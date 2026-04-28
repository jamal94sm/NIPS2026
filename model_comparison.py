"""
model_comparison.py
===================
Compares all biometric models on:
  - Total parameters
  - Trainable parameters
  - GFLOPs for a single forward pass (batch=1, input 224×224)

Models: CompNet, PPNet, CCNet, CO3Net, SF2Net,
        PalmBridge (CompNet backbone), ConvNeXt, DINOv2,
        GIFT (GIFTBackbone — inference model only, FSM inactive),
        TSCAN (PalmNet / FeatureEncoder — inference model only)

Run on the server:
    python model_comparison.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models as tv_models
from fvcore.nn import FlopCountAnalysis

DEVICE = torch.device("cpu")   # FLOPs measured on CPU


# ══════════════════════════════════════════════════════════════
#  SHARED COMPONENTS
# ══════════════════════════════════════════════════════════════

# ── CompNet / PalmBridge Gabor backbone ───────────────────────

class LearnableGaborLayer(nn.Module):
    def __init__(self, num_filters=32, kernel_size=15):
        super().__init__()
        n = num_filters; ks = kernel_size
        self.theta = nn.Parameter(torch.linspace(0.0, math.pi, n + 1)[:-1])
        self.sigma = nn.Parameter(torch.full((n,), 3.0))
        self.lambd = nn.Parameter(torch.full((n,), 6.0))
        self.psi   = nn.Parameter(torch.zeros(n))
        self.gamma = nn.Parameter(torch.full((n,), 0.5))
        half = ks // 2
        ys = torch.arange(-half, half + 1, dtype=torch.float32)
        xs = torch.arange(-half, half + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("xx", xx)
        self.register_buffer("yy", yy)
        self.ks = ks

    def _build_filters(self):
        theta = self.theta
        sigma = self.sigma.abs().clamp(min=0.5)
        lambd = self.lambd.abs().clamp(min=1.0)
        psi   = self.psi
        gamma = self.gamma.abs().clamp(min=0.1)
        xx = self.xx.unsqueeze(0); yy = self.yy.unsqueeze(0)
        cos_t = torch.cos(theta).view(-1, 1, 1)
        sin_t = torch.sin(theta).view(-1, 1, 1)
        sigma = sigma.view(-1, 1, 1); lambd = lambd.view(-1, 1, 1)
        psi   = psi.view(-1, 1, 1);   gamma = gamma.view(-1, 1, 1)
        x_rot =  xx * cos_t + yy * sin_t
        y_rot = -xx * sin_t + yy * cos_t
        envelope = torch.exp(-(x_rot**2 + gamma**2 * y_rot**2) / (2.0 * sigma**2))
        kernel   = envelope * torch.cos(2.0 * math.pi * x_rot / lambd + psi)
        kernel   = kernel - kernel.mean(dim=(1, 2), keepdim=True)
        return kernel.unsqueeze(1).contiguous()

    def forward(self, x):
        return F.conv2d(x, self._build_filters(), padding=self.ks // 2)


class CompetitivePool(nn.Module):
    def forward(self, x):
        out, _ = x.abs().max(dim=1, keepdim=True)
        return out


class CompNetBackbone(nn.Module):
    """Shared backbone for CompNet and PalmBridge."""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.gabor   = LearnableGaborLayer()
        self.compete = CompetitivePool()
        self.gbn     = nn.BatchNorm2d(1)

        def _block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin,  cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.block1 = _block(1,   32);  self.pool1 = nn.MaxPool2d(2)
        self.block2 = _block(32,  64);  self.pool2 = nn.MaxPool2d(2)
        self.block3 = _block(64,  128); self.pool3 = nn.MaxPool2d(2)
        self.block4 = _block(128, 256); self.gap   = nn.AdaptiveAvgPool2d((4, 4))
        self.embed  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, x):
        g = F.relu(self.gabor(x))
        g = self.gbn(self.compete(g))
        f = self.pool1(self.block1(g))
        f = self.pool2(self.block2(f))
        f = self.pool3(self.block3(f))
        f = self.gap(self.block4(f))
        return F.normalize(self.embed(f), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  COMPNET
# ══════════════════════════════════════════════════════════════

class CompNet(nn.Module):
    def __init__(self, num_classes=190, feature_dim=512):
        super().__init__()
        self.backbone = CompNetBackbone(feature_dim)

    def forward(self, x):
        return self.backbone(x)


# ══════════════════════════════════════════════════════════════
#  PPNET  (CE + Contrastive, L2 distance matching)
# ══════════════════════════════════════════════════════════════

class PPNet(nn.Module):
    def __init__(self, num_classes=190, feature_dim=512):
        super().__init__()
        self.gabor   = LearnableGaborLayer()
        self.compete = CompetitivePool()
        self.gbn     = nn.BatchNorm2d(1)

        def _block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin,  cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.block1 = _block(1,   32);  self.pool1 = nn.MaxPool2d(2)
        self.block2 = _block(32,  64);  self.pool2 = nn.MaxPool2d(2)
        self.block3 = _block(64,  128); self.pool3 = nn.MaxPool2d(2)
        self.block4 = _block(128, 256); self.gap   = nn.AdaptiveAvgPool2d((4, 4))
        self.embed  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
        )
        self.l2_reg = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        g = F.relu(self.gabor(x))
        g = self.gbn(self.compete(g))
        f = self.pool1(self.block1(g))
        f = self.pool2(self.block2(f))
        f = self.pool3(self.block3(f))
        f = self.gap(self.block4(f))
        z = self.embed(f)
        return z   # L2 distance used at eval


# ══════════════════════════════════════════════════════════════
#  CCNET / CO3NET  (Gabor Competitive Block)
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, ksize, stride=1, padding=0, init_ratio=1.):
        super().__init__()
        r = init_ratio
        self.ch_in = ch_in; self.ch_out = ch_out
        self.ksize  = ksize; self.stride = stride; self.padding = padding
        self.gamma  = nn.Parameter(torch.FloatTensor([2.0]))
        self.sigma  = nn.Parameter(torch.FloatTensor([9.2 * r]))
        self.theta  = nn.Parameter(
            torch.arange(ch_out).float() * math.pi / ch_out, requires_grad=False)
        self.f      = nn.Parameter(torch.FloatTensor([0.057 / r]))
        self.psi    = nn.Parameter(torch.FloatTensor([0.0]), requires_grad=False)

    def _build_bank(self):
        xm  = self.ksize // 2
        rng = torch.arange(-xm, xm + 1).float()
        y   = rng.view(1, -1).repeat(self.ch_out, self.ch_in, self.ksize, 1)
        x   = rng.view(-1, 1).repeat(self.ch_out, self.ch_in, 1, self.ksize)
        x   = x.to(self.sigma.device); y = y.to(self.sigma.device)
        th  = self.theta.view(-1, 1, 1, 1)
        xt  =  x * torch.cos(th) + y * torch.sin(th)
        yt  = -x * torch.sin(th) + y * torch.cos(th)
        gb  = -torch.exp(
            -0.5 * ((self.gamma * xt) ** 2 + yt ** 2)
            / (8 * self.sigma.view(-1, 1, 1, 1) ** 2)
        ) * torch.cos(2 * math.pi * self.f.view(-1, 1, 1, 1) * xt
                      + self.psi.view(-1, 1, 1, 1))
        return gb - gb.mean(dim=[2, 3], keepdim=True)

    def forward(self, x):
        return F.conv2d(x, self._build_bank(), stride=self.stride, padding=self.padding)


class SELayer(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(ch, ch, bias=False), nn.ReLU(inplace=True),
            nn.Linear(ch, ch, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)


class CompetitiveBlock(nn.Module):
    def __init__(self, ch_in, n_comp, ksize, weight=0.8, init_ratio=1., o1=32):
        super().__init__()
        nc2 = n_comp * 2; nc4 = n_comp * 4
        self.g1 = GaborConv2d(ch_in, n_comp, ksize, 2, ksize // 2, init_ratio)
        self.g2 = GaborConv2d(nc2, nc2, ksize, 2, ksize // 2, init_ratio)
        if ksize == 35:
            self.c1a = nn.Conv2d(ch_in,  n_comp, 7, 1, 0)
            self.c1b = nn.Conv2d(n_comp, n_comp, 5, 2, 5)
            self.c2a = nn.Conv2d(nc2,    nc2,    7, 1, 0)
            self.c2b = nn.Conv2d(nc2,    nc2,    5, 2, 5)
        elif ksize == 17:
            self.c1a = nn.Conv2d(ch_in,  n_comp, 5, 1, 0)
            self.c1b = nn.Conv2d(n_comp, n_comp, 3, 2, 3)
            self.c2a = nn.Conv2d(nc2,    nc2,    5, 1, 0)
            self.c2b = nn.Conv2d(nc2,    nc2,    3, 2, 3)
        else:
            self.c1a = nn.Conv2d(ch_in,  n_comp, 3, 1, 0)
            self.c1b = nn.Conv2d(n_comp, n_comp, 1, 2, 1)
            self.c2a = nn.Conv2d(nc2,    nc2,    3, 1, 0)
            self.c2b = nn.Conv2d(nc2,    nc2,    1, 2, 1)
        self.sm_c = nn.Softmax(dim=1)
        self.sm_h = nn.Softmax(dim=2)
        self.sm_w = nn.Softmax(dim=3)
        self.se1  = SELayer(nc2); self.se2 = SELayer(nc4)
        self.ppu1 = nn.Conv2d(nc2, o1 // 2, 5, 2, 0)
        self.ppu2 = nn.Conv2d(nc4, o1 // 2, 5, 2, 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.wc   = weight; self.ws = (1. - weight) / 2.

    def _compete(self, x):
        return self.wc * self.sm_c(x) + self.ws * (self.sm_h(x) + self.sm_w(x))

    def forward(self, x):
        f  = torch.cat([self.g1(x), self.c1b(self.c1a(x))], dim=1)
        x1 = self.pool(self.ppu1(self.se1(self._compete(f))))
        f  = torch.cat([self.g2(f), self.c2b(self.c2a(f))], dim=1)
        x2 = self.pool(self.ppu2(self.se2(self._compete(f))))
        return torch.cat([x1.flatten(1), x2.flatten(1)], dim=1)


def _infer_competitive_fc_in(input_tensor):
    """
    Dynamically compute the flattened feature size produced by the three
    CompetitiveBlocks (cb1/cb2/cb3) for an arbitrary spatial input size,
    avoiding the hardcoded 13152 that was calibrated for 112×112 only.
    """
    cb1 = CompetitiveBlock(1, 9,  35, 0.8, 1.00)
    cb2 = CompetitiveBlock(1, 36, 17, 0.8, 0.50)
    cb3 = CompetitiveBlock(1, 9,   7, 0.8, 0.25)
    with torch.no_grad():
        f = torch.cat([cb1(input_tensor), cb2(input_tensor), cb3(input_tensor)], dim=1)
    return f.shape[1]


class CCNet(nn.Module):
    """1-channel input, feature_dim=2048. fc size inferred from input shape."""
    def __init__(self, num_classes=190, sample_input=None):
        super().__init__()
        self.cb1 = CompetitiveBlock(1, 9, 35, 0.8, 1.00)
        self.cb2 = CompetitiveBlock(1, 36, 17, 0.8, 0.50)
        self.cb3 = CompetitiveBlock(1, 9,  7, 0.8, 0.25)

        # Infer fc input dimension from a sample forward pass
        if sample_input is None:
            sample_input = torch.zeros(1, 1, 224, 224)
        fc_in = _infer_competitive_fc_in(sample_input)

        self.fc  = nn.Linear(fc_in, 2048)
        self.bn  = nn.BatchNorm1d(2048)

    def forward(self, x):
        f = torch.cat([self.cb1(x), self.cb2(x), self.cb3(x)], dim=1)
        return F.normalize(self.bn(self.fc(f)), p=2, dim=1)


class CO3Net(nn.Module):
    """Same architecture as CCNet. fc size inferred from input shape."""
    def __init__(self, num_classes=190, sample_input=None):
        super().__init__()
        self.cb1 = CompetitiveBlock(1, 9, 35, 0.8, 1.00)
        self.cb2 = CompetitiveBlock(1, 36, 17, 0.8, 0.50)
        self.cb3 = CompetitiveBlock(1, 9,  7, 0.8, 0.25)

        if sample_input is None:
            sample_input = torch.zeros(1, 1, 224, 224)
        fc_in = _infer_competitive_fc_in(sample_input)

        self.fc  = nn.Linear(fc_in, 2048)
        self.bn  = nn.BatchNorm1d(2048)

    def forward(self, x):
        f = torch.cat([self.cb1(x), self.cb2(x), self.cb3(x)], dim=1)
        return F.normalize(self.bn(self.fc(f)), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  SF2NET  (Gabor + Triplet, 1024-d)
# ══════════════════════════════════════════════════════════════

class SF2Net(nn.Module):
    def __init__(self, num_classes=190, feature_dim=1024):
        super().__init__()
        self.gabor   = LearnableGaborLayer()
        self.compete = CompetitivePool()
        self.gbn     = nn.BatchNorm2d(1)

        def _block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin,  cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.block1 = _block(1,   64);  self.pool1 = nn.MaxPool2d(2)
        self.block2 = _block(64,  128); self.pool2 = nn.MaxPool2d(2)
        self.block3 = _block(128, 256); self.pool3 = nn.MaxPool2d(2)
        self.block4 = _block(256, 512); self.gap   = nn.AdaptiveAvgPool2d((4, 4))
        self.embed  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, x):
        g = F.relu(self.gabor(x))
        g = self.gbn(self.compete(g))
        f = self.pool1(self.block1(g))
        f = self.pool2(self.block2(f))
        f = self.pool3(self.block3(f))
        f = self.gap(self.block4(f))
        return F.normalize(self.embed(f), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  PALMBRIDGE  (CompNet backbone + codebook)
# ══════════════════════════════════════════════════════════════

class PalmBridgeModel(nn.Module):
    """CompNet backbone only — codebook is separate (not counted in inference)."""
    def __init__(self, feature_dim=512, num_pb_vectors=512):
        super().__init__()
        self.backbone = CompNetBackbone(feature_dim)
        # PalmBridge codebook P ∈ R^{K×D}
        self.P = nn.Parameter(
            F.normalize(torch.randn(num_pb_vectors, feature_dim), p=2, dim=1))
        self.W_ORI = 0.7; self.W_MAP = 0.3

    def forward(self, x):
        z = self.backbone(x)
        # Nearest-vector lookup (argmin L2)
        dists   = (z.pow(2).sum(1, keepdim=True)
                   + self.P.pow(2).sum(1).unsqueeze(0)
                   - 2.0 * (z @ self.P.t()))
        z_tilde = self.P[dists.argmin(dim=1)]
        return F.normalize(self.W_ORI * z + self.W_MAP * z_tilde, p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  CONVNEXT
# ══════════════════════════════════════════════════════════════

class ConvNeXtModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone  = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=0)
        self.embed_dim = self.backbone.num_features

    def forward(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  DINOV2
# ══════════════════════════════════════════════════════════════

class DINOv2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone  = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
        self.embed_dim = 384

    def forward(self, x):
        out = self.backbone.forward_features(x)
        return F.normalize(out["x_norm_clstoken"], p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  TSCAN  (inference model only — PalmNet / FeatureEncoder)
#  Source: "Teacher-Student Co-learning with Adversarial
#           Normalization" adapted for Palm-Auth
#
#  Inference path: PalmNet.get_features(x)
#    → FeatureEncoder: frozen ResNet-18 stem+layer1-3
#                    + trainable layer4
#                    + Linear(512→feat_dim) + Tanh
#  Excluded (training-only): DomainDiscriminator, AdaFaceLoss
#  Input: 3×224×224
# ══════════════════════════════════════════════════════════════

class TSCANModel(nn.Module):
    """
    Inference-only wrapper for TSCAN's PalmNet.
    Mirrors FeatureEncoder exactly — frozen_layers are included
    because they run during inference (no_grad scope is a training
    optimisation, not an architectural exclusion).
    """
    def __init__(self, feat_dim=256):
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        children = list(backbone.children())
        # stem + layer1 + layer2 + layer3  (frozen during training, but
        # still execute at inference — must be included in the model)
        self.frozen_layers = nn.Sequential(*children[:7])
        self.layer4        = children[7]          # trainable
        self.avgpool       = children[8]
        self.flatten       = nn.Flatten()
        self.linear        = nn.Linear(512, feat_dim, bias=True)
        self.hash          = nn.Tanh()

    def forward(self, x):
        x    = self.frozen_layers(x)
        x    = self.layer4(x)
        x    = self.avgpool(x)
        bb   = self.flatten(x)
        feat = self.hash(self.linear(bb))
        return F.normalize(feat, p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  GIFT  (GIFTBackbone — inference model only)
#  Source: "Generating Stylized Features for Single-Source
#           Cross-Dataset Palmprint Recognition", TIP 2024
#
#  At inference the FSMs are inactive (pass-through), so they
#  contribute parameters but add zero FLOPs.
#  Input: 3×112×112 RGB  (eval_transform in the original code)
# ══════════════════════════════════════════════════════════════

class _FeatureStylizationModule(nn.Module):
    """FSM — inactive at eval, so zero extra FLOPs during inference."""
    def __init__(self, gamma=0.2):
        super().__init__()
        self.gamma  = gamma
        self.active = False   # always False at eval

    def forward(self, f):
        # During eval (active=False) this is a pure pass-through.
        return f, f


class GIFTModel(nn.Module):
    """
    Inference-only wrapper around GIFTBackbone.
    Matches the original architecture exactly:
      ResNet-18 backbone + 5 FSM modules + linear projection to emb_dim.
    FSMs are never activated here, so they cost zero FLOPs.
    """
    def __init__(self, emb_dim=128, gamma=0.2):
        super().__init__()
        resnet = tv_models.resnet18(weights=None)   # no pretrained weights needed for param/FLOP count

        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.fsm0    = _FeatureStylizationModule(gamma)

        self.layer1  = resnet.layer1; self.fsm1 = _FeatureStylizationModule(gamma)
        self.layer2  = resnet.layer2; self.fsm2 = _FeatureStylizationModule(gamma)
        self.layer3  = resnet.layer3; self.fsm3 = _FeatureStylizationModule(gamma)
        self.layer4  = resnet.layer4; self.fsm4 = _FeatureStylizationModule(gamma)

        self.avgpool = resnet.avgpool
        self.fc      = nn.Linear(resnet.fc.in_features, emb_dim)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x, _ = self.fsm0(x)
        x = self.layer1(x); x, _ = self.fsm1(x)
        x = self.layer2(x); x, _ = self.fsm2(x)
        x = self.layer3(x); x, _ = self.fsm3(x)
        x = self.layer4(x); x, _ = self.fsm4(x)
        x   = self.avgpool(x).flatten(1)
        return F.normalize(self.fc(x), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  ANALYSIS
# ══════════════════════════════════════════════════════════════

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, input_tensor):
    """Return GFLOPs for a single forward pass."""
    try:
        model.eval()
        flops = FlopCountAnalysis(model, input_tensor)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9   # GFLOPs
    except Exception as e:
        print(f"    [FLOP count failed: {e}]")
        return float("nan")


def fmt_params(n):
    if n >= 1e6: return f"{n/1e6:.2f} M"
    if n >= 1e3: return f"{n/1e3:.1f} K"
    return str(n)


def run():
    x1  = torch.zeros(1, 1, 224, 224)   # grayscale 224×224
    x3  = torch.zeros(1, 3, 224, 224)   # RGB 224×224
    x3s = torch.zeros(1, 3, 224, 224)   # RGB 224×224  (standardised input)

    MODELS = [
        ("CompNet",          lambda: CompNet(),                    x1),
        ("PPNet",            lambda: PPNet(),                      x1),
        ("CCNet",            lambda: CCNet(sample_input=x1),      x1),
        ("CO3Net",           lambda: CO3Net(sample_input=x1),     x1),
        ("SF2Net",           lambda: SF2Net(),                     x1),
        ("PalmBridge",       lambda: PalmBridgeModel(),            x1),
        ("ConvNeXtV2-Tiny",  lambda: ConvNeXtModel(),             x3),
        ("DINOv2 ViT-S/14", lambda: DINOv2Model(),               x3),
        ("GIFT (ResNet-18)", lambda: GIFTModel(),                 x3s),
        ("TSCAN (PalmNet)",  lambda: TSCANModel(),                x3s),
    ]

    # Column widths
    W_NAME  = 20
    W_TOTAL = 16
    W_TRAIN = 16
    W_FLOPS = 12

    header = (f"{'Model':<{W_NAME}}"
              f"{'Total Params':>{W_TOTAL}}"
              f"{'Trainable':>{W_TRAIN}}"
              f"{'GFLOPs':>{W_FLOPS}}")
    sep = "─" * len(header)

    print("\n" + "=" * len(header))
    print("Model Comparison — Parameters & FLOPs (single sample, input 224×224)")
    print("=" * len(header))
    print(header)
    print(sep)

    results = []
    for name, build_fn, x in MODELS:
        print(f"  Analysing {name} ...", flush=True)
        try:
            model = build_fn().to(DEVICE).eval()
            total, trainable = count_params(model)
            gflops           = count_flops(model, x)

            results.append({
                "name": name, "total": total,
                "trainable": trainable, "gflops": gflops,
            })

            gflops_str = f"{gflops:.3f}" if not math.isnan(gflops) else "N/A"
            print(f"    {name:<{W_NAME}}"
                  f"{fmt_params(total):>{W_TOTAL}}"
                  f"{fmt_params(trainable):>{W_TRAIN}}"
                  f"{gflops_str:>{W_FLOPS}}")

        except Exception as e:
            print(f"  ERROR — {name}: {e}")

    print(sep)

    # ── Save to txt ────────────────────────────────────────────
    out_path = "model_comparison.txt"
    with open(out_path, "w") as f:
        f.write("Model Comparison — Parameters & FLOPs (input 224×224)\n")
        f.write(sep + "\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for r in results:
            gflops_str = f"{r['gflops']:.3f}" if not math.isnan(r['gflops']) else "N/A"
            f.write(f"{r['name']:<{W_NAME}}"
                    f"{fmt_params(r['total']):>{W_TOTAL}}"
                    f"{fmt_params(r['trainable']):>{W_TRAIN}}"
                    f"{gflops_str:>{W_FLOPS}}\n")
        f.write(sep + "\n")
        f.write("\nNotes:\n")
        f.write("  - GFLOPs measured for a single sample (batch=1)\n")
        f.write("  - Grayscale models (CompNet/PPNet/CCNet/CO3Net/SF2Net/PalmBridge): input 1×224×224\n")
        f.write("  - RGB models (ConvNeXt/DINOv2): input 3×224×224\n")
        f.write("  - GIFT: input 3×224×224 (standardised to match other models)\n")
        f.write("  - GIFT FSMs are inactive at inference → zero extra FLOPs, but counted in params\n")
        f.write("  - TSCAN: inference model is PalmNet (FeatureEncoder); discriminator & AdaFace excluded\n")
        f.write("  - DINOv2 requires input size multiple of patch_size=14 → 224×224\n")
        f.write("  - PalmBridge includes codebook P ∈ R^{512×512} in parameter count\n")
        f.write("  - CCNet/CO3Net fc input size inferred dynamically (not hardcoded)\n")
    print(f"\nTable saved to: {out_path}")


if __name__ == "__main__":
    run()
