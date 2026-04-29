"""
model_comparison.py
===================
Compares all biometric models on:
  - Total parameters
  - Trainable parameters
  - GFLOPs for a single forward pass (batch=1, input 112×112)

Models: CompNet, PPNet, CCNet, CO3Net, SF2Net,
        PalmBridge (CompNet backbone), ConvNeXt, DINOv2,
        GIFT (GIFTBackbone — inference model only, FSM inactive),
        TSCAN (PalmNet / FeatureEncoder — inference model only),
        PDFG (MultiDatasetExtractors — inference model only, N=2 heads),
        ArcFace iResNet100 (loaded from ONNX via onnx2torch, 112×112 RGB),
        MagFace iResNet100 (loaded from .pth checkpoint,    112×112 RGB)

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

# ── Checkpoint paths ──────────────────────────────────────────
ARCFACE_ONNX_PATH  = "/home/pai-ng/Jamal/NIPS2026/face_models/checkpoints/r100_glint360k.onnx"
MAGFACE_CKPT_PATH  = "/home/pai-ng/Jamal/NIPS2026/face_models/checkpoints/magface_iresnet100.pth"


# ══════════════════════════════════════════════════════════════
#  SHARED COMPONENTS
# ══════════════════════════════════════════════════════════════

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
#  PPNET
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
        return self.embed(f)


# ══════════════════════════════════════════════════════════════
#  CCNET / CO3NET
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
    cb1 = CompetitiveBlock(1, 9,  35, 0.8, 1.00)
    cb2 = CompetitiveBlock(1, 36, 17, 0.8, 0.50)
    cb3 = CompetitiveBlock(1, 9,   7, 0.8, 0.25)
    with torch.no_grad():
        f = torch.cat([cb1(input_tensor), cb2(input_tensor), cb3(input_tensor)], dim=1)
    return f.shape[1]


class CCNet(nn.Module):
    def __init__(self, num_classes=190, sample_input=None):
        super().__init__()
        self.cb1 = CompetitiveBlock(1, 9, 35, 0.8, 1.00)
        self.cb2 = CompetitiveBlock(1, 36, 17, 0.8, 0.50)
        self.cb3 = CompetitiveBlock(1, 9,  7, 0.8, 0.25)
        if sample_input is None:
            sample_input = torch.zeros(1, 1, 112, 112)
        fc_in = _infer_competitive_fc_in(sample_input)
        self.fc  = nn.Linear(fc_in, 2048)
        self.bn  = nn.BatchNorm1d(2048)

    def forward(self, x):
        f = torch.cat([self.cb1(x), self.cb2(x), self.cb3(x)], dim=1)
        return F.normalize(self.bn(self.fc(f)), p=2, dim=1)


class CO3Net(nn.Module):
    def __init__(self, num_classes=190, sample_input=None):
        super().__init__()
        self.cb1 = CompetitiveBlock(1, 9, 35, 0.8, 1.00)
        self.cb2 = CompetitiveBlock(1, 36, 17, 0.8, 0.50)
        self.cb3 = CompetitiveBlock(1, 9,  7, 0.8, 0.25)
        if sample_input is None:
            sample_input = torch.zeros(1, 1, 112, 112)
        fc_in = _infer_competitive_fc_in(sample_input)
        self.fc  = nn.Linear(fc_in, 2048)
        self.bn  = nn.BatchNorm1d(2048)

    def forward(self, x):
        f = torch.cat([self.cb1(x), self.cb2(x), self.cb3(x)], dim=1)
        return F.normalize(self.bn(self.fc(f)), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  SF2NET
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
#  PALMBRIDGE
# ══════════════════════════════════════════════════════════════

class PalmBridgeModel(nn.Module):
    def __init__(self, feature_dim=512, num_pb_vectors=512):
        super().__init__()
        self.backbone = CompNetBackbone(feature_dim)
        self.P = nn.Parameter(
            F.normalize(torch.randn(num_pb_vectors, feature_dim), p=2, dim=1))
        self.W_ORI = 0.7; self.W_MAP = 0.3

    def forward(self, x):
        z = self.backbone(x)
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
#  PDFG
# ══════════════════════════════════════════════════════════════

class _PDFGSharedLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16,  3, stride=4, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=1)
        self.conv2 = nn.Conv2d(16, 32,  5, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d(2, stride=1)
        self.conv3 = nn.Conv2d(32, 64,  3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=1)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))
        x = self.pool3(self.act(self.conv4(x)))
        return x


class PDFGModel(nn.Module):
    def __init__(self, n_heads=2, feature_dim=128, sample_input=None):
        super().__init__()
        self.shared = _PDFGSharedLayers()
        if sample_input is None:
            sample_input = torch.zeros(1, 3, 112, 112)
        with torch.no_grad():
            flat_dim = self.shared(sample_input).view(1, -1).shape[1]
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flat_dim, 1024), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1024, 512),      nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, feature_dim),
            ) for _ in range(n_heads)
        ])

    def forward(self, x):
        shared = self.shared(x).view(x.size(0), -1)
        per_head = torch.stack(
            [F.normalize(h(shared), p=2, dim=1) for h in self.heads], dim=0)
        return F.normalize(per_head.mean(dim=0), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  TSCAN
# ══════════════════════════════════════════════════════════════

class TSCANModel(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        children = list(backbone.children())
        self.frozen_layers = nn.Sequential(*children[:7])
        self.layer4        = children[7]
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
#  GIFT
# ══════════════════════════════════════════════════════════════

class _FeatureStylizationModule(nn.Module):
    def __init__(self, gamma=0.2):
        super().__init__()
        self.gamma  = gamma
        self.active = False

    def forward(self, f):
        return f, f


class GIFTModel(nn.Module):
    def __init__(self, emb_dim=128, gamma=0.2):
        super().__init__()
        resnet = tv_models.resnet18(weights=None)
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
        x = self.avgpool(x).flatten(1)
        return F.normalize(self.fc(x), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  iResNet100  (shared architecture for ArcFace and MagFace)
# ══════════════════════════════════════════════════════════════

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride,
                     padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2   = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3   = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample; self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x);    out = self.conv1(out)
        out = self.bn2(out);  out = self.prelu(out)
        out = self.conv2(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    def __init__(self, block, layers, num_features=512):
        super().__init__()
        self.inplanes = 64
        self.conv1  = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu  = nn.PReLU(64)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2    = nn.BatchNorm2d(512, eps=1e-05)
        self.fc     = nn.Linear(512 * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes, eps=1e-05))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.bn2(x)
        x = F.adaptive_avg_pool2d(x, (7, 7))   # handles any input spatial size
        x = x.flatten(1)
        x = self.fc(x);  x = self.features(x)
        return x


def iresnet100(num_features=512):
    return IResNet(IBasicBlock, [3, 13, 30, 3], num_features=num_features)


# ══════════════════════════════════════════════════════════════
#  ARCFACE iResNet100  (loaded from ONNX via onnx2torch)
#  Input: 3×112×112 RGB, mean=0.5, std=0.5
#  Freeze: first 75% of parameter tensors
# ══════════════════════════════════════════════════════════════

class ArcFaceModel(nn.Module):
    """
    ArcFace iResNet100 from InsightFace ONNX (R100, Glint360K).
    Loaded via onnx2torch — no custom architecture mapping needed.
    Freeze ratio: first 75% of parameter tensors frozen.
    """
    def __init__(self, onnx_path, freeze_ratio=0.75):
        super().__init__()
        import onnx
        from onnx2torch import convert
        self.net = convert(onnx.load(onnx_path))

        all_params = list(self.net.parameters())
        n_freeze   = int(len(all_params) * freeze_ratio)
        for i, p in enumerate(all_params):
            p.requires_grad = (i >= n_freeze)

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return F.normalize(out, p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  MAGFACE iResNet100  (loaded from .pth checkpoint)
#  Input: 3×112×112 RGB, mean=0.5, std=0.5
#  Freeze: first 75% of parameter tensors
# ══════════════════════════════════════════════════════════════

class MagFaceModel(nn.Module):
    """
    MagFace iResNet100 from official MagFace checkpoint (MS1MV2).
    Checkpoint key format: "features.module.*" → stripped to match IResNet.
    Freeze ratio: first 75% of parameter tensors frozen.
    Input resized to 112×112 before network (matches InsightFace convention
    and makes FLOPs comparable with ArcFace).
    """
    def __init__(self, ckpt_path, freeze_ratio=0.75):
        super().__init__()
        self.net = iresnet100()

        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("features.module.", ""): v
                 for k, v in state.items()
                 if k.startswith("features.module.")}
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if missing:    print(f"  [MagFace] Missing keys    : {len(missing)}")
        if unexpected: print(f"  [MagFace] Unexpected keys : {len(unexpected)}")

        all_params = list(self.net.parameters())
        n_freeze   = int(len(all_params) * freeze_ratio)
        for i, p in enumerate(all_params):
            p.requires_grad = (i >= n_freeze)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  ANALYSIS
# ══════════════════════════════════════════════════════════════

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, input_tensor):
    try:
        model.eval()
        flops = FlopCountAnalysis(model, input_tensor)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9
    except Exception as e:
        print(f"    [FLOP count failed: {e}]")
        return float("nan")


def fmt_params(n):
    if n >= 1e6: return f"{n/1e6:.2f} M"
    if n >= 1e3: return f"{n/1e3:.1f} K"
    return str(n)


def run():
    x1   = torch.zeros(1, 1, 112, 112)   # grayscale 112×112
    x3   = torch.zeros(1, 3, 112, 112)   # RGB 112×112
    x3s  = torch.zeros(1, 3, 112, 112)   # RGB 112×112

    MODELS = [
        ("CompNet",             lambda: CompNet(),                     x1),
        ("PPNet",               lambda: PPNet(),                       x1),
        ("CCNet",               lambda: CCNet(sample_input=x1),        x1),
        ("CO3Net",              lambda: CO3Net(sample_input=x1),       x1),
        ("SF2Net",              lambda: SF2Net(),                      x1),
        ("PalmBridge",          lambda: PalmBridgeModel(),             x1),
        ("ConvNeXtV2-Tiny",     lambda: ConvNeXtModel(),              x3),
        ("DINOv2 ViT-S/14",    lambda: DINOv2Model(),                x3),
        ("GIFT (ResNet-18)",    lambda: GIFTModel(),                  x3s),
        ("TSCAN (PalmNet)",     lambda: TSCANModel(),                  x3s),
        ("PDFG",                lambda: PDFGModel(sample_input=x3s),  x3s),
        ("ArcFace iResNet100",  lambda: ArcFaceModel(ARCFACE_ONNX_PATH), x3),
        ("MagFace iResNet100",  lambda: MagFaceModel(MAGFACE_CKPT_PATH), x3),
    ]

    W_NAME  = 22
    W_TOTAL = 16
    W_TRAIN = 16
    W_FLOPS = 12

    header = (f"{'Model':<{W_NAME}}"
              f"{'Total Params':>{W_TOTAL}}"
              f"{'Trainable':>{W_TRAIN}}"
              f"{'GFLOPs':>{W_FLOPS}}")
    sep = "─" * len(header)

    print("\n" + "=" * len(header))
    print("Model Comparison — Parameters & FLOPs (input 112×112)")
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
            results.append({"name": name, "total": total,
                            "trainable": trainable, "gflops": gflops})
            gflops_str = f"{gflops:.3f}" if not math.isnan(gflops) else "N/A"
            print(f"    {name:<{W_NAME}}"
                  f"{fmt_params(total):>{W_TOTAL}}"
                  f"{fmt_params(trainable):>{W_TRAIN}}"
                  f"{gflops_str:>{W_FLOPS}}")
        except Exception as e:
            print(f"  ERROR — {name}: {e}")

    print(sep)

    out_path = "model_comparison.txt"
    with open(out_path, "w") as f:
        f.write("Model Comparison — Parameters & FLOPs (input 112×112 for all models)\n")
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
        f.write("  - Grayscale models (CompNet/PPNet/CCNet/CO3Net/SF2Net/PalmBridge): input 1×112×112\n")
        f.write("  - RGB models (ConvNeXt/DINOv2/GIFT/TSCAN/PDFG/ArcFace/MagFace): input 3×112×112\n")
        f.write("  - All models evaluated at 112×112 input for fair comparison\n")
        f.write("  - ArcFace loaded from ONNX via onnx2torch (R100, Glint360K)\n")
        f.write("  - MagFace loaded from .pth checkpoint (R100, MS1MV2, epoch 25)\n")
        f.write("  - Both face models: first 75% of parameter tensors frozen\n")
        f.write("  - GIFT FSMs inactive at inference → zero extra FLOPs\n")
        f.write("  - TSCAN: FeatureEncoder only; discriminator & AdaFace excluded\n")
        f.write("  - PDFG: MultiDatasetExtractors (N=2 heads); ArcFace loss excluded\n")
        f.write("  - PalmBridge includes codebook P in R^{512x512}\n")
    print(f"\nTable saved to: {out_path}")


if __name__ == "__main__":
    run()
