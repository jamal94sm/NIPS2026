"""
MSPHNet on CASIA-MS Dataset  —  FIXED VERSION
==================================================
Changes vs original:
  FIX 1  (critical) TransformerBranch: pos_embedding pre-allocated in __init__,
          not re-created inside forward(). The original created it dynamically,
          so it was never in the optimizer's parameter groups and never trained.
  FIX 2  (critical) PHEB: added pre_proj Conv2d(in_channels→1) before the Gabor
          filter so the Gabor always operates on a single-channel map instead of
          a 64/128/256-channel one. The original caused a massive bottleneck
          (e.g. 256→36 channels = 86% information lost at the deepest layers).
          Also removed the dead `is_first` branch (both cases were identical).
          Added a residual 1×1 shortcut around the whole PHEB so gradients can
          flow back to the encoder without going through the Gabor path.
  FIX 3  (moderate) MSPHNet: passes spatial_size to each PHEB/TransformerBranch
          so the pre-allocated positional embedding has the right shape per level.
  FIX 4  (config)   lr reset to 0.0001 (paper default), batch_size reset to 32,
          StepLR replaced with CosineAnnealingLR, 10-epoch linear warmup added.
  FIX 5  (stability) DataLoader: persistent_workers=True, error-handling wrapper
          in __getitem__ so a single corrupt image doesn't crash the whole run.
  FIX 6  (training)  Validation pass now uses ground-truth labels so ArcFace
          loss is computed the same way as during training (with margin).
"""

CONFIG = {
    "protocol"         : "open-set",   # "closed-set" | "open-set"
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "results_dir"      : "./rst_msphnet_casia_ms",
    "img_side"         : 128,
    "batch_size"       : 32,           # FIX 4: paper default (was 64)
    "num_epochs"       : 1000,
    "lr"               : 0.0001,       # FIX 4: paper default (was 0.001)
    "warmup_epochs"    : 10,           # FIX 4: new — linear LR warmup
    "lr_step"          : 300,
    "lr_gamma"         : 0.5,
    "dropout"          : 0.5,
    "arcface_s"        : 30.0,
    "arcface_m"        : 0.50,
    "embedding_dim"    : 1024,
    "gabor_filters"    : 36,
    "gabor_kernel"     : 17,
    "transformer_depth": 2,
    "transformer_dim"  : 128,
    "transformer_heads": 8,
    "patch_size"       : 8,
    "ca_reduction"     : 16,
    "train_ratio"      : 0.80,
    "gallery_ratio"    : 0.50,
    "val_ratio"        : 0.10,
    "random_seed"      : 42,
    "save_every"       : 10,
    "eval_every"       : 50,
    "num_workers"      : 4,
}

import os, sys, math, time, random, pickle, warnings
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange
except ImportError:
    os.system("pip install einops")
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange

warnings.filterwarnings("ignore")

SEED = CONFIG["random_seed"]
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════
#  ARCFACE
# ══════════════════════════════════════════════════════════════

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.s, self.m    = s, m
        self.weight       = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin  = easy_margin
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is not None:
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
            phi  = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = one_hot * phi + (1.0 - one_hot) * cosine
            return output * self.s
        return cosine * self.s


# ══════════════════════════════════════════════════════════════
#  LEARNABLE GABOR FILTER
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=0.5):
        super().__init__()
        self.channel_in  = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding

        SIGMA = 9.2 * init_ratio
        self.sigma     = nn.Parameter(torch.FloatTensor([SIGMA]))
        self.gamma     = nn.Parameter(torch.FloatTensor([2.0]))
        self.theta     = nn.Parameter(
            torch.arange(0, channel_out).float() * math.pi / channel_out,
            requires_grad=False)
        self.frequency = nn.Parameter(torch.FloatTensor([0.057 / init_ratio]))
        self.psi       = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def _get_gabor_kernel(self):
        xm = self.kernel_size // 2
        x0 = torch.arange(-xm, xm + 1).float()
        y0 = torch.arange(-xm, xm + 1).float()
        k  = 2 * xm + 1
        x  = x0.view(-1, 1).repeat(self.channel_out, self.channel_in, 1, k).to(self.sigma.device)
        y  = y0.view(1, -1).repeat(self.channel_out, self.channel_in, k, 1).to(self.sigma.device)
        th = self.theta.view(-1, 1, 1, 1)
        x_t = x * torch.cos(th) + y * torch.sin(th)
        y_t = -x * torch.sin(th) + y * torch.cos(th)
        g   = -torch.exp(-0.5 * ((self.gamma * x_t)**2 + y_t**2)
                         / (8 * self.sigma.view(-1,1,1,1)**2)) \
              * torch.cos(2 * math.pi * self.frequency.view(-1,1,1,1) * x_t
                          + self.psi.view(-1,1,1,1))
        return g - g.mean(dim=[2, 3], keepdim=True)

    def forward(self, x):
        return F.conv2d(x, self._get_gabor_kernel(),
                        stride=self.stride, padding=self.padding)


# ══════════════════════════════════════════════════════════════
#  COMPREHENSIVE ATTENTION BLOCK (CAB)
# ══════════════════════════════════════════════════════════════

class XDirectionAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(1, channels // reduction)
        self.conv1 = nn.Conv2d(channels, r, 1)
        self.conv2 = nn.Conv2d(r, channels, 1)
    def forward(self, x):
        a = torch.mean(x, dim=2, keepdim=True)
        a = torch.sigmoid(self.conv2(F.relu(self.conv1(a))))
        return x * a

class YDirectionAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(1, channels // reduction)
        self.conv1 = nn.Conv2d(channels, r, 1)
        self.conv2 = nn.Conv2d(r, channels, 1)
    def forward(self, x):
        a = torch.mean(x, dim=3, keepdim=True)
        a = torch.sigmoid(self.conv2(F.relu(self.conv1(a))))
        return x * a

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(1, channels // reduction)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, r, 1)
        self.conv2 = nn.Conv2d(r, channels, 1)
    def forward(self, x):
        a = self.pool(x)
        a = torch.sigmoid(self.conv2(F.relu(self.conv1(a))))
        return x * a

class PixelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        a = torch.sigmoid(self.conv2(F.relu(self.conv1(x))))
        return x * a

class CAB(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.xa    = XDirectionAttention(channels, reduction)
        self.ya    = YDirectionAttention(channels, reduction)
        self.ca    = ChannelAttention(channels, reduction)
        self.pa    = PixelAttention(channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        sa  = self.alpha * self.xa(x) + self.beta * self.ya(x)
        ca  = self.ca(sa)
        return self.pa(ca)


# ══════════════════════════════════════════════════════════════
#  TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner  = dim_head * heads
        self.heads = heads; self.scale = dim_head ** -0.5
        self.norm  = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.drop  = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
    def forward(self, x):
        x   = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t,'b n (h d)->b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1,-2)) * self.scale
        attn = self.drop(self.attend(dots))
        out  = rearrange(torch.matmul(attn, v), 'b h n d->b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.ff   = FeedForward(dim, mlp_dim, dropout)
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


# ══════════════════════════════════════════════════════════════
#  FIX 1: TransformerBranch — pos_embedding pre-allocated in __init__
# ══════════════════════════════════════════════════════════════

class TransformerBranch(nn.Module):
    """
    FIX 1: pos_embedding is now an nn.Parameter created in __init__ with
    the correct size derived from spatial_size and patch_size. The original
    code created it lazily inside forward() which meant the optimizer never
    saw it and it received no gradient updates throughout training.
    """
    def __init__(self, in_channels, spatial_size, patch_size=8,
                 dim=128, depth=2, heads=8, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.dim        = dim

        num_patches_side = spatial_size // patch_size
        num_patches      = num_patches_side * num_patches_side
        patch_dim        = in_channels * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # FIX 1: allocated here, registered with the optimizer from the start
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)

        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, max(1, dim // heads), mlp_dim, dropout)
            for _ in range(depth)])
        self.norm    = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        nh = H // self.patch_size
        nw = W // self.patch_size
        x  = self.to_patch_embedding(x)          # [B, N, dim]
        x  = x + self.pos_embedding               # works because N is fixed per level
        x  = self.dropout(x)
        for blk in self.transformer:
            x = blk(x)
        x = self.norm(x)
        return rearrange(x, 'b (h w) d -> b d h w', h=nh, w=nw)


# ══════════════════════════════════════════════════════════════
#  CNN BRANCH
# ══════════════════════════════════════════════════════════════

class CNNBranch(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.cab = CAB(out_channels, reduction)
    def forward(self, x):
        return self.cab(self.conv(x))


# ══════════════════════════════════════════════════════════════
#  FIX 2: PHEB — pre_proj before Gabor, residual shortcut, no dead is_first
# ══════════════════════════════════════════════════════════════

class PHEB(nn.Module):
    """
    FIX 2a: pre_proj compresses in_channels → 1 before the Gabor filter.
      The original applied Gabor to 64/128/256-channel maps which (a) created
      a severe information bottleneck (256→36) and (b) is architecturally
      inconsistent with how Gabor filters work on texture images.

    FIX 2b: residual shortcut (1×1 conv) added around the whole block so
      gradients flow back to the encoder without traversing the Gabor path.

    FIX 3: removed dead `is_first` branch (both sides were identical).

    spatial_size must match the spatial resolution of the input feature map
    so TransformerBranch can pre-allocate pos_embedding correctly.
    """
    def __init__(self, in_channels, out_channels, gabor_filters=36, gabor_kernel=17,
                 patch_size=8, trans_dim=128, trans_depth=2, trans_heads=8,
                 ca_reduction=16, spatial_size=128):
        super().__init__()

        # FIX 2a: project to single channel before Gabor
        self.pre_proj = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1))

        self.gabor = GaborConv2d(
            channel_in=1, channel_out=gabor_filters,
            kernel_size=gabor_kernel, stride=1, padding=gabor_kernel // 2,
            init_ratio=0.5)
        self.gabor_bn = nn.BatchNorm2d(gabor_filters)

        self.cnn_branch   = CNNBranch(gabor_filters, out_channels // 2, ca_reduction)

        # FIX 1: spatial_size propagated here
        self.trans_branch = TransformerBranch(
            in_channels=gabor_filters, spatial_size=spatial_size,
            patch_size=patch_size, dim=trans_dim,
            depth=trans_depth, heads=trans_heads,
            mlp_dim=trans_dim * 2, dropout=0.1)

        self.trans_proj = nn.Sequential(
            nn.Conv2d(trans_dim, out_channels // 2, 1),
            nn.BatchNorm2d(out_channels // 2), nn.ReLU(inplace=True))

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

        # FIX 2b: residual shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        res = self.shortcut(x)

        # FIX 2a: single-channel projection before Gabor
        proj = F.relu(self.pre_proj(x))

        g = F.relu(self.gabor_bn(self.gabor(proj)))

        cnn_out   = self.cnn_branch(g)
        trans_out = self.trans_proj(self.trans_branch(g))

        if trans_out.shape[2:] != cnn_out.shape[2:]:
            trans_out = F.interpolate(trans_out, size=cnn_out.shape[2:],
                                      mode='bilinear', align_corners=False)

        out = self.fusion(torch.cat([cnn_out, trans_out], dim=1))
        return F.relu(out + res)   # FIX 2b: residual


# ══════════════════════════════════════════════════════════════
#  DOWN / UP SAMPLING
# ══════════════════════════════════════════════════════════════

class DownSample(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(ic, oc, 4, stride=2, padding=1),
            nn.BatchNorm2d(oc), nn.ReLU(inplace=True))
    def forward(self, x): return self.down(x)

class UpSample(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ic, oc, 4, stride=2, padding=1),
            nn.BatchNorm2d(oc), nn.ReLU(inplace=True))
    def forward(self, x): return self.up(x)


# ══════════════════════════════════════════════════════════════
#  FIX 3: MSPHNet — passes spatial_size to each PHEB
# ══════════════════════════════════════════════════════════════

class MSPHNet(nn.Module):
    """
    FIX 3: each PHEB now receives the correct spatial_size for its level
    so the TransformerBranch pre-allocates the right pos_embedding.
    Spatial sizes (with img_side=128):
      pheb1: 128×128
      pheb2: 64×64   (after down1)
      pheb3: 32×32   (after down2, bottleneck)
      pheb4: 64×64   (after up1 + skip)
      pheb5: 128×128 (after up2 + skip)
    """
    def __init__(self, num_classes, img_side=128, embedding_dim=1024,
                 gabor_filters=36, gabor_kernel=17,
                 trans_dim=128, trans_depth=2, trans_heads=8, patch_size=8,
                 ca_reduction=16, dropout=0.5, arcface_s=30.0, arcface_m=0.50):
        super().__init__()
        self.num_classes   = num_classes
        self.embedding_dim = embedding_dim

        c1, c2, c3 = 64, 128, 256
        s1 = img_side          # 128
        s2 = img_side // 2     # 64
        s3 = img_side // 4     # 32

        self.init_conv = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True))

        pheb_kw = dict(gabor_filters=gabor_filters, gabor_kernel=gabor_kernel,
                       patch_size=patch_size, trans_dim=trans_dim,
                       trans_depth=trans_depth, trans_heads=trans_heads,
                       ca_reduction=ca_reduction)

        # Encoder
        self.pheb1 = PHEB(c1,    c1,    spatial_size=s1, **pheb_kw)
        self.down1 = DownSample(c1, c2)
        self.pheb2 = PHEB(c2,    c2,    spatial_size=s2, **pheb_kw)
        self.down2 = DownSample(c2, c3)
        self.pheb3 = PHEB(c3,    c3,    spatial_size=s3, **pheb_kw)

        # Decoder
        self.up1   = UpSample(c3, c2)
        self.pheb4 = PHEB(c2*2,  c2,    spatial_size=s2, **pheb_kw)
        self.up2   = UpSample(c2, c1)
        self.pheb5 = PHEB(c1*2,  c1,    spatial_size=s1, **pheb_kw)

        self.final_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.final_conv = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True))

        fc_dim = c3 * 4 * 4   # 4096
        self.fc = nn.Sequential(
            nn.Linear(fc_dim, 2048), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(2048, embedding_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))

        self.arcface = ArcMarginProduct(embedding_dim, num_classes,
                                        s=arcface_s, m=arcface_m)

    def _encode(self, x):
        x  = self.init_conv(x)
        e1 = self.pheb1(x)
        d1 = self.down1(e1);  e2 = self.pheb2(d1)
        d2 = self.down2(e2);  e3 = self.pheb3(d2)
        u1 = self.up1(e3);    u1 = torch.cat([u1, e2], dim=1)
        r1 = self.pheb4(u1)
        u2 = self.up2(r1);    u2 = torch.cat([u2, e1], dim=1)
        r2 = self.pheb5(u2)
        feat = self.final_conv(self.final_pool(r2))
        return self.fc(feat.view(feat.size(0), -1))

    def forward(self, x, y=None):
        emb    = self._encode(x)
        output = self.arcface(emb, y)
        return output, F.normalize(emb, dim=-1)

    def get_feature_vector(self, x):
        return F.normalize(self._encode(x), dim=-1)


# ══════════════════════════════════════════════════════════════
#  NORMALISATION
# ══════════════════════════════════════════════════════════════

class NormSingleROI(object):
    def __init__(self, outchannels=1): self.outchannels = outchannels
    def __call__(self, tensor):
        c, h, w = tensor.size()
        t = tensor.view(c, h * w)
        idx = t > 0; vals = t[idx]
        if len(vals) > 0:
            t[idx] = (vals - vals.mean()) / (vals.std() + 1e-6)
        tensor = t.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, self.outchannels, dim=0)
        return tensor


# ══════════════════════════════════════════════════════════════
#  DATASET — CASIA-MS-ROI
# ══════════════════════════════════════════════════════════════

def parse_casia_ms(data_root):
    id2paths = defaultdict(list)
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")):
            continue
        parts = fname.split("_")
        if len(parts) < 4: continue
        identity = parts[0] + "_" + parts[1]
        id2paths[identity].append(os.path.join(data_root, fname))
    return dict(id2paths)


def split_closed_set(id2paths, train_ratio=0.8, seed=42):
    rng = random.Random(seed)
    label_map, train_samples, test_samples = {}, [], []
    for idx, (identity, paths) in enumerate(sorted(id2paths.items())):
        label_map[identity] = idx
        ps = list(paths); rng.shuffle(ps)
        n  = max(1, int(len(ps) * train_ratio))
        for p in ps[:n]:  train_samples.append((p, idx))
        for p in ps[n:]:  test_samples.append((p, idx))
    return train_samples, test_samples, label_map


def split_open_set(id2paths, train_ratio=0.8, gallery_ratio=0.5,
                   val_ratio=0.10, seed=42):
    rng = random.Random(seed)
    ids = sorted(id2paths.keys()); rng.shuffle(ids)
    n_tr  = max(1, int(len(ids) * train_ratio))
    train_ids, test_ids = ids[:n_tr], ids[n_tr:]

    train_label_map = {k: i for i, k in enumerate(sorted(train_ids))}
    all_tr = [(p, train_label_map[id]) for id in train_ids for p in id2paths[id]]

    rng2 = random.Random(seed + 1); rng2.shuffle(all_tr)
    n_v  = max(1, int(len(all_tr) * val_ratio))
    val_samples, train_samples = all_tr[:n_v], all_tr[n_v:]

    test_label_map = {k: i for i, k in enumerate(sorted(test_ids))}
    gallery_samples, probe_samples = [], []
    for id in test_ids:
        lab = test_label_map[id]
        ps  = list(id2paths[id]); rng.shuffle(ps)
        n_g = max(1, int(len(ps) * gallery_ratio))
        for p in ps[:n_g]: gallery_samples.append((p, lab))
        for p in ps[n_g:]: probe_samples.append((p, lab))

    return (train_samples, val_samples, gallery_samples, probe_samples,
            train_label_map, test_label_map)


# FIX 5: __getitem__ wraps open() in try/except so a single corrupt file
#         doesn't crash the DataLoader worker and terminate the whole run.
class CASIAMSDataset(Dataset):
    def __init__(self, samples, img_side=128, train=True):
        self.samples = samples
        self.train   = train
        if train:
            self.transform = T.Compose([
                T.Resize(img_side),
                T.RandomChoice([
                    T.ColorJitter(brightness=0, contrast=0.05),
                    T.RandomResizedCrop(img_side, scale=(0.8,1.0), ratio=(1.,1.)),
                    T.RandomPerspective(distortion_scale=0.15, p=1),
                    T.RandomChoice([
                        T.RandomRotation(10, interpolation=Image.BICUBIC, expand=False,
                                         center=(0.5*img_side, 0.0)),
                        T.RandomRotation(10, interpolation=Image.BICUBIC, expand=False,
                                         center=(0.0, 0.5*img_side)),
                    ]),
                ]),
                T.ToTensor(), NormSingleROI(1)])
        else:
            self.transform = T.Compose([
                T.Resize(img_side), T.ToTensor(), NormSingleROI(1)])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("L")
            return self.transform(img), label
        except Exception as e:
            # FIX 5: corrupt file → return a neighbouring sample
            print(f"  [warn] could not load {path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))


class CASIAMSDatasetSingle(Dataset):
    def __init__(self, samples, img_side=128):
        self.samples   = samples
        self.transform = T.Compose([T.Resize(img_side), T.ToTensor(), NormSingleROI(1)])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            return self.transform(Image.open(path).convert("L")), label
        except Exception as e:
            print(f"  [warn] {path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))


# ══════════════════════════════════════════════════════════════
#  FIX 6: run_one_epoch — validation also passes labels so ArcFace
#          computes the loss with margin (consistent with training).
# ══════════════════════════════════════════════════════════════

def run_one_epoch(epoch, model, loader, criterion, optimizer, device, phase):
    is_train = (phase == "training")
    model.train() if is_train else model.eval()

    running_loss = running_correct = total = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            if is_train: optimizer.zero_grad()

            # FIX 6: pass labels during validation too
            output, _ = model(imgs, targets)
            loss = criterion(output, targets)

            if is_train:
                loss.backward(); optimizer.step()

            running_loss    += loss.item() * imgs.size(0)
            running_correct += output.data.max(1)[1].eq(targets).sum().item()
            total           += imgs.size(0)

    return running_loss / max(total,1), 100.0 * running_correct / max(total,1)


# ══════════════════════════════════════════════════════════════
#  EVALUATION — EER + Rank-1
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    for imgs, labs in loader:
        feats.append(model.get_feature_vector(imgs.to(device)).cpu().numpy())
        labels.append(labs.numpy() if not isinstance(labs, list) else np.array(labs))
    return np.concatenate(feats), np.concatenate(labels)


def compute_eer(scores_arr):
    inscore  = scores_arr[scores_arr[:,1]==1,  0]
    outscore = scores_arr[scores_arr[:,1]==-1, 0]
    if len(inscore)==0 or len(outscore)==0: return 1.0, 0.0
    if inscore.mean() < outscore.mean():
        inscore, outscore = -inscore, -outscore
    y   = np.concatenate([np.ones(len(inscore)), np.zeros(len(outscore))])
    s   = np.concatenate([inscore, outscore])
    fpr, tpr, thr = roc_curve(y, s, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer, float(interp1d(fpr, thr)(eer))


def evaluate(model, probe_loader, gallery_loader, device, out_dir=".", tag="eval"):
    pf, pl = extract_features(model, probe_loader,   device)
    gf, gl = extract_features(model, gallery_loader, device)
    np_, ng = len(pf), len(gf)

    scores_list, labels_list = [], []
    dist_matrix = np.zeros((np_, ng))
    for i in range(np_):
        cs   = np.dot(gf, pf[i])
        dists = np.arccos(np.clip(cs, -1, 1)) / np.pi
        dist_matrix[i] = dists
        for j in range(ng):
            scores_list.append(dists[j])
            labels_list.append(1 if pl[i]==gl[j] else -1)

    scores_arr = np.column_stack([scores_list, labels_list])
    pair_eer, _ = compute_eer(scores_arr)

    aggr_s, aggr_l = [], []
    for i in range(np_-1):
        for j in range(i+1, np_):
            d = np.arccos(np.clip(np.dot(pf[i], pf[j]), -1, 1)) / np.pi
            aggr_s.append(d); aggr_l.append(1 if pl[i]==pl[j] else -1)
    aggr_eer = compute_eer(np.column_stack([aggr_s, aggr_l]))[0] if aggr_s else 1.0

    rank1 = 100.0 * sum(pl[i]==gl[np.argmin(dist_matrix[i])] for i in range(np_)) / max(np_,1)

    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for sv, lv in zip(scores_list, labels_list): f.write(f"{sv} {lv}\n")
    _save_roc_det(scores_arr, out_dir, tag)
    print(f"  [{tag}]  pairEER={pair_eer*100:.4f}%  aggrEER={aggr_eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return pair_eer, aggr_eer, rank1


def _save_roc_det(scores_arr, out_dir, tag):
    ins = scores_arr[scores_arr[:,1]==1, 0]
    out = scores_arr[scores_arr[:,1]==-1,0]
    if len(ins)==0 or len(out)==0: return
    if ins.mean() < out.mean(): ins, out = -ins, -out
    y = np.concatenate([np.ones(len(ins)), np.zeros(len(out))])
    s = np.concatenate([ins, out])
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    try:
        pdf = PdfPages(os.path.join(out_dir, f"roc_det_{tag}.pdf"))
        for (xd,yd,xl,yl,t) in [
            (fpr*100, tpr*100, 'FAR(%)', 'GAR(%)', 'ROC'),
            (fpr*100, fnr*100, 'FAR(%)', 'FRR(%)', 'DET')]:
            fig, ax = plt.subplots()
            ax.plot(xd, yd, 'b-^', markersize=2); ax.grid(True)
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{t} — {tag}")
            pdf.savefig(fig); plt.close(fig)
        pdf.close()
    except Exception as e:
        print(f"  [warn] plot: {e}")


def plot_loss_acc(tl, vl, ta, va, results_dir):
    try:
        ep = range(1, len(tl)+1)
        for (d1,d2,lbl,fn) in [(tl,vl,'loss','losses.png'),(ta,va,'accuracy (%)','accuracy.png')]:
            fig, ax = plt.subplots()
            ax.plot(ep, d1, 'b', label='train'); ax.plot(ep, d2, 'r', label='val')
            ax.legend(); ax.set_xlabel('epoch'); ax.set_ylabel(lbl)
            fig.savefig(os.path.join(results_dir, fn)); plt.close(fig)
    except Exception: pass


# ══════════════════════════════════════════════════════════════
#  FIX 4: MAIN — CosineAnnealingLR + 10-epoch linear warmup
# ══════════════════════════════════════════════════════════════

def main():
    C           = CONFIG
    protocol    = C["protocol"]
    data_root   = C["data_root"]
    results_dir = C["results_dir"]
    img_side    = C["img_side"]
    batch_size  = C["batch_size"]
    num_epochs  = C["num_epochs"]
    lr          = C["lr"]
    warmup_ep   = C["warmup_epochs"]
    dropout     = C["dropout"]
    embedding_dim = C["embedding_dim"]
    gabor_filters = C["gabor_filters"]
    gabor_kernel  = C["gabor_kernel"]
    trans_depth   = C["transformer_depth"]
    trans_dim     = C["transformer_dim"]
    trans_heads   = C["transformer_heads"]
    patch_size    = C["patch_size"]
    ca_reduction  = C["ca_reduction"]
    train_ratio   = C["train_ratio"]
    gallery_ratio = C["gallery_ratio"]
    val_ratio     = C["val_ratio"]
    seed          = C["random_seed"]
    save_every    = C["save_every"]
    eval_every    = C["eval_every"]
    nw            = C["num_workers"]

    assert protocol in ("closed-set","open-set")
    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  MSPHNet (fixed) on CASIA-MS")
    print(f"  Protocol : {protocol}  |  Device : {device}")
    print(f"  Patch: {patch_size}×{patch_size}  |  Heads: {trans_heads}")
    print(f"  lr: {lr}  |  Batch: {batch_size}  |  Warmup: {warmup_ep} ep")
    print(f"{'='*60}\n")

    print("Scanning dataset …")
    id2paths    = parse_casia_ms(data_root)
    n_ids       = len(id2paths)
    n_imgs      = sum(len(v) for v in id2paths.values())
    print(f"  Found {n_ids} identities, {n_imgs} images.\n")

    # FIX 5: persistent_workers avoids worker restart on each epoch
    dl_kw = dict(batch_size=batch_size, num_workers=nw,
                 pin_memory=True, persistent_workers=(nw>0))

    if protocol == "closed-set":
        tr_s, te_s, lmap = split_closed_set(id2paths, train_ratio, seed)
        num_classes = len(lmap)
        train_loader   = DataLoader(CASIAMSDataset(tr_s, img_side, True),  shuffle=True,  **dl_kw)
        val_loader     = DataLoader(CASIAMSDataset(te_s, img_side, False), shuffle=False, **dl_kw)
        gallery_loader = DataLoader(CASIAMSDatasetSingle(tr_s, img_side),  shuffle=False, **dl_kw)
        probe_loader   = DataLoader(CASIAMSDatasetSingle(te_s, img_side),  shuffle=False, **dl_kw)
        print(f"  [closed-set] #classes={num_classes}")
    else:
        (tr_s, va_s, ga_s, pr_s, tr_lmap, _) = split_open_set(
            id2paths, train_ratio, gallery_ratio, val_ratio, seed)
        num_classes = len(tr_lmap)
        train_loader   = DataLoader(CASIAMSDataset(tr_s, img_side, True),  shuffle=True,  **dl_kw)
        val_loader     = DataLoader(CASIAMSDataset(va_s, img_side, False), shuffle=False, **dl_kw)
        gallery_loader = DataLoader(CASIAMSDatasetSingle(ga_s, img_side),  shuffle=False, **dl_kw)
        probe_loader   = DataLoader(CASIAMSDatasetSingle(pr_s, img_side),  shuffle=False, **dl_kw)
        print(f"  [open-set] #train_classes={num_classes}")

    print(f"\nBuilding MSPHNet — num_classes={num_classes} …")
    net = MSPHNet(
        num_classes=num_classes, img_side=img_side,
        embedding_dim=embedding_dim, gabor_filters=gabor_filters,
        gabor_kernel=gabor_kernel, trans_dim=trans_dim,
        trans_depth=trans_depth, trans_heads=trans_heads,
        patch_size=patch_size, ca_reduction=ca_reduction,
        dropout=dropout, arcface_s=C["arcface_s"], arcface_m=C["arcface_m"])
    net.to(device)

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}\n")

    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
        net = DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # FIX 4: CosineAnnealingLR instead of StepLR
    cos_sched = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.01)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0
    best_eer     = 1.0
    last_eer, last_rank1 = float("nan"), float("nan")

    print(f"Starting training for {num_epochs} epochs …")
    print(f"  EER / Rank-1 computed every {eval_every} epochs.\n")

    for epoch in range(num_epochs):

        # FIX 4: linear LR warmup for the first `warmup_ep` epochs
        if epoch < warmup_ep:
            warm_lr = lr * (epoch + 1) / warmup_ep
            for pg in optimizer.param_groups:
                pg["lr"] = warm_lr

        t_loss, t_acc = run_one_epoch(epoch, net, train_loader,
                                       criterion, optimizer, device, "training")
        v_loss, v_acc = run_one_epoch(epoch, net, val_loader,
                                       criterion, optimizer, device, "testing")

        if epoch >= warmup_ep:
            cos_sched.step()

        train_losses.append(t_loss); val_losses.append(v_loss)
        train_accs.append(t_acc);   val_accs.append(v_acc)

        _net = net.module if isinstance(net, DataParallel) else net

        if epoch % eval_every == 0 or epoch == num_epochs-1:
            tag = f"ep{epoch:04d}_{protocol.replace('-','')}"
            cur_eer, cur_aggr_eer, cur_rank1 = evaluate(
                _net, probe_loader, gallery_loader, device, rst_eval, tag)
            last_eer, last_rank1 = cur_eer, cur_rank1
            if cur_eer < best_eer:
                best_eer = cur_eer
                torch.save(_net.state_dict(),
                           os.path.join(results_dir,"net_params_best_eer.pth"))
                print(f"  *** New best EER: {best_eer*100:.4f}% ***")

        if epoch % 10 == 0 or epoch == num_epochs-1:
            ts  = time.strftime("%H:%M:%S")
            cur_lr  = optimizer.param_groups[0]["lr"]
            eer_str  = f"{last_eer*100:.4f}%"  if not math.isnan(last_eer)   else "N/A"
            r1_str   = f"{last_rank1:.2f}%"    if not math.isnan(last_rank1) else "N/A"
            print(f"[{ts}] ep {epoch:04d} | lr={cur_lr:.6f} | "
                  f"loss t={t_loss:.5f} v={v_loss:.5f} | "
                  f"acc t={t_acc:.2f}% v={v_acc:.2f}% | "
                  f"EER={eer_str} R1={r1_str}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(_net.state_dict(), os.path.join(results_dir,"net_params_best.pth"))

        if epoch % save_every == 0 or epoch == num_epochs-1:
            torch.save(_net.state_dict(), os.path.join(results_dir,"net_params.pth"))
            plot_loss_acc(train_losses, val_losses, train_accs, val_accs, results_dir)

    print("\n=== Final evaluation with best EER model ===")
    best_path = os.path.join(results_dir,"net_params_best_eer.pth")
    if not os.path.exists(best_path):
        best_path = os.path.join(results_dir,"net_params_best.pth")

    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_path, map_location=device))
    final_eer, final_aggr, final_r1 = evaluate(
        eval_net, probe_loader, gallery_loader, device,
        rst_eval, f"FINAL_{protocol.replace('-','')}")

    print(f"\n{'='*60}")
    print(f"  PROTOCOL : {protocol}")
    print(f"  FINAL Pairwise EER   : {final_eer*100:.4f}%")
    print(f"  FINAL Aggregated EER : {final_aggr*100:.4f}%")
    print(f"  FINAL Rank-1         : {final_r1:.3f}%")
    print(f"  Results saved to     : {results_dir}")
    print(f"{'='*60}\n")

    with open(os.path.join(results_dir,"summary.txt"),"w") as f:
        f.write(f"Protocol  : {protocol}\n")
        f.write(f"Data root : {data_root}\n")
        f.write(f"Identities: {n_ids}\n")
        f.write(f"Images    : {n_imgs}\n")
        f.write(f"Classes (train): {num_classes}\n")
        f.write(f"Best val acc       : {best_val_acc:.3f}%\n")
        f.write(f"Final Pairwise EER : {final_eer*100:.6f}%\n")
        f.write(f"Final Aggreg. EER  : {final_aggr*100:.6f}%\n")
        f.write(f"Final Rank-1       : {final_r1:.3f}%\n")


if __name__ == "__main__":
    main()
