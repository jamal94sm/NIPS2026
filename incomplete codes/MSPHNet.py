"""
MSPHNet on CASIA-MS Dataset (Paper-Faithful Implementation)
============================================================
Multi-Scale Parallel Hybrid Network for Palmprint Recognition (ICASSP 2025)

This implementation follows the paper exactly:
  - PHEB: Gabor + CNN(CAB) + Transformer branches
  - CAB: XA + YA → CA → PA (Equations 1-4)
  - Transformer: 8×8 patches, 8 attention heads (Equation 5)
  - U-Net encoder-decoder with skip connections

Speed optimization: Patch-based attention (8×8 patches = 64 patches per 64×64 map)
  → Attention is O(64²) = 4096, not O(4096²) = 16M
"""

# ==============================================================
#  CONFIG
# ==============================================================
CONFIG = {
    "protocol"        : "open-set",
    "data_root"       : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "results_dir"     : "./rst_msphnet_casia_ms",
    "img_side"        : 128,
    "batch_size"      : 64,
    "num_epochs"      : 1000,
    "lr"              : 0.001,
    "lr_step"         : 500,
    "lr_gamma"        : 0.8,
    "dropout"         : 0.5,
    "arcface_s"       : 30.0,
    "arcface_m"       : 0.50,
    "embedding_dim"   : 1024,
    "gabor_filters"   : 36,
    "gabor_kernel"    : 17,
    "patch_size"      : 8,              # Paper: 8×8 patches
    "num_heads"       : 8,              # Paper: 8 attention heads
    "ca_reduction"    : 16,
    "train_ratio"     : 0.60,
    "gallery_ratio"   : 0.10,
    "val_ratio"       : 0.10,
    "random_seed"     : 42,
    "save_every"      : 10,
    "eval_every"      : 50,
    "num_workers"     : 4,
}

import os
import math
import time
import random
import warnings
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

warnings.filterwarnings("ignore")

SEED = CONFIG["random_seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════
#  ARCFACE
# ══════════════════════════════════════════════════════════════

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.training and label is not None:
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            output = self.s * cosine
        return output


# ══════════════════════════════════════════════════════════════
#  LEARNABLE GABOR FILTER
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    """Learnable Gabor Convolution for texture extraction."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Learnable parameters
        self.sigma = nn.Parameter(torch.tensor([4.6]))
        self.gamma = nn.Parameter(torch.tensor([2.0]))
        self.freq = nn.Parameter(torch.tensor([0.114]))

        # Fixed orientations
        theta = torch.arange(0, out_ch).float() * math.pi / out_ch
        self.register_buffer('theta', theta)

        # Coordinate grids
        half = kernel_size // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1).float(),
            torch.arange(-half, half + 1).float(),
            indexing='ij'
        )
        self.register_buffer('x_grid', x)
        self.register_buffer('y_grid', y)

    def forward(self, x):
        theta = self.theta.view(-1, 1, 1)
        x_theta = self.x_grid * torch.cos(theta) + self.y_grid * torch.sin(theta)
        y_theta = -self.x_grid * torch.sin(theta) + self.y_grid * torch.cos(theta)

        sigma_sq = 2 * self.sigma ** 2
        gabor = torch.exp(-((self.gamma ** 2) * (x_theta ** 2) + y_theta ** 2) / sigma_sq)
        gabor = gabor * torch.cos(2 * math.pi * self.freq * x_theta)
        gabor = gabor - gabor.mean(dim=[1, 2], keepdim=True)

        kernel = gabor.unsqueeze(1).repeat(1, self.in_ch, 1, 1)
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


# ══════════════════════════════════════════════════════════════
#  CAB - Comprehensive Attention Block (Paper Equations 1-4)
# ══════════════════════════════════════════════════════════════

class XAttention(nn.Module):
    """X-direction Attention (Eq. 1): Pool along height, attend along width."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Mean over H: [B, C, H, W] → [B, C, 1, W]
        attn = x.mean(dim=2, keepdim=True)
        attn = self.fc(attn)
        return x * attn


class YAttention(nn.Module):
    """Y-direction Attention (Eq. 2): Pool along width, attend along height."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Mean over W: [B, C, H, W] → [B, C, H, 1]
        attn = x.mean(dim=3, keepdim=True)
        attn = self.fc(attn)
        return x * attn


class ChannelAttention(nn.Module):
    """Channel Attention (Eq. 3): Global pooling → channel weights."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class PixelAttention(nn.Module):
    """Pixel Attention (Eq. 4): Per-pixel attention weights."""
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class CAB(nn.Module):
    """
    Comprehensive Attention Block (Paper Equations 1-4).
    Flow: Input → (XA + YA weighted sum) → CA → PA → Output
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.xa = XAttention(channels, reduction)
        self.ya = YAttention(channels, reduction)
        self.ca = ChannelAttention(channels, reduction)
        self.pa = PixelAttention(channels)

        # Learnable weights for XA + YA combination
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # Spatial attention: weighted sum of XA and YA
        xa_out = self.xa(x)
        ya_out = self.ya(x)
        sa_out = self.alpha * xa_out + self.beta * ya_out

        # Channel attention
        ca_out = self.ca(sa_out)

        # Pixel attention
        pa_out = self.pa(ca_out)

        return pa_out


# ══════════════════════════════════════════════════════════════
#  TRANSFORMER BRANCH (Paper: 8×8 patches, 8 heads)
# ══════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """Convert feature map to patch embeddings."""
    def __init__(self, in_channels, embed_dim, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, H, W] → [B, embed_dim, H/P, W/P]
        x = self.proj(x)
        B, C, H, W = x.shape
        # Reshape to [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention (Paper Eq. 5)."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and FFN."""
    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBranch(nn.Module):
    """
    Transformer-based branch for global features (Paper description).
    Uses 8×8 patches and 8 attention heads.
    """
    def __init__(self, in_channels, out_channels, patch_size=8, num_heads=8, depth=1):
        super().__init__()
        self.patch_size = patch_size
        embed_dim = out_channels

        # Patch embedding
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)

        # Positional embedding (learnable)
        self.pos_embed = None  # Will be initialized dynamically

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=2.0, dropout=0.1)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Project back to spatial
        self.proj_out = nn.Conv2d(embed_dim, out_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch embedding
        x, pH, pW = self.patch_embed(x)  # [B, num_patches, embed_dim]
        num_patches = pH * pW

        # Add positional embedding
        if self.pos_embed is None or self.pos_embed.size(1) != num_patches:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, x.size(-1), device=x.device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        x = x + self.pos_embed[:, :num_patches]

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Reshape back to spatial: [B, num_patches, C] → [B, C, pH, pW]
        x = x.transpose(1, 2).reshape(B, -1, pH, pW)

        # Upsample to original resolution
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = self.proj_out(x)

        return x


# ══════════════════════════════════════════════════════════════
#  CNN BRANCH (with CAB)
# ══════════════════════════════════════════════════════════════

class CNNBranch(nn.Module):
    """CNN-based branch with CAB for local features."""
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cab = CAB(out_channels, reduction)

    def forward(self, x):
        return self.cab(self.conv(x))


# ══════════════════════════════════════════════════════════════
#  PHEB - Parallel Hybrid Feature Extraction Block
# ══════════════════════════════════════════════════════════════

class PHEB(nn.Module):
    """
    Parallel Hybrid Feature Extraction Block (Paper architecture).
    - Learnable Gabor filter for texture
    - CNN branch with CAB for local features
    - Transformer branch for global features
    - Concatenation + Conv + BN
    """
    def __init__(self, in_ch, out_ch, gabor_filters=36, gabor_kernel=17,
                 patch_size=8, num_heads=8, reduction=16, use_gabor=True):
        super().__init__()
        self.use_gabor = use_gabor

        # Gabor filter
        if use_gabor:
            self.gabor = nn.Sequential(
                GaborConv2d(in_ch, gabor_filters, gabor_kernel, padding=gabor_kernel // 2),
                nn.BatchNorm2d(gabor_filters),
                nn.ReLU(inplace=True)
            )
            branch_in = gabor_filters
        else:
            self.gabor = None
            branch_in = in_ch

        # CNN branch (local features)
        cnn_out = out_ch // 2
        self.cnn_branch = CNNBranch(branch_in, cnn_out, reduction)

        # Transformer branch (global features)
        trans_out = out_ch - cnn_out
        self.trans_branch = TransformerBranch(branch_in, trans_out, patch_size, num_heads, depth=1)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.gabor is not None:
            x = self.gabor(x)

        cnn_out = self.cnn_branch(x)
        trans_out = self.trans_branch(x)

        concat = torch.cat([cnn_out, trans_out], dim=1)
        return self.fusion(concat)


# ══════════════════════════════════════════════════════════════
#  ENCODER-DECODER BLOCKS
# ══════════════════════════════════════════════════════════════

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


# ══════════════════════════════════════════════════════════════
#  MSPHNet
# ══════════════════════════════════════════════════════════════

class MSPHNet(nn.Module):
    """
    Multi-Scale Parallel Hybrid Network (Paper-faithful).

    Architecture:
    - Encoder: PHEB → Down → PHEB → Down → PHEB (bottleneck)
    - Decoder: Up → PHEB (skip) → Up → PHEB (skip)
    - Final: Pool → Conv → FC → Embedding
    """
    def __init__(self, num_classes, embedding_dim=1024, gabor_filters=36,
                 gabor_kernel=17, patch_size=8, num_heads=8, reduction=16,
                 dropout=0.5, arcface_s=30.0, arcface_m=0.50):
        super().__init__()

        c1, c2, c3 = 64, 128, 256

        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        # Encoder
        self.pheb1 = PHEB(c1, c1, gabor_filters, gabor_kernel, patch_size, num_heads, reduction, use_gabor=True)
        self.down1 = DownBlock(c1, c2)

        self.pheb2 = PHEB(c2, c2, gabor_filters, gabor_kernel, patch_size, num_heads, reduction, use_gabor=False)
        self.down2 = DownBlock(c2, c3)

        # Bottleneck
        self.pheb3 = PHEB(c3, c3, gabor_filters, gabor_kernel, patch_size, num_heads, reduction, use_gabor=False)

        # Decoder
        self.up1 = UpBlock(c3, c2)
        self.pheb4 = PHEB(c2 * 2, c2, gabor_filters, gabor_kernel, patch_size, num_heads, reduction, use_gabor=False)

        self.up2 = UpBlock(c2, c1)
        self.pheb5 = PHEB(c1 * 2, c1, gabor_filters, gabor_kernel, patch_size, num_heads, reduction, use_gabor=False)

        # Final
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        # FC
        self.fc = nn.Sequential(
            nn.Linear(c2 * 16, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Classification
        self.arcface = ArcMarginProduct(embedding_dim, num_classes, arcface_s, arcface_m)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _features(self, x):
        x = self.stem(x)

        e1 = self.pheb1(x)
        e2 = self.pheb2(self.down1(e1))
        e3 = self.pheb3(self.down2(e2))

        d1 = self.pheb4(torch.cat([self.up1(e3), e2], 1))
        d2 = self.pheb5(torch.cat([self.up2(d1), e1], 1))

        feat = self.final(d2)
        return self.fc(feat.view(feat.size(0), -1))

    def forward(self, x, y=None):
        emb = self._features(x)
        out = self.arcface(emb, y)
        return out, F.normalize(emb, dim=-1)

    def get_feature_vector(self, x):
        return F.normalize(self._features(x), dim=-1)


# ══════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════

class NormSingleROI:
    def __call__(self, t):
        c, h, w = t.shape
        t = t.view(c, -1)
        mask = t > 0
        if mask.any():
            vals = t[mask]
            t[mask] = (vals - vals.mean()) / (vals.std() + 1e-6)
        return t.view(c, h, w)


def parse_casia_ms(root):
    id2paths = defaultdict(list)
    for f in sorted(os.listdir(root)):
        if f.lower().endswith(('.jpg', '.jpeg', '.bmp', '.png')):
            parts = f.split('_')
            if len(parts) >= 4:
                id2paths[f"{parts[0]}_{parts[1]}"].append(os.path.join(root, f))
    return dict(id2paths)


def split_closed_set(id2paths, ratio=0.8, seed=42):
    rng = random.Random(seed)
    train, test, lmap = [], [], {}
    for idx, (k, paths) in enumerate(sorted(id2paths.items())):
        lmap[k] = idx
        p = list(paths)
        rng.shuffle(p)
        n = max(1, int(len(p) * ratio))
        train.extend((x, idx) for x in p[:n])
        test.extend((x, idx) for x in p[n:])
    return train, test, lmap


def split_open_set(id2paths, train_r=0.8, gal_r=0.5, val_r=0.1, seed=42):
    rng = random.Random(seed)
    ids = sorted(id2paths.keys())
    rng.shuffle(ids)
    n = max(1, int(len(ids) * train_r))
    train_ids, test_ids = ids[:n], ids[n:]

    train_lm = {k: i for i, k in enumerate(sorted(train_ids))}
    all_train = [(p, train_lm[k]) for k in train_ids for p in id2paths[k]]
    rng2 = random.Random(seed + 1)
    rng2.shuffle(all_train)
    nv = max(1, int(len(all_train) * val_r))
    val, train = all_train[:nv], all_train[nv:]

    test_lm = {k: i for i, k in enumerate(sorted(test_ids))}
    gal, prb = [], []
    for k in test_ids:
        lab = test_lm[k]
        p = list(id2paths[k])
        rng.shuffle(p)
        ng = max(1, int(len(p) * gal_r))
        gal.extend((x, lab) for x in p[:ng])
        prb.extend((x, lab) for x in p[ng:])

    return train, val, gal, prb, train_lm, test_lm


class DS(Dataset):
    def __init__(self, samples, size=128, train=True):
        self.samples = samples
        if train:
            self.tf = T.Compose([
                T.Resize(size),
                T.RandomChoice([
                    T.ColorJitter(contrast=0.05),
                    T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(1., 1.)),
                    T.RandomPerspective(0.15, p=1),
                    T.RandomRotation(10, interpolation=Image.BICUBIC),
                ]),
                T.ToTensor(), NormSingleROI()
            ])
        else:
            self.tf = T.Compose([T.Resize(size), T.ToTensor(), NormSingleROI()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, l = self.samples[i]
        return self.tf(Image.open(p).convert('L')), l


# ══════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, opt, device, train):
    model.train() if train else model.eval()
    loss_sum, correct, total = 0., 0, 0
    for imgs, labs in loader:
        imgs, labs = imgs.to(device), labs.to(device)
        if train:
            opt.zero_grad()
            out, _ = model(imgs, labs)
        else:
            with torch.no_grad():
                out, _ = model(imgs)
        loss = criterion(out, labs)
        loss_sum += loss.item() * imgs.size(0)
        correct += out.argmax(1).eq(labs).sum().item()
        total += imgs.size(0)
        if train:
            loss.backward()
            opt.step()
    return loss_sum / total, 100. * correct / total


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    F, L = [], []
    for imgs, labs in loader:
        F.append(model.get_feature_vector(imgs.to(device)).cpu().numpy())
        L.append(labs.numpy() if isinstance(labs, torch.Tensor) else np.array(labs))
    return np.concatenate(F), np.concatenate(L)


def compute_eer(arr):
    ins, outs = arr[arr[:, 1] == 1, 0], arr[arr[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0:
        return 1., 0.
    if ins.mean() < outs.mean():
        ins, outs = -ins, -outs
    y = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s = np.concatenate([ins, outs])
    fpr, tpr, th = roc_curve(y, s, pos_label=1)
    eer = brentq(lambda x: 1 - x - interp1d(fpr, tpr)(x), 0, 1)
    return eer, float(interp1d(fpr, th)(eer))


def evaluate(model, prb_loader, gal_loader, device, out_dir, tag):
    pf, pl = extract(model, prb_loader, device)
    gf, gl = extract(model, gal_loader, device)
    np_, ng = len(pf), len(gf)

    scores, labels = [], []
    dmat = np.zeros((np_, ng))
    for i in range(np_):
        cos = np.dot(gf, pf[i])
        d = np.arccos(np.clip(cos, -1, 1)) / np.pi
        dmat[i] = d
        for j in range(ng):
            scores.append(d[j])
            labels.append(1 if pl[i] == gl[j] else -1)

    arr = np.column_stack([scores, labels])
    pair_eer, _ = compute_eer(arr)

    # Aggregated EER
    as_, al_ = [], []
    for i in range(np_ - 1):
        for j in range(i + 1, np_):
            d = np.arccos(np.clip(np.dot(pf[i], pf[j]), -1, 1)) / np.pi
            as_.append(d)
            al_.append(1 if pl[i] == pl[j] else -1)
    aggr_eer = compute_eer(np.column_stack([as_, al_]))[0] if as_ else 1.

    # Rank-1
    r1 = 100. * sum(pl[i] == gl[np.argmin(dmat[i])] for i in range(np_)) / np_

    with open(os.path.join(out_dir, f"scores_{tag}.txt"), 'w') as f:
        for s, l in zip(scores, labels):
            f.write(f"{s} {l}\n")

    print(f"  [{tag}] pairEER={pair_eer*100:.4f}% aggrEER={aggr_eer*100:.4f}% Rank-1={r1:.2f}%")
    return pair_eer, aggr_eer, r1


def plot_curves(tl, vl, ta, va, d):
    try:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        a1.plot(tl, 'b', label='train')
        a1.plot(vl, 'r', label='val')
        a1.legend()
        a1.set_xlabel('epoch')
        a1.set_ylabel('loss')
        a2.plot(ta, 'b', label='train')
        a2.plot(va, 'r', label='val')
        a2.legend()
        a2.set_xlabel('epoch')
        a2.set_ylabel('acc')
        fig.savefig(os.path.join(d, 'curves.png'))
        plt.close(fig)
    except:
        pass


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    cfg = CONFIG
    protocol = cfg["protocol"]

    os.makedirs(cfg["results_dir"], exist_ok=True)
    eval_dir = os.path.join(cfg["results_dir"], "eval")
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  MSPHNet (Paper-Faithful) on CASIA-MS")
    print(f"  Protocol: {protocol} | Device: {device}")
    print(f"  Patch: {cfg['patch_size']}×{cfg['patch_size']} | Heads: {cfg['num_heads']}")
    print(f"  LR: {cfg['lr']} | Batch: {cfg['batch_size']}")
    print(f"{'='*60}\n")

    # Data
    print("Loading dataset...")
    id2paths = parse_casia_ms(cfg["data_root"])
    print(f"  {len(id2paths)} identities, {sum(len(v) for v in id2paths.values())} images\n")

    if protocol == "closed-set":
        train_s, test_s, _ = split_closed_set(id2paths, cfg["train_ratio"], cfg["random_seed"])
        nc = len(set(s[1] for s in train_s))
        train_ds = DS(train_s, cfg["img_side"], True)
        val_ds = DS(test_s, cfg["img_side"], False)
        gal_ds = DS(train_s, cfg["img_side"], False)
        prb_ds = DS(test_s, cfg["img_side"], False)
    else:
        train_s, val_s, gal_s, prb_s, tlm, _ = split_open_set(
            id2paths, cfg["train_ratio"], cfg["gallery_ratio"], cfg["val_ratio"], cfg["random_seed"])
        nc = len(tlm)
        train_ds = DS(train_s, cfg["img_side"], True)
        val_ds = DS(val_s, cfg["img_side"], False)
        gal_ds = DS(gal_s, cfg["img_side"], False)
        prb_ds = DS(prb_s, cfg["img_side"], False)

    print(f"  [{protocol}] #classes={nc}\n")

    bs, nw = cfg["batch_size"], cfg["num_workers"]
    train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, bs, num_workers=nw, pin_memory=True)
    gal_loader = DataLoader(gal_ds, bs, num_workers=nw, pin_memory=True)
    prb_loader = DataLoader(prb_ds, bs, num_workers=nw, pin_memory=True)

    # Model
    print("Building MSPHNet...")
    net = MSPHNet(
        nc, cfg["embedding_dim"], cfg["gabor_filters"], cfg["gabor_kernel"],
        cfg["patch_size"], cfg["num_heads"], cfg["ca_reduction"],
        cfg["dropout"], cfg["arcface_s"], cfg["arcface_m"]
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in net.parameters()):,}\n")

    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=cfg["lr"])
    sched = lr_scheduler.StepLR(opt, cfg["lr_step"], cfg["lr_gamma"])

    # Training
    tl, vl, ta, va = [], [], [], []
    best_eer, best_acc = 1., 0.
    last_eer, last_r1 = float('nan'), float('nan')

    print(f"Training for {cfg['num_epochs']} epochs...\n")

    for ep in range(cfg["num_epochs"]):
        t_loss, t_acc = run_epoch(net, train_loader, crit, opt, device, True)
        v_loss, v_acc = run_epoch(net, val_loader, crit, opt, device, False)
        sched.step()

        tl.append(t_loss)
        vl.append(v_loss)
        ta.append(t_acc)
        va.append(v_acc)

        _net = net.module if isinstance(net, DataParallel) else net

        if ep % cfg["eval_every"] == 0 or ep == cfg["num_epochs"] - 1:
            eer, _, r1 = evaluate(_net, prb_loader, gal_loader, device, eval_dir, f"ep{ep:04d}")
            last_eer, last_r1 = eer, r1
            if eer < best_eer:
                best_eer = eer
                torch.save(_net.state_dict(), os.path.join(cfg["results_dir"], "best_eer.pth"))
                print(f"  *** Best EER: {best_eer*100:.4f}% ***")

        if ep % 10 == 0 or ep == cfg["num_epochs"] - 1:
            es = f"{last_eer*100:.4f}%" if not math.isnan(last_eer) else "N/A"
            rs = f"{last_r1:.2f}%" if not math.isnan(last_r1) else "N/A"
            print(f"[{time.strftime('%H:%M:%S')}] ep {ep:04d} | "
                  f"loss t={t_loss:.5f} v={v_loss:.5f} | "
                  f"acc t={t_acc:.2f}% v={v_acc:.2f}% | EER={es} R1={rs}")

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(_net.state_dict(), os.path.join(cfg["results_dir"], "best.pth"))

        if ep % cfg["save_every"] == 0 or ep == cfg["num_epochs"] - 1:
            torch.save(_net.state_dict(), os.path.join(cfg["results_dir"], "latest.pth"))
            plot_curves(tl, vl, ta, va, cfg["results_dir"])

    # Final
    print("\n=== Final Evaluation ===")
    bp = os.path.join(cfg["results_dir"], "best_eer.pth")
    if not os.path.exists(bp):
        bp = os.path.join(cfg["results_dir"], "best.pth")
    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(bp, map_location=device))

    f_eer, f_aggr, f_r1 = evaluate(eval_net, prb_loader, gal_loader, device, eval_dir, "FINAL")

    print(f"\n{'='*60}")
    print(f"  FINAL: EER={f_eer*100:.4f}% Aggr={f_aggr*100:.4f}% R1={f_r1:.2f}%")
    print(f"{'='*60}\n")

    with open(os.path.join(cfg["results_dir"], "summary.txt"), 'w') as f:
        f.write(f"Protocol: {protocol}\n")
        f.write(f"Best acc: {best_acc:.3f}%\n")
        f.write(f"Final EER: {f_eer*100:.6f}%\n")
        f.write(f"Final Aggr: {f_aggr*100:.6f}%\n")
        f.write(f"Final R1: {f_r1:.3f}%\n")


if __name__ == "__main__":
    main()
