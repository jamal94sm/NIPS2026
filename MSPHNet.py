"""
MSPHNet on CASIA-MS Dataset
==================================================
Single-file implementation with closed-set and open-set protocols.
Implements the Multi-Scale Parallel Hybrid Network (MSPHNet) from:
"Multi-Scale Parallel Hybrid Network for Palmprint Recognition" (ICASSP 2025)

Architecture:
  - U-Net inspired dual-tier Encoder-Decoder
  - PHEB (Parallel Hybrid Feature Extraction Block):
    * Learnable Gabor filter for multi-order texture
    * CNN-based branch with CAB (Comprehensive Attention Block)
    * Transformer-based branch for global features
  - CAB: Spatial Attention (XA + YA + CA) + Pixel Attention (PA)
  - Skip connections between encoder and decoder

PROTOCOL options (edit CONFIG below):
  'closed-set' : 80% of samples per identity → train | 20% → test
  'open-set'   : 80% of identities → train | 20% of identities → test

Dataset: CASIA-MS-ROI
  Filename format : {subjectID}_{handSide}_{spectrum}_{iteration}.jpg
  Identity key    : subjectID + handSide  (e.g. "001_L")

Design decisions for unspecified parameters:
  - Channel progression: 1→64→128→256→128→64 (U-Net style)
  - Gabor: kernel=17, 36 filters, init_ratio=0.5 (from CO3Net/CompNet)
  - Transformer: depth=2, dim=128, 8 heads, 8×8 patches (paper + SF2Net)
  - CAB reduction ratio: 16 (from SE-Net reference)
  - Embedding: 1024-d (common in palmprint recognition)
  - Down/Up sampling: kernel=4, stride=2 (standard U-Net)
"""

# ==============================================================
#  CONFIG  — edit this block only
# ==============================================================
CONFIG = {
    "protocol"        : "open-set",   # "closed-set" | "open-set"
    "data_root"       : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "results_dir"     : "./rst_msphnet_casia_ms",
    "img_side"        : 128,            # input image size (128×128)
    "batch_size"      : 32,             # MSPHNet paper default
    "num_epochs"      : 1000,           # reasonable for palmprint
    "lr"              : 0.0001,         # MSPHNet paper default
    "lr_step"         : 300,            # learning rate decay step
    "lr_gamma"        : 0.5,            # learning rate decay factor
    "dropout"         : 0.5,            # dropout rate
    "arcface_s"       : 30.0,           # ArcFace scale
    "arcface_m"       : 0.50,           # ArcFace margin
    "embedding_dim"   : 1024,           # final embedding dimension
    "gabor_filters"   : 36,             # number of Gabor filters (from CO3Net)
    "gabor_kernel"    : 17,             # Gabor kernel size
    "transformer_depth": 2,             # Transformer block depth
    "transformer_dim" : 128,            # Transformer hidden dimension
    "transformer_heads": 8,             # number of attention heads (paper)
    "patch_size"      : 8,              # Transformer patch size (paper)
    "ca_reduction"    : 16,             # Channel attention reduction (SE-Net)
    "train_ratio"     : 0.80,
    "gallery_ratio"   : 0.50,
    "val_ratio"       : 0.10,
    "random_seed"     : 42,
    "save_every"      : 10,
    "eval_every"      : 50,
    "num_workers"     : 4,
}
# ==============================================================

import os
import sys
import math
import time
import random
import pickle
import warnings
import numpy as np
from collections import defaultdict, Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# einops for Transformer
try:
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange
except ImportError:
    print("Installing einops...")
    os.system("pip install einops")
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────
SEED = CONFIG["random_seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════
#  ARCFACE (for classification head)
# ══════════════════════════════════════════════════════════════

class ArcMarginProduct(nn.Module):
    """ArcFace head — large margin arc distance."""
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if self.training:
            assert label is not None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            output = self.s * cosine
        return output


# ══════════════════════════════════════════════════════════════
#  LEARNABLE GABOR FILTER (from CO3Net/CompNet)
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    """Learnable Gabor Convolution (LGC) Layer for texture extraction."""
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=0.5):
        super(GaborConv2d, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_ratio = init_ratio

        self.SIGMA = 9.2 * init_ratio
        self.GAMMA = 2.0
        self.FREQUENCY = 0.057 / init_ratio

        self.sigma = nn.Parameter(torch.FloatTensor([self.SIGMA]), requires_grad=True)
        self.gamma = nn.Parameter(torch.FloatTensor([self.GAMMA]), requires_grad=True)
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
            requires_grad=False)
        self.frequency = nn.Parameter(torch.FloatTensor([self.FREQUENCY]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def forward(self, x):
        kernel = self._get_gabor_kernel()
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)
        return out

    def _get_gabor_kernel(self):
        x_max = self.kernel_size // 2
        y_max = self.kernel_size // 2
        x_min, y_min = -x_max, -y_max
        k_size = x_max - x_min + 1

        x_0 = torch.arange(x_min, x_max + 1).float()
        y_0 = torch.arange(y_min, y_max + 1).float()

        x = x_0.view(-1, 1).repeat(self.channel_out, self.channel_in, 1, k_size)
        y = y_0.view(1, -1).repeat(self.channel_out, self.channel_in, k_size, 1)

        x = x.float().to(self.sigma.device)
        y = y.float().to(self.sigma.device)

        x_theta = x * torch.cos(self.theta.view(-1, 1, 1, 1)) + y * torch.sin(self.theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(self.theta.view(-1, 1, 1, 1)) + y * torch.cos(self.theta.view(-1, 1, 1, 1))

        gabor = -torch.exp(
            -0.5 * ((self.gamma * x_theta) ** 2 + y_theta ** 2) / (8 * self.sigma.view(-1, 1, 1, 1) ** 2)) \
            * torch.cos(2 * math.pi * self.frequency.view(-1, 1, 1, 1) * x_theta + self.psi.view(-1, 1, 1, 1))

        gabor = gabor - gabor.mean(dim=[2, 3], keepdim=True)
        return gabor


# ══════════════════════════════════════════════════════════════
#  COMPREHENSIVE ATTENTION BLOCK (CAB) - Equations 1-4 from paper
# ══════════════════════════════════════════════════════════════

class XDirectionAttention(nn.Module):
    """X-direction Attention (Eq. 1): Compresses feature map vertically."""
    def __init__(self, channels, reduction=16):
        super(XDirectionAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        # Global average pooling along height (vertical compression)
        # Result: [B, C, 1, W]
        x_avg = torch.mean(x, dim=2, keepdim=True)
        
        # Conv → ReLU → Conv → Sigmoid
        attn = self.conv1(x_avg)
        attn = F.relu(attn)
        attn = self.conv2(attn)
        attn = torch.sigmoid(attn)
        
        return x * attn


class YDirectionAttention(nn.Module):
    """Y-direction Attention (Eq. 2): Compresses feature map horizontally."""
    def __init__(self, channels, reduction=16):
        super(YDirectionAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        # Global average pooling along width (horizontal compression)
        # Result: [B, C, H, 1]
        y_avg = torch.mean(x, dim=3, keepdim=True)
        
        # Conv → ReLU → Conv → Sigmoid
        attn = self.conv1(y_avg)
        attn = F.relu(attn)
        attn = self.conv2(attn)
        attn = torch.sigmoid(attn)
        
        return x * attn


class ChannelAttention(nn.Module):
    """Channel Attention (Eq. 3): SE-style channel attention."""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        # Global average pooling: [B, C, H, W] → [B, C, 1, 1]
        ca = self.avg_pool(x)
        
        # Conv → ReLU → Conv → Sigmoid
        ca = self.conv1(ca)
        ca = F.relu(ca)
        ca = self.conv2(ca)
        ca = torch.sigmoid(ca)
        
        return x * ca


class PixelAttention(nn.Module):
    """Pixel Attention (Eq. 4): Assigns different weights to different pixels."""
    def __init__(self, channels):
        super(PixelAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # Conv → ReLU → Conv → Sigmoid
        pa = self.conv1(x)
        pa = F.relu(pa)
        pa = self.conv2(pa)
        pa = torch.sigmoid(pa)
        
        return x * pa


class CAB(nn.Module):
    """
    Comprehensive Attention Block (CAB).
    Integrates Spatial Attention (XA + YA + CA) and Pixel Attention (PA).
    """
    def __init__(self, channels, reduction=16):
        super(CAB, self).__init__()
        
        # Spatial Attention components
        self.xa = XDirectionAttention(channels, reduction)
        self.ya = YDirectionAttention(channels, reduction)
        self.ca = ChannelAttention(channels, reduction)
        
        # Pixel Attention
        self.pa = PixelAttention(channels)
        
        # Learnable weights for XA and YA fusion
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # X-direction and Y-direction attention
        xa_out = self.xa(x)
        ya_out = self.ya(x)
        
        # Weighted addition of XA and YA (as per paper description)
        sa_out = self.alpha * xa_out + self.beta * ya_out
        
        # Channel Attention
        ca_out = self.ca(sa_out)
        
        # Pixel Attention
        pa_out = self.pa(ca_out)
        
        return pa_out


# ══════════════════════════════════════════════════════════════
#  TRANSFORMER BLOCK (for global feature extraction)
# ══════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """MLP for Transformer."""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention (Eq. 5 from paper)."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Standard Transformer Block with multi-head attention and feed-forward."""
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=256, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class TransformerBranch(nn.Module):
    """
    Transformer-based branch for global feature extraction.
    Patches input, adds positional encoding, applies transformer blocks.
    """
    def __init__(self, in_channels, patch_size=8, dim=128, depth=2, heads=8, mlp_dim=256, dropout=0.1):
        super(TransformerBranch, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        
        # Patch embedding: flatten patches and project to dim
        patch_dim = in_channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        # Positional embedding (will be initialized based on actual patch count)
        self.pos_embedding = None
        
        # Transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, dim // heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Calculate number of patches
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Patch embedding
        x = self.to_patch_embedding(x)  # [B, num_patches, dim]
        
        # Initialize positional embedding if needed
        if self.pos_embedding is None or self.pos_embedding.size(1) != num_patches:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, num_patches, self.dim, device=x.device)
            )
        
        # Add positional embedding
        x = x + self.pos_embedding[:, :num_patches, :]
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer:
            x = block(x)
        
        x = self.norm(x)
        
        # Reshape back to spatial format for concatenation with CNN branch
        # [B, num_patches, dim] → [B, dim, num_patches_h, num_patches_w]
        x = rearrange(x, 'b (h w) d -> b d h w', h=num_patches_h, w=num_patches_w)
        
        return x


# ══════════════════════════════════════════════════════════════
#  CNN-BASED BRANCH (with CAB)
# ══════════════════════════════════════════════════════════════

class CNNBranch(nn.Module):
    """CNN-based branch with Comprehensive Attention Block (CAB)."""
    def __init__(self, in_channels, out_channels, reduction=16):
        super(CNNBranch, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cab = CAB(out_channels, reduction)

    def forward(self, x):
        x = self.conv(x)
        x = self.cab(x)
        return x


# ══════════════════════════════════════════════════════════════
#  PARALLEL HYBRID FEATURE EXTRACTION BLOCK (PHEB)
# ══════════════════════════════════════════════════════════════

class PHEB(nn.Module):
    """
    Parallel Hybrid Feature Extraction Block.
    - Learnable Gabor filter for texture extraction
    - CNN-based branch with CAB for local features
    - Transformer-based branch for global features
    - Concatenation + Conv + BN for fusion
    """
    def __init__(self, in_channels, out_channels, gabor_filters=36, gabor_kernel=17,
                 patch_size=8, trans_dim=128, trans_depth=2, trans_heads=8,
                 ca_reduction=16, is_first=False):
        super(PHEB, self).__init__()
        
        self.is_first = is_first
        
        # Learnable Gabor filter for texture extraction
        if is_first:
            gabor_in = in_channels
        else:
            gabor_in = in_channels
        
        self.gabor = GaborConv2d(
            channel_in=gabor_in,
            channel_out=gabor_filters,
            kernel_size=gabor_kernel,
            stride=1,
            padding=gabor_kernel // 2,
            init_ratio=0.5
        )
        self.gabor_bn = nn.BatchNorm2d(gabor_filters)
        
        # CNN-based branch with CAB
        self.cnn_branch = CNNBranch(gabor_filters, out_channels // 2, ca_reduction)
        
        # Transformer-based branch
        self.trans_branch = TransformerBranch(
            in_channels=gabor_filters,
            patch_size=patch_size,
            dim=trans_dim,
            depth=trans_depth,
            heads=trans_heads,
            mlp_dim=trans_dim * 2,
            dropout=0.1
        )
        
        # Project transformer output to match CNN branch channels
        self.trans_proj = nn.Sequential(
            nn.Conv2d(trans_dim, out_channels // 2, kernel_size=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Upsample transformer features to match CNN spatial size
        self.trans_upsample = None  # Will be created dynamically if needed
        
        # Fusion: concatenate + conv + BN
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Gabor filter for texture extraction
        gabor_out = self.gabor(x)
        gabor_out = self.gabor_bn(gabor_out)
        gabor_out = F.relu(gabor_out)
        
        # CNN branch (local features)
        cnn_out = self.cnn_branch(gabor_out)
        
        # Transformer branch (global features)
        trans_out = self.trans_branch(gabor_out)
        trans_out = self.trans_proj(trans_out)
        
        # Upsample transformer output to match CNN spatial size if needed
        if trans_out.shape[2:] != cnn_out.shape[2:]:
            trans_out = F.interpolate(trans_out, size=cnn_out.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate CNN and Transformer outputs
        concat = torch.cat([cnn_out, trans_out], dim=1)
        
        # Fusion
        out = self.fusion(concat)
        
        return out


# ══════════════════════════════════════════════════════════════
#  DOWN-SAMPLING AND UP-SAMPLING BLOCKS
# ══════════════════════════════════════════════════════════════

class DownSample(nn.Module):
    """Down-sampling block: Conv with stride 2."""
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    """Up-sampling block: Transposed Conv with stride 2."""
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


# ══════════════════════════════════════════════════════════════
#  MSPHNet MODEL
# ══════════════════════════════════════════════════════════════

class MSPHNet(nn.Module):
    """
    Multi-Scale Parallel Hybrid Network for Palmprint Recognition.
    
    Architecture:
    - Initial Conv to expand channels
    - Encoder: PHEB → DownSample → PHEB → DownSample → PHEB (bottleneck)
    - Decoder: UpSample → PHEB (+ skip) → UpSample → PHEB (+ skip)
    - Final: MaxPool → Conv → FC → Embedding
    """
    def __init__(self, num_classes, img_side=128, embedding_dim=1024,
                 gabor_filters=36, gabor_kernel=17,
                 trans_dim=128, trans_depth=2, trans_heads=8, patch_size=8,
                 ca_reduction=16, dropout=0.5,
                 arcface_s=30.0, arcface_m=0.50):
        super(MSPHNet, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Channel progression (U-Net style)
        c1, c2, c3 = 64, 128, 256
        
        # Initial conv to expand input channels
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        
        # Encoder path
        # Level 1: 128x128 → PHEB
        self.pheb1 = PHEB(c1, c1, gabor_filters, gabor_kernel, patch_size,
                          trans_dim, trans_depth, trans_heads, ca_reduction, is_first=True)
        
        # Level 2: 64x64 → DownSample + PHEB
        self.down1 = DownSample(c1, c2)
        self.pheb2 = PHEB(c2, c2, gabor_filters, gabor_kernel, patch_size,
                          trans_dim, trans_depth, trans_heads, ca_reduction)
        
        # Level 3 (Bottleneck): 32x32 → DownSample + PHEB
        self.down2 = DownSample(c2, c3)
        self.pheb3 = PHEB(c3, c3, gabor_filters, gabor_kernel, patch_size,
                          trans_dim, trans_depth, trans_heads, ca_reduction)
        
        # Decoder path
        # Level 2: 64x64 → UpSample + PHEB (with skip connection)
        self.up1 = UpSample(c3, c2)
        self.pheb4 = PHEB(c2 * 2, c2, gabor_filters, gabor_kernel, patch_size,
                          trans_dim, trans_depth, trans_heads, ca_reduction)  # *2 for skip concat
        
        # Level 1: 128x128 → UpSample + PHEB (with skip connection)
        self.up2 = UpSample(c2, c1)
        self.pheb5 = PHEB(c1 * 2, c1, gabor_filters, gabor_kernel, patch_size,
                          trans_dim, trans_depth, trans_heads, ca_reduction)  # *2 for skip concat
        
        # Final feature extraction
        self.final_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.final_conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )
        
        # Fully connected layers
        fc_input_dim = c3 * 4 * 4  # 256 * 4 * 4 = 4096
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2048, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # ArcFace classification head
        self.arcface = ArcMarginProduct(embedding_dim, num_classes, s=arcface_s, m=arcface_m)

    def forward(self, x, y=None):
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder
        e1 = self.pheb1(x)          # 128x128, c1
        
        d1 = self.down1(e1)         # 64x64, c2
        e2 = self.pheb2(d1)         # 64x64, c2
        
        d2 = self.down2(e2)         # 32x32, c3
        e3 = self.pheb3(d2)         # 32x32, c3 (bottleneck)
        
        # Decoder with skip connections
        u1 = self.up1(e3)           # 64x64, c2
        u1 = torch.cat([u1, e2], dim=1)  # Skip connection
        d1_out = self.pheb4(u1)     # 64x64, c2
        
        u2 = self.up2(d1_out)       # 128x128, c1
        u2 = torch.cat([u2, e1], dim=1)  # Skip connection
        d2_out = self.pheb5(u2)     # 128x128, c1
        
        # Final feature extraction
        feat = self.final_pool(d2_out)
        feat = self.final_conv(feat)
        feat = feat.view(feat.size(0), -1)
        
        # FC layers
        embedding = self.fc(feat)
        
        # ArcFace
        output = self.arcface(embedding, y)
        
        return output, F.normalize(embedding, dim=-1)

    def get_feature_vector(self, x):
        """Extract L2-normalized embedding for matching."""
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder
        e1 = self.pheb1(x)
        d1 = self.down1(e1)
        e2 = self.pheb2(d1)
        d2 = self.down2(e2)
        e3 = self.pheb3(d2)
        
        # Decoder with skip connections
        u1 = self.up1(e3)
        u1 = torch.cat([u1, e2], dim=1)
        d1_out = self.pheb4(u1)
        
        u2 = self.up2(d1_out)
        u2 = torch.cat([u2, e1], dim=1)
        d2_out = self.pheb5(u2)
        
        # Final feature extraction
        feat = self.final_pool(d2_out)
        feat = self.final_conv(feat)
        feat = feat.view(feat.size(0), -1)
        
        # FC layers
        embedding = self.fc(feat)
        
        # L2 normalize
        return F.normalize(embedding, dim=-1)


# ══════════════════════════════════════════════════════════════
#  NORMALISATION (NormSingleROI - same as CO3Net/SF2Net)
# ══════════════════════════════════════════════════════════════

class NormSingleROI(object):
    """Normalise non-black pixels to zero mean, unit std."""
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        c, h, w = tensor.size()
        tensor = tensor.view(c, h * w)
        idx = tensor > 0
        t = tensor[idx]
        if len(t) > 0:
            m = t.mean()
            s = t.std()
            t = t.sub_(m).div_(s + 1e-6)
            tensor[idx] = t
        tensor = tensor.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats=self.outchannels, dim=0)
        return tensor


# ══════════════════════════════════════════════════════════════
#  DATASET — for CASIA-MS-ROI
# ══════════════════════════════════════════════════════════════

def parse_casia_ms(data_root):
    """
    Scan CASIA-MS-ROI folder. Filename format:
        {subjectID}_{handSide}_{spectrum}_{iteration}.jpg
    Identity key = subjectID + "_" + handSide (e.g. "001_L").
    Returns dict {identity_key: [path1, path2, …]}
    """
    id2paths = defaultdict(list)
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".bmp", ".png")):
            continue
        parts = fname.split("_")
        if len(parts) < 4:
            continue
        identity = parts[0] + "_" + parts[1]
        id2paths[identity].append(os.path.join(data_root, fname))
    return dict(id2paths)


def split_closed_set(id2paths, train_ratio=0.8, seed=42):
    """Closed-set split: 80% samples per identity for train, 20% for test."""
    rng = random.Random(seed)
    label_map = {}
    train_samples, test_samples = [], []

    for idx, (identity, paths) in enumerate(sorted(id2paths.items())):
        label_map[identity] = idx
        paths_shuffled = list(paths)
        rng.shuffle(paths_shuffled)
        n_train = max(1, int(len(paths_shuffled) * train_ratio))
        for p in paths_shuffled[:n_train]:
            train_samples.append((p, idx))
        for p in paths_shuffled[n_train:]:
            test_samples.append((p, idx))

    return train_samples, test_samples, label_map


def split_open_set(id2paths, train_ratio=0.8, gallery_ratio=0.5, val_ratio=0.10, seed=42):
    """Open-set split: 80% identities for train, 20% for test."""
    rng = random.Random(seed)
    identities = sorted(id2paths.keys())
    rng.shuffle(identities)
    n_train_ids = max(1, int(len(identities) * train_ratio))

    train_ids = identities[:n_train_ids]
    test_ids = identities[n_train_ids:]

    train_label_map = {k: i for i, k in enumerate(sorted(train_ids))}
    all_train_samples = []
    for identity in train_ids:
        lab = train_label_map[identity]
        for p in id2paths[identity]:
            all_train_samples.append((p, lab))

    rng2 = random.Random(seed + 1)
    rng2.shuffle(all_train_samples)
    n_val = max(1, int(len(all_train_samples) * val_ratio))
    val_samples = all_train_samples[:n_val]
    train_samples = all_train_samples[n_val:]

    test_label_map = {k: i for i, k in enumerate(sorted(test_ids))}
    gallery_samples, probe_samples = [], []
    for identity in test_ids:
        lab = test_label_map[identity]
        paths_shuffled = list(id2paths[identity])
        rng.shuffle(paths_shuffled)
        n_gal = max(1, int(len(paths_shuffled) * gallery_ratio))
        for p in paths_shuffled[:n_gal]:
            gallery_samples.append((p, lab))
        for p in paths_shuffled[n_gal:]:
            probe_samples.append((p, lab))

    return (train_samples, val_samples, gallery_samples, probe_samples,
            train_label_map, test_label_map)


class CASIAMSDataset(Dataset):
    """Dataset for training/validation with data augmentation."""
    def __init__(self, samples, img_side=128, train=True):
        super().__init__()
        self.samples = samples
        self.img_side = img_side
        self.train = train

        if train:
            self.transform = T.Compose([
                T.Resize(img_side),
                T.RandomChoice(transforms=[
                    T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                    T.RandomResizedCrop(size=img_side, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                    T.RandomPerspective(distortion_scale=0.15, p=1),
                    T.RandomChoice(transforms=[
                        T.RandomRotation(degrees=10, interpolation=Image.BICUBIC,
                                         expand=False, center=(0.5 * img_side, 0.0)),
                        T.RandomRotation(degrees=10, interpolation=Image.BICUBIC,
                                         expand=False, center=(0.0, 0.5 * img_side)),
                    ]),
                ]),
                T.ToTensor(),
                NormSingleROI(outchannels=1),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(img_side),
                T.ToTensor(),
                NormSingleROI(outchannels=1),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, label


class CASIAMSDatasetSingle(Dataset):
    """Single-image dataset for evaluation."""
    def __init__(self, samples, img_side=128):
        super().__init__()
        self.samples = samples
        self.transform = T.Compose([
            T.Resize(img_side),
            T.ToTensor(),
            NormSingleROI(outchannels=1),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, label


# ══════════════════════════════════════════════════════════════
#  TRAINING — one-epoch worker
# ══════════════════════════════════════════════════════════════

def run_one_epoch(epoch, model, loader, criterion, optimizer, device, phase):
    """Run one epoch of training or validation."""
    is_train = (phase == "training")
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad()
            output, embedding = model(imgs, targets)
        else:
            with torch.no_grad():
                output, embedding = model(imgs, None)

        loss = criterion(output, targets)

        running_loss += loss.item() * imgs.size(0)
        preds = output.data.max(dim=1)[1]
        running_correct += preds.eq(targets).cpu().sum().item()
        total += imgs.size(0)

        if is_train:
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = 100.0 * running_correct / max(total, 1)
    return epoch_loss, epoch_acc


# ══════════════════════════════════════════════════════════════
#  EVALUATION — EER + Rank-1
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device):
    """Extract L2-normalized embeddings."""
    model.eval()
    feats, labels = [], []
    for imgs, labs in loader:
        imgs = imgs.to(device)
        codes = model.get_feature_vector(imgs)
        feats.append(codes.cpu().numpy())
        labels.append(np.array(labs) if isinstance(labs, list) else labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def compute_eer(scores_array):
    """Compute EER from scores array."""
    inscore = scores_array[scores_array[:, 1] == 1, 0]
    outscore = scores_array[scores_array[:, 1] == -1, 0]
    if len(inscore) == 0 or len(outscore) == 0:
        return 1.0, 0.0

    mIn, mOut = inscore.mean(), outscore.mean()
    flipped = False
    if mIn < mOut:
        inscore, outscore = -inscore, -outscore
        flipped = True

    y = np.concatenate([np.ones(len(inscore)), np.zeros(len(outscore))])
    s = np.concatenate([inscore, outscore])
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))
    if flipped:
        thresh = -thresh
    return eer, thresh


def evaluate(model, probe_loader, gallery_loader, device, out_dir=".", tag="eval"):
    """Evaluate model: Pairwise EER, Aggregated EER, Rank-1."""
    probe_feats, probe_labels = extract_features(model, probe_loader, device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)

    n_probe, n_gallery = len(probe_feats), len(gallery_feats)

    # Pairwise matching
    scores_list, labels_list = [], []
    dist_matrix = np.zeros((n_probe, n_gallery))

    for i in range(n_probe):
        cos_sim = np.dot(gallery_feats, probe_feats[i])
        dists = np.arccos(np.clip(cos_sim, -1, 1)) / np.pi
        dist_matrix[i] = dists
        for j in range(n_gallery):
            scores_list.append(dists[j])
            labels_list.append(1 if probe_labels[i] == gallery_labels[j] else -1)

    scores_arr = np.column_stack([scores_list, labels_list])
    pair_eer, _ = compute_eer(scores_arr)

    # Aggregated EER
    aggr_s, aggr_l = [], []
    for i in range(n_probe - 1):
        for j in range(i + 1, n_probe):
            cos_sim = np.dot(probe_feats[i], probe_feats[j])
            d = np.arccos(np.clip(cos_sim, -1, 1)) / np.pi
            aggr_s.append(d)
            aggr_l.append(1 if probe_labels[i] == probe_labels[j] else -1)

    aggr_eer = 1.0
    if aggr_s:
        aggr_arr = np.column_stack([aggr_s, aggr_l])
        aggr_eer, _ = compute_eer(aggr_arr)

    # Rank-1
    correct = sum(1 for i in range(n_probe) if probe_labels[i] == gallery_labels[np.argmin(dist_matrix[i])])
    rank1 = 100.0 * correct / max(n_probe, 1)

    # Save scores
    score_path = os.path.join(out_dir, f"scores_{tag}.txt")
    with open(score_path, "w") as f:
        for s_val, l_val in zip(scores_list, labels_list):
            f.write(f"{s_val} {l_val}\n")

    _save_roc_det(scores_arr, out_dir, tag)

    print(f"  [{tag}]  pairEER={pair_eer*100:.4f}%  aggrEER={aggr_eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return pair_eer, aggr_eer, rank1


def _save_roc_det(scores_arr, out_dir, tag):
    """Save ROC/DET plots."""
    inscore = scores_arr[scores_arr[:, 1] == 1, 0]
    outscore = scores_arr[scores_arr[:, 1] == -1, 0]
    if len(inscore) == 0 or len(outscore) == 0:
        return

    mIn, mOut = inscore.mean(), outscore.mean()
    if mIn < mOut:
        inscore, outscore = -inscore, -outscore

    y = np.concatenate([np.ones(len(inscore)), np.zeros(len(outscore))])
    s = np.concatenate([inscore, outscore])
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr

    try:
        pdf = PdfPages(os.path.join(out_dir, f"roc_det_{tag}.pdf"))
        
        # ROC
        fig, ax = plt.subplots()
        ax.plot(fpr * 100, tpr * 100, 'b-^', markersize=2)
        ax.set_xlim([0, 5]); ax.set_ylim([90, 100])
        ax.grid(True); ax.set_title(f'ROC — {tag}')
        ax.set_xlabel('FAR (%)'); ax.set_ylabel('GAR (%)')
        pdf.savefig(fig); plt.close(fig)

        # DET
        fig, ax = plt.subplots()
        ax.plot(fpr * 100, fnr * 100, 'b-^', markersize=2)
        ax.set_xlim([0, 5]); ax.set_ylim([0, 5])
        ax.grid(True); ax.set_title(f'DET — {tag}')
        ax.set_xlabel('FAR (%)'); ax.set_ylabel('FRR (%)')
        pdf.savefig(fig); plt.close(fig)

        pdf.close()
    except Exception as e:
        print(f"  [warn] plot save failed: {e}")


def plot_loss_acc(train_losses, val_losses, train_accs, val_accs, results_dir):
    """Plot loss and accuracy curves."""
    try:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(train_losses) + 1), train_losses, 'b', label='train loss')
        ax.plot(range(1, len(val_losses) + 1), val_losses, 'r', label='val loss')
        ax.legend(); ax.set_xlabel('epoch'); ax.set_ylabel('loss')
        fig.savefig(os.path.join(results_dir, "losses.png"))
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(range(1, len(train_accs) + 1), train_accs, 'b', label='train acc')
        ax.plot(range(1, len(val_accs) + 1), val_accs, 'r', label='val acc')
        ax.legend(); ax.grid(True)
        ax.set_xlabel('epoch'); ax.set_ylabel('accuracy (%)')
        fig.savefig(os.path.join(results_dir, "accuracy.png"))
        plt.close(fig)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    protocol = CONFIG["protocol"]
    data_root = CONFIG["data_root"]
    results_dir = CONFIG["results_dir"]
    img_side = CONFIG["img_side"]
    batch_size = CONFIG["batch_size"]
    num_epochs = CONFIG["num_epochs"]
    lr = CONFIG["lr"]
    lr_step = CONFIG["lr_step"]
    lr_gamma = CONFIG["lr_gamma"]
    dropout = CONFIG["dropout"]
    arcface_s = CONFIG["arcface_s"]
    arcface_m = CONFIG["arcface_m"]
    embedding_dim = CONFIG["embedding_dim"]
    gabor_filters = CONFIG["gabor_filters"]
    gabor_kernel = CONFIG["gabor_kernel"]
    trans_depth = CONFIG["transformer_depth"]
    trans_dim = CONFIG["transformer_dim"]
    trans_heads = CONFIG["transformer_heads"]
    patch_size = CONFIG["patch_size"]
    ca_reduction = CONFIG["ca_reduction"]
    train_ratio = CONFIG["train_ratio"]
    gallery_ratio = CONFIG["gallery_ratio"]
    val_ratio = CONFIG["val_ratio"]
    seed = CONFIG["random_seed"]
    save_every = CONFIG["save_every"]
    eval_every = CONFIG["eval_every"]
    nw = CONFIG["num_workers"]

    assert protocol in ("closed-set", "open-set")

    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  MSPHNet on CASIA-MS")
    print(f"  Protocol : {protocol}")
    print(f"  Device   : {device}")
    print(f"  Data     : {data_root}")
    print(f"  Loss     : CrossEntropy + ArcFace")
    print(f"  Embedding: {embedding_dim}-d")
    print(f"{'='*60}\n")

    # Parse dataset
    print("Scanning dataset …")
    id2paths = parse_casia_ms(data_root)
    n_total_ids = len(id2paths)
    n_total_imgs = sum(len(v) for v in id2paths.values())
    print(f"  Found {n_total_ids} identities, {n_total_imgs} images total.\n")

    # Protocol-specific split
    if protocol == "closed-set":
        train_samples, test_samples, label_map = split_closed_set(id2paths, train_ratio, seed)
        num_classes = len(label_map)

        train_dataset = CASIAMSDataset(train_samples, img_side, train=True)
        val_dataset = CASIAMSDataset(test_samples, img_side, train=False)
        gallery_eval = CASIAMSDatasetSingle(train_samples, img_side)
        probe_eval = CASIAMSDatasetSingle(test_samples, img_side)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=nw, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        gallery_loader = DataLoader(gallery_eval, batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        probe_loader = DataLoader(probe_eval, batch_size, shuffle=False, num_workers=nw, pin_memory=True)

        print(f"  [closed-set] #classes={num_classes}\n")

    else:  # open-set
        (train_samples, val_samples, gallery_samples, probe_samples,
         train_label_map, test_label_map) = split_open_set(id2paths, train_ratio, gallery_ratio, val_ratio, seed)
        num_classes = len(train_label_map)

        train_dataset = CASIAMSDataset(train_samples, img_side, train=True)
        val_dataset = CASIAMSDataset(val_samples, img_side, train=False)
        gallery_eval = CASIAMSDatasetSingle(gallery_samples, img_side)
        probe_eval = CASIAMSDatasetSingle(probe_samples, img_side)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=nw, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        gallery_loader = DataLoader(gallery_eval, batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        probe_loader = DataLoader(probe_eval, batch_size, shuffle=False, num_workers=nw, pin_memory=True)

        print(f"  [open-set] #train_classes={num_classes}\n")

    # Build model
    print(f"Building MSPHNet — num_classes={num_classes} …")
    net = MSPHNet(
        num_classes=num_classes,
        img_side=img_side,
        embedding_dim=embedding_dim,
        gabor_filters=gabor_filters,
        gabor_kernel=gabor_kernel,
        trans_dim=trans_dim,
        trans_depth=trans_depth,
        trans_heads=trans_heads,
        patch_size=patch_size,
        ca_reduction=ca_reduction,
        dropout=dropout,
        arcface_s=arcface_s,
        arcface_m=arcface_m
    )
    net.to(device)

    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
        net = DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_eer = 1.0
    last_eer, last_rank1 = float("nan"), float("nan")

    print(f"\nStarting training for {num_epochs} epochs …")
    print(f"  EER / Rank-1 computed every {eval_every} epochs.\n")

    for epoch in range(num_epochs):
        t_loss, t_acc = run_one_epoch(epoch, net, train_loader, criterion, optimizer, device, "training")
        v_loss, v_acc = run_one_epoch(epoch, net, val_loader, criterion, optimizer, device, "testing")
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        _net = net.module if isinstance(net, DataParallel) else net

        # Periodic evaluation
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            tag = f"ep{epoch:04d}_{protocol.replace('-','')}"
            cur_eer, cur_aggr_eer, cur_rank1 = evaluate(_net, probe_loader, gallery_loader, device, rst_eval, tag)
            last_eer, last_rank1 = cur_eer, cur_rank1

            if cur_eer < best_eer:
                best_eer = cur_eer
                torch.save(_net.state_dict(), os.path.join(results_dir, "net_params_best_eer.pth"))
                print(f"  *** New best EER: {best_eer*100:.4f}% ***")

        # Console output every 10 epochs
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            ts = time.strftime("%H:%M:%S")
            eer_str = f"{last_eer*100:.4f}%" if not math.isnan(last_eer) else "N/A"
            rank1_str = f"{last_rank1:.2f}%" if not math.isnan(last_rank1) else "N/A"
            print(f"[{ts}] ep {epoch:04d} | loss train={t_loss:.5f} val={v_loss:.5f} | "
                  f"acc train={t_acc:.2f}% val={v_acc:.2f}% | EER={eer_str} Rank-1={rank1_str}")

        # Save best model
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(_net.state_dict(), os.path.join(results_dir, "net_params_best.pth"))

        # Periodic checkpoint
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            torch.save(_net.state_dict(), os.path.join(results_dir, "net_params.pth"))
            plot_loss_acc(train_losses, val_losses, train_accs, val_accs, results_dir)

    # Final evaluation
    print("\n=== Final evaluation with best EER model ===")
    best_model_path = os.path.join(results_dir, "net_params_best_eer.pth")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(results_dir, "net_params_best.pth")

    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_model_path, map_location=device))

    final_eer, final_aggr_eer, final_rank1 = evaluate(
        eval_net, probe_loader, gallery_loader, device, rst_eval, f"FINAL_{protocol.replace('-','')}")

    print(f"\n{'='*60}")
    print(f"  PROTOCOL : {protocol}")
    print(f"  FINAL Pairwise EER   : {final_eer*100:.4f}%")
    print(f"  FINAL Aggregated EER : {final_aggr_eer*100:.4f}%")
    print(f"  FINAL Rank-1         : {final_rank1:.3f}%")
    print(f"  Results saved to: {results_dir}")
    print(f"{'='*60}\n")

    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"Protocol  : {protocol}\n")
        f.write(f"Data root : {data_root}\n")
        f.write(f"Identities: {n_total_ids}\n")
        f.write(f"Images    : {n_total_imgs}\n")
        f.write(f"Classes (train): {num_classes}\n")
        f.write(f"Best val acc       : {best_val_acc:.3f}%\n")
        f.write(f"Final Pairwise EER : {final_eer*100:.6f}%\n")
        f.write(f"Final Aggreg. EER  : {final_aggr_eer*100:.6f}%\n")
        f.write(f"Final Rank-1       : {final_rank1:.3f}%\n")


if __name__ == "__main__":
    main()
