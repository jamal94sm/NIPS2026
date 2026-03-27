"""
SF2Net on CASIA-MS Dataset
==================================================
Single-file implementation with closed-set and open-set protocols.
Faithfully preserves the official SF2Net architecture, TripletLoss,
triplet-sample contrastive training, and NormSingleROI normalisation.

PROTOCOL options (edit CONFIG below):
  'closed-set' : 80% of samples per identity → train | 20% → test
                 Evaluation: test probe vs train gallery (Rank-1 + EER)

  'open-set'   : 80% of identities → train | 20% of identities → test
                 Within test identities: 50% samples → gallery, 50% → probe
                 Evaluation: Rank-1 identification + EER

Dataset: CASIA-MS-ROI
  Filename format : {subjectID}_{handSide}_{spectrum}_{iteration}.jpg
  Identity key    : subjectID + handSide  (e.g. "001_L")
  All spectra and iterations are treated as samples of the same identity.

Architecture: SF2Net (unchanged from official repo)
  - Multi-order Gabor texture extraction (1st and 2nd order)
  - Sequence Feature Extractor (SFE) with first-k/last-k selection
  - Dual ViT for sequence feature processing
  - Weighted fusion of CNN and ViT features
  - Embedding: L2-normalised 1024-d
  - ArcFace classification head

Loss: CrossEntropy + TripletLoss (SRT distance)
Training: Triplet sampling (anchor, positive, negative)
"""

# ==============================================================
#  CONFIG  — edit this block only
# ==============================================================
CONFIG = {
    "protocol"        : "open-set",   # "closed-set" | "open-set"
    "data_root"       : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "results_dir"     : "./rst_sf2net_casia_ms",
    "img_side"        : 128,            # input image size (128×128)
    "batch_size"      : 500,            # SF2Net default
    "num_epochs"      : 100,           # SF2Net default
    "lr"              : 0.001,          # SF2Net default
    "lr_step"         : 50,            # SF2Net default (redstep)
    "lr_gamma"        : 0.8,            # SF2Net default
    "dropout"         : 0.5,            # SF2Net default
    "arcface_s"       : 30.0,
    "arcface_m"       : 0.50,
    "ce_weight"       : 0.7,            # cross-entropy loss weight (loss_ce)
    "tl_weight"       : 0.3,            # triplet loss weight (loss_tl)
    "vit_floor_num"   : 10,             # SF2Net default: first-k and last-k in SFE
    "cnn_vit_weight"  : 0.7,            # weight for CNN features (ViT gets 1-weight)
    "embedding_dim"   : 1024,           # SF2Net final embedding
    "train_ratio"     : 0.60,
    "gallery_ratio"   : 0.10,
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

# einops for ViT
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
#  TRIPLET LOSS (exact copy from SF2Net/model/component/triplet_loss.py)
# ══════════════════════════════════════════════════════════════

class TripletLoss(nn.Module):
    """
    Triplet loss with SRT (Soft Relative Triplet) distance.
    https://www.sciencedirect.com/science/article/pii/S003132032100649X
    """
    def __init__(self, margin=2.0, alpha=0.95, distance="SRT"):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.distance = distance
        self.tripletMargin = torch.nn.TripletMarginLoss(margin=1.0, swap=True, reduction='mean')

    def dis(self, anchor, positive):
        pos_dist = torch.sum((torch.sub(anchor, positive).pow(2)), 1)
        return pos_dist

    def cosine(self, emb1, emb2):
        uv = torch.sum(emb1 * emb2, dim=1)
        uu = torch.sum(emb2 * emb2, dim=1)
        vv = torch.sum(emb1 * emb1, dim=1)
        dist = torch.div(uv, torch.sqrt(uu * vv))
        return dist

    def forward(self, anchor, positive, negative, size_average=True):
        if self.distance == "Triplet" or self.distance == "TTriplet":
            self.margin = 1.0
            if self.distance == "TTriplet":
                self.margin = 1
                losses = self.tripletMargin(anchor, positive, negative)
                return losses, losses, losses, losses

            positive = F.normalize(positive, p=2, dim=1)
            anchor = F.normalize(anchor, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)

            positive_loss = self.dis(anchor, positive)
            distance_negative = self.dis(anchor, negative)
            distance_p_n = self.dis(positive, negative)
            losses = F.relu(positive_loss + self.margin - distance_negative).mean()
            return losses, positive_loss.mean(), distance_negative.mean(), distance_p_n.mean()

        elif self.distance == "SRT":
            self.margin = 2.0
            positive = F.normalize(positive, p=2, dim=1)
            anchor = F.normalize(anchor, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)

            positive_loss = self.dis(anchor, positive)
            distance_negative = self.dis(anchor, negative)
            distance_p_n = self.dis(positive, negative)

            cond = distance_negative.mean() >= distance_p_n.mean()

            ls = torch.where(cond, (positive_loss + self.margin - distance_p_n.mean()),
                             (positive_loss + self.margin - distance_negative))
            losses = F.relu(ls).mean()
            return losses, positive_loss.mean(), distance_negative.mean(), distance_p_n.mean()
        else:
            # cosine distance
            positive_loss = torch.abs(1.0 - self.cosine(anchor, positive).mean())
            distance_negative = self.cosine(anchor, negative)
            negative_loss = F.relu(distance_negative - self.margin).mean()
            losses = self.alpha * positive_loss + (1.0 - self.alpha) * negative_loss
            return losses, positive_loss, negative_loss, torch.tensor(0.0)


# ══════════════════════════════════════════════════════════════
#  ARCFACE (exact copy from SF2Net/model/component/arcface.py)
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
#  GABOR CONV (exact copy from SF2Net/model/component/gabor.py)
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    """Learnable Gabor Convolution (LGC) Layer."""
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_ratio = init_ratio
        self.kernel = 0

        self.SIGMA = 9.2 * self.init_ratio
        self.GAMMA = 2.0
        self.FREQUENCY = 0.057 / self.init_ratio

        self.sigma = nn.Parameter(torch.FloatTensor([self.SIGMA]), requires_grad=True)
        self.gamma = nn.Parameter(torch.FloatTensor([self.GAMMA]), requires_grad=True)
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
            requires_grad=False)
        self.frequency = nn.Parameter(torch.FloatTensor([self.FREQUENCY]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def forward(self, x):
        self.kernel = self.get_gabor()
        out = F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)
        return out

    def get_gabor(self):
        x_max = self.kernel_size // 2
        y_max = self.kernel_size // 2
        x_min = -x_max
        y_min = -y_max

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
#  SE MODULE (exact copy from SF2Net/model/component/squeeze_and_excitation.py)
# ══════════════════════════════════════════════════════════════

class SEModule(nn.Module):
    """Squeeze-and-Excitation module."""
    def __init__(self, channel, reduction=1):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x)
        return x * se_weight


# ══════════════════════════════════════════════════════════════
#  SEQUENCE FEATURE EXTRACTOR (from SF2Net/model/feature_extraction.py)
# ══════════════════════════════════════════════════════════════

def get_sequence_feature(feature_tensor, vit_floor_num):
    """Extract first-k and last-k features from sorted channel responses."""
    feature_tensor_for_channel = torch.softmax(feature_tensor, dim=1)
    feature_tensor_for_channel_front = feature_tensor_for_channel[:, :vit_floor_num, :, :]
    feature_tensor_for_channel_back = feature_tensor_for_channel[:, -vit_floor_num:, :, :]
    feature_tensor = torch.cat((feature_tensor_for_channel_front, feature_tensor_for_channel_back), dim=1)
    return feature_tensor


class FeatureExtraction(nn.Module):
    """
    Local feature extraction with multi-order Gabor texture.
    First-order: gradual texture; Second-order: abrupt texture changes.
    """
    def __init__(self, channel_in, filter_num, kernel_size, stride, padding,
                 init_ratio, label_num, vit_floor_num, channel_out=36):
        super(FeatureExtraction, self).__init__()

        self.channel_in = channel_in
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_ratio = init_ratio
        self.label_num = label_num
        self.vit_floor_num = vit_floor_num
        self.channel_out = channel_out

        # Multi-order Gabor convolution
        self.gabor_conv2d_1 = GaborConv2d(
            channel_in=self.channel_in, channel_out=self.filter_num,
            kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, init_ratio=self.init_ratio)
        self.gabor_conv2d_2 = GaborConv2d(
            channel_in=self.filter_num, channel_out=self.filter_num,
            kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, init_ratio=self.init_ratio)

        # SE module
        self.squeeze_and_excitation = SEModule(channel=self.filter_num)

        # Conv layers for processing blocks
        self.conv_0 = nn.Conv2d(in_channels=self.filter_num, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv_1 = nn.Conv2d(in_channels=self.filter_num, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0)

        # Pooling
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, feature_tensor):
        # First-order texture (gradual)
        first_order_feature_tensor = self.gabor_conv2d_1(feature_tensor)
        # Second-order texture (abrupt changes)
        second_order_feature_tensor = self.gabor_conv2d_2(first_order_feature_tensor)

        # Process blocks
        first_order_feature_tensor = self.process_block(first_order_feature_tensor, conv=self.conv_0)
        second_order_feature_tensor = self.process_block(second_order_feature_tensor, conv=self.conv_1)

        # Secondary convolution
        f_order_feature_tensor = self.conv_2(first_order_feature_tensor)
        s_order_feature_tensor = self.conv_3(second_order_feature_tensor)

        # Flatten and concatenate: 32*14*14 + 32*6*6 = 7424
        feature_tensor = torch.cat((
            f_order_feature_tensor.view(f_order_feature_tensor.shape[0], -1),
            s_order_feature_tensor.view(s_order_feature_tensor.shape[0], -1)), dim=1)

        # Sequence features for ViT
        first_order_seq = get_sequence_feature(first_order_feature_tensor, self.vit_floor_num)
        second_order_seq = get_sequence_feature(second_order_feature_tensor, self.vit_floor_num)

        return feature_tensor, first_order_seq, second_order_seq

    def process_block(self, feature_tensor, conv):
        feature_tensor = self.squeeze_and_excitation(feature_tensor)
        feature_tensor = conv(feature_tensor)
        feature_tensor = torch.relu(feature_tensor)
        feature_tensor = self.max_pool(feature_tensor)
        return feature_tensor


# ══════════════════════════════════════════════════════════════
#  VIT (exact copy from SF2Net/model/vit.py)
# ══════════════════════════════════════════════════════════════

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    """MLP for Transformer."""
    def __init__(self, dim, dim_for_mlp, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_for_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_for_mlp, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head attention."""
    def __init__(self, dim, heads, dim_for_head, dropout=0.1):
        super(Attention, self).__init__()
        self.heads = heads
        self.dim_for_head = dim_for_head
        self.dim_for_head_inner = dim_for_head * heads
        self.scale = dim_for_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, self.dim_for_head_inner * 3, bias=False)

        project_out = not (heads == 1 and dim_for_head == dim)
        self.to_out = nn.Sequential(
            nn.Linear(self.dim_for_head_inner, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

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


class Transformer(nn.Module):
    """Transformer encoder."""
    def __init__(self, dim, depth, heads, dim_for_head, dim_for_mlp, dropout=0.1):
        super(Transformer, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_for_head=dim_for_head, dropout=dropout),
                FeedForward(dim=dim, dim_for_mlp=dim_for_mlp, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    """Vision Transformer for sequence feature processing."""
    def __init__(self, *, image_size, patch_size, channels, num_classes, depth, heads,
                 dim, dim_for_head, dim_for_mlp, pool='cls', dropout=0.1, emb_dropout=0.1):
        super(ViT, self).__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (image_height % patch_height == 0 and image_width % patch_width == 0), \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads,
            dim_for_head=dim_for_head, dim_for_mlp=dim_for_mlp, dropout=dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, feature_tensor):
        x = self.to_patch_embedding(feature_tensor)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.to_latent(x)
        return x


# ══════════════════════════════════════════════════════════════
#  SF2NET MODEL (exact copy from SF2Net/model/sf2net.py)
# ══════════════════════════════════════════════════════════════

class SF2Net(nn.Module):
    """
    SF2Net: Sequence Feature Fusion Network for Palmprint Verification
    https://doi.org/10.1109/TIFS.2025.3611692
    """
    def __init__(self, num_classes, vit_floor_num=10, weight=0.7,
                 dropout=0.5, arcface_s=30.0, arcface_m=0.50):
        super(SF2Net, self).__init__()

        self.num_classes = num_classes
        self.vit_floor_num = vit_floor_num
        self.weight = weight

        # Local feature extraction
        self.feature_extraction = FeatureExtraction(
            channel_in=1, filter_num=36,
            kernel_size=17, stride=2, padding=17 // 2,
            init_ratio=0.5, label_num=num_classes,
            vit_floor_num=vit_floor_num)

        # ViT for sequence features
        self.vit_0 = ViT(
            image_size=30, patch_size=5, channels=vit_floor_num * 2,
            num_classes=num_classes, depth=2, heads=16, dim=128,
            dim_for_head=64, dim_for_mlp=256, dropout=0.1, emb_dropout=0.1)
        self.vit_1 = ViT(
            image_size=14, patch_size=2, channels=vit_floor_num * 2,
            num_classes=num_classes, depth=2, heads=16, dim=128,
            dim_for_head=64, dim_for_mlp=256, dropout=0.1, emb_dropout=0.1)

        # FC layers for CNN features (7424 → 2048 → 1024)
        self.fully_connection_1 = nn.Linear(7424, 2048)
        self.fully_connection_2 = nn.Linear(2048, 1024)

        # FC layers for ViT features (11136 → 4096 → 1024)
        self.fully_connection_for_vit_1 = nn.Linear(11136, 4096)
        self.fully_connection_for_vit_2 = nn.Linear(4096, 1024)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # ArcFace
        self.arcface = ArcMarginProduct(
            in_features=1024, out_features=num_classes,
            s=arcface_s, m=arcface_m, easy_margin=False)

    def forward(self, x, y=None):
        # Processing
        x = self.processing(x)

        # Dropout
        x = self.dropout(x)

        # ArcFace
        output = self.arcface(x, y)

        return output, F.normalize(x, dim=-1)

    def get_feature_vector(self, x):
        """Extract 1024-d L2-normalised embedding for matching."""
        x = self.processing(x)
        return x / torch.norm(x, p=2, dim=1, keepdim=True)

    def processing(self, x):
        # Feature extraction
        feature_tensor, first_order_seq, second_order_seq = self.feature_extraction(x)

        # ViT processing
        first_order_vit = self.vit_0(first_order_seq)
        second_order_vit = self.vit_1(second_order_seq)

        # Concatenate ViT outputs
        feature_tensor_for_vit = torch.cat((first_order_vit, second_order_vit), dim=1)
        feature_tensor_for_vit = feature_tensor_for_vit.view(feature_tensor_for_vit.shape[0], -1)

        # FC layers
        feature_tensor = self.fully_connection_1(feature_tensor)
        feature_tensor = self.fully_connection_2(feature_tensor)

        feature_tensor_for_vit = self.fully_connection_for_vit_1(feature_tensor_for_vit)
        feature_tensor_for_vit = self.fully_connection_for_vit_2(feature_tensor_for_vit)

        # Weighted fusion
        feature_tensor = feature_tensor * self.weight + feature_tensor_for_vit * (1 - self.weight)

        return feature_tensor


# ══════════════════════════════════════════════════════════════
#  NORMALISATION (same as CO3Net — NormSingleROI)
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
        m = t.mean()
        s = t.std()
        t = t.sub_(m).div_(s + 1e-6)
        tensor[idx] = t
        tensor = tensor.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats=self.outchannels, dim=0)
        return tensor


# ══════════════════════════════════════════════════════════════
#  DATASET — triplet & single, for CASIA-MS-ROI
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


# ────────── closed-set split ──────────
def split_closed_set(id2paths, train_ratio=0.8, seed=42):
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


# ────────── open-set split ──────────
def split_open_set(id2paths, train_ratio=0.8, gallery_ratio=0.5,
                   val_ratio=0.10, seed=42):
    rng = random.Random(seed)
    identities = sorted(id2paths.keys())
    rng.shuffle(identities)
    n_train_ids = max(1, int(len(identities) * train_ratio))

    train_ids = identities[:n_train_ids]
    test_ids = identities[n_train_ids:]

    # Training identities → train + val
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

    # Test identities → gallery + probe
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


# ────────── triplet dataset (for training / val) ──────────
class CASIAMSDatasetTriplet(Dataset):
    """
    Returns (anchor, positive, negative), (label_a, label_p, label_n).
    Training: positive = different sample of same identity
              negative = sample of different identity
    Testing:  positive = negative = anchor (duplicate)
    Mirrors the official SF2Net MyDataset behaviour.
    """
    def __init__(self, samples, img_side=128, train=True):
        super().__init__()
        self.samples = samples
        self.train = train
        self.img_side = img_side

        # Build label → indices map
        self.label2idxs = defaultdict(list)
        self.labels = []
        for i, (_, lab) in enumerate(samples):
            self.label2idxs[lab].append(i)
            self.labels.append(lab)
        self.labels = np.array(self.labels)

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
        path_anchor, label_anchor = self.samples[index]

        if self.train:
            # Positive: different sample, same label
            pos_idxs = self.label2idxs[label_anchor]
            pos_idx = index
            while pos_idx == index and len(pos_idxs) > 1:
                pos_idx = random.choice(pos_idxs)
            path_positive = self.samples[pos_idx][0]
            label_positive = label_anchor

            # Negative: sample from different label
            neg_idxs = np.where(self.labels != label_anchor)[0]
            neg_idx = random.choice(neg_idxs)
            path_negative, label_negative = self.samples[neg_idx]
        else:
            path_positive = path_anchor
            label_positive = label_anchor
            path_negative = path_anchor
            label_negative = label_anchor

        # Load images
        img_anchor = Image.open(path_anchor).convert("L")
        img_positive = Image.open(path_positive).convert("L")
        img_negative = Image.open(path_negative).convert("L")

        img_anchor = self.transform(img_anchor)
        img_positive = self.transform(img_positive)
        img_negative = self.transform(img_negative)

        return ([img_anchor, img_positive, img_negative],
                [label_anchor, label_positive, label_negative])


# ────────── single-image dataset (for evaluation) ──────────
class CASIAMSDatasetSingle(Dataset):
    def __init__(self, samples, img_side=128):
        super().__init__()
        self.samples = samples
        self.img_side = img_side
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

def run_one_epoch(epoch, model, loader, criterion, tl_criterion,
                  optimizer, device, phase,
                  ce_weight=0.7, tl_weight=0.3):
    """
    Mirrors SF2Net train.py `fit()`.
    phase ∈ {'training', 'testing'}
    """
    is_train = (phase == "training")
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for datas, targets in loader:
        anchor_data = datas[0].to(device)
        positive_data = datas[1].to(device)
        negative_data = datas[2].to(device)

        anchor_target = targets[0].to(device)
        positive_target = targets[1].to(device)
        negative_target = targets[2].to(device)

        if is_train:
            optimizer.zero_grad()
            output, fe1 = model(anchor_data, anchor_target)
            output2, fe2 = model(positive_data, positive_target)
            output3, fe3 = model(negative_data, negative_target)
        else:
            with torch.no_grad():
                output, fe1 = model(anchor_data, None)
                output2, fe2 = model(positive_data, None)
                output3, fe3 = model(negative_data, None)

        # CrossEntropy on anchor
        ce_loss = criterion(output, anchor_target)

        # TripletLoss on outputs (SF2Net uses output logits for triplet loss)
        tl_loss, _, _, _ = tl_criterion(output, output2, output3)

        loss = ce_weight * ce_loss + tl_weight * tl_loss

        running_loss += loss.item() * anchor_data.size(0)
        preds = output.data.max(dim=1)[1]
        running_correct += preds.eq(anchor_target).cpu().sum().item()
        total += anchor_data.size(0)

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
    """Extract 1024-d L2-normalised embeddings."""
    model.eval()
    feats, labels = [], []
    for imgs, labs in loader:
        imgs = imgs.to(device)
        codes = model.get_feature_vector(imgs)
        feats.append(codes.cpu().numpy())
        labels.append(np.array(labs) if isinstance(labs, list) else labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def compute_eer(scores_array):
    """
    scores_array: (N, 2) — col 0 = matching score, col 1 = label (+1 / -1).
    Returns (eer, threshold).
    """
    inscore = scores_array[scores_array[:, 1] == 1, 0]
    outscore = scores_array[scores_array[:, 1] == -1, 0]
    if len(inscore) == 0 or len(outscore) == 0:
        return 1.0, 0.0

    mIn = inscore.mean()
    mOut = outscore.mean()
    flipped = False
    if mIn < mOut:
        inscore = -inscore
        outscore = -outscore
        flipped = True

    y = np.concatenate([np.ones(len(inscore)), np.zeros(len(outscore))])
    s = np.concatenate([inscore, outscore])
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))
    if flipped:
        thresh = -thresh
    return eer, thresh


def evaluate(model, probe_loader, gallery_loader, device,
             out_dir=".", tag="eval"):
    """
    1) Extract features for probe and gallery.
    2) Pairwise EER (probe vs gallery).
    3) Aggregated EER (all-vs-all within probe set).
    4) Rank-1 accuracy.
    Returns (pairwise_eer, aggregated_eer, rank1_acc).
    """
    probe_feats, probe_labels = extract_features(model, probe_loader, device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)

    n_probe = len(probe_feats)
    n_gallery = len(gallery_feats)

    # Pairwise matching: probe vs gallery (cosine → arc distance)
    scores_list = []
    labels_list = []
    dist_matrix = np.zeros((n_probe, n_gallery))

    for i in range(n_probe):
        cos_sim = np.dot(gallery_feats, probe_feats[i])
        dists = np.arccos(np.clip(cos_sim, -1, 1)) / np.pi
        dist_matrix[i] = dists
        for j in range(n_gallery):
            scores_list.append(dists[j])
            labels_list.append(1 if probe_labels[i] == gallery_labels[j] else -1)

    scores_arr = np.column_stack([scores_list, labels_list])
    pair_eer, pair_th = compute_eer(scores_arr)

    # Aggregated EER: all-vs-all within PROBE set
    aggr_s, aggr_l = [], []
    for i in range(n_probe - 1):
        for j in range(i + 1, n_probe):
            cos_sim = np.dot(probe_feats[i], probe_feats[j])
            d = np.arccos(np.clip(cos_sim, -1, 1)) / np.pi
            aggr_s.append(d)
            aggr_l.append(1 if probe_labels[i] == probe_labels[j] else -1)

    if aggr_s:
        aggr_arr = np.column_stack([aggr_s, aggr_l])
        aggr_eer, aggr_th = compute_eer(aggr_arr)
    else:
        aggr_eer = 1.0

    # Rank-1 identification
    correct = 0
    for i in range(n_probe):
        best_j = np.argmin(dist_matrix[i])
        if probe_labels[i] == gallery_labels[best_j]:
            correct += 1
    rank1 = 100.0 * correct / max(n_probe, 1)

    # Save score file + plots
    score_path = os.path.join(out_dir, f"scores_{tag}.txt")
    with open(score_path, "w") as f:
        for s_val, l_val in zip(scores_list, labels_list):
            f.write(f"{s_val} {l_val}\n")

    _save_roc_det(scores_arr, out_dir, tag, pair_eer, pair_th)

    print(f"  [{tag}]  pairEER={pair_eer*100:.4f}%  aggrEER={aggr_eer*100:.4f}%  "
          f"Rank-1={rank1:.2f}%")
    return pair_eer, aggr_eer, rank1


def _save_roc_det(scores_arr, out_dir, tag, eer, thresh):
    """Save ROC / DET / FAR-FRR plots."""
    inscore = scores_arr[scores_arr[:, 1] == 1, 0]
    outscore = scores_arr[scores_arr[:, 1] == -1, 0]
    if len(inscore) == 0 or len(outscore) == 0:
        return

    mIn = inscore.mean()
    mOut = outscore.mean()
    if mIn < mOut:
        inscore = -inscore
        outscore = -outscore

    y = np.concatenate([np.ones(len(inscore)), np.zeros(len(outscore))])
    s = np.concatenate([inscore, outscore])
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr

    pdf_path = os.path.join(out_dir, f"roc_det_{tag}.pdf")
    try:
        pdf = PdfPages(pdf_path)

        fpr_p = fpr * 100
        tpr_p = tpr * 100
        fnr_p = fnr * 100

        # ROC
        fig, ax = plt.subplots()
        ax.plot(fpr_p, tpr_p, 'b-^', label='ROC curve', markersize=2)
        ax.plot(np.linspace(0, 100, 101), np.linspace(100, 0, 101), 'k-', label='EER')
        ax.set_xlim([0, 5]); ax.set_ylim([90, 100])
        ax.legend(); ax.grid(True)
        ax.set_title(f'ROC — {tag}'); ax.set_xlabel('FAR (%)'); ax.set_ylabel('GAR (%)')
        pdf.savefig(fig); plt.close(fig)

        # DET
        fig, ax = plt.subplots()
        ax.plot(fpr_p, fnr_p, 'b-^', label='DET curve', markersize=2)
        ax.plot(np.linspace(0, 100, 101), np.linspace(0, 100, 101), 'k-', label='EER')
        ax.set_xlim([0, 5]); ax.set_ylim([0, 5])
        ax.legend(); ax.grid(True)
        ax.set_title(f'DET — {tag}'); ax.set_xlabel('FAR (%)'); ax.set_ylabel('FRR (%)')
        pdf.savefig(fig); plt.close(fig)

        # FAR/FRR vs threshold
        fig, ax = plt.subplots()
        ax.plot(thresholds, fpr_p, 'r-.', label='FAR', markersize=2)
        ax.plot(thresholds, fnr_p, 'b-^', label='FRR', markersize=2)
        ax.legend(); ax.grid(True)
        ax.set_title(f'FAR & FRR — {tag}')
        ax.set_xlabel('Threshold'); ax.set_ylabel('FAR, FRR (%)')
        pdf.savefig(fig); plt.close(fig)

        pdf.close()
    except Exception as e:
        print(f"  [warn] plot save failed: {e}")


# ══════════════════════════════════════════════════════════════
#  PLOTTING HELPER — loss & accuracy curves
# ══════════════════════════════════════════════════════════════

def plot_loss_acc(train_losses, val_losses,
                  train_accs, val_accs, results_dir):
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
    protocol       = CONFIG["protocol"]
    data_root      = CONFIG["data_root"]
    results_dir    = CONFIG["results_dir"]
    img_side       = CONFIG["img_side"]
    batch_size     = CONFIG["batch_size"]
    num_epochs     = CONFIG["num_epochs"]
    lr             = CONFIG["lr"]
    lr_step        = CONFIG["lr_step"]
    lr_gamma       = CONFIG["lr_gamma"]
    dropout        = CONFIG["dropout"]
    arcface_s      = CONFIG["arcface_s"]
    arcface_m      = CONFIG["arcface_m"]
    ce_weight      = CONFIG["ce_weight"]
    tl_weight      = CONFIG["tl_weight"]
    vit_floor_num  = CONFIG["vit_floor_num"]
    cnn_vit_weight = CONFIG["cnn_vit_weight"]
    train_ratio    = CONFIG["train_ratio"]
    gallery_ratio  = CONFIG["gallery_ratio"]
    val_ratio      = CONFIG["val_ratio"]
    seed           = CONFIG["random_seed"]
    save_every     = CONFIG["save_every"]
    eval_every     = CONFIG["eval_every"]
    nw             = CONFIG["num_workers"]

    assert protocol in ("closed-set", "open-set")

    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SF2Net on CASIA-MS")
    print(f"  Protocol : {protocol}")
    print(f"  Device   : {device}")
    print(f"  Data     : {data_root}")
    print(f"  Loss     : {ce_weight}*CE + {tl_weight}*TripletLoss(SRT)")
    print(f"  ViT floor: {vit_floor_num}, CNN/ViT weight: {cnn_vit_weight}")
    print(f"{'='*60}\n")

    # ---------- parse dataset ----------
    print("Scanning dataset …")
    id2paths = parse_casia_ms(data_root)
    n_total_ids = len(id2paths)
    n_total_imgs = sum(len(v) for v in id2paths.values())
    print(f"  Found {n_total_ids} identities, {n_total_imgs} images total.\n")

    # ---------- protocol-specific split ----------
    if protocol == "closed-set":
        train_samples, test_samples, label_map = split_closed_set(
            id2paths, train_ratio=train_ratio, seed=seed)
        num_classes = len(label_map)

        # Triplet datasets for training/val
        train_dataset = CASIAMSDatasetTriplet(train_samples, img_side=img_side, train=True)
        val_dataset = CASIAMSDatasetTriplet(test_samples, img_side=img_side, train=False)

        # Single-image datasets for evaluation
        gallery_eval = CASIAMSDatasetSingle(train_samples, img_side=img_side)
        probe_eval = CASIAMSDatasetSingle(test_samples, img_side=img_side)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        gallery_loader = DataLoader(gallery_eval, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        probe_loader = DataLoader(probe_eval, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

        print(f"  [closed-set] #classes={num_classes}\n")

    else:  # open-set
        (train_samples, val_samples, gallery_samples, probe_samples,
         train_label_map, test_label_map) = split_open_set(
            id2paths, train_ratio=train_ratio,
            gallery_ratio=gallery_ratio, val_ratio=val_ratio, seed=seed)
        num_classes = len(train_label_map)

        # Triplet datasets for training/val
        train_dataset = CASIAMSDatasetTriplet(train_samples, img_side=img_side, train=True)
        val_dataset = CASIAMSDatasetTriplet(val_samples, img_side=img_side, train=False)

        # Single-image datasets for evaluation
        gallery_eval = CASIAMSDatasetSingle(gallery_samples, img_side=img_side)
        probe_eval = CASIAMSDatasetSingle(probe_samples, img_side=img_side)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        gallery_loader = DataLoader(gallery_eval, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        probe_loader = DataLoader(probe_eval, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

        print(f"  [open-set] #train_classes={num_classes}\n")

    # ---------- model ----------
    print(f"Building SF2Net — num_classes={num_classes} …")
    net = SF2Net(
        num_classes=num_classes,
        vit_floor_num=vit_floor_num,
        weight=cnn_vit_weight,
        dropout=dropout,
        arcface_s=arcface_s,
        arcface_m=arcface_m)
    net.to(device)

    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
        net = DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    tl_criterion = TripletLoss(distance="SRT")
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    # ---------- training loop ----------
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_eer = 1.0

    last_eer = float("nan")
    last_rank1 = float("nan")

    print(f"\nStarting training for {num_epochs} epochs …")
    print(f"  EER / Rank-1 computed every {eval_every} epochs.\n")

    for epoch in range(num_epochs):
        t_loss, t_acc = run_one_epoch(
            epoch, net, train_loader, criterion, tl_criterion,
            optimizer, device, "training",
            ce_weight=ce_weight, tl_weight=tl_weight)
        v_loss, v_acc = run_one_epoch(
            epoch, net, val_loader, criterion, tl_criterion,
            optimizer, device, "testing",
            ce_weight=ce_weight, tl_weight=tl_weight)
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        _net = net.module if isinstance(net, DataParallel) else net

        # Periodic evaluation
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            tag = f"ep{epoch:04d}_{protocol.replace('-','')}"
            cur_eer, cur_aggr_eer, cur_rank1 = evaluate(
                _net, probe_loader, gallery_loader,
                device, out_dir=rst_eval, tag=tag)
            last_eer = cur_eer
            last_rank1 = cur_rank1

            if cur_eer < best_eer:
                best_eer = cur_eer
                torch.save(_net.state_dict(),
                           os.path.join(results_dir, "net_params_best_eer.pth"))
                print(f"  *** New best EER: {best_eer*100:.4f}% ***")

        # Every-10-epoch console print
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            ts = time.strftime("%H:%M:%S")
            eer_str = f"{last_eer*100:.4f}%" if not math.isnan(last_eer) else "N/A"
            rank1_str = f"{last_rank1:.2f}%" if not math.isnan(last_rank1) else "N/A"
            print(
                f"[{ts}] ep {epoch:04d} | "
                f"loss  train={t_loss:.5f}  val={v_loss:.5f} | "
                f"cls-acc  train={t_acc:.2f}%  val={v_acc:.2f}% | "
                f"EER={eer_str}  Rank-1={rank1_str}")

        # Save best classification model
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(_net.state_dict(),
                       os.path.join(results_dir, "net_params_best.pth"))

        # Periodic checkpoint
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            torch.save(_net.state_dict(),
                       os.path.join(results_dir, "net_params.pth"))
            plot_loss_acc(train_losses, val_losses, train_accs, val_accs, results_dir)

    # ---------- final evaluation ----------
    print("\n=== Final evaluation with best EER model ===")
    best_model_path = os.path.join(results_dir, "net_params_best_eer.pth")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(results_dir, "net_params_best.pth")

    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_model_path, map_location=device))

    final_eer, final_aggr_eer, final_rank1 = evaluate(
        eval_net, probe_loader, gallery_loader,
        device, out_dir=rst_eval,
        tag=f"FINAL_{protocol.replace('-','')}")

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
