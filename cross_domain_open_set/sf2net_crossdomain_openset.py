"""
SF2Net — Cross-Domain Open-Set Evaluations on Palm-Auth
=========================================================
All evaluations follow an open-set protocol: subject IDs are split
80 % / 20 % into disjoint train and test partitions.

  train_ratio = 0.80  (152 / 190 IDs used for training)
  test_ratio  = 0.20  (38 / 190 IDs used for evaluation only)

Gallery and probe are built from the TEST IDs only, with a 50 / 50
sample-level split so every test identity appears in both sets.

Settings (13 total)
────────────────────
  S_scanner         │ Train : perspective (all)  for train IDs
                    │ Test  : scanner            for test IDs

  S_scanner_to_persp│ Train : scanner            for scanner IDs
                    │ Test  : perspective (all)  for IDs with NO scanner data

  S_C (×11)         │ Train : perspective (¬C) + scanner  for train IDs
                    │ Test  : perspective (C)              for test IDs

Conditions: bf | close | far | fl | jf | pitch | roll | rnd | sf | text | wet

Scanner spectra kept: green | ir | yellow | pink | white

Results saved to:
  {BASE_RESULTS_DIR}/setting_scanner/
  {BASE_RESULTS_DIR}/setting_{C}/
  {BASE_RESULTS_DIR}/results_summary.txt
"""

# ==============================================================
#  CONFIG
# ==============================================================
CONFIG = {
    "palm_auth_data_root"  : "/home/pai-ng/Jamal/smartphone_data",
    "scanner_spectra"      : {"green", "ir", "yellow", "pink", "white"},

    # Open-set split
    "train_id_ratio"       : 0.80,   # fraction of IDs used for training
    "test_gallery_ratio"   : 0.50,   # sample-level gallery/probe split

    # Model
    "img_side"             : 128,
    "embedding_dim"        : 512,
    "dropout"              : 0.25,
    "arcface_s"            : 30.0,
    "arcface_m"            : 0.50,

    # Training
    "batch_size"             : 256,
    "num_epochs"             : 200,
    "lr"             : 0.001,
    "lr_step"             : 17,
    "lr_gamma"             : 0.8,
    "augment_factor"             : 4,

    # Misc
    "base_results_dir"     : "./rst_sf2net_crossdomain_openset",
    "random_seed"          : 42,
    "save_every"           : 50,
    "eval_every"           : 50,
    "num_workers"          : 4,
}

ALL_CONDITIONS = ["bf", "close", "far", "fl", "jf",
                  "pitch", "roll", "rnd", "sf", "text", "wet"]
# ==============================================================

import os
import json
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

try:
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange
except ImportError:
    os.system("pip install einops --quiet")
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange

warnings.filterwarnings("ignore")

IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


# ══════════════════════════════════════════════════════════════
#  TRIPLET LOSS  (exact official SF2Net — unchanged)
# ══════════════════════════════════════════════════════════════

class TripletLoss(nn.Module):
    """Triplet loss with SRT (Soft Relative Triplet) distance."""
    def __init__(self, margin=2.0, alpha=0.95, distance="SRT"):
        super().__init__()
        self.margin = margin
        self.alpha  = alpha
        self.distance = distance
        self.tripletMargin = nn.TripletMarginLoss(margin=1.0, swap=True, reduction='mean')

    def dis(self, a, b):
        return torch.sum((a - b).pow(2), 1)

    def forward(self, anchor, positive, negative, size_average=True):
        if self.distance == "SRT":
            self.margin = 2.0
            anchor   = F.normalize(anchor,   p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)
            pos_d  = self.dis(anchor, positive)
            neg_d  = self.dis(anchor, negative)
            pn_d   = self.dis(positive, negative)
            cond   = neg_d.mean() >= pn_d.mean()
            ls     = torch.where(cond,
                                 pos_d + self.margin - pn_d.mean(),
                                 pos_d + self.margin - neg_d)
            losses = F.relu(ls).mean()
            return losses, pos_d.mean(), neg_d.mean(), pn_d.mean()
        else:
            raise ValueError(f"Unsupported distance: {self.distance}. Use 'SRT'.")


# ══════════════════════════════════════════════════════════════
#  ARCFACE  (exact official SF2Net — unchanged)
# ══════════════════════════════════════════════════════════════

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super().__init__()
        self.s = s; self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m

    def forward(self, inp, label=None):
        cosine = F.linear(F.normalize(inp), F.normalize(self.weight))
        if self.training:
            assert label is not None
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
            phi  = cosine * self.cos_m - sine * self.sin_m
            phi  = (torch.where(cosine > 0, phi, cosine) if self.easy_margin
                    else torch.where(cosine > self.th, phi, cosine - self.mm))
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            return self.s * ((one_hot * phi) + ((1.0 - one_hot) * cosine))
        return self.s * cosine


# ══════════════════════════════════════════════════════════════
#  GABOR CONV  (exact official SF2Net — unchanged)
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super().__init__()
        self.channel_in  = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.init_ratio  = init_ratio
        self.kernel      = 0
        self.sigma     = nn.Parameter(torch.FloatTensor([9.2  * init_ratio]), requires_grad=True)
        self.gamma     = nn.Parameter(torch.FloatTensor([2.0]),               requires_grad=True)
        self.theta     = nn.Parameter(
            torch.arange(0, channel_out).float() * math.pi / channel_out,
            requires_grad=False)
        self.frequency = nn.Parameter(torch.FloatTensor([0.057 / init_ratio]), requires_grad=True)
        self.psi       = nn.Parameter(torch.FloatTensor([0]),                  requires_grad=False)

    def get_gabor(self):
        half = self.kernel_size // 2
        x_0 = torch.arange(-half, half + 1).float()
        y_0 = torch.arange(-half, half + 1).float()
        k   = self.kernel_size
        x = x_0.view(-1, 1).repeat(self.channel_out, self.channel_in, 1, k)
        y = y_0.view(1, -1).repeat(self.channel_out, self.channel_in, k, 1)
        x = x.float().to(self.sigma.device)
        y = y.float().to(self.sigma.device)
        xt =  x*torch.cos(self.theta.view(-1,1,1,1)) + y*torch.sin(self.theta.view(-1,1,1,1))
        yt = -x*torch.sin(self.theta.view(-1,1,1,1)) + y*torch.cos(self.theta.view(-1,1,1,1))
        gb = -torch.exp(
            -0.5*((self.gamma*xt)**2 + yt**2) / (8*self.sigma.view(-1,1,1,1)**2)
        ) * torch.cos(2*math.pi*self.frequency.view(-1,1,1,1)*xt + self.psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2,3], keepdim=True)

    def forward(self, x):
        self.kernel = self.get_gabor()
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)


# ══════════════════════════════════════════════════════════════
#  SE MODULE  (exact official SF2Net — unchanged)
# ══════════════════════════════════════════════════════════════

class SEModule(nn.Module):
    def __init__(self, channel, reduction=1):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


# ══════════════════════════════════════════════════════════════
#  SEQUENCE FEATURE EXTRACTOR  (exact official SF2Net — unchanged)
# ══════════════════════════════════════════════════════════════

def get_sequence_feature(feature_tensor, vit_floor_num):
    ft    = torch.softmax(feature_tensor, dim=1)
    front = ft[:, :vit_floor_num, :, :]
    back  = ft[:, -vit_floor_num:, :, :]
    return torch.cat((front, back), dim=1)


class FeatureExtraction(nn.Module):
    def __init__(self, channel_in, filter_num, kernel_size, stride, padding,
                 init_ratio, label_num, vit_floor_num):
        super().__init__()
        self.vit_floor_num  = vit_floor_num
        self.gabor_conv2d_1 = GaborConv2d(channel_in, filter_num, kernel_size,
                                          stride, padding, init_ratio)
        self.gabor_conv2d_2 = GaborConv2d(filter_num, filter_num, kernel_size,
                                          stride, padding, init_ratio)
        self.se     = SEModule(channel=filter_num)
        self.conv_0 = nn.Conv2d(filter_num, 64, 5, 1, 0)
        self.conv_1 = nn.Conv2d(filter_num, 64, 5, 1, 0)
        self.conv_2 = nn.Conv2d(64, 32, 3, 2, 0)
        self.conv_3 = nn.Conv2d(64, 32, 3, 2, 0)
        self.max_pool = nn.MaxPool2d(2, 2)

    def process_block(self, x, conv):
        x = self.se(x); x = conv(x); x = torch.relu(x); x = self.max_pool(x)
        return x

    def forward(self, x):
        f1 = self.gabor_conv2d_1(x)
        f2 = self.gabor_conv2d_2(f1)
        f1p = self.process_block(f1, self.conv_0)
        f2p = self.process_block(f2, self.conv_1)
        out1 = self.conv_2(f1p)
        out2 = self.conv_3(f2p)
        feat = torch.cat((out1.flatten(1), out2.flatten(1)), dim=1)
        seq1 = get_sequence_feature(f1p, self.vit_floor_num)
        seq2 = get_sequence_feature(f2p, self.vit_floor_num)
        return feat, seq1, seq2


# ══════════════════════════════════════════════════════════════
#  VIT  (exact official SF2Net — unchanged)
# ══════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    def __init__(self, dim, dim_for_mlp, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim_for_mlp), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_for_mlp, dim), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_for_head, dropout=0.1):
        super().__init__()
        self.heads   = heads
        inner        = dim_for_head * heads
        self.scale   = dim_for_head ** -0.5
        self.norm    = nn.LayerNorm(dim)
        self.attend  = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv  = nn.Linear(dim, inner * 3, bias=False)
        self.to_out  = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out  = rearrange(torch.matmul(attn, v), 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_for_head, dim_for_mlp, dropout=0.1):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_for_head, dropout),
                FeedForward(dim, dim_for_mlp, dropout)
            ]) for _ in range(depth)])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x; x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, channels, num_classes, depth,
                 heads, dim, dim_for_head, dim_for_mlp, pool='cls',
                 dropout=0.1, emb_dropout=0.1):
        super().__init__()
        ih, iw = image_size, image_size
        ph, pw = patch_size, patch_size
        assert ih % ph == 0 and iw % pw == 0
        num_patches = (ih // ph) * (iw // pw)
        patch_dim   = channels * ph * pw
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=ph, p2=pw),
            nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token     = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout       = nn.Dropout(emb_dropout)
        self.transformer   = Transformer(dim, depth, heads, dim_for_head, dim_for_mlp, dropout)
        self.to_latent     = nn.Identity()

    def forward(self, x):
        x = self.to_patch(x)
        b, n, _ = x.shape
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(self.dropout(x))
        return self.to_latent(x)


# ══════════════════════════════════════════════════════════════
#  SF2NET MODEL  (exact official SF2Net — unchanged)
# ══════════════════════════════════════════════════════════════

class SF2Net(nn.Module):
    """SF2Net: Sequence Feature Fusion Network for Palmprint Verification."""
    def __init__(self, num_classes, vit_floor_num=10, weight=0.7,
                 dropout=0.5, arcface_s=30.0, arcface_m=0.50):
        super().__init__()
        self.num_classes   = num_classes
        self.vit_floor_num = vit_floor_num
        self.weight        = weight

        self.feature_extraction = FeatureExtraction(
            channel_in=1, filter_num=36, kernel_size=17, stride=2, padding=8,
            init_ratio=0.5, label_num=num_classes, vit_floor_num=vit_floor_num)

        self.vit_0 = ViT(image_size=30, patch_size=5, channels=vit_floor_num*2,
                         num_classes=num_classes, depth=2, heads=16, dim=128,
                         dim_for_head=64, dim_for_mlp=256, dropout=0.1, emb_dropout=0.1)
        self.vit_1 = ViT(image_size=14, patch_size=2, channels=vit_floor_num*2,
                         num_classes=num_classes, depth=2, heads=16, dim=128,
                         dim_for_head=64, dim_for_mlp=256, dropout=0.1, emb_dropout=0.1)

        self.fc1  = nn.Linear(7424, 2048)
        self.fc2  = nn.Linear(2048, 1024)
        self.vfc1 = nn.Linear(11136, 4096)
        self.vfc2 = nn.Linear(4096, 1024)

        self.dropout = nn.Dropout(p=dropout)
        self.arcface = ArcMarginProduct(1024, num_classes, s=arcface_s, m=arcface_m)

    def _process(self, x):
        feat, seq1, seq2 = self.feature_extraction(x)
        vit1 = self.vit_0(seq1); vit2 = self.vit_1(seq2)
        vit_cat = torch.cat((vit1, vit2), dim=1).flatten(1)
        cnn_out = self.fc2(self.fc1(feat))
        vit_out = self.vfc2(self.vfc1(vit_cat))
        return cnn_out * self.weight + vit_out * (1 - self.weight)

    def forward(self, x, y=None):
        x   = self._process(x)
        out = self.arcface(self.dropout(x), y)
        return out, F.normalize(x, dim=-1)

    @torch.no_grad()
    def get_embedding(self, x):
        """L2-normalised 1024-d embedding for matching."""
        return F.normalize(self._process(x), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  NORMALISATION
# ══════════════════════════════════════════════════════════════

class NormSingleROI:
    def __init__(self, outchannels=1): self.outchannels = outchannels
    def __call__(self, tensor):
        c, h, w = tensor.size(); tensor = tensor.view(c, h * w)
        idx = tensor > 0; t = tensor[idx]
        tensor[idx] = t.sub_(t.mean()).div_(t.std() + 1e-6)
        tensor = tensor.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, self.outchannels, dim=0)
        return tensor


# ══════════════════════════════════════════════════════════════
#  DATA COLLECTION HELPERS
# ══════════════════════════════════════════════════════════════

def _collect_perspective(data_root):
    """condition → identity → [path, ...]"""
    cond_paths = defaultdict(lambda: defaultdict(list))
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir = os.path.join(data_root, subject_id)
        if not os.path.isdir(subject_dir): continue
        roi_dir = os.path.join(subject_dir, "roi_perspective")
        if not os.path.isdir(roi_dir): continue
        for fname in sorted(os.listdir(roi_dir)):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
            parts = os.path.splitext(fname)[0].split("_")
            if len(parts) < 3: continue
            identity  = parts[0] + "_" + parts[1].lower()
            condition = parts[2].lower()
            cond_paths[condition][identity].append(os.path.join(roi_dir, fname))
    return cond_paths


def _collect_scanner(data_root, scanner_spectra):
    """identity → [path, ...]"""
    scanner_paths = defaultdict(list)
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir = os.path.join(data_root, subject_id)
        if not os.path.isdir(subject_dir): continue
        scan_dir = os.path.join(subject_dir, "roi_scanner")
        if not os.path.isdir(scan_dir): continue
        for fname in sorted(os.listdir(scan_dir)):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
            parts = os.path.splitext(fname)[0].split("_")
            if len(parts) < 4: continue
            if parts[2].lower() not in scanner_spectra: continue
            identity = parts[0] + "_" + parts[1].lower()
            scanner_paths[identity].append(os.path.join(scan_dir, fname))
    return scanner_paths


def _split_ids(ids, train_ratio, seed):
    """Split a sorted list of IDs into (train_ids, test_ids) with no overlap."""
    rng = random.Random(seed)
    ids = list(ids); rng.shuffle(ids)
    n_train = max(1, int(len(ids) * train_ratio))
    return sorted(ids[:n_train]), sorted(ids[n_train:])


def _all_samples(id2paths, label_map):
    """Flatten id2paths → flat (path, label) list (used for training)."""
    return [(p, label_map[ident])
            for ident, paths in id2paths.items()
            for p in paths]


def _gallery_probe_split(id2paths, label_map, gallery_ratio, rng):
    """50/50 sample-level gallery/probe split — every test ID appears in both."""
    gallery, probe = [], []
    for ident, paths in id2paths.items():
        paths = list(paths); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        n_gal = min(n_gal, len(paths) - 1)   # guarantee at least 1 probe
        if len(paths) == 1:                   # edge case: duplicate single image
            gallery.append((paths[0], label_map[ident]))
            probe.append((paths[0], label_map[ident]))
        else:
            for p in paths[:n_gal]: gallery.append((p, label_map[ident]))
            for p in paths[n_gal:]: probe.append((p, label_map[ident]))
    return gallery, probe


# ══════════════════════════════════════════════════════════════
#  PARSERS — OPEN-SET SETTINGS
# ══════════════════════════════════════════════════════════════

def parse_setting_scanner(cond_paths, scanner_paths,
                          train_id_ratio, gallery_ratio, seed):
    """
    S_scanner — Perspective (train IDs) → Scanner (test IDs)
    ──────────────────────────────────────────────────────────
    1. From IDs that HAVE scanner data, select 38 (20% of 190) as test IDs.
    2. Test  : scanner images for those 38 IDs → 50/50 gallery/probe
    3. Train : the remaining 152 IDs (all perspective images, NO scanner)
    """
    rng = random.Random(seed)

    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)

    all_persp_ids  = sorted(persp_all.keys())          # 190 IDs
    scanner_ids    = sorted(scanner_paths.keys())       # 148 IDs

    # Pick n_test IDs from scanner IDs to form the test set
    n_test  = len(all_persp_ids) - int(len(all_persp_ids) * train_id_ratio)  # 38
    rng_obj = random.Random(seed)
    test_ids  = sorted(rng_obj.sample(scanner_ids, n_test))
    train_ids = sorted(set(all_persp_ids) - set(test_ids))                   # 152

    train_label_map = {ident: i for i, ident in enumerate(train_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(test_ids)}
    num_train_cls   = len(train_ids)

    # Train: perspective only for all 152 train IDs
    train_samples = _all_samples(
        {i: persp_all[i] for i in train_ids if i in persp_all},
        train_label_map)

    # Test: scanner images for the 38 test IDs, 50/50 gallery/probe
    gallery_samples, probe_samples = _gallery_probe_split(
        {i: scanner_paths[i] for i in test_ids},
        test_label_map, gallery_ratio, rng)

    _print_stats("S_scanner | Perspective (train IDs) → Scanner (test IDs)",
                 len(train_ids), len(test_ids), len(train_samples),
                 len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_train_cls

def parse_setting_scanner_to_perspective(cond_paths, scanner_paths,
                                         train_id_ratio, gallery_ratio, seed):
    """
    S_scanner_to_persp — Scanner (scanner IDs) → Perspective (no-scanner IDs)
    ────────────────────────────────────────────────────────────────────────────
    Train : ALL scanner images for IDs that HAVE scanner data
    Test  : ALL perspective images for IDs that have NO scanner data
            → 50/50 gallery/probe by samples
    """
    rng = random.Random(seed)
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)

    scanner_ids    = sorted(scanner_paths.keys())
    no_scanner_ids = sorted(set(persp_all.keys()) - set(scanner_ids))

    if not no_scanner_ids:
        raise ValueError("S_scanner_to_persp: every perspective ID also has scanner data!")

    train_label_map = {ident: i for i, ident in enumerate(scanner_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(no_scanner_ids)}
    num_train_cls   = len(scanner_ids)

    train_samples = _all_samples(scanner_paths, train_label_map)
    gallery_samples, probe_samples = _gallery_probe_split(
        {i: persp_all[i] for i in no_scanner_ids},
        test_label_map, gallery_ratio, rng)

    _print_stats("S_scanner_to_persp | Scanner (all) → Perspective (no-scanner IDs)",
                 len(scanner_ids), len(no_scanner_ids), len(train_samples),
                 len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_train_cls


def parse_setting_leave_one_condition(target_condition, cond_paths, scanner_paths,
                                      train_id_ratio, gallery_ratio, seed):
    """
    S_C — Leave-one-condition-out (open set)
    ─────────────────────────────────────────
    IDs with condition C images are split 80/20:
      Train (80 %): perspective (¬C) + scanner for train IDs
      Test  (20 %): condition C images for test IDs → 50/50 gallery/probe
    """
    rng = random.Random(seed)
    test_id2paths = cond_paths.get(target_condition, {})
    if not test_id2paths:
        raise ValueError(f"No images for condition '{target_condition}'")

    all_cond_ids = sorted(test_id2paths.keys())
    train_ids, test_ids = _split_ids(all_cond_ids, train_id_ratio, seed)

    train_label_map = {ident: i for i, ident in enumerate(train_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(test_ids)}
    num_train_cls   = len(train_ids)

    # Train = non-target perspective + scanner, for train_ids only
    train_samples = []
    for cond, cond_dict in cond_paths.items():
        if cond == target_condition: continue
        for ident in train_ids:
            for p in cond_dict.get(ident, []):
                train_samples.append((p, train_label_map[ident]))
    for ident in train_ids:
        for p in scanner_paths.get(ident, []):
            train_samples.append((p, train_label_map[ident]))

    # Test = condition C samples for test_ids, split gallery/probe
    gallery_samples, probe_samples = _gallery_probe_split(
        {i: test_id2paths[i] for i in test_ids},
        test_label_map, gallery_ratio, rng)

    _print_stats(
        f"S_{target_condition} | Perspective (¬{target_condition}) + Scanner (train IDs)"
        f" → {target_condition} (test IDs)",
        len(train_ids), len(test_ids), len(train_samples),
        len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_train_cls


def _print_stats(name, n_train_ids, n_test_ids, train_n, gallery_n, probe_n):
    print(f"\n  [{name}]")
    print(f"    Train IDs / Test IDs  : {n_train_ids} / {n_test_ids}")
    print(f"    Train images          : {train_n}")
    print(f"    Gallery / Probe       : {gallery_n} / {probe_n}")


# ══════════════════════════════════════════════════════════════
#  FIXED MODEL INITIALISATION
# ══════════════════════════════════════════════════════════════

def get_or_create_init_weights(net, num_classes, cache_dir, device):
    os.makedirs(cache_dir, exist_ok=True)
    model_name   = type(net.module if isinstance(net, DataParallel) else net).__name__
    weights_path = os.path.join(cache_dir,
                                f"init_weights_{model_name}_nc{num_classes}.pth")
    _net = net.module if isinstance(net, DataParallel) else net
    if os.path.exists(weights_path):
        print(f"  Loading cached init weights: {weights_path}")
        _net.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"  Saving init weights: {weights_path}")
        torch.save(_net.state_dict(), weights_path)
    return net


# ══════════════════════════════════════════════════════════════
#  PYTORCH DATASETS
# ══════════════════════════════════════════════════════════════

class SingleDataset(Dataset):
    def __init__(self, samples, img_side=128):
        self.samples   = samples
        self.transform = T.Compose([T.Resize(img_side), T.ToTensor(),
                                    NormSingleROI(outchannels=1)])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("L")), label


class AugmentedDataset(Dataset):
    def __init__(self, samples, img_side=128, augment_factor=1):
        self.samples = samples; self.augment_factor = augment_factor
        self.aug_transform = T.Compose([
            T.Resize(img_side),
            T.RandomChoice([
                T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                T.RandomResizedCrop(img_side, scale=(0.8,1.0), ratio=(1.0,1.0)),
                T.RandomPerspective(distortion_scale=0.15, p=1),
                T.RandomChoice([
                    T.RandomRotation(10, interpolation=Image.BICUBIC,
                                     expand=False, center=(0.5*img_side, 0.0)),
                    T.RandomRotation(10, interpolation=Image.BICUBIC,
                                     expand=False, center=(0.0, 0.5*img_side)),
                ]),
            ]),
            T.ToTensor(), NormSingleROI(outchannels=1),
        ])
    def __len__(self): return len(self.samples) * self.augment_factor
    def __getitem__(self, index):
        real_idx = index % len(self.samples)
        path, label = self.samples[real_idx]
        return self.aug_transform(Image.open(path).convert("L")), label


# ══════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════

def run_one_epoch(model, loader, criterion, optimizer, device, phase):
    is_train = (phase == "training")
    model.train() if is_train else model.eval()
    running_loss = 0.0; running_correct = 0; total = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            if is_train: optimizer.zero_grad()
            output = model(data, target if is_train else None)
            loss   = criterion(output, target)
            if is_train: loss.backward(); optimizer.step()
            running_loss    += loss.item() * data.size(0)
            running_correct += output.data.max(1)[1].eq(target).sum().item()
            total           += data.size(0)
    return running_loss / max(total, 1), 100.0 * running_correct / max(total, 1)


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval(); feats, labels = [], []
    for imgs, labs in loader:
        feats.append(model.get_embedding(imgs.to(device)).cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def compute_eer(scores_array):
    ins  = scores_array[scores_array[:, 1] ==  1, 0]
    outs = scores_array[scores_array[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0: return 1.0, 0.0
    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s   = np.concatenate([ins, outs])
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    eer    = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))
    return eer, thresh


def evaluate(model, probe_loader, gallery_loader, device,
             out_dir=".", tag="eval"):
    probe_feats,   probe_labels   = extract_features(model, probe_loader,   device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)
    n_probe    = len(probe_feats)
    sim_matrix = probe_feats @ gallery_feats.T

    scores_list, labels_list = [], []
    for i in range(n_probe):
        for j in range(sim_matrix.shape[1]):
            scores_list.append(float(sim_matrix[i, j]))
            labels_list.append(1 if probe_labels[i] == gallery_labels[j] else -1)

    scores_arr = np.column_stack([scores_list, labels_list])
    eer, _     = compute_eer(scores_arr)

    nn_idx  = np.argmax(sim_matrix, axis=1)
    correct = sum(probe_labels[i] == gallery_labels[nn_idx[i]] for i in range(n_probe))
    rank1   = 100.0 * correct / max(n_probe, 1)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list): f.write(f"{s} {l}\n")

    print(f"  [{tag}]  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return eer, rank1


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════

def run_experiment(train_samples, gallery_samples, probe_samples,
                   num_classes, cfg, results_dir, device):
    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    img_side       = cfg["img_side"]
    batch_size     = cfg["batch_size"]
    num_epochs     = cfg["num_epochs"]
    augment_factor = cfg["augment_factor"]
    nw             = cfg["num_workers"]
    eval_every     = cfg["eval_every"]
    save_every     = cfg["save_every"]

    train_loader = DataLoader(
        TripletDataset(train_samples, img_side, augment_factor),
        batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    gallery_loader = DataLoader(
        SingleDataset(gallery_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    probe_loader = DataLoader(
        SingleDataset(probe_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    net = SF2Net(num_classes=num_classes, vit_floor_num=cfg["vit_floor_num"],
                  weight=cfg["cnn_vit_weight"], dropout=cfg["dropout"],
                  arcface_s=cfg["arcface_s"], arcface_m=cfg["arcface_m"])
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    net = get_or_create_init_weights(net, num_classes,
                                     cfg["base_results_dir"], device)

    criterion    = nn.CrossEntropyLoss()
    tl_criterion = TripletLoss(distance="SRT")
    optimizer = optim.Adam(net.parameters(), lr=cfg["lr"])
    scheduler = lr_scheduler.StepLR(optimizer, cfg["lr_step"], cfg["lr_gamma"])

    _net = net.module if isinstance(net, DataParallel) else net
    pre_eer, pre_r1 = evaluate(_net, probe_loader, gallery_loader,
                                device, out_dir=rst_eval, tag="ep-001_pretrain")
    best_eer = pre_eer; last_eer = pre_eer; last_rank1 = pre_r1
    torch.save(_net.state_dict(),
               os.path.join(results_dir, "net_params_best_eer.pth"))

    train_losses, train_accs = [], []

    for epoch in range(num_epochs):
        t_loss, t_acc = run_one_epoch(
            net, train_loader, criterion, tl_criterion,
            optimizer, device, "training",
            ce_weight=cfg["ce_weight"], tl_weight=cfg["tl_weight"])
        scheduler.step()
        train_losses.append(t_loss); train_accs.append(t_acc)
        _net = net.module if isinstance(net, DataParallel) else net

        if (epoch % eval_every == 0 and epoch > 0) or epoch == num_epochs - 1:
            cur_eer, cur_rank1 = evaluate(
                _net, probe_loader, gallery_loader,
                device, out_dir=rst_eval, tag=f"ep{epoch:04d}")
            last_eer, last_rank1 = cur_eer, cur_rank1
            if cur_eer < best_eer:
                best_eer = cur_eer
                torch.save(_net.state_dict(),
                           os.path.join(results_dir, "net_params_best_eer.pth"))
                print(f"  *** New best EER: {best_eer*100:.4f}% ***")

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            ts = time.strftime("%H:%M:%S")
            eer_str   = f"{last_eer*100:.4f}%"  if not math.isnan(last_eer)   else "N/A"
            rank1_str = f"{last_rank1:.2f}%"     if not math.isnan(last_rank1) else "N/A"
            print(f"  [{ts}] ep {epoch:04d} | loss={t_loss:.4f} | acc={t_acc:.2f}% | "
                  f"EER={eer_str}  Rank-1={rank1_str}")

        if epoch % save_every == 0 or epoch == num_epochs - 1:
            torch.save(_net.state_dict(),
                       os.path.join(results_dir, "net_params.pth"))

    best_path = os.path.join(results_dir, "net_params_best_eer.pth")
    if not os.path.exists(best_path):
        best_path = os.path.join(results_dir, "net_params.pth")
    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_path, map_location=device))
    final_eer, final_rank1 = evaluate(
        eval_net, probe_loader, gallery_loader,
        device, out_dir=rst_eval, tag="FINAL")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(train_losses, 'b'); axes[0].set_title("Train Loss")
        axes[0].set_xlabel("epoch"); axes[0].grid(True)
        axes[1].plot(train_accs,   'b'); axes[1].set_title("Train Acc (%)")
        axes[1].set_xlabel("epoch"); axes[1].grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "train_curves.png"))
        plt.close(fig)
    except Exception:
        pass

    return final_eer, final_rank1


# ══════════════════════════════════════════════════════════════
#  RESULTS SUMMARY TABLE
# ══════════════════════════════════════════════════════════════

def print_and_save_summary(all_results, out_path):
    col_w  = 14
    header = (f"{'Setting':<22}"
              f"{'Train domain':<38}"
              f"{'Test domain':<26}"
              f"{'EER (%)':>{col_w}}"
              f"{'Rank-1 (%)':>{col_w}}")
    sep = "─" * len(header)
    lines = ["\nCross-Domain Open-Set Results — Palm-Auth (CompNet)", sep, header, sep]

    for r in all_results:
        eer_str   = f"{r['eer']:.2f}"   if r['eer']   is not None else "—"
        rank1_str = f"{r['rank1']:.2f}" if r['rank1'] is not None else "—"
        lines.append(f"{r['setting']:<22}"
                     f"{r['train_desc']:<38}"
                     f"{r['test_desc']:<26}"
                     f"{eer_str:>{col_w}}"
                     f"{rank1_str:>{col_w}}")
    lines.append(sep)

    text = "\n".join(lines)
    print(text)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nSummary saved to: {out_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    cfg  = CONFIG
    seed = cfg["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_results_dir = cfg["base_results_dir"]
    os.makedirs(base_results_dir, exist_ok=True)

    train_ratio  = cfg["train_id_ratio"]
    gallery_ratio = cfg["test_gallery_ratio"]

    print(f"\n{'='*60}")
    print(f"  CompNet — Cross-Domain Open-Set (Palm-Auth)")
    print(f"  Protocol  : open set (80/20 ID split, no overlap)")
    print(f"  Train IDs : {train_ratio*100:.0f}%   Test IDs: {(1-train_ratio)*100:.0f}%")
    print(f"  Gallery/Probe: {gallery_ratio*100:.0f}/{(1-gallery_ratio)*100:.0f} sample split")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {cfg['num_epochs']}")
    print(f"  Settings  : 2 scanner + {len(ALL_CONDITIONS)} leave-one-condition-out")
    print(f"  Results   : {base_results_dir}")
    print(f"{'='*60}")

    print("\n  Scanning dataset …")
    cond_paths    = _collect_perspective(cfg["palm_auth_data_root"])
    scanner_paths = _collect_scanner(cfg["palm_auth_data_root"],
                                     cfg["scanner_spectra"])
    print(f"  Perspective conditions found : {sorted(cond_paths.keys())}")
    print(f"  Scanner identities found     : {len(scanner_paths)}")

    SETTINGS = []

    SETTINGS.append({
        "tag"        : "setting_scanner",
        "label"      : "S_scanner",
        "train_desc" : "Perspective (all, train IDs)",
        "test_desc"  : "Scanner (test IDs)",
        "parser"     : lambda: parse_setting_scanner(
                           cond_paths, scanner_paths,
                           train_ratio, gallery_ratio, seed),
    })

    SETTINGS.append({
        "tag"        : "setting_scanner_to_persp",
        "label"      : "S_scanner_to_persp",
        "train_desc" : "Scanner (all scanner IDs)",
        "test_desc"  : "Perspective (no-scanner IDs)",
        "parser"     : lambda: parse_setting_scanner_to_perspective(
                           cond_paths, scanner_paths,
                           train_ratio, gallery_ratio, seed),
    })

    conditions_found = sorted(cond_paths.keys())
    for cond in ALL_CONDITIONS:
        if cond not in conditions_found:
            print(f"  [WARN] condition '{cond}' not found — skipping")
            continue
        c = cond
        SETTINGS.append({
            "tag"        : f"setting_{c}",
            "label"      : f"S_{c}",
            "train_desc" : f"Perspective (¬{c}) + Scanner (train IDs)",
            "test_desc"  : f"Perspective ({c}) (test IDs)",
            "parser"     : (lambda c=c: parse_setting_leave_one_condition(
                                c, cond_paths, scanner_paths,
                                train_ratio, gallery_ratio, seed)),
        })

    print(f"\n  Total settings to run : {len(SETTINGS)}")

    all_results = []

    for idx, s in enumerate(SETTINGS, 1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{len(SETTINGS)}] {s['label']}")
        print(f"  Train : {s['train_desc']}")
        print(f"  Test  : {s['test_desc']}")
        print(f"{'='*60}")

        results_dir = os.path.join(base_results_dir, s["tag"])
        t_start     = time.time()
        try:
            train_s, gal_s, probe_s, n_cls = s["parser"]()
            eer, rank1 = run_experiment(
                train_s, gal_s, probe_s, n_cls, cfg, results_dir, device)
            elapsed = time.time() - t_start
            print(f"\n  ✓  {s['label']}:  EER={eer*100:.4f}%  "
                  f"Rank-1={rank1:.2f}%  Time={elapsed/60:.1f} min")
            with open(os.path.join(results_dir, "results.json"), "w") as f:
                json.dump({"setting": s["label"], "train_desc": s["train_desc"],
                           "test_desc": s["test_desc"], "num_train_classes": n_cls,
                           "EER_pct": eer*100, "Rank1_pct": rank1}, f, indent=2)
            all_results.append({"setting": s["label"], "train_desc": s["train_desc"],
                                 "test_desc": s["test_desc"],
                                 "eer": eer*100, "rank1": rank1})
        except Exception as e:
            print(f"\n  ✗  {s['label']} FAILED: {e}")
            all_results.append({"setting": s["label"], "train_desc": s["train_desc"],
                                 "test_desc": s["test_desc"], "eer": None, "rank1": None})

    print(f"\n\n{'='*60}")
    print(f"  ALL {len(SETTINGS)} SETTINGS COMPLETE")
    print(f"{'='*60}")
    print_and_save_summary(
        all_results,
        os.path.join(base_results_dir, "results_summary.txt"))


if __name__ == "__main__":
    main()
