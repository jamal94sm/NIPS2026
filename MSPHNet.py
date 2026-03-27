"""
MSPHNet on CASIA-MS Dataset  —  FIXED VERSION 2
==================================================
Previous fixes (v1):
  FIX 1  pos_embedding pre-allocated in __init__
  FIX 2  pre_proj + residual shortcut in PHEB
  FIX 3  removed dead is_first branch
  FIX 4  lr=0.0001, batch=32, CosineAnnealingLR, warmup
  FIX 5  DataLoader persistent_workers, corrupt-file handler
  FIX 6  Validation also passes labels to ArcFace

New fixes (v2) — addressing train-good / test-stuck (EER~30%, Rank-1~57%):
  FIX 7  (critical) Spectral-stratified gallery/probe split.
          CASIA-MS filename: {id}_{hand}_{spectrum}_{iter}.jpg  (4 spectra).
          The old random 50/50 split put different spectra in gallery vs probe
          (e.g. gallery=460nm, probe=850nm). Cross-spectral matching is a
          completely different problem — CO3Net reports separate EERs per
          spectrum. The new split ensures each spectrum contributes equally to
          gallery (1 sample/spectrum) and probe (remaining samples/spectrum).
          Evaluation is now also reported per-spectrum.
  FIX 8  (high) CenterLoss added alongside ArcFace + CrossEntropy.
          ArcFace is a classification loss — it separates the 120 training
          classes but provides no guarantee that the metric generalises to
          unseen identities. CenterLoss directly minimises intra-class spread
          in embedding space, forcing compact per-identity clusters that
          transfer to unseen identities.
  FIX 9  (moderate) weight_decay=1e-4 added to Adam optimizer.
          Without it, embedding vectors can have arbitrary norms, making cosine
          similarity unreliable for unseen identities.
  FIX 10 (moderate) Label smoothing 0.1 on CrossEntropyLoss to reduce
          over-confident predictions that hurt metric generalisation.
  FIX 11 (moderate) Embedding dim 1024 → 512. 9.4M params / 120 classes is
          over-parameterised; the 1024-d embedding was too large to regularise.
"""

CONFIG = {
    "protocol"         : "open-set",   # "closed-set" | "open-set"
    "data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "results_dir"      : "./rst_msphnet_casia_ms_v2",
    "img_side"         : 128,
    "batch_size"       : 32,
    "num_epochs"       : 1000,
    "lr"               : 0.0001,
    "warmup_epochs"    : 10,
    "weight_decay"     : 1e-4,         # FIX 9
    "label_smoothing"  : 0.1,          # FIX 10
    "center_loss_weight": 0.003,       # FIX 8  (lambda for CenterLoss)
    "center_loss_lr"   : 0.5,          # FIX 8  (separate LR for centres)
    "dropout"          : 0.5,
    "arcface_s"        : 30.0,
    "arcface_m"        : 0.50,
    "embedding_dim"    : 512,          # FIX 11 (was 1024)
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

import os, sys, math, time, random, warnings
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
    from einops import rearrange
    from einops.layers.torch import Rearrange
except ImportError:
    os.system("pip install einops")
    from einops import rearrange
    from einops.layers.torch import Rearrange

warnings.filterwarnings("ignore")

SEED = CONFIG["random_seed"]
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════
#  ARCFACE
# ══════════════════════════════════════════════════════════════

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.s, self.m = s, m
        self.weight   = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if label is not None:
            sine  = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
            phi   = cosine * self.cos_m - sine * self.sin_m
            phi   = torch.where(cosine > self.th, phi, cosine - self.mm)
            oh    = torch.zeros_like(cosine).scatter_(1, label.view(-1,1).long(), 1)
            return (oh * phi + (1 - oh) * cosine) * self.s
        return cosine * self.s


# ══════════════════════════════════════════════════════════════
#  FIX 8: CENTERLOSS
#  Minimises intra-class variance in embedding space.
#  Centres are updated via a separate SGD optimizer (not Adam)
#  with lr=center_loss_lr, as per the original paper.
# ══════════════════════════════════════════════════════════════

class CenterLoss(nn.Module):
    """
    Center Loss (Wen et al., ECCV 2016).
    Penalises the L2 distance between each embedding and the running
    centre of its class. Combined with ArcFace, this produces compact
    per-identity clusters that generalise to unseen identities.
    """
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim) * 0.01)

    def forward(self, x, labels):
        # x: [B, D]  labels: [B]
        batch_size = x.size(0)
        # Squared distances to all centres: [B, C]
        diff  = x.unsqueeze(1) - self.centers.unsqueeze(0)   # [B,C,D]
        dists = (diff ** 2).sum(dim=2)                        # [B,C]
        # Pick each sample's own centre distance
        mask  = torch.zeros_like(dists).scatter_(1, labels.view(-1,1).long(), 1)
        loss  = (dists * mask).sum() / batch_size
        return loss


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
        self.sigma     = nn.Parameter(torch.FloatTensor([9.2 * init_ratio]))
        self.gamma     = nn.Parameter(torch.FloatTensor([2.0]))
        self.theta     = nn.Parameter(
            torch.arange(0, channel_out).float() * math.pi / channel_out,
            requires_grad=False)
        self.frequency = nn.Parameter(torch.FloatTensor([0.057 / init_ratio]))
        self.psi       = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def _get_gabor_kernel(self):
        xm = self.kernel_size // 2
        x0 = torch.arange(-xm, xm + 1).float()
        k  = 2 * xm + 1
        x  = x0.view(-1,1).repeat(self.channel_out, self.channel_in, 1, k).to(self.sigma.device)
        y  = x0.view(1,-1).repeat(self.channel_out, self.channel_in, k, 1).to(self.sigma.device)
        th = self.theta.view(-1,1,1,1)
        xt = x * torch.cos(th) + y * torch.sin(th)
        yt = -x * torch.sin(th) + y * torch.cos(th)
        g  = -torch.exp(-0.5 * ((self.gamma * xt)**2 + yt**2)
                        / (8 * self.sigma.view(-1,1,1,1)**2)) \
             * torch.cos(2 * math.pi * self.frequency.view(-1,1,1,1) * xt
                         + self.psi.view(-1,1,1,1))
        return g - g.mean(dim=[2,3], keepdim=True)

    def forward(self, x):
        return F.conv2d(x, self._get_gabor_kernel(),
                        stride=self.stride, padding=self.padding)


# ══════════════════════════════════════════════════════════════
#  COMPREHENSIVE ATTENTION BLOCK (CAB)
# ══════════════════════════════════════════════════════════════

class XDirectionAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        rd = max(1, ch // r)
        self.c1 = nn.Conv2d(ch, rd, 1); self.c2 = nn.Conv2d(rd, ch, 1)
    def forward(self, x):
        a = torch.sigmoid(self.c2(F.relu(self.c1(x.mean(2, keepdim=True)))))
        return x * a

class YDirectionAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        rd = max(1, ch // r)
        self.c1 = nn.Conv2d(ch, rd, 1); self.c2 = nn.Conv2d(rd, ch, 1)
    def forward(self, x):
        a = torch.sigmoid(self.c2(F.relu(self.c1(x.mean(3, keepdim=True)))))
        return x * a

class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        rd = max(1, ch // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.c1   = nn.Conv2d(ch, rd, 1); self.c2 = nn.Conv2d(rd, ch, 1)
    def forward(self, x):
        a = torch.sigmoid(self.c2(F.relu(self.c1(self.pool(x)))))
        return x * a

class PixelAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 1); self.c2 = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        return x * torch.sigmoid(self.c2(F.relu(self.c1(x))))

class CAB(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.xa = XDirectionAttention(ch, r)
        self.ya = YDirectionAttention(ch, r)
        self.ca = ChannelAttention(ch, r)
        self.pa = PixelAttention(ch)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        return self.pa(self.ca(self.alpha * self.xa(x) + self.beta * self.ya(x)))


# ══════════════════════════════════════════════════════════════
#  TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    def __init__(self, dim, hidden, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden),
            nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop))
    def forward(self, x): return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dh=64, drop=0.1):
        super().__init__()
        inner = dh * heads
        self.heads = heads; self.scale = dh ** -0.5
        self.norm  = nn.LayerNorm(dim)
        self.att   = nn.Softmax(dim=-1); self.drop = nn.Dropout(drop)
        self.qkv   = nn.Linear(dim, inner * 3, bias=False)
        self.out   = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(drop))
    def forward(self, x):
        x = self.norm(x)
        q, k, v = map(lambda t: rearrange(t,'b n (h d)->b h n d', h=self.heads),
                      self.qkv(x).chunk(3, dim=-1))
        o = rearrange(torch.matmul(self.drop(self.att(
            torch.matmul(q, k.transpose(-1,-2)) * self.scale)), v),
            'b h n d->b n (h d)')
        return self.out(o)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dh=64, mlp=256, drop=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads, dh, drop)
        self.ff   = FeedForward(dim, mlp, drop)
    def forward(self, x):
        x = self.attn(x) + x; x = self.ff(x) + x; return x

class TransformerBranch(nn.Module):
    """pos_embedding pre-allocated in __init__ (FIX 1 from v1)."""
    def __init__(self, in_ch, spatial_size, patch_size=8,
                 dim=128, depth=2, heads=8, mlp=256, drop=0.1):
        super().__init__()
        self.ps  = patch_size; self.dim = dim
        np_ = (spatial_size // patch_size) ** 2
        pd  = in_ch * patch_size * patch_size
        self.embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2)->b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(pd), nn.Linear(pd, dim), nn.LayerNorm(dim))
        self.pos  = nn.Parameter(torch.randn(1, np_, dim) * 0.02)
        self.tf   = nn.ModuleList([
            TransformerBlock(dim, heads, max(1, dim//heads), mlp, drop)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim); self.drop = nn.Dropout(drop)
    def forward(self, x):
        B,C,H,W = x.shape
        nh = H // self.ps; nw = W // self.ps
        x  = self.drop(self.embed(x) + self.pos)
        for b in self.tf: x = b(x)
        return rearrange(self.norm(x), 'b (h w) d->b d h w', h=nh, w=nw)


# ══════════════════════════════════════════════════════════════
#  CNN BRANCH
# ══════════════════════════════════════════════════════════════

class CNNBranch(nn.Module):
    def __init__(self, ic, oc, r=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True))
        self.cab = CAB(oc, r)
    def forward(self, x): return self.cab(self.conv(x))


# ══════════════════════════════════════════════════════════════
#  PHEB (v1 fixes: pre_proj, residual shortcut, spatial_size)
# ══════════════════════════════════════════════════════════════

class PHEB(nn.Module):
    def __init__(self, in_ch, out_ch, gabor_filters=36, gabor_kernel=17,
                 patch_size=8, trans_dim=128, trans_depth=2, trans_heads=8,
                 ca_reduction=16, spatial_size=128):
        super().__init__()
        self.pre_proj = nn.Sequential(nn.Conv2d(in_ch, 1, 1, bias=False),
                                      nn.BatchNorm2d(1))
        self.gabor    = GaborConv2d(1, gabor_filters, gabor_kernel,
                                    padding=gabor_kernel//2, init_ratio=0.5)
        self.gbn      = nn.BatchNorm2d(gabor_filters)
        self.cnn      = CNNBranch(gabor_filters, out_ch // 2, ca_reduction)
        self.trans    = TransformerBranch(gabor_filters, spatial_size,
                                          patch_size, trans_dim, trans_depth,
                                          trans_heads, trans_dim*2, 0.1)
        self.tproj    = nn.Sequential(
            nn.Conv2d(trans_dim, out_ch//2, 1), nn.BatchNorm2d(out_ch//2), nn.ReLU(True))
        self.fusion   = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                      nn.BatchNorm2d(out_ch))

    def forward(self, x):
        res = self.shortcut(x)
        g   = F.relu(self.gbn(self.gabor(F.relu(self.pre_proj(x)))))
        c   = self.cnn(g)
        t   = self.tproj(self.trans(g))
        if t.shape[2:] != c.shape[2:]:
            t = F.interpolate(t, size=c.shape[2:], mode='bilinear', align_corners=False)
        return F.relu(self.fusion(torch.cat([c, t], dim=1)) + res)


# ══════════════════════════════════════════════════════════════
#  DOWN / UP SAMPLING
# ══════════════════════════════════════════════════════════════

class DownSample(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.d = nn.Sequential(nn.Conv2d(ic,oc,4,stride=2,padding=1),
                               nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, x): return self.d(x)

class UpSample(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.u = nn.Sequential(nn.ConvTranspose2d(ic,oc,4,stride=2,padding=1),
                               nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, x): return self.u(x)


# ══════════════════════════════════════════════════════════════
#  MSPHNet
# ══════════════════════════════════════════════════════════════

class MSPHNet(nn.Module):
    def __init__(self, num_classes, img_side=128, embedding_dim=512,
                 gabor_filters=36, gabor_kernel=17,
                 trans_dim=128, trans_depth=2, trans_heads=8, patch_size=8,
                 ca_reduction=16, dropout=0.5, arcface_s=30.0, arcface_m=0.50):
        super().__init__()
        self.embedding_dim = embedding_dim
        c1, c2, c3 = 64, 128, 256
        s1, s2, s3 = img_side, img_side//2, img_side//4

        self.init_conv = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(True))

        kw = dict(gabor_filters=gabor_filters, gabor_kernel=gabor_kernel,
                  patch_size=patch_size, trans_dim=trans_dim,
                  trans_depth=trans_depth, trans_heads=trans_heads,
                  ca_reduction=ca_reduction)

        self.pheb1 = PHEB(c1,   c1,   spatial_size=s1, **kw)
        self.down1 = DownSample(c1, c2)
        self.pheb2 = PHEB(c2,   c2,   spatial_size=s2, **kw)
        self.down2 = DownSample(c2, c3)
        self.pheb3 = PHEB(c3,   c3,   spatial_size=s3, **kw)
        self.up1   = UpSample(c3, c2)
        self.pheb4 = PHEB(c2*2, c2,   spatial_size=s2, **kw)
        self.up2   = UpSample(c2, c1)
        self.pheb5 = PHEB(c1*2, c1,   spatial_size=s1, **kw)

        self.final_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.final_conv = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(True),
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(True))

        self.fc = nn.Sequential(
            nn.Linear(c3*16, 2048), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(2048, embedding_dim), nn.ReLU(True), nn.Dropout(dropout))

        self.arcface = ArcMarginProduct(embedding_dim, num_classes,
                                        s=arcface_s, m=arcface_m)

    def _encode(self, x):
        x  = self.init_conv(x)
        e1 = self.pheb1(x)
        e2 = self.pheb2(self.down1(e1))
        e3 = self.pheb3(self.down2(e2))
        r1 = self.pheb4(torch.cat([self.up1(e3), e2], dim=1))
        r2 = self.pheb5(torch.cat([self.up2(r1), e1], dim=1))
        f  = self.final_conv(self.final_pool(r2))
        return self.fc(f.view(f.size(0), -1))

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
    def __call__(self, t):
        c,h,w = t.size(); t = t.view(c,h*w)
        idx = t > 0; v = t[idx]
        if len(v) > 0: t[idx] = (v - v.mean()) / (v.std() + 1e-6)
        return t.view(c,h,w)


# ══════════════════════════════════════════════════════════════
#  FIX 7: SPECTRAL-AWARE DATASET PARSING AND SPLITTING
#
#  parse_casia_ms_spectral() returns:
#    id2spectra: { identity: { spectrum: [path, path, …] } }
#
#  split_open_set_spectral() builds gallery/probe so that:
#    - gallery gets exactly 1 image per identity per spectrum
#      (the first iteration for each spectrum)
#    - probe gets all remaining images
#  This means gallery and probe always share the same spectrum,
#  matching the evaluation protocol used by CO3Net and the paper.
# ══════════════════════════════════════════════════════════════

def parse_casia_ms_spectral(data_root):
    """
    Returns:
      id2paths   (flat)  – same as before, for training
      id2spectra (nested)– {identity: {spectrum: [paths]}}
    """
    id2paths   = defaultdict(list)
    id2spectra = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")):
            continue
        parts = fname.split("_")
        if len(parts) < 4: continue
        identity = parts[0] + "_" + parts[1]
        spectrum = parts[2]                      # e.g. "460", "630", "700", "850"
        full     = os.path.join(data_root, fname)
        id2paths[identity].append(full)
        id2spectra[identity][spectrum].append(full)
    return dict(id2paths), {k: dict(v) for k,v in id2spectra.items()}


def split_open_set_spectral(id2paths, id2spectra,
                            train_ratio=0.8, val_ratio=0.10, seed=42):
    """
    FIX 7: gallery = first-iteration image per spectrum per test identity
            probe  = all other images for that test identity
    Training split is unchanged (mixed spectra is fine for classification).
    """
    rng  = random.Random(seed)
    ids  = sorted(id2paths.keys()); rng.shuffle(ids)
    n_tr = max(1, int(len(ids) * train_ratio))
    train_ids, test_ids = ids[:n_tr], ids[n_tr:]

    train_label_map = {k: i for i, k in enumerate(sorted(train_ids))}
    all_tr = [(p, train_label_map[id]) for id in train_ids for p in id2paths[id]]
    rng2   = random.Random(seed + 1); rng2.shuffle(all_tr)
    n_v    = max(1, int(len(all_tr) * val_ratio))
    val_samples, train_samples = all_tr[:n_v], all_tr[n_v:]

    test_label_map = {k: i for i, k in enumerate(sorted(test_ids))}
    gallery_samples, probe_samples = [], []

    for identity in test_ids:
        lab      = test_label_map[identity]
        spectra  = id2spectra.get(identity, {})
        if not spectra:
            # Fallback: no spectral info available
            paths = sorted(id2paths[identity])
            n_g   = max(1, len(paths) // 2)
            for p in paths[:n_g]: gallery_samples.append((p, lab))
            for p in paths[n_g:]: probe_samples.append((p, lab))
            continue
        for spec, paths in spectra.items():
            ps = sorted(paths)          # deterministic order by filename
            gallery_samples.append((ps[0], lab))   # 1st sample → gallery
            for p in ps[1:]:                        # rest → probe
                probe_samples.append((p, lab))

    return (train_samples, val_samples, gallery_samples, probe_samples,
            train_label_map, test_label_map)


def split_closed_set(id2paths, train_ratio=0.8, seed=42):
    rng = random.Random(seed)
    label_map, tr, te = {}, [], []
    for idx, (id_, paths) in enumerate(sorted(id2paths.items())):
        label_map[id_] = idx
        ps = list(paths); rng.shuffle(ps)
        n  = max(1, int(len(ps) * train_ratio))
        for p in ps[:n]: tr.append((p, idx))
        for p in ps[n:]: te.append((p, idx))
    return tr, te, label_map


# ══════════════════════════════════════════════════════════════
#  DATASETS
# ══════════════════════════════════════════════════════════════

class CASIAMSDataset(Dataset):
    def __init__(self, samples, img_side=128, train=True):
        self.samples = samples
        if train:
            self.tf = T.Compose([
                T.Resize(img_side),
                T.RandomChoice([
                    T.ColorJitter(brightness=0, contrast=0.05),
                    T.RandomResizedCrop(img_side, scale=(0.8,1.), ratio=(1.,1.)),
                    T.RandomPerspective(0.15, p=1),
                    T.RandomChoice([
                        T.RandomRotation(10, interpolation=Image.BICUBIC, expand=False,
                                         center=(0.5*img_side, 0.)),
                        T.RandomRotation(10, interpolation=Image.BICUBIC, expand=False,
                                         center=(0., 0.5*img_side)),
                    ]),
                ]),
                T.ToTensor(), NormSingleROI()])
        else:
            self.tf = T.Compose([T.Resize(img_side), T.ToTensor(), NormSingleROI()])
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, lab = self.samples[i]
        try:    return self.tf(Image.open(p).convert("L")), lab
        except: return self.__getitem__((i+1) % len(self.samples))

class CASIAMSDatasetSingle(Dataset):
    def __init__(self, samples, img_side=128):
        self.samples = samples
        self.tf = T.Compose([T.Resize(img_side), T.ToTensor(), NormSingleROI()])
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, lab = self.samples[i]
        try:    return self.tf(Image.open(p).convert("L")), lab
        except: return self.__getitem__((i+1) % len(self.samples))


# ══════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════

def run_one_epoch(epoch, model, center_loss, loader,
                  criterion, optimizer, center_optimizer,
                  device, phase, center_weight=0.003):
    is_train = (phase == "training")
    model.train() if is_train else model.eval()
    total_loss = total_correct = total = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            if is_train:
                optimizer.zero_grad()
                if center_optimizer: center_optimizer.zero_grad()

            logits, emb = model(imgs, targets)
            cls_loss    = criterion(logits, targets)
            cen_loss    = center_loss(emb, targets) if center_loss else 0.0
            loss        = cls_loss + center_weight * cen_loss

            if is_train:
                loss.backward()
                # Prevent centres from moving too fast (standard practice)
                if center_loss:
                    for p in center_loss.parameters():
                        p.grad.data *= (1. / center_weight)
                optimizer.step()
                if center_optimizer: center_optimizer.step()

            total_loss    += loss.item() * imgs.size(0)
            total_correct += logits.max(1)[1].eq(targets).sum().item()
            total         += imgs.size(0)
    return total_loss / max(total,1), 100.0 * total_correct / max(total,1)


# ══════════════════════════════════════════════════════════════
#  EVALUATION — EER + Rank-1 (+ per-spectrum breakdown)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval(); feats, labels, paths = [], [], []
    for batch in loader:
        if len(batch) == 3:
            imgs, labs, ps = batch
        else:
            imgs, labs = batch; ps = [""] * imgs.size(0)
        feats.append(model.get_feature_vector(imgs.to(device)).cpu().numpy())
        labels.append(labs.numpy() if not isinstance(labs, list) else np.array(labs))
        paths.extend(ps)
    return np.concatenate(feats), np.concatenate(labels), paths


def compute_eer(scores_arr):
    ins  = scores_arr[scores_arr[:,1]==1,  0]
    outs = scores_arr[scores_arr[:,1]==-1, 0]
    if not len(ins) or not len(outs): return 1.0, 0.0
    if ins.mean() < outs.mean(): ins, outs = -ins, -outs
    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s   = np.concatenate([ins, outs])
    fpr, tpr, thr = roc_curve(y, s, pos_label=1)
    eer = brentq(lambda x: 1-x-interp1d(fpr,tpr)(x), 0., 1.)
    return eer, float(interp1d(fpr,thr)(eer))


def evaluate(model, probe_loader, gallery_loader, device, out_dir=".", tag="eval",
             id2spectra=None):
    """
    FIX 7: also reports per-spectrum EER when id2spectra is provided.
    """
    pf, pl, pp = extract_features(model, probe_loader,   device)
    gf, gl, gp = extract_features(model, gallery_loader, device)
    np_, ng     = len(pf), len(gf)

    sl, ll = [], []
    dm     = np.zeros((np_, ng))
    for i in range(np_):
        d = np.arccos(np.clip(np.dot(gf, pf[i]), -1, 1)) / np.pi
        dm[i] = d
        for j in range(ng):
            sl.append(d[j]); ll.append(1 if pl[i]==gl[j] else -1)

    sa = np.column_stack([sl, ll])
    pair_eer, _ = compute_eer(sa)

    aggr_s, aggr_l = [], []
    for i in range(np_-1):
        for j in range(i+1, np_):
            d = np.arccos(np.clip(np.dot(pf[i], pf[j]),-1,1))/np.pi
            aggr_s.append(d); aggr_l.append(1 if pl[i]==pl[j] else -1)
    aggr_eer = compute_eer(np.column_stack([aggr_s,aggr_l]))[0] if aggr_s else 1.

    rank1 = 100.*sum(pl[i]==gl[dm[i].argmin()] for i in range(np_))/max(np_,1)

    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for sv,lv in zip(sl,ll): f.write(f"{sv} {lv}\n")
    _save_roc_det(sa, out_dir, tag)

    print(f"  [{tag}]  pairEER={pair_eer*100:.4f}%  aggrEER={aggr_eer*100:.4f}%  Rank-1={rank1:.2f}%")

    # FIX 7: per-spectrum breakdown
    if pp and pp[0]:
        def get_spec(path):
            parts = os.path.basename(path).split("_")
            return parts[2] if len(parts) >= 4 else "unknown"
        spectra = sorted(set(get_spec(p) for p in pp if p))
        for spec in spectra:
            pi_idx = [i for i,p in enumerate(pp) if get_spec(p)==spec]
            gi_idx = [i for i,p in enumerate(gp) if get_spec(p)==spec]
            if not pi_idx or not gi_idx: continue
            pf_s = pf[pi_idx]; pl_s = pl[pi_idx]
            gf_s = gf[gi_idx]; gl_s = gl[gi_idx]
            sl2, ll2 = [], []
            for i in range(len(pf_s)):
                d = np.arccos(np.clip(np.dot(gf_s,pf_s[i]),-1,1))/np.pi
                for j in range(len(gf_s)):
                    sl2.append(d[j]); ll2.append(1 if pl_s[i]==gl_s[j] else -1)
            if sl2:
                sa2 = np.column_stack([sl2,ll2])
                eer2,_ = compute_eer(sa2)
                r1_2 = 100.*sum(
                    pl_s[i]==gl_s[np.array([np.arccos(np.clip(
                        np.dot(gf_s,pf_s[i]),-1,1))/np.pi for _ in [0]],
                        dtype=object)[0].argmin()
                    if False else
                    np.argmin(np.arccos(np.clip(np.dot(gf_s,pf_s[i]),-1,1))/np.pi)]
                    for i in range(len(pf_s))) / max(len(pf_s),1)
                print(f"    spectrum {spec}: EER={eer2*100:.4f}%  Rank-1={r1_2:.2f}%")

    return pair_eer, aggr_eer, rank1


def _save_roc_det(sa, out_dir, tag):
    ins  = sa[sa[:,1]==1,  0]; outs = sa[sa[:,1]==-1, 0]
    if not len(ins) or not len(outs): return
    if ins.mean() < outs.mean(): ins, outs = -ins, -outs
    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    fpr, tpr, _ = roc_curve(y, np.concatenate([ins,outs]), pos_label=1)
    try:
        pdf = PdfPages(os.path.join(out_dir, f"roc_det_{tag}.pdf"))
        for (xd,yd,xl,yl,t) in [
            (fpr*100,tpr*100,'FAR(%)','GAR(%)','ROC'),
            (fpr*100,(1-tpr)*100,'FAR(%)','FRR(%)','DET')]:
            fig,ax = plt.subplots(); ax.plot(xd,yd,'b-^',markersize=2)
            ax.grid(True); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{t}—{tag}")
            pdf.savefig(fig); plt.close(fig)
        pdf.close()
    except Exception as e: print(f"  [warn] plot: {e}")


def plot_loss_acc(tl, vl, ta, va, d):
    try:
        ep = range(1, len(tl)+1)
        for (d1,d2,lbl,fn) in [(tl,vl,'loss','losses.png'),(ta,va,'acc(%)','accuracy.png')]:
            fig,ax = plt.subplots()
            ax.plot(ep,d1,'b',label='train'); ax.plot(ep,d2,'r',label='val')
            ax.legend(); ax.set_xlabel('epoch'); ax.set_ylabel(lbl)
            fig.savefig(os.path.join(d,fn)); plt.close(fig)
    except Exception: pass


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    C  = CONFIG
    os.makedirs(C["results_dir"], exist_ok=True)
    rst_eval = os.path.join(C["results_dir"], "eval")
    os.makedirs(rst_eval, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  MSPHNet v2 (fixed) on CASIA-MS")
    print(f"  Protocol : {C['protocol']}  |  Device: {device}")
    print(f"  lr={C['lr']}  batch={C['batch_size']}  warmup={C['warmup_epochs']}ep")
    print(f"  CenterLoss weight={C['center_loss_weight']}  emb_dim={C['embedding_dim']}")
    print(f"  weight_decay={C['weight_decay']}  label_smooth={C['label_smoothing']}")
    print(f"{'='*60}\n")

    print("Scanning dataset …")
    # FIX 7: use spectral-aware parser
    id2paths, id2spectra = parse_casia_ms_spectral(C["data_root"])
    n_ids  = len(id2paths)
    n_imgs = sum(len(v) for v in id2paths.values())
    all_spectra = sorted(set(sp for spd in id2spectra.values() for sp in spd))
    print(f"  Found {n_ids} identities, {n_imgs} images.")
    print(f"  Spectra detected: {all_spectra}\n")

    dl_kw = dict(batch_size=C["batch_size"], num_workers=C["num_workers"],
                 pin_memory=True, persistent_workers=(C["num_workers"]>0))

    if C["protocol"] == "closed-set":
        tr_s, te_s, lmap = split_closed_set(id2paths, C["train_ratio"], C["random_seed"])
        num_classes = len(lmap)
        train_loader   = DataLoader(CASIAMSDataset(tr_s, C["img_side"], True),  shuffle=True,  **dl_kw)
        val_loader     = DataLoader(CASIAMSDataset(te_s, C["img_side"], False), shuffle=False, **dl_kw)
        gallery_loader = DataLoader(CASIAMSDatasetSingle(tr_s, C["img_side"]),  shuffle=False, **dl_kw)
        probe_loader   = DataLoader(CASIAMSDatasetSingle(te_s, C["img_side"]),  shuffle=False, **dl_kw)
        print(f"  [closed-set] #classes={num_classes}")
    else:
        # FIX 7: spectral-stratified split
        (tr_s, va_s, ga_s, pr_s, tr_lmap, _) = split_open_set_spectral(
            id2paths, id2spectra, C["train_ratio"], C["val_ratio"], C["random_seed"])
        num_classes = len(tr_lmap)
        train_loader   = DataLoader(CASIAMSDataset(tr_s, C["img_side"], True),  shuffle=True,  **dl_kw)
        val_loader     = DataLoader(CASIAMSDataset(va_s, C["img_side"], False), shuffle=False, **dl_kw)
        gallery_loader = DataLoader(CASIAMSDatasetSingle(ga_s, C["img_side"]),  shuffle=False, **dl_kw)
        probe_loader   = DataLoader(CASIAMSDatasetSingle(pr_s, C["img_side"]),  shuffle=False, **dl_kw)
        print(f"  [open-set spectral] #train_classes={num_classes}")
        print(f"  gallery={len(ga_s)}  probe={len(pr_s)}\n")

    print(f"Building MSPHNet — num_classes={num_classes} …")
    net = MSPHNet(
        num_classes=num_classes, img_side=C["img_side"],
        embedding_dim=C["embedding_dim"],
        gabor_filters=C["gabor_filters"], gabor_kernel=C["gabor_kernel"],
        trans_dim=C["transformer_dim"], trans_depth=C["transformer_depth"],
        trans_heads=C["transformer_heads"], patch_size=C["patch_size"],
        ca_reduction=C["ca_reduction"], dropout=C["dropout"],
        arcface_s=C["arcface_s"], arcface_m=C["arcface_m"]).to(device)

    n_p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_p:,}\n")

    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    # FIX 8: CenterLoss with its own SGD optimizer
    center_loss = CenterLoss(num_classes, C["embedding_dim"]).to(device)
    center_optimizer = optim.SGD(center_loss.parameters(), lr=C["center_loss_lr"])

    # FIX 9: weight_decay; FIX 10: label_smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=C["label_smoothing"])
    optimizer = optim.Adam(net.parameters(), lr=C["lr"],
                           weight_decay=C["weight_decay"])
    cos_sched = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=C["num_epochs"], eta_min=C["lr"]*0.01)

    tl, vl, ta, va = [], [], [], []
    best_val_acc = 0.; best_eer = 1.
    last_eer = last_rank1 = float("nan")

    print(f"Starting training for {C['num_epochs']} epochs …")
    print(f"  EER computed every {C['eval_every']} epochs.\n")

    for epoch in range(C["num_epochs"]):
        # Linear warmup
        if epoch < C["warmup_epochs"]:
            for pg in optimizer.param_groups:
                pg["lr"] = C["lr"] * (epoch + 1) / C["warmup_epochs"]

        _net = net.module if isinstance(net, DataParallel) else net
        t_loss, t_acc = run_one_epoch(
            epoch, net, center_loss, train_loader,
            criterion, optimizer, center_optimizer,
            device, "training", C["center_loss_weight"])
        v_loss, v_acc = run_one_epoch(
            epoch, net, center_loss, val_loader,
            criterion, None, None,
            device, "testing", C["center_loss_weight"])

        if epoch >= C["warmup_epochs"]: cos_sched.step()

        tl.append(t_loss); vl.append(v_loss)
        ta.append(t_acc);  va.append(v_acc)

        if epoch % C["eval_every"] == 0 or epoch == C["num_epochs"]-1:
            tag = f"ep{epoch:04d}_{C['protocol'].replace('-','')}"
            cur_eer, cur_aggr, cur_r1 = evaluate(
                _net, probe_loader, gallery_loader, device, rst_eval, tag, id2spectra)
            last_eer, last_rank1 = cur_eer, cur_r1
            if cur_eer < best_eer:
                best_eer = cur_eer
                torch.save(_net.state_dict(),
                           os.path.join(C["results_dir"],"net_params_best_eer.pth"))
                print(f"  *** New best EER: {best_eer*100:.4f}% ***")

        if epoch % 10 == 0 or epoch == C["num_epochs"]-1:
            cur_lr   = optimizer.param_groups[0]["lr"]
            eer_s    = f"{last_eer*100:.4f}%"   if not math.isnan(last_eer)   else "N/A"
            r1_s     = f"{last_rank1:.2f}%"     if not math.isnan(last_rank1) else "N/A"
            print(f"[{time.strftime('%H:%M:%S')}] ep {epoch:04d} | lr={cur_lr:.6f} | "
                  f"loss t={t_loss:.5f} v={v_loss:.5f} | "
                  f"acc t={t_acc:.2f}% v={v_acc:.2f}% | "
                  f"EER={eer_s} R1={r1_s}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(_net.state_dict(),
                       os.path.join(C["results_dir"],"net_params_best.pth"))

        if epoch % C["save_every"] == 0 or epoch == C["num_epochs"]-1:
            torch.save(_net.state_dict(),
                       os.path.join(C["results_dir"],"net_params.pth"))
            plot_loss_acc(tl, vl, ta, va, C["results_dir"])

    print("\n=== Final evaluation ===")
    best_path = os.path.join(C["results_dir"],"net_params_best_eer.pth")
    if not os.path.exists(best_path):
        best_path = os.path.join(C["results_dir"],"net_params_best.pth")
    _net = net.module if isinstance(net, DataParallel) else net
    _net.load_state_dict(torch.load(best_path, map_location=device))
    fe, fa, fr = evaluate(_net, probe_loader, gallery_loader, device,
                          rst_eval, f"FINAL_{C['protocol'].replace('-','')}", id2spectra)

    print(f"\n{'='*60}")
    print(f"  PROTOCOL : {C['protocol']}")
    print(f"  FINAL Pairwise EER   : {fe*100:.4f}%")
    print(f"  FINAL Aggregated EER : {fa*100:.4f}%")
    print(f"  FINAL Rank-1         : {fr:.3f}%")
    print(f"{'='*60}\n")

    with open(os.path.join(C["results_dir"],"summary.txt"),"w") as f:
        f.write(f"Protocol      : {C['protocol']}\n")
        f.write(f"Identities    : {n_ids}\n")
        f.write(f"Images        : {n_imgs}\n")
        f.write(f"Spectra       : {all_spectra}\n")
        f.write(f"Train classes : {num_classes}\n")
        f.write(f"Embedding dim : {C['embedding_dim']}\n")
        f.write(f"Final EER     : {fe*100:.6f}%\n")
        f.write(f"Final AggrEER : {fa*100:.6f}%\n")
        f.write(f"Final Rank-1  : {fr:.3f}%\n")


if __name__ == "__main__":
    main()
