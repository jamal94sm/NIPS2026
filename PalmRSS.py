"""
PalmRSS — Single Source Domain Generalization for Palm Biometrics
Paper: Jia et al., Pattern Recognition 2025
https://doi.org/10.1016/j.patcog.2025.111620

Cross-spectral experiment on CASIA-MS dataset.

Filename format : {id}_{hand}_{spectrum}_{iter}.jpg
                  e.g.  018_r_WHT_02.jpg

CASIA-MS collection protocol (Section 4.1):
  100 subjects × 6 spectra × 2 sessions × 6 samples = 7200 images
  Session 1: iterations 01-06  (first collection time)
  Session 2: iterations 07-12  (second collection time, ~1 month later)

Dataset partitioning (Section 4.2):
  D1 = session 1 images of SOURCE_SPECTRUM  (iter 01-06)
  D2 = session 2 images of SOURCE_SPECTRUM  (iter 07-12)
  "D1 and D2 correspond to different data collection times" — equal 1:1 split
  Dt = all images of TARGET_SPECTRA         (unseen during training)

Evaluation gallery (paper protocol):
  Gallery = D0 = D1 ∪ D2  (full source spectrum, both sessions enrolled)
  Probe   = Dt             (unseen target spectra)

Image alignment (Eq. 1-3):
  F1     = FAT(x_D1, x_D2)   Fourier Alignment Transform
  F2     = HM(x_D1, x_D2)    Histogram Matching
  F_cat  = Concat(F1, F2)     2-channel input to CCNet

Loss (Eq. 13 / 15):
  L_hyb  = 0.8*L_ce + 0.1*L_con + 0.1*L_sim
  L_total = L_adv + 1.0 * L_hyb
"""

import os, math, time, copy
import numpy as np
from collections import defaultdict
from PIL import Image
from sklearn import metrics
from sklearn.metrics import auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS  — edit these lines only
# ============================================================

# --- Paths ---
DATA_PATH       = "/home/pai-ng/Jamal/CASIA-MS-ROI"
OUTPUT_DIR      = "./results_casia_ms"
GPU_ID          = "0"

# --- Cross-spectral experiment (Table 1 / Table 3) ---
# CASIA-MS spectra: 460, 700, 850 (selected per Section 4.1)
# Set SOURCE_SPECTRUM to one; TARGET_SPECTRA = the other two.
# Examples from Table 3:
#   460 → [700, 850]
#   700 → [460, 850]
#   850 → [460, 700]
SOURCE_SPECTRUM = "460"
TARGET_SPECTRA  = ["700"]

# --- Architecture ---
COM_WEIGHT      = 0.8
ARC_S           = 30.0
ARC_M           = 0.5
FC_DIM1         = 4096
FC_DIM2         = 2048
DROPOUT         = 0.5

# --- Loss weights (Eq. 13) ---
W_CE            = 0.8
W_CON           = 0.1
W_SIM           = 0.1
LAMBDA_HYB      = 1.0
TEMPERATURE     = 0.07
BASE_TEMP       = 0.07

# --- FDA (Eq. 5-6) ---
BETA            = 0.1

# --- Training ---
BATCH_SIZE      = 512
EPOCH_NUM       = 500
LR              = 0.001
LR_STEP         = 500
LR_GAMMA        = 0.8
IMSIDE          = 128

# --- Logging ---
PRINT_INTERVAL  = 10     # print train + target metrics every N epochs
SAVE_INTERVAL   = 50    # save checkpoint every N epochs

# ============================================================
# (nothing to edit below this line)
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {device}")


# ============================================================
# 1.  DATA SPLITTING  (session-based, Section 4.2)
#
# CASIA-MS: 2 sessions × 6 samples per identity per spectrum
#   Session 1 (D1): iterations 01-06  — first collection time
#   Session 2 (D2): iterations 07-12  — second collection time (~1 month later)
#
# We determine session by sorting all iterations for each
# (identity, spectrum) group and splitting at the midpoint.
# This is robust regardless of exact iteration numbering.
# ============================================================

def parse_filename(fname):
    """Parse {id}_{hand}_{spectrum}_{iter}.jpg"""
    stem  = os.path.splitext(fname)[0]
    parts = stem.split("_")
    if len(parts) < 4 or not parts[0].isdigit():
        return None
    return dict(id=parts[0], hand=parts[1],
                spectrum=parts[2], iteration=parts[3])


def build_splits(data_root, source_spectrum, target_spectra):
    """
    Session-based split for CASIA-MS (Section 4.2).

    For each (identity, spectrum) group:
      - Sort all images by iteration number
      - First half  → Session 1 → D1  (earlier collection time)
      - Second half → Session 2 → D2  (later  collection time)

    This matches: "D1 and D2 correspond to different data collection times"
    with equal 1:1 split.

    Returns
    -------
    d1_list     : [(path, label), ...]  session-1 source images
    d2_list     : [(path, label), ...]  session-2 source images
    dt_list     : [(path, label), ...]  all target-spectrum images
    num_classes : int
    """
    exts  = {".jpg", ".jpeg", ".png"}
    files = sorted(f for f in os.listdir(data_root)
                   if os.path.splitext(f)[1].lower() in exts)

    # Group by (identity_key, spectrum)
    src_groups = defaultdict(list)   # source spectrum only
    tgt_groups = defaultdict(list)   # target spectra

    for f in files:
        m = parse_filename(f)
        if m is None:
            continue
        key_id = f"{m['id']}_{m['hand']}"
        path   = os.path.join(data_root, f)
        if m['spectrum'] == source_spectrum:
            src_groups[key_id].append((m['iteration'], path))
        elif m['spectrum'] in target_spectra:
            tgt_groups[(key_id, m['spectrum'])].append(
                (m['iteration'], path))

    # Assign integer labels from source identities
    sorted_ids  = sorted(src_groups.keys())
    label_map   = {k: i for i, k in enumerate(sorted_ids)}
    num_classes = len(sorted_ids)

    d1_list, d2_list = [], []
    session_summary  = []   # for verification printout

    for key_id in sorted_ids:
        lbl = label_map[key_id]
        # Sort by iteration string (01, 02, ... 12)
        items = sorted(src_groups[key_id], key=lambda x: x[0])
        n     = len(items)
        half  = n // 2   # session boundary: first half = session 1

        sess1 = [p for _, p in items[:half]]
        sess2 = [p for _, p in items[half:]]

        for p in sess1:
            d1_list.append((p, lbl))
        for p in sess2:
            d2_list.append((p, lbl))

        # Record first identity for verification
        if len(session_summary) < 3:
            iters = [it for it, _ in items]
            session_summary.append(
                f"  {key_id}: session1={iters[:half]}  "
                f"session2={iters[half:]}")

    # Target domain: assign labels by identity (open-set)
    tgt_label_map = {}
    dt_list       = []
    for (key_id, spec), items in tgt_groups.items():
        if key_id not in tgt_label_map:
            tgt_label_map[key_id] = len(tgt_label_map)
        lbl = tgt_label_map[key_id]
        for _, p in sorted(items, key=lambda x: x[0]):
            dt_list.append((p, lbl))

    print(f"\n  Source spectrum  : {source_spectrum}")
    print(f"  Target spectra   : {target_spectra}")
    print(f"  Source identities: {num_classes}")
    print(f"  D1 (session 1)   : {len(d1_list)} images")
    print(f"  D2 (session 2)   : {len(d2_list)} images")
    print(f"  Dt (target)      : {len(dt_list)} images "
          f"({len(tgt_label_map)} identities)")
    print(f"\n  Session split verification (first 3 identities):")
    for s in session_summary:
        print(s)

    if len(d1_list) == 0:
        print(f"\n  ERROR: No images found for source spectrum "
              f"'{source_spectrum}'.")
        exts2 = {".jpg", ".jpeg", ".png"}
        all_files = [f for f in os.listdir(data_root)
                     if os.path.splitext(f)[1].lower() in exts2]
        found_spectra = set()
        for f in all_files[:300]:
            m = parse_filename(f)
            if m:
                found_spectra.add(m['spectrum'])
        print(f"  Available spectra in {data_root}:")
        print(f"  {sorted(found_spectra)}")

    return d1_list, d2_list, dt_list, num_classes


def write_txt(lst, path):
    with open(path, "w") as f:
        for img_path, label in lst:
            f.write(f"{img_path} {label}\n")


# ============================================================
# 2.  DATASET
# ============================================================

class NormSingleROI:
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        c, h, w = tensor.size()
        flat = tensor.view(c, h * w)
        idx  = flat > 0
        t    = flat[idx]
        if t.numel() > 1:
            flat[idx] = (t - t.mean()) / (t.std() + 1e-6)
        tensor = flat.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, self.outchannels, dim=0)
        return tensor


class PalmDataset(Dataset):
    def __init__(self, samples, train=True, imside=IMSIDE):
        self.samples = samples
        self.train   = train
        self.labels  = [s[1] for s in samples]

        if train:
            self.tf = T.Compose([
                T.Resize(imside),
                T.RandomChoice([
                    T.ColorJitter(brightness=0, contrast=0.05),
                    T.RandomResizedCrop(imside, scale=(0.8, 1.0),
                                        ratio=(1., 1.)),
                    T.RandomPerspective(distortion_scale=0.15, p=1),
                    T.RandomChoice([
                        T.RandomRotation(10, expand=False,
                                         center=(int(0.5 * imside), 0)),
                        T.RandomRotation(10, expand=False,
                                         center=(0, int(0.5 * imside))),
                    ]),
                ]),
                T.ToTensor(),
                NormSingleROI(1),
            ])
        else:
            self.tf = T.Compose([
                T.Resize(imside),
                T.ToTensor(),
                NormSingleROI(1),
            ])

    def __len__(self):
        return len(self.samples)

    def _load(self, idx):
        path, label = self.samples[idx]
        return self.tf(Image.open(path).convert("L")), label

    def __getitem__(self, idx):
        img1, label = self._load(idx)
        same = [i for i, l in enumerate(self.labels) if l == label]
        idx2 = idx
        if self.train and len(same) > 1:
            while idx2 == idx:
                idx2 = int(np.random.choice(same))
        img2, _ = self._load(idx2)
        return (img1, img2), label


# ============================================================
# 3.  MODEL  (CCNet — ccnet_2.py, 2-channel input)
# ============================================================

class GaborConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, ksize, stride=1,
                 padding=0, init_ratio=1.):
        super().__init__()
        r = init_ratio
        self.ch_in = ch_in; self.ch_out = ch_out
        self.ksize = ksize; self.stride = stride
        self.padding = padding; self.kernel = None
        self.gamma = nn.Parameter(torch.FloatTensor([2.0]))
        self.sigma = nn.Parameter(torch.FloatTensor([9.2 * r]))
        self.theta = nn.Parameter(
            torch.arange(ch_out).float() * math.pi / ch_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([0.057 / r]))
        self.psi = nn.Parameter(torch.FloatTensor([0.0]),
                                requires_grad=False)

    def _build_bank(self):
        xm  = self.ksize // 2
        rng = torch.arange(-xm, xm + 1).float()
        y   = rng.view(1, -1).repeat(self.ch_out, self.ch_in,
                                      self.ksize, 1)
        x   = rng.view(-1, 1).repeat(self.ch_out, self.ch_in,
                                      1, self.ksize)
        x   = x.to(self.sigma.device)
        y   = y.to(self.sigma.device)
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
        self.kernel = self._build_bank()
        return F.conv2d(x, self.kernel, stride=self.stride,
                        padding=self.padding)


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
    def __init__(self, ch_in, n_comp, ksize, weight,
                 init_ratio=1., o1=32):
        super().__init__()
        nc2 = n_comp * 2; nc4 = n_comp * 4
        self.g1 = GaborConv2d(ch_in, n_comp, ksize, 2,
                               ksize // 2, init_ratio)
        self.g2 = GaborConv2d(nc2, nc2, ksize, 2,
                               ksize // 2, init_ratio)
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
        else:   # ksize == 7
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
        return (self.wc * self.sm_c(x)
                + self.ws * (self.sm_h(x) + self.sm_w(x)))

    def forward(self, x):
        f  = torch.cat([self.g1(x), self.c1b(self.c1a(x))], dim=1)
        x1 = self.pool(self.ppu1(self.se1(self._compete(f))))
        f  = torch.cat([self.g2(f), self.c2b(self.c2a(f))], dim=1)
        x2 = self.pool(self.ppu2(self.se2(self._compete(f))))
        return torch.cat([x1.flatten(1), x2.flatten(1)], dim=1)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_f, out_f, s=ARC_S, m=ARC_M):
        super().__init__()
        self.s     = s
        self.w     = Parameter(torch.FloatTensor(out_f, in_f))
        nn.init.xavier_uniform_(self.w)
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        cos = F.linear(F.normalize(x), F.normalize(self.w))
        if self.training and label is not None:
            sin = torch.sqrt((1. - cos ** 2).clamp(0., 1.))
            phi = cos * self.cos_m - sin * self.sin_m
            phi = torch.where(cos > self.th, phi, cos - self.mm)
            oh  = torch.zeros_like(cos).scatter_(
                1, label.view(-1, 1).long(), 1)
            return ((oh * phi) + ((1. - oh) * cos)) * self.s
        return self.s * cos

    def cosine_scores(self, x):
        """No-margin cosine logits — used for monitoring accuracy only"""
        return F.linear(F.normalize(x), F.normalize(self.w)) * self.s


class CCNet(nn.Module):
    """
    CCNet (ccnet_2.py):
      CB1 ksize=35 / CB2 ksize=17 / CB3 ksize=7
      2-layer FC + Dropout + ArcFace
      Input: 2-channel [FAT | HM]
    """
    def __init__(self, num_classes, weight=COM_WEIGHT):
        super().__init__()
        self.cb1  = CompetitiveBlock(2,  9, 35, weight, init_ratio=1.00)
        self.cb2  = CompetitiveBlock(2, 36, 17, weight, init_ratio=0.50)
        self.cb3  = CompetitiveBlock(2,  9,  7, weight, init_ratio=0.25)
        self.fc   = nn.Linear(13152, FC_DIM1)
        self.fc1  = nn.Linear(FC_DIM1, FC_DIM2)
        self.drop = nn.Dropout(DROPOUT)
        self.arc  = ArcMarginProduct(FC_DIM2, num_classes,
                                     s=ARC_S, m=ARC_M)

    def _backbone(self, x):
        return torch.cat(
            [self.cb1(x), self.cb2(x), self.cb3(x)], dim=1)

    def forward(self, x, y=None):
        h1  = self.fc(self._backbone(x))
        h2  = self.fc1(h1)
        fe  = torch.cat([h1, h2], dim=1)
        out = self.arc(self.drop(h2), y)
        return out, F.normalize(fe, dim=-1)

    def cosine_classify(self, x):
        """No-margin cosine logits — real training accuracy metric"""
        h2 = self.fc1(self.fc(self._backbone(x)))
        return self.arc.cosine_scores(self.drop(h2))

    def getFeatureCode(self, x):
        return F.normalize(
            self.fc1(self.fc(self._backbone(x))), dim=-1)


# ============================================================
# 4.  DOMAIN DISCRIMINATOR  (L_adv, Eq. 14)
# ============================================================

class DomainDiscriminator(nn.Module):
    """Separates D1 features (label=1) from D2 features (label=0)."""
    def __init__(self, input_dim=FC_DIM1 + FC_DIM2, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 5.  LOSSES
# ============================================================

class SupConLoss(nn.Module):
    """Supervised contrastive loss (Eq. 11)"""
    def __init__(self, temperature=TEMPERATURE,
                 base_temperature=BASE_TEMP):
        super().__init__()
        self.T = temperature; self.base = base_temperature

    def forward(self, features, labels):
        dev  = features.device
        bsz  = features.shape[0]; n = features.shape[1]
        mask = torch.eq(
            labels.view(-1, 1), labels.view(1, -1)).float().to(dev)
        contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
        dot      = torch.div(
            torch.matmul(contrast, contrast.T), self.T)
        lm, _    = torch.max(dot, dim=1, keepdim=True)
        logits   = dot - lm.detach()
        mask     = mask.repeat(n, n)
        lmask    = 1. - torch.eye(bsz * n, device=dev)
        mask     = mask * lmask
        exp_log  = torch.exp(logits) * lmask
        log_prob = logits - torch.log(
            exp_log.sum(1, keepdim=True) + 1e-9)
        denom    = mask.sum(1).clamp(min=1.)
        return (-(self.T / self.base)
                * (mask * log_prob).sum(1) / denom).mean()


def feature_similarity_loss(v, v_aug):
    """L_sim = mean cosine distance (Eq. 12)"""
    return (1. - F.cosine_similarity(v, v_aug, dim=-1)).mean()


def adversarial_loss(disc, v1, v2):
    """L_adv (Eq. 14): D1 → label 1, D2 → label 0"""
    bce  = nn.BCEWithLogitsLoss()
    lbl1 = torch.ones (v1.size(0), 1, device=v1.device)
    lbl2 = torch.zeros(v2.size(0), 1, device=v2.device)
    return bce(disc(v1), lbl1) + bce(disc(v2), lbl2)


# ============================================================
# 6.  IMAGE ALIGNMENT  (Section 3.2)
# ============================================================

def _hist_match_np(src: np.ndarray,
                   tgt: np.ndarray) -> np.ndarray:
    """CDF-based histogram matching (Eq. 7-9), NumPy-only."""
    matched = np.empty_like(src)
    for c in range(src.shape[2]):
        s = src[..., c].ravel().astype(np.float64)
        t = tgt[..., c].ravel().astype(np.float64)
        s_min, s_max = s.min(), s.max()
        t_min, t_max = t.min(), t.max()
        if s_max == s_min or t_max == t_min:
            matched[..., c] = src[..., c]; continue
        s_n = (s - s_min) / (s_max - s_min)
        t_n = (t - t_min) / (t_max - t_min)
        bins = 256
        s_cnt, _ = np.histogram(s_n, bins=bins, range=(0., 1.))
        t_cnt, _ = np.histogram(t_n, bins=bins, range=(0., 1.))
        s_cdf = np.cumsum(s_cnt).astype(np.float64)
        s_cdf /= s_cdf[-1]
        t_cdf = np.cumsum(t_cnt).astype(np.float64)
        t_cdf /= t_cdf[-1]
        edges   = np.linspace(0., 1., bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.
        t_idx   = np.searchsorted(t_cdf, s_cdf).clip(0, bins - 1)
        lut     = centers[t_idx] * (s_max - s_min) + s_min
        pix_bin = np.searchsorted(edges[1:], s_n).clip(0, bins - 1)
        matched[..., c] = (lut[pix_bin]
                           .reshape(src.shape[:2])
                           .astype(np.float32))
    return matched.astype(np.float32)


def hm_batch(src_batch, tgt_batch):
    """F2 = HM(x_D1, x_D2) — CPU tensors [B,C,H,W]"""
    rows = []
    for s, t in zip(src_batch, tgt_batch):
        s_np = s.permute(1, 2, 0).numpy()
        t_np = t.permute(1, 2, 0).numpy()
        rows.append(
            torch.from_numpy(
                _hist_match_np(s_np, t_np)).permute(2, 0, 1))
    return torch.stack(rows).float()


def fat_batch(src, tgt, beta=BETA):
    """F1 = FAT(x_D1, x_D2) — Eq. 6, CPU tensors [B,C,H,W]"""
    fs  = torch.fft.rfft2(src, dim=(-2, -1))
    ft  = torch.fft.rfft2(tgt, dim=(-2, -1))
    as_ = torch.abs(fs).clone()
    ps  = torch.angle(fs)
    at  = torch.abs(ft)
    _, _, h, w2 = as_.shape
    bh = int(np.floor(beta * h))
    bw = int(np.floor(beta * w2 * 2))
    b  = min(bh, bw)
    if b > 0:
        as_[:, :, :b,      :b] = at[:, :, :b,      :b]
        as_[:, :, h-b+1:h, :b] = at[:, :, h-b+1:h, :b]
    rec = torch.fft.irfft2(
        torch.complex(torch.cos(ps) * as_,
                      torch.sin(ps) * as_),
        dim=(-2, -1), s=[h, w2 * 2])
    return rec[..., :src.shape[-2], :src.shape[-1]]


def make_2ch(src, tgt):
    """F_cat = Concat(FAT, HM) — returns [B,2,H,W] CPU tensor"""
    return torch.cat([fat_batch(src, tgt), hm_batch(src, tgt)], dim=1)


def make_2ch_identity(x):
    """Test-time: duplicate single channel → [B,2,H,W]"""
    return torch.cat([x, x], dim=1)


# ============================================================
# 7.  EVALUATION
#
# Gallery = D0 = D1 ∪ D2  (full source domain, both sessions)
# Probe   = Dt             (unseen target spectra)
#
# This matches the paper's protocol. Using only D1 as gallery
# gives artificially lower EER (fewer genuine pairs) and lower
# Rank-1 (fewer enrolled templates).
# ============================================================

def compute_eer(ins, outs):
    if ins.mean() < outs.mean():
        ins, outs = -ins, -outs
    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    sc  = np.concatenate([ins, outs])
    fpr, tpr, _ = metrics.roc_curve(y, sc, pos_label=1)
    roc_auc     = auc(fpr, tpr)
    eer = brentq(
        lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100., roc_auc


def extract_features(model, loader):
    model.eval()
    feats, ids = [], []
    with torch.no_grad():
        for (d, _), target in loader:
            codes = model.getFeatureCode(
                make_2ch_identity(d).to(device))
            feats.append(codes.cpu().numpy())
            ids.append(target.numpy())
    return np.concatenate(feats), np.concatenate(ids)


def quick_eval(model, d1_loader, d2_loader, probe_loader):
    """
    Gallery = D0 = D1 ∪ D2  (full source domain, both sessions)
    Probe   = Dt             (unseen target spectra)

    Using D1 ∪ D2 as gallery matches the paper:
      - More genuine pairs  → EER harder and more realistic
      - More templates      → Rank-1 higher and more realistic
    """
    ft_d1, id_d1 = extract_features(model, d1_loader)
    ft_d2, id_d2 = extract_features(model, d2_loader)

    # D0 = D1 ∪ D2  (full source domain gallery)
    ft_g = np.concatenate([ft_d1, ft_d2], axis=0)
    id_g = np.concatenate([id_d1, id_d2], axis=0)

    ft_p, id_p = extract_features(model, probe_loader)

    # Vectorised cosine similarity [M_probe × N_gallery]
    sim = ft_p @ ft_g.T

    # Rank-1
    preds = id_g[sim.argmax(axis=1)]
    rank1 = 100. * (preds == id_p).mean()

    # EER — vectorised pairwise distance matrix
    dis   = np.arccos(np.clip(sim, -1., 1.)) / np.pi
    n_g   = len(id_g)
    n_p   = len(id_p)
    # label: 1=genuine, -1=impostor  for every (probe_i, gallery_j) pair
    gal_ids_tiled  = np.tile(id_g, n_p)          # [n_p * n_g]
    probe_ids_rep  = np.repeat(id_p, n_g)         # [n_p * n_g]
    l     = np.where(gal_ids_tiled == probe_ids_rep, 1, -1)
    s     = dis.ravel()
    ins   = 1. - s[l ==  1]
    outs  = 1. - s[l == -1]
    eer, roc_auc = compute_eer(ins, outs)
    return rank1, eer, roc_auc


def full_evaluate(model, d1_loader, d2_loader,
                  probe_loader, tag, out_dir):
    """Full evaluation with file output — used at final checkpoints."""
    print(f"\n{'='*65}")
    print(f"  Full Evaluation : {tag}")
    print(f"  Gallery = D0 = D1 ∪ D2  |  Probe = Dt (unseen spectra)")
    print(f"{'='*65}")
    rank1, eer, roc_auc = quick_eval(
        model, d1_loader, d2_loader, probe_loader)
    print(f"  Rank-1  : {rank1:.3f}%")
    print(f"  EER     : {eer:.4f}%")
    print(f"  AUC     : {roc_auc:.6f}")
    print(f"{'='*65}\n")
    ev_dir = os.path.join(out_dir, tag)
    os.makedirs(ev_dir, exist_ok=True)
    with open(os.path.join(ev_dir, "results.txt"), "w") as f:
        f.write(f"EER    : {eer:.4f}%\n")
        f.write(f"Rank-1 : {rank1:.3f}%\n")
        f.write(f"AUC    : {roc_auc:.6f}\n")
    return eer, rank1


# ============================================================
# 8.  TRAINING LOOP  (Eq. 13, 14, 15)
# ============================================================

def fit_epoch(epoch, model, disc,
              d1_loader, d2_iter_ref,
              criterion, con_crit,
              opt_model, opt_disc):
    """
    L_hyb  = W_CE*L_ce + W_CON*L_con + W_SIM*L_sim   (Eq. 13)
    L_total = L_adv + LAMBDA_HYB * L_hyb               (Eq. 15)

    arc_acc  — ArcFace-penalised (~0% early, expected by design)
    cos_acc  — cosine accuracy without margin (real progress metric)
    """
    model.train(); disc.train()
    run_loss = 0.; arc_corr = 0; cos_corr = 0; total = 0

    for (x_d1, x_d1_aug), y_d1 in d1_loader:
        try:
            (x_d2, _), _ = next(d2_iter_ref[0])
        except StopIteration:
            d2_iter_ref[0] = iter(d2_iter_ref[1])
            (x_d2, _), _  = next(d2_iter_ref[0])

        y_d1 = y_d1.to(device)

        data     = make_2ch(x_d1,     x_d2).to(device)
        data_aug = make_2ch(x_d1_aug, x_d2).to(device)
        data_d2  = make_2ch(x_d2,     x_d1).to(device)

        opt_model.zero_grad(); opt_disc.zero_grad()

        out1, fe1 = model(data,     y_d1)
        out2, fe2 = model(data_aug, y_d1)

        with torch.no_grad():
            _, fe_d2 = model(data_d2, None)

        l_ce  = criterion(out1, y_d1)
        l_con = con_crit(torch.stack([fe1, fe2], dim=1), y_d1)
        l_sim = feature_similarity_loss(fe1, fe2)
        l_hyb = W_CE * l_ce + W_CON * l_con + W_SIM * l_sim
        l_adv = adversarial_loss(
            disc, fe1.detach(), fe_d2.detach())
        loss  = l_adv + LAMBDA_HYB * l_hyb

        loss.backward()
        opt_model.step(); opt_disc.step()

        run_loss += loss.item() * y_d1.size(0)
        total    += y_d1.size(0)

        with torch.no_grad():
            arc_corr += out1.argmax(1).eq(y_d1).sum().item()
            model.eval()
            cos_corr += (model.cosine_classify(data)
                         .argmax(1).eq(y_d1).sum().item())
            model.train(); disc.train()

    return (run_loss / total,
            100. * arc_corr / total,
            100. * cos_corr / total)


# ============================================================
# 9.  MAIN
# ============================================================

def main():
    # ── 1. Session-based splits ──────────────────────────────
    print("\n[1] Building session-based splits ...")
    print(f"    Source: {SOURCE_SPECTRUM}  "
          f"Target: {TARGET_SPECTRA}")
    d1_list, d2_list, dt_list, num_classes = build_splits(
        DATA_PATH, SOURCE_SPECTRUM, TARGET_SPECTRA)

    if len(d1_list) == 0:
        return

    write_txt(d1_list, os.path.join(OUTPUT_DIR, "D1_session1.txt"))
    write_txt(d2_list, os.path.join(OUTPUT_DIR, "D2_session2.txt"))
    write_txt(dt_list, os.path.join(OUTPUT_DIR, "Dt_target.txt"))

    # ── 2. Dataloaders ───────────────────────────────────────
    print("\n[2] Building dataloaders ...")
    d1_ds = PalmDataset(d1_list, train=True)
    d2_ds = PalmDataset(d2_list, train=True)
    dt_ds = PalmDataset(dt_list, train=False)

    kw = dict(num_workers=4, pin_memory=True)
    d1_loader = DataLoader(d1_ds, batch_size=BATCH_SIZE,
                           shuffle=True, drop_last=True, **kw)
    d2_loader = DataLoader(d2_ds, batch_size=BATCH_SIZE,
                           shuffle=True, drop_last=True, **kw)
    dt_loader = DataLoader(dt_ds, batch_size=BATCH_SIZE,
                           shuffle=False, **kw)

    # ── 3. Model + discriminator ─────────────────────────────
    print(f"\n[3] Building CCNet (num_classes={num_classes}) "
          f"+ Discriminator ...")
    net      = CCNet(num_classes, COM_WEIGHT).to(device)
    best_net = CCNet(num_classes, COM_WEIGHT).to(device)
    disc     = DomainDiscriminator(FC_DIM1 + FC_DIM2, 1024).to(device)

    criterion = nn.CrossEntropyLoss()
    con_crit  = SupConLoss(TEMPERATURE, BASE_TEMP)
    opt_model = optim.Adam(net.parameters(), lr=LR)
    opt_disc  = optim.Adam(disc.parameters(), lr=LR)
    sched     = lr_scheduler.StepLR(
        opt_model, step_size=LR_STEP, gamma=LR_GAMMA)

    print(f"  Loss: L_adv + {LAMBDA_HYB} × "
          f"({W_CE}·L_ce + {W_CON}·L_con + {W_SIM}·L_sim)")
    print(f"  Gallery for eval: D0 = D1 ∪ D2  "
          f"(both sessions, paper protocol)")

    # ── 4. Training ──────────────────────────────────────────
    print("\n[4] Training ...")
    print("  arc_acc near 0% early = expected (ArcFace margin).")
    print("  cos_acc = real convergence.  Dt-EER = paper metric.\n")

    # Column header
    header = (f"{'Epoch':>6}  "
              f"{'Tr-Loss':>9}  "
              f"{'Tr-ArcAcc':>10}  "
              f"{'Tr-CosAcc':>10}  "
              f"{'Dt-Rank1':>9}  "
              f"{'Dt-EER':>8}  "
              f"{'Dt-AUC':>8}  "
              f"{'Time':>7}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    best_eer     = 100.
    best_cos_acc = 0.
    loss_hist, arc_hist, cos_hist = [], [], []
    eer_hist, rank1_hist          = [], []
    d2_iter_ref = [iter(d2_loader), d2_loader]

    log_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,tr_loss,tr_arc_acc,tr_cos_acc,"
                "dt_rank1,dt_eer,dt_auc\n")

    for epoch in range(EPOCH_NUM):
        t0 = time.time()

        loss, arc_acc, cos_acc = fit_epoch(
            epoch, net, disc, d1_loader, d2_iter_ref,
            criterion, con_crit, opt_model, opt_disc)
        sched.step()

        loss_hist.append(loss)
        arc_hist.append(arc_acc)
        cos_hist.append(cos_acc)

        if epoch % PRINT_INTERVAL == 0:
            # Evaluate with full D0 gallery (paper protocol)
            rank1, eer, roc_auc = quick_eval(
                net, d1_loader, d2_loader, dt_loader)
            eer_hist.append(eer)
            rank1_hist.append(rank1)

            elapsed = time.time() - t0
            print(f"{epoch:>6}  "
                  f"{loss:>9.5f}  "
                  f"{arc_acc:>9.2f}%  "
                  f"{cos_acc:>9.2f}%  "
                  f"{rank1:>8.3f}%  "
                  f"{eer:>7.4f}%  "
                  f"{roc_auc:>8.6f}  "
                  f"{elapsed:>5.1f}s  "
                  f"[{time.strftime('%H:%M:%S')}]")

            with open(log_path, "a") as f:
                f.write(f"{epoch},{loss:.6f},"
                        f"{arc_acc:.4f},{cos_acc:.4f},"
                        f"{rank1:.4f},{eer:.4f},{roc_auc:.6f}\n")

            if eer < best_eer:
                best_eer = eer
                torch.save(net.state_dict(), os.path.join(
                    OUTPUT_DIR, "best_eer_model.pth"))
                print(f"  >>> New best EER: {best_eer:.4f}%  "
                      f"Rank-1: {rank1:.3f}%  — saved")

        if cos_acc >= best_cos_acc:
            best_cos_acc = cos_acc
            torch.save(net.state_dict(), os.path.join(
                OUTPUT_DIR, "best_model.pth"))
            best_net.load_state_dict(
                copy.deepcopy(net.state_dict()))

        if epoch % SAVE_INTERVAL == 0 and epoch > 0:
            torch.save(net.state_dict(), os.path.join(
                OUTPUT_DIR, f"epoch_{epoch}.pth"))

    # ── 5. Final evaluation (full D0 gallery) ────────────────
    print(f"\n{sep}")
    print(f"Training complete.  Best EER: {best_eer:.4f}%")
    print(sep)
    full_evaluate(net,      d1_loader, d2_loader,
                  dt_loader, "final_last", OUTPUT_DIR)
    full_evaluate(best_net, d1_loader, d2_loader,
                  dt_loader, "final_best", OUTPUT_DIR)

    # ── 6. Training curves ───────────────────────────────────
    epochs_all  = list(range(EPOCH_NUM))
    epochs_eval = list(range(0, EPOCH_NUM, PRINT_INTERVAL))

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    axes[0, 0].plot(epochs_all, loss_hist)
    axes[0, 0].set_title("Train loss")
    axes[0, 0].set_xlabel("Epoch")

    axes[0, 1].plot(epochs_all, arc_hist,
                    label="arc_acc (penalised)", alpha=0.5)
    axes[0, 1].plot(epochs_all, cos_hist,
                    label="cos_acc (real)")
    axes[0, 1].set_title("Train accuracy (%)")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].legend()

    axes[0, 2].plot(epochs_eval[:len(eer_hist)],
                    eer_hist, color="red")
    axes[0, 2].set_title("Target EER % (gallery=D0)")
    axes[0, 2].set_xlabel("Epoch")

    axes[1, 0].plot(epochs_eval[:len(rank1_hist)],
                    rank1_hist, color="green")
    axes[1, 0].set_title("Target Rank-1 % (gallery=D0)")
    axes[1, 0].set_xlabel("Epoch")

    axes[1, 1].plot(epochs_all, loss_hist,
                    label="Train loss", alpha=0.7)
    ax2 = axes[1, 1].twinx()
    ax2.plot(epochs_eval[:len(eer_hist)], eer_hist,
             color="red", label="Target EER")
    axes[1, 1].set_title("Loss vs Target EER")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend(loc="upper left")
    ax2.legend(loc="upper right")

    axes[1, 2].plot(epochs_all, cos_hist,
                    label="Cos acc", alpha=0.7)
    ax3 = axes[1, 2].twinx()
    ax3.plot(epochs_eval[:len(rank1_hist)], rank1_hist,
             color="green", label="Target Rank-1")
    axes[1, 2].set_title("Cos acc vs Target Rank-1")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].legend(loc="upper left")
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"),
                dpi=120)
    plt.close()

    torch.save(net.state_dict(),
               os.path.join(OUTPUT_DIR, "last_model.pth"))
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
