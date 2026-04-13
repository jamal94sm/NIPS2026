"""
CO3Net on CASIA-MS Dataset
==================================================
Single-file implementation with closed-set and open-set protocols.
Faithfully preserves the official CO3Net architecture, SupConLoss,
paired-sample contrastive training, and NormSingleROI normalisation.

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

Architecture: CO3Net (unchanged from official repo)
  CompetitiveBlock (dual LGC + CoordAtt + soft-argmax + PPU)
  FC: 17328 → 4096 → 2048 + ArcFace output
  Contrastive feature: L2-normalised 6144-d (4096+2048 concat)
  Embedding (for matching): L2-normalised 2048-d
"""

# ==============================================================
#  CONFIG  — edit this block only
# ==============================================================
CONFIG = {
    "train_data"           : "Smartphone",   # "Smartphone" | "CASIA-MS"
    "test_data"            : "Smartphone",     # "Smartphone" | "CASIA-MS"
    "data_root"            : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "test_data_root"       : "/home/pai-ng/Jamal/smartphone_data",
    "train_subject_ratio"  : 0.70,   # 70% of subjects → train
    "test_gallery_ratio"   : 0.30,   # 30% of test-subject images → gallery
    "results_dir"          : "./rst_co3net_casia_ms",
    "img_side"             : 128,
    "batch_size"           : 256,    # ← was 1024
    "num_epochs"           : 200,    # ← was 150
    "lr"                   : 0.001,
    "lr_step"              : 30,     # ← was 500
    "lr_gamma"             : 0.6,    # ← was 0.8
    "dropout"              : 0.5,
    "arcface_s"            : 20.0,   # ← was 30.0
    "arcface_m"            : 0.30,   # ← was 0.50
    "ce_weight"            : 0.8,
    "con_weight"           : 0.2,
    "temperature"          : 0.07,
    "n_casia_subjects"     : 190,
    "n_casia_samples"      : 2776,
    "augment_factor"       : 3,      # ← increase to 3 to compensate small dataset
    "random_seed"          : 42,
    "save_every"           : 50,
    "eval_every"           : 50,
    "num_workers"          : 4,
    "eval_only"            : False,
    "protocol"             : "open-set"
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
#  SUPCONLOSS  (exact copy from CO3Net/loss.py)
# ══════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# ══════════════════════════════════════════════════════════════
#  MODEL  (exact copy of CO3Net/models/compnet.py — unchanged)
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    """Learnable Gabor Convolution (LGC) layer."""
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_ratio = init_ratio
        self.kernel = 0

        if init_ratio <= 0:
            init_ratio = 1.0
            print('input error!!!, require init_ratio > 0.0, using default...')

        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ  = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]), requires_grad=True)
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([self._FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def genGaborBank(self, kernel_size, channel_in, channel_out,
                     sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax
        ksize = xmax - xmin + 1

        y_0 = torch.arange(ymin, ymax + 1).float()
        x_0 = torch.arange(xmin, xmax + 1).float()

        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1)
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize)

        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        x_theta =  x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8 * sigma.view(-1, 1, 1, 1) ** 2)) \
            * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

        gb = gb - gb.mean(dim=[2, 3], keepdim=True)
        return gb

    def forward(self, x):
        kernel = self.genGaborBank(self.kernel_size, self.channel_in,
                                   self.channel_out, self.sigma, self.gamma,
                                   self.theta, self.f, self.psi)
        self.kernel = kernel
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate Attention module from CO3Net."""
    def __init__(self, inp, oup, reduction=1):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out


class CompetitiveBlock(nn.Module):
    """
    CO3Net Competitive Block:
    [LGC1 → CoordAtt → LGC2 → CoordAtt → soft-argmax → PPU]
    """
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 init_ratio=1, o1=32, o2=12):
        super(CompetitiveBlock, self).__init__()
        self.channel_in = channel_in
        self.n_competitor = n_competitor
        self.init_ratio = init_ratio

        # Dual LGC
        self.gabor_conv2d  = GaborConv2d(channel_in=channel_in,
                                         channel_out=n_competitor,
                                         kernel_size=ksize, stride=stride,
                                         padding=ksize // 2,
                                         init_ratio=init_ratio)
        self.gabor_conv2d2 = GaborConv2d(channel_in=n_competitor,
                                         channel_out=n_competitor,
                                         kernel_size=ksize, stride=1,
                                         padding=ksize // 2,
                                         init_ratio=init_ratio)

        # Coordinate Attention (replaces SE in CCNet)
        self.cooratt1 = CoordAtt(n_competitor, n_competitor)
        self.cooratt2 = CoordAtt(n_competitor, n_competitor)

        # soft-argmax
        self.a = nn.Parameter(torch.FloatTensor([1]))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.argmax = nn.Softmax(dim=1)

        # PPU
        self.conv1   = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.gabor_conv2d(x)
        x = self.cooratt1(x)
        x = self.gabor_conv2d2(x)
        x = self.cooratt2(x)

        x = (x - self.b) * self.a
        x = self.argmax(x)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        return x


class ArcMarginProduct(nn.Module):
    """ArcFace head — large margin arc distance."""
    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if self.training:
            assert label is not None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine   = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi    = cosine * self.cos_m - sine * self.sin_m

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


class co3net(nn.Module):
    """
    CO3Net = CB1 // CB2 // CB3  +  FC  +  Dropout  +  ArcFace
    https://ieeexplore.ieee.org/document/10124928
    """
    def __init__(self, num_classes, dropout=0.5,
                 arcface_s=30.0, arcface_m=0.50):
        super(co3net, self).__init__()
        self.num_classes = num_classes

        # Three parallel Competitive Blocks (official CO3Net config)
        self.cb1 = CompetitiveBlock(channel_in=1, n_competitor=9,
                                    ksize=35, stride=3, padding=17,
                                    init_ratio=1)
        self.cb2 = CompetitiveBlock(channel_in=1, n_competitor=36,
                                    ksize=17, stride=3, padding=8,
                                    init_ratio=0.5, o2=24)
        self.cb3 = CompetitiveBlock(channel_in=1, n_competitor=9,
                                    ksize=7, stride=3, padding=3,
                                    init_ratio=0.25)

        # FC layers  (17328 = 12*19*19 + 24*19*19 + 12*19*19)
        self.fc  = nn.Linear(17328, 4096)
        self.fc1 = nn.Linear(4096, 2048)
        self.drop = nn.Dropout(p=dropout)

        self.arclayer = ArcMarginProduct(2048, num_classes,
                                         s=arcface_s, m=arcface_m,
                                         easy_margin=False)

    def forward(self, x, y=None):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x1 = self.fc(x)
        x  = self.fc1(x1)

        # contrastive feature: 4096 + 2048 = 6144-d, L2-normalised
        fe = torch.cat((x1, x), dim=1)

        x = self.drop(x)
        x = self.arclayer(x, y)

        return x, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        """Extract 2048-d L2-normalised embedding for matching."""
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        x = self.fc1(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x


# ══════════════════════════════════════════════════════════════
#  NORMALISATION  (exact copy from CO3Net/models/dataset.py)
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
#  DATASET  — paired & single, for CASIA-MS-ROI
# ══════════════════════════════════════════════════════════════
def parse_smartphone_data(data_root):
    """
    Scan smartphone_data folder.
    Structure : {data_root}/{ID}/roi_square/{ID}_{hand}_{condition}.jpg
    Identity key = "{ID}_{hand}"  e.g. "1_left", "1_right"
    Returns dict  {identity_key: [path1, path2, …]}
    """
    id2paths = defaultdict(list)
    for subject_id in sorted(os.listdir(data_root)):
        roi_dir = os.path.join(data_root, subject_id, "roi_square")
        if not os.path.isdir(roi_dir):
            continue
        for fname in sorted(os.listdir(roi_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".bmp", ".png")):
                continue
            stem  = os.path.splitext(fname)[0]      # "1_left_jf"
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            # parts[0] = ID, parts[1] = hand (left/right)
            identity = parts[0] + "_" + parts[1]    # "1_left"
            id2paths[identity].append(os.path.join(roi_dir, fname))
    return dict(id2paths)


def split_smartphone_eval(id2paths, gallery_ratio=0.3, seed=42):
    """
    Split smartphone identities into gallery and probe.
    Each identity keeps its own label (0 … N-1).
    Returns (gallery_samples, probe_samples, label_map)
    where each sample is (path, label).
    """
    rng = random.Random(seed)
    label_map       = {}
    gallery_samples = []
    probe_samples   = []

    for idx, identity in enumerate(sorted(id2paths.keys())):
        label_map[identity] = idx
        paths = list(id2paths[identity])
        rng.shuffle(paths)
        n_gallery = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gallery]:
            gallery_samples.append((p, idx))
        for p in paths[n_gallery:]:
            probe_samples.append((p, idx))

    return gallery_samples, probe_samples, label_map


def parse_casia_ms(data_root, n_subjects=190, n_total_samples=2776, seed=42):
    """
    Select n_subjects identities from CASIA-MS with near-uniform sample counts.

    Distribution logic (example: N=190, S=2776):
      - target per ID  : S // N = 14,  remainder = S % N = 116
        → 116 IDs get 15 images, 74 IDs get 14 images
      - target per spectrum (6 spectra, e.g. target=14):
          base_per_spectrum = 14 // 6 = 2, remainder = 14 % 6 = 2
        → 2 spectra get 3 images, 4 spectra get 2 images  (total = 14)
      - which IDs / spectra get the extra image is randomised via seed.

    Returns dict  {identity_key: [path1, path2, …]}
    """
    rng = random.Random(seed)

    # ── Step 1: index all files by (identity, spectrum) ──────────────────
    # id_spec[(identity, spectrum)] = [path, …]
    id_spec = defaultdict(lambda: defaultdict(list))

    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".bmp", ".png")):
            continue
        stem  = os.path.splitext(fname)[0]          # "025_l_940_04"
        parts = stem.split("_")
        if len(parts) < 4:
            continue
        identity = parts[0] + "_" + parts[1]        # "025_l"
        spectrum = parts[2]                          # "940"
        id_spec[identity][spectrum].append(
            os.path.join(data_root, fname))

    all_identities = sorted(id_spec.keys())
    if n_subjects > len(all_identities):
        raise ValueError(
            f"Requested {n_subjects} subjects but only "
            f"{len(all_identities)} available in {data_root}.")

    # ── Step 2: randomly select N identities ─────────────────────────────
    selected = sorted(rng.sample(all_identities, n_subjects))

    # ── Step 3: assign per-ID sample targets ─────────────────────────────
    # base IDs get (S // N) images; the first `remainder` IDs get one extra
    base_per_id   = n_total_samples // n_subjects   # e.g. 14
    remainder_ids = n_total_samples %  n_subjects   # e.g. 116

    id_list = list(selected)
    rng.shuffle(id_list)                            # randomise who gets the extra
    id_target = {
        ident: base_per_id + (1 if i < remainder_ids else 0)
        for i, ident in enumerate(id_list)
    }

    # ── Step 4: for each identity distribute target across spectra ────────
    id2paths = {}
    actual_total = 0

    for ident in selected:
        target   = id_target[ident]
        spectra  = sorted(id_spec[ident].keys())
        n_spec   = len(spectra)

        if n_spec == 0:
            id2paths[ident] = []
            continue

        # base images per spectrum; first `rem_spec` spectra get one extra
        base_per_spec = target // n_spec            # e.g. 2
        rem_spec      = target %  n_spec            # e.g. 2

        spec_list = list(spectra)
        rng.shuffle(spec_list)                      # randomise which spectra get extra

        chosen = []
        for j, spectrum in enumerate(spec_list):
            k         = base_per_spec + (1 if j < rem_spec else 0)
            available = id_spec[ident][spectrum]
            k         = min(k, len(available))      # guard: never exceed available
            chosen.extend(rng.sample(available, k))

        id2paths[ident] = chosen
        actual_total   += len(chosen)

    # ── Step 5: summary ───────────────────────────────────────────────────
    counts      = [len(v) for v in id2paths.values()]
    spec_counts = []
    for ident in selected:
        for spectrum, paths in id_spec[ident].items():
            # how many were actually chosen from this spectrum
            chosen_from_spec = [
                p for p in id2paths[ident]
                if f"_{spectrum}_" in os.path.basename(p)
            ]
            spec_counts.append(len(chosen_from_spec))

    print(f"  Selected  : {len(id2paths)} identities")
    print(f"  Total     : {actual_total} images  (target={n_total_samples}, "
          f"diff={actual_total - n_total_samples})")
    print(f"  Per-ID    : min={min(counts)}  max={max(counts)}  "
          f"mean={sum(counts)/len(counts):.2f}")
    print(f"  Per-spec  : min={min(spec_counts)}  max={max(spec_counts)}  "
          f"mean={sum(spec_counts)/len(spec_counts):.2f}")

    return id2paths


def get_parser(dataset_name, casia_root, smartphone_root,
               n_subjects, n_samples, seed):
    """
    Returns (parse_fn, root_path) for the requested dataset name.
    parse_fn() → id2paths dict.
    """
    name = dataset_name.strip().lower().replace("-", "").replace("_", "")
    if name == "casiams":
        return (lambda: parse_casia_ms(casia_root,
                                       n_subjects=n_subjects,
                                       n_total_samples=n_samples,
                                       seed=seed),
                casia_root)
    elif name == "smartphone":
        return (lambda: parse_smartphone_data(smartphone_root),
                smartphone_root)
    else:
        raise ValueError(f"Unknown dataset name: '{dataset_name}'. "
                         f"Use 'CASIA-MS' or 'Smartphone'.")


def split_same_dataset(id2paths, train_subject_ratio=0.70,
                       gallery_ratio=0.50, seed=42):
    """
    Used when train and test come from the same dataset.
    - Splits subjects:  train_subject_ratio → train | rest → test
    - Within test subjects: gallery_ratio of images → gallery | rest → probe
    Returns (train_samples, gallery_samples, probe_samples,
             train_label_map, test_label_map)
    """
    rng = random.Random(seed)
    identities = sorted(id2paths.keys())
    rng.shuffle(identities)

    n_train = max(1, int(len(identities) * train_subject_ratio))
    train_ids = identities[:n_train]
    test_ids  = identities[n_train:]

    train_label_map = {k: i for i, k in enumerate(train_ids)}
    test_label_map  = {k: i for i, k in enumerate(test_ids)}

    train_samples   = [(p, train_label_map[ident])
                       for ident in train_ids
                       for p in id2paths[ident]]

    gallery_samples, probe_samples = [], []
    for ident in test_ids:
        paths = list(id2paths[ident])
        rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gal]:
            gallery_samples.append((p, test_label_map[ident]))
        for p in paths[n_gal:]:
            probe_samples.append((p, test_label_map[ident]))

    return (train_samples, gallery_samples, probe_samples,
            train_label_map, test_label_map)


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
    test_ids  = identities[n_train_ids:]

    # --- training identities → train + val ---
    train_label_map = {k: i for i, k in enumerate(sorted(train_ids))}
    all_train_samples = []
    for identity in train_ids:
        lab = train_label_map[identity]
        for p in id2paths[identity]:
            all_train_samples.append((p, lab))

    rng2 = random.Random(seed + 1)
    rng2.shuffle(all_train_samples)
    n_val = max(1, int(len(all_train_samples) * val_ratio))
    val_samples   = all_train_samples[:n_val]
    train_samples = all_train_samples[n_val:]

    # --- test identities → gallery + probe ---
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


# ────────── paired dataset (for training / val) ──────────
class CASIAMSDataset(Dataset):
    def __init__(self, samples, img_side=128, train=True, augment_factor=1):
        super().__init__()
        self.samples        = samples
        self.train          = train
        self.img_side       = img_side
        self.augment_factor = augment_factor if train else 1  # never augment val/test

        self.label2idxs = defaultdict(list)
        for i, (_, lab) in enumerate(samples):
            self.label2idxs[lab].append(i)

        # augmentation transform (used for all K copies during training)
        self.aug_transform = T.Compose([
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

        # clean transform (no augmentation — used for val/test)
        self.clean_transform = T.Compose([
            T.Resize(img_side),
            T.ToTensor(),
            NormSingleROI(outchannels=1),
        ])

    def __len__(self):
        # effective dataset is K × larger than the original
        return len(self.samples) * self.augment_factor

    def __getitem__(self, index):
        # map the virtual index back to a real sample
        real_idx   = index % len(self.samples)
        path1, label = self.samples[real_idx]

        if self.train:
            # pair with a different sample of the same identity
            idxs = self.label2idxs[label]
            idx2 = real_idx
            while idx2 == real_idx and len(idxs) > 1:
                idx2 = random.choice(idxs)
            path2 = self.samples[idx2][0]

            img1 = Image.open(path1).convert("L")
            img2 = Image.open(path2).convert("L")
            # every copy — including the first — gets its own random augmentation
            img1 = self.aug_transform(img1)
            img2 = self.aug_transform(img2)
        else:
            img1 = Image.open(path1).convert("L")
            img2 = img1.copy()
            img1 = self.clean_transform(img1)
            img2 = self.clean_transform(img2)

        return [img1, img2], label


# ────────── single-image dataset (for evaluation) ──────────
class CASIAMSDatasetSingle(Dataset):
    def __init__(self, samples, img_side=128):
        super().__init__()
        self.samples  = samples
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
#  TRAINING  — one-epoch worker
# ══════════════════════════════════════════════════════════════

def run_one_epoch(epoch, model, loader, criterion, con_criterion,
                  optimizer, device, phase,
                  ce_weight=0.8, con_weight=0.2):
    """
    Mirrors CO3Net train.py `fit()`.
    phase ∈ {'training', 'testing'}
    """
    is_train = (phase == "training")
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss    = 0.0
    running_correct = 0
    total           = 0

    for datas, target in loader:
        data1 = datas[0].to(device)
        data2 = datas[1].to(device)
        target = target.to(device)

        if is_train:
            optimizer.zero_grad()
            output,  fe1 = model(data1, target)
            output2, fe2 = model(data2, target)
            fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                output,  fe1 = model(data1, None)
                output2, fe2 = model(data2, None)
                fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)

        ce_loss  = criterion(output, target)
        con_loss = con_criterion(fe, target)
        loss     = ce_weight * ce_loss + con_weight * con_loss

        running_loss += loss.item() * data1.size(0)
        preds = output.data.max(dim=1)[1]
        running_correct += preds.eq(target).cpu().sum().item()
        total += data1.size(0)

        if is_train:
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / max(total, 1)
    epoch_acc  = 100.0 * running_correct / max(total, 1)
    return epoch_loss, epoch_acc


# ══════════════════════════════════════════════════════════════
#  EVALUATION  — EER + Rank-1 (mirrors CO3Net test())
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device):
    """Extract 2048-d L2-normalised embeddings."""
    model.eval()
    feats, labels = [], []
    for imgs, labs in loader:
        imgs = imgs.to(device)
        codes = model.getFeatureCode(imgs)
        feats.append(codes.cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def compute_eer(scores_array):
    """
    scores_array : (N, 2)  — col 0 = matching score, col 1 = label (+1 / -1).
    Returns (eer, threshold).
    """
    inscore  = scores_array[scores_array[:, 1] ==  1, 0]
    outscore = scores_array[scores_array[:, 1] == -1, 0]
    if len(inscore) == 0 or len(outscore) == 0:
        return 1.0, 0.0

    # inner should be bigger than outer for roc_curve
    mIn  = inscore.mean()
    mOut = outscore.mean()
    flipped = False
    if mIn < mOut:
        inscore  = -inscore
        outscore = -outscore
        flipped  = True

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
    2) Pairwise EER  (probe vs gallery).
    3) Aggregated EER (probe-probe, all-vs-all within probe set).
    4) Rank-1 accuracy.
    Returns (pairwise_eer, aggregated_eer, rank1_acc).
    """
    probe_feats,   probe_labels   = extract_features(model, probe_loader, device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)

    n_probe   = len(probe_feats)
    n_gallery = len(gallery_feats)

    # ── pairwise matching: probe vs gallery (cosine → arc distance) ──
    scores_list = []
    labels_list = []
    dist_matrix = np.zeros((n_probe, n_gallery))

    for i in range(n_probe):
        cos_sim = np.dot(gallery_feats, probe_feats[i])          # (n_gallery,)
        dists   = np.arccos(np.clip(cos_sim, -1, 1)) / np.pi    # 0‒1
        dist_matrix[i] = dists
        for j in range(n_gallery):
            scores_list.append(dists[j])
            labels_list.append(1 if probe_labels[i] == gallery_labels[j] else -1)

    scores_arr = np.column_stack([scores_list, labels_list])
    pair_eer, pair_th = compute_eer(scores_arr)

    # ── aggregated EER: all-vs-all within PROBE set ──
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

    # ── Rank-1 identification ──
    correct = 0
    for i in range(n_probe):
        best_j = np.argmin(dist_matrix[i])
        if probe_labels[i] == gallery_labels[best_j]:
            correct += 1
    rank1 = 100.0 * correct / max(n_probe, 1)

    # ── save score file + plots ──
    score_path = os.path.join(out_dir, f"scores_{tag}.txt")
    with open(score_path, "w") as f:
        for s_val, l_val in zip(scores_list, labels_list):
            f.write(f"{s_val} {l_val}\n")

    _save_roc_det(scores_arr, out_dir, tag, pair_eer, pair_th)

    print(f"  [{tag}]  pairEER={pair_eer*100:.4f}%  aggrEER={aggr_eer*100:.4f}%  "
          f"Rank-1={rank1:.2f}%")
    return pair_eer, aggr_eer, rank1


def _save_roc_det(scores_arr, out_dir, tag, eer, thresh):
    """Save ROC / DET / FAR-FRR plots — mirrors CO3Net getEER.py."""
    inscore  = scores_arr[scores_arr[:, 1] ==  1, 0]
    outscore = scores_arr[scores_arr[:, 1] == -1, 0]
    if len(inscore) == 0 or len(outscore) == 0:
        return

    mIn  = inscore.mean()
    mOut = outscore.mean()
    if mIn < mOut:
        inscore  = -inscore
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
#  PLOTTING HELPER  — loss & accuracy curves
# ══════════════════════════════════════════════════════════════

def plot_loss_acc(train_losses, val_losses,
                  train_accs, val_accs, results_dir):
    try:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(train_losses) + 1), train_losses, 'b', label='train loss')
        ax.plot(range(1, len(val_losses) + 1),   val_losses,   'r', label='val loss')
        ax.legend(); ax.set_xlabel('epoch'); ax.set_ylabel('loss')
        fig.savefig(os.path.join(results_dir, "losses.png"))
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(range(1, len(train_accs) + 1), train_accs, 'b', label='train acc')
        ax.plot(range(1, len(val_accs) + 1),   val_accs,   'r', label='val acc')
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
    train_data          = CONFIG["train_data"]
    test_data           = CONFIG["test_data"]
    data_root           = CONFIG["data_root"]           # CASIA-MS path
    test_data_root      = CONFIG["test_data_root"]      # Smartphone path
    test_gallery_ratio  = CONFIG["test_gallery_ratio"]
    train_subject_ratio = CONFIG["train_subject_ratio"]
    results_dir         = CONFIG["results_dir"]
    img_side            = CONFIG["img_side"]
    batch_size          = CONFIG["batch_size"]
    num_epochs          = CONFIG["num_epochs"]
    lr                  = CONFIG["lr"]
    lr_step             = CONFIG["lr_step"]
    lr_gamma            = CONFIG["lr_gamma"]
    dropout             = CONFIG["dropout"]
    arcface_s           = CONFIG["arcface_s"]
    arcface_m           = CONFIG["arcface_m"]
    ce_weight           = CONFIG["ce_weight"]
    con_weight          = CONFIG["con_weight"]
    temperature         = CONFIG["temperature"]
    seed                = CONFIG["random_seed"]
    save_every          = CONFIG["save_every"]
    eval_every          = CONFIG["eval_every"]
    nw                  = CONFIG["num_workers"]
    augment_factor      = CONFIG["augment_factor"]
    n_subjects          = CONFIG["n_casia_subjects"]
    n_samples           = CONFIG["n_casia_samples"]

    same_dataset = (train_data.strip().lower().replace("-", "") ==
                    test_data.strip().lower().replace("-", ""))

    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  CO3Net Palmprint Recognition")
    print(f"  Device         : {device}")
    print(f"  Train dataset  : {train_data}")
    print(f"  Test dataset   : {test_data}")
    if same_dataset:
        print(f"  Mode           : same-dataset split  "
              f"({int(train_subject_ratio*100)}% subjects train / "
              f"{int((1-train_subject_ratio)*100)}% subjects test)")
    print(f"  Loss           : {ce_weight}*CE + {con_weight}*SupCon(τ={temperature})")
    print(f"  Augment factor : {augment_factor}x")
    print(f"{'='*60}\n")

    # ── get parsers ───────────────────────────────────────────────────────
    train_parser, train_root = get_parser(train_data, data_root, test_data_root,
                                          n_subjects, n_samples, seed)
    test_parser,  test_root  = get_parser(test_data,  data_root, test_data_root,
                                          n_subjects, n_samples, seed)

    # ══════════════════════════════════════════════════════════════════════
    #  CASE 1 — different datasets
    # ══════════════════════════════════════════════════════════════════════
    if not same_dataset:
        # ── training set ─────────────────────────────────────────────────
        print(f"Scanning {train_data} (train) …")
        train_id2paths = train_parser()
        n_train_ids    = len(train_id2paths)
        n_train_imgs   = sum(len(v) for v in train_id2paths.values())
        print(f"  Found {n_train_ids} identities, {n_train_imgs} images.\n")

        train_label_map = {k: i for i, k in enumerate(sorted(train_id2paths.keys()))}
        train_samples   = [(p, train_label_map[ident])
                           for ident, paths in train_id2paths.items()
                           for p in paths]
        num_classes = len(train_label_map)

        # ── test set ─────────────────────────────────────────────────────
        print(f"Scanning {test_data} (test) …")
        test_id2paths = test_parser()
        n_test_ids    = len(test_id2paths)
        n_test_imgs   = sum(len(v) for v in test_id2paths.values())
        print(f"  Found {n_test_ids} identities, {n_test_imgs} images.\n")

        gallery_samples, probe_samples, _ = split_smartphone_eval(
            test_id2paths, gallery_ratio=test_gallery_ratio, seed=seed)

    # ══════════════════════════════════════════════════════════════════════
    #  CASE 2 — same dataset, split by subject
    # ══════════════════════════════════════════════════════════════════════
    else:
        print(f"Scanning {train_data} (shared train+test) …")
        all_id2paths = train_parser()           # parse once
        n_total_ids  = len(all_id2paths)
        n_total_imgs = sum(len(v) for v in all_id2paths.values())
        print(f"  Found {n_total_ids} identities, {n_total_imgs} images.\n")

        (train_samples, gallery_samples, probe_samples,
         train_label_map, _) = split_same_dataset(
            all_id2paths,
            train_subject_ratio = train_subject_ratio,
            gallery_ratio       = test_gallery_ratio,
            seed                = seed)

        num_classes  = len(train_label_map)
        n_train_ids  = num_classes
        n_train_imgs = len(train_samples)
        n_test_ids   = n_total_ids - n_train_ids
        n_test_imgs  = len(gallery_samples) + len(probe_samples)

    # ── loaders (shared for both cases) ──────────────────────────────────
    train_dataset = CASIAMSDataset(train_samples, img_side=img_side,
                                   train=True, augment_factor=augment_factor)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=nw, pin_memory=True)
    gallery_loader = DataLoader(
        CASIAMSDatasetSingle(gallery_samples, img_side=img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    probe_loader   = DataLoader(
        CASIAMSDatasetSingle(probe_samples, img_side=img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"  Train subjects : {n_train_ids}  |  "
          f"Train images : {n_train_imgs} (+aug → {n_train_imgs*augment_factor})")
    print(f"  Test subjects  : {n_test_ids}  |  "
          f"Gallery : {len(gallery_samples)}  |  Probe : {len(probe_samples)}")
    print(f"  Num classes    : {num_classes}\n")

    # ── model ─────────────────────────────────────────────────────────────
    print(f"Building CO3Net — num_classes={num_classes} …")
    net = co3net(num_classes=num_classes, dropout=dropout,
                 arcface_s=arcface_s, arcface_m=arcface_m)
    net.to(device)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
        net = DataParallel(net)
      
    '''
    # ── resume from checkpoint if one exists ──────────────────────────────
    for ckpt_path in [
        os.path.join(results_dir, "net_params_best_eer.pth"),
        os.path.join(results_dir, "net_params_best.pth"),
        os.path.join(results_dir, "net_params.pth"),
    ]:
        if os.path.exists(ckpt_path):
            _net = net.module if isinstance(net, DataParallel) else net
            _net.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"  Resumed from checkpoint : {ckpt_path}\n")
            break
    else:
        print(f"  No checkpoint found — training from scratch.\n")
    '''
  
    criterion     = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=temperature, base_temperature=temperature)
    optimizer     = optim.Adam(net.parameters(), lr=lr)
    scheduler     = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    # ── training loop ─────────────────────────────────────────────────────
    train_losses = []
    train_accs   = []
    best_eer     = 1.0
    last_eer     = float("nan")
    last_rank1   = float("nan")

    print(f"Starting training for {num_epochs} epochs …")
    print(f"  EER / Rank-1 evaluated every {eval_every} epochs.\n")

    if CONFIG.get("eval_only", False):
        print("  eval_only=True — skipping training.\n")
    else:
        for epoch in range(num_epochs):
            t_loss, t_acc = run_one_epoch(
                epoch, net, train_loader, criterion, con_criterion,
                optimizer, device, "training",
                ce_weight=ce_weight, con_weight=con_weight)
            scheduler.step()

            train_losses.append(t_loss)
            train_accs.append(t_acc)

            _net = net.module if isinstance(net, DataParallel) else net

            # ── periodic evaluation ───────────────────────────────────────
            if epoch % eval_every == 0 or epoch == num_epochs - 1:
                tag = f"ep{epoch:04d}_{test_data.replace('-','')}"
                cur_eer, cur_aggr_eer, cur_rank1 = evaluate(
                    _net, probe_loader, gallery_loader,
                    device, out_dir=rst_eval, tag=tag)
                last_eer   = cur_eer
                last_rank1 = cur_rank1

                if cur_eer < best_eer:
                    best_eer = cur_eer
                    torch.save(_net.state_dict(),
                               os.path.join(results_dir, "net_params_best_eer.pth"))
                    print(f"  *** New best EER: {best_eer*100:.4f}% ***")

            # ── periodic console print ────────────────────────────────────
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                ts        = time.strftime("%H:%M:%S")
                eer_str   = f"{last_eer*100:.4f}%"  if not math.isnan(last_eer)   else "N/A"
                rank1_str = f"{last_rank1:.2f}%"     if not math.isnan(last_rank1) else "N/A"
                print(
                    f"[{ts}] ep {epoch:04d} | "
                    f"loss={t_loss:.5f} | cls-acc={t_acc:.2f}% | "
                    f"EER={eer_str}  Rank-1={rank1_str}")

            # ── periodic checkpoint ───────────────────────────────────────
            if epoch % save_every == 0 or epoch == num_epochs - 1:
                torch.save(_net.state_dict(),
                           os.path.join(results_dir, "net_params.pth"))
                try:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].plot(train_losses, 'b')
                    axes[0].set_title("Train Loss")
                    axes[0].set_xlabel("epoch"); axes[0].grid(True)
                    axes[1].plot(train_accs, 'b')
                    axes[1].set_title("Train Acc (%)")
                    axes[1].set_xlabel("epoch"); axes[1].grid(True)
                    fig.tight_layout()
                    fig.savefig(os.path.join(results_dir, "train_curves.png"))
                    plt.close(fig)
                except Exception:
                    pass

    # ── final evaluation ──────────────────────────────────────────────────
    print(f"\n=== Final evaluation on {test_data} (best EER model) ===")
    best_model_path = os.path.join(results_dir, "net_params_best_eer.pth")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(results_dir, "net_params.pth")

    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_model_path, map_location=device))

    saved_name = (f"CO3Net_train{train_data.replace('-','').replace(' ','')}"
                  f"_test{test_data.replace('-','').replace(' ','')}.pth")
    torch.save(eval_net.state_dict(), os.path.join(results_dir, saved_name))
    print(f"  Model saved as {saved_name}")

    final_eer, final_aggr_eer, final_rank1 = evaluate(
        eval_net, probe_loader, gallery_loader,
        device, out_dir=rst_eval,
        tag=f"FINAL_{test_data.replace('-','')}")

    print(f"\n{'='*60}")
    print(f"  Train          : {train_data} ({n_train_ids} subjects, {n_train_imgs} imgs)")
    print(f"  Test           : {test_data}  ({n_test_ids} subjects, {n_test_imgs} imgs)")
    print(f"  FINAL Pairwise EER   : {final_eer*100:.4f}%")
    print(f"  FINAL Aggregated EER : {final_aggr_eer*100:.4f}%")
    print(f"  FINAL Rank-1         : {final_rank1:.3f}%")
    print(f"  Results saved to     : {results_dir}")
    print(f"{'='*60}\n")

    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"Train dataset      : {train_data}\n")
        f.write(f"Train subjects     : {n_train_ids}\n")
        f.write(f"Train images       : {n_train_imgs}\n")
        f.write(f"Augment factor     : {augment_factor}x\n")
        f.write(f"Num classes        : {num_classes}\n")
        f.write(f"Test dataset       : {test_data}\n")
        f.write(f"Test subjects      : {n_test_ids}\n")
        f.write(f"Test images        : {n_test_imgs}\n")
        f.write(f"Gallery samples    : {len(gallery_samples)}\n")
        f.write(f"Probe samples      : {len(probe_samples)}\n")
        f.write(f"Final Pairwise EER : {final_eer*100:.6f}%\n")
        f.write(f"Final Aggreg. EER  : {final_aggr_eer*100:.6f}%\n")
        f.write(f"Final Rank-1       : {final_rank1:.3f}%\n")


if __name__ == "__main__":
    main()
