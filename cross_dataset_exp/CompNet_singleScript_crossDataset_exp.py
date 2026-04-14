"""
CompNet — Cross-Dataset Palmprint Recognition
==================================================
Augmentation: general spatial aug + label-free FFT amplitude mixup (4x expansion).

4x expansion per batch:
  view 1 — original (clean)
  view 2 — spatial aug (general_aug)
  view 3 — FFT mixup  (random batch pairs, low-frequency amplitude swap)
  view 4 — 50% spatial aug  OR  50% FFT mixup  (xor view)

All 4 views share the same identity label.
"""

# ==============================================================
#  CONFIG  — edit only this block
# ==============================================================
CONFIG = {
    # ── Dataset selection ──────────────────────────────────────
    # Choices: "CASIA-MS" | "Smartphone" | "MPDv2"
    "train_data"           : "Smartphone",
    "test_data"            : "Smartphone",

    # ── Dataset paths ──────────────────────────────────────────
    "casiams_data_root"    : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "smartphone_data_root" : "/home/pai-ng/Jamal/smartphone_data",
    "mpd_data_root"        : "/home/pai-ng/Jamal/MPDv2_mediapipe_roi",

    # ── Splitting ──────────────────────────────────────────────
    "train_subject_ratio"  : 0.80,
    "test_gallery_ratio"   : 0.50,

    # ── CASIA-MS sampling ──────────────────────────────────────
    "n_casia_subjects"     : 190,
    "n_casia_samples"      : 2776,

    # ── MPDv2 sampling ─────────────────────────────────────────
    "n_mpd_subjects"       : 190,
    "n_mpd_samples"        : 2850,

    # ── Model ──────────────────────────────────────────────────
    "img_side"             : 128,
    "embedding_dim"        : 512,
    "dropout"              : 0.25,
    "arcface_s"            : 30.0,
    "arcface_m"            : 0.50,

    # ── Training ───────────────────────────────────────────────
    "batch_size"           : 128,
    "num_epochs"           : 200,
    "lr"                   : 0.001,
    "lr_step"              : 30,
    "lr_gamma"             : 0.8,

    # ── FFT mixup ──────────────────────────────────────────────
    "fft_beta_min"         : 0.01,
    "fft_beta_max"         : 0.20,
    "fft_xor_prob"         : 0.50,

    # ── Misc ───────────────────────────────────────────────────
    "results_dir"          : "./rst_compnet",
    "random_seed"          : 42,
    "eval_every"           : 50,
    "num_workers"          : 4,
    "eval_only"            : False,
}
# ==============================================================

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
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

SEED = CONFIG["random_seed"]
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════
#  AUGMENTATION
# ══════════════════════════════════════════════════════════════

general_aug = T.Compose([
    T.RandomResizedCrop(size=128, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
    T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2.0)], p=0.2),
    T.RandomApply([T.RandomAutocontrast()], p=0.2),
])


def fft_mixup(x, beta_min=0.01, beta_max=0.20):
    """
    Label-free FFT amplitude mixup.
    Random batch-pair permutation, low-frequency amplitude swap.
    """
    B, C, H, W = x.shape

    perm = torch.randperm(B, device=x.device)
    while B > 1 and (perm == torch.arange(B, device=x.device)).any():
        perm = torch.randperm(B, device=x.device)

    fft             = torch.fft.fft2(x, dim=(-2, -1))
    amp             = torch.abs(fft)
    pha             = torch.angle(fft)
    amp_shifted     = torch.fft.fftshift(amp, dim=(-2, -1))
    amp_shifted_trg = amp_shifted[perm]

    beta = float(np.random.uniform(beta_min, beta_max))
    b    = int(np.floor(min(H, W) * beta))
    c_h, c_w = H // 2, W // 2

    if b > 0:
        amp_shifted[..., c_h - b: c_h + b,
                         c_w - b: c_w + b] = \
            amp_shifted_trg[..., c_h - b: c_h + b,
                                  c_w - b: c_w + b]

    amp_mixed = torch.fft.ifftshift(amp_shifted, dim=(-2, -1))
    x_aug     = torch.fft.ifft2(
        amp_mixed * torch.exp(1j * pha), dim=(-2, -1)).real
    return torch.clamp(x_aug, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════
#  MODEL
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
        self.init_ratio  = max(init_ratio, 1e-6)
        self.kernel      = 0

        _S = 9.2   * self.init_ratio
        _F = 0.057 / self.init_ratio
        _G = 2.0

        self.gamma = nn.Parameter(torch.FloatTensor([_G]))
        self.sigma = nn.Parameter(torch.FloatTensor([_S]))
        self.theta = nn.Parameter(
            torch.arange(0, channel_out).float() * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([_F]))
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def _gen(self, ksize, c_in, c_out, sigma, gamma, theta, f, psi):
        half = ksize // 2; ksz = 2 * half + 1
        y0   = torch.arange(-half, half + 1).float()
        x0   = torch.arange(-half, half + 1).float()
        y    = y0.view(1, -1).repeat(c_out, c_in, ksz, 1)
        x    = x0.view(-1, 1).repeat(c_out, c_in, 1, ksz)
        x    = x.to(sigma.device); y = y.to(sigma.device)
        xt   =  x * torch.cos(theta.view(-1,1,1,1)) + y * torch.sin(theta.view(-1,1,1,1))
        yt   = -x * torch.sin(theta.view(-1,1,1,1)) + y * torch.cos(theta.view(-1,1,1,1))
        gb   = -torch.exp(
            -0.5 * ((gamma * xt)**2 + yt**2) / (8 * sigma.view(-1,1,1,1)**2)
        ) * torch.cos(2 * math.pi * f.view(-1,1,1,1) * xt + psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2, 3], keepdim=True)

    def forward(self, x):
        self.kernel = self._gen(self.kernel_size, self.channel_in,
                                self.channel_out, self.sigma, self.gamma,
                                self.theta, self.f, self.psi)
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)


class CompetitiveBlock(nn.Module):
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 init_ratio=1, o1=32, o2=12):
        super().__init__()
        self.gabor   = GaborConv2d(channel_in, n_competitor, ksize,
                                   stride, padding, init_ratio)
        self.a       = nn.Parameter(torch.FloatTensor([1]))
        self.b       = nn.Parameter(torch.FloatTensor([0]))
        self.argmax  = nn.Softmax(dim=1)
        self.conv1   = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.gabor(x)
        x = self.argmax((x - self.b) * self.a)
        return self.conv2(self.maxpool(self.conv1(x)))


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super().__init__()
        self.s = s; self.m = m
        self.weight      = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.training:
            assert label is not None
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
            phi  = cosine * self.cos_m - sine * self.sin_m
            phi  = (torch.where(cosine > 0, phi, cosine)
                    if self.easy_margin
                    else torch.where(cosine > self.th, phi, cosine - self.mm))
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            return self.s * ((one_hot * phi) + ((1 - one_hot) * cosine))
        return self.s * cosine


class CompNet(nn.Module):
    """CompNet = CB1 ∥ CB2 ∥ CB3 + FC(9708→emb_dim) + Dropout + ArcFace"""
    def __init__(self, num_classes, embedding_dim=512,
                 arcface_s=30.0, arcface_m=0.50, dropout=0.25):
        super().__init__()
        self.cb1  = CompetitiveBlock(1, 9, 35, 3, 0, init_ratio=1.00)
        self.cb2  = CompetitiveBlock(1, 9, 17, 3, 0, init_ratio=0.50)
        self.cb3  = CompetitiveBlock(1, 9,  7, 3, 0, init_ratio=0.25)
        self.fc   = nn.Linear(9708, embedding_dim)
        self.drop = nn.Dropout(p=dropout)
        self.arc  = ArcMarginProduct(embedding_dim, num_classes,
                                     s=arcface_s, m=arcface_m)

    def _backbone(self, x):
        x1 = self.cb1(x).flatten(1)
        x2 = self.cb2(x).flatten(1)
        x3 = self.cb3(x).flatten(1)
        return self.fc(torch.cat([x1, x2, x3], dim=1))

    def forward(self, x, y=None):
        return self.arc(self.drop(self._backbone(x)), y)

    @torch.no_grad()
    def get_embedding(self, x):
        return F.normalize(self._backbone(x), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  NORMALISATION
# ══════════════════════════════════════════════════════════════

class NormSingleROI:
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        c, h, w = tensor.size()
        tensor  = tensor.view(c, h * w)
        idx     = tensor > 0; t = tensor[idx]
        tensor[idx] = t.sub_(t.mean()).div_(t.std() + 1e-6)
        tensor = tensor.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, self.outchannels, dim=0)
        return tensor


# ══════════════════════════════════════════════════════════════
#  DATASET PARSERS
# ══════════════════════════════════════════════════════════════

def parse_casia_ms(data_root, n_subjects=190, n_total_samples=2776, seed=42):
    rng     = random.Random(seed)
    id_spec = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")): continue
        stem  = os.path.splitext(fname)[0]; parts = stem.split("_")
        if len(parts) < 4: continue
        id_spec[parts[0]+"_"+parts[1]][parts[2]].append(
            os.path.join(data_root, fname))
    all_ids = sorted(id_spec.keys())
    if n_subjects > len(all_ids):
        raise ValueError(f"Requested {n_subjects} but only {len(all_ids)} available.")
    selected    = sorted(rng.sample(all_ids, n_subjects))
    base_per_id = n_total_samples // n_subjects
    rem_ids     = n_total_samples %  n_subjects
    id_list     = list(selected); rng.shuffle(id_list)
    id_target   = {ident: base_per_id + (1 if i < rem_ids else 0)
                   for i, ident in enumerate(id_list)}
    id2paths = {}; actual_total = 0
    for ident in selected:
        target    = id_target[ident]
        spec_list = list(sorted(id_spec[ident].keys())); rng.shuffle(spec_list)
        n_spec    = len(spec_list); base_s = target // n_spec; rem_s = target % n_spec
        chosen    = []
        for j, sp in enumerate(spec_list):
            k = min(base_s + (1 if j < rem_s else 0), len(id_spec[ident][sp]))
            chosen.extend(rng.sample(id_spec[ident][sp], k))
        id2paths[ident] = chosen; actual_total += len(chosen)
    counts = [len(v) for v in id2paths.values()]
    print(f"  [CASIA-MS] ids={len(id2paths)}  total={actual_total}  "
          f"per-id min/max/mean={min(counts)}/{max(counts)}/{sum(counts)/len(counts):.1f}")
    return id2paths


def parse_smartphone_data(data_root):
    id2paths = defaultdict(list)
    for subject_id in sorted(os.listdir(data_root)):
        roi_dir = os.path.join(data_root, subject_id, "roi_square")
        if not os.path.isdir(roi_dir): continue
        for fname in sorted(os.listdir(roi_dir)):
            if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")): continue
            parts = os.path.splitext(fname)[0].split("_")
            if len(parts) < 3: continue
            id2paths[parts[0]+"_"+parts[1]].append(os.path.join(roi_dir, fname))
    return dict(id2paths)


def parse_mpd_data(data_root, n_subjects=190, n_total_samples=2850, seed=42):
    rng           = random.Random(seed)
    images_per_id = math.ceil(n_total_samples / n_subjects)
    id_dev = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")): continue
        stem  = os.path.splitext(fname)[0]; parts = stem.split("_")
        if len(parts) != 5: continue
        subject, session, device, hand_side, iteration = parts
        if device not in ("h","m") or hand_side not in ("l","r"): continue
        id_dev[subject+"_"+hand_side][device].append(os.path.join(data_root, fname))
    def _qualifies(devs):
        h = len(devs.get("h",[])); m = len(devs.get("m",[]))
        return (h >= 8 and m >= 7) or (h >= 7 and m >= 8)
    eligible = {ident: devs for ident, devs in id_dev.items() if _qualifies(devs)}
    print(f"  [MPDv2] Eligible IDs: {len(eligible)} / {len(id_dev)}")
    if n_subjects > len(eligible):
        raise ValueError(f"Requested {n_subjects} but only {len(eligible)} qualify.")
    selected = sorted(rng.sample(sorted(eligible.keys()), n_subjects))
    id2paths = {}; actual_total = 0
    for ident in selected:
        devs  = eligible[ident]
        h_cnt = len(devs["h"]); m_cnt = len(devs["m"])
        if h_cnt > m_cnt:   alloc = {"h": 8, "m": 7}
        elif m_cnt > h_cnt: alloc = {"h": 7, "m": 8}
        else: alloc = {"h":8,"m":7} if rng.random() < 0.5 else {"h":7,"m":8}
        chosen = []
        for device, k in alloc.items():
            chosen.extend(rng.sample(devs[device], k))
        id2paths[ident] = chosen; actual_total += len(chosen)
    counts = [len(v) for v in id2paths.values()]
    print(f"  [MPDv2] ids={len(id2paths)}  total={actual_total}  "
          f"per-id min/max/mean={min(counts)}/{max(counts)}/{sum(counts)/len(counts):.1f}")
    return id2paths


def get_parser(dataset_name, cfg):
    name = dataset_name.strip().lower().replace("-","").replace("_","")
    seed = cfg["random_seed"]
    if name == "casiams":
        return lambda: parse_casia_ms(cfg["casiams_data_root"],
            n_subjects=cfg["n_casia_subjects"],
            n_total_samples=cfg["n_casia_samples"], seed=seed)
    elif name == "smartphone":
        return lambda: parse_smartphone_data(cfg["smartphone_data_root"])
    elif name == "mpdv2":
        return lambda: parse_mpd_data(cfg["mpd_data_root"],
            n_subjects=cfg["n_mpd_subjects"],
            n_total_samples=cfg["n_mpd_samples"], seed=seed)
    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'.")


# ══════════════════════════════════════════════════════════════
#  SPLITS
# ══════════════════════════════════════════════════════════════

def split_same_dataset(id2paths, train_subject_ratio=0.80,
                       gallery_ratio=0.50, seed=42):
    rng = random.Random(seed)
    identities = sorted(id2paths.keys()); rng.shuffle(identities)
    n_train   = max(1, int(len(identities) * train_subject_ratio))
    train_ids = identities[:n_train]; test_ids = identities[n_train:]
    train_label_map = {k: i for i, k in enumerate(train_ids)}
    test_label_map  = {k: i for i, k in enumerate(test_ids)}
    train_samples   = [(p, train_label_map[ident])
                       for ident in train_ids for p in id2paths[ident]]
    gallery_samples, probe_samples = [], []
    for ident in test_ids:
        paths = list(id2paths[ident]); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gal]: gallery_samples.append((p, test_label_map[ident]))
        for p in paths[n_gal:]: probe_samples.append((p, test_label_map[ident]))
    return train_samples, gallery_samples, probe_samples, train_label_map, test_label_map


def split_cross_dataset_test(id2paths, gallery_ratio=0.50, seed=42):
    rng       = random.Random(seed)
    label_map = {k: i for i, k in enumerate(sorted(id2paths.keys()))}
    gallery_samples, probe_samples = [], []
    for ident, paths in id2paths.items():
        paths = list(paths); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gal]: gallery_samples.append((p, label_map[ident]))
        for p in paths[n_gal:]: probe_samples.append((p, label_map[ident]))
    return gallery_samples, probe_samples, label_map


# ══════════════════════════════════════════════════════════════
#  PYTORCH DATASETS
# ══════════════════════════════════════════════════════════════

class PairedAugDataset(Dataset):
    """Returns (img_orig, img_spatial, label). FFT views built in run_one_epoch."""
    def __init__(self, samples, img_side=128):
        self.samples  = samples
        self.clean_tf = T.Compose([
            T.Resize(img_side), T.ToTensor(), NormSingleROI(outchannels=1)])
        self.spatial_tf = T.Compose([
            T.Resize(img_side), general_aug,
            T.ToTensor(), NormSingleROI(outchannels=1)])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return self.clean_tf(img), self.spatial_tf(img), label


class SingleDataset(Dataset):
    """Single-image dataset for gallery and probe."""
    def __init__(self, samples, img_side=128):
        self.samples   = samples
        self.transform = T.Compose([
            T.Resize(img_side), T.ToTensor(), NormSingleROI(outchannels=1)])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("L")), label


# ══════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════

def run_one_epoch(model, loader, criterion, optimizer, device, phase, cfg):
    is_train = (phase == "training")
    beta_min = cfg["fft_beta_min"]
    beta_max = cfg["fft_beta_max"]
    xor_prob = cfg["fft_xor_prob"]

    model.train() if is_train else model.eval()
    running_loss = 0.0; running_correct = 0; total = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for img_orig, img_spatial, target in loader:
            img_orig    = img_orig.to(device)
            img_spatial = img_spatial.to(device)
            target      = target.to(device)

            if is_train:
                optimizer.zero_grad()

                img_fft      = fft_mixup(img_orig, beta_min, beta_max)
                use_fft_mask = (torch.rand(img_orig.size(0), device=device) < xor_prob)
                mask         = use_fft_mask.view(-1, 1, 1, 1).float()
                img_xor      = mask * img_fft + (1 - mask) * img_spatial

                images_all = torch.cat([img_orig, img_spatial,
                                        img_fft, img_xor], dim=0)
                labels_all = target.repeat(4)

                output = model(images_all, labels_all)
                loss   = criterion(output, labels_all)
                loss.backward()
                optimizer.step()

                running_loss    += loss.item() * images_all.size(0)
                running_correct += output.data.max(1)[1].eq(labels_all).sum().item()
                total           += images_all.size(0)
            else:
                output = model(img_orig, None)
                loss   = criterion(output, target)
                running_loss    += loss.item() * img_orig.size(0)
                running_correct += output.data.max(1)[1].eq(target).sum().item()
                total           += img_orig.size(0)

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
    ins  = scores_array[scores_array[:,1] ==  1, 0]
    outs = scores_array[scores_array[:,1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0: return 1.0, 0.0
    flipped = ins.mean() < outs.mean()
    if flipped: ins, outs = -ins, -outs
    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s   = np.concatenate([ins, outs])
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    eer    = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))
    return eer, (-thresh if flipped else thresh)


def evaluate(model, probe_loader, gallery_loader, device,
             out_dir=".", tag="eval"):
    probe_feats,   probe_labels   = extract_features(model, probe_loader, device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)
    n_probe = len(probe_feats); n_gallery = len(gallery_feats)

    scores_list, labels_list = [], []
    dist_matrix = np.zeros((n_probe, n_gallery))
    for i in range(n_probe):
        cos_sim = np.dot(gallery_feats, probe_feats[i])
        dists   = np.arccos(np.clip(cos_sim, -1, 1)) / np.pi
        dist_matrix[i] = dists
        for j in range(n_gallery):
            scores_list.append(dists[j])
            labels_list.append(1 if probe_labels[i] == gallery_labels[j] else -1)

    scores_arr  = np.column_stack([scores_list, labels_list])
    pair_eer, _ = compute_eer(scores_arr)

    aggr_s, aggr_l = [], []
    for i in range(n_probe - 1):
        for j in range(i + 1, n_probe):
            d = np.arccos(np.clip(np.dot(probe_feats[i], probe_feats[j]), -1, 1)) / np.pi
            aggr_s.append(d)
            aggr_l.append(1 if probe_labels[i] == probe_labels[j] else -1)
    aggr_eer = compute_eer(np.column_stack([aggr_s, aggr_l]))[0] if aggr_s else 1.0

    correct = sum(probe_labels[i] == gallery_labels[np.argmin(dist_matrix[i])]
                  for i in range(n_probe))
    rank1   = 100.0 * correct / max(n_probe, 1)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list): f.write(f"{s} {l}\n")
    _save_roc_det(scores_arr, out_dir, tag)

    print(f"  [{tag}]  pairEER={pair_eer*100:.4f}%  "
          f"aggrEER={aggr_eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return pair_eer, aggr_eer, rank1


def _save_roc_det(scores_arr, out_dir, tag):
    ins  = scores_arr[scores_arr[:,1] ==  1, 0]
    outs = scores_arr[scores_arr[:,1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0: return
    if ins.mean() < outs.mean(): ins, outs = -ins, -outs
    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s   = np.concatenate([ins, outs])
    fpr, tpr, thr = roc_curve(y, s, pos_label=1); fnr = 1 - tpr
    try:
        pdf = PdfPages(os.path.join(out_dir, f"roc_det_{tag}.pdf"))
        for (xd, yd, xl, yl, title, xlim, ylim) in [
            (fpr*100, tpr*100, "FAR (%)", "GAR (%)", f"ROC — {tag}", [0,5], [90,100]),
            (fpr*100, fnr*100, "FAR (%)", "FRR (%)", f"DET — {tag}", [0,5], [0,5]),
        ]:
            fig, ax = plt.subplots()
            ax.plot(xd, yd, 'b-^', markersize=2)
            ax.plot(np.linspace(0,100,101),
                    np.linspace(100,0,101) if "ROC" in title else np.linspace(0,100,101),
                    'k-')
            ax.set(xlim=xlim, ylim=ylim, xlabel=xl, ylabel=yl, title=title)
            ax.grid(True); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots()
        ax.plot(thr, fpr*100, 'r-.', label='FAR', markersize=2)
        ax.plot(thr, fnr*100, 'b-^', label='FRR', markersize=2)
        ax.set(xlabel="Threshold", ylabel="Rate (%)", title=f"FAR/FRR — {tag}")
        ax.legend(); ax.grid(True); pdf.savefig(fig); plt.close(fig)
        pdf.close()
    except Exception as e:
        print(f"  [warn] plot failed: {e}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    train_data          = CONFIG["train_data"]
    test_data           = CONFIG["test_data"]
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
    embedding_dim       = CONFIG["embedding_dim"]
    seed                = CONFIG["random_seed"]
    eval_every          = CONFIG["eval_every"]
    nw                  = CONFIG["num_workers"]

    same_dataset = (train_data.strip().lower().replace("-","") ==
                    test_data.strip().lower().replace("-",""))

    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  CompNet — FFT Mixup + Spatial Aug (4×)")
    print(f"  Device         : {device}")
    print(f"  Train dataset  : {train_data}")
    print(f"  Test dataset   : {test_data}")
    if same_dataset:
        print(f"  Mode           : same-dataset split "
              f"({int(train_subject_ratio*100)}% train / "
              f"{int((1-train_subject_ratio)*100)}% test)")
    print(f"  Augmentation   : general_aug + FFT mixup (β∈"
          f"[{CONFIG['fft_beta_min']},{CONFIG['fft_beta_max']}])")
    print(f"  Loss           : CrossEntropy  (4× batch expansion)")
    print(f"{'='*60}\n")

    train_parser = get_parser(train_data, CONFIG)
    test_parser  = get_parser(test_data,  CONFIG)

    # ── dataset setup ─────────────────────────────────────────
    if same_dataset:
        print(f"Scanning {train_data} (shared train+test) …")
        all_id2paths = train_parser()
        n_total_ids  = len(all_id2paths)
        n_total_imgs = sum(len(v) for v in all_id2paths.values())
        print(f"  Found {n_total_ids} identities, {n_total_imgs} images.\n")
        (train_samples, gallery_samples, probe_samples,
         train_label_map, _) = split_same_dataset(
            all_id2paths,
            train_subject_ratio=train_subject_ratio,
            gallery_ratio=test_gallery_ratio, seed=seed)
        num_classes  = len(train_label_map)
        n_train_ids  = num_classes
        n_train_imgs = len(train_samples)
        n_test_ids   = n_total_ids - n_train_ids
        n_test_imgs  = len(gallery_samples) + len(probe_samples)
    else:
        print(f"Scanning {train_data} (train) …")
        train_id2paths = train_parser()
        n_train_ids    = len(train_id2paths)
        n_train_imgs   = sum(len(v) for v in train_id2paths.values())
        print(f"  Found {n_train_ids} identities, {n_train_imgs} images.\n")
        train_label_map = {k: i for i, k in enumerate(sorted(train_id2paths))}
        train_samples   = [(p, train_label_map[ident])
                           for ident, paths in train_id2paths.items()
                           for p in paths]
        num_classes = len(train_label_map)
        print(f"Scanning {test_data} (test) …")
        test_id2paths = test_parser()
        n_test_ids    = len(test_id2paths)
        n_test_imgs   = sum(len(v) for v in test_id2paths.values())
        print(f"  Found {n_test_ids} identities, {n_test_imgs} images.\n")
        gallery_samples, probe_samples, _ = split_cross_dataset_test(
            test_id2paths, gallery_ratio=test_gallery_ratio, seed=seed)

    # ── data loaders ──────────────────────────────────────────
    train_loader = DataLoader(
        PairedAugDataset(train_samples, img_side),
        batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=True)

    gallery_loader = DataLoader(
        SingleDataset(gallery_samples, img_side),
        batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True)

    probe_loader = DataLoader(
        SingleDataset(probe_samples, img_side),
        batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True)

    print(f"  Train  : {n_train_ids} subjects | "
          f"{n_train_imgs} imgs → {n_train_imgs*4} effective (4×)")
    print(f"  Test   : {n_test_ids} subjects | "
          f"Gallery {len(gallery_samples)} | Probe {len(probe_samples)}")
    print(f"  Classes: {num_classes}\n")

    # ── model ─────────────────────────────────────────────────
    print(f"Building CompNet — num_classes={num_classes} …")
    net = CompNet(num_classes, embedding_dim=embedding_dim,
                  arcface_s=arcface_s, arcface_m=arcface_m, dropout=dropout)
    net.to(device)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs")
        net = DataParallel(net)

    print("  Training from scratch.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, lr_step, lr_gamma)

    # ── training loop ─────────────────────────────────────────
    train_losses, train_accs = [], []
    best_eer = 1.0; last_eer = float("nan"); last_rank1 = float("nan")

    print(f"\nStarting training for {num_epochs} epochs …")
    print(f"  EER / Rank-1 evaluated every {eval_every} epochs.\n")

    if CONFIG.get("eval_only", False):
        print("  eval_only=True — skipping training.\n")
    else:
        for epoch in range(num_epochs):
            t_loss, t_acc = run_one_epoch(
                net, train_loader, criterion, optimizer,
                device, "training", CONFIG)
            scheduler.step()

            train_losses.append(t_loss); train_accs.append(t_acc)
            _net = net.module if isinstance(net, DataParallel) else net

            if epoch % eval_every == 0 or epoch == num_epochs - 1:
                tag = f"ep{epoch:04d}_{test_data.replace('-','')}"
                cur_eer, _, cur_rank1 = evaluate(
                    _net, probe_loader, gallery_loader,
                    device, out_dir=rst_eval, tag=tag)
                last_eer, last_rank1 = cur_eer, cur_rank1
                if cur_eer < best_eer:
                    best_eer = cur_eer
                    print(f"  *** New best EER: {best_eer*100:.4f}% ***")

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                ts        = time.strftime("%H:%M:%S")
                eer_str   = f"{last_eer*100:.4f}%"  if not math.isnan(last_eer)   else "N/A"
                rank1_str = f"{last_rank1:.2f}%"     if not math.isnan(last_rank1) else "N/A"
                print(f"[{ts}] ep {epoch:04d} | "
                      f"loss={t_loss:.5f} | acc={t_acc:.2f}% | "
                      f"EER={eer_str}  Rank-1={rank1_str}")

            if epoch % 50 == 0 or epoch == num_epochs - 1:
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

    # ── final evaluation ──────────────────────────────────────
    print(f"\n=== Final evaluation on {test_data} ===")
    _net = net.module if isinstance(net, DataParallel) else net

    final_eer, final_aggr_eer, final_rank1 = evaluate(
        _net, probe_loader, gallery_loader,
        device, out_dir=rst_eval,
        tag=f"FINAL_{test_data.replace('-','')}")

    print(f"\n{'='*60}")
    print(f"  Train  : {train_data} ({n_train_ids} subjects, {n_train_imgs} imgs)")
    print(f"  Test   : {test_data}  ({n_test_ids} subjects, {n_test_imgs} imgs)")
    print(f"  FINAL Pairwise EER   : {final_eer*100:.4f}%")
    print(f"  FINAL Aggregated EER : {final_aggr_eer*100:.4f}%")
    print(f"  FINAL Rank-1         : {final_rank1:.3f}%")
    print(f"  Results saved to     : {results_dir}")
    print(f"{'='*60}\n")

    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"Train dataset      : {train_data}\n")
        f.write(f"Train subjects     : {n_train_ids}\n")
        f.write(f"Train images       : {n_train_imgs}\n")
        f.write(f"4x batch expansion : via general_aug + FFT mixup\n")
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
