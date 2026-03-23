"""
TSCAN v4

Key change from v3:
  Source domain split: 80% gallery (train), 20% query (eval) per identity.
  Stratified: for each identity, last ceil(20%) images held out as query.
  Every identity appears in both sets — required for EER pair building.

Phase 1 printed metrics (every EVAL_EVERY epochs):
  Loss      : AdaFace training loss on gallery
  Eval ID   : identification acc  — query features matched against gallery
  Eval EER  : verification EER    — genuine/impostor pairs from query set
  Target ID : identification acc on target spectrum
  Target EER: verification EER on target spectrum
  Pseudo%   : % of target samples passing confidence threshold

Removing train acc/EER — replaced by Eval acc/EER (query→gallery).
Gap between Eval and Target metrics = domain shift magnitude.
Gap between Train loss and Eval metrics = overfitting signal.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import glob
import copy
import time
import random
import itertools
import math
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T


# =============================================================================
# PARAMETERS
# =============================================================================

DATA_ROOT       = "/home/pai-ng/Jamal/CASIA-MS-ROI"

ALL_SPECTRUMS   = ['460', '630', '700', '850', '940', 'White']
SOURCE_SPECTRUM = '460'
TARGET_SPECTRUM = '630'
SEPARATE_HANDS  = True

# ── Source split ──────────────────────────────────────────────────────────────
EVAL_SPLIT      = 0.2   # fraction of each identity's images held out as query

# ── Model ─────────────────────────────────────────────────────────────────────
FEATURE_DIM     = 256

# ── AdaFace ───────────────────────────────────────────────────────────────────
ADAFACE_M0      = 0.5
ADAFACE_MMIN    = 0.25
ADAFACE_S       = 32.0

# ── Stage 1 ───────────────────────────────────────────────────────────────────
S1_EPOCHS        = 100
S1_LR_HEAD       = 1e-3
S1_LR_BACKBONE   = 1e-4
S1_WEIGHT_DECAY  = 5e-4
S1_BATCH_SIZE    = 64
S1_WARMUP_EPOCHS = 5

# ── Stage 2 ───────────────────────────────────────────────────────────────────
S2_EPOCHS        = 60
S2_LR_HEAD       = 5e-5
S2_LR_BACKBONE   = 5e-6
S2_WEIGHT_DECAY  = 5e-4
S2_BATCH_SIZE    = 32
S2_WARMUP_EPOCHS = 3

# ── Co-learning ───────────────────────────────────────────────────────────────
EMA_DECAY            = 0.999
PSEUDO_LABEL_THRESH  = 0.6
ALPHA                = 1.0
BETA                 = 0.8
GAMMA_LOSS           = 0.3

# ── Augmentation ──────────────────────────────────────────────────────────────
RESIZE_SIZE     = 124
CROP_SIZE       = 112

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS     = 8
PIN_MEMORY      = True
SEED            = 42
EVAL_EVERY      = 5
MAX_PAIRS       = 500_000


# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0;  self.count = 0

    def reset(self):
        self.sum = 0.0;  self.count = 0

    def update(self, val, n=1):
        self.sum += val * n;  self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


# =============================================================================
# AUGMENTATION
# =============================================================================

class GaussianNoise:
    def __init__(self, std=0.02):
        self.std = std
    def __call__(self, t):
        return (t + torch.randn_like(t) * self.std).clamp(0., 1.)


def weak_transform():
    return T.Compose([
        T.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        T.RandomCrop(CROP_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def strong_transform():
    return T.Compose([
        T.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        T.RandomCrop(CROP_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.4, saturation=0.4, hue=0.1),
        T.RandomAutocontrast(p=0.3),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        GaussianNoise(std=0.02),
    ])


def eval_transform():
    return T.Compose([
        T.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        T.CenterCrop(CROP_SIZE),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# =============================================================================
# DATASET
# =============================================================================

def parse_filename(filepath):
    base  = os.path.splitext(os.path.basename(filepath))[0]
    parts = base.split('_')
    if len(parts) < 4:
        return None, None
    identity = f"{parts[0]}_{parts[1]}" if SEPARATE_HANDS else parts[0]
    spectrum = parts[2]
    return identity, spectrum


def scan_spectrum(spectrum):
    """Return sorted list of (filepath, identity) for one spectrum."""
    files = sorted(glob.glob(os.path.join(DATA_ROOT, "*.jpg")))
    if not files:
        files = sorted(glob.glob(os.path.join(DATA_ROOT, "**", "*.jpg"),
                                 recursive=True))
    records = []
    for fp in files:
        identity, spec = parse_filename(fp)
        if identity is not None and spec == spectrum:
            records.append((fp, identity))
    return records


def build_label_map(records):
    identities = sorted(set(r[1] for r in records))
    return {name: idx for idx, name in enumerate(identities)}


def stratified_split(records, eval_fraction=0.2):
    """
    Split records into (gallery, query) with stratification per identity.

    For each identity:
      - Sort images deterministically by filepath.
      - Hold out the last ceil(eval_fraction * n) images as query.
      - Keep the rest as gallery (train).

    This guarantees:
      - Every identity has at least 1 image in each split (provided ≥2 imgs/id).
      - The split is deterministic (reproducible without a seed).
      - No data leakage: query images never appear in gallery.

    Args:
        records       : list of (filepath, identity_str)
        eval_fraction : fraction to hold out per identity (default 0.2)

    Returns:
        gallery : list of (filepath, identity_str)  — 80%, used for training
        query   : list of (filepath, identity_str)  — 20%, used for evaluation
    """
    # Group by identity, sort files within each group
    by_identity = defaultdict(list)
    for fp, ident in records:
        by_identity[ident].append(fp)
    for ident in by_identity:
        by_identity[ident].sort()

    gallery, query = [], []
    skipped = 0
    for ident, fps in sorted(by_identity.items()):
        n      = len(fps)
        n_eval = max(1, math.ceil(n * eval_fraction))

        if n < 2:
            # Cannot split a single image — put in gallery only, skip query
            gallery.extend([(fp, ident) for fp in fps])
            skipped += 1
            continue

        # Last n_eval images → query; rest → gallery
        gallery.extend([(fp, ident) for fp in fps[:-n_eval]])
        query.extend  ([(fp, ident) for fp in fps[-n_eval:]])

    if skipped:
        log(f"  [split] {skipped} identities with <2 images skipped from query")

    return gallery, query


class RecordDataset(Dataset):
    """
    Generic dataset from a pre-built list of (filepath, identity_str) records.
    Accepts an explicit label_map so gallery and query share the same class indices.
    """
    def __init__(self, records, transform, label_map):
        self.transform = transform
        self.label_map = label_map
        # Keep only records whose identity is in label_map
        self.records   = [(fp, ident) for fp, ident in records
                          if ident in label_map]

    @property
    def num_classes(self):
        return len(self.label_map)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fp, ident = self.records[idx]
        img = Image.open(fp).convert("RGB")
        return self.transform(img), self.label_map[ident]


class UnlabeledTargetDataset(Dataset):
    """Target domain: returns (weak_aug, strong_aug) pair, no labels."""
    def __init__(self, spectrum):
        self.weak   = weak_transform()
        self.strong = strong_transform()
        records     = scan_spectrum(spectrum)
        assert records, f"No target images found for spectrum '{spectrum}'"
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img = Image.open(self.records[idx][0]).convert("RGB")
        return self.weak(img), self.strong(img)


# =============================================================================
# MODEL
# =============================================================================

class FeatureEncoder(nn.Module):
    """
    ResNet18 partial unfreeze:
      Frozen    : conv1, bn1, relu, maxpool, layer1, layer2
      Trainable : layer3, layer4  (lower LR)
      Trainable : linear(512→feat_dim), Tanh  (higher LR)
    """
    def __init__(self, feat_dim=256, pretrained=True):
        super().__init__()
        weights  = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        self.frozen_layers    = nn.Sequential(*list(backbone.children())[:6])
        self.trainable_layers = nn.Sequential(*list(backbone.children())[6:9])
        self.flatten          = nn.Flatten()
        self.linear           = nn.Linear(512, feat_dim, bias=True)
        self.hash             = nn.Tanh()

        for param in self.frozen_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.frozen_layers(x)
        x    = self.trainable_layers(x)
        bb   = self.flatten(x)
        feat = self.hash(self.linear(bb))
        return feat, bb

    def backbone_parameters(self):
        return list(self.trainable_layers.parameters())

    def head_parameters(self):
        return list(self.linear.parameters())


class PalmNet(nn.Module):
    def __init__(self, feat_dim=256, pretrained=True):
        super().__init__()
        self.encoder = FeatureEncoder(feat_dim=feat_dim, pretrained=pretrained)

    def forward(self, x):
        return self.encoder(x)

    def get_features(self, x):
        return self.encoder(x)[0]

    def backbone_parameters(self):
        return self.encoder.backbone_parameters()

    def head_parameters(self):
        return self.encoder.head_parameters()


# =============================================================================
# ADAFACE LOSS
# =============================================================================

class AdaFaceLoss(nn.Module):
    def __init__(self, num_classes, feat_dim=256, m0=0.5, m_min=0.25, s=32.0):
        super().__init__()
        self.m0     = m0
        self.m_min  = m_min
        self.s      = s
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def _margin(self, norms):
        lo    = norms.min().detach()
        hi    = norms.max().detach()
        denom = (hi - lo).clamp(min=1e-8)
        return (self.m_min + (self.m0 - self.m_min) *
                (norms - lo) / denom).clamp(self.m_min, self.m0)

    def forward(self, features, labels):
        norms   = features.norm(dim=1)
        margins = self._margin(norms)
        feat_n  = F.normalize(features, dim=1)
        w_n     = F.normalize(self.weight, dim=1)
        cosine  = (feat_n @ w_n.T).clamp(-1 + 1e-7, 1 - 1e-7)
        theta   = torch.acos(cosine)
        m_col   = margins.unsqueeze(1)
        cos_m_  = cosine * torch.cos(m_col) - torch.sin(theta) * torch.sin(m_col)
        one_hot = F.one_hot(labels, self.weight.size(0)).float()
        logits  = self.s * (one_hot * cos_m_ + (1 - one_hot) * cosine)
        return F.cross_entropy(logits, labels)

    def get_logits(self, features):
        return (F.normalize(features, dim=1) @
                F.normalize(self.weight, dim=1).T * self.s)


# =============================================================================
# GRL + DISCRIMINATOR
# =============================================================================

class GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class DomainDiscriminator(nn.Module):
    def __init__(self, feat_dim=256, hidden=128, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.net   = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        return self.net(GRLFunction.apply(feat, self.alpha))

    def set_alpha(self, alpha):
        self.alpha = alpha


# =============================================================================
# TRAINING HELPERS
# =============================================================================

@torch.no_grad()
def ema_update(teacher, student, decay=0.999):
    for t_p, s_p in zip(teacher.parameters(), student.parameters()):
        t_p.data.mul_(decay).add_(s_p.data * (1.0 - decay))


def grl_alpha(cur_iter, max_iter, alpha_max=1.0):
    p = cur_iter / max(max_iter, 1)
    return float(alpha_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


@torch.no_grad()
def generate_pseudo_labels(teacher, weak_imgs, adaface, threshold):
    teacher.eval()
    feat, _   = teacher(weak_imgs)
    probs     = F.softmax(adaface.get_logits(feat), dim=1)
    max_p, pl = probs.max(dim=1)
    mask      = max_p >= threshold
    pl[~mask] = -1
    return pl, mask


def domain_loss(discriminator, src_feat, tgt_feat):
    src_lbl = torch.zeros(src_feat.size(0), 1, device=src_feat.device)
    tgt_lbl = torch.ones (tgt_feat.size(0), 1, device=tgt_feat.device)
    preds   = discriminator(torch.cat([src_feat, tgt_feat], dim=0))
    return F.binary_cross_entropy(preds, torch.cat([src_lbl, tgt_lbl], dim=0))


def make_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs,
                                eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def extract_features(model, loader):
    """Extract L2-normalised features and labels from a loader."""
    model.eval()
    feats, labs = [], []
    for imgs, labels in loader:
        feat = F.normalize(model.get_features(imgs.to(DEVICE)), dim=1)
        feats.append(feat.cpu().numpy())
        labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def compute_eer_and_acc(scores, is_genuine):
    """Sweep 1000 thresholds, return (best_acc, eer)."""
    gen = scores[ is_genuine]
    imp = scores[~is_genuine]
    far_arr, frr_arr, acc_arr = [], [], []
    for thr in np.linspace(-1.0, 1.0, 1000):
        TP = int((gen >= thr).sum());  FN = int((gen  < thr).sum())
        FP = int((imp >= thr).sum());  TN = int((imp  < thr).sum())
        far_arr.append(FP / max(FP + TN, 1))
        frr_arr.append(FN / max(TP + FN, 1))
        acc_arr.append((TP + TN) / max(TP + TN + FP + FN, 1))
    far_arr = np.array(far_arr);  frr_arr = np.array(frr_arr)
    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    eer     = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)
    acc     = float(np.array(acc_arr).max())
    return acc, eer


def build_pairs_gallery_query(gallery_feats, gallery_labels,
                               query_feats,   query_labels):
    """
    Build genuine/impostor pairs using QUERY as probe and GALLERY as reference.

    Genuine  : query[i] vs gallery[j] where query_labels[i] == gallery_labels[j]
    Impostor : query[i] vs gallery[j] where query_labels[i] != gallery_labels[j]

    This is the standard closed-set identification / open-set verification
    protocol used in biometric benchmarks.
    """
    rng = np.random.RandomState(42)
    genuine, impostor = [], []

    # Index gallery by identity for fast lookup
    gallery_by_id = defaultdict(list)
    for idx, lbl in enumerate(gallery_labels):
        gallery_by_id[lbl].append(idx)

    all_gallery_idx = np.arange(len(gallery_labels))

    for q_idx, q_lbl in enumerate(query_labels):
        q_feat = query_feats[q_idx]

        # Genuine: match against all gallery samples of same identity
        for g_idx in gallery_by_id.get(q_lbl, []):
            sim = float(np.dot(q_feat, gallery_feats[g_idx]))
            genuine.append(sim)

        # Impostor: sample a few random gallery entries of different identity
        n_imp = min(5, len(all_gallery_idx))
        sampled = rng.choice(len(all_gallery_idx), size=n_imp * 3, replace=False)
        count = 0
        for g_idx in sampled:
            if gallery_labels[g_idx] != q_lbl:
                sim = float(np.dot(q_feat, gallery_feats[g_idx]))
                impostor.append(sim)
                count += 1
                if count >= n_imp:
                    break

    scores     = np.array(genuine + impostor, dtype=np.float32)
    is_genuine = np.array([True] * len(genuine) + [False] * len(impostor))
    return scores, is_genuine


def identification_acc_gallery_query(model, adaface,
                                      gallery_loader, query_loader):
    """
    Closed-set identification accuracy:
      For each query, find nearest gallery sample by cosine similarity.
      Correct if the nearest gallery sample has the same identity label.
    """
    model.eval();  adaface.eval()

    # Build gallery feature matrix
    g_feats, g_labels = extract_features(model, gallery_loader)  # (N_g, D)

    # Match each query against gallery
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in query_loader:
            imgs   = imgs.to(DEVICE)
            q_feat = F.normalize(model.get_features(imgs), dim=1).cpu().numpy()
            # Cosine similarity: q_feat (B,D) @ g_feats.T (D,N_g) → (B, N_g)
            sims   = q_feat @ g_feats.T
            pred_idx    = sims.argmax(axis=1)          # nearest gallery index
            pred_labels = g_labels[pred_idx]           # predicted identity
            correct += int((pred_labels == labels.numpy()).sum())
            total   += labels.size(0)

    return correct / max(total, 1)


def evaluate_gallery_query(model, adaface, gallery_loader, query_loader):
    """
    Full eval: identification accuracy + verification EER.
    Gallery = probe reference, Query = probes.
    Returns (id_acc, eer).
    """
    g_feats, g_labels = extract_features(model, gallery_loader)
    q_feats, q_labels = extract_features(model, query_loader)

    # Identification accuracy (query → nearest gallery)
    sims        = q_feats @ g_feats.T               # (N_q, N_g)
    pred_labels = g_labels[sims.argmax(axis=1)]
    id_acc      = float((pred_labels == q_labels).mean())

    # Verification EER (genuine/impostor pairs, query as probe)
    scores, is_genuine = build_pairs_gallery_query(
        g_feats, g_labels, q_feats, q_labels
    )
    _, eer = compute_eer_and_acc(scores, is_genuine)

    return id_acc, eer


def evaluate_single_set(model, loader):
    """
    Verification EER within a single set (all-vs-all pairs).
    Used for target domain evaluation where there is no separate gallery/query.
    Returns (id_acc_via_cosine_logits, eer).
    """
    feats, labels = extract_features(model, loader)
    rng = np.random.RandomState(42)
    genuine, impostor = [], []

    by_id = defaultdict(list)
    for idx, lbl in enumerate(labels):
        by_id[lbl].append(idx)

    for uid, idxs in by_id.items():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                genuine.append(float(np.dot(feats[idxs[i]], feats[idxs[j]])))

    n_imp = min(len(genuine) * 5, MAX_PAIRS)
    N, seen = len(labels), 0
    while len(impostor) < n_imp and seen < n_imp * 3:
        i, j = rng.choice(N, 2, replace=False)
        if labels[i] != labels[j]:
            impostor.append(float(np.dot(feats[i], feats[j])))
        seen += 1

    scores     = np.array(genuine + impostor, dtype=np.float32)
    is_genuine = np.array([True] * len(genuine) + [False] * len(impostor))
    acc, eer   = compute_eer_and_acc(scores, is_genuine)
    return acc, eer


@torch.no_grad()
def pseudo_acceptance_rate(teacher, adaface, loader, threshold):
    """Fraction of target samples whose max confidence >= threshold."""
    teacher.eval()
    accept = total = 0
    for imgs, _ in loader:
        feat, _ = teacher(imgs.to(DEVICE))
        max_p   = F.softmax(adaface.get_logits(feat), dim=1).max(dim=1).values
        accept += (max_p >= threshold).sum().item()
        total  += imgs.size(0)
    return accept / max(total, 1)


# =============================================================================
# DATASET & MODEL SETUP
# =============================================================================

set_seed(SEED)

log("=" * 75)
log(f"TSCAN v4  |  {SOURCE_SPECTRUM} → {TARGET_SPECTRUM}  |  device={DEVICE}")
log(f"Source split: {int((1-EVAL_SPLIT)*100)}% gallery (train) / "
    f"{int(EVAL_SPLIT*100)}% query (eval)  — stratified per identity")
log("=" * 75)

# ── Source domain split ───────────────────────────────────────────────────────
all_source_records = scan_spectrum(SOURCE_SPECTRUM)
assert all_source_records, f"No source images found for spectrum '{SOURCE_SPECTRUM}'"

gallery_records, query_records = stratified_split(all_source_records, EVAL_SPLIT)

# Label map built from ALL source records so gallery and query share indices
LABEL_MAP   = build_label_map(all_source_records)
NUM_CLASSES = len(LABEL_MAP)

log(f"Source [{SOURCE_SPECTRUM}]: {len(all_source_records)} total images, "
    f"{NUM_CLASSES} identities")
log(f"  Gallery (train) : {len(gallery_records)} images "
    f"({len(gallery_records)/len(all_source_records)*100:.1f}%)")
log(f"  Query   (eval)  : {len(query_records)} images "
    f"({len(query_records)/len(all_source_records)*100:.1f}%)")

# Verify every identity appears in both sets
gallery_ids = set(r[1] for r in gallery_records)
query_ids   = set(r[1] for r in query_records)
missing     = query_ids - gallery_ids
if missing:
    log(f"  WARNING: {len(missing)} identities in query have no gallery samples")

target_records = scan_spectrum(TARGET_SPECTRUM)
log(f"Target  [{TARGET_SPECTRUM}]: {len(target_records)} images (unlabeled for adaptation)")

# ── DataLoaders ───────────────────────────────────────────────────────────────

# Phase 1 training: gallery with training augmentation
s1_gallery_loader = DataLoader(
    RecordDataset(gallery_records, weak_transform(), LABEL_MAP),
    batch_size=S1_BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)

# Evaluation: gallery and query with deterministic transform
gallery_eval_loader = DataLoader(
    RecordDataset(gallery_records, eval_transform(), LABEL_MAP),
    batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

query_eval_loader = DataLoader(
    RecordDataset(query_records, eval_transform(), LABEL_MAP),
    batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

# Target domain evaluation (all target images, eval transform)
tgt_eval_loader = DataLoader(
    RecordDataset(target_records, eval_transform(), LABEL_MAP),
    batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

# Phase 2 loaders
s2_gallery_loader = DataLoader(
    RecordDataset(gallery_records, strong_transform(), LABEL_MAP),
    batch_size=S2_BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)

s2_tgt_loader = DataLoader(
    UnlabeledTargetDataset(TARGET_SPECTRUM),
    batch_size=S2_BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)

# ── Models ───────────────────────────────────────────────────────────────────
teacher = PalmNet(feat_dim=FEATURE_DIM, pretrained=True).to(DEVICE)
adaface = AdaFaceLoss(num_classes=NUM_CLASSES, feat_dim=FEATURE_DIM,
                      m0=ADAFACE_M0, m_min=ADAFACE_MMIN, s=ADAFACE_S).to(DEVICE)

frozen_p   = sum(p.numel() for p in teacher.encoder.frozen_layers.parameters())
backbone_p = sum(p.numel() for p in teacher.backbone_parameters())
head_p     = sum(p.numel() for p in teacher.head_parameters())
adaface_p  = sum(p.numel() for p in adaface.parameters())
log(f"\nFrozen     : {frozen_p/1e6:.2f}M  (conv1, bn1, layer1, layer2)")
log(f"Backbone   : {backbone_p/1e6:.2f}M  trainable (layer3+layer4)  LR={S1_LR_BACKBONE}")
log(f"Head       : {head_p/1e6:.4f}M  trainable (linear+Tanh)    LR={S1_LR_HEAD}")
log(f"AdaFace W  : {adaface_p/1e6:.4f}M  trainable                  LR={S1_LR_HEAD}")


# =============================================================================
# PHASE 1 — TEACHER INITIALIZATION
# =============================================================================

log("\n" + "=" * 75)
log("PHASE 1 — Teacher Initialization")
log(f"  Training on: gallery ({int((1-EVAL_SPLIT)*100)}% of source)")
log(f"  Evaluation : query→gallery  (source held-out)")
log(f"               all-vs-all     (target domain)\n")
hdr = (f"{'Epoch':>6}  {'Loss':>8}  "
       f"{'Eval ID':>8}  {'Eval EER':>9}  "
       f"{'Tgt ID':>8}  {'Tgt EER':>8}  "
       f"{'Pseudo%':>8}")
log(hdr)
log("-" * 75)

s1_optimizer = optim.AdamW([
    {'params': teacher.backbone_parameters(), 'lr': S1_LR_BACKBONE},
    {'params': teacher.head_parameters(),     'lr': S1_LR_HEAD},
    {'params': adaface.parameters(),          'lr': S1_LR_HEAD},
], weight_decay=S1_WEIGHT_DECAY)

s1_scheduler = make_warmup_cosine_scheduler(
    s1_optimizer, S1_WARMUP_EPOCHS, S1_EPOCHS)

best_s1_eval_eer = 1.0
best_s1_state    = None
best_s1_adaface  = None

for epoch in range(1, S1_EPOCHS + 1):
    teacher.train();  adaface.train()
    loss_m = AverageMeter()

    for imgs, labels in s1_gallery_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        feat, _  = teacher(imgs)
        loss     = adaface(feat, labels)
        s1_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            teacher.backbone_parameters() +
            teacher.head_parameters() +
            list(adaface.parameters()), max_norm=5.0)
        s1_optimizer.step()
        loss_m.update(loss.item(), imgs.size(0))

    s1_scheduler.step()

    if epoch % EVAL_EVERY == 0 or epoch == S1_EPOCHS:
        # Source: query probes matched against gallery
        eval_id, eval_eer = evaluate_gallery_query(
            teacher, adaface, gallery_eval_loader, query_eval_loader)

        # Target: all-vs-all within target spectrum
        tgt_id, tgt_eer = evaluate_single_set(teacher, tgt_eval_loader)

        # Pseudo-label acceptance rate on target
        pseudo_pct = pseudo_acceptance_rate(
            teacher, adaface, tgt_eval_loader, PSEUDO_LABEL_THRESH) * 100

        log(f"{epoch:>6}  {loss_m.avg:>8.4f}  "
            f"{eval_id*100:>7.2f}%  {eval_eer*100:>8.2f}%  "
            f"{tgt_id*100:>7.2f}%  {tgt_eer*100:>7.2f}%  "
            f"{pseudo_pct:>7.1f}%")

        # Save best by source eval EER (held-out, not target)
        if eval_eer < best_s1_eval_eer:
            best_s1_eval_eer = eval_eer
            best_s1_state    = copy.deepcopy(teacher.state_dict())
            best_s1_adaface  = copy.deepcopy(adaface.state_dict())
            log(f"         ★ best Eval EER so far")
    else:
        log(f"{epoch:>6}  {loss_m.avg:>8.4f}")

log("-" * 75)
log(f"Phase 1 done  |  Best Src Eval EER = {best_s1_eval_eer*100:.2f}%")


# =============================================================================
# PHASE 2 — TEACHER-STUDENT CO-LEARNING
# =============================================================================

log("\n" + "=" * 75)
log("PHASE 2 — Teacher-Student Co-Learning (Domain Adaptation)")
log(f"  Source gallery → student (labeled)   |   Target → teacher pseudo-labels\n")
hdr2 = (f"{'Epoch':>6}  {'L_total':>8}  {'L_sup':>7}  "
        f"{'L_uns':>7}  {'L_dis':>7}  {'Pseudo%':>8}  "
        f"{'Eval ID':>8}  {'Eval EER':>9}  "
        f"{'Tgt ID':>8}  {'Tgt EER':>8}")
log(hdr2)
log("-" * 85)

# Load best Phase 1 weights
teacher.load_state_dict(best_s1_state)
adaface.load_state_dict(best_s1_adaface)

student = PalmNet(feat_dim=FEATURE_DIM, pretrained=True).to(DEVICE)
student.load_state_dict(best_s1_state)   # start identical to teacher

discriminator = DomainDiscriminator(
    feat_dim=FEATURE_DIM, hidden=128, alpha=1.0).to(DEVICE)

for p in teacher.parameters():          # teacher: EMA only, no gradient
    p.requires_grad = False

s2_optimizer = optim.AdamW([
    {'params': student.backbone_parameters(), 'lr': S2_LR_BACKBONE},
    {'params': student.head_parameters(),     'lr': S2_LR_HEAD},
    {'params': adaface.parameters(),          'lr': S2_LR_HEAD},
    {'params': discriminator.parameters(),    'lr': S2_LR_HEAD},
], weight_decay=S2_WEIGHT_DECAY)

s2_scheduler = make_warmup_cosine_scheduler(
    s2_optimizer, S2_WARMUP_EPOCHS, S2_EPOCHS)

total_steps = len(s2_gallery_loader) * S2_EPOCHS
global_step = 0

for epoch in range(1, S2_EPOCHS + 1):
    student.train();  teacher.eval()
    discriminator.train();  adaface.train()

    loss_t_m = AverageMeter();  loss_s_m = AverageMeter()
    loss_u_m = AverageMeter();  loss_d_m = AverageMeter()
    pseudo_m = AverageMeter()

    max_steps = max(len(s2_gallery_loader), len(s2_tgt_loader))
    src_iter  = itertools.cycle(s2_gallery_loader)
    tgt_iter  = itertools.cycle(s2_tgt_loader)

    for _ in range(max_steps):
        alpha = grl_alpha(global_step, total_steps, alpha_max=1.0)
        discriminator.set_alpha(alpha)

        src_imgs, src_lbl = next(src_iter)
        tgt_weak, tgt_str = next(tgt_iter)
        src_imgs = src_imgs.to(DEVICE);  src_lbl  = src_lbl.to(DEVICE)
        tgt_weak = tgt_weak.to(DEVICE);  tgt_str  = tgt_str.to(DEVICE)

        pl, mask    = generate_pseudo_labels(
            teacher, tgt_weak, adaface, PSEUDO_LABEL_THRESH)
        src_feat, _ = student(src_imgs)
        tgt_feat, _ = student(tgt_str)

        L_sup   = adaface(src_feat, src_lbl)
        L_unsup = adaface(tgt_feat[mask], pl[mask]) if mask.any() \
                  else torch.tensor(0.0, device=DEVICE)
        L_dis   = domain_loss(discriminator, src_feat, tgt_feat)
        L_total = ALPHA * L_sup + BETA * L_unsup + GAMMA_LOSS * L_dis

        s2_optimizer.zero_grad()
        L_total.backward()
        nn.utils.clip_grad_norm_(
            student.backbone_parameters() +
            student.head_parameters() +
            list(discriminator.parameters()) +
            list(adaface.parameters()), max_norm=5.0)
        s2_optimizer.step()
        ema_update(teacher, student, decay=EMA_DECAY)

        bs = src_imgs.size(0)
        loss_t_m.update(L_total.item(), bs)
        loss_s_m.update(L_sup.item(),   bs)
        loss_u_m.update(L_unsup.item() if mask.any() else 0., bs)
        loss_d_m.update(L_dis.item(),   bs)
        pseudo_m.update(mask.float().mean().item(), bs)
        global_step += 1

    s2_scheduler.step()

    if epoch % EVAL_EVERY == 0 or epoch == S2_EPOCHS:
        # Source: query→gallery (same split as Phase 1)
        eval_id, eval_eer = evaluate_gallery_query(
            student, adaface, gallery_eval_loader, query_eval_loader)
        # Target: all-vs-all
        tgt_id, tgt_eer   = evaluate_single_set(student, tgt_eval_loader)

        log(f"{epoch:>6}  {loss_t_m.avg:>8.4f}  {loss_s_m.avg:>7.4f}  "
            f"{loss_u_m.avg:>7.4f}  {loss_d_m.avg:>7.4f}  "
            f"{pseudo_m.avg*100:>7.1f}%  "
            f"{eval_id*100:>7.2f}%  {eval_eer*100:>8.2f}%  "
            f"{tgt_id*100:>7.2f}%  {tgt_eer*100:>7.2f}%")
    else:
        log(f"{epoch:>6}  {loss_t_m.avg:>8.4f}  {loss_s_m.avg:>7.4f}  "
            f"{loss_u_m.avg:>7.4f}  {loss_d_m.avg:>7.4f}  "
            f"{pseudo_m.avg*100:>7.1f}%")

log("-" * 85)
log("TSCAN v4 complete.")
