"""
TSCAN v5

Changes from v4:
  1. Accuracy → Rank-1: for each query sample, find 1-nearest gallery sample
     by cosine similarity. Rank-1 = fraction where top-1 match is correct identity.
  2. Pseudo% column removed from all printed output.
  3. Phase 2 adaptation fixes:
     - GAMMA_LOSS 0.3 → 1.0  (stronger domain alignment pressure)
     - EMA_DECAY  0.999 → 0.99 (teacher updates faster from student)
     - PSEUDO_LABEL_THRESH 0.6 → 0.5 (accept more pseudo-labels early on)
     - S2_LR_HEAD 5e-5 → 1e-4 (student head learns faster)
     - Phase 2 now also reports whether target EER improved vs Phase 1 baseline
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
EVAL_SPLIT      = 0.2

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
S2_LR_HEAD       = 1e-4      # was 5e-5 — student needs to learn faster
S2_LR_BACKBONE   = 5e-6
S2_WEIGHT_DECAY  = 5e-4
S2_BATCH_SIZE    = 32
S2_WARMUP_EPOCHS = 3

# ── Co-learning ───────────────────────────────────────────────────────────────
EMA_DECAY            = 0.99   # was 0.999 — faster teacher update from student
PSEUDO_LABEL_THRESH  = 0.5    # was 0.6 — accept more pseudo-labels
ALPHA                = 1.0
BETA                 = 0.8
GAMMA_LOSS           = 1.0    # was 0.3 — stronger domain alignment

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
    return identity, parts[2]


def scan_spectrum(spectrum):
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
    return {name: idx for idx, name in
            enumerate(sorted(set(r[1] for r in records)))}


def stratified_split(records, eval_fraction=0.2):
    """
    Per-identity split: last ceil(eval_fraction * n) images → query.
    Deterministic (sorted filenames), no randomness.
    """
    by_identity = defaultdict(list)
    for fp, ident in records:
        by_identity[ident].append(fp)
    for ident in by_identity:
        by_identity[ident].sort()

    gallery, query = [], []
    for ident, fps in sorted(by_identity.items()):
        n      = len(fps)
        n_eval = max(1, math.ceil(n * eval_fraction))
        if n < 2:
            gallery.extend([(fp, ident) for fp in fps])
            continue
        gallery.extend([(fp, ident) for fp in fps[:-n_eval]])
        query.extend  ([(fp, ident) for fp in fps[-n_eval:]])
    return gallery, query


class RecordDataset(Dataset):
    def __init__(self, records, transform, label_map):
        self.transform = transform
        self.label_map = label_map
        self.records   = [(fp, ident) for fp, ident in records
                          if ident in label_map]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fp, ident = self.records[idx]
        return (self.transform(Image.open(fp).convert("RGB")),
                self.label_map[ident])


class UnlabeledTargetDataset(Dataset):
    def __init__(self, spectrum):
        self.weak   = weak_transform()
        self.strong = strong_transform()
        self.records = scan_spectrum(spectrum)
        assert self.records, f"No target images: '{spectrum}'"

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
    Frozen  : conv1, bn1, relu, maxpool, layer1, layer2
    Trainable: layer3, layer4 (low LR) + linear(512→feat_dim) + Tanh (high LR)
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
        for p in self.frozen_layers.parameters():
            p.requires_grad = False

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
def ema_update(teacher, student, decay=0.99):
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
    return F.binary_cross_entropy(preds,
                                   torch.cat([src_lbl, tgt_lbl], dim=0))


def make_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer,
                                T_max=total_epochs - warmup_epochs,
                                eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def extract_features(model, loader):
    """Extract L2-normalised features and integer labels from loader."""
    model.eval()
    feats, labs = [], []
    for imgs, labels in loader:
        feat = F.normalize(model.get_features(imgs.to(DEVICE)), dim=1)
        feats.append(feat.cpu().numpy())
        labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def rank1_gallery_query(gallery_feats, gallery_labels,
                         query_feats,   query_labels):
    """
    Rank-1 identification accuracy: query → gallery.

    For each query sample:
      1. Compute cosine similarity against every gallery sample.
      2. Find the gallery sample with highest similarity (rank-1 match).
      3. Correct if rank-1 gallery label == query label.

    Rank-1 Acc = # correct rank-1 matches / total query samples
    """
    # Full similarity matrix: (N_query, N_gallery)
    sims        = query_feats @ gallery_feats.T       # cosine sim (already L2-normed)
    top1_idx    = sims.argmax(axis=1)                 # index of best gallery match
    top1_labels = gallery_labels[top1_idx]            # identity of best match
    correct     = int((top1_labels == query_labels).sum())
    total       = len(query_labels)
    return correct / max(total, 1)


def rank1_single_set(feats, labels):
    """
    Rank-1 accuracy within a single set (target domain, no separate gallery).
    For each sample, exclude itself, find nearest neighbour, check identity.
    """
    # Full similarity matrix
    sims = feats @ feats.T                             # (N, N)
    np.fill_diagonal(sims, -2.0)                      # exclude self-match
    top1_idx    = sims.argmax(axis=1)
    top1_labels = labels[top1_idx]
    correct     = int((top1_labels == labels).sum())
    return correct / max(len(labels), 1)


def compute_eer(scores, is_genuine):
    """Sweep 1000 thresholds, return EER."""
    gen = scores[ is_genuine]
    imp = scores[~is_genuine]
    far_arr, frr_arr = [], []
    for thr in np.linspace(-1.0, 1.0, 1000):
        TP = int((gen >= thr).sum());  FN = int((gen  < thr).sum())
        FP = int((imp >= thr).sum());  TN = int((imp  < thr).sum())
        far_arr.append(FP / max(FP + TN, 1))
        frr_arr.append(FN / max(TP + FN, 1))
    far_arr = np.array(far_arr);  frr_arr = np.array(frr_arr)
    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    return float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)


def build_pairs_gallery_query(g_feats, g_labels, q_feats, q_labels):
    """
    Genuine : (query_i, gallery_j) where labels match
    Impostor: (query_i, gallery_j) where labels differ (random sample)
    """
    rng = np.random.RandomState(42)
    g_by_id = defaultdict(list)
    for idx, lbl in enumerate(g_labels):
        g_by_id[lbl].append(idx)

    genuine, impostor = [], []
    for q_idx, q_lbl in enumerate(q_labels):
        q_feat = q_feats[q_idx]
        for g_idx in g_by_id.get(q_lbl, []):
            genuine.append(float(np.dot(q_feat, g_feats[g_idx])))
        sampled = rng.choice(len(g_labels), size=min(15, len(g_labels)),
                              replace=False)
        for g_idx in sampled:
            if g_labels[g_idx] != q_lbl:
                impostor.append(float(np.dot(q_feat, g_feats[g_idx])))

    scores     = np.array(genuine + impostor, dtype=np.float32)
    is_genuine = np.array([True] * len(genuine) + [False] * len(impostor))
    return scores, is_genuine


def build_pairs_single_set(feats, labels):
    """All genuine pairs + random impostor pairs within one set."""
    rng = np.random.RandomState(42)
    by_id = defaultdict(list)
    for idx, lbl in enumerate(labels):
        by_id[lbl].append(idx)

    genuine, impostor = [], []
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
    return scores, is_genuine


def evaluate_source(model, gallery_loader, query_loader):
    """
    Source domain: query → gallery.
    Returns (rank1_acc, eer).
    Rank-1: each query matched to nearest gallery sample.
    EER   : genuine/impostor pairs, query as probe.
    """
    g_feats, g_labels = extract_features(model, gallery_loader)
    q_feats, q_labels = extract_features(model, query_loader)

    r1  = rank1_gallery_query(g_feats, g_labels, q_feats, q_labels)
    scores, is_genuine = build_pairs_gallery_query(
        g_feats, g_labels, q_feats, q_labels)
    eer = compute_eer(scores, is_genuine)
    return r1, eer


def evaluate_target(model, loader):
    """
    Target domain: all-vs-all (no separate gallery).
    Returns (rank1_acc, eer).
    Rank-1: each sample matched to nearest neighbour (self excluded).
    EER   : all genuine/impostor pairs within set.
    """
    feats, labels = extract_features(model, loader)
    r1            = rank1_single_set(feats, labels)
    scores, is_genuine = build_pairs_single_set(feats, labels)
    eer           = compute_eer(scores, is_genuine)
    return r1, eer


# =============================================================================
# DATASET & MODEL SETUP
# =============================================================================

set_seed(SEED)

log("=" * 72)
log(f"TSCAN v5  |  {SOURCE_SPECTRUM} → {TARGET_SPECTRUM}  |  device={DEVICE}")
log(f"Accuracy metric : Rank-1 (query → nearest gallery sample)")
log(f"Source split    : {int((1-EVAL_SPLIT)*100)}% gallery / "
    f"{int(EVAL_SPLIT*100)}% query  (stratified per identity)")
log("=" * 72)

# Source split
all_src       = scan_spectrum(SOURCE_SPECTRUM)
assert all_src, f"No source images for '{SOURCE_SPECTRUM}'"
gallery_recs, query_recs = stratified_split(all_src, EVAL_SPLIT)
LABEL_MAP     = build_label_map(all_src)
NUM_CLASSES   = len(LABEL_MAP)

log(f"Source [{SOURCE_SPECTRUM}] : {len(all_src)} total | "
    f"gallery={len(gallery_recs)} | query={len(query_recs)} | "
    f"identities={NUM_CLASSES}")
log(f"Target [{TARGET_SPECTRUM}] : {len(scan_spectrum(TARGET_SPECTRUM))} images")

# DataLoaders
s1_gallery_loader = DataLoader(
    RecordDataset(gallery_recs, weak_transform(), LABEL_MAP),
    batch_size=S1_BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)

gallery_eval_loader = DataLoader(
    RecordDataset(gallery_recs, eval_transform(), LABEL_MAP),
    batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

query_eval_loader = DataLoader(
    RecordDataset(query_recs, eval_transform(), LABEL_MAP),
    batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

tgt_eval_loader = DataLoader(
    RecordDataset(scan_spectrum(TARGET_SPECTRUM), eval_transform(), LABEL_MAP),
    batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

s2_gallery_loader = DataLoader(
    RecordDataset(gallery_recs, strong_transform(), LABEL_MAP),
    batch_size=S2_BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)

s2_tgt_loader = DataLoader(
    UnlabeledTargetDataset(TARGET_SPECTRUM),
    batch_size=S2_BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)

# Models
teacher = PalmNet(feat_dim=FEATURE_DIM, pretrained=True).to(DEVICE)
adaface = AdaFaceLoss(num_classes=NUM_CLASSES, feat_dim=FEATURE_DIM,
                      m0=ADAFACE_M0, m_min=ADAFACE_MMIN,
                      s=ADAFACE_S).to(DEVICE)

frozen_p   = sum(p.numel() for p in teacher.encoder.frozen_layers.parameters())
backbone_p = sum(p.numel() for p in teacher.backbone_parameters())
head_p     = sum(p.numel() for p in teacher.head_parameters())
log(f"\nFrozen : {frozen_p/1e6:.2f}M | "
    f"Backbone (trainable): {backbone_p/1e6:.2f}M | "
    f"Head: {head_p/1e6:.4f}M | "
    f"AdaFace W: {sum(p.numel() for p in adaface.parameters())/1e6:.4f}M")


# =============================================================================
# PHASE 1 — TEACHER INITIALIZATION
# =============================================================================

log("\n" + "=" * 72)
log("PHASE 1 — Teacher Initialization")
log("  Rank-1: each query matched to nearest gallery sample (cosine sim)")
log("  EER   : genuine/impostor pairs, query as probe, gallery as reference\n")
log(f"{'Epoch':>6}  {'Loss':>8}  "
    f"{'Src R1':>8}  {'Src EER':>8}  "
    f"{'Tgt R1':>8}  {'Tgt EER':>8}")
log("-" * 56)

s1_optimizer = optim.AdamW([
    {'params': teacher.backbone_parameters(), 'lr': S1_LR_BACKBONE},
    {'params': teacher.head_parameters(),     'lr': S1_LR_HEAD},
    {'params': adaface.parameters(),          'lr': S1_LR_HEAD},
], weight_decay=S1_WEIGHT_DECAY)
s1_scheduler = make_warmup_cosine_scheduler(
    s1_optimizer, S1_WARMUP_EPOCHS, S1_EPOCHS)

best_s1_eer    = 1.0
best_s1_state  = None
best_s1_adaface = None

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
        src_r1, src_eer = evaluate_source(teacher,
                                           gallery_eval_loader,
                                           query_eval_loader)
        tgt_r1, tgt_eer = evaluate_target(teacher, tgt_eval_loader)

        marker = " ★" if src_eer < best_s1_eer else ""
        log(f"{epoch:>6}  {loss_m.avg:>8.4f}  "
            f"{src_r1*100:>7.2f}%  {src_eer*100:>7.2f}%  "
            f"{tgt_r1*100:>7.2f}%  {tgt_eer*100:>7.2f}%{marker}")

        if src_eer < best_s1_eer:
            best_s1_eer     = src_eer
            best_s1_state   = copy.deepcopy(teacher.state_dict())
            best_s1_adaface = copy.deepcopy(adaface.state_dict())
    else:
        log(f"{epoch:>6}  {loss_m.avg:>8.4f}")

log("-" * 56)
log(f"Phase 1 done  |  Best Src EER = {best_s1_eer*100:.2f}%")

# Record Phase 1 target baseline for Phase 2 comparison
teacher.load_state_dict(best_s1_state)
adaface.load_state_dict(best_s1_adaface)
p1_tgt_r1, p1_tgt_eer = evaluate_target(teacher, tgt_eval_loader)
log(f"Phase 1 target baseline  →  Rank-1={p1_tgt_r1*100:.2f}%  "
    f"EER={p1_tgt_eer*100:.2f}%")


# =============================================================================
# PHASE 2 — TEACHER-STUDENT CO-LEARNING
# =============================================================================

log("\n" + "=" * 72)
log("PHASE 2 — Teacher-Student Co-Learning (Domain Adaptation)")
log("  Src Rank-1/EER : student performance on source query→gallery")
log("                   (tracks: is student forgetting source knowledge?)")
log("  Tgt Rank-1/EER : student performance on target all-vs-all")
log("                   (tracks: is adaptation actually working?)")
log(f"  Phase-1 target baseline: "
    f"Rank-1={p1_tgt_r1*100:.2f}%  EER={p1_tgt_eer*100:.2f}%\n")

log(f"{'Epoch':>6}  {'L_total':>8}  {'L_sup':>7}  "
    f"{'L_uns':>7}  {'L_dis':>7}  "
    f"{'Src R1':>8}  {'Src EER':>8}  "
    f"{'Tgt R1':>8}  {'Tgt EER':>8}")
log("-" * 75)

# Initialise student from best Phase 1 teacher
student = PalmNet(feat_dim=FEATURE_DIM, pretrained=True).to(DEVICE)
student.load_state_dict(best_s1_state)

discriminator = DomainDiscriminator(
    feat_dim=FEATURE_DIM, hidden=128, alpha=1.0).to(DEVICE)

for p in teacher.parameters():
    p.requires_grad = False    # teacher: EMA only

s2_optimizer = optim.AdamW([
    {'params': student.backbone_parameters(), 'lr': S2_LR_BACKBONE},
    {'params': student.head_parameters(),     'lr': S2_LR_HEAD},
    {'params': adaface.parameters(),          'lr': S2_LR_HEAD},
    {'params': discriminator.parameters(),    'lr': S2_LR_HEAD},
], weight_decay=S2_WEIGHT_DECAY)
s2_scheduler = make_warmup_cosine_scheduler(
    s2_optimizer, S2_WARMUP_EPOCHS, S2_EPOCHS)

best_s2_tgt_eer = p1_tgt_eer    # must beat Phase 1 baseline to count
total_steps     = len(s2_gallery_loader) * S2_EPOCHS
global_step     = 0

for epoch in range(1, S2_EPOCHS + 1):
    student.train();  teacher.eval()
    discriminator.train();  adaface.train()

    loss_t_m = AverageMeter();  loss_s_m = AverageMeter()
    loss_u_m = AverageMeter();  loss_d_m = AverageMeter()

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
        global_step += 1

    s2_scheduler.step()

    if epoch % EVAL_EVERY == 0 or epoch == S2_EPOCHS:
        src_r1, src_eer = evaluate_source(student,
                                           gallery_eval_loader,
                                           query_eval_loader)
        tgt_r1, tgt_eer = evaluate_target(student, tgt_eval_loader)

        # Show delta vs Phase 1 target baseline
        delta_r1  = (tgt_r1  - p1_tgt_r1)  * 100
        delta_eer = (p1_tgt_eer - tgt_eer) * 100   # positive = improvement
        marker = (f"  [Tgt EER Δ{delta_eer:+.2f}%  R1 Δ{delta_r1:+.2f}%]"
                  if epoch % EVAL_EVERY == 0 else "")

        log(f"{epoch:>6}  {loss_t_m.avg:>8.4f}  {loss_s_m.avg:>7.4f}  "
            f"{loss_u_m.avg:>7.4f}  {loss_d_m.avg:>7.4f}  "
            f"{src_r1*100:>7.2f}%  {src_eer*100:>7.2f}%  "
            f"{tgt_r1*100:>7.2f}%  {tgt_eer*100:>7.2f}%{marker}")

        if tgt_eer < best_s2_tgt_eer:
            best_s2_tgt_eer = tgt_eer
            log(f"         ★ best target EER so far  ({tgt_eer*100:.2f}%)")
    else:
        log(f"{epoch:>6}  {loss_t_m.avg:>8.4f}  {loss_s_m.avg:>7.4f}  "
            f"{loss_u_m.avg:>7.4f}  {loss_d_m.avg:>7.4f}")

log("-" * 75)
log(f"\nFinal summary:")
log(f"  Phase 1 source  →  Src EER={best_s1_eer*100:.2f}%")
log(f"  Phase 1 target  →  Tgt R1={p1_tgt_r1*100:.2f}%  "
    f"Tgt EER={p1_tgt_eer*100:.2f}%")
log(f"  Phase 2 best    →  Tgt EER={best_s2_tgt_eer*100:.2f}%  "
    f"(Δ {(p1_tgt_eer-best_s2_tgt_eer)*100:+.2f}%)")
log("TSCAN v5 complete.")
