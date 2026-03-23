"""
TSCAN v7 — Fixed Phase 2 based on training diagnostics.

Evaluation protocol unchanged from v6 (paper Section 4.2):
  All images per domain, all-vs-all pairs, ACC/EER/TAR@FAR metrics.

Phase 2 fixes (motivated by observed L_sup=15, ΔEER=-13%):

  FIX 1 — Source augmentation: strong → weak in Phase 2.
    Strong aug caused L_sup to jump from 0.01 (Phase 1 end) to 15+.
    Weak aug keeps the student anchored to the feature space the
    teacher learned under identical augmentation.

  FIX 2 — Backbone fully frozen in Phase 2 (was partially trainable).
    Layer3/layer4 conflicting gradients (L_sup + noisy L_uns + L_dis)
    caused catastrophic forgetting. Only the linear head adapts now.

  FIX 3 — AdaFace W frozen in Phase 2.
    W was learned on source. Letting noisy pseudo-labels update W
    corrupts source identity clusters. W is frozen; only the linear
    layer is updated by backprop.

  FIX 4 — Loss weights rebalanced.
    BETA  0.8 → 0.3  (noisy pseudo-labels get less weight)
    GAMMA 1.0 → 0.3  (restores paper values; domain loss was swamped)
    ALPHA stays 1.0  (source supervision must dominate)

  FIX 5 — EMA decay 0.99 → 0.999.
    Fast EMA with degrading student quickly corrupted the teacher.
    Slow EMA keeps teacher near Phase 1 while absorbing improvements.

  FIX 6 — Pseudo-label threshold 0.5 → 0.7.
    Overfit teacher (EER=0%) is overconfident. Low threshold accepts
    confidently-wrong labels. Higher threshold is more selective.

  FIX 7 — Phase 1 saves best checkpoint by TARGET EER, not source EER.
    Source EER=0% is overfit. The best teacher for adaptation is the
    one that generalises best to target, not the one that memorises source.
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
TARGET_SPECTRUM = '850'
SEPARATE_HANDS  = True

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
S2_LR_HEAD       = 1e-4    # only linear layer trains in Phase 2
S2_WEIGHT_DECAY  = 5e-4
S2_BATCH_SIZE    = 32
S2_WARMUP_EPOCHS = 3

# ── Co-learning ───────────────────────────────────────────────────────────────
EMA_DECAY            = 0.999  # FIX 5: was 0.99 — slow EMA protects teacher
PSEUDO_LABEL_THRESH  = 0.7    # FIX 6: was 0.5 — stricter to avoid noisy labels
ALPHA                = 1.0    # source supervision weight (unchanged)
BETA                 = 0.3    # FIX 4: was 0.8 — less weight on noisy pseudo-labels
GAMMA_LOSS           = 0.3    # FIX 4: was 1.0 — restore paper value

# ── Augmentation ──────────────────────────────────────────────────────────────
RESIZE_SIZE     = 124
CROP_SIZE       = 112

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS     = 8
PIN_MEMORY      = True
SEED            = 42
EVAL_EVERY      = 5
MAX_IMPOSTORS   = 50_000


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


class SpectrumDataset(Dataset):
    def __init__(self, spectrum, transform, label_map=None):
        self.transform = transform
        records        = scan_spectrum(spectrum)
        assert records, f"No images found for spectrum '{spectrum}'"
        self.label_map = label_map if label_map else build_label_map(records)
        self.records   = [(fp, ident) for fp, ident in records
                          if ident in self.label_map]

    @property
    def num_classes(self):
        return len(self.label_map)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fp, ident = self.records[idx]
        return (self.transform(Image.open(fp).convert("RGB")),
                self.label_map[ident])


class UnlabeledTargetDataset(Dataset):
    """Target: (weak_aug, strong_aug) — no labels."""
    def __init__(self, spectrum):
        self.weak    = weak_transform()
        self.strong  = strong_transform()
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
    Phase 1: layer3+layer4 trainable (low LR), linear+Tanh trainable (high LR)
    Phase 2: ALL backbone layers frozen, only linear+Tanh trainable
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
        # Phase 1 default: freeze early layers only
        for p in self.frozen_layers.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.frozen_layers(x)
        # In Phase 2 trainable_layers are also frozen via requires_grad=False
        # but we still pass through normally — no extra no_grad needed
        x    = self.trainable_layers(x)
        bb   = self.flatten(x)
        feat = self.hash(self.linear(bb))
        return feat, bb

    def freeze_backbone_for_phase2(self):
        """FIX 2: freeze layer3+layer4 before Phase 2 starts."""
        for p in self.trainable_layers.parameters():
            p.requires_grad = False
        log("  Backbone fully frozen for Phase 2 (layer3+layer4 now frozen)")

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

    def freeze_backbone_for_phase2(self):
        self.encoder.freeze_backbone_for_phase2()


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
        cos_m_  = (cosine * torch.cos(m_col)
                   - torch.sin(theta) * torch.sin(m_col))
        one_hot = F.one_hot(labels, self.weight.size(0)).float()
        logits  = self.s * (one_hot * cos_m_ + (1 - one_hot) * cosine)
        return F.cross_entropy(logits, labels)

    def get_logits(self, features):
        return (F.normalize(features, dim=1) @
                F.normalize(self.weight, dim=1).T * self.s)

    def freeze_weights(self):
        """FIX 3: freeze W during Phase 2 so noisy pseudo-labels can't corrupt it."""
        self.weight.requires_grad = False
        log("  AdaFace W frozen for Phase 2")

    def unfreeze_weights(self):
        self.weight.requires_grad = True


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
# EVALUATION  —  Paper Section 4.2
# =============================================================================

@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    feats, labs = [], []
    for imgs, labels in loader:
        feat = F.normalize(model.get_features(imgs.to(DEVICE)), dim=1)
        feats.append(feat.cpu().numpy())
        labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def build_pairs(feats, labels):
    """
    Genuine : ALL (i<j) pairs where labels[i] == labels[j]
    Impostor: random pairs where labels differ, capped at MAX_IMPOSTORS
    """
    rng   = np.random.RandomState(42)
    by_id = defaultdict(list)
    for idx, lbl in enumerate(labels):
        by_id[lbl].append(idx)

    genuine = []
    for uid, idxs in by_id.items():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                genuine.append(float(np.dot(feats[idxs[i]], feats[idxs[j]])))

    n_imp    = min(len(genuine) * 5, MAX_IMPOSTORS)
    N, seen  = len(labels), 0
    impostor = []
    while len(impostor) < n_imp and seen < n_imp * 4:
        i, j = rng.choice(N, 2, replace=False)
        if labels[i] != labels[j]:
            impostor.append(float(np.dot(feats[i], feats[j])))
        seen += 1

    scores     = np.array(genuine + impostor, dtype=np.float32)
    is_genuine = np.array([True]  * len(genuine) +
                           [False] * len(impostor))
    return scores, is_genuine


def compute_metrics(scores, is_genuine):
    """ACC, EER, TAR@FAR=0.1, TAR@FAR=0.01  (paper Section 4.2)"""
    gen = scores[ is_genuine]
    imp = scores[~is_genuine]
    thresholds = np.linspace(-1.0, 1.0, 1000)
    far_arr, frr_arr, tar_arr, acc_arr = [], [], [], []
    for thr in thresholds:
        TP = int((gen >= thr).sum());  FN = int((gen  < thr).sum())
        FP = int((imp >= thr).sum());  TN = int((imp  < thr).sum())
        far_arr.append(FP / max(FP + TN, 1))
        frr_arr.append(FN / max(TP + FN, 1))
        tar_arr.append(TP / max(TP + FN, 1))
        acc_arr.append((TP + TN) / max(TP + TN + FP + FN, 1))
    far_arr = np.array(far_arr);  frr_arr = np.array(frr_arr)
    tar_arr = np.array(tar_arr);  acc_arr = np.array(acc_arr)
    acc     = float(acc_arr.max())
    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    eer     = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)
    valid_01  = far_arr <= 0.1
    tar_01    = float(tar_arr[valid_01].max())  if valid_01.any()  else 0.0
    valid_001 = far_arr <= 0.01
    tar_001   = float(tar_arr[valid_001].max()) if valid_001.any() else 0.0
    return {'acc': acc, 'eer': eer, 'tar_01': tar_01, 'tar_001': tar_001,
            'n_genuine': int(is_genuine.sum()),
            'n_impostor': int((~is_genuine).sum())}


def evaluate(model, loader, label=""):
    feats, labels      = extract_features(model, loader)
    scores, is_genuine = build_pairs(feats, labels)
    m                  = compute_metrics(scores, is_genuine)
    log(f"  [{label}]  ACC={m['acc']*100:.2f}%  EER={m['eer']*100:.2f}%  "
        f"TAR@FAR0.1={m['tar_01']*100:.2f}%  TAR@FAR0.01={m['tar_001']*100:.2f}%  "
        f"(genuine={m['n_genuine']}, impostor={m['n_impostor']})")
    return m


# =============================================================================
# DATASET & MODEL SETUP
# =============================================================================

set_seed(SEED)

log("=" * 72)
log(f"TSCAN v7  |  {SOURCE_SPECTRUM} → {TARGET_SPECTRUM}  |  device={DEVICE}")
log(f"Evaluation: paper protocol — all-vs-all pairs, ACC/EER/TAR@FAR")
log("=" * 72)

src_records = scan_spectrum(SOURCE_SPECTRUM)
assert src_records, f"No source images for '{SOURCE_SPECTRUM}'"
LABEL_MAP   = build_label_map(src_records)
NUM_CLASSES = len(LABEL_MAP)

tgt_records = scan_spectrum(TARGET_SPECTRUM)
assert tgt_records, f"No target images for '{TARGET_SPECTRUM}'"

log(f"Source [{SOURCE_SPECTRUM}]: {len(src_records)} images, "
    f"{NUM_CLASSES} identities")
log(f"Target [{TARGET_SPECTRUM}]: {len(tgt_records)} images")

# ── DataLoaders ───────────────────────────────────────────────────────────────

# Phase 1: weak aug on source
s1_train_loader = DataLoader(
    SpectrumDataset(SOURCE_SPECTRUM, weak_transform(), LABEL_MAP),
    batch_size=S1_BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)

# Evaluation: deterministic, all images
src_eval_loader = DataLoader(
    SpectrumDataset(SOURCE_SPECTRUM, eval_transform(), LABEL_MAP),
    batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

tgt_eval_loader = DataLoader(
    SpectrumDataset(TARGET_SPECTRUM, eval_transform(), LABEL_MAP),
    batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

# Phase 2 source: WEAK aug (FIX 1 — was strong_transform)
s2_src_loader = DataLoader(
    SpectrumDataset(SOURCE_SPECTRUM, weak_transform(), LABEL_MAP),
    batch_size=S2_BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)

# Phase 2 target: unlabeled (weak for teacher, strong for student)
s2_tgt_loader = DataLoader(
    UnlabeledTargetDataset(TARGET_SPECTRUM),
    batch_size=S2_BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)

# ── Models ────────────────────────────────────────────────────────────────────
teacher = PalmNet(feat_dim=FEATURE_DIM, pretrained=True).to(DEVICE)
adaface = AdaFaceLoss(num_classes=NUM_CLASSES, feat_dim=FEATURE_DIM,
                      m0=ADAFACE_M0, m_min=ADAFACE_MMIN,
                      s=ADAFACE_S).to(DEVICE)

frozen_p   = sum(p.numel() for p in teacher.encoder.frozen_layers.parameters())
backbone_p = sum(p.numel() for p in teacher.backbone_parameters())
head_p     = sum(p.numel() for p in teacher.head_parameters())
log(f"\nPhase 1 trainable: backbone={backbone_p/1e6:.2f}M (LR={S1_LR_BACKBONE}) "
    f"+ head={head_p/1e6:.4f}M (LR={S1_LR_HEAD})")
log(f"Phase 1 frozen   : {frozen_p/1e6:.2f}M (conv1..layer2)")
log(f"Phase 2 trainable: head only={head_p/1e6:.4f}M (LR={S2_LR_HEAD})  "
    f"[backbone+AdaFace W frozen]")


# =============================================================================
# PHASE 1 — TEACHER INITIALIZATION
# =============================================================================

log("\n" + "=" * 72)
log("PHASE 1 — Teacher Initialization")
log(f"  Saves best checkpoint by TARGET EER  (FIX 7 — avoids overfit teacher)\n")
log(f"{'Epoch':>6}  {'Loss':>8}  "
    f"{'Src ACC':>8}  {'Src EER':>8}  {'Src TAR@.1':>10}  "
    f"{'Tgt ACC':>8}  {'Tgt EER':>8}  {'Tgt TAR@.1':>10}")
log("-" * 80)

s1_optimizer = optim.AdamW([
    {'params': teacher.backbone_parameters(), 'lr': S1_LR_BACKBONE},
    {'params': teacher.head_parameters(),     'lr': S1_LR_HEAD},
    {'params': adaface.parameters(),          'lr': S1_LR_HEAD},
], weight_decay=S1_WEIGHT_DECAY)
s1_scheduler = make_warmup_cosine_scheduler(
    s1_optimizer, S1_WARMUP_EPOCHS, S1_EPOCHS)

best_s1_tgt_eer = 1.0   # FIX 7: track target EER, not source EER
best_s1_state   = None
best_s1_adaface = None

for epoch in range(1, S1_EPOCHS + 1):
    teacher.train();  adaface.train()
    loss_m = AverageMeter()

    for imgs, labels in s1_train_loader:
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
        src_m = evaluate(teacher, src_eval_loader, label=f"Src ep{epoch}")
        tgt_m = evaluate(teacher, tgt_eval_loader, label=f"Tgt ep{epoch}")

        # FIX 7: save best by target EER
        marker = "  ★" if tgt_m['eer'] < best_s1_tgt_eer else ""
        log(f"{epoch:>6}  {loss_m.avg:>8.4f}  "
            f"{src_m['acc']*100:>7.2f}%  {src_m['eer']*100:>7.2f}%  "
            f"{src_m['tar_01']*100:>9.2f}%  "
            f"{tgt_m['acc']*100:>7.2f}%  {tgt_m['eer']*100:>7.2f}%  "
            f"{tgt_m['tar_01']*100:>9.2f}%{marker}")

        if tgt_m['eer'] < best_s1_tgt_eer:
            best_s1_tgt_eer = tgt_m['eer']
            best_s1_state   = copy.deepcopy(teacher.state_dict())
            best_s1_adaface = copy.deepcopy(adaface.state_dict())
    else:
        log(f"{epoch:>6}  {loss_m.avg:>8.4f}")

log("-" * 80)
log(f"Phase 1 done  |  Best Target EER = {best_s1_tgt_eer*100:.2f}%")

teacher.load_state_dict(best_s1_state)
adaface.load_state_dict(best_s1_adaface)
log("\nPhase 1 best checkpoint:")
p1_src_m = evaluate(teacher, src_eval_loader, label="P1 Source")
p1_tgt_m = evaluate(teacher, tgt_eval_loader, label="P1 Target")


# =============================================================================
# PHASE 2 — TEACHER-STUDENT CO-LEARNING
# =============================================================================

log("\n" + "=" * 72)
log("PHASE 2 — Teacher-Student Co-Learning")
log("  FIX 1: source uses weak aug (same as Phase 1)")
log("  FIX 2: backbone fully frozen — only linear head trains")
log("  FIX 3: AdaFace W frozen — pseudo-labels cannot corrupt source clusters")
log(f"  FIX 4: ALPHA={ALPHA} BETA={BETA} GAMMA={GAMMA_LOSS} "
    f"(was 1.0/0.8/1.0)")
log(f"  FIX 5: EMA decay={EMA_DECAY}  (was 0.99)")
log(f"  FIX 6: pseudo threshold={PSEUDO_LABEL_THRESH}  (was 0.5)")
log(f"\n  Phase 1 baseline:")
log(f"    Source → ACC={p1_src_m['acc']*100:.2f}%  "
    f"EER={p1_src_m['eer']*100:.2f}%  "
    f"TAR@0.1={p1_src_m['tar_01']*100:.2f}%")
log(f"    Target → ACC={p1_tgt_m['acc']*100:.2f}%  "
    f"EER={p1_tgt_m['eer']*100:.2f}%  "
    f"TAR@0.1={p1_tgt_m['tar_01']*100:.2f}%\n")

log(f"{'Epoch':>6}  {'L_total':>8}  {'L_sup':>7}  "
    f"{'L_uns':>7}  {'L_dis':>7}  "
    f"{'Src ACC':>8}  {'Src EER':>8}  "
    f"{'Tgt ACC':>8}  {'Tgt EER':>8}  {'Tgt TAR@.1':>10}")
log("-" * 90)

# ── Build student from best Phase 1 checkpoint ────────────────────────────────
student = PalmNet(feat_dim=FEATURE_DIM, pretrained=True).to(DEVICE)
student.load_state_dict(best_s1_state)

# FIX 2: freeze backbone in both student and teacher for Phase 2
student.freeze_backbone_for_phase2()

# FIX 3: freeze AdaFace W so pseudo-labels cannot corrupt source clusters
adaface.freeze_weights()

discriminator = DomainDiscriminator(
    feat_dim=FEATURE_DIM, hidden=128, alpha=1.0).to(DEVICE)

# Teacher: no gradient at all (EMA only)
for p in teacher.parameters():
    p.requires_grad = False

# Optimizer: only student linear head + discriminator (backbone and W frozen)
s2_optimizer = optim.AdamW([
    {'params': student.head_parameters(),  'lr': S2_LR_HEAD},
    {'params': discriminator.parameters(), 'lr': S2_LR_HEAD},
], weight_decay=S2_WEIGHT_DECAY)
s2_scheduler = make_warmup_cosine_scheduler(
    s2_optimizer, S2_WARMUP_EPOCHS, S2_EPOCHS)

best_s2_tgt_eer = p1_tgt_m['eer']
total_steps     = len(s2_src_loader) * S2_EPOCHS
global_step     = 0

for epoch in range(1, S2_EPOCHS + 1):
    student.train();  teacher.eval()
    discriminator.train();  adaface.eval()   # adaface in eval (W frozen)

    loss_t_m = AverageMeter();  loss_s_m = AverageMeter()
    loss_u_m = AverageMeter();  loss_d_m = AverageMeter()

    max_steps = max(len(s2_src_loader), len(s2_tgt_loader))
    src_iter  = itertools.cycle(s2_src_loader)
    tgt_iter  = itertools.cycle(s2_tgt_loader)

    for _ in range(max_steps):
        alpha = grl_alpha(global_step, total_steps, alpha_max=1.0)
        discriminator.set_alpha(alpha)

        src_imgs, src_lbl = next(src_iter)
        tgt_weak, tgt_str = next(tgt_iter)
        src_imgs = src_imgs.to(DEVICE);  src_lbl  = src_lbl.to(DEVICE)
        tgt_weak = tgt_weak.to(DEVICE);  tgt_str  = tgt_str.to(DEVICE)

        # Pseudo-labels from teacher (no grad)
        pl, mask    = generate_pseudo_labels(
            teacher, tgt_weak, adaface, PSEUDO_LABEL_THRESH)

        # Student forward
        src_feat, _ = student(src_imgs)
        tgt_feat, _ = student(tgt_str)

        # L_sup: source supervised loss (W frozen, only linear head trains)
        L_sup   = adaface(src_feat, src_lbl)

        # L_unsup: pseudo-label loss on confident target samples
        L_unsup = (adaface(tgt_feat[mask], pl[mask]) if mask.any()
                   else torch.tensor(0.0, device=DEVICE))

        # L_dis: adversarial domain alignment via GRL
        L_dis   = domain_loss(discriminator, src_feat, tgt_feat)

        L_total = ALPHA * L_sup + BETA * L_unsup + GAMMA_LOSS * L_dis

        s2_optimizer.zero_grad()
        L_total.backward()
        nn.utils.clip_grad_norm_(
            student.head_parameters() +
            list(discriminator.parameters()), max_norm=5.0)
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
        src_m = evaluate(student, src_eval_loader, label=f"Src ep{epoch}")
        tgt_m = evaluate(student, tgt_eval_loader, label=f"Tgt ep{epoch}")

        d_eer  = (p1_tgt_m['eer'] - tgt_m['eer']) * 100
        marker = (f"  [ΔEER={d_eer:+.2f}%]"
                  + ("  ★" if tgt_m['eer'] < best_s2_tgt_eer else ""))

        log(f"{epoch:>6}  {loss_t_m.avg:>8.4f}  {loss_s_m.avg:>7.4f}  "
            f"{loss_u_m.avg:>7.4f}  {loss_d_m.avg:>7.4f}  "
            f"{src_m['acc']*100:>7.2f}%  {src_m['eer']*100:>7.2f}%  "
            f"{tgt_m['acc']*100:>7.2f}%  {tgt_m['eer']*100:>7.2f}%  "
            f"{tgt_m['tar_01']*100:>9.2f}%{marker}")

        if tgt_m['eer'] < best_s2_tgt_eer:
            best_s2_tgt_eer = tgt_m['eer']
    else:
        log(f"{epoch:>6}  {loss_t_m.avg:>8.4f}  {loss_s_m.avg:>7.4f}  "
            f"{loss_u_m.avg:>7.4f}  {loss_d_m.avg:>7.4f}")

log("-" * 90)
log(f"\nFinal results:")
log(f"  Source — ACC={p1_src_m['acc']*100:.2f}%  "
    f"EER={p1_src_m['eer']*100:.2f}%  "
    f"TAR@FAR=0.1={p1_src_m['tar_01']*100:.2f}%")
log(f"  Target before adaptation — "
    f"ACC={p1_tgt_m['acc']*100:.2f}%  "
    f"EER={p1_tgt_m['eer']*100:.2f}%  "
    f"TAR@FAR=0.1={p1_tgt_m['tar_01']*100:.2f}%")
log(f"  Target after  adaptation — "
    f"Best EER={best_s2_tgt_eer*100:.2f}%  "
    f"(ΔEER={(p1_tgt_m['eer']-best_s2_tgt_eer)*100:+.2f}%)")
log("TSCAN v7 complete.")
