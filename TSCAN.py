"""
TSCAN: Teacher-Student Co-Learning Adaptive Network
for Cross-Spectrum Palmprint Recognition on CASIA-MS

Single-file implementation following paper structure exactly.
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
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T


# =============================================================================
# PARAMETERS
# =============================================================================

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT        = "/home/pai-ng/Jamal/CASIA-MS-ROI"
SAVE_DIR         = "./tscan_checkpoints"
LOG_DIR          = "./tscan_logs"

# ── Dataset ──────────────────────────────────────────────────────────────────
# Filename format: {subject}_{hand}_{spectrum}_{iteration}.jpg
# Spectrums are the 6 domains in CASIA-MS
ALL_SPECTRUMS    = ['460', '630', '700', '850', '940', 'White']
SOURCE_SPECTRUM  = '460'      # source domain spectrum
TARGET_SPECTRUM  = '630'      # target domain spectrum
SEPARATE_HANDS   = True       # True → subject_L ≠ subject_R (more identities)

# ── Model ─────────────────────────────────────────────────────────────────────
FEATURE_DIM      = 128        # hash layer output dimension
BACKBONE_DIM     = 512        # ResNet18 avgpool output

# ── AdaFace Loss (Section 3.1) ────────────────────────────────────────────────
ADAFACE_M0       = 0.5        # base / maximum margin
ADAFACE_MMIN     = 0.25       # minimum margin for low-quality samples
ADAFACE_S        = 64.0       # feature scale factor

# ── Stage 1 — Teacher Initialization ─────────────────────────────────────────
S1_EPOCHS        = 60
S1_LR            = 1e-3
S1_WEIGHT_DECAY  = 5e-4
S1_BATCH_SIZE    = 64
S1_LR_MILESTONES = [30, 50]
S1_LR_GAMMA      = 0.1

# ── Stage 2 — Teacher-Student Co-Learning ────────────────────────────────────
S2_EPOCHS        = 50
S2_LR            = 1e-4
S2_WEIGHT_DECAY  = 5e-4
S2_BATCH_SIZE    = 32
S2_LR_MILESTONES = [25, 40]
S2_LR_GAMMA      = 0.1

EMA_DECAY             = 0.99   # λ: θ_teacher ← λ·θ_teacher + (1−λ)·θ_student
PSEUDO_LABEL_THRESH   = 0.8    # confidence threshold for pseudo-label acceptance
ALPHA                 = 1.0    # weight for L_sup
BETA                  = 0.8    # weight for L_unsup
GAMMA_LOSS            = 0.3    # weight for L_dis

# ── Augmentation ─────────────────────────────────────────────────────────────
RESIZE_SIZE      = 124
CROP_SIZE        = 112

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS      = 8
PIN_MEMORY       = True
SEED             = 42

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_EVERY       = 5           # evaluate every N epochs in Stage 2
MAX_PAIRS        = 500_000     # cap on impostor pairs during evaluation


# =============================================================================
# FUNCTIONS AND CLASSES
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def log(msg, also_print=True):
    """Simple logger that writes to file and optionally prints."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "tscan_run.log")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line)
    with open(log_path, "a") as f:
        f.write(line + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# AverageMeter
# ─────────────────────────────────────────────────────────────────────────────
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum   += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation Transforms  (Section 3.2.1)
# ─────────────────────────────────────────────────────────────────────────────
class GaussianNoise:
    def __init__(self, std=0.02):
        self.std = std
    def __call__(self, t):
        return (t + torch.randn_like(t) * self.std).clamp(0., 1.)


def weak_transform():
    """
    Teacher input / pseudo-label generation.
    Preserves domain style; ensures reliable pseudo-labels.
    resize 124 → crop 112 → hflip → rotate
    """
    return T.Compose([
        T.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        T.RandomCrop(CROP_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def strong_transform():
    """
    Student input. Expands distribution, forces learning of core textures.
    Adds color jitter, contrast, blur, grayscale, Gaussian noise on top of weak.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Utilities
# ─────────────────────────────────────────────────────────────────────────────
def parse_filename(filepath, separate_hands=True):
    """
    Parse {subject}_{hand}_{spectrum}_{iteration}.jpg
    Returns (identity_key, spectrum) or (None, None).
    """
    base   = os.path.splitext(os.path.basename(filepath))[0]
    parts  = base.split('_')
    if len(parts) < 4:
        return None, None
    subject  = parts[0]
    hand     = parts[1]
    spectrum = parts[2]           # e.g. '460', '630', 'White'
    identity = f"{subject}_{hand}" if separate_hands else subject
    return identity, spectrum


def scan_spectrum(spectrum, separate_hands=True):
    """Return sorted list of (filepath, identity_str) for one spectrum."""
    files = sorted(glob.glob(os.path.join(DATA_ROOT, "*.jpg")))
    if not files:
        files = sorted(glob.glob(os.path.join(DATA_ROOT, "**", "*.jpg"),))
    records = []
    for fp in files:
        identity, spec = parse_filename(fp, separate_hands)
        if identity is not None and spec == spectrum:
            records.append((fp, identity))
    return records


def build_label_map(records):
    """Map sorted identity strings to integer indices."""
    identities = sorted(set(r[1] for r in records))
    return {name: idx for idx, name in enumerate(identities)}


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────
class LabeledDataset(Dataset):
    """Single-spectrum labeled dataset (Stage 1 + source side of Stage 2)."""

    def __init__(self, spectrum, transform, label_map=None):
        self.transform = transform
        records        = scan_spectrum(spectrum, SEPARATE_HANDS)
        assert records, f"No images found for spectrum '{spectrum}' in {DATA_ROOT}"
        self.label_map = label_map if label_map else build_label_map(records)
        self.records   = [(fp, ident) for fp, ident in records
                          if ident in self.label_map]
        print(f"  LabeledDataset  [{spectrum}]: "
              f"{len(self.records)} images, {len(self.label_map)} identities")

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
    """Target domain: returns (weak_aug, strong_aug) — no labels."""

    def __init__(self, spectrum):
        self.weak   = weak_transform()
        self.strong = strong_transform()
        records     = scan_spectrum(spectrum, SEPARATE_HANDS)
        assert records, f"No target images found for spectrum '{spectrum}'"
        self.records = records
        print(f"  UnlabeledDataset [{spectrum}]: {len(self.records)} images (unlabeled)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fp, _ = self.records[idx]
        img   = Image.open(fp).convert("RGB")
        return self.weak(img), self.strong(img)


# ─────────────────────────────────────────────────────────────────────────────
# AdaFace Loss  (Section 3.1)
# ─────────────────────────────────────────────────────────────────────────────
class AdaFaceLoss(nn.Module):
    """
    Adaptive Margin Softmax Loss.
    Feature norm ‖z_i‖ acts as quality proxy.
    High-quality → larger margin (tighter clusters).
    Low-quality  → smaller margin (avoid over-penalty).

    m(‖z‖) linearly scales from m_min (lowest norm) to m0 (highest norm).
    L = -log [ e^{s·cos(θ_yi + m)} / (e^{s·cos(θ_yi + m)} + Σ_{j≠y} e^{s·cosθ_j}) ]
    """

    def __init__(self, num_classes, feat_dim=128,
                 m0=0.5, m_min=0.25, s=64.0):
        super().__init__()
        self.m0          = m0
        self.m_min       = m_min
        self.s           = s
        self.weight      = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def _adaptive_margin(self, norms):
        """Per-sample margin ∈ [m_min, m0] based on feature norm."""
        lo    = norms.min().detach()
        hi    = norms.max().detach()
        denom = (hi - lo).clamp(min=1e-8)
        return (self.m_min + (self.m0 - self.m_min) * (norms - lo) / denom
                ).clamp(self.m_min, self.m0)

    def forward(self, features, labels):
        norms   = features.norm(dim=1)                              # (B,)
        margins = self._adaptive_margin(norms)                      # (B,)

        feat_n  = F.normalize(features, dim=1)                      # (B, D)
        w_n     = F.normalize(self.weight, dim=1)                   # (C, D)
        cosine  = (feat_n @ w_n.T).clamp(-1 + 1e-7, 1 - 1e-7)     # (B, C)

        # cos(θ + m) = cosθ·cosm − sinθ·sinm
        theta   = torch.acos(cosine)                                # (B, C)
        m_col   = margins.unsqueeze(1)                              # (B, 1)
        cos_m   = torch.cos(m_col)
        sin_m   = torch.sin(m_col)
        cos_m_  = cosine * cos_m - torch.sin(theta) * sin_m        # (B, C)

        one_hot = F.one_hot(labels, self.weight.size(0)).float()
        logits  = one_hot * cos_m_ + (1 - one_hot) * cosine        # (B, C)
        logits  = self.s * logits
        return F.cross_entropy(logits, labels)

    def get_logits(self, features):
        """Cosine logits for accuracy/EER computation (no margin)."""
        feat_n = F.normalize(features, dim=1)
        w_n    = F.normalize(self.weight, dim=1)
        return feat_n @ w_n.T * self.s


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture  (Section 3.1, Figure 1)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureEncoder(nn.Module):
    """
    ResNet18 (no FC) → Flatten → Linear(512→128) → Tanh
    Outputs 128-dim hash features z_i ∈ (−1, 1).
    Feature norm ‖z_i‖ is used as quality proxy by AdaFace.
    """
    def __init__(self, feat_dim=128, pretrained=True):
        super().__init__()
        weights        = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone       = resnet18(weights=weights)
        self.backbone  = nn.Sequential(*list(backbone.children())[:-1])  # → (B,512,1,1)
        self.flatten   = nn.Flatten()
        self.linear    = nn.Linear(512, feat_dim, bias=True)             # linear layer
        self.hash      = nn.Tanh()                                        # hash layer

    def forward(self, x):
        bb   = self.flatten(self.backbone(x))   # (B, 512)
        feat = self.hash(self.linear(bb))        # (B, 128)
        return feat, bb


class PalmNet(nn.Module):
    """Complete palmprint recognition model (teacher or student)."""
    def __init__(self, feat_dim=128, pretrained=True):
        super().__init__()
        self.encoder = FeatureEncoder(feat_dim=feat_dim, pretrained=pretrained)

    def forward(self, x):
        return self.encoder(x)      # returns (feat, backbone_feat)

    def get_features(self, x):
        feat, _ = self.encoder(x)
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Reversal Layer  (Section 3.2.3)
# ─────────────────────────────────────────────────────────────────────────────
class GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GRLFunction.apply(x, self.alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Domain Discriminator  (Section 3.2.3, Figure 5)
# ─────────────────────────────────────────────────────────────────────────────
class DomainDiscriminator(nn.Module):
    """
    GRL → FC(128→64) → BN → ReLU → Dropout → FC(64→1) → Sigmoid
    Distinguishes source (0) from target (1).
    """
    def __init__(self, feat_dim=128, hidden=64, alpha=1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        return self.net(self.grl(feat))

    def set_alpha(self, alpha):
        self.grl.alpha = alpha


# ─────────────────────────────────────────────────────────────────────────────
# EMA Update  (Section 3.2.2)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def ema_update(teacher, student, decay=0.99):
    """θ_teacher ← λ·θ_teacher + (1−λ)·θ_student"""
    for t_p, s_p in zip(teacher.parameters(), student.parameters()):
        t_p.data.mul_(decay).add_(s_p.data * (1.0 - decay))


# ─────────────────────────────────────────────────────────────────────────────
# GRL Alpha Schedule
# ─────────────────────────────────────────────────────────────────────────────
def grl_alpha(cur_iter, max_iter, alpha_max=1.0):
    """Gradually increase GRL strength: 0 → alpha_max (Ganin & Lempitsky 2015)."""
    p = cur_iter / max(max_iter, 1)
    return float(alpha_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-Label Generation  (Section 3.2.2)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_pseudo_labels(teacher, weak_imgs, adaface, threshold=0.8):
    """
    Teacher generates pseudo-labels for unlabeled target images.
    Only accepted when max softmax probability >= threshold (0.8).
    Returns (pseudo_labels, mask) where mask[i]=True means accepted.
    """
    teacher.eval()
    feat, _   = teacher(weak_imgs)
    feat_n    = F.normalize(feat, dim=1)
    w_n       = F.normalize(adaface.weight, dim=1)
    probs     = F.softmax(feat_n @ w_n.T * adaface.s, dim=1)   # (B, C)
    max_p, pl = probs.max(dim=1)
    mask      = max_p >= threshold
    pl[~mask] = -1
    return pl, mask


# ─────────────────────────────────────────────────────────────────────────────
# Domain Loss  (Section 3.2.3)
# ─────────────────────────────────────────────────────────────────────────────
def domain_loss(discriminator, src_feat, tgt_feat):
    """
    L_dis = (1/N+M) Σ [d_i·log(q_i) + (1−d_i)·log(1−q_i)]
    source → label 0, target → label 1
    """
    N, M      = src_feat.size(0), tgt_feat.size(0)
    src_lbl   = torch.zeros(N, 1, device=src_feat.device)
    tgt_lbl   = torch.ones (M, 1, device=tgt_feat.device)
    all_feat  = torch.cat([src_feat, tgt_feat], dim=0)
    all_lbl   = torch.cat([src_lbl,  tgt_lbl],  dim=0)
    preds     = discriminator(all_feat)
    return F.binary_cross_entropy(preds, all_lbl)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: ACC, EER, TAR@FAR  (Section 4.2)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    feats, labs = [], []
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        feat   = model.get_features(imgs)
        feat   = F.normalize(feat, dim=1)
        feats.append(feat.cpu().numpy())
        labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def build_pairs(features, labels):
    """Build genuine + impostor cosine similarity scores."""
    rng = np.random.RandomState(42)
    genuine_scores, impostor_scores = [], []

    for uid in np.unique(labels):
        idx = np.where(labels == uid)[0]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                s = float(np.dot(features[idx[i]], features[idx[j]]))
                genuine_scores.append(s)

    n_imp = min(len(genuine_scores) * 5, MAX_PAIRS)
    N     = len(labels)
    seen  = 0
    while len(impostor_scores) < n_imp and seen < n_imp * 3:
        i, j = rng.choice(N, 2, replace=False)
        if labels[i] != labels[j]:
            impostor_scores.append(float(np.dot(features[i], features[j])))
        seen += 1

    scores    = np.array(genuine_scores + impostor_scores, dtype=np.float32)
    is_genuine = np.array([True] * len(genuine_scores) +
                           [False] * len(impostor_scores))
    return scores, is_genuine


def tar_at_far(far_arr, tar_arr, target_far):
    valid = far_arr <= target_far
    if not valid.any():
        return 0.0
    idx = np.where(valid)[0]
    return float(tar_arr[idx[np.argmax(tar_arr[idx])]])


def compute_metrics(scores, is_genuine):
    """
    Returns dict: acc, eer, tar_far01, tar_far001
    Uses 1000 thresholds over [-1, 1].
    """
    thresholds = np.linspace(-1.0, 1.0, 1000)
    gen = scores[ is_genuine]
    imp = scores[~is_genuine]

    far_arr, frr_arr, tar_arr, acc_arr = [], [], [], []
    for thr in thresholds:
        TP = int((gen >= thr).sum());  FN = int((gen <  thr).sum())
        FP = int((imp >= thr).sum());  TN = int((imp <  thr).sum())
        far_arr.append(FP / max(FP + TN, 1))
        frr_arr.append(FN / max(TP + FN, 1))
        tar_arr.append(TP / max(TP + FN, 1))
        acc_arr.append((TP + TN) / max(TP + TN + FP + FN, 1))

    far_arr = np.array(far_arr)
    frr_arr = np.array(frr_arr)
    tar_arr = np.array(tar_arr)
    acc_arr = np.array(acc_arr)

    best_acc   = float(acc_arr.max())
    diff       = np.abs(far_arr - frr_arr)
    eer_idx    = np.argmin(diff)
    eer        = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)
    tar_far01  = tar_at_far(far_arr, tar_arr, 0.1)
    tar_far001 = tar_at_far(far_arr, tar_arr, 0.01)

    return {
        "acc": best_acc, "eer": eer,
        "tar_far01": tar_far01, "tar_far001": tar_far001,
    }


def evaluate(model, loader, split=""):
    """Full eval pipeline: extract → pairs → metrics. Returns metrics dict."""
    features, labels = extract_features(model, loader)
    scores, is_genuine = build_pairs(features, labels)
    m = compute_metrics(scores, is_genuine)
    log(f"  [{split}]  ACC={m['acc']*100:.2f}%  EER={m['eer']*100:.2f}%  "
        f"TAR@FAR0.1={m['tar_far01']*100:.2f}%  "
        f"TAR@FAR0.01={m['tar_far001']*100:.2f}%  "
        f"(genuine={is_genuine.sum()}, impostor={(~is_genuine).sum()})")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Top-1 Accuracy helper (training loop quick estimate)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def batch_accuracy(logits, labels):
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def save(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load(path):
    return torch.load(path, map_location=DEVICE)


# =============================================================================
# DATASET AND MODEL LOADING / SETUP
# =============================================================================

set_seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

log("=" * 70)
log("TSCAN — Cross-Spectrum Palmprint Recognition on CASIA-MS")
log(f"  Source: {SOURCE_SPECTRUM}  →  Target: {TARGET_SPECTRUM}")
log(f"  Device: {DEVICE}")
log("=" * 70)

# ── Stage 1 labeled source dataset ──────────────────────────────────────────
log("\n[Setup] Building datasets...")
s1_train_dataset = LabeledDataset(SOURCE_SPECTRUM, weak_transform())
NUM_CLASSES      = s1_train_dataset.num_classes
LABEL_MAP        = s1_train_dataset.label_map

s1_train_loader  = DataLoader(
    s1_train_dataset,
    batch_size  = S1_BATCH_SIZE,
    shuffle     = True,
    num_workers = NUM_WORKERS,
    pin_memory  = PIN_MEMORY,
    drop_last   = True,
)

# ── Stage 1 eval loaders (source and target, deterministic) ─────────────────
src_eval_dataset = LabeledDataset(SOURCE_SPECTRUM, eval_transform(), LABEL_MAP)
tgt_eval_dataset = LabeledDataset(TARGET_SPECTRUM, eval_transform(), LABEL_MAP)

src_eval_loader  = DataLoader(src_eval_dataset, batch_size=128,
                               shuffle=False, num_workers=NUM_WORKERS)
tgt_eval_loader  = DataLoader(tgt_eval_dataset, batch_size=128,
                               shuffle=False, num_workers=NUM_WORKERS)

# ── Stage 2 loaders (source labeled strong aug + target unlabeled) ───────────
s2_src_dataset  = LabeledDataset(SOURCE_SPECTRUM, strong_transform(), LABEL_MAP)
s2_tgt_dataset  = UnlabeledTargetDataset(TARGET_SPECTRUM)

s2_src_loader   = DataLoader(s2_src_dataset, batch_size=S2_BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, drop_last=True)
s2_tgt_loader   = DataLoader(s2_tgt_dataset, batch_size=S2_BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, drop_last=True)

# ── Teacher model + AdaFace loss ─────────────────────────────────────────────
log(f"\n[Setup] Building models (num_classes={NUM_CLASSES})...")
teacher  = PalmNet(feat_dim=FEATURE_DIM, pretrained=True).to(DEVICE)
adaface  = AdaFaceLoss(
    num_classes = NUM_CLASSES,
    feat_dim    = FEATURE_DIM,
    m0          = ADAFACE_M0,
    m_min       = ADAFACE_MMIN,
    s           = ADAFACE_S,
).to(DEVICE)

log(f"  Teacher      : ResNet18 → Linear(512→{FEATURE_DIM}) → Tanh")
log(f"  AdaFace      : m0={ADAFACE_M0}, m_min={ADAFACE_MMIN}, s={ADAFACE_S}")
log(f"  Parameters   : "
    f"{sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M (teacher) + "
    f"{sum(p.numel() for p in adaface.parameters()) / 1e6:.3f}M (AdaFace head)")

STAGE1_CKPT = os.path.join(SAVE_DIR, "teacher_stage1_best.pth")
STAGE2_CKPT = os.path.join(SAVE_DIR, "student_stage2_best.pth")


# =============================================================================
# PHASE 1 — TEACHER INITIALIZATION  (Section 3.1)
# =============================================================================
log("\n" + "=" * 70)
log("PHASE 1 — Teacher Initialization (AdaFace on Source Domain)")
log(f"  Epochs={S1_EPOCHS}, LR={S1_LR}, Batch={S1_BATCH_SIZE}")
log("=" * 70)

s1_optimizer = optim.Adam(
    list(teacher.parameters()) + list(adaface.parameters()),
    lr=S1_LR, weight_decay=S1_WEIGHT_DECAY,
)
s1_scheduler = MultiStepLR(s1_optimizer, milestones=S1_LR_MILESTONES,
                            gamma=S1_LR_GAMMA)

best_s1_acc = 0.0

for epoch in range(1, S1_EPOCHS + 1):
    teacher.train()
    adaface.train()

    loss_m = AverageMeter()
    acc_m  = AverageMeter()
    t0     = time.time()

    for imgs, labels in s1_train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        feat, _  = teacher(imgs)
        loss     = adaface(feat, labels)

        s1_optimizer.zero_grad()
        loss.backward()
        s1_optimizer.step()

        with torch.no_grad():
            logits = adaface.get_logits(feat)
            acc    = batch_accuracy(logits, labels)

        bs = imgs.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update(acc, bs)

    s1_scheduler.step()

    # ── Per-epoch metrics ────────────────────────────────────────────────
    train_loss = loss_m.avg
    train_acc  = acc_m.avg

    # EER on source train features (quick, no pair expansion)
    src_m = evaluate(teacher, src_eval_loader, split="S1 Src-Eval")
    tgt_m = evaluate(teacher, tgt_eval_loader, split="S1 Tgt-Eval")

    log(f"\n[Phase1 | Epoch {epoch:3d}/{S1_EPOCHS}]  "
        f"Train Loss={train_loss:.4f}  Train Acc={train_acc*100:.2f}%  "
        f"LR={s1_scheduler.get_last_lr()[0]:.2e}  "
        f"Time={time.time()-t0:.1f}s")
    log(f"  Src → ACC={src_m['acc']*100:.2f}%  EER={src_m['eer']*100:.2f}%  "
        f"TAR@0.1={src_m['tar_far01']*100:.2f}%  TAR@0.01={src_m['tar_far001']*100:.2f}%")
    log(f"  Tgt → ACC={tgt_m['acc']*100:.2f}%  EER={tgt_m['eer']*100:.2f}%  "
        f"TAR@0.1={tgt_m['tar_far01']*100:.2f}%  TAR@0.01={tgt_m['tar_far001']*100:.2f}%")

    # ── Save best checkpoint ─────────────────────────────────────────────
    if src_m['acc'] > best_s1_acc:
        best_s1_acc = src_m['acc']
        save({
            "epoch"             : epoch,
            "model_state_dict"  : teacher.state_dict(),
            "adaface_state_dict": adaface.state_dict(),
            "src_metrics"       : src_m,
            "tgt_metrics"       : tgt_m,
            "label_map"         : LABEL_MAP,
            "num_classes"       : NUM_CLASSES,
        }, STAGE1_CKPT)
        log(f"  ★ New best Src ACC={best_s1_acc*100:.2f}% — saved to {STAGE1_CKPT}")

log(f"\n[Phase 1 Complete]  Best Source ACC = {best_s1_acc*100:.2f}%")


# =============================================================================
# PHASE 2 — TEACHER-STUDENT CO-LEARNING  (Section 3.2)
# =============================================================================
log("\n" + "=" * 70)
log("PHASE 2 — Teacher-Student Co-Learning (Cross-Domain Adaptation)")
log(f"  Epochs={S2_EPOCHS}, LR={S2_LR}, Batch={S2_BATCH_SIZE}")
log(f"  EMA λ={EMA_DECAY}  Threshold={PSEUDO_LABEL_THRESH}")
log(f"  Loss weights: α={ALPHA} β={BETA} γ={GAMMA_LOSS}")
log("=" * 70)

# ── Initialize student from best Stage-1 teacher ────────────────────────────
ckpt    = load(STAGE1_CKPT)
teacher.load_state_dict(ckpt["model_state_dict"])
adaface.load_state_dict(ckpt["adaface_state_dict"])

student      = PalmNet(feat_dim=FEATURE_DIM, pretrained=True).to(DEVICE)
student.load_state_dict(ckpt["model_state_dict"])   # student starts = teacher

discriminator = DomainDiscriminator(
    feat_dim=FEATURE_DIM, hidden=64, alpha=1.0
).to(DEVICE)

# Teacher parameters: no gradient (EMA only)
for p in teacher.parameters():
    p.requires_grad = False

log(f"  Teacher & Student initialised from Phase-1 checkpoint.")
log(f"  Discriminator: GRL → FC(128→64) → BN → ReLU → Dropout → FC→Sigmoid")

# ── Optimizer (student + discriminator + adaface only) ──────────────────────
s2_optimizer = optim.Adam(
    list(student.parameters()) +
    list(discriminator.parameters()) +
    list(adaface.parameters()),
    lr=S2_LR, weight_decay=S2_WEIGHT_DECAY,
)
s2_scheduler = MultiStepLR(s2_optimizer, milestones=S2_LR_MILESTONES,
                            gamma=S2_LR_GAMMA)

best_s2_tgt_acc = 0.0
total_steps     = len(s2_src_loader) * S2_EPOCHS
global_step     = 0

for epoch in range(1, S2_EPOCHS + 1):
    student.train()
    teacher.eval()
    discriminator.train()
    adaface.train()

    loss_total_m  = AverageMeter()
    loss_sup_m    = AverageMeter()
    loss_unsup_m  = AverageMeter()
    loss_dis_m    = AverageMeter()
    pseudo_ratio_m = AverageMeter()
    train_acc_m   = AverageMeter()

    max_steps = max(len(s2_src_loader), len(s2_tgt_loader))
    src_iter  = itertools.cycle(s2_src_loader)
    tgt_iter  = itertools.cycle(s2_tgt_loader)
    t0        = time.time()

    for step in range(max_steps):
        # ── GRL alpha schedule ─────────────────────────────────────────
        alpha = grl_alpha(global_step, total_steps, alpha_max=1.0)
        discriminator.set_alpha(alpha)

        # ── Fetch batches ──────────────────────────────────────────────
        src_imgs, src_lbl = next(src_iter)
        tgt_weak, tgt_str = next(tgt_iter)

        src_imgs = src_imgs.to(DEVICE)
        src_lbl  = src_lbl.to(DEVICE)
        tgt_weak = tgt_weak.to(DEVICE)
        tgt_str  = tgt_str.to(DEVICE)

        # ── Pseudo-label generation (teacher, no grad) ─────────────────
        pl, mask = generate_pseudo_labels(
            teacher, tgt_weak, adaface, PSEUDO_LABEL_THRESH
        )
        pseudo_ratio_m.update(mask.float().mean().item(), tgt_weak.size(0))

        # ── Student forward ────────────────────────────────────────────
        src_feat, _ = student(src_imgs)     # (N, 128)
        tgt_feat, _ = student(tgt_str)      # (M, 128)

        # ── L_sup : supervised source loss ─────────────────────────────
        L_sup = adaface(src_feat, src_lbl)

        # ── L_unsup : pseudo-label target loss ─────────────────────────
        if mask.any():
            L_unsup = adaface(tgt_feat[mask], pl[mask])
        else:
            L_unsup = torch.tensor(0.0, device=DEVICE)

        # ── L_dis : adversarial domain loss ────────────────────────────
        L_dis = domain_loss(discriminator, src_feat, tgt_feat)

        # ── Total loss: α·Lsup + β·Lunsup + γ·Ldis ────────────────────
        L_total = ALPHA * L_sup + BETA * L_unsup + GAMMA_LOSS * L_dis

        # ── Backprop ───────────────────────────────────────────────────
        s2_optimizer.zero_grad()
        L_total.backward()
        nn.utils.clip_grad_norm_(
            list(student.parameters()) +
            list(discriminator.parameters()) +
            list(adaface.parameters()),
            max_norm=5.0,
        )
        s2_optimizer.step()

        # ── EMA teacher update ─────────────────────────────────────────
        ema_update(teacher, student, decay=EMA_DECAY)

        # ── Training accuracy estimate (source batch) ──────────────────
        with torch.no_grad():
            logits = adaface.get_logits(src_feat)
            acc    = batch_accuracy(logits, src_lbl)

        bs = src_imgs.size(0)
        loss_total_m.update(L_total.item(), bs)
        loss_sup_m.update(L_sup.item(),     bs)
        loss_unsup_m.update(L_unsup.item() if mask.any() else 0., bs)
        loss_dis_m.update(L_dis.item(),     bs)
        train_acc_m.update(acc,             bs)
        global_step += 1

    s2_scheduler.step()

    # ── Per-epoch training summary ───────────────────────────────────────
    log(f"\n[Phase2 | Epoch {epoch:3d}/{S2_EPOCHS}]  "
        f"L_total={loss_total_m.avg:.4f}  "
        f"L_sup={loss_sup_m.avg:.4f}  "
        f"L_unsup={loss_unsup_m.avg:.4f}  "
        f"L_dis={loss_dis_m.avg:.4f}  "
        f"Train Acc={train_acc_m.avg*100:.2f}%  "
        f"PseudoAccepted={pseudo_ratio_m.avg*100:.1f}%  "
        f"GRL_α={alpha:.3f}  "
        f"LR={s2_scheduler.get_last_lr()[0]:.2e}  "
        f"Time={time.time()-t0:.1f}s")

    # ── Evaluation every EVAL_EVERY epochs ──────────────────────────────
    if epoch % EVAL_EVERY == 0 or epoch == S2_EPOCHS:
        log(f"  → Student evaluation at epoch {epoch}:")
        src_m = evaluate(student, src_eval_loader, split="S2 Src")
        tgt_m = evaluate(student, tgt_eval_loader, split="S2 Tgt")

        log(f"  [Student | Source]  "
            f"ACC={src_m['acc']*100:.2f}%  EER={src_m['eer']*100:.2f}%  "
            f"TAR@0.1={src_m['tar_far01']*100:.2f}%  "
            f"TAR@0.01={src_m['tar_far001']*100:.2f}%")
        log(f"  [Student | Target]  "
            f"ACC={tgt_m['acc']*100:.2f}%  EER={tgt_m['eer']*100:.2f}%  "
            f"TAR@0.1={tgt_m['tar_far01']*100:.2f}%  "
            f"TAR@0.01={tgt_m['tar_far001']*100:.2f}%")

        # ── Also evaluate the teacher ────────────────────────────────
        log(f"  → Teacher evaluation at epoch {epoch}:")
        tch_src_m = evaluate(teacher, src_eval_loader, split="S2 Teacher-Src")
        tch_tgt_m = evaluate(teacher, tgt_eval_loader, split="S2 Teacher-Tgt")
        log(f"  [Teacher | Source]  "
            f"ACC={tch_src_m['acc']*100:.2f}%  EER={tch_src_m['eer']*100:.2f}%")
        log(f"  [Teacher | Target]  "
            f"ACC={tch_tgt_m['acc']*100:.2f}%  EER={tch_tgt_m['eer']*100:.2f}%")

        # ── Save best student by target ACC ─────────────────────────
        if tgt_m['acc'] > best_s2_tgt_acc:
            best_s2_tgt_acc = tgt_m['acc']
            save({
                "epoch"               : epoch,
                "student_state_dict"  : student.state_dict(),
                "teacher_state_dict"  : teacher.state_dict(),
                "adaface_state_dict"  : adaface.state_dict(),
                "src_metrics"         : src_m,
                "tgt_metrics"         : tgt_m,
                "label_map"           : LABEL_MAP,
                "source_spectrum"     : SOURCE_SPECTRUM,
                "target_spectrum"     : TARGET_SPECTRUM,
            }, STAGE2_CKPT)
            log(f"  ★ New best Target ACC={best_s2_tgt_acc*100:.2f}% — saved")


# =============================================================================
# FINAL RESULTS
# =============================================================================
log("\n" + "=" * 70)
log("FINAL RESULTS")
log("=" * 70)

# Load best student checkpoint
best_ckpt = load(STAGE2_CKPT)
student.load_state_dict(best_ckpt["student_state_dict"])

log(f"Best checkpoint: epoch={best_ckpt['epoch']}")
log(f"\n[Best Student | Source ({SOURCE_SPECTRUM})]")
log(f"  ACC        = {best_ckpt['src_metrics']['acc']*100:.2f}%")
log(f"  EER        = {best_ckpt['src_metrics']['eer']*100:.2f}%")
log(f"  TAR@FAR0.1 = {best_ckpt['src_metrics']['tar_far01']*100:.2f}%")
log(f"  TAR@FAR0.01= {best_ckpt['src_metrics']['tar_far001']*100:.2f}%")

log(f"\n[Best Student | Target ({TARGET_SPECTRUM})]")
log(f"  ACC        = {best_ckpt['tgt_metrics']['acc']*100:.2f}%")
log(f"  EER        = {best_ckpt['tgt_metrics']['eer']*100:.2f}%")
log(f"  TAR@FAR0.1 = {best_ckpt['tgt_metrics']['tar_far01']*100:.2f}%")
log(f"  TAR@FAR0.01= {best_ckpt['tgt_metrics']['tar_far001']*100:.2f}%")

log(f"\nLog saved → {os.path.join(LOG_DIR, 'tscan_run.log')}")
log(f"Checkpoints → {SAVE_DIR}/")
log("=" * 70)
