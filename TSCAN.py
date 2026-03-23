"""
TSCAN: Teacher-Student Co-Learning Adaptive Network
for Cross-Spectrum Palmprint Recognition on CASIA-MS

Backbone (ResNet18) is FROZEN. Only linear, hash layers and AdaFace
weight matrix are trainable in both phases.
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

DATA_ROOT        = "/home/pai-ng/Jamal/CASIA-MS-ROI"

ALL_SPECTRUMS    = ['460', '630', '700', '850', '940', 'White']
SOURCE_SPECTRUM  = '460'
TARGET_SPECTRUM  = '630'
SEPARATE_HANDS   = True

FEATURE_DIM      = 128
ADAFACE_M0       = 0.5
ADAFACE_MMIN     = 0.25
ADAFACE_S        = 64.0

S1_EPOCHS        = 60
S1_LR            = 1e-3
S1_WEIGHT_DECAY  = 5e-4
S1_BATCH_SIZE    = 64
S1_LR_MILESTONES = [30, 50]
S1_LR_GAMMA      = 0.1

S2_EPOCHS        = 50
S2_LR            = 1e-4
S2_WEIGHT_DECAY  = 5e-4
S2_BATCH_SIZE    = 32
S2_LR_MILESTONES = [25, 40]
S2_LR_GAMMA      = 0.1

EMA_DECAY             = 0.99
PSEUDO_LABEL_THRESH   = 0.8
ALPHA                 = 1.0
BETA                  = 0.8
GAMMA_LOSS            = 0.3

RESIZE_SIZE      = 124
CROP_SIZE        = 112

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS      = 8
PIN_MEMORY       = True
SEED             = 42
EVAL_EVERY       = 5
MAX_PAIRS        = 500_000


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
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum   += val * n
        self.count += n

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
    subject  = parts[0]
    hand     = parts[1]
    spectrum = parts[2]
    identity = f"{subject}_{hand}" if SEPARATE_HANDS else subject
    return identity, spectrum


def scan_spectrum(spectrum):
    files   = sorted(glob.glob(os.path.join(DATA_ROOT, "*.jpg")))
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


class LabeledDataset(Dataset):
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
        img = Image.open(fp).convert("RGB")
        return self.transform(img), self.label_map[ident]


class UnlabeledTargetDataset(Dataset):
    def __init__(self, spectrum):
        self.weak   = weak_transform()
        self.strong = strong_transform()
        records     = scan_spectrum(spectrum)
        assert records, f"No target images found for spectrum '{spectrum}'"
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fp, _ = self.records[idx]
        img   = Image.open(fp).convert("RGB")
        return self.weak(img), self.strong(img)


# =============================================================================
# MODEL
# =============================================================================

class FeatureEncoder(nn.Module):
    """
    ResNet18 backbone (FROZEN) → Linear(512→128) → Tanh
    Only self.linear and self.hash are updated during training.
    """
    def __init__(self, feat_dim=128, pretrained=True):
        super().__init__()
        weights       = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone      = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.flatten  = nn.Flatten()
        self.linear   = nn.Linear(512, feat_dim, bias=True)   # trainable
        self.hash     = nn.Tanh()                              # trainable (no params)

        # ── Freeze the entire backbone ──────────────────────────────────
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():                    # no gradient through backbone
            bb = self.flatten(self.backbone(x))  # (B, 512)
        feat = self.hash(self.linear(bb))        # (B, 128)  ← gradient flows here
        return feat, bb


class PalmNet(nn.Module):
    def __init__(self, feat_dim=128, pretrained=True):
        super().__init__()
        self.encoder = FeatureEncoder(feat_dim=feat_dim, pretrained=pretrained)

    def forward(self, x):
        return self.encoder(x)

    def get_features(self, x):
        feat, _ = self.encoder(x)
        return feat

    def trainable_parameters(self):
        """Return only the unfrozen parameters (linear + hash)."""
        return [p for p in self.parameters() if p.requires_grad]


class AdaFaceLoss(nn.Module):
    def __init__(self, num_classes, feat_dim=128, m0=0.5, m_min=0.25, s=64.0):
        super().__init__()
        self.m0    = m0
        self.m_min = m_min
        self.s     = s
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def _margin(self, norms):
        lo    = norms.min().detach()
        hi    = norms.max().detach()
        denom = (hi - lo).clamp(min=1e-8)
        return (self.m_min + (self.m0 - self.m_min) * (norms - lo) / denom
                ).clamp(self.m_min, self.m0)

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
        feat_n = F.normalize(features, dim=1)
        w_n    = F.normalize(self.weight, dim=1)
        return feat_n @ w_n.T * self.s


class GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class DomainDiscriminator(nn.Module):
    def __init__(self, feat_dim=128, hidden=64, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.net   = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        feat = GRLFunction.apply(feat, self.alpha)
        return self.net(feat)

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
def generate_pseudo_labels(teacher, weak_imgs, adaface, threshold=0.8):
    teacher.eval()
    feat, _   = teacher(weak_imgs)
    feat_n    = F.normalize(feat, dim=1)
    w_n       = F.normalize(adaface.weight, dim=1)
    probs     = F.softmax(feat_n @ w_n.T * adaface.s, dim=1)
    max_p, pl = probs.max(dim=1)
    mask      = max_p >= threshold
    pl[~mask] = -1
    return pl, mask


def domain_loss(discriminator, src_feat, tgt_feat):
    N, M     = src_feat.size(0), tgt_feat.size(0)
    src_lbl  = torch.zeros(N, 1, device=src_feat.device)
    tgt_lbl  = torch.ones (M, 1, device=tgt_feat.device)
    preds    = discriminator(torch.cat([src_feat, tgt_feat], dim=0))
    labels   = torch.cat([src_lbl, tgt_lbl], dim=0)
    return F.binary_cross_entropy(preds, labels)


# =============================================================================
# EVALUATION  (ACC, EER, TAR@FAR)
# =============================================================================

@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    feats, labs = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        feat = F.normalize(model.get_features(imgs), dim=1)
        feats.append(feat.cpu().numpy())
        labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def build_pairs(features, labels):
    rng = np.random.RandomState(42)
    genuine_scores, impostor_scores = [], []
    for uid in np.unique(labels):
        idx = np.where(labels == uid)[0]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                genuine_scores.append(float(np.dot(features[idx[i]], features[idx[j]])))
    n_imp = min(len(genuine_scores) * 5, MAX_PAIRS)
    N     = len(labels)
    seen  = 0
    while len(impostor_scores) < n_imp and seen < n_imp * 3:
        i, j = rng.choice(N, 2, replace=False)
        if labels[i] != labels[j]:
            impostor_scores.append(float(np.dot(features[i], features[j])))
        seen += 1
    scores     = np.array(genuine_scores + impostor_scores, dtype=np.float32)
    is_genuine = np.array([True] * len(genuine_scores) +
                           [False] * len(impostor_scores))
    return scores, is_genuine


def compute_metrics(scores, is_genuine):
    thresholds = np.linspace(-1.0, 1.0, 1000)
    gen = scores[ is_genuine]
    imp = scores[~is_genuine]
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
    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    eer     = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)
    acc     = float(acc_arr.max())
    return {"acc": acc, "eer": eer}


def get_eer(model, loader):
    """Returns (identification_acc, verification_eer) for a loader."""
    features, labels = extract_features(model, loader)
    scores, is_genuine = build_pairs(features, labels)
    m = compute_metrics(scores, is_genuine)
    return m["acc"], m["eer"]


@torch.no_grad()
def identification_accuracy(model, adaface, loader):
    """Top-1 closed-set identification accuracy (classification via cosine logits)."""
    model.eval()
    adaface.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        feat, _  = model(imgs)
        logits   = adaface.get_logits(feat)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total   += labels.size(0)
    return correct / max(total, 1)


# =============================================================================
# DATASET AND MODEL SETUP
# =============================================================================

set_seed(SEED)

log("=" * 65)
log(f"TSCAN  |  {SOURCE_SPECTRUM} → {TARGET_SPECTRUM}  |  device={DEVICE}")
log(f"Backbone: FROZEN  |  Trainable: linear(512→128) + AdaFace W")
log("=" * 65)

# Datasets
src_train_ds = LabeledDataset(SOURCE_SPECTRUM, weak_transform())
NUM_CLASSES  = src_train_ds.num_classes
LABEL_MAP    = src_train_ds.label_map
log(f"Source [{SOURCE_SPECTRUM}]: {len(src_train_ds)} images, {NUM_CLASSES} identities")

tgt_eval_ds  = LabeledDataset(TARGET_SPECTRUM, eval_transform(), LABEL_MAP)
log(f"Target [{TARGET_SPECTRUM}]: {len(tgt_eval_ds)} images (eval only in Phase 1)")

# DataLoaders
s1_train_loader = DataLoader(src_train_ds, batch_size=S1_BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, drop_last=True)

src_eval_loader = DataLoader(
    LabeledDataset(SOURCE_SPECTRUM, eval_transform(), LABEL_MAP),
    batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

tgt_eval_loader = DataLoader(tgt_eval_ds, batch_size=128,
                              shuffle=False, num_workers=NUM_WORKERS)

s2_src_ds = LabeledDataset(SOURCE_SPECTRUM, strong_transform(), LABEL_MAP)
s2_tgt_ds = UnlabeledTargetDataset(TARGET_SPECTRUM)

s2_src_loader = DataLoader(s2_src_ds, batch_size=S2_BATCH_SIZE,
                            shuffle=True, num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY, drop_last=True)
s2_tgt_loader = DataLoader(s2_tgt_ds, batch_size=S2_BATCH_SIZE,
                            shuffle=True, num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY, drop_last=True)

# Models
teacher  = PalmNet(feat_dim=FEATURE_DIM, pretrained=True).to(DEVICE)
adaface  = AdaFaceLoss(num_classes=NUM_CLASSES, feat_dim=FEATURE_DIM,
                       m0=ADAFACE_M0, m_min=ADAFACE_MMIN, s=ADAFACE_S).to(DEVICE)

# Report trainable vs frozen params
total_p     = sum(p.numel() for p in teacher.parameters())
trainable_p = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
frozen_p    = total_p - trainable_p
log(f"Teacher params  — total: {total_p/1e6:.2f}M  "
    f"| trainable: {trainable_p/1e6:.4f}M  "
    f"| frozen: {frozen_p/1e6:.2f}M")


# =============================================================================
# PHASE 1 — TEACHER INITIALIZATION
# =============================================================================

log("\n" + "=" * 65)
log("PHASE 1 — Teacher Initialization")
log(f"{'Epoch':>6}  {'Loss':>8}  "
    f"{'Train ID Acc':>13}  {'Train EER':>10}  "
    f"{'Test ID Acc':>12}  {'Test EER':>9}")
log("-" * 65)

# Only linear layer + AdaFace weight matrix — backbone is frozen
s1_optimizer = optim.Adam(
    teacher.trainable_parameters() + list(adaface.parameters()),
    lr=S1_LR, weight_decay=S1_WEIGHT_DECAY,
)
s1_scheduler = MultiStepLR(s1_optimizer, milestones=S1_LR_MILESTONES,
                            gamma=S1_LR_GAMMA)

# Save best teacher state in memory (no file I/O)
best_s1_state    = None
best_s1_adaface  = None
best_s1_tgt_eer  = 1.0

for epoch in range(1, S1_EPOCHS + 1):
    teacher.train()
    adaface.train()

    loss_m = AverageMeter()
    acc_m  = AverageMeter()

    for imgs, labels in s1_train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        feat, _  = teacher(imgs)
        loss     = adaface(feat, labels)
        s1_optimizer.zero_grad()
        loss.backward()
        s1_optimizer.step()
        with torch.no_grad():
            acc = (adaface.get_logits(feat).argmax(1) == labels).float().mean().item()
        loss_m.update(loss.item(), imgs.size(0))
        acc_m.update(acc,          imgs.size(0))

    s1_scheduler.step()

    # ── Metrics ─────────────────────────────────────────────────────────
    train_id_acc = identification_accuracy(teacher, adaface, s1_train_loader)
    train_id_acc_batch = acc_m.avg          # fast batch-level estimate

    # Verification EER on train set (genuine/impostor pairs)
    _, train_eer = get_eer(teacher, src_eval_loader)

    # Test metrics on target spectrum
    test_id_acc  = identification_accuracy(teacher, adaface, tgt_eval_loader)
    _, test_eer  = get_eer(teacher, tgt_eval_loader)

    log(f"{epoch:>6}  {loss_m.avg:>8.4f}  "
        f"{train_id_acc*100:>12.2f}%  {train_eer*100:>9.2f}%  "
        f"{test_id_acc*100:>11.2f}%  {test_eer*100:>8.2f}%")

    # Keep best teacher state in memory
    if test_eer < best_s1_tgt_eer:
        best_s1_tgt_eer = test_eer
        best_s1_state   = copy.deepcopy(teacher.state_dict())
        best_s1_adaface = copy.deepcopy(adaface.state_dict())

log("-" * 65)
log(f"Phase 1 done  |  Best Test EER = {best_s1_tgt_eer*100:.2f}%")


# =============================================================================
# PHASE 2 — TEACHER-STUDENT CO-LEARNING
# =============================================================================

log("\n" + "=" * 65)
log("PHASE 2 — Teacher-Student Co-Learning (Domain Adaptation)")
log(f"{'Epoch':>6}  {'L_total':>8}  {'L_sup':>7}  {'L_uns':>7}  {'L_dis':>7}  "
    f"{'Train ID Acc':>13}  {'Train EER':>10}  "
    f"{'Test ID Acc':>12}  {'Test EER':>9}")
log("-" * 65)

# Load best Phase-1 weights into both teacher and student
teacher.load_state_dict(best_s1_state)
adaface.load_state_dict(best_s1_adaface)

student = PalmNet(feat_dim=FEATURE_DIM, pretrained=True).to(DEVICE)
student.load_state_dict(best_s1_state)      # student starts identical to teacher

discriminator = DomainDiscriminator(
    feat_dim=FEATURE_DIM, hidden=64, alpha=1.0
).to(DEVICE)

# Teacher: no gradient (EMA only)
for p in teacher.parameters():
    p.requires_grad = False

# Only student trainable layers + discriminator + adaface
s2_optimizer = optim.Adam(
    student.trainable_parameters() +
    list(discriminator.parameters()) +
    list(adaface.parameters()),
    lr=S2_LR, weight_decay=S2_WEIGHT_DECAY,
)
s2_scheduler = MultiStepLR(s2_optimizer, milestones=S2_LR_MILESTONES,
                            gamma=S2_LR_GAMMA)

total_steps = len(s2_src_loader) * S2_EPOCHS
global_step = 0

for epoch in range(1, S2_EPOCHS + 1):
    student.train()
    teacher.eval()
    discriminator.train()
    adaface.train()

    loss_t_m  = AverageMeter()
    loss_s_m  = AverageMeter()
    loss_u_m  = AverageMeter()
    loss_d_m  = AverageMeter()

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

        pl, mask = generate_pseudo_labels(
            teacher, tgt_weak, adaface, PSEUDO_LABEL_THRESH
        )

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
            student.trainable_parameters() +
            list(discriminator.parameters()) +
            list(adaface.parameters()),
            max_norm=5.0,
        )
        s2_optimizer.step()
        ema_update(teacher, student, decay=EMA_DECAY)

        bs = src_imgs.size(0)
        loss_t_m.update(L_total.item(), bs)
        loss_s_m.update(L_sup.item(),   bs)
        loss_u_m.update(L_unsup.item() if mask.any() else 0., bs)
        loss_d_m.update(L_dis.item(),   bs)
        global_step += 1

    s2_scheduler.step()

    # ── Metrics every EVAL_EVERY epochs ─────────────────────────────────
    if epoch % EVAL_EVERY == 0 or epoch == S2_EPOCHS:
        train_id_acc = identification_accuracy(student, adaface, s2_src_loader)
        _, train_eer = get_eer(student, src_eval_loader)
        test_id_acc  = identification_accuracy(student, adaface, tgt_eval_loader)
        _, test_eer  = get_eer(student, tgt_eval_loader)

        log(f"{epoch:>6}  {loss_t_m.avg:>8.4f}  "
            f"{loss_s_m.avg:>7.4f}  {loss_u_m.avg:>7.4f}  {loss_d_m.avg:>7.4f}  "
            f"{train_id_acc*100:>12.2f}%  {train_eer*100:>9.2f}%  "
            f"{test_id_acc*100:>11.2f}%  {test_eer*100:>8.2f}%")
    else:
        # Non-eval epochs: print losses only
        log(f"{epoch:>6}  {loss_t_m.avg:>8.4f}  "
            f"{loss_s_m.avg:>7.4f}  {loss_u_m.avg:>7.4f}  {loss_d_m.avg:>7.4f}  "
            f"{'---':>13}  {'---':>10}  {'---':>12}  {'---':>9}")

log("-" * 65)
log("TSCAN training complete.")
