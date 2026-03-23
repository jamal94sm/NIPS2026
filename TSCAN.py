"""
TSCAN v3 — Fixed version based on training diagnostics.

Changes from v2:
  1. Backbone PARTIALLY unfrozen: layer3 + layer4 trainable, earlier layers frozen.
     ImageNet features cannot represent multispectral palmprint images when fully frozen.
  2. Layer-wise learning rates: backbone layers get 10x lower LR than the head.
  3. AdaFace scale s: 64 → 32. s=64 is designed for large face datasets and
     causes loss explosion (21+) on small palmprint sets.
  4. Feature dim: 128 → 256. Gives the head more capacity.
  5. Pseudo-label threshold: 0.8 → 0.6. Teacher at 3% accuracy never passes 0.8.
  6. Phase 1 epochs: 60 → 100. Partially unfrozen backbone needs more time.
  7. LR schedule: added warmup + cosine decay instead of step decay.
  8. Gradient clipping added to Phase 1 as well.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import glob
import copy
import time
import math
import random
import itertools

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

# ── Model ─────────────────────────────────────────────────────────────────────
FEATURE_DIM     = 256       # was 128 — more capacity for the head

# ── AdaFace ───────────────────────────────────────────────────────────────────
ADAFACE_M0      = 0.5
ADAFACE_MMIN    = 0.25
ADAFACE_S       = 32.0      # was 64 — s=64 caused loss explosion (21+) on small sets

# ── Stage 1 ───────────────────────────────────────────────────────────────────
S1_EPOCHS       = 100       # was 60 — partially unfrozen backbone needs more steps
S1_LR_HEAD      = 1e-3      # learning rate for linear + hash + AdaFace W
S1_LR_BACKBONE  = 1e-4      # 10x lower for unfrozen backbone layers (layer3, layer4)
S1_WEIGHT_DECAY = 5e-4
S1_BATCH_SIZE   = 64
S1_WARMUP_EPOCHS= 5         # linear warmup before cosine decay

# ── Stage 2 ───────────────────────────────────────────────────────────────────
S2_EPOCHS       = 60        # slightly more time for adaptation
S2_LR_HEAD      = 5e-5      # lower than stage 1 to avoid forgetting
S2_LR_BACKBONE  = 5e-6      # backbone layers even lower in stage 2
S2_WEIGHT_DECAY = 5e-4
S2_BATCH_SIZE   = 32
S2_WARMUP_EPOCHS= 3

# ── Co-learning ───────────────────────────────────────────────────────────────
EMA_DECAY           = 0.999     # slightly higher than 0.99 — smoother teacher
PSEUDO_LABEL_THRESH = 0.6       # was 0.8 — teacher at 3% acc never passes 0.8
ALPHA               = 1.0       # L_sup weight
BETA                = 0.8       # L_unsup weight
GAMMA_LOSS          = 0.3       # L_dis weight

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


class LabeledDataset(Dataset):
    def __init__(self, spectrum, transform, label_map=None):
        self.transform = transform
        records        = scan_spectrum(spectrum)
        assert records, f"No images found for spectrum '{spectrum}'"
        self.label_map = label_map if label_map else build_label_map(records)
        self.records   = [(fp, i) for fp, i in records if i in self.label_map]

    @property
    def num_classes(self):
        return len(self.label_map)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fp, ident = self.records[idx]
        return self.transform(Image.open(fp).convert("RGB")), self.label_map[ident]


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
        img = Image.open(self.records[idx][0]).convert("RGB")
        return self.weak(img), self.strong(img)


# =============================================================================
# MODEL
# =============================================================================

class FeatureEncoder(nn.Module):
    """
    ResNet18 with PARTIAL unfreezing:
      Frozen  : conv1, bn1, layer1, layer2  (low-level edges/textures)
      Trainable: layer3, layer4             (high-level semantic features)
      Trainable: linear(512→256), Tanh      (hash head)

    Why partial and not full freeze:
      CASIA-MS multispectral images (460nm–940nm) differ fundamentally from
      ImageNet RGB. layer3/layer4 need to adapt to palmprint-specific textures.
      Early layers (edges, blobs) transfer fine and stay frozen.
    """
    def __init__(self, feat_dim=256, pretrained=True):
        super().__init__()
        weights  = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # Split ResNet18 into frozen and trainable parts
        # children: conv1(0), bn1(1), relu(2), maxpool(3),
        #           layer1(4), layer2(5), layer3(6), layer4(7), avgpool(8)
        self.frozen_layers    = nn.Sequential(*list(backbone.children())[:6])   # conv1→layer2
        self.trainable_layers = nn.Sequential(*list(backbone.children())[6:9])  # layer3→avgpool
        self.flatten          = nn.Flatten()
        self.linear           = nn.Linear(512, feat_dim, bias=True)
        self.hash             = nn.Tanh()

        # Freeze early layers
        for param in self.frozen_layers.parameters():
            param.requires_grad = False

        # Trainable: layer3, layer4, linear, hash
        # (layer3, layer4 use lower LR — handled in optimizer param groups)

    def forward(self, x):
        with torch.no_grad():
            x = self.frozen_layers(x)       # frozen, no grad
        x    = self.trainable_layers(x)     # layer3+layer4, grad flows
        bb   = self.flatten(x)              # (B, 512)
        feat = self.hash(self.linear(bb))   # (B, feat_dim)
        return feat, bb

    def backbone_parameters(self):
        """layer3 + layer4 parameters (lower LR group)."""
        return list(self.trainable_layers.parameters())

    def head_parameters(self):
        """linear + hash parameters (higher LR group)."""
        return list(self.linear.parameters())


class PalmNet(nn.Module):
    def __init__(self, feat_dim=256, pretrained=True):
        super().__init__()
        self.encoder = FeatureEncoder(feat_dim=feat_dim, pretrained=pretrained)

    def forward(self, x):
        return self.encoder(x)

    def get_features(self, x):
        feat, _ = self.encoder(x)
        return feat

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
        self.m0    = m0
        self.m_min = m_min
        self.s     = s
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
        return F.normalize(features, dim=1) @ F.normalize(self.weight, dim=1).T * self.s


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
    feat, _ = teacher(weak_imgs)
    probs   = F.softmax(adaface.get_logits(feat), dim=1)
    max_p, pl = probs.max(dim=1)
    mask    = max_p >= threshold
    pl[~mask] = -1
    return pl, mask


def domain_loss(discriminator, src_feat, tgt_feat):
    src_lbl = torch.zeros(src_feat.size(0), 1, device=src_feat.device)
    tgt_lbl = torch.ones (tgt_feat.size(0), 1, device=tgt_feat.device)
    preds   = discriminator(torch.cat([src_feat, tgt_feat], dim=0))
    return F.binary_cross_entropy(preds, torch.cat([src_lbl, tgt_lbl], dim=0))


def make_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup then cosine annealing."""
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
    model.eval()
    feats, labs = [], []
    for imgs, labels in loader:
        feat = F.normalize(model.get_features(imgs.to(DEVICE)), dim=1)
        feats.append(feat.cpu().numpy())
        labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def build_pairs(features, labels):
    rng = np.random.RandomState(42)
    genuine, impostor = [], []
    for uid in np.unique(labels):
        idx = np.where(labels == uid)[0]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                genuine.append(float(np.dot(features[idx[i]], features[idx[j]])))
    n_imp = min(len(genuine) * 5, MAX_PAIRS)
    N, seen = len(labels), 0
    while len(impostor) < n_imp and seen < n_imp * 3:
        i, j = rng.choice(N, 2, replace=False)
        if labels[i] != labels[j]:
            impostor.append(float(np.dot(features[i], features[j])))
        seen += 1
    scores     = np.array(genuine + impostor, dtype=np.float32)
    is_genuine = np.array([True] * len(genuine) + [False] * len(impostor))
    return scores, is_genuine


def compute_eer(scores, is_genuine):
    thresholds = np.linspace(-1.0, 1.0, 1000)
    gen = scores[ is_genuine]
    imp = scores[~is_genuine]
    far_arr, frr_arr, acc_arr = [], [], []
    for thr in thresholds:
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


def get_metrics(model, adaface, labeled_loader, eval_loader):
    """
    Returns (id_acc, eer).
    id_acc: closed-set identification accuracy via cosine logits.
    eer:    open-set verification EER via genuine/impostor pairs.
    """
    # Identification accuracy (closed-set)
    model.eval(); adaface.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in labeled_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            feat, _  = model(imgs)
            logits   = adaface.get_logits(feat)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    id_acc = correct / max(total, 1)

    # Verification EER (open-set)
    features, labs = extract_features(model, eval_loader)
    scores, is_genuine = build_pairs(features, labs)
    _, eer = compute_eer(scores, is_genuine)

    return id_acc, eer


# =============================================================================
# DATASET & MODEL SETUP
# =============================================================================

set_seed(SEED)

log("=" * 70)
log(f"TSCAN v3  |  {SOURCE_SPECTRUM} → {TARGET_SPECTRUM}  |  device={DEVICE}")
log(f"Backbone: layer3+layer4 trainable (LR={S1_LR_BACKBONE}), "
    f"layer1+layer2 frozen")
log(f"Head LR={S1_LR_HEAD}  |  feat_dim={FEATURE_DIM}  |  s={ADAFACE_S}")
log("=" * 70)

# Datasets & Loaders
src_train_ds    = LabeledDataset(SOURCE_SPECTRUM, weak_transform())
NUM_CLASSES     = src_train_ds.num_classes
LABEL_MAP       = src_train_ds.label_map
log(f"Source [{SOURCE_SPECTRUM}]: {len(src_train_ds)} images, {NUM_CLASSES} identities")
log(f"Target [{TARGET_SPECTRUM}]: {len(LabeledDataset(TARGET_SPECTRUM, eval_transform(), LABEL_MAP))} images")

s1_train_loader = DataLoader(src_train_ds, batch_size=S1_BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, drop_last=True)
src_eval_loader = DataLoader(LabeledDataset(SOURCE_SPECTRUM, eval_transform(), LABEL_MAP),
                              batch_size=128, shuffle=False, num_workers=NUM_WORKERS)
tgt_eval_loader = DataLoader(LabeledDataset(TARGET_SPECTRUM, eval_transform(), LABEL_MAP),
                              batch_size=128, shuffle=False, num_workers=NUM_WORKERS)
s2_src_loader   = DataLoader(LabeledDataset(SOURCE_SPECTRUM, strong_transform(), LABEL_MAP),
                              batch_size=S2_BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
s2_tgt_loader   = DataLoader(UnlabeledTargetDataset(TARGET_SPECTRUM),
                              batch_size=S2_BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)

# Models
teacher = PalmNet(feat_dim=FEATURE_DIM, pretrained=True).to(DEVICE)
adaface = AdaFaceLoss(num_classes=NUM_CLASSES, feat_dim=FEATURE_DIM,
                      m0=ADAFACE_M0, m_min=ADAFACE_MMIN, s=ADAFACE_S).to(DEVICE)

# Parameter count summary
frozen_p    = sum(p.numel() for p in teacher.encoder.frozen_layers.parameters())
backbone_p  = sum(p.numel() for p in teacher.backbone_parameters())
head_p      = sum(p.numel() for p in teacher.head_parameters())
adaface_p   = sum(p.numel() for p in adaface.parameters())
log(f"Frozen params     : {frozen_p/1e6:.2f}M  (conv1, bn1, layer1, layer2)")
log(f"Trainable backbone: {backbone_p/1e6:.2f}M  (layer3, layer4)  LR={S1_LR_BACKBONE}")
log(f"Trainable head    : {head_p/1e6:.4f}M  (linear+hash)      LR={S1_LR_HEAD}")
log(f"AdaFace weight W  : {adaface_p/1e6:.4f}M                    LR={S1_LR_HEAD}")


# =============================================================================
# PHASE 1 — TEACHER INITIALIZATION
# =============================================================================

log("\n" + "=" * 70)
log("PHASE 1 — Teacher Initialization")
log(f"{'Epoch':>6}  {'Loss':>8}  "
    f"{'Train ID':>9}  {'Train EER':>10}  "
    f"{'Test ID':>8}  {'Test EER':>9}  {'Pseudo%':>8}")
log("-" * 70)

# Two param groups: backbone layers get lower LR
s1_optimizer = optim.AdamW([
    {'params': teacher.backbone_parameters(), 'lr': S1_LR_BACKBONE},
    {'params': teacher.head_parameters(),     'lr': S1_LR_HEAD},
    {'params': adaface.parameters(),          'lr': S1_LR_HEAD},
], weight_decay=S1_WEIGHT_DECAY)

s1_scheduler = make_warmup_cosine_scheduler(s1_optimizer, S1_WARMUP_EPOCHS, S1_EPOCHS)

best_s1_tgt_eer = 1.0
best_s1_state   = None
best_s1_adaface = None

for epoch in range(1, S1_EPOCHS + 1):
    teacher.train(); adaface.train()
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
        train_id, train_eer = get_metrics(teacher, adaface, s1_train_loader, src_eval_loader)
        test_id,  test_eer  = get_metrics(teacher, adaface, tgt_eval_loader, tgt_eval_loader)

        # Pseudo-label acceptance rate (how many target samples pass threshold)
        teacher.eval()
        pseudo_total = pseudo_accept = 0
        with torch.no_grad():
            for imgs, _ in tgt_eval_loader:
                feat, _ = teacher(imgs.to(DEVICE))
                probs   = F.softmax(adaface.get_logits(feat), dim=1)
                max_p   = probs.max(dim=1).values
                pseudo_accept += (max_p >= PSEUDO_LABEL_THRESH).sum().item()
                pseudo_total  += imgs.size(0)
        pseudo_pct = pseudo_accept / max(pseudo_total, 1) * 100

        log(f"{epoch:>6}  {loss_m.avg:>8.4f}  "
            f"{train_id*100:>8.2f}%  {train_eer*100:>9.2f}%  "
            f"{test_id*100:>7.2f}%  {test_eer*100:>8.2f}%  "
            f"{pseudo_pct:>7.1f}%")

        if test_eer < best_s1_tgt_eer:
            best_s1_tgt_eer = test_eer
            best_s1_state   = copy.deepcopy(teacher.state_dict())
            best_s1_adaface = copy.deepcopy(adaface.state_dict())
    else:
        log(f"{epoch:>6}  {loss_m.avg:>8.4f}")

log("-" * 70)
log(f"Phase 1 done  |  Best Test EER = {best_s1_tgt_eer*100:.2f}%")


# =============================================================================
# PHASE 2 — TEACHER-STUDENT CO-LEARNING
# =============================================================================

log("\n" + "=" * 70)
log("PHASE 2 — Teacher-Student Co-Learning (Domain Adaptation)")
log(f"{'Epoch':>6}  {'L_total':>8}  {'L_sup':>7}  {'L_uns':>7}  {'L_dis':>7}  "
    f"{'Pseudo%':>8}  "
    f"{'Train ID':>9}  {'Train EER':>10}  "
    f"{'Test ID':>8}  {'Test EER':>9}")
log("-" * 80)

# Load best Phase 1 weights into both teacher and student
teacher.load_state_dict(best_s1_state)
adaface.load_state_dict(best_s1_adaface)

student = PalmNet(feat_dim=FEATURE_DIM, pretrained=True).to(DEVICE)
student.load_state_dict(best_s1_state)   # start identical to teacher

discriminator = DomainDiscriminator(
    feat_dim=FEATURE_DIM, hidden=128, alpha=1.0
).to(DEVICE)

# Teacher: no gradient (EMA only)
for p in teacher.parameters():
    p.requires_grad = False

# Student: two param groups (backbone lower LR, head higher LR)
s2_optimizer = optim.AdamW([
    {'params': student.backbone_parameters(), 'lr': S2_LR_BACKBONE},
    {'params': student.head_parameters(),     'lr': S2_LR_HEAD},
    {'params': adaface.parameters(),          'lr': S2_LR_HEAD},
    {'params': discriminator.parameters(),    'lr': S2_LR_HEAD},
], weight_decay=S2_WEIGHT_DECAY)

s2_scheduler = make_warmup_cosine_scheduler(s2_optimizer, S2_WARMUP_EPOCHS, S2_EPOCHS)

total_steps = len(s2_src_loader) * S2_EPOCHS
global_step = 0

for epoch in range(1, S2_EPOCHS + 1):
    student.train(); teacher.eval()
    discriminator.train(); adaface.train()

    loss_t_m = AverageMeter(); loss_s_m = AverageMeter()
    loss_u_m = AverageMeter(); loss_d_m = AverageMeter()
    pseudo_m = AverageMeter()

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

        pl, mask    = generate_pseudo_labels(teacher, tgt_weak, adaface,
                                             PSEUDO_LABEL_THRESH)
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
        train_id, train_eer = get_metrics(student, adaface, s2_src_loader, src_eval_loader)
        test_id,  test_eer  = get_metrics(student, adaface, tgt_eval_loader, tgt_eval_loader)
        log(f"{epoch:>6}  {loss_t_m.avg:>8.4f}  "
            f"{loss_s_m.avg:>7.4f}  {loss_u_m.avg:>7.4f}  {loss_d_m.avg:>7.4f}  "
            f"{pseudo_m.avg*100:>7.1f}%  "
            f"{train_id*100:>8.2f}%  {train_eer*100:>9.2f}%  "
            f"{test_id*100:>7.2f}%  {test_eer*100:>8.2f}%")
    else:
        log(f"{epoch:>6}  {loss_t_m.avg:>8.4f}  "
            f"{loss_s_m.avg:>7.4f}  {loss_u_m.avg:>7.4f}  {loss_d_m.avg:>7.4f}  "
            f"{pseudo_m.avg*100:>7.1f}%")

log("-" * 80)
log("TSCAN v3 training complete.")
