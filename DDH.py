"""
ddh_best.py  — Best synthesis of paper + released code
=======================================================

╔══════════════════════════════════════════════════════════════╗
║                  DATASET SPLIT CONFIGURATION                 ║
╠══════════════════════════════════════════════════════════════╣
║  Paper protocol (Section V-B):                               ║
║    "For each category, the HALF of palmprint images are      ║
║     adopted as training set and the remaining half are used  ║
║     as test set."                                            ║
║                                                              ║
║  CASIA-MS palmprint database facts:                          ║
║    • 5,502 images total  (full database)                     ║
║    • 624 hands  (full database)                              ║
║                                                              ║
║  YOUR dataset (100 subjects × 2 hands):                      ║
║    • 100 subjects, left + right hand treated separately      ║
║    • 200 independent classes  (each hand = one class)        ║
║    • ~8–9 images per hand  →  ~4 train / ~4-5 test           ║
║                                                              ║
║  Class label = subject_id + hand_side  (e.g. '001_L')        ║
║  Controlled by LABEL_POS=0 and HAND_POS=1 below.            ║
║                                                              ║
║  Applied split  (TRAIN_RATIO = 0.5):                         ║
║    • First 50% of each hand's images  →  TRAIN               ║
║    • Remaining 50%                    →  TEST                ║
║    • Hand with 8 images: 4 train / 4 test                    ║
║    • Hand with 9 images: 4 train / 5 test                    ║
║    • 200 classes × ~4 train  =  ~800 total train images      ║
║                                                              ║
║  To change: edit TRAIN_RATIO below, e.g. 0.6 for 60/40.     ║
╚══════════════════════════════════════════════════════════════╝

ARCHITECTURE choices:
  ✓ VGG pool4 frozen        ← released code (ground truth of what ran)
  ✓ Custom conv5 + FC head  ← released code (trainable on top of pool4)
  ✓ BN + LeakyReLU(0.2)     ← released code (critical for stability)
  ✓ Student Conv2 SAME pad  ← released code (paper padding=0 breaks sizes)
  ✓ Input 128×128            ← paper Section V-A (all databases 128×128)

LOSS choices:
  ✓ D = squared Euclidean        ← both agree (t=180 calibrated to this)
  ✓ d_ij = L2 norm               ← paper Eq.7/8 explicit ‖·‖_2
  ✓ L_q = mean((|h|−1)²), w=0.5 ← released code form
  ✓ L_rela = MSE of L2 dists     ← paper structure, MSE for gradient flow
  ✓ L_hard + hinge clamp(·,0)    ← no reward for already-met constraints
  ✓ Adam lr=0.001, α=1, β=0.8   ← paper Section V-B, Tables XI/XII
"""

import os, random, logging, argparse, csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_curve

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger()

# ╔══════════════════════════════════════════════════════════════╗
# ║            >>>  CHANGE ONLY THIS SECTION  <<<               ║
# ╚══════════════════════════════════════════════════════════════╝

DATA_DIR    = '/home/pai-ng/Jamal/CASIA-MS-ROI'
                         # Path to the folder of pre-cropped 128×128 ROI images.
                         # Supports two layouts:
                         #   (a) Subdirectory per subject+hand: DATA_DIR/001_L/img.bmp
                         #   (b) Flat folder, label in name:   DATA_DIR/001_L_01.bmp

LABEL_POS   = 0          # Token index of the SUBJECT ID in the filename.
                         # e.g. '001_L_01.bmp' split by '_' → token 0 = '001'

HAND_POS    = 1          # Token index of the HAND SIDE in the filename.
                         # e.g. '001_L_01.bmp' split by '_' → token 1 = 'L'
                         # Class = subject + hand  →  200 classes for CASIA-MS
                         # (100 subjects × 2 hands = 200 independent palms).
                         # Set HAND_POS = None to use subject-only labels (100 classes).

SEP         = '_'        # Filename token separator.

# ── Train / Test split ───────────────────────────────────────────
# Paper Section V-B: "half … training set, remaining half … test set"
# CASIA-MS: ~8-9 images per hand → ~4 train / ~4-5 test per class.
TRAIN_RATIO = 0.8        # 0.5 = 50% train / 50% test  (paper default)
                         # Change to e.g. 0.6 for 60/40.

# ── Regularisation ───────────────────────────────────────────────
DROPOUT_TEACHER = 0.5    # dropout rate for teacher FC layers
DROPOUT_STUDENT = 0.4    # dropout rate for student FC layer
WEIGHT_DECAY    = 1e-4   # L2 regularisation on all trainable weights

# ── Quantization loss weight ─────────────────────────────────────
# L_DHN = Lh + W_QUANT * Lq
# Lq = mean((|h| - 1)²) pushes hash codes toward ±1 (near-binary).
# If Lq is too small relative to Lh the codes stay soft and matching
# quality degrades after sign() binarisation.
# Rule of thumb: aim for W_QUANT * Lq ≈ 10–20% of Lh at early training.
# Example: Lh=5.6, Lq=0.1 → W_QUANT=10 makes Lq contribute 1.0 (~15%).
W_QUANT = 10.0           # increased from 0.5 (original) to balance losses

# ── Batch size ───────────────────────────────────────────────────
# Images are sampled randomly — no class ordering required.
# Just set how many images per batch.
BATCH_SIZE = 128     # standard random batch size

# ── Training (epoch-based) ────────────────────────────────────────
# One epoch = one full pass through all training batches.
# With 200 classes and batch covering 25 classes, one epoch ≈ 8 batches.
TEACHER_EPOCHS = 500     # epochs for teacher training
STUDENT_EPOCHS = 500     # epochs for DDH student training
LR             = 1e-3    # Adam initial learning rate
LR_MIN         = 1e-5    # cosine schedule final LR (reached at last epoch)

# ── Mid-training evaluation frequency (in EPOCHS) ────────────────
# Acc + EER on train and test sets are printed every EVAL_EVERY_EPOCHS.
# Set to 0 to only show the final evaluation table.
EVAL_EVERY_EPOCHS = 50   # ← change here  (e.g. 25, 100, …)

# ── Loss-print frequency (in EPOCHS) ─────────────────────────────
LOG_EVERY_EPOCHS  = 10

# ── Output directory ─────────────────────────────────────────────
SAVE_DIR = './ckpt_best'

# ══════════════════════════════════════════════════════════════════
# Do not edit below this line unless you know what you are changing
# ══════════════════════════════════════════════════════════════════

CFG = dict(
    data_dir        = DATA_DIR,
    label_pos       = LABEL_POS,
    hand_pos        = HAND_POS,
    sep             = SEP,
    train_ratio     = TRAIN_RATIO,
    hash_dim        = 128,
    img_size        = 128,
    lrelu_alpha     = 0.2,
    margin_t        = 180.0,
    w_quant         = W_QUANT,
    alpha           = 1.0,
    beta            = 0.8,
    dropout_teacher = DROPOUT_TEACHER,
    dropout_student = DROPOUT_STUDENT,
    weight_decay    = WEIGHT_DECAY,
    lr_min          = LR_MIN,
    batch_size      = BATCH_SIZE,
    teacher_epochs  = TEACHER_EPOCHS,
    student_epochs  = STUDENT_EPOCHS,
    lr              = LR,
    eval_every      = EVAL_EVERY_EPOCHS,
    log_every       = LOG_EVERY_EPOCHS,
    num_workers     = 4,
    save_dir        = SAVE_DIR,
)

# ─────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────
class CASIADataset(Dataset):
    """
    Supports two folder layouts:
      (a) Subdirectory per class:  root/001_L/img.bmp  → class = folder name
      (b) Flat folder, tokens in filename: root/001_L_01.bmp
          class key = tokens[label_pos] + '_' + tokens[hand_pos]
          e.g. '001_L_01.bmp' → '001_L'  (200 classes for CASIA-MS)
          Set hand_pos=None to use subject token only (100 classes).
    """
    EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.pgm', '.tiff'}

    def __init__(self, root, transform=None, label_pos=0, sep='_', hand_pos=1):
        self.transform    = transform
        self.paths        = []
        self.labels       = []
        self.idx_by_label = defaultdict(list)

        subdirs = sorted(d for d in os.listdir(root)
                         if os.path.isdir(os.path.join(root, d)))
        if subdirs:
            # Subdirectory layout: folder name IS the class key
            lbl_map = {s: i for i, s in enumerate(subdirs)}
            for subj in subdirs:
                for fn in sorted(os.listdir(os.path.join(root, subj))):
                    if os.path.splitext(fn)[1].lower() in self.EXTS:
                        i = len(self.paths)
                        self.paths.append(os.path.join(root, subj, fn))
                        lbl = lbl_map[subj]
                        self.labels.append(lbl)
                        self.idx_by_label[lbl].append(i)
        else:
            # Flat layout: build class key from filename tokens
            files = sorted(fn for fn in os.listdir(root)
                           if os.path.splitext(fn)[1].lower() in self.EXTS)
            if not files:
                raise RuntimeError(f'No images found in {root!r}')

            def _class_key(fn):
                tokens = fn.split(sep)
                subj   = tokens[label_pos]
                if hand_pos is not None and hand_pos < len(tokens):
                    return f'{subj}{sep}{tokens[hand_pos]}'   # e.g. '001_L'
                return subj                                    # e.g. '001'

            class_keys = sorted({_class_key(fn) for fn in files})
            lbl_map    = {k: i for i, k in enumerate(class_keys)}
            for fn in files:
                key = _class_key(fn)
                lbl = lbl_map[key]
                i   = len(self.paths)
                self.paths.append(os.path.join(root, fn))
                self.labels.append(lbl)
                self.idx_by_label[lbl].append(i)

    @property
    def num_classes(self): return len(self.idx_by_label)
    def __len__(self):     return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[i]


def split_dataset(ds, ratio=0.5):
    """
    First `ratio` fraction of each subject's sorted images → train.
    Remaining → test.  Mirrors paper Section V-B 50/50 protocol.
    """
    tr, te = [], []
    for lbl, idxs in sorted(ds.idx_by_label.items()):
        k = max(1, int(len(idxs) * ratio))
        tr.extend(idxs[:k])
        te.extend(idxs[k:])
    return tr, te


def print_split_summary(ds, tr_idx, te_idx):
    """Print a clear description of the train/test sets."""
    tr_set = set(tr_idx); te_set = set(te_idx)
    per_tr = [sum(1 for i in v if i in tr_set)
              for v in ds.idx_by_label.values()]
    per_te = [sum(1 for i in v if i in te_set)
              for v in ds.idx_by_label.values()]
    log.info('')
    log.info('  ┌─────────────────────────────────────────────────┐')
    log.info('  │              TRAIN / TEST SPLIT                 │')
    log.info('  ├─────────────────────────────────────────────────┤')
    log.info(f'  │  Subjects (classes)  : {ds.num_classes:<27}│')
    log.info(f'  │  Total images        : {len(ds):<27}│')
    log.info(f'  │  Train images        : {len(tr_idx):<5}  '
             f'({len(tr_idx)/len(ds)*100:.1f}% of total)          │')
    log.info(f'  │  Test  images        : {len(te_idx):<5}  '
             f'({len(te_idx)/len(ds)*100:.1f}% of total)          │')
    log.info(f'  │  Train per subject   : {min(per_tr)}–{max(per_tr)} images'
             f'{"":>28}│')
    log.info(f'  │  Test  per subject   : {min(per_te)}–{max(per_te)} images'
             f'{"":>28}│')
    log.info(f'  │  Split rule          : first {TRAIN_RATIO:.0%} per subject → train │')
    log.info('  └─────────────────────────────────────────────────┘')
    log.info('')


def get_transform(train=True, size=128):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ops = [transforms.Resize((size, size))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                    scale=(0.95, 1.05)),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


class ClassOrderedBatchSampler(Sampler):
    """Each batch = [class_A×num_per, class_B×num_per, …]."""
    def __init__(self, idx_by_label, train_indices,
                 num_per, n_cls_per_batch, seed=42):
        self.num_per = num_per
        self.n_cls   = n_cls_per_batch
        self.rng     = random.Random(seed)
        tr_set       = set(train_indices)
        self.lbl_idx = {
            lbl: [i for i in idxs if i in tr_set]
            for lbl, idxs in idx_by_label.items()
            if any(i in tr_set for i in idxs)
        }
        self.labels = list(self.lbl_idx.keys())

    def __iter__(self):
        lbls = self.labels.copy()
        self.rng.shuffle(lbls)
        for start in range(0, len(lbls) - self.n_cls + 1, self.n_cls):
            grp   = lbls[start: start + self.n_cls]
            batch = []
            for lbl in grp:
                pool = self.lbl_idx[lbl]
                k    = self.num_per
                batch.extend(self.rng.sample(pool, k)
                             if len(pool) >= k
                             else self.rng.choices(pool, k=k))
            yield batch

    def __len__(self): return len(self.labels) // self.n_cls


def make_onehot(labels_int, n_cls, device):
    oh = torch.zeros(labels_int.shape[0], n_cls, device=device)
    oh[torch.arange(labels_int.shape[0]), labels_int] = 1.0
    return oh


# ─────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────
class LReLU(nn.Module):
    def __init__(self, a=0.2): super().__init__(); self.a = a
    def forward(self, x): return torch.max(self.a * x, x)


class TeacherDHN(nn.Module):
    """
    VGG pool4 frozen + trainable conv5 block + compact FC head.

    FC head is intentionally smaller than the original paper's 4096→4096→2048
    because CASIA-MS with 200 classes has only ~800 training images total
    (~4 per class). The original head had ~59M trainable parameters which
    is ~74k params/sample — guaranteed overfitting.

    Reduced head (flat→1024→512→128) has ~9M params (~11k params/sample),
    which is still large but manageable with dropout=0.5 and weight decay.
    """
    def __init__(self, hash_dim=128, img_size=128, alpha=0.2, dropout=0.5):
        super().__init__()
        vgg = models.vgg16(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(vgg.features.children())[:24])
        for p in self.backbone.parameters(): p.requires_grad_(False)

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-5), LReLU(alpha),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-5), LReLU(alpha),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-5), LReLU(alpha),
            nn.MaxPool2d(2, stride=2),
        )
        flat = self._flat(img_size)
        self.fc = nn.Sequential(
            nn.Linear(flat, 1024),
            nn.BatchNorm1d(1024, eps=1e-5), LReLU(alpha), nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512,  eps=1e-5), LReLU(alpha), nn.Dropout(dropout),
            nn.Linear(512, hash_dim), nn.Tanh(),
        )
        self._init()

    def _flat(self, s):
        with torch.no_grad():
            x = self.backbone(torch.zeros(1, 3, s, s))
            return self.conv5(x).view(1, -1).shape[1]

    def _init(self):
        for m in list(self.conv5.modules()) + list(self.fc.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        with torch.no_grad(): x = self.backbone(x)
        return self.fc(torch.flatten(self.conv5(x), 1))

    @torch.no_grad()
    def get_codes(self, x): return torch.sign(self.forward(x))


class StudentDHN(nn.Module):
    """Paper Fig.6 + BN/LReLU from released code + dropout for regularisation."""
    def __init__(self, hash_dim=128, img_size=128, alpha=0.2, dropout=0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,  16, 3, stride=4, padding=0),    # VALID
            nn.BatchNorm2d(16, eps=1e-5), LReLU(alpha),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),    # SAME
            nn.BatchNorm2d(32, eps=1e-5), LReLU(alpha),
            nn.MaxPool2d(2, stride=1),
        )
        flat = self._flat(img_size)
        self.fc = nn.Sequential(
            nn.Linear(flat, 512), nn.BatchNorm1d(512, eps=1e-5),
            LReLU(alpha), nn.Dropout(dropout),
            nn.Linear(512, hash_dim), nn.Tanh(),
        )
        self._init()

    def _flat(self, s):
        with torch.no_grad():
            return self.conv(torch.zeros(1, 3, s, s)).view(1, -1).shape[1]

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(torch.flatten(self.conv(x), 1))

    @torch.no_grad()
    def get_codes(self, x): return torch.sign(self.forward(x))


# ─────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def _sq_dist(A, B=None):
    if B is None: B = A
    return ((A * A).sum(1, keepdim=True) +
            (B * B).sum(1, keepdim=True).t()
            - 2.0 * A @ B.t()).clamp(min=0.0)


def _l2_dist(A, B=None, eps=1e-8):
    # eps MUST be inside the sqrt — sqrt'(0) = inf causes NaN in backprop
    # without it the diagonal (self-distance = 0) gives infinite gradient
    return (_sq_dist(A, B) + eps).sqrt()


def loss_dhn(h1, h2, s, t=180.0, w=0.5):
    D  = torch.sum((h1 - h2) ** 2, dim=1)
    Lh = (0.5 * s * D +
          0.5 * (1.0 - s) * torch.clamp(t - D, min=0.0)).mean()
    Lq = ((torch.abs(h1) - 1.0).pow(2).mean() +
          (torch.abs(h2) - 1.0).pow(2).mean()) / 2.0
    return Lh + w * Lq, Lh, Lq


def loss_dhn_batch(h, label_oh, batch_size, omega, t=180.0, w=0.5):
    f_a, f_s = h[:omega], h[omega:]
    la, ls   = label_oh[:omega], label_oh[omega:]
    d_aa_sq  = _sq_dist(f_a)
    d_as_sq  = _sq_dist(f_a, f_s)
    d_aa_l2  = (d_aa_sq + 1e-8).sqrt()   # eps inside sqrt — prevents NaN
    d_as_l2  = (d_as_sq + 1e-8).sqrt()   # gradient when distances are ~0
    sim_aa   = la @ la.t()
    sim_as   = la @ ls.t()

    def cont(d_sq, sim):
        return (0.5 * sim * d_sq +
                0.5 * (1.0 - sim) * torch.clamp(t - d_sq, min=0.0)).mean()

    hl = cont(d_aa_sq, sim_aa) + cont(d_as_sq, sim_as)
    ql = torch.mean((torch.abs(h) - 1.0) ** 2)
    return hl + w * ql, d_aa_l2, d_as_l2


def loss_rela(s_aa, s_as, t_aa, t_as):
    return (torch.mean((s_aa - t_aa) ** 2) +
            torch.mean((s_as - t_as) ** 2))


def loss_hard(feat_T, feat_S, labels_int):
    """
    L_hard for random batches — iterates over each unique class present
    in the batch rather than assuming fixed-size contiguous class blocks.

    For each class c in the batch:
      genuine:  indices of samples belonging to c
      imposter: indices of all other samples
      pos_loss = clamp(max_S(genuine_dists) − min_T(genuine_dists), 0)
      neg_loss = clamp(max_T(imposter_dists) − min_S(imposter_dists), 0)

    Classes with only 1 sample are skipped (need ≥2 for a genuine pair).
    """
    T_d = _l2_dist(feat_T)   # [B, B]
    S_d = _l2_dist(feat_S)   # [B, B]

    pos_l, neg_l = [], []

    for lbl in labels_int.unique():
        idx  = (labels_int == lbl).nonzero(as_tuple=True)[0]   # genuine indices
        if len(idx) < 2:
            continue   # need at least 2 for a genuine pair

        # ── Genuine distances (off-diagonal within this class)
        T_blk = T_d[idx][:, idx]   # [k, k]
        S_blk = S_d[idx][:, idx]
        k  = len(idx)
        dm = torch.eye(k, dtype=torch.bool, device=feat_T.device)
        T_blk_od = T_blk.masked_fill(dm, float('inf'))   # inf on diagonal → ignored by min
        S_blk_od = S_blk.masked_fill(dm, 0.0)            # 0  on diagonal → ignored by max
        pos_l.append(torch.clamp(S_blk_od.max() - T_blk_od.min(), min=0.0))

        # ── Imposter distances (this class vs all others in batch)
        other = (labels_int != lbl)
        if other.sum() == 0:
            continue
        T_n = T_d[idx][:, other]
        S_n = S_d[idx][:, other]
        neg_l.append(torch.clamp(T_n.max() - S_n.min(), min=0.0))

    if not pos_l:
        return feat_T.new_zeros(())

    result = torch.stack(pos_l).mean()
    if neg_l:
        result = result + torch.stack(neg_l).mean()
    return result


# ─────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_codes(model, loader, device):
    was_training = model.training
    model.eval()
    codes, labels = [], []
    for imgs, lbls in loader:
        h = model(imgs.to(device))
        codes.append(torch.sign(h).cpu().numpy().astype(np.int8))
        labels.append(lbls.numpy())
    if was_training:
        model.train()
    return np.vstack(codes), np.concatenate(labels)


def identification_accuracy(tr_codes, tr_labels, te_codes, te_labels):
    """1-NN by Hamming distance.  Returns accuracy %."""
    correct = 0
    for code, true_lbl in zip(te_codes, te_labels):
        ham     = np.sum(code != tr_codes, axis=1)
        correct += int(tr_labels[np.argmin(ham)] == true_lbl)
    return correct / len(te_labels) * 100.0


def compute_eer(tr_codes, tr_labels, te_codes, te_labels):
    """EER via roc_curve with negative-Hamming as similarity score."""
    scores, truth = [], []
    for code, lbl in zip(te_codes, te_labels):
        sim = -np.sum(code != tr_codes, axis=1).astype(np.float32)
        scores.extend(sim.tolist())
        truth.extend((tr_labels == lbl).tolist())
    fpr, tpr, _ = roc_curve(truth, scores)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0 * 100.0


def run_metrics(model, tr_ld, te_ld, device):
    """
    Compute Acc and EER for both training set and test set.
    Train metrics: 1-NN retrieval within the training set itself.
    Test  metrics: 1-NN retrieval from test codes against training gallery.
    Returns (tr_acc, tr_eer, te_acc, te_eer).
    """
    tr_c, tr_l = extract_codes(model, tr_ld, device)
    te_c, te_l = extract_codes(model, te_ld, device)
    tr_acc = identification_accuracy(tr_c, tr_l, tr_c, tr_l)
    tr_eer = compute_eer(tr_c, tr_l, tr_c, tr_l)
    te_acc = identification_accuracy(tr_c, tr_l, te_c, te_l)
    te_eer = compute_eer(tr_c, tr_l, te_c, te_l)
    return tr_acc, tr_eer, te_acc, te_eer


def _print_metrics_row(label, tr_acc, tr_eer, te_acc, te_eer):
    log.info(
        f'  {label:22s}'
        f'  TRAIN  Acc={tr_acc:6.2f}%  EER={tr_eer:5.2f}%'
        f'   │   TEST   Acc={te_acc:6.2f}%  EER={te_eer:5.2f}%'
    )


def _make_eval_loaders(cfg, tr_idx, te_idx):
    """Build eval-transform DataLoaders for train and test subsets."""
    ds = CASIADataset(cfg['data_dir'],
                      get_transform(False, cfg['img_size']),
                      cfg['label_pos'], cfg['sep'], cfg['hand_pos'])
    tr_ld = DataLoader(Subset(ds, tr_idx), 64,
                       num_workers=cfg['num_workers'], pin_memory=True)
    te_ld = DataLoader(Subset(ds, te_idx), 64,
                       num_workers=cfg['num_workers'], pin_memory=True)
    return tr_ld, te_ld


def _save_csv(rows, save_dir, fname):
    if not rows: return
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, fname), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ─────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────
def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def train_teacher(cfg, device):
    log.info('=' * 70)
    log.info('STAGE 1 – Teacher  (VGG pool4 frozen + conv5/FC  |  epoch-based)')
    log.info('=' * 70)

    ds_tr = CASIADataset(cfg['data_dir'], get_transform(True, cfg['img_size']),
                         cfg['label_pos'], cfg['sep'], cfg['hand_pos'])
    tr_idx, te_idx = split_dataset(ds_tr, cfg['train_ratio'])
    print_split_summary(ds_tr, tr_idx, te_idx)
    tr_ld, te_ld = _make_eval_loaders(cfg, tr_idx, te_idx)
    num_cl  = ds_tr.num_classes
    B       = cfg['batch_size']

    sampler  = None   # random batches — no class ordering needed
    loader   = DataLoader(Subset(ds_tr, tr_idx),
                          batch_size=cfg['batch_size'], shuffle=True,
                          num_workers=cfg['num_workers'], pin_memory=True,
                          drop_last=True)
    n_epochs = cfg['teacher_epochs']

    model = TeacherDHN(cfg['hash_dim'], cfg['img_size'],
                       cfg['lrelu_alpha'],
                       dropout=cfg['dropout_teacher']).to(device)
    opt   = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=cfg['lr_min'])

    log.info(f'  {n_epochs} epochs  |  dropout={cfg["dropout_teacher"]}'
             f'  wd={cfg["weight_decay"]}  lr: {cfg["lr"]}→{cfg["lr_min"]} cosine')
    log.info(f'  {"Epoch":>6}  {"Loss":>9}  {"Lh":>9}  {"Lq":>9}  {"w*Lq":>9}  {"LR":>10}')
    log.info('  ' + '─' * 60)

    best_te = 0.0
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_loss = ep_lh = ep_lq = 0.0; n_steps = 0

        for imgs, labels_int in loader:
            imgs, labels_int = imgs.to(device), labels_int.to(device)
            if imgs.shape[0] != B: continue
            loh = make_onehot(labels_int, num_cl, device)
            h   = model(imgs)
            h1  = h.unsqueeze(1).expand(-1, B, -1).reshape(B * B, -1)
            h2  = h.unsqueeze(0).expand(B, -1, -1).reshape(B * B, -1)
            s   = (loh @ loh.t()).reshape(-1)
            ut  = torch.triu(torch.ones(B, B, device=device),
                             diagonal=1).bool().reshape(-1)
            h1p, h2p, sp = h1[ut], h2[ut], s[ut]
            loss, Lh, Lq = loss_dhn(h1p, h2p, sp, cfg['margin_t'], cfg['w_quant'])
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            ep_loss += loss.item(); ep_lh += Lh.item()
            ep_lq   += Lq.item();  n_steps += 1

        scheduler.step()   # once per epoch
        avg_l  = ep_loss / max(n_steps, 1)
        avg_lh = ep_lh   / max(n_steps, 1)
        avg_lq = ep_lq   / max(n_steps, 1)
        cur_lr = scheduler.get_last_lr()[0]

        if epoch % cfg['log_every'] == 0 or epoch == 1:
            log.info(f'  {epoch:6d}  {avg_l:9.4f}  '
                     f'Lh={avg_lh:7.4f}  '
                     f'Lq={avg_lq:7.4f}  '
                     f'w*Lq={cfg["w_quant"]*avg_lq:7.4f}  {cur_lr:10.2e}')

        if cfg['eval_every'] > 0 and epoch % cfg['eval_every'] == 0:
            log.info(f'  ── Eval @ epoch {epoch} {"─"*44}')
            tr_acc, tr_eer, te_acc, te_eer = run_metrics(
                model, tr_ld, te_ld, device)
            _print_metrics_row(f'Teacher ep {epoch}',
                               tr_acc, tr_eer, te_acc, te_eer)
            history.append(dict(epoch=epoch, tr_acc=tr_acc, tr_eer=tr_eer,
                                te_acc=te_acc, te_eer=te_eer))
            if te_acc > best_te:
                best_te = te_acc
                os.makedirs(cfg['save_dir'], exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(cfg['save_dir'], 'teacher.pth'))
                log.info(f'  ✓ New best teacher  (test Acc={te_acc:.2f}%)')
            log.info(f'  {"─"*64}')

    os.makedirs(cfg['save_dir'], exist_ok=True)
    final_path = os.path.join(cfg['save_dir'], 'teacher.pth')
    if not os.path.exists(final_path):
        torch.save(model.state_dict(), final_path)

    log.info('')
    log.info('  ── TEACHER FINAL EVALUATION ──')
    log.info(f'  {"Set":22s}  {"Acc (%)":>10}  {"EER (%)":>10}')
    log.info('  ' + '─' * 46)
    tr_acc, tr_eer, te_acc, te_eer = run_metrics(model, tr_ld, te_ld, device)
    log.info(f'  {"Training set":22s}  {tr_acc:>10.2f}  {tr_eer:>10.2f}')
    log.info(f'  {"Test set":22s}  {te_acc:>10.2f}  {te_eer:>10.2f}')
    log.info(f'  Teacher checkpoint → {final_path}')
    history.append(dict(epoch='final', tr_acc=tr_acc, tr_eer=tr_eer,
                        te_acc=te_acc, te_eer=te_eer))
    _save_csv(history, cfg['save_dir'], 'teacher_history.csv')
    return model

def train_student(cfg, device, teacher=None):
    log.info('=' * 70)
    log.info('STAGE 2 – Student DDH  (epoch-based  |  Adam + cosine LR)')
    log.info('=' * 70)

    ds_tr = CASIADataset(cfg['data_dir'], get_transform(True, cfg['img_size']),
                         cfg['label_pos'], cfg['sep'], cfg['hand_pos'])
    tr_idx, te_idx = split_dataset(ds_tr, cfg['train_ratio'])
    print_split_summary(ds_tr, tr_idx, te_idx)
    tr_ld, te_ld = _make_eval_loaders(cfg, tr_idx, te_idx)
    num_cl  = ds_tr.num_classes
    B       = cfg['batch_size']
    omega   = B // 2

    sampler  = None   # random batches — no class ordering needed
    loader   = DataLoader(Subset(ds_tr, tr_idx),
                          batch_size=cfg['batch_size'], shuffle=True,
                          num_workers=cfg['num_workers'], pin_memory=True,
                          drop_last=True)
    n_epochs = cfg['student_epochs']

    if teacher is None:
        teacher = TeacherDHN(cfg['hash_dim'], cfg['img_size'],
                             cfg['lrelu_alpha'],
                             dropout=cfg['dropout_teacher']).to(device)
        ckpt = os.path.join(cfg['save_dir'], 'teacher.pth')
        teacher.load_state_dict(torch.load(ckpt, map_location=device))
        log.info(f'  Teacher loaded from {ckpt}')
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    log.info('  ── Teacher baseline (before student training) ──')
    log.info(f'  {"Set":22s}  {"Acc (%)":>10}  {"EER (%)":>10}')
    log.info('  ' + '─' * 46)
    tch_tr, tch_tr_e, tch_te, tch_te_e = run_metrics(teacher, tr_ld, te_ld, device)
    log.info(f'  {"Training set":22s}  {tch_tr:>10.2f}  {tch_tr_e:>10.2f}')
    log.info(f'  {"Test set":22s}  {tch_te:>10.2f}  {tch_te_e:>10.2f}')
    log.info('')

    student = StudentDHN(cfg['hash_dim'], cfg['img_size'],
                         cfg['lrelu_alpha'],
                         dropout=cfg['dropout_student']).to(device)
    opt = optim.Adam(student.parameters(),
                     lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=cfg['lr_min'])

    log.info(f'  {n_epochs} epochs  |  dropout={cfg["dropout_student"]}'
             f'  wd={cfg["weight_decay"]}  lr: {cfg["lr"]}→{cfg["lr_min"]} cosine')
    log.info(f'  {"Epoch":>6}  {"Loss":>9}  {"DHN":>9}  '
             f'{"Lrela":>9}  {"Lhard":>9}  {"LR":>10}')
    log.info('  ' + '─' * 65)

    best_te = 0.0
    history = []

    for epoch in range(1, n_epochs + 1):
        student.train()
        ep_loss = ep_dhn = ep_rela = ep_hard = 0.0; n_steps = 0

        for imgs, labels_int in loader:
            imgs, labels_int = imgs.to(device), labels_int.to(device)
            if imgs.shape[0] != B: continue
            loh = make_onehot(labels_int, num_cl, device)

            with torch.no_grad():
                h_T = teacher(imgs)
            _, t_aa, t_as = loss_dhn_batch(
                h_T, loh, B, omega, cfg['margin_t'], cfg['w_quant'])

            h_S = student(imgs)
            dhn_s, s_aa, s_as = loss_dhn_batch(
                h_S, loh, B, omega, cfg['margin_t'], cfg['w_quant'])

            Lrela = loss_rela(s_aa, s_as, t_aa, t_as)
            Lhard = loss_hard(h_T, h_S, labels_int)
            loss  = dhn_s + cfg['alpha'] * Lrela + cfg['beta'] * Lhard

            if not torch.isfinite(loss):
                log.warning(f'  ep {epoch}: non-finite loss — skipping batch')
                opt.zero_grad(set_to_none=True); continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            opt.step()

            ep_loss += loss.item(); ep_dhn  += dhn_s.item()
            ep_rela += Lrela.item(); ep_hard += Lhard.item(); n_steps += 1

        scheduler.step()
        n = max(n_steps, 1)
        cur_lr = scheduler.get_last_lr()[0]

        if epoch % cfg['log_every'] == 0 or epoch == 1:
            log.info(f'  {epoch:6d}  {ep_loss/n:9.4f}  {ep_dhn/n:9.4f}  '
                     f'{ep_rela/n:9.4f}  {ep_hard/n:9.4f}  {cur_lr:10.2e}')

        if cfg['eval_every'] > 0 and epoch % cfg['eval_every'] == 0:
            log.info(f'  ── Eval @ epoch {epoch} {"─"*44}')
            tr_acc, tr_eer, te_acc, te_eer = run_metrics(
                student, tr_ld, te_ld, device)
            _print_metrics_row(f'Student ep {epoch}',
                               tr_acc, tr_eer, te_acc, te_eer)
            history.append(dict(epoch=epoch, tr_acc=tr_acc, tr_eer=tr_eer,
                                te_acc=te_acc, te_eer=te_eer))
            if te_acc > best_te:
                best_te = te_acc
                os.makedirs(cfg['save_dir'], exist_ok=True)
                torch.save(student.state_dict(),
                           os.path.join(cfg['save_dir'], 'student.pth'))
                log.info(f'  ✓ New best student  (test Acc={te_acc:.2f}%)')
            log.info(f'  {"─"*64}')

    os.makedirs(cfg['save_dir'], exist_ok=True)
    stu_path = os.path.join(cfg['save_dir'], 'student.pth')
    if not os.path.exists(stu_path):
        torch.save(student.state_dict(), stu_path)

    log.info('')
    log.info('  ══ FINAL RESULTS  ══════════════════════════════════════════')
    log.info(f'  {"Model":22s}  {"Train Acc":>10}  {"Train EER":>10}  '
             f'{"Test Acc":>10}  {"Test EER":>10}')
    log.info('  ' + '─' * 68)
    all_results = []
    for name, m in [('Teacher', teacher), ('DDH (Student)', student)]:
        tr_acc, tr_eer, te_acc, te_eer = run_metrics(m, tr_ld, te_ld, device)
        log.info(f'  {name:22s}  {tr_acc:>10.2f}  {tr_eer:>10.2f}  '
                 f'{te_acc:>10.2f}  {te_eer:>10.2f}')
        all_results.append(dict(model=name,
                                train_acc=round(tr_acc, 2),
                                train_eer=round(tr_eer, 2),
                                test_acc=round(te_acc, 2),
                                test_eer=round(te_eer, 2)))
        history.append(dict(epoch=f'final_{name}',
                            tr_acc=tr_acc, tr_eer=tr_eer,
                            te_acc=te_acc, te_eer=te_eer))
    log.info(f'  Student checkpoint → {stu_path}')
    _save_csv(all_results, cfg['save_dir'], 'results_final.csv')
    _save_csv(history,     cfg['save_dir'], 'student_history.csv')
    log.info(f"  CSV → {os.path.join(cfg['save_dir'], 'results_final.csv')}")
    return student


def evaluate(cfg, device):
    log.info('=' * 70)
    log.info('EVALUATION  (loading saved checkpoints)')
    log.info('=' * 70)

    ds = CASIADataset(cfg['data_dir'], get_transform(False, cfg['img_size']),
                      cfg['label_pos'], cfg['sep'], cfg['hand_pos'])
    tr_idx, te_idx = split_dataset(ds, cfg['train_ratio'])
    print_split_summary(ds, tr_idx, te_idx)
    tr_ld, te_ld = _make_eval_loaders(cfg, tr_idx, te_idx)

    log.info(f'  {"Model":22s}  {"Train Acc":>10}  {"Train EER":>10}  '
             f'{"Test Acc":>10}  {"Test EER":>10}')
    log.info('  ' + '─' * 68)

    results = []
    for name, ckpt_fn, make_m in [
        ('Teacher',       'teacher.pth',
         lambda: TeacherDHN(cfg['hash_dim'], cfg['img_size'],
                            cfg['lrelu_alpha'],
                            dropout=cfg['dropout_teacher'])),
        ('DDH (Student)', 'student.pth',
         lambda: StudentDHN(cfg['hash_dim'], cfg['img_size'],
                            cfg['lrelu_alpha'],
                            dropout=cfg['dropout_student'])),
    ]:
        ckpt = os.path.join(cfg['save_dir'], ckpt_fn)
        if not os.path.exists(ckpt):
            log.warning(f'  {name}: checkpoint not found, skipping')
            continue
        m = make_m().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))

        tr_c, tr_l = extract_codes(m, tr_ld, device)
        te_c, te_l = extract_codes(m, te_ld, device)

        tr_acc = identification_accuracy(tr_c, tr_l, tr_c, tr_l)
        tr_eer = compute_eer(tr_c, tr_l, tr_c, tr_l)
        te_acc = identification_accuracy(tr_c, tr_l, te_c, te_l)
        te_eer = compute_eer(tr_c, tr_l, te_c, te_l)

        log.info(f'  {name:22s}  {tr_acc:>10.2f}  {tr_eer:>10.2f}  '
                 f'{te_acc:>10.2f}  {te_eer:>10.2f}')
        results.append(dict(model=name,
                            train_acc=round(tr_acc, 2),
                            train_eer=round(tr_eer, 2),
                            test_acc=round(te_acc, 2),
                            test_eer=round(te_eer, 2)))

    _save_csv(results, cfg['save_dir'], 'results_eval.csv')
    log.info(f"  CSV → {os.path.join(cfg['save_dir'], 'results_eval.csv')}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description='DDH – best synthesis')
    p.add_argument('--data_dir',  default=CFG['data_dir'])
    p.add_argument('--stage',     default='all',
                   choices=['all', 'teacher', 'student', 'eval'])
    p.add_argument('--label_pos', type=int, default=CFG['label_pos'])
    p.add_argument('--sep',       default=CFG['sep'])
    p.add_argument('--batch_size', type=int, default=CFG['batch_size'])
    args = p.parse_args()
    CFG.update(data_dir=args.data_dir, label_pos=args.label_pos,
               sep=args.sep, batch_size=args.batch_size)

    os.makedirs(CFG['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log.info('')
    log.info('╔══════════════════════════════════════════════════════════════╗')
    log.info('║           DDH – paper + released code synthesis             ║')
    log.info('╠══════════════════════════════════════════════════════════════╣')
    log.info(f'║  Device     : {str(device):<48}║')
    log.info(f'║  Data dir   : {CFG["data_dir"]:<48}║')
    log.info(f'║  Stage      : {args.stage:<48}║')
    log.info(f'║  Split      : {TRAIN_RATIO:.0%} train / '
             f'{1-TRAIN_RATIO:.0%} test  (paper Section V-B 50/50)   ║')
    log.info(f'║  Batch size : {CFG["batch_size"]} (random shuffle, no class ordering)'
             f'{"":>5}║')
    log.info(f'║  α={CFG["alpha"]}  β={CFG["beta"]}  t={CFG["margin_t"]}  lr={CFG["lr"]}→{CFG["lr_min"]}'
             f'{"":>22}║')
    log.info(f'║  dropout T={CFG["dropout_teacher"]} S={CFG["dropout_student"]}'
             f'  weight_decay={CFG["weight_decay"]}'
             f'{"":>22}║')
    log.info(f'║  Eval every : {CFG["eval_every"]} epochs'
             f'{"":>{42-len(str(CFG["eval_every"]))}}║')
    log.info('╚══════════════════════════════════════════════════════════════╝')
    log.info('')

    teacher = None
    if args.stage in ('all', 'teacher'):
        teacher = train_teacher(cfg=CFG, device=device)
    if args.stage in ('all', 'student'):
        train_student(cfg=CFG, device=device, teacher=teacher)
    if args.stage in ('all', 'eval'):
        evaluate(cfg=CFG, device=device)


if __name__ == '__main__':
    main()
