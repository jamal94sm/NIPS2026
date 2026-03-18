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
║    • 5,502 images total                                      ║
║    • 624 hands (subjects / classes)                          ║
║    • ~8–9 images per hand (two sessions, ~4–5 each)          ║
║                                                              ║
║  Applied split  (TRAIN_RATIO = 0.5):                         ║
║    • First 50% of each subject's images  →  TRAIN            ║
║    • Remaining 50%                       →  TEST             ║
║    • Subject with 8 images: 4 train / 4 test                 ║
║    • Subject with 9 images: 4 train / 5 test                 ║
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
                         #   (a) Subdirectory per subject:  DATA_DIR/001/img1.bmp
                         #   (b) Flat folder, label in name: DATA_DIR/001_L_01.bmp

LABEL_POS   = 0          # Which token in the filename is the subject ID.
                         # e.g. '001_L_01.bmp' split by '_' → token 0 = '001'

SEP         = '_'        # Filename token separator.

# ── Train / Test split ───────────────────────────────────────────
# Paper Section V-B: "half … training set, remaining half … test set"
# CASIA-MS has ~8-9 images per hand → 4 train / 4-5 test per subject.
TRAIN_RATIO = 0.8        # 0.5 = 50% train / 50% test  (paper default)
                         # Change to e.g. 0.6 for 60/40.

# ── Regularisation ───────────────────────────────────────────────
# CASIA-MS has ~35-46 train images per subject, which causes severe
# overfitting when the FC head has no dropout.  These two controls
# are the single most important change for generalisation.
DROPOUT_TEACHER = 0.5    # dropout rate for teacher FC layers
DROPOUT_STUDENT = 0.4    # dropout rate for student FC layer
WEIGHT_DECAY    = 1e-4   # L2 regularisation on all trainable weights

# ── Batch composition ────────────────────────────────────────────
# With ~35-46 train images per subject we can afford more per-class
# images and more classes per batch, giving a richer contrastive signal.
# batch_size = NUM_PER × N_CLS_PER_BATCH
NUM_PER         = 6      # images per class per batch  (≤ min train per subj)
N_CLS_PER_BATCH = 25     # classes per batch  →  batch = 6 × 25 = 150

# ── Training ─────────────────────────────────────────────────────
# With a larger dataset (35-46 train images vs the paper's 4-10),
# more iterations are needed for the model to see enough variety.
TEACHER_ITERS = 40000    # increased from 10,000
STUDENT_ITERS = 40000    # increased from 10,000
LR            = 1e-3     # Adam initial learning rate
# LR is annealed via cosine schedule down to LR_MIN over all iters
LR_MIN        = 1e-5     # cosine schedule final LR

# ── Mid-training evaluation frequency ────────────────────────────
# Every EVAL_EVERY_K iterations, Acc + EER are computed and printed
# for BOTH the training set and test set.  This is the only place
# metrics are printed during training — NOT at every loss step.
# Set to 0 to skip mid-training eval (final table is always printed).
EVAL_EVERY_K = 4000     # ← change K here  (e.g. 2000, 8000, …)

# ── Loss-print frequency ─────────────────────────────────────────
LOG_EVERY  = 1000

# ── Output directory ─────────────────────────────────────────────
SAVE_DIR = './ckpt_best'

# ══════════════════════════════════════════════════════════════════
# Do not edit below this line unless you know what you are changing
# ══════════════════════════════════════════════════════════════════

CFG = dict(
    data_dir        = DATA_DIR,
    label_pos       = LABEL_POS,
    sep             = SEP,
    train_ratio     = TRAIN_RATIO,
    hash_dim        = 128,
    img_size        = 128,
    lrelu_alpha     = 0.2,
    margin_t        = 180.0,
    w_quant         = 0.5,
    alpha           = 1.0,
    beta            = 0.8,
    dropout_teacher = DROPOUT_TEACHER,
    dropout_student = DROPOUT_STUDENT,
    weight_decay    = WEIGHT_DECAY,
    lr_min          = LR_MIN,
    num_per         = NUM_PER,
    n_cls_per_batch = N_CLS_PER_BATCH,
    batch_size      = NUM_PER * N_CLS_PER_BATCH,
    teacher_iters   = TEACHER_ITERS,
    student_iters   = STUDENT_ITERS,
    lr              = LR,
    eval_every      = EVAL_EVERY_K,
    log_every       = LOG_EVERY,
    num_workers     = 4,
    save_dir        = SAVE_DIR,
)

# ─────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────
class CASIADataset(Dataset):
    EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.pgm', '.tiff'}

    def __init__(self, root, transform=None, label_pos=0, sep='_'):
        self.transform    = transform
        self.paths        = []
        self.labels       = []
        self.idx_by_label = defaultdict(list)

        subdirs = sorted(d for d in os.listdir(root)
                         if os.path.isdir(os.path.join(root, d)))
        if subdirs:
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
            files = sorted(fn for fn in os.listdir(root)
                           if os.path.splitext(fn)[1].lower() in self.EXTS)
            if not files:
                raise RuntimeError(f'No images found in {root!r}')
            subjects = sorted({fn.split(sep)[label_pos] for fn in files})
            lbl_map  = {s: i for i, s in enumerate(subjects)}
            for fn in files:
                sid = fn.split(sep)[label_pos]
                lbl = lbl_map[sid]
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
    """VGG pool4 frozen + trainable conv5 block + FC head.
    Dropout added to all intermediate FC layers to prevent overfitting
    on datasets with many images per subject (e.g. CASIA-MS ~35-46/subj).
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
        # Dropout between every intermediate FC layer is the primary
        # defence against overfitting on large-per-subject datasets.
        self.fc = nn.Sequential(
            nn.Linear(flat, 4096),
            nn.BatchNorm1d(4096, eps=1e-5), LReLU(alpha), nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096, eps=1e-5), LReLU(alpha), nn.Dropout(dropout),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048, eps=1e-5), LReLU(alpha), nn.Dropout(dropout),
            nn.Linear(2048, hash_dim), nn.Tanh(),
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


def loss_hard(feat_T, feat_S, label_oh, batch_size, num_per):
    T_d   = _l2_dist(feat_T)   # eps already inside sqrt — no NaN gradient
    S_d   = _l2_dist(feat_S)
    lbl_m = label_oh @ label_oh.t()
    T_pos = lbl_m * T_d;  T_neg = (1.0 - lbl_m) * T_d
    S_pos = lbl_m * S_d;  S_neg = (1.0 - lbl_m) * S_d
    pos_l, neg_l = [], []
    n_blk = batch_size // num_per

    # Pre-build diagonal mask for genuine blocks (size num_per × num_per).
    # The diagonal is the self-distance ≈ 0 (eps only).
    # Without masking, T_blk.min() = 0 always → genuine constraint is
    # always violated → loss is always large → gradients explode.
    diag_mask = torch.eye(num_per, dtype=torch.bool, device=feat_T.device)

    for i in range(n_blk):
        s, e = i * num_per, (i + 1) * num_per

        # ── Genuine block: off-diagonal pairs only
        T_blk = T_pos[s:e, s:e]
        S_blk = S_pos[s:e, s:e]
        # Mask diagonal out of min/max so self-distances don't dominate
        T_blk_od = T_blk.masked_fill(diag_mask, float('inf'))   # for min
        S_blk_od = S_blk.masked_fill(diag_mask, 0.0)            # for max
        pos_l.append(torch.clamp(S_blk_od.max() - T_blk_od.min(), min=0.0))

        # ── Imposter block: all columns outside class i
        T_n = torch.cat([T_neg[s:e, :s], T_neg[s:e, e:]], dim=1)
        S_n = torch.cat([S_neg[s:e, :s], S_neg[s:e, e:]], dim=1)
        neg_l.append(torch.clamp(T_n.max() - S_n.min(), min=0.0))

    return torch.stack(pos_l).mean() + torch.stack(neg_l).mean()


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
                      cfg['label_pos'], cfg['sep'])
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
    log.info('STAGE 1 – Teacher  (VGG pool4 frozen + conv5/FC  |  Adam + cosine LR)')
    log.info('=' * 70)

    # ── Build datasets
    ds_tr = CASIADataset(cfg['data_dir'], get_transform(True,  cfg['img_size']),
                         cfg['label_pos'], cfg['sep'])
    tr_idx, te_idx = split_dataset(ds_tr, cfg['train_ratio'])
    print_split_summary(ds_tr, tr_idx, te_idx)
    tr_ld, te_ld = _make_eval_loaders(cfg, tr_idx, te_idx)
    num_cl = ds_tr.num_classes
    B      = cfg['batch_size']

    sampler = ClassOrderedBatchSampler(
        ds_tr.idx_by_label, tr_idx, cfg['num_per'], cfg['n_cls_per_batch'])
    loader = DataLoader(ds_tr, batch_sampler=sampler,
                        num_workers=cfg['num_workers'], pin_memory=True)

    model = TeacherDHN(cfg['hash_dim'], cfg['img_size'],
                       cfg['lrelu_alpha'],
                       dropout=cfg['dropout_teacher']).to(device)
    opt   = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    # Cosine LR annealing: smoothly reduces LR from lr to lr_min
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg['teacher_iters'], eta_min=cfg['lr_min'])

    model.train()
    it      = _infinite(loader)
    best_te = 0.0
    history = []

    log.info(f'  dropout={cfg["dropout_teacher"]}  weight_decay={cfg["weight_decay"]}'
             f'  lr: {cfg["lr"]} → {cfg["lr_min"]} (cosine)')
    log.info(f'  {"Iter":>6}  {"Loss":>9}  {"Lh":>9}  {"Lq":>9}  {"LR":>10}')
    log.info('  ' + '─' * 50)

    for step in range(1, cfg['teacher_iters'] + 1):
        imgs, labels_int = next(it)
        imgs, labels_int = imgs.to(device), labels_int.to(device)
        if imgs.shape[0] != B: continue

        loh = make_onehot(labels_int, num_cl, device)
        h   = model(imgs)

        h1 = h.unsqueeze(1).expand(-1, B, -1).reshape(B * B, -1)
        h2 = h.unsqueeze(0).expand(B, -1, -1).reshape(B * B, -1)
        s  = (loh @ loh.t()).reshape(-1)
        ut = torch.triu(torch.ones(B, B, device=device),
                        diagonal=1).bool().reshape(-1)
        h1p, h2p, sp = h1[ut], h2[ut], s[ut]

        loss, Lh, Lq = loss_dhn(h1p, h2p, sp, cfg['margin_t'], cfg['w_quant'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scheduler.step()

        if step % cfg['log_every'] == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            log.info(f'  {step:6d}  {loss.item():9.4f}  '
                     f'{Lh.item():9.4f}  {Lq.item():9.4f}  {current_lr:10.2e}')

        # ── Periodic train + test evaluation
        if cfg['eval_every'] > 0 and step % cfg['eval_every'] == 0:
            log.info(f'  ── Eval @ iter {step} {"─"*45}')
            tr_acc, tr_eer, te_acc, te_eer = run_metrics(
                model, tr_ld, te_ld, device)
            _print_metrics_row(f'Teacher iter {step}',
                               tr_acc, tr_eer, te_acc, te_eer)
            history.append(dict(iter=step, tr_acc=tr_acc, tr_eer=tr_eer,
                                te_acc=te_acc, te_eer=te_eer))
            if te_acc > best_te:
                best_te = te_acc
                os.makedirs(cfg['save_dir'], exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(cfg['save_dir'], 'teacher.pth'))
                log.info(f'  ✓ New best teacher  (test Acc={te_acc:.2f}%)')
            log.info(f'  {"─"*64}')

    # ── Final save
    os.makedirs(cfg['save_dir'], exist_ok=True)
    final_path = os.path.join(cfg['save_dir'], 'teacher.pth')
    if not os.path.exists(final_path):
        torch.save(model.state_dict(), final_path)

    # ── Final evaluation
    log.info('')
    log.info('  ── TEACHER FINAL EVALUATION ──')
    log.info(f'  {"Set":22s}  {"Acc (%)":>10}  {"EER (%)":>10}')
    log.info('  ' + '─' * 46)
    tr_acc, tr_eer, te_acc, te_eer = run_metrics(model, tr_ld, te_ld, device)
    log.info(f'  {"Training set":22s}  {tr_acc:>10.2f}  {tr_eer:>10.2f}')
    log.info(f'  {"Test set":22s}  {te_acc:>10.2f}  {te_eer:>10.2f}')
    log.info(f'  Teacher checkpoint → {final_path}')

    history.append(dict(iter='final', tr_acc=tr_acc, tr_eer=tr_eer,
                        te_acc=te_acc, te_eer=te_eer))
    _save_csv(history, cfg['save_dir'], 'teacher_history.csv')
    return model


def train_student(cfg, device, teacher=None):
    log.info('=' * 70)
    log.info('STAGE 2 – Student DDH  (Adam  |  α=1  β=0.8  |  10 000 iters)')
    log.info('=' * 70)

    # ── Build datasets
    ds_tr = CASIADataset(cfg['data_dir'], get_transform(True,  cfg['img_size']),
                         cfg['label_pos'], cfg['sep'])
    tr_idx, te_idx = split_dataset(ds_tr, cfg['train_ratio'])
    print_split_summary(ds_tr, tr_idx, te_idx)
    tr_ld, te_ld = _make_eval_loaders(cfg, tr_idx, te_idx)
    num_cl = ds_tr.num_classes
    B      = cfg['batch_size']
    omega  = B // 2

    sampler = ClassOrderedBatchSampler(
        ds_tr.idx_by_label, tr_idx, cfg['num_per'], cfg['n_cls_per_batch'])
    loader = DataLoader(ds_tr, batch_sampler=sampler,
                        num_workers=cfg['num_workers'], pin_memory=True)

    # ── Load teacher
    if teacher is None:
        teacher = TeacherDHN(cfg['hash_dim'], cfg['img_size'],
                             cfg['lrelu_alpha'],
                             dropout=cfg['dropout_teacher']).to(device)
        ckpt = os.path.join(cfg['save_dir'], 'teacher.pth')
        teacher.load_state_dict(torch.load(ckpt, map_location=device))
        log.info(f'  Teacher loaded from {ckpt}')
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    # ── Baseline: teacher performance before student training starts
    log.info('  ── Teacher baseline (before student training) ──')
    log.info(f'  {"Set":22s}  {"Acc (%)":>10}  {"EER (%)":>10}')
    log.info('  ' + '─' * 46)
    tch_tr_acc, tch_tr_eer, tch_te_acc, tch_te_eer = run_metrics(
        teacher, tr_ld, te_ld, device)
    log.info(f'  {"Training set":22s}  {tch_tr_acc:>10.2f}  {tch_tr_eer:>10.2f}')
    log.info(f'  {"Test set":22s}  {tch_te_acc:>10.2f}  {tch_te_eer:>10.2f}')
    log.info('')

    student = StudentDHN(cfg['hash_dim'], cfg['img_size'],
                         cfg['lrelu_alpha'],
                         dropout=cfg['dropout_student']).to(device)
    opt = optim.Adam(student.parameters(),
                     lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg['student_iters'], eta_min=cfg['lr_min'])

    student.train()
    it      = _infinite(loader)
    best_te = 0.0
    history = []

    log.info(f'  dropout={cfg["dropout_student"]}  weight_decay={cfg["weight_decay"]}'
             f'  lr: {cfg["lr"]} → {cfg["lr_min"]} (cosine)')
    log.info(f'  {"Iter":>6}  {"Loss":>9}  {"DHN":>9}  '
             f'{"Lrela":>9}  {"Lhard":>9}  {"LR":>10}')
    log.info('  ' + '─' * 65)

    for step in range(1, cfg['student_iters'] + 1):
        imgs, labels_int = next(it)
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
        Lhard = loss_hard(h_T, h_S, loh, B, cfg['num_per'])
        loss  = dhn_s + cfg['alpha'] * Lrela + cfg['beta'] * Lhard

        # Guard: skip batch and warn if any loss component is NaN/Inf
        if not torch.isfinite(loss):
            log.warning(f'  iter {step}: non-finite loss '
                        f'(DHN={dhn_s.item():.4f} '
                        f'Lrela={Lrela.item():.4f} '
                        f'Lhard={Lhard.item():.4f}) — skipping batch')
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        opt.step()
        scheduler.step()

        if step % cfg['log_every'] == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            log.info(f'  {step:6d}  {loss.item():9.4f}  '
                     f'{dhn_s.item():9.4f}  '
                     f'{Lrela.item():9.4f}  '
                     f'{Lhard.item():9.4f}  {current_lr:10.2e}')

        # ── Periodic train + test evaluation
        if cfg['eval_every'] > 0 and step % cfg['eval_every'] == 0:
            log.info(f'  ── Eval @ iter {step} {"─"*45}')
            tr_acc, tr_eer, te_acc, te_eer = run_metrics(
                student, tr_ld, te_ld, device)
            _print_metrics_row(f'Student iter {step}',
                               tr_acc, tr_eer, te_acc, te_eer)
            history.append(dict(iter=step, tr_acc=tr_acc, tr_eer=tr_eer,
                                te_acc=te_acc, te_eer=te_eer))
            if te_acc > best_te:
                best_te = te_acc
                os.makedirs(cfg['save_dir'], exist_ok=True)
                torch.save(student.state_dict(),
                           os.path.join(cfg['save_dir'], 'student.pth'))
                log.info(f'  ✓ New best student  (test Acc={te_acc:.2f}%)')
            log.info(f'  {"─"*64}')

    # ── Final save
    os.makedirs(cfg['save_dir'], exist_ok=True)
    stu_path = os.path.join(cfg['save_dir'], 'student.pth')
    if not os.path.exists(stu_path):
        torch.save(student.state_dict(), stu_path)

    # ── Final comparison table: Teacher vs Student/DDH
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
        history.append(dict(iter=f'final_{name}',
                            tr_acc=tr_acc, tr_eer=tr_eer,
                            te_acc=te_acc, te_eer=te_eer))

    log.info(f'  Student checkpoint → {stu_path}')
    _save_csv(all_results, cfg['save_dir'], 'results_final.csv')
    _save_csv(history,     cfg['save_dir'], 'student_history.csv')
    log.info(f"  CSV → {os.path.join(cfg['save_dir'], 'results_final.csv')}")
    return student


# ─────────────────────────────────────────────────────────────────
# STANDALONE EVALUATION  (--stage eval)
# ─────────────────────────────────────────────────────────────────
def evaluate(cfg, device):
    log.info('=' * 70)
    log.info('EVALUATION  (loading saved checkpoints)')
    log.info('=' * 70)

    ds = CASIADataset(cfg['data_dir'], get_transform(False, cfg['img_size']),
                      cfg['label_pos'], cfg['sep'])
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
    p.add_argument('--num_per',   type=int, default=CFG['num_per'])
    p.add_argument('--n_cls',     type=int, default=CFG['n_cls_per_batch'])
    args = p.parse_args()
    CFG.update(data_dir=args.data_dir, label_pos=args.label_pos,
               sep=args.sep, num_per=args.num_per,
               n_cls_per_batch=args.n_cls)
    CFG['batch_size'] = CFG['num_per'] * CFG['n_cls_per_batch']

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
    log.info(f'║  Batch      : {CFG["num_per"]} per class × {CFG["n_cls_per_batch"]} classes = {CFG["batch_size"]}'
             f'{"":>{30 - len(str(CFG["batch_size"]))}}║')
    log.info(f'║  α={CFG["alpha"]}  β={CFG["beta"]}  t={CFG["margin_t"]}  lr={CFG["lr"]}→{CFG["lr_min"]}'
             f'{"":>22}║')
    log.info(f'║  dropout T={CFG["dropout_teacher"]} S={CFG["dropout_student"]}'
             f'  weight_decay={CFG["weight_decay"]}'
             f'{"":>22}║')
    log.info(f'║  Eval every : {CFG["eval_every"]} iters  (K={CFG["eval_every"]})'
             f'{"":>{30-len(str(CFG["eval_every"]))}}║')
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
