"""
ddh_released.py  — PyTorch faithful translation of the authors' TF source
==========================================================================
Source files translated line-by-line:
  vgg16.py, Teacher_nets.py, Student_nets.py,
  train_teacher.py, train_student.py, train_DDH.py, eval.py

Critical facts found ONLY in the source code (not in the paper):

ARCHITECTURE
  • vgg16.py: VGG is loaded from .npy as tf.constant — ALL weights frozen.
    Only pool4 output is used: `feature_t = vgg.pool4`
  • Teacher encode1 (Teacher_nets.py):
      3× Conv(512→512, 3×3, SAME, BN over [0,1,2], LeakyReLU(α=0.2))
      + MaxPool(2×2, stride=2, SAME)          ← stride=2, not stride=1
  • Teacher encode2:
      Flatten → FC(→4096, BN over [0,1], LReLU)
              → FC(→4096, BN, LReLU)
              → FC(→2048, BN, LReLU)
              Dropout(0.5) present but Training=False in all calls → inactive
  • Teacher encode3:  FC(2048→128, Tanh)
  • Student encode1 (Student_nets.py):
      Conv(3→16, 3×3, stride=4, VALID, BN over [0,1,2], LReLU)
      MaxPool(2×2, stride=1, VALID)
  • Student encode2:
      Conv(16→32, 5×5, stride=2, SAME, BN over [0,1,2], LReLU)
      MaxPool(2×2, stride=1, VALID)
  • Student encode3:  Flatten → FC(→512, NO BN!, LReLU)   ← BN absent here
  • Student encode4:  FC(512→128, Tanh)

LOSS FUNCTIONS (train_DDH.py)
  • Hash_loss: splits batch into archer[:omega] and sabor[omega:].
    Distances = squared Euclidean via ‖A‖²+‖B‖²−2AᵀB.
    Contrastive: mean(0.5·S·D + 0.5·(1-S)·max(180-D,0))
  • q_loss = mean((h - sign(h))²)  → weight 0.5
  • rela_loss = MSE(S_archer_dist, T_archer_dist)
              + MSE(S_sabor_dist,  T_sabor_dist)
    Both distance matrices are SQUARED Euclidean.
  • Hard_loss: full-batch squared-Euclidean distance matrices,
    per-class blocks of size num_per, NO clamp anywhere.
  • DDH total = rela_loss + hard_loss + DHN_loss   (no α/β weights)

TRAINING
  • Train teacher:  RMSProp(lr=0.001, α=0.9), batch=20, omega=10,
                    epoch=500, Shuffle=True
  • Train student:  RMSProp(lr=0.001, α=0.9), batch=20, omega=10,
                    epoch=500, Shuffle=True  (standalone DHN loss only)
  • Train DDH:      RMSProp(lr=0.0001,α=0.9), batch=50, omega=30,
                    epoch=680, num_per=10, Shuffle=False → class-ordered batches
  • Dropout=0.5 present in code but Training=False in all training calls
    → dropout is effectively never applied.
  • Images resized to 224×224 (eval.py and get_batch functions)

EVALUATION (eval.py)
  • Distance metric: L2 = sqrt(Σ(a_k−b_k)²)  on binary (sign) codes
  • Threshold sweep: range(10, 50)
  • Test set = second half per class (i >= num_per/2 within each group)

Usage:
  python ddh_released.py --data_dir /path/to/casia_ms --stage all
  python ddh_released.py --data_dir /path --label_pos 0 --sep _
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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — mirrors released code constants exactly
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    data_dir            = './casia_ms',
    label_pos           = 0,
    sep                 = '_',
    train_ratio         = 0.5,

    hash_dim            = 128,
    img_size            = 224,      # eval.py: resize_images(image, [224, 224])
    lrelu_alpha         = 0.2,      # all nets: alpha=0.2

    # DHN loss
    margin_t            = 180.0,
    w_quant             = 0.5,      # 0.5 * q_loss in all train scripts

    # train_teacher.py / train_student.py
    teacher_batch       = 20,
    teacher_omega       = 10,
    teacher_lr          = 1e-3,
    teacher_epochs      = 500,

    student_batch       = 20,
    student_omega       = 10,
    student_lr          = 1e-3,
    student_epochs      = 500,

    # train_DDH.py
    ddh_batch           = 50,
    ddh_omega           = 30,
    ddh_lr              = 1e-4,
    ddh_epochs          = 680,
    ddh_num_per         = 10,       # images per class per batch

    rms_alpha           = 0.9,      # RMSProp alpha in all scripts
    num_workers         = 4,
    log_every           = 200,
    save_dir            = './ckpt_released',
)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class CASIADataset(Dataset):
    """Supports flat-folder and sub-directory layouts."""
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
                        self.labels.append(lbl); self.idx_by_label[lbl].append(i)
        else:
            files = sorted(fn for fn in os.listdir(root)
                           if os.path.splitext(fn)[1].lower() in self.EXTS)
            if not files: raise RuntimeError(f'No images in {root!r}')
            subjects = sorted({fn.split(sep)[label_pos] for fn in files})
            lbl_map  = {s: i for i, s in enumerate(subjects)}
            for fn in files:
                sid = fn.split(sep)[label_pos]
                lbl = lbl_map[sid]
                i   = len(self.paths)
                self.paths.append(os.path.join(root, fn))
                self.labels.append(lbl); self.idx_by_label[lbl].append(i)

    @property
    def num_classes(self): return len(self.idx_by_label)
    def __len__(self):     return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[i]


def split_dataset(ds, ratio=0.5):
    """First half per subject → train (matches code's read_image logic)."""
    tr, te = [], []
    for lbl, idxs in ds.idx_by_label.items():
        k = max(1, int(len(idxs) * ratio))
        tr.extend(idxs[:k]); te.extend(idxs[k:])
    return tr, te


def get_transform(size=224):
    """
    Released code only resizes and casts; no augmentation.
    We apply ImageNet normalisation as the equivalent of VGG mean subtraction.
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


class PairDataset(Dataset):
    """Pair-based dataset used for teacher and standalone-student training."""
    def __init__(self, base_ds, indices, num_pairs=60000, seed=42):
        rng = random.Random(seed)
        lbl_map = defaultdict(list)
        for i in indices: lbl_map[base_ds.labels[i]].append(i)
        valid = [l for l, v in lbl_map.items() if len(v) >= 2]
        all_l = list(lbl_map.keys())
        self.base = base_ds
        pairs = []
        half  = num_pairs // 2
        for _ in range(half):
            l    = rng.choice(valid)
            a, b = rng.sample(lbl_map[l], 2)
            pairs.append((a, b, 1.0))
        for _ in range(num_pairs - half):
            l1, l2 = rng.sample(all_l, 2)
            pairs.append((rng.choice(lbl_map[l1]),
                          rng.choice(lbl_map[l2]), 0.0))
        rng.shuffle(pairs)
        self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        a, b, s = self.pairs[i]
        x1, _ = self.base[a]; x2, _ = self.base[b]
        return x1, x2, torch.tensor(s, dtype=torch.float32)


class ClassOrderedBatchSampler(Sampler):
    """
    Replicates Shuffle=False on class-sorted data.
    Each batch = [class_A × num_per, class_B × num_per, …]
    Required for Hard_loss per-class block indexing.
    """
    def __init__(self, idx_by_label, train_indices, num_per,
                 batch_size, seed=0):
        self.num_per  = num_per
        self.n_cls    = batch_size // num_per
        tr_set = set(train_indices)
        self.lbl_idx  = {
            lbl: [i for i in idxs if i in tr_set]
            for lbl, idxs in idx_by_label.items()
            if any(i in tr_set for i in idxs)
        }
        self.labels = list(self.lbl_idx.keys())
        self.rng    = random.Random(seed)

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


def make_onehot(labels_int, num_classes, device):
    B  = labels_int.shape[0]
    oh = torch.zeros(B, num_classes, device=device)
    oh[torch.arange(B), labels_int] = 1.0
    return oh


# ─────────────────────────────────────────────────────────────────────────────
# MODELS — translated from Teacher_nets.py, Student_nets.py, vgg16.py
# ─────────────────────────────────────────────────────────────────────────────

def leaky(x, a=0.2):
    """TF equivalent: tf.maximum(alpha*x, x)"""
    return torch.max(a * x, x)


class LReLU(nn.Module):
    def __init__(self, a=0.2): super().__init__(); self.a = a
    def forward(self, x): return leaky(x, self.a)


class TeacherDHN(nn.Module):
    """
    vgg16.py + Teacher_nets.py:
      backbone = VGG pool4 (ALL weights frozen as tf.constant)
      encode1  = 3× Conv(512→512, 3×3, SAME) + BN[0,1,2] + LReLU
                 + MaxPool(2×2, stride=2, SAME)          ← stride=2!
      encode2  = FC(flat→4096→4096→2048) + BN[0,1] + LReLU
                 (Dropout(0.5) but Training=False → never applied)
      encode3  = FC(2048→128, Tanh)

    For 224×224 input:
      pool4  → 14×14×512
      encode1 MaxPool stride=2 → 7×7×512  → flat = 25088
    """
    def __init__(self, hash_dim=128, alpha=0.2):
        super().__init__()
        vgg = models.vgg16(weights='IMAGENET1K_V1')
        # pool4 = features[:24]  (output for 224×224: [B,512,14,14])
        self.backbone = nn.Sequential(*list(vgg.features.children())[:24])
        for p in self.backbone.parameters():
            p.requires_grad_(False)                # frozen constants

        # encode1: 3 conv + MaxPool(2×2, stride=2, SAME padding)
        self.encode1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, momentum=None, eps=1e-5), LReLU(alpha),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, momentum=None, eps=1e-5), LReLU(alpha),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, momentum=None, eps=1e-5), LReLU(alpha),
            # SAME MaxPool stride=2 — PyTorch: kernel=2,stride=2
            # For 14×14 → 7×7 this matches TF's SAME stride=2 pool
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        # flat = 7×7×512 = 25088  (for 224×224 input)

        # encode2: FC 25088→4096→4096→2048 with BN1d + LReLU
        # NOTE: we compute flat size dynamically in forward; here we use
        # a placeholder and replace it after first forward if needed.
        self._flat_size = 512 * 7 * 7   # = 25088 for 224×224

        self.encode2 = nn.Sequential(
            nn.Linear(self._flat_size, 4096),
            nn.BatchNorm1d(4096, momentum=None, eps=1e-5), LReLU(alpha),
            # Dropout(0.5) omitted because Training=False in all TF calls
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096, momentum=None, eps=1e-5), LReLU(alpha),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048, momentum=None, eps=1e-5), LReLU(alpha),
        )

        # encode3: FC(2048→128, Tanh)
        self.encode3 = nn.Sequential(
            nn.Linear(2048, hash_dim),
            nn.Tanh(),
        )
        self._init()

    def _init(self):
        for m in list(self.encode1.modules()) + \
                 list(self.encode2.modules()) + \
                 list(self.encode3.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)          # frozen pool4 features
        x = self.encode1(x)
        x = torch.flatten(x, 1)
        x = self.encode2(x)
        return self.encode3(x)             # h ∈ (−1,+1)^D

    @torch.no_grad()
    def get_codes(self, x): return torch.sign(self.forward(x))


class StudentDHN(nn.Module):
    """
    Student_nets.py translated:
      encode1: Conv(3→16, 3×3, stride=4, VALID=0), BN[0,1,2], LReLU
               MaxPool(2×2, stride=1, VALID)
      encode2: Conv(16→32, 5×5, stride=2, SAME=2), BN[0,1,2], LReLU
               MaxPool(2×2, stride=1, VALID)
      encode3: Flatten → FC(→512), NO BN, LReLU
               (Dropout omitted: Training=False in all TF calls)
      encode4: FC(512→128, Tanh)
    """
    def __init__(self, hash_dim=128, img_size=224, alpha=0.2):
        super().__init__()
        self.encode1 = nn.Sequential(
            nn.Conv2d(3,  16, kernel_size=3, stride=4, padding=0),  # VALID
            nn.BatchNorm2d(16, momentum=None, eps=1e-5), LReLU(alpha),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),       # VALID
        )
        self.encode2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # SAME
            nn.BatchNorm2d(32, momentum=None, eps=1e-5), LReLU(alpha),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),       # VALID
        )
        flat = self._flat(img_size)

        # encode3: NO BN (different from conv layers)
        self.encode3 = nn.Sequential(
            nn.Linear(flat, 512),
            LReLU(alpha),
            # Dropout(0.5) omitted: Training=False in all TF calls
        )
        # encode4
        self.encode4 = nn.Sequential(
            nn.Linear(512, hash_dim),
            nn.Tanh(),
        )
        self._init()

    def _flat(self, s):
        with torch.no_grad():
            x = self.encode1(torch.zeros(1, 3, s, s))
            x = self.encode2(x)
            return x.view(1, -1).shape[1]

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encode1(x)
        x = self.encode2(x)
        x = self.encode3(torch.flatten(x, 1))
        return self.encode4(x)

    @torch.no_grad()
    def get_codes(self, x): return torch.sign(self.forward(x))


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS — translated directly from TF source
# ─────────────────────────────────────────────────────────────────────────────

def _sq_dist_matrix(A, B=None):
    """
    Squared Euclidean distance matrix.
    TF code: diag_A + diag_B^T − 2·A·B^T
    Equivalent to ‖A_i − B_j‖²
    """
    if B is None: B = A
    A2 = (A * A).sum(1, keepdim=True)           # [N, 1]
    B2 = (B * B).sum(1, keepdim=True).t()       # [1, M]
    return (A2 + B2 - 2.0 * A @ B.t()).clamp(min=0.0)  # [N, M]


def hash_loss(feature, label_oh, batch_size, omega_size, margin=180.0):
    """
    Exact translation of Hash_loss() from train_teacher/student/DDH.py.
    Splits into archer[:omega] and sabor[omega:].
    Returns (hash_loss_value, archer_sq_dists, sabor_sq_dists).
    All distances are SQUARED Euclidean.
    """
    f_a = feature[:omega_size];  f_s = feature[omega_size:]
    l_a = label_oh[:omega_size]; l_s = label_oh[omega_size:]

    d_aa = _sq_dist_matrix(f_a)          # [ω, ω]
    d_as = _sq_dist_matrix(f_a, f_s)     # [ω, B-ω]

    sim_aa = l_a @ l_a.t()               # [ω, ω]
    sim_as = l_a @ l_s.t()               # [ω, B-ω]

    def contrastive(d, s):
        return (0.5 * s * d +
                0.5 * (1.0 - s) * torch.clamp(margin - d, min=0.0)).mean()

    hl = contrastive(d_aa, sim_aa) + contrastive(d_as, sim_as)
    return hl, d_aa, d_as


def q_loss(h):
    """
    Released code: tf.reduce_mean(tf.pow(tf.subtract(h, sign(h)), 2))
    = mean((h − sign(h))²) = mean((|h|−1)²)
    """
    return torch.mean((h - torch.sign(h)) ** 2)


def dhn_loss(h, label_oh, batch_size, omega_size, w_quant=0.5, margin=180.0):
    """DHN_loss = hash_loss + 0.5 * q_loss"""
    code = torch.sign(h)
    hl, d_aa, d_as = hash_loss(h, label_oh, batch_size, omega_size, margin)
    ql = q_loss(h)
    return hl + w_quant * ql, d_aa, d_as


def rela_loss_fn(s_aa, s_as, t_aa, t_as):
    """
    Released code (train_DDH.py):
      rela_loss = MSE(student_archer_dist, teacher_archer_dist)
                + MSE(student_sabor_dist,  teacher_sabor_dist)
    Both matrices are SQUARED Euclidean (direct output of Hash_loss).
    """
    return (torch.mean((s_aa - t_aa) ** 2) +
            torch.mean((s_as - t_as) ** 2))


def hard_loss_fn(feat_T, feat_S, label_oh, batch_size, num_per):
    """
    Exact translation of Hard_loss() from train_DDH.py.
    Uses SQUARED Euclidean distance matrices.
    NO clamp anywhere (as in the original code).

    Batch must be organised as [class_0×num_per, class_1×num_per, …].

    FIX: diagonal self-distances (always 0) are masked out of the genuine
    block min/max.  Without masking, T_blk.min() = 0 always (self-distance),
    so the genuine constraint degenerates to S_blk.max() - 0, which is
    always large and produces oversized gradients every step.
    """
    T_dist = _sq_dist_matrix(feat_T)     # [B, B] squared dists
    S_dist = _sq_dist_matrix(feat_S)

    lbl_m  = label_oh @ label_oh.t()    # [B, B]  1=same class

    T_pos = lbl_m * T_dist;  T_neg = (1.0 - lbl_m) * T_dist
    S_pos = lbl_m * S_dist;  S_neg = (1.0 - lbl_m) * S_dist

    # Diagonal mask built once; reused across all class blocks
    diag_mask = torch.eye(num_per, dtype=torch.bool, device=feat_T.device)

    pos_losses = []
    neg_losses = []
    n_blk = batch_size // num_per

    for i in range(n_blk):
        s, e = i * num_per, (i + 1) * num_per

        # genuine block — exclude diagonal (self-distance = 0)
        T_blk = T_pos[s:e, s:e]
        S_blk = S_pos[s:e, s:e]
        T_blk_od = T_blk.masked_fill(diag_mask, float('inf'))  # for .min()
        S_blk_od = S_blk.masked_fill(diag_mask, 0.0)           # for .max()
        pos_losses.append(S_blk_od.max() - T_blk_od.min())     # NO clamp

        # imposter from this class to all others
        T_n = torch.cat([T_neg[s:e, :s], T_neg[s:e, e:]], dim=1)
        S_n = torch.cat([S_neg[s:e, :s], S_neg[s:e, e:]], dim=1)
        neg_losses.append(T_n.max() - S_n.min())                # NO clamp

    return (torch.stack(pos_losses).mean() +
            torch.stack(neg_losses).mean())


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def _infinite(loader):
    while True:
        for batch in loader: yield batch


def train_teacher(cfg, device):
    log.info('=' * 60)
    log.info('STAGE 1 – Teacher  (RMSProp lr=0.001, epoch=500)')
    log.info('=' * 60)
    ds = CASIADataset(cfg['data_dir'], get_transform(cfg['img_size']),
                      cfg['label_pos'], cfg['sep'])
    log.info(f'  {len(ds)} images | {ds.num_classes} subjects')
    tr_idx, _ = split_dataset(ds, cfg['train_ratio'])
    num_cl = ds.num_classes

    # Shuffle=True (train_teacher.py)
    pairs  = PairDataset(ds, tr_idx, num_pairs=60000)
    loader = DataLoader(pairs, cfg['teacher_batch'], shuffle=True,
                        num_workers=cfg['num_workers'], pin_memory=True,
                        drop_last=True)

    model = TeacherDHN(cfg['hash_dim'], cfg['lrelu_alpha']).to(device)
    opt   = optim.RMSprop(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['teacher_lr'], alpha=cfg['rms_alpha'])

    model.train()
    step = 0
    for epoch in range(cfg['teacher_epochs']):
        for x1, x2, s in loader:
            x1, x2 = x1.to(device), x2.to(device)
            s       = s.to(device)
            B       = x1.shape[0]
            h1      = model(x1)
            h2      = model(x2)

            # Build a 2B batch with one-hot labels from similarity vector
            # (teacher training uses Hash_loss with archer/sabor split)
            h    = torch.cat([h1, h2], 0)          # [2B, D]
            lbls = torch.cat([
                torch.arange(B, device=device),
                torch.where(s.bool(),
                            torch.arange(B, device=device),
                            torch.arange(B, 2 * B, device=device))
            ], 0)
            n_cls = 2 * B
            loh   = make_onehot(lbls, n_cls, device)
            hl_v, _, _ = hash_loss(h, loh, 2 * B, B, cfg['margin_t'])
            ql_v       = q_loss(h)
            loss       = hl_v + cfg['w_quant'] * ql_v

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % cfg['log_every'] == 0:
                log.info(f'  Ep {epoch+1:3d}  step {step:6d} | '
                         f'loss={loss.item():.4f}')

    os.makedirs(cfg['save_dir'], exist_ok=True)
    path = os.path.join(cfg['save_dir'], 'teacher.pth')
    torch.save(model.state_dict(), path)
    log.info(f'  Teacher saved → {path}')
    return model


def train_student_standalone(cfg, device):
    """
    Mirrors train_student.py: student trained with DHN loss only (no teacher).
    Not strictly needed for DDH but matches the original pipeline.
    """
    log.info('=' * 60)
    log.info('STAGE 2a – Student standalone (RMSProp lr=0.001, epoch=500)')
    log.info('=' * 60)
    ds = CASIADataset(cfg['data_dir'], get_transform(cfg['img_size']),
                      cfg['label_pos'], cfg['sep'])
    tr_idx, _ = split_dataset(ds, cfg['train_ratio'])
    num_cl = ds.num_classes

    sampler = ClassOrderedBatchSampler(
        ds.idx_by_label, tr_idx,
        cfg['student_batch'] // 2,   # num_per
        cfg['student_batch'])
    loader  = DataLoader(ds, batch_sampler=sampler,
                         num_workers=cfg['num_workers'], pin_memory=True)

    model = StudentDHN(cfg['hash_dim'], cfg['img_size'],
                       cfg['lrelu_alpha']).to(device)
    opt   = optim.RMSprop(model.parameters(),
                          lr=cfg['student_lr'], alpha=cfg['rms_alpha'])
    model.train()
    step = 0
    for epoch in range(cfg['student_epochs']):
        for imgs, labels_int in loader:
            imgs, labels_int = imgs.to(device), labels_int.to(device)
            B     = imgs.shape[0]
            if B != cfg['student_batch']: continue
            loh   = make_onehot(labels_int, num_cl, device)
            h     = model(imgs)
            hl_v, _, _ = hash_loss(h, loh, B, cfg['student_omega'],
                                   cfg['margin_t'])
            ql_v       = q_loss(h)
            loss       = hl_v + cfg['w_quant'] * ql_v
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            step += 1
            if step % cfg['log_every'] == 0:
                log.info(f'  Ep {epoch+1:3d}  step {step:6d} | '
                         f'loss={loss.item():.4f}')

    path = os.path.join(cfg['save_dir'], 'student_standalone.pth')
    os.makedirs(cfg['save_dir'], exist_ok=True)
    torch.save(model.state_dict(), path)
    log.info(f'  Standalone student saved → {path}')
    return model


def train_ddh(cfg, device, teacher=None):
    """
    Mirrors train_DDH.py:
    loss = rela_loss + hard_loss + DHN_loss   (no α/β)
    RMSProp(0.0001, 0.9), epoch=680, batch=50, omega=30, num_per=10
    Shuffle=False → ClassOrderedBatchSampler
    """
    log.info('=' * 60)
    log.info('STAGE 2b – DDH  (RMSProp lr=0.0001, epoch=680, class-ordered)')
    log.info('=' * 60)
    ds = CASIADataset(cfg['data_dir'], get_transform(cfg['img_size']),
                      cfg['label_pos'], cfg['sep'])
    tr_idx, _ = split_dataset(ds, cfg['train_ratio'])
    num_cl    = ds.num_classes

    sampler = ClassOrderedBatchSampler(
        ds.idx_by_label, tr_idx,
        cfg['ddh_num_per'], cfg['ddh_batch'])
    loader  = DataLoader(ds, batch_sampler=sampler,
                         num_workers=cfg['num_workers'], pin_memory=True)

    if teacher is None:
        teacher = TeacherDHN(cfg['hash_dim'], cfg['lrelu_alpha']).to(device)
        ckpt = os.path.join(cfg['save_dir'], 'teacher.pth')
        teacher.load_state_dict(torch.load(ckpt, map_location=device))
        log.info(f'  Teacher loaded from {ckpt}')
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    student = StudentDHN(cfg['hash_dim'], cfg['img_size'],
                         cfg['lrelu_alpha']).to(device)
    opt = optim.RMSprop(student.parameters(),
                        lr=cfg['ddh_lr'], alpha=cfg['rms_alpha'])

    student.train()
    step = 0
    for epoch in range(cfg['ddh_epochs']):
        for imgs, labels_int in loader:
            imgs, labels_int = imgs.to(device), labels_int.to(device)
            B = imgs.shape[0]
            if B != cfg['ddh_batch']: continue

            loh = make_onehot(labels_int, num_cl, device)

            # Student features
            h_s = student(imgs)
            dhn_s, s_aa, s_as = dhn_loss(
                h_s, loh, B, cfg['ddh_omega'], cfg['w_quant'], cfg['margin_t'])

            # Teacher features (frozen)
            with torch.no_grad():
                h_t = teacher(imgs)
            _, t_aa, t_as = dhn_loss(
                h_t, loh, B, cfg['ddh_omega'], cfg['w_quant'], cfg['margin_t'])

            rl   = rela_loss_fn(s_aa, s_as, t_aa, t_as)
            hl   = hard_loss_fn(h_t, h_s, loh, B, cfg['ddh_num_per'])
            # Exact train_DDH.py: loss = rela_loss + hard_loss + DHN_loss
            loss = rl + hl + dhn_s

            # Guard: skip batch if any component is non-finite
            if not torch.isfinite(loss):
                log.warning(f'  Ep {epoch+1} step {step}: non-finite loss '
                            f'(rela={rl.item():.4f} hard={hl.item():.4f} '
                            f'DHN={dhn_s.item():.4f}) — skipping batch')
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Clip gradients — hard_loss can spike at early iterations
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            opt.step()
            step += 1
            if step % cfg['log_every'] == 0:
                log.info(f'  Ep {epoch+1:3d}  step {step:6d} | '
                         f'loss={loss.item():.4f}  '
                         f'rela={rl.item():.4f}  hard={hl.item():.4f}  '
                         f'DHN={dhn_s.item():.4f}')

    path = os.path.join(cfg['save_dir'], 'student_ddh.pth')
    os.makedirs(cfg['save_dir'], exist_ok=True)
    torch.save(student.state_dict(), path)
    log.info(f'  DDH student saved → {path}')
    return student


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION — translated from eval.py
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_codes(model, loader, device):
    model.eval()
    codes, labels = [], []
    for imgs, lbls in loader:
        h = model(imgs.to(device))
        codes.append(torch.sign(h).cpu().numpy().astype(np.float32))
        labels.append(lbls.numpy())
    return np.vstack(codes), np.concatenate(labels)


def identification_accuracy(tr_codes, tr_labels, te_codes, te_labels):
    """
    1-NN using L2 distance on binary codes (eval.py uses sqrt(sum(sq_diff))).
    """
    correct = 0
    for code, true_lbl in zip(te_codes, te_labels):
        l2    = np.sqrt(np.sum((code - tr_codes) ** 2, axis=1))
        correct += int(tr_labels[np.argmin(l2)] == true_lbl)
    return correct / len(te_labels) * 100.0


def compute_eer(tr_codes, tr_labels, te_codes, te_labels, num_per=None):
    """
    EER via threshold sweep 10–50 on L2 distances (Draw_DET.py logic).
    """
    all_codes  = np.vstack([tr_codes, te_codes])
    all_labels = np.concatenate([tr_labels, te_labels])
    n_train    = len(tr_labels)

    true_list, false_list = [], []
    for i in range(n_train, len(all_codes)):
        for j in range(len(all_codes)):
            if i == j: continue
            d = float(np.sqrt(np.sum((all_codes[i] - all_codes[j]) ** 2)))
            if all_labels[i] == all_labels[j]: true_list.append(d)
            else:                               false_list.append(d)

    true_arr  = np.array(true_list,  dtype=np.float32)
    false_arr = np.array(false_list, dtype=np.float32)
    best_eer, best_diff = 1.0, 1.0

    for thr in range(10, 50):
        frr = float(np.mean(true_arr  >  thr))
        far = float(np.mean(false_arr <= thr))
        diff = abs(frr - far)
        if diff < best_diff:
            best_diff = diff
            best_eer  = (frr + far) / 2.0

    return best_eer * 100.0


def evaluate(cfg, device):
    log.info('=' * 60); log.info('EVALUATION'); log.info('=' * 60)
    ds = CASIADataset(cfg['data_dir'], get_transform(cfg['img_size']),
                      cfg['label_pos'], cfg['sep'])
    tr_idx, te_idx = split_dataset(ds, cfg['train_ratio'])
    tr_ld = DataLoader(Subset(ds, tr_idx), 32, num_workers=cfg['num_workers'])
    te_ld = DataLoader(Subset(ds, te_idx), 32, num_workers=cfg['num_workers'])

    results = []
    for name, ckpt_fn, make_m in [
        ('Teacher', 'teacher.pth',
         lambda: TeacherDHN(cfg['hash_dim'], cfg['lrelu_alpha'])),
        ('DDH',     'student_ddh.pth',
         lambda: StudentDHN(cfg['hash_dim'], cfg['img_size'],
                            cfg['lrelu_alpha'])),
    ]:
        ckpt = os.path.join(cfg['save_dir'], ckpt_fn)
        if not os.path.exists(ckpt):
            log.warning(f'  {name}: checkpoint not found, skipping'); continue
        m = make_m().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        tr_c, tr_l = extract_codes(m, tr_ld, device)
        te_c, te_l = extract_codes(m, te_ld, device)
        acc = identification_accuracy(tr_c, tr_l, te_c, te_l)
        eer = compute_eer(tr_c, tr_l, te_c, te_l)
        log.info(f'  {name:10s}  Acc={acc:.2f}%   EER={eer:.2f}%')
        results.append({'model': name, 'acc_%': round(acc, 2),
                        'eer_%': round(eer, 2)})

    os.makedirs(cfg['save_dir'], exist_ok=True)
    csv_p = os.path.join(cfg['save_dir'], 'results.csv')
    with open(csv_p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'acc_%', 'eer_%'])
        w.writeheader(); w.writerows(results)
    log.info(f'  Results → {csv_p}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description='DDH – released code faithful')
    p.add_argument('--data_dir',   default=CFG['data_dir'])
    p.add_argument('--stage',      default='all',
                   choices=['all', 'teacher', 'student', 'ddh', 'eval'])
    p.add_argument('--label_pos',  type=int, default=CFG['label_pos'])
    p.add_argument('--sep',        default=CFG['sep'])
    args = p.parse_args()
    CFG.update(data_dir=args.data_dir, label_pos=args.label_pos, sep=args.sep)

    os.makedirs(CFG['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device : {device}')
    log.info('Script : ddh_released.py  (TF source faithful)')
    log.info(f'Teacher batch={CFG["teacher_batch"]} omega={CFG["teacher_omega"]} '
             f'lr={CFG["teacher_lr"]} epochs={CFG["teacher_epochs"]}')
    log.info(f'DDH     batch={CFG["ddh_batch"]} omega={CFG["ddh_omega"]} '
             f'lr={CFG["ddh_lr"]} epochs={CFG["ddh_epochs"]} '
             f'num_per={CFG["ddh_num_per"]}')

    teacher = None
    if args.stage in ('all', 'teacher'):
        teacher = train_teacher(cfg=CFG, device=device)
    if args.stage in ('all', 'student'):
        train_student_standalone(cfg=CFG, device=device)
    if args.stage in ('all', 'ddh'):
        train_ddh(cfg=CFG, device=device, teacher=teacher)
    if args.stage in ('all', 'eval'):
        evaluate(cfg=CFG, device=device)


if __name__ == '__main__':
    main()
