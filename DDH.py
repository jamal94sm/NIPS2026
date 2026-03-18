"""
ddh_best.py  — Best synthesis of paper + released code
=======================================================
Every decision below is annotated with WHY it was chosen.

ARCHITECTURE
  ✓ VGG pool4 frozen  ← released code (ground truth of what produced results).
    Paper Fig.5 is misleading; code shows ALL VGG weights are tf.constant.
  ✓ Custom conv5 block + FC head trainable  ← released code.
  ✓ BN + LeakyReLU(0.2) everywhere  ← released code; critical for stability.
  ✓ Student Conv2 SAME padding  ← released code (paper says padding=0 which
    conflicts with Fig.6 output sizes).
  ✓ Input 128×128  ← paper (all databases use 128×128 ROI, Section V-A).
    We adapt the VGG for 128×128 dynamically; this is MORE faithful to the
    paper's dataset setting even though the TF code uses 224×224.

DISTANCES
  ✓ D in L_h = squared Euclidean  ← both paper (t=180 only makes sense
    for squared) and released code agree.
  ✓ d_ij for rela/hard = L2 norm  ← paper Eq.7/8 explicitly write ‖·‖_2.
    Released code uses squared; we prefer paper's explicit formula here.

QUANTIZATION LOSS
  ✓ L_q = mean((|h|−1)²), weight=0.5  ← released code (numerically stable;
    equivalent to paper Eq.2 but in MSE form with empirically tuned weight).

L_rela
  ✓ MSE of L2 distances (not squared)  ← paper Eq.6 structure (‖d^T−d^S‖)
    using MSE instead of vector-norm for better gradient flow.

L_hard
  ✓ Per-class block structure  ← released code (required for meaningful
    max/min computation over genuine/imposter sets).
  ✓ hinge clamp(·, min=0)  ← mathematically correct. The loss should be
    zero when the constraint is already satisfied. Released code's omission
    of the clamp causes negative losses that reward already-good solutions
    and can destabilise training. Paper Eq.9 implicitly assumes constraints
    are violated (otherwise there is no loss to minimise).

OPTIMISER
  ✓ Adam lr=0.001  ← paper Section V-B explicitly states "Adam Optimizer".
  ✓ α=1, β=0.8    ← paper Tables XI/XII, best across all sub-datasets.
  ✓ 10,000 iterations  ← paper Table XV, same for all models.

EVALUATION
  ✓ Hamming distance on binary codes  ← paper Section V-C/D
    ("XOR operation ... Hamming distance"). Released code uses L2 as a
    proxy but Hamming is the stated metric and correct for ±1 binary codes.

Usage:
  python ddh_best.py --data_dir /path/to/casia_ms --stage all
  python ddh_best.py --data_dir /path --label_pos 0 --sep _ --num_per 5
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
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    data_dir        = "/home/pai-ng/Jamal/CASIA-MS-ROI",
    label_pos       = 0,
    sep             = '_',
    train_ratio     = 0.5,

    hash_dim        = 128,
    img_size        = 128,           # paper ROI size; VGG adapted dynamically
    lrelu_alpha     = 0.2,

    # DHN loss
    margin_t        = 180.0,
    w_quant         = 0.5,           # released code: more stable than 1.0

    # Distillation (paper Tables XI/XII best)
    alpha           = 1.0,
    beta            = 0.8,

    # Batch: num_per × n_cls_per_batch = total batch size
    # CASIA-MS typically has ~5 train images per subject → num_per=5
    num_per         = 5,
    n_cls_per_batch = 10,            # → batch_size = 50

    # Training (paper Section V-B, Table XV)
    teacher_iters   = 10000,
    student_iters   = 10000,
    lr              = 1e-3,          # Adam

    num_workers     = 4,
    log_every       = 500,
    save_dir        = './ckpt_best',
)
CFG['batch_size'] = CFG['num_per'] * CFG['n_cls_per_batch']

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
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
    tr, te = [], []
    for lbl, idxs in ds.idx_by_label.items():
        k = max(1, int(len(idxs) * ratio))
        tr.extend(idxs[:k]); te.extend(idxs[k:])
    return tr, te


def get_transform(train=True, size=128):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ops = [transforms.Resize((size, size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(0.3),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1)]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


class ClassOrderedBatchSampler(Sampler):
    """
    Each batch = [class_A × num_per, class_B × num_per, …].
    Required for Hard_loss per-class block indexing.
    """
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
        self.labels  = list(self.lbl_idx.keys())

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


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────
class LReLU(nn.Module):
    def __init__(self, a=0.2): super().__init__(); self.a = a
    def forward(self, x): return torch.max(self.a * x, x)


class TeacherDHN(nn.Module):
    """
    VGG pool4 frozen (released code ground truth) + trainable conv5 + FC.
    BN + LReLU (released code, training stability).
    Input adapted to any size: flat dimension computed dynamically.
    """
    def __init__(self, hash_dim=128, img_size=128, alpha=0.2):
        super().__init__()
        vgg = models.vgg16(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(vgg.features.children())[:24])
        for p in self.backbone.parameters(): p.requires_grad_(False)

        # conv5: 3 convs, BN, LReLU + MaxPool stride=2 (released code)
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
            nn.Linear(flat, 4096),
            nn.BatchNorm1d(4096, eps=1e-5), LReLU(alpha),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096, eps=1e-5), LReLU(alpha),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048, eps=1e-5), LReLU(alpha),
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
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu',
                                        a=0.2)
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
    """
    Paper Fig.6 layout + released code improvements:
      BN + LReLU (stability), Conv2 SAME padding (released code).
    """
    def __init__(self, hash_dim=128, img_size=128, alpha=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,  16, 3, stride=4, padding=0),     # VALID
            nn.BatchNorm2d(16, eps=1e-5), LReLU(alpha),
            nn.MaxPool2d(2, stride=1),

            nn.Conv2d(16, 32, 5, stride=2, padding=2),     # SAME
            nn.BatchNorm2d(32, eps=1e-5), LReLU(alpha),
            nn.MaxPool2d(2, stride=1),
        )
        flat = self._flat(img_size)
        self.fc = nn.Sequential(
            nn.Linear(flat, 512),  nn.BatchNorm1d(512, eps=1e-5), LReLU(alpha),
            nn.Linear(512, hash_dim), nn.Tanh(),
        )
        self._init()

    def _flat(self, s):
        with torch.no_grad():
            return self.conv(torch.zeros(1, 3, s, s)).view(1, -1).shape[1]

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu',
                                        a=0.2)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(torch.flatten(self.conv(x), 1))

    @torch.no_grad()
    def get_codes(self, x): return torch.sign(self.forward(x))


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def _sq_dist(A, B=None):
    """‖A_i − B_j‖²  [N, M]  via dot-product expansion."""
    if B is None: B = A
    return (((A * A).sum(1, keepdim=True) +
             (B * B).sum(1, keepdim=True).t()
             - 2.0 * A @ B.t())).clamp(min=0.0)


def _l2_dist(A, B=None):
    """‖A_i − B_j‖_2  [N, M]  (L2 norm)."""
    return _sq_dist(A, B).sqrt()


def loss_dhn(h1, h2, s, t=180.0, w=0.5):
    """
    D = squared Euclidean (consistent with t=180 range, both paper+code agree).
    L_q = mean((|h|−1)²), weight w=0.5 (released code).
    """
    D  = torch.sum((h1 - h2) ** 2, dim=1)
    Lh = (0.5 * s * D +
          0.5 * (1.0 - s) * torch.clamp(t - D, min=0.0)).mean()
    Lq = ((torch.abs(h1) - 1.0).pow(2).mean() +
          (torch.abs(h2) - 1.0).pow(2).mean()) / 2.0
    return Lh + w * Lq, Lh, Lq


def loss_dhn_batch(h, label_oh, batch_size, omega, t=180.0, w=0.5):
    """
    Hash_loss on a class-ordered batch using archer/sabor split.
    Returns (total, archer_L2_dists, cross_L2_dists).
    Uses L2 distances (paper Eq.7/8) for rela/hard.
    """
    f_a, f_s   = h[:omega],         h[omega:]
    la, ls     = label_oh[:omega],  label_oh[omega:]

    d_aa_sq = _sq_dist(f_a)          # [ω, ω]
    d_as_sq = _sq_dist(f_a, f_s)     # [ω, B-ω]
    d_aa_l2 = d_aa_sq.sqrt()         # for rela/hard
    d_as_l2 = d_as_sq.sqrt()

    sim_aa  = la @ la.t()
    sim_as  = la @ ls.t()

    def cont(d_sq, sim):
        return (0.5 * sim * d_sq +
                0.5 * (1.0 - sim) * torch.clamp(t - d_sq, min=0.0)).mean()

    hl = cont(d_aa_sq, sim_aa) + cont(d_as_sq, sim_as)
    ql = torch.mean((torch.abs(h) - 1.0) ** 2)
    return hl + w * ql, d_aa_l2, d_as_l2


def loss_rela(s_aa, s_as, t_aa, t_as):
    """
    MSE of L2 distances (paper Eq.7/8 for d_ij, MSE form for stability).
    Rela = MSE(d_aa_S, d_aa_T) + MSE(d_as_S, d_as_T)
    """
    return (torch.mean((s_aa - t_aa) ** 2) +
            torch.mean((s_as - t_as) ** 2))


def loss_hard(feat_T, feat_S, label_oh, batch_size, num_per):
    """
    Per-class blocks (released code structure) with L2 distances (paper Eq.7/8).
    Hinge clamp at 0 (mathematically correct):
      genuine:  clamp(max_S_genuine − min_T_genuine, min=0)
      imposter: clamp(max_T_imposter − min_S_imposter, min=0)
    """
    T_d = _l2_dist(feat_T)            # [B, B] L2 distances
    S_d = _l2_dist(feat_S)

    lbl_m = label_oh @ label_oh.t()  # [B, B]  1=same class

    T_pos = lbl_m * T_d;  T_neg = (1.0 - lbl_m) * T_d
    S_pos = lbl_m * S_d;  S_neg = (1.0 - lbl_m) * S_d

    pos_l, neg_l = [], []
    n_blk = batch_size // num_per

    for i in range(n_blk):
        s, e = i * num_per, (i + 1) * num_per

        # genuine block: want max_S_genuine < min_T_genuine
        T_blk = T_pos[s:e, s:e]; S_blk = S_pos[s:e, s:e]
        pos_l.append(torch.clamp(S_blk.max() - T_blk.min(), min=0.0))

        # imposter: want min_S_imposter > max_T_imposter
        T_n = torch.cat([T_neg[s:e, :s], T_neg[s:e, e:]], dim=1)
        S_n = torch.cat([S_neg[s:e, :s], S_neg[s:e, e:]], dim=1)
        neg_l.append(torch.clamp(T_n.max() - S_n.min(), min=0.0))

    return (torch.stack(pos_l).mean() + torch.stack(neg_l).mean())


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def _infinite(loader):
    while True:
        for batch in loader: yield batch


def train_teacher(cfg, device):
    log.info('=' * 60)
    log.info('STAGE 1 – Teacher  (Adam lr=0.001, 10 000 iters)')
    log.info('=' * 60)

    ds = CASIADataset(cfg['data_dir'],
                      get_transform(True,  cfg['img_size']),
                      cfg['label_pos'], cfg['sep'])
    log.info(f'  {len(ds)} images | {ds.num_classes} subjects')
    tr_idx, _ = split_dataset(ds, cfg['train_ratio'])
    num_cl    = ds.num_classes
    B         = cfg['batch_size']

    sampler = ClassOrderedBatchSampler(
        ds.idx_by_label, tr_idx, cfg['num_per'], cfg['n_cls_per_batch'])
    loader  = DataLoader(ds, batch_sampler=sampler,
                         num_workers=cfg['num_workers'], pin_memory=True)

    model = TeacherDHN(cfg['hash_dim'], cfg['img_size'],
                       cfg['lrelu_alpha']).to(device)
    opt   = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'])

    model.train()
    it   = _infinite(loader)
    best = float('inf')

    for step in range(1, cfg['teacher_iters'] + 1):
        imgs, labels_int = next(it)
        imgs, labels_int = imgs.to(device), labels_int.to(device)
        if imgs.shape[0] != B: continue

        loh  = make_onehot(labels_int, num_cl, device)
        h    = model(imgs)

        # All pairwise pairs (upper triangle) for contrastive loss
        h1 = h.unsqueeze(1).expand(-1, B, -1).reshape(B * B, -1)
        h2 = h.unsqueeze(0).expand(B, -1, -1).reshape(B * B, -1)
        s  = (loh @ loh.t()).reshape(-1)
        ut = torch.triu(torch.ones(B, B, device=device), diagonal=1).bool()
        ut = ut.reshape(-1)
        h1p, h2p, sp = h1[ut], h2[ut], s[ut]

        loss, Lh, Lq = loss_dhn(h1p, h2p, sp, cfg['margin_t'], cfg['w_quant'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg['log_every'] == 0 or step == 1:
            log.info(f'  iter {step:5d} | loss={loss.item():.4f} '
                     f'Lh={Lh.item():.4f} Lq={Lq.item():.4f}')
        if loss.item() < best:
            best = loss.item()
            os.makedirs(cfg['save_dir'], exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(cfg['save_dir'], 'teacher.pth'))

    log.info(f'  Teacher saved (best loss={best:.4f})')
    return model


def train_student(cfg, device, teacher=None):
    log.info('=' * 60)
    log.info('STAGE 2 – Student DDH  (Adam, α=1, β=0.8, 10 000 iters)')
    log.info('=' * 60)

    ds = CASIADataset(cfg['data_dir'],
                      get_transform(True,  cfg['img_size']),
                      cfg['label_pos'], cfg['sep'])
    tr_idx, _ = split_dataset(ds, cfg['train_ratio'])
    num_cl    = ds.num_classes
    B         = cfg['batch_size']
    omega     = B // 2                 # archer half of each batch

    sampler = ClassOrderedBatchSampler(
        ds.idx_by_label, tr_idx, cfg['num_per'], cfg['n_cls_per_batch'])
    loader  = DataLoader(ds, batch_sampler=sampler,
                         num_workers=cfg['num_workers'], pin_memory=True)

    if teacher is None:
        teacher = TeacherDHN(cfg['hash_dim'], cfg['img_size'],
                             cfg['lrelu_alpha']).to(device)
        ckpt = os.path.join(cfg['save_dir'], 'teacher.pth')
        teacher.load_state_dict(torch.load(ckpt, map_location=device))
        log.info(f'  Teacher loaded from {ckpt}')
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    student = StudentDHN(cfg['hash_dim'], cfg['img_size'],
                         cfg['lrelu_alpha']).to(device)
    opt = optim.Adam(student.parameters(), lr=cfg['lr'])

    student.train()
    it   = _infinite(loader)
    best = float('inf')

    for step in range(1, cfg['student_iters'] + 1):
        imgs, labels_int = next(it)
        imgs, labels_int = imgs.to(device), labels_int.to(device)
        if imgs.shape[0] != B: continue

        loh  = make_onehot(labels_int, num_cl, device)

        # Teacher (frozen)
        with torch.no_grad():
            h_T = teacher(imgs)
        _, t_aa, t_as = loss_dhn_batch(
            h_T, loh, B, omega, cfg['margin_t'], cfg['w_quant'])

        # Student
        h_S = student(imgs)
        dhn_s, s_aa, s_as = loss_dhn_batch(
            h_S, loh, B, omega, cfg['margin_t'], cfg['w_quant'])

        Lrela = loss_rela(s_aa, s_as, t_aa, t_as)
        Lhard = loss_hard(h_T, h_S, loh, B, cfg['num_per'])

        # Eq.10: L = L_DHN + α·L_rela + β·L_hard
        loss  = dhn_s + cfg['alpha'] * Lrela + cfg['beta'] * Lhard

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg['log_every'] == 0 or step == 1:
            log.info(f'  iter {step:5d} | loss={loss.item():.4f} '
                     f'DHN={dhn_s.item():.4f} '
                     f'Lrela={Lrela.item():.4f} '
                     f'Lhard={Lhard.item():.4f}')
        if loss.item() < best:
            best = loss.item()
            os.makedirs(cfg['save_dir'], exist_ok=True)
            torch.save(student.state_dict(),
                       os.path.join(cfg['save_dir'], 'student.pth'))

    log.info(f'  Student saved (best loss={best:.4f})')
    return student


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_codes(model, loader, device):
    model.eval()
    codes, labels = [], []
    for imgs, lbls in loader:
        h = model(imgs.to(device))
        codes.append(torch.sign(h).cpu().numpy().astype(np.int8))
        labels.append(lbls.numpy())
    return np.vstack(codes), np.concatenate(labels)


def identification_accuracy(tr_codes, tr_labels, te_codes, te_labels):
    """1-NN by Hamming distance on ±1 binary codes."""
    correct = 0
    for code, true_lbl in zip(te_codes, te_labels):
        ham     = np.sum(code != tr_codes, axis=1)
        correct += int(tr_labels[np.argmin(ham)] == true_lbl)
    return correct / len(te_labels) * 100.0


def compute_eer(tr_codes, tr_labels, te_codes, te_labels):
    """EER via sklearn roc_curve with Hamming similarity."""
    scores, truth = [], []
    for code, lbl in zip(te_codes, te_labels):
        sim = -np.sum(code != tr_codes, axis=1).astype(np.float32)
        scores.extend(sim.tolist())
        truth.extend((tr_labels == lbl).tolist())
    fpr, tpr, _ = roc_curve(truth, scores)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0 * 100.0


def evaluate(cfg, device):
    log.info('=' * 60); log.info('EVALUATION'); log.info('=' * 60)
    ds = CASIADataset(cfg['data_dir'],
                      get_transform(False, cfg['img_size']),
                      cfg['label_pos'], cfg['sep'])
    tr_idx, te_idx = split_dataset(ds, cfg['train_ratio'])
    tr_ld = DataLoader(Subset(ds, tr_idx), 64, num_workers=cfg['num_workers'])
    te_ld = DataLoader(Subset(ds, te_idx), 64, num_workers=cfg['num_workers'])

    log.info(f'  {"Model":12s} {"Acc (%)":>10} {"EER (%)":>10}')
    log.info('  ' + '-' * 34)
    results = []
    for name, ckpt_fn, make_m in [
        ('Teacher', 'teacher.pth',
         lambda: TeacherDHN(cfg['hash_dim'], cfg['img_size'],
                            cfg['lrelu_alpha'])),
        ('DDH',     'student.pth',
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
        log.info(f'  {name:12s} {acc:>10.2f} {eer:>10.2f}')
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
    p = argparse.ArgumentParser(description='DDH – best synthesis')
    p.add_argument('--data_dir',   default=CFG['data_dir'])
    p.add_argument('--stage',      default='all',
                   choices=['all', 'teacher', 'student', 'eval'])
    p.add_argument('--label_pos',  type=int, default=CFG['label_pos'])
    p.add_argument('--sep',        default=CFG['sep'])
    p.add_argument('--num_per',    type=int, default=CFG['num_per'],
                   help='train images per class per batch (≤ your per-subject '
                        'train count). For CASIA-MS ≈5 images/subject → 5.')
    p.add_argument('--n_cls',      type=int, default=CFG['n_cls_per_batch'],
                   help='classes per batch  (batch_size = num_per × n_cls)')
    args = p.parse_args()
    CFG.update(data_dir=args.data_dir, label_pos=args.label_pos,
               sep=args.sep, num_per=args.num_per,
               n_cls_per_batch=args.n_cls)
    CFG['batch_size'] = CFG['num_per'] * CFG['n_cls_per_batch']

    os.makedirs(CFG['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device     : {device}')
    log.info(f'Script     : ddh_best.py  (paper + released code synthesis)')
    log.info(f'Batch      : {CFG["batch_size"]}  '
             f'({CFG["num_per"]} per class × {CFG["n_cls_per_batch"]} classes)')
    log.info(f'α={CFG["alpha"]}  β={CFG["beta"]}  '
             f't={CFG["margin_t"]}  lr={CFG["lr"]}  '
             f'iters={CFG["teacher_iters"]}')

    teacher = None
    if args.stage in ('all', 'teacher'):
        teacher = train_teacher(cfg=CFG, device=device)
    if args.stage in ('all', 'student'):
        train_student(cfg=CFG, device=device, teacher=teacher)
    if args.stage in ('all', 'eval'):
        evaluate(cfg=CFG, device=device)


if __name__ == '__main__':
    main()
