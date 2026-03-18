"""
ddh_paper.py  — Implementation faithful to the PAPER text only
================================================================
Paper: "Towards Efficient Unconstrained Palmprint Recognition via
        Deep Distillation Hashing", Shao et al.

Every equation implemented exactly as written, every architecture
decision taken from the paper text and figures only.

KEY PAPER CHOICES (vs. released code):
  • Teacher: full VGG-16 up to pool5. batch1–4 fine-tuned from ImageNet,
    batch5 + FC head trained from scratch. (paper Fig.5 and Section IV-A)
  • Student: Conv1(3×3,16ch,stride=4,VALID)→ReLU→MaxPool(2×2,stride=1)
             Conv2(5×5,32ch,stride=2,VALID)→ReLU→MaxPool(2×2,stride=1)
             FC(512,ReLU)→FC(128,Tanh)→sign()
    Standard ReLU, NO batch-norm. (paper Fig.6)
  • L_q = ‖|h|-1‖_2  per sample, vector L2 norm.  (Eq.2)
  • d_ij = ‖b_i - b_j‖_2  L2 norm.  (Eq.7/8)
  • L_rela = ‖d^T - d^S‖_2  (Eq.6)
  • L_hard: NO clamp — raw differences.  (Eq.9 as written)
  • L = L_DHN + α·L_rela + β·L_hard,  α=1, β=0.8 (Tables XI/XII best)
  • Adam lr=0.001  (Section V-B)
  • 10,000 iterations  (Table XV)
  • Input 128×128  (all databases use 128×128 ROI, Section V-A)

Usage:
  python ddh_paper.py --data_dir /path/to/casia_ms --stage all
  python ddh_paper.py --data_dir /path/to/casia_ms --stage teacher
  python ddh_paper.py --data_dir /path/to/casia_ms --stage student
  python ddh_paper.py --data_dir /path/to/casia_ms --stage eval
  # For CASIA-MS flat folder with filenames like 001_2_L_03.jpg:
  python ddh_paper.py --data_dir /path --label_pos 0 --sep _
"""

import os, random, logging, argparse, csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
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
    train_ratio     = 0.5,           # first 50 % per subject → train

    hash_dim        = 128,
    img_size        = 128,           # paper: 128×128 ROI (Section V-A)

    # DHN loss (Eq.1-3)
    margin_t        = 180.0,         # t in Eq.1 (paper: "set to 180, like [7]")
    w_quant         = 1.0,           # w in Eq.3 (paper does not fix value;
                                     # 1.0 = equal weight as a plain reading implies)

    # Distillation weights (Eq.10, Tables XI/XII best)
    alpha           = 1.0,
    beta            = 0.8,

    # Optimisation (Section V-B, Table XV)
    teacher_iters   = 10000,
    student_iters   = 10000,
    lr              = 1e-3,          # Adam lr=0.001

    batch_size      = 32,            # pairs per batch
    num_pairs       = 40000,         # pairs pre-sampled per pass

    num_workers     = 4,
    log_every       = 500,
    eval_every      = 2000,          # evaluate train+test every N iterations
    save_dir        = './ckpt_paper',
)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class CASIADataset(Dataset):
    """
    Supports both layouts:
      • Flat folder:  root/001_L_01.jpg  (label from filename token)
      • Subdirectory: root/001/01.jpg    (label from folder name)
    """
    EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.pgm', '.tiff'}

    def __init__(self, root, transform=None, label_pos=0, sep='_'):
        self.transform    = transform
        self.paths        = []
        self.labels       = []
        self.idx_by_label = defaultdict(list)

        subdirs = sorted(d for d in os.listdir(root)
                         if os.path.isdir(os.path.join(root, d)))
        if subdirs:                                # subdirectory layout
            lbl_map = {s: i for i, s in enumerate(subdirs)}
            for subj in subdirs:
                for fn in sorted(os.listdir(os.path.join(root, subj))):
                    if os.path.splitext(fn)[1].lower() in self.EXTS:
                        idx = len(self.paths)
                        self.paths.append(os.path.join(root, subj, fn))
                        lbl = lbl_map[subj]
                        self.labels.append(lbl)
                        self.idx_by_label[lbl].append(idx)
        else:                                      # flat layout
            files = sorted(fn for fn in os.listdir(root)
                           if os.path.splitext(fn)[1].lower() in self.EXTS)
            if not files:
                raise RuntimeError(f'No images found in {root!r}')
            subjects = sorted({fn.split(sep)[label_pos] for fn in files})
            lbl_map  = {s: i for i, s in enumerate(subjects)}
            for fn in files:
                sid = fn.split(sep)[label_pos]
                lbl = lbl_map[sid]
                idx = len(self.paths)
                self.paths.append(os.path.join(root, fn))
                self.labels.append(lbl)
                self.idx_by_label[lbl].append(idx)

    @property
    def num_classes(self): return len(self.idx_by_label)
    def __len__(self):     return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[i]


def split_dataset(ds, ratio=0.5):
    """First `ratio` fraction of each subject's images → train."""
    train_idx, test_idx = [], []
    for lbl, idxs in ds.idx_by_label.items():
        k = max(1, int(len(idxs) * ratio))
        train_idx.extend(idxs[:k])
        test_idx.extend(idxs[k:])
    return train_idx, test_idx


class PairDataset(Dataset):
    """
    Pre-generates (img_i, img_j, S_ij) triples.
    S_ij = 1  for genuine (same subject), 0 for imposter.
    """
    def __init__(self, base_ds, indices, num_pairs, seed=42):
        self.base = base_ds
        rng = random.Random(seed)
        lbl_map = defaultdict(list)
        for i in indices:
            lbl_map[base_ds.labels[i]].append(i)
        valid_lbls = [l for l, v in lbl_map.items() if len(v) >= 2]
        all_lbls   = list(lbl_map.keys())

        pairs = []
        half  = num_pairs // 2
        # genuine pairs
        for _ in range(half):
            l    = rng.choice(valid_lbls)
            a, b = rng.sample(lbl_map[l], 2)
            pairs.append((a, b, 1.0))
        # imposter pairs
        for _ in range(num_pairs - half):
            l1, l2 = rng.sample(all_lbls, 2)
            a = rng.choice(lbl_map[l1])
            b = rng.choice(lbl_map[l2])
            pairs.append((a, b, 0.0))
        rng.shuffle(pairs)
        self.pairs = pairs

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        a, b, s = self.pairs[i]
        x1, _  = self.base[a]
        x2, _  = self.base[b]
        return x1, x2, torch.tensor(s, dtype=torch.float32)


def get_transform(train=True, size=128):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ops = [transforms.Resize((size, size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(0.3),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1)]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


# ─────────────────────────────────────────────────────────────────────────────
# MODELS  — exact paper descriptions
# ─────────────────────────────────────────────────────────────────────────────
class TeacherDHN(nn.Module):
    """
    Paper Fig.5 / Section IV-A:
    VGG-16 backbone (all 5 conv-batches + pooling = features[0:31]).
    • Batch 1–4 (features[0:23]) : pretrained ImageNet weights, fine-tuned.
    • Batch 5   (features[23:30]): re-initialised, trained from scratch.
    • AdaptiveAvgPool → Flatten
    • FC head (trained from scratch):
        Linear(flat→4096, ReLU, Dropout) →
        Linear(4096→4096, ReLU, Dropout) →
        Linear(4096→128,  Tanh)          ← coding layer
    Paper says "DHN transforms the softmax layer of VGG-16 into a coding
    layer"; original VGG has FC(4096)→FC(4096)→FC(1000), so we replace
    the last layer with FC(128, tanh).
    """
    def __init__(self, hash_dim=128, pretrained=True):
        super().__init__()
        vgg = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)

        # ── Backbone: all 5 conv batches + pooling layers
        self.features   = vgg.features          # indices 0-30  (31 layers)
        self.avgpool    = nn.AdaptiveAvgPool2d((4, 4))   # safe for any input

        # ── batch5 (features[24:30]) trained from scratch per paper
        for m in list(self.features.children())[24:30]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

        # ── FC head: replaces VGG classifier; all trained from scratch
        flat = 512 * 4 * 4
        self.head = nn.Sequential(
            nn.Linear(flat, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, hash_dim),             # coding layer
            nn.Tanh(),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)                       # h ∈ (−1,+1)^D

    @torch.no_grad()
    def get_codes(self, x):
        return torch.sign(self.forward(x))        # b ∈ {−1,+1}^D


class StudentDHN(nn.Module):
    """
    Paper Fig.6 exactly:
      Conv1: 3×3,16ch,stride=4,padding=0(VALID) → ReLU → MaxPool(2×2,stride=1)
      Conv2: 5×5,32ch,stride=2,padding=0(VALID) → ReLU → MaxPool(2×2,stride=1)
      FC1:   → 512, ReLU
      FC3:   → 128, Tanh        (paper labels it FC3; it is the second FC layer)
      Code:  sign() at inference
    Standard ReLU, NO batch-norm. (paper Fig.6 shows ReLU boxes, no BN mentioned)
    """
    def __init__(self, hash_dim=128, img_size=128):
        super().__init__()
        self.conv = nn.Sequential(
            # Conv1 (paper: 3×3×16, stride=4, padding=0)
            nn.Conv2d(3,  16, kernel_size=3, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),    # 2×2, stride=1

            # Conv2 (paper: 5×5×32, stride=2, padding=0)
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),    # 2×2, stride=1
        )
        flat = self._flat_dim(img_size)
        self.fc = nn.Sequential(
            nn.Linear(flat, 512),     nn.ReLU(inplace=True),   # FC1
            nn.Linear(512, hash_dim), nn.Tanh(),                # FC3 / coding
        )
        self._init_weights()

    def _flat_dim(self, s):
        with torch.no_grad():
            return self.conv(torch.zeros(1, 3, s, s)).view(1, -1).shape[1]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(torch.flatten(self.conv(x), 1))

    @torch.no_grad()
    def get_codes(self, x): return torch.sign(self.forward(x))


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS — strict paper equations
# ─────────────────────────────────────────────────────────────────────────────
def loss_dhn_paper(h1, h2, s, t=180.0, w=1.0):
    """
    Eq.1:  L_h_{i,j} = 0.5·S·D + 0.5·(1−S)·max(t−D,0)
           D = ‖h_i − h_j‖²  (squared Euclidean — t=180 is calibrated to this)
    Eq.2:  L_q_i = ‖|h_i| − 1‖_2   (vector L2 norm per sample, then mean)
    Eq.3:  L_DHN = L_h + w·L_q
    """
    # Eq.1 — squared Euclidean distance
    D  = torch.sum((h1 - h2) ** 2, dim=1)                    # [B]
    Lh = (0.5 * s * D +
          0.5 * (1.0 - s) * torch.clamp(t - D, min=0.0)).mean()

    # Eq.2 — VECTOR L2 norm per sample (not MSE), then mean
    Lq = (torch.norm(torch.abs(h1) - 1.0, p=2, dim=1).mean() +
          torch.norm(torch.abs(h2) - 1.0, p=2, dim=1).mean()) / 2.0

    return Lh + w * Lq, Lh, Lq


def loss_rela_paper(bT1, bT2, bS1, bS2):
    """
    Eq.6:  L_rela = ‖d^T − d^S‖_2
    Eq.7:  d_ij^T = ‖b_i^T − b_j^T‖_2  (L2 norm, NOT squared)
    Eq.8:  d_ij^S = ‖b_i^S − b_j^S‖_2  (L2 norm)
    The outer ‖·‖_2 in Eq.6 is the L2 norm of the vector of differences.
    """
    dT = torch.norm(bT1 - bT2, p=2, dim=1)   # [B]
    dS = torch.norm(bS1 - bS2, p=2, dim=1)   # [B]
    return torch.norm(dT - dS, p=2)           # scalar


def loss_hard_paper(bT1, bT2, bS1, bS2, s):
    """
    Eq.9:  L_hard (exactly as written — NO clamp):
      For genuines  (S=1): max_S(d_genuine)  − min_T(d_genuine)
      For imposters (S=0): max_T(d_imposter) − min_S(d_imposter)
    d_ij = ‖b_i − b_j‖_2  (L2 norm, consistent with Eq.7/8)
    """
    dT = torch.norm(bT1 - bT2, p=2, dim=1)
    dS = torch.norm(bS1 - bS2, p=2, dim=1)
    gm = s.bool(); im = ~gm

    loss = bT1.new_zeros(())
    if gm.sum() >= 2:
        loss = loss + (dS[gm].max() - dT[gm].min())    # Eq.9 genuine term
    if im.sum() >= 2:
        loss = loss + (dT[im].max() - dS[im].min())    # Eq.9 imposter term
    return loss


def loss_student_total(bT1, bT2, bS1, bS2, s, cfg):
    """Eq.10: L = L_DHN + α·L_rela + β·L_hard"""
    Ldhn, Lh, Lq = loss_dhn_paper(bS1, bS2, s, cfg['margin_t'], cfg['w_quant'])
    Lrela         = loss_rela_paper(bT1, bT2, bS1, bS2)
    Lhard         = loss_hard_paper(bT1, bT2, bS1, bS2, s)
    total = Ldhn + cfg['alpha'] * Lrela + cfg['beta'] * Lhard
    return total, Ldhn, Lrela, Lhard


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS  (used during training and at final eval)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _extract_codes_internal(model, loader, device):
    """Extract binary codes; preserves model.training state."""
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


def _id_acc(tr_codes, tr_labels, te_codes, te_labels):
    """1-NN by Hamming distance. Training set used as gallery."""
    correct = 0
    for code, true_lbl in zip(te_codes, te_labels):
        ham = np.sum(code != tr_codes, axis=1)
        correct += int(tr_labels[np.argmin(ham)] == true_lbl)
    return correct / len(te_labels) * 100.0


def _eer(tr_codes, tr_labels, te_codes, te_labels):
    """EER via sklearn roc_curve with negative-Hamming similarity."""
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
    Returns (train_acc, train_eer, test_acc, test_eer).
    Train set is used as gallery for both train-query and test-query.
    """
    tr_c, tr_l = _extract_codes_internal(model, tr_ld, device)
    te_c, te_l = _extract_codes_internal(model, te_ld, device)
    tr_acc = _id_acc(tr_c, tr_l, tr_c, tr_l)
    tr_eer = _eer(tr_c, tr_l, tr_c, tr_l)
    te_acc = _id_acc(tr_c, tr_l, te_c, te_l)
    te_eer = _eer(tr_c, tr_l, te_c, te_l)
    return tr_acc, tr_eer, te_acc, te_eer


def _save_csv(rows, save_dir, fname):
    if not rows: return
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, fname), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def _infinite(loader):
    """Cycle a DataLoader infinitely — for iteration-based training."""
    while True:
        for batch in loader:
            yield batch


def train_teacher(cfg, device):
    log.info('=' * 65)
    log.info('STAGE 1 – Teacher (VGG-16 DHN, paper Fig.5)')
    log.info('=' * 65)

    ds_tr   = CASIADataset(cfg['data_dir'],
                           get_transform(True, cfg['img_size']),
                           cfg['label_pos'], cfg['sep'])
    ds_eval = CASIADataset(cfg['data_dir'],
                           get_transform(False, cfg['img_size']),
                           cfg['label_pos'], cfg['sep'])
    tr_idx, te_idx = split_dataset(ds_tr, cfg['train_ratio'])
    log.info(f'  {len(ds_tr)} images | {ds_tr.num_classes} subjects')
    log.info(f'  Train: {len(tr_idx)} images  |  Test: {len(te_idx)} images')

    pairs  = PairDataset(ds_tr, tr_idx, cfg['num_pairs'])
    loader = DataLoader(pairs, cfg['batch_size'], shuffle=True,
                        num_workers=cfg['num_workers'], pin_memory=True,
                        drop_last=True)
    tr_ld  = DataLoader(Subset(ds_eval, tr_idx), 64,
                        num_workers=cfg['num_workers'], pin_memory=True)
    te_ld  = DataLoader(Subset(ds_eval, te_idx), 64,
                        num_workers=cfg['num_workers'], pin_memory=True)

    model = TeacherDHN(cfg['hash_dim'], pretrained=True).to(device)
    opt   = optim.Adam(model.parameters(), lr=cfg['lr'])

    model.train()
    it      = _infinite(loader)
    best_te = 0.0
    history = []

    log.info(f'  {"Iter":>6}  {"Loss":>9}  {"Lh":>9}  {"Lq":>9}')
    log.info('  ' + '─' * 42)

    for step in range(1, cfg['teacher_iters'] + 1):
        x1, x2, s = next(it)
        x1, x2, s = x1.to(device), x2.to(device), s.to(device)
        h1, h2 = model(x1), model(x2)
        loss, Lh, Lq = loss_dhn_paper(h1, h2, s, cfg['margin_t'], cfg['w_quant'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg['log_every'] == 0 or step == 1:
            log.info(f'  {step:6d}  {loss.item():9.4f}  '
                     f'{Lh.item():9.4f}  {Lq.item():9.4f}')

        # ── Periodic train + test evaluation
        if cfg['eval_every'] > 0 and step % cfg['eval_every'] == 0:
            log.info(f'  ── Eval @ iter {step} {"─"*42}')
            tr_acc, tr_eer, te_acc, te_eer = run_metrics(
                model, tr_ld, te_ld, device)
            log.info(f'  TRAIN  Acc={tr_acc:6.2f}%  EER={tr_eer:5.2f}%   │   '
                     f'TEST   Acc={te_acc:6.2f}%  EER={te_eer:5.2f}%')
            history.append(dict(iter=step, tr_acc=tr_acc, tr_eer=tr_eer,
                                te_acc=te_acc, te_eer=te_eer))
            if te_acc > best_te:
                best_te = te_acc
                os.makedirs(cfg['save_dir'], exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(cfg['save_dir'], 'teacher.pth'))
                log.info(f'  ✓ New best teacher  (test Acc={te_acc:.2f}%)')
            log.info(f'  {"─"*60}')

    # ── Always save at end if eval_every=0 or no save triggered
    os.makedirs(cfg['save_dir'], exist_ok=True)
    ckpt_path = os.path.join(cfg['save_dir'], 'teacher.pth')
    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    # ── Final evaluation
    log.info('')
    log.info('  ── TEACHER FINAL EVALUATION ──')
    log.info(f'  {"Set":20s}  {"Acc (%)":>10}  {"EER (%)":>10}')
    log.info('  ' + '─' * 44)
    tr_acc, tr_eer, te_acc, te_eer = run_metrics(model, tr_ld, te_ld, device)
    log.info(f'  {"Training set":20s}  {tr_acc:>10.2f}  {tr_eer:>10.2f}')
    log.info(f'  {"Test set":20s}  {te_acc:>10.2f}  {te_eer:>10.2f}')
    log.info(f'  Teacher → {ckpt_path}')

    history.append(dict(iter='final', tr_acc=tr_acc, tr_eer=tr_eer,
                        te_acc=te_acc, te_eer=te_eer))
    _save_csv(history, cfg['save_dir'], 'teacher_history.csv')
    return model


def train_student(cfg, device, teacher=None):
    log.info('=' * 65)
    log.info('STAGE 2 – Student DDH (Eq.10, α=1, β=0.8)')
    log.info('=' * 65)

    ds_tr   = CASIADataset(cfg['data_dir'],
                           get_transform(True, cfg['img_size']),
                           cfg['label_pos'], cfg['sep'])
    ds_eval = CASIADataset(cfg['data_dir'],
                           get_transform(False, cfg['img_size']),
                           cfg['label_pos'], cfg['sep'])
    tr_idx, te_idx = split_dataset(ds_tr, cfg['train_ratio'])
    log.info(f'  Train: {len(tr_idx)} images  |  Test: {len(te_idx)} images')

    pairs  = PairDataset(ds_tr, tr_idx, cfg['num_pairs'])
    loader = DataLoader(pairs, cfg['batch_size'], shuffle=True,
                        num_workers=cfg['num_workers'], pin_memory=True,
                        drop_last=True)
    tr_ld  = DataLoader(Subset(ds_eval, tr_idx), 64,
                        num_workers=cfg['num_workers'], pin_memory=True)
    te_ld  = DataLoader(Subset(ds_eval, te_idx), 64,
                        num_workers=cfg['num_workers'], pin_memory=True)

    if teacher is None:
        teacher = TeacherDHN(cfg['hash_dim'], pretrained=False).to(device)
        ckpt = os.path.join(cfg['save_dir'], 'teacher.pth')
        teacher.load_state_dict(torch.load(ckpt, map_location=device))
        log.info(f'  Teacher loaded from {ckpt}')
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── Teacher baseline before student training
    log.info('  ── Teacher baseline ──')
    log.info(f'  {"Set":20s}  {"Acc (%)":>10}  {"EER (%)":>10}')
    log.info('  ' + '─' * 44)
    tch_tr_acc, tch_tr_eer, tch_te_acc, tch_te_eer = run_metrics(
        teacher, tr_ld, te_ld, device)
    log.info(f'  {"Training set":20s}  {tch_tr_acc:>10.2f}  {tch_tr_eer:>10.2f}')
    log.info(f'  {"Test set":20s}  {tch_te_acc:>10.2f}  {tch_te_eer:>10.2f}')
    log.info('')

    student = StudentDHN(cfg['hash_dim'], cfg['img_size']).to(device)
    opt     = optim.Adam(student.parameters(), lr=cfg['lr'])

    student.train()
    it      = _infinite(loader)
    best_te = 0.0
    history = []

    log.info(f'  {"Iter":>6}  {"Loss":>9}  {"DHN":>9}  '
             f'{"Lrela":>9}  {"Lhard":>9}')
    log.info('  ' + '─' * 54)

    for step in range(1, cfg['student_iters'] + 1):
        x1, x2, s = next(it)
        x1, x2, s = x1.to(device), x2.to(device), s.to(device)

        with torch.no_grad():
            bT1, bT2 = teacher(x1), teacher(x2)
        bS1, bS2 = student(x1), student(x2)

        loss, Ldhn, Lrela, Lhard = loss_student_total(
            bT1, bT2, bS1, bS2, s, cfg)

        # Guard: skip batch if non-finite
        if not torch.isfinite(loss):
            log.warning(f'  iter {step}: non-finite loss '
                        f'(DHN={Ldhn.item():.4f} '
                        f'Lrela={Lrela.item():.4f} '
                        f'Lhard={Lhard.item():.4f}) — skipping batch')
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        opt.step()

        if step % cfg['log_every'] == 0 or step == 1:
            log.info(f'  {step:6d}  {loss.item():9.4f}  '
                     f'{Ldhn.item():9.4f}  '
                     f'{Lrela.item():9.4f}  '
                     f'{Lhard.item():9.4f}')

        # ── Periodic train + test evaluation
        if cfg['eval_every'] > 0 and step % cfg['eval_every'] == 0:
            log.info(f'  ── Eval @ iter {step} {"─"*42}')
            tr_acc, tr_eer, te_acc, te_eer = run_metrics(
                student, tr_ld, te_ld, device)
            log.info(f'  TRAIN  Acc={tr_acc:6.2f}%  EER={tr_eer:5.2f}%   │   '
                     f'TEST   Acc={te_acc:6.2f}%  EER={te_eer:5.2f}%')
            history.append(dict(iter=step, tr_acc=tr_acc, tr_eer=tr_eer,
                                te_acc=te_acc, te_eer=te_eer))
            if te_acc > best_te:
                best_te = te_acc
                os.makedirs(cfg['save_dir'], exist_ok=True)
                torch.save(student.state_dict(),
                           os.path.join(cfg['save_dir'], 'student.pth'))
                log.info(f'  ✓ New best student  (test Acc={te_acc:.2f}%)')
            log.info(f'  {"─"*60}')

    # ── Save at end if nothing was saved yet
    os.makedirs(cfg['save_dir'], exist_ok=True)
    stu_path = os.path.join(cfg['save_dir'], 'student.pth')
    if not os.path.exists(stu_path):
        torch.save(student.state_dict(), stu_path)

    # ── Final comparison table: Teacher vs DDH
    log.info('')
    log.info('  ══ FINAL RESULTS ══════════════════════════════════════════')
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

    log.info(f'  Student → {stu_path}')
    _save_csv(all_results, cfg['save_dir'], 'results_final.csv')
    _save_csv(history,     cfg['save_dir'], 'student_history.csv')
    log.info(f"  CSV → {os.path.join(cfg['save_dir'], 'results_final.csv')}")
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
    """1-NN identification by Hamming distance (Section V-C)."""
    correct = 0
    for code, true_lbl in zip(te_codes, te_labels):
        ham     = np.sum(code != tr_codes, axis=1)
        correct += int(tr_labels[np.argmin(ham)] == true_lbl)
    return correct / len(te_labels) * 100.0


def compute_eer(tr_codes, tr_labels, te_codes, te_labels):
    """EER via sklearn roc_curve (Section V-D)."""
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
    log.info('=' * 65)
    log.info('EVALUATION  (loading saved checkpoints)')
    log.info('=' * 65)
    ds = CASIADataset(cfg['data_dir'],
                      get_transform(False, cfg['img_size']),
                      cfg['label_pos'], cfg['sep'])
    tr_idx, te_idx = split_dataset(ds, cfg['train_ratio'])
    log.info(f'  Train: {len(tr_idx)} images  |  Test: {len(te_idx)} images')
    tr_ld = DataLoader(Subset(ds, tr_idx), 64, num_workers=cfg['num_workers'])
    te_ld = DataLoader(Subset(ds, te_idx), 64, num_workers=cfg['num_workers'])

    log.info(f'  {"Model":22s}  {"Train Acc":>10}  {"Train EER":>10}  '
             f'{"Test Acc":>10}  {"Test EER":>10}')
    log.info('  ' + '─' * 68)

    results = []
    for name, ckpt_fn, make_model in [
        ('Teacher',       'teacher.pth',
         lambda: TeacherDHN(cfg['hash_dim'], pretrained=False)),
        ('DDH (Student)', 'student.pth',
         lambda: StudentDHN(cfg['hash_dim'], cfg['img_size'])),
    ]:
        ckpt = os.path.join(cfg['save_dir'], ckpt_fn)
        if not os.path.exists(ckpt):
            log.warning(f'  {name}: checkpoint not found, skipping'); continue
        m = make_model().to(device)
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description='DDH – paper equations')
    p.add_argument('--data_dir',   default=CFG['data_dir'])
    p.add_argument('--stage',      default='all',
                   choices=['all', 'teacher', 'student', 'eval'])
    p.add_argument('--label_pos',  type=int, default=CFG['label_pos'],
                   help='token index of subject ID in filename')
    p.add_argument('--sep',        default=CFG['sep'],
                   help='filename separator')
    args = p.parse_args()
    CFG.update(data_dir=args.data_dir, label_pos=args.label_pos, sep=args.sep)

    os.makedirs(CFG['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device : {device}')
    log.info(f'Script : ddh_paper.py  (paper equations only)')
    log.info(f'α={CFG["alpha"]}  β={CFG["beta"]}  t={CFG["margin_t"]}  '
             f'lr={CFG["lr"]}  iters={CFG["teacher_iters"]}')

    teacher = None
    if args.stage in ('all', 'teacher'):
        teacher = train_teacher(CFG, device)
    if args.stage in ('all', 'student'):
        train_student(CFG, device, teacher)
    if args.stage in ('all', 'eval'):
        evaluate(CFG, device)


if __name__ == '__main__':
    main()
  
