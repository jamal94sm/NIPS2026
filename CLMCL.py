#!/usr/bin/env python3
"""
C-LMCL for Open-Set Palmprint Recognition — CASIA-MS Dataset
=============================================================
Paper : Zhong & Zhu, "Centralized Large Margin Cosine Loss for Open-Set
        Deep Palmprint Recognition", IEEE TCSVT, Vol. 30, No. 6, 2020.

Dataset layout
--------------
    DATA_ROOT/{subjectID}_{handSide}_{spectrum}_{iteration}.jpg
    e.g.  001_L_460_01.jpg

Identity mapping
----------------
    Each (subjectID, hand) pair  →  one class label.
    100 subjects × 2 hands = 200 classes.
    Labels are assigned in sorted order: 001_L=0, 001_R=1, 002_L=2 …

Protocols
---------
    cross_subject  — 80/20 identity split, 8-fold CV, all spectra pooled.
                     Train classes ≠ test classes.  (Table II / III in paper)
    cross_spectrum — Leave-one-spectrum-out.  Same 200 IDs in train & test,
                     but the held-out spectrum is unseen during training.

Usage
-----
    # Full run (both protocols, paper hyper-params)
    python clmcl_casia.py --mode both

    # Quick smoke-test  (100 iterations, 1 fold)
    python clmcl_casia.py --mode both --max_iter 100 --n_folds 1

    # One protocol only
    python clmcl_casia.py --mode cross_subject
    python clmcl_casia.py --mode cross_spectrum

Requirements
------------
    pip install torch torchvision numpy scipy scikit-learn matplotlib Pillow
"""

import os, sys, glob, random, math, argparse, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════
# 0.  GLOBAL CONFIGURATION  (match paper Sec. IV-B exactly)
# ══════════════════════════════════════════════════════════════════════
CFG = {
    # ── Data ──────────────────────────────────────────────────────────
    'data_root'   : '/home/pai-ng/Jamal/CASIA-MS-ROI',
    'img_size'    : 224,       # ROI resize target (paper Sec. IV-B-1)

    # ── C-LMCL hyper-parameters (paper Sec. IV-B-3) ──────────────────
    's'           : 30,        # feature-norm scale
    'm'           : 0.65,      # cosine margin
    'lambda_c'    : 0.10,      # center-loss weight  λ
    'feat_dim'    : 128,       # FC1 output dimension

    # ── Optimiser (paper Sec. IV-B-3) ────────────────────────────────
    'batch_size'  : 55,
    'lr'          : 0.01,      # SGD base LR
    'weight_decay': 5e-4,
    'momentum'    : 0.9,
    'max_iter'    : 30_000,    # gradient steps
    'lr_steps'    : [16_000, 24_000, 28_000],   # divide LR by 10 at each
    'lr_gamma'    : 0.1,
    'center_lr'   : 0.5,       # α in eq. (6-7) for center update

    # ── Cross-subject protocol ────────────────────────────────────────
    'train_ratio' : 0.80,      # fraction of identities for training
    'n_folds'     : 8,         # independent splits

    # ── Misc ──────────────────────────────────────────────────────────
    'out_dir'     : './clmcl_results',
    'seed'        : 42,
    'num_workers' : 4,
    'device'      : 'cuda' if torch.cuda.is_available() else 'cpu',
}


# ══════════════════════════════════════════════════════════════════════
# 1.  DATASET
# ══════════════════════════════════════════════════════════════════════
class CASIAMultispectralDataset(Dataset):
    """
    Loads CASIA-MS-ROI palmprint images from the flat directory.

    Filtering
    ---------
    allowed_spectra : list[str] | None   — keep only these spectrum codes
    allowed_ids     : set[int]  | None   — keep only these class labels
    """

    def __init__(self, data_root, transform=None,
                 allowed_spectra=None, allowed_ids=None):
        self.root      = Path(data_root)
        self.transform = transform
        self.samples   = []     # (path_str, class_id, spectrum, iteration)

        self._build_index(allowed_spectra, allowed_ids)

    # ──────────────────────────────────────────────────────────────────
    def _build_index(self, allowed_spectra, allowed_ids):
        paths = sorted(self.root.glob('*.jpg')) or sorted(self.root.glob('*.png'))
        assert paths, f"No images found in {self.root}"

        # Pass 1 — discover all (subject, hand) pairs for a stable mapping
        id_set = set()
        for p in paths:
            parts = p.stem.split('_')
            id_set.add((parts[0], parts[1]))

        # Deterministic label assignment
        sorted_ids = sorted(id_set)
        self.class_to_idx = {f'{s}_{h}': i for i, (s, h) in enumerate(sorted_ids)}
        self.idx_to_class  = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes   = len(self.class_to_idx)

        # Pass 2 — collect filtered samples
        all_spectra = set()
        for p in paths:
            parts     = p.stem.split('_')
            subject   = parts[0]
            hand      = parts[1]
            spectrum  = parts[2]
            iteration = parts[3] if len(parts) > 3 else '01'
            all_spectra.add(spectrum)

            if allowed_spectra and spectrum not in allowed_spectra:
                continue
            class_id = self.class_to_idx[f'{subject}_{hand}']
            if allowed_ids is not None and class_id not in allowed_ids:
                continue
            self.samples.append((str(p), class_id, spectrum, iteration))

        self.all_spectra = sorted(all_spectra)

    # ──────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_id, spectrum, iteration = self.samples[idx]
        # Grayscale → RGB by replication  (as per paper — 3-channel input)
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, class_id, spectrum, iteration


# ── Lightweight subset wrappers ───────────────────────────────────────
class _MappedSubset(Dataset):
    """Subset with optional class-label remapping (used for training)."""
    def __init__(self, base, indices, label_remap, transform):
        self.base       = base
        self.indices    = indices
        self.remap      = label_remap
        self.transform  = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, old_lbl, spec, it = self.base.samples[self.indices[i]]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.remap[old_lbl], spec, it


class _Subset(Dataset):
    """Subset without label remapping (used for evaluation)."""
    def __init__(self, base, indices, transform):
        self.base       = base
        self.indices    = indices
        self.transform  = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, lbl, spec, it = self.base.samples[self.indices[i]]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, lbl, spec, it


# ── Transforms (paper Sec. IV-B-1) ───────────────────────────────────
def get_transform(img_size=224, augment=True):
    if augment:
        return transforms.Compose([
            transforms.Resize(img_size + 32),
            # Resized crop: range [240, 256] then 224×224 patch
            transforms.RandomResizedCrop(img_size, scale=(240/256, 1.0)),
            transforms.RandomRotation(5),                     # ±5°
            transforms.ColorJitter(brightness=0.4, contrast=0.4),   # α∈[0.5,1.5], β∈[-50,+50]
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),           # pixel/128 − 1
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])


def make_loader(ds, batch_size, shuffle, num_workers=4):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=shuffle)


# ══════════════════════════════════════════════════════════════════════
# 2.  MODEL — ResNet-20 (Table I + Fig. 2 in paper)
# ══════════════════════════════════════════════════════════════════════
# Layer count breakdown:
#   Stage 1: 1 plain + 1 residual (2 convs)  =  3 convolutions
#   Stage 2: 1 plain + 2 residuals (4 convs) =  5 convolutions
#   Stage 3: 1 plain + 4 residuals (8 convs) =  9 convolutions
#   Stage 4: 1 plain + 1 residual (2 convs)  =  3 convolutions
#   Total: 20 convolutions → "ResNet-20"

class _Plain(nn.Module):
    """Single 3×3 conv-BN-ReLU (stride 2) — spatial downsampling."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.seq(x)


class _Residual(nn.Module):
    """Two 3×3 convs with identity / projection shortcut (stride 1)."""
    def __init__(self, ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(ch, ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
    def forward(self, x): return F.relu(self.seq(x) + x)


class PalmResNet20(nn.Module):
    """
    Architecture from Table I of the paper.

    Input  : 3 × 224 × 224
    Stage 1: plain(3→64, /2) + 1 residual   → 64  × 112 × 112
    Stage 2: plain(64→128,/2) + 2 residuals → 128 ×  56 ×  56
    Stage 3: plain(128→256,/2)+ 4 residuals → 256 ×  28 ×  28
    Stage 4: plain(256→512,/2)+ 1 residual  → 512 ×  14 ×  14
    GAP    :                                 → 512 ×   1 ×   1
    FC1    :                                 → 128-dim feature
    """
    def __init__(self, feat_dim=128):
        super().__init__()
        self.stage1 = nn.Sequential(_Plain(3, 64), _Residual(64))
        self.stage2 = nn.Sequential(_Plain(64, 128), _Residual(128), _Residual(128))
        self.stage3 = nn.Sequential(_Plain(128, 256),
                                    _Residual(256), _Residual(256),
                                    _Residual(256), _Residual(256))
        self.stage4 = nn.Sequential(_Plain(256, 512), _Residual(512))
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(512, feat_dim)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        return x   # raw 128-dim embedding (not yet L2-normalised)


# ══════════════════════════════════════════════════════════════════════
# 3.  LOSSES
# ══════════════════════════════════════════════════════════════════════

class LMCLLoss(nn.Module):
    """
    Large Margin Cosine Loss — eq. (4) in paper.
    (Wang et al. "CosFace", CVPR 2018)

    L_lmc = -(1/N) Σ log[ e^{s(cosθ_{yi}−m)} / (e^{s(cosθ_{yi}−m)} + Σ_{j≠yi} e^{s·cosθ_j}) ]
    """
    def __init__(self, num_classes, feat_dim, s=30.0, m=0.65):
        super().__init__()
        self.s  = s
        self.m  = m
        self.W  = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, feat, labels):
        feat_n = F.normalize(feat, p=2, dim=1)          # (N, D)
        W_n    = F.normalize(self.W,  p=2, dim=1)        # (C, D)
        cosine = feat_n @ W_n.t()                         # (N, C)

        # Subtract margin only from the ground-truth class cosine
        margin_mask = torch.zeros_like(cosine)
        margin_mask.scatter_(1, labels.unsqueeze(1), self.m)
        logits = self.s * (cosine - margin_mask)

        loss = F.cross_entropy(logits, labels)
        return loss, cosine.detach()


class CenterLoss(nn.Module):
    """
    Center Loss — eq. (5) in paper.
    (Wen et al. ECCV 2016)

    L_c = (1/2) Σ_i ||x_i − c_{y_i}||²

    Centers are updated via a dedicated SGD optimiser (α in eq. 6-7).
    """
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, labels):
        c    = self.centers[labels]                   # (N, D)
        loss = 0.5 * ((feat - c) ** 2).sum(dim=1).mean()
        return loss


class CLMCLoss(nn.Module):
    """
    Centralized LMCL — eq. (8) in paper.

    L = L_lmc + λ · L_c
    """
    def __init__(self, num_classes, feat_dim, s=30, m=0.65, lam=0.1):
        super().__init__()
        self.lmcl   = LMCLLoss(num_classes, feat_dim, s, m)
        self.center = CenterLoss(num_classes, feat_dim)
        self.lam    = lam

    def forward(self, feat, labels):
        l_lmc, cosine = self.lmcl(feat, labels)
        l_c            = self.center(feat, labels)
        total          = l_lmc + self.lam * l_c
        return total, l_lmc.item(), l_c.item(), cosine


# ══════════════════════════════════════════════════════════════════════
# 4.  TRAINING
# ══════════════════════════════════════════════════════════════════════
def train_model(model, criterion, optimizer, center_opt,
                loader, device, max_iter, lr_steps, lr_gamma,
                out_dir, tag):
    """
    Train for exactly max_iter gradient steps.
    LR schedule: divide by 10 at each milestone in lr_steps.
    Returns the trained model.
    """
    model.train()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_steps, gamma=lr_gamma)

    log_path = os.path.join(out_dir, f'train_{tag}.log')
    log_f    = open(log_path, 'w', buffering=1)

    it         = 0
    loss_buf   = []

    while it < max_iter:
        for batch in loader:
            if it >= max_iter:
                break

            imgs, labels, _, _ = batch
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            center_opt.zero_grad()

            feat = model(imgs)
            loss, l_lmc, l_c, _ = criterion(feat, labels)
            loss.backward()

            optimizer.step()

            # Center update: scale gradient by 1/(1 + batch_size) as per Wen et al.
            for p in criterion.center.parameters():
                if p.grad is not None:
                    p.grad.data *= 1.0 / (1 + imgs.size(0))
            center_opt.step()
            scheduler.step()

            loss_buf.append(loss.item())
            it += 1

            if it % 500 == 0 or it == 1:
                avg = np.mean(loss_buf[-500:])
                lr  = optimizer.param_groups[0]['lr']
                msg = (f'[{tag}] iter {it:6d}/{max_iter} | '
                       f'total={avg:.4f}  lmcl={l_lmc:.4f}  '
                       f'center={l_c:.4f}  lr={lr:.2e}')
                print(msg)
                log_f.write(msg + '\n')

    log_f.close()
    return model


# ══════════════════════════════════════════════════════════════════════
# 5.  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_features(model, loader, device):
    """
    Returns (feats, labels, spectra, iters).

    Feature = concat(f_original, f_mirror) → 256-dim, then L2-normalised.
    (Paper Sec. IV-B-4: "we concatenate features of the original image
     and its mirror image together as the final palmprint representation")
    """
    model.eval()
    all_feats, all_labels, all_spectra, all_iters = [], [], [], []

    for imgs, labels, spectra, iters in loader:
        imgs     = imgs.to(device, non_blocking=True)
        imgs_mir = torch.flip(imgs, dims=[-1])        # horizontal mirror

        f_orig = model(imgs)
        f_mir  = model(imgs_mir)
        feat   = torch.cat([f_orig, f_mir], dim=1)   # (N, 256)
        feat   = F.normalize(feat, p=2, dim=1)        # L2 normalise

        all_feats.append(feat.cpu().numpy())
        all_labels.extend(labels.numpy().tolist())
        all_spectra.extend(list(spectra))
        all_iters.extend(list(iters))

    return (np.vstack(all_feats),
            np.array(all_labels, dtype=np.int64),
            np.array(all_spectra),
            np.array(all_iters))


# ══════════════════════════════════════════════════════════════════════
# 6.  METRICS
# ══════════════════════════════════════════════════════════════════════
def split_gallery_probe(iters):
    """
    First-half iterations → gallery  |  second-half → probe.
    Mirrors the paper's session-1 / session-2 split.
    Returns gallery_mask and probe_mask (bool arrays).
    """
    unique_iters = sorted(set(iters))
    half         = max(1, len(unique_iters) // 2)
    gallery_set  = set(unique_iters[:half])
    probe_set    = set(unique_iters[half:]) or set(unique_iters)  # fallback

    g_mask = np.array([it in gallery_set for it in iters])
    p_mask = np.array([it in probe_set   for it in iters])
    return g_mask, p_mask


def rank1(g_feats, g_labels, p_feats, p_labels):
    """Cosine-distance Rank-1 identification rate (%)."""
    sim  = p_feats @ g_feats.T                    # (P, G)
    pred = g_labels[sim.argmax(axis=1)]
    return (pred == p_labels).mean() * 100.0


def verification_metrics(g_feats, g_labels, p_feats, p_labels):
    """
    Compute EER, TAR@FAR=0.01%, TAR@FAR=0.1%, and full (FPR, TPR) curve.
    """
    sim    = (p_feats @ g_feats.T).ravel()
    n_p, n_g = p_feats.shape[0], g_feats.shape[0]

    # Build ground-truth labels: 1 = genuine, 0 = impostor
    p_rep = np.repeat(p_labels, n_g)              # (P*G,)
    g_tile = np.tile(g_labels, n_p)               # (P*G,)
    y_true = (p_rep == g_tile).astype(np.int32)

    fpr, tpr, _ = roc_curve(y_true, sim, pos_label=1)
    fnr = 1.0 - tpr

    # EER — crossover of FAR and FRR
    diff = np.abs(fpr - fnr)
    idx  = diff.argmin()
    eer  = float((fpr[idx] + fnr[idx]) / 2.0 * 100.0)

    # TAR at fixed FAR using linear interpolation
    def _tar_at_far(target_far):
        try:
            return float(np.interp(target_far, fpr, tpr)) * 100.0
        except Exception:
            return float('nan')

    tar_001 = _tar_at_far(0.0001)   # FAR = 0.01%
    tar_01  = _tar_at_far(0.001)    # FAR = 0.1%

    return eer, tar_001, tar_01, fpr, tpr


def evaluate(model, test_ds, device, batch_size=128, tag=''):
    """
    Run full gallery/probe evaluation and print results.
    Returns a result dict.
    """
    loader = make_loader(test_ds, batch_size, shuffle=False,
                         num_workers=CFG['num_workers'])
    feats, labels, spectra, iters = extract_features(model, loader, device)

    g_mask, p_mask = split_gallery_probe(iters)
    g_feats, g_labels = feats[g_mask], labels[g_mask]
    p_feats, p_labels = feats[p_mask], labels[p_mask]

    r1                        = rank1(g_feats, g_labels, p_feats, p_labels)
    eer, tar_001, tar_01, fpr, tpr = verification_metrics(
        g_feats, g_labels, p_feats, p_labels)

    sep = '─' * 58
    print(f'\n{sep}')
    print(f'  {tag}')
    print(f'  Gallery : {g_mask.sum()} imgs | Probe : {p_mask.sum()} imgs')
    print(f'  Rank-1 Identification  :  {r1:.2f}%')
    print(f'  EER                    :  {eer:.4f}%')
    print(f'  TAR @ FAR = 0.01 %     :  {tar_001:.2f}%')
    print(f'  TAR @ FAR = 0.1  %     :  {tar_01:.2f}%')
    print(f'{sep}\n')

    return dict(rank1=r1, eer=eer, tar_001=tar_001, tar_01=tar_01,
                fpr=fpr, tpr=tpr)


# ══════════════════════════════════════════════════════════════════════
# 7.  CROSS-SUBJECT EXPERIMENT
# ══════════════════════════════════════════════════════════════════════
def run_cross_subject(cfg):
    """
    Protocol (paper Sec. IV-A / IV-C):
      • 80% identities  → training   (160 of 200)
      • 20% identities  → testing    (40  of 200)
      • All spectra pooled (spectrum label ignored)
      • 8 independent random splits, metrics averaged
    """
    print('\n' + '═'*60)
    print('  PROTOCOL 1 — CROSS-SUBJECT')
    print('═'*60)

    device  = cfg['device']
    out_dir = os.path.join(cfg['out_dir'], 'cross_subject')
    os.makedirs(out_dir, exist_ok=True)

    # Load full dataset (no transform yet) for index manipulation
    full_ds = CASIAMultispectralDataset(cfg['data_root'])
    ids     = list(range(full_ds.num_classes))

    fold_results = []

    for fold in range(cfg['n_folds']):
        print(f'\n── Fold {fold + 1} / {cfg["n_folds"]} ─────────────────────────')

        random.shuffle(ids)
        n_train    = int(len(ids) * cfg['train_ratio'])
        train_ids  = set(ids[:n_train])
        test_ids   = set(ids[n_train:])

        # Index arrays into full_ds.samples
        tr_idx = [i for i, (_, c, _, _) in enumerate(full_ds.samples)
                  if c in train_ids]
        te_idx = [i for i, (_, c, _, _) in enumerate(full_ds.samples)
                  if c in test_ids]

        # Re-map training labels to 0 … N_train-1
        id_remap = {old: new for new, old in enumerate(sorted(train_ids))}

        train_ds = _MappedSubset(full_ds, tr_idx, id_remap,
                                  get_transform(cfg['img_size'], augment=True))
        test_ds  = _Subset(full_ds, te_idx,
                            get_transform(cfg['img_size'], augment=False))

        print(f'  Train : {len(train_ds)} imgs, {len(train_ids)} IDs  |  '
              f'Test : {len(test_ds)} imgs, {len(test_ids)} IDs')

        loader = make_loader(train_ds, cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'])

        # ── Build model + losses ──────────────────────────────────────
        model     = PalmResNet20(feat_dim=cfg['feat_dim']).to(device)
        criterion = CLMCLoss(len(train_ids), cfg['feat_dim'],
                              cfg['s'], cfg['m'], cfg['lambda_c']).to(device)

        # Main optimiser covers backbone + LMCL weight matrix W
        optimizer  = torch.optim.SGD(
            list(model.parameters()) + list(criterion.lmcl.parameters()),
            lr=cfg['lr'], momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])
        # Separate optimiser for center vectors (α = center_lr)
        center_opt = torch.optim.SGD(
            criterion.center.parameters(), lr=cfg['center_lr'])

        # ── Train ─────────────────────────────────────────────────────
        model = train_model(model, criterion, optimizer, center_opt,
                             loader, device,
                             cfg['max_iter'], cfg['lr_steps'], cfg['lr_gamma'],
                             out_dir, tag=f'fold{fold + 1}')

        torch.save(model.state_dict(),
                   os.path.join(out_dir, f'model_fold{fold + 1}.pth'))

        # ── Evaluate ──────────────────────────────────────────────────
        result = evaluate(model, test_ds, device,
                           tag=f'Cross-Subject  Fold {fold + 1}')
        fold_results.append(result)

    # ── Aggregate across folds ────────────────────────────────────────
    print('\n' + '═'*60)
    print('  CROSS-SUBJECT — AGGREGATE OVER FOLDS')
    print(f'  {"Metric":20s}  {"Mean":>10}  {"Std":>10}')
    print('  ' + '─'*44)
    for k, label in [('rank1','Rank-1 (%)'),
                      ('eer',  'EER (%)'),
                      ('tar_001','TAR@FAR=0.01% (%)'),
                      ('tar_01', 'TAR@FAR=0.1%  (%)')]:
        vals = [r[k] for r in fold_results]
        print(f'  {label:20s}  {np.mean(vals):10.4f}  {np.std(vals):10.4f}')

    _plot_roc_folds(fold_results,
                    os.path.join(out_dir, 'roc_cross_subject.pdf'),
                    'Cross-Subject ROC Curves')
    _save_results_txt(fold_results,
                      os.path.join(out_dir, 'results_cross_subject.txt'))
    return fold_results


# ══════════════════════════════════════════════════════════════════════
# 8.  CROSS-SPECTRUM EXPERIMENT
# ══════════════════════════════════════════════════════════════════════
def run_cross_spectrum(cfg):
    """
    Protocol (user specification):
      • Leave-one-spectrum-out.
      • All 200 identities appear in BOTH train and test sets.
      • Train spectra ≠ test spectra  (domain shift evaluation).
      • Gallery / Probe split within the test spectrum by iteration.
    """
    print('\n' + '═'*60)
    print('  PROTOCOL 2 — CROSS-SPECTRUM  (leave-one-out)')
    print('═'*60)

    device  = cfg['device']
    out_dir = os.path.join(cfg['out_dir'], 'cross_spectrum')
    os.makedirs(out_dir, exist_ok=True)

    # Discover all spectra
    meta_ds     = CASIAMultispectralDataset(cfg['data_root'])
    all_spectra = sorted(meta_ds.all_spectra)
    num_classes  = meta_ds.num_classes
    print(f'\n  Spectra found : {all_spectra}')
    print(f'  Total classes : {num_classes}')

    spectrum_results = {}

    for test_spec in all_spectra:
        train_spectra = [s for s in all_spectra if s != test_spec]
        print(f'\n── Test spec : {test_spec}  |  Train spectra : {train_spectra} ──')

        train_ds = CASIAMultispectralDataset(
            cfg['data_root'],
            transform=get_transform(cfg['img_size'], augment=True),
            allowed_spectra=train_spectra)

        test_ds  = CASIAMultispectralDataset(
            cfg['data_root'],
            transform=get_transform(cfg['img_size'], augment=False),
            allowed_spectra=[test_spec])

        print(f'  Train : {len(train_ds)} imgs  |  Test : {len(test_ds)} imgs')

        loader = make_loader(train_ds, cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'])

        model     = PalmResNet20(feat_dim=cfg['feat_dim']).to(device)
        criterion = CLMCLoss(num_classes, cfg['feat_dim'],
                              cfg['s'], cfg['m'], cfg['lambda_c']).to(device)

        optimizer  = torch.optim.SGD(
            list(model.parameters()) + list(criterion.lmcl.parameters()),
            lr=cfg['lr'], momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])
        center_opt = torch.optim.SGD(
            criterion.center.parameters(), lr=cfg['center_lr'])

        model = train_model(model, criterion, optimizer, center_opt,
                             loader, device,
                             cfg['max_iter'], cfg['lr_steps'], cfg['lr_gamma'],
                             out_dir, tag=f'spec_{test_spec}')

        torch.save(model.state_dict(),
                   os.path.join(out_dir, f'model_spec_{test_spec}.pth'))

        result = evaluate(model, test_ds, device,
                           tag=f'Cross-Spectrum  test={test_spec}')
        spectrum_results[test_spec] = result

    # ── Summary table ─────────────────────────────────────────────────
    print('\n' + '═'*60)
    print('  CROSS-SPECTRUM — RESULTS PER HELD-OUT SPECTRUM')
    hdr = f'  {"Spectrum":>10}  {"Rank-1":>8}  {"EER":>8}  {"TAR@0.01%":>10}  {"TAR@0.1%":>9}'
    print(hdr)
    print('  ' + '─' * (len(hdr) - 2))
    for spec, r in spectrum_results.items():
        print(f'  {spec:>10}  {r["rank1"]:8.2f}  {r["eer"]:8.4f}'
              f'  {r["tar_001"]:10.2f}  {r["tar_01"]:9.2f}')

    _plot_roc_spectra(spectrum_results,
                      os.path.join(out_dir, 'roc_cross_spectrum.pdf'))
    _save_spectrum_txt(spectrum_results,
                       os.path.join(out_dir, 'results_cross_spectrum.txt'))
    return spectrum_results


# ══════════════════════════════════════════════════════════════════════
# 9.  PLOTTING  &  SAVE HELPERS
# ══════════════════════════════════════════════════════════════════════
def _plot_roc_folds(fold_results, save_path, title):
    """Plot per-fold ROC + mean ROC on a single figure."""
    plt.figure(figsize=(7, 6))
    mean_fpr = np.linspace(0, 0.04, 1000)
    tprs_interp = []

    for i, r in enumerate(fold_results):
        tpr_i = np.interp(mean_fpr, r['fpr'], r['tpr'])
        tprs_interp.append(tpr_i)
        plt.plot(mean_fpr * 100, (1 - tpr_i) * 100,
                 lw=0.8, alpha=0.4, label=f'Fold {i+1} EER={r["eer"]:.3f}%')

    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_eer = np.mean([r['eer'] for r in fold_results])
    plt.plot(mean_fpr * 100, (1 - mean_tpr) * 100,
             'k-', lw=2.5, label=f'Mean  EER={mean_eer:.3f}%')
    plt.plot([0, 4], [0, 4], 'r--', lw=1, label='EER line')

    plt.xlim(0, 4); plt.ylim(0, 4)
    plt.xlabel('FAR (%)'); plt.ylabel('FRR (%)')
    plt.title(title); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f'  ROC saved → {save_path}')


def _plot_roc_spectra(spectrum_results, save_path):
    """One ROC curve per test spectrum."""
    plt.figure(figsize=(7, 6))
    max_far = 0.04
    fpr_grid = np.linspace(0, max_far, 1000)

    for spec, r in spectrum_results.items():
        tpr_i = np.interp(fpr_grid, r['fpr'], r['tpr'])
        plt.plot(fpr_grid * 100, (1 - tpr_i) * 100,
                 lw=1.5, label=f'Spec {spec}  EER={r["eer"]:.3f}%')

    plt.plot([0, max_far*100], [0, max_far*100], 'k--', lw=1, label='EER line')
    plt.xlim(0, max_far*100); plt.ylim(0, max_far*100)
    plt.xlabel('FAR (%)'); plt.ylabel('FRR (%)')
    plt.title('Cross-Spectrum ROC Curves'); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()
    print(f'  ROC saved → {save_path}')


def _save_results_txt(fold_results, path):
    lines = ['fold,rank1,eer,tar_001,tar_01']
    for i, r in enumerate(fold_results):
        lines.append(f'{i+1},{r["rank1"]:.4f},{r["eer"]:.4f},'
                     f'{r["tar_001"]:.4f},{r["tar_01"]:.4f}')
    means = {k: np.mean([r[k] for r in fold_results])
             for k in ('rank1','eer','tar_001','tar_01')}
    stds  = {k: np.std( [r[k] for r in fold_results])
             for k in ('rank1','eer','tar_001','tar_01')}
    lines.append(f'mean,{means["rank1"]:.4f},{means["eer"]:.4f},'
                 f'{means["tar_001"]:.4f},{means["tar_01"]:.4f}')
    lines.append(f'std,{stds["rank1"]:.4f},{stds["eer"]:.4f},'
                 f'{stds["tar_001"]:.4f},{stds["tar_01"]:.4f}')
    Path(path).write_text('\n'.join(lines))
    print(f'  Results CSV → {path}')


def _save_spectrum_txt(spectrum_results, path):
    lines = ['spectrum,rank1,eer,tar_001,tar_01']
    for spec, r in spectrum_results.items():
        lines.append(f'{spec},{r["rank1"]:.4f},{r["eer"]:.4f},'
                     f'{r["tar_001"]:.4f},{r["tar_01"]:.4f}')
    Path(path).write_text('\n'.join(lines))
    print(f'  Results CSV → {path}')


# ══════════════════════════════════════════════════════════════════════
# 10.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='C-LMCL palmprint recognition on CASIA-MS dataset')
    parser.add_argument('--mode', default='both',
                        choices=['cross_subject', 'cross_spectrum', 'both'],
                        help='Evaluation protocol to run')
    parser.add_argument('--data_root', default=CFG['data_root'],
                        help='Path to CASIA-MS-ROI directory')
    parser.add_argument('--max_iter', type=int, default=CFG['max_iter'],
                        help='Gradient steps per training run')
    parser.add_argument('--n_folds', type=int, default=CFG['n_folds'],
                        help='Number of cross-subject folds')
    parser.add_argument('--batch_size', type=int, default=CFG['batch_size'])
    parser.add_argument('--out_dir', default=CFG['out_dir'])
    parser.add_argument('--smoke_test', action='store_true',
                        help='Quick run: 100 iterations, 1 fold (code check only)')
    args = parser.parse_args()

    # Apply CLI overrides
    CFG.update({
        'data_root'  : args.data_root,
        'max_iter'   : args.max_iter,
        'n_folds'    : args.n_folds,
        'batch_size' : args.batch_size,
        'out_dir'    : args.out_dir,
    })
    if args.smoke_test:
        CFG['max_iter'] = 100
        CFG['n_folds']  = 1
        print('[smoke-test] max_iter=100, n_folds=1')

    # ── Reproducibility ───────────────────────────────────────────────
    random.seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    torch.manual_seed(CFG['seed'])
    if CFG['device'] == 'cuda':
        torch.cuda.manual_seed_all(CFG['seed'])
        torch.backends.cudnn.benchmark = True

    os.makedirs(CFG['out_dir'], exist_ok=True)

    # ── Dataset summary ───────────────────────────────────────────────
    meta = CASIAMultispectralDataset(CFG['data_root'])
    print('\n' + '═'*60)
    print('  DATASET SUMMARY')
    print('═'*60)
    print(f'  Root           : {CFG["data_root"]}')
    print(f'  Total images   : {len(meta)}')
    print(f'  Identities     : {meta.num_classes}  (subjects × hands)')
    print(f'  Spectra        : {meta.all_spectra}')
    print(f'  Device         : {CFG["device"]}')
    print(f'  Output dir     : {CFG["out_dir"]}')
    print(f'  Max iterations : {CFG["max_iter"]}')
    print('═'*60)

    # ── Run protocols ─────────────────────────────────────────────────
    if args.mode in ('cross_subject', 'both'):
        run_cross_subject(CFG)

    if args.mode in ('cross_spectrum', 'both'):
        run_cross_spectrum(CFG)

    print('\nAll done.  Results in:', CFG['out_dir'])


if __name__ == '__main__':
    main()
