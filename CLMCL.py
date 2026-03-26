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
    Labels assigned in sorted order: 001_L=0, 001_R=1, 002_L=2 …

Protocols
---------
    cross_subject  — 80/20 identity split, 8-fold CV, all spectra pooled.
    cross_spectrum — Leave-one-spectrum-out. Same 200 IDs in train & test,
                     but held-out spectrum is unseen during training.

Logged metrics (every log_every iterations)
-------------------------------------------
    • Total loss  (L = L_lmc + λ·L_c)
    • LMCL loss   (L_lmc)
    • Center loss (L_c)
    • Train top-1 accuracy  (on the current mini-batch, closed-set)
    • Test  top-1 accuracy  (Rank-1, open-set, full test set)
    • Test  EER             (open-set verification)

All hyper-parameters are set directly from the paper (Sec. IV-B).

Usage
-----
    python clmcl_casia.py --mode both
    python clmcl_casia.py --mode cross_subject
    python clmcl_casia.py --mode cross_spectrum
    python clmcl_casia.py --smoke_test   # 200 iters, 1 fold — quick check
"""

import os, random, warnings
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

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION  —  all values from the paper (Sec. IV-B)
# ══════════════════════════════════════════════════════════════════════
CFG = {
    # ── Data ──────────────────────────────────────────────────────────
    'data_root'    : '/home/pai-ng/Jamal/CASIA-MS-ROI',
    'img_size'     : 224,        # paper: "ROI images resized to 224×224"

    # ── Network ───────────────────────────────────────────────────────
    'feat_dim'     : 128,        # paper: "128-dimensional feature embedding"

    # ── C-LMCL hyper-parameters  (paper Sec. IV-B-3) ─────────────────
    's'            : 30,         # feature-norm scale
    'm'            : 0.65,       # cosine margin (best on Tongji/PolyU)
    'lambda_c'     : 0.10,       # center-loss weight  λ
    'center_lr'    : 0.5,        # α  in eq. (6-7)

    # ── SGD optimiser  (paper Sec. IV-B-3) ───────────────────────────
    'batch_size'   : 55,         # paper: "batch size of 55 on a single GPU"
    'lr'           : 0.01,       # paper: "learning rate starts with 0.01"
    'momentum'     : 0.9,
    'weight_decay' : 5e-4,       # paper: "weight decay fixed to 0.0005"

    # ── LR schedule  (paper Sec. IV-B-3) ─────────────────────────────
    'max_iter'     : 30_000,     # paper: "finish training at 30k iterations"
    'lr_steps'     : [16_000, 24_000, 28_000],
    'lr_gamma'     : 0.1,        # paper: "divided by 10 at 16k, 24k, 28k"

    # ── Cross-subject protocol ────────────────────────────────────────
    'train_ratio'  : 0.80,       # paper: "80% of identities for training"
    'n_folds'      : 8,          # paper: "repeated 8 times"

    # ── Logging & evaluation cadence ─────────────────────────────────
    'log_every'    : 500,        # iterations between loss/acc log entries
    'eval_every'   : 2_000,      # iterations between full test evaluations

    # ── Misc ──────────────────────────────────────────────────────────
    'out_dir'      : './clmcl_results',
    'seed'         : 42,
    'num_workers'  : 4,
    'device'       : 'cuda' if torch.cuda.is_available() else 'cpu',
}


# ══════════════════════════════════════════════════════════════════════
# 1.  DATASET
# ══════════════════════════════════════════════════════════════════════
class CASIAMultispectralDataset(Dataset):
    """
    Flat-directory loader for CASIA-MS-ROI.
    Filename : {subject}_{hand}_{spectrum}_{iteration}.jpg
    Identity : (subject, hand)  →  integer class label
    """
    def __init__(self, data_root, transform=None,
                 allowed_spectra=None, allowed_ids=None):
        self.root      = Path(data_root)
        self.transform = transform
        self.samples   = []   # (path, class_id, spectrum, iteration)
        self._build_index(allowed_spectra, allowed_ids)

    def _build_index(self, allowed_spectra, allowed_ids):
        paths = sorted(self.root.glob('*.jpg'))
        if not paths:
            paths = sorted(self.root.glob('*.png'))
        assert paths, f"No images found in {self.root}"

        # Stable (subject, hand) → label mapping built from ALL files
        id_set = set()
        for p in paths:
            parts = p.stem.split('_')
            id_set.add((parts[0], parts[1]))
        sorted_ids        = sorted(id_set)
        self.class_to_idx = {f'{s}_{h}': i for i,(s,h) in enumerate(sorted_ids)}
        self.idx_to_class  = {v: k for k,v in self.class_to_idx.items()}
        self.num_classes   = len(self.class_to_idx)

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

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        path, class_id, spectrum, iteration = self.samples[idx]
        img = Image.open(path).convert('RGB')   # grayscale → 3-ch replication
        if self.transform:
            img = self.transform(img)
        return img, class_id, spectrum, iteration


# ── Subset helpers ────────────────────────────────────────────────────
class _MappedSubset(Dataset):
    """Subset with class-label remapping (used for training set)."""
    def __init__(self, base, indices, label_remap, transform):
        self.base = base; self.indices = indices
        self.remap = label_remap; self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        path, old_lbl, spec, it = self.base.samples[self.indices[i]]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.remap[old_lbl], spec, it


class _Subset(Dataset):
    """Subset without label remapping (used for test set)."""
    def __init__(self, base, indices, transform):
        self.base = base; self.indices = indices; self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        path, lbl, spec, it = self.base.samples[self.indices[i]]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, lbl, spec, it


# ── Transforms  (paper Sec. IV-B-1) ──────────────────────────────────
def get_transform(img_size=224, augment=True):
    """
    Paper augmentations (all applied with p=0.6):
      Rotation ±5° | Resized crop [240,256]→224 | Contrast α∈[0.5,1.5]
      Brightness β∈[−50,+50] | Smoothing/sharpening | Color shift ±25
    Normalisation: pixel/128 − 1  (paper: "subtracting 128, dividing by 128")
    """
    if augment:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(img_size, scale=(240/256, 1.0)),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=50/128, contrast=0.5,
                                   saturation=0.1, hue=25/255),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
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
# 2.  MODEL — ResNet-20  (Table I + Fig. 2 in paper)
# ══════════════════════════════════════════════════════════════════════
# Conv count: Stage1(3) + Stage2(5) + Stage3(9) + Stage4(3) = 20
class _Plain(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.seq(x)


class _Residual(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(ch, ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch))
    def forward(self, x): return F.relu(self.seq(x) + x)


class PalmResNet20(nn.Module):
    """
    ResNet-20 as per Table I.
    3×224×224 → Stage1(64)→Stage2(128)→Stage3(256)→Stage4(512)→GAP→FC(128)
    """
    def __init__(self, feat_dim=128):
        super().__init__()
        self.stage1 = nn.Sequential(_Plain(3, 64),   _Residual(64))
        self.stage2 = nn.Sequential(_Plain(64,128),  _Residual(128),  _Residual(128))
        self.stage3 = nn.Sequential(_Plain(128,256), _Residual(256),  _Residual(256),
                                    _Residual(256),  _Residual(256))
        self.stage4 = nn.Sequential(_Plain(256,512), _Residual(512))
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(512, feat_dim)
    def forward(self, x):
        for s in (self.stage1, self.stage2, self.stage3, self.stage4):
            x = s(x)
        return self.fc(self.gap(x).flatten(1))   # 128-dim raw embedding


# ══════════════════════════════════════════════════════════════════════
# 3.  LOSSES
# ══════════════════════════════════════════════════════════════════════
class LMCLLoss(nn.Module):
    """Large Margin Cosine Loss — eq. (4)."""
    def __init__(self, num_classes, feat_dim, s=30.0, m=0.65):
        super().__init__()
        self.s = s; self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, feat, labels):
        feat_n = F.normalize(feat,   p=2, dim=1)
        W_n    = F.normalize(self.W, p=2, dim=1)
        cosine = feat_n @ W_n.t()
        margin = torch.zeros_like(cosine).scatter_(
                     1, labels.unsqueeze(1), self.m)
        logits = self.s * (cosine - margin)
        loss   = F.cross_entropy(logits, labels)
        return loss, cosine.detach(), logits.detach()


class CenterLoss(nn.Module):
    """Center Loss — eq. (5)."""
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, labels):
        c = self.centers[labels]
        return 0.5 * ((feat - c) ** 2).sum(dim=1).mean()


class CLMCLoss(nn.Module):
    """Centralized LMCL — eq. (8):  L = L_lmc + λ·L_c"""
    def __init__(self, num_classes, feat_dim, s=30, m=0.65, lam=0.1):
        super().__init__()
        self.lmcl   = LMCLLoss(num_classes, feat_dim, s, m)
        self.center = CenterLoss(num_classes, feat_dim)
        self.lam    = lam

    def forward(self, feat, labels):
        l_lmc, cosine, logits = self.lmcl(feat, labels)
        l_c                   = self.center(feat, labels)
        total                 = l_lmc + self.lam * l_c
        return total, l_lmc, l_c, cosine, logits


# ══════════════════════════════════════════════════════════════════════
# 4.  METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════
def batch_top1(logits, labels):
    """Mini-batch closed-set top-1 accuracy (%)."""
    return (logits.argmax(dim=1) == labels).float().mean().item() * 100.0


def split_gallery_probe(iters):
    """First-half iterations → gallery; second-half → probe."""
    unique = sorted(set(iters))
    half   = max(1, len(unique) // 2)
    g_set  = set(unique[:half])
    p_set  = set(unique[half:]) if len(unique) > 1 else set(unique)
    g_mask = np.array([it in g_set for it in iters])
    p_mask = np.array([it in p_set for it in iters])
    return g_mask, p_mask


@torch.no_grad()
def extract_features(model, loader, device):
    """
    Feature = concat(f_orig, f_mirror), L2-normalised → 256-dim.
    Paper Sec. IV-B-4.
    """
    model.eval()
    feats, labels, spectra, iters = [], [], [], []
    for imgs, lbls, spec, it in loader:
        imgs = imgs.to(device, non_blocking=True)
        f    = F.normalize(
                   torch.cat([model(imgs), model(torch.flip(imgs, [-1]))], dim=1),
                   p=2, dim=1)
        feats.append(f.cpu().numpy())
        labels.extend(lbls.numpy().tolist())
        spectra.extend(list(spec)); iters.extend(list(it))
    model.train()
    return (np.vstack(feats), np.array(labels, dtype=np.int64),
            np.array(spectra), np.array(iters))


def rank1_acc(g_feats, g_labels, p_feats, p_labels):
    pred = g_labels[(p_feats @ g_feats.T).argmax(axis=1)]
    return (pred == p_labels).mean() * 100.0


def compute_verification_metrics(g_feats, g_labels, p_feats, p_labels):
    """Returns EER (%), TAR@FAR=0.01%, TAR@FAR=0.1%, fpr, tpr."""
    sim    = (p_feats @ g_feats.T).ravel()
    n_p, n_g = len(p_labels), len(g_labels)
    y_true = (np.repeat(p_labels, n_g) == np.tile(g_labels, n_p)).astype(np.int32)
    fpr, tpr, _ = roc_curve(y_true, sim, pos_label=1)
    fnr  = 1.0 - tpr
    idx  = np.abs(fpr - fnr).argmin()
    eer  = float((fpr[idx] + fnr[idx]) / 2.0 * 100.0)
    tar_001 = float(np.interp(1e-4, fpr, tpr)) * 100.0
    tar_01  = float(np.interp(1e-3, fpr, tpr)) * 100.0
    return eer, tar_001, tar_01, fpr, tpr


def test_metrics(model, test_ds, device, batch_size=128):
    """Quick evaluation during training: returns (rank1%, EER%)."""
    loader = make_loader(test_ds, batch_size, shuffle=False,
                          num_workers=CFG['num_workers'])
    feats, labels, _, iters = extract_features(model, loader, device)
    g_mask, p_mask = split_gallery_probe(iters)
    g_f, g_l = feats[g_mask], labels[g_mask]
    p_f, p_l = feats[p_mask], labels[p_mask]
    r1           = rank1_acc(g_f, g_l, p_f, p_l)
    eer, _, _, _, _ = compute_verification_metrics(g_f, g_l, p_f, p_l)
    return r1, eer


# ══════════════════════════════════════════════════════════════════════
# 5.  METRICS TRACKER
# ══════════════════════════════════════════════════════════════════════
class MetricsTracker:
    """Accumulates per-iteration scalars and streams them to a CSV log."""
    def __init__(self, log_path):
        self.iters        = []
        self.total_loss   = []
        self.lmcl_loss    = []
        self.center_loss  = []
        self.train_acc    = []
        self.eval_iters   = []
        self.test_rank1   = []
        self.test_eer     = []
        self._f = open(log_path, 'w', buffering=1)
        self._f.write('iter,total_loss,lmcl_loss,center_loss,train_acc\n')

    def log_train(self, it, total, lmcl, center, acc):
        self.iters.append(it);       self.total_loss.append(total)
        self.lmcl_loss.append(lmcl); self.center_loss.append(center)
        self.train_acc.append(acc)
        self._f.write(f'{it},{total:.6f},{lmcl:.6f},{center:.6f},{acc:.4f}\n')

    def log_eval(self, it, rank1, eer):
        self.eval_iters.append(it)
        self.test_rank1.append(rank1); self.test_eer.append(eer)
        self._f.write(f'# EVAL  iter={it}  rank1={rank1:.2f}%  eer={eer:.4f}%\n')

    def close(self): self._f.close()


# ══════════════════════════════════════════════════════════════════════
# 6.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════
def train_model(model, criterion, optimizer, center_opt,
                train_loader, test_ds,
                device, max_iter, lr_steps, lr_gamma,
                log_every, eval_every, out_dir, tag):

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_steps, gamma=lr_gamma)

    tracker = MetricsTracker(os.path.join(out_dir, f'metrics_{tag}.csv'))

    # Rolling buffers (reset every log_every steps for smooth averages)
    buf = dict(total=[], lmcl=[], center=[], acc=[])

    model.train()
    it = 0

    while it < max_iter:
        for batch in train_loader:
            if it >= max_iter:
                break

            imgs, labels, _, _ = batch
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            center_opt.zero_grad()

            feat  = model(imgs)
            total, l_lmc, l_c, _, logits = criterion(feat, labels)
            total.backward()

            optimizer.step()
            # Center gradient scaled by 1/(1+N) per Wen et al. eq.(6-7)
            for p in criterion.center.parameters():
                if p.grad is not None:
                    p.grad.data *= 1.0 / (1 + imgs.size(0))
            center_opt.step()
            scheduler.step()

            it += 1
            buf['total'].append(total.item())
            buf['lmcl'].append(l_lmc.item())
            buf['center'].append(l_c.item())
            buf['acc'].append(batch_top1(logits, labels))

            # ── Log training metrics ───────────────────────────────
            if it % log_every == 0 or it == 1:
                w    = min(log_every, len(buf['total']))
                avgs = {k: np.mean(buf[k][-w:]) for k in buf}
                tracker.log_train(it, avgs['total'], avgs['lmcl'],
                                   avgs['center'], avgs['acc'])
                lr = optimizer.param_groups[0]['lr']
                print(f'[{tag}] iter {it:6d}/{max_iter} | '
                      f'Loss={avgs["total"]:.4f} '
                      f'(LMCL={avgs["lmcl"]:.4f} Ctr={avgs["center"]:.4f}) | '
                      f'TrainAcc={avgs["acc"]:.2f}%  LR={lr:.2e}')

            # ── Evaluate on test set ───────────────────────────────
            if (it % eval_every == 0 or it == max_iter) and test_ds is not None:
                r1, eer = test_metrics(model, test_ds, device)
                tracker.log_eval(it, r1, eer)
                print(f'  ↳ [TEST] iter={it:6d} | '
                      f'Rank-1={r1:.2f}%  EER={eer:.4f}%')
                model.train()

    tracker.close()
    return model, tracker


# ══════════════════════════════════════════════════════════════════════
# 7.  FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════
def evaluate_final(model, test_ds, device, tag=''):
    loader = make_loader(test_ds, 128, shuffle=False,
                          num_workers=CFG['num_workers'])
    feats, labels, _, iters = extract_features(model, loader, device)
    g_mask, p_mask = split_gallery_probe(iters)
    g_f, g_l = feats[g_mask], labels[g_mask]
    p_f, p_l = feats[p_mask], labels[p_mask]

    r1                   = rank1_acc(g_f, g_l, p_f, p_l)
    eer, t001, t01, fpr, tpr = compute_verification_metrics(g_f, g_l, p_f, p_l)

    sep = '─' * 62
    print(f'\n{sep}')
    print(f'  {tag}')
    print(f'  Gallery: {g_mask.sum()} imgs  |  Probe: {p_mask.sum()} imgs')
    print(f'  Rank-1 Identification  :  {r1:.4f} %')
    print(f'  EER                    :  {eer:.4f} %')
    print(f'  TAR @ FAR = 0.01 %     :  {t001:.4f} %')
    print(f'  TAR @ FAR = 0.1  %     :  {t01:.4f} %')
    print(f'{sep}\n')

    return dict(rank1=r1, eer=eer, tar_001=t001, tar_01=t01, fpr=fpr, tpr=tpr)


# ══════════════════════════════════════════════════════════════════════
# 8.  PLOTTING
# ══════════════════════════════════════════════════════════════════════
def plot_training_curves(tracker, out_dir, tag):
    """
    4-panel figure saved as PDF:
      (a) Loss curves (Total / LMCL / Center) vs. iteration
      (b) Training top-1 accuracy vs. iteration
      (c) Test Rank-1 accuracy vs. iteration
      (d) Test EER vs. iteration
    Red dashed verticals mark LR drop milestones.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f'Training Curves — {tag}', fontsize=13, fontweight='bold')

    iters = tracker.iters
    ei    = tracker.eval_iters

    # (a) Loss
    ax = axes[0, 0]
    ax.plot(iters, tracker.total_loss,  lw=1.6, label='Total loss  $L$')
    ax.plot(iters, tracker.lmcl_loss,   lw=1.2, ls='--', label='LMCL loss  $L_{lmc}$')
    ax.plot(iters, tracker.center_loss, lw=1.2, ls=':',  label='Center loss  $L_c$')
    for step in CFG['lr_steps']:
        ax.axvline(step, color='red', ls='--', lw=0.8, alpha=0.55)
    ax.text(CFG['lr_steps'][0]+50, ax.get_ylim()[0]*1.05,
            'LR÷10', color='red', fontsize=7)
    ax.set_title('(a) Loss'); ax.set_xlabel('Iteration'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (b) Train accuracy
    ax = axes[0, 1]
    ax.plot(iters, tracker.train_acc, color='steelblue', lw=1.6,
            label='Train top-1 acc (batch)')
    for step in CFG['lr_steps']:
        ax.axvline(step, color='red', ls='--', lw=0.8, alpha=0.55)
    ax.set_ylim(0, 105)
    ax.set_title('(b) Training Accuracy'); ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy (%)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (c) Test Rank-1
    ax = axes[1, 0]
    if ei:
        ax.plot(ei, tracker.test_rank1, 'o-', color='darkorange', lw=1.6,
                ms=5, label='Test Rank-1 (%)')
    ax.set_ylim(0, 105)
    ax.set_title('(c) Test Rank-1 Accuracy'); ax.set_xlabel('Iteration')
    ax.set_ylabel('Rank-1 (%)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (d) Test EER
    ax = axes[1, 1]
    if ei:
        ax.plot(ei, tracker.test_eer, 'D-', color='crimson', lw=1.6,
                ms=5, label='Test EER (%)')
    ax.set_title('(d) Test EER'); ax.set_xlabel('Iteration')
    ax.set_ylabel('EER (%)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f'training_curves_{tag}.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Training curves → {path}')


def plot_roc_folds(fold_results, out_dir):
    plt.figure(figsize=(7, 6))
    max_far  = 0.04
    mean_fpr = np.linspace(0, max_far, 1000)
    tprs_i   = []
    for i, r in enumerate(fold_results):
        ti = np.interp(mean_fpr, r['fpr'], r['tpr'])
        tprs_i.append(ti)
        plt.plot(mean_fpr*100, (1-ti)*100, lw=0.8, alpha=0.35,
                 label=f'Fold {i+1}  EER={r["eer"]:.3f}%')
    mean_tpr = np.mean(tprs_i, axis=0)
    mean_eer = np.mean([r['eer'] for r in fold_results])
    plt.plot(mean_fpr*100, (1-mean_tpr)*100, 'k-', lw=2.5,
             label=f'Mean  EER={mean_eer:.3f}%')
    plt.plot([0, max_far*100], [0, max_far*100], 'r--', lw=1, label='EER line')
    plt.xlim(0, max_far*100); plt.ylim(0, max_far*100)
    plt.xlabel('FAR (%)'); plt.ylabel('FRR (%)')
    plt.title('Cross-Subject — ROC Curves'); plt.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(out_dir, 'roc_cross_subject.pdf')
    plt.savefig(path, dpi=200); plt.close()
    print(f'  ROC curve → {path}')


def plot_roc_spectra(spectrum_results, out_dir):
    plt.figure(figsize=(7, 6))
    max_far  = 0.04
    fpr_grid = np.linspace(0, max_far, 1000)
    for spec, r in spectrum_results.items():
        ti = np.interp(fpr_grid, r['fpr'], r['tpr'])
        plt.plot(fpr_grid*100, (1-ti)*100, lw=1.5,
                 label=f'Spec {spec}  EER={r["eer"]:.3f}%')
    plt.plot([0, max_far*100], [0, max_far*100], 'k--', lw=1, label='EER line')
    plt.xlim(0, max_far*100); plt.ylim(0, max_far*100)
    plt.xlabel('FAR (%)'); plt.ylabel('FRR (%)')
    plt.title('Cross-Spectrum — ROC Curves'); plt.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, 'roc_cross_spectrum.pdf')
    plt.savefig(path, dpi=200); plt.close()
    print(f'  ROC curve → {path}')


# ══════════════════════════════════════════════════════════════════════
# 9.  RESULT SAVE HELPERS
# ══════════════════════════════════════════════════════════════════════
def save_fold_results(fold_results, path):
    lines = ['fold,rank1,eer,tar_001,tar_01']
    for i, r in enumerate(fold_results):
        lines.append(f'{i+1},{r["rank1"]:.4f},{r["eer"]:.4f},'
                     f'{r["tar_001"]:.4f},{r["tar_01"]:.4f}')
    m = {k: np.mean([r[k] for r in fold_results])
         for k in ('rank1','eer','tar_001','tar_01')}
    s = {k: np.std ([r[k] for r in fold_results])
         for k in ('rank1','eer','tar_001','tar_01')}
    lines += [f'mean,{m["rank1"]:.4f},{m["eer"]:.4f},{m["tar_001"]:.4f},{m["tar_01"]:.4f}',
              f'std,{s["rank1"]:.4f},{s["eer"]:.4f},{s["tar_001"]:.4f},{s["tar_01"]:.4f}']
    Path(path).write_text('\n'.join(lines))
    print(f'  Results CSV → {path}')


def save_spectrum_results(spectrum_results, path):
    lines = ['spectrum,rank1,eer,tar_001,tar_01']
    for spec, r in spectrum_results.items():
        lines.append(f'{spec},{r["rank1"]:.4f},{r["eer"]:.4f},'
                     f'{r["tar_001"]:.4f},{r["tar_01"]:.4f}')
    Path(path).write_text('\n'.join(lines))
    print(f'  Results CSV → {path}')


# ══════════════════════════════════════════════════════════════════════
# 10.  PROTOCOL A — CROSS-SUBJECT
# ══════════════════════════════════════════════════════════════════════
def run_cross_subject(cfg):
    print('\n' + '═'*62)
    print('  PROTOCOL 1 — CROSS-SUBJECT')
    print('═'*62)

    device  = cfg['device']
    out_dir = os.path.join(cfg['out_dir'], 'cross_subject')
    os.makedirs(out_dir, exist_ok=True)

    full_ds = CASIAMultispectralDataset(cfg['data_root'])
    ids     = list(range(full_ds.num_classes))
    fold_results = []

    for fold in range(cfg['n_folds']):
        print(f'\n── Fold {fold+1} / {cfg["n_folds"]} ──────────────────────────────')

        random.shuffle(ids)
        n_train   = int(len(ids) * cfg['train_ratio'])
        train_ids = set(ids[:n_train])
        test_ids  = set(ids[n_train:])

        tr_idx = [i for i,(_, c, _, _) in enumerate(full_ds.samples) if c in train_ids]
        te_idx = [i for i,(_, c, _, _) in enumerate(full_ds.samples) if c in test_ids]
        id_remap = {old: new for new,old in enumerate(sorted(train_ids))}

        train_ds = _MappedSubset(full_ds, tr_idx, id_remap,
                                  get_transform(cfg['img_size'], augment=True))
        test_ds  = _Subset(full_ds, te_idx,
                            get_transform(cfg['img_size'], augment=False))

        print(f'  Train: {len(train_ds)} imgs / {len(train_ids)} IDs  |  '
              f'Test: {len(test_ds)} imgs / {len(test_ids)} IDs')

        loader = make_loader(train_ds, cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'])

        model     = PalmResNet20(feat_dim=cfg['feat_dim']).to(device)
        criterion = CLMCLoss(len(train_ids), cfg['feat_dim'],
                              cfg['s'], cfg['m'], cfg['lambda_c']).to(device)
        optimizer  = torch.optim.SGD(
            list(model.parameters()) + list(criterion.lmcl.parameters()),
            lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
        center_opt = torch.optim.SGD(
            criterion.center.parameters(), lr=cfg['center_lr'])

        model, tracker = train_model(
            model, criterion, optimizer, center_opt,
            loader, test_ds, device,
            cfg['max_iter'], cfg['lr_steps'], cfg['lr_gamma'],
            cfg['log_every'], cfg['eval_every'],
            out_dir, tag=f'fold{fold+1}')

        plot_training_curves(tracker, out_dir, tag=f'fold{fold+1}')
        torch.save(model.state_dict(),
                   os.path.join(out_dir, f'model_fold{fold+1}.pth'))

        result = evaluate_final(model, test_ds, device,
                                 tag=f'Cross-Subject  Fold {fold+1}')
        fold_results.append(result)

    # Aggregate
    print('\n' + '═'*62)
    print('  CROSS-SUBJECT — AGGREGATE  (mean ± std over folds)')
    print(f'  {"Metric":24s}  {"Mean":>10}  {"Std":>10}')
    print('  ' + '─'*48)
    for k, lbl in [('rank1','Rank-1 (%)'), ('eer','EER (%)'),
                   ('tar_001','TAR@FAR=0.01% (%)'), ('tar_01','TAR@FAR=0.1% (%)')]:
        vals = [r[k] for r in fold_results]
        print(f'  {lbl:24s}  {np.mean(vals):10.4f}  {np.std(vals):10.4f}')

    plot_roc_folds(fold_results, out_dir)
    save_fold_results(fold_results,
                      os.path.join(out_dir, 'results_cross_subject.csv'))
    return fold_results


# ══════════════════════════════════════════════════════════════════════
# 11.  PROTOCOL B — CROSS-SPECTRUM
# ══════════════════════════════════════════════════════════════════════
def run_cross_spectrum(cfg):
    print('\n' + '═'*62)
    print('  PROTOCOL 2 — CROSS-SPECTRUM  (leave-one-spectrum-out)')
    print('═'*62)

    device  = cfg['device']
    out_dir = os.path.join(cfg['out_dir'], 'cross_spectrum')
    os.makedirs(out_dir, exist_ok=True)

    meta_ds     = CASIAMultispectralDataset(cfg['data_root'])
    all_spectra = sorted(meta_ds.all_spectra)
    num_classes  = meta_ds.num_classes
    print(f'\n  Spectra found : {all_spectra}')
    print(f'  Total classes : {num_classes}')

    spectrum_results = {}

    for test_spec in all_spectra:
        train_spectra = [s for s in all_spectra if s != test_spec]
        print(f'\n── Hold-out: {test_spec}  |  Train on: {train_spectra} ──')

        train_ds = CASIAMultispectralDataset(
            cfg['data_root'],
            transform=get_transform(cfg['img_size'], augment=True),
            allowed_spectra=train_spectra)

        test_ds = CASIAMultispectralDataset(
            cfg['data_root'],
            transform=get_transform(cfg['img_size'], augment=False),
            allowed_spectra=[test_spec])

        print(f'  Train: {len(train_ds)} imgs  |  Test: {len(test_ds)} imgs')

        loader = make_loader(train_ds, cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'])

        model     = PalmResNet20(feat_dim=cfg['feat_dim']).to(device)
        criterion = CLMCLoss(num_classes, cfg['feat_dim'],
                              cfg['s'], cfg['m'], cfg['lambda_c']).to(device)
        optimizer  = torch.optim.SGD(
            list(model.parameters()) + list(criterion.lmcl.parameters()),
            lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
        center_opt = torch.optim.SGD(
            criterion.center.parameters(), lr=cfg['center_lr'])

        model, tracker = train_model(
            model, criterion, optimizer, center_opt,
            loader, test_ds, device,
            cfg['max_iter'], cfg['lr_steps'], cfg['lr_gamma'],
            cfg['log_every'], cfg['eval_every'],
            out_dir, tag=f'spec_{test_spec}')

        plot_training_curves(tracker, out_dir, tag=f'spec_{test_spec}')
        torch.save(model.state_dict(),
                   os.path.join(out_dir, f'model_spec_{test_spec}.pth'))

        result = evaluate_final(model, test_ds, device,
                                 tag=f'Cross-Spectrum  test={test_spec}')
        spectrum_results[test_spec] = result

    print('\n' + '═'*62)
    print('  CROSS-SPECTRUM — RESULTS PER HELD-OUT SPECTRUM')
    hdr = (f'  {"Spectrum":>10}  {"Rank-1":>8}  {"EER":>8}  '
           f'{"TAR@0.01%":>10}  {"TAR@0.1%":>9}')
    print(hdr); print('  ' + '─'*(len(hdr)-2))
    for spec, r in spectrum_results.items():
        print(f'  {spec:>10}  {r["rank1"]:8.4f}  {r["eer"]:8.4f}'
              f'  {r["tar_001"]:10.4f}  {r["tar_01"]:9.4f}')

    plot_roc_spectra(spectrum_results, out_dir)
    save_spectrum_results(spectrum_results,
                          os.path.join(out_dir, 'results_cross_spectrum.csv'))
    return spectrum_results


# ══════════════════════════════════════════════════════════════════════
# 12.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='C-LMCL on CASIA-MS palmprint dataset')
    parser.add_argument('--mode', default='both',
                        choices=['cross_subject', 'cross_spectrum', 'both'])
    parser.add_argument('--data_root',  default=CFG['data_root'])
    parser.add_argument('--max_iter',   type=int, default=CFG['max_iter'])
    parser.add_argument('--n_folds',    type=int, default=CFG['n_folds'])
    parser.add_argument('--batch_size', type=int, default=CFG['batch_size'])
    parser.add_argument('--out_dir',    default=CFG['out_dir'])
    parser.add_argument('--log_every',  type=int, default=CFG['log_every'])
    parser.add_argument('--eval_every', type=int, default=CFG['eval_every'])
    parser.add_argument('--smoke_test', action='store_true',
                        help='200 iters, 1 fold, eval every 100 (quick check)')
    args = parser.parse_args()

    CFG.update({k: getattr(args, k)
                for k in ('data_root','max_iter','n_folds','batch_size',
                           'out_dir','log_every','eval_every')})
    if args.smoke_test:
        CFG.update(max_iter=200, n_folds=1, log_every=50, eval_every=100)
        print('[smoke-test]  max_iter=200  n_folds=1  eval_every=100')

    random.seed(CFG['seed']); np.random.seed(CFG['seed'])
    torch.manual_seed(CFG['seed'])
    if CFG['device'] == 'cuda':
        torch.cuda.manual_seed_all(CFG['seed'])
        torch.backends.cudnn.benchmark = True

    os.makedirs(CFG['out_dir'], exist_ok=True)

    meta = CASIAMultispectralDataset(CFG['data_root'])
    print('\n' + '═'*62)
    print('  DATASET  &  HYPER-PARAMETER SUMMARY  (paper values)')
    print('═'*62)
    print(f'  Root              : {CFG["data_root"]}')
    print(f'  Total images      : {len(meta)}')
    print(f'  Identities        : {meta.num_classes}  (subjects × hands)')
    print(f'  Spectra           : {meta.all_spectra}')
    print(f'  Device            : {CFG["device"]}')
    print()
    print(f'  ── Model (Table I) ────────────────────────────────')
    print(f'  Backbone          : ResNet-20')
    print(f'  Feature dim       : {CFG["feat_dim"]}  (FC1 output)')
    print()
    print(f'  ── C-LMCL (Sec. IV-B-3) ──────────────────────────')
    print(f'  Scale  s          : {CFG["s"]}')
    print(f'  Margin m          : {CFG["m"]}')
    print(f'  Lambda λ          : {CFG["lambda_c"]}')
    print(f'  Center LR α       : {CFG["center_lr"]}')
    print()
    print(f'  ── SGD (Sec. IV-B-3) ──────────────────────────────')
    print(f'  Batch size        : {CFG["batch_size"]}')
    print(f'  Initial LR        : {CFG["lr"]}')
    print(f'  Momentum          : {CFG["momentum"]}')
    print(f'  Weight decay      : {CFG["weight_decay"]}')
    print(f'  Max iterations    : {CFG["max_iter"]}')
    print(f'  LR drop at        : {CFG["lr_steps"]}  (×{CFG["lr_gamma"]})')
    print()
    print(f'  ── Logging ────────────────────────────────────────')
    print(f'  Log loss/acc every: {CFG["log_every"]} iters')
    print(f'  Eval test set every: {CFG["eval_every"]} iters')
    print('═'*62)

    if args.mode in ('cross_subject', 'both'):
        run_cross_subject(CFG)

    if args.mode in ('cross_spectrum', 'both'):
        run_cross_spectrum(CFG)

    print('\nAll done.  Results in:', CFG['out_dir'])


if __name__ == '__main__':
    main()
