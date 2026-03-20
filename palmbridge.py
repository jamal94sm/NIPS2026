"""
palmbridge_single.py — PalmBridge: A Plug-and-Play Feature Alignment
Framework for Open-Set Palmprint Verification (Zhang et al., 2026)

Single-file implementation · CompNet backbone · CASIA-MS ROI dataset
Filename convention: {subject}_{hand}_{spectrum}_{iteration}.jpg
  e.g.  001_L_460_1.jpg

To run: python palmbridge_single.py
All settings are in the PARAMETERS block immediately below the imports.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# Standard library
# ═══════════════════════════════════════════════════════════════════════════
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# Third-party
# ═══════════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          PARAMETERS                                       ║
# ║  Edit everything in this block — nowhere else needs to be touched.        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = "/home/pai-ng/Jamal/CASIA-MS-ROI"  # folder containing all ROI .jpg files
SPECTRA     = ["460", "630", "700", "850", "940", "WHT"]  # spectral channels to include
IMG_SIZE    = 128                   # ROIs are resized to (IMG_SIZE × IMG_SIZE)

# ── Protocol selection ────────────────────────────────────────────────────────
# Choose one of: "intra" | "cross" | "closed" | "all"
PROTOCOL    = "all"

# ── Protocol-specific split settings (§IV-A-6) ───────────────────────────────
#
# INTRA-DATASET OPEN-SET
#   All identities split 50/50 into two non-overlapping groups (no ID overlap).
#   Group 1 (training IDs) : images split 80 % train / 20 % internal-test.
#   Group 2 (open-set IDs) : images split 50 % gallery / 50 % query.
INTRA_TRAIN_ID_RATIO     = 0.5   # fraction of identities → training group
INTRA_TRAIN_IMG_RATIO    = 0.8   # within training group: train / internal-test

#
# CROSS-DATASET OPEN-SET  (adapted for CASIA-MS single-dataset setup)
#   Train  : ALL 4 spectral domains, subject IDs  1 – 150  (indices 0-149)
#   Eval   : 2 target spectral domains, subject IDs 151 – 200 (indices 150-199)
#   Gallery / Query split: 50 % / 50 % of each eval identity's images.
CROSS_TRAIN_SPECTRA      = ["460", "630", "WHT", "940"]  # all 4 source domains
CROSS_TEST_SPECTRA       = ["700", "850"]                # 2 target domains
# CROSS_TRAIN_ID_RATIO: fraction of identity keys used for training.
# Using a ratio (not a hardcoded subject-index threshold) makes this robust
# regardless of how subjects are numbered in the dataset.
# 0.75 = first 75% of sorted identity keys → train,  last 25% → eval.
CROSS_TRAIN_ID_RATIO     = 0.75

#
# CLOSED-SET
#   All 200 identities included (no cross-dataset separation).
#   First 80 % of each identity's images → training.
#   Remaining 20 % split 50 % gallery / 50 % query.
CLOSED_TRAIN_IMG_RATIO   = 0.8   # 80 % of images/identity → training

# ── Feature blending  (Eq. 4, Fig. 4 optimal) ────────────────────────────────
W_MAP       = 0.3   # weight of the mapped PalmBridge vector  z̃
W_ORI       = 0.7   # weight of the original backbone vector  z

# ── Blending coefficient sweep  (Fig. 4) ─────────────────────────────────────
RUN_SWEEP   = False  # set True to reproduce Fig. 4
SWEEP_STEPS = 10     # number of (w_map : w_ori) operating points

# ── Plug-and-play mode  (Table XV) ───────────────────────────────────────────
# Set PLUG_AND_PLAY = True and point PB_CKPT to a saved checkpoint to inject
# a trained PalmBridge codebook into a fresh (untrained) backbone.
PLUG_AND_PLAY = False
PB_CKPT       = None   # e.g. "checkpoints/best_intra.pt"

# ── Model architecture ────────────────────────────────────────────────────────
FEATURE_DIM       = 512   # D — backbone output / embedding dimension
NUM_PB_VECTORS    = 512   # K — PalmBridge codebook size (Table XIV best)
NUM_GABOR_FILTERS = 32    # orientations in the learnable Gabor bank
GABOR_KERNEL_SIZE = 15    # spatial size of each Gabor kernel (px)

# ── Loss weights  (Eq. 8) ────────────────────────────────────────────────────
# WHY ALPHA=1, BETA=1:  The codebook is pre-aligned during warmup (L_con is
# trained from epoch 1 even though z_hat is not used for ArcFace yet).  By the
# time PalmBridge activates, L_con is already near-zero — no amplification
# needed.  ALPHA=10 caused a sudden loss spike at epoch 21 (8.18 vs 0.26)
# because the misaligned codebook produced L_con~0.79 × 10 = 7.9 added to bak.
ALPHA      = 0.1    # weight of L_con  (codebook pre-aligned → no amplification needed)
BETA       = 1.0    # weight of L_o    (orthogonality regularisation)
LAMBDA_CON = 0.25   # λ inside L_con   (Eq. 6, §III-C-1)

# ── ArcFace  (L_bak) ─────────────────────────────────────────────────────────
# WHY s=32, m=0.2:  s=64/m=0.5 inflates initial loss to ~18 (random features
# give ln(C) ≈ 4.6 normally).  This makes gradients huge and unstable before
# the backbone has learned anything useful.  s=32, m=0.2 keeps initial loss
# near ln(100)=4.6, allowing proper convergence.  Once the backbone is warm,
# ArcFace naturally becomes discriminative even with a softer margin.
ARC_S      = 48.0   # feature scale  (32 collapses bak too fast; 48 keeps it meaningful)
ARC_M      = 0.40   # angular margin (reduced from 0.50 — easier to optimise)

# ── Training  (§IV-A-4) ──────────────────────────────────────────────────────
BATCH_SIZE   = 16
LR           = 1e-3
WEIGHT_DECAY = 5e-4
EPOCHS       = 100
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS  = 4

# ── PalmBridge warm-up ────────────────────────────────────────────────────────
# WHY warm-up:  PalmBridge's argmin is non-differentiable.  At epoch 1 the
# codebook is random, so z_tilde maps all identities to shared vectors,
# poisoning the ArcFace gradient before the backbone has learned anything.
# Solution: train backbone alone for WARMUP_EPOCHS (w_map=0), then activate
# PalmBridge with the full blending weight.  This is the two-phase strategy
# implied by the paper's joint optimisation once features are meaningful.
WARMUP_EPOCHS = 5   # epochs to train backbone only (w_map forced to 0)

# ── Evaluation ────────────────────────────────────────────────────────────────
N_EER_THRESHOLDS = 2000   # resolution of the FAR / FRR sweep

# ── Misc ─────────────────────────────────────────────────────────────────────
SEED         = 42
SAVE_DIR     = "checkpoints"
PLOT_DIR     = "plots"
LOG_EPOCHS   = 5           # evaluate and print EER/ACC every N epochs

# ╚═══════════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. DATASET                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _parse_filename(fname: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parses  {subject}_{hand}_{spectrum}_{iteration}.jpg
    Returns (subject, hand, spectrum, iteration) or None on mismatch.
    """
    stem  = Path(fname).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def build_index(
    data_root: str,
    spectra:   List[str],
) -> Dict[str, List[str]]:
    """
    Scans data_root (recursively) for image files matching the naming
    convention {subject}_{hand}_{spectrum}_{iteration}.<ext>.

    Handles:
      - Flat folder or nested subdirectories  (rglob, not glob)
      - Case-insensitive extensions           (.jpg / .JPG / .jpeg / .png / .bmp)
      - Clear diagnostic output if no files are found

    Returns  { "{subject}_{hand}" : [sorted absolute file paths] }
    Left and right palms are treated as separate identities.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"DATA_ROOT does not exist: {data_root!r}\n"
            f"  Please set DATA_ROOT at the top of the script to the folder "
            f"containing your CASIA-MS ROI images."
        )

    # Collect all image files (recursive, case-insensitive extension)
    EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    all_files = [
        f for f in sorted(root.rglob("*"))
        if f.is_file() and f.suffix.lower() in EXTS
    ]

    if not all_files:
        raise RuntimeError(
            f"No image files found under {data_root!r}\n"
            f"  Searched extensions: {sorted(EXTS)}\n"
            f"  Check that DATA_ROOT points to the correct folder and that "
            f"image files are present (including in subdirectories)."
        )

    index: Dict[str, List[str]] = defaultdict(list)
    n_skipped_parse   = 0
    n_skipped_spectrum = 0

    for f in all_files:
        parsed = _parse_filename(f.name)
        if parsed is None:
            n_skipped_parse += 1
            continue
        subject, hand, spectrum, _ = parsed
        if spectrum not in spectra:
            n_skipped_spectrum += 1
            continue
        index[f"{subject}_{hand}"].append(str(f))

    # ── Diagnostic summary ────────────────────────────────────────────────
    total_matched = sum(len(v) for v in index.values())
    print(
        f"[build_index] root='{data_root}'  spectra={spectra}\n"
        f"  Total files found   : {len(all_files)}\n"
        f"  Matched (parsed+spectrum): {total_matched}\n"
        f"  Skipped (parse fail): {n_skipped_parse}  "
        f"Skipped (wrong spectrum): {n_skipped_spectrum}\n"
        f"  Unique identity keys: {len(index)}"
    )

    if not index:
        # Show a sample of actual filenames to help the user debug
        samples = [f.name for f in all_files[:8]]
        raise RuntimeError(
            f"build_index found {len(all_files)} image files but NONE matched "
            f"the expected pattern  {{subject}}_{{hand}}_{{spectrum}}_{{iteration}}.<ext>\n"
            f"  Sample filenames found:\n"
            + "\n".join(f"    {s}" for s in samples) +
            f"\n  Expected spectra: {spectra}\n"
            f"  Please verify DATA_ROOT and filename convention."
        )

    return {k: sorted(v) for k, v in index.items()}


class PalmprintDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")     # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transform(train: bool) -> transforms.Compose:
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.02, 0.02),
                                    scale=(0.95, 1.05)),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),    # → [-1, 1]
    ]
    return transforms.Compose(ops)


def make_loader(
    samples: List[Tuple[str, int]],
    train:   bool = False,
    shuffle: bool = False,
) -> DataLoader:
    if len(samples) == 0:
        raise RuntimeError(
            "make_loader received an empty sample list.\n"
            "  This usually means build_index found no files, or the "
            "protocol split left one partition empty.\n"
            "  Check: (1) DATA_ROOT path, (2) filename convention, "
            "(3) SPECTRA list matches your actual filenames."
        )
    # Clamp batch size so it never exceeds dataset size (avoids drop_last
    # removing all samples in tiny splits during debugging).
    effective_bs = min(BATCH_SIZE, len(samples))
    dataset = PalmprintDataset(samples, transform=get_transform(train))
    return DataLoader(
        dataset,
        batch_size  = effective_bs,
        shuffle     = shuffle,
        num_workers = NUM_WORKERS,
        pin_memory  = (DEVICE == "cuda"),
        drop_last   = train and len(samples) > effective_bs,
    )


# ── Protocol 1: Intra-Dataset Open-Set  (§IV-A-6, Tables I–IV) ─────────────

def build_intra_dataset_splits() -> Tuple[List, List, List, List]:
    """
    Paper §IV-A-6 — Intra-dataset open-set evaluation:

    "For each dataset, all identities are evenly divided into two groups
     with no identity overlap.
     In the first group, images of each identity are further split into
     training and testing subsets for model optimization and internal
     evaluation.
     In the second group, images are equally partitioned into gallery
     and query subsets for open-set verification."

    Concretely for CASIA-MS:
      • All identity keys (subject_hand) are shuffled and split 50 / 50
        into Group A (training) and Group B (open-set) — zero overlap.
      • Group A: 80 % of each identity's images → training,
                 20 % → internal test/val (not used in open-set scoring).
      • Group B: 50 % of each identity's images → gallery,
                 50 % → query          (used for open-set EER / ACC).

    Returns
    -------
    train_samples, val_samples, gallery_samples, query_samples
    """
    rng   = np.random.default_rng(SEED)
    index = build_index(DATA_ROOT, SPECTRA)
    ids   = sorted(index.keys())

    # ── Step 1: 50/50 identity-level split (no identity overlap) ──────────
    perm      = rng.permutation(len(ids))
    n_train   = int(len(ids) * INTRA_TRAIN_ID_RATIO)   # e.g. 50 % → Group A
    train_ids = [ids[i] for i in sorted(perm[:n_train])]
    open_ids  = [ids[i] for i in sorted(perm[n_train:])]

    # Label spaces are entirely separate
    train_label = {iid: idx for idx, iid in enumerate(train_ids)}
    open_label  = {iid: idx for idx, iid in enumerate(open_ids)}

    train_s, val_s, gallery_s, query_s = [], [], [], []

    # ── Step 2: Group A → train / internal-test split ─────────────────────
    for iid in train_ids:
        files    = index[iid]
        n        = len(files)
        n_tr     = max(1, int(n * INTRA_TRAIN_IMG_RATIO))   # 80 % train
        for f in files[:n_tr]: train_s.append((f, train_label[iid]))
        for f in files[n_tr:]: val_s.append((f,   train_label[iid]))

    # ── Step 3: Group B → gallery / query split (50 / 50) ─────────────────
    for iid in open_ids:
        files = index[iid]
        n     = len(files)
        half  = max(1, n // 2)                              # 50 % gallery
        for f in files[:half]: gallery_s.append((f, open_label[iid]))
        for f in files[half:]: query_s.append((f,  open_label[iid]))

    print(
        f"[Intra]  "
        f"Group A (train) IDs: {len(train_ids)} | "
        f"  train imgs: {len(train_s)}  val imgs: {len(val_s)}\n"
        f"         "
        f"Group B (open)  IDs: {len(open_ids)}  | "
        f"  gallery: {len(gallery_s)}  query: {len(query_s)}"
    )
    return train_s, val_s, gallery_s, query_s


# ── Protocol 2: Cross-Dataset Open-Set  (§IV-A-6, Table V) ─────────────────

def build_cross_dataset_splits() -> Tuple[List, List, List]:
    """
    Paper §IV-A-6 — Cross-dataset open-set evaluation (CASIA-MS adaptation):

    "In each evaluation round, one dataset is used exclusively for model
     training and validation, while the remaining datasets are reserved
     solely for gallery-query verification."

    Adapted for CASIA-MS using spectral bands as proxy domains:
      • Source domain (train) : CROSS_TRAIN_SPECTRA, first CROSS_TRAIN_ID_RATIO
                                fraction of all sorted identity keys.
      • Target domain (eval)  : CROSS_TEST_SPECTRA, remaining identity keys.
      • Zero identity overlap guaranteed by positional split on sorted key list.

    WHY positional split instead of subject-index threshold:
      The original code used a hardcoded threshold (e.g. subjects > 150) which
      silently produces an empty eval set when the dataset has fewer subjects
      than the threshold (e.g. CASIA-MS with 100 subjects, all indexed 1-100).
      Splitting by position on the sorted key list works for any number of
      subjects regardless of how they are numbered.

    Returns
    -------
    train_samples, gallery_samples, query_samples
    """
    # ── Build full index for both spectral subsets ────────────────────────
    full_train_index = build_index(DATA_ROOT, CROSS_TRAIN_SPECTRA)
    full_eval_index  = build_index(DATA_ROOT, CROSS_TEST_SPECTRA)

    # ── Collect all unique subject IDs (not identity keys) ───────────────
    # A "subject" is the person, a "identity key" is subject+hand.
    # We split at the subject level so that both hands of a subject always
    # land in the same partition — preventing data leakage between train/eval.
    all_subjects = sorted(set(
        iid.split("_")[0]
        for iid in set(full_train_index.keys()) | set(full_eval_index.keys())
    ))
    n_train_subj = max(1, int(len(all_subjects) * CROSS_TRAIN_ID_RATIO))
    train_subjects = set(all_subjects[:n_train_subj])   # first 75%
    eval_subjects  = set(all_subjects[n_train_subj:])   # last  25%

    # ── Training samples: source spectra, train subjects ─────────────────
    train_ids = sorted(
        iid for iid in full_train_index
        if iid.split("_")[0] in train_subjects
    )
    train_label = {iid: idx for idx, iid in enumerate(train_ids)}
    train_s = [
        (f, train_label[iid])
        for iid in train_ids
        for f in full_train_index[iid]
    ]

    # ── Eval samples: target spectra, eval subjects ───────────────────────
    eval_ids = sorted(
        iid for iid in full_eval_index
        if iid.split("_")[0] in eval_subjects
    )
    eval_label = {iid: idx for idx, iid in enumerate(eval_ids)}

    gallery_s, query_s = [], []
    for iid in eval_ids:
        files = full_eval_index[iid]
        n     = len(files)
        half  = max(1, n // 2)            # 50% gallery / 50% query
        for f in files[:half]: gallery_s.append((f, eval_label[iid]))
        for f in files[half:]: query_s.append((f,  eval_label[iid]))

    print(
        f"[Cross]  "
        f"Total subjects: {len(all_subjects)}  "
        f"Train: {len(train_subjects)} subj  Eval: {len(eval_subjects)} subj\n"
        f"         "
        f"Train spectra={CROSS_TRAIN_SPECTRA}  "
        f"IDs: {len(train_ids)}  imgs: {len(train_s)}\n"
        f"         "
        f"Eval  spectra={CROSS_TEST_SPECTRA}  "
        f"IDs: {len(eval_ids)}  "
        f"gallery: {len(gallery_s)}  query: {len(query_s)}"
    )

    if len(eval_ids) == 0:
        raise RuntimeError(
            f"Cross-dataset eval set is empty!\n"
            f"  Total subjects found: {len(all_subjects)}\n"
            f"  CROSS_TRAIN_ID_RATIO={CROSS_TRAIN_ID_RATIO} reserved "
            f"{len(train_subjects)} for train, {len(eval_subjects)} for eval.\n"
            f"  Lower CROSS_TRAIN_ID_RATIO (e.g. 0.6) to leave more subjects for eval."
        )

    return train_s, gallery_s, query_s


# ── Protocol 3: Closed-Set  (§IV-A-6, Tables VI–VII) ───────────────────────

def build_closed_set_splits() -> Tuple[List, List]:
    """
    Paper §IV-A-6 — Closed-set evaluation:

    "In the closed-set scenario, all identities within each dataset are
     preserved without cross-dataset separation.
     In CASIA-MS, all 200 identities are included, with the first 80 %
     of images per identity used for training, and the remaining images
     used for testing and verification."

    There is NO separate gallery / query split in the closed-set protocol.
    The training embeddings (80 %) act as the enrolled templates (gallery),
    and the test embeddings (20 %) are the probes.  Both sets share the
    same identity label space (0 … N_identities-1), so every test sample
    has at least one matching training sample to compare against.

    Evaluation:
      • Rank-1 ACC — each test embed is matched to its nearest training embed.
      • EER        — computed over all genuine/impostor (test vs train) pairs.

    Returns
    -------
    train_samples : list of (path, label)  — 80 % per identity (enrolled templates)
    test_samples  : list of (path, label)  — 20 % per identity (probes)
    """
    index     = build_index(DATA_ROOT, SPECTRA)
    ids       = sorted(index.keys())
    label_map = {iid: idx for idx, iid in enumerate(ids)}

    train_s, test_s = [], []

    for iid in ids:
        files   = index[iid]
        n       = len(files)
        lbl     = label_map[iid]
        n_train = max(1, int(n * CLOSED_TRAIN_IMG_RATIO))   # 80 % enrolled
        for f in files[:n_train]: train_s.append((f, lbl))
        for f in files[n_train:]: test_s.append((f,  lbl))

    n_subj = len(set(iid.split("_")[0] for iid in ids))
    print(
        f"[Closed] "
        f"Subjects: {n_subj}  Identity keys: {len(ids)}  "
        f"(all 4 spectra x both hands)\n"
        f"         "
        f"Train (enrolled): {len(train_s)} imgs  "
        f"Test  (probes)  : {len(test_s)} imgs  "
        f"({int(CLOSED_TRAIN_IMG_RATIO*100)}/{int((1-CLOSED_TRAIN_IMG_RATIO)*100)} split)"
    )
    return train_s, test_s


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. COMPNET BACKBONE                                                       ║
# ║  Liang et al., IEEE Signal Processing Letters, 2021 [20]                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class LearnableGaborLayer(nn.Module):
    """
    Bank of NUM_GABOR_FILTERS Gabor kernels whose five parameters
    (θ, σ, λ, ψ, γ) are nn.Parameters learned end-to-end.

    Input : [B, 1, H, W]
    Output: [B, NUM_GABOR_FILTERS, H, W]  (SAME padding)
    """
    def __init__(self):
        super().__init__()
        n  = NUM_GABOR_FILTERS
        ks = GABOR_KERNEL_SIZE

        # Evenly-spaced initial orientations in [0, π)
        self.theta = nn.Parameter(torch.linspace(0.0, math.pi, n + 1)[:-1])
        self.sigma = nn.Parameter(torch.full((n,), 3.0))
        self.lambd = nn.Parameter(torch.full((n,), 6.0))
        self.psi   = nn.Parameter(torch.zeros(n))
        self.gamma = nn.Parameter(torch.full((n,), 0.5))

        # Fixed spatial grid — moves with .to(device)
        half = ks // 2
        ys   = torch.arange(-half, half + 1, dtype=torch.float32)
        xs   = torch.arange(-half, half + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("xx", xx)
        self.register_buffer("yy", yy)
        self.ks = ks

    def _build_filters(self) -> torch.Tensor:
        """
        Fully vectorised Gabor filter construction — no Python loop.
        All F filters are built in parallel using broadcast arithmetic,
        which also avoids the non-contiguous stride issue in PyTorch 2.x
        that occurs when stacking tensors built from separate scalar params.

        Shapes:
          theta, sigma, lambd, psi, gamma : [F]
          xx, yy (buffers)               : [k, k]
          x_rot, y_rot                   : [F, k, k]
          kernel                         : [F, k, k]
          return                         : [F, 1, k, k]  contiguous
        """
        # Clamp to safe ranges                               [F]
        theta = self.theta                                   # [F]
        sigma = self.sigma.abs().clamp(min=0.5)             # [F]
        lambd = self.lambd.abs().clamp(min=1.0)             # [F]
        psi   = self.psi                                     # [F]
        gamma = self.gamma.abs().clamp(min=0.1)             # [F]

        # Broadcast grid to [F, k, k] by inserting a leading dim
        xx = self.xx.unsqueeze(0)   # [1, k, k]
        yy = self.yy.unsqueeze(0)   # [1, k, k]

        # Reshape params to [F, 1, 1] for broadcasting
        cos_t = torch.cos(theta).view(-1, 1, 1)
        sin_t = torch.sin(theta).view(-1, 1, 1)
        sigma = sigma.view(-1, 1, 1)
        lambd = lambd.view(-1, 1, 1)
        psi   = psi.view(-1, 1, 1)
        gamma = gamma.view(-1, 1, 1)

        # Rotated coordinates                               [F, k, k]
        x_rot =  xx * cos_t + yy * sin_t
        y_rot = -xx * sin_t + yy * cos_t

        # Gabor = Gaussian envelope × cosine carrier       [F, k, k]
        envelope = torch.exp(
            -(x_rot**2 + gamma**2 * y_rot**2) / (2.0 * sigma**2)
        )
        kernel = envelope * torch.cos(2.0 * math.pi * x_rot / lambd + psi)

        # Zero-mean per filter (suppress DC component)
        kernel = kernel - kernel.mean(dim=(1, 2), keepdim=True)

        # [F, 1, k, k]  — .contiguous() ensures valid strides for conv2d
        return kernel.unsqueeze(1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]  →  [B, F, H, W]
        filters = self._build_filters()             # [F, 1, k, k]  contiguous
        return F.conv2d(x, filters, padding=self.ks // 2)


class CompetitivePool(nn.Module):
    """
    Element-wise max across the orientation dimension — the competition
    mechanism core to CompNet.

    Input : [B, F, H, W]
    Output: [B, 1, H, W]
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = x.abs().max(dim=1, keepdim=True)
        return out


class CompNet(nn.Module):
    """
    Full CompNet pipeline:
      [B,1,H,W]
        → LearnableGaborLayer → ReLU → CompetitivePool → BN
        → Conv block ×4 with MaxPool → AdaptiveAvgPool(4,4)
        → Linear(FEATURE_DIM) → BN → L2-Norm  →  [B, FEATURE_DIM]
    """
    def __init__(self, num_classes: int):
        super().__init__()

        # Stage 1 — Learnable Gabor + Competition
        self.gabor   = LearnableGaborLayer()
        self.compete = CompetitivePool()
        self.gbn     = nn.BatchNorm2d(1)

        # Stage 2 — Deep CNN
        def _block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin,  cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.block1 = _block(1,   32);  self.pool1 = nn.MaxPool2d(2)
        self.block2 = _block(32,  64);  self.pool2 = nn.MaxPool2d(2)
        self.block3 = _block(64,  128); self.pool3 = nn.MaxPool2d(2)
        self.block4 = _block(128, 256); self.gap   = nn.AdaptiveAvgPool2d((4, 4))

        # Stage 3 — Embedding head
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, FEATURE_DIM, bias=False),
            nn.BatchNorm1d(FEATURE_DIM),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,1,H,W]  →  z: [B, FEATURE_DIM]  L2-normalised"""
        g = F.relu(self.gabor(x))           # [B, F, H, W]
        g = self.gbn(self.compete(g))       # [B, 1, H, W]
        f = self.pool1(self.block1(g))
        f = self.pool2(self.block2(f))
        f = self.pool3(self.block3(f))
        f = self.gap(self.block4(f))
        z = self.embed(f)
        return F.normalize(z, p=2, dim=1)   # unit hypersphere


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. PALMBRIDGE MODULE  (§III-B and §III-C)                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class PalmBridge(nn.Module):
    """
    Eq.(1)  k*_i = argmin_k ||z_i - p_k||²        nearest-vector index
    Eq.(2)  z̃_i  = p_{k*_i}                        mapped vector
    Eq.(4)  ẑ_i  = W_ORI·z_i + W_MAP·z̃_i          blended feature
    Eq.(6)  L_con  feature-consistency with sg(·)
    Eq.(7)  L_o    orthogonality on codebook P
    Eq.(13) p_same   assignment consistency
    Eq.(15) p_collide collision rate
    """
    def __init__(self):
        super().__init__()
        self.K = NUM_PB_VECTORS
        # Representative vectors P ∈ R^{K×D} — initialised on unit sphere
        self.P = nn.Parameter(
            F.normalize(torch.randn(NUM_PB_VECTORS, FEATURE_DIM), p=2, dim=1)
        )

    # ── Eq. (1)–(2): nearest-vector search ─────────────────────────────────

    def _nearest_vector(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorised pairwise L2:  ||z-p||² = ||z||² + ||p||² - 2·z·Pᵀ
        Returns z_tilde [B, D] and assignment indices [B].
        """
        dists = (
            z.pow(2).sum(1, keepdim=True)           # [B, 1]
            + self.P.pow(2).sum(1).unsqueeze(0)     # [1, K]
            - 2.0 * (z @ self.P.t())                # [B, K]
        )
        idx     = dists.argmin(dim=1)               # [B]     Eq. (1)
        z_tilde = self.P[idx]                       # [B, D]  Eq. (2)
        return z_tilde, idx

    # ── Eq. (4): blending ───────────────────────────────────────────────────

    def _blend(self, z: torch.Tensor, z_tilde: torch.Tensor) -> torch.Tensor:
        """ẑ_i = W_ORI·z_i + W_MAP·z̃_i"""
        return W_ORI * z + W_MAP * z_tilde

    # ── Full forward ─────────────────────────────────────────────────────────

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z_hat   [B, D] — blended feature ẑ (passed to ArcFace)
        z_tilde [B, D] — raw mapped vectors  (needed for L_con)
        indices [B]    — codebook assignments (diagnostics)
        """
        z_tilde, indices = self._nearest_vector(z)
        z_hat            = self._blend(z, z_tilde)
        return z_hat, z_tilde, indices

    # ── Eq. (6): Feature-Consistency Loss ───────────────────────────────────

    def loss_consistency(
        self, z: torch.Tensor, z_tilde: torch.Tensor
    ) -> torch.Tensor:
        """
        L_con = (1/B) Σ_i [
            ||p_i  - sg(z_i)||²        ← updates P only (backbone blocked)
          + λ·||sg(p_i) - z_i||²       ← updates backbone only (P blocked)
        ]
        sg(·) = .detach()
        """
        t1 = (z_tilde - z.detach()).pow(2).sum(1).mean()
        t2 = (z - z_tilde.detach()).pow(2).sum(1).mean()
        return t1 + LAMBDA_CON * t2

    # ── Eq. (7): Orthogonality Loss ─────────────────────────────────────────

    def loss_orthogonal(self) -> torch.Tensor:
        """
        W  = L2-norm(P)  ∈ R^{K×D}
        S  = W·Wᵀ        ∈ R^{K×K}   (pairwise cosine similarities)
        L_o = (1/K²) Σ_{ij} (S_ij - δ_ij)²   →  drives S toward I_K
        """
        W = F.normalize(self.P, p=2, dim=1)
        S = W @ W.t()
        I = torch.eye(self.K, device=S.device, dtype=S.dtype)
        return ((S - I).pow(2)).sum() / (self.K ** 2)

    # ── Eq. (13): Assignment consistency ────────────────────────────────────

    @torch.no_grad()
    def assignment_consistency(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> float:
        """p_same = Pr[Π(z1)==Π(z2) | same identity]"""
        _, i1 = self._nearest_vector(z1)
        _, i2 = self._nearest_vector(z2)
        return (i1 == i2).float().mean().item()

    # ── Eq. (15): Collision rate ─────────────────────────────────────────────

    @torch.no_grad()
    def collision_rate(
        self, za: torch.Tensor, zb: torch.Tensor
    ) -> float:
        """p_collide = Pr[Π(za)==Π(zb) | different identity]"""
        _, ia = self._nearest_vector(za)
        _, ib = self._nearest_vector(zb)
        return (ia == ib).float().mean().item()

    # ── Codebook diagnostics ─────────────────────────────────────────────────

    @torch.no_grad()
    def codebook_usage(self) -> dict:
        W   = F.normalize(self.P, p=2, dim=1)
        S   = W @ W.t()
        off = S[~torch.eye(self.K, dtype=torch.bool, device=S.device)]
        return {"mean_cosine":    off.mean().item(),
                "near_duplicate": (off > 0.9).float().mean().item()}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. ARCFACE LOSS  (L_bak)                                                 ║
# ║  Deng et al., CVPR 2019 [5]                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ArcFaceLoss(nn.Module):
    """
    Additive angular margin cross-entropy applied to the blended feature ẑ.
    s = ARC_S = 64,  m = ARC_M = 0.5  (paper §IV-A).
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.s     = ARC_S
        self.cos_m = math.cos(ARC_M)
        self.sin_m = math.sin(ARC_M)
        self.th    = math.cos(math.pi - ARC_M)
        self.mm    = math.sin(math.pi - ARC_M) * ARC_M

        self.weight = nn.Parameter(torch.empty(num_classes, FEATURE_DIM))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        W           = F.normalize(self.weight, p=2, dim=1)
        cos_theta   = (z @ W.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sin_theta   = (1.0 - cos_theta**2).sqrt()
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m,
                                  cos_theta - self.mm)
        one_hot = torch.zeros_like(cos_theta).scatter_(
            1, labels.view(-1, 1), 1.0
        )
        logits = self.s * (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta)
        return self.ce(logits, labels)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5. TRAINING   L = L_bak + α·L_con + β·L_o   (Eq. 8)                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def train_one_epoch(
    backbone:   CompNet,
    palmbridge: PalmBridge,
    arcface:    ArcFaceLoss,
    loader:     DataLoader,
    optimizer:  optim.Optimizer,
    epoch:      int,
) -> dict:
    """
    One training epoch.  No per-batch printing — caller decides when to log.
    Returns mean losses + train classification accuracy (from ArcFace logits).
    """
    backbone.train(); palmbridge.train(); arcface.train()
    totals = {"loss": 0.0, "bak": 0.0, "con": 0.0, "orth": 0.0}
    n_correct = 0
    n_total   = 0

    # During warm-up phase, disable PalmBridge blending so the backbone
    # can learn discriminative features before the codebook interferes.
    pb_active = (epoch > WARMUP_EPOCHS)

    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # ── Forward ───────────────────────────────────────────────────────
        z                 = backbone(imgs)            # [B, D]  Eq. (9)
        z_hat, z_tilde, _ = palmbridge(z)             # Eq. (1)–(5)

        # ── Losses  Eq. (8): L = L_bak + α·L_con + β·L_o ─────────────────
        # During warm-up:
        #   - ArcFace uses raw z (not blended z_hat) so backbone trains cleanly
        #   - L_con IS computed so the codebook silently aligns with backbone
        #     features. By epoch WARMUP_EPOCHS, L_con is near-zero, meaning
        #     PalmBridge activation at epoch 21 causes no loss spike.
        #   - L_o always trains to keep codebook vectors diverse
        # After warm-up:
        #   - ArcFace uses blended z_hat (full PalmBridge pipeline)
        #   - All three loss terms contribute normally
        feat_for_arc = z_hat if pb_active else z
        L_bak = arcface(feat_for_arc, labels)
        L_con = palmbridge.loss_consistency(z, z_tilde)  # always — pre-aligns during warmup
        L_o   = palmbridge.loss_orthogonal()              # always — keeps codebook diverse
        loss  = L_bak + ALPHA * L_con + BETA * L_o

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(palmbridge.parameters()),
            max_norm=10.0,
        )
        optimizer.step()

        # ── Track train classification accuracy from ArcFace logits ───────
        with torch.no_grad():
            W       = F.normalize(arcface.weight, p=2, dim=1)
            logits  = ARC_S * (z_hat @ W.t())         # [B, C] cosine logits
            preds   = logits.argmax(dim=1)
            n_correct += (preds == labels).sum().item()
            n_total   += labels.size(0)

        totals["loss"] += loss.item()
        totals["bak"]  += L_bak.item()
        totals["con"]  += L_con.item()
        totals["orth"] += L_o.item()

    nb = max(len(loader), 1)
    metrics = {k: v / nb for k, v in totals.items()}
    metrics["train_acc"] = n_correct / max(n_total, 1)
    return metrics


def build_optimizer(backbone, palmbridge, arcface) -> optim.Optimizer:
    params = (list(backbone.parameters())
              + list(palmbridge.parameters())
              + list(arcface.parameters()))
    return optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)


def build_scheduler(optimizer):
    """
    Linear warmup for first WARMUP_EPOCHS epochs, then cosine anneal.
    Warmup prevents large random gradients from corrupting early weights.
    """
    def lr_lambda(epoch):
        # epoch is 0-indexed here (scheduler steps after each epoch)
        if epoch < WARMUP_EPOCHS:
            # Linear ramp: 0.1 → 1.0 over WARMUP_EPOCHS
            return 0.1 + 0.9 * (epoch / max(WARMUP_EPOCHS - 1, 1))
        # Cosine decay from 1.0 → 1e-3 (relative) over remaining epochs
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(backbone, palmbridge, arcface, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"epoch": epoch,
                "backbone":   backbone.state_dict(),
                "palmbridge": palmbridge.state_dict(),
                "arcface":    arcface.state_dict(),
                "optimizer":  optimizer.state_dict()}, path)
    print(f"[ckpt] Saved → {path}")


def load_checkpoint(path, backbone, palmbridge, arcface=None,
                    optimizer=None) -> int:
    ckpt = torch.load(path, map_location=DEVICE)
    backbone.load_state_dict(ckpt["backbone"])
    palmbridge.load_state_dict(ckpt["palmbridge"])
    if arcface   and "arcface"   in ckpt: arcface.load_state_dict(ckpt["arcface"])
    if optimizer and "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[ckpt] Loaded ← {path}  (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6. EVALUATION  (EER, ACC, ROC, GI distributions, blending sweep)        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def extract_features(
    backbone:         CompNet,
    palmbridge:       PalmBridge,
    loader:           DataLoader,
    apply_palmbridge: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    backbone.eval(); palmbridge.eval()
    feats, lbls = [], []
    for imgs, labels in loader:
        z = backbone(imgs.to(DEVICE))
        if apply_palmbridge:
            z, _, _ = palmbridge(z)     # blended ẑ
        feats.append(z.cpu().numpy())
        lbls.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(lbls)


def cosine_sim_matrix(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Returns [Nq, Ng] cosine similarity matrix."""
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
    return q @ g.T


def compute_eer(
    q_feats: np.ndarray, g_feats: np.ndarray,
    q_labels: np.ndarray, g_labels: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweeps N_EER_THRESHOLDS thresholds over all genuine/impostor pairs.
    Returns  eer, far_arr, tpr_arr, thresholds, genuine_scores, impostor_scores
    """
    sim      = cosine_sim_matrix(q_feats, g_feats)
    match    = q_labels[:, None] == g_labels[None, :]
    genuine  = sim[match]
    impostor = sim[~match]

    all_s      = np.concatenate([genuine, impostor])
    thresholds = np.linspace(all_s.min(), all_s.max(), N_EER_THRESHOLDS)

    far = np.array([(impostor >= t).mean() for t in thresholds])
    frr = np.array([(genuine  <  t).mean() for t in thresholds])
    tpr = 1.0 - frr

    idx = np.argmin(np.abs(far - frr))
    eer = float((far[idx] + frr[idx]) / 2.0)

    return eer, far, tpr, thresholds, genuine, impostor


def compute_rank1_acc(
    q_feats: np.ndarray, g_feats: np.ndarray,
    q_labels: np.ndarray, g_labels: np.ndarray,
) -> float:
    sim      = cosine_sim_matrix(q_feats, g_feats)
    pred_lbl = g_labels[sim.argmax(axis=1)]
    return float((pred_lbl == q_labels).mean())


def plot_roc_and_gi(
    far: np.ndarray, tpr: np.ndarray, eer: float,
    genuine: np.ndarray, impostor: np.ndarray,
    tag: str, plot_dir: str,
) -> None:
    """Saves combined ROC + GI plot (reproduces Fig. 2 and Fig. 3 style)."""
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    axes[0].plot(far, tpr, linewidth=1.5,
                 label=f"{tag}  EER={eer*100:.2f}%")
    axes[0].set_xlabel("FAR"); axes[0].set_ylabel("GAR (1−FRR)")
    axes[0].set_title("ROC Curve"); axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # GI score distribution
    lo, hi = min(genuine.min(), impostor.min()), max(genuine.max(), impostor.max())
    bins   = np.linspace(lo, hi, 60)
    axes[1].hist(genuine,  bins=bins, density=True, alpha=0.6,
                 label="Genuine",  color="tab:green")
    axes[1].hist(impostor, bins=bins, density=True, alpha=0.6,
                 label="Impostor", color="tab:red")
    axes[1].set_xlabel("Cosine Similarity"); axes[1].set_ylabel("Density")
    axes[1].set_title("GI Score Distribution"); axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"PalmBridge — {tag}"); plt.tight_layout()
    path = os.path.join(plot_dir, f"{tag}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [plot] {path}")


def run_evaluation(
    backbone:         CompNet,
    palmbridge:       PalmBridge,
    gallery_loader:   DataLoader,
    query_loader:     DataLoader,
    tag:              str,
    apply_palmbridge: bool = True,
    save_plots:       bool = True,
    plot_dir:         str  = PLOT_DIR,
) -> dict:
    g_feats, g_labels = extract_features(backbone, palmbridge,
                                         gallery_loader, apply_palmbridge)
    q_feats, q_labels = extract_features(backbone, palmbridge,
                                         query_loader,  apply_palmbridge)

    eer, far, tpr, thresh, genuine, impostor = compute_eer(
        q_feats, g_feats, q_labels, g_labels
    )
    acc   = compute_rank1_acc(q_feats, g_feats, q_labels, g_labels)
    stats = palmbridge.codebook_usage()

    print(f"  [{tag}] EER={eer*100:.4f}%  ACC={acc*100:.2f}%  "
          f"Genuine:{len(genuine)}  Impostor:{len(impostor)}  "
          f"cb_cos={stats['mean_cosine']:.3f}")

    if save_plots:
        plot_roc_and_gi(far, tpr, eer, genuine, impostor, tag, plot_dir)

    return {"eer": eer, "acc": acc, "far": far, "tpr": tpr,
            "thresholds": thresh, "genuine": genuine, "impostor": impostor}


def run_evaluation_closed(
    backbone:         CompNet,
    palmbridge:       PalmBridge,
    train_loader:     DataLoader,
    test_loader:      DataLoader,
    tag:              str,
    apply_palmbridge: bool = True,
    save_plots:       bool = True,
    plot_dir:         str  = PLOT_DIR,
) -> dict:
    """
    Closed-set evaluation: compare test (probe) embeddings against
    training (enrolled template) embeddings within the SAME identity space.

    No gallery/query split — training features ARE the enrolled gallery.
    Metrics:
      • Rank-1 ACC  — nearest-neighbour identity prediction.
      • EER         — FAR/FRR sweep over all genuine and impostor
                      (test-vs-train) similarity pairs.
    """
    # Extract embeddings from both sets
    enroll_feats, enroll_labels = extract_features(
        backbone, palmbridge, train_loader, apply_palmbridge
    )
    probe_feats, probe_labels = extract_features(
        backbone, palmbridge, test_loader, apply_palmbridge
    )

    # Rank-1 ACC: each probe matched to its nearest enrolled template
    acc = compute_rank1_acc(probe_feats, enroll_feats, probe_labels, enroll_labels)

    # EER: probe vs enrolled similarity scores split into genuine / impostor
    eer, far, tpr, thresh, genuine, impostor = compute_eer(
        probe_feats, enroll_feats, probe_labels, enroll_labels
    )

    stats = palmbridge.codebook_usage()
    print(f"  [{tag}] EER={eer*100:.4f}%  ACC={acc*100:.2f}%  "
          f"Enrolled:{len(enroll_feats)}  Probes:{len(probe_feats)}  "
          f"Genuine pairs:{len(genuine)}  Impostor pairs:{len(impostor)}  "
          f"cb_cos={stats['mean_cosine']:.3f}")

    if save_plots:
        plot_roc_and_gi(far, tpr, eer, genuine, impostor, tag, plot_dir)

    return {"eer": eer, "acc": acc, "far": far, "tpr": tpr,
            "thresholds": thresh, "genuine": genuine, "impostor": impostor}


def blending_sweep(
    backbone:       CompNet,
    palmbridge:     PalmBridge,
    gallery_loader: DataLoader,
    query_loader:   DataLoader,
    plot_dir:       str,
) -> Dict[float, float]:
    """
    Sweeps W_MAP ∈ [0, 1] (W_ORI = 1 − W_MAP) and records EER.
    Reproduces Fig. 4 of the paper.
    """
    os.makedirs(plot_dir, exist_ok=True)
    backbone.eval(); palmbridge.eval()

    @torch.no_grad()
    def _raw(loader):
        feats, lbls = [], []
        for imgs, labels in loader:
            feats.append(backbone(imgs.to(DEVICE)).cpu().numpy())
            lbls.append(labels.numpy())
        return np.concatenate(feats), np.concatenate(lbls)

    g_raw, g_lbl = _raw(gallery_loader)
    q_raw, q_lbl = _raw(query_loader)
    P = palmbridge.P.detach().cpu().numpy()             # [K, D]

    def _blend_np(z, wm, wo):
        dists = ((z**2).sum(1, keepdims=True) + (P**2).sum(1)
                 - 2.0 * z @ P.T)
        return wo * z + wm * P[dists.argmin(axis=1)]

    ratios   = np.linspace(0.0, 1.0, SWEEP_STEPS + 1)
    eer_vals = []
    results  = {}

    for wm in ratios:
        wo      = 1.0 - wm
        eer, *_ = compute_eer(_blend_np(q_raw, wm, wo),
                               _blend_np(g_raw, wm, wo), q_lbl, g_lbl)
        results[float(wm)] = float(eer)
        eer_vals.append(eer * 100)
        print(f"  [sweep] wmap={wm:.2f} wori={wo:.2f}  EER={eer*100:.4f}%")

    # Plot (Fig. 4)
    fig, ax = plt.subplots(figsize=(7, 4))
    xlabels = [f"{wm:.1f}:{(1-wm):.1f}" for wm in ratios]
    ax.plot(range(len(ratios)), eer_vals, "o-", lw=1.5, ms=5)
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("w_map : w_ori"); ax.set_ylabel("EER (%)")
    ax.set_title("EER vs. Blending Coefficient  (Fig. 4)")
    ax.grid(True, alpha=0.3); plt.tight_layout()
    path = os.path.join(plot_dir, "fig4_blending_sweep.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [sweep] Fig.4 → {path}")
    return results


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  7. GENERIC TRAIN + EVAL RUNNER                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def train_and_eval(
    train_samples:   list,
    gallery_samples: list,
    query_samples:   list,
    protocol_name:   str,
    ckpt_name:       str,
) -> dict:
    num_classes = len(set(lbl for _, lbl in train_samples))
    pdir        = os.path.join(PLOT_DIR, protocol_name)
    os.makedirs(pdir,     exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_loader   = make_loader(train_samples,   train=True,  shuffle=True)
    gallery_loader = make_loader(gallery_samples, train=False, shuffle=False)
    query_loader   = make_loader(query_samples,   train=False, shuffle=False)

    backbone   = CompNet(num_classes).to(DEVICE)
    palmbridge = PalmBridge().to(DEVICE)
    arcface    = ArcFaceLoss(num_classes).to(DEVICE)

    # ── Plug-and-play mode (Table XV) ───────────────────────────────────────
    if PLUG_AND_PLAY and PB_CKPT:
        print(f"\n[Plug-and-Play] loading PalmBridge codebook from {PB_CKPT}")
        ckpt = torch.load(PB_CKPT, map_location=DEVICE)
        palmbridge.load_state_dict(ckpt["palmbridge"])

        print("── Naive baseline ──────────────────────────────────────────────")
        run_evaluation(backbone, palmbridge, gallery_loader, query_loader,
                       f"{protocol_name}_pnp_naive", False, True, pdir)
        print("── PalmBridge plug-and-play ────────────────────────────────────")
        return run_evaluation(backbone, palmbridge, gallery_loader, query_loader,
                              f"{protocol_name}_pnp_pb", True, True, pdir)

    # ── Joint training ───────────────────────────────────────────────────────
    optimizer = build_optimizer(backbone, palmbridge, arcface)
    scheduler = build_scheduler(optimizer)
    best_eer  = float("inf")
    ckpt_path = os.path.join(SAVE_DIR, ckpt_name)

    print(f"\n{'='*64}")
    print(f"  PROTOCOL : {protocol_name.upper()}")
    print(f"  Device:{DEVICE}  Classes:{num_classes}  Epochs:{EPOCHS}")
    print(f"  K={NUM_PB_VECTORS}  w_map={W_MAP}  w_ori={W_ORI}")
    print(f"{'='*64}")

    for epoch in range(1, EPOCHS + 1):
        m = train_one_epoch(backbone, palmbridge, arcface,
                            train_loader, optimizer, epoch)
        scheduler.step()

        # ── Compact single-line epoch summary (always printed) ─────────────
        phase = "WARMUP" if epoch <= WARMUP_EPOCHS else "PalmBridge"
        print(f"Epoch {epoch:03d}/{EPOCHS} [{phase}]  "
              f"loss={m['loss']:.4f}  bak={m['bak']:.4f}  "
              f"con={m['con']:.4f}  orth={m['orth']:.4f}  "
              f"train_acc={m['train_acc']*100:.2f}%  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # ── Full EER / ACC evaluation every LOG_EPOCHS epochs ──────────────
        if epoch % LOG_EPOCHS == 0 or epoch == EPOCHS:
            # During warm-up, evaluate without PalmBridge (not meaningful yet)
            use_pb = (epoch > WARMUP_EPOCHS)
            res = run_evaluation(backbone, palmbridge,
                                 gallery_loader, query_loader,
                                 protocol_name, apply_palmbridge=use_pb,
                                 save_plots=(epoch == EPOCHS), plot_dir=pdir)
            pb_label = "PB" if use_pb else "Naive"
            print(f"  >> [eval @{epoch} {pb_label}] "
                  f"EER={res['eer']*100:.4f}%  ACC={res['acc']*100:.2f}%")
            if res["eer"] < best_eer:
                best_eer = res["eer"]
                save_checkpoint(backbone, palmbridge, arcface,
                                optimizer, epoch, ckpt_path)

    # ── Final comparison: naive baseline vs PalmBridge ──────────────────────
    print("\n── Naive baseline ──────────────────────────────────────────────────")
    naive = run_evaluation(backbone, palmbridge, gallery_loader, query_loader,
                           f"{protocol_name}_naive", False, True, pdir)
    print(f"  >> Naive  EER={naive['eer']*100:.4f}%  ACC={naive['acc']*100:.2f}%")

    print("\n── PalmBridge ──────────────────────────────────────────────────────")
    final = run_evaluation(backbone, palmbridge, gallery_loader, query_loader,
                           f"{protocol_name}_pb", True, True, pdir)
    print(f"  >> PalmBridge  EER={final['eer']*100:.4f}%  ACC={final['acc']*100:.2f}%")

    if RUN_SWEEP:
        print("\n── Blending coefficient sweep (Fig. 4) ─────────────────────────")
        blending_sweep(backbone, palmbridge, gallery_loader, query_loader, pdir)

    return final


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  8. PROTOCOL RUNNERS                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_intra():
    """Intra-dataset open-set — Tables I–IV equivalent."""
    train_s, _, gallery_s, query_s = build_intra_dataset_splits()
    return train_and_eval(train_s, gallery_s, query_s, "intra", "best_intra.pt")


def run_cross():
    """Cross-spectrum open-set — Table V equivalent."""
    train_s, gallery_s, query_s = build_cross_dataset_splits()
    return train_and_eval(train_s, gallery_s, query_s, "cross", "best_cross.pt")


def run_closed():
    """
    Closed-set verification — Tables VI-VII equivalent.

    Training (80 %) embeddings = enrolled templates (gallery).
    Test     (20 %) embeddings = probes.
    Both sets share the same identity label space — no open-set identities.
    Evaluation via run_evaluation_closed (not the open-set run_evaluation).
    """
    train_s, test_s = build_closed_set_splits()
    num_classes     = len(set(lbl for _, lbl in train_s))
    pdir            = os.path.join(PLOT_DIR, "closed")
    os.makedirs(pdir,     exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_loader = make_loader(train_s, train=True,  shuffle=True)
    test_loader  = make_loader(test_s,  train=False, shuffle=False)

    backbone   = CompNet(num_classes).to(DEVICE)
    palmbridge = PalmBridge().to(DEVICE)
    arcface    = ArcFaceLoss(num_classes).to(DEVICE)

    # ── Plug-and-play mode ───────────────────────────────────────────────────
    if PLUG_AND_PLAY and PB_CKPT:
        print(f"\n[Plug-and-Play] loading PalmBridge codebook from {PB_CKPT}")
        ckpt = torch.load(PB_CKPT, map_location=DEVICE)
        palmbridge.load_state_dict(ckpt["palmbridge"])
        print("-- Naive baseline -----------------------------------------")
        run_evaluation_closed(backbone, palmbridge, train_loader, test_loader,
                              "closed_pnp_naive", False, True, pdir)
        print("-- PalmBridge plug-and-play --------------------------------")
        return run_evaluation_closed(backbone, palmbridge, train_loader, test_loader,
                                     "closed_pnp_pb", True, True, pdir)

    # ── Joint training ───────────────────────────────────────────────────────
    optimizer = build_optimizer(backbone, palmbridge, arcface)
    scheduler = build_scheduler(optimizer)
    best_eer  = float("inf")
    ckpt_path = os.path.join(SAVE_DIR, "best_closed.pt")

    print(f"\n{'='*64}")
    print(f"  PROTOCOL : CLOSED-SET")
    print(f"  Device:{DEVICE}  Classes:{num_classes}  Epochs:{EPOCHS}")
    print(f"  K={NUM_PB_VECTORS}  w_map={W_MAP}  w_ori={W_ORI}")
    print(f"{'='*64}")

    for epoch in range(1, EPOCHS + 1):
        m = train_one_epoch(backbone, palmbridge, arcface,
                            train_loader, optimizer, epoch)
        scheduler.step()

        # ── Compact single-line epoch summary ──────────────────────────────
        phase = "WARMUP" if epoch <= WARMUP_EPOCHS else "PalmBridge"
        print(f"Epoch {epoch:03d}/{EPOCHS} [{phase}]  "
              f"loss={m['loss']:.4f}  bak={m['bak']:.4f}  "
              f"con={m['con']:.4f}  orth={m['orth']:.4f}  "
              f"train_acc={m['train_acc']*100:.2f}%  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # ── Full EER / ACC evaluation every LOG_EPOCHS epochs ──────────────
        if epoch % LOG_EPOCHS == 0 or epoch == EPOCHS:
            use_pb = (epoch > WARMUP_EPOCHS)
            res = run_evaluation_closed(
                backbone, palmbridge, train_loader, test_loader,
                "closed", apply_palmbridge=use_pb,
                save_plots=(epoch == EPOCHS), plot_dir=pdir
            )
            pb_label = "PB" if use_pb else "Naive"
            print(f"  >> [eval @{epoch} {pb_label}] "
                  f"EER={res['eer']*100:.4f}%  ACC={res['acc']*100:.2f}%")
            if res["eer"] < best_eer:
                best_eer = res["eer"]
                save_checkpoint(backbone, palmbridge, arcface,
                                optimizer, epoch, ckpt_path)

    # ── Final comparison: naive baseline vs PalmBridge ──────────────────────
    print("\n-- Naive baseline (train embeds vs test embeds) ------------------")
    naive = run_evaluation_closed(backbone, palmbridge, train_loader, test_loader,
                                  "closed_naive", False, True, pdir)
    print(f"  >> Naive  EER={naive['eer']*100:.4f}%  ACC={naive['acc']*100:.2f}%")

    print("\n-- PalmBridge ---------------------------------------------------")
    final = run_evaluation_closed(backbone, palmbridge, train_loader, test_loader,
                                  "closed_pb", True, True, pdir)
    print(f"  >> PalmBridge  EER={final['eer']*100:.4f}%  ACC={final['acc']*100:.2f}%")

    if RUN_SWEEP:
        print("\n-- Blending coefficient sweep (Fig. 4) --------------------------")
        blending_sweep(backbone, palmbridge, train_loader, test_loader, pdir)

    return final


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  9. ENTRY POINT                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    if PROTOCOL in ("intra",  "all"): run_intra()
    if PROTOCOL in ("cross",  "all"): run_cross()
    if PROTOCOL in ("closed", "all"): run_closed()
