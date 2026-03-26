"""
W2ML — Weight-based Meta Metric Learning for Open-Set Palmprint Recognition
============================================================================
Paper: "Towards open-set touchless palmprint recognition via weight-based
        meta metric learning", Shao & Zhong, Pattern Recognition 2022.

Hard-mining inspired by: "Multi-Similarity Loss With General Pair Weighting
        for Deep Metric Learning", Wang et al., CVPR 2019.

Single-file implementation for CASIA-MS dataset.
Filename format:  {subjectID}_{handSide}_{spectrum}_{iteration}.jpg
  e.g.            001_L_460_01.jpg
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import os
import random
import time
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Third-party
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
#  PARAMETERS  —  edit this block only
# ═══════════════════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = "/home/pai-ng/Jamal/CASIA-MS-ROI"   # folder containing all ROI .jpg files
SAVE_DIR  = "checkpoints"       # where best.pth / latest.pth are written
RESUME    = None                # path to a .pth checkpoint to resume from
                                # e.g. "checkpoints/best.pth"

# ── Evaluation protocol ───────────────────────────────────────────────────────
EVAL_PROTOCOL  = 'cross_subject'
TRAIN_RATIO    = 0.8

ALL_SPECTRA    = ['460', '630', '700', '850', '940', 'White']
TRAIN_SPECTRA  = ['460', '630', '700']
TEST_SPECTRA   = ['850', '940', 'White']

# ── Image ─────────────────────────────────────────────────────────────────────
IMG_SIZE = 128

# ── Episode sampling ──────────────────────────────────────────────────────────
N           = 32    # number of classes per episode
# K is NO LONGER set here — it is computed dynamically from the data as:
#   K = min_images_per_identity_in_training_set - Q_PER_CLASS
# This ensures every identity can contribute support + query without running out.
Q_PER_CLASS = 5     # query images per class (fixed); the rest become support (K)

EPISODES_PER_EPOCH = 200

# ── Model ─────────────────────────────────────────────────────────────────────
# Choose backbone:
#   'custom'   — lightweight CNN trained from scratch (original paper default)
#   'resnet18' — ResNet-18 pretrained on ImageNet, layer1-3 frozen,
#                layer4 + embedding head fine-tuned (paper Section 4.2)
BACKBONE   = 'resnet18'
EMBED_DIM  = 128

# Differential learning rates (only used when BACKBONE = 'resnet18')
#   LR_LAYER4 : fine-tune rate for the unfrozen ResNet layer4
#   LR_HEAD   : full learning rate for the new 128-d embedding head
# The frozen layers (layer1-3) receive no gradient at all.
LR_LAYER4  = 2e-5   # 10× lower than head — gentle fine-tuning of layer4
LR_HEAD    = 2e-4   # matches the paper's Adam base lr (same as LR above)

# ── Loss hyper-parameters (Section 3.3) ──────────────────────────────────────
ALPHA  = 2.0
BETA   = 40.0
GAMMA  = 0.5
MARGIN = 0.05

# ── Training ──────────────────────────────────────────────────────────────────
NUM_EPOCHS   = 60
LR           = 2e-4
WEIGHT_DECAY = 1e-4
LR_STEP      = 20
LR_GAMMA     = 0.5
GRAD_CLIP    = 5.0

DEVICE      = 'cuda'
NUM_WORKERS = 4
SEED        = 42
LOG_INTERVAL = 50


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def parse_casia_filename(fname: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse  {subjectID}_{handSide}_{spectrum}_{iteration}.ext
    Returns (subject_id, side, spectrum, iteration) or None.

    CASIA-MS example:  001_L_460_01.jpg
      → subject='001', side='L', spectrum='460', iteration='01'

    Identity = subjectID + '_' + side  (left/right treated as distinct classes).

    NOTE: subjectID is assumed to contain NO underscores (e.g. '001', '002').
    If your dataset uses multi-part IDs, adjust the split index accordingly.
    """
    name  = os.path.splitext(fname)[0]
    parts = name.split('_')
    # Minimum 4 parts: subjectID, side, spectrum, iteration
    if len(parts) < 4:
        return None
    # parts[-1] = iteration, parts[-2] = spectrum, parts[-3] = side,
    # parts[:-3] joined = subjectID (handles IDs that contain underscores)
    iteration = parts[-1]
    spectrum  = parts[-2]
    side      = parts[-3]
    subject   = '_'.join(parts[:-3])   # robust to multi-part subject IDs
    return subject, side, spectrum, iteration


def build_identity_index(root: str,
                         spectra: Optional[List[str]] = None
                         ) -> Dict[str, List[str]]:
    """
    Single-pass scan of DATA_ROOT.
    Returns  {identity_string: [list_of_full_file_paths]}
    where identity = '{subjectID}_{side}'.

    Having one consolidated scan avoids the double-pass inconsistency that
    existed in the original get_all_subjects + split_subjects pair.
    """
    allowed = set(spectra) if spectra else set(ALL_SPECTRA)
    index: Dict[str, List[str]] = defaultdict(list)
    for fname in sorted(os.listdir(root)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        parsed = parse_casia_filename(fname)
        if parsed is None:
            continue
        subject, side, spectrum, _ = parsed
        if spectrum not in allowed:
            continue
        identity = f"{subject}_{side}"
        index[identity].append(os.path.join(root, fname))
    return dict(index)


def split_subjects(root: str,
                   train_ratio: float = TRAIN_RATIO,
                   spectra: Optional[List[str]] = None
                   ) -> Tuple[List[str], List[str]]:
    """
    Open-set subject split (Section 4.2).

    Returns (train_identities, test_identities).
    Zero category overlap guaranteed.
    Sorted subject list → deterministic split (same as original).
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"TRAIN_RATIO must be in (0, 1), got {train_ratio}")

    index = build_identity_index(root, spectra)

    # Derive unique subject IDs from the identities (strip the '_side' suffix)
    # Identities have the form '{subjectID}_{side}', where side ∈ {L, R}.
    # We split on the last '_' to recover the subject.
    subjects = sorted({ident.rsplit('_', 1)[0] for ident in index})
    n_train  = max(1, int(len(subjects) * train_ratio))
    train_subjects = set(subjects[:n_train])
    test_subjects  = set(subjects[n_train:])

    all_identities = sorted(index.keys())
    train_ids = [i for i in all_identities if i.rsplit('_', 1)[0] in train_subjects]
    test_ids  = [i for i in all_identities if i.rsplit('_', 1)[0] in test_subjects]
    return train_ids, test_ids


def print_identity_counts(dataset: 'CASIAMSDataset',
                          tag: str = '',
                          top_n_outliers: int = 10) -> None:
    """
    Print per-identity image counts for a CASIAMSDataset.

    Shows:
      • Summary statistics (min / median / max / mean)
      • Distribution histogram (bucketed)
      • The `top_n_outliers` identities with the fewest images (potential issues)
      • The `top_n_outliers` identities with the most images
    """
    counts: Dict[str, int] = defaultdict(int)
    # Reverse the identity_to_idx map so we can print human-readable names
    idx_to_identity = {v: k for k, v in dataset.identity_to_idx.items()}

    for _, label, _ in dataset.samples:
        counts[idx_to_identity[label]] += 1

    if not counts:
        print(f"  [{tag}] No samples found.")
        return

    values = sorted(counts.values())
    n      = len(values)
    print(f"\n{'─'*60}")
    print(f"  Identity image counts  [{tag}]")
    print(f"{'─'*60}")
    print(f"  Total identities : {n}")
    print(f"  Total images     : {sum(values)}")
    print(f"  Min              : {values[0]}")
    print(f"  Median           : {values[n // 2]}")
    print(f"  Mean             : {sum(values) / n:.1f}")
    print(f"  Max              : {values[-1]}")

    # Histogram
    bucket_size = max(1, (values[-1] - values[0]) // 10 + 1)
    buckets: Dict[int, int] = defaultdict(int)
    for v in values:
        buckets[(v // bucket_size) * bucket_size] += 1
    print(f"\n  Count distribution (bucket size = {bucket_size}):")
    for bucket_start in sorted(buckets):
        bar = '█' * buckets[bucket_start]
        print(f"    [{bucket_start:4d}–{bucket_start + bucket_size - 1:4d}]  "
              f"{bar}  ({buckets[bucket_start]})")

    # Bottom outliers
    sorted_by_count = sorted(counts.items(), key=lambda x: x[1])
    print(f"\n  {top_n_outliers} identities with fewest images:")
    for ident, cnt in sorted_by_count[:top_n_outliers]:
        print(f"    {ident:<20s}  {cnt:3d} images")

    # Top identities
    print(f"\n  {top_n_outliers} identities with most images:")
    for ident, cnt in sorted_by_count[-top_n_outliers:][::-1]:
        print(f"    {ident:<20s}  {cnt:3d} images")

    print(f"{'─'*60}\n")


class CASIAMSDataset(Dataset):
    """
    CASIA-MS ROI dataset with optional spectrum filtering.
    Returns (img_tensor, int_label, spectrum_str) per sample.
    """

    def __init__(
        self,
        root:       str,
        identities: List[str],
        spectra:    Optional[List[str]] = None,
        transform=None,
    ):
        self.root      = root
        self.spectra   = set(spectra) if spectra else set(ALL_SPECTRA)
        self.transform = transform

        self.identity_to_idx: Dict[str, int] = {
            ident: i for i, ident in enumerate(sorted(identities))
        }
        self.samples: List[Tuple[str, int, str]] = []
        valid_ids = set(identities)

        for fname in sorted(os.listdir(root)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            parsed = parse_casia_filename(fname)
            if parsed is None:
                continue
            subject, side, spectrum, _ = parsed
            if spectrum not in self.spectra:
                continue
            identity = f"{subject}_{side}"
            if identity not in valid_ids:
                continue
            label = self.identity_to_idx[identity]
            self.samples.append((os.path.join(root, fname), label, spectrum))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label, spectrum = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, spectrum


def get_transforms(train: bool = True) -> T.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if train:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — EPISODE SAMPLER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dynamic_k(dataset: CASIAMSDataset,
                      q_per_class: int = Q_PER_CLASS) -> int:
    """
    Compute K = min_per_identity_count - q_per_class.

    We use the minimum image count across ALL identities in the dataset so that
    every identity can participate in every episode without running out of images.
    A warning is printed if the resulting K is small.
    """
    counts: Dict[int, int] = defaultdict(int)
    for _, label, _ in dataset.samples:
        counts[label] += 1

    if not counts:
        raise ValueError("Dataset is empty — cannot compute K.")

    min_count = min(counts.values())
    k = min_count - q_per_class

    if k <= 0:
        raise ValueError(
            f"min images per identity = {min_count}, Q_PER_CLASS = {q_per_class}. "
            f"K = {k} ≤ 0. Either reduce Q_PER_CLASS or ensure each identity "
            f"has more than {q_per_class} images."
        )
    if k < 4:
        print(f"  WARNING: K={k} is very small (min_count={min_count}, "
              f"Q_PER_CLASS={q_per_class}). "
              f"Consider filtering out identities with few samples.")
    return k


class EpisodeSampler:
    """
    Samples one episode: N classes × (K support + Q_PER_CLASS query) images.

    K is passed in at construction time (computed dynamically from the dataset).
    """

    def __init__(self, dataset: CASIAMSDataset, k: int):
        self.k   = k
        self._label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for i, (_, label, _) in enumerate(dataset.samples):
            self._label_to_indices[label].append(i)

        min_needed = k + Q_PER_CLASS
        self.valid_labels: List[int] = [
            lbl for lbl, idxs in self._label_to_indices.items()
            if len(idxs) >= min_needed
        ]
        if len(self.valid_labels) < N:
            raise ValueError(
                f"Only {len(self.valid_labels)} classes have ≥ {min_needed} "
                f"samples (K={k} + Q={Q_PER_CLASS}), but N={N} are needed "
                f"per episode.\n"
                f"  → Either lower N, lower Q_PER_CLASS, or check for missing "
                f"images in your dataset."
            )
        print(f"  EpisodeSampler ready:  K={k}  Q={Q_PER_CLASS}  "
              f"valid_classes={len(self.valid_labels)}/{len(self._label_to_indices)}")
        self.dataset = dataset

    def sample_episode(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        support_imgs   : (N*K, C, H, W)
        support_labels : (N*K,)   local labels 0 … N-1
        query_imgs     : (N*Q, C, H, W)
        query_labels   : (N*Q,)   local labels 0 … N-1
        """
        chosen = random.sample(self.valid_labels, N)
        s_imgs, s_labels, q_imgs, q_labels = [], [], [], []

        for local_lbl, global_lbl in enumerate(chosen):
            pool   = self._label_to_indices[global_lbl]
            picked = random.sample(pool, self.k + Q_PER_CLASS)
            for idx in picked[:self.k]:
                img, _, _ = self.dataset[idx]
                s_imgs.append(img);  s_labels.append(local_lbl)
            for idx in picked[self.k:]:
                img, _, _ = self.dataset[idx]
                q_imgs.append(img);  q_labels.append(local_lbl)

        return (
            torch.stack(s_imgs),
            torch.tensor(s_labels, dtype=torch.long),
            torch.stack(q_imgs),
            torch.tensor(q_labels, dtype=torch.long),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class W2MLModel(nn.Module):
    """
    Custom CNN trained from scratch (no pretrained weights).

    Architecture:
      Conv1 : 3×3, 16 filters, stride 4, Leaky ReLU(0.1)
      MaxPool: 2×2, stride 1
      Conv2 : 5×5, 32 filters, stride 2, Leaky ReLU(0.1)
      MaxPool: 2×2, stride 1
      Conv3 : 3×3, 64 filters, stride 1, Leaky ReLU(0.1)
      Conv4 : 3×3, 128 filters, stride 1, Leaky ReLU(0.1)
      MaxPool: 2×2, stride 1
      FC1   : 1024, Leaky ReLU(0.1)
      FC2   : 512,  Leaky ReLU(0.1)
      FC3   : 128,  no activation  ← embedding output
      L2-normalise → EMBED_DIM-d unit hypersphere
    """

    def __init__(self):
        super().__init__()
        lrelu = partial(nn.LeakyReLU, negative_slope=0.1, inplace=True)

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(16),
            lrelu(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            lrelu(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            lrelu(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            lrelu(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            feat_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, 1024, bias=False),
            nn.BatchNorm1d(1024),
            lrelu(),
            nn.Dropout(p=0.3),

            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            lrelu(),
            nn.Dropout(p=0.3),

            nn.Linear(512, EMBED_DIM, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.classifier(self.features(x)), p=2, dim=1)

    def param_groups(self, lr: float) -> list:
        return [{'params': self.parameters(), 'lr': lr}]

    def trainable_parameters(self):
        return list(self.parameters())


class ResNet18Model(nn.Module):
    """
    ResNet-18 backbone as used in the paper (Section 4.2).

    Freezing strategy
    ─────────────────
    Frozen  (requires_grad=False) : conv1, bn1, maxpool, layer1, layer2, layer3
    Unfrozen (fine-tuned)         : layer4  ← rich semantic features, adapt to palmprint
    Unfrozen (trained from init)  : embedding head (Linear 512→EMBED_DIM, no bias)

    The original ResNet-18 avgpool + fc are replaced by:
        AdaptiveAvgPool2d(1) → Flatten → Linear(512, EMBED_DIM, bias=False)
    followed by L2 normalisation onto the unit hypersphere.

    Param groups (for differential LR in Adam)
    ──────────────────────────────────────────
      group 0 — layer4          : LR_LAYER4  (gentle fine-tune)
      group 1 — embedding head  : LR_HEAD    (full rate, new weights)

    IMG_SIZE note
    ─────────────
    The paper uses 224×224 ROIs.  This model works at any size ≥ 32 px because
    of the AdaptiveAvgPool2d, but 224 is recommended for best transfer quality.
    Set IMG_SIZE = 224 in the PARAMETERS block when using this backbone.
    """

    # Modules that stay frozen — listed by attribute name on the ResNet object
    _FROZEN_MODULES = ('conv1', 'bn1', 'relu', 'maxpool',
                       'layer1', 'layer2', 'layer3')

    def __init__(self):
        super().__init__()
        import torchvision.models as models

        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # ── Freeze layers 1–3 ────────────────────────────────────────────
        for name in self._FROZEN_MODULES:
            module = getattr(base, name)
            for param in module.parameters():
                param.requires_grad = False

        # ── Keep layer4 (unfrozen by default) ────────────────────────────
        self.frozen_body = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3,
        )
        self.layer4 = base.layer4          # unfrozen — fine-tuned at LR_LAYER4

        # ── New embedding head (replaces avgpool + fc) ────────────────────
        # ResNet-18 layer4 output: (B, 512, H/32, W/32)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.head     = nn.Linear(512, EMBED_DIM, bias=False)  # trained at LR_HEAD

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frozen_body(x)   # frozen — no grad flows here
        x = self.layer4(x)        # fine-tuned
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)

    def param_groups(self, lr: float) -> list:
        """
        Two groups with differential learning rates.
        `lr` argument is accepted for API compatibility but ignored here —
        LR_LAYER4 and LR_HEAD from the global config are used directly.
        """
        return [
            {'params': self.layer4.parameters(),
             'lr': LR_LAYER4,
             'name': 'layer4'},
            {'params': self.head.parameters(),
             'lr': LR_HEAD,
             'name': 'head'},
        ]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @staticmethod
    def frozen_module_names() -> Tuple[str, ...]:
        return ResNet18Model._FROZEN_MODULES


# ─────────────────────────────────────────────────────────────────────────────

def build_model(backbone: str = BACKBONE) -> nn.Module:
    """
    Factory that returns the selected model.

    backbone='custom'   → W2MLModel  (from-scratch CNN, any IMG_SIZE)
    backbone='resnet18' → ResNet18Model  (ImageNet pretrained, IMG_SIZE=224 recommended)
    """
    backbone = backbone.lower().strip()
    if backbone == 'custom':
        return W2MLModel()
    elif backbone == 'resnet18':
        return ResNet18Model()
    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Choose 'custom' or 'resnet18'."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — W2ML LOSS  (Equations 2–8)
# ═══════════════════════════════════════════════════════════════════════════════

def build_meta_support_sets(
    support_embs:   torch.Tensor,   # (N*K, D)
    support_labels: torch.Tensor,   # (N*K,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eq. 2 — S_j_meta = mean_i f(x^j_i), then re-normalise.
    Returns meta_embs (N, D) and meta_labels (N,).
    """
    unique_labels = torch.unique(support_labels, sorted=True)
    meta_embs = torch.stack([
        support_embs[support_labels == lbl].mean(0) for lbl in unique_labels
    ])
    return F.normalize(meta_embs, p=2, dim=1), unique_labels


def mine_hard_pairs(
    pos_dists: torch.Tensor,
    neg_dists: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eq. 4 — positive selected iff  d_pos < max(d_neg) + m
    Eq. 5 — negative selected iff  d_neg > min(d_pos) - m
    """
    hard_pos_mask = pos_dists < neg_dists.max() + MARGIN
    hard_neg_mask = neg_dists > pos_dists.min() - MARGIN
    return hard_pos_mask, hard_neg_mask


def w2ml_loss(
    query_embs:   torch.Tensor,
    query_labels: torch.Tensor,
    meta_embs:    torch.Tensor,
    meta_labels:  torch.Tensor,
) -> torch.Tensor:
    """
    Eq. 8 — Episode loss averaged over all l query samples.

    Loss is expressed in DISTANCE space (d = 1 − cosine_similarity).

      Positive term: (1/α) · log(1 + Σ_P exp(+α(d_p − γ)))
      Negative term: (1/β) · log(1 + Σ_N exp(−β(d_n − γ)))
    """
    MAX_EXP = 80.0
    dist_mat = 1.0 - torch.mm(query_embs, meta_embs.t())   # (Q, N)

    per_query_losses = []

    for q_idx in range(query_embs.size(0)):
        q_lbl    = query_labels[q_idx]
        dists    = dist_mat[q_idx]
        pos_mask = meta_labels == q_lbl
        neg_mask = ~pos_mask

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        pos_dists = dists[pos_mask]
        neg_dists = dists[neg_mask]

        hp_mask, hn_mask = mine_hard_pairs(pos_dists, neg_dists)
        if hp_mask.sum() == 0 or hn_mask.sum() == 0:
            continue

        hp = pos_dists[hp_mask]
        hn = neg_dists[hn_mask]

        pos_exp  = torch.clamp(+ALPHA * (hp - GAMMA), max=MAX_EXP)
        pos_term = (1.0 / ALPHA) * torch.log1p(torch.exp(pos_exp).sum())

        neg_exp  = torch.clamp(-BETA  * (hn - GAMMA), max=MAX_EXP)
        neg_term = (1.0 / BETA)  * torch.log1p(torch.exp(neg_exp).sum())

        per_query_losses.append(pos_term + neg_term)

    if not per_query_losses:
        return dist_mat.sum() * 0.0

    return torch.stack(per_query_losses).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(
    model:   nn.Module,
    dataset: CASIAMSDataset,
    device:  torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    model.eval()
    all_embs, all_labels = [], []
    for imgs, labels, *_ in tqdm(loader, desc='  Extracting', leave=False):
        all_embs.append(model(imgs.to(device)).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_embs), np.concatenate(all_labels).astype(np.int32)


def identification(embs: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    g_embs, g_labs, p_embs, p_labs = [], [], [], []

    for lbl in unique_labels:
        idxs = np.where(labels == lbl)[0]
        g_embs.append(embs[idxs[0]]);  g_labs.append(lbl)
        for i in idxs[1:]:
            p_embs.append(embs[i]);  p_labs.append(lbl)

    g_embs = np.array(g_embs);  g_labs = np.array(g_labs)
    p_embs = np.array(p_embs);  p_labs = np.array(p_labs)

    preds = g_labs[np.argmax(p_embs @ g_embs.T, axis=1)]
    return float((preds == p_labs).mean())


def compute_eer(genuine: np.ndarray, imposter: np.ndarray) -> float:
    thresholds = np.linspace(
        min(genuine.min(), imposter.min()),
        max(genuine.max(), imposter.max()),
        1000,
    )
    far = np.array([(imposter <= t).mean() for t in thresholds])
    frr = np.array([(genuine  >  t).mean() for t in thresholds])
    idx = np.argmin(np.abs(far - frr))
    return float((far[idx] + frr[idx]) / 2.0)


def verification(embs: np.ndarray, labels: np.ndarray,
                 max_imp: int = 200_000) -> float:
    rng   = np.random.default_rng(42)
    l2idx = defaultdict(list)
    for i, lbl in enumerate(labels):
        l2idx[int(lbl)].append(i)

    genuine = []
    for idxs in l2idx.values():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                genuine.append(1.0 - float(embs[idxs[i]] @ embs[idxs[j]]))
    genuine = np.array(genuine, dtype=np.float32)

    uniq    = list(l2idx.keys())
    n_imp   = min(max_imp, len(genuine))
    imposter = []
    for _ in range(n_imp * 5):
        if len(imposter) >= n_imp:
            break
        a, b = rng.choice(uniq, 2, replace=False)
        ia   = rng.choice(l2idx[a])
        ib   = rng.choice(l2idx[b])
        imposter.append(1.0 - float(embs[ia] @ embs[ib]))
    imposter = np.array(imposter[:n_imp], dtype=np.float32)

    return compute_eer(genuine, imposter)


def evaluate(
    model:   nn.Module,
    dataset: CASIAMSDataset,
    device:  torch.device,
    tag:     str = '',
) -> Dict[str, float]:
    embs, labels = extract_features(model, dataset, device)
    acc = identification(embs, labels)
    eer = verification(embs, labels)
    prefix = f"[{tag}] " if tag else ""
    print(f"  {prefix}Acc={acc*100:.2f}%  EER={eer*100:.2f}%")
    return {'accuracy': acc, 'eer': eer}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(
    model:   nn.Module,
    sampler: EpisodeSampler,
    device:  torch.device,
) -> torch.Tensor:
    s_imgs, s_labels, q_imgs, q_labels = sampler.sample_episode()
    s_imgs, s_labels = s_imgs.to(device), s_labels.to(device)
    q_imgs, q_labels = q_imgs.to(device), q_labels.to(device)

    s_embs = model(s_imgs)
    q_embs = model(q_imgs)
    meta_embs, meta_labels = build_meta_support_sets(s_embs, s_labels)
    return w2ml_loss(q_embs, q_labels, meta_embs, meta_labels)


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\nW2ML Palmprint Recognition")
    print(f"  Device    : {device}")
    print(f"  Backbone  : {BACKBONE}")
    print(f"  Protocol  : {EVAL_PROTOCOL}")
    print(f"  Q_PER_CLASS (fixed) : {Q_PER_CLASS}")
    print(f"  K will be computed dynamically from training data\n")

    # ── Build datasets ───────────────────────────────────────────────────
    train_ids, test_ids = split_subjects(DATA_ROOT, TRAIN_RATIO)
    print(f"Subjects:  {len(train_ids)} train identities / "
          f"{len(test_ids)} test identities  "
          f"(ratio={TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%})")

    if EVAL_PROTOCOL == 'cross_subject':
        tr_spec = ALL_SPECTRA
        te_spec = ALL_SPECTRA
    elif EVAL_PROTOCOL == 'cross_spectrum':
        tr_spec = TRAIN_SPECTRA
        te_spec = TEST_SPECTRA
        print(f"  Train spectra : {tr_spec}")
        print(f"  Test  spectra : {te_spec}")
    else:
        raise ValueError(f"Unknown EVAL_PROTOCOL: {EVAL_PROTOCOL}")

    # Eval datasets (no augmentation)
    train_dataset = CASIAMSDataset(DATA_ROOT, train_ids, tr_spec,
                                   get_transforms(train=False))
    test_dataset  = CASIAMSDataset(DATA_ROOT, test_ids,  te_spec,
                                   get_transforms(train=False))

    # Augmented dataset used only during training episodes
    train_dataset_aug = CASIAMSDataset(DATA_ROOT, train_ids, tr_spec,
                                       get_transforms(train=True))

    print(f"Samples:   {len(train_dataset)} train / {len(test_dataset)} test\n")

    # ── Print per-identity image counts ─────────────────────────────────
    # Uses the non-augmented dataset (same file list, no transform difference)
    print_identity_counts(train_dataset, tag='TRAIN')
    print_identity_counts(test_dataset,  tag='TEST')

    # ── Compute K dynamically ────────────────────────────────────────────
    # K = min images per identity in the AUGMENTED training set - Q_PER_CLASS
    # (aug dataset has identical files to train_dataset, just different transforms)
    K = compute_dynamic_k(train_dataset_aug, Q_PER_CLASS)
    print(f"Dynamic K  : {K}  (= min_per_identity - Q_PER_CLASS = "
          f"min_per_identity - {Q_PER_CLASS})")
    print(f"Images per episode class : {K} support + {Q_PER_CLASS} query = "
          f"{K + Q_PER_CLASS} total\n")

    train_sampler = EpisodeSampler(train_dataset_aug, k=K)

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(BACKBONE).to(device)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_frozen    = n_total - n_trainable

    print(f"Backbone   : {BACKBONE}")
    if BACKBONE == 'resnet18':
        print(f"  Frozen modules  : {ResNet18Model.frozen_module_names()}")
        print(f"  Unfrozen modules: layer4  (lr={LR_LAYER4:.1e}), "
              f"head  (lr={LR_HEAD:.1e})")
    print(f"Params     : {n_trainable:,} trainable  /  "
          f"{n_frozen:,} frozen  /  {n_total:,} total\n")

    optimizer = Adam(model.param_groups(LR), weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)
    os.makedirs(SAVE_DIR, exist_ok=True)

    start_epoch = 0
    best_eer    = float('inf')
    best_acc    = 0.0

    if RESUME and os.path.isfile(RESUME):
        ckpt = torch.load(RESUME, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        best_eer    = ckpt.get('best_eer', float('inf'))
        best_acc    = ckpt.get('best_acc', 0.0)
        print(f"Resumed from {RESUME}  (epoch {start_epoch})\n")

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for ep_idx in range(EPISODES_PER_EPOCH):
            optimizer.zero_grad()
            loss = run_episode(model, train_sampler, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()

            if (ep_idx + 1) % LOG_INTERVAL == 0:
                avg = epoch_loss / (ep_idx + 1)
                lr  = scheduler.get_last_lr()[0]
                print(f"  Ep {epoch+1:03d} [{ep_idx+1:4d}/{EPISODES_PER_EPOCH}]"
                      f"  loss={avg:.4f}  lr={lr:.2e}")

        scheduler.step()
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch+1:03d}/{NUM_EPOCHS}  "
              f"avg_loss={epoch_loss/EPISODES_PER_EPOCH:.4f}  "
              f"time={elapsed:.0f}s")

        print("  Evaluating train set ...")
        tr_metrics = evaluate(model, train_dataset, device, tag='train')

        print("  Evaluating test set ...")
        te_metrics = evaluate(model, test_dataset,  device, tag=EVAL_PROTOCOL)

        tr_acc, tr_eer = tr_metrics['accuracy'], tr_metrics['eer']
        te_acc, te_eer = te_metrics['accuracy'], te_metrics['eer']

        print(f"  Summary  │  Train: Acc={tr_acc*100:.2f}%  EER={tr_eer*100:.2f}%"
              f"  │  Test:  Acc={te_acc*100:.2f}%  EER={te_eer*100:.2f}%")

        if te_eer < best_eer or (te_eer == best_eer and te_acc > best_acc):
            best_eer, best_acc = te_eer, te_acc
            torch.save({
                'epoch':     epoch + 1,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_eer':  best_eer,
                'best_acc':  best_acc,
                'K':         K,
                'backbone':  BACKBONE,
            }, os.path.join(SAVE_DIR, 'best.pth'))
            print(f"  ✓ Best checkpoint saved  "
                  f"(Test EER={best_eer*100:.2f}%  Test Acc={best_acc*100:.2f}%)")

        torch.save({
            'epoch':    epoch + 1,
            'model':    model.state_dict(),
            'best_eer': best_eer,
            'best_acc': best_acc,
            'K':        K,
            'backbone': BACKBONE,
        }, os.path.join(SAVE_DIR, 'latest.pth'))
        print()

    # ── Final per-spectrum breakdown (cross_spectrum only) ───────────────
    if EVAL_PROTOCOL == 'cross_spectrum':
        print("\n── Per-spectrum breakdown (test subjects) ──")
        ckpt = torch.load(os.path.join(SAVE_DIR, 'best.pth'), map_location=device)
        model.load_state_dict(ckpt['model'])
        for spec in TEST_SPECTRA:
            ds = CASIAMSDataset(DATA_ROOT, test_ids, [spec],
                                get_transforms(train=False))
            if len(ds) == 0:
                print(f"  [{spec}] no samples — skip"); continue
            evaluate(model, ds, device, tag=f'spectrum={spec}')

    print(f"\nDone.  Best Test EER={best_eer*100:.2f}%  "
          f"Best Test Acc={best_acc*100:.2f}%  K={K}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
