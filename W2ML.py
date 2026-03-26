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

FIX (data loading):
  Spectrum names are now normalised to lowercase at parse time.
  ALL_SPECTRA / TRAIN_SPECTRA / TEST_SPECTRA use lowercase 'white'.
  A startup diagnostic verifies every expected spectrum is present on disk
  and warns about any unexpected ones, preventing silent under-loading.
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
DATA_ROOT = "/home/pai-ng/Jamal/CASIA-MS-ROI"
SAVE_DIR  = "checkpoints"
RESUME    = None

# ── Evaluation protocol ───────────────────────────────────────────────────────
EVAL_PROTOCOL  = 'cross_subject'
TRAIN_RATIO    = 0.8

# ── Spectrum lists — all lowercase so they match parse_casia_filename output ──
#    parse_casia_filename does  spectrum = parts[-2].lower()
#    so 'White', 'WHITE', 'white' in filenames all map to 'white' here.
ALL_SPECTRA    = ['460', '630', '700', '850', '940', 'wht']   # ← lowercase; 'WHT' in filenames → 'wht'
TRAIN_SPECTRA  = ['460', '630', '700']
TEST_SPECTRA   = ['850', '940', 'wht']                         # ← lowercase

# Expected dataset shape — used by the startup diagnostic
EXPECTED_IDENTITIES  = 200   # unique subjectID × side pairs
EXPECTED_SPECTRA     = 6     # entries in ALL_SPECTRA
EXPECTED_ITERATIONS  = 6     # captures per (identity, spectrum)
EXPECTED_PER_IDENTITY = EXPECTED_SPECTRA * EXPECTED_ITERATIONS  # 36

# ── Image ─────────────────────────────────────────────────────────────────────
IMG_SIZE = 128

# ── Episode sampling ──────────────────────────────────────────────────────────
N           = 32
Q_PER_CLASS = 5

EPISODES_PER_EPOCH = 200

# ── Model ─────────────────────────────────────────────────────────────────────
BACKBONE   = 'custom'
EMBED_DIM  = 128

LR_LAYER4  = 2e-5
LR_HEAD    = 2e-4

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
#  SECTION 0 — STARTUP DIAGNOSTIC
#  Scans DATA_ROOT once and verifies the dataset matches expectations.
#  Catches spectrum name mismatches, missing files, etc. before training starts.
# ═══════════════════════════════════════════════════════════════════════════════

def run_dataset_diagnostic(root: str) -> None:
    """
    One-pass scan of DATA_ROOT.  Prints:
      • Total images found / skipped
      • Spectra actually present on disk  vs  ALL_SPECTRA
      • Identities found  vs  EXPECTED_IDENTITIES
      • Images per identity: min / median / max  vs  EXPECTED_PER_IDENTITY
      • Any spectrum in ALL_SPECTRA with zero images  ← catches case bugs
      • Any spectrum found on disk NOT in ALL_SPECTRA ← catches typos
    Raises RuntimeError if a critical mismatch is detected.
    """
    from collections import Counter
    print(f"\n{'═'*60}")
    print(f"  DATASET DIAGNOSTIC  —  {root}")
    print(f"{'═'*60}")

    found_spectra:   Counter = Counter()
    identity_counts: Dict[str, int] = defaultdict(int)
    skipped = 0

    for fname in sorted(os.listdir(root)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        parsed = _parse_raw(fname)     # internal helper, no filtering
        if parsed is None:
            skipped += 1
            continue
        subject, side, spectrum, _ = parsed
        found_spectra[spectrum] += 1
        identity_counts[f"{subject}_{side}"] += 1

    total = sum(found_spectra.values())
    print(f"  Total images parsed  : {total:,}")
    print(f"  Skipped (bad names)  : {skipped}")
    print(f"  Unique identities    : {len(identity_counts)}  "
          f"(expected {EXPECTED_IDENTITIES})")

    # ── Spectrum audit ────────────────────────────────────────────────────
    expected_set = set(ALL_SPECTRA)
    found_set    = set(found_spectra.keys())

    missing_from_disk = expected_set - found_set
    extra_on_disk     = found_set    - expected_set

    print(f"\n  Spectra found on disk : {sorted(found_set)}")
    print(f"  ALL_SPECTRA (config)  : {sorted(expected_set)}")

    if missing_from_disk:
        # This is the critical failure mode: spectrum exists in config but not
        # on disk → those images will be silently excluded from every dataset.
        raise RuntimeError(
            f"\n  ✗ CRITICAL — spectra in ALL_SPECTRA with ZERO images on disk:\n"
            f"      {sorted(missing_from_disk)}\n"
            f"  Likely cause: case mismatch between filenames and ALL_SPECTRA.\n"
            f"  Filenames use: {sorted(found_set)}\n"
            f"  Fix: ensure ALL_SPECTRA matches exactly, or update the filenames."
        )
    if extra_on_disk:
        print(f"  ⚠  Spectra on disk NOT in ALL_SPECTRA (will be ignored): "
              f"{sorted(extra_on_disk)}")

    print(f"\n  Images per spectrum:")
    for spec in sorted(found_spectra):
        marker = "✓" if spec in expected_set else "⚠"
        print(f"    {marker}  {spec:<10}  {found_spectra[spec]:,}")

    # ── Per-identity counts ───────────────────────────────────────────────
    counts = sorted(identity_counts.values())
    n      = len(counts)
    median = counts[n // 2]
    mean   = sum(counts) / n if n else 0
    print(f"\n  Images per identity:")
    print(f"    Min    : {counts[0]}  "
          f"{'✓' if counts[0] == EXPECTED_PER_IDENTITY else f'⚠ expected {EXPECTED_PER_IDENTITY}'}")
    print(f"    Median : {median}")
    print(f"    Mean   : {mean:.1f}")
    print(f"    Max    : {counts[-1]}")

    if counts[0] < EXPECTED_PER_IDENTITY:
        low = [(ident, cnt) for ident, cnt in identity_counts.items()
               if cnt < EXPECTED_PER_IDENTITY]
        print(f"\n  ⚠  {len(low)} identities below {EXPECTED_PER_IDENTITY} images:")
        for ident, cnt in sorted(low, key=lambda x: x[1])[:20]:
            print(f"      {ident:<20}  {cnt}")
        if len(low) > 20:
            print(f"      … and {len(low)-20} more")
    else:
        print(f"\n  ✓ All identities have {EXPECTED_PER_IDENTITY} images each.")

    print(f"{'═'*60}\n")


def _parse_raw(fname: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Internal helper used only by the diagnostic.
    Parses without any spectrum filtering, so we can see the raw on-disk names.
    Spectrum is still normalised to lowercase (same as parse_casia_filename).
    """
    name  = os.path.splitext(fname)[0]
    parts = name.split('_')
    if len(parts) < 4:
        return None
    iteration = parts[-1]
    spectrum  = parts[-2].lower()   # ← normalise here too
    side      = parts[-3]
    subject   = '_'.join(parts[:-3])
    return subject, side, spectrum, iteration


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

    NOTE: spectrum is normalised to lowercase so that 'White', 'WHITE', and
    'white' in filenames all resolve to 'white', matching ALL_SPECTRA entries.
    """
    name  = os.path.splitext(fname)[0]
    parts = name.split('_')
    if len(parts) < 4:
        return None
    iteration = parts[-1]
    spectrum  = parts[-2].lower()   # ← FIX: normalise to lowercase
    side      = parts[-3]
    subject   = '_'.join(parts[:-3])
    return subject, side, spectrum, iteration


def build_identity_index(root: str,
                         spectra: Optional[List[str]] = None
                         ) -> Dict[str, List[str]]:
    """
    Single-pass scan of DATA_ROOT.
    Returns  {identity_string: [list_of_full_file_paths]}
    where identity = '{subjectID}_{side}'.
    """
    # spectra list is already lowercase (from ALL_SPECTRA or caller);
    # parse_casia_filename also returns lowercase spectrum → safe comparison.
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
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"TRAIN_RATIO must be in (0, 1), got {train_ratio}")

    index = build_identity_index(root, spectra)
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
    counts: Dict[str, int] = defaultdict(int)
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
    print(f"  Min              : {values[0]}  "
          f"{'✓' if values[0] == EXPECTED_PER_IDENTITY else f'⚠ expected {EXPECTED_PER_IDENTITY}'}")
    print(f"  Median           : {values[n // 2]}")
    print(f"  Mean             : {sum(values) / n:.1f}")
    print(f"  Max              : {values[-1]}")

    bucket_size = max(1, (values[-1] - values[0]) // 10 + 1)
    buckets: Dict[int, int] = defaultdict(int)
    for v in values:
        buckets[(v // bucket_size) * bucket_size] += 1
    print(f"\n  Count distribution (bucket size = {bucket_size}):")
    for bucket_start in sorted(buckets):
        bar = '█' * buckets[bucket_start]
        print(f"    [{bucket_start:4d}–{bucket_start + bucket_size - 1:4d}]  "
              f"{bar}  ({buckets[bucket_start]})")

    sorted_by_count = sorted(counts.items(), key=lambda x: x[1])
    print(f"\n  {top_n_outliers} identities with fewest images:")
    for ident, cnt in sorted_by_count[:top_n_outliers]:
        print(f"    {ident:<20s}  {cnt:3d} images")

    print(f"\n  {top_n_outliers} identities with most images:")
    for ident, cnt in sorted_by_count[-top_n_outliers:][::-1]:
        print(f"    {ident:<20s}  {cnt:3d} images")

    print(f"{'─'*60}\n")


class CASIAMSDataset(Dataset):
    """
    CASIA-MS ROI dataset with optional spectrum filtering.
    Returns (img_tensor, int_label, spectrum_str) per sample.

    Spectrum strings are always lowercase (normalised in parse_casia_filename).
    Pass spectra as lowercase strings, e.g. ['460', 'white'] not ['White'].
    """

    def __init__(
        self,
        root:       str,
        identities: List[str],
        spectra:    Optional[List[str]] = None,
        transform=None,
    ):
        self.root      = root
        # Normalise to lowercase for safety even if caller passes mixed case
        self.spectra   = (set(s.lower() for s in spectra)
                          if spectra else set(ALL_SPECTRA))
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
            # spectrum is already lowercase from parse_casia_filename
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
    """
    Augmentation strategy for a small dataset (36 images / identity).

    Train pipeline — ordered from least to most destructive:
      1. Spatial : flip, rotation, affine shear  — preserve palmprint structure
      2. Photometric : jitter, blur, grayscale   — simulate multi-spectrum variation
      3. Tensor conversion + normalise
      4. RandomErasing x2 (independent patches)  — simulate partial occlusion

    Two-pass RandomErasing applies two independent random patches, giving
    stronger occlusion regularisation without needing extra libraries.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if train:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            # ── Spatial ──────────────────────────────────────────────────
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=20),
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),   # mild translation
                shear=8,                  # slight shear — realistic wrist angle
            ),
            T.RandomPerspective(distortion_scale=0.2, p=0.4),
            # ── Photometric ──────────────────────────────────────────────
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            T.RandomGrayscale(p=0.15),
            # ── Tensor + normalise ───────────────────────────────────────
            T.ToTensor(),
            T.Normalize(mean, std),
            # ── Occlusion regularisation (two independent patches) ───────
            T.RandomErasing(p=0.4, scale=(0.02, 0.12), ratio=(0.3, 3.0), value=0),
            T.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.0), value=0),
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

class _ResBlock(nn.Module):
    """
    Pre-activation residual block: BN → LeakyReLU → Conv → BN → LeakyReLU → Conv.

    Pre-activation (He et al. 2016) puts BN+activation before each conv so
    gradients flow cleanly through the skip connection without going through
    any non-linearity — important for a relatively shallow network like this.

    A 1×1 projection skip is added when stride > 1 or channels change.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),

            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if (stride != 1 or in_ch != out_ch) else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


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
            nn.Dropout(p=0.5),

            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            lrelu(),
            nn.Dropout(p=0.5),

            nn.Linear(512, EMBED_DIM, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.classifier(self.features(x)), p=2, dim=1)

    def param_groups(self, lr: float) -> list:
        return [{'params': self.parameters(), 'lr': lr}]

    def trainable_parameters(self):
        return list(self.parameters())


class ResNet18Model(nn.Module):
    _FROZEN_MODULES = ('conv1', 'bn1', 'relu', 'maxpool',
                       'layer1', 'layer2', 'layer3')

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        for name in self._FROZEN_MODULES:
            for param in getattr(base, name).parameters():
                param.requires_grad = False

        self.frozen_body = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3,
        )
        self.layer4  = base.layer4
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.head    = nn.Linear(512, EMBED_DIM, bias=False)

    def forward(self, x):
        x = self.frozen_body(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return F.normalize(self.head(x), p=2, dim=1)

    def param_groups(self, lr):
        return [
            {'params': self.layer4.parameters(), 'lr': LR_LAYER4, 'name': 'layer4'},
            {'params': self.head.parameters(),   'lr': LR_HEAD,   'name': 'head'},
        ]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @staticmethod
    def frozen_module_names():
        return ResNet18Model._FROZEN_MODULES


def build_model(backbone: str = BACKBONE) -> nn.Module:
    backbone = backbone.lower().strip()
    if backbone == 'custom':
        return W2MLModel()
    elif backbone == 'resnet18':
        return ResNet18Model()
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose 'custom' or 'resnet18'.")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — W2ML LOSS  (Equations 2–8)
# ═══════════════════════════════════════════════════════════════════════════════

def build_meta_support_sets(support_embs, support_labels):
    unique_labels = torch.unique(support_labels, sorted=True)
    meta_embs = torch.stack([
        support_embs[support_labels == lbl].mean(0) for lbl in unique_labels
    ])
    return F.normalize(meta_embs, p=2, dim=1), unique_labels


def mine_hard_pairs(pos_dists, neg_dists):
    hard_pos_mask = pos_dists < neg_dists.max() + MARGIN
    hard_neg_mask = neg_dists > pos_dists.min() - MARGIN
    return hard_pos_mask, hard_neg_mask


def w2ml_loss(query_embs, query_labels, meta_embs, meta_labels):
    MAX_EXP  = 80.0
    dist_mat = 1.0 - torch.mm(query_embs, meta_embs.t())

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

        neg_exp  = torch.clamp(-BETA * (hn - GAMMA), max=MAX_EXP)
        neg_term = (1.0 / BETA)  * torch.log1p(torch.exp(neg_exp).sum())

        per_query_losses.append(pos_term + neg_term)

    if not per_query_losses:
        return dist_mat.sum() * 0.0
    return torch.stack(per_query_losses).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    model.eval()
    all_embs, all_labels = [], []
    for imgs, labels, *_ in tqdm(loader, desc='  Extracting', leave=False):
        all_embs.append(model(imgs.to(device)).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_embs), np.concatenate(all_labels).astype(np.int32)


def identification(embs, labels):
    unique_labels = np.unique(labels)
    g_embs, g_labs, p_embs, p_labs = [], [], [], []
    for lbl in unique_labels:
        idxs = np.where(labels == lbl)[0]
        g_embs.append(embs[idxs[0]]);  g_labs.append(lbl)
        for i in idxs[1:]:
            p_embs.append(embs[i]);  p_labs.append(lbl)
    g_embs = np.array(g_embs);  g_labs = np.array(g_labs)
    p_embs = np.array(p_embs);  p_labs = np.array(p_labs)
    preds  = g_labs[np.argmax(p_embs @ g_embs.T, axis=1)]
    return float((preds == p_labs).mean())


def compute_eer(genuine, imposter):
    thresholds = np.linspace(
        min(genuine.min(), imposter.min()),
        max(genuine.max(), imposter.max()), 1000)
    far = np.array([(imposter <= t).mean() for t in thresholds])
    frr = np.array([(genuine  >  t).mean() for t in thresholds])
    idx = np.argmin(np.abs(far - frr))
    return float((far[idx] + frr[idx]) / 2.0)


def verification(embs, labels, max_imp=200_000):
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

    uniq     = list(l2idx.keys())
    n_imp    = min(max_imp, len(genuine))
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


def evaluate(model, dataset, device, tag=''):
    embs, labels = extract_features(model, dataset, device)
    acc = identification(embs, labels)
    eer = verification(embs, labels)
    prefix = f"[{tag}] " if tag else ""
    print(f"  {prefix}Acc={acc*100:.2f}%  EER={eer*100:.2f}%")
    return {'accuracy': acc, 'eer': eer}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(model, sampler, device):
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

    # ── Step 0: Run diagnostic before anything else ──────────────────────
    # This will raise RuntimeError early if a spectrum name mismatch is found,
    # rather than silently training on an incomplete dataset.
    run_dataset_diagnostic(DATA_ROOT)

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

    train_dataset     = CASIAMSDataset(DATA_ROOT, train_ids, tr_spec,
                                       get_transforms(train=False))
    test_dataset      = CASIAMSDataset(DATA_ROOT, test_ids,  te_spec,
                                       get_transforms(train=False))
    train_dataset_aug = CASIAMSDataset(DATA_ROOT, train_ids, tr_spec,
                                       get_transforms(train=True))

    print(f"Samples:   {len(train_dataset)} train / {len(test_dataset)} test\n")

    print_identity_counts(train_dataset, tag='TRAIN')
    print_identity_counts(test_dataset,  tag='TEST')

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
