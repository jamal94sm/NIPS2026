"""
dataset.py — Dataset parsers, train/test splits, and PyTorch Dataset classes.

Supported datasets
------------------
  CASIA-MS   : {subjectID}_{handSide}_{spectrum}_{iter}.jpg
                 identity = subjectID + "_" + handSide
  Smartphone : {ID}/roi_square/{ID}_{hand}_{condition}.jpg
                 identity = ID + "_" + hand
  MPDv2      : {subject}_{session}_{device}_{handSide}_{iter}.jpg
                 identity = subject + "_" + handSide
"""

import os
import math
import random
from collections import defaultdict

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


# ══════════════════════════════════════════════════════════════
#  Normalisation (from CO3Net/models/dataset.py)
# ══════════════════════════════════════════════════════════════

class NormSingleROI:
    """Normalise non-black pixels to zero mean, unit std."""

    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        c, h, w = tensor.size()
        tensor  = tensor.view(c, h * w)
        idx     = tensor > 0
        t       = tensor[idx]
        m, s    = t.mean(), t.std()
        tensor[idx] = t.sub_(m).div_(s + 1e-6)
        tensor  = tensor.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, self.outchannels, dim=0)
        return tensor


# ══════════════════════════════════════════════════════════════
#  CASIA-MS parser
# ══════════════════════════════════════════════════════════════

def parse_casia_ms(data_root, n_subjects=190, n_total_samples=2776, seed=42):
    """
    Select n_subjects identities, each with near-uniform sample counts,
    distributed evenly across spectra.
    Returns {identity_key: [path, …]}
    """
    rng = random.Random(seed)

    id_spec = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".bmp", ".png")):
            continue
        stem  = os.path.splitext(fname)[0]
        parts = stem.split("_")
        if len(parts) < 4:
            continue
        identity = parts[0] + "_" + parts[1]
        spectrum = parts[2]
        id_spec[identity][spectrum].append(os.path.join(data_root, fname))

    all_ids = sorted(id_spec.keys())
    if n_subjects > len(all_ids):
        raise ValueError(f"Requested {n_subjects} subjects but only "
                         f"{len(all_ids)} available in {data_root}.")

    selected     = sorted(rng.sample(all_ids, n_subjects))
    base_per_id  = n_total_samples // n_subjects
    rem_ids      = n_total_samples %  n_subjects

    id_list = list(selected)
    rng.shuffle(id_list)
    id_target = {ident: base_per_id + (1 if i < rem_ids else 0)
                 for i, ident in enumerate(id_list)}

    id2paths     = {}
    actual_total = 0
    for ident in selected:
        target    = id_target[ident]
        spec_list = list(sorted(id_spec[ident].keys()))
        rng.shuffle(spec_list)
        n_spec       = len(spec_list)
        base_per_spec = target // n_spec
        rem_spec      = target %  n_spec
        chosen = []
        for j, sp in enumerate(spec_list):
            k = base_per_spec + (1 if j < rem_spec else 0)
            k = min(k, len(id_spec[ident][sp]))
            chosen.extend(rng.sample(id_spec[ident][sp], k))
        id2paths[ident]  = chosen
        actual_total    += len(chosen)

    counts = [len(v) for v in id2paths.values()]
    print(f"  [CASIA-MS] Selected={len(id2paths)}  "
          f"Total={actual_total}  "
          f"Per-ID min/max/mean={min(counts)}/{max(counts)}"
          f"/{sum(counts)/len(counts):.1f}")
    return id2paths


# ══════════════════════════════════════════════════════════════
#  Smartphone parser
# ══════════════════════════════════════════════════════════════

def parse_smartphone_data(data_root):
    """
    Structure: {data_root}/{ID}/roi_square/{ID}_{hand}_{condition}.jpg
    Identity  : ID + "_" + hand  (e.g. "1_left")
    Returns {identity_key: [path, …]}
    """
    id2paths = defaultdict(list)
    for subject_id in sorted(os.listdir(data_root)):
        roi_dir = os.path.join(data_root, subject_id, "roi_square")
        if not os.path.isdir(roi_dir):
            continue
        for fname in sorted(os.listdir(roi_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".bmp", ".png")):
                continue
            parts    = os.path.splitext(fname)[0].split("_")
            if len(parts) < 3:
                continue
            identity = parts[0] + "_" + parts[1]
            id2paths[identity].append(os.path.join(roi_dir, fname))
    return dict(id2paths)


# ══════════════════════════════════════════════════════════════
#  MPDv2 parser
# ══════════════════════════════════════════════════════════════

def parse_mpd_data(data_root, n_subjects=190, n_total_samples=2850, seed=42):
    """
    Select n_subjects identities from MPDv2, each with exactly
    ceil(n_total_samples / n_subjects) = 15 images, split 7/8 or 8/7
    across devices 'h' and 'm'.

    Eligibility: both devices must have ≥7 images, with at least one
    device having ≥8 (so a 7+8=15 split is always possible).

    Filename: {subject}_{session}_{device}_{handSide}_{iter}.jpg
    Identity: subject + "_" + handSide  (e.g. "191_l")
    """
    rng = random.Random(seed)

    images_per_id = math.ceil(n_total_samples / n_subjects)  # 15

    id_dev = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".bmp", ".png")):
            continue
        stem  = os.path.splitext(fname)[0]
        parts = stem.split("_")
        if len(parts) != 5:
            continue
        subject, session, device, hand_side, iteration = parts
        if device not in ("h", "m") or hand_side not in ("l", "r"):
            continue
        identity = subject + "_" + hand_side
        id_dev[identity][device].append(os.path.join(data_root, fname))

    def _qualifies(devs):
        h = len(devs.get("h", []))
        m = len(devs.get("m", []))
        return (h >= 8 and m >= 7) or (h >= 7 and m >= 8)

    eligible = {ident: devs for ident, devs in id_dev.items()
                if _qualifies(devs)}

    print(f"  [MPDv2] Eligible IDs (≥7 one device, ≥8 other): "
          f"{len(eligible)} / {len(id_dev)}")

    if n_subjects > len(eligible):
        raise ValueError(
            f"Requested {n_subjects} IDs but only {len(eligible)} qualify "
            f"in {data_root}. Lower n_mpd_subjects in CONFIG.")

    selected = sorted(rng.sample(sorted(eligible.keys()), n_subjects))

    id2paths     = {}
    actual_total = 0
    for ident in selected:
        devs  = eligible[ident]
        h_cnt = len(devs["h"])
        m_cnt = len(devs["m"])
        if h_cnt > m_cnt:
            alloc = {"h": 8, "m": 7}
        elif m_cnt > h_cnt:
            alloc = {"h": 7, "m": 8}
        else:
            alloc = ({"h": 8, "m": 7} if rng.random() < 0.5
                     else {"h": 7, "m": 8})
        chosen = []
        for device, k in alloc.items():
            chosen.extend(rng.sample(devs[device], k))
        id2paths[ident]  = chosen
        actual_total    += len(chosen)

    counts = [len(v) for v in id2paths.values()]
    print(f"  [MPDv2] Selected={len(id2paths)}  "
          f"Total={actual_total}  "
          f"Per-ID min/max/mean={min(counts)}/{max(counts)}"
          f"/{sum(counts)/len(counts):.1f}")
    return id2paths


# ══════════════════════════════════════════════════════════════
#  get_parser — factory
# ══════════════════════════════════════════════════════════════

def get_parser(dataset_name, cfg):
    """
    Returns a zero-argument callable that parses the requested dataset
    and returns an id2paths dict.
    """
    name = dataset_name.strip().lower().replace("-", "").replace("_", "")
    seed = cfg["random_seed"]

    if name == "casiams":
        return lambda: parse_casia_ms(
            cfg["casiams_data_root"],
            n_subjects=cfg["n_casia_subjects"],
            n_total_samples=cfg["n_casia_samples"],
            seed=seed)

    elif name == "smartphone":
        return lambda: parse_smartphone_data(cfg["smartphone_data_root"])

    elif name == "mpdv2":
        return lambda: parse_mpd_data(
            cfg["mpd_data_root"],
            n_subjects=cfg["n_mpd_subjects"],
            n_total_samples=cfg["n_mpd_samples"],
            seed=seed)

    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'. "
                         f"Choose 'CASIA-MS', 'Smartphone', or 'MPDv2'.")


# ══════════════════════════════════════════════════════════════
#  Splits
# ══════════════════════════════════════════════════════════════

def split_same_dataset(id2paths, train_subject_ratio=0.70,
                       gallery_ratio=0.50, seed=42):
    """
    Split a single dataset by subject:
      train_subject_ratio of subjects → training
      rest of subjects → gallery + probe
    Returns (train_samples, gallery_samples, probe_samples,
             train_label_map, test_label_map)
    """
    rng        = random.Random(seed)
    identities = sorted(id2paths.keys())
    rng.shuffle(identities)

    n_train   = max(1, int(len(identities) * train_subject_ratio))
    train_ids = identities[:n_train]
    test_ids  = identities[n_train:]

    train_label_map = {k: i for i, k in enumerate(train_ids)}
    test_label_map  = {k: i for i, k in enumerate(test_ids)}

    train_samples = [(p, train_label_map[ident])
                     for ident in train_ids
                     for p in id2paths[ident]]

    gallery_samples, probe_samples = [], []
    for ident in test_ids:
        paths = list(id2paths[ident])
        rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gal]:
            gallery_samples.append((p, test_label_map[ident]))
        for p in paths[n_gal:]:
            probe_samples.append((p, test_label_map[ident]))

    return (train_samples, gallery_samples, probe_samples,
            train_label_map, test_label_map)


def split_cross_dataset_test(id2paths, gallery_ratio=0.5, seed=42):
    """
    Split a test dataset (all subjects) into gallery and probe.
    Returns (gallery_samples, probe_samples, label_map)
    """
    rng       = random.Random(seed)
    label_map = {k: i for i, k in enumerate(sorted(id2paths.keys()))}
    gallery_samples, probe_samples = [], []

    for ident, paths in id2paths.items():
        paths = list(paths)
        rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gal]:
            gallery_samples.append((p, label_map[ident]))
        for p in paths[n_gal:]:
            probe_samples.append((p, label_map[ident]))

    return gallery_samples, probe_samples, label_map


# ══════════════════════════════════════════════════════════════
#  PyTorch Dataset classes
# ══════════════════════════════════════════════════════════════

class PairedDataset(Dataset):
    """
    Paired dataset for training with SupConLoss.
    Each sample returns (img1, img2) of the same identity.
    augment_factor > 1 repeats the dataset K times with fresh augmentation.
    """

    def __init__(self, samples, img_side=128, train=True, augment_factor=1):
        super().__init__()
        self.samples        = samples
        self.train          = train
        self.augment_factor = augment_factor if train else 1

        self.label2idxs = defaultdict(list)
        for i, (_, lab) in enumerate(samples):
            self.label2idxs[lab].append(i)

        self.aug_transform = T.Compose([
            T.Resize(img_side),
            T.RandomChoice([
                T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                T.RandomResizedCrop(img_side, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                T.RandomPerspective(distortion_scale=0.15, p=1),
                T.RandomChoice([
                    T.RandomRotation(10, interpolation=Image.BICUBIC,
                                     expand=False, center=(0.5*img_side, 0.0)),
                    T.RandomRotation(10, interpolation=Image.BICUBIC,
                                     expand=False, center=(0.0, 0.5*img_side)),
                ]),
            ]),
            T.ToTensor(),
            NormSingleROI(outchannels=1),
        ])
        self.clean_transform = T.Compose([
            T.Resize(img_side),
            T.ToTensor(),
            NormSingleROI(outchannels=1),
        ])

    def __len__(self):
        return len(self.samples) * self.augment_factor

    def __getitem__(self, index):
        real_idx     = index % len(self.samples)
        path1, label = self.samples[real_idx]

        if self.train:
            idxs = self.label2idxs[label]
            idx2 = real_idx
            while idx2 == real_idx and len(idxs) > 1:
                idx2 = random.choice(idxs)
            path2 = self.samples[idx2][0]
            img1  = self.aug_transform(Image.open(path1).convert("L"))
            img2  = self.aug_transform(Image.open(path2).convert("L"))
        else:
            img1  = self.clean_transform(Image.open(path1).convert("L"))
            img2  = img1.clone()

        return [img1, img2], label


class SingleDataset(Dataset):
    """Single-image dataset for gallery / probe evaluation."""

    def __init__(self, samples, img_side=128):
        super().__init__()
        self.samples   = samples
        self.transform = T.Compose([
            T.Resize(img_side),
            T.ToTensor(),
            NormSingleROI(outchannels=1),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(Image.open(path).convert("L")), label
