"""
dataset.py
==========
Single source of truth for data collection, ID splitting, the 12
train/test settings, augmentation, and PyTorch Dataset classes shared by
every baseline.

This module merges the "identical to CompNet" data-handling blocks that
were duplicated across all 10 original NIPS2026 baseline scripts. Because
every method now reads splits from the same cached SPLITS_FILE with the
same seed, ALL BASELINES TRAIN/TEST ON THE EXACT SAME DATA SPLIT for a
given setting.
"""
import os
import json
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

import config as C

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# 10 paired-condition settings + S_scanner + S_scanner_to_persp = 12 settings total
PAIRED_CONDITIONS = [
    ("wet",  "text"),
    ("wet",  "rnd"),
    ("rnd",  "text"),
    ("sf",   "roll"),
    ("jf",   "pitch"),
    ("bf",   "far"),
    ("roll", "close"),
    ("far",  "jf"),
    ("fl",   "sf"),
    ("roll", "pitch"),
]


# ══════════════════════════════════════════════════════════════
#  NORMALISATION  (shared by every grayscale/ROI-based baseline)
# ══════════════════════════════════════════════════════════════

class NormSingleROI:
    """Per-sample z-score normalisation over nonzero pixels (identical
    across ccnet/co3net/compnet/ppnet/sf2net/palmbridge in the original
    code, up to a `numel() > 1` guard which we keep for safety)."""
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        c, h, w = tensor.size()
        tensor = tensor.view(c, h * w)
        idx = tensor > 0
        t = tensor[idx]
        if t.numel() > 1:
            tensor[idx] = t.sub_(t.mean()).div_(t.std() + 1e-6)
        tensor = tensor.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, self.outchannels, dim=0)
        return tensor


# ══════════════════════════════════════════════════════════════
#  AUGMENTATION  (unified "CompNet-style" transform used by every
#  baseline; see fairness discussion in config.py for augment_factor)
# ══════════════════════════════════════════════════════════════

def make_aug_transform(img_side, channels=1, normalize="roi"):
    """Canonical augmentation shared by all baselines: RandomChoice among
    ColorJitter / RandomResizedCrop / RandomPerspective / a nested choice
    of two axis-centered +/-10deg rotations.
    `normalize` selects the final normalisation step:
      "roi"      -> NormSingleROI (grayscale ROI baselines)
      "imagenet" -> ImageNet mean/std (RGB pretrained backbones)
      "unit"     -> [-1, 1] normalisation (ArcFace/MagFace RGB backbones)
    """
    resize = T.Resize((img_side, img_side)) if channels == 1 else T.Resize(img_side)
    core = [
        resize,
        T.RandomChoice([
            T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
            T.RandomResizedCrop(img_side, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            T.RandomPerspective(distortion_scale=0.15, p=1),
            T.RandomChoice([
                T.RandomRotation(10, interpolation=Image.BICUBIC,
                                 expand=False, center=(0.5 * img_side, 0.0)),
                T.RandomRotation(10, interpolation=Image.BICUBIC,
                                 expand=False, center=(0.0, 0.5 * img_side)),
            ]),
        ]),
        T.ToTensor(),
    ]
    return T.Compose(core + [_final_norm(normalize, channels)])


def make_eval_transform(img_side, channels=1, normalize="roi"):
    resize = T.Resize((img_side, img_side)) if channels == 1 else T.Resize(img_side)
    return T.Compose([resize, T.ToTensor(), _final_norm(normalize, channels)])


def _final_norm(normalize, channels):
    if normalize == "roi":
        return NormSingleROI(outchannels=channels)
    if normalize == "imagenet":
        return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if normalize == "unit":
        return T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    raise ValueError(normalize)


# ══════════════════════════════════════════════════════════════
#  DATA COLLECTION  (identical across every original baseline file)
# ══════════════════════════════════════════════════════════════

def collect_perspective(data_root):
    """condition -> identity -> [path, ...]"""
    cond_paths = defaultdict(lambda: defaultdict(list))
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir = os.path.join(data_root, subject_id)
        if not os.path.isdir(subject_dir):
            continue
        roi_dir = os.path.join(subject_dir, "roi_perspective")
        if not os.path.isdir(roi_dir):
            continue
        for fname in sorted(os.listdir(roi_dir)):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                continue
            parts = os.path.splitext(fname)[0].split("_")
            if len(parts) < 3:
                continue
            ident = parts[0] + "_" + parts[1].lower()
            cond = parts[2].lower()
            cond_paths[cond][ident].append(os.path.join(roi_dir, fname))
    return cond_paths


def collect_scanner(data_root, scanner_spectra):
    """identity -> [path, ...] restricted to the configured scanner spectra"""
    scanner_paths = defaultdict(list)
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir = os.path.join(data_root, subject_id)
        if not os.path.isdir(subject_dir):
            continue
        scan_dir = os.path.join(subject_dir, "roi_scanner")
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                continue
            parts = os.path.splitext(fname)[0].split("_")
            if len(parts) < 4:
                continue
            if parts[2].lower() not in scanner_spectra:
                continue
            ident = parts[0] + "_" + parts[1].lower()
            scanner_paths[ident].append(os.path.join(scan_dir, fname))
    return scanner_paths


def _all_samples(id2paths, label_map):
    return [(p, label_map[ident]) for ident, paths in id2paths.items() for p in paths]


def _gallery_probe_split(id2paths, label_map, gallery_ratio, rng):
    gallery, probe = [], []
    for ident, paths in id2paths.items():
        paths = list(paths)
        rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        n_gal = min(n_gal, len(paths) - 1)
        if len(paths) == 1:
            gallery.append((paths[0], label_map[ident]))
            probe.append((paths[0], label_map[ident]))
        else:
            for p in paths[:n_gal]:
                gallery.append((p, label_map[ident]))
            for p in paths[n_gal:]:
                probe.append((p, label_map[ident]))
    return gallery, probe


# ══════════════════════════════════════════════════════════════
#  DETERMINISTIC ID SPLITS  (shared by ALL methods and ALL 12 settings)
# ══════════════════════════════════════════════════════════════

def generate_all_splits(cond_paths, scanner_paths, train_id_ratio, seed):
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    all_persp_ids = sorted(persp_all.keys())
    scanner_ids = sorted(scanner_paths.keys())
    n_test = len(all_persp_ids) - int(len(all_persp_ids) * train_id_ratio)

    splits = {}
    n_test_scanner = min(n_test, len(scanner_ids))
    test_ids = sorted(random.Random(seed).sample(scanner_ids, n_test_scanner))
    train_ids = sorted(set(all_persp_ids) - set(test_ids))
    splits["S_scanner"] = {"train_ids": train_ids, "test_ids": test_ids}

    no_scanner_ids = sorted(set(all_persp_ids) - set(scanner_ids))
    splits["S_scanner_to_persp"] = {"train_ids": scanner_ids, "test_ids": no_scanner_ids}

    for cond_a, cond_b in PAIRED_CONDITIONS:
        paths_a = cond_paths.get(cond_a, {})
        paths_b = cond_paths.get(cond_b, {})
        eligible_ids = sorted(set(paths_a.keys()) & set(paths_b.keys()))
        if not eligible_ids:
            continue
        n_t = min(n_test, len(eligible_ids))
        test_ids = sorted(random.Random(seed).sample(eligible_ids, n_t))
        train_ids = sorted(set(all_persp_ids) - set(test_ids))
        splits[f"S_{cond_a}_{cond_b}"] = {"train_ids": train_ids, "test_ids": test_ids}
    return splits


def load_or_generate_splits(cond_paths, scanner_paths, train_id_ratio, seed):
    if os.path.exists(C.SPLITS_FILE):
        with open(C.SPLITS_FILE) as f:
            splits = json.load(f)
        print(f"  Loaded existing ID splits from: {C.SPLITS_FILE}")
    else:
        print(f"  Generating ID splits (seed={seed}) -> {C.SPLITS_FILE}")
        splits = generate_all_splits(cond_paths, scanner_paths, train_id_ratio, seed)
        with open(C.SPLITS_FILE, "w") as f:
            json.dump(splits, f, indent=2)
    for key, val in splits.items():
        print(f"    {key:<30}  train={len(val['train_ids'])}  test={len(val['test_ids'])}")
    return splits


# ══════════════════════════════════════════════════════════════
#  SETTING PARSERS  (build train / gallery / probe sample lists)
# ══════════════════════════════════════════════════════════════

def parse_setting_scanner(cond_paths, scanner_paths, splits, gallery_ratio, seed):
    rng = random.Random(seed)
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    train_ids, test_ids = splits["train_ids"], splits["test_ids"]
    train_label_map = {ident: i for i, ident in enumerate(train_ids)}
    test_label_map = {ident: i for i, ident in enumerate(test_ids)}
    train_samples = _all_samples({i: persp_all[i] for i in train_ids if i in persp_all}, train_label_map)
    gallery, probe = _gallery_probe_split(
        {i: scanner_paths[i] for i in test_ids if i in scanner_paths}, test_label_map, gallery_ratio, rng)
    return train_samples, gallery, probe, len(train_ids)


def parse_setting_scanner_to_perspective(cond_paths, scanner_paths, splits, gallery_ratio, seed):
    rng = random.Random(seed)
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    train_ids, test_ids = splits["train_ids"], splits["test_ids"]
    train_label_map = {ident: i for i, ident in enumerate(train_ids)}
    test_label_map = {ident: i for i, ident in enumerate(test_ids)}
    train_samples = _all_samples(scanner_paths, train_label_map)
    gallery, probe = _gallery_probe_split(
        {i: persp_all[i] for i in test_ids if i in persp_all}, test_label_map, gallery_ratio, rng)
    return train_samples, gallery, probe, len(train_ids)


def parse_setting_paired_conditions(cond_a, cond_b, cond_paths, scanner_paths, splits, seed):
    rng = random.Random(seed)
    paths_a = cond_paths.get(cond_a, {})
    paths_b = cond_paths.get(cond_b, {})
    train_ids, test_ids = splits["train_ids"], splits["test_ids"]
    train_label_map = {ident: i for i, ident in enumerate(train_ids)}
    test_label_map = {ident: i for i, ident in enumerate(test_ids)}

    train_samples = []
    for cond, cond_dict in cond_paths.items():
        if cond in (cond_a, cond_b):
            continue
        for ident in train_ids:
            for p in cond_dict.get(ident, []):
                train_samples.append((p, train_label_map[ident]))
    for ident in train_ids:
        for p in scanner_paths.get(ident, []):
            train_samples.append((p, train_label_map[ident]))

    gallery_samples, probe_samples = [], []
    for ident in test_ids:
        label = test_label_map[ident]
        a_imgs = list(paths_a.get(ident, [])); rng.shuffle(a_imgs)
        b_imgs = list(paths_b.get(ident, [])); rng.shuffle(b_imgs)
        if not a_imgs or not b_imgs:
            continue
        if rng.random() < 0.5:
            gallery_samples.append((a_imgs[0], label)); probe_samples.append((b_imgs[0], label))
        else:
            gallery_samples.append((b_imgs[0], label)); probe_samples.append((a_imgs[0], label))
    return train_samples, gallery_samples, probe_samples, len(train_ids)


def build_settings(cond_paths, scanner_paths, all_splits, gallery_ratio, seed):
    """Returns the list of 12 settings, each a dict with tag/label/train_desc/
    test_desc/parser. `parser()` returns (train_samples, gallery, probe, n_classes),
    IDENTICAL for every baseline that calls it for a given setting -> same split."""
    settings = []
    settings.append({
        "tag": "setting_scanner", "label": "S_scanner",
        "train_desc": "Perspective (train IDs)", "test_desc": "Scanner (test IDs)",
        "parser": lambda: parse_setting_scanner(
            cond_paths, scanner_paths, all_splits["S_scanner"], gallery_ratio, seed)})
    settings.append({
        "tag": "setting_scanner_to_persp", "label": "S_scanner_to_persp",
        "train_desc": "Scanner (all scanner IDs)", "test_desc": "Perspective (no-scanner IDs)",
        "parser": lambda: parse_setting_scanner_to_perspective(
            cond_paths, scanner_paths, all_splits["S_scanner_to_persp"], gallery_ratio, seed)})

    conditions_found = sorted(cond_paths.keys())
    for cond_a, cond_b in PAIRED_CONDITIONS:
        if cond_a not in conditions_found or cond_b not in conditions_found:
            continue
        split_key = f"S_{cond_a}_{cond_b}"
        if split_key not in all_splits:
            continue
        settings.append({
            "tag": f"setting_{cond_a}_{cond_b}", "label": split_key,
            "train_desc": f"Perspective(not {cond_a}/{cond_b})+Scanner",
            "test_desc": f"{cond_a}/{cond_b} (test IDs)",
            "parser": (lambda ca=cond_a, cb=cond_b: parse_setting_paired_conditions(
                ca, cb, cond_paths, scanner_paths, all_splits[f"S_{ca}_{cb}"], seed))})
    return settings


def get_settings():
    """Top-level convenience: scans the dataset once and returns the 12
    settings, shared identically by every method/run in benchmarking.py."""
    cond_paths = collect_perspective(C.DATA_ROOT)
    scanner_paths = collect_scanner(C.DATA_ROOT, C.SCANNER_SPECTRA)
    all_splits = load_or_generate_splits(cond_paths, scanner_paths, C.TRAIN_ID_RATIO, C.SEED)
    return build_settings(cond_paths, scanner_paths, all_splits, C.TEST_GALLERY_RATIO, C.SEED)


# ══════════════════════════════════════════════════════════════
#  GENERIC DATASET CLASSES
#  view_mode: "single" | "paired" | "triplet"  (see config.py METHODS)
# ══════════════════════════════════════════════════════════════

class SingleDataset(Dataset):
    """1 view per sample. Used for eval (all methods) and for train when
    view_mode == 'single' (arcface, magface, palmbridge, compnet, ppnet).
    `augment_factor` inflates dataset length so that, combined with a
    random `transform`, each image is revisited `augment_factor` times
    per epoch with an independent random augmentation draw."""
    def __init__(self, samples, transform, augment_factor=1, channels=1):
        self.samples = samples
        self.transform = transform
        self.augment_factor = max(1, augment_factor)
        self.channels = channels

    def __len__(self):
        return len(self.samples) * self.augment_factor

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        path, label = self.samples[real_idx]
        mode = "L" if self.channels == 1 else "RGB"
        img = Image.open(path).convert(mode)
        return self.transform(img), label


class PairedDataset(Dataset):
    """2 augmented views per __getitem__ for SupCon-style losses
    (ccnet, co3net, convnext, dino). Prefers a different real photo of the
    same identity for the second view; falls back to a second augmented
    draw of the same photo when the identity has only one sample."""
    def __init__(self, samples, transform, augment_factor=1, channels=1):
        self.samples = samples
        self.transform = transform
        self.augment_factor = max(1, augment_factor)
        self.channels = channels
        self.label2idxs = defaultdict(list)
        for i, (_, lab) in enumerate(samples):
            self.label2idxs[lab].append(i)

    def __len__(self):
        return len(self.samples) * self.augment_factor

    def _load(self, path):
        mode = "L" if self.channels == 1 else "RGB"
        return self.transform(Image.open(path).convert(mode))

    def __getitem__(self, index):
        real_idx = index % len(self.samples)
        path1, label = self.samples[real_idx]
        idxs = self.label2idxs[label]
        idx2 = real_idx
        while idx2 == real_idx and len(idxs) > 1:
            idx2 = random.choice(idxs)
        path2 = self.samples[idx2][0]
        return [self._load(path1), self._load(path2)], label


class TripletDataset(Dataset):
    """(anchor, positive, negative) per __getitem__ for triplet losses
    (sf2net). Structural floor of 3 touches/image regardless of
    augment_factor; see fairness discussion in config.py."""
    def __init__(self, samples, transform, augment_factor=1, channels=1):
        self.samples = samples
        self.transform = transform
        self.augment_factor = max(1, augment_factor)
        self.channels = channels
        self.labels = [lab for _, lab in samples]
        self.label2idxs = defaultdict(list)
        for i, (_, lab) in enumerate(samples):
            self.label2idxs[lab].append(i)

    def __len__(self):
        return len(self.samples) * self.augment_factor

    def _load(self, path):
        mode = "L" if self.channels == 1 else "RGB"
        return self.transform(Image.open(path).convert(mode))

    def __getitem__(self, index):
        real_idx = index % len(self.samples)
        path_a, label_a = self.samples[real_idx]

        pos_idxs = self.label2idxs[label_a]
        pos_idx = real_idx
        while pos_idx == real_idx and len(pos_idxs) > 1:
            pos_idx = random.choice(pos_idxs)
        path_p, label_p = self.samples[pos_idx]

        neg_candidates = [i for i, l in enumerate(self.labels) if l != label_a]
        neg_idx = random.choice(neg_candidates)
        path_n, label_n = self.samples[neg_idx]

        return ([self._load(path_a), self._load(path_p), self._load(path_n)],
                [label_a, label_p, label_n])


def make_loader(samples, method_name, train, batch_size, num_workers, drop_last=None):
    """Central factory: builds the right Dataset/DataLoader for a method
    based on its registry entry in config.METHODS, using make_aug_transform
    for train and make_eval_transform for eval."""
    from torch.utils.data import DataLoader
    m = C.METHODS[method_name]
    channels = m.get("channels", 1)
    img_side = m["img_side"]
    normalize = m.get("normalize", "roi")

    if train:
        transform = make_aug_transform(img_side, channels, normalize)
        augment_factor = m["augment_factor"]
        view_mode = m["view_mode"]
        cls = {"single": SingleDataset, "paired": PairedDataset,
               "triplet": TripletDataset}[view_mode]
        ds = cls(samples, transform, augment_factor=augment_factor, channels=channels)
    else:
        transform = make_eval_transform(img_side, channels, normalize)
        ds = SingleDataset(samples, transform, augment_factor=1, channels=channels)

    bs = min(batch_size, len(samples)) if len(samples) > 0 else batch_size
    if drop_last is None:
        drop_last = train and len(samples) > batch_size
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=num_workers,
                       pin_memory=True, drop_last=drop_last)
