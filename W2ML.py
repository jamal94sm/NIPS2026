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
from torchvision import models
import torchvision.transforms as T
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
#  PARAMETERS  —  edit this block only
# ═══════════════════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = "/home/pai-ng/Jamal/CASIA-MS-ROI"       # folder containing all ROI .jpg files
SAVE_DIR  = "checkpoints"         # where best.pth / latest.pth are written
RESUME    = None                  # path to a .pth checkpoint to resume from
                                  # e.g. "checkpoints/best.pth"

# ── Evaluation protocol ───────────────────────────────────────────────────────
# 'cross_subject'  : train on first 50 % of subjects, test on last 50 %
#                   (all 6 spectra used for both splits)
# 'cross_spectrum' : train on TRAIN_SPECTRA, test on TEST_SPECTRA
#                   (still with a 50/50 subject split)
EVAL_PROTOCOL  = 'cross_subject'

ALL_SPECTRA    = ['460', '630', '700', '850', '940', 'White']
TRAIN_SPECTRA  = ['460', '630', '700']   # used only for cross_spectrum
TEST_SPECTRA   = ['850', '940', 'White'] # used only for cross_spectrum

# ── Image ─────────────────────────────────────────────────────────────────────
IMG_SIZE = 224                    # paper uses 224 × 224 ROIs

# ── Episode sampling  (Table 1) ───────────────────────────────────────────────
N           = 32    # number of classes per episode          (paper: 32)
K           = 4     # support images per class               (paper: 4)
Q_PER_CLASS = 4     # query images per class  (not specified; match K)

EPISODES_PER_EPOCH = 500
VAL_EPISODES       = 200          # validation episodes (informational only)

# ── Model ─────────────────────────────────────────────────────────────────────
EMBED_DIM  = 128    # embedding dimensionality                (paper: 128)
PRETRAINED = True   # ImageNet init

# ── Loss hyper-parameters  (Section 3.3, optimal from Tables 5 & 6) ──────────
ALPHA  = 2.0        # positive weighting scale               (paper: 2,  fixed)
BETA   = 40.0       # negative weighting scale               (paper: 40, fixed)
GAMMA  = 0.5        # similarity margin for weighting        (paper: 0.5 optimal)
MARGIN = 0.05       # hard-mining margin m                   (paper: 0.05 optimal)

# ── Training ──────────────────────────────────────────────────────────────────
NUM_EPOCHS   = 60
LR           = 1e-4             # reduced: head-only training needs smaller LR
WEIGHT_DECAY = 1e-4
LR_STEP      = 20               # StepLR: decay every N epochs
LR_GAMMA     = 0.5
GRAD_CLIP    = 5.0

DEVICE      = 'cuda'            # 'cuda' or 'cpu'
NUM_WORKERS = 4
SEED        = 42
LOG_INTERVAL = 50               # print every N episodes


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATASET  (dataset.py)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_casia_filename(fname: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse  {subjectID}_{handSide}_{spectrum}_{iteration}.ext
    Returns (subject_id, side, spectrum, iteration) or None.
    Identity = subjectID + side  →  left/right treated as distinct classes.
    """
    name  = os.path.splitext(fname)[0]
    parts = name.split('_')
    if len(parts) < 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def get_all_subjects(root: str) -> List[str]:
    subjects = set()
    for fname in os.listdir(root):
        parsed = parse_casia_filename(fname)
        if parsed:
            subjects.add(parsed[0])
    return sorted(subjects)


def split_subjects_50_50(root: str) -> Tuple[List[str], List[str]]:
    """
    Open-set 50/50 subject split (Section 4.2):
    first half → train, second half → test, zero category overlap.
    Returns (train_identities, test_identities)  where identity = 'sub_side'.
    """
    subjects = get_all_subjects(root)
    mid      = len(subjects) // 2
    train_subs = set(subjects[:mid])
    test_subs  = set(subjects[mid:])

    all_ids = sorted({
        f"{p[0]}_{p[1]}"
        for fname in os.listdir(root)
        if (p := parse_casia_filename(fname)) is not None
    })
    train_ids = [i for i in all_ids if i.rsplit('_', 1)[0] in train_subs]
    test_ids  = [i for i in all_ids if i.rsplit('_', 1)[0] in test_subs]
    return train_ids, test_ids


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
            sub, side, spectrum, _ = parsed
            if spectrum not in self.spectra:
                continue
            identity = f"{sub}_{side}"
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
            T.RandomErasing(p=0.2, scale=(0.02, 0.15)),   # must be after ToTensor
        ])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — EPISODE SAMPLER  (Algorithm 1, steps 1–2)
# ═══════════════════════════════════════════════════════════════════════════════

class EpisodeSampler:
    """
    Samples one episode: N classes × (K support + Q_PER_CLASS query) images.
    """

    def __init__(self, dataset: CASIAMSDataset):
        self._label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for i, (_, label, _) in enumerate(dataset.samples):
            self._label_to_indices[label].append(i)

        min_needed = K + Q_PER_CLASS
        self.valid_labels: List[int] = [
            lbl for lbl, idxs in self._label_to_indices.items()
            if len(idxs) >= min_needed
        ]
        if len(self.valid_labels) < N:
            raise ValueError(
                f"Only {len(self.valid_labels)} classes have ≥ {min_needed} "
                f"samples, but N={N} are needed per episode."
            )
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
            picked = random.sample(pool, K + Q_PER_CLASS)
            for idx in picked[:K]:
                img, _, _ = self.dataset[idx]
                s_imgs.append(img);  s_labels.append(local_lbl)
            for idx in picked[K:]:
                img, _, _ = self.dataset[idx]
                q_imgs.append(img);  q_labels.append(local_lbl)

        return (
            torch.stack(s_imgs),
            torch.tensor(s_labels, dtype=torch.long),
            torch.stack(q_imgs),
            torch.tensor(q_labels, dtype=torch.long),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MODEL  (Section 4.2)
# ═══════════════════════════════════════════════════════════════════════════════

class W2MLModel(nn.Module):
    """
    ResNet-18 backbone fully frozen + trainable BN→Dropout→Linear head.

    With only 108 train identities (~3 K images), any backbone fine-tuning
    causes severe overfitting (train EER→3%, test EER stuck at 21%).
    Solution: freeze the entire backbone and make the 65K embed head more
    expressive by adding BatchNorm1d before the projection.

      Frozen   : entire ResNet-18 backbone (ImageNet weights preserved)
      Trainable: BN(512) → Dropout(0.3) → Linear(512→128)   [~66K params]

    BatchNorm re-centres and re-scales the 512-d backbone features per
    mini-batch, giving the head adaptive normalisation without adding
    overfittable convolutional weights.
    """

    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if PRETRAINED else None
        resnet  = models.resnet18(weights=weights)

        self.stem    = nn.Sequential(resnet.conv1, resnet.bn1,
                                     resnet.relu, resnet.maxpool)
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool

        # Freeze entire backbone
        for module in [self.stem, self.layer1, self.layer2]:
            for param in module.parameters():
                param.requires_grad = False

        # Trainable head: BN adapts feature scale, Dropout regularises
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512),        # re-normalise backbone features
            nn.Dropout(p=0.5),
            nn.Linear(512, EMBED_DIM, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
        return F.normalize(self.embed(x), p=2, dim=1)

    def param_groups(self, lr: float) -> list:
        """Single group — only the embed head is trained."""
        return [{'params': self.embed.parameters(), 'lr': lr}]

    def trainable_parameters(self):
        """All parameters that require grad (for counting)."""
        return [p for p in self.parameters() if p.requires_grad]


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — W2ML LOSS  (Equations 2–8)
# ═══════════════════════════════════════════════════════════════════════════════

def build_meta_support_sets(
    support_embs:   torch.Tensor,   # (N*K, D)
    support_labels: torch.Tensor,   # (N*K,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eq. 2 — S_j_meta = mean_i f(x^j_i),  then re-normalise.
    Returns meta_embs (N, D) and meta_labels (N,).
    """
    unique_labels = torch.unique(support_labels, sorted=True)
    meta_embs = torch.stack([
        support_embs[support_labels == lbl].mean(0) for lbl in unique_labels
    ])                                                      # (N, D)
    return F.normalize(meta_embs, p=2, dim=1), unique_labels


def mine_hard_pairs(
    pos_dists: torch.Tensor,   # (P,)
    neg_dists: torch.Tensor,   # (N_neg,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eq. 4 — positive selected iff  d_pos < max(d_neg) + m
    Eq. 5 — negative selected iff  d_neg > min(d_pos) - m
    Returns boolean masks (hard_pos_mask, hard_neg_mask).
    """
    hard_pos_mask = pos_dists < neg_dists.max() + MARGIN    # Eq. 4
    hard_neg_mask = neg_dists > pos_dists.min() - MARGIN    # Eq. 5
    return hard_pos_mask, hard_neg_mask


def w2ml_loss(
    query_embs:   torch.Tensor,   # (Q, D)  L2-normalised
    query_labels: torch.Tensor,   # (Q,)
    meta_embs:    torch.Tensor,   # (N, D)  L2-normalised
    meta_labels:  torch.Tensor,   # (N,)
) -> torch.Tensor:
    """
    Eq. 8 — Episode loss averaged over all l query samples.

    Loss is expressed in DISTANCE space (d = 1 − cosine_similarity).
    Substituting S = 1 − d into the original similarity-space MS Loss gives:

      Positive term: (1/α) · log(1 + Σ_P exp(+α(d_p − γ)))
                     penalises large positive distances  ✓

      Negative term: (1/β) · log(1 + Σ_N exp(−β(d_n − γ)))
                     penalises small negative distances  ✓

    FIX vs. original code: both exponent signs were inverted, causing the model
    to push same-class embeddings apart and pull different-class ones together.
    """
    MAX_EXP = 80.0   # clamp to prevent inf with β=40
    dist_mat = 1.0 - torch.mm(query_embs, meta_embs.t())   # (Q, N)

    per_query_losses = []

    for q_idx in range(query_embs.size(0)):
        q_lbl    = query_labels[q_idx]
        dists    = dist_mat[q_idx]                          # (N,)
        pos_mask = meta_labels == q_lbl                     # (N,) bool
        neg_mask = ~pos_mask

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        pos_dists = dists[pos_mask]   # (P,)  typically 1
        neg_dists = dists[neg_mask]   # (N-1,)

        # Hard-mining  (Eq. 4 & 5)
        hp_mask, hn_mask = mine_hard_pairs(pos_dists, neg_dists)
        if hp_mask.sum() == 0 or hn_mask.sum() == 0:
            continue

        hp = pos_dists[hp_mask]
        hn = neg_dists[hn_mask]

        # FIX: distance-space signs (were both inverted in original code)
        # Eq. 8 positive term — penalise large positive distances
        pos_exp  = torch.clamp(+ALPHA * (hp - GAMMA), max=MAX_EXP)   # was: -ALPHA
        pos_term = (1.0 / ALPHA) * torch.log1p(torch.exp(pos_exp).sum())

        # Eq. 8 negative term — penalise small negative distances
        neg_exp  = torch.clamp(-BETA  * (hn - GAMMA), max=MAX_EXP)   # was: +BETA
        neg_term = (1.0 / BETA)  * torch.log1p(torch.exp(neg_exp).sum())

        per_query_losses.append(pos_term + neg_term)

    if not per_query_losses:
        return dist_mat.sum() * 0.0   # safe zero with gradient

    return torch.stack(per_query_losses).mean()   # Eq. 8 : 1/l · Σ


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — EVALUATION  (Section 4.3)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(
    model:   nn.Module,
    dataset: CASIAMSDataset,
    device:  torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward full dataset → (embeddings (N,D), labels (N,))."""
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
    """
    Rank-1 identification accuracy.
    Gallery = first sample per class; probe = all remaining.
    """
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
    """EER via FAR / FRR sweep over 1 000 thresholds."""
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
    """EER from all genuine pairs + sampled imposter pairs."""
    rng   = np.random.default_rng(42)
    l2idx = defaultdict(list)
    for i, lbl in enumerate(labels):
        l2idx[int(lbl)].append(i)

    # All genuine pairs
    genuine = []
    for idxs in l2idx.values():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                genuine.append(1.0 - float(embs[idxs[i]] @ embs[idxs[j]]))
    genuine = np.array(genuine, dtype=np.float32)

    # Sampled imposter pairs
    uniq    = list(l2idx.keys())
    n_imp   = min(max_imp, len(genuine))
    imposter = []
    for _ in range(n_imp * 5):            # over-sample then truncate
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
#  SECTION 6 — TRAINING LOOP  (Algorithm 1, full)
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(
    model:   nn.Module,
    sampler: EpisodeSampler,
    device:  torch.device,
    train:   bool = True,
) -> torch.Tensor:
    """One episode: sample → forward → W2ML loss."""
    s_imgs, s_labels, q_imgs, q_labels = sampler.sample_episode()
    s_imgs, s_labels = s_imgs.to(device), s_labels.to(device)
    q_imgs, q_labels = q_imgs.to(device), q_labels.to(device)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        s_embs = model(s_imgs)                                    # (N*K, D)
        q_embs = model(q_imgs)                                    # (N*Q, D)
        meta_embs, meta_labels = build_meta_support_sets(s_embs, s_labels)
        loss = w2ml_loss(q_embs, q_labels, meta_embs, meta_labels)
    return loss


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\nW2ML Palmprint Recognition")
    print(f"  Device   : {device}")
    print(f"  Protocol : {EVAL_PROTOCOL}\n")

    # ── Build datasets ───────────────────────────────────────────────────
    train_ids, test_ids = split_subjects_50_50(DATA_ROOT)
    print(f"Subjects:  {len(train_ids)} train identities / "
          f"{len(test_ids)} test identities")

    if EVAL_PROTOCOL == 'cross_subject':
        tr_spec  = ALL_SPECTRA
        te_spec  = ALL_SPECTRA
    elif EVAL_PROTOCOL == 'cross_spectrum':
        tr_spec  = TRAIN_SPECTRA
        te_spec  = TEST_SPECTRA
        print(f"  Train spectra : {tr_spec}")
        print(f"  Test  spectra : {te_spec}")
    else:
        raise ValueError(f"Unknown EVAL_PROTOCOL: {EVAL_PROTOCOL}")

    train_dataset = CASIAMSDataset(DATA_ROOT, train_ids, tr_spec,
                                   get_transforms(train=False))  # no augment for eval
    test_dataset  = CASIAMSDataset(DATA_ROOT, test_ids,  te_spec,
                                   get_transforms(train=False))

    # Separate augmented dataset used only during training episodes
    train_dataset_aug = CASIAMSDataset(DATA_ROOT, train_ids, tr_spec,
                                       get_transforms(train=True))

    print(f"Samples:   {len(train_dataset)} train / {len(test_dataset)} test\n")

    train_sampler = EpisodeSampler(train_dataset_aug)

    # ── Model ────────────────────────────────────────────────────────────
    model = W2MLModel().to(device)
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"Params:    {n_trainable:,} trainable / {n_total:,} total "
          f"(backbone fully frozen, head only)\n")
    # Differential LR: layer4 at LR/10, embed head at full LR
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
            loss = run_episode(model, train_sampler, device, train=True)
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

        # ── Epoch-end evaluation — train and test ────────────────────────
        print("  Evaluating train set ...")
        tr_metrics = evaluate(model, train_dataset, device, tag='train')

        print("  Evaluating test set ...")
        te_metrics = evaluate(model, test_dataset,  device, tag=EVAL_PROTOCOL)

        tr_acc, tr_eer = tr_metrics['accuracy'], tr_metrics['eer']
        te_acc, te_eer = te_metrics['accuracy'], te_metrics['eer']

        print(f"  Summary  │  Train: Acc={tr_acc*100:.2f}%  EER={tr_eer*100:.2f}%"
              f"  │  Test:  Acc={te_acc*100:.2f}%  EER={te_eer*100:.2f}%")

        # Save best checkpoint based on test EER (lower is better)
        if te_eer < best_eer or (te_eer == best_eer and te_acc > best_acc):
            best_eer, best_acc = te_eer, te_acc
            torch.save({
                'epoch':     epoch + 1,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_eer':  best_eer,
                'best_acc':  best_acc,
            }, os.path.join(SAVE_DIR, 'best.pth'))
            print(f"  ✓ Best checkpoint saved  "
                  f"(Test EER={best_eer*100:.2f}%  Test Acc={best_acc*100:.2f}%)")

        torch.save({
            'epoch':    epoch + 1,
            'model':    model.state_dict(),
            'best_eer': best_eer,
            'best_acc': best_acc,
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

    print(f"\nDone.  Best Test EER={best_eer*100:.2f}%  Best Test Acc={best_acc*100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
