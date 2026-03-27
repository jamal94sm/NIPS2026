"""
PPNet on CASIA-MS Dataset
==================================================
Single-file implementation with closed-set and open-set protocols.
Faithfully preserves the official PPNet architecture, loss function,
normalisation (NormSingleROI), and training procedure.

PROTOCOL options (edit CONFIG below):
  'closed-set' : 80% of samples per identity → train | 20% → test
                 Evaluation: test probe vs train gallery (Rank-1 + EER)

  'open-set'   : 80% of identities → train | 20% of identities → test
                 Within test identities: 50% samples → gallery, 50% → probe
                 Evaluation: Rank-1 identification + EER

Dataset: CASIA-MS-ROI
  Filename format : {subjectID}_{handSide}_{spectrum}_{iteration}.jpg
  Identity key    : subjectID + handSide  (e.g. "001_L")
  All spectra and iterations are treated as samples of the same identity.

Architecture: PPNet (unchanged from official repo)
  5 conv layers + 2 FC layers + PairwiseDistance + classifier
  Input: 128×128 grayscale   FC1 input: 43264   Embedding dim: 512
"""

# ==============================================================
#  CONFIG  — edit this block only
# ==============================================================
CONFIG = {
    "protocol"        : "open-set",   # "closed-set" | "open-set"
    "data_root"       : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "results_dir"     : "./rst_ppnet_casia_ms",
    "img_side"        : 128,            # input image size (128×128 → fc1=43264)
    "batch_size"      : 8,              # MUST be even (PPNet contrastive loss)
    "num_epochs"      : 100,            # PPNet default is 3000; 500 for CASIA-MS
    "lr"              : 0.0001,         # PPNet default
    "lr_step"         : 100,            # PPNet default
    "lr_gamma"        : 0.8,            # PPNet default
    "dropout"         : 0.25,           # PPNet default
    "contrastive_margin" : 5.0,         # PPNet default
    "w_l2"            : 1e-4,           # PPNet L2 reg weight
    "w_contra"        : 2e-4,           # PPNet contrastive loss weight
    "w_dis"           : 1e-4,           # PPNet distance penalty weight
    "embedding_dim"   : 512,
    "train_ratio"     : 0.60,           # fraction of samples (closed) or IDs (open)
    "gallery_ratio"   : 0.10,           # open-set: fraction of test-ID samples → gallery
    "val_ratio"       : 0.20,           # fraction of train samples held out for val
    "random_seed"     : 42,
    "save_every"      : 10,             # save model every N epochs
    "eval_every"      : 50,             # run full evaluation every N epochs
    "num_workers"     : 4,
}
# ==============================================================

import os
import sys
import math
import time
import random
import pickle
import warnings
import numpy as np
from collections import defaultdict, Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────
SEED = CONFIG["random_seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════
#  MODEL  (exact copy of PPNet/models/ppnet.py — unchanged)
# ══════════════════════════════════════════════════════════════

class ppnet(nn.Module):
    def __init__(self, num_classes):
        super(ppnet, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module("conv", nn.Conv2d(1, 16, 5, 1))
        self.layer1.add_module("bn", nn.BatchNorm2d(16))

        self.layer2 = nn.Sequential()
        self.layer2.add_module("conv", nn.Conv2d(16, 32, 1, 1))
        self.layer2.add_module("bn", nn.BatchNorm2d(32, momentum=0.001, affine=True, track_running_stats=True))
        self.layer2.add_module("sigmoid", nn.Sigmoid())
        self.layer2.add_module("avgpool", nn.AvgPool2d(2, 2))

        self.layer3 = nn.Sequential()
        self.layer3.add_module("conv", nn.Conv2d(32, 64, 3, 1))
        self.layer3.add_module("bn", nn.BatchNorm2d(64, momentum=0.001, affine=True, track_running_stats=True))
        self.layer3.add_module("sigmoid", nn.Sigmoid())
        self.layer3.add_module("avgpool", nn.AvgPool2d(2, 2))

        self.layer4 = nn.Sequential()
        self.layer4.add_module("conv", nn.Conv2d(64, 64, 3, 1))
        self.layer4.add_module("bn", nn.BatchNorm2d(64, momentum=0.001, affine=True, track_running_stats=True))
        self.layer4.add_module("relu", nn.ReLU())

        self.layer5 = nn.Sequential()
        self.layer5.add_module("conv", nn.Conv2d(64, 256, 3, 1))
        self.layer5.add_module("bn", nn.BatchNorm2d(256, momentum=0.001, affine=True, track_running_stats=True))
        self.layer5.add_module("relu", nn.ReLU())
        self.layer5.add_module("maxpool", nn.MaxPool2d(2, 2))

        self.fc1 = nn.Linear(43264, 512)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.001, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.001, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.25)

        self.dis = nn.PairwiseDistance(p=2)

        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        b, _ = x.size()
        o1 = x[:b // 2, :]
        o2 = x[b // 2:, :]
        dis = self.dis(o1, o2)

        x = self.fc3(x)

        return x, dis

    def getFeatureCode(self, x):
        """Return 512-d embedding (before dropout and classifier)."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


# ══════════════════════════════════════════════════════════════
#  NORMALISATION  (exact copy of PPNet/models/dataset.py)
# ══════════════════════════════════════════════════════════════

class NormSingleROI(object):
    """
    Normalize the input image (exclude the black region) to be 0 mean and 1 std.
    [c, h, w]
    """
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        c, h, w = tensor.size()
        if c != 1:
            raise TypeError('only support grayscale image.')

        tensor = tensor.view(c, h * w)
        idx = tensor > 0
        t = tensor[idx]
        m = t.mean()
        s = t.std()
        t = t.sub_(m).div_(s + 1e-6)
        tensor[idx] = t
        tensor = tensor.view(c, h, w)

        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats=self.outchannels, dim=0)

        return tensor


# ══════════════════════════════════════════════════════════════
#  DATASET UTILITIES
# ══════════════════════════════════════════════════════════════

class CASIAMSDataset(Dataset):
    """
    Dataset for CASIA-MS ROI images.
    Uses NormSingleROI normalisation (matching official PPNet).
    Applies data augmentation during training (matching official PPNet).

    Parameters
    ----------
    samples  : list of (image_path, int_label)
    img_side : resize target (default 128)
    train    : if True, apply PPNet augmentation
    """
    def __init__(self, samples, img_side=128, train=False):
        self.samples  = samples
        self.img_side = img_side

        if train:
            self.transforms = T.Compose([
                T.Resize(img_side),
                T.RandomChoice(transforms=[
                    T.ColorJitter(brightness=0.3, contrast=0.3),
                    T.RandomResizedCrop(size=img_side, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                    T.RandomRotation(degrees=8, expand=False),
                    T.RandomPerspective(distortion_scale=0.15, p=0.8),
                ]),
                T.ToTensor(),
                NormSingleROI(outchannels=1),
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(img_side),
                T.ToTensor(),
                NormSingleROI(outchannels=1),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transforms(img)
        return img, label


# ══════════════════════════════════════════════════════════════
#  DATA LOADING & SPLIT LOGIC
# ══════════════════════════════════════════════════════════════

def parse_casia_ms(data_root):
    """
    Scan data_root for files matching  {subjectID}_{handSide}_{spectrum}_{iter}.jpg
    Returns
    -------
    id2paths : dict  identity_key → sorted list of absolute paths
    """
    id2paths = defaultdict(list)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for fname in sorted(os.listdir(data_root)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in exts:
            continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) < 4:
            print(f"  [WARN] Skipping unexpected filename: {fname}")
            continue
        subject_id = parts[0]
        hand_side  = parts[1]
        identity   = f"{subject_id}_{hand_side}"
        id2paths[identity].append(os.path.join(data_root, fname))

    return dict(id2paths)


def make_label_map(identities_sorted):
    """Map sorted identity keys to consecutive integer labels starting at 0."""
    return {ident: idx for idx, ident in enumerate(sorted(identities_sorted))}


def split_closed_set(id2paths, train_ratio=0.80, seed=42):
    """
    Closed-set split:
      - All identities appear in both train and test.
      - Per identity: first `train_ratio` fraction of samples → train,
        remainder → test.
    """
    rng = random.Random(seed)
    identities = sorted(id2paths.keys())
    label_map  = make_label_map(identities)

    train_samples, test_samples = [], []
    for ident in identities:
        paths = id2paths[ident][:]
        rng.shuffle(paths)
        label   = label_map[ident]
        n_train = max(1, int(len(paths) * train_ratio))
        for p in paths[:n_train]:
            train_samples.append((p, label))
        for p in paths[n_train:]:
            test_samples.append((p, label))

    print(f"  [closed-set] identities: {len(identities)} | "
          f"train: {len(train_samples)} | test: {len(test_samples)}")
    return train_samples, test_samples, label_map


def split_open_set(id2paths, train_ratio=0.80, gallery_ratio=0.50,
                   val_ratio=0.10, seed=42):
    """
    Open-set split:
      - 80% of identities → training.
      - 20% of identities → test, never seen during training.
      - Within train identities: `val_ratio` of samples → validation,
        rest → training.
      - Within test identities: `gallery_ratio` of samples → gallery,
        rest → probe.
    """
    rng = random.Random(seed)
    identities = sorted(id2paths.keys())
    rng_ids    = identities[:]
    rng.shuffle(rng_ids)

    n_train = max(1, int(len(rng_ids) * train_ratio))
    train_ids = sorted(rng_ids[:n_train])
    test_ids  = sorted(rng_ids[n_train:])

    train_label_map = make_label_map(train_ids)
    test_label_map  = make_label_map(test_ids)

    train_samples   = []
    val_samples     = []
    gallery_samples = []
    probe_samples   = []

    for ident in train_ids:
        paths = id2paths[ident][:]
        rng.shuffle(paths)
        label = train_label_map[ident]
        n_val = max(1, int(len(paths) * val_ratio))
        for p in paths[:n_val]:
            val_samples.append((p, label))
        for p in paths[n_val:]:
            train_samples.append((p, label))

    for ident in test_ids:
        paths = id2paths[ident][:]
        rng.shuffle(paths)
        label     = test_label_map[ident]
        n_gallery = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gallery]:
            gallery_samples.append((p, label))
        for p in paths[n_gallery:]:
            probe_samples.append((p, label))

    print(f"  [open-set] train IDs: {len(train_ids)} | test IDs: {len(test_ids)}")
    print(f"             train samples: {len(train_samples)} | "
          f"val samples: {len(val_samples)} | "
          f"gallery: {len(gallery_samples)} | probe: {len(probe_samples)}")
    return (train_samples, val_samples, gallery_samples, probe_samples,
            train_label_map, test_label_map)


# ══════════════════════════════════════════════════════════════
#  CONTRASTIVE LOSS  (exact copy from PPNet/train.py)
# ══════════════════════════════════════════════════════════════

def contrastive_loss(target, dis, margin, device):
    """
    PPNet contrastive loss.
    Splits the batch labels in half: first half vs second half.
    Same-class pairs attract (minimise distance),
    different-class pairs repel (push beyond margin).
    """
    n = len(target) // 2
    y1 = target[:n]
    y2 = target[n:]

    y = torch.zeros(n, device=device)
    y[y1 == y2] = 1.0

    margin_t = torch.full((n,), margin, device=device)

    contra = torch.mean(
        y * torch.pow(dis, 2)
        + (1 - y) * torch.pow(torch.clamp(margin_t - dis, min=0.0), 2)
    )
    return contra


# ══════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_features(net, data_loader, device):
    """
    Returns
    -------
    feats  : np.ndarray  [N, embedding_dim]
    labels : np.ndarray  [N]
    """
    net.eval()
    feats_list  = []
    labels_list = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            codes = net.getFeatureCode(data)
            feats_list.append(codes.cpu().numpy())
            labels_list.append(target.numpy())
    feats  = np.concatenate(feats_list,  axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return feats, labels


# ══════════════════════════════════════════════════════════════
#  MATCHING & METRICS
# ══════════════════════════════════════════════════════════════

def l2_distance(f1, f2):
    """L2 (Euclidean) distance between two feature vectors (matches PPNet test.py)."""
    return np.linalg.norm(f1 - f2, 2)


def compute_eer(scores, labels):
    """
    Compute EER from a list of (score, label) pairs.
    Scores are L2 distances (lower = more similar → genuine scores are smaller).
    Handles direction automatically (matching PPNet/getEER.py).
    """
    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels)

    in_scores  = scores[labels ==  1]
    out_scores = scores[labels == -1]

    # getEER.py: if genuine mean < impostor mean → negate (roc_curve needs sim)
    mIn  = in_scores.mean()
    mOut = out_scores.mean()
    flipped = False
    if mIn < mOut:
        in_scores  = -in_scores
        out_scores = -out_scores
        flipped = True

    y    = np.concatenate([np.ones(len(in_scores)), np.zeros(len(out_scores))])
    sall = np.concatenate([in_scores, out_scores])

    fpr, tpr, thresholds = roc_curve(y, sall, pos_label=1)
    roc_auc = auc(fpr, tpr)

    eer    = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))

    # reverse threshold back if we flipped
    if flipped:
        thresh = -thresh
        thresholds = -thresholds

    diff     = np.abs(fpr - (1 - tpr))
    idx      = np.argmin(diff)
    eer_half = (fpr[idx] + (1 - tpr[idx])) / 2.0

    return eer, thresh, roc_auc, eer_half, fpr, tpr, thresholds


def compute_rank1(probe_feats, probe_labels,
                  gallery_feats, gallery_labels,
                  dist_matrix=None):
    """Rank-1 identification accuracy using L2 distance."""
    n_probe   = probe_feats.shape[0]
    n_gallery = gallery_feats.shape[0]

    if dist_matrix is None:
        dist = np.zeros((n_probe, n_gallery))
        for i in range(n_probe):
            for j in range(n_gallery):
                dist[i, j] = l2_distance(probe_feats[i], gallery_feats[j])
    else:
        dist = dist_matrix

    correct = 0
    for i in range(n_probe):
        best_j = int(np.argmin(dist[i]))
        if probe_labels[i] == gallery_labels[best_j]:
            correct += 1
    rank1 = correct / n_probe * 100.0
    return rank1, dist


def compute_aggregated_eer(dist_matrix, prb_labels, gal_labels):
    """
    Aggregated EER (min-distance per gallery class per probe).
    Matches the aggregation in PPNet/test.py.
    """
    class_ids = sorted(set(gal_labels.tolist()))
    n_probe   = dist_matrix.shape[0]

    aggr_s, aggr_l = [], []
    for i in range(n_probe):
        for cls in class_ids:
            cls_mask = (gal_labels == cls)
            min_dist = dist_matrix[i, cls_mask].min()
            aggr_s.append(min_dist)
            aggr_l.append(1 if prb_labels[i] == cls else -1)

    return aggr_s, aggr_l


# ══════════════════════════════════════════════════════════════
#  PLOTTING & SAVING
# ══════════════════════════════════════════════════════════════

def save_scores_txt(scores, labels, path):
    with open(path, "w") as f:
        for s, l in zip(scores, labels):
            f.write(f"{s} {l}\n")


def plot_and_save(fpr, tpr, fnr, thresholds, eer, rank1, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    fpr_pct = fpr * 100
    tpr_pct = tpr * 100
    fnr_pct = fnr * 100

    pdf_path = os.path.join(out_dir, f"roc_det_{tag}.pdf")
    with PdfPages(pdf_path) as pdf:
        # ROC
        plt.figure()
        plt.plot(fpr_pct, tpr_pct, "b-^", label="ROC")
        plt.plot(np.linspace(0, 100, 101), np.linspace(100, 0, 101), "k-", label="EER")
        plt.xlim([0, 5]); plt.ylim([90, 100])
        plt.legend(); plt.grid(True)
        plt.title(f"ROC  |  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
        plt.xlabel("FAR (%)"); plt.ylabel("GAR (%)")
        plt.savefig(os.path.join(out_dir, f"ROC_{tag}.png"))
        pdf.savefig(); plt.close()

        # DET
        plt.figure()
        plt.plot(fpr_pct, fnr_pct, "b-^", label="DET")
        plt.plot(np.linspace(0, 100, 101), np.linspace(0, 100, 101), "k-", label="EER")
        plt.xlim([0, 5]); plt.ylim([0, 5])
        plt.legend(); plt.grid(True)
        plt.title("DET curve")
        plt.xlabel("FAR (%)"); plt.ylabel("FRR (%)")
        plt.savefig(os.path.join(out_dir, f"DET_{tag}.png"))
        pdf.savefig(); plt.close()

        # FAR/FRR vs threshold
        plt.figure()
        plt.plot(thresholds, fpr_pct, "r-.", label="FAR")
        plt.plot(thresholds, fnr_pct, "b-^", label="FRR")
        plt.legend(); plt.grid(True)
        plt.title("FAR and FRR Curves")
        plt.xlabel("Threshold"); plt.ylabel("FAR / FRR (%)")
        plt.savefig(os.path.join(out_dir, f"FAR_FRR_{tag}.png"))
        pdf.savefig(); plt.close()


def plot_loss_acc(train_losses, val_losses, train_acc, val_acc, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ep = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(ep, train_losses, "b", label="train loss")
    plt.plot(ep, val_losses,   "r", label="val loss")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.savefig(os.path.join(out_dir, "losses.png")); plt.close()

    plt.figure()
    plt.plot(ep, train_acc, "b", label="train acc")
    plt.plot(ep, val_acc,   "r", label="val acc")
    plt.legend(); plt.grid(True)
    plt.xlabel("epoch"); plt.ylabel("accuracy (%)")
    plt.savefig(os.path.join(out_dir, "accuracy.png")); plt.close()


def plot_gi_histogram(in_scores, out_scores, out_dir, tag):
    """Genuine-Impostor matching score distribution."""
    os.makedirs(out_dir, exist_ok=True)
    samples = 100
    in_arr  = np.array(in_scores)
    out_arr = np.array(out_scores)

    def normalise_hist(arr):
        lo, hi = arr.min(), arr.max()
        idx_arr = np.round((arr - lo) / (hi - lo + 1e-10) * samples).astype(int)
        h = np.zeros(samples + 1)
        for v in idx_arr:
            h[v] += 1
        h = h / h.sum() * 100
        x = np.linspace(0, 1, samples + 1) * (hi - lo) + lo
        return x, h

    xi, hi = normalise_hist(in_arr)
    xo, ho = normalise_hist(out_arr)

    plt.figure()
    plt.plot(xo, ho, "r", label="Impostor")
    plt.plot(xi, hi, "b", label="Genuine")
    plt.legend(fontsize=13)
    plt.xlabel("Matching Score", fontsize=13)
    plt.ylabel("Percentage (%)", fontsize=13)
    plt.ylim([0, 1.2 * max(hi.max(), ho.max())])
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"GI_{tag}.png")); plt.close()


# ══════════════════════════════════════════════════════════════
#  TRAINING  (matches PPNet/train.py loss & logic)
# ══════════════════════════════════════════════════════════════

def run_one_epoch(epoch, model, loader, criterion, optimizer, device,
                  phase="training",
                  margin=5.0, w_l2=1e-4, w_contra=2e-4, w_dis=1e-4):
    """
    PPNet training / validation epoch.

    Loss = CrossEntropy + w_l2 * L2_reg + w_contra * contrastive + w_dis * mean(dis²)

    Batch must be even for contrastive pairing.  If the last batch is odd,
    we pad it by duplicating the first sample (matching PPNet/train.py).
    """
    if phase == "training":
        model.train()
    else:
        model.eval()

    running_loss    = 0.0
    running_correct = 0
    num_samples     = 0

    for data, target in loader:
        # PPNet requires even batch for contrastive loss
        if len(target) % 2 != 0:
            target = torch.cat((target, target[0:1]), dim=0)
            data   = torch.cat((data,   data[0:1]),   dim=0)

        data, target = data.to(device), target.to(device)

        if phase == "training":
            optimizer.zero_grad()

        if phase == "training":
            output, dis = model(data)
        else:
            with torch.no_grad():
                output, dis = model(data)

        # --- PPNet composite loss ---
        cross = criterion(output, target)

        # Retrieve the underlying model for weight access
        _m = model.module if isinstance(model, DataParallel) else model
        l2_reg = torch.norm(_m.fc2.weight, 2) + torch.norm(_m.fc3.weight, 2)

        contra = contrastive_loss(target, dis, margin, device)

        loss = cross + w_l2 * l2_reg + w_contra * contra + w_dis * torch.mean(torch.pow(dis, 2))

        running_loss += loss.item()

        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().item()
        num_samples += len(target)

        if phase == "training":
            loss.backward()
            optimizer.step()

    avg_loss = running_loss / num_samples
    acc      = 100.0 * running_correct / num_samples
    return avg_loss, acc


# ══════════════════════════════════════════════════════════════
#  EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════

def evaluate(net, probe_loader, gallery_loader,
             device, out_dir, tag="eval"):
    """
    Shared evaluation for both protocols.
    Uses L2 distance (matching PPNet/test.py).
    Always computes both pairwise and aggregated EER.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- extract features ---
    print("  Extracting gallery features …")
    gal_feats, gal_labels = extract_features(net, gallery_loader, device)

    print("  Extracting probe features …")
    prb_feats, prb_labels = extract_features(net, probe_loader, device)

    n_probe   = prb_feats.shape[0]
    n_gallery = gal_feats.shape[0]
    print(f"  probe: {n_probe}  gallery: {n_gallery}")

    # --- compute all pairwise L2 distances ---
    print("  Computing pairwise L2 distances …")
    s, l = [], []
    dist_matrix = np.zeros((n_probe, n_gallery))
    for i in range(n_probe):
        for j in range(n_gallery):
            d = l2_distance(prb_feats[i], gal_feats[j])
            dist_matrix[i, j] = d
            s.append(d)
            l.append(1 if prb_labels[i] == gal_labels[j] else -1)

    # save raw scores
    scores_path = os.path.join(out_dir, f"scores_{tag}.txt")
    save_scores_txt(s, l, scores_path)

    # --- Pairwise EER ---
    eer, thresh, roc_auc, eer_half, fpr, tpr, thresholds = compute_eer(s, l)
    fnr = 1 - tpr
    print(f"  Pairwise EER: {eer*100:.4f}%  |  thresh: {thresh:.4f}  |  AUC: {roc_auc:.6f}")
    print(f"  Pairwise EER½: {eer_half*100:.4f}%")

    # --- Rank-1 ---
    rank1, _ = compute_rank1(prb_feats, prb_labels,
                              gal_feats, gal_labels,
                              dist_matrix=dist_matrix)
    print(f"  Rank-1 acc: {rank1:.3f}%")

    # --- GI histogram ---
    in_scores  = [s[k] for k in range(len(s)) if l[k] ==  1]
    out_scores = [s[k] for k in range(len(s)) if l[k] == -1]
    plot_gi_histogram(in_scores, out_scores, out_dir, tag)

    # --- plots ---
    plot_and_save(fpr, tpr, fnr, thresholds, eer, rank1, out_dir, tag)

    # save text summary
    with open(os.path.join(out_dir, f"rst_{tag}.txt"), "w") as f:
        f.write(f"Pairwise EER  : {eer*100:.6f}%\n")
        f.write(f"Pairwise EER½ : {eer_half*100:.6f}%\n")
        f.write(f"Threshold     : {thresh:.4f}\n")
        f.write(f"AUC           : {roc_auc:.10f}\n")
        f.write(f"Rank-1        : {rank1:.3f}%\n")

    # --- Aggregated EER (always compute when multi-sample gallery) ---
    n_gallery_classes = len(set(gal_labels.tolist()))
    aggr_eer = eer  # default fallback
    if n_gallery_classes < n_gallery:
        print("  Computing aggregated EER …")
        aggr_s, aggr_l = compute_aggregated_eer(dist_matrix, prb_labels, gal_labels)
        (aggr_eer, aggr_thresh, aggr_auc,
         aggr_eer_half, *_) = compute_eer(aggr_s, aggr_l)
        print(f"  Aggregated EER: {aggr_eer*100:.4f}%  |  AUC: {aggr_auc:.6f}")
        with open(os.path.join(out_dir, f"rst_{tag}.txt"), "a") as f:
            f.write(f"\nAggregated EER      : {aggr_eer*100:.6f}%\n")
            f.write(f"Aggregated EER_half : {aggr_eer_half*100:.6f}%\n")
            f.write(f"Aggregated AUC      : {aggr_auc:.10f}\n")

    return eer, aggr_eer, rank1


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    # ---------- unpack config ----------
    protocol       = CONFIG["protocol"]
    data_root      = CONFIG["data_root"]
    results_dir    = CONFIG["results_dir"]
    img_side       = CONFIG["img_side"]
    batch_size     = CONFIG["batch_size"]
    num_epochs     = CONFIG["num_epochs"]
    lr             = CONFIG["lr"]
    lr_step        = CONFIG["lr_step"]
    lr_gamma       = CONFIG["lr_gamma"]
    dropout        = CONFIG["dropout"]
    margin         = CONFIG["contrastive_margin"]
    w_l2           = CONFIG["w_l2"]
    w_contra       = CONFIG["w_contra"]
    w_dis          = CONFIG["w_dis"]
    emb_dim        = CONFIG["embedding_dim"]
    train_ratio    = CONFIG["train_ratio"]
    gallery_ratio  = CONFIG["gallery_ratio"]
    val_ratio      = CONFIG["val_ratio"]
    seed           = CONFIG["random_seed"]
    save_every     = CONFIG["save_every"]
    eval_every     = CONFIG["eval_every"]
    nw             = CONFIG["num_workers"]

    assert protocol in ("closed-set", "open-set"), \
        f"Unknown protocol: {protocol}. Use 'closed-set' or 'open-set'."
    assert batch_size % 2 == 0, \
        f"batch_size must be even for PPNet contrastive loss, got {batch_size}."

    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Protocol : {protocol}")
    print(f"  Device   : {device}")
    print(f"  Data     : {data_root}")
    print(f"{'='*60}\n")

    # ---------- parse dataset ----------
    print("Scanning dataset …")
    id2paths = parse_casia_ms(data_root)
    n_total_ids = len(id2paths)
    n_total_imgs = sum(len(v) for v in id2paths.values())
    print(f"  Found {n_total_ids} identities, {n_total_imgs} images total.\n")

    # ---------- protocol-specific split ----------
    if protocol == "closed-set":
        train_samples, test_samples, label_map = split_closed_set(
            id2paths, train_ratio=train_ratio, seed=seed)
        num_classes = len(label_map)

        train_dataset   = CASIAMSDataset(train_samples, img_side=img_side, train=True)
        test_dataset    = CASIAMSDataset(test_samples,  img_side=img_side, train=False)
        train_gal_data  = CASIAMSDataset(train_samples, img_side=img_side, train=False)

        train_loader    = DataLoader(train_dataset,  batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=True)
        val_loader      = DataLoader(test_dataset,   batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        gallery_loader  = DataLoader(train_gal_data, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        probe_loader    = val_loader

        print(f"  [closed-set] #classes={num_classes}\n")

    else:  # open-set
        (train_samples, val_samples, gallery_samples, probe_samples,
         train_label_map, test_label_map) = split_open_set(
            id2paths, train_ratio=train_ratio,
            gallery_ratio=gallery_ratio,
            val_ratio=val_ratio, seed=seed)
        num_classes = len(train_label_map)

        train_dataset   = CASIAMSDataset(train_samples,   img_side=img_side, train=True)
        val_dataset     = CASIAMSDataset(val_samples,      img_side=img_side, train=False)
        gallery_dataset = CASIAMSDataset(gallery_samples,  img_side=img_side, train=False)
        probe_dataset   = CASIAMSDataset(probe_samples,    img_side=img_side, train=False)

        train_loader    = DataLoader(train_dataset,   batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=True)
        val_loader      = DataLoader(val_dataset,     batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        gallery_loader  = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        probe_loader    = DataLoader(probe_dataset,   batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

        print(f"  [open-set] #train_classes={num_classes}\n")

    # ---------- model ----------
    print(f"Building PPNet — num_classes={num_classes} …")
    net = ppnet(num_classes=num_classes)
    net.to(device)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
        net = DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    # ---------- training loop ----------
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0
    best_eer     = 1.0

    last_eer   = float("nan")
    last_rank1 = float("nan")

    print(f"\nStarting training for {num_epochs} epochs …")
    print(f"  EER / Rank-1 computed every {eval_every} epochs and shown in every 10-epoch log.")
    print(f"  Loss = CE + {w_l2}*L2 + {w_contra}*contrastive(margin={margin}) + {w_dis}*mean(dis²)\n")

    for epoch in range(num_epochs):
        t_loss, t_acc = run_one_epoch(
            epoch, net, train_loader, criterion, optimizer, device,
            "training", margin=margin, w_l2=w_l2, w_contra=w_contra, w_dis=w_dis)
        v_loss, v_acc = run_one_epoch(
            epoch, net, val_loader, criterion, optimizer, device,
            "validation", margin=margin, w_l2=w_l2, w_contra=w_contra, w_dis=w_dis)
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        _net = net.module if isinstance(net, DataParallel) else net

        # ── periodic evaluation (EER / Rank-1) ───────────────────────────────
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            eval_net = _net
            tag = f"ep{epoch:04d}_{protocol.replace('-','')}"
            cur_eer, cur_aggr_eer, cur_rank1 = evaluate(
                eval_net,
                probe_loader,
                gallery_loader,
                device,
                out_dir=rst_eval,
                tag=tag,
            )
            last_eer   = cur_eer
            last_rank1 = cur_rank1

            if cur_eer < best_eer:
                best_eer = cur_eer
                torch.save(_net.state_dict(),
                           os.path.join(results_dir, "net_params_best_eer.pth"))
                print(f"  *** New best EER: {best_eer*100:.4f}% ***")

        # ── every-10-epoch console print ──────────────────────────────────────
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            ts = time.strftime("%H:%M:%S")
            eer_str   = f"{last_eer*100:.4f}%"   if not math.isnan(last_eer)   else "N/A"
            rank1_str = f"{last_rank1:.2f}%"      if not math.isnan(last_rank1) else "N/A"
            print(
                f"[{ts}] ep {epoch:04d} | "
                f"loss  train={t_loss:.5f}  val={v_loss:.5f} | "
                f"cls-acc  train={t_acc:.2f}%  val={v_acc:.2f}% | "
                f"EER={eer_str}  Rank-1={rank1_str}"
            )

        # ── save best classification model ────────────────────────────────────
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(_net.state_dict(),
                       os.path.join(results_dir, "net_params_best.pth"))

        # ── periodic checkpoint + loss/acc plots ──────────────────────────────
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            torch.save(_net.state_dict(),
                       os.path.join(results_dir, "net_params.pth"))
            plot_loss_acc(train_losses, val_losses, train_accs, val_accs, results_dir)

    # ---------- final evaluation with best model ----------
    print("\n=== Final evaluation with best EER model ===")
    best_model_path = os.path.join(results_dir, "net_params_best_eer.pth")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(results_dir, "net_params_best.pth")

    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_model_path, map_location=device))

    final_eer, final_aggr_eer, final_rank1 = evaluate(
        eval_net,
        probe_loader,
        gallery_loader,
        device,
        out_dir=rst_eval,
        tag=f"FINAL_{protocol.replace('-','')}",
    )

    print(f"\n{'='*60}")
    print(f"  PROTOCOL : {protocol}")
    print(f"  FINAL Pairwise EER   : {final_eer*100:.4f}%")
    print(f"  FINAL Aggregated EER : {final_aggr_eer*100:.4f}%")
    print(f"  FINAL Rank-1         : {final_rank1:.3f}%")
    print(f"  Results saved to: {results_dir}")
    print(f"{'='*60}\n")

    # save final summary
    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"Protocol  : {protocol}\n")
        f.write(f"Data root : {data_root}\n")
        f.write(f"Identities: {n_total_ids}\n")
        f.write(f"Images    : {n_total_imgs}\n")
        f.write(f"Classes (train): {num_classes}\n")
        f.write(f"Best val acc       : {best_val_acc:.3f}%\n")
        f.write(f"Final Pairwise EER : {final_eer*100:.6f}%\n")
        f.write(f"Final Aggreg. EER  : {final_aggr_eer*100:.6f}%\n")
        f.write(f"Final Rank-1       : {final_rank1:.3f}%\n")


if __name__ == "__main__":
    main()
