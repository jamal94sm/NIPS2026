"""
CompNet — Cross-Domain Closed-Set Evaluation on Palm-Auth
===========================================================
Training domain : smartphone images  (roi_perspective)
Test domain     : scanner images     (roi_scanner)

Protocol: CLOSED SET
  - Exactly the same subject IDs appear in both train and test.
  - Only subjects that have images in BOTH domains are included.
  - Train  : ALL roi_perspective images for every shared subject.
  - Gallery: 50 % of roi_scanner images per subject (random split).
  - Probe  : remaining 50 % of roi_scanner images per subject.

Scanner spectra kept: green | ir | yellow | pink | white

Results are saved to:
  {BASE_RESULTS_DIR}/eval/          ← per-checkpoint score files
  {BASE_RESULTS_DIR}/results.txt    ← final EER / Rank-1
  {BASE_RESULTS_DIR}/train_curves.png
"""

# ==============================================================
#  CONFIG
# ==============================================================
CONFIG = {
    # ── Dataset path ───────────────────────────────────────────
    "palm_auth_data_root"  : "/home/pai-ng/Jamal/smartphone_data",

    # ── Scanner spectra to include ─────────────────────────────
    "scanner_spectra"      : {"green", "ir", "yellow", "pink", "white"},

    # ── Gallery / probe split ratio (scanner side) ─────────────
    "test_gallery_ratio"   : 0.50,

    # ── Model ──────────────────────────────────────────────────
    "img_side"             : 128,
    "embedding_dim"        : 512,
    "dropout"              : 0.25,
    "arcface_s"            : 30.0,
    "arcface_m"            : 0.50,

    # ── Training ───────────────────────────────────────────────
    "batch_size"           : 128,
    "num_epochs"           : 300,
    "lr"                   : 0.001,
    "lr_step"              : 30,
    "lr_gamma"             : 0.8,
    "augment_factor"       : 2,

    # ── Misc ───────────────────────────────────────────────────
    "base_results_dir"     : "./rst_compnet_crossdomain",
    "random_seed"          : 42,
    "save_every"           : 50,
    "eval_every"           : 50,
    "num_workers"          : 4,
}
# ==============================================================

import os
import json
import math
import time
import random
import warnings
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


# ══════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super().__init__()
        self.channel_in = channel_in; self.channel_out = channel_out
        self.kernel_size = kernel_size; self.stride = stride
        self.padding = padding; self.init_ratio = max(init_ratio, 1e-6)
        self.kernel = 0
        _S = 9.2 * self.init_ratio; _F = 0.057 / self.init_ratio; _G = 2.0
        self.gamma = nn.Parameter(torch.FloatTensor([_G]))
        self.sigma = nn.Parameter(torch.FloatTensor([_S]))
        self.theta = nn.Parameter(
            torch.arange(0, channel_out).float() * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([_F]))
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def _gen(self, ksize, c_in, c_out, sigma, gamma, theta, f, psi):
        half = ksize // 2; ksz = 2 * half + 1
        y0 = torch.arange(-half, half + 1).float()
        x0 = torch.arange(-half, half + 1).float()
        y  = y0.view(1,-1).repeat(c_out, c_in, ksz, 1)
        x  = x0.view(-1,1).repeat(c_out, c_in, 1, ksz)
        x  = x.to(sigma.device); y = y.to(sigma.device)
        xt =  x*torch.cos(theta.view(-1,1,1,1)) + y*torch.sin(theta.view(-1,1,1,1))
        yt = -x*torch.sin(theta.view(-1,1,1,1)) + y*torch.cos(theta.view(-1,1,1,1))
        gb = -torch.exp(-0.5*((gamma*xt)**2+yt**2)/(8*sigma.view(-1,1,1,1)**2)
            ) * torch.cos(2*math.pi*f.view(-1,1,1,1)*xt+psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2,3], keepdim=True)

    def forward(self, x):
        self.kernel = self._gen(self.kernel_size, self.channel_in,
                                self.channel_out, self.sigma, self.gamma,
                                self.theta, self.f, self.psi)
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)


class CompetitiveBlock(nn.Module):
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 init_ratio=1, o1=32, o2=12):
        super().__init__()
        self.gabor   = GaborConv2d(channel_in, n_competitor, ksize,
                                   stride, padding, init_ratio)
        self.a       = nn.Parameter(torch.FloatTensor([1]))
        self.b       = nn.Parameter(torch.FloatTensor([0]))
        self.argmax  = nn.Softmax(dim=1)
        self.conv1   = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.gabor(x)
        x = self.argmax((x - self.b) * self.a)
        return self.conv2(self.maxpool(self.conv1(x)))


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super().__init__()
        self.s = s; self.m = m
        self.weight      = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.training:
            assert label is not None
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
            phi  = cosine * self.cos_m - sine * self.sin_m
            phi  = (torch.where(cosine > 0, phi, cosine) if self.easy_margin
                    else torch.where(cosine > self.th, phi, cosine - self.mm))
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            return self.s * ((one_hot * phi) + ((1 - one_hot) * cosine))
        return self.s * cosine


class CompNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=512,
                 arcface_s=30.0, arcface_m=0.50, dropout=0.25):
        super().__init__()
        self.cb1  = CompetitiveBlock(1, 9, 35, 3, 0, init_ratio=1.00)
        self.cb2  = CompetitiveBlock(1, 9, 17, 3, 0, init_ratio=0.50)
        self.cb3  = CompetitiveBlock(1, 9,  7, 3, 0, init_ratio=0.25)
        self.fc   = nn.Linear(9708, embedding_dim)
        self.drop = nn.Dropout(p=dropout)
        self.arc  = ArcMarginProduct(embedding_dim, num_classes,
                                     s=arcface_s, m=arcface_m)

    def _backbone(self, x):
        x1 = self.cb1(x).flatten(1); x2 = self.cb2(x).flatten(1)
        x3 = self.cb3(x).flatten(1)
        return self.fc(torch.cat([x1, x2, x3], dim=1))

    def forward(self, x, y=None):
        return self.arc(self.drop(self._backbone(x)), y)

    @torch.no_grad()
    def get_embedding(self, x):
        return F.normalize(self._backbone(x), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  NORMALISATION
# ══════════════════════════════════════════════════════════════

class NormSingleROI:
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        c, h, w = tensor.size(); tensor = tensor.view(c, h * w)
        idx = tensor > 0; t = tensor[idx]
        tensor[idx] = t.sub_(t.mean()).div_(t.std() + 1e-6)
        tensor = tensor.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, self.outchannels, dim=0)
        return tensor


# ══════════════════════════════════════════════════════════════
#  DATASET PARSER  (Palm-Auth, cross-domain, closed set)
# ══════════════════════════════════════════════════════════════

def parse_palm_auth_cross_domain(data_root, scanner_spectra, gallery_ratio=0.50, seed=42):
    """
    Collect images from roi_perspective (train domain) and roi_scanner (test domain).

    Only subjects that have at least one image in BOTH domains are kept.
    The same label map is used for all three splits → closed-set protocol.

    Returns
    -------
    train_samples   : list of (path, label)  — smartphone / perspective
    gallery_samples : list of (path, label)  — scanner, gallery half
    probe_samples   : list of (path, label)  — scanner, probe half
    num_classes     : int
    """
    rng = random.Random(seed)

    # ── Collect perspective images ─────────────────────────────────────────
    # Subject identity key: "{subject_folder}_{hand_id}" (e.g. "001_L")
    # Filename pattern: {hand_id}_{session}_{condition}.jpg  → parts[0]_parts[1]
    persp_paths = defaultdict(list)
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir = os.path.join(data_root, subject_id)
        if not os.path.isdir(subject_dir): continue
        roi_dir = os.path.join(subject_dir, "roi_perspective")
        if not os.path.isdir(roi_dir): continue
        for fname in sorted(os.listdir(roi_dir)):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
            parts = os.path.splitext(fname)[0].split("_")
            if len(parts) < 3: continue
            identity = parts[0] + "_" + parts[1]
            persp_paths[identity].append(os.path.join(roi_dir, fname))

    # ── Collect scanner images ─────────────────────────────────────────────
    # Filename pattern: {hand_id}_{session}_{spectrum}_{...}.jpg
    # Subject folder acts as subject_id; scanner identity key: "{subject_id}_{spectrum}"
    # We keep all spectra in scanner_spectra set.
    # Identity must match perspective key, so we use subject_folder + hand_id.
    scanner_paths = defaultdict(list)
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir = os.path.join(data_root, subject_id)
        if not os.path.isdir(subject_dir): continue
        scan_dir = os.path.join(subject_dir, "roi_scanner")
        if not os.path.isdir(scan_dir): continue
        for fname in sorted(os.listdir(scan_dir)):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
            parts = os.path.splitext(fname)[0].split("_")
            if len(parts) < 4: continue
            spectrum = parts[2].lower()
            if spectrum not in scanner_spectra: continue
            # Use same identity key as perspective: parts[0]_parts[1]
            identity = parts[0] + "_" + parts[1]
            scanner_paths[identity].append(os.path.join(scan_dir, fname))

    # ── Keep only shared identities ────────────────────────────────────────
    shared_ids = sorted(set(persp_paths.keys()) & set(scanner_paths.keys()))
    if len(shared_ids) == 0:
        raise ValueError("No shared identities found between perspective and scanner!")

    label_map = {ident: i for i, ident in enumerate(shared_ids)}
    num_classes = len(shared_ids)

    # ── Build train samples (all perspective images) ───────────────────────
    train_samples = []
    for ident in shared_ids:
        for path in persp_paths[ident]:
            train_samples.append((path, label_map[ident]))

    # ── Build gallery / probe (scanner images, 50/50 per identity) ─────────
    gallery_samples, probe_samples = [], []
    for ident in shared_ids:
        paths = list(scanner_paths[ident])
        rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gal]: gallery_samples.append((p, label_map[ident]))
        for p in paths[n_gal:]: probe_samples.append((p, label_map[ident]))

    # ── Stats ──────────────────────────────────────────────────────────────
    p_counts = [len(persp_paths[i])   for i in shared_ids]
    s_counts = [len(scanner_paths[i]) for i in shared_ids]
    print(f"\n  [Palm-Auth Cross-Domain | Closed Set]")
    print(f"    Shared subjects       : {num_classes}")
    print(f"    Perspective (train)   : {len(train_samples)} images  "
          f"(min={min(p_counts)} max={max(p_counts)} mean={sum(p_counts)/num_classes:.1f})")
    print(f"    Scanner    (test)     : {len(gallery_samples)+len(probe_samples)} images  "
          f"(min={min(s_counts)} max={max(s_counts)} mean={sum(s_counts)/num_classes:.1f})")
    print(f"    Gallery / Probe       : {len(gallery_samples)} / {len(probe_samples)}")
    print(f"    Spectra kept          : {sorted(scanner_spectra)}")

    return train_samples, gallery_samples, probe_samples, num_classes


# ══════════════════════════════════════════════════════════════
#  FIXED MODEL INITIALISATION
# ══════════════════════════════════════════════════════════════

def get_or_create_init_weights(net, cfg, num_classes, device):
    cache_dir    = os.path.abspath(cfg["base_results_dir"])
    os.makedirs(cache_dir, exist_ok=True)
    model_name   = type(net.module if isinstance(net, DataParallel) else net).__name__
    weights_path = os.path.join(cache_dir,
                                f"init_weights_{model_name}_nc{num_classes}.pth")
    _net = net.module if isinstance(net, DataParallel) else net
    if os.path.exists(weights_path):
        print(f"  Loading cached init weights: {weights_path}")
        _net.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"  Saving init weights: {weights_path}")
        torch.save(_net.state_dict(), weights_path)
    return net


# ══════════════════════════════════════════════════════════════
#  PYTORCH DATASETS
# ══════════════════════════════════════════════════════════════

class SingleDataset(Dataset):
    def __init__(self, samples, img_side=128):
        self.samples   = samples
        self.transform = T.Compose([T.Resize(img_side), T.ToTensor(),
                                    NormSingleROI(outchannels=1)])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("L")), label


class AugmentedDataset(Dataset):
    def __init__(self, samples, img_side=128, augment_factor=1):
        self.samples = samples; self.augment_factor = augment_factor
        self.aug_transform = T.Compose([
            T.Resize(img_side),
            T.RandomChoice([
                T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                T.RandomResizedCrop(img_side, scale=(0.8,1.0), ratio=(1.0,1.0)),
                T.RandomPerspective(distortion_scale=0.15, p=1),
                T.RandomChoice([
                    T.RandomRotation(10, interpolation=Image.BICUBIC,
                                     expand=False, center=(0.5*img_side, 0.0)),
                    T.RandomRotation(10, interpolation=Image.BICUBIC,
                                     expand=False, center=(0.0, 0.5*img_side)),
                ]),
            ]),
            T.ToTensor(), NormSingleROI(outchannels=1),
        ])
    def __len__(self): return len(self.samples) * self.augment_factor
    def __getitem__(self, index):
        real_idx = index % len(self.samples)
        path, label = self.samples[real_idx]
        return self.aug_transform(Image.open(path).convert("L")), label


# ══════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════

def run_one_epoch(model, loader, criterion, optimizer, device, phase):
    is_train = (phase == "training")
    model.train() if is_train else model.eval()
    running_loss = 0.0; running_correct = 0; total = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            if is_train: optimizer.zero_grad()
            output = model(data, target if is_train else None)
            loss   = criterion(output, target)
            if is_train: loss.backward(); optimizer.step()
            running_loss    += loss.item() * data.size(0)
            running_correct += output.data.max(1)[1].eq(target).sum().item()
            total           += data.size(0)
    return running_loss / max(total, 1), 100.0 * running_correct / max(total, 1)


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval(); feats, labels = [], []
    for imgs, labs in loader:
        feats.append(model.get_embedding(imgs.to(device)).cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def compute_eer(scores_array):
    ins  = scores_array[scores_array[:, 1] ==  1, 0]
    outs = scores_array[scores_array[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0: return 1.0, 0.0
    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s   = np.concatenate([ins, outs])
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    eer    = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))
    return eer, thresh


def evaluate(model, probe_loader, gallery_loader, device,
             out_dir=".", tag="eval"):
    probe_feats,   probe_labels   = extract_features(model, probe_loader,   device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)
    n_probe    = len(probe_feats)
    sim_matrix = probe_feats @ gallery_feats.T

    scores_list, labels_list = [], []
    for i in range(n_probe):
        for j in range(sim_matrix.shape[1]):
            scores_list.append(float(sim_matrix[i, j]))
            labels_list.append(1 if probe_labels[i] == gallery_labels[j] else -1)

    scores_arr = np.column_stack([scores_list, labels_list])
    eer, _     = compute_eer(scores_arr)

    nn_idx  = np.argmax(sim_matrix, axis=1)
    correct = sum(probe_labels[i] == gallery_labels[nn_idx[i]] for i in range(n_probe))
    rank1   = 100.0 * correct / max(n_probe, 1)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list): f.write(f"{s} {l}\n")

    print(f"  [{tag}]  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return eer, rank1


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    cfg  = CONFIG
    seed = cfg["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_results_dir = cfg["base_results_dir"]
    os.makedirs(base_results_dir, exist_ok=True)
    rst_eval = os.path.join(base_results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CompNet — Cross-Domain Closed-Set (Palm-Auth)")
    print(f"  Train domain : roi_perspective (smartphone)")
    print(f"  Test  domain : roi_scanner")
    print(f"  Protocol     : closed set (shared IDs)")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {cfg['num_epochs']}")
    print(f"  Results dir  : {base_results_dir}")
    print(f"{'='*60}")

    # ── Parse dataset ─────────────────────────────────────────────────────
    train_samples, gallery_samples, probe_samples, num_classes = \
        parse_palm_auth_cross_domain(
            data_root       = cfg["palm_auth_data_root"],
            scanner_spectra = cfg["scanner_spectra"],
            gallery_ratio   = cfg["test_gallery_ratio"],
            seed            = seed,
        )

    # ── Data loaders ──────────────────────────────────────────────────────
    img_side       = cfg["img_side"]
    batch_size     = cfg["batch_size"]
    augment_factor = cfg["augment_factor"]
    nw             = cfg["num_workers"]

    train_loader = DataLoader(
        AugmentedDataset(train_samples, img_side, augment_factor),
        batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    gallery_loader = DataLoader(
        SingleDataset(gallery_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    probe_loader = DataLoader(
        SingleDataset(probe_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    net = CompNet(num_classes,
                  embedding_dim = cfg["embedding_dim"],
                  arcface_s     = cfg["arcface_s"],
                  arcface_m     = cfg["arcface_m"],
                  dropout       = cfg["dropout"])
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    net = get_or_create_init_weights(net, cfg, num_classes, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg["lr"])
    scheduler = lr_scheduler.StepLR(optimizer, cfg["lr_step"], cfg["lr_gamma"])

    # ── Pre-training baseline (scanner → scanner, random weights) ─────────
    _net = net.module if isinstance(net, DataParallel) else net
    pre_eer, pre_r1 = evaluate(_net, probe_loader, gallery_loader,
                                device, out_dir=rst_eval,
                                tag="ep-001_pretrain")
    best_eer = pre_eer; last_eer = pre_eer; last_rank1 = pre_r1
    torch.save(_net.state_dict(),
               os.path.join(base_results_dir, "net_params_best_eer.pth"))

    train_losses, train_accs = [], []

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(cfg["num_epochs"]):
        t_loss, t_acc = run_one_epoch(
            net, train_loader, criterion, optimizer, device, "training")
        scheduler.step()
        train_losses.append(t_loss); train_accs.append(t_acc)
        _net = net.module if isinstance(net, DataParallel) else net

        if (epoch % cfg["eval_every"] == 0 and epoch > 0) or epoch == cfg["num_epochs"] - 1:
            cur_eer, cur_rank1 = evaluate(
                _net, probe_loader, gallery_loader,
                device, out_dir=rst_eval, tag=f"ep{epoch:04d}")
            last_eer, last_rank1 = cur_eer, cur_rank1
            if cur_eer < best_eer:
                best_eer = cur_eer
                torch.save(_net.state_dict(),
                           os.path.join(base_results_dir, "net_params_best_eer.pth"))
                print(f"  *** New best EER: {best_eer*100:.4f}% ***")

        if epoch % 10 == 0 or epoch == cfg["num_epochs"] - 1:
            ts        = time.strftime("%H:%M:%S")
            eer_str   = f"{last_eer*100:.4f}%"  if not math.isnan(last_eer)   else "N/A"
            rank1_str = f"{last_rank1:.2f}%"     if not math.isnan(last_rank1) else "N/A"
            print(f"  [{ts}] ep {epoch:04d} | loss={t_loss:.4f} | acc={t_acc:.2f}% | "
                  f"EER={eer_str}  Rank-1={rank1_str}")

        if epoch % cfg["save_every"] == 0 or epoch == cfg["num_epochs"] - 1:
            torch.save(_net.state_dict(),
                       os.path.join(base_results_dir, "net_params.pth"))

    # ── Final evaluation (best checkpoint) ────────────────────────────────
    best_path = os.path.join(base_results_dir, "net_params_best_eer.pth")
    if not os.path.exists(best_path):
        best_path = os.path.join(base_results_dir, "net_params.pth")
    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_path, map_location=device))
    final_eer, final_rank1 = evaluate(
        eval_net, probe_loader, gallery_loader,
        device, out_dir=rst_eval, tag="FINAL")

    # ── Save results ───────────────────────────────────────────────────────
    result_txt = os.path.join(base_results_dir, "results.txt")
    with open(result_txt, "w") as f:
        f.write("CompNet — Cross-Domain Closed-Set (Palm-Auth)\n")
        f.write("  Train domain : roi_perspective (smartphone)\n")
        f.write("  Test  domain : roi_scanner\n")
        f.write(f"  Subjects     : {num_classes}\n")
        f.write(f"  Train images : {len(train_samples)}\n")
        f.write(f"  Gallery      : {len(gallery_samples)}\n")
        f.write(f"  Probe        : {len(probe_samples)}\n\n")
        f.write(f"  EER    : {final_eer*100:.4f}%\n")
        f.write(f"  Rank-1 : {final_rank1:.2f}%\n")
    print(f"\n  Results saved to: {result_txt}")
    print(f"  EER={final_eer*100:.4f}%  Rank-1={final_rank1:.2f}%")

    with open(os.path.join(base_results_dir, "results_raw.json"), "w") as f:
        json.dump({"EER_pct": final_eer*100, "Rank1_pct": final_rank1,
                   "num_classes": num_classes,
                   "train_images": len(train_samples),
                   "gallery": len(gallery_samples),
                   "probe": len(probe_samples)}, f, indent=2)

    # ── Train curves ───────────────────────────────────────────────────────
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(train_losses, 'b'); axes[0].set_title("Train Loss")
        axes[0].set_xlabel("epoch"); axes[0].grid(True)
        axes[1].plot(train_accs,   'b'); axes[1].set_title("Train Acc (%)")
        axes[1].set_xlabel("epoch"); axes[1].grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(base_results_dir, "train_curves.png"))
        plt.close(fig)
    except Exception:
        pass


if __name__ == "__main__":
    main()
