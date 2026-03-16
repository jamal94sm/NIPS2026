"""
Reproduction of CCNet on CASIA-MS dataset
Paper: "Single Source Domain Generalization for Palm Biometrics"

Filename format: {id}_{hand}_{spectrum}_{iter}.jpg  e.g. 018_r_WHT_02.jpg
Identity key  : subject_id + hand  (left/right treated as separate classes)
Spectra       : treated as different samples of the same identity
"""

import os, sys, math, time, copy
import numpy as np
from collections import defaultdict
from PIL import Image
from sklearn import metrics
from sklearn.metrics import auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms as T
from skimage import exposure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS  — edit these lines only
# ============================================================

# --- Paths ---
DATA_PATH     = "/home/pai-ng/Jamal/CASIA-MS-ROI"
OUTPUT_DIR    = "./results_casia_ms"
GPU_ID        = "0"

# --- Dataset ---
# CASIA-MS: 460 subjects × 2 hands = 920 identities (Multi-Spec 500 variant = 500)
# Set to actual number after generate_splits() prints it, or leave 0 to auto-detect
NUM_CLASSES   = 0        # 0 = auto-detect from data
TRAIN_RATIO   = 0.5      # first 50% of samples per identity → train
IMSIDE        = 128      # input image side length (px)
OUT_CHANNELS  = 1        # grayscale

# --- Architecture (from paper Table I / code defaults) ---
COM_WEIGHT    = 0.8      # channel competition weight (α) inside Competitive Block
ARC_S         = 30.0     # ArcFace scale s
ARC_M         = 0.5      # ArcFace margin m
FC_DIM1       = 4096     # first FC layer output dim
FC_DIM2       = 2048     # second FC layer output dim (embedding size)
DROPOUT       = 0.5

# --- Training (from paper / train.py / train_cc20.py) ---
BATCH_SIZE    = 1024
EPOCH_NUM     = 3000
LR            = 0.001
LR_STEP       = 500      # StepLR step size (epochs)
LR_GAMMA      = 0.8      # StepLR multiplicative factor
WEIGHT_CE     = 0.8      # λ₁  cross-entropy loss weight
WEIGHT_CON    = 0.2      # λ₂  supervised contrastive loss weight
TEMPERATURE   = 0.07     # SupConLoss temperature τ
BASE_TEMP     = 0.07     # SupConLoss base temperature

# --- Domain adaptation (FDA + histogram matching, train_cc20.py) ---
FDA_L         = 0.1      # low-frequency ratio for FDA transfer

# --- Logging ---
TEST_INTERVAL = 500      # evaluate every N epochs
SAVE_INTERVAL = 500      # save checkpoint every N epochs

# ============================================================
# (nothing to edit below this line)
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATA SPLIT
# ============================================================

def parse_filename(fname):
    stem  = os.path.splitext(fname)[0]
    parts = stem.split("_")
    if len(parts) < 4 or not parts[0].isdigit():
        return None
    return dict(id=parts[0], hand=parts[1], spectrum=parts[2], iteration=parts[3])


def generate_splits(data_root, train_ratio):
    exts   = {".jpg", ".jpeg", ".png"}
    files  = sorted(f for f in os.listdir(data_root)
                    if os.path.splitext(f)[1].lower() in exts)
    groups = defaultdict(list)
    for f in files:
        m = parse_filename(f)
        if m is None:
            continue
        key = f"{m['id']}_{m['hand']}"
        groups[key].append(os.path.join(data_root, f))

    sorted_keys = sorted(groups.keys())
    label_map   = {k: i for i, k in enumerate(sorted_keys)}
    num_classes  = len(sorted_keys)

    train_list, test_list = [], []
    for key in sorted_keys:
        paths   = sorted(groups[key])
        n_train = max(1, int(len(paths) * train_ratio))
        lbl     = label_map[key]
        for p in paths[:n_train]:
            train_list.append((p, lbl))
        for p in paths[n_train:]:
            test_list.append((p, lbl))

    print(f"  Identities   : {num_classes}")
    print(f"  Train samples: {len(train_list)}")
    print(f"  Test  samples: {len(test_list)}")
    return train_list, test_list, num_classes


def write_txt(lst, path):
    with open(path, "w") as f:
        for img_path, label in lst:
            f.write(f"{img_path} {label}\n")


# ============================================================
# DATASET
# ============================================================

class NormSingleROI:
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        c, h, w = tensor.size()
        flat    = tensor.view(c, h * w)
        idx     = flat > 0
        t       = flat[idx]
        if t.numel() > 1:
            flat[idx] = (t - t.mean()) / (t.std() + 1e-6)
        tensor = flat.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, self.outchannels, dim=0)
        return tensor


class PalmDataset(Dataset):
    def __init__(self, samples, train=True, imside=IMSIDE, outchannels=OUT_CHANNELS):
        self.samples = samples
        self.train   = train
        self.labels  = [s[1] for s in samples]

        if train:
            self.tf = T.Compose([
                T.Resize(imside),
                T.RandomChoice([
                    T.ColorJitter(brightness=0, contrast=0.05),
                    T.RandomResizedCrop(imside, scale=(0.8, 1.0), ratio=(1., 1.)),
                    T.RandomPerspective(distortion_scale=0.15, p=1),
                    T.RandomChoice([
                        T.RandomRotation(10, expand=False,
                                         center=(0.5 * imside, 0.0)),
                        T.RandomRotation(10, expand=False,
                                         center=(0.0, 0.5 * imside)),
                    ]),
                ]),
                T.ToTensor(),
                NormSingleROI(outchannels),
            ])
        else:
            self.tf = T.Compose([
                T.Resize(imside),
                T.ToTensor(),
                NormSingleROI(outchannels),
            ])

    def __len__(self):
        return len(self.samples)

    def _load(self, idx):
        path, label = self.samples[idx]
        return self.tf(Image.open(path).convert("L")), label

    def __getitem__(self, idx):
        img1, label = self._load(idx)
        same = [i for i, l in enumerate(self.labels) if l == label]
        idx2 = idx
        if self.train and len(same) > 1:
            while idx2 == idx:
                idx2 = int(np.random.choice(same))
        img2, _ = self._load(idx2)
        return (img1, img2), label


# ============================================================
# MODEL  (ccnet_2.py — 2-channel input, Gabor + conv blocks)
# ============================================================

class GaborConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, ksize, stride=1, padding=0, init_ratio=1.):
        super().__init__()
        r            = init_ratio
        self.ch_in   = ch_in
        self.ch_out  = ch_out
        self.ksize   = ksize
        self.stride  = stride
        self.padding = padding
        self.kernel  = None
        self.gamma   = nn.Parameter(torch.FloatTensor([2.0]))
        self.sigma   = nn.Parameter(torch.FloatTensor([9.2 * r]))
        self.theta   = nn.Parameter(
            torch.arange(ch_out).float() * math.pi / ch_out, requires_grad=False)
        self.f       = nn.Parameter(torch.FloatTensor([0.057 / r]))
        self.psi     = nn.Parameter(torch.FloatTensor([0.0]),    requires_grad=False)

    def _build_bank(self):
        xm  = self.ksize // 2
        rng = torch.arange(-xm, xm + 1).float()
        y   = rng.view(1, -1).repeat(self.ch_out, self.ch_in, self.ksize, 1)
        x   = rng.view(-1, 1).repeat(self.ch_out, self.ch_in, 1, self.ksize)
        x   = x.to(self.sigma.device)
        y   = y.to(self.sigma.device)
        th  = self.theta.view(-1, 1, 1, 1)
        xt  =  x * torch.cos(th) + y * torch.sin(th)
        yt  = -x * torch.sin(th) + y * torch.cos(th)
        gb  = -torch.exp(
            -0.5 * ((self.gamma * xt) ** 2 + yt ** 2)
            / (8 * self.sigma.view(-1, 1, 1, 1) ** 2)
        ) * torch.cos(2 * math.pi * self.f.view(-1, 1, 1, 1) * xt
                      + self.psi.view(-1, 1, 1, 1))
        return gb - gb.mean(dim=[2, 3], keepdim=True)

    def forward(self, x):
        self.kernel = self._build_bank()
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)


class SELayer(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(ch, ch, bias=False), nn.ReLU(inplace=True),
            nn.Linear(ch, ch, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)


class CompetitiveBlock(nn.Module):
    def __init__(self, ch_in, n_comp, ksize, weight, init_ratio=1., o1=32):
        super().__init__()
        nc2 = n_comp * 2
        nc4 = n_comp * 4
        self.g1 = GaborConv2d(ch_in, n_comp, ksize, 2, ksize // 2, init_ratio)
        self.g2 = GaborConv2d(nc2,   nc2,    ksize, 2, ksize // 2, init_ratio)

        if ksize == 35:
            self.c1a = nn.Conv2d(ch_in, n_comp, 7, 1, 0)
            self.c1b = nn.Conv2d(n_comp, n_comp, 5, 2, 5)
            self.c2a = nn.Conv2d(nc2,    nc2,    7, 1, 0)
            self.c2b = nn.Conv2d(nc2,    nc2,    5, 2, 5)
        elif ksize == 17:
            self.c1a = nn.Conv2d(ch_in, n_comp, 5, 1, 0)
            self.c1b = nn.Conv2d(n_comp, n_comp, 3, 2, 3)
            self.c2a = nn.Conv2d(nc2,    nc2,    5, 1, 0)
            self.c2b = nn.Conv2d(nc2,    nc2,    3, 2, 3)
        else:                                           # ksize == 7
            self.c1a = nn.Conv2d(ch_in, n_comp, 3, 1, 0)
            self.c1b = nn.Conv2d(n_comp, n_comp, 1, 2, 1)
            self.c2a = nn.Conv2d(nc2,    nc2,    3, 1, 0)
            self.c2b = nn.Conv2d(nc2,    nc2,    1, 2, 1)

        self.sm_c = nn.Softmax(dim=1)
        self.sm_h = nn.Softmax(dim=2)
        self.sm_w = nn.Softmax(dim=3)
        self.se1  = SELayer(nc2)
        self.se2  = SELayer(nc4)
        self.ppu1 = nn.Conv2d(nc2, o1 // 2, 5, 2, 0)
        self.ppu2 = nn.Conv2d(nc4, o1 // 2, 5, 2, 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.wc   = weight
        self.ws   = (1. - weight) / 2.

    def _compete(self, x):
        return self.wc * self.sm_c(x) + self.ws * (self.sm_h(x) + self.sm_w(x))

    def forward(self, x):
        # 1st order
        f = torch.cat([self.g1(x), self.c1b(self.c1a(x))], dim=1)
        x1 = self.pool(self.ppu1(self.se1(self._compete(f))))
        # 2nd order
        f = torch.cat([self.g2(f), self.c2b(self.c2a(f))], dim=1)
        x2 = self.pool(self.ppu2(self.se2(self._compete(f))))
        return torch.cat([x1.flatten(1), x2.flatten(1)], dim=1)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_f, out_f, s=ARC_S, m=ARC_M):
        super().__init__()
        self.s   = s
        self.w   = Parameter(torch.FloatTensor(out_f, in_f))
        nn.init.xavier_uniform_(self.w)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        cos = F.linear(F.normalize(x), F.normalize(self.w))
        if self.training and label is not None:
            sin = torch.sqrt((1. - cos ** 2).clamp(0., 1.))
            phi = cos * self.cos_m - sin * self.sin_m
            phi = torch.where(cos > self.th, phi, cos - self.mm)
            oh  = torch.zeros_like(cos).scatter_(1, label.view(-1, 1).long(), 1)
            return ((oh * phi) + ((1. - oh) * cos)) * self.s
        return self.s * cos


class CCNet(nn.Module):
    """
    CCNet  (ccnet_2.py variant):
      3 Competitive Blocks (CB1 ksize=35 / CB2 ksize=17 / CB3 ksize=7)
      + 2-layer FC  + Dropout  + ArcFace head
      Input: 2-channel (histogram-matched | FDA-transferred)
    """
    def __init__(self, num_classes, weight=COM_WEIGHT):
        super().__init__()
        self.cb1  = CompetitiveBlock(2,  9, 35, weight, init_ratio=1.00)
        self.cb2  = CompetitiveBlock(2, 36, 17, weight, init_ratio=0.50)
        self.cb3  = CompetitiveBlock(2,  9,  7, weight, init_ratio=0.25)
        self.fc   = nn.Linear(13152, FC_DIM1)
        self.fc1  = nn.Linear(FC_DIM1, FC_DIM2)
        self.drop = nn.Dropout(DROPOUT)
        self.arc  = ArcMarginProduct(FC_DIM2, num_classes, s=ARC_S, m=ARC_M)

    def _backbone(self, x):
        return torch.cat([self.cb1(x), self.cb2(x), self.cb3(x)], dim=1)

    def forward(self, x, y=None):
        h1  = self.fc(self._backbone(x))
        h2  = self.fc1(h1)
        fe  = torch.cat([h1, h2], dim=1)
        out = self.arc(self.drop(h2), y)
        return out, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        return F.normalize(self.fc1(self.fc(self._backbone(x))), dim=-1)

    def getFeatureCode2(self, x):
        h1 = self.fc(self._backbone(x))
        h2 = self.fc1(h1)
        return F.normalize(torch.cat([h1, h2], dim=1), dim=-1)


# ============================================================
# LOSSES
# ============================================================

class SupConLoss(nn.Module):
    def __init__(self, temperature=TEMPERATURE, base_temperature=BASE_TEMP):
        super().__init__()
        self.T    = temperature
        self.base = base_temperature

    def forward(self, features, labels):
        dev  = features.device
        bsz  = features.shape[0]
        n    = features.shape[1]               # number of views (2)
        mask = torch.eq(labels.view(-1, 1),
                        labels.view(1, -1)).float().to(dev)
        contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
        dot      = torch.div(torch.matmul(contrast, contrast.T), self.T)
        lm, _    = torch.max(dot, dim=1, keepdim=True)
        logits   = dot - lm.detach()
        mask     = mask.repeat(n, n)
        lmask    = 1. - torch.eye(bsz * n, device=dev)
        mask     = mask * lmask
        exp_log  = torch.exp(logits) * lmask
        log_prob = logits - torch.log(exp_log.sum(1, keepdim=True) + 1e-9)
        denom    = mask.sum(1).clamp(min=1.)
        loss     = -(self.T / self.base) * (mask * log_prob).sum(1) / denom
        return loss.mean()


# ============================================================
# DOMAIN ADAPTATION  (histogram matching + FDA, train_cc20.py)
# ============================================================

def _fda(src, tgt, L=FDA_L):
    fs  = torch.fft.rfft2(src, dim=(-2, -1))
    ft  = torch.fft.rfft2(tgt, dim=(-2, -1))
    as_, ps = torch.abs(fs), torch.angle(fs)
    at        = torch.abs(ft)
    _, _, h, w = as_.shape
    b  = int(np.floor(0.5 * min(h, w * 2) * L))
    if b > 0:
        as_[:, :, :b,      :b] = at[:, :, :b,      :b]
        as_[:, :, h-b+1:h, :b] = at[:, :, h-b+1:h, :b]
    out = torch.fft.irfft2(torch.complex(torch.cos(ps) * as_,
                                          torch.sin(ps) * as_),
                            dim=(-2, -1), s=[h, w * 2])
    return out[..., :src.shape[-2], :src.shape[-1]]


def _hist(src_batch, tgt_batch):
    rows = []
    for s, t in zip(src_batch, tgt_batch):
        s_np = s.permute(1, 2, 0).numpy()
        t_np = t.permute(1, 2, 0).numpy()
        rows.append(torch.from_numpy(
            exposure.match_histograms(s_np, t_np)).permute(2, 0, 1))
    return torch.stack(rows, dim=0).float()


def make_2ch(src, tgt):
    """[hist-matched | FDA-transferred]  → 2-channel input tensor"""
    return torch.cat([_hist(src, tgt), _fda(src, tgt)], dim=1)


def make_2ch_identity(x):
    """At test time: duplicate single channel to get 2-channel input"""
    return torch.cat([x, x], dim=1)


# ============================================================
# EVALUATION
# ============================================================

def compute_eer(ins, outs):
    if ins.mean() < outs.mean():
        ins, outs = -ins, -outs
    y  = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    sc = np.concatenate([ins, outs])
    fpr, tpr, _ = metrics.roc_curve(y, sc, pos_label=1)
    roc_auc     = auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100., roc_auc


def extract_features(model, loader):
    model.eval()
    feats, ids = [], []
    with torch.no_grad():
        for (d, _), target in loader:
            codes = model.getFeatureCode(make_2ch_identity(d).to(device))
            feats.append(codes.cpu().numpy())
            ids.append(target.numpy())
    return np.concatenate(feats), np.concatenate(ids)


def evaluate(model, tr_loader, te_loader, tag, out_dir):
    print(f"\n--- Evaluation: {tag} ---")
    ft_tr, id_tr = extract_features(model, tr_loader)
    ft_te, id_te = extract_features(model, te_loader)

    s, l = [], []
    for i in range(len(ft_te)):
        for j in range(len(ft_tr)):
            sim = np.dot(ft_te[i], ft_tr[j])
            dis = np.arccos(np.clip(sim, -1., 1.)) / np.pi
            s.append(dis)
            l.append(1 if id_te[i] == id_tr[j] else -1)

    s, l = np.array(s), np.array(l)
    ins   = 1. - s[l ==  1]
    outs  = 1. - s[l == -1]
    eer, roc_auc = compute_eer(ins, outs)

    # Rank-1
    cnt, corr = 0, 0
    for i in range(len(ft_te)):
        dis = s[cnt: cnt + len(ft_tr)]
        cnt += len(ft_tr)
        if id_te[i] == id_tr[np.argmin(dis)]:
            corr += 1
    rank1 = 100. * corr / len(ft_te)

    print(f"  EER    : {eer:.4f}%")
    print(f"  Rank-1 : {rank1:.3f}%")
    print(f"  AUC    : {roc_auc:.6f}")

    ev_dir = os.path.join(out_dir, tag)
    os.makedirs(ev_dir, exist_ok=True)
    with open(os.path.join(ev_dir, "scores.txt"), "w") as f:
        for sc, lb in zip(1. - s, l):
            f.write(f"{sc:.6f} {lb}\n")
    with open(os.path.join(ev_dir, "results.txt"), "w") as f:
        f.write(f"EER    : {eer:.4f}%\n")
        f.write(f"Rank-1 : {rank1:.3f}%\n")
        f.write(f"AUC    : {roc_auc:.6f}\n")

    return eer, rank1


# ============================================================
# TRAINING LOOP  (mirrors train_cc20.py → fit())
# ============================================================

def fit_epoch(epoch, model, src_loader, tgt_iter_ref,
              criterion, con_crit, optimizer):
    model.train()
    run_loss, run_corr, total = 0., 0, 0

    for (src1, src2), targets in src_loader:
        try:
            (tgt1, _), _ = next(tgt_iter_ref[0])
        except StopIteration:
            tgt_iter_ref[0] = iter(tgt_iter_ref[1])   # reset
            (tgt1, _), _   = next(tgt_iter_ref[0])

        targets = targets.to(device)

        data     = make_2ch(src1, tgt1).to(device)
        data_con = make_2ch(src2, tgt1).to(device)

        optimizer.zero_grad()
        out1, fe1 = model(data,     targets)
        out2, fe2 = model(data_con, targets)
        fe        = torch.stack([fe1, fe2], dim=1)

        ce  = criterion(out1, targets) + criterion(out2, targets)
        con = con_crit(fe, targets)
        loss = WEIGHT_CE * ce + WEIGHT_CON * con

        loss.backward()
        optimizer.step()

        run_loss += loss.item() * targets.size(0)
        run_corr += out1.argmax(1).eq(targets).sum().item()
        total    += targets.size(0)

    loss_avg = run_loss / total
    acc      = 100. * run_corr / total
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:4d} | loss {loss_avg:.5f} | train acc {acc:.2f}%"
              f"  [{time.strftime('%H:%M:%S')}]")
    return loss_avg, acc


# ============================================================
# MAIN
# ============================================================

def main():
    # ── 1. Splits ────────────────────────────────────────────
    print("\n[1] Generating train/test splits ...")
    train_list, test_list, num_classes = generate_splits(DATA_PATH, TRAIN_RATIO)
    nc = NUM_CLASSES if NUM_CLASSES > 0 else num_classes
    print(f"  Using num_classes = {nc}")

    write_txt(train_list, os.path.join(OUTPUT_DIR, "train.txt"))
    write_txt(test_list,  os.path.join(OUTPUT_DIR, "test.txt"))

    # ── 2. Loaders ───────────────────────────────────────────
    print("\n[2] Building dataloaders ...")
    train_ds  = PalmDataset(train_list, train=True)
    test_ds   = PalmDataset(test_list,  train=False)
    target_ds = PalmDataset(test_list,  train=True)   # target domain (unlabelled)

    kw = dict(num_workers=4, pin_memory=True)
    train_loader  = DataLoader(train_ds,  batch_size=BATCH_SIZE, shuffle=True,
                               drop_last=True,  **kw)
    test_loader   = DataLoader(test_ds,   batch_size=BATCH_SIZE, shuffle=False, **kw)
    target_loader = DataLoader(target_ds, batch_size=BATCH_SIZE, shuffle=True,
                               drop_last=True,  **kw)

    # ── 3. Model ─────────────────────────────────────────────
    print(f"\n[3] Building CCNet  (num_classes={nc}, com_weight={COM_WEIGHT}) ...")
    net      = CCNet(nc, COM_WEIGHT).to(device)
    best_net = CCNet(nc, COM_WEIGHT).to(device)

    criterion = nn.CrossEntropyLoss()
    con_crit  = SupConLoss(TEMPERATURE, BASE_TEMP)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    # ── 4. Training ──────────────────────────────────────────
    print("\n[4] Training ...")
    best_acc          = 0.
    loss_hist, acc_hist = [], []
    tgt_iter_ref      = [iter(target_loader), target_loader]   # mutable ref

    for epoch in range(EPOCH_NUM):
        loss, acc = fit_epoch(epoch, net, train_loader, tgt_iter_ref,
                              criterion, con_crit, optimizer)
        scheduler.step()
        loss_hist.append(loss)
        acc_hist.append(acc)

        if acc >= best_acc:
            best_acc = acc
            torch.save(net.state_dict(),
                       os.path.join(OUTPUT_DIR, "best_model.pth"))
            best_net.load_state_dict(copy.deepcopy(net.state_dict()))

        if epoch % SAVE_INTERVAL == 0 and epoch > 0:
            torch.save(net.state_dict(),
                       os.path.join(OUTPUT_DIR, f"epoch_{epoch}.pth"))

        if epoch % TEST_INTERVAL == 0 and epoch > 0:
            evaluate(net, train_loader, test_loader,
                     f"ep{epoch}", OUTPUT_DIR)
            net.train()

    # ── 5. Final evaluation ──────────────────────────────────
    print("\n[5] Final evaluation ...")
    evaluate(net,      train_loader, test_loader, "last", OUTPUT_DIR)
    evaluate(best_net, train_loader, test_loader, "best", OUTPUT_DIR)

    # ── 6. Training curve ────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(loss_hist); ax1.set_title("Train Loss");     ax1.set_xlabel("Epoch")
    ax2.plot(acc_hist);  ax2.set_title("Train Acc (%)"); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curve.png"))
    plt.close()

    torch.save(net.state_dict(), os.path.join(OUTPUT_DIR, "last_model.pth"))
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
