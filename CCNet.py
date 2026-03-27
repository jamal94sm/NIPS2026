"""
CCNet on CASIA-MS Dataset
==================================================
Single-file implementation with closed-set and open-set protocols.
Faithfully preserves the official CCNet architecture, SupConLoss,
paired-sample contrastive training, and NormSingleROI normalisation.

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

Architecture: CCNet (unchanged from official repo)
  CompetitiveBlock_Mul_Ord_Comp (multi-order Gabor + spatial/channel competition + SE)
  FC: 13152 → 4096 → 2048 + ArcFace output
  Embedding: L2-normalised 2048-d
"""

# ==============================================================
#  CONFIG  — edit this block only
# ==============================================================
CONFIG = {
    "protocol"        : "open-set",   # "closed-set" | "open-set"
    "data_root"       : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "results_dir"     : "./rst_ccnet_casia_ms",
    "img_side"        : 128,            # input image size (128×128 → fc=13152)
    "batch_size"      : 1024,           # CCNet default
    "num_epochs"      : 500,            # CCNet default is 3000; adjust as needed
    "lr"              : 0.001,          # CCNet default
    "lr_step"         : 500,            # CCNet default (redstep)
    "lr_gamma"        : 0.8,            # CCNet default
    "dropout"         : 0.5,            # CCNet default
    "arcface_s"       : 30.0,
    "arcface_m"       : 0.50,
    "comp_weight"     : 0.8,            # channel competition weight (CCNet default)
    "ce_weight"       : 0.8,            # cross-entropy loss weight (weight1)
    "con_weight"      : 0.2,            # contrastive loss weight (weight2)
    "temperature"     : 0.07,           # SupConLoss temperature (CCNet default)
    "embedding_dim"   : 2048,           # CCNet fc1 output
    "train_ratio"     : 0.80,
    "gallery_ratio"   : 0.50,
    "val_ratio"       : 0.10,
    "random_seed"     : 42,
    "save_every"      : 10,
    "eval_every"      : 50,
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
from torch.nn import Parameter, DataParallel
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
#  SUPCONLOSS  (exact copy from CCNet/loss.py)
# ══════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# ══════════════════════════════════════════════════════════════
#  MODEL  (exact copy of CCNet/models/ccnet.py — unchanged)
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    """Learnable Gabor Convolution (LGC) layer."""
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()
        self.channel_in  = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.init_ratio  = init_ratio
        self.kernel      = 0

        if init_ratio <= 0:
            init_ratio = 1.0

        _SIGMA = 9.2   * init_ratio
        _FREQ  = 0.057 / init_ratio
        _GAMMA = 2.0

        self.gamma = nn.Parameter(torch.FloatTensor([_GAMMA]), requires_grad=True)
        self.sigma = nn.Parameter(torch.FloatTensor([_SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([_FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def genGaborBank(self, kernel_size, channel_in, channel_out,
                     sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin, ymin = -xmax, -ymax
        ksize = xmax - xmin + 1

        y_0 = torch.arange(ymin, ymax + 1).float()
        x_0 = torch.arange(xmin, xmax + 1).float()

        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1)
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize)
        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        x_theta = ( x * torch.cos(theta.view(-1, 1, 1, 1))
                  + y * torch.sin(theta.view(-1, 1, 1, 1)))
        y_theta = (-x * torch.sin(theta.view(-1, 1, 1, 1))
                  + y * torch.cos(theta.view(-1, 1, 1, 1)))

        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2)
            / (8 * sigma.view(-1, 1, 1, 1) ** 2)
        ) * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta
                      + psi.view(-1, 1, 1, 1))
        gb = gb - gb.mean(dim=[2, 3], keepdim=True)
        return gb

    def forward(self, x):
        kernel = self.genGaborBank(
            self.kernel_size, self.channel_in, self.channel_out,
            self.sigma, self.gamma, self.theta, self.f, self.psi)
        self.kernel = kernel
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer."""
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CompetitiveBlock_Mul_Ord_Comp(nn.Module):
    """
    Multi-Order Comprehensive Competition Block.
    1st order: LGC → spatial+channel competition → SE → conv → pool
    2nd order: LGC(on 1st Gabor output) → spatial+channel competition → SE → conv → pool
    Output: concatenation of 1st and 2nd order features.
    """
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 weight, init_ratio=1, o1=32, o2=12):
        super(CompetitiveBlock_Mul_Ord_Comp, self).__init__()
        self.channel_in   = channel_in
        self.n_competitor  = n_competitor
        self.init_ratio    = init_ratio

        self.gabor_conv2d = GaborConv2d(
            channel_in=channel_in, channel_out=n_competitor,
            kernel_size=ksize, stride=2, padding=ksize // 2,
            init_ratio=init_ratio)
        self.gabor_conv2d2 = GaborConv2d(
            channel_in=n_competitor, channel_out=n_competitor,
            kernel_size=ksize, stride=2, padding=ksize // 2,
            init_ratio=init_ratio)

        # competition mechanisms
        self.argmax   = nn.Softmax(dim=1)   # channel competition
        self.argmax_x = nn.Softmax(dim=2)   # spatial-x competition
        self.argmax_y = nn.Softmax(dim=3)   # spatial-y competition

        # PPU layers
        self.conv1_1 = nn.Conv2d(n_competitor, o1 // 2, 5, 2, 0)
        self.conv2_1 = nn.Conv2d(n_competitor, o1 // 2, 5, 2, 0)
        self.maxpool = nn.MaxPool2d(2, 2)

        # SE layers
        self.se1 = SELayer(n_competitor)
        self.se2 = SELayer(n_competitor)

        # competition weights
        self.weight_chan = weight
        self.weight_spa  = (1 - weight) / 2

    def forward(self, x):
        # 1st order
        x = self.gabor_conv2d(x)

        x1_1 = self.argmax(x)
        x1_2 = self.argmax_x(x)
        x1_3 = self.argmax_y(x)
        x_1 = self.weight_chan * x1_1 + self.weight_spa * (x1_2 + x1_3)

        x_1 = self.se1(x_1)
        x_1 = self.conv1_1(x_1)
        x_1 = self.maxpool(x_1)

        # 2nd order
        x = self.gabor_conv2d2(x)

        x2_1 = self.argmax(x)
        x2_2 = self.argmax_x(x)
        x2_3 = self.argmax_y(x)
        x_2 = self.weight_chan * x2_1 + self.weight_spa * (x2_2 + x2_3)

        x_2 = self.se2(x_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.maxpool(x_2)

        xx = torch.cat((x_1.view(x_1.shape[0], -1),
                         x_2.view(x_2.shape[0], -1)), dim=1)
        return xx


class ArcMarginProduct(nn.Module):
    """ArcFace angular margin product layer (CCNet version)."""
    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, inp, label=None):
        cosine = F.linear(F.normalize(inp), F.normalize(self.weight))
        if self.training:
            assert label is not None
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi  = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            # CCNet: no assertion on label in eval mode
            output = self.s * cosine
        return output


class ccnet(nn.Module):
    """
    CCNet = CB1 // CB2 // CB3 + FC(13152→4096) + FC1(4096→2048) + Dropout + ArcFace
    forward returns (logits, normalised_features_for_contrastive)
    """
    def __init__(self, num_classes, weight=0.8):
        super(ccnet, self).__init__()
        self.num_classes = num_classes

        self.cb1 = CompetitiveBlock_Mul_Ord_Comp(
            channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17,
            init_ratio=1, weight=weight)
        self.cb2 = CompetitiveBlock_Mul_Ord_Comp(
            channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8,
            init_ratio=0.5, o2=24, weight=weight)
        self.cb3 = CompetitiveBlock_Mul_Ord_Comp(
            channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3,
            init_ratio=0.25, weight=weight)

        self.fc       = nn.Linear(13152, 4096)
        self.fc1      = nn.Linear(4096, 2048)
        self.drop     = nn.Dropout(p=0.5)
        self.arclayer_ = ArcMarginProduct(2048, num_classes, s=30, m=0.5,
                                           easy_margin=False)

    def forward(self, x, y=None):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)
        x = torch.cat((x1, x2, x3), dim=1)

        x1 = self.fc(x)
        x  = self.fc1(x1)
        fe = torch.cat((x1, x), dim=1)       # 4096 + 2048 = 6144-d
        x  = self.drop(x)
        x  = self.arclayer_(x, y)

        return x, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        """Return L2-normalised 2048-d embedding."""
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        x = self.fc1(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x


# ══════════════════════════════════════════════════════════════
#  NORMALISATION  (exact copy from CCNet/models/dataset.py)
# ══════════════════════════════════════════════════════════════

class NormSingleROI(object):
    """Normalize the input image (exclude the black region) to 0 mean, 1 std."""
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
    Dataset for CASIA-MS ROI images with paired same-class sampling.
    Matches CCNet/models/dataset.py:
      - Training: returns [img, img2] where img2 is a random DIFFERENT
        sample of the same identity (for SupConLoss).
      - Testing:  returns [img, img] (same image duplicated).
    Uses NormSingleROI + CCNet-specific augmentation.
    """
    def __init__(self, samples, img_side=128, train=False):
        self.samples  = samples
        self.img_side = img_side
        self.train    = train

        # build label → indices map for same-class sampling
        self.label2indices = defaultdict(list)
        for idx, (_, label) in enumerate(samples):
            self.label2indices[label].append(idx)

        if train:
            self.transforms = T.Compose([
                T.Resize(img_side),
                T.RandomChoice(transforms=[
                    T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                    T.RandomResizedCrop(size=img_side, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                    T.RandomPerspective(distortion_scale=0.15, p=1),
                    T.RandomChoice(transforms=[
                        T.RandomRotation(degrees=10, expand=False,
                                         center=(0.5 * img_side, 0.0)),
                        T.RandomRotation(degrees=10, expand=False,
                                         center=(0.0, 0.5 * img_side)),
                    ]),
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

        # CCNet paired sampling: training → different same-class sample
        if self.train:
            same_class = self.label2indices[label]
            idx2 = idx
            while idx2 == idx and len(same_class) > 1:
                idx2 = random.choice(same_class)
            path2, _ = self.samples[idx2]
        else:
            path2 = path  # testing: same image

        img  = Image.open(path).convert("L")
        img  = self.transforms(img)
        img2 = Image.open(path2).convert("L")
        img2 = self.transforms(img2)

        return [img, img2], label


class CASIAMSDatasetSingle(Dataset):
    """
    Single-image dataset for feature extraction during evaluation.
    No paired sampling — just returns (image, label).
    """
    def __init__(self, samples, img_side=128):
        self.samples  = samples
        self.img_side = img_side
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
    """Scan data_root for {subjectID}_{handSide}_{spectrum}_{iter}.jpg"""
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
        identity = f"{parts[0]}_{parts[1]}"
        id2paths[identity].append(os.path.join(data_root, fname))
    return dict(id2paths)


def make_label_map(identities_sorted):
    return {ident: idx for idx, ident in enumerate(sorted(identities_sorted))}


def split_closed_set(id2paths, train_ratio=0.80, seed=42):
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
    rng = random.Random(seed)
    identities = sorted(id2paths.keys())
    rng_ids = identities[:]
    rng.shuffle(rng_ids)

    n_train = max(1, int(len(rng_ids) * train_ratio))
    train_ids = sorted(rng_ids[:n_train])
    test_ids  = sorted(rng_ids[n_train:])

    train_label_map = make_label_map(train_ids)
    test_label_map  = make_label_map(test_ids)

    train_samples, val_samples = [], []
    gallery_samples, probe_samples = [], []

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
    print(f"             train: {len(train_samples)} | val: {len(val_samples)} | "
          f"gallery: {len(gallery_samples)} | probe: {len(probe_samples)}")
    return (train_samples, val_samples, gallery_samples, probe_samples,
            train_label_map, test_label_map)


# ══════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_features(net, data_loader, device):
    """Extract L2-normalised embeddings from a single-image DataLoader."""
    net.eval()
    feats_list, labels_list = [], []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            codes = net.getFeatureCode(data)
            feats_list.append(codes.cpu().numpy())
            labels_list.append(target.numpy())
    return np.concatenate(feats_list, axis=0), np.concatenate(labels_list, axis=0)


# ══════════════════════════════════════════════════════════════
#  MATCHING & METRICS
# ══════════════════════════════════════════════════════════════

def angular_distance(f1, f2):
    """Normalised angular distance (matches CCNet test.py)."""
    cos = np.dot(f1, f2)
    return np.arccos(np.clip(cos, -1.0, 1.0)) / np.pi


def compute_eer(scores, labels):
    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels)
    sim = -scores
    in_scores  = sim[labels ==  1]
    out_scores = sim[labels == -1]
    y    = np.concatenate([np.ones(len(in_scores)), np.zeros(len(out_scores))])
    sall = np.concatenate([in_scores, out_scores])
    fpr, tpr, thresholds = roc_curve(y, sall, pos_label=1)
    roc_auc = auc(fpr, tpr)
    eer    = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))
    diff    = np.abs(fpr - (1 - tpr))
    idx     = np.argmin(diff)
    eer_half = (fpr[idx] + (1 - tpr[idx])) / 2.0
    return eer, thresh, roc_auc, eer_half, fpr, tpr, thresholds


def compute_rank1(probe_feats, probe_labels,
                  gallery_feats, gallery_labels, dist_matrix=None):
    n_probe   = probe_feats.shape[0]
    n_gallery = gallery_feats.shape[0]
    if dist_matrix is None:
        dist = np.zeros((n_probe, n_gallery))
        for i in range(n_probe):
            for j in range(n_gallery):
                dist[i, j] = angular_distance(probe_feats[i], gallery_feats[j])
    else:
        dist = dist_matrix
    correct = 0
    for i in range(n_probe):
        best_j = int(np.argmin(dist[i]))
        if probe_labels[i] == gallery_labels[best_j]:
            correct += 1
    return correct / n_probe * 100.0, dist


def compute_aggregated_eer(dist_matrix, prb_labels, gal_labels):
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
    fpr_pct, tpr_pct, fnr_pct = fpr * 100, tpr * 100, fnr * 100
    pdf_path = os.path.join(out_dir, f"roc_det_{tag}.pdf")
    with PdfPages(pdf_path) as pdf:
        plt.figure()
        plt.plot(fpr_pct, tpr_pct, "b-^", label="ROC")
        plt.plot(np.linspace(0, 100, 101), np.linspace(100, 0, 101), "k-", label="EER")
        plt.xlim([0, 5]); plt.ylim([90, 100])
        plt.legend(); plt.grid(True)
        plt.title(f"ROC  |  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
        plt.xlabel("FAR (%)"); plt.ylabel("GAR (%)")
        plt.savefig(os.path.join(out_dir, f"ROC_{tag}.png"))
        pdf.savefig(); plt.close()

        plt.figure()
        plt.plot(fpr_pct, fnr_pct, "b-^", label="DET")
        plt.plot(np.linspace(0, 100, 101), np.linspace(0, 100, 101), "k-", label="EER")
        plt.xlim([0, 5]); plt.ylim([0, 5])
        plt.legend(); plt.grid(True)
        plt.title("DET curve"); plt.xlabel("FAR (%)"); plt.ylabel("FRR (%)")
        plt.savefig(os.path.join(out_dir, f"DET_{tag}.png"))
        pdf.savefig(); plt.close()

        plt.figure()
        plt.plot(thresholds, fpr_pct, "r-.", label="FAR")
        plt.plot(thresholds, fnr_pct, "b-^", label="FRR")
        plt.legend(); plt.grid(True)
        plt.title("FAR and FRR Curves"); plt.xlabel("Threshold"); plt.ylabel("FAR / FRR (%)")
        plt.savefig(os.path.join(out_dir, f"FAR_FRR_{tag}.png"))
        pdf.savefig(); plt.close()


def plot_loss_acc(train_losses, val_losses, train_acc, val_acc, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ep = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(ep, train_losses, "b", label="train loss")
    plt.plot(ep, val_losses, "r", label="val loss")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.savefig(os.path.join(out_dir, "losses.png")); plt.close()

    plt.figure()
    plt.plot(ep, train_acc, "b", label="train acc")
    plt.plot(ep, val_acc, "r", label="val acc")
    plt.legend(); plt.grid(True)
    plt.xlabel("epoch"); plt.ylabel("accuracy (%)")
    plt.savefig(os.path.join(out_dir, "accuracy.png")); plt.close()


def plot_gi_histogram(in_scores, out_scores, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    samples = 100
    in_arr, out_arr = np.array(in_scores), np.array(out_scores)
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
    plt.legend(fontsize=13); plt.xlabel("Matching Score", fontsize=13)
    plt.ylabel("Percentage (%)", fontsize=13)
    plt.ylim([0, 1.2 * max(hi.max(), ho.max())])
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"GI_{tag}.png")); plt.close()


# ══════════════════════════════════════════════════════════════
#  TRAINING  (matches CCNet/train.py loss & logic)
# ══════════════════════════════════════════════════════════════

def run_one_epoch(epoch, model, loader, criterion, con_criterion,
                  optimizer, device, phase="training",
                  ce_weight=0.8, con_weight=0.2):
    """
    CCNet training / validation epoch.
    Loss = ce_weight * CrossEntropy + con_weight * SupConLoss

    Dataset returns [img, img2] pairs for contrastive learning.
    """
    if phase == "training":
        model.train()
    else:
        model.eval()

    running_loss    = 0.0
    running_correct = 0
    num_samples     = 0

    for datas, target in loader:
        data     = datas[0].to(device)
        data_con = datas[1].to(device)
        target   = target.to(device)

        if phase == "training":
            optimizer.zero_grad()
            output, fe1 = model(data, target)
            output2, fe2 = model(data_con, target)
            fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                output, fe1 = model(data, None)
                output2, fe2 = model(data_con, None)
                fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)

        ce  = criterion(output, target)
        ce2 = con_criterion(fe, target)
        loss = ce_weight * ce + con_weight * ce2

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

def evaluate(net, probe_loader, gallery_loader, device, out_dir, tag="eval"):
    """
    Evaluation using angular distance on L2-normalised embeddings.
    Uses single-image loaders (CASIAMSDatasetSingle).
    Always computes pairwise EER, Rank-1, and aggregated EER.
    """
    os.makedirs(out_dir, exist_ok=True)

    print("  Extracting gallery features …")
    gal_feats, gal_labels = extract_features(net, gallery_loader, device)
    print("  Extracting probe features …")
    prb_feats, prb_labels = extract_features(net, probe_loader, device)

    n_probe, n_gallery = prb_feats.shape[0], gal_feats.shape[0]
    print(f"  probe: {n_probe}  gallery: {n_gallery}")

    # pairwise angular distances
    print("  Computing pairwise distances …")
    s, l = [], []
    dist_matrix = np.zeros((n_probe, n_gallery))
    for i in range(n_probe):
        for j in range(n_gallery):
            d = angular_distance(prb_feats[i], gal_feats[j])
            dist_matrix[i, j] = d
            s.append(d)
            l.append(1 if prb_labels[i] == gal_labels[j] else -1)

    save_scores_txt(s, l, os.path.join(out_dir, f"scores_{tag}.txt"))

    # Pairwise EER
    eer, thresh, roc_auc, eer_half, fpr, tpr, thresholds = compute_eer(s, l)
    fnr = 1 - tpr
    print(f"  Pairwise EER: {eer*100:.4f}%  |  thresh: {thresh:.4f}  |  AUC: {roc_auc:.6f}")
    print(f"  Pairwise EER½: {eer_half*100:.4f}%")

    # Rank-1
    rank1, _ = compute_rank1(prb_feats, prb_labels, gal_feats, gal_labels,
                              dist_matrix=dist_matrix)
    print(f"  Rank-1 acc: {rank1:.3f}%")

    # GI histogram + plots
    in_scores  = [s[k] for k in range(len(s)) if l[k] ==  1]
    out_scores = [s[k] for k in range(len(s)) if l[k] == -1]
    plot_gi_histogram(in_scores, out_scores, out_dir, tag)
    plot_and_save(fpr, tpr, fnr, thresholds, eer, rank1, out_dir, tag)

    with open(os.path.join(out_dir, f"rst_{tag}.txt"), "w") as f:
        f.write(f"Pairwise EER  : {eer*100:.6f}%\n")
        f.write(f"Pairwise EER½ : {eer_half*100:.6f}%\n")
        f.write(f"Threshold     : {thresh:.4f}\n")
        f.write(f"AUC           : {roc_auc:.10f}\n")
        f.write(f"Rank-1        : {rank1:.3f}%\n")

    # Aggregated EER
    n_gallery_classes = len(set(gal_labels.tolist()))
    aggr_eer = eer
    if n_gallery_classes < n_gallery:
        print("  Computing aggregated EER …")
        aggr_s, aggr_l = compute_aggregated_eer(dist_matrix, prb_labels, gal_labels)
        (aggr_eer, aggr_thresh, aggr_auc, aggr_eer_half, *_) = compute_eer(aggr_s, aggr_l)
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
    arc_s          = CONFIG["arcface_s"]
    arc_m          = CONFIG["arcface_m"]
    comp_weight    = CONFIG["comp_weight"]
    ce_weight      = CONFIG["ce_weight"]
    con_weight     = CONFIG["con_weight"]
    temperature    = CONFIG["temperature"]
    emb_dim        = CONFIG["embedding_dim"]
    train_ratio    = CONFIG["train_ratio"]
    gallery_ratio  = CONFIG["gallery_ratio"]
    val_ratio      = CONFIG["val_ratio"]
    seed           = CONFIG["random_seed"]
    save_every     = CONFIG["save_every"]
    eval_every     = CONFIG["eval_every"]
    nw             = CONFIG["num_workers"]

    assert protocol in ("closed-set", "open-set")

    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Protocol : {protocol}")
    print(f"  Device   : {device}")
    print(f"  Data     : {data_root}")
    print(f"  Loss     : {ce_weight}*CE + {con_weight}*SupCon(τ={temperature})")
    print(f"  Comp wt  : {comp_weight}")
    print(f"{'='*60}\n")

    # ---------- parse dataset ----------
    print("Scanning dataset …")
    id2paths = parse_casia_ms(data_root)
    n_total_ids  = len(id2paths)
    n_total_imgs = sum(len(v) for v in id2paths.values())
    print(f"  Found {n_total_ids} identities, {n_total_imgs} images total.\n")

    # ---------- protocol-specific split ----------
    if protocol == "closed-set":
        train_samples, test_samples, label_map = split_closed_set(
            id2paths, train_ratio=train_ratio, seed=seed)
        num_classes = len(label_map)

        # Paired datasets for training/val
        train_dataset = CASIAMSDataset(train_samples, img_side=img_side, train=True)
        val_dataset   = CASIAMSDataset(test_samples,  img_side=img_side, train=False)

        # Single-image datasets for evaluation
        gallery_eval  = CASIAMSDatasetSingle(train_samples, img_side=img_side)
        probe_eval    = CASIAMSDatasetSingle(test_samples,  img_side=img_side)

        train_loader   = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=True)
        val_loader     = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        gallery_loader = DataLoader(gallery_eval,  batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        probe_loader   = DataLoader(probe_eval,    batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

        print(f"  [closed-set] #classes={num_classes}\n")

    else:  # open-set
        (train_samples, val_samples, gallery_samples, probe_samples,
         train_label_map, test_label_map) = split_open_set(
            id2paths, train_ratio=train_ratio,
            gallery_ratio=gallery_ratio, val_ratio=val_ratio, seed=seed)
        num_classes = len(train_label_map)

        # Paired datasets for training/val
        train_dataset = CASIAMSDataset(train_samples, img_side=img_side, train=True)
        val_dataset   = CASIAMSDataset(val_samples,   img_side=img_side, train=False)

        # Single-image datasets for evaluation
        gallery_eval  = CASIAMSDatasetSingle(gallery_samples, img_side=img_side)
        probe_eval    = CASIAMSDatasetSingle(probe_samples,   img_side=img_side)

        train_loader   = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=True)
        val_loader     = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        gallery_loader = DataLoader(gallery_eval,  batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        probe_loader   = DataLoader(probe_eval,    batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

        print(f"  [open-set] #train_classes={num_classes}\n")

    # ---------- model ----------
    print(f"Building CCNet — num_classes={num_classes}, comp_weight={comp_weight} …")
    net = ccnet(num_classes=num_classes, weight=comp_weight)
    net.to(device)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
        net = DataParallel(net)

    criterion     = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=temperature, base_temperature=temperature)
    optimizer     = optim.Adam(net.parameters(), lr=lr)
    scheduler     = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    # ---------- training loop ----------
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0
    best_eer     = 1.0

    last_eer   = float("nan")
    last_rank1 = float("nan")

    print(f"\nStarting training for {num_epochs} epochs …")
    print(f"  EER / Rank-1 computed every {eval_every} epochs.\n")

    for epoch in range(num_epochs):
        t_loss, t_acc = run_one_epoch(
            epoch, net, train_loader, criterion, con_criterion,
            optimizer, device, "training",
            ce_weight=ce_weight, con_weight=con_weight)
        v_loss, v_acc = run_one_epoch(
            epoch, net, val_loader, criterion, con_criterion,
            optimizer, device, "testing",
            ce_weight=ce_weight, con_weight=con_weight)
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        _net = net.module if isinstance(net, DataParallel) else net

        # ── periodic evaluation ───────────────────────────────────────────────
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            tag = f"ep{epoch:04d}_{protocol.replace('-','')}"
            cur_eer, cur_aggr_eer, cur_rank1 = evaluate(
                _net, probe_loader, gallery_loader,
                device, out_dir=rst_eval, tag=tag)
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
                f"EER={eer_str}  Rank-1={rank1_str}")

        # ── save best classification model ────────────────────────────────────
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(_net.state_dict(),
                       os.path.join(results_dir, "net_params_best.pth"))

        # ── periodic checkpoint ───────────────────────────────────────────────
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            torch.save(_net.state_dict(),
                       os.path.join(results_dir, "net_params.pth"))
            plot_loss_acc(train_losses, val_losses, train_accs, val_accs, results_dir)

    # ---------- final evaluation ----------
    print("\n=== Final evaluation with best EER model ===")
    best_model_path = os.path.join(results_dir, "net_params_best_eer.pth")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(results_dir, "net_params_best.pth")

    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_model_path, map_location=device))

    final_eer, final_aggr_eer, final_rank1 = evaluate(
        eval_net, probe_loader, gallery_loader,
        device, out_dir=rst_eval,
        tag=f"FINAL_{protocol.replace('-','')}")

    print(f"\n{'='*60}")
    print(f"  PROTOCOL : {protocol}")
    print(f"  FINAL Pairwise EER   : {final_eer*100:.4f}%")
    print(f"  FINAL Aggregated EER : {final_aggr_eer*100:.4f}%")
    print(f"  FINAL Rank-1         : {final_rank1:.3f}%")
    print(f"  Results saved to: {results_dir}")
    print(f"{'='*60}\n")

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
