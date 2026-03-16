"""
JPFA — Joint Pixel and Feature Alignment
Paper: Shao & Zhong, IEEE TIP 2021

Adapted for CASIA-MS cross-spectral evaluation.

Filename format : {id}_{hand}_{spectrum}_{iter}.jpg
                  e.g. 005_r_460_01.jpg
Spectra         : 460, 630, 700, 850, 940, WHT
Iterations      : 01-06 (6 images per identity per spectrum)

PIPELINE
--------
Phase 1 — CycleGAN (pixel-level alignment, Section III-B)
  Source images   → G_S2T → fake target images  (same identity)
  Target images   → G_T2S → fake source images
  Losses          : GAN + cycle-consistency + identity (Eq. 4-6)
  Identity loss   : Euclidean dist of DHN features (Eq. 5)

Phase 2 — JPFA feature-level training (Section III-C, train.py)
  Input           : source, fake (from Phase 1), unlabeled target
  Backbone        : VGG16 pretrained (torchvision), pool4 frozen
  Two extractors  : FS (source branch), FF (fake branch)
  Losses          : DHN + MK-MMD + quantization + consistency (Eq. 11)
  Total loss      : mmd_t_s + mmd_t_f + src_dhn + fake_dhn
                    + Q_loss + 1.5 * consistency

Evaluation (Section IV-B)
  Gallery = source dataset (all images, D1 ∪ D2)
  Probe   = target dataset
  Metric  = Hamming distance of binarised codes → EER + Rank-1
  Best of FS, FF, avg reported (paper protocol)

DATA SPLIT for CASIA-MS
  6 images per identity per spectrum (no explicit sessions)
  D1 = iter 01-03 (first half)
  D2 = iter 04-06 (second half)
  Equal 1:1 split as stated in paper Section 4.2
"""

import os, math, time, copy, itertools
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
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS  — edit these lines only
# ============================================================

# --- Paths ---
DATA_PATH        = "/home/pai-ng/Jamal/CASIA-MS-ROI"
OUTPUT_DIR       = "./results_jpfa_casia"
GPU_ID           = "0"

# --- Cross-spectral experiment ---
# Paper uses 460, 700, 850 for CasiaM (Section 4.1)
# Table 3: set SOURCE_SPECTRUM to one, TARGET_SPECTRA = the other two
SOURCE_SPECTRUM  = "460"
TARGET_SPECTRA   = ["700", "850"]

# --- CycleGAN Phase 1 ---
CYC_EPOCHS       = 200       # epochs for CycleGAN training
CYC_LR           = 0.0002    # Adam lr (standard CycleGAN)
CYC_LAMBDA_CYC   = 10.0      # cycle-consistency weight
CYC_LAMBDA_ID    = 1.0       # identity loss weight (Eq. 5)
CYC_BATCH        = 4         # batch size (GAN is memory-heavy)
CYC_DECAY_EPOCH  = 100       # start LR decay at this epoch
N_RESIDUALS      = 9         # ResNet blocks in generator (224px → 9)
GAN_IMG_SIZE     = 128       # resize for CycleGAN (save memory)

# --- JPFA Phase 2 (from train.py) ---
HASH_DIM         = 128       # hashing code length
FC_DIMS          = [4096, 4096, 2048, 128]
MARGIN           = 180       # hashing loss margin m
ALPHA_Q          = 0.5       # quantization loss weight α
MMD_BANDWIDTHS   = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
BETA_CONSIS      = 1.5       # consistency loss weight β
BATCH_SIZE       = 30        # from train.py
OMEGA_SIZE       = 20        # anchor samples in hashing loss
JPFA_STEPS       = 50000     # training steps (from train.py)
JPFA_LR          = 0.0001    # RMSProp lr (from train.py)
JPFA_LR_DECAY    = 0.96      # decay factor every 100 steps
VGG_IMG_SIZE     = 224       # VGG16 input size

# --- Logging ---
PRINT_INTERVAL   = 500
EVAL_INTERVAL    = 2000

# ============================================================
# (nothing to edit below)
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "cyclegan"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "fake_images"), exist_ok=True)
print(f"Device: {device}")


# ============================================================
# 1.  DATA HELPERS
# ============================================================

def parse_filename(fname):
    """Parse {id}_{hand}_{spectrum}_{iter}.jpg"""
    stem  = os.path.splitext(fname)[0]
    parts = stem.split("_")
    if len(parts) < 4 or not parts[0].isdigit():
        return None
    return dict(id=parts[0], hand=parts[1],
                spectrum=parts[2], iteration=parts[3])


def build_lists(data_root, source_spectrum, target_spectra):
    """
    D1 = iter 01-03 of source spectrum (first half)
    D2 = iter 04-06 of source spectrum (second half)
    D0 = D1 ∪ D2 = all source spectrum images
    Dt = all images of target spectra (unlabeled at train time)

    Returns: d0, d1, d2, dt lists and num_classes
    """
    exts  = {".jpg", ".jpeg", ".png"}
    files = sorted(f for f in os.listdir(data_root)
                   if os.path.splitext(f)[1].lower() in exts)

    src_groups = defaultdict(list)   # key=(id_hand), value=[(iter,path)]
    tgt_groups = defaultdict(list)   # key=(id_hand,spec)

    for f in files:
        m = parse_filename(f)
        if m is None:
            continue
        key_id = f"{m['id']}_{m['hand']}"
        path   = os.path.join(data_root, f)
        if m['spectrum'] == source_spectrum:
            src_groups[key_id].append((m['iteration'], path))
        elif m['spectrum'] in target_spectra:
            tgt_groups[(key_id, m['spectrum'])].append(
                (m['iteration'], path))

    sorted_ids  = sorted(src_groups.keys())
    label_map   = {k: i for i, k in enumerate(sorted_ids)}
    num_classes = len(sorted_ids)

    d0_list, d1_list, d2_list = [], [], []
    for key_id in sorted_ids:
        lbl   = label_map[key_id]
        items = sorted(src_groups[key_id], key=lambda x: x[0])
        n     = len(items); half = n // 2
        for _, p in items:
            d0_list.append((p, lbl))
        for _, p in items[:half]:    # D1: first half (iter 01-03)
            d1_list.append((p, lbl))
        for _, p in items[half:]:    # D2: second half (iter 04-06)
            d2_list.append((p, lbl))

    tgt_label_map = {}
    dt_list = []
    for (key_id, spec), items in tgt_groups.items():
        if key_id not in tgt_label_map:
            tgt_label_map[key_id] = len(tgt_label_map)
        lbl = tgt_label_map[key_id]
        for _, p in sorted(items, key=lambda x: x[0]):
            dt_list.append((p, lbl))

    print(f"\n  Source spectrum  : {source_spectrum}")
    print(f"  Target spectra   : {target_spectra}")
    print(f"  D0 (all source)  : {len(d0_list)} images  "
          f"({num_classes} identities)")
    print(f"  D1 (iter 01-03)  : {len(d1_list)} images")
    print(f"  D2 (iter 04-06)  : {len(d2_list)} images")
    print(f"  Dt (target)      : {len(dt_list)} images  "
          f"({len(tgt_label_map)} identities)")

    if len(d0_list) == 0:
        found = set()
        for f in files[:300]:
            m2 = parse_filename(f)
            if m2: found.add(m2['spectrum'])
        print(f"  Available spectra: {sorted(found)}")

    return d0_list, d1_list, d2_list, dt_list, num_classes


def write_txt(lst, path):
    with open(path, "w") as f:
        for img_path, label in lst:
            f.write(f"{img_path} {label}\n")


# ============================================================
# 2.  DATASETS
# ============================================================

class PalmDatasetCycleGAN(Dataset):
    """Single-domain dataset for CycleGAN (unpaired).
    Returns normalised 3-channel image at GAN_IMG_SIZE.
    """
    def __init__(self, samples):
        self.samples = samples
        self.tf = transforms.Compose([
            transforms.Resize(GAN_IMG_SIZE),
            transforms.CenterCrop(GAN_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.5]*3, [0.5]*3),   # [-1, 1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.tf(Image.open(path).convert("L"))
        return img, label, path   # path kept for saving fake images


class PalmDatasetJPFA(Dataset):
    """Dataset for JPFA feature training.
    Returns VGG16-normalised 3-channel image at VGG_IMG_SIZE.
    """
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.labels  = [s[1] for s in samples]
        base = [
            transforms.Resize(VGG_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.485]*3, [0.229]*3),
        ]
        if augment:
            base = [
                transforms.Resize(VGG_IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=10, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize([0.485]*3, [0.229]*3),
            ]
        self.tf = transforms.Compose(base)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.tf(Image.open(path).convert("L"))
        return img, label


class FakeDatasetJPFA(Dataset):
    """Loads pre-generated fake images saved by CycleGAN phase."""
    def __init__(self, fake_dir, source_samples):
        """
        fake_dir      : directory where fake images are saved
        source_samples: original source list to map labels
        """
        self.samples = []
        for path, label in source_samples:
            fname    = os.path.basename(path)
            fake_path = os.path.join(fake_dir, fname)
            if os.path.exists(fake_path):
                self.samples.append((fake_path, label))
            else:
                # Fallback to original if fake not generated
                self.samples.append((path, label))
        self.labels = [s[1] for s in self.samples]
        self.tf = transforms.Compose([
            transforms.Resize(VGG_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.485]*3, [0.229]*3),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.tf(Image.open(path).convert("L"))
        return img, label


# ============================================================
# 3.  CYCLEGAN ARCHITECTURE  (Section III-B)
#     Standard ResNet generator + PatchGAN discriminator
#     matching the architecture used in train.py comments
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class CycleGenerator(nn.Module):
    """ResNet-based generator (n_residuals=9 for 224px input)."""
    def __init__(self, in_ch=3, out_ch=3, n_res=N_RESIDUALS, ngf=64):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),
        ]
        # Downsampling
        for mult in [1, 2]:
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2,
                          3, 2, 1),
                nn.InstanceNorm2d(ngf * mult * 2), nn.ReLU(True),
            ]
        # Residual blocks
        for _ in range(n_res):
            layers.append(ResidualBlock(ngf * 4))
        # Upsampling
        for mult in [2, 1]:
            layers += [
                nn.ConvTranspose2d(ngf * mult * 2, ngf * mult,
                                   3, 2, 1, output_padding=1),
                nn.InstanceNorm2d(ngf * mult), nn.ReLU(True),
            ]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CycleDiscriminator(nn.Module):
    """PatchGAN discriminator."""
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        def block(ic, oc, stride=2, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, stride, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        self.model = nn.Sequential(
            *block(in_ch, ndf,    norm=False),
            *block(ndf,   ndf*2),
            *block(ndf*2, ndf*4),
            *block(ndf*4, ndf*8, stride=1),
            nn.Conv2d(ndf*8, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.model(x)


class ImageBuffer:
    """50-image buffer to stabilise GAN training."""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data     = []

    def push_and_pop(self, images):
        out = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(img)
                out.append(img)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(self.max_size)
                    out.append(self.data[idx].clone())
                    self.data[idx] = img
                else:
                    out.append(img)
        return torch.cat(out, dim=0)


# ============================================================
# 4.  CYCLEGAN TRAINING  (Phase 1)
# ============================================================

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)


def linear_decay(epoch, n_epochs, decay_start):
    """LR multiplier: 1.0 until decay_start, then linear decay to 0."""
    if epoch < decay_start:
        return 1.0
    return max(0., 1. - (epoch - decay_start) / (n_epochs - decay_start))


def train_cyclegan(src_list, tgt_list, fake_dir):
    """
    Phase 1: CycleGAN training (Eq. 4-6).

    G_S2T : source → target-style fake images
    G_T2S : target → source-style fake images
    D_S   : discriminator for source domain
    D_T   : discriminator for target domain

    Identity loss uses a lightweight feature extractor
    (pretrained VGG16 pool1 features) to match Euclidean
    distance between source and fake features (Eq. 5).

    Saves fake source-spectrum images to fake_dir.
    These are used as the fake dataset in Phase 2.
    """
    print("\n" + "="*60)
    print("  PHASE 1 — CycleGAN pixel-level alignment")
    print(f"  Source: {SOURCE_SPECTRUM}  →  Target: {TARGET_SPECTRA}")
    print(f"  Epochs: {CYC_EPOCHS}  |  Batch: {CYC_BATCH}")
    print("="*60 + "\n")

    # ── Datasets ─────────────────────────────────────────────
    src_ds = PalmDatasetCycleGAN(src_list)
    tgt_ds = PalmDatasetCycleGAN(tgt_list)
    src_loader = DataLoader(src_ds, batch_size=CYC_BATCH,
                            shuffle=True,  drop_last=True,
                            num_workers=2, pin_memory=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=CYC_BATCH,
                            shuffle=True,  drop_last=True,
                            num_workers=2, pin_memory=True)

    # ── Models ───────────────────────────────────────────────
    G_S2T = CycleGenerator().to(device)
    G_T2S = CycleGenerator().to(device)
    D_S   = CycleDiscriminator().to(device)
    D_T   = CycleDiscriminator().to(device)

    G_S2T.apply(weights_init)
    G_T2S.apply(weights_init)
    D_S.apply(weights_init)
    D_T.apply(weights_init)

    # Identity loss feature extractor:
    # VGG16 pool1 features (light, identity-preserving)
    vgg_id = models.vgg16(
        weights=models.VGG16_Weights.IMAGENET1K_V1)
    id_feat = nn.Sequential(
        *list(vgg_id.features.children())[:5]).to(device)
    for p in id_feat.parameters():
        p.requires_grad = False
    id_feat.eval()

    # ── Optimisers ───────────────────────────────────────────
    opt_G = optim.Adam(
        itertools.chain(G_S2T.parameters(), G_T2S.parameters()),
        lr=CYC_LR, betas=(0.5, 0.999))
    opt_D_S = optim.Adam(D_S.parameters(),
                         lr=CYC_LR, betas=(0.5, 0.999))
    opt_D_T = optim.Adam(D_T.parameters(),
                         lr=CYC_LR, betas=(0.5, 0.999))

    sched_G   = optim.lr_scheduler.LambdaLR(
        opt_G,   lr_lambda=lambda e: linear_decay(
            e, CYC_EPOCHS, CYC_DECAY_EPOCH))
    sched_D_S = optim.lr_scheduler.LambdaLR(
        opt_D_S, lr_lambda=lambda e: linear_decay(
            e, CYC_EPOCHS, CYC_DECAY_EPOCH))
    sched_D_T = optim.lr_scheduler.LambdaLR(
        opt_D_T, lr_lambda=lambda e: linear_decay(
            e, CYC_EPOCHS, CYC_DECAY_EPOCH))

    crit_GAN  = nn.MSELoss()   # LSGAN (more stable than BCE)
    crit_cyc  = nn.L1Loss()
    crit_id   = nn.MSELoss()   # Euclidean distance for id loss (Eq. 5)

    buf_S = ImageBuffer(); buf_T = ImageBuffer()

    cyc_log = os.path.join(OUTPUT_DIR, "cyclegan", "log.csv")
    with open(cyc_log, "w") as f:
        f.write("epoch,loss_G,loss_D_S,loss_D_T\n")

    for epoch in range(1, CYC_EPOCHS + 1):
        G_S2T.train(); G_T2S.train()
        D_S.train();   D_T.train()
        epoch_G = epoch_DS = epoch_DT = 0.; n_batches = 0

        tgt_iter = iter(tgt_loader)
        for src_imgs, src_lbls, _ in src_loader:
            try:
                tgt_imgs, _, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_imgs, _, _ = next(tgt_iter)

            real_S = src_imgs.to(device)
            real_T = tgt_imgs.to(device)
            bs     = real_S.size(0)

            # Discriminator patch targets
            real_lbl = torch.ones (bs, 1,
                *D_T(real_T).shape[2:], device=device)
            fake_lbl = torch.zeros(bs, 1,
                *D_T(real_T).shape[2:], device=device)

            # ── Generator update ─────────────────────────────
            opt_G.zero_grad()

            fake_T  = G_S2T(real_S)   # source → fake target
            fake_S  = G_T2S(real_T)   # target → fake source
            rec_S   = G_T2S(fake_T)   # cycle: fake target → source
            rec_T   = G_S2T(fake_S)   # cycle: fake source → target

            # GAN losses
            l_gan_S2T = crit_GAN(D_T(fake_T), real_lbl)
            l_gan_T2S = crit_GAN(D_S(fake_S), real_lbl)

            # Cycle-consistency losses
            l_cyc_S = crit_cyc(rec_S, real_S) * CYC_LAMBDA_CYC
            l_cyc_T = crit_cyc(rec_T, real_T) * CYC_LAMBDA_CYC

            # Identity loss (Eq. 5) using VGG features
            # Resize to 224 for VGG feature extraction
            rs = F.interpolate(real_S, 224, mode='bilinear',
                               align_corners=False)
            ft = F.interpolate(fake_T, 224, mode='bilinear',
                               align_corners=False)
            # Normalise from [-1,1] to ImageNet space
            mean = torch.tensor([0.485, 0.456, 0.406],
                                 device=device).view(1,3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225],
                                 device=device).view(1,3,1,1)
            rs_n = (rs * 0.5 + 0.5 - mean) / std
            ft_n = (ft * 0.5 + 0.5 - mean) / std
            f_rs = id_feat(rs_n).flatten(1)
            f_ft = id_feat(ft_n).flatten(1)
            l_id = crit_id(f_rs, f_ft) * CYC_LAMBDA_ID

            loss_G = (l_gan_S2T + l_gan_T2S
                      + l_cyc_S + l_cyc_T + l_id)
            loss_G.backward()
            opt_G.step()

            # ── Discriminator D_T update ──────────────────────
            opt_D_T.zero_grad()
            fake_T_buf = buf_T.push_and_pop(fake_T.detach())
            l_DT = (crit_GAN(D_T(real_T), real_lbl)
                    + crit_GAN(D_T(fake_T_buf), fake_lbl)) * 0.5
            l_DT.backward()
            opt_D_T.step()

            # ── Discriminator D_S update ──────────────────────
            opt_D_S.zero_grad()
            fake_S_buf = buf_S.push_and_pop(fake_S.detach())
            l_DS = (crit_GAN(D_S(real_S), real_lbl)
                    + crit_GAN(D_S(fake_S_buf), fake_lbl)) * 0.5
            l_DS.backward()
            opt_D_S.step()

            epoch_G  += loss_G.item()
            epoch_DS += l_DS.item()
            epoch_DT += l_DT.item()
            n_batches += 1

        sched_G.step(); sched_D_S.step(); sched_D_T.step()

        avg_G  = epoch_G  / n_batches
        avg_DS = epoch_DS / n_batches
        avg_DT = epoch_DT / n_batches

        if epoch % 10 == 0 or epoch == 1:
            print(f"  CycleGAN Epoch {epoch:3d}/{CYC_EPOCHS}  "
                  f"G={avg_G:.4f}  D_S={avg_DS:.4f}  "
                  f"D_T={avg_DT:.4f}")

        with open(cyc_log, "a") as f:
            f.write(f"{epoch},{avg_G:.5f},"
                    f"{avg_DS:.5f},{avg_DT:.5f}\n")

    # ── Save generators ──────────────────────────────────────
    torch.save(G_S2T.state_dict(),
               os.path.join(OUTPUT_DIR, "cyclegan", "G_S2T.pth"))
    torch.save(G_T2S.state_dict(),
               os.path.join(OUTPUT_DIR, "cyclegan", "G_T2S.pth"))

    # ── Generate and save fake images ────────────────────────
    print(f"\n  Generating fake images → {fake_dir}")
    G_S2T.eval()
    gen_tf = transforms.Compose([
        transforms.Resize(GAN_IMG_SIZE),
        transforms.CenterCrop(GAN_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    to_pil = transforms.ToPILImage()
    n_saved = 0
    with torch.no_grad():
        for path, _ in src_list:
            img_t = gen_tf(
                Image.open(path).convert("L")).unsqueeze(0).to(device)
            fake_t = G_S2T(img_t).squeeze(0).cpu()
            # Denormalise from [-1,1] to [0,1]
            fake_t = (fake_t * 0.5 + 0.5).clamp(0., 1.)
            out_path = os.path.join(fake_dir,
                                    os.path.basename(path))
            to_pil(fake_t).convert("RGB").save(out_path)
            n_saved += 1
    print(f"  Saved {n_saved} fake images to {fake_dir}")
    return G_S2T


# ============================================================
# 5.  VGG16 FEATURE EXTRACTOR  (from VGG16_net.py)
#     Common F (VGG Batch1-4, frozen) +
#     domain-specific head (Batch5 + custom FC)
# ============================================================

class FeatureExtractor(nn.Module):
    """
    Matches VGG16_net.py exactly:
      Map()  : VGG pool4 → 3×Conv(512,512,3) + BN + ReLU → MaxPool
      FC()   : flatten → 4096 → 4096 → 2048 → 128 (tanh)

    Frozen: VGG features[:24] (Batch1-4 up to pool4)
    Trained: features[24:] + extra_conv + fc
    """
    def __init__(self):
        super().__init__()
        # Load pretrained VGG16 as specified
        vgg16 = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Frozen backbone: Batch1-4 (features[0:24])
        self.backbone = nn.Sequential(
            *list(vgg16.features.children())[:24])
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Trainable Batch5 (features[24:])
        self.batch5 = nn.Sequential(
            *list(vgg16.features.children())[24:])

        # Extra conv layers from Map() in VGG16_net.py
        # 3 × Conv(512→512, 3×3, padding=1) + BN + ReLU
        # followed by MaxPool
        self.extra_conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        # FC layers from FC() in VGG16_net.py
        # For 224×224 input:
        #   pool4 output: 14×14×512
        #   after batch5 + extra_conv: 7×7×512 = 25088
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, FC_DIMS[0]),
            nn.BatchNorm1d(FC_DIMS[0]), nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(FC_DIMS[0], FC_DIMS[1]),
            nn.BatchNorm1d(FC_DIMS[1]), nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(FC_DIMS[1], FC_DIMS[2]),
            nn.BatchNorm1d(FC_DIMS[2]), nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(FC_DIMS[2], FC_DIMS[3]),
            nn.Tanh(),   # tanh → codes in [-1,1]
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)    # frozen Batch1-4
        x = self.batch5(x)          # trainable Batch5
        x = self.extra_conv(x)      # extra conv + pool
        x = x.flatten(1)            # [B, 25088]
        return self.fc(x)           # [B, 128] tanh codes


# ============================================================
# 6.  JPFA LOSSES  (from H_loss.py, M_loss.py, train.py)
# ============================================================

def hashing_loss(feature, label_onehot):
    """
    DHN hashing + quantization loss (H_loss.py, Eq. 1-3).
    feature     : [B, 128]  tanh codes
    label_onehot: [B, C]    one-hot labels
    Uses OMEGA_SIZE as split point between anchor / positive set.
    """
    omega = OMEGA_SIZE
    a_code = feature[:omega]; s_code = feature[omega:]
    a_lbl  = label_onehot[:omega]; s_lbl = label_onehot[omega:]

    def euclid(A, B):
        a2 = (A**2).sum(1, keepdim=True)
        b2 = (B**2).sum(1, keepdim=True)
        return (a2 + b2.T - 2 * A @ B.T).clamp(min=0.)

    aa_dist = euclid(a_code, a_code)   # [omega × omega]
    as_dist = euclid(a_code, s_code)   # [omega × (B-omega)]
    aa_sim  = (a_lbl @ a_lbl.T > 0).float()
    as_sim  = (a_lbl @ s_lbl.T  > 0).float()

    l_aa = (0.5 * aa_sim * aa_dist
            + 0.5 * (1-aa_sim) * F.relu(MARGIN - aa_dist)).mean()
    l_as = (0.5 * as_sim * as_dist
            + 0.5 * (1-as_sim) * F.relu(MARGIN - as_dist)).mean()

    # Quantization loss (Eq. 2)
    q = ((feature.abs() - 1.) ** 2).mean()

    return l_aa + l_as + ALPHA_Q * q


def mkmmd_loss(X, Y):
    """MK-MMD loss (M_loss.py, Eq. 7-9)."""
    XX = X @ X.T; XY = X @ Y.T; YY = Y @ Y.T
    x2 = XX.diag(); y2 = YY.diag()
    K_XX = K_XY = K_YY = 0.
    for sigma in MMD_BANDWIDTHS:
        g = 1. / (2 * sigma**2)
        K_XX = K_XX + torch.exp(-g * (
            x2.unsqueeze(1) + x2.unsqueeze(0) - 2*XX))
        K_XY = K_XY + torch.exp(-g * (
            x2.unsqueeze(1) + y2.unsqueeze(0) - 2*XY))
        K_YY = K_YY + torch.exp(-g * (
            y2.unsqueeze(1) + y2.unsqueeze(0) - 2*YY))
    n = float(X.size(0)); m = float(Y.size(0))
    mmd2 = (K_XX.sum()/(n*n) + K_YY.sum()/(m*m)
            - 2*K_XY.sum()/(n*m))
    return torch.sqrt(mmd2.clamp(min=1e-8))


def consistency_loss(code_s, code_f):
    """
    Consistency loss (Eq. 10, train.py: distance_loss).
    Mean absolute difference between binarised codes.
    """
    return (code_s.sign() - code_f.sign()).abs().mean()


# ============================================================
# 7.  EVALUATION  (Section IV-B, Hamming distance protocol)
# ============================================================

def compute_eer(ins, outs):
    if ins.mean() < outs.mean():
        ins, outs = -ins, -outs
    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    sc  = np.concatenate([ins, outs])
    fpr, tpr, _ = metrics.roc_curve(y, sc, pos_label=1)
    roc_auc     = auc(fpr, tpr)
    eer = brentq(
        lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100., roc_auc


def extract_codes(model, loader):
    model.eval()
    codes, ids = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            c = model(imgs.to(device)).sign()   # binarise
            codes.append(c.cpu().numpy())
            ids.append(labels.numpy())
    return np.concatenate(codes), np.concatenate(ids)


def hamming_sim(A, B):
    """
    Hamming similarity: higher = more similar.
    sim = (D - hamming_dist) / D
        = A @ B.T / D   (for codes in {-1,+1})
    """
    D = A.shape[1]
    return (A @ B.T) / D   # in [-1, 1]


def evaluate(fs_model, ff_model,
             src_loader, tgt_loader,
             tag, out_dir):
    """
    Gallery = source (D0 = D1 ∪ D2).  Probe = target Dt.
    Two extractors FS and FF evaluated; best result reported.
    """
    print(f"\n{'='*60}")
    print(f"  Evaluation : {tag}")
    print(f"  Gallery=D0 (source)  |  Probe=Dt (target)")
    print(f"{'='*60}")

    gc_s, g_id = extract_codes(fs_model, src_loader)
    gc_f, _    = extract_codes(ff_model, src_loader)
    pc_s, p_id = extract_codes(fs_model, tgt_loader)
    pc_f, _    = extract_codes(ff_model, tgt_loader)

    best_eer   = 100.; best_rank1 = 0.
    best_auc   = 0.;   best_lbl   = ""

    for gc, pc, lbl in [
        (gc_s, pc_s, "FS"),
        (gc_f, pc_f, "FF"),
        ((gc_s+gc_f)/2., (pc_s+pc_f)/2., "avg"),
    ]:
        sim   = hamming_sim(pc, gc)               # [M, N]
        preds = g_id[sim.argmax(axis=1)]
        rank1 = 100. * (preds == p_id).mean()

        n_g = len(g_id)
        s   = sim.ravel()
        l   = np.where(
            np.tile(g_id, len(p_id)) == np.repeat(p_id, n_g),
            1, -1)
        ins  = s[l ==  1]
        outs = s[l == -1]
        eer, roc_auc = compute_eer(ins, outs)

        print(f"  {lbl:4s}  Rank-1={rank1:.3f}%  "
              f"EER={eer:.4f}%  AUC={roc_auc:.6f}")
        if eer < best_eer:
            best_eer   = eer; best_rank1 = rank1
            best_auc   = roc_auc; best_lbl = lbl

    print(f"  Best ({best_lbl}): Rank-1={best_rank1:.3f}%  "
          f"EER={best_eer:.4f}%")
    print(f"{'='*60}\n")

    ev_dir = os.path.join(out_dir, tag)
    os.makedirs(ev_dir, exist_ok=True)
    with open(os.path.join(ev_dir, "results.txt"), "w") as f:
        f.write(f"EER    : {best_eer:.4f}%\n")
        f.write(f"Rank-1 : {best_rank1:.3f}%\n")
        f.write(f"AUC    : {best_auc:.6f}\n")
    return best_eer, best_rank1


# ============================================================
# 8.  JPFA TRAINING  (Phase 2, train.py)
# ============================================================

def make_onehot(labels, num_classes):
    oh = torch.zeros(len(labels), num_classes, device=device)
    oh.scatter_(1, labels.to(device).view(-1,1).long(), 1.)
    return oh


def train_jpfa(src_list, fake_dir, tgt_list, num_classes):
    """
    Phase 2: JPFA feature-level training (Section III-C).

    Exactly mirrors train.py:
      source + target → FS → source_feature, target_source_feature
      fake   + target → FF → fake_feature,   target_fake_feature

      source_loss  = DHN(source_feature, source_label)
      fake_loss    = DHN(fake_feature,   fake_label)
      Q_loss       = quantization on target codes
      mmd_t_s      = MK-MMD(target_source_feature, source_feature)
      mmd_t_f      = MK-MMD(target_fake_feature,   fake_feature)
      dist_loss    = consistency(target_source, target_fake)

      total = mmd_t_s + mmd_t_f + source_loss + fake_loss
              + Q_loss + 1.5 * dist_loss
    """
    print("\n" + "="*60)
    print("  PHASE 2 — JPFA feature-level training")
    print(f"  Steps: {JPFA_STEPS}  |  Batch: {BATCH_SIZE}")
    print("="*60 + "\n")

    # ── Datasets ─────────────────────────────────────────────
    src_ds  = PalmDatasetJPFA(src_list,  augment=True)
    fake_ds = FakeDatasetJPFA(fake_dir,  src_list)
    tgt_ds  = PalmDatasetJPFA(tgt_list,  augment=False)

    def inf_loader(ds, bs):
        while True:
            for batch in DataLoader(ds, batch_size=bs,
                                    shuffle=True, drop_last=True,
                                    num_workers=2, pin_memory=True):
                yield batch

    src_iter  = inf_loader(src_ds,  BATCH_SIZE)
    fake_iter = inf_loader(fake_ds, BATCH_SIZE)
    tgt_iter  = inf_loader(tgt_ds,  BATCH_SIZE)

    # Evaluation loaders (full D0 as gallery)
    eval_src = DataLoader(
        PalmDatasetJPFA(src_list,  augment=False),
        batch_size=64, shuffle=False, num_workers=2)
    eval_tgt = DataLoader(
        PalmDatasetJPFA(tgt_list,  augment=False),
        batch_size=64, shuffle=False, num_workers=2)

    # ── Models ───────────────────────────────────────────────
    FS = FeatureExtractor().to(device)
    FF = FeatureExtractor().to(device)
    best_FS = copy.deepcopy(FS)
    best_FF = copy.deepcopy(FF)

    # Trainable parameters only (backbone frozen)
    trainable = (
        [p for p in FS.parameters() if p.requires_grad] +
        [p for p in FF.parameters() if p.requires_grad])

    # RMSProp with alpha=0.9 (from train.py)
    optimizer = optim.RMSprop(trainable, lr=JPFA_LR, alpha=0.9)
    # Exponential decay: every 100 steps × 0.96 (from train.py)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=JPFA_LR_DECAY)

    print(f"  Trainable params: "
          f"{sum(p.numel() for p in trainable):,}")

    # ── Training log ─────────────────────────────────────────
    header = (f"{'Step':>6}  {'Loss':>8}  {'SrcDHN':>8}  "
              f"{'FkDHN':>8}  {'MMD_s':>7}  {'MMD_f':>7}  "
              f"{'Consis':>7}  {'Rank1':>7}  {'EER':>7}  Time")
    sep = "-" * len(header)
    print(header); print(sep)

    log_path = os.path.join(OUTPUT_DIR, "jpfa_log.csv")
    with open(log_path, "w") as f:
        f.write("step,loss,src_dhn,fake_dhn,"
                "mmd_s,mmd_f,consis,rank1,eer\n")

    best_eer   = 100.
    loss_hist  = []
    eer_hist   = []
    rank1_hist = []
    eval_steps = []
    t0 = time.time()

    for step in range(1, JPFA_STEPS + 1):
        FS.train(); FF.train()

        # Sample batches
        src_imgs, src_lbls = next(src_iter)
        fk_imgs,  fk_lbls  = next(fake_iter)
        tgt_imgs, _        = next(tgt_iter)   # target unlabeled

        src_imgs = src_imgs.to(device)
        fk_imgs  = fk_imgs.to(device)
        tgt_imgs = tgt_imgs.to(device)

        src_oh = make_onehot(src_lbls, num_classes)
        fk_oh  = make_onehot(fk_lbls,  num_classes)

        optimizer.zero_grad()

        # ── Forward passes (matching train.py) ───────────────
        # FS: processes source + target
        src_feat   = FS(src_imgs)    # source feature
        tgt_feat_s = FS(tgt_imgs)    # target feature via FS

        # FF: processes fake + target
        fake_feat  = FF(fk_imgs)     # fake feature
        tgt_feat_f = FF(tgt_imgs)    # target feature via FF

        # ── Losses (train.py) ─────────────────────────────────
        src_dhn  = hashing_loss(src_feat,  src_oh)
        fake_dhn = hashing_loss(fake_feat, fk_oh)

        # Quantization on target codes (Eq. 2 applied to target)
        ts_sign = tgt_feat_s.sign(); tf_sign = tgt_feat_f.sign()
        q_loss  = 0.5 * (
            ((ts_sign - tgt_feat_s)**2).mean() +
            ((tf_sign - tgt_feat_f)**2).mean())

        mmd_s    = mkmmd_loss(tgt_feat_s, src_feat)
        mmd_f    = mkmmd_loss(tgt_feat_f, fake_feat)
        consis   = consistency_loss(tgt_feat_s, tgt_feat_f)

        # Total loss (train.py, line 44)
        loss = (mmd_s + mmd_f
                + src_dhn + fake_dhn + q_loss
                + BETA_CONSIS * consis)

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_hist.append(loss.item())

        if step % PRINT_INTERVAL == 0:
            elapsed = time.time() - t0
            print(f"{step:>6}  {loss.item():>8.4f}  "
                  f"{src_dhn.item():>8.4f}  "
                  f"{fake_dhn.item():>8.4f}  "
                  f"{mmd_s.item():>7.4f}  "
                  f"{mmd_f.item():>7.4f}  "
                  f"{consis.item():>7.4f}  "
                  f"{'--':>7}  {'--':>7}  {elapsed:.0f}s")
            t0 = time.time()

        if step % EVAL_INTERVAL == 0:
            eer, rank1 = evaluate(
                FS, FF, eval_src, eval_tgt,
                f"step_{step:05d}", OUTPUT_DIR)
            eer_hist.append(eer)
            rank1_hist.append(rank1)
            eval_steps.append(step)

            with open(log_path, "a") as f:
                f.write(f"{step},{loss.item():.5f},"
                        f"{src_dhn.item():.5f},"
                        f"{fake_dhn.item():.5f},"
                        f"{mmd_s.item():.5f},"
                        f"{mmd_f.item():.5f},"
                        f"{consis.item():.5f},"
                        f"{rank1:.4f},{eer:.4f}\n")

            if eer < best_eer:
                best_eer = eer
                torch.save(FS.state_dict(),
                           os.path.join(OUTPUT_DIR, "best_FS.pth"))
                torch.save(FF.state_dict(),
                           os.path.join(OUTPUT_DIR, "best_FF.pth"))
                best_FS.load_state_dict(copy.deepcopy(FS.state_dict()))
                best_FF.load_state_dict(copy.deepcopy(FF.state_dict()))
                print(f"  >>> Best EER: {best_eer:.4f}%  "
                      f"Rank-1: {rank1:.3f}%  — saved")

    # ── Final evaluation ─────────────────────────────────────
    print(f"\n{sep}")
    print(f"JPFA complete.  Best EER: {best_eer:.4f}%")
    evaluate(FS,      FF,      eval_src, eval_tgt,
             "final_last", OUTPUT_DIR)
    evaluate(best_FS, best_FF, eval_src, eval_tgt,
             "final_best", OUTPUT_DIR)

    # ── Curves ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(loss_hist)
    axes[0].set_title("Total loss"); axes[0].set_xlabel("Step")
    axes[1].plot(eval_steps, eer_hist, color="red")
    axes[1].set_title("Target EER %"); axes[1].set_xlabel("Step")
    axes[2].plot(eval_steps, rank1_hist, color="green")
    axes[2].set_title("Target Rank-1 %"); axes[2].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "jpfa_curves.png"), dpi=120)
    plt.close()

    torch.save(FS.state_dict(),
               os.path.join(OUTPUT_DIR, "last_FS.pth"))
    torch.save(FF.state_dict(),
               os.path.join(OUTPUT_DIR, "last_FF.pth"))
    return best_FS, best_FF


# ============================================================
# 9.  MAIN
# ============================================================

def main():
    print(f"\n{'='*60}")
    print(f"  JPFA  |  Source: {SOURCE_SPECTRUM}  "
          f"→  Target: {TARGET_SPECTRA}")
    print(f"{'='*60}")

    # ── 1. Build data lists ───────────────────────────────────
    print("\n[1] Building data lists ...")
    d0_list, d1_list, d2_list, dt_list, num_classes = build_lists(
        DATA_PATH, SOURCE_SPECTRUM, TARGET_SPECTRA)

    if len(d0_list) == 0:
        return

    write_txt(d0_list, os.path.join(OUTPUT_DIR, "D0_source.txt"))
    write_txt(d1_list, os.path.join(OUTPUT_DIR, "D1_first_half.txt"))
    write_txt(d2_list, os.path.join(OUTPUT_DIR, "D2_second_half.txt"))
    write_txt(dt_list, os.path.join(OUTPUT_DIR, "Dt_target.txt"))

    fake_dir = os.path.join(OUTPUT_DIR, "fake_images")

    # ── 2. Phase 1: CycleGAN ─────────────────────────────────
    cyc_done_flag = os.path.join(
        OUTPUT_DIR, "cyclegan", "G_S2T.pth")
    if os.path.exists(cyc_done_flag):
        print("\n[2] CycleGAN weights found — skipping training.")
        print(f"    Using existing fake images in {fake_dir}")
        print(f"    Delete {cyc_done_flag} to retrain CycleGAN.")
    else:
        print("\n[2] Starting CycleGAN training ...")
        train_cyclegan(d0_list, dt_list, fake_dir)

    # ── 3. Phase 2: JPFA ─────────────────────────────────────
    print("\n[3] Starting JPFA feature-level training ...")
    best_FS, best_FF = train_jpfa(
        d0_list, fake_dir, dt_list, num_classes)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
