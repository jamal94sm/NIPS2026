"""
JPFA — Joint Pixel and Feature Alignment
Paper: Shao & Zhong, IEEE TIP 2021

Adapted for CASIA-MS cross-spectral evaluation.

Filename format : {id}_{hand}_{spectrum}_{iter}.jpg
                  e.g. 005_r_460_01.jpg
Spectra         : 460, 630, 700, 850, 940, WHT
Iterations      : 01-06  →  D1 = iter 01-03, D2 = iter 04-06

PIXEL ALIGNMENT TOGGLE
----------------------
PIXEL_METHOD = "cyclegan"  — Phase 1 trains CycleGAN (Eq. 4-6).
PIXEL_METHOD = "palmrss"   — FAT + HM on-the-fly (faster).

FIXES APPLIED
-------------
[F1] BNNoScale: gamma=1 frozen, matches VGG16_net.py (scale=None in TF).

[F2] Epoch display: every print shows step and equivalent epoch.

[F3] Full print table: TrAcc, TrEER, TgtR1, TgtEER every PRINT_INTERVAL.

[F4] CRITICAL — Wrong features in hashing loss:
     H_loss in the repo operates on the 2048-dim intermediate ReLU
     features (FC3/fc_feat), NOT the 128-dim tanh codes.
     For 128-dim codes, max Euclidean dist ≈ 22.6.
     MARGIN=180 >> 22.6 means all impostor pairs are always at max
     penalty, genuine pairs get almost zero gradient — no learning.
     Fix: FeatureExtractor returns (fc_feat [2048], code [128]).
          hashing_loss uses fc_feat; MMD/consistency use code.

[F5] CRITICAL — LR decay was catastrophically aggressive:
     StepLR(step_size=100, gamma=0.96) → LR decays to ~1e-8 by
     step 20000, essentially killing training.
     Fix: step_size=5000 (decay every ~125 epochs, not every 2.5).
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
SOURCE_SPECTRUM  = "460"
TARGET_SPECTRA   = ["700", "850"]

# ──────────────────────────────────────────────────────────────
# PIXEL ALIGNMENT TOGGLE
#   "cyclegan" : CycleGAN (paper, Eq. 4-6)
#   "palmrss"  : FAT + HM on-the-fly
# ──────────────────────────────────────────────────────────────
PIXEL_METHOD     = "palmrss"

# --- CycleGAN settings ---
CYC_EPOCHS       = 200
CYC_LR           = 0.0002
CYC_LAMBDA_CYC   = 10.0
CYC_LAMBDA_ID    = 1.0
CYC_BATCH        = 4
CYC_DECAY_EPOCH  = 100
N_RESIDUALS      = 9
GAN_IMG_SIZE     = 128

# --- PalmRSS settings ---
BETA_FAT         = 0.1

# --- JPFA Phase 2 ---
HASH_DIM         = 128
FC_DIMS          = [4096, 4096, 2048, 128]   # FC1→FC2→FC3(feat)→FC4(code)
MARGIN           = 180      # margin for hashing loss on 2048-dim fc_feat [F4]
ALPHA_Q          = 0.5
MMD_BANDWIDTHS   = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
BETA_CONSIS      = 1.5
BATCH_SIZE       = 30
OMEGA_SIZE       = 20
JPFA_STEPS       = 50000
JPFA_LR          = 0.0001
# [F5] Fixed: was step_size=100 (killed LR by step 10000)
# Now step_size=5000 → one decay every ~125 epochs, gentle and correct
JPFA_LR_STEP     = 5000
JPFA_LR_GAMMA    = 0.96
VGG_IMG_SIZE     = 224

# --- Logging ---
PRINT_INTERVAL   = 500
EVAL_INTERVAL    = 2000

# ============================================================
# (nothing to edit below)
# ============================================================

assert PIXEL_METHOD in ("cyclegan", "palmrss"), \
    "PIXEL_METHOD must be 'cyclegan' or 'palmrss'"

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "cyclegan"),    exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "fake_images"), exist_ok=True)
print(f"Device       : {device}")
print(f"Pixel method : {PIXEL_METHOD.upper()}")


# ============================================================
# 1.  DATA HELPERS
# ============================================================

def parse_filename(fname):
    stem  = os.path.splitext(fname)[0]
    parts = stem.split("_")
    if len(parts) < 4 or not parts[0].isdigit():
        return None
    return dict(id=parts[0], hand=parts[1],
                spectrum=parts[2], iteration=parts[3])


def build_lists(data_root, source_spectrum, target_spectra):
    exts  = {".jpg", ".jpeg", ".png"}
    files = sorted(f for f in os.listdir(data_root)
                   if os.path.splitext(f)[1].lower() in exts)

    src_groups = defaultdict(list)
    tgt_groups = defaultdict(list)

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
        half  = len(items) // 2
        for _, p in items:
            d0_list.append((p, lbl))
        for _, p in items[:half]:
            d1_list.append((p, lbl))
        for _, p in items[half:]:
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
            if m2:
                found.add(m2['spectrum'])
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
    def __init__(self, samples):
        self.samples = samples
        self.tf = transforms.Compose([
            transforms.Resize(GAN_IMG_SIZE),
            transforms.CenterCrop(GAN_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.tf(Image.open(path).convert("L")), label, path


class PalmDatasetJPFA(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.labels  = [s[1] for s in samples]
        if augment:
            self.tf = transforms.Compose([
                transforms.Resize(VGG_IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=10, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize([0.485]*3, [0.229]*3),
            ])
        else:
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
        return self.tf(Image.open(path).convert("L")), label


class FakeDatasetCycleGAN(Dataset):
    def __init__(self, fake_dir, source_samples):
        self.samples = []
        n_missing = 0
        for path, label in source_samples:
            fname     = os.path.basename(path)
            fake_path = os.path.join(fake_dir, fname)
            if os.path.exists(fake_path):
                self.samples.append((fake_path, label))
            else:
                self.samples.append((path, label))
                n_missing += 1
        if n_missing:
            print(f"  [FakeDataset] {n_missing} missing, using originals.")
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
        return self.tf(Image.open(path).convert("L")), label


# ============================================================
# 3A.  CYCLEGAN  (PIXEL_METHOD = "cyclegan")
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class CycleGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, n_res=N_RESIDUALS, ngf=64):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7),
                  nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        for mult in [1, 2]:
            layers += [nn.Conv2d(ngf*mult, ngf*mult*2, 3, 2, 1),
                       nn.InstanceNorm2d(ngf*mult*2), nn.ReLU(True)]
        for _ in range(n_res):
            layers.append(ResidualBlock(ngf*4))
        for mult in [2, 1]:
            layers += [nn.ConvTranspose2d(ngf*mult*2, ngf*mult,
                                          3, 2, 1, output_padding=1),
                       nn.InstanceNorm2d(ngf*mult), nn.ReLU(True)]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7),
                   nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CycleDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        def blk(ic, oc, stride=2, norm=True):
            L = [nn.Conv2d(ic, oc, 4, stride, 1)]
            if norm: L.append(nn.InstanceNorm2d(oc))
            L.append(nn.LeakyReLU(0.2, True)); return L
        self.model = nn.Sequential(
            *blk(in_ch, ndf, norm=False), *blk(ndf, ndf*2),
            *blk(ndf*2, ndf*4), *blk(ndf*4, ndf*8, stride=1),
            nn.Conv2d(ndf*8, 1, 4, 1, 1))

    def forward(self, x):
        return self.model(x)


class ImageBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size; self.data = []

    def push_and_pop(self, images):
        out = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(img); out.append(img)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(self.max_size)
                    out.append(self.data[idx].clone())
                    self.data[idx] = img
                else:
                    out.append(img)
        return torch.cat(out, dim=0)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)


def linear_decay(epoch, n_epochs, decay_start):
    if epoch < decay_start: return 1.0
    return max(0., 1.-(epoch-decay_start)/(n_epochs-decay_start))


def run_cyclegan(src_list, tgt_list, fake_dir):
    print("\n"+"="*65)
    print("  PHASE 1 — CycleGAN (Eq. 4-6, not in GitHub repo)")
    print(f"  Epochs: {CYC_EPOCHS}  |  Batch: {CYC_BATCH}")
    print("="*65+"\n")

    src_ds = PalmDatasetCycleGAN(src_list)
    tgt_ds = PalmDatasetCycleGAN(tgt_list)
    src_loader = DataLoader(src_ds, batch_size=CYC_BATCH, shuffle=True,
                            drop_last=True, num_workers=2, pin_memory=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=CYC_BATCH, shuffle=True,
                            drop_last=True, num_workers=2, pin_memory=True)

    G_S2T = CycleGenerator().to(device)
    G_T2S = CycleGenerator().to(device)
    D_S   = CycleDiscriminator().to(device)
    D_T   = CycleDiscriminator().to(device)
    for net in [G_S2T, G_T2S, D_S, D_T]:
        net.apply(weights_init)

    vgg_tmp = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    id_feat = nn.Sequential(*list(vgg_tmp.features.children())[:5]).to(device)
    for p in id_feat.parameters(): p.requires_grad = False
    id_feat.eval()

    opt_G  = optim.Adam(itertools.chain(G_S2T.parameters(),
                                        G_T2S.parameters()),
                        lr=CYC_LR, betas=(0.5, 0.999))
    opt_DS = optim.Adam(D_S.parameters(), lr=CYC_LR, betas=(0.5, 0.999))
    opt_DT = optim.Adam(D_T.parameters(), lr=CYC_LR, betas=(0.5, 0.999))
    sch_G  = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lambda e:
        linear_decay(e, CYC_EPOCHS, CYC_DECAY_EPOCH))
    sch_DS = optim.lr_scheduler.LambdaLR(opt_DS, lr_lambda=lambda e:
        linear_decay(e, CYC_EPOCHS, CYC_DECAY_EPOCH))
    sch_DT = optim.lr_scheduler.LambdaLR(opt_DT, lr_lambda=lambda e:
        linear_decay(e, CYC_EPOCHS, CYC_DECAY_EPOCH))

    crit_GAN = nn.MSELoss(); crit_cyc = nn.L1Loss(); crit_id = nn.MSELoss()
    buf_S = ImageBuffer(); buf_T = ImageBuffer()
    mean_ = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
    std_  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)

    cyc_log = os.path.join(OUTPUT_DIR, "cyclegan", "log.csv")
    with open(cyc_log, "w") as f: f.write("epoch,loss_G,loss_DS,loss_DT\n")

    for epoch in range(1, CYC_EPOCHS+1):
        G_S2T.train(); G_T2S.train(); D_S.train(); D_T.train()
        eg = eds = edt = 0.; nb = 0
        tgt_it = iter(tgt_loader)
        for src_imgs, _, _ in src_loader:
            try: tgt_imgs, _, _ = next(tgt_it)
            except StopIteration:
                tgt_it = iter(tgt_loader); tgt_imgs, _, _ = next(tgt_it)
            real_S = src_imgs.to(device); real_T = tgt_imgs.to(device)
            bs = real_S.size(0); patch = D_T(real_T).shape[2:]
            rl = torch.ones(bs, 1, *patch, device=device)
            fl = torch.zeros(bs, 1, *patch, device=device)

            opt_G.zero_grad()
            fake_T = G_S2T(real_S); fake_S = G_T2S(real_T)
            rec_S  = G_T2S(fake_T); rec_T  = G_S2T(fake_S)
            l_gan  = crit_GAN(D_T(fake_T), rl) + crit_GAN(D_S(fake_S), rl)
            l_cyc  = (crit_cyc(rec_S,real_S)+crit_cyc(rec_T,real_T))*CYC_LAMBDA_CYC
            rs_n = (F.interpolate(real_S,224,mode='bilinear',align_corners=False)*0.5+0.5-mean_)/std_
            ft_n = (F.interpolate(fake_T,224,mode='bilinear',align_corners=False)*0.5+0.5-mean_)/std_
            l_id = crit_id(id_feat(rs_n).flatten(1), id_feat(ft_n).flatten(1))*CYC_LAMBDA_ID
            (l_gan+l_cyc+l_id).backward(); opt_G.step()

            opt_DT.zero_grad()
            ft_buf = buf_T.push_and_pop(fake_T.detach())
            loss_DT = (crit_GAN(D_T(real_T),rl)+crit_GAN(D_T(ft_buf),fl))*0.5
            loss_DT.backward(); opt_DT.step()

            opt_DS.zero_grad()
            fs_buf = buf_S.push_and_pop(fake_S.detach())
            loss_DS = (crit_GAN(D_S(real_S),rl)+crit_GAN(D_S(fs_buf),fl))*0.5
            loss_DS.backward(); opt_DS.step()
            eg += (l_gan+l_cyc+l_id).item(); eds += loss_DS.item()
            edt += loss_DT.item(); nb += 1

        sch_G.step(); sch_DS.step(); sch_DT.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{CYC_EPOCHS}  "
                  f"G={eg/nb:.4f}  DS={eds/nb:.4f}  DT={edt/nb:.4f}")
        with open(cyc_log, "a") as f:
            f.write(f"{epoch},{eg/nb:.5f},{eds/nb:.5f},{edt/nb:.5f}\n")

    torch.save(G_S2T.state_dict(), os.path.join(OUTPUT_DIR,"cyclegan","G_S2T.pth"))
    torch.save(G_T2S.state_dict(), os.path.join(OUTPUT_DIR,"cyclegan","G_T2S.pth"))

    print(f"\n  Generating fake images → {fake_dir}")
    G_S2T.eval()
    gen_tf = transforms.Compose([
        transforms.Resize(GAN_IMG_SIZE), transforms.CenterCrop(GAN_IMG_SIZE),
        transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize([0.5]*3, [0.5]*3)])
    to_pil = transforms.ToPILImage()
    with torch.no_grad():
        for path, _ in src_list:
            img_t = gen_tf(Image.open(path).convert("L")).unsqueeze(0).to(device)
            fake_t = (G_S2T(img_t).squeeze(0).cpu()*0.5+0.5).clamp(0.,1.)
            to_pil(fake_t).convert("RGB").save(
                os.path.join(fake_dir, os.path.basename(path)))
    print(f"  Saved {len(src_list)} fake images.")


# ============================================================
# 3B.  PALMRSS ALIGNMENT
# ============================================================

def _hist_match_np(src, tgt):
    matched = np.empty_like(src)
    for c in range(src.shape[2]):
        s = src[...,c].ravel().astype(np.float64)
        t = tgt[...,c].ravel().astype(np.float64)
        s_min,s_max = s.min(),s.max(); t_min,t_max = t.min(),t.max()
        if s_max==s_min or t_max==t_min: matched[...,c]=src[...,c]; continue
        s_n=(s-s_min)/(s_max-s_min); t_n=(t-t_min)/(t_max-t_min)
        bins=256
        s_cnt,_=np.histogram(s_n,bins=bins,range=(0.,1.))
        t_cnt,_=np.histogram(t_n,bins=bins,range=(0.,1.))
        s_cdf=np.cumsum(s_cnt).astype(np.float64); s_cdf/=s_cdf[-1]
        t_cdf=np.cumsum(t_cnt).astype(np.float64); t_cdf/=t_cdf[-1]
        edges=np.linspace(0.,1.,bins+1); centers=(edges[:-1]+edges[1:])/2.
        t_idx=np.searchsorted(t_cdf,s_cdf).clip(0,bins-1)
        lut=centers[t_idx]*(s_max-s_min)+s_min
        pb=np.searchsorted(edges[1:],s_n).clip(0,bins-1)
        matched[...,c]=lut[pb].reshape(src.shape[:2]).astype(np.float32)
    return matched.astype(np.float32)


def hm_batch(src_batch, tgt_batch):
    rows = []
    for s,t in zip(src_batch,tgt_batch):
        rows.append(torch.from_numpy(
            _hist_match_np(s.permute(1,2,0).numpy(),
                           t.permute(1,2,0).numpy())).permute(2,0,1))
    return torch.stack(rows).float()


def fat_batch(src, tgt, beta=BETA_FAT):
    fs=torch.fft.rfft2(src,dim=(-2,-1)); ft=torch.fft.rfft2(tgt,dim=(-2,-1))
    as_=torch.abs(fs).clone(); ps=torch.angle(fs); at=torch.abs(ft)
    _,_,h,w2=as_.shape
    b=min(int(np.floor(beta*h)),int(np.floor(beta*w2*2)))
    if b>0:
        as_[:,:,:b,:b]=at[:,:,:b,:b]; as_[:,:,h-b+1:h,:b]=at[:,:,h-b+1:h,:b]
    rec=torch.fft.irfft2(torch.complex(torch.cos(ps)*as_,torch.sin(ps)*as_),
                         dim=(-2,-1),s=[h,w2*2])
    return rec[...,:src.shape[-2],:src.shape[-1]]


def make_fake_palmrss(src_cpu, tgt_cpu):
    return (fat_batch(src_cpu,tgt_cpu)+hm_batch(src_cpu,tgt_cpu))/2.


# ============================================================
# 4.  VGG16 FEATURE EXTRACTOR  (VGG16_net.py)
#
# [F1] BNNoScale: gamma=1 frozen → matches TF scale=None
# [F4] Returns (fc_feat [2048-dim ReLU], code [128-dim tanh])
#      fc_feat → used for hashing loss   (margin=180 valid here)
#      code    → used for MMD, consistency, evaluation
# ============================================================

class BNNoScale(nn.Module):
    """BatchNorm with gamma=1 frozen. Matches VGG16_net.py scale=None."""
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=True)
        nn.init.ones_(self.bn.weight)
        self.bn.weight.requires_grad = False

    def forward(self, x):
        return self.bn(x)


class BNNoScale1d(nn.Module):
    """Same as BNNoScale but for FC layers."""
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=True)
        nn.init.ones_(self.bn.weight)
        self.bn.weight.requires_grad = False

    def forward(self, x):
        return self.bn(x)


class FeatureExtractor(nn.Module):
    """
    Matches VGG16_net.py architecture exactly.

    Map():
      VGG pool4 (frozen, 14×14×512)
      → 3×Conv(512,512,3,pad=1)+BNNoScale+ReLU
      → MaxPool → 7×7×512

    FC():
      flatten(25088)
      → FC1(4096) + BNNoScale1d + ReLU + Dropout
      → FC2(4096) + BNNoScale1d + ReLU + Dropout
      → FC3(2048) + BNNoScale1d + ReLU + Dropout  ← fc_feat [F4]
      → FC4(128)  + Tanh                           ← code    [F4]

    forward() returns (fc_feat, code):
      fc_feat : 2048-dim ReLU → passed to hashing_loss (margin=180 valid)
      code    : 128-dim tanh  → passed to MMD, consistency, evaluation
    """
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Frozen VGG Batch1-4 (up to pool4, 14×14×512)
        self.backbone = nn.Sequential(*list(vgg16.features.children())[:24])
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Map(): 3 convs + maxpool (14×14 → 7×7)
        self.extra_conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), BNNoScale(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), BNNoScale(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), BNNoScale(512), nn.ReLU(True),
            nn.MaxPool2d(2, 2),   # 14×14 → 7×7 → flatten=25088
        )

        # FC1 + FC2 + FC3 (shared, returns fc_feat)
        self.fc123 = nn.Sequential(
            nn.Linear(512*7*7, FC_DIMS[0]),
            BNNoScale1d(FC_DIMS[0]), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(FC_DIMS[0], FC_DIMS[1]),
            BNNoScale1d(FC_DIMS[1]), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(FC_DIMS[1], FC_DIMS[2]),     # FC3: 2048-dim
            BNNoScale1d(FC_DIMS[2]), nn.ReLU(True), nn.Dropout(0.5),
        )

        # FC4: code layer (128-dim tanh)
        self.fc4 = nn.Sequential(
            nn.Linear(FC_DIMS[2], FC_DIMS[3]),     # FC4: 128-dim
            nn.Tanh(),
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)          # frozen → 14×14×512
        x        = self.extra_conv(x)     # → 7×7×512
        fc_feat  = self.fc123(x.flatten(1))  # → 2048-dim ReLU [F4]
        code     = self.fc4(fc_feat)         # → 128-dim tanh  [F4]
        return fc_feat, code


# ============================================================
# 5.  JPFA LOSSES
# ============================================================

def hashing_loss(fc_feat, label_onehot):
    """
    DHN hashing + quantization loss (H_loss.py, Eq. 1-3).

    [F4] Uses fc_feat (2048-dim ReLU), NOT tanh codes.
    For 2048-dim ReLU features, typical Euclidean distances are
    large enough for MARGIN=180 to be meaningful.
    """
    a_feat = fc_feat[:OMEGA_SIZE]; s_feat = fc_feat[OMEGA_SIZE:]
    a_lbl  = label_onehot[:OMEGA_SIZE]; s_lbl = label_onehot[OMEGA_SIZE:]

    def euclid(A, B):
        a2 = (A**2).sum(1, keepdim=True); b2 = (B**2).sum(1, keepdim=True)
        return (a2 + b2.T - 2*A@B.T).clamp(min=0.)

    aa_d = euclid(a_feat, a_feat); as_d = euclid(a_feat, s_feat)
    aa_s = (a_lbl@a_lbl.T > 0).float(); as_s = (a_lbl@s_lbl.T > 0).float()

    l_hash = (
        (0.5*aa_s*aa_d + 0.5*(1-aa_s)*F.relu(MARGIN-aa_d)).mean()
        + (0.5*as_s*as_d + 0.5*(1-as_s)*F.relu(MARGIN-as_d)).mean()
    )
    # Quantization applied to codes (sign operation during eval)
    # Note: codes are in fc4 output (128-dim tanh), passed separately
    return l_hash


def quantization_loss(code):
    """Q-loss (Eq. 2): push tanh codes toward ±1."""
    return ALPHA_Q * ((code.abs() - 1.)**2).mean()


def mkmmd_loss(X, Y):
    """MK-MMD on 128-dim codes (M_loss.py, Eq. 7-9)."""
    XX=X@X.T; XY=X@Y.T; YY=Y@Y.T
    x2=XX.diag(); y2=YY.diag()
    K_XX=K_XY=K_YY=0.
    for sigma in MMD_BANDWIDTHS:
        g=1./(2*sigma**2)
        K_XX+=torch.exp(-g*(x2.unsqueeze(1)+x2.unsqueeze(0)-2*XX))
        K_XY+=torch.exp(-g*(x2.unsqueeze(1)+y2.unsqueeze(0)-2*XY))
        K_YY+=torch.exp(-g*(y2.unsqueeze(1)+y2.unsqueeze(0)-2*YY))
    n=float(X.size(0)); m=float(Y.size(0))
    mmd2=K_XX.sum()/(n*n)+K_YY.sum()/(m*m)-2*K_XY.sum()/(n*m)
    return torch.sqrt(mmd2.clamp(min=1e-8))


def consistency_loss(code_s, code_f):
    """Consistency on 128-dim codes (train.py distance_loss, Eq. 10)."""
    return (code_s.sign() - code_f.sign()).abs().mean()


# ============================================================
# 6.  EVALUATION  (Hamming distance on binarised 128-dim codes)
# ============================================================

def compute_eer(ins, outs):
    if ins.mean() < outs.mean(): ins, outs = -ins, -outs
    y  = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    sc = np.concatenate([ins, outs])
    fpr,tpr,_ = metrics.roc_curve(y, sc, pos_label=1)
    roc_auc   = auc(fpr, tpr)
    eer = brentq(lambda x: 1.-x-interp1d(fpr,tpr)(x), 0., 1.)
    return eer*100., roc_auc


def extract_codes(model, loader):
    """Extract binarised 128-dim codes for evaluation."""
    model.eval()
    codes, ids = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            _, code = model(imgs.to(device))   # use code not fc_feat [F4]
            codes.append(code.sign().cpu().numpy())
            ids.append(labels.numpy())
    return np.concatenate(codes), np.concatenate(ids)


def hamming_sim(A, B):
    return (A@B.T) / A.shape[1]


def compute_metrics(gc, g_id, pc, p_id):
    sim   = hamming_sim(pc, gc)
    preds = g_id[sim.argmax(axis=1)]
    rank1 = 100.*(preds==p_id).mean()
    n_g   = len(g_id); s = sim.ravel()
    l = np.where(np.tile(g_id,len(p_id))==np.repeat(p_id,n_g), 1, -1)
    eer, roc_auc = compute_eer(s[l==1], s[l==-1])
    return rank1, eer, roc_auc


def quick_eval(fs_model, ff_model, src_loader, tgt_loader):
    gc_s,g_id = extract_codes(fs_model, src_loader)
    gc_f,_    = extract_codes(ff_model, src_loader)
    pc_s,p_id = extract_codes(fs_model, tgt_loader)
    pc_f,_    = extract_codes(ff_model, tgt_loader)
    best_eer=100.; best_rank1=0.; best_lbl=""
    for gc,pc,lbl in [(gc_s,pc_s,"FS"),(gc_f,pc_f,"FF"),
                      ((gc_s+gc_f)/2.,(pc_s+pc_f)/2.,"avg")]:
        rank1,eer,_ = compute_metrics(gc,g_id,pc,p_id)
        if eer<best_eer: best_eer=eer; best_rank1=rank1; best_lbl=lbl
    return best_rank1, best_eer, best_lbl


def full_evaluate(fs_model, ff_model, src_loader, tgt_loader, tag, out_dir):
    print(f"\n{'='*65}\n  Evaluation : {tag}  [{PIXEL_METHOD.upper()}]")
    print(f"  Gallery=D0 (source)  |  Probe=Dt (target)\n{'='*65}")
    gc_s,g_id = extract_codes(fs_model,src_loader)
    gc_f,_    = extract_codes(ff_model,src_loader)
    pc_s,p_id = extract_codes(fs_model,tgt_loader)
    pc_f,_    = extract_codes(ff_model,tgt_loader)
    best_eer=100.; best_rank1=0.; best_lbl=""
    for gc,pc,lbl in [(gc_s,pc_s,"FS"),(gc_f,pc_f,"FF"),
                      ((gc_s+gc_f)/2.,(pc_s+pc_f)/2.,"avg")]:
        rank1,eer,roc_auc = compute_metrics(gc,g_id,pc,p_id)
        print(f"  {lbl:4s}  Rank-1={rank1:.3f}%  "
              f"EER={eer:.4f}%  AUC={roc_auc:.6f}")
        if eer<best_eer: best_eer=eer; best_rank1=rank1; best_lbl=lbl
    print(f"  Best ({best_lbl}): Rank-1={best_rank1:.3f}%  "
          f"EER={best_eer:.4f}%\n{'='*65}\n")
    ev_dir = os.path.join(out_dir, tag)
    os.makedirs(ev_dir, exist_ok=True)
    with open(os.path.join(ev_dir,"results.txt"),"w") as f:
        f.write(f"Method : {PIXEL_METHOD}\nEER    : {best_eer:.4f}%\n"
                f"Rank-1 : {best_rank1:.3f}%\n")
    return best_eer, best_rank1


def train_source_eval(fs_model, src_loader):
    gc, g_id = extract_codes(fs_model, src_loader)
    n   = len(g_id)
    sim = hamming_sim(gc, gc)

    # Rank-1: leave-one-out (fill diagonal with -inf so self is never top-1)
    sim_loo = sim.copy()
    np.fill_diagonal(sim_loo, -np.inf)
    preds = g_id[sim_loo.argmax(axis=1)]
    rank1 = 100.*(preds==g_id).mean()

    # EER: exclude diagonal entirely (self-matches are trivially genuine)
    # Build same/diff label matrix in 2D (n×n), then flatten with off_diag
    off_diag  = ~np.eye(n, dtype=bool)               # 2D mask (n×n)
    same_mat  = g_id[:, None] == g_id[None, :]       # 2D (n×n) bool
    s    = sim[off_diag]                              # 1D, non-self scores
    same = same_mat[off_diag]                         # 1D, genuine/impostor
    ins  = s[same]; outs = s[~same]
    eer, _ = compute_eer(ins, outs)
    return rank1, eer


# ============================================================
# 7.  JPFA TRAINING LOOP
# ============================================================

def make_onehot(labels, num_classes):
    oh = torch.zeros(len(labels), num_classes, device=device)
    oh.scatter_(1, labels.to(device).view(-1,1).long(), 1.)
    return oh


def train_jpfa(src_list, fake_dir, tgt_list, num_classes):
    steps_per_epoch = max(1, len(src_list) // BATCH_SIZE)

    print("\n"+"="*65)
    method_label = ("CycleGAN fake images" if PIXEL_METHOD=="cyclegan"
                    else "PalmRSS on-the-fly (FAT+HM)")
    print(f"  PHASE 2 — JPFA feature-level training")
    print(f"  Fake source  : {method_label}")
    print(f"  Steps        : {JPFA_STEPS}  "
          f"(≈ {JPFA_STEPS/steps_per_epoch:.0f} epochs)")
    print(f"  Batch        : {BATCH_SIZE}  "
          f"(steps/epoch ≈ {steps_per_epoch})")
    print(f"  LR decay     : every {JPFA_LR_STEP} steps × {JPFA_LR_GAMMA}  "
          f"[F5 fix: was 100 steps]")
    print(f"  Hashing loss : on fc_feat (2048-dim)  [F4 fix: was 128-dim]")
    print("="*65+"\n")

    src_ds  = PalmDatasetJPFA(src_list, augment=True)
    tgt_ds  = PalmDatasetJPFA(tgt_list, augment=False)
    fake_ds = (FakeDatasetCycleGAN(fake_dir, src_list)
               if PIXEL_METHOD=="cyclegan"
               else PalmDatasetJPFA(src_list, augment=False))

    def inf_loader(ds, bs):
        while True:
            for batch in DataLoader(ds, batch_size=bs, shuffle=True,
                                    drop_last=True, num_workers=2,
                                    pin_memory=True):
                yield batch

    src_iter  = inf_loader(src_ds,  BATCH_SIZE)
    fake_iter = inf_loader(fake_ds, BATCH_SIZE)
    tgt_iter  = inf_loader(tgt_ds,  BATCH_SIZE)

    eval_src = DataLoader(PalmDatasetJPFA(src_list, augment=False),
                          batch_size=64, shuffle=False, num_workers=2)
    eval_tgt = DataLoader(PalmDatasetJPFA(tgt_list, augment=False),
                          batch_size=64, shuffle=False, num_workers=2)

    FS = FeatureExtractor().to(device)
    FF = FeatureExtractor().to(device)
    best_FS = copy.deepcopy(FS)
    best_FF = copy.deepcopy(FF)

    trainable = ([p for p in FS.parameters() if p.requires_grad] +
                 [p for p in FF.parameters() if p.requires_grad])
    optimizer = optim.RMSprop(trainable, lr=JPFA_LR, alpha=0.9)

    # [F5] Fixed: step_size=JPFA_LR_STEP (5000) not 100
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=JPFA_LR_STEP, gamma=JPFA_LR_GAMMA)

    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    # Print header
    header = (f"{'Step':>6}  {'Epoch':>6}  {'LR':>8}  {'Loss':>8}  "
              f"{'SrcDHN':>8}  {'FkDHN':>8}  "
              f"{'TrAcc':>7}  {'TrEER':>7}  "
              f"{'TgtR1':>7}  {'TgtEER':>7}  "
              f"{'Time':>6}")
    sep = "-" * len(header)
    print(header); print(sep)

    log_path = os.path.join(OUTPUT_DIR, f"jpfa_log_{PIXEL_METHOD}.csv")
    with open(log_path, "w") as f:
        f.write("step,epoch,lr,loss,src_dhn,fake_dhn,"
                "tr_acc,tr_eer,tgt_rank1,tgt_eer\n")

    best_eer   = 100.
    loss_hist  = []
    tr_acc_hist=[]; tr_eer_hist=[]
    tgt_r1_hist=[]; tgt_eer_hist=[]
    eval_steps = []

    run_corr=0; run_total=0
    t0 = time.time()

    for step in range(1, JPFA_STEPS+1):
        FS.train(); FF.train()

        src_imgs, src_lbls = next(src_iter)
        tgt_imgs, _        = next(tgt_iter)

        if PIXEL_METHOD=="cyclegan":
            fk_imgs, fk_lbls = next(fake_iter)
            fk_imgs = fk_imgs.to(device); fk_lbls_dev = fk_lbls
        else:
            fk_raw, fk_lbls = next(fake_iter)
            fk_imgs     = make_fake_palmrss(fk_raw.cpu(), tgt_imgs.cpu()).to(device)
            fk_lbls_dev = fk_lbls

        src_imgs = src_imgs.to(device); tgt_imgs = tgt_imgs.to(device)
        src_oh   = make_onehot(src_lbls,    num_classes)
        fk_oh    = make_onehot(fk_lbls_dev, num_classes)

        optimizer.zero_grad()

        # [F4] Both extractors return (fc_feat, code)
        src_feat,  src_code  = FS(src_imgs)
        tgt_feat_s,tgt_code_s= FS(tgt_imgs)
        fake_feat, fake_code = FF(fk_imgs)
        tgt_feat_f,tgt_code_f= FF(tgt_imgs)

        # [F4] hashing_loss on fc_feat (2048-dim) — margin=180 valid
        src_dhn  = hashing_loss(src_feat,  src_oh)
        fake_dhn = hashing_loss(fake_feat, fk_oh)

        # Quantization on tanh codes
        q_loss = (quantization_loss(src_code) + quantization_loss(fake_code)
                  + quantization_loss(tgt_code_s) + quantization_loss(tgt_code_f))

        # [F4] MK-MMD on codes (128-dim tanh)
        mmd_s  = mkmmd_loss(tgt_code_s, src_code)
        mmd_f  = mkmmd_loss(tgt_code_f, fake_code)

        # [F4] Consistency on codes (128-dim tanh)
        consis = consistency_loss(tgt_code_s, tgt_code_f)

        # Total loss (train.py line 44)
        loss = (mmd_s + mmd_f
                + src_dhn + fake_dhn + q_loss
                + BETA_CONSIS * consis)

        loss.backward(); optimizer.step(); scheduler.step()
        loss_hist.append(loss.item())

        # Running train accuracy (batch nearest-neighbour on codes)
        with torch.no_grad():
            sc = src_code.sign().cpu().numpy()
            si = src_lbls.numpy()
            sm = hamming_sim(sc, sc); np.fill_diagonal(sm, -np.inf)
            run_corr  += (si[sm.argmax(axis=1)]==si).sum()
            run_total += len(si)

        if step % PRINT_INTERVAL == 0:
            epoch   = step / steps_per_epoch
            elapsed = time.time() - t0
            cur_lr  = optimizer.param_groups[0]['lr']

            tr_acc = 100.*run_corr/run_total
            run_corr=0; run_total=0

            tr_rank1, tr_eer   = train_source_eval(FS, eval_src)
            tgt_rank1, tgt_eer, _ = quick_eval(FS, FF, eval_src, eval_tgt)

            print(f"{step:>6}  {epoch:>6.1f}  {cur_lr:>8.2e}  "
                  f"{loss.item():>8.4f}  "
                  f"{src_dhn.item():>8.4f}  {fake_dhn.item():>8.4f}  "
                  f"{tr_acc:>6.2f}%  {tr_eer:>6.2f}%  "
                  f"{tgt_rank1:>6.2f}%  {tgt_eer:>6.2f}%  "
                  f"{elapsed:>5.0f}s  [{time.strftime('%H:%M:%S')}]")

            tr_acc_hist.append(tr_acc); tr_eer_hist.append(tr_eer)
            tgt_eer_hist.append(tgt_eer); tgt_r1_hist.append(tgt_rank1)
            eval_steps.append(step)

            with open(log_path, "a") as f:
                f.write(f"{step},{epoch:.2f},{cur_lr:.6e},{loss.item():.5f},"
                        f"{src_dhn.item():.5f},{fake_dhn.item():.5f},"
                        f"{tr_acc:.4f},{tr_eer:.4f},"
                        f"{tgt_rank1:.4f},{tgt_eer:.4f}\n")
            t0 = time.time()

        if step % EVAL_INTERVAL == 0:
            eer, rank1 = full_evaluate(FS, FF, eval_src, eval_tgt,
                                       f"step_{step:05d}_{PIXEL_METHOD}",
                                       OUTPUT_DIR)
            if eer < best_eer:
                best_eer = eer
                torch.save(FS.state_dict(),
                           os.path.join(OUTPUT_DIR, f"best_FS_{PIXEL_METHOD}.pth"))
                torch.save(FF.state_dict(),
                           os.path.join(OUTPUT_DIR, f"best_FF_{PIXEL_METHOD}.pth"))
                best_FS.load_state_dict(copy.deepcopy(FS.state_dict()))
                best_FF.load_state_dict(copy.deepcopy(FF.state_dict()))
                print(f"  >>> Best EER: {best_eer:.4f}%  "
                      f"Rank-1: {rank1:.3f}%  — saved\n")

    print(f"\n{sep}\nJPFA complete.  Best EER: {best_eer:.4f}%")
    full_evaluate(FS,      FF,      eval_src, eval_tgt,
                  f"final_last_{PIXEL_METHOD}", OUTPUT_DIR)
    full_evaluate(best_FS, best_FF, eval_src, eval_tgt,
                  f"final_best_{PIXEL_METHOD}", OUTPUT_DIR)

    # Curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes[0,0].plot(eval_steps, [loss_hist[s-1] for s in eval_steps])
    axes[0,0].set_title(f"Total loss [{PIXEL_METHOD}]"); axes[0,0].set_xlabel("Step")
    axes[0,1].plot(eval_steps, tr_acc_hist,  color="blue")
    axes[0,1].set_title("Train Acc % (batch)"); axes[0,1].set_xlabel("Step")
    axes[0,2].plot(eval_steps, tr_eer_hist,  color="navy")
    axes[0,2].set_title("Train EER %"); axes[0,2].set_xlabel("Step")
    axes[1,0].plot(eval_steps, tgt_r1_hist,  color="green")
    axes[1,0].set_title("Target Rank-1 %"); axes[1,0].set_xlabel("Step")
    axes[1,1].plot(eval_steps, tgt_eer_hist, color="red")
    axes[1,1].set_title("Target EER %"); axes[1,1].set_xlabel("Step")
    ax = axes[1,2]
    ax.plot(eval_steps, tr_eer_hist,  color="navy",  label="Train EER")
    ax.plot(eval_steps, tgt_eer_hist, color="red",   label="Target EER")
    ax.set_title("Train vs Target EER"); ax.set_xlabel("Step"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"curves_{PIXEL_METHOD}.png"), dpi=120)
    plt.close()

    torch.save(FS.state_dict(), os.path.join(OUTPUT_DIR, f"last_FS_{PIXEL_METHOD}.pth"))
    torch.save(FF.state_dict(), os.path.join(OUTPUT_DIR, f"last_FF_{PIXEL_METHOD}.pth"))
    return best_FS, best_FF


# ============================================================
# 8.  MAIN
# ============================================================

def main():
    print(f"\n{'='*65}")
    print(f"  JPFA  |  Source={SOURCE_SPECTRUM}  Target={TARGET_SPECTRA}")
    print(f"  Pixel : {PIXEL_METHOD.upper()}")
    print(f"{'='*65}")

    print("\n[1] Building data lists ...")
    d0, d1, d2, dt, num_classes = build_lists(
        DATA_PATH, SOURCE_SPECTRUM, TARGET_SPECTRA)
    if len(d0) == 0: return

    write_txt(d0, os.path.join(OUTPUT_DIR, "D0_source.txt"))
    write_txt(d1, os.path.join(OUTPUT_DIR, "D1_first_half.txt"))
    write_txt(d2, os.path.join(OUTPUT_DIR, "D2_second_half.txt"))
    write_txt(dt, os.path.join(OUTPUT_DIR, "Dt_target.txt"))

    fake_dir = os.path.join(OUTPUT_DIR, "fake_images")

    if PIXEL_METHOD == "cyclegan":
        cyc_w = os.path.join(OUTPUT_DIR, "cyclegan", "G_S2T.pth")
        if os.path.exists(cyc_w):
            print(f"\n[2] CycleGAN found — skipping. Delete {cyc_w} to retrain.")
        else:
            print("\n[2] Running CycleGAN ...")
            run_cyclegan(d0, dt, fake_dir)
    else:
        print("\n[2] PalmRSS — no Phase 1. Fake images on-the-fly via FAT+HM.")

    print("\n[3] Running JPFA feature-level training ...")
    train_jpfa(d0, fake_dir, dt, num_classes)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
