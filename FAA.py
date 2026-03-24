"""
FFT-BASED ADVERSARIAL AUGMENTATION FOR PALMPRINT RECOGNITION
Cross-domain evaluation on CASIA-MS-ROI (train: 460+630nm, test: 850nm+white)

Two methods compared:
  PROPOSED  — trainable FFT amplitude augmentation with adversarial training
               Option A (single): ConvVAE perturbs one image's amplitude
               Option B (pair  ): AdaIN or CNN mixer fuses two images' amplitudes
               Training: alternate aug ↔ rec updates (per batch or per epoch)

  BASELINE  — static amplitude mixup (no trainable augmentation module)
               α·A_x + (1-α)·A_shuffle, fixed α=0.5 or dynamic
               One-step training per batch (only recognition network updated)

Architecture choices:
  ConvVAE    : encoder-decoder on log-amplitude, reparameterisation gives diversity,
               KL term prevents adversarial collapse. Best for single-image Option A.
  AdaIN mixer: MLP predicts γ, β from amp1 stats, applies to normalised amp2.
               Lightweight, directly analogous to neural style transfer in freq domain.
  CNN mixer  : channel-concat of both amplitudes → conv layers → output amplitude.
               More expressive, learns non-linear mixing relations.
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # ── Dataset ───────────────────────────────────────────────────────────
    'data_path'        : '/home/pai-ng/Jamal/CASIA-MS-ROI',
    # Filename: {subject}_{hand}_{spectrum}_{iteration}.jpg

    # ── Cross-domain split ────────────────────────────────────────────────
    'train_spectra'    : ['460', '630'],   # source domain
    'test_spectra'     : ['850', 'white'], # target domain
    # cross_subject=False → all subjects in both train & test (pure cross-domain)
    # cross_subject=True  → 70% subjects train-only, 30% test-only (cross-domain + open-set)
    'cross_subject'    : True,
    'train_ratio'      : 0.7,             # used only when cross_subject=True
    'random_seed'      : 42,

    # ── Method toggle ─────────────────────────────────────────────────────
    # 'proposed' | 'baseline'
    'method'           : 'proposed',

    # ── Proposed: augmentation input mode ────────────────────────────────
    # 'single' → Option A: ConvVAE on one image's amplitude
    # 'pair'   → Option B: mixer on two images' amplitudes
    'aug_input'        : 'single',

    # ── Proposed: amplitude model (used for respective aug_input) ─────────
    # aug_input='single' → only 'conv_vae' applies
    # aug_input='pair'   → 'adain' | 'cnn'
    'amp_model'        : 'conv_vae',      # 'conv_vae' | 'adain' | 'cnn'

    # ── Proposed: adversarial alternation schedule ────────────────────────
    'alternate_per'    : 'batch',         # 'batch' | 'epoch'

    # ── Proposed: aug loss weights ────────────────────────────────────────
    'lambda_kl'        : 0.01,            # KL weight for ConvVAE (keeps latent sane)

    # ── Baseline: amplitude mixup ─────────────────────────────────────────
    'dynamic_mixup'    : False,           # False → fixed α; True → α ~ Beta(0.4, 0.4)
    'mixup_alpha'      : 0.5,             # used when dynamic_mixup=False

    # ── Common training ───────────────────────────────────────────────────
    'img_size'         : 112,
    'batch_size'       : 32,
    'num_workers'      : 4,
    'num_epochs'       : 50,
    'lr_rec'           : 0.01,            # recognition network LR
    'lr_aug'           : 1e-3,            # augmentation module LR (proposed only)
    'warmup_epochs'    : 5,
    'weight_decay'     : 5e-4,
    'grad_clip'        : 5.0,
    'aug_gamma'        : 0.5,             # fraction of batch to augment
    'eval_freq'        : 5,
    'save_freq'        : 10,

    # ── ConvVAE ───────────────────────────────────────────────────────────
    'latent_dim'       : 256,

    # ── ArcFace ───────────────────────────────────────────────────────────
    'feature_dim'      : 512,
    'arcface_s'        : 64.0,
    'arcface_m'        : 0.5,

    # ── Checkpoint paths ──────────────────────────────────────────────────
    'save_dir'         : 'checkpoints_fft',
}

# ============================================================================
# IMPORTS
# ============================================================================

import os, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_curve
import argparse
from datetime import datetime

torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])
random.seed(CONFIG['random_seed'])

args = argparse.Namespace(**CONFIG)

print("=" * 70)
print("  FFT AUGMENTATION — CROSS-DOMAIN PALMPRINT RECOGNITION")
print("=" * 70)
for k, v in CONFIG.items():
    print(f"  {k:<22} = {v}")
print("=" * 70 + "\n")


# ============================================================================
# SECTION 1: DATA LOADING WITH CROSS-DOMAIN SPLIT
# ============================================================================

def load_casia_samples(data_path):
    """
    Parse CASIA-MS-ROI directory.
    Filename: {subject}_{hand}_{spectrum}_{iteration}.jpg
    Identity key: subject + hand
    """
    samples = []
    for root, _, files in os.walk(data_path):
        for fname in sorted(files):
            if not fname.lower().endswith('.jpg'):
                continue
            parts = fname[:-4].split('_')
            if len(parts) != 4:
                continue
            subject_id, hand, spectrum, iteration = parts
            samples.append({
                'path'     : os.path.join(root, fname),
                'subject'  : subject_id,
                'hand'     : hand,
                'spectrum' : spectrum,
                'iteration': iteration,
                'hand_id'  : f'{subject_id}_{hand}',
            })
    return samples


def make_cross_domain_split(samples, cfg):
    """
    Build train / test sample lists for cross-domain evaluation.

    cross_subject=False:
        All subjects appear in both train and test.
        Train = train_spectra images, Test = test_spectra images.
        Pure cross-domain challenge: model must generalise across spectral domains.

    cross_subject=True:
        Subjects split 70/30 (train subjects vs test subjects).
        Train = train subjects × train_spectra.
        Test  = test  subjects × test_spectra.
        Combined cross-domain + open-set challenge.
    """
    train_sp = set(cfg.train_spectra)
    test_sp  = set(cfg.test_spectra)

    if not cfg.cross_subject:
        train_s = [s for s in samples if s['spectrum'] in train_sp]
        test_s  = [s for s in samples if s['spectrum'] in test_sp]
    else:
        np.random.seed(cfg.random_seed)
        all_ids  = sorted(set(s['hand_id'] for s in samples))
        np.random.shuffle(all_ids)
        n_train  = int(len(all_ids) * cfg.train_ratio)
        train_ids = set(all_ids[:n_train])
        test_ids  = set(all_ids[n_train:])
        train_s  = [s for s in samples if s['hand_id'] in train_ids
                    and s['spectrum'] in train_sp]
        test_s   = [s for s in samples if s['hand_id'] in test_ids
                    and s['spectrum'] in test_sp]

    # Build integer identity labels
    train_hand_ids = sorted(set(s['hand_id'] for s in train_s))
    test_hand_ids  = sorted(set(s['hand_id'] for s in test_s))
    train_id_map   = {h: i for i, h in enumerate(train_hand_ids)}
    test_id_map    = {h: i for i, h in enumerate(test_hand_ids)}

    for s in train_s:
        s['label'] = train_id_map[s['hand_id']]
    for s in test_s:
        s['label'] = test_id_map[s['hand_id']]

    print(f"[Data] Train: {len(set(s['hand_id'] for s in train_s))} identities, "
          f"{len(train_s)} samples  (spectra: {cfg.train_spectra})")
    print(f"[Data] Test : {len(set(s['hand_id'] for s in test_s))} identities, "
          f"{len(test_s)} samples  (spectra: {cfg.test_spectra})")
    return train_s, test_s, len(train_hand_ids)


class PalmDataset(Dataset):
    def __init__(self, samples, img_size=112):
        self.samples   = samples
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        return {
            'img'     : self.transform(img),
            'label'   : s['label'],
            'hand_id' : s['hand_id'],
            'spectrum': s['spectrum'],
        }


def create_dataloaders(cfg):
    print(f"\n[Data] Loading CASIA-MS-ROI from {cfg.data_path}")
    samples          = load_casia_samples(cfg.data_path)
    print(f"[Data] Total samples: {len(samples)}")
    print(f"[Data] Spectra found: {sorted(set(s['spectrum'] for s in samples))}")

    train_s, test_s, num_classes = make_cross_domain_split(samples, cfg)

    train_ds     = PalmDataset(train_s, cfg.img_size)
    test_ds      = PalmDataset(test_s,  cfg.img_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    return train_loader, test_loader, num_classes, test_s


# ============================================================================
# SECTION 2: FFT UTILITIES
# ============================================================================

def image_to_amp_phase(x):
    """
    x: (B, C, H, W) in [-1, 1]
    Returns:
        amplitude : (B, C, H, W) — non-negative magnitude
        phase     : (B, C, H, W) — angle in [-π, π]
        log_amp   : (B, C, H, W) — log(1 + amplitude), numerically stable
    """
    fft       = torch.fft.fft2(x, norm='ortho')
    amplitude = torch.abs(fft)
    phase     = torch.angle(fft)
    log_amp   = torch.log1p(amplitude)
    return amplitude, phase, log_amp


def amp_phase_to_image(amplitude, phase):
    """
    Reconstruct image from amplitude and phase via inverse FFT.
    amplitude: (B, C, H, W) — must be non-negative
    phase    : (B, C, H, W)
    Returns  : (B, C, H, W) in [-1, 1]
    """
    real        = amplitude * torch.cos(phase)
    imag        = amplitude * torch.sin(phase)
    fft_complex = torch.complex(real, imag)
    x           = torch.fft.ifft2(fft_complex, norm='ortho').real
    return x.clamp(-1.0, 1.0)


def log_amp_to_amp(log_amp):
    """Inverse of log1p: recover amplitude from log-amplitude."""
    return torch.expm1(log_amp.clamp(min=0.0))


# ============================================================================
# SECTION 3: AMPLITUDE AUGMENTATION MODULES
# ============================================================================

# ── Option A: ConvVAE ────────────────────────────────────────────────────────

class ConvVAE(nn.Module):
    """
    Variational autoencoder on log-amplitude.

    Encoder: (B, 3, 112, 112) → μ, logvar  (both shape: B × latent_dim)
    Decoder: (B, latent_dim) → (B, 3, 112, 112) log-amplitude

    Why ConvVAE for Option A:
      - Reparameterisation trick gives diversity: each forward pass samples a
        different amplitude perturbation → varied augmentations from one image.
      - KL term during adversarial training prevents the latent from collapsing
        to a single degenerate hard point — maintains augmentation diversity.
      - Convolutional layers capture spatial frequency patterns better than
        a flat MLP on the full 112×112 map.
    """
    def __init__(self, img_size=112, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 112 → 56 → 28 → 14 → 7
        self.encoder = nn.Sequential(
            nn.Conv2d(3,   32,  4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32,  64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2),
            nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        # After 4 stride-2 convs: 112 → 7
        self.flat_dim  = 256 * (img_size // 16) * (img_size // 16)
        self.fc_mu     = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.ConvTranspose2d(32,  3,   4, stride=2, padding=1),
            nn.Softplus(),   # amplitude must be non-negative; Softplus is smooth
        )
        self._spatial = img_size // 16

    def encode(self, log_amp):
        h      = self.encoder(log_amp).flatten(1)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, self._spatial, self._spatial)
        return self.decoder(h)   # outputs non-negative log-amplitude

    def forward(self, log_amp):
        mu, logvar = self.encode(log_amp)
        z          = self.reparameterise(mu, logvar)
        log_amp_aug = self.decode(z)
        return log_amp_aug, mu, logvar


# ── Option B — AdaIN amplitude mixer ─────────────────────────────────────────

class AdaINAmpMixer(nn.Module):
    """
    Transfers amplitude statistics of x1 onto the structure of x2.

    A small MLP predicts per-channel γ and β from the mean+std of amp1,
    then applies them to instance-normalised amp2.

    Why AdaIN for Option B:
      - Directly analogous to neural style transfer: amplitude statistics
        (mean, std) encode the global frequency profile (brightness, contrast,
        spectral energy distribution) — exactly the 'style' we want to transfer.
      - The MLP adds trainable parameters so the adversarial update can learn
        which statistical transformations produce the hardest samples.
      - Very lightweight (~few hundred parameters) → fast convergence.
    """
    def __init__(self, channels=3, hidden=64):
        super().__init__()
        # MLP: [mean(amp1), std(amp1)] per channel → γ, β for amp2
        self.style_net = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels * 2),
        )
        self.channels = channels

    def forward(self, amp1, amp2):
        """
        amp1: (B, C, H, W) — style source amplitude
        amp2: (B, C, H, W) — identity source amplitude (phase kept from amp2's image)
        Returns: mixed amplitude (B, C, H, W), non-negative
        """
        C = self.channels

        # Statistics of amp1 → predict affine params
        mean1   = amp1.mean(dim=[2, 3])                  # (B, C)
        std1    = amp1.std(dim=[2, 3]).clamp(min=1e-8)   # (B, C)
        stats   = torch.cat([mean1, std1], dim=1)        # (B, 2C)
        params  = self.style_net(stats)                  # (B, 2C)
        gamma   = params[:, :C].unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        beta    = params[:, C:].unsqueeze(-1).unsqueeze(-1)

        # Instance-normalise amp2 then re-scale with learned params from amp1
        mean2   = amp2.mean(dim=[2, 3], keepdim=True)
        std2    = amp2.std(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        normed  = (amp2 - mean2) / std2
        out     = gamma * normed + beta

        return F.softplus(out)   # enforce non-negativity smoothly


# ── Option B — CNN amplitude mixer ────────────────────────────────────────────

class CNNAmpMixer(nn.Module):
    """
    Learns to mix two amplitude maps via convolutional layers.

    Concatenates amp1 and amp2 along the channel dimension and passes
    through a shallow residual CNN to produce the output amplitude.

    Why CNN for Option B:
      - More expressive than AdaIN: learns pixel-level mixing patterns
        (e.g. keep high-frequency content from one, low-frequency from other).
      - Residual connections ensure the output stays close to valid amplitudes.
      - 6-channel input (concat) lets the network learn asymmetric mixing.
    """
    def __init__(self, channels=3, hidden=32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden),
        )
        self.out = nn.Conv2d(hidden, channels, 1)
        # Residual skip from input to output via 1×1 conv
        self.skip = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, amp1, amp2):
        """
        amp1, amp2: (B, C, H, W) — both non-negative amplitudes
        Returns   : mixed amplitude (B, C, H, W), non-negative
        """
        x    = torch.cat([amp1, amp2], dim=1)    # (B, 2C, H, W)
        skip = self.skip(x)
        h    = self.head(x)
        h    = h + self.res1(h)
        h    = h + self.res2(h)
        out  = self.out(h) + skip
        return F.softplus(out)   # non-negative


# ============================================================================
# SECTION 4: FFT AUGMENTATION MODULE  (wraps amplitude models)
# ============================================================================

class FFTAugModule(nn.Module):
    """
    Unified FFT augmentation module.

    forward(x) or forward(x1, x2):
      1. Extract amplitude and phase from input image(s) via FFT
      2. Feed amplitude through the chosen trainable model → augmented amplitude
      3. Reconstruct augmented image via IFFT using original phase

    aug_input='single' (Option A):
      x → FFT → (A, P) → ConvVAE(log(A)) → A_aug → IFFT(A_aug, P) → x_aug
      x_aug has x's identity (same phase) but perturbed texture (new amplitude)

    aug_input='pair' (Option B):
      x1, x2 → FFT → (A1, P1), (A2, P2)
      mixer(A1, A2) → A_aug → IFFT(A_aug, P2) → x_aug
      x_aug has x2's identity (P2 unchanged) with style influence from x1 (A1)
      x1 is typically a shuffled version of the batch (different identity)
    """
    def __init__(self, cfg):
        super().__init__()
        self.aug_input = cfg.aug_input
        self.amp_model = cfg.amp_model

        if cfg.aug_input == 'single':
            # Only ConvVAE makes sense for single-image option
            assert cfg.amp_model == 'conv_vae', \
                "aug_input='single' requires amp_model='conv_vae'"
            self.model = ConvVAE(cfg.img_size, cfg.latent_dim)

        elif cfg.aug_input == 'pair':
            if cfg.amp_model == 'adain':
                self.model = AdaINAmpMixer(channels=3)
            elif cfg.amp_model == 'cnn':
                self.model = CNNAmpMixer(channels=3)
            else:
                raise ValueError(f"amp_model='{cfg.amp_model}' not valid for pair input. "
                                 f"Use 'adain' or 'cnn'.")
        else:
            raise ValueError(f"aug_input must be 'single' or 'pair', got '{cfg.aug_input}'")

    def forward(self, x, x_style=None):
        """
        x       : (B, 3, H, W) — identity source (phase always kept from x)
        x_style : (B, 3, H, W) — style source for pair option (shuffled batch)
                  Ignored for single option.
        Returns : (x_aug, extra) where extra = (mu, logvar) for ConvVAE, else None
        """
        amp, phase, log_amp = image_to_amp_phase(x)

        if self.aug_input == 'single':
            log_amp_aug, mu, logvar = self.model(log_amp)
            amp_aug = log_amp_to_amp(log_amp_aug)
            extra   = (mu, logvar)

        else:   # pair
            assert x_style is not None, "aug_input='pair' requires x_style argument"
            amp_style, _, _ = image_to_amp_phase(x_style)
            amp_aug = self.model(amp_style, amp)   # style from x_style, structure from x
            extra   = None

        x_aug = amp_phase_to_image(amp_aug, phase)
        return x_aug, extra


# ============================================================================
# SECTION 5: BASELINE AUGMENTATION  (no trainable module)
# ============================================================================

class BaselineAmpMixup:
    """
    Simple within-batch amplitude mixup.
    For each image x in the batch:
      1. Take FFT → amplitude A_x, phase P_x
      2. Pick a random other image x_j from the batch
      3. A_aug = α·A_x + (1-α)·A_j
      4. x_aug = IFFT(A_aug, P_x)   ← phase from x, identity preserved

    dynamic_mixup=False : α = mixup_alpha (fixed, default 0.5)
    dynamic_mixup=True  : α ~ Beta(0.4, 0.4) sampled per batch
    """
    def __init__(self, cfg):
        self.dynamic = cfg.dynamic_mixup
        self.alpha   = cfg.mixup_alpha

    @torch.no_grad()
    def augment(self, x):
        B              = x.size(0)
        amp, phase, _  = image_to_amp_phase(x)

        # Random permutation for the style source within batch
        perm   = torch.randperm(B, device=x.device)
        amp_j  = amp[perm]

        if self.dynamic:
            alpha = float(np.random.beta(0.4, 0.4))
        else:
            alpha = self.alpha

        amp_aug = alpha * amp + (1.0 - alpha) * amp_j
        x_aug   = amp_phase_to_image(amp_aug, phase)
        return x_aug


# ============================================================================
# SECTION 6: RECOGNITION NETWORK  (ArcFace + ResNet-50)
# ============================================================================

class ArcFaceLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s = s; self.m = m
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m
        self.W     = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, feats, labels):
        feats   = F.normalize(feats, dim=1)
        W       = F.normalize(self.W,  dim=1)
        cos_t   = torch.clamp(feats @ W.T, -1 + 1e-7, 1 - 1e-7)
        sin_t   = torch.sqrt(torch.clamp(1 - cos_t**2, min=1e-8))
        cos_tm  = cos_t * self.cos_m - sin_t * self.sin_m
        cos_tm  = torch.where(cos_t > self.th, cos_tm, cos_t - self.mm)
        one_hot = torch.zeros_like(cos_t)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        output  = one_hot * cos_tm + (1 - one_hot) * cos_t
        return F.cross_entropy(output * self.s, labels)

    def get_logits(self, feats):
        return F.normalize(feats, dim=1) @ F.normalize(self.W, dim=1).T


class RecognitionNet(nn.Module):
    def __init__(self, num_classes, feature_dim=512, s=64.0, m=0.5):
        super().__init__()
        from torchvision.models import resnet50
        r             = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(r.children())[:-1])
        self.bn1      = nn.BatchNorm1d(2048)
        self.fc       = nn.Linear(2048, feature_dim)
        self.bn2      = nn.BatchNorm1d(feature_dim)
        self.head     = ArcFaceLoss(feature_dim, num_classes, s, m)

    def extract(self, x):
        f = self.backbone(x).flatten(1)
        f = self.bn2(self.fc(self.bn1(f)))
        return f

    def forward(self, x):
        f = self.extract(x)
        return self.head.get_logits(f), f

    def compute_loss(self, feats, labels):
        return self.head(feats, labels)

    def get_features(self, x):
        return F.normalize(self.extract(x), dim=1)


# ============================================================================
# SECTION 7: LR SCHEDULE
# ============================================================================

def build_lr_schedule(optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-5):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr / base_lr + (1 - min_lr / base_lr) * cosine
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# SECTION 8: EVALUATOR
# ============================================================================

class Evaluator:
    def __init__(self, rec_net, device):
        self.rec = rec_net
        self.dev = device

    @torch.no_grad()
    def extract(self, loader):
        feats, ids, specs = [], [], []
        self.rec.eval()
        for batch in tqdm(loader, desc='[Eval] Extracting', leave=False):
            x = batch['img'].to(self.dev)
            f = self.rec.get_features(x)
            feats.append(f.cpu().numpy())
            ids.extend(batch['label'].numpy().tolist())
            specs.extend(batch['spectrum'])
        return np.concatenate(feats), np.array(ids), np.array(specs)

    def verification(self, feats, ids, far_targets=(1e-6, 1e-5, 1e-4, 1e-3)):
        sim = feats @ feats.T
        gen_s, imp_s = [], []
        n = len(feats)
        for i in range(n):
            for j in range(i + 1, n):
                (gen_s if ids[i] == ids[j] else imp_s).append(float(sim[i, j]))
        gen_s = np.array(gen_s); imp_s = np.array(imp_s)

        results = {}
        imp_sorted = np.sort(imp_s)[::-1]
        for far in far_targets:
            idx = int(len(imp_s) * far)
            thr = imp_sorted[min(idx, len(imp_s) - 1)]
            results[f'TAR@FAR={far:.0e}'] = float((gen_s >= thr).mean())

        all_s = np.concatenate([gen_s, imp_s])
        all_l = np.concatenate([np.ones(len(gen_s)), np.zeros(len(imp_s))])
        fpr, tpr, _ = roc_curve(all_l, all_s)
        diff = np.abs(fpr - (1 - tpr))
        eer_idx = np.argmin(diff)
        results['EER'] = float((fpr[eer_idx] + (1 - tpr[eer_idx])) / 2)
        return results, gen_s, imp_s

    def identification(self, feats, ids, specs):
        feats_n = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        uid_list = np.unique(ids)
        g_f, g_id, p_f, p_id = [], [], [], []
        for uid in uid_list:
            idx = np.where(ids == uid)[0]
            if len(idx) < 2: continue
            gal_set = set()
            for spec in np.unique(specs[idx]):
                si = idx[specs[idx] == spec]
                gal_set.add(si[0])
                g_f.append(feats_n[si[0]]); g_id.append(uid)
            for i in idx:
                if i not in gal_set:
                    p_f.append(feats_n[i]); p_id.append(uid)
        if not g_f or not p_f:
            return {f'Rank-{k}': 0.0 for k in [1, 5, 10]}
        G, Gid = np.array(g_f), np.array(g_id)
        P, Pid = np.array(p_f), np.array(p_id)
        sim    = P @ G.T
        ranked = Gid[np.argsort(-sim, axis=1)]
        return {f'Rank-{k}': float(np.any(
            ranked[:, :min(k, len(Gid))] == Pid[:, None], axis=1).mean())
            for k in [1, 5, 10]}

    def evaluate(self, loader, far_targets=(1e-6, 1e-5, 1e-4, 1e-3)):
        feats, ids, specs   = self.extract(loader)
        ver, gen_s, imp_s   = self.verification(feats, ids, far_targets)
        idf                 = self.identification(feats, ids, specs)
        return {**ver, **idf,
                'num_samples'   : len(feats),
                'num_identities': int(np.unique(ids).size),
                'num_genuine'   : len(gen_s),
                'num_imposter'  : len(imp_s)}

    def print_results(self, res, tag=''):
        print(f"\n{'='*60}")
        print(f"  RESULTS  {tag}")
        print(f"{'='*60}")
        print(f"  Samples: {res['num_samples']}  "
              f"Identities: {res['num_identities']}")
        print(f"  Genuine pairs : {res['num_genuine']:,}")
        print(f"  Imposter pairs: {res['num_imposter']:,}")
        print("\n  Verification:")
        for k in sorted(res):
            if 'TAR' in k or k == 'EER':
                print(f"    {k:<28} {res[k]:.4f}")
        print("\n  Identification:")
        for k in sorted(res):
            if 'Rank' in k:
                print(f"    {k:<28} {res[k]:.4f}")
        print('='*60 + "\n")


# ============================================================================
# SECTION 9: PROPOSED METHOD TRAINER
# ============================================================================

class ProposedTrainer:
    """
    Adversarial FFT augmentation trainer.

    Per-batch (default):
      Step 1 — freeze F_θ, update AugModule to MAXIMISE classification loss
      Step 2 — freeze AugModule, update F_θ to MINIMISE loss on [x, x_aug]

    Per-epoch:
      Odd epochs  → step 1 only (update AugModule)
      Even epochs → step 2 only (update F_θ)
    """
    def __init__(self, cfg, num_classes, device):
        self.cfg = cfg
        self.dev = device

        self.rec = RecognitionNet(
            num_classes, cfg.feature_dim, cfg.arcface_s, cfg.arcface_m
        ).to(device)

        self.aug = FFTAugModule(cfg).to(device)

        self.opt_rec = optim.SGD(self.rec.parameters(), lr=cfg.lr_rec,
                                 momentum=0.9, weight_decay=cfg.weight_decay)
        self.opt_aug = optim.Adam(self.aug.parameters(), lr=cfg.lr_aug,
                                  betas=(0.9, 0.999))

        self.sched_rec = build_lr_schedule(
            self.opt_rec, cfg.warmup_epochs, cfg.num_epochs, cfg.lr_rec)

        self.best_tar  = 0.0
        self._phase    = 'aug'   # for per-epoch alternation

    # ── per-batch helpers ────────────────────────────────────────────────

    def _freeze(self, net):
        for p in net.parameters():
            p.requires_grad_(False)

    def _unfreeze(self, net):
        for p in net.parameters():
            p.requires_grad_(True)

    def _make_x_aug(self, x_sub):
        """Run augmentation on subset. Returns x_aug and any VAE extras."""
        if self.cfg.aug_input == 'single':
            x_aug, extra = self.aug(x_sub)
        else:
            perm          = torch.randperm(x_sub.size(0), device=x_sub.device)
            x_style       = x_sub[perm]
            x_aug, extra  = self.aug(x_sub, x_style)
        return x_aug, extra

    def _aug_loss(self, cls_loss, extra):
        """
        Augmentation module loss = -cls_loss (maximise hardness)
                                  + λ_kl * KL  (for ConvVAE only)
        """
        loss = -cls_loss
        if extra is not None:
            mu, logvar = extra
            kl  = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
            loss = loss + self.cfg.lambda_kl * kl
        return loss

    # ── single batch step ────────────────────────────────────────────────

    def train_batch(self, x, labels, update_aug=True, update_rec=True):
        B        = x.size(0)
        aug_size = max(1, int(B * self.cfg.aug_gamma))
        idx      = torch.randperm(B, device=self.dev)[:aug_size]
        x_sub    = x[idx]
        lb_sub   = labels[idx]

        losses = {}

        # ── Step 1: update augmentation module ───────────────────────────
        if update_aug:
            self._freeze(self.rec)
            self._unfreeze(self.aug)

            x_aug, extra = self._make_x_aug(x_sub)
            _, feats     = self.rec(x_aug)
            cls_loss     = self.rec.compute_loss(feats, lb_sub)
            aug_loss     = self._aug_loss(cls_loss, extra)

            self.opt_aug.zero_grad()
            aug_loss.backward()
            self.opt_aug.step()
            losses['aug_cls'] = cls_loss.item()

            self._unfreeze(self.rec)

        # ── Step 2: update recognition network ───────────────────────────
        if update_rec:
            self._freeze(self.aug)
            self._unfreeze(self.rec)

            with torch.no_grad():
                x_aug, _ = self._make_x_aug(x_sub)

            x_all  = torch.cat([x, x_aug],       dim=0)
            lb_all = torch.cat([labels, lb_sub],  dim=0)

            _, feats  = self.rec(x_all)
            rec_loss  = self.rec.compute_loss(feats, lb_all)

            self.opt_rec.zero_grad()
            rec_loss.backward()
            nn.utils.clip_grad_norm_(self.rec.parameters(), self.cfg.grad_clip)
            self.opt_rec.step()
            losses['rec'] = rec_loss.item()

            self._unfreeze(self.aug)

        return losses

    # ── training epoch ───────────────────────────────────────────────────

    def train_epoch(self, epoch, loader):
        self.rec.train()
        self.aug.train()

        # Per-epoch alternation: decide which update runs this epoch
        if self.cfg.alternate_per == 'epoch':
            do_aug = (epoch % 2 == 0)
            do_rec = not do_aug
        else:
            do_aug = do_rec = True    # both run every batch

        total_rec = 0.0
        total_aug = 0.0
        n_rec = n_aug = 0

        pbar = tqdm(loader, desc=f'[Proposed] Ep {epoch+1}', leave=False)
        for batch in pbar:
            x      = batch['img'].to(self.dev)
            labels = batch['label'].to(self.dev)

            losses = self.train_batch(x, labels, update_aug=do_aug, update_rec=do_rec)

            if 'rec' in losses:
                total_rec += losses['rec']; n_rec += 1
            if 'aug_cls' in losses:
                total_aug += losses['aug_cls']; n_aug += 1

            pbar.set_postfix({k: f'{v:.3f}' for k, v in losses.items()})

        self.sched_rec.step()

        avg_rec = total_rec / max(n_rec, 1)
        avg_aug = total_aug / max(n_aug, 1)
        print(f'  [Proposed] Epoch {epoch+1:3d}  rec={avg_rec:.4f}  '
              f'aug_cls={avg_aug:.4f}  '
              f'lr={self.opt_rec.param_groups[0]["lr"]:.5f}')
        return avg_rec

    def save(self, epoch, tag=''):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        path = os.path.join(self.cfg.save_dir, f'proposed{tag}.pt')
        torch.save({'epoch': epoch, 'rec': self.rec.state_dict(),
                    'aug': self.aug.state_dict()}, path)
        print(f'  [Proposed] Saved → {path}')


# ============================================================================
# SECTION 10: BASELINE TRAINER
# ============================================================================

class BaselineTrainer:
    """
    Amplitude mixup baseline — no trainable augmentation module.
    One training step per batch: just update F_θ on [x, x_aug].
    """
    def __init__(self, cfg, num_classes, device):
        self.cfg  = cfg
        self.dev  = device
        self.rec  = RecognitionNet(
            num_classes, cfg.feature_dim, cfg.arcface_s, cfg.arcface_m
        ).to(device)
        self.aug  = BaselineAmpMixup(cfg)
        self.opt  = optim.SGD(self.rec.parameters(), lr=cfg.lr_rec,
                              momentum=0.9, weight_decay=cfg.weight_decay)
        self.sched = build_lr_schedule(
            self.opt, cfg.warmup_epochs, cfg.num_epochs, cfg.lr_rec)
        self.best_tar = 0.0

    def train_epoch(self, epoch, loader):
        self.rec.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f'[Baseline] Ep {epoch+1}', leave=False)
        for batch in pbar:
            x      = batch['img'].to(self.dev)
            labels = batch['label'].to(self.dev)
            B      = x.size(0)

            aug_size = max(1, int(B * self.cfg.aug_gamma))
            idx      = torch.randperm(B, device=self.dev)[:aug_size]
            x_sub    = x[idx]
            lb_sub   = labels[idx]

            x_aug  = self.aug.augment(x_sub)
            x_all  = torch.cat([x, x_aug],      dim=0)
            lb_all = torch.cat([labels, lb_sub], dim=0)

            _, feats = self.rec(x_all)
            loss     = self.rec.compute_loss(feats, lb_all)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.rec.parameters(), self.cfg.grad_clip)
            self.opt.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.3f}')

        self.sched.step()
        avg = total_loss / len(loader)
        print(f'  [Baseline]  Epoch {epoch+1:3d}  loss={avg:.4f}  '
              f'lr={self.opt.param_groups[0]["lr"]:.5f}')
        return avg

    def save(self, epoch, tag=''):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        path = os.path.join(self.cfg.save_dir, f'baseline{tag}.pt')
        torch.save({'epoch': epoch, 'rec': self.rec.state_dict()}, path)
        print(f'  [Baseline] Saved → {path}')


# ============================================================================
# SECTION 11: COMPARISON RUNNER
# ============================================================================

def run_comparison(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n[Init] Device: {device}')

    train_loader, test_loader, num_classes, test_samples = create_dataloaders(cfg)
    print(f'[Init] num_classes (train): {num_classes}')

    results_proposed = None
    results_baseline = None

    # ── Run proposed method ───────────────────────────────────────────────
    if cfg.method in ('proposed', 'both'):
        print(f'\n{"="*60}')
        print(f'  PROPOSED  aug_input={cfg.aug_input}  '
              f'amp_model={cfg.amp_model}  alternate_per={cfg.alternate_per}')
        print(f'{"="*60}')

        trainer_p = ProposedTrainer(cfg, num_classes, device)
        ev_p      = Evaluator(trainer_p.rec, device)

        for epoch in range(cfg.num_epochs):
            trainer_p.train_epoch(epoch, train_loader)

            if (epoch + 1) % cfg.eval_freq == 0 or epoch == cfg.num_epochs - 1:
                res = ev_p.evaluate(test_loader)
                ev_p.print_results(res, tag=f'PROPOSED @ epoch {epoch+1}')

                tar = res.get('TAR@FAR=1e-05', 0.0)
                if tar > trainer_p.best_tar:
                    trainer_p.best_tar = tar
                    trainer_p.save(epoch, tag='_best')

            if (epoch + 1) % cfg.save_freq == 0:
                trainer_p.save(epoch, tag=f'_ep{epoch+1}')

        results_proposed = ev_p.evaluate(test_loader)
        ev_p.print_results(results_proposed, tag='PROPOSED  [FINAL]')

    # ── Run baseline ──────────────────────────────────────────────────────
    if cfg.method in ('baseline', 'both'):
        print(f'\n{"="*60}')
        print(f'  BASELINE  mixup_alpha={cfg.mixup_alpha}  '
              f'dynamic={cfg.dynamic_mixup}')
        print(f'{"="*60}')

        trainer_b = BaselineTrainer(cfg, num_classes, device)
        ev_b      = Evaluator(trainer_b.rec, device)

        for epoch in range(cfg.num_epochs):
            trainer_b.train_epoch(epoch, train_loader)

            if (epoch + 1) % cfg.eval_freq == 0 or epoch == cfg.num_epochs - 1:
                res = ev_b.evaluate(test_loader)
                ev_b.print_results(res, tag=f'BASELINE @ epoch {epoch+1}')

                tar = res.get('TAR@FAR=1e-05', 0.0)
                if tar > trainer_b.best_tar:
                    trainer_b.best_tar = tar
                    trainer_b.save(epoch, tag='_best')

            if (epoch + 1) % cfg.save_freq == 0:
                trainer_b.save(epoch, tag=f'_ep{epoch+1}')

        results_baseline = ev_b.evaluate(test_loader)
        ev_b.print_results(results_baseline, tag='BASELINE  [FINAL]')

    # ── Side-by-side comparison ───────────────────────────────────────────
    if results_proposed is not None and results_baseline is not None:
        print('\n' + '='*60)
        print('  CROSS-DOMAIN COMPARISON  (test spectra: '
              + str(cfg.test_spectra) + ')')
        print('='*60)
        print(f'  {"Metric":<28}  {"Proposed":>10}  {"Baseline":>10}  {"Delta":>10}')
        print(f'  {"-"*28}  {"-"*10}  {"-"*10}  {"-"*10}')
        for k in sorted(results_proposed):
            if not isinstance(results_proposed[k], float):
                continue
            vp = results_proposed[k]
            vb = results_baseline[k]
            delta = vp - vb
            sign  = '+' if delta >= 0 else ''
            print(f'  {k:<28}  {vp:>10.4f}  {vb:>10.4f}  {sign}{delta:>9.4f}')
        print('='*60 + '\n')

    return results_proposed, results_baseline


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print('\n' + '='*70)
    print('  FFT-BASED ADVERSARIAL AUGMENTATION — CROSS-DOMAIN EVALUATION')
    print('='*70 + '\n')

    # Run both methods for direct comparison
    cfg        = argparse.Namespace(**CONFIG)
    cfg.method = 'both'

    run_comparison(cfg)
