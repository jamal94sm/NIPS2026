"""
UNIFIED ADVERSARIAL AUGMENTATION (UAA) - SEQUENTIAL (UAA-s)
Aligned with Jin et al., ICCV 2025.
"""

import os, math, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve

# ============================================================================
# 1. CONFIGURATION (§4.3)
# ============================================================================
CONFIG = {
    'data_path': '/home/pai-ng/Jamal/CASIA-MS-ROI',
    'train_ratio': 0.5,           # Paper 1:1 open-set split [cite: 291]
    'batch_size': 32,            # Adapted for single GPU [cite: 309]
    'img_size': 112,
    'feature_dim': 512,
    'style_dim': 16,             # Paper §4.6 [cite: 365]
    'num_epochs': 50,
    'lr': 0.01,                  # Scaled LR for batch 32 [cite: 308]
    'warmup_epochs': 5,
    'weight_decay': 5e-4,
    'gamma': 0.5,                # Augmentation rate γ=0.5 [cite: 312]
    'momentum_geo': 0.5,         # β=0.5 [cite: 312]
    'momentum_tex': 0.25,        # β=0.25 [cite: 312]
    'gen_lr': 1e-3,              # [cite: 305]
    'gen_pretrain_epochs': 60,   # [cite: 306]
    'arcface_s': 64.0,           # [cite: 307]
    'arcface_m': 0.5,            # [cite: 307]
}

# ============================================================================
# 2. MODULES (§3.3, §3.4)
# ============================================================================

class SpatialTransformer(nn.Module):
    """Differentiable spatial transformation module[cite: 80, 217]."""
    def forward(self, x, params):
        tx, ty, theta, ts = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        c, s = torch.cos(theta), torch.sin(theta)
        # Affine matrix construction [cite: 237-242]
        row1 = torch.stack([ts*c, -ts*s, tx], dim=1)
        row2 = torch.stack([ts*s, ts*c, ty], dim=1)
        M = torch.stack([row1, row2], dim=1)
        grid = F.affine_grid(M, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode='reflection')

    @staticmethod
    def constrain(p):
        """Project parameters into the allowed perturbation set S[cite: 203, 246]."""
        tx = torch.clamp(p[:, 0:1], -0.2, 0.2)      # |tx| < 0.2 [cite: 248]
        ty = torch.clamp(p[:, 1:2], -0.2, 0.2)      # |ty| < 0.2 [cite: 248]
        theta = torch.clamp(p[:, 2:3], -0.25, 0.25) # |tθ| < 0.25 [cite: 248]
        ts = 1.0 + torch.clamp(p[:, 3:4] - 1.0, -0.2, 0.2) # |ts-1| < 0.2 [cite: 248]
        return torch.cat([tx, ty, theta, ts], dim=1)

class AdaIN(nn.Module):
    """Adaptive Instance Normalization[cite: 255]."""
    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.fc_gamma = nn.Linear(style_dim, channels)
        self.fc_beta = nn.Linear(style_dim, channels)

    def forward(self, x, s):
        g = self.fc_gamma(s).view(s.size(0), -1, 1, 1)
        b = self.fc_beta(s).view(s.size(0), -1, 1, 1)
        return g * self.norm(x) + b

class PalmGenerator(nn.Module):
    """Identity-preserving generation network[cite: 96, 252]."""
    def __init__(self, style_dim=16):
        super().__init__()
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, style_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim * 4, style_dim)
        )
        self.fc = nn.Linear(style_dim, 512 * 7 * 7)
        # Spatial identity features (1024) concatenated with style (512) [cite: 263]
        self.conv1 = nn.Conv2d(1024 + 512, 256, 3, padding=1)
        self.adain1 = AdaIN(256, style_dim)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.out_conv = nn.Conv2d(256, 3, 3, padding=1)

    def forward(self, z, id_feat):
        # L2 normalization of style code [cite: 261]
        z_mod = self.style_mlp(F.normalize(z, p=2, dim=1))
        z_feat = self.fc(z_mod).view(z.size(0), 512, 7, 7)
        x = torch.cat([z_feat, id_feat], dim=1)
        x = F.relu(self.adain1(self.conv1(x), z_mod))
        return torch.tanh(self.out_conv(self.upsample(x)))

# ============================================================================
# 3. MOMENTUM SAMPLER (§3.2, Eq. 5)
# ============================================================================

class MomentumSampler:
    """Momentum-based dynamic sampling[cite: 91, 211]."""
    def __init__(self, dim, momentum):
        self.dim, self.momentum = dim, momentum
        self.z_prev = None

    def sample(self, B, device):
        mu = torch.zeros(self.dim, device=device)
        if self.z_prev is not None:
            # β·z*_{t-1} + (1-β)·μ [cite: 213]
            mu = self.momentum * self.z_prev.to(device) + (1 - self.momentum) * mu
        return mu.unsqueeze(0) + 0.1 * torch.randn(B, self.dim, device=device)

    def update(self, z_opt):
        self.z_prev = z_opt.mean(0).detach().cpu()

# ============================================================================
# 4. ADVERSARIAL OPTIMIZER (Eq. 3)
# ============================================================================

class AdversarialOptimizer:
    """PGD-based control vector optimization[cite: 200, 202]."""
    def __init__(self, aug_module, rec_net):
        self.aug = aug_module
        self.rec = rec_net

    def _get_alpha(self):
        # α ~ N(0.1, 0.001) [cite: 317]
        return torch.normal(0.1, 0.001, (1,)).item()

    def optimize(self, x, labels, z_init, aug_type='geometric', id_feat=None):
        self.rec.eval()
        for p in self.rec.parameters(): p.requires_grad_(False)

        # K=1 for geo, K=2 for tex [cite: 312]
        steps = 1 if aug_type == 'geometric' else 2
        z = z_init.clone().detach().requires_grad_(True)
        alpha = self._get_alpha()

        for _ in range(steps):
            if z.grad is not None: z.grad.zero_()
            
            if aug_type == 'geometric':
                x_aug = self.aug.stn(x, z)
            else:
                x_aug = self.aug.gen(z, id_feat)
            
            feats = self.rec(x_aug)
            loss = self.rec.compute_loss(feats, labels)
            loss.backward()

            if z.grad is not None:
                with torch.no_grad():
                    z.data = z.data + alpha * torch.sign(z.grad)
                    # Projection Π [cite: 203]
                    if aug_type == 'geometric':
                        z.data = SpatialTransformer.constrain(z.data)
                    else:
                        z.data = torch.clamp(z.data, -1.0, 1.0)

        for p in self.rec.parameters(): p.requires_grad_(True)
        self.rec.train()
        return z.detach()

# ============================================================================
# 5. MAIN TRAINER (Eq. 4)
# ============================================================================

class UAATrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Modules initialized here...
        self.s_geo = MomentumSampler(4, cfg['momentum_geo'])
        self.s_tex = MomentumSampler(cfg['style_dim'], cfg['momentum_tex'])

    def train_epoch(self, loader):
        self.rec.train()
        for batch in loader:
            x, labels = batch['img'].to(self.dev), batch['id'].to(self.dev)
            B = x.size(0)

            # --- Stage 1: Adversarial Augmentation (Sequential UAA-s)  ---
            
            # A. Geometric PGD (K=1) [cite: 312]
            z_geo_init = self.s_geo.sample(B, self.dev)
            z_geo_opt = self.pgd.optimize(x, labels, z_geo_init, 'geometric')
            
            # Apply geometric warp for textural identity condition
            with torch.no_grad():
                x_warped = self.aug.stn(x, z_geo_opt)
                id_feat = self.aug.get_id_feat(x_warped) # Identity conditions [cite: 115, 147]

            # B. Textural PGD (K=2) [cite: 312]
            z_tex_init = self.s_tex.sample(B, self.dev)
            z_tex_opt = self.pgd.optimize(x_warped, labels, z_tex_init, 'textural', id_feat)
            
            self.s_geo.update(z_geo_opt)
            self.s_tex.update(z_tex_opt)

            # --- Stage 2: Recognition Training (Eq. 4) [cite: 206] ---
            
            with torch.no_grad():
                n_aug = int(B * self.cfg['gamma']) # Augment γ fraction [cite: 312]
                x_aug = self.aug.gen(z_tex_opt[:n_aug], id_feat[:n_aug])

            # Combine original and challenging data [cite: 204]
            x_combined = torch.cat([x, x_aug], dim=0)
            lb_combined = torch.cat([labels, labels[:n_aug]], dim=0)

            feats = self.rec(x_combined)
            loss = self.rec.compute_loss(feats, lb_combined) # Minimize classification loss [cite: 140]
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
