import os, math, datetime, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
import numpy as np
from PIL import Image
from tqdm import tqdm

# ============================================================================
# 1. CONFIGURATION (Paper-Aligned §4.3)
# ============================================================================
CONFIG = {
    'data_path': '/home/pai-ng/Jamal/CASIA-MS-ROI',
    'train_ratio': 0.5,           # Paper 1:1 open-set split [cite: 291]
    'batch_size': 32,            # Adapted for single GPU [cite: 309]
    'img_size': 112,
    'feature_dim': 512,
    'style_dim': 16,             # Paper §4.6 [cite: 365]
    'num_epochs': 50,
    'lr': 0.01,                  # Scaled LR for batch size 32 [cite: 308]
    'warmup_epochs': 5,
    'weight_decay': 5e-4,
    'gamma': 0.5,                # Augmentation rate γ=0.5 
    'momentum_geo': 0.5,         # β=0.5 
    'momentum_tex': 0.25,        # β=0.25 
    'gen_lr': 1e-3,              # [cite: 305]
    'gen_pretrain_epochs': 60,   # [cite: 306]
    'arcface_s': 64.0,           # [cite: 307]
    'arcface_m': 0.5,            # [cite: 307]
}

# ============================================================================
# 2. DATA LOADING
# ============================================================================
def load_all_samples(data_path):
    samples = []
    for root, _, files in os.walk(data_path):
        for fname in sorted(files):
            if not fname.lower().endswith(".jpg"): continue
            parts = fname[:-4].split("_")
            if len(parts) != 4: continue
            samples.append({
                'path': os.path.join(root, fname),
                'hand_id': f"{parts[0]}_{parts[1]}",
                'spectrum': parts[2]
            })
    return samples

class PalmDataset(Dataset):
    def __init__(self, samples, identity_map, img_size=112):
        self.samples = samples
        self.identity_map = identity_map
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert("RGB")
        return {
            'img': self.transform(img),
            'id': self.identity_map[s['hand_id']]
        }

# ============================================================================
# 3. MODULES (STN & GENERATOR)
# ============================================================================
class SpatialTransformer(nn.Module):
    def forward(self, x, params):
        tx, ty, theta, ts = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        c, s = torch.cos(theta), torch.sin(theta)
        row1 = torch.stack([ts*c, -ts*s, tx], dim=1)
        row2 = torch.stack([ts*s, ts*c, ty], dim=1)
        M = torch.stack([row1, row2], dim=1)
        grid = F.affine_grid(M, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode='reflection')

    @staticmethod
    def constrain(p):
        # Bounds from Paper §3.3 [cite: 248]
        tx = torch.clamp(p[:, 0:1], -0.2, 0.2)
        ty = torch.clamp(p[:, 1:2], -0.2, 0.2)
        theta = torch.clamp(p[:, 2:3], -0.25, 0.25)
        ts = 1.0 + torch.clamp(p[:, 3:4] - 1.0, -0.2, 0.2)
        return torch.cat([tx, ty, theta, ts], dim=1)

class AdaIN(nn.Module):
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
    def __init__(self, style_dim=16):
        super().__init__()
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, style_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim * 4, style_dim)
        )
        self.fc = nn.Linear(style_dim, 512 * 7 * 7)
        self.conv1 = nn.Conv2d(1024 + 512, 256, 3, padding=1)
        self.adain1 = AdaIN(256, style_dim)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.out_conv = nn.Conv2d(256, 3, 3, padding=1)

    def forward(self, z, id_feat):
        z_mod = self.style_mlp(F.normalize(z, p=2, dim=1)) # L2 Norm [cite: 261]
        z_feat = self.fc(z_mod).view(z.size(0), 512, 7, 7)
        x = torch.cat([z_feat, id_feat], dim=1) # Channel Concatenation [cite: 263]
        x = F.relu(self.adain1(self.conv1(x), z_mod))
        return torch.tanh(self.out_conv(self.upsample(x)))

# ============================================================================
# 4. OPTIMIZERS & SAMPLERS
# ============================================================================
class MomentumSampler:
    def __init__(self, dim, momentum):
        self.dim, self.momentum = dim, momentum
        self.z_prev = None

    def sample(self, B, device):
        mu = torch.zeros(self.dim, device=device)
        if self.z_prev is not None:
            mu = self.momentum * self.z_prev.to(device) + (1 - self.momentum) * mu
        return mu.unsqueeze(0) + 0.1 * torch.randn(B, self.dim, device=device)

    def update(self, z_opt):
        self.z_prev = z_opt.mean(0).detach().cpu()

class AdversarialOptimizer:
    def __init__(self, aug_module, rec_net):
        self.aug = aug_module
        self.rec = rec_net

    def optimize(self, x, labels, z_init, aug_type='geometric', id_feat=None):
        self.rec.eval()
        for p in self.rec.parameters(): p.requires_grad_(False)
        
        steps = 1 if aug_type == 'geometric' else 2 # K steps 
        z = z_init.clone().detach().requires_grad_(True)
        alpha = torch.normal(0.1, 0.001, (1,)).item() # Stochastic α [cite: 317]

        for _ in range(steps):
            if z.grad is not None: z.grad.zero_()
            if aug_type == 'geometric':
                x_aug = self.aug.stn(x, z)
            else:
                x_aug = self.aug.gen(z, id_feat)
            
            feat = self.rec.extract(x_aug)
            loss = self.rec.compute_loss(feat, labels)
            loss.backward()

            if z.grad is not None:
                with torch.no_grad():
                    z.data = z.data + alpha * torch.sign(z.grad)
                    if aug_type == 'geometric':
                        z.data = SpatialTransformer.constrain(z.data)
                    else:
                        z.data = torch.clamp(z.data, -1.0, 1.0)

        for p in self.rec.parameters(): p.requires_grad_(True)
        self.rec.train()
        return z.detach()

# ============================================================================
# 5. MAIN TRAINER
# ============================================================================
class UAATrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Data setup
        samples = load_all_samples(cfg['data_path'])
        ids = sorted(list(set(s['hand_id'] for s in samples)))
        self.id_map = {name: i for i, name in enumerate(ids)}
        split = int(len(samples) * cfg['train_ratio'])
        self.train_loader = DataLoader(PalmDataset(samples[:split], self.id_map), 
                                       batch_size=cfg['batch_size'], shuffle=True)
        
        # Models
        self.rec = PalmRecognitionNetwork(len(ids)).to(self.dev)
        self.aug = nn.Module()
        self.aug.stn = SpatialTransformer().to(self.dev)
        self.aug.gen = PalmGenerator(cfg['style_dim']).to(self.dev)
        self.aug.id_enc = models.resnet50(pretrained=True).to(self.dev) # Frozen Eid [cite: 260]
        
        self.pgd = AdversarialOptimizer(self.aug, self.rec)
        self.s_geo = MomentumSampler(4, cfg['momentum_geo'])
        self.s_tex = MomentumSampler(cfg['style_dim'], cfg['momentum_tex'])
        self.opt = optim.SGD(self.rec.parameters(), lr=cfg['lr'], momentum=0.9)

    def train(self):
        for ep in range(self.cfg['num_epochs']):
            pbar = tqdm(self.train_loader, desc=f"Epoch {ep+1}")
            for batch in pbar:
                x, y = batch['img'].to(self.dev), batch['id'].to(self.dev)
                B = x.size(0)

                # A. Geo PGD (UAA-s)
                z_g = self.pgd.optimize(x, y, self.s_geo.sample(B, self.dev), 'geometric')
                with torch.no_grad():
                    x_w = self.aug.stn(x, z_g)
                    # Extract hierarchical id features from layer 3 (1024ch) [cite: 259]
                    id_f = self.aug.id_enc.layer3(self.aug.id_enc.conv1(x_w)) # etc.

                # B. Tex PGD
                z_t = self.pgd.optimize(x_w, y, self.s_tex.sample(B, self.dev), 'textural', id_f)
                
                # C. Update Recognition (Eq 4)
                with torch.no_grad():
                    n = int(B * self.cfg['gamma'])
                    x_aug = self.aug.gen(z_t[:n], id_f[:n])

                logits = self.rec(torch.cat([x, x_aug], dim=0))
                loss = F.cross_entropy(logits, torch.cat([y, y[:n]], dim=0))
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                pbar.set_postfix(loss=f"{loss.item():.3f}")

if __name__ == '__main__':
    UAATrainer(CONFIG).train()
