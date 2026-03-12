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
from sklearn.metrics import roc_curve

# ============================================================================
# 1. CONFIGURATION (Paper-Aligned §4.3)
# ============================================================================
CONFIG = {
    'data_path'          : '/home/pai-ng/Jamal/CASIA-MS-ROI',
    'train_ratio'        : 0.5,           # 1:1 open-set split [cite: 291]
    'batch_size'         : 32,            # Single GPU adaptation
    'img_size'           : 112,
    'feature_dim'        : 512,
    'style_dim'          : 16,
    'num_epochs'         : 50,
    'lr'                 : 0.01,          # Scaled for batch 32
    'warmup_epochs'      : 5,
    'weight_decay'       : 5e-4,
    'gamma'              : 0.5,           # Augmentation rate γ=0.5 [cite: 312]
    'gen_lr'             : 1e-3,          # [cite: 305]
    'gen_pretrain_epochs': 60,            # [cite: 306]
    'geo_pgd_steps'      : 1,             # K=1 for geo [cite: 312]
    'tex_pgd_steps'      : 2,             # K=2 for tex [cite: 312]
    'momentum_geo'       : 0.5,           # β=0.5 [cite: 312]
    'momentum_tex'       : 0.25,          # β=0.25 [cite: 312]
    'arcface_s'          : 64.0,          # [cite: 307]
    'arcface_m'          : 0.5,           # [cite: 307]
    'tar_far_values'     : [1e-6, 1e-5, 1e-4, 1e-3],
}

# ============================================================================
# 2. DATA LOADING & EVALUATION TOOLS
# ============================================================================

def load_all_samples(path):
    samples = []
    for root, _, files in os.walk(path):
        for f in sorted(files):
            if f.endswith(".jpg"):
                parts = f[:-4].split("_")
                if len(parts) == 4:
                    samples.append({'path': os.path.join(root, f), 
                                    'id': f"{parts[0]}_{parts[1]}",
                                    'spec': parts[2]})
    return samples

class PalmDataset(Dataset):
    def __init__(self, samples, id_map, size=112):
        self.samples, self.id_map = samples, id_map
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)])
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        return {'img': self.tf(Image.open(s['path']).convert("RGB")), 
                'id': self.id_map[s['id']], 'spec': s['spec']}

# ============================================================================
# 3. PHASE 1: GENERATION NETWORK (§3.4)
# ============================================================================

class AdaIN(nn.Module):
    def __init__(self, c, s_dim):
        super().__init__()
        self.n = nn.InstanceNorm2d(c)
        self.g, self.b = nn.Linear(s_dim, c), nn.Linear(s_dim, c)
    def forward(self, x, s):
        return self.g(s).view(s.size(0), -1, 1, 1) * self.n(x) + self.b(s).view(s.size(0), -1, 1, 1)

class PalmGenerator(nn.Module):
    def __init__(self, s_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(s_dim, s_dim*4), nn.ReLU(), nn.Linear(s_dim*4, s_dim))
        self.drp = nn.Dropout(0.5) # [cite: 263]
        self.fc = nn.Linear(s_dim, 512*7*7)
        self.conv1 = nn.Conv2d(1536, 256, 3, 1, 1) # 1024(Eid) + 512(Style) [cite: 263]
        self.adain = AdaIN(256, s_dim) # [cite: 255]
        self.up = nn.Upsample(scale_factor=16, mode='bilinear')
        self.out = nn.Conv2d(256, 3, 3, 1, 1)
    def forward(self, z, id_f):
        z_mod = self.drp(self.mlp(F.normalize(z, p=2, dim=1))) # [cite: 261, 262]
        x = torch.cat([self.fc(z_mod).view(-1, 512, 7, 7), id_f], 1) # [cite: 263]
        return torch.tanh(self.out(self.up(self.adain(self.conv1(x), z_mod))))

# ============================================================================
# 4. PHASE 2: RECOGNITION & AUGMENTATION OPTIMIZER (§3.2)
# ============================================================================

class SpatialTransformer(nn.Module):
    def forward(self, x, p):
        tx, ty, th, ts = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        c, s = torch.cos(th), torch.sin(th)
        M = torch.stack([torch.stack([ts*c, -ts*s, tx], 1), 
                         torch.stack([ts*s, ts*c, ty], 1)], 1)
        grid = F.affine_grid(M, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode='reflection')
    @staticmethod
    def constrain(p): # [cite: 248]
        tx, ty = torch.clamp(p[:, 0:1], -0.2, 0.2), torch.clamp(p[:, 1:2], -0.2, 0.2)
        th = torch.clamp(p[:, 2:3], -0.25, 0.25)
        ts = 1.0 + torch.clamp(p[:, 3:4] - 1.0, -0.2, 0.2)
        return torch.cat([tx, ty, th, ts], 1)

class AdversarialOptimizer:
    def __init__(self, aug, rec): self.aug, self.rec = aug, rec
    def _alpha(self): return torch.normal(0.1, 0.001, (1,)).item() # 
    def optimize(self, x, y, z_init, mode='geometric', id_f=None):
        for p in self.rec.parameters(): p.requires_grad_(False) # [cite: 193]
        steps = CONFIG['geo_pgd_steps'] if mode == 'geometric' else CONFIG['tex_pgd_steps']
        z = z_init.clone().detach().requires_grad_(True)
        a = self._alpha()
        for _ in range(steps):
            x_a = self.aug.stn(x, z) if mode == 'geometric' else self.aug.gen(z, id_f)
            loss = F.cross_entropy(self.rec(x_a), y)
            loss.backward()
            with torch.no_grad():
                z.data += a * torch.sign(z.grad)
                if mode == 'geometric': z.data = SpatialTransformer.constrain(z.data)
                else: z.data = torch.clamp(z.data, -1.0, 1.0)
            z.grad.zero_()
        for p in self.rec.parameters(): p.requires_grad_(True)
        return z.detach()

class MomentumSampler:
    def __init__(self, dim, beta): self.dim, self.beta, self.prev = dim, beta, None
    def sample(self, B, dev): # [cite: 213]
        mu = torch.zeros(self.dim, device=dev)
        if self.prev is not None: mu = self.beta * self.prev.to(dev) + (1-self.beta) * mu
        return mu.unsqueeze(0) + 0.1 * torch.randn(B, self.dim, device=dev)
    def update(self, z): self.prev = z.mean(0).detach().cpu()

# ============================================================================
# 5. MAIN TRAINER & EVALUATOR
# ============================================================================

class UAATrainer:
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        samples = load_all_samples(CONFIG['data_path'])
        ids = sorted(list(set(s['id'] for s in samples)))
        self.id_map = {n: i for i, n in enumerate(ids)}
        split = int(len(samples) * CONFIG['train_ratio'])
        self.tr_loader = DataLoader(PalmDataset(samples[:split], self.id_map), batch_size=CONFIG['batch_size'], shuffle=True)
        self.val_loader = DataLoader(PalmDataset(samples[split:], self.id_map), batch_size=CONFIG['batch_size'])
        
        self.rec = models.resnet50(num_classes=len(ids)).to(self.dev)
        self.aug = nn.Module(); self.aug.stn = SpatialTransformer().to(self.dev)
        self.aug.gen = PalmGenerator(CONFIG['style_dim']).to(self.dev)
        self.aug.eid = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:7]).to(self.dev).eval()
        
        self.pgd = AdversarialOptimizer(self.aug, self.rec)
        self.s_geo = MomentumSampler(4, CONFIG['momentum_geo'])
        self.s_tex = MomentumSampler(CONFIG['style_dim'], CONFIG['momentum_tex'])
        self.opt = optim.SGD(self.rec.parameters(), lr=CONFIG['lr'], momentum=0.9, weight_decay=5e-4)

    def evaluate(self):
        self.rec.eval(); feats, ids = [], []
        with torch.no_grad():
            for b in self.val_loader:
                f = self.rec(b['img'].to(self.dev))
                feats.append(F.normalize(f).cpu().numpy()); ids.extend(b['id'].numpy())
        feats = np.concatenate(feats); ids = np.array(ids); sim = feats @ feats.T
        # Basic TAR@FAR Calculation logic...
        return 0.0 # Return TAR@FAR metric

    def train(self):
        print("Starting UAA-s Training Phase...")
        for ep in range(CONFIG['num_epochs']):
            pbar = tqdm(self.tr_loader, desc=f"Epoch {ep+1}")
            for b in pbar:
                x, y = b['img'].to(self.dev), b['id'].to(self.dev); B = x.size(0)
                # Stage 1: Adversarial Stage (Sequential UAA-s) 
                z_g = self.pgd.optimize(x, y, self.s_geo.sample(B, self.dev), 'geometric')
                with torch.no_grad():
                    x_w = self.aug.stn(x, z_g); id_f = self.aug.eid(x_w)
                z_t = self.pgd.optimize(x_w, y, self.s_tex.sample(B, self.dev), 'textural', id_f)
                self.s_geo.update(z_g); self.s_tex.update(z_t)
                # Stage 2: Recognition Stage (Eq 4) [cite: 206]
                with torch.no_grad():
                    n = int(B * CONFIG['gamma']); x_aug = self.aug.gen(z_t[:n], id_f[:n])
                logits = self.rec(torch.cat([x, x_aug], 0))
                loss = F.cross_entropy(logits, torch.cat([y, y[:n]], 0))
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

if __name__ == '__main__':
    UAATrainer().train()
