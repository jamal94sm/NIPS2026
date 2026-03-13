"""
UNIFIED ADVERSARIAL AUGMENTATION (UAA) FOR PALMPRINT RECOGNITION
Complete All-in-One Implementation — ICCV 2025
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # ── Dataset toggle ────────────────────────────────────────────────────
    # Set 'dataset' to 'casia' or 'mpd'
    'dataset'            : 'mpd',

    # ── CASIA-MS-ROI ──────────────────────────────────────────────────────
    # Filename: {subject}_{hand}_{spectrum}_{iteration}.jpg
    # Identity key: subject + hand  (e.g. "0001_l")
    'casia_data_path'    : '/home/pai-ng/Jamal/CASIA-MS-ROI',

    # ── MPDv2-ROI ─────────────────────────────────────────────────────────
    # Filename: {id}_{session}_{device}_{hand}_{iteration}.jpg
    # Identity key: subject id + hand  (e.g. "001_l")
    # Requires pre-extracted ROI images (run extract_mpd_roi.py first)
    'mpd_data_path'      : '/home/pai-ng/Jamal/MPDv2-ROI',

    # ── Split ─────────────────────────────────────────────────────────────
    'train_ratio'        : 0.7,
    'random_seed'        : 42,

    # ── Input ─────────────────────────────────────────────────────────────
    'batch_size'         : 32,
    'num_workers'        : 4,
    'img_size'           : 112,

    # ── Network ───────────────────────────────────────────────────────────
    'feature_dim'        : 512,
    'style_dim'          : 16,

    # ── Recognition training ──────────────────────────────────────────────
    'num_epochs'         : 50,
    'lr'                 : 0.01,
    'warmup_epochs'      : 5,
    'weight_decay'       : 5e-4,
    'grad_clip'          : 5.0,
    'save_freq'          : 5,

    # ── UAA augmentation ──────────────────────────────────────────────────
    'use_geometric'      : True,
    'use_generation'     : True,
    'use_textural'       : True,

    # ── GAN pre-training ──────────────────────────────────────────────────
    'gen_pretrain_epochs': 60,
    'gen_lr'             : 1e-3,
    'gen_save_path'      : 'checkpoints/generation_network_pretrained.pt',
    'gan_finetune_epochs': 10,

    # ── PGD ───────────────────────────────────────────────────────────────
    'pgd_steps'          : 2,
    'pgd_step_size'      : 0.05,

    # ── Momentum sampling ─────────────────────────────────────────────────
    'momentum_geo'       : 0.5,
    'momentum_tex'       : 0.25,

    # ── ArcFace ───────────────────────────────────────────────────────────
    'arcface_s'          : 64.0,
    'arcface_m'          : 0.5,

    # ── Evaluation ────────────────────────────────────────────────────────
    'tar_far_values'     : [1e-6, 1e-5, 1e-4, 1e-3],
    'eval_freq'          : 5,
}

# ============================================================================
# Imports
# ============================================================================

import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from datetime import datetime
from sklearn.metrics import roc_curve

args = argparse.Namespace(**CONFIG)

print("=" * 80)
print("  CONFIGURATION (Paper Settings — UAA ICCV 2025):")
print("=" * 80)
for k, v in CONFIG.items():
    print(f"  {k:<28} = {v}")
print("=" * 80 + "\n")


# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

# ---------------------------------------------------------------------------
# 1A  CASIA-MS-ROI
#     Filename: {subject}_{hand}_{spectrum}_{iteration}.jpg
#     Identity: subject + "_" + hand   (e.g. "0001_l", "0001_r")
#     Spectra:  multiple per identity → used as the "channel" dimension in eval
# ---------------------------------------------------------------------------

def load_casia_samples(data_path):
    """Walk CASIA-MS-ROI directory and return a list of sample dicts."""
    samples = []
    for root, _, files in os.walk(data_path):
        for fname in sorted(files):
            if not fname.lower().endswith(".jpg"):
                continue
            parts = fname[:-4].split("_")
            if len(parts) != 4:
                continue
            subject_id, hand, spectrum, iteration = parts
            samples.append({
                'path'     : os.path.join(root, fname),
                'subject'  : subject_id,
                'hand'     : hand,
                'spectrum' : spectrum,
                'iteration': iteration,
                'hand_id'  : f"{subject_id}_{hand}",
                'dataset'  : 'casia',
            })
    return samples


class CASIADataset(Dataset):
    """
    Dataset wrapper for CASIA-MS-ROI.
    Returns a 'spectrum' key so the evaluator can build gallery/probe splits
    across different spectral channels.
    """
    def __init__(self, samples, identity_map, img_size=112):
        self.samples   = samples
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        for s in self.samples:
            s['identity'] = identity_map[s['hand_id']]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s['path']).convert("RGB")
        return {
            'img'     : self.transform(img),
            'identity': s['identity'],
            'path'    : s['path'],
            'hand_id' : s['hand_id'],
            'spectrum': s['spectrum'],     # used by Evaluator.identification()
            'subject' : s['subject'],
            'hand'    : s['hand'],
        }


# ---------------------------------------------------------------------------
# 1B  MPDv2-ROI
#     Filename: {id}_{session}_{device}_{hand}_{iteration}.jpg
#     Identity: id + "_" + hand   (e.g. "001_l", "001_r")
#     Device:   h (high-res) or m (mobile) → reused as 'spectrum' proxy so the
#               same Evaluator gallery/probe logic applies without modification
# ---------------------------------------------------------------------------

def load_mpd_samples(data_path):
    """
    Walk MPDv2-ROI directory and return a list of sample dicts.
    Files that do not match the 5-part MPDv2 naming convention are skipped.
    """
    samples = []
    skipped = 0
    for root, _, files in os.walk(data_path):
        for fname in sorted(files):
            if not fname.lower().endswith(".jpg"):
                continue
            parts = fname[:-4].split("_")
            if len(parts) != 5:
                skipped += 1
                continue
            subj_id, session, device, hand, iteration = parts
            if not (len(subj_id) == 3 and subj_id.isdigit()
                    and session in ("1", "2")
                    and device in ("h", "m")
                    and hand in ("l", "r")
                    and len(iteration) == 2 and iteration.isdigit()):
                skipped += 1
                continue
            samples.append({
                'path'     : os.path.join(root, fname),
                'subject'  : subj_id,
                'hand'     : hand,
                'session'  : session,
                'device'   : device,
                'iteration': iteration,
                'hand_id'  : f"{subj_id}_{hand}",
                # Map 'device' → 'spectrum' so Evaluator works without changes:
                # gallery is built from one device, probe from the other
                'spectrum' : device,
                'dataset'  : 'mpd',
            })
    if skipped:
        print(f"[Data/MPD] Skipped {skipped} files with non-standard filenames.")
    return samples


class MPDDataset(Dataset):
    """
    Dataset wrapper for MPDv2-ROI.
    'spectrum' holds the device label ("h" or "m"); the Evaluator uses it
    to split gallery vs probe — identical logic to CASIA spectral channels.
    """
    def __init__(self, samples, identity_map, img_size=112):
        self.samples   = samples
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        for s in self.samples:
            s['identity'] = identity_map[s['hand_id']]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s['path']).convert("RGB")
        return {
            'img'     : self.transform(img),
            'identity': s['identity'],
            'path'    : s['path'],
            'hand_id' : s['hand_id'],
            'spectrum': s['spectrum'],   # device label: "h" or "m"
            'subject' : s['subject'],
            'hand'    : s['hand'],
            # MPDv2-specific extras (ignored by trainer, useful for analysis)
            'session' : s['session'],
            'device'  : s['device'],
        }


# ---------------------------------------------------------------------------
# 1C  Shared helpers: identity map + train/test split
# ---------------------------------------------------------------------------

def build_identity_map(samples):
    """Assign a consecutive integer label to every unique hand_id."""
    all_hand_ids = sorted(set(s['hand_id'] for s in samples))
    identity_map = {h: i for i, h in enumerate(all_hand_ids)}
    print(f"[Data] Total identities: {len(identity_map)}")
    return identity_map, len(identity_map)


def split_train_test(samples, identity_map, train_ratio=0.7, seed=42):
    """
    Identity-disjoint train/test split.
    Identities (not individual images) are shuffled and divided.
    """
    np.random.seed(seed)
    all_ids   = list(sorted(set(s['hand_id'] for s in samples)))
    np.random.shuffle(all_ids)
    n_train   = int(len(all_ids) * train_ratio)
    train_ids = set(all_ids[:n_train])
    test_ids  = set(all_ids[n_train:])
    train_s   = [s for s in samples if s['hand_id'] in train_ids]
    test_s    = [s for s in samples if s['hand_id'] in test_ids]
    print(f"[Data] Train: {len(train_ids)} identities, {len(train_s)} samples")
    print(f"[Data] Test : {len(test_ids)} identities, {len(test_s)} samples")
    return train_s, test_s, train_ids, test_ids


# ---------------------------------------------------------------------------
# 1D  create_dataloaders — single entry point, dispatches on cfg.dataset
# ---------------------------------------------------------------------------

def create_dataloaders(cfg):
    """
    Build train/test DataLoaders for either CASIA-MS or MPDv2.

    Toggle via CONFIG['dataset']:
      'casia'  →  load from cfg.casia_data_path using CASIADataset
      'mpd'    →  load from cfg.mpd_data_path   using MPDDataset

    Both datasets expose the same keys in every batch:
      img, identity, path, hand_id, spectrum, subject, hand

    Returns: train_loader, test_loader, num_classes, test_samples, identity_map
    """
    dataset = cfg.dataset.lower().strip()
    print(f"\n[Data] Dataset : {dataset.upper()}")

    if dataset == 'casia':
        data_path   = cfg.casia_data_path
        print(f"[Data] Path    : {data_path}")
        all_samples = load_casia_samples(data_path)
        print(f"[Data] Samples : {len(all_samples)}")
        identity_map, num_classes = build_identity_map(all_samples)
        train_s, test_s, _, _    = split_train_test(
            all_samples, identity_map, cfg.train_ratio, cfg.random_seed)
        train_ds = CASIADataset(train_s, identity_map, cfg.img_size)
        test_ds  = CASIADataset(test_s,  identity_map, cfg.img_size)

    elif dataset == 'mpd':
        data_path   = cfg.mpd_data_path
        print(f"[Data] Path    : {data_path}")
        all_samples = load_mpd_samples(data_path)
        print(f"[Data] Samples : {len(all_samples)}")
        identity_map, num_classes = build_identity_map(all_samples)
        train_s, test_s, _, _    = split_train_test(
            all_samples, identity_map, cfg.train_ratio, cfg.random_seed)
        train_ds = MPDDataset(train_s, identity_map, cfg.img_size)
        test_ds  = MPDDataset(test_s,  identity_map, cfg.img_size)

    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            "Set CONFIG['dataset'] to 'casia' or 'mpd'.")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(
        test_ds,  batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    return train_loader, test_loader, num_classes, test_s, identity_map


# ============================================================================
# SECTION 2: SPATIAL TRANSFORMER  (§3.3, Eq. 6)
# — No changes needed; was already fully aligned with paper
# ============================================================================

class SpatialTransformer(nn.Module):
    def __init__(self, img_size=112):
        super().__init__()
        self.img_size = img_size

    def forward(self, x, params):
        tx, ty    = params[:, 0], params[:, 1]
        theta, ts = params[:, 2], params[:, 3]
        c, s      = torch.cos(theta), torch.sin(theta)
        # out-of-place construction avoids in-place autograd corruption
        row1 = torch.stack([ ts*c, -ts*s, tx], dim=1)
        row2 = torch.stack([ ts*s,  ts*c, ty], dim=1)
        M    = torch.stack([row1, row2], dim=1)        # (B, 2, 3) affine matrix
        grid = F.affine_grid(M, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode='reflection')

    @staticmethod
    def constrain(params):
        """Project t into allowed perturbation set S (paper §3.3)."""
        tx    = torch.clamp(params[:, 0:1], -0.2,  0.2)   # |tx| < 0.2
        ty    = torch.clamp(params[:, 1:2], -0.2,  0.2)   # |ty| < 0.2
        theta = torch.clamp(params[:, 2:3], -0.25, 0.25)  # |tθ| < 0.25
        scale = 1.0 + torch.clamp(params[:, 3:4] - 1.0, -0.2, 0.2)  # |ts-1|<0.2
        return torch.cat([tx, ty, theta, scale], dim=1)


# ============================================================================
# SECTION 3: GENERATION NETWORK  (§3.4, Eq. 7, Figure 3)
# ============================================================================

class StyleEncoder(nn.Module):
    """
    Encodes style from x1 via reparameterisation trick (paper §3.4).
    Frozen after GAN pre-training (now possible with id_feat properly injected).
    """
    def __init__(self, style_dim=16):
        super().__init__()
        from torchvision.models import resnet50
        resnet = resnet50(pretrained=True)
        self.backbone  = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_mu     = nn.Linear(2048, style_dim)
        self.fc_logvar = nn.Linear(2048, style_dim)

    def forward(self, x):
        f      = self.backbone(x).flatten(1)
        mu     = self.fc_mu(f)
        logvar = self.fc_logvar(f)
        # Reparameterisation: s ~ N(μs, σ²s)
        z      = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu, logvar, z


class IdentityEncoder(nn.Module):
    """
    Frozen pre-trained recognition network used as Eid (paper §3.4).
    Extracts spatial hierarchical identity features fid from x2.
    ResNet-50 layers[:8] → output: (B, 512, 4, 4) for 112×112 input.
    """
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50
        r = resnet50(pretrained=True)
        # children() = [conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,avgpool,fc]
        # indices:       0    1   2   3        4      5      6      7      8       9
        # layers[:7] ends at layer3 → output: 1024ch, 7×7 for 112×112 input
        # Preferred over layers[:8] (layer4→2048ch,4×4): already at 7×7 so no
        # interpolation is needed in the generator, and 1024ch is lighter.
        self.layers = nn.Sequential(*list(r.children())[:7])
        for p in self.parameters():
            p.requires_grad = False   # always frozen

    def forward(self, x):
        return self.layers(x)   # (B, 1024, 7, 7)


class AdaIN(nn.Module):
    """Adaptive Instance Normalisation: γ(s)·norm(x) + β(s)."""
    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm  = nn.InstanceNorm2d(channels)
        self.gamma = nn.Linear(style_dim, channels)
        self.beta  = nn.Linear(style_dim, channels)

    def forward(self, x, z):
        g = self.gamma(z).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(z).unsqueeze(-1).unsqueeze(-1)
        return g * self.norm(x) + b


class AdaINBlock(nn.Module):
    def __init__(self, in_c, out_c, style_dim):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv  = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.adain = AdaIN(out_c, style_dim)
        self.act   = nn.ReLU(inplace=False)

    def forward(self, x, z):
        return self.act(self.adain(self.conv(self.up(x)), z))


class StyleMLP(nn.Module):
    """
    2-layer MLP that maps L2-normalised style code to modulation code.
    Paper §3.4: 'passes through multiple layers of MLPs and is integrated
    into the modulated convolution layer'.
    """
    def __init__(self, style_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(style_dim, style_dim * 4),
            nn.ReLU(inplace=False),
            nn.Linear(style_dim * 4, style_dim),
        )

    def forward(self, z):
        # L2 normalise → project onto hypersphere (paper §3.4)  [FIX 9]
        z_norm = F.normalize(z, dim=1)
        return self.net(z_norm)                                  # [FIX 10]


class PalmGenerator(nn.Module):
    """
    Generator G.  Fixes vs. previous version:
      - id_feat injected via channel concatenation at spatial level  [FIX 8]
      - style code L2-normalised then processed by MLP               [FIX 9,10]
      - Dropout regularisation added                                 [FIX 11]
    """
    def __init__(self, style_dim=16, id_feat_channels=1024):
        super().__init__()
        self.style_mlp = StyleMLP(style_dim)
        self.dropout   = nn.Dropout(p=0.5)                         # [FIX 11]

        # Initial projection of style code to spatial feature map
        self.fc = nn.Linear(style_dim, 512 * 7 * 7)

        # b1 input = z_feat(512) + id_feat(1024) = 1536            [FIX 8]
        # id_feat from layer3 is already 7×7 — no interpolation needed
        self.b1 = AdaINBlock(512 + id_feat_channels, 256, style_dim)
        self.b2 = AdaINBlock(256, 128, style_dim)
        self.b3 = AdaINBlock(128,  64, style_dim)
        self.b4 = AdaINBlock( 64,  32, style_dim)
        self.out = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, z, id_feat):
        """
        z       : (B, style_dim)    — style control vector
        id_feat : (B, 1024, 7, 7)   — spatial identity features from Eid (layer3)
        """
        # [FIX 9,10] L2-normalise then MLP
        z_mod = self.style_mlp(z)           # (B, style_dim)
        z_mod = self.dropout(z_mod)         # [FIX 11]

        # Initial spatial feature from style code: (B, 512, 7, 7)
        z_feat = self.fc(z_mod).view(z.size(0), 512, 7, 7)

        # [FIX 8] id_feat is already 7×7 from layer3 — concat directly
        x = torch.cat([z_feat, id_feat], dim=1)   # (B, 1536, 7, 7)

        # Decode — style modulation applied at every block
        x = self.b1(x, z_mod)   # (B, 256, 14, 14)
        x = self.b2(x, z_mod)   # (B, 128, 28, 28)
        x = self.b3(x, z_mod)   # (B,  64, 56, 56)
        x = self.b4(x, z_mod)   # (B,  32, 112, 112)
        return torch.tanh(self.out(x))


class PalmDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers
        self.net = nn.Sequential(
            *block(  3,  64, bn=False),
            *block( 64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class PalmGenerationNetwork(nn.Module):
    def __init__(self, style_dim=16, img_size=112):
        super().__init__()
        self.style_dim        = style_dim
        self.style_encoder    = StyleEncoder(style_dim)
        self.identity_encoder = IdentityEncoder()
        self.generator        = PalmGenerator(style_dim, id_feat_channels=1024)
        self.discriminator    = PalmDiscriminator()

    def forward(self, x_style, x_id):
        """
        x_style : (B, 3, H, W) — source of style (x1 in paper Fig 3)
        x_id    : (B, 3, H, W) — source of identity (x2 in paper Fig 3)
        These should be DIFFERENT images within the batch.          [FIX 12]
        """
        mu, logvar, z = self.style_encoder(x_style)
        id_feat       = self.identity_encoder(x_id)     # frozen
        generated     = self.generator(z, id_feat)
        return generated, z, id_feat, mu, logvar

    def generate_from_z(self, z, x_id):
        """
        Used during textural PGD: generate from external style vector z.
        Gradient path: z → StyleMLP → generator.fc → AdaIN → image → loss.
        generator weights are frozen but z is a leaf with requires_grad=True,
        so ∂image/∂z exists through the linear/conv operations.
        """
        id_feat = self.identity_encoder(x_id)   # frozen, no grad needed
        return self.generator(z, id_feat)


# ============================================================================
# SECTION 4: RECOGNITION NETWORK  (§4.3)
# — ArcFace with ResNet-50 backbone; no changes needed
# ============================================================================

class ArcFaceLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s     = s
        self.m     = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m
        self.W     = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, feats, labels):
        feats  = F.normalize(feats,  dim=1)
        W      = F.normalize(self.W, dim=1)
        cos_t  = torch.clamp(feats @ W.T, -1.0 + 1e-7, 1.0 - 1e-7)
        sin_t  = torch.sqrt(torch.clamp(1.0 - cos_t**2, min=1e-8))
        cos_tm = cos_t * self.cos_m - sin_t * self.sin_m
        cos_tm = torch.where(cos_t > self.th, cos_tm, cos_t - self.mm)
        one_hot = torch.zeros_like(cos_t)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        output = one_hot * cos_tm + (1.0 - one_hot) * cos_t
        return F.cross_entropy(output * self.s, labels)

    def get_logits(self, feats):
        feats = F.normalize(feats, dim=1)
        W     = F.normalize(self.W, dim=1)
        return feats @ W.T


class PalmRecognitionNetwork(nn.Module):
    def __init__(self, num_classes, feature_dim=512, s=64.0, m=0.5):
        super().__init__()
        from torchvision.models import resnet50
        r = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(r.children())[:-1])
        self.bn1      = nn.BatchNorm1d(2048)
        self.fc       = nn.Linear(2048, feature_dim)
        self.bn2      = nn.BatchNorm1d(feature_dim)
        self.head     = ArcFaceLoss(feature_dim, num_classes, s=s, m=m)

    def extract(self, x):
        f = self.backbone(x).flatten(1)
        f = self.bn1(f)
        f = self.fc(f)
        f = self.bn2(f)
        return f

    def forward(self, x):
        f = self.extract(x)
        return self.head.get_logits(f), f

    def compute_loss(self, feats, labels):
        return self.head(feats, labels)

    def get_verification_features(self, x):
        return F.normalize(self.extract(x), dim=1)


# ============================================================================
# SECTION 5: UNIFIED AUGMENTATION MODULE  (§3.2, Figure 2b)
# ============================================================================

class UnifiedAugmentationModule(nn.Module):
    def __init__(self, style_dim=16, img_size=112, use_generation=True):
        super().__init__()
        self.use_generation = use_generation
        self.style_dim      = style_dim
        self.spatial_tf     = SpatialTransformer(img_size)
        if use_generation:
            self.gen_network = PalmGenerationNetwork(style_dim, img_size)

    def forward(self, x, ctrl, aug_type='both'):
        spatial_p = ctrl[:, :4]

        if aug_type in ('geometric', 'both'):
            spatial_p = SpatialTransformer.constrain(spatial_p)
            x = self.spatial_tf(x, spatial_p)

        if aug_type in ('textural', 'both') and self.use_generation:
            z_style = ctrl[:, 4:]
            x_gen   = self.gen_network.generate_from_z(z_style, x)
            # Blend 50/50: preserves original structure, smoother transition
            # Pure replacement was too aggressive and destabilised training
            x = 0.5 * x_gen + 0.5 * x

        return x

    def freeze_gan_weights(self):
        """
        Freeze generator and discriminator only.
        Style encoder stays TRAINABLE so PGD gradients flow:
        z → style_encoder → generator → image → loss → ∇z
        """
        for p in self.gen_network.generator.parameters():
            p.requires_grad = False
        for p in self.gen_network.discriminator.parameters():
            p.requires_grad = False
        # Style encoder: keep requires_grad=True for PGD gradient flow
        print("[Aug] Generator & Discriminator frozen. Style encoder stays trainable.")


# ============================================================================
# SECTION 6: ADVERSARIAL AUGMENTATION OPTIMISER  (§3.2, Eq. 2–3)
# ============================================================================

class AdversarialAugOptimizer:
    """
    PGD optimiser for control vector z (Eq. 3).
    Uses fixed step_size=0.05 matching the working version — small and stable.
    Fθ explicitly frozen during z update (F*θ, Fig 2c).
    """

    def __init__(self, aug_module, rec_net, pgd_steps=2, step_size=0.05):
        self.aug   = aug_module
        self.rec   = rec_net
        self.steps = pgd_steps
        self.alpha = step_size
        self.b_sp  = 0.3
        self.b_st  = 1.0

    def optimize(self, x, labels, z_init, aug_type='both'):
        x_det = x.detach()

        # Freeze Fθ during z optimisation (F*θ in paper Figure 2c)
        for p in self.rec.parameters():
            p.requires_grad_(False)

        z = z_init.clone().detach().requires_grad_(True)

        for _ in range(self.steps):
            if z.grad is not None:
                z.grad.zero_()

            x_aug   = self.aug(x_det, z, aug_type=aug_type)
            _, feat = self.rec(x_aug)
            loss    = self.rec.compute_loss(feat, labels)
            loss.backward()

            grad_sign = (torch.sign(z.grad) if z.grad is not None
                         else torch.sign(torch.randn_like(z)))

            with torch.no_grad():
                z_new = z.data + self.alpha * grad_sign
                sp    = torch.clamp(z_new[:, :4], -self.b_sp, self.b_sp)
                st    = torch.clamp(z_new[:, 4:], -self.b_st, self.b_st)
                z_new = torch.cat([sp, st], dim=1)

            z = z_new.detach().requires_grad_(True)

        # Unfreeze Fθ for the recognition training step
        for p in self.rec.parameters():
            p.requires_grad_(True)

        return z.detach()


# ============================================================================
# SECTION 7: MOMENTUM SAMPLER  (§3.2, Eq. 5)
# — No changes needed; was already fully aligned with paper
# ============================================================================

class MomentumSampler:
    """
    Eq. 5: z^0_t ~ N(β·z*_{t-1} + (1-β)·μ, Σ)
    Dynamically shifts sampling distribution toward challenging regions.
    """
    def __init__(self, dim, momentum=0.5, std=0.1):
        self.dim      = dim
        self.momentum = momentum
        self.std      = std
        self.z_prev   = None

    def sample(self, B, device):
        base = torch.zeros(self.dim, device=device)   # μ = 0
        if self.z_prev is not None:
            # β·z*_{t-1} + (1-β)·μ
            base = (self.momentum * self.z_prev.squeeze(0).to(device)
                    + (1 - self.momentum) * base)
        return base.unsqueeze(0) + self.std * torch.randn(B, self.dim, device=device)

    def update(self, z_opt):
        self.z_prev = z_opt.mean(0, keepdim=True).detach().cpu()


# ============================================================================
# SECTION 8: GAN PRE-TRAINER  (§3.4, §4.3)
# ============================================================================

class GANPretrainer:
    """
    Trains the identity-preserving generation network.
    Fixes vs. previous version:
      - x_style ≠ x_id: different images per batch (paper Fig 3) [FIX 12]
      - λ_1 = 1.0 (paper Eq.7)                                   [FIX 13]
      - Linear LR decay schedule (paper §4.3)                    [FIX 14]
    """
    def __init__(self, gen_net, device, lr=1e-3, epochs=60, save_path=None):
        self.gen    = gen_net
        self.dev    = device
        self.epochs = epochs
        self.save   = save_path

        gen_params = (list(gen_net.style_encoder.parameters()) +
                      list(gen_net.generator.parameters()))
        self.opt_G = optim.Adam(gen_params, lr=lr, betas=(0.5, 0.99))
        self.opt_D = optim.Adam(gen_net.discriminator.parameters(),
                                lr=lr, betas=(0.5, 0.99))

        # [FIX 14] Linear LR decay: lr → 0 over `epochs` epochs
        self.sched_G = optim.lr_scheduler.LambdaLR(
            self.opt_G, lr_lambda=lambda ep: 1.0 - ep / epochs)
        self.sched_D = optim.lr_scheduler.LambdaLR(
            self.opt_D, lr_lambda=lambda ep: 1.0 - ep / epochs)

    def train_batch(self, x):
        """
        x: (B, 3, H, W) — one batch of real palmprints

        [FIX 12] Create x2 by shuffling batch indices so x_style ≠ x_id.
        This is essential for the disentanglement signal:
          - L_1  forces s to capture the style of x1
          - L_ID forces the output to have the identity of x2 (not x1)
          Together these decouple style from identity.
        """
        B  = x.size(0)
        x1 = x                                           # style source
        x2 = x[torch.randperm(B, device=x.device)]      # identity source [FIX 12]

        # ── Discriminator step ───────────────────────────────────────────
        gen, _, _, mu, logvar = self.gen(x1, x2)
        d_real = self.gen.discriminator(x)
        d_fake = self.gen.discriminator(gen.detach())
        # Hinge loss (paper uses GAN adversarial loss L_GAN)
        l_D = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
        self.opt_D.zero_grad()
        l_D.backward()
        self.opt_D.step()

        # ── Generator step ───────────────────────────────────────────────
        gen2, _, _, mu2, lv2 = self.gen(x1, x2)
        d_fake2 = self.gen.discriminator(gen2)

        l_adv = -d_fake2.mean()                          # L_GAN
        l_rec = F.l1_loss(gen2, x1)                      # L_1  (style recon)
        l_kl  = -0.5 * (1 + lv2 - mu2.pow(2) -
                        lv2.exp()).sum(1).mean()          # L_KL

        # L_ID: cosine similarity between identity features of generated and x2
        with torch.no_grad():
            fi_r = self.gen.identity_encoder(x2).mean([2, 3])   # x2 identity
        fi_g  = self.gen.identity_encoder(gen2).mean([2, 3])    # generated identity
        l_id  = 1.0 - F.cosine_similarity(fi_r, fi_g, dim=1).mean()

        # Eq. 7: L_GEN = λ_KL·L_KL + λ_GAN·L_GAN + λ_1·L_1 + λ_ID·L_ID
        # Paper: λ_KL=0.01, λ_GAN=1.0, λ_1=1.0, λ_ID=5.0      [FIX 13]
        l_G = l_adv * 1.0 + l_rec * 1.0 + l_kl * 0.01 + l_id * 5.0

        self.opt_G.zero_grad()
        l_G.backward()
        self.opt_G.step()

        return {'G': l_G.item(), 'D': l_D.item(),
                'rec': l_rec.item(), 'id': l_id.item()}

    def run(self, loader):
        print(f"\n{'='*80}")
        print(f"  GAN PRE-TRAINING ({self.epochs} epochs)")
        print(f"  Losses — G: total generator | D: discriminator hinge")
        print(f"           rec: L1 style recon | id: identity preservation")
        print(f"  Loss weights: λ_GAN=1.0, λ_1=1.0, λ_KL=0.01, λ_ID=5.0")
        print(f"{'='*80}")
        self.gen.train()

        for ep in range(self.epochs):
            totals = {}
            pbar   = tqdm(loader, desc=f"[GAN] Epoch {ep+1}/{self.epochs}")
            for batch in pbar:
                x      = batch['img'].to(self.dev)
                losses = self.train_batch(x)
                for k, v in losses.items():
                    totals[k] = totals.get(k, 0) + v
                pbar.set_postfix({k: f"{v:.3f}" for k, v in losses.items()})

            n   = len(loader)
            avg = {k: v / n for k, v in totals.items()}
            print(f"[GAN] Epoch {ep+1:3d}  " +
                  "  ".join(f"{k}={v:.4f}" for k, v in avg.items()))

            # [FIX 14] Step linear LR decay
            self.sched_G.step()
            self.sched_D.step()

        if self.save:
            os.makedirs(os.path.dirname(self.save), exist_ok=True)
            torch.save(self.gen.state_dict(), self.save)
            print(f"[GAN] Saved → {self.save}")


# ============================================================================
# SECTION 9: EVALUATION
# — No changes needed
# ============================================================================

class Evaluator:
    def __init__(self, rec_net, device):
        self.rec = rec_net
        self.dev = device

    @torch.no_grad()
    def extract(self, loader):
        feats, ids, specs = [], [], []
        self.rec.eval()
        for batch in tqdm(loader, desc="[Eval] Extracting features"):
            x = batch['img'].to(self.dev)
            f = self.rec.get_verification_features(x)
            feats.append(f.cpu().numpy())
            ids.extend(batch['identity'].numpy().tolist())
            specs.extend(batch['spectrum'])
        return (np.concatenate(feats),
                np.array(ids),
                np.array(specs))

    def verification(self, feats, ids, far_targets):
        print("[Eval] Computing verification metrics...")
        sim = feats @ feats.T
        gen_scores, imp_scores = [], []
        n = len(feats)
        for i in range(n):
            for j in range(i + 1, n):
                (gen_scores if ids[i] == ids[j] else imp_scores).append(float(sim[i, j]))
        gen_scores = np.array(gen_scores)
        imp_scores = np.array(imp_scores)
        print(f"       Genuine: {len(gen_scores):,}   Imposter: {len(imp_scores):,}")

        results    = {}
        imp_sorted = np.sort(imp_scores)[::-1]
        for far in far_targets:
            idx = int(len(imp_scores) * far)
            thr = imp_sorted[min(idx, len(imp_scores) - 1)]
            tar = float((gen_scores >= thr).mean())
            results[f"TAR@FAR={far:.0e}"] = tar

        all_s = np.concatenate([gen_scores, imp_scores])
        all_l = np.concatenate([np.ones(len(gen_scores)), np.zeros(len(imp_scores))])
        fpr, tpr, _ = roc_curve(all_l, all_s)
        diff        = np.abs(fpr - (1 - tpr))
        eer_idx     = np.argmin(diff)
        results["EER"] = float((fpr[eer_idx] + (1 - tpr[eer_idx])) / 2)
        return results, gen_scores, imp_scores

    def identification(self, feats, ids, specs):
        print("[Eval] Computing identification metrics (1:N)...")
        feats_n  = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        uid_list = np.unique(ids)
        g_f, g_id, p_f, p_id = [], [], [], []

        for uid in uid_list:
            idx = np.where(ids == uid)[0]
            if len(idx) < 2:
                continue
            gal_set = set()
            for spec in np.unique(specs[idx]):
                si = idx[specs[idx] == spec]
                gal_set.add(si[0])
                g_f.append(feats_n[si[0]])
                g_id.append(uid)
            for i in idx:
                if i not in gal_set:
                    p_f.append(feats_n[i])
                    p_id.append(uid)

        if not g_f or not p_f:
            print("[Eval] Warning: empty gallery or probe set.")
            return {f"Rank-{k}": 0.0 for k in [1, 5, 10]}

        G, Gid = np.array(g_f), np.array(g_id)
        P, Pid = np.array(p_f), np.array(p_id)
        print(f"       Gallery: {len(G)}   Probes: {len(P)}")
        sim    = P @ G.T
        ranked = Gid[np.argsort(-sim, axis=1)]
        results = {}
        for k in [1, 5, 10]:
            kk = min(k, len(Gid))
            results[f"Rank-{k}"] = float(
                np.any(ranked[:, :kk] == Pid[:, None], axis=1).mean())
        return results

    def evaluate(self, loader, far_targets):
        feats, ids, specs = self.extract(loader)
        ver, gen, imp     = self.verification(feats, ids, far_targets)
        idf               = self.identification(feats, ids, specs)
        return {**ver, **idf,
                'num_samples'   : len(feats),
                'num_identities': int(np.unique(ids).size),
                'num_genuine'   : len(gen),
                'num_imposter'  : len(imp)}

    def print_results(self, res):
        print("\n" + "=" * 80)
        print("  EVALUATION RESULTS")
        print("=" * 80)
        print(f"  Samples: {res['num_samples']}   "
              f"Identities: {res['num_identities']}")
        print(f"  Genuine pairs : {res['num_genuine']:,}")
        print(f"  Imposter pairs: {res['num_imposter']:,}")
        print("\n  Verification (TAR@FAR):")
        for k in sorted(res):
            if 'TAR' in k or k == 'EER':
                print(f"    {k:<28} {res[k]:.4f}")
        print("\n  Identification:")
        for k in sorted(res):
            if 'Rank' in k:
                print(f"    {k:<28} {res[k]:.4f}")
        print("=" * 80 + "\n")


# ============================================================================
# SECTION 10: LR SCHEDULE  (§4.3)
# ============================================================================

def build_lr_schedule(optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-5):
    """
    Linear warmup then cosine decay.
    Warmup protects pretrained ResNet50 weights from large gradient updates
    in early epochs — essential at small batch size (32 vs paper's 256).
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # linear ramp 0→1
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr / base_lr + (1 - min_lr / base_lr) * cosine
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# SECTION 11: MAIN TRAINER
# ============================================================================

class UAATrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Device: {self.dev}")

        (self.train_loader, self.test_loader,
         self.num_classes, self.test_samples,
         self.identity_map) = create_dataloaders(cfg)

        self.rec = PalmRecognitionNetwork(
            self.num_classes, cfg.feature_dim,
            s=cfg.arcface_s, m=cfg.arcface_m
        ).to(self.dev)

        self.aug = UnifiedAugmentationModule(
            style_dim=cfg.style_dim,
            img_size=cfg.img_size,
            use_generation=cfg.use_generation
        ).to(self.dev)

        # SGD + warmup + cosine decay (scale-adapted from paper §4.3)
        self.opt   = optim.SGD(self.rec.parameters(),
                               lr=cfg.lr, momentum=0.9,
                               weight_decay=cfg.weight_decay)
        self.sched = build_lr_schedule(
            self.opt, cfg.warmup_epochs, cfg.num_epochs, cfg.lr)

        self.pgd   = AdversarialAugOptimizer(
            self.aug, self.rec,
            pgd_steps=cfg.pgd_steps, step_size=cfg.pgd_step_size)
        self.s_geo = MomentumSampler(4,             momentum=cfg.momentum_geo)
        self.s_tex = MomentumSampler(cfg.style_dim, momentum=cfg.momentum_tex)

        self.writer   = SummaryWriter(
            f'runs/{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.step     = 0
        self.best_tar = 0.0

    # ── Phase 1: GAN pre-training ──────────────────────────────────────────
    def pretrain_gan(self):
        if not self.cfg.use_generation:
            print("[GAN] Skipping — use_generation=False")
            return

        if (self.cfg.gen_save_path and
                os.path.exists(self.cfg.gen_save_path)):
            print(f"[GAN] Loading pre-trained weights from {self.cfg.gen_save_path}")
            ckpt = torch.load(self.cfg.gen_save_path, map_location=self.dev)

            # Filter checkpoint: keep only keys that exist in the current model
            # AND whose tensor shape matches exactly.
            # strict=False alone still crashes on size mismatches (e.g. b1.conv
            # changed from 512→1024 input channels after id_feat injection fix).
            current_state = self.aug.gen_network.state_dict()
            filtered, skipped_missing, skipped_shape = {}, [], []
            for k, v in ckpt.items():
                if k not in current_state:
                    skipped_missing.append(k)
                elif v.shape != current_state[k].shape:
                    skipped_shape.append(
                        f"{k}: ckpt{list(v.shape)} vs model{list(current_state[k].shape)}")
                else:
                    filtered[k] = v

            loaded   = len(filtered)
            total    = len(current_state)
            print(f"[GAN]   Loaded   : {loaded}/{total} keys matched shape exactly")
            if skipped_shape:
                print(f"[GAN]   Skipped (shape mismatch, will re-init): "
                      f"{len(skipped_shape)} keys")
                for s in skipped_shape[:6]:
                    print(f"          {s}")
                if len(skipped_shape) > 6:
                    print(f"          ... and {len(skipped_shape)-6} more")
            if skipped_missing:
                print(f"[GAN]   Skipped (not in current model): "
                      f"{len(skipped_missing)} keys")

            # Load only the safe, shape-compatible subset
            missing, unexpected = self.aug.gen_network.load_state_dict(
                filtered, strict=False)

            needs_finetune = len(skipped_shape) > 0 or len(missing) > 0
            if needs_finetune:
                print(f"[GAN] Architecture mismatch detected — "
                      f"running {self.cfg.gan_finetune_epochs} fine-tune epochs "
                      f"to train re-initialised layers before freezing.")
                ft_trainer = GANPretrainer(
                    self.aug.gen_network, self.dev,
                    lr=self.cfg.gen_lr * 0.1,   # lower LR: pretrained layers stay stable
                    epochs=self.cfg.gan_finetune_epochs,
                    save_path=None)              # don't overwrite the original checkpoint
                ft_trainer.run(self.train_loader)
            else:
                print("[GAN]   All weights loaded successfully — no fine-tuning needed.")
        else:
            trainer = GANPretrainer(
                self.aug.gen_network, self.dev,
                lr=self.cfg.gen_lr,
                epochs=self.cfg.gen_pretrain_epochs,
                save_path=self.cfg.gen_save_path)
            trainer.run(self.train_loader)

        # Freeze generator & discriminator; style_encoder stays trainable for PGD
        self.aug.freeze_gan_weights()

    # ── Phase 2: recognition training epoch ───────────────────────────────
    def train_epoch(self, epoch):
        self.rec.train()
        if self.cfg.use_generation:
            # Generator & discriminator frozen; style_encoder stays trainable
            # so PGD gradients flow: z → style_encoder → generator → image
            self.aug.gen_network.generator.eval()
            self.aug.gen_network.discriminator.eval()
            self.aug.gen_network.style_encoder.train()
            self.aug.gen_network.identity_encoder.eval()

        total_loss = 0.0
        pbar = tqdm(self.train_loader,
                    desc=f"[Train] Epoch {epoch+1}/{self.cfg.num_epochs}")

        for batch in pbar:
            x      = batch['img'].to(self.dev)
            labels = batch['identity'].to(self.dev)
            B      = x.size(0)

            # Sample initial control vectors via momentum sampler (Eq. 5)
            z_geo  = self.s_geo.sample(B, self.dev)
            z_tex  = self.s_tex.sample(B, self.dev)
            z_init = torch.cat([z_geo, z_tex], dim=1)

            # Geometric PGD on whole batch
            if self.cfg.use_geometric:
                z_opt = self.pgd.optimize(x, labels, z_init, aug_type='geometric')
                self.s_geo.update(z_opt[:, :4])
            else:
                z_opt = z_init

            # Textural PGD on whole batch (sequential on same z)
            if self.cfg.use_textural and self.cfg.use_generation:
                z_opt2 = self.pgd.optimize(x, labels, z_opt, aug_type='textural')
                self.s_tex.update(z_opt2[:, 4:])
                z_final = z_opt2
            else:
                z_final = z_opt

            # Apply augmentation — no grad needed here
            with torch.no_grad():
                x_aug = self.aug(x, z_final, aug_type='both')

            # Eq. 4: train on original + augmented (whole batch each)
            x_all  = torch.cat([x, x_aug], dim=0)
            lb_all = torch.cat([labels, labels], dim=0)

            _, feats = self.rec(x_all)
            loss     = self.rec.compute_loss(feats, lb_all)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.rec.parameters(), self.cfg.grad_clip)
            self.opt.step()

            total_loss += loss.item()
            self.step  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{self.opt.param_groups[0]['lr']:.6f}")

            if self.step % 100 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.step)

        self.sched.step()
        avg = total_loss / len(self.train_loader)
        print(f"[Epoch {epoch+1:3d}] Loss: {avg:.4f}  "
              f"LR: {self.opt.param_groups[0]['lr']:.6f}")
        return avg

    def validate(self, epoch):
        ev  = Evaluator(self.rec, self.dev)
        res = ev.evaluate(self.test_loader, self.cfg.tar_far_values)
        ev.print_results(res)

        for k, v in res.items():
            if isinstance(v, float):
                self.writer.add_scalar(f'val/{k}', v, epoch)

        tar_key = 'TAR@FAR=1e-05'
        tar     = res.get(tar_key, 0.0)
        if tar > self.best_tar:
            self.best_tar = tar
            self.save_checkpoint(epoch, best=True)
            print(f"  ★ New best {tar_key}: {self.best_tar:.4f}")

        return res

    def save_checkpoint(self, epoch, best=False):
        os.makedirs('checkpoints', exist_ok=True)
        state = {
            'epoch'     : epoch,
            'rec_net'   : self.rec.state_dict(),
            'aug_module': self.aug.state_dict(),
            'optimizer' : self.opt.state_dict(),
            'best_tar'  : self.best_tar,
        }
        tag  = '_best' if best else f'_ep{epoch+1}'
        path = f'checkpoints/uaa{tag}.pt'
        torch.save(state, path)
        print(f"[Save] {path}")

    def train(self):
        print(f"\n{'='*80}")
        print("  PHASE 1 — GAN PRE-TRAINING  (paper §3.4, §4.3)")
        print(f"{'='*80}")
        self.pretrain_gan()

        print(f"\n{'='*80}")
        print("  PHASE 2 — RECOGNITION TRAINING WITH UAA  (paper §3.2, §4.3)")
        print(f"{'='*80}")

        for epoch in range(self.cfg.num_epochs):
            self.train_epoch(epoch)

            if ((epoch + 1) % self.cfg.eval_freq == 0
                    or epoch == self.cfg.num_epochs - 1):
                self.validate(epoch)

            if (epoch + 1) % self.cfg.save_freq == 0:
                self.save_checkpoint(epoch, best=False)

        print(f"\n[Done] Best TAR@FAR=1e-5: {self.best_tar:.4f}")
        self.writer.close()

        print("\n[Final] Comprehensive evaluation...")
        ev  = Evaluator(self.rec, self.dev)
        res = ev.evaluate(self.test_loader, self.cfg.tar_far_values)
        ev.print_results(res)
        return res


# ============================================================================
# SECTION 12: INFERENCE HELPER
# ============================================================================

class PalmInference:
    def __init__(self, ckpt_path, num_classes, feature_dim=512, device='cuda'):
        self.dev = torch.device(device)
        self.rec = PalmRecognitionNetwork(num_classes, feature_dim).to(self.dev)
        ck = torch.load(ckpt_path, map_location=self.dev)
        self.rec.load_state_dict(ck['rec_net'])
        self.rec.eval()
        self.tf = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def embed(self, path):
        img = self.tf(Image.open(path).convert('RGB')).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            return self.rec.get_verification_features(img)[0].cpu().numpy()

    def verify(self, p1, p2, thr=0.5):
        f1, f2 = self.embed(p1), self.embed(p2)
        score  = float(np.dot(f1, f2))
        return score >= thr, score


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("  UAA PALMPRINT RECOGNITION — PAPER ALIGNED IMPLEMENTATION")
    print("  Jin et al., ICCV 2025")
    print("=" * 80 + "\n")

    trainer = UAATrainer(args)
    results = trainer.train()

    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE")
    print("=" * 80)
    print("  Checkpoints : checkpoints/")
    print("  Best model  : checkpoints/uaa_best.pt")
    print("  TensorBoard : tensorboard --logdir=runs")
    print("\n  Final Metrics:")
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"    {k:<32} {v:.4f}")
        else:
            print(f"    {k:<32} {v}")
    print("=" * 80 + "\n")
