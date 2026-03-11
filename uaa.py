"""
UNIFIED ADVERSARIAL AUGMENTATION (UAA) FOR PALMPRINT RECOGNITION
Complete All-in-One Implementation
ICCV 2025 Paper Implementation

All components combined into a single self-contained script.
Author: Claude (Implementation)
Based on: "Unified Adversarial Augmentation for Improving Palmprint Recognition"
"""

# ============================================================================
# ⚙️ CONFIGURATION - SET ALL PARAMETERS HERE AT THE TOP
# ============================================================================

CONFIG = {
    # ========== Dataset Configuration ==========
    'data_path': '/home/pai-ng/Jamal/CASIA-MS-ROI',  # Path to CASIA-MS dataset
    'train_ratio': 0.7,                              # Proportion of identities for training
    'random_seed': 42,                               # Random seed for reproducibility

    # ========== Data Loading ==========
    'batch_size': 32,                                 # Reduce if OOM (try 16, 8)
    'num_workers': 4,                                 # Number of data loading workers
    'img_size': 112,                                  # Input image resolution

    # ========== Model Architecture ==========
    'feature_dim': 512,                               # Recognition feature dimension
    'style_dim': 16,                                  # Style code dimension (8, 16, 32)

    # ========== Training ==========
    'num_epochs': 30,                                 # Number of training epochs
    'lr': 0.1,                                        # Learning rate
    'save_freq': 5,                                   # Save checkpoint every N epochs

    # ========== Augmentation Module ==========
    'use_geometric': True,                            # Enable spatial/geometric augmentation
    'use_generation': False,                          # Enable GAN-based generation
    'use_textural': False,                            # Enable textural augmentation (requires use_generation=True)
    'geometric_rate': 0.5,                            # Proportion of samples to augment geometrically
    'textural_rate': 0.5,                             # Proportion of samples to augment texturally

    # ========== Generation Network (if use_generation=True) ==========
    'gen_pretrain_epochs': 10,                        # GAN pretraining epochs
    'gen_lr': 1e-3,                                   # Generator learning rate

    # ========== Adversarial Augmentation ==========
    'pgd_steps': 1,                                   # Number of PGD optimization steps (1-3, more = harder samples, slower)
    'pgd_step_size': 0.01,                            # Step size for PGD updates

    # ========== Momentum Sampling ==========
    'momentum_geo': 0.5,                              # Momentum for geometric sampling
    'momentum_tex': 0.25,                             # Momentum for textural sampling

    # ========== Evaluation ==========
    'tar_far_values': [1e-5, 1e-4, 1e-3, 1e-2],      # FAR thresholds for TAR@FAR computation
}

# ============================================================================
# 📋 CONFIGURATION PRESETS (Uncomment one below to override CONFIG above)
# ============================================================================

"""
# PRESET 1: Quick Test (Validate everything works, ~2-5 min on GPU)
CONFIG.update({
    'batch_size': 8,
    'num_epochs': 1,
    'use_geometric': True,
    'use_generation': False,
})

# PRESET 2: Geometric Only - FAST (Spatial augmentation only, ~30 min on GPU)
CONFIG.update({
    'batch_size': 32,
    'num_epochs': 30,
    'use_geometric': True,
    'use_generation': False,
})

# PRESET 3: Full UAA - COMPREHENSIVE (Geometric + Textural, ~2-3 hours on GPU)
CONFIG.update({
    'batch_size': 32,
    'num_epochs': 30,
    'use_generation': True,
    'use_geometric': True,
    'use_textural': True,
    'gen_pretrain_epochs': 10,
})

# PRESET 4: Advanced - BEST RESULTS (More iterations, ~5-6 hours on GPU)
CONFIG.update({
    'batch_size': 32,
    'num_epochs': 50,
    'use_generation': True,
    'use_geometric': True,
    'use_textural': True,
    'gen_pretrain_epochs': 30,
    'pgd_steps': 2,
    'pgd_step_size': 0.05,
    'style_dim': 32,
    'geometric_rate': 0.7,
    'textural_rate': 0.7,
})

# PRESET 5: Low Memory (For machines with limited GPU memory)
CONFIG.update({
    'batch_size': 8,
    'num_epochs': 30,
    'use_geometric': True,
    'use_generation': False,
    'num_workers': 2,
})

# PRESET 6: CPU Only (Much slower, but works without GPU)
CONFIG.update({
    'batch_size': 4,
    'num_epochs': 5,
    'use_geometric': True,
    'use_generation': False,
    'num_workers': 0,
})
"""

# ============================================================================
# Import Required Libraries
# ============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
from datetime import datetime
import math
from sklearn.metrics import roc_curve

# ============================================================================
# Convert CONFIG to args object for code compatibility
# ============================================================================
args = argparse.Namespace(**CONFIG)
print("=" * 80)
print("⚙️  CONFIGURATION LOADED:")
print("=" * 80)
for key, value in CONFIG.items():
    print(f"  {key:<30} = {value}")
print("=" * 80 + "\n")


# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

def load_all_samples(data_path):
    """Load all samples from dataset"""
    samples = []

    for root, _, files in os.walk(data_path):
        files.sort()
        for fname in files:
            if not fname.lower().endswith(".jpg"):
                continue
            parts = fname[:-4].split("_")
            if len(parts) != 4:
                continue
            subject_id, hand, spectrum, iteration = parts
            img_path = os.path.join(root, fname)
            hand_id = f"{subject_id}_{hand}"

            samples.append({
                'path': img_path,
                'subject': subject_id,
                'hand': hand,
                'spectrum': spectrum,
                'iteration': iteration,
                'hand_id': hand_id
            })

    return samples


def build_identity_map(samples):
    """Build identity label mapping from all samples"""
    all_hand_ids = sorted(set(s['hand_id'] for s in samples))
    identity_map = {h: i for i, h in enumerate(all_hand_ids)}
    num_classes = len(identity_map)
    print(f"[Data] Total identities: {num_classes}")
    return identity_map, num_classes


def split_train_test(samples, identity_map, train_ratio=0.7, seed=42):
    """
    Split data into train and test sets.
    Ensures identities are not mixed between train and test.
    """
    np.random.seed(seed)

    identity_samples = {}
    for sample in samples:
        hand_id = sample['hand_id']
        if hand_id not in identity_samples:
            identity_samples[hand_id] = []
        identity_samples[hand_id].append(sample)

    all_identities = list(identity_samples.keys())
    num_train = int(len(all_identities) * train_ratio)

    np.random.shuffle(all_identities)
    train_identities = set(all_identities[:num_train])
    test_identities = set(all_identities[num_train:])

    train_samples = []
    test_samples = []

    for sample in samples:
        if sample['hand_id'] in train_identities:
            train_samples.append(sample)
        else:
            test_samples.append(sample)

    print(f"[Data] Train identities: {len(train_identities)}, Train samples: {len(train_samples)}")
    print(f"[Data] Test identities: {len(test_identities)}, Test samples: {len(test_samples)}")

    return train_samples, test_samples, train_identities, test_identities


class PalmDataset(Dataset):
    """CASIA-MS Palmprint Dataset"""

    def __init__(self, samples, identity_map, img_size=112, is_test=False):
        self.samples = samples
        self.identity_map = identity_map
        self.img_size = img_size
        self.is_test = is_test

        for sample in self.samples:
            sample['identity'] = identity_map[sample['hand_id']]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['path']).convert("RGB")
        img_tensor = self.transform(img)

        return {
            'img': img_tensor,
            'identity': sample['identity'],
            'path': sample['path'],
            'hand_id': sample['hand_id'],
            'spectrum': sample['spectrum'],
            'subject': sample['subject'],
            'hand': sample['hand']
        }


def create_dataloaders(data_path, batch_size=32, num_workers=4, img_size=112,
                       train_ratio=0.7, seed=42):
    all_samples = load_all_samples(data_path)
    print(f"[Data] Total samples loaded: {len(all_samples)}")

    identity_map, num_classes = build_identity_map(all_samples)

    train_samples, test_samples, train_ids, test_ids = split_train_test(
        all_samples, identity_map, train_ratio, seed
    )

    train_dataset = PalmDataset(train_samples, identity_map, img_size, is_test=False)
    test_dataset = PalmDataset(test_samples, identity_map, img_size, is_test=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    return train_loader, test_loader, num_classes, test_samples


# ============================================================================
# SECTION 2: SPATIAL TRANSFORMER MODULE
# ============================================================================

class SpatialTransformer(nn.Module):
    """Differentiable Spatial Transformer for geometric augmentation"""

    def __init__(self, img_size=112):
        super(SpatialTransformer, self).__init__()
        self.img_size = img_size

    def forward(self, x, transform_params):
        """Apply geometric transformation to images — no in-place ops"""
        batch_size = x.size(0)
        device = x.device

        tx      = transform_params[:, 0]
        ty      = transform_params[:, 1]
        ttheta  = transform_params[:, 2]
        ts      = transform_params[:, 3]

        cos_theta = torch.cos(ttheta)
        sin_theta = torch.sin(ttheta)

        # Build affine matrix using stack (avoids in-place assignment)
        row1 = torch.stack([ ts * cos_theta, -ts * sin_theta, tx], dim=1)  # (B, 3)
        row2 = torch.stack([ ts * sin_theta,  ts * cos_theta, ty], dim=1)  # (B, 3)
        affine_matrix = torch.stack([row1, row2], dim=1)                    # (B, 2, 3)

        grid = F.affine_grid(affine_matrix, x.size(), align_corners=False)
        transformed = F.grid_sample(x, grid, align_corners=False,
                                    padding_mode='reflection')
        return transformed

    def get_constraint_params(self, unconstrained_params):
        """Apply constraints to transformation parameters — no in-place ops"""
        tx      = torch.clamp(unconstrained_params[:, 0:1], -0.2,  0.2)
        ty      = torch.clamp(unconstrained_params[:, 1:2], -0.2,  0.2)
        ttheta  = torch.clamp(unconstrained_params[:, 2:3], -0.25, 0.25)
        ts_raw  = unconstrained_params[:, 3:4]
        ts      = 1.0 + torch.clamp(ts_raw - 1.0, -0.2, 0.2)
        return torch.cat([tx, ty, ttheta, ts], dim=1)


# ============================================================================
# SECTION 3: IDENTITY-PRESERVING GENERATION NETWORK
# ============================================================================

class StyleEncoder(nn.Module):
    """Style encoder to extract style code from palmprint images"""

    def __init__(self, style_dim=16):
        super(StyleEncoder, self).__init__()
        self.style_dim = style_dim

        from torchvision.models import resnet50
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_mu = nn.Linear(2048, style_dim)
        self.fc_logvar = nn.Linear(2048, style_dim)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        style_code = mu + eps * std
        return mu, logvar, style_code


class IdentityEncoder(nn.Module):
    """Identity encoder (frozen pre-trained network)"""

    def __init__(self):
        super(IdentityEncoder, self).__init__()
        from torchvision.models import resnet50
        resnet = resnet50(pretrained=True)
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer4 = nn.Sequential(*list(resnet.children())[7:8])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_high = self.layer4(x)
        return feat_high


class AdaptiveInstanceNorm(nn.Module):
    """Adaptive Instance Normalization (AdaIN)"""

    def __init__(self, num_features, style_dim):
        super(AdaptiveInstanceNorm, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features)
        self.fc_gamma = nn.Linear(style_dim, num_features)
        self.fc_beta = nn.Linear(style_dim, num_features)

    def forward(self, x, style_code):
        normalized = self.instance_norm(x)
        gamma = self.fc_gamma(style_code).unsqueeze(-1).unsqueeze(-1)
        beta = self.fc_beta(style_code).unsqueeze(-1).unsqueeze(-1)
        return gamma * normalized + beta


class AdaINBlock(nn.Module):
    """Adaptive Instance Normalization Block"""

    def __init__(self, in_channels, out_channels, style_dim):
        super(AdaINBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.adain = AdaptiveInstanceNorm(out_channels, style_dim)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, style_code):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.adain(x, style_code)
        x = self.relu(x)
        return x


class PalmGenerator(nn.Module):
    """Generator network for palmprint synthesis"""

    def __init__(self, style_dim=16, img_size=112):
        super(PalmGenerator, self).__init__()
        self.style_dim = style_dim
        self.img_size = img_size
        self.fc = nn.Linear(style_dim, 512 * 7 * 7)
        self.block1 = AdaINBlock(512, 256, style_dim)
        self.block2 = AdaINBlock(256, 128, style_dim)
        self.block3 = AdaINBlock(128, 64, style_dim)
        self.block4 = AdaINBlock(64, 32, style_dim)
        self.to_rgb = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, style_code, identity_feat):
        batch_size = style_code.size(0)
        x = self.fc(style_code)
        x = x.view(batch_size, 512, 7, 7)
        x = self.block1(x, style_code)
        x = self.block2(x, style_code)
        x = self.block3(x, style_code)
        x = self.block4(x, style_code)
        img = self.to_rgb(x)
        img = torch.tanh(img)
        return img


class PalmDiscriminator(nn.Module):
    """Discriminator network for palmprint generation"""

    def __init__(self):
        super(PalmDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.main(x).view(x.size(0), -1)


class PalmGenerationNetwork(nn.Module):
    """Complete identity-preserving palmprint generation network"""

    def __init__(self, style_dim=16, img_size=112):
        super(PalmGenerationNetwork, self).__init__()
        self.style_encoder = StyleEncoder(style_dim=style_dim)
        self.identity_encoder = IdentityEncoder()
        self.generator = PalmGenerator(style_dim=style_dim, img_size=img_size)
        self.discriminator = PalmDiscriminator()

    def forward(self, x_style, x_identity=None):
        if x_identity is None:
            x_identity = x_style
        mu, logvar, style_code = self.style_encoder(x_style)
        identity_feat = self.identity_encoder(x_identity)
        generated = self.generator(style_code, identity_feat)
        return generated, style_code, identity_feat, mu, logvar


# ============================================================================
# SECTION 4: RECOGNITION NETWORK (ARCFACE)
# ============================================================================

class ArcFaceHead(nn.Module):
    """ArcFace Head for palmprint recognition"""

    def __init__(self, in_features=512, num_classes=1000, s=64.0, m=0.5):
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        self.register_parameter('bias', None)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        W = F.normalize(self.weight, dim=1)
        logits = F.linear(x, W)
        return logits


class ArcFaceLoss(nn.Module):
    """ArcFace Loss function"""

    def __init__(self, num_classes=1000, s=64.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m
        self.threshold = math.cos(math.pi - m)

    def forward(self, logits, labels):
        logits = torch.clamp(logits, -1.0, 1.0)
        theta = torch.acos(logits)
        target_logit = torch.cos(theta + self.m)
        # Use scatter instead of in-place index assignment
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        output = one_hot * target_logit + (1.0 - one_hot) * logits
        output = output * self.s
        loss = F.cross_entropy(output, labels)
        return loss


class PalmRecognitionNetwork(nn.Module):
    """Palmprint Recognition Network with ResNet-50 and ArcFace"""

    def __init__(self, num_classes, feature_dim=512, input_size=112):
        super(PalmRecognitionNetwork, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        from torchvision.models import resnet50
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_bn = nn.BatchNorm1d(2048)
        self.feature_fc = nn.Linear(2048, feature_dim)
        self.feature_bn2 = nn.BatchNorm1d(feature_dim)
        self.arcface_head = ArcFaceHead(in_features=feature_dim, num_classes=num_classes, s=64.0, m=0.5)
        self.arcface_loss = ArcFaceLoss(num_classes=num_classes, s=64.0, m=0.5)

    def extract_features(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.feature_bn(x)
        x = self.feature_fc(x)
        x = self.feature_bn2(x)
        return x

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.arcface_head(features)
        return logits, features

    def compute_loss(self, logits, labels):
        return self.arcface_loss(logits, labels)

    def get_verification_features(self, x):
        features = self.extract_features(x)
        features = F.normalize(features, dim=1)
        return features


# ============================================================================
# SECTION 5: AUGMENTATION MODULE (UAA)
# ============================================================================

class UnifiedAugmentationModule(nn.Module):
    """Unified augmentation module combining spatial and generative components"""

    def __init__(self, style_dim=16, img_size=112, use_generation=True):
        super(UnifiedAugmentationModule, self).__init__()
        self.style_dim = style_dim
        self.img_size = img_size
        self.use_generation = use_generation
        self.control_dim = 4 + style_dim
        self.spatial_transformer = SpatialTransformer(img_size=img_size)

        if use_generation:
            self.generation_network = PalmGenerationNetwork(style_dim=style_dim, img_size=img_size)

    def forward(self, x, control_vector, augmentation_type='both'):
        spatial_params = control_vector[:, :4]
        style_code = control_vector[:, 4:]

        if augmentation_type in ('geometric', 'both'):
            spatial_params = self.spatial_transformer.get_constraint_params(spatial_params)
            x = self.spatial_transformer(x, spatial_params)

        if augmentation_type in ('textural', 'both'):
            if self.use_generation:
                x_augmented, _, _, _, _ = self.generation_network(x, x)
                alpha = 0.5
                x = alpha * x_augmented + (1 - alpha) * x

        return x

    def get_frozen_params(self):
        params = []
        if self.use_generation:
            params.extend(self.generation_network.parameters())
        return params


class AdversarialAugmentationOptimizer(nn.Module):
    """Optimize control vector to generate challenging samples"""

    def __init__(self, augmentation_module, recognition_network,
                 pgd_steps=1, pgd_step_size=0.01, control_dim=20):
        super(AdversarialAugmentationOptimizer, self).__init__()
        self.augmentation_module = augmentation_module
        self.recognition_network = recognition_network
        self.pgd_steps = pgd_steps
        self.pgd_step_size = pgd_step_size
        self.control_dim = control_dim
        self.perturbation_bounds = {'spatial': 0.3, 'style': 1.0}

    def optimize_control_vector(self, x, labels, z_init, augmentation_type='both'):
        # Detach input so we don't backprop through the data pipeline
        x_detached = x.detach()

        # Start from a fresh leaf tensor
        z = z_init.clone().detach().requires_grad_(True)

        for step in range(self.pgd_steps):
            # Zero grad if it exists
            if z.grad is not None:
                z.grad.zero_()

            # Forward pass
            x_augmented = self.augmentation_module(x_detached, z,
                                                   augmentation_type=augmentation_type)
            logits, features = self.recognition_network(x_augmented)
            loss = self.recognition_network.compute_loss(logits, labels)

            # Backward pass
            loss.backward()

            # PGD update — all out-of-place
            with torch.no_grad():
                grad_sign = torch.sign(z.grad)
                z_new = z.data + self.pgd_step_size * grad_sign
                z_new = self._project_onto_bounds(z_new)

            # Fresh leaf tensor for next iteration
            z = z_new.detach().requires_grad_(True)

        return z.detach()

    def _project_onto_bounds(self, z):
        z_spatial = torch.clamp(z[:, :4],
                                -self.perturbation_bounds['spatial'],
                                 self.perturbation_bounds['spatial'])
        z_style   = torch.clamp(z[:, 4:],
                                -self.perturbation_bounds['style'],
                                 self.perturbation_bounds['style'])
        return torch.cat([z_spatial, z_style], dim=1)


class MomentumDynamicSampler(nn.Module):
    """Momentum-based dynamic sampling strategy"""

    def __init__(self, control_dim, momentum=0.5):
        super(MomentumDynamicSampler, self).__init__()
        self.control_dim = control_dim
        self.momentum = momentum
        self.register_buffer('mean', torch.zeros(control_dim))
        self.register_buffer('std', torch.ones(control_dim) * 0.1)
        self.z_prev = None

    def sample(self, batch_size, device):
        if self.z_prev is None:
            z = (torch.randn(batch_size, self.control_dim, device=device)
                 * self.std.to(device) + self.mean.to(device))
        else:
            mean_dynamic = (self.momentum * self.z_prev
                            + (1 - self.momentum) * self.mean.to(device))
            z = (torch.randn(batch_size, self.control_dim, device=device)
                 * self.std.to(device) + mean_dynamic)
        return z

    def update_prev(self, z_opt):
        self.z_prev = z_opt.mean(dim=0, keepdim=True).detach()


class AugmentationRateController(nn.Module):
    """Controls proportion of samples to augment"""

    def __init__(self, geometric_rate=0.5, textural_rate=0.5):
        super(AugmentationRateController, self).__init__()
        self.geometric_rate = geometric_rate
        self.textural_rate = textural_rate

    def get_augmentation_mask(self, batch_size, augmentation_type='both', device='cpu'):
        if augmentation_type == 'geometric':
            num_to_augment = int(batch_size * self.geometric_rate)
        elif augmentation_type == 'textural':
            num_to_augment = int(batch_size * self.textural_rate)
        else:
            num_to_augment = int(batch_size * max(self.geometric_rate, self.textural_rate))

        mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        indices = torch.randperm(batch_size, device=device)[:num_to_augment]
        mask[indices] = True
        return mask


# ============================================================================
# SECTION 6: EVALUATION METRICS
# ============================================================================

class PalmRecognitionEvaluator:
    """Comprehensive evaluator for palmprint recognition"""

    def __init__(self, recognition_network, device):
        self.recognition_network = recognition_network
        self.device = device
        self.recognition_network.eval()

    def extract_features(self, dataloader):
        features_list = []
        identities_list = []
        paths_list = []
        spectrums_list = []
        hand_ids_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                x = batch['img'].to(self.device)
                identities = batch['identity'].cpu().numpy()
                paths = batch['path']
                spectrums = batch['spectrum']
                hand_ids = batch['hand_id']

                features = self.recognition_network.get_verification_features(x)

                features_list.append(features.cpu().numpy())
                identities_list.extend(identities)
                paths_list.extend(paths)
                spectrums_list.extend(spectrums)
                hand_ids_list.extend(hand_ids)

        features = np.concatenate(features_list, axis=0)
        identities = np.array(identities_list)

        return features, identities, paths_list, spectrums_list, hand_ids_list

    def compute_verification_metrics(self, features, identities,
                                     tar_far_values=[1e-5, 1e-4, 1e-3, 1e-2]):
        print("[Eval] Computing verification metrics...")

        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        similarities = features_norm @ features_norm.T

        genuine_scores = []
        imposter_scores = []

        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                score = similarities[i, j]
                if identities[i] == identities[j]:
                    genuine_scores.append(score)
                else:
                    imposter_scores.append(score)

        genuine_scores = np.array(genuine_scores)
        imposter_scores = np.array(imposter_scores)

        print(f"  Genuine pairs: {len(genuine_scores)}")
        print(f"  Imposter pairs: {len(imposter_scores)}")

        results = {}

        for far_threshold in tar_far_values:
            num_imposter = len(imposter_scores)
            num_false_accepts = int(num_imposter * far_threshold)

            sorted_imposters = np.sort(imposter_scores)[::-1]

            if num_false_accepts < num_imposter:
                decision_threshold = sorted_imposters[num_false_accepts]
            else:
                decision_threshold = -1.0

            num_genuine = len(genuine_scores)
            num_genuine_accepts = np.sum(genuine_scores >= decision_threshold)
            tar = num_genuine_accepts / num_genuine if num_genuine > 0 else 0.0

            results[f'TAR@FAR={far_threshold:.0e}'] = tar

        all_scores = np.concatenate([genuine_scores, imposter_scores])
        all_labels = np.concatenate([np.ones_like(genuine_scores),
                                     np.zeros_like(imposter_scores)])

        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        eer = np.min(np.abs(fpr + (1 - tpr)))
        results['EER'] = eer

        return results, genuine_scores, imposter_scores

    def compute_identification_metrics(self, features, identities, spectrums, hand_ids):
        print("[Eval] Computing identification metrics...")

        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

        unique_ids = np.unique(identities)

        gallery_features = []
        gallery_identities = []

        probe_features = []
        probe_identities = []

        for uid in unique_ids:
            indices = np.where(identities == uid)[0]

            if len(indices) < 2:
                continue

            spectrum_indices = {}
            for idx in indices:
                spec = spectrums[idx]
                if spec not in spectrum_indices:
                    spectrum_indices[spec] = []
                spectrum_indices[spec].append(idx)

            gallery_selected = set()
            for spec, spec_idxs in spectrum_indices.items():
                gallery_idx = spec_idxs[0]
                gallery_selected.add(gallery_idx)
                gallery_features.append(features_norm[gallery_idx])
                gallery_identities.append(uid)

            for idx in indices:
                if idx not in gallery_selected:
                    probe_features.append(features_norm[idx])
                    probe_identities.append(uid)

        if len(gallery_features) == 0 or len(probe_features) == 0:
            print("[Eval] Warning: No valid gallery or probe samples")
            return {f'Rank-{k}': 0.0 for k in [1, 5, 10]}

        gallery_features = np.array(gallery_features)
        gallery_identities = np.array(gallery_identities)
        probe_features = np.array(probe_features)
        probe_identities = np.array(probe_identities)

        print(f"  Gallery size: {len(gallery_features)}")
        print(f"  Probe size: {len(probe_features)}")

        similarities = probe_features @ gallery_features.T
        ranked_indices = np.argsort(-similarities, axis=1)
        ranked_gallery_ids = gallery_identities[ranked_indices]

        results = {}
        for k in [1, 5, 10]:
            if k > len(gallery_identities):
                results[f'Rank-{k}'] = 0.0
                continue
            matches = ranked_gallery_ids[:, :k] == probe_identities.reshape(-1, 1)
            results[f'Rank-{k}'] = np.any(matches, axis=1).mean()

        return results

    def evaluate(self, test_loader, test_samples=None,
                 tar_far_values=[1e-5, 1e-4, 1e-3, 1e-2]):
        print("\n" + "=" * 80)
        print("EVALUATION PHASE")
        print("=" * 80 + "\n")

        features, identities, paths, spectrums, hand_ids = self.extract_features(test_loader)

        ver_results, genuine_scores, imposter_scores = self.compute_verification_metrics(
            features, identities, tar_far_values
        )

        id_results = self.compute_identification_metrics(
            features, identities, spectrums, hand_ids
        )

        all_results = {
            **ver_results,
            **id_results,
            'num_samples': len(features),
            'num_identities': len(np.unique(identities)),
            'num_genuine_pairs': len(genuine_scores),
            'num_imposter_pairs': len(imposter_scores),
        }

        return all_results

    def print_results(self, results):
        print("\n" + "=" * 80)
        print("PALMPRINT RECOGNITION EVALUATION RESULTS")
        print("=" * 80)

        print(f"\nDataset Statistics:")
        print(f"  Total samples:    {results.get('num_samples', 'N/A')}")
        print(f"  Total identities: {results.get('num_identities', 'N/A')}")
        print(f"  Genuine pairs:    {results.get('num_genuine_pairs', 'N/A')}")
        print(f"  Imposter pairs:   {results.get('num_imposter_pairs', 'N/A')}")

        print(f"\nVerification Metrics (TAR@FAR):")
        for key in sorted(results.keys()):
            if 'TAR@FAR' in key:
                print(f"  {key}: {results[key]:.4f}")

        if 'EER' in results:
            print(f"  EER: {results['EER']:.4f}")

        print(f"\nIdentification Metrics (1:N Matching):")
        for key in sorted(results.keys()):
            if 'Rank-' in key:
                print(f"  {key} Accuracy: {results[key]:.4f}")

        print("\n" + "=" * 80 + "\n")


# ============================================================================
# SECTION 7: TRAINING
# ============================================================================

class GenerationNetworkTrainer:
    """Pre-train the generation network"""

    def __init__(self, model, device, lr=1e-3, num_epochs=60):
        self.model = model
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.optimizer_gen = optim.Adam(
            list(model.style_encoder.parameters()) + list(model.generator.parameters()),
            lr=lr, betas=(0.5, 0.99)
        )
        self.optimizer_disc = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.99))

    def compute_losses(self, x_real, x_style, generated, disc_real, disc_fake, mu, logvar):
        loss_gan = (torch.nn.ReLU()(1.0 - disc_real).mean()
                    + torch.nn.ReLU()(1.0 + disc_fake).mean())
        loss_l1 = torch.abs(generated - x_style).mean()
        loss_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        with torch.no_grad():
            identity_feat_real = self.model.identity_encoder(x_real)
        identity_feat_gen = self.model.identity_encoder(generated)
        loss_id = 1.0 - F.cosine_similarity(
            identity_feat_real.mean(dim=[2, 3]),
            identity_feat_gen.mean(dim=[2, 3]), dim=1
        ).mean()

        return {'gan': loss_gan, 'l1': loss_l1, 'kl': loss_kl, 'id': loss_id}

    def train_on_batch(self, x_style, x_identity):
        generated, style_code, identity_feat, mu, logvar = self.model(x_style, x_identity)
        disc_real = self.model.discriminator(x_identity)
        disc_fake = self.model.discriminator(generated.detach())
        losses = self.compute_losses(x_identity, x_style, generated,
                                     disc_real, disc_fake, mu, logvar)

        self.optimizer_disc.zero_grad()
        loss_disc = losses['gan']
        loss_disc.backward()
        self.optimizer_disc.step()

        generated = self.model.generator(style_code, identity_feat)
        disc_fake = self.model.discriminator(generated)
        losses = self.compute_losses(x_identity, x_style, generated,
                                     disc_real, disc_fake, mu, logvar)

        loss_gen = (losses['gan'] * 1.0 + losses['l1'] * 1.0
                    + losses['kl'] * 0.01 + losses['id'] * 5.0)
        self.optimizer_gen.zero_grad()
        loss_gen.backward()
        self.optimizer_gen.step()

        return {
            'loss_gen': loss_gen.item(),
            'loss_disc': loss_disc.item(),
            **{f'loss_{k}': v.item() for k, v in losses.items()}
        }


class UAATrainer:
    """Main UAA training framework"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Using device: {self.device}")

        self.train_loader, self.test_loader, self.num_classes, self.test_samples = \
            create_dataloaders(
                data_path=args.data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                img_size=args.img_size,
                train_ratio=args.train_ratio,
                seed=args.random_seed
            )

        self._create_networks()
        self._create_optimizers()
        self._create_augmentation_components()
        self.writer = SummaryWriter(f'runs/{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.global_step = 0

    def _create_networks(self):
        self.recognition_net = PalmRecognitionNetwork(
            num_classes=self.num_classes,
            feature_dim=self.args.feature_dim,
            input_size=self.args.img_size
        ).to(self.device)

        self.augmentation_module = UnifiedAugmentationModule(
            style_dim=self.args.style_dim,
            img_size=self.args.img_size,
            use_generation=self.args.use_generation
        ).to(self.device)

        if self.args.use_generation:
            self.generation_network = self.augmentation_module.generation_network

        print(f"[Networks] Recognition network created with {self.num_classes} classes")

    def _create_optimizers(self):
        self.optimizer_rec = optim.SGD(
            self.recognition_net.parameters(),
            lr=self.args.lr, momentum=0.9, weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_rec, T_max=self.args.num_epochs, eta_min=1e-4
        )

    def _create_augmentation_components(self):
        self.adv_optimizer = AdversarialAugmentationOptimizer(
            augmentation_module=self.augmentation_module,
            recognition_network=self.recognition_net,
            pgd_steps=self.args.pgd_steps,
            pgd_step_size=self.args.pgd_step_size,
            control_dim=4 + self.args.style_dim
        )
        self.sampler_geo = MomentumDynamicSampler(
            control_dim=4, momentum=self.args.momentum_geo
        )
        self.sampler_tex = MomentumDynamicSampler(
            control_dim=self.args.style_dim, momentum=self.args.momentum_tex
        )
        self.rate_controller = AugmentationRateController(
            geometric_rate=self.args.geometric_rate,
            textural_rate=self.args.textural_rate
        )

    def pretrain_generation_network(self):
        if not self.args.use_generation:
            print("[Pretrain] Skipping generation network pretraining")
            return
        print("[Pretrain] Starting generation network pretraining...")
        trainer = GenerationNetworkTrainer(
            self.generation_network, self.device,
            lr=self.args.gen_lr, num_epochs=self.args.gen_pretrain_epochs
        )
        for epoch in range(self.args.gen_pretrain_epochs):
            total_loss = 0
            pbar = tqdm(self.train_loader,
                        desc=f"[GenPreTrain] Epoch {epoch + 1}/{self.args.gen_pretrain_epochs}")
            for batch_idx, batch in enumerate(pbar):
                x = batch['img'].to(self.device)
                losses = trainer.train_on_batch(x, x)
                total_loss += losses['loss_gen']
                pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
            print(f"[Pretrain] Epoch {epoch + 1}: "
                  f"Loss = {total_loss / len(self.train_loader):.4f}")

    def train_epoch(self, epoch):
        self.recognition_net.train()
        total_loss = 0
        pbar = tqdm(self.train_loader,
                    desc=f"[Train] Epoch {epoch + 1}/{self.args.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            x = batch['img'].to(self.device)
            labels = batch['identity'].to(self.device)

            # ── Sample fresh z vectors every batch ──────────────────────────
            z_geo = self.sampler_geo.sample(len(x), self.device)   # (B, 4)
            z_tex = self.sampler_tex.sample(len(x), self.device)   # (B, style_dim)
            z_init = torch.cat([z_geo, z_tex], dim=1)              # (B, 4+style_dim)

            # ── Geometric adversarial optimisation ───────────────────────────
            if self.args.use_geometric:
                z_opt_geo = self.adv_optimizer.optimize_control_vector(
                    x, labels, z_init, augmentation_type='geometric'
                )
                self.sampler_geo.update_prev(z_opt_geo[:, :4])
            else:
                z_opt_geo = z_init[:, :4]

            # ── Textural adversarial optimisation ────────────────────────────
            z_full = torch.cat([z_opt_geo, z_init[:, 4:]], dim=1)
            if self.args.use_textural and self.args.use_generation:
                z_opt_tex = self.adv_optimizer.optimize_control_vector(
                    x, labels, z_full, augmentation_type='textural'
                )
                self.sampler_tex.update_prev(z_opt_tex[:, 4:])
            else:
                z_opt_tex = z_full

            # ── Apply augmentation & train recognition network ───────────────
            x_aug = self.augmentation_module(x, z_opt_tex, augmentation_type='both')
            x_combined = torch.cat([x, x_aug], dim=0)
            labels_combined = torch.cat([labels, labels], dim=0)

            logits, features = self.recognition_net(x_combined)
            loss = self.recognition_net.compute_loss(logits, labels_combined)

            self.optimizer_rec.zero_grad()
            loss.backward()
            self.optimizer_rec.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            self.global_step += 1

            if self.global_step % 100 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)

        self.scheduler.step()
        avg_loss = total_loss / len(self.train_loader)
        print(f"[Epoch {epoch + 1}] Train Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, epoch):
        self.recognition_net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"[Val] Epoch {epoch + 1}"):
                x = batch['img'].to(self.device)
                labels = batch['identity'].to(self.device)
                logits, _ = self.recognition_net(x)
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"[Epoch {epoch + 1}] Validation Accuracy: {accuracy:.4f}")
        self.writer.add_scalar('val/accuracy', accuracy, epoch)
        return accuracy

    def evaluate_final(self):
        print("\n[Eval] Starting comprehensive evaluation...")
        evaluator = PalmRecognitionEvaluator(self.recognition_net, self.device)
        results = evaluator.evaluate(
            self.test_loader, self.test_samples,
            tar_far_values=self.args.tar_far_values
        )
        evaluator.print_results(results)
        return results

    def train(self):
        print("[Train] Starting UAA training...")
        self.pretrain_generation_network()
        best_acc = 0

        for epoch in range(self.args.num_epochs):
            self.train_epoch(epoch)
            acc = self.validate(epoch)

            if acc > best_acc:
                best_acc = acc
                self.save_checkpoint(epoch, best=True)

            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(epoch, best=False)

        print(f"[Train] Training complete! Best validation accuracy: {best_acc:.4f}")
        self.writer.close()

        print("\n[Train] Running final evaluation with verification and identification metrics...")
        final_results = self.evaluate_final()

        return final_results

    def save_checkpoint(self, epoch, best=False):
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'recognition_net': self.recognition_net.state_dict(),
            'augmentation_module': self.augmentation_module.state_dict(),
            'optimizer': self.optimizer_rec.state_dict(),
        }
        if self.args.use_generation:
            checkpoint['generation_network'] = self.generation_network.state_dict()

        suffix = '_best' if best else ''
        save_path = f'checkpoints/checkpoint_epoch{epoch + 1}{suffix}.pt'
        torch.save(checkpoint, save_path)
        print(f"[Save] Checkpoint saved to {save_path}")


# ============================================================================
# SECTION 8: INFERENCE
# ============================================================================

class PalmRecognitionInference:
    """Inference engine for palmprint recognition"""

    def __init__(self, model_path, num_classes, device='cuda'):
        self.device = torch.device(device)
        self.model = PalmRecognitionNetwork(
            num_classes=num_classes, feature_dim=512, input_size=112
        ).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['recognition_net'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        print(f"[Inference] Model loaded from {model_path}")

    def extract_feature(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model.get_verification_features(img_tensor)
        return feature[0].cpu().numpy()

    def compute_similarity(self, feature1, feature2):
        similarity = np.dot(feature1, feature2) / (
            np.linalg.norm(feature1) * np.linalg.norm(feature2) + 1e-6
        )
        return (similarity + 1) / 2

    def verify_pair(self, image_path1, image_path2, threshold=0.5):
        feature1 = self.extract_feature(image_path1)
        feature2 = self.extract_feature(image_path2)
        similarity = self.compute_similarity(feature1, feature2)
        match = similarity >= threshold
        return match, similarity


# ============================================================================
# SECTION 9: MAIN ENTRY POINT
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("🚀 STARTING UAA TRAINING")
    print("=" * 80 + "\n")

    trainer = UAATrainer(args)
    final_results = trainer.train()

    print("\n" + "=" * 80)
    print("✅ TRAINING & EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"Checkpoints saved to: checkpoints/")
    print(f"TensorBoard logs saved to: runs/")
    print(f"Best model: checkpoints/checkpoint_*_best.pt")
    print("\nFinal Evaluation Summary:")
    print("-" * 80)

    for key, value in sorted(final_results.items()):
        if isinstance(value, float):
            print(f"  {key:<30} {value:.4f}")
        else:
            print(f"  {key:<30} {value}")

    print("-" * 80)
    print("\nTo view training curves:")
    print("  tensorboard --logdir=runs")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
