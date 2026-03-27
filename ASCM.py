"""
SAC-Net: Scale-Aware Competition Network for Palmprint Recognition
Implementation for CASIA-MS Dataset

Based on: "Scale-Aware Competition Network for Palmprint Recognition" (ICASSP 2024)
Authors: Gao et al.
"""

import os
import random
import numpy as np
from PIL import Image
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Dataset
    data_root = '/home/pai-ng/Jamal/CASIA-MS-ROI'
    img_size = 128  # Resize images to this size
    
    # Model
    num_orientations = 6  # Number of Gabor filter orientations
    gabor_scales = [7, 17, 35]  # Tiny, Middle, Large filter sizes
    num_heads = 4  # Multi-head self-attention heads
    feature_dim = 512  # Final feature dimension
    
    # Training
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.0003
    contrastive_margin = 0.5
    contrastive_weight = 0.5  # Weight for contrastive loss
    
    # Evaluation protocol
    train_ratio = 0.8  # 80% IDs for training
    gallery_ratio = 0.1  # 10% of test images as gallery
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reproducibility
    seed = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Dataset
# ============================================================================
class CASIAMSDataset(Dataset):
    """
    CASIA Multi-Spectral Palmprint Dataset
    Filename structure: {subjectID}_{handSide}_{spectrum}_{iteration}.jpg
    Different hands are treated as different identities.
    """
    
    def __init__(self, file_list: List[str], id_to_label: Dict[str, int], 
                 data_root: str, img_size: int = 128, augment: bool = False):
        self.file_list = file_list
        self.id_to_label = id_to_label
        self.data_root = data_root
        self.img_size = img_size
        self.augment = augment
    
    def __len__(self):
        return len(self.file_list)
    
    def _parse_filename(self, filename: str) -> Tuple[str, str, str, str]:
        """Parse filename to extract subject, hand, spectrum, iteration"""
        parts = filename.replace('.jpg', '').replace('.JPG', '').split('_')
        subject_id = parts[0]
        hand_side = parts[1]
        spectrum = parts[2]
        iteration = parts[3]
        return subject_id, hand_side, spectrum, iteration
    
    def _get_identity(self, filename: str) -> str:
        """Get unique identity (subject + hand)"""
        subject_id, hand_side, _, _ = self._parse_filename(filename)
        return f"{subject_id}_{hand_side}"
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        img_path = os.path.join(self.data_root, filename)
        
        # Load and preprocess image
        img = Image.open(img_path).convert('L')  # Grayscale
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        
        # Data augmentation (only for training)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
            # Random rotation (-10 to 10 degrees)
            # Slight brightness variation
            img = img * (0.9 + 0.2 * random.random())
            img = np.clip(img, 0, 1)
        
        # Convert to tensor [1, H, W]
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        
        # Get label
        identity = self._get_identity(filename)
        label = self.id_to_label[identity]
        
        return img_tensor, label, filename


def prepare_dataset(config: Config):
    """
    Prepare dataset with open-set evaluation protocol.
    - 80% IDs for training
    - 20% IDs for testing (10% gallery, 90% probe)
    """
    print("=" * 60)
    print("Scanning dataset...")
    print("=" * 60)
    
    # Get all files
    all_files = [f for f in os.listdir(config.data_root) 
                 if f.endswith('.jpg') or f.endswith('.JPG')]
    
    # Parse files and organize by identity
    id_to_files = defaultdict(list)
    spectrums_found = set()
    
    for f in all_files:
        parts = f.replace('.jpg', '').replace('.JPG', '').split('_')
        subject_id = parts[0]
        hand_side = parts[1]
        spectrum = parts[2]
        identity = f"{subject_id}_{hand_side}"
        id_to_files[identity].append(f)
        spectrums_found.add(spectrum)
    
    # Dataset statistics
    all_identities = sorted(id_to_files.keys())
    num_identities = len(all_identities)
    total_images = len(all_files)
    
    print(f"Data root: {config.data_root}")
    print(f"Total images: {total_images}")
    print(f"Unique identities (subject + hand): {num_identities}")
    print(f"Spectrums found: {sorted(spectrums_found)}")
    print(f"Avg images per identity: {total_images / num_identities:.1f}")
    
    # Split identities into train/test
    random.shuffle(all_identities)
    num_train_ids = int(num_identities * config.train_ratio)
    train_identities = set(all_identities[:num_train_ids])
    test_identities = set(all_identities[num_train_ids:])
    
    print(f"\nTrain identities: {len(train_identities)}")
    print(f"Test identities: {len(test_identities)}")
    
    # Create label mappings (only for train identities during training)
    train_id_to_label = {id_: idx for idx, id_ in enumerate(sorted(train_identities))}
    # For test, we still need labels for evaluation
    test_id_to_label = {id_: idx for idx, id_ in enumerate(sorted(test_identities))}
    
    # Prepare file lists
    train_files = []
    for identity in train_identities:
        train_files.extend(id_to_files[identity])
    
    # Split test files into gallery and probe
    gallery_files = []
    probe_files = []
    for identity in test_identities:
        files = id_to_files[identity].copy()
        random.shuffle(files)
        num_gallery = max(1, int(len(files) * config.gallery_ratio))
        gallery_files.extend(files[:num_gallery])
        probe_files.extend(files[num_gallery:])
    
    print(f"\nTrain images: {len(train_files)}")
    print(f"Test gallery images: {len(gallery_files)}")
    print(f"Test probe images: {len(probe_files)}")
    
    # Create datasets
    train_dataset = CASIAMSDataset(
        train_files, train_id_to_label, config.data_root, 
        config.img_size, augment=True
    )
    gallery_dataset = CASIAMSDataset(
        gallery_files, test_id_to_label, config.data_root,
        config.img_size, augment=False
    )
    probe_dataset = CASIAMSDataset(
        probe_files, test_id_to_label, config.data_root,
        config.img_size, augment=False
    )
    
    return (train_dataset, gallery_dataset, probe_dataset, 
            len(train_identities), len(test_identities))


# ============================================================================
# Learnable Gabor Filter Layer
# ============================================================================
class LearnableGaborFilter(nn.Module):
    """
    Learnable Gabor Filter Layer
    Parameters (λ, θ, ψ, σ, γ) are learnable for each orientation.
    """
    
    def __init__(self, in_channels: int, num_orientations: int, kernel_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_orientations = num_orientations
        self.kernel_size = kernel_size
        
        # Learnable parameters for each orientation
        # Initialize θ evenly distributed across orientations
        theta_init = torch.linspace(0, np.pi, num_orientations + 1)[:-1]
        self.theta = nn.Parameter(theta_init)
        
        # Initialize other parameters
        self.sigma = nn.Parameter(torch.ones(num_orientations) * (kernel_size / 4))
        self.lambd = nn.Parameter(torch.ones(num_orientations) * (kernel_size / 2))
        self.gamma = nn.Parameter(torch.ones(num_orientations) * 0.5)
        self.psi = nn.Parameter(torch.zeros(num_orientations))
        
        # Create coordinate grids
        half_size = kernel_size // 2
        y, x = torch.meshgrid(
            torch.arange(-half_size, half_size + 1, dtype=torch.float32),
            torch.arange(-half_size, half_size + 1, dtype=torch.float32),
            indexing='ij'
        )
        self.register_buffer('x', x)
        self.register_buffer('y', y)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Output tensor [B, num_orientations, H, W]
        """
        batch_size = x.size(0)
        
        # Generate Gabor kernels
        kernels = self._generate_kernels()  # [num_orientations, 1, K, K]
        
        # Apply convolution for each input channel and sum
        outputs = []
        for i in range(self.in_channels):
            out = F.conv2d(x[:, i:i+1], kernels, padding=self.kernel_size // 2)
            outputs.append(out)
        
        output = torch.stack(outputs, dim=0).sum(dim=0)
        return output
    
    def _generate_kernels(self) -> torch.Tensor:
        """Generate Gabor filter kernels from learnable parameters"""
        kernels = []
        
        for i in range(self.num_orientations):
            theta = self.theta[i]
            sigma = torch.clamp(self.sigma[i], min=1.0)
            lambd = torch.clamp(self.lambd[i], min=2.0)
            gamma = torch.clamp(self.gamma[i], min=0.1, max=1.0)
            psi = self.psi[i]
            
            # Rotation
            x_theta = self.x * torch.cos(theta) + self.y * torch.sin(theta)
            y_theta = -self.x * torch.sin(theta) + self.y * torch.cos(theta)
            
            # Gabor function
            gaussian = torch.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2)
            sinusoid = torch.cos(2 * np.pi * x_theta / lambd + psi)
            kernel = gaussian * sinusoid
            
            # Normalize
            kernel = kernel - kernel.mean()
            kernel = kernel / (kernel.std() + 1e-8)
            
            kernels.append(kernel)
        
        kernels = torch.stack(kernels, dim=0).unsqueeze(1)  # [num_orientations, 1, K, K]
        return kernels


# ============================================================================
# Multi-Head Self-Attention
# ============================================================================
class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention for capturing long-range dependencies"""
    
    def __init__(self, in_channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)
        
        self.norm = nn.LayerNorm(in_channels)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence: [B, H*W, C]
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Compute Q, K, V
        Q = self.query(x_flat).view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_flat).view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_flat).view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, H*W, C)
        out = self.out_proj(out)
        
        # Residual connection and normalization
        out = self.norm(out + x_flat)
        
        # Reshape back to [B, C, H, W]
        out = out.transpose(1, 2).view(B, C, H, W)
        
        return out


# ============================================================================
# Inner-Scale Competition Module (ISCM)
# ============================================================================
class ISCM(nn.Module):
    """
    Inner-Scale Competition Module
    - Two Learnable Gabor Filter layers
    - Multi-Head Self-Attention
    - Softmax-based Competitive Coding along orientation dimension
    """
    
    def __init__(self, num_orientations: int, kernel_size: int, num_heads: int = 4):
        super().__init__()
        self.num_orientations = num_orientations
        
        # Two LGF layers
        self.lgf1 = LearnableGaborFilter(1, num_orientations, kernel_size)
        self.lgf2 = LearnableGaborFilter(num_orientations, num_orientations, kernel_size)
        
        # Batch normalization after LGF
        self.bn1 = nn.BatchNorm2d(num_orientations)
        self.bn2 = nn.BatchNorm2d(num_orientations)
        
        # Multi-Head Self-Attention
        self.msa = MultiHeadSelfAttention(num_orientations, num_heads)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, 1, H, W]
        Returns:
            F_inner: Competition features along orientation [B, num_orientations, H, W]
            F_msa: Features before softmax (for ASCM) [B, num_orientations, H, W]
        """
        # LGF layers
        out = self.lgf1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.lgf2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        # Multi-Head Self-Attention
        F_msa = self.msa(out)
        
        # Softmax-based Competitive Coding along orientation (channel) dimension
        F_inner = F.softmax(F_msa, dim=1)
        
        return F_inner, F_msa


# ============================================================================
# Across-Scale Competition Module (ASCM)
# ============================================================================
class ASCM(nn.Module):
    """
    Across-Scale Competition Module
    - Concatenates features from multiple scales
    - Applies softmax competition across scales
    """
    
    def __init__(self, num_orientations: int, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        
        # 1x1 convolution to combine scale information
        self.conv = nn.Conv2d(num_orientations * num_scales, num_orientations * num_scales, 1)
        self.bn = nn.BatchNorm2d(num_orientations * num_scales)
    
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_list: List of F_msa from each scale branch
                          Each tensor: [B, num_orientations, H, W]
        Returns:
            F_across: Competition features across scales [B, num_orientations * num_scales, H, W]
        """
        # Concatenate features from all scales
        F_concat = torch.cat(features_list, dim=1)  # [B, num_orientations * num_scales, H, W]
        
        # Apply convolution
        F_concat = self.conv(F_concat)
        F_concat = self.bn(F_concat)
        
        # Softmax competition across all scale-orientation combinations
        F_across = F.softmax(F_concat, dim=1)
        
        return F_across


# ============================================================================
# SAC-Net Model
# ============================================================================
class SACNet(nn.Module):
    """
    Scale-Aware Competition Network
    - Three branches: Tiny, Middle, Large scale
    - Each branch has ISCM
    - ASCM combines features across scales
    - Classification head
    """
    
    def __init__(self, num_classes: int, config: Config):
        super().__init__()
        self.num_orientations = config.num_orientations
        self.scales = config.gabor_scales
        self.num_scales = len(self.scales)
        
        # ISCM for each scale branch
        self.iscm_tiny = ISCM(config.num_orientations, self.scales[0], config.num_heads)
        self.iscm_middle = ISCM(config.num_orientations, self.scales[1], config.num_heads)
        self.iscm_large = ISCM(config.num_orientations, self.scales[2], config.num_heads)
        
        # ASCM
        self.ascm = ASCM(config.num_orientations, self.num_scales)
        
        # Feature dimension after ISCM and ASCM
        total_channels = config.num_orientations * self.num_scales  # From ASCM
        total_channels += config.num_orientations * self.num_scales  # From all ISCMs
        
        # Pooling and FC layers
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(total_channels, config.feature_dim),
            nn.BatchNorm1d(config.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
        # Classification head
        self.classifier = nn.Linear(config.feature_dim, num_classes)
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x: Input tensor [B, 1, H, W]
            return_features: If True, return features instead of logits
        Returns:
            logits or features depending on return_features flag
        """
        # ISCM for each scale
        F_inner_tiny, F_msa_tiny = self.iscm_tiny(x)
        F_inner_middle, F_msa_middle = self.iscm_middle(x)
        F_inner_large, F_msa_large = self.iscm_large(x)
        
        # ASCM
        F_across = self.ascm([F_msa_tiny, F_msa_middle, F_msa_large])
        
        # Concatenate all features
        F_all = torch.cat([
            F_inner_tiny, F_inner_middle, F_inner_large,
            F_across
        ], dim=1)
        
        # Global pooling
        F_pooled = self.pool(F_all).flatten(1)
        
        # FC layer to get features
        features = self.fc(F_pooled)
        
        if return_features:
            return F.normalize(features, p=2, dim=1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits, features


# ============================================================================
# Losses
# ============================================================================
class ContrastiveLoss(nn.Module):
    """Contrastive loss for feature learning"""
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Normalized feature vectors [B, D]
            labels: Class labels [B]
        Returns:
            Contrastive loss value
        """
        batch_size = features.size(0)
        
        # Compute pairwise distances
        distances = torch.cdist(features, features, p=2)
        
        # Create positive/negative masks
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()
        
        # Remove diagonal
        eye = torch.eye(batch_size, device=features.device)
        positive_mask = positive_mask - eye
        
        # Positive loss: minimize distance for same class
        positive_loss = (positive_mask * distances).sum() / (positive_mask.sum() + 1e-8)
        
        # Negative loss: push apart different classes (with margin)
        negative_loss = (negative_mask * F.relu(self.margin - distances)).sum() / (negative_mask.sum() + 1e-8)
        
        return positive_loss + negative_loss


# ============================================================================
# Evaluation Functions
# ============================================================================
def extract_features(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """Extract features from all samples in the dataloader"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            features = model(images, return_features=True)
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_features, all_labels


def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    """Compute Equal Error Rate"""
    # Combine scores and labels
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    
    # Compute ROC
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Find EER
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    return eer * 100  # Return as percentage


def compute_rank1_accuracy(gallery_features: torch.Tensor, gallery_labels: torch.Tensor,
                           probe_features: torch.Tensor, probe_labels: torch.Tensor) -> float:
    """Compute Rank-1 identification accuracy"""
    # Compute cosine similarity (features are already normalized)
    similarity = torch.mm(probe_features, gallery_features.t())
    
    # Get top-1 predictions
    _, top1_indices = similarity.topk(1, dim=1)
    predicted_labels = gallery_labels[top1_indices.squeeze()]
    
    # Compute accuracy
    correct = (predicted_labels == probe_labels).sum().item()
    accuracy = correct / len(probe_labels) * 100
    
    return accuracy


def evaluate(model: nn.Module, gallery_loader: DataLoader, probe_loader: DataLoader,
             device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model on gallery and probe sets
    Returns: (EER, Rank-1 accuracy)
    """
    # Extract features
    gallery_features, gallery_labels = extract_features(model, gallery_loader, device)
    probe_features, probe_labels = extract_features(model, probe_loader, device)
    
    # Compute similarity scores for EER
    similarity = torch.mm(probe_features, gallery_features.t())
    
    # For each probe, compute genuine and impostor scores
    genuine_scores = []
    impostor_scores = []
    
    for i in range(len(probe_features)):
        probe_label = probe_labels[i].item()
        
        # Genuine: scores with same identity in gallery
        genuine_mask = (gallery_labels == probe_label)
        if genuine_mask.sum() > 0:
            genuine_scores.extend(similarity[i, genuine_mask].numpy().tolist())
        
        # Impostor: scores with different identities
        impostor_mask = (gallery_labels != probe_label)
        impostor_scores.extend(similarity[i, impostor_mask].numpy().tolist())
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    # Compute EER
    eer = compute_eer(genuine_scores, impostor_scores)
    
    # Compute Rank-1 accuracy
    rank1 = compute_rank1_accuracy(gallery_features, gallery_labels, 
                                   probe_features, probe_labels)
    
    return eer, rank1


# ============================================================================
# Training
# ============================================================================
def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                ce_criterion: nn.Module, contrastive_criterion: nn.Module,
                device: torch.device, contrastive_weight: float) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, features = model(images)
        
        # Compute losses
        ce_loss = ce_criterion(logits, labels)
        features_normalized = F.normalize(features, p=2, dim=1)
        contrastive_loss = contrastive_criterion(features_normalized, labels)
        
        # Combined loss
        loss = ce_loss + contrastive_weight * contrastive_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    
    return avg_loss, accuracy


def main():
    # Set seed for reproducibility
    set_seed(Config.seed)
    
    print("=" * 60)
    print("SAC-Net: Scale-Aware Competition Network")
    print("CASIA Multi-Spectral Palmprint Recognition")
    print("=" * 60)
    print(f"\nDevice: {Config.device}")
    
    # Prepare dataset
    train_dataset, gallery_dataset, probe_dataset, num_train_ids, num_test_ids = \
        prepare_dataset(Config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=Config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    probe_loader = DataLoader(
        probe_dataset, batch_size=Config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Create model
    model = SACNet(num_classes=num_train_ids, config=Config)
    model = model.to(Config.device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")
    
    # Loss functions
    ce_criterion = nn.CrossEntropyLoss()
    contrastive_criterion = ContrastiveLoss(margin=Config.contrastive_margin)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.num_epochs, eta_min=1e-6
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    best_eer = 100.0
    best_rank1 = 0.0
    
    for epoch in range(1, Config.num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, ce_criterion, contrastive_criterion,
            Config.device, Config.contrastive_weight
        )
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            eer, rank1 = evaluate(model, gallery_loader, probe_loader, Config.device)
            
            if eer < best_eer:
                best_eer = eer
            if rank1 > best_rank1:
                best_rank1 = rank1
            
            print(f"Epoch [{epoch:3d}/{Config.num_epochs}] | "
                  f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"EER: {eer:.2f}% | Rank-1: {rank1:.2f}%")
        else:
            print(f"Epoch [{epoch:3d}/{Config.num_epochs}] | "
                  f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    eer, rank1 = evaluate(model, gallery_loader, probe_loader, Config.device)
    
    print(f"\nFinal Results:")
    print(f"  EER: {eer:.4f}%")
    print(f"  Rank-1 Accuracy: {rank1:.2f}%")
    print(f"\nBest Results during training:")
    print(f"  Best EER: {best_eer:.4f}%")
    print(f"  Best Rank-1: {best_rank1:.2f}%")


if __name__ == "__main__":
    main()
