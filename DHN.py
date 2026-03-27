"""
Deep Hashing Network (DHN) for Palmprint Recognition
Implementation for CASIA-MS Dataset

Based on: "Palmprint-Palmvein Fusion Recognition Based on Deep Hashing Network"
(IEEE Access, 2021) - Wu et al.

Key components:
- Spatial Transformer Network (STN) for alignment
- Modified CNN-F backbone (5 conv + 3 FC layers)
- PReLU activation
- Distance Loss + Quantization Loss
- Binary hash code output via tanh + sign
"""

import os
import random
import math
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
    img_size = 128  # Paper uses 128x128 input
    
    # Model (following paper Table 2)
    hash_bit = 128  # Hash code length (64, 128, or 256)
    use_stn = True  # Use Spatial Transformer Network
    
    # Training
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001  # Standard for CNN training
    alpha = 0.5  # Weight for distance loss (paper Section III.C)
    
    # Distance loss threshold T (depends on hash_bit)
    # Paper: "T is determined by the length of the hashing code"
    # Typically T = hash_bit / 2
    threshold_T = hash_bit // 2
    
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
    
    # Create label mappings
    train_id_to_label = {id_: idx for idx, id_ in enumerate(sorted(train_identities))}
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
# Spatial Transformer Network (Section III.A)
# ============================================================================
class STN(nn.Module):
    """
    Spatial Transformer Network (Figure 1)
    
    Consists of:
    1. Localization Network - regresses transformation parameters θ
    2. Grid Generator - computes sampling grid
    3. Sampler - samples input at grid locations
    
    Uses affine transformation (6 parameters)
    """
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        
        # Calculate the size after localization conv layers
        # For 128x128 input: 128 -> 64 -> 32 -> 16
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)  # 6 parameters for affine transformation
        )
        
        # Initialize to identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Transformed tensor [B, C, H, W]
        """
        batch_size = x.size(0)
        
        # Localization network
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        
        # Predict transformation parameters (Equation 1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # Grid generator and sampler
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        
        return x_transformed


# ============================================================================
# Modified CNN-F Network (Section III.B, Table 2, Figure 2)
# ============================================================================
class DHN(nn.Module):
    """
    Deep Hashing Network with Modified CNN-F backbone
    
    Architecture (Table 2):
    - Input: 128×128×1
    - STN: 128×128×1
    - Conv1: 16×3×3, stride 4, pad 0, BN, PReLU → 16×32×32
    - MaxPool: 2×2, stride 1, pad 0 → 16×31×31
    - Conv2: 32×5×5, stride 2, pad 2, BN, PReLU → 32×16×16
    - MaxPool: 2×2, stride 1, pad 0 → 32×15×15
    - Conv3: 64×3×3, stride 1, pad 1, PReLU → 64×15×15
    - Conv4: 64×3×3, stride 1, pad 1, PReLU → 64×15×15
    - Conv5: 128×3×3, stride 1, pad 1, PReLU → 128×15×15
    - MaxPool: 2×2, stride 1, pad 0 → 128×14×14
    - FC6: 2048
    - FC7: 2048
    - FC8: hash_bit with tanh + sgn
    """
    
    def __init__(self, hash_bit: int = 128, use_stn: bool = True):
        super().__init__()
        self.hash_bit = hash_bit
        self.use_stn = use_stn
        
        # Spatial Transformer Network
        if use_stn:
            self.stn = STN(in_channels=1)
        
        # Conv1: 16×3×3, stride 4, pad 0, BN, PReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=4, padding=0),
            nn.BatchNorm2d(16),
            nn.PReLU()
        )
        # MaxPool: 2×2, stride 1, pad 0
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        
        # Conv2: 32×5×5, stride 2, pad 2, BN, PReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        # MaxPool: 2×2, stride 1, pad 0
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        
        # Conv3: 64×3×3, stride 1, pad 1, PReLU
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        
        # Conv4: 64×3×3, stride 1, pad 1, PReLU
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        
        # Conv5: 128×3×3, stride 1, pad 1, PReLU
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        # MaxPool: 2×2, stride 1, pad 0
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        
        # Calculate feature map size after conv layers
        # Input: 128x128
        # After conv1 (stride 4): (128-3)/4 + 1 = 32 -> 32x32
        # After pool1 (stride 1, k=2): 32-1 = 31 -> 31x31
        # After conv2 (stride 2, pad 2): (31+4-5)/2 + 1 = 16 -> 16x16
        # After pool2: 16-1 = 15 -> 15x15
        # After conv3,4,5: 15x15 (stride 1, pad 1)
        # After pool3: 15-1 = 14 -> 14x14
        # Feature: 128 * 14 * 14 = 25088
        
        fc_input_size = 128 * 14 * 14
        
        # FC6: 2048
        self.fc6 = nn.Sequential(
            nn.Linear(fc_input_size, 2048),
            nn.PReLU(),
            nn.Dropout(0.5)
        )
        
        # FC7: 2048
        self.fc7 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.PReLU(),
            nn.Dropout(0.5)
        )
        
        # FC8: hash_bit with tanh activation
        self.fc8 = nn.Linear(2048, hash_bit)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, return_hash: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 1, 128, 128]
            return_hash: If True, return binary hash codes
        Returns:
            Hash features (continuous) or binary hash codes
        """
        # STN
        if self.use_stn:
            x = self.stn(x)
        
        # Convolutional layers
        x = self.conv1(x)   # [B, 16, 32, 32]
        x = self.pool1(x)   # [B, 16, 31, 31]
        
        x = self.conv2(x)   # [B, 32, 16, 16]
        x = self.pool2(x)   # [B, 32, 15, 15]
        
        x = self.conv3(x)   # [B, 64, 15, 15]
        x = self.conv4(x)   # [B, 64, 15, 15]
        x = self.conv5(x)   # [B, 128, 15, 15]
        x = self.pool3(x)   # [B, 128, 14, 14]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [B, 25088]
        
        # Fully connected layers
        x = self.fc6(x)     # [B, 2048]
        x = self.fc7(x)     # [B, 2048]
        x = self.fc8(x)     # [B, hash_bit]
        
        # Tanh activation (paper: "Tanh function is used as the activation 
        # function of the output layer")
        hash_features = torch.tanh(x)
        
        if return_hash:
            # Sign function for binary codes (paper: "Sign function is used 
            # to quantize the output")
            # Use sign for inference, but tanh for training (gradient flow)
            hash_codes = torch.sign(hash_features)
            return hash_codes
        
        return hash_features


# ============================================================================
# Loss Functions (Section III.C)
# ============================================================================
class DHNLoss(nn.Module):
    """
    DHN Loss Function (Section III.C, Equations 2-6)
    
    L = α * L_S + L_Q
    
    where:
    - L_S: Distance loss (enlarge inter-class, reduce intra-class)
    - L_Q: Quantization loss (push outputs toward ±1)
    - α: Scaling factor (0.5 per paper)
    """
    
    def __init__(self, alpha: float = 0.5, threshold_T: int = 64):
        super().__init__()
        self.alpha = alpha
        self.threshold_T = threshold_T
    
    def forward(self, hash_features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hash_features: Output of tanh layer [B, hash_bit]
            labels: Class labels [B]
        Returns:
            total_loss, distance_loss, quantization_loss
        """
        batch_size = hash_features.size(0)
        hash_bit = hash_features.size(1)
        
        # ==================== Distance Loss (Equations 2-4) ====================
        # Compute pairwise Hamming distance approximation
        # For continuous features in [-1, 1], we use:
        # D(f_i, f_j) ≈ (hash_bit - f_i · f_j) / 2
        # This approximates Hamming distance when features are binary
        
        # Compute similarity matrix (dot product)
        similarity = torch.mm(hash_features, hash_features.t())  # [B, B]
        
        # Approximate Hamming distance
        # D = (hash_bit - similarity) / 2
        hamming_dist = (hash_bit - similarity) / 2
        
        # Create label similarity matrix
        # l_ij = 1 if same class, 0 otherwise
        labels_col = labels.view(-1, 1)
        labels_row = labels.view(1, -1)
        same_class_mask = (labels_col == labels_row).float()  # [B, B]
        diff_class_mask = 1 - same_class_mask
        
        # Remove diagonal
        eye = torch.eye(batch_size, device=hash_features.device)
        same_class_mask = same_class_mask - eye
        
        # S(x_i, x_j) = 0.5 * l_ij * D(f_i, f_j)  (Equation 3)
        # Minimize distance for same class
        same_class_loss = 0.5 * same_class_mask * hamming_dist
        
        # S_bar(x_i, x_j) = 0.5 * (1 - l_ij) * max(T - D(f_i, f_j), 0)  (Equation 4)
        # Push apart different classes if distance < T
        margin_loss = F.relu(self.threshold_T - hamming_dist)
        diff_class_loss = 0.5 * diff_class_mask * margin_loss
        
        # L_S = sum of both (Equation 2)
        num_pairs = batch_size * (batch_size - 1)
        if num_pairs > 0:
            distance_loss = (same_class_loss.sum() + diff_class_loss.sum()) / num_pairs
        else:
            distance_loss = torch.tensor(0.0, device=hash_features.device)
        
        # ==================== Quantization Loss (Equation 5) ====================
        # L_Q = sum_i (1/2) * ||1 - |f_i|||_2
        # Push each element toward ±1
        quantization_loss = 0.5 * torch.mean((1 - torch.abs(hash_features)) ** 2)
        
        # ==================== Total Loss (Equation 6) ====================
        total_loss = self.alpha * distance_loss + quantization_loss
        
        return total_loss, distance_loss, quantization_loss


# ============================================================================
# Evaluation Functions
# ============================================================================
def extract_hash_codes(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """Extract binary hash codes from all samples"""
    model.eval()
    all_codes = []
    all_labels = []
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            # Get binary hash codes
            hash_codes = model(images, return_hash=True)
            all_codes.append(hash_codes.cpu())
            all_labels.append(labels)
    
    all_codes = torch.cat(all_codes, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_codes, all_labels


def compute_hamming_distance(codes1: torch.Tensor, codes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Hamming distance between two sets of binary codes
    
    Args:
        codes1: [N, hash_bit] binary codes (-1 or 1)
        codes2: [M, hash_bit] binary codes (-1 or 1)
    Returns:
        distances: [N, M] Hamming distances
    """
    # Convert from {-1, 1} to {0, 1}
    codes1_binary = (codes1 > 0).float()
    codes2_binary = (codes2 > 0).float()
    
    # Hamming distance = number of differing bits
    # Using XOR: different bits give 1, same bits give 0
    # D = hash_bit - 2 * (codes1 @ codes2.T) when codes are in {0, 1}
    # Simpler: D = (hash_bit - codes1 @ codes2.T) / 2 when codes are in {-1, 1}
    
    hash_bit = codes1.size(1)
    similarity = torch.mm(codes1, codes2.t())  # Range: [-hash_bit, hash_bit]
    distances = (hash_bit - similarity) / 2
    
    return distances


def compute_eer(genuine_distances: np.ndarray, impostor_distances: np.ndarray) -> float:
    """Compute Equal Error Rate using distances (lower is more similar)"""
    if len(genuine_distances) == 0 or len(impostor_distances) == 0:
        return 50.0
    
    # Convert distances to similarity scores (negate so higher is more similar)
    genuine_scores = -genuine_distances
    impostor_scores = -impostor_distances
    
    # Combine scores and labels
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    
    # Compute ROC
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Find EER
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    return eer * 100


def compute_rank1_accuracy(gallery_codes: torch.Tensor, gallery_labels: torch.Tensor,
                           probe_codes: torch.Tensor, probe_labels: torch.Tensor) -> float:
    """Compute Rank-1 identification accuracy using Hamming distance"""
    # Compute Hamming distances
    distances = compute_hamming_distance(probe_codes, gallery_codes)
    
    # Get nearest neighbor (minimum distance)
    _, top1_indices = distances.topk(1, dim=1, largest=False)
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Extract hash codes
    gallery_codes, gallery_labels = extract_hash_codes(model, gallery_loader, device)
    probe_codes, probe_labels = extract_hash_codes(model, probe_loader, device)
    
    # Compute Hamming distances
    distances = compute_hamming_distance(probe_codes, gallery_codes)
    
    # For each probe, compute genuine and impostor distances
    genuine_distances = []
    impostor_distances = []
    
    for i in range(len(probe_codes)):
        probe_label = probe_labels[i].item()
        
        # Genuine: distances to same identity in gallery
        genuine_mask = (gallery_labels == probe_label)
        if genuine_mask.sum() > 0:
            genuine_distances.extend(distances[i, genuine_mask].numpy().tolist())
        
        # Impostor: distances to different identities
        impostor_mask = (gallery_labels != probe_label)
        impostor_distances.extend(distances[i, impostor_mask].numpy().tolist())
    
    genuine_distances = np.array(genuine_distances)
    impostor_distances = np.array(impostor_distances)
    
    # Compute EER
    eer = compute_eer(genuine_distances, impostor_distances)
    
    # Compute Rank-1 accuracy
    rank1 = compute_rank1_accuracy(gallery_codes, gallery_labels, 
                                   probe_codes, probe_labels)
    
    return eer, rank1


# ============================================================================
# Training
# ============================================================================
def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dist_loss = 0
    total_quant_loss = 0
    num_batches = 0
    
    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        hash_features = model(images, return_hash=False)
        
        # Compute loss
        loss, dist_loss, quant_loss = criterion(hash_features, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_dist_loss += dist_loss.item()
        total_quant_loss += quant_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_dist_loss = total_dist_loss / num_batches
    avg_quant_loss = total_quant_loss / num_batches
    
    return avg_loss, avg_dist_loss, avg_quant_loss


def main():
    # Set seed for reproducibility
    set_seed(Config.seed)
    
    print("=" * 60)
    print("DHN: Deep Hashing Network for Palmprint Recognition")
    print("CASIA Multi-Spectral Dataset")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Image size: {Config.img_size}x{Config.img_size}")
    print(f"  Hash bit: {Config.hash_bit}")
    print(f"  Use STN: {Config.use_stn}")
    print(f"  Batch size: {Config.batch_size}")
    print(f"  Learning rate: {Config.learning_rate}")
    print(f"  Alpha (distance loss weight): {Config.alpha}")
    print(f"  Threshold T: {Config.threshold_T}")
    print(f"  Device: {Config.device}")
    
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
    model = DHN(hash_bit=Config.hash_bit, use_stn=Config.use_stn)
    model = model.to(Config.device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")
    
    # Loss function
    criterion = DHNLoss(alpha=Config.alpha, threshold_T=Config.threshold_T)
    
    # Optimizer (using Adam as commonly used for deep hashing)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    best_eer = 100.0
    best_rank1 = 0.0
    
    for epoch in range(1, Config.num_epochs + 1):
        # Train
        train_loss, dist_loss, quant_loss = train_epoch(
            model, train_loader, optimizer, criterion, Config.device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            eer, rank1 = evaluate(model, gallery_loader, probe_loader, Config.device)
            
            if eer < best_eer:
                best_eer = eer
            if rank1 > best_rank1:
                best_rank1 = rank1
            
            print(f"Epoch [{epoch:3d}/{Config.num_epochs}] | "
                  f"Loss: {train_loss:.4f} (D:{dist_loss:.4f} Q:{quant_loss:.4f}) | "
                  f"EER: {eer:.2f}% | Rank-1: {rank1:.2f}%")
        else:
            print(f"Epoch [{epoch:3d}/{Config.num_epochs}] | "
                  f"Loss: {train_loss:.4f} (D:{dist_loss:.4f} Q:{quant_loss:.4f})")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    eer, rank1 = evaluate(model, gallery_loader, probe_loader, Config.device)
    
    print(f"\nFinal Results:")
    print(f"  EER: {eer:.4f}%")
    print(f"  Rank-1 Accuracy: {rank1:.2f}%")
    print(f"\nBest Results during training:")
    print(f"  Best EER: {best_eer:.4f}%")
    print(f"  Best Rank-1: {best_rank1:.2f}%")


if __name__ == "__main__":
    main()
