import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import cv2
from pytorch_metric_learning.losses import ArcFaceLoss

batch_size = 32
margin=0.3
scale=16
lr=1e-3
weight_decay=1e-4
epochs = 50

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




################################################## Mixure of Experts
from typing import Tuple, List

# ----------------------------
# LoRA Expert
# ----------------------------
class LoRAExpert(nn.Module):
    """Single LoRA expert (same as before)"""
    def __init__(self, dim, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.scaling = alpha / r
        self.fc1 = nn.Linear(dim, r, bias=False)
        self.fc2 = nn.Linear(r, dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.fc1.weight, a=5**0.5)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        return self.scaling * self.fc2(self.dropout(self.act(self.fc1(x))))

# ----------------------------
# Top-K Router / Gating Network
# ----------------------------
class GatingNetwork(nn.Module):
    """Gating network for MoE"""
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, num_experts]
        gate_logits = self.gate(x)
        top_k_gates, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_gates, dim=-1)
        return top_k_gates, top_k_indices

# ----------------------------
# MoE-LoRA Adapter
# ----------------------------
class MoELoRA(nn.Module):
    """Mixture-of-Experts LoRA adapter parallel to frozen MLP"""
    def __init__(self, dim, num_experts=4, top_k=2, r=8, alpha=16):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Create multiple LoRA experts
        self.experts = nn.ModuleList([LoRAExpert(dim, r, alpha) for _ in range(num_experts)])
        self.router = GatingNetwork(dim, num_experts, top_k)

    def forward(self, x):
        # x: [batch, seq_len, dim] or [batch, dim]
        orig_shape = x.shape
        if x.dim() == 3:
            # flatten batch and seq for routing: [B*S, D]
            x_flat = x.reshape(-1, x.shape[-1])
        else:
            x_flat = x

        # Router -> top-k gates and indices
        top_k_gates, top_k_indices = self.router(x_flat)  # [B*S, top_k], [B*S, top_k]

        # Accumulate expert outputs
        expert_out = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            idx = top_k_indices[:, i]  # [B*S]
            g = top_k_gates[:, i].unsqueeze(-1)  # [B*S, 1]

            # select corresponding experts
            expert_outputs = torch.stack([self.experts[j](x_flat) for j in range(self.num_experts)], dim=0)  # [num_experts, B*S, D]
            selected = expert_outputs[idx, torch.arange(x_flat.shape[0]), :]  # [B*S, D]

            expert_out += g * selected

        # restore original shape
        expert_out = expert_out.view(*orig_shape)
        return expert_out

# ----------------------------
# MLP + MoE-LoRA block
# ----------------------------
class MLPWithMoELoRA(nn.Module):
    """Frozen MLP + parallel MoE LoRA"""
    def __init__(self, mlp, dim, num_experts=4, top_k=2, r=8, alpha=16):
        super().__init__()
        self.mlp = mlp
        self.moe = MoELoRA(dim, num_experts, top_k, r, alpha)
        # Freeze original MLP
        for p in self.mlp.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.mlp(x) + self.moe(x)

# ----------------------------
# Replace ViT block with MoE-LoRA
# ----------------------------
class ViTBlockWithMoELoRA(nn.Module):
    """ViT block with frozen attention and MLP + MoE-LoRA"""
    def __init__(self, block, r=8, alpha=16, num_experts=4, top_k=2):
        super().__init__()
        self.norm1 = block.norm1
        self.attn = block.attn
        self.norm2 = block.norm2
        for p in self.attn.parameters():
            p.requires_grad = False

        dim = block.attn.qkv.in_features
        self.mlp = MLPWithMoELoRA(block.mlp, dim, num_experts, top_k, r, alpha)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------
# Embedding Model Wrapper
# --------------------------------------------------
class ViTEmbeddingModel(nn.Module):
    def __init__(self, backbone, embed_dim=512):
        super().__init__()
        self.backbone = backbone
        self.backbone.reset_classifier(0)  # remove classification head
        self.embedding = nn.Linear(backbone.num_features, embed_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x







######################################################### DATA
class CASIA_MS_Dataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        self.hand_id_map = {}
        self.domain_map = {}

        hand_id_counter = 0
        domain_counter = 0

        for root, _, files in os.walk(data_path):
            for fname in files:
                if not fname.lower().endswith(".jpg"):
                    continue

                # Expected format: ID_{l|r}_{spectrum}_{iteration}.jpg
                parts = fname[:-4].split("_")
                if len(parts) != 4:
                    continue

                subject_id, hand, spectrum, iteration = parts
                hand_id = f"{subject_id}_{hand}"

                if hand_id not in self.hand_id_map:
                    self.hand_id_map[hand_id] = hand_id_counter
                    hand_id_counter += 1

                if spectrum not in self.domain_map:
                    self.domain_map[spectrum] = domain_counter
                    domain_counter += 1

                img_path = os.path.join(root, fname)
                self.samples.append(
                    (img_path,
                     self.hand_id_map[hand_id],   # y_i (ID label)
                     self.domain_map[spectrum])   # y_d (domain)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y_i, y_d = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        # Resize AFTER ROI extraction
        img_np = cv2.resize(img_np, (224,224), interpolation=cv2.INTER_LINEAR)

        img = torch.tensor(
            img_np, dtype=torch.float32
        ).permute(2, 0, 1) / 255.0

        return img, y_i, y_d


# ================================
# Dataset
# ================================
data_path = "/home/pai-ng/Jamal/CASIA-MS-ROI"
dataset = CASIA_MS_Dataset(data_path)

num_classes = len(dataset.hand_id_map)
num_domains = len(dataset.domain_map)

print("Total samples:", len(dataset))
print("Hand ID classes:", num_classes)
print("Domains:", dataset.domain_map)

# ================================
# Cross-Domain Split Configuration
# ================================

# Choose domains by NAME: 460, 630, 700, WHT, 850, 940
train_domains = ["WHT", "460"]   # training spectra
test_domains  = ["700"]          # unseen test spectrum

# Reverse domain map: id → name
inv_domain_map = {v: k for k, v in dataset.domain_map.items()}

train_indices = []
test_indices = []

for idx, (_, _, y_d) in enumerate(dataset.samples):
    domain_name = inv_domain_map[y_d]

    if domain_name in train_domains:
        train_indices.append(idx)
    elif domain_name in test_domains:
        test_indices.append(idx)

# ================================
# Subsets
# ================================
train_dataset = Subset(dataset, train_indices)
test_dataset  = Subset(dataset, test_indices)

print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

# ================================
# DataLoaders
# ================================
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ================================
# Sanity Check
# ================================
images, y_i, y_d = next(iter(train_loader))

# Verify domain separation
train_domains_seen = set(inv_domain_map[d.item()] for d in y_d)
print("Domains seen in train batch:", train_domains_seen)


images, y_i, y_d = next(iter(test_loader))

# Verify domain separation
test_domains_seen = set(inv_domain_map[d.item()] for d in y_d)
print("Domains seen in test batch:", test_domains_seen)







################################################################# MODELS
class CustomCNN160(nn.Module):
    def __init__(self, input_channels=3):
        super(CustomCNN160, self).__init__()

        # --- Feature Extraction ---
        self.features = nn.Sequential(
            # Output: 16 x 40 x 40 -> MaxPool: 16 x 39 x 39
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=4, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=1),

            # Output: 32 x 19 x 19 -> MaxPool: 32 x 18 x 18
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            # Output: 128 x 18 x 18 -> MaxPool: 128 x 17 x 17
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        # --- Fully Connected Layers ---
        self.flatten = nn.Flatten()

        # Calculation: 128 * 17 * 17 = 36992
        self.classifier = nn.Sequential(
            nn.Linear(41472, 1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x





############################################################ MAIN

# ----------------------------
# Main Script for Custom ViT + MoE-LoRA + ArcFace
# ----------------------------


# ================================
# Backbone: DeiT + MoE-LoRA
# ================================
# 21M params, inference cost ? GFLOPs ---> deit_small_patch16_224
# 5M params, inference cost ? GFLOPs ---> deit_tiny_patch16_224
# 0.9M params, inference cost ? GFLOPs ---> test_vit3.r160_in1k
# inference cost 0.0885 GFLOPs ---> test_vit2.r160_in1k
# 0.4M params, inference cost 0.03 GFLOPs (less than baseline's CNN model) ---> test_vit.r160_in1k

'''
backbone = timm.create_model(
    "deit_tiny_patch16_224",
    pretrained=True,
    num_classes=0
)

# Freeze backbone
for p in backbone.parameters():
    p.requires_grad = False

# Replace blocks with MoE-LoRA
for i, block in enumerate(backbone.blocks):
    backbone.blocks[i] = ViTBlockWithMoELoRA(
        block,
        r=8,
        alpha=16,
        num_experts=4,
        top_k=2
    )
embedding_dim = 512
model = ViTEmbeddingModel(backbone, embed_dim=embedding_dim).to(device)



print("Loading ConvNeXt V2-Tiny...")
model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0).to(device)

# Freeze logic (Applied to base_encoder before passing to MoCo wrapper)
for p in model.parameters():
    p.requires_grad = False

print("Unfreezing the last stage (stage 3) and norm layers of ConvNeXt...")
for p in model.stages[3].parameters():
    p.requires_grad = True
if hasattr(model, 'norm'):
     for p in model.norm.parameters():
         p.requires_grad = True
'''

# Loading dinov2_vits14 (Small version, patch size 14)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)

# Freeze most layers, unfreeze last (two) blocks
# In DINOv2 hub models, blocks are accessed via .blocks
for name, p in model.named_parameters():
    p.requires_grad = False
    # Unfreezing the last two blocks (block 10 and 11 for ViT-S)
    if "blocks.10" in name or "blocks.11" in name:
        p.requires_grad = True
        
# ================================
# 5. ArcFace Loss & Optimizer
# ================================
#embedding_dim = model.num_features # ConvNeXt
embedding_dim = 384
num_classes = 200
criterion = ArcFaceLoss(
    num_classes=num_classes,
    embedding_size=embedding_dim,
    margin=margin,
    scale=scale
).to(device)

optimizer = optim.AdamW(
    list(model.parameters()) + list(criterion.parameters()),
    lr=lr,
    weight_decay=weight_decay
)

# ================================
# 6. Training + Evaluation Loop
# ================================
for epoch in range(epochs):
    # -------- Training --------
    model.train()
    train_loss, train_correct, total_train = 0.0, 0, 0

    # FIXED: Added the third value '_' for unpacking
    for images, y_i, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        images, y_i = images.to(device), y_i.to(device)

        optimizer.zero_grad()
        embeddings = model(images)
        loss = criterion(embeddings, y_i)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = criterion.get_logits(embeddings).argmax(dim=1)
        train_correct += (preds == y_i).sum().item()
        total_train += y_i.size(0)

    # -------- Evaluation --------
    model.eval()
    test_loss, test_correct, total_test = 0.0, 0, 0
    with torch.no_grad():
        for images, y_i, _ in tqdm(test_loader, desc=f"Epoch {epoch+1} [Test]"):
            images, y_i = images.to(device), y_i.to(device)
            embeddings = model(images)
            loss = criterion(embeddings, y_i)
            test_loss += loss.item()
            preds = criterion.get_logits(embeddings).argmax(dim=1)
            test_correct += (preds == y_i).sum().item()
            total_test += y_i.size(0)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_correct/total_train:.4f} | Test Acc: {test_correct/total_test:.4f}")



def save_model_to_drive(model, save_path, filename="model.pth"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    full_path = os.path.join(save_path, filename)

    # Save the weights (state_dict)
    # Moving to CPU before saving is a best practice for compatibility
    torch.save(model.state_dict(), full_path)
    print(f"✅ Model weights successfully saved to: {full_path}")

# Usage:
DRIVE_PATH = "/home/pai-ng/Jamal/CovNeXt.pth"
save_model_to_drive(model, DRIVE_PATH)


'''
def load_moe_vit_model(checkpoint_path, device, num_experts=4, top_k=2):
    # 1. Rebuild the exact same architecture
    backbone = timm.create_model(
    "test_vit2.r160_in1k",
    pretrained=True,
    num_classes=0
    )
    for i, block in enumerate(backbone.blocks):
        backbone.blocks[i] = ViTBlockWithMoELoRA(
            block,
            r=8,
            alpha=16,
            num_experts=num_experts,
            top_k=top_k
        )

    # 2. Wrap and move to device
    model = ViTEmbeddingModel(backbone, embed_dim=512).to(device)

    # 3. Load the weights
    if os.path.exists(checkpoint_path):
        # map_location=device handles the CPU -> GPU transfer if needed
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print("✅ Weights loaded and architecture matched successfully.")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    return model

# Usage for TTA:
CHECKPOINT = os.path.join(DRIVE_PATH, 'model.pth')
model = load_moe_vit_model(CHECKPOINT, device, num_experts=4, top_k=2)

'''






#################################################################### Test Time Adaptation
# --------------------------------------------------------------------------------
# Test-Time Adaptation (Episodic Entropy Minimization)
# --------------------------------------------------------------------------------
print("\nStarting Global TTA Evaluation...")

# 1. Create a CPU backup for episodic reset
# Since 'model' is already loaded via load_moe_vit_model, we clone its current state.
cpu_reset_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

'''
# 2. Freeze all parameters except MoE Experts/Routers
for p in model.parameters():
    p.requires_grad = False

for name, p in model.named_parameters():
    if any(k in name.lower() for k in ["lora", "expert", "gate", "router"]):
        p.requires_grad = True
'''


total_baseline_acc = 0.0
total_tta_acc = 0.0
num_batches = len(test_loader)

# 3. Iterate through all batches
for images, labels, _ in tqdm(test_loader, desc="TTA on Test Set"):
    images, labels = images.to(device), labels.to(device)

    # --- Step A: Episodic Reset ---
    # Return to the clean, trained state for every new batch
    model.load_state_dict(cpu_reset_state)

    # --- Step B: Baseline Check (Before TTA) ---
    model.eval()
    with torch.no_grad():
        embeddings = model(images)
        logits = criterion.get_logits(embeddings)
        baseline_batch_acc = (logits.argmax(dim=1) == labels).float().mean().item()
        total_baseline_acc += baseline_batch_acc

    # --- Step C: Perform TTA (Entropy Minimization) ---
    # Optimizer specifically for MoE/LoRA parameters
    tta_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    model.train()
    for _ in range(20): # 20 steps based on your BS=32 observations
        tta_optimizer.zero_grad()
        embeddings = model(images)
        logits = criterion.get_logits(embeddings)
        probs = F.softmax(logits, dim=1)

        # Entropy loss calculation: -sum(p * log(p))
        entropy_loss = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
        entropy_loss.backward()
        tta_optimizer.step()

    # --- Step D: Inference (After TTA) ---
    model.eval()
    with torch.no_grad():
        embeddings = model(images)
        logits = criterion.get_logits(embeddings)
        tta_batch_acc = (logits.argmax(dim=1) == labels).float().mean().item()
        total_tta_acc += tta_batch_acc

# 4. Final Results Summary
final_baseline = total_baseline_acc / num_batches
final_tta = total_tta_acc / num_batches

print("\n" + "="*40)
print(f"FINAL TTA RESULTS ({num_batches} Batches)")
print(f"Average Baseline Acc: {final_baseline:.4f}")
print(f"Average TTA Acc:      {final_tta:.4f}")
print(f"Absolute Improvement: {final_tta - final_baseline:.4f}")
print("="*40)

