import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import cv2
from pytorch_metric_learning.losses import ArcFaceLoss

# ----------------------------
# Configuration
# ----------------------------
batch_size = 32
margin = 0.3
scale = 16
lr = 1e-4             
weight_decay = 1e-4
epochs = 20

# Domain Selection
train_domains = ["WHT", "460"]   # training spectra
test_domains  = ["700"]          # unseen test spectrum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 1. Dataset Class
# ----------------------------
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

        # Resize (Standard for ConvNeXt is 224x224)
        img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Convert to Tensor (Normalize 0-1)
        img = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return img, y_i, y_d

# ----------------------------
# 2. Data Setup
# ----------------------------
data_path = "/home/pai-ng/Jamal/CASIA-MS-ROI"
dataset = CASIA_MS_Dataset(data_path)

# Split by Domain
inv_domain_map = {v: k for k, v in dataset.domain_map.items()}
train_indices = []
test_indices = []

for idx, (_, _, y_d) in enumerate(dataset.samples):
    domain_name = inv_domain_map[y_d]
    if domain_name in train_domains:
        train_indices.append(idx)
    elif domain_name in test_domains:
        test_indices.append(idx)

# Create Subsets & Loaders
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"Total: {len(dataset)} | Train: {len(train_indices)} | Test: {len(test_indices)}")
print(f"Classes: {len(dataset.hand_id_map)}")

# ----------------------------
# 2.1 Domain Sanity Check (ADDED)
# ----------------------------
print("\n--- Verifying Domain Split ---")

# Check Train Batch
try:
    _, _, y_d_train = next(iter(train_loader))
    train_domains_seen = set(inv_domain_map[d.item()] for d in y_d_train)
    print(f"Training batch contains domains: {train_domains_seen}")
    if not train_domains_seen.issubset(set(train_domains)):
        print(f"WARNING: Found unexpected domains in Training set!")
except StopIteration:
    print("Training loader is empty!")

# Check Test Batch
try:
    _, _, y_d_test = next(iter(test_loader))
    test_domains_seen = set(inv_domain_map[d.item()] for d in y_d_test)
    print(f"Testing batch contains domains:  {test_domains_seen}")
    if not test_domains_seen.issubset(set(test_domains)):
        print(f"WARNING: Found unexpected domains in Test set!")
except StopIteration:
    print("Test loader is empty!")

print("------------------------------\n")

# ----------------------------
# 3. Model Setup (ConvNeXt-Tiny)
# ----------------------------
print("Loading ConvNeXt Tiny...")
# num_classes=0 removes the head, giving us the raw embedding
model = timm.create_model('convnext_tiny', pretrained=True, num_classes=0).to(device)

# --- Freezing Logic ---
# 1. Freeze EVERYTHING
for p in model.parameters():
    p.requires_grad = False

# 2. Unfreeze the last TWO stages (ConvNeXt has stages 0, 1, 2, 3)
for p in model.stages[2].parameters():
    p.requires_grad = True
for p in model.stages[3].parameters():
    p.requires_grad = True

# 3. Unfreeze the final Norm (Important for feature statistics)
if hasattr(model, 'norm'):
    for p in model.norm.parameters():
        p.requires_grad = True

embedding_dim = model.num_features
print(f"Model Ready. Embedding Dim: {embedding_dim}")

# ----------------------------
# 4. Loss & Optimizer
# ----------------------------
num_classes = len(dataset.hand_id_map)

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

# ----------------------------
# 5. Training Loop
# ----------------------------
for epoch in range(epochs):
    # -------- Training --------
    model.train()
    criterion.train() 
    
    train_loss, train_correct, total_train = 0.0, 0, 0

    for images, y_i, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        images, y_i = images.to(device), y_i.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(images)
        
        # Loss
        loss = criterion(embeddings, y_i)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        # Accuracy
        preds = criterion.get_logits(embeddings).argmax(dim=1)
        train_correct += (preds == y_i).sum().item()
        total_train += y_i.size(0)

    # -------- Evaluation --------
    model.eval()
    criterion.eval()
    
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

    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_correct/total_train:.4f} | "
          f"Test Acc: {test_correct/total_test:.4f}")
