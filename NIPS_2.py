import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import cv2
from pytorch_metric_learning.losses import ArcFaceLoss

# ----------------------------
# Hyperparameters
# ----------------------------
batch_size = 32
margin = 0.3
scale = 16
lr = 1e-4          # Lowered slightly for fine-tuning
weight_decay = 1e-4
epochs = 50

# Choose domains by NAME
train_domains = ["WHT", "460"]   # training spectra
test_domains  = ["630"]          # unseen test spectrum

# Device
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

        # Resize
        img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # To Tensor + Normalize (0-1)
        img = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Optional: Add ImageNet normalization if using pretrained models
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # img = normalize(img)

        return img, y_i, y_d

# ----------------------------
# 2. Setup Data
# ----------------------------
data_path = "/home/pai-ng/Jamal/CASIA-MS-ROI"
dataset = CASIA_MS_Dataset(data_path)

# Domain Split
inv_domain_map = {v: k for k, v in dataset.domain_map.items()}
train_indices = []
test_indices = []

for idx, (_, _, y_d) in enumerate(dataset.samples):
    domain_name = inv_domain_map[y_d]
    if domain_name in train_domains:
        train_indices.append(idx)
    elif domain_name in test_domains:
        test_indices.append(idx)

train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train samples: {len(train_indices)} | Test samples: {len(test_indices)}")

# ----------------------------
# 3. Model: ConvNeXt Tiny
# ----------------------------
print("Loading ConvNeXt Tiny...")
# We use num_classes=0 to get the pooling layer output directly (embedding)
model = timm.create_model('convnext_tiny', pretrained=True, num_classes=0).to(device)

# 3.1 Freeze ALL parameters first
for param in model.parameters():
    param.requires_grad = False

# 3.2 Unfreeze the last TWO stages (Stage 2 and Stage 3)
# ConvNeXt stages are typically indexed 0, 1, 2, 3
for param in model.stages[2].parameters():
    param.requires_grad = True
    
for param in model.stages[3].parameters():
    param.requires_grad = True

# 3.3 Unfreeze the final Norm layer
# Important: The final norm processes features before they exit the backbone
if hasattr(model, 'norm'):
    for param in model.norm.parameters():
        param.requires_grad = True

# Get embedding dimension
embedding_dim = model.num_features 
print(f"Model Ready. Embedding Dim: {embedding_dim}")

# ----------------------------
# 4. Loss & Optimizer
# ----------------------------
num_classes = len(dataset.hand_id_map)

# Note: pytorch_metric_learning's ArcFaceLoss creates its own learnable weights 'W'
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
def get_arcface_preds(embeddings, criterion_layer):
    """
    Helper to calculate accuracy from ArcFaceLoss.
    Computes Cosine Similarity between embeddings and class centers (W).
    """
    # Normalize features and weights
    features_norm = F.normalize(embeddings, p=2, dim=1)
    weights_norm = F.normalize(criterion_layer.W, p=2, dim=1)
    
    # Cosine similarity [Batch, Num_Classes]
    cosine_sim = torch.mm(features_norm, weights_norm.t())
    return cosine_sim.argmax(dim=1)

for epoch in range(epochs):
    # -------- Training --------
    model.train()
    # ArcFace loss layer also needs to be in train mode (for updating W)
    criterion.train() 
    
    train_loss, train_correct, total_train = 0.0, 0, 0

    for images, y_i, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        images, y_i = images.to(device), y_i.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(images)
        
        # Loss calculation
        loss = criterion(embeddings, y_i)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        # Calculate Accuracy
        preds = get_arcface_preds(embeddings, criterion)
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
            
            # Loss
            loss = criterion(embeddings, y_i)
            test_loss += loss.item()
            
            # Accuracy
            preds = get_arcface_preds(embeddings, criterion)
            test_correct += (preds == y_i).sum().item()
            total_test += y_i.size(0)

    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_correct/total_train:.4f} | "
          f"Test Acc: {test_correct/total_test:.4f}")
