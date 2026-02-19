import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import cv2
from pytorch_metric_learning import losses

# ----------------------------
# Configuration
# ----------------------------
batch_size = 32   
margin = 0.3
scale = 16
lr = 1e-3
weight_decay = 1e-4
epochs = 20
lamb = 1.0        # Weight for Contrastive Loss

# Choose domains by NAME
train_domains = ["WHT", "460"]   
test_domains  = ["700"]          

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 1. Transforms (Separated)
# ----------------------------
# Transform 1: CLEAN (Original)
# Used for ArcFace classification and as the 'Anchor' for SupCon
orig_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(...) # Add if needed
])

# Transform 2: AUGMENTED
# Used only for the Contrastive 'Positive' view
aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.6, 1.0)),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.ToTensor(),
    # transforms.Normalize(...) # Add if needed
])

# ----------------------------
# 2. Dataset Class (Updated)
# ----------------------------
class CASIA_MS_Dataset(Dataset):
    def __init__(self, data_path, target_domains, orig_transform=None, aug_transform=None, is_train=True):
        self.samples = []
        self.hand_id_map = {}
        self.orig_transform = orig_transform
        self.aug_transform = aug_transform
        self.is_train = is_train
        
        hand_id_counter = 0

        for root, _, files in os.walk(data_path):
            files.sort()
            for fname in files:
                if not fname.lower().endswith(".jpg"):
                    continue

                parts = fname[:-4].split("_")
                if len(parts) != 4:
                    continue

                subject_id, hand, spectrum, iteration = parts
                
                if spectrum not in target_domains:
                    continue

                hand_id = f"{subject_id}_{hand}"

                if hand_id not in self.hand_id_map:
                    self.hand_id_map[hand_id] = hand_id_counter
                    hand_id_counter += 1
                
                img_path = os.path.join(root, fname)
                self.samples.append((img_path, self.hand_id_map[hand_id]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y_i = self.samples[idx]
        
        img = Image.open(img_path).convert("RGB")
        
        # Apply Original Transform (Always)
        if self.orig_transform:
            img_orig = self.orig_transform(img)
        else:
            img_orig = transforms.Resize((224, 224))(img)
            img_orig = transforms.ToTensor()(img_orig)

        # If training, ALSO return the Augmented version
        if self.is_train and self.aug_transform:
            img_aug = self.aug_transform(img)
            return img_orig, img_aug, y_i
        
        # If testing, just return original
        return img_orig, y_i

# ----------------------------
# 3. Data Loading
# ----------------------------
data_path = "/home/pai-ng/Jamal/CASIA-MS-ROI"

print("Creating Training Dataset...")
# Pass BOTH transforms here
train_dataset = CASIA_MS_Dataset(
    data_path, 
    target_domains=train_domains, 
    orig_transform=orig_transform, 
    aug_transform=aug_transform,
    is_train=True
)

print("Creating Test Dataset...")
# Pass only orig_transform, is_train=False
test_dataset  = CASIA_MS_Dataset(
    data_path, 
    target_domains=test_domains, 
    orig_transform=orig_transform,
    is_train=False
)

# Standard Shuffle=True DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

# ----------------------------
# 4. Model Setup
# ----------------------------
print("Loading ConvNeXt V2-Tiny...")
model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0).to(device)
embedding_dim = model.num_features 

# Freeze Logic
for p in model.parameters():
    p.requires_grad = False
for p in model.stages[3].parameters():
    p.requires_grad = True
if hasattr(model, 'norm'):
     for p in model.norm.parameters():
         p.requires_grad = True

# SupCon Projection Head
class ProjectionHead(nn.Module):
    def __init__(self, dim_in, dim_out=128):
        super(ProjectionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_out)
        )
    def forward(self, x):
        return F.normalize(self.head(x), dim=1)

proj_head = ProjectionHead(embedding_dim).to(device)

# ----------------------------
# 5. Losses & Optimizer
# ----------------------------
num_classes = len(train_dataset.hand_id_map)

criterion_arc = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_dim, margin=margin, scale=scale).to(device)
criterion_supcon = losses.SupConLoss(temperature=0.1).to(device)

optimizer = optim.AdamW(
    list(model.parameters()) + list(criterion_arc.parameters()) + list(proj_head.parameters()), 
    lr=lr, weight_decay=weight_decay
)

# ----------------------------
# 6. Training Loop
# ----------------------------
for epoch in range(epochs):
    # -------- Training --------
    model.train(); proj_head.train(); criterion_arc.train()
    train_loss, train_correct, total_train = 0.0, 0, 0

    # Note: Unpacking 3 items now: orig, aug, label
    for img_orig, img_aug, y_i in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        img_orig = img_orig.to(device)
        img_aug  = img_aug.to(device)
        y_i      = y_i.to(device)

        optimizer.zero_grad()
        
        # Concatenate for efficient forward pass (Batch size becomes 2*B = 64)
        images = torch.cat([img_orig, img_aug], dim=0)
        
        # 1. Forward Pass
        embeddings = model(images)           
        
        # Split embeddings back into Original and Augmented
        # First half is Original, Second half is Augmented
        bs = img_orig.size(0)
        emb_orig = embeddings[:bs]
        emb_aug  = embeddings[bs:]
        
        # 2. Losses
        
        # A) ArcFace: Apply ONLY to the Original (Clean) images
        # This keeps the class centers stable and noise-free
        loss_arc = criterion_arc(emb_orig, y_i)
        
        # B) SupCon: Apply to BOTH (Original + Augmented)
        # We project them first
        proj_orig = proj_head(emb_orig)
        proj_aug  = proj_head(emb_aug)
        
        # SupCon expects stacked features [Batch, n_views, dim]
        # We create a stack of 2 views per image
        features_stacked = torch.stack([proj_orig, proj_aug], dim=1)
        loss_con = criterion_supcon(features_stacked, y_i)
        
        # Combined Loss
        loss = loss_arc + lamb * loss_con
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        # Accuracy: Based on the Clean images
        preds = criterion_arc.get_logits(emb_orig).argmax(dim=1)
        train_correct += (preds == y_i).sum().item()
        total_train += y_i.size(0)

    # -------- Evaluation --------
    model.eval(); criterion_arc.eval()
    test_loss, test_correct, total_test = 0.0, 0, 0
    
    with torch.no_grad():
        # Test loader only returns img_orig and label
        for img_orig, y_i in tqdm(test_loader, desc=f"Epoch {epoch+1} [Test]"):
            img_orig, y_i = img_orig.to(device), y_i.to(device)
            
            embeddings = model(img_orig)
            
            test_loss += criterion_arc(embeddings, y_i).item()
            preds = criterion_arc.get_logits(embeddings).argmax(dim=1)
            test_correct += (preds == y_i).sum().item()
            total_test += y_i.size(0)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_correct/total_train:.4f} | Test Acc: {test_correct/total_test:.4f}")
