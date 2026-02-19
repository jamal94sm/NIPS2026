import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import cv2
from pytorch_metric_learning import losses
from collections import defaultdict

# ----------------------------
# Configuration
# ----------------------------
batch_size = 32   
m_per_class = 4   # 8 classes * 4 samples = 32
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
# 1. Augmentations (Standard/Strong for SupCon)
# ----------------------------
# Strong Augmentation for Training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.6, 1.0)),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Critical for spectra
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Clean Transform for Testing
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------------------
# 2. Custom Sampler
# ----------------------------
class MPerClassSampler(Sampler):
    def __init__(self, labels, m, batch_size=None, length_before_new_iter=100000):
        self.m = m
        self.batch_size = batch_size
        self.labels = np.array(labels)
        self.unique_labels = list(set(self.labels))
        self.length_before_new_iter = length_before_new_iter

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        self.classes_per_batch = self.batch_size // self.m

    def __len__(self):
        return self.length_before_new_iter

    def __iter__(self):
        num_batches = self.length_before_new_iter // self.batch_size
        for _ in range(num_batches):
            batch = []
            selected_classes = np.random.choice(self.unique_labels, self.classes_per_batch, replace=False)
            for cls in selected_classes:
                indices = self.label_to_indices[cls]
                selected_indices = np.random.choice(indices, self.m, replace=True) 
                batch.extend(selected_indices)
            yield batch

# ----------------------------
# 3. Dataset Class
# ----------------------------
class CASIA_MS_Dataset(Dataset):
    def __init__(self, data_path, target_domains, transform=None):
        self.samples = []
        self.hand_id_map = {}
        self.transform = transform
        
        hand_id_counter = 0

        # Note: We assume IDs are consistent across domains
        for root, _, files in os.walk(data_path):
            files.sort() # Ensure deterministic order
            for fname in files:
                if not fname.lower().endswith(".jpg"):
                    continue

                parts = fname[:-4].split("_")
                if len(parts) != 4:
                    continue

                subject_id, hand, spectrum, iteration = parts
                
                # Internal Filtering
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
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.Resize((224, 224))(img)
            img = transforms.ToTensor()(img)

        return img, y_i

# ----------------------------
# 4. Data Loading
# ----------------------------
data_path = "/home/pai-ng/Jamal/CASIA-MS-ROI"

print("Creating Training Dataset...")
train_dataset = CASIA_MS_Dataset(data_path, target_domains=train_domains, transform=train_transform)

print("Creating Test Dataset...")
test_dataset  = CASIA_MS_Dataset(data_path, target_domains=test_domains, transform=val_transform)

# Sampler Setup
train_labels = [s[1] for s in train_dataset.samples]
sampler = MPerClassSampler(train_labels, m=m_per_class, batch_size=batch_size, length_before_new_iter=len(train_dataset))

train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

# ----------------------------
# 5. Model Setup
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
# 6. Losses & Optimizer
# ----------------------------
num_classes = len(train_dataset.hand_id_map)

criterion_arc = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_dim, margin=margin, scale=scale).to(device)
criterion_supcon = losses.SupConLoss(temperature=0.1).to(device)

optimizer = optim.AdamW(
    list(model.parameters()) + list(criterion_arc.parameters()) + list(proj_head.parameters()), 
    lr=lr, weight_decay=weight_decay
)

# ----------------------------
# 7. Training Loop
# ----------------------------
for epoch in range(epochs):
    # -------- Training --------
    model.train(); proj_head.train(); criterion_arc.train()
    train_loss, train_correct, total_train = 0.0, 0, 0

    for images, y_i in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        images, y_i = images.to(device), y_i.to(device)

        optimizer.zero_grad()
        
        # Forward
        embeddings = model(images)           
        projections = proj_head(embeddings)  
        
        # Combined Loss
        loss = criterion_arc(embeddings, y_i) + lamb * criterion_supcon(projections, y_i)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = criterion_arc.get_logits(embeddings).argmax(dim=1)
        train_correct += (preds == y_i).sum().item()
        total_train += y_i.size(0)

    # -------- Evaluation --------
    model.eval(); criterion_arc.eval()
    test_loss, test_correct, total_test = 0.0, 0, 0
    
    with torch.no_grad():
        for images, y_i in tqdm(test_loader, desc=f"Epoch {epoch+1} [Test]"):
            images, y_i = images.to(device), y_i.to(device)
            embeddings = model(images)
            
            # Test Loss & Accuracy (ArcFace only)
            test_loss += criterion_arc(embeddings, y_i).item()
            preds = criterion_arc.get_logits(embeddings).argmax(dim=1)
            test_correct += (preds == y_i).sum().item()
            total_test += y_i.size(0)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_correct/total_train:.4f} | Test Acc: {test_correct/total_test:.4f}")
