import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
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
batch_size = 32   # Must be divisible by m (samples_per_class)
m_per_class = 4   # 8 classes * 4 samples = 32 batch size
margin = 0.3
scale = 16
lr = 1e-3         # Lowered slightly for stability
weight_decay = 1e-4
epochs = 20
lamb = 1.0

# Choose domains by NAME
train_domains = ["WHT", "460"]   
test_domains  = ["700"]          

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 1. Custom Sampler (Crucial for Batch Size 32)
# ----------------------------
class MPerClassSampler(Sampler):
    """
    Ensures every batch contains 'm' samples for every selected class.
    Essential for Contrastive Loss with small batches.
    """
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
                # Replace=True allows sampling if a class has fewer than m images
                selected_indices = np.random.choice(indices, self.m, replace=True) 
                batch.extend(selected_indices)
            
            yield batch

# ----------------------------
# 2. Dataset Class
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
        
        # Standard ImageNet normalization (Required for Pretrained Models)
        #self.normalize = transforms.Normalize(
           # mean=[0.485, 0.456, 0.406], 
            #std=[0.229, 0.224, 0.225]
        #)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y_i, y_d = self.samples[idx]
        
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        img = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
        #img = self.normalize(img)

        return img, y_i, y_d

# ----------------------------
# 3. Data Setup & Split
# ----------------------------
data_path = "/home/pai-ng/Jamal/CASIA-MS-ROI"
dataset = CASIA_MS_Dataset(data_path)

inv_domain_map = {v: k for k, v in dataset.domain_map.items()}
train_indices = []
test_indices = []

for idx, (_, _, y_d) in enumerate(dataset.samples):
    domain_name = inv_domain_map[y_d]
    if domain_name in train_domains:
        train_indices.append(idx)
    elif domain_name in test_domains:
        test_indices.append(idx)

# Subsets
train_dataset = Subset(dataset, train_indices)
test_dataset  = Subset(dataset, test_indices)

# --- Sampler Integration ---
# Extract labels for the sampler
train_labels = [dataset.samples[i][1] for i in train_indices]

sampler = MPerClassSampler(
    labels=train_labels, 
    m=m_per_class, 
    batch_size=batch_size, 
    length_before_new_iter=len(train_indices)
)

# Loaders
train_loader = DataLoader(
    train_dataset,
    batch_sampler=sampler,  # Use batch_sampler (Mutually exclusive with shuffle=True)
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

print(f"Train samples: {len(train_indices)} | Test samples: {len(test_indices)}")
print(f"Classes: {len(dataset.hand_id_map)}")

# ----------------------------
# 4. Model Setup
# ----------------------------
print("Loading ConvNeXt V2-Tiny...")
model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0).to(device)
embedding_dim = model.num_features 

# Freeze Logic
for p in model.parameters():
    p.requires_grad = False

print("Unfreezing stage 3 and norm layers...")
for p in model.stages[3].parameters():
    p.requires_grad = True
if hasattr(model, 'norm'):
     for p in model.norm.parameters():
         p.requires_grad = True

# --- SupCon Projection Head ---
# We need a small head to project features for contrastive loss
# This is standard practice (SimCLR/MoCo) to avoid hurting the main features
class ProjectionHead(nn.Module):
    def __init__(self, dim_in, dim_out=128):
        super(ProjectionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_out)
        )
    def forward(self, x):
        feat = self.head(x)
        return F.normalize(feat, dim=1)

proj_head = ProjectionHead(embedding_dim).to(device)

# ----------------------------
# 5. Losses & Optimizer
# ----------------------------
num_classes = len(dataset.hand_id_map)

# 1. Classification Loss
criterion_arc = losses.ArcFaceLoss(
    num_classes=num_classes,
    embedding_size=embedding_dim,
    margin=margin,
    scale=scale
).to(device)

# 2. Contrastive Loss (Supervised)
criterion_supcon = losses.SupConLoss(temperature=0.1).to(device)

# Optimizer
optimizer = optim.AdamW(
    list(model.parameters()) + 
    list(criterion_arc.parameters()) + 
    list(proj_head.parameters()),  # Don't forget to optimize the new head!
    lr=lr,
    weight_decay=weight_decay
)

# ----------------------------
# 6. Training Loop
# ----------------------------
for epoch in range(epochs):
    # -------- Training --------
    model.train()
    proj_head.train()
    criterion_arc.train()
    
    train_loss, train_correct, total_train = 0.0, 0, 0

    for images, y_i, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        images, y_i = images.to(device), y_i.to(device)

        optimizer.zero_grad()
        
        # 1. Forward Pass
        embeddings = model(images)           # [B, 768]
        projections = proj_head(embeddings)  # [B, 128] for SupCon
        
        # 2. Calculate Losses
        loss_arc = criterion_arc(embeddings, y_i)
        loss_con = criterion_supcon(projections, y_i)
        
        # Combined Loss (Lambda usually 0.1 - 1.0)
        loss = loss_arc + lamb * loss_con

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        # Accuracy (Based on ArcFace only)
        preds = criterion_arc.get_logits(embeddings).argmax(dim=1)
        train_correct += (preds == y_i).sum().item()
        total_train += y_i.size(0)

    # -------- Evaluation --------
    model.eval()
    proj_head.eval()
    criterion_arc.eval()
    
    test_loss, test_correct, total_test = 0.0, 0, 0
    
    with torch.no_grad():
        for images, y_i, _ in tqdm(test_loader, desc=f"Epoch {epoch+1} [Test]"):
            images, y_i = images.to(device), y_i.to(device)
            
            embeddings = model(images)
            loss = criterion_arc(embeddings, y_i) # Test only on classification
            
            test_loss += loss.item()
            
            preds = criterion_arc.get_logits(embeddings).argmax(dim=1)
            test_correct += (preds == y_i).sum().item()
            total_test += y_i.size(0)

    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_correct/total_train:.4f} | "
          f"Test Acc: {test_correct/total_test:.4f}")
