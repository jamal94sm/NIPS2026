import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import copy
import numpy as np
from PIL import Image
from pytorch_metric_learning import losses

# ----------------------------
# Configuration
# ----------------------------
batch_size = 32   
margin = 0.3
scale = 16
lr = 1e-3
weight_decay = 1e-4
epochs = 100
lamb = 0.2       # Weight for Contrastive Loss
moco_m = 0.99    # Momentum for Teacher update (0.99 is best for small batches)

# Choose domains by NAME
train_domains = ["WHT", "460"]   
test_domains  = ["700"]          

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 1. Transforms (Separated)
# ----------------------------
orig_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2.0)], p=0.2),
    transforms.RandomApply([transforms.RandomAutocontrast()], p=0.2),
    transforms.ToTensor(),
])

# ----------------------------
# 2. Dataset Class
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
        
        if self.orig_transform:
            img_orig = self.orig_transform(img)
        else:
            img_orig = transforms.Resize((224, 224))(img)
            img_orig = transforms.ToTensor()(img_orig)

        if self.is_train and self.aug_transform:
            img_aug = self.aug_transform(img)
            return img_orig, img_aug, y_i
        
        return img_orig, y_i

# ----------------------------
# 3. Data Loading
# ----------------------------
data_path = "/home/pai-ng/Jamal/CASIA-MS-ROI"

print("Creating Training Dataset...")
train_dataset = CASIA_MS_Dataset(data_path, train_domains, orig_transform, aug_transform, True)
print("Creating Test Dataset...")
test_dataset  = CASIA_MS_Dataset(data_path, test_domains, orig_transform, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

# ----------------------------
# 4. Model Setup (MoCo v3 Dual Architecture)
# ----------------------------
print("Loading ConvNeXt V2-Tiny...")

# A. Student Networks (Learn via Backprop)
student_backbone = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0).to(device)
embedding_dim = student_backbone.num_features 

# Freeze early layers
for p in student_backbone.parameters(): p.requires_grad = False
for p in student_backbone.stages[3].parameters(): p.requires_grad = True
if hasattr(student_backbone, 'norm'):
     for p in student_backbone.norm.parameters(): p.requires_grad = True

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, dim_out=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_out)
        )
    def forward(self, x):
        return F.normalize(self.head(x), dim=1)

# MoCo v3 specific: Predictor Head on the Student
class PredictionHead(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return F.normalize(self.head(x), dim=1)

student_proj = ProjectionHead(embedding_dim).to(device)
student_pred = PredictionHead(128).to(device)

# B. Teacher Networks (Learn via EMA only)
teacher_backbone = copy.deepcopy(student_backbone)
teacher_proj = copy.deepcopy(student_proj)

# Freeze Teacher completely
for p in teacher_backbone.parameters(): p.requires_grad = False
for p in teacher_proj.parameters(): p.requires_grad = False

# Function to update Teacher
@torch.no_grad()
def update_momentum_teacher(m):
    for param_s, param_t in zip(student_backbone.parameters(), teacher_backbone.parameters()):
        param_t.data = param_t.data * m + param_s.data * (1. - m)
    for param_s, param_t in zip(student_proj.parameters(), teacher_proj.parameters()):
        param_t.data = param_t.data * m + param_s.data * (1. - m)

# ----------------------------
# 5. Losses & Optimizer
# ----------------------------
num_classes = len(train_dataset.hand_id_map)
criterion_arc = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_dim, margin=margin, scale=scale).to(device)

# Custom Supervised MoCo Loss with Memory Queue
class SupMoCoLoss(nn.Module):
    def __init__(self, queue_size=4096, feat_dim=128, temperature=0.1):
        super().__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.register_buffer("queue", nn.functional.normalize(torch.randn(feat_dim, queue_size), dim=0))
        self.register_buffer("queue_labels", -torch.ones(queue_size, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_labels[ptr:ptr + batch_size] = labels
        self.ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, q, k, labels):
        # Compare Student Query (q) against Teacher Key (k) AND the Memory Queue
        all_keys = torch.cat([k.T, self.queue.clone().detach()], dim=1) 
        all_labels = torch.cat([labels, self.queue_labels.clone().detach()], dim=0) 
        
        logits = torch.matmul(q, all_keys) / self.temperature
        mask = torch.eq(labels.unsqueeze(1), all_labels.unsqueeze(0)).float() 
        
        # Stability shift
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - max_logits.detach()
        
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = -mean_log_prob_pos.mean()
        
        self.enqueue(k, labels)
        return loss

criterion_moco = SupMoCoLoss(queue_size=4096, feat_dim=128, temperature=0.1).to(device)

# Optimizer strictly applies to Student networks and ArcFace
all_params = list(student_backbone.parameters()) + list(criterion_arc.parameters()) + \
             list(student_proj.parameters()) + list(student_pred.parameters())

optimizer = optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

# ----------------------------
# 6. Training Loop
# ----------------------------
for epoch in range(epochs):
    # -------- Training --------
    student_backbone.train(); student_proj.train(); student_pred.train()
    criterion_arc.train()
    # Teacher is permanently in eval mode
    teacher_backbone.eval(); teacher_proj.eval()
    
    train_loss = 0.0
    train_correct = 0
    total_train = 0

    for img_orig, img_aug, y_i in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        img_orig, img_aug, y_i = img_orig.to(device), img_aug.to(device), y_i.to(device)
        optimizer.zero_grad()
        
        # --- 1. Student Forward Pass (Requires Grad) ---
        emb_orig_s = student_backbone(img_orig)
        emb_aug_s  = student_backbone(img_aug)
        
        # Projections -> Predictions
        q1 = student_pred(student_proj(emb_orig_s))
        q2 = student_pred(student_proj(emb_aug_s))

        # --- 2. Teacher Forward Pass (No Grad) ---
        with torch.no_grad():
            emb_orig_t = teacher_backbone(img_orig)
            emb_aug_t  = teacher_backbone(img_aug)
            
            k1 = teacher_proj(emb_orig_t)
            k2 = teacher_proj(emb_aug_t)

        # --- 3. Losses ---
        # A) ArcFace Loss (Uses Student Original Embeddings only)
        loss_arc = criterion_arc(emb_orig_s, y_i)
        
        # B) Symmetric SupMoCo Loss (Student vs Teacher)
        loss_moco = (criterion_moco(q1, k2, y_i) + criterion_moco(q2, k1, y_i)) / 2.0
        
        loss = loss_arc + lamb * loss_moco
        loss.backward()
        optimizer.step()

        # --- 4. EMA Momentum Update ---
        update_momentum_teacher(moco_m)

        train_loss += loss.item()
        
        # Accuracy: Based on Student ArcFace
        preds = criterion_arc.get_logits(emb_orig_s).argmax(dim=1)
        train_correct += (preds == y_i).sum().item()
        total_train += img_orig.size(0)

    # -------- Evaluation (Using Student Only) --------
    student_backbone.eval(); criterion_arc.eval()
    test_loss = 0.0
    test_correct = 0
    total_test = 0
    
    with torch.no_grad():
        for img_orig, y_i in tqdm(test_loader, desc=f"Epoch {epoch+1} [Test]"):
            img_orig, y_i = img_orig.to(device), y_i.to(device)
            
            embeddings = student_backbone(img_orig)
            loss = criterion_arc(embeddings, y_i)
            test_loss += loss.item()
            
            preds = criterion_arc.get_logits(embeddings).argmax(dim=1)
            test_correct += (preds == y_i).sum().item()
            total_test += y_i.size(0)

    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Train Loss: {train_loss/len(train_loader):.4f} "
          f"Train Acc: {train_correct/total_train:.4f} | "
          f"Test Acc: {test_correct/total_test:.4f}")

    scheduler.step()
