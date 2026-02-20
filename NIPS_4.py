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
epochs = 100
lamb = 0.2       # Weight for Contrastive Loss

# Choose domains by NAME
train_domains = ["WHT", "460", "700"]   
test_domains  = ["850"]          

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
# 2. Dataset Class (Updated for Domain Labels)
# ----------------------------
class CASIA_MS_Dataset(Dataset):
    def __init__(self, data_path, target_domains, orig_transform=None, aug_transform=None, is_train=True):
        self.samples = []
        self.hand_id_map = {}
        # Map spectrum name to an integer domain label (e.g., WHT->0, 460->1, 700->2)
        self.domain_map = {d: i for i, d in enumerate(target_domains)}
        
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
                y_d = self.domain_map[spectrum] # Get domain label
                self.samples.append((img_path, self.hand_id_map[hand_id], y_d))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y_i, y_d = self.samples[idx]
        
        img = Image.open(img_path).convert("RGB")
        
        if self.orig_transform:
            img_orig = self.orig_transform(img)
        else:
            img_orig = transforms.Resize((224, 224))(img)
            img_orig = transforms.ToTensor()(img_orig)

        if self.is_train and self.aug_transform:
            img_aug = self.aug_transform(img)
            # Return original, augmented, identity label, and domain label
            return img_orig, img_aug, y_i, y_d
        
        # Testing returns original, identity, and domain
        return img_orig, y_i, y_d

# ----------------------------
# 3. Data Loading
# ----------------------------
data_path = "/home/pai-ng/Jamal/CASIA-MS-ROI"

print("Creating Training Dataset...")
train_dataset = CASIA_MS_Dataset(
    data_path, 
    target_domains=train_domains, 
    orig_transform=orig_transform, 
    aug_transform=aug_transform,
    is_train=True
)

print("Creating Test Dataset...")
test_dataset  = CASIA_MS_Dataset(
    data_path, 
    target_domains=test_domains, 
    orig_transform=orig_transform,
    is_train=False
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

# ----------------------------
# 4. Model Setup (with GRL)
# ----------------------------
print("Loading ConvNeXt V2-Tiny...")
model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0).to(device)
embedding_dim = model.num_features 

for p in model.parameters(): p.requires_grad = False
for p in model.stages[3].parameters(): p.requires_grad = True
if hasattr(model, 'norm'):
     for p in model.norm.parameters(): p.requires_grad = True

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

# --- NEW: Gradient Reversal Layer ---
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Multiply gradient by negative alpha
        output = grad_output.neg() * ctx.alpha
        return output, None

# --- NEW: Domain Classifier ---
class DomainClassifier(nn.Module):
    def __init__(self, dim_in, num_domains):
        super(DomainClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2),
            nn.BatchNorm1d(dim_in // 2),
            nn.ReLU(True),
            nn.Linear(dim_in // 2, num_domains)
        )
    def forward(self, x, alpha):
        # Pass through GRL first
        x_rev = GradientReversal.apply(x, alpha)
        return self.net(x_rev)

# Initialize Domain Classifier
num_domains = len(train_dataset.domain_map)
domain_classifier = DomainClassifier(embedding_dim, num_domains).to(device)

# ----------------------------
# 5. Losses & Optimizer
# ----------------------------
num_classes = len(train_dataset.hand_id_map)

criterion_arc = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_dim, margin=margin, scale=scale).to(device)
criterion_supcon = losses.SupConLoss(temperature=0.1).to(device)
criterion_domain = nn.CrossEntropyLoss().to(device) # Loss for GRL

# Add Domain Classifier to optimizer
all_params = list(model.parameters()) + list(criterion_arc.parameters()) + \
             list(proj_head.parameters()) + list(domain_classifier.parameters())

optimizer = optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

# ----------------------------
# 6. Training Loop
# ----------------------------
total_batches = len(train_loader) * epochs

for epoch in range(epochs):
    # -------- Training --------
    model.train(); proj_head.train(); criterion_arc.train(); domain_classifier.train()
    
    train_loss = 0.0
    train_correct = 0
    total_train = 0

    # Unpack 4 items now: orig, aug, identity_label, domain_label
    for batch_idx, (img_orig, img_aug, y_i, y_d) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")):
        img_orig = img_orig.to(device)
        img_aug  = img_aug.to(device)
        y_i      = y_i.to(device)
        y_d      = y_d.to(device)

        # Calculate GRL Alpha (Ramps from 0 to 1 over the course of training)
        p = float(batch_idx + epoch * len(train_loader)) / total_batches
        alpha = 2. / (1. + np.exp(-25 * p)) - 1

        optimizer.zero_grad()
        
        # 1. Forward Pass
        images_all = torch.cat([img_orig, img_aug], dim=0)
        embeddings_all = model(images_all)
        projections_all = proj_head(embeddings_all)

        # Split embeddings
        batch_size_curr = img_orig.size(0)
        emb_orig = embeddings_all[:batch_size_curr]

        # 2. Domain Classifier Forward
        # We need domain labels for both orig and aug
        y_d_all = torch.cat([y_d, y_d], dim=0)
        domain_logits = domain_classifier(embeddings_all, alpha)

        # 3. Calculate Losses
        
        # A) ArcFace
        loss_arc = criterion_arc(emb_orig, y_i)
        
        # B) SupCon 
        labels_all = torch.cat([y_i, y_i], dim=0)
        loss_con = criterion_supcon(projections_all, labels_all)
        
        # C) Domain Loss
        loss_domain = criterion_domain(domain_logits, y_d_all)
        
        # Total Loss
        # Note: GRL handles the subtraction automatically during backward()
        loss = loss_arc + (lamb * loss_con) + loss_domain
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        # Identity Accuracy
        preds = criterion_arc.get_logits(emb_orig).argmax(dim=1)
        train_correct += (preds == y_i).sum().item()
        total_train += batch_size_curr

    # -------- Evaluation --------
    model.eval(); criterion_arc.eval()
    test_loss = 0.0
    test_correct = 0
    total_test = 0
    
    with torch.no_grad():
        # Unpack 3 items now
        for img_orig, y_i, y_d in tqdm(test_loader, desc=f"Epoch {epoch+1} [Test]"):
            img_orig, y_i = img_orig.to(device), y_i.to(device)
            
            embeddings = model(img_orig)
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
