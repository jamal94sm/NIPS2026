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
from typing import Tuple, List

batch_size = 32
margin=0.3
scale=16
lr=1e-3
weight_decay=1e-4
epochs = 50

# Choose domains by NAME: 460, 630, 700, WHT, 850, 940
train_domains = ["WHT", "460"]   # training spectra
test_domains  = ["700"]          # unseen test spectrum


# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







##################################################################
##################################################### Data Process
##################################################################
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






########################################################################
################################################################# MODELS
########################################################################
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


def save_model_to_drive(model, save_path, filename="model.pth"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    full_path = os.path.join(save_path, filename)

    # Save the weights (state_dict)
    # Moving to CPU before saving is a best practice for compatibility
    torch.save(model.state_dict(), full_path)
    print(f"✅ Model weights successfully saved to: {full_path}")





####################################################################
###################################################### The MAIN LOOP
####################################################################

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



####################################
print("Loading ConvNeXt V2-Tiny...")
model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0).to(device)
embedding_dim = model.num_features 

# Freeze logic (Applied to base_encoder before passing to MoCo wrapper)
for p in model.parameters():
    p.requires_grad = False

print("Unfreezing the last stage (stage 3) and norm layers of ConvNeXt...")
for p in model.stages[3].parameters():
    p.requires_grad = True
if hasattr(model, 'norm'):
     for p in model.norm.parameters():
         p.requires_grad = True
# ================================
# 5. ArcFace Loss & Optimizer
# ================================
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

