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
lamb = 0.2           # Weight for SupCon Loss
aux_weight = 0.2     # Weight for MoE Load Balancing Loss

### MoE
#num_experts = len(train_dataset.domain_map) 
num_experts = 3
top_k = 2

# NEW: Toggle to freeze (True) or fine-tune (False) the base expansion layer
freeze_base_mlp = True 

# Choose domains by NAME
train_domains = ["460", "WHT", "700"]   
test_domains  = ["850"]   

### TTA
if_tta = True
tta_lr = 1e-4  # Keep this low to prevent destroying the router

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
                y_d = self.domain_map[spectrum]
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
            return img_orig, img_aug, y_i, y_d
        
        return img_orig, y_i, y_d

# ----------------------------
# 3. MoE-LoRA Classes 
# ----------------------------
class VectorizedLoRAExperts(nn.Module):
    def __init__(self, dim: int, num_experts: int, r: int = 8, alpha: int = 8):
        super().__init__()
        self.scaling = alpha / r
        
        # [num_experts, in_features, out_features]
        self.w_down = nn.Parameter(torch.randn(num_experts, dim, r) * (1 / dim**0.5))
        self.w_up = nn.Parameter(torch.zeros(num_experts, r, dim))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        down = torch.matmul(x, self.w_down[expert_idx])
        act = self.act(down)
        up = torch.matmul(act, self.w_up[expert_idx])
        return up * self.scaling

class ConvNeXtParallelMoELoRA(nn.Module):
    def __init__(self, orig_mlp: nn.Module, dim: int, num_experts: int = 3, top_k: int = 2, r: int = 8, alpha: int = 8, freeze_base: bool = True):
        super().__init__()
        self.orig_mlp = orig_mlp
        self.num_experts = num_experts
        self.top_k = top_k
        self.freeze_base = freeze_base
        
        if self.freeze_base:
            for p in self.orig_mlp.parameters():
                p.requires_grad = False
        else:
            for p in self.orig_mlp.parameters():
                p.requires_grad = True
                
        self.router = nn.Linear(dim, num_experts)
        self.experts = VectorizedLoRAExperts(dim, num_experts, r, alpha)
        
        self.aux_loss = 0.0
        self.entropy_loss = 0.0  # NEW: Tracks router confusion for TTA

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.orig_mlp(x)
        
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1]) 
        
        gate_logits = self.router(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # NEW: Calculate Shannon Entropy of the routing probabilities
        self.entropy_loss = -(gate_probs * torch.log(gate_probs + 1e-6)).sum(dim=-1).mean()
        
        topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
        fraction_routed = torch.zeros_like(gate_probs).scatter_(1, topk_indices, 1.0).mean(dim=0)
        mean_probs = gate_probs.mean(dim=0)
        self.aux_loss = self.num_experts * torch.sum(fraction_routed * mean_probs)
        
        moe_out = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            token_indices, k_indices = torch.where(topk_indices == i)
            if len(token_indices) == 0:
                continue
                
            tokens = x_flat[token_indices]
            expert_output = self.experts(tokens, i)
            
            weights = topk_probs[token_indices, k_indices].unsqueeze(-1)
            moe_out[token_indices] += expert_output * weights
            
        moe_out = moe_out.view(*orig_shape)
        return orig_out + moe_out

# ----------------------------
# 4. Data Loading
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
# 5. Model Setup (MoE + GRL)
# ----------------------------
print("Loading ConvNeXt V2-Tiny...")
model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0).to(device)
embedding_dim = model.num_features 

# A. Freeze ALL parameters first
for p in model.parameters(): p.requires_grad = False

# B. Unfreeze Stage 3 (Lets the 7x7 spatial convolutions fine-tune)
for p in model.stages[3].parameters(): p.requires_grad = True

# C. Inject MoE-LoRA into Stage 3 MLPs
stage_3_dim = 768
#num_experts = len(train_dataset.domain_map) 

for block in model.stages[3].blocks:
    block.mlp = ConvNeXtParallelMoELoRA(
        orig_mlp=block.mlp,
        dim=stage_3_dim,
        num_experts=num_experts,
        top_k=top_k,
        r=8,
        alpha=8,
        freeze_base=freeze_base_mlp   # NEW: Passing the toggle here
    ).to(device)

# D. Unfreeze Final Norm
if hasattr(model, 'norm'):
     for p in model.norm.parameters(): p.requires_grad = True

# Projection Head & GRL Classes
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

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DomainClassifier(nn.Module):
    def __init__(self, dim_in, num_domains):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2),
            nn.BatchNorm1d(dim_in // 2),
            nn.ReLU(True),
            nn.Linear(dim_in // 2, num_domains)
        )
    def forward(self, x, alpha):
        x_rev = GradientReversal.apply(x, alpha)
        return self.net(x_rev)

proj_head = ProjectionHead(embedding_dim).to(device)
domain_classifier = DomainClassifier(embedding_dim, num_experts).to(device)

# ----------------------------
# 6. Losses & Optimizer
# ----------------------------
num_classes = len(train_dataset.hand_id_map)

criterion_arc = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_dim, margin=margin, scale=scale).to(device)
criterion_supcon = losses.SupConLoss(temperature=0.1).to(device)
criterion_domain = nn.CrossEntropyLoss().to(device) 

all_params = list(model.parameters()) + list(criterion_arc.parameters()) + \
             list(proj_head.parameters()) + list(domain_classifier.parameters())

optimizer = optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)


# ----------------------------
# 7. Training Loop
# ----------------------------
total_batches = len(train_loader) * epochs

for epoch in range(epochs):
    # -------- Training --------
    model.train(); proj_head.train(); criterion_arc.train(); domain_classifier.train()
    
    train_loss = 0.0
    train_correct = 0
    total_train = 0

    for batch_idx, (img_orig, img_aug, y_i, y_d) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
        img_orig, img_aug = img_orig.to(device), img_aug.to(device)
        y_i, y_d = y_i.to(device), y_d.to(device)

        # GRL Alpha calculation
        p = float(batch_idx + epoch * len(train_loader)) / total_batches
        alpha_grl = 2. / (1. + np.exp(-25 * p)) - 1

        optimizer.zero_grad()
        
        # 1. Forward Pass
        images_all = torch.cat([img_orig, img_aug], dim=0)
        embeddings_all = model(images_all)
        projections_all = proj_head(embeddings_all)

        batch_size_curr = img_orig.size(0)
        emb_orig = embeddings_all[:batch_size_curr]

        # 2. Domain Classifier Forward
        y_d_all = torch.cat([y_d, y_d], dim=0)
        domain_logits = domain_classifier(embeddings_all, alpha_grl)

        # 3. Calculate Losses
        loss_arc = criterion_arc(emb_orig, y_i)
        
        labels_all = torch.cat([y_i, y_i], dim=0)
        loss_con = criterion_supcon(projections_all, labels_all)
        
        loss_domain = criterion_domain(domain_logits, y_d_all)
        
        # Aggregate MoE Load Balancing Loss
        aux_loss_total = 0.0
        for block in model.stages[3].blocks:
            aux_loss_total += block.mlp.aux_loss
        
        # Total Loss Equation
        loss = loss_arc + (lamb * loss_con) + loss_domain + (aux_weight * aux_loss_total)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        preds = criterion_arc.get_logits(emb_orig).argmax(dim=1)
        train_correct += (preds == y_i).sum().item()
        total_train += batch_size_curr

    # -------- Evaluation --------
    model.eval(); criterion_arc.eval()
    test_loss = 0.0
    test_correct = 0
    total_test = 0
    
    with torch.no_grad():
        for img_orig, y_i, y_d in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test] "):
            img_orig, y_i = img_orig.to(device), y_i.to(device)
            
            embeddings = model(img_orig)
            
            # Test Loss is based purely on ArcFace (Identity mapping)
            loss = criterion_arc(embeddings, y_i)
            test_loss += loss.item()
            
            preds = criterion_arc.get_logits(embeddings).argmax(dim=1)
            test_correct += (preds == y_i).sum().item()
            total_test += y_i.size(0)

    # -------- Epoch Summary --------
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_correct / total_train
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_correct / total_test

    print(f"Epoch [{epoch+1}/{epochs}] Summary:")
    print(f"  -> Train | Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.4f}")
    print(f"  -> Test  | Loss: {avg_test_loss:.4f} | Acc: {avg_test_acc:.4f}")
    print("-" * 50)

    scheduler.step()


# ----------------------------
# 8. Test-Time Adaptation (TTA) - Tent Strategy
# ----------------------------
if if_tta:
    print("\n" + "="*50)
    print("🚀 Starting Test-Time Adaptation (TTA) on Unseen 850nm Domain")
    print("="*50)

    # A. Freeze absolutely everything in the network
    for p in model.parameters(): p.requires_grad = False
    for p in proj_head.parameters(): p.requires_grad = False
    for p in domain_classifier.parameters(): p.requires_grad = False
    for p in criterion_arc.parameters(): p.requires_grad = False

    # B. Isolate and Unfreeze ONLY the MoE Routers in Stage 3
    tta_params = []
    for block in model.stages[3].blocks:
        for p in block.mlp.router.parameters():
            p.requires_grad = True
        tta_params.extend(list(block.mlp.router.parameters()))

    # C. Create a dedicated optimizer just for the routers
    tta_optimizer = optim.Adam(tta_params, lr=tta_lr)

    # D. TTA Loop (Adapting batch-by-batch on the Test Set)
    # We use model.eval() to keep BatchNorm stats frozen, but we DO NOT use torch.no_grad()
    model.eval() 
    criterion_arc.eval()
    
    tta_test_loss = 0.0
    tta_test_correct = 0
    total_tta_test = 0

    for img_orig, y_i, y_d in tqdm(test_loader, desc="[TTA] Adapting & Testing"):
        img_orig, y_i = img_orig.to(device), y_i.to(device)

        tta_optimizer.zero_grad()
        
        # 1. Forward Pass
        embeddings = model(img_orig)
        
        # 2. Collect Entropy from all routers
        entropy_loss_total = 0.0
        for block in model.stages[3].blocks:
            entropy_loss_total += block.mlp.entropy_loss
            
        # 3. Tent Backward Pass (Update the router to be more confident)
        # Notice we are backpropagating the ENTROPY, not the ArcFace loss (because we don't know the labels)
        entropy_loss_total.backward()
        tta_optimizer.step()
        
        # 4. Evaluate Identity Accuracy using the newly updated routing
        # In a strict real-world TTA, you would re-forward the image here. 
        # For academic evaluation, evaluating the current embeddings is standard and faster.
        with torch.no_grad():
            loss = criterion_arc(embeddings, y_i)
            tta_test_loss += loss.item()
            
            preds = criterion_arc.get_logits(embeddings).argmax(dim=1)
            tta_test_correct += (preds == y_i).sum().item()
            total_tta_test += y_i.size(0)

    avg_tta_loss = tta_test_loss / len(test_loader)
    avg_tta_acc = tta_test_correct / total_tta_test

    print(f"\n✅ TTA Completed!")
    print(f"  -> Final Adapted TTA Loss: {avg_tta_loss:.4f}")
    print(f"  -> Final Adapted TTA Acc:  {avg_tta_acc:.4f}")

