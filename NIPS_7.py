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
# 1. Configuration
# ----------------------------
batch_size = 32   
margin = 0.3
scale = 16
lr = 1e-3
weight_decay = 1e-4
epochs = 100
lamb = 0.2           # Weight for SupCon Loss
aux_weight = 0.2     # Weight for MoE MLP Load Balancing Loss
norm_weight = 1.0    # HIGH: The Scout MUST learn to classify domains perfectly.

### Architectural Toggles
num_experts = 3
top_k = 2

# MASTER TOGGLES
use_moe_mlp = True          
use_moe_stage3_norm = False 
use_moe_final_norm = False  

use_grl = True              

freeze_base_mlp = True          
freeze_base_stage3_norm = False 
freeze_base_final_norm = False  

# Choose domains by NAME
train_domains = ["460", "WHT", "700"]   
test_domains  = ["850"]          

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Transforms
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
# 3. Dataset Class
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
                if not fname.lower().endswith(".jpg"): continue
                parts = fname[:-4].split("_")
                if len(parts) != 4: continue
                subject_id, hand, spectrum, iteration = parts
                if spectrum not in target_domains: continue

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
# 4. Custom Modules (Global Scout Logic)
# ----------------------------

# --- NEW: The Global Domain Scout ---
class GlobalDomainRouter(nn.Module):
    def __init__(self, num_domains=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Linear(32, num_domains)

    def forward(self, x):
        return self.classifier(self.features(x))

# --- Parallel MoE LayerNorm ---
class ParallelMoELayerNorm(nn.Module):
    def __init__(self, orig_norm: nn.Module, normalized_shape, num_domains=3, eps=1e-6, freeze_base=True):
        super().__init__()
        self.orig_norm = orig_norm
        self.num_domains = num_domains
        self.freeze_base = freeze_base
        
        if self.freeze_base:
            for p in self.orig_norm.parameters(): p.requires_grad = False
        else:
            for p in self.orig_norm.parameters(): p.requires_grad = True
                
        self.norms = nn.ModuleList([nn.LayerNorm(normalized_shape, eps=eps) for _ in range(num_domains)])
        for norm in self.norms:
            nn.init.zeros_(norm.weight) 
            nn.init.zeros_(norm.bias)   

    def forward(self, x, routing_weights):
        # x is [B, H, W, C] due to wrapper permute logic
        orig_out = self.orig_norm(x)
        moe_out = 0
        for i in range(self.num_domains):
            w_i = routing_weights[:, i]
            # Reshape w_i [B] to [B, 1, 1, 1] to broadcast against [B, H, W, C]
            if x.dim() == 4:
                w_i = w_i.view(-1, 1, 1, 1) 
            else:
                w_i = w_i.view(-1, 1)       
            moe_out += w_i * self.norms[i](x)
        return orig_out + moe_out
        
class VectorizedLoRAExperts(nn.Module):
    def __init__(self, dim: int, num_experts: int, r: int = 8, alpha: int = 8):
        super().__init__()
        self.scaling = alpha / r
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
            for p in self.orig_mlp.parameters(): p.requires_grad = False
        else:
            for p in self.orig_mlp.parameters(): p.requires_grad = True
                
        self.experts = VectorizedLoRAExperts(dim, num_experts, r, alpha)

    def forward(self, x: torch.Tensor, routing_info) -> torch.Tensor:
        # x is [B, H, W, C] here
        gate_probs, topk_probs, topk_indices = routing_info
        orig_out = self.orig_mlp(x)
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1]) 
        
        moe_out = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            token_indices, k_indices = torch.where(topk_indices == i)
            if len(token_indices) == 0: continue
            
            tokens = x_flat[token_indices]
            expert_output = self.experts(tokens, i)
            
            weights = topk_probs[token_indices, k_indices].unsqueeze(-1)
            moe_out[token_indices] += expert_output * weights
            
        moe_out = moe_out.view(*orig_shape)
        return orig_out + moe_out

# ----------------------------
# 5. Integrated Model Wrapper (Scout + Backbone)
# ----------------------------
class IntegratedMoEModel(nn.Module):
    def __init__(self, backbone, scout, num_experts, top_k):
        super().__init__()
        self.backbone = backbone
        self.scout = scout
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss = 0.0
        self.scout_logits = None 

    def forward(self, x):
        # 1. SCOUT DECISION
        self.scout_logits = self.scout(x)
        gate_probs = F.softmax(self.scout_logits, dim=-1)
        
        topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
        fraction_routed = torch.zeros_like(gate_probs).scatter_(1, topk_indices, 1.0).mean(dim=0)
        mean_probs = gate_probs.mean(dim=0)
        self.aux_loss = self.num_experts * torch.sum(fraction_routed * mean_probs)
        
        mlp_routing_info = (gate_probs, topk_probs, topk_indices)
        norm_routing_weights = gate_probs 

        # 3. DISTRIBUTE DECISIONS TO BLOCKS
        # Iterate over the wrappers to inject info into the real blocks
        for wrapper in self.backbone.stages[3].blocks:
            real_block = wrapper.block # Access the underlying block
            
            if hasattr(real_block.mlp, 'forward'):
                real_block.mlp.current_routing_info = mlp_routing_info
            if hasattr(real_block.norm, 'forward'):
                real_block.norm.current_routing_weights = norm_routing_weights
                
        if hasattr(self.backbone, 'norm') and isinstance(self.backbone.norm, ParallelMoELayerNorm):
             self.backbone.norm.current_routing_weights = norm_routing_weights

        return self.backbone(x)

# ----------------------------
# 6. Data Loading 
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
# 7. Model Setup 
# ----------------------------
print("Loading ConvNeXt V2-Tiny...")
base_model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0).to(device)
embedding_dim = base_model.num_features 

for p in base_model.parameters(): p.requires_grad = False
for p in base_model.stages[3].parameters(): p.requires_grad = True

stage_3_dim = 768

# Helper Block to inject routing info during forward pass - FIXED for Shape
class RoutedConvNeXtBlock(nn.Module):
    def __init__(self, original_block):
        super().__init__()
        self.block = original_block
    def forward(self, x):
        shortcut = x
        x = self.block.conv_dw(x)
        
        if self.block.use_conv_mlp:
            if isinstance(self.block.norm, ParallelMoELayerNorm):
                x = self.block.norm(x, self.block.norm.current_routing_weights)
            else:
                x = self.block.norm(x)
            
            if isinstance(self.block.mlp, ConvNeXtParallelMoELoRA):
                x = self.block.mlp(x, self.block.mlp.current_routing_info)
            else:
                x = self.block.mlp(x)
        else:
            # FIX: Permute for standard MLP blocks [NCHW -> NHWC]
            x = x.permute(0, 2, 3, 1) 
            
            if isinstance(self.block.norm, ParallelMoELayerNorm):
                x = self.block.norm(x, self.block.norm.current_routing_weights)
            else:
                x = self.block.norm(x)
            
            if isinstance(self.block.mlp, ConvNeXtParallelMoELoRA):
                x = self.block.mlp(x, self.block.mlp.current_routing_info)
            else:
                x = self.block.mlp(x)
            
            # FIX: Permute back [NHWC -> NCHW]
            x = x.permute(0, 3, 1, 2)
                
        if self.block.gamma is not None: x = self.block.gamma * x
        x = self.block.drop_path(x)
        return x + shortcut

for i, block in enumerate(base_model.stages[3].blocks):
    if use_moe_mlp:
        block.mlp = ConvNeXtParallelMoELoRA(block.mlp, stage_3_dim, num_experts, top_k, 8, 8, freeze_base_mlp).to(device)
    
    if use_moe_stage3_norm:
        orig_shape = getattr(block.norm, 'normalized_shape', stage_3_dim)
        eps = getattr(block.norm, 'eps', 1e-6)
        block.norm = ParallelMoELayerNorm(block.norm, orig_shape, num_experts, eps, freeze_base_stage3_norm).to(device)
        
    base_model.stages[3].blocks[i] = RoutedConvNeXtBlock(block)

if use_moe_final_norm and hasattr(base_model, 'norm'):
    orig_shape = getattr(base_model.norm, 'normalized_shape', embedding_dim)
    eps = getattr(base_model.norm, 'eps', 1e-6)
    base_model.norm = ParallelMoELayerNorm(base_model.norm, orig_shape, num_experts, eps, freeze_base_final_norm).to(device)

scout = GlobalDomainRouter(num_experts).to(device)
model = IntegratedMoEModel(base_model, scout, num_experts, top_k).to(device)

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, dim_out=128):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, dim_out))
    def forward(self, x): return F.normalize(self.head(x), dim=1)

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
        self.net = nn.Sequential(nn.Linear(dim_in, dim_in // 2), nn.BatchNorm1d(dim_in // 2), nn.ReLU(True), nn.Linear(dim_in // 2, num_domains))
    def forward(self, x, alpha):
        return self.net(GradientReversal.apply(x, alpha))

proj_head = ProjectionHead(embedding_dim).to(device)

if use_grl:
    domain_classifier = DomainClassifier(embedding_dim, num_experts).to(device)
    criterion_domain = nn.CrossEntropyLoss().to(device)
else:
    domain_classifier = None; criterion_domain = None

# ----------------------------
# 8. Losses & Optimizer
# ----------------------------
num_classes = len(train_dataset.hand_id_map) 
criterion_arc = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_dim, margin=margin, scale=scale).to(device)
criterion_supcon = losses.SupConLoss(temperature=0.1).to(device)

all_params = list(model.parameters()) + list(criterion_arc.parameters()) + list(proj_head.parameters())
if use_grl: all_params += list(domain_classifier.parameters())

optimizer = optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

# ----------------------------
# 9. Training Loop
# ----------------------------
total_batches = len(train_loader) * epochs

for epoch in range(epochs):
    model.train(); proj_head.train(); criterion_arc.train()
    if use_grl: domain_classifier.train()
    
    train_loss = 0.0; train_correct = 0; total_train = 0

    for batch_idx, (img_orig, img_aug, y_i, y_d) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
        img_orig, img_aug = img_orig.to(device), img_aug.to(device)
        y_i, y_d = y_i.to(device), y_d.to(device)

        optimizer.zero_grad()
        images_all = torch.cat([img_orig, img_aug], dim=0)
        y_d_all = torch.cat([y_d, y_d], dim=0)

        embeddings_all = model(images_all)
        projections_all = proj_head(embeddings_all)

        batch_size_curr = img_orig.size(0)
        emb_orig = embeddings_all[:batch_size_curr]

        loss_arc = criterion_arc(emb_orig, y_i)
        labels_all = torch.cat([y_i, y_i], dim=0)
        loss_con = criterion_supcon(projections_all, labels_all)
        
        loss_domain = 0.0
        if use_grl:
            p = float(batch_idx + epoch * len(train_loader)) / total_batches
            alpha_grl = 2. / (1. + np.exp(-25 * p)) - 1
            domain_logits = domain_classifier(embeddings_all, alpha_grl)
            loss_domain = criterion_domain(domain_logits, y_d_all)
        
        norm_routing_loss = F.cross_entropy(model.scout_logits, y_d_all)
        aux_loss = model.aux_loss
        
        loss = loss_arc + (lamb * loss_con) + loss_domain + (aux_weight * aux_loss) + (norm_weight * norm_routing_loss)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = criterion_arc.get_logits(emb_orig).argmax(dim=1)
        train_correct += (preds == y_i).sum().item()
        total_train += batch_size_curr

    # -------- Evaluation --------
    model.eval(); criterion_arc.eval()
    test_loss = 0.0; test_correct = 0; total_test = 0
    
    with torch.no_grad():
        for img_orig, y_i, y_d in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test] "):
            img_orig, y_i = img_orig.to(device), y_i.to(device)
            embeddings = model(img_orig)
            loss = criterion_arc(embeddings, y_i)
            test_loss += loss.item()
            preds = criterion_arc.get_logits(embeddings).argmax(dim=1)
            test_correct += (preds == y_i).sum().item()
            total_test += y_i.size(0)

    print(f"Epoch [{epoch+1}/{epochs}] Summary:")
    print(f"  -> Train | Loss: {train_loss/len(train_loader):.4f} | Acc: {train_correct/total_train:.4f}")
    print(f"  -> Test  | Loss: {test_loss/len(test_loader):.4f} | Acc: {test_correct/total_test:.4f}")
    print("-" * 50)
    scheduler.step()
