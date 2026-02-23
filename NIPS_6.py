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
aux_weight = 0.2     # Weight for MoE MLP Load Balancing Loss
norm_weight = 0.2    # Weight for MoE LayerNorm Routing Loss

### Architectural Toggles
num_experts = 3
top_k = 2

# MASTER TOGGLES
use_moe_mlp = False          
use_moe_stage3_norm = True  
use_moe_final_norm = True  

use_grl = True              

freeze_base_mlp = False          
freeze_base_stage3_norm = True 
freeze_base_final_norm = True  

# Choose domains by NAME
train_domains = ["460", "WHT", "700"]   
test_domains  = ["850"]          

### Test-Time Strategies Configuration
run_continuous_tta = True
run_episodic_tta = True
run_ttaug = True

tta_lr = 1e-4  
episodic_steps = 15 # Number of optimization steps per batch for Episodic TTA
ttaug_size = 4      # Number of augmented views to generate per test sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 1. Transforms
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

# GPU-Accelerated Test-Time Augmentation Pipeline (Tensor Safe)
ttaug_pipeline = transforms.Compose([
    transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
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
# 3. Custom Modules 
# ----------------------------
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
            
        router_dim = normalized_shape[0] if isinstance(normalized_shape, (tuple, list)) else normalized_shape
        self.router = nn.Linear(router_dim, num_domains)
        self.router_logits = None 

    def forward(self, x):
        orig_out = self.orig_norm(x)
        x_pooled = x.mean(dim=(1, 2)) if x.dim() == 4 else x
        self.router_logits = self.router(x_pooled.detach()) 
        weights = F.softmax(self.router_logits, dim=-1) 

        moe_out = 0
        for i in range(self.num_domains):
            w_i = weights[:, i]
            w_i = w_i.view(-1, 1, 1, 1) if x.dim() == 4 else w_i.view(-1, 1)
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
                
        self.router = nn.Linear(dim, num_experts)
        self.experts = VectorizedLoRAExperts(dim, num_experts, r, alpha)
        self.aux_loss = 0.0
        self.entropy_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.orig_mlp(x)
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1]) 
        
        gate_logits = self.router(x_flat.detach())
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        self.entropy_loss = -(gate_probs * torch.log(gate_probs + 1e-6)).sum(dim=-1).mean()
        
        topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
        fraction_routed = torch.zeros_like(gate_probs).scatter_(1, topk_indices, 1.0).mean(dim=0)
        mean_probs = gate_probs.mean(dim=0)
        self.aux_loss = self.num_experts * torch.sum(fraction_routed * mean_probs)
        
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
# 5. Model Setup 
# ----------------------------
print("Loading ConvNeXt V2-Tiny...")
model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0).to(device)
embedding_dim = model.num_features 

for p in model.parameters(): p.requires_grad = False
for p in model.stages[3].parameters(): p.requires_grad = True

stage_3_dim = 768
for block in model.stages[3].blocks:
    if use_moe_mlp:
        block.mlp = ConvNeXtParallelMoELoRA(orig_mlp=block.mlp, dim=stage_3_dim, num_experts=num_experts, top_k=top_k, r=8, alpha=8, freeze_base=freeze_base_mlp).to(device)
    else:
        for p in block.mlp.parameters(): p.requires_grad = not freeze_base_mlp
    
    if use_moe_stage3_norm and hasattr(block, 'norm'):
        orig_shape = getattr(block.norm, 'normalized_shape', stage_3_dim)
        eps = getattr(block.norm, 'eps', 1e-6)
        block.norm = ParallelMoELayerNorm(orig_norm=block.norm, normalized_shape=orig_shape, num_domains=num_experts, eps=eps, freeze_base=freeze_base_stage3_norm).to(device)
    elif hasattr(block, 'norm'):
        for p in block.norm.parameters(): p.requires_grad = not freeze_base_stage3_norm

if hasattr(model, 'norm'):
    if use_moe_final_norm:
        orig_shape = getattr(model.norm, 'normalized_shape', embedding_dim)
        eps = getattr(model.norm, 'eps', 1e-6)
        model.norm = ParallelMoELayerNorm(orig_norm=model.norm, normalized_shape=orig_shape, num_domains=num_experts, eps=eps, freeze_base=freeze_base_final_norm).to(device)
    else:
        for p in model.norm.parameters(): p.requires_grad = not freeze_base_final_norm

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
# 6. Losses & Optimizer
# ----------------------------
num_classes = len(train_dataset.hand_id_map)
criterion_arc = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_dim, margin=margin, scale=scale).to(device)
criterion_supcon = losses.SupConLoss(temperature=0.1).to(device)

all_params = list(model.parameters()) + list(criterion_arc.parameters()) + list(proj_head.parameters())
if use_grl: all_params += list(domain_classifier.parameters())

optimizer = optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

# ----------------------------
# 7. Training Loop
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
        
        aux_loss_total = 0.0
        if use_moe_mlp:
            for block in model.stages[3].blocks: aux_loss_total += block.mlp.aux_loss
            
        norm_routing_loss = 0.0; norm_count = 0
        for module in model.modules():
            if isinstance(module, ParallelMoELayerNorm) and module.router_logits is not None:
                norm_routing_loss += F.cross_entropy(module.router_logits, y_d_all)
                norm_count += 1
        if norm_count > 0: norm_routing_loss = norm_routing_loss / norm_count
        
        loss = loss_arc + (lamb * loss_con) + loss_domain + (aux_weight * aux_loss_total) + (norm_weight * norm_routing_loss)
        
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

# --------------------------------------------------------------------------------
# 8. Test-Time Adaptation & Augmentation (TTA / TTAug)
# --------------------------------------------------------------------------------

print("\nCreating Clean Model Snapshot for Evaluation...")
clean_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# NEW HELPER: Dynamically unfreezes params based on TTA Strategy
def get_tta_params(target_model, unfreeze_experts=False):
    for p in target_model.parameters(): p.requires_grad = False
    for p in proj_head.parameters(): p.requires_grad = False
    if use_grl:
        for p in domain_classifier.parameters(): p.requires_grad = False
    for p in criterion_arc.parameters(): p.requires_grad = False

    tta_params = []
    
    if use_moe_mlp:
        for block in target_model.stages[3].blocks:
            for p in block.mlp.router.parameters(): p.requires_grad = True
            tta_params.extend(list(block.mlp.router.parameters()))
            # NEW: Allow MLP experts to adapt if requested
            if unfreeze_experts:
                for p in block.mlp.experts.parameters(): p.requires_grad = True
                tta_params.extend(list(block.mlp.experts.parameters()))
            
    for module in target_model.modules():
        if isinstance(module, ParallelMoELayerNorm):
            for p in module.router.parameters(): p.requires_grad = True
            tta_params.extend(list(module.router.parameters()))
            # NEW: Allow Norm experts to adapt if requested
            if unfreeze_experts:
                for p in module.norms.parameters(): p.requires_grad = True
                tta_params.extend(list(module.norms.parameters()))
            
    return tta_params

# ==========================================
# APPROACH A: Continuous TTA (Drift)
# ==========================================
if run_continuous_tta:
    print("\n" + "="*50)
    print("🚀 APPROACH A: Continuous TTA (Tent - Routers Only)")
    print("="*50)
    model.load_state_dict({k: v.to(device) for k, v in clean_state_dict.items()})
    
    # Continuous TTA is kept strictly to Routers to prevent catastrophic forgetting
    tta_params = get_tta_params(model, unfreeze_experts=False)
    tta_optimizer = optim.Adam(tta_params, lr=tta_lr)

    model.eval(); criterion_arc.eval()
    tta_test_correct = 0; total_tta_test = 0

    for img_orig, y_i, y_d in tqdm(test_loader, desc="[Continuous TTA]"):
        img_orig, y_i = img_orig.to(device), y_i.to(device)

        tta_optimizer.zero_grad()
        embeddings = model(img_orig)
        
        entropy_loss_total = 0.0
        if use_moe_mlp:
            for block in model.stages[3].blocks: entropy_loss_total += block.mlp.entropy_loss
                
        if isinstance(entropy_loss_total, torch.Tensor):
            entropy_loss_total.backward()
            tta_optimizer.step()
        
        with torch.no_grad():
            updated_embeddings = model(img_orig)
            preds = criterion_arc.get_logits(updated_embeddings).argmax(dim=1)
            tta_test_correct += (preds == y_i).sum().item()
            total_tta_test += y_i.size(0)

    print(f"\n✅ Continuous TTA Acc: {tta_test_correct / total_tta_test:.4f}")

# ==========================================
# APPROACH B: Episodic TTA (Batch Reset)
# ==========================================
if run_episodic_tta:
    print("\n" + "="*50)
    print("🚀 APPROACH B: Episodic TTA (Routers + Experts via Identity Entropy)")
    print("="*50)
    episodic_test_correct = 0; total_episodic_test = 0
    
    for img_orig, y_i, y_d in tqdm(test_loader, desc="[Episodic TTA]"):
        img_orig, y_i = img_orig.to(device), y_i.to(device)

        model.load_state_dict({k: v.to(device) for k, v in clean_state_dict.items()})
        
        # UPDATED: We unfreeze BOTH the routers and the LoRA/Norm experts here
        tta_params = get_tta_params(model, unfreeze_experts=True)
        tta_optimizer = optim.Adam(tta_params, lr=tta_lr)
        
        model.eval() 
        for _ in range(episodic_steps):
            tta_optimizer.zero_grad()
            embeddings = model(img_orig)
            
            # 1. Identity Entropy (This pulls gradients through the Experts)
            logits = criterion_arc.get_logits(embeddings)
            probs = F.softmax(logits, dim=1)
            identity_entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
            
            # 2. Router Entropy (This keeps the Routers stable during updates)
            router_entropy = 0.0
            if use_moe_mlp:
                for block in model.stages[3].blocks: router_entropy += block.mlp.entropy_loss
                    
            # 3. Combine losses so both sub-systems update cooperatively
            total_batch_loss = identity_entropy + router_entropy
            total_batch_loss.backward()
            tta_optimizer.step()
                
        with torch.no_grad():
            final_batch_embeddings = model(img_orig)
            preds = criterion_arc.get_logits(final_batch_embeddings).argmax(dim=1)
            episodic_test_correct += (preds == y_i).sum().item()
            total_episodic_test += y_i.size(0)

    print(f"\n✅ Episodic TTA Acc: {episodic_test_correct / total_episodic_test:.4f}")

# ==========================================
# APPROACH C: Test-Time Augmentation (TTAug)
# ==========================================
if run_ttaug:
    print("\n" + "="*50)
    print(f"🚀 APPROACH C: Test-Time Augmentation (Size: {ttaug_size} Augs)")
    print("="*50)
    
    # Restore clean state (No weights will be updated here)
    model.load_state_dict({k: v.to(device) for k, v in clean_state_dict.items()})
    model.eval(); criterion_arc.eval()
    
    embed_correct = 0; vote_correct = 0; total_ttaug_test = 0

    with torch.no_grad():
        for img_orig, y_i, y_d in tqdm(test_loader, desc="[TTAug Evaluate]"):
            img_orig, y_i = img_orig.to(device), y_i.to(device)
            B, C, H, W = img_orig.shape

            # 1. Create original + augmented views directly on GPU
            views = [img_orig]
            for _ in range(ttaug_size):
                views.append(ttaug_pipeline(img_orig))
            
            # Combine into single batch: Shape [Batch * Total_Views, C, H, W]
            total_views = 1 + ttaug_size
            all_views = torch.stack(views, dim=1).view(-1, C, H, W)

            # 2. Forward Pass for all views simultaneously
            all_embeddings = model(all_views) # Shape: [B*Total_Views, Embedding_Dim]
            _, D = all_embeddings.shape
            
            # Reshape back to [Batch, Total_Views, Embedding_Dim]
            all_embeddings = all_embeddings.view(B, total_views, D)

            # ----------------------------------------------------
            # Strategy 1: Embedding Average (Feature Smoothing)
            # ----------------------------------------------------
            avg_embedding = all_embeddings.mean(dim=1)
            logits_embed = criterion_arc.get_logits(avg_embedding)
            preds_embed = logits_embed.argmax(dim=1)
            embed_correct += (preds_embed == y_i).sum().item()

            # ----------------------------------------------------
            # Strategy 2: Majority Vote (Decision Smoothing)
            # ----------------------------------------------------
            logits_all = criterion_arc.get_logits(all_embeddings.view(-1, D))
            preds_all = logits_all.argmax(dim=1).view(B, total_views)
            
            preds_vote, _ = torch.mode(preds_all, dim=1)
            vote_correct += (preds_vote == y_i).sum().item()

            total_ttaug_test += B

    print(f"\n✅ TTAug (Embedding Avg) Acc: {embed_correct / total_ttaug_test:.4f}")
    print(f"✅ TTAug (Majority Vote) Acc: {vote_correct / total_ttaug_test:.4f}")
