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
import random
from PIL import Image, ImageFilter, ImageEnhance
from pytorch_metric_learning import losses

# ----------------------------
# 1. Configuration & Toggles
# ----------------------------
batch_size = 32      
margin = 0.3
scale = 16
lr = 1e-3
weight_decay = 1e-4
epochs = 200
lamb = 0.2           # SupCon Weight
aux_weight = 0.2     # MoE Balance Weight
norm_weight = 1.0    # Scout Weight (Only used if Global Scout is ON)

# --- EXPANSION MODE ---
augmentation_expansion_mode = '4x' 

# --- VISIBILITY TOGGLE ---
use_aug_only_for_supcon = False 

# --- AUGMENTATION SETTINGS ---
use_dynamic_beta = True

# --- ARCHITECTURE TOGGLES ---
# Master Switch: 
# True  = Global Scout (Your Method: One brain routing all layers)
# False = Normal MoE (Baseline: Each layer routes itself independently)
use_global_scout = True     

num_experts = 3
top_k = 2

use_moe_mlp = True          
use_moe_stage3_norm = False 
use_moe_final_norm = False

use_grl = True               

freeze_base_mlp = True          
freeze_base_stage3_norm = False 
freeze_base_final_norm = False  

train_domains = ["460", "WHT"]   
test_domains  = ["700"]          

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Augmentation Logic
# ----------------------------
general_aug = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2.0)], p=0.2),
    transforms.RandomApply([transforms.RandomAutocontrast()], p=0.2),
])

def label_guided_fft_mixup(x, labels):
    B, C, H, W = x.shape
    fft = torch.fft.fft2(x, dim=(-2, -1))
    amp, pha = torch.abs(fft), torch.angle(fft)
    amp_shifted = torch.fft.fftshift(amp, dim=(-2, -1))
    
    sorted_idx = torch.argsort(labels)
    target_indices = torch.roll(sorted_idx, shifts=B // 2, dims=0)
    amp_shifted_trg = amp_shifted[target_indices]
    
    if use_dynamic_beta:
        beta = np.random.uniform(0.01, 0.20)
    else:
        beta = 0.15
    
    b = int(np.floor(np.amin((H, W)) * beta))
    c_h, c_w = int(np.floor(H / 2.0)), int(np.floor(W / 2.0))
    
    if b > 0:
        amp_shifted[..., c_h-b:c_h+b, c_w-b:c_w+b] = amp_shifted_trg[..., c_h-b:c_h+b, c_w-b:c_w+b]
    
    amp_mixed = torch.fft.ifftshift(amp_shifted, dim=(-2, -1))
    x_aug = torch.fft.ifft2(amp_mixed * torch.exp(1j * pha), dim=(-2, -1)).real
    return torch.clamp(x_aug, 0, 1)

# ----------------------------
# 3. Dataset Class
# ----------------------------
class CASIA_MS_Dataset(Dataset):
    def __init__(self, data_path, target_domains, is_train=True):
        self.samples = []
        self.hand_id_map = {}
        self.domain_map = {d: i for i, d in enumerate(target_domains)}
        self.is_train = is_train
        self.to_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.general_aug = general_aug
        
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
                self.samples.append((img_path, self.hand_id_map[hand_id], y_d, spectrum))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, y_i, y_d, spectrum = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_orig = self.to_tensor(img)

        if self.is_train:
            img_aug_pil = self.general_aug(img)
            return img_orig, self.to_tensor(img_aug_pil), y_i, y_d
        
        return img_orig, y_i, y_d

# ----------------------------
# 4. Modules & Routers
# ----------------------------
class GlobalDomainRouter(nn.Module):
    def __init__(self, num_domains=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 7, 4, 3), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.classifier = nn.Linear(32, num_domains)
    def forward(self, x): return self.classifier(self.features(x))

class ParallelMoELayerNorm(nn.Module):
    def __init__(self, orig_norm, normalized_shape, num_domains=3, eps=1e-6, freeze_base=True):
        super().__init__()
        self.orig_norm = orig_norm; self.num_domains = num_domains
        if freeze_base: 
            for p in self.orig_norm.parameters(): p.requires_grad = False
        else:
            for p in self.orig_norm.parameters(): p.requires_grad = True
        self.norms = nn.ModuleList([nn.LayerNorm(normalized_shape, eps=eps) for _ in range(num_domains)])
        for n in self.norms: nn.init.zeros_(n.weight); nn.init.zeros_(n.bias)
    def forward(self, x, weights):
        out = self.orig_norm(x)
        moe = 0
        for i in range(self.num_domains):
            w = weights[:, i]
            if x.dim()==4: w = w.view(-1, 1, 1, 1)
            else: w = w.view(-1, 1)
            moe += w * self.norms[i](x)
        return out + moe

class VectorizedLoRAExperts(nn.Module):
    def __init__(self, dim, num_experts, r=8, alpha=8):
        super().__init__()
        self.scaling = alpha/r
        self.w_down = nn.Parameter(torch.randn(num_experts, dim, r)*(1/dim**0.5))
        self.w_up = nn.Parameter(torch.zeros(num_experts, r, dim))
        self.act = nn.GELU()
    def forward(self, x, idx):
        return torch.matmul(self.act(torch.matmul(x, self.w_down[idx])), self.w_up[idx]) * self.scaling

# --- HYBRID EXPERT LAYER (Supports Global OR Local Routing) ---
class ConvNeXtParallelMoELoRA(nn.Module):
    def __init__(self, orig_mlp, dim, num_experts=3, top_k=2, r=8, alpha=8, freeze_base=True, use_global_routing=True):
        super().__init__()
        self.orig_mlp = orig_mlp; self.num_experts = num_experts; self.top_k = top_k
        self.use_global_routing = use_global_routing
        self.local_aux_loss = 0.0 # Store loss if local
        
        if freeze_base: 
            for p in self.orig_mlp.parameters(): p.requires_grad = False
        else:
            for p in self.orig_mlp.parameters(): p.requires_grad = True
            
        self.experts = VectorizedLoRAExperts(dim, num_experts, r, alpha)
        
        # If Local Routing (Normal MoE), we need a router PER LAYER
        if not self.use_global_routing:
            self.router = nn.Linear(dim, num_experts) 

    def forward(self, x, info=None):
        # 1. Determine Routing Info (Gate, TopK_Probs, TopK_Idx)
        if self.use_global_routing:
            # Global Mode: Router info comes from outside (Scout)
            gate, topk_probs, topk_idx = info
        else:
            # Local Mode: Calculate routing based on THIS layer's input
            # x shape in ConvNeXt MLP is (Batch, Height, Width, Channels)
            B, H, W, C = x.shape
            x_flat = x.view(B, -1, C).mean(dim=1) # Global Avg Pool to (B, C)
            
            logits = self.router(x_flat)
            gate = F.softmax(logits, dim=-1)
            
            # TopK logic
            topk_probs, topk_idx = torch.topk(gate, self.top_k, dim=-1)
            topk_probs = topk_probs / (topk_probs.sum(-1, keepdim=True) + 1e-6) # Normalize
            
            # Calculate Load Balancing Loss Locally
            frac = torch.zeros_like(gate).scatter_(1, topk_idx, 1.0).mean(0)
            self.local_aux_loss = self.num_experts * torch.sum(frac * gate.mean(0))

        # 2. Apply Experts
        orig = self.orig_mlp(x); shape = x.shape
        x_flat = x.view(-1, shape[-1]); moe = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            idx, k_idx = torch.where(topk_idx == i)
            if len(idx)==0: continue
            moe[idx] += self.experts(x_flat[idx], i) * topk_probs[idx, k_idx].unsqueeze(-1)
            
        return orig + moe.view(*shape)

class IntegratedMoEModel(nn.Module):
    def __init__(self, backbone, scout, num_experts, top_k):
        super().__init__()
        self.backbone = backbone; self.scout = scout; self.num_experts = num_experts; self.top_k = top_k
        self.aux_loss = 0.0; self.scout_logits = None
    def forward(self, x):
        self.scout_logits = self.scout(x)
        gate = F.softmax(self.scout_logits, dim=-1)
        topk_p, topk_i = torch.topk(gate, self.top_k, dim=-1)
        topk_p = topk_p / (topk_p.sum(-1, keepdim=True)+1e-6)
        frac = torch.zeros_like(gate).scatter_(1, topk_i, 1.0).mean(0)
        self.aux_loss = self.num_experts * torch.sum(frac * gate.mean(0))
        
        info = (gate, topk_p, topk_i)
        for w in self.backbone.stages[3].blocks:
            if hasattr(w.block.mlp, 'forward'): w.block.mlp.current_routing_info = info
            if hasattr(w.block.norm, 'forward'): w.block.norm.current_routing_weights = gate
        
        if hasattr(self.backbone, 'norm') and isinstance(self.backbone.norm, ParallelMoELayerNorm):
             self.backbone.norm.current_routing_weights = gate
             
        return self.backbone(x)

# ----------------------------
# 5. Data Loading & Setup
# ----------------------------
data_path = "/home/pai-ng/Jamal/CASIA-MS-ROI"
print(f"Creating Training Dataset... (Batch Size: {batch_size})")
train_dataset = CASIA_MS_Dataset(data_path, train_domains, is_train=True)
test_dataset  = CASIA_MS_Dataset(data_path, test_domains, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# --- MODEL SETUP ---
print(f"Model Mode: {'MoE + Global Scout' if use_global_scout else 'Normal MoE (Local Routing)'}")

base_model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0).to(device)
embedding_dim = base_model.num_features
stage_3_dim = 768

# Inject MoE layers (works for both modes, but configured differently)
for p in base_model.parameters(): p.requires_grad = False
for p in base_model.stages[3].parameters(): p.requires_grad = True

class RoutedConvNeXtBlock(nn.Module):
    def __init__(self, original_block):
        super().__init__(); self.block = original_block
    def forward(self, x):
        shortcut = x; x = self.block.conv_dw(x)
        if self.block.use_conv_mlp:
            x = self.block.norm(x, self.block.norm.current_routing_weights) if isinstance(self.block.norm, ParallelMoELayerNorm) else self.block.norm(x)
            # Pass 'None' if local routing; layer handles it self.
            info = getattr(self.block.mlp, 'current_routing_info', None)
            x = self.block.mlp(x, info) if isinstance(self.block.mlp, ConvNeXtParallelMoELoRA) else self.block.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.block.norm(x, self.block.norm.current_routing_weights) if isinstance(self.block.norm, ParallelMoELayerNorm) else self.block.norm(x)
            info = getattr(self.block.mlp, 'current_routing_info', None)
            x = self.block.mlp(x, info) if isinstance(self.block.mlp, ConvNeXtParallelMoELoRA) else self.block.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.block.gamma is not None: x = self.block.gamma * x
        return self.block.drop_path(x) + shortcut

for i, block in enumerate(base_model.stages[3].blocks):
    if use_moe_mlp: 
        # INJECT HYBRID EXPERTS
        # use_global_routing=False means Normal MoE (internal Linear router)
        block.mlp = ConvNeXtParallelMoELoRA(block.mlp, stage_3_dim, num_experts, top_k, 8, 8, freeze_base_mlp, use_global_routing=use_global_scout).to(device)
    base_model.stages[3].blocks[i] = RoutedConvNeXtBlock(block)

if use_moe_final_norm and hasattr(base_model, 'norm'):
    orig_shape = getattr(base_model.norm, 'normalized_shape', embedding_dim)
    eps = getattr(base_model.norm, 'eps', 1e-6)
    base_model.norm = ParallelMoELayerNorm(base_model.norm, orig_shape, num_experts, eps, freeze_base_final_norm).to(device)

if use_global_scout:
    scout = GlobalDomainRouter(num_experts).to(device)
    model = IntegratedMoEModel(base_model, scout, num_experts, top_k).to(device)
else:
    # Just the Backbone (with internal local routers)
    model = base_model

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, dim_out=128):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(True), nn.Linear(dim_in, dim_out))
    def forward(self, x): return F.normalize(self.head(x), dim=1)

class GradientReversal(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, x, alpha): ctx.alpha=alpha; return x.view_as(x)
    @staticmethod 
    def backward(ctx, grad_output): return grad_output.neg() * ctx.alpha, None

class DomainClassifier(nn.Module):
    def __init__(self, dim_in, num_domains):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, dim_in//2), nn.BatchNorm1d(dim_in//2), nn.ReLU(True), nn.Linear(dim_in//2, num_domains))
    def forward(self, x, alpha): return self.net(GradientReversal.apply(x, alpha))

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
print(f"Starting Training | Mode: {augmentation_expansion_mode} | SupCon Only Aug: {use_aug_only_for_supcon} | Global Scout: {use_global_scout}")

for epoch in range(epochs):
    model.train(); proj_head.train(); criterion_arc.train()
    if use_grl: domain_classifier.train()
    train_loss = 0.0; train_correct = 0; total_train = 0

    for batch_idx, (img_orig, img_spatial, y_i, y_d) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
        img_orig, img_spatial, y_i, y_d = img_orig.to(device), img_spatial.to(device), y_i.to(device), y_d.to(device)

        # === DATA GENERATION ===
        images_list = [img_orig]; labels_list = [y_i]; domains_list = [y_d]

        # 1. XOR View
        if torch.rand(1) < 0.5:
            img_xor = label_guided_fft_mixup(img_orig, y_d)
        else:
            img_xor = img_spatial

        if augmentation_expansion_mode == '2x':
            images_list.append(img_xor); labels_list.append(y_i); domains_list.append(y_d)
        elif augmentation_expansion_mode == '4x':
            img_fft = label_guided_fft_mixup(img_orig, y_d)
            images_list.extend([img_spatial, img_fft, img_xor])
            labels_list.extend([y_i, y_i, y_i])
            domains_list.extend([y_d, y_d, y_d])

        images_all = torch.cat(images_list, dim=0)
        y_i_all = torch.cat(labels_list, dim=0)
        y_d_all = torch.cat(domains_list, dim=0)

        # === FORWARD ===
        optimizer.zero_grad()
        embeddings_all = model(images_all)
        projections_all = proj_head(embeddings_all)

        batch_curr = img_orig.size(0)
        emb_orig = embeddings_all[:batch_curr]

        # === LOSSES ===
        # 1. Classifiers
        norm_routing_loss = 0.0
        if use_aug_only_for_supcon:
            loss_arc = criterion_arc(emb_orig, y_i)
            # Only calc routing loss if Global Scout is ON
            if use_global_scout:
                norm_routing_loss = F.cross_entropy(model.scout_logits[:batch_curr], y_d)
        else:
            loss_arc = criterion_arc(embeddings_all, y_i_all)
            if use_global_scout:
                norm_routing_loss = F.cross_entropy(model.scout_logits, y_d_all)

        # 2. SupCon
        loss_con = criterion_supcon(projections_all, y_i_all)
        
        # 3. GRL
        loss_domain = 0.0
        if use_grl:
            p = float(batch_idx + epoch * len(train_loader)) / (len(train_loader) * epochs)
            alpha_grl = 2. / (1. + np.exp(-10 * p)) - 1
            loss_domain = criterion_domain(domain_classifier(emb_orig, alpha_grl), y_d)
        
        # 4. Aux Loss Calculation (Global vs Local)
        loss_aux = 0.0
        if use_global_scout:
            # Global: Calculated once in IntegratedMoEModel
            loss_aux = model.aux_loss
        else:
            # Local: Sum up aux_loss from every expert layer
            # We traverse the model to find all MoE layers
            aux_losses = []
            for module in model.modules():
                if isinstance(module, ConvNeXtParallelMoELoRA):
                    aux_losses.append(module.local_aux_loss)
            if aux_losses:
                loss_aux = sum(aux_losses)

        # Total
        loss = loss_arc + (lamb * loss_con) + loss_domain + (aux_weight * loss_aux) + (norm_weight * norm_routing_loss)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        if use_aug_only_for_supcon:
             preds = criterion_arc.get_logits(emb_orig).argmax(dim=1)
             train_correct += (preds == y_i).sum().item()
             total_train += img_orig.size(0)
        else:
             preds = criterion_arc.get_logits(embeddings_all).argmax(dim=1)
             train_correct += (preds == y_i_all).sum().item()
             total_train += images_all.size(0)

    # -------- Evaluation --------
    model.eval(); criterion_arc.eval()
    test_correct = 0; total_test = 0
    with torch.no_grad():
        for img_orig, y_i, y_d in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]"):
            img_orig, y_i = img_orig.to(device), y_i.to(device)
            embeddings = model(img_orig)
            preds = criterion_arc.get_logits(embeddings).argmax(dim=1)
            test_correct += (preds == y_i).sum().item()
            total_test += y_i.size(0)

    print(f"Epoch [{epoch+1}/{epochs}] Summary:")
    print(f"  -> Train | Loss: {train_loss/len(train_loader):.4f} | Acc: {train_correct/total_train:.4f}")
    print(f"  -> Test  | Acc: {test_correct/total_test:.4f}")
    print("-" * 50)
    scheduler.step()
