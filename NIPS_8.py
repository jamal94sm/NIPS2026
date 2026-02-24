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
norm_weight = 1.0    # HIGH: The Scout acts as the "General" for the network.

### Architectural Toggles
num_experts = 3
top_k = 2

# MASTER TOGGLES
use_moe_mlp = True          
use_moe_stage3_norm = False 
use_moe_final_norm = False  
use_grl = True              

use_global_scout = True     
use_spectral_scout = False  # RGB is more stable for the Scout

freeze_base_mlp = True          
freeze_base_stage3_norm = False 
freeze_base_final_norm = False  

# Choose domains by NAME
train_domains = ["460", "WHT", "700"]   
test_domains  = ["850"]          

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Transforms (Standard + FFT Plan)
# ----------------------------
orig_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# The "General" Augmentations you requested
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
# 3. FFT Batch Mixer (The Swapping Mechanism)
# ----------------------------
def label_guided_fft_mixup(x, labels, beta=0.15):
    """
    Swaps low-freq amplitude between images in the batch.
    It sorts the batch by label to maximize cross-domain swapping.
    """
    B, C, H, W = x.shape
    
    # 1. FFT to Frequency Domain
    fft = torch.fft.fft2(x, dim=(-2, -1))
    amp = torch.abs(fft)
    pha = torch.angle(fft)
    
    # 2. Shift to center (Low Frequencies)
    amp_shifted = torch.fft.fftshift(amp, dim=(-2, -1))
    
    # 3. Target Selection (Label-Guided Sorting)
    # We sort by label, then roll by B/2. 
    # This pairs Domain A images with Domain B images (if batch is balanced).
    sorted_idx = torch.argsort(labels)
    roll_amount = B // 2
    target_indices = torch.roll(sorted_idx, shifts=roll_amount, dims=0)
    
    # Get the "Style" (Amplitude) from the target images
    amp_shifted_trg = amp_shifted[target_indices]
    
    # 4. Swap Low Frequencies (The "Style" Box)
    b = int(np.floor(np.amin((H, W)) * beta))
    c_h, c_w = int(np.floor(H / 2.0)), int(np.floor(W / 2.0))
    
    # Replace Source Low-Freqs with Target Low-Freqs
    amp_shifted[..., c_h-b:c_h+b, c_w-b:c_w+b] = amp_shifted_trg[..., c_h-b:c_h+b, c_w-b:c_w+b]
    
    # 5. Inverse FFT
    amp_mixed = torch.fft.ifftshift(amp_shifted, dim=(-2, -1))
    fft_new = amp_mixed * torch.exp(1j * pha) # New Amp + Original Phase
    x_aug = torch.fft.ifft2(fft_new, dim=(-2, -1)).real
    
    return torch.clamp(x_aug, 0, 1)

# ----------------------------
# 4. Dataset Class
# ----------------------------
class CASIA_MS_Dataset(Dataset):
    def __init__(self, data_path, target_domains, orig_transform=None, aug_transform=None, is_train=True):
        self.samples = []
        self.hand_id_map = {}
        self.domain_map = {d: i for i, d in enumerate(target_domains)}
        
        self.orig_transform = orig_transform
        self.aug_transform = aug_transform # Standard Augs
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
        
        # 1. Clean
        if self.orig_transform:
            img_orig = self.orig_transform(img)
        else:
            img_orig = transforms.Resize((224, 224))(img)
            img_orig = transforms.ToTensor()(img_orig)

        # 2. Augmented (Standard Spatial Augs)
        if self.is_train and self.aug_transform:
            img_aug = self.aug_transform(img)
            return img_orig, img_aug, y_i, y_d
        
        return img_orig, y_i, y_d

# ----------------------------
# 5. Modules & Scout
# ----------------------------
class SpectralPreprocess(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        fft = torch.fft.fft2(x, dim=(-2, -1))
        x_amp = torch.fft.ifft2(torch.abs(fft) * torch.exp(1j * torch.zeros_like(fft)), dim=(-2, -1)).real
        return torch.clamp(x_amp, 0, 1)

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

class ConvNeXtParallelMoELoRA(nn.Module):
    def __init__(self, orig_mlp, dim, num_experts=3, top_k=2, r=8, alpha=8, freeze_base=True):
        super().__init__()
        self.orig_mlp = orig_mlp; self.num_experts = num_experts; self.top_k = top_k
        if freeze_base: 
            for p in self.orig_mlp.parameters(): p.requires_grad = False
        else:
            for p in self.orig_mlp.parameters(): p.requires_grad = True
        self.experts = VectorizedLoRAExperts(dim, num_experts, r, alpha)
    def forward(self, x, info):
        gate, topk_probs, topk_idx = info
        orig = self.orig_mlp(x); shape = x.shape
        x_flat = x.view(-1, shape[-1]); moe = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            idx, k_idx = torch.where(topk_idx == i)
            if len(idx)==0: continue
            moe[idx] += self.experts(x_flat[idx], i) * topk_probs[idx, k_idx].unsqueeze(-1)
        return orig + moe.view(*shape)

class IntegratedMoEModel(nn.Module):
    def __init__(self, backbone, scout, num_experts, top_k, use_spectral_scout=False):
        super().__init__()
        self.backbone = backbone; self.scout = scout; self.num_experts = num_experts; self.top_k = top_k
        self.use_spectral_scout = use_spectral_scout
        if use_spectral_scout: self.spectral_split = SpectralPreprocess()
        self.aux_loss = 0.0; self.scout_logits = None
    def forward(self, x):
        scout_in = self.spectral_split(x) if self.use_spectral_scout else x
        self.scout_logits = self.scout(scout_in)
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
# 6. Data Loading 
# ----------------------------
data_path = "/home/pai-ng/Jamal/CASIA-MS-ROI"

print("Creating Training Dataset...")
# Note: aug_transform passed here contains only standard augs. FFT happens in loop.
train_dataset = CASIA_MS_Dataset(data_path, train_domains, orig_transform=orig_transform, aug_transform=aug_transform, is_train=True)
print("Creating Test Dataset...")
test_dataset  = CASIA_MS_Dataset(data_path, test_domains, orig_transform=orig_transform, is_train=False)

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

class RoutedConvNeXtBlock(nn.Module):
    def __init__(self, original_block):
        super().__init__(); self.block = original_block
    def forward(self, x):
        shortcut = x; x = self.block.conv_dw(x)
        if self.block.use_conv_mlp:
            x = self.block.norm(x, self.block.norm.current_routing_weights) if isinstance(self.block.norm, ParallelMoELayerNorm) else self.block.norm(x)
            x = self.block.mlp(x, self.block.mlp.current_routing_info) if isinstance(self.block.mlp, ConvNeXtParallelMoELoRA) else self.block.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.block.norm(x, self.block.norm.current_routing_weights) if isinstance(self.block.norm, ParallelMoELayerNorm) else self.block.norm(x)
            x = self.block.mlp(x, self.block.mlp.current_routing_info) if isinstance(self.block.mlp, ConvNeXtParallelMoELoRA) else self.block.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.block.gamma is not None: x = self.block.gamma * x
        return self.block.drop_path(x) + shortcut

for i, block in enumerate(base_model.stages[3].blocks):
    if use_moe_mlp: block.mlp = ConvNeXtParallelMoELoRA(block.mlp, stage_3_dim, num_experts, top_k, 8, 8, freeze_base_mlp).to(device)
    base_model.stages[3].blocks[i] = RoutedConvNeXtBlock(block)

if use_moe_final_norm and hasattr(base_model, 'norm'):
    orig_shape = getattr(base_model.norm, 'normalized_shape', embedding_dim)
    eps = getattr(base_model.norm, 'eps', 1e-6)
    base_model.norm = ParallelMoELayerNorm(base_model.norm, orig_shape, num_experts, eps, freeze_base_final_norm).to(device)

if use_global_scout:
    scout = GlobalDomainRouter(num_experts).to(device)
    model = IntegratedMoEModel(base_model, scout, num_experts, top_k, use_spectral_scout).to(device)
else:
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
for epoch in range(epochs):
    model.train(); proj_head.train(); criterion_arc.train()
    if use_grl: domain_classifier.train()
    train_loss = 0.0; train_correct = 0; total_train = 0

    for batch_idx, (img_orig, img_aug, y_i, y_d) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
        img_orig, img_aug = img_orig.to(device), img_aug.to(device)
        y_i, y_d = y_i.to(device), y_d.to(device)

        # === FFT BATCH MIXUP (Label-Guided) ===
        # We apply this to the Augmented images to make them "Hardest Negatives"
        # 50% chance to replace standard augs with Amplitude Swapped version
        if torch.rand(1) < 0.99:
            # Swap amplitudes between different domains if possible
            img_aug = label_guided_fft_mixup(img_aug, y_d, beta=0.15)

        optimizer.zero_grad()
        images_all = torch.cat([img_orig, img_aug], dim=0)
        y_d_all = torch.cat([y_d, y_d], dim=0)

        # Forward
        embeddings_all = model(images_all)
        projections_all = proj_head(embeddings_all)

        batch_size_curr = img_orig.size(0)
        emb_orig = embeddings_all[:batch_size_curr]

        loss_arc = criterion_arc(emb_orig, y_i)
        labels_all = torch.cat([y_i, y_i], dim=0)
        loss_con = criterion_supcon(projections_all, labels_all)
        
        # GRL Loss
        loss_domain = 0.0
        if use_grl:
            p = float(batch_idx + epoch * len(train_loader)) / (len(train_loader) * epochs)
            alpha_grl = 2. / (1. + np.exp(-10 * p)) - 1
            domain_logits = domain_classifier(embeddings_all, alpha_grl)
            loss_domain = criterion_domain(domain_logits, y_d_all)
        
        # Scout Supervision
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
