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

train_domains = ["460", "700", "630"]   
test_domains  = ["940"]          

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
# ── Build shared label map from train domains ─────────────────────────────────
# Train and test share the same identities. Building the map from train domains
# and passing it to both datasets ensures integer label k means the same
# hand_id everywhere — essential for ArcFace evaluation on the test set.
data_path = "/home/pai-ng/Jamal/CASIA-MS-ROI"

_all_hand_ids = set()
for root, _, files in os.walk(data_path):
    files.sort()
    for fname in files:
        if not fname.lower().endswith(".jpg"): continue
        parts = fname[:-4].split("_")
        if len(parts) != 4: continue
        subject_id, hand, spectrum, _ = parts
        if spectrum in set(train_domains):
            _all_hand_ids.add(f"{subject_id}_{hand}")
shared_label_map  = {h: i for i, h in enumerate(sorted(_all_hand_ids))}
num_total_classes = len(shared_label_map)
print(f"Shared identity space: {num_total_classes} identities (same in train & test)")

class CASIA_MS_Dataset(Dataset):
    """
    Uses a shared_label_map so that integer labels are consistent across
    train and test datasets. Required for valid ArcFace classification
    accuracy on the test set.
    """
    def __init__(self, data_path, target_domains, shared_label_map, is_train=True):
        self.samples      = []
        self.hand_id_map  = shared_label_map          # shared — same integers everywhere
        self.domain_map   = {d: i for i, d in enumerate(target_domains)}
        self.is_train     = is_train
        self.to_tensor    = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.general_aug  = general_aug
        
        for root, _, files in os.walk(data_path):
            files.sort()
            for fname in files:
                if not fname.lower().endswith(".jpg"): continue
                parts = fname[:-4].split("_")
                if len(parts) != 4: continue
                subject_id, hand, spectrum, iteration = parts
                if spectrum not in target_domains: continue
                hand_id = f"{subject_id}_{hand}"
                if hand_id not in self.hand_id_map: continue   # identity not in shared map
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

class ConvNeXtParallelMoELoRA(nn.Module):
    def __init__(self, orig_mlp, dim, num_experts=3, top_k=2, r=8, alpha=8, freeze_base=True, use_global_routing=True):
        super().__init__()
        self.orig_mlp = orig_mlp; self.num_experts = num_experts; self.top_k = top_k
        self.use_global_routing = use_global_routing
        self.local_aux_loss = 0.0
        if freeze_base: 
            for p in self.orig_mlp.parameters(): p.requires_grad = False
        else:
            for p in self.orig_mlp.parameters(): p.requires_grad = True
        self.experts = VectorizedLoRAExperts(dim, num_experts, r, alpha)
        if not self.use_global_routing:
            self.router = nn.Linear(dim, num_experts) 

    def forward(self, x, info=None):
        if self.use_global_routing:
            gate, topk_probs, topk_idx = info
        else:
            B, H, W, C = x.shape
            x_flat = x.view(B, -1, C).mean(dim=1)
            logits = self.router(x_flat)
            gate = F.softmax(logits, dim=-1)
            topk_probs, topk_idx = torch.topk(gate, self.top_k, dim=-1)
            topk_probs = topk_probs / (topk_probs.sum(-1, keepdim=True) + 1e-6)
            frac = torch.zeros_like(gate).scatter_(1, topk_idx, 1.0).mean(0)
            self.local_aux_loss = self.num_experts * torch.sum(frac * gate.mean(0))

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
print(f"Creating Training Dataset... (Batch Size: {batch_size})")
train_dataset = CASIA_MS_Dataset(data_path, train_domains, shared_label_map, is_train=True)
test_dataset  = CASIA_MS_Dataset(data_path, test_domains,  shared_label_map, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# ── Gallery / Query split of test set ────────────────────────────────────────
# Gallery : first image per identity  → registered template
# Query   : all remaining images      → probe set
# Sets are disjoint so no self-match issue in split evaluation.
_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

class _ListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label

_gallery_samples, _query_samples = [], []
_seen_ids = set()
for path, y_i, y_d, spec in test_dataset.samples:
    if y_i not in _seen_ids:
        _seen_ids.add(y_i)
        _gallery_samples.append((path, y_i))
    else:
        _query_samples.append((path, y_i))

gallery_loader = DataLoader(_ListDataset(_gallery_samples, _tf),
                            batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
query_loader   = DataLoader(_ListDataset(_query_samples, _tf),
                            batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

# --- MODEL SETUP ---
print(f"Model Mode: {'MoE + Global Scout' if use_global_scout else 'Normal MoE (Local Routing)'}")

base_model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0).to(device)
embedding_dim = base_model.num_features
stage_3_dim = 768

for p in base_model.parameters(): p.requires_grad = False
for p in base_model.stages[3].parameters(): p.requires_grad = True

class RoutedConvNeXtBlock(nn.Module):
    def __init__(self, original_block):
        super().__init__(); self.block = original_block
    def forward(self, x):
        shortcut = x; x = self.block.conv_dw(x)
        if self.block.use_conv_mlp:
            x = self.block.norm(x, self.block.norm.current_routing_weights) if isinstance(self.block.norm, ParallelMoELayerNorm) else self.block.norm(x)
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
# Use num_total_classes (shared map size) so ArcFace W has one row per identity,
# with the same integer→identity mapping as the test set.
criterion_arc    = losses.ArcFaceLoss(num_classes=num_total_classes, embedding_size=embedding_dim, margin=margin, scale=scale).to(device)
criterion_supcon = losses.SupConLoss(temperature=0.1).to(device)

all_params = list(model.parameters()) + list(criterion_arc.parameters()) + list(proj_head.parameters())
if use_grl: all_params += list(domain_classifier.parameters())

optimizer = optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

# ----------------------------
# 7. Evaluation helpers
# ----------------------------
def compute_eer(scores_gen, scores_imp):
    """Compute EER (%) from genuine and impostor score lists."""
    if len(scores_gen) == 0 or len(scores_imp) == 0:
        return float("nan")
    gen  = np.array(scores_gen)
    imp  = np.array(scores_imp)
    thrs = np.linspace(min(gen.min(), imp.min()), max(gen.max(), imp.max()), 500)
    eer  = min(
        ((abs((imp >= t).mean() - (gen < t).mean()),
          ((imp >= t).mean() + (gen < t).mean()) / 2)
         for t in thrs),
        key=lambda x: x[0],
    )[1] * 100
    return eer

@torch.no_grad()
def _extract_embeddings(loader):
    """Extract L2-normalised embeddings for an entire loader."""
    feats, labels = [], []
    for batch in loader:
        # test_loader returns (img, y_i, y_d); gallery/query loaders return (img, y_i)
        imgs = batch[0].to(device)
        lbl  = batch[1]
        emb  = model(imgs)
        feats.append(F.normalize(emb, p=2, dim=1).cpu())
        labels.append(lbl)
    return torch.cat(feats), torch.cat(labels)   # [M, d], [M]

@torch.no_grad()
def evaluate(epoch):
    model.eval()
    criterion_arc.eval()

    # ── 1. Full similarity-based accuracy ────────────────────────────────────
    # All M test images act as both gallery and query.
    # Cosine similarity matrix [M, M]; diagonal set to -1 to exclude self-match.
    # Rank-1: nearest neighbour = predicted identity.
    # EER  : all unique pairs (i<j) split into genuine / impostor by label.
    all_feats, all_labels = _extract_embeddings(test_loader)
    M   = len(all_labels)
    sim = torch.mm(all_feats, all_feats.t())   # [M, M] cosine similarity
    sim.fill_diagonal_(-1.0)                   # exclude self-match

    pred_full = all_labels[sim.argmax(dim=1)]
    acc_full  = (pred_full == all_labels).float().mean().item() * 100

    sim_np = sim.numpy()
    gen_full, imp_full = [], []
    for i in range(M):
        for j in range(i + 1, M):
            s = sim_np[i, j]
            (gen_full if all_labels[i] == all_labels[j] else imp_full).append(s)
    eer_full = compute_eer(gen_full, imp_full)

    # ── 2. Split similarity-based accuracy ───────────────────────────────────
    # Gallery: first image per identity (registered template).
    # Query  : remaining images (probe set). Sets are disjoint.
    # Rank-1: each query matched to nearest gallery sample.
    # EER  : all (query_i, gallery_j) pairs split by label equality.
    if len(_query_samples) > 0:
        gal_feats, gal_labels = _extract_embeddings(gallery_loader)   # [G, d]
        qry_feats, qry_labels = _extract_embeddings(query_loader)     # [Q, d]

        sim_split  = torch.mm(qry_feats, gal_feats.t())               # [Q, G]
        pred_split = gal_labels[sim_split.argmax(dim=1)]
        acc_split  = (pred_split == qry_labels).float().mean().item() * 100

        sim_split_np = sim_split.numpy()
        gen_split, imp_split = [], []
        for i in range(len(qry_labels)):
            for j in range(len(gal_labels)):
                s = sim_split_np[i, j]
                (gen_split if qry_labels[i] == gal_labels[j] else imp_split).append(s)
        eer_split = compute_eer(gen_split, imp_split)
    else:
        acc_split = eer_split = float("nan")

    # ── 3. ArcFace classification accuracy ───────────────────────────────────
    # ArcFace W: [num_total_classes, embedding_dim] — learned class prototypes.
    # At inference (no margin): score(class c) = cosine_sim(embedding, W[c])
    #                                           = L2_norm(emb) @ L2_norm(W[c])
    # Since embeddings are L2-normalised and W rows are L2-normalised,
    # this is a pure dot product. argmax → predicted identity integer.
    # Because shared_label_map is used everywhere, predicted integer directly
    # matches test label integer — no remapping needed.
    #
    # get_logits() from pytorch-metric-learning applies scale*cos(theta) without
    # margin (margin is training-only). argmax is scale-invariant so both
    # get_logits().argmax() and raw cosine argmax give identical predictions.
    arc_correct = 0
    arc_total   = 0
    for batch in test_loader:
        imgs = batch[0].to(device)
        lbl  = batch[1]
        emb  = model(imgs)
        # get_logits: scale * cos(theta), no margin — correct for inference
        preds = criterion_arc.get_logits(emb).argmax(dim=1).cpu()
        arc_correct += (preds == lbl).sum().item()
        arc_total   += lbl.size(0)
    acc_arc = 100.0 * arc_correct / arc_total

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n  ┌─ Evaluation | Epoch {epoch} | Test domain: {test_domains[0]} | M={M}")
    print(f"  │  [1] Full similarity  │ Rank-1: {acc_full:6.2f}%  EER: {eer_full:5.2f}%"
          f"  — all {M} images, self excluded")
    if not np.isnan(acc_split):
        print(f"  │  [2] Split similarity │ Rank-1: {acc_split:6.2f}%  EER: {eer_split:5.2f}%"
              f"  — gallery={len(_gallery_samples)}, query={len(_query_samples)}")
    else:
        print(f"  │  [2] Split similarity │ N/A — each identity has only 1 image")
    print(f"  │  [3] ArcFace classify │ Acc:    {acc_arc:6.2f}%"
          f"  — {num_total_classes} classes, no margin at inference")
    print(f"  └{'─'*65}")

    return acc_full, eer_full, acc_arc

# ----------------------------
# 8. Training Loop
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
        y_i_all    = torch.cat(labels_list, dim=0)
        y_d_all    = torch.cat(domains_list, dim=0)

        # === FORWARD ===
        optimizer.zero_grad()
        embeddings_all  = model(images_all)
        projections_all = proj_head(embeddings_all)

        batch_curr = img_orig.size(0)
        emb_orig   = embeddings_all[:batch_curr]

        # === LOSSES ===
        norm_routing_loss = 0.0
        if use_aug_only_for_supcon:
            loss_arc = criterion_arc(emb_orig, y_i)
            if use_global_scout:
                norm_routing_loss = F.cross_entropy(model.scout_logits[:batch_curr], y_d)
        else:
            loss_arc = criterion_arc(embeddings_all, y_i_all)
            if use_global_scout:
                norm_routing_loss = F.cross_entropy(model.scout_logits, y_d_all)

        loss_con = criterion_supcon(projections_all, y_i_all)
        
        loss_domain = 0.0
        if use_grl:
            p = float(batch_idx + epoch * len(train_loader)) / (len(train_loader) * epochs)
            alpha_grl = 2. / (1. + np.exp(-10 * p)) - 1
            loss_domain = criterion_domain(domain_classifier(emb_orig, alpha_grl), y_d)
        
        loss_aux = 0.0
        if use_global_scout:
            loss_aux = model.aux_loss
        else:
            aux_losses = []
            for module in model.modules():
                if isinstance(module, ConvNeXtParallelMoELoRA):
                    aux_losses.append(module.local_aux_loss)
            if aux_losses:
                loss_aux = sum(aux_losses)

        loss = loss_arc + (lamb * loss_con) + loss_domain + (aux_weight * loss_aux) + (norm_weight * norm_routing_loss)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        if use_aug_only_for_supcon:
            preds = criterion_arc.get_logits(emb_orig).argmax(dim=1)
            train_correct += (preds == y_i).sum().item()
            total_train   += img_orig.size(0)
        else:
            preds = criterion_arc.get_logits(embeddings_all).argmax(dim=1)
            train_correct += (preds == y_i_all).sum().item()
            total_train   += images_all.size(0)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc  = train_correct / total_train

    print(f"Epoch [{epoch+1}/{epochs}] Summary:")
    print(f"  -> Train | Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.4f}")

    # ── Evaluation (all three modes) ─────────────────────────────────────────
    acc_full, eer_full, acc_arc = evaluate(epoch + 1)
    print("-" * 50)

    scheduler.step()
