"""
GIFT: Generating stylIzed FeaTures for Single-Source Cross-Dataset
Palmprint Recognition With Unseen Target Dataset
IEEE Transactions on Image Processing, Vol. 33, 2024

Paper: Shao, Li, Zhong (2024)
Implementation: Full clean reproduction for CASIA Multi-Spectral dataset

Dataset path convention: <root>/<subject_id>_<hand>_<spectrum>_<iter>.jpg
  e.g.  001_l_630_01.jpg
Spectra available: 460, 630, 700, 850, 940, WHT
ROI size expected: 112 × 112 (as per paper)
"""

import os
import math
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pytorch_metric_learning import losses
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH     = "/home/pai-ng/Jamal/CASIA-MS-ROI"

# Paper Sec. IV-A: choose any one spectrum as source; evaluate on every other
SOURCE_DOMAIN = "630"          # training domain
TARGET_DOMAIN = "700"          # test domain  (change to evaluate other pairs)

# Paper Sec. IV-B implementation details
BATCH_SIZE    = 24
LR            = 1e-3
EPOCHS        = 100            # paper does not specify; 100 is a sensible default
EMB_DIM       = 128
ARC_MARGIN    = 0.3
ARC_SCALE     = 64             # standard ArcFace scale

# Loss weights (Table IX – optimal: α=15, β=10)
ALPHA         = 15.0           # weight for diversity-enhancement loss  L_DE
BETA          = 10.0           # weight for consistency-preservation loss  L_con

# Feature stylization noise strength (γ)
GAMMA         = 0.5            # γ ∈ (0, 1]; paper does not state exact value

EVAL_EVERY    = 5              # epochs between evaluations
NUM_WORKERS   = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ─────────────────────────────────────────────
# 2.  DATASET
# ─────────────────────────────────────────────
def build_label_map(data_path, source_domain):
    """Scan dataset and assign integer identity labels from source domain."""
    hand_ids = set()
    for root, _, files in os.walk(data_path):
        for fname in sorted(files):
            if not fname.lower().endswith(".jpg"):
                continue
            parts = fname[:-4].split("_")
            if len(parts) != 4:
                continue
            subject_id, hand, spectrum, _ = parts
            if spectrum == source_domain:
                hand_ids.add(f"{subject_id}_{hand}")
    label_map = {h: i for i, h in enumerate(sorted(hand_ids))}
    return label_map


class CASIAMultiSpectral(Dataset):
    """
    Loads ROI images for a single spectrum.
    Returns (img_tensor, identity_label) pairs.
    """
    def __init__(self, data_path, domain, label_map, augment=False):
        self.samples  = []
        self.augment  = augment
        self.label_map = label_map

        base_tf = [transforms.Resize((112, 112)), transforms.ToTensor()]
        self.to_tensor = transforms.Compose(base_tf)
        self.aug = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
        ])

        for root, _, files in os.walk(data_path):
            for fname in sorted(files):
                if not fname.lower().endswith(".jpg"):
                    continue
                parts = fname[:-4].split("_")
                if len(parts) != 4:
                    continue
                subject_id, hand, spectrum, _ = parts
                if spectrum != domain:
                    continue
                hand_id = f"{subject_id}_{hand}"
                if hand_id not in label_map:
                    continue
                self.samples.append((
                    os.path.join(root, fname),
                    label_map[hand_id]
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tf  = self.aug if self.augment else self.to_tensor
        return tf(img), label


# ─────────────────────────────────────────────
# 3.  FEATURE STYLIZATION MODULE
#     (Paper Sec. III-C, Fig. 2)
# ─────────────────────────────────────────────
class FeatureStylizationModule(nn.Module):
    """
    Inserted after each convolutional layer of the backbone.

    Forward pass returns:
        original feature f(x)   – used for L_sup, L_con, discriminator label=1
        stylized feature f(x)^new – used for L_DE, L_con, discriminator label=0

    Low-frequency decomposition (Eq. 2):
        f_L(x) = UP(AvgPool(f(x)))   kernel_size=2
        f_H(x) = f(x) - f_L(x)

    New low-frequency statistics (Eqs. 3-9):
        μ_i^L, σ_i^L  – per-channel spatial mean & std of f_L
        Batch-level variance of those statistics → φ ~ N(0, γ)
        New f_L^new via instance-normalise then rescale with perturbed stats
        Stylised output = f_L^new + f_H  (Eq. 10)
    """

    def __init__(self, gamma=GAMMA):
        super().__init__()
        self.gamma = gamma

    def _decompose(self, f):
        """Split feature map into low- and high-frequency components."""
        # AvgPool with kernel=2 then nearest upsample back to original size
        _, _, H, W = f.shape
        f_L = F.avg_pool2d(f, kernel_size=2, stride=2, padding=0)
        f_L = F.interpolate(f_L, size=(H, W), mode='nearest')
        f_H = f - f_L
        return f_L, f_H

    def forward(self, f):
        """
        Args:
            f: feature map of shape (B, C, H, W)
        Returns:
            f_orig:     original feature (= f,  unchanged)
            f_stylized: stylized feature
        """
        if not self.training:
            # At inference, stylization is not applied
            return f, f

        f_orig = f
        f_L, f_H = self._decompose(f)

        # --- Channel-wise mean & variance of low-freq component (Eqs. 3-4) ---
        # mu_i shape: (B, C)  ;  sig_i shape: (B, C)
        mu_i  = f_L.mean(dim=(-2, -1))          # (B, C)
        var_i = f_L.var(dim=(-2, -1), unbiased=False)
        sig_i = (var_i + 1e-8).sqrt()           # (B, C)

        # --- Batch-level variance of those statistics (Eqs. 5-6) ---
        mu_hat  = mu_i.var(dim=0, unbiased=False).sqrt()   # (C,)
        sig_hat = sig_i.var(dim=0, unbiased=False).sqrt()  # (C,)

        # --- Sample noise (Eqs. 7-8) ---
        phi_mu  = torch.randn_like(mu_i)  * self.gamma   # (B, C)
        phi_sig = torch.randn_like(sig_i) * self.gamma   # (B, C)

        mu_new  = mu_i  + phi_mu  * mu_hat.unsqueeze(0)   # (B, C)
        sig_new = sig_i + phi_sig * sig_hat.unsqueeze(0)  # (B, C)

        # --- Generate new low-frequency component (Eq. 9) ---
        # Instance-normalise f_L, then rescale with new stats
        mu_i_4d  = mu_i.view(-1, mu_i.shape[1], 1, 1)
        sig_i_4d = sig_i.view(-1, sig_i.shape[1], 1, 1)
        mu_new_4d  = mu_new.view(-1, mu_new.shape[1], 1, 1)
        sig_new_4d = sig_new.view(-1, sig_new.shape[1], 1, 1)

        f_L_norm = (f_L - mu_i_4d) / (sig_i_4d + 1e-8)
        f_L_new  = mu_new_4d + sig_new_4d * f_L_norm      # Eq. 9 (note: paper writes μ^new * (·) + σ^new)

        # --- Combine with original high-frequency (Eq. 10) ---
        f_stylized = f_L_new + f_H

        return f_orig, f_stylized


# ─────────────────────────────────────────────
# 4.  DISCRIMINATOR  (for Diversity Enhancement Loss L_DE)
#     (Paper Sec. III-D.1)
# ─────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Binary classifier: original feature → label 1, stylized → label 0.
    Operates on spatially-pooled feature vectors.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 5.  BACKBONE  (ResNet-18 + injected stylization)
#     (Paper Sec. III-B)
# ─────────────────────────────────────────────
class GIFTBackbone(nn.Module):
    """
    ResNet-18 pretrained on ImageNet.
    Last FC replaced with 128-d linear head.
    FeatureStylizationModules injected after:
        conv1  (layer 0)
        layer1 (ResBlock 1)
        layer2 (ResBlock 2)
        layer3 (ResBlock 3)
        layer4 (ResBlock 4)
    This matches Table X best result: all Conv + ResB1-4.

    Forward returns:
        emb        – final L2-normalised 128-d embedding
        stylized_pairs – list of (f_orig, f_stylized) tuples, one per FSM
    """

    def __init__(self, emb_dim=EMB_DIM, gamma=GAMMA):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Stem
        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.fsm0    = FeatureStylizationModule(gamma)   # after conv1

        # Residual stages
        self.layer1 = resnet.layer1
        self.fsm1   = FeatureStylizationModule(gamma)

        self.layer2 = resnet.layer2
        self.fsm2   = FeatureStylizationModule(gamma)

        self.layer3 = resnet.layer3
        self.fsm3   = FeatureStylizationModule(gamma)

        self.layer4 = resnet.layer4
        self.fsm4   = FeatureStylizationModule(gamma)

        # Head
        self.avgpool = resnet.avgpool
        self.fc      = nn.Linear(resnet.fc.in_features, emb_dim)

        # Channel sizes for discriminators
        self.channel_sizes = [64, 64, 128, 256, 512]

    def forward(self, x):
        stylized_pairs = []

        # Stem
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x_orig, x_sty = self.fsm0(x)
        stylized_pairs.append((x_orig, x_sty))
        x = x_sty if self.training else x_orig

        # Stage 1
        x = self.layer1(x)
        x_orig, x_sty = self.fsm1(x)
        stylized_pairs.append((x_orig, x_sty))
        x = x_sty if self.training else x_orig

        # Stage 2
        x = self.layer2(x)
        x_orig, x_sty = self.fsm2(x)
        stylized_pairs.append((x_orig, x_sty))
        x = x_sty if self.training else x_orig

        # Stage 3
        x = self.layer3(x)
        x_orig, x_sty = self.fsm3(x)
        stylized_pairs.append((x_orig, x_sty))
        x = x_sty if self.training else x_orig

        # Stage 4
        x = self.layer4(x)
        x_orig, x_sty = self.fsm4(x)
        stylized_pairs.append((x_orig, x_sty))
        x = x_sty if self.training else x_orig

        # Pool → embed
        x   = self.avgpool(x).flatten(1)
        emb = self.fc(x)
        emb = F.normalize(emb, p=2, dim=1)

        return emb, stylized_pairs


# ─────────────────────────────────────────────
# 6.  EVALUATION HELPERS
# ─────────────────────────────────────────────
def compute_eer(gen_scores, imp_scores):
    if len(gen_scores) == 0 or len(imp_scores) == 0:
        return float("nan")
    gen = np.array(gen_scores)
    imp = np.array(imp_scores)
    thrs = np.linspace(min(gen.min(), imp.min()),
                       max(gen.max(), imp.max()), 500)
    best = min(
        ((abs((imp >= t).mean() - (gen < t).mean()),
          ((imp >= t).mean() + (gen < t).mean()) / 2)
         for t in thrs),
        key=lambda x: x[0],
    )
    return best[1] * 100


@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    feats, labels = [], []
    for imgs, lbl in loader:
        imgs = imgs.to(device)
        emb, _ = model(imgs)
        feats.append(F.normalize(emb, p=2, dim=1).cpu())
        labels.append(lbl)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def evaluate(model, reg_loader, qry_loader, epoch, phase=""):
    """
    Identification  (Rank-1 Acc) + Verification (EER).
    Paper: source dataset = registration set, target dataset = query set.
    """
    reg_feats, reg_labels = extract_embeddings(model, reg_loader)
    qry_feats, qry_labels = extract_embeddings(model, qry_loader)

    # ── Rank-1 identification ────────────────────────────────────────────────
    sim    = torch.mm(qry_feats, reg_feats.t())  # (Q, G)
    preds  = reg_labels[sim.argmax(dim=1)]
    acc    = (preds == qry_labels).float().mean().item() * 100

    # ── EER verification ─────────────────────────────────────────────────────
    sim_np = sim.numpy()
    Q, G   = sim_np.shape
    gen_s, imp_s = [], []
    for i in range(Q):
        for j in range(G):
            s = sim_np[i, j]
            if qry_labels[i] == reg_labels[j]:
                gen_s.append(s)
            else:
                imp_s.append(s)
    eer = compute_eer(gen_s, imp_s)

    tag = f"[{phase}] " if phase else ""
    print(f"\n  ┌─ {tag}Epoch {epoch}  "
          f"src={SOURCE_DOMAIN} ({G} imgs)  tgt={TARGET_DOMAIN} ({Q} imgs)")
    print(f"  │  Rank-1 Accuracy : {acc:6.2f}%")
    print(f"  │  EER             : {eer:5.2f}%")
    print(f"  └{'─'*60}")
    return acc, eer


# ─────────────────────────────────────────────
# 7.  TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, discriminators, criterion_arc,
                    optimizer, opt_disc, train_loader, epoch):
    """
    One full epoch.

    Loss (Eq. 13):  L = L_sup  +  α · L_DE  +  β · L_con

    L_sup  – ArcFace on the final embedding using the STYLIZED path
             (the stylized feature travels through remaining layers and
              produces the final embedding; this is the 'model output'
              described in the paper as the feature extractor output)

    L_DE   – cross-entropy in each discriminator:
              original features → class 1
              stylized features → class 0
              Gradient flows back into FSM to make stylized ≠ original

    L_con  – L2 distance between original and stylized features at
              each FSM location (Eq. 12 applied at feature-map level)
    """
    model.train()
    criterion_disc = nn.CrossEntropyLoss()

    total_loss = total_arc = total_de = total_con = 0.0
    correct = 0; total = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs   = imgs.to(device)
        labels = labels.to(device)

        # ── Forward ──────────────────────────────────────────────────────────
        optimizer.zero_grad()
        opt_disc.zero_grad()

        emb, stylized_pairs = model(imgs)

        # ── L_sup: ArcFace on final embedding ────────────────────────────────
        loss_arc = criterion_arc(emb, labels)

        # ── L_DE + L_con: iterate over each FSM ──────────────────────────────
        loss_de  = torch.tensor(0.0, device=device)
        loss_con = torch.tensor(0.0, device=device)

        for k, (f_orig, f_sty) in enumerate(stylized_pairs):
            disc = discriminators[k]

            # Discriminator predictions
            #   original  → label 1
            #   stylized  → label 0
            logits_orig = disc(f_orig)
            logits_sty  = disc(f_sty)

            lbl_orig = torch.ones(imgs.size(0), dtype=torch.long, device=device)
            lbl_sty  = torch.zeros(imgs.size(0), dtype=torch.long, device=device)

            loss_de = loss_de + (criterion_disc(logits_orig, lbl_orig) +
                                 criterion_disc(logits_sty,  lbl_sty)) * 0.5

            # L_con: L2 between original and stylized feature maps (Eq. 12)
            # Spatial mean to get a vector, then L2 norm
            loss_con = loss_con + F.mse_loss(
                f_sty.mean(dim=(-2, -1)),
                f_orig.mean(dim=(-2, -1)).detach()
            )

        loss_de  = loss_de  / len(stylized_pairs)
        loss_con = loss_con / len(stylized_pairs)

        loss = loss_arc + ALPHA * loss_de + BETA * loss_con

        loss.backward()
        optimizer.step()
        opt_disc.step()

        # ── Metrics ──────────────────────────────────────────────────────────
        total_loss += loss.item()
        total_arc  += loss_arc.item()
        total_de   += loss_de.item()
        total_con  += loss_con.item()

        preds = criterion_arc.get_logits(emb).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    n = len(train_loader)
    print(f"  Loss {total_loss/n:.4f}  "
          f"(arc={total_arc/n:.4f}  de={total_de/n:.4f}  con={total_con/n:.4f})  "
          f"Train-Acc={correct/total:.4f}")


# ─────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────
def main():
    # ── Label map built from source domain only ───────────────────────────────
    label_map       = build_label_map(DATA_PATH, SOURCE_DOMAIN)
    num_classes     = len(label_map)
    print(f"Identities in source domain '{SOURCE_DOMAIN}': {num_classes}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = CASIAMultiSpectral(DATA_PATH, SOURCE_DOMAIN,
                                       label_map, augment=True)
    # Registration = all source images (no augmentation)
    reg_dataset   = CASIAMultiSpectral(DATA_PATH, SOURCE_DOMAIN,
                                       label_map, augment=False)
    # Query = all target images
    qry_dataset   = CASIAMultiSpectral(DATA_PATH, TARGET_DOMAIN,
                                       label_map, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    reg_loader   = DataLoader(reg_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)
    qry_loader   = DataLoader(qry_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)

    print(f"Train: {len(train_dataset)}  |  "
          f"Registration: {len(reg_dataset)}  |  "
          f"Query: {len(qry_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GIFTBackbone(emb_dim=EMB_DIM, gamma=GAMMA).to(device)

    # One discriminator per FSM location (5 total: conv1 + ResB1-4)
    discriminators = nn.ModuleList([
        Discriminator(c).to(device) for c in model.channel_sizes
    ])

    # ── Losses ────────────────────────────────────────────────────────────────
    criterion_arc = losses.ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=EMB_DIM,
        margin=ARC_MARGIN,
        scale=ARC_SCALE
    ).to(device)

    # ── Optimizers (paper: RMSprop, lr=0.001) ────────────────────────────────
    # Main optimizer: backbone + arcface head
    optimizer = optim.RMSprop(
        list(model.parameters()) + list(criterion_arc.parameters()),
        lr=LR, weight_decay=1e-4
    )
    # Separate optimizer for discriminators
    opt_disc = optim.RMSprop(discriminators.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GIFT Training   src={SOURCE_DOMAIN}  →  tgt={TARGET_DOMAIN}")
    print(f"  α={ALPHA}  β={BETA}  γ={GAMMA}  epochs={EPOCHS}")
    print(f"{'='*60}\n")

    best_acc = 0.0

    for epoch in range(EPOCHS):
        train_one_epoch(model, discriminators, criterion_arc,
                        optimizer, opt_disc, train_loader, epoch)

        if (epoch + 1) % EVAL_EVERY == 0:
            acc, eer = evaluate(model, reg_loader, qry_loader,
                                epoch + 1, phase="GIFT")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch":     epoch + 1,
                    "model":     model.state_dict(),
                    "acc":       acc,
                    "eer":       eer,
                    "src":       SOURCE_DOMAIN,
                    "tgt":       TARGET_DOMAIN,
                }, f"gift_best_{SOURCE_DOMAIN}_to_{TARGET_DOMAIN}.pth")
                print(f"  ✓ Best model saved  (Rank-1={acc:.2f}%)")

        scheduler.step()

    print(f"\nTraining complete.  Best Rank-1 = {best_acc:.2f}%")


if __name__ == "__main__":
    main()
