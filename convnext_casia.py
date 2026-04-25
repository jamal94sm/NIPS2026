"""
ConvNeXtV2-Tiny finetuning on CASIA-MS
=======================================
Architecture : ConvNeXtV2-Tiny (pretrained ImageNet)
               Frozen : stem + stages 0-2
               Trainable : stage 3 + final norm
Loss         : ArcFace  +  λ · SupConLoss
Evaluation   : Rank-1 Accuracy  +  EER
               Registration = source domain images
               Query         = target domain images
"""

import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH     = "/home/pai-ng/Jamal/CASIA-MS-ROI"
TRAIN_DOMAINS = ["460", "WHT"]
TEST_DOMAINS  = ["700"]

BATCH_SIZE    = 32
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 200
LAMB          = 0.2       # SupCon weight
MARGIN        = 0.3       # ArcFace margin
SCALE         = 16        # ArcFace scale
EVAL_EVERY    = 5
NUM_WORKERS   = 4
SEED          = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


set_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE = 224

base_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

aug_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomChoice([
        transforms.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomPerspective(distortion_scale=0.15, p=1),
        transforms.RandomChoice([
            transforms.RandomRotation(10, interpolation=Image.BICUBIC,
                                      expand=False,
                                      center=(int(0.5*IMG_SIZE), 0)),
            transforms.RandomRotation(10, interpolation=Image.BICUBIC,
                                      expand=False,
                                      center=(0, int(0.5*IMG_SIZE))),
        ]),
    ]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

# Build shared identity label map from training domains
_all_hand_ids = set()
for root, _, files in os.walk(DATA_PATH):
    for fname in sorted(files):
        if not fname.lower().endswith(".jpg"): continue
        parts = fname[:-4].split("_")
        if len(parts) != 4: continue
        subject_id, hand, spectrum, _ = parts
        if spectrum in set(TRAIN_DOMAINS):
            _all_hand_ids.add(f"{subject_id}_{hand}")

shared_label_map  = {h: i for i, h in enumerate(sorted(_all_hand_ids))}
num_total_classes = len(shared_label_map)
print(f"Shared identity space: {num_total_classes} identities")


class CASIADataset(Dataset):
    """
    Training  : returns (img_orig, img_aug, label)
                img_orig = base transform (no augmentation)
                img_aug  = CompNet-style random augmentation
    Inference : returns (img_orig, label)
    """
    def __init__(self, data_path, domains, label_map, train=True):
        self.samples   = []
        self.label_map = label_map
        self.train     = train

        for root, _, files in os.walk(data_path):
            for fname in sorted(files):
                if not fname.lower().endswith(".jpg"): continue
                parts = fname[:-4].split("_")
                if len(parts) != 4: continue
                subject_id, hand, spectrum, _ = parts
                if spectrum not in domains: continue
                hand_id = f"{subject_id}_{hand}"
                if hand_id not in label_map: continue
                self.samples.append((
                    os.path.join(root, fname),
                    label_map[hand_id],
                ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.train:
            return base_transform(img), aug_transform(img), label
        return base_transform(img), label


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ConvNeXtFinetune(nn.Module):
    """
    ConvNeXtV2-Tiny with:
      - stem + stage 0/1/2 : frozen
      - stage 3 + final norm : trainable
    Returns L2-normalised embeddings.
    """
    def __init__(self):
        super().__init__()
        backbone = timm.create_model(
            'convnextv2_tiny', pretrained=True, num_classes=0)

        # Freeze everything
        for p in backbone.parameters():
            p.requires_grad = False

        # Unfreeze stage 3 + final norm
        for p in backbone.stages[3].parameters():
            p.requires_grad = True
        if hasattr(backbone, 'norm'):
            for p in backbone.norm.parameters():
                p.requires_grad = True

        self.backbone    = backbone
        self.embed_dim   = backbone.num_features

    def forward(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# PROJECTION HEAD  (for SupCon)
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, dim_out=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_out))
    def forward(self, x):
        return F.normalize(self.head(x), dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(gen_scores, imp_scores):
    if len(gen_scores) == 0 or len(imp_scores) == 0:
        return float("nan")
    gen  = np.array(gen_scores); imp = np.array(imp_scores)
    thrs = np.linspace(min(gen.min(), imp.min()),
                       max(gen.max(), imp.max()), 500)
    eer  = min(
        ((abs((imp >= t).mean() - (gen < t).mean()),
          ((imp >= t).mean() + (gen < t).mean()) / 2)
         for t in thrs),
        key=lambda x: x[0],
    )[1] * 100
    return eer


@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    feats, labels = [], []
    for imgs, lbl in loader:
        feats.append(model(imgs.to(DEVICE)).cpu())
        labels.append(lbl)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def evaluate(model, gal_loader, prb_loader, epoch):
    gal_feats, gal_labels = extract_embeddings(model, gal_loader)
    prb_feats, prb_labels = extract_embeddings(model, prb_loader)

    sim    = torch.mm(prb_feats, gal_feats.t())
    nn_idx = sim.argmax(dim=1)
    acc    = (gal_labels[nn_idx] == prb_labels).float().mean().item() * 100

    sim_np = sim.numpy()
    gen_scores, imp_scores = [], []
    for i in range(len(prb_labels)):
        for j in range(len(gal_labels)):
            s = sim_np[i, j]
            if prb_labels[i] == gal_labels[j]: gen_scores.append(s)
            else:                               imp_scores.append(s)
    eer = compute_eer(gen_scores, imp_scores)

    print(f"\n  ┌─ Epoch {epoch} | gallery={len(gal_labels)} "
          f"| probe={len(prb_labels)} | domain={TEST_DOMAINS[0]}")
    print(f"  │  Rank-1 : {acc:6.2f}%")
    print(f"  │  EER    : {eer:5.2f}%")
    print(f"  └{'─'*55}")
    return acc, eer


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\nDevice: {DEVICE}")
    print(f"Train domains : {TRAIN_DOMAINS}   Test domain: {TEST_DOMAINS}")
    print(f"Epochs: {EPOCHS}   LR: {LR}   λ_SupCon: {LAMB}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    # ── Collect test-domain samples, split per-ID 50/50 into gallery/probe ──
    test_ds_all = CASIADataset(DATA_PATH, TEST_DOMAINS, shared_label_map, train=False)
    by_id = {}
    for path, label in test_ds_all.samples:
        by_id.setdefault(label, []).append(path)

    rng = random.Random(SEED)
    gallery_samples, probe_samples = [], []
    for label, paths in sorted(by_id.items()):
        paths = list(paths); rng.shuffle(paths)
        n_gal = max(1, len(paths) // 2)
        n_gal = min(n_gal, len(paths) - 1) if len(paths) > 1 else n_gal
        for p in paths[:n_gal]: gallery_samples.append((p, label))
        for p in paths[n_gal:]: probe_samples.append((p, label))

    class _SampleDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            path, label = self.samples[idx]
            return base_transform(Image.open(path).convert("RGB")), label

    train_ds     = CASIADataset(DATA_PATH, TRAIN_DOMAINS, shared_label_map, train=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    gal_loader   = DataLoader(_SampleDataset(gallery_samples), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    prb_loader   = DataLoader(_SampleDataset(probe_samples),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train images   : {len(train_ds)}")
    print(f"Gallery images : {len(gallery_samples)}   "
          f"Probe images: {len(probe_samples)}")
    print(f"Test IDs       : {len(by_id)}")


    # ── Model ─────────────────────────────────────────────────────────────────
    model   = ConvNeXtFinetune().to(DEVICE)
    proj    = ProjectionHead(model.embed_dim).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nModel params — trainable: {trainable/1e6:.2f}M / total: {total/1e6:.2f}M")

    # ── Losses ────────────────────────────────────────────────────────────────
    criterion_arc    = losses.ArcFaceLoss(
        num_classes=num_total_classes, embedding_size=model.embed_dim,
        margin=MARGIN, scale=SCALE).to(DEVICE)
    criterion_supcon = losses.SupConLoss(temperature=0.1).to(DEVICE)

    all_params = (list(model.parameters()) +
                  list(proj.parameters()) +
                  list(criterion_arc.parameters()))

    optimizer = optim.AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    # ── Training ──────────────────────────────────────────────────────────────
    best_rank1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train(); proj.train(); criterion_arc.train()
        ep_loss = 0.0; ep_arc = 0.0; ep_con = 0.0
        ep_corr = 0;   ep_tot = 0

        for img_orig, img_aug, y_i in tqdm(train_loader,
                                           desc=f"Epoch {epoch}/{EPOCHS}",
                                           leave=False):
            img_orig = img_orig.to(DEVICE)
            img_aug  = img_aug.to(DEVICE)
            y_i      = y_i.to(DEVICE)

            # Stack: original + augmented  (2× per sample)
            # img_orig: no augmentation (base transform only)
            # img_aug : CompNet-style random augmentation
            imgs_all = torch.cat([img_orig, img_aug], dim=0)
            y_all    = torch.cat([y_i, y_i], dim=0)

            optimizer.zero_grad()
            emb_all  = model(imgs_all)    # [2B, D]  L2-normalised
            proj_all = proj(emb_all)      # [2B, 128]

            # ArcFace on all views
            loss_arc = criterion_arc(emb_all, y_all)

            # SupCon on projected embeddings — original and aug form a positive pair
            loss_con = criterion_supcon(proj_all, y_all)

            loss = loss_arc + LAMB * loss_con
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 5.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_arc  += loss_arc.item()
            ep_con  += loss_con.item()

            with torch.no_grad():
                preds    = criterion_arc.get_logits(emb_all).argmax(dim=1)
                ep_corr += (preds == y_all).sum().item()
                ep_tot  += y_all.size(0)

        scheduler.step()

        n       = len(train_loader)
        avg_acc = 100.0 * ep_corr / ep_tot
        print(f"Epoch [{epoch}/{EPOCHS}]  "
              f"loss={ep_loss/n:.4f}  arc={ep_arc/n:.4f}  "
              f"con={ep_con/n:.4f}  train_acc={avg_acc:.2f}%")

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            acc, eer = evaluate(model, gal_loader, prb_loader, epoch)
            if acc > best_rank1:
                best_rank1 = acc
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "acc": acc, "eer": eer}, "best_model.pth")
                print(f"  ✓ New best Rank-1: {best_rank1:.2f}%  → saved")

    print(f"\nDone.  Best Rank-1 = {best_rank1:.2f}%")


if __name__ == "__main__":
    main()
