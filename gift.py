"""
GIFT: Generating stylIzed FeaTures for Single-Source Cross-Dataset
Palmprint Recognition With Unseen Target Dataset
IEEE Transactions on Image Processing, Vol. 33, 2024

Fixes applied vs. v1:
  1. ArcFace scale lowered (64→16) and warm-up phase added before FSM activates
  2. FSM noise clamped; variance computed with stability eps
  3. Loss weights α, β warmed up gradually instead of fixed large values
  4. Gradient clipping added
  5. NaN guard in training loop
"""

import os
import math
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
SOURCE_DOMAIN = "630"
TARGET_DOMAIN = "700"

BATCH_SIZE    = 24
LR            = 1e-3
WARMUP_EPOCHS = 10      # train with ArcFace ONLY, FSM disabled → lets backbone stabilise first
EPOCHS        = 100     # total epochs (warmup + main)
EMB_DIM       = 128
ARC_MARGIN    = 0.3
ARC_SCALE     = 16      # FIX: was 64; 16 is stable for small datasets

# Loss weights — ramped up after warmup (paper: α=15, β=10 at convergence)
ALPHA_FINAL   = 15.0
BETA_FINAL    = 10.0

# FSM noise strength — clamped inside module
GAMMA         = 0.3     # FIX: was 0.5; lower = less noise explosion early

EVAL_EVERY    = 5
NUM_WORKERS   = 2
GRAD_CLIP     = 1.0     # FIX: gradient clipping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ─────────────────────────────────────────────
# 2.  DATASET
# ─────────────────────────────────────────────
def build_label_map(data_path, source_domain):
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
    return {h: i for i, h in enumerate(sorted(hand_ids))}


class CASIAMultiSpectral(Dataset):
    def __init__(self, data_path, domain, label_map, augment=False):
        self.samples   = []
        self.label_map = label_map
        self.to_tensor = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),  # FIX: normalise input
        ])
        self.aug = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
        self.augment = augment

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
# 3.  FEATURE STYLIZATION MODULE  (Eqs. 2–10)
# ─────────────────────────────────────────────
class FeatureStylizationModule(nn.Module):
    """
    FIX summary vs v1:
      - std computation uses unbiased=False + larger eps (1e-5)
      - phi (noise) clamped to [-2, 2] to prevent extreme samples
      - sig_new clamped to > eps so instance-norm denominator never 0
      - disabled entirely during warmup (controlled by .active flag)
    """

    def __init__(self, gamma=GAMMA):
        super().__init__()
        self.gamma  = gamma
        self.active = False   # toggled to True after warmup

    def _decompose(self, f):
        _, _, H, W = f.shape
        f_L = F.avg_pool2d(f, kernel_size=2, stride=2, padding=0)
        f_L = F.interpolate(f_L, size=(H, W), mode='nearest')
        f_H = f - f_L
        return f_L, f_H

    def forward(self, f):
        if not self.training or not self.active:
            return f, f   # identity — no stylization

        f_orig = f
        f_L, f_H = self._decompose(f)

        # Per-sample channel statistics  (B, C)
        mu_i  = f_L.mean(dim=(-2, -1))
        sig_i = f_L.std(dim=(-2, -1), unbiased=False).clamp(min=1e-5)

        # Batch-level std of those statistics  (C,)
        mu_hat  = mu_i.std(dim=0,  unbiased=False).clamp(min=1e-5)
        sig_hat = sig_i.std(dim=0, unbiased=False).clamp(min=1e-5)

        # Sample and clamp noise
        phi_mu  = torch.randn_like(mu_i).clamp(-2, 2) * self.gamma
        phi_sig = torch.randn_like(sig_i).clamp(-2, 2) * self.gamma

        mu_new  = mu_i  + phi_mu  * mu_hat.unsqueeze(0)
        sig_new = (sig_i + phi_sig * sig_hat.unsqueeze(0)).clamp(min=1e-5)

        # Reshape for broadcasting  (B, C, 1, 1)
        def _4d(t): return t.view(t.shape[0], t.shape[1], 1, 1)

        f_L_norm  = (f_L - _4d(mu_i))  / (_4d(sig_i)  + 1e-5)
        f_L_new   = _4d(mu_new) + _4d(sig_new) * f_L_norm

        f_stylized = f_L_new + f_H
        return f_orig, f_stylized


# ─────────────────────────────────────────────
# 4.  DISCRIMINATOR
# ─────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, max(in_channels // 2, 32)),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // 2, 32), 2),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 5.  BACKBONE  (ResNet-18 + FSMs)
# ─────────────────────────────────────────────
class GIFTBackbone(nn.Module):
    """
    ResNet-18 pretrained on ImageNet.
    FSM injected after conv1 + all 4 residual stages.
    """

    def __init__(self, emb_dim=EMB_DIM, gamma=GAMMA):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.fsm0    = FeatureStylizationModule(gamma)

        self.layer1 = resnet.layer1
        self.fsm1   = FeatureStylizationModule(gamma)

        self.layer2 = resnet.layer2
        self.fsm2   = FeatureStylizationModule(gamma)

        self.layer3 = resnet.layer3
        self.fsm3   = FeatureStylizationModule(gamma)

        self.layer4 = resnet.layer4
        self.fsm4   = FeatureStylizationModule(gamma)

        self.avgpool = resnet.avgpool
        self.fc      = nn.Linear(resnet.fc.in_features, emb_dim)

        self.channel_sizes = [64, 64, 128, 256, 512]
        self.fsm_list      = [self.fsm0, self.fsm1,
                              self.fsm2, self.fsm3, self.fsm4]

    def activate_fsm(self):
        """Call after warmup to enable stylization."""
        for fsm in self.fsm_list:
            fsm.active = True
        print("  ✓ FSM stylization activated.")

    def forward(self, x):
        pairs = []

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        o, s = self.fsm0(x);  pairs.append((o, s));  x = s if (self.training and self.fsm0.active) else o

        x = self.layer1(x)
        o, s = self.fsm1(x);  pairs.append((o, s));  x = s if (self.training and self.fsm1.active) else o

        x = self.layer2(x)
        o, s = self.fsm2(x);  pairs.append((o, s));  x = s if (self.training and self.fsm2.active) else o

        x = self.layer3(x)
        o, s = self.fsm3(x);  pairs.append((o, s));  x = s if (self.training and self.fsm3.active) else o

        x = self.layer4(x)
        o, s = self.fsm4(x);  pairs.append((o, s));  x = s if (self.training and self.fsm4.active) else o

        x   = self.avgpool(x).flatten(1)
        emb = F.normalize(self.fc(x), p=2, dim=1)
        return emb, pairs


# ─────────────────────────────────────────────
# 6.  EVALUATION
# ─────────────────────────────────────────────
def compute_eer(gen_scores, imp_scores):
    if not gen_scores or not imp_scores:
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
def evaluate(model, reg_loader, qry_loader, epoch, phase=""):
    model.eval()
    def extract(loader):
        feats, labels = [], []
        for imgs, lbl in loader:
            emb, _ = model(imgs.to(device))
            feats.append(F.normalize(emb, p=2, dim=1).cpu())
            labels.append(lbl)
        return torch.cat(feats), torch.cat(labels)

    rf, rl = extract(reg_loader)
    qf, ql = extract(qry_loader)

    sim   = torch.mm(qf, rf.t())
    acc   = (rl[sim.argmax(dim=1)] == ql).float().mean().item() * 100

    s = sim.numpy()
    gen_s, imp_s = [], []
    for i in range(len(ql)):
        for j in range(len(rl)):
            (gen_s if ql[i] == rl[j] else imp_s).append(s[i, j])
    eer = compute_eer(gen_s, imp_s)

    tag = f"[{phase}] " if phase else ""
    print(f"\n  ┌─ {tag}Epoch {epoch}  "
          f"src={SOURCE_DOMAIN} ({len(rl)} imgs)  tgt={TARGET_DOMAIN} ({len(ql)} imgs)")
    print(f"  │  Rank-1 Accuracy : {acc:6.2f}%")
    print(f"  │  EER             : {eer:5.2f}%")
    print(f"  └{'─'*60}")
    return acc, eer


# ─────────────────────────────────────────────
# 7.  TRAINING
# ─────────────────────────────────────────────
def get_loss_weights(epoch):
    """
    Ramp α and β linearly from 0 to their final values over 20 epochs
    after the warmup ends. This prevents the large auxiliary losses from
    overwhelming ArcFace before the backbone has learned basic features.
    """
    if epoch < WARMUP_EPOCHS:
        return 0.0, 0.0
    ramp = min((epoch - WARMUP_EPOCHS) / 20.0, 1.0)
    return ALPHA_FINAL * ramp, BETA_FINAL * ramp


def train_one_epoch(model, discriminators, criterion_arc,
                    optimizer, opt_disc, loader, epoch):
    model.train()
    criterion_disc = nn.CrossEntropyLoss()
    alpha, beta = get_loss_weights(epoch)
    fsm_on      = epoch >= WARMUP_EPOCHS

    tot_loss = tot_arc = tot_de = tot_con = 0.0
    correct = total = 0
    nan_batches = 0

    for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        opt_disc.zero_grad()

        emb, pairs = model(imgs)

        loss_arc = criterion_arc(emb, labels)

        # ── NaN guard ────────────────────────────────────────────────────────
        if not torch.isfinite(loss_arc):
            nan_batches += 1
            continue

        loss_de  = torch.zeros(1, device=device)
        loss_con = torch.zeros(1, device=device)

        if fsm_on and alpha > 0:
            for k, (f_orig, f_sty) in enumerate(pairs):
                disc = discriminators[k]
                lo   = disc(f_orig)
                ls   = disc(f_sty)
                lbl1 = torch.ones(imgs.size(0),  dtype=torch.long, device=device)
                lbl0 = torch.zeros(imgs.size(0), dtype=torch.long, device=device)
                loss_de = loss_de + 0.5 * (criterion_disc(lo, lbl1) +
                                           criterion_disc(ls, lbl0))
                loss_con = loss_con + F.mse_loss(
                    f_sty.mean(dim=(-2, -1)),
                    f_orig.mean(dim=(-2, -1)).detach()
                )
            loss_de  = loss_de  / len(pairs)
            loss_con = loss_con / len(pairs)

        loss = loss_arc + alpha * loss_de + beta * loss_con

        loss.backward()
        # FIX: clip gradients before step
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(discriminators.parameters(), GRAD_CLIP)

        optimizer.step()
        opt_disc.step()

        tot_loss += loss.item()
        tot_arc  += loss_arc.item()
        tot_de   += loss_de.item()
        tot_con  += loss_con.item()

        with torch.no_grad():
            preds = criterion_arc.get_logits(emb).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    if nan_batches:
        print(f"  ⚠ {nan_batches} NaN batches skipped")

    n = max(len(loader) - nan_batches, 1)
    phase = "WARMUP" if not fsm_on else f"GIFT (α={alpha:.1f} β={beta:.1f})"
    print(f"  [{phase}]  Loss {tot_loss/n:.4f}  "
          f"(arc={tot_arc/n:.4f}  de={tot_de/n:.4f}  con={tot_con/n:.4f})  "
          f"Train-Acc={correct/max(total,1):.4f}")


# ─────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────
def main():
    label_map   = build_label_map(DATA_PATH, SOURCE_DOMAIN)
    num_classes = len(label_map)
    print(f"Identities: {num_classes}  |  src={SOURCE_DOMAIN}  tgt={TARGET_DOMAIN}")

    train_ds = CASIAMultiSpectral(DATA_PATH, SOURCE_DOMAIN, label_map, augment=True)
    reg_ds   = CASIAMultiSpectral(DATA_PATH, SOURCE_DOMAIN, label_map, augment=False)
    qry_ds   = CASIAMultiSpectral(DATA_PATH, TARGET_DOMAIN, label_map, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    reg_loader   = DataLoader(reg_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    qry_loader   = DataLoader(qry_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train {len(train_ds)} | Reg {len(reg_ds)} | Query {len(qry_ds)}")

    model          = GIFTBackbone(emb_dim=EMB_DIM, gamma=GAMMA).to(device)
    discriminators = nn.ModuleList([
        Discriminator(c).to(device) for c in model.channel_sizes
    ])

    criterion_arc = losses.ArcFaceLoss(
        num_classes=num_classes, embedding_size=EMB_DIM,
        margin=ARC_MARGIN, scale=ARC_SCALE
    ).to(device)

    optimizer = optim.RMSprop(
        list(model.parameters()) + list(criterion_arc.parameters()),
        lr=LR, weight_decay=1e-4
    )
    opt_disc = optim.RMSprop(discriminators.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1
    )

    print(f"\n{'='*60}")
    print(f"  Phase 1: {WARMUP_EPOCHS} warmup epochs  (ArcFace only, FSM off)")
    print(f"  Phase 2: {EPOCHS - WARMUP_EPOCHS} main epochs  (all losses, FSM on)")
    print(f"  α→{ALPHA_FINAL}  β→{BETA_FINAL}  γ={GAMMA}  scale={ARC_SCALE}")
    print(f"{'='*60}\n")

    best_acc = 0.0

    for epoch in range(EPOCHS):

        # Activate FSM exactly at the warmup boundary
        if epoch == WARMUP_EPOCHS:
            model.activate_fsm()

        train_one_epoch(model, discriminators, criterion_arc,
                        optimizer, opt_disc, train_loader, epoch)

        if (epoch + 1) % EVAL_EVERY == 0:
            acc, eer = evaluate(model, reg_loader, qry_loader,
                                epoch + 1, phase="GIFT")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch": epoch + 1, "model": model.state_dict(),
                    "acc": acc, "eer": eer,
                    "src": SOURCE_DOMAIN, "tgt": TARGET_DOMAIN,
                }, f"gift_best_{SOURCE_DOMAIN}_to_{TARGET_DOMAIN}.pth")
                print(f"  ✓ Best saved  (Rank-1={acc:.2f}%)")

        scheduler.step()

    print(f"\nDone.  Best Rank-1 = {best_acc:.2f}%")


if __name__ == "__main__":
    main()
