"""
DFMT: Distilling From Multi-Teacher for Multi-Target Cross-Dataset
Palmprint Recognition
IEEE Transactions on Instrumentation and Measurement, Vol. 72, 2023
Shao & Zhong

Configuration:
  Source  : 630
  Targets : 460, 700, 850
  Backbone: Paper's custom CNN (Fig. 3) — Conv×4 + FC×3 → 128-d

Algorithm 1 (faithful reproduction):
  Each iteration:
    1. Sample n source images + m target images per target domain
    2. Update each teacher  with  L_teacher = L_arc + L_mkmmd
    3. Update student        with  L_student = L_arc + α·L_fea_distill
                                               + β·L_con_distill
       where L_fea_distill = L_dis_fea + L_angle_fea   (Eqs. 8–12)
             L_con_distill  = ||σ(f_tea_conv) - σ(f_stu_conv)||₂  (Eq. 7)
       Student receives distillation from each teacher IN TURN within
       the same batch (Algorithm 1, lines 5–6 are inside the same loop).
"""

import os
import itertools
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from pytorch_metric_learning import losses
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────────────────
DATA_PATH      = "/home/pai-ng/Jamal/CASIA-MS-ROI"

SOURCE_DOMAIN  = "630"
TARGET_DOMAINS = ["460", "700", "850"]   # N = 3

EMB_DIM        = 128
BATCH_SRC      = 32    # n  — source images per iteration
BATCH_TGT      = 10    # m  — target images per target domain per iteration
                        #      total batch ≈ 32 + 3×10 = 62

LR             = 1e-4  # paper: 0.0001
ARC_MARGIN     = 0.5   # paper: m = 0.5
ARC_SCALE      = 32    # not stated; 32 is safe for 200 classes
ALPHA          = 1.0   # weight for L_fea_distill  (paper: α = 1)
BETA           = 0.5   # weight for L_con_distill  (paper: β = 0.5)
EPOCHS         = 100
EVAL_EVERY     = 5
NUM_WORKERS    = 2

# MK-MMD kernel bandwidths (standard choice from literature)
MMD_KERNELS    = [0.5, 1.0, 2.0, 4.0, 8.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"Source : {SOURCE_DOMAIN}  |  Targets : {TARGET_DOMAINS}")


# ─────────────────────────────────────────────────────────
# 2.  DATASET
# ─────────────────────────────────────────────────────────
def build_label_map(data_path, source_domain):
    """Identity labels are built from the source domain only."""
    hand_ids = set()
    for root, _, files in os.walk(data_path):
        for fname in sorted(files):
            if not fname.lower().endswith(".jpg"):
                continue
            parts = fname[:-4].split("_")
            if len(parts) != 4:
                continue
            subj, hand, spectrum, _ = parts
            if spectrum == source_domain:
                hand_ids.add(f"{subj}_{hand}")
    return {h: i for i, h in enumerate(sorted(hand_ids))}


class CASIADomain(Dataset):
    """Single-domain loader returning (image_tensor, identity_label)."""

    _tf_base = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    _tf_aug = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, data_path, domain, label_map, augment=False):
        self.samples  = []
        self.augment  = augment
        for root, _, files in os.walk(data_path):
            for fname in sorted(files):
                if not fname.lower().endswith(".jpg"):
                    continue
                parts = fname[:-4].split("_")
                if len(parts) != 4:
                    continue
                subj, hand, spectrum, _ = parts
                if spectrum != domain:
                    continue
                hand_id = f"{subj}_{hand}"
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
        tf  = self._tf_aug if self.augment else self._tf_base
        return tf(img), label


# ─────────────────────────────────────────────────────────
# 3.  BACKBONE  — paper Fig. 3
#     Conv1: 3×3×16, stride 4  + MaxPool 2×2 stride 1
#     Conv2: 5×5×32, stride 2  + MaxPool 2×2 stride 1
#     Conv3: 3×3×64, stride 1
#     Conv4: 3×3×128, stride 1 + MaxPool 2×2 stride 1
#     FC1: 1024  LeakyReLU
#     FC2: 512   LeakyReLU
#     FC3: 128   (no activation — raw embedding)
# ─────────────────────────────────────────────────────────
class PaperCNN(nn.Module):
    """
    Exact architecture from Fig. 3.
    Returns (embedding, top_conv_feat) where:
      embedding     — 128-d L2-normalised feature  (for ArcFace + L_fea_distill)
      top_conv_feat — spatially-pooled output of Conv4  (for L_con_distill Eq. 7)
    """

    def __init__(self):
        super().__init__()
        # ── Convolutional stem ──────────────────────────────────────────────
        self.conv1   = nn.Conv2d(3, 16,  3, stride=4, padding=1)
        self.pool1   = nn.MaxPool2d(2, stride=1)

        self.conv2   = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.pool2   = nn.MaxPool2d(2, stride=1)

        self.conv3   = nn.Conv2d(32, 64, 3, stride=1, padding=1)

        self.conv4   = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.pool3   = nn.MaxPool2d(2, stride=1)

        self.act     = nn.LeakyReLU(0.2, inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)   # σ(·) in Eq. 7

        # ── Fully-connected head ────────────────────────────────────────────
        # Compute flat dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 112, 112)
            flat  = self._conv_forward(dummy).shape[1]

        self.fc1 = nn.Linear(flat, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)

    def _conv_forward(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))
        x = self.pool3(self.act(self.conv4(x)))
        return self.avgpool(x).flatten(1)

    def forward(self, x):
        # Top conv output for L_con_distill (before FC, after spatial pooling)
        x  = self.pool1(self.act(self.conv1(x)))
        x  = self.pool2(self.act(self.conv2(x)))
        x  = self.act(self.conv3(x))
        x  = self.pool3(self.act(self.conv4(x)))
        top_conv = self.avgpool(x).flatten(1)   # σ(f^con(·))  — Eq. 7

        h  = self.act(self.fc1(top_conv))
        h  = self.act(self.fc2(h))
        emb = self.fc3(h)
        emb = F.normalize(emb, p=2, dim=1)     # L2-normalise
        return emb, top_conv


# ─────────────────────────────────────────────────────────
# 4.  MK-MMD  (Eq. 4–5)
# ─────────────────────────────────────────────────────────
def mk_mmd(source_feats, target_feats, kernels=MMD_KERNELS):
    """
    Multiple-Kernel Maximum Mean Discrepancy.
    source_feats, target_feats : (n, d) and (m, d) tensors
    kernels                    : list of RBF bandwidths σ²
    Returns scalar MMD² loss.
    """
    n, m = source_feats.shape[0], target_feats.shape[0]

    def rbf_kernel(x, y, sigma):
        # (|x|², |y|²) pairwise squared distances
        xx = (x * x).sum(1, keepdim=True)          # (n,1)
        yy = (y * y).sum(1, keepdim=True)          # (m,1)
        dist = xx + yy.t() - 2.0 * torch.mm(x, y.t())  # (n,m)
        return torch.exp(-dist / (2.0 * sigma))

    # Eq. 4: k(xS,xS) - 2·k(xS,xT) + k(xT,xT)  summed over kernels
    loss = torch.tensor(0.0, device=source_feats.device)
    for sigma in kernels:
        kss = rbf_kernel(source_feats, source_feats, sigma)  # (n,n)
        ktt = rbf_kernel(target_feats, target_feats, sigma)  # (m,m)
        kst = rbf_kernel(source_feats, target_feats, sigma)  # (n,m)
        loss = loss + (kss.sum() / (n * n)
                       - 2.0 * kst.sum() / (n * m)
                       + ktt.sum() / (m * m))
    return loss / len(kernels)


# ─────────────────────────────────────────────────────────
# 5.  DISTILLATION LOSSES
# ─────────────────────────────────────────────────────────
def huber(a, b):
    """Scalar Huber loss between two tensors (Eq. 9)."""
    diff = (a - b).abs()
    return torch.where(diff <= 1.0,
                       0.5 * diff ** 2,
                       diff - 0.5).mean()


def l_dis_fea(f_tea, f_stu):
    """
    Distance-based feature distillation  (Eq. 8).
    For every pair (i, j) in the batch, match pairwise cosine distances.
    Using all O(B²) pairs is expensive; we sample consecutive pairs.
    """
    # Pairwise L2 distances  (B, B)
    d_tea = torch.cdist(f_tea, f_tea, p=2)
    d_stu = torch.cdist(f_stu, f_stu, p=2)
    # Upper-triangle only (avoid diagonal & double-counting)
    mask  = torch.triu(torch.ones_like(d_tea, dtype=torch.bool), diagonal=1)
    return huber(d_tea[mask], d_stu[mask])


def l_angle_fea(f_tea, f_stu):
    """
    Angle-based feature distillation  (Eqs. 10–12).
    For each anchor j, compute angles ψ(i,j,k) using all ordered
    (i, k) pairs where i ≠ j and k ≠ j and i ≠ k.
    Approximation: use consecutive triplets to keep cost O(B).
    """
    B = f_tea.shape[0]
    if B < 3:
        return torch.tensor(0.0, device=f_tea.device)

    # Roll to get shifted versions → fast consecutive-triplet approximation
    fi = f_tea                          # anchor i
    fj = torch.roll(f_tea, 1, dims=0)  # anchor j
    fk = torch.roll(f_tea, 2, dims=0)  # anchor k

    fi_s = f_stu
    fj_s = torch.roll(f_stu, 1, dims=0)
    fk_s = torch.roll(f_stu, 2, dims=0)

    def angle(fa, fb, fc):
        eij = F.normalize(fa - fb, dim=1)
        ekj = F.normalize(fc - fb, dim=1)
        return (eij * ekj).sum(dim=1)   # cos of angle at j

    psi_tea = angle(fi,   fj,   fk)
    psi_stu = angle(fi_s, fj_s, fk_s)
    return huber(psi_tea, psi_stu)


def l_con_distill(top_conv_tea, top_conv_stu):
    """
    Convolution-level distillation  (Eq. 7).
    top_conv_tea, top_conv_stu : (B, C) — σ(f^con(·)) already pooled.
    """
    return F.mse_loss(top_conv_stu, top_conv_tea.detach())


# ─────────────────────────────────────────────────────────
# 6.  EVALUATION
# ─────────────────────────────────────────────────────────
def compute_eer(gen_s, imp_s):
    if not gen_s or not imp_s:
        return float("nan")
    gen = np.array(gen_s); imp = np.array(imp_s)
    thrs = np.linspace(min(gen.min(), imp.min()),
                       max(gen.max(), imp.max()), 500)
    best = min(
        ((abs((imp >= t).mean() - (gen < t).mean()),
          ((imp >= t).mean() + (gen < t).mean()) / 2)
         for t in thrs),
        key=lambda x: x[0]
    )
    return best[1] * 100


@torch.no_grad()
def _extract(model, loader):
    model.eval()
    feats, labels = [], []
    for imgs, lbl in loader:
        emb, _ = model(imgs.to(device))
        feats.append(F.normalize(emb, p=2, dim=1).cpu())
        labels.append(lbl)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def evaluate(student, reg_loader, qry_loaders, epoch):
    """
    Identification (Rank-1 Acc) and Verification (EER).
    reg_loader  : source domain (registration set)
    qry_loaders : dict {domain: loader}  (query sets — one per target)
    """
    rf, rl = _extract(student, reg_loader)

    print(f"\n  ┌─ Epoch {epoch}  src={SOURCE_DOMAIN} ({len(rl)} imgs)")
    results = {}
    for dom, qry_loader in qry_loaders.items():
        qf, ql = _extract(student, qry_loader)
        sim     = torch.mm(qf, rf.t())
        acc     = (rl[sim.argmax(1)] == ql).float().mean().item() * 100
        s = sim.numpy()
        gen_s, imp_s = [], []
        for i in range(len(ql)):
            for j in range(len(rl)):
                (gen_s if ql[i] == rl[j] else imp_s).append(s[i, j])
        eer = compute_eer(gen_s, imp_s)
        print(f"  │  [{dom}]  Rank-1={acc:6.2f}%  EER={eer:5.2f}%")
        results[dom] = (acc, eer)
    print(f"  └{'─'*55}")
    return results


# ─────────────────────────────────────────────────────────
# 7.  INFINITE DOMAIN ITERATORS  (for mixed batches)
# ─────────────────────────────────────────────────────────
def inf_loader(dataset, batch_size):
    """Yields batches endlessly from a dataset."""
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=NUM_WORKERS,
                        pin_memory=True, drop_last=True)
    while True:
        yield from loader


# ─────────────────────────────────────────────────────────
# 8.  TRAINING
# ─────────────────────────────────────────────────────────
def train_one_epoch(teachers, student,
                    tea_optims, stu_optim,
                    arc_criteria,        # list[ArcFaceLoss], one per teacher + 1 student
                    src_iter, tgt_iters,
                    n_batches, epoch):
    """
    One epoch following Algorithm 1:
      for each iteration:
        step 2: update teachers (L_arc + MK-MMD)
        step 3: update student  (L_arc + L_fea_distill + L_con_distill)
    """
    N = len(TARGET_DOMAINS)
    for m in teachers: m.train()
    student.train()
    for c in arc_criteria: c.train()

    tea_arc_loss   = arc_criteria[:N]    # one ArcFace per teacher
    stu_arc_loss   = arc_criteria[N]     # student ArcFace

    tot_tea = tot_stu = 0.0
    correct = total  = 0

    for _ in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{EPOCHS}"):

        # ── Sample source batch ───────────────────────────────────────────
        src_imgs, src_lbl = next(src_iter)
        src_imgs = src_imgs.to(device)
        src_lbl  = src_lbl.to(device)

        # ── Sample one target batch per domain ───────────────────────────
        tgt_batches = []
        for it in tgt_iters:
            imgs, _ = next(it)          # target labels not used (unsupervised)
            tgt_batches.append(imgs.to(device))

        # ══════════════════════════════════════════════════════════════════
        # STEP 2 — Update each teacher independently
        # L_teacher = L_arc(source) + L_mkmmd(source, target_j)   (Eq. 6)
        # ══════════════════════════════════════════════════════════════════
        for j, (teacher, t_opt, t_arc) in enumerate(
                zip(teachers, tea_optims, tea_arc_loss)):

            t_opt.zero_grad()

            src_emb, _ = teacher(src_imgs)
            tgt_emb, _ = teacher(tgt_batches[j])

            loss_arc  = t_arc(src_emb, src_lbl)
            loss_mmd  = mk_mmd(src_emb, tgt_emb)
            loss_tea  = loss_arc + loss_mmd

            if torch.isfinite(loss_tea):
                loss_tea.backward()
                nn.utils.clip_grad_norm_(teacher.parameters(), 5.0)
                t_opt.step()
                tot_tea += loss_tea.item()

        # ══════════════════════════════════════════════════════════════════
        # STEP 3 — Update student, receiving distillation from each teacher
        # IN TURN (Algorithm 1 line 6 is inside the same for-loop as line 5)
        # L_student = L_arc + α·(L_dis_fea + L_angle_fea) + β·L_con_distill
        # ══════════════════════════════════════════════════════════════════
        stu_optim.zero_grad()

        # Student forward on source (for ArcFace)
        stu_emb_src, stu_conv_src = student(src_imgs)
        loss_stu = stu_arc_loss(stu_emb_src, src_lbl)

        # Distillation from each teacher in turn
        # Paper: student receives from each teacher sequentially within
        # the iteration → accumulate distillation losses across teachers
        loss_distill = torch.tensor(0.0, device=device)
        for j, teacher in enumerate(teachers):
            # Teacher inference on source (no grad needed for teacher targets)
            with torch.no_grad():
                tea_emb_src, tea_conv_src = teacher(src_imgs)

            # L_fea_distill = L_dis_fea + L_angle_fea  (Eq. 13)
            ld = l_dis_fea(tea_emb_src, stu_emb_src)
            la = l_angle_fea(tea_emb_src, stu_emb_src)

            # Student also processes target images for conv-level distillation
            tgt_imgs_j = tgt_batches[j]
            stu_emb_tgt, stu_conv_tgt = student(tgt_imgs_j)
            with torch.no_grad():
                _, tea_conv_tgt = teacher(tgt_imgs_j)

            lc = l_con_distill(tea_conv_tgt, stu_conv_tgt)

            loss_distill = loss_distill + (ALPHA * (ld + la) + BETA * lc)

        loss_distill = loss_distill / N   # average over teachers
        loss_stu_total = loss_stu + loss_distill

        if torch.isfinite(loss_stu_total):
            loss_stu_total.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            stu_optim.step()
            tot_stu += loss_stu_total.item()

        # Track student train accuracy on source
        with torch.no_grad():
            preds = stu_arc_loss.get_logits(stu_emb_src).argmax(1)
            correct += (preds == src_lbl).sum().item()
            total   += src_lbl.size(0)

    print(f"  Tea-loss={tot_tea/n_batches:.4f}  "
          f"Stu-loss={tot_stu/n_batches:.4f}  "
          f"Train-Acc={correct/max(total,1):.4f}")


# ─────────────────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────────────────
def main():
    N = len(TARGET_DOMAINS)

    # ── Label map from source domain ────────────────────────────────────────
    label_map   = build_label_map(DATA_PATH, SOURCE_DOMAIN)
    num_classes = len(label_map)
    print(f"Identities: {num_classes}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    src_train_ds = CASIADomain(DATA_PATH, SOURCE_DOMAIN, label_map, augment=True)
    src_reg_ds   = CASIADomain(DATA_PATH, SOURCE_DOMAIN, label_map, augment=False)
    tgt_train_ds = [CASIADomain(DATA_PATH, d, label_map, augment=True)
                    for d in TARGET_DOMAINS]
    tgt_qry_ds   = [CASIADomain(DATA_PATH, d, label_map, augment=False)
                    for d in TARGET_DOMAINS]

    reg_loader = DataLoader(src_reg_ds, batch_size=64,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True)
    qry_loaders = {d: DataLoader(ds, batch_size=64, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)
                   for d, ds in zip(TARGET_DOMAINS, tgt_qry_ds)}

    # Infinite iterators for training
    src_iter  = inf_loader(src_train_ds, BATCH_SRC)
    tgt_iters = [inf_loader(ds, BATCH_TGT) for ds in tgt_train_ds]

    # Steps per epoch = ceil(source dataset / BATCH_SRC)
    n_batches = math.ceil(len(src_train_ds) / BATCH_SRC)

    print(f"Source train: {len(src_train_ds)}  |  "
          + "  ".join(f"{d}: {len(ds)}"
                      for d, ds in zip(TARGET_DOMAINS, tgt_train_ds)))

    # ── Models ───────────────────────────────────────────────────────────────
    # N teachers + 1 student, same architecture
    teachers = nn.ModuleList([PaperCNN().to(device) for _ in range(N)])
    student  = PaperCNN().to(device)

    # ── ArcFace: one per teacher + one for student ───────────────────────────
    arc_criteria = nn.ModuleList([
        losses.ArcFaceLoss(num_classes=num_classes,
                           embedding_size=EMB_DIM,
                           margin=ARC_MARGIN,
                           scale=ARC_SCALE).to(device)
        for _ in range(N + 1)
    ])

    # ── Optimizers (RMSprop, lr=0.0001 — paper Sec. IV-B) ───────────────────
    tea_optims = [
        optim.RMSprop(
            list(teachers[j].parameters()) +
            list(arc_criteria[j].parameters()),
            lr=LR, weight_decay=1e-4
        )
        for j in range(N)
    ]
    stu_optim = optim.RMSprop(
        list(student.parameters()) +
        list(arc_criteria[N].parameters()),
        lr=LR, weight_decay=1e-4
    )

    schedulers = [
        optim.lr_scheduler.StepLR(o, step_size=30, gamma=0.1)
        for o in tea_optims + [stu_optim]
    ]

    # ── Training ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DFMT  src={SOURCE_DOMAIN}  →  tgts={TARGET_DOMAINS}")
    print(f"  α={ALPHA}  β={BETA}  lr={LR}  arc_m={ARC_MARGIN}")
    print(f"{'='*60}\n")

    best_mean_acc = 0.0

    for epoch in range(EPOCHS):
        train_one_epoch(
            teachers, student,
            tea_optims, stu_optim,
            list(arc_criteria),
            src_iter, tgt_iters,
            n_batches, epoch
        )

        if (epoch + 1) % EVAL_EVERY == 0:
            results = evaluate(student, reg_loader, qry_loaders, epoch + 1)
            mean_acc = np.mean([v[0] for v in results.values()])
            mean_eer = np.mean([v[1] for v in results.values()])
            print(f"  Mean Rank-1={mean_acc:.2f}%  Mean EER={mean_eer:.2f}%")

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc
                torch.save({
                    "epoch":    epoch + 1,
                    "student":  student.state_dict(),
                    "teachers": [t.state_dict() for t in teachers],
                    "results":  results,
                }, f"dfmt_best_{SOURCE_DOMAIN}_to_{'_'.join(TARGET_DOMAINS)}.pth")
                print(f"  ✓ Best student saved  (mean Rank-1={mean_acc:.2f}%)")

        for s in schedulers:
            s.step()

    print(f"\nDone.  Best mean Rank-1 = {best_mean_acc:.2f}%")


# ─────────────────────────────────────────────────────────
# missing import
# ─────────────────────────────────────────────────────────
import math

if __name__ == "__main__":
    main()
