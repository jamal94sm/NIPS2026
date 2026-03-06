"""
DFMT: Distilling From Multi-Teacher  (IEEE TIM 2023, Shao & Zhong)
Fixed version — v2

Root causes of v1 not learning:
  1. MK-MMD ~5-6 units drowning ArcFace signal in teachers
     FIX: lambda_mmd=0.1 coefficient, ramped up after warmup
  2. Student distilling from untrained teachers (random→random)
     FIX: Warmup phase — student + teachers train with ArcFace only
  3. ArcFace scale=32 too aggressive for untrained features
     FIX: scale=16, warmup then normal

Configuration:
  Source  : 630
  Targets : 460, 700, 850  (N=3)
  Backbone: Paper's custom CNN  (Fig. 3) — Conv×4 + FC×3 → 128-d
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
from torchvision import transforms
from pytorch_metric_learning import losses
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────────────────
DATA_PATH      = "/home/pai-ng/Jamal/CASIA-MS-ROI"

SOURCE_DOMAIN  = "460"
TARGET_DOMAINS = ["630", "850"]
N_TARGETS      = len(TARGET_DOMAINS)

EMB_DIM        = 128
BATCH_SRC      = 32
BATCH_TGT      = 10

LR             = 1e-4       # paper: RMSprop 0.0001
ARC_MARGIN     = 0.3        # paper: m = 0.5
ARC_SCALE      = 16         # FIX: was 32; 16 stable for small datasets

# Loss weights
ALPHA          = 1.0        # L_fea_distill weight  (paper: α = 1)
BETA           = 0.5        # L_con_distill weight  (paper: β = 0.5)
LAMBDA_MMD     = 0.1        # FIX: scale MK-MMD so it doesn't crush ArcFace

# Warmup: train ONLY with ArcFace (no distillation, no MMD)
# until teachers produce meaningful features
WARMUP_EPOCHS  = 20         # FIX: was 0

EPOCHS         = 100
EVAL_EVERY     = 5
NUM_WORKERS    = 2

MMD_KERNELS    = [0.5, 1.0, 2.0, 4.0, 8.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {device}")
print(f"Source  : {SOURCE_DOMAIN}  |  Targets : {TARGET_DOMAINS}")
print(f"Warmup  : {WARMUP_EPOCHS} epochs (ArcFace only)")
print(f"Main    : {EPOCHS - WARMUP_EPOCHS} epochs (full DFMT)")


# ─────────────────────────────────────────────────────────
# 2.  DATASET
# ─────────────────────────────────────────────────────────
def build_label_map(data_path, source_domain):
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
        self.samples = []
        self.augment = augment
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
                self.samples.append((os.path.join(root, fname),
                                     label_map[hand_id]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return (self._tf_aug if self.augment else self._tf_base)(img), label


# ─────────────────────────────────────────────────────────
# 3.  BACKBONE — Paper's custom CNN  (Fig. 3)
#     Conv1: 3×3×16  stride 4  LeakyReLU + MaxPool 2×2 stride 1
#     Conv2: 5×5×32  stride 2  LeakyReLU + MaxPool 2×2 stride 1
#     Conv3: 3×3×64  stride 1  LeakyReLU
#     Conv4: 3×3×128 stride 1  LeakyReLU + MaxPool 2×2 stride 1
#     FC1: 1024  LeakyReLU
#     FC2: 512   LeakyReLU
#     FC3: 128   (no activation)
#
#     Returns (embedding, top_conv_feat):
#       embedding     — 128-d L2-normed  (ArcFace + distillation)
#       top_conv_feat — σ(Conv4 output)  (L_con_distill Eq. 7)
# ─────────────────────────────────────────────────────────
class FeatureExtractor(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.conv1  = nn.Conv2d(3,  16,  3, stride=4, padding=1)
        self.pool1  = nn.MaxPool2d(2, stride=1)
        self.conv2  = nn.Conv2d(16, 32,  5, stride=2, padding=2)
        self.pool2  = nn.MaxPool2d(2, stride=1)
        self.conv3  = nn.Conv2d(32, 64,  3, stride=1, padding=1)
        self.conv4  = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.pool3  = nn.MaxPool2d(2, stride=1)
        self.act    = nn.LeakyReLU(0.2, inplace=True)
        self.gap    = nn.AdaptiveAvgPool2d(1)   # σ(·) in Eq. 7

        # Compute flat dim after conv stack
        with torch.no_grad():
            dummy    = torch.zeros(1, 3, 112, 112)
            flat_dim = self._conv_out(dummy).shape[1]

        self.fc1 = nn.Linear(flat_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, emb_dim)

    def _conv_out(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))
        x = self.pool3(self.act(self.conv4(x)))
        return self.gap(x).flatten(1)

    def forward(self, x):
        top_conv = self._conv_out(x)                        # σ(f^con) — Eq. 7
        h        = self.act(self.fc1(top_conv))
        h        = self.act(self.fc2(h))
        emb      = F.normalize(self.fc3(h), p=2, dim=1)
        return emb, top_conv


# ─────────────────────────────────────────────────────────
# 4.  MK-MMD  (Eqs. 4–5)
# ─────────────────────────────────────────────────────────
def mk_mmd(src, tgt, kernels=MMD_KERNELS):
    n, m = src.shape[0], tgt.shape[0]

    def rbf(x, y, s):
        xx = (x * x).sum(1, keepdim=True)
        yy = (y * y).sum(1, keepdim=True)
        d  = xx + yy.t() - 2.0 * torch.mm(x, y.t())
        return torch.exp(-d / (2.0 * s))

    loss = src.new_zeros(1).squeeze()
    for s in kernels:
        kss = rbf(src, src, s); ktt = rbf(tgt, tgt, s); kst = rbf(src, tgt, s)
        loss = loss + (kss.sum()/(n*n) - 2.0*kst.sum()/(n*m) + ktt.sum()/(m*m))
    return loss / len(kernels)


# ─────────────────────────────────────────────────────────
# 5.  DISTILLATION LOSSES  (Eqs. 7–12)
# ─────────────────────────────────────────────────────────
def huber(a, b):
    d = (a - b).abs()
    return torch.where(d <= 1.0, 0.5 * d**2, d - 0.5).mean()


def l_dis_fea(f_tea, f_stu):
    """Pairwise distance matching — Eq. 8"""
    d_tea = torch.cdist(f_tea, f_tea, p=2)
    d_stu = torch.cdist(f_stu, f_stu, p=2)
    mask  = torch.triu(torch.ones_like(d_tea, dtype=torch.bool), diagonal=1)
    return huber(d_tea[mask], d_stu[mask])


def l_angle_fea(f_tea, f_stu):
    """Angle matching via consecutive-triplet approximation — Eqs. 10–12"""
    B = f_tea.shape[0]
    if B < 3:
        return f_tea.new_zeros(1).squeeze()

    def angle(fa, fb, fc):
        return (F.normalize(fa - fb, dim=1) *
                F.normalize(fc - fb, dim=1)).sum(dim=1)

    psi_t = angle(f_tea, torch.roll(f_tea, 1, 0), torch.roll(f_tea, 2, 0))
    psi_s = angle(f_stu, torch.roll(f_stu, 1, 0), torch.roll(f_stu, 2, 0))
    return huber(psi_t, psi_s)


def l_con_distill(tc_tea, tc_stu):
    """Conv-level distillation — Eq. 7"""
    return F.mse_loss(tc_stu, tc_tea.detach())


# ─────────────────────────────────────────────────────────
# 6.  EVALUATION
# ─────────────────────────────────────────────────────────
def compute_eer(gen_s, imp_s):
    if not gen_s or not imp_s:
        return float("nan")
    gen = np.array(gen_s); imp = np.array(imp_s)
    thrs = np.linspace(min(gen.min(), imp.min()),
                       max(gen.max(), imp.max()), 500)
    best = min(((abs((imp>=t).mean()-(gen<t).mean()),
                 ((imp>=t).mean()+(gen<t).mean())/2) for t in thrs),
               key=lambda x: x[0])
    return best[1] * 100


@torch.no_grad()
def evaluate(student, reg_loader, qry_loaders, epoch):
    student.eval()

    def extract(loader):
        fs, ls = [], []
        for imgs, lbl in loader:
            emb, _ = student(imgs.to(device))
            fs.append(F.normalize(emb, p=2, dim=1).cpu()); ls.append(lbl)
        return torch.cat(fs), torch.cat(ls)

    rf, rl = extract(reg_loader)
    print(f"\n  ┌─ Epoch {epoch}  src={SOURCE_DOMAIN}  reg={len(rl)} imgs")

    results = {}
    all_qf, all_ql = [], []   # pooled across all target domains

    for dom, qry_loader in qry_loaders.items():
        qf, ql = extract(qry_loader)
        all_qf.append(qf); all_ql.append(ql)

        sim = torch.mm(qf, rf.t())
        acc = (rl[sim.argmax(1)] == ql).float().mean().item() * 100
        s   = sim.numpy()
        gen_s, imp_s = [], []
        for i in range(len(ql)):
            for j in range(len(rl)):
                (gen_s if ql[i]==rl[j] else imp_s).append(s[i,j])
        eer = compute_eer(gen_s, imp_s)
        print(f"  │  [{dom}]  Rank-1={acc:6.2f}%  EER={eer:5.2f}%")
        results[dom] = (acc, eer)

    # ── Total accuracy across ALL target domains pooled ───────────────────
    all_qf = torch.cat(all_qf)   # (total_queries, D)
    all_ql = torch.cat(all_ql)   # (total_queries,)
    sim_all = torch.mm(all_qf, rf.t())
    total_acc = (rl[sim_all.argmax(1)] == all_ql).float().mean().item() * 100
    s_all     = sim_all.numpy()
    gen_all, imp_all = [], []
    for i in range(len(all_ql)):
        for j in range(len(rl)):
            (gen_all if all_ql[i]==rl[j] else imp_all).append(s_all[i,j])
    total_eer = compute_eer(gen_all, imp_all)

    print(f"  ├─ {'─'*53}")
    print(f"  │  [TOTAL]  Rank-1={total_acc:6.2f}%  EER={total_eer:5.2f}%"
          f"  ({len(all_ql)} query imgs across {len(qry_loaders)} domains)")
    print(f"  └{'─'*55}")

    results["TOTAL"] = (total_acc, total_eer)
    return results


# ─────────────────────────────────────────────────────────
# 7.  INFINITE LOADER
# ─────────────────────────────────────────────────────────
def inf_loader(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        drop_last=True)
    while True:
        yield from loader


# ─────────────────────────────────────────────────────────
# 8.  TRAINING — Algorithm 1
# ─────────────────────────────────────────────────────────
def get_weights(epoch):
    """
    Ramp up auxiliary losses after warmup.
    Warmup  : only ArcFace — lets backbone features stabilise first
    Ramp    : linearly ramp MMD and distillation over 20 epochs
    Full    : paper's original weights
    """
    if epoch < WARMUP_EPOCHS:
        return 0.0, 0.0, 0.0     # lambda_mmd, alpha, beta  all off
    ramp = min((epoch - WARMUP_EPOCHS) / 20.0, 1.0)
    return LAMBDA_MMD * ramp, ALPHA * ramp, BETA * ramp


def train_one_epoch(teachers, student, tea_optims, stu_optim,
                    arc_criteria, src_iter, tgt_iters,
                    n_batches, epoch):
    for m in teachers: m.train()
    student.train()
    for c in arc_criteria: c.train()

    tea_arcs = arc_criteria[:N_TARGETS]
    stu_arc  = arc_criteria[N_TARGETS]

    lmd_mmd, alpha, beta = get_weights(epoch)
    warmup = (epoch < WARMUP_EPOCHS)
    phase  = "WARMUP" if warmup else f"DFMT (λ={lmd_mmd:.3f} α={alpha:.1f} β={beta:.2f})"

    tot_tea = tot_stu = 0.0
    correct = total   = 0

    for _ in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{EPOCHS} [{phase}]"):

        # ── Source + target batches ───────────────────────────────────────
        src_imgs, src_lbl = next(src_iter)
        src_imgs = src_imgs.to(device)
        src_lbl  = src_lbl.to(device)
        tgt_imgs = [next(it)[0].to(device) for it in tgt_iters]

        # ══════════════════════════════════════════════════════════════════
        # STEP 2 — Teachers: ArcFace + (MK-MMD if not warmup)
        # ══════════════════════════════════════════════════════════════════
        for j, (teacher, t_opt, t_arc) in enumerate(
                zip(teachers, tea_optims, tea_arcs)):

            t_opt.zero_grad()
            src_emb, _ = teacher(src_imgs)
            loss_tea   = t_arc(src_emb, src_lbl)

            if not warmup:
                tgt_emb, _ = teacher(tgt_imgs[j])
                loss_tea   = loss_tea + lmd_mmd * mk_mmd(src_emb, tgt_emb)

            if torch.isfinite(loss_tea):
                loss_tea.backward()
                nn.utils.clip_grad_norm_(teacher.parameters(), 5.0)
                t_opt.step()
                tot_tea += loss_tea.item()

        # ══════════════════════════════════════════════════════════════════
        # STEP 3 — Student: ArcFace + distillation from each teacher in turn
        # ══════════════════════════════════════════════════════════════════
        stu_optim.zero_grad()
        stu_emb_src, stu_conv_src = student(src_imgs)
        loss_stu = stu_arc(stu_emb_src, src_lbl)

        if not warmup:
            loss_distill = src_imgs.new_zeros(1).squeeze()
            for j, teacher in enumerate(teachers):
                with torch.no_grad():
                    tea_emb_src, _            = teacher(src_imgs)
                    _,           tea_conv_tgt = teacher(tgt_imgs[j])

                ld = l_dis_fea(tea_emb_src, stu_emb_src)
                la = l_angle_fea(tea_emb_src, stu_emb_src)

                _, stu_conv_tgt = student(tgt_imgs[j])
                lc = l_con_distill(tea_conv_tgt, stu_conv_tgt)

                loss_distill = loss_distill + alpha*(ld+la) + beta*lc

            loss_stu = loss_stu + loss_distill / N_TARGETS

        if torch.isfinite(loss_stu):
            loss_stu.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            stu_optim.step()
            tot_stu += loss_stu.item()

        with torch.no_grad():
            preds    = stu_arc.get_logits(stu_emb_src).argmax(1)
            correct += (preds == src_lbl).sum().item()
            total   += src_lbl.size(0)

    n = max(n_batches, 1)
    print(f"  Tea={tot_tea/n:.4f}  Stu={tot_stu/n:.4f}  "
          f"Train-Acc={correct/max(total,1):.4f}")


# ─────────────────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────────────────
def main():
    label_map   = build_label_map(DATA_PATH, SOURCE_DOMAIN)
    num_classes = len(label_map)
    print(f"Identities: {num_classes}")

    src_train = CASIADomain(DATA_PATH, SOURCE_DOMAIN, label_map, augment=True)
    src_reg   = CASIADomain(DATA_PATH, SOURCE_DOMAIN, label_map, augment=False)
    tgt_train = [CASIADomain(DATA_PATH, d, label_map, augment=True)
                 for d in TARGET_DOMAINS]
    tgt_qry   = [CASIADomain(DATA_PATH, d, label_map, augment=False)
                 for d in TARGET_DOMAINS]

    reg_loader  = DataLoader(src_reg,  batch_size=64, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    qry_loaders = {d: DataLoader(ds, batch_size=64, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)
                   for d, ds in zip(TARGET_DOMAINS, tgt_qry)}

    src_iter  = inf_loader(src_train, BATCH_SRC)
    tgt_iters = [inf_loader(ds, BATCH_TGT) for ds in tgt_train]
    n_batches = math.ceil(len(src_train) / BATCH_SRC)

    # ── Models ───────────────────────────────────────────────────────────
    teachers = nn.ModuleList([FeatureExtractor().to(device)
                               for _ in range(N_TARGETS)])
    student  = FeatureExtractor().to(device)

    arc_criteria = nn.ModuleList([
        losses.ArcFaceLoss(num_classes=num_classes, embedding_size=EMB_DIM,
                           margin=ARC_MARGIN, scale=ARC_SCALE).to(device)
        for _ in range(N_TARGETS + 1)
    ])

    # ── Optimizers ───────────────────────────────────────────────────────
    tea_optims = [
        optim.RMSprop(list(teachers[j].parameters()) +
                      list(arc_criteria[j].parameters()),
                      lr=LR, weight_decay=1e-4)
        for j in range(N_TARGETS)
    ]
    stu_optim = optim.RMSprop(
        list(student.parameters()) + list(arc_criteria[N_TARGETS].parameters()),
        lr=LR, weight_decay=1e-4
    )
    schedulers = [optim.lr_scheduler.StepLR(o, step_size=30, gamma=0.1)
                  for o in tea_optims + [stu_optim]]

    print(f"\n{'='*60}")
    print(f"  DFMT v2  src={SOURCE_DOMAIN}  →  {TARGET_DOMAINS}")
    print(f"  Backbone : Paper CNN (Conv×4 + FC×3, scratch)")
    print(f"  α={ALPHA}  β={BETA}  λ_mmd={LAMBDA_MMD}  scale={ARC_SCALE}")
    print(f"  Warmup={WARMUP_EPOCHS} epochs  Total={EPOCHS} epochs")
    print(f"{'='*60}\n")

    best_mean_acc = 0.0

    for epoch in range(EPOCHS):
        train_one_epoch(teachers, student, tea_optims, stu_optim,
                        list(arc_criteria), src_iter, tgt_iters,
                        n_batches, epoch)

        if (epoch + 1) % EVAL_EVERY == 0:
            results      = evaluate(student, reg_loader, qry_loaders, epoch+1)
            total_acc    = results["TOTAL"][0]
            total_eer    = results["TOTAL"][1]
            domain_accs  = [v[0] for k, v in results.items() if k != "TOTAL"]
            mean_acc     = np.mean(domain_accs)
            print(f"  Mean domain Rank-1={mean_acc:.2f}%  "
                  f"Total Rank-1={total_acc:.2f}%  Total EER={total_eer:.2f}%")

            if total_acc > best_mean_acc:
                best_mean_acc = total_acc
                torch.save({
                    "epoch":    epoch + 1,
                    "student":  student.state_dict(),
                    "teachers": [t.state_dict() for t in teachers],
                    "results":  results,
                }, f"dfmt_v2_best_{SOURCE_DOMAIN}.pth")
                print(f"  ✓ Best saved  (Total Rank-1={total_acc:.2f}%)")

        for s in schedulers: s.step()

    print(f"\nDone.  Best mean Rank-1 = {best_mean_acc:.2f}%")


if __name__ == "__main__":
    main()
