"""
ArcFace iResNet100 — Cross-Domain Open-Set Evaluation on Palm-Auth
===================================================================
Backbone : iResNet100 pretrained on Glint360K (InsightFace ONNX)
           loaded via onnx2torch, first 75% of tensors frozen
Loss     : ArcFace (additive angular margin, s=64, m=0.50)
Input    : 112×112 RGB, mean=0.5, std=0.5

Open-set protocol:
  IDs split 80/20 into disjoint train/test partitions.
  Gallery and probe built from TEST IDs only (50/50 sample split).
  Splits saved to palm_auth_openset_splits.json for fair comparison.

Results saved to:
  {SAVE_DIR}/eval/
  {SAVE_DIR}/best_model.pth
  {SAVE_DIR}/results.json
"""

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "palm_auth_data_root"  : "/home/pai-ng/Jamal/smartphone_data",
    "scanner_spectra"      : {"green", "ir", "yellow", "pink", "white"},
    "pretrained_weights"   : "/home/pai-ng/Jamal/NIPS2026/face_models/checkpoints/r100_glint360k.onnx",

    "train_id_ratio"       : 0.80,
    "test_gallery_ratio"   : 0.50,
    "splits_file"          : "./palm_auth_openset_splits.json",

    "arcface_s"            : 64.0,
    "arcface_m"            : 0.50,

    "img_side"             : 112,
    "batch_size"           : 32,
    "num_epochs"           : 100,
    "lr"                   : 1e-4,
    "weight_decay"         : 5e-4,
    "eval_every"           : 5,
    "num_workers"          : 4,

    "save_dir"             : "./rst_arcface_openset",
    "random_seed"          : 42,
}
# ─────────────────────────────────────────────────────────────

import os, json, math, time, random, warnings
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")
IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


# ══════════════════════════════════════════════════════════════
#  MODEL  (unchanged)
# ══════════════════════════════════════════════════════════════

class ArcFaceBackbone(nn.Module):
    def __init__(self, pretrained_path, freeze_ratio=0.75):
        super().__init__()
        import onnx
        from onnx2torch import convert
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"ONNX not found: {pretrained_path}")
        print(f"  Loading ONNX model: {pretrained_path}")
        self.net = convert(onnx.load(pretrained_path))
        all_params = list(self.net.parameters())
        n_freeze   = int(len(all_params) * freeze_ratio)
        for i, p in enumerate(all_params):
            p.requires_grad = (i >= n_freeze)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (list, tuple)): out = out[0]
        return F.normalize(out, p=2, dim=1)


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size=512, s=64.0, m=0.50):
        super().__init__()
        self.s     = s
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        W           = F.normalize(self.weight, p=2, dim=1)
        cos_theta   = (embeddings @ W.t()).clamp(-1+1e-7, 1-1e-7)
        sin_theta   = (1.0 - cos_theta**2).sqrt()
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)
        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1,1), 1.0)
        logits  = self.s * (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta)
        return self.ce(logits, labels)

    @torch.no_grad()
    def get_logits(self, embeddings):
        W = F.normalize(self.weight, p=2, dim=1)
        return self.s * (embeddings @ W.t())


# ══════════════════════════════════════════════════════════════
#  TRANSFORMS & DATASETS
# ══════════════════════════════════════════════════════════════

def _base_tf(img_side):
    return transforms.Compose([
        transforms.Resize((img_side, img_side)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

def _aug_tf(img_side):
    return transforms.Compose([
        transforms.Resize((img_side, img_side)),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
            transforms.RandomResizedCrop(img_side, scale=(0.8,1.0), ratio=(1.0,1.0)),
            transforms.RandomPerspective(distortion_scale=0.15, p=1),
            transforms.RandomChoice([
                transforms.RandomRotation(10, interpolation=Image.BICUBIC,
                                          expand=False, center=(int(0.5*img_side),0)),
                transforms.RandomRotation(10, interpolation=Image.BICUBIC,
                                          expand=False, center=(0,int(0.5*img_side))),
            ]),
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])


class TrainDataset(Dataset):
    def __init__(self, samples, img_side):
        self.samples   = samples
        self.transform = _aug_tf(img_side)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


class EvalDataset(Dataset):
    def __init__(self, samples, img_side):
        self.samples   = samples
        self.transform = _base_tf(img_side)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


def make_loader(samples, train, cfg):
    ds = TrainDataset(samples, cfg["img_side"]) if train \
         else EvalDataset(samples, cfg["img_side"])
    return DataLoader(ds, batch_size=min(cfg["batch_size"], len(samples)),
                      shuffle=train, num_workers=cfg["num_workers"],
                      pin_memory=True, drop_last=train and len(samples)>cfg["batch_size"])


# ══════════════════════════════════════════════════════════════
#  DATA COLLECTION & OPEN-SET SPLITS
# ══════════════════════════════════════════════════════════════

def collect_palm_auth(data_root, scanner_spectra):
    id2paths = defaultdict(list)
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir = os.path.join(data_root, subject_id)
        if not os.path.isdir(subject_dir): continue
        roi_dir = os.path.join(subject_dir, "roi_perspective")
        if os.path.isdir(roi_dir):
            for fname in sorted(os.listdir(roi_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                parts = os.path.splitext(fname)[0].split("_")
                if len(parts) < 3: continue
                id2paths[parts[0]+"_"+parts[1].lower()].append(os.path.join(roi_dir, fname))
        scan_dir = os.path.join(subject_dir, "roi_scanner")
        if os.path.isdir(scan_dir):
            for fname in sorted(os.listdir(scan_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                parts = os.path.splitext(fname)[0].split("_")
                if len(parts) < 4: continue
                if parts[2].lower() not in scanner_spectra: continue
                id2paths[parts[0]+"_"+parts[1].lower()].append(os.path.join(scan_dir, fname))
    result = dict(id2paths); counts = [len(v) for v in result.values()]
    print(f"  [Palm-Auth] ids={len(result)}  total={sum(counts)}  "
          f"min/max/mean={min(counts)}/{max(counts)}/{sum(counts)/len(counts):.1f}")
    return result


def load_or_generate_splits(id2paths, train_ratio, gallery_ratio, seed, splits_file):
    if os.path.exists(splits_file):
        with open(splits_file) as f: splits = json.load(f)
        print(f"  Loaded splits from: {splits_file}")
        train_ids = splits["train_ids"]; test_ids = splits["test_ids"]
    else:
        rng   = random.Random(seed)
        ids   = sorted(id2paths.keys()); rng.shuffle(ids)
        n_tr  = max(1, int(len(ids) * train_ratio))
        train_ids = ids[:n_tr]; test_ids = ids[n_tr:]
        with open(splits_file, "w") as f:
            json.dump({"train_ids": train_ids, "test_ids": test_ids}, f, indent=2)
        print(f"  Generated splits → {splits_file}")

    print(f"  Train IDs: {len(train_ids)}  Test IDs: {len(test_ids)}")

    rng             = random.Random(seed)
    train_label_map = {k: i for i, k in enumerate(train_ids)}
    test_label_map  = {k: i for i, k in enumerate(test_ids)}
    train_samples   = [(p, train_label_map[ident])
                       for ident in train_ids if ident in id2paths
                       for p in id2paths[ident]]
    gallery, probe  = [], []
    for ident in test_ids:
        if ident not in id2paths: continue
        paths = list(id2paths[ident]); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        n_gal = min(n_gal, len(paths)-1) if len(paths) > 1 else n_gal
        for p in paths[:n_gal]: gallery.append((p, test_label_map[ident]))
        for p in paths[n_gal:]: probe.append((p, test_label_map[ident]))

    return train_samples, gallery, probe, len(train_ids)


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval(); feats, labels = [], []
    for imgs, lbl in loader:
        emb = model(imgs.to(device))
        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        feats.append(emb.cpu().numpy()); labels.append(lbl.numpy())
    feats = np.concatenate(feats)
    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, np.concatenate(labels)


def compute_eer(scores_array):
    ins  = scores_array[scores_array[:,1]==1, 0]
    outs = scores_array[scores_array[:,1]==-1, 0]
    if len(ins)==0 or len(outs)==0: return 1.0, 0.0
    y = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s = np.concatenate([ins, outs])
    if not np.isfinite(s).all() or np.unique(s).size < 2: return 1.0, 0.0
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    eer    = brentq(lambda x: 1.0-x-interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))
    return eer, thresh


def evaluate(model, gallery_loader, probe_loader, device, out_dir, tag):
    gal_feats, gal_labels = extract_embeddings(model, gallery_loader, device)
    prb_feats, prb_labels = extract_embeddings(model, probe_loader,   device)
    sim   = prb_feats @ gal_feats.T
    rank1 = 100.0 * (gal_labels[sim.argmax(axis=1)] == prb_labels).mean()
    scores_list, labels_list = [], []
    for i in range(len(prb_labels)):
        for j in range(len(gal_labels)):
            scores_list.append(float(sim[i,j]))
            labels_list.append(1 if prb_labels[i]==gal_labels[j] else -1)
    scores_arr = np.column_stack([scores_list, labels_list])
    eer, _     = compute_eer(scores_arr)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list): f.write(f"{s} {l}\n")
    print(f"  [{tag}]  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return eer, rank1


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    cfg  = CONFIG
    seed = cfg["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = cfg["save_dir"]
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ArcFace iResNet100 — Palm-Auth Open-Set")
    print(f"  Device : {device}  Epochs : {cfg['num_epochs']}")
    print(f"  Train/Test split : {cfg['train_id_ratio']*100:.0f}/{(1-cfg['train_id_ratio'])*100:.0f}")
    print(f"{'='*60}\n")

    print("  Collecting Palm-Auth data …")
    id2paths = collect_palm_auth(cfg["palm_auth_data_root"], cfg["scanner_spectra"])

    train_samples, gallery_samples, probe_samples, num_classes = load_or_generate_splits(
        id2paths, cfg["train_id_ratio"], cfg["test_gallery_ratio"],
        seed, cfg["splits_file"])

    print(f"  Train images: {len(train_samples)}  Gallery: {len(gallery_samples)}  Probe: {len(probe_samples)}")

    train_loader   = make_loader(train_samples,   train=True,  cfg=cfg)
    gallery_loader = make_loader(gallery_samples, train=False, cfg=cfg)
    probe_loader   = make_loader(probe_samples,   train=False, cfg=cfg)

    model     = ArcFaceBackbone(cfg["pretrained_weights"]).to(device)
    criterion = ArcFaceLoss(num_classes, embedding_size=512,
                            s=cfg["arcface_s"], m=cfg["arcface_m"]).to(device)

    trainable_params = ([p for p in model.parameters()     if p.requires_grad] +
                        [p for p in criterion.parameters() if p.requires_grad])
    optimizer = optim.AdamW(trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["num_epochs"], eta_min=1e-6)

    best_rank1 = 0.0
    ckpt_path  = os.path.join(save_dir, "best_model.pth")

    evaluate(model, gallery_loader, probe_loader, device, eval_dir, "pretrain")

    for epoch in range(1, cfg["num_epochs"]+1):
        model.train(); criterion.train()
        ep_loss=0.0; ep_corr=0; ep_tot=0

        for imgs, labels in train_loader:
            imgs=imgs.to(device); labels=labels.to(device)
            optimizer.zero_grad()
            embeddings = model(imgs)
            loss       = criterion(embeddings, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
            optimizer.step()
            ep_loss += loss.item()
            with torch.no_grad():
                preds    = criterion.get_logits(embeddings).argmax(dim=1)
                ep_corr += (preds==labels).sum().item()
                ep_tot  += labels.size(0)

        scheduler.step()
        n=len(train_loader); acc=100.*ep_corr/max(ep_tot,1)
        ts=time.strftime("%H:%M:%S")
        print(f"  [{ts}] ep {epoch:03d}/{cfg['num_epochs']}  "
              f"loss={ep_loss/n:.4f}  acc={acc:.2f}%")

        if epoch%cfg["eval_every"]==0 or epoch==cfg["num_epochs"]:
            cur_eer, cur_rank1 = evaluate(
                model, gallery_loader, probe_loader, device, eval_dir, f"ep{epoch:04d}")
            if cur_rank1 > best_rank1:
                best_rank1 = cur_rank1
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "arc": criterion.state_dict(),
                            "rank1": cur_rank1, "eer": cur_eer}, ckpt_path)
                print(f"  *** New best Rank-1: {best_rank1:.2f}% ***")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final_eer, final_rank1 = evaluate(
        model, gallery_loader, probe_loader, device, eval_dir, "FINAL")

    result = {"EER_pct": final_eer*100, "Rank1_pct": final_rank1,
              "num_train_classes": num_classes}
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  FINAL  EER={final_eer*100:.4f}%  Rank-1={final_rank1:.2f}%")


if __name__ == "__main__":
    main()
