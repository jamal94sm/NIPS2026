"""
utils.py
========
Shared, method-agnostic utilities: seeding, embedding extraction,
EER/Rank-1 evaluation (one canonical evaluator used by all 10 baselines,
replacing each method's slightly-different bespoke eval code), checkpoint
helpers, and the final results-table builder.
"""
import os
import random
import numpy as np
import torch

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════
#  EMBEDDING EXTRACTION / EER / RANK-1  (one canonical implementation)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(embed_fn, loader, device):
    """embed_fn(batch_imgs) -> L2-normalised (or raw, for MagFace) embeddings.
    embed_fn is provided per-method by model.py (e.g. wraps PalmBridge's
    z_hat, or MagFace's raw backbone output)."""
    feats, labels = [], []
    for imgs, lbl in loader:
        emb = embed_fn(imgs.to(device))
        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        feats.append(emb.cpu().numpy())
        labels.append(lbl.numpy())
    feats = np.concatenate(feats) if feats else np.zeros((0, 1))
    if feats.size and not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, np.concatenate(labels) if labels else np.zeros((0,))


def compute_eer(scores_array):
    ins = scores_array[scores_array[:, 1] == 1, 0]
    outs = scores_array[scores_array[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0:
        return 1.0, 0.0
    y = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s = np.concatenate([ins, outs])
    if not np.isfinite(s).all() or np.unique(s).size < 2:
        return 1.0, 0.0
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer, float(interp1d(fpr, thresholds)(eer))


def evaluate(embed_fn, gallery_loader, probe_loader, device, out_dir=None, tag="eval"):
    """Cosine-similarity, all-pairs EER + argmax Rank-1. Identical logic
    for every method -- normalize embeddings to unit length beforehand if
    the method's embed_fn doesn't already (see model.py)."""
    gal_feats, gal_labels = extract_embeddings(embed_fn, gallery_loader, device)
    prb_feats, prb_labels = extract_embeddings(embed_fn, probe_loader, device)

    gal_n = gal_feats / (np.linalg.norm(gal_feats, axis=1, keepdims=True) + 1e-8)
    prb_n = prb_feats / (np.linalg.norm(prb_feats, axis=1, keepdims=True) + 1e-8)
    sim = prb_n @ gal_n.T

    rank1 = 100.0 * (gal_labels[sim.argmax(axis=1)] == prb_labels).mean()

    scores_list, labels_list = [], []
    for i in range(len(prb_labels)):
        row = sim[i]
        same = (gal_labels == prb_labels[i])
        scores_list.extend(row.tolist())
        labels_list.extend(np.where(same, 1, -1).tolist())
    scores_arr = np.column_stack([scores_list, labels_list])
    eer, _ = compute_eer(scores_arr)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
            for s, l in zip(scores_list, labels_list):
                f.write(f"{s} {l}\n")
    print(f"    [{tag}]  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return eer * 100.0, rank1


# ══════════════════════════════════════════════════════════════
#  CHECKPOINTING
# ══════════════════════════════════════════════════════════════

def save_best(state_dict_fn, path, epoch, eer, rank1):
    torch.save({"epoch": epoch, "eer": eer, "rank1": rank1, **state_dict_fn()}, path)


def load_ckpt(path, device):
    return torch.load(path, map_location=device, weights_only=False)


# ══════════════════════════════════════════════════════════════
#  RESULTS TABLE  (rows = 12 settings, columns = 10 methods)
# ══════════════════════════════════════════════════════════════

def build_tables(results, settings_labels, method_order, out_dir):
    """`results[(setting_label, method)] = {"eer": float, "rank1": float}`
    (eer/rank1 already correspond to the checkpoint with the BEST Rank-1
    seen during training, per benchmarking.py). Writes eer_table.csv,
    rank1_table.csv and a combined markdown summary; returns both
    pandas DataFrames."""
    import pandas as pd

    eer_rows, r1_rows = [], []
    for s in settings_labels:
        eer_row = {"setting": s}
        r1_row = {"setting": s}
        for m in method_order:
            r = results.get((s, m))
            eer_row[m] = None if r is None else round(r["eer"], 4)
            r1_row[m] = None if r is None else round(r["rank1"], 2)
        eer_rows.append(eer_row)
        r1_rows.append(r1_row)

    eer_df = pd.DataFrame(eer_rows).set_index("setting")[method_order]
    r1_df = pd.DataFrame(r1_rows).set_index("setting")[method_order]

    os.makedirs(out_dir, exist_ok=True)
    eer_df.to_csv(os.path.join(out_dir, "eer_table.csv"))
    r1_df.to_csv(os.path.join(out_dir, "rank1_table.csv"))

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("# EER (%) -- rows = settings, columns = methods\n\n")
        f.write(eer_df.to_markdown())
        f.write("\n\n# Rank-1 (%) -- best-Rank-1 checkpoint per run\n\n")
        f.write(r1_df.to_markdown())
        f.write("\n")

    return eer_df, r1_df
