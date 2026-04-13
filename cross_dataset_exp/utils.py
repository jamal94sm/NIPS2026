"""
utils.py — Evaluation helpers: feature extraction, EER, Rank-1, ROC/DET plots.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# ──────────────────────────────────────────────────────────────
#  Feature extraction
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, loader, device):
    """Return (feats, labels) as numpy arrays."""
    model.eval()
    feats, labels = [], []
    for imgs, labs in loader:
        feats.append(model.get_embedding(imgs.to(device)).cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


# ──────────────────────────────────────────────────────────────
#  EER
# ──────────────────────────────────────────────────────────────

def compute_eer(scores_array):
    """
    scores_array : (N, 2)  col-0 = score, col-1 = +1 (genuine) / -1 (impostor)
    Returns (eer, threshold).
    """
    inscore  = scores_array[scores_array[:, 1] ==  1, 0]
    outscore = scores_array[scores_array[:, 1] == -1, 0]
    if len(inscore) == 0 or len(outscore) == 0:
        return 1.0, 0.0

    flipped = inscore.mean() < outscore.mean()
    if flipped:
        inscore, outscore = -inscore, -outscore

    y   = np.concatenate([np.ones(len(inscore)), np.zeros(len(outscore))])
    s   = np.concatenate([inscore, outscore])
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    eer    = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))
    return eer, (-thresh if flipped else thresh)


# ──────────────────────────────────────────────────────────────
#  Full evaluation
# ──────────────────────────────────────────────────────────────

def evaluate(model, probe_loader, gallery_loader, device,
             out_dir=".", tag="eval"):
    """
    Compute pairwise EER, aggregated EER, and Rank-1.
    Saves score file and ROC/DET PDF.
    Returns (pairwise_eer, aggregated_eer, rank1_acc).
    """
    probe_feats,   probe_labels   = extract_features(model, probe_loader, device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)

    n_probe   = len(probe_feats)
    n_gallery = len(gallery_feats)

    scores_list, labels_list = [], []
    dist_matrix = np.zeros((n_probe, n_gallery))

    for i in range(n_probe):
        cos_sim        = np.dot(gallery_feats, probe_feats[i])
        dists          = np.arccos(np.clip(cos_sim, -1, 1)) / np.pi
        dist_matrix[i] = dists
        for j in range(n_gallery):
            scores_list.append(dists[j])
            labels_list.append(1 if probe_labels[i] == gallery_labels[j] else -1)

    scores_arr    = np.column_stack([scores_list, labels_list])
    pair_eer, _   = compute_eer(scores_arr)

    # aggregated EER (probe vs probe)
    aggr_s, aggr_l = [], []
    for i in range(n_probe - 1):
        for j in range(i + 1, n_probe):
            d = np.arccos(np.clip(np.dot(probe_feats[i], probe_feats[j]), -1, 1)) / np.pi
            aggr_s.append(d)
            aggr_l.append(1 if probe_labels[i] == probe_labels[j] else -1)
    aggr_eer = compute_eer(np.column_stack([aggr_s, aggr_l]))[0] if aggr_s else 1.0

    # Rank-1
    correct = sum(probe_labels[i] == gallery_labels[np.argmin(dist_matrix[i])]
                  for i in range(n_probe))
    rank1   = 100.0 * correct / max(n_probe, 1)

    # save scores
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list):
            f.write(f"{s} {l}\n")

    _save_roc_det(scores_arr, out_dir, tag)

    print(f"  [{tag}]  pairEER={pair_eer*100:.4f}%  "
          f"aggrEER={aggr_eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return pair_eer, aggr_eer, rank1


def _save_roc_det(scores_arr, out_dir, tag):
    ins  = scores_arr[scores_arr[:, 1] ==  1, 0]
    outs = scores_arr[scores_arr[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0:
        return
    if ins.mean() < outs.mean():
        ins, outs = -ins, -outs

    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s   = np.concatenate([ins, outs])
    fpr, tpr, thr = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr

    try:
        pdf = PdfPages(os.path.join(out_dir, f"roc_det_{tag}.pdf"))

        fig, ax = plt.subplots()
        ax.plot(fpr*100, tpr*100, 'b-^', markersize=2, label="ROC")
        ax.plot(np.linspace(0,100,101), np.linspace(100,0,101), 'k-', label="EER")
        ax.set(xlim=[0,5], ylim=[90,100], xlabel="FAR (%)", ylabel="GAR (%)",
               title=f"ROC — {tag}")
        ax.legend(); ax.grid(True)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(fpr*100, fnr*100, 'b-^', markersize=2, label="DET")
        ax.plot(np.linspace(0,100,101), np.linspace(0,100,101), 'k-', label="EER")
        ax.set(xlim=[0,5], ylim=[0,5], xlabel="FAR (%)", ylabel="FRR (%)",
               title=f"DET — {tag}")
        ax.legend(); ax.grid(True)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(thr, fpr*100, 'r-.', markersize=2, label="FAR")
        ax.plot(thr, fnr*100, 'b-^', markersize=2, label="FRR")
        ax.set(xlabel="Threshold", ylabel="Rate (%)", title=f"FAR/FRR — {tag}")
        ax.legend(); ax.grid(True)
        pdf.savefig(fig); plt.close(fig)

        pdf.close()
    except Exception as e:
        print(f"  [warn] plot failed: {e}")


# ──────────────────────────────────────────────────────────────
#  Training curve plot
# ──────────────────────────────────────────────────────────────

def plot_train_curves(losses, accs, results_dir):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(losses, 'b'); axes[0].set_title("Train Loss")
        axes[0].set_xlabel("epoch"); axes[0].grid(True)
        axes[1].plot(accs,   'b'); axes[1].set_title("Train Acc (%)")
        axes[1].set_xlabel("epoch"); axes[1].grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "train_curves.png"))
        plt.close(fig)
    except Exception:
        pass
