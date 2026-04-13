"""
main.py — Entry point for CO3Net palmprint recognition.

Usage
-----
    python main.py

All settings are in config.py.
"""

import os
import math
import time
import random
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from config  import CONFIG
from models  import CO3Net
from losses  import SupConLoss
from dataset import (get_parser, split_same_dataset,
                     split_cross_dataset_test,
                     PairedDataset, SingleDataset)
from train   import run_one_epoch
from utils   import evaluate, plot_train_curves


# ──────────────────────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────────────────────
SEED = CONFIG["random_seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    # ── unpack config ─────────────────────────────────────────
    train_data          = CONFIG["train_data"]
    test_data           = CONFIG["test_data"]
    test_gallery_ratio  = CONFIG["test_gallery_ratio"]
    train_subject_ratio = CONFIG["train_subject_ratio"]
    results_dir         = CONFIG["results_dir"]
    img_side            = CONFIG["img_side"]
    batch_size          = CONFIG["batch_size"]
    num_epochs          = CONFIG["num_epochs"]
    lr                  = CONFIG["lr"]
    lr_step             = CONFIG["lr_step"]
    lr_gamma            = CONFIG["lr_gamma"]
    dropout             = CONFIG["dropout"]
    arcface_s           = CONFIG["arcface_s"]
    arcface_m           = CONFIG["arcface_m"]
    ce_weight           = CONFIG["ce_weight"]
    con_weight          = CONFIG["con_weight"]
    temperature         = CONFIG["temperature"]
    seed                = CONFIG["random_seed"]
    save_every          = CONFIG["save_every"]
    eval_every          = CONFIG["eval_every"]
    nw                  = CONFIG["num_workers"]
    augment_factor      = CONFIG["augment_factor"]
    use_triplet         = CONFIG.get("use_triplet",     False)
    triplet_weight      = CONFIG.get("triplet_weight",  0.10)
    triplet_margin      = CONFIG.get("triplet_margin",  0.25)

    same_dataset = (train_data.strip().lower().replace("-", "") ==
                    test_data.strip().lower().replace("-", ""))

    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  CO3Net Palmprint Recognition")
    print(f"  Device         : {device}")
    print(f"  Train dataset  : {train_data}")
    print(f"  Test dataset   : {test_data}")
    if same_dataset:
        print(f"  Mode           : same-dataset split "
              f"({int(train_subject_ratio*100)}% train / "
              f"{int((1-train_subject_ratio)*100)}% test)")
    print(f"  Loss           : {ce_weight}×CE + {con_weight}×SupCon"
          f"(τ={temperature})")
    print(f"  Augment factor : {augment_factor}×")
    print(f"{'='*60}\n")

    # ── parsers ───────────────────────────────────────────────
    train_parser = get_parser(train_data, CONFIG)
    test_parser  = get_parser(test_data,  CONFIG)

    # ══════════════════════════════════════════════════════════
    #  Dataset setup — same dataset
    # ══════════════════════════════════════════════════════════
    if same_dataset:
        print(f"Scanning {train_data} (shared train+test) …")
        all_id2paths = train_parser()
        n_total_ids  = len(all_id2paths)
        n_total_imgs = sum(len(v) for v in all_id2paths.values())
        print(f"  Found {n_total_ids} identities, {n_total_imgs} images.\n")

        (train_samples, gallery_samples, probe_samples,
         train_label_map, _) = split_same_dataset(
            all_id2paths,
            train_subject_ratio=train_subject_ratio,
            gallery_ratio=test_gallery_ratio,
            seed=seed)

        num_classes  = len(train_label_map)
        n_train_ids  = num_classes
        n_train_imgs = len(train_samples)
        n_test_ids   = n_total_ids - n_train_ids
        n_test_imgs  = len(gallery_samples) + len(probe_samples)

    # ══════════════════════════════════════════════════════════
    #  Dataset setup — cross dataset
    # ══════════════════════════════════════════════════════════
    else:
        print(f"Scanning {train_data} (train) …")
        train_id2paths = train_parser()
        n_train_ids    = len(train_id2paths)
        n_train_imgs   = sum(len(v) for v in train_id2paths.values())
        print(f"  Found {n_train_ids} identities, {n_train_imgs} images.\n")

        train_label_map = {k: i for i, k in enumerate(sorted(train_id2paths))}
        train_samples   = [(p, train_label_map[ident])
                           for ident, paths in train_id2paths.items()
                           for p in paths]
        num_classes = len(train_label_map)

        print(f"Scanning {test_data} (test) …")
        test_id2paths = test_parser()
        n_test_ids    = len(test_id2paths)
        n_test_imgs   = sum(len(v) for v in test_id2paths.values())
        print(f"  Found {n_test_ids} identities, {n_test_imgs} images.\n")

        gallery_samples, probe_samples, _ = split_cross_dataset_test(
            test_id2paths, gallery_ratio=test_gallery_ratio, seed=seed)

    # ── data loaders ──────────────────────────────────────────
    train_loader = DataLoader(
        PairedDataset(train_samples, img_side, train=True,
                      augment_factor=augment_factor),
        batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=True)

    gallery_loader = DataLoader(
        SingleDataset(gallery_samples, img_side),
        batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True)

    probe_loader = DataLoader(
        SingleDataset(probe_samples, img_side),
        batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True)

    print(f"  Train  : {n_train_ids} subjects | "
          f"{n_train_imgs} imgs (+aug → {n_train_imgs*augment_factor})")
    print(f"  Test   : {n_test_ids} subjects | "
          f"Gallery {len(gallery_samples)} | Probe {len(probe_samples)}")
    print(f"  Classes: {num_classes}\n")

    # ── model ─────────────────────────────────────────────────
    print(f"Building CO3Net — num_classes={num_classes} …")
    net = CO3Net(num_classes, dropout=dropout,
                 arcface_s=arcface_s, arcface_m=arcface_m)
    net.to(device)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs")
        net = DataParallel(net)

    criterion     = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=temperature,
                               base_temperature=temperature)
    optimizer     = optim.Adam(net.parameters(), lr=lr)
    scheduler     = lr_scheduler.StepLR(optimizer, lr_step, lr_gamma)

    # ── training loop ─────────────────────────────────────────
    train_losses, train_accs = [], []
    best_eer   = 1.0
    last_eer   = float("nan")
    last_rank1 = float("nan")

    print(f"\nStarting training for {num_epochs} epochs …")
    print(f"  EER / Rank-1 evaluated every {eval_every} epochs.\n")

    if CONFIG.get("eval_only", False):
        print("  eval_only=True — skipping training.\n")
    else:
        for epoch in range(num_epochs):
            t_loss, t_acc = run_one_epoch(
                net, train_loader, criterion, con_criterion,
                optimizer, device, "training",
                ce_weight=ce_weight, con_weight=con_weight,
                use_triplet=use_triplet,
                triplet_weight=triplet_weight,
                triplet_margin=triplet_margin)

            train_losses.append(t_loss)
            train_accs.append(t_acc)

            _net = net.module if isinstance(net, DataParallel) else net

            # periodic evaluation
            if epoch % eval_every == 0 or epoch == num_epochs - 1:
                tag = f"ep{epoch:04d}_{test_data.replace('-','')}"
                cur_eer, _, cur_rank1 = evaluate(
                    _net, probe_loader, gallery_loader,
                    device, out_dir=rst_eval, tag=tag)
                last_eer, last_rank1 = cur_eer, cur_rank1
                if cur_eer < best_eer:
                    best_eer = cur_eer
                    torch.save(_net.state_dict(),
                               os.path.join(results_dir,
                                            "net_params_best_eer.pth"))
                    print(f"  *** New best EER: {best_eer*100:.4f}% ***")

            # console print every 10 epochs
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                ts        = time.strftime("%H:%M:%S")
                eer_str   = (f"{last_eer*100:.4f}%"
                             if not math.isnan(last_eer) else "N/A")
                rank1_str = (f"{last_rank1:.2f}%"
                             if not math.isnan(last_rank1) else "N/A")
                print(f"[{ts}] ep {epoch:04d} | "
                      f"loss={t_loss:.5f} | acc={t_acc:.2f}% | "
                      f"EER={eer_str}  Rank-1={rank1_str}")

            # checkpoint + curves
            if epoch % save_every == 0 or epoch == num_epochs - 1:
                torch.save(_net.state_dict(),
                           os.path.join(results_dir, "net_params.pth"))
                plot_train_curves(train_losses, train_accs, results_dir)

    # ── final evaluation ──────────────────────────────────────
    print(f"\n=== Final evaluation on {test_data} ===")
    best_path = os.path.join(results_dir, "net_params_best_eer.pth")
    if not os.path.exists(best_path):
        best_path = os.path.join(results_dir, "net_params.pth")

    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_path, map_location=device))

    saved_name = (f"CO3Net"
                  f"_train{train_data.replace('-','').replace(' ','')}"
                  f"_test{test_data.replace('-','').replace(' ','')}.pth")
    torch.save(eval_net.state_dict(),
               os.path.join(results_dir, saved_name))
    print(f"  Model saved as {saved_name}")

    final_eer, final_aggr_eer, final_rank1 = evaluate(
        eval_net, probe_loader, gallery_loader,
        device, out_dir=rst_eval,
        tag=f"FINAL_{test_data.replace('-','')}")

    print(f"\n{'='*60}")
    print(f"  Train  : {train_data} ({n_train_ids} subjects, {n_train_imgs} imgs)")
    print(f"  Test   : {test_data}  ({n_test_ids} subjects, {n_test_imgs} imgs)")
    print(f"  FINAL Pairwise EER   : {final_eer*100:.4f}%")
    print(f"  FINAL Aggregated EER : {final_aggr_eer*100:.4f}%")
    print(f"  FINAL Rank-1         : {final_rank1:.3f}%")
    print(f"  Results saved to     : {results_dir}")
    print(f"{'='*60}\n")

    # summary file
    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"Train dataset      : {train_data}\n")
        f.write(f"Train subjects     : {n_train_ids}\n")
        f.write(f"Train images       : {n_train_imgs}\n")
        f.write(f"Augment factor     : {augment_factor}×\n")
        f.write(f"Num classes        : {num_classes}\n")
        f.write(f"Test dataset       : {test_data}\n")
        f.write(f"Test subjects      : {n_test_ids}\n")
        f.write(f"Test images        : {n_test_imgs}\n")
        f.write(f"Gallery samples    : {len(gallery_samples)}\n")
        f.write(f"Probe samples      : {len(probe_samples)}\n")
        f.write(f"Final Pairwise EER : {final_eer*100:.6f}%\n")
        f.write(f"Final Aggreg. EER  : {final_aggr_eer*100:.6f}%\n")
        f.write(f"Final Rank-1       : {final_rank1:.3f}%\n")


if __name__ == "__main__":
    main()
