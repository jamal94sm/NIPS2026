"""
benchmarking.py
================
Runs every method in config.METHOD_ORDER on every one of the 12
train/test settings (config.py / dataset.py), using the SAME cached,
seeded ID split for every method on a given setting, and reports:

  - EER (%) and Rank-1 (%) at the checkpoint with the best Rank-1 seen
    during training, for each (setting, method) pair
  - two final tables: rows = 12 settings, columns = 10 methods

Usage:
    python benchmarking.py                       # run everything
    python benchmarking.py --methods arcface,ccnet
    python benchmarking.py --settings S_scanner,S_wet_text
    python benchmarking.py --quick                # smoke-test: 1 epoch, eval every epoch
"""
import os
import json
import time
import argparse
import traceback

import torch

import config as C
import dataset as D
import model as M
import utils as U


def get_or_create_init_state(baseline, method_name, num_classes):
    """Every setting for a given method starts from the SAME initial
    weights (cached & reused across the 12 settings), so that
    setting-to-setting differences aren't confounded by random init.
    This mirrors `get_or_create_init_weights` in the original scripts."""
    cache_dir = os.path.join(C.BASE_RESULTS_DIR, "init_weights")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{method_name}_nc{num_classes}.pth")
    if os.path.exists(path):
        baseline.load_state_dict(torch.load(path, map_location=C.DEVICE, weights_only=False))
    else:
        torch.save(baseline.state_dict(), path)


def run_one(method_name, setting, train_samples, gallery_samples, probe_samples,
            num_classes, quick=False):
    cfg = C.METHODS[method_name]
    num_epochs = 1 if quick else cfg["num_epochs"]
    eval_every = 1 if quick else C.EVAL_EVERY

    results_dir = os.path.join(C.BASE_RESULTS_DIR, setting["tag"], method_name)
    eval_dir = os.path.join(results_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    train_loader = D.make_loader(train_samples, method_name, True, cfg["batch_size"], C.NUM_WORKERS)
    gallery_loader = D.make_loader(gallery_samples, method_name, False, cfg["batch_size"], C.NUM_WORKERS)
    probe_loader = D.make_loader(probe_samples, method_name, False, cfg["batch_size"], C.NUM_WORKERS)

    baseline = M.REGISTRY[method_name](num_classes, cfg, C.DEVICE)
    get_or_create_init_state(baseline, method_name, num_classes)
    optimizer, scheduler = baseline.build_optimizer()

    best_rank1, best_eer = -1.0, None
    ckpt_path = os.path.join(results_dir, "best_model.pth")

    batch_times, epoch_times = [], []
    for epoch in range(1, num_epochs + 1):
        if hasattr(baseline, "epoch"):
            baseline.epoch = epoch          # drives PalmBridge's warmup schedule
        baseline.train_mode()
        ep_loss, n_batches = 0.0, 0
        epoch_t0 = time.time()
        for batch in train_loader:
            if C.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            batch_t0 = time.time()
            loss = baseline.train_step(batch, optimizer)
            if C.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            batch_times.append(time.time() - batch_t0)
            ep_loss += loss
            n_batches += 1
        epoch_times.append(time.time() - epoch_t0)
        scheduler.step()

        if epoch % eval_every == 0 or epoch == num_epochs:
            baseline.eval_mode()
            eer, rank1 = U.evaluate(baseline.embed, gallery_loader, probe_loader, C.DEVICE,
                                     eval_dir, tag=f"ep{epoch:04d}")
            if rank1 > best_rank1:
                best_rank1, best_eer = rank1, eer
                torch.save(baseline.state_dict(), ckpt_path)
            avg_loss = ep_loss / max(n_batches, 1)
            print(f"    ep {epoch:04d}/{num_epochs}  loss={avg_loss:.4f}  "
                  f"EER={eer:.4f}%  Rank-1={rank1:.2f}%  (best R1={best_rank1:.2f}%)")

    if os.path.exists(ckpt_path):
        baseline.load_state_dict(torch.load(ckpt_path, map_location=C.DEVICE, weights_only=False))
    baseline.eval_mode()

    # Dedicated inference-timing pass (kept separate from the EER/Rank-1 eval
    # calls above, which run periodically during training and would skew the
    # average if included).
    infer_samples = gallery_samples + probe_samples
    infer_loader = D.make_loader(infer_samples, method_name, False, cfg["batch_size"], C.NUM_WORKERS)
    if C.DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for imgs, _ in infer_loader:
            baseline.embed(imgs.to(C.DEVICE))
    if C.DEVICE.type == "cuda":
        torch.cuda.synchronize()
    infer_total_s = time.time() - t0
    infer_time_per_100 = (infer_total_s / max(len(infer_samples), 1)) * 100 * 1000  # ms / 100 samples

    final_eer, final_rank1 = U.evaluate(baseline.embed, gallery_loader, probe_loader, C.DEVICE,
                                         eval_dir, tag="FINAL")

    timing = {
        "train_time_per_epoch_s": sum(epoch_times) / max(len(epoch_times), 1),
        "train_time_per_batch_ms": (sum(batch_times) / max(len(batch_times), 1)) * 1000,
        "infer_time_per_100_samples_ms": infer_time_per_100,
    }

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump({"setting": setting["label"], "method": method_name,
                    "num_train_classes": num_classes,
                    "EER_pct": final_eer, "Rank1_pct": final_rank1, **timing}, f, indent=2)
    return final_eer, final_rank1, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", type=str, default=None,
                     help="comma-separated subset of config.METHOD_ORDER")
    ap.add_argument("--settings", type=str, default=None,
                     help="comma-separated subset of setting labels (e.g. S_scanner)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 epoch per run")
    args = ap.parse_args()

    U.seed_everything(C.SEED)

    methods = args.methods.split(",") if args.methods else C.METHOD_ORDER
    for m in methods:
        assert m in C.METHODS, f"unknown method {m}"

    print("Scanning dataset and building the 12 settings (shared across all methods)...")
    settings = D.get_settings()
    if args.settings:
        wanted = set(args.settings.split(","))
        settings = [s for s in settings if s["label"] in wanted]
    print(f"Running {len(methods)} methods x {len(settings)} settings "
          f"= {len(methods) * len(settings)} experiments\n")

    results = {}
    timing_results = {m: [] for m in methods}
    for s_idx, setting in enumerate(settings, 1):
        print(f"{'='*70}\n[{s_idx}/{len(settings)}] SETTING {setting['label']}\n"
              f"  Train: {setting['train_desc']}   Test: {setting['test_desc']}\n{'='*70}")
        train_samples, gallery_samples, probe_samples, num_classes = setting["parser"]()
        print(f"  train={len(train_samples)}  gallery={len(gallery_samples)}  "
              f"probe={len(probe_samples)}  classes={num_classes}\n")

        for method_name in methods:
            print(f"  --- {method_name} ---")
            t0 = time.time()
            try:
                eer, rank1, timing = run_one(method_name, setting, train_samples, gallery_samples,
                                              probe_samples, num_classes, quick=args.quick)
                results[(setting["label"], method_name)] = {"eer": eer, "rank1": rank1}
                timing_results[method_name].append(timing)
                print(f"  {method_name}: EER={eer:.4f}%  Rank-1={rank1:.2f}%  "
                      f"train={timing['train_time_per_epoch_s']:.2f}s/epoch  "
                      f"infer={timing['infer_time_per_100_samples_ms']:.1f}ms/100  "
                      f"({(time.time()-t0)/60:.1f} min)\n")
            except Exception as e:
                print(f"  [FAILED] {method_name} on {setting['label']}: {e}")
                traceback.print_exc()
                results[(setting["label"], method_name)] = {"eer": None, "rank1": None}

    print(f"\n{'='*70}\nALL RUNS COMPLETE -- building tables\n{'='*70}")
    setting_labels = [s["label"] for s in settings]
    eer_df, rank1_df = U.build_tables(results, setting_labels, methods, C.BASE_RESULTS_DIR)

    print("\nEER (%) table (rows=settings, cols=methods):")
    print(eer_df.to_string())
    print("\nRank-1 (%) table (rows=settings, cols=methods), best-Rank-1 checkpoint per run:")
    print(rank1_df.to_string())
    print(f"\nSaved: {C.BASE_RESULTS_DIR}/eer_table.csv, rank1_table.csv, summary.md")

    timing_df = U.build_timing_table(timing_results, methods, C.BASE_RESULTS_DIR)
    print("\nTiming table (avg per method across all completed settings):")
    print(timing_df.to_string())
    print(f"Saved: {C.BASE_RESULTS_DIR}/timing_table.csv")


if __name__ == "__main__":
    main()
