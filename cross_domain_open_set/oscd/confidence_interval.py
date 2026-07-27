"""
confidence_interval.py
=======================
Reviewer question: "How stable are the open-set results under more identity
splits or bootstrap confidence intervals?"

Two complementary experiments, CompNet only, X-Palm, open-set cross-domain,
all 12 train/test settings from dataset.py:

  PHASE 1 -- bootstrap CI
    Train a fresh CompNet at config.TRAIN_ID_RATIO (the same ratio used
    everywhere else in the benchmark) for each of the 12 settings, then
    bootstrap-resample the *test identities* (not raw score pairs, which
    are correlated) B times to get a 95% CI on EER and Rank-1 from the
    fixed similarity matrix -- no retraining inside the bootstrap loop.

  PHASE 2 -- repeated identity splits (K=3 fold, adjustable)
    CompNet only, all 12 settings. Every identity is tested in exactly one
    fold, and -- critically -- every fold uses DIFFERENT TEST IDENTITIES
    for every one of the 12 settings, including S_scanner_to_persp.

    S_scanner_to_persp needs special handling: the original protocol
    (dataset.parse_setting_scanner_to_perspective) hardcodes
    train_ids == scanner_ids and ignores whatever test_ids it's given, so
    it cannot be folded by just feeding it a different split. Instead,
    this file partitions the FULL identity population into k folds; each
    fold's identities become that fold's test set (evaluated on their
    perspective images), and training uses the scanner images of every
    remaining identity that has scanner data. This preserves the setting's
    "train on scanner domain, test cross-domain generalization on
    perspective domain" character while giving genuinely different held-
    out test identities each fold, exactly like the other 11 settings.

Usage:
    python confidence_interval.py                  # full run, both phases
    python confidence_interval.py --quick            # smoke test: 1 epoch/run
    python confidence_interval.py --n-bootstrap 1000
    python confidence_interval.py --k 3
    python confidence_interval.py --phase 1          # phase 1 only
    python confidence_interval.py --phase 2          # phase 2 only
"""
import os
import copy
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

import config as C
import dataset as D
from dataset import _all_samples, _gallery_probe_split
import model as M
import utils as U

METHOD = "compnet"


# ══════════════════════════════════════════════════════════════
#  SHARED: train CompNet -- thin wrappers over model.py's generic,
#  method-agnostic training loop (so this stays importable from ONE
#  shared place -- model.py/utils.py -- rather than other scripts having
#  to reach into this experiment script; see imbalance_experiment.py)
# ══════════════════════════════════════════════════════════════

def train_compnet_model(train_samples, gallery_samples, probe_samples, num_classes,
                         init_tag, num_epochs=None, eval_every=None):
    """CompNet-specific wrapper over model.train_baseline_model()."""
    return M.train_baseline_model("compnet", train_samples, gallery_samples, probe_samples,
                                   num_classes, init_tag, num_epochs=num_epochs,
                                   eval_every=eval_every)


def train_compnet(train_samples, gallery_samples, probe_samples, num_classes,
                   init_tag, num_epochs=None, eval_every=None):
    """CompNet-specific wrapper over model.train_baseline(). Returns
    (gal_feats, gal_labels, prb_feats, prb_labels) from the best-Rank-1
    checkpoint."""
    return M.train_baseline("compnet", train_samples, gallery_samples, probe_samples,
                             num_classes, init_tag, num_epochs=num_epochs,
                             eval_every=eval_every)


# point_eer_rank1 now lives in utils.py (shared, method-agnostic) --
# re-exported here so existing `from confidence_interval import
# point_eer_rank1` call sites elsewhere keep working unchanged.
point_eer_rank1 = U.point_eer_rank1


def _eval_setting_or_skip(s, gallery_samples, probe_samples, num_classes):
    if len(gallery_samples) == 0 or len(probe_samples) == 0 or num_classes == 0:
        print(f"    {s['label']}: SKIPPED (empty gallery/probe/train for this setting)")
        return False
    return True


# ══════════════════════════════════════════════════════════════
#  PHASE 1 -- bootstrap CI (identity-level resampling)
# ══════════════════════════════════════════════════════════════

def identity_bootstrap_ci(sim, gal_labels, prb_labels, n_bootstrap=1000, ci=0.95, seed=0):
    """Resamples TEST IDENTITIES (with replacement), not raw score pairs --
    probe/gallery entries from the same identity are correlated, so
    pair-level bootstrap would understate variance."""
    rng = np.random.default_rng(seed)
    identities = np.array(sorted(set(gal_labels.tolist()) | set(prb_labels.tolist())))
    gal_by_id = {i: np.where(gal_labels == i)[0] for i in identities}
    prb_by_id = {i: np.where(prb_labels == i)[0] for i in identities}

    eers, rank1s = [], []
    for _ in range(n_bootstrap):
        sampled = rng.choice(identities, size=len(identities), replace=True)
        gal_idx = np.concatenate([gal_by_id[i] for i in sampled if len(gal_by_id[i])])
        prb_idx = np.concatenate([prb_by_id[i] for i in sampled if len(prb_by_id[i])])
        if len(gal_idx) == 0 or len(prb_idx) == 0:
            continue
        sub_sim = sim[np.ix_(prb_idx, gal_idx)]
        sub_gal_labels = gal_labels[gal_idx]
        sub_prb_labels = prb_labels[prb_idx]

        rank1 = 100.0 * (sub_gal_labels[sub_sim.argmax(axis=1)] == sub_prb_labels).mean()
        same = (sub_prb_labels[:, None] == sub_gal_labels[None, :])
        scores = sub_sim.ravel()
        labels = np.where(same, 1, -1).ravel()
        eer, _ = U.compute_eer(np.column_stack([scores, labels]))

        eers.append(eer * 100.0)
        rank1s.append(rank1)

    alpha = (1 - ci) / 2
    eer_lo, eer_hi = np.percentile(eers, [alpha * 100, (1 - alpha) * 100])
    r1_lo, r1_hi = np.percentile(rank1s, [alpha * 100, (1 - alpha) * 100])
    return {"eer_ci_lo": eer_lo, "eer_ci_hi": eer_hi,
            "rank1_ci_lo": r1_lo, "rank1_ci_hi": r1_hi,
            "n_bootstrap_used": len(eers)}


def run_phase1(n_bootstrap, ci, quick=False):
    print(f"\n{'='*70}\nPHASE 1 -- bootstrap CI, CompNet, ratio={C.TRAIN_ID_RATIO}\n{'='*70}")
    settings = D.get_settings()    # uses cached SPLITS_FILE + config.TRAIN_ID_RATIO
    rows = []
    for s in settings:
        train_samples, gallery_samples, probe_samples, num_classes = s["parser"]()
        print(f"  {s['label']}: train={len(train_samples)} gallery={len(gallery_samples)} "
              f"probe={len(probe_samples)}")
        if not _eval_setting_or_skip(s, gallery_samples, probe_samples, num_classes):
            rows.append({"setting": s["label"], "n_test_ids": 0,
                         "EER_pct": None, "EER_CI_lo": None, "EER_CI_hi": None,
                         "Rank1_pct": None, "Rank1_CI_lo": None, "Rank1_CI_hi": None})
            continue
        n_test_ids = len(set(l for _, l in gallery_samples))
        t0 = time.time()
        gal_feats, gal_labels, prb_feats, prb_labels = train_compnet(
            train_samples, gallery_samples, probe_samples, num_classes,
            init_tag="main_protocol", num_epochs=1 if quick else None,
            eval_every=1 if quick else None)
        eer, rank1, sim = point_eer_rank1(gal_feats, gal_labels, prb_feats, prb_labels)
        ci_res = identity_bootstrap_ci(sim, gal_labels, prb_labels,
                                        n_bootstrap=n_bootstrap, ci=ci, seed=C.SEED)
        print(f"    EER={eer:.3f}% [{ci_res['eer_ci_lo']:.3f}, {ci_res['eer_ci_hi']:.3f}]  "
              f"Rank1={rank1:.2f}% [{ci_res['rank1_ci_lo']:.2f}, {ci_res['rank1_ci_hi']:.2f}]  "
              f"({(time.time()-t0)/60:.1f} min)")
        rows.append({
            "setting": s["label"], "n_test_ids": n_test_ids,
            "EER_pct": round(eer, 3), "EER_CI_lo": round(ci_res["eer_ci_lo"], 3),
            "EER_CI_hi": round(ci_res["eer_ci_hi"], 3),
            "Rank1_pct": round(rank1, 2), "Rank1_CI_lo": round(ci_res["rank1_ci_lo"], 2),
            "Rank1_CI_hi": round(ci_res["rank1_ci_hi"], 2),
        })

    df = pd.DataFrame(rows)
    out_dir = C.BASE_RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "phase1_bootstrap_ci.csv"), index=False)

    print(f"\n{'-'*70}\nPHASE 1 SUMMARY -- CompNet, ratio={C.TRAIN_ID_RATIO}, "
          f"bootstrap 95% CI (identity-level resampling, B={n_bootstrap})\n{'-'*70}")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_dir}/phase1_bootstrap_ci.csv")
    return df


# ══════════════════════════════════════════════════════════════
#  PHASE 2 -- K-fold repeated identity splits
# ══════════════════════════════════════════════════════════════

def kfold_partition(pool, k, seed):
    pool = sorted(pool)
    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    return [shuffled[i::k] for i in range(k)]     # near-equal-sized folds


def parse_scanner_to_persp_fold(cond_paths, scanner_paths, test_ids, gallery_ratio, seed):
    """Fold-aware replacement for dataset.parse_setting_scanner_to_perspective.

    The original parser hardcodes train_ids == scanner_ids and ignores
    whatever test_ids it's given (it just uses the full scanner_paths for
    training), so it cannot produce different test identities per fold.
    This version: test_ids (this fold's held-out identities, from the FULL
    identity population, not just the non-scanner subset) are evaluated on
    their perspective images; training uses the scanner images of every
    OTHER identity that has scanner data. Preserves the "train on scanner
    domain, test cross-domain generalization on perspective domain"
    character while giving genuinely different held-out test identities
    each fold.
    """
    rng = random.Random(seed)
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)

    scanner_ids = sorted(scanner_paths.keys())
    test_id_set = set(test_ids)
    train_ids = sorted(set(scanner_ids) - test_id_set)

    train_label_map = {ident: i for i, ident in enumerate(train_ids)}
    test_label_map = {ident: i for i, ident in enumerate(sorted(test_id_set))}

    train_samples = _all_samples({i: scanner_paths[i] for i in train_ids}, train_label_map)
    gallery, probe = _gallery_probe_split(
        {i: persp_all[i] for i in test_id_set if i in persp_all}, test_label_map, gallery_ratio, rng)
    return train_samples, gallery, probe, len(train_ids)


def generate_kfold_splits(cond_paths, scanner_paths, k, seed):
    """Partitions each setting's identity pool into k disjoint folds (each
    identity tested exactly once). For S_scanner and the 10 paired-condition
    settings, this matches dataset.generate_all_splits' pools exactly (just
    folded instead of randomly sampled at one ratio). For S_scanner_to_persp,
    the pool is the FULL identity population (see parse_scanner_to_persp_fold)
    so that setting also gets genuinely different test identities per fold."""
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    all_persp_ids = sorted(persp_all.keys())
    scanner_ids = sorted(scanner_paths.keys())

    folds_by_key = {
        "S_scanner": kfold_partition(scanner_ids, k, seed),
        "S_scanner_to_persp": kfold_partition(all_persp_ids, k, seed),
    }
    for cond_a, cond_b in D.PAIRED_CONDITIONS:
        paths_a = cond_paths.get(cond_a, {})
        paths_b = cond_paths.get(cond_b, {})
        eligible_ids = sorted(set(paths_a.keys()) & set(paths_b.keys()))
        if eligible_ids:
            folds_by_key[f"S_{cond_a}_{cond_b}"] = kfold_partition(eligible_ids, k, seed)

    kfold_splits = []
    for fold_idx in range(k):
        splits = {}
        test_ids = sorted(folds_by_key["S_scanner"][fold_idx])
        splits["S_scanner"] = {"train_ids": sorted(set(all_persp_ids) - set(test_ids)),
                                "test_ids": test_ids}

        sp_test_ids = sorted(folds_by_key["S_scanner_to_persp"][fold_idx])
        sp_train_ids = sorted(set(scanner_ids) - set(sp_test_ids))
        splits["S_scanner_to_persp"] = {"train_ids": sp_train_ids, "test_ids": sp_test_ids}

        for cond_a, cond_b in D.PAIRED_CONDITIONS:
            key = f"S_{cond_a}_{cond_b}"
            if key not in folds_by_key:
                continue
            test_ids = sorted(folds_by_key[key][fold_idx])
            splits[key] = {"train_ids": sorted(set(all_persp_ids) - set(test_ids)),
                            "test_ids": test_ids}
        kfold_splits.append(splits)
    return kfold_splits


def build_settings_for_fold(cond_paths, scanner_paths, splits, gallery_ratio, seed):
    """dataset.build_settings(), with S_scanner_to_persp's parser swapped for
    the fold-aware version (dataset.py's built-in parser ignores test_ids
    and would silently reuse the full scanner_ids -> no fold variation)."""
    settings = D.build_settings(cond_paths, scanner_paths, splits, gallery_ratio, seed)
    sp_test_ids = splits["S_scanner_to_persp"]["test_ids"]
    for s in settings:
        if s["label"] == "S_scanner_to_persp":
            s["parser"] = (lambda tids=sp_test_ids, sd=seed: parse_scanner_to_persp_fold(
                cond_paths, scanner_paths, tids, gallery_ratio, sd))
    return settings


def run_phase2(k, quick=False):
    print(f"\n{'='*70}\nPHASE 2 -- {k}-fold identity splits, CompNet, all 12 settings\n"
          f"(every fold uses different TEST IDENTITIES for all 12 settings, "
          f"including S_scanner_to_persp)\n{'='*70}")
    cond_paths = D.collect_perspective(C.DATA_ROOT)
    scanner_paths = D.collect_scanner(C.DATA_ROOT, C.SCANNER_SPECTRA)
    kfold_splits = generate_kfold_splits(cond_paths, scanner_paths, k, C.SEED)

    per_setting = defaultdict(lambda: {"eer": [], "rank1": []})
    setting_labels = None
    for fold_idx, splits in enumerate(kfold_splits):
        fold_seed = C.SEED + fold_idx      # also varies gallery/probe assignment per fold
        settings = build_settings_for_fold(cond_paths, scanner_paths, splits,
                                            C.TEST_GALLERY_RATIO, fold_seed)
        if setting_labels is None:
            setting_labels = [s["label"] for s in settings]
        for s in settings:
            train_samples, gallery_samples, probe_samples, num_classes = s["parser"]()
            if not _eval_setting_or_skip(s, gallery_samples, probe_samples, num_classes):
                per_setting[s["label"]]["eer"].append(np.nan)
                per_setting[s["label"]]["rank1"].append(np.nan)
                continue
            print(f"  [fold {fold_idx+1}/{k}] {s['label']}: train={len(train_samples)} "
                  f"gallery={len(gallery_samples)} probe={len(probe_samples)}")
            t0 = time.time()
            gal_feats, gal_labels, prb_feats, prb_labels = train_compnet(
                train_samples, gallery_samples, probe_samples, num_classes,
                init_tag=f"kfold{k}",
                num_epochs=1 if quick else None, eval_every=1 if quick else None)
            eer, rank1, _ = point_eer_rank1(gal_feats, gal_labels, prb_feats, prb_labels)
            print(f"    EER={eer:.3f}%  Rank1={rank1:.2f}%  ({(time.time()-t0)/60:.1f} min)")
            per_setting[s["label"]]["eer"].append(eer)
            per_setting[s["label"]]["rank1"].append(rank1)

    rows = []
    for label in setting_labels:
        eers = per_setting[label]["eer"]
        r1s = per_setting[label]["rank1"]
        row = {"setting": label}
        for i, (e, r) in enumerate(zip(eers, r1s), 1):
            row[f"fold{i}_EER"] = round(e, 3)
            row[f"fold{i}_Rank1"] = round(r, 2)
        row["EER_mean"] = round(float(np.nanmean(eers)), 3)
        row["EER_std"] = round(float(np.nanstd(eers)), 3)
        row["EER_min"] = round(float(np.nanmin(eers)), 3)
        row["EER_max"] = round(float(np.nanmax(eers)), 3)
        row["Rank1_mean"] = round(float(np.nanmean(r1s)), 2)
        row["Rank1_std"] = round(float(np.nanstd(r1s)), 2)
        row["Rank1_min"] = round(float(np.nanmin(r1s)), 2)
        row["Rank1_max"] = round(float(np.nanmax(r1s)), 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = C.BASE_RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "phase2_kfold_splits.csv"), index=False)

    print(f"\n{'-'*70}\nPHASE 2 SUMMARY -- CompNet, {k}-fold identity-disjoint splits "
          f"(different test IDs every fold, all 12 settings)\n{'-'*70}")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_dir}/phase2_kfold_splits.csv")
    return df


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=None, choices=[1, 2],
                     help="run only phase 1 or phase 2 (default: both)")
    ap.add_argument("--n-bootstrap", type=int, default=1000, help="Phase 1: bootstrap iterations")
    ap.add_argument("--ci", type=float, default=0.95, help="Phase 1: confidence level")
    ap.add_argument("--k", type=int, default=3, help="Phase 2: number of folds")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 epoch per run")
    args = ap.parse_args()

    U.seed_everything(C.SEED)

    if args.phase in (None, 1):
        run_phase1(args.n_bootstrap, args.ci, quick=args.quick)
    if args.phase in (None, 2):
        run_phase2(args.k, quick=args.quick)


if __name__ == "__main__":
    main()
