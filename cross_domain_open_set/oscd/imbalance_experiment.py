"""
imbalance_experiment.py
========================
Reviewer question: dataset is imbalanced -- not all identities have scanner
data, some have only smartphone (perspective) data. Two experiments,
CompNet only, quantifying the effect of that imbalance separately at
train time and at inference time.

Shared vocabulary used throughout:
    dual_ids             identities with BOTH scanner and smartphone data
    smartphone_only_ids  identities with ONLY smartphone (perspective) data

EXPERIMENT 1 -- effect of the imbalance on TRAINING
    Held out: 40 identities from dual_ids as a FIXED test set (both domains
    pooled per identity, 50/50 gallery/probe split -- config.TEST_GALLERY_RATIO).
    Two training configurations, SAME total ID count (120) and therefore
    the SAME classification-head size, so any gap is attributable to domain
    COMPOSITION, not population size or head size:
      Mode A ("balanced")   : 120 identities, ALL dual-domain (both domains
                              used for every training identity).
      Mode B ("imbalanced") : 90 dual-domain identities (both domains) + 30
                              smartphone-only identities (smartphone only)
                              -- mirrors the dataset's real imbalance.
    Caveat (stated explicitly in the printed output, not hidden): Mode B
    has FEWER total training images than Mode A, since its 30 smartphone-
    only identities contribute one domain's worth of images instead of two.
    A gap between modes is therefore "imbalance + somewhat less data",
    not purely "imbalance" -- report accordingly.

EXPERIMENT 2 -- effect of the imbalance on INFERENCE
    SAME 40-identity test set as Experiment 1 (same seed -> same split).
    Train ONCE on everything else (remaining dual_ids, both domains, +
    smartphone_only_ids, smartphone domain only -- i.e. "the rest of the
    data" as it actually exists, no manipulation).
    Two gallery/probe configurations evaluated from that SAME trained model
    (no retraining, no extra forward passes for probe -- see below):
      Mode 1 (full)          : gallery + probe both pooled from both domains
                                per test identity (the natural 50/50 split).
      Mode 2 (smartphone-only gallery) : gallery = Mode 1's gallery with
                                scanner entries DROPPED (not moved to
                                probe, per your instruction); probe =
                                UNCHANGED from Mode 1 (already a natural
                                domain mix, so re-using it is correct, not
                                a shortcut).
    Mode 2's gallery is a strict subset of Mode 1's gallery, so this only
    requires ONE embedding-extraction pass over Mode 1's gallery -- Mode 2's
    embeddings are just a boolean-masked subset of that same array.

Usage:
    python imbalance_experiment.py                      # both experiments
    python imbalance_experiment.py --experiment 1
    python imbalance_experiment.py --experiment 2
    python imbalance_experiment.py --quick               # smoke test: 1 epoch
    python imbalance_experiment.py --n-test-ids 40 --n-train-a 120 \
        --n-train-b-dual 90 --n-train-b-sp 30
"""
import os
import random
import time
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

import config as C
import dataset as D
from dataset import _gallery_probe_split
from confidence_interval import train_compnet, train_compnet_model, point_eer_rank1
import utils as U

METHOD = "compnet"

# ── Default identity-count targets (all adjustable via CLI) ─────────
N_TEST_IDS          = 40    # held out from dual_ids, shared by both experiments
N_TRAIN_MODE_A      = 120   # Experiment 1, Mode A: all-dual-domain training IDs
N_TRAIN_MODE_B_DUAL = 90    # Experiment 1, Mode B: dual-domain portion
N_TRAIN_MODE_B_SP   = 30    # Experiment 1, Mode B: smartphone-only portion


# ══════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════


def point_eer_rank1(gal_feats, gal_labels, prb_feats, prb_labels):
    """Same metric definition as utils.evaluate(), vectorised (no I/O)."""
    gal_n = gal_feats / (np.linalg.norm(gal_feats, axis=1, keepdims=True) + 1e-8)
    prb_n = prb_feats / (np.linalg.norm(prb_feats, axis=1, keepdims=True) + 1e-8)
    sim = prb_n @ gal_n.T
    rank1 = 100.0 * (gal_labels[sim.argmax(axis=1)] == prb_labels).mean()
    same = (prb_labels[:, None] == gal_labels[None, :])
    scores = sim.ravel()
    labels = np.where(same, 1, -1).ravel()
    eer, _ = U.compute_eer(np.column_stack([scores, labels]))
    return eer * 100.0, rank1, sim

def train_compnet_model(train_samples, gallery_samples, probe_samples, num_classes,
                         init_tag, num_epochs=None, eval_every=None):
    """Trains CompNet exactly as train_compnet() does below, but returns the
    trained (best-Rank-1 checkpoint loaded) `baseline` object itself -- not
    pre-extracted embeddings -- plus the gallery/probe loaders used for its
    internal periodic checkpoint-selection eval. This lets callers extract
    embeddings for additional/different gallery-probe configurations
    afterward without retraining (see imbalance_experiment.py)."""
    cfg = dict(C.METHODS[METHOD])
    num_epochs = num_epochs or cfg["num_epochs"]
    eval_every = eval_every or C.EVAL_EVERY

    train_loader = D.make_loader(train_samples, METHOD, True, cfg["batch_size"], C.NUM_WORKERS)
    gallery_loader = D.make_loader(gallery_samples, METHOD, False, cfg["batch_size"], C.NUM_WORKERS)
    probe_loader = D.make_loader(probe_samples, METHOD, False, cfg["batch_size"], C.NUM_WORKERS)

    baseline = M.REGISTRY[METHOD](num_classes, cfg, C.DEVICE)
    get_or_create_init_state(baseline, num_classes, init_tag)     # same init across folds
    optimizer, scheduler = baseline.build_optimizer()

    best_rank1 = -1.0
    best_state = None
    for epoch in range(1, num_epochs + 1):
        baseline.train_mode()
        for batch in train_loader:
            baseline.train_step(batch, optimizer)
        scheduler.step()
        if epoch % eval_every == 0 or epoch == num_epochs:
            baseline.eval_mode()
            eer, rank1 = U.evaluate(baseline.embed, gallery_loader, probe_loader, C.DEVICE)
            if rank1 > best_rank1:
                best_rank1 = rank1
                best_state = copy.deepcopy(baseline.state_dict())

    if best_state is not None:
        baseline.load_state_dict(best_state)
    baseline.eval_mode()
    return baseline, gallery_loader, probe_loader


def train_compnet(train_samples, gallery_samples, probe_samples, num_classes,
                   init_tag, num_epochs=None, eval_every=None):
    """Thin wrapper over train_compnet_model() preserving the original
    return contract (embeddings, not the model) used by Phase 1 / Phase 2.
    Returns (gal_feats, gal_labels, prb_feats, prb_labels) from the
    best-Rank-1 checkpoint."""
    baseline, gallery_loader, probe_loader = train_compnet_model(
        train_samples, gallery_samples, probe_samples, num_classes,
        init_tag, num_epochs=num_epochs, eval_every=eval_every)
    baseline.eval_mode()
    gal_feats, gal_labels = U.extract_embeddings(baseline.embed, gallery_loader, C.DEVICE)
    prb_feats, prb_labels = U.extract_embeddings(baseline.embed, probe_loader, C.DEVICE)
    return gal_feats, gal_labels, prb_feats, prb_labels


def collect_pools():
    """Returns (persp_all, scanner_paths, dual_ids, smartphone_only_ids).
    persp_all: identity -> [smartphone/perspective image paths] (all conditions pooled).
    dual_ids: identities with BOTH scanner and smartphone data.
    smartphone_only_ids: identities with smartphone data but NO scanner data."""
    cond_paths = D.collect_perspective(C.DATA_ROOT)
    scanner_paths = D.collect_scanner(C.DATA_ROOT, C.SCANNER_SPECTRA)
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    all_persp_ids = set(persp_all.keys())
    scanner_ids = set(scanner_paths.keys())
    dual_ids = sorted(all_persp_ids & scanner_ids)
    smartphone_only_ids = sorted(all_persp_ids - scanner_ids)
    return persp_all, scanner_paths, dual_ids, smartphone_only_ids


def pooled_samples(ident_list, persp_all, scanner_paths, label_map, domains):
    """(path, label) samples for the given identities, restricted to the
    requested domain(s) ('smartphone', 'scanner', or both)."""
    samples = []
    for ident in ident_list:
        if "smartphone" in domains:
            for p in persp_all.get(ident, []):
                samples.append((p, label_map[ident]))
        if "scanner" in domains:
            for p in scanner_paths.get(ident, []):
                samples.append((p, label_map[ident]))
    return samples


def build_test_split(test_ids, persp_all, scanner_paths, gallery_ratio, seed):
    """Pools BOTH domains for each test identity and splits into gallery/
    probe via dataset.py's own _gallery_probe_split (same logic S_scanner
    uses), so this is consistent with the rest of the project."""
    rng = random.Random(seed)
    label_map = {ident: i for i, ident in enumerate(sorted(test_ids))}
    pooled = {ident: list(persp_all.get(ident, [])) + list(scanner_paths.get(ident, []))
              for ident in test_ids}
    gallery, probe = _gallery_probe_split(pooled, label_map, gallery_ratio, rng)
    return gallery, probe, label_map


def sample_test_ids(dual_ids, n_test_ids, seed):
    rng = random.Random(seed)
    shuffled = dual_ids[:]
    rng.shuffle(shuffled)
    n = min(n_test_ids, len(shuffled))
    if n < n_test_ids:
        print(f"  [WARN] requested {n_test_ids} test IDs but only {len(shuffled)} "
              f"dual-domain identities exist -- using {n}.")
    test_ids = sorted(shuffled[:n])
    remaining_dual = sorted(set(dual_ids) - set(test_ids))
    return test_ids, remaining_dual


def sample_ids(pool, n, seed, tag=""):
    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    k = min(n, len(shuffled))
    if k < n:
        print(f"  [WARN] requested {n} {tag} IDs but only {len(shuffled)} available -- using {k}.")
    return sorted(shuffled[:k])


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 1 -- imbalance effect on TRAINING
# ══════════════════════════════════════════════════════════════

def run_experiment1(n_test_ids=N_TEST_IDS, n_train_a=N_TRAIN_MODE_A,
                     n_train_b_dual=N_TRAIN_MODE_B_DUAL, n_train_b_sp=N_TRAIN_MODE_B_SP,
                     seed=None, quick=False):
    seed = C.SEED if seed is None else seed
    print(f"\n{'='*70}\nEXPERIMENT 1 -- imbalance effect on TRAINING, CompNet\n{'='*70}")

    persp_all, scanner_paths, dual_ids, smartphone_only_ids = collect_pools()
    print(f"  dual_ids (scanner+smartphone): {len(dual_ids)}")
    print(f"  smartphone_only_ids          : {len(smartphone_only_ids)}")

    test_ids, remaining_dual = sample_test_ids(dual_ids, n_test_ids, seed)
    print(f"  test_ids: {len(test_ids)} (from dual_ids)  remaining_dual: {len(remaining_dual)}")

    gallery, probe, _ = build_test_split(test_ids, persp_all, scanner_paths,
                                          C.TEST_GALLERY_RATIO, seed)
    print(f"  [shared test set] gallery={len(gallery)} probe={len(probe)}")

    # Mode A: n_train_a identities, ALL dual-domain (both domains each)
    train_ids_a = sample_ids(remaining_dual, n_train_a, seed + 1, "Mode-A")
    label_map_a = {ident: i for i, ident in enumerate(train_ids_a)}
    train_samples_a = pooled_samples(train_ids_a, persp_all, scanner_paths, label_map_a,
                                      domains=("smartphone", "scanner"))

    # Mode B: n_train_b_dual dual-domain (both domains) + n_train_b_sp
    # smartphone-only (smartphone domain only) -- mirrors the real imbalance
    train_ids_b_dual = sample_ids(remaining_dual, n_train_b_dual, seed + 2, "Mode-B-dual")
    train_ids_b_sp = sample_ids(smartphone_only_ids, n_train_b_sp, seed + 3, "Mode-B-smartphone-only")
    train_ids_b = sorted(train_ids_b_dual + train_ids_b_sp)
    label_map_b = {ident: i for i, ident in enumerate(train_ids_b)}
    train_samples_b = (
        pooled_samples(train_ids_b_dual, persp_all, scanner_paths, label_map_b,
                        domains=("smartphone", "scanner"))
        + pooled_samples(train_ids_b_sp, persp_all, scanner_paths, label_map_b,
                          domains=("smartphone",)))

    print(f"  Mode A (balanced)  : {len(train_ids_a)} IDs, all dual-domain, "
          f"{len(train_samples_a)} training images, {len(train_ids_a)}-way head")
    print(f"  Mode B (imbalanced): {len(train_ids_b_dual)} dual + {len(train_ids_b_sp)} "
          f"smartphone-only = {len(train_ids_b)} IDs, {len(train_samples_b)} training images, "
          f"{len(train_ids_b)}-way head")
    if len(train_samples_a) != len(train_samples_b):
        print(f"  [NOTE] Mode B has {len(train_samples_a) - len(train_samples_b)} fewer training "
              f"images than Mode A (expected: smartphone-only IDs contribute one domain, "
              f"not two) -- keep this in mind when interpreting any gap below.")

    rows = []
    for mode_name, train_samples, train_ids in [
        ("Mode A (balanced: all-dual)", train_samples_a, train_ids_a),
        ("Mode B (imbalanced: dual+smartphone-only)", train_samples_b, train_ids_b),
    ]:
        print(f"\n  --- {mode_name} ---")
        t0 = time.time()
        gal_feats, gal_labels, prb_feats, prb_labels = train_compnet(
            train_samples, gallery, probe, len(train_ids),
            init_tag=f"exp1_{mode_name[5]}",
            num_epochs=1 if quick else None, eval_every=1 if quick else None)
        eer, rank1, _ = point_eer_rank1(gal_feats, gal_labels, prb_feats, prb_labels)
        print(f"    EER={eer:.3f}%  Rank1={rank1:.2f}%  ({(time.time()-t0)/60:.1f} min)")
        rows.append({"mode": mode_name, "n_train_ids": len(train_ids),
                     "n_train_images": len(train_samples),
                     "EER_pct": round(eer, 3), "Rank1_pct": round(rank1, 2)})

    df = pd.DataFrame(rows).set_index("mode")
    out_dir = C.BASE_RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "imbalance_experiment1_training.csv"))

    print(f"\n{'-'*70}\nEXPERIMENT 1 SUMMARY -- same test set, same #train-IDs (120), "
          f"same head size (120-way)\n{'-'*70}")
    print(df.to_string())
    print(f"\nSaved: {out_dir}/imbalance_experiment1_training.csv")
    return df


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 2 -- imbalance effect on INFERENCE
# ══════════════════════════════════════════════════════════════

def run_experiment2(n_test_ids=N_TEST_IDS, seed=None, quick=False):
    seed = C.SEED if seed is None else seed
    print(f"\n{'='*70}\nEXPERIMENT 2 -- imbalance effect on INFERENCE, CompNet\n{'='*70}")

    persp_all, scanner_paths, dual_ids, smartphone_only_ids = collect_pools()
    test_ids, remaining_dual = sample_test_ids(dual_ids, n_test_ids, seed)
    print(f"  test_ids: {len(test_ids)} (from dual_ids, SAME split as Experiment 1 "
          f"given the same seed)  remaining_dual: {len(remaining_dual)}")

    # Train ONCE on "the rest of the data", as it actually exists -- no
    # manipulation of domain composition here, unlike Experiment 1.
    train_ids = sorted(remaining_dual + smartphone_only_ids)
    label_map_train = {ident: i for i, ident in enumerate(train_ids)}
    train_samples = (
        pooled_samples(remaining_dual, persp_all, scanner_paths, label_map_train,
                        domains=("smartphone", "scanner"))
        + pooled_samples(smartphone_only_ids, persp_all, scanner_paths, label_map_train,
                          domains=("smartphone",)))
    print(f"  train: {len(train_ids)} IDs ({len(remaining_dual)} dual + "
          f"{len(smartphone_only_ids)} smartphone-only), {len(train_samples)} images")

    # Mode 1 gallery/probe: pool BOTH domains per test identity, 50/50 split.
    gallery_mode1, probe_mode1, _ = build_test_split(
        test_ids, persp_all, scanner_paths, C.TEST_GALLERY_RATIO, seed)
    print(f"  Mode 1 (full)              : gallery={len(gallery_mode1)} probe={len(probe_mode1)}")

    # Mode 2: gallery = Mode 1's gallery with scanner entries DROPPED (not
    # moved to probe); probe = UNCHANGED (already a natural domain mix).
    scanner_path_set = {p for paths in scanner_paths.values() for p in paths}
    keep_mask = np.array([p not in scanner_path_set for p, _ in gallery_mode1])
    gallery_mode2 = [s for s, keep in zip(gallery_mode1, keep_mask) if keep]
    probe_mode2 = probe_mode1
    print(f"  Mode 2 (smartphone-only gal): gallery={len(gallery_mode2)} "
          f"(dropped {len(gallery_mode1) - len(gallery_mode2)} scanner entries, not moved to probe)  "
          f"probe={len(probe_mode2)} (unchanged from Mode 1)")

    ids_with_gallery_mode1 = set(l for _, l in gallery_mode1)
    ids_with_gallery_mode2 = set(l for _, l in gallery_mode2)
    zero_gallery_ids = ids_with_gallery_mode1 - ids_with_gallery_mode2
    if zero_gallery_ids:
        print(f"  [NOTE] {len(zero_gallery_ids)} test identities have ZERO smartphone-domain "
              f"gallery entries under Mode 2 (all their Mode-1 gallery images happened to be "
              f"scanner-domain) -- they become effectively unrecoverable in Mode 2's Rank-1, "
              f"which is a real, expected consequence of a smartphone-only enrollment policy, "
              f"not a bug.")

    print(f"\n  --- training (once) ---")
    t0 = time.time()
    baseline, _, _ = train_compnet_model(
        train_samples, gallery_mode1, probe_mode1, len(train_ids),
        init_tag="exp2_shared_model",
        num_epochs=1 if quick else None, eval_every=1 if quick else None)
    print(f"    done ({(time.time()-t0)/60:.1f} min)")

    # ONE embedding pass over Mode 1's (full) gallery + probe; Mode 2's
    # gallery embeddings are just the boolean-masked subset of the same array.
    cfg = dict(C.METHODS[METHOD])
    gallery_loader = D.make_loader(gallery_mode1, METHOD, False, cfg["batch_size"], C.NUM_WORKERS)
    probe_loader = D.make_loader(probe_mode1, METHOD, False, cfg["batch_size"], C.NUM_WORKERS)
    gal_feats_full, gal_labels_full = U.extract_embeddings(baseline.embed, gallery_loader, C.DEVICE)
    prb_feats, prb_labels = U.extract_embeddings(baseline.embed, probe_loader, C.DEVICE)

    eer1, rank1_1, _ = point_eer_rank1(gal_feats_full, gal_labels_full, prb_feats, prb_labels)
    eer2, rank1_2, _ = point_eer_rank1(gal_feats_full[keep_mask], gal_labels_full[keep_mask],
                                        prb_feats, prb_labels)

    print(f"\n  Mode 1 (full gallery)              : EER={eer1:.3f}%  Rank1={rank1_1:.2f}%")
    print(f"  Mode 2 (smartphone-only gallery)   : EER={eer2:.3f}%  Rank1={rank1_2:.2f}%")

    df = pd.DataFrame([
        {"mode": "Mode 1 (full gallery, both domains)", "gallery_size": len(gallery_mode1),
         "probe_size": len(probe_mode1), "EER_pct": round(eer1, 3), "Rank1_pct": round(rank1_1, 2)},
        {"mode": "Mode 2 (smartphone-only gallery)", "gallery_size": len(gallery_mode2),
         "probe_size": len(probe_mode2), "EER_pct": round(eer2, 3), "Rank1_pct": round(rank1_2, 2)},
    ]).set_index("mode")

    out_dir = C.BASE_RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "imbalance_experiment2_inference.csv"))

    print(f"\n{'-'*70}\nEXPERIMENT 2 SUMMARY -- ONE trained model, two gallery configurations, "
          f"SAME probe\n{'-'*70}")
    print(df.to_string())
    print(f"\nSaved: {out_dir}/imbalance_experiment2_inference.csv")
    return df


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=int, default=None, choices=[1, 2],
                     help="run only experiment 1 or 2 (default: both)")
    ap.add_argument("--n-test-ids", type=int, default=N_TEST_IDS)
    ap.add_argument("--n-train-a", type=int, default=N_TRAIN_MODE_A)
    ap.add_argument("--n-train-b-dual", type=int, default=N_TRAIN_MODE_B_DUAL)
    ap.add_argument("--n-train-b-sp", type=int, default=N_TRAIN_MODE_B_SP)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 epoch per run")
    args = ap.parse_args()

    U.seed_everything(C.SEED)

    if args.experiment in (None, 1):
        run_experiment1(args.n_test_ids, args.n_train_a, args.n_train_b_dual,
                         args.n_train_b_sp, seed=args.seed, quick=args.quick)
    if args.experiment in (None, 2):
        run_experiment2(args.n_test_ids, seed=args.seed, quick=args.quick)


if __name__ == "__main__":
    main()
