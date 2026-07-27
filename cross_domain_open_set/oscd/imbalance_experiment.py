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
import model as M
import utils as U

METHOD = "compnet"

# ── Default identity-count targets (all adjustable via CLI) ─────────
N_TEST_IDS          = 40    # held out from dual_ids, shared by both experiments
N_TRAIN_MODE_A      = 120   # Experiment 1, Mode A: all-dual-domain training IDs
N_TRAIN_MODE_B_DUAL = 90    # Experiment 1, Mode B: dual-domain portion
N_TRAIN_MODE_B_SP   = 30    # Experiment 1, Mode B: smartphone-only portion
N_BOOTSTRAP          = 1000  # identity-level bootstrap iterations for both experiments
CI_LEVEL             = 0.95


# ══════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════

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


def _three_way_split(id2paths, label_map, gallery_frac, probe_frac, rng):
    """Per-identity random 3-way split into (gallery, probe, discarded),
    using gallery_frac/probe_frac of EACH identity's available images (not
    a fixed count). Small-sample handling mirrors dataset.py's
    _gallery_probe_split: an identity with only 1 image gets that same
    image in both gallery and probe; identities with 2+ images always get
    at least 1 in each, capped so gallery+probe never exceeds what's
    available."""
    gallery, probe = [], []
    for ident, paths in id2paths.items():
        paths = list(paths)
        rng.shuffle(paths)
        n = len(paths)
        if n == 0:
            continue
        if n == 1:
            gallery.append((paths[0], label_map[ident]))
            probe.append((paths[0], label_map[ident]))
            continue
        n_gal = min(max(1, round(n * gallery_frac)), n - 1)
        n_prb = min(max(1, round(n * probe_frac)), n - n_gal)
        for p in paths[:n_gal]:
            gallery.append((p, label_map[ident]))
        for p in paths[n_gal:n_gal + n_prb]:
            probe.append((p, label_map[ident]))
        # remaining paths[n_gal + n_prb:] are deliberately discarded
    return gallery, probe


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


def _match_total_ids(train_ids_a, train_ids_b_dual, train_ids_b_sp, seed):
    """Enforce that Mode A and Mode B end up with the SAME total identity
    count -- the actual design goal of Experiment 1 (any EER/Rank-1 gap
    should be attributable to domain COMPOSITION, not classification-head
    size / population size). Without this, independently clamping each
    mode's own request against its own pool's availability can silently
    leave the two modes with different totals (e.g. Mode A's single pool
    runs out before Mode B's two pools do), reintroducing exactly the
    confound this experiment is designed to control for.

    Trims whichever mode ended up larger:
      - Mode A: trimmed directly (single homogeneous pool).
      - Mode B: trimmed from the DUAL portion FIRST, preserving the
        smartphone-only count exactly, since that count is the actual
        variable under test. Only trims the smartphone-only portion too
        if the dual portion alone can't absorb the full reduction (rare
        -- a loud warning is printed if this happens, since it means the
        intended smartphone-only count silently changed).
    """
    n_a = len(train_ids_a)
    n_b = len(train_ids_b_dual) + len(train_ids_b_sp)
    if n_a == n_b:
        return train_ids_a, train_ids_b_dual, train_ids_b_sp

    rng = random.Random(seed)
    if n_a > n_b:
        excess = n_a - n_b
        print(f"  [MATCH] Mode A ({n_a}) > Mode B ({n_b}) -- trimming {excess} "
              f"identities from Mode A so both modes have the SAME total ID count.")
        shuffled = train_ids_a[:]
        rng.shuffle(shuffled)
        train_ids_a = sorted(shuffled[excess:])
    else:
        excess = n_b - n_a
        print(f"  [MATCH] Mode B ({n_b}) > Mode A ({n_a}) -- trimming {excess} "
              f"identities from Mode B's DUAL portion (preserving its smartphone-only "
              f"count exactly) so both modes have the SAME total ID count.")
        if excess <= len(train_ids_b_dual):
            shuffled = train_ids_b_dual[:]
            rng.shuffle(shuffled)
            train_ids_b_dual = sorted(shuffled[excess:])
        else:
            print(f"  [WARN] Mode B's dual portion ({len(train_ids_b_dual)}) alone can't "
                  f"absorb the full excess ({excess}) -- also trimming Mode B's "
                  f"smartphone-only portion, which changes the REQUESTED smartphone-only "
                  f"count. Consider lowering --n-train-a or --n-train-b-dual/--n-train-b-sp.")
            remaining_excess = excess - len(train_ids_b_dual)
            train_ids_b_dual = []
            shuffled_sp = train_ids_b_sp[:]
            rng.shuffle(shuffled_sp)
            train_ids_b_sp = sorted(shuffled_sp[remaining_excess:])

    n_a_final = len(train_ids_a)
    n_b_final = len(train_ids_b_dual) + len(train_ids_b_sp)
    assert n_a_final == n_b_final, "internal error: ID-count matching failed"
    return train_ids_a, train_ids_b_dual, train_ids_b_sp


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 1 -- imbalance effect on TRAINING
# ══════════════════════════════════════════════════════════════

def run_experiment1(n_test_ids=N_TEST_IDS, n_train_a=N_TRAIN_MODE_A,
                     n_train_b_dual=N_TRAIN_MODE_B_DUAL, n_train_b_sp=N_TRAIN_MODE_B_SP,
                     seed=None, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, quick=False):
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

    # Mode B: n_train_b_dual dual-domain (both domains) + n_train_b_sp
    # smartphone-only (smartphone domain only) -- mirrors the real imbalance
    train_ids_b_dual = sample_ids(remaining_dual, n_train_b_dual, seed + 2, "Mode-B-dual")
    train_ids_b_sp = sample_ids(smartphone_only_ids, n_train_b_sp, seed + 3, "Mode-B-smartphone-only")

    # Enforce SAME total ID count in both modes (see _match_total_ids docstring
    # -- without this, independent per-pool clamping can silently leave the
    # two modes with different totals/head-sizes, reintroducing the exact
    # confound this experiment exists to control for).
    train_ids_a, train_ids_b_dual, train_ids_b_sp = _match_total_ids(
        train_ids_a, train_ids_b_dual, train_ids_b_sp, seed + 4)

    label_map_a = {ident: i for i, ident in enumerate(train_ids_a)}
    train_samples_a = pooled_samples(train_ids_a, persp_all, scanner_paths, label_map_a,
                                      domains=("smartphone", "scanner"))

    train_ids_b = sorted(train_ids_b_dual + train_ids_b_sp)
    label_map_b = {ident: i for i, ident in enumerate(train_ids_b)}
    train_samples_b = (
        pooled_samples(train_ids_b_dual, persp_all, scanner_paths, label_map_b,
                        domains=("smartphone", "scanner"))
        + pooled_samples(train_ids_b_sp, persp_all, scanner_paths, label_map_b,
                          domains=("smartphone",)))

    assert len(train_ids_a) == len(train_ids_b), \
        f"Mode A ({len(train_ids_a)}) and Mode B ({len(train_ids_b)}) ID counts still differ"

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
        gal_feats, gal_labels, prb_feats, prb_labels = M.train_baseline(
            METHOD, train_samples, gallery, probe, len(train_ids),
            init_tag=f"exp1_{mode_name[5]}",
            num_epochs=1 if quick else None, eval_every=1 if quick else None)
        eer, rank1, sim = U.point_eer_rank1(gal_feats, gal_labels, prb_feats, prb_labels)
        ci_res = U.identity_bootstrap_ci(sim, gal_labels, prb_labels,
                                          n_bootstrap=n_bootstrap, ci=ci, seed=seed)
        print(f"    EER={eer:.3f}% [{ci_res['eer_ci_lo']:.3f}, {ci_res['eer_ci_hi']:.3f}]  "
              f"Rank1={rank1:.2f}% [{ci_res['rank1_ci_lo']:.2f}, {ci_res['rank1_ci_hi']:.2f}]  "
              f"({(time.time()-t0)/60:.1f} min)")
        rows.append({"mode": mode_name, "n_train_ids": len(train_ids),
                     "n_train_images": len(train_samples),
                     "EER_pct": round(eer, 3),
                     "EER_CI_lo": round(ci_res["eer_ci_lo"], 3),
                     "EER_CI_hi": round(ci_res["eer_ci_hi"], 3),
                     "Rank1_pct": round(rank1, 2),
                     "Rank1_CI_lo": round(ci_res["rank1_ci_lo"], 2),
                     "Rank1_CI_hi": round(ci_res["rank1_ci_hi"], 2)})

    df = pd.DataFrame(rows).set_index("mode")
    out_dir = C.BASE_RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "imbalance_experiment1_training.csv"))

    print(f"\n{'-'*70}\nEXPERIMENT 1 SUMMARY -- same test set, same #train-IDs ({len(train_ids_a)}), "
          f"same head size ({len(train_ids_a)}-way), bootstrap 95% CI "
          f"(identity-level resampling, B={n_bootstrap})\n{'-'*70}")
    print(df.to_string())
    print(f"\nSaved: {out_dir}/imbalance_experiment1_training.csv")
    return df


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 2 -- imbalance effect on INFERENCE
#  (fair version: Mode 1 uses 25% gallery + 25% probe of each identity's
#  full pool -- half the samples overall; Mode 2 uses ALL smartphone
#  samples for gallery + 50% of the ORIGINAL full probe, with any
#  gallery/probe overlap removed to avoid trivial self-match leakage)
# ══════════════════════════════════════════════════════════════

def run_experiment2(n_test_ids=N_TEST_IDS, seed=None, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL,
                     quick=False):
    seed = C.SEED if seed is None else seed
    print(f"\n{'='*70}\nEXPERIMENT 2 -- imbalance effect on INFERENCE, CompNet\n{'='*70}")

    persp_all, scanner_paths, dual_ids, smartphone_only_ids = collect_pools()
    test_ids, remaining_dual = sample_test_ids(dual_ids, n_test_ids, seed)
    print(f"  test_ids: {len(test_ids)} (from dual_ids, SAME split as Experiment 1 "
          f"given the same seed)  remaining_dual: {len(remaining_dual)}")

    train_ids = sorted(remaining_dual + smartphone_only_ids)
    label_map_train = {ident: i for i, ident in enumerate(train_ids)}
    train_samples = (
        pooled_samples(remaining_dual, persp_all, scanner_paths, label_map_train,
                        domains=("smartphone", "scanner"))
        + pooled_samples(smartphone_only_ids, persp_all, scanner_paths, label_map_train,
                          domains=("smartphone",)))
    print(f"  train: {len(train_ids)} IDs ({len(remaining_dual)} dual + "
          f"{len(smartphone_only_ids)} smartphone-only), {len(train_samples)} images")

    # "Previous version": the original full 50/50 pooled-both-domain split.
    # Kept only as the base pool Mode 2's probe is drawn from below -- not
    # used directly as either mode's final gallery/probe anymore.
    prev_gallery, prev_probe, test_label_map = build_test_split(
        test_ids, persp_all, scanner_paths, C.TEST_GALLERY_RATIO, seed)
    print(f"  [previous version, reference only] gallery={len(prev_gallery)} "
          f"probe={len(prev_probe)}")

    # Mode 1 (fair): 25% gallery + 25% probe of each test identity's FULL
    # pool (both domains) -- half of each identity's total samples used
    # overall, the other half discarded.
    pooled = {ident: list(persp_all.get(ident, [])) + list(scanner_paths.get(ident, []))
              for ident in test_ids}
    rng1 = random.Random(seed + 100)
    gallery_mode1, probe_mode1 = _three_way_split(pooled, test_label_map, 0.25, 0.25, rng1)
    print(f"  Mode 1 (25% gallery + 25% probe, both domains): gallery={len(gallery_mode1)} "
          f"probe={len(probe_mode1)}")

    # Mode 2 (fair): gallery = ALL smartphone samples for every test identity
    # (the realistic "everything available" smartphone-only enrollment --
    # not just whatever fraction happened to land in a random split); probe
    # = 50% random subsample of the PREVIOUS probe. Any of those probe
    # candidates that are ALSO in Mode 2's gallery are removed first: an
    # image appearing in both gallery and probe would let the model
    # trivially self-match rather than genuinely recognize the identity.
    gallery_mode2 = pooled_samples(test_ids, persp_all, scanner_paths, test_label_map,
                                    domains=("smartphone",))
    gallery_mode2_paths = {p for p, _ in gallery_mode2}

    rng2 = random.Random(seed + 200)
    prev_probe_shuffled = prev_probe[:]
    rng2.shuffle(prev_probe_shuffled)
    n_probe2 = max(1, round(len(prev_probe_shuffled) * 0.5))
    probe_mode2_candidates = prev_probe_shuffled[:n_probe2]
    leaked = [s for s in probe_mode2_candidates if s[0] in gallery_mode2_paths]
    probe_mode2 = [s for s in probe_mode2_candidates if s[0] not in gallery_mode2_paths]
    if leaked:
        print(f"  [NOTE] Removed {len(leaked)} sample(s) from Mode 2's probe that were also "
              f"present in Mode 2's all-smartphone gallery (would otherwise let the model "
              f"trivially self-match instead of genuinely recognizing the identity).")
    print(f"  Mode 2 (all-smartphone gallery, 50% of previous probe): "
          f"gallery={len(gallery_mode2)} probe={len(probe_mode2)}")

    print(f"\n  --- training (once) ---")
    t0 = time.time()
    # Best-Rank-1 checkpoint tracked against Mode 1 (the smaller, "fair,
    # matched" config) during training -- Mode 2's gallery is now large and
    # not a subset of anything Mode 1 uses, so it can't double as the
    # tracking target too.
    baseline, _, _ = M.train_baseline_model(
        METHOD, train_samples, gallery_mode1, probe_mode1, len(train_ids),
        init_tag="exp2_shared_model_v2",
        num_epochs=1 if quick else None, eval_every=1 if quick else None)
    print(f"    done ({(time.time()-t0)/60:.1f} min)")

    cfg = dict(C.METHODS[METHOD])

    def _embed(samples):
        loader = D.make_loader(samples, METHOD, False, cfg["batch_size"], C.NUM_WORKERS)
        return U.extract_embeddings(baseline.embed, loader, C.DEVICE)

    gal1_feats, gal1_labels = _embed(gallery_mode1)
    prb1_feats, prb1_labels = _embed(probe_mode1)
    gal2_feats, gal2_labels = _embed(gallery_mode2)
    prb2_feats, prb2_labels = _embed(probe_mode2)

    eer1, rank1_1, sim1 = U.point_eer_rank1(gal1_feats, gal1_labels, prb1_feats, prb1_labels)
    eer2, rank1_2, sim2 = U.point_eer_rank1(gal2_feats, gal2_labels, prb2_feats, prb2_labels)

    ci1 = U.identity_bootstrap_ci(sim1, gal1_labels, prb1_labels,
                                   n_bootstrap=n_bootstrap, ci=ci, seed=seed)
    ci2 = U.identity_bootstrap_ci(sim2, gal2_labels, prb2_labels,
                                   n_bootstrap=n_bootstrap, ci=ci, seed=seed)

    print(f"\n  Mode 1 (25%/25% split)             : EER={eer1:.3f}% "
          f"[{ci1['eer_ci_lo']:.3f}, {ci1['eer_ci_hi']:.3f}]  "
          f"Rank1={rank1_1:.2f}% [{ci1['rank1_ci_lo']:.2f}, {ci1['rank1_ci_hi']:.2f}]")
    print(f"  Mode 2 (all-smartphone gallery)    : EER={eer2:.3f}% "
          f"[{ci2['eer_ci_lo']:.3f}, {ci2['eer_ci_hi']:.3f}]  "
          f"Rank1={rank1_2:.2f}% [{ci2['rank1_ci_lo']:.2f}, {ci2['rank1_ci_hi']:.2f}]")

    df = pd.DataFrame([
        {"mode": "Mode 1 (25% gallery + 25% probe, both domains)",
         "gallery_size": len(gallery_mode1), "probe_size": len(probe_mode1),
         "EER_pct": round(eer1, 3),
         "EER_CI_lo": round(ci1["eer_ci_lo"], 3), "EER_CI_hi": round(ci1["eer_ci_hi"], 3),
         "Rank1_pct": round(rank1_1, 2),
         "Rank1_CI_lo": round(ci1["rank1_ci_lo"], 2), "Rank1_CI_hi": round(ci1["rank1_ci_hi"], 2)},
        {"mode": "Mode 2 (all-smartphone gallery, 50% of previous probe)",
         "gallery_size": len(gallery_mode2), "probe_size": len(probe_mode2),
         "EER_pct": round(eer2, 3),
         "EER_CI_lo": round(ci2["eer_ci_lo"], 3), "EER_CI_hi": round(ci2["eer_ci_hi"], 3),
         "Rank1_pct": round(rank1_2, 2),
         "Rank1_CI_lo": round(ci2["rank1_ci_lo"], 2), "Rank1_CI_hi": round(ci2["rank1_ci_hi"], 2)},
    ]).set_index("mode")

    out_dir = C.BASE_RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "imbalance_experiment2_inference.csv"))

    print(f"\n{'-'*70}\nEXPERIMENT 2 SUMMARY -- ONE trained model, two FAIR gallery/probe "
          f"configurations, bootstrap 95% CI (identity-level resampling, B={n_bootstrap})\n{'-'*70}")
    print(df.to_string())
    print(f"\nSaved: {out_dir}/imbalance_experiment2_inference.csv")
    return df



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=int, default=None, choices=[1, 2],
                     help="run only experiment 1 or 2 (default: both)")
    ap.add_argument("--n-test-ids", type=int, default=N_TEST_IDS)
    ap.add_argument("--n-train-a", type=int, default=N_TRAIN_MODE_A)
    ap.add_argument("--n-train-b-dual", type=int, default=N_TRAIN_MODE_B_DUAL)
    ap.add_argument("--n-train-b-sp", type=int, default=N_TRAIN_MODE_B_SP)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                     help="identity-level bootstrap iterations for both experiments")
    ap.add_argument("--ci", type=float, default=CI_LEVEL, help="confidence level")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 epoch per run")
    args = ap.parse_args()

    U.seed_everything(C.SEED)

    if args.experiment in (None, 1):
        run_experiment1(args.n_test_ids, args.n_train_a, args.n_train_b_dual,
                         args.n_train_b_sp, seed=args.seed,
                         n_bootstrap=args.n_bootstrap, ci=args.ci, quick=args.quick)
    if args.experiment in (None, 2):
        run_experiment2(args.n_test_ids, seed=args.seed,
                         n_bootstrap=args.n_bootstrap, ci=args.ci, quick=args.quick)


if __name__ == "__main__":
    main()
