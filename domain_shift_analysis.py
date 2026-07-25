"""
Cross-domain / cross-dataset feature-space domain-shift analysis for palmprint
data, aimed at answering a specific reviewer question: "what makes the method
undergo a significant performance drop on your dataset [X-Palm]?" -- i.e. is
X-Palm's cross-domain difficulty quantifiably larger than the other
benchmarks', not just an impression.

The previous script in this pair (image_quality_analysis.py) addresses the
"is it just image quality?" question. This script addresses the complementary
"is there a genuine distributional/domain gap, and is it worse for X-Palm?"
question, using feature-space distance metrics (MMD, Proxy A-Distance,
Frechet Feature Distance) computed on CNN embeddings -- the standard toolkit
from the domain-adaptation / OOD-detection literature.

Four tables, in order of how directly they answer the reviewer:

  Table A - Between-Dataset Shift: treats each whole dataset as one
            distribution and computes pairwise shift across all 4 datasets.
            This is the headline comparison: are the X-Palm-involving pairs
            systematically larger than the other pairs?
  Table B - Within-Dataset Heterogeneity: how spread out a dataset's own
            captured sub-domains are from each other (pooled, all pairs).
            Supports "X-Palm is intrinsically a harder/more heterogeneous
            benchmark", independent of any other dataset.
  Table C - Cross-Sensor Shift: a like-for-like comparison, since X-Palm
            (scanner vs. smartphone) and MPDv2 (device h vs. device m) both
            have a genuine sensor-change axis, while XJTU-UP's devices give
            a second one -- lets you say "X-Palm's sensor-change shift is
            X times larger than MPDv2's", not just "shift exists".
  Table D - Hardest Sub-Domain Pairs: the single worst-case pairs pooled
            across every dataset, ranked once by MMD and then reported
            consistently across all three metrics (the original script's
            "Top 5" logic ranked each metric independently, which could
            silently report three different sets of pairs under one label
            -- fixed here).

Statistical fixes relative to a naive MMD/PAD/FID implementation, all
important given that per-sub-domain sample sizes here are small (tens to a
few hundred), not the tens-of-thousands typical generative-model FID is
designed for:

  - All embeddings are standardized then PCA-reduced (fit ONCE globally, on
    every image pooled together, so every downstream comparison shares the
    same coordinate system). This is not optional at these sample sizes:
    Frechet distance needs a well-conditioned covariance estimate, which
    requires N notably larger than the feature dimensionality -- 2048-D raw
    ResNet features with a few hundred images per sub-domain is exactly the
    regime where that breaks down.
  - Proxy A-Distance uses stratified k-fold cross-validated accuracy (not one
    train/test split) and a fixed regularization strength, plus a larger
    minimum-sample-size floor: with as few as ~10 samples in a high-D
    embedding space, a linear SVM will separate almost any two groups
    trivially, reporting spuriously maximal shift regardless of whether a
    real difference exists.
  - MMD uses the unbiased estimator (excludes self-similarity diagonal terms)
    with a data-adaptive median-heuristic kernel bandwidth (rather than
    sklearn's default gamma=1/n_features, which is arbitrary here), plus an
    optional label-permutation test for a significance flag.
  - FFD is computed on size-balanced random subsamples (both groups
    downsampled to the smaller group's N, repeated and averaged) so that
    differences in sample count across sub-domains -- which will differ a
    lot between datasets -- don't masquerade as differences in shift.

A caveat worth stating explicitly in a rebuttal: these metrics are computed
on generic ImageNet-pretrained features, which is standard practice but is a
domain-mismatched proxy for what your actual recognizer responds to. See
`EMBEDDING_SOURCE` below -- swapping in your trained recognition backbone's
own embeddings ties the shift measurement directly to the feature space that
determines matching performance, which is the more causally convincing
version of this analysis for a reviewer asking "why does performance drop".
"""

import os
import pickle
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from scipy.linalg import sqrtm
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# Configuration
# ==========================================
DATA_ROOTS = {
    "CASIA-MS": "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "MPDv2": "/home/pai-ng/Jamal/MPDv2_mediapipe_manual_roi",
    "XJTU-UP": "/home/pai-ng/Jamal/XJTU-UP",
    "X-Palm": "/home/pai-ng/Jamal/xpalm",
}

OUTPUT_DIR = "domain_shift_outputs"
FEATURE_CACHE = os.path.join(OUTPUT_DIR, "raw_features_cache.pkl")

EMBEDDING_SOURCE = "imagenet_resnet50"   # or "task_model" -- see extract_task_specific_feature()
PCA_DIM = 64                              # shared, globally-fit reduced dimensionality
MIN_SAMPLES_PAD = 20                      # PAD reliability floor (see module docstring)
MIN_SAMPLES_SHIFT = 10                    # general floor for MMD/FFD to even attempt a pair
N_PERMUTATIONS = 100                      # MMD permutation-test resamples (0 disables)
MAX_PERM_SAMPLES = 200                    # cap per-group size used inside the permutation test only
FFD_RESAMPLES = 10                        # size-balanced FFD resampling repeats
TOP_K_HARDEST = 5
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Feature extraction
# ==========================================
def load_imagenet_extractor():
    print(f"Loading ImageNet-pretrained ResNet-50 feature extractor on {device}...")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).to(device)
    model.fc = torch.nn.Identity()
    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(232),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extractor(path):
        img = cv2.imread(path)
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(tensor).cpu().numpy().flatten()

    return extractor


def extract_task_specific_feature():
    """
    Hook for the more causally convincing version of this analysis: swap in
    embeddings from YOUR trained palmprint recognition backbone instead of a
    generic ImageNet CNN. Generic features measure "does this look visually
    different"; your model's own penultimate-layer embeddings measure "does
    this look different in the exact space where matching decisions are
    made" -- much stronger evidence for a reviewer asking why performance
    drops specifically on X-Palm.

    Implement by loading your checkpoint, running the same forward pass used
    at inference time (minus the final classification/margin head), and
    returning a flat embedding vector per image path. Then set
    EMBEDDING_SOURCE = "task_model" above.
    """
    raise NotImplementedError(
        "Point this at your trained recognition model's embedding layer, "
        "then set EMBEDDING_SOURCE = 'task_model'."
    )


def get_extractor():
    if EMBEDDING_SOURCE == "task_model":
        return extract_task_specific_feature()
    return load_imagenet_extractor()


# ==========================================
# Dataset parsing (kept close to the original; adds SensorTag / ConditionTag
# so a "cross-sensor" comparison can be built generically for every dataset
# that actually has a sensor axis, rather than special-casing one dataset)
# ==========================================
def parse_casia_ms(data_root):
    records = []
    if not os.path.exists(data_root):
        return records
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg", ".png", ".bmp")):
            continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) < 3:
            continue
        records.append({
            "Dataset": "CASIA-MS", "SubDomain": f"Spectrum_{parts[2]}",
            "SensorTag": "MS-Sensor", "ConditionTag": f"Spectrum_{parts[2]}",
            "Path": os.path.join(data_root, fname),
        })
    return records


def parse_mpd_data(data_root):
    records = []
    if not os.path.exists(data_root):
        return records
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg", ".bmp", ".png")):
            continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) != 5:
            continue
        _, session, device_id, _, _ = parts
        if device_id not in ("h", "m"):
            continue
        records.append({
            "Dataset": "MPDv2", "SubDomain": f"Device_{device_id}",
            "SensorTag": f"Device_{device_id}", "ConditionTag": f"Session_{session}",
            "Path": os.path.join(data_root, fname),
        })
    return records


def parse_xjtu_domains(data_root):
    records = []
    if not os.path.exists(data_root):
        return records
    IMG_EXTS = {".jpg", ".bmp", ".png"}
    for dev in os.listdir(data_root):
        dev_dir = os.path.join(data_root, dev)
        if not os.path.isdir(dev_dir):
            continue
        for condition in os.listdir(dev_dir):
            cond_dir = os.path.join(dev_dir, condition)
            if not os.path.isdir(cond_dir):
                continue
            for id_folder in sorted(os.listdir(cond_dir)):
                id_dir = os.path.join(cond_dir, id_folder)
                if not os.path.isdir(id_dir):
                    continue
                for fname in sorted(os.listdir(id_dir)):
                    if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                        continue
                    records.append({
                        "Dataset": "XJTU-UP", "SubDomain": f"{dev}_{condition}",
                        "SensorTag": f"Device_{dev}", "ConditionTag": f"Condition_{condition}",
                        "Path": os.path.join(id_dir, fname),
                    })
    return records


def parse_xpalm(data_root):
    records = []
    unmatched = 0
    if not os.path.exists(data_root):
        return records, unmatched
    IMG_EXTS = {".jpg", ".png", ".bmp"}
    scanner_targets = ["pink", "green", "white", "ir", "blue", "yellow"]
    smartphone_targets = ["wet", "text", "jf", "sf", "bf", "close", "far", "pitch", "roll", "fl", "rnd"]

    scanner_dir = os.path.join(data_root, "scanner_roi")
    if os.path.isdir(scanner_dir):
        for subj in sorted(os.listdir(scanner_dir)):
            subj_dir = os.path.join(scanner_dir, subj)
            if not os.path.isdir(subj_dir):
                continue
            for fname in sorted(os.listdir(subj_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                    continue
                fname_lower = fname.lower()
                matched = next((t for t in scanner_targets if t in fname_lower), None)
                if matched:
                    records.append({
                        "Dataset": "X-Palm", "SubDomain": f"Scanner_{matched}",
                        "SensorTag": "Scanner", "ConditionTag": matched,
                        "Path": os.path.join(subj_dir, fname),
                    })
                else:
                    unmatched += 1

    phone_dir = os.path.join(data_root, "smartphone_roi")
    if os.path.isdir(phone_dir):
        for subj in sorted(os.listdir(phone_dir)):
            subj_dir = os.path.join(phone_dir, subj)
            if not os.path.isdir(subj_dir):
                continue
            for fname in sorted(os.listdir(subj_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                    continue
                fname_lower = fname.lower()
                matched = next((t for t in smartphone_targets if t in fname_lower), None)
                if matched:
                    records.append({
                        "Dataset": "X-Palm", "SubDomain": f"Smartphone_{matched}",
                        "SensorTag": "Smartphone", "ConditionTag": matched,
                        "Path": os.path.join(subj_dir, fname),
                    })
                else:
                    unmatched += 1

    return records, unmatched


def gather_all_records():
    all_records = []
    print("Parsing dataset structures...")
    for name, fn in (("CASIA-MS", parse_casia_ms), ("MPDv2", parse_mpd_data), ("XJTU-UP", parse_xjtu_domains)):
        recs = fn(DATA_ROOTS[name])
        print(f"  {name:10s}: {len(recs)} images parsed")
        all_records.extend(recs)
    xpalm_recs, xpalm_unmatched = parse_xpalm(DATA_ROOTS["X-Palm"])
    print(f"  {'X-Palm':10s}: {len(xpalm_recs)} images parsed "
          f"({xpalm_unmatched} file(s) matched no known target keyword and were skipped)")
    all_records.extend(xpalm_recs)
    return all_records


# ==========================================
# Feature extraction with disk caching (keyed by absolute path, so reruns
# after adding a few files don't recompute everything)
# ==========================================
def extract_all_features(records, extractor):
    cache = {}
    if os.path.exists(FEATURE_CACHE):
        with open(FEATURE_CACHE, "rb") as f:
            cache = pickle.load(f)

    updated = False
    for rec in tqdm(records, desc="Feature Extraction", unit="img"):
        path = rec["Path"]
        if path in cache:
            continue
        feat = extractor(path)
        cache[path] = feat
        updated = True

    if updated:
        with open(FEATURE_CACHE, "wb") as f:
            pickle.dump(cache, f)

    paths, feats, kept_records = [], [], []
    for rec in records:
        feat = cache.get(rec["Path"])
        if feat is None:
            continue
        paths.append(rec["Path"])
        feats.append(feat)
        kept_records.append(rec)
    return kept_records, np.vstack(feats)


# ==========================================
# Shared PCA pipeline (fit once, globally)
# ==========================================
def build_feature_pipeline(raw_features, pca_dim=PCA_DIM, random_state=RANDOM_STATE):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw_features)
    n_comp = max(1, min(pca_dim, scaled.shape[0] - 1, scaled.shape[1]))
    pca = PCA(n_components=n_comp, random_state=random_state)
    reduced = pca.fit_transform(scaled)
    print(f"PCA: reduced {raw_features.shape[1]}-D -> {n_comp}-D "
          f"(explained variance retained: {pca.explained_variance_ratio_.sum():.1%})")
    return reduced


# ==========================================
# Shift metrics
# ==========================================
def median_heuristic_gamma(X):
    d2 = euclidean_distances(X, X, squared=True)
    iu = np.triu_indices_from(d2, k=1)
    med = np.median(d2[iu]) if iu[0].size else 0.0
    return 1.0 / (2 * med) if med > 0 else 1.0


def _mmd2_unbiased(a, b, gamma):
    Kaa, Kbb, Kab = rbf_kernel(a, a, gamma), rbf_kernel(b, b, gamma), rbf_kernel(a, b, gamma)
    na, nb = len(a), len(b)
    sum_aa = (Kaa.sum() - np.trace(Kaa)) / (na * (na - 1))
    sum_bb = (Kbb.sum() - np.trace(Kbb)) / (nb * (nb - 1))
    return max(0.0, sum_aa + sum_bb - 2 * Kab.mean())


def compute_mmd(X_A, X_B, n_permutations=N_PERMUTATIONS, random_state=RANDOM_STATE):
    if len(X_A) < MIN_SAMPLES_SHIFT or len(X_B) < MIN_SAMPLES_SHIFT:
        return np.nan, np.nan
    X_full = np.vstack([X_A, X_B])
    gamma = median_heuristic_gamma(X_full)
    observed = _mmd2_unbiased(X_A, X_B, gamma)

    p_value = np.nan
    if n_permutations > 0:
        rng = np.random.default_rng(random_state)
        na = min(len(X_A), MAX_PERM_SAMPLES)
        nb = min(len(X_B), MAX_PERM_SAMPLES)
        a_sub = X_A[rng.choice(len(X_A), na, replace=False)]
        b_sub = X_B[rng.choice(len(X_B), nb, replace=False)]
        pooled = np.vstack([a_sub, b_sub])
        gamma_sub = median_heuristic_gamma(pooled)
        obs_sub = _mmd2_unbiased(a_sub, b_sub, gamma_sub)
        count = 0
        n_total = na + nb
        for _ in range(n_permutations):
            idx = rng.permutation(n_total)
            a_perm, b_perm = pooled[idx[:na]], pooled[idx[na:]]
            if _mmd2_unbiased(a_perm, b_perm, gamma_sub) >= obs_sub:
                count += 1
        p_value = (count + 1) / (n_permutations + 1)

    return observed, p_value


def compute_proxy_a_distance(X_A, X_B, min_samples=MIN_SAMPLES_PAD, random_state=RANDOM_STATE):
    if len(X_A) < min_samples or len(X_B) < min_samples:
        return np.nan
    X = np.vstack([X_A, X_B])
    y = np.hstack([np.zeros(len(X_A)), np.ones(len(X_B))])
    n_splits = min(5, len(X_A), len(X_B))
    if n_splits < 2:
        return np.nan
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = LinearSVC(random_state=random_state, max_iter=5000, dual=False, C=0.1)
    try:
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    except ValueError:
        return np.nan
    error = 1.0 - scores.mean()
    return max(0.0, 2 * (1 - 2 * error))


def _ffd_raw(X_A, X_B, eps=1e-6):
    mu_A, mu_B = np.mean(X_A, axis=0), np.mean(X_B, axis=0)
    sigma_A = np.cov(X_A, rowvar=False) + np.eye(X_A.shape[1]) * eps
    sigma_B = np.cov(X_B, rowvar=False) + np.eye(X_B.shape[1]) * eps
    diff = mu_A - mu_B
    covmean, _ = sqrtm(sigma_A.dot(sigma_B), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return max(0.0, float(diff.dot(diff) + np.trace(sigma_A + sigma_B - 2 * covmean)))


def compute_ffd(X_A, X_B, n_resamples=FFD_RESAMPLES, random_state=RANDOM_STATE):
    if len(X_A) < MIN_SAMPLES_SHIFT or len(X_B) < MIN_SAMPLES_SHIFT:
        return np.nan, 0
    n = min(len(X_A), len(X_B))
    rng = np.random.default_rng(random_state)
    vals = []
    for _ in range(n_resamples):
        idx_a = rng.choice(len(X_A), n, replace=False)
        idx_b = rng.choice(len(X_B), n, replace=False)
        vals.append(_ffd_raw(X_A[idx_a], X_B[idx_b]))
    return float(np.mean(vals)), n


def compute_all_metrics(X_A, X_B):
    mmd, mmd_p = compute_mmd(X_A, X_B)
    pad = compute_proxy_a_distance(X_A, X_B)
    ffd, ffd_n = compute_ffd(X_A, X_B)
    return {"MMD": mmd, "MMD_p": mmd_p, "PAD": pad, "FFD": ffd, "FFD_n": ffd_n,
            "N_A": len(X_A), "N_B": len(X_B)}


def format_metric(x, digits=3):
    return f"{x:.{digits}f}" if np.isfinite(x) else "N/A"


# ==========================================
# Hook: linking domain shift to recognition performance
# ==========================================
def correlate_shift_with_performance(pairwise_shift_df, recognition_results_df):
    """
    The tables in this script establish *that* a shift exists and *how it
    compares* across datasets -- they don't by themselves prove it's what
    hurts recognition. To make that link (the actual answer a reviewer wants
    to "what causes the drop"):

      1. Join each sub-domain-pair's MMD/PAD/FFD (this script's per-pair CSV)
         against the recognition performance drop (e.g. EER or Rank-1
         delta) observed when training on one side of the pair and testing
         on the other.
      2. Spearman-correlate shift magnitude against performance drop across
         all pairs, pooled across all four datasets. A positive, significant
         correlation is direct evidence that distributional shift (not just
         a labeling artifact) predicts where the method struggles.
      3. Check whether X-Palm's pairs sit above the fitted trend line, not
         just at higher absolute shift -- if X-Palm pairs show a *larger*
         performance drop than other datasets' pairs at the *same* shift
         magnitude, that points to something specific to X-Palm beyond
         generic distributional distance (e.g. its sensor/condition changes
         alter exactly the texture cues the recognizer relies on).
      4. Cross-reference against the companion quality-analysis script: if
         a sub-domain pair's shift is largely explained by an image-quality
         gap (see that script's per-image CSV), say so; if the shift
         persists in quality-matched subsets, that strengthens the
         "genuine domain gap" argument for this reviewer specifically.
    """
    raise NotImplementedError(
        "Plug in your recognition pipeline's per-pair performance numbers "
        "here; see the docstring for the recommended analysis steps."
    )


# ==========================================
# Main
# ==========================================
def main():
    records = gather_all_records()
    if not records:
        print("No images found. Please verify root paths.")
        return

    extractor = get_extractor()
    records, raw_features = extract_all_features(records, extractor)
    print(f"\nExtracted features for {len(records)} images "
          f"(raw dim={raw_features.shape[1]}).")

    reduced = build_feature_pipeline(raw_features)

    df_meta = pd.DataFrame(records)
    df_meta["_row"] = np.arange(len(df_meta))

    def feats_for(mask):
        return reduced[df_meta.loc[mask, "_row"].values]

    datasets = sorted(df_meta["Dataset"].unique())

    # ---------------------------------------------------------
    # TABLE A: Between-Dataset Shift (headline comparison)
    # ---------------------------------------------------------
    dataset_pool = {ds: feats_for(df_meta["Dataset"] == ds) for ds in datasets}
    rows_a = []
    for ds_a, ds_b in combinations(datasets, 2):
        m = compute_all_metrics(dataset_pool[ds_a], dataset_pool[ds_b])
        rows_a.append({"Dataset A": ds_a, "Dataset B": ds_b,
                        "N_A": m["N_A"], "N_B": m["N_B"],
                        "MMD": format_metric(m["MMD"]),
                        "MMD_p": format_metric(m["MMD_p"], 3),
                        "PAD": format_metric(m["PAD"]),
                        "FFD": format_metric(m["FFD"], 1)})
    table_a = pd.DataFrame(rows_a)
    print("\n" + "=" * 100)
    print("TABLE A: Between-Dataset Domain Shift (each dataset pooled as one distribution)")
    print("=" * 100)
    print(table_a.to_markdown(index=False))
    table_a.to_csv(os.path.join(OUTPUT_DIR, "tableA_between_dataset.csv"), index=False)

    # ---------------------------------------------------------
    # TABLE B: Within-Dataset Heterogeneity (all sub-domain pairs, flat)
    # + collect every pair record (tagged by dataset) for Table D
    # ---------------------------------------------------------
    rows_b, all_pair_records = [], []
    for ds in datasets:
        sub = df_meta[df_meta["Dataset"] == ds]
        subdomains = sorted(sub["SubDomain"].unique())
        if len(subdomains) < 2:
            continue
        sub_feats = {sd: feats_for((df_meta["Dataset"] == ds) & (df_meta["SubDomain"] == sd))
                     for sd in subdomains}
        mmd_list, pad_list, ffd_list = [], [], []
        for sd_a, sd_b in combinations(subdomains, 2):
            m = compute_all_metrics(sub_feats[sd_a], sub_feats[sd_b])
            all_pair_records.append({"Dataset": ds, "SubDomain_A": sd_a, "SubDomain_B": sd_b, **m})
            if np.isfinite(m["MMD"]):
                mmd_list.append(m["MMD"])
            if np.isfinite(m["PAD"]):
                pad_list.append(m["PAD"])
            if np.isfinite(m["FFD"]):
                ffd_list.append(m["FFD"])
        rows_b.append({
            "Dataset": ds, "N_SubDomains": len(subdomains), "Pairs_Evaluated": len(mmd_list),
            "Mean_MMD": format_metric(np.mean(mmd_list)) if mmd_list else "N/A",
            "Mean_PAD": format_metric(np.mean(pad_list)) if pad_list else "N/A",
            "Mean_FFD": format_metric(np.mean(ffd_list), 1) if ffd_list else "N/A",
        })
    table_b = pd.DataFrame(rows_b)
    print("\n" + "=" * 100)
    print("TABLE B: Within-Dataset Internal Heterogeneity (mean over all own sub-domain pairs)")
    print("=" * 100)
    print(table_b.to_markdown(index=False))
    table_b.to_csv(os.path.join(OUTPUT_DIR, "tableB_within_dataset.csv"), index=False)

    pair_df = pd.DataFrame(all_pair_records)
    pair_df.to_csv(os.path.join(OUTPUT_DIR, "all_subdomain_pairs.csv"), index=False)
    print(f"\nSaved every sub-domain pair's raw MMD/PAD/FFD to "
          f"{os.path.join(OUTPUT_DIR, 'all_subdomain_pairs.csv')} "
          "(join against recognition performance for the causal analysis).")

    # ---------------------------------------------------------
    # TABLE C: Cross-Sensor Shift (like-for-like across datasets that have
    # a genuine sensor-change axis)
    # ---------------------------------------------------------
    rows_c = []
    for ds in datasets:
        sub = df_meta[df_meta["Dataset"] == ds]
        sensor_tags = sorted(sub["SensorTag"].unique())
        if len(sensor_tags) < 2:
            continue
        sensor_feats = {st: feats_for((df_meta["Dataset"] == ds) & (df_meta["SensorTag"] == st))
                         for st in sensor_tags}
        for st_a, st_b in combinations(sensor_tags, 2):
            m = compute_all_metrics(sensor_feats[st_a], sensor_feats[st_b])
            rows_c.append({"Dataset": ds, "Sensor A": st_a, "Sensor B": st_b,
                           "N_A": m["N_A"], "N_B": m["N_B"],
                           "MMD": format_metric(m["MMD"]), "MMD_p": format_metric(m["MMD_p"], 3),
                           "PAD": format_metric(m["PAD"]), "FFD": format_metric(m["FFD"], 1)})
    table_c = pd.DataFrame(rows_c)
    print("\n" + "=" * 100)
    print("TABLE C: Cross-Sensor Shift within each dataset (like-for-like comparison)")
    print("=" * 100)
    if not table_c.empty:
        print(table_c.to_markdown(index=False))
        table_c.to_csv(os.path.join(OUTPUT_DIR, "tableC_cross_sensor.csv"), index=False)
    else:
        print("No dataset had >=2 distinct SensorTag groups with enough samples.")

    # ---------------------------------------------------------
    # TABLE D: Hardest sub-domain pairs, pooled across all datasets, ranked
    # ONCE by MMD (fixes the independent-per-metric-sort bug)
    # ---------------------------------------------------------
    if not pair_df.empty:
        ranked = pair_df.sort_values("MMD", ascending=False, na_position="last")
        table_d = ranked.head(TOP_K_HARDEST).copy()
        for col in ("MMD", "PAD", "FFD"):
            table_d[col] = table_d[col].map(lambda x: format_metric(x))
        print("\n" + "=" * 100)
        print(f"TABLE D: Top {TOP_K_HARDEST} Hardest Sub-Domain Pairs Overall (ranked once by MMD, "
              "same pairs reported for PAD/FFD)")
        print("=" * 100)
        print(table_d.to_markdown(index=False))
        table_d.to_csv(os.path.join(OUTPUT_DIR, "tableD_hardest_pairs.csv"), index=False)

    print("\nGuidance: to establish that this shift actually drives performance (not just that "
          "it exists), join all_subdomain_pairs.csv against your verification pipeline's "
          "per-pair results and use `correlate_shift_with_performance()` in this file as a "
          "starting point.")


if __name__ == "__main__":
    main()
