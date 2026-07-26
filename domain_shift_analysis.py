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

Tables, in order of how directly they answer the reviewer:

  Table A  - Between-Dataset Shift: treats each whole dataset as one
             distribution and computes pairwise shift across all 4 datasets.
             This is the headline comparison: are the X-Palm-involving pairs
             systematically larger than the other pairs? (Cross-dataset.)
  Table B  - Within-Dataset Heterogeneity: how spread out a dataset's own
             captured sub-domains are from each other, now reported with
             dispersion (not just the mean pairwise MMD/PAD/FFD) plus two
             cluster-separation indices -- Calinski-Harabasz and Silhouette
             -- computed once per dataset over ALL its sub-domains jointly,
             rather than by averaging pairwise comparisons (which favors
             datasets with more sub-domains simply because they contribute
             more pairs). We intentionally do NOT pair-count-match across
             datasets here: a dataset covering more distinct capture
             conditions is, by design, a more useful benchmark, and the raw
             pairwise mean is left as-is so that richer condition coverage
             is visible rather than normalized away. (Intra-dataset only.)
  Table B2 - Per-Identity Cross-Domain Variability: Table B answers "how
             different are this dataset's conditions from each other",
             which says nothing about whether a *given subject* stays
             recognizable across them. For every identity with samples in
             >=2 sub-domains, we compute a within-identity, one-way-ANOVA-
             style F-ratio: (between-domain scatter of that identity's own
             per-domain centroids) / (within-domain scatter of that
             identity's own samples around each centroid) -- an
             identity-level analog of the Calinski-Harabasz idea in Table B,
             but with "domain" as the grouping factor *within* one person's
             data instead of across the whole dataset. A high ratio means a
             typical subject's embedding moves a lot when the capture
             condition changes, relative to how much it naturally varies
             within a single condition -- the quantity most directly
             relevant to whether cross-domain matching will be hard for a
             typical enrolled subject. (Intra-dataset only.)
  Table C  - Cross-Sensor Shift: a like-for-like comparison, since X-Palm
             (scanner vs. smartphone) and MPDv2 (device h vs. device m) both
             have a genuine sensor-change axis, while XJTU-UP's devices give
             a second one -- lets you say "X-Palm's sensor-change shift is
             X times larger than MPDv2's", not just "shift exists".
             (Intra-dataset only.)
  Table D  - Leave-One-Condition-Out Sensitivity: targeted diagnostic for
             the two comparisons a reviewer is most likely to press on --
             X-Palm's Scanner subset vs. CASIA-MS, and X-Palm's Smartphone
             subset vs. XJTU-UP. We compute the baseline shift with all of
             X-Palm's relevant conditions pooled, then recompute with each
             condition removed in turn. A large drop when a single
             condition is excluded means that one acquisition condition --
             not the sensor/dataset comparison broadly -- is driving the
             apparent shift (this is the check motivated by an earlier
             finding that a single scanner illumination condition dominated
             the dataset's hardest sub-domain pairs). (Cross-dataset by
             construction -- this is a diagnostic FOR a Table A/C finding,
             not an addition to the intra-dataset metrics above.)
  Table E  - Hardest Sub-Domain Pairs Overall: the single worst-case pairs
             pooled across every dataset's own sub-domains, ranked once by
             MMD and reported consistently across all three metrics (the
             original script's "Top 5" logic ranked each metric
             independently, which could silently report three different
             sets of pairs under one label -- fixed here). This is what
             motivated Table D above. (Intra-dataset only.)

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
on generic, task-agnostic features (ResNet-50 by default; DINOv2 is also
available as a toggle -- see `EMBEDDING_SOURCE` below), which is standard
practice but is a domain-mismatched proxy for what your actual recognizer
responds to. Swapping in your trained recognition backbone's own embeddings
(`EMBEDDING_SOURCE = "task_model"`) ties the shift measurement directly to
the feature space that determines matching performance, which is the more
causally convincing version of this analysis for a reviewer asking "why
does performance drop".
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
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, euclidean_distances
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
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

EMBEDDING_SOURCE = "imagenet_resnet50"   # "imagenet_resnet50" | "dinov2" | "task_model"
DINOV2_MODEL_NAME = "dinov2_vits14"   # or dinov2_vitb14 / dinov2_vitl14 / dinov2_vitg14
                                        # (larger = slower + higher-dim; vits14=384-D,
                                        # vitb14=768-D, vitl14=1024-D, vitg14=1536-D)
                                        # only used when EMBEDDING_SOURCE == "dinov2"

PCA_DIM = 64                              # shared, globally-fit reduced dimensionality
MIN_SAMPLES_PAD = 20                      # PAD reliability floor (see module docstring)
MIN_SAMPLES_SHIFT = 10                    # general floor for MMD/FFD to even attempt a pair
N_PERMUTATIONS = 100                      # MMD permutation-test resamples (0 disables)
MAX_PERM_SAMPLES = 200                    # cap per-group size used inside the permutation test only
FFD_RESAMPLES = 10                        # size-balanced FFD resampling repeats
KID_DEGREE = 3                            # polynomial kernel degree for KID (Binkowski et al., 2018)
SWD_N_PROJECTIONS = 100                   # random projections for Sliced Wasserstein Distance
TOP_K_HARDEST = 5
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cache_path():
    """Feature cache is keyed by which embedding model produced it, not just
    by image path -- otherwise switching EMBEDDING_SOURCE (e.g. ResNet-50 ->
    DINOv2) and rerunning would silently reuse stale cached vectors from the
    OLD model for every path already in the cache, quietly mixing two
    incompatible embedding spaces into one "reduced" feature matrix with no
    error or warning. Every table downstream would still run; the numbers
    would just be wrong."""
    if EMBEDDING_SOURCE == "dinov2":
        tag = f"dinov2_{DINOV2_MODEL_NAME}"
    elif EMBEDDING_SOURCE == "task_model":
        tag = "task_model"
    else:
        tag = "imagenet_resnet50"
    return os.path.join(OUTPUT_DIR, f"raw_features_cache_{tag}.pkl")


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


def load_dinov2_extractor(model_name=DINOV2_MODEL_NAME):
    """
    DINOv2 (Oquab et al., 2023) self-supervised ViT features, loaded via
    torch.hub. Unlike the ImageNet-supervised ResNet-50 below, DINOv2 is
    trained with a self-supervised objective over a much larger and more
    visually diverse corpus and is widely reported to transfer better to
    content unlike typical ImageNet photos (close-up textures, documents,
    biometric imagery) -- plausibly a better-behaved embedding space for
    measuring domain shift on palmprint ROIs than a supervised natural-photo
    classifier. It is still a generic, task-agnostic embedding, not a
    substitute for the task-specific hook in extract_task_specific_feature().

    Requires internet access on first call (weights are fetched via
    torch.hub and cached under ~/.cache/torch/hub, or $TORCH_HOME if set).
    Input must be resized to a multiple of the model's 14x14 patch size;
    224 (=16x14) is the standard choice used here. Calling the loaded model
    directly (backbone(tensor)) returns the pooled per-image embedding,
    matching the flat feature-vector-per-image contract the rest of this
    script expects -- no extra pooling/indexing needed.
    """
    print(f"Loading DINOv2 ({model_name}) feature extractor on {device}...")
    try:
        backbone = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    except TypeError:
        # older torch versions don't accept trust_repo
        backbone = torch.hub.load("facebookresearch/dinov2", model_name)
    backbone = backbone.to(device)
    backbone.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
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
            return backbone(tensor).cpu().numpy().flatten()

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
    if EMBEDDING_SOURCE == "dinov2":
        return load_dinov2_extractor()
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
            "ID": f"{parts[0]}_{parts[1]}",
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
        subject, session, device_id, hand_side, _ = parts
        if device_id not in ("h", "m"):
            continue
        records.append({
            "Dataset": "MPDv2", "SubDomain": f"Device_{device_id}",
            "SensorTag": f"Device_{device_id}", "ConditionTag": f"Session_{session}",
            "ID": f"{subject}_{hand_side}",
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
                    # NOTE: assumes id_folder naming (e.g. "L_01") is used
                    # consistently for the same physical subject across every
                    # device/condition subfolder. If your capture protocol
                    # does not guarantee that, Table B2 below will undercount
                    # or mis-pair identities for this dataset.
                    records.append({
                        "Dataset": "XJTU-UP", "SubDomain": f"{dev}_{condition}",
                        "SensorTag": f"Device_{dev}", "ConditionTag": f"Condition_{condition}",
                        "ID": id_folder,
                        "Path": os.path.join(id_dir, fname),
                    })
    return records


import re

_HAND_TOKEN_PATTERN = re.compile(r"(?:^|_)(l|r|left|right|lh|rh)(?:_|$)", re.IGNORECASE)


def extract_hand(fname_no_ext):
    """Best-effort extraction of hand laterality from an X-Palm filename.
    X-Palm's true identity unit is subject+hand (a person's two palms are
    different biometric patterns), but scanner_roi/smartphone_roi only
    separate by SUBJECT folder -- both hands' images live in the same
    folder, so laterality must be recovered from the filename itself.

    *** ADAPT THE TOKEN LIST ABOVE (_HAND_TOKEN_PATTERN) to your actual
    naming convention before trusting this. *** Verify first, e.g.:
        ls scanner_roi/<some_subject_folder>/ | head -20
        ls smartphone_roi/<some_subject_folder>/ | head -20
    and check that "l"/"r" (or whatever token you actually use) appears as
    its own underscore-delimited token in those filenames.

    Returns 'L', 'R', or None if no recognized token was found. A None
    result falls back to subject-only identity for that one file (the old
    behavior) and is counted by the caller, so a systematic mismatch
    between this pattern and your real filenames shows up as a loud
    fallback-rate warning instead of silently mis-grouping identities.
    """
    m = _HAND_TOKEN_PATTERN.search(fname_no_ext.lower())
    if not m:
        return None
    token = m.group(1)
    return "R" if token in ("r", "right", "rh") else "L"


def parse_xpalm(data_root):
    records = []
    unmatched = 0
    no_hand_detected = 0
    if not os.path.exists(data_root):
        return records, unmatched, no_hand_detected
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
                    hand = extract_hand(os.path.splitext(fname)[0])
                    if hand is None:
                        no_hand_detected += 1
                    subject_id = f"{subj}_{hand}" if hand else subj
                    records.append({
                        "Dataset": "X-Palm", "SubDomain": f"Scanner_{matched}",
                        "SensorTag": "Scanner", "ConditionTag": matched,
                        "ID": subject_id,
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
                    hand = extract_hand(os.path.splitext(fname)[0])
                    if hand is None:
                        no_hand_detected += 1
                    subject_id = f"{subj}_{hand}" if hand else subj
                    records.append({
                        "Dataset": "X-Palm", "SubDomain": f"Smartphone_{matched}",
                        "SensorTag": "Smartphone", "ConditionTag": matched,
                        "ID": subject_id,
                        "Path": os.path.join(subj_dir, fname),
                    })
                else:
                    unmatched += 1

    return records, unmatched, no_hand_detected


def gather_all_records():
    all_records = []
    print("Parsing dataset structures...")
    for name, fn in (("CASIA-MS", parse_casia_ms), ("MPDv2", parse_mpd_data), ("XJTU-UP", parse_xjtu_domains)):
        recs = fn(DATA_ROOTS[name])
        print(f"  {name:10s}: {len(recs)} images parsed")
        all_records.extend(recs)
    xpalm_recs, xpalm_unmatched, xpalm_no_hand = parse_xpalm(DATA_ROOTS["X-Palm"])
    print(f"  {'X-Palm':10s}: {len(xpalm_recs)} images parsed "
          f"({xpalm_unmatched} file(s) matched no known target keyword and were skipped)")
    if xpalm_recs:
        no_hand_rate = xpalm_no_hand / len(xpalm_recs)
        if no_hand_rate > 0.02:
            print(f"  WARNING: hand laterality could not be detected for "
                  f"{xpalm_no_hand}/{len(xpalm_recs)} X-Palm files ({no_hand_rate:.1%}) -- "
                  f"these fell back to subject-only ID. _HAND_TOKEN_PATTERN in parse_xpalm() "
                  f"almost certainly does not match your actual filename convention; inspect a "
                  f"few real filenames and adjust it before trusting Table B2's X-Palm rows.")
        else:
            print(f"  X-Palm hand detection: {xpalm_no_hand}/{len(xpalm_recs)} files fell back "
                  f"to subject-only ID.")
    all_records.extend(xpalm_recs)
    return all_records


# ==========================================
# Feature extraction with disk caching (keyed by absolute path, so reruns
# after adding a few files don't recompute everything)
# ==========================================
def extract_all_features(records, extractor):
    cache_path = get_cache_path()
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
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
        with open(cache_path, "wb") as f:
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
    """Returns (PAD, raw_error). PAD = 2(1-2*error) saturates near its
    ceiling of 2 once error gets close to 0, which compresses real
    differences between highly-separable sub-domain pairs into a narrow
    band (see Table B discussion). The raw cross-validated error doesn't
    have that compression -- report it alongside PAD, not as a replacement
    for it."""
    if len(X_A) < min_samples or len(X_B) < min_samples:
        return np.nan, np.nan
    X = np.vstack([X_A, X_B])
    y = np.hstack([np.zeros(len(X_A)), np.ones(len(X_B))])
    n_splits = min(5, len(X_A), len(X_B))
    if n_splits < 2:
        return np.nan, np.nan
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = LinearSVC(random_state=random_state, max_iter=5000, dual=False, C=0.1)
    try:
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    except ValueError:
        return np.nan, np.nan
    error = 1.0 - scores.mean()
    pad = max(0.0, 2 * (1 - 2 * error))
    return pad, float(error)


def _ffd_components(X_A, X_B, eps=1e-6):
    """Splits FFD into its two additive terms, plus CORAL, so a large FFD
    can be attributed to a genuine cause rather than left ambiguous:
      - mean_term: ||mu_A - mu_B||^2 -- do the sub-domains differ in
        average appearance?
      - cov_term: Tr(Sigma_A + Sigma_B - 2*sqrtm(Sigma_A @ Sigma_B)) -- the
        Frechet covariance-mismatch term (FFD = mean_term + cov_term).
      - coral: ||Sigma_A - Sigma_B||_F^2 (Sun, Feng & Saenko, 2016) -- a
        simpler, non-Frechet covariance-mismatch score, useful as a
        cross-check on cov_term since it needs no matrix square root.
    """
    mu_A, mu_B = np.mean(X_A, axis=0), np.mean(X_B, axis=0)
    sigma_A = np.cov(X_A, rowvar=False) + np.eye(X_A.shape[1]) * eps
    sigma_B = np.cov(X_B, rowvar=False) + np.eye(X_B.shape[1]) * eps
    mean_term = float(np.sum((mu_A - mu_B) ** 2))
    covmean, _ = sqrtm(sigma_A.dot(sigma_B), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    cov_term = max(0.0, float(np.trace(sigma_A + sigma_B - 2 * covmean)))
    coral = float(np.sum((sigma_A - sigma_B) ** 2))
    return mean_term, cov_term, coral


def _ffd_raw(X_A, X_B, eps=1e-6):
    mean_term, cov_term, _ = _ffd_components(X_A, X_B, eps)
    return max(0.0, mean_term + cov_term)


def compute_ffd(X_A, X_B, n_resamples=FFD_RESAMPLES, random_state=RANDOM_STATE):
    """Returns (FFD, n, mean_term, cov_term, CORAL), all averaged over the
    same size-balanced resamples used for FFD itself, so the decomposition
    is directly comparable to the headline FFD number."""
    if len(X_A) < MIN_SAMPLES_SHIFT or len(X_B) < MIN_SAMPLES_SHIFT:
        return np.nan, 0, np.nan, np.nan, np.nan
    n = min(len(X_A), len(X_B))
    rng = np.random.default_rng(random_state)
    ffd_vals, mean_vals, cov_vals, coral_vals = [], [], [], []
    for _ in range(n_resamples):
        idx_a = rng.choice(len(X_A), n, replace=False)
        idx_b = rng.choice(len(X_B), n, replace=False)
        mean_term, cov_term, coral = _ffd_components(X_A[idx_a], X_B[idx_b])
        ffd_vals.append(max(0.0, mean_term + cov_term))
        mean_vals.append(mean_term)
        cov_vals.append(cov_term)
        coral_vals.append(coral)
    return (float(np.mean(ffd_vals)), n, float(np.mean(mean_vals)),
            float(np.mean(cov_vals)), float(np.mean(coral_vals)))


def compute_kid(X_A, X_B, degree=KID_DEGREE, coef0=1.0):
    """Kernel Inception Distance (Binkowski et al., 2018): MMD^2 with a
    polynomial kernel instead of an RBF kernel. Purpose-built as a
    lower-bias, lower-variance alternative to Frechet-style distances in
    the small-sample regime (FID/FFD assume tens of thousands of samples;
    our sub-domains have tens to a few hundred) -- a cross-check on FFD
    that doesn't share its Gaussian-covariance assumption or its small-N
    bias."""
    if len(X_A) < MIN_SAMPLES_SHIFT or len(X_B) < MIN_SAMPLES_SHIFT:
        return np.nan
    gamma = 1.0 / X_A.shape[1]
    Kaa = polynomial_kernel(X_A, X_A, degree=degree, gamma=gamma, coef0=coef0)
    Kbb = polynomial_kernel(X_B, X_B, degree=degree, gamma=gamma, coef0=coef0)
    Kab = polynomial_kernel(X_A, X_B, degree=degree, gamma=gamma, coef0=coef0)
    na, nb = len(X_A), len(X_B)
    sum_aa = (Kaa.sum() - np.trace(Kaa)) / (na * (na - 1))
    sum_bb = (Kbb.sum() - np.trace(Kbb)) / (nb * (nb - 1))
    return max(0.0, float(sum_aa + sum_bb - 2 * Kab.mean()))


def compute_swd(X_A, X_B, n_projections=SWD_N_PROJECTIONS, random_state=RANDOM_STATE):
    """Sliced Wasserstein Distance: approximates the (generally intractable
    in >1D) Wasserstein-1 distance by averaging the exact closed-form 1-D
    Wasserstein distance over many random projections. Makes NO Gaussian
    assumption about either sub-domain, unlike FFD -- if SWD and FFD agree
    in ranking, that corroborates FFD's Gaussian assumption; if they
    diverge, FFD's magnitude may partly reflect non-Gaussian sub-domain
    shape rather than genuine separation."""
    if len(X_A) < MIN_SAMPLES_SHIFT or len(X_B) < MIN_SAMPLES_SHIFT:
        return np.nan
    rng = np.random.default_rng(random_state)
    d = X_A.shape[1]
    dists = []
    for _ in range(n_projections):
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction) + 1e-12
        proj_a = np.sort(X_A @ direction)
        proj_b = np.sort(X_B @ direction)
        n = min(len(proj_a), len(proj_b))
        qa = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(proj_a)), proj_a)
        qb = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(proj_b)), proj_b)
        dists.append(np.mean(np.abs(qa - qb)))
    return float(np.mean(dists))


def compute_nn_same_domain_fraction(X_A, X_B, max_samples=MAX_PERM_SAMPLES, random_state=RANDOM_STATE):
    """Friedman-Rafsky (1979) 1-nearest-neighbor two-sample statistic: pool
    both sub-domains, and for every point check whether its nearest OTHER
    point came from the same sub-domain. The fraction that do is a
    distribution-free separability score -- no kernel bandwidth, no
    Gaussian assumption, no classifier training -- a third, methodologically
    independent check alongside MMD and PAD. Near 0.5 under the null
    (identical distributions); near 1.0 means the sub-domains barely
    overlap in feature space."""
    if len(X_A) < MIN_SAMPLES_SHIFT or len(X_B) < MIN_SAMPLES_SHIFT:
        return np.nan
    rng = np.random.default_rng(random_state)
    a = X_A if len(X_A) <= max_samples else X_A[rng.choice(len(X_A), max_samples, replace=False)]
    b = X_B if len(X_B) <= max_samples else X_B[rng.choice(len(X_B), max_samples, replace=False)]
    X = np.vstack([a, b])
    labels = np.array([0] * len(a) + [1] * len(b))
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    _, indices = nn.kneighbors(X)
    nearest_other = indices[:, 1]
    return float((labels[nearest_other] == labels).mean())


def compute_all_metrics(X_A, X_B):
    mmd, mmd_p = compute_mmd(X_A, X_B)
    pad, pad_error = compute_proxy_a_distance(X_A, X_B)
    ffd, ffd_n, ffd_mean_term, ffd_cov_term, coral = compute_ffd(X_A, X_B)
    kid = compute_kid(X_A, X_B)
    swd = compute_swd(X_A, X_B)
    nn_frac = compute_nn_same_domain_fraction(X_A, X_B)
    return {
        "MMD": mmd, "MMD_p": mmd_p,
        "PAD": pad, "PAD_Error": pad_error,
        "FFD": ffd, "FFD_n": ffd_n,
        "FFD_MeanTerm": ffd_mean_term, "FFD_CovTerm": ffd_cov_term, "CORAL": coral,
        "KID": kid, "SWD": swd, "NN_SameDomainFrac": nn_frac,
        "N_A": len(X_A), "N_B": len(X_B),
    }


def format_metric(x, digits=3):
    return f"{x:.{digits}f}" if np.isfinite(x) else "N/A"


# ==========================================
# Intra-dataset heterogeneity metrics (Table B / Table B2)
# ==========================================
def safe_cluster_scores(features, labels, random_state=RANDOM_STATE,
                         silhouette_sample_cap=5000):
    """Calinski-Harabasz and Silhouette scores for a dataset's sub-domain
    grouping, computed ONCE over the whole dataset (all sub-domains
    jointly) rather than by averaging pairwise comparisons. This avoids the
    pairwise-mean's implicit bias toward datasets with more sub-domains
    (more sub-domains -> more pairs -> more chances for the mean to include
    an extreme pair) -- both indices are defined directly over an arbitrary
    number of groups, so they aren't in the sub-domain count.

    CH is unbounded (higher = more separated); Silhouette is bounded in
    [-1, 1] and is comparable in scale across datasets regardless of how
    many sub-domains each has, which the raw mean-pairwise metrics are not.
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(features) <= len(unique_labels):
        return np.nan, np.nan

    try:
        ch = calinski_harabasz_score(features, labels)
    except Exception:
        ch = np.nan

    try:
        if len(features) > silhouette_sample_cap:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(features), silhouette_sample_cap, replace=False)
            sil = silhouette_score(features[idx], labels[idx])
        else:
            sil = silhouette_score(features, labels)
    except Exception:
        sil = np.nan

    return ch, sil


def per_id_domain_variability(df_meta, reduced, dataset, domain_col="SubDomain",
                               min_domains=2, min_within_samples=2):
    """Identity-level analog of the Calinski-Harabasz ratio in Table B, but
    computed WITHIN each identity's own samples, with sub-domain as the
    grouping factor, instead of across the whole dataset.

    For identity i with samples spanning domains d in D_i:
      - domain centroid  mu_{i,d} = mean of i's samples captured in d
      - grand mean       mu_i     = mean of ALL of i's samples in this dataset
      - MS_between = [sum_d n_{i,d} * ||mu_{i,d} - mu_i||^2] / (|D_i| - 1)
      - MS_within  = [sum_d sum_{x in d} ||x - mu_{i,d}||^2] / (sum_d (n_{i,d}-1))
                      (only over domains with >= min_within_samples)
      - F_ratio    = MS_between / MS_within

    F_ratio is this identity's own "domain-driven displacement" relative to
    its own natural sample-to-sample noise within one condition -- directly
    relevant to whether cross-domain matching will be hard for a *typical*
    subject, which dataset-level heterogeneity (Table B) cannot tell you.
    Requires the "ID" field added by the dataset parsers.
    """
    if "ID" not in df_meta.columns:
        return pd.DataFrame(columns=["ID", "N_Domains", "MS_Between", "MS_Within", "F_Ratio"])

    sub = df_meta[df_meta["Dataset"] == dataset]
    results = []
    for id_val, grp in sub.groupby("ID"):
        domains = grp[domain_col].unique()
        if len(domains) < min_domains:
            continue

        feats_by_domain = {}
        for d in domains:
            rows = grp.loc[grp[domain_col] == d, "_row"].values
            feats_by_domain[d] = reduced[rows]

        all_feats = np.vstack(list(feats_by_domain.values()))
        grand_mean = all_feats.mean(axis=0)

        ss_between, df_between = 0.0, len(domains) - 1
        ss_within, n_within_df = 0.0, 0
        for f in feats_by_domain.values():
            centroid = f.mean(axis=0)
            n_d = len(f)
            ss_between += n_d * np.sum((centroid - grand_mean) ** 2)
            if n_d >= min_within_samples:
                ss_within += np.sum((f - centroid) ** 2)
                n_within_df += (n_d - 1)

        ms_between = ss_between / df_between if df_between > 0 else np.nan
        ms_within = ss_within / n_within_df if n_within_df > 0 else np.nan
        f_ratio = (ms_between / ms_within
                   if (n_within_df > 0 and np.isfinite(ms_within) and ms_within > 0)
                   else np.nan)

        results.append({"ID": id_val, "N_Domains": len(domains),
                         "MS_Between": ms_between, "MS_Within": ms_within,
                         "F_Ratio": f_ratio})

    return pd.DataFrame(results)


# ==========================================
# Leave-one-condition-out diagnostic (Table D)
# ==========================================
def leave_one_condition_out(df_meta, feats_for_fn, group_dataset, group_sensor_tag,
                             other_dataset, condition_col="ConditionTag"):
    """Baseline: pool ALL of `group_dataset`'s sub-domains under
    `group_sensor_tag` (e.g. every X-Palm scanner illumination) against the
    whole of `other_dataset`, then remove one condition at a time and
    recompute. A large drop in a metric when a single condition is excluded
    means that one acquisition condition -- not the sensor/dataset
    comparison broadly -- is driving the apparent shift.
    """
    mask_group_all = (df_meta["Dataset"] == group_dataset) & (df_meta["SensorTag"] == group_sensor_tag)
    mask_other = df_meta["Dataset"] == other_dataset
    conditions = sorted(df_meta.loc[mask_group_all, condition_col].unique())

    other_feats = feats_for_fn(mask_other)
    baseline = compute_all_metrics(feats_for_fn(mask_group_all), other_feats)
    rows = [{"Condition_Removed": "(none -- all conditions)", "N_Group": baseline["N_A"], **baseline}]

    for c in conditions:
        mask_c = mask_group_all & (df_meta[condition_col] != c)
        feats_c = feats_for_fn(mask_c)
        if len(feats_c) < MIN_SAMPLES_SHIFT:
            continue
        m = compute_all_metrics(feats_c, other_feats)
        rows.append({"Condition_Removed": f"-{c}", "N_Group": m["N_A"], **m})

    out = pd.DataFrame(rows)
    base_mmd = out.loc[out["Condition_Removed"] == "(none -- all conditions)", "MMD"].values[0]
    out["Delta_MMD_vs_Baseline"] = base_mmd - out["MMD"]
    out.attrs["group_dataset"] = group_dataset
    out.attrs["group_sensor_tag"] = group_sensor_tag
    out.attrs["other_dataset"] = other_dataset
    return out.sort_values("Delta_MMD_vs_Baseline", ascending=False, na_position="last")


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
    # TABLE B: Within-Dataset Heterogeneity -- mean AND dispersion over all
    # own sub-domain pairs, plus dataset-level Calinski-Harabasz and
    # Silhouette scores (computed once over all sub-domains jointly, so they
    # don't share the pairwise mean's implicit bias toward datasets with
    # more sub-domains). We deliberately do NOT pair-count-match across
    # datasets: more sub-domains reflects richer condition coverage, which
    # is a benchmark strength, not a statistical artifact to normalize away.
    # + collect every pair record (tagged by dataset) for Table E
    # ---------------------------------------------------------
    rows_b, all_pair_records = [], []
    for ds in datasets:
        sub = df_meta[df_meta["Dataset"] == ds]
        subdomains = sorted(sub["SubDomain"].unique())
        if len(subdomains) < 2:
            continue
        sub_feats = {sd: feats_for((df_meta["Dataset"] == ds) & (df_meta["SubDomain"] == sd))
                     for sd in subdomains}
        metric_lists = {k: [] for k in (
            "MMD", "PAD", "PAD_Error", "FFD", "FFD_MeanTerm", "FFD_CovTerm",
            "CORAL", "KID", "SWD", "NN_SameDomainFrac")}
        for sd_a, sd_b in combinations(subdomains, 2):
            m = compute_all_metrics(sub_feats[sd_a], sub_feats[sd_b])
            all_pair_records.append({"Dataset": ds, "SubDomain_A": sd_a, "SubDomain_B": sd_b, **m})
            for key in metric_lists:
                if np.isfinite(m[key]):
                    metric_lists[key].append(m[key])

        ch_index, sil_score = safe_cluster_scores(feats_for(df_meta["Dataset"] == ds),
                                                    sub["SubDomain"].values)

        def _mean_std(key, digits=3):
            vals = metric_lists[key]
            if not vals:
                return "N/A", "N/A"
            return format_metric(np.mean(vals), digits), format_metric(np.std(vals), digits)

        mean_mmd, std_mmd = _mean_std("MMD")
        mean_pad, std_pad = _mean_std("PAD")
        mean_pad_err, std_pad_err = _mean_std("PAD_Error")
        mean_ffd, std_ffd = _mean_std("FFD", 1)
        mean_ffd_mean, std_ffd_mean = _mean_std("FFD_MeanTerm", 1)
        mean_ffd_cov, std_ffd_cov = _mean_std("FFD_CovTerm", 1)
        mean_coral, std_coral = _mean_std("CORAL", 1)
        mean_kid, std_kid = _mean_std("KID")
        mean_swd, std_swd = _mean_std("SWD")
        mean_nn, std_nn = _mean_std("NN_SameDomainFrac")

        rows_b.append({
            "Dataset": ds, "N_SubDomains": len(subdomains),
            "Pairs_Evaluated": len(metric_lists["MMD"]),
            "Mean_MMD": mean_mmd, "Std_MMD": std_mmd,
            "Mean_PAD": mean_pad, "Std_PAD": std_pad,
            "Mean_PAD_Error": mean_pad_err, "Std_PAD_Error": std_pad_err,
            "Mean_FFD": mean_ffd, "Std_FFD": std_ffd,
            "Mean_FFD_MeanTerm": mean_ffd_mean, "Std_FFD_MeanTerm": std_ffd_mean,
            "Mean_FFD_CovTerm": mean_ffd_cov, "Std_FFD_CovTerm": std_ffd_cov,
            "Mean_CORAL": mean_coral, "Std_CORAL": std_coral,
            "Mean_KID": mean_kid, "Std_KID": std_kid,
            "Mean_SWD": mean_swd, "Std_SWD": std_swd,
            "Mean_NN_SameDomainFrac": mean_nn, "Std_NN_SameDomainFrac": std_nn,
            "Calinski_Harabasz": format_metric(ch_index, 1),
            "Silhouette": format_metric(sil_score),
        })
    table_b = pd.DataFrame(rows_b)
    print("\n" + "=" * 100)
    print("TABLE B: Within-Dataset Internal Heterogeneity "
          "(pairwise mean +/- std, and whole-dataset cluster-separation indices)")
    print("=" * 100)
    print(table_b.to_markdown(index=False))
    table_b.to_csv(os.path.join(OUTPUT_DIR, "tableB_within_dataset.csv"), index=False)
    print("Note: Calinski-Harabasz is unbounded (higher = more separated); Silhouette is "
          "bounded in [-1, 1] and is the more directly cross-dataset-comparable of the two "
          "since it does not scale with sub-domain count the way the pairwise means can.\n"
          "New metrics: KID (Binkowski et al. 2018) is a lower-bias small-sample alternative "
          "to FFD; SWD makes no Gaussian assumption, unlike FFD, and is a check on that "
          "assumption; NN_SameDomainFrac (Friedman & Rafsky 1979) is a kernel-free, "
          "classifier-free separability check (~0.5 = fully overlapping, ~1.0 = non-overlapping); "
          "FFD_MeanTerm/FFD_CovTerm/CORAL decompose FFD into 'different average appearance' vs. "
          "'different spread/noisiness' so a large FFD can be attributed to a specific cause; "
          "PAD_Error is the raw cross-validated classifier error PAD is computed from, without "
          "PAD's ceiling-saturating 2(1-2*error) transform.")

    pair_df = pd.DataFrame(all_pair_records)
    pair_df.to_csv(os.path.join(OUTPUT_DIR, "all_subdomain_pairs.csv"), index=False)
    print(f"\nSaved every sub-domain pair's raw MMD/PAD/FFD to "
          f"{os.path.join(OUTPUT_DIR, 'all_subdomain_pairs.csv')} "
          "(join against recognition performance for the causal analysis).")

    # ---------------------------------------------------------
    # TABLE B2: Per-Identity Cross-Domain Variability (intra-dataset only).
    # Complements Table B: instead of "how separated are this dataset's
    # conditions", this asks "how much does a typical enrolled subject's own
    # embedding move between conditions, relative to their own natural
    # within-condition noise" -- see per_id_domain_variability() docstring.
    # ---------------------------------------------------------
    rows_b2 = []
    all_id_records = []
    for ds in datasets:
        id_df = per_id_domain_variability(df_meta, reduced, ds)
        n_total_ids = df_meta.loc[df_meta["Dataset"] == ds, "ID"].nunique() if "ID" in df_meta.columns else 0
        valid = id_df.dropna(subset=["F_Ratio"]) if not id_df.empty else id_df
        if not id_df.empty:
            id_df = id_df.copy()
            id_df.insert(0, "Dataset", ds)
            all_id_records.append(id_df)

        rows_b2.append({
            "Dataset": ds,
            "N_IDs_Total": n_total_ids,
            "N_IDs_Multi_Domain": len(id_df),
            "N_IDs_With_Valid_Ratio": len(valid),
            "Mean_MS_Between": format_metric(id_df["MS_Between"].mean()) if not id_df.empty else "N/A",
            "Mean_MS_Within": format_metric(id_df["MS_Within"].mean()) if not id_df.empty else "N/A",
            "Mean_F_Ratio": format_metric(valid["F_Ratio"].mean()) if not valid.empty else "N/A",
            "Median_F_Ratio": format_metric(valid["F_Ratio"].median()) if not valid.empty else "N/A",
            "Std_F_Ratio": format_metric(valid["F_Ratio"].std()) if not valid.empty else "N/A",
        })
    table_b2 = pd.DataFrame(rows_b2)
    print("\n" + "=" * 100)
    print("TABLE B2: Per-Identity Cross-Domain Variability "
          "(within-identity, between-domain-scatter / within-domain-scatter F-ratio, "
          "averaged over identities with samples in >=2 sub-domains)")
    print("=" * 100)
    print(table_b2.to_markdown(index=False))
    table_b2.to_csv(os.path.join(OUTPUT_DIR, "tableB2_per_identity.csv"), index=False)
    print("Note: mean F-ratio can be pulled up by a small number of identities with an "
          "unusually small within-domain denominator -- the median is the more robust "
          "summary; per-identity raw values are saved separately below.")

    if all_id_records:
        id_raw_df = pd.concat(all_id_records, ignore_index=True)
        id_raw_df.to_csv(os.path.join(OUTPUT_DIR, "tableB2_per_id_raw.csv"), index=False)
        print(f"Saved every identity's own F-ratio to "
              f"{os.path.join(OUTPUT_DIR, 'tableB2_per_id_raw.csv')}.")

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
    # TABLE D: Leave-one-condition-out sensitivity for the two comparisons a
    # reviewer is most likely to press on: X-Palm Scanner vs. CASIA-MS, and
    # X-Palm Smartphone vs. XJTU-UP. Tests whether a single acquisition
    # condition is driving the apparent shift rather than the sensor/dataset
    # change broadly.
    # ---------------------------------------------------------
    loo_configs = [
        ("X-Palm", "Scanner", "CASIA-MS", "TABLE D1: Leave-One-Out -- X-Palm Scanner vs. CASIA-MS"),
        ("X-Palm", "Smartphone", "XJTU-UP", "TABLE D2: Leave-One-Out -- X-Palm Smartphone vs. XJTU-UP"),
    ]
    for group_ds, sensor_tag, other_ds, title in loo_configs:
        if group_ds not in datasets or other_ds not in datasets:
            print(f"\nSkipping {title}: one of the two datasets was not found.")
            continue
        loo_table = leave_one_condition_out(df_meta, feats_for, group_ds, sensor_tag, other_ds)
        display = loo_table.copy()
        for col in ("MMD", "MMD_p", "PAD", "PAD_Error", "FFD", "FFD_MeanTerm", "FFD_CovTerm",
                    "CORAL", "KID", "SWD", "NN_SameDomainFrac", "Delta_MMD_vs_Baseline"):
            display[col] = display[col].map(lambda x: format_metric(x, 3))
        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
        print(display.to_markdown(index=False))
        fname = f"tableD_loo_{group_ds}_{sensor_tag}_vs_{other_ds}.csv".replace(" ", "")
        loo_table.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
        top_condition = loo_table.iloc[0]
        if top_condition["Condition_Removed"] != "(none -- all conditions)":
            print(f"Largest single-condition contribution: removing "
                  f"'{top_condition['Condition_Removed'].lstrip('-')}' drops MMD by "
                  f"{format_metric(top_condition['Delta_MMD_vs_Baseline'], 3)} "
                  f"relative to the all-conditions baseline.")

    # ---------------------------------------------------------
    # TABLE E: Hardest sub-domain pairs, pooled across all datasets' own
    # sub-domains, ranked ONCE by MMD (fixes the independent-per-metric-sort
    # bug in an earlier version of this table). This is what motivated the
    # Table D leave-one-out check above.
    # ---------------------------------------------------------
    if not pair_df.empty:
        ranked = pair_df.sort_values("MMD", ascending=False, na_position="last")
        table_e = ranked.head(TOP_K_HARDEST).copy()
        for col in ("MMD", "PAD", "FFD"):
            table_e[col] = table_e[col].map(lambda x: format_metric(x))
        print("\n" + "=" * 100)
        print(f"TABLE E: Top {TOP_K_HARDEST} Hardest Sub-Domain Pairs Overall (ranked once by MMD, "
              "same pairs reported for PAD/FFD)")
        print("=" * 100)
        print(table_e.to_markdown(index=False))
        table_e.to_csv(os.path.join(OUTPUT_DIR, "tableE_hardest_pairs.csv"), index=False)

    print("\nGuidance: to establish that this shift actually drives performance (not just that "
          "it exists), join all_subdomain_pairs.csv against your verification pipeline's "
          "per-pair results and use `correlate_shift_with_performance()` in this file as a "
          "starting point.")


if __name__ == "__main__":
    main()
