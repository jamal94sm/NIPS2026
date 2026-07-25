"""
Cross-domain palmprint image quality analysis.

Goal: give a reviewer-facing answer to "is cross-domain recognition failure
caused by image quality degradation, or a genuinely novel biometric domain
gap?" This requires more than a table of mean IQA scores per dataset, so the
script is organized around three questions:

  1. Are the datasets even comparable in the first place? (capture
     characteristics: resolution, aspect ratio, color vs. grayscale, format)
  2. Do the datasets differ in quality, and is that difference statistically
     significant / practically large? (native-scale AND standardized-scale
     metrics, non-parametric tests, effect sizes)
  3. Does quality actually explain recognition performance differences, or
     is there a residual domain gap after controlling for quality? (hook for
     joining with your verification pipeline's scores -- see
     `analyze_quality_vs_performance` at the bottom)

Design choices worth calling out explicitly (useful to cite in a rebuttal):

  - Metrics are computed at BOTH native resolution (as captured by the
    sensor) and at the standardized size fed to the recognizer (default
    112x112). Native-scale tells you about sensor/capture quality; eval-scale
    tells you what the recognizer actually "sees". Comparing the two shows
    how much of a raw sensor-quality gap survives the pipeline's own
    preprocessing.
  - All quality metrics are computed on grayscale intensity, deliberately.
    Some source datasets are captured in color and others as grayscale scans;
    evaluating everyone in the same (grayscale) modality removes color model
    as a confound in the comparison. Native color/mono composition is still
    recorded per dataset so this choice is transparent, not hidden.
  - NIQE/BRISQUE are natural-photograph NR-IQA models -- a domain mismatch
    for close-up skin/ridge textures. They're kept as they are established
    reference metrics, but supplemented with:
      * a Gabor-bank "ridge energy" score, since palmprint matchers
        (CompCode/PalmCode-style) are themselves banks of Gabor filter
        responses -- this is a quality proxy aligned with the recognition
        mechanism, unlike generic photo-IQA.
      * MUSIQ (optional, loaded defensively) as a more modern learned NR-IQA
        cross-check against NIQE/BRISQUE.
  - Downsampling uses INTER_AREA (not the previous default bilinear) to
    avoid aliasing that would inconsistently distort sharpness/noise metrics
    depending on each dataset's native resolution.
"""

import os
import warnings
import numpy as np
import pandas as pd
import cv2
import torch
import pyiqa
from tqdm import tqdm
from scipy import stats

# ==========================================
# Configuration
# ==========================================
DATA_ROOTS = {
    "CASIA-MS": "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "MPDv2": "/home/pai-ng/Jamal/MPDv2_mediapipe_manual_roi",
    "XJTU-UP": "/home/pai-ng/Jamal/XJTU-UP",
    "X-Palm": "/home/pai-ng/Jamal/xpalm",
}

EVAL_SIZE = 112              # standardized size fed to the recognizer
RESIZE_MODE = "stretch"      # "stretch" matches typical recognizer preprocessing;
                              # "pad" preserves aspect ratio -- switch to compare
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
USE_DEEP_METRIC = True        # try to also load MUSIQ (optional, more robust off-domain)
OUTPUT_DIR = "quality_analysis_outputs"

DATASET_ORDER = ["XJTU-UP", "MPDv2", "CASIA-MS", "X-Palm (scanner)", "X-Palm (smartphone)"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# pyiqa model loading (defensive: optional deep metric may not be cached)
# ==========================================
def init_pyiqa_metrics(use_deep_metric=True):
    print(f"Loading IQA models on {device}...")
    metrics = {}
    for name in ("niqe", "brisque"):
        try:
            metrics[name.upper()] = pyiqa.create_metric(name).to(device)
        except Exception as e:
            warnings.warn(f"Could not load required pyiqa metric '{name}': {e}")
    if use_deep_metric:
        try:
            metrics["MUSIQ"] = pyiqa.create_metric("musiq").to(device)
        except Exception as e:
            warnings.warn(f"Optional pyiqa metric 'musiq' unavailable, skipping: {e}")
    return metrics


def compute_pyiqa_metrics(gray_eval_img, metrics):
    """pyiqa expects an RGB tensor; we feed the grayscale-derived image
    (duplicated across channels) to keep modality consistent across datasets
    -- see module docstring for rationale."""
    img_rgb = cv2.cvtColor(gray_eval_img, cv2.COLOR_GRAY2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)
    out = {}
    with torch.no_grad():
        for name, model in metrics.items():
            try:
                out[name] = model(tensor).item()
            except Exception:
                out[name] = np.nan
    return out


# ==========================================
# Handcrafted quality metrics (scale-agnostic: usable at native or eval size)
# ==========================================
_GABOR_BANK = None


def _get_gabor_bank(ksize=15, sigma=4.0, lambd=8.0, gamma=0.5, n_orientations=6):
    global _GABOR_BANK
    if _GABOR_BANK is None:
        _GABOR_BANK = [
            cv2.getGaborKernel((ksize, ksize), sigma, i * np.pi / n_orientations,
                                lambd, gamma, 0, ktype=cv2.CV_64F)
            for i in range(n_orientations)
        ]
    return _GABOR_BANK


def gabor_ridge_energy(gray_img):
    """Mean max-orientation Gabor response: a palmprint-relevant proxy for
    ridge/line clarity, aligned with how Gabor-code matchers actually read
    the image (unlike generic natural-photo IQA)."""
    gray_f = gray_img.astype(np.float32) / 255.0
    responses = np.stack(
        [cv2.filter2D(gray_f, cv2.CV_64F, k) for k in _get_gabor_bank()], axis=0
    )
    return float(np.max(np.abs(responses), axis=0).mean())


def shannon_entropy(gray_img):
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def illumination_nonuniformity(gray_img):
    """Std of block-mean brightness across a coarse grid: higher = more
    uneven lighting (a classic scanner-vs-phone / cross-domain confound)."""
    h, w = gray_img.shape
    bs = max(8, min(h, w) // 8)
    means = [
        gray_img[y:y + bs, x:x + bs].mean()
        for y in range(0, h - bs + 1, bs)
        for x in range(0, w - bs + 1, bs)
    ]
    return float(np.std(means)) if len(means) > 1 else 0.0


def handcrafted_metrics(gray_img):
    gray_f = gray_img.astype(np.float64)

    laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()

    gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx ** 2 + gy ** 2))

    rms_contrast = float(gray_img.std())

    edges = cv2.Canny(gray_img, 50, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    mse = np.mean((gray_f - blurred.astype(np.float64)) ** 2)
    pseudo_psnr = 100.0 if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))

    return {
        "Laplacian_Var": laplacian_var,
        "Tenengrad": tenengrad,
        "RMS_Contrast": rms_contrast,
        "Entropy": shannon_entropy(gray_img),
        "Edge_Density": edge_density,
        "Pseudo_PSNR": pseudo_psnr,
        "Illum_Nonuniformity": illumination_nonuniformity(gray_img),
        "Ridge_Energy": gabor_ridge_energy(gray_img),
    }


# ==========================================
# Image I/O + resizing (fairness-relevant choices explained inline)
# ==========================================
def read_image_with_native_info(path):
    """Reads with IMREAD_UNCHANGED so we can honestly report native channel
    count / bit depth (cv2.imread's default flag silently forces 3-channel
    color, which would hide whether a source was actually grayscale)."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, None

    native_channels = 1 if raw.ndim == 2 else raw.shape[2]
    native_h, native_w = raw.shape[:2]
    meta = {
        "Native_Width": native_w,
        "Native_Height": native_h,
        "Native_Channels": native_channels,
        "Native_Dtype": str(raw.dtype),
        "Native_AspectRatio": native_w / native_h if native_h else np.nan,
    }

    img = raw
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        gray = img
    elif img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray, meta


def resize_for_eval(gray_img, size=EVAL_SIZE, mode=RESIZE_MODE):
    h, w = gray_img.shape[:2]
    interp = cv2.INTER_AREA if (w > size or h > size) else cv2.INTER_LINEAR

    if mode == "stretch":
        return cv2.resize(gray_img, (size, size), interpolation=interp)

    # "pad": preserve aspect ratio, letterbox onto a square canvas
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(gray_img, (new_w, new_h), interpolation=interp)
    canvas = np.full((size, size), int(resized.mean()), dtype=np.uint8)
    y0, x0 = (size - new_h) // 2, (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def compute_metrics_for_image(path, pyiqa_metrics):
    gray_native, meta = read_image_with_native_info(path)
    if gray_native is None:
        return None

    gray_eval = resize_for_eval(gray_native)

    row = dict(meta)
    for k, v in handcrafted_metrics(gray_native).items():
        row[f"Native_{k}"] = v
    for k, v in handcrafted_metrics(gray_eval).items():
        row[f"Eval_{k}"] = v
    for k, v in compute_pyiqa_metrics(gray_eval, pyiqa_metrics).items():
        row[f"Eval_{k}"] = v

    return row


# ==========================================
# Dataset parsing (naming-pattern logic unchanged from the original script;
# added: a completeness check so silently-skipped files are visible)
# ==========================================
def _count_images_on_disk(root):
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                total += 1
    return total


def parse_casia_ms(data_root):
    records = []
    if not os.path.exists(data_root):
        return records
    for fname in sorted(os.listdir(data_root)):
        if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
            continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) < 4:
            continue
        identity = f"{parts[0]}_{parts[1]}"
        records.append({"Dataset": "CASIA-MS", "Subset": "All", "ID": identity,
                         "Path": os.path.join(data_root, fname)})
    return records


def parse_mpd_data(data_root):
    records = []
    if not os.path.exists(data_root):
        return records
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".bmp", ".png")):
            continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) != 5:
            continue
        subject, session, device_id, hand_side, iteration = parts
        if device_id not in ("h", "m") or hand_side not in ("l", "r"):
            continue
        identity = f"{subject}_{hand_side}"
        records.append({"Dataset": "MPDv2", "Subset": "All", "ID": identity,
                         "Path": os.path.join(data_root, fname)})
    return records


def parse_xjtu_domains(data_root):
    records = []
    if not os.path.exists(data_root):
        return records
    for device in os.listdir(data_root):
        dev_dir = os.path.join(data_root, device)
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
                parts = id_folder.split("_")
                if len(parts) < 2 or parts[0].upper() not in ("L", "R"):
                    continue
                for fname in sorted(os.listdir(id_dir)):
                    if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                        continue
                    records.append({"Dataset": "XJTU-UP", "Subset": "All", "ID": id_folder,
                                     "Path": os.path.join(id_dir, fname)})
    return records


def parse_xpalm(data_root):
    records = []
    if not os.path.exists(data_root):
        return records

    scanner_dir = os.path.join(data_root, "scanner_roi")
    if os.path.isdir(scanner_dir):
        for subj_folder in sorted(os.listdir(scanner_dir)):
            subj_dir = os.path.join(scanner_dir, subj_folder)
            if not os.path.isdir(subj_dir):
                continue
            for fname in sorted(os.listdir(subj_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                    continue
                parts = os.path.splitext(fname)[0].split("_")
                if len(parts) < 4:
                    continue
                records.append({"Dataset": "X-Palm (scanner)", "Subset": "scanner",
                                 "ID": subj_folder, "Path": os.path.join(subj_dir, fname)})

    phone_dir = os.path.join(data_root, "smartphone_roi")
    if os.path.isdir(phone_dir):
        for subj_folder in sorted(os.listdir(phone_dir)):
            subj_dir = os.path.join(phone_dir, subj_folder)
            if not os.path.isdir(subj_dir):
                continue
            for fname in sorted(os.listdir(subj_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                    continue
                records.append({"Dataset": "X-Palm (smartphone)", "Subset": "smartphone",
                                 "ID": subj_folder, "Path": os.path.join(subj_dir, fname)})
    return records


def gather_all_records():
    parsers = {
        "CASIA-MS": (parse_casia_ms, DATA_ROOTS["CASIA-MS"]),
        "MPDv2": (parse_mpd_data, DATA_ROOTS["MPDv2"]),
        "XJTU-UP": (parse_xjtu_domains, DATA_ROOTS["XJTU-UP"]),
        "X-Palm": (parse_xpalm, DATA_ROOTS["X-Palm"]),
    }
    all_records = []
    print("\nDataset parsing / completeness check:")
    for name, (fn, root) in parsers.items():
        records = fn(root)
        on_disk = _count_images_on_disk(root) if os.path.exists(root) else 0
        rate = (len(records) / on_disk * 100) if on_disk else 0.0
        flag = "" if on_disk == 0 or rate >= 95 else "  <-- check naming assumptions"
        print(f"  {name:10s}: {len(records):6d} parsed / {on_disk:6d} image files on disk "
              f"({rate:5.1f}%){flag}")
        all_records.extend(records)
    return all_records


# ==========================================
# Statistics helpers
# ==========================================
def holm_bonferroni(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvals[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def cliffs_delta(u_stat, n1, n2):
    return (2 * u_stat) / (n1 * n2) - 1


def kruskal_wallis_table(df, metrics, group_col="Dataset"):
    rows = []
    groups = [g for g in df[group_col].unique()]
    for m in metrics:
        samples = [df.loc[df[group_col] == g, m].dropna().values for g in groups]
        samples = [s for s in samples if len(s) > 0]
        if len(samples) < 2:
            continue
        h_stat, p_val = stats.kruskal(*samples)
        rows.append({"Metric": m, "H": h_stat, "p_value": p_val, "n_groups": len(samples)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm_bonferroni(out["p_value"].values)
        out["significant_(p<0.05)"] = out["p_holm"] < 0.05
    return out


def pairwise_mannwhitney(df, metrics, group_col, group_a, group_b):
    rows = []
    for m in metrics:
        x = df.loc[df[group_col] == group_a, m].dropna().values
        y = df.loc[df[group_col] == group_b, m].dropna().values
        if len(x) < 2 or len(y) < 2:
            continue
        u_stat, p_val = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append({
            "Metric": m, f"{group_a}_median": np.median(x), f"{group_b}_median": np.median(y),
            "p_value": p_val, "cliffs_delta": cliffs_delta(u_stat, len(x), len(y)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm_bonferroni(out["p_value"].values)
    return out


def paired_wilcoxon(df, metrics, id_col, group_col, group_a, group_b):
    """Paired test for X-Palm scanner vs. smartphone: same subject, two
    sensors -- the cleanest identity-controlled comparison available here."""
    wide = df.pivot_table(index=id_col, columns=group_col, values=metrics, aggfunc="mean")
    rows = []
    for m in metrics:
        if (m, group_a) not in wide.columns or (m, group_b) not in wide.columns:
            continue
        paired = wide[[(m, group_a), (m, group_b)]].dropna()
        if len(paired) < 2:
            continue
        a, b = paired[(m, group_a)].values, paired[(m, group_b)].values
        try:
            w_stat, p_val = stats.wilcoxon(a, b)
        except ValueError:
            continue
        rows.append({
            "Metric": m, "n_pairs": len(paired),
            f"median_{group_a}": np.median(a), f"median_{group_b}": np.median(b),
            "median_paired_diff": float(np.median(a - b)), "p_value": p_val,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm_bonferroni(out["p_value"].values)
    return out


def format_mean_std(x):
    return f"{x.mean():.2f} ± {x.std():.2f}"


def summarize(df, group_col, metrics):
    table = df.groupby(group_col)[metrics].agg(format_mean_std).T
    cols = [c for c in DATASET_ORDER if c in table.columns]
    return table[cols] if cols else table


# ==========================================
# Hook: linking quality to recognition performance
# ==========================================
def analyze_quality_vs_performance(quality_df, recognition_scores_df,
                                    id_col="ID", dataset_col="Dataset"):
    """
    This is the piece that actually answers the reviewer's question -- a
    quality table by itself cannot. `recognition_scores_df` should come from
    your verification pipeline and contain, at minimum, a joinable identity/
    pair key plus a per-sample outcome (match score, or correct/incorrect at
    your operating threshold).

    Recommended steps once real recognition scores are available:
      1. Join mean per-image (or per-verification-pair) quality metrics onto
         the recognition results.
      2. Spearman correlation between each quality metric and the outcome,
         computed within each dataset and pooled -- shows how much
         performance variance quality alone explains.
      3. Fit outcome ~ quality_metrics + C(dataset) (logistic if outcome is
         correct/incorrect, linear/OLS if it's a continuous match score) and
         test whether the dataset term stays significant after controlling
         for quality. A significant residual dataset effect supports a
         genuine domain gap; a non-significant one supports quality as the
         main driver.
      4. Quality-matched subsampling: bin samples into quality quantiles
         per dataset, then compare cross-domain accuracy within the same
         quality bin. A performance gap that persists even in the top
         quality bin is the cleanest evidence of a genuine domain gap.
    """
    raise NotImplementedError(
        "Plug in your verification pipeline's per-pair scores here; see the "
        "docstring for the recommended analysis steps."
    )


# ==========================================
# Main
# ==========================================
def main():
    all_records = gather_all_records()
    if not all_records:
        print("No images found. Please check dataset root paths.")
        return

    pyiqa_metrics = init_pyiqa_metrics(USE_DEEP_METRIC)

    print(f"\nTotal images to process: {len(all_records)}")
    results, n_failed = [], 0
    for record in tqdm(all_records, desc="Extracting Quality Metrics", unit="img"):
        metrics = compute_metrics_for_image(record["Path"], pyiqa_metrics)
        if metrics is None:
            n_failed += 1
            continue
        record.update(metrics)
        results.append(record)
    if n_failed:
        print(f"WARNING: {n_failed} image(s) failed to load and were skipped.")

    df = pd.DataFrame(results)
    raw_csv = os.path.join(OUTPUT_DIR, "per_image_quality_metrics.csv")
    df.to_csv(raw_csv, index=False)
    print(f"\nSaved per-image metrics to {raw_csv} "
          f"(join this against recognition scores for the causal analysis).")

    eval_metrics = [c for c in df.columns if c.startswith("Eval_")]
    native_handcrafted = [c for c in df.columns if c.startswith("Native_")
                           and c not in ("Native_Width", "Native_Height",
                                         "Native_Channels", "Native_Dtype", "Native_AspectRatio")]

    # ---------------------------------------------------------
    # TABLE 0: Capture / sensor characteristics per dataset
    # ---------------------------------------------------------
    cap_cols = ["Native_Width", "Native_Height", "Native_AspectRatio"]
    table0 = df.groupby("Dataset")[cap_cols].agg(format_mean_std).T
    mono_pct = df.groupby("Dataset")["Native_Channels"].apply(lambda s: (s == 1).mean() * 100)
    table0.loc["Pct_Native_Grayscale"] = mono_pct
    counts = df.groupby("Dataset").size()
    table0.loc["N_Images"] = counts
    cols = [c for c in DATASET_ORDER if c in table0.columns]
    table0 = table0[cols] if cols else table0

    print("\n" + "=" * 80)
    print("TABLE 0: Capture Characteristics by Dataset (native resolution, before any resize)")
    print("=" * 80)
    print(table0.to_markdown())
    table0.to_csv(os.path.join(OUTPUT_DIR, "table0_capture_characteristics.csv"))

    # ---------------------------------------------------------
    # TABLE 1a: Global comparison, pooled per-image (standardized eval size)
    # ---------------------------------------------------------
    table1a = summarize(df, "Dataset", eval_metrics)
    print("\n" + "=" * 80)
    print(f"TABLE 1a: Image Quality by Dataset, pooled per-image (standardized {EVAL_SIZE}x{EVAL_SIZE})")
    print("=" * 80)
    print(table1a.to_markdown())
    table1a.to_csv(os.path.join(OUTPUT_DIR, "table1a_global_pooled.csv"))

    # ---------------------------------------------------------
    # TABLE 1b: Same, but identity-hierarchical (guards against subject imbalance)
    # ---------------------------------------------------------
    per_identity = df.groupby(["Dataset", "ID"])[eval_metrics].mean().reset_index()
    table1b = summarize(per_identity, "Dataset", eval_metrics)
    print("\n" + "=" * 80)
    print("TABLE 1b: Same metrics, averaged per identity first (checks subject-count imbalance)")
    print("=" * 80)
    print(table1b.to_markdown())
    table1b.to_csv(os.path.join(OUTPUT_DIR, "table1b_identity_hierarchical.csv"))

    # ---------------------------------------------------------
    # TABLE 1c: Native vs. standardized scale (handcrafted metrics only)
    # ---------------------------------------------------------
    handcrafted_names = [c.replace("Native_", "") for c in native_handcrafted]
    rows = []
    for name in handcrafted_names:
        native_col, eval_col = f"Native_{name}", f"Eval_{name}"
        if native_col not in df.columns or eval_col not in df.columns:
            continue
        for ds in df["Dataset"].unique():
            sub = df[df["Dataset"] == ds]
            rows.append({"Metric": name, "Dataset": ds,
                         "Native": sub[native_col].mean(), "Eval": sub[eval_col].mean()})
    table1c = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("TABLE 1c: Native-resolution vs. standardized-eval-size means "
          "(shows how much the resize pipeline compresses away sensor-level differences)")
    print("=" * 80)
    print(table1c.to_markdown(index=False))
    table1c.to_csv(os.path.join(OUTPUT_DIR, "table1c_native_vs_eval.csv"), index=False)

    # ---------------------------------------------------------
    # TABLE 2: X-Palm subject-level breakdown (scanner vs. smartphone, paired)
    # ---------------------------------------------------------
    xpalm_df = df[df["Dataset"].str.contains("X-Palm")]
    if not xpalm_df.empty:
        table2 = xpalm_df.groupby(["ID", "Subset"])[eval_metrics].agg(format_mean_std).unstack(level="Subset")
        table2.columns = [f"{metric}_{subset}" for metric, subset in table2.columns]
        paired_cols = []
        for m in eval_metrics:
            if f"{m}_scanner" in table2.columns:
                paired_cols.append(f"{m}_scanner")
            if f"{m}_smartphone" in table2.columns:
                paired_cols.append(f"{m}_smartphone")
        table2 = table2[paired_cols]

        print("\n" + "=" * 100)
        print("TABLE 2: X-Palm Quality by Subject ID (Scanner vs Smartphone)")
        print("=" * 100)
        print(table2.to_markdown())
        table2.to_csv(os.path.join(OUTPUT_DIR, "table2_xpalm_by_subject.csv"))

        # Paired Wilcoxon: same subjects, sensor is the only thing that changed
        wilcoxon_table = paired_wilcoxon(xpalm_df, eval_metrics, "ID", "Subset", "scanner", "smartphone")
        if not wilcoxon_table.empty:
            print("\nX-Palm paired Wilcoxon signed-rank test (scanner vs. smartphone, same subjects):")
            print(wilcoxon_table.to_markdown(index=False))
            wilcoxon_table.to_csv(os.path.join(OUTPUT_DIR, "table2b_xpalm_wilcoxon.csv"), index=False)
    else:
        print("\nNo X-Palm data found to generate Table 2.")

    # ---------------------------------------------------------
    # TABLE 3: Cross-dataset significance testing
    # ---------------------------------------------------------
    kw_table = kruskal_wallis_table(df, eval_metrics, "Dataset")
    print("\n" + "=" * 80)
    print("TABLE 3: Kruskal-Wallis test across all datasets per metric (Holm-corrected)")
    print("=" * 80)
    if not kw_table.empty:
        print(kw_table.to_markdown(index=False))
        kw_table.to_csv(os.path.join(OUTPUT_DIR, "table3_kruskal_wallis.csv"), index=False)

    print("\nGuidance: to actually answer the reviewer's causal question, join "
          f"{raw_csv} against your verification pipeline's per-pair scores and "
          "use `analyze_quality_vs_performance()` in this file as a starting point.")


if __name__ == "__main__":
    main()
