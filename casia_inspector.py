

import os
from collections import Counter

DATA_ROOT  = "/home/pai-ng/Jamal/smartphone_data"
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}

def count_images(folder):
    if not os.path.isdir(folder):
        return None          # folder does not exist
    return sum(1 for f in os.listdir(folder)
               if os.path.splitext(f)[1].lower() in IMG_EXTS)

persp_counts   = []
scanner_counts = []
rows           = []

for subject_id in sorted(os.listdir(DATA_ROOT), key=lambda x: int(x) if x.isdigit() else float("inf")):
    subject_dir = os.path.join(DATA_ROOT, subject_id)
    if not os.path.isdir(subject_dir):
        continue

    n_persp   = count_images(os.path.join(subject_dir, "roi_perspective"))
    n_scanner = count_images(os.path.join(subject_dir, "roi_scanner"))

    persp_str   = str(n_persp)   if n_persp   is not None else "—"
    scanner_str = str(n_scanner) if n_scanner is not None else "—"

    rows.append((subject_id, persp_str, scanner_str))
    if n_persp   is not None: persp_counts.append(n_persp)
    if n_scanner is not None: scanner_counts.append(n_scanner)

# ── Print table ───────────────────────────────────────────────
print(f"\n{'ID':<8} {'roi_perspective':>16} {'roi_scanner':>12}")
print("-" * 40)
for sid, p, s in rows:
    print(f"{sid:<8} {p:>16} {s:>12}")

# ── Summary ───────────────────────────────────────────────────
n_has_persp   = len(persp_counts)
n_has_scanner = len(scanner_counts)
total_ids     = len(rows)

print(f"\n{'='*40}")
print(f"  Total subject folders : {total_ids}")
print(f"\n  roi_perspective")
print(f"    IDs with folder     : {n_has_persp}")
if persp_counts:
    print(f"    Min / Max / Mean    : {min(persp_counts)} / {max(persp_counts)} / {sum(persp_counts)/n_has_persp:.1f}")
    print(f"    Total images        : {sum(persp_counts)}")
    dist = Counter(persp_counts)
    print(f"    Distribution        : { {k: dist[k] for k in sorted(dist)} }")

print(f"\n  roi_scanner")
print(f"    IDs with folder     : {n_has_scanner}")
if scanner_counts:
    print(f"    Min / Max / Mean    : {min(scanner_counts)} / {max(scanner_counts)} / {sum(scanner_counts)/n_has_scanner:.1f}")
    print(f"    Total images        : {sum(scanner_counts)}")
    dist = Counter(scanner_counts)
    print(f"    Distribution        : { {k: dist[k] for k in sorted(dist)} }")

print(f"{'='*40}\n")




'''
# ── MPDv2 samples-per-ID diagnostic ──────────────────────────
import os, math
from collections import defaultdict

data_root = "/home/pai-ng/Jamal/MPDv2_mediapipe_manual_roi"

id_dev = defaultdict(lambda: defaultdict(list))
for fname in sorted(os.listdir(data_root)):
    if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")):
        continue
    stem  = os.path.splitext(fname)[0]
    parts = stem.split("_")
    if len(parts) != 5:
        continue
    subject, session, device, hand_side, iteration = parts
    if device not in ("h","m") or hand_side not in ("l","r"):
        continue
    identity = subject + "_" + hand_side
    id_dev[identity][device].append(fname)

print(f"Total IDs found : {len(id_dev)}\n")

# Per-ID summary
counts_total = []
counts_h     = []
counts_m     = []
for ident, devs in sorted(id_dev.items()):
    nh = len(devs.get("h", []))
    nm = len(devs.get("m", []))
    counts_h.append(nh)
    counts_m.append(nm)
    counts_total.append(nh + nm)

print(f"{'Metric':<25} {'h':>6} {'m':>6} {'total':>6}")
print("-" * 45)
print(f"{'Min':<25} {min(counts_h):>6} {min(counts_m):>6} {min(counts_total):>6}")
print(f"{'Max':<25} {max(counts_h):>6} {max(counts_m):>6} {max(counts_total):>6}")
print(f"{'Mean':<25} {sum(counts_h)/len(counts_h):>6.1f} "
      f"{sum(counts_m)/len(counts_m):>6.1f} "
      f"{sum(counts_total)/len(counts_total):>6.1f}")
print()

# Distribution of total samples per ID
from collections import Counter
dist = Counter(counts_total)
print("Distribution of total samples per ID:")
for n in sorted(dist):
    print(f"  {n:>3} samples : {dist[n]:>4} IDs")

# Eligibility breakdown (7+8 or 8+7 threshold)
def qualifies(devs):
    h = len(devs.get("h", [])); m = len(devs.get("m", []))
    return (h >= 8 and m >= 7) or (h >= 7 and m >= 8)

eligible = {k: v for k, v in id_dev.items() if qualifies(v)}
print(f"\nEligible IDs (≥7 on one device, ≥8 on other): "
      f"{len(eligible)} / {len(id_dev)}")
'''



########################################################################################################




'''

# ── Set this to your dataset path ────────────────────────────────────────────
DATA_ROOT = "/home/pai-ng/Jamal/CASIA-MS-ROI"
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import time
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ═══════════════════════════════════════════════════════════════════════════════
#  1. Parse all filenames
# ═══════════════════════════════════════════════════════════════════════════════

def parse(fname):
    """
    {subjectID}_{side}_{spectrum}_{iteration}.jpg
    e.g.  001_L_460_01.jpg  →  ('001', 'L', '460', '01')
    Left (L) and Right (R) hands are treated as separate identities.
    Returns None if the filename doesn't match.
    """
    name  = os.path.splitext(fname)[0]
    parts = name.split('_')
    if len(parts) < 4:
        return None
    iteration = parts[-1]
    spectrum  = parts[-2]
    side      = parts[-3]
    subject   = '_'.join(parts[:-3])
    return subject, side, spectrum, iteration


records = []   # list of dicts, one per image file
skipped = []   # files that couldn't be parsed

for fname in sorted(os.listdir(DATA_ROOT)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    parsed = parse(fname)
    if parsed is None:
        skipped.append(fname)
        continue
    subject, side, spectrum, iteration = parsed
    records.append({
        'filename' : fname,
        'subject'  : subject,
        'side'     : side,
        'identity' : f"{subject}_{side}",   # L and R → distinct identities
        'spectrum' : spectrum,
        'iteration': iteration,
    })

df = pd.DataFrame(records)

print(f"{'═'*60}")
print(f"  CASIA-MS-ROI  —  Dataset Inspection")
print(f"  (Left hand / Right hand treated as separate identities)")
print(f"{'═'*60}")
print(f"  Total image files parsed : {len(df):,}")
print(f"  Skipped (bad filename)   : {len(skipped)}")
print(f"  Unique subjects          : {df['subject'].nunique()}")
print(f"  Unique identities        : {df['identity'].nunique()}  "
      f"(subject × side, L/R separate)")
print(f"  Unique spectra           : {df['spectrum'].nunique()}  "
      f"→  {sorted(df['spectrum'].unique())}")
print(f"  Unique iterations        : {sorted(df['iteration'].unique())}")
print()

if skipped:
    print(f"  ⚠  Skipped files (first 10): {skipped[:10]}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Images per identity  (L and R are distinct)
# ═══════════════════════════════════════════════════════════════════════════════

per_identity = (df.groupby('identity')
                  .size()
                  .rename('n_images')
                  .sort_values(ascending=False))

time.sleep(10)
print(f"{'─'*60}")
print(f"  Images per identity  (total {len(per_identity)} identities)")
print(f"  Each identity = subjectID_L  or  subjectID_R")
print(f"{'─'*60}")
print(f"  Min    : {per_identity.min()}")
print(f"  Median : {per_identity.median():.0f}")
print(f"  Mean   : {per_identity.mean():.1f}")
print(f"  Max    : {per_identity.max()}")
print()

print("  Full identity count table:")
print(f"  {'Identity':<20} {'Images':>7}")
print(f"  {'-'*28}")
for ident, n in per_identity.items():
    flag = "  ⚠ INCOMPLETE" if n < per_identity.max() else ""
    print(f"  {ident:<20} {n:>7}{flag}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Images per spectrum
# ═══════════════════════════════════════════════════════════════════════════════

per_spectrum = (df.groupby('spectrum')
                  .size()
                  .rename('n_images')
                  .sort_index())

time.sleep(10)
print(f"{'─'*60}")
print(f"  Images per spectrum")
print(f"{'─'*60}")
print(f"  {'Spectrum':<12} {'Images':>8}")
print(f"  {'-'*22}")
for spec, n in per_spectrum.items():
    print(f"  {spec:<12} {n:>8,}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Identity × Spectrum cross-table
#     Rows   = identity (subjectID_L or subjectID_R)
#     Columns = spectrum wavelength
#     Values  = number of sample images for that (identity, spectrum) pair
# ═══════════════════════════════════════════════════════════════════════════════

cross = (df.groupby(['identity', 'spectrum'])
           .size()
           .unstack(fill_value=0))

# Sort rows: group by subject, then L before R
cross = cross.sort_index()

# Expected images per cell = number of unique iterations
expected = df['iteration'].nunique()

time.sleep(10)
print(f"{'═'*60}")
print(f"  Identity × Spectrum Sample Count Table")
print(f"  Rows    = identity (subjectID_L / subjectID_R)")
print(f"  Columns = spectrum wavelength")
print(f"  Values  = number of images  (expected {expected} per cell)")
print(f"{'═'*60}")

# Pretty-print with a 'TOTAL' column appended
cross_display = cross.copy()
cross_display.insert(0, 'TOTAL', cross_display.sum(axis=1))
print(cross_display.to_string())
print()

# Identities where ANY cell (excluding TOTAL) < expected
incomplete = cross[(cross < expected).any(axis=1)]
if len(incomplete):
    print(f"  ⚠  {len(incomplete)} identities with at least one missing "
          f"capture (< {expected} images in some spectrum):")
    inc_display = incomplete.copy()
    inc_display.insert(0, 'TOTAL', inc_display.sum(axis=1))
    print(inc_display.to_string())
else:
    print(f"  ✓  All identities have exactly {expected} images per spectrum.")
print()

# Also export the table to CSV for reference
csv_path = os.path.join(os.path.dirname(DATA_ROOT) or ".", "identity_spectrum_table.csv")
cross_display.to_csv(csv_path)
print(f"  Identity × Spectrum table saved to: {csv_path}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Plots
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("CASIA-MS-ROI Dataset Statistics  (L / R = distinct identities)",
             fontsize=14, fontweight='bold')

# ── 5a. Histogram of images per identity ─────────────────────────────────────
ax = axes[0]
per_identity.hist(ax=ax, bins=20, color='steelblue', edgecolor='white')
ax.set_title("Images per Identity (histogram)")
ax.set_xlabel("Number of images")
ax.set_ylabel("Number of identities")
ax.axvline(per_identity.median(), color='red',    linestyle='--',
           linewidth=1.5, label=f"Median={per_identity.median():.0f}")
ax.axvline(per_identity.mean(),   color='orange', linestyle=':',
           linewidth=1.5, label=f"Mean={per_identity.mean():.1f}")
ax.legend(fontsize=9)

# ── 5b. Images per spectrum (bar chart) ──────────────────────────────────────
ax = axes[1]
per_spectrum.plot(kind='bar', ax=ax, color='teal', edgecolor='white', rot=0)
ax.set_title("Images per Spectrum")
ax.set_xlabel("Spectrum")
ax.set_ylabel("Number of images")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + per_spectrum.max() * 0.01,
            f"{int(bar.get_height()):,}",
            ha='center', va='bottom', fontsize=9)

# ── 5c. Heatmap: identity × spectrum ─────────────────────────────────────────
ax = axes[2]
im = ax.imshow(cross.values, aspect='auto', cmap='YlOrRd',
               vmin=0, vmax=expected)
ax.set_title(f"Images per Identity × Spectrum\n"
             f"(expected {expected} per cell)")
ax.set_xticks(range(len(cross.columns)))
ax.set_xticklabels(cross.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(cross.index)))
ax.set_yticklabels(cross.index, fontsize=6)
ax.set_xlabel("Spectrum")
ax.set_ylabel("Identity  (subject_L / subject_R)")
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label='image count')

plt.tight_layout()
plt.savefig("casia_ms_stats.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Plot saved to casia_ms_stats.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Summary for episode sampler
# ═══════════════════════════════════════════════════════════════════════════════

time.sleep(10)
print(f"{'═'*60}")
print(f"  Episode Sampler Summary  (Q_PER_CLASS = 5)")
print(f"{'═'*60}")
Q = 5
safe_k   = per_identity.min() - Q
viable32 = (per_identity >= (safe_k + Q)).sum()
print(f"  Min images per identity          : {per_identity.min()}")
print(f"  Safe K (= min - Q)               : {safe_k}")
print(f"  Identities viable for episodes   : {viable32} / {len(per_identity)}")
print(f"  → With N=32, need ≥32 viable classes. "
      f"{'✓ OK' if viable32 >= 32 else '✗ TOO FEW — lower N'}")
'''
