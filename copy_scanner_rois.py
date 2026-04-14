"""
Copy scanner ROIs into smartphone_data.

For each subject ID found in smartphone_data/:
  - Check if scanner_mediapipe_roi/{ID}/ exists
  - If yes, copy it to smartphone_data/{ID}/roi_scanner/

Result: smartphone_data/{ID}/roi_scanner/{images}
"""

import os
import shutil

# ── Paths — edit these ────────────────────────────────────────
SMARTPHONE_ROOT  = "/home/pai-ng/Jamal/smartphone_data"
SCANNER_ROI_ROOT = "/home/pai-ng/Jamal/scanner_mediapipe_roi"
# ─────────────────────────────────────────────────────────────

copied   = []
skipped  = []
no_match = []

for subject_id in sorted(os.listdir(SMARTPHONE_ROOT)):
    phone_dir   = os.path.join(SMARTPHONE_ROOT,  subject_id)
    scanner_dir = os.path.join(SCANNER_ROI_ROOT, subject_id)
    dst_dir     = os.path.join(phone_dir, "roi_scanner")

    if not os.path.isdir(phone_dir):
        continue

    if not os.path.isdir(scanner_dir):
        no_match.append(subject_id)
        continue

    if os.path.exists(dst_dir):
        print(f"  [SKIP] {subject_id}/roi_scanner already exists")
        skipped.append(subject_id)
        continue

    shutil.copytree(scanner_dir, dst_dir)
    n_files = len([f for f in os.listdir(dst_dir)
                   if os.path.isfile(os.path.join(dst_dir, f))])
    print(f"  [OK]   {subject_id}/roi_scanner  ({n_files} files)")
    copied.append(subject_id)

print(f"\n{'='*50}")
print(f"  Copied   : {len(copied)}")
print(f"  Skipped  : {len(skipped)}  (roi_scanner already existed)")
print(f"  No match : {len(no_match)}  (not in scanner_mediapipe_roi)")
if no_match:
    print(f"  IDs with no scanner ROI: {no_match}")
print(f"{'='*50}")
