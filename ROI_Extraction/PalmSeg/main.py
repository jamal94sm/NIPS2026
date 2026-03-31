"""
PalmNet ROI Extractor
=====================
Edit the three settings below, then run:

    python main.py
"""

import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from palmnet_roi.params import get_params
from palmnet_roi.segment_palms import segment_palms
from palmnet_roi.find_roi import find_roi_from_shape

# =============================================================================
# USER SETTINGS  –  only edit these three lines
# =============================================================================

DATABASE   = 'MPDv2'       # CASIA | IITD | REST | Tongji | MPDv2
INPUT_DIR  = '/home/pai-ng/Jamal/MPDv2/'   # folder containing input palm images
OUTPUT_DIR = '/home/pai-ng/Jamal/MPDv2_ROI_PalmSeg/'    # folder where extracted ROIs will be saved

# =============================================================================
# Processing  –  do not edit below this line
# =============================================================================

def process_image(img_path: Path, param: dict):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    shape_final, centroid, _, bw = segment_palms(img, param)
    roi, error, results = find_roi_from_shape(img, bw, shape_final, centroid, param)
    return roi, error, results


def main():
    param   = get_params(DATABASE)
    ext     = param['meta']['image_ext']
    in_dir  = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(in_dir.glob(f'*.{ext}'))
    if not images:
        print(f"[WARN] No .{ext} files found in '{in_dir}'")
        return

    ok, fail = 0, 0
    for i, img_path in enumerate(images, 1):
        print(f"[{i:4d}/{len(images)}] {img_path.name}: ", end='', flush=True)
        try:
            roi, error, _ = process_image(img_path, param)
            if error == 0 and roi is not None:
                roi_u8 = (np.clip(roi, 0, 1) * 255).astype(np.uint8)
                if roi_u8.ndim == 3:
                    roi_u8 = cv2.cvtColor(roi_u8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_dir / img_path.name), roi_u8)
                print("OK")
                ok += 1
            else:
                print("FAILED – cannot extract ROI")
                fail += 1
        except Exception as exc:
            print(f"ERROR – {exc}")
            traceback.print_exc()
            fail += 1

    print(f"\nDone.  Success: {ok}   Failed: {fail}")


if __name__ == '__main__':
    main()
