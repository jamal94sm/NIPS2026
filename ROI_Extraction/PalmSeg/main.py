"""
Main script: process a directory of palmprint images and save ROIs.
Equivalent to MATLAB launch_PalmSeg.m.

Usage
-----
    python main.py --db Tongji --input ./images/ --output ./ROIs/
    python main.py --db CASIA  --input ./images/ --output ./ROIs/ --ext bmp

Supported databases: CASIA, IITD, REST, Tongji
"""

import os
import argparse
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from palmnet_roi import get_params, segment_palms, find_roi_from_shape


# ---------------------------------------------------------------------------

def process_image(img_path: Path, param: dict, verbose: bool = True):
    """
    Full pipeline for one image.

    Returns
    -------
    roi      : float (H, W) or (H, W, 3) array, or None on failure.
    results  : dict with valley_points, grad_refined.
    """
    # Load image
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0

    # --- Segmentation (shape + binary mask) --------------------------------
    shape_final, centroid, orient_m, bw = segment_palms(img, param)

    # --- ROI extraction ----------------------------------------------------
    roi, error, results = find_roi_from_shape(
        img, bw, shape_final, centroid, param)

    return roi, results, error


def main():
    parser = argparse.ArgumentParser(
        description="PalmNet ROI extractor – Python translation of launch_PalmSeg.m")
    parser.add_argument('--db',     default='Tongji',
                        choices=['CASIA', 'IITD', 'REST', 'Tongji'],
                        help='Database name (selects parameter set)')
    parser.add_argument('--input',  default='./images/',
                        help='Directory of input palm images')
    parser.add_argument('--output', default='./ROIs/',
                        help='Directory to save extracted ROIs')
    parser.add_argument('--ext',    default='bmp',
                        help='Image file extension (bmp, jpg, png, …)')
    parser.add_argument('--quiet',  action='store_true',
                        help='Suppress per-image messages')
    args = parser.parse_args()

    param = get_params(args.db)
    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(in_dir.glob(f'*.{args.ext}'))
    if not images:
        print(f"[WARN] No .{args.ext} files found in {in_dir}")
        return

    ok_count, fail_count = 0, 0

    for i, img_path in enumerate(images, 1):
        print(f"[{i:4d}/{len(images)}] {img_path.name}: ", end='', flush=True)
        try:
            roi, results, error = process_image(img_path, param,
                                                verbose=not args.quiet)
            if error == 0 and roi is not None:
                out_path = out_dir / img_path.name
                # Save as the same extension; convert float → uint8
                roi_u8 = (np.clip(roi, 0, 1) * 255).astype(np.uint8)
                if roi_u8.ndim == 3:
                    roi_u8 = cv2.cvtColor(roi_u8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), roi_u8)
                print("OK")
                ok_count += 1
            else:
                print("FAILED – cannot extract ROI")
                fail_count += 1
        except Exception as exc:
            print(f"ERROR – {exc}")
            if not args.quiet:
                traceback.print_exc()
            fail_count += 1

    print(f"\nDone. Success: {ok_count}  Failed: {fail_count}")


if __name__ == '__main__':
    main()
