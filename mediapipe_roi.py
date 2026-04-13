"""
MediaPipe Palm ROI Extraction
==============================
Improved keypoint detection pipeline for high-resolution,
complex-background smartphone images (MPDv2, Scanner, generic).

Detection strategy:
  Stage 1 — Skin-guided pre-crop
             Detect the skin region using HSV colour,
             crop tightly around it, run MediaPipe on the crop.
             This dramatically helps on large images where
             the hand occupies a small fraction of the frame.

  Stage 2 — Multi-scale full-image fallback
             If Stage 1 fails, try the full image at multiple
             downscale levels.

  Stage 3 — Preprocessing variants
             Each scale attempt is also tried with CLAHE,
             gamma correction, and unsharp-mask sharpening.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import json
from PIL import Image
from tqdm import tqdm

# ============================================================
#  PATHS & SETTINGS  — edit only this block
# ============================================================

SRC_ROOT   = "/home/pai-ng/Jamal/MPDv2"
DST_ROOT   = "/home/pai-ng/Jamal/MPDv2_mediapipe_roi"
MODEL_PATH = "/home/pai-ng/Jamal/NIPS2026/ROI_Extraction/hand_landmarker.task"

ROI_SIZE    = 224     # output square size in pixels
SAVE_FAILED = False   # True → save resized full image when extraction fails

FAILED_JSON_PATH = os.path.join(DST_ROOT, "failed_samples.json")

# Downscale sizes tried for full-image fallback (longest edge in pixels)
DETECTION_SCALES = [1024, 640, 1280, 512, 384]

# Skin-crop padding — fraction of crop size added on each side
SKIN_CROP_PADDING = 0.25

# Minimum palm span as fraction of max(H, W) — only rejects truly tiny detections
MIN_PALM_RATIO = 0.02

# ============================================================
#  MEDIAPIPE INIT
# ============================================================

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.10,   # low — preprocessing handles quality
    min_hand_presence_confidence=0.10,
    min_tracking_confidence=0.10,
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.HandLandmarker.create_from_options(options)


# ============================================================
#  IMAGE PREPROCESSING VARIANTS
# ============================================================

def _clahe(bgr):
    """CLAHE contrast enhancement in LAB space."""
    lab     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _gamma(bgr, g=1.5):
    """Gamma correction — brightens underexposed images."""
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / g)
    lut = (lut * 255).clip(0, 255).astype(np.uint8)
    return cv2.LUT(bgr, lut)


def _sharpen(bgr):
    """Unsharp mask — enhances edges to help landmark localisation."""
    blur    = cv2.GaussianBlur(bgr, (0, 0), sigmaX=3)
    sharp   = cv2.addWeighted(bgr, 1.5, blur, -0.5, 0)
    return sharp


def _get_variants(bgr):
    """
    Return a list of preprocessed versions of the image.
    Tried in order — first successful detection wins.
    """
    return [
        bgr,                   # 1. original
        _clahe(bgr),           # 2. contrast enhanced
        _gamma(bgr, 1.5),      # 3. brightened
        _gamma(bgr, 0.7),      # 4. darkened  (overexposed images)
        _sharpen(bgr),         # 5. sharpened
        _clahe(_sharpen(bgr)), # 6. sharpen then enhance
    ]


# ============================================================
#  SKIN-GUIDED PRE-CROP
# ============================================================

def _skin_bbox(bgr, padding=SKIN_CROP_PADDING):
    """
    Detect the largest skin-coloured region and return a padded
    bounding box (x0, y0, x1, y1) in original image coordinates.
    Returns None if no skin region is found.

    Works across a wide range of skin tones using a union of
    HSV ranges and a YCrCb range.
    """
    h, w = bgr.shape[:2]

    # HSV skin ranges (light to dark tones)
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1   = cv2.inRange(hsv, np.array([0,  15,  40], np.uint8),
                            np.array([25, 255, 255], np.uint8))
    m2   = cv2.inRange(hsv, np.array([155, 15,  40], np.uint8),
                            np.array([180, 255, 255], np.uint8))

    # YCrCb skin range
    ycr  = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m3   = cv2.inRange(ycr, np.array([0,  133, 77], np.uint8),
                            np.array([255, 173, 127], np.uint8))

    mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)

    # Clean up noise
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

    # Find the largest connected skin blob
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None

    # Ignore background (label 0), pick largest foreground blob
    areas = stats[1:, cv2.CC_STAT_AREA]
    best  = int(np.argmax(areas)) + 1

    # Must cover at least 1% of the image — avoids tiny false positives
    if float(areas[best - 1]) < 0.01 * h * w:
        return None

    bx = int(stats[best, cv2.CC_STAT_LEFT])
    by = int(stats[best, cv2.CC_STAT_TOP])
    bw = int(stats[best, cv2.CC_STAT_WIDTH])
    bh = int(stats[best, cv2.CC_STAT_HEIGHT])

    # Add padding
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x0 = max(0,     bx - pad_x)
    y0 = max(0,     by - pad_y)
    x1 = min(w - 1, bx + bw + pad_x)
    y1 = min(h - 1, by + bh + pad_y)

    return x0, y0, x1, y1


# ============================================================
#  CORE DETECTION
# ============================================================

def _run_detector(image_bgr):
    """Run MediaPipe on a single BGR image. Returns (pts, hand) or (None, None)."""
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None, None
    scores     = [result.handedness[i][0].score
                  for i in range(len(result.hand_landmarks))]
    best       = int(np.argmax(scores))
    lm_list    = result.hand_landmarks[best]
    handedness = result.handedness[best][0].category_name
    dh, dw     = image_bgr.shape[:2]
    pts        = [(int(p.x * dw), int(p.y * dh)) for p in lm_list]
    return pts, handedness


def _detect_on_image(image_bgr):
    """
    Try all preprocessing variants on a given image.
    Returns (pts, hand) in image_bgr coordinate space, or (None, None).
    """
    for variant in _get_variants(image_bgr):
        pts, hand = _run_detector(variant)
        if pts is not None:
            return pts, hand
    return None, None


def _detect_at_scale(image_bgr, max_dim, offset_xy=(0, 0)):
    """
    Downscale to max_dim, run all variants, remap to original coords.
    offset_xy: (ox, oy) shift to add when remapping (for crops).
    """
    h, w  = image_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        dw, dh   = int(w * scale), int(h * scale)
        small    = cv2.resize(image_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
    else:
        small    = image_bgr
        scale    = 1.0

    pts, hand = _detect_on_image(small)
    if pts is None:
        return None, None

    # Remap to original (or full-image) coordinates
    ox, oy = offset_xy
    pts = [(int(x / scale) + ox, int(y / scale) + oy) for x, y in pts]
    return pts, hand


def _run_mp_hands_robust(image_bgr):
    """
    Full detection cascade:

    Stage 1 — skin-guided pre-crop (best for high-res images with small hands)
      a) detect skin bbox
      b) crop + run all preprocessing variants at multiple scales

    Stage 2 — full-image multi-scale fallback
      Try each scale in DETECTION_SCALES with all preprocessing variants.

    Returns (pts, handedness) in ORIGINAL image coordinates, or (None, None).
    """
    h, w = image_bgr.shape[:2]

    # ── Stage 1: skin-guided crop ─────────────────────────────
    bbox = _skin_bbox(image_bgr)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        crop = image_bgr[y0:y1, x0:x1]
        if crop.size > 0:
            for max_dim in [640, 1024, 512, 384]:
                pts, hand = _detect_at_scale(crop, max_dim,
                                             offset_xy=(x0, y0))
                if pts is not None:
                    return pts, hand

    # ── Stage 2: full-image multi-scale ───────────────────────
    for max_dim in DETECTION_SCALES:
        pts, hand = _detect_at_scale(image_bgr, max_dim)
        if pts is not None:
            return pts, hand

    return None, None


# ============================================================
#  ROI GEOMETRY
# ============================================================

def _midpoints(pairs):
    return [((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            for p1, p2 in pairs]


def _calculate_point_c(m1, m2, thumb):
    m1, m2, thumb = map(np.asarray, (m1, m2, thumb))
    O    = (m1 + m2) / 2.0
    AB   = m2 - m1
    L    = float(np.linalg.norm(AB))
    if L == 0:
        raise ValueError("Midpoints coincide.")
    ABu  = AB / L
    perp = np.array([-ABu[1], ABu[0]])
    cz   = float(ABu[0]) * float((thumb - O)[1]) \
         - float(ABu[1]) * float((thumb - O)[0])
    if cz < 0:
        perp = -perp
    C = O + 1.8 * L * perp
    return int(C[0]), int(C[1])


def _extract_roi(img, mid1, mid2, C, thumb, hand_type):
    vec   = np.array(mid2) - np.array(mid1)
    angle = float(np.degrees(np.arctan2(float(vec[1]), float(vec[0]))))
    C     = (int(C[0]), int(C[1]))
    if hand_type.lower() == "right":
        if np.dot(vec, np.array(thumb) - np.array(C)) > 0:
            angle += 180.0
    else:
        if np.dot(vec, np.array(thumb) - np.array(C)) < 0:
            angle += 180.0
    side  = max(float(np.linalg.norm(vec)) * 2.5, 10.0)
    rect  = (C, (side, side), angle)
    M     = cv2.getRotationMatrix2D(C, angle, 1.0)
    rot   = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    roi   = cv2.getRectSubPix(rot, (int(side), int(side)), C)
    box   = cv2.boxPoints(rect).astype(np.int32)
    return roi, box


# ============================================================
#  MAIN EXTRACTION FUNCTION
# ============================================================

def extract_palm_roi(image_bgr):
    lms, hand_type = _run_mp_hands_robust(image_bgr)
    if lms is None:
        return None, None, None

    idx = lambda i: lms[i]

    mids4 = _midpoints([
        (idx(17), idx(18)), (idx(14), idx(13)),
        (idx(10), idx(9)),  (idx(6),  idx(5)),
    ])
    adj = _midpoints([
        (mids4[0], mids4[1]),
        (mids4[1], mids4[2]),
        (mids4[2], mids4[3]),
    ])
    roi_mid1 = ((adj[0][0] + adj[1][0]) / 2.0,
                (adj[0][1] + adj[1][1]) / 2.0)
    roi_mid2 = ((adj[1][0] + adj[2][0]) / 2.0,
                (adj[1][1] + adj[2][1]) / 2.0)
    thumb = idx(2)

    try:
        C = _calculate_point_c(roi_mid1, roi_mid2, thumb)
    except ValueError:
        return None, None, None

    roi, box = _extract_roi(image_bgr, roi_mid1, roi_mid2,
                            C, thumb, hand_type)
    if roi is None or roi.size == 0:
        return None, None, None

    roi = cv2.rotate(roi, cv2.ROTATE_180)

    ann = image_bgr.copy()
    for x, y in lms:
        cv2.circle(ann, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.polylines(ann, [box], True, (0, 255, 0), 2)
    cv2.circle(ann, C, 6, (0, 0, 255), -1)

    return roi, ann, hand_type


# ============================================================
#  FILENAME PARSER
# ============================================================

def parse_filename(fname):
    """
    MPDv2   : 001_1_h_l_01.jpg   (5 parts, parts[2]=='h')
    Scanner : 065_S2_Left_magenta.jpg (4 parts)
    Generic : anything else — always accepted
    """
    stem  = os.path.splitext(fname)[0]
    parts = stem.split("_")

    if len(parts) == 5 and parts[2].lower() == "h":
        side = "Left" if parts[3].lower() == "l" else "Right"
        return dict(id=parts[0], session=parts[1],
                    hand=side, iter=parts[4], fmt="mpd")

    if len(parts) == 4:
        sid, ses, hand, extra = parts
        if sid.isdigit() and ses.startswith("S") and hand in ("Left","Right"):
            return dict(id=sid, session=ses, hand=hand,
                        color=extra, fmt="scanner")

    return dict(id=parts[0] if parts else stem,
                session="", hand="unknown", fmt="generic")


# ============================================================
#  MAIN LOOP
# ============================================================

def main():
    os.makedirs(DST_ROOT, exist_ok=True)

    # ── Diagnostic scan ───────────────────────────────────────
    print(f"\nScanning {SRC_ROOT} ...")
    raw_count, samples = 0, []
    for root, _, files in os.walk(SRC_ROOT):
        for f in sorted(files):
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                full = os.path.join(root, f)
                if raw_count < 5: samples.append(full)
                raw_count += 1

    print(f"  Raw image files : {raw_count}")
    if raw_count == 0:
        print(f"\n  ERROR: No images found in {SRC_ROOT}")
        print(f"  Check that SRC_ROOT is set correctly.")
        return
    print(f"  First few paths :")
    for p in samples:
        print(f"    {p}")
    print()

    # ── Collect all images ────────────────────────────────────
    all_images = []
    for root, _, files in os.walk(SRC_ROOT):
        for f in sorted(files):
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                all_images.append((os.path.join(root, f), parse_filename(f)))

    print(f"Found {len(all_images)} images.")
    print(f"Detection scales  : {DETECTION_SCALES}")
    print(f"Output dir        : {DST_ROOT}\n")

    num_success, num_failed = 0, 0
    failed_samples = []

    pbar = tqdm(all_images, desc="Extracting palm ROIs")
    for src_path, meta in pbar:
        rel_path = os.path.relpath(src_path, SRC_ROOT)
        dst_path = os.path.join(DST_ROOT, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Load
        try:
            img_rgb = np.array(Image.open(src_path).convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            failed_samples.append({"path": src_path,
                                   "reason": f"load_error: {e}"})
            num_failed += 1
            pbar.set_postfix(ok=num_success, fail=num_failed)
            continue

        # Extract
        roi_bgr, _, hand_type = extract_palm_roi(img_bgr)

        if roi_bgr is None or roi_bgr.size == 0:
            failed_samples.append({"path": src_path,
                                   "reason": "no_detection"})
            num_failed += 1
            if SAVE_FAILED:
                cv2.imwrite(dst_path,
                            cv2.resize(img_bgr, (ROI_SIZE, ROI_SIZE)))
        else:
            roi_out = cv2.resize(roi_bgr, (ROI_SIZE, ROI_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
            # Save as RGB
            cv2.imwrite(dst_path,
                        cv2.cvtColor(roi_out, cv2.COLOR_BGR2RGB))
            num_success += 1

        pbar.set_postfix(ok=num_success, fail=num_failed)

    # ── Save failure report ───────────────────────────────────
    with open(FAILED_JSON_PATH, "w") as fp:
        json.dump(failed_samples, fp, indent=2)

    # ── Summary ───────────────────────────────────────────────
    total = num_success + num_failed
    print(f"\n{'='*52}")
    print(f"  Done.")
    print(f"  Success  : {num_success:>6} / {total}  "
          f"({100*num_success/max(total,1):.1f}%)")
    print(f"  Failed   : {num_failed:>6} / {total}  "
          f"({100*num_failed/max(total,1):.1f}%)")
    print(f"  Failures : {FAILED_JSON_PATH}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
