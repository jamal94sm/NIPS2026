"""
MediaPipe Palm ROI Extraction
==============================
Robust against high-resolution, complex-background smartphone images.

Improvements over baseline:
  1. Multi-scale detection — tries multiple downscale sizes before giving up
  2. CLAHE contrast enhancement — second pass on difficult images
  3. Landmark geometric validation — rejects hallucinated detections
  4. num_hands=2 — picks the highest-confidence hand
  5. Coordinate mapping — landmarks always mapped back to original resolution
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

ROI_SIZE   = 224          # output square size in pixels
SAVE_FAILED = False       # True → save resized full image for failed samples

FAILED_JSON_PATH = os.path.join(DST_ROOT, "failed_samples.json")

# Detection scales tried in order (long-edge pixels).
# Smaller values run faster; larger values catch small/distant hands.
DETECTION_SCALES = [1024, 640, 1280, 512, 384]

# Minimum fraction of image width the detected palm must span.
# Raise to reject tiny/distant detections; lower if hands are far from camera.
MIN_PALM_WIDTH_RATIO = 0.05

# ============================================================
#  MEDIAPIPE INIT
# ============================================================

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,                          # detect up to 2; pick best score
    min_hand_detection_confidence=0.15,
    min_hand_presence_confidence=0.15,
    min_tracking_confidence=0.15,
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.HandLandmarker.create_from_options(options)


# ============================================================
#  DETECTION HELPERS
# ============================================================

def _enhance_for_detection(bgr):
    """
    CLAHE contrast enhancement in LAB space.
    Helps MediaPipe on low-contrast, shadowed, or overexposed images.
    """
    lab       = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b   = cv2.split(lab)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l         = clahe.apply(l)
    enhanced  = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _detect_at_scale(image_bgr, max_dim):
    """
    Downscale image so its longest edge = max_dim, run detector,
    then map landmark coordinates back to original resolution.
    Returns (pts, handedness_str) or (None, None).
    """
    h, w  = image_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))

    if scale < 1.0:
        dw, dh    = int(w * scale), int(h * scale)
        det_img   = cv2.resize(image_bgr, (dw, dh),
                               interpolation=cv2.INTER_AREA)
    else:
        det_img   = image_bgr

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
    result   = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None, None

    # Pick the hand with the highest detection confidence
    scores = [result.handedness[i][0].score
              for i in range(len(result.hand_landmarks))]
    best   = int(np.argmax(scores))

    lm_list    = result.hand_landmarks[best]
    handedness = result.handedness[best][0].category_name   # 'Left' | 'Right'

    # Map back to ORIGINAL image resolution
    pts = [(int(p.x * w), int(p.y * h)) for p in lm_list]
    return pts, handedness


def _run_mp_hands_robust(image_bgr):
    """
    Multi-scale detection cascade.
    Pass 1: original image at each scale.
    Pass 2: CLAHE-enhanced image at each scale.
    Returns (pts, handedness) or (None, None).
    """
    # Pass 1 — original colours
    for max_dim in DETECTION_SCALES:
        pts, hand = _detect_at_scale(image_bgr, max_dim)
        if pts is not None:
            return pts, hand

    # Pass 2 — contrast-enhanced
    enhanced = _enhance_for_detection(image_bgr)
    for max_dim in DETECTION_SCALES:
        pts, hand = _detect_at_scale(enhanced, max_dim)
        if pts is not None:
            return pts, hand

    return None, None


# ============================================================
#  LANDMARK VALIDATION
# ============================================================

def _validate_landmarks(pts, img_shape):
    """
    Geometric sanity checks on the 21 detected landmarks.
    Returns False if the detection is likely a false positive.
    """
    h, w    = img_shape[:2]
    arr     = np.array(pts, dtype=float)

    # 1. All landmarks must be inside the image
    if (np.any(arr[:, 0] < 0) or np.any(arr[:, 0] >= w) or
            np.any(arr[:, 1] < 0) or np.any(arr[:, 1] >= h)):
        return False

    # 2. Palm span (wrist → pinky base) must be at least MIN_PALM_WIDTH_RATIO
    palm_span = float(np.linalg.norm(arr[0] - arr[17]))
    if palm_span < MIN_PALM_WIDTH_RATIO * max(h, w):
        return False

    # 3. Wrist (0) should be below the middle fingertip (12) — palm facing up
    #    (y increases downward; allow ±20% tolerance for tilted hands)
    wrist_y    = float(arr[0,  1])
    mid_tip_y  = float(arr[12, 1])
    if wrist_y < mid_tip_y - 0.20 * h:
        return False

    # 4. Fingertips (4,8,12,16,20) must be spatially spread out
    tips   = arr[[4, 8, 12, 16, 20]]
    spread = float(np.std(tips[:, 0]))
    if spread < 0.01 * w:
        return False

    # 5. The five finger tips should not all be clustered at one point
    tip_range = float(np.ptp(tips, axis=0).max())
    if tip_range < 0.02 * max(h, w):
        return False

    return True


# ============================================================
#  ROI GEOMETRY HELPERS
# ============================================================

def _midpoints(pairs):
    return [((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            for p1, p2 in pairs]


def _calculate_point_c(m1, m2, thumb):
    """
    Compute palm centre C:
    1.8× the knuckle-line length, offset perpendicular toward the wrist.
    """
    m1, m2, thumb = map(np.asarray, (m1, m2, thumb))
    O    = (m1 + m2) / 2.0
    AB   = m2 - m1
    L    = float(np.linalg.norm(AB))
    if L == 0:
        raise ValueError("Midpoints coincide — degenerate hand pose.")
    ABu  = AB / L
    perp = np.array([-ABu[1], ABu[0]])
    # Flip perpendicular so it points toward the wrist side (thumb side)
    cross_z = ABu[0] * float((thumb - O)[1]) - ABu[1] * float((thumb - O)[0])
    if cross_z < 0:
        perp = -perp
    C = O + 1.8 * L * perp
    return int(C[0]), int(C[1])


def _extract_roi(img, mid1, mid2, C, thumb, hand_type):
    """
    Rotate the image so the palm knuckle line is horizontal,
    then crop a square centred at C.
    """
    vec   = np.array(mid2) - np.array(mid1)
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
    C     = (int(C[0]), int(C[1]))

    if hand_type.lower() == "right":
        if np.dot(vec, np.array(thumb) - np.array(C)) > 0:
            angle += 180.0
    else:
        if np.dot(vec, np.array(thumb) - np.array(C)) < 0:
            angle += 180.0

    side = float(np.linalg.norm(vec)) * 2.5
    side = max(side, 10.0)          # guard against degenerate crops

    rect = (C, (side, side), angle)
    M    = cv2.getRotationMatrix2D(C, angle, 1.0)
    rot  = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    roi  = cv2.getRectSubPix(rot, (int(side), int(side)), C)
    box  = cv2.boxPoints(rect).astype(np.int32)
    return roi, box


# ============================================================
#  MAIN EXTRACTION FUNCTION
# ============================================================

def extract_palm_roi(image_bgr):
    """
    Full pipeline: detect → validate → crop → rotate 180°.
    Returns (roi_bgr, annotated_bgr, hand_type) or (None, None, None).
    """
    lms, hand_type = _run_mp_hands_robust(image_bgr)

    if lms is None:
        return None, None, None

    if not _validate_landmarks(lms, image_bgr.shape):
        return None, None, None

    idx = lambda i: lms[i]

    # Build the two palm reference points from knuckle midpoints
    mids4 = _midpoints([
        (idx(17), idx(18)),
        (idx(14), idx(13)),
        (idx(10), idx(9)),
        (idx(6),  idx(5)),
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

    # Flip so fingers point upward
    roi = cv2.rotate(roi, cv2.ROTATE_180)

    # Annotated image for debugging
    ann = image_bgr.copy()
    for x, y in lms:
        cv2.circle(ann, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.polylines(ann, [box], True, (0, 255, 0), 2)
    cv2.circle(ann, C, 6, (0, 0, 255), -1)

    return roi, ann, hand_type


# ============================================================
#  FILENAME PARSER
# ============================================================

def parse_scanner_filename(fname):
    """
    Parses filenames like: 065_S2_Left_magenta.jpg
    Returns dict or None if the filename does not match.
    """
    stem  = os.path.splitext(fname)[0]
    parts = stem.split("_")
    if len(parts) != 4:
        return None
    subj_id, session, hand, color = parts
    if (subj_id.isdigit()
            and session.startswith("S")
            and hand in ("Left", "Right")):
        return dict(id=subj_id, session=session, hand=hand, color=color)
    return None


# ============================================================
#  MAIN LOOP
# ============================================================

def main():
    os.makedirs(DST_ROOT, exist_ok=True)

    # Collect all matching images
    all_images = []
    for root, _, files in os.walk(SRC_ROOT):
        for f in sorted(files):
            if not f.lower().endswith(".jpg"):
                continue
            parsed = parse_scanner_filename(f)
            if parsed:
                all_images.append((os.path.join(root, f), parsed))

    print(f"\nFound {len(all_images)} images.")
    print(f"Detection scales : {DETECTION_SCALES}")
    print(f"Output dir       : {DST_ROOT}\n")

    num_success  = 0
    num_failed   = 0
    failed_samples = []
    scale_usage  = {}       # track which scale succeeded most often

    pbar = tqdm(all_images, desc="Extracting palm ROIs")

    for src_path, meta in pbar:
        rel_path = os.path.relpath(src_path, SRC_ROOT)
        dst_path = os.path.join(DST_ROOT, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Load image
        try:
            img_rgb = np.array(Image.open(src_path).convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            failed_samples.append({"path": src_path, "reason": f"load_error: {e}"})
            num_failed += 1
            pbar.set_postfix(ok=num_success, fail=num_failed)
            continue

        # Extract ROI
        roi_bgr, ann_bgr, hand_type = extract_palm_roi(img_bgr)

        if roi_bgr is None or roi_bgr.size == 0:
            failed_samples.append({"path": src_path, "reason": "no_detection"})
            num_failed += 1
            if SAVE_FAILED:
                fallback = cv2.resize(img_bgr, (ROI_SIZE, ROI_SIZE))
                cv2.imwrite(dst_path, fallback)
        else:
            roi_out = cv2.resize(roi_bgr, (ROI_SIZE, ROI_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(dst_path, roi_out)
            num_success += 1

        pbar.set_postfix(ok=num_success, fail=num_failed)

    # ── Save failure report ───────────────────────────────────
    os.makedirs(os.path.dirname(FAILED_JSON_PATH) or ".", exist_ok=True)
    with open(FAILED_JSON_PATH, "w") as fp:
        json.dump(failed_samples, fp, indent=2)

    # ── Summary ───────────────────────────────────────────────
    total = num_success + num_failed
    print(f"\n{'='*50}")
    print(f"  Done.")
    print(f"  Success  : {num_success} / {total}  "
          f"({100*num_success/max(total,1):.1f}%)")
    print(f"  Failed   : {num_failed} / {total}  "
          f"({100*num_failed/max(total,1):.1f}%)")
    print(f"  Failures : {FAILED_JSON_PATH}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
