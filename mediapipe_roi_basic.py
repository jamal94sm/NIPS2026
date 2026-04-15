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
# PATHS  — edit these lines
# ============================================================
SRC_ROOT        = "/home/pai-ng/Jamal/CASIA_Palmprint_V1"
DST_ROOT        = "/home/pai-ng/Jamal/CASIA_mediapipe_roi"
MODEL_PATH      = "/home/pai-ng/Jamal/NIPS2026/ROI_Extraction/hand_landmarker.task"
ROI_SIZE        = 224
SAVE_FAILED     = False   # True → save resized full image for failed samples
FAILED_JSON_PATH = os.path.join(DST_ROOT, "failed_samples.json")

# ============================================================
# Initialize MediaPipe Tasks Detector
# ============================================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.2,
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.HandLandmarker.create_from_options(options)


def _run_mp_hands_new(image_bgr):
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)

    if not detection_result.hand_landmarks:
        return None, None

    lm_list    = detection_result.hand_landmarks[0]
    handedness = detection_result.handedness[0][0].category_name  # 'Left' or 'Right'

    h, w = image_bgr.shape[:2]
    pts  = [(int(p.x * w), int(p.y * h)) for p in lm_list]
    return pts, handedness


# ============================================================
# ROI MATH HELPERS
# ============================================================

def _midpoints(pairs):
    return [((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            for p1, p2 in pairs]


def _calculate_point_c(m1, m2, thumb):
    m1, m2, thumb = map(np.asarray, (m1, m2, thumb))
    O    = (m1 + m2) / 2.0
    AB   = m2 - m1
    L    = np.linalg.norm(AB)
    if L == 0:
        raise ValueError("Midpoints coincide")
    ABu  = AB / L
    perp = np.array([-ABu[1], ABu[0]])
    cross_z = ABu[0] * float((thumb - O)[1]) - ABu[1] * float((thumb - O)[0])
    if cross_z < 0:
        perp = -perp
    C = O + 1.8 * L * perp
    return int(C[0]), int(C[1])


def _extract_roi(img, mid1, mid2, C, thumb, hand_type):
    vec   = np.array(mid2) - np.array(mid1)
    angle = float(np.degrees(np.arctan2(float(vec[1]), float(vec[0]))))
    C     = (int(C[0]), int(C[1]))

    if hand_type.lower() == "right":
        if np.dot(vec, np.array(thumb) - np.array(C)) > 0:
            angle += 180
    else:
        if np.dot(vec, np.array(thumb) - np.array(C)) < 0:
            angle += 180

    side = float(np.linalg.norm(vec)) * 2.5
    side = max(side, 10.0)

    rect   = (C, (side, side), angle)
    M      = cv2.getRotationMatrix2D(C, angle, 1.0)
    rot    = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    roi    = cv2.getRectSubPix(rot, (int(side), int(side)), C)
    box    = cv2.boxPoints(rect).astype(np.int32)
    return roi, box


def extract_palm_roi(image_bgr):
    lms, hand_type = _run_mp_hands_new(image_bgr)
    if lms is None:
        return None, None, None

    idx   = lambda i: lms[i]
    mids4 = _midpoints([
        (idx(17), idx(18)), (idx(14), idx(13)),
        (idx(10), idx(9)),  (idx(6),  idx(5)),
    ])
    adj = _midpoints([
        (mids4[0], mids4[1]),
        (mids4[1], mids4[2]),
        (mids4[2], mids4[3]),
    ])
    roi_mid1 = ((adj[0][0] + adj[1][0]) / 2, (adj[0][1] + adj[1][1]) / 2)
    roi_mid2 = ((adj[1][0] + adj[2][0]) / 2, (adj[1][1] + adj[2][1]) / 2)

    thumb = idx(2)

    try:
        C = _calculate_point_c(roi_mid1, roi_mid2, thumb)
    except ValueError:
        return None, None, None

    roi, box = _extract_roi(image_bgr, roi_mid1, roi_mid2, C, thumb, hand_type)

    if roi is None or roi.size == 0:
        return None, None, None

    # Rotate 180° so fingers point up
    roi = cv2.rotate(roi, cv2.ROTATE_180)

    ann = image_bgr.copy()
    for x, y in lms:
        cv2.circle(ann, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.polylines(ann, [box], True, (0, 255, 0), 2)
    cv2.circle(ann, C, 6, (0, 0, 255), -1)

    return roi, ann, hand_type


# ============================================================
# Filename Parser  — handles MPDv2, Scanner, and generic files
# ============================================================

def parse_filename(fname):
    """
    Accepts any image file regardless of naming format.

    MPDv2   : {subject}_{session}_{device}_{hand}_{iter}.jpg
              e.g. 001_1_h_l_01.jpg  (5 parts, device ∈ {h, m})
    Scanner : {id}_{session}_{hand}_{color}.jpg
              e.g. 065_S2_Left_magenta.jpg  (4 parts, hand ∈ {Left, Right})
    Generic : anything else — accepted with minimal metadata
    """
    stem  = os.path.splitext(fname)[0]
    parts = stem.split("_")

    # MPDv2: {subject}_{session}_{device}_{hand}_{iter}
    if len(parts) == 5 and parts[2].lower() in ("h", "m"):
        side = "Left" if parts[3].lower() == "l" else "Right"
        return dict(id=parts[0], session=parts[1],
                    device=parts[2], hand=side, iter=parts[4])

    # Scanner: {id}_{session}_{hand}_{color}
    if len(parts) == 4 and parts[2] in ("Left", "Right"):
        return dict(id=parts[0], session=parts[1],
                    hand=parts[2], color=parts[3])

    # Generic fallback — never skip a file due to naming
    return dict(id=parts[0] if parts else stem,
                session="", hand="unknown")


# ============================================================
# Main Loop
# ============================================================

def main():
    os.makedirs(DST_ROOT, exist_ok=True)

    # Collect all images (any common extension)
    all_images = []
    for root, _, files in os.walk(SRC_ROOT):
        for f in sorted(files):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            meta = parse_filename(f)           # always returns a dict
            all_images.append((os.path.join(root, f), meta))

    print(f"Found {len(all_images)} images.")
    if len(all_images) == 0:
        print(f"  ERROR: No images found in {SRC_ROOT}")
        print(f"  Check that SRC_ROOT is correct.")
        return

    # Print a few sample paths so the user can confirm the scan is working
    print("  First few paths found:")
    for src_path, _ in all_images[:5]:
        print(f"    {src_path}")
    print()

    num_success  = 0
    num_failed   = 0
    failed_samples = []

    pbar = tqdm(all_images, desc="Processing images")
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
            pbar.set_postfix(success=num_success, failed=num_failed)
            continue

        # Extract ROI
        roi_bgr, _, _ = extract_palm_roi(img_bgr)

        if roi_bgr is None or roi_bgr.size == 0:
            failed_samples.append({"path": src_path, "reason": "no_detection"})
            num_failed += 1
            if SAVE_FAILED:
                cv2.imwrite(dst_path,
                            cv2.resize(img_bgr, (ROI_SIZE, ROI_SIZE)))
        else:
            roi_out = cv2.resize(roi_bgr, (ROI_SIZE, ROI_SIZE))
            # Save as RGB
            cv2.imwrite(dst_path,
                        cv2.cvtColor(roi_out, cv2.COLOR_BGR2RGB))
            num_success += 1

        pbar.set_postfix(success=num_success, failed=num_failed)

    # Save failure report
    with open(FAILED_JSON_PATH, "w") as f:
        json.dump(failed_samples, f, indent=2)

    total = num_success + num_failed
    print(f"\nDone.")
    print(f"  Success : {num_success} / {total}  "
          f"({100*num_success/max(total,1):.1f}%)")
    print(f"  Failed  : {num_failed} / {total}  "
          f"({100*num_failed/max(total,1):.1f}%)")
    print(f"  Failures saved to : {FAILED_JSON_PATH}")


if __name__ == "__main__":
    main()
