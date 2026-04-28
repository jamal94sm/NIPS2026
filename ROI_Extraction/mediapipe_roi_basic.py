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
# PATHS & DATASET  — edit these lines
# ============================================================
# DATASET choices:
#   "generic"  → flat or arbitrary folder layout (original behaviour)
#   "xjtu-up"  → {device}/{condition}/{hand}_{id}/*.JPG
DATASET = "xjtu-up"

SRC_ROOT         = "/home/pai-ng/Jamal/CASIA_Palmprint_V1"
DST_ROOT         = "/home/pai-ng/Jamal/CASIA_mediapipe_roi"

# XJTU-UP paths (only used when DATASET == "xjtu-up")
XJTU_SRC_ROOT    = "/mnt/data/FingerprintDatasets/Combined/combineddataset/XJTU-UP"
XJTU_DST_ROOT    = "/home/pai-ng/Jamal/XJTU_mediapipe_roi"

MODEL_PATH       = "/home/pai-ng/Jamal/NIPS2026/ROI_Extraction/hand_landmarker.task"
ROI_SIZE         = 224
SAVE_FAILED      = False
FAILED_JSON_PATH = os.path.join(
    XJTU_DST_ROOT if DATASET == "xjtu-up" else DST_ROOT,
    "failed_samples.json")

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
    handedness = detection_result.handedness[0][0].category_name

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

    rect = (C, (side, side), angle)
    M    = cv2.getRotationMatrix2D(C, angle, 1.0)
    rot  = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    roi  = cv2.getRectSubPix(rot, (int(side), int(side)), C)
    box  = cv2.boxPoints(rect).astype(np.int32)
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
    thumb    = idx(2)

    try:
        C = _calculate_point_c(roi_mid1, roi_mid2, thumb)
    except ValueError:
        return None, None, None

    roi, box = _extract_roi(image_bgr, roi_mid1, roi_mid2, C, thumb, hand_type)

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
# Filename Parser  (generic mode only)
# ============================================================

def parse_filename(fname):
    """
    MPDv2   : {subject}_{session}_{device}_{hand}_{iter}.jpg
    Scanner : {id}_{session}_{hand}_{color}.jpg
    Generic : anything else
    """
    stem  = os.path.splitext(fname)[0]
    parts = stem.split("_")

    if len(parts) == 5 and parts[2].lower() in ("h", "m"):
        side = "Left" if parts[3].lower() == "l" else "Right"
        return dict(id=parts[0], session=parts[1],
                    device=parts[2], hand=side, iter=parts[4])

    if len(parts) == 4 and parts[2] in ("Left", "Right"):
        return dict(id=parts[0], session=parts[1],
                    hand=parts[2], color=parts[3])

    return dict(id=parts[0] if parts else stem,
                session="", hand="unknown")


# ============================================================
# XJTU-UP Image Collector
# ============================================================

def collect_xjtu_up_images(src_root):
    """
    Walk the XJTU-UP directory tree and collect all images.

    Tree layout:
      {src_root}/
        {device}/          e.g. huawei, iPhone
          {condition}/     e.g. Flash, Nature
            {hand}_{id}/   e.g. L_001, R_100  ← identity folder
              *.JPG        individual images

    Identity key  : "{hand}_{id}"   e.g. "L_001", "R_100"
    Hand          : "Left" if starts with "L", else "Right"
    Metadata dict : {id, hand, device, condition, identity_folder}

    Returns list of (src_path, meta) tuples.
    """
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    images   = []

    for device in sorted(os.listdir(src_root)):
        device_dir = os.path.join(src_root, device)
        if not os.path.isdir(device_dir):
            continue

        for condition in sorted(os.listdir(device_dir)):
            cond_dir = os.path.join(device_dir, condition)
            if not os.path.isdir(cond_dir):
                continue

            for identity_folder in sorted(os.listdir(cond_dir)):
                id_dir = os.path.join(cond_dir, identity_folder)
                if not os.path.isdir(id_dir):
                    continue

                # identity_folder is like "L_001" or "R_100"
                parts = identity_folder.split("_")
                if len(parts) < 2:
                    continue
                hand_code = parts[0].upper()         # "L" or "R"
                subject_id = "_".join(parts[1:])     # "001" or "100"
                hand = "Left" if hand_code == "L" else "Right"

                for fname in sorted(os.listdir(id_dir)):
                    if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                        continue
                    src_path = os.path.join(id_dir, fname)
                    meta = dict(
                        id              = subject_id,
                        hand            = hand,
                        device          = device,
                        condition       = condition,
                        identity_folder = identity_folder,  # e.g. "R_100"
                    )
                    images.append((src_path, meta))

    return images


# ============================================================
# Generic Image Collector
# ============================================================

def collect_generic_images(src_root):
    """Walk src_root recursively and collect all images."""
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    images   = []
    for root, _, files in os.walk(src_root):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in IMG_EXTS:
                continue
            meta = parse_filename(f)
            images.append((os.path.join(root, f), meta))
    return images


# ============================================================
# Main Loop
# ============================================================

def main():
    dataset = DATASET.strip().lower().replace("-", "").replace("_", "")

    if dataset == "xjtuu" or dataset == "xjtuup":
        src_root = XJTU_SRC_ROOT
        dst_root = XJTU_DST_ROOT
        print(f"Dataset mode   : XJTU-UP")
        print(f"Source         : {src_root}")
        print(f"Destination    : {dst_root}")
        all_images = collect_xjtu_up_images(src_root)
    else:
        src_root = SRC_ROOT
        dst_root = DST_ROOT
        print(f"Dataset mode   : generic")
        print(f"Source         : {src_root}")
        print(f"Destination    : {dst_root}")
        all_images = collect_generic_images(src_root)

    os.makedirs(dst_root, exist_ok=True)

    print(f"\nFound {len(all_images)} images.")
    if len(all_images) == 0:
        print(f"  ERROR: No images found in {src_root}")
        return

    print("  First few paths found:")
    for src_path, _ in all_images[:5]:
        print(f"    {src_path}")
    print()

    # Print XJTU-UP identity summary
    if dataset in ("xjtuu", "xjtuup"):
        from collections import defaultdict, Counter
        id_counts = defaultdict(int)
        for _, meta in all_images:
            id_counts[meta["identity_folder"]] += 1
        total_ids = len(id_counts)
        counts    = list(id_counts.values())
        print(f"  XJTU-UP summary:")
        print(f"    Identity folders : {total_ids}")
        print(f"    Images per ID    : min={min(counts)}  max={max(counts)}  "
              f"mean={sum(counts)/len(counts):.1f}")
        devices    = sorted({m["device"]    for _, m in all_images})
        conditions = sorted({m["condition"] for _, m in all_images})
        print(f"    Devices          : {', '.join(devices)}")
        print(f"    Conditions       : {', '.join(conditions)}")
        print()

    num_success    = 0
    num_failed     = 0
    failed_samples = []

    pbar = tqdm(all_images, desc="Processing images")
    for src_path, meta in pbar:
        # Preserve relative path from src_root → dst_root
        rel_path = os.path.relpath(src_path, src_root)
        dst_path = os.path.join(dst_root, rel_path)
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
            failed_samples.append({"path": src_path,
                                   "reason": "no_detection",
                                   "meta": meta})
            num_failed += 1
            if SAVE_FAILED:
                cv2.imwrite(dst_path,
                            cv2.resize(img_bgr, (ROI_SIZE, ROI_SIZE)))
        else:
            roi_out = cv2.resize(roi_bgr, (ROI_SIZE, ROI_SIZE))
            cv2.imwrite(dst_path,
                        cv2.cvtColor(roi_out, cv2.COLOR_BGR2RGB))
            num_success += 1

        pbar.set_postfix(success=num_success, failed=num_failed)

    # Save failure report
    failed_json = os.path.join(dst_root, "failed_samples.json")
    with open(failed_json, "w") as f:
        json.dump(failed_samples, f, indent=2)

    total = num_success + num_failed
    print(f"\nDone.")
    print(f"  Success : {num_success} / {total}  "
          f"({100*num_success/max(total,1):.1f}%)")
    print(f"  Failed  : {num_failed} / {total}  "
          f"({100*num_failed/max(total,1):.1f}%)")
    print(f"  Failures saved to : {failed_json}")


if __name__ == "__main__":
    main()
