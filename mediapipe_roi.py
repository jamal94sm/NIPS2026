import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv
from PIL import Image
from tqdm import tqdm

# ============================================================
# PATHS  — edit these lines
# ============================================================
SRC_ROOT = "/home/pai-ng/Jamal/MPDv2"
DST_ROOT = "/home/pai-ng/Jamal/MPDv2_mediapipe_roi"
MODEL_PATH = "hand_landmarker.task"  # Ensure this file is in the same folder
ROI_SIZE = 224
SAVE_FAILED = False  # ← Set to True to save fallback ROIs, False to skip them

# ============================================================
# NEW: Initialize MediaPipe Tasks Detector
# ============================================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.2,
    running_mode=vision.RunningMode.IMAGE
)
# Create detector once to reuse in the loop
detector = vision.HandLandmarker.create_from_options(options)

def _run_mp_hands_new(image_bgr):
    """
    Updated for MediaPipe 0.10.30+ using Tasks API
    """
    # Convert BGR to MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    
    # Run detection
    detection_result = detector.detect(mp_image)
    
    if not detection_result.hand_landmarks:
        return None, None

    # Extract landmarks for the first hand detected
    lm_list = detection_result.hand_landmarks[0]
    handedness = detection_result.handedness[0][0].category_name # 'Left' or 'Right'

    h, w = image_bgr.shape[:2]
    # Tasks API uses normalized coordinates (0.0 to 1.0) just like legacy
    pts = [(int(p.x * w), int(p.y * h)) for p in lm_list]
    
    return pts, handedness

# ============================================================
# ROI MATH HELPERS (Unchanged logic, just updated call)
# ============================================================

def _midpoints(pairs):
    return [((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2) for p1, p2 in pairs]

def _calculate_point_c(m1, m2, thumb):
    m1, m2, thumb = map(np.asarray, (m1, m2, thumb))
    O = (m1 + m2) / 2.0
    AB = m2 - m1
    L = np.linalg.norm(AB)
    if L == 0: raise ValueError("Midpoints coincide")
    ABu = AB / L
    perp = np.array([-ABu[1], ABu[0]])
    cross_z = ABu[0] * (thumb - O)[1] - ABu[1] * (thumb - O)[0]
    if cross_z < 0: perp = -perp
    C = O + 1.8 * L * perp
    return int(C[0]), int(C[1])

def _extract_roi(img, mid1, mid2, C, thumb, hand_type):
    vec = np.array(mid2) - np.array(mid1)
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    C = (int(C[0]), int(C[1]))
    if hand_type.lower() == "right":
        if np.dot(vec, np.array(thumb) - np.array(C)) > 0: angle += 180
    else:
        if np.dot(vec, np.array(thumb) - np.array(C)) < 0: angle += 180
    side = np.linalg.norm(vec) * 2.5
    rect = (C, (side, side), angle)
    M = cv2.getRotationMatrix2D(C, angle, 1.0)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    roi = cv2.getRectSubPix(rot, (int(side), int(side)), C)
    box = cv2.boxPoints(rect).astype(int)
    return roi, box

def extract_palm_roi(image_bgr):
    # Updated call to the new detection function
    lms, hand_type = _run_mp_hands_new(image_bgr)
    if lms is None:
        return None, None, None

    idx = lambda i: lms[i]
    mids4 = _midpoints([(idx(17), idx(18)), (idx(14), idx(13)), (idx(10), idx(9)), (idx(6), idx(5))])
    adj = _midpoints([(mids4[0], mids4[1]), (mids4[1], mids4[2]), (mids4[2], mids4[3])])
    roi_mid1 = ((adj[0][0] + adj[1][0]) / 2, (adj[0][1] + adj[1][1]) / 2)
    roi_mid2 = ((adj[1][0] + adj[2][0]) / 2, (adj[1][1] + adj[2][1]) / 2)

    thumb = idx(2)
    C = _calculate_point_c(roi_mid1, roi_mid2, thumb)
    roi, box = _extract_roi(image_bgr, roi_mid1, roi_mid2, C, thumb, hand_type)

    ann = image_bgr.copy()
    for x, y in lms: cv2.circle(ann, (x, y), 3, (0, 255, 0), -1)
    cv2.polylines(ann, [box], True, (0, 255, 0), 2)
    cv2.circle(ann, C, 6, (0, 0, 255), -1)

    return roi, ann, hand_type

# ============================================================
# Filename Parser & Main Loop (Remains largely the same)
# ============================================================

def parse_mpd_filename(fname):
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")
    if len(parts) != 5: return None
    subj_id, session, device, hand, iteration = parts
    if (len(subj_id) == 3 and subj_id.isdigit() and session in ("1", "2")
            and device in ("h", "m") and hand in ("l", "r")
            and len(iteration) == 2 and iteration.isdigit()):
        return dict(id=subj_id, session=session, device=device, hand=hand, iteration=iteration)
    return None

def main():
    os.makedirs(DST_ROOT, exist_ok=True)
    all_images = []
    for root, _, files in os.walk(SRC_ROOT):
        for f in sorted(files):
            if not f.lower().endswith(".jpg"): continue
            parsed = parse_mpd_filename(f)
            if parsed: all_images.append((os.path.join(root, f), parsed))

    num_success = 0
    num_fallback = 0
    report_rows = []

    pbar = tqdm(all_images, desc="Processing MPDv2")
    for src_path, meta in pbar:
        rel_path = os.path.relpath(src_path, SRC_ROOT)
        dst_path = os.path.join(DST_ROOT, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        try:
            img_rgb = np.array(Image.open(src_path).convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            num_fallback += 1
            pbar.set_postfix(success=num_success, failed=num_fallback)
            continue

        roi_bgr, _, _ = extract_palm_roi(img_bgr)

        if roi_bgr is None or roi_bgr.size == 0:
            status = "failed"
            num_fallback += 1
            if SAVE_FAILED:
                roi_bgr = cv2.resize(img_bgr, (ROI_SIZE, ROI_SIZE))
                cv2.imwrite(dst_path, roi_bgr)
            report_rows.append({**meta, 'path': rel_path, 'status': status})
        else:
            roi_bgr = cv2.resize(roi_bgr, (ROI_SIZE, ROI_SIZE))
            status = "ok"
            num_success += 1
            cv2.imwrite(dst_path, roi_bgr)
            report_rows.append({**meta, 'path': rel_path, 'status': status})

        pbar.set_postfix(success=num_success, failed=num_fallback)

    print(f"\nDone. Success: {num_success}, Fallback: {num_fallback}")

if __name__ == "__main__":
    main()
