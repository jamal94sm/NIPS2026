
import cv2
import numpy as np
import mediapipe as mp

# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _run_mp_hands(image, min_det_conf=0.2):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_det_conf
    ) as hands:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None, None

        lm = res.multi_hand_landmarks[0]
        handedness = (
            res.multi_handedness[0].classification[0].label
            if res.multi_handedness else "Unknown"
        )

        h, w = image.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
        return pts, handedness


def _midpoints(pairs):
    return [((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2) for p1, p2 in pairs]


def _calculate_point_c(m1, m2, thumb):
    m1, m2, thumb = map(np.asarray, (m1, m2, thumb))

    O = (m1 + m2) / 2.0
    AB = m2 - m1
    L = np.linalg.norm(AB)
    if L == 0:
        raise ValueError("Midpoints coincide")

    ABu = AB / L
    perp = np.array([-ABu[1], ABu[0]])

    # ---- NumPy 2.0–safe 2D cross product (scalar z-component)
    cross_z = ABu[0] * (thumb - O)[1] - ABu[1] * (thumb - O)[0]
    if cross_z < 0:
        perp = -perp

    C = O + 1.8 * L * perp
    return int(C[0]), int(C[1])  # ensure Python ints


def _extract_roi(img, mid1, mid2, C, thumb, hand_type):
    vec = np.array(mid2) - np.array(mid1)
    angle = np.degrees(np.arctan2(vec[1], vec[0]))

    C = (int(C[0]), int(C[1]))  # OpenCV-safe center

    if hand_type.lower() == "right":
        if np.dot(vec, np.array(thumb) - np.array(C)) > 0:
            angle += 180
    else:  # left / unknown
        if np.dot(vec, np.array(thumb) - np.array(C)) < 0:
            angle += 180

    side = np.linalg.norm(vec) * 2.5
    rect = (C, (side, side), angle)

    M = cv2.getRotationMatrix2D(C, angle, 1.0)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    roi = cv2.getRectSubPix(rot, (int(side), int(side)), C)

    box = cv2.boxPoints(rect).astype(int)
    return roi, box

# ----------------------------------------------------------------------
# Public function
# ----------------------------------------------------------------------

def extract_palm_roi(image_bgr):
    """
    Parameters
    ----------
    image_bgr : np.ndarray
        BGR image.

    Returns
    -------
    roi_bgr, annotated_bgr, hand_type
        If landmarks fail → returns (None, None, None)
    """
    lms, hand_type = _run_mp_hands(image_bgr)
    if lms is None:
        return None, None, None

    idx = lambda i: lms[i]

    mids4 = _midpoints([
        (idx(17), idx(18)),
        (idx(14), idx(13)),
        (idx(10), idx(9)),
        (idx(6),  idx(5))
    ])

    adj = _midpoints([
        (mids4[0], mids4[1]),
        (mids4[1], mids4[2]),
        (mids4[2], mids4[3])
    ])

    roi_mid1 = ((adj[0][0] + adj[1][0]) / 2,
                (adj[0][1] + adj[1][1]) / 2)

    roi_mid2 = ((adj[1][0] + adj[2][0]) / 2,
                (adj[1][1] + adj[2][1]) / 2)

    thumb = idx(2)
    C = _calculate_point_c(roi_mid1, roi_mid2, thumb)

    roi, box = _extract_roi(image_bgr, roi_mid1, roi_mid2, C, thumb, hand_type)

    # Annotated image
    ann = image_bgr.copy()
    for x, y in lms:
        cv2.circle(ann, (x, y), 3, (0, 255, 0), -1)

    cv2.polylines(ann, [box], True, (0, 255, 0), 2)
    cv2.circle(ann, C, 6, (0, 0, 255), -1)

    return roi, ann, hand_type




import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ================================
# Paths
# ================================
SRC_ROOT = "/content/drive/MyDrive/CASIA-MS"
DST_ROOT = "/content/drive/MyDrive/CASIA-MS-ROI"

os.makedirs(DST_ROOT, exist_ok=True)


# ================================
# Collect all images
# ================================
all_images = []
for root, _, files in os.walk(SRC_ROOT):
    for f in files:
        if f.lower().endswith(".jpg"):
            all_images.append(os.path.join(root, f))

print(f"Total images found: {len(all_images)}")


# ================================
# Counters
# ================================
num_success = 0
num_failed = 0


# ================================
# Batch processing
# ================================
for src_path in tqdm(all_images):
    rel_path = os.path.relpath(src_path, SRC_ROOT)
    dst_path = os.path.join(DST_ROOT, rel_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # ---- Load image (RGB)
    img_rgb = np.array(Image.open(src_path).convert("RGB"))

    # ---- Convert to BGR for ROI extractor
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ---- Extract palm ROI (advanced method)
    roi_bgr, _, _ = extract_palm_roi(img_bgr)

    # ---- Check extraction result
    if roi_bgr is None:
        num_failed += 1
        roi_bgr = img_bgr  # fallback
        roi_bgr = cv2.resize(roi_bgr, (160, 160))
    else:
        num_success += 1
        roi_bgr = cv2.resize(roi_bgr, (160, 160))

    # ---- Save ROI
    cv2.imwrite(dst_path, roi_bgr)

