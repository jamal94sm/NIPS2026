"""
Palm ROI Extraction — Python translation of the full MATLAB pipeline
=====================================================================

Original MATLAB pipeline (two separate scripts):
  1. MarkToolForPalmprintPoint  — human GUI that manually marks finger-gap
                                   points A & B on each image and saves them
                                   to a paired .mat file.
  2. generateROI.m              — reads the .mat points, runs the geometric
                                   transform, and crops the ROI.

This script merges both steps into one, operating directly on JPGs:
  Step 1 — Detect finger-gap points A and B automatically from the image
            using YCrCb skin segmentation + convexity defects (replaces the
            manual .mat annotation).
  Step 2 — Run the exact generateROI.m geometric algorithm on those points
            (pad → scale → rotate → re-project → crop).

No intermediate files are written. Each image is processed end-to-end.

Filename format (MPDv2 / MPD)
------------------------------
  {id}_{session}_{device}_{hand}_{iter}.jpg
  e.g.  009_2_h_l_10.jpg
"""

import math
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ============================================================
# PATHS  — edit these two lines
# ============================================================
SRC_ROOT = "/home/pai-ng/Jamal/MPDv2"       # source directory
DST_ROOT = "/home/pai-ng/Jamal/MPDv2-ROI"   # output directory
ROI_SIZE = 224                               # output side length in pixels
SAVE_FAILED = False  # ← Set to True to save fallback ROIs, False to skip them

# ============================================================
# STEP 1 — DETECT FINGER-GAP POINTS A AND B
# (replaces the .mat annotation created by MarkToolForPalmprintPoint)
#
# Point A = midpoint of the gap between index and middle fingers
# Point B = midpoint of the gap between ring and little fingers
# (as defined in Fig. 1 of the DeepMPV paper)
# ============================================================

def detect_ab_points(image_bgr, hand_side):
    """
    Automatically locate finger-gap points A and B.

    Algorithm
    ---------
    1. YCrCb skin segmentation → morphological clean-up → largest contour.
    2. Convexity defects of the hand contour → filter by depth and position
       to keep only the 2–4 inter-finger valley points.
    3. Sort surviving defect points left→right by x-coordinate.
    4. Assign A / B based on hand side (read from filename):
         right hand : defects are ordered little→index  (leftmost = B, rightmost = A)
         left  hand : defects are ordered index→little  (leftmost = A, rightmost = B)

    Parameters
    ----------
    image_bgr : np.ndarray   BGR image
    hand_side : str          'l' or 'r'  (from the filename)

    Returns
    -------
    point_a, point_b : (x, y) int tuples  —  or  (None, None) on failure
    """
    h, w = image_bgr.shape[:2]

    # --- 1. Skin mask (YCrCb is robust across illuminations) ----------
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    mask  = cv2.inRange(ycrcb,
                        np.array([0,   133,  77], dtype=np.uint8),
                        np.array([255, 173, 127], dtype=np.uint8))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

    # --- 2. Largest contour = hand ------------------------------------
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    contour = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.04 * h * w:
        return None, None

    # --- 3. Convexity defects (inter-finger valleys) ------------------
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return None, None
    defects = cv2.convexityDefects(contour, hull_idx)
    if defects is None:
        return None, None

    # Keep defects whose depth > 3 % of image diagonal and that are
    # in the upper 85 % of the image (exclude wrist area)
    diag      = math.hypot(h, w)
    min_depth = 0.03 * diag
    valleys   = []
    for i in range(defects.shape[0]):
        _, _, f, d = defects[i, 0]
        depth = d / 256.0
        pt    = tuple(contour[f][0])
        if depth > min_depth and pt[1] < 0.85 * h:
            valleys.append((pt, depth))

    if len(valleys) < 2:
        return None, None

    # Take the 4 deepest, sort left→right
    valleys.sort(key=lambda v: -v[1])
    top_pts = sorted([v[0] for v in valleys[:4]], key=lambda p: p[0])

    # --- 4. Assign A and B -------------------------------------------
    # When the palm faces the camera:
    #   right hand fingers run little(left) → index(right)
    #     → leftmost valley  = ring–little gap  = B
    #     → rightmost valley = index–middle gap = A
    #   left hand fingers run index(left) → little(right)
    #     → leftmost valley  = index–middle gap = A
    #     → rightmost valley = ring–little gap  = B
    if hand_side == 'r':
        point_b = top_pts[0]    # ring–little (leftmost)
        point_a = top_pts[-1]   # index–middle (rightmost)
    else:
        point_a = top_pts[0]    # index–middle (leftmost)
        point_b = top_pts[-1]   # ring–little (rightmost)

    return point_a, point_b


# ============================================================
# STEP 2 — GENERATE ROI  (direct translation of generateROI.m)
# ============================================================

def _vec_rotate(x, y, angle_deg):
    """
    MATLAB:  x1 =  x*cos(r) + y*sin(r)
             y1 = -x*sin(r) + y*cos(r)
    """
    r = math.radians(angle_deg)
    c, s = math.cos(r), math.sin(r)
    return x * c + y * s, -x * s + y * c


def _rotation_angle(x1, y1, x2, y2):
    """
    MATLAB:
        vec = normalise([x2-x1, y2-y1])
        if vec(2) < 0 : angle = acosd(-vec(1)) - 180
        else           : angle = abs(acosd(-vec(1)) - 180)
    """
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0
    vx, vy = dx / length, dy / length
    angle  = math.degrees(math.acos(max(-1.0, min(1.0, -vx)))) - 180.0
    if vy >= 0:
        angle = abs(angle)
    return angle


def _make_square(image):
    """
    MATLAB:
        if wid > len : img(len+1:wid, :, :) = 255
        else         : img(:, wid+1:len, :) = 255
    """
    h, w = image.shape[:2]
    if w == h:
        return image.copy()
    edge   = max(h, w)
    square = np.full((edge, edge, image.shape[2]), 255, dtype=image.dtype)
    square[:h, :w] = image
    return square


def _rotate_loose(image, angle_deg):
    """
    MATLAB: imrotate(img, angle, 'bilinear', 'loose')
    Expands canvas so nothing is clipped.
    cv2 convention is clockwise-positive → negate angle.
    """
    h, w   = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    rad    = math.radians(angle_deg)
    ca, sa = abs(math.cos(rad)), abs(math.sin(rad))
    new_w  = int(h * sa + w * ca)
    new_h  = int(h * ca + w * sa)
    M      = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += new_h / 2.0 - cy
    return cv2.warpAffine(image, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def generate_roi(image, point_a, point_b):
    """
    Direct translation of generateROI.m.

    MATLAB steps reproduced here line-by-line
    ------------------------------------------
    edge = max(len, wid)
    img  = pad-to-square(origin)

    labelMarks = marks - 0.5          % 1-based → pixel centres
    x1 = labelMarks(slot1,1) - edge/2
    y1 = labelMarks(slot1,2) - edge/2
    x2 = labelMarks(slot2,1) - edge/2
    y2 = labelMarks(slot2,2) - edge/2

    sideLen = norm([x2-x1, y2-y1])
    angle   = rotation_angle(vec)

    temp   = imrotate(img, angle, 'bilinear', 'loose')
    mEdge  = size(temp, 1)

    [x1,y1] = vecrotate(x1, y1, angle);   x1 += mEdge/2 ...
    x0 = (x1+x2)/2;  y0 = (y1+y2)/2

    ROI = imcrop(temp, [x0-sideLen*7/12,  y0+sideLen*1/6,
                         sideLen*7/6,       sideLen*7/6])

    Parameters
    ----------
    image   : BGR np.ndarray
    point_a : (x, y)  index–middle finger gap
    point_b : (x, y)  ring–little finger gap

    Returns
    -------
    roi : np.ndarray or None
    """
    h, w = image.shape[:2]
    edge = max(h, w)

    # pad to square
    img = _make_square(image)

    # In MATLAB the marks came from the .mat file (1-based pixel coords).
    # Here point_a / point_b are already 0-based pixel coords from OpenCV,
    # which is equivalent to MATLAB's marks - 0.5 ≈ marks - 1.
    # We subtract edge/2 to centre-on-origin, matching the MATLAB code.
    x1 = point_a[0] - edge / 2.0
    y1 = point_a[1] - edge / 2.0
    x2 = point_b[0] - edge / 2.0
    y2 = point_b[1] - edge / 2.0

    side_len = math.hypot(x2 - x1, y2 - y1)
    if side_len == 0:
        return None

    angle   = _rotation_angle(x1, y1, x2, y2)
    rotated = _rotate_loose(img, angle)
    m_edge  = rotated.shape[0]

    # vecrotate then shift back
    rx1, ry1 = _vec_rotate(x1, y1, angle)
    rx2, ry2 = _vec_rotate(x2, y2, angle)
    rx1 += m_edge / 2.0;  ry1 += m_edge / 2.0
    rx2 += m_edge / 2.0;  ry2 += m_edge / 2.0

    x0 = (rx1 + rx2) / 2.0
    y0 = (ry1 + ry2) / 2.0

    # imcrop(temp, [x0-sideLen*7/12, y0+sideLen*1/6, sideLen*7/6, sideLen*7/6])
    cx    = int(round(x0 - side_len * 7.0 / 12.0))
    cy    = int(round(y0 + side_len * 1.0 / 6.0))
    csize = int(round(side_len * 7.0 / 6.0))

    if (cx < 0 or cy < 0
            or cx + csize > rotated.shape[1]
            or cy + csize > rotated.shape[0]
            or csize <= 0):
        return None

    return rotated[cy: cy + csize, cx: cx + csize]


# ============================================================
# FILENAME PARSER
# ============================================================

def parse_mpd_filename(fname):
    """
    Accepts:
      5-part MPDv2/MPD : {id}_{session}_{device}_{hand}_{iter}.jpg
                         e.g. 009_2_h_l_10.jpg
      4-part MPDv2 alt : {id}_{session}_{hand}_{iter}.jpg
                         e.g. 001_1_l_01.jpg
    Returns metadata dict or None.
    """
    stem  = os.path.splitext(fname)[0]
    parts = stem.split("_")

    if len(parts) == 5:
        subj_id, session, device, hand, iteration = parts
        if (len(subj_id) == 3 and subj_id.isdigit()
                and session in ("1", "2")
                and device in ("h", "m")
                and hand in ("l", "r")
                and len(iteration) == 2 and iteration.isdigit()):
            return dict(id=subj_id, session=session, device=device,
                        hand=hand, iteration=iteration)

    if len(parts) == 4:
        subj_id, session, hand, iteration = parts
        if (len(subj_id) == 3 and subj_id.isdigit()
                and session in ("1", "2")
                and hand in ("l", "r")
                and len(iteration) == 2 and iteration.isdigit()):
            return dict(id=subj_id, session=session,
                        hand=hand, iteration=iteration)

    return None


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    os.makedirs(DST_ROOT, exist_ok=True)

    all_images = []
    for root, _, files in os.walk(SRC_ROOT):
        for f in sorted(files):
            if not f.lower().endswith(".jpg"):
                continue
            parsed = parse_mpd_filename(f)
            if parsed:
                all_images.append((os.path.join(root, f), parsed))

    print(f"Found {len(all_images)} images.")

    num_success  = 0
    num_fallback = 0
    report_rows  = []

    pbar = tqdm(all_images, desc="Extracting palm ROIs")
    for src_path, meta in pbar:
        rel_path = os.path.relpath(src_path, SRC_ROOT)
        stem     = os.path.splitext(os.path.basename(src_path))[0]
        dst_dir  = os.path.join(DST_ROOT, os.path.dirname(rel_path))
        dst_path = os.path.join(dst_dir, stem + "_ROI.jpeg")
        os.makedirs(dst_dir, exist_ok=True)

        # Load image
        try:
            img_rgb = np.array(Image.open(src_path).convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            num_fallback += 1
            report_rows.append({**meta, "path": rel_path, "status": "load_error"})
            pbar.set_postfix(success=num_success, fallback=num_fallback, refresh=True)
            continue

        # Step 1: detect A and B directly from the image
        point_a, point_b = detect_ab_points(img_bgr, meta["hand"])

        # Step 2: run generateROI.m algorithm
        roi_bgr = None
        if point_a is not None and point_b is not None:
            roi_bgr = generate_roi(img_bgr, point_a, point_b)

        if roi_bgr is None or roi_bgr.size == 0:
            status = "fallback"
            num_fallback += 1
            if SAVE_FAILED:
                roi_bgr = cv2.resize(img_bgr, (ROI_SIZE, ROI_SIZE))
                cv2.imwrite(dst_path, roi_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
            report_rows.append({**meta, "path": rel_path, "status": status})
        else:
            roi_bgr = cv2.resize(roi_bgr, (ROI_SIZE, ROI_SIZE))
            status  = "ok"
            num_success += 1
            cv2.imwrite(dst_path, roi_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
            report_rows.append({**meta, "path": rel_path, "status": status})

        pbar.set_postfix(success=num_success, fallback=num_fallback, refresh=True)

    print(f"\nDone.  Success: {num_success},  Fallback: {num_fallback}")


if __name__ == "__main__":
    main()
