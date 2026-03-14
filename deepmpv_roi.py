"""
Palm ROI Extraction
===================
Python translation of the MATLAB `generateROI.m` script from:

  "Pay by Showing Your Palm: A Study of Palmprint Verification on Mobile Platforms"
  IEEE ICME 2019  —  https://ieeexplore.ieee.org/document/8785020

Algorithm
---------
For each image the paired .mat annotation file contains:
  - marks : (N, 2) array of (x, y) keypoint coordinates (1-indexed, MATLAB)
  - slots : indices of the two landmarks that define the palm axis

Steps:
  1. Pad the image to a square with white.
  2. Scale marks to the square coordinate system.
  3. Compute the rotation angle that makes the palm axis horizontal.
  4. Rotate image (expanding canvas, bilinear — equiv. to MATLAB 'loose').
  5. Re-project landmarks into the rotated frame via vec_rotate().
  6. Crop a (7/6 * sideLen) square at offset (x_mid − 7/12·L, y_mid + 1/6·L).
  7. Save to DST_ROOT preserving relative folder structure.
"""

import math
import os

import cv2
import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm

# ============================================================
# PATHS  — edit these lines
# ============================================================
SRC_ROOT = "/home/pai-ng/Jamal/MPDv2"               # source directory
DST_ROOT = "/home/pai-ng/Jamal/MPDv2-ROI"           # output directory
ROI_SIZE = 128                         # output side length in pixels

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ============================================================
# GEOMETRY HELPERS
# ============================================================

def vec_rotate(x, y, angle_deg):
    """
    Rotate point (x, y) around the origin by angle_deg degrees.

    Matches the MATLAB nested helper::
        x1 =  x*cos(r) + y*sin(r)
        y1 = -x*sin(r) + y*cos(r)
    """
    r = math.radians(angle_deg)
    cos_r, sin_r = math.cos(r), math.sin(r)
    return x * cos_r + y * sin_r, -x * sin_r + y * cos_r


def rotation_angle(x1, y1, x2, y2):
    """
    Angle (degrees) that rotates the vector p1->p2 to horizontal.

    Mirrors the MATLAB convention::
        vec = normalise([x2-x1, y2-y1])
        if vec_y < 0 : angle = acos(-vec_x) - 180   (negative)
        else          : angle = |acos(-vec_x) - 180| (positive / zero)
    """
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0
    vx = dx / length
    vy = dy / length

    angle = math.degrees(math.acos(max(-1.0, min(1.0, -vx)))) - 180.0
    if vy >= 0:
        angle = abs(angle)
    return angle


# ============================================================
# IMAGE HELPERS
# ============================================================

def make_square(image):
    """
    Pad the shorter dimension with white so the image becomes square.

    Replicates MATLAB::
        if wid > len : img(len+1:wid, :, :) = 255
        else         : img(:, wid+1:len, :) = 255
    """
    h, w = image.shape[:2]
    if w == h:
        return image.copy()
    edge = max(h, w)
    square = np.full((edge, edge, image.shape[2]), 255, dtype=image.dtype)
    square[:h, :w] = image
    return square


def rotate_image(image, angle_deg):
    """
    Rotate image by angle_deg degrees with bilinear interpolation,
    expanding the canvas so no content is clipped.

    Equivalent to MATLAB's imrotate(..., 'bilinear', 'loose').
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    rad = math.radians(angle_deg)
    cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # cv2 is clockwise-positive, MATLAB CCW-positive -> negate angle
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += new_h / 2.0 - cy

    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


# ============================================================
# ANNOTATION LOADER
# ============================================================

def load_mat_annotation(mat_path):
    """
    Load marks and slots from a .mat file, converting from MATLAB's
    1-based indexing to 0-based Python indexing.

    Returns
    -------
    marks : np.ndarray (N, 2)  float64,  0-based pixel (x, y)
    slots : np.ndarray (2,)    int,      0-based landmark indices
    """
    data = sio.loadmat(str(mat_path))

    marks = data["marks"].astype(float) - 1.0   # 1-based -> 0-based

    raw_slots = data["slots"].flatten().astype(int)
    slots = raw_slots[:2] - 1                   # 1-based -> 0-based

    return marks, slots


# ============================================================
# CORE ROI EXTRACTION  (mirrors generateROI.m step-by-step)
# ============================================================

def extract_roi(image, marks, slots):
    """
    Extract the palm ROI from image given annotated keypoints.

    Parameters
    ----------
    image  : np.ndarray  BGR image
    marks  : np.ndarray  (N, 2) float, 0-based pixel (x, y)
    slots  : array-like  two 0-based indices into marks

    Returns
    -------
    roi : np.ndarray or None
    """
    h, w = image.shape[:2]
    edge = max(h, w)

    # --- 1. Make square -----------------------------------------------
    square = make_square(image)

    # --- 2. Scale marks to the square coordinate system ---------------
    # MATLAB: labelMarks(:,1) = (marks-0.5) / wid * edge  (x-column)
    #         labelMarks(:,2) = (marks-0.5) / len * edge  (y-column)
    # Python marks are already 0-based (equivalent to MATLAB marks-1),
    # so we just scale without the extra -0.5 shift.
    lm = marks.copy()
    lm[:, 0] = lm[:, 0] / w * edge   # x
    lm[:, 1] = lm[:, 1] / h * edge   # y

    idx1, idx2 = int(slots[0]), int(slots[1])
    x1, y1 = lm[idx1, 0], lm[idx1, 1]
    x2, y2 = lm[idx2, 0], lm[idx2, 1]

    side_len = math.hypot(x2 - x1, y2 - y1)
    if side_len == 0:
        return None

    # --- 3. Rotation angle --------------------------------------------
    angle = rotation_angle(x1, y1, x2, y2)

    # --- 4. Rotate image (loose canvas) --------------------------------
    rotated = rotate_image(square, angle)
    m_edge = rotated.shape[0]

    # --- 5. Re-project landmarks into rotated frame -------------------
    # Centre on origin, rotate, shift back
    cx1, cy1 = x1 - edge / 2.0, y1 - edge / 2.0
    cx2, cy2 = x2 - edge / 2.0, y2 - edge / 2.0

    rx1, ry1 = vec_rotate(cx1, cy1, angle)
    rx2, ry2 = vec_rotate(cx2, cy2, angle)

    rx1 += m_edge / 2.0;  ry1 += m_edge / 2.0
    rx2 += m_edge / 2.0;  ry2 += m_edge / 2.0

    x0 = (rx1 + rx2) / 2.0   # mid-point x
    y0 = (ry1 + ry2) / 2.0   # mid-point y

    # --- 6. Crop window (matches MATLAB imcrop call exactly) ----------
    # imcrop(temp, [x0 - sideLen*7/12,  y0 + sideLen*1/6,
    #               sideLen*7/6,         sideLen*7/6])
    crop_x    = int(round(x0  - side_len * 7.0 / 12.0))
    crop_y    = int(round(y0  + side_len * 1.0 / 6.0))
    crop_size = int(round(side_len * 7.0 / 6.0))

    if (crop_x < 0 or crop_y < 0
            or crop_x + crop_size > rotated.shape[1]
            or crop_y + crop_size > rotated.shape[0]
            or crop_size <= 0):
        return None

    return rotated[crop_y: crop_y + crop_size,
                   crop_x: crop_x + crop_size]


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    os.makedirs(DST_ROOT, exist_ok=True)

    # Collect all images that have a paired .mat file
    all_images = []
    for root, _, files in os.walk(SRC_ROOT):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            mat_name = os.path.splitext(f)[0] + ".mat"
            mat_path = os.path.join(root, mat_name)
            if os.path.exists(mat_path):
                all_images.append((os.path.join(root, f), mat_path))

    num_success = 0
    num_fallback = 0
    report_rows = []

    for src_path, mat_path in tqdm(all_images, desc="Extracting palm ROIs"):
        # Preserve relative directory structure under DST_ROOT
        rel_path = os.path.relpath(src_path, SRC_ROOT)
        stem     = os.path.splitext(os.path.basename(src_path))[0]
        dst_dir  = os.path.join(DST_ROOT, os.path.dirname(rel_path))
        dst_path = os.path.join(dst_dir, stem + "_ROI.jpeg")
        os.makedirs(dst_dir, exist_ok=True)

        # Load image via PIL (consistent with reference script)
        try:
            img_rgb = np.array(Image.open(src_path).convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            num_fallback += 1
            report_rows.append({"path": rel_path, "status": "load_error"})
            continue

        # Load annotation
        try:
            marks, slots = load_mat_annotation(mat_path)
        except Exception:
            num_fallback += 1
            report_rows.append({"path": rel_path, "status": "mat_error"})
            continue

        # Extract ROI
        roi_bgr = extract_roi(img_bgr, marks, slots)

        if roi_bgr is None or roi_bgr.size == 0:
            roi_bgr = cv2.resize(img_bgr, (ROI_SIZE, ROI_SIZE))
            status = "fallback"
            num_fallback += 1
        else:
            roi_bgr = cv2.resize(roi_bgr, (ROI_SIZE, ROI_SIZE))
            status = "ok"
            num_success += 1

        cv2.imwrite(dst_path, roi_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
        report_rows.append({"path": rel_path, "status": status})

    print(f"\nDone.  Success: {num_success},  Fallback/skipped: {num_fallback}")


if __name__ == "__main__":
    main()
