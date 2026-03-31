"""
ROI extraction: valley-point detection → rotation → crop → resize.
Translated from MATLAB findROI_fromShape.m and its util subfunctions.
"""

import numpy as np
from itertools import combinations
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from skimage.draw import polygon as sk_polygon
from skimage.measure import label as sklabel

import cv2

from utils import (
    smooth_ma, poly2mask, pdist_max, find_most_dist_points,
    triangle_angles, est_orient, find_neigh, num_cons_els,
    imrotate,
)


# ---------------------------------------------------------------------------
# CHECK-EXT-AREA (checkExtArea.m)
# ---------------------------------------------------------------------------

def check_ext_area(bw, central_pt, other_pt, param):
    """
    Walk from other_pt away from central_pt and check if the path stays
    inside the palm mask (bw=1).  Returns True if majority are inside.
    Equivalent to MATLAB checkExtArea.
    central_pt, other_pt: (x, y) = (col, row).
    """
    rp = param['rejectPoints']
    n_steps = rp['numStepsCheckExtArea']
    perc    = rp['percCheckExtArea']
    dist_red = 1.0 / n_steps

    dx = float(other_pt[0] - central_pt[0])
    dy = float(other_pt[1] - central_pt[1])

    inside = []
    for mult in range(1, n_steps + 1):
        nx = other_pt[0] + mult * dist_red * dx
        ny = other_pt[1] + mult * dist_red * dy

        if nx <= 1 or nx >= bw.shape[1] - 1:
            break
        if ny <= 1 or ny >= bw.shape[0] - 1:
            break

        val = bool(bw[int(round(ny)), int(round(nx))])
        inside.append(val)

    if not inside:
        return False
    return sum(inside) > len(inside) * perc


# ---------------------------------------------------------------------------
# EVAL-COND (evalCond.m)
# ---------------------------------------------------------------------------

def eval_cond(point, bw, param):
    """
    Test whether a point qualifies as a valley point via 3 neighbourhood
    ring conditions.  Equivalent to MATLAB evalCond.
    point: (x, y) = (col, row).
    """
    lp = param['localsearch']
    beta  = lp['beta']
    alpha = lp['alpha']
    mu    = lp['mu']

    for offset_a in range(0, 91, lp['stepAngle']):
        # --- Cond 1: exactly 1 zero in inner ring of 4 neighbours ----------
        n4 = find_neigh(bw, point, beta, offset_a, 4)
        cond1 = True if n4.size == 0 else (int(np.sum(n4 == 0)) == 1)

        # --- Cond 2: ≥1 zero in mid ring, max consecutive zeros ≤ 4 --------
        n8 = find_neigh(bw, point, beta + alpha, offset_a, 8)
        if n8.size == 0:
            cond2 = True
        else:
            nz8 = int(np.sum(n8 == 0))
            cond2 = nz8 >= 1 and num_cons_els(n8, 0) <= 4

        # --- Cond 3: 1–7 zeros in outer ring of 16 neighbours --------------
        n16 = find_neigh(bw, point, beta + alpha + mu, offset_a, 16)
        if n16.size == 0:
            cond3 = True
        else:
            nz16 = int(np.sum(n16 == 0))
            cond3 = 1 <= nz16 <= 7

        if cond1 and cond2 and cond3:
            return True, True, True

    return cond1, cond2, cond3


# ---------------------------------------------------------------------------
# LOCAL-SEARCH-VALLEYS (localSearchValleys.m)
# ---------------------------------------------------------------------------

def local_search_valleys(im, bw, shape_f, ind_peaks, param):
    """
    Refine valley-point indices by local neighbourhood search.
    shape_f: (N, 2) with col 0 = row, col 1 = col  (post-swap in MATLAB).
    Returns refined index array.
    """
    lp = param['localsearch']
    ind_new = list(ind_peaks)

    for ii, idx in enumerate(ind_peaks):
        half = int(lp['offset'] / 2)
        lo = max(0, idx - half)
        hi = min(len(shape_f) - 1, idx + half)

        candidates = []
        for s in range(lo, hi + 1, lp['stepSearch']):
            # shape_f col 0 = y (row), col 1 = x (col) after the MATLAB swap
            pt = (float(shape_f[s, 1]), float(shape_f[s, 0]))  # (x, y)
            if cdist([pt], [(float(shape_f[idx, 1]),
                              float(shape_f[idx, 0]))])[0, 0] > lp['maxDistance']:
                continue
            c1, c2, c3 = eval_cond(pt, bw, param)
            if c1 and c2 and c3:
                candidates.append(s)

        if candidates:
            ind_new[ii] = int(round(np.mean(candidates)))

    return ind_new


# ---------------------------------------------------------------------------
# DISCARD-OUTLIERS (discardOutliers.m)
# ---------------------------------------------------------------------------

def discard_outliers(sort_coord, im, bw, param):
    """
    From ≥3 candidate valley points, pick the best triplet.
    sort_coord: (N, 2) with col 0 = x (col), col 1 = y (row).
    Returns (best_3_points, error_code).  error_code = -1 on failure.
    """
    rp = param['rejectPoints']
    n  = len(sort_coord)
    all_idx = list(range(n))
    h, w = im.shape[:2]

    best_score = 1e9
    best_trio  = None

    for trio in combinations(all_idx, 3):
        pts = sort_coord[list(trio)]            # (3, 2): x, y
        pts_sorted = pts[np.lexsort((pts[:, 0], pts[:, 1]))]

        # CHECK 6a: triangle interior must be mostly inside palm
        bwt = poly2mask(pts_sorted[:, 0], pts_sorted[:, 1], h, w)
        num_z   = int(np.sum(bwt & ~bw))
        num_tot = int(np.sum(bwt))
        if num_tot == 0 or (num_z / num_tot) > rp['percBlackPixels']:
            continue

        # CHECK 6b: 2 of the 3 triangle angles must be < thAngle
        angles = triangle_angles(pts_sorted, fmt='d')
        if int(np.sum(angles < rp['thAngle'])) < 2:
            continue

        # CHECK 6c: extension beyond each outer point must be inside palm
        dm = cdist(pts_sorted, pts_sorted)
        central_idx = int(np.argmin(dm.mean(axis=1)))
        other_idx   = [i for i in range(3) if i != central_idx]
        central_pt  = pts_sorted[central_idx]

        ok_c = True
        for oi in other_idx:
            if not check_ext_area(bw, central_pt, pts_sorted[oi], param):
                ok_c = False
                break
        if not ok_c:
            continue

        # CHECK 6d: no point too close to image border
        bd = rp['minDistanceBorder']
        too_close = (
            np.any(pts_sorted[:, 1] < bd) or
            np.any(pts_sorted[:, 1] > h - bd) or
            np.any(pts_sorted[:, 0] < bd) or
            np.any(pts_sorted[:, 0] > w - bd)
        )
        if too_close:
            continue

        # CHECK 6e: minimise combined score = min mean dist + mean of 2 smallest angles
        score = float(dm.mean(axis=1).min()) + float(np.sort(angles)[:2].mean())
        if score < best_score:
            best_score = score
            best_trio  = list(trio)

    if best_trio is None:
        return sort_coord, -1

    return sort_coord[best_trio], 0


# ---------------------------------------------------------------------------
# FIND-ROI-SIZE (findROIsize.m)
# ---------------------------------------------------------------------------

def find_roi_size(sort_coord, param):
    """
    Compute ROI dimensions from inter-valley distances.
    sort_coord: (3, 2) with x, y columns.
    Returns (dist_valleys, roi_w, roi_h, x_offset).
    """
    rp = param['ROIsize']
    dist = float(pdist_max(sort_coord))
    roi_w = int(round(dist * rp['multX']))
    roi_h = int(round(dist * rp['multY']))
    x_off = int(round(roi_w / 2 + dist * rp['multOffset']))
    return dist, roi_w, roi_h, x_off


# ---------------------------------------------------------------------------
# CENTROID-IMAGE HELPERS (createCentroidImages.m)
# ---------------------------------------------------------------------------

def create_centroid_images(h_orig, w_orig, sort_coord, i_p1, i_p2):
    """
    Create zero images with a small marker at two key valley points.
    sort_coord col 0 = x (col), col 1 = y (row).
    Returns (zero_im1, zero_im2) with values 0 or 255.
    """
    def _mark(h, w, x, y):
        img = np.zeros((h, w), dtype=float)
        r = int(round(y))
        c = int(round(x))
        r0, r1 = max(0, r - 2), min(h, r + 3)
        c0, c1 = max(0, c - 2), min(w, c + 3)
        img[r0:r1, c0:c1] = 255.0
        return img

    z1 = _mark(h_orig, w_orig,
               sort_coord[i_p1, 0], sort_coord[i_p1, 1])
    z2 = _mark(h_orig, w_orig,
               sort_coord[i_p2, 0], sort_coord[i_p2, 1])
    return z1, z2


# ---------------------------------------------------------------------------
# COMPUTE-CENTER-ROI  (computeCenterROI.m)
# ---------------------------------------------------------------------------

def compute_center_roi(top1, top2, bottom1, bottom2, x_offset):
    """
    Compute centre of the ROI from the two reference landmark coordinates.
    top1, top2 = col, row arrays from the first centroid image.
    bottom1, bottom2 = col, row arrays from the second centroid image.
    """
    cx = int(round((np.mean(top1) + np.mean(bottom1)) / 2)) + x_offset
    cy = int(round(((min(top2) + max(top2)) / 2 + np.mean(bottom2)) / 2))
    return np.array([cx, cy])


# ---------------------------------------------------------------------------
# COMPUTE-RANGES-ROI  (computeRangesROI.m)
# ---------------------------------------------------------------------------

def compute_ranges_roi(new_c, roi_w, roi_h):
    x0 = new_c[0] - roi_w // 2 + 1
    x1 = new_c[0] + roi_w // 2
    y0 = new_c[1] - roi_h // 2 + 1
    y1 = new_c[1] + roi_h // 2
    return int(x0), int(x1), int(y0), int(y1)


# ---------------------------------------------------------------------------
# CHECK-INDEXES-ROI  (checkIndexesROI.m)
# ---------------------------------------------------------------------------

def check_indexes_roi(x0, x1, y0, y1, bw):
    """Return True if all ranges are within image bounds."""
    return (y0 > 0 and y1 <= bw.shape[0] and
            x0 > 0 and x1 <= bw.shape[1])


# ---------------------------------------------------------------------------
# COMPUTE-ROI  (computeROI.m)
# ---------------------------------------------------------------------------

def compute_roi(im, bw, z1, z2, grad, grad_refined,
                roi_w, roi_h, x_offset, param):
    """
    Rotate image, find crop centre from landmark images, extract ROI.
    Returns (roi_out, rot_palm, new_c, top1, top2, bottom1, bottom2, error).
    """
    angle = grad + 180 if grad_refined else grad

    rot_palm = imrotate(im,  -angle, interp='bilinear', crop=False)
    rot_bw   = imrotate(bw.astype(float), -angle, interp='nearest', crop=False) > 0.5
    rot_z1   = imrotate(z1,  -angle, interp='bilinear', crop=False)
    rot_z2   = imrotate(z2,  -angle, interp='bilinear', crop=False)

    # Find landmark positions after rotation (threshold smeared values)
    top2, top1     = np.where(rot_z1 > 127)   # (rows, cols) of marker 1
    bottom2, bottom1 = np.where(rot_z2 > 127) # (rows, cols) of marker 2

    if top1.size == 0 or bottom1.size == 0:
        return None, rot_palm, None, top1, top2, bottom1, bottom2, -1

    new_c = compute_center_roi(top1, top2, bottom1, bottom2, x_offset)
    x0, x1, y0, y1 = compute_ranges_roi(new_c, roi_w, roi_h)

    if not check_indexes_roi(x0, x1, y0, y1, rot_bw):
        return None, rot_palm, new_c, top1, top2, bottom1, bottom2, -1

    roi_out  = rot_palm[y0:y1, x0:x1]
    bw_patch = rot_bw[y0:y1, x0:x1]

    # Reject if ROI is mostly background
    pct_black = 1.0 - float(bw_patch.mean())
    if pct_black > param['ROIsize']['percBlackPixels']:
        return None, rot_palm, new_c, top1, top2, bottom1, bottom2, -1

    return roi_out, rot_palm, new_c, top1, top2, bottom1, bottom2, 0


# ---------------------------------------------------------------------------
# REFINE-GRAD  (refineGrad.m)
# ---------------------------------------------------------------------------

def refine_grad(im, bw, z1, z2, grad, roi_w, roi_h, x_offset, param):
    """
    Decide whether to add 180° to the orientation angle.
    Returns True if grad should be incremented by 180.
    """
    rot_bw = imrotate(bw.astype(float), -grad, interp='nearest', crop=False) > 0.5
    rot_z1 = imrotate(z1, -grad, interp='bilinear', crop=False)
    rot_z2 = imrotate(z2, -grad, interp='bilinear', crop=False)

    top2,    top1    = np.where(rot_z1 > 127)
    bottom2, bottom1 = np.where(rot_z2 > 127)

    if top1.size == 0 or bottom1.size == 0:
        return True

    new_c = compute_center_roi(top1, top2, bottom1, bottom2, x_offset)
    x0, x1, y0, y1 = compute_ranges_roi(new_c, roi_w, roi_h)

    if not check_indexes_roi(x0, x1, y0, y1, rot_bw):
        return True

    bw_patch = rot_bw[y0:y1, x0:x1]
    pct_black = 1.0 - float(bw_patch.mean())
    return pct_black > param['ROIsize']['percBlackPixels']


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT  (findROI_fromShape.m)
# ---------------------------------------------------------------------------

def find_roi_from_shape(im, bw, shape_f, centroid, param):
    """
    Detect valley points, estimate orientation, and extract the palm ROI.
    Equivalent to MATLAB findROI_fromShape.

    Parameters
    ----------
    im       : float (H, W[, 3]) image in [0, 1].
    bw       : bool (H, W) binary palm mask.
    shape_f  : (N, 2) contour points with shape_f[:,0]=x, shape_f[:,1]=y.
    centroid : (2,) [x, y] = [col, row].
    param    : parameter dict.

    Returns
    -------
    roi     : float array – extracted/resized ROI, or None on failure.
    error   : 0 on success, -1 on failure.
    results : dict with 'valley_points' and 'grad_refined' keys.
    """
    h_orig, w_orig = im.shape[:2]
    error_c = [0, 0]
    results = {}

    # ------------------------------------------------------------------
    # MATLAB swaps shapeF cols: new[:,0]=old[:,1]=y, new[:,1]=old[:,0]=x
    # So after swap: shape_f2[:,0] = row (y), shape_f2[:,1] = col (x)
    shape_f2 = np.column_stack([shape_f[:, 1], shape_f[:, 0]])  # (row, col)

    for _iteration in range(param['peakFind']['numIterCentroid']):

        # ---- Build distance-from-centroid array --------------------------
        # regionLine: [col(x), row(y), dist]
        cx, cy = float(centroid[0]), float(centroid[1])
        dists = np.sqrt((shape_f2[:, 1] - cx) ** 2 +
                        (shape_f2[:, 0] - cy) ** 2)
        region_line = np.column_stack([shape_f2[:, 1],   # col = x
                                        shape_f2[:, 0],   # row = y
                                        dists])

        # ---- Smooth distances and find peaks (= valleys in the hand) ----
        smoothF  = param['peakFind']['smoothF']
        min_dist = int(param['peakFind']['minPeakDistance'])

        smoothed = smooth_ma(-region_line[:, 2], smoothF)
        peaks, props = find_peaks(smoothed,
                                  distance=min_dist,
                                  prominence=5)

        if len(peaks) < 3:
            error_c[0] = -1
            break

        coordinates_orig = region_line[peaks, :2]   # (x, y)

        # ---- Local search to refine valley positions --------------------
        peaks_ref = local_search_valleys(im, bw, shape_f2, list(peaks), param)
        coordinates = region_line[peaks_ref, :2]    # (x, y)

        # ---- Sort by y then x (MATLAB sortrows(coordinates, [2 1])) -----
        sort_idx  = np.lexsort((coordinates[:, 0], coordinates[:, 1]))
        sort_coord = coordinates[sort_idx]          # (x, y) columns

        # ---- Discard outliers, keep best 3 ------------------------------
        sort_coord, err = discard_outliers(sort_coord, im, bw, param)
        if err == -1:
            error_c[0] = -1
            break

        # ---- ROI size ---------------------------------------------------
        _, roi_w, roi_h, x_offset = find_roi_size(sort_coord, param)

        # ---- Two most distant valley points for orientation -------------
        i_p1, i_p2 = find_most_dist_points(sort_coord)

        # ---- Orientation estimate ---------------------------------------
        grad = est_orient(sort_coord, i_p1, i_p2)

        # ---- Centroid marker images ------------------------------------
        z1, z2 = create_centroid_images(
            h_orig, w_orig, sort_coord, i_p1, i_p2)

        # ---- Optionally refine orientation (flip 180°) -----------------
        if param['ROIsize']['useRefineGrad']:
            grad_refined = refine_grad(im, bw, z1, z2, grad,
                                       roi_w, roi_h, x_offset, param)
        else:
            grad_refined = False

        # ---- Extract ROI ------------------------------------------------
        roi_out, rot_palm, new_c, top1, top2, bot1, bot2, err2 = \
            compute_roi(im, bw, z1, z2, grad, grad_refined,
                        roi_w, roi_h, x_offset, param)

        error_c[1] = err2
        if err2 == -1:
            break

        # Save results
        results['grad_refined']   = grad_refined
        results['valley_points']  = sort_coord   # (3, 2): x, y

        # Update centroid for next iteration (if numIterCentroid > 1)
        if new_c is not None:
            # new_c is in rotated frame; use the original centroid for now
            pass

    # ------------------------------------------------------------------
    # Final resize
    if error_c[0] == 0 and error_c[1] == 0 and roi_out is not None:
        target_h, target_w = param['ROIsize']['sizeROI']
        roi_resized = cv2.resize(
            roi_out.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_CUBIC
        )
        return roi_resized, 0, results
    else:
        return None, -1, results
