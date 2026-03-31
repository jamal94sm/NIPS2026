"""
Hand segmentation: produces the palm contour (shapeFinal), centroid,
orientation, and binary palm mask (bw).
Translated from MATLAB segmentPalms.m and findOrientBasedonEdge.m.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours, label as sklabel, regionprops
from skimage.filters import sobel_h as _sobel_h, sobel_v as _sobel_v

import cv2

from utils import (
    convert_image_single_channel, normalize_img, rescale_img,
    imadjust, zero_border, poly2mask, resizem, smooth_ma,
    big_conn_comp, remove_cc_area, imfill_holes,
    imclose_bin, imopen_bin, imdilate_bin,
    imrotate, compensate_crop_bb, _make_se,
)
from vessel_extract import vessel_extract
from threshold_palm import graythresh, threshold_palm


# ---------------------------------------------------------------------------
# Edge-orientation search (findOrientBasedonEdge.m)
# ---------------------------------------------------------------------------

def _auto_sobel_threshold(img):
    """Approximate MATLAB's automatic Sobel threshold."""
    gx = _sobel_v(img.astype(float))
    gy = _sobel_h(img.astype(float))
    mag = np.sqrt(gx**2 + gy**2)
    return np.sqrt(2) * float(mag.mean())


def _sobel_edges(img, direction, threshold):
    """Binary Sobel edges in one direction."""
    if direction == 'horizontal':
        grad = np.abs(_sobel_h(img.astype(float)))
    else:
        grad = np.abs(_sobel_v(img.astype(float)))
    return grad > threshold


def find_orient_based_on_edge(C, C_uint8, thVessF, param):
    """
    Estimate palm orientation and extract horizontal/vertical edge maps.
    Equivalent to MATLAB findOrientBasedonEdge.

    Returns
    -------
    refl_ho : float array  – horizontal vessel-edge map
    refl_ve : float array  – vertical vessel-edge map
    orient_m : int         – dominant orientation angle (degrees)
    """
    h_orig, w_orig = C_uint8.shape[:2]
    p = param['segm']

    # -------- Step 1: find orientation that minimises ho/ve edge ratio --------
    # Two-pass coarse→fine: 13 + 7 = 20 iterations instead of 37
    def _score_angles(angles):
        best_angle, best_cond = 0, 1e9
        for angle in angles:
            C_rot = imrotate(C, angle, crop=True)
            th_e  = _auto_sobel_threshold(C_rot)
            if th_e == 0:
                continue
            e_ho = _sobel_edges(C_rot, 'horizontal', th_e)
            e_ve = _sobel_edges(C_rot, 'vertical',   th_e)
            e_ho_back = compensate_crop_bb(
                imrotate(e_ho.astype(float), -angle, crop=False), h_orig, w_orig)
            e_ve_back = compensate_crop_bb(
                imrotate(e_ve.astype(float), -angle, crop=False), h_orig, w_orig)
            area_ve = float(e_ve_back.sum())
            if area_ve == 0:
                continue
            cond = float(e_ho_back.sum()) / area_ve
            if cond < best_cond:
                best_cond, best_angle = cond, angle
        return best_angle

    coarse   = _score_angles(range(-90, 91, 15))          # step 15° → 13 iters
    orient_m = _score_angles(                              # step  5° →  7 iters
        range(max(-90, coarse - 15), min(90, coarse + 15) + 1, 5)
    )

    # -------- Step 2: build vessel-edge maps at optimum orientation -----------
    ind_ver = p['indKirschVer']
    ind_hor = p['indKirschHor']

    refl_ho = np.zeros((h_orig, w_orig), dtype=float)
    for ang in range(orient_m - p['edgeHoSearchAngle'],
                     orient_m + p['edgeHoSearchAngle'] + 1):
        C_rot = imrotate(C_uint8.astype(float), ang, crop=False)
        vess  = vessel_extract(C_rot, 0, 'horizontal', ind_ver, ind_hor)
        vess_back = imrotate(vess, -ang, crop=False)
        vess_crop = compensate_crop_bb(vess_back, h_orig, w_orig)
        vess_crop = zero_border(vess_crop)
        refl_ho = np.maximum(refl_ho, vess_crop)

    refl_ve = np.zeros((h_orig, w_orig), dtype=float)
    for ang in range(orient_m - p['edgeVeSearchAngle'],
                     orient_m + p['edgeVeSearchAngle'] + 1):
        C_rot = imrotate(C_uint8.astype(float), ang, crop=False)
        vess  = vessel_extract(C_rot, 0, 'vertical', ind_ver, ind_hor)
        vess_back = imrotate(vess, -ang, crop=False)
        vess_crop = compensate_crop_bb(vess_back, h_orig, w_orig)
        vess_crop = zero_border(vess_crop)
        refl_ve = np.maximum(refl_ve, vess_crop)

    return refl_ho, refl_ve, orient_m


# ---------------------------------------------------------------------------
# Main segmentation function
# ---------------------------------------------------------------------------

def segment_palms(input_image, param):
    """
    Segment the palm and return its contour, centroid, orientation, mask.
    Equivalent to MATLAB segmentPalms.

    Parameters
    ----------
    input_image : uint8 or float ndarray, shape (H, W) or (H, W, 3).
    param       : parameter dict (from params.py).

    Returns
    -------
    shape_final : (N, 2) float array  – contour points [x, y]
    centroid    : (2,) float array    – [x, y] = [col, row]
    orient_m    : int                 – dominant orientation (degrees)
    bw_smooth   : (H, W) bool array  – binary palm mask
    """
    p = param['segm']

    # --- Convert to float [0, 1] -------------------------------------------
    img_orig = input_image.astype(float)
    if img_orig.max() > 1.0:
        img_orig = img_orig / 255.0

    # --- Convert to single channel -----------------------------------------
    img_single, _ = convert_image_single_channel(img_orig, p['colorSpaceTrans'])

    # --- Resize ------------------------------------------------------------
    rf = p['resizeF']
    if rf != 1:
        new_h = int(img_single.shape[0] * rf)
        new_w = int(img_single.shape[1] * rf)
        img_single = cv2.resize(img_single, (new_w, new_h))
        img_color  = cv2.resize(img_orig,   (new_w, new_h))
    else:
        img_color = img_orig.copy()

    # --- Contrast stretch --------------------------------------------------
    img_single = imadjust(img_single)

    # --- Optional normalisation --------------------------------------------
    if p.get('normalizza', 0):
        img_single, _, _ = normalize_img(img_single)
        img_color,  _, _ = normalize_img(img_color)

    # --- Gaussian smooth ---------------------------------------------------
    C = gaussian_filter(img_single, sigma=p['fGauss_sigma'])
    C_u8 = np.clip(C * 255, 0, 255).astype(np.uint8)

    # --- Binarise (Otsu) ---------------------------------------------------
    binar = threshold_palm(C, param)

    # --- Vessel / edge extraction (all Kirsch directions) ------------------
    vess, min_vess, max_vess = normalize_img(
        vessel_extract(C_u8, 0))
    vess2 = zero_border(vess)
    th_vess_f, _ = graythresh(vess2)
    edge_added = vess2 > th_vess_f

    se_med  = _make_se(p['typeStrel'], p['sizeStrel_medium'])
    se_lrg  = _make_se(p['typeStrel'], p['sizeStrel_large'])

    edge_added = imclose_bin(edge_added, se_med)
    edge_added = imopen_bin( edge_added, se_med)

    # --- Optionally add edge to binary mask --------------------------------
    if p['useEdgeAdd']:
        binar_pe = binar | edge_added
        binar_pe = imclose_bin(binar_pe, se_med)
        binar_pe = imfill_holes(binar_pe)
        binar_pe = imopen_bin(binar_pe, se_med)
    else:
        binar_pe = binar.copy()
    binar_pe = big_conn_comp(binar_pe, fill=True)

    # --- Orientation and directional edge maps -----------------------------
    refl_ho, refl_ve, orient_m = find_orient_based_on_edge(
        C, C_u8.astype(float), th_vess_f, param)

    # Rescale to [0, 1] using the same stats as the vessel image
    refl_ho = rescale_img(refl_ho, min_vess, max_vess)
    refl_ve = rescale_img(refl_ve, min_vess, max_vess)

    # Threshold edge maps
    th_ve = th_vess_f + p['thVessModVe']
    th_ho = th_vess_f + p['thVessModHo']
    edge_removed = refl_ve > th_ve
    refl_thresh  = refl_ho > th_ho

    orient_line = 90 - orient_m     # for line strel

    # --- Subtract vertical edges -------------------------------------------
    if p['subEdgesVer']:
        se_e_sm  = _make_se(p['typeStrelEdge'], p['sizeStrelEdge_small'],  orient_line)
        se_e_md  = _make_se(p['typeStrelEdge'], p['sizeStrelEdge_medium'], orient_line)
        se_e_hg  = _make_se(p['typeStrelEdge'], p['sizeStrelEdge_huge'],   orient_line)

        er = edge_removed.astype(bool) & ~refl_thresh.astype(bool)
        er = imdilate_bin(er, se_e_sm)
        er = imclose_bin( er, se_e_hg)
        er = imopen_bin(  er, se_e_md)
        er = remove_cc_area(er, p['thArea'])

        binar_me = binar_pe.astype(bool) & ~er
        se_op = _make_se(p['typeStrel'], p['sizeStrel_medium'])
        binar_me = imopen_bin(binar_me, se_op)
    else:
        er = edge_removed.copy()
        binar_me = binar_pe.copy()

    # --- Add horizontal edges (reflections) --------------------------------
    if p['useRefl']:
        orient_ho = orient_line - 90
        se_e_sm2 = _make_se(p['typeStrelEdge'], p['sizeStrelEdge_small'],  orient_ho)
        se_e_lr2 = _make_se(p['typeStrelEdge'], p['sizeStrelEdge_large'],  orient_ho)

        refl = refl_thresh.astype(bool) & ~er.astype(bool)
        refl = imdilate_bin(refl, se_e_sm2)
        refl = imclose_bin( refl, se_e_lr2)

        binar_pr = binar_me.astype(bool) | refl
        binar_pr = imfill_holes(binar_pr)
        se_lrg2 = _make_se(p['typeStrel'], p['sizeStrel_large'])
        binar_pr = imopen_bin(binar_pr, se_lrg2)
    else:
        binar_pr = binar_me.copy()

    binar_pr = big_conn_comp(binar_pr, fill=True)

    # --- Invert resize (go back to original resolution) -------------------
    h_orig, w_orig = input_image.shape[:2]
    if rf != 1:
        binar_pr_orig = cv2.resize(
            binar_pr.astype(np.uint8), (w_orig, h_orig)) > 0
    else:
        binar_pr_orig = binar_pr.astype(bool)

    # --- Boundary extraction ----------------------------------------------
    contours = find_contours(binar_pr_orig.astype(float), 0.5)
    if not contours:
        raise RuntimeError("No contour found after segmentation.")
    contour = max(contours, key=len)    # longest = outer boundary

    # find_contours returns (row, col); match MATLAB's flipud(B{1})
    outline = contour[::-1]             # flip order (like flipud)
    shape_final_raw = np.column_stack([outline[:, 1],   # col = x
                                       outline[:, 0]])   # row = y

    # --- Centroid ---------------------------------------------------------
    props = regionprops(sklabel(binar_pr_orig.astype(int), connectivity=2))
    if props:
        cy, cx = props[0].centroid      # skimage: (row, col)
        centroid = np.array([cx, cy])   # → (col, row) = (x, y)
    else:
        centroid = np.array([w_orig / 2, h_orig / 2])

    # --- Interpolate contour to fixed number of points --------------------
    n_interp = p['sizeShapeInterp']
    shape_final = resizem(shape_final_raw, n_interp)

    # --- Smooth contour ---------------------------------------------------
    win = p['smoothShapeSize']
    shape_final[:, 0] = smooth_ma(shape_final[:, 0], win)
    shape_final[:, 1] = smooth_ma(shape_final[:, 1], win)

    # --- Binary mask from smoothed contour --------------------------------
    bw_smooth = poly2mask(shape_final[:, 0], shape_final[:, 1],
                          h_orig, w_orig)

    return shape_final, centroid, orient_m, bw_smooth
