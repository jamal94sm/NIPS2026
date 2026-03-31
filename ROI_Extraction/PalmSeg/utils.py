"""
Shared utility functions.
Translated from MATLAB util/*.m files.
"""

import numpy as np
from scipy.ndimage import (
    binary_fill_holes, rotate as ndrotate, uniform_filter1d
)
from scipy.spatial.distance import cdist, pdist as scipy_pdist
from skimage.measure import label as sklabel, regionprops
from skimage.morphology import disk, binary_closing, binary_opening
from skimage.morphology import binary_dilation, binary_erosion
from skimage.draw import polygon as sk_polygon
from skimage.exposure import rescale_intensity


# ---------------------------------------------------------------------------
# Color-space conversion  (matches MATLAB rgb2ycbcr / rgb2hsv)
# ---------------------------------------------------------------------------

def rgb2ycbcr(img):
    """
    MATLAB-compatible RGB → YCbCr.
    Input img: float [0, 1], shape (H, W, 3).
    Returns float array (H, W, 3) with channels Y, Cb, Cr.
    """
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y  =  0.2989 * r + 0.5870 * g + 0.1140 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5000 * b + 0.5
    cr =  0.5000 * r - 0.4187 * g - 0.0813 * b + 0.5
    return np.stack([y, cb, cr], axis=-1)


def rgb2hsv(img):
    """
    MATLAB-compatible RGB → HSV.
    Input img: float [0, 1].  Returns float (H, W, 3) H/S/V each in [0, 1].
    """
    from matplotlib.colors import rgb_to_hsv as _rgb_to_hsv
    return _rgb_to_hsv(img)


def hsv2rgb(img):
    """HSV → RGB, values in [0, 1]."""
    from matplotlib.colors import hsv_to_rgb as _hsv_to_rgb
    return _hsv_to_rgb(img)


def convert_image_single_channel(img, color_space='rgb2gray'):
    """
    Equivalent to MATLAB convertImageSingleChannel.
    Returns (single_channel_float, full_converted_image_or_None).
    """
    if img.ndim == 2:
        return img.astype(float), None

    img_f = img.astype(float)
    if img_f.max() > 1.0:
        img_f = img_f / 255.0

    if color_space == 'rgb2gray':
        # BT.601 luminance
        single = 0.2989 * img_f[..., 0] + 0.5870 * img_f[..., 1] + 0.1140 * img_f[..., 2]
        return single, None
    elif color_space == 'rgb2ycbcr':
        conv = rgb2ycbcr(img_f)
        return conv[..., 0], conv          # Y channel
    elif color_space == 'rgb2hsv':
        conv = rgb2hsv(img_f)
        return conv[..., 2], conv          # V channel
    else:
        raise ValueError(f"Unknown color space: {color_space}")


# ---------------------------------------------------------------------------
# Image normalisation / adjustment
# ---------------------------------------------------------------------------

def normalize_img(img, norm_type='std'):
    """
    Equivalent to MATLAB normalizzaImg.
    Returns (normalised_img, min_val, max_val).
    """
    img = img.astype(float)
    if norm_type == 'std':
        minv = img.min()
    elif norm_type == 'nozeros':
        nonzero = img[img != 0]
        minv = float(nonzero.min()) if nonzero.size > 0 else 0.0
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

    img = img - minv
    maxv = float(img.max())
    if maxv > 0:
        img = img / maxv
    return img, minv, maxv


def rescale_img(img, min_val, max_val):
    """Equivalent to MATLAB rescaleImg."""
    img = img.astype(float) - min_val
    return img / max_val if max_val != 0 else img


def imadjust(img, lo_pct=0.01, hi_pct=0.99):
    """
    Equivalent to MATLAB imadjust(I, stretchlim(I,[lo hi])).
    Clips to [lo_pct, hi_pct] percentile and rescales to [0, 1].
    """
    p_lo = np.percentile(img, lo_pct * 100)
    p_hi = np.percentile(img, hi_pct * 100)
    return rescale_intensity(img.astype(float), in_range=(p_lo, p_hi),
                             out_range=(0.0, 1.0))


# ---------------------------------------------------------------------------
# Morphological helpers
# ---------------------------------------------------------------------------

def _make_se(type_str, size, angle=0):
    """Create structuring element.  type_str: 'disk' | 'line'."""
    if type_str == 'disk':
        return disk(size)
    elif type_str == 'line':
        return _strel_line(size, angle)
    raise ValueError(f"Unknown strel type: {type_str}")


def _strel_line(length, angle_deg):
    """MATLAB strel('line', length, angle) equivalent."""
    angle_rad = np.deg2rad(angle_deg)
    half = int(length) // 2
    steps = np.arange(-half, half + 1)
    rows = np.round(-steps * np.sin(angle_rad)).astype(int)
    cols = np.round( steps * np.cos(angle_rad)).astype(int)
    r0, c0 = rows.min(), cols.min()
    h = rows.max() - r0 + 1
    w = cols.max() - c0 + 1
    se = np.zeros((max(1, h), max(1, w)), dtype=bool)
    se[rows - r0, cols - c0] = True
    return se


def imclose_bin(bw, se):
    return binary_closing(bw.astype(bool), se)


def imopen_bin(bw, se):
    return binary_opening(bw.astype(bool), se)


def imdilate_bin(bw, se):
    return binary_dilation(bw.astype(bool), se)


def imerode_bin(bw, se):
    return binary_erosion(bw.astype(bool), se)


def imfill_holes(bw):
    return binary_fill_holes(bw.astype(bool))


# ---------------------------------------------------------------------------
# Connected-component helpers
# ---------------------------------------------------------------------------

def big_conn_comp(bw, fill=True):
    """
    Keep only the largest connected component (8-connectivity).
    Equivalent to MATLAB bigConnComp.
    """
    bw = bw.astype(bool)
    labeled = sklabel(bw, connectivity=2)
    props = regionprops(labeled)
    if not props:
        return bw
    max_area = max(p.area for p in props)
    result = np.zeros_like(bw)
    for p in props:
        if p.area == max_area:
            result[labeled == p.label] = True
    if fill:
        result = binary_fill_holes(result)
    return result.astype(bool)


def remove_cc_area(bw, th_area):
    """
    Remove connected components with area < th_area.
    Equivalent to MATLAB removeCCArea.
    """
    labeled = sklabel(bw.astype(bool), connectivity=2)
    result = np.zeros_like(bw, dtype=bool)
    for p in regionprops(labeled):
        if p.area >= th_area:
            result[labeled == p.label] = True
    return result


# ---------------------------------------------------------------------------
# Shape / contour helpers
# ---------------------------------------------------------------------------

def poly2mask(x_coords, y_coords, height, width):
    """
    Equivalent to MATLAB poly2mask(x, y, h, w).
    x_coords → columns, y_coords → rows.
    """
    mask = np.zeros((height, width), dtype=bool)
    rr, cc = sk_polygon(np.asarray(y_coords), np.asarray(x_coords),
                        shape=(height, width))
    mask[rr, cc] = True
    return mask


def resizem(shape_pts, n_out):
    """
    Interpolate an (N, 2) array of shape points to n_out points.
    Equivalent to MATLAB resizem(..., 'bilinear').
    """
    from scipy.interpolate import interp1d
    n_in = shape_pts.shape[0]
    t_in  = np.linspace(0, 1, n_in)
    t_out = np.linspace(0, 1, n_out)
    out = np.zeros((n_out, shape_pts.shape[1]))
    for col in range(shape_pts.shape[1]):
        f = interp1d(t_in, shape_pts[:, col], kind='linear',
                     fill_value='extrapolate')
        out[:, col] = f(t_out)
    return out


def smooth_ma(x, window):
    """
    Moving-average smoothing — equivalent to MATLAB smooth(x, window).
    """
    return uniform_filter1d(np.asarray(x, dtype=float), size=int(window),
                            mode='reflect')


def zero_border(img, border=5):
    """
    Set a `border`-pixel border to zero.
    Equivalent to MATLAB zeroBorder (which uses 5 pixels).
    """
    out = np.zeros_like(img)
    out[border-1:-(border), border-1:-(border)] = \
        img[border-1:-(border), border-1:-(border)]
    return out


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def imrotate(img, angle_deg, interp='bilinear', crop=False):
    """
    Rotate image by angle_deg (counterclockwise, like MATLAB imrotate).
    crop=True → output same size as input (MATLAB 'crop' option).
    interp: 'bilinear' (order=1) or 'nearest' (order=0).
    """
    order = 1 if interp == 'bilinear' else 0
    return ndrotate(img.astype(float), angle_deg,
                    reshape=not crop, order=order, cval=0.0)


def compensate_crop_bb(rot_img, h_orig, w_orig):
    """
    Centre-crop a rotated image back to original size.
    Equivalent to MATLAB compensateCropBB.
    """
    import cv2 as _cv2
    h_rot, w_rot = rot_img.shape[:2]
    x0 = max(0, int(w_rot / 2 - w_orig / 2))
    y0 = max(0, int(h_rot / 2 - h_orig / 2))
    x1 = min(w_rot, x0 + w_orig)
    y1 = min(h_rot, y0 + h_orig)
    cropped = rot_img[y0:y1, x0:x1]
    if cropped.shape[:2] != (h_orig, w_orig):
        cropped = _cv2.resize(cropped, (w_orig, h_orig))
    return cropped


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def pdist2(a, b):
    """Pairwise Euclidean distances between rows of a and b (MATLAB pdist2)."""
    return cdist(a, b)


def pdist_max(pts):
    """Maximum pairwise distance among rows of pts."""
    if pts.shape[0] < 2:
        return 0.0
    return float(scipy_pdist(pts).max())


def find_most_dist_points(sort_coord):
    """
    Return indices of the two most distant points.
    Equivalent to MATLAB findMostDistPoints.
    """
    dm = cdist(sort_coord, sort_coord)
    max_v = dm.max()
    idx = np.argwhere(dm == max_v)
    return int(idx[0, 0]), int(idx[0, 1])


def triangle_angles(pts, fmt='d'):
    """
    Angles of the triangle defined by 3 points (rows of pts).
    Equivalent to MATLAB triangle_angles.
    fmt='d' → degrees, 'r' → radians.
    """
    L = np.array([
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[1] - pts[2]),
        np.linalg.norm(pts[2] - pts[0]),
    ])
    s = L.sum() / 2
    area_sq = s * (s - L[0]) * (s - L[1]) * (s - L[2])
    h = (2 / L[2]) * np.sqrt(max(area_sq, 0.0))
    x = (L[0]**2 - L[1]**2 + L[2]**2) / (2 * L[2])

    if fmt == 'd':
        a1 = np.degrees(np.arcsin(np.clip(h / L[0], -1, 1)))
        if x < 0:
            a1 = 180 - a1
        a3 = np.degrees(np.arcsin(np.clip(h / L[1], -1, 1)))
        if x > L[2]:
            a3 = 180 - a3
        a2 = 180 - a3 - a1
    else:
        a1 = np.arcsin(np.clip(h / L[0], -1, 1))
        if x < 0:
            a1 = np.pi - a1
        a3 = np.arcsin(np.clip(h / L[1], -1, 1))
        if x > L[2]:
            a3 = np.pi - a3
        a2 = np.pi - a3 - a1

    return np.array([a1, a2, a3])


def est_orient(sort_coord, i_p1, i_p2):
    """
    Estimate rotation angle from two key valley points.
    Equivalent to MATLAB estOrient.
    """
    diff_y = sort_coord[i_p2, 1] - sort_coord[i_p1, 1]
    diff_x = sort_coord[i_p2, 0] - sort_coord[i_p1, 0]
    grad = np.degrees(np.arctan2(diff_y, diff_x))
    grad_mod = -90 - grad
    return grad_mod


# ---------------------------------------------------------------------------
# Neighbourhood ring sampler (for evalCond)
# ---------------------------------------------------------------------------

def find_neigh(bw, point, distance, offset_a, num_neigh):
    """
    Sample `num_neigh` pixels on a circle of `distance` radius.
    Equivalent to MATLAB findNeigh.
    point = (x, y) = (col, row).
    """
    px = int(round(point[0]))   # col
    py = int(round(point[1]))   # row

    # Boundary pre-check
    if (py + distance >= bw.shape[0] or px + distance >= bw.shape[1] or
            py - distance <= 0 or px - distance <= 0):
        return np.array([])

    step = 360.0 / num_neigh
    neigh = np.zeros(num_neigh, dtype=float)
    for k, angle in enumerate(np.arange(0, 360, step)):
        a_rad = np.deg2rad(angle + offset_a)
        row = int(round(py + distance * np.sin(a_rad)))
        col = int(round(px + distance * np.cos(a_rad)))
        if not (0 <= row < bw.shape[0] and 0 <= col < bw.shape[1]):
            return np.array([])
        neigh[k] = bw[row, col]
    return neigh


def num_cons_els(vector, value):
    """
    Maximum consecutive run of `value` in a circular arrangement.
    Equivalent to MATLAB numConsEls.
    """
    v = np.asarray(vector)
    n = len(v)
    if n == 0:
        return -1
    # Circular doubling trick
    doubled = np.tile(v, 2)
    max_run = -1
    cur = 0
    for i in range(2 * n):
        if doubled[i] == value:
            cur += 1
            if cur <= n:
                max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run
