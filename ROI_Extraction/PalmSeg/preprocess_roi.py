"""
ROI preprocessing for PalmNet feature extraction.
Translated from MATLAB applyPreproc.m and preProcessROI.m.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from skimage.exposure import equalize_adapthist

from utils import convert_image_single_channel, rgb2hsv, hsv2rgb


# ---------------------------------------------------------------------------
# fspecial('laplacian', alpha) kernel
# ---------------------------------------------------------------------------

def _laplacian_kernel(alpha=0.2):
    """
    Matches MATLAB fspecial('laplacian', alpha).
    Default alpha = 0.2.
    """
    return np.array([
        [alpha / 4,       (1 - alpha) / 4, alpha / 4],
        [(1 - alpha) / 4, -1.0,            (1 - alpha) / 4],
        [alpha / 4,       (1 - alpha) / 4, alpha / 4],
    ])


# ---------------------------------------------------------------------------
# apply_preproc  (applyPreproc.m)
# ---------------------------------------------------------------------------

def apply_preproc(roi, param):
    """
    Apply CLAHE → Laplacian sharpening → Gaussian smoothing to a single-
    channel ROI image.
    Equivalent to MATLAB applyPreproc.

    Parameters
    ----------
    roi   : 2-D float array in [0, 1].
    param : parameter dict containing 'prepProc' sub-dict.

    Returns
    -------
    roi_out : 2-D float array in [0, 1].
    """
    pp = param['prepProc']
    pad   = int(pp['padSize'])
    alpha = float(pp.get('alphaLaplacian', 0.2))
    sigma = float(pp.get('sigmaGauss', 1.0))

    roi = roi.astype(float)

    # 1. CLAHE (adapthisteq in MATLAB)
    roi_clahe = equalize_adapthist(roi)

    # 2. Laplacian sharpening
    roi_pad = np.pad(roi_clahe, pad, mode='reflect')
    h_lap   = _laplacian_kernel(alpha)
    lap     = convolve(roi_pad, h_lap, mode='reflect')
    roi_sharp = roi_pad - lap

    # Crop back
    roi_crop = roi_sharp[pad: pad + roi.shape[0],
                         pad: pad + roi.shape[1]]

    # 3. Gaussian smoothing
    roi_out = gaussian_filter(roi_crop, sigma=sigma)

    return roi_out


# ---------------------------------------------------------------------------
# preprocess_roi  (preProcessROI.m)
# ---------------------------------------------------------------------------

def preprocess_roi(roi, param):
    """
    Full ROI preprocessing pipeline.
    Equivalent to MATLAB preProcessROI.

    Parameters
    ----------
    roi   : float array (H, W) or (H, W, 3) in [0, 1].
    param : parameter dict.

    Returns
    -------
    roi_proc       : 2-D float array – processed single-channel ROI.
    roi_proc_color : 3-D float array or None – processed colour ROI.
    """
    pp = param['prepProc']
    roi = roi.astype(float)
    if roi.max() > 1.0:
        roi = roi / 255.0

    # Single-channel version
    roi_single, _ = convert_image_single_channel(
        roi, pp.get('colorSpaceTransSingle', 'rgb2gray'))
    roi_proc = apply_preproc(roi_single, param)

    # Colour version (process luminance channel in HSV)
    roi_proc_color = None
    if roi.ndim == 3:
        hsv = rgb2hsv(roi)
        v_channel = apply_preproc(hsv[:, :, 2], param)
        hsv[:, :, 2] = v_channel
        roi_proc_color = hsv2rgb(hsv)

    return roi_proc, roi_proc_color
