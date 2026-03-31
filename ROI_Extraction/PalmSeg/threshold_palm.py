"""
Palm-image binarization using Otsu's method with effectiveness test.
Translated from MATLAB thresholdPalm.m.
"""

import numpy as np
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.measure import label as sklabel


# ---------------------------------------------------------------------------
# Otsu threshold + effectiveness  (MATLAB graythresh)
# ---------------------------------------------------------------------------

def graythresh(img):
    """
    Otsu threshold and inter-class separability (effectiveness).

    Returns
    -------
    thresh : float  – threshold in [0, 1]
    em     : float  – effectiveness in [0, 1]
    """
    img = img.astype(float)
    # Normalise to [0, 1] if needed
    if img.max() > 1.0:
        img = img / 255.0

    thresh = float(threshold_otsu(img))

    # Otsu effectiveness  η = σ_B² / σ_T²
    pix = img.ravel()
    sigma_t2 = float(pix.var())
    if sigma_t2 == 0:
        return thresh, 0.0

    mu_t = float(pix.mean())
    w0 = float((pix <= thresh).mean())
    w1 = 1.0 - w0

    if w0 == 0 or w1 == 0:
        return thresh, 0.0

    mu0 = float(pix[pix <= thresh].mean())
    mu1 = float(pix[pix > thresh].mean())

    sigma_b2 = w0 * (mu0 - mu_t) ** 2 + w1 * (mu1 - mu_t) ** 2
    em = sigma_b2 / sigma_t2
    return thresh, em


def multithresh(img, n=2):
    """
    Multi-level Otsu (MATLAB multithresh).
    Returns (thresholds_array, effectiveness).
    """
    img = img.astype(float)
    if img.max() > 1.0:
        img = img / 255.0

    thresholds = threshold_multiotsu(img, classes=n + 1)

    # Effectiveness: between-class / total variance
    pix = img.ravel()
    sigma_t2 = float(pix.var())
    if sigma_t2 == 0:
        return thresholds, 0.0

    mu_t = float(pix.mean())
    boundaries = np.concatenate([[pix.min()], thresholds, [pix.max() + 1e-9]])
    sigma_b2 = 0.0
    for i in range(len(thresholds) + 1):
        mask = (pix >= boundaries[i]) & (pix < boundaries[i + 1])
        w_k = float(mask.mean())
        if w_k > 0:
            mu_k = float(pix[mask].mean())
            sigma_b2 += w_k * (mu_k - mu_t) ** 2

    em = sigma_b2 / sigma_t2
    return thresholds, em


# ---------------------------------------------------------------------------
# Main threshold function
# ---------------------------------------------------------------------------

def threshold_palm(img, param):
    """
    Binarize a grayscale palm image.
    Equivalent to MATLAB thresholdPalm.

    Parameters
    ----------
    img   : float 2-D array in [0, 1].
    param : parameter dict (segm sub-dict required).

    Returns
    -------
    binar : bool 2-D array.
    """
    p = param['segm']
    th_em      = p['th_em']
    th_red     = p['thSegmRed']
    num_cc_max = p['numCCbinar']

    thresh_s, em_s = graythresh(img)

    if em_s >= th_em:
        # Good separability → single Otsu threshold
        binar = img > (thresh_s + th_red)
    else:
        # Poor separability → try multi-level Otsu (2 thresholds → 3 classes)
        thresholds_m, em_m = multithresh(img, n=2)
        if em_m > th_em:
            seg = np.digitize(img, thresholds_m + th_red)   # 0, 1, 2
            binar = (seg == 1) | (seg == 2)
            # Sanity check: too many connected components → fall back
            _, n_cc = sklabel(binar, connectivity=2, return_num=True)
            if n_cc > num_cc_max:
                binar = img > thresh_s
        else:
            binar = img > thresh_s

    # Collapse any residual non-binary values and cast
    binar = binar.astype(bool)
    return binar
