"""
Kirsch-template edge/vessel extraction.
Translated from MATLAB VesselExtract.m.

Reference:
  Kirsch, R. (1971). "Computer determination of the constituent structure of
  biological images." Computers and Biomedical Research. 4: 315-328.
"""

import numpy as np
from scipy.ndimage import correlate  # filter2 in MATLAB = correlation


# Eight Kirsch compass kernels (normalised by /15)
_KIRSCH_KERNELS = [
    np.array([[ 5, -3, -3],
              [ 5,  0, -3],
              [ 5, -3, -3]], dtype=float) / 15,   # h1  (West)

    np.array([[-3, -3,  5],
              [-3,  0,  5],
              [-3, -3,  5]], dtype=float) / 15,   # h2  (East)

    np.array([[-3, -3, -3],
              [ 5,  0, -3],
              [ 5,  5, -3]], dtype=float) / 15,   # h3  (SW)

    np.array([[-3,  5,  5],
              [-3,  0,  5],
              [-3, -3, -3]], dtype=float) / 15,   # h4  (NE)

    np.array([[-3, -3, -3],
              [-3,  0, -3],
              [ 5,  5,  5]], dtype=float) / 15,   # h5  (South)

    np.array([[ 5,  5,  5],
              [-3,  0, -3],
              [-3, -3, -3]], dtype=float) / 15,   # h6  (North)

    np.array([[-3, -3, -3],
              [-3,  0,  5],
              [-3,  5,  5]], dtype=float) / 15,   # h7  (SE)

    np.array([[ 5,  5, -3],
              [ 5,  0, -3],
              [-3, -3, -3]], dtype=float) / 15,   # h8  (NW)
]


def vessel_extract(in_img, threshold=0, direction='all',
                   ind_vert=None, ind_hor=None):
    """
    Extract edges/vessels using Kirsch compass kernels.

    Parameters
    ----------
    in_img     : 2-D float or uint8 array.
    threshold  : Minimum response to keep (0 = keep all positives).
    direction  : 'all' | 'vertical' | 'horizontal'
    ind_vert   : 1-indexed list of kernel indices for vertical direction.
    ind_hor    : 1-indexed list of kernel indices for horizontal direction.

    Returns
    -------
    blood_vessels : 2-D float array, same spatial size as in_img.
    """
    in_img = in_img.astype(float)

    # Symmetric padding (matches MATLAB padarray with 'symmetric')
    in_pad = np.pad(in_img, 2, mode='reflect')

    # Apply all 8 Kirsch kernels via 2-D correlation (= MATLAB filter2)
    responses = np.stack(
        [correlate(in_pad, k, mode='constant', cval=0.0)
         for k in _KIRSCH_KERNELS],
        axis=-1
    )

    # Select indices to consider (convert 1-indexed MATLAB → 0-indexed)
    if direction == 'all':
        idx = list(range(8))
    elif direction == 'vertical':
        idx = [i - 1 for i in (ind_vert or [1, 2])]
    elif direction == 'horizontal':
        idx = [i - 1 for i in (ind_hor or [5, 6])]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # Keep only selected directions, then take pixel-wise max
    selected = np.full_like(responses, -np.inf)
    selected[..., idx] = responses[..., idx]
    bv = selected.max(axis=-1)

    # Threshold
    bv[bv < threshold] = 0.0

    # Crop back to original size (undo the 2-pixel padding on each side)
    h, w = in_img.shape
    blood_vessels = bv[2:2 + h, 2:2 + w]

    return blood_vessels
