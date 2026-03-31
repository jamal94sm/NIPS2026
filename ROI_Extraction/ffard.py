#!/usr/bin/env python3
"""
extract_mpd_rois.py
===================
Full FFARD pipeline — hand segmentation (CHSST) → rotation alignment
(PalmAlNet/LANet) → constrained inscribed-circle ROI extraction.

Usage
-----
  python extract_mpd_rois.py

Edit the CONFIG block below before running.  Run from the RDRLA repo
root so that `CHSST.models.*` imports resolve.

Requirements
------------
  pip install torch torchvision opencv-python numpy
  pip install mmcv==1.4.0 mmseg==0.20.0          # for CHSST backbone

Output
------
  ROI_OUT_DIR/  – 128×128 JPEGs, 20 augmented crops per input image
                  filename pattern: <base>_<aug_angle>.jpg
  SEG_TMP_DIR/  – intermediate segmented palms (black background)
  VIS_OUT_DIR/  – optional debug images showing both inscribed circles
"""

# ─────────────────────────────────────────────────────────────────
#  CONFIG  ← edit these before running
# ─────────────────────────────────────────────────────────────────
MPD_RAW_DIR = "/home/pai-ng/Jamal/MPDv2"
CHSST_CKPT  = "EP7-iou0.951562-pacc0.977916.pth"
LANET_CKPT  = "LANet_v1.pkl"
SEG_TMP_DIR = "tmp/MPD_segmented"        # intermediate; can be /tmp
ROI_OUT_DIR = "/home/pai-ng/Jamal/MPDv2_ROI_FFARD"         # ← main output
VIS_OUT_DIR = "MPD_vis/"     # circle debug visualisations
SAVE_VIS    = False                        # set True to write debug images
DEVICE      = "cuda"                       # "cuda" or "cpu"
# ─────────────────────────────────────────────────────────────────

import os, sys, math, warnings, tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

warnings.filterwarnings("ignore")

# Add repo root so CHSST.models.* imports work when run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════
#  1. LANet (PalmAlNet) — rotation angle predictor
#     Source: adaptive_PROIE/LANet.py
# ═══════════════════════════════════════════════════════════════════

def _make_vgg_p3():
    """First 3 blocks of VGG-16 (up to layer index 17 — ReLU after block-3 pool)."""
    try:
        vgg = torchvision.models.vgg16(weights=None)
    except TypeError:
        vgg = torchvision.models.vgg16(pretrained=False)
    return vgg.features[:17]


class LAnet(nn.Module):
    """
    Rotation-angle regression network.
    Input : (B, 3, 56, 56)  — grayscale repeated in channels 0-1,
                               2-D Gaussian heatmap in channel 2.
    Output: (B, 1)           — tanh-scaled angle in [-1, 1]
                               (multiply by pi to get radians).
    """
    def __init__(self, ipt_size: int = 56):
        super().__init__()
        self.vgg_p3 = _make_vgg_p3()
        feat_dim = (ipt_size * ipt_size * 256) // 64     # 12 544
        self.extra = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(feat_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extra(self.vgg_p3(x))


# ═══════════════════════════════════════════════════════════════════
#  2. Reconstructed utilities (utills.py — missing from repo)
#     Implemented directly from paper Section III-B and dataset.py
# ═══════════════════════════════════════════════════════════════════

def padding_img(img: np.ndarray, padding: int = 20) -> np.ndarray:
    """
    Add a black border of `padding` pixels on all sides.
    Used after rotation so the rotated palm keeps its full extent.
    """
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    out = np.zeros((h + 2 * padding, w + 2 * padding, c) if img.ndim == 3
                   else (h + 2 * padding, w + 2 * padding), dtype=img.dtype)
    out[padding:padding + h, padding:padding + w] = img
    return out


def _dist_transform(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance transform of a uint8 binary mask."""
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2,
                                 cv2.DIST_MASK_PRECISE)


def find_circle_inform_hard(mask: np.ndarray):
    """
    Find the 'hard' inscribed circle centre and radius.
    Mirrors dataset.py implementation exactly.

    Strategy: find the pixel that sits deepest inside the palm AND is
    toward the upper half — i.e. the palm centre rather than the wrist.

    Returns: center (cx, cy), radius r  (all ints)
    """
    dist_map = _dist_transform(mask)
    _, max_val, _, center = cv2.minMaxLoc(dist_map)
    center_x, center_y = int(center[0]), int(center[1])

    # Region within 90 % of the maximum inscribed-circle radius
    good_val    = max_val * 0.9
    good_region = dist_map.copy()
    good_region[good_region < good_val] = 0
    good_region = good_region.astype(np.uint8)

    # How far inside `good_region` is the existing centre?
    inner_dist = _dist_transform(good_region)
    radius = inner_dist[center_y, center_x]

    # Remove the inscribed circle from good_region so only the surrounding
    # ring remains, then keep only the upper half of that ring.
    good_region_copy = good_region.copy()
    cv2.circle(good_region_copy, (center_x, center_y), int(radius), 0, -1)
    black_image = np.zeros_like(good_region_copy)
    black_image[:center_y] = good_region_copy[:center_y]
    black_image[black_image > 0] = 255

    contours, _ = cv2.findContours(black_image.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        return (center_x, center_y), int(dist_map[center_y, center_x])

    # Largest blob in the upper ring → centroid is the refined centre
    best  = max(contours, key=cv2.contourArea)
    M     = cv2.moments(best)
    if M["m00"] == 0:
        return (center_x, center_y), int(dist_map[center_y, center_x])

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    r  = int(dist_map[cy, cx])
    return (cx, cy), r


def circle_better(img: np.ndarray,
                  visualize_img: np.ndarray,
                  mask: np.ndarray,
                  rate: float = 1.1):
    """
    Reconstructed from paper Section III-B-2 (Adaptive ROI Detection
    with Constraints).  This is the core of FFARD.

    Algorithm  (parameters from paper: t1=0.85, t2=1.1)
    ─────────────────────────────────────────────────────
    Step 1  Dc  = distance transform of `mask`
    Step 2  PC  = argmax(Dc),  r = max(Dc)              ← max inscribed circle
    Step 3  S   = {p | Dc(p) > t1·r}                    ← acceptable region
    Step 4  Ds  = distance transform of S
    Step 5  T   = {p | ‖p − PC‖₂ > Ds(PC),  p.y ≥ y₀}  ← lower / outer band
    Step 6  S'  = S ∩ T
    Step 7  P'C = centroid of largest contour in S'
    Step 8  final_r = t2 · Dc(P'C)                      ← constrained radius
    Step 9  Crop a square of side 2·final_r centred at P'C (padded to exact size)

    Returns: (cropped_palm_img, visualisation_img)
    """
    t1 = 0.85
    t2 = rate          # 1.1

    h, w = mask.shape[:2]

    # ── Steps 1-2 ───────────────────────────────────────────────
    dc        = _dist_transform(mask)
    _, max_r, _, pc = cv2.minMaxLoc(dc)
    x0, y0    = int(pc[0]), int(pc[1])

    # ── Step 3: S ───────────────────────────────────────────────
    s_mask    = (dc > t1 * max_r).astype(np.uint8) * 255

    # ── Step 4: Ds ──────────────────────────────────────────────
    ds        = _dist_transform(s_mask)
    ds_at_pc  = float(ds[y0, x0])

    # ── Step 5: T ───────────────────────────────────────────────
    yy, xx    = np.mgrid[0:h, 0:w]
    dist_to_pc = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2).astype(np.float32)
    t_mask    = ((dist_to_pc > ds_at_pc) & (yy >= y0)).astype(np.uint8) * 255

    # ── Step 6: S' ──────────────────────────────────────────────
    s_prime   = cv2.bitwise_and(s_mask, t_mask)

    # ── Step 7: P'C ─────────────────────────────────────────────
    x0p, y0p  = x0, y0     # fallback = original PC
    contours, _ = cv2.findContours(s_prime, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if contours:
        best = max(contours, key=cv2.contourArea)
        M    = cv2.moments(best)
        if M["m00"] != 0:
            x0p = int(M["m10"] / M["m00"])
            y0p = int(M["m01"] / M["m00"])

    # ── Step 8: constrained radius ──────────────────────────────
    final_r  = int(t2 * float(dc[y0p, x0p]))
    if final_r <= 4:
        final_r = max(4, int(max_r))     # safety: never degenerate

    # ── Step 9: exact-size square crop centred at P'C ──────────
    side     = 2 * final_r
    # Source window (clamped to image boundaries)
    src_x1   = max(0, x0p - final_r);  src_x2 = min(w, x0p + final_r)
    src_y1   = max(0, y0p - final_r);  src_y2 = min(h, y0p + final_r)
    # Destination offsets inside the `side × side` canvas
    dst_x1   = src_x1 - (x0p - final_r)
    dst_y1   = src_y1 - (y0p - final_r)
    dst_x2   = dst_x1 + (src_x2 - src_x1)
    dst_y2   = dst_y1 + (src_y2 - src_y1)

    canvas   = np.zeros((side, side, 3), dtype=img.dtype)
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

    # ── Visualisation ────────────────────────────────────────────
    vis = visualize_img.copy()
    cv2.circle(vis, (x0,  y0),  int(max_r),  (0, 200, 0),  2)   # green = max
    cv2.circle(vis, (x0p, y0p), final_r,      (0, 0, 220),  2)   # blue  = constrained
    cv2.circle(vis, (x0p, y0p), 4,             (220, 0, 0), -1)  # red dot = P'C

    return canvas, vis


# ═══════════════════════════════════════════════════════════════════
#  3. Shared geometry helpers
# ═══════════════════════════════════════════════════════════════════

def generate_heatmap(keypoint_location, heatmap_size, variance):
    """
    2-D Gaussian heatmap centred at keypoint_location (x, y).
    Source: adaptive_PROIE/dataset.py
    """
    x, y     = keypoint_location
    x_range  = torch.arange(0, heatmap_size[1])
    y_range  = torch.arange(0, heatmap_size[0])
    # meshgrid: default indexing is 'ij' (torch ≤ 1.12) → explicit kwarg for ≥ 1.13
    try:
        X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
    except TypeError:
        X, Y = torch.meshgrid(x_range, y_range)
    pos     = torch.stack((X, Y), dim=2)
    heatmap = torch.exp(
        -(torch.sum((pos - torch.tensor([x, y])) ** 2, dim=2)) / (2.0 * variance ** 2)
    )
    return heatmap


def center_and_pad_image(img: np.ndarray, kpts: np.ndarray):
    """
    Pad `img` to a square (max dimension) with black borders.
    Shift `kpts` (N×2 array of (x,y)) by the same offset.
    Source: adaptive_PROIE/dataset.py
    """
    h, w   = img.shape[:2]
    sz     = int(max(h, w))
    xoff   = (sz - w) // 2
    yoff   = (sz - h) // 2
    pad    = (np.zeros((sz, sz, 3), dtype=img.dtype)
              if img.ndim == 3 else np.zeros((sz, sz), dtype=img.dtype))
    pad[yoff:yoff + h, xoff:xoff + w] = img
    kp     = kpts.copy().astype(float)
    kp[:, 0] += xoff
    kp[:, 1] += yoff
    return pad, kp


def get_inter_square(img: np.ndarray, rotate_theta: float):
    """
    Rotate `img` by `rotate_theta` degrees around its centre, then crop
    the largest axis-aligned square that fits inside the rotated image.
    Source: adaptive_PROIE/ROI_extraction.py
    """
    h, w   = img.shape[:2]
    cx, cy = w // 2, h // 2
    half   = int(math.sqrt(2) * min(w, h) / 4)   # inscribed square half-side
    left   = cx - half;  right  = cx + half
    top    = cy - half;  bottom = cy + half
    M      = cv2.getRotationMatrix2D((cx, cy), rotate_theta, scale=1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    square  = rotated[max(0, top):min(h, bottom), max(0, left):min(w, right)]
    return square, rotated


# ═══════════════════════════════════════════════════════════════════
#  4. Stage 1 — CHSST hand segmentation
# ═══════════════════════════════════════════════════════════════════

def load_chsst(ckpt_path: str):
    """
    Load Seaformernet (preferred) or Topformernet.
    The training script saves the full model with torch.save(model, path),
    so we try loading directly first; fall back to state-dict loading.
    """
    # Try Seaformer first (model used in training.py with args.model=="Sea")
    model = None
    try:
        from CHSST.models.toptransformer.seaformer import Seaformernet
        model = Seaformernet()
        print("[CHSST] backbone: Seaformernet")
    except Exception as e_sea:
        try:
            from CHSST.models.toptransformer.basemodel import Topformernet
            model = Topformernet()
            print("[CHSST] backbone: Topformernet (Seaformer unavailable)")
        except Exception as e_top:
            raise RuntimeError(
                f"Could not import CHSST model.\n"
                f"  Seaformer: {e_sea}\n  Topformer: {e_top}\n"
                f"Ensure mmcv==1.4.0 and mmseg==0.20.0 are installed and "
                f"run from the RDRLA repo root."
            )

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Case A: torch.save(model, ...) was used → ckpt is the model object
    if hasattr(ckpt, "state_dict"):
        model = ckpt
    # Case B: state_dict saved directly or under a key
    elif isinstance(ckpt, dict):
        state = ckpt.get("state_dict", ckpt)
        # Strip leading "module." from DataParallel saves
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    else:
        raise ValueError(f"Unrecognised checkpoint format: {type(ckpt)}")

    model.to(DEVICE).eval()
    return model


def segment_palm(raw_path: str, seg_model) -> np.ndarray | None:
    """
    Run CHSST on one raw image.
    Returns: palm region on black background, tightly cropped to the hand
             bounding box — exactly what the LANet pipeline expects.
    Source: CHSST/palmSegmentation.py  process_one_img() + segfile()
    """
    img = cv2.imread(raw_path)
    if img is None:
        return None

    # Orientation normalisation based on L/R suffix in filename
    basename = os.path.basename(raw_path)
    suffix   = basename.split("_")[-1][0] if "_" in basename else "R"
    h, w     = img.shape[:2]
    if suffix == "L":
        if w > h:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.flip(img, 1)
    else:
        if w > h:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    h, w = img.shape[:2]

    # Forward pass — resize to 448×448, model output at 1/8 scale (56×56)
    inp = cv2.resize(img, (448, 448))
    inp_t = torch.tensor(
        np.transpose(inp, (2, 0, 1)) / 255.0, dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = seg_model(inp_t)   # [1, 2, H', W']

    seg_logit = out[0].cpu().numpy()                           # [2, H', W']
    seg_mask  = seg_logit.argmax(axis=0).astype(np.uint8) * 255
    seg_mask  = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Keep only the largest connected region (remove segmentation noise)
    contours, _ = cv2.findContours(seg_mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 100:          # degenerate mask
        return None

    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(filled, [best], 255)
    filled_3ch = cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR)

    palm = cv2.bitwise_and(img, filled_3ch)

    # Tight crop to bounding box
    pts       = best.reshape(-1, 2)
    xs, ys    = pts.min(0)
    xe, ye    = pts.max(0)
    crop      = palm[ys:ye, xs:xe]
    return crop if crop.size > 0 else None


# ═══════════════════════════════════════════════════════════════════
#  5. Stage 2 — PalmAlNet input preparation
# ═══════════════════════════════════════════════════════════════════

def prepare_lanet_input(seg_palm: np.ndarray):
    """
    Build the 56×56 three-channel tensor that LANet expects, together
    with the square-padded raw image and the rotation centre.

    Channel layout:  [0] B  [1] G  [2] heatmap
    (original code replaces channel-2 with the heatmap regardless of the
    underlying colour content — we replicate that exactly)

    Source: adaptive_PROIE/ROI_extraction.py  generate_net_ipt()

    Returns:
        tensor      (1, 3, 56, 56) on DEVICE,  or None on failure
        raw_img     square-padded BGR image (the 'large' input for warpAffine)
        r_center    (int, int) rotation centre used by cv2.getRotationMatrix2D
                    NOTE: the original code passes (cy, cx) — i.e. swapped —
                    to getRotationMatrix2D; we replicate that convention so
                    the pretrained weights produce correct results.
    """
    # Black-background mask (matches original: np.where((img==0), 0, 255)[:,:,2])
    mask_raw = np.where((seg_palm == 0), 0, 255)[:, :, 2].astype(np.uint8)

    # Inscribed circle centre on the full segmented image
    center_full, _ = find_circle_inform_hard(mask_raw)   # (cx, cy)

    kpts = np.array([[center_full[0], center_full[1]]], dtype=float)

    # Bounding-box crop (mirrors original: find contour → crop)
    gray = cv2.cvtColor(seg_palm, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None

    best = max(contours, key=cv2.contourArea)
    pts  = best.reshape(-1, 2)
    xs, ys = map(int, pts.min(0))
    xe, ye = map(int, pts.max(0))

    cropped = seg_palm[ys:ye, xs:xe]
    if cropped.size == 0:
        return None, None, None

    # Shift keypoint to cropped-image space
    kpts[:, 0] -= xs
    kpts[:, 1] -= ys

    # Pad to square, update keypoint
    padded, kpts_pad = center_and_pad_image(cropped, kpts)
    raw_img = padded.copy()
    h, w    = padded.shape[:2]

    cx_pad  = kpts_pad[0, 0]   # float x in padded image
    cy_pad  = kpts_pad[0, 1]   # float y in padded image

    # Heatmap coordinates in 56-pixel space
    # (original: center_changed = [int(center[1]*56/w), int(center[0]*56/h)])
    # where center = (cx_pad, cy_pad), so center[1]=cy_pad, center[0]=cx_pad
    hmap_x  = int(cy_pad * 56 / w)     # note: swapped, matches original exactly
    hmap_y  = int(cx_pad * 56 / h)
    hmap    = generate_heatmap((hmap_x, hmap_y), (56, 56), variance=2)

    # Build 56×56 input tensor
    small   = cv2.resize(padded, (56, 56))
    arr     = np.transpose(small, (2, 0, 1)) / 255.0
    tensor  = torch.from_numpy(arr.copy()).float()
    tensor[2] = hmap       # overwrite channel 2 with heatmap

    # Rotation centre (original: (int(center[1]), int(center[0])) = (cy, cx))
    # OpenCV getRotationMatrix2D expects (x, y), so this effectively swaps
    # them — we replicate the original convention unchanged.
    r_center = (int(cy_pad), int(cx_pad))

    return tensor.unsqueeze(0).to(DEVICE), raw_img, r_center


# ═══════════════════════════════════════════════════════════════════
#  6. Model loaders
# ═══════════════════════════════════════════════════════════════════

def load_lanet(ckpt_path: str) -> LAnet:
    """
    Load LANet weights.
    Training saves: {"LANet": state_dict}  (see adaptive_PROIE/train_LANet.py)
    """
    net  = LAnet().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("LANet", ckpt)   # handle both dict-wrapped and raw
    net.load_state_dict(state, strict=True)
    net.eval()
    print(f"[LANet] loaded from {ckpt_path}")
    return net


# ═══════════════════════════════════════════════════════════════════
#  7. Per-image ROI extraction (stages 2 + 3 combined)
# ═══════════════════════════════════════════════════════════════════

def extract_rois(seg_palm: np.ndarray,
                 lanet: LAnet,
                 base_name: str,
                 roi_dir: str,
                 vis_dir: str) -> int:
    """
    Given a segmented palm image (black background), run:
      1. PalmAlNet → predicted rotation angle θ
      2. Rotate image by θ around inscribed-circle centre
      3. Constrained inscribed-circle search (circle_better)
      4. 20 augmented 128×128 square crops (angles −30…+27°, step 3)

    Returns the number of ROI files successfully written.
    Source: adaptive_PROIE/ROI_extraction.py  process_single_img_ipt()
    """
    # ── Step 1: build LANet input ───────────────────────────────
    tensor_ipt, raw_img, r_center = prepare_lanet_input(seg_palm)
    if tensor_ipt is None:
        return 0

    # ── Step 2: predict rotation angle ─────────────────────────
    with torch.no_grad():
        theta = lanet(tensor_ipt).cpu().numpy()[0, 0]   # tanh ∈ (−1, 1)
    angle_deg = float(np.degrees(theta * np.pi))

    # ── Step 3: rotate raw (padded) image ─────────────────────
    M_rot    = cv2.getRotationMatrix2D(r_center, angle_deg, scale=1.0)
    aligned  = cv2.warpAffine(raw_img, M_rot,
                               (raw_img.shape[1], raw_img.shape[0]))
    aligned  = padding_img(aligned, padding=20)   # small black border so the
                                                   # circle search has room

    # Build binary mask of the aligned palm
    raw_mask = np.where(aligned <= 10, 0, 255)[:, :, 2].astype(np.uint8)

    # ── Step 4: constrained inscribed-circle search ─────────────
    vis_in          = aligned.copy()
    final_circle, vis_out = circle_better(aligned, vis_in, raw_mask, rate=1.1)

    if final_circle is None or final_circle.size == 0:
        return 0

    # Optional visualisation save
    if SAVE_VIS:
        h_v = 200
        w_v = max(1, int(200 * vis_out.shape[1] / max(vis_out.shape[0], 1)))
        cv2.imwrite(os.path.join(vis_dir, base_name + "_vis.jpg"),
                    cv2.resize(vis_out, (w_v, h_v)))

    # ── Step 5: rotational augmentation → 20 × 128×128 ROIs ────
    # angles: range(−30, 30, 3) = −30, −27, …, −3, 0, 3, …, 27  (20 values)
    saved = 0
    for aug_angle in range(-30, 30, 3):
        try:
            square_roi, _ = get_inter_square(final_circle, aug_angle)
        except Exception:
            continue
        if square_roi is None or square_roi.size == 0:
            continue
        roi_128 = cv2.resize(square_roi, (128, 128))
        fname   = f"{base_name}_{aug_angle}.jpg"
        cv2.imwrite(os.path.join(roi_dir, fname), roi_128)
        saved += 1

    return saved


# ═══════════════════════════════════════════════════════════════════
#  8. Main pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    for d in [SEG_TMP_DIR, ROI_OUT_DIR, VIS_OUT_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Load models ─────────────────────────────────────────────
    print("Loading CHSST segmentation model …")
    chsst = load_chsst(CHSST_CKPT)

    print("Loading LANet rotation predictor …")
    lanet = load_lanet(LANET_CKPT)

    # ── Collect images ───────────────────────────────────────────
    exts      = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    raw_imgs  = sorted(
        f for f in os.listdir(MPD_RAW_DIR)
        if os.path.splitext(f)[1].lower() in exts
    )
    print(f"Found {len(raw_imgs)} images in {MPD_RAW_DIR}\n")

    total_rois = 0
    seg_fail   = 0
    roi_fail   = 0

    for imgn in tqdm.tqdm(raw_imgs, desc="Extracting ROIs", unit="img"):
        raw_path  = os.path.join(MPD_RAW_DIR, imgn)
        base_name = os.path.splitext(imgn)[0]
        seg_path  = os.path.join(SEG_TMP_DIR, imgn)

        # ── Stage 1: hand segmentation (CHSST) ──────────────────
        try:
            palm_seg = segment_palm(raw_path, chsst)
        except Exception as e:
            tqdm.tqdm.write(f"  [seg-error] {imgn}: {e}")
            seg_fail += 1
            continue

        if palm_seg is None:
            tqdm.tqdm.write(f"  [seg-empty] {imgn}")
            seg_fail += 1
            continue

        cv2.imwrite(seg_path, palm_seg)   # save intermediate for inspection

        # ── Stage 2+3: align + ROI extraction ───────────────────
        try:
            n = extract_rois(palm_seg, lanet, base_name,
                             ROI_OUT_DIR, VIS_OUT_DIR)
        except Exception as e:
            tqdm.tqdm.write(f"  [roi-error] {imgn}: {e}")
            roi_fail += 1
            continue

        if n == 0:
            tqdm.tqdm.write(f"  [roi-empty] {imgn}")
            roi_fail += 1
        total_rois += n

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Input images   : {len(raw_imgs)}")
    print(f"  Seg failures   : {seg_fail}")
    print(f"  ROI failures   : {roi_fail}")
    print(f"  ROIs saved     : {total_rois}  (~{total_rois//max(1,len(raw_imgs)-seg_fail)} per image)")
    print(f"  Output dir     : {ROI_OUT_DIR}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
