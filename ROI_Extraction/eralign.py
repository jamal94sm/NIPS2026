"""
ROI Extraction — CASIA-MS / MPDv2 / Smartphone palmprint datasets
==================================================================
All user-facing settings are in the CONFIG block at the top.
No edits needed below the  ═══ END OF CONFIG ═══  line.
"""

# ══════════════════════════════════════════════════════════════
#  USER CONFIG  — edit only this block
# ══════════════════════════════════════════════════════════════

# ── Paths ──────────────────────────────────────────────────────
DIR_SOURCE = "/home/pai-ng/Jamal/MPDv2"
DIR_OUTPUT = "/home/pai-ng/Jamal/MPDv2_eralign_roi"

# ── Dataset preset ─────────────────────────────────────────────
#    "Smartphone"  |  "CASIA-MS"  |  "MPDv2"  |  "Generic"
DATASET_PRESET = "Smartphone"

# ── Debug options ──────────────────────────────────────────────
DEBUG             = True    # per-image console output
SAVE_OVERLAY_PASS = False   # save overlay for PASSED images too
                            # (overlays always saved for rejected)

# ── Validation & rejected-image saving ─────────────────────────
VALIDATE_ROI  = True   # run quality checks on every extracted ROI
SAVE_REJECTED = True   # save rejected ROIs + overlays to _rejected/ folder

# ── Inspect a single tricky image (set to None to skip) ────────
DEBUG_SINGLE_IMAGE = None
# DEBUG_SINGLE_IMAGE = "/home/pai-ng/Jamal/MPDv2/subject01/img001.jpg"

# ── Quality thresholds ─────────────────────────────────────────
#    Tuned for smartphone images with complex backgrounds.
#    A valid palm ROI is a near-square crop with:
#      - moderate, spatially uniform texture
#      - visible skin-colour content
#      - oriented palm-line structure (Gabor energy)
QUALITY_CONFIG = {
    "min_size"         : 80,     # min width AND height in pixels
    "max_size"         : 3000,   # max width OR height (sanity cap)
    "min_aspect"       : 0.5,    # width / height — palms are near-square
    "max_aspect"       : 2.0,
    "min_texture_std"  : 8.0,    # global pixel std-dev; blank bg → ~0
    "max_texture_std"  : 90.0,   # very high → likely a busy background
    "min_skin_ratio"   : 0.20,   # fraction of pixels in skin-HSV range
    "min_sharpness"    : 15.0,   # Laplacian variance; blurry → low
    "min_gabor_energy" : 3.0,    # oriented line energy; no lines → low
    "max_local_var_std": 55.0,   # std of per-block variance; background
                                 # grabs are spatially non-uniform → high
}

# ── Dataset-specific primary processing configs ─────────────────
DATASET_CONFIGS = {

    # Smartphone: complex / cluttered backgrounds, colour, variable lighting
    "Smartphone": {
        "THRESHOLD_SEG"           : 0,       # 0 = use Otsu (adapts to bg)
        "BI_THRESHOLD_SEG"        : False,
        "THRESHOLD_EDGE"          : 4,       # low → catch more edges in busy scene
        "BLUR"                    : True,
        "BLUR_SIGMA"              : 1.5,
        "KERNEL_SIZE"             : None,    # None = auto from image size
        "BLUR_ROTATE"             : True,
        "BLUR_SIGMA_ROTATE"       : 1.5,
        "BI_THRESHOLD_SEG_ROTATE" : False,
        "THRESHOLD_SEG_ROTATE"    : 0,
        "ratio_rotate"            : 2.0,
        "ratio"                   : 1.0,
    },

    # CASIA-MS: controlled lab, multi-spectral, clean dark background
    "CASIA-MS": {
        "THRESHOLD_SEG"           : 90,
        "BI_THRESHOLD_SEG"        : True,
        "THRESHOLD_EDGE"          : 10,
        "BLUR"                    : False,
        "BLUR_SIGMA"              : 0.05,
        "KERNEL_SIZE"             : (45, 45),
        "BLUR_ROTATE"             : True,
        "BLUR_SIGMA_ROTATE"       : 0.05,
        "BI_THRESHOLD_SEG_ROTATE" : False,
        "THRESHOLD_SEG_ROTATE"    : 90,
        "ratio_rotate"            : 1.0,
        "ratio"                   : 1.0,
    },

    # MPDv2: mixed indoor/outdoor conditions
    "MPDv2": {
        "THRESHOLD_SEG"           : 0,
        "BI_THRESHOLD_SEG"        : False,
        "THRESHOLD_EDGE"          : 5,
        "BLUR"                    : True,
        "BLUR_SIGMA"              : 1.0,
        "KERNEL_SIZE"             : None,
        "BLUR_ROTATE"             : True,
        "BLUR_SIGMA_ROTATE"       : 1.0,
        "BI_THRESHOLD_SEG_ROTATE" : False,
        "THRESHOLD_SEG_ROTATE"    : 0,
        "ratio_rotate"            : 2.0,
        "ratio"                   : 1.0,
    },

    # Generic: safe defaults for unknown datasets
    "Generic": {
        "THRESHOLD_SEG"           : 0,
        "BI_THRESHOLD_SEG"        : False,
        "THRESHOLD_EDGE"          : 7,
        "BLUR"                    : True,
        "BLUR_SIGMA"              : 1.5,
        "KERNEL_SIZE"             : None,
        "BLUR_ROTATE"             : True,
        "BLUR_SIGMA_ROTATE"       : 1.0,
        "BI_THRESHOLD_SEG_ROTATE" : False,
        "THRESHOLD_SEG_ROTATE"    : 0,
        "ratio_rotate"            : 2.0,
        "ratio"                   : 1.0,
    },
}

# ── Fallback cascade ───────────────────────────────────────────
#    Each entry overrides only the listed keys; the rest come
#    from the primary config. Tried in order when primary fails.
FALLBACK_CONFIGS = [
    # fallback 1 — stronger blur, keep Otsu
    {
        "THRESHOLD_SEG"    : 0,
        "BI_THRESHOLD_SEG" : False,
        "THRESHOLD_EDGE"   : 4,
        "BLUR"             : True,
        "BLUR_SIGMA"       : 2.5,
        "KERNEL_SIZE"      : None,
    },
    # fallback 2 — fixed low threshold + moderate blur
    {
        "THRESHOLD_SEG"    : 30,
        "BI_THRESHOLD_SEG" : True,
        "THRESHOLD_EDGE"   : 5,
        "BLUR"             : True,
        "BLUR_SIGMA"       : 1.5,
        "KERNEL_SIZE"      : None,
    },
    # fallback 3 — mid threshold + strong blur
    {
        "THRESHOLD_SEG"    : 60,
        "BI_THRESHOLD_SEG" : True,
        "THRESHOLD_EDGE"   : 3,
        "BLUR"             : True,
        "BLUR_SIGMA"       : 3.0,
        "KERNEL_SIZE"      : None,
    },
    # fallback 4 — high threshold, minimal blur
    {
        "THRESHOLD_SEG"    : 120,
        "BI_THRESHOLD_SEG" : True,
        "THRESHOLD_EDGE"   : 3,
        "BLUR"             : True,
        "BLUR_SIGMA"       : 0.5,
        "KERNEL_SIZE"      : None,
    },
]

# ══════════════════════════════════════════════════════════════
#  END OF CONFIG
# ══════════════════════════════════════════════════════════════

import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
import networkx as nx


# ──────────────────────────────────────────────────────────────
#  Gabor filter
# ──────────────────────────────────────────────────────────────

def GaborFilter(ksize, num_direction, sigma, lambd, gamma):
    assert num_direction % 2 == 0
    half   = ksize // 2
    sigma2 = 2 * sigma ** 2
    F      = np.zeros((num_direction, ksize, ksize))
    x, y   = np.meshgrid(range(-half, half + 1), range(-half, half + 1))
    for a in range(num_direction):
        th   = np.pi * a / num_direction
        xt   =  x * np.cos(th) + y * np.sin(th)
        yt   = -x * np.sin(th) + y * np.cos(th)
        F[a] = (np.exp(-(xt**2 + (gamma*yt)**2) / sigma2)
                * np.cos(2*np.pi*xt / lambd))
        F[a] -= F[a].mean()
    return F


# ──────────────────────────────────────────────────────────────
#  PalmBasic
# ──────────────────────────────────────────────────────────────

class PalmBasic:

    def gaussian_blur(self, img, kernel_size=(5, 5), sigma=2, blur=True):
        return cv2.GaussianBlur(img, kernel_size, sigma) if blur else img

    def threshold_image(self, img, threshold_val=20, bi_threshold=True):
        if bi_threshold:
            _, t = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)
        else:
            _, t = cv2.threshold(img, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return t

    def compute_edist(self, p1, p2):
        return float(np.linalg.norm(np.array(p1, float) - np.array(p2, float)))

    def find_largest_component(self, img):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(img)
        if len(stats) <= 1:
            return np.zeros_like(img), None, np.zeros_like(img)
        lbl  = int(np.argmax(stats[1:, -1])) + 1
        comp = np.zeros_like(img); comp[labels == lbl] = 255
        cnt  = self.find_contour(comp)
        cimg = cv2.drawContours(np.zeros_like(img, np.uint8), [cnt], -1, 255, 1)
        return comp, cnt, cimg

    def find_contour(self, img):
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return max(cnts, key=cv2.contourArea)

    def fill_contour(self, img, contour, filled=False):
        out = np.zeros_like(img, np.uint8)
        cv2.drawContours(out, [contour], -1, 255, cv2.FILLED if filled else 1)
        return out

    def erode_binary_image(self, img, kernel_size=(7,7), iterations=1, th_binary=180):
        er = cv2.erode(img, np.ones(kernel_size, np.uint8), iterations=iterations)
        _, b = cv2.threshold(er, th_binary, 255, cv2.THRESH_BINARY)
        return b

    def resize_image(self, img, ratio):
        return cv2.resize(img, (img.shape[1]//ratio, img.shape[0]//ratio))

    def hull_image(self, img, contour):
        eps  = 0.01 * cv2.arcLength(contour, True)
        apx  = cv2.approxPolyDP(contour, eps, True)
        hull = cv2.convexHull(apx, clockwise=False, returnPoints=True)
        hi   = cv2.drawContours(np.zeros_like(img, np.uint8),
                                 [hull], -1, 255, cv2.FILLED)
        return hull, hi

    def get_skeleton(self, img):
        return (skeletonize(img == 255) * 255).astype(np.uint8)

    def find_closest_path_graph(self, matrix):
        cleaned = np.zeros_like(matrix)
        G       = self._build_graph(matrix)
        deg1    = [n for n in G.nodes() if G.degree[n] == 1]
        if not deg1:
            return [cleaned, []]
        src  = min(deg1, key=lambda n: n[1])
        best = []
        for node in deg1:
            if node == src:
                continue
            try:
                p = nx.shortest_path(G, src, node)
                if len(p) > len(best):
                    best = p
            except nx.NetworkXNoPath:
                pass
        for pt in best:
            cleaned[pt[0], pt[1]] = 255
        return [cleaned, best]

    def _build_graph(self, matrix):
        nz = np.argwhere(matrix == 255)
        if len(nz) == 0:
            return nx.Graph()
        r0, c0 = nz.min(axis=0); r1, c1 = nz.max(axis=0)
        G = nx.Graph()
        dirs = [(-1,0),(-1,-1),(0,-1),(1,-1)]
        for i in range(r0, r1+1):
            for j in range(c0, c1+1):
                if matrix[i,j] == 255:
                    G.add_node((i,j))
                    for d in dirs:
                        nb = (i+d[0], j+d[1])
                        if r0<=nb[0]<=r1 and c0<=nb[1]<=c1 and matrix[nb[0],nb[1]]==255:
                            G.add_edge((i,j), nb)
        return G

    def judge_valley(self, path):
        if not path:
            return False, 0
        ri  = max(range(len(path)), key=lambda k: path[k][1])
        n1, n2 = len(path[ri+1:]), len(path[:ri+1])
        return min(n1,n2) > len(path)/3, ri

    def find_farthest_point(self, path, th):
        lp   = len(path)
        path = np.array(path)[:, [1,0]]
        li, ri = int(lp/5*2), lp-1
        ln, rn = path[li], path[ri]

        def _flag(li, ri, ln, rn):
            fv  = ln - rn
            est = path[li+1:ri] - rn
            if len(est) == 0:
                return False
            cp  = np.cross(est, fv)
            dp  = np.dot(est, fv)
            ang = np.degrees(np.arctan2(cp, dp))
            return np.abs(np.abs(np.mean(ang)) - np.mean(np.abs(ang))) > 1.5

        flag = _flag(li, ri, ln, rn)
        while flag and li < lp-1 and ri > 0:
            if li < lp//3:
                li += 2; ln = path[li]
            elif ri > int(lp/5*4):
                ri -= 2; rn = path[ri]
            else:
                break
            flag = _flag(li, ri, ln, rn)

        pv   = path[li:ri+1]
        if len(pv) < 3:
            return pv
        x1,y1 = pv[0]; x2,y2 = pv[-1]
        A,B   = y2-y1, x1-x2
        C     = x2*y1 - x1*y2
        inner = pv[1:-1]
        dist  = np.abs(A*inner[:,0]+B*inner[:,1]+C) / (np.sqrt(A**2+B**2)+1e-9)
        mx    = int(np.argmax(dist))
        if dist[mx] > th:
            pv = pv[:mx+4]
        return pv

    def find_closest_white_point(self, image, point):
        nz = cv2.findNonZero(image)
        if nz is None:
            return tuple(np.array(point, int)), float('inf')
        dist = np.linalg.norm(nz - point, axis=2)
        idx  = int(np.argmin(dist))
        return tuple(nz[idx][0]), float(dist[idx])


# ──────────────────────────────────────────────────────────────
#  ROI geometry
# ──────────────────────────────────────────────────────────────

pad = 0

def extract_roi(p1, p2, img, color=(0,255,0), thickness=2):
    d    = np.linalg.norm(p2 - p1)
    dirv = (p2 - p1) / (d + 1e-9)
    norm = np.array([dirv[1], -dirv[0]])
    s    = int(d/6*7); he = int(d/12); ds = int(d/6)

    C = (p1 + (ds+s)*norm - he*dirv).astype(np.int32)
    D = (C + s*dirv).astype(np.int32)
    E = (D - s*norm).astype(np.int32)
    F = (C - s*norm).astype(np.int32)

    pad_img = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                  cv2.BORDER_CONSTANT, value=0)
    Cp,Dp,Ep,Fp = C+pad, D+pad, E+pad, F+pad
    pts    = np.array([Cp,Dp,Ep,Fp], dtype="float32")
    width  = max(np.linalg.norm(Ep-Dp), np.linalg.norm(Fp-Cp))
    height = max(np.linalg.norm(Dp-Cp), np.linalg.norm(Ep-Fp))
    dst    = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], "float32")
    M      = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(pad_img, M, (int(width), int(height)))
    cv2.polylines(pad_img, [np.array([Cp,Dp,Ep,Fp])], True, color, thickness)
    corners = [(Cp[0],Cp[1]),(Dp[0],Dp[1]),(Ep[0],Ep[1]),(Fp[0],Fp[1])]
    return warped, corners, pad_img


# ──────────────────────────────────────────────────────────────
#  Kernel size helper
# ──────────────────────────────────────────────────────────────

def _resolve_kernel(cfg, shape):
    if cfg.get("KERNEL_SIZE") is not None:
        return cfg["KERNEL_SIZE"]
    k = max(15, min(shape[0], shape[1]) // 10)
    return (k if k%2==1 else k+1,) * 2


# ──────────────────────────────────────────────────────────────
#  GetROI — config-driven
# ──────────────────────────────────────────────────────────────

class GetROI(PalmBasic):

    def __init__(self, img, config):
        self.cfg          = config
        self.ratio_rotate = config["ratio_rotate"]
        self.ratio        = config["ratio"]
        self.ori_img      = img
        self.w, self.h    = img.shape
        self.w_rotate     = max(1, int(self.w / self.ratio_rotate))
        self.h_rotate     = max(1, int(self.h / self.ratio_rotate))
        self._ksize       = _resolve_kernel(config, img.shape)

    # ── helpers ───────────────────────────────────────────────

    def _rotate(self, img, angle):
        cx, cy = img.shape[1]//2, img.shape[0]//2
        M = cv2.getRotationMatrix2D((cx,cy), angle, 1.0)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    def rotate_keypoints(self, img, angle, kps):
        cx, cy = img.shape[1]//2, img.shape[0]//2
        M  = cv2.getRotationMatrix2D((cx,cy), angle, 1.0)
        pt = np.hstack([kps, np.ones((kps.shape[0],1))])
        return np.round(pt @ M.T).astype(int)

    def _detect_edges(self, img, thr=10):
        lap   = cv2.Laplacian(img, cv2.CV_64F, ksize=7).clip(min=0)
        edges = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, e  = cv2.threshold(edges, thr, 255, cv2.THRESH_BINARY)
        return cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    def _select_palm(self, edges_img):
        _, lbl, stats = cv2.connectedComponentsWithStats(edges_img, connectivity=8)[:3]
        return lbl, np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]

    def _rough_orientation(self, labels, sidx, hull_img, gabors):
        eimg = np.zeros((self.w_rotate, self.h_rotate))
        for idx in sidx[1:4]:
            if np.sum(labels == idx+1) > 20/self.ratio:
                eimg[labels == idx+1] = 255
        pts = np.argwhere(eimg > 0)
        if len(pts) == 0:
            return 0.0
        cx, cy   = np.mean(pts,axis=0)[1], np.mean(pts,axis=0)[0]
        fhull    = cv2.convexHull(pts[:,[1,0]])
        fhfill   = cv2.drawContours(np.zeros_like(eimg, np.uint8),
                                     [fhull], -1, 255, cv2.FILLED)
        rem      = np.argwhere((fhfill & hull_img) ^ hull_img)
        if len(rem) == 0:
            return 0.0
        ctr      = np.mean(rem, axis=0)
        vect     = np.array([cx-ctr[1], cy-ctr[0]])
        ang_r    = np.degrees(np.arctan2(vect[1], vect[0]))
        mx, my   = eimg.shape[1]//2, eimg.shape[0]//2
        Mr       = cv2.getRotationMatrix2D((mx,my), ang_r, 1)
        erot     = cv2.warpAffine(eimg, Mr, (eimg.shape[1], eimg.shape[0]))
        res      = np.stack([cv2.filter2D(erot.astype(np.float32), -1, gabors[j])
                             for j in range(len(gabors))])
        vi       = np.argmax(res, axis=0)[erot == 255]
        if len(vi) == 0:
            return ang_r + 180
        u, c     = np.unique(vi, return_counts=True)
        fine     = 90 - 180/len(gabors) * u[np.argmax(c)]
        return ang_r - fine + 180

    # ── rotation stage ────────────────────────────────────────

    def run_rotate(self):
        cfg    = self.cfg
        img    = self.resize_image(self.ori_img, int(self.ratio_rotate))
        gabors = GaborFilter(15, 12, sigma=3, lambd=10, gamma=0.2)
        blur   = self.gaussian_blur(img, sigma=cfg["BLUR_SIGMA_ROTATE"],
                                    blur=cfg["BLUR_ROTATE"])
        binary = self.threshold_image(blur,
                                      threshold_val=cfg["THRESHOLD_SEG_ROTATE"],
                                      bi_threshold=cfg["BI_THRESHOLD_SEG_ROTATE"])
        comp, cnt, _   = self.find_largest_component(binary)
        hull, himg     = self.hull_image(img, cnt)
        edges          = self._detect_edges(blur & comp, cfg["THRESHOLD_EDGE"])
        lbl, sidx      = self._select_palm(edges & himg)
        self.rotation_angle = self._rough_orientation(lbl, sidx, himg, gabors)
        self.norm_img  = self._rotate(self.ori_img, self.rotation_angle)
        h, w           = self.norm_img.shape[:2]
        self.cut_norm_img = self.norm_img[0:h, 0:int(w/4*3)]

    # ── localisation stage ────────────────────────────────────

    def run_localization(self):
        cfg = self.cfg
        img = self.cut_norm_img
        self.w, self.h = img.shape

        blur   = self.gaussian_blur(img, sigma=cfg["BLUR_SIGMA"], blur=cfg["BLUR"])
        binary = self.threshold_image(blur,
                                      threshold_val=cfg["THRESHOLD_SEG"],
                                      bi_threshold=cfg["BI_THRESHOLD_SEG"])
        comp, cnt, _ = self.find_largest_component(binary)
        if cnt is None:
            return False

        cs = self.erode_binary_image(comp, self._ksize)
        if cv2.countNonZero(cs) == 0:
            cs = self.erode_binary_image(comp, (15,15))
        if cv2.countNonZero(cs) == 0:
            return False

        hull, himg     = self.hull_image(img, cnt)
        edges          = self._detect_edges(blur & comp, cfg["THRESHOLD_EDGE"])
        hs             = self.erode_binary_image(himg, self._ksize)
        if cv2.countNonZero(hs) == 0:
            hs = self.erode_binary_image(himg, (15,15))
        if cv2.countNonZero(hs) == 0:
            return False

        hcnt_s   = self.find_contour(hs)
        hcimg_s  = self.fill_contour(img, hcnt_s)
        cnt_img  = cv2.drawContours(np.zeros_like(img,np.uint8), [cnt], -1, 255, 1)
        concave  = cnt_img & hs
        lbl, sidx = self._select_palm(edges & hs)
        palm_len = (np.max(hull[:,0,:][:,0]) - np.min(hull[:,0,:][:,0]))

        self.finger_edges_img = self._process_contour(
            sidx, lbl, palm_len, hcimg_s, concave, cnt_img)
        self.palm_contour_img, self.palm_contour_coord = \
            self._inner_contour(self.finger_edges_img)
        self.finger_lines        = self._concave_fingers(
            hcnt_s, hs, palm_len,
            np.max(hull[:,0,:][:,0]), self.palm_contour_coord)
        self.finger_lines_sorted = self._sort_lines(self.finger_lines)
        self.keypoints           = self._detect_keypoints(self.finger_lines_sorted)
        self.show_img            = cv2.cvtColor(np.copy(self.norm_img),
                                               cv2.COLOR_GRAY2BGR)
        if self.keypoints is None:
            return False

        idx = np.argsort(self.keypoints[:,1])
        self.keypoints = self.keypoints[idx]
        self._localise_keypoints(self.keypoints)
        return True

    # ── keypoint detection ────────────────────────────────────

    def _localise_keypoints(self, kps):
        if len(kps) == 4:
            d1 = (self.compute_edist(kps[0],kps[2]) +
                  self.compute_edist(kps[0],kps[1]))
            d2 = (self.compute_edist(kps[1],kps[3]) +
                  self.compute_edist(kps[2],kps[3]))
            kp1,kp2 = (kps[0],kps[2]) if d2>d1 else (kps[1],kps[3])
        else:
            kp1,kp2 = kps[0], kps[1]
        self.keypoints_localization = [kp1, kp2]

    def _detect_keypoints(self, lines):
        def right_of(p1,p2,p):
            return ((p2[0]-p1[0])*(p[1]-p1[1])-(p2[1]-p1[1])*(p[0]-p1[0])) < 0
        def best(s1,s2):
            for a in s1:
                for b in s2:
                    if (all(not right_of(a,b,p) for p in s1 if p is not a) and
                            all(not right_of(a,b,p) for p in s2 if p is not b)):
                        return a,b
            return s1[np.argmax(s1[:,0])], s2[np.argmax(s2[:,0])]
        if len(lines) == 4:
            kps=np.zeros((4,2),np.int32)
            r1=best(lines[0],lines[2]); r2=best(lines[1],lines[3])
            kps[0],kps[2]=r1; kps[1],kps[3]=r2
        elif len(lines) in (2,3):
            kps=np.zeros((2,2),np.int32)
            r=best(lines[0],lines[-1]); kps[0],kps[1]=r
        else:
            return None
        return kps

    def _sort_lines(self, lines, num_pts=80):
        half = num_pts//2; out=[]
        for line in sorted(lines, key=lambda l: np.min(l[:,1])):
            mi=np.argmax(line[:,0])
            out.append(line[max(0,mi-half):min(line.shape[0],mi+half+1)])
        return out

    def _concave_fingers(self, hcnt_s, hs, palm_len, hull_right, palm_cnt):
        mask = np.array([
            cv2.pointPolygonTest(hcnt_s[:,0,:].astype(np.float32),
                                 (float(p[0]),float(p[1])), True) > 0.5
            for p in palm_cnt])
        trans = np.where(mask[:-1]!=mask[1:])[0]+1
        if mask[0]:  trans=np.insert(trans,0,0)
        if mask[-1]: trans=np.append(trans,len(palm_cnt))
        rough=[palm_cnt[s:e] for s,e in zip(trans[::2],trans[1::2]) if e-s>5]
        if len(rough)<=4: return rough
        fingers=[]
        for i in np.argsort([r.shape[0] for r in rough])[::-1]:
            if len(fingers)>=4: break
            line=rough[i]
            if line.shape[0]<=20: continue
            rpt=line[np.argmax(line[:,0])]
            cp,cd=self.find_closest_white_point(hs,rpt)
            rx=np.max(line[:,0])
            if cd < 20/self.ratio:
                if cd/line.shape[0] > 1/3: fingers.append(line)
            else:
                if hull_right-rx > palm_len/4: fingers.append(line)
        return fingers

    def _inner_contour(self, fe):
        cnts,_ = cv2.findContours(fe, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        top2   = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
        sel    = top2[1] if len(top2)>1 else top2[0]
        return self.fill_contour(self.ori_img, sel), sel[:,0,:]

    def _process_contour(self, sidx, labels, palm_len,
                         hcimg_s, concave, cnt_img):
        ch=concave|hcimg_s; fe=concave|hcimg_s
        th_rm=6/self.ratio; count=0
        hpts=np.argwhere(cnt_img==255)

        for ind in sidx:
            if count>3: break
            ge=np.zeros((self.w,self.h)); ge[labels==ind+1]=255
            sk=self.get_skeleton(ge)
            if sk.sum()/255 < 25: continue
            res=self.find_closest_path_graph(sk)
            if not res[1]: continue
            valley,_=self.judge_valley(res[1])

            if not valley:
                ln=res[1][0]
                lhp=hpts[hpts[:,0]==ln[0]]
                if len(lhp)==0: continue
                if ln[1]-np.min(lhp[:,1]) > palm_len/6: continue
                pv=self.find_farthest_point(res[1], th_rm)
                sk=np.zeros_like(self.cut_norm_img)
                cv2.polylines(sk,[pv],False,255,1)
                cp,_=self.find_closest_white_point(ch,pv[0])
                cv2.line(sk,cp,tuple(pv[0]),255,1)
            else:
                dn,sn=res[1][-1],res[1][0]
                dhp=hpts[hpts[:,0]==dn[0]]; shp=hpts[hpts[:,0]==sn[0]]
                if len(dhp)==0 or len(shp)==0: continue
                if min(dn[1]-np.min(dhp[:,1]), sn[1]-np.min(shp[:,1])) > palm_len/5:
                    continue
                sk=res[0].astype(np.uint8)
                cp,_=self.find_closest_white_point(ch,dn[::-1])
                cv2.line(sk,cp,dn[::-1],255,1)
                cp,_=self.find_closest_white_point(ch,sn[::-1])
                cv2.line(sk,cp,sn[::-1],255,1)

            count+=1; fe=fe|sk

        return fe.astype(np.uint8)


# ──────────────────────────────────────────────────────────────
#  Quality validation
# ──────────────────────────────────────────────────────────────

def _skin_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1  = cv2.inRange(hsv, np.array([0,  15, 40],np.uint8),
                           np.array([25,255,255],np.uint8))
    m2  = cv2.inRange(hsv, np.array([155,15, 40],np.uint8),
                           np.array([180,255,255],np.uint8))
    return m1 | m2


def _gabor_energy(gray):
    vals=[]
    for ang in range(0,180,30):
        k = cv2.getGaborKernel((21,21), sigma=4.0,
                               theta=np.deg2rad(ang),
                               lambd=10.0, gamma=0.5, psi=0,
                               ktype=cv2.CV_32F)
        k /= k.sum()+1e-9
        r  = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, k)
        vals.append(float(np.mean(np.abs(r))))
    return float(np.mean(vals))


def _local_var_std(gray, block=32):
    h,w=gray.shape; g=gray.astype(np.float32); blocks=[]
    for r in range(0,h-block,block):
        for c in range(0,w-block,block):
            blocks.append(float(g[r:r+block,c:c+block].var()))
    return float(np.std(blocks)) if blocks else 0.0


def validate_roi(roi_bgr, qcfg=None):
    """
    Returns (passed:bool, reasons:list[str], scores:dict).
    All checks can be tuned via QUALITY_CONFIG at the top.
    """
    if qcfg is None:
        qcfg = QUALITY_CONFIG
    failed=[]; scores={}
    h,w = roi_bgr.shape[:2]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 1. size
    scores["w"]=w; scores["h"]=h
    if w<qcfg["min_size"] or h<qcfg["min_size"]:
        failed.append(f"too_small({w}x{h})")
    if w>qcfg["max_size"] or h>qcfg["max_size"]:
        failed.append(f"too_large({w}x{h})")

    # 2. aspect ratio
    asp=w/(h+1e-9); scores["aspect"]=round(asp,3)
    if not (qcfg["min_aspect"]<=asp<=qcfg["max_aspect"]):
        failed.append(f"bad_aspect({asp:.2f})")

    # 3. texture std-dev — uniform skin has moderate std
    tstd=float(gray.std()); scores["texture_std"]=round(tstd,2)
    if tstd<qcfg["min_texture_std"]:  failed.append(f"low_texture({tstd:.1f})")
    if tstd>qcfg["max_texture_std"]:  failed.append(f"high_texture({tstd:.1f})")

    # 4. skin colour ratio
    skin=_skin_mask(roi_bgr)
    skrat=float(np.count_nonzero(skin))/(w*h+1e-9)
    scores["skin_ratio"]=round(skrat,3)
    if skrat<qcfg["min_skin_ratio"]:  failed.append(f"low_skin({skrat:.2f})")

    # 5. sharpness (Laplacian variance)
    sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    scores["sharpness"]=round(sharp,2)
    if sharp<qcfg["min_sharpness"]:   failed.append(f"blurry({sharp:.1f})")

    # 6. Gabor palm-line energy
    ge=_gabor_energy(gray); scores["gabor_energy"]=round(ge,3)
    if ge<qcfg["min_gabor_energy"]:   failed.append(f"no_lines({ge:.2f})")

    # 7. spatial uniformity — real palm ROIs have low local-variance spread
    lvs=_local_var_std(gray); scores["local_var_std"]=round(lvs,2)
    if lvs>qcfg["max_local_var_std"]: failed.append(f"non_uniform({lvs:.1f})")

    return len(failed)==0, failed, scores


# ──────────────────────────────────────────────────────────────
#  Overlay visualiser
# ──────────────────────────────────────────────────────────────

def save_overlay(orig_color, get_roi_obj, out_path,
                 scores=None, reasons=None):
    vis = (orig_color.copy() if len(orig_color.shape)==3
           else cv2.cvtColor(orig_color, cv2.COLOR_GRAY2BGR))
    kk  = np.array(get_roi_obj.keypoints_localization)
    kkt = get_roi_obj.rotate_keypoints(
        get_roi_obj.norm_img, -get_roi_obj.rotation_angle, kk)

    for pt in kkt:
        cv2.circle(vis, tuple(pt.astype(int)), 10, (0,0,255), -1)

    d    = np.linalg.norm(kkt[1]-kkt[0])
    dirv = (kkt[1]-kkt[0])/(d+1e-9)
    norm = np.array([dirv[1],-dirv[0]])
    s    = int(d/6*7); he=int(d/12); ds=int(d/6)
    C=(kkt[0]+(ds+s)*norm-he*dirv).astype(np.int32)
    D=(C+s*dirv).astype(np.int32)
    E=(D-s*norm).astype(np.int32)
    F=(C-s*norm).astype(np.int32)
    cv2.polylines(vis,[np.array([C,D,E,F])],True,(0,255,0),3)

    if reasons is not None:
        label  = "PASS" if not reasons else "FAIL: "+" | ".join(reasons)
        colour = (0,200,0) if not reasons else (0,0,220)
        cv2.putText(vis, label, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
    if scores is not None:
        y=80
        for k,v in scores.items():
            cv2.putText(vis,f"{k}={v}",(10,y),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,0),1)
            y+=24

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, vis)


# ──────────────────────────────────────────────────────────────
#  Core: try primary then fallbacks
# ──────────────────────────────────────────────────────────────

def _merged(primary, override, shape):
    m=dict(primary); m.update(override)
    m["KERNEL_SIZE"]=_resolve_kernel(m, shape)
    return m


def try_extract(gray_img, color_img, primary_cfg, fallback_cfgs, debug=False):
    """
    Returns (roi_bgr, cfg_label, get_roi_obj)  or  (None, None, None).
    Geometry runs on gray_img; final warp runs on color_img.
    """
    attempts = [("primary", primary_cfg)] + \
               [(f"fallback_{i+1}", fb) for i,fb in enumerate(fallback_cfgs)]

    for label, override in attempts:
        cfg = _merged(primary_cfg, override, gray_img.shape)
        try:
            gr = GetROI(gray_img, cfg)
            gr.run_rotate()
            gr.norm_img = gr._rotate(gr.ori_img, gr.rotation_angle)
            h,w = gr.norm_img.shape[:2]
            gr.cut_norm_img = gr.norm_img[0:h, 0:int(w/4*3)]
            ok  = gr.run_localization()
        except Exception as e:
            if debug: print(f"      [{label}] exception: {e}")
            continue
        if ok:
            kk  = np.array(gr.keypoints_localization)
            kkt = gr.rotate_keypoints(gr.norm_img, -gr.rotation_angle, kk)
            roi,_,_ = extract_roi(kkt[0], kkt[1], color_img,
                                  color=(0,255,0), thickness=2)
            return roi, label, gr
        if debug: print(f"      [{label}] localization failed")

    return None, None, None


# ──────────────────────────────────────────────────────────────
#  Single-image debug helper
# ──────────────────────────────────────────────────────────────

def debug_single_image(img_path, preset=None):
    if preset is None: preset=DATASET_PRESET
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None: print(f"Cannot read: {img_path}"); return
    cfg  = dict(DATASET_CONFIGS[preset])
    cfg["KERNEL_SIZE"] = _resolve_kernel(cfg, gray.shape)
    out  = os.path.dirname(img_path)

    cv2.imwrite(os.path.join(out,"dbg_0_raw.jpg"), gray)

    blur = (cv2.GaussianBlur(gray,(5,5),cfg["BLUR_SIGMA"])
            if cfg["BLUR"] else gray.copy())
    cv2.imwrite(os.path.join(out,"dbg_1_blurred.jpg"), blur)

    otsu_val,_ = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if cfg["BI_THRESHOLD_SEG"]:
        _,thr = cv2.threshold(blur, cfg["THRESHOLD_SEG"], 255, cv2.THRESH_BINARY)
    else:
        _,thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(out,"dbg_2_threshold.jpg"), thr)

    _,lbl,stats,_ = cv2.connectedComponentsWithStats(thr)
    comp=np.zeros_like(gray)
    if len(stats)>1: comp[lbl==int(np.argmax(stats[1:,-1]))+1]=255
    cv2.imwrite(os.path.join(out,"dbg_3_component.jpg"), comp)

    eroded=cv2.erode(comp, np.ones(cfg["KERNEL_SIZE"],np.uint8))
    cv2.imwrite(os.path.join(out,"dbg_4_eroded.jpg"), eroded)

    lap   = cv2.Laplacian(blur&comp, cv2.CV_64F, ksize=7).clip(min=0)
    edges = cv2.normalize(lap,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    _,eb  = cv2.threshold(edges, cfg["THRESHOLD_EDGE"], 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(out,"dbg_5_edges.jpg"), eb)

    color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if color is not None:
        cv2.imwrite(os.path.join(out,"dbg_6_skin.jpg"), _skin_mask(color))

    print(f"\nDebug images → {out}")
    print(f"  Size     : {gray.shape[1]}x{gray.shape[0]}")
    print(f"  Otsu val : {otsu_val:.1f}")
    print(f"  Kernel   : {cfg['KERNEL_SIZE']}")
    print(f"  Preset   : {preset}")


# ──────────────────────────────────────────────────────────────
#  Main extraction runner
# ──────────────────────────────────────────────────────────────

def run_extraction(dir_source, dir_output,
                   dataset_preset=None, debug=None):
    if dataset_preset is None: dataset_preset=DATASET_PRESET
    if debug          is None: debug=DEBUG

    if dataset_preset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown preset '{dataset_preset}'. "
                         f"Valid: {list(DATASET_CONFIGS.keys())}")

    primary_cfg = DATASET_CONFIGS[dataset_preset]
    rej_root    = dir_output+"_rejected"
    ov_root     = dir_output+"_overlays"
    os.makedirs(dir_output, exist_ok=True)

    all_images=[]
    for root,_,files in os.walk(dir_source):
        for fn in sorted(files):
            if fn.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                all_images.append((root,fn))

    print(f"\n{'='*58}")
    print(f"  Source   : {dir_source}")
    print(f"  Output   : {dir_output}")
    print(f"  Preset   : {dataset_preset}")
    print(f"  Validate : {VALIDATE_ROI}")
    print(f"  Images   : {len(all_images)}")
    print(f"{'='*58}\n")

    n_ok=n_fail=n_reject=0; usage={}

    for idx,(root,fn) in enumerate(all_images):
        img_path = os.path.join(root,fn)
        rel      = os.path.relpath(root, dir_source)
        out_dir  = os.path.join(dir_output, rel)
        os.makedirs(out_dir, exist_ok=True)

        color_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if color_img is None:
            print(f"[{idx:05d}] SKIP (unreadable): {fn}")
            n_fail+=1; continue

        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        if debug:
            print(f"[{idx:05d}] {fn}  ({color_img.shape[1]}x{color_img.shape[0]})")

        roi, used, gr = try_extract(gray_img, color_img,
                                    primary_cfg, FALLBACK_CONFIGS, debug=debug)

        if roi is None:
            print(f"[{idx:05d}] FAILED all configs: {fn}")
            n_fail+=1; continue

        # quality check
        if VALIDATE_ROI:
            passed,reasons,scores = validate_roi(roi)
        else:
            passed,reasons,scores = True,[],{}

        # BGR → RGB for saving
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        if passed:
            out_path = os.path.join(out_dir, fn)
            cv2.imwrite(out_path, roi_rgb)
            usage[used] = usage.get(used,0)+1

            if SAVE_OVERLAY_PASS and gr is not None:
                ov_dir=os.path.join(ov_root,rel); os.makedirs(ov_dir,exist_ok=True)
                save_overlay(color_img, gr, os.path.join(ov_dir,fn),
                             scores=scores, reasons=reasons)
            if debug:
                print(f"        OK  [{used}]  {scores}")
            n_ok+=1

        else:
            if SAVE_REJECTED and gr is not None:
                rej_dir=os.path.join(rej_root,rel); os.makedirs(rej_dir,exist_ok=True)
                cv2.imwrite(os.path.join(rej_dir,fn), roi_rgb)
                save_overlay(color_img, gr,
                             os.path.join(rej_dir,"overlay_"+fn),
                             scores=scores, reasons=reasons)
            print(f"[{idx:05d}] REJECTED {reasons}  {fn}")
            n_reject+=1

    # ── final summary ─────────────────────────────────────────
    total = n_ok+n_fail+n_reject
    print(f"\n{'='*58}")
    print(f"  Done.")
    print(f"  Passed   : {n_ok}")
    print(f"  Rejected : {n_reject}  (saved → {rej_root})")
    print(f"  Failed   : {n_fail}")
    print(f"  Total    : {total}")
    if usage:
        print(f"\n  Config usage:")
        for lbl,cnt in sorted(usage.items()):
            print(f"    {lbl:15s}: {cnt:5d}  ({100*cnt/max(n_ok,1):.1f}%)")
    print(f"{'='*58}\n")


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if DEBUG_SINGLE_IMAGE is not None:
        debug_single_image(DEBUG_SINGLE_IMAGE, preset=DATASET_PRESET)

    run_extraction(
        dir_source     = DIR_SOURCE,
        dir_output     = DIR_OUTPUT,
        dataset_preset = DATASET_PRESET,
        debug          = DEBUG,
    )
