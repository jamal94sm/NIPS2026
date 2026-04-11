"""
Palm ROI Annotation Tool
=========================
Exactly 5 keypoints per image:
  P1 — index base
  P2 — mid-ring web
  P3 — little base
  P4 — wrist
  P5 — thumb

Left-click  → place next point (blocked after 5)
Right-click → remove last point
Save & Next → enforce 5 pts → save keypoints JSON → extract ROI → advance
Skip        → mark image as unusable, record in skipped_images.json
Back        → go to previous image

Output layout (all relative to the opened image folder):
  ./keypoints/<imagename>.json   ← per-image keypoint file
  ./roi/<imagename>.<ext>        ← extracted square ROI (224x224)
  skipped_images.json            ← list of skipped filenames

ROI method — Lin et al. 2024:
  D      = |P1 - P3|
  M      = midpoint(P1, P3)
  perp   = unit vector perpendicular to P1->P3, validated toward P4 (wrist)
  O'     = M + 0.8 * D * perp
  side   = 1.2 * D
  Extraction via single PIL affine transform — no rotate-then-crop step.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
import os
import glob
import math
import numpy as np

import pillow_heif
pillow_heif.register_heif_opener()

# ── Configuration ──────────────────────────────────────────────────────────────
POINT_RADIUS  = 6
MAX_DISPLAY_W = 900
MAX_DISPLAY_H = 700
MAX_POINTS    = 5
ROI_SIZE      = 224

OFFSET_RATIO  = 0.45
SIDE_RATIO    = 0.85

POINT_COLORS = ["#00FF88", "#FFD700", "#FF6B6B", "#87CEEB", "#FF8C00"]
POINT_LABELS = [
    "P1 index corner",
    "P2 mid-ring web",
    "P3 little corner",
    "P4 wrist",
    "P5 thumb",
]
LINE_COLOR  = "#00FF88"
LABEL_COLOR = "#FFFFFF"
# ──────────────────────────────────────────────────────────────────────────────


def _compute_roi_corners(pts_original, offset_ratio, side_ratio):
    """
    Compute the 4 corners of the Lin et al. 2024 ROI square in
    original image coordinates.

    Returns [top_left, top_right, bottom_right, bottom_left] where
    'bottom' is the wrist side and 'top' is the finger side.

    The perpendicular direction is validated against P4 (wrist) so the
    ROI centre always falls inside the palm regardless of hand orientation.
    """
    p1x, p1y = pts_original[0]   # index base
    p3x, p3y = pts_original[2]   # little base
    p4x, p4y = pts_original[3]   # wrist (used only for direction check)

    ax = p3x - p1x
    ay = p3y - p1y
    D = math.hypot(ax, ay)
    if D < 1:
        raise ValueError("P1 and P3 are too close together.")

    ux = ax / D   # unit along P1->P3
    uy = ay / D

    # CCW perpendicular candidate
    px = -uy
    py =  ux

    mx = (p1x + p3x) / 2.0
    my = (p1y + p3y) / 2.0

    # Flip perpendicular if it points away from wrist
    if (p4x - mx) * px + (p4y - my) * py < 0:
        px, py = -px, -py

    side = side_ratio * D
    half = side / 2.0

    ox = mx + offset_ratio * D * px
    oy = my + offset_ratio * D * py

    top_left     = (ox - half * ux - half * px, oy - half * uy - half * py)
    top_right    = (ox + half * ux - half * px, oy + half * uy - half * py)
    bottom_right = (ox + half * ux + half * px, oy + half * uy + half * py)
    bottom_left  = (ox - half * ux + half * px, oy - half * uy + half * py)

    return [top_left, top_right, bottom_right, bottom_left]


def _compute_inscribed_rect(pts_original):
    """
    Find the largest rectangle that fits entirely within the palm quad
    P1→P3→P4→P5, with no constraint on which edge it touches.

    Strategy: for a convex polygon the largest inscribed axis-aligned
    rectangle has at least one side flush with an edge.  We try every
    quad edge as the u-axis, run a full 2-D search (both v_top AND
    v_bot free) for each orientation, and return the globally largest
    result — completely unbiased toward the P1-P3 line.

    The winning rectangle is then rotated (in 90° steps) so that its
    top edge is closest to the P1-P3 finger-base midpoint, giving a
    consistent finger-bases-at-top orientation.

    Returns [top_left, top_right, bottom_right, bottom_left] in image px.
    """
    p1 = np.array(pts_original[0], dtype=float)
    p3 = np.array(pts_original[2], dtype=float)
    p4 = np.array(pts_original[3], dtype=float)
    p5 = np.array(pts_original[4], dtype=float)

    quad_img = [p1, p3, p4, p5]          # clockwise vertex order

    def _best_for_axis(u_hat):
        v_hat  = np.array([-u_hat[1], u_hat[0]])
        origin = quad_img[0]

        def to_local(p):
            d = p - origin
            return float(np.dot(d, u_hat)), float(np.dot(d, v_hat))

        local = [to_local(p) for p in quad_img]
        n     = len(local)

        def boundaries_at(v):
            xs = []
            for i in range(n):
                au, av = local[i]
                bu, bv = local[(i + 1) % n]
                lo, hi = min(av, bv), max(av, bv)
                if lo <= v <= hi:
                    if abs(bv - av) < 1e-9:
                        xs.extend([au, bu])
                    else:
                        t = (v - av) / (bv - av)
                        xs.append(au + t * (bu - au))
            if not xs:
                return None, None
            return min(xs), max(xs)

        all_v        = [q[1] for q in local]
        v_min, v_max = min(all_v), max(all_v)
        N            = 200
        vs           = [v_min + (v_max - v_min) * k / N for k in range(N + 1)]

        Lv, Rv = [], []
        for v in vs:
            l, r = boundaries_at(v)
            Lv.append(l if l is not None else  float('inf'))
            Rv.append(r if r is not None else float('-inf'))

        best_area = 0.0
        best_vt   = vs[0]; best_vb = vs[-1]
        best_ul   = 0.0;   best_ur = 0.0

        for i in range(N):
            max_l = Lv[i]
            min_r = Rv[i]
            for j in range(i + 1, N + 1):
                max_l = max(max_l, Lv[j])
                min_r = min(min_r, Rv[j])
                w = min_r - max_l
                h = vs[j] - vs[i]
                if w > 0 and w * h > best_area:
                    best_area = w * h
                    best_vt, best_vb = vs[i], vs[j]
                    best_ul, best_ur = max_l, min_r

        return best_area, (best_vt, best_vb, best_ul, best_ur), u_hat, v_hat, origin

    # ── try every quad edge as the u-axis ────────────────────────────
    n_q              = len(quad_img)
    best_area_global = 0.0
    best_result      = None

    for i in range(n_q):
        a = quad_img[i]
        b = quad_img[(i + 1) % n_q]
        edge     = b - a
        edge_len = float(np.linalg.norm(edge))
        if edge_len < 1:
            continue
        u_hat = edge / edge_len
        area, params, u_h, v_h, origin = _best_for_axis(u_hat)
        if area > best_area_global:
            best_area_global = area
            best_result      = (params, u_h, v_h, origin)

    params, u_hat, v_hat, origin = best_result
    vt, vb, ul, ur = params

    def to_img(u, v):
        pt = origin + u * u_hat + v * v_hat
        return (float(pt[0]), float(pt[1]))

    tl = to_img(ul, vt)
    tr = to_img(ur, vt)
    br = to_img(ur, vb)
    bl = to_img(ul, vb)

    # ── orient: top edge closest to P1-P3 finger-base midpoint ───────
    finger_mid   = (p1 + p3) / 2.0
    corners_list = [tl, tr, br, bl]

    def top_centre(c):
        return (np.array(c[0]) + np.array(c[1])) / 2.0

    best_rot  = 0
    best_dist = np.linalg.norm(top_centre(corners_list) - finger_mid)
    for rot in range(1, 4):
        rotated = corners_list[rot:] + corners_list[:rot]
        d = np.linalg.norm(top_centre(rotated) - finger_mid)
        if d < best_dist:
            best_dist = d
            best_rot  = rot

    corners_list = corners_list[best_rot:] + corners_list[:best_rot]
    return corners_list


def _get_perspective_coeffs(src_pts, dst_pts):
    """
    Compute the 8 PIL PERSPECTIVE transform coefficients.

    PIL maps each OUTPUT pixel (x, y) to an INPUT pixel (X, Y) via:
        X = (a*x + b*y + c) / (g*x + h*y + 1)
        Y = (d*x + e*y + f) / (g*x + h*y + 1)

    src_pts : 4 points in the *input* image  [(x0,y0), ...]
    dst_pts : 4 corresponding *output* pixel coords [(x0,y0), ...]
    Returns a tuple (a, b, c, d, e, f, g, h).
    """
    A, rhs = [], []
    for (xd, yd), (xs, ys) in zip(dst_pts, src_pts):
        A.append([xd, yd, 1,  0,  0,  0, -xd * xs, -yd * xs])
        rhs.append(xs)
        A.append([ 0,  0,  0, xd, yd,  1, -xd * ys, -yd * ys])
        rhs.append(ys)
    coeffs = np.linalg.solve(np.array(A, dtype=float),
                             np.array(rhs, dtype=float))
    return tuple(float(c) for c in coeffs)


def _get_palm_quad(pts_original, top_padding=0.0, bottom_padding=0.04):
    """
    Return the 4 palm landmark corners for perspective extraction.

    Corner mapping (output image order):
        P1 (index corner)  → top-left   (0, 0)
        P3 (little corner)  → top-right  (W, 0)
        P4 (wrist right)    → bot-right  (W, H)
        P5 (thumb)          → bot-left   (0, H)

    Asymmetric padding:
      top_padding    = 0.0  — P1/P3 stay exact (no finger-skin artifacts)
      bottom_padding = 0.04 — P4/P5 nudged outward for a wrist safety margin
    """
    p1 = pts_original[0]
    p3 = pts_original[2]
    p4 = pts_original[3]
    p5 = pts_original[4]

    cx = (p1[0] + p3[0] + p4[0] + p5[0]) / 4.0
    cy = (p1[1] + p3[1] + p4[1] + p5[1]) / 4.0

    def _pad(p, frac):
        if frac == 0.0:
            return p
        return (p[0] + frac * (p[0] - cx),
                p[1] + frac * (p[1] - cy))

    p1 = _pad(p1, top_padding)
    p3 = _pad(p3, top_padding)
    p4 = _pad(p4, bottom_padding)
    p5 = _pad(p5, bottom_padding)

    return [p1, p3, p4, p5]   # TL, TR, BR, BL


class PalmAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Palm ROI Annotator")
        self.root.configure(bg="#1e1e2e")

        self.image_paths = []
        self.current_idx = 0
        self.folder = ""
        self.annotations = {}
        self.skipped = []
        self.points_canvas = []
        self.points_original = []
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.tk_image = None

        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        toolbar = tk.Frame(self.root, bg="#313244", pady=6)
        toolbar.pack(fill="x")

        btn = {"relief": "flat", "padx": 14, "pady": 5,
               "font": ("Segoe UI", 10, "bold"), "cursor": "hand2",
               "fg": "#1e1e2e"}

        tk.Button(toolbar, text="Open Folder", bg="#89b4fa",
                  command=self._open_folder, **btn).pack(side="left", padx=6)

        self.progress_label = tk.Label(
            toolbar, text="No images loaded",
            bg="#313244", fg="#cdd6f4", font=("Segoe UI", 10))
        self.progress_label.pack(side="right", padx=12)

        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(self.root, variable=self.progress_var,
                        maximum=100).pack(fill="x")

        main = tk.Frame(self.root, bg="#1e1e2e")
        main.pack(fill="both", expand=True)

        cf = tk.Frame(main, bg="#181825")
        cf.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self.canvas = tk.Canvas(cf, bg="#181825", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

        side = tk.Frame(main, bg="#313244", width=230)
        side.pack(side="right", fill="y", padx=(0, 8), pady=8)
        side.pack_propagate(False)

        tk.Label(side, text="Keypoints  (exactly 5)",
                 bg="#313244", fg="#cdd6f4",
                 font=("Segoe UI", 11, "bold")).pack(pady=(12, 4))

        lf = tk.Frame(side, bg="#313244")
        lf.pack(fill="both", expand=True, padx=8)
        sb = tk.Scrollbar(lf)
        sb.pack(side="right", fill="y")
        self.points_listbox = tk.Listbox(
            lf, yscrollcommand=sb.set, bg="#1e1e2e", fg="#a6e3a1",
            font=("Courier New", 9), selectbackground="#89b4fa",
            borderwidth=0, highlightthickness=0)
        self.points_listbox.pack(fill="both", expand=True)
        sb.config(command=self.points_listbox.yview)

        self.count_label = tk.Label(side, text="Points: 0 / 5",
                                    bg="#313244", fg="#f38ba8",
                                    font=("Segoe UI", 10, "bold"))
        self.count_label.pack(pady=4)

        self.next_hint = tk.Label(side, text="", bg="#313244", fg="#FAC775",
                                  font=("Segoe UI", 9, "italic"),
                                  wraplength=200, justify="center")
        self.next_hint.pack(pady=(0, 6))

        b2 = {"relief": "flat", "padx": 10, "pady": 6,
              "font": ("Segoe UI", 10, "bold"), "cursor": "hand2",
              "fg": "#1e1e2e", "width": 18}

        tk.Button(side, text="Clear Points", bg="#fab387",
                  command=self._clear_points, **b2).pack(padx=8, pady=3, fill="x")
        tk.Button(side, text="Back", bg="#cba6f7",
                  command=self._go_back, **b2).pack(padx=8, pady=3, fill="x")
        tk.Button(side, text="Save & Next", bg="#a6e3a1",
                  command=self._save_and_next, **b2).pack(padx=8, pady=3, fill="x")
        tk.Button(side, text="Skip (unusable)", bg="#f38ba8",
                  command=self._skip_image, **b2).pack(padx=8, pady=(12, 3), fill="x")

        tips = ("P1 -> index corner\n"
                "P2 -> mid-ring web\n"
                "P3 -> little corner\n"
                "P4 -> wrist\n"
                "P5 -> thumb\n\n"
                "Right-click removes last.\n"
                "Yellow dashed = P1-P3 axis.\n"
                "Blue   dashed = square ROI.\n"
                "Orange dashed = rect ROI.\n"
                "Green  dashed = persp quad.")
        tk.Label(side, text=tips, bg="#313244", fg="#6c7086",
                 font=("Segoe UI", 8), justify="left").pack(pady=8, padx=8)

        self.status_var = tk.StringVar(value="Open a folder to begin.")
        tk.Label(self.root, textvariable=self.status_var,
                 bg="#181825", fg="#6c7086",
                 font=("Segoe UI", 9), anchor="w").pack(
            fill="x", side="bottom", padx=8, pady=2)

    # ── Folder / Session ───────────────────────────────────────────────────────

    def _open_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp",
                "*.tiff", "*.tif", "*.webp")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(folder, ext)))
            paths.extend(glob.glob(os.path.join(folder, ext.upper())))
        paths = sorted(set(os.path.normpath(p) for p in paths))

        if not paths:
            messagebox.showwarning("No Images",
                                   "No supported images found in that folder.")
            return

        self.folder = folder
        self.image_paths = paths
        self.current_idx = 0
        self.annotations = {}
        self.skipped = []

        os.makedirs(os.path.join(folder, "keypoints"),    exist_ok=True)
        os.makedirs(os.path.join(folder, "roi_square"),    exist_ok=True)
        os.makedirs(os.path.join(folder, "roi_rectangle"), exist_ok=True)
        os.makedirs(os.path.join(folder, "roi_perspective"), exist_ok=True)

        self._load_existing_session()
        self._advance_to_first_pending()
        self._display_image()
        self.status_var.set(f"Loaded {len(paths)} images from: {folder}")

    def _load_existing_session(self):
        kp_dir = os.path.join(self.folder, "keypoints")
        for path in self.image_paths:
            fname = os.path.basename(path)
            stem = os.path.splitext(fname)[0]
            jpath = os.path.join(kp_dir, stem + ".json")
            if os.path.exists(jpath):
                try:
                    with open(jpath) as f:
                        data = json.load(f)
                    self.annotations[fname] = data.get("keypoints", [])
                except Exception:
                    pass

        skip_path = os.path.join(self.folder, "skipped_images.json")
        if os.path.exists(skip_path):
            try:
                with open(skip_path) as f:
                    self.skipped = json.load(f)
            except Exception:
                self.skipped = []

    def _advance_to_first_pending(self):
        done = set(self.annotations.keys()) | set(self.skipped)
        for i, p in enumerate(self.image_paths):
            if os.path.basename(p) not in done:
                self.current_idx = i
                return
        self.current_idx = 0

    # ── Display ────────────────────────────────────────────────────────────────

    def _display_image(self):
        if not self.image_paths:
            return

        path = self.image_paths[self.current_idx]
        fname = os.path.basename(path)

        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        scale = min(MAX_DISPLAY_W / orig_w, MAX_DISPLAY_H / orig_h, 1.0)
        disp_w = int(orig_w * scale)
        disp_h = int(orig_h * scale)
        self.scale_x = orig_w / disp_w
        self.scale_y = orig_h / disp_h

        img_resized = img.resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img_resized)

        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        self.points_canvas.clear()
        self.points_original.clear()

        if fname in self.annotations:
            for pt in self.annotations[fname]:
                ox, oy = pt["x"], pt["y"]
                self.points_canvas.append(
                    (int(ox / self.scale_x), int(oy / self.scale_y)))
                self.points_original.append((ox, oy))

        self._redraw_points()
        self._update_listbox()
        self._update_progress()

        tag = ""
        if fname in self.skipped:
            tag = "  [SKIPPED]"
        elif fname in self.annotations:
            tag = "  [annotated]"
        self.root.title(
            f"Palm Annotator  [{self.current_idx+1}/{len(self.image_paths)}]"
            f"  {fname}{tag}")

    # ── Click Handlers ─────────────────────────────────────────────────────────

    def _on_left_click(self, event):
        if not self.image_paths:
            return
        if len(self.points_canvas) >= MAX_POINTS:
            self.status_var.set(
                "5 points placed. Right-click to remove the last one.")
            return
        cx, cy = event.x, event.y
        self.points_canvas.append((cx, cy))
        self.points_original.append(
            (int(cx * self.scale_x), int(cy * self.scale_y)))
        self._redraw_points()
        self._update_listbox()

    def _on_right_click(self, event):
        if self.points_canvas:
            self.points_canvas.pop()
            self.points_original.pop()
            self._redraw_points()
            self._update_listbox()

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _redraw_points(self):
        self.canvas.delete("annotation")
        pts = self.points_canvas
        r = POINT_RADIUS

        # Polygon edges
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                self.canvas.create_line(
                    pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1],
                    fill=LINE_COLOR, width=1.5, dash=(4, 3),
                    tags="annotation")
            if len(pts) == MAX_POINTS:
                self.canvas.create_line(
                    pts[-1][0], pts[-1][1], pts[0][0], pts[0][1],
                    fill=LINE_COLOR, width=1, dash=(2, 4),
                    tags="annotation")

        # P1-P3 axis (yellow) when both placed
        if len(pts) >= 3:
            self.canvas.create_line(
                pts[0][0], pts[0][1], pts[2][0], pts[2][1],
                fill="#FFD700", width=2, dash=(6, 3), tags="annotation")

        # Individual dots + number labels
        for i, (cx, cy) in enumerate(pts):
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=POINT_COLORS[i], outline="white", width=1.5,
                tags="annotation")
            self.canvas.create_text(
                cx + r + 4, cy - r, text=f"P{i+1}",
                fill=LABEL_COLOR, font=("Segoe UI", 8, "bold"),
                anchor="nw", tags="annotation")

        # ROI preview when all 5 placed
        if len(pts) == MAX_POINTS:
            self._draw_roi_preview()

        n = len(pts)
        if n < MAX_POINTS:
            self.next_hint.config(text=f"Place: {POINT_LABELS[n]}")
        else:
            self.next_hint.config(text="All 5 points placed.")

    def _draw_roi_preview(self):
        if len(self.points_canvas) < MAX_POINTS:
            return

        # ── square ROI (original Lin et al. method) — blue ────────────
        try:
            sq_orig = _compute_roi_corners(
                self.points_original, OFFSET_RATIO, SIDE_RATIO)
            sq = [(x / self.scale_x, y / self.scale_y) for x, y in sq_orig]
            flat_sq = [c for corner in sq for c in corner]
            self.canvas.create_polygon(
                *flat_sq, outline="#185FA5", fill="", width=2,
                dash=(8, 4), tags="annotation")
            cx_sq = sum(c[0] for c in sq) / 4
            cy_sq = sum(c[1] for c in sq) / 4
            self.canvas.create_oval(
                cx_sq - 4, cy_sq - 4, cx_sq + 4, cy_sq + 4,
                fill="#185FA5", outline="white", width=1,
                tags="annotation")
            self.canvas.create_text(
                cx_sq + 6, cy_sq, text="sq", fill="#89b4fa",
                font=("Segoe UI", 8, "bold"), anchor="w",
                tags="annotation")
        except Exception:
            pass

        # ── perspective quad (P1→P3→P4→P5) — green ──────────────────────
        try:
            persp_orig = _get_palm_quad(self.points_original)
            persp = [(x / self.scale_x, y / self.scale_y)
                     for x, y in persp_orig]
            flat_persp = [c for corner in persp for c in corner]
            self.canvas.create_polygon(
                *flat_persp, outline="#00CC55", fill="", width=2,
                dash=(8, 4), tags="annotation")
            cx_p = sum(c[0] for c in persp) / 4
            cy_p = sum(c[1] for c in persp) / 4
            self.canvas.create_text(
                cx_p + 6, cy_p, text="persp", fill="#00CC55",
                font=("Segoe UI", 8, "bold"), anchor="w",
                tags="annotation")
        except Exception:
            pass
        try:
            rect_orig = _compute_inscribed_rect(self.points_original)
            rect = [(x / self.scale_x, y / self.scale_y)
                    for x, y in rect_orig]
            flat_rect = [c for corner in rect for c in corner]
            self.canvas.create_polygon(
                *flat_rect, outline="#FF8C00", fill="", width=2,
                dash=(8, 4), tags="annotation")
            cx_r = sum(c[0] for c in rect) / 4
            cy_r = sum(c[1] for c in rect) / 4
            self.canvas.create_oval(
                cx_r - 4, cy_r - 4, cx_r + 4, cy_r + 4,
                fill="#FF8C00", outline="white", width=1,
                tags="annotation")
            self.canvas.create_text(
                cx_r + 6, cy_r, text="rect", fill="#FF8C00",
                font=("Segoe UI", 8, "bold"), anchor="w",
                tags="annotation")
        except Exception:
            pass

    # ── Listbox / Progress ─────────────────────────────────────────────────────

    def _update_listbox(self):
        self.points_listbox.delete(0, "end")
        for i, (ox, oy) in enumerate(self.points_original):
            lbl = POINT_LABELS[i][3:] if i < len(POINT_LABELS) else f"P{i+1}"
            self.points_listbox.insert(
                "end", f"P{i+1} {lbl:14s}  {ox:4d},{oy:4d}")
        self.count_label.config(
            text=f"Points: {len(self.points_original)} / {MAX_POINTS}")

    def _update_progress(self):
        total = len(self.image_paths)
        done = len(self.annotations) + len(self.skipped)
        pct = (done / total * 100) if total else 0
        self.progress_var.set(pct)
        self.progress_label.config(
            text=(f"{len(self.annotations)} annotated  "
                  f"{len(self.skipped)} skipped  /  {total}  ({pct:.0f}%)"))

    # ── Save / Navigation ──────────────────────────────────────────────────────

    def _save_and_next(self):
        if not self.image_paths:
            return
        if len(self.points_original) != MAX_POINTS:
            messagebox.showwarning(
                "Need 5 points",
                f"Please place exactly 5 keypoints before saving.\n"
                f"Currently placed: {len(self.points_original)}")
            return

        fname = os.path.basename(self.image_paths[self.current_idx])
        self._save_keypoints(fname)

        try:
            self._extract_roi(fname)
            self.status_var.set(
                f"Saved keypoints + roi_square + roi_rectangle + roi_perspective for {fname}")
        except Exception as e:
            self.status_var.set(f"Keypoints saved. ROI error: {e}")

        self._go_to_next()

    def _save_keypoints(self, fname):
        pts = [{"x": x, "y": y} for x, y in self.points_original]
        self.annotations[fname] = pts
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(self.folder, "keypoints", stem + ".json")
        payload = {
            "image": fname,
            "keypoints": pts,
            "labels": POINT_LABELS,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _extract_roi(self, fname):
        """
        Extract three ROIs and save them to separate folders:
          ./roi_square/      — Lin et al. 2024 square (right hand flipped)
          ./roi_rectangle/   — largest inscribed rectangle (no flip)
          ./roi_perspective/ — perspective warp P1→P3→P4→P5 (no flip)
        """
        path = self.image_paths[self.current_idx]
        img  = Image.open(path).convert("RGB")

        def _affine_extract(corners):
            c0, c1, _c2, c3 = corners
            ux = (c1[0] - c0[0]) / ROI_SIZE
            uy = (c1[1] - c0[1]) / ROI_SIZE
            vx = (c3[0] - c0[0]) / ROI_SIZE
            vy = (c3[1] - c0[1]) / ROI_SIZE
            coeffs = (ux, vx, c0[0], uy, vy, c0[1])
            return img.transform(
                (ROI_SIZE, ROI_SIZE),
                Image.AFFINE,
                coeffs,
                resample=Image.BICUBIC,
            )

        # ── 1. Square ROI (Lin et al. 2024) → roi_square/ ─────────────
        sq_corners = _compute_roi_corners(
            self.points_original, OFFSET_RATIO, SIDE_RATIO)
        roi_square = _affine_extract(sq_corners)
        if "right" in fname.lower():
            roi_square = roi_square.transpose(Image.FLIP_LEFT_RIGHT)
            print(f"Flipped right-hand square ROI for: {fname}")
        roi_square.save(os.path.join(self.folder, "roi_square", fname))

        # ── 2. Rectangle ROI (largest inscribed) → roi_rectangle/ ─────
        rect_corners = _compute_inscribed_rect(self.points_original)
        roi_rect     = _affine_extract(rect_corners)   # no flip
        roi_rect.save(os.path.join(self.folder, "roi_rectangle", fname))

        # ── 3. Perspective ROI (P1→P3→P4→P5 warp) → roi_perspective/ ─
        W = H = ROI_SIZE
        src_pts = _get_palm_quad(self.points_original)
        dst_pts = [(0, 0), (W, 0), (W, H), (0, H)]
        persp_coeffs = _get_perspective_coeffs(src_pts, dst_pts)
        roi_persp = img.transform(
            (ROI_SIZE, ROI_SIZE),
            Image.PERSPECTIVE,
            persp_coeffs,
            resample=Image.BICUBIC,
        )
        if "right" in fname.lower():
            roi_persp = roi_persp.transpose(Image.FLIP_LEFT_RIGHT)
            print(f"Flipped right-hand perspective ROI for: {fname}")
        roi_persp.save(os.path.join(self.folder, "roi_perspective", fname))

    def _go_to_next(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.points_canvas.clear()
            self.points_original.clear()
            self._display_image()
        else:
            messagebox.showinfo(
                "Done!",
                f"All images processed!\n\n"
                f"Annotated : {len(self.annotations)}\n"
                f"Skipped   : {len(self.skipped)}\n\n"
                f"Keypoints     -> ./keypoints/\n"
                f"Square ROIs   -> ./roi_square/\n"
                f"Rect ROIs     -> ./roi_rectangle/\n"
                f"Persp ROIs    -> ./roi_perspective/\n"
                f"Skipped       -> skipped_images.json")

    def _go_back(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.points_canvas.clear()
            self.points_original.clear()
            self._display_image()

    def _clear_points(self):
        self.points_canvas.clear()
        self.points_original.clear()
        self._redraw_points()
        self._update_listbox()

    # ── Skip ───────────────────────────────────────────────────────────────────

    def _skip_image(self):
        if not self.image_paths:
            return
        fname = os.path.basename(self.image_paths[self.current_idx])
        if fname not in self.skipped:
            self.skipped.append(fname)
        self._save_skipped_list()
        self.status_var.set(f"Skipped: {fname}")
        self._go_to_next()

    def _save_skipped_list(self):
        out_path = os.path.join(self.folder, "skipped_images.json")
        with open(out_path, "w") as f:
            json.dump(self.skipped, f, indent=2)


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.minsize(960, 640)
    PalmAnnotator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
