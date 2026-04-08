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

# ── Configuration ──────────────────────────────────────────────────────────────
POINT_RADIUS  = 6
MAX_DISPLAY_W = 900
MAX_DISPLAY_H = 700
MAX_POINTS    = 5
ROI_SIZE      = 224

OFFSET_RATIO  = 0.8
SIDE_RATIO    = 1.2

POINT_COLORS = ["#00FF88", "#FFD700", "#FF6B6B", "#87CEEB", "#FF8C00"]
POINT_LABELS = [
    "P1 index base",
    "P2 mid-ring web",
    "P3 little base",
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

        tips = ("P1 -> index base\n"
                "P2 -> mid-ring web\n"
                "P3 -> little base\n"
                "P4 -> wrist\n"
                "P5 -> thumb\n\n"
                "Right-click removes last.\n"
                "Yellow dashed = P1-P3 axis.\n"
                "Blue dashed   = ROI preview.")
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
        paths = sorted(set(paths))

        if not paths:
            messagebox.showwarning("No Images",
                                   "No supported images found in that folder.")
            return

        self.folder = folder
        self.image_paths = paths
        self.current_idx = 0
        self.annotations = {}
        self.skipped = []

        os.makedirs(os.path.join(folder, "keypoints"), exist_ok=True)
        os.makedirs(os.path.join(folder, "roi"), exist_ok=True)

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
        try:
            corners_orig = _compute_roi_corners(
                self.points_original, OFFSET_RATIO, SIDE_RATIO)
            corners = [(x / self.scale_x, y / self.scale_y)
                       for x, y in corners_orig]
            flat = [c for corner in corners for c in corner]
            self.canvas.create_polygon(
                *flat, outline="#185FA5", fill="", width=2,
                dash=(8, 4), tags="annotation")
            cx_avg = sum(c[0] for c in corners) / 4
            cy_avg = sum(c[1] for c in corners) / 4
            self.canvas.create_oval(
                cx_avg - 4, cy_avg - 4, cx_avg + 4, cy_avg + 4,
                fill="#185FA5", outline="white", width=1,
                tags="annotation")
            self.canvas.create_text(
                cx_avg + 6, cy_avg, text="O'", fill="#89b4fa",
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
            self.status_var.set(f"Saved keypoints + ROI for {fname}")
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
        Extract the ROI via a single PIL affine transform.

        _compute_roi_corners() returns the 4 corners of the target square
        in original image space. We then build the inverse affine mapping:
          dst (col, row) -> src (x, y)
        so PIL can sample from the source at full quality in one pass.
        """
        path = self.image_paths[self.current_idx]
        img = Image.open(path).convert("RGB")

        corners = _compute_roi_corners(
            self.points_original, OFFSET_RATIO, SIDE_RATIO)
        c0, c1, _c2, c3 = corners   # top-left, top-right, bottom-right, bottom-left

        # Vectors in source coords per output pixel
        #   u-axis (output col direction): c0 -> c1
        #   v-axis (output row direction): c0 -> c3
        ux = (c1[0] - c0[0]) / ROI_SIZE
        uy = (c1[1] - c0[1]) / ROI_SIZE
        vx = (c3[0] - c0[0]) / ROI_SIZE
        vy = (c3[1] - c0[1]) / ROI_SIZE

        # PIL AFFINE transform: (src_x, src_y) = M * (dst_col, dst_row) + offset
        #   src_x = ux * col + vx * row + c0[0]
        #   src_y = uy * col + vy * row + c0[1]
        coeffs = (ux, vx, c0[0],
                  uy, vy, c0[1])

        roi = img.transform(
            (ROI_SIZE, ROI_SIZE),
            Image.AFFINE,
            coeffs,
            resample=Image.BICUBIC,
        )

        out_path = os.path.join(self.folder, "roi", fname)
        roi.save(out_path)

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
                f"Keypoints -> ./keypoints/\n"
                f"ROIs      -> ./roi/\n"
                f"Skipped   -> skipped_images.json")

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
