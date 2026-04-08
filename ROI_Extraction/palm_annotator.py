"""
Palm ROI Annotation Tool
========================
Left-click  → place a point
Right-click → remove last point
Save & Next → save coordinates and go to next image
Back        → go to previous image
Clear       → remove all points on current image
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
import csv
import os
import glob


# ── Configuration ─────────────────────────────────────────────────────────────
POINT_RADIUS   = 6       # dot size on canvas
POINT_COLOR    = "#00FF88"
LINE_COLOR     = "#00FF88"
LABEL_COLOR    = "#FFFFFF"
DRAW_POLYGON   = True    # connect points with lines
MAX_DISPLAY_W  = 900     # max canvas width
MAX_DISPLAY_H  = 700     # max canvas height
OUTPUT_FORMAT  = "json"  # "json" or "csv"
# ──────────────────────────────────────────────────────────────────────────────


class PalmAnnotator:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Palm ROI Annotator")
        self.root.configure(bg="#1e1e2e")

        # State
        self.image_paths: list[str] = []
        self.current_idx: int = 0
        self.annotations: dict[str, list[dict]] = {}  # filename → [{"x":…,"y":…}]
        self.points_canvas: list[tuple[int, int]] = []  # display coords
        self.points_original: list[tuple[int, int]] = []  # original image coords
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.tk_image = None
        self.output_path: str = ""

        self._build_ui()

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top toolbar ──
        toolbar = tk.Frame(self.root, bg="#313244", pady=6)
        toolbar.pack(fill="x")

        btn_style = {"bg": "#89b4fa", "fg": "#1e1e2e", "font": ("Segoe UI", 10, "bold"),
                     "relief": "flat", "padx": 14, "pady": 5, "cursor": "hand2"}

        tk.Button(toolbar, text="📂  Open Folder", command=self._open_folder,
                  **btn_style).pack(side="left", padx=6)

        tk.Button(toolbar, text="💾  Save Format: JSON/CSV",
                  command=self._toggle_format, **{**btn_style, "bg": "#a6e3a1"}).pack(side="left", padx=4)

        self.format_label = tk.Label(toolbar, text=f"Format: {OUTPUT_FORMAT.upper()}",
                                     bg="#313244", fg="#cdd6f4", font=("Segoe UI", 10))
        self.format_label.pack(side="left", padx=4)

        self.progress_label = tk.Label(toolbar, text="No images loaded",
                                       bg="#313244", fg="#cdd6f4", font=("Segoe UI", 10))
        self.progress_label.pack(side="right", padx=12)

        # ── Progress bar ──
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var,
                                            maximum=100, length=400)
        self.progress_bar.pack(fill="x", padx=0)

        # ── Main area: canvas + side panel ──
        main = tk.Frame(self.root, bg="#1e1e2e")
        main.pack(fill="both", expand=True)

        # Canvas
        canvas_frame = tk.Frame(main, bg="#181825")
        canvas_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.canvas = tk.Canvas(canvas_frame, bg="#181825", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

        # Side panel
        side = tk.Frame(main, bg="#313244", width=220)
        side.pack(side="right", fill="y", padx=(0, 8), pady=8)
        side.pack_propagate(False)

        tk.Label(side, text="ROI Points", bg="#313244", fg="#cdd6f4",
                 font=("Segoe UI", 12, "bold")).pack(pady=(12, 4))

        # Points list
        list_frame = tk.Frame(side, bg="#313244")
        list_frame.pack(fill="both", expand=True, padx=8)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.points_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                         bg="#1e1e2e", fg="#a6e3a1",
                                         font=("Courier New", 10),
                                         selectbackground="#89b4fa",
                                         borderwidth=0, highlightthickness=0)
        self.points_listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.points_listbox.yview)

        self.count_label = tk.Label(side, text="Points: 0", bg="#313244",
                                    fg="#f38ba8", font=("Segoe UI", 10, "bold"))
        self.count_label.pack(pady=4)

        # Action buttons
        btn2 = {"bg": "#f38ba8", "fg": "#1e1e2e", "font": ("Segoe UI", 10, "bold"),
                 "relief": "flat", "padx": 10, "pady": 5, "cursor": "hand2", "width": 18}

        tk.Button(side, text="🗑  Clear Points", command=self._clear_points,
                  **{**btn2, "bg": "#fab387"}).pack(padx=8, pady=3, fill="x")

        tk.Button(side, text="⬅  Back", command=self._go_back,
                  **{**btn2, "bg": "#cba6f7"}).pack(padx=8, pady=3, fill="x")

        tk.Button(side, text="✅  Save & Next", command=self._save_and_next,
                  **{**btn2, "bg": "#a6e3a1"}).pack(padx=8, pady=3, fill="x")

        tk.Button(side, text="💾  Export All Now", command=self._export_all,
                  **{**btn2, "bg": "#89b4fa"}).pack(padx=8, pady=(12, 3), fill="x")

        # Tip label
        tips = ("Left-click → add point\n"
                "Right-click → remove last\n"
                "Save & Next → save + advance")
        tk.Label(side, text=tips, bg="#313244", fg="#6c7086",
                 font=("Segoe UI", 9), justify="left").pack(pady=8, padx=8)

        # Status bar
        self.status_var = tk.StringVar(value="Open a folder to begin.")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              bg="#181825", fg="#6c7086",
                              font=("Segoe UI", 9), anchor="w")
        status_bar.pack(fill="x", side="bottom", padx=8, pady=2)

    # ── Folder / Image Loading ─────────────────────────────────────────────────

    def _open_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(folder, ext)))
            paths.extend(glob.glob(os.path.join(folder, ext.upper())))

        paths = sorted(set(paths))
        if not paths:
            messagebox.showwarning("No Images", "No supported images found in that folder.")
            return

        self.image_paths = paths
        self.current_idx = 0
        self.annotations = {}

        # Default output path next to images
        self.output_path = os.path.join(folder, "roi_annotations")
        self.status_var.set(f"Loaded {len(paths)} images from: {folder}")
        self._load_existing_annotations(folder)
        self._display_image()

    def _load_existing_annotations(self, folder: str):
        """Load previously saved annotations if they exist."""
        json_path = os.path.join(folder, "roi_annotations.json")
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    self.annotations = json.load(f)
                self.status_var.set(f"Resumed existing session — {len(self.annotations)} images already annotated.")
            except Exception:
                pass

    # ── Image Display ──────────────────────────────────────────────────────────

    def _display_image(self):
        if not self.image_paths:
            return

        path = self.image_paths[self.current_idx]
        fname = os.path.basename(path)

        # Load & resize to fit canvas
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

        # Restore existing points for this image
        self.points_canvas.clear()
        self.points_original.clear()

        if fname in self.annotations:
            for pt in self.annotations[fname]:
                ox, oy = pt["x"], pt["y"]
                cx = int(ox / self.scale_x)
                cy = int(oy / self.scale_y)
                self.points_canvas.append((cx, cy))
                self.points_original.append((ox, oy))

        self._redraw_points()
        self._update_listbox()
        self._update_progress()
        self.root.title(f"Palm Annotator — [{self.current_idx + 1}/{len(self.image_paths)}] {fname}")

    # ── Click Handlers ─────────────────────────────────────────────────────────

    def _on_left_click(self, event):
        if not self.image_paths:
            return
        cx, cy = event.x, event.y
        ox = int(cx * self.scale_x)
        oy = int(cy * self.scale_y)
        self.points_canvas.append((cx, cy))
        self.points_original.append((ox, oy))
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

        if DRAW_POLYGON and len(pts) > 1:
            flat = [coord for p in pts for coord in p]
            self.canvas.create_line(*flat, fill=LINE_COLOR, width=2,
                                    tags="annotation", dash=(4, 2))
            if len(pts) > 2:  # close the polygon preview
                self.canvas.create_line(pts[-1][0], pts[-1][1],
                                        pts[0][0], pts[0][1],
                                        fill=LINE_COLOR, width=1,
                                        tags="annotation", dash=(2, 4))

        for i, (cx, cy) in enumerate(pts):
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=POINT_COLOR, outline="white", width=1,
                                    tags="annotation")
            self.canvas.create_text(cx + r + 4, cy - r, text=str(i + 1),
                                    fill=LABEL_COLOR, font=("Segoe UI", 8, "bold"),
                                    anchor="nw", tags="annotation")

    # ── Listbox / Progress ─────────────────────────────────────────────────────

    def _update_listbox(self):
        self.points_listbox.delete(0, "end")
        for i, (ox, oy) in enumerate(self.points_original):
            self.points_listbox.insert("end", f"P{i+1:02d}  x={ox:4d}  y={oy:4d}")
        self.count_label.config(text=f"Points: {len(self.points_original)}")

    def _update_progress(self):
        total = len(self.image_paths)
        done = len(self.annotations)
        pct = (done / total * 100) if total else 0
        self.progress_var.set(pct)
        self.progress_label.config(text=f"{done}/{total} annotated  ({pct:.0f}%)")

    # ── Save / Navigation ──────────────────────────────────────────────────────

    def _save_current(self):
        if not self.image_paths:
            return
        fname = os.path.basename(self.image_paths[self.current_idx])
        self.annotations[fname] = [{"x": x, "y": y} for x, y in self.points_original]
        self.status_var.set(f"Saved {len(self.points_original)} points for {fname}")

    def _save_and_next(self):
        self._save_current()
        self._export_all(silent=True)
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.points_canvas.clear()
            self.points_original.clear()
            self._display_image()
        else:
            messagebox.showinfo("Done!", "All images have been annotated! 🎉\n"
                                         f"Annotations saved to:\n{self.output_path}")

    def _go_back(self):
        self._save_current()
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

    # ── Export ─────────────────────────────────────────────────────────────────

    def _export_all(self, silent=False):
        if not self.annotations:
            if not silent:
                messagebox.showwarning("Nothing to export", "No annotations yet.")
            return

        global OUTPUT_FORMAT
        if OUTPUT_FORMAT == "json":
            path = self.output_path + ".json"
            with open(path, "w") as f:
                json.dump(self.annotations, f, indent=2)
        else:
            path = self.output_path + ".csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "point_index", "x", "y"])
                for fname, pts in self.annotations.items():
                    for i, pt in enumerate(pts):
                        writer.writerow([fname, i + 1, pt["x"], pt["y"]])

        if not silent:
            messagebox.showinfo("Exported", f"Annotations saved to:\n{path}")
        self.status_var.set(f"Auto-saved → {os.path.basename(path)}")

    # ── Format Toggle ──────────────────────────────────────────────────────────

    def _toggle_format(self):
        global OUTPUT_FORMAT
        OUTPUT_FORMAT = "csv" if OUTPUT_FORMAT == "json" else "json"
        self.format_label.config(text=f"Format: {OUTPUT_FORMAT.upper()}")


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.minsize(900, 600)
    app = PalmAnnotator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
