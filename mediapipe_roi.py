"""
MPDv2 Palm ROI Extraction
=========================
Filename convention: {id}_{session}_{device}_{hand}_{iteration}.jpg
  id        : 001 … 200
  session   : 1 | 2
  device    : h | m
  hand      : l | r
  iteration : 01 … 10

Walks SRC_ROOT recursively, extracts palm ROI via MediaPipe,
resizes to 160×160, and saves to DST_ROOT preserving the
original relative directory structure and filename.

Fallback: images where MediaPipe fails → original image resized to 160×160.
A CSV summary is written to DST_ROOT/extraction_report.csv.
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import csv
from PIL import Image
from tqdm import tqdm


# ============================================================
# PATHS  — edit these two lines
# ============================================================
SRC_ROOT = "/mnt/data/FingerprintDatasets/Combined/combineddataset/MPDv2"
DST_ROOT = "/mnt/data/FingerprintDatasets/Combined/combineddataset/MPDv2-ROI"
ROI_SIZE = 160          # output square size in pixels


# ============================================================
# Internal helpers (unchanged from original)
# ============================================================

def _run_mp_hands(image, min_det_conf=0.2):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_det_conf,
    ) as hands:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None, None

        lm = res.multi_hand_landmarks[0]
        handedness = (
            res.multi_handedness[0].classification[0].label
            if res.multi_handedness else "Unknown"
        )

        h, w = image.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
        return pts, handedness


def _midpoints(pairs):
    return [((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2) for p1, p2 in pairs]


def _calculate_point_c(m1, m2, thumb):
    m1, m2, thumb = map(np.asarray, (m1, m2, thumb))
    O   = (m1 + m2) / 2.0
    AB  = m2 - m1
    L   = np.linalg.norm(AB)
    if L == 0:
        raise ValueError("Midpoints coincide")
    ABu  = AB / L
    perp = np.array([-ABu[1], ABu[0]])
    cross_z = ABu[0] * (thumb - O)[1] - ABu[1] * (thumb - O)[0]
    if cross_z < 0:
        perp = -perp
    C = O + 1.8 * L * perp
    return int(C[0]), int(C[1])


def _extract_roi(img, mid1, mid2, C, thumb, hand_type):
    vec   = np.array(mid2) - np.array(mid1)
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    C     = (int(C[0]), int(C[1]))

    if hand_type.lower() == "right":
        if np.dot(vec, np.array(thumb) - np.array(C)) > 0:
            angle += 180
    else:
        if np.dot(vec, np.array(thumb) - np.array(C)) < 0:
            angle += 180

    side = np.linalg.norm(vec) * 2.5
    rect = (C, (side, side), angle)

    M   = cv2.getRotationMatrix2D(C, angle, 1.0)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    roi = cv2.getRectSubPix(rot, (int(side), int(side)), C)

    box = cv2.boxPoints(rect).astype(int)
    return roi, box


def extract_palm_roi(image_bgr):
    """
    Returns (roi_bgr, annotated_bgr, hand_type)
    or      (None, None, None) if landmarks not found.
    """
    lms, hand_type = _run_mp_hands(image_bgr)
    if lms is None:
        return None, None, None

    idx = lambda i: lms[i]

    mids4 = _midpoints([
        (idx(17), idx(18)),
        (idx(14), idx(13)),
        (idx(10), idx(9)),
        (idx(6),  idx(5)),
    ])
    adj = _midpoints([
        (mids4[0], mids4[1]),
        (mids4[1], mids4[2]),
        (mids4[2], mids4[3]),
    ])

    roi_mid1 = ((adj[0][0] + adj[1][0]) / 2, (adj[0][1] + adj[1][1]) / 2)
    roi_mid2 = ((adj[1][0] + adj[2][0]) / 2, (adj[1][1] + adj[2][1]) / 2)

    thumb = idx(2)
    C     = _calculate_point_c(roi_mid1, roi_mid2, thumb)
    roi, box = _extract_roi(image_bgr, roi_mid1, roi_mid2, C, thumb, hand_type)

    ann = image_bgr.copy()
    for x, y in lms:
        cv2.circle(ann, (x, y), 3, (0, 255, 0), -1)
    cv2.polylines(ann, [box], True, (0, 255, 0), 2)
    cv2.circle(ann, C, 6, (0, 0, 255), -1)

    return roi, ann, hand_type


# ============================================================
# MPDv2 filename parser
# ============================================================

def parse_mpd_filename(fname):
    """
    Parse '{id}_{session}_{device}_{hand}_{iteration}.jpg'
    Returns a dict with keys: id, session, device, hand, iteration
    or None if the name does not match the convention.
    """
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")
    if len(parts) != 5:
        return None
    subj_id, session, device, hand, iteration = parts

    # Basic validation
    if (len(subj_id) == 3 and subj_id.isdigit()
            and session in ("1", "2")
            and device in ("h", "m")
            and hand in ("l", "r")
            and len(iteration) == 2 and iteration.isdigit()):
        return dict(id=subj_id, session=session, device=device,
                    hand=hand, iteration=iteration)
    return None


# ============================================================
# Main processing loop
# ============================================================

def main():
    os.makedirs(DST_ROOT, exist_ok=True)

    # ── Collect all matching images ───────────────────────────────────────
    all_images = []
    skipped_names = []
    for root, _, files in os.walk(SRC_ROOT):
        for f in sorted(files):
            if not f.lower().endswith(".jpg"):
                continue
            parsed = parse_mpd_filename(f)
            if parsed is None:
                skipped_names.append(os.path.join(root, f))
                continue
            all_images.append((os.path.join(root, f), parsed))

    print(f"Total valid MPDv2 images : {len(all_images)}")
    if skipped_names:
        print(f"Skipped (bad filename)   : {len(skipped_names)}")
        for p in skipped_names[:5]:
            print(f"  {p}")

    # ── Statistics ────────────────────────────────────────────────────────
    num_success  = 0
    num_fallback = 0
    report_rows  = []   # for CSV

    # ── Process ───────────────────────────────────────────────────────────
    for src_path, meta in tqdm(all_images, desc="Extracting ROIs"):
        rel_path = os.path.relpath(src_path, SRC_ROOT)
        dst_path = os.path.join(DST_ROOT, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Load
        try:
            img_rgb = np.array(Image.open(src_path).convert("RGB"))
        except Exception as e:
            tqdm.write(f"[LOAD ERROR] {src_path}: {e}")
            num_fallback += 1
            report_rows.append({**meta, 'path': rel_path, 'status': 'load_error'})
            continue

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Extract
        roi_bgr, _, detected_hand = extract_palm_roi(img_bgr)

        if roi_bgr is None or roi_bgr.size == 0:
            # Fallback: use original image
            roi_bgr = img_bgr
            status  = "fallback"
            num_fallback += 1
        else:
            # Optional sanity-check: MediaPipe hand side vs filename hand
            if detected_hand is not None:
                mp_hand  = detected_hand.lower()[0]   # 'l' or 'r'
                # MediaPipe labels from camera perspective (mirrored for selfie)
                # so we just log a warning rather than discard
                if mp_hand != meta['hand']:
                    tqdm.write(
                        f"[HAND MISMATCH] {rel_path}: "
                        f"filename={meta['hand']} mp={mp_hand}")
            status = "ok"
            num_success += 1

        roi_bgr = cv2.resize(roi_bgr, (ROI_SIZE, ROI_SIZE))
        cv2.imwrite(dst_path, roi_bgr)
        report_rows.append({**meta, 'path': rel_path, 'status': status})

    # ── Write CSV report ──────────────────────────────────────────────────
    report_path = os.path.join(DST_ROOT, "extraction_report.csv")
    with open(report_path, "w", newline="") as f:
        fieldnames = ["id", "session", "device", "hand", "iteration",
                      "path", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    # ── Summary ───────────────────────────────────────────────────────────
    total = num_success + num_fallback
    print(f"\n{'='*50}")
    print(f"  Done.  Total processed : {total}")
    print(f"  ROI extracted (ok)     : {num_success}  "
          f"({100*num_success/max(total,1):.1f}%)")
    print(f"  Fallback (original)    : {num_fallback}  "
          f"({100*num_fallback/max(total,1):.1f}%)")
    print(f"  Output directory       : {DST_ROOT}")
    print(f"  Report CSV             : {report_path}")
    print(f"{'='*50}\n")

    # Per-subset breakdown
    from collections import Counter
    breakdown = Counter(
        (r['session'], r['device'], r['hand'])
        for r in report_rows if r['status'] == 'ok'
    )
    print("  Successful extractions by (session, device, hand):")
    for key in sorted(breakdown):
        print(f"    session={key[0]} device={key[1]} hand={key[2]} : "
              f"{breakdown[key]}")
    print()


if __name__ == "__main__":
    main()
