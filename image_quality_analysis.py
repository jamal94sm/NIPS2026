import os
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import pyiqa

# ==========================================
# Configuration & Paths
# ==========================================
casiams_data_root   = "/home/pai-ng/Jamal/CASIA-MS-ROI"
palm_auth_data_root = "/home/pai-ng/Jamal/xpalm"
mpd_data_root       = "/home/pai-ng/Jamal/MPDv2_mediapipe_manual_roi"
xjtu_data_root      = "/home/pai-ng/Jamal/XJTU-UP"

# ==========================================
# Metric Initialization
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading IQA models on {device}...")
niqe_metric = pyiqa.create_metric('niqe').to(device)
brisque_metric = pyiqa.create_metric('brisque').to(device)

def compute_metrics(image_path):
    """Loads an image, resizes to 112x112, converts to grayscale, and computes metrics."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None
    
    # Resize to evaluation size
    img_resized = cv2.resize(img_bgr, (112, 112))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 1. Laplacian Variance (Sharpness)
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()

    # 2. RMS Contrast (Illumination Variance)
    rms_contrast = img_gray.std()

    # 3. Pseudo-PSNR (Noise proxy)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    mse = np.mean((img_gray.astype(np.float64) - blurred.astype(np.float64)) ** 2)
    psnr_val = 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

    # 4. pyiqa metrics (Requires RGB tensor)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB) 
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        niqe_val = niqe_metric(img_tensor).item()
        brisque_val = brisque_metric(img_tensor).item()

    return {
        "NIQE": niqe_val,
        "BRISQUE": brisque_val,
        "Laplacian_Var": laplacian_var,
        "RMS_Contrast": rms_contrast,
        "Pseudo_PSNR": psnr_val
    }

# ==========================================
# Dataset Parsing Functions (All Images)
# ==========================================

def parse_casia_ms(data_root):
    records = []
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if not os.path.exists(data_root): return records
    
    for fname in sorted(os.listdir(data_root)):
        if os.path.splitext(fname)[1].lower() not in img_exts: continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) < 4: continue
        
        identity = f"{parts[0]}_{parts[1]}"
        path = os.path.join(data_root, fname)
        records.append({"Dataset": "CASIA-MS", "Subset": "All", "ID": identity, "Path": path})
    return records

def parse_mpd_data(data_root):
    records = []
    if not os.path.exists(data_root): return records
    
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")): continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) != 5: continue
        
        subject, session, device_id, hand_side, iteration = parts
        if device_id not in ("h","m") or hand_side not in ("l","r"): continue
        
        identity = subject + "_" + hand_side
        path = os.path.join(data_root, fname)
        records.append({"Dataset": "MPDv2", "Subset": "All", "ID": identity, "Path": path})
    return records

def parse_xjtu_domains(data_root):
    records = []
    if not os.path.exists(data_root): return records
    IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}
    
    # Traverse Phone/Condition/Identity structure
    for device in os.listdir(data_root):
        dev_dir = os.path.join(data_root, device)
        if not os.path.isdir(dev_dir): continue
        
        for condition in os.listdir(dev_dir):
            cond_dir = os.path.join(dev_dir, condition)
            if not os.path.isdir(cond_dir): continue
            
            for id_folder in sorted(os.listdir(cond_dir)):
                id_dir = os.path.join(cond_dir, id_folder)
                if not os.path.isdir(id_dir): continue
                
                parts = id_folder.split("_")
                if len(parts) < 2 or parts[0].upper() not in ("L", "R"): continue
                
                for fname in sorted(os.listdir(id_dir)):
                    if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                    path = os.path.join(id_dir, fname)
                    records.append({"Dataset": "XJTU-UP", "Subset": "All", "ID": id_folder, "Path": path})
    return records

def parse_xpalm(data_root):
    records = []
    if not os.path.exists(data_root): return records
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Scanner
    scanner_dir = os.path.join(data_root, "scanner_roi")
    if os.path.isdir(scanner_dir):
        for subj_folder in sorted(os.listdir(scanner_dir)):
            subj_dir = os.path.join(scanner_dir, subj_folder)
            if not os.path.isdir(subj_dir): continue
            
            for fname in sorted(os.listdir(subj_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                parts = os.path.splitext(fname)[0].split("_")
                if len(parts) < 4: continue
                
                # Use raw subject ID (e.g., '001') for cross-domain matching table
                subj_id = subj_folder 
                path = os.path.join(subj_dir, fname)
                records.append({"Dataset": "X-Palm (scanner)", "Subset": "scanner", "ID": subj_id, "Path": path})

    # Smartphone
    phone_dir = os.path.join(data_root, "smartphone_roi")
    if os.path.isdir(phone_dir):
        for subj_folder in sorted(os.listdir(phone_dir)):
            subj_dir = os.path.join(phone_dir, subj_folder)
            if not os.path.isdir(subj_dir): continue
            
            for fname in sorted(os.listdir(subj_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                
                subj_id = subj_folder
                path = os.path.join(subj_dir, fname)
                records.append({"Dataset": "X-Palm (smartphone)", "Subset": "smartphone", "ID": subj_id, "Path": path})

    return records

# ==========================================
# Main Execution & Table Generation
# ==========================================
def main():
    # 1. Gather all file paths
    print("Parsing directories...")
    all_records = []
    all_records.extend(parse_casia_ms(casiams_data_root))
    all_records.extend(parse_mpd_data(mpd_data_root))
    all_records.extend(parse_xjtu_domains(xjtu_data_root))
    all_records.extend(parse_xpalm(palm_auth_data_root))

    if not all_records:
        print("No images found. Please check dataset root paths.")
        return

    # 2. Compute metrics
    print(f"Computing metrics for {len(all_records)} images. This will take time...")
    results = []
    for i, record in enumerate(all_records):
        if i > 0 and i % 1000 == 0:
            print(f"  Processed {i}/{len(all_records)} images...")
            
        metrics = compute_metrics(record["Path"])
        if metrics:
            record.update(metrics)
            results.append(record)

    df = pd.DataFrame(results)

    # Helper function for mean +- std
    def format_mean_std(x):
        return f"{x.mean():.2f} ± {x.std():.2f}"

    metrics_cols = ["NIQE", "BRISQUE", "Laplacian_Var", "RMS_Contrast", "Pseudo_PSNR"]

    # ---------------------------------------------------------
    # TABLE 1: Global Dataset Comparison
    # ---------------------------------------------------------
    table1 = df.groupby('Dataset')[metrics_cols].agg(format_mean_std).T
    
    # Enforce requested column order
    cols_order = ["XJTU-UP", "MPDv2", "CASIA-MS", "X-Palm (scanner)", "X-Palm (smartphone)"]
    valid_cols = [c for c in cols_order if c in table1.columns]
    table1 = table1[valid_cols]

    print("\n" + "="*80)
    print("TABLE 1: Image Quality Metrics by Dataset (Computed on 112x112 ROIs)")
    print("="*80)
    print(table1.to_markdown())

    # ---------------------------------------------------------
    # TABLE 2: X-Palm Subject-Level Breakdown
    # ---------------------------------------------------------
    xpalm_df = df[df['Dataset'].str.contains("X-Palm")]
    
    if not xpalm_df.empty:
        table2 = xpalm_df.groupby(['ID', 'Subset'])[metrics_cols].agg(format_mean_std).unstack(level='Subset')
        
        # Flatten multi-index columns: e.g., ('NIQE', 'scanner') -> 'NIQE_scanner'
        table2.columns = [f"{metric}_{subset}" for metric, subset in table2.columns]
        
        # Reorder columns to pair metrics (e.g., NIQE_scanner, NIQE_smartphone, ...)
        paired_cols = []
        for m in metrics_cols:
            if f"{m}_scanner" in table2.columns:
                paired_cols.append(f"{m}_scanner")
            if f"{m}_smartphone" in table2.columns:
                paired_cols.append(f"{m}_smartphone")
                
        table2 = table2[paired_cols]

        print("\n" + "="*100)
        print("TABLE 2: X-Palm Dataset Quality by Subject ID (Scanner vs Smartphone)")
        print("="*100)
        print(table2.to_markdown())
    else:
        print("\nNo X-Palm data found to generate Table 2.")

if __name__ == "__main__":
    main()