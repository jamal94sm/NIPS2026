import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import sqrtm
from tqdm import tqdm
from itertools import combinations
import warnings

# Suppress sklearn/scipy convergence warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# Configuration & Paths
# ==========================================
casiams_data_root   = "/home/pai-ng/Jamal/CASIA-MS-ROI"
palm_auth_data_root = "/home/pai-ng/Jamal/xpalm"
mpd_data_root       = "/home/pai-ng/Jamal/MPDv2_mediapipe_manual_roi"
xjtu_data_root      = "/home/pai-ng/Jamal/XJTU-UP"

# ==========================================
# Distance Metrics Formulation
# ==========================================
def compute_proxy_a_distance(X_A, X_B):
    if len(X_A) < 10 or len(X_B) < 10: return np.nan
    X = np.vstack((X_A, X_B))
    y = np.hstack((np.zeros(len(X_A)), np.ones(len(X_B))))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LinearSVC(random_state=42, max_iter=1000, dual=False)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    error = 1.0 - accuracy_score(y_test, y_pred)
    return max(0.0, 2 * (1 - 2 * error))

def compute_mmd(X_A, X_B, gamma=None):
    if len(X_A) < 2 or len(X_B) < 2: return np.nan
    XX = rbf_kernel(X_A, X_A, gamma)
    YY = rbf_kernel(X_B, X_B, gamma)
    XY = rbf_kernel(X_A, X_B, gamma)
    return max(0.0, XX.mean() + YY.mean() - 2 * XY.mean())

def compute_ffd(X_A, X_B):
    if len(X_A) < 2 or len(X_B) < 2: return np.nan
    eps = 1e-6 
    mu_A, sigma_A = np.mean(X_A, axis=0), np.cov(X_A, rowvar=False) + np.eye(X_A.shape[1]) * eps
    mu_B, sigma_B = np.mean(X_B, axis=0), np.cov(X_B, rowvar=False) + np.eye(X_B.shape[1]) * eps
    diff = mu_A - mu_B
    covmean, _ = sqrtm(sigma_A.dot(sigma_B), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return max(0.0, diff.dot(diff) + np.trace(sigma_A + sigma_B - 2 * covmean))

# ==========================================
# Dataset Parsing logic
# ==========================================
def parse_casia_ms(data_root):
    records = []
    if not os.path.exists(data_root): return records
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg", ".png", ".bmp")): continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) < 3: continue
        records.append({"Dataset": "CASIA-MS", "SubDomain": f"Spectrum_{parts[2]}", "Path": os.path.join(data_root, fname)})
    return records

def parse_mpd_data(data_root):
    records = []
    if not os.path.exists(data_root): return records
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg", ".bmp", ".png")): continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) != 5: continue
        device_id = parts[2]
        if device_id not in ("h", "m"): continue
        records.append({"Dataset": "MPDv2", "SubDomain": f"Device_{device_id}", "Path": os.path.join(data_root, fname)})
    return records

def parse_xjtu_domains(data_root):
    records = []
    if not os.path.exists(data_root): return records
    IMG_EXTS = {".jpg", ".bmp", ".png"}
    for device in os.listdir(data_root):
        dev_dir = os.path.join(data_root, device)
        if not os.path.isdir(dev_dir): continue
        for condition in os.listdir(dev_dir):
            cond_dir = os.path.join(dev_dir, condition)
            if not os.path.isdir(cond_dir): continue
            subdomain = f"{device}_{condition}"
            for id_folder in sorted(os.listdir(cond_dir)):
                id_dir = os.path.join(cond_dir, id_folder)
                if not os.path.isdir(id_dir): continue
                for fname in sorted(os.listdir(id_dir)):
                    if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                    records.append({"Dataset": "XJTU-UP", "SubDomain": subdomain, "Path": os.path.join(id_dir, fname)})
    return records

def parse_xpalm(data_root):
    records = []
    if not os.path.exists(data_root): return records
    IMG_EXTS = {".jpg", ".png", ".bmp"}
    
    # Explicit domain targets
    scanner_targets = ["pink", "green", "white", "ir", "blue", "yellow"]
    smartphone_targets = ["wet", "text", "jf", "sf", "bf", "close", "far", "pitch", "roll", "fl", "rnd"]
    
    # Process Scanner
    scanner_dir = os.path.join(data_root, "scanner_roi")
    if os.path.isdir(scanner_dir):
        for subj in sorted(os.listdir(scanner_dir)):
            subj_dir = os.path.join(scanner_dir, subj)
            if not os.path.isdir(subj_dir): continue
            for fname in sorted(os.listdir(subj_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                # Identify which scanner target matches the filename
                fname_lower = fname.lower()
                matched_target = next((t for t in scanner_targets if t in fname_lower), None)
                if matched_target:
                    records.append({"Dataset": "X-Palm", "SubDomain": f"Scanner_{matched_target}", "Path": os.path.join(subj_dir, fname)})

    # Process Smartphone
    phone_dir = os.path.join(data_root, "smartphone_roi")
    if os.path.isdir(phone_dir):
        for subj in sorted(os.listdir(phone_dir)):
            subj_dir = os.path.join(phone_dir, subj)
            if not os.path.isdir(subj_dir): continue
            for fname in sorted(os.listdir(subj_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                # Identify which smartphone target matches the filename
                fname_lower = fname.lower()
                matched_target = next((t for t in smartphone_targets if t in fname_lower), None)
                if matched_target:
                    records.append({"Dataset": "X-Palm", "SubDomain": f"Smartphone_{matched_target}", "Path": os.path.join(subj_dir, fname)})
                    
    return records

# ==========================================
# Main Execution Protocol
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ResNet-50 feature extractor on {device}...")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).to(device)
    model.fc = torch.nn.Identity()
    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(232),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Parsing dataset structures...")
    all_records = []
    all_records.extend(parse_casia_ms(casiams_data_root))
    all_records.extend(parse_mpd_data(mpd_data_root))
    all_records.extend(parse_xjtu_domains(xjtu_data_root))
    all_records.extend(parse_xpalm(palm_auth_data_root))

    if not all_records:
        print("No images found. Please verify root paths.")
        return

    print(f"Extracting 2048-D features for {len(all_records)} total images...")
    features_dict = {"CASIA-MS": {}, "MPDv2": {}, "XJTU-UP": {}, "X-Palm": {}}

    for record in tqdm(all_records, desc="Feature Extraction"):
        ds, sub, path = record["Dataset"], record["SubDomain"], record["Path"]
        if sub not in features_dict[ds]: features_dict[ds][sub] = []
        
        img = cv2.imread(path)
        if img is None: continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = model(tensor).cpu().numpy().flatten()
            features_dict[ds][sub].append(feat)

    print("\nComputing domain shift metrics (PAD, MMD, FFD)...")
    results = []

    for dataset_name, subdomains in features_dict.items():
        keys = list(subdomains.keys())
        subdomain_count = len(keys)
        if subdomain_count < 2: continue
            
        pairs = list(combinations(keys, 2))
        
        # Standard tracking for non X-Palm datasets
        mmd_scores, pad_scores, ffd_scores = [], [], []
        
        # Granular tracking specifically for X-Palm
        xp_svphone = {"mmd": [], "pad": [], "ffd": []}
        xp_svs = {"mmd": [], "pad": [], "ffd": []}
        xp_pvp = {"mmd": [], "pad": [], "ffd": []}

        # Progress bar for metric calculations
        for dom_A, dom_B in tqdm(pairs, desc=f"Calculating metrics for {dataset_name}", leave=False):
            X_A, X_B = np.array(subdomains[dom_A]), np.array(subdomains[dom_B])
            if X_A.shape[0] == 0 or X_B.shape[0] == 0: continue
            
            m_val = compute_mmd(X_A, X_B)
            p_val = compute_proxy_a_distance(X_A, X_B)
            f_val = compute_ffd(X_A, X_B)

            # Store standard values
            if not np.isnan(m_val): mmd_scores.append(m_val)
            if not np.isnan(p_val): pad_scores.append(p_val)
            if not np.isnan(f_val): ffd_scores.append(f_val)

            # Route X-Palm specific variations
            if dataset_name == "X-Palm":
                is_A_scan = "Scanner" in dom_A
                is_B_scan = "Scanner" in dom_B
                is_A_phone = "Smartphone" in dom_A
                is_B_phone = "Smartphone" in dom_B

                if (is_A_scan and is_B_phone) or (is_A_phone and is_B_scan):
                    target_dict = xp_svphone
                elif is_A_scan and is_B_scan:
                    target_dict = xp_svs
                elif is_A_phone and is_B_phone:
                    target_dict = xp_pvp
                else:
                    continue

                if not np.isnan(m_val): target_dict["mmd"].append(m_val)
                if not np.isnan(p_val): target_dict["pad"].append(p_val)
                if not np.isnan(f_val): target_dict["ffd"].append(f_val)

        # Helper to format metrics safely
        def format_metric(arr):
            return f"{np.mean(arr):.3f} ± {np.std(arr):.3f}" if len(arr) > 0 else "N/A"

        if dataset_name != "X-Palm":
            results.append({
                "Dataset": dataset_name,
                "Comparison Type": "All Sub-Domains",
                "Pairs Evaluated": len(mmd_scores),
                "Mean MMD (↑)": format_metric(mmd_scores),
                "Mean Proxy A-Dist (↑)": format_metric(pad_scores),
                "Mean Fréchet Dist (↑)": f"{np.mean(ffd_scores):.1f} ± {np.std(ffd_scores):.1f}" if ffd_scores else "N/A"
            })
        else:
            # 1. Scanner vs Smartphone
            results.append({
                "Dataset": "X-Palm",
                "Comparison Type": "Scanner vs. Smartphone",
                "Pairs Evaluated": len(xp_svphone['mmd']),
                "Mean MMD (↑)": format_metric(xp_svphone["mmd"]),
                "Mean Proxy A-Dist (↑)": format_metric(xp_svphone["pad"]),
                "Mean Fréchet Dist (↑)": f"{np.mean(xp_svphone['ffd']):.1f} ± {np.std(xp_svphone['ffd']):.1f}" if xp_svphone["ffd"] else "N/A"
            })
            
            # 2. Scanner vs Scanner
            results.append({
                "Dataset": "X-Palm",
                "Comparison Type": "Scanner vs. Scanner (Spectrums)",
                "Pairs Evaluated": len(xp_svs['mmd']),
                "Mean MMD (↑)": format_metric(xp_svs["mmd"]),
                "Mean Proxy A-Dist (↑)": format_metric(xp_svs["pad"]),
                "Mean Fréchet Dist (↑)": f"{np.mean(xp_svs['ffd']):.1f} ± {np.std(xp_svs['ffd']):.1f}" if xp_svs["ffd"] else "N/A"
            })
            
            # 3. Smartphone vs Smartphone
            results.append({
                "Dataset": "X-Palm",
                "Comparison Type": "Smartphone vs. Smartphone (All Vars)",
                "Pairs Evaluated": len(xp_pvp['mmd']),
                "Mean MMD (↑)": format_metric(xp_pvp["mmd"]),
                "Mean Proxy A-Dist (↑)": format_metric(xp_pvp["pad"]),
                "Mean Fréchet Dist (↑)": f"{np.mean(xp_pvp['ffd']):.1f} ± {np.std(xp_pvp['ffd']):.1f}" if xp_pvp["ffd"] else "N/A"
            })
            
            # 4. Smartphone vs Smartphone (Top 5 Shifts)
            # Sort arrays descending independently to find the highest shift values recorded
            top5_mmd = sorted(xp_pvp["mmd"], reverse=True)[:5]
            top5_pad = sorted(xp_pvp["pad"], reverse=True)[:5]
            top5_ffd = sorted(xp_pvp["ffd"], reverse=True)[:5]
            
            results.append({
                "Dataset": "X-Palm",
                "Comparison Type": "Smartphone vs. Smartphone (Top 5 Shifts)",
                "Pairs Evaluated": len(top5_mmd),
                "Mean MMD (↑)": format_metric(top5_mmd),
                "Mean Proxy A-Dist (↑)": format_metric(top5_pad),
                "Mean Fréchet Dist (↑)": f"{np.mean(top5_ffd):.1f} ± {np.std(top5_ffd):.1f}" if top5_ffd else "N/A"
            })

    # Output Final Table
    df = pd.DataFrame(results)
    
    print("\n" + "="*120)
    print("TABLE: Inter-Domain Distribution Shift by Dataset")
    print("="*120)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
