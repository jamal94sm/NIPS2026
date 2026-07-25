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
    """Computes PAD by training a linear SVM to distinguish domains."""
    if len(X_A) < 10 or len(X_B) < 10: return np.nan
    
    X = np.vstack((X_A, X_B))
    y = np.hstack((np.zeros(len(X_A)), np.ones(len(X_B))))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LinearSVC(random_state=42, max_iter=1000, dual=False)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    error = 1.0 - accuracy_score(y_test, y_pred)
    pad = 2 * (1 - 2 * error)
    return max(0.0, pad)

def compute_mmd(X_A, X_B, gamma=None):
    """Computes empirical Maximum Mean Discrepancy using an RBF kernel."""
    if len(X_A) < 2 or len(X_B) < 2: return np.nan
    
    XX = rbf_kernel(X_A, X_A, gamma)
    YY = rbf_kernel(X_B, X_B, gamma)
    XY = rbf_kernel(X_A, X_B, gamma)
    
    return max(0.0, XX.mean() + YY.mean() - 2 * XY.mean())

def compute_ffd(X_A, X_B):
    """Computes Fréchet Feature Distance between two domain distributions."""
    if len(X_A) < 2 or len(X_B) < 2: return np.nan
    
    eps = 1e-6 # Add tiny noise to diagonal to prevent singular matrix errors
    mu_A, sigma_A = np.mean(X_A, axis=0), np.cov(X_A, rowvar=False) + np.eye(X_A.shape[1]) * eps
    mu_B, sigma_B = np.mean(X_B, axis=0), np.cov(X_B, rowvar=False) + np.eye(X_B.shape[1]) * eps
    diff = mu_A - mu_B
    
    covmean, _ = sqrtm(sigma_A.dot(sigma_B), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    ffd = diff.dot(diff) + np.trace(sigma_A + sigma_B - 2 * covmean)
    return max(0.0, ffd)

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
        # Assume parts[2] represents the spectral channel (e.g. 460nm, 850nm)
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
    # Navigating: Device -> Condition -> Identity (Yields 4 SubDomains: 2 devices * 2 lightings)
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
    
    scanner_dir = os.path.join(data_root, "scanner_roi")
    if os.path.isdir(scanner_dir):
        for subj in sorted(os.listdir(scanner_dir)):
            subj_dir = os.path.join(scanner_dir, subj)
            if not os.path.isdir(subj_dir): continue
            for fname in sorted(os.listdir(subj_dir)):
                if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                    records.append({"Dataset": "X-Palm", "SubDomain": "Scanner", "Path": os.path.join(subj_dir, fname)})

    phone_dir = os.path.join(data_root, "smartphone_roi")
    if os.path.isdir(phone_dir):
        for subj in sorted(os.listdir(phone_dir)):
            subj_dir = os.path.join(phone_dir, subj)
            if not os.path.isdir(subj_dir): continue
            for fname in sorted(os.listdir(subj_dir)):
                if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                    # Depending on X-Palm internal file naming, we could expand subdomains here
                    records.append({"Dataset": "X-Palm", "SubDomain": "Smartphone", "Path": os.path.join(subj_dir, fname)})
    return records

# ==========================================
# Main Execution Protocol
# ==========================================
def main():
    # 1. Initialize Feature Extractor (Neutral ResNet-50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ResNet-50 feature extractor on {device}...")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).to(device)
    model.fc = torch.nn.Identity() # Remove classification head to get raw 2048-D features
    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(232),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Parse Datasets
    print("Parsing dataset structures...")
    all_records = []
    all_records.extend(parse_casia_ms(casiams_data_root))
    all_records.extend(parse_mpd_data(mpd_data_root))
    all_records.extend(parse_xjtu_domains(xjtu_data_root))
    all_records.extend(parse_xpalm(palm_auth_data_root))

    if not all_records:
        print("No images found. Please verify root paths.")
        return

    # 3. Extract Features
    print(f"Extracting 2048-D features for {len(all_records)} total images...")
    features_dict = {
        "CASIA-MS": {},
        "MPDv2": {},
        "XJTU-UP": {},
        "X-Palm": {}
    }

    for record in tqdm(all_records, desc="Feature Extraction"):
        ds = record["Dataset"]
        sub = record["SubDomain"]
        path = record["Path"]
        
        if sub not in features_dict[ds]:
            features_dict[ds][sub] = []
            
        img = cv2.imread(path)
        if img is None: continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = model(tensor).cpu().numpy().flatten()
            features_dict[ds][sub].append(feat)

    # 4. Compute Pairwise Sub-Domain Distances
    print("\nComputing domain shift metrics (PAD, MMD, FFD)...")
    results = []

    for dataset_name, subdomains in features_dict.items():
        keys = list(subdomains.keys())
        subdomain_count = len(keys)
        
        if subdomain_count < 2:
            continue # Needs at least 2 domains to compute shift
            
        mmd_scores, pad_scores, ffd_scores = [], [], []
        
        # Calculate for all unique pairs within the dataset
        for dom_A, dom_B in combinations(keys, 2):
            X_A = np.array(subdomains[dom_A])
            X_B = np.array(subdomains[dom_B])
            
            # Skip if arrays are empty
            if X_A.shape[0] == 0 or X_B.shape[0] == 0: continue
            
            mmd_scores.append(compute_mmd(X_A, X_B))
            pad_scores.append(compute_proxy_a_distance(X_A, X_B))
            ffd_scores.append(compute_ffd(X_A, X_B))
            
        # Aggregate stats
        mmd_scores = [s for s in mmd_scores if not np.isnan(s)]
        pad_scores = [s for s in pad_scores if not np.isnan(s)]
        ffd_scores = [s for s in ffd_scores if not np.isnan(s)]
        
        results.append({
            "Dataset": dataset_name,
            "Sub-Domain Count": subdomain_count,
            "Mean Pairwise MMD (↑)": f"{np.mean(mmd_scores):.3f} ± {np.std(mmd_scores):.3f}" if mmd_scores else "N/A",
            "Mean Proxy A-Dist (↑)": f"{np.mean(pad_scores):.2f} ± {np.std(pad_scores):.2f}" if pad_scores else "N/A",
            "Mean Fréchet Dist (↑)": f"{np.mean(ffd_scores):.1f} ± {np.std(ffd_scores):.1f}" if ffd_scores else "N/A"
        })

    # 5. Output Final Table
    df = pd.DataFrame(results).set_index("Dataset")
    
    # Order rows cleanly
    order = ["XJTU-UP", "MPDv2", "CASIA-MS", "X-Palm"]
    df = df.reindex([d for d in order if d in df.index])

    print("\n" + "="*90)
    print("TABLE: Inter-Domain Distribution Shift by Dataset")
    print("="*90)
    print(df.to_markdown())

if __name__ == "__main__":
    main()