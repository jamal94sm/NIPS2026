"""
config.py
=========
One place to set every parameter for every baseline. Defaults reproduce
each method's original published hyperparameters exactly, EXCEPT
`augment_factor`, which has been re-balanced across all 10 methods per
the augmentation-fairness analysis (see README.md):

  - Methods whose loss needs >1 view per image (SupCon pairs, triplets)
    get augment_factor=1: their native view count IS their augmentation
    budget, with no extra dataset-length multiplication on top.
  - Methods whose loss only needs 1 view get augment_factor=2, so every
    method ends up exposed to ~2 augmented views/image/epoch --
    except sf2net, whose triplet loss has an irreducible 3-view floor
    (augment_factor=1 there means "no augmentation beyond the minimum
    the loss requires", not "equal total exposure to the other 9").
"""
import os
import torch

# ══════════════════════════════════════════════════════════════
#  GLOBAL / SHARED ACROSS ALL METHODS AND ALL 12 SETTINGS
# ══════════════════════════════════════════════════════════════

DATA_ROOT            = "/home/pai-ng/Jamal/smartphone_data"   # <-- set to your data path
SCANNER_SPECTRA    = {"green", "ir", "yellow", "pink", "white"}


BASE_RESULTS_DIR    = "./benchmark_results"




SEED                = 42
SPLITS_FILE        = "./palm_auth_openset_splits_50-50.json"       # shared, cached, seeded
TRAIN_ID_RATIO       = 0.50




TEST_GALLERY_RATIO   = 0.50

DEVICE              = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS          = 4
EVAL_EVERY           = 10       # epochs between full gallery/probe evaluations

os.makedirs(BASE_RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  PER-METHOD CONFIG
#  Fields common to every entry:
#    img_side        : model input resolution
#    channels         : 1 (grayscale ROI) or 3 (RGB)
#    normalize        : "roi" | "imagenet" | "unit"  (see dataset.py)
#    view_mode        : "single" | "paired" | "triplet"  (see dataset.py)
#    augment_factor   : fairness-adjusted (see module docstring above)
#    batch_size, num_epochs, lr, ...  : original per-method training hp
# ══════════════════════════════════════════════════════════════

METHODS = {

    "compnet": {
        "img_side": 128, "channels": 1, "normalize": "roi",
        "view_mode": "single", "augment_factor": 2,
        "batch_size": 128, "num_epochs": 300, "lr": 1e-3,
        "lr_step": 30, "lr_gamma": 0.8,
        "embedding_dim": 512, "arcface_s": 30.0, "arcface_m": 0.50, "dropout": 0.25,
    },
  
    "arcface": {
        "img_side": 112, "channels": 3, "normalize": "unit",
        "view_mode": "single", "augment_factor": 2,
        "batch_size": 32, "num_epochs": 100, "lr": 1e-4, "weight_decay": 5e-4,
        "arcface_s": 64.0, "arcface_m": 0.50,
        "pretrained_weights": "/home/pai-ng/Jamal/NIPS2026/face_models/checkpoints/r100_glint360k.onnx",  # <-- set this
        "freeze_ratio": 0.75,
    },

    "magface": {
        "img_side": 112, "channels": 3, "normalize": "unit",
        "view_mode": "single", "augment_factor": 2,
        "batch_size": 32, "num_epochs": 100, "lr": 1e-4, "weight_decay": 5e-4,
        "arc_s": 64.0, "m_l": 0.45, "m_u": 0.80, "l_a": 10.0, "u_a": 110.0,
        "lambda_g": 20.0,
        "pretrained_weights": "/home/pai-ng/Jamal/NIPS2026/face_models/checkpoints/magface_iresnet100.pth",  # <-- set this
        "freeze_ratio": 0.75,
    },

    "ppnet": {
        "img_side": 128, "channels": 1, "normalize": "roi",
        "view_mode": "single", "augment_factor": 2,
        "batch_size": 64, "num_epochs": 200, "lr": 1e-4,
        "lr_step": 17, "lr_gamma": 0.8,
        "contrastive_margin": 5.0, "w_l2": 1e-4, "w_contra": 2e-4, "w_dis": 1e-4,
    },

    "palmbridge": {
        "img_side": 128, "channels": 1, "normalize": "roi",
        "view_mode": "single", "augment_factor": 2,
        "batch_size": 16, "num_epochs": 100, "lr": 1e-3, "warmup_epochs": 5,
        "feature_dim": 512, "num_pb_vectors": 512,
        "num_gabor_filters": 32, "gabor_kernel_size": 15,
        "w_ori": 0.7, "w_map": 0.3, "lambda_con": 0.25,
        "arc_s": 48.0, "arc_m": 0.40, "alpha": 0.1, "beta": 1.0,
    },

    "ccnet": {
        "img_side": 128, "channels": 1, "normalize": "roi",
        "view_mode": "paired", "augment_factor": 1,
        "batch_size": 256, "num_epochs": 200, "lr": 1e-3,
        "lr_step": 17, "lr_gamma": 0.8,
        "comp_weight": 0.8, "dropout": 0.5, "arcface_s": 20.0, "arcface_m": 0.30,
        "ce_weight": 0.8, "con_weight": 0.2, "temperature": 0.07,
    },

    "co3net": {
        "img_side": 128, "channels": 1, "normalize": "roi",
        "view_mode": "paired", "augment_factor": 1,
        "batch_size": 256, "num_epochs": 200, "lr": 1e-3,
        "lr_step": 30, "lr_gamma": 0.6,
        "dropout": 0.5, "arcface_s": 20.0, "arcface_m": 0.30,
        "ce_weight": 0.8, "con_weight": 0.2, "temperature": 0.07,
    },

    "convnext": {
        "img_side": 112, "channels": 3, "normalize": "imagenet",
        "view_mode": "paired", "augment_factor": 1,
        "batch_size": 32, "num_epochs": 100, "lr": 1e-3,
        "margin": 0.50, "scale": 64.0, "lambda_supcon": 0.2, "supcon_temperature": 0.1,
    },

    "dino": {
        "img_side": 224, "channels": 3, "normalize": "imagenet",
        "view_mode": "paired", "augment_factor": 1,
        "batch_size": 32, "num_epochs": 100, "lr": 1e-3,
        "margin": 0.50, "scale": 64.0, "lambda_supcon": 0.2, "supcon_temperature": 0.1,
    },

    "sf2net": {
        "img_side": 128, "channels": 1, "normalize": "roi",
        "view_mode": "triplet", "augment_factor": 1,
        "batch_size": 256, "num_epochs": 200, "lr": 1e-3,
        "lr_step": 17, "lr_gamma": 0.8,
        "dropout": 0.5, "arcface_s": 30.0, "arcface_m": 0.50,
        "ce_weight": 0.7, "tl_weight": 0.3, "triplet_margin": 2.0,
        "vit_floor_num": 10,
    },
}

# Order in which benchmarking.py iterates methods (also the column order
# in the final EER / Rank-1 tables)
METHOD_ORDER = ["arcface", "magface", "compnet", "ppnet", "palmbridge",
                 "ccnet", "co3net", "convnext", "dino", "sf2net"]
