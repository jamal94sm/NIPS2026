"""
config.py — Central configuration for CO3Net palmprint recognition.
Edit only this file to change datasets, hyperparameters, or paths.
"""

CONFIG = {
    # ── Dataset selection ───────────────────────────────────────────────
    # Choices: "CASIA-MS" | "Smartphone" | "MPDv2"
    "train_data"           : "MPDv2",
    "test_data"            : "MPDv2",

    # ── Dataset paths ───────────────────────────────────────────────────
    "casiams_data_root"    : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    "smartphone_data_root" : "/home/pai-ng/Jamal/smartphone_data",
    "mpd_data_root"        : "/home/pai-ng/Jamal/MPDv2_mediapipe_roi",

    # ── Splitting ratios ────────────────────────────────────────────────
    "train_subject_ratio"  : 0.80,   # fraction of subjects → train (same-dataset mode)
    "test_gallery_ratio"   : 0.50,   # fraction of test-subject images → gallery

    # ── CASIA-MS sampling ───────────────────────────────────────────────
    "n_casia_subjects"     : 190,
    "n_casia_samples"      : 2776,

    # ── MPDv2 sampling ──────────────────────────────────────────────────
    "n_mpd_subjects"       : 190,
    "n_mpd_samples"        : 2850,   # 190 × 15

    # ── Training ────────────────────────────────────────────────────────
    "img_side"             : 128,
    "batch_size"           : 256,
    "num_epochs"           : 100,
    "lr"                   : 0.001,
    "lr_step"              : 30,
    "lr_gamma"             : 0.6,
    "dropout"              : 0.5,
    "arcface_s"            : 20.0,
    "arcface_m"            : 0.30,
    "ce_weight"            : 0.8,
    "con_weight"           : 0.2,
    "temperature"          : 0.07,
    "augment_factor"       : 4,

    # ── Misc ────────────────────────────────────────────────────────────
    "results_dir"          : "./rst_co3net",
    "random_seed"          : 42,
    "save_every"           : 50,
    "eval_every"           : 50,
    "num_workers"          : 4,
    "eval_only"            : False,

}
