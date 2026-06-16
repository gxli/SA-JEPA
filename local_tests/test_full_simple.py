#!/usr/bin/env python3
"""Train gen_186 MHD run 2 via API, then run inference on two different files."""
from __future__ import annotations

import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, "local_tests", ".mplconfig"))
sys.path.insert(0, ROOT)

from sajepa import ScaleAwareJEPA

DATA_NPY = os.path.join(ROOT, "data", "C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy")
TRAIN_DIR = os.path.join(ROOT, "local_tests", "sessions_full_simple")
TRAIN_NAME = "mhd_ms1p2_pooled_api_demo"
INFER_DIR = os.path.join(ROOT, "local_tests", "inference_full_simple")

config = {
    "data": {
        "data_root": os.path.join(ROOT, "data"),
        "npy_pattern": os.path.basename(DATA_NPY),
        "num_samples": 200,
        "d4_augment": True,
    },
    "model": {"mask_size_scaling": 1.2},
    "train": {
        "epochs": 10,
        "spread_regularizer": {
            "type": "std_hinge", "target": "context",
            "spatial_mode": "pooled", "weight": 5.0, "target_std": 1.0,
        },
    },
}


def main() -> int:
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(INFER_DIR, exist_ok=True)

    # 1. Train
    model = ScaleAwareJEPA().train(configs=config, config_name=TRAIN_NAME, sessions_dir=TRAIN_DIR)
    train_session = model.session_dir
    print(f"training session:  {train_session}")
    model.generate_dashboard(os.path.join(train_session, "dashboard.html"))

    # 2. Inference on the original data file
    print("\n--- inference round 1: original data ---")
    inf1 = ScaleAwareJEPA.infer_from_session(train_session, DATA_NPY, INFER_DIR, make_dashboard=True)
    print(f"inference session:  {inf1}")

    # 3. Inference on the SAME file (change this to a different file to test)
    print("\n--- inference round 2: same file (change DATA_NPY2 to test different) ---")
    DATA_NPY2 = os.environ.get("INFER_FILE2", DATA_NPY)
    inf2 = ScaleAwareJEPA.infer_from_session(train_session, DATA_NPY2, INFER_DIR, make_dashboard=True)
    print(f"inference session:  {inf2}")

    # 4. Verification section — inference on a DIFFERENT file.
    #    Set DIFFERENT_FILE to a different .npy to test cross-file inference.
    DIFFERENT_FILE = os.environ.get("DIFFERENT_FILE", "")
    if DIFFERENT_FILE and os.path.exists(DIFFERENT_FILE):
        print(f"\n--- verification: inference on different file ---")
        print(f"  file: {DIFFERENT_FILE}")
        inf3 = ScaleAwareJEPA.infer_from_session(
            train_session, DIFFERENT_FILE, os.path.join(INFER_DIR, "verification"),
            make_dashboard=True,
        )
        print(f"  session: {inf3}")
        m3 = ScaleAwareJEPA.load_session(inf3)
        m3.generate_dashboard(os.path.join(inf3, "dashboard.html"))
        print(f"  dashboard: {os.path.join(inf3, 'dashboard.html')}")
    else:
        print(f"\n--- verification: skipped (set DIFFERENT_FILE env var) ---")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
