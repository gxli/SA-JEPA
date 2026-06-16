#!/usr/bin/env python3
"""Minimal public-API demo for gen_186 MHD run 002.

Runs the ms=1.2 pooled MHD baseline using default config + inline overrides,
saves a training session, generates dashboards, then demonstrates the separate
inference-session workflow on an `.npy` file.
"""

from __future__ import annotations

import os
import sys


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
    "model": {
        "mask_size_scaling": 1.2,
    },
    "train": {
        "epochs": 10,
        "spread_regularizer": {
            "type": "std_hinge",
            "target": "context",
            "spatial_mode": "pooled",
            "weight": 5.0,
            "target_std": 1.0,
        },
    },
}


def main() -> int:
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(INFER_DIR, exist_ok=True)

    # 1. Train from default config + inline overrides.
    model = ScaleAwareJEPA().train(
        configs=config,
        config_name=TRAIN_NAME,
        sessions_dir=TRAIN_DIR,
    )
    train_session = model.session_dir

    # 2. Generate the training-session dashboard.
    print(f"training session:  {train_session}")
    train_dashboard = os.path.join(train_session, "dashboard.html")
    model.generate_dashboard(train_dashboard)
    print(f"training dashboard: {train_dashboard}")

    # 3. Run inference on an .npy into a separate inference-only session.
    inference_session = ScaleAwareJEPA.infer_from_session(
        train_session,
        DATA_NPY,
        INFER_DIR,
        make_dashboard=True,
    )

    # 4. Load the inference session and generate its dashboard.
    print(f"inference session:  {inference_session}")
    inference = ScaleAwareJEPA.load_session(inference_session)
    infer_dashboard = os.path.join(inference_session, "dashboard.html")
    inference.generate_dashboard(infer_dashboard)
    print(f"inference dashboard: {infer_dashboard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
