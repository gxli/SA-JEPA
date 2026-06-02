#!/usr/bin/env python3
"""Generate gen_67 MHD symmetric pattern-separation configs with lattice sampling."""
from __future__ import annotations

import json
import os
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "configs", "experiments")

BASE_CONFIG = "../base_pyramid_scaleaware_convnext.json"
MASK_SIZE_SCALINGS = (0.4, 0.8, 1.2, 1.6)


def _fmt(v: float) -> str:
    return f"{v:.1f}".replace(".", "p")


def write_config(mscale: float) -> str:
    cfg = {
        "base_config": BASE_CONFIG,
        "model": {
            "mode": "pyramid",
            "model_key": "cdd_scaleaware_convnext",
            "mask_scale_factor": float(mscale),
            "mask_spacing_scaling": 2.0,
            "mask_footprint_px": 0,
            "normalize_loss_l2": True,
            "use_symmetric_feature_loss": True,
            "scaleaware_norm_per_scale": True,
            "target_sampling_mode": "grid",
        },
        "train": {
            "epochs": 10,
            "log_interval": 1,
            "prediction_loss_weight": 100.0,
            "spread_regularizer": {"type": "std_hinge", "target": "context", "weight": 1.0, "target_std": 1.0, "eps": 1e-4},
            "inference_tta_enabled": True,
            "inference_tta_mode": "flip4",
        },
    }
    name = f"gen_67_run_1_mhd_symmetric_patternsep_lattice_l2_mss_{_fmt(mscale)}"
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(deepcopy(cfg), f, indent=2)
        f.write("\n")
    return path


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = [write_config(mscale) for mscale in MASK_SIZE_SCALINGS]
    for path in paths:
        print(os.path.relpath(path, ROOT))
    print(f"\nTotal: {len(paths)} configs written to {os.path.relpath(OUT_DIR, ROOT)}")


if __name__ == "__main__":
    main()
