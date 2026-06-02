#!/usr/bin/env python3
"""Generate gen_66 MHD symmetric pattern-separation configs."""
from __future__ import annotations

import json
import os
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "configs", "experiments")

BASE_CONFIG = "../base_pyramid_scaleaware_convnext.json"
MASK_SIZE_SCALINGS = (0.4, 0.8, 1.2, 1.6, 2.0, 2.4)
BOX_SIZES = (3, 5, 7, 9, 11)


def _fmt(v: float) -> str:
    return f"{v:.1f}".replace(".", "p")


def _base_config(mscale: float, mask_footprint_px: int) -> dict:
    return {
        "base_config": BASE_CONFIG,
        "model": {
            "mode": "pyramid",
            "model_key": "cdd_scaleaware_convnext",
            "mask_scale_factor": float(mscale),
            "mask_spacing_scaling": 2.0,
            "mask_footprint_px": int(mask_footprint_px),
            "normalize_loss_l2": True,
            "use_symmetric_feature_loss": True,
            "scaleaware_norm_per_scale": True,
            "target_sampling_mode": "priority_sampling",
            "priority_top_percent": 15.0,
            "priority_n_target": "auto",
            "priority_min_targets_per_map": 10,
            "priority_dithering_pixels": 6,
        },
        "train": {
            "epochs": 10,
            "log_interval": 1,
            "prediction_loss_weight": 100.0,
            "experimental_losses": {"vicreg_var_weight": 15.0, "vicreg_cov_weight": 15.0},
            "spread_regularizer": {"type": "std_hinge", "target": "context", "weight": 5.0, "target_std": 1.0, "eps": 1e-4},
            "vicreg_spatial_mode": "pooled",
            "inference_tta_enabled": True,
            "inference_tta_mode": "flip4",
        },
    }


def write_config(name: str, cfg: dict) -> str:
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(deepcopy(cfg), f, indent=2)
        f.write("\n")
    return path


def write_scale_config(mscale: float) -> str:
    cfg = _base_config(mscale=mscale, mask_footprint_px=0)
    name = f"gen_66_run_1_mhd_symmetric_patternsep_priority_l2_mss_{_fmt(mscale)}"
    return write_config(name, cfg)


def write_box_config(mask_footprint_px: int) -> str:
    cfg = _base_config(mscale=0.0, mask_footprint_px=mask_footprint_px)
    name = f"gen_66_run_2_mhd_symmetric_patternsep_priority_l2_mss_0p0_mbox_{mask_footprint_px:02d}"
    return write_config(name, cfg)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = [write_scale_config(mscale) for mscale in MASK_SIZE_SCALINGS]
    paths.extend(write_box_config(mask_footprint_px) for mask_footprint_px in BOX_SIZES)
    for path in paths:
        print(os.path.relpath(path, ROOT))
    print(f"\nTotal: {len(paths)} configs written to {os.path.relpath(OUT_DIR, ROOT)}")


if __name__ == "__main__":
    main()
