#!/usr/bin/env python3
"""Generate gen_68 cross-dataset configs at mask_size_scaling=1.2."""
from __future__ import annotations

import json
import os
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "configs", "experiments")

BASE_CONFIG = "../base_pyramid_scaleaware_convnext.json"


def _base_cfg(npy_pattern: str) -> dict:
    return {
        "base_config": BASE_CONFIG,
        "data": {
            "npy_pattern": npy_pattern,
        },
        "model": {
            "mode": "pyramid",
            "model_key": "cdd_scaleaware_convnext",
            "mask_size_scaling": 1.2,
            "mask_spacing_scaling": 2.0,
            "mask_box_size": 0,
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
            "jepa_loss_weight": 100.0,
            "vicreg_var_weight": 0.0,
            "vicreg_cov_weight": 0.0,
            "sigreg_weight": 1.0,
            "sigreg_sketch_dim": 64,
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


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    configs = [
        (
            "gen_68_run_1_jhu_symmetric_patternsep_priority_l2_mss_1p2",
            _base_cfg("decomp_E_rot.npy_slice_sm_1.0.npy"),
        ),
        (
            "gen_68_run_2_ngc_symmetric_patternsep_priority_l2_mss_1p2",
            _base_cfg("ngc3627_12m+7m+tp_co21_strict_mom0.npy_sm.npy"),
        ),
    ]
    paths = [write_config(name, cfg) for name, cfg in configs]
    for path in paths:
        print(os.path.relpath(path, ROOT))
    print(f"\nTotal: {len(paths)} configs written to {os.path.relpath(OUT_DIR, ROOT)}")


if __name__ == "__main__":
    main()
