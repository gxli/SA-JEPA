#!/usr/bin/env python3
"""Generate gen_69 random mask-size ablation configs."""
from __future__ import annotations

import json
import os
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "configs", "experiments")

BASE_CONFIG = "../base_pyramid_scaleaware_convnext.json"
MASK_SCALE_RANGE = (0.4, 2.4)
MASK_BOX_RANGE = (5, 11)


def _base_config() -> dict:
    return {
        "base_config": BASE_CONFIG,
        "model": {
            "mode": "pyramid",
            "model_key": "cdd_scaleaware_convnext",
            "mask_size_scaling": 1.2,
            "mask_spacing_scaling": 2.0,
            "mask_box_size": 0,
            "normalize_loss_l2": True,
            "use_symmetric_feature_loss": True,
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


def scale_only_config() -> tuple[str, dict]:
    cfg = _base_config()
    cfg["model"]["mask_size_scaling"] = list(MASK_SCALE_RANGE)
    cfg["model"]["mask_box_size"] = 0
    name = "gen_69_run_1_mhd_scaleaware_symmetric_sigreg_l2_random_mscale"
    return name, cfg


def box_only_config() -> tuple[str, dict]:
    cfg = _base_config()
    cfg["model"]["mask_size_scaling"] = 0.0
    cfg["model"]["mask_box_size"] = list(MASK_BOX_RANGE)
    name = "gen_69_run_2_mhd_scaleaware_symmetric_sigreg_l2_random_mbox"
    return name, cfg


def hybrid_config() -> tuple[str, dict]:
    cfg = _base_config()
    cfg["model"]["mask_size_scaling"] = list(MASK_SCALE_RANGE)
    cfg["model"]["mask_box_size"] = list(MASK_BOX_RANGE)
    name = "gen_69_run_3_mhd_scaleaware_symmetric_sigreg_l2_random_hybrid"
    return name, cfg


def image_dense_config() -> tuple[str, dict]:
    cfg = _base_config()
    cfg["model"]["mode"] = "image"
    cfg["model"]["model_key"] = "convnext_dense_masktoken"
    cfg["model"]["mask_size_scaling"] = 0.0
    cfg["model"]["mask_box_size"] = list(MASK_BOX_RANGE)
    cfg["model"].pop("use_symmetric_feature_loss", None)
    name = "gen_69_run_4_mhd_convnext_dense_masktoken_sigreg_l2_random_mbox"
    return name, cfg


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = []
    for name, cfg in (
        scale_only_config(),
        box_only_config(),
        hybrid_config(),
        image_dense_config(),
    ):
        paths.append(write_config(name, cfg))

    for path in paths:
        print(os.path.relpath(path, ROOT))
    print(f"\nTotal: {len(paths)} configs written to {os.path.relpath(OUT_DIR, ROOT)}")


if __name__ == "__main__":
    main()
