#!/usr/bin/env python3
"""Generate gen_69 random mask-size ablation configs for MHD, NGC, and JHU."""
from __future__ import annotations

import json
import os
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "configs", "experiments")

BASE_CONFIG = "../base_pyramid_scaleaware_convnext.json"
MASK_SCALE_RANGE = (0.4, 2.4)
MASK_BOX_RANGE = (3, 11)

DATASETS = {
    "mhd": {
        "label": "mhd",
    },
    "ngc": {
        "label": "ngc",
        "npy_pattern": "ngc3627_12m+7m+tp_co21_strict_mom0.npy_sm.npy",
    },
    "jhu": {
        "label": "jhu",
        "npy_pattern": "decomp_E_rot.npy_slice_sm_1.0.npy",
    },
}


def _base_config(dataset: str) -> dict:
    ds = DATASETS[dataset]
    cfg: dict = {
        "base_config": BASE_CONFIG,
        "model": {
            "mode": "pyramid",
            "model_key": "cdd_scaleaware_convnext",
            "mask_scale_factor": 1.2,
            "mask_spacing_scaling": 2.0,
            "mask_footprint_px": 0,
            "normalize_loss_l2": True,
            "use_symmetric_feature_loss": True,
            "target_sampling_mode": "priority",
            "priority_top_percent": 15.0,
            "priority_n_target": "auto",
            "priority_min_targets_per_map": 10,
            "priority_dithering_pixels": 6,
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
    if "npy_pattern" in ds:
        cfg["data"] = {"npy_pattern": ds["npy_pattern"]}
    return cfg


def write_config(name: str, cfg: dict) -> str:
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(deepcopy(cfg), f, indent=2)
        f.write("\n")
    return path


def scale_only_config(dataset: str) -> tuple[str, dict]:
    ds = DATASETS[dataset]
    cfg = _base_config(dataset)
    cfg["model"]["mask_scale_factor"] = list(MASK_SCALE_RANGE)
    cfg["model"]["mask_footprint_px"] = 0
    name = f"gen_69_run_1_{ds['label']}_scaleaware_symmetric_spread_l2_random_mscale"
    return name, cfg


def box_only_config(dataset: str) -> tuple[str, dict]:
    ds = DATASETS[dataset]
    cfg = _base_config(dataset)
    cfg["model"]["mask_scale_factor"] = 0.0
    cfg["model"]["mask_footprint_px"] = list(MASK_BOX_RANGE)
    name = f"gen_69_run_2_{ds['label']}_scaleaware_symmetric_spread_l2_random_mbox"
    return name, cfg


def hybrid_config(dataset: str) -> tuple[str, dict]:
    ds = DATASETS[dataset]
    cfg = _base_config(dataset)
    cfg["model"]["mask_scale_factor"] = list(MASK_SCALE_RANGE)
    cfg["model"]["mask_footprint_px"] = list(MASK_BOX_RANGE)
    name = f"gen_69_run_3_{ds['label']}_scaleaware_symmetric_spread_l2_random_hybrid"
    return name, cfg


def image_dense_config(dataset: str) -> tuple[str, dict]:
    ds = DATASETS[dataset]
    cfg = _base_config(dataset)
    cfg["model"]["mode"] = "image"
    cfg["model"]["model_key"] = "convnext_dense_masktoken"
    cfg["model"]["mask_scale_factor"] = 0.0
    cfg["model"]["mask_footprint_px"] = list(MASK_BOX_RANGE)
    cfg["model"].pop("use_symmetric_feature_loss", None)
    name = f"gen_69_run_4_{ds['label']}_convnext_dense_masktoken_spread_l2_random_mbox"
    return name, cfg


VARIANTS = [
    scale_only_config,
    box_only_config,
    hybrid_config,
    image_dense_config,
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = []
    for dataset in DATASETS:
        for variant_fn in VARIANTS:
            name, cfg = variant_fn(dataset)
            paths.append(write_config(name, cfg))

    for path in paths:
        print(os.path.relpath(path, ROOT))
    print(f"\nTotal: {len(paths)} configs written to {os.path.relpath(OUT_DIR, ROOT)}")


if __name__ == "__main__":
    main()
