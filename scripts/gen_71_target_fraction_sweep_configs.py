#!/usr/bin/env python3
"""Generate gen_71 target-fraction sweep configs (priority_top_percent=30%)."""
from __future__ import annotations

import json
import os
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "configs", "experiments")

BASE_CONFIG = "../base_pyramid_scaleaware_convnext.json"
MASK_BOX_RANGE = (3, 11)
TARGET_FRACTIONS = [1.0, 0.5, 0.3]

DATASETS = {
    "mhd": {"label": "mhd"},
    "ngc": {
        "label": "ngc",
        "npy_pattern": "ngc3627_12m+7m+tp_co21_strict_mom0.npy_sm.npy",
    },
}


def _base_config(dataset: str) -> dict:
    ds = DATASETS[dataset]
    cfg: dict = {
        "base_config": BASE_CONFIG,
        "model": {
            "mode": "pyramid",
            "model_key": "cdd_scaleaware_convnext",
            "mask_scale_factor": 0.0,
            "mask_spacing_scaling": 2.0,
            "mask_footprint_px": list(MASK_BOX_RANGE),
            "normalize_loss_l2": True,
            "use_symmetric_feature_loss": True,
            "priority_top_percent": 30.0,
            "target_sampling_mode": "priority",
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
            "umap": {
                "l2_normalize": True,
            },
        },
    }
    if "npy_pattern" in ds:
        cfg.setdefault("data", {})["npy_pattern"] = ds["npy_pattern"]
    return cfg


def write_config(name: str, cfg: dict) -> str:
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(deepcopy(cfg), f, indent=2)
        f.write("\n")
    return path


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = []
    run = 0
    for dataset in DATASETS:
        for frac in TARGET_FRACTIONS:
            run += 1
            ds = DATASETS[dataset]
            cfg = _base_config(dataset)
            cfg["model"]["active_target_fraction"] = float(frac)
            name = f"gen_71_run_{run}_{ds['label']}_symmetric_spread_l2_mbox_toppct30_targetfrac_{str(frac).replace('.', 'p')}"
            paths.append(write_config(name, cfg))

    for path in paths:
        print(os.path.relpath(path, ROOT))
    print(f"\nTotal: {len(paths)} configs written to {os.path.relpath(OUT_DIR, ROOT)}")


if __name__ == "__main__":
    main()
