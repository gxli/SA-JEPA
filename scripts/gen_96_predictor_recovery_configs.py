#!/usr/bin/env python3
"""Generate gen_96 predictor near-collapse recovery ablations."""
from __future__ import annotations

import json
import os
from glob import glob
from copy import deepcopy

from rename_gen_96_configs import canonical_config_name

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "configs", "experiments")
BASE_CONFIG = "../base_pyramid_scaleaware_convnext.json"

DATASETS = [
    "C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy",
    "ngc3627_12m+7m+tp_co21_strict_mom0.npy_sm.npy",
    "chengdu.npy",
]


def _base_config(npy_pattern: str) -> dict:
    cfg = {
        "base_config": BASE_CONFIG,
        "data": {
            "npy_pattern": npy_pattern,
        },
        "model": {
            "mode": "pyramid",
            "model_key": "cdd_scaleaware_convnext",
            "mask_spacing_scaling": 2.0,
            "normalize_loss_l2": True,
            "target_sampling_mode": "priority",
            "priority_top_percent": 15.0,
            "priority_n_target": "auto",
            "priority_min_targets_per_map": 10,
            "priority_dithering_pixels": 6,
            "target_nonoverlap": True,
            "predictor_layernorm": False,
            "predictor_spatial_conv": True,
            "predictor_hidden": 96,
            "scaleaware_norm_per_scale": False,
            "mask_scale_factor": 1.0,
            "mask_footprint_px": 0,
            "use_symmetric_feature_loss": False,
            "post_log_transform": True,
        },
        "train": {
            "vicreg_spatial_mode": "pooled",
            "ema_momentum_base": 0.99,
            "ema_momentum_final": 0.9999,
            "ema_warmup_fraction": 0.25,
            "epochs": 5,
            "log_interval": 1,
            "prediction_loss_weight": 50.0,
            "spread_regularizer": {"type": "std_hinge", "target": "context", "weight": 10.0, "target_std": 1.0, "eps": 1e-4},
            "symmetry_loss_weight": 0.0,
            "inference_tta_enabled": True,
            "inference_tta_mode": "flip4",
        },
    }
    if npy_pattern == "chengdu.npy":
        cfg["train"]["batch_size"] = 1
    return cfg


# The ms1p0 entry is the real candidate. The other entries change one knob at a time.
VARIANTS = [
    ({
        "mask_scale_factor": 0.8,
    },),
    ({},),
    ({
        "mask_scale_factor": 1.2,
    },),
    ({
        "scaleaware_norm_per_scale": True,
    },),
    ({
        "use_symmetric_feature_loss": True,
    }, {
        "symmetry_loss_weight": 1.0,
    }),
    ({
        "predictor_spatial_conv": False,
    },),
    ({
        "predictor_hidden": 64,
    },),
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for stale_path in glob(os.path.join(OUT_DIR, "gen_96_run_*.json")):
        os.remove(stale_path)
    paths = []
    for npy_pattern in DATASETS:
        for run, variant in enumerate(VARIANTS, start=1):
            model_overrides, *train_override_values = variant
            cfg = _base_config(npy_pattern)
            cfg["model"].update(model_overrides)
            if train_override_values:
                cfg["train"].update(train_override_values[0])
            name = canonical_config_name(run, cfg)
            path = os.path.join(OUT_DIR, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(deepcopy(cfg), f, indent=2)
                f.write("\n")
            paths.append(path)

    for path in paths:
        print(os.path.relpath(path, ROOT))
    print(f"\nTotal: {len(paths)} configs written to {os.path.relpath(OUT_DIR, ROOT)}")


if __name__ == "__main__":
    main()
