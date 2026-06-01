#!/usr/bin/env python3
"""Generate gen_98 configs: symmetry-on with norm toggles."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "configs", "experiments")
BASE_CONFIG = "../base_pyramid_scaleaware_convnext.json"

DATASETS = [
    ("mhd", "MHD_2D.npy"),
    ("ngc", "ngc3627_12m+7m+tp_co21_strict_mom0.npy_sm.npy"),
    ("chengdu", "chengdu.npy"),
]

# (scaleaware_norm_per_scale, scaleaware_adapter_norm)
VARIANTS = [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
]

MASK_SIZE_SCALING = 1.0
SYMMETRY_WEIGHT = 0.02


def _config(
    npy_pattern: str,
    norm_per_scale: bool,
    adapter_norm: bool,
) -> dict:
    cfg = {
        "base_config": BASE_CONFIG,
        "data": {
            "npy_pattern": npy_pattern,
            "log_transform": True,
            "norm_before_cdd": True,
        },
        "model": {
            "mode": "pyramid",
            "model_key": "cdd_scaleaware_convnext",
            "mask_spacing_scaling": 2.0,
            "mask_size_scaling": MASK_SIZE_SCALING,
            "mask_box_size": 0,
            "normalize_loss_l2": True,
            "predictor_spatial_conv": True,
            "predictor_layernorm": False,
            "predictor_hidden": 96,
            "scaleaware_norm_per_scale": bool(norm_per_scale),
            "scaleaware_adapter_norm": bool(adapter_norm),
            "post_log_transform": True,
            "use_grn": True,
            "use_symmetric_feature_loss": True,
            "target_sampling_mode": "priority_sampling",
            "priority_top_percent": 15.0,
            "priority_n_target": "auto",
            "priority_min_targets_per_map": 10,
            "priority_dithering_pixels": 6,
            "target_nonoverlap": True,
        },
        "train": {
            "epochs": 5,
            "log_interval": 1,
            "vicreg_spatial_mode": "pooled",
            "ema_momentum_base": 0.99,
            "ema_momentum_final": 0.9999,
            "ema_warmup_fraction": 0.25,
            "mse_loss_weight": 50.0,
            "vicreg_var_weight": 0.0,
            "vicreg_cov_weight": 0.0,
            "sigreg_on_pred": True,
            "sigreg_weight": 10.0,
            "sigreg_sketch_dim": 64,
            "symmetric_feature_loss_weight": SYMMETRY_WEIGHT,
            "inference_tta_enabled": True,
            "inference_tta_mode": "flip4",
        },
    }
    if npy_pattern == "chengdu.npy":
        cfg["train"]["batch_size"] = 1
    return cfg


def _name(dataset: str, run: int, norm_per_scale: bool, adapter_norm: bool) -> str:
    return (
        f"gen_98_{dataset}_run_{run:03d}"
        "_ms1p0"
        "_pred3x3_h96_predln_off"
        f"_perscalenorm_{'on' if norm_per_scale else 'off'}"
        f"_adapternorm_{'on' if adapter_norm else 'off'}"
        "_symw0p02"
        "_sigpred_on"
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for stale_path in glob(os.path.join(OUT_DIR, "gen_98_*.json")):
        os.remove(stale_path)

    paths = []
    for dataset, npy_pattern in DATASETS:
        for run, (norm_per_scale, adapter_norm) in enumerate(VARIANTS, start=1):
            cfg = _config(npy_pattern, norm_per_scale, adapter_norm)
            name = _name(dataset, run, norm_per_scale, adapter_norm)
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
