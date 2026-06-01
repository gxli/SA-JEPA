#!/usr/bin/env python3
"""Generate gen_97 compact cross-field stability configs."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "configs", "experiments")
BASE_CONFIG = "../base_pyramid_scaleaware_convnext.json"

DATASETS = [
    ("c12", "C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy"),
    ("ngc", "ngc3627_12m+7m+tp_co21_strict_mom0.npy_sm.npy"),
    ("chengdu", "chengdu.npy"),
]

# (mask_size_scaling, symmetry loss weight, scale-aware per-scale normalization)
VARIANTS = [
    (1.0, 0.0, False),
    (1.2, 0.0, False),
    (1.0, 0.01, False),
    (1.0, 0.0, True),
    (1.2, 0.0, True),
    (1.0, 0.01, True),
]


def _float_tag(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _symmetry_tag(value: float) -> str:
    return "0" if float(value) == 0.0 else _float_tag(value)


def _config(
    npy_pattern: str,
    mask_size_scaling: float,
    symmetry_weight: float,
    norm_per_scale: bool,
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
            "mask_size_scaling": float(mask_size_scaling),
            "mask_box_size": 0,
            "normalize_loss_l2": True,
            "predictor_spatial_conv": True,
            "predictor_layernorm": False,
            "predictor_hidden": 96,
            "scaleaware_norm_per_scale": bool(norm_per_scale),
            "post_log_transform": True,
            "use_grn": True,
            "use_symmetric_feature_loss": symmetry_weight > 0.0,
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
            "symmetric_feature_loss_weight": float(symmetry_weight),
            "inference_tta_enabled": True,
            "inference_tta_mode": "flip4",
        },
    }
    if npy_pattern == "chengdu.npy":
        cfg["train"]["batch_size"] = 1
    return cfg


def _name(
    dataset: str,
    run: int,
    mask_size_scaling: float,
    symmetry_weight: float,
    norm_per_scale: bool,
) -> str:
    return (
        f"gen_97_{dataset}_run_{run:03d}"
        f"_ms{_float_tag(mask_size_scaling)}"
        "_pred3x3_h96_predln_off"
        f"_perscalenorm_{'on' if norm_per_scale else 'off'}"
        f"_symw{_symmetry_tag(symmetry_weight)}"
        "_sigpred_on"
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for stale_path in glob(os.path.join(OUT_DIR, "gen_97_*.json")):
        os.remove(stale_path)

    paths = []
    for dataset, npy_pattern in DATASETS:
        for run, (mask_size_scaling, symmetry_weight, norm_per_scale) in enumerate(VARIANTS, start=1):
            cfg = _config(npy_pattern, mask_size_scaling, symmetry_weight, norm_per_scale)
            name = _name(dataset, run, mask_size_scaling, symmetry_weight, norm_per_scale)
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
