#!/usr/bin/env python3
"""Generate gen_77 configs testing shared ConvNeXt + FiLM scale-aware encoders.

Variants:
  - cdd_film_scaleaware_convnext (shared ConvNeXt + scale FiLM)
  - cdd_film_scaleaware_convnext + per-scale adapters
  - cdd_film_scaleaware_convnext (pure shared, no FiLM)
  - baseline cdd_scaleaware_convnext (separate branches, for reference)
"""
from __future__ import annotations

import json
import os
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "configs", "experiments")

BASE_CONFIG = "../base_pyramid_scaleaware_convnext.json"
MASK_BOX_RANGE = [3, 11]

DATASETS = {
    "mhd": {"label": "mhd"},
    "ngc": {
        "label": "ngc",
        "npy_pattern": "ngc3627_12m+7m+tp_co21_strict_mom0.npy_sm.npy",
    },
    "jhu": {
        "label": "jhu",
        "npy_pattern": "C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy",
    },
}

# (mask_size_scaling, mask_box_size, mask_label)
MASK_VARIANTS = [
    (0.8, 0, "pyramid_ms0p8"),
    (1.2, 0, "pyramid_ms1p2"),
    (0.0, MASK_BOX_RANGE, "random_mbox"),
]

# (encoder_type, use_film, use_per_scale_adapters, encoder_label)
ENCODER_VARIANTS = [
    ("cdd_film_scaleaware_convnext", True,  False, "film"),
    ("cdd_film_scaleaware_convnext", True,  True,  "film_adapters"),
    ("cdd_film_scaleaware_convnext", False, False, "shared_pure"),
    ("cdd_scaleaware_convnext",      None,  None,  "separate_branches_baseline"),
]


def _base_config() -> dict:
    return {
        "base_config": BASE_CONFIG,
        "model": {
            "mode": "pyramid",
            "mask_spacing_scaling": 2.0,
            "normalize_loss_l2": True,
            "use_symmetric_feature_loss": True,
            "priority_top_percent": 15.0,
            "target_sampling_mode": "priority_sampling",
            "priority_n_target": "auto",
            "priority_min_targets_per_map": 10,
            "priority_dithering_pixels": 6,
            "target_nonoverlap": True,
            "predictor_layernorm": False,
            "predictor_spatial_conv": False,
        },
        "train": {
            "epochs": 10,
            "log_interval": 1,
            "mse_loss_weight": 100.0,
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
    paths = []
    run = 0
    for dkey, ds in DATASETS.items():
        for mscale, mbox, mask_label in MASK_VARIANTS:
            for enc_type, use_film, use_adapters, enc_label in ENCODER_VARIANTS:
                run += 1
                cfg = _base_config()
                cfg["model"]["model_key"] = enc_type
                cfg["model"]["mask_size_scaling"] = mscale
                cfg["model"]["mask_box_size"] = mbox

                if use_film is not None:
                    cfg["model"]["use_film"] = use_film
                if use_adapters is not None:
                    cfg["model"]["use_per_scale_adapters"] = use_adapters

                if "npy_pattern" in ds:
                    cfg["data"] = cfg.get("data", {})
                    cfg["data"]["npy_pattern"] = ds["npy_pattern"]

                name = f"gen_77_run_{run}_{dkey}_symmetric_sigreg_l2_{mask_label}_{enc_label}"
                paths.append(write_config(name, cfg))

    for path in paths:
        print(os.path.relpath(path, ROOT))
    print(f"\nTotal: {len(paths)} configs written to {os.path.relpath(OUT_DIR, ROOT)}")


if __name__ == "__main__":
    main()
