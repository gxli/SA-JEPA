#!/usr/bin/env python3
"""Generate gen_103 MHD/C12 mask-geometry sweep configs with +1 ConvNeXt layer."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "configs", "experiments")
BASE_CONFIG = "../base_pyramid_scaleaware_convnext.json"
MHD_PATTERN = "C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy"

FIXED_MASK_SCALES = (0.4, 0.8, 1.2, 1.6, 2.0)
FIXED_MASK_BOXES = (3, 7, 11, 15)
RANDOM_MASK_SCALE_RANGE = (0.4, 2.0)
RANDOM_MASK_BOX_RANGE = (3, 15)


def _float_tag(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _range_tag(values) -> str:
    return "_".join(_float_tag(v) for v in values)


def _int_range_tag(values) -> str:
    return "_".join(str(int(v)) for v in values)


def _pyramid_base() -> dict:
    return {
        "base_config": BASE_CONFIG,
        "data": {
            "npy_pattern": MHD_PATTERN,
            "log_transform": True,
            "norm_before_cdd": True,
        },
        "model": {
            "mode": "pyramid",
            "model_key": "cdd_scaleaware_convnext",
            "encoder_depth": 5,
            "mask_spacing_scaling": 2.0,
            "mask_size_scaling": 1.1,
            "mask_box_size": 0,
            "convnext_layer_dilations": [1, 1, 1, 1, 1],
            "predictor_spatial_conv": True,
            "predictor_hidden": 96,
            "predictor_layernorm": False,
            "scaleaware_norm_per_scale": False,
            "scaleaware_adapter_norm": True,
            "scaleaware_stem_norm": True,
            "scaleaware_final_norm": False,
            "use_grn": True,
            "post_log_transform": True,
            "normalize_loss_l2": True,
            "use_symmetric_feature_loss": False,
            "target_sampling_mode": "priority_sampling",
            "priority_top_percent": 15.0,
            "priority_n_target": "auto",
            "priority_min_targets_per_map": 10,
            "priority_dithering_pixels": 6,
            "target_nonoverlap": True,
            "target_allow_partial_overlap": 0.0,
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
            "symmetric_feature_loss_weight": 0.0,
            "compute_effective_rank": True,
            "scale_probe_enabled": True,
            "inference_tta_enabled": True,
            "inference_tta_mode": "flip4",
        },
    }


def _image_base() -> dict:
    cfg = _pyramid_base()
    cfg["model"]["mode"] = "image"
    cfg["model"]["model_key"] = "convnext_dense_masktoken"
    cfg["train"]["scale_probe_enabled"] = False
    return cfg


def _write(name: str, cfg: dict) -> str:
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(deepcopy(cfg), f, indent=2)
        f.write("\n")
    return path


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for stale_path in glob(os.path.join(OUT_DIR, "gen_103_*.json")):
        os.remove(stale_path)

    paths = []
    run = 1

    cfg = _pyramid_base()
    paths.append(_write(f"gen_103_mhd_run_{run:03d}_pyramid_ms1p1", cfg))
    run += 1

    for mask_scale in FIXED_MASK_SCALES:
        cfg = _pyramid_base()
        cfg["model"]["mask_size_scaling"] = float(mask_scale)
        paths.append(_write(f"gen_103_mhd_run_{run:03d}_pyramid_ms{_float_tag(mask_scale)}", cfg))
        run += 1

    for mask_box in FIXED_MASK_BOXES:
        cfg = _pyramid_base()
        cfg["model"]["mask_size_scaling"] = 0.0
        cfg["model"]["mask_box_size"] = int(mask_box)
        paths.append(_write(f"gen_103_mhd_run_{run:03d}_pyramid_fixed_mbox{mask_box}", cfg))
        run += 1

    cfg = _pyramid_base()
    cfg["model"]["mask_size_scaling"] = list(RANDOM_MASK_SCALE_RANGE)
    paths.append(_write(f"gen_103_mhd_run_{run:03d}_pyramid_random_ms{_range_tag(RANDOM_MASK_SCALE_RANGE)}", cfg))
    run += 1

    cfg = _pyramid_base()
    cfg["model"]["mask_size_scaling"] = 0.0
    cfg["model"]["mask_box_size"] = list(RANDOM_MASK_BOX_RANGE)
    paths.append(_write(f"gen_103_mhd_run_{run:03d}_pyramid_random_mbox{_int_range_tag(RANDOM_MASK_BOX_RANGE)}", cfg))
    run += 1

    for mask_box in FIXED_MASK_BOXES:
        cfg = _image_base()
        cfg["model"]["mask_size_scaling"] = 0.0
        cfg["model"]["mask_box_size"] = int(mask_box)
        paths.append(_write(f"gen_103_mhd_run_{run:03d}_image_masktoken_fixed_mbox{mask_box}", cfg))
        run += 1

    cfg = _image_base()
    cfg["model"]["mask_size_scaling"] = 0.0
    cfg["model"]["mask_box_size"] = list(RANDOM_MASK_BOX_RANGE)
    paths.append(_write(f"gen_103_mhd_run_{run:03d}_image_masktoken_random_mbox{_int_range_tag(RANDOM_MASK_BOX_RANGE)}", cfg))

    for path in paths:
        print(os.path.relpath(path, ROOT))
    print(f"\nTotal: {len(paths)} configs written to {os.path.relpath(OUT_DIR, ROOT)}")


if __name__ == "__main__":
    main()
