#!/usr/bin/env python3
"""Generate gen_52 MHD configs (mask_scale sweep, auto targets, min 10/map)."""
from __future__ import annotations

import json
import os
from copy import deepcopy

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

BASE_DATA = {
    "data_root": "data",
    "npy_pattern": "C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy",
    "num_samples": 200,
    "log_eps": 1e-06,
    "cdd_mode": "log",
    "cdd_constrained": True,
    "cdd_sm_mode": "reflect",
    "cube_slice_strategy": "center",
    "cube_slice_axis": 0,
    "cube_slice_index": 0,
    "random_roll_max": 0,
    "d4_augment": True,
}

BASE_TRAIN = {
    "batch_size": 8,
    "epochs": 10,
    "lr": 0.0001,
    "weight_decay": 1e-05,
    "num_workers": 0,
    "log_interval": 1,
    "prediction_loss_weight": 100.0,
    "spread_regularizer": {"type": "std_hinge", "target": "context", "weight": 1.0, "target_std": 1.0, "eps": 1e-4},
    "force_recompute_inference": True,
    "umap": {
        "n_neighbors": 30,
        "min_dist": 0.15,
        "metric": "euclidean",
        "random_state": 42,
        "l2_normalize": False,
        "standardize": False,
    },
    "viz_crop_border": True,
    "compute_effective_rank": True,
    "vicreg_spatial_mode": "dense",
    "ema_momentum_base": 0.996,
    "ema_momentum_final": 1.0,
}

BASE_MODEL = {
    "mode": "pyramid",
    "model_key": "cdd_scaleaware_convnext-pyramid-scaleaware",
    "sigmas": [2, 4, 8, 16],
    "latent_channels": 32,
    "encoder_width": 64,
    "encoder_depth": 4,
    "encoder_kernel_size": 7,
    "scaleaware_feat_channels": 8,
    "scaleaware_adapter_kernel_size": 3,
    "scaleaware_fusion_type": "topdown",
    "scaleaware_norm_per_scale": True,
    "cdd_append_last_residual": True,
    "global_shift": False,
    "align_scales": True,
    "constant_mask_box": False,
    "mask_footprint_px": 0,
    "cdd_mode": "log",
    "cdd_constrained": True,
    "cdd_sm_mode": "reflect",
    "post_log_transform": True,
    "log_eps": 1e-06,
    "cdd_log_std_floor_mult": 0.05,
    "ema_momentum": 0.996,
    "normalize_loss_l2": False,
    "predictor_layernorm": True,
    "mask_scale_factor": 1.0,
    "mask_spacing_scaling": 2.0,
    "target_invalid_region_skip": False,
    "target_sampling_mode": "priority_sampling",
    "priority_top_percent": 15.0,
    "priority_n_target": "auto",
    "priority_min_targets_per_map": 10,
    "priority_dithering_pixels": 6,
    "patch_size": 3,
    "active_target_fraction": 1.0,
}

MASK_SCALES = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]


def _fmt(v: float) -> str:
    return f"{v:.1f}".replace(".", "p")


def write_config(name: str, model_cfg: dict) -> None:
    cfg = {
        "data": deepcopy(BASE_DATA),
        "model": model_cfg,
        "train": deepcopy(BASE_TRAIN),
    }
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"  {name}.json")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    count = 0
    for mscale in MASK_SCALES:
        m = deepcopy(BASE_MODEL)
        m["mask_scale_factor"] = float(mscale)
        name = (
            "gen_52_run_1_mhd_cdd_scaleaware_convnext-pyramid-scaleaware_"
            f"afrac_1p0_mscale_{_fmt(mscale)}_mbox_00_pmin_10"
        )
        write_config(name, m)
        count += 1
    print(f"\nTotal: {count} configs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
