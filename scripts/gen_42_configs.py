#!/usr/bin/env python3
"""Generate run_42_g1 configs (12 total)."""
from __future__ import annotations

import json
import os

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

PYRAMID_MODEL_BASE = {
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
    "target_sampling_mode": "priority",
    "priority_top_percent": 15.0,
    "priority_n_target": 20,
    "priority_dithering_pixels": 6,
    "patch_size": 3,
}

IMAGE_MODEL_BASE = {
    "mode": "image",
    "model_key": "convnext_image_dense_masked",
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
    "constant_mask_box": True,
    "cdd_mode": "log",
    "cdd_constrained": True,
    "cdd_sm_mode": "reflect",
    "post_log_transform": True,
    "log_eps": 1e-06,
    "cdd_log_std_floor_mult": 0.05,
    "ema_momentum": 0.996,
    "normalize_loss_l2": False,
    "predictor_layernorm": True,
    "mask_scale_factor": 0.0,
    "mask_spacing_scaling": 2.0,
    "target_invalid_region_skip": False,
    "target_sampling_mode": "priority",
    "priority_top_percent": 15.0,
    "priority_n_target": 20,
    "priority_dithering_pixels": 6,
    "patch_size": 3,
    "mask_fraction": 0.0,
}


def _fmt(v):
    if isinstance(v, float):
        return str(v).replace(".", "p")
    return str(v)


def write_config(name, model_cfg):
    cfg = {
        "data": dict(BASE_DATA),
        "model": model_cfg,
        "train": dict(BASE_TRAIN),
    }
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"  {name}.json")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Run 1: pyramid scaleaware, vary mask_fraction ──
    for mfrac in [0.4, 0.8, 1.2, 1.6]:
        m = dict(PYRAMID_MODEL_BASE)
        m["mask_fraction"] = mfrac
        m["mask_scale_factor"] = 1.0
        m["mask_footprint_px"] = 0
        m.pop("constant_mask_box", None)
        name = f"gen_42_run_1_cdd_scaleaware_convnext-pyramid-scaleaware_mfrac_{_fmt(mfrac)}_mbox_00"
        write_config(name, m)

    # ── Run 2: pyramid scaleaware, mask pixels (box_size), scaling=0 ──
    for mbox in [5, 7, 9, 11]:
        m = dict(PYRAMID_MODEL_BASE)
        m["mask_fraction"] = 0.0
        m["mask_scale_factor"] = 0.0
        m["mask_footprint_px"] = mbox
        m["constant_mask_box"] = True
        name = f"gen_42_run_2_cdd_scaleaware_convnext-pyramid-scaleaware_mfrac_0p0_mbox_{mbox:02d}"
        write_config(name, m)

    # ── Run 3: image convnext masked, mask pixels (box_size) ──
    for mbox in [5, 7, 9, 11]:
        m = dict(IMAGE_MODEL_BASE)
        m["mask_footprint_px"] = mbox
        name = f"gen_42_run_3_convnext_image_dense_masked_mfrac_0p0_mbox_{mbox:02d}"
        write_config(name, m)

    print(f"\nTotal: 12 configs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
