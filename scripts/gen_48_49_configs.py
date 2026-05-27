#!/usr/bin/env python3
"""Generate gen_48 and gen_49 sweep configs.

gen_48: cdd_scaleaware_convnext-pyramid-scaleaware
gen_49: rescnn-pyramid-scaleaware

For each generation:
- Run 1: sweep mask_fraction in [0.2, 0.4, ..., 2.0] with mask_box_size=0
- Run 2: fixed mask box sizes [5, 9, 13]
"""
from __future__ import annotations

import json
import os
from copy import deepcopy

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

BASE_DATA = {
    "data_root": "data",
    "npy_pattern": "C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy",
    "num_samples": 200,
    "image_size": 256,
    "log_transform": True,
    "log_eps": 1e-06,
    "cdd_scales": [2, 4, 8, 16],
    "cdd_strength": 1,
    "cdd_clip": True,
    "norm_before_cdd": True,
    "cdd_mode": "log",
    "cdd_constrained": True,
    "cdd_sm_mode": "reflect",
    "cube_slice_strategy": "center",
    "cube_slice_axis": 0,
    "cube_slice_index": 0,
    "random_roll_max": 0,
    "cache_cdd": True,
    "cache_random_slices": True,
    "precompute_cdd_cache_all_slices": True,
    "d4_augment": True,
}

BASE_TRAIN = {
    "batch_size": 8,
    "epochs": 10,
    "lr": 0.0001,
    "weight_decay": 1e-05,
    "num_workers": 0,
    "log_interval": 1,
    "jepa_loss_weight": 100.0,
    "vicreg_var_weight": 0.0,
    "vicreg_cov_weight": 0.0,
    "sigreg_weight": 1.0,
    "sigreg_sketch_dim": 64,
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

BASE_MODEL_COMMON = {
    "mode": "pyramid",
    "blur_mode": "cdd",
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
    "mask_box_size": 0,
    "cdd_mode": "log",
    "cdd_constrained": True,
    "cdd_sm_mode": "reflect",
    "post_log_transform": True,
    "log_eps": 1e-06,
    "cdd_log_std_floor_mult": 0.05,
    "ema_momentum": 0.996,
    "normalize_loss_l2": False,
    "predictor_layernorm": True,
    "mask_size_scaling": 1.0,
    "mask_spacing_scaling": 2.0,
    "target_invalid_region_skip": False,
    "target_sampling_mode": "priority_sampling",
    "priority_top_percent": 15.0,
    "priority_n_target": 20,
    "priority_dithering_pixels": 6,
    "patch_size": 3,
}

MASK_FRACTIONS = [round(0.2 * i, 1) for i in range(1, 11)]  # 0.2..2.0
FIXED_BOX_SIZES = [5, 9, 13, 15, 17]


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


def make_base_model(model_key: str) -> dict:
    m = deepcopy(BASE_MODEL_COMMON)
    m["model_key"] = model_key
    return m


def generate_gen(gen_id: int, model_key: str) -> int:
    count = 0

    # Run 1: mask_fraction sweep with mask_box_size=0
    for mfrac in MASK_FRACTIONS:
        m = make_base_model(model_key)
        m["mask_fraction"] = float(mfrac)
        m["mask_box_size"] = 0
        m["constant_mask_box"] = False
        m["mask_size_scaling"] = 1.0
        name = (
            f"gen_{gen_id}_run_1_{model_key}_"
            f"mfrac_{_fmt(mfrac)}_mbox_00"
        )
        write_config(name, m)
        count += 1

    # Run 2: fixed box size sweep
    for mbox in FIXED_BOX_SIZES:
        m = make_base_model(model_key)
        m["mask_fraction"] = 0.0
        m["mask_box_size"] = int(mbox)
        m["constant_mask_box"] = True
        m["mask_size_scaling"] = 0.0
        name = (
            f"gen_{gen_id}_run_2_{model_key}_"
            f"mfrac_0p0_mbox_{mbox:02d}"
        )
        write_config(name, m)
        count += 1

    return count


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    total += generate_gen(48, "cdd_scaleaware_convnext-pyramid-scaleaware")
    total += generate_gen(49, "rescnn-pyramid-scaleaware")
    print(f"\nTotal: {total} configs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
