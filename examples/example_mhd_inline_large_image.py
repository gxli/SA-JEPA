#!/usr/bin/env python3
"""MHD-style inline example on the Gulf of Mexico large image."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sajepa import ScaleAwareJEPA

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("SAJEPA_ENABLE_CPU_UMAP", "1")
os.environ.setdefault("SAJEPA_ENABLE_TORCHDR_UMAP", "0")
os.environ.setdefault("SAJEPA_STRICT_UMAP", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

config = {
    "data": {
        "data_root": os.path.join(ROOT, "data", "local_data"),
        "npy_pattern": "mexico_temp.npy",
        "input_type": "image",
        "num_samples": 500,
        "log_eps": 1e-6,
        "cdd_mode": "log",
        "cdd_constrained": True,
        "cdd_sm_mode": "reflect",
        "cdd_precompute": True,
        "crop_mode": "random",
        "crop_size": 256,
        "crop_min_valid_fraction": 0.8,
        "native_invalid_border_rejection": True,
        "d4_augment": True,
    },
    "model": {
        "mode": "pyramid",
        "model_key": "cdd_scaleaware_convnext",
        "encoder_width": 64,
        "encoder_depth": 4,
        "encoder_kernel_size": 7,
        "latent_channels": 32,
        "scaleaware_feat_channels": 8,
        "scaleaware_adapter_kernel_size": 3,
        "scaleaware_fusion_type": "topdown",
        "scaleaware_norm_per_scale": True,
        "scaleaware_final_norm": True,
        "scaleaware_stem_norm": True,
        "scaleaware_adapter_norm": True,
        "normalize_loss_l2": False,
        "predictor_layernorm": True,
        "predictor_spatial_conv": True,
        "predictor_hidden": 96,
        "predictor_residual": False,
        "use_grn": True,
        "post_log_transform": False,
        "use_symmetric_feature_loss": False,
        "ema_momentum": 0.996,
        "cdd_mode": "log",
        "cdd_constrained": True,
        "cdd_sm_mode": "reflect",
        "cdd_log_std_floor_mult": 0.05,
        "cdd_append_last_residual": True,
        "sigmas": [2, 4, 8, 16, 32],
        "align_scales": True,
        "patch_size": 3,
        "mask_spacing_scaling": 2.0,
        "mask_size_scaling": 1.2,
        "mask_size": 0,
        "target_sampling_mode": "random",
        "active_target_fraction": 1.0,
        "target_nonoverlap": True,
        "target_allow_partial_overlap": 0.0,
        "target_invalid_region_skip": True,
        "target_invalid_region_values": ["nan"],
        "priority_top_percent": 100,
        "priority_n_target": "auto",
        "priority_min_targets_per_map": 10,
        "priority_dithering_pixels": 6,
        "priority_candidate_oversample": 0,
        "inference_mask_border": True,
        "mask_box_hardcap": 48,
        "nan_border_sigma_multiplier": 1.25,
        "invalid_support_border_mode": "encoder_width_half_plus_one",
    },
    "train": {
        "epochs": 10,
        "batch_size": 4,
        "gradient_accumulation_mode": "step",
        "gradient_accumulation_steps": 1,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "num_workers": 8,
        "diagnostic_interval": 1,
        "full_visit_map": True,
        "symmetry_loss_weight": 0.0,
        "ema_momentum_base": 0.99,
        "ema_momentum_final": 0.9999,
        "prediction_loss_weight": 50,
        "vicreg_spatial_mode": "pooled",
        "scale_probe_enabled": True,
        "inference_tta_enabled": False,
        "compute_effective_rank": True,
        "post_training_artifacts": True,
        "force_recompute_inference": True,
        "mask_predict_mode": "composite",
        "mask_predict_stride": 1,
        "mask_predict_chunk_size": 128,
        "inference_tile_size": 512,
        "inference_tile_overlap": 128,
        "spread_regularizer": {
            "type": "std_hinge",
            "target": "context",
            "spatial_mode": "pooled",
            "weight": 5.0,
            "target_std": 1.0,
            "eps": 1e-4,
        },
        "umap": {
            "metric": "euclidean",
            "standardize": True,
            "l2_normalize": False,
            "n_neighbors": 50,
            "min_dist": 0.2,
            "save_umap_weights": True,
            "reuse_umap_weights": True,
            "fit_max_tokens": 4096,
            "transform_batch": 4096,
        },
    },
}

model = ScaleAwareJEPA(config=config)
model.train(
    config_name="example_mhd_inline_large_image",
    sessions_dir=os.path.join(ROOT, "sessions"),
    dashboard=True,
)

dashboard = os.path.join(model.session_dir, "dashboard.html")
results_dir = os.path.join(model.session_dir, "results")
umap_npy = os.path.join(results_dir, "predict_umap_xyz.npy")
umap_html = os.path.join(results_dir, "interactive_umap_predict.html")
interactive = None
if os.path.exists(umap_npy):
    interactive = model.save_interactive_umap(umap_npy, umap_html)

print(
    "\nDone."
    f"\n  session:          {model.session_dir}"
    f"\n  dashboard:        {dashboard}"
    f"\n  interactive_umap: {interactive or umap_html}"
)
