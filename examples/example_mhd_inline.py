#!/usr/bin/env python3
"""Fully annotated MHD inline config — all knobs explained."""
from __future__ import annotations

import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sajepa import ScaleAwareJEPA

config = {
    "data": {
        "data_root": os.path.join(ROOT, "data"),
        "npy_pattern": "C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy",
        "num_samples": 200,
        "d4_augment": True,
    },
    "model": {
        "mask_size_scaling": 1.2,
        # "sigmas": [2, 4, 8, 16],
        # "convnext_layer_dilations": [1,1,2,4],
        # "mask_box_hardcap": 48,
        "normalize_loss_l2": False,
        # "use_symmetric_feature_loss": False,
    },
    "train": {
        "epochs": 10,
        "batch_size": 4,
        # "gradient_accumulation_steps": 2,
        "symmetry_loss_weight": 0.0,
        "spread_regularizer": {
            "type": "std_hinge",
            "target": "context",
            "spatial_mode": "pooled",
            "weight": 5.0,
            "target_std": 1.0,
        },
        # "prediction_loss_weight": 50,
        # "symmetry_loss_weight": 0.003,      # optional; costs extra memory
    },
}

model = ScaleAwareJEPA(config=config)
model.train(config_name="mhd_inline_annotated",
            sessions_dir=os.path.join(ROOT, "sessions"),
            dashboard=True)
dashboard = os.path.join(model.session_dir, "dashboard.html")
umap_npy = os.path.join(model.session_dir, "results", "predict_umap_xyz.npy")
umap_html = os.path.join(model.session_dir, "results", "interactive_umap_predict.html")
if os.path.exists(umap_npy):
    model.save_interactive_umap(umap_npy, umap_html)
print(f"\nDone.\n  session:          {model.session_dir}\n  dashboard:        {dashboard}\n  interactive_umap: {umap_html}")
