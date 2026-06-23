#!/usr/bin/env python3
"""Quickstart MHD example — inline config, 10 epochs."""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, ROOT)
from sajepa import ScaleAwareJEPA

DATA = os.path.join(ROOT, "data", "C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy")

model = ScaleAwareJEPA(config={
    "data": {"data_root": os.path.dirname(DATA), "npy_pattern": os.path.basename(DATA),
             "num_samples": 200, "d4_augment": True},
    "model": {"mask_size_scaling": 1.2},
    "train": {"epochs": 10, "symmetry_loss_weight": 0.0,
              "spread_regularizer": {"type": "std_hinge", "target": "context",
              "spatial_mode": "pooled", "weight": 5.0, "target_std": 1.0}},
})

model.train(config_name="quickstart", sessions_dir=os.path.join(ROOT, "sessions"), dashboard=True)
dashboard = os.path.join(model.session_dir, "dashboard.html")
umap_npy = os.path.join(model.session_dir, "results", "predict_umap_xyz.npy")
umap_html = os.path.join(model.session_dir, "results", "interactive_umap_predict.html")
if os.path.exists(umap_npy):
    model.save_interactive_umap(umap_npy, umap_html)
print(f"\nDone.\n  session:          {model.session_dir}\n  dashboard:        {dashboard}\n  interactive_umap: {umap_html}")
