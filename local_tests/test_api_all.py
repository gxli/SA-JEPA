#!/usr/bin/env python3
"""Smoke-test sajepa API — inline config, zero file reads, includes resume test."""
from __future__ import annotations

import os, sys, copy, time
import numpy as np
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"))

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _PROJ)

from sajepa import ScaleAwareJEPA

SESS = os.path.join(_HERE, "sessions")
DATA_DIR = os.path.join(SESS, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CFG = {
    "data": {
        "data_root": DATA_DIR,
        "npy_pattern": "_input.npy",
        "cdd_precompute": False,
        "num_samples": 2,
    },
    "model": {
        "mode": "pyramid", "model_key": "cdd_scaleaware_convnext",
        "encoder_width": 32, "encoder_depth": 2, "encoder_kernel_size": 5,
        "latent_channels": 16, "scaleaware_feat_channels": 8,
        "predictor_hidden": 32, "scaleaware_fusion_type": "concat",
        "scaleaware_final_norm": True, "scaleaware_stem_norm": True,
        "scaleaware_adapter_norm": True, "normalize_loss_l2": False,
        "use_grn": True, "post_log_transform": True,
        "use_symmetric_feature_loss": False, "cdd_append_last_residual": True,
    },
    "train": {
        "batch_size": 1, "epochs": 1, "num_workers": 0,
        "inference_num_samples": 1,
        "lr": 1e-4, "weight_decay": 1e-5,
        "prediction_loss_weight": 50.0, "symmetry_loss_weight": 0.0,
        "spread_regularizer": {
            "type": "std_hinge", "target": "context",
            "spatial_mode": "pooled", "weight": 5.0,
            "target_std": 1.0, "eps": 0.0001,
        },
    },
}


def ok(s): print(f"  OK   {s}")
def fail(s): print(f"  FAIL {s}")

t0 = time.time()
field = torch.randn(64, 64)
ok(f"field {tuple(field.shape)}")
np.save(os.path.join(DATA_DIR, "_input.npy"), field.numpy().astype(np.float32))

# ── fresh fit (no session dir → temp) ──
model = ScaleAwareJEPA(config=copy.deepcopy(CFG))
ok("init")

try:
    model.fit(field, epochs=1)
    ok("fit (temp)")
except Exception as e:
    fail(f"fit: {e}")

try:
    z = model.extract(field)
    ok(f"extract {tuple(z.shape)}")
except Exception as e:
    fail(f"extract: {e}")

try:
    model.project(field)
    ok("project")
except Exception as e:
    fail(f"project: {e}")

# ── resume (explicit session dir, add more epochs) ──
RESUME_DIR = os.path.join(SESS, "api_resume_test")
os.makedirs(RESUME_DIR, exist_ok=True)
cfg2 = copy.deepcopy(CFG)
cfg2["train"]["epochs"] = 2  # 1 extra epoch on top

model2 = ScaleAwareJEPA(config=cfg2)
try:
    model2.fit(field, epochs=2, session_dir=RESUME_DIR)
    ok("fit (resume session_dir)")
    model2.fit(field, epochs=2, session_dir=RESUME_DIR)
    ok("fit (re-resume, should skip completed epochs)")
except Exception as e:
    fail(f"resume: {e}")

try:
    z2 = model2.extract(field)
    ok(f"extract resume {tuple(z2.shape)}")
except Exception as e:
    fail(f"extract resume: {e}")

# ── dashboard ──
dash = os.path.join(_HERE, "api_test_output", "smoke.html")
os.makedirs(os.path.dirname(dash), exist_ok=True)
try:
    model2.generate_dashboard(dash)
    ok(f"dashboard ({os.path.getsize(dash)//1024} KB)" if os.path.exists(dash) else "dashboard MISSING")
except Exception as e:
    fail(f"dashboard: {e}")

print(f"\n  done ({time.time()-t0:.1f}s)")
