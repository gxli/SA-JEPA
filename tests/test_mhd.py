"""Train sajepa on the C12 MHD test data using gen_139 ms=1.2 config defaults."""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from src.api import ScaleAwareJEPA
from src.utils.npy import _safe_load_npy

# Load MHD data
arr = _safe_load_npy("data/C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy")
field = torch.from_numpy(arr.astype(np.float32))

# Use the canonical base config.
model = ScaleAwareJEPA(config="configs/base_pyramid_scaleaware_convnext.yaml")

# Train, extract, save
out_dir = "examples/output"
os.makedirs(out_dir, exist_ok=True)
model.fit(field, epochs=10)
latent = model.extract(field)
model.save_session(os.path.join(out_dir, "session_mhd"))
model.generate_dashboard()

# Save latent
np.save(os.path.join(out_dir, "latent_mhd.npy"), latent.cpu().numpy())
print(f"\nSaved: examples/output/latent_mhd.npy  shape={tuple(latent.shape)}")
print(f"Dashboard: examples/output/session_mhd/dashboard.html")
