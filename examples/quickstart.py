"""Quickstart example: train sajepa on synthetic data and extract latent embeddings."""

import os
import sys

# Ensure the project root is on the Python path.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Enable MPS fallback for ops not yet supported on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from src.api import ScaleAwareJEPA


def main():
    # 1. Generate synthetic physical field (128×128 log-normal turbulence-like)
    rng = np.random.default_rng(42)
    field = rng.lognormal(mean=0.0, sigma=0.4, size=(128, 128)).astype(np.float32)
    field = torch.from_numpy(field)

    # 2. Create model with minimal config for quick training
    config = {
        "model": {
            "encoder_depth": 4,
            "mask_size_scaling": 1.2,
            "normalize_loss_l2": False,
            "predictor_layernorm": True,
        },
        "train": {
            "epochs": 3,
            "batch_size": 4,
            "target_batch_size": 32,
            "auto_scale_batch_size": "power_of_two",
            "precision": "bf16",
        },
    }

    model = ScaleAwareJEPA(config=config)

    # 3. Train and extract
    latent_atlas = model.fit_and_extract(field)

    print(f"\n{'='*60}")
    print(f"Training complete")
    print(f"{'='*60}")
    print(f"Latent atlas shape:  {tuple(latent_atlas.shape)}")
    print(f"  channels={latent_atlas.shape[0]}, spatial={latent_atlas.shape[1]}×{latent_atlas.shape[2]}")
    print(f"  dtype={latent_atlas.dtype}, device={latent_atlas.device}")
    print(f"  value range: [{float(latent_atlas.min()):.4f}, {float(latent_atlas.max()):.4f}]  mean={float(latent_atlas.mean()):.4f}")
    std_per_ch = latent_atlas.float().std(dim=(1, 2))
    dead = (std_per_ch < 1e-5).sum().item()
    print(f"  per-channel std:  mean={float(std_per_ch.mean()):.4f}  min={float(std_per_ch.min()):.4f}  max={float(std_per_ch.max()):.4f}")
    print(f"  dead channels (std<1e-5): {dead}/{latent_atlas.shape[0]}")
    print(f"{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
