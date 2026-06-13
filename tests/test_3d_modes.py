from __future__ import annotations

import tempfile

import numpy as np
import torch

from src.inference import run_full_volume_inference_3d
from src.train import _is_3d_jepa_mode, build_model3d_from_config


def test_3d_full_volume_mode_builds():
    model = build_model3d_from_config(
        {
            "mode": "3d_full_volume",
            "encoder_type": "convnext_dense3d",
            "latent_channels": 4,
            "scale_channels": 4,
            "encoder_depth": 1,
            "encoder_kernel_size": 3,
            "patch_size": 1,
            "slab_depth": 3,
            "num_targets": 2,
        },
        train_cfg={},
        device=torch.device("cpu"),
    )

    assert _is_3d_jepa_mode("3d_full_volume")
    assert model.mode == "3d_full_volume"
    assert model.full_volume_training is True


class _IdentityContext(torch.nn.Module):
    def forward(self, slab, mask_tokens=None):
        return slab


class _FakeSlabModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.context_encoder = _IdentityContext()
        self.full_volume_training = False
        self.slab_depth = 2
        self._param = torch.nn.Parameter(torch.zeros(()))


def test_full_volume_inference_writes_covered_target_regions_without_gaps():
    model = _FakeSlabModel()
    volume = np.arange(6, dtype=np.float32).reshape(1, 6, 1, 1)

    with tempfile.TemporaryDirectory() as tmp:
        run_full_volume_inference_3d(
            model=model,
            cdd_cache={("/tmp/cube.npy", None): volume},
            session_dir=tmp,
            config_name="test",
            device=torch.device("cpu"),
            slab_depth=4,
            overlap=0,
            post_log_transform=False,
        )
        saved = np.load(f"{tmp}/cube_context_map_3d.npz")["arr"]

    assert saved.shape == volume.shape
    np.testing.assert_array_equal(saved, volume)
