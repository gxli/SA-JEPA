from __future__ import annotations

import pytest
import torch

from src.models.build_jepa3d import PyramidGridJEPA3D
from src.train import _is_3d_jepa_mode, build_model3d_from_config


def test_3d_full_volume_mode_is_removed():
    assert not _is_3d_jepa_mode("3d_full_volume")
    with pytest.raises(ValueError, match="3d_slab"):
        build_model3d_from_config(
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


def test_3d_slab_training_uses_required_depth_and_target_width_latents():
    model = PyramidGridJEPA3D(
        latent_channels=4,
        scale_channels=4,
        num_scales=1,
        encoder_type="convnext_dense3d",
        patch_size=1,
        slab_depth=4,
        num_targets=2,
        encoder_depth=1,
        encoder_kernel_size=3,
        mask_box_size=1,
        num_mask_boxes=1,
        use_grn=False,
    )
    model.train()
    x = torch.rand(1, 1, 12, 24, 24)

    out = model(x)

    assert model.required_input_depth == model.encoder_receptive_field_depth + model.slab_depth - 1
    assert int(out["selected_slab_depth"][0]) == model.slab_depth
    assert out["x_clean_full"].shape[2] == model.required_input_depth
    assert out["x_clean"].shape[2] == model.slab_depth
    assert out["context_map"].shape[-3:] == (model.slab_depth, 24, 24)
    assert out["gt_map"].shape[-3:] == (model.slab_depth, 24, 24)
    assert out["pred_map"].shape[-3:] == (model.slab_depth, 24, 24)
    assert "context_map_3d" not in out
    assert "gt_map_3d" not in out
