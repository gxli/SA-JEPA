import torch

from src.train import build_model_from_config


def _build(model_cfg):
    cfg = {
        "mode": "pyramid",
        "model_key": "convnext_dense_pyramid",
        "sigmas": [2, 4],
        "latent_channels": 8,
        "encoder_width": 8,
        "encoder_depth": 1,
        "patch_size": 1,
        "target_sampling_mode": "random",
        **model_cfg,
    }
    return build_model_from_config(cfg, data_cfg={}, train_cfg={}, device=torch.device("cpu"))


def test_mask_size_scaling_and_mask_size_accept_inline_ranges():
    model = _build(
        {
            "mask_size_scaling": [0.4, 1.6],
            "mask_size": [3, 15],
        }
    )

    assert model.mask_scale == 1.0
    assert model.mask_scale_range == (0.4, 1.6)
    assert model.mask_box_size == 9
    assert model.mask_box_size_range == (3, 15)


def test_random_mask_box_per_target_keeps_range_for_candidate_sampling():
    model = _build(
        {
            "mask_size_scaling": 0,
            "mask_size": [3, 15],
            "random_mask_box_per_target": True,
        }
    )

    assert model.random_mask_box_per_target is True
    assert model.mask_box_size == 9
    assert model.mask_box_size_range == (3, 15)
    assert model.sample_mask_params(device=torch.device("cpu")) == (0.0, 9)


def test_mask_size_manual_overrides_with_fixed_per_channel_sizes():
    model = _build(
        {
            "mask_size_scaling": [8.0, 10.0],
            "mask_size": [3, 15],
            "mask_size_manual": [5, 9],
        }
    )

    assert model.mask_scale == 9.0
    assert model.mask_scale_range == (8.0, 10.0)
    assert model.mask_box_size == 9
    assert model.mask_box_size_range == (3, 15)
    assert model.manual_mask_box_sizes == (5, 9)


def test_mask_size_manual_accepts_scalar_fixed_size():
    model = _build(
        {
            "mask_size_scaling": 1.0,
            "mask_size": 5,
            "mask_size_manual": 7,
        }
    )

    assert model.mask_scale == 1.0
    assert model.mask_scale_range is None
    assert model.manual_mask_box_sizes == (7,)


def test_legacy_mask_keys_are_ignored_by_config_parser():
    model = _build(
        {
            "mask_scale_factor": 9.0,
            "mask_footprint_px": 5,
        }
    )

    assert model.mask_scale == 1.0
    assert model.mask_scale_range is None
    assert model.mask_box_size == 16
