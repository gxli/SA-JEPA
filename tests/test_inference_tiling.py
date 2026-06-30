from __future__ import annotations

import numpy as np
import torch

from scripts.session_to_dash import compute_dash_data
from src.inference_from_session import load_raw_data, save_inference_session
import src.inference as inference_mod
from src.inference import _run_tiled_dense_inference_2d, run_post_training_inference


def test_tiled_inference_keeps_only_more_than_half_valid_cutouts(tmp_path):
    arr = np.zeros((8, 8), dtype=np.float32)
    arr[:4, :4] = 1.0
    arr[4:, 4:] = 2.0
    path = tmp_path / "field.npy"
    np.save(path, arr)

    tensor, layout = load_raw_data(
        str(path),
        crop_size=4,
        crop_mode="tile",
        crop_min_valid_fraction=0.5,
        return_layout=True,
    )

    assert tuple(tensor.shape) == (2, 1, 4, 4)
    assert layout is not None
    assert layout.origins == ((0, 0), (4, 4))
    assert layout.visit_map is not None
    assert int(layout.visit_map[:4, :4].min()) == 1
    assert int(layout.visit_map[4:, 4:].min()) == 1
    assert int(layout.visit_map[:4, 4:].max()) == 0
    assert int(layout.visit_map[4:, :4].max()) == 0


def test_tiled_inference_errors_when_no_tile_is_valid(tmp_path):
    arr = np.zeros((8, 8), dtype=np.float32)
    path = tmp_path / "empty.npy"
    np.save(path, arr)

    try:
        load_raw_data(
            str(path),
            crop_size=4,
            crop_mode="tile",
            crop_min_valid_fraction=0.5,
            return_layout=True,
        )
    except ValueError as exc:
        assert "No inference tiles passed" in str(exc)
    else:
        raise AssertionError("Expected invalid tiled input to raise")


def test_inference_session_saves_tile_visit_heatmap_for_dashboard(tmp_path):
    arr = np.zeros((8, 8), dtype=np.float32)
    arr[:4, :4] = 1.0
    arr[4:, 4:] = 2.0
    path = tmp_path / "field.npy"
    np.save(path, arr)
    _tensor, layout = load_raw_data(
        str(path),
        crop_size=4,
        crop_mode="tile",
        crop_min_valid_fraction=0.5,
        return_layout=True,
    )
    assert layout is not None

    outputs = {
        "x_clean": torch.ones((1, 1, 8, 8), dtype=torch.float32),
        "x_context": torch.ones((1, 1, 8, 8), dtype=torch.float32),
        "target_locations": torch.zeros((1, 1, 2), dtype=torch.long),
        "target_valid": torch.ones((1, 1), dtype=torch.bool),
        "pred_map": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
        "gt_map": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
        "context_map": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
    }

    session_dir = tmp_path / "inference"
    save_inference_session(
        outputs,
        str(session_dir),
        {"data": {}, "model": {}, "train": {}},
        str(path),
        crop_size=4,
        crop_min_valid_fraction=0.5,
        make_dashboard=False,
        tile_layout=layout,
    )
    dash_path = compute_dash_data(str(session_dir), overwrite=True)

    visit = np.load(session_dir / "tile_visit_map.npy")
    assert visit.shape == (8, 8)
    with np.load(dash_path) as data:
        assert str(data["visit_heatmap_kind"]) == "Tile Coverage Heatmap"
        np.testing.assert_array_equal(data["visit_heatmap"], visit.astype(np.float32))


def test_post_training_inference_exports_clean_and_masked_predictions_separately(tmp_path):
    class FakeModel(torch.nn.Module):
        patch_size = 1
        mask_fraction = 0.25
        priority_n_target = 1
        priority_min_targets_per_map = 0
        target_invalid_region_skip = False
        target_invalid_region_values = (0.0, "nan")

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.mask_inference_calls = []

        def forward(
            self,
            x,
            return_debug=False,
            enable_grid_jitter=False,
            enable_target_dithering=False,
            lattice_shift_override=(0, 0),
            mask_inference=True,
            cdd_orig=None,
        ):
            self.mask_inference_calls.append(bool(mask_inference))
            pred_value = 1.0 if mask_inference else 2.0
            pred = torch.full((x.shape[0], 3, 4, 4), pred_value, device=x.device)
            gt = torch.full_like(pred, 3.0)
            out = {
                "x_clean": x,
                "x_context": x,
                "x_clean_raw": x,
                "x_context_raw": x,
                "target_locations": torch.zeros((x.shape[0], 1, 2), dtype=torch.long, device=x.device),
                "target_scales": torch.ones((x.shape[0], 1), dtype=torch.long, device=x.device),
                "target_valid": torch.ones((x.shape[0], 1), dtype=torch.bool, device=x.device),
                "pred_map": pred,
                "gt_map": gt,
                "context_map": pred + 4.0,
                "pred_patches": pred[:, :, :1, :1].unsqueeze(1),
                "gt_patches": gt[:, :, :1, :1].unsqueeze(1),
            }
            if return_debug:
                out["target_mask_map"] = torch.ones((x.shape[0], 1, x.shape[-2], x.shape[-1]), device=x.device)
            return out

    def energy_fn(outputs, normalize=False):
        return torch.tensor(0.0)

    def energy_map_fn(outputs, image_size):
        b = outputs["pred_map"].shape[0]
        h, w = image_size
        zeros = torch.zeros((b, 1, h, w), dtype=outputs["pred_map"].dtype)
        return {
            "energy_rel_sym": zeros,
            "energy_raw": zeros,
            "energy_rel_gt": zeros,
            "energy_cosine": zeros,
        }

    session_dir = tmp_path / "post_training"
    session_dir.mkdir()
    x = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    model = FakeModel()
    run_post_training_inference(
        model=model,
        dataloader=[x],
        session_dir=str(session_dir),
        config_name="fake",
        visit_counts=None,
        force_recompute_inference=True,
        inference_mask_passes=1,
        mask_inference=True,
        inference_mask_border=False,
        compute_jepa_energy_fn=energy_fn,
        compute_target_energy_map_fn=energy_map_fn,
        tile_size=None,
    )

    saved = torch.load(session_dir / "inference_outputs.pt", map_location="cpu")
    assert model.mask_inference_calls == [True, False]
    assert torch.all(saved["pred_map"] == 2.0)
    assert torch.all(saved["masked_pred_map"] == 1.0)


def test_tiled_cdd_masked_inference_uses_shared_encoder_wrapper(monkeypatch):
    class TokenAwareEncoder(torch.nn.Module):
        def forward(self, x, mask_tokens=None):
            if mask_tokens is None:
                return x
            return x + 10.0 * mask_tokens

    class FakeScaleAware(torch.nn.Module):
        encoder_type = "cdd_scaleaware_convnext"
        post_log_transform = False
        scaleaware_norm_per_scale = False
        predictor_spatial_conv = False
        encoder_depth = 0
        encoder_kernel_size = 1
        sigmas = (1.0, 2.0)
        mask_fraction = 0.25
        spacing_scale = 2.0
        global_shift = False
        align_scales = True
        mask_box_size_range = None
        manual_mask_box_sizes = None
        cdd_mode = "log"
        cdd_constrained = True
        cdd_sm_mode = "reflect"
        cdd_append_last_residual = True
        cdd_pre_log_transform = False
        patch_size = 1
        target_invalid_region_skip = False
        target_invalid_region_values = (0.0, "nan")
        target_sampling_mode = "random"
        priority_top_percent = 100
        priority_n_target = 1
        priority_min_targets_per_map = 0
        priority_dithering_pixels = 0
        priority_candidate_oversample = 0
        target_nonoverlap = False
        target_allow_partial_overlap = 0.0
        mask_box_hardcap = None

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.context_encoder = TokenAwareEncoder()
            self.target_encoder = TokenAwareEncoder()
            self.projector = torch.nn.Identity()
            self.predictor = torch.nn.Identity()
            self.target_projector = torch.nn.Identity()

        def sample_mask_params(self, device):
            return torch.tensor(1.0, device=device), torch.tensor(1, device=device)

    x_raw = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    cdd_raw = torch.full((1, 2, 4, 4), 2.0, dtype=torch.float32)
    cdd_masked = torch.full((1, 2, 4, 4), 3.0, dtype=torch.float32)
    mask_token = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    mask_token[:, 0] = 0.25
    mask_token[:, 1] = 0.5

    def fail_prepare_context_batch(**_kwargs):
        raise AssertionError("tiled encoder should use already-built full-frame mask tensors")

    monkeypatch.setattr(inference_mod, "prepare_context_batch", fail_prepare_context_batch)

    out = _run_tiled_dense_inference_2d(
        model=FakeScaleAware(),
        x_raw=x_raw,
        cdd_raw=cdd_raw,
        cdd_masked_raw=cdd_masked,
        mask_token_raw=mask_token,
        tile_size=4,
        tile_overlap=0,
        config_name="fake",
        mask_inference=True,
    )

    expected_masked = cdd_masked + 10.0 * mask_token
    np.testing.assert_allclose(out["pred_map"].numpy(), cdd_raw.numpy())
    np.testing.assert_allclose(out["masked_pred_map"].numpy(), expected_masked.numpy())
