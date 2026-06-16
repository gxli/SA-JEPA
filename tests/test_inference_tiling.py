from __future__ import annotations

import numpy as np
import torch

from scripts.session_to_dash import compute_dash_data
from src.inference_from_session import load_raw_data, save_inference_session


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
