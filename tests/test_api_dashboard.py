from __future__ import annotations

import torch
import numpy as np


def test_public_api_generates_dashboard_from_import_route(tmp_path):
    from sajepa import ScaleAwareJEPA

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    torch.save({"context_map": torch.randn(1, 4, 8, 8)}, session_dir / "inference_outputs.pt")

    model = ScaleAwareJEPA()
    model._session_dir = str(session_dir)
    model._is_trained = True

    out_html = tmp_path / "dashboard.html"
    model.generate_dashboard(str(out_html))

    assert out_html.exists()
    assert "<html" in out_html.read_text(encoding="utf-8").lower()


def test_target_region_mask_uses_last_two_location_coordinates_for_3d():
    from src.utils.viz import _target_region_mask_from_outputs

    outputs = {
        "target_locations": torch.tensor([[[4, 0, 7], [2, 7, 2]]], dtype=torch.long),
        "target_valid": torch.tensor([[True, True]]),
        "x_clean": torch.zeros(1, 1, 8, 8),
        "patch_size": 1,
    }

    mask = _target_region_mask_from_outputs(outputs, 8, 8)

    assert mask[0, 7]
    assert mask[7, 2]
    assert not mask[4, 0]
    assert not mask[2, 7]


def test_inference_embeddings_exclude_zero_input_background(tmp_path):
    from src.utils import viz

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    x_clean = torch.zeros(1, 1, 8, 8)
    x_clean[:, :, :4, :4] = 1.0
    outputs = {
        "x_clean": x_clean,
        "x_context": x_clean.clone(),
        "pred_map": torch.randn(1, 4, 4, 4),
        "gt_map": torch.randn(1, 4, 4, 4),
        "context_map": torch.randn(1, 4, 4, 4),
        "target_locations": torch.tensor([[[0, 0], [3, 3]]], dtype=torch.long),
        "target_valid": torch.tensor([[True, True]]),
    }
    old_umap = viz._compute_umap_nd
    viz._compute_umap_nd = lambda x, **_kwargs: x[:, :3].astype("float32")
    try:
        viz.save_inference_dashboard(str(session_dir), outputs, umap_cfg={})
    finally:
        viz._compute_umap_nd = old_umap

    pred_pca = np.load(session_dir / "results" / "predict_pca_xyz.npy")
    finite = np.isfinite(pred_pca.transpose(1, 2, 0).reshape(-1, 3)).all(axis=1)
    assert int(finite.sum()) == 4
    assert not finite.reshape(4, 4)[-1, -1]


def test_dashboard_does_not_compute_umap_by_default(tmp_path):
    from src import dashboard

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    outputs = {
        "x_clean": torch.ones(1, 1, 8, 8),
        "x_context": torch.ones(1, 1, 8, 8),
        "pred_map": torch.randn(1, 4, 4, 4),
        "gt_map": torch.randn(1, 4, 4, 4),
        "context_map": torch.randn(1, 4, 4, 4),
        "target_locations": torch.tensor([[[0, 0], [7, 7]]], dtype=torch.long),
        "target_valid": torch.tensor([[True, True]]),
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")

    old_compute_umap = dashboard._compute_umap_nd
    old_dashboard_compute_umap = dashboard.DASHBOARD_COMPUTE_UMAP
    dashboard.DASHBOARD_COMPUTE_UMAP = False
    dashboard._compute_umap_nd = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("UMAP called"))
    try:
        dash_npz = dashboard.compute_dash_data(str(session_dir), overwrite=True)
    finally:
        dashboard._compute_umap_nd = old_compute_umap
        dashboard.DASHBOARD_COMPUTE_UMAP = old_dashboard_compute_umap

    data = np.load(dash_npz)
    assert np.isfinite(data["pred_pca3d"]).all()
    assert not np.isfinite(data["pred_umap3d"]).any()

    html_path = dashboard.plot_dash_html(str(session_dir), overwrite=True)
    html = html_path and (session_dir / "dashboard.html").read_text(encoding="utf-8")
    assert "Predict PCA RGB" in html
    assert "Predict UMAP RGB" in html
    assert "Predict UMAP 3D Scatter" in html


def test_dashboard_keeps_total_only_loss_history(tmp_path):
    from src import dashboard

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    outputs = {
        "x_clean": torch.ones(1, 1, 8, 8),
        "x_context": torch.ones(1, 1, 8, 8),
        "pred_map": torch.randn(1, 4, 4, 4),
        "gt_map": torch.randn(1, 4, 4, 4),
        "context_map": torch.randn(1, 4, 4, 4),
        "target_locations": torch.tensor([[[0, 0], [7, 7]]], dtype=torch.long),
        "target_valid": torch.tensor([[True, True]]),
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")
    (session_dir / "metrics.csv").write_text(
        "epoch,train_loss\n0,3.5\n1,2.25\n",
        encoding="utf-8",
    )

    dash_npz = dashboard.compute_dash_data(str(session_dir), overwrite=True)
    data = np.load(dash_npz)
    np.testing.assert_allclose(data["loss_x"], np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(data["loss_total"], np.array([3.5, 2.25], dtype=np.float32))
    assert np.isnan(data["loss_prediction"]).all()

    html_path = dashboard.plot_dash_html(str(session_dir), overwrite=True)
    html = (session_dir / "dashboard.html").read_text(encoding="utf-8")
    assert html_path.endswith("dashboard.html")
    assert "Active Loss Terms (Weighted)" in html
    assert "loss_total" in html


def test_dashboard_target_mask_map_does_not_clip_energy_or_visit_panels(tmp_path):
    from src import dashboard

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    target_energy = torch.zeros(1, 1, 8, 8)
    target_energy[0, 0, 7, 7] = 5.0
    target_mask_map = torch.zeros(1, 1, 8, 8)
    target_mask_map[0, 0, 0, 0] = 1.0
    outputs = {
        "x_clean": torch.ones(1, 1, 8, 8),
        "x_context": torch.ones(1, 1, 8, 8),
        "pred_map": torch.randn(1, 4, 8, 8),
        "gt_map": torch.randn(1, 4, 8, 8),
        "context_map": torch.randn(1, 4, 8, 8),
        "target_locations": torch.tensor([[[7, 7]]], dtype=torch.long),
        "target_valid": torch.tensor([[True]]),
        "target_energy_map": target_energy,
        "target_mask_map": target_mask_map,
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")
    np.savez_compressed(session_dir / "visited_target_frequency_canonical.npz", arr=np.eye(8, dtype=np.float32))

    old_umap = dashboard._compute_umap_nd
    dashboard._compute_umap_nd = lambda x, **_kwargs: x[:, :3].astype("float32")
    try:
        dash_npz = dashboard.compute_dash_data(str(session_dir), overwrite=True)
    finally:
        dashboard._compute_umap_nd = old_umap

    data = np.load(dash_npz)
    assert data["energy_map"][7, 7] == 5.0
    assert data["target_loc_heatmap"][7, 7] == 1.0
    assert data["visit_heatmap"][7, 7] == 1.0


def test_inference_target_map_is_centers_not_mask_boxes():
    from src.inference import _target_center_map

    target_locations = torch.tensor([[[3, 4]]], dtype=torch.long)
    target_valid = torch.tensor([[True]])

    target_map = _target_center_map(target_locations, target_valid, (8, 8))

    assert target_map[0, 0, 3, 4] == 1.0
    assert target_map[0, 0, 0, 0] == 0.0
    assert int(target_map.sum().item()) == 9


def test_volumetric_umap_defaults_to_full_volume(tmp_path):
    from src.utils import viz

    x_clean = torch.ones(1, 1, 2, 4, 4)
    x_clean[:, :, :, 0, :] = 0.0
    outputs = {
        "context_map": torch.randn(1, 4, 2, 4, 4),
        "pred_map": torch.randn(1, 4, 2, 4, 4),
        "x_clean": x_clean,
    }
    old_umap = viz._compute_umap_nd
    viz._compute_umap_nd = lambda x, **_kwargs: x[:, :3].astype("float32")
    try:
        meta_path = viz.save_volumetric_umap_embeddings(str(tmp_path), outputs, umap_cfg={})
    finally:
        viz._compute_umap_nd = old_umap

    import json

    meta = json.loads((tmp_path / "volumetric_umap_meta.json").read_text(encoding="utf-8"))
    assert meta_path.endswith("volumetric_umap_meta.json")
    assert meta["n_total_voxels"] == 32
    assert meta["n_valid_voxels"] == 24
    assert meta["n_selected"] == 24
    assert meta["max_points"] == 100000
    assert meta["selection"] == "full_valid_inference_extent"
    assert np.load(tmp_path / "results" / "volumetric_umap_xyz.npy").shape == (24, 3)
