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
    assert str(np.asarray(data["dashboard_model"]).reshape(-1)[0]) == "full"
    assert np.isfinite(data["pred_umap3d"]).all()
    np.testing.assert_allclose(data["pred_umap3d"], data["pred_full_latent3d"])

    html_path = dashboard.plot_dash_html(str(session_dir), overwrite=True)
    html = html_path and (session_dir / "dashboard.html").read_text(encoding="utf-8")
    assert "Predict PCA RGB" in html
    assert "Predict Full Latent RGB" in html
    assert "Predict Full Latent 3D Scatter" in html


def test_umap_dashboard_recomputes_pca_from_same_slice_latents(tmp_path):
    from src import dashboard

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    results_dir = session_dir / "results"
    results_dir.mkdir()
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
    np.save(results_dir / "predict_pca_xyz.npy", np.full((3, 4, 4), 123.0, dtype=np.float32))

    old_external_umap = dashboard._compute_external_umap_nd
    dashboard._compute_external_umap_nd = lambda x, **_kwargs: x[:, :3].astype("float32")
    try:
        dash_npz = dashboard.compute_dash_data(str(session_dir), overwrite=True, model="umap")
    finally:
        dashboard._compute_external_umap_nd = old_external_umap

    with np.load(dash_npz) as data:
        assert str(np.asarray(data["dashboard_model"]).reshape(-1)[0]) == "umap"
        assert data["pred_pca3d"].shape == data["pred_umap3d"].shape == (16, 3)
        assert not np.allclose(data["pred_pca3d"], 123.0, equal_nan=True)
        np.testing.assert_allclose(
            data["pred_full_latent3d"][:, :3],
            outputs["pred_map"][0].permute(1, 2, 0).reshape(-1, 4).numpy()[:, :3],
        )


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


def test_dashboard_omits_blank_rank_cards_when_rank_data_missing(tmp_path):
    from src import dashboard

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    outputs = {
        "x_clean": torch.ones(1, 1, 4, 4),
        "x_context": torch.ones(1, 1, 4, 4),
        "pred_map": torch.randn(1, 4, 4, 4),
        "gt_map": torch.randn(1, 4, 4, 4),
        "context_map": torch.randn(1, 4, 4, 4),
        "target_locations": torch.tensor([[[1, 1]]], dtype=torch.long),
        "target_valid": torch.tensor([[True]]),
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")

    old_umap = dashboard._compute_umap_nd
    dashboard._compute_umap_nd = lambda x, **_kwargs: x[:, :3].astype("float32")
    try:
        dashboard.plot_dash_html(str(session_dir), overwrite=True)
    finally:
        dashboard._compute_umap_nd = old_umap

    html = (session_dir / "dashboard.html").read_text(encoding="utf-8")
    assert "Manifold Diagnostics" in html
    assert "Rank Energy Concentration" in html
    assert "Rank Energy Top-k" in html
    assert 'data-group="eff-rank"' in html
    assert "effective_rank=not computed" not in html


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


def test_dashboard_keeps_full_field_visit_frequency_for_cropped_sample(tmp_path):
    from src import dashboard

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    outputs = {
        "x_clean": torch.ones(1, 1, 4, 4),
        "x_context": torch.ones(1, 1, 4, 4),
        "pred_map": torch.randn(1, 4, 4, 4),
        "gt_map": torch.randn(1, 4, 4, 4),
        "context_map": torch.randn(1, 4, 4, 4),
        "target_locations": torch.tensor([[[1, 1]]], dtype=torch.long),
        "target_valid": torch.tensor([[True]]),
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")

    full_field_visits = np.zeros((8, 8), dtype=np.float32)
    full_field_visits[7, 7] = 3.0
    np.savez_compressed(session_dir / "visited_target_frequency_canonical.npz", arr=full_field_visits)

    old_umap = dashboard._compute_umap_nd
    dashboard._compute_umap_nd = lambda x, **_kwargs: x[:, :3].astype("float32")
    try:
        dash_npz = dashboard.compute_dash_data(str(session_dir), overwrite=True)
    finally:
        dashboard._compute_umap_nd = old_umap

    data = np.load(dash_npz)
    assert data["visit_heatmap"].shape == (8, 8)
    assert data["visit_heatmap"][7, 7] == 3.0


def test_dashboard_target_coverage_takes_priority_over_tile_coverage(tmp_path):
    from src import dashboard

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    outputs = {
        "x_clean": torch.ones(1, 1, 4, 4),
        "x_context": torch.ones(1, 1, 4, 4),
        "pred_map": torch.randn(1, 4, 4, 4),
        "gt_map": torch.randn(1, 4, 4, 4),
        "context_map": torch.randn(1, 4, 4, 4),
        "target_locations": torch.tensor([[[1, 1]]], dtype=torch.long),
        "target_valid": torch.tensor([[True]]),
        "tile_visit_map": torch.full((4, 4), 9.0),
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")
    np.savez_compressed(session_dir / "tile_visit_map.npz", arr=np.full((4, 4), 9.0, dtype=np.float32))
    target_visits = np.zeros((4, 4), dtype=np.float32)
    target_visits[2, 3] = 5.0
    np.savez_compressed(session_dir / "visited_target_frequency.npz", arr=target_visits)

    old_umap = dashboard._compute_umap_nd
    dashboard._compute_umap_nd = lambda x, **_kwargs: x[:, :3].astype("float32")
    try:
        dash_npz = dashboard.compute_dash_data(str(session_dir), overwrite=True)
    finally:
        dashboard._compute_umap_nd = old_umap

    data = np.load(dash_npz)
    assert str(data["visit_heatmap_kind"]) == "Target Coverage Heatmap"
    np.testing.assert_array_equal(data["visit_heatmap"], target_visits)


def test_dashboard_visit_heatmap_respects_configured_zero_invalid_region(tmp_path):
    from src import dashboard

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    x_clean = torch.zeros(1, 1, 4, 4)
    x_clean[:, :, :2, :2] = 1.0
    outputs = {
        "x_clean": x_clean,
        "x_context": x_clean.clone(),
        "pred_map": torch.randn(1, 4, 4, 4),
        "gt_map": torch.randn(1, 4, 4, 4),
        "context_map": torch.randn(1, 4, 4, 4),
        "target_locations": torch.tensor([[[0, 0], [3, 3]]], dtype=torch.long),
        "target_valid": torch.tensor([[True, True]]),
        "target_energy_map": torch.ones(1, 1, 4, 4),
        "target_energy_count_map": torch.ones(1, 1, 4, 4),
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")
    (session_dir / "config_used.json").write_text(
        '{"model":{"target_invalid_region_skip":true,"target_invalid_region_values":[0,"nan"]}}',
        encoding="utf-8",
    )

    dash_npz = dashboard.compute_dash_data(str(session_dir), overwrite=True)
    data = np.load(dash_npz)
    assert data["visit_heatmap"][0, 0] == 1.0
    assert data["visit_heatmap"][3, 3] == 0.0
    assert data["energy_map"][3, 3] == 0.0
    assert data["target"][3, 3] == 0.0


def test_dashboard_does_not_invent_zero_mask_when_config_allows_zero_targets(tmp_path):
    from src import dashboard

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    x_clean = torch.zeros(1, 1, 4, 4)
    outputs = {
        "x_clean": x_clean,
        "x_context": x_clean.clone(),
        "pred_map": torch.randn(1, 4, 4, 4),
        "gt_map": torch.randn(1, 4, 4, 4),
        "context_map": torch.randn(1, 4, 4, 4),
        "target_locations": torch.tensor([[[3, 3]]], dtype=torch.long),
        "target_valid": torch.tensor([[True]]),
        "target_energy_count_map": torch.ones(1, 1, 4, 4),
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")
    (session_dir / "config_used.json").write_text(
        '{"model":{"target_invalid_region_skip":false,"target_invalid_region_values":[0,"nan"]}}',
        encoding="utf-8",
    )

    dash_npz = dashboard.compute_dash_data(str(session_dir), overwrite=True)
    data = np.load(dash_npz)
    assert data["visit_heatmap"][3, 3] == 1.0
    assert data["target"][3, 3] == 1.0


def test_dashboard_ignores_stale_masked_predict_pca_artifact(tmp_path):
    import os
    from src import dashboard

    session_dir = tmp_path / "session"
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True)
    h = w = 4
    pred = torch.randn(1, 4, h, w)
    yy = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(1, 1, h, w)
    xx = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(1, 1, h, w)
    masked = torch.cat([xx, yy, xx + yy, xx - yy], dim=1)
    outputs = {
        "x_clean": torch.ones(1, 1, h, w),
        "x_context": torch.ones(1, 1, h, w),
        "pred_map": pred,
        "masked_pred_map": masked,
        "gt_map": torch.randn(1, 4, h, w),
        "context_map": torch.randn(1, 4, h, w),
        "target_locations": torch.tensor([[[1, 1]]], dtype=torch.long),
        "target_valid": torch.tensor([[True]]),
    }

    stale_pca = np.zeros((3, h, w), dtype=np.float32)
    stale_umap = np.zeros((3, h, w), dtype=np.float32)
    for branch in ("predict", "masked_predict", "target", "context"):
        np.save(results_dir / f"{branch}_spatial_shape.npy", np.asarray([h, w], dtype=np.int64))
        np.save(results_dir / f"{branch}_pca_xyz.npy", stale_pca)
        np.save(results_dir / f"{branch}_umap_xyz.npy", stale_umap)
    old_time = 1_700_000_000
    for path in results_dir.glob("*.npy"):
        os.utime(path, (old_time, old_time))

    torch.save(outputs, session_dir / "inference_outputs.pt")

    old_umap = dashboard._compute_umap_nd
    old_dashboard_compute_umap = dashboard.DASHBOARD_COMPUTE_UMAP
    dashboard.DASHBOARD_COMPUTE_UMAP = False
    dashboard._compute_umap_nd = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("UMAP called"))
    try:
        dash_npz = dashboard.compute_dash_data(str(session_dir), overwrite=True)
    finally:
        dashboard._compute_umap_nd = old_umap
        dashboard.DASHBOARD_COMPUTE_UMAP = old_dashboard_compute_umap

    data = np.load(dash_npz)
    assert np.nanstd(data["masked_pred_pca3d"]) > 0.0
    assert not np.allclose(data["masked_pred_pca3d"], 0.0, equal_nan=True)


def test_dashboard_recomputes_fresh_collapsed_pca_when_latents_have_spread(tmp_path):
    from src import dashboard

    session_dir = tmp_path / "session"
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True)
    h = w = 4
    yy = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(1, 1, h, w)
    xx = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(1, 1, h, w)
    masked = torch.cat([xx, yy, xx + yy, xx - yy], dim=1)
    outputs = {
        "x_clean": torch.ones(1, 1, h, w),
        "x_context": torch.ones(1, 1, h, w),
        "pred_map": torch.randn(1, 4, h, w),
        "masked_pred_map": masked,
        "gt_map": torch.randn(1, 4, h, w),
        "context_map": torch.randn(1, 4, h, w),
        "target_locations": torch.tensor([[[1, 1]]], dtype=torch.long),
        "target_valid": torch.tensor([[True]]),
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")

    zero_pca = np.zeros((3, h, w), dtype=np.float32)
    spread_umap = np.stack(
        [
            np.tile(np.linspace(-1, 1, w, dtype=np.float32), (h, 1)),
            np.tile(np.linspace(-1, 1, h, dtype=np.float32)[:, None], (1, w)),
            np.ones((h, w), dtype=np.float32),
        ],
        axis=0,
    )
    for branch in ("predict", "masked_predict", "target", "context"):
        np.save(results_dir / f"{branch}_spatial_shape.npy", np.asarray([h, w], dtype=np.int64))
        np.save(results_dir / f"{branch}_pca_xyz.npy", zero_pca)
        np.save(results_dir / f"{branch}_umap_xyz.npy", spread_umap)

    old_umap = dashboard._compute_umap_nd
    old_dashboard_compute_umap = dashboard.DASHBOARD_COMPUTE_UMAP
    dashboard.DASHBOARD_COMPUTE_UMAP = False
    dashboard._compute_umap_nd = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("UMAP called"))
    try:
        dash_npz = dashboard.compute_dash_data(str(session_dir), overwrite=True)
    finally:
        dashboard._compute_umap_nd = old_umap
        dashboard.DASHBOARD_COMPUTE_UMAP = old_dashboard_compute_umap

    data = np.load(dash_npz)
    assert np.nanstd(data["masked_pred_pca3d"]) > 0.0
    notes = "\n".join(str(x) for x in data["embedding_health_notes"].reshape(-1))
    assert "masked_pred: PCA collapsed" in notes
    assert data["masked_pred_latent_norm"].shape == (h, w)


def test_dashboard_warns_when_inference_tensors_contain_nan(tmp_path):
    from src import dashboard

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    masked = torch.randn(1, 4, 4, 4)
    masked[0, 0, 0, 0] = float("nan")
    outputs = {
        "x_clean": torch.ones(1, 1, 4, 4),
        "x_context": torch.ones(1, 1, 4, 4),
        "pred_map": torch.randn(1, 4, 4, 4),
        "masked_pred_map": masked,
        "gt_map": torch.randn(1, 4, 4, 4),
        "context_map": torch.randn(1, 4, 4, 4),
        "target_locations": torch.tensor([[[1, 1]]], dtype=torch.long),
        "target_valid": torch.tensor([[True]]),
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")

    old_umap = dashboard._compute_umap_nd
    dashboard._compute_umap_nd = lambda x, **_kwargs: x[:, :3].astype("float32")
    try:
        html_path = dashboard.plot_dash_html(str(session_dir), overwrite=True)
    finally:
        dashboard._compute_umap_nd = old_umap

    html = (session_dir / "dashboard.html").read_text(encoding="utf-8")
    assert html_path.endswith("dashboard.html")
    assert "Numerical health warning" not in html
    assert "masked_pred_map: nonfinite" not in html


def test_dashboard_warns_when_masked_predict_pca_collapses(tmp_path):
    from src import dashboard

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    outputs = {
        "x_clean": torch.ones(1, 1, 4, 4),
        "x_context": torch.ones(1, 1, 4, 4),
        "pred_map": torch.randn(1, 4, 4, 4),
        "masked_pred_map": torch.ones(1, 4, 4, 4),
        "gt_map": torch.randn(1, 4, 4, 4),
        "context_map": torch.randn(1, 4, 4, 4),
        "target_locations": torch.tensor([[[1, 1]]], dtype=torch.long),
        "target_valid": torch.tensor([[True]]),
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")

    old_umap = dashboard._compute_umap_nd
    dashboard._compute_umap_nd = lambda x, **_kwargs: x[:, :3].astype("float32")
    try:
        html_path = dashboard.plot_dash_html(str(session_dir), overwrite=True)
    finally:
        dashboard._compute_umap_nd = old_umap

    html = (session_dir / "dashboard.html").read_text(encoding="utf-8")
    assert html_path.endswith("dashboard.html")
    assert "masked_pred_map: COLLAPSED latent map" not in html
    assert "masked_pred_pca3d: COLLAPSED embedding" not in html


def test_dashboard_places_latent_norm_after_embedding_pairs(tmp_path):
    from src import dashboard

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    outputs = {
        "x_clean": torch.ones(1, 1, 4, 4),
        "x_context": torch.ones(1, 1, 4, 4),
        "pred_map": torch.randn(1, 4, 4, 4),
        "masked_pred_map": torch.randn(1, 4, 4, 4),
        "gt_map": torch.randn(1, 4, 4, 4),
        "context_map": torch.randn(1, 4, 4, 4),
        "target_locations": torch.tensor([[[1, 1]]], dtype=torch.long),
        "target_valid": torch.tensor([[True]]),
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")

    old_umap = dashboard._compute_umap_nd
    dashboard._compute_umap_nd = lambda x, **_kwargs: x[:, :3].astype("float32")
    try:
        dashboard.plot_dash_html(str(session_dir), overwrite=True)
    finally:
        dashboard._compute_umap_nd = old_umap

    html = (session_dir / "dashboard.html").read_text(encoding="utf-8")
    assert html.index('data-group="context-pca"') < html.index('data-group="context-full-latent"')
    assert html.index('data-group="context-full-latent"') < html.index('data-group="masked_pred-pca"')
    assert html.index('data-group="gt-full-latent"') < html.index('data-group="context-latent-norm"')


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
