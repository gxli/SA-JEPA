import numpy as np
import torch

from scripts.legacy.check_session_integrity import check_session
from scripts.legacy.migrate_embedding_artifacts import migrate_results
from scripts.session_to_dash import compute_dash_data, plot_dash_html
from src.models.masking import (
    ALLOWED_TARGET_SAMPLING_MODES,
    _build_priority_catalogue_from_cdd_ratio,
    _fractional_spatial_target_budget,
    make_pyramid_grid_context,
    normalize_target_sampling_mode,
)
from src.utils.viz import save_inference_dashboard


def test_fractional_spatial_target_budget_matches_area_ratio_for_integer_case():
    budget = _fractional_spatial_target_budget(
        height=64,
        width=64,
        box_size=16,
        oversample=3.0,
        device=torch.device("cpu"),
    )

    assert budget == 48


def test_fractional_spatial_target_budget_scales_with_allowed_overlap():
    budget = _fractional_spatial_target_budget(
        height=64,
        width=64,
        box_size=16,
        oversample=3.0,
        device=torch.device("cpu"),
        overlap_fraction=0.5,
    )

    assert budget == 96


def test_priority_catalogue_is_sorted_by_ratio_descending():
    cdd = np.ones((3, 5, 5), dtype=np.float32)
    cdd[2] = 20.0
    cdd[0, 1, 1] = 100.0
    cdd[0, 2, 2] = 50.0
    cdd[0, 3, 3] = 25.0

    catalogue = _build_priority_catalogue_from_cdd_ratio(
        cdd_orig=cdd,
        top_percent=25.0,
        patch_size=1,
        h=5,
        w=5,
    )

    assert catalogue[:3] == [(1, 1), (2, 2), (3, 3)]


def test_priority_sampling_prescreens_to_fractional_candidate_budget():
    torch.manual_seed(0)
    x_clean = torch.ones((1, 1, 64, 64), dtype=torch.float32)
    cdd_orig = torch.ones((1, 3, 64, 64), dtype=torch.float32)

    result = make_pyramid_grid_context(
        x_clean=x_clean,
        sigmas=(1,),
        mask_fraction=1.0,
        mask_scale=0.0,
        spacing_scale=1.0,
        global_shift=False,
        align_scales=True,
        mask_box_size=15,
        inner_target_size=1,
        return_debug=True,
        enable_grid_jitter=False,
        enable_target_dithering=False,
        target_sampling_mode="priority",
        priority_top_percent=100.0,
        priority_n_target="auto",
        priority_candidate_oversample=3.0,
        cdd_orig_in=cdd_orig,
    )

    debug = result[-1]
    assert debug["priority_good_candidates"].item() > 1000
    assert debug["priority_prescreen_candidates"].item() <= 55


def test_random_sampling_uses_full_valid_candidate_pool_not_priority_prescreen():
    torch.manual_seed(0)
    x_clean = torch.ones((1, 1, 32, 32), dtype=torch.float32)
    cdd_orig = torch.ones((1, 3, 32, 32), dtype=torch.float32)
    cdd_orig[:, 0, :4, :4] = 100.0

    result = make_pyramid_grid_context(
        x_clean=x_clean,
        sigmas=(1,),
        mask_fraction=1.0,
        mask_scale=0.0,
        spacing_scale=1.0,
        global_shift=False,
        align_scales=True,
        mask_box_size=5,
        inner_target_size=1,
        return_debug=True,
        enable_grid_jitter=False,
        enable_target_dithering=False,
        target_sampling_mode="random",
        priority_top_percent=1.0,
        priority_n_target=32,
        priority_candidate_oversample=1.0,
        cdd_orig_in=cdd_orig,
    )

    _x_context, target_locations, _target_scales, target_valid, debug = result
    valid_locs = target_locations[0][target_valid[0]]
    assert debug["priority_good_candidates"].item() > 500
    assert debug["priority_prescreen_candidates"].item() == debug["priority_good_candidates"].item()
    assert 0 < valid_locs.shape[0] <= 32


def test_allowed_target_sampling_modes_are_explicit():
    assert ALLOWED_TARGET_SAMPLING_MODES == ("random", "priority", "priority_small_scale", "lattice")
    assert normalize_target_sampling_mode("priority_sampling") == "priority"
    assert normalize_target_sampling_mode("grid") == "lattice"
    for removed_mode in ("uniform", "random_uniform", "monte_carlo"):
        try:
            normalize_target_sampling_mode(removed_mode)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{removed_mode} should not be an allowed target sampling mode")


def test_sampled_modes_reject_zero_and_nan_input_before_candidate_count():
    torch.manual_seed(0)
    x_clean = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    x_clean[:, :, 8:24, 8:24] = 1.0
    x_clean[:, :, 0, :] = float("nan")
    cdd_orig = torch.ones((1, 3, 32, 32), dtype=torch.float32)

    result = make_pyramid_grid_context(
        x_clean=x_clean,
        sigmas=(1,),
        mask_fraction=1.0,
        mask_scale=0.0,
        spacing_scale=1.0,
        global_shift=False,
        align_scales=True,
        mask_box_size=5,
        inner_target_size=1,
        return_debug=True,
        enable_grid_jitter=False,
        enable_target_dithering=False,
        target_sampling_mode="random",
        priority_n_target=32,
        cdd_orig_in=cdd_orig,
    )

    _x_context, target_locations, _target_scales, target_valid, debug = result
    valid_locs = target_locations[0][target_valid[0]]
    assert debug["priority_good_candidates"].item() == 256
    assert 0 < valid_locs.shape[0] <= 32
    assert torch.all((valid_locs[:, 0] >= 8) & (valid_locs[:, 0] < 24))
    assert torch.all((valid_locs[:, 1] >= 8) & (valid_locs[:, 1] < 24))


def test_sampled_modes_reject_overlapping_pyramids_by_default():
    torch.manual_seed(4)
    x_clean = torch.ones((1, 1, 32, 32), dtype=torch.float32)
    cdd_orig = torch.ones((1, 3, 32, 32), dtype=torch.float32)

    result = make_pyramid_grid_context(
        x_clean=x_clean,
        sigmas=(1,),
        mask_fraction=1.0,
        mask_scale=0.0,
        spacing_scale=1.0,
        global_shift=False,
        align_scales=True,
        mask_box_size=7,
        inner_target_size=1,
        return_debug=True,
        enable_grid_jitter=False,
        enable_target_dithering=False,
        target_sampling_mode="random",
        priority_n_target=64,
        cdd_orig_in=cdd_orig,
    )

    _x_context, target_locations, _target_scales, target_valid, _debug = result
    valid_locs = target_locations[0][target_valid[0]]
    occ = torch.zeros((32, 32), dtype=torch.bool)
    half = 7 // 2
    for cy_t, cx_t in valid_locs:
        cy = int(cy_t.item())
        cx = int(cx_t.item())
        y0 = max(0, cy - half)
        y1 = min(32, cy + 7 - half)
        x0 = max(0, cx - half)
        x1 = min(32, cx + 7 - half)
        assert not occ[y0:y1, x0:x1].any()
        occ[y0:y1, x0:x1] = True


def test_random_mask_box_per_target_assigns_candidate_footprints_before_rejection():
    torch.manual_seed(7)
    x_clean = torch.ones((1, 1, 48, 48), dtype=torch.float32)
    cdd_orig = torch.ones((1, 2, 48, 48), dtype=torch.float32)

    result = make_pyramid_grid_context(
        x_clean=x_clean,
        sigmas=(1,),
        mask_fraction=1.0,
        mask_scale=0.0,
        spacing_scale=1.0,
        global_shift=False,
        align_scales=True,
        mask_box_size=9,
        mask_box_size_range=(3, 15),
        random_mask_box_per_target=True,
        inner_target_size=1,
        return_debug=True,
        enable_grid_jitter=False,
        enable_target_dithering=False,
        target_sampling_mode="random",
        priority_n_target=48,
        cdd_orig_in=cdd_orig,
    )

    _x_context, target_locations, _target_scales, target_valid, debug = result
    valid_boxes = debug["target_box_sizes"][0][target_valid[0]]
    valid_locs = target_locations[0][target_valid[0]]
    assert valid_boxes.numel() > 1
    assert torch.all(valid_boxes >= 3)
    assert torch.all(valid_boxes <= 15)
    assert torch.all((valid_boxes % 2) == 1)
    assert torch.unique(valid_boxes).numel() > 1

    occ = torch.zeros((48, 48), dtype=torch.bool)
    for (cy_t, cx_t), box_t in zip(valid_locs, valid_boxes):
        cy = int(cy_t.item())
        cx = int(cx_t.item())
        box = int(box_t.item())
        half = box // 2
        y0 = max(0, cy - half)
        y1 = min(48, cy + box - half)
        x0 = max(0, cx - half)
        x1 = min(48, cx + box - half)
        assert not occ[y0:y1, x0:x1].any()
        occ[y0:y1, x0:x1] = True


def test_manual_mask_box_sizes_reuse_last_when_shorter_than_cdd_channels():
    x_clean = torch.ones((1, 1, 32, 32), dtype=torch.float32)
    cdd_orig = torch.ones((1, 4, 32, 32), dtype=torch.float32)

    result = make_pyramid_grid_context(
        x_clean=x_clean,
        sigmas=(2, 4, 8, 16),
        mask_fraction=1.0,
        mask_scale=99.0,
        spacing_scale=1.0,
        global_shift=False,
        align_scales=True,
        mask_box_size=99,
        manual_mask_box_sizes=[5, 9],
        inner_target_size=1,
        return_debug=True,
        enable_grid_jitter=False,
        enable_target_dithering=False,
        cdd_orig_in=cdd_orig,
    )

    debug = result[-1]
    assert debug["cdd_box_sizes"][0].tolist() == [5.0, 9.0, 9.0, 9.0]


def test_manual_mask_box_sizes_ignore_extra_values():
    x_clean = torch.ones((1, 1, 32, 32), dtype=torch.float32)
    cdd_orig = torch.ones((1, 2, 32, 32), dtype=torch.float32)

    result = make_pyramid_grid_context(
        x_clean=x_clean,
        sigmas=(2, 4),
        mask_fraction=1.0,
        mask_scale=0.0,
        spacing_scale=1.0,
        global_shift=False,
        align_scales=True,
        mask_box_size=3,
        manual_mask_box_sizes=[7, 11, 15, 19],
        inner_target_size=1,
        return_debug=True,
        enable_grid_jitter=False,
        enable_target_dithering=False,
        cdd_orig_in=cdd_orig,
    )

    debug = result[-1]
    assert debug["cdd_box_sizes"][0].tolist() == [7.0, 11.0]


def test_dashboard_reconstructs_pyramid_mask_stack_from_cdd_diff(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    cdd_orig = torch.ones((1, 3, 8, 8), dtype=torch.float32)
    cdd_masked = cdd_orig.clone()
    cdd_masked[:, 0, 1:3, 1:3] = 0.0
    cdd_masked[:, 1, 4:6, 4:6] = 0.0
    outputs = {
        "x_clean": torch.ones((1, 1, 8, 8), dtype=torch.float32),
        "x_context": torch.ones((1, 1, 8, 8), dtype=torch.float32),
        "target_locations": torch.zeros((1, 1, 2), dtype=torch.long),
        "target_valid": torch.ones((1, 1), dtype=torch.bool),
        "pred_map": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
        "gt_map": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
        "pred_patches": torch.zeros((1, 1, 2, 1, 1), dtype=torch.float32),
        "gt_patches": torch.zeros((1, 1, 2, 1, 1), dtype=torch.float32),
        "cdd_channels_orig": cdd_orig,
        "cdd_channels_masked": cdd_masked,
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")

    dash_path = compute_dash_data(str(session_dir), overwrite=True)
    with np.load(dash_path) as data:
        stack = data["pyramid_mask_stack"]

    assert stack.shape == (3, 8, 8)
    assert int(np.count_nonzero(stack)) == 8


def test_dashboard_accepts_chw_embedding_artifacts(tmp_path):
    session_dir = tmp_path / "session_chw_embeddings"
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True)
    outputs = {
        "x_clean": torch.ones((1, 1, 8, 8), dtype=torch.float32),
        "x_context": torch.ones((1, 1, 8, 8), dtype=torch.float32),
        "target_locations": torch.zeros((1, 1, 2), dtype=torch.long),
        "target_valid": torch.ones((1, 1), dtype=torch.bool),
        "pred_map": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
        "gt_map": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
        "context_map": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
        "pred_patches": torch.zeros((1, 1, 4, 1, 1), dtype=torch.float32),
        "gt_patches": torch.zeros((1, 1, 4, 1, 1), dtype=torch.float32),
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")
    base = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    for branch in ("predict", "target", "context"):
        np.save(results_dir / f"{branch}_spatial_shape.npy", np.asarray([4, 4], dtype=np.int64))
        np.save(results_dir / f"{branch}_pca_xyz.npy", base + 10.0)
        np.save(results_dir / f"{branch}_umap_xyz.npy", base + 20.0)

    dash_path = compute_dash_data(str(session_dir), overwrite=True)
    with np.load(dash_path) as data:
        pred_pca = data["pred_pca3d"]
        pred_umap = data["pred_umap3d"]

    assert pred_pca.shape == (16, 3)
    assert pred_umap.shape == (16, 3)
    np.testing.assert_allclose(pred_pca[0], np.array([10.0, 26.0, 42.0], dtype=np.float32))
    np.testing.assert_allclose(pred_umap[0], np.array([20.0, 36.0, 52.0], dtype=np.float32))

    html_path = plot_dash_html(str(session_dir), overwrite=True)
    html = open(html_path, "r", encoding="utf-8").read()
    assert "Predict UMAP RGB" in html
    assert "Predict UMAP 3D Scatter" in html


def test_dashboard_uses_session_level_umap_artifact_and_keeps_partial_finite_rows(tmp_path):
    session_dir = tmp_path / "session_default_umap"
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True)
    outputs = {
        "x_clean": torch.ones((1, 1, 8, 8), dtype=torch.float32),
        "x_context": torch.ones((1, 1, 8, 8), dtype=torch.float32),
        "target_locations": torch.zeros((1, 1, 2), dtype=torch.long),
        "target_valid": torch.ones((1, 1), dtype=torch.bool),
        "pred_map": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
        "gt_map": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
        "context_map": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")
    base = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    umap = base + 30.0
    umap[:, 0, 0] = np.nan
    np.save(results_dir / "pca_xyz.npy", base + 10.0)
    np.save(results_dir / "umap_xyz.npy", umap)

    dash_path = compute_dash_data(str(session_dir), overwrite=True)
    with np.load(dash_path) as data:
        pred_umap = data["pred_umap3d"]

    assert pred_umap.shape == (16, 3)
    assert np.isnan(pred_umap[0]).all()
    np.testing.assert_allclose(pred_umap[1], np.array([31.0, 47.0, 63.0], dtype=np.float32))

    html_path = plot_dash_html(str(session_dir), overwrite=True)
    html = open(html_path, "r", encoding="utf-8").read()
    assert "Predict UMAP RGB" in html


def test_inference_dashboard_masks_invalid_edges_in_saved_embedding_artifacts(tmp_path):
    from src.utils import viz

    session_dir = tmp_path / "session_embedding_edges"
    session_dir.mkdir()
    x_clean = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    x_clean[:, :, 0, :] = 0.0
    outputs = {
        "x_clean": x_clean,
        "x_clean_raw": x_clean.clone(),
        "x_context": x_clean.clone(),
        "target_locations": torch.zeros((1, 1, 2), dtype=torch.long),
        "target_valid": torch.ones((1, 1), dtype=torch.bool),
        "pred_map": torch.randn((1, 4, 4, 4), dtype=torch.float32),
        "gt_map": torch.randn((1, 4, 4, 4), dtype=torch.float32),
        "context_map": torch.randn((1, 4, 4, 4), dtype=torch.float32),
    }
    old_umap = viz._compute_umap_nd
    viz._compute_umap_nd = lambda x, **_kwargs: x[:, :3].astype("float32")
    try:
        viz.save_inference_dashboard(str(session_dir), outputs, umap_cfg={})
    finally:
        viz._compute_umap_nd = old_umap

    pred_pca = np.load(session_dir / "results" / "predict_pca_xyz.npy").transpose(1, 2, 0)
    pred_umap = np.load(session_dir / "results" / "predict_umap_xyz.npy").transpose(1, 2, 0)

    assert np.isnan(pred_pca[0]).all()
    assert np.isnan(pred_umap[0]).all()
    assert np.isfinite(pred_pca[1:]).all()
    assert np.isfinite(pred_umap[1:]).all()


def test_inference_dashboard_respects_target_allowed_mask_for_embeddings(tmp_path):
    from src.utils import viz

    session_dir = tmp_path / "session_target_allowed"
    session_dir.mkdir()
    x_clean = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    target_allowed = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    target_allowed[:, :, :4, :4] = 1.0
    outputs = {
        "x_clean": x_clean,
        "x_clean_raw": x_clean.clone(),
        "x_context": x_clean.clone(),
        "target_locations": torch.zeros((1, 1, 2), dtype=torch.long),
        "target_valid": torch.ones((1, 1), dtype=torch.bool),
        "target_allowed_mask_map": target_allowed,
        "pred_map": torch.randn((1, 4, 4, 4), dtype=torch.float32),
        "gt_map": torch.randn((1, 4, 4, 4), dtype=torch.float32),
        "context_map": torch.randn((1, 4, 4, 4), dtype=torch.float32),
    }
    old_umap = viz._compute_umap_nd
    viz._compute_umap_nd = lambda x, **_kwargs: x[:, :3].astype("float32")
    try:
        viz.save_inference_dashboard(str(session_dir), outputs, umap_cfg={})
    finally:
        viz._compute_umap_nd = old_umap

    pred_pca = np.load(session_dir / "results" / "predict_pca_xyz.npy").transpose(1, 2, 0)
    pred_umap = np.load(session_dir / "results" / "predict_umap_xyz.npy").transpose(1, 2, 0)

    finite = np.isfinite(pred_pca).all(axis=-1)
    assert int(finite.sum()) == 4
    assert np.isfinite(pred_umap[:2, :2]).all()
    assert np.isnan(pred_pca[2:, :]).all()
    assert np.isnan(pred_pca[:, 2:]).all()


def test_inference_dashboard_exports_embedding_maps_not_flat_triplets(tmp_path):
    session_dir = tmp_path / "export_session"
    outputs = {
        "x_clean": torch.ones((1, 1, 8, 8), dtype=torch.float32),
        "x_clean_raw": torch.ones((1, 1, 8, 8), dtype=torch.float32),
        "x_context": torch.ones((1, 1, 8, 8), dtype=torch.float32),
        "target_locations": torch.zeros((1, 1, 2), dtype=torch.long),
        "target_valid": torch.ones((1, 1), dtype=torch.bool),
        "pred_map": torch.randn((1, 4, 4, 4), dtype=torch.float32),
        "gt_map": torch.randn((1, 4, 4, 4), dtype=torch.float32),
        "context_map": torch.randn((1, 4, 4, 4), dtype=torch.float32),
    }

    save_inference_dashboard(str(session_dir), outputs, umap_cfg={"n_neighbors": 4})
    results_dir = session_dir / "results"

    assert np.load(results_dir / "latent_vectors_full.npy").shape == (4, 4, 4)
    assert np.load(results_dir / "pca_xyz.npy").shape == (3, 4, 4)
    assert np.load(results_dir / "umap_xyz.npy").shape == (3, 4, 4)
    assert np.load(results_dir / "predict_latent_vectors_full.npy").shape == (4, 4, 4)
    assert np.load(results_dir / "predict_pca_xyz.npy").shape == (3, 4, 4)
    assert np.load(results_dir / "predict_umap_xyz.npy").shape == (3, 4, 4)
    assert not (results_dir / "predict_umap_x.npy").exists()
    assert not (results_dir / "predict_umap_y.npy").exists()
    assert not (results_dir / "predict_umap_z.npy").exists()


def test_embedding_migration_refuses_guessed_square_reshape(tmp_path):
    results_dir = tmp_path / "session" / "results"
    results_dir.mkdir(parents=True)
    np.save(results_dir / "predict_latent_vectors_full.npy", np.zeros((16, 4), dtype=np.float32))

    try:
        migrate_results(results_dir, dry_run=True)
    except ValueError as e:
        assert "refusing guessed reshape" in str(e)
    else:
        raise AssertionError("migration should refuse reshaping without spatial_shape")


def test_session_integrity_rejects_flat_embedding_maps(tmp_path):
    session_dir = tmp_path / "session"
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True)
    outputs = {
        "x_clean": torch.ones((1, 1, 8, 8), dtype=torch.float32),
        "pred_map": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
    }
    torch.save(outputs, session_dir / "inference_outputs.pt")
    for branch in ("predict", "target"):
        np.save(results_dir / f"{branch}_spatial_shape.npy", np.asarray([4, 4], dtype=np.int64))
        np.save(results_dir / f"{branch}_pca_xyz.npy", np.zeros((16, 3), dtype=np.float32))
        np.save(results_dir / f"{branch}_umap_xyz.npy", np.zeros((3, 4, 4), dtype=np.float32))

    report = check_session(str(session_dir))

    assert not report.ok
    assert any(f"invalid_shape:predict_pca_xyz.npy expected=(3,4,4) got=(16, 3)" in issue for issue in report.issues)
