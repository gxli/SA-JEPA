import numpy as np
import torch

from src.models.masking import (
    _build_priority_catalogue_from_cdd_ratio,
    _fractional_spatial_target_budget,
    make_pyramid_grid_context,
)


def test_fractional_spatial_target_budget_matches_area_ratio_for_integer_case():
    budget = _fractional_spatial_target_budget(
        height=64,
        width=64,
        box_size=16,
        oversample=3.0,
        device=torch.device("cpu"),
    )

    assert budget == 48


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
        target_sampling_mode="priority_sampling",
        priority_top_percent=100.0,
        priority_n_target="auto",
        priority_candidate_oversample=3.0,
        cdd_orig_in=cdd_orig,
    )

    debug = result[-1]
    assert debug["priority_good_candidates"].item() > 1000
    assert debug["priority_prescreen_candidates"].item() <= 55
