from __future__ import annotations

import numpy as np

from src.dataset import JEPADataset
from src.train import _inverse_augmented_yx_to_native


def test_cdd_cache_crop_size_int_is_square_tuple(tmp_path):
    data_path = tmp_path / "field.npy"
    np.save(data_path, np.ones((8, 8), dtype=np.float32))

    cdd_cache = {
        (str(data_path), None): np.ones((4, 8, 8), dtype=np.float32),
    }
    dataset = JEPADataset(
        num_samples=1,
        data_root=str(tmp_path),
        npy_pattern="field.npy",
        crop_mode="center",
        crop_size=4,
        cdd_cache=cdd_cache,
    )

    cdd_orig, x_clean = dataset[0]

    assert tuple(dataset.crop_size) == (4, 4)
    assert tuple(cdd_orig.shape) == (4, 4, 4)
    assert tuple(x_clean.shape) == (1, 4, 4)


def test_dataset_metadata_tracks_crop_for_native_visit_counts(tmp_path):
    data_path = tmp_path / "field.npy"
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    np.save(data_path, arr)

    dataset = JEPADataset(
        num_samples=1,
        data_root=str(tmp_path),
        npy_pattern="field.npy",
        crop_mode="center",
        crop_size=4,
        d4_augment=False,
        return_metadata=True,
    )

    sample, meta = dataset[0]

    assert tuple(sample.shape) == (1, 4, 4)
    assert meta["full_h"] == 8
    assert meta["full_w"] == 8
    assert meta["crop_y0"] == 2
    assert meta["crop_x0"] == 2
    assert _inverse_augmented_yx_to_native(1, 1, meta) == (3, 3)


def test_inverse_augmented_yx_to_native_undoes_d4_transform():
    # Forward transform for a 4x4 crop: rot90(k=1), then horizontal flip.
    # Native crop coordinate (1, 2) -> after rot90 => (1, 1), after flip_x => (1, 2).
    meta = {
        "full_h": 4,
        "full_w": 4,
        "crop_y0": 0,
        "crop_x0": 0,
        "pre_aug_h": 4,
        "pre_aug_w": 4,
        "post_aug_h": 4,
        "post_aug_w": 4,
        "rot_k": 1,
        "flip_x": True,
        "flip_y": False,
    }

    assert _inverse_augmented_yx_to_native(1, 2, meta) == (1, 2)
