from __future__ import annotations

import numpy as np

from src.dataset import JEPADataset, resolve_input_files
from src.train import _input_inference_dir, _inverse_augmented_yx_to_native


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


def test_dataset_accepts_explicit_input_files_in_order(tmp_path):
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.ones((4, 4), dtype=np.float32))
    np.save(second, np.ones((4, 4), dtype=np.float32) * 2)

    files = resolve_input_files(data_root=str(tmp_path), input_files=["second.npy", "first.npy"])
    dataset = JEPADataset(
        num_samples=4,
        data_root=str(tmp_path),
        input_files=["second.npy", "first.npy"],
    )

    assert files == [str(second), str(first)]
    assert dataset.sample_index == [(str(second), None), (str(first), None)]
    assert tuple(dataset[0].shape) == (1, 4, 4)
    assert tuple(dataset[1].shape) == (1, 4, 4)


def test_input_inference_dir_is_stable_and_numbered(tmp_path):
    out = _input_inference_dir(str(tmp_path), 3, ("/data/a weird:name.npy", None))

    assert out.endswith("inference_inputs/003_a_weird_name")
