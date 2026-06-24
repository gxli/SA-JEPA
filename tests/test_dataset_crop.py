from __future__ import annotations

import numpy as np

from src.dataset import JEPADataset


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
