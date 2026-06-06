from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.dataset import JEPADataset
from src.inference import _apply_tta_2d, _tta_views_2d
from src.inference_from_session import TileLayout2D, _stitch_tile_tensor
from src.models.symmetry import _crop_from_square, _pad_to_square


class ReviewFixesTests(unittest.TestCase):
    def test_square_padding_is_symmetric_and_crops_back(self) -> None:
        x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4)

        padded, orig_shape = _pad_to_square(x)
        restored = _crop_from_square(padded, orig_shape)

        self.assertEqual(tuple(padded.shape), (1, 1, 4, 4))
        torch.testing.assert_close(padded[..., 1:3, :], x)
        torch.testing.assert_close(restored, x)

    def test_preprocess_arr2d_normalizes_using_finite_values_only(self) -> None:
        dataset = JEPADataset.__new__(JEPADataset)
        arr = np.array([[np.nan, 100.0], [200.0, np.inf]], dtype=np.float32)

        out = dataset._preprocess_arr2d(arr)

        self.assertEqual(float(out[0, 0]), 0.0)
        self.assertEqual(float(out[0, 1]), 0.0)
        self.assertEqual(float(out[1, 0]), 1.0)
        self.assertEqual(float(out[1, 1]), 0.0)

    def test_d4_augment_can_apply_rot90_on_square_tensors(self) -> None:
        dataset = JEPADataset.__new__(JEPADataset)
        dataset.d4_augment = True
        dataset.random_roll_max = 0
        x = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)

        with patch("src.dataset.np.random.randint", side_effect=[1, 0]):
            (out,) = dataset._apply_augmentations(x)

        torch.testing.assert_close(out, torch.rot90(x, k=1, dims=(-2, -1)))

    def test_tile_stitching_restores_full_image_canvas(self) -> None:
        tiles = torch.zeros((4, 1, 4, 4), dtype=torch.float32)
        tiles[0, 0, :4, :4] = 1.0
        tiles[1, 0, :4, :2] = 2.0
        tiles[2, 0, :2, :4] = 3.0
        tiles[3, 0, :2, :2] = 4.0
        layout = TileLayout2D(
            original_shape=(6, 6),
            crop_size=4,
            origins=((0, 0), (0, 4), (4, 0), (4, 4)),
            valid_shapes=((4, 4), (4, 2), (2, 4), (2, 2)),
        )

        stitched = _stitch_tile_tensor(tiles, layout)

        self.assertEqual(tuple(stitched.shape), (1, 1, 6, 6))
        self.assertEqual(float(stitched[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(stitched[0, 0, 0, 5]), 2.0)
        self.assertEqual(float(stitched[0, 0, 5, 0]), 3.0)
        self.assertEqual(float(stitched[0, 0, 5, 5]), 4.0)

    def test_d4_tta_views_align_back_to_original_shape(self) -> None:
        x = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)

        restored = [_apply_tta_2d(name, view) for name, view in _tta_views_2d(x, "d4")]

        self.assertEqual(len(restored), 8)
        for item in restored:
            self.assertEqual(tuple(item.shape), tuple(x.shape))
            torch.testing.assert_close(item, x)


if __name__ == "__main__":
    unittest.main()
