from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.dataset import JEPADataset
from src.models.cdd_inspect import CDDOperatorFeatures2D
from src.models.symmetry import _crop_from_square, _pad_to_square


class ReviewFixesTests(unittest.TestCase):
    def test_cdd_stack_lognorm_preserves_positive_sign_below_one(self) -> None:
        op = CDDOperatorFeatures2D(
            features=("x",),
            expect_3d_pyramid=False,
            apply_lognorm=True,
            lognorm_on_stack=True,
            lognorm_mode="signed",
        )
        x = torch.full((1, 1, 4, 4), 1e-4)

        stack = op(x)["stack"]

        self.assertGreaterEqual(float(stack.min().item()), 0.0)

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


if __name__ == "__main__":
    unittest.main()
