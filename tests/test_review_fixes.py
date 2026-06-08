from __future__ import annotations

import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import JEPADataset
from src.inference import _apply_tta_2d, _forward_tta_streaming_2d, _iter_tta_views_2d
from src.inference_from_session import TileLayout2D, _stitch_tile_tensor, load_raw_data, run_inference_on_data
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

    def test_tile_stitching_scales_origins_for_downsampled_maps(self) -> None:
        tiles = torch.zeros((4, 1, 2, 2), dtype=torch.float32)
        tiles[0, 0, :2, :2] = 1.0
        tiles[1, 0, :2, :1] = 2.0
        tiles[2, 0, :1, :2] = 3.0
        tiles[3, 0, :1, :1] = 4.0
        layout = TileLayout2D(
            original_shape=(6, 6),
            crop_size=4,
            origins=((0, 0), (0, 4), (4, 0), (4, 4)),
            valid_shapes=((4, 4), (4, 2), (2, 4), (2, 2)),
        )

        stitched = _stitch_tile_tensor(tiles, layout)

        self.assertEqual(tuple(stitched.shape), (1, 1, 3, 3))
        self.assertEqual(float(stitched[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(stitched[0, 0, 0, 2]), 2.0)
        self.assertEqual(float(stitched[0, 0, 2, 0]), 3.0)
        self.assertEqual(float(stitched[0, 0, 2, 2]), 4.0)

    def test_tiled_raw_data_uses_global_normalization(self) -> None:
        arr = np.arange(24, dtype=np.float32).reshape(4, 6)
        with tempfile.NamedTemporaryFile(suffix=".npy") as f:
            np.save(f.name, arr)
            data, layout = load_raw_data(
                f.name,
                crop_size=4,
                crop_mode="tile",
                mode="image",
                return_layout=True,
            )

        self.assertIsNotNone(layout)
        self.assertEqual(tuple(data.shape), (6, 1, 4, 4))
        # arr[0, 2] appears in the first tile at [0, 2] and the second at [0, 0].
        self.assertEqual(float(data[0, 0, 0, 2]), float(data[1, 0, 0, 0]))

    def test_3d_slab_raw_data_returns_5d_sliding_slabs(self) -> None:
        volume = np.arange(5 * 3 * 4, dtype=np.float32).reshape(5, 3, 4)
        with tempfile.NamedTemporaryFile(suffix=".npy") as f:
            np.save(f.name, volume)
            data, layout = load_raw_data(
                f.name,
                mode="3d_slab",
                slice_axis=0,
                slab_depth=3,
                return_layout=True,
            )

        self.assertIsNone(layout)
        self.assertEqual(tuple(data.shape), (5, 1, 3, 3, 4))

    def test_run_inference_returns_target_metadata(self) -> None:
        class FakeModel:
            mode = "image"

            def eval(self) -> None:
                return None

            def __call__(self, x: torch.Tensor, mask_inference: bool) -> dict:
                b = x.shape[0]
                return {
                    "pred_map": x + 1.0,
                    "gt_map": x + 2.0,
                    "context_map": x + 3.0,
                    "target_locations": torch.zeros((b, 2, 2), dtype=torch.long, device=x.device),
                    "target_scales": torch.ones((b, 2), dtype=x.dtype, device=x.device),
                    "target_valid": torch.ones((b, 2), dtype=torch.bool, device=x.device),
                }

        data = torch.zeros((3, 1, 4, 4), dtype=torch.float32)
        loader = DataLoader(data, batch_size=2)

        outputs = run_inference_on_data(FakeModel(), loader, torch.device("cpu"))

        self.assertEqual(tuple(outputs["target_locations"].shape), (3, 2, 2))
        self.assertEqual(tuple(outputs["target_scales"].shape), (3, 2))
        self.assertEqual(tuple(outputs["target_valid"].shape), (3, 2))

    def test_d4_tta_views_align_back_to_original_shape(self) -> None:
        x = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)

        restored = [_apply_tta_2d(name, view) for name, view in _iter_tta_views_2d(x, "d4")]

        self.assertEqual(len(restored), 8)
        for item in restored:
            self.assertEqual(tuple(item.shape), tuple(x.shape))
            torch.testing.assert_close(item, x)

    def test_streaming_tta_matches_stack_average(self) -> None:
        x = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)

        def forward_one(xv: torch.Tensor, _cdv: torch.Tensor | None) -> dict:
            return {
                "pred_map": xv + 1.0,
                "gt_map": xv * 2.0,
                "cdd_channels_orig": xv + 3.0,
                "target_locations": torch.zeros((1, 1, 2), dtype=torch.long),
            }

        expected_pred = None
        expected_gt = None
        expected_count = 0
        for name, view in _iter_tta_views_2d(x, "d4"):
            out = forward_one(view, None)
            pred = _apply_tta_2d(name, out["pred_map"])
            gt = _apply_tta_2d(name, out["gt_map"])
            expected_pred = pred if expected_pred is None else expected_pred + pred
            expected_gt = gt if expected_gt is None else expected_gt + gt
            expected_count += 1
        expected_pred = expected_pred / float(expected_count)
        expected_gt = expected_gt / float(expected_count)

        streamed, n_views = _forward_tta_streaming_2d(
            x=x,
            mode="d4",
            forward_one=forward_one,
        )

        self.assertEqual(n_views, 8)
        torch.testing.assert_close(streamed["pred_map"], expected_pred)
        torch.testing.assert_close(streamed["gt_map"], expected_gt)
        torch.testing.assert_close(streamed["cdd_channels_orig"], x + 3.0)


if __name__ == "__main__":
    unittest.main()
