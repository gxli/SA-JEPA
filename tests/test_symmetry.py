from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from src.models.symmetry import symmetric_forward_2d


class CountingIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_examples = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.num_examples += int(x.shape[0])
        return x


class SymmetryTests(unittest.TestCase):
    def test_symmetric_forward_2d_defaults_to_flip4_views(self) -> None:
        encoder = CountingIdentity()
        x = torch.randn(2, 3, 5, 7)

        avg, var = symmetric_forward_2d(encoder, x, return_var=True, max_views_per_forward=3)

        self.assertEqual(tuple(avg.shape), tuple(x.shape))
        self.assertEqual(tuple(var.shape), tuple(x.shape))
        self.assertEqual(encoder.num_examples, 4 * x.shape[0])
        self.assertTrue(torch.allclose(avg, x, atol=1e-6))
        self.assertLess(float(var.abs().max().item()), 1e-10)

    def test_symmetric_forward_2d_can_use_d4_views(self) -> None:
        encoder = CountingIdentity()
        x = torch.randn(2, 3, 5, 7)

        avg, var = symmetric_forward_2d(encoder, x, return_var=True, max_views_per_forward=3, view_mode="d4")

        self.assertEqual(tuple(avg.shape), tuple(x.shape))
        self.assertEqual(tuple(var.shape), tuple(x.shape))
        self.assertEqual(encoder.num_examples, 8 * x.shape[0])
        self.assertTrue(torch.allclose(avg, x, atol=1e-6))
        self.assertLess(float(var.abs().max().item()), 1e-10)


if __name__ == "__main__":
    unittest.main()
