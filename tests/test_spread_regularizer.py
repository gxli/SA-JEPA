from __future__ import annotations

import unittest

import torch

from src.losses import (
    embedding_spread_stats,
    parse_spread_regularizer_config,
    spread_regularizer_loss,
)


class SpreadRegularizerTests(unittest.TestCase):
    def test_near_collapsed_embeddings_produce_nonzero_gradient(self) -> None:
        torch.manual_seed(0)
        z = (torch.randn(32, 8) * 1e-3).requires_grad_()

        loss = spread_regularizer_loss(z, target_std=1.0, eps=1e-4)
        loss.backward()

        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(float(z.grad.abs().sum().item()), 0.0)

    def test_healthy_embeddings_produce_low_spread_loss(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(4096, 8) * 1.2

        loss = spread_regularizer_loss(z, target_std=1.0, eps=1e-4)

        self.assertLess(float(loss.item()), 1e-3)

    def test_perfect_collapse_is_detected_in_diagnostics(self) -> None:
        stats = embedding_spread_stats(torch.zeros(32, 8), target_std=1.0)

        self.assertEqual(stats["dead_channel_count"], 8)
        self.assertEqual(stats["embed_under_spread_frac"], 1.0)
        self.assertEqual(stats["context_manifold_size"], 0.0)

    def test_spread_regularizer_schema_is_explicit(self) -> None:
        cfg = parse_spread_regularizer_config(
            {
                "spread_regularizer": {
                    "type": "std_hinge",
                    "target": "context",
                    "weight": 2,
                    "target_std": 1.0,
                    "eps": 1e-4,
                }
            }
        )

        self.assertEqual(cfg["type"], "std_hinge")
        self.assertEqual(cfg["target"], "context")
        self.assertEqual(cfg["weight"], 2.0)

    def test_flat_spread_regularizer_keys_are_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            parse_spread_regularizer_config({"spread_regularizer_weight": 2})

    def test_invalid_spread_regularizer_settings_are_rejected(self) -> None:
        invalid_blocks = (
            {"type": "other", "target": "context", "weight": 2},
            {"type": "std_hinge", "target": "predictor", "weight": 2},
            {"type": "std_hinge", "target": "context", "weight": -1},
        )
        for block in invalid_blocks:
            with self.subTest(block=block), self.assertRaises(AssertionError):
                parse_spread_regularizer_config({"spread_regularizer": block})


if __name__ == "__main__":
    unittest.main()
