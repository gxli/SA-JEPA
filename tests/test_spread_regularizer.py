from __future__ import annotations

import unittest

import torch

from src.losses import (
    compute_output_spread_regularizer_loss,
    compute_spread_regularizer_loss,
    embedding_spread_stats,
    extract_valid_dense_embeddings,
    parse_spread_regularizer_config,
    sketched_sigreg_loss,
    spread_regularizer_loss,
    weak_sigreg_loss,
)


class SpreadRegularizerTests(unittest.TestCase):
    def test_extract_valid_dense_embeddings_keeps_spatial_tokens(self) -> None:
        patches = torch.randn(2, 3, 4, 5, 5)
        valid = torch.tensor([[True, False, True], [False, True, True]])
        outputs = {"context_patches": patches, "target_valid": valid}

        z = extract_valid_dense_embeddings(outputs, key="context_patches")

        self.assertEqual(tuple(z.shape), (4 * 5 * 5, 4))

    def test_output_sigreg_defaults_to_pooled_patch_tokens(self) -> None:
        torch.manual_seed(0)
        context = torch.randn(2, 3, 4, 5, 5).requires_grad_()
        valid = torch.tensor([[True, False, True], [False, True, True]])
        outputs = {"context_patches": context, "target_valid": valid}

        loss, z_ctx = compute_output_spread_regularizer_loss(
            outputs,
            {"type": "std_hinge", "target": "context", "target_std": 1.0, "eps": 1e-4},
        )
        loss.backward()

        self.assertEqual(tuple(z_ctx.shape), (4, 4))
        self.assertGreater(float(context.grad.abs().sum().item()), 0.0)

    def test_output_sigreg_can_use_dense_spatial_tokens(self) -> None:
        torch.manual_seed(0)
        context = torch.randn(2, 3, 4, 5, 5).requires_grad_()
        valid = torch.tensor([[True, False, True], [False, True, True]])
        outputs = {"context_patches": context, "target_valid": valid}

        loss, z_ctx = compute_output_spread_regularizer_loss(
            outputs,
            {"type": "std_hinge", "target": "context", "target_std": 1.0, "eps": 1e-4, "spatial_mode": "dense"},
        )
        loss.backward()

        self.assertEqual(tuple(z_ctx.shape), (4 * 5 * 5, 4))
        self.assertGreater(float(context.grad.abs().sum().item()), 0.0)

    def test_output_sigreg_defaults_to_context_not_predictor(self) -> None:
        torch.manual_seed(0)
        context = (torch.randn(2, 3, 4, 5, 5) * 1e-3).requires_grad_()
        pred = (torch.randn(2, 3, 4, 5, 5) * 1e-3).requires_grad_()
        valid = torch.tensor([[True, False, True], [False, True, True]])
        outputs = {"context_patches": context, "pred_patches": pred, "target_valid": valid}

        loss, z_ctx = compute_output_spread_regularizer_loss(
            outputs,
            {"type": "std_hinge", "target_std": 1.0, "eps": 1e-4},
        )
        loss.backward()

        self.assertEqual(tuple(z_ctx.shape), (4, 4))
        self.assertGreater(float(context.grad.abs().sum().item()), 0.0)
        self.assertIsNone(pred.grad)

    def test_output_sigreg_can_include_predictor_safety_term(self) -> None:
        torch.manual_seed(0)
        context = (torch.randn(2, 3, 4, 5, 5) * 1e-3).requires_grad_()
        pred = (torch.randn(2, 3, 4, 5, 5) * 1e-3).requires_grad_()
        valid = torch.tensor([[True, False, True], [False, True, True]])
        outputs = {"context_patches": context, "pred_patches": pred, "target_valid": valid}

        loss, _ = compute_output_spread_regularizer_loss(
            outputs,
            {"type": "std_hinge", "target": "context", "target_std": 1.0, "eps": 1e-4},
            include_predictor=True,
        )
        loss.backward()

        self.assertGreater(float(context.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(pred.grad.abs().sum().item()), 0.0)

    def test_output_sigreg_can_target_predictor(self) -> None:
        torch.manual_seed(0)
        context = (torch.randn(2, 3, 4, 5, 5) * 1e-3).requires_grad_()
        pred = (torch.randn(2, 3, 4, 5, 5) * 1e-3).requires_grad_()
        valid = torch.tensor([[True, False, True], [False, True, True]])
        outputs = {"context_patches": context, "pred_patches": pred, "target_valid": valid}

        loss, z_spread = compute_output_spread_regularizer_loss(
            outputs,
            {"type": "std_hinge", "target": "predictor", "target_std": 1.0, "eps": 1e-4},
        )
        loss.backward()

        self.assertEqual(tuple(z_spread.shape), (4, 4))
        self.assertIsNone(context.grad)
        self.assertGreater(float(pred.grad.abs().sum().item()), 0.0)

    def test_near_collapsed_embeddings_produce_nonzero_gradient(self) -> None:
        torch.manual_seed(0)
        z = (torch.randn(32, 8) * 1e-3).requires_grad_()

        loss = spread_regularizer_loss(z, target_std=1.0, eps=1e-4)
        loss.backward()

        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(float(z.grad.abs().sum().item()), 0.0)

    def test_healthy_embeddings_produce_low_spread_loss(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(4096, 8) * 1.6

        loss = spread_regularizer_loss(z, target_std=1.0, eps=1e-4)

        self.assertLess(float(loss.item()), 1e-3)

    def test_spread_regularizer_repeated_batch_does_not_boost_gradient(self) -> None:
        torch.manual_seed(0)
        z_small = (torch.randn(128, 8) * 1e-3).requires_grad_()
        z_large = z_small.detach().repeat(4, 1).requires_grad_()

        loss_small = spread_regularizer_loss(z_small, target_std=1.0, eps=1e-4)
        loss_large = spread_regularizer_loss(z_large, target_std=1.0, eps=1e-4)
        loss_small.backward()
        loss_large.backward()

        self.assertAlmostEqual(float(loss_small.item()), float(loss_large.item()), places=3)
        self.assertAlmostEqual(
            float(z_large.grad.abs().sum().item()),
            float(z_small.grad.abs().sum().item()),
            delta=1e-3,
        )

    def test_weak_sigreg_produces_gradient(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(128, 32).requires_grad_()

        loss = weak_sigreg_loss(z, sketch_dim=8, eps=1e-6)
        loss.backward()

        self.assertGreaterEqual(float(loss.item()), 0.0)
        self.assertGreater(float(z.grad.abs().sum().item()), 0.0)

    def test_weak_sigreg_near_collapse_keeps_gradient(self) -> None:
        torch.manual_seed(0)
        z = (torch.randn(128, 32) * 1e-6).requires_grad_()

        loss = weak_sigreg_loss(z, sketch_dim=8, eps=1e-4)
        loss.backward()

        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(float(z.grad.abs().sum().item()), 0.0)

    def test_weak_sigreg_collapse_is_variance_hinge_not_covariance_mse(self) -> None:
        z = torch.zeros(128, 32)

        loss = weak_sigreg_loss(z, sketch_dim=8, eps=1e-4)

        self.assertGreater(float(loss.item()), 0.98)
        self.assertLessEqual(float(loss.item()), 1.0)

    def test_weak_sigreg_uses_target_std_for_full_variance_hinge(self) -> None:
        z = torch.zeros(128, 32)

        loss_low_target = weak_sigreg_loss(z, target_std=0.5, sketch_dim=8, eps=1e-4)
        loss_high_target = weak_sigreg_loss(z, target_std=1.0, sketch_dim=8, eps=1e-4)

        self.assertAlmostEqual(float(loss_low_target.item()), 0.49, delta=1e-4)
        self.assertAlmostEqual(float(loss_high_target.item()), 0.99, delta=1e-4)

    def test_weak_sigreg_repeated_batch_does_not_boost_gradient(self) -> None:
        torch.manual_seed(0)
        z_small = (torch.randn(128, 32) * 1e-3).requires_grad_()
        z_large = z_small.detach().repeat(4, 1).requires_grad_()

        torch.manual_seed(123)
        loss_small = weak_sigreg_loss(z_small, sketch_dim=8, eps=1e-4)
        torch.manual_seed(123)
        loss_large = weak_sigreg_loss(z_large, sketch_dim=8, eps=1e-4)
        loss_small.backward()
        loss_large.backward()

        self.assertAlmostEqual(float(loss_small.item()), float(loss_large.item()), places=3)
        self.assertAlmostEqual(
            float(z_large.grad.abs().sum().item()),
            float(z_small.grad.abs().sum().item()),
            places=3,
        )

    def test_weak_sigreg_dispatch(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(128, 32).requires_grad_()
        cfg = {"type": "weak_sigreg", "target_std": 1.0, "sketch_dim": 8, "eps": 1e-4}

        loss = compute_spread_regularizer_loss(z, cfg)
        loss.backward()

        self.assertGreaterEqual(float(loss.item()), 0.0)
        self.assertGreater(float(z.grad.abs().sum().item()), 0.0)

    def test_sketched_sigreg_legacy_dispatch(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(128, 32).requires_grad_()
        cfg = {"type": "sketched_sigreg", "sketch_dim": 8, "eps": 1e-6}

        torch.manual_seed(123)
        direct = sketched_sigreg_loss(z, target_std=1.0, sketch_dim=8, eps=1e-6)
        torch.manual_seed(123)
        loss = compute_spread_regularizer_loss(z, cfg)
        loss.backward()

        self.assertAlmostEqual(float(loss.item()), float(direct.item()), places=6)
        self.assertGreaterEqual(float(loss.item()), 0.0)
        self.assertGreater(float(z.grad.abs().sum().item()), 0.0)

    def test_sketched_sigreg_has_escape_gradient_near_collapse(self) -> None:
        torch.manual_seed(0)
        z = (1e-4 * torch.randn(128, 32)).requires_grad_()

        loss = sketched_sigreg_loss(z, target_std=1.0, sketch_dim=8, eps=1e-4)
        loss.backward()

        self.assertGreater(float(loss.item()), 1.0)
        self.assertGreater(float(z.grad.abs().sum().item()), 0.0)

    def test_sketched_sigreg_penalizes_rank_one_channel_copy(self) -> None:
        torch.manual_seed(0)
        base = torch.randn(128, 1)
        z = base.repeat(1, 32).requires_grad_()

        loss = sketched_sigreg_loss(z, target_std=1.0, sketch_dim=8, eps=1e-4)
        loss.backward()

        self.assertGreater(float(loss.item()), 0.5)
        self.assertGreater(float(z.grad.abs().sum().item()), 0.0)

    def test_sketched_sigreg_has_gradient_near_collapse(self) -> None:
        torch.manual_seed(0)
        z = (torch.randn(128, 32) * 1e-6).requires_grad_()

        loss = sketched_sigreg_loss(z, target_std=1.0, sketch_dim=8, eps=1e-4)
        loss.backward()

        self.assertGreater(float(loss.item()), 1.0)
        self.assertGreater(float(z.grad.abs().sum().item()), 0.0)

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
        self.assertEqual(cfg["spatial_mode"], "pooled")

    def test_weak_sigreg_schema_is_explicit(self) -> None:
        cfg = parse_spread_regularizer_config(
            {
                "spread_regularizer": {
                    "type": "weak_sigreg",
                    "target": "context",
                    "weight": 0.5,
                    "sketch_dim": 64,
                    "eps": 1e-6,
                }
            }
        )

        self.assertEqual(cfg["type"], "weak_sigreg")
        self.assertEqual(cfg["target"], "context")
        self.assertEqual(cfg["weight"], 0.5)
        self.assertEqual(cfg["sketch_dim"], 64)

    def test_spread_regularizer_schema_accepts_predictor_target(self) -> None:
        cfg = parse_spread_regularizer_config(
            {
                "spread_regularizer": {
                    "type": "std_hinge",
                    "target": "predictor",
                    "weight": 2,
                }
            }
        )

        self.assertEqual(cfg["target"], "predictor")

    def test_flat_spread_regularizer_keys_are_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            parse_spread_regularizer_config({"spread_regularizer_weight": 2})

    def test_invalid_spread_regularizer_settings_are_rejected(self) -> None:
        invalid_blocks = (
            {"type": "other", "target": "context", "weight": 2},
            {"type": "std_hinge", "target": "target", "weight": 2},
            {"type": "std_hinge", "target": "context", "weight": -1},
            {"type": "std_hinge", "target": "context", "weight": 2, "spatial_mode": "tokens"},
            {"type": "weak_sigreg", "target": "context", "weight": 0.5, "sketch_dim": 0},
        )
        for block in invalid_blocks:
            with self.subTest(block=block), self.assertRaises(AssertionError):
                parse_spread_regularizer_config({"spread_regularizer": block})


if __name__ == "__main__":
    unittest.main()
