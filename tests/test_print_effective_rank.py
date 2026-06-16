from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "print_effective_rank.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("print_effective_rank", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrintEffectiveRankTests(unittest.TestCase):
    def test_summary_prints_per_row_energy_and_loss_ratios(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, energy, total0, total1, pred0, pred1, spread0, spread1, weighted0, weighted1 in (
                ("run_a", 1.25, 11.0, 1.0, 8.0, 2.0, 4.0, 2.0, 5.0, 1.0),
                ("run_b", 9.50, 12.0, 9.0, 4.0, 6.0, 2.0, 6.0, 7.0, 14.0),
            ):
                session = root / name
                session.mkdir()
                (session / "config_used.json").write_text(
                    json.dumps(
                        {
                            "model": {
                                "mode": "pyramid",
                                "mask_scale_factor": 1.2,
                                "mask_footprint_px": 0,
                            },
                            "train": {},
                        }
                    ),
                    encoding="utf-8",
                )
                (session / "effective_rank.txt").write_text("2.0\n", encoding="utf-8")
                (session / "rank_diagnostics.json").write_text(
                    json.dumps({"energy": energy}),
                    encoding="utf-8",
                )
                (session / "metrics.csv").write_text(
                    "\n".join(
                        [
                            "epoch,loss_total,loss_prediction,loss_spread,weighted_spread",
                            f"0,{total0},{pred0},{spread0},{weighted0}",
                            f"1,{total1},{pred1},{spread1},{weighted1}",
                        ]
                    ),
                    encoding="utf-8",
                )

            old_argv = sys.argv
            sys.argv = ["print_effective_rank.py", str(root)]
            out = io.StringIO()
            try:
                with contextlib.redirect_stdout(out):
                    self.assertEqual(module.main(), 0)
                rows_by_name = {
                    row[0]: row
                    for row in module.rank_summary([str(root / "run_a"), str(root / "run_b")])
                }
            finally:
                sys.argv = old_argv

        lines = out.getvalue().splitlines()
        self.assertIn("Sampled Embedding Diagnostics (sorted by session filename A-Z)", lines)
        self.assertTrue(any("target_effrank" in line for line in lines))
        header = next(line for line in lines if line.startswith("session "))
        self.assertIn("status", header)
        self.assertIn("cdd_n", header)
        self.assertIn("dil", header)
        self.assertIn("hardcap", header)
        self.assertIn("spread_w", header)
        self.assertIn("total", header)
        self.assertIn("pred", header)
        self.assertIn("spread", header)
        self.assertNotIn("cos_diag", header)
        self.assertNotIn("cos_r", header)
        self.assertNotIn("sim_r", header)
        self.assertNotIn("hinge_r", header)
        self.assertNotIn("hinge_w", header)
        self.assertNotIn("sig_w", header)
        self.assertIn("predictor_effrank", header)
        self.assertIn("context_effrank", header)
        self.assertNotIn("vicvar_w", header)

        run_a = next(line for line in lines if line.startswith("run_a "))
        run_b = next(line for line in lines if line.startswith("run_b "))

        self.assertIn("1.2500", run_a)
        self.assertIn("2.0000", run_a)
        self.assertIn("1.0000", run_a)
        self.assertIn("9.5000", run_b)
        self.assertIn("9.0000", run_b)
        self.assertIn("6.0000", run_b)
        self.assertNotEqual(rows_by_name["run_a"][19:22], rows_by_name["run_b"][19:22])

    def test_summary_shows_vicreg_columns_only_when_active(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "run_vic"
            session.mkdir()
            (session / "config_used.json").write_text(
                json.dumps(
                    {
                        "model": {"mode": "pyramid", "mask_scale_factor": 1.2},
                        "train": {
                            "spread_regularizer": {"type": "std_hinge", "weight": 2},
                            "vicreg_var_weight": 1.0,
                            "vicreg_cov_weight": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (session / "metrics.csv").write_text(
                "\n".join(
                    [
                        "epoch,sim,loss_spread,weighted_spread,loss_vicreg_var,loss_vicreg_cov,weighted_vicreg_var,weighted_vicreg_cov",
                        "0,1,2,2,3,4,3,2",
                        "1,2,1,1,6,2,6,1",
                    ]
                ),
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                module.print_rank_table([str(session)])

        header = next(line for line in out.getvalue().splitlines() if line.startswith("session "))
        self.assertIn("vicvar_w", header)
        self.assertIn("viccov_w", header)
        self.assertIn("hinge_r", header)
        self.assertIn("vicv_r", header)
        self.assertIn("wvc_r", header)

    def test_summary_shows_sim_ratio_for_l2_prediction_runs(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "run_l2"
            session.mkdir()
            (session / "config_used.json").write_text(
                json.dumps(
                    {
                        "model": {"mode": "pyramid", "mask_scale_factor": 1.2, "normalize_loss_l2": True},
                        "train": {"spread_regularizer": {"type": "std_hinge", "weight": 0}},
                    }
                ),
                encoding="utf-8",
            )
            (session / "metrics.csv").write_text(
                "\n".join(
                    [
                        "epoch,sim,loss_total,loss_prediction,loss_spread,weighted_spread",
                        "0,0.5,3,2,0,0",
                        "1,1.0,1,1,0,0",
                    ]
                ),
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                module.print_rank_table([str(session)])

        header = next(line for line in out.getvalue().splitlines() if line.startswith("session "))
        self.assertIn("sim_r", header)
        self.assertNotIn("hinge_r", header)

    def test_summary_hides_vicreg_columns_when_weights_are_zero(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "run_vic_off"
            session.mkdir()
            (session / "config_used.json").write_text(
                json.dumps(
                    {
                        "model": {"mode": "pyramid", "mask_scale_factor": 1.2},
                        "train": {
                            "spread_regularizer": {"type": "std_hinge", "weight": 2},
                            "vicreg_var_weight": 0.0,
                            "vicreg_cov_weight": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (session / "metrics.csv").write_text(
                "\n".join(
                    [
                        "epoch,sim,loss_spread,weighted_spread,loss_vicreg_var,loss_vicreg_cov,weighted_vicreg_var,weighted_vicreg_cov",
                        "0,1,2,2,3,4,0,0",
                        "1,2,1,1,6,2,0,0",
                    ]
                ),
                encoding="utf-8",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                module.print_rank_table([str(session)])

        header = next(line for line in out.getvalue().splitlines() if line.startswith("session "))
        self.assertNotIn("vicvar_w", header)
        self.assertNotIn("vicv_r", header)

    def test_api_rank_summary_columns_match_script_rows(self) -> None:
        from src.api import ScaleAwareJEPA

        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            (session / "config_used.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "mode": "3d_full_volume",
                            "mask_size_scaling": 1.2,
                            "mask_size": 5,
                            "encoder_depth": 2,
                            "sigmas": [2, 4, 8, 16],
                            "dilations": [1, 2],
                            "mask_box_hardcap": 9,
                        },
                        "train": {},
                    }
                ),
                encoding="utf-8",
            )
            (session / "rank_diagnostics.json").write_text(json.dumps({"energy": 3.5}), encoding="utf-8")
            model = ScaleAwareJEPA(config={"model": {}, "train": {}, "data": {}})
            model._session_dir = str(session)

            summary = model.analyze_rank()

        self.assertEqual(summary["depth"], "2")
        self.assertEqual(summary["dilations"], "[1,2]")
        self.assertEqual(summary["hardcap"], "9")
        self.assertEqual(summary["cdd_scales"], "4")
        self.assertEqual(summary["energy"], "   3.5000")


if __name__ == "__main__":
    unittest.main()
