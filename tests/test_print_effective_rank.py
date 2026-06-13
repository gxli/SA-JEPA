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
            for name, energy, sim0, sim1, spread0, spread1, weighted0, weighted1 in (
                ("run_a", 1.25, 2.0, 1.0, 4.0, 2.0, 5.0, 1.0),
                ("run_b", 9.50, 3.0, 12.0, 2.0, 6.0, 7.0, 14.0),
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
                            "epoch,sim,loss_spread,weighted_spread",
                            f"0,{sim0},{spread0},{weighted0}",
                            f"1,{sim1},{spread1},{weighted1}",
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
            finally:
                sys.argv = old_argv

        lines = out.getvalue().splitlines()
        run_a = next(line for line in lines if line.startswith("run_a "))
        run_b = next(line for line in lines if line.startswith("run_b "))

        self.assertIn("   1.2500  0.5000  0.5000  0.2000", run_a)
        self.assertIn("   9.5000  4.0000  3.0000  2.0000", run_b)
        self.assertNotEqual(run_a.split()[13:17], run_b.split()[13:17])

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
        self.assertEqual(summary["energy"], "   3.5000")


if __name__ == "__main__":
    unittest.main()
