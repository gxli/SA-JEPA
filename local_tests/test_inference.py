#!/usr/bin/env python3
"""Local API inference smoke test.

Example:
    python local_tests/test_inference.py \
        --results-dir sessions/gen_186_mhd_run_002_ms1p2_pooled \
        --input data/C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy \
        --inference-dir local_tests/inference_output
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.api import ScaleAwareJEPA


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the source-results-dir + new-.npy -> inference-dir pipeline, "
            "then build the dashboard by loading the inference dir."
        )
    )
    parser.add_argument(
        "--results-dir",
        "--session",
        dest="results_dir",
        default=os.environ.get("SAJEPA_TEST_SESSION"),
        help="Trained results/session directory",
    )
    parser.add_argument("--input", default=os.environ.get("SAJEPA_TEST_NPY"), help="Input .npy file")
    parser.add_argument(
        "--inference-dir",
        "--out",
        dest="inference_dir",
        default=os.path.join(ROOT, "local_tests", "inference_output"),
        help="Output inference-only session directory",
    )
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--crop-mode", default="center", choices=("center", "tile"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-mask-inference", action="store_true")
    args = parser.parse_args()

    if not args.results_dir or not os.path.isdir(args.results_dir):
        parser.error("--results-dir must point to a trained results/session directory")
    if not args.input or not os.path.exists(args.input):
        parser.error("--input must point to an .npy file")

    out_dir = ScaleAwareJEPA.infer_from_session(
        args.results_dir,
        args.input,
        args.inference_dir,
        crop_size=args.crop_size,
        crop_mode=args.crop_mode,
        batch_size=args.batch_size,
        device=args.device,
        mask_inference=not args.no_mask_inference,
        make_dashboard=False,
    )

    # The dashboard stage must be able to treat the output as a standalone
    # inference-only session. This is the production path for arbitrary new data.
    inference = ScaleAwareJEPA.load_session(out_dir)
    inference.generate_dashboard(os.path.join(out_dir, "dashboard.html"))

    expected = [
        "config_used.json",
        "inference_outputs.pt",
        "dashboard.html",
        os.path.join("results", "predict_umap_xyz.npy"),
        os.path.join("results", "predict_pca_xyz.npy"),
    ]
    missing = [name for name in expected if not os.path.exists(os.path.join(out_dir, name))]
    if missing:
        raise RuntimeError(f"Missing expected inference artifacts in {out_dir}: {missing}")

    outputs = torch.load(os.path.join(out_dir, "inference_outputs.pt"), map_location="cpu", weights_only=False)
    print(f"[test_inference] source_results_dir={args.results_dir}")
    print(f"[test_inference] inference_dir={out_dir}")
    print(f"[test_inference] pred_map={tuple(outputs['pred_map'].shape)}")
    print(f"[test_inference] dashboard={os.path.join(out_dir, 'dashboard.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
