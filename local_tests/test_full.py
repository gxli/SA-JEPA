#!/usr/bin/env python3
"""Reproduce gen_186 MHD run 002 (ms=1.2 pooled) through the public API.

This intentionally uses an inline API override dict. The API supplies the
canonical default config, and this file only overrides the gen_186 run-002
differences from that default.

Example:
    python local_tests/test_full.py
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, "local_tests", ".mplconfig"))
sys.path.insert(0, ROOT)

import torch

from sajepa import ScaleAwareJEPA


GEN_186_MHD_RUN_002_OVERRIDES = {
    "data": {
        "data_root": os.path.join(ROOT, "data"),
        "npy_pattern": "C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy",
        "num_samples": 200,
        "d4_augment": True,
    },
    "model": {
        "mask_size_scaling": 1.2,
    },
    "train": {
        "epochs": 10,
        "spread_regularizer": {
            "type": "std_hinge",
            "target": "context",
            "spatial_mode": "pooled",
            "weight": 5.0,
            "target_std": 1.0,
            "eps": 0.0001,
        },
    },
}


def _ok(msg: str) -> None:
    print(f"  OK   {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Full API reproduction for gen_186_mhd_run_002_ms1p2_pooled.")
    parser.add_argument("--sessions-dir", default=os.path.join(ROOT, "local_tests", "sessions_full"))
    parser.add_argument("--name", default="gen_186_mhd_run_002_ms1p2_pooled_api")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for a quicker smoke run.")
    parser.add_argument("--num-samples", type=int, default=None, help="Override data.num_samples for a quicker smoke run.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override training.num_workers.")
    parser.add_argument(
        "--cdd-precompute",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto keeps CDD precompute on when CUDA/MPS is available, off on CPU-only machines.",
    )
    parser.add_argument("--no-dashboard", action="store_true")
    args = parser.parse_args()

    cfg = copy.deepcopy(GEN_186_MHD_RUN_002_OVERRIDES)
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.num_samples is not None:
        cfg.setdefault("data", {})["num_samples"] = int(args.num_samples)
    if args.num_workers is not None:
        cfg.setdefault("train", {})["num_workers"] = int(args.num_workers)
    has_accel = torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if args.cdd_precompute == "off" or (args.cdd_precompute == "auto" and not has_accel):
        cfg.setdefault("data", {})["cdd_precompute"] = False

    os.makedirs(args.sessions_dir, exist_ok=True)

    t0 = time.time()
    model = ScaleAwareJEPA().train(
        configs=cfg,
        config_name=args.name,
        sessions_dir=args.sessions_dir,
    )
    session_dir = model.session_dir
    _ok(f"trained session={session_dir}")

    rank = model.analyze_rank()
    _ok(
        "rank "
        f"target_effrank={rank.get('target_effrank', '')} "
        f"predictor_effrank={rank.get('predictor_effrank', '')} "
        f"target_part={rank.get('target_part', '')}"
    )

    if not args.no_dashboard:
        dash = os.path.join(session_dir, "dashboard.html")
        model.generate_dashboard(dash)
        _ok(f"dashboard={dash}")

    print(f"  done ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
