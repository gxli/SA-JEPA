#!/usr/bin/env python3
"""Create a deterministic MHD slice with a few positive Gaussian contaminants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy"
DEFAULT_OUTPUT = ROOT / "data" / "C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5_gaussian_contaminated.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=21812)
    parser.add_argument("--num-gaussians", type=int, default=6)
    parser.add_argument("--amp-low-std", type=float, default=8.0)
    parser.add_argument("--amp-high-std", type=float, default=20.0)
    parser.add_argument("--sigma-low-px", type=float, default=5.0)
    parser.add_argument("--sigma-high-px", type=float, default=18.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output

    arr = np.load(source)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D MHD slice, got shape {arr.shape} from {source}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Source contains non-finite values: {source}")

    base = arr.astype(np.float64, copy=False)
    contaminated = base.copy()
    finite_std = float(np.std(base))
    if finite_std <= 0.0:
        raise ValueError(f"Source has zero standard deviation: {source}")

    rng = np.random.default_rng(args.seed)
    h, w = base.shape
    yy, xx = np.mgrid[0:h, 0:w]
    gaussians = []
    margin = int(np.ceil(args.sigma_high_px * 2.0))

    for _ in range(args.num_gaussians):
        cx = float(rng.uniform(margin, max(margin + 1, w - margin)))
        cy = float(rng.uniform(margin, max(margin + 1, h - margin)))
        sigma = float(rng.uniform(args.sigma_low_px, args.sigma_high_px))
        amp = float(rng.uniform(args.amp_low_std, args.amp_high_std) * finite_std)
        bump = amp * np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))
        contaminated += bump
        gaussians.append({"cx": cx, "cy": cy, "sigma_px": sigma, "amplitude": amp})

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, contaminated.astype(arr.dtype, copy=False))

    meta = {
        "source": str(source.relative_to(ROOT)),
        "output": str(output.relative_to(ROOT)),
        "seed": args.seed,
        "num_gaussians": args.num_gaussians,
        "source_min": float(np.min(base)),
        "source_max": float(np.max(base)),
        "source_mean": float(np.mean(base)),
        "source_std": finite_std,
        "output_min": float(np.min(contaminated)),
        "output_max": float(np.max(contaminated)),
        "output_mean": float(np.mean(contaminated)),
        "output_std": float(np.std(contaminated)),
        "gaussians": gaussians,
    }
    meta_path = output.with_suffix(output.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {output}")
    print(f"wrote {meta_path}")
    print(
        "stats "
        f"source_max={meta['source_max']:.6e} output_max={meta['output_max']:.6e} "
        f"source_std={meta['source_std']:.6e} output_std={meta['output_std']:.6e}"
    )


if __name__ == "__main__":
    main()
