from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.cdd_inspect import CDDOperatorFeatures2D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save CDD operator feature images from .npy input.")
    p.add_argument("input_npy", type=str, help="Input .npy with shape S,H,W / B,S,H,W / S,D,H,W / B,S,D,H,W")
    p.add_argument("--out-dir", type=str, default="results/feature_images")
    p.add_argument("--features", type=str, default="x,grad_x,grad_y,gradmag,lap,abslap,local_mean,local_std")
    p.add_argument("--stack-features", type=str, default="x,gradmag,abslap,local_std")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--depth-idx", type=int, default=0, help="Used only for 5D tensors B,C,D,H,W")
    p.add_argument("--max-scales", type=int, default=None)
    p.add_argument("--local-std-kernel", type=int, default=7)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--padding-mode", type=str, default="reflect")
    p.add_argument("--apply-lognorm", action="store_true")
    p.add_argument("--log-eps", type=float, default=1e-30)
    p.add_argument("--log-std-floor-mult", type=float, default=0.05)
    p.add_argument("--unified-lognorm", action="store_true")
    p.add_argument(
        "--lognorm-mode",
        type=str,
        default="auto",
        choices=["auto", "positive", "signed"],
    )
    p.add_argument("--percentile-low", type=float, default=1.0)
    p.add_argument("--percentile-high", type=float, default=99.0)
    p.add_argument("--expect-3d-pyramid", action="store_true", default=True)
    p.add_argument("--no-expect-3d-pyramid", dest="expect_3d_pyramid", action="store_false")
    return p.parse_args()


def to_batched(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, None, ...]  # 1,1,H,W
    if arr.ndim == 3:
        return arr[None, ...]  # 1,S,H,W
    if arr.ndim == 4:
        return arr
    if arr.ndim == 5:
        return arr
    raise ValueError(f"Unsupported input shape {arr.shape}")


def normalize_for_vis(x2d: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    x2d = np.nan_to_num(x2d, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(x2d, p_low)
    hi = np.percentile(x2d, p_high)
    if hi <= lo + 1e-12:
        return np.zeros_like(x2d, dtype=np.float32)
    return np.clip((x2d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    arr = to_batched(np.asarray(np.load(args.input_npy), dtype=np.float32))
    cdd = torch.from_numpy(arr).float()

    op = CDDOperatorFeatures2D(
        features=tuple(x.strip() for x in args.stack_features.split(",") if x.strip()),
        local_std_kernel=args.local_std_kernel,
        eps=args.eps,
        padding_mode=args.padding_mode,
        expect_3d_pyramid=args.expect_3d_pyramid,
        apply_lognorm=args.apply_lognorm,
        log_eps=args.log_eps,
        log_std_floor_mult=args.log_std_floor_mult,
        lognorm_mode=args.lognorm_mode,
        unified_lognorm=args.unified_lognorm,
    )
    attrs = op(cdd)

    names = [x.strip() for x in args.features.split(",") if x.strip()]
    for name in names:
        t = attrs[name].detach().cpu()
        if t.ndim not in (4, 5):
            continue

        if args.sample_idx < 0 or args.sample_idx >= t.shape[0]:
            raise ValueError(f"sample_idx out of range: {args.sample_idx} vs batch={t.shape[0]}")

        if t.ndim == 4:
            # B,S,H,W
            n_scales = t.shape[1] if args.max_scales is None else min(t.shape[1], args.max_scales)
            for s in range(n_scales):
                vis = normalize_for_vis(
                    t[args.sample_idx, s].numpy(), args.percentile_low, args.percentile_high
                )
                path = os.path.join(args.out_dir, f"{name}_sample{args.sample_idx:02d}_scale{s:02d}.png")
                plt.imsave(path, vis, cmap="gray")
        else:
            # B,S,D,H,W
            if args.depth_idx < 0 or args.depth_idx >= t.shape[2]:
                raise ValueError(f"depth_idx out of range: {args.depth_idx} vs depth={t.shape[2]}")
            n_scales = t.shape[1] if args.max_scales is None else min(t.shape[1], args.max_scales)
            for s in range(n_scales):
                vis = normalize_for_vis(
                    t[args.sample_idx, s, args.depth_idx].numpy(),
                    args.percentile_low,
                    args.percentile_high,
                )
                path = os.path.join(
                    args.out_dir,
                    f"{name}_sample{args.sample_idx:02d}_scale{s:02d}_depth{args.depth_idx:03d}.png",
                )
                plt.imsave(path, vis, cmap="gray")

    print(f"Saved feature images to: {args.out_dir}")


if __name__ == "__main__":
    main()
