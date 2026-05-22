from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.cdd_inspect import CDDOperatorFeatures2D


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load CDD .npy and print CDD operator feature tensors."
    )
    p.add_argument("input_npy", type=str, help="Path to input .npy")
    p.add_argument(
        "--features",
        type=str,
        default="x,gradmag,abslap,local_std",
        help="Comma-separated feature list for stack",
    )
    p.add_argument("--local-std-kernel", type=int, default=7)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--padding-mode", type=str, default="reflect")
    p.add_argument("--normalize-stack", action="store_true")
    p.add_argument("--apply-lognorm", action="store_true")
    p.add_argument("--log-eps", type=float, default=1e-30)
    p.add_argument("--log-std-floor-mult", type=float, default=0.05)
    p.add_argument("--unified-lognorm", action="store_true")
    p.add_argument(
        "--lognorm-mode",
        type=str,
        default="auto",
        choices=["auto", "positive", "signed"],
        help="auto: signed for grad_x/grad_y/lap, positive for others",
    )
    p.add_argument(
        "--expect-3d-pyramid",
        action="store_true",
        default=True,
        help="Expect B,S,D,H,W input (default true).",
    )
    p.add_argument(
        "--no-expect-3d-pyramid",
        dest="expect_3d_pyramid",
        action="store_false",
        help="Allow 2D input B,S,H,W without 3D expectation.",
    )
    p.add_argument(
        "--print-all",
        action="store_true",
        help="Print full tensor values for each feature map.",
    )
    return p.parse_args()


def _to_batched_cdd(x: np.ndarray) -> np.ndarray:
    # Accepts:
    # H,W -> 1,1,H,W
    # S,H,W -> 1,S,H,W
    # B,S,H,W -> unchanged
    # S,D,H,W -> 1,S,D,H,W
    # B,S,D,H,W -> unchanged
    if x.ndim == 2:
        return x[None, None, ...]
    if x.ndim == 3:
        return x[None, ...]
    if x.ndim == 4:
        return x
    if x.ndim == 5:
        return x
    raise ValueError(f"Unsupported input shape {x.shape}; expected 3D/4D/5D array.")


def _print_tensor_summary(name: str, t: torch.Tensor) -> None:
    x = t.detach().float()
    finite = torch.isfinite(x)
    finite_ratio = float(finite.float().mean().item())
    x_safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    q = torch.quantile(x_safe.reshape(-1), torch.tensor([0.01, 0.5, 0.99]))
    print(f"[{name}]")
    print(f"  shape={tuple(x.shape)}")
    print(
        "  min={:.6g} max={:.6g} mean={:.6g} std={:.6g} p01={:.6g} p50={:.6g} p99={:.6g} finite={:.6f}".format(
            float(x_safe.min().item()),
            float(x_safe.max().item()),
            float(x_safe.mean().item()),
            float(x_safe.std(unbiased=False).item()),
            float(q[0].item()),
            float(q[1].item()),
            float(q[2].item()),
            finite_ratio,
        )
    )


def main() -> None:
    args = _parse_args()
    arr = np.asarray(np.load(args.input_npy), dtype=np.float32)
    arr = _to_batched_cdd(arr)

    cdd = torch.from_numpy(arr).float()
    feature_names = tuple(s.strip() for s in args.features.split(",") if s.strip())

    op = CDDOperatorFeatures2D(
        features=feature_names,
        local_std_kernel=args.local_std_kernel,
        eps=args.eps,
        padding_mode=args.padding_mode,
        normalize_stack=args.normalize_stack,
        expect_3d_pyramid=args.expect_3d_pyramid,
        apply_lognorm=args.apply_lognorm,
        log_eps=args.log_eps,
        log_std_floor_mult=args.log_std_floor_mult,
        lognorm_mode=args.lognorm_mode,
        unified_lognorm=args.unified_lognorm,
    )

    attrs = op(cdd)
    print("input_shape:", tuple(cdd.shape))
    print("features_for_stack:", feature_names)

    for name in ["x", "grad_x", "grad_y", "gradmag", "lap", "abslap", "local_mean", "local_std", "stack"]:
        _print_tensor_summary(name, attrs[name])
        if args.print_all:
            print(attrs[name].detach().cpu().numpy())


if __name__ == "__main__":
    main()
