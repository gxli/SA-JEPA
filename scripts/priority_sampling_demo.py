#!/usr/bin/env python3
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import JEPADataset
from src.models.masking import make_pyramid_grid_context


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_dataset(data_cfg: dict) -> JEPADataset:
    # Keep demo aligned with cdd-mode training: no dataset CDD, no pre-log.
    return JEPADataset(
        num_samples=max(1, int(data_cfg.get("num_samples", 1))),
        image_size=int(data_cfg.get("image_size", 256)),
        data_root=data_cfg.get("data_root", "data"),
        npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
        log_transform=False,
        log_eps=float(data_cfg.get("log_eps", 1.0)),
        cdd_scales=data_cfg.get("cdd_scales", [2, 4, 8, 16]),
        cdd_strength=float(data_cfg.get("cdd_strength", 1.0)),
        cdd_clip=bool(data_cfg.get("cdd_clip", True)),
        norm_before_cdd=bool(data_cfg.get("norm_before_cdd", True)),
        cdd_mode=data_cfg.get("cdd_mode", "log"),
        cdd_constrained=bool(data_cfg.get("cdd_constrained", True)),
        cdd_sm_mode=data_cfg.get("cdd_sm_mode", "reflect"),
        apply_cdd=False,
        cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
        cube_slice_axis=int(data_cfg.get("cube_slice_axis", 0)),
        cube_slice_index=int(data_cfg.get("cube_slice_index", 0)),
    )


def _to_np01(x: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))
    if hi <= lo + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Priority sampling masking demo")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-percent", type=float, default=None)
    ap.add_argument("--n-target", type=int, default=None)
    ap.add_argument("--out", type=str, default="priority_sampling_demo.png")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    model_cfg = dict(cfg.get("model", {}))
    data_cfg = dict(cfg.get("data", {}))
    model_cfg.setdefault("target_sampling_mode", "priority_sampliyg")
    if args.top_percent is not None:
        model_cfg["priority_top_percent"] = float(args.top_percent)
    if args.n_target is not None:
        model_cfg["priority_n_target"] = int(args.n_target)

    ds = _build_dataset(data_cfg)
    x = ds[int(args.sample_index)].unsqueeze(0)  # 1x1xHxW

    torch.manual_seed(int(args.seed))
    x_ctx, tloc, _, tvalid, debug = make_pyramid_grid_context(
        x_clean=x,
        sigmas=tuple(model_cfg.get("sigmas", [2, 4, 8, 16])),
        cell_sizes=tuple(model_cfg.get("cell_sizes", [16, 32, 64, 128])),
        mask_fraction=float(model_cfg.get("mask_fraction", 1.0)),
        box_sigma_mult=float(model_cfg.get("box_sigma_mult", 4.0)),
        mask_scale=float(model_cfg.get("mask_scale", 1.0)),
        spacing_scale=float(model_cfg.get("spacing_scale", 2.0)),
        mask_size=float(model_cfg.get("mask_size", 0.0)),
        full_grid=bool(model_cfg.get("full_grid", True)),
        global_shift=bool(model_cfg.get("global_shift", True)),
        align_scales=bool(model_cfg.get("align_scales", True)),
        mask_box_size=int(model_cfg.get("mask_box_size", 16)),
        blur_mode=str(model_cfg.get("blur_mode", "cdd")),
        cdd_mode=str(model_cfg.get("cdd_mode", "log")),
        cdd_constrained=bool(model_cfg.get("cdd_constrained", True)),
        cdd_sm_mode=str(model_cfg.get("cdd_sm_mode", "reflect")),
        mask_fill_mode=str(model_cfg.get("mask_fill_mode", "gaussian_dip")),
        dip_sigma_mult=1.0,
        scaleaware_gaussian_ratios=tuple(model_cfg.get("scaleaware_gaussian_ratios", [0.25, 0.5, 1.0, 2.0])),
        cdd_append_last_residual=bool(model_cfg.get("cdd_append_last_residual", True)),
        inner_target_size=int(model_cfg.get("patch_size", 2)),
        return_debug=True,
        target_sampling_mode=str(model_cfg.get("target_sampling_mode", "priority_sampliyg")),
        priority_top_percent=float(model_cfg.get("priority_top_percent", 5.0)),
        priority_n_target=int(model_cfg.get("priority_n_target", 20)),
    )

    x_np = x[0, 0].cpu().numpy()
    ctx_np = x_ctx[0, 0].cpu().numpy()
    cdd_orig = debug["cdd_channels_orig"][0].cpu().numpy()
    small = cdd_orig[0] + cdd_orig[1] if cdd_orig.shape[0] > 1 else cdd_orig[0]
    ratio = small / np.maximum(np.sum(cdd_orig, axis=0), 1e-8)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

    loc = tloc[0].cpu().numpy()
    valid = tvalid[0].cpu().numpy().astype(bool)
    pts = loc[valid]

    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    axes[0].imshow(_to_np01(x_np), cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(_to_np01(ratio), cmap="magma")
    axes[1].set_title("Priority Ratio")
    axes[1].axis("off")
    if pts.shape[0] > 0:
        axes[1].scatter(pts[:, 1], pts[:, 0], s=12, c="cyan", marker="x")

    axes[2].imshow(_to_np01(ctx_np), cmap="gray")
    axes[2].set_title("Masked Context")
    axes[2].axis("off")

    delta = ctx_np - x_np
    lim = float(np.percentile(np.abs(delta), 99.0))
    lim = max(lim, 1e-8)
    im = axes[3].imshow(delta, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[3].set_title("Context - Input")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close(fig)

    print(
        json.dumps(
            {
                "out": args.out,
                "target_sampling_mode": str(model_cfg.get("target_sampling_mode", "priority_sampliyg")),
                "priority_top_percent": float(model_cfg.get("priority_top_percent", 5.0)),
                "priority_n_target": int(model_cfg.get("priority_n_target", 20)),
                "n_valid_targets": int(valid.sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
