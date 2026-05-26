#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import JEPADataset
from src.models.masking import prepare_context_batch


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(data_cfg: dict, for_cdd_masking: bool = False) -> JEPADataset:
    ds_log_transform = bool(data_cfg.get("log_transform", True))
    # Keep demo input in raw image space by default to avoid CDD dependency path.
    ds_apply_cdd = False
    if for_cdd_masking:
        ds_log_transform = False
        ds_apply_cdd = False
    return JEPADataset(
        num_samples=max(1, int(data_cfg.get("num_samples", 1))),
        image_size=int(data_cfg.get("image_size", 256)),
        data_root=data_cfg.get("data_root", "data"),
        npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
        log_transform=ds_log_transform,
        log_eps=float(data_cfg.get("log_eps", 1.0)),
        cdd_scales=data_cfg.get("cdd_scales", [2, 4, 8]),
        cdd_strength=float(data_cfg.get("cdd_strength", 1.0)),
        cdd_clip=bool(data_cfg.get("cdd_clip", True)),
        norm_before_cdd=bool(data_cfg.get("norm_before_cdd", True)),
        cdd_mode=data_cfg.get("cdd_mode", "log"),
        cdd_constrained=bool(data_cfg.get("cdd_constrained", True)),
        cdd_sm_mode=data_cfg.get("cdd_sm_mode", "reflect"),
        apply_cdd=ds_apply_cdd,
        cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
        cube_slice_axis=int(data_cfg.get("cube_slice_axis", 0)),
        cube_slice_index=int(data_cfg.get("cube_slice_index", 0)),
    )


def build_context(x_t: torch.Tensor, cfg: dict):
    m = cfg.get("model", {})
    force_blur_mode = str(m.get("_force_blur_mode", "")).strip().lower()
    blur_mode = force_blur_mode if force_blur_mode in {"gaussian", "cdd"} else str(m.get("blur_mode", "cdd"))
    return prepare_context_batch(
        x_clean=x_t,
        sigmas=tuple(m.get("sigmas", [2, 4, 8, 16])),
        cell_sizes=tuple(m.get("cell_sizes", [16, 32, 64, 128])),
        mask_fraction=float(m.get("mask_fraction", 1.0)),
        box_sigma_mult=float(m.get("box_sigma_mult", 4.0)),
        mask_scale=float(m.get("mask_scaling_box", m.get("mask_scale", 1.0))),
        spacing_scale=float(m.get("mask_spacing_scaling", m.get("spacing_scale", 1.5))),
        mask_size=float(m.get("mask_size", 0.0)),
        full_grid=bool(m.get("full_grid", True)),
        global_shift=bool(m.get("global_shift", True)),
        align_scales=bool(m.get("align_scales", True)),
        mask_box_size=int(m.get("mask_box_size", 16)),
        blur_mode=blur_mode,
        cdd_mode=str(m.get("cdd_mode", "log")),
        cdd_constrained=bool(m.get("cdd_constrained", True)),
        cdd_sm_mode=str(m.get("cdd_sm_mode", "reflect")),
        mask_fill_mode=str(m.get("mask_fill_mode", "zero")),
        dip_sigma_mult=1.0,
        constant_gaussian_sigma=float(m.get("constant_gaussian_sigma", 1.0)),
        scaleaware_gaussian_ratios=tuple(m.get("scaleaware_gaussian_ratios", [0.25, 0.5, 1.0, 2.0])),
        cdd_append_last_residual=bool(m.get("cdd_append_last_residual", True)),
        patch_size=int(m.get("patch_size", 3)),
        return_debug=True,
        target_invalid_region_skip=bool(m.get("target_invalid_region_skip", False)),
        target_invalid_region_values=tuple(m.get("target_invalid_region_values", (0.0, "nan"))),
        target_sampling_mode=str(m.get("target_sampling_mode", "grid")),
        priority_top_percent=float(m.get("priority_top_percent", 5.0)),
        priority_n_target=int(m.get("priority_n_target", 20)),
        target_dithering_pixels=int(m.get("target_dithering_pixels", 6)),
        cdd_use_gpu=False,
    )


def extract_centers(target_locations: torch.Tensor, target_valid: torch.Tensor):
    loc = target_locations[0].cpu().numpy()
    val = target_valid[0].cpu().numpy().astype(bool)
    pts = [(int(loc[i, 0]), int(loc[i, 1])) for i in range(loc.shape[0]) if val[i]]
    if not pts:
        raise RuntimeError("No valid targets")
    return pts


def stamp_target_mask(z: np.ndarray, target_mask: np.ndarray, value: float = -2.0) -> np.ndarray:
    out = np.array(z, dtype=np.float32, copy=True)
    out[np.asarray(target_mask, dtype=np.float32) > 0.5] = float(value)
    return out


def contour_trace(z: np.ndarray, color: str = "red"):
    contour_src = np.nan_to_num(np.asarray(z, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return go.Contour(
        z=contour_src,
        contours=dict(start=0.5, end=0.5, size=1.0, coloring="none"),
        line=dict(color=color, width=2),
        showscale=False,
        showlegend=False,
        hoverinfo="skip",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--crop", type=int, default=16)
    ap.add_argument("--center-box", type=int, default=3)
    ap.add_argument("--binarize-mask", action="store_true")
    ap.add_argument("--force-blur-mode", choices=["gaussian", "cdd"], default=None)
    ap.add_argument("--mask-fraction", type=float, default=None)
    ap.add_argument("--mask-scale", type=float, default=None)
    ap.add_argument("--mask-box-size", type=int, default=None)
    ap.add_argument("--cols", type=int, default=1)
    ap.add_argument("--panel-px", type=int, default=220)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.force_blur_mode is not None:
        cfg.setdefault("model", {})["_force_blur_mode"] = str(args.force_blur_mode)
    if args.mask_fraction is not None:
        cfg.setdefault("model", {})["mask_fraction"] = float(args.mask_fraction)
    if args.mask_scale is not None:
        cfg.setdefault("model", {})["mask_scale"] = float(args.mask_scale)
        cfg.setdefault("model", {})["mask_scaling_box"] = float(args.mask_scale)
    if args.mask_box_size is not None:
        cfg.setdefault("model", {})["mask_box_size"] = int(args.mask_box_size)
    eff_blur_mode = str(cfg.get("model", {}).get("_force_blur_mode", cfg.get("model", {}).get("blur_mode", "cdd")))
    ds = build_dataset(cfg.get("data", {}), for_cdd_masking=(eff_blur_mode == "cdd"))
    x = ds[int(args.sample_index) % len(ds)][0].numpy().astype(np.float32)
    x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)

    _x_ctx, target_locations, _target_scales, target_valid, debug = build_context(x_t, cfg)
    dip_t = debug.get("dip_field_per_channel")
    if dip_t is not None and dip_t.numel() > 0:
        dip = dip_t[0].cpu().numpy().astype(np.float32)
    else:
        # Non-CDD fallback: use hard mask map as a single-channel diagnostic.
        mm = debug.get("mask_map")
        if mm is None or mm.numel() == 0:
            raise RuntimeError("Missing both dip_field_per_channel and mask_map")
        mm_np = mm[0].cpu().numpy().astype(np.float32)
        if mm_np.ndim == 3:
            mm_np = mm_np[0]
        dip = np.expand_dims(np.clip(mm_np, 0.0, 1.0), axis=0)
    if args.binarize_mask:
        dip = (dip > 1e-6).astype(np.float32)
    agg = dip.max(axis=0)
    m = cfg.get("model", {})

    centers = extract_centers(target_locations, target_valid)
    cy, cx = centers[len(centers) // 2]

    h, w = agg.shape
    target_mask = np.zeros((h, w), dtype=np.float32)
    target_patch = max(1, int(m.get("patch_size", 3)))
    if target_patch % 2 == 0:
        target_patch += 1
    target_half_lo = target_patch // 2
    target_half_hi = target_patch - target_half_lo
    for ty, tx in centers:
        ty0 = max(0, int(ty) - target_half_lo)
        ty1 = min(h, int(ty) + target_half_hi)
        tx0 = max(0, int(tx) - target_half_lo)
        tx1 = min(w, int(tx) + target_half_hi)
        if ty1 > ty0 and tx1 > tx0:
            target_mask[ty0:ty1, tx0:tx1] = 1.0

    hc = max(1, int(args.crop) // 2)
    y0, y1 = max(0, cy - hc), min(h, cy + hc)
    x0, x1 = max(0, cx - hc), min(w, cx + hc)
    if (y1 - y0) < int(args.crop):
        y0 = max(0, min(y0, h - int(args.crop))); y1 = min(h, y0 + int(args.crop))
    if (x1 - x0) < int(args.crop):
        x0 = max(0, min(x0, w - int(args.crop))); x1 = min(w, x0 + int(args.crop))

    target_crop = target_mask[y0:y1, x0:x1]
    agg = agg[y0:y1, x0:x1]
    dip = dip[:, y0:y1, x0:x1]

    n_ch = int(dip.shape[0])
    box_sizes_t = debug.get("cdd_box_sizes")
    if box_sizes_t is not None and box_sizes_t.numel() > 0:
        box_sizes = [int(round(float(v))) for v in box_sizes_t[0].detach().cpu().flatten().tolist()]
    else:
        fallback = max(1, int(args.center_box))
        if fallback % 2 == 0:
            fallback += 1
        box_sizes = [fallback] * n_ch
    if len(box_sizes) < n_ch:
        box_sizes = (box_sizes + [box_sizes[-1] if box_sizes else 3] * n_ch)[:n_ch]
    box_sizes = [b + 1 if b % 2 == 0 else b for b in box_sizes[:n_ch]]

    # Channel plots disabled (gaussian dip removed).
    # panel_count = n_ch + 1
    panel_count = 1
    cols = max(1, int(args.cols))
    cols = min(cols, panel_count)
    rows = int(math.ceil(panel_count / cols))
    titles = [f"agg ({len(centers)} targets, patch={target_patch})"]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.045 if rows > 1 else 0.01,
        horizontal_spacing=0.035 if cols > 1 else 0.01,
    )

    def panel_rc(panel_idx: int) -> tuple[int, int]:
        return panel_idx // cols + 1, panel_idx % cols + 1

    def add(z, contour_z, panel_idx, box_size, scale=False, colorscale="Viridis", zmin=0.0, zmax=1.0):
        r, c = panel_rc(panel_idx)
        fig.add_trace(go.Heatmap(z=z, colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=scale), row=r, col=c)
        if contour_z is not None:
            fig.add_trace(contour_trace(contour_z), row=r, col=c)
        axis_idx = panel_idx + 1
        x_axis = "x" if axis_idx == 1 else f"x{axis_idx}"
        fig.update_xaxes(row=r, col=c, showticklabels=False)
        fig.update_yaxes(
            row=r,
            col=c,
            showticklabels=False,
            autorange="reversed",
            scaleanchor=x_axis,
            scaleratio=1,
        )

    energy_marker_colorscale = [[0.0, "#ff3b30"], [0.6666667, "#440154"], [1.0, "#fde725"]]

    target_contour = target_crop.astype(np.float32)
    agg_plot = stamp_target_mask(agg, target_contour, value=-2.0)
    dip_plot = [stamp_target_mask(dip[i], target_contour, value=-2.0) for i in range(n_ch)]

    add(
        agg_plot,
        target_contour,
        0,
        max(box_sizes) if box_sizes else max(1, int(args.center_box)),
        scale=True,
        colorscale=energy_marker_colorscale,
        zmin=-2.0,
        zmax=1.0,
    )
    for i in range(n_ch):
        add(dip_plot[i], target_contour, i + 1, box_sizes[i], colorscale=energy_marker_colorscale, zmin=-2.0, zmax=1.0)

    title = (
        f"Mask Diagnostic: {os.path.basename(args.config)} | "
        f"blur={eff_blur_mode} "
        f"mask_box={m.get('mask_box_size')} "
        f"mask_scale={m.get('mask_scale', m.get('mask_scaling_box'))} "
        f"mask_fraction={m.get('mask_fraction')} boxes={box_sizes}"
    )
    panel_px = max(160, int(args.panel_px))
    fig.update_layout(
        height=max(panel_px + 120, panel_px * rows + 120),
        width=max(360, panel_px * cols + 130),
        title=title,
        plot_bgcolor="#050505",
    )
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    meta = {
        "config": os.path.abspath(args.config),
        "pipeline_entry": "src.models.masking.prepare_context_batch",
        "masking_function": "src.models.masking.make_pyramid_grid_context",
        "blur_mode": eff_blur_mode,
        "mask_box_size": int(m.get("mask_box_size", 0)),
        "mask_scale": float(m.get("mask_scale", m.get("mask_scaling_box", 1.0))),
        "mask_fraction": float(m.get("mask_fraction", 1.0)),
        "target_sampling_mode": str(m.get("target_sampling_mode", "grid")),
        "priority_n_target": int(m.get("priority_n_target", 20)),
        "valid_target_count": int(target_valid[0].sum().item()),
        "target_patch_size": target_patch,
        "target_patch_value_in_energy_channels": -2.0,
        "contour_source": "target_patch_mask_from_target_locations",
        "layout_cols": cols,
        "cdd_box_sizes": box_sizes,
    }
    meta_path = os.path.splitext(out_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(out_path)
    print(meta_path)


if __name__ == "__main__":
    main()
