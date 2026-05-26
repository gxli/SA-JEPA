#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import JEPADataset
from src.models.build_jepa import make_pyramid_grid_context


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(data_cfg: dict, for_cdd_masking: bool) -> JEPADataset:
    ds_log_transform = bool(data_cfg.get("log_transform", True))
    ds_apply_cdd = True
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
        random_roll_max=int(max(0, data_cfg.get("random_roll_max", 0))),
        d4_augment=bool(data_cfg.get("d4_augment", False)),
    )


def main():
    parser = argparse.ArgumentParser(description="Dry-run deterministic inference mask sweep and plot all target locations")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sessions-dir", type=str, default="sessions")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    session_dir = os.path.join(args.sessions_dir, config_name)
    os.makedirs(session_dir, exist_ok=True)

    blur_mode = str(model_cfg.get("blur_mode", "cdd"))
    ds = build_dataset(data_cfg, for_cdd_masking=(blur_mode == "cdd"))
    x = ds[int(args.sample_index) % len(ds)][0].numpy().astype(np.float32)
    x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    h, w = int(x.shape[0]), int(x.shape[1])

    sigmas = tuple(float(v) for v in model_cfg.get("sigmas", [2, 4, 8, 16]))
    largest_sigma = float(max(sigmas))
    mask_scale = float(model_cfg.get("mask_scale", 1.0))
    spacing_scale = float(model_cfg.get("spacing_scale", 1.5))
    mask_box_size = int(model_cfg.get("mask_box_size", 16))
    max_box = round(largest_sigma * mask_scale + mask_box_size)
    spacing = int(max(1, round(float(max_box) * spacing_scale)))

    half = spacing // 2
    # Centered pixel shifts over one lattice period:
    # dy,dx in [-0.5*spacing, +0.5*spacing) approximately.
    shift_vals = list(range(-half, spacing - half))
    all_shifts = [(dy, dx) for dy in shift_vals for dx in shift_vals]
    total_possible_shifts = len(all_shifts)
    # Always run full centered sweep so coverage matches real deterministic inference.
    shifts = all_shifts

    visit = np.zeros((h, w), dtype=np.float64)
    hard_union = np.zeros((h, w), dtype=np.float64)
    total_points = 0
    shift_points = []
    for dy, dx in shifts:
        _, tloc, _, tvalid, debug = make_pyramid_grid_context(
            x_clean=x_t,
            sigmas=sigmas,
            cell_sizes=tuple(model_cfg.get("cell_sizes", [16, 32, 64, 128])),
            mask_fraction=float(model_cfg.get("mask_fraction", 1.0)),
            box_sigma_mult=float(model_cfg.get("box_sigma_mult", 4.0)),
            mask_scale=mask_scale,
            spacing_scale=spacing_scale,
            full_grid=bool(model_cfg.get("full_grid", True)),
            global_shift=bool(model_cfg.get("global_shift", True)),
            align_scales=bool(model_cfg.get("align_scales", True)),
            mask_box_size=mask_box_size,
            blur_mode=blur_mode,
            cdd_mode=model_cfg.get("cdd_mode", "log"),
            cdd_constrained=bool(model_cfg.get("cdd_constrained", True)),
            cdd_sm_mode=model_cfg.get("cdd_sm_mode", "reflect"),
            mask_fill_mode=model_cfg.get("mask_fill_mode", "zero"),
            dip_sigma_mult=1.0,
            constant_gaussian_sigma=float(model_cfg.get("constant_gaussian_sigma", 1.0)),
            inner_target_size=int(model_cfg.get("patch_size", 2)),
            return_debug=True,
            forced_grid_shift=(int(dy), int(dx)),
            enable_grid_jitter=False,
        )
        pts = tloc[0].cpu().numpy()
        valid = tvalid[0].cpu().numpy().astype(bool)
        for i in range(pts.shape[0]):
            if not bool(valid[i]):
                continue
            yy = int(pts[i, 0])
            xx = int(pts[i, 1])
            if 0 <= yy < h and 0 <= xx < w:
                visit[yy, xx] += 1.0
                shift_points.append((int(dy), int(dx), yy, xx))
                total_points += 1
        hard_union = np.maximum(hard_union, debug["mask_map"][0].cpu().numpy().astype(np.float64))

    out_npy = os.path.join(session_dir, "test_inference_mask_target_locations.npy")
    out_mask_npy = os.path.join(session_dir, "test_inference_mask_hard_union.npy")
    out_html = os.path.join(session_dir, "test_inference_mask_target_locations.html")
    out_shift_html = os.path.join(session_dir, "test_inference_mask_shift_scatter.html")
    np.save(out_npy, visit.astype(np.float32))
    np.save(out_mask_npy, hard_union.astype(np.float32))

    fig = go.Figure(
        go.Heatmap(
            z=visit.astype(np.float32),
            colorscale="Magma",
            zmin=0.0,
            zmax=float(max(1.0, visit.max())),
            colorbar=dict(title="visit count"),
        )
    )
    fig.update_layout(
        title=(
            f"Deterministic Inference Mask Sweep: Target Locations<br>"
            f"config={config_name} shifts={len(shifts)}/{total_possible_shifts} "
            f"spacing={spacing} total_points={total_points}"
        ),
        template="plotly_white",
        height=760,
        width=760,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    fig.update_xaxes(showticklabels=False, constrain="domain")
    fig.update_yaxes(showticklabels=False, scaleanchor="x", scaleratio=1, constrain="domain")
    fig.write_html(out_html, include_plotlyjs="cdn")

    # Per-shift scatter view to verify centers actually move.
    uniq_shifts = []
    seen_s = set()
    for dy, dx, _, _ in shift_points:
        key = (dy, dx)
        if key not in seen_s:
            seen_s.add(key)
            uniq_shifts.append(key)
    traces = []
    for i, (dy, dx) in enumerate(uniq_shifts):
        xs = [p[3] for p in shift_points if p[0] == dy and p[1] == dx]
        ys = [p[2] for p in shift_points if p[0] == dy and p[1] == dx]
        traces.append(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=5),
                name=f"shift({dy},{dx})",
                visible=(i == 0),
            )
        )
    fig_shift = go.Figure(data=traces)
    steps = []
    for i, (dy, dx) in enumerate(uniq_shifts):
        vis = [False] * len(uniq_shifts)
        vis[i] = True
        steps.append(
            dict(
                method="update",
                args=[{"visible": vis}, {"title": f"Per-Shift Target Centers: shift=({dy},{dx})"}],
                label=str(i),
            )
        )
    fig_shift.update_layout(
        title=(f"Per-Shift Target Centers: shift={uniq_shifts[0] if uniq_shifts else '(none)'}"),
        template="plotly_white",
        height=760,
        width=760,
        xaxis=dict(range=[0, w - 1], showticklabels=False),
        yaxis=dict(range=[h - 1, 0], showticklabels=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=70, b=20),
        sliders=[dict(active=0, currentvalue={"prefix": "shift idx: "}, steps=steps)],
    )
    fig_shift.write_html(out_shift_html, include_plotlyjs="cdn")

    covered = int((visit > 0).sum())
    print(f"test_inference_mask_saved_html={out_html}")
    print(f"test_inference_mask_saved_npy={out_npy}")
    print(f"test_inference_mask_saved_union_npy={out_mask_npy}")
    print(f"test_inference_mask_saved_shift_html={out_shift_html}")
    print(
        f"test_inference_mask_summary config={config_name} sample_index={int(args.sample_index)} "
        f"spacing={spacing} shifts={len(shifts)}/{total_possible_shifts} "
        f"shift_min={shift_vals[0]} shift_max={shift_vals[-1]} "
        f"covered_pixels={covered} total_pixels={h*w} total_points={total_points}"
    )


if __name__ == "__main__":
    main()
