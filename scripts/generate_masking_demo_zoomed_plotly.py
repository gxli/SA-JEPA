#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.masking_demo import build_dataset, extract_centers_from_targets, load_config, make_context_and_debug
from src.train import build_model_from_config


def _zoom_panel_from_config(config_path: str, sample_index: int, crop: int, center_box: int, binarize_mask: bool):
    cfg = load_config(config_path)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    ds = build_dataset(data_cfg)
    x = ds[int(sample_index) % len(ds)][0].numpy().astype(np.float32)
    x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)

    x_ctx, target_locations, target_scales, target_valid, debug = make_context_and_debug(x_t, model_cfg, seed=42)
    centers = extract_centers_from_targets(target_locations, target_valid)
    if centers.size == 0:
        raise RuntimeError(f"No valid target centers from config: {config_path}")

    model = build_model_from_config(model_cfg, data_cfg, cfg.get("train", {}), torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        out = model(x_t, context_data=(x_ctx, target_locations, target_scales, target_valid, debug))

    h_enc = int(out["pred_map"].shape[-2])
    w_enc = int(out["pred_map"].shape[-1])
    h_img = int(x_t.shape[-2])
    w_img = int(x_t.shape[-1])

    dip_t = debug.get("dip_field_per_channel")
    if dip_t is None or dip_t.numel() == 0:
        raise RuntimeError(f"dip_field_per_channel missing for config: {config_path}")

    dip = F.interpolate(dip_t[:1], size=(h_enc, w_enc), mode="nearest")[0].cpu().numpy()

    if binarize_mask:
        mask_agg = (dip.max(axis=0) > 1e-6).astype(np.float32)
        dip = (dip > 1e-6).astype(np.float32)
    else:
        mask_agg = dip.max(axis=0).astype(np.float32)

    ctr = np.array([h_img / 2.0, w_img / 2.0], dtype=np.float32)
    best = min(centers.tolist(), key=lambda p: (p[0] - ctr[0]) ** 2 + (p[1] - ctr[1]) ** 2)
    cy, cx = int(best[0]), int(best[1])

    ye = int(np.clip(np.floor((float(cy) + 0.5) * float(h_enc) / float(h_img)), 0, h_enc - 1))
    xe = int(np.clip(np.floor((float(cx) + 0.5) * float(w_enc) / float(w_img)), 0, w_enc - 1))

    cbox = max(1, int(center_box))
    if cbox % 2 == 0:
        cbox += 1
    half = cbox // 2

    half_crop = max(1, int(crop) // 2)
    y0, y1 = max(0, ye - half_crop), min(h_enc, ye + half_crop)
    x0, x1 = max(0, xe - half_crop), min(w_enc, xe + half_crop)
    if (y1 - y0) < int(crop):
        if y0 == 0:
            y1 = min(h_enc, int(crop))
        elif y1 == h_enc:
            y0 = max(0, h_enc - int(crop))
    if (x1 - x0) < int(crop):
        if x0 == 0:
            x1 = min(w_enc, int(crop))
        elif x1 == w_enc:
            x0 = max(0, w_enc - int(crop))

    dip_zoom = dip[:, y0:y1, x0:x1].astype(np.float32)
    agg_zoom = mask_agg[y0:y1, x0:x1].astype(np.float32)

    h_z, w_z = (y1 - y0), (x1 - x0)
    outline = np.zeros((h_z, w_z), dtype=np.float32)
    cyz, cxz = (ye - y0), (xe - x0)
    yy0z, yy1z = max(0, cyz - half), min(h_z, cyz + half + 1)
    xx0z, xx1z = max(0, cxz - half), min(w_z, cxz + half + 1)
    if yy1z > yy0z and xx1z > xx0z:
        outline[yy0z:yy1z, xx0z] = 1.0
        outline[yy0z:yy1z, xx1z - 1] = 1.0
        outline[yy0z, xx0z:xx1z] = 1.0
        outline[yy1z - 1, xx0z:xx1z] = 1.0

    return agg_zoom, dip_zoom, outline


def add_panel(fig, z, outline, row, col, title, showscale=False):
    fig.add_trace(
        go.Heatmap(
            z=z,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            showscale=showscale,
            colorbar=dict(title="mask") if showscale else None,
        ),
        row=row,
        col=col,
    )
    ys, xs = np.where(outline > 0.5)
    if len(xs) > 0:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(color="red", size=4),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.update_xaxes(title_text="x", row=row, col=col)
    fig.update_yaxes(title_text="y", row=row, col=col, autorange="reversed", scaleanchor=f"x{(row-1)*2+col}", scaleratio=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--crop", type=int, default=16)
    ap.add_argument("--center-box", type=int, default=3)
    ap.add_argument("--binarize-mask", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    agg, dip, outline = _zoom_panel_from_config(
        args.config, args.sample_index, args.crop, args.center_box, args.binarize_mask
    )
    n_ch = int(dip.shape[0])
    rows = n_ch + 1
    titles = ["Aggregate"] + [f"ch{i}" for i in range(n_ch)]
    subplot_titles = []
    for t in titles:
        subplot_titles.extend([t, t])

    fig = make_subplots(rows=rows, cols=2, subplot_titles=subplot_titles, horizontal_spacing=0.04, vertical_spacing=0.04)

    add_panel(fig, agg, outline, 1, 1, "agg", showscale=False)
    add_panel(fig, agg, outline, 1, 2, "agg", showscale=True)

    for i in range(n_ch):
        add_panel(fig, dip[i], outline, i + 2, 1, f"ch{i}", showscale=False)
        add_panel(fig, dip[i], outline, i + 2, 2, f"ch{i}", showscale=False)

    h = max(800, 260 * rows)
    fig.update_layout(title=f"Mask Diagnostic: {os.path.basename(args.config)}", height=h, width=1200)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(out_path)


if __name__ == "__main__":
    main()
