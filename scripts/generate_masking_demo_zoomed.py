#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Reuse built-in demo helpers instead of rebuilding masking args here.
from scripts.masking_demo import (
    build_dataset,
    extract_centers_from_targets,
    load_config,
    make_context_and_debug,
)
from src.train import build_model_from_config


def _zoom_panel_from_config(
    config_path: str,
    sample_index: int,
    crop: int,
    center_box: int,
    center_value: float,
    binarize_mask: bool,
):
    cfg = load_config(config_path)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    ds = build_dataset(data_cfg)
    x = ds[int(sample_index) % len(ds)][0].numpy().astype(np.float32)
    x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)

    # Built-in helper: uses pipeline masking construction.
    x_ctx, target_locations, target_scales, target_valid, debug = make_context_and_debug(
        x_t, model_cfg, seed=42
    )
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
    else:
        mask_agg = dip.max(axis=0).astype(np.float32)

    # Pick nearest valid center to image center.
    ctr = np.array([h_img / 2.0, w_img / 2.0], dtype=np.float32)
    best = min(centers.tolist(), key=lambda p: (p[0] - ctr[0]) ** 2 + (p[1] - ctr[1]) ** 2)
    cy, cx = int(best[0]), int(best[1])
    # Map pixel-center coordinates from image grid to encoder grid.
    # Using +0.5 center mapping removes half-cell shift bias.
    ye = int(np.clip(np.floor((float(cy) + 0.5) * float(h_enc) / float(h_img)), 0, h_enc - 1))
    xe = int(np.clip(np.floor((float(cx) + 0.5) * float(w_enc) / float(w_img)), 0, w_enc - 1))

    cbox = max(1, int(center_box))
    if cbox % 2 == 0:
        cbox += 1
    half = cbox // 2

    # Zoom crop.
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
    if binarize_mask:
        dip_zoom = (dip_zoom > 1e-6).astype(np.float32)

    # Build target outline overlay separately (do not modify mask values).
    h_z, w_z = (y1 - y0), (x1 - x0)
    target_outline = np.zeros((h_z, w_z), dtype=np.float32)
    cyz, cxz = (ye - y0), (xe - x0)
    yy0z, yy1z = max(0, cyz - half), min(h_z, cyz + half + 1)
    xx0z, xx1z = max(0, cxz - half), min(w_z, cxz + half + 1)
    if yy1z > yy0z and xx1z > xx0z:
        target_outline[yy0z:yy1z, xx0z] = 1.0
        target_outline[yy0z:yy1z, xx1z - 1] = 1.0
        target_outline[yy0z, xx0z:xx1z] = 1.0
        target_outline[yy1z - 1, xx0z:xx1z] = 1.0

    return mask_agg[y0:y1, x0:x1], target_outline, dip_zoom


def main() -> None:
    parser = argparse.ArgumentParser(description="Zoomed mask demo using built-in masking helpers.")
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--label-a", default="Config A")
    parser.add_argument("--label-b", default="Config B")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--crop", type=int, default=16)
    parser.add_argument("--center-box", type=int, default=4)
    parser.add_argument("--center-value", type=float, default=2.0)
    parser.add_argument("--num-channels", type=int, default=6)
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="Show only aggregate + RGB composite (first 3 channels) per config.",
    )
    parser.add_argument(
        "--binarize-mask",
        action="store_true",
        help="If set, convert mask maps to binary (0/1) before overlay.",
    )
    parser.add_argument("--out", default="sessions/masking_demo_zoomed_compare.png")
    args = parser.parse_args()

    panel_a, overlay_a, ch_a = _zoom_panel_from_config(
        args.config_a, args.sample_index, args.crop, args.center_box, args.center_value, args.binarize_mask
    )
    panel_b, overlay_b, ch_b = _zoom_panel_from_config(
        args.config_b, args.sample_index, args.crop, args.center_box, args.center_value, args.binarize_mask
    )

    n_ch = int(max(1, min(args.num_channels, ch_a.shape[0], ch_b.shape[0])))
    n_rows = 2 if args.rgb_only else (n_ch + 1)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 3.0 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    vmax = 1.0
    for col, (title, panel, overlay) in enumerate(
        [(f"{args.label_a} (agg)", panel_a, overlay_a), (f"{args.label_b} (agg)", panel_b, overlay_b)]
    ):
        ax = axes[0, col]
        im = ax.imshow(panel, cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.contour(overlay, levels=[0.5], colors=["#ff0000"], linewidths=1.2)
        ax.set_title(f"{title} zoom {args.crop}x{args.crop}")
        ax.set_xticks(np.arange(0, panel.shape[1], max(1, panel.shape[1] // 4)))
        ax.set_yticks(np.arange(0, panel.shape[0], max(1, panel.shape[0] // 4)))
        ax.grid(color="white", alpha=0.25, linewidth=0.6)

    if args.rgb_only:
        for col, (title, ch, overlay) in enumerate([(args.label_a, ch_a, overlay_a), (args.label_b, ch_b, overlay_b)]):
            ax = axes[1, col]
            rgb = np.zeros((ch.shape[1], ch.shape[2], 3), dtype=np.float32)
            c = min(3, ch.shape[0])
            rgb[..., :c] = np.transpose(ch[:c], (1, 2, 0))
            rgb = np.clip(rgb / max(1e-8, float(vmax)), 0.0, 1.0)
            ax.imshow(rgb, interpolation="nearest")
            ax.contour(overlay, levels=[0.5], colors=["#ff0000"], linewidths=1.0)
            ax.set_title(f"{title} RGB(ch0-2)")
            ax.set_xticks(np.arange(0, rgb.shape[1], max(1, rgb.shape[1] // 4)))
            ax.set_yticks(np.arange(0, rgb.shape[0], max(1, rgb.shape[0] // 4)))
            ax.grid(color="white", alpha=0.25, linewidth=0.6)
    else:
        for i in range(n_ch):
            for col, (title, ch, overlay) in enumerate([(args.label_a, ch_a, overlay_a), (args.label_b, ch_b, overlay_b)]):
                ax = axes[i + 1, col]
                z = ch[i]
                ax.imshow(z, cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")
                ax.contour(overlay, levels=[0.5], colors=["#ff0000"], linewidths=1.0)
                ax.set_title(f"{title} ch{i}")
                ax.set_xticks(np.arange(0, z.shape[1], max(1, z.shape[1] // 4)))
                ax.set_yticks(np.arange(0, z.shape[0], max(1, z.shape[0] // 4)))
                ax.grid(color="white", alpha=0.25, linewidth=0.6)

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    plt.tight_layout()
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=240)
    print(out_path)


if __name__ == "__main__":
    main()
