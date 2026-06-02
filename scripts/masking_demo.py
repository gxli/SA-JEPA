#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import JEPADataset
from src.models.build_jepa import PyramidGridJEPA, make_pyramid_grid_context

DEMO_MASK_FILL_MODE = "zero"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm01(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    if hi <= lo + 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def log_norm(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    z = np.log1p(np.abs(arr) / float(eps))
    lo = float(np.percentile(z, 1.0))
    hi = float(np.percentile(z, 99.0))
    if hi <= lo + 1e-12:
        return np.zeros_like(z, dtype=np.float32)
    return np.clip((z - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def signed_log_norm(arr: np.ndarray, p: float = 99.0, eps: float = 1e-12) -> np.ndarray:
    z = np.sign(arr) * np.log1p(np.abs(arr) / float(eps))
    lim = float(np.percentile(np.abs(z), p))
    if lim <= 1e-12:
        lim = 1e-12
    return np.clip(0.5 + 0.5 * (z / lim), 0.0, 1.0).astype(np.float32)


def shared_log_pair(a: np.ndarray, b: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    e = max(1e-30, float(eps))
    a_log = np.log(np.clip(a, a_min=0.0, a_max=None) + e).astype(np.float32)
    b_log = np.log(np.clip(b, a_min=0.0, a_max=None) + e).astype(np.float32)
    return a_log, b_log


def display_soft(arr: np.ndarray, p_low: float = 0.2, p_high: float = 99.8, compress: float = 3.0) -> np.ndarray:
    # Gentle contrast for weak structures: percentile clip + asinh compression.
    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    if hi <= lo + 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    x = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    y = np.arcsinh(compress * x) / np.arcsinh(compress)
    return y.astype(np.float32)


def display_soft_pair(a: np.ndarray, b: np.ndarray, p_low: float = 0.2, p_high: float = 99.8, compress: float = 3.0):
    pair = np.concatenate([a.reshape(-1), b.reshape(-1)]).astype(np.float32)
    lo = float(np.percentile(pair, p_low))
    hi = float(np.percentile(pair, p_high))
    if hi <= lo + 1e-12:
        z = np.zeros_like(a, dtype=np.float32)
        return z, np.zeros_like(b, dtype=np.float32)

    def _map(x: np.ndarray) -> np.ndarray:
        xn = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        return (np.arcsinh(compress * xn) / np.arcsinh(compress)).astype(np.float32)

    return _map(a), _map(b)


def build_dataset(data_cfg: dict) -> JEPADataset:
    # Keep demo data path aligned with training: normalize in the dataset and
    # perform any CDD decomposition only in model-side masking.
    return JEPADataset(
        num_samples=max(1, int(data_cfg.get("num_samples", 1))),
        data_root=data_cfg.get("data_root", "data"),
        npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
        cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
        cube_slice_axis=int(data_cfg.get("cube_slice_axis", 0)),
        cube_slice_index=int(data_cfg.get("cube_slice_index", 0)),
    )


def make_context_and_debug(x: torch.Tensor, model_cfg: dict, seed: int):
    torch.manual_seed(seed)
    return make_pyramid_grid_context(
        x_clean=x,
        sigmas=tuple(model_cfg.get("sigmas", [2, 4, 8, 16])),
        mask_fraction=float(model_cfg.get("active_target_fraction", model_cfg.get("mask_fraction", 1.0))),
        mask_scale=float(model_cfg.get("mask_scale_factor", 1.0)),
        spacing_scale=float(model_cfg.get("mask_spacing_scaling", 1.5)),
        global_shift=bool(model_cfg.get("global_shift", True)),
        align_scales=bool(model_cfg.get("align_scales", True)),
        mask_box_size=int(model_cfg.get("mask_footprint_px", 16)),
        cdd_mode=model_cfg.get("cdd_mode", "log"),
        cdd_constrained=bool(model_cfg.get("cdd_constrained", True)),
        cdd_sm_mode=model_cfg.get("cdd_sm_mode", "reflect"),
        return_debug=True,
    )


def extract_centers(debug: dict) -> np.ndarray:
    arr = debug["unique_centers"][0].cpu().numpy()
    return np.array([(int(y), int(x)) for y, x in arr.tolist() if int(y) >= 0 and int(x) >= 0], dtype=np.int64)


def extract_centers_from_targets(target_locations: torch.Tensor, target_valid: torch.Tensor) -> np.ndarray:
    loc = target_locations[0].cpu().numpy()
    val = target_valid[0].cpu().numpy().astype(bool)
    out = []
    for i in range(loc.shape[0]):
        if not val[i]:
            continue
        out.append((int(loc[i, 0]), int(loc[i, 1])))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(out, dtype=np.int64)


def compute_boundary(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0.5).astype(np.uint8)
    h, w = m.shape
    out = np.zeros_like(m, dtype=np.float32)
    out[1:, :] = np.maximum(out[1:, :], (m[1:, :] != m[:-1, :]).astype(np.float32))
    out[:, 1:] = np.maximum(out[:, 1:], (m[:, 1:] != m[:, :-1]).astype(np.float32))
    out[: h - 1, :] = np.maximum(out[: h - 1, :], (m[: h - 1, :] != m[1:, :]).astype(np.float32))
    out[:, : w - 1] = np.maximum(out[:, : w - 1], (m[:, : w - 1] != m[:, 1:]).astype(np.float32))
    return out


def plot_overview(x: np.ndarray, ctx: np.ndarray, centers: np.ndarray, out_path: str):
    delta = ctx - x

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(display_soft(x), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(display_soft(ctx), cmap="gray")
    axes[1].set_title("Masked/Context")
    axes[1].axis("off")

    axes[2].imshow(display_soft(x), cmap="gray")
    axes[2].set_title("Mask Centers")
    axes[2].axis("off")
    if centers.size > 0:
        axes[2].scatter(centers[:, 1], centers[:, 0], c="red", s=12, marker="x")

    im = axes[3].imshow(signed_log_norm(delta), cmap="seismic")
    axes[3].set_title("Context - Original (signed)")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overview_html(x: np.ndarray, ctx: np.ndarray, centers: np.ndarray, out_html: str):
    delta = ctx - x
    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=("Original", "Masked/Context", "Mask Centers", "Context-Original"),
        horizontal_spacing=0.015,
    )
    center_map = np.zeros_like(x, dtype=np.float32)
    if centers.size > 0:
        center_map[centers[:, 0], centers[:, 1]] = 1.0
    # Shared display scaling avoids visually "inflating" masked regions.
    x_vis, ctx_vis = display_soft_pair(x, ctx)
    panels = [x_vis, ctx_vis, center_map, delta]
    for i, arr in enumerate(panels, start=1):
        if i in (1, 2):
            lo = float(np.percentile(arr, 1.0))
            hi = float(np.percentile(arr, 99.0))
            fig.add_trace(go.Heatmap(z=arr, colorscale="Viridis", zmin=lo, zmax=hi, showscale=(i == 4)), row=1, col=i)
        elif i == 3:
            fig.add_trace(go.Heatmap(z=arr, colorscale="Magma", zmin=0.0, zmax=1.0, showscale=False), row=1, col=i)
        else:
            lim = float(np.percentile(np.abs(arr), 99.0))
            if lim <= 1e-12:
                lim = 1e-12
            fig.add_trace(
                go.Heatmap(z=arr, colorscale="RdBu", zmin=-lim, zmax=lim, zmid=0.0, showscale=True),
                row=1,
                col=i,
            )
        fig.update_xaxes(constrain="domain", row=1, col=i)
        fig.update_yaxes(scaleanchor=f"x{i}", scaleratio=1, row=1, col=i)
    fig.update_layout(height=460, width=1680, title="Masking Overview (Interactive)", margin=dict(l=20, r=20, t=50, b=20))
    fig.write_html(out_html, include_plotlyjs="cdn")


def plot_channels(debug: dict, centers: np.ndarray, out_path: str):
    cdd_orig_t = debug.get("cdd_channels_orig")
    cdd_mask_t = debug.get("cdd_channels_masked")
    if cdd_orig_t is None or cdd_mask_t is None or cdd_orig_t.numel() == 0 or cdd_mask_t.numel() == 0:
        return
    cdd_orig = cdd_orig_t[0].cpu().numpy()
    cdd_mask = cdd_mask_t[0].cpu().numpy()

    n_ch = cdd_orig.shape[0]
    fig, axes = plt.subplots(n_ch, 3, figsize=(13, 3.2 * n_ch))
    if n_ch == 1:
        axes = np.array([axes])

    for i in range(n_ch):
        ch0 = cdd_orig[i]
        ch1 = cdd_mask[i]
        d = ch1 - ch0

        axes[i, 0].imshow(log_norm(ch0), cmap="viridis")
        axes[i, 0].set_title(f"CDD Ch {i} Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(log_norm(ch1), cmap="viridis")
        axes[i, 1].set_title("CDD Ch Zeroed")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(log_norm(d), cmap="magma")
        axes[i, 2].set_title("Delta")
        axes[i, 2].axis("off")

        if centers.size > 0:
            for ax in (axes[i, 0], axes[i, 1], axes[i, 2]):
                ax.scatter(centers[:, 1], centers[:, 0], c="red", s=10, marker="x")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_channels_html(debug: dict, out_html: str):
    cdd_orig_t = debug.get("cdd_channels_orig")
    cdd_mask_t = debug.get("cdd_channels_masked")
    if cdd_orig_t is None or cdd_mask_t is None or cdd_orig_t.numel() == 0 or cdd_mask_t.numel() == 0:
        return
    cdd_orig = cdd_orig_t[0].cpu().numpy()
    cdd_mask = cdd_mask_t[0].cpu().numpy()
    n = cdd_orig.shape[0]
    fig = make_subplots(
        rows=n,
        cols=3,
        subplot_titles=sum(([f"Ch {i} orig", f"Ch {i} masked", f"Ch {i} delta"] for i in range(n)), []),
        horizontal_spacing=0.02,
        vertical_spacing=0.04,
    )
    for i in range(n):
        orig = cdd_orig[i]
        masked = cdd_mask[i]
        delta = masked - orig
        pair = np.concatenate([orig.ravel(), masked.ravel()])
        lo = float(np.percentile(pair, 1.0))
        hi = float(np.percentile(pair, 99.0))
        dlim = float(np.max(np.abs(delta)))
        if dlim <= 1e-12:
            dlim = 1e-12
        fig.add_trace(
            go.Heatmap(
                z=orig,
                colorscale="Viridis",
                zmin=lo,
                zmax=hi,
                showscale=False,
                hovertemplate="y=%{y}, x=%{x}, v=%{z:.6g}<extra></extra>",
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=masked,
                colorscale="Viridis",
                zmin=lo,
                zmax=hi,
                showscale=False,
                hovertemplate="y=%{y}, x=%{x}, v=%{z:.6g}<extra></extra>",
            ),
            row=i + 1,
            col=2,
        )
        fig.add_trace(
            go.Heatmap(
                z=delta,
                colorscale="RdBu",
                zmin=-dlim,
                zmax=dlim,
                zmid=0.0,
                showscale=False,
                hovertemplate="y=%{y}, x=%{x}, Δ=%{z:.6g}<extra></extra>",
            ),
            row=i + 1,
            col=3,
        )
        fig.update_xaxes(constrain="domain", row=i + 1, col=1)
        fig.update_yaxes(scaleanchor=f"x{3*i+1}", scaleratio=1, row=i + 1, col=1)
        fig.update_xaxes(constrain="domain", row=i + 1, col=2)
        fig.update_yaxes(scaleanchor=f"x{3*i+2}", scaleratio=1, row=i + 1, col=2)
        fig.update_xaxes(constrain="domain", row=i + 1, col=3)
        fig.update_yaxes(scaleanchor=f"x{3*i+3}", scaleratio=1, row=i + 1, col=3)
    fig.update_layout(
        height=max(360, 240 * n),
        width=1560,
        title="CDD Channels (Interactive)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def plot_dip_field(debug: dict, out_path: str):
    dip_t = debug.get("dip_field")
    if dip_t is None or dip_t.numel() == 0:
        return
    dip = dip_t[0].cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(dip, cmap="magma")
    ax.set_title("Applied CDD Channel Mask Field")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_dip_field_per_channel(debug: dict, out_path: str):
    dip_t = debug.get("dip_field_per_channel")
    if dip_t is None or dip_t.numel() == 0:
        return
    dip = dip_t[0].cpu().numpy()  # C x H x W
    c = dip.shape[0]
    fig, axes = plt.subplots(c, 2, figsize=(11, 3.0 * c))
    if c == 1:
        axes = np.array([axes])
    for i, ax in enumerate(axes):
        lin = np.clip(dip[i], 0.0, 1.0)
        logv = np.log1p(1000.0 * lin) / np.log1p(1000.0)

        im0 = ax[0].imshow(lin, cmap="magma", vmin=0.0, vmax=1.0)
        ax[0].set_title(f"Dip Ch {i} (linear)")
        ax[0].axis("off")
        im1 = ax[1].imshow(logv, cmap="magma", vmin=0.0, vmax=1.0)
        ax[1].set_title(f"Dip Ch {i} (log)")
        ax[1].axis("off")

        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)




def plot_dip_proto_per_channel(debug: dict, out_path: str):
    dip_t = debug.get("dip_proto_per_channel")
    if dip_t is None or dip_t.numel() == 0:
        return
    dip = dip_t[0].cpu().numpy()
    c = dip.shape[0]
    fig, axes = plt.subplots(c, 3, figsize=(15, 3.0 * c))
    if c == 1:
        axes = np.array([axes])
    for i, ax in enumerate(axes):
        lin = np.clip(dip[i], 0.0, 1.0)
        logv = np.log1p(1000.0 * lin) / np.log1p(1000.0)
        # Contrast view for near-zero fields.
        p = float(np.percentile(lin, 99.9))
        vmax = max(1e-8, p)
        im0 = ax[0].imshow(lin, cmap="magma", vmin=0.0, vmax=vmax)
        ax[0].set_title(f"Dip Proto Ch {i} (linear)")
        ax[0].axis("off")
        im1 = ax[1].imshow(logv, cmap="magma", vmin=0.0, vmax=1.0)
        ax[1].set_title(f"Dip Proto Ch {i} (log)")
        ax[1].axis("off")
        # Local crop around the peak to make width visually measurable.
        py, px = np.unravel_index(np.argmax(lin), lin.shape)
        rr = 48
        y0, y1 = max(0, py - rr), min(lin.shape[0], py + rr)
        x0, x1 = max(0, px - rr), min(lin.shape[1], px + rr)
        im2 = ax[2].imshow(lin[y0:y1, x0:x1], cmap="magma", vmin=0.0, vmax=1.0)
        ax[2].set_title(f"Dip Proto Ch {i} (crop)")
        ax[2].axis("off")
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def build_model(model_cfg: dict, data_cfg: dict) -> PyramidGridJEPA:
    model_post_log = bool(model_cfg.get("post_log_transform", True))
    return PyramidGridJEPA(
        mode=model_cfg.get("mode", "image"),
        latent_channels=model_cfg.get("latent_channels", 32),
        predictor_hidden=model_cfg.get("predictor_hidden"),
        patch_size=model_cfg.get("patch_size", 2),
        sigmas=tuple(model_cfg.get("sigmas", [2, 4, 8, 16])),
        mask_fraction=model_cfg.get("active_target_fraction", model_cfg.get("mask_fraction", 1.0)),
        mask_scale=model_cfg.get("mask_scale_factor", 1.0),
        spacing_scale=model_cfg.get("mask_spacing_scaling", 1.5),
        global_shift=model_cfg.get("global_shift", True),
        align_scales=model_cfg.get("align_scales", True),
        mask_box_size=model_cfg.get("mask_footprint_px", 16),
        cdd_mode=model_cfg.get("cdd_mode", "log"),
        cdd_constrained=model_cfg.get("cdd_constrained", True),
        cdd_sm_mode=model_cfg.get("cdd_sm_mode", "reflect"),
        cdd_append_last_residual=model_cfg.get("cdd_append_last_residual", True),
        post_log_transform=model_post_log,
        log_eps=model_cfg.get("log_eps", float(data_cfg.get("log_eps", 1.0))),
        cdd_log_std_floor_mult=model_cfg.get("cdd_log_std_floor_mult", 0.05),
        ema_momentum=model_cfg.get("ema_momentum", 0.996),
        normalize_loss_l2=model_cfg.get("normalize_loss_l2", True),
        predictor_layernorm=model_cfg.get("predictor_layernorm", False),
        encoder_type=model_cfg.get("encoder_type", "convnext_dense_masktoken"),
        encoder_width=model_cfg.get("encoder_width", model_cfg.get("latent_channels", 32)),
        encoder_depth=model_cfg.get("encoder_depth", 4),
        encoder_kernel_size=model_cfg.get("encoder_kernel_size", 7),
        scaleaware_feat_channels=model_cfg.get("scaleaware_feat_channels", 8),
        scaleaware_adapter_kernel_size=model_cfg.get("scaleaware_adapter_kernel_size", 3),
        scaleaware_fusion_type=model_cfg.get("scaleaware_fusion_type", "concat"),
        scaleaware_norm_per_scale=model_cfg.get("scaleaware_norm_per_scale", False),
    )


def load_model_checkpoint_if_available(model: PyramidGridJEPA, session_dir: str) -> Optional[str]:
    ckpt_path = os.path.join(session_dir, "model_last.pt")
    if not os.path.exists(ckpt_path):
        return None
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return ckpt_path


def plot_network_outputs(x_clean: np.ndarray, x_context: np.ndarray, pred_norm: np.ndarray, gt_norm: np.ndarray, out_path: str):
    diff = pred_norm - gt_norm
    dlim = float(np.percentile(np.abs(diff), 99.0))
    if dlim <= 1e-12:
        dlim = 1e-12
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    axes[0].imshow(display_soft(x_clean), cmap="gray")
    axes[0].set_title("Net Input Clean")
    axes[0].axis("off")
    axes[1].imshow(display_soft(x_context), cmap="gray")
    axes[1].set_title("Net Input Context")
    axes[1].axis("off")
    axes[2].imshow(display_soft(pred_norm), cmap="viridis")
    axes[2].set_title("Pred Latent Norm")
    axes[2].axis("off")
    axes[3].imshow(display_soft(gt_norm), cmap="viridis")
    axes[3].set_title("GT Latent Norm")
    axes[3].axis("off")
    im = axes[4].imshow(diff, cmap="RdBu", vmin=-dlim, vmax=dlim)
    axes[4].set_title("Pred-GT Norm")
    axes[4].axis("off")
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_network_outputs_html(
    x_clean: np.ndarray, x_context: np.ndarray, pred_norm: np.ndarray, gt_norm: np.ndarray, out_html: str
):
    diff = pred_norm - gt_norm
    dlim = float(np.percentile(np.abs(diff), 99.0))
    if dlim <= 1e-12:
        dlim = 1e-12
    fig = make_subplots(
        rows=1,
        cols=5,
        subplot_titles=("Net Input Clean", "Net Input Context", "Pred Latent Norm", "GT Latent Norm", "Pred-GT Norm"),
        horizontal_spacing=0.015,
    )
    panels = [display_soft(x_clean), display_soft(x_context), display_soft(pred_norm), display_soft(gt_norm), diff]
    for i, arr in enumerate(panels, start=1):
        if i < 5:
            lo = float(np.percentile(arr, 1.0))
            hi = float(np.percentile(arr, 99.0))
            fig.add_trace(go.Heatmap(z=arr, colorscale="Viridis", zmin=lo, zmax=hi, showscale=False), row=1, col=i)
        else:
            fig.add_trace(go.Heatmap(z=arr, colorscale="RdBu", zmin=-dlim, zmax=dlim, zmid=0.0, showscale=True), row=1, col=i)
        fig.update_xaxes(constrain="domain", row=1, col=i)
        fig.update_yaxes(scaleanchor=f"x{i}", scaleratio=1, row=1, col=i)
    fig.update_layout(height=460, width=2100, title="Network Outputs (Interactive)", margin=dict(l=20, r=20, t=50, b=20))
    fig.write_html(out_html, include_plotlyjs="cdn")


def build_overview_figure(
    x_net: np.ndarray,
    ctx_net: np.ndarray,
    centers: np.ndarray,
    debug: dict,
    mode: str,
    title: str,
    frac_i1: np.ndarray,
    frac_i2: np.ndarray,
    diff_i1: np.ndarray,
    diff_i2: np.ndarray,
):
    delta = diff_i2 - diff_i1
    # Exact requested fractional difference with a robust denominator floor to
    # avoid low-signal blow-ups outside masked regions.
    frac_den = np.clip(frac_i1 + frac_i2, a_min=0.0, a_max=None).astype(np.float32)
    den_floor = max(1e-12, float(np.percentile(frac_den, 25.0)) * 0.1)
    frac = (frac_i1 - frac_i2) / np.maximum(frac_den, den_floor)
    mask_map = debug["mask_map"][0].cpu().numpy().astype(np.float32)
    dip_t = debug.get("dip_field")
    if mode == "channel_mask" and dip_t is not None and dip_t.numel() > 0:
        applied_mask = np.clip(dip_t[0].cpu().numpy().astype(np.float32), 0.0, 1.0)
    else:
        applied_mask = np.clip(mask_map, 0.0, 1.0)
    # Mask delta and frac to only show differences inside masked regions.
    delta[applied_mask <= 0] = 0.0
    frac[applied_mask <= 0] = 0.0
    fig = make_subplots(
        rows=1,
        cols=6,
        subplot_titles=(
            "Original",
            "Masked/Context",
            "Mask Centers",
            "Applied Mask (0-1)",
            "Context-Original",
            "Fractional Diff (I1-I2)/(I1+I2)",
        ),
        horizontal_spacing=0.015,
    )
    center_map = np.zeros_like(x_net, dtype=np.float32)
    if centers.size > 0:
        for cy, cx in centers:
            y0, y1 = max(0, int(cy) - 1), min(center_map.shape[0], int(cy) + 2)
            x0, x1 = max(0, int(cx) - 1), min(center_map.shape[1], int(cx) + 2)
            center_map[y0:y1, x0:x1] = 1.0
    x_vis, ctx_vis = display_soft_pair(x_net, ctx_net)
    panels = [x_vis, ctx_vis, center_map, applied_mask, delta, frac]
    for i, arr in enumerate(panels, start=1):
        if i in (1, 2):
            lo = float(np.percentile(arr, 1.0))
            hi = float(np.percentile(arr, 99.0))
            fig.add_trace(go.Heatmap(z=arr, colorscale="Viridis", zmin=lo, zmax=hi, showscale=False), row=1, col=i)
        elif i == 3:
            fig.add_trace(go.Heatmap(z=arr, colorscale="Magma", zmin=0.0, zmax=1.0, showscale=False), row=1, col=i)
        elif i == 4:
            fig.add_trace(go.Heatmap(z=arr, colorscale="Magma", zmin=0.0, zmax=1.0, showscale=True), row=1, col=i)
        elif i == 5:
            lim = float(np.percentile(np.abs(arr), 99.0))
            if lim <= 1e-12:
                lim = 1e-12
            fig.add_trace(
                go.Heatmap(z=arr, colorscale="RdBu", zmin=-lim, zmax=lim, zmid=0.0, showscale=True),
                row=1,
                col=i,
            )
        else:
            fmax = float(np.max(np.abs(arr)))
            if fmax <= 1e-12:
                fmax = 1e-12
            fig.add_trace(
                go.Heatmap(z=arr, colorscale="RdBu", zmin=-fmax, zmax=fmax, zmid=0.0, showscale=True),
                row=1,
                col=i,
            )
        fig.update_xaxes(constrain="domain", row=1, col=i)
        fig.update_yaxes(scaleanchor=f"x{i}", scaleratio=1, row=1, col=i)

    if mode == "zero":
        boundary = compute_boundary(applied_mask)
        by, bx = np.where(boundary > 0.5)
        if by.size > 0:
            for col in (1, 2):
                fig.add_trace(
                    go.Scattergl(
                        x=bx.tolist(),
                        y=by.tolist(),
                        mode="markers",
                        marker={"size": 2, "color": "yellow"},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )
    if mode == "channel_mask" and centers.size > 0:
        for col in (1, 2):
            fig.add_trace(
                go.Scattergl(
                    x=centers[:, 1].tolist(),
                    y=centers[:, 0].tolist(),
                    mode="markers",
                    marker={"size": 5, "color": "yellow", "symbol": "x"},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
    # Red + markers on fractional diff (col 6) to show target locations.
    if centers.size > 0:
        fig.add_trace(
            go.Scattergl(
                x=centers[:, 1].tolist(),
                y=centers[:, 0].tolist(),
                mode="markers",
                marker={"size": 7, "color": "red", "symbol": "cross"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=6,
        )
    fig.update_layout(height=460, width=2450, title=title, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def build_channels_figure(debug: dict, title: str):
    cdd_orig_t = debug.get("cdd_channels_orig")
    cdd_mask_t = debug.get("cdd_channels_masked")
    if cdd_orig_t is None or cdd_mask_t is None or cdd_orig_t.numel() == 0 or cdd_mask_t.numel() == 0:
        return None
    cdd_orig = cdd_orig_t[0].cpu().numpy()
    cdd_mask = cdd_mask_t[0].cpu().numpy()
    n = cdd_orig.shape[0]

    box_sizes_t = debug.get("cdd_box_sizes")
    blur_sigmas_t = debug.get("cdd_blur_sigmas")
    box_sizes = box_sizes_t[0].cpu().numpy() if box_sizes_t is not None and box_sizes_t.numel() > 0 else np.full(n, -1)
    blur_sigmas = blur_sigmas_t[0].cpu().numpy() if blur_sigmas_t is not None and blur_sigmas_t.numel() > 0 else np.zeros(n)
    chan_labels = []
    for i in range(n):
        bs = int(box_sizes[i])
        sigma = float(blur_sigmas[i])
        fwhm = 2.355 * sigma
        if sigma > 0:
            info = f"box={bs}px σ={sigma:.1f} FWHM={fwhm:.1f}px"
        else:
            info = f"box={bs}px"
        chan_labels.extend([f"Ch{i} {info} orig", f"Ch{i} masked", f"Ch{i} delta"])

    fig = make_subplots(
        rows=n,
        cols=3,
        subplot_titles=chan_labels,
        horizontal_spacing=0.02,
        vertical_spacing=0.04,
    )
    for i in range(n):
        orig = cdd_orig[i]
        masked = cdd_mask[i]
        delta = masked - orig
        pair = np.concatenate([orig.ravel(), masked.ravel()])
        lo = float(np.percentile(pair, 1.0))
        hi = float(np.percentile(pair, 99.0))
        dlim = float(np.max(np.abs(delta)))
        if dlim <= 1e-12:
            dlim = 1e-12
        fig.add_trace(go.Heatmap(z=orig, colorscale="Viridis", zmin=lo, zmax=hi, showscale=False), row=i + 1, col=1)
        fig.add_trace(go.Heatmap(z=masked, colorscale="Viridis", zmin=lo, zmax=hi, showscale=False), row=i + 1, col=2)
        fig.add_trace(go.Heatmap(z=delta, colorscale="RdBu", zmin=-dlim, zmax=dlim, zmid=0.0, showscale=False), row=i + 1, col=3)
        # Contour overlay on delta to highlight non-zero regions.
        delta_abs = np.abs(delta)
        contour_level = max(float(np.percentile(delta_abs, 80.0)), dlim * 0.1)
        if contour_level > 1e-12 and np.any(delta_abs > contour_level):
            fig.add_trace(
                go.Contour(
                    z=delta_abs,
                    contours=dict(start=contour_level, end=float(delta_abs.max()), size=contour_level),
                    contours_coloring="lines",
                    line=dict(width=1, color="lime"),
                    showscale=False,
                    hoverinfo="skip",
                ),
                row=i + 1,
                col=3,
            )
        c1 = 3 * i + 1
        c2 = 3 * i + 2
        c3 = 3 * i + 3
        fig.update_xaxes(constrain="domain", row=i + 1, col=1)
        fig.update_yaxes(scaleanchor=f"x{c1}", scaleratio=1, row=i + 1, col=1)
        fig.update_xaxes(constrain="domain", row=i + 1, col=2)
        fig.update_yaxes(scaleanchor=f"x{c2}", scaleratio=1, row=i + 1, col=2)
        fig.update_xaxes(constrain="domain", row=i + 1, col=3)
        fig.update_yaxes(scaleanchor=f"x{c3}", scaleratio=1, row=i + 1, col=3)
    fig.update_layout(
        height=max(420, 300 * n),
        width=1200,
        title=title,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_network_figure(x_clean: np.ndarray, x_context: np.ndarray, pred_norm: np.ndarray, gt_norm: np.ndarray, title: str):
    diff = pred_norm - gt_norm
    dlim = float(np.percentile(np.abs(diff), 99.0))
    if dlim <= 1e-12:
        dlim = 1e-12
    ng = np.concatenate([pred_norm.ravel(), gt_norm.ravel()])
    nlo = float(np.percentile(ng, 1.0))
    nhi = float(np.percentile(ng, 99.5))
    if nhi <= nlo + 1e-12:
        nhi = nlo + 1e-12
    fig = make_subplots(
        rows=1,
        cols=5,
        subplot_titles=("Net Input Clean", "Net Input Context", "Pred Latent Norm", "GT Latent Norm", "Pred-GT Norm"),
        horizontal_spacing=0.015,
    )
    fig.add_trace(go.Heatmap(z=display_soft(x_clean), colorscale="Viridis", showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=display_soft(x_context), colorscale="Viridis", showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=pred_norm, colorscale="Viridis", zmin=nlo, zmax=nhi, showscale=False), row=1, col=3)
    fig.add_trace(go.Heatmap(z=gt_norm, colorscale="Viridis", zmin=nlo, zmax=nhi, showscale=False), row=1, col=4)
    fig.add_trace(go.Heatmap(z=diff, colorscale="RdBu", zmin=-dlim, zmax=dlim, zmid=0.0, showscale=True), row=1, col=5)
    for i in range(1, 6):
        fig.update_xaxes(constrain="domain", row=1, col=i)
        fig.update_yaxes(scaleanchor=f"x{i}", scaleratio=1, row=1, col=i)
    fig.update_layout(height=520, width=1500, title=title, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def build_channel_mask_figure(debug: dict, title: str):
    dip_t = debug.get("dip_field_per_channel")
    if dip_t is None or dip_t.numel() == 0:
        return None, None
    dip = dip_t[0].cpu().numpy().astype(np.float32)  # C,H,W
    c = dip.shape[0]
    fig = make_subplots(
        rows=1,
        cols=c,
        subplot_titles=[f"Mask Ch {i}" for i in range(c)],
        horizontal_spacing=0.01,
    )
    for i in range(c):
        fig.add_trace(go.Heatmap(z=np.clip(dip[i], 0.0, 1.0), colorscale="Magma", zmin=0.0, zmax=1.0, showscale=(i == c - 1)), row=1, col=i + 1)
        fig.update_xaxes(constrain="domain", row=1, col=i + 1)
        fig.update_yaxes(scaleanchor=f"x{i+1}", scaleratio=1, row=1, col=i + 1)
    fig.update_layout(height=320, width=max(900, 280 * c), title=title, margin=dict(l=20, r=20, t=50, b=20))
    return fig, dip


def evaluate_mask_symmetry(
    ds: JEPADataset,
    model_cfg_run: dict,
    n_samples: int,
    base_seed: int,
) -> tuple[np.ndarray, dict]:
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    n = int(max(1, n_samples))
    x0 = ds[0][0].numpy().astype(np.float32)
    h, w = int(x0.shape[0]), int(x0.shape[1])
    acc = np.zeros((h, w), dtype=np.float64)
    for i in range(n):
        x = ds[i % len(ds)][0].numpy().astype(np.float32)
        x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
        _, _, _, _, debug = make_context_and_debug(x_t, model_cfg_run, int(base_seed + i))
        dip_t = debug.get("dip_field")
        if str(model_cfg_run.get("mask_fill_mode", DEMO_MASK_FILL_MODE)) == "channel_mask" and dip_t is not None and dip_t.numel() > 0:
            mask = np.clip(dip_t[0].cpu().numpy().astype(np.float32), 0.0, 1.0)
        else:
            mask = np.clip(debug["mask_map"][0].cpu().numpy().astype(np.float32), 0.0, 1.0)
        acc += mask.astype(np.float64)
    heat = (acc / float(n)).astype(np.float32)

    lr_flip = heat[:, ::-1]
    tb_flip = heat[::-1, :]
    lr_mae = float(np.mean(np.abs(heat - lr_flip)))
    tb_mae = float(np.mean(np.abs(heat - tb_flip)))
    top = float(np.mean(heat[: (h // 2), :]))
    bot = float(np.mean(heat[(h // 2) :, :]))
    left = float(np.mean(heat[:, : (w // 2)]))
    right = float(np.mean(heat[:, (w // 2) :]))
    metrics = {
        "samples": int(n),
        "mean_mask_value": float(np.mean(heat)),
        "lr_symmetry_mae": lr_mae,
        "tb_symmetry_mae": tb_mae,
        "top_minus_bottom": float(top - bot),
        "left_minus_right": float(left - right),
    }
    return heat, metrics


def main():
    parser = argparse.ArgumentParser(description="Masking demo wrapper over core pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sessions-dir", type=str, default="sessions")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-mode", type=str, choices=["config", "zero", "channel_mask", "both"], default="config")
    parser.add_argument("--rigid-mask-box", action="store_true", help="Keep constant box mask size (disable adaptive sizing)")
    parser.add_argument("--eval-samples", type=int, default=0, help="If >0, run aggregate mask symmetry eval with this many samples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    session_dir = os.path.join(args.sessions_dir, config_name)
    os.makedirs(session_dir, exist_ok=True)

    ds = build_dataset(data_cfg)
    x = ds[int(args.sample_index) % len(ds)][0].numpy().astype(np.float32)
    x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)

    modes = []
    if args.mask_mode == "both":
        modes = ["zero", "channel_mask"]
    elif args.mask_mode == "config":
        modes = [DEMO_MASK_FILL_MODE]
    else:
        modes = [args.mask_mode]

    all_meta = {"config": args.config, "sample_index": int(args.sample_index), "seed": int(args.seed), "modes": []}
    effective_single_mode = modes[0] if len(modes) == 1 else None
    if effective_single_mode == "zero":
        report_name = f"masking_demo_{config_name}_box_masked.html"
    elif effective_single_mode == "channel_mask":
        report_name = f"masking_demo_{config_name}_channel_mask_only.html"
    else:
        report_name = f"masking_demo_{config_name}_all.html"
    report_html = os.path.join(session_dir, report_name)
    html_parts = [
        "<html><head><meta charset='utf-8'><title>Masking Demo All</title>"
        "<style>body{font-family:Arial,sans-serif;margin:16px;} .panel{max-width:1500px;margin:0 auto 28px auto;}</style>"
        "</head><body>",
        "<h1>Masking Demo Report</h1>",
    ]

    for mode in modes:
        model_cfg_run = dict(model_cfg)
        model_cfg_run["mask_fill_mode"] = mode
        # Default demo behavior: adaptive per-scale box masks.
        if args.rigid_mask_box:
            model_cfg_run["mask_scale_factor"] = 0.0
        else:
            model_cfg_run["mask_scale_factor"] = float(model_cfg.get("mask_scale_factor", 1.0))
        suffix = f"_{mode}"

        x_ctx, target_locations, target_scales, target_valid, debug = make_context_and_debug(x_t, model_cfg_run, int(args.seed))
        ctx = x_ctx[0, 0].cpu().numpy().astype(np.float32)
        mask = debug["mask_map"][0].cpu().numpy().astype(np.uint8)
        centers = extract_centers(debug)
        if centers.size == 0:
            centers = extract_centers_from_targets(target_locations, target_valid)

        # Overview panels use raw data so Original is uncontaminated and
        # Context-Original / Fractional Diff show the actual masking effect.
        x_display = x
        ctx_display = ctx
        x_diff = x.astype(np.float32)
        ctx_diff = ctx.astype(np.float32)
        x_frac = np.clip(x, 0.0, None).astype(np.float32)
        ctx_frac = np.clip(ctx, 0.0, None).astype(np.float32)

        model = build_model(model_cfg_run, data_cfg)
        ckpt_used = load_model_checkpoint_if_available(model, session_dir)
        model.eval()
        with torch.no_grad():
            out = model(x_t)
        x_clean_net = out["x_clean"][0, 0].cpu().numpy().astype(np.float32)
        x_context_net = out["x_context"][0, 0].cpu().numpy().astype(np.float32)
        pred_norm = out["pred_map"][0].detach().cpu().norm(dim=0).numpy().astype(np.float32)
        gt_norm = out["gt_map"][0].detach().cpu().norm(dim=0).numpy().astype(np.float32)

        html_parts.append(f"<h2>Mode: {mode}</h2>")
        fig_over = build_overview_figure(
            x_display,
            ctx_display,
            centers,
            debug,
            mode,
            title=f"Masking Overview ({mode})",
            frac_i1=x_frac,
            frac_i2=ctx_frac,
            diff_i1=x_diff,
            diff_i2=ctx_diff,
        )
        html_parts.append("<div class='panel'>")
        html_parts.append(pio.to_html(fig_over, include_plotlyjs="cdn", full_html=False, config={"responsive": True}))
        html_parts.append("</div>")
        fig_ch = build_channels_figure(debug, title=f"CDD Channels ({mode})")
        if fig_ch is not None:
            html_parts.append("<div class='panel'>")
            html_parts.append(pio.to_html(fig_ch, include_plotlyjs=False, full_html=False, config={"responsive": True}))
            html_parts.append("</div>")
        fig_net = build_network_figure(
            x_clean_net, x_context_net, pred_norm, gt_norm, title=f"Network Outputs ({mode})"
        )
        html_parts.append("<div class='panel'>")
        html_parts.append(pio.to_html(fig_net, include_plotlyjs=False, full_html=False, config={"responsive": True}))
        html_parts.append("</div>")
        if mode == "channel_mask":
            fig_mask_ch, dip_ch = build_channel_mask_figure(debug, title="CDD Channel Mask Per Channel")
            if fig_mask_ch is not None:
                html_parts.append("<div class='panel'>")
                html_parts.append(pio.to_html(fig_mask_ch, include_plotlyjs=False, full_html=False, config={"responsive": True}))
                html_parts.append("</div>")
                mask_path = os.path.join(session_dir, "masking_demo_cdd_mask_per_channel.npy")
                np.save(mask_path, dip_ch)
            else:
                mask_path = None
        else:
            mask_path = None

        eval_metrics = None
        eval_heatmap_path = None
        if int(args.eval_samples) > 0:
            heat, eval_metrics = evaluate_mask_symmetry(
                ds=ds,
                model_cfg_run=model_cfg_run,
                n_samples=int(args.eval_samples),
                base_seed=int(args.seed),
            )
            fig_eval = make_subplots(
                rows=1,
                cols=1,
                subplot_titles=(f"Mask Frequency Heatmap ({int(args.eval_samples)} samples)",),
            )
            fig_eval.add_trace(
                go.Heatmap(z=heat, colorscale="Magma", zmin=0.0, zmax=float(max(1e-6, np.max(heat))), showscale=True),
                row=1,
                col=1,
            )
            fig_eval.update_xaxes(constrain="domain", row=1, col=1)
            fig_eval.update_yaxes(scaleanchor="x1", scaleratio=1, row=1, col=1)
            fig_eval.update_layout(height=620, width=760, title=f"Mask Symmetry Eval ({mode})", margin=dict(l=20, r=20, t=50, b=20))
            html_parts.append("<div class='panel'>")
            html_parts.append(pio.to_html(fig_eval, include_plotlyjs=False, full_html=False, config={"responsive": True}))
            html_parts.append("</div>")
            html_parts.append("<div class='panel'>")
            html_parts.append(
                "<pre style='font-size:13px;background:#f7f7f7;border:1px solid #ddd;padding:10px;'>"
                f"Mask Symmetry Metrics ({mode})\\n"
                f"samples: {eval_metrics['samples']}\\n"
                f"mean_mask_value: {eval_metrics['mean_mask_value']:.6g}\\n"
                f"lr_symmetry_mae: {eval_metrics['lr_symmetry_mae']:.6g}\\n"
                f"tb_symmetry_mae: {eval_metrics['tb_symmetry_mae']:.6g}\\n"
                f"top_minus_bottom: {eval_metrics['top_minus_bottom']:.6g}\\n"
                f"left_minus_right: {eval_metrics['left_minus_right']:.6g}"
                "</pre>"
            )
            html_parts.append("</div>")
            eval_heatmap_path = os.path.join(session_dir, f"mask_symmetry_heatmap_{mode}.npy")
            np.save(eval_heatmap_path, heat)

        # CDD consistency checks for in-dashboard review.
        cdd_summary = None
        cdd_orig_t = debug.get("cdd_channels_orig")
        cdd_mask_t = debug.get("cdd_channels_masked")
        if cdd_orig_t is not None and cdd_mask_t is not None and cdd_orig_t.numel() > 0 and cdd_mask_t.numel() > 0:
            c0 = cdd_orig_t[0].cpu().numpy().astype(np.float64)
            c1 = cdd_mask_t[0].cpu().numpy().astype(np.float64)
            s0 = c0.sum(axis=0)
            s1 = c1.sum(axis=0)
            d = s0 - s1
            cdd_summary = {
                "all_c0_nonnegative": bool((c0 >= 0.0).all()),
                "all_c1_nonnegative": bool((c1 >= 0.0).all()),
                "all_sum_diff_nonnegative": bool((d >= 0.0).all()),
                "any_sum_masked_gt_orig": bool((s1 > s0).any()),
                "sum_diff_min": float(d.min()),
                "sum_diff_max": float(d.max()),
            }
            html_parts.append("<div class='panel'>")
            html_parts.append(
                "<pre style='font-size:13px;background:#f7f7f7;border:1px solid #ddd;padding:10px;'>"
                f"CDD Summary ({mode})\\n"
                f"all cdd_channels_orig >= 0: {cdd_summary['all_c0_nonnegative']}\\n"
                f"all cdd_channels_masked >= 0: {cdd_summary['all_c1_nonnegative']}\\n"
                f"all (sum_orig - sum_masked) >= 0: {cdd_summary['all_sum_diff_nonnegative']}\\n"
                f"any sum_masked > sum_orig: {cdd_summary['any_sum_masked_gt_orig']}\\n"
                f"min(sum_orig - sum_masked): {cdd_summary['sum_diff_min']:.6g}\\n"
                f"max(sum_orig - sum_masked): {cdd_summary['sum_diff_max']:.6g}"
                "</pre>"
            )
            html_parts.append("</div>")

        # Per-channel mask geometry table.
        box_sizes_t = debug.get("cdd_box_sizes")
        blur_sigmas_t = debug.get("cdd_blur_sigmas")
        if box_sizes_t is not None and box_sizes_t.numel() > 0:
            box_sizes = box_sizes_t[0].cpu().numpy()
            blur_sigmas = blur_sigmas_t[0].cpu().numpy() if blur_sigmas_t is not None and blur_sigmas_t.numel() > 0 else np.zeros_like(box_sizes)
            rows = []
            for i in range(len(box_sizes)):
                bs = float(blur_sigmas[i])
                fwhm = 2.355 * bs if bs > 0 else 0.0
                rows.append(f"  ch {i}: box={int(box_sizes[i])}px  blur_sigma={bs:.2f}px  FWHM={fwhm:.1f}px  2*FWHM={2*fwhm:.1f}px")
            html_parts.append("<div class='panel'>")
            html_parts.append(
                "<pre style='font-size:13px;background:#f7f7f7;border:1px solid #ddd;padding:10px;'>"
                f"Mask Geometry ({mode})\\n"
                + "\\n".join(rows)
                + "</pre>"
            )
            html_parts.append("</div>")

        meta = {
            "mask_fill_mode": mode,
            "masking_mode": "cdd",
            "checkpoint_used": ckpt_used,
            "use_cdd": bool(data_cfg.get("use_cdd", True)),
            "mask_fraction": float(model_cfg_run.get("active_target_fraction", model_cfg_run.get("mask_fraction", 1.0))),
            "mask_scale_factor": float(model_cfg_run.get("mask_scale_factor", 1.0)),
            "mask_spacing_scaling": float(model_cfg_run.get("mask_spacing_scaling", 1.5)),
            "sigmas": list(model_cfg_run.get("sigmas", [2, 4, 8, 16])),
            "global_realized_fraction": float(mask.mean()),
            "unique_centers": [[int(y), int(x)] for y, x in centers.tolist()],
            "target_scales": [float(s) for s in target_scales[0].cpu().numpy().tolist()],
            "target_locations": [[int(y), int(x)] for y, x in target_locations[0].cpu().numpy().tolist()],
            "cdd_mask_per_channel_path": mask_path,
            "mask_symmetry_heatmap_path": eval_heatmap_path,
            "mask_symmetry_metrics": eval_metrics,
            "cdd_summary": cdd_summary,
        }
        all_meta["modes"].append(meta)
        print(f"realized_fraction={meta['global_realized_fraction']:.6f}")

    html_parts.append("</body></html>")
    with open(report_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    if effective_single_mode == "zero":
        meta_name = f"masking_demo_meta_{config_name}_box_masked.json"
    elif effective_single_mode == "channel_mask":
        meta_name = f"masking_demo_meta_{config_name}_channel_mask_only.json"
    else:
        meta_name = f"masking_demo_meta_{config_name}_all.json"
    out_meta = os.path.join(session_dir, meta_name)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)
    print(f"session_saved={session_dir}")
    print(f"saved={report_html}")
    print(f"meta={out_meta}")


if __name__ == "__main__":
    main()
