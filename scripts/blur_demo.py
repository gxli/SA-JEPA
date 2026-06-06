#!/usr/bin/env python3
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

##
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import JEPADataset


def pyramid_blur_center(
    x: torch.Tensor,
    center_y: int,
    center_x: int,
    scales=(2, 4, 8, 16),
    radius: int = 32,
) -> torch.Tensor:
    x_context = x.clone()

    _, c, h, w = x.shape
    # Channel-matched masking: zero out scale-sized boxes.
    if len(scales) == 0:
        return x_context
    max_sigma = float(max(scales))
    for ch in range(c):
        sigma = float(scales[min(ch, len(scales) - 1)])
        rr = max(2, int(round(radius * (sigma / max_sigma)))) if max_sigma > 0 else int(radius)
        y0 = max(0, center_y - rr)
        y1 = min(h, center_y + rr)
        x0 = max(0, center_x - rr)
        x1 = min(w, center_x + rr)
        x_context[:, ch : ch + 1, y0:y1, x0:x1] = 0.0
    return x_context


def constrained_diffusion_decomposition(
    arr: np.ndarray,
    scales=(2, 4, 8),
    strength: float = 1.0,
    mode: str = "log",
    constrained: bool = True,
    sm_mode: str = "reflect",
) -> np.ndarray:
    # Use the package's native constrained_diffusion_decomposition implementation.
    import constrained_diffusion as cdd

    num_channels = max(1, len(scales))
    result, residual, out_scales = cdd.constrained_diffusion_decomposition(
        arr,
        num_channels=num_channels,
        mode=mode,
        constrained=constrained,
        sm_mode=sm_mode,
        return_scales=True,
        verbose=False,
        use_gpu=False,
    )

    # Reconstruct from true decomposition outputs.
    result = np.asarray(result, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    orig = np.asarray(arr, dtype=np.float32)
    cdd_sum = np.sum(result, axis=0)
    recon = cdd_sum + residual
    detail = (recon - orig) * float(strength)

    channels = np.stack([orig, recon, detail], axis=0).astype(np.float32)
    meta = {
        "package_scales": [float(s) for s in np.asarray(out_scales).tolist()],
        "num_channels": int(result.shape[0]),
        "mode": mode,
        "constrained": bool(constrained),
        "sm_mode": sm_mode,
    }
    return channels, meta, result.astype(np.float32), residual.astype(np.float32)


def _choose_log_eps(arr: np.ndarray, cfg_eps: float = None) -> float:
    pos = arr[arr > 0]
    if pos.size == 0:
        return float(cfg_eps) if cfg_eps is not None else 1e-12
    p10 = float(np.percentile(pos, 10))
    auto_eps = max(1e-30, p10 * 1e-3)
    if cfg_eps is None:
        return auto_eps
    # Safety cap: prevent oversized epsilon from flattening tiny-valued fields.
    return min(float(cfg_eps), max(1e-30, p10 * 1e-2))


def load_input(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D or 3D npy array, got shape {arr.shape}")
        arr = np.asarray(arr, dtype=np.float32)
    else:
        img = plt.imread(path)
        if img.ndim == 3:
            arr = img.mean(axis=2).astype(np.float32)
        elif img.ndim == 2:
            arr = img.astype(np.float32)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def post_log_transform_channels(ch: np.ndarray, log_eps: float = None) -> np.ndarray:
    out = ch.copy()
    eps0 = _choose_log_eps(out[0], cfg_eps=log_eps)
    eps1 = _choose_log_eps(out[1], cfg_eps=log_eps)
    out[0] = np.log(np.clip(out[0], a_min=0.0, a_max=None) + eps0)
    out[1] = np.log(np.clip(out[1], a_min=0.0, a_max=None) + eps1)
    out[2] = np.sign(out[2]) * np.log1p(np.abs(out[2]))
    return out


def to_display_image(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    if hi <= lo + 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def to_display_diverging(arr: np.ndarray, p: float = 99.0) -> np.ndarray:
    lim = float(np.percentile(np.abs(arr), p))
    if lim <= 1e-12:
        lim = 1e-12
    out = 0.5 + 0.5 * (arr / lim)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def identical_log_map(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Shared monotonic compression used consistently across panels for visibility.
    return np.sign(arr) * np.log1p(np.abs(arr) / float(eps))


def _robust_norm(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if hi <= lo + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def pair_display_maps(orig: np.ndarray, mod: np.ndarray, signed: bool = False):
    if signed:
        eps = max(1e-12, float(np.percentile(np.abs(np.concatenate([orig.ravel(), mod.ravel()])), 10)) * 1e-3)
        o = identical_log_map(orig, eps=eps)
        m = identical_log_map(mod, eps=eps)
        lim = max(1e-12, float(np.percentile(np.abs(np.concatenate([o.ravel(), m.ravel()])), 99.5)))
        o_vis = np.clip(0.5 + 0.5 * (o / lim), 0.0, 1.0).astype(np.float32)
        m_vis = np.clip(0.5 + 0.5 * (m / lim), 0.0, 1.0).astype(np.float32)
        return o_vis, m_vis, "seismic"

    pair = np.concatenate([orig.ravel(), mod.ravel()])
    nz = pair[pair > 0]
    eps = max(1e-20, float(np.percentile(nz, 5)) * 1e-2) if nz.size > 0 else 1e-20
    o = np.log1p(np.clip(orig, a_min=0.0, a_max=None) / eps)
    m = np.log1p(np.clip(mod, a_min=0.0, a_max=None) / eps)
    z = np.concatenate([o.ravel(), m.ravel()])
    lo = float(np.percentile(z, 0.5))
    hi = float(np.percentile(z, 99.5))
    if hi <= lo + 1e-12:
        # Fallback to z-score style contrast if percentile span collapses.
        mu = float(np.mean(z))
        sd = float(np.std(z)) + 1e-12
        o_vis = np.clip(0.5 + 0.2 * ((o - mu) / sd), 0.0, 1.0).astype(np.float32)
        m_vis = np.clip(0.5 + 0.2 * ((m - mu) / sd), 0.0, 1.0).astype(np.float32)
    else:
        o_vis = np.clip((o - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        m_vis = np.clip((m - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return o_vis, m_vis, "viridis"


def channel_sigma_map(num_channels: int, scales) -> list:
    if len(scales) == 0:
        return [0.0 for _ in range(num_channels)]
    return [float(scales[i % len(scales)]) for i in range(num_channels)]


def find_input_path(data_cfg: dict) -> str:
    if data_cfg.get("input_path"):
        return data_cfg["input_path"]
    data_root = data_cfg.get("data_root", "data")
    npy_pattern = data_cfg.get("npy_pattern", "*.npy")
    import glob

    files = sorted(glob.glob(os.path.join(data_root, npy_pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in {data_root} with pattern {npy_pattern}")
    return files[0]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Config-driven blur demo")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--sessions-dir", type=str, default="sessions", help="Session output root")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    demo_cfg = cfg.get("blur_demo", {})
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    session_dir = os.path.join(args.sessions_dir, config_name)
    os.makedirs(session_dir, exist_ok=True)

    input_path = find_input_path(data_cfg)
    cfg_log_eps = data_cfg.get("log_eps", None)

    # Use the main dataset pipeline for full consistency with training.
    dataset = JEPADataset(
        num_samples=max(1, int(data_cfg.get("num_samples", 1))),
        data_root=data_cfg.get("data_root", "data"),
        npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
        cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
        cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
        cube_slice_index=data_cfg.get("cube_slice_index", 0),
    )
    sample_idx = int(demo_cfg.get("sample_index", 0))
    x_sample = dataset[sample_idx]  # 3 x H x W (training-view channels)
    x_np = x_sample.numpy().astype(np.float32)

    h, w = x_np.shape[-2], x_np.shape[-1]
    cy = demo_cfg.get("center_y")
    cx = demo_cfg.get("center_x")
    cy = h // 2 if cy is None else int(cy)
    cx = w // 2 if cx is None else int(cx)
    cy = int(np.clip(cy, 0, h - 1))
    cx = int(np.clip(cx, 0, w - 1))

    scales = tuple(demo_cfg.get("scales", [1, 2, 4]))
    make_channel_plot = bool(demo_cfg.get("make_channel_plot", True))
    radius = int(demo_cfg.get("radius", 32))
    num_centers_cfg = demo_cfg.get("num_random_centers", "auto")
    use_grid_centers = bool(demo_cfg.get("use_grid_centers", True))
    center_jitter = int(demo_cfg.get("center_jitter", 8))
    grid_size = int(demo_cfg.get("grid_size", 8))
    grid_rows = demo_cfg.get("grid_rows")
    grid_cols = demo_cfg.get("grid_cols")
    no_gap_boxes = bool(demo_cfg.get("no_gap_boxes", False))
    spacing_consistent = bool(demo_cfg.get("spacing_consistent", True))
    mask_scale = float(demo_cfg.get("mask_scale_factor", 1.0))
    pyramid_spacing_mult = float(demo_cfg.get("pyramid_spacing_mult", 2.0))
    seed = int(demo_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    cdd_scales = tuple(model_cfg.get("sigmas", [2, 4, 8, 16]))
    cdd_mode = data_cfg.get("cdd_mode", "log")
    cdd_constrained = bool(data_cfg.get("cdd_constrained", True))
    cdd_sm_mode = data_cfg.get("cdd_sm_mode", "reflect")
    cdd_meta = {
        "num_channels": 3,
        "mode": cdd_mode,
        "constrained": cdd_constrained,
        "sm_mode": cdd_sm_mode,
        "source": "JEPADataset",
    }
    # Run package CDD directly on original image to get true scale-component channels.
    arr_linear = load_input(input_path)
    amin = float(arr_linear.min())
    amax = float(arr_linear.max())
    denom = amax - amin
    if denom > 1e-20:
        arr_linear = (arr_linear - amin) / denom
    else:
        arr_linear = np.zeros_like(arr_linear, dtype=np.float32)
    _, cdd_pkg_meta, cdd_full_result, cdd_residual = constrained_diffusion_decomposition(
        arr_linear,
        scales=cdd_scales,
        strength=1.0,
        mode=cdd_mode,
        constrained=cdd_constrained,
        sm_mode=cdd_sm_mode,
    )
    # Blur the actual CDD scale components (all channels), then reconstruct image.
    x_comp = torch.from_numpy(cdd_full_result.astype(np.float32)).float().unsqueeze(0)  # 1 x Nc x H x W
    x = x_comp

    # Non-overlapping centers: fixed grid + one global random shift (x/y),
    # then discard centers too close to boundaries.
    centers = []
    explicit_box_half = demo_cfg.get("box_half")
    draw_half = int(explicit_box_half) if explicit_box_half is not None else int(radius)
    if use_grid_centers:
        # Determine base box half-size from config/grid intent.
        if grid_rows is not None and grid_cols is not None:
            nrows = max(1, int(grid_rows))
            ncols = max(1, int(grid_cols))
        else:
            nrows = max(1, grid_size)
            ncols = max(1, grid_size)
        cell_h = max(1, h // nrows)
        cell_w = max(1, w // ncols)
        mean_spacing = float((cell_h + cell_w) / 2.0)
        if explicit_box_half is None and spacing_consistent:
            draw_half = int(max(2, round(0.5 * mean_spacing)))
            center_jitter = int(max(0, round(0.5 * mean_spacing)))
        if explicit_box_half is None:
            draw_half = int(max(2, min(cell_h, cell_w) // 2))
        draw_half = int(max(2, round(draw_half * mask_scale)))

        # Largest active mask half-size from channel-matched scales.
        c_eff = max(1, x_np.shape[0])
        max_sigma = float(max(scales)) if len(scales) > 0 else 1.0
        largest_active_sigma = float(scales[min(c_eff - 1, len(scales) - 1)]) if len(scales) > 0 else 1.0
        largest_box_half = int(max(2, round(draw_half * (largest_active_sigma / max_sigma)))) if max_sigma > 0 else draw_half

        # Spacing rule requested:
        # spacing_px = largest_scale * mask_scale * spacing_scale
        effective_half = float(largest_box_half)
        largest_scale = float(max(scales)) if len(scales) > 0 else 1.0
        spacing_target = float(largest_scale) * float(mask_scale) * float(pyramid_spacing_mult)
        margin_y = int(max(0, largest_box_half))
        margin_x = int(max(0, largest_box_half))

        auto_pyramid_count = bool(
            demo_cfg.get(
                "auto_pyramid_count",
                isinstance(num_centers_cfg, str) and num_centers_cfg.lower() == "auto",
            )
        )
        spacing_px = int(max(1, round(spacing_target)))

        # Build grid-like lattice either from fixed rows/cols or from spacing-driven auto packing.
        if auto_pyramid_count:
            y_positions = np.arange(margin_y, h - margin_y, spacing_px, dtype=int)
            x_positions = np.arange(margin_x, w - margin_x, spacing_px, dtype=int)
        else:
            if nrows > 1:
                y_positions = np.round(np.linspace(margin_y, h - 1 - margin_y, nrows)).astype(int)
            else:
                y_positions = np.array([(h - 1) // 2], dtype=int)
            if ncols > 1:
                x_positions = np.round(np.linspace(margin_x, w - 1 - margin_x, ncols)).astype(int)
            else:
                x_positions = np.array([(w - 1) // 2], dtype=int)

        # Strict lattice: spacing from largest mask footprint, one global shift.
        spacing_px = int(max(1, round(spacing_target)))
        shift_y = int(rng.integers(0, spacing_px))
        shift_x = int(rng.integers(0, spacing_px))

        y_start = int(margin_y + shift_y)
        x_start = int(margin_x + shift_x)
        y_limit = int(h - margin_y)
        x_limit = int(w - margin_x)

        y_shifted = np.arange(y_start, y_limit, spacing_px, dtype=int)
        x_shifted = np.arange(x_start, x_limit, spacing_px, dtype=int)
        if y_shifted.size == 0:
            y_shifted = np.array([int((margin_y + y_limit - 1) // 2)], dtype=int)
        if x_shifted.size == 0:
            x_shifted = np.array([int((margin_x + x_limit - 1) // 2)], dtype=int)

        shifted = [(int(yy), int(xx)) for yy in y_shifted for xx in x_shifted]

        if isinstance(num_centers_cfg, str) and num_centers_cfg.lower() == "auto":
            num_centers = len(shifted)
        else:
            num_centers = int(num_centers_cfg)
        centers = shifted[:num_centers]
    else:
        centers = [(cy, cx)]
        num_centers = 1
        trials = 0
        while len(centers) < num_centers and trials < 5000:
            trials += 1
            yy = int(rng.integers(0, h))
            xx = int(rng.integers(0, w))
            ok = True
            for py, px in centers:
                min_sep = max(2 * draw_half + 2, 8)
                if (yy - py) ** 2 + (xx - px) ** 2 < (min_sep * min_sep):
                    ok = False
                    break
            if ok:
                centers.append((yy, xx))

    sigma_map = channel_sigma_map(x.shape[1], scales)
    max_sigma = float(max(scales)) if len(scales) > 0 else 1.0
    if bool(demo_cfg.get("channel_box_from_scales", True)):
        channel_box_half = [int(max(1, min(draw_half, round(s)))) for s in sigma_map]
    else:
        channel_box_half = [
            max(2, int(round(draw_half * (s / max_sigma)))) if max_sigma > 0 else draw_half for s in sigma_map
        ]
    x_blur = x.clone()
    for yy, xx in centers:
        x_blur = pyramid_blur_center(
            x_blur,
            center_y=yy,
            center_x=xx,
            scales=scales,
            radius=draw_half,
        )

    # Reconstruct original/modified from CDD components + residual for main 4-panel checks.
    orig = np.sum(x.cpu().numpy()[0], axis=0) + cdd_residual
    blur = np.sum(x_blur.cpu().numpy()[0], axis=0) + cdd_residual
    # Apply identical log transform before display normalization.
    shared_eps = float(max(1e-12, np.percentile(np.abs(orig), 10) * 1e-3))
    orig_log = identical_log_map(orig, eps=shared_eps)
    blur_log = identical_log_map(blur, eps=shared_eps)
    orig_vis = to_display_image(orig_log)
    blur_vis = to_display_image(blur_log)
    ratio = blur / np.clip(orig, 1e-12, None)
    ratio_log = identical_log_map(ratio, eps=shared_eps)
    ratio_vis = to_display_image(ratio_log, p_low=2.0, p_high=98.0)
    frac = (blur - orig) / np.clip(blur + orig, 1e-12, None)
    frac_log = identical_log_map(frac, eps=max(1e-12, np.percentile(np.abs(frac), 10) * 1e-3))
    frac_vis = to_display_diverging(frac_log, p=99.0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, img, title, cmap in [
        (axes[0], orig_vis, "Original", "viridis"),
        (axes[1], blur_vis, "Center Masked", "viridis"),
        (axes[2], ratio_vis, "Ratio (I2 / I1)", "viridis"),
        (axes[3], frac_vis, "Frac Change (I2-I1)/(I2+I1)", "seismic"),
    ]:
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        ys = [p[0] for p in centers]
        xs = [p[1] for p in centers]
        ax.scatter(xs, ys, s=14, c="red", marker="x")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join(session_dir, "blur_demo.png")
    plt.savefig(out_path, dpi=180)
    plt.close(fig)

    channels_path = os.path.join(session_dir, "blur_demo_channels.png")
    if make_channel_plot:
        # Per-channel visualization with explicit masking assignment on CDD scale components.
        n_ch = x.shape[1]
        fig2, axes2 = plt.subplots(n_ch, 3, figsize=(14, 3.2 * n_ch))
        if n_ch == 1:
            axes2 = np.array([axes2])
        for ch in range(n_ch):
            ch_orig = x[0, ch].cpu().numpy()
            ch_blur = x_blur[0, ch].cpu().numpy()
            signed = bool(np.nanmin(ch_orig) < 0 or np.nanmin(ch_blur) < 0)
            ch_orig_vis, ch_blur_vis, cmap = pair_display_maps(ch_orig, ch_blur, signed=signed)
            ch_delta = ch_blur - ch_orig
            delta_eps = max(1e-20, float(np.percentile(np.abs(ch_delta), 10)) * 1e-2)
            ch_delta_vis = to_display_diverging(identical_log_map(ch_delta, eps=delta_eps))

            ax_l = axes2[ch, 0]
            ax_r = axes2[ch, 1]
            ax_d = axes2[ch, 2]
            ax_l.imshow(ch_orig_vis, cmap=cmap)
            ax_l.set_title(f"Channel {ch} Original")
            ax_l.axis("off")

            ax_r.imshow(ch_blur_vis, cmap=cmap)
            ax_r.set_title(f"Channel {ch} Masked (scale={sigma_map[ch]:g})")
            ax_r.axis("off")

            ax_d.imshow(ch_delta_vis, cmap="seismic")
            ax_d.set_title(f"Channel {ch} Delta")
            ax_d.axis("off")

            for ax in (ax_l, ax_r, ax_d):
                for yy, xx in centers:
                    rect = plt.Rectangle(
                        (xx - channel_box_half[ch], yy - channel_box_half[ch]),
                        2 * channel_box_half[ch],
                        2 * channel_box_half[ch],
                        fill=False,
                        edgecolor="red",
                        linewidth=1.3,
                    )
                    ax.add_patch(rect)

        plt.tight_layout()
        plt.savefig(channels_path, dpi=180)
        plt.close(fig2)
    # Save full package CDD decomposition channels as .npy for direct reference.
    # Layout: [channel, y, x] includes all channels returned by package CDD.
    cdd_npy_path = os.path.join(session_dir, "cdd_result.npy")
    np.save(cdd_npy_path, cdd_full_result.astype(np.float32))
    cdd_residual_path = os.path.join(session_dir, "cdd_residual.npy")
    np.save(cdd_residual_path, cdd_residual.astype(np.float32))

    meta = {
        "config": args.config,
        "input_path": input_path,
        "log_eps_used": _choose_log_eps(np.load(input_path).astype(np.float32), cfg_eps=cfg_log_eps)
        if input_path.endswith(".npy")
        else cfg_log_eps,
        "scales": list(scales),
        "channel_sigma_map": sigma_map,
        "model_sigmas": list(cdd_scales),
        "decomposition_name": "constrained_diffusion_decomposition",
        "decomposition_then_log": True,
        "cdd_meta": cdd_meta,
        "cdd_package_meta": cdd_pkg_meta,
        "radius": radius,
        "draw_half": draw_half,
        "channel_box_half": channel_box_half,
        "centers": centers,
        "use_grid_centers": use_grid_centers,
        "grid_size": grid_size,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "no_gap_boxes": no_gap_boxes,
        "spacing_consistent": spacing_consistent,
        "mask_scale_factor": mask_scale,
        "effective_mask_half": int(round(effective_half)) if use_grid_centers else int(draw_half),
        "effective_mask_full": int(round(2.0 * effective_half)) if use_grid_centers else int(2 * draw_half),
        "spacing_target": float(spacing_target) if use_grid_centers else None,
        "spacing_px": int(spacing_px) if use_grid_centers else None,
        "spacing_formula": "largest_scale * mask_scale_factor * spacing_scale",
        "spacing_px": int(spacing_px) if use_grid_centers else None,
        "auto_pyramid_count": auto_pyramid_count if use_grid_centers else None,
        "pyramid_spacing_mult": pyramid_spacing_mult,
        "center_jitter": center_jitter,
        "num_random_centers": num_centers,
        "seed": seed,
        "global_shift_mode": True,
        "global_shift_xy": [shift_y, shift_x] if use_grid_centers else [0, 0],
        "shared_log_eps": shared_eps,
        "output": out_path,
        "channels_output": channels_path if make_channel_plot else None,
        "cdd_result_npy": cdd_npy_path,
        "cdd_residual_npy": cdd_residual_path,
        "num_channels": int(x.shape[1]),
        "num_scales": int(len(scales)),
        "make_channel_plot": make_channel_plot,
    }
    with open(os.path.join(session_dir, "blur_demo_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"num_scales={len(scales)}")
    print(f"channel_plot_generated={make_channel_plot}")
    print(f"session_saved={session_dir}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
