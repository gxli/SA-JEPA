import argparse
import csv
import json
import os
import shutil
import sys

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _compute_pca_2d(x: np.ndarray) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA

        return PCA(n_components=2).fit_transform(x)
    except Exception:
        x_t = torch.from_numpy(x.astype(np.float32))
        x_t = x_t - x_t.mean(dim=0, keepdim=True)
        u, s, _ = torch.pca_lowrank(x_t, q=2)
        return (u[:, :2] * s[:2]).cpu().numpy()


def _compute_umap_2d(x: np.ndarray) -> np.ndarray:
    try:
        from cuml.manifold import UMAP as CuMLUMAP

        return CuMLUMAP(n_components=2, random_state=42).fit_transform(x)
    except Exception:
        pass

    try:
        import torchdr

        if hasattr(torchdr, "UMAP"):
            model = torchdr.UMAP(n_components=2)
            z = model.fit_transform(torch.from_numpy(x.astype(np.float32)))
            if isinstance(z, torch.Tensor):
                return z.cpu().numpy()
            return np.asarray(z)
    except Exception:
        pass

    try:
        import umap

        return umap.UMAP(n_components=2, random_state=42).fit_transform(x)
    except Exception:
        pass

    return _compute_pca_2d(x)


def _compute_pca_3d(x: np.ndarray) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA

        return PCA(n_components=3).fit_transform(x)
    except Exception:
        x_t = torch.from_numpy(x.astype(np.float32))
        x_t = x_t - x_t.mean(dim=0, keepdim=True)
        u, s, _ = torch.pca_lowrank(x_t, q=3)
        return (u[:, :3] * s[:3]).cpu().numpy()


def _compute_umap_3d(x: np.ndarray) -> np.ndarray:
    try:
        from cuml.manifold import UMAP as CuMLUMAP

        return CuMLUMAP(n_components=3, random_state=42).fit_transform(x)
    except Exception:
        pass

    try:
        import torchdr

        if hasattr(torchdr, "UMAP"):
            model = torchdr.UMAP(n_components=3)
            z = model.fit_transform(torch.from_numpy(x.astype(np.float32)))
            if isinstance(z, torch.Tensor):
                return z.cpu().numpy()
            return np.asarray(z)
    except Exception:
        pass

    try:
        import umap

        return umap.UMAP(n_components=3, random_state=42).fit_transform(x)
    except Exception:
        pass

    return _compute_pca_3d(x)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def _rgb_from_xyz(
    pts_xyz: np.ndarray,
    keep_mask: np.ndarray,
    h: int,
    w: int,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build HxWx3 uint8 image + flat RGB + normalization bounds from Nx3 points."""
    pts = np.asarray(pts_xyz, dtype=np.float32)
    if pts.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {pts.shape}")
    lo = np.percentile(pts, lo_pct, axis=0)
    hi = np.percentile(pts, hi_pct, axis=0)
    den = np.clip(hi - lo, 1e-8, None)
    rgb_valid = np.clip((pts - lo) / den, 0.0, 1.0)
    rgb_valid = np.clip(np.round(rgb_valid * 255.0), 0, 255).astype(np.uint8)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[keep_mask] = rgb_valid
    return rgb, rgb_valid, lo.astype(np.float32), hi.astype(np.float32)


def _resolve_crop_border(cfg: dict) -> int:
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    if "viz_crop_border_size" in train_cfg:
        return max(0, int(train_cfg.get("viz_crop_border_size", 0)))
    if "max_conv_size" in model_cfg:
        return max(0, int(model_cfg.get("max_conv_size", 0)))
    sigmas = model_cfg.get("sigmas", [])
    if isinstance(sigmas, (list, tuple)) and len(sigmas) > 0:
        return max(0, int(max(float(s) for s in sigmas)))
    return 0


def _center_mask(h: int, w: int, border: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    if border <= 0:
        m[:, :] = True
        return m
    y0 = border
    y1 = h - border
    x0 = border
    x1 = w - border
    if y1 <= y0 or x1 <= x0:
        # Degenerate case: keep all rather than empty.
        m[:, :] = True
        return m
    m[y0:y1, x0:x1] = True
    return m


def _nan_border(arr: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    out[~keep_mask] = np.nan
    return out


def _canonical_from_outputs(outputs: dict) -> dict:
    """
    Canonical fields:
      orig: HxW
      context: HxW
      target: HxW
      pred_latent: BxCxHxW
      gt_latent: BxCxHxW
    """
    if "x_clean" in outputs and "x_context" in outputs and "pred_map" in outputs and "gt_map" in outputs:
        x_clean = outputs["x_clean"]
        x_context = outputs["x_context"]
        pred_latent = outputs["pred_map"]
        gt_latent = outputs["gt_map"]
        orig = x_clean[0, 0].detach().cpu().numpy()
        context = x_context[0, 0].detach().cpu().numpy()
        if "target_map" in outputs:
            target = outputs["target_map"][0, 0].detach().cpu().numpy()
        else:
            target_locations = outputs["target_locations"]
            h, w = orig.shape
            target = np.zeros((h, w), dtype=np.float32)
            for i in range(target_locations.shape[1]):
                cy = int(target_locations[0, i, 0].item())
                cx = int(target_locations[0, i, 1].item())
                if 0 <= cy < h and 0 <= cx < w:
                    target[cy, cx] = 1.0
        return {
            "orig": orig,
            "context": context,
            "target": target,
            "pred_latent": pred_latent,
            "gt_latent": gt_latent,
        }

    # Legacy segmentation-like schema fallback.
    x_raw = outputs.get("x_raw")
    true_mask = outputs["true_mask"]
    pred_mask_logits = outputs["pred_mask_logits"]
    pred_latent = outputs["pred_latent"]
    gt_latent = outputs["gt_latent"]
    if x_raw is not None:
        orig = x_raw[0, 0].detach().cpu().numpy()
        context = x_raw[0, 1].detach().cpu().numpy() if x_raw.shape[1] > 1 else orig
    else:
        h, w = true_mask.shape[-2], true_mask.shape[-1]
        orig = np.zeros((h, w), dtype=np.float32)
        context = np.zeros((h, w), dtype=np.float32)
    target = true_mask[0, 0].detach().cpu().numpy()
    _ = pred_mask_logits
    return {
        "orig": orig,
        "context": context,
        "target": target,
        "pred_latent": pred_latent,
        "gt_latent": gt_latent,
    }


def compute_dash_data(session_dir: str, overwrite: bool = False) -> str:
    inf_path = os.path.join(session_dir, "inference_outputs.pt")
    out_npz = os.path.join(session_dir, "dash_data.npz")
    if os.path.exists(out_npz) and not overwrite:
        try:
            existing = np.load(out_npz)
            required = {
                "orig", "blurred", "target", "pred_mask", "target_loc_heatmap", "energy_map",
                "context_pca3d", "context_umap3d", "context_pca_rgb", "context_pca_rgb_flat", "context_pca_lo", "context_pca_hi",
                "context_umap_rgb", "context_umap_rgb_flat", "context_umap_lo", "context_umap_hi",
                "pred_pca3d", "pred_umap3d", "pred_pca_rgb", "pred_pca_rgb_flat", "pred_pca_lo", "pred_pca_hi",
                "pred_umap_rgb", "pred_umap_rgb_flat", "pred_umap_lo", "pred_umap_hi",
                "gt_pca3d", "gt_umap3d", "gt_pca_rgb", "gt_pca_rgb_flat", "gt_pca_lo", "gt_pca_hi",
                "gt_umap_rgb", "gt_umap_rgb_flat", "gt_umap_lo", "gt_umap_hi",
            }
            if required.issubset(set(existing.files)):
                existing.close()
                return out_npz
            existing.close()
        except Exception:
            pass
    if not os.path.exists(inf_path):
        raise FileNotFoundError(f"Missing inference outputs: {inf_path}")

    outputs = torch.load(inf_path, map_location="cpu")
    cfg_path = os.path.join(session_dir, "config_used.json")
    # Always L2-normalize vectors for PCA/UMAP visualizations.
    viz_l2_norm = True
    viz_crop_border = False
    viz_crop_border_size = 0
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            _ = bool(cfg.get("train", {}).get("viz_l2_normalize", True))
            viz_crop_border = bool(cfg.get("train", {}).get("viz_crop_border", False))
            if viz_crop_border:
                viz_crop_border_size = _resolve_crop_border(cfg)
        except Exception:
            pass
    canon = _canonical_from_outputs(outputs)
    orig = canon["orig"]
    blurred = canon["context"]
    target = canon["target"]
    energy_map = np.asarray(outputs.get("target_energy_map", torch.zeros((1, orig.shape[0], orig.shape[1])))[0], dtype=np.float32)
    context_latent = outputs.get("context_map", canon["pred_latent"])
    pred_latent = canon["pred_latent"]
    gt_latent = canon["gt_latent"]
    pred0 = pred_latent[0].detach().cpu()
    gt0 = gt_latent[0].detach().cpu()
    num = (pred0 - gt0).norm(dim=0)
    den = pred0.norm(dim=0) + gt0.norm(dim=0)
    pred_mask = (num / torch.clamp(den, min=1e-8)).numpy()
    target_loc_heatmap = np.zeros_like(pred_mask, dtype=np.float32)
    if "target_locations" in outputs and "target_valid" in outputs:
        tloc = outputs["target_locations"].detach().cpu().numpy()
        tvalid = outputs["target_valid"].detach().cpu().numpy().astype(bool)
        h_map, w_map = target_loc_heatmap.shape
        for bi in range(tloc.shape[0]):
            for ki in range(tloc.shape[1]):
                if not bool(tvalid[bi, ki]):
                    continue
                yy = int(tloc[bi, ki, 0])
                xx = int(tloc[bi, ki, 1])
                if 0 <= yy < h_map and 0 <= xx < w_map:
                    target_loc_heatmap[yy, xx] += 1.0
    visit_freq_path = os.path.join(session_dir, "visited_target_frequency.npy")
    if os.path.exists(visit_freq_path):
        try:
            target_loc_heatmap = np.asarray(np.load(visit_freq_path), dtype=np.float32)
        except Exception:
            pass
    # Prefer session-wide visitation for the Target Locations panel too,
    # so it reflects aggregate behavior rather than a sparse single sample.
    target_panel = target_loc_heatmap if np.any(np.isfinite(target_loc_heatmap)) else target
    h_lat, w_lat = int(pred_latent.shape[-2]), int(pred_latent.shape[-1])
    keep_mask = _center_mask(h_lat, w_lat, viz_crop_border_size if viz_crop_border else 0)

    def _vec_from_map(lat: torch.Tensor) -> np.ndarray:
        # Use sample 0 only, matching the displayed image panels.
        grid0 = lat.detach().cpu().permute(0, 2, 3, 1).numpy()[0]  # H,W,C
        if viz_crop_border:
            return grid0[keep_mask].reshape(-1, lat.shape[1])
        return grid0.reshape(-1, lat.shape[1])

    context_vec = _vec_from_map(context_latent)
    pred_vec = _vec_from_map(pred_latent)
    gt_vec = _vec_from_map(gt_latent)
    if viz_l2_norm:
        context_vec = _l2_normalize_rows(context_vec)
        pred_vec = _l2_normalize_rows(pred_vec)
        gt_vec = _l2_normalize_rows(gt_vec)

    context_pca3d = _compute_pca_3d(context_vec)
    context_umap3d = _compute_umap_3d(context_vec)
    pred_pca3d = _compute_pca_3d(pred_vec)
    pred_umap3d = _compute_umap_3d(pred_vec)
    gt_pca3d = _compute_pca_3d(gt_vec)
    gt_umap3d = _compute_umap_3d(gt_vec)

    context_pca_rgb, context_pca_rgb_flat, context_pca_lo, context_pca_hi = _rgb_from_xyz(context_pca3d, keep_mask, h_lat, w_lat)
    pred_pca_rgb, pred_pca_rgb_flat, pred_pca_lo, pred_pca_hi = _rgb_from_xyz(pred_pca3d, keep_mask, h_lat, w_lat)
    gt_pca_rgb, gt_pca_rgb_flat, gt_pca_lo, gt_pca_hi = _rgb_from_xyz(gt_pca3d, keep_mask, h_lat, w_lat)
    context_umap_rgb, context_umap_rgb_flat, context_umap_lo, context_umap_hi = _rgb_from_xyz(context_umap3d, keep_mask, h_lat, w_lat)
    pred_umap_rgb, pred_umap_rgb_flat, pred_umap_lo, pred_umap_hi = _rgb_from_xyz(pred_umap3d, keep_mask, h_lat, w_lat)
    gt_umap_rgb, gt_umap_rgb_flat, gt_umap_lo, gt_umap_hi = _rgb_from_xyz(gt_umap3d, keep_mask, h_lat, w_lat)

    metrics_path = os.path.join(session_dir, "metrics.csv")
    loss_x, loss_total, loss_jepa, loss_valid_frac = [], [], [], []
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ep = float(row["epoch"])
                    ba = float(row["batch"])
                    loss_x.append(ep + 0.001 * ba)
                    loss_total.append(float(row["total_loss"]))
                    loss_jepa.append(float(row["loss_jepa"]))
                    if "valid_frac" in row and row["valid_frac"] not in ("", None):
                        loss_valid_frac.append(float(row["valid_frac"]))
        except Exception:
            pass

    # Keep full-boundary fields for inference/header/error maps.
    # Border crop applies only to latent tokens used for PCA/UMAP vectors.
    orig_show = orig
    blurred_show = blurred
    target_show = target_panel
    pred_mask_show = pred_mask

    # Persist the scalar error map in session for downstream analysis.
    np.save(os.path.join(session_dir, "fractional_prediction_error.npy"), pred_mask_show.astype(np.float32))

    np.savez_compressed(
        out_npz,
        orig=orig_show,
        blurred=blurred_show,
        target=target_show,
        pred_mask=pred_mask_show,
        target_loc_heatmap=target_loc_heatmap.astype(np.float32),
        energy_map=energy_map.astype(np.float32),
        context_pca3d=context_pca3d,
        context_umap3d=context_umap3d,
        context_pca_rgb=context_pca_rgb,
        context_pca_rgb_flat=context_pca_rgb_flat,
        context_pca_lo=context_pca_lo,
        context_pca_hi=context_pca_hi,
        context_umap_rgb=context_umap_rgb,
        context_umap_rgb_flat=context_umap_rgb_flat,
        context_umap_lo=context_umap_lo,
        context_umap_hi=context_umap_hi,
        pred_pca3d=pred_pca3d,
        pred_umap3d=pred_umap3d,
        pred_pca_rgb=pred_pca_rgb,
        pred_pca_rgb_flat=pred_pca_rgb_flat,
        pred_pca_lo=pred_pca_lo,
        pred_pca_hi=pred_pca_hi,
        pred_umap_rgb=pred_umap_rgb,
        pred_umap_rgb_flat=pred_umap_rgb_flat,
        pred_umap_lo=pred_umap_lo,
        pred_umap_hi=pred_umap_hi,
        gt_pca3d=gt_pca3d,
        gt_umap3d=gt_umap3d,
        gt_pca_rgb=gt_pca_rgb,
        gt_pca_rgb_flat=gt_pca_rgb_flat,
        gt_pca_lo=gt_pca_lo,
        gt_pca_hi=gt_pca_hi,
        gt_umap_rgb=gt_umap_rgb,
        gt_umap_rgb_flat=gt_umap_rgb_flat,
        gt_umap_lo=gt_umap_lo,
        gt_umap_hi=gt_umap_hi,
        loss_x=np.asarray(loss_x, dtype=np.float32),
        loss_total=np.asarray(loss_total, dtype=np.float32),
        loss_jepa=np.asarray(loss_jepa, dtype=np.float32),
        loss_valid_frac=np.asarray(loss_valid_frac, dtype=np.float32),
    )
    return out_npz


def _axis_range_xy(x: np.ndarray, y: np.ndarray, pad_frac: float = 0.05):
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xr = max(1e-8, xmax - xmin)
    yr = max(1e-8, ymax - ymin)
    xpad = xr * pad_frac
    ypad = yr * pad_frac
    return [xmin - xpad, xmax + xpad], [ymin - ypad, ymax + ypad]


def _axis_range_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, pad_frac: float = 0.05):
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    zmin, zmax = float(np.min(z)), float(np.max(z))
    xr = max(1e-8, xmax - xmin)
    yr = max(1e-8, ymax - ymin)
    zr = max(1e-8, zmax - zmin)
    return (
        [xmin - xr * pad_frac, xmax + xr * pad_frac],
        [ymin - yr * pad_frac, ymax + yr * pad_frac],
        [zmin - zr * pad_frac, zmax + zr * pad_frac],
    )


def plot_dash_html(session_dir: str, overwrite: bool = False) -> str:
    npz_path = os.path.join(session_dir, "dash_data.npz")
    out_html = os.path.join(session_dir, "dashboard.html")
    if os.path.exists(out_html) and not overwrite:
        return out_html
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing computed dash data: {npz_path}")

    data = np.load(npz_path)
    required = {
        "target_loc_heatmap", "energy_map",
        "context_pca3d", "context_pca_rgb", "context_pca_rgb_flat", "context_pca_lo", "context_pca_hi",
        "context_umap3d", "context_umap_rgb", "context_umap_rgb_flat", "context_umap_lo", "context_umap_hi",
        "pred_pca3d", "pred_pca_rgb", "pred_pca_rgb_flat", "pred_pca_lo", "pred_pca_hi",
        "pred_umap3d", "pred_umap_rgb", "pred_umap_rgb_flat", "pred_umap_lo", "pred_umap_hi",
        "gt_pca3d", "gt_pca_rgb", "gt_pca_rgb_flat", "gt_pca_lo", "gt_pca_hi",
        "gt_umap3d", "gt_umap_rgb", "gt_umap_rgb_flat", "gt_umap_lo", "gt_umap_hi",
    }
    if not required.issubset(set(data.files)):
        data.close()
        compute_dash_data(session_dir, overwrite=True)
        data = np.load(npz_path)
    orig = data["orig"]
    blurred = data["blurred"]
    target = data["target"]
    target_loc_heatmap = data["target_loc_heatmap"] if "target_loc_heatmap" in data.files else target
    energy_map = data["energy_map"] if "energy_map" in data.files else np.zeros_like(target, dtype=np.float32)
    def _latent_bundle(prefix: str, kind: str) -> dict:
        k = f"{prefix}_{kind}"
        return {
            "xyz": data[f"{k}3d"],
            "rgb": data[f"{k}_rgb"],
            "rgb_flat": data[f"{k}_rgb_flat"],
            "lo": data[f"{k}_lo"],
            "hi": data[f"{k}_hi"],
        }

    sections = [
        ("Context", _latent_bundle("context", "pca"), _latent_bundle("context", "umap")),
        ("Predict", _latent_bundle("pred", "pca"), _latent_bundle("pred", "umap")),
        ("Target", _latent_bundle("gt", "pca"), _latent_bundle("gt", "umap")),
    ]
    loss_x = data["loss_x"] if "loss_x" in data.files else np.asarray([], dtype=np.float32)
    loss_total = data["loss_total"] if "loss_total" in data.files else np.asarray([], dtype=np.float32)
    loss_jepa = data["loss_jepa"] if "loss_jepa" in data.files else np.asarray([], dtype=np.float32)
    loss_valid_frac = data["loss_valid_frac"] if "loss_valid_frac" in data.files else np.asarray([], dtype=np.float32)

    fig = make_subplots(
        rows=10,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "scene"}],
            [{"type": "xy"}, {"type": "scene"}],
            [{"type": "xy"}, {"type": "scene"}],
            [{"type": "xy"}, {"type": "scene"}],
            [{"type": "xy"}, {"type": "scene"}],
            [{"type": "xy"}, {"type": "scene"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "Input (Log-Norm)", "CDD Blurred/Add-Back",
            "Target Locations", "Loss Curve",
            "Context PCA Color", "Context PCA Scatter",
            "Context UMAP Color", "Context UMAP Scatter",
            "Predict PCA Color", "Predict PCA Scatter",
            "Predict UMAP Color", "Predict UMAP Scatter",
            "Target PCA Color", "Target PCA Scatter",
            "Target UMAP Color", "Target UMAP Scatter",
            "Target Location Heatmap", "",
            "Energy Map", "",
        ),
        horizontal_spacing=0.06,
        vertical_spacing=0.06,
    )

    def _panel_heatmap(arr: np.ndarray, colorscale: str = "Viridis"):
        z = np.asarray(arr, dtype=np.float32)
        finite = z[np.isfinite(z)]
        if finite.size == 0:
            z = np.zeros_like(z, dtype=np.float32)
            zmin, zmax = 0.0, 1.0
        else:
            zmin = float(np.percentile(finite, 1.0))
            zmax = float(np.percentile(finite, 99.0))
            if zmax <= zmin + 1e-12:
                zmax = zmin + 1.0
        return go.Heatmap(z=z, colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=False)

    def _scatter3d(xyz: np.ndarray, rgb_flat: np.ndarray, name: str):
        return go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
            marker={"size": 2, "opacity": 0.45, "color": rgb_flat, "showscale": False},
            hovertemplate="x=%{x:.6g}<br>y=%{y:.6g}<br>z=%{z:.6g}<extra>" + name + "</extra>",
            name=name,
        )

    fig.add_trace(_panel_heatmap(orig, colorscale="Viridis"), row=1, col=1)
    fig.add_trace(_panel_heatmap(blurred, colorscale="Viridis"), row=1, col=2)
    fig.add_trace(_panel_heatmap(target, colorscale="Magma"), row=2, col=1)
    if loss_x.size > 0:
        fig.add_trace(go.Scattergl(x=loss_x, y=loss_total, mode="lines", name="total_loss"), row=2, col=2)
        fig.add_trace(go.Scattergl(x=loss_x, y=loss_jepa, mode="lines", name="loss_jepa"), row=2, col=2)
    fig.update_xaxes(title_text="epoch+0.001*batch", row=2, col=2)
    fig.update_yaxes(title_text="loss", row=2, col=2)

    scene_base = dict(aspectmode="cube", camera=dict(projection=dict(type="orthographic")))
    scene_layout_updates = {}
    scene_idx = 1
    start_row = 3
    for i, (title, pca, umap) in enumerate(sections):
        row_pca = start_row + 2 * i
        row_umap = row_pca + 1

        fig.add_trace(go.Image(z=pca["rgb"], name=f"{title} PCA map"), row=row_pca, col=1)
        fig.add_trace(_scatter3d(pca["xyz"], pca["rgb_flat"], f"{title} PCA3D"), row=row_pca, col=2)
        fig.add_trace(go.Image(z=umap["rgb"], name=f"{title} UMAP map"), row=row_umap, col=1)
        fig.add_trace(_scatter3d(umap["xyz"], umap["rgb_flat"], f"{title} UMAP3D"), row=row_umap, col=2)

        scene_layout_updates[f"scene{'' if scene_idx == 1 else scene_idx}"] = dict(
            scene_base,
            xaxis=dict(range=[float(pca["lo"][0]), float(pca["hi"][0])]),
            yaxis=dict(range=[float(pca["lo"][1]), float(pca["hi"][1])]),
            zaxis=dict(range=[float(pca["lo"][2]), float(pca["hi"][2])]),
        )
        scene_idx += 1
        scene_layout_updates[f"scene{scene_idx}"] = dict(
            scene_base,
            xaxis=dict(range=[float(umap["lo"][0]), float(umap["hi"][0])]),
            yaxis=dict(range=[float(umap["lo"][1]), float(umap["hi"][1])]),
            zaxis=dict(range=[float(umap["lo"][2]), float(umap["hi"][2])]),
        )
        scene_idx += 1

    fig.add_trace(_panel_heatmap(target_loc_heatmap, colorscale="Magma"), row=9, col=1)
    fig.add_trace(_panel_heatmap(energy_map, colorscale="Inferno"), row=10, col=1)

    for r in range(1, 11):
        sub = fig.get_subplot(r, 1)
        xname = str(sub.xaxis.plotly_name).replace("axis", "")
        fig.update_xaxes(showticklabels=False, row=r, col=1, constrain="domain")
        fig.update_yaxes(showticklabels=False, row=r, col=1, scaleanchor=xname, scaleratio=1, constrain="domain")
    # Keep CDD imshow panel square as well.
    fig.update_xaxes(showticklabels=False, row=1, col=2, constrain="domain")
    fig.update_yaxes(showticklabels=False, row=1, col=2, scaleanchor="x2", scaleratio=1, constrain="domain")

    fig.update_layout(**scene_layout_updates)
    fig.update_layout(
        title={"text": f"JEPA Session Dashboard: {os.path.basename(session_dir)}", "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        width=1700,
        height=4400,
        margin=dict(l=20, r=20, t=90, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def _preferred_html_for_export(session_dir: str, fallback_html: str) -> str:
    # Export rebuilt dashboard and append masking-demo panels when present.
    demo_files = sorted(
        [
            fn
            for fn in os.listdir(session_dir)
            if fn.startswith("masking_demo_") and fn.endswith(".html")
        ]
    )
    if not demo_files:
        return fallback_html
    with open(fallback_html, "r", encoding="utf-8") as f:
        html = f.read()
    parts = [
        "<hr/>",
        "<h2 style='font-family:sans-serif;margin:16px 0 8px 0;'>Masking Demo Panels</h2>",
    ]
    for fn in demo_files:
        parts.append(
            f"<div style='margin:10px 0;'>"
            f"<div style='font-family:sans-serif;font-size:14px;margin:4px 0;'>{fn}</div>"
            f"<iframe src=\"{fn}\" style='width:100%;height:980px;border:1px solid #ddd;border-radius:6px;'></iframe>"
            f"</div>"
        )
    html = html.replace("</body>", "\n" + "\n".join(parts) + "\n</body>")
    out_html = os.path.join(session_dir, "dashboard_with_masking_demo.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html


def _regenerate_inference_if_missing(session_dir: str) -> bool:
    """
    Regenerate inference artifacts for a session without retraining.
    Returns True if inference_outputs.pt exists after this call.
    """
    inf_path = os.path.join(session_dir, "inference_outputs.pt")
    if os.path.exists(inf_path):
        return True

    cfg_candidates = [
        os.path.join(session_dir, "config_used.json"),
        os.path.join(session_dir, "resolved_config.json"),
    ]
    cfg_path = next((p for p in cfg_candidates if os.path.exists(p)), None)
    if cfg_path is None:
        return False

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.setdefault("train", {})
        # Skip training loop and force post-training inference rebuild.
        cfg["train"]["epochs"] = 0
        cfg["train"]["force_recompute_inference"] = True

        from src.train import run_training

        config_name = os.path.basename(session_dir.rstrip(os.sep))
        sessions_root = os.path.dirname(session_dir.rstrip(os.sep))
        print(f"regen_inference_start session={config_name}")
        run_training(cfg, config_name=config_name, sessions_root=sessions_root)
        print(f"regen_inference_done session={config_name}")
    except Exception as e:
        print(f"regen_inference_failed session={session_dir} reason={type(e).__name__}: {e}")
        return False

    return os.path.exists(inf_path)


def main():
    parser = argparse.ArgumentParser(description="Build dashboards from existing sessions")
    parser.add_argument("--sessions-dir", type=str, default="sessions")
    parser.add_argument("--export-dir", type=str, default="results/dashboard", help="Export dashboards here for fast sync")
    parser.add_argument("--stage", type=str, choices=["compute", "plot", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if output exists")
    parser.add_argument("--reset", action="store_true", help="Delete session dash artifacts first, then rebuild")
    parser.add_argument(
        "--regen-missing-inference",
        action="store_true",
        default=False,
        help="If inference_outputs.pt is missing, regenerate it from session config/checkpoint.",
    )
    parser.add_argument(
        "--no-regen-missing-inference",
        action="store_false",
        dest="regen_missing_inference",
        help="Disable auto-regeneration when inference_outputs.pt is missing.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.sessions_dir):
        raise FileNotFoundError(f"Sessions dir not found: {args.sessions_dir}")
    export_dir = args.export_dir
    if not os.path.isabs(export_dir):
        export_dir = os.path.join(os.getcwd(), export_dir)
    os.makedirs(export_dir, exist_ok=True)
    processed = 0
    skipped = 0
    exported = 0
    missing_inference = []
    failed_sessions = []
    for name in sorted(os.listdir(args.sessions_dir)):
        session_dir = os.path.join(args.sessions_dir, name)
        if not os.path.isdir(session_dir):
            continue
        inf_path = os.path.join(session_dir, "inference_outputs.pt")
        if not os.path.exists(inf_path):
            if args.regen_missing_inference:
                ok = _regenerate_inference_if_missing(session_dir)
                if ok:
                    inf_path = os.path.join(session_dir, "inference_outputs.pt")
            if not os.path.exists(inf_path):
                print(f"skip_no_inference={session_dir}")
                missing_inference.append(session_dir)
                skipped += 1
                continue

        try:
            if args.reset:
                for p in (
                    os.path.join(session_dir, "dash_data.npz"),
                    os.path.join(session_dir, "dashboard.html"),
                ):
                    if os.path.exists(p):
                        os.remove(p)
                # Remove exported dashboard files for this session so run is guaranteed fresh.
                safe_name = str(name).replace("/", "_")
                for p in (
                    os.path.join(export_dir, f"{safe_name}.html"),
                ):
                    if os.path.exists(p):
                        os.remove(p)
            if args.stage in ("compute", "all"):
                npz_path = compute_dash_data(session_dir, overwrite=args.overwrite)
                print(f"dash_data_saved={npz_path}")
            if args.stage in ("plot", "all"):
                if not os.path.exists(os.path.join(session_dir, "dash_data.npz")):
                    compute_dash_data(session_dir, overwrite=args.overwrite)
                html_path = plot_dash_html(session_dir, overwrite=args.overwrite)
                print(f"dashboard_html_saved={html_path}")
                # Export canonical HTML dashboard to a flat file: results/plots/<session_name>.html
                safe_name = str(name).replace("/", "_")
                html_src = _preferred_html_for_export(session_dir, html_path)
                html_export = os.path.join(export_dir, f"{safe_name}.html")
                shutil.copy2(html_src, html_export)
                print(f"dashboard_html_exported={html_export}")
                # Keep plot artifacts centralized under export_dir only.
                if os.path.exists(html_path):
                    os.remove(html_path)
                exported += 1
            processed += 1
        except FileNotFoundError as e:
            print(f"skip_missing_data={session_dir} reason={e}")
            failed_sessions.append((session_dir, f"FileNotFoundError: {e}"))
            skipped += 1
        except Exception as e:
            print(f"skip_error={session_dir} reason={type(e).__name__}: {e}")
            failed_sessions.append((session_dir, f"{type(e).__name__}: {e}"))
            skipped += 1

    print(
        f"dash_summary processed={processed} exported={exported} skipped={skipped} "
        f"missing_inference={len(missing_inference)} failed={len(failed_sessions)} sessions_dir={args.sessions_dir}"
    )
    if missing_inference:
        print("missing_inference_sessions_begin")
        for s in missing_inference:
            print(s)
        print("missing_inference_sessions_end")
    if failed_sessions:
        print("failed_sessions_begin")
        for s, msg in failed_sessions:
            print(f"{s} :: {msg}")
        print("failed_sessions_end")


if __name__ == "__main__":
    main()
