from __future__ import annotations

import argparse
import csv
import os
import shutil
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch


DASHBOARD_VERSION = "scatter3d-v1"

DASH_DATA_REQUIRED = {
    "orig",
    "target",
    "target_loc_heatmap",
    "energy_map",
    "visit_heatmap",
    "context_pca3d",
    "context_umap3d",
    "context_pca_rgb",
    "context_pca_rgb_flat",
    "context_umap_rgb",
    "context_umap_rgb_flat",
    "pred_pca3d",
    "pred_umap3d",
    "pred_pca_rgb",
    "pred_pca_rgb_flat",
    "pred_umap_rgb",
    "pred_umap_rgb_flat",
    "gt_pca3d",
    "gt_umap3d",
    "gt_pca_rgb",
    "gt_pca_rgb_flat",
    "gt_umap_rgb",
    "gt_umap_rgb_flat",
}


def _verbose_artifact_report(session_dir: str) -> list[str]:
    missing: list[str] = []
    results_dir = os.path.join(session_dir, "results")
    if not os.path.isdir(results_dir):
        missing.append(f"missing_dir: {results_dir}")
        return missing
    # Core branch artifacts expected from src/train.py save_inference_dashboard().
    for branch in ("predict", "target"):
        for fn in (
            f"{branch}_pca_xyz.npy",
            f"{branch}_umap_x.npy",
            f"{branch}_umap_y.npy",
            f"{branch}_umap_z.npy",
            f"{branch}_spatial_shape.npy",
        ):
            p = os.path.join(results_dir, fn)
            if not os.path.exists(p):
                missing.append(f"missing_file[{branch}]: {p}")
    # Context branch is optional: if absent, we fallback to predict branch.
    ctx_files = [
        os.path.join(results_dir, "context_pca_xyz.npy"),
        os.path.join(results_dir, "context_umap_x.npy"),
        os.path.join(results_dir, "context_umap_y.npy"),
        os.path.join(results_dir, "context_umap_z.npy"),
        os.path.join(results_dir, "context_spatial_shape.npy"),
    ]
    if any(not os.path.exists(p) for p in ctx_files):
        missing.append(
            "optional_context_missing: one or more context_* artifacts missing; "
            "dashboard will fallback to predict_* for context panels"
        )
        for p in ctx_files:
            if not os.path.exists(p):
                missing.append(f"missing_file[context_optional]: {p}")
    return missing


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _has_required_branch_artifacts(results_dir: str, branch: str) -> bool:
    required = (
        f"{branch}_pca_xyz.npy",
        f"{branch}_umap_x.npy",
        f"{branch}_umap_y.npy",
        f"{branch}_umap_z.npy",
        f"{branch}_spatial_shape.npy",
    )
    return all(os.path.exists(os.path.join(results_dir, fn)) for fn in required)


def _has_min_dashboard_artifacts(session_dir: str) -> bool:
    results_dir = os.path.join(session_dir, "results")
    if not os.path.isdir(results_dir):
        return False
    return _has_required_branch_artifacts(results_dir, "predict") and _has_required_branch_artifacts(results_dir, "target")


def _missing_dashboard_artifacts(session_dir: str) -> list[str]:
    missing: list[str] = []
    results_dir = os.path.join(session_dir, "results")
    if not os.path.isdir(results_dir):
        missing.append(f"missing_dir:{results_dir}")
        return missing
    for branch in ("predict", "target"):
        for fn in (
            f"{branch}_pca_xyz.npy",
            f"{branch}_umap_x.npy",
            f"{branch}_umap_y.npy",
            f"{branch}_umap_z.npy",
            f"{branch}_spatial_shape.npy",
        ):
            p = os.path.join(results_dir, fn)
            if not os.path.exists(p):
                missing.append(f"missing_file:{p}")
    return missing


def _extract_hw_map(src: dict, keys: tuple[str, ...], shape: tuple[int, int]) -> np.ndarray | None:
    for k in keys:
        if k not in src:
            continue
        arr = _to_np(src[k]).astype(np.float32)
        if arr.ndim == 4:
            arr = arr[0, 0]
        elif arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 2 and arr.shape == shape:
            return np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    return None


def _rgb_from_xyz(xyz: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    lo = np.percentile(xyz, 1.0, axis=0)
    hi = np.percentile(xyz, 99.0, axis=0)
    den = np.clip(hi - lo, 1e-8, None)
    rgb_flat = np.clip(np.round(np.clip((xyz - lo) / den, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    rgb = rgb_flat.reshape(h, w, 3)
    return rgb, rgb_flat


def compute_dash_data(session_dir: str, overwrite: bool = False) -> str:
    out_npz = os.path.join(session_dir, "dash_data.npz")
    if os.path.exists(out_npz) and not overwrite:
        try:
            existing = np.load(out_npz)
            missing = sorted(DASH_DATA_REQUIRED.difference(existing.files))
            existing.close()
            if not missing:
                return out_npz
            print(f"dash_data_stale_recompute={out_npz} missing={','.join(missing)}")
        except Exception as e:
            print(f"dash_data_stale_recompute={out_npz} reason={type(e).__name__}: {e}")

    inf_path = os.path.join(session_dir, "inference_outputs.pt")
    if not os.path.exists(inf_path):
        raise FileNotFoundError(f"Missing inference outputs: {inf_path}")
    outputs = torch.load(inf_path, map_location="cpu")

    x_clean = outputs.get("x_clean")
    if x_clean is None:
        raise RuntimeError(f"{session_dir}: inference outputs missing x_clean")
    orig = _to_np(x_clean)[0, 0].astype(np.float32)
    h, w = orig.shape

    context = outputs.get("x_context", outputs.get("x_context_raw", x_clean))
    blurred = _to_np(context)[0, 0].astype(np.float32)

    # Always render target locations as center points (not square footprints).
    target_locations = outputs.get("target_locations")
    target_valid = outputs.get("target_valid")
    if target_locations is None:
        raise RuntimeError(f"{session_dir}: missing target_locations")
    target = np.zeros((h, w), dtype=np.float32)
    tloc = _to_np(target_locations)
    tvalid = _to_np(target_valid).astype(bool) if target_valid is not None else np.ones(tloc.shape[:2], dtype=bool)
    for bi in range(min(1, tloc.shape[0])):
        for ki in range(tloc.shape[1]):
            if not tvalid[bi, ki]:
                continue
            yy, xx = int(tloc[bi, ki, 0]), int(tloc[bi, ki, 1])
            if 0 <= yy < h and 0 <= xx < w:
                target[yy, xx] = 1.0

    target_loc_heatmap = _extract_hw_map(
        outputs,
        ("target_loc_heatmap", "target_location_heatmap", "target_heatmap"),
        (h, w),
    )
    if target_loc_heatmap is None:
        target_loc_heatmap = target.copy()

    energy_map = _extract_hw_map(outputs, ("target_energy_map",), (h, w))
    if energy_map is None:
        energy_npy = os.path.join(session_dir, "target_energy_map.npy")
        if os.path.exists(energy_npy):
            energy_map = np.asarray(np.load(energy_npy), dtype=np.float32)
            if energy_map.ndim == 4:
                energy_map = energy_map[0, 0]
            elif energy_map.ndim == 3:
                energy_map = energy_map[0]
            if energy_map.shape != (h, w):
                energy_map = np.zeros((h, w), dtype=np.float32)
        else:
            energy_map = np.zeros((h, w), dtype=np.float32)

    visit_path = os.path.join(session_dir, "visited_target_frequency.npy")
    visit_heatmap = np.asarray(np.load(visit_path), dtype=np.float32) if os.path.exists(visit_path) else np.zeros((h, w), dtype=np.float32)
    if visit_heatmap.shape != (h, w):
        visit_heatmap = np.zeros((h, w), dtype=np.float32)

    # Load precomputed PCA/UMAP artifacts saved by training-time pipeline.
    results_dir = os.path.join(session_dir, "results")
    if not os.path.isdir(results_dir):
        raise RuntimeError(f"{session_dir}: missing required directory {results_dir}")
    if not _has_required_branch_artifacts(results_dir, "predict"):
        raise RuntimeError(f"{session_dir}: missing required artifacts under {results_dir} for branch=predict")
    if not _has_required_branch_artifacts(results_dir, "target"):
        raise RuntimeError(f"{session_dir}: missing required artifacts under {results_dir} for branch=target")
    verbose_missing = _verbose_artifact_report(session_dir)
    for line in verbose_missing:
        print(f"dashboard_artifact_check={line}")

    pred_map = outputs.get("pred_map")
    if pred_map is None:
        raise RuntimeError(f"{session_dir}: inference outputs missing pred_map")
    h_lat, w_lat = int(pred_map.shape[-2]), int(pred_map.shape[-1])

    def _load_xyz_triplet(prefix: str, kind: str) -> np.ndarray:
        if kind == "pca":
            path = os.path.join(results_dir, f"{prefix}_pca_xyz.npy")
            if not os.path.exists(path):
                raise RuntimeError(
                    f"{session_dir}: missing required artifact {path}\n"
                    f"hint: run training/inference export that writes results/{prefix}_pca_xyz.npy"
                )
            xyz = np.asarray(np.load(path), dtype=np.float32)
            if xyz.ndim != 2 or xyz.shape[1] != 3:
                raise RuntimeError(f"{session_dir}: malformed PCA artifact {path} shape={xyz.shape}")
            return xyz
        ux = os.path.join(results_dir, f"{prefix}_umap_x.npy")
        uy = os.path.join(results_dir, f"{prefix}_umap_y.npy")
        uz = os.path.join(results_dir, f"{prefix}_umap_z.npy")
        if not (os.path.exists(ux) and os.path.exists(uy) and os.path.exists(uz)):
            raise RuntimeError(
                f"{session_dir}: missing required UMAP artifacts for {prefix} "
                f"(expected {ux}, {uy}, {uz})\n"
                f"hint: run training/inference export that writes results/{prefix}_umap_[x|y|z].npy"
            )
        x = np.asarray(np.load(ux), dtype=np.float32).reshape(-1)
        y = np.asarray(np.load(uy), dtype=np.float32).reshape(-1)
        z = np.asarray(np.load(uz), dtype=np.float32).reshape(-1)
        n = min(x.size, y.size, z.size)
        return np.stack([x[:n], y[:n], z[:n]], axis=1).astype(np.float32)

    def _load_hw(prefix: str) -> tuple[int, int]:
        shp = os.path.join(results_dir, f"{prefix}_spatial_shape.npy")
        if os.path.exists(shp):
            arr = np.asarray(np.load(shp), dtype=np.int64).reshape(-1)
            if arr.size >= 2 and int(arr[0]) > 0 and int(arr[1]) > 0:
                return int(arr[0]), int(arr[1])
        return h_lat, w_lat

    bundles = {}
    for prefix_saved, prefix_out in (("context", "context"), ("predict", "pred"), ("target", "gt")):
        src_prefix = prefix_saved
        try:
            hh, ww = _load_hw(src_prefix)
            pca = _load_xyz_triplet(src_prefix, "pca")
            um = _load_xyz_triplet(src_prefix, "umap")
        except Exception:
            if prefix_saved == "context":
                src_prefix = "predict"
                hh, ww = _load_hw(src_prefix)
                pca = _load_xyz_triplet(src_prefix, "pca")
                um = _load_xyz_triplet(src_prefix, "umap")
                print(
                    f"dashboard_note={session_dir}: missing context embeddings; "
                    "using predict embeddings for context panels"
                )
            else:
                raise
        if pca.shape[0] != hh * ww or um.shape[0] != hh * ww:
            raise RuntimeError(
                f"{session_dir}: embedding length mismatch for {src_prefix} "
                f"(shape={hh}x{ww}, pca_n={pca.shape[0]}, umap_n={um.shape[0]})"
            )
        pca_rgb, pca_rgb_flat = _rgb_from_xyz(pca, hh, ww)
        um_rgb, um_rgb_flat = _rgb_from_xyz(um, hh, ww)
        bundles[prefix_out] = {
            "pca3d": pca,
            "umap3d": um,
            "pca_rgb": pca_rgb,
            "pca_rgb_flat": pca_rgb_flat,
            "umap_rgb": um_rgb,
            "umap_rgb_flat": um_rgb_flat,
        }

    metrics_path = os.path.join(session_dir, "metrics.csv")
    loss_x, loss_total, loss_jepa = [], [], []
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ep = float(row.get("epoch", "nan"))
                    ba = float(row.get("batch", row.get("step", "nan")))
                    tl = float(row.get("total_loss", row.get("loss_total", row.get("loss", "nan"))))
                    jl = float(row.get("loss_jepa", row.get("jepa_loss", "nan")))
                except Exception:
                    continue
                if np.isfinite(ep) and np.isfinite(ba):
                    loss_x.append(ep + 0.001 * ba)
                    loss_total.append(tl)
                    loss_jepa.append(jl)

    np.savez_compressed(
        out_npz,
        orig=orig,
        blurred=blurred,
        target=target.astype(np.float32),
        target_loc_heatmap=target_loc_heatmap.astype(np.float32),
        energy_map=energy_map.astype(np.float32),
        visit_heatmap=visit_heatmap.astype(np.float32),
        context_pca3d=bundles["context"]["pca3d"],
        context_umap3d=bundles["context"]["umap3d"],
        context_pca_rgb=bundles["context"]["pca_rgb"],
        context_pca_rgb_flat=bundles["context"]["pca_rgb_flat"],
        context_umap_rgb=bundles["context"]["umap_rgb"],
        context_umap_rgb_flat=bundles["context"]["umap_rgb_flat"],
        pred_pca3d=bundles["pred"]["pca3d"],
        pred_umap3d=bundles["pred"]["umap3d"],
        pred_pca_rgb=bundles["pred"]["pca_rgb"],
        pred_pca_rgb_flat=bundles["pred"]["pca_rgb_flat"],
        pred_umap_rgb=bundles["pred"]["umap_rgb"],
        pred_umap_rgb_flat=bundles["pred"]["umap_rgb_flat"],
        gt_pca3d=bundles["gt"]["pca3d"],
        gt_umap3d=bundles["gt"]["umap3d"],
        gt_pca_rgb=bundles["gt"]["pca_rgb"],
        gt_pca_rgb_flat=bundles["gt"]["pca_rgb_flat"],
        gt_umap_rgb=bundles["gt"]["umap_rgb"],
        gt_umap_rgb_flat=bundles["gt"]["umap_rgb_flat"],
        loss_x=np.asarray(loss_x, dtype=np.float32),
        loss_total=np.asarray(loss_total, dtype=np.float32),
        loss_jepa=np.asarray(loss_jepa, dtype=np.float32),
    )
    return out_npz


def plot_dash_html(session_dir: str, overwrite: bool = False) -> str:
    npz_path = os.path.join(session_dir, "dash_data.npz")
    out_html = os.path.join(session_dir, "dashboard.html")
    # Always regenerate plot HTML, even if an existing dashboard file is present.
    # This keeps plots in sync with the latest artifacts without requiring --overwrite.
    if not os.path.exists(npz_path):
        compute_dash_data(session_dir, overwrite=False)
    data = np.load(npz_path)
    missing = sorted(DASH_DATA_REQUIRED.difference(data.files))
    if missing:
        data.close()
        print(f"dash_data_stale_recompute={npz_path} missing={','.join(missing)}")
        compute_dash_data(session_dir, overwrite=True)
        data = np.load(npz_path)

    def heat(title: str, z: np.ndarray, colorscale: str) -> go.Figure:
        vals = np.asarray(z, dtype=np.float32)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            vals = np.zeros_like(vals)
            zmin, zmax = 0.0, 1.0
        else:
            zmin, zmax = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
            if zmax <= zmin + 1e-12:
                zmax = zmin + 1.0
        fig = go.Figure([go.Heatmap(z=vals, colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=False)])
        fig.update_layout(template="plotly_white", title={"text": title, "x": 0.02}, margin=dict(l=8, r=8, t=36, b=8), height=330)
        fig.update_xaxes(showticklabels=False, constrain="domain")
        fig.update_yaxes(showticklabels=False, scaleanchor="x", scaleratio=1, constrain="domain", autorange="reversed")
        return fig

    def img(title: str, rgb: np.ndarray) -> go.Figure:
        fig = go.Figure([go.Image(z=np.asarray(rgb))])
        fig.update_layout(template="plotly_white", title={"text": title, "x": 0.02}, margin=dict(l=8, r=8, t=36, b=8), height=330)
        fig.update_xaxes(showticklabels=False, constrain="domain")
        fig.update_yaxes(showticklabels=False, scaleanchor="x", scaleratio=1, constrain="domain")
        return fig

    def scatter3d(title: str, xyz: np.ndarray, rgb_flat: np.ndarray) -> tuple[go.Figure, int, int]:
        pts = np.asarray(xyz, dtype=np.float32)
        rgb = np.asarray(rgb_flat)
        source_n = int(pts.shape[0]) if pts.ndim == 2 and pts.shape[1] >= 3 else 0
        if source_n == 0:
            x, y, z, colors = [], [], [], None
            rendered_n = 0
        else:
            n = source_n
            if rgb.ndim == 2:
                n = min(n, int(rgb.shape[0]))
            pts = pts[:n]
            if rgb.ndim == 2:
                rgb = rgb[:n]
            # Keep HTML size sane while still showing dense point clouds.
            if n > 65536:
                step = int(np.ceil(n / 65536.0))
                pts = pts[::step]
                if rgb.ndim == 2:
                    rgb = rgb[::step]
            rendered_n = int(pts.shape[0])
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            colors = None
            if rgb.ndim == 2 and rgb.shape[1] >= 3:
                colors = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in rgb]
        fig = go.Figure(
            [
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(size=2, opacity=0.82, color=colors),
                    showlegend=False,
                )
            ]
        )
        fig.update_layout(
            template="plotly_white",
            title={"text": title, "x": 0.02},
            margin=dict(l=8, r=8, t=36, b=8),
            height=430,
            scene=dict(
                xaxis_title="dim-1",
                yaxis_title="dim-2",
                zaxis_title="dim-3",
                aspectmode="data",
            ),
        )
        return fig, source_n, rendered_n

    loss_x = np.asarray(data["loss_x"], dtype=np.float32) if "loss_x" in data.files else np.asarray([], dtype=np.float32)
    loss_total = np.asarray(data["loss_total"], dtype=np.float32) if "loss_total" in data.files else np.asarray([], dtype=np.float32)
    loss_jepa = np.asarray(data["loss_jepa"], dtype=np.float32) if "loss_jepa" in data.files else np.asarray([], dtype=np.float32)
    n = min(loss_x.size, loss_total.size, loss_jepa.size) if (loss_x.size and loss_total.size and loss_jepa.size) else 0
    fig_loss = go.Figure()
    if n > 0:
        fig_loss.add_trace(go.Scattergl(x=loss_x[:n], y=loss_total[:n], mode="lines", name="total_loss"))
        fig_loss.add_trace(go.Scattergl(x=loss_x[:n], y=loss_jepa[:n], mode="lines", name="loss_jepa"))
    fig_loss.update_layout(
        template="plotly_white",
        title={"text": "Loss Curve", "x": 0.02},
        margin=dict(l=42, r=8, t=36, b=36),
        height=330,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    fig_loss.update_xaxes(title_text="epoch+0.001*batch")
    fig_loss.update_yaxes(title_text="loss")

    cards: list[tuple[str, go.Figure]] = [
        ("Input (Log-Norm)", heat("Input (Log-Norm)", data["orig"], "Viridis")),
        ("Loss Curve", fig_loss),
        ("Target Locations", heat("Target Locations", data["target"], "Magma")),
        ("Target Location Heatmap", heat("Target Location Heatmap", data["target_loc_heatmap"], "Magma")),
        ("Energy Map", heat("Energy Map", data["energy_map"], "Inferno")),
        ("Visit Frequency Heatmap", heat("Visit Frequency Heatmap", data["visit_heatmap"], "Cividis")),
    ]
    for name, stem in (("Context", "context"), ("Predict", "pred"), ("Target", "gt")):
        pca_scatter, _, _ = scatter3d(f"{name} PCA 3D Scatter", data[f"{stem}_pca3d"], data[f"{stem}_pca_rgb_flat"])
        umap_scatter, _, _ = scatter3d(f"{name} UMAP 3D Scatter", data[f"{stem}_umap3d"], data[f"{stem}_umap_rgb_flat"])
        cards.append((f"{name} PCA Color", img(f"{name} PCA Color", data[f"{stem}_pca_rgb"])))
        cards.append((f"{name} PCA 3D Scatter", pca_scatter))
        cards.append((f"{name} UMAP Color", img(f"{name} UMAP Color", data[f"{stem}_umap_rgb"])))
        cards.append((f"{name} UMAP 3D Scatter", umap_scatter))

    rendered = []
    for i, (_, fig) in enumerate(cards):
        rendered.append(f'<section class="card">{fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False), config={"responsive": True, "displaylogo": False})}</section>')

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>JEPA Dashboard - {os.path.basename(session_dir)}</title>
  <meta name="jepa-dashboard-version" content="{DASHBOARD_VERSION}" />
  <style>
    body {{ margin: 14px; background: #f4f6fa; color: #0d1527; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    h1 {{ margin: 0 0 12px 2px; font-size: 24px; font-weight: 650; }}
    .version {{ color: #596275; font-size: 13px; font-weight: 500; margin-left: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(420px, 1fr)); gap: 12px; }}
    .card {{ background: #fff; border: 1px solid #d9deea; border-radius: 10px; box-shadow: 0 1px 2px rgba(10,20,40,0.08); padding: 6px; overflow: hidden; }}
    @media (max-width: 1120px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>JEPA Session Dashboard: {os.path.basename(session_dir)} <span class="version">{DASHBOARD_VERSION}</span></h1>
  <div class="grid">{''.join(rendered)}</div>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    # Mandatory diagnostics summary output.
    print(f"dashboard_plot_summary_begin session={session_dir}")
    print(f"dashboard_plot_summary_cards={len(cards)}")
    print(f"dashboard_plot_item=Loss Curve: {'ok' if n > 0 else 'empty'} (total_points={n} jepa_points={n})")
    for name, stem in (("Context", "context"), ("Predict", "pred"), ("Target", "gt")):
        pca_arr = np.asarray(data[f"{stem}_pca3d"], dtype=np.float32)
        um_arr = np.asarray(data[f"{stem}_umap3d"], dtype=np.float32)
        _, pca_source_n, pca_rendered_n = scatter3d(f"{name} PCA 3D Scatter", pca_arr, data[f"{stem}_pca_rgb_flat"])
        _, umap_source_n, umap_rendered_n = scatter3d(f"{name} UMAP 3D Scatter", um_arr, data[f"{stem}_umap_rgb_flat"])
        same_shape = pca_arr.shape == um_arr.shape
        if same_shape and pca_arr.size > 0:
            diff_l2 = float(np.linalg.norm((pca_arr - um_arr).reshape(-1)))
            pca_l2 = float(np.linalg.norm(pca_arr.reshape(-1))) + 1e-12
            rel_diff = diff_l2 / pca_l2
            umap_equals_pca = bool(rel_diff < 1e-6)
        else:
            rel_diff = float("nan")
            umap_equals_pca = False
        print(f"dashboard_plot_item={name} PCA Color: ok")
        print(
            f"dashboard_plot_item={name} PCA 3D Scatter: "
            f"{'ok' if pca_rendered_n > 0 else 'empty'} "
            f"(source_points={pca_source_n} rendered_points={pca_rendered_n})"
        )
        print(f"dashboard_plot_item={name} UMAP Color: ok")
        print(
            f"dashboard_plot_item={name} UMAP 3D Scatter: "
            f"{'ok' if umap_rendered_n > 0 else 'empty'} "
            f"(source_points={umap_source_n} rendered_points={umap_rendered_n})"
        )
        print(
            f"dashboard_plot_item={name} UMAP_vs_PCA: "
            f"{'same' if umap_equals_pca else 'different'} "
            f"(relative_l2_diff={rel_diff:.6g})"
        )
    for title, key in (
        ("Input (Log-Norm)", "orig"),
        ("Target Locations", "target"),
        ("Target Location Heatmap", "target_loc_heatmap"),
        ("Energy Map", "energy_map"),
        ("Visit Frequency Heatmap", "visit_heatmap"),
    ):
        finite = int(np.isfinite(np.asarray(data[key])).sum())
        print(f"dashboard_plot_item={title}: {'ok' if finite > 0 else 'empty'} (finite_pixels={finite})")
    print(f"dashboard_plot_summary_end session={session_dir} out_html={out_html}")

    data.close()
    return out_html


def plot_dash(session_dir: str, overwrite: bool = False) -> str:
    return plot_dash_html(session_dir, overwrite=overwrite)


def _preferred_html_for_export(session_dir: str, fallback_html: str) -> str:
    demo_files = sorted([fn for fn in os.listdir(session_dir) if fn.startswith("masking_demo_") and fn.endswith(".html")])
    if not demo_files:
        return fallback_html
    with open(fallback_html, "r", encoding="utf-8") as f:
        html = f.read()
    parts = ["<hr/>", "<h2 style='font-family:sans-serif;margin:16px 0 8px 0;'>Masking Demo Panels</h2>"]
    for fn in demo_files:
        parts.append(
            f"<div style='margin:10px 0;'><div style='font-family:sans-serif;font-size:14px;margin:4px 0;'>{fn}</div>"
            f"<iframe src=\"{fn}\" style='width:100%;height:980px;border:1px solid #ddd;border-radius:6px;'></iframe></div>"
        )
    html = html.replace("</body>", "\n" + "\n".join(parts) + "\n</body>")
    out_html = os.path.join(session_dir, "dashboard_with_masking_demo.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html


def main():
    parser = argparse.ArgumentParser(description="Build dashboards from existing sessions")
    parser.add_argument("--sessions-dir", type=str, default="sessions")
    parser.add_argument("--export-dir", type=str, default="results/dashboard")
    parser.add_argument("--stage", type=str, choices=["compute", "plot", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.sessions_dir):
        raise FileNotFoundError(f"Sessions dir not found: {args.sessions_dir}")
    export_dir = args.export_dir if os.path.isabs(args.export_dir) else os.path.join(os.getcwd(), args.export_dir)
    os.makedirs(export_dir, exist_ok=True)

    processed = 0
    skipped = 0
    exported = 0
    for name in sorted(os.listdir(args.sessions_dir)):
        session_dir = os.path.join(args.sessions_dir, name)
        if not os.path.isdir(session_dir):
            continue
        print("=" * 72)
        print(f"dashboard_session_begin={session_dir}")
        inf_path = os.path.join(session_dir, "inference_outputs.pt")
        if not os.path.exists(inf_path):
            print(f"skip_no_inference={session_dir}")
            skipped += 1
            continue
        if args.stage in ("compute", "all", "plot"):
            has_dash_npz = os.path.exists(os.path.join(session_dir, "dash_data.npz"))
            if (not has_dash_npz) and (not _has_min_dashboard_artifacts(session_dir)):
                missing = _missing_dashboard_artifacts(session_dir)
                if missing:
                    print(f"skip_no_dashboard_inputs={session_dir} missing={';'.join(missing)}")
                else:
                    print(f"skip_no_dashboard_inputs={session_dir}")
                skipped += 1
                continue
        try:
            if args.reset:
                for p in (os.path.join(session_dir, "dash_data.npz"), os.path.join(session_dir, "dashboard.html")):
                    if os.path.exists(p):
                        os.remove(p)
                exp = os.path.join(export_dir, f"{name.replace('/', '_')}.html")
                if os.path.exists(exp):
                    os.remove(exp)
            if args.stage in ("compute", "all"):
                npz = compute_dash_data(session_dir, overwrite=args.overwrite)
                print(f"dash_data_saved={npz}")
            if args.stage in ("plot", "all"):
                if not os.path.exists(os.path.join(session_dir, "dash_data.npz")):
                    compute_dash_data(session_dir, overwrite=args.overwrite)
                html = plot_dash_html(session_dir, overwrite=args.overwrite)
                print(f"dashboard_html_saved={html}")
                export_path = os.path.join(export_dir, f"{name.replace('/', '_')}.html")
                src = _preferred_html_for_export(session_dir, html)
                shutil.copy2(src, export_path)
                print(f"dashboard_html_exported={export_path}")
                exported += 1
            processed += 1
        except Exception as e:
            print(f"skip_error={session_dir} reason={type(e).__name__}: {e}")
            skipped += 1
        finally:
            print(f"dashboard_session_end={session_dir}")

    print(
        f"dash_summary processed={processed} exported={exported} skipped={skipped} "
        f"sessions_dir={args.sessions_dir}"
    )


if __name__ == "__main__":
    main()
