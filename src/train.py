from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.dataset import JEPADataset
from src.inference import run_post_training_inference
from src.models.build_jepa import PyramidGridJEPA


def _fmt_metric(v: float) -> str:
    x = float(v)
    ax = abs(x)
    if ax == 0.0:
        return "0.0000"
    if ax < 1e-3 or ax >= 1e3:
        return f"{x:.3e}"
    return f"{x:.4f}"


def _compute_pca_2d(x: np.ndarray) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA

        return PCA(n_components=2).fit_transform(x)
    except Exception as e:
        print(f"[warning] sklearn PCA(2D) failed: {type(e).__name__}: {e}; falling back to torch.pca_lowrank")
        x_t = torch.from_numpy(x.astype(np.float32))
        x_t = x_t - x_t.mean(dim=0, keepdim=True)
        u, s, _ = torch.pca_lowrank(x_t, q=2)
        return (u[:, :2] * s[:2]).cpu().numpy()


def _compute_pca_3d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    try:
        from sklearn.decomposition import PCA

        return PCA(n_components=3).fit_transform(x)
    except Exception as e:
        print(f"[warning] sklearn PCA(3D) failed: {type(e).__name__}: {e}; falling back to numpy SVD")
        u, s, _ = np.linalg.svd(x.astype(np.float64), full_matrices=False)
        z = (u[:, :3] * s[:3]).astype(np.float32)
        return z


def _preprocess_latents_for_umap(x: np.ndarray, l2_normalize: bool = False, standardize: bool = False) -> np.ndarray:
    z = np.asarray(x, dtype=np.float32)
    if l2_normalize:
        denom = np.linalg.norm(z, axis=1, keepdims=True)
        z = z / np.clip(denom, 1e-12, None)
    if standardize:
        mu = z.mean(axis=0, keepdims=True)
        sd = z.std(axis=0, keepdims=True)
        z = (z - mu) / np.clip(sd, 1e-6, None)
    return z


def _compute_umap_nd(
    x: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.05,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    try:
        from cuml.manifold import UMAP as CuMLUMAP

        return CuMLUMAP(
            n_components=n_components,
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            metric=str(metric),
            random_state=int(random_state),
        ).fit_transform(x)
    except Exception as e:
        print(f"[warning] cuML UMAP failed: {type(e).__name__}: {e}")

    try:
        import torchdr

        if hasattr(torchdr, "UMAP"):
            model = torchdr.UMAP(
                n_components=n_components,
                n_neighbors=int(n_neighbors),
                min_dist=float(min_dist),
            )
            z = model.fit_transform(torch.from_numpy(x.astype(np.float32)))
            if isinstance(z, torch.Tensor):
                return z.cpu().numpy()
            return np.asarray(z)
    except Exception as e:
        print(f"[warning] torchdr UMAP failed: {type(e).__name__}: {e}")

    try:
        import umap

        return umap.UMAP(
            n_components=n_components,
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            metric=str(metric),
            random_state=int(random_state),
        ).fit_transform(x)
    except Exception as e:
        print(f"[warning] umap-learn failed: {type(e).__name__}: {e}")

    if n_components == 2:
        return _compute_pca_2d(x)
    p2 = _compute_pca_2d(x)
    z = np.zeros((p2.shape[0], n_components), dtype=np.float32)
    z[:, :2] = p2.astype(np.float32)
    return z


def _save_latent_overview_html(session_dir: str, pca_points: np.ndarray, umap_points: np.ndarray, h: int, w: int) -> str:
    out_path = os.path.join(session_dir, "latent_overview_4panel.html")
    pca = np.asarray(pca_points, dtype=np.float32)
    umap = np.asarray(umap_points, dtype=np.float32)
    if pca.shape[1] != 3 or umap.shape[1] != 3:
        raise ValueError(f"Expected 3D points for PCA/UMAP, got pca={pca.shape}, umap={umap.shape}")
    n = int(h * w)
    if pca.shape[0] != n or umap.shape[0] != n:
        raise ValueError(f"Point count mismatch with map shape: h*w={n}, pca={pca.shape[0]}, umap={umap.shape[0]}")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Latent Overview: PCA/UMAP Color Maps vs XYZ</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 14px; color: #111; }}
    .topbar {{ display: flex; gap: 14px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }}
    .topbar label {{ font-size: 13px; }}
    #plot {{ width: 100%; height: 920px; }}
  </style>
</head>
<body>
  <h2>Latent Overview: PCA/UMAP Color Maps vs XYZ</h2>
  <div class="topbar">
    <label>Low percentile <input id="pctLow" type="number" min="0" max="99" step="0.1" value="1.0" /></label>
    <label>High percentile <input id="pctHigh" type="number" min="1" max="100" step="0.1" value="99.0" /></label>
    <button id="applyBtn" type="button">Apply Range</button>
    <span id="status"></span>
  </div>
  <div id="plot"></div>
<script>
const H = {int(h)};
const W = {int(w)};
const pca = {json.dumps(pca.tolist())};
const umap = {json.dumps(umap.tolist())};
function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
function percentile(sortedArr, pct) {{
  if (sortedArr.length === 0) return 0.0;
  const p = clamp(pct, 0, 100) / 100.0;
  const idx = (sortedArr.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedArr[lo];
  const t = idx - lo;
  return sortedArr[lo] * (1.0 - t) + sortedArr[hi] * t;
}}
function mapRgb(points, loPct, hiPct) {{
  const n = points.length;
  const d0 = new Array(n);
  const d1 = new Array(n);
  const d2 = new Array(n);
  for (let i = 0; i < n; i++) {{
    d0[i] = points[i][0];
    d1[i] = points[i][1];
    d2[i] = points[i][2];
  }}
  const s0 = d0.slice().sort((a, b) => a - b);
  const s1 = d1.slice().sort((a, b) => a - b);
  const s2 = d2.slice().sort((a, b) => a - b);
  const lo0 = percentile(s0, loPct), hi0 = percentile(s0, hiPct);
  const lo1 = percentile(s1, loPct), hi1 = percentile(s1, hiPct);
  const lo2 = percentile(s2, loPct), hi2 = percentile(s2, hiPct);
  const r = new Uint8Array(n), g = new Uint8Array(n), b = new Uint8Array(n);
  const colors = new Array(n);
  const rgbImage = new Array(H);
  for (let y = 0; y < H; y++) rgbImage[y] = new Array(W);
  for (let i = 0; i < n; i++) {{
    const rr = clamp((d0[i] - lo0) / (Math.max(hi0 - lo0, 1e-8)), 0.0, 1.0);
    const gg = clamp((d1[i] - lo1) / (Math.max(hi1 - lo1, 1e-8)), 0.0, 1.0);
    const bb = clamp((d2[i] - lo2) / (Math.max(hi2 - lo2, 1e-8)), 0.0, 1.0);
    r[i] = Math.round(rr * 255.0);
    g[i] = Math.round(gg * 255.0);
    b[i] = Math.round(bb * 255.0);
    colors[i] = `rgb(${{r[i]}},${{g[i]}},${{b[i]}})`;
    const yy = Math.floor(i / W);
    const xx = i - yy * W;
    rgbImage[yy][xx] = [r[i], g[i], b[i]];
  }}
  return {{ rgbImage, colors }};
}}
function mkScatter(points, colors, sceneName) {{
  return {{
    type: "scatter3d",
    mode: "markers",
    x: points.map(p => p[0]),
    y: points.map(p => p[1]),
    z: points.map(p => p[2]),
    marker: {{ size: 2, opacity: 0.5, color: colors }},
    scene: sceneName,
    showlegend: false,
  }};
}}
function render(loPct, hiPct) {{
  const lo = Number.isFinite(loPct) ? loPct : 1.0;
  const hi = Number.isFinite(hiPct) ? hiPct : 99.0;
  const pcaMapped = mapRgb(pca, lo, hi);
  const umapMapped = mapRgb(umap, lo, hi);
  const traces = [
    {{ type: "image", z: pcaMapped.rgbImage, xaxis: "x", yaxis: "y", hoverinfo: "skip" }},
    mkScatter(pca, pcaMapped.colors, "scene"),
    {{ type: "image", z: umapMapped.rgbImage, xaxis: "x2", yaxis: "y2", hoverinfo: "skip" }},
    mkScatter(umap, umapMapped.colors, "scene2"),
  ];
  const layout = {{
    width: 1400, height: 920, template: "plotly_white",
    margin: {{l: 30, r: 10, t: 70, b: 20}},
    annotations: [
      {{text: "PCA Color Map", x: 0.18, y: 1.03, xref: "paper", yref: "paper", showarrow: false}},
      {{text: "PCA XYZ", x: 0.72, y: 1.03, xref: "paper", yref: "paper", showarrow: false}},
      {{text: "UMAP Color Map", x: 0.18, y: 0.48, xref: "paper", yref: "paper", showarrow: false}},
      {{text: "UMAP XYZ", x: 0.72, y: 0.48, xref: "paper", yref: "paper", showarrow: false}},
    ],
    xaxis: {{domain: [0.0, 0.44], showticklabels: false}},
    yaxis: {{domain: [0.55, 1.0], showticklabels: false, scaleanchor: "x", scaleratio: 1}},
    xaxis2: {{domain: [0.0, 0.44], showticklabels: false}},
    yaxis2: {{domain: [0.0, 0.45], showticklabels: false, scaleanchor: "x2", scaleratio: 1}},
    scene: {{domain: {{x: [0.52, 1.0], y: [0.55, 1.0]}}, aspectmode: "cube", xaxis: {{title: "PC1"}}, yaxis: {{title: "PC2"}}, zaxis: {{title: "PC3"}}}},
    scene2: {{domain: {{x: [0.52, 1.0], y: [0.0, 0.45]}}, aspectmode: "cube", xaxis: {{title: "U1"}}, yaxis: {{title: "U2"}}, zaxis: {{title: "U3"}}}},
  }};
  Plotly.newPlot("plot", traces, layout, {{responsive: true}});
  document.getElementById("status").textContent = `Applied range: low=${{lo.toFixed(1)}} high=${{hi.toFixed(1)}}`;
}}
document.getElementById("applyBtn").addEventListener("click", () => {{
  const lo = parseFloat(document.getElementById("pctLow").value);
  const hi = parseFloat(document.getElementById("pctHigh").value);
  render(lo, hi);
}});
render(1.0, 99.0);
</script>
</body></html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def save_blurred_debug_images(
    project_root: str,
    session_name: str,
    x_clean_raw: torch.Tensor,
    x_context_raw: torch.Tensor,
    max_images: int = 8,
) -> str:
    out_dir = os.path.join(project_root, "results", "debug_blurred_images", session_name)
    os.makedirs(out_dir, exist_ok=True)
    n = min(int(max_images), int(x_context_raw.shape[0]))
    for i in range(n):
        clean = x_clean_raw[i, 0].detach().cpu().numpy().astype(np.float32)
        ctx = x_context_raw[i, 0].detach().cpu().numpy().astype(np.float32)
        delta = clean - ctx
        plt.imsave(os.path.join(out_dir, f"{i:03d}_clean.png"), clean, cmap="gray")
        plt.imsave(os.path.join(out_dir, f"{i:03d}_context_blurred.png"), ctx, cmap="gray")
        plt.imsave(os.path.join(out_dir, f"{i:03d}_clean_minus_context.png"), delta, cmap="coolwarm")
    return out_dir


def save_blurred_reference_images(session_dir: str, x_clean_raw: torch.Tensor, x_context_raw: torch.Tensor) -> str:
    out_dir = os.path.join(session_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    clean = x_clean_raw[0, 0].detach().cpu().numpy().astype(np.float32)
    ctx = x_context_raw[0, 0].detach().cpu().numpy().astype(np.float32)
    delta = clean - ctx
    plt.imsave(os.path.join(out_dir, "reference_000_clean.png"), clean, cmap="gray")
    plt.imsave(os.path.join(out_dir, "reference_000_context_blurred.png"), ctx, cmap="gray")
    plt.imsave(os.path.join(out_dir, "reference_000_clean_minus_context.png"), delta, cmap="coolwarm")
    return out_dir


def save_inference_dashboard(session_dir: str, outputs: dict, umap_cfg: dict | None = None) -> str:
    umap_cfg = dict(umap_cfg or {})
    umap_n_neighbors = int(umap_cfg.get("n_neighbors", 15))
    umap_min_dist = float(umap_cfg.get("min_dist", 0.05))
    umap_metric = str(umap_cfg.get("metric", "cosine"))
    umap_random_state = int(umap_cfg.get("random_state", 42))
    umap_l2_normalize = bool(umap_cfg.get("l2_normalize", False))
    umap_standardize = bool(umap_cfg.get("standardize", False))

    x_clean_raw = outputs.get("x_clean_raw", outputs["x_clean"])
    x_context_raw = outputs.get("x_context_raw", outputs["x_context"])
    x_clean = outputs["x_clean"]
    x_context = outputs["x_context"]
    target_locations = outputs["target_locations"]
    pred_map = outputs["pred_map"]
    gt_map = outputs["gt_map"]
    context_map = outputs.get("context_map")

    orig = x_clean_raw[0, 0].detach().cpu().numpy()
    ctx = x_context_raw[0, 0].detach().cpu().numpy()

    # Render sampled target locations for first sample.
    target_vis = np.zeros_like(orig, dtype=np.float32)
    for i in range(target_locations.shape[1]):
        cy = int(target_locations[0, i, 0].item())
        cx = int(target_locations[0, i, 1].item())
        if 0 <= cy < target_vis.shape[0] and 0 <= cx < target_vis.shape[1]:
            target_vis[cy, cx] = 1.0

    pred_vec = pred_map.detach().cpu().permute(0, 2, 3, 1).reshape(-1, pred_map.shape[1]).numpy()
    gt_vec = gt_map.detach().cpu().permute(0, 2, 3, 1).reshape(-1, gt_map.shape[1]).numpy()
    x = np.concatenate([pred_vec, gt_vec], axis=0)

    pca_cache = os.path.join(session_dir, "pca_embeddings.npy")
    umap_cache_key = hashlib.md5(json.dumps(umap_cfg, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    umap_cache = os.path.join(session_dir, f"umap_embeddings_{umap_cache_key}.npy")
    x_umap = _preprocess_latents_for_umap(
        x,
        l2_normalize=umap_l2_normalize,
        standardize=umap_standardize,
    )
    if os.path.exists(pca_cache):
        try:
            pca_2d = np.load(pca_cache)
        except Exception as e:
            print(f"[warning] failed to load PCA cache {pca_cache}: {type(e).__name__}: {e}; recomputing")
            pca_2d = _compute_pca_2d(x)
            np.save(pca_cache, pca_2d)
    else:
        pca_2d = _compute_pca_2d(x)
        np.save(pca_cache, pca_2d)
    if os.path.exists(umap_cache):
        try:
            umap_3d = np.load(umap_cache)
        except Exception as e:
            print(f"[warning] failed to load UMAP cache {umap_cache}: {type(e).__name__}: {e}; recomputing")
            umap_3d = _compute_umap_nd(
                x_umap,
                n_components=3,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric=umap_metric,
                random_state=umap_random_state,
            )
            np.save(umap_cache, umap_3d)
    else:
        umap_3d = _compute_umap_nd(
            x_umap,
            n_components=3,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=umap_random_state,
        )
        np.save(umap_cache, umap_3d)
    # Session plot compatibility artifacts.
    results_dir = os.path.join(session_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    save_blurred_reference_images(session_dir, x_clean_raw, x_context_raw)
    np.save(os.path.join(results_dir, "latent_vectors_full.npy"), x.astype(np.float32))
    np.save(os.path.join(results_dir, "umap_x.npy"), umap_3d[:, 0].astype(np.float32))
    np.save(os.path.join(results_dir, "umap_y.npy"), umap_3d[:, 1].astype(np.float32))
    np.save(os.path.join(results_dir, "umap_z.npy"), umap_3d[:, 2].astype(np.float32))

    def _save_branch_embeddings(branch_name: str, fmap: torch.Tensor):
        # Use sample-0 dense latent map (H*W tokens) for branch-specific plotly 2D color + 3D scatter.
        h_map = int(fmap.shape[-2])
        w_map = int(fmap.shape[-1])
        z = fmap[0].detach().cpu().permute(1, 2, 0).reshape(-1, fmap.shape[1]).numpy().astype(np.float32)
        z_umap = _preprocess_latents_for_umap(
            z,
            l2_normalize=umap_l2_normalize,
            standardize=umap_standardize,
        )
        pca3 = _compute_pca_3d(z).astype(np.float32)
        umap3 = _compute_umap_nd(
            z_umap,
            n_components=3,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=umap_random_state,
        ).astype(np.float32)
        np.save(os.path.join(results_dir, f"{branch_name}_spatial_shape.npy"), np.asarray([h_map, w_map], dtype=np.int64))
        np.save(os.path.join(results_dir, f"{branch_name}_latent_vectors_full.npy"), z)
        np.save(os.path.join(results_dir, f"{branch_name}_pca_xyz.npy"), pca3)
        np.save(os.path.join(results_dir, f"{branch_name}_pca_x.npy"), pca3[:, 0])
        np.save(os.path.join(results_dir, f"{branch_name}_pca_y.npy"), pca3[:, 1])
        np.save(os.path.join(results_dir, f"{branch_name}_pca_z.npy"), pca3[:, 2])
        np.save(os.path.join(results_dir, f"{branch_name}_umap_x.npy"), umap3[:, 0])
        np.save(os.path.join(results_dir, f"{branch_name}_umap_y.npy"), umap3[:, 1])
        np.save(os.path.join(results_dir, f"{branch_name}_umap_z.npy"), umap3[:, 2])

    _save_branch_embeddings("predict", pred_map)
    _save_branch_embeddings("target", gt_map)
    if context_map is not None:
        _save_branch_embeddings("context", context_map)

    # Spatial latent map overview (sample-0 pred map only): PCA/UMAP colormap + XYZ scatter.
    pred0 = pred_map[0].detach().cpu().permute(1, 2, 0).reshape(-1, pred_map.shape[1]).numpy().astype(np.float32)
    pred0_umap = _preprocess_latents_for_umap(
        pred0,
        l2_normalize=umap_l2_normalize,
        standardize=umap_standardize,
    )
    pca0_3d = _compute_pca_3d(pred0)
    umap0_3d = _compute_umap_nd(
        pred0_umap,
        n_components=3,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=umap_random_state,
    )
    latent_html_path = _save_latent_overview_html(session_dir, pca0_3d, umap0_3d, pred_map.shape[-2], pred_map.shape[-1])

    # Historical target-location heatmap loaded from session CSV log.
    hist_vis = np.zeros_like(orig, dtype=np.float32)
    hist_path = os.path.join(session_dir, "visited_target_locations.csv")
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cy = int(float(row["y"]))
                    cx = int(float(row["x"]))
                    if 0 <= cy < hist_vis.shape[0] and 0 <= cx < hist_vis.shape[1]:
                        hist_vis[cy, cx] += 1.0
        except Exception as e:
            print(f"[warning] failed to read {hist_path}: {type(e).__name__}: {e}")
    if float(hist_vis.max()) > 0.0:
        hist_vis = hist_vis / float(hist_vis.max())
    np.save(os.path.join(results_dir, "target_locations_vis.npy"), target_vis.astype(np.float32))
    np.save(os.path.join(results_dir, "target_locations_hist_vis.npy"), hist_vis.astype(np.float32))
    target_vis_img = os.path.join(results_dir, "target_locations_vis.png")
    hist_vis_img = os.path.join(results_dir, "target_locations_hist_vis.png")
    plt.imsave(target_vis_img, target_vis, cmap="magma")
    plt.imsave(hist_vis_img, hist_vis, cmap="viridis")

    # Build a single assembled dashboard entrypoint for this session.
    dashboard_path = os.path.join(session_dir, "dashboard.html")
    latent_name = os.path.basename(latent_html_path)
    metrics_path = os.path.join(session_dir, "metrics.csv")
    loss_x = []
    loss_total = []
    loss_jepa = []
    loss_pixel = []
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    loss_x.append(float(row["epoch"]) + 0.001 * float(row["batch"]))
                    loss_total.append(float(row["total_loss"]))
                    loss_jepa.append(float(row["loss_jepa"]))
                    loss_pixel.append(float(row["loss_pixel"]))
        except Exception as e:
            print(f"[warning] failed to read {metrics_path}: {type(e).__name__}: {e}")
    ref_clean = os.path.join("results", "reference_000_clean.png")
    ref_ctx = os.path.join("results", "reference_000_context_blurred.png")
    ref_delta = os.path.join("results", "reference_000_clean_minus_context.png")
    target_vis_rel = os.path.join("results", "target_locations_vis.png")
    hist_vis_rel = os.path.join("results", "target_locations_hist_vis.png")
    loss_html = (
        '<div id="loss-plot" style="width: 100%; height: 420px; border: 1px solid #ddd; border-radius: 6px;"></div>'
        if len(loss_x) > 0
        else "<p>Loss data not available yet.</p>"
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>JEPA Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 20px; color: #111; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .section {{ margin: 0 0 24px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(240px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #ddd; padding: 8px; border-radius: 6px; background: #fff; }}
    img {{ width: 100%; height: auto; display: block; }}
    iframe {{ width: 100%; height: 920px; border: 1px solid #ddd; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>JEPA Session Dashboard</h1>
  <div class="section">
    <h2>Loss Curve</h2>
    {loss_html}
  </div>
  <div class="section">
    <h2>Reference Images (Sample 0)</h2>
    <div class="grid">
      <div class="card"><p>Clean</p><img src="{ref_clean}" alt="reference_clean" /></div>
      <div class="card"><p>Blurred Context</p><img src="{ref_ctx}" alt="reference_context" /></div>
      <div class="card"><p>Clean - Context</p><img src="{ref_delta}" alt="reference_delta" /></div>
    </div>
  </div>
  <div class="section">
    <h2>Target Sampling Diagnostics</h2>
    <div class="grid">
      <div class="card"><p>Current Sample Target Locations</p><img src="{target_vis_rel}" alt="target_locations_vis" /></div>
      <div class="card"><p>Historical Target Visit Heatmap</p><img src="{hist_vis_rel}" alt="target_locations_hist_vis" /></div>
    </div>
  </div>
  <div class="section">
    <h2>Latent Overview</h2>
    <iframe src="{latent_name}" title="latent_overview_4panel"></iframe>
  </div>
</body>
</html>
"""
    if len(loss_x) > 0:
        script = f"""
<script>
const lossX = {json.dumps(loss_x)};
const lossTotal = {json.dumps(loss_total)};
const lossJepa = {json.dumps(loss_jepa)};
const lossPixel = {json.dumps(loss_pixel)};
Plotly.newPlot('loss-plot', [
  {{x: lossX, y: lossTotal, mode: 'lines', name: 'total_loss'}},
  {{x: lossX, y: lossJepa, mode: 'lines', name: 'loss_jepa'}},
  {{x: lossX, y: lossPixel, mode: 'lines', name: 'loss_pixel'}}
], {{
  title: 'Training Loss Curve',
  xaxis: {{title: 'epoch + 0.001*batch'}},
  yaxis: {{title: 'loss'}},
  template: 'plotly_white',
  margin: {{l: 60, r: 20, t: 50, b: 55}}
}}, {{responsive: true}});
</script>
"""
        html = html.replace("</body>", script + "\n</body>")
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html)
    return dashboard_path


def save_loss_curve(session_dir: str):
    metrics_path = os.path.join(session_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        return None
    x_ep = []
    total = []
    jepa = []
    pixel = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_ep.append(float(row["epoch"]) + 0.001 * float(row["batch"]))
            total.append(float(row["total_loss"]))
            jepa.append(float(row["loss_jepa"]))
            pixel.append(float(row["loss_pixel"]))
    if len(x_ep) == 0:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(x_ep, total, label="total_loss")
    ax.plot(x_ep, jepa, label="loss_jepa")
    ax.plot(x_ep, pixel, label="loss_pixel")
    ax.set_title("Training Loss Curve")
    ax.set_xlabel("epoch + 0.001*batch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    out_path = os.path.join(session_dir, "loss_curve.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _offdiag(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("offdiag expects square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def extract_valid_pooled_embeddings(outputs: dict, key: str = "pred_patches") -> torch.Tensor:
    patches = outputs[key]  # B,K,C,P,P
    valid = outputs["target_valid"]  # B,K
    _, _, c, _, _ = patches.shape
    pooled = patches.mean(dim=(3, 4))  # B,K,C
    vm = valid.reshape(-1)
    z = pooled.reshape(-1, c)[vm]
    return z


def sketched_sigreg_loss(z: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    """
    Lightweight SIGReg-style isotropic Gaussian regularization.
    Encourages projected embeddings to have mean 0 and variance 1.
    """
    if z.numel() == 0:
        return z.sum() * 0.0
    if z.shape[0] < 2:
        return z.sum() * 0.0

    z = z - z.mean(dim=0, keepdim=True)
    c = z.shape[1]
    sketch_dim = int(max(1, sketch_dim))
    a = torch.randn((c, sketch_dim), device=z.device, dtype=z.dtype)
    a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-6)
    y = z @ a  # N,sketch_dim

    mean_loss = y.mean(dim=0).pow(2).mean()
    var_loss = (y.var(dim=0, unbiased=False) - 1.0).pow(2).mean()
    return mean_loss + var_loss


def compute_sim_var_cov(outputs: dict) -> tuple[float, float, float]:
    pred = outputs["pred_patches"].detach()  # B,K,C,P,P
    gt = outputs["gt_patches"].detach()  # B,K,C,P,P
    valid = outputs["target_valid"].detach()  # B,K

    b, k, c, p, _ = pred.shape
    # Pool spatial dimensions so VICReg acts on patch-level concepts, not adjacent pixels.
    pred_v = pred.mean(dim=(3, 4))  # B,K,C
    gt_v = gt.mean(dim=(3, 4))  # B,K,C
    vm = valid.reshape(-1)
    z1 = pred_v.reshape(-1, c)[vm]
    z2 = gt_v.reshape(-1, c)[vm]
    if z1.numel() == 0 or z2.numel() == 0:
        return 0.0, 0.0, 0.0

    # sim: cosine similarity (higher is better)
    sim = torch.nn.functional.cosine_similarity(z1, z2, dim=1).mean()

    if z1.shape[0] < 2:
        return float(sim.item()), 0.0, 0.0

    # var: VICReg variance regularizer term (lower is better; 0 ideal)
    std_z1 = torch.sqrt(z1.var(dim=0, unbiased=False) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0, unbiased=False) + 1e-4)
    var_term = 0.5 * (torch.relu(1.0 - std_z1).mean() + torch.relu(1.0 - std_z2).mean())

    # cov: VICReg covariance regularizer term (lower is better; 0 ideal)
    z1c = z1 - z1.mean(dim=0, keepdim=True)
    z2c = z2 - z2.mean(dim=0, keepdim=True)
    cov_z1 = (z1c.T @ z1c) / max(1, z1c.shape[0] - 1)
    cov_z2 = (z2c.T @ z2c) / max(1, z2c.shape[0] - 1)
    cov_term = 0.5 * ((_offdiag(cov_z1).pow(2).mean()) + (_offdiag(cov_z2).pow(2).mean()))

    return float(sim.item()), float(var_term.item()), float(cov_term.item())


def compute_sim_var_cov_torch(outputs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = outputs["pred_patches"]  # keep graph
    gt = outputs["gt_patches"]  # keep graph (target branch already no-grad in forward)
    valid = outputs["target_valid"]

    b, k, c, p, _ = pred.shape
    # Pool spatial dimensions so VICReg acts on patch-level concepts, not adjacent pixels.
    pred_v = pred.mean(dim=(3, 4))  # B,K,C
    gt_v = gt.mean(dim=(3, 4))  # B,K,C
    vm = valid.reshape(-1)
    z1 = pred_v.reshape(-1, c)[vm]
    z2 = gt_v.reshape(-1, c)[vm]
    if z1.numel() == 0 or z2.numel() == 0:
        z = pred.sum() * 0.0
        return z, z, z

    sim = torch.nn.functional.cosine_similarity(z1, z2, dim=1).mean()
    if z1.shape[0] < 2:
        z = sim * 0.0
        return sim, z, z

    std_z1 = torch.sqrt(z1.var(dim=0, unbiased=False) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0, unbiased=False) + 1e-4)
    var_term = 0.5 * (torch.relu(1.0 - std_z1).mean() + torch.relu(1.0 - std_z2).mean())

    z1c = z1 - z1.mean(dim=0, keepdim=True)
    z2c = z2 - z2.mean(dim=0, keepdim=True)
    cov_z1 = (z1c.T @ z1c) / max(1, z1c.shape[0] - 1)
    cov_z2 = (z2c.T @ z2c) / max(1, z2c.shape[0] - 1)
    cov_term = 0.5 * ((_offdiag(cov_z1).pow(2).mean()) + (_offdiag(cov_z2).pow(2).mean()))
    return sim, var_term, cov_term


def compute_raw_mse_and_norm_err(outputs: dict) -> tuple[float, float]:
    pred = outputs["pred_patches"].detach()  # B,K,C,P,P
    gt = outputs["gt_patches"].detach()  # B,K,C,P,P
    valid = outputs["target_valid"].detach()  # B,K

    b, k, c, p, _ = pred.shape
    pred_v = pred.permute(0, 1, 3, 4, 2).reshape(b, k, p * p, c)
    gt_v = gt.permute(0, 1, 3, 4, 2).reshape(b, k, p * p, c)
    vm = valid.unsqueeze(-1).unsqueeze(-1).expand(b, k, p * p, 1).reshape(-1)
    z1 = pred_v.reshape(-1, c)[vm]
    z2 = gt_v.reshape(-1, c)[vm]
    if z1.numel() == 0 or z2.numel() == 0:
        return 0.0, 0.0

    raw_mse = torch.mean((z1 - z2) ** 2)
    norm_err = torch.mean(torch.abs(torch.norm(z1, dim=1) - torch.norm(z2, dim=1)))
    return float(raw_mse.item()), float(norm_err.item())


def compute_jepa_energy(outputs: dict, normalize: bool = False) -> float:
    pred = outputs["pred_patches"]
    gt = outputs["gt_patches"].detach()
    valid = outputs["target_valid"]
    if normalize:
        pred = torch.nn.functional.normalize(pred, dim=2)
        gt = torch.nn.functional.normalize(gt, dim=2)
    energy_per_target = (pred - gt).pow(2).mean(dim=(2, 3, 4))
    if bool(valid.any()):
        return float(energy_per_target[valid].mean().item())
    return 0.0


def compute_target_energy_map(outputs: dict, image_size: tuple[int, int]) -> torch.Tensor:
    # Dense full-image energy from latent map reconstruction error.
    # This intentionally does NOT depend on sparse target points.
    pred_map = outputs["pred_map"]
    gt_map = outputs["gt_map"].detach()
    h, w = int(image_size[0]), int(image_size[1])
    # Per-pixel latent MSE across channels -> Bx1xH_latxW_lat
    energy_lat = (pred_map - gt_map).pow(2).mean(dim=1, keepdim=True)
    if energy_lat.shape[-2:] != (h, w):
        energy_lat = F.interpolate(energy_lat, size=(h, w), mode="bilinear", align_corners=False)
    return energy_lat


def compute_effective_rank_from_features(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 1:
        return 0.0
    z = z - z.mean(axis=0, keepdims=True)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    evals = np.linalg.eigvalsh(cov)
    evals = np.clip(evals, 0.0, None)
    s = float(evals.sum())
    if s <= 0.0:
        return 0.0
    p = evals / s
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    h = float(-np.sum(p * np.log(p)))
    return float(np.exp(h))


def compute_error_by_scale(outputs: dict) -> dict[float, float]:
    pred = outputs["pred_patches"].detach()  # B,K,C,P,P
    gt = outputs["gt_patches"].detach()  # B,K,C,P,P
    scales = outputs["target_scales"].detach()  # B,K
    valid = outputs["target_valid"].detach()  # B,K

    # Per-target MSE averaged over C,P,P
    mse_bk = torch.mean((pred - gt) ** 2, dim=(2, 3, 4))  # B,K
    out = defaultdict(list)
    b, k = mse_bk.shape
    for bi in range(b):
        for ki in range(k):
            if not bool(valid[bi, ki].item()):
                continue
            s = round(float(scales[bi, ki].item()), 6)
            out[s].append(float(mse_bk[bi, ki].item()))
    return {float(s): float(np.mean(v)) for s, v in out.items() if len(v) > 0}


@torch.no_grad()
def evaluate_validation(model: PyramidGridJEPA, val_loader: DataLoader, device: torch.device, max_batches: int | None = None) -> dict:
    model.eval()
    n = 0
    loss_sum = 0.0
    sim_sum = 0.0
    scale_mse = defaultdict(list)
    for batch_idx, x_raw in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x_raw = x_raw.to(device, non_blocking=True)
        outputs = model(x_raw)
        loss = model.compute_loss(outputs)
        sim_val, _, _ = compute_sim_var_cov(outputs)
        ebs = compute_error_by_scale(outputs)
        for s, v in ebs.items():
            scale_mse[s].append(float(v))
        loss_sum += float(loss.item())
        sim_sum += float(sim_val)
        n += 1

    if n == 0:
        return {"val_loss": 0.0, "val_sim": 0.0, "val_error_by_scale": {}}
    return {
        "val_loss": loss_sum / n,
        "val_sim": sim_sum / n,
        "val_error_by_scale": {float(s): float(np.mean(v)) for s, v in scale_mse.items()},
    }


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_session_dir(root: str, config_name: str) -> str:
    path = os.path.join(root, config_name)
    os.makedirs(path, exist_ok=True)
    return path


def resolve_pipeline_config(data_cfg: dict, model_cfg: dict) -> tuple[bool, bool, bool]:
    blur_mode = str(model_cfg.get("blur_mode", "gaussian"))
    if blur_mode not in ("gaussian", "cdd"):
        raise ValueError(
            f"Unsupported blur_mode={blur_mode}. "
            "Allowed blur_mode values are 'gaussian' and 'cdd'."
        )
    # Policy:
    # - gaussian mode: dataset runs CDD, no pre-log, model may apply post-log.
    # - cdd mode: dataset skips CDD and pre-log, model performs CDD masking.
    dataset_apply_cdd = (blur_mode == "gaussian")
    dataset_log_transform = False
    model_post_log = bool(model_cfg.get("post_log_transform", data_cfg.get("log_transform", True)))
    return dataset_apply_cdd, dataset_log_transform, model_post_log


def run_training(config: dict, config_name: str, sessions_root: str = "sessions") -> str:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    print(
        f"[{config_name}] backend_discovered device={device.type} "
        f"cuda_available={torch.cuda.is_available()} mps_available={mps_available}"
    )

    train_cfg = config["train"]
    model_cfg = config["model"]
    data_cfg = config["data"]

    session_dir = make_session_dir(sessions_root, config_name)
    os.makedirs(session_dir, exist_ok=True)
    model_ckpt_path = os.path.join(session_dir, "model_last.pt")
    resume_ckpt_path = os.path.join(session_dir, "checkpoint_last.pt")
    resume_from_existing = os.path.exists(model_ckpt_path)

    with open(os.path.join(session_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    blur_mode = str(model_cfg.get("blur_mode", "gaussian"))
    # Naming cleanup (backward compatible):
    # - box scaling: mask_scaling_box (legacy: mask_scale)
    # - gaussian dip scaling: mask_scaling_gaussian (legacy: dip_sigma_mult)
    mask_scaling_box = float(model_cfg.get("mask_scaling_box", model_cfg.get("mask_scale", 1.0)))
    mask_scaling_gaussian = float(model_cfg.get("mask_scaling_gaussian", model_cfg.get("dip_sigma_mult", 1.0)))
    mask_spacing_scaling = float(model_cfg.get("mask_spacing_scaling", model_cfg.get("spacing_scale", 1.5)))
    mask_size = float(model_cfg.get("mask_size", 0.0))
    dataset_apply_cdd, dataset_log_transform, model_post_log = resolve_pipeline_config(data_cfg=data_cfg, model_cfg=model_cfg)

    print(
        f"[{config_name}] resolved_pipeline "
        f"blur_mode={blur_mode} "
        f"dataset_apply_cdd={dataset_apply_cdd} "
        f"dataset_log_transform={dataset_log_transform} "
        f"model_post_log_transform={model_post_log} "
        f"data.log_transform={data_cfg.get('log_transform', True)} "
        f"model.post_log_transform={model_cfg.get('post_log_transform', '<unset>')} "
        f"data.cdd_mode={data_cfg.get('cdd_mode', 'log')} "
        f"model.cdd_mode={model_cfg.get('cdd_mode', 'log')}"
    )

    model = PyramidGridJEPA(
        latent_channels=model_cfg.get("latent_channels", 32),
        predictor_hidden=model_cfg.get("predictor_hidden"),
        patch_size=model_cfg.get("patch_size", 2),
        sigmas=tuple(model_cfg.get("sigmas", [2, 4, 8, 16])),
        cell_sizes=tuple(model_cfg.get("cell_sizes", [16, 32, 64, 128])),
        mask_fraction=model_cfg.get("mask_fraction", 1.0),
        box_sigma_mult=model_cfg.get("box_sigma_mult", 4.0),
        mask_scale=mask_scaling_box,
        min_mask_scale=model_cfg.get("min_mask_scale", 0.0),
        spacing_scale=mask_spacing_scaling,
        mask_size=mask_size,
        full_grid=model_cfg.get("full_grid", True),
        global_shift=model_cfg.get("global_shift", True),
        align_scales=model_cfg.get("align_scales", True),
        constant_mask_box=model_cfg.get("constant_mask_box", True),
        mask_box_size=model_cfg.get("mask_box_size", 16),
        blur_mode=blur_mode,
        cdd_mode=model_cfg.get("cdd_mode", "log"),
        cdd_constrained=model_cfg.get("cdd_constrained", True),
        cdd_sm_mode=model_cfg.get("cdd_sm_mode", "reflect"),
        mask_fill_mode=model_cfg.get("mask_fill_mode", "zero"),
        dip_sigma_mult=mask_scaling_gaussian,
        constant_gaussian_sigma=model_cfg.get("constant_gaussian_sigma", 1.0),
        post_log_transform=model_cfg.get("post_log_transform", model_post_log),
        log_eps=model_cfg.get("log_eps", float(data_cfg.get("log_eps", 1.0))),
        cdd_log_std_floor_mult=model_cfg.get("cdd_log_std_floor_mult", 0.05),
        ema_momentum=model_cfg.get("ema_momentum", train_cfg.get("momentum", 0.996)),
        normalize_loss=model_cfg.get("normalize_loss", True),
        predictor_layernorm=model_cfg.get("predictor_layernorm", False),
        mode=model_cfg.get("mode", "image"),
        encoder_type=model_cfg.get("encoder_type", "fullres"),
        encoder_width=model_cfg.get("encoder_width", model_cfg.get("latent_channels", 32)),
        encoder_depth=model_cfg.get("encoder_depth", 4),
        encoder_kernel_size=model_cfg.get("encoder_kernel_size", 7),
        encoder_norm_type=model_cfg.get("encoder_norm_type"),
        encoder_norm_groups=model_cfg.get("encoder_norm_groups"),
        encoder_norm_eps=model_cfg.get("encoder_norm_eps"),
        mfae_scales=tuple(model_cfg.get("mfae_scales", [1, 2, 4])),
        mfae_features=tuple(model_cfg.get("mfae_features", ["x", "gradmag", "abslap", "local_std"])),
        mfae_normalize_attributes=bool(model_cfg.get("mfae_normalize_attributes", False)),
        mfae_include_mask_tokens=bool(model_cfg.get("mfae_include_mask_tokens", True)),
    ).to(device)
    allow_partial_resume = bool(train_cfg.get("allow_partial_resume", False))
    resume_mismatch_action = str(train_cfg.get("resume_mismatch_action", "skip")).lower()
    if resume_mismatch_action not in ("skip", "error"):
        raise ValueError(
            f"Unsupported resume_mismatch_action={resume_mismatch_action}. "
            "Use 'skip' or 'error'."
        )
    optimizer_mismatch_action = str(train_cfg.get("optimizer_mismatch_action", "continue_fresh_optimizer")).lower()
    if optimizer_mismatch_action not in ("continue_fresh_optimizer", "restart_epoch0"):
        raise ValueError(
            f"Unsupported optimizer_mismatch_action={optimizer_mismatch_action}. "
            "Use 'continue_fresh_optimizer' or 'restart_epoch0'."
        )

    start_epoch = 0
    resume_state = None
    if os.path.exists(resume_ckpt_path):
        resume_state = torch.load(resume_ckpt_path, map_location=device)
        if "model_state_dict" in resume_state:
            missing, unexpected = model.load_state_dict(resume_state["model_state_dict"], strict=False)
            print(f"[{config_name}] resume_model missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
            if missing:
                print(f"[{config_name}] resume_model missing_keys_list={missing}")
            if unexpected:
                print(f"[{config_name}] resume_model unexpected_keys_list={unexpected}")
            if (missing or unexpected) and not allow_partial_resume:
                if resume_mismatch_action == "error":
                    raise RuntimeError(
                        "Checkpoint model-state mismatch detected and allow_partial_resume=False. "
                        "Set train.allow_partial_resume=true to permit partial model resume."
                    )
                print(
                    f"[{config_name}] warning: checkpoint model-state mismatch; "
                    "skipping resume checkpoint and starting fresh model/optimizer/scaler."
                )
                resume_state = None
                start_epoch = 0
                model = PyramidGridJEPA(
                    latent_channels=model_cfg.get("latent_channels", 32),
                    predictor_hidden=model_cfg.get("predictor_hidden"),
                    patch_size=model_cfg.get("patch_size", 2),
                    sigmas=tuple(model_cfg.get("sigmas", [2, 4, 8, 16])),
                    cell_sizes=tuple(model_cfg.get("cell_sizes", [16, 32, 64, 128])),
                    mask_fraction=model_cfg.get("mask_fraction", 1.0),
                    box_sigma_mult=model_cfg.get("box_sigma_mult", 4.0),
                    mask_scale=mask_scaling_box,
                    min_mask_scale=model_cfg.get("min_mask_scale", 0.0),
                    spacing_scale=mask_spacing_scaling,
                    mask_size=mask_size,
                    full_grid=model_cfg.get("full_grid", True),
                    global_shift=model_cfg.get("global_shift", True),
                    align_scales=model_cfg.get("align_scales", True),
                    constant_mask_box=model_cfg.get("constant_mask_box", True),
                    mask_box_size=model_cfg.get("mask_box_size", 16),
                    blur_mode=blur_mode,
                    cdd_mode=model_cfg.get("cdd_mode", "log"),
                    cdd_constrained=model_cfg.get("cdd_constrained", True),
                    cdd_sm_mode=model_cfg.get("cdd_sm_mode", "reflect"),
                    mask_fill_mode=model_cfg.get("mask_fill_mode", "zero"),
                    dip_sigma_mult=mask_scaling_gaussian,
                    constant_gaussian_sigma=model_cfg.get("constant_gaussian_sigma", 1.0),
                    post_log_transform=model_cfg.get("post_log_transform", model_post_log),
                    log_eps=model_cfg.get("log_eps", float(data_cfg.get("log_eps", 1.0))),
                    cdd_log_std_floor_mult=model_cfg.get("cdd_log_std_floor_mult", 0.05),
                    ema_momentum=model_cfg.get("ema_momentum", train_cfg.get("momentum", 0.996)),
                    normalize_loss=model_cfg.get("normalize_loss", True),
                    predictor_layernorm=model_cfg.get("predictor_layernorm", False),
                    mode=model_cfg.get("mode", "image"),
                    encoder_type=model_cfg.get("encoder_type", "fullres"),
                    encoder_width=model_cfg.get("encoder_width", model_cfg.get("latent_channels", 32)),
                    encoder_depth=model_cfg.get("encoder_depth", 4),
                    encoder_kernel_size=model_cfg.get("encoder_kernel_size", 7),
                    encoder_norm_type=model_cfg.get("encoder_norm_type"),
                    encoder_norm_groups=model_cfg.get("encoder_norm_groups"),
                    encoder_norm_eps=model_cfg.get("encoder_norm_eps"),
                    mfae_scales=tuple(model_cfg.get("mfae_scales", [1, 2, 4])),
                    mfae_features=tuple(model_cfg.get("mfae_features", ["x", "gradmag", "abslap", "local_std"])),
                    mfae_normalize_attributes=bool(model_cfg.get("mfae_normalize_attributes", False)),
                    mfae_include_mask_tokens=bool(model_cfg.get("mfae_include_mask_tokens", True)),
                ).to(device)
                print(f"[{config_name}] resume_checkpoint_ignored={resume_ckpt_path}")
        if resume_state is not None:
            start_epoch = int(resume_state.get("epoch", 0))
            print(f"resume_checkpoint={resume_ckpt_path} start_epoch={start_epoch}")
    elif resume_from_existing:
        missing, unexpected = model.load_state_dict(torch.load(model_ckpt_path, map_location=device), strict=False)
        print(f"[{config_name}] resume_model missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        if missing:
            print(f"[{config_name}] resume_model missing_keys_list={missing}")
        if unexpected:
            print(f"[{config_name}] resume_model unexpected_keys_list={unexpected}")
        if (missing or unexpected) and not allow_partial_resume:
            if resume_mismatch_action == "error":
                raise RuntimeError(
                    "Model checkpoint mismatch detected and allow_partial_resume=False. "
                    "Set train.allow_partial_resume=true to permit partial model resume."
                )
            print(
                f"[{config_name}] warning: model checkpoint mismatch; "
                "ignoring model_last and starting fresh model/optimizer/scaler."
            )
            model = PyramidGridJEPA(
                latent_channels=model_cfg.get("latent_channels", 32),
                predictor_hidden=model_cfg.get("predictor_hidden"),
                patch_size=model_cfg.get("patch_size", 2),
                sigmas=tuple(model_cfg.get("sigmas", [2, 4, 8, 16])),
                cell_sizes=tuple(model_cfg.get("cell_sizes", [16, 32, 64, 128])),
                mask_fraction=model_cfg.get("mask_fraction", 1.0),
                box_sigma_mult=model_cfg.get("box_sigma_mult", 4.0),
                mask_scale=mask_scaling_box,
                min_mask_scale=model_cfg.get("min_mask_scale", 0.0),
                spacing_scale=mask_spacing_scaling,
                mask_size=mask_size,
                full_grid=model_cfg.get("full_grid", True),
                global_shift=model_cfg.get("global_shift", True),
                align_scales=model_cfg.get("align_scales", True),
                constant_mask_box=model_cfg.get("constant_mask_box", True),
                mask_box_size=model_cfg.get("mask_box_size", 16),
                blur_mode=blur_mode,
                cdd_mode=model_cfg.get("cdd_mode", "log"),
                cdd_constrained=model_cfg.get("cdd_constrained", True),
                cdd_sm_mode=model_cfg.get("cdd_sm_mode", "reflect"),
                mask_fill_mode=model_cfg.get("mask_fill_mode", "zero"),
                dip_sigma_mult=mask_scaling_gaussian,
                constant_gaussian_sigma=model_cfg.get("constant_gaussian_sigma", 1.0),
                post_log_transform=model_cfg.get("post_log_transform", model_post_log),
                log_eps=model_cfg.get("log_eps", float(data_cfg.get("log_eps", 1.0))),
                cdd_log_std_floor_mult=model_cfg.get("cdd_log_std_floor_mult", 0.05),
                ema_momentum=model_cfg.get("ema_momentum", train_cfg.get("momentum", 0.996)),
                normalize_loss=model_cfg.get("normalize_loss", True),
                predictor_layernorm=model_cfg.get("predictor_layernorm", False),
                mode=model_cfg.get("mode", "image"),
                encoder_type=model_cfg.get("encoder_type", "fullres"),
                encoder_width=model_cfg.get("encoder_width", model_cfg.get("latent_channels", 32)),
                encoder_depth=model_cfg.get("encoder_depth", 4),
                encoder_kernel_size=model_cfg.get("encoder_kernel_size", 7),
                encoder_norm_type=model_cfg.get("encoder_norm_type"),
                encoder_norm_groups=model_cfg.get("encoder_norm_groups"),
                encoder_norm_eps=model_cfg.get("encoder_norm_eps"),
                mfae_scales=tuple(model_cfg.get("mfae_scales", [1, 2, 4])),
                mfae_features=tuple(model_cfg.get("mfae_features", ["x", "gradmag", "abslap", "local_std"])),
                mfae_normalize_attributes=bool(model_cfg.get("mfae_normalize_attributes", False)),
                mfae_include_mask_tokens=bool(model_cfg.get("mfae_include_mask_tokens", True)),
            ).to(device)
            print(f"[{config_name}] resume_model_ignored={model_ckpt_path}")
        else:
            print(f"resume_model={model_ckpt_path}")

    scale_max = float(max(model_cfg.get("sigmas", [2, 4, 8, 16])))
    auto_roll_max = max(1, int(round(scale_max * mask_scaling_box * mask_spacing_scaling)))

    dataset = JEPADataset(
        num_samples=data_cfg.get("num_samples", 2000),
        image_size=data_cfg.get("image_size", 256),
        data_root=data_cfg.get("data_root", "data"),
        npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
        log_transform=dataset_log_transform,
        log_eps=data_cfg.get("log_eps", 1.0),
        cdd_scales=data_cfg.get("cdd_scales", [2, 4, 8, 16]),
        cdd_strength=data_cfg.get("cdd_strength", 1.0),
        cdd_clip=data_cfg.get("cdd_clip", True),
        norm_before_cdd=data_cfg.get("norm_before_cdd", True),
        cdd_mode=data_cfg.get("cdd_mode", "log"),
        cdd_constrained=data_cfg.get("cdd_constrained", True),
        cdd_sm_mode=data_cfg.get("cdd_sm_mode", "reflect"),
        apply_cdd=dataset_apply_cdd,
        cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
        cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
        cube_slice_index=data_cfg.get("cube_slice_index", 0),
        random_roll_max=int(max(0, data_cfg.get("random_roll_max", auto_roll_max))),
        d4_augment=bool(data_cfg.get("d4_augment", False)),
        cache_cdd=bool(data_cfg.get("cache_cdd", True)),
        cdd_cache_dir=data_cfg.get("cdd_cache_dir"),
        cdd_mem_cache_max=int(data_cfg.get("cdd_mem_cache_max", 64)),
        cache_random_slices=bool(data_cfg.get("cache_random_slices", False)),
        precompute_cdd_cache_all_slices=bool(data_cfg.get("precompute_cdd_cache_all_slices", False)),
        cache_cdd_in_ram_all=bool(data_cfg.get("cache_cdd_in_ram_all", False)),
    )
    val_fraction = float(train_cfg.get("val_fraction", 0.1))
    val_fraction = min(max(val_fraction, 0.0), 0.95)
    total_idx = list(dataset.sample_index)
    n_total = len(total_idx)
    n_val_idx = int(round(n_total * val_fraction)) if n_total > 1 else 0
    if val_fraction > 0.0 and n_val_idx == 0 and n_total > 1:
        n_val_idx = 1
    n_train_idx = max(1, n_total - n_val_idx)
    train_idx = total_idx[:n_train_idx]
    val_idx = total_idx[n_train_idx:] if n_val_idx > 0 else []

    train_dataset = dataset
    train_dataset.sample_index = train_idx
    train_dataset.num_samples = int(train_cfg.get("num_samples", data_cfg.get("num_samples", 2000)))

    val_dataset = None
    if len(val_idx) > 0:
        val_dataset = JEPADataset(
            num_samples=max(1, int(train_cfg.get("val_num_samples", max(16, int(0.25 * train_dataset.num_samples))))),
            image_size=data_cfg.get("image_size", 256),
            data_root=data_cfg.get("data_root", "data"),
            npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
            log_transform=dataset_log_transform,
            log_eps=data_cfg.get("log_eps", 1.0),
            cdd_scales=data_cfg.get("cdd_scales", [2, 4, 8, 16]),
            cdd_strength=data_cfg.get("cdd_strength", 1.0),
            cdd_clip=data_cfg.get("cdd_clip", True),
            norm_before_cdd=data_cfg.get("norm_before_cdd", True),
            cdd_mode=data_cfg.get("cdd_mode", "log"),
            cdd_constrained=data_cfg.get("cdd_constrained", True),
            cdd_sm_mode=data_cfg.get("cdd_sm_mode", "reflect"),
            apply_cdd=dataset_apply_cdd,
            cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
            cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
            cube_slice_index=data_cfg.get("cube_slice_index", 0),
            random_roll_max=int(max(0, data_cfg.get("random_roll_max", auto_roll_max))),
            d4_augment=False,
            cache_cdd=bool(data_cfg.get("cache_cdd", True)),
            cdd_cache_dir=data_cfg.get("cdd_cache_dir"),
            cdd_mem_cache_max=int(data_cfg.get("cdd_mem_cache_max", 64)),
            cache_random_slices=bool(data_cfg.get("cache_random_slices", False)),
            precompute_cdd_cache_all_slices=bool(data_cfg.get("precompute_cdd_cache_all_slices", False)),
            cache_cdd_in_ram_all=bool(data_cfg.get("cache_cdd_in_ram_all", False)),
        )
        val_dataset.sample_index = val_idx
    print(
        f"[{config_name}] dataset_split total_index={n_total} train_index={len(train_idx)} "
        f"val_index={len(val_idx)} val_fraction={val_fraction:.3f}"
    )
    print(
        f"[{config_name}] data_jitter random_roll_max={dataset.random_roll_max} "
        f"(symmetric inclusive roll in [-max,+max])"
    )
    requested_workers = int(train_cfg.get("num_workers", 4))
    # macOS/MPS-safe default: avoid multiprocessing worker hangs unless explicitly set.
    if "num_workers" in train_cfg:
        num_workers = requested_workers
    else:
        num_workers = 4 if device.type == "cuda" else 0
    pin_memory = bool(device.type == "cuda")
    persistent_workers = bool(num_workers > 0)
    print(
        f"[{config_name}] dataloader_setup num_workers={num_workers} "
        f"pin_memory={pin_memory} persistent_workers={persistent_workers}"
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 32),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    # Inference must use canonical orientation (no D4 augmentation).
    inference_dataset = JEPADataset(
        num_samples=train_dataset.num_samples,
        image_size=data_cfg.get("image_size", 256),
        data_root=data_cfg.get("data_root", "data"),
        npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
        log_transform=dataset_log_transform,
        log_eps=data_cfg.get("log_eps", 1.0),
        cdd_scales=data_cfg.get("cdd_scales", [2, 4, 8, 16]),
        cdd_strength=data_cfg.get("cdd_strength", 1.0),
        cdd_clip=data_cfg.get("cdd_clip", True),
        norm_before_cdd=data_cfg.get("norm_before_cdd", True),
        cdd_mode=data_cfg.get("cdd_mode", "log"),
        cdd_constrained=data_cfg.get("cdd_constrained", True),
        cdd_sm_mode=data_cfg.get("cdd_sm_mode", "reflect"),
        apply_cdd=dataset_apply_cdd,
        cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
        cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
        cube_slice_index=data_cfg.get("cube_slice_index", 0),
        random_roll_max=int(max(0, data_cfg.get("random_roll_max", auto_roll_max))),
        d4_augment=False,
        cache_cdd=bool(data_cfg.get("cache_cdd", True)),
        cdd_cache_dir=data_cfg.get("cdd_cache_dir"),
        cdd_mem_cache_max=int(data_cfg.get("cdd_mem_cache_max", 64)),
        cache_random_slices=bool(data_cfg.get("cache_random_slices", False)),
        precompute_cdd_cache_all_slices=bool(data_cfg.get("precompute_cdd_cache_all_slices", False)),
        cache_cdd_in_ram_all=bool(data_cfg.get("cache_cdd_in_ram_all", False)),
    )
    inference_dataset.sample_index = list(train_idx)
    inference_dataset.num_samples = train_dataset.num_samples
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    if resume_state is not None:
        optimizer_state_loaded = False
        if "optimizer_state_dict" in resume_state:
            try:
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
                optimizer_state_loaded = True
            except ValueError as e:
                # Model parameterization changed (e.g., architecture update): choose explicit behavior.
                if optimizer_mismatch_action == "restart_epoch0":
                    print(f"[{config_name}] warning: optimizer_state_incompatible, restarting epoch counter at 0: {e}")
                    start_epoch = 0
                else:
                    print(
                        f"[{config_name}] warning: optimizer_state_incompatible, "
                        f"continuing from epoch {start_epoch} with fresh optimizer: {e}"
                    )
        if optimizer_state_loaded and "scaler_state_dict" in resume_state and torch.cuda.is_available():
            try:
                scaler.load_state_dict(resume_state["scaler_state_dict"])
            except Exception as e:
                print(f"[{config_name}] warning: scaler_state_incompatible, starting scaler fresh: {e}")

    epochs = train_cfg.get("epochs", 20)
    log_interval = train_cfg.get("log_interval", 10)
    force_recompute_inference = bool(train_cfg.get("force_recompute_inference", False))
    inference_mask_passes = int(train_cfg.get("inference_mask_passes", 1))
    viz_crop_border = bool(train_cfg.get("viz_crop_border", False))
    viz_crop_border_px = train_cfg.get("viz_crop_border_px")
    umap_cfg = dict(train_cfg.get("umap", {}))
    compute_effective_rank = bool(train_cfg.get("compute_effective_rank", False))
    print(f"[{config_name}] umap_config={json.dumps(umap_cfg, sort_keys=True)}")
    jepa_loss_weight = float(train_cfg.get("jepa_loss_weight", 100.0))
    vicreg_var_weight = float(train_cfg.get("vicreg_var_weight", 1.0))
    vicreg_cov_weight = float(train_cfg.get("vicreg_cov_weight", 0.1))
    sigreg_weight = float(train_cfg.get("sigreg_weight", 0.0))
    sigreg_sketch_dim = int(train_cfg.get("sigreg_sketch_dim", 64))

    metrics_path = os.path.join(session_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "batch",
                    "total_loss",
                    "loss_jepa",
                    "loss_pixel",
                    "sim",
                    "var",
                    "cov",
                    "raw_mse",
                    "norm_err",
                    "valid_frac",
                    "time_sec",
                ]
            )
    masked_scales_log_path = os.path.join(session_dir, "masked_scales_log.csv")
    if not os.path.exists(masked_scales_log_path):
        with open(masked_scales_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "batch", "scale", "count"])
    epoch_summary_path = os.path.join(session_dir, "epoch_summary.csv")
    if not os.path.exists(epoch_summary_path):
        with open(epoch_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_sim", "val_error_by_scale_json"])
    visited_targets_log_path = os.path.join(session_dir, "visited_target_locations.csv")
    if not os.path.exists(visited_targets_log_path):
        with open(visited_targets_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "batch", "sample_idx", "target_idx", "y", "x", "scale"])

    model.train()
    start = time.time()
    visit_counts = None
    if start_epoch >= int(epochs):
        print(f"[{config_name}] checkpoint epoch {start_epoch} already >= configured epochs {epochs}, skipping training loop")
    for epoch in range(start_epoch, epochs):
        epoch_total = 0.0
        epoch_jepa = 0.0
        epoch_pixel = 0.0
        epoch_sim = 0.0
        epoch_var = 0.0
        epoch_cov = 0.0
        epoch_sigreg = 0.0
        epoch_valid_frac = 0.0
        epoch_batches = 0
        metrics_rows = []
        masked_scale_rows = []
        visited_rows = []
        for batch_idx, x_raw in enumerate(dataloader):
            x_raw = x_raw.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(x_raw)
                loss_jepa = model.compute_loss(outputs)
                _, var_term_t, cov_term_t = compute_sim_var_cov_torch(outputs)
                z_pred = extract_valid_pooled_embeddings(outputs, key="pred_patches")
                loss_sigreg = sketched_sigreg_loss(z_pred, sketch_dim=sigreg_sketch_dim)
                total_loss = (
                    (jepa_loss_weight * loss_jepa)
                    + (vicreg_var_weight * var_term_t)
                    + (vicreg_cov_weight * cov_term_t)
                    + (sigreg_weight * loss_sigreg)
                )
                loss_pixel_val = 0.0

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            model.update_target_encoder()
            sim_val, var_val, cov_val = compute_sim_var_cov(outputs)
            raw_mse_val, norm_err_val = compute_raw_mse_and_norm_err(outputs)
            valid_frac = float(outputs["target_valid"].float().mean().item())

            elapsed = time.time() - start
            metrics_rows.append(
                [
                    epoch + 1,
                    batch_idx,
                    float(total_loss.item()),
                    float(loss_jepa.item()),
                    float(loss_pixel_val),
                    float(sim_val),
                    float(var_val),
                    float(cov_val),
                    float(raw_mse_val),
                    float(norm_err_val),
                    float(valid_frac),
                    round(elapsed, 4),
                ]
            )
            # Save masked-scale usage as training log in session dir.
            scales = outputs["target_scales"].detach().cpu().numpy()
            valid = outputs["target_valid"].detach().cpu().numpy().astype(bool)
            valid_scales = scales[valid]
            if "cdd_channels_masked" in outputs:
                cube_path = os.path.join(session_dir, "example_masked_channel_cube.npy")
                if not os.path.exists(cube_path):
                    np.save(
                        cube_path,
                        outputs["cdd_channels_masked"][0].detach().cpu().numpy().astype(np.float32),
                    )
            if valid_scales.size > 0:
                uniq, cnt = np.unique(np.round(valid_scales.astype(np.float32), 6), return_counts=True)
                for s, c in zip(uniq.tolist(), cnt.tolist()):
                    masked_scale_rows.append([epoch + 1, batch_idx, float(s), int(c)])
            # Save visited target locations for full-session diagnostics.
            tloc = outputs["target_locations"].detach().cpu().numpy()
            tvalid = outputs["target_valid"].detach().cpu().numpy().astype(bool)
            tscale = outputs["target_scales"].detach().cpu().numpy()
            if visit_counts is None:
                hh, ww = int(outputs["x_clean"].shape[-2]), int(outputs["x_clean"].shape[-1])
                visit_counts = np.zeros((hh, ww), dtype=np.float32)
            for bi in range(tloc.shape[0]):
                for ki in range(tloc.shape[1]):
                    if not bool(tvalid[bi, ki]):
                        continue
                    yy = int(tloc[bi, ki, 0])
                    xx = int(tloc[bi, ki, 1])
                    if 0 <= yy < visit_counts.shape[0] and 0 <= xx < visit_counts.shape[1]:
                        visit_counts[yy, xx] += 1.0
            bsz = tloc.shape[0]
            ksz = tloc.shape[1]
            for bi in range(bsz):
                for ki in range(ksz):
                    if not bool(tvalid[bi, ki]):
                        continue
                    visited_rows.append(
                        [
                            epoch + 1,
                            batch_idx,
                            bi,
                            ki,
                            int(tloc[bi, ki, 0]),
                            int(tloc[bi, ki, 1]),
                            float(tscale[bi, ki]),
                        ]
                    )
            if batch_idx % log_interval == 0:
                print(
                    f"[{config_name}] Epoch {epoch + 1}/{epochs} Batch {batch_idx}/{len(dataloader)} "
                    f"total={_fmt_metric(total_loss.item())} jepa={_fmt_metric(loss_jepa.item())} pixel={_fmt_metric(loss_pixel_val)} "
                    f"sigreg={_fmt_metric(loss_sigreg.item())} "
                    f"sim={_fmt_metric(sim_val)} var={_fmt_metric(var_val)} cov={_fmt_metric(cov_val)} "
                    f"raw_mse={_fmt_metric(raw_mse_val)} norm_err={_fmt_metric(norm_err_val)} "
                    f"valid_frac={_fmt_metric(valid_frac)}"
                )
            epoch_total += float(total_loss.item())
            epoch_jepa += float(loss_jepa.item())
            epoch_pixel += float(loss_pixel_val)
            epoch_sim += float(sim_val)
            epoch_var += float(var_val)
            epoch_cov += float(cov_val)
            epoch_sigreg += float(loss_sigreg.item())
            epoch_valid_frac += float(valid_frac)
            epoch_batches += 1

        if metrics_rows:
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(metrics_rows)
        if masked_scale_rows:
            with open(masked_scales_log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(masked_scale_rows)
        if visited_rows:
            with open(visited_targets_log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(visited_rows)
        if visit_counts is not None:
            np.save(os.path.join(session_dir, "visited_target_frequency.npy"), visit_counts.astype(np.float32))

        if epoch_batches > 0:
            print(
                f"[{config_name}] Epoch {epoch + 1}/{epochs} summary "
                f"avg_total={_fmt_metric(epoch_total/epoch_batches)} "
                f"avg_jepa={_fmt_metric(epoch_jepa/epoch_batches)} "
                f"avg_pixel={_fmt_metric(epoch_pixel/epoch_batches)} "
                f"avg_sigreg={_fmt_metric(epoch_sigreg/epoch_batches)} "
                f"avg_sim={_fmt_metric(epoch_sim/epoch_batches)} "
                f"avg_var={_fmt_metric(epoch_var/epoch_batches)} "
                f"avg_cov={_fmt_metric(epoch_cov/epoch_batches)} "
                f"avg_valid_frac={_fmt_metric(epoch_valid_frac/epoch_batches)}"
            )
        val_loss = 0.0
        val_sim = 0.0
        val_error_by_scale = {}
        if val_loader is not None:
            v = evaluate_validation(
                model=model,
                val_loader=val_loader,
                device=device,
                max_batches=train_cfg.get("val_max_batches"),
            )
            val_loss = float(v["val_loss"])
            val_sim = float(v["val_sim"])
            val_error_by_scale = dict(v["val_error_by_scale"])
            print(
                f"[{config_name}] Epoch {epoch + 1}/{epochs} validation "
                f"val_loss={_fmt_metric(val_loss)} val_sim={_fmt_metric(val_sim)} "
                f"val_error_by_scale={json.dumps(val_error_by_scale, sort_keys=True)}"
            )
        with open(epoch_summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    round(epoch_total / max(1, epoch_batches), 8),
                    round(val_loss, 8),
                    round(val_sim, 8),
                    json.dumps(val_error_by_scale, sort_keys=True),
                ]
            )
        model.train()
        # Save resumable checkpoint at the end of every epoch.
        torch.save(
            {
                "epoch": int(epoch + 1),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config_name": config_name,
            },
            resume_ckpt_path,
        )
        # Keep model_last in sync for inference-only resume paths.
        torch.save(model.state_dict(), model_ckpt_path)
        print(f"[{config_name}] checkpoint_saved={resume_ckpt_path} epoch={epoch + 1}")

    torch.save(model.state_dict(), os.path.join(session_dir, "model_last.pt"))

    session_dir = run_post_training_inference(
        model=model,
        dataloader=inference_loader,
        session_dir=session_dir,
        config_name=config_name,
        visit_counts=visit_counts,
        force_recompute_inference=force_recompute_inference,
        inference_mask_passes=inference_mask_passes,
        viz_crop_border=viz_crop_border,
        viz_crop_border_px=viz_crop_border_px,
        compute_jepa_energy_fn=compute_jepa_energy,
        compute_target_energy_map_fn=compute_target_energy_map,
    )
    # Keep dashboard artifacts in sync with inference outputs for all runs.
    # This writes session/results/* embedding files required by session_to_dash.py.
    inf_path = os.path.join(session_dir, "inference_outputs.pt")
    if os.path.exists(inf_path):
        try:
            outputs = torch.load(inf_path, map_location="cpu")
            dash_path = save_inference_dashboard(session_dir, outputs, umap_cfg=umap_cfg)
            print(f"[{config_name}] dashboard_saved={dash_path}")
            effective_rank = ""
            if compute_effective_rank:
                try:
                    pred_map = outputs.get("pred_map")
                    if pred_map is not None:
                        pm = torch.as_tensor(pred_map)
                        z = pm[0].detach().cpu().permute(1, 2, 0).reshape(-1, int(pm.shape[1])).numpy()
                        effective_rank = f"{compute_effective_rank_from_features(z):.8f}"
                except Exception as er:
                    print(f"[{config_name}] warning: effective_rank_failed: {type(er).__name__}: {er}")
            # Dedicated artifact for simple downstream collection.
            # Empty string means rank was not computed for this run.
            with open(os.path.join(session_dir, "effective_rank.txt"), "w", encoding="utf-8") as f:
                f.write(f"{effective_rank}\n")
            with open(os.path.join(session_dir, "effective_rank.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": int(time.time()),
                        "config_name": config_name,
                        "compute_effective_rank": bool(compute_effective_rank),
                        "effective_rank": (None if effective_rank == "" else float(effective_rank)),
                    },
                    f,
                    indent=2,
                )
            run_results_path = os.path.join(session_dir, "run_results.csv")
            if not os.path.exists(run_results_path):
                with open(run_results_path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(["timestamp", "config_name", "compute_effective_rank", "effective_rank"])
            with open(run_results_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([int(time.time()), config_name, int(compute_effective_rank), effective_rank])
        except Exception as e:
            print(f"[{config_name}] warning: dashboard generation failed: {type(e).__name__}: {e}")
    else:
        print(f"[{config_name}] warning: inference_outputs.pt missing; skip dashboard generation")
    return session_dir
