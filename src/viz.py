from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


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
    init: str = "spectral",
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    init_mode = str(init).lower()
    if init_mode not in ("spectral", "random"):
        init_mode = "spectral"
    try:
        from cuml.manifold import UMAP as CuMLUMAP

        return CuMLUMAP(
            n_components=n_components,
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            metric=str(metric),
            random_state=int(random_state),
            init=init_mode,
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
            init=init_mode,
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
  // Filter NaN sentinels for percentile computation.
  const valid0 = d0.filter(v => !isNaN(v));
  const valid1 = d1.filter(v => !isNaN(v));
  const valid2 = d2.filter(v => !isNaN(v));
  const s0 = valid0.slice().sort((a, b) => a - b);
  const s1 = valid1.slice().sort((a, b) => a - b);
  const s2 = valid2.slice().sort((a, b) => a - b);
  const lo0 = percentile(s0, loPct), hi0 = percentile(s0, hiPct);
  const lo1 = percentile(s1, loPct), hi1 = percentile(s1, hiPct);
  const lo2 = percentile(s2, loPct), hi2 = percentile(s2, hiPct);
  const r = new Uint8Array(n), g = new Uint8Array(n), b = new Uint8Array(n);
  const colors = new Array(n);
  const rgbImage = new Array(H);
  for (let y = 0; y < H; y++) rgbImage[y] = new Array(W);
  for (let i = 0; i < n; i++) {{
    if (isNaN(d0[i]) || isNaN(d1[i]) || isNaN(d2[i])) {{
      r[i] = 0; g[i] = 0; b[i] = 0;
      colors[i] = `rgb(${{r[i]}},${{g[i]}},${{b[i]}})`;
      const yy = Math.floor(i / W);
      const xx = i - yy * W;
      rgbImage[yy][xx] = [r[i], g[i], b[i]];
      continue;
    }}
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


def _build_input_validity_mask(x_clean_raw: torch.Tensor, target_h: int, target_w: int) -> np.ndarray:
    """Build a bool validity mask at latent-map resolution from the raw input.

    A position is True when the corresponding input region contains mostly valid
    (non-zero, non-NaN) pixels.  Uses average-pool downsampling so that isolated
    single-pixel artefacts do not dominate the mask.
    """
    if x_clean_raw.dim() != 4:
        return np.ones((target_h, target_w), dtype=bool)
    inp = x_clean_raw[0:1, 0:1]  # [1, 1, H_in, W_in]
    valid = (inp.abs() > 1e-12) & torch.isfinite(inp)
    valid_f = valid.float()
    h_in, w_in = int(inp.shape[-2]), int(inp.shape[-1])
    k_h = max(1, h_in // target_h)
    k_w = max(1, w_in // target_w)
    stride_h = max(1, h_in // target_h)
    stride_w = max(1, w_in // target_w)
    pooled = F.avg_pool2d(valid_f, kernel_size=(k_h, k_w), stride=(stride_h, stride_w))
    if pooled.shape[-2] != target_h or pooled.shape[-1] != target_w:
        pooled = F.interpolate(pooled, size=(target_h, target_w), mode="bilinear", align_corners=False)
    mask = (pooled[0, 0] > 0.5).cpu().numpy()
    return mask


def _filtered_embedding(
    latents_flat: np.ndarray,
    mask_flat: np.ndarray,
    compute_fn,
) -> np.ndarray:
    """Run *compute_fn* only on valid (mask==True) rows and insert NaN sentinels.

    Returns an array of shape (N, D) where D is determined by *compute_fn*.
    """
    n_total = latents_flat.shape[0]
    if not mask_flat.any():
        # No valid positions → all-NaN placeholder; guess D=3
        return np.full((n_total, 3), np.nan, dtype=np.float32)
    if mask_flat.all():
        return compute_fn(latents_flat)
    latents_valid = latents_flat[mask_flat]
    emb_valid = compute_fn(latents_valid)
    n_emb = emb_valid.shape[1]
    result = np.full((n_total, n_emb), np.nan, dtype=np.float32)
    result[mask_flat] = emb_valid
    return result


def save_inference_dashboard(session_dir: str, outputs: dict, umap_cfg: dict | None = None) -> str:
    umap_cfg = dict(umap_cfg or {})
    umap_n_neighbors = int(umap_cfg.get("n_neighbors", 15))
    umap_min_dist = float(umap_cfg.get("min_dist", 0.05))
    umap_metric = str(umap_cfg.get("metric", "cosine"))
    umap_random_state = int(umap_cfg.get("random_state", 42))
    umap_init = str(umap_cfg.get("init", "spectral")).lower()
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

    # Build input-validity mask at latent-map resolution (shared by all branches).
    h_lat = int(pred_map.shape[-2])
    w_lat = int(pred_map.shape[-1])
    valid_mask_2d = _build_input_validity_mask(x_clean_raw, h_lat, w_lat)  # [H_lat, W_lat] bool
    valid_mask_flat = valid_mask_2d.reshape(-1)  # [H_lat * W_lat]

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
                init=umap_init,
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
            init=umap_init,
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

        # Filter invalid-region latents from PCA/UMAP, keep NaN sentinels.
        pca3 = _filtered_embedding(z, valid_mask_flat, _compute_pca_3d).astype(np.float32)

        if valid_mask_flat.all():
            z_umap = _preprocess_latents_for_umap(
                z,
                l2_normalize=umap_l2_normalize,
                standardize=umap_standardize,
            )
            umap3 = _compute_umap_nd(
                z_umap,
                n_components=3,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric=umap_metric,
                random_state=umap_random_state,
                init=umap_init,
            ).astype(np.float32)
        else:
            z_valid = z[valid_mask_flat]
            if z_valid.shape[0] == 0:
                umap3 = np.full((z.shape[0], 3), np.nan, dtype=np.float32)
            else:
                z_umap_valid = _preprocess_latents_for_umap(
                    z_valid,
                    l2_normalize=umap_l2_normalize,
                    standardize=umap_standardize,
                )
                umap3_valid = _compute_umap_nd(
                    z_umap_valid,
                    n_components=3,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    metric=umap_metric,
                    random_state=umap_random_state,
                    init=umap_init,
                ).astype(np.float32)
                umap3 = np.full((z.shape[0], 3), np.nan, dtype=np.float32)
                umap3[valid_mask_flat] = umap3_valid

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
    pca0_3d = _filtered_embedding(pred0, valid_mask_flat, _compute_pca_3d).astype(np.float32)
    if valid_mask_flat.all():
        pred0_umap = _preprocess_latents_for_umap(
            pred0,
            l2_normalize=umap_l2_normalize,
            standardize=umap_standardize,
        )
        umap0_3d = _compute_umap_nd(
            pred0_umap,
            n_components=3,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=umap_random_state,
            init=umap_init,
        )
    else:
        pred0_valid = pred0[valid_mask_flat]
        if pred0_valid.shape[0] == 0:
            umap0_3d = np.full((pred0.shape[0], 3), np.nan, dtype=np.float32)
        else:
            pred0_umap_valid = _preprocess_latents_for_umap(
                pred0_valid,
                l2_normalize=umap_l2_normalize,
                standardize=umap_standardize,
            )
            umap0_3d_valid = _compute_umap_nd(
                pred0_umap_valid,
                n_components=3,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric=umap_metric,
                random_state=umap_random_state,
                init=umap_init,
            )
            umap0_3d = np.full((pred0.shape[0], 3), np.nan, dtype=np.float32)
            umap0_3d[valid_mask_flat] = umap0_3d_valid
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


