from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _generate_masking_diagnostic_for_dashboard(session_dir: str) -> str | None:
    cfg_path = os.path.join(session_dir, "config_used.json")
    if not os.path.exists(cfg_path):
        return None
    script_path = os.path.join(ROOT_DIR, "scripts", "generate_masking_diagnostics_plotly.py")
    if not os.path.exists(script_path):
        return None
    out_html = os.path.join(session_dir, "masking_demo_plotly_dashboard.html")
    cmd = [
        sys.executable,
        script_path,
        "--config",
        cfg_path,
        "--sample-index",
        "0",
        "--crop",
        "16",
        "--binarize-mask",
        "--cols",
        "3",
        "--panel-px",
        "220",
        "--out",
        out_html,
    ]
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-cache")
    try:
        subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=True, text=True, capture_output=True)
    except Exception as e:
        print(f"[warning] masking diagnostic dashboard generation failed: {type(e).__name__}: {e}")
        return None
    return os.path.basename(out_html)


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

    if n_components == 3:
        return _compute_pca_3d(x)
    return _compute_pca_2d(x)


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
function _pctRange(points, loPct, hiPct) {{
  // Pre-compute sorted channel arrays and return lo/hi per channel.
  const n = points.length;
  const d0 = new Array(n), d1 = new Array(n), d2 = new Array(n);
  for (let i = 0; i < n; i++) {{
    d0[i] = points[i][0]; d1[i] = points[i][1]; d2[i] = points[i][2];
  }}
  const valid0 = d0.filter(v => !isNaN(v));
  const valid1 = d1.filter(v => !isNaN(v));
  const valid2 = d2.filter(v => !isNaN(v));
  const s0 = valid0.slice().sort((a, b) => a - b);
  const s1 = valid1.slice().sort((a, b) => a - b);
  const s2 = valid2.slice().sort((a, b) => a - b);
  return {{
    lo0: percentile(s0, loPct), hi0: percentile(s0, hiPct),
    lo1: percentile(s1, loPct), hi1: percentile(s1, hiPct),
    lo2: percentile(s2, loPct), hi2: percentile(s2, hiPct),
  }};
}}
function mapRgbImage(points, loPct, hiPct) {{
  const n = points.length;
  const rng = _pctRange(points, loPct, hiPct);
  const rgbImage = new Array(H);
  for (let y = 0; y < H; y++) rgbImage[y] = new Array(W);
  for (let i = 0; i < n; i++) {{
    const yy = Math.floor(i / W), xx = i - yy * W;
    const v0 = points[i][0], v1 = points[i][1], v2 = points[i][2];
    if (isNaN(v0) || isNaN(v1) || isNaN(v2)) {{
      rgbImage[yy][xx] = [0, 0, 0]; continue;
    }}
    rgbImage[yy][xx] = [
      Math.round(clamp((v0 - rng.lo0) / Math.max(rng.hi0 - rng.lo0, 1e-8), 0, 1) * 255),
      Math.round(clamp((v1 - rng.lo1) / Math.max(rng.hi1 - rng.lo1, 1e-8), 0, 1) * 255),
      Math.round(clamp((v2 - rng.lo2) / Math.max(rng.hi2 - rng.lo2, 1e-8), 0, 1) * 255),
    ];
  }}
  rgbImage.reverse();
  return rgbImage;
}}
function mapColors(points, loPct, hiPct) {{
  const n = points.length;
  const rng = _pctRange(points, loPct, hiPct);
  const colors = new Array(n);
  for (let i = 0; i < n; i++) {{
    const v0 = points[i][0], v1 = points[i][1], v2 = points[i][2];
    if (isNaN(v0) || isNaN(v1) || isNaN(v2)) {{ colors[i] = "rgb(0,0,0)"; continue; }}
    const rr = clamp((v0 - rng.lo0) / Math.max(rng.hi0 - rng.lo0, 1e-8), 0, 1);
    const gg = clamp((v1 - rng.lo1) / Math.max(rng.hi1 - rng.lo1, 1e-8), 0, 1);
    const bb = clamp((v2 - rng.lo2) / Math.max(rng.hi2 - rng.lo2, 1e-8), 0, 1);
    colors[i] = `rgb(${{Math.round(rr*255)}},${{Math.round(gg*255)}},${{Math.round(bb*255)}})`;
  }}
  return colors;
}}
// RGB images are fixed at full range — slider only affects scatter markers.
const _pcaRgbImage = mapRgbImage(pca, 0.0, 100.0);
const _umapRgbImage = mapRgbImage(umap, 0.0, 100.0);
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
let _plotReady = false;
function render(loPct, hiPct) {{
  const lo = Number.isFinite(loPct) ? loPct : 1.0;
  const hi = Number.isFinite(hiPct) ? hiPct : 99.0;
  const pcaColors = mapColors(pca, lo, hi);
  const umapColors = mapColors(umap, lo, hi);
  if (!_plotReady) {{
    const traces = [
      {{ type: "image", z: _pcaRgbImage, xaxis: "x", yaxis: "y", hoverinfo: "skip" }},
      mkScatter(pca, pcaColors, "scene"),
      {{ type: "image", z: _umapRgbImage, xaxis: "x2", yaxis: "y2", hoverinfo: "skip" }},
      mkScatter(umap, umapColors, "scene2"),
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
    _plotReady = true;
  }} else {{
    // Update only scatter marker colors — RGB images are untouched.
    Plotly.restyle("plot", {{ "marker.color": [pcaColors] }}, [1]);
    Plotly.restyle("plot", {{ "marker.color": [umapColors] }}, [3]);
  }}
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
    # Prefer the real 2-channel ConvNeXt input (e.g. convnext_dense_masktoken).
    # Fallback to shared transformed x_context, not x_context_raw from
    # prepare_context_batch, to avoid a misleading blurred/drop-shadow look.
    x_context_display = outputs.get("network_context_in", x_context)
    ctx = x_context_display[0, 0].detach().cpu().numpy()

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

    return results_dir


def save_volumetric_umap_embeddings(session_dir: str, outputs: dict, umap_cfg: dict | None = None) -> str:
    """Train UMAP on a random fraction of volumetric latent voxels and save artifacts."""
    umap_cfg = dict(umap_cfg or {})
    umap_n_neighbors = int(umap_cfg.get("n_neighbors", 15))
    umap_min_dist = float(umap_cfg.get("min_dist", 0.05))
    umap_metric = str(umap_cfg.get("metric", "cosine"))
    umap_random_state = int(umap_cfg.get("random_state", 42))
    umap_init = str(umap_cfg.get("init", "spectral")).lower()
    umap_l2_normalize = bool(umap_cfg.get("l2_normalize", False))
    umap_standardize = bool(umap_cfg.get("standardize", False))
    sample_fraction = float(umap_cfg.get("volumetric_sample_fraction", 0.05))
    sample_fraction = min(max(sample_fraction, 0.0), 1.0)
    max_points = int(max(128, umap_cfg.get("volumetric_max_points", 50000)))
    sample_seed = int(umap_cfg.get("volumetric_sample_seed", umap_random_state))

    fmap = outputs.get("context_map")
    if fmap is None:
        fmap = outputs["pred_map"]
    if fmap.dim() != 5:
        raise ValueError(f"Expected 3D map B,C,D,H,W, got shape={tuple(fmap.shape)}")

    z = fmap[0].detach().cpu().permute(1, 2, 3, 0).reshape(-1, fmap.shape[1]).numpy().astype(np.float32)
    n_total = int(z.shape[0])
    rng = np.random.default_rng(sample_seed)
    n_pick = int(round(n_total * sample_fraction)) if sample_fraction < 1.0 else n_total
    n_pick = max(128, min(n_total, n_pick, max_points))
    idx = rng.choice(n_total, size=n_pick, replace=False)
    z_sub = z[idx]

    z_sub_umap = _preprocess_latents_for_umap(
        z_sub,
        l2_normalize=umap_l2_normalize,
        standardize=umap_standardize,
    )
    umap3 = _compute_umap_nd(
        z_sub_umap,
        n_components=3,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=umap_random_state,
        init=umap_init,
    ).astype(np.float32)
    pca3 = _compute_pca_3d(z_sub).astype(np.float32)

    results_dir = os.path.join(session_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "volumetric_umap_indices.npy"), idx.astype(np.int64))
    np.save(os.path.join(results_dir, "volumetric_umap_latents.npy"), z_sub.astype(np.float32))
    np.save(os.path.join(results_dir, "volumetric_umap_xyz.npy"), umap3)
    np.save(os.path.join(results_dir, "volumetric_pca_xyz.npy"), pca3)

    meta = {
        "n_total_voxels": n_total,
        "n_sampled": int(n_pick),
        "sample_fraction": float(sample_fraction),
        "max_points": int(max_points),
        "sample_seed": int(sample_seed),
    }
    meta_path = os.path.join(session_dir, "volumetric_umap_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path
