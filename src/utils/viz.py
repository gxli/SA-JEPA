from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_PCA_FIT_TOKENS = 65536


def _deep_merge_dict(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _read_config_file(path: str, seen: set[str] | None = None) -> dict:
    seen = set() if seen is None else seen
    abs_path = os.path.abspath(path)
    if abs_path in seen:
        raise ValueError(f"Cyclic base_config reference detected: {abs_path}")
    seen.add(abs_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        if abs_path.endswith((".yaml", ".yml")):
            import yaml as _yaml

            cfg = _yaml.safe_load(f) or {}
        else:
            cfg = json.load(f)
    if not isinstance(cfg, dict):
        seen.remove(abs_path)
        return {}
    base_ref = cfg.pop("base_config", None)
    if base_ref is not None:
        base_path = base_ref if os.path.isabs(base_ref) else os.path.join(os.path.dirname(abs_path), base_ref)
        cfg = _deep_merge_dict(_read_config_file(base_path, seen), cfg)
    seen.remove(abs_path)
    return cfg


def _session_config_candidates(session_dir: str) -> list[str]:
    name = os.path.basename(os.path.abspath(session_dir))
    env_cfg_dir = os.environ.get("SESSION_DASH_CONFIG_DIR", "").strip()
    candidates: list[str] = []
    if env_cfg_dir:
        for ext in (".yaml", ".yml", ".json"):
            candidates.append(os.path.join(env_cfg_dir, "local_configs", f"{name}{ext}"))
            candidates.append(os.path.join(env_cfg_dir, f"{name}{ext}"))
    candidates.extend([
        os.path.join(ROOT_DIR, "configs", "local_configs", f"{name}.yaml"),
        os.path.join(ROOT_DIR, "configs", "local_configs", f"{name}.yml"),
        os.path.join(ROOT_DIR, "configs", "local_configs", f"{name}.json"),
        os.path.join(session_dir, "resolved_config.json"),
        os.path.join(session_dir, "config_used.json"),
    ])
    return [os.path.abspath(path) for path in candidates]


def _load_session_config(session_dir: str) -> dict:
    for path in _session_config_candidates(session_dir):
        if not os.path.exists(path):
            continue
        try:
            return _read_config_file(path)
        except Exception:
            continue
    return {}


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
    env.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl-cache"))
    try:
        subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=True, text=True, capture_output=True)
    except Exception as e:
        print(f"[warning] masking diagnostic dashboard generation failed: {type(e).__name__}: {e}")
        return None
    return os.path.basename(out_html)


def _encoder_fov_border_from_config(session_dir: str) -> int:
    cfg = _load_session_config(session_dir)
    if not cfg:
        return 0
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    if not isinstance(model_cfg, dict):
        return 0
    depth = int(model_cfg.get("encoder_depth", 3))
    kernel = int(model_cfg.get("encoder_kernel_size", 5))
    mode = str(model_cfg.get("mode", "")).strip().lower()
    encoder_type = str(model_cfg.get("encoder_type", "")).strip().lower()
    if mode.startswith("3d") or "3d" in encoder_type:
        rf = 1 + 2 * (3 - 1) + max(0, depth) * max(0, kernel - 1)
        return max(0, rf // 2)
    dilations = model_cfg.get("convnext_layer_dilations")
    if dilations is None:
        dil_list = [1] * max(0, depth)
    else:
        try:
            dil_list = [int(v) for v in dilations]
        except TypeError:
            dil_list = [1] * max(0, depth)
        if len(dil_list) < depth and dil_list:
            reps = (depth + len(dil_list) - 1) // len(dil_list)
            dil_list = (dil_list * reps)[:depth]
        else:
            dil_list = dil_list[:depth]
    rf = 1 + 2 + 2
    for dilation in dil_list:
        rf += max(0, kernel - 1) * max(1, int(dilation))
    return max(0, rf // 2)


def _latent_border_from_summary_or_config(session_dir: str) -> int:
    cfg = _load_session_config(session_dir)
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    source_cfg = model_cfg if isinstance(model_cfg, dict) and "viz_crop_border" in model_cfg else train_cfg
    if isinstance(source_cfg, dict) and bool(source_cfg.get("viz_crop_border", False)):
        try:
            value = source_cfg.get("viz_crop_border_px", source_cfg.get("inference_discard_margin", "auto"))
            if value is None or str(value).strip().lower() == "auto":
                return _encoder_fov_border_from_config(session_dir)
            return int(max(0, value or 0))
        except (TypeError, ValueError):
            return 0
    summary_path = os.path.join(session_dir, "jepa_energy_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            if "inference_discard_margin" in summary:
                return int(max(0, summary.get("inference_discard_margin") or 0))
        except Exception:
            pass
    return 0


def _compute_pca_2d(x: np.ndarray, fit_max_tokens: int = MAX_PCA_FIT_TOKENS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    fit_x = x
    if x.shape[0] > fit_max_tokens:
        rng = np.random.default_rng(42)
        fit_x = x[rng.choice(x.shape[0], size=fit_max_tokens, replace=False)]
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42).fit(fit_x)
        return pca.transform(x).astype(np.float32)
    except Exception as e:
        print(f"[warning] sklearn PCA(2D) failed: {type(e).__name__}: {e}; falling back to torch.pca_lowrank")
        fit_t = torch.from_numpy(fit_x.astype(np.float32))
        mean = fit_t.mean(dim=0, keepdim=True)
        fit_t = fit_t - mean
        _u, _s, v = torch.pca_lowrank(fit_t, q=2)
        x_t = torch.from_numpy(x.astype(np.float32)) - mean
        return (x_t @ v[:, :2]).cpu().numpy().astype(np.float32)


def _robust_pca(z_flat: np.ndarray, mask_flat: np.ndarray, fit_max_tokens: int = MAX_PCA_FIT_TOKENS) -> np.ndarray:
    """Strip NaN/Inf from input + mask, run PCA, never crash."""
    combined = mask_flat.copy() & np.isfinite(z_flat).all(axis=-1)
    if not combined.any():
        return np.full((z_flat.shape[0], 3), np.nan, dtype=np.float32)
    emb = _compute_pca_3d(z_flat[combined], fit_max_tokens=fit_max_tokens).astype(np.float32)
    out = np.full((z_flat.shape[0], emb.shape[1]), np.nan, dtype=np.float32)
    out[combined] = emb
    return out


def _robust_umap(
    z_flat: np.ndarray,
    mask_flat: np.ndarray,
    l2_normalize: bool = False,
    standardize: bool = False,
    n_neighbors: int = 50,
    min_dist: float = 0.2,
    metric: str = "euclidean",
    random_state: int = 42,
    init: str = "auto",
    fit_max_tokens: int = MAX_PCA_FIT_TOKENS,
) -> np.ndarray:
    """Strip NaN/Inf from input + mask, run UMAP, never crash."""
    combined = mask_flat.copy() & np.isfinite(z_flat).all(axis=-1)
    if not combined.any():
        return np.full((z_flat.shape[0], 3), np.nan, dtype=np.float32)
    z_clean = _preprocess_latents_for_umap(
        z_flat[combined],
        l2_normalize=l2_normalize,
        standardize=standardize,
    )
    emb = _compute_umap_nd(
        z_clean,
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        init=init,
        fit_max_tokens=fit_max_tokens,
    ).astype(np.float32)
    out = np.full((z_flat.shape[0], emb.shape[1]), np.nan, dtype=np.float32)
    out[combined] = emb
    return out


def _compute_pca_3d(x: np.ndarray, fit_max_tokens: int = MAX_PCA_FIT_TOKENS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] > fit_max_tokens:
        rng = np.random.default_rng(42)
        idx = rng.choice(x.shape[0], size=fit_max_tokens, replace=False)
        fit_x = x[idx]
    else:
        fit_x = x

    mean = fit_x.mean(axis=0, keepdims=True)
    fit_centered = fit_x - mean
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=3, random_state=42).fit(fit_x)
        return pca.transform(x).astype(np.float32)
    except Exception as e:
        print(f"[warning] sklearn PCA(3D) failed: {type(e).__name__}: {e}; falling back to numpy SVD")
        try:
            _, _, vt = np.linalg.svd(fit_centered.astype(np.float64), full_matrices=False)
        except np.linalg.LinAlgError:
            try:
                fit_jitter = fit_centered.astype(np.float64) + np.eye(fit_centered.shape[0], fit_centered.shape[1]) * 1e-5
                _, _, vt = np.linalg.svd(fit_jitter, full_matrices=False)
            except np.linalg.LinAlgError:
                raise
        comps = vt[:3].T.astype(np.float32)
        return ((x - mean) @ comps).astype(np.float32)


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
    fit_max_tokens: int = 65536,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    init_mode = str(init).lower()
    if init_mode not in ("spectral", "random"):
        init_mode = "spectral"
    fallback = _compute_pca_3d(x, fit_max_tokens=fit_max_tokens)
    if int(n_components) != 3:
        fallback = fallback[:, : int(n_components)]

    if x.shape[0] > fit_max_tokens:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(x.shape[0], size=fit_max_tokens, replace=False)
        fit_x = x[idx]
        needs_fit_transform = True
    else:
        needs_fit_transform = False

    if torch.cuda.is_available():
        try:
            from cuml.manifold import UMAP as CuMLUMAP

            print("[inference] UMAP backend: cuML (GPU)")
            model = CuMLUMAP(
                n_components=n_components,
                n_neighbors=int(n_neighbors),
                min_dist=float(min_dist),
                metric=str(metric),
                random_state=int(random_state),
                init=init_mode,
            )
            if needs_fit_transform:
                model.fit(fit_x)
                return model.transform(x)
            return model.fit_transform(x)
        except ModuleNotFoundError as e:
            if e.name == "cuml" or str(e).endswith("No module named 'cuml'"):
                print("[inference] UMAP backend: cuML not installed; trying CPU/torch alternatives")
            else:
                print(f"[warning] cuML UMAP import failed: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"[warning] cuML UMAP failed: {type(e).__name__}: {e}")

    enable_torchdr = os.environ.get("SAJEPA_ENABLE_TORCHDR_UMAP", "").strip().lower() in {"1", "true", "yes", "on"}
    if enable_torchdr:
        try:
            import torchdr

            if hasattr(torchdr, "UMAP"):
                model = torchdr.UMAP(
                    n_components=n_components,
                    n_neighbors=int(n_neighbors),
                    min_dist=float(min_dist),
                )
                if needs_fit_transform:
                    model.fit(torch.from_numpy(fit_x.astype(np.float32)))
                    z = model.transform(torch.from_numpy(x.astype(np.float32)))
                else:
                    z = model.fit_transform(torch.from_numpy(x.astype(np.float32)))
                if isinstance(z, torch.Tensor):
                    return z.cpu().numpy()
                return np.asarray(z)
        except Exception as e:
            print(f"[warning] torchdr UMAP failed: {type(e).__name__}: {e}")

    enable_cpu_umap_default = sys.platform != "darwin"
    enable_cpu_umap = os.environ.get("SAJEPA_ENABLE_CPU_UMAP", str(int(enable_cpu_umap_default))).strip().lower()
    if enable_cpu_umap in {"1", "true", "yes", "on"}:
        try:
            import umap

            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=int(n_neighbors),
                min_dist=float(min_dist),
                metric=str(metric),
                random_state=int(random_state),
                init=init_mode,
            )
            if needs_fit_transform:
                model.fit(fit_x)
                return model.transform(x)
            return model.fit_transform(x)
        except Exception as e:
            print(f"[warning] umap-learn failed: {type(e).__name__}: {e}")

    print("[warning] UMAP backend unavailable/disabled; using PCA coordinates as UMAP fallback")
    return fallback.astype(np.float32, copy=False)


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


def _target_allowed_validity_mask(outputs: dict, target_h: int, target_w: int) -> np.ndarray | None:
    """Return the user-provided valid-target region at latent resolution."""
    target_allowed = outputs.get("target_allowed_mask_map")
    if target_allowed is None:
        return None
    tm = torch.as_tensor(target_allowed)
    if tm.dim() == 2:
        tm = tm.unsqueeze(0).unsqueeze(0)
    elif tm.dim() == 3:
        tm = tm.unsqueeze(1)
    elif tm.dim() == 5:
        # B,C,D,H,W: use the center depth plane for 2D dashboard embeddings.
        tm = tm[:, :, tm.shape[2] // 2]
    if tm.dim() != 4 or tm.shape[0] < 1:
        return None
    tm0 = tm[0:1, 0:1].float()
    if int(tm0.shape[-2]) != int(target_h) or int(tm0.shape[-1]) != int(target_w):
        tm0 = F.interpolate(tm0, size=(target_h, target_w), mode="nearest")
    return (tm0[0, 0] > 0.5).detach().cpu().numpy().astype(bool)


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
    latents_valid = latents_flat[mask_flat]
    # Drop rows where the model produced NaN/Inf latents.
    finite_rows = np.isfinite(latents_valid).all(axis=-1)
    if not finite_rows.any():
        return np.full((n_total, 3), np.nan, dtype=np.float32)
    if finite_rows.all():
        emb_valid = compute_fn(latents_valid)
    else:
        emb_finite = compute_fn(latents_valid[finite_rows])
        emb_valid = np.full((latents_valid.shape[0], emb_finite.shape[1]), np.nan, dtype=np.float32)
        emb_valid[finite_rows] = emb_finite
    n_emb = emb_valid.shape[1]
    result = np.full((n_total, n_emb), np.nan, dtype=np.float32)
    result[mask_flat] = emb_valid
    return result


def _target_location_yx(tloc: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    """Return (loc_y, loc_x) from target_locations tensor (B,K,2) or (B,K,3)."""
    tloc = torch.as_tensor(tloc)
    if tloc.dim() == 3 and tloc.shape[-1] >= 2:
        if tloc.shape[-1] >= 3:
            return tloc[..., -2].long(), tloc[..., -1].long()
        return tloc[..., 0].long(), tloc[..., 1].long()
    raise ValueError(f"Expected target_locations shape (B,K,2) or (B,K,3), got {tuple(tloc.shape)}")


def _target_region_mask_from_outputs(outputs: dict, target_h: int, target_w: int) -> np.ndarray:
    """Return valid target/mask positions at latent-map resolution for sample 0."""
    target_mask = outputs.get("target_mask_map")
    if target_mask is not None:
        tm = torch.as_tensor(target_mask)
        if tm.dim() == 3:
            tm = tm.unsqueeze(1)
        if tm.dim() == 4 and tm.shape[0] > 0:
            tm0 = tm[0:1, 0:1].float()
            if int(tm0.shape[-2]) != int(target_h) or int(tm0.shape[-1]) != int(target_w):
                tm0 = F.interpolate(tm0, size=(target_h, target_w), mode="nearest")
            return (tm0[0, 0] > 0.5).detach().cpu().numpy().astype(bool)

    mask = np.zeros((target_h, target_w), dtype=bool)
    target_locations = outputs.get("target_locations")
    if target_locations is None:
        return mask
    tloc = torch.as_tensor(target_locations)
    if tloc.dim() != 3 or tloc.shape[0] < 1:
        return mask
    target_valid = outputs.get("target_valid")
    if target_valid is None:
        tvalid = torch.ones(tloc.shape[:2], dtype=torch.bool)
    else:
        tvalid = torch.as_tensor(target_valid).bool()
    x_ref = outputs.get("x_clean_raw", outputs.get("x_clean"))
    if x_ref is not None and torch.as_tensor(x_ref).dim() >= 4:
        h_in = max(1, int(torch.as_tensor(x_ref).shape[-2]))
        w_in = max(1, int(torch.as_tensor(x_ref).shape[-1]))
    else:
        h_in = int(target_h)
        w_in = int(target_w)
    scale_y = float(target_h) / float(h_in)
    scale_x = float(target_w) / float(w_in)
    patch_size = int(max(1, outputs.get("patch_size", 1)))
    rad_y = max(0, int(np.ceil(0.5 * patch_size * scale_y)))
    rad_x = max(0, int(np.ceil(0.5 * patch_size * scale_x)))
    loc_y, loc_x = _target_location_yx(tloc)
    for i in range(int(tloc.shape[1])):
        if i >= int(tvalid.shape[1]) or not bool(tvalid[0, i].item()):
            continue
        cy = int(round(float(loc_y[0, i].item()) * scale_y))
        cx = int(round(float(loc_x[0, i].item()) * scale_x))
        y0, y1 = max(0, cy - rad_y), min(target_h, cy + rad_y + 1)
        x0, x1 = max(0, cx - rad_x), min(target_w, cx + rad_x + 1)
        if y0 < y1 and x0 < x1:
            mask[y0:y1, x0:x1] = True
    return mask


def save_inference_dashboard(session_dir: str, outputs: dict, umap_cfg: dict | None = None) -> str:
    umap_cfg = dict(umap_cfg or {})
    fit_max_tokens = int(umap_cfg.get("fit_max_tokens", 65536))
    umap_n_neighbors = int(umap_cfg.get("n_neighbors", 15))
    umap_min_dist = float(umap_cfg.get("min_dist", 0.05))
    umap_metric = str(umap_cfg.get("metric", "cosine"))
    umap_random_state = int(umap_cfg.get("random_state", 42))
    umap_init = str(umap_cfg.get("init", "spectral")).lower()
    umap_l2_normalize = bool(umap_cfg.get("l2_normalize", False))
    umap_standardize = bool(umap_cfg.get("standardize", False))

    x_clean = outputs.get("x_clean")
    if x_clean is None:
        x_clean = outputs.get("x_clean_raw")
    x_context = outputs.get("x_context")
    if x_context is None:
        x_context = outputs.get("x_context_raw", x_clean)
    if x_clean is None:
        raise RuntimeError("Inference dashboard requires x_clean or x_clean_raw in outputs.")
    x_clean_raw = outputs.get("x_clean_raw", x_clean)
    x_context_raw = outputs.get("x_context_raw", x_context)
    target_locations = outputs["target_locations"]
    pred_map = outputs["pred_map"]
    masked_pred_map = outputs.get("masked_pred_map", pred_map)
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
    border_px = int(max(0, min(_latent_border_from_summary_or_config(session_dir), h_lat // 2, w_lat // 2)))
    if border_px > 0:
        valid_mask_2d = valid_mask_2d.copy()
        valid_mask_2d[:border_px, :] = False
        valid_mask_2d[h_lat - border_px :, :] = False
        valid_mask_2d[:, :border_px] = False
        valid_mask_2d[:, w_lat - border_px :] = False
        print(f"[dashboard] latent PCA/UMAP border mask: border_px={border_px}")
    target_allowed_mask = _target_allowed_validity_mask(outputs, h_lat, w_lat)
    if target_allowed_mask is not None:
        valid_mask_2d = valid_mask_2d & target_allowed_mask
        print(
            "[dashboard] latent PCA/UMAP target-allowed mask: "
            f"valid_pixels={int(valid_mask_2d.sum())}/{int(valid_mask_2d.size)}"
        )
    valid_mask_flat = valid_mask_2d.reshape(-1)  # [H_lat * W_lat]

    # Render sampled target locations for first sample.
    target_vis = np.zeros_like(orig, dtype=np.float32)
    loc_y, loc_x = _target_location_yx(target_locations)
    for i in range(target_locations.shape[1]):
        cy = int(loc_y[0, i].item())
        cx = int(loc_x[0, i].item())
        if 0 <= cy < target_vis.shape[0] and 0 <= cx < target_vis.shape[1]:
            target_vis[cy, cx] = 1.0

    # Session plot compatibility artifacts.
    results_dir = os.path.join(session_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    for legacy_name in (
        "pca_embeddings.npy",
        "umap_x.npy",
        "umap_y.npy",
        "umap_z.npy",
        "pca_x.npy",
        "pca_y.npy",
        "pca_z.npy",
    ):
        legacy_path = os.path.join(results_dir, legacy_name)
        if os.path.exists(legacy_path):
            os.remove(legacy_path)
    for legacy_name in os.listdir(session_dir):
        if legacy_name == "pca_embeddings.npy" or re.match(r"umap_embeddings_[0-9a-f]+\.npy$", legacy_name):
            legacy_path = os.path.join(session_dir, legacy_name)
            if os.path.exists(legacy_path):
                os.remove(legacy_path)

    def _save_branch_embeddings(branch_name: str, fmap: torch.Tensor):
        # Use sample-0 dense latent map (H*W tokens) for branch-specific plotly 2D color + 3D scatter.
        h_map = int(fmap.shape[-2])
        w_map = int(fmap.shape[-1])
        latent_t = fmap[0]
        if latent_t.dim() == 4:
            # C,D,H,W -> C,H,W center slice for 3D slab inference dashboards.
            latent_t = latent_t[:, latent_t.shape[1] // 2]
        latent_map = latent_t.detach().cpu().numpy().astype(np.float32)
        if latent_map.ndim != 3:
            raise RuntimeError(
                f"{branch_name} latent map must be CxHxW after slicing, got shape={latent_map.shape}"
            )
        z = np.transpose(latent_map, (1, 2, 0)).reshape(-1, fmap.shape[1]).astype(np.float32)
        pca3 = _robust_pca(z, valid_mask_flat, fit_max_tokens=fit_max_tokens)
        umap3 = _robust_umap(
            z, valid_mask_flat,
            l2_normalize=umap_l2_normalize,
            standardize=umap_standardize,
            n_neighbors=umap_n_neighbors, min_dist=umap_min_dist,
            metric=umap_metric, random_state=umap_random_state,
            init=umap_init, fit_max_tokens=fit_max_tokens,
        )

        expected_n = h_map * w_map
        if pca3.shape != (expected_n, 3):
            raise RuntimeError(f"PCA embedding shape {pca3.shape} cannot reshape exactly to (3,{h_map},{w_map})")
        if umap3.shape != (expected_n, 3):
            raise RuntimeError(f"UMAP embedding shape {umap3.shape} cannot reshape exactly to (3,{h_map},{w_map})")
        pca_map = np.transpose(pca3.reshape(h_map, w_map, 3), (2, 0, 1)).astype(np.float32)
        umap_map = np.transpose(umap3.reshape(h_map, w_map, 3), (2, 0, 1)).astype(np.float32)
        np.save(os.path.join(results_dir, f"{branch_name}_spatial_shape.npy"), np.asarray([h_map, w_map], dtype=np.int64))
        np.save(os.path.join(results_dir, f"{branch_name}_latent_vectors_full.npy"), latent_map)
        np.save(os.path.join(results_dir, f"{branch_name}_pca_xyz.npy"), pca_map)
        np.save(os.path.join(results_dir, f"{branch_name}_umap_xyz.npy"), umap_map)
        for legacy_name in (
            f"{branch_name}_pca_x.npy",
            f"{branch_name}_pca_y.npy",
            f"{branch_name}_pca_z.npy",
            f"{branch_name}_umap_x.npy",
            f"{branch_name}_umap_y.npy",
            f"{branch_name}_umap_z.npy",
        ):
            legacy_path = os.path.join(results_dir, legacy_name)
            if os.path.exists(legacy_path):
                os.remove(legacy_path)
        return latent_map, pca_map, umap_map

    default_latent_map, default_pca_map, default_umap_map = _save_branch_embeddings("predict", pred_map)
    _save_branch_embeddings("masked_predict", masked_pred_map)
    np.save(os.path.join(results_dir, "latent_vectors_full.npy"), default_latent_map)
    np.save(os.path.join(results_dir, "pca_xyz.npy"), default_pca_map)
    np.save(os.path.join(results_dir, "umap_xyz.npy"), default_umap_map)
    _save_branch_embeddings("target", gt_map)
    if context_map is not None:
        _save_branch_embeddings("context", context_map)

    return results_dir


def _build_volume_validity_mask(x_clean_raw: torch.Tensor, target_d: int, target_h: int, target_w: int) -> np.ndarray:
    """Build a bool validity mask at latent volume resolution from the raw input."""
    if x_clean_raw is None:
        return np.ones((target_d, target_h, target_w), dtype=bool)
    x = torch.as_tensor(x_clean_raw)
    if x.dim() == 4:
        valid_2d = _build_input_validity_mask(x, target_h, target_w)
        return np.broadcast_to(valid_2d[None, :, :], (target_d, target_h, target_w)).copy()
    if x.dim() != 5:
        return np.ones((target_d, target_h, target_w), dtype=bool)
    inp = x[0:1, 0:1]
    valid = ((inp.abs() > 1e-12) & torch.isfinite(inp)).float()
    pooled = F.interpolate(valid, size=(target_d, target_h, target_w), mode="nearest")
    return (pooled[0, 0] > 0.5).detach().cpu().numpy().astype(bool)


def save_volumetric_umap_embeddings(session_dir: str, outputs: dict, umap_cfg: dict | None = None) -> str:
    """Train UMAP on valid volumetric latent voxels and save artifacts."""
    umap_cfg = dict(umap_cfg or {})
    umap_n_neighbors = int(umap_cfg.get("n_neighbors", 15))
    umap_min_dist = float(umap_cfg.get("min_dist", 0.05))
    umap_metric = str(umap_cfg.get("metric", "cosine"))
    umap_random_state = int(umap_cfg.get("random_state", 42))
    umap_init = str(umap_cfg.get("init", "spectral")).lower()
    umap_l2_normalize = bool(umap_cfg.get("l2_normalize", False))
    umap_standardize = bool(umap_cfg.get("standardize", False))
    max_points_raw = umap_cfg.get("volumetric_max_points", 100000)
    max_points = None if max_points_raw is None else int(max(128, max_points_raw))
    sample_seed = int(umap_cfg.get("volumetric_sample_seed", umap_random_state))

    fmap = outputs.get("context_map")
    if fmap is None:
        fmap = outputs["pred_map"]
    if fmap.dim() != 5:
        raise ValueError(f"Expected 3D map B,C,D,H,W, got shape={tuple(fmap.shape)}")

    d_map, h_map, w_map = int(fmap.shape[-3]), int(fmap.shape[-2]), int(fmap.shape[-1])
    z = fmap[0].detach().cpu().permute(1, 2, 3, 0).reshape(-1, fmap.shape[1]).numpy().astype(np.float32)
    input_ref = outputs.get("x_clean_raw", outputs.get("x_clean"))
    input_valid = _build_volume_validity_mask(input_ref, d_map, h_map, w_map).reshape(-1)
    finite_valid = np.isfinite(z).all(axis=1)
    valid = input_valid & finite_valid
    valid_indices = np.flatnonzero(valid).astype(np.int64)
    if valid_indices.size < 4:
        raise RuntimeError(f"volumetric UMAP: fewer than 4 valid latent voxels ({valid_indices.size})")
    n_valid = int(valid_indices.size)
    rng = np.random.default_rng(sample_seed)
    if max_points is not None and n_valid > max_points:
        selected_indices = rng.choice(valid_indices, size=int(max_points), replace=False)
        selection = "absolute_cap"
    else:
        selected_indices = valid_indices
        selection = "full_valid_inference_extent"
    selected_indices = np.sort(selected_indices.astype(np.int64, copy=False))
    z_sub = z[selected_indices]

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
    np.save(os.path.join(results_dir, "volumetric_umap_indices.npy"), selected_indices.astype(np.int64))
    np.save(os.path.join(results_dir, "volumetric_umap_latents.npy"), z_sub.astype(np.float32))
    np.save(os.path.join(results_dir, "volumetric_umap_xyz.npy"), umap3)
    np.save(os.path.join(results_dir, "volumetric_pca_xyz.npy"), pca3)

    meta = {
        "n_total_voxels": int(z.shape[0]),
        "n_finite_voxels": int(np.count_nonzero(finite_valid)),
        "n_valid_voxels": n_valid,
        "n_selected": int(selected_indices.size),
        "max_points": None if max_points is None else int(max_points),
        "sample_seed": int(sample_seed),
        "selection": selection,
    }
    meta_path = os.path.join(session_dir, "volumetric_umap_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path
