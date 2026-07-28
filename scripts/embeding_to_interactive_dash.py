#!/usr/bin/env python
"""Build a self-contained click-to-distance HTML dashboard for embedding maps."""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import json
from pathlib import Path

import numpy as np


def _as_hwc_features(arr: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D {name} array, got shape {arr.shape}")
    if arr.shape[-1] <= 128 and arr.shape[0] > arr.shape[-1] and arr.shape[1] > arr.shape[-1]:
        pass
    elif arr.shape[0] <= 128 and arr.shape[1] > 1 and arr.shape[2] > 1:
        arr = np.moveaxis(arr, 0, -1)
    elif arr.shape[-1] <= 128:
        pass
    else:
        raise ValueError(
            "expected shape (C,H,W) or (H,W,C) with a reasonable channel count; "
            f"got {arr.shape}"
        )
    return arr


def _display_xyz(features: np.ndarray) -> np.ndarray:
    if features.shape[-1] < 3:
        raise ValueError(f"display array needs at least 3 channels, got shape {features.shape}")
    return np.asarray(features[..., :3], dtype=np.float32)


def _robust_rgb(display_xyz: np.ndarray, similarity_features: np.ndarray, lo_pct: float, hi_pct: float) -> tuple[np.ndarray, dict]:
    valid = np.isfinite(display_xyz).all(axis=-1) & np.isfinite(similarity_features).all(axis=-1)
    rgb = np.zeros((*display_xyz.shape[:2], 4), dtype=np.uint8)
    rgb[..., 3] = np.where(valid, 255, 0).astype(np.uint8)

    stats = {
        "shape": list(display_xyz.shape),
        "similarity_shape": list(similarity_features.shape),
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.size),
        "channels": [],
    }
    if not valid.any():
        return rgb, stats

    for channel in range(3):
        values = display_xyz[..., channel][valid]
        lo, hi = np.nanpercentile(values, [lo_pct, hi_pct])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            rgb[..., channel] = 127
            lo = hi = float(values[0]) if values.size else 0.0
        else:
            scaled = (display_xyz[..., channel] - lo) / (hi - lo)
            scaled = np.clip(scaled, 0.0, 1.0)
            rgb[..., channel] = np.where(valid, np.round(scaled * 255.0), 0).astype(np.uint8)
        stats["channels"].append(
            {
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "lo_pct": float(lo),
                "hi_pct": float(hi),
            }
        )
    return rgb, stats


def _rgba_base64(rgba: np.ndarray) -> tuple[str, int, int]:
    """Return (base64, width, height) of raw RGBA bytes."""
    h, w = rgba.shape[:2]
    raw = np.ascontiguousarray(rgba, dtype=np.uint8).tobytes()
    return base64.b64encode(raw).decode("ascii"), w, h


def _float32_base64(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr.astype("<f4", copy=False))
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def build_html(
    input_path: Path,
    output_path: Path | None,
    lo_pct: float,
    hi_pct: float,
    *,
    display_label: str = "embedding",
) -> Path:
    input_path = input_path.expanduser().resolve()
    if output_path is None:
        output_path = input_path.with_suffix(".html")
    else:
        output_path = output_path.expanduser().resolve()

    display_features = _as_hwc_features(np.load(input_path), name="display")
    display_xyz = _display_xyz(display_features)
    similarity_features = display_features
    rgba, stats = _robust_rgb(display_xyz, similarity_features, lo_pct=lo_pct, hi_pct=hi_pct)
    rgba_b64, img_w, img_h = _rgba_base64(rgba)
    rgba_shape_json = json.dumps([img_h, img_w])
    display_b64 = _float32_base64(display_xyz)
    similarity_b64 = _float32_base64(similarity_features)

    display_label = str(display_label)
    similarity_label = display_label
    title = f"{display_label} interactive dashboard"
    stats_json = json.dumps(stats, separators=(",", ":"))
    source_text = html_lib.escape(str(input_path))
    title_text = html_lib.escape(title)
    display_label_text = html_lib.escape(display_label)
    similarity_label_text = html_lib.escape(similarity_label)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_text}</title>
<style>
  :root {{
    color-scheme: dark;
    --bg: #111317;
    --panel: #1d222a;
    --text: #e9edf4;
    --muted: #9aa6b8;
    --line: #374151;
    --accent: #67d4ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    min-height: 100vh;
    background: var(--bg);
    color: var(--text);
    font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--line);
    background: var(--panel);
  }}
  h1 {{
    margin: 0;
    font-size: 16px;
    font-weight: 650;
  }}
  .meta {{
    color: var(--muted);
    font-size: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .toolbar {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }}
  .hint {{
    color: var(--muted);
    font-size: 12px;
  }}
  button {{
    border: 1px solid var(--line);
    border-radius: 6px;
    background: #252c36;
    color: var(--text);
    padding: 6px 10px;
    cursor: pointer;
  }}
  button:hover {{ border-color: var(--accent); }}
  label.control {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--muted);
    font-size: 12px;
    white-space: nowrap;
  }}
  input[type="range"] {{
    width: 130px;
    accent-color: var(--accent);
  }}
  select {{
    border: 1px solid var(--line);
    border-radius: 6px;
    background: #252c36;
    color: var(--text);
    padding: 5px 8px;
  }}
  main {{
    height: calc(100vh - 58px);
    display: grid;
    grid-template-columns: minmax(0, 1fr) 280px;
  }}
  .stage {{
    position: relative;
    overflow: hidden;
    background:
      linear-gradient(45deg, #0d0f13 25%, transparent 25%),
      linear-gradient(-45deg, #0d0f13 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, #0d0f13 75%),
      linear-gradient(-45deg, transparent 75%, #0d0f13 75%);
    background-size: 24px 24px;
    background-position: 0 0, 0 12px, 12px -12px, -12px 0;
  }}
  canvas {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    image-rendering: pixelated;
    cursor: grab;
  }}
  canvas.dragging {{ cursor: grabbing; }}
  aside {{
    border-left: 1px solid var(--line);
    background: var(--panel);
    padding: 14px;
    overflow: auto;
  }}
  dl {{ margin: 0; }}
  dt {{ color: var(--muted); font-size: 12px; margin-top: 12px; }}
  dd {{ margin: 3px 0 0; overflow-wrap: anywhere; }}
  .channel {{
    display: grid;
    grid-template-columns: 20px 1fr;
    gap: 8px;
    align-items: center;
    margin-top: 10px;
  }}
  .swatch {{ width: 16px; height: 16px; border-radius: 4px; }}
  .r {{ background: #ff5b6b; }}
  .g {{ background: #34d399; }}
  .b {{ background: #60a5fa; }}
  .colorbar-wrap {{
    margin-top: 6px;
  }}
  #colorbar {{
    display: block;
    width: 100%;
    height: 18px;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: #0d0f13;
  }}
  .colorbar-labels {{
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin-top: 4px;
    color: var(--muted);
    font-size: 11px;
  }}
  .colorbar-labels span:last-child {{
    text-align: right;
  }}
  @media (max-width: 860px) {{
    main {{ grid-template-columns: 1fr; grid-template-rows: minmax(380px, 1fr) auto; }}
    aside {{ border-left: 0; border-top: 1px solid var(--line); }}
  }}
</style>
</head>
<body>
<header>
  <div>
    <h1>{title_text}</h1>
    <div class="meta">source={display_label_text}: {source_text}</div>
  </div>
  <div class="toolbar">
    <button id="fit" type="button">Fit</button>
    <button id="one" type="button">1:1</button>
    <button id="rgb" type="button">RGB</button>
    <button id="flipY" type="button">Flip Display Y</button>
    <button id="saveNpy" type="button">Save NPY</button>
    <label class="control">Distance <select id="distanceMetric">
      <option value="euclidean">Euclidean</option>
      <option value="cosine">Cosine</option>
    </select></label>
    <label class="control">Gamma <input id="gamma" type="range" min="0.05" max="8" step="0.05" value="0.65"></label>
    <label class="control">Map <select id="colormap">
      <option value="turbo">Turbo</option>
      <option value="viridis">Viridis</option>
      <option value="plasma">Plasma</option>
      <option value="inferno">Inferno</option>
      <option value="magma">Magma</option>
      <option value="cividis">Cividis</option>
      <option value="cubehelix">Cubehelix</option>
      <option value="icefire">Ice/Fire</option>
      <option value="fire">Fire</option>
      <option value="cyanmagenta">Cyan/Magenta</option>
      <option value="redblue">Red/Blue</option>
      <option value="gray">Gray</option>
    </select></label>
  </div>
</header>
<main>
  <section class="stage"><canvas id="view"></canvas></section>
  <aside>
    <dl>
      <dt>Display array</dt><dd id="shape"></dd>
      <dt>Similarity array</dt><dd id="simshape"></dd>
      <dt>Valid pixels</dt><dd id="valid"></dd>
      <dt>Cursor</dt><dd id="cursor">-</dd>
      <dt>Selected</dt><dd id="selected">click a valid pixel</dd>
      <dt>View</dt><dd id="mode">RGB projection</dd>
      <dt>Distance scale</dt><dd id="window">-</dd>
      <dt>Colorbar</dt>
      <dd>
        <div class="colorbar-wrap">
          <canvas id="colorbar" width="220" height="18"></canvas>
          <div class="colorbar-labels">
            <span id="colorbarNear">near / low distance</span>
            <span id="colorbarFar">far / high distance</span>
          </div>
        </div>
      </dd>
      <dt>Save orientation</dt><dd>original array shape/orientation</dd>
      <dt>Zoom</dt><dd id="zoom">-</dd>
    </dl>
    <p class="hint">Click a valid pixel to draw a distance map against the selected {similarity_label_text} vector. Use the toolbar to switch Euclidean or cosine distance.</p>
    <div class="channel"><span class="swatch r"></span><span id="ch0"></span></div>
    <div class="channel"><span class="swatch g"></span><span id="ch1"></span></div>
    <div class="channel"><span class="swatch b"></span><span id="ch2"></span></div>
  </aside>
</main>
<script>
const RGBA_B64 = {json.dumps(rgba_b64)};
const IMG_W = {img_w};
const IMG_H = {img_h};
const DISPLAY_B64 = {json.dumps(display_b64)};
const SIMILARITY_B64 = {json.dumps(similarity_b64)};
const STATS = {stats_json};
const DISPLAY_LABEL = {json.dumps(display_label)};
const SIMILARITY_LABEL = {json.dumps(similarity_label)};
const DISPLAY_CHANNELS = 3;
const SIMILARITY_CHANNELS = STATS.similarity_shape[2];
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
// Decode raw RGBA to a canvas; no image library needed.
function decodeRGBA(b64, w, h) {{
  const raw = atob(b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  const imgData = new ImageData(new Uint8ClampedArray(bytes.buffer, bytes.byteOffset, bytes.byteLength), w, h);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").putImageData(imgData, 0, 0);
  return c;
}}
const rgbCanvas = decodeRGBA(RGBA_B64, IMG_W, IMG_H);
const img = {{ width: IMG_W, height: IMG_H }};
let xyz = null;
let similarity = null;
let simCanvas = null;
let simValues = null;
let simMin = 0;
let simMax = 1;
let selectedPx = null;
let selectedPy = null;
let mode = "rgb";
let flipY = false;
let scale = 1;
let offsetX = 0;
let offsetY = 0;
let dragging = false;
let moved = false;
let downX = 0;
let downY = 0;
let lastX = 0;
let lastY = 0;

function decodeFloat32(base64) {{
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}}

function finiteVector(values, index, channels) {{
  for (let c = 0; c < channels; c++) {{
    if (!Number.isFinite(values[index + c])) return false;
  }}
  return true;
}}

function fmt(v) {{
  return Number.isFinite(v) ? v.toFixed(4) : String(v);
}}

function lerp(a, b, t) {{
  return a + (b - a) * t;
}}

function lerpColor(a, b, t) {{
  return [
    Math.round(lerp(a[0], b[0], t)),
    Math.round(lerp(a[1], b[1], t)),
    Math.round(lerp(a[2], b[2], t))
  ];
}}

function rampColor(stops, t) {{
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {{
    const left = stops[i];
    const right = stops[i + 1];
    if (t <= right[0]) {{
      const local = (t - left[0]) / Math.max(1e-12, right[0] - left[0]);
      return lerpColor(left[1], right[1], local);
    }}
  }}
  return stops[stops.length - 1][1];
}}

function resize() {{
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  renderColorbar();
  draw();
}}

function fit() {{
  scale = Math.min(canvas.width / IMG_W, canvas.height / IMG_H);
  offsetX = (canvas.width - IMG_W * scale) / 2;
  offsetY = (canvas.height - IMG_H * scale) / 2;
  draw();
}}

function oneToOne() {{
  const dpr = window.devicePixelRatio || 1;
  scale = dpr;
  offsetX = (canvas.width - IMG_W * scale) / 2;
  offsetY = (canvas.height - IMG_H * scale) / 2;
  draw();
}}

function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = false;
  const source = mode === "sim" && simCanvas ? simCanvas : rgbCanvas;
  ctx.save();
  if (flipY) {{
    ctx.translate(0, canvas.height);
    ctx.scale(1, -1);
    ctx.drawImage(source, offsetX, canvas.height - offsetY - IMG_H * scale, IMG_W * scale, IMG_H * scale);
  }} else {{
    ctx.drawImage(source, offsetX, offsetY, IMG_W * scale, IMG_H * scale);
  }}
  ctx.restore();
  drawCrosshair();
  document.getElementById("zoom").textContent = `${{(scale / (window.devicePixelRatio || 1) * 100).toFixed(1)}}%`;
}}

function selectedCanvasPoint() {{
  if (selectedPx === null || selectedPy === null) return null;
  if (selectedPx < 0 || selectedPy < 0 || selectedPx >= IMG_W || selectedPy >= IMG_H) return null;
  const cx = offsetX + (selectedPx + 0.5) * scale;
  const imageY = offsetY + (selectedPy + 0.5) * scale;
  const cy = flipY ? canvas.height - imageY : imageY;
  return {{ cx, cy }};
}}

function drawCrosshair() {{
  const pt = selectedCanvasPoint();
  if (!pt) return;
  const dpr = window.devicePixelRatio || 1;
  const arm = Math.max(8 * dpr, Math.min(18 * dpr, scale * 3));
  const gap = Math.max(2 * dpr, Math.min(5 * dpr, scale * 0.6));
  const radius = Math.max(3 * dpr, Math.min(8 * dpr, scale * 0.7));
  ctx.save();
  ctx.lineWidth = Math.max(1.5 * dpr, 2);
  ctx.strokeStyle = "rgba(0, 0, 0, 0.9)";
  strokeCrosshair(pt.cx, pt.cy, arm, gap, radius);
  ctx.lineWidth = Math.max(1 * dpr, 1.25);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.98)";
  strokeCrosshair(pt.cx, pt.cy, arm, gap, radius);
  ctx.restore();
}}

function strokeCrosshair(cx, cy, arm, gap, radius) {{
  ctx.beginPath();
  ctx.moveTo(cx - arm, cy);
  ctx.lineTo(cx - gap, cy);
  ctx.moveTo(cx + gap, cy);
  ctx.lineTo(cx + arm, cy);
  ctx.moveTo(cx, cy - arm);
  ctx.lineTo(cx, cy - gap);
  ctx.moveTo(cx, cy + gap);
  ctx.lineTo(cx, cy + arm);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();
}}

function canvasToPixel(clientX, clientY) {{
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const x = (clientX - rect.left) * dpr;
  const y = (clientY - rect.top) * dpr;
  return {{
    px: Math.floor((x - offsetX) / scale),
    py: Math.floor(((flipY ? canvas.height - y : y) - offsetY) / scale)
  }};
}}

function turboColor(t) {{
  t = Math.max(0, Math.min(1, t));
  const r = 34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05))));
  const g = 23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (1073.77 + t * 707.56))));
  const b = 27.2 + t * (3211.1 + t * (-15327.97 + t * (27814.0 + t * (-22569.18 + t * 6838.66))));
  return [
    Math.max(0, Math.min(255, Math.round(r))),
    Math.max(0, Math.min(255, Math.round(g))),
    Math.max(0, Math.min(255, Math.round(b)))
  ];
}}

function colorFor(t) {{
  const map = document.getElementById("colormap").value;
  if (map === "turbo") return turboColor(t);
  if (map === "viridis") return rampColor([
    [0.00, [68, 1, 84]],
    [0.25, [59, 82, 139]],
    [0.50, [33, 145, 140]],
    [0.75, [94, 201, 98]],
    [1.00, [253, 231, 37]]
  ], t);
  if (map === "plasma") return rampColor([
    [0.00, [13, 8, 135]],
    [0.25, [126, 3, 168]],
    [0.50, [204, 71, 120]],
    [0.75, [248, 149, 64]],
    [1.00, [240, 249, 33]]
  ], t);
  if (map === "inferno") return rampColor([
    [0.00, [0, 0, 4]],
    [0.25, [87, 15, 109]],
    [0.50, [187, 55, 84]],
    [0.75, [249, 142, 8]],
    [1.00, [252, 255, 164]]
  ], t);
  if (map === "magma") return rampColor([
    [0.00, [0, 0, 4]],
    [0.25, [80, 18, 123]],
    [0.50, [182, 54, 121]],
    [0.75, [251, 136, 97]],
    [1.00, [252, 253, 191]]
  ], t);
  if (map === "cividis") return rampColor([
    [0.00, [0, 34, 77]],
    [0.25, [39, 75, 108]],
    [0.50, [104, 113, 111]],
    [0.75, [177, 155, 84]],
    [1.00, [253, 231, 55]]
  ], t);
  if (map === "cubehelix") return rampColor([
    [0.00, [0, 0, 0]],
    [0.20, [29, 43, 80]],
    [0.40, [27, 102, 91]],
    [0.60, [126, 121, 70]],
    [0.80, [199, 148, 194]],
    [1.00, [255, 255, 255]]
  ], t);
  if (map === "icefire") return rampColor([
    [0.00, [0, 18, 97]],
    [0.25, [0, 157, 196]],
    [0.48, [236, 248, 255]],
    [0.52, [255, 243, 224]],
    [0.75, [226, 84, 35]],
    [1.00, [90, 0, 0]]
  ], t);
  if (map === "fire") return rampColor([
    [0.00, [0, 0, 0]],
    [0.25, [92, 0, 80]],
    [0.50, [210, 40, 40]],
    [0.75, [255, 169, 36]],
    [1.00, [255, 255, 210]]
  ], t);
  if (map === "cyanmagenta") return rampColor([
    [0.00, [0, 0, 0]],
    [0.25, [0, 178, 210]],
    [0.50, [235, 235, 235]],
    [0.75, [228, 44, 180]],
    [1.00, [255, 235, 255]]
  ], t);
  if (map === "redblue") return rampColor([
    [0.00, [49, 54, 149]],
    [0.45, [116, 173, 209]],
    [0.50, [247, 247, 247]],
    [0.55, [244, 109, 67]],
    [1.00, [165, 0, 38]]
  ], t);
  const g = Math.round(t * 255);
  return [g, g, g];
}}

function renderColorbar() {{
  const bar = document.getElementById("colorbar");
  const bctx = bar.getContext("2d");
  const rect = bar.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  bar.width = Math.max(2, Math.round(rect.width * dpr));
  bar.height = Math.max(2, Math.round(rect.height * dpr));
  const gamma = Number(document.getElementById("gamma").value);
  const out = new ImageData(bar.width, bar.height);
  for (let x = 0; x < bar.width; x++) {{
    const distanceT = x / Math.max(1, bar.width - 1);
    const closenessT = 1 - distanceT;
    const color = colorFor(Math.pow(closenessT, gamma));
    for (let y = 0; y < bar.height; y++) {{
      const o = (y * bar.width + x) * 4;
      out.data[o] = color[0];
      out.data[o + 1] = color[1];
      out.data[o + 2] = color[2];
      out.data[o + 3] = 255;
    }}
  }}
  bctx.putImageData(out, 0, 0);
  const near = simValues ? `${{fmt(simMin)}} near` : "near / low distance";
  const far = simValues ? `${{fmt(simMax)}} far` : "far / high distance";
  document.getElementById("colorbarNear").textContent = near;
  document.getElementById("colorbarFar").textContent = far;
}}

function renderSimilarity() {{
  if (!simValues) return;
  const gamma = Number(document.getElementById("gamma").value);
  const lo = simMin;
  const hi = simMax;
  const out = new ImageData(IMG_W, IMG_H);
  for (let i = 0, o = 0; i < simValues.length; i++, o += 4) {{
    const sim = simValues[i];
    if (!Number.isFinite(sim)) {{
      out.data[o + 3] = 0;
      continue;
    }}
    const distanceT = Math.max(0, Math.min(1, (sim - lo) / Math.max(1e-12, hi - lo)));
    const closenessT = 1 - distanceT;
    const color = colorFor(Math.pow(closenessT, gamma));
    out.data[o] = color[0];
    out.data[o + 1] = color[1];
    out.data[o + 2] = color[2];
    out.data[o + 3] = 255;
  }}
  simCanvas = document.createElement("canvas");
  simCanvas.width = IMG_W;
  simCanvas.height = IMG_H;
  simCanvas.getContext("2d").putImageData(out, 0, 0);
  document.getElementById("window").textContent = `${{fmt(lo)}} to ${{fmt(hi)}}, bright = near, gamma=${{fmt(gamma)}}`;
  renderColorbar();
  if (mode === "sim") draw();
}}

function npyFloat32Blob(values, shape) {{
  const magic = new Uint8Array([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]);
  const version = new Uint8Array([1, 0]);
  let shapeText = shape.join(", ");
  if (shape.length === 1) shapeText += ",";
  let header = `{{'descr': '<f4', 'fortran_order': False, 'shape': (${{shapeText}}), }}`;
  const encoder = new TextEncoder();
  let headerLen = encoder.encode(header + "\\n").length;
  const preambleLen = magic.length + version.length + 2;
  const padLen = (16 - ((preambleLen + headerLen) % 16)) % 16;
  header = header + " ".repeat(padLen) + "\\n";
  const headerBytes = encoder.encode(header);
  const headerSize = new Uint8Array(2);
  headerSize[0] = headerBytes.length & 0xff;
  headerSize[1] = (headerBytes.length >> 8) & 0xff;
  const dataBytes = new Uint8Array(values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength));
  return new Blob([magic, version, headerSize, headerBytes, dataBytes], {{ type: "application/octet-stream" }});
}}

function downloadBlob(blob, filename) {{
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}}

function saveDisplayedNpy() {{
  if (mode !== "sim" || !simValues) {{
    document.getElementById("window").textContent = "click a pixel first, then save the distance map";
    return;
  }}
  const blob = npyFloat32Blob(simValues, [IMG_H, IMG_W]);
  const metric = document.getElementById("distanceMetric").value;
  const filename = `${{metric}}_distance_x${{selectedPx}}_y${{selectedPy}}.npy`;
  downloadBlob(blob, filename);
}}

function activeDistanceMetric() {{
  return document.getElementById("distanceMetric").value === "cosine" ? "cosine" : "euclidean";
}}

function distanceMetricLabel(metric) {{
  return metric === "cosine" ? "cosine distance" : "Euclidean distance";
}}

function computeSimilarity(px, py) {{
  if (!similarity || px < 0 || py < 0 || px >= IMG_W || py >= IMG_H) return;
  const metric = activeDistanceMetric();
  const seedIndex = (py * IMG_W + px) * SIMILARITY_CHANNELS;
  if (!finiteVector(similarity, seedIndex, SIMILARITY_CHANNELS)) return;

  let seedNorm2 = 0;
  let seedNorm = 0;
  if (metric === "cosine") {{
    for (let c = 0; c < SIMILARITY_CHANNELS; c++) {{
      const v = similarity[seedIndex + c];
      seedNorm2 += v * v;
    }}
    seedNorm = Math.sqrt(seedNorm2);
    if (!Number.isFinite(seedNorm) || seedNorm <= 1e-12) return;
  }}

  simValues = new Float32Array(IMG_W * IMG_H);
  simValues.fill(Number.NaN);
  let minDistance = Infinity;
  let maxDistance = -Infinity;
  for (let y = 0; y < IMG_H; y++) {{
    for (let x = 0; x < IMG_W; x++) {{
      const i = (y * IMG_W + x) * SIMILARITY_CHANNELS;
      if (!finiteVector(similarity, i, SIMILARITY_CHANNELS)) {{
        continue;
      }}
      let distance = 0;
      let dot = 0;
      let norm2 = 0;
      for (let c = 0; c < SIMILARITY_CHANNELS; c++) {{
        const sv = similarity[seedIndex + c];
        const vv = similarity[i + c];
        if (metric === "cosine") {{
          dot += sv * vv;
          norm2 += vv * vv;
        }} else {{
          const delta = vv - sv;
          distance += delta * delta;
        }}
      }}
      if (metric === "cosine") {{
        const norm = Math.sqrt(norm2);
        if (!Number.isFinite(norm) || norm <= 1e-12) {{
          continue;
        }}
        const cosineSimilarity = dot / (seedNorm * norm);
        distance = 1 - Math.max(-1, Math.min(1, cosineSimilarity));
      }} else {{
        distance = Math.sqrt(distance);
      }}
      if (!Number.isFinite(distance)) continue;
      minDistance = Math.min(minDistance, distance);
      maxDistance = Math.max(maxDistance, distance);
      simValues[y * IMG_W + x] = distance;
    }}
  }}
  if (!Number.isFinite(minDistance) || !Number.isFinite(maxDistance)) return;

  simMin = minDistance;
  simMax = maxDistance;
  selectedPx = px;
  selectedPy = py;
  renderSimilarity();
  mode = "sim";
  const preview = [];
  for (let c = 0; c < Math.min(6, SIMILARITY_CHANNELS); c++) preview.push(fmt(similarity[seedIndex + c]));
  const suffix = SIMILARITY_CHANNELS > 6 ? ", ..." : "";
  document.getElementById("mode").textContent = `${{SIMILARITY_LABEL}} ${{distanceMetricLabel(metric)}} map, ${{fmt(minDistance)}} to ${{fmt(maxDistance)}}`;
  document.getElementById("selected").textContent = `x=${{px}}, y=${{py}}, ${{SIMILARITY_LABEL}}[${{SIMILARITY_CHANNELS}}]=(${{preview.join(", ")}}${{suffix}})`;
  draw();
}}

function recomputeSelectedDistance() {{
  if (selectedPx === null || selectedPy === null) return;
  computeSimilarity(selectedPx, selectedPy);
}}

function showRgb() {{
  mode = "rgb";
  document.getElementById("mode").textContent = `${{DISPLAY_LABEL}} RGB projection`;
  document.getElementById("window").textContent = "-";
  renderColorbar();
  draw();
}}

function toggleFlipY() {{
  flipY = !flipY;
  document.getElementById("flipY").textContent = flipY ? "Unflip Display Y" : "Flip Display Y";
  draw();
}}

canvas.addEventListener("wheel", (event) => {{
  event.preventDefault();
  const before = canvasToPixel(event.clientX, event.clientY);
  const factor = event.deltaY < 0 ? 1.18 : 1 / 1.18;
  scale = Math.min(80, Math.max(0.02, scale * factor));
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const mx = (event.clientX - rect.left) * dpr;
  const my = (event.clientY - rect.top) * dpr;
  offsetX = mx - before.px * scale;
  offsetY = (flipY ? canvas.height - my : my) - before.py * scale;
  draw();
}}, {{ passive: false }});

canvas.addEventListener("pointerdown", (event) => {{
  dragging = true;
  moved = false;
  downX = event.clientX;
  downY = event.clientY;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.classList.add("dragging");
  canvas.setPointerCapture(event.pointerId);
}});
canvas.addEventListener("pointermove", (event) => {{
  const pix = canvasToPixel(event.clientX, event.clientY);
  document.getElementById("cursor").textContent =
    pix.px >= 0 && pix.py >= 0 && pix.px < IMG_W && pix.py < IMG_H
      ? `x=${{pix.px}}, y=${{pix.py}}`
      : "-";
  if (!dragging) return;
  const dpr = window.devicePixelRatio || 1;
  if (Math.hypot(event.clientX - downX, event.clientY - downY) > 3) moved = true;
  offsetX += (event.clientX - lastX) * dpr;
  offsetY += (event.clientY - lastY) * dpr;
  lastX = event.clientX;
  lastY = event.clientY;
  draw();
}});
canvas.addEventListener("pointerup", (event) => {{
  if (!moved) {{
    const pix = canvasToPixel(event.clientX, event.clientY);
    computeSimilarity(pix.px, pix.py);
  }}
  dragging = false;
  canvas.classList.remove("dragging");
}});

document.getElementById("fit").addEventListener("click", fit);
document.getElementById("one").addEventListener("click", oneToOne);
document.getElementById("rgb").addEventListener("click", showRgb);
document.getElementById("flipY").addEventListener("click", toggleFlipY);
document.getElementById("saveNpy").addEventListener("click", saveDisplayedNpy);
document.getElementById("gamma").addEventListener("input", () => {{ renderSimilarity(); renderColorbar(); }});
document.getElementById("colormap").addEventListener("change", () => {{ renderSimilarity(); renderColorbar(); }});
document.getElementById("distanceMetric").addEventListener("change", recomputeSelectedDistance);
window.addEventListener("resize", resize);

document.getElementById("shape").textContent = STATS.shape.join(" x ");
document.getElementById("simshape").textContent = STATS.similarity_shape.join(" x ");
document.getElementById("valid").textContent = `${{STATS.valid_pixels}} / ${{STATS.total_pixels}}`;
for (let i = 0; i < 3; i++) {{
  const ch = STATS.channels[i] || {{}};
  document.getElementById(`ch${{i}}`).textContent =
    `raw ${{fmt(ch.min)}} to ${{fmt(ch.max)}}, scaled ${{fmt(ch.lo_pct)}} to ${{fmt(ch.hi_pct)}}`;
}}

xyz = decodeFloat32(DISPLAY_B64);
similarity = decodeFloat32(SIMILARITY_B64);
renderColorbar();
resize(); fit();
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Embedding .npy, shape (C,H,W) or (H,W,C)")
    parser.add_argument("--output", type=Path, default=None, help="Output dashboard .html path; defaults to input with .html")
    parser.add_argument("--display-label", default="embedding", help="Human label for the embedding")
    parser.add_argument("--lo-pct", type=float, default=1.0, help="Lower percentile for RGB display scaling")
    parser.add_argument("--hi-pct", type=float, default=99.0, help="Upper percentile for RGB display scaling")
    args = parser.parse_args()

    out = build_html(
        args.input,
        args.output,
        args.lo_pct,
        args.hi_pct,
        display_label=args.display_label,
    )
    print(f"html_saved={out}")
    print(f"open {out}")


if __name__ == "__main__":
    main()
