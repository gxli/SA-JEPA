from __future__ import annotations

import argparse
import base64
import csv
import html as html_lib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
from src.utils.viz import _compute_pca_3d, _compute_umap_nd, _preprocess_latents_for_umap, _target_region_mask_from_outputs


DASHBOARD_VERSION = "production-diagnostics-v20-card-local-controls"
CONTROL_SCRIPT_SENTINEL = "window.JEPADashboardControls"
DASHBOARD_COMPUTE_UMAP = os.environ.get("DASHBOARD_COMPUTE_UMAP", "").strip().lower() in {"1", "true", "yes", "on"}
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(ROOT_DIR, "scripts")

DASH_DATA_REQUIRED = {
    "dashboard_version",
    "rgb_render_source",
    "dashboard_config_source",
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
    "masked_pred_pca3d",
    "masked_pred_umap3d",
    "masked_pred_pca_rgb",
    "masked_pred_pca_rgb_flat",
    "masked_pred_umap_rgb",
    "masked_pred_umap_rgb_flat",
    "gt_pca3d",
    "gt_umap3d",
    "gt_pca_rgb",
    "gt_pca_rgb_flat",
    "gt_umap_rgb",
    "gt_umap_rgb_flat",
    "pyramid_mask_stack",
}

SCALE_PROBE_KEYS = {
    "scale_probe_sensitivity_maps",
    "scale_probe_scale_only_sim_maps",
    "scale_probe_winner_map",
    "scale_probe_names",
}


def _session_config_candidates(session_dir: str) -> list[str]:
    name = os.path.basename(os.path.abspath(session_dir))
    env_cfg_dir = os.environ.get("SESSION_DASH_CONFIG_DIR", "").strip()
    candidates = []
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
    candidates.extend([
        os.path.join(ROOT_DIR, "configs", "experiments", f"{name}.json"),
        os.path.join(ROOT_DIR, "configs", f"{name}.json"),
    ])
    return [os.path.abspath(path) for path in candidates]


def _read_config_file(path: str, seen: set[str] | None = None) -> dict[str, Any]:
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
        base_cfg = _read_config_file(base_path, seen)
        cfg = _deep_merge_dict(base_cfg, cfg)
    seen.remove(abs_path)
    return cfg


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _find_session_config(session_dir: str) -> str | None:
    for path in _session_config_candidates(session_dir):
        if os.path.exists(path):
            return path
    return None


def _find_readable_session_config(session_dir: str) -> tuple[dict[str, Any], str | None]:
    for path in _session_config_candidates(session_dir):
        if not os.path.exists(path):
            continue
        try:
            cfg = _read_config_file(path)
            if isinstance(cfg, dict):
                return cfg, path
            print(f"dashboard_config_read_failed={path} reason=not_object")
        except Exception as e:
            print(f"dashboard_config_read_failed={path} reason={type(e).__name__}: {e}")
            continue
    return {}, None


def _find_readable_session_config_path(session_dir: str) -> str | None:
    _cfg, cfg_path = _find_readable_session_config(session_dir)
    return cfg_path


def _dashboard_html_is_current(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return False
    return (
        f'name="jepa-dashboard-version" content="{DASHBOARD_VERSION}"' in text
        and CONTROL_SCRIPT_SENTINEL in text
    )


def _load_session_config(session_dir: str) -> tuple[dict[str, Any], str | None]:
    return _find_readable_session_config(session_dir)


def _generate_masking_diagnostic_for_dashboard(session_dir: str) -> str | None:
    cfg_path = _find_readable_session_config_path(session_dir)
    if cfg_path is None:
        print(f"dashboard_masking_demo_skip={session_dir} reason=no_config")
        return None

    out_html = os.path.join(session_dir, "masking_demo_plotly_dashboard.html")
    script_path = os.path.join(SCRIPT_DIR, "generate_masking_diagnostics_plotly.py")
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
        proc = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception as e:
        print(f"dashboard_masking_demo_failed={session_dir} reason={type(e).__name__}: {e}")
        return None

    for line in proc.stdout.splitlines():
        if line.strip():
            print(f"dashboard_masking_demo={line.strip()}")
    for line in proc.stderr.splitlines():
        if line.strip():
            print(f"dashboard_masking_demo_stderr={line.strip()}")
    return out_html


def _verbose_artifact_report(session_dir: str) -> list[str]:
    missing: list[str] = []
    results_dir = os.path.join(session_dir, "results")
    if not os.path.isdir(results_dir):
        missing.append(f"missing_dir: {results_dir}")
        return missing
    # Core branch artifacts expected from src/train.py save_inference_dashboard().
    for branch in ("predict", "target"):
        if not _has_required_branch_artifacts(results_dir, branch):
            missing.append(f"missing_branch_artifacts[{branch}]: map or legacy coordinate files absent")
    # Context branch is optional: if absent, we fallback to predict branch.
    if not _has_required_branch_artifacts(results_dir, "context"):
        missing.append(
            "optional_context_missing: one or more context_* artifacts missing; "
            "dashboard will fallback to predict_* for context panels"
        )
    return missing


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _load_array(path: str) -> np.ndarray:
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            return np.asarray(loaded["arr"])
        finally:
            loaded.close()
    return np.asarray(loaded)


def _prefer_npz(path: str) -> str:
    npz_path = os.path.splitext(path)[0] + ".npz"
    return npz_path if os.path.exists(npz_path) else path


def _find_scale_probe_artifacts(session_dir: str) -> tuple[str | None, str | None]:
    run_name = os.path.basename(os.path.abspath(session_dir))
    preferred_pt = os.path.join(session_dir, f"{run_name}_scale_response.pt")
    preferred_json = os.path.join(session_dir, f"{run_name}_report.json")
    if os.path.exists(preferred_pt):
        return preferred_pt, preferred_json if os.path.exists(preferred_json) else None

    pts = sorted(
        os.path.join(session_dir, fn)
        for fn in os.listdir(session_dir)
        if fn.endswith("_scale_response.pt")
    )
    if not pts:
        return None, None
    pt_path = pts[0]
    stem = os.path.basename(pt_path)[: -len("_scale_response.pt")]
    report_path = os.path.join(session_dir, f"{stem}_report.json")
    return pt_path, report_path if os.path.exists(report_path) else None


def _load_scale_probe_dash_data(session_dir: str) -> dict[str, np.ndarray]:
    pt_path, report_path = _find_scale_probe_artifacts(session_dir)
    if pt_path is None:
        return {}
    try:
        probe = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        probe = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        print(f"dashboard_scale_probe_skip={session_dir} reason={type(e).__name__}: {e}")
        return {}

    sensitivity = _to_np(probe.get("sensitivity_maps", np.asarray([]))).astype(np.float32)
    if sensitivity.ndim != 4 or sensitivity.shape[0] <= 0 or sensitivity.shape[1] <= 0:
        print(f"dashboard_scale_probe_skip={session_dir} reason=malformed_sensitivity shape={sensitivity.shape}")
        return {}
    scale_only = _to_np(probe.get("scale_only_sim_maps", np.zeros_like(sensitivity))).astype(np.float32)
    if scale_only.shape != sensitivity.shape:
        scale_only = np.zeros_like(sensitivity, dtype=np.float32)
    winner = _to_np(probe.get("winner_map", sensitivity[0].argmax(axis=0))).astype(np.float32)
    if winner.ndim != 2:
        winner = sensitivity[0].argmax(axis=0).astype(np.float32)
    pred_sensitivity = _to_np(probe.get("pred_sensitivity_maps", np.asarray([]))).astype(np.float32)
    if pred_sensitivity.shape != sensitivity.shape:
        pred_sensitivity = np.asarray([], dtype=np.float32)
    probe_input = _to_np(probe.get("input_map", np.asarray([]))).astype(np.float32)
    if probe_input.ndim != 2:
        x_pyr = _to_np(probe.get("x_pyr", np.asarray([]))).astype(np.float32)
        if x_pyr.ndim == 4 and x_pyr.shape[0] > 0:
            probe_input = x_pyr[0].sum(axis=0).astype(np.float32)
        else:
            probe_input = np.asarray([], dtype=np.float32)

    scale_names: list[str] = []
    report = {}
    if report_path is not None:
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            scale_names = [str(v) for v in report.get("scale_names", [])]
        except Exception:
            report = {}
    if len(scale_names) != int(sensitivity.shape[1]):
        scale_names = [f"scale_{i}" for i in range(int(sensitivity.shape[1]))]

    def _target_hw_from_report() -> tuple[int, int] | None:
        shape = report.get("input_shape") if isinstance(report, dict) else None
        if isinstance(shape, (list, tuple)) and len(shape) >= 4:
            h0, w0 = int(shape[-2]), int(shape[-1])
            if h0 > 0 and w0 > 0:
                return h0, w0
        return None

    target_hw = tuple(int(v) for v in probe_input.shape[-2:]) if probe_input.ndim == 2 else _target_hw_from_report()

    def _coerce_maps_to_hw(arr: np.ndarray, hw: tuple[int, int] | None, *, mode: str = "bilinear") -> np.ndarray:
        vals = np.asarray(arr, dtype=np.float32)
        if hw is None or vals.size == 0 or vals.ndim < 2:
            return vals
        h0, w0 = int(hw[0]), int(hw[1])
        if vals.shape[-2:] == (h0, w0):
            return vals
        if vals.shape[-2] * vals.shape[-1] == h0 * w0:
            return vals.reshape(*vals.shape[:-2], h0, w0).astype(np.float32, copy=False)
        leading = vals.shape[:-2]
        flat = vals.reshape(-1, 1, vals.shape[-2], vals.shape[-1])
        kwargs = {"size": (h0, w0), "mode": mode}
        if mode != "nearest":
            kwargs["align_corners"] = False
        resized = torch.nn.functional.interpolate(torch.from_numpy(flat), **kwargs).numpy()
        return resized.reshape(*leading, h0, w0).astype(np.float32, copy=False)

    sensitivity = _coerce_maps_to_hw(sensitivity, target_hw)
    scale_only = _coerce_maps_to_hw(scale_only, target_hw)
    if pred_sensitivity.size > 0:
        pred_sensitivity = _coerce_maps_to_hw(pred_sensitivity, target_hw)
    winner = _coerce_maps_to_hw(winner, target_hw, mode="nearest")
    if winner.ndim != 2:
        winner = sensitivity[0].argmax(axis=0).astype(np.float32)

    sens_global = sensitivity.mean(axis=(0, 2, 3))
    sens_frac = sens_global / max(float(np.sum(sens_global)), 1e-12)
    pred_global = (
        pred_sensitivity.mean(axis=(0, 2, 3))
        if pred_sensitivity.size > 0
        else np.asarray([], dtype=np.float32)
    )
    pred_frac = (
        pred_global / max(float(np.sum(pred_global)), 1e-12)
        if pred_global.size > 0
        else np.asarray([], dtype=np.float32)
    )
    sim_global = scale_only.mean(axis=(0, 2, 3))

    return {
        "scale_probe_sensitivity_maps": sensitivity[0].astype(np.float32),
        "scale_probe_scale_only_sim_maps": scale_only[0].astype(np.float32),
        "scale_probe_winner_map": winner.astype(np.float32),
        "scale_probe_pred_sensitivity_maps": pred_sensitivity[0].astype(np.float32) if pred_sensitivity.size > 0 else np.asarray([], dtype=np.float32),
        "scale_probe_sensitivity_mean": sens_global.astype(np.float32),
        "scale_probe_sensitivity_fraction": sens_frac.astype(np.float32),
        "scale_probe_pred_sensitivity_mean": pred_global.astype(np.float32),
        "scale_probe_pred_sensitivity_fraction": pred_frac.astype(np.float32),
        "scale_probe_scale_only_similarity": sim_global.astype(np.float32),
        "scale_probe_names": np.array(scale_names, dtype=str),
        "scale_probe_source": np.array([os.path.basename(pt_path)], dtype=str),
        "scale_probe_report_json": np.array([json.dumps(report, sort_keys=True)], dtype=str),
        "scale_probe_input_map": probe_input.astype(np.float32) if probe_input.ndim == 2 else np.asarray([], dtype=np.float32),
    }


def _has_required_branch_artifacts(results_dir: str, branch: str) -> bool:
    generic_pca = "pca_xyz.npy" if branch == "predict" else None
    generic_umap = "umap_xyz.npy" if branch == "predict" else None
    candidates = [
        (f"{branch}_spatial_shape.npy", None),
        (f"{branch}_pca_xyz.npy", generic_pca or f"volumetric_pca_xyz.npy"),
        (f"{branch}_umap_xyz.npy", generic_umap or f"volumetric_umap_xyz.npy"),
    ]
    has_shape = False
    has_pca = False
    has_umap = False
    for name, fallback in candidates:
        if os.path.exists(os.path.join(results_dir, name)):
            if "spatial_shape" in name:
                has_shape = True
            elif "pca" in name:
                has_pca = True
            elif "umap" in name:
                has_umap = True
        elif fallback and os.path.exists(os.path.join(results_dir, fallback)):
            if "pca" in fallback:
                has_pca = True
            elif "umap" in fallback:
                has_umap = True
    has_umap_legacy = all(
        os.path.exists(os.path.join(results_dir, f"{branch}_umap_{axis}.npy"))
        for axis in ("x", "y", "z")
    )
    # 3D volumetric sessions use volumetric_* naming; spatial_shape is optional.
    return has_pca and (has_umap or has_umap_legacy)


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
        if not _has_required_branch_artifacts(results_dir, branch):
            missing.append(f"missing_branch_artifacts:{branch}")
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
        if arr.ndim == 2 and arr.shape == (shape[1], shape[0]):
            arr = arr.T
            return np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    return None


def _canonicalize_cube_hw(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return an S,H,W cube aligned to the dashboard image shape when possible."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        return np.zeros((1, shape[0], shape[1]), dtype=np.float32)
    if arr.shape[-2:] == shape:
        out = arr
    elif arr.shape[-2:] == (shape[1], shape[0]):
        out = arr.transpose(0, 2, 1)
    else:
        out = np.zeros((arr.shape[0], shape[0], shape[1]), dtype=np.float32)
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32, copy=False)


def _display_scalar_from_batched_tensor(x: Any) -> np.ndarray:
    arr = _to_np(x).astype(np.float32)
    if arr.ndim == 5:
        # B,C,D,H,W. Dashboard displays a scalar field; multi-channel CDD
        # inputs are reconstructed by summing channels before taking a slice.
        arr = arr[0]
        if arr.shape[0] > 1:
            arr = arr.sum(axis=0)
        else:
            arr = arr[0]
        return arr[arr.shape[0] // 2] if arr.ndim == 3 else arr
    if arr.ndim == 4:
        # B,C,H,W or B,D,H,W. Treat C as channels when there are multiple maps.
        arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] > 1:
            arr = arr.sum(axis=0)
        elif arr.ndim == 3:
            arr = arr[0]
        return arr
    if arr.ndim == 3:
        return arr[arr.shape[0] // 2]
    if arr.ndim == 2:
        return arr
    return np.zeros((1, 1), dtype=np.float32)


def _rgb_from_xyz(
    xyz: np.ndarray,
    h: int,
    w: int,
    bright_top: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    # Compute percentiles only on finite (non-NaN) entries so invalid-region
    # sentinels do not contaminate the colour range.
    fin = np.isfinite(xyz).all(axis=1)
    if fin.any():
        lo = np.percentile(xyz[fin], 1.0, axis=0)
        hi = np.percentile(xyz[fin], 99.0, axis=0)
    else:
        lo = np.zeros(xyz.shape[1], dtype=np.float64)
        hi = np.ones(xyz.shape[1], dtype=np.float64)
    den = np.clip(hi - lo, 1e-8, None)
    clipped = np.clip((xyz - lo) / den, 0.0, 1.0)
    if not bright_top:
        clipped = 1.0 - clipped
    clipped[~fin] = 0.0
    rgb_flat = np.clip(np.round(clipped * 255.0), 0, 255).astype(np.uint8)
    alpha = fin.astype(np.uint8) * 255
    rgb_flat = np.column_stack([rgb_flat, alpha])
    rgb = rgb_flat.reshape(h, w, 4)
    return rgb, rgb_flat


def _fill_invalid_rgb_from_nearest(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid display pixels from nearest valid row/column, preserving shape."""
    out = np.asarray(rgb, dtype=np.uint8).copy()
    mask = np.asarray(valid, dtype=bool)
    if out.ndim != 3 or out.shape[:2] != mask.shape or not mask.any() or mask.all():
        return out
    yy, xx = np.nonzero(mask)
    y0, y1 = int(yy.min()), int(yy.max())
    x0, x1 = int(xx.min()), int(xx.max())
    bad_y, bad_x = np.nonzero(~mask)
    if bad_y.size == 0:
        return out
    # Only fill interior holes (within the valid bounding box).
    # Border NaN pixels stay unfilled → render as gray, keeping FOV border visible.
    inside = (bad_y >= y0) & (bad_y <= y1) & (bad_x >= x0) & (bad_x <= x1)
    bad_y = bad_y[inside]
    bad_x = bad_x[inside]
    if bad_y.size == 0:
        return out
    src_y = np.clip(bad_y, y0, y1)
    src_x = np.clip(bad_x, x0, x1)
    fallback = out[src_y, src_x]
    still_bad = ~mask[src_y, src_x]
    if np.any(still_bad):
        # Rare non-rectangular holes: use the median valid colour rather than
        # turning missing display pixels into a fake black structure.
        fallback[still_bad] = np.median(out[mask], axis=0).astype(np.uint8)
    out[bad_y, bad_x] = fallback
    return out


def _xyz_from_feature_map(feat: np.ndarray) -> np.ndarray:
    """Fallback embedding: flatten CHW feature map to N x 3 via first channels."""
    arr = np.asarray(feat, dtype=np.float32)
    if arr.ndim == 5:
        # B,C,D,H,W -> C,H,W center slice. These are latent maps, not CDD.
        arr = arr[0]
        arr = arr[:, arr.shape[1] // 2]
    elif arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return np.zeros((0, 3), dtype=np.float32)
    c, h, w = arr.shape
    flat = np.transpose(arr, (1, 2, 0)).reshape(h * w, c)
    if c >= 3:
        xyz = flat[:, :3]
    elif c == 2:
        xyz = np.concatenate([flat, np.zeros((flat.shape[0], 1), dtype=np.float32)], axis=1)
    elif c == 1:
        xyz = np.concatenate([flat, flat, flat], axis=1)
    else:
        xyz = np.zeros((h * w, 3), dtype=np.float32)
    return xyz.astype(np.float32, copy=False)


def _latent_vectors_from_feature_map(feat: np.ndarray) -> np.ndarray:
    arr = np.asarray(feat, dtype=np.float32)
    if arr.ndim == 5:
        arr = arr[0]
        arr = arr[:, arr.shape[1] // 2]
    elif arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return np.zeros((0, 0), dtype=np.float32)
    c, h, w = arr.shape
    return np.transpose(arr, (1, 2, 0)).reshape(h * w, c).astype(np.float32, copy=False)


def _apply_latent_border_nan(z: np.ndarray, h: int, w: int, border_px: int) -> np.ndarray:
    b = int(max(0, min(int(border_px), h // 2, w // 2)))
    arr = np.asarray(z, dtype=np.float32)
    if b <= 0 or arr.ndim != 2 or arr.shape[0] != h * w:
        return arr
    out = arr.reshape(h, w, arr.shape[1]).copy()
    out[:b, :, :] = np.nan
    out[h - b :, :, :] = np.nan
    out[:, :b, :] = np.nan
    out[:, w - b :, :] = np.nan
    return out.reshape(h * w, arr.shape[1])


def _expected_full_frame_hw(session_dir: str) -> tuple[int, int] | None:
    profile_path = os.path.join(session_dir, "data_profile.json")
    if not os.path.exists(profile_path):
        return None
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        files = profile.get("files", []) if isinstance(profile, dict) else []
        if not files:
            return None
        shape = files[0].get("shape")
        if not isinstance(shape, list) or len(shape) < 2:
            return None
        return int(shape[-2]), int(shape[-1])
    except Exception:
        return None


def _assert_not_stale_cropped_inference(session_dir: str, observed_hw: tuple[int, int]) -> None:
    cfg, _cfg_path = _load_session_config(session_dir)
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    max_diag = train_cfg.get("inference_max_diagnostic_size") if isinstance(train_cfg, dict) else None
    try:
        full_frame_requested = max_diag is None or int(max_diag) <= 0
    except (TypeError, ValueError):
        full_frame_requested = str(max_diag).strip().lower() in ("", "none", "false", "full")
    if not full_frame_requested:
        return
    expected_hw = _expected_full_frame_hw(session_dir)
    if expected_hw is None or tuple(observed_hw) == tuple(expected_hw):
        return
    raise RuntimeError(
        f"{session_dir}: stale/cropped inference_outputs.pt has image shape {tuple(observed_hw)} "
        f"but config requests full-frame inference and data_profile shape is {tuple(expected_hw)}. "
        "Delete inference_outputs.pt and dash_data.npz, then rerun with --recompute-inference."
    )


def _apply_xyz_border_nan(xyz: np.ndarray, h: int, w: int, border_px: int) -> np.ndarray:
    b = int(max(0, min(int(border_px), h // 2, w // 2)))
    arr = np.asarray(xyz, dtype=np.float32)
    if b <= 0 or arr.shape != (h * w, 3):
        return arr
    out = arr.reshape(h, w, 3).copy()
    out[:b, :, :] = np.nan
    out[h - b :, :, :] = np.nan
    out[:, :b, :] = np.nan
    out[:, w - b :, :] = np.nan
    return out.reshape(h * w, 3)


def _apply_image_border_nan(arr: np.ndarray, border_px: int) -> np.ndarray:
    img = np.asarray(arr, dtype=np.float32)
    if img.ndim != 2:
        return img
    h, w = img.shape
    b = int(max(0, min(int(border_px), h // 2, w // 2)))
    if b <= 0:
        return img
    out = img.copy()
    out[:b, :] = np.nan
    out[h - b :, :] = np.nan
    out[:, :b] = np.nan
    out[:, w - b :] = np.nan
    return out


def _summary_discard_margin(session_dir: str) -> int:
    summary_path = os.path.join(session_dir, "jepa_energy_summary.json")
    if not os.path.exists(summary_path):
        return 0
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return int(max(0, summary.get("inference_discard_margin") or 0))
    except Exception:
        return 0


def _viz_crop_border_from_config(session_dir: str) -> tuple[bool, int, str | None]:
    cfg, cfg_path = _load_session_config(session_dir)
    summary_margin = _summary_discard_margin(session_dir)
    if not cfg_path:
        return summary_margin > 0, summary_margin, None
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    source_cfg = model_cfg if isinstance(model_cfg, dict) and "viz_crop_border" in model_cfg else train_cfg
    if not isinstance(source_cfg, dict):
        return summary_margin > 0, summary_margin, cfg_path
    enabled = bool(source_cfg.get("viz_crop_border", False))
    if not enabled:
        return summary_margin > 0, summary_margin, cfg_path
    value = source_cfg.get("viz_crop_border_px", source_cfg.get("inference_discard_margin", "auto"))
    try:
        if value is None or str(value).strip().lower() == "auto":
            return True, max(summary_margin, _encoder_fov_border_from_config_dict(cfg)), cfg_path
        return True, max(summary_margin, int(max(0, value or 0))), cfg_path
    except (TypeError, ValueError):
        return summary_margin > 0, summary_margin, cfg_path


def _encoder_fov_border_from_config_dict(cfg: dict) -> int:
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


def compute_dash_data(session_dir: str, overwrite: bool = False) -> str:
    out_npz = os.path.join(session_dir, "dash_data.npz")
    if os.path.exists(out_npz) and not overwrite:
        try:
            existing = np.load(out_npz)
            missing = sorted(DASH_DATA_REQUIRED.difference(existing.files))
            scale_pt, _ = _find_scale_probe_artifacts(session_dir)
            if scale_pt is not None:
                missing.extend(sorted(SCALE_PROBE_KEYS.difference(existing.files)))
            if "dashboard_version" in existing.files:
                version_arr = np.asarray(existing["dashboard_version"]).reshape(-1)
                version_str = str(version_arr[0]) if version_arr.size else ""
                if version_str != DASHBOARD_VERSION:
                    missing.append("dashboard_version")
            existing.close()
            npz_mtime = os.path.getmtime(out_npz)
            stale_inputs = []
            for dep_name in (
                "inference_outputs.pt",
                "jepa_energy_summary.json",
                "metrics.csv",
                "loss_weights.json",
                "rank_diagnostics.json",
            ):
                dep_path = os.path.join(session_dir, dep_name)
                if os.path.exists(dep_path) and os.path.getmtime(dep_path) > npz_mtime:
                    stale_inputs.append(dep_name)
            cfg_path = _find_readable_session_config_path(session_dir)
            if cfg_path and os.path.exists(cfg_path) and os.path.getmtime(cfg_path) > npz_mtime:
                stale_inputs.append(os.path.basename(cfg_path))
            if stale_inputs:
                print(f"dash_data_stale_recompute={out_npz} newer={','.join(stale_inputs)}")
                missing.extend(stale_inputs)
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
    ctx_raw = outputs.get("x_context", outputs.get("x_context_raw", x_clean))
    orig = _display_scalar_from_batched_tensor(x_clean)
    blurred = _display_scalar_from_batched_tensor(ctx_raw)
    h, w = orig.shape
    _assert_not_stale_cropped_inference(session_dir, (h, w))

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
            if tloc.shape[-1] >= 3:
                yy, xx = int(tloc[bi, ki, -2]), int(tloc[bi, ki, -1])
            else:
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
        energy_npy = _prefer_npz(os.path.join(session_dir, "target_energy_map.npy"))
        if os.path.exists(energy_npy):
            energy_map = np.asarray(_load_array(energy_npy), dtype=np.float32)
            if energy_map.ndim == 4:
                energy_map = energy_map[0, 0]
            elif energy_map.ndim == 3:
                energy_map = energy_map[0]
            if energy_map.shape != (h, w):
                energy_map = np.zeros((h, w), dtype=np.float32)
        else:
            energy_map = np.zeros((h, w), dtype=np.float32)

    visit_path = _prefer_npz(os.path.join(session_dir, "visited_target_frequency_canonical.npy"))
    if not os.path.exists(visit_path):
        visit_path = _prefer_npz(os.path.join(session_dir, "visited_target_frequency.npy"))
    # Auto-detect tile visit map from tiled inference
    tile_visit_path = _prefer_npz(os.path.join(session_dir, "tile_visit_map.npy"))
    visit_heatmap_kind = "Visit Frequency Heatmap"
    if os.path.exists(tile_visit_path):
        visit_heatmap = np.asarray(_load_array(tile_visit_path), dtype=np.float32)
        if visit_heatmap.shape != (h, w):
            visit_heatmap = np.zeros((h, w), dtype=np.float32)
        else:
            visit_heatmap_kind = "Tile Coverage Heatmap"
    elif os.path.exists(visit_path):
        visit_heatmap = np.asarray(_load_array(visit_path), dtype=np.float32)
        if visit_heatmap.shape != (h, w):
            visit_heatmap = np.zeros((h, w), dtype=np.float32)
    else:
        visit_heatmap = np.zeros((h, w), dtype=np.float32)

    # Dashboard-only pyramid mask stack (S,H,W), reconstructed from inference
    # tensors/artifacts. This is not written as a standalone debug file.
    pyramid_mask_stack = None
    for mask_name in ("dip_field_per_channel", "pyramid_mask_token"):
        mask_path = _prefer_npz(os.path.join(session_dir, f"{mask_name}.npy"))
        if os.path.exists(mask_path):
            arr = np.asarray(_load_array(mask_path), dtype=np.float32)
            if arr.ndim == 4 and arr.shape[0] > 0:
                pyramid_mask_stack = arr[0]
                break
    if pyramid_mask_stack is None:
        tok = outputs.get("dip_field_per_channel", outputs.get("pyramid_mask_token"))
        if tok is not None:
            tok = _to_np(tok).astype(np.float32)
            if tok.ndim == 4 and tok.shape[0] > 0:
                pyramid_mask_stack = tok[0]
    if pyramid_mask_stack is None:
        cdd_o = outputs.get("cdd_channels_orig")
        cdd_m = outputs.get("cdd_channels_masked")
        if cdd_o is not None and cdd_m is not None:
            orig_arr = _to_np(cdd_o).astype(np.float32)
            masked_arr = _to_np(cdd_m).astype(np.float32)
            if orig_arr.ndim == 4 and masked_arr.shape == orig_arr.shape and orig_arr.shape[0] > 0:
                pyramid_mask_stack = (np.abs(orig_arr[0] - masked_arr[0]) > 1e-8).astype(np.float32)
    if pyramid_mask_stack is None:
        cdd_o_path = _prefer_npz(os.path.join(session_dir, "cdd_channels_orig.npy"))
        cdd_m_path = _prefer_npz(os.path.join(session_dir, "cdd_channels_masked.npy"))
        if os.path.exists(cdd_o_path) and os.path.exists(cdd_m_path):
            orig_arr = np.asarray(_load_array(cdd_o_path), dtype=np.float32)
            masked_arr = np.asarray(_load_array(cdd_m_path), dtype=np.float32)
            if orig_arr.ndim == 4 and masked_arr.shape == orig_arr.shape and orig_arr.shape[0] > 0:
                pyramid_mask_stack = (np.abs(orig_arr[0] - masked_arr[0]) > 1e-8).astype(np.float32)
    if pyramid_mask_stack is None:
        tok = outputs.get("mask_cube")
        if tok is not None:
            tok = _to_np(tok).astype(np.float32)
            if tok.ndim == 5 and tok.shape[0] > 0 and tok.shape[1] > 0:
                pyramid_mask_stack = tok[0, 0]
    if pyramid_mask_stack is None:
        pyramid_mask_stack = np.zeros((1, h, w), dtype=np.float32)
    else:
        pyramid_mask_stack = _canonicalize_cube_hw(pyramid_mask_stack, (h, w))

    # Load precomputed PCA/UMAP artifacts saved by training-time pipeline.
    results_dir = os.path.join(session_dir, "results")
    has_results_dir = os.path.isdir(results_dir)
    has_predict_branch = has_results_dir and _has_required_branch_artifacts(results_dir, "predict")
    has_target_branch = has_results_dir and _has_required_branch_artifacts(results_dir, "target")
    fallback_mode = not (has_predict_branch and has_target_branch)
    inference_summary = {}
    summary_path = os.path.join(session_dir, "jepa_energy_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                inference_summary = json.load(f)
        except Exception:
            inference_summary = {}
    if has_results_dir:
        verbose_missing = _verbose_artifact_report(session_dir)
        for line in verbose_missing:
            print(f"dashboard_artifact_check={line}")
    if fallback_mode:
        print(
            f"dashboard_fallback_mode={session_dir} "
            f"reason=missing_branch_embeddings predict={int(has_predict_branch)} target={int(has_target_branch)}"
        )

    pred_map = outputs.get("pred_map")
    if pred_map is None:
        raise RuntimeError(f"{session_dir}: inference outputs missing pred_map")
    h_lat, w_lat = int(pred_map.shape[-2]), int(pred_map.shape[-1])

    dashboard_config_source = _find_readable_session_config_path(session_dir)

    def _chw_or_n3_to_xyz(arr: np.ndarray, hh: int, ww: int, path: str) -> np.ndarray:
        xyz = np.asarray(arr, dtype=np.float32)
        if xyz.ndim == 3:
            if xyz.shape[0] != 3 or xyz.shape[1:] != (hh, ww):
                raise RuntimeError(
                    f"{session_dir}: malformed map artifact {path} shape={xyz.shape}, "
                    f"expected (3,{hh},{ww})"
                )
            return np.transpose(xyz, (1, 2, 0)).reshape(hh * ww, 3).astype(np.float32)
        if xyz.ndim == 2 and xyz.shape[1:] == (3,):
            # Accept any N×3 shape (volumetric artifacts may differ from hh*ww).
            return xyz.astype(np.float32)
        if xyz.ndim == 2 and xyz.shape == (hh * ww, 3):
            return xyz.astype(np.float32)
        raise RuntimeError(
            f"{session_dir}: malformed embedding artifact {path} shape={xyz.shape}, "
            f"expected (3,{hh},{ww}) or (N,3)"
        )

    def _resolve_artifact_path(base_name: str, volumetric_name: str, generic_name: str | None = None) -> str | None:
        path = os.path.join(results_dir, base_name)
        if os.path.exists(path):
            return path
        if generic_name:
            gpath = os.path.join(results_dir, generic_name)
            if os.path.exists(gpath):
                return gpath
        vpath = os.path.join(results_dir, volumetric_name)
        if os.path.exists(vpath):
            return vpath
        return None

    def _load_xyz_triplet(prefix: str, kind: str, hh: int, ww: int) -> np.ndarray:
        if kind == "pca":
            path = _resolve_artifact_path(
                f"{prefix}_pca_xyz.npy",
                "volumetric_pca_xyz.npy",
                "pca_xyz.npy" if prefix == "predict" else None,
            )
            if not path:
                raise RuntimeError(
                    f"{session_dir}: missing required PCA artifact "
                    f"(results/{prefix}_pca_xyz.npy or results/volumetric_pca_xyz.npy)\n"
                    "hint: run training/inference to generate PCA artifacts"
                )
            return _chw_or_n3_to_xyz(np.load(path), hh, ww, path)
        xyz_path = _resolve_artifact_path(
            f"{prefix}_umap_xyz.npy",
            "volumetric_umap_xyz.npy",
            "umap_xyz.npy" if prefix == "predict" else None,
        )
        if xyz_path:
            return _chw_or_n3_to_xyz(np.load(xyz_path), hh, ww, xyz_path)
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
        if n != hh * ww:
            raise RuntimeError(
                f"{session_dir}: malformed legacy UMAP artifacts for {prefix}, "
                f"expected {hh * ww} points, got {n}"
            )
        return np.stack([x[:n], y[:n], z[:n]], axis=1).astype(np.float32)

    def _load_hw(prefix: str) -> tuple[int, int]:
        shp = os.path.join(results_dir, f"{prefix}_spatial_shape.npy")
        if os.path.exists(shp):
            arr = np.asarray(np.load(shp), dtype=np.int64).reshape(-1)
            if arr.size >= 2 and int(arr[0]) > 0 and int(arr[1]) > 0:
                return int(arr[0]), int(arr[1])
        return h_lat, w_lat

    def _slice_latent_xyz(prefix_out: str) -> np.ndarray:
        if prefix_out == "pred":
            src_map = outputs.get("pred_map", outputs.get("context_map"))
        elif prefix_out == "masked_pred":
            src_map = outputs.get("masked_pred_map", outputs.get("pred_map", outputs.get("context_map")))
        elif prefix_out == "gt":
            src_map = outputs.get("gt_map", outputs.get("pred_map"))
        else:
            src_map = outputs.get("context_map", outputs.get("pred_map"))
        if src_map is None:
            return np.zeros((h_lat * w_lat, 3), dtype=np.float32)
        xyz = _xyz_from_feature_map(_to_np(src_map))
        if xyz.shape[0] != h_lat * w_lat:
            return np.zeros((h_lat * w_lat, 3), dtype=np.float32)
        return xyz

    def _slice_latent_vectors(prefix_out: str) -> np.ndarray:
        if prefix_out == "pred":
            src_map = outputs.get("pred_map", outputs.get("context_map"))
        elif prefix_out == "masked_pred":
            src_map = outputs.get("masked_pred_map", outputs.get("pred_map", outputs.get("context_map")))
        elif prefix_out == "gt":
            src_map = outputs.get("gt_map", outputs.get("pred_map"))
        else:
            src_map = outputs.get("context_map", outputs.get("pred_map"))
        if src_map is None:
            return np.zeros((0, 0), dtype=np.float32)
        z = _latent_vectors_from_feature_map(_to_np(src_map))
        if z.shape[0] != h_lat * w_lat:
            return np.zeros((0, 0), dtype=np.float32)
        return z

    def _compute_slice_pca_umap(prefix_out: str) -> tuple[np.ndarray, np.ndarray]:
        z = _slice_latent_vectors(prefix_out)
        if z.ndim != 2 or z.shape[0] != h_lat * w_lat or z.shape[1] == 0:
            empty = np.full((h_lat * w_lat, 3), np.nan, dtype=np.float32)
            return empty, empty.copy()
        valid = np.isfinite(z).all(axis=1)
        pca = np.full((z.shape[0], 3), np.nan, dtype=np.float32)
        um = np.full((z.shape[0], 3), np.nan, dtype=np.float32)
        if int(np.count_nonzero(valid)) < 4:
            return pca, um
        z_valid = z[valid].astype(np.float32, copy=False)
        pca_valid = _compute_pca_3d(z_valid).astype(np.float32, copy=False)
        if pca_valid.ndim == 2 and pca_valid.shape[1] < 3:
            pca_pad = np.full((pca_valid.shape[0], 3), np.nan, dtype=np.float32)
            pca_pad[:, : pca_valid.shape[1]] = pca_valid
            pca_valid = pca_pad
        pca[valid] = pca_valid
        if DASHBOARD_COMPUTE_UMAP:
            umap_valid = _compute_umap_nd(
                _preprocess_latents_for_umap(z_valid),
                n_components=3,
                n_neighbors=15,
                min_dist=0.05,
                metric="cosine",
                random_state=42,
                init="spectral",
                fit_max_tokens=65536,
            ).astype(np.float32, copy=False)
            um[valid] = umap_valid
        return pca, um

    bundles = {}
    for prefix_saved, prefix_out in (
        ("context", "context"),
        ("predict", "pred"),
        ("masked_predict", "masked_pred"),
        ("target", "gt"),
    ):
        src_prefix = prefix_saved
        try:
            hh, ww = _load_hw(src_prefix)
            pca = _load_xyz_triplet(src_prefix, "pca", hh, ww)
            um = _load_xyz_triplet(src_prefix, "umap", hh, ww)
        except Exception:
            if prefix_saved == "context":
                src_prefix = "predict"
                try:
                    hh, ww = _load_hw(src_prefix)
                    pca = _load_xyz_triplet(src_prefix, "pca", hh, ww)
                    um = _load_xyz_triplet(src_prefix, "umap", hh, ww)
                except Exception:
                    pca, um = _compute_slice_pca_umap(prefix_out)
                    hh, ww = h_lat, w_lat
                else:
                    print(
                        f"dashboard_note={session_dir}: missing context embeddings; "
                        "using predict embeddings for context panels"
                    )
            else:
                pca, um = _compute_slice_pca_umap(prefix_out)
                hh, ww = h_lat, w_lat
        if pca.shape[0] != hh * ww or um.shape[0] != hh * ww:
            # Scatter artifacts can be sampled/volumetric and therefore cannot
            # be reshaped into the displayed slice. Use the actual slice latent
            # map for image-grid panels instead of rendering black placeholders.
            print(
                f"dashboard_note={session_dir}: {prefix_out} embedding point count "
                f"does not match slice grid pca={pca.shape[0]} umap={um.shape[0]} "
                f"grid={hh * ww}; using slice PCA fallback and blank UMAP grid"
            )
            pca, um = _compute_slice_pca_umap(prefix_out)
            hh, ww = h_lat, w_lat
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
    loss_x, loss_total, loss_prediction = [], [], []
    loss_spread = []
    loss_symmetry, weighted_symmetry = [], []
    weighted_prediction, weighted_spread = [], []
    embed_spread_mean, embed_spread_min, embed_under_spread_frac, dead_channel_count = [], [], [], []
    targets_per_image, mask_footprint_mean_px, mask_scale_factor = [], [], []
    if os.path.exists(metrics_path):
        def _row_float(row: dict[str, str], *keys: str) -> float:
            for key in keys:
                value = row.get(key)
                if value is None or value == "":
                    continue
                try:
                    return float(value)
                except Exception:
                    continue
            return float("nan")

        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep = _row_float(row, "epoch")
                ba = _row_float(row, "batch", "step")
                gs = _row_float(row, "global_step")
                tl = _row_float(row, "loss_total", "total_loss", "train_loss", "loss")
                jl = _row_float(row, "loss_prediction", "loss_jepa", "jepa_loss", "loss_mse")
                sl = _row_float(row, "loss_spread", "loss_sigreg")
                syml = _row_float(row, "loss_symmetry", "loss_symmetric")
                wj = _row_float(row, "weighted_prediction", "weighted_jepa", "weighted_mse")
                ws = _row_float(row, "weighted_spread", "weighted_sigreg")
                wsym = _row_float(row, "weighted_symmetry", "weighted_symmetric")
                esm = _row_float(row, "embed_spread_mean", "ctx_std_mean")
                esmin = _row_float(row, "embed_spread_min", "ctx_std_min")
                euf = _row_float(row, "embed_under_spread_frac")
                dcc = _row_float(row, "dead_channel_count")
                tpi = _row_float(row, "targets_per_image")
                mfpx = _row_float(row, "mask_footprint_mean_px")
                msf = _row_float(row, "mask_scale_factor")
                if np.isfinite(gs):
                    loss_x.append(gs)
                elif np.isfinite(ep) and np.isfinite(ba):
                    loss_x.append(ep + 0.001 * ba)
                elif np.isfinite(ep):
                    loss_x.append(ep)
                else:
                    continue
                if not (np.isfinite(tl) or np.isfinite(jl) or np.isfinite(sl) or np.isfinite(syml)):
                    loss_x.pop()
                    continue
                loss_total.append(tl if np.isfinite(tl) else np.nan)
                loss_prediction.append(jl if np.isfinite(jl) else np.nan)
                loss_spread.append(sl if np.isfinite(sl) else np.nan)
                loss_symmetry.append(syml if np.isfinite(syml) else np.nan)
                weighted_prediction.append(wj if np.isfinite(wj) else np.nan)
                weighted_spread.append(ws if np.isfinite(ws) else np.nan)
                weighted_symmetry.append(wsym if np.isfinite(wsym) else np.nan)
                embed_spread_mean.append(esm if np.isfinite(esm) else np.nan)
                embed_spread_min.append(esmin if np.isfinite(esmin) else np.nan)
                embed_under_spread_frac.append(euf if np.isfinite(euf) else np.nan)
                dead_channel_count.append(dcc if np.isfinite(dcc) else np.nan)
                targets_per_image.append(tpi if np.isfinite(tpi) else np.nan)
                mask_footprint_mean_px.append(mfpx if np.isfinite(mfpx) else np.nan)
                mask_scale_factor.append(msf if np.isfinite(msf) else np.nan)
    effective_rank_x, effective_rank_y = [], []
    run_results_path = os.path.join(session_dir, "run_results.csv")
    if os.path.exists(run_results_path):
        with open(run_results_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = float(row.get("timestamp", "nan"))
                    er = float(row.get("effective_rank", "nan"))
                except Exception:
                    continue
                if np.isfinite(ts) and np.isfinite(er):
                    effective_rank_x.append(ts)
                    effective_rank_y.append(er)

    rank_diag = {}
    rank_diag_path = os.path.join(session_dir, "rank_diagnostics.json")
    if os.path.exists(rank_diag_path):
        try:
            import json

            with open(rank_diag_path, "r", encoding="utf-8") as f:
                rank_diag = json.load(f)
        except Exception:
            rank_diag = {}
    def _rd(branch: str, key: str, legacy_key: str | None = None, default: float = np.nan) -> float:
        try:
            values = rank_diag.get(branch, {})
            return float(values.get(key, values.get(legacy_key, default)))
        except Exception:
            return float(default)
    rd_rank_match_ratio = float(rank_diag.get("rank_match_ratio", rank_diag.get("pred_gt_erank_ratio", np.nan))) if isinstance(rank_diag, dict) else np.nan
    rd_volume_match_ratio = float(rank_diag.get("volume_match_ratio", rank_diag.get("pred_gt_participation_ratio", np.nan))) if isinstance(rank_diag, dict) else np.nan

    # Compute mask sizes per sigma from config
    mask_sigma_names: list[str] = []
    mask_sigma_sizes: list[int] = []
    mask_config_summary: list[str] = []
    cfg, cfg_source = _load_session_config(session_dir)
    if cfg:
        try:
            mc = cfg.get("model", {})
            _sigmas = mc.get("sigmas", [2, 4, 8, 16])
            _ms_raw = mc.get("mask_size_scaling", 1.0)
            _ms = float(_ms_raw[0] if isinstance(_ms_raw, (list, tuple)) else _ms_raw)
            _mb_raw = mc.get("mask_size", 0)
            _mb = int(_mb_raw[0] if isinstance(_mb_raw, (list, tuple)) else _mb_raw)
            _random_box_per_target = bool(mc.get("random_mask_box_per_target", False))
            _manual_boxes_raw = mc.get("mask_size_manual")
            _manual_boxes = None
            if _manual_boxes_raw is not None:
                if isinstance(_manual_boxes_raw, str):
                    _manual_boxes = [int(round(float(v.strip()))) for v in _manual_boxes_raw.split(",") if v.strip()]
                else:
                    try:
                        _manual_boxes = [int(round(float(v))) for v in list(_manual_boxes_raw)]
                    except TypeError:
                        _manual_boxes = [int(round(float(_manual_boxes_raw)))]
            _mf = float(mc.get("active_target_fraction", mc.get("mask_fraction", 1.0)))
            _ps = int(mc.get("patch_size", 3))
            _symmetric = bool(mc.get("use_symmetric_feature_loss", False))
            _norm_l2 = bool(mc.get("normalize_loss_l2", mc.get("normalize_loss", False)))
            _sampling = str(mc.get("target_sampling_mode", "random"))
            _enc = str(mc.get("model_key", mc.get("encoder_type", "unknown")))
            mask_config_summary = [
                f"config={os.path.basename(cfg_source) if cfg_source else 'unknown'}",
                f"encoder={_enc}",
                f"mask_strategy={'random-box-per-target' if _random_box_per_target else 'standard'}",
                f"mask_size_scaling={_ms_raw}",
                f"mask_size={_mb_raw}",
                f"mask_size_manual={_manual_boxes}" if _manual_boxes else "mask_size_manual=None",
                f"active_target_fraction={_mf}",
                f"patch_size={_ps}",
                f"target_sampling={_sampling}",
                f"use_symmetric_feature_loss={_symmetric}",
                f"normalize_loss_l2={_norm_l2}",
            ]
            if _manual_boxes:
                if len(_manual_boxes) < len(_sigmas):
                    mask_config_summary.append("mask_size_manual shorter than sigmas; last size reused")
                elif len(_manual_boxes) > len(_sigmas):
                    mask_config_summary.append("mask_size_manual longer than sigmas; extras ignored")
            def _summary_box(i, s):
                if _manual_boxes:
                    return max(_ps, int(_manual_boxes[min(i, len(_manual_boxes) - 1)]))
                return max(_ps, round(float(s) * _ms + _mb))
            for i, s in enumerate(_sigmas):
                box = _summary_box(i, s)
                mask_sigma_names.append(f"σ={s}")
                mask_sigma_sizes.append(box)
            # Add overall summary
            if _random_box_per_target and isinstance(_mb_raw, (list, tuple)):
                _computed = f"per-target random boxes from [{_mb_raw[0]}, {_mb_raw[1]}]px before rejection"
            else:
                _computed = ", ".join(f"σ={s}→{_summary_box(i, s)}px" for i, s in enumerate(_sigmas))
            mask_config_summary.append(f"computed: {_computed}")
        except Exception:
            pass
    tsel_path = os.path.join(session_dir, "target_selection_summary.json")
    if os.path.exists(tsel_path):
        try:
            with open(tsel_path, "r", encoding="utf-8") as f:
                tsel = json.load(f)
            mask_config_summary.append(
                "target_select: "
                f"priority_n_target={tsel.get('priority_n_target_config')} "
                f"priority_min_targets_per_map={int(tsel.get('priority_min_targets_per_map_config', 0))} "
                f"active_target_fraction={float(tsel.get('active_target_fraction', np.nan)):.3g} "
                f"good_candidates_mean={float(tsel.get('priority_good_candidates_mean', 0.0)):.3g} "
                f"nonzero_mean={float(tsel.get('priority_nonzero_mean_mean', 1.0)):.3g} "
                f"auto_base_mean={float(tsel.get('priority_auto_base_targets_mean', 0.0)):.3g} "
                f"effective_mean={float(tsel.get('priority_effective_targets_mean', 0.0)):.3g}"
            )
        except Exception:
            pass

    scale_probe_data = _load_scale_probe_dash_data(session_dir)

    dash_payload = dict(
        dashboard_version=np.asarray(DASHBOARD_VERSION),
        rgb_render_source=np.asarray("inference-fov-nan"),
        dashboard_config_source=np.asarray(dashboard_config_source or "none"),
        orig=orig,
        blurred=blurred,
        target=target.astype(np.float32),
        target_loc_heatmap=target_loc_heatmap.astype(np.float32),
        energy_map=energy_map.astype(np.float32),
        visit_heatmap=visit_heatmap.astype(np.float32),
        visit_heatmap_kind=np.asarray(visit_heatmap_kind),
        pyramid_mask_stack=pyramid_mask_stack.astype(np.float32),
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
        masked_pred_pca3d=bundles["masked_pred"]["pca3d"],
        masked_pred_umap3d=bundles["masked_pred"]["umap3d"],
        masked_pred_pca_rgb=bundles["masked_pred"]["pca_rgb"],
        masked_pred_pca_rgb_flat=bundles["masked_pred"]["pca_rgb_flat"],
        masked_pred_umap_rgb=bundles["masked_pred"]["umap_rgb"],
        masked_pred_umap_rgb_flat=bundles["masked_pred"]["umap_rgb_flat"],
        gt_pca3d=bundles["gt"]["pca3d"],
        gt_umap3d=bundles["gt"]["umap3d"],
        gt_pca_rgb=bundles["gt"]["pca_rgb"],
        gt_pca_rgb_flat=bundles["gt"]["pca_rgb_flat"],
        gt_umap_rgb=bundles["gt"]["umap_rgb"],
        gt_umap_rgb_flat=bundles["gt"]["umap_rgb_flat"],
        loss_x=np.asarray(loss_x, dtype=np.float32),
        loss_total=np.asarray(loss_total, dtype=np.float32),
        loss_prediction=np.asarray(loss_prediction, dtype=np.float32),
        loss_spread=np.asarray(loss_spread, dtype=np.float32),
        loss_symmetry=np.asarray(loss_symmetry, dtype=np.float32),
        weighted_prediction=np.asarray(weighted_prediction, dtype=np.float32),
        weighted_spread=np.asarray(weighted_spread, dtype=np.float32),
        weighted_symmetry=np.asarray(weighted_symmetry, dtype=np.float32),
        embed_spread_mean=np.asarray(embed_spread_mean, dtype=np.float32),
        embed_spread_min=np.asarray(embed_spread_min, dtype=np.float32),
        embed_under_spread_frac=np.asarray(embed_under_spread_frac, dtype=np.float32),
        dead_channel_count=np.asarray(dead_channel_count, dtype=np.float32),
        targets_per_image=np.asarray(targets_per_image, dtype=np.float32),
        mask_footprint_mean_px=np.asarray(mask_footprint_mean_px, dtype=np.float32),
        mask_scale_factor=np.asarray(mask_scale_factor, dtype=np.float32),
        effective_rank_x=np.asarray(effective_rank_x, dtype=np.float64),
        effective_rank_y=np.asarray(effective_rank_y, dtype=np.float32),
        rank_context_erank=np.asarray([_rd("context", "erank")], dtype=np.float32),
        rank_pred_erank=np.asarray([_rd("pred", "erank")], dtype=np.float32),
        rank_gt_erank=np.asarray([_rd("gt", "erank")], dtype=np.float32),
        rank_context_manifold_size=np.asarray([_rd("context", "manifold_size", "participation_rank")], dtype=np.float32),
        rank_predicted_manifold_size=np.asarray([_rd("pred", "manifold_size", "participation_rank")], dtype=np.float32),
        rank_target_manifold_size=np.asarray([_rd("gt", "manifold_size", "participation_rank")], dtype=np.float32),
        rank_context_dead=np.asarray([_rd("context", "dead_channel_fraction")], dtype=np.float32),
        rank_pred_dead=np.asarray([_rd("pred", "dead_channel_fraction")], dtype=np.float32),
        rank_gt_dead=np.asarray([_rd("gt", "dead_channel_fraction")], dtype=np.float32),
        rank_context_top1=np.asarray([_rd("context", "top1_energy")], dtype=np.float32),
        rank_pred_top1=np.asarray([_rd("pred", "top1_energy")], dtype=np.float32),
        rank_gt_top1=np.asarray([_rd("gt", "top1_energy")], dtype=np.float32),
        rank_context_top4=np.asarray([_rd("context", "top4_energy")], dtype=np.float32),
        rank_pred_top4=np.asarray([_rd("pred", "top4_energy")], dtype=np.float32),
        rank_gt_top4=np.asarray([_rd("gt", "top4_energy")], dtype=np.float32),
        rank_context_top8=np.asarray([_rd("context", "top8_energy")], dtype=np.float32),
        rank_pred_top8=np.asarray([_rd("pred", "top8_energy")], dtype=np.float32),
        rank_gt_top8=np.asarray([_rd("gt", "top8_energy")], dtype=np.float32),
        rank_match_ratio=np.asarray([rd_rank_match_ratio], dtype=np.float32),
        volume_match_ratio=np.asarray([rd_volume_match_ratio], dtype=np.float32),
        mask_sigma_names=np.array(mask_sigma_names, dtype=str),
        mask_sigma_sizes=np.asarray(mask_sigma_sizes, dtype=np.int32),
        mask_config_summary=np.array(mask_config_summary, dtype=str),
    )
    dash_payload.update(scale_probe_data)
    np.savez_compressed(out_npz, **dash_payload)
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
    scale_pt, _ = _find_scale_probe_artifacts(session_dir)
    if scale_pt is not None:
        missing.extend(sorted(SCALE_PROBE_KEYS.difference(data.files)))
    if missing:
        data.close()
        print(f"dash_data_stale_recompute={npz_path} missing={','.join(missing)}")
        compute_dash_data(session_dir, overwrite=True)
        data = np.load(npz_path)
    model_mode = "unknown"
    cfg_used, _cfg_source = _load_session_config(session_dir)
    if cfg_used:
        model_mode = str(cfg_used.get("model", {}).get("mode", "unknown")).lower()

    def _safe_download_name(name: str) -> str:
        safe = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
        return safe or "panel"

    def _raw_png_data_url(arr: np.ndarray) -> str:
        vals = np.asarray(arr)
        if vals.ndim == 3 and vals.shape[-1] in (3, 4):
            out = vals
            if not np.issubdtype(out.dtype, np.integer):
                max_v = float(np.nanmax(out)) if out.size else 1.0
                scale = 255.0 if max_v <= 1.0 + 1e-6 else 1.0
                out = out.astype(np.float32) * scale
            out = np.clip(np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)
        elif vals.ndim == 2:
            raw = vals.astype(np.float32, copy=False)
            finite = raw[np.isfinite(raw)]
            if finite.size == 0:
                gray = np.full(raw.shape, 230, dtype=np.uint8)
            else:
                lo = float(np.min(finite))
                hi = float(np.max(finite))
                if hi <= lo + 1e-12:
                    hi = lo + 1.0
                gray = np.clip(np.round((np.nan_to_num(raw, nan=lo) - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)
                gray[~np.isfinite(raw)] = 230
            alpha = np.full(raw.shape, 255, dtype=np.uint8)
            out = np.dstack([gray, gray, gray, alpha])
        else:
            out = np.zeros((1, 1, 4), dtype=np.uint8)
        from PIL import Image

        bio = io.BytesIO()
        Image.fromarray(out).save(bio, format="PNG")
        return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode("ascii")

    def _image_axis_range(shape: tuple[int, int]) -> tuple[list[float], list[float]]:
        h, w = int(shape[0]), int(shape[1])
        return [-0.5, float(w) - 0.5], [float(h) - 0.5, -0.5]

    def _image_xy(shape: tuple[int, int], *, explicit_y_down: bool = False) -> tuple[np.ndarray, np.ndarray]:
        h, w = int(shape[0]), int(shape[1])
        x = np.arange(w, dtype=np.float32)
        y = np.arange(h, dtype=np.float32)
        if explicit_y_down:
            y = -y
        return x, y

    def _apply_image_axes(
        fig: go.Figure,
        shape: tuple[int, int],
        *,
        row: int | None = None,
        col: int | None = None,
        scaleanchor: str | None = "x",
        explicit_y_down: bool = False,
    ) -> None:
        if explicit_y_down:
            h, w = int(shape[0]), int(shape[1])
            xr, yr = [-0.5, float(w) - 0.5], [-(float(h) - 0.5), 0.5]
        else:
            xr, yr = _image_axis_range(shape)
        fig.update_xaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            constrain="domain",
            range=xr,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            scaleanchor=scaleanchor,
            scaleratio=1,
            constrain="domain",
            range=yr,
            row=row,
            col=col,
        )

    def heat(
        title: str,
        z: np.ndarray,
        colorscale: str,
        *,
        percentile_scale: bool = True,
        log1p_nonzero_nan: bool = False,
    ) -> go.Figure:
        vals = np.asarray(z, dtype=np.float32)
        if log1p_nonzero_nan:
            raw = np.asarray(vals, dtype=np.float32)
            out = np.full_like(raw, np.nan, dtype=np.float32)
            m = raw > 0.0
            if np.any(m):
                out[m] = np.log1p(raw[m]).astype(np.float32)
            vals = out
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            vals = np.zeros_like(vals)
            zmin, zmax = 0.0, 1.0
        else:
            if percentile_scale:
                zmin, zmax = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
            else:
                zmin, zmax = float(np.min(finite)), float(np.max(finite))
            if zmax <= zmin + 1e-12:
                zmax = zmin + 1.0
        h_v, w_v = vals.shape[-2:]
        x_v, y_v = _image_xy((h_v, w_v))
        fig = go.Figure(
            [
                go.Heatmap(
                    z=vals,
                    x=x_v,
                    y=y_v,
                    colorscale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                    showscale=False,
                )
            ]
        )
        fig.update_layout(
            template="plotly_white",
            title={"text": title, "x": 0.02},
            margin=dict(l=8, r=8, t=36, b=8),
            height=330,
            plot_bgcolor="#e6e8ef",
        )
        _apply_image_axes(fig, vals.shape[-2:])
        return fig

    def img(title: str, rgb: np.ndarray) -> go.Figure:
        vals = np.asarray(rgb)
        fig = go.Figure([go.Image(z=vals)])
        fig.update_layout(
            template="plotly_white",
            title={"text": title, "x": 0.02},
            margin=dict(l=8, r=8, t=36, b=8),
            height=330,
            plot_bgcolor="#e6e8ef",
        )
        _apply_image_axes(fig, vals.shape[:2])
        return fig

    def scatter3d(title: str, xyz: np.ndarray, rgb_flat: np.ndarray) -> tuple[go.Figure, int, int]:
        pts = np.asarray(xyz, dtype=np.float32)
        if pts.ndim == 3 and pts.shape[0] == 3:
            pts = np.transpose(pts, (1, 2, 0)).reshape(-1, 3)
        elif pts.ndim == 3 and pts.shape[-1] == 3:
            pts = pts.reshape(-1, 3)
        elif pts.ndim == 4 and pts.shape[0] == 1 and pts.shape[1] == 3:
            pts = np.transpose(pts[0], (1, 2, 0)).reshape(-1, 3)
        elif pts.ndim == 4 and pts.shape[0] == 1 and pts.shape[-1] == 3:
            pts = pts[0].reshape(-1, 3)
        rgb = np.asarray(rgb_flat)
        if rgb.ndim == 3 and rgb.shape[-1] == 3:
            rgb = rgb.reshape(-1, 3)
        source_n = int(pts.shape[0]) if pts.ndim == 2 and pts.shape[1] >= 3 else 0
        if source_n == 0:
            x, y, z = [], [], []
            colors = []
            rendered_n = 0
        else:
            n = source_n
            if rgb.ndim == 2:
                n = min(n, int(rgb.shape[0]))
            pts = pts[:n]
            if rgb.ndim == 2:
                rgb = rgb[:n]
            finite = np.isfinite(pts[:, :3]).all(axis=1)
            finite_pts = pts[finite]
            finite_rgb = rgb[finite] if rgb.ndim == 2 else rgb
            pts = pts[finite]
            if rgb.ndim == 2:
                rgb = rgb[finite]
            if pts.shape[0] > 0:
                lo = np.percentile(pts[:, :3], 1.0, axis=0)
                hi = np.percentile(pts[:, :3], 99.0, axis=0)
                valid_range = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
                keep = np.ones((pts.shape[0],), dtype=bool)
                for ax in range(3):
                    if valid_range[ax]:
                        keep &= (pts[:, ax] >= lo[ax]) & (pts[:, ax] <= hi[ax])
                min_keep = max(16, min(256, int(0.001 * pts.shape[0])))
                if int(np.count_nonzero(keep)) >= min_keep:
                    pts = pts[keep]
                    if rgb.ndim == 2:
                        rgb = rgb[keep]
            if pts.shape[0] == 0 and finite_pts.shape[0] > 0:
                pts = finite_pts
                if rgb.ndim == 2:
                    rgb = finite_rgb
            max_scatter_points = 50000
            if pts.shape[0] > max_scatter_points:
                sample_idx = np.linspace(0, pts.shape[0] - 1, max_scatter_points, dtype=np.int64)
                pts = pts[sample_idx]
                if rgb.ndim == 2:
                    rgb = rgb[sample_idx]
            rendered_n = int(pts.shape[0])
            x, y, z = pts[:, 0], -pts[:, 1], pts[:, 2]
            if rgb.ndim == 2 and rgb.shape[1] >= 3:
                colors = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in rgb]
            else:
                colors = ["rgb(127,127,127)"] * rendered_n
        fig = go.Figure(
            [
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(
                        size=3,
                        opacity=0.96,
                        color=colors,
                        showscale=False,
                        line=dict(width=0),
                    ),
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
                aspectmode="cube",
                camera=dict(projection=dict(type="orthographic")),
            ),
        )
        return fig, source_n, rendered_n

    def mask3d(title: str, cube: np.ndarray) -> go.Figure:
        c = np.asarray(cube, dtype=np.float32)
        if c.ndim != 3:
            c = np.zeros((1, 1, 1), dtype=np.float32)
        s, h0, w0 = c.shape
        zz, yy, xx = np.nonzero(c > 1e-6)
        if xx.size == 0:
            xx = np.asarray([], dtype=np.float32)
            yy = np.asarray([], dtype=np.float32)
            zz = np.asarray([], dtype=np.float32)
            vv = np.asarray([], dtype=np.float32)
        else:
            vv = c[zz, yy, xx]
        fig = go.Figure(
            [
                go.Scatter3d(
                    x=xx.astype(np.float32),
                    y=-yy.astype(np.float32),
                    z=zz.astype(np.float32),
                    mode="markers",
                    marker=dict(size=2, opacity=0.7, color=vv, colorscale="Viridis", showscale=True, colorbar=dict(title="mask")),
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
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="scale",
                aspectmode="manual",
                aspectratio=dict(x=w0, y=h0, z=max(w0, h0)),
                camera=dict(projection=dict(type="orthographic")),
            ),
        )
        return fig

    def scale_probe_bar(title: str, names: np.ndarray, sensitivity: np.ndarray, fraction: np.ndarray, sim: np.ndarray) -> go.Figure:
        labels = [str(v) for v in np.asarray(names).reshape(-1)]
        sens = np.asarray(sensitivity, dtype=np.float32).reshape(-1)
        frac = np.asarray(fraction, dtype=np.float32).reshape(-1)
        simv = np.asarray(sim, dtype=np.float32).reshape(-1)
        n0 = min(len(labels), sens.size, frac.size)
        labels = labels[:n0]
        sens = sens[:n0]
        frac = frac[:n0]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=sens, name="drop sensitivity", marker=dict(color="#F58518")))
        fig.add_trace(go.Scatter(x=labels, y=frac, mode="lines+markers", name="fraction", yaxis="y2", line=dict(color="#636EFA", width=2)))
        if simv.size >= n0:
            fig.add_trace(go.Scatter(x=labels, y=simv[:n0], mode="lines+markers", name="scale-only sim", yaxis="y2", line=dict(color="#00CC96", width=2)))
        fig.update_layout(
            template="plotly_white",
            title={"text": title, "x": 0.02},
            margin=dict(l=42, r=42, t=36, b=42),
            height=330,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
            yaxis=dict(title="mean response drop"),
            yaxis2=dict(title="fraction / similarity", overlaying="y", side="right", range=[0.0, 1.0]),
        )
        return fig

    def scale_probe_importance_scatter(title: str, names: np.ndarray, sensitivity: np.ndarray, fraction: np.ndarray, sim: np.ndarray) -> go.Figure:
        labels = [str(v) for v in np.asarray(names).reshape(-1)]
        sens = np.asarray(sensitivity, dtype=np.float32).reshape(-1)
        frac = np.asarray(fraction, dtype=np.float32).reshape(-1)
        simv = np.asarray(sim, dtype=np.float32).reshape(-1)
        n0 = min(len(labels), sens.size, frac.size)
        labels = labels[:n0]
        sens = sens[:n0]
        frac = frac[:n0]
        sim_plot = simv[:n0] if simv.size >= n0 else np.full(n0, np.nan, dtype=np.float32)
        marker_size = 12.0 + 56.0 * np.clip(frac, 0.0, 1.0)
        custom = np.stack([sens, frac, sim_plot], axis=1) if n0 > 0 else np.zeros((0, 3), dtype=np.float32)
        fig = go.Figure(
            [
                go.Scatter(
                    x=np.arange(n0),
                    y=sens,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    customdata=custom,
                    marker=dict(
                        size=marker_size,
                        color=frac,
                        colorscale="Viridis",
                        cmin=0.0,
                        cmax=max(1.0, float(np.nanmax(frac)) if frac.size else 1.0),
                        showscale=True,
                        colorbar=dict(title="importance frac"),
                        line=dict(color="#1f2937", width=1),
                        opacity=0.86,
                    ),
                    hovertemplate=(
                        "channel=%{text}<br>"
                        "drop=%{customdata[0]:.5g}<br>"
                        "importance=%{customdata[1]:.3f}<br>"
                        "scale-only sim=%{customdata[2]:.3f}<extra></extra>"
                    ),
                    showlegend=False,
                )
            ]
        )
        fig.update_layout(
            template="plotly_white",
            title={"text": title, "x": 0.02},
            margin=dict(l=42, r=42, t=46, b=42),
            height=330,
            xaxis=dict(title="CDD channel", tickmode="array", tickvals=list(range(n0)), ticktext=labels),
            yaxis=dict(title="mean response drop"),
        )
        return fig

    def scale_probe_spatial_scatter3d(title: str, maps: np.ndarray, names: np.ndarray) -> go.Figure:
        arr = np.asarray(maps, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] <= 0:
            arr = np.zeros((1, 1, 1), dtype=np.float32)
        s, h0, w0 = arr.shape
        labels = [str(v) for v in np.asarray(names).reshape(-1)]
        if len(labels) != s:
            labels = [f"scale_{i}" for i in range(s)]
        valid = np.isfinite(arr) & (arr > 0.0)
        finite_pos = arr[valid]
        threshold = 0.0
        if finite_pos.size > 0:
            threshold = float(np.percentile(finite_pos, 75.0))
        zz, yy, xx = np.nonzero(valid & (arr >= threshold))
        if xx.size == 0:
            zz, yy, xx = np.nonzero(valid)
        if xx.size == 0:
            xx = np.asarray([0], dtype=np.int32)
            yy = np.asarray([0], dtype=np.int32)
            zz = np.asarray([0], dtype=np.int32)
            vv = np.asarray([0.0], dtype=np.float32)
        else:
            vv = arr[zz, yy, xx].astype(np.float32)
        max_points = 65000
        if vv.size > max_points:
            order = np.argsort(vv)[-max_points:]
            xx = xx[order]
            yy = yy[order]
            zz = zz[order]
            vv = vv[order]
        tickvals = list(range(s))
        fig = go.Figure(
            [
                go.Scatter3d(
                    x=xx.astype(np.float32),
                    y=(-yy).astype(np.float32),
                    z=zz.astype(np.float32),
                    mode="markers",
                    marker=dict(
                        size=2,
                        opacity=0.72,
                        color=vv,
                        colorscale="Inferno",
                        showscale=True,
                        colorbar=dict(title="response"),
                    ),
                    text=[labels[int(i)] for i in zz],
                    customdata=yy.astype(np.float32),
                    hovertemplate="x=%{x}<br>row=%{customdata}<br>channel=%{text}<br>response=%{marker.color:.5g}<extra></extra>",
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
                xaxis_title="x",
                yaxis_title="y",
                zaxis=dict(title="CDD channel", tickmode="array", tickvals=tickvals, ticktext=labels),
                aspectmode="manual",
                aspectratio=dict(x=w0, y=h0, z=max(w0, h0)),
                camera=dict(projection=dict(type="orthographic")),
            ),
        )
        return fig

    def scale_probe_maps(title: str, maps: np.ndarray, names: np.ndarray, colorscale: str, zmid: float | None = None) -> go.Figure:
        arr = np.asarray(maps, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] <= 0:
            return heat(title, np.zeros((1, 1), dtype=np.float32), colorscale)
        labels = [str(v) for v in np.asarray(names).reshape(-1)]
        if len(labels) != arr.shape[0]:
            labels = [f"scale_{i}" for i in range(arr.shape[0])]
        cols = min(4, int(arr.shape[0]))
        rows = int(np.ceil(arr.shape[0] / max(1, cols)))
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=labels)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            zmin = float(np.percentile(finite, 1))
            zmax = float(np.percentile(finite, 99))
            if zmax <= zmin + 1e-12:
                zmax = zmin + 1.0
        else:
            zmin, zmax = 0.0, 1.0
        for i in range(arr.shape[0]):
            r = i // cols + 1
            c = i % cols + 1
            h_i, w_i = arr[i].shape[-2:]
            x_i, y_i = _image_xy((h_i, w_i), explicit_y_down=True)
            kwargs = dict(
                z=arr[i],
                x=x_i,
                y=y_i,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                showscale=(i == arr.shape[0] - 1),
            )
            if zmid is not None:
                kwargs["zmid"] = zmid
            fig.add_trace(go.Heatmap(**kwargs), row=r, col=c)
            axis_idx = (r - 1) * cols + c
            _apply_image_axes(
                fig,
                arr[i].shape,
                row=r,
                col=c,
                scaleanchor="x" if axis_idx == 1 else f"x{axis_idx}",
                explicit_y_down=True,
            )
        fig.update_layout(template="plotly_white", title={"text": title, "x": 0.02}, margin=dict(l=8, r=8, t=56, b=8), height=max(330, 260 * rows))
        return fig

    def winner_heat(title: str, winner: np.ndarray, names: np.ndarray) -> go.Figure:
        vals = np.asarray(winner, dtype=np.float32)
        labels = [str(v) for v in np.asarray(names).reshape(-1)]
        n_scales = max(1, len(labels))
        x_v, y_v = _image_xy(vals.shape[-2:], explicit_y_down=True)
        fig = go.Figure(
            [
                go.Heatmap(
                    z=vals,
                    x=x_v,
                    y=y_v,
                    colorscale="Turbo",
                    zmin=-0.5,
                    zmax=float(n_scales) - 0.5,
                    colorbar=dict(
                        title="scale",
                        tickmode="array",
                        tickvals=list(range(n_scales)),
                        ticktext=labels if labels else [str(i) for i in range(n_scales)],
                    ),
                )
            ]
        )
        fig.update_layout(template="plotly_white", title={"text": title, "x": 0.02}, margin=dict(l=8, r=8, t=36, b=8), height=330)
        _apply_image_axes(fig, vals.shape[-2:], explicit_y_down=True)
        return fig

    def _npz_array(name: str, *legacy_names: str) -> np.ndarray:
        for key in (name, *legacy_names):
            if key in data.files:
                return np.asarray(data[key], dtype=np.float32)
        return np.asarray([], dtype=np.float32)

    def _orient_cdd_probe_for_display(arr: np.ndarray) -> np.ndarray:
        vals = np.asarray(arr, dtype=np.float32)
        return vals.astype(np.float32, copy=False)

    loss_x = _npz_array("loss_x")
    loss_total = np.asarray(data["loss_total"], dtype=np.float32) if "loss_total" in data.files else np.asarray([], dtype=np.float32)
    loss_prediction = _npz_array("loss_prediction", "loss_jepa")
    loss_spread = _npz_array("loss_spread", "loss_sigreg")
    loss_symmetry = _npz_array("loss_symmetry", "loss_symmetric")
    loss_var = np.asarray(data["loss_var"], dtype=np.float32) if "loss_var" in data.files else np.asarray([], dtype=np.float32)
    loss_cov = np.asarray(data["loss_cov"], dtype=np.float32) if "loss_cov" in data.files else np.asarray([], dtype=np.float32)
    weighted_prediction = _npz_array("weighted_prediction", "weighted_jepa")
    weighted_spread = _npz_array("weighted_spread", "weighted_sigreg")
    weighted_symmetry = _npz_array("weighted_symmetry", "weighted_symmetric")
    weighted_var = np.asarray(data["weighted_var"], dtype=np.float32) if "weighted_var" in data.files else np.asarray([], dtype=np.float32)
    weighted_cov = np.asarray(data["weighted_cov"], dtype=np.float32) if "weighted_cov" in data.files else np.asarray([], dtype=np.float32)
    embed_spread_mean = _npz_array("embed_spread_mean")
    embed_spread_min = _npz_array("embed_spread_min")
    embed_under_spread_frac = _npz_array("embed_under_spread_frac")
    dead_channel_count = _npz_array("dead_channel_count")
    targets_per_image = _npz_array("targets_per_image")
    mask_footprint_mean_px = _npz_array("mask_footprint_mean_px")
    mask_scale_factor = _npz_array("mask_scale_factor")
    def _smooth(y: np.ndarray, win: int = 25) -> np.ndarray:
        arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if arr.size < 3 or win <= 1:
            return arr
        valid = np.isfinite(arr)
        if not valid.any():
            return np.full_like(arr, np.nan, dtype=np.float32)
        w = min(win, max(3, arr.size // 5))
        if w % 2 == 0:
            w += 1
        pad = w // 2
        values = np.where(valid, arr, 0.0).astype(np.float32)
        weights = valid.astype(np.float32)
        values_pad = np.pad(values, (pad, pad), mode="edge")
        weights_pad = np.pad(weights, (pad, pad), mode="edge")
        kernel = np.ones(w, dtype=np.float32)
        num = np.convolve(values_pad, kernel, mode="valid")
        den = np.convolve(weights_pad, kernel, mode="valid")
        out = np.divide(num, np.maximum(den, 1e-6), dtype=np.float32)
        out[den <= 0.0] = np.nan
        return out
    n = min(loss_x.size, loss_total.size, loss_prediction.size) if (loss_x.size and loss_total.size and loss_prediction.size) else 0
    loss_terms = (
        ("prediction", "prediction_loss_weight", loss_prediction, weighted_prediction, "#636EFA"),
        ("spread", "spread_regularizer.weight", loss_spread, weighted_spread, "#EF553B"),
        ("symmetry", "symmetry_loss_weight", loss_symmetry, weighted_symmetry, "#00CC96"),
    )
    loss_weights = {}
    loss_weights_path = os.path.join(session_dir, "loss_weights.json")
    if os.path.exists(loss_weights_path):
        try:
            with open(loss_weights_path, "r", encoding="utf-8") as f:
                loss_weights = json.load(f)
        except Exception:
            loss_weights = {}
    active_loss_terms = []
    if n > 0:
        for name, weight_key, raw_arr, weighted_arr, color in loss_terms:
            if raw_arr.size < n:
                continue
            if weight_key == "spread_regularizer.weight":
                weight = loss_weights.get("spread_regularizer", {}).get("weight")
            else:
                weight = loss_weights.get(weight_key)
            raw = raw_arr[:n]
            if weighted_arr.size >= n:
                weighted = weighted_arr[:n]
            elif weight is not None:
                weighted = raw * float(weight)
            else:
                weighted = np.full(n, np.nan, dtype=np.float32)
            observed_active = np.isfinite(weighted).any() and np.nanmax(np.abs(weighted)) > 1e-12
            if weight is not None:
                active = abs(float(weight)) > 1e-12
            else:
                active = observed_active
            if active:
                active_loss_terms.append((name, raw, weighted, color))

    active_loss_legend = dict(
        orientation="v",
        x=0.985,
        y=0.985,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.72)",
        bordercolor="rgba(120,120,120,0.35)",
        borderwidth=1,
    )

    def _add_loss_trace(fig: go.Figure, *, x: np.ndarray, y: np.ndarray, name: str, color: str) -> None:
        values = np.where(np.isfinite(y), y, np.nan).astype(np.float32)
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=values,
                mode="lines",
                name=f"{name} raw",
                line=dict(width=1.8, color="#aaaaaa", dash="dot"),
                opacity=0.55,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=x, y=_smooth(values), mode="lines", name=name,
                line=dict(width=2, color=color),
            )
        )

    fig_loss_components = go.Figure()
    if n > 0:
        lx = loss_x[:n]
        for name, raw_arr, _, color in active_loss_terms:
            _add_loss_trace(fig_loss_components, x=lx, y=raw_arr, name=f"loss_{name}", color=color)
    fig_loss_components.update_layout(
        template="plotly_white",
        title={"text": "Active Loss Terms (Unweighted)", "x": 0.02},
        margin=dict(l=42, r=8, t=36, b=36),
        height=330,
        legend=active_loss_legend,
    )
    fig_loss_components.update_xaxes(title_text="global_step")
    fig_loss_components.update_yaxes(title_text="loss")
    fig_weighted_components = go.Figure()
    if n > 0:
        lx = loss_x[:n]
        for name, _, weighted_arr, color in active_loss_terms:
            _add_loss_trace(fig_weighted_components, x=lx, y=weighted_arr, name=f"weighted_{name}", color=color)
        _add_loss_trace(fig_weighted_components, x=lx, y=loss_total[:n], name="loss_total", color="#222222")
    fig_weighted_components.update_layout(
        template="plotly_white",
        title={"text": "Active Loss Terms (Weighted into loss_total)", "x": 0.02},
        margin=dict(l=42, r=8, t=36, b=36),
        height=330,
        legend=active_loss_legend,
    )
    fig_weighted_components.update_xaxes(title_text="global_step")
    fig_weighted_components.update_yaxes(title_text="weighted contribution")
    def _latest_finite(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        finite = arr[np.isfinite(arr)]
        return float(finite[-1]) if finite.size > 0 else float("nan")

    def _fmt_table_value(value: float, *, percent: bool = False, integer: bool = False) -> str:
        if not np.isfinite(value):
            return "-"
        if percent:
            return f"{100.0 * value:.1f}%"
        if integer:
            return str(int(round(value)))
        return f"{value:.4g}"

    spread_cfg = loss_weights.get("spread_regularizer", {})
    spread_target = float(spread_cfg.get("target_std", loss_weights.get("embed_spread_target", 1.0)))
    latest_step = _latest_finite(loss_x)
    fig_spread_health = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Metric", "Latest", "Healthy direction"],
                    fill_color="#E8EEF7",
                    align=["left", "right", "left"],
                    font=dict(size=13, color="#1F2D3D"),
                ),
                cells=dict(
                    values=[
                        [
                            "Average embedding spread",
                            "Weakest dimension spread",
                            "Under-spread dimensions",
                            "Dead channels",
                        ],
                        [
                            _fmt_table_value(_latest_finite(embed_spread_mean)),
                            _fmt_table_value(_latest_finite(embed_spread_min)),
                            _fmt_table_value(_latest_finite(embed_under_spread_frac), percent=True),
                            _fmt_table_value(_latest_finite(dead_channel_count), integer=True),
                        ],
                        [
                            f">= {spread_target:g}",
                            f">= {spread_target:g}",
                            "toward 0%",
                            "0",
                        ],
                    ],
                    fill_color="#FFFFFF",
                    align=["left", "right", "left"],
                    height=31,
                    font=dict(size=13, color="#263238"),
                ),
            )
        ]
    )
    fig_spread_health.update_layout(
        template="plotly_white",
        title={"text": f"Embedding Spread Health (latest step: {_fmt_table_value(latest_step, integer=True)})", "x": 0.02},
        margin=dict(l=8, r=8, t=42, b=8),
        height=330,
    )
    fig_mask_geometry = go.Figure()
    if loss_x.size > 0:
        for values, name, color in (
            (targets_per_image, "Targets per image", "#00CC96"),
            (mask_footprint_mean_px, "Mask size", "#636EFA"),
            (mask_scale_factor, "Mask scale", "#EF553B"),
        ):
            m = min(loss_x.size, values.size)
            if m > 0:
                _add_loss_trace(fig_mask_geometry, x=loss_x[:m], y=values[:m], name=name, color=color)
    fig_mask_geometry.update_layout(
        template="plotly_white",
        title={"text": "Mask Geometry", "x": 0.02},
        margin=dict(l=42, r=8, t=36, b=36),
        height=330,
    )
    er_x = np.asarray(data["effective_rank_x"], dtype=np.float64) if "effective_rank_x" in data.files else np.asarray([], dtype=np.float64)
    er_y = np.asarray(data["effective_rank_y"], dtype=np.float32) if "effective_rank_y" in data.files else np.asarray([], dtype=np.float32)
    fig_eff_rank = go.Figure()
    if er_x.size > 0 and er_y.size > 0:
        m = min(er_x.size, er_y.size)
        fig_eff_rank.add_trace(go.Scatter(x=er_x[:m], y=er_y[:m], mode="lines+markers", name="effective_rank"))
    fig_eff_rank.update_layout(
        template="plotly_white",
        title={"text": "Effective Rank", "x": 0.02},
        margin=dict(l=42, r=8, t=36, b=36),
        height=330,
    )
    fig_eff_rank.update_xaxes(title_text="timestamp")
    fig_eff_rank.update_yaxes(title_text="effective_rank")
    def _scalar(name: str, *legacy_names: str) -> float:
        for key in (name, *legacy_names):
            if key not in data.files:
                continue
            arr = np.asarray(data[key]).reshape(-1)
            if arr.size > 0 and np.isfinite(arr[0]):
                return float(arr[0])
        return float("nan")

    rank_branches = ["context", "pred", "gt"]
    rank_erank = [_scalar("rank_context_erank"), _scalar("rank_pred_erank"), _scalar("rank_gt_erank")]
    rank_pr = [
        _scalar("rank_context_manifold_size", "rank_context_pr"),
        _scalar("rank_predicted_manifold_size", "rank_pred_pr"),
        _scalar("rank_target_manifold_size", "rank_gt_pr"),
    ]
    rank_dead = [_scalar("rank_context_dead"), _scalar("rank_pred_dead"), _scalar("rank_gt_dead")]
    rank_top1 = [_scalar("rank_context_top1"), _scalar("rank_pred_top1"), _scalar("rank_gt_top1")]
    rank_top4 = [_scalar("rank_context_top4"), _scalar("rank_pred_top4"), _scalar("rank_gt_top4")]
    rank_top8 = [_scalar("rank_context_top8"), _scalar("rank_pred_top8"), _scalar("rank_gt_top8")]
    rank_ratio = _scalar("rank_match_ratio", "rank_pred_gt_erank_ratio")

    fig_rank_diag = go.Figure()
    fig_rank_diag.add_trace(go.Bar(name="effective rank", x=rank_branches, y=rank_erank))
    fig_rank_diag.add_trace(go.Bar(name="manifold size", x=rank_branches, y=rank_pr))
    fig_rank_diag.add_trace(go.Bar(name="dead channels", x=rank_branches, y=rank_dead))
    subtitle = ""
    if np.isfinite(rank_ratio):
        subtitle = f" (rank match={rank_ratio:.3f})"
    fig_rank_diag.update_layout(
        barmode="group",
        template="plotly_white",
        title={"text": f"Manifold Diagnostics{subtitle}", "x": 0.02},
        margin=dict(l=42, r=8, t=36, b=36),
        height=330,
    )
    fig_rank_energy = go.Figure()
    fig_rank_energy.add_trace(go.Bar(name="top1_energy", x=rank_branches, y=rank_top1))
    fig_rank_energy.add_trace(go.Bar(name="top4_energy", x=rank_branches, y=rank_top4))
    fig_rank_energy.add_trace(go.Bar(name="top8_energy", x=rank_branches, y=rank_top8))
    fig_rank_energy.update_layout(
        barmode="group",
        template="plotly_white",
        title={"text": "Rank Energy Concentration (Top-k)", "x": 0.02},
        margin=dict(l=42, r=8, t=36, b=36),
        height=330,
    )
    energy_vals = np.asarray(data["energy_map"], dtype=np.float32).reshape(-1)
    energy_vals = energy_vals[np.isfinite(energy_vals)]
    fig_energy_dist = go.Figure()
    fig_energy_dist.add_trace(
        go.Histogram(
            x=energy_vals.tolist(),
            nbinsx=90,
            marker=dict(color="#F58518"),
            name="energy",
            showlegend=False,
        )
    )
    fig_energy_dist.update_layout(
        template="plotly_white",
        title={"text": "Energy Map Distribution (Masked Predict - Target, 0-1.5)", "x": 0.02},
        margin=dict(l=42, r=8, t=36, b=36),
        height=330,
        xaxis=dict(title="energy value", range=[0.0, 1.5]),
        yaxis=dict(title="count"),
    )

    cards: list[dict] = []
    for name, stem in (("Context", "context"), ("Masked Predict", "masked_pred"), ("Predict", "pred"), ("Target", "gt")):
        pca_scatter, _, _ = scatter3d(f"{name} PCA 3D Scatter", data[f"{stem}_pca3d"], data[f"{stem}_pca_rgb_flat"])
        umap_scatter, _, _ = scatter3d(f"{name} UMAP 3D Scatter", data[f"{stem}_umap3d"], data[f"{stem}_umap_rgb_flat"])
        # Keep strict left-right pairing: RGB map (left), RGB scatter (right).
        cards.append(
            {
                "title": f"{name} PCA RGB",
                "fig": img(f"{name} PCA RGB", data[f"{stem}_pca_rgb"]),
                "group": f"{stem}-pca",
                "raw_png": _raw_png_data_url(data[f"{stem}_pca_rgb"]),
            }
        )
        cards.append({"title": f"{name} PCA RGB Scatter", "fig": pca_scatter, "group": f"{stem}-pca"})
        cards.append(
            {
                "title": f"{name} UMAP RGB",
                "fig": img(f"{name} UMAP RGB", data[f"{stem}_umap_rgb"]),
                "group": f"{stem}-umap",
                "raw_png": _raw_png_data_url(data[f"{stem}_umap_rgb"]),
            }
        )
        cards.append({"title": f"{name} UMAP RGB Scatter", "fig": umap_scatter, "group": f"{stem}-umap"})
    if "scale_probe_sensitivity_maps" in data.files and "scale_probe_names" in data.files:
        sp_names = data["scale_probe_names"]
        sp_sens_maps = _orient_cdd_probe_for_display(data["scale_probe_sensitivity_maps"])
        sp_sim_maps = _orient_cdd_probe_for_display(data["scale_probe_scale_only_sim_maps"]) if "scale_probe_scale_only_sim_maps" in data.files else np.asarray([], dtype=np.float32)
        sp_winner = _orient_cdd_probe_for_display(data["scale_probe_winner_map"]) if "scale_probe_winner_map" in data.files else np.asarray([], dtype=np.float32)
        sp_sens_mean = np.asarray(data["scale_probe_sensitivity_mean"], dtype=np.float32) if "scale_probe_sensitivity_mean" in data.files else np.asarray([], dtype=np.float32)
        sp_sens_frac = np.asarray(data["scale_probe_sensitivity_fraction"], dtype=np.float32) if "scale_probe_sensitivity_fraction" in data.files else np.asarray([], dtype=np.float32)
        sp_sim_mean = np.asarray(data["scale_probe_scale_only_similarity"], dtype=np.float32) if "scale_probe_scale_only_similarity" in data.files else np.asarray([], dtype=np.float32)
        sp_input = _orient_cdd_probe_for_display(data["scale_probe_input_map"]) if "scale_probe_input_map" in data.files else np.asarray([], dtype=np.float32)
        cards.extend(
            [
                {
                    "title": "CDD Scale Response Summary",
                    "fig": scale_probe_bar("CDD Scale Response Summary", sp_names, sp_sens_mean, sp_sens_frac, sp_sim_mean),
                    "group": "scale-probe-summary",
                },
                {
                    "title": "CDD Channel Importance Scatter",
                    "fig": scale_probe_importance_scatter("CDD Channel Importance Scatter", sp_names, sp_sens_mean, sp_sens_frac, sp_sim_mean),
                    "group": "scale-probe-importance-scatter",
                },
                {
                    "title": "CDD Spatial Response Scatter",
                    "fig": scale_probe_spatial_scatter3d("CDD Spatial Response Scatter (top quartile)", sp_sens_maps, sp_names),
                    "group": "scale-probe-spatial-scatter",
                },
                *(
                    [
                        {
                            "title": "Scale Probe Input (CDD Sum)",
                            "fig": heat("Scale Probe Input (CDD Sum)", sp_input, "Viridis"),
                            "group": "scale-probe-input",
                        }
                    ]
                    if sp_input.ndim == 2 and sp_input.size > 0
                    else []
                ),
                {
                    "title": "CDD Scale Drop Sensitivity",
                    "fig": scale_probe_maps("CDD Scale Drop Sensitivity", sp_sens_maps, sp_names, "Inferno"),
                    "group": "scale-probe-sensitivity",
                },
            ]
        )
        if sp_sim_maps.ndim == 3 and sp_sim_maps.size > 0:
            cards.append(
                {
                    "title": "CDD Scale-Only Similarity",
                    "fig": scale_probe_maps("CDD Scale-Only Similarity", sp_sim_maps, sp_names, "RdYlBu_r", zmid=0.0),
                    "group": "scale-probe-only-sim",
                }
            )
        if sp_winner.ndim == 2 and sp_winner.size > 0:
            cards.append(
                {
                    "title": "Dominant CDD Scale Map",
                    "fig": winner_heat("Dominant CDD Scale Map", sp_winner, sp_names),
                    "group": "scale-probe-winner",
                }
            )
        if "scale_probe_pred_sensitivity_maps" in data.files:
            sp_pred_maps = _orient_cdd_probe_for_display(data["scale_probe_pred_sensitivity_maps"])
            if sp_pred_maps.ndim == 3 and sp_pred_maps.size > 0:
                cards.append(
                    {
                        "title": "Predictor CDD Scale Drop Sensitivity",
                        "fig": scale_probe_maps("Predictor CDD Scale Drop Sensitivity", sp_pred_maps, sp_names, "Inferno"),
                        "group": "scale-probe-pred-sensitivity",
                    }
                )
    # Non-pair panels afterwards.
    cards.extend(
        [
            {"title": "Input (Log-Norm)", "fig": heat("Input (Log-Norm)", data["orig"], "Viridis"), "group": "input", "raw_png": _raw_png_data_url(data["orig"])},
            {"title": "Effective Rank", "fig": fig_eff_rank, "group": "eff-rank"},
            {"title": "Active Loss Terms (Unweighted)", "fig": fig_loss_components, "group": "loss-components"},
            {"title": "Active Loss Terms (Weighted)", "fig": fig_weighted_components, "group": "weighted-loss-components"},
            {"title": "Embedding Spread Health", "fig": fig_spread_health, "group": "spread-health"},
            {"title": "Mask Geometry", "fig": fig_mask_geometry, "group": "mask-geometry"},
            {"title": "Manifold Diagnostics", "fig": fig_rank_diag, "group": "rank-diag"},
            {"title": "Rank Energy Top-k", "fig": fig_rank_energy, "group": "rank-energy"},
            {"title": "Energy Distribution", "fig": fig_energy_dist, "group": "energy-dist"},
            {"title": "Target Locations", "fig": heat("Target Locations", data["target"], "Magma"), "group": "target-loc", "raw_png": _raw_png_data_url(data["target"])},
            {"title": "Target Location Heatmap", "fig": heat("Target Location Heatmap", data["target_loc_heatmap"], "Magma"), "group": "target-heat", "raw_png": _raw_png_data_url(data["target_loc_heatmap"])},
            {"title": "Energy Map (Masked Predict - Target)", "fig": heat("Energy Map (Masked Predict - Target)", data["energy_map"], "Inferno"), "group": "energy", "raw_png": _raw_png_data_url(data["energy_map"])},
            {
                "title": str(data["visit_heatmap_kind"]) if "visit_heatmap_kind" in data.files else "Visit Frequency Heatmap",
                "fig": heat(
                    f"{str(data['visit_heatmap_kind']) if 'visit_heatmap_kind' in data.files else 'Visit Frequency Heatmap'} (log1p, zero=NaN)",
                    data["visit_heatmap"],
                    "Cividis",
                    percentile_scale=False,
                    log1p_nonzero_nan=True,
                ),
                "group": "visit",
                "raw_png": _raw_png_data_url(data["visit_heatmap"]),
            },
        ]
    )
    if model_mode in ("pyramid", "3d_slab") and "pyramid_mask_stack" in data.files:
        cards.append(
            {
                "title": "Pyramid Mask Stack (Sample-0)",
                "fig": mask3d("Pyramid Mask Stack (Sample-0)", data["pyramid_mask_stack"]),
                "group": "pyr-mask-stack",
            }
        )
    rendered = []
    for i, card in enumerate(cards):
        fig = card["fig"]
        group = card["group"]
        panel_title = str(card.get("title", f"panel_{i+1}"))
        panel_title_html = html_lib.escape(panel_title, quote=True)
        raw_png = card.get("raw_png")
        if raw_png:
            href = html_lib.escape(str(raw_png), quote=True)
            file_name = html_lib.escape(f"{_safe_download_name(panel_title)}.png", quote=True)
            save_control = f'<a class="save-panel" download="{file_name}" href="{href}">Save raw PNG</a>'
        else:
            save_control = f'<button class="save-panel" type="button" data-panel-title="{panel_title_html}">Save plot PNG</button>'
        controls = ""
        control_id = f"panel-{i}"
        control_kind = "scatter" if "Scatter" in panel_title else "image"
        if "-pca" in group or "-umap" in group:
            if control_kind == "image":
                axis_controls = (
                    f'<label>xmin <input type="number" step="any" data-k="xmin" placeholder="auto"></label>'
                    f'<label>xmax <input type="number" step="any" data-k="xmax" placeholder="auto"></label>'
                    f'<label>ymin <input type="number" step="any" data-k="ymin" placeholder="auto"></label>'
                    f'<label>ymax <input type="number" step="any" data-k="ymax" placeholder="auto"></label>'
                    f'<label>color low <input type="number" step="any" data-k="color_low" placeholder="visible 1%"></label>'
                    f'<label>color high <input type="number" step="any" data-k="color_high" placeholder="visible 99%"></label>'
                    f'<label><input type="checkbox" data-k="invert_x"> invert R</label>'
                    f'<label><input type="checkbox" data-k="invert_y"> invert G</label>'
                    f'<label><input type="checkbox" data-k="invert_z"> invert B</label>'
                    f'<label><input type="checkbox" data-k="invert_color"> invert all</label>'
                )
            else:
                axis_controls = (
                    f'<label>xmin <input type="number" step="any" data-k="xmin" placeholder="auto"></label>'
                    f'<label>xmax <input type="number" step="any" data-k="xmax" placeholder="auto"></label>'
                    f'<label>ymin <input type="number" step="any" data-k="ymin" placeholder="auto"></label>'
                    f'<label>ymax <input type="number" step="any" data-k="ymax" placeholder="auto"></label>'
                    f'<label>zmin <input type="number" step="any" data-k="zmin" placeholder="auto"></label>'
                    f'<label>zmax <input type="number" step="any" data-k="zmax" placeholder="auto"></label>'
                    f'<label>color low <input type="number" step="any" data-k="color_low" placeholder="visible 1%"></label>'
                    f'<label>color high <input type="number" step="any" data-k="color_high" placeholder="visible 99%"></label>'
                    f'<label><input type="checkbox" data-k="invert_x"> invert x</label>'
                    f'<label><input type="checkbox" data-k="invert_y"> invert y</label>'
                    f'<label><input type="checkbox" data-k="invert_z"> invert z</label>'
                    f'<label><input type="checkbox" data-k="invert_color"> invert color</label>'
                )
            controls = (
                f'<div class="controls local-controls" data-control-id="{control_id}" data-kind="{control_kind}">'
                f'{axis_controls}'
                f'<button class="apply-local" type="button" data-control-id="{control_id}">Apply</button>'
                f"</div>"
            )
        rendered.append(
            f'<section class="card" data-group="{group}" data-control-id="{control_id}">'
            f'<div class="card-tools">{save_control}</div>'
            f'{controls}'
            f'{fig.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False), config={"responsive": True, "displaylogo": False, "modeBarButtonsToRemove": ["toImage"]})}'
            f'</section>'
        )
    # Keep grid alignment stable: if odd number of cards, append a dummy placeholder.
    if len(rendered) % 2 == 1:
        rendered.append('<section class="card card-dummy" aria-hidden="true"></section>')

    # Build mask config summary from NPZ data
    ms_names = data.get("mask_sigma_names", np.array([], dtype=str))
    ms_sizes = data.get("mask_sigma_sizes", np.array([], dtype=np.int32))
    ms_summary = data.get("mask_config_summary", np.array([], dtype=str))
    er_y = data.get("effective_rank_y", np.array([], dtype=np.float32))
    latest_er = float(er_y[-1]) if len(er_y) > 0 and np.isfinite(er_y[-1]) else None
    dash_data_version = str(np.asarray(data.get("dashboard_version", np.asarray("missing"))).reshape(-1)[0])
    rgb_render_source = str(np.asarray(data.get("rgb_render_source", np.asarray("missing"))).reshape(-1)[0])
    dashboard_config_source = str(np.asarray(data.get("dashboard_config_source", np.asarray("missing"))).reshape(-1)[0])
    dashboard_config_label = dashboard_config_source
    if dashboard_config_source not in ("missing", "none"):
        try:
            dashboard_config_label = os.path.relpath(dashboard_config_source, ROOT_DIR)
        except ValueError:
            dashboard_config_label = dashboard_config_source
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    build_banner_html = (
        f'<span class="build-chip strong">html={html_lib.escape(DASHBOARD_VERSION)}</span>'
        f'<span class="build-chip">dash_data={html_lib.escape(dash_data_version)}</span>'
        f'<span class="build-chip">rgb={html_lib.escape(rgb_render_source)}</span>'
        f'<span class="build-chip">config_loaded={html_lib.escape(dashboard_config_label)}</span>'
        f'<span class="build-chip">generated={html_lib.escape(generated_utc)}</span>'
    )

    summary_parts = []
    for s in ms_summary:
        summary_parts.append(f'<span class="val">{s}</span>')
    if latest_er is not None:
        summary_parts.append(f'<span class="erank"><span class="erank-label">erank</span> <span class="erank-value">{latest_er:.3f}</span></span>')
    mask_summary_html = " ".join(summary_parts) if summary_parts else "(mask config unavailable)"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>JEPA Dashboard - {os.path.basename(session_dir)}</title>
  <meta name="jepa-dashboard-version" content="{DASHBOARD_VERSION}" />
  <style>
    body {{ margin: 14px; background: #f4f6fa; color: #0d1527; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .topbar {{ margin: 0 0 12px 2px; }}
    h1 {{ margin: 0; font-size: 24px; font-weight: 650; color: #0d1527; }}
    .version {{ color: #596275; font-size: 13px; font-weight: 600; margin-left: 8px; }}
    .build-banner {{ margin: 8px 0 10px 0; padding: 10px 12px; background: #fff7d6; border: 2px solid #f0b429; border-radius: 8px; color: #1f2937; font-size: 14px; font-weight: 650; }}
    .build-chip {{ display: inline-block; margin-right: 14px; }}
    .build-chip.strong {{ color: #9a3412; font-weight: 800; }}
    .mask-summary {{ margin: 0 0 18px 2px; padding: 14px 18px; background: #fff; border: 1px solid #d9deea; border-radius: 8px; font-size: 15px; line-height: 1.8; }}
    .mask-summary .val {{ color: #3a4055; margin-right: 20px; }}
    .mask-summary .erank {{ display: inline-block; margin-left: 8px; padding: 4px 14px; background: #1a1a2e; color: #fde725; border-radius: 6px; font-weight: 700; font-size: 17px; }}
    .mask-summary .erank-label {{ color: #7a7d8a; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .mask-summary .erank-value {{ font-size: 24px; margin-left: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(420px, 1fr)); gap: 12px; }}
    .controls {{ display:flex; flex-wrap:wrap; gap:8px; margin: 0 0 12px 2px; align-items:center; }}
    .controls input {{ width:88px; }}
    .controls input[type="checkbox"] {{ width:auto; }}
    .local-controls {{ margin: 2px 2px 8px 2px; padding: 6px; background: #f7f9ff; border: 1px solid #dde3f0; border-radius: 6px; }}
    .card-tools {{ display: flex; justify-content: flex-end; margin: 2px 2px 6px 2px; }}
    .save-panel {{ border: 1px solid #c7d2e8; background: #ffffff; color: #23304f; border-radius: 6px; padding: 4px 10px; font-size: 12px; font-weight: 600; cursor: pointer; }}
    .save-panel:hover {{ background: #eef3ff; }}
    .card {{ background: #fff; border: 1px solid #d9deea; border-radius: 10px; box-shadow: 0 1px 2px rgba(10,20,40,0.08); padding: 6px; overflow: hidden; }}
    .card-dummy {{ visibility: hidden; min-height: 340px; }}
    @media (max-width: 1120px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="topbar">
    <h1>JEPA Session Dashboard: {os.path.basename(session_dir)} <span class="version">{DASHBOARD_VERSION}</span></h1>
    <div class="build-banner">{build_banner_html}</div>
  </div>
  <div class="mask-summary">{mask_summary_html}</div>
  <div class="grid">{''.join(rendered)}</div>
  <script>
  function num(el) {{
    const v = parseFloat(el.value);
    return Number.isFinite(v) ? v : null;
  }}
  function clamp01(v) {{
    return Math.max(0.0, Math.min(1.0, v));
  }}
  function toArray(a) {{
    return Array.from(a || [], Number);
  }}
  function percentileRange(arr, loPct, hiPct) {{
    const vals = [];
    for (let i = 0; i < arr.length; i++) {{
      const v = Number(arr[i]);
      if (Number.isFinite(v)) vals.push(v);
    }}
    if (vals.length === 0) return [0.0, 1.0];
    vals.sort((a, b) => a - b);
    const q = (p) => {{
      const idx = (vals.length - 1) * Math.max(0.0, Math.min(1.0, p));
      const i0 = Math.floor(idx);
      const i1 = Math.ceil(idx);
      if (i0 === i1) return vals[i0];
      const t = idx - i0;
      return vals[i0] * (1.0 - t) + vals[i1] * t;
    }};
    let lo = q(loPct);
    let hi = q(hiPct);
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) {{
      lo = vals[0];
      hi = vals[vals.length - 1];
    }}
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [0.0, 1.0];
    return [lo, hi];
  }}
  function finitePercentileMask(x, y, z, loPct, hiPct) {{
    const n = Math.min(x.length, y.length, z.length);
    const xf = [], yf = [], zf = [];
    const finite = new Array(n);
    for (let i = 0; i < n; i++) {{
      const xx = Number(x[i]), yy = Number(y[i]), zz = Number(z[i]);
      const ok = Number.isFinite(xx) && Number.isFinite(yy) && Number.isFinite(zz);
      finite[i] = ok;
      if (ok) {{ xf.push(xx); yf.push(yy); zf.push(zz); }}
    }}
    const xr = percentileRange(xf, loPct, hiPct);
    const yr = percentileRange(yf, loPct, hiPct);
    const zr = percentileRange(zf, loPct, hiPct);
    const mask = new Array(n);
    for (let i = 0; i < n; i++) {{
      const xx = Number(x[i]), yy = Number(y[i]), zz = Number(z[i]);
      mask[i] = finite[i] && xx >= xr[0] && xx <= xr[1] && yy >= yr[0] && yy <= yr[1] && zz >= zr[0] && zz <= zr[1];
    }}
    let kept = 0;
    for (let i = 0; i < n; i++) if (mask[i]) kept++;
    const minKeep = Math.max(16, Math.min(256, Math.floor(0.001 * xf.length)));
    if (kept < minKeep) {{
      for (let i = 0; i < n; i++) mask[i] = finite[i];
    }}
    return {{ mask, xr, yr, zr }};
  }}
  function map255(v, lo, hi, invertColor) {{
    const den = Math.max(1e-12, hi - lo);
    let y = clamp01((Number(v) - lo) / den);
    if (invertColor) y = 1.0 - y;
    return Math.round(y * 255.0);
  }}
  function normalizeChannel(v, lo, hi, invertAxis) {{
    const den = Math.max(1e-12, hi - lo);
    let y = clamp01((Number(v) - lo) / den);
    return invertAxis ? 1.0 - y : y;
  }}
  function imageVisibleRange(baseImage, loPct, hiPct) {{
    const vals = [];
    (baseImage || []).forEach((row) => {{
      (row || []).forEach((pixel) => {{
        if (Array.isArray(pixel) || ArrayBuffer.isView(pixel)) {{
          for (let i = 0; i < Math.min(3, pixel.length); i++) {{
            const v = Number(pixel[i]);
            if (Number.isFinite(v)) vals.push(v);
          }}
        }}
      }});
    }});
    return percentileRange(vals, loPct, hiPct);
  }}
  function ensureTraceBases(gd) {{
    (gd.data || []).forEach((tr) => {{
      if (!tr) return;
      if (tr.type === "scatter3d") {{
        if (!tr._jepaBaseX) tr._jepaBaseX = toArray(tr.x);
        if (!tr._jepaBaseY) tr._jepaBaseY = toArray(tr.y);
        if (!tr._jepaBaseZCoord) tr._jepaBaseZCoord = toArray(tr.z);
      }} else if (tr.type === "image") {{
        if (!tr._jepaBaseImage) {{
          tr._jepaBaseImage = (tr.z || []).map((row) => Array.from(row || []).map((pixel) => {{
            if (Array.isArray(pixel) || ArrayBuffer.isView(pixel)) {{
              return [Number(pixel[0]) || 0, Number(pixel[1]) || 0, Number(pixel[2]) || 0];
            }}
            return [0, 0, 0];
          }}));
        }}
      }}
    }});
  }}
  function applyRangesForControl(controlId) {{
    const controls = document.querySelector('.local-controls[data-control-id="' + controlId + '"]');
    if (!controls || !window.Plotly) return;
    const card = controls.closest(".card");
    if (!card) return;
    const get = (k) => {{
      const el = controls.querySelector('input[data-k="' + k + '"]');
      return el ? num(el) : null;
    }};
    const xmin = get("xmin");
    const xmax = get("xmax");
    const ymin = get("ymin");
    const ymax = get("ymax");
    const zmin = get("zmin");
    const zmax = get("zmax");
    const colorLow = get("color_low");
    const colorHigh = get("color_high");
    const invertX = !!(controls.querySelector('input[data-k="invert_x"]') || {{}}).checked;
    const invertY = !!(controls.querySelector('input[data-k="invert_y"]') || {{}}).checked;
    const invertZ = !!(controls.querySelector('input[data-k="invert_z"]') || {{}}).checked;
    const invertColor = !!(controls.querySelector('input[data-k="invert_color"]') || {{}}).checked;
    const plots = card.querySelectorAll(".js-plotly-plot");
    plots.forEach((gd) => {{
      ensureTraceBases(gd);
      (gd.data || []).forEach((tr, i) => {{
        if (!tr) return;
        if (tr.type === "image") {{
          const autoRange = imageVisibleRange(tr._jepaBaseImage, 0.01, 0.99);
          const lo = colorLow !== null ? colorLow : autoRange[0];
          const hi = colorHigh !== null ? colorHigh : autoRange[1];
          const recolored = tr._jepaBaseImage.map((row) => row.map((pixel) => {{
            let r = invertX ? 255.0 - pixel[0] : pixel[0];
            let g = invertY ? 255.0 - pixel[1] : pixel[1];
            let b = invertZ ? 255.0 - pixel[2] : pixel[2];
            return [map255(r, lo, hi, invertColor), map255(g, lo, hi, invertColor), map255(b, lo, hi, invertColor)];
          }}));
          Plotly.restyle(gd, {{ z: [recolored] }}, [i]);
          const relImg = {{}};
          if (xmin !== null || xmax !== null) relImg["xaxis.range"] = [xmin !== null ? xmin : 0, xmax !== null ? xmax : recolored[0].length];
          if (ymin !== null || ymax !== null) relImg["yaxis.range"] = [ymin !== null ? ymin : 0, ymax !== null ? ymax : recolored.length];
          if (Object.keys(relImg).length > 0) Plotly.relayout(gd, relImg);
          Plotly.redraw(gd);
          return;
        }}
        if (tr.type !== "scatter3d") return;
        const bx = tr._jepaBaseX;
        const by = tr._jepaBaseY;
        const bz = tr._jepaBaseZCoord;
        const fx = invertX ? bx.map((v) => -v) : bx.slice();
        const fy = invertY ? by.map((v) => -v) : by.slice();
        const fz = invertZ ? bz.map((v) => -v) : bz.slice();
        const filt = finitePercentileMask(fx, fy, fz, 0.01, 0.99);
        const lo = colorLow !== null ? colorLow : 0.0;
        const hi = colorHigh !== null ? colorHigh : 255.0;
        const n = Math.min(bx.length, by.length, bz.length);
        const x = [], y = [], z = [], colors = [];
        const maxScatterPoints = 50000;
        let keptTotal = 0;
        for (let k = 0; k < n; k++) if (filt.mask[k]) keptTotal++;
        const stride = keptTotal > maxScatterPoints ? Math.ceil(keptTotal / maxScatterPoints) : 1;
        let keptSeen = 0;
        for (let k = 0; k < n; k++) {{
          if (!filt.mask[k]) continue;
          if ((keptSeen % stride) !== 0) {{ keptSeen++; continue; }}
          keptSeen++;
          x.push(fx[k]); y.push(fy[k]); z.push(fz[k]);
          const r0 = normalizeChannel(fx[k], filt.xr[0], filt.xr[1], false) * 255.0;
          const g0 = normalizeChannel(fy[k], filt.yr[0], filt.yr[1], false) * 255.0;
          const b0 = normalizeChannel(fz[k], filt.zr[0], filt.zr[1], false) * 255.0;
          colors.push(`rgb(${{map255(r0, lo, hi, invertColor)}},${{map255(g0, lo, hi, invertColor)}},${{map255(b0, lo, hi, invertColor)}})`);
        }}
        Plotly.restyle(gd, {{ x: [x], y: [y], z: [z], "marker.color": [colors] }}, [i]);
        const relayoutOpts = {{
          "scene.camera.projection.type": "orthographic",
          "scene.aspectmode": "cube",
        }};
        relayoutOpts["scene.xaxis.range"] = [xmin !== null ? xmin : filt.xr[0], xmax !== null ? xmax : filt.xr[1]];
        relayoutOpts["scene.yaxis.range"] = [ymin !== null ? ymin : filt.yr[0], ymax !== null ? ymax : filt.yr[1]];
        relayoutOpts["scene.zaxis.range"] = [zmin !== null ? zmin : filt.zr[0], zmax !== null ? zmax : filt.zr[1]];
        Plotly.relayout(gd, relayoutOpts);
        Plotly.redraw(gd);
      }});
    }});
  }}
  function initControlDefaults() {{
    document.querySelectorAll('.card .js-plotly-plot').forEach(ensureTraceBases);
  }}
  function bindLocalControls() {{
    document.querySelectorAll(".apply-local").forEach((btn) => {{
      btn.addEventListener("click", () => applyRangesForControl(btn.getAttribute("data-control-id")));
    }});
    document.querySelectorAll('.local-controls input').forEach((input) => {{
      input.addEventListener("change", () => {{
        const controls = input.closest(".local-controls");
        if (controls) applyRangesForControl(controls.getAttribute("data-control-id"));
      }});
    }});
    initControlDefaults();
  }}
  window.JEPADashboardControls = {{
    applyRangesForControl,
    applyRangesForGroup: applyRangesForControl,
    bindLocalControls,
  }};
  window.applyRangesForControl = applyRangesForControl;
  window.applyRangesForGroup = applyRangesForControl;
  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", bindLocalControls);
  }} else {{
    bindLocalControls();
  }}
	  function _safeFileName(name) {{
	    return String(name || "panel")
	      .toLowerCase()
	      .replace(/[^a-z0-9]+/g, "_")
	      .replace(/^_+|_+$/g, "") || "panel";
	  }}
	  function _axisLayoutKey(axisRef) {{
	    const ref = String(axisRef || "x");
	    const prefix = ref.charAt(0) === "y" ? "yaxis" : "xaxis";
	    const suffix = ref.length > 1 ? ref.slice(1) : "";
	    return prefix + suffix;
	  }}
	  function _traceRasterShape(trace) {{
	    if (!trace || (trace.type !== "image" && trace.type !== "heatmap")) return null;
	    const z = trace.z || [];
	    const h = Array.isArray(z) || ArrayBuffer.isView(z) ? z.length : 0;
	    const row0 = h > 0 ? z[0] : [];
	    const w = Array.isArray(row0) || ArrayBuffer.isView(row0) ? row0.length : 0;
	    return h > 0 && w > 0 ? {{ h, w }} : null;
	  }}
	  function _axisDomain(layout, key) {{
	    const axis = (layout || {{}})[key] || {{}};
	    const domain = axis.domain || [0, 1];
	    const lo = Number(domain[0]);
	    const hi = Number(domain[1]);
	    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return 1.0;
	    return Math.max(1e-6, hi - lo);
	  }}
	  function _dominantRasterShape(gd) {{
	    let best = null;
	    (gd.data || []).forEach((trace) => {{
	      const shape = _traceRasterShape(trace);
	      if (!shape) return;
	      const area = shape.w * shape.h;
	      if (!best || area > best.area) best = {{ ...shape, area }};
	    }});
	    return best;
	  }}
	  function _dominantRasterTrace(gd) {{
	    let best = null;
	    (gd.data || []).forEach((trace) => {{
	      const shape = _traceRasterShape(trace);
	      if (!shape) return;
	      const area = shape.w * shape.h;
	      if (!best || area > best.area) best = {{ trace, shape, area }};
	    }});
	    return best;
	  }}
	  function _finiteRange2d(z, fallbackLo, fallbackHi) {{
	    let lo = Number(fallbackLo);
	    let hi = Number(fallbackHi);
	    if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) return [lo, hi];
	    lo = Infinity;
	    hi = -Infinity;
	    (z || []).forEach((row) => {{
	      Array.from(row || []).forEach((v) => {{
	        const x = Number(v);
	        if (!Number.isFinite(x)) return;
	        lo = Math.min(lo, x);
	        hi = Math.max(hi, x);
	      }});
	    }});
	    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [0, 1];
	    return [lo, hi];
	  }}
	  function _toByte(v, lo = 0, hi = 255) {{
	    const x = Number(v);
	    if (!Number.isFinite(x)) return 0;
	    const y = (x - lo) / Math.max(1e-12, hi - lo);
	    return Math.max(0, Math.min(255, Math.round(y * 255)));
	  }}
	  function _downloadRawRasterPng(gd, title) {{
	    const item = _dominantRasterTrace(gd);
	    if (!item) return false;
	    const trace = item.trace;
	    const z = trace.z || [];
	    const width = item.shape.w;
	    const height = item.shape.h;
	    const canvas = document.createElement("canvas");
	    canvas.width = width;
	    canvas.height = height;
	    const ctx = canvas.getContext("2d");
	    if (!ctx) return false;
	    const image = ctx.createImageData(width, height);
	    const data = image.data;
	    const scalarRange = trace.type === "heatmap" ? _finiteRange2d(z, trace.zmin, trace.zmax) : [0, 255];
	    for (let y = 0; y < height; y++) {{
	      const row = z[y] || [];
	      for (let x = 0; x < width; x++) {{
	        const pixel = row[x];
	        const idx = 4 * (y * width + x);
	        if (Array.isArray(pixel) || ArrayBuffer.isView(pixel)) {{
	          data[idx] = _toByte(pixel[0]);
	          data[idx + 1] = _toByte(pixel[1]);
	          data[idx + 2] = _toByte(pixel[2]);
	          data[idx + 3] = pixel.length > 3 ? _toByte(pixel[3]) : 255;
	        }} else {{
	          const g = _toByte(pixel, scalarRange[0], scalarRange[1]);
	          data[idx] = g;
	          data[idx + 1] = g;
	          data[idx + 2] = g;
	          data[idx + 3] = Number.isFinite(Number(pixel)) ? 255 : 0;
	        }}
	      }}
	    }}
	    ctx.putImageData(image, 0, 0);
	    const a = document.createElement("a");
	    a.href = canvas.toDataURL("image/png");
	    a.download = _safeFileName(title) + ".png";
	    document.body.appendChild(a);
	    a.click();
	    a.remove();
	    return true;
	  }}
	  function _panelExportSize(gd) {{
	    const layout = gd.layout || {{}};
	    const margin = layout.margin || {{}};
	    const ml = Number.isFinite(Number(margin.l)) ? Number(margin.l) : 80;
	    const mr = Number.isFinite(Number(margin.r)) ? Number(margin.r) : 40;
	    const mt = Number.isFinite(Number(margin.t)) ? Number(margin.t) : 80;
	    const mb = Number.isFinite(Number(margin.b)) ? Number(margin.b) : 50;
	    let width = Math.max(1200, gd.clientWidth || 1200);
	    let height = Math.max(900, gd.clientHeight || 900);
	    let plotW = 0;
	    let plotH = 0;
	    (gd.data || []).forEach((trace) => {{
	      const shape = _traceRasterShape(trace);
	      if (!shape) return;
	      const xDomain = _axisDomain(layout, _axisLayoutKey(trace.xaxis || "x"));
	      const yDomain = _axisDomain(layout, _axisLayoutKey(trace.yaxis || "y"));
	      plotW = Math.max(plotW, shape.w / xDomain);
	      plotH = Math.max(plotH, shape.h / yDomain);
	    }});
	    const nativeWidth = plotW > 0 ? Math.ceil(plotW + ml + mr) : 0;
	    const nativeHeight = plotH > 0 ? Math.ceil(plotH + mt + mb) : 0;
	    const rasterShape = _dominantRasterShape(gd);
	    if (rasterShape) {{
	      const contentW = Math.max(nativeWidth > 0 ? nativeWidth - ml - mr : 0, gd.clientWidth || 0, 1200);
	      const contentH = Math.max(1, Math.ceil(contentW * rasterShape.h / Math.max(1, rasterShape.w)));
	      width = Math.max(width, Math.ceil(contentW + ml + mr));
	      height = Math.max(gd.clientHeight || 0, Math.ceil(contentH + mt + mb));
	    }}
	    let scale = Math.max(
	      4,
	      nativeWidth > 0 ? nativeWidth / width : 0,
	      nativeHeight > 0 ? nativeHeight / height : 0,
	    );
	    const maxDim = 16384;
	    const maxPixels = 120000000;
	    scale = Math.min(scale, maxDim / width, maxDim / height);
	    scale = Math.min(scale, Math.sqrt(maxPixels / Math.max(1, width * height)));
	    scale = Math.max(1, scale);
	    if (plotW <= 0 || plotH <= 0) {{
	      return {{ width, height, scale: Math.max(2, scale) }};
	    }}
	    return {{ width, height, scale }};
	  }}
	  async function savePanelPng(btn) {{
	    const card = btn.closest(".card");
	    const gd = card ? card.querySelector(".js-plotly-plot") : null;
	    if (!gd || !window.Plotly || typeof window.Plotly.toImage !== "function") {{
	      alert("Panel export unavailable for this card.");
	      return;
	    }}
	    const title = btn.getAttribute("data-panel-title") || "panel";
	    const exportSize = _panelExportSize(gd);
	    try {{
	      if (_downloadRawRasterPng(gd, title)) return;
	      const dataUrl = await window.Plotly.toImage(
	        gd,
	        {{ format: "png", width: exportSize.width, height: exportSize.height, scale: exportSize.scale }}
	      );
	      const a = document.createElement("a");
	      a.href = dataUrl;
	      a.download = _safeFileName(title) + ".png";
	      document.body.appendChild(a);
	      a.click();
	      a.remove();
	    }} catch (err) {{
	      console.error("savePanelPng failed", err);
	      alert("Failed to save panel PNG.");
	    }}
	  }}
	  document.querySelectorAll("button.save-panel").forEach((btn) => {{
	    btn.addEventListener("click", () => savePanelPng(btn));
	  }});
	  </script>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    # Mandatory diagnostics summary output.
    print(f"dashboard_plot_summary_begin session={session_dir}")
    print(f"dashboard_plot_summary_cards={len(cards)}")
    print(
        f"dashboard_plot_item=Active Loss Terms: {'ok' if n > 0 else 'empty'} "
        f"(points={n} active={','.join(name for name, _, _, _ in active_loss_terms)})"
    )
    er_n = int(min(er_x.size, er_y.size)) if (er_x.size and er_y.size) else 0
    print(f"dashboard_plot_item=Effective Rank: {'ok' if er_n > 0 else 'empty'} (points={er_n})")
    if "scale_probe_sensitivity_maps" in data.files:
        sp = np.asarray(data["scale_probe_sensitivity_maps"], dtype=np.float32)
        src = str(np.asarray(data["scale_probe_source"]).reshape(-1)[0]) if "scale_probe_source" in data.files and np.asarray(data["scale_probe_source"]).size else "unknown"
        print(f"dashboard_plot_item=CDD Scale Response: ok (source={src} shape={tuple(sp.shape)})")
    else:
        print("dashboard_plot_item=CDD Scale Response: empty (missing *_scale_response.pt)")
    for name, stem in (("Context", "context"), ("Masked Predict", "masked_pred"), ("Predict", "pred"), ("Target", "gt")):
        pca_arr = np.asarray(data[f"{stem}_pca3d"], dtype=np.float32)
        um_arr = np.asarray(data[f"{stem}_umap3d"], dtype=np.float32)
        print(f"dashboard_plot_item={name} PCA Array Shape: shape={tuple(pca_arr.shape)}")
        print(f"dashboard_plot_item={name} UMAP Array Shape: shape={tuple(um_arr.shape)}")
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
        (str(data["visit_heatmap_kind"]) if "visit_heatmap_kind" in data.files else "Visit Frequency Heatmap", "visit_heatmap"),
    ):
        finite = int(np.isfinite(np.asarray(data[key])).sum())
        print(f"dashboard_plot_item={title}: {'ok' if finite > 0 else 'empty'} (finite_pixels={finite})")
    print(f"dashboard_plot_summary_end session={session_dir} out_html={out_html}")

    data.close()
    return out_html


def plot_dash(session_dir: str, overwrite: bool = False) -> str:
    return plot_dash_html(session_dir, overwrite=overwrite)


def _preferred_html_for_export(session_dir: str, fallback_html: str) -> str:
    def _inline_masking_iframe_srcs(page_html: str) -> str:
        pattern = re.compile(r"""src=(["'])(masking_demo_[^"']+\.html)\1""")

        def _repl(match: re.Match[str]) -> str:
            demo_fn = match.group(2)
            demo_path = os.path.join(session_dir, demo_fn)
            if not os.path.exists(demo_path):
                return match.group(0)
            with open(demo_path, "r", encoding="utf-8") as demo_f:
                srcdoc = html_lib.escape(demo_f.read(), quote=True)
            return f'srcdoc="{srcdoc}"'

        return pattern.sub(_repl, page_html)

    with open(fallback_html, "r", encoding="utf-8") as f:
        html_raw = f.read()
    html_inlined = _inline_masking_iframe_srcs(html_raw)
    if 'data-mask-demo-auto="true"' in html_raw:
        if html_inlined == html_raw:
            return fallback_html
        out_html = os.path.join(session_dir, "dashboard_with_masking_demo.html")
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html_inlined)
        return out_html

    page_html = html_inlined
    demo_files = sorted([fn for fn in os.listdir(session_dir) if fn.startswith("masking_demo_") and fn.endswith(".html")])
    if not demo_files:
        return fallback_html
    parts = ["<hr/>", "<h2 style='font-family:sans-serif;margin:16px 0 8px 0;'>Masking Demo Panels</h2>"]
    for fn in demo_files:
        demo_path = os.path.join(session_dir, fn)
        if not os.path.exists(demo_path):
            continue
        with open(demo_path, "r", encoding="utf-8") as demo_f:
            srcdoc = html_lib.escape(demo_f.read(), quote=True)
        parts.append(
            f"<div style='margin:10px 0;'><div style='font-family:sans-serif;font-size:14px;margin:4px 0;'>{fn}</div>"
            f"<iframe srcdoc=\"{srcdoc}\" style='width:100%;height:980px;border:1px solid #ddd;border-radius:6px;'></iframe></div>"
        )
    page_html = page_html.replace("</body>", "\n" + "\n".join(parts) + "\n</body>")
    out_html = os.path.join(session_dir, "dashboard_with_masking_demo.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(page_html)
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
    root_name = os.path.basename(os.path.abspath(args.sessions_dir))
    if os.path.exists(os.path.join(args.sessions_dir, "inference_outputs.pt")):
        session_items = [(root_name, args.sessions_dir)]
    else:
        session_items = [
            (name, os.path.join(args.sessions_dir, name))
            for name in sorted(os.listdir(args.sessions_dir))
            if os.path.isdir(os.path.join(args.sessions_dir, name))
        ]
    for name, session_dir in session_items:
        print("=" * 72)
        print(f"dashboard_session_begin={session_dir}")
        dash_html_path = os.path.join(session_dir, "dashboard.html")
        export_path = os.path.join(export_dir, f"{name.replace('/', '_')}.html")
        # Plain exists skip — mirrors movie PNG behavior: if both files are
        # already there and we're not forcing overwrite/reset, skip.
        if (not args.overwrite) and (not args.reset) and os.path.exists(dash_html_path) and os.path.exists(export_path):
            print(
                f"skip_dashboard_exists={session_dir} "
                f"session_html={dash_html_path} export_html={export_path}"
            )
            skipped += 1
            print(f"dashboard_session_end={session_dir}")
            continue
        inf_path = os.path.join(session_dir, "inference_outputs.pt")
        if not os.path.exists(inf_path):
            print(f"skip_no_inference={session_dir}")
            skipped += 1
            continue
        if args.stage == "plot":
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
