from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch


DASHBOARD_VERSION = "aligned-loss-layout-v3"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    "pyramid_mask_cube",
}


def _find_session_config(session_dir: str) -> str | None:
    name = os.path.basename(os.path.abspath(session_dir))
    env_cfg_dir = os.environ.get("SESSION_DASH_CONFIG_DIR", "").strip()
    candidates = [
        os.path.join(session_dir, "config_used.json"),
    ]
    if env_cfg_dir:
        candidates.append(os.path.join(env_cfg_dir, f"{name}.json"))
    candidates.extend([
        os.path.join(ROOT_DIR, "configs", "experiments", f"{name}.json"),
        os.path.join(ROOT_DIR, "configs", f"{name}.json"),
    ])
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def _generate_masking_diagnostic_for_dashboard(session_dir: str) -> str | None:
    cfg_path = _find_session_config(session_dir)
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
    # NaN sentinels → black (no-data marker)
    clipped[~fin] = 0.0
    rgb_flat = np.clip(np.round(clipped * 255.0), 0, 255).astype(np.uint8)
    rgb = rgb_flat.reshape(h, w, 3)
    return rgb, rgb_flat


def _xyz_from_feature_map(feat: np.ndarray) -> np.ndarray:
    """Fallback embedding: flatten CHW feature map to N x 3 via first channels."""
    arr = np.asarray(feat, dtype=np.float32)
    if arr.ndim == 4:
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
    xyz = np.where(np.isfinite(xyz), xyz, 0.0).astype(np.float32)
    return xyz


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

    visit_path = os.path.join(session_dir, "visited_target_frequency_canonical.npy")
    if not os.path.exists(visit_path):
        visit_path = os.path.join(session_dir, "visited_target_frequency.npy")
    visit_heatmap = np.asarray(np.load(visit_path), dtype=np.float32) if os.path.exists(visit_path) else np.zeros((h, w), dtype=np.float32)
    if visit_heatmap.shape != (h, w):
        visit_heatmap = np.zeros((h, w), dtype=np.float32)

    # Load first-sample pyramid mask cube (S,H,W), save a canonical artifact in session dir.
    mask_cube = None
    pmt_path = os.path.join(session_dir, "pyramid_mask_token.npy")
    if os.path.exists(pmt_path):
        arr = np.asarray(np.load(pmt_path), dtype=np.float32)
        # Expected saved inference artifact shape: B,S,H,W
        if arr.ndim == 4 and arr.shape[0] > 0:
            mask_cube = arr[0]
    if mask_cube is None:
        tok = outputs.get("dip_field_per_channel", outputs.get("pyramid_mask_token"))
        if tok is not None:
            tok = _to_np(tok).astype(np.float32)
            if tok.ndim == 4 and tok.shape[0] > 0:
                mask_cube = tok[0]
    if mask_cube is None:
        tok = outputs.get("mask_cube")
        if tok is not None:
            tok = _to_np(tok).astype(np.float32)
            if tok.ndim == 5 and tok.shape[0] > 0 and tok.shape[1] > 0:
                mask_cube = tok[0, 0]
    if mask_cube is None:
        mask_cube = np.zeros((1, h, w), dtype=np.float32)
    else:
        mask_cube = np.where(np.isfinite(mask_cube), mask_cube, 0.0).astype(np.float32)
    np.save(os.path.join(session_dir, "example_pyramid_mask_cube.npy"), mask_cube.astype(np.float32, copy=False))

    # Load precomputed PCA/UMAP artifacts saved by training-time pipeline.
    results_dir = os.path.join(session_dir, "results")
    has_results_dir = os.path.isdir(results_dir)
    has_predict_branch = has_results_dir and _has_required_branch_artifacts(results_dir, "predict")
    has_target_branch = has_results_dir and _has_required_branch_artifacts(results_dir, "target")
    fallback_mode = not (has_predict_branch and has_target_branch)
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
        if fallback_mode:
            if prefix_out == "pred":
                src_map = outputs.get("pred_map", outputs.get("context_map"))
            elif prefix_out == "gt":
                src_map = outputs.get("gt_map", outputs.get("pred_map"))
            else:
                src_map = outputs.get("context_map", outputs.get("pred_map"))
            if src_map is None:
                xyz = np.zeros((h_lat * w_lat, 3), dtype=np.float32)
            else:
                xyz = _xyz_from_feature_map(_to_np(src_map))
                if xyz.shape[0] != h_lat * w_lat:
                    xyz = np.zeros((h_lat * w_lat, 3), dtype=np.float32)
            pca = xyz
            um = xyz
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
            continue
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
    loss_sigreg, loss_var, loss_cov = [], [], []
    loss_symmetric, weighted_symmetric = [], []
    weighted_jepa, weighted_sigreg, weighted_var, weighted_cov = [], [], [], []
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ep = float(row.get("epoch", "nan"))
                    ba = float(row.get("batch", row.get("step", "nan")))
                    gs = float(row.get("global_step", "nan"))
                    tl = float(row.get("total_loss", row.get("loss_total", row.get("loss", "nan"))))
                    jl = float(row.get("loss_jepa", row.get("jepa_loss", row.get("loss_mse", "nan"))))
                    sl = float(row.get("loss_sigreg", "nan"))
                    syml = float(row.get("loss_symmetric", "nan"))
                    vl = float(row.get("loss_var", "nan"))
                    cl = float(row.get("loss_cov", "nan"))
                    wj = float(row.get("weighted_jepa", row.get("weighted_mse", "nan")))
                    ws = float(row.get("weighted_sigreg", "nan"))
                    wsym = float(row.get("weighted_symmetric", "nan"))
                    wv = float(row.get("weighted_var", "nan"))
                    wc = float(row.get("weighted_cov", "nan"))
                except Exception:
                    continue
                if np.isfinite(gs):
                    loss_x.append(gs)
                elif np.isfinite(ep) and np.isfinite(ba):
                    loss_x.append(ep + 0.001 * ba)
                else:
                    continue
                if np.isfinite(tl) and np.isfinite(jl):
                    loss_total.append(tl)
                    loss_jepa.append(jl)
                    loss_sigreg.append(sl if np.isfinite(sl) else np.nan)
                    loss_symmetric.append(syml if np.isfinite(syml) else np.nan)
                    loss_var.append(vl if np.isfinite(vl) else np.nan)
                    loss_cov.append(cl if np.isfinite(cl) else np.nan)
                    weighted_jepa.append(wj if np.isfinite(wj) else np.nan)
                    weighted_sigreg.append(ws if np.isfinite(ws) else np.nan)
                    weighted_symmetric.append(wsym if np.isfinite(wsym) else np.nan)
                    weighted_var.append(wv if np.isfinite(wv) else np.nan)
                    weighted_cov.append(wc if np.isfinite(wc) else np.nan)
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
    def _rd(branch: str, key: str, default: float = np.nan) -> float:
        try:
            return float(rank_diag.get(branch, {}).get(key, default))
        except Exception:
            return float(default)
    rd_pred_gt_erank_ratio = float(rank_diag.get("pred_gt_erank_ratio", np.nan)) if isinstance(rank_diag, dict) else np.nan

    # Compute mask sizes per sigma from config
    cfg_used_path = os.path.join(session_dir, "config_used.json")
    mask_sigma_names: list[str] = []
    mask_sigma_sizes: list[int] = []
    mask_config_summary: list[str] = []
    if os.path.exists(cfg_used_path):
        try:
            with open(cfg_used_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            mc = cfg.get("model", {})
            _sigmas = mc.get("sigmas", [2, 4, 8, 16])
            _ms = float(mc.get("mask_size_scaling", 1.0))
            _mb = int(mc.get("mask_box_size", 0))
            _mf = float(mc.get("active_target_fraction", mc.get("mask_fraction", 1.0)))
            _ps = int(mc.get("patch_size", 3))
            _symmetric = bool(mc.get("use_symmetric_feature_loss", False))
            _norm_l2 = bool(mc.get("normalize_loss_l2", mc.get("normalize_loss", False)))
            _sampling = str(mc.get("target_sampling_mode", "grid"))
            _enc = str(mc.get("model_key", mc.get("encoder_type", "unknown")))
            mask_config_summary = [
                f"encoder={_enc}",
                f"mask_size_scaling={_ms}",
                f"mask_box_size={_mb}",
                f"active_target_fraction={_mf}",
                f"patch_size={_ps}",
                f"target_sampling={_sampling}",
                f"use_symmetric_feature_loss={_symmetric}",
                f"normalize_loss_l2={_norm_l2}",
            ]
            for s in _sigmas:
                box = max(_ps, round(float(s) * _ms + _mb))
                mask_sigma_names.append(f"σ={s}")
                mask_sigma_sizes.append(box)
            # Add overall summary
            _computed = ", ".join(f"σ={s}→{max(_ps, round(float(s)*_ms+_mb))}px" for s in _sigmas)
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
        loss_sigreg=np.asarray(loss_sigreg, dtype=np.float32),
        loss_symmetric=np.asarray(loss_symmetric, dtype=np.float32),
        loss_var=np.asarray(loss_var, dtype=np.float32),
        loss_cov=np.asarray(loss_cov, dtype=np.float32),
        weighted_jepa=np.asarray(weighted_jepa, dtype=np.float32),
        weighted_sigreg=np.asarray(weighted_sigreg, dtype=np.float32),
        weighted_symmetric=np.asarray(weighted_symmetric, dtype=np.float32),
        weighted_var=np.asarray(weighted_var, dtype=np.float32),
        weighted_cov=np.asarray(weighted_cov, dtype=np.float32),
        effective_rank_x=np.asarray(effective_rank_x, dtype=np.float64),
        effective_rank_y=np.asarray(effective_rank_y, dtype=np.float32),
        rank_context_erank=np.asarray([_rd("context", "erank")], dtype=np.float32),
        rank_pred_erank=np.asarray([_rd("pred", "erank")], dtype=np.float32),
        rank_gt_erank=np.asarray([_rd("gt", "erank")], dtype=np.float32),
        rank_context_pr=np.asarray([_rd("context", "participation_rank")], dtype=np.float32),
        rank_pred_pr=np.asarray([_rd("pred", "participation_rank")], dtype=np.float32),
        rank_gt_pr=np.asarray([_rd("gt", "participation_rank")], dtype=np.float32),
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
        rank_pred_gt_erank_ratio=np.asarray([rd_pred_gt_erank_ratio], dtype=np.float32),
        pyramid_mask_cube=mask_cube.astype(np.float32),
        mask_sigma_names=np.array(mask_sigma_names, dtype=str),
        mask_sigma_sizes=np.asarray(mask_sigma_sizes, dtype=np.int32),
        mask_config_summary=np.array(mask_config_summary, dtype=str),
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
    model_mode = "unknown"
    cfg_used_path = os.path.join(session_dir, "config_used.json")
    if os.path.exists(cfg_used_path):
        try:
            import json

            with open(cfg_used_path, "r", encoding="utf-8") as f:
                cfg_used = json.load(f)
            model_mode = str(cfg_used.get("model", {}).get("mode", "unknown")).lower()
        except Exception:
            model_mode = "unknown"

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
            # Keep HTML size sane while still showing dense point clouds.
            if n > 65536:
                step = int(np.ceil(n / 65536.0))
                pts = pts[::step]
                if rgb.ndim == 2:
                    rgb = rgb[::step]
            rendered_n = int(pts.shape[0])
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
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
                    marker=dict(size=2, opacity=0.82, color=colors, showscale=False),
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
            xx = np.asarray([0], dtype=np.int32)
            yy = np.asarray([0], dtype=np.int32)
            zz = np.asarray([0], dtype=np.int32)
            vv = np.asarray([0.0], dtype=np.float32)
        else:
            vv = c[zz, yy, xx]
        fig = go.Figure(
            [
                go.Scatter3d(
                    x=xx.astype(np.float32),
                    y=yy.astype(np.float32),
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

    loss_x = np.asarray(data["loss_x"], dtype=np.float32) if "loss_x" in data.files else np.asarray([], dtype=np.float32)
    loss_total = np.asarray(data["loss_total"], dtype=np.float32) if "loss_total" in data.files else np.asarray([], dtype=np.float32)
    loss_jepa = np.asarray(data["loss_jepa"], dtype=np.float32) if "loss_jepa" in data.files else np.asarray([], dtype=np.float32)
    loss_sigreg = np.asarray(data["loss_sigreg"], dtype=np.float32) if "loss_sigreg" in data.files else np.asarray([], dtype=np.float32)
    loss_symmetric = np.asarray(data["loss_symmetric"], dtype=np.float32) if "loss_symmetric" in data.files else np.asarray([], dtype=np.float32)
    loss_var = np.asarray(data["loss_var"], dtype=np.float32) if "loss_var" in data.files else np.asarray([], dtype=np.float32)
    loss_cov = np.asarray(data["loss_cov"], dtype=np.float32) if "loss_cov" in data.files else np.asarray([], dtype=np.float32)
    weighted_jepa = np.asarray(data["weighted_jepa"], dtype=np.float32) if "weighted_jepa" in data.files else np.asarray([], dtype=np.float32)
    weighted_sigreg = np.asarray(data["weighted_sigreg"], dtype=np.float32) if "weighted_sigreg" in data.files else np.asarray([], dtype=np.float32)
    weighted_symmetric = np.asarray(data["weighted_symmetric"], dtype=np.float32) if "weighted_symmetric" in data.files else np.asarray([], dtype=np.float32)
    weighted_var = np.asarray(data["weighted_var"], dtype=np.float32) if "weighted_var" in data.files else np.asarray([], dtype=np.float32)
    weighted_cov = np.asarray(data["weighted_cov"], dtype=np.float32) if "weighted_cov" in data.files else np.asarray([], dtype=np.float32)
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
    n = min(loss_x.size, loss_total.size, loss_jepa.size) if (loss_x.size and loss_total.size and loss_jepa.size) else 0
    loss_terms = (
        ("jepa", "mse_loss_weight", loss_jepa, weighted_jepa, "#636EFA"),
        ("sigreg", "sigreg_weight", loss_sigreg, weighted_sigreg, "#EF553B"),
        ("symmetric", "symmetric_feature_loss_weight", loss_symmetric, weighted_symmetric, "#00CC96"),
        ("var", "vicreg_var_weight", loss_var, weighted_var, "#AB63FA"),
        ("cov", "vicreg_cov_weight", loss_cov, weighted_cov, "#FFA15A"),
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
            if raw_arr.size < n or weighted_arr.size < n:
                continue
            weighted = weighted_arr[:n]
            weight = loss_weights.get(weight_key)
            configured_active = weight is not None and abs(float(weight)) > 1e-12
            observed_active = np.isfinite(weighted).any() and np.nanmax(np.abs(weighted)) > 1e-12
            if configured_active or observed_active:
                active_loss_terms.append((name, raw_arr[:n], weighted, color))

    def _add_loss_trace(fig: go.Figure, *, x: np.ndarray, y: np.ndarray, name: str, color: str) -> None:
        values = np.where(np.isfinite(y), y, np.nan).astype(np.float32)
        fig.add_trace(
            go.Scattergl(
                x=x, y=values, mode="lines", name=name,
                line=dict(width=1, color=color), opacity=0.12, showlegend=False,
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
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    fig_loss_components.update_xaxes(title_text="global_step")
    fig_loss_components.update_yaxes(title_text="loss")
    fig_weighted_components = go.Figure()
    if n > 0:
        lx = loss_x[:n]
        for name, _, weighted_arr, color in active_loss_terms:
            _add_loss_trace(fig_weighted_components, x=lx, y=weighted_arr, name=f"weighted_{name}", color=color)
        _add_loss_trace(fig_weighted_components, x=lx, y=loss_total[:n], name="total_loss", color="#222222")
    fig_weighted_components.update_layout(
        template="plotly_white",
        title={"text": "Active Loss Terms (Weighted into total_loss)", "x": 0.02},
        margin=dict(l=42, r=8, t=36, b=36),
        height=330,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    fig_weighted_components.update_xaxes(title_text="global_step")
    fig_weighted_components.update_yaxes(title_text="weighted contribution")
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
    def _scalar(name: str) -> float:
        if name in data.files:
            arr = np.asarray(data[name]).reshape(-1)
            if arr.size > 0 and np.isfinite(arr[0]):
                return float(arr[0])
        return float("nan")

    rank_branches = ["context", "pred", "gt"]
    rank_erank = [_scalar("rank_context_erank"), _scalar("rank_pred_erank"), _scalar("rank_gt_erank")]
    rank_pr = [_scalar("rank_context_pr"), _scalar("rank_pred_pr"), _scalar("rank_gt_pr")]
    rank_dead = [_scalar("rank_context_dead"), _scalar("rank_pred_dead"), _scalar("rank_gt_dead")]
    rank_top1 = [_scalar("rank_context_top1"), _scalar("rank_pred_top1"), _scalar("rank_gt_top1")]
    rank_top4 = [_scalar("rank_context_top4"), _scalar("rank_pred_top4"), _scalar("rank_gt_top4")]
    rank_top8 = [_scalar("rank_context_top8"), _scalar("rank_pred_top8"), _scalar("rank_gt_top8")]
    rank_ratio = _scalar("rank_pred_gt_erank_ratio")

    fig_rank_diag = go.Figure()
    fig_rank_diag.add_trace(go.Bar(name="erank", x=rank_branches, y=rank_erank))
    fig_rank_diag.add_trace(go.Bar(name="participation_rank", x=rank_branches, y=rank_pr))
    fig_rank_diag.add_trace(go.Bar(name="dead_channel_fraction", x=rank_branches, y=rank_dead))
    subtitle = ""
    if np.isfinite(rank_ratio):
        subtitle = f" (pred/gt erank ratio={rank_ratio:.3f})"
    fig_rank_diag.update_layout(
        barmode="group",
        template="plotly_white",
        title={"text": f"Rank Diagnostics{subtitle}", "x": 0.02},
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
        title={"text": "Energy Map Distribution (0-1.5)", "x": 0.02},
        margin=dict(l=42, r=8, t=36, b=36),
        height=330,
        xaxis=dict(title="energy value", range=[0.0, 1.5]),
        yaxis=dict(title="count"),
    )

    cards: list[dict] = []
    for name, stem in (("Context", "context"), ("Predict", "pred"), ("Target", "gt")):
        pca_scatter, _, _ = scatter3d(f"{name} PCA 3D Scatter", data[f"{stem}_pca3d"], data[f"{stem}_pca_rgb_flat"])
        umap_scatter, _, _ = scatter3d(f"{name} UMAP 3D Scatter", data[f"{stem}_umap3d"], data[f"{stem}_umap_rgb_flat"])
        # Keep strict left-right pairing: RGB map (left), RGB scatter (right).
        cards.append({"title": f"{name} PCA RGB", "fig": img(f"{name} PCA RGB", data[f"{stem}_pca_rgb"]), "group": f"{stem}-pca"})
        cards.append({"title": f"{name} PCA RGB Scatter", "fig": pca_scatter, "group": f"{stem}-pca"})
        cards.append({"title": f"{name} UMAP RGB", "fig": img(f"{name} UMAP RGB", data[f"{stem}_umap_rgb"]), "group": f"{stem}-umap"})
        cards.append({"title": f"{name} UMAP RGB Scatter", "fig": umap_scatter, "group": f"{stem}-umap"})
    # Non-pair panels afterwards.
    cards.extend(
        [
            {"title": "Input (Log-Norm)", "fig": heat("Input (Log-Norm)", data["orig"], "Viridis"), "group": "input"},
            {"title": "Effective Rank", "fig": fig_eff_rank, "group": "eff-rank"},
            {"title": "Active Loss Terms (Unweighted)", "fig": fig_loss_components, "group": "loss-components"},
            {"title": "Active Loss Terms (Weighted)", "fig": fig_weighted_components, "group": "weighted-loss-components"},
            {"title": "Rank Diagnostics", "fig": fig_rank_diag, "group": "rank-diag"},
            {"title": "Rank Energy Top-k", "fig": fig_rank_energy, "group": "rank-energy"},
            {"title": "Energy Distribution", "fig": fig_energy_dist, "group": "energy-dist"},
            {"title": "Target Locations", "fig": heat("Target Locations", data["target"], "Magma"), "group": "target-loc"},
            {"title": "Target Location Heatmap", "fig": heat("Target Location Heatmap", data["target_loc_heatmap"], "Magma"), "group": "target-heat"},
            {"title": "Energy Map", "fig": heat("Energy Map", data["energy_map"], "Inferno"), "group": "energy"},
            {
                "title": "Visit Frequency Heatmap",
                "fig": heat(
                    "Visit Frequency Heatmap (log1p, unvisited=NaN)",
                    data["visit_heatmap"],
                    "Cividis",
                    percentile_scale=False,
                    log1p_nonzero_nan=True,
                ),
                "group": "visit",
            },
        ]
    )
    if model_mode in ("pyramid", "3d_slab") and "pyramid_mask_cube" in data.files:
        cards.append(
            {
                "title": "Pyramid Mask 3D (Sample-0)",
                "fig": mask3d("Pyramid Mask 3D (Sample-0)", data["pyramid_mask_cube"]),
                "group": "pyr-mask-3d",
            }
        )

    rendered = []
    seen_groups: set[str] = set()
    for i, card in enumerate(cards):
        fig = card["fig"]
        group = card["group"]
        panel_title = str(card.get("title", f"panel_{i+1}"))
        panel_title_html = html_lib.escape(panel_title, quote=True)
        controls = ""
        if group not in seen_groups and ("-pca" in group or "-umap" in group):
            controls = (
                f'<div class="controls local-controls" data-group="{group}">'
                f'<label>xmin <input type="number" step="any" data-k="xmin" placeholder="auto"></label>'
                f'<label>xmax <input type="number" step="any" data-k="xmax" placeholder="auto"></label>'
                f'<label>ymin <input type="number" step="any" data-k="ymin" placeholder="auto"></label>'
                f'<label>ymax <input type="number" step="any" data-k="ymax" placeholder="auto"></label>'
                f'<label>zmin <input type="number" step="any" data-k="zmin" placeholder="auto"></label>'
                f'<label>zmax <input type="number" step="any" data-k="zmax" placeholder="auto"></label>'
                f'<button class="apply-local" type="button" data-group="{group}">Apply</button>'
                f"</div>"
            )
            seen_groups.add(group)
        rendered.append(
            f'<section class="card" data-group="{group}">'
            f'<div class="card-tools"><button class="save-panel" type="button" data-panel-title="{panel_title_html}">Save PNG</button></div>'
            f'{controls}'
            f'{fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False), config={"responsive": True, "displaylogo": False})}'
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
    .mask-summary {{ margin: 0 0 18px 2px; padding: 14px 18px; background: #fff; border: 1px solid #d9deea; border-radius: 8px; font-size: 15px; line-height: 1.8; }}
    .mask-summary .val {{ color: #3a4055; margin-right: 20px; }}
    .mask-summary .erank {{ display: inline-block; margin-left: 8px; padding: 4px 14px; background: #1a1a2e; color: #fde725; border-radius: 6px; font-weight: 700; font-size: 17px; }}
    .mask-summary .erank-label {{ color: #7a7d8a; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .mask-summary .erank-value {{ font-size: 24px; margin-left: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(420px, 1fr)); gap: 12px; }}
    .controls {{ display:flex; flex-wrap:wrap; gap:8px; margin: 0 0 12px 2px; align-items:center; }}
    .controls input {{ width:88px; }}
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
  </div>
  <div class="mask-summary">{mask_summary_html}</div>
  <div class="grid">{''.join(rendered)}</div>
  <script>
  function num(el) {{
    const v = parseFloat(el.value);
    return Number.isFinite(v) ? v : null;
  }}
  function applyRangesForGroup(group) {{
    const controls = document.querySelector('.local-controls[data-group="' + group + '"]');
    if (!controls) return;
    const get = (k) => {{
      const el = controls.querySelector('input[data-k="' + k + '"]');
      return el ? num(el) : null;
    }};
    const xmin = get("xmin"), xmax = get("xmax");
    const ymin = get("ymin"), ymax = get("ymax");
    const zmin = get("zmin"), zmax = get("zmax");
    const sections = document.querySelectorAll('.card[data-group="' + group + '"] .js-plotly-plot');
    function clamp01(v) {{
      return Math.max(0.0, Math.min(1.0, v));
    }}
    function map01(v, lo, hi) {{
      const den = Math.max(1e-12, hi - lo);
      return clamp01((v - lo) / den);
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
        const idx = (vals.length - 1) * p;
        const i0 = Math.floor(idx);
        const i1 = Math.ceil(idx);
        if (i0 === i1) return vals[i0];
        const t = idx - i0;
        return vals[i0] * (1 - t) + vals[i1] * t;
      }};
      let lo = q(Math.max(0.0, Math.min(1.0, loPct)));
      let hi = q(Math.max(0.0, Math.min(1.0, hiPct)));
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) {{
        lo = vals[0];
        hi = vals[vals.length - 1];
      }}
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [0.0, 1.0];
      return [lo, hi];
    }}
    sections.forEach((gd) => {{
      const fl = gd._fullLayout || {{}};
      if (fl.scene) {{
        const rl = {{
          "scene.camera.projection.type": "orthographic",
          "scene.aspectmode": "cube",
        }};
        if (xmin !== null && xmax !== null) rl["scene.xaxis.range"] = [xmin, xmax];
        if (ymin !== null && ymax !== null) rl["scene.yaxis.range"] = [ymin, ymax];
        if (zmin !== null && zmax !== null) rl["scene.zaxis.range"] = [zmin, zmax];
        Plotly.relayout(gd, rl);
        (gd.data || []).forEach((tr, i) => {{
          if (!tr || tr.type !== "scatter3d") return;
          const xs = tr.x || [];
          const ys = tr.y || [];
          const zs = tr.z || [];
          // Robust recoloring from visible-range stats to avoid black collapse.
          const xr = (xmin !== null && xmax !== null) ? [xmin, xmax] : percentileRange(xs, 0.01, 0.99);
          const yr = (ymin !== null && ymax !== null) ? [ymin, ymax] : percentileRange(ys, 0.01, 0.99);
          const zr = (zmin !== null && zmax !== null) ? [zmin, zmax] : percentileRange(zs, 0.01, 0.99);
          const n = Math.min(xs.length || 0, ys.length || 0, zs.length || 0);
          const colors = new Array(n);
          for (let k = 0; k < n; k++) {{
            const r = Math.round(map01(Number(xs[k]), xr[0], xr[1]) * 255.0);
            const g = Math.round(map01(Number(ys[k]), yr[0], yr[1]) * 255.0);
            const b = Math.round(map01(Number(zs[k]), zr[0], zr[1]) * 255.0);
            colors[k] = `rgb(${{r}},${{g}},${{b}})`;
          }}
          Plotly.restyle(gd, {{"marker.color": [colors]}}, [i]);
        }});
      }}
    }});
  }}
	  function initControlDefaults() {{
    document.querySelectorAll('.local-controls').forEach((controls) => {{
      const group = controls.getAttribute("data-group");
      const gd = document.querySelector('.card[data-group="' + group + '"] .js-plotly-plot');
      if (!gd || !gd._fullLayout || !gd._fullLayout.scene) return;
      const s = gd._fullLayout.scene;
      const defs = {{
        xmin: s.xaxis && s.xaxis.range ? s.xaxis.range[0] : null,
        xmax: s.xaxis && s.xaxis.range ? s.xaxis.range[1] : null,
        ymin: s.yaxis && s.yaxis.range ? s.yaxis.range[0] : null,
        ymax: s.yaxis && s.yaxis.range ? s.yaxis.range[1] : null,
        zmin: s.zaxis && s.zaxis.range ? s.zaxis.range[0] : null,
        zmax: s.zaxis && s.zaxis.range ? s.zaxis.range[1] : null,
      }};
      Object.keys(defs).forEach((k) => {{
        const el = controls.querySelector('input[data-k="' + k + '"]');
        const v = defs[k];
        if (el && Number.isFinite(v)) el.value = String(v);
      }});
    }});
  }}
	  window.addEventListener("load", initControlDefaults);
	  document.querySelectorAll(".apply-local").forEach((btn) => {{
	    btn.addEventListener("click", () => applyRangesForGroup(btn.getAttribute("data-group")));
	  }});
	  function _safeFileName(name) {{
	    return String(name || "panel")
	      .toLowerCase()
	      .replace(/[^a-z0-9]+/g, "_")
	      .replace(/^_+|_+$/g, "") || "panel";
	  }}
	  async function savePanelPng(btn) {{
	    const card = btn.closest(".card");
	    const gd = card ? card.querySelector(".js-plotly-plot") : null;
	    if (!gd || !window.Plotly || typeof window.Plotly.toImage !== "function") {{
	      alert("Panel export unavailable for this card.");
	      return;
	    }}
	    const title = btn.getAttribute("data-panel-title") || "panel";
	    const width = Math.max(1200, gd.clientWidth || 1200);
	    const height = Math.max(900, gd.clientHeight || 900);
	    try {{
	      const dataUrl = await window.Plotly.toImage(gd, {{ format: "png", width, height, scale: 2 }});
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
	  document.querySelectorAll(".save-panel").forEach((btn) => {{
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
    for name, stem in (("Context", "context"), ("Predict", "pred"), ("Target", "gt")):
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
    for name in sorted(os.listdir(args.sessions_dir)):
        session_dir = os.path.join(args.sessions_dir, name)
        if not os.path.isdir(session_dir):
            continue
        print("=" * 72)
        print(f"dashboard_session_begin={session_dir}")
        dash_html_path = os.path.join(session_dir, "dashboard.html")
        export_path = os.path.join(export_dir, f"{name.replace('/', '_')}.html")
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
