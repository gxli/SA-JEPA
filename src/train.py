from __future__ import annotations

import csv
import glob
import hashlib
import json
import logging
import math
import warnings

from tqdm import tqdm
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.dataset import JEPADataset, resolve_input_files
from src.dataset3d import JEPA3DCropDataset
from src.diagnostics import (
    compute_effective_rank_from_features,
    compute_error_by_scale,
    rank_dashboard,
)
from src.inference import run_post_training_inference, run_post_training_inference_3d
from src.losses import (
    anchored_spread_hinge_loss,
    compute_jepa_energy,
    compute_output_spread_regularizer_loss,
    compute_raw_mse_and_norm_err,
    compute_sim_var_cov,
    compute_sim_var_cov_torch,
    compute_target_energy_map,
    embedding_channel_std,
    embedding_std_hinge_loss,
    embedding_spread_stats,
    extract_valid_dense_embeddings,
    extract_valid_pooled_embeddings,
    parse_spread_regularizer_config,
)
from src.models.build_jepa import CDD_CUBE_ENCODER_TYPES, CDD_DEBUG_ENCODER_TYPES, MASK_MAP_ENCODER_TYPES, PyramidGridJEPA
from src.models.build_jepa3d import PyramidGridJEPA3D, compute_3d_encoder_receptive_field_depth
from src.models.masking import _max_effective_mask_box_size, pack_target_mask_passes, prepare_context_batch
from src.utils import log_error, set_error_log_path
from src.utils.cdd_import import import_constrained_diffusion, safe_constrained_diffusion_decomposition
from src.utils.npy import _safe_load_npy
from src.utils.support import additional_crop_from_config
from src.utils.viz import _target_location_yx, export_inference_dashboard_artifacts, save_volumetric_umap_embeddings

LOGGER = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore",
    message=r"The epoch parameter in `scheduler\.step\(\)` was not necessary.*",
    category=UserWarning,
    module=r"torch\.optim\.lr_scheduler",
)


def _ensure_training_logging() -> None:
    """Install a small default logger for API/imported training runs."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.setLevel(logging.INFO)


def log_info(*parts: object) -> None:
    LOGGER.info(" ".join(str(part) for part in parts))


def _fmt_metric(v: float) -> str:
    x = float(v)
    ax = abs(x)
    if ax == 0.0:
        return "0.0000"
    if ax < 1e-3 or ax >= 1e3:
        return f"{x:.3e}"
    return f"{x:.4f}"


def _format_metric_dict(metrics: dict[str, str]) -> str:
    return " ".join(f"{key}={value}" for key, value in metrics.items())


def _seed_dataloader_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _format_progress_line(
    prefix: str,
    losses: dict[str, str],
    diagnostics: dict[str, str],
    optim_state: dict[str, str] | None = None,
) -> str:
    parts = [prefix, _format_metric_dict(losses), _format_metric_dict(diagnostics)]
    if optim_state:
        parts.append(_format_metric_dict(optim_state))
    return " | ".join(part for part in parts if part)


def _format_target_z_counts(
    target_locations: torch.Tensor,
    target_valid: torch.Tensor,
    depth: int | None = None,
) -> str:
    if target_locations.dim() < 3 or int(target_locations.shape[-1]) < 3:
        return ""
    if target_valid.dim() < 2:
        return ""
    with torch.no_grad():
        loc = target_locations.detach()
        valid = target_valid.detach().bool()
        if loc.shape[:2] != valid.shape[:2]:
            return ""
        z = loc[..., 0].long()
        max_depth = int(depth) if depth is not None and int(depth) > 0 else int(z[valid].max().item() + 1) if bool(valid.any().item()) else 0
        if max_depth <= 0:
            return ""
        counts = []
        denom = max(1, int(valid.shape[0]))
        for zi in range(max_depth):
            n = float(((z == zi) & valid).sum().item()) / float(denom)
            counts.append(f"{zi}:{n:.1f}")
    return ",".join(counts)


def _inverse_augmented_yx_to_native(yy: int, xx: int, meta: dict | None) -> tuple[int, int]:
    if not isinstance(meta, dict):
        return int(yy), int(xx)
    y = int(yy)
    x = int(xx)
    post_h = int(meta.get("post_aug_h", meta.get("pre_aug_h", 0)) or 0)
    post_w = int(meta.get("post_aug_w", meta.get("pre_aug_w", 0)) or 0)
    pre_h = int(meta.get("pre_aug_h", post_h) or post_h)
    pre_w = int(meta.get("pre_aug_w", post_w) or post_w)
    if bool(meta.get("flip_x", False)) and post_w > 0:
        x = post_w - 1 - x
    if bool(meta.get("flip_y", False)) and post_h > 0:
        y = post_h - 1 - y
    rot_k = int(meta.get("rot_k", 0) or 0) % 4
    if rot_k == 1:
        y, x = x, pre_h - 1 - y
    elif rot_k == 2:
        y, x = pre_h - 1 - y, pre_w - 1 - x
    elif rot_k == 3:
        y, x = pre_w - 1 - x, y
    y += int(meta.get("crop_y0", 0) or 0)
    x += int(meta.get("crop_x0", 0) or 0)
    return int(y), int(x)


def _format_active_loss_terms(
    *,
    total: float,
    total_label: str = "total",
    prediction: float,
    prediction_weight: float,
    spread: float,
    spread_weight: float,
    symmetry: float,
    symmetry_weight: float,
    vicreg_var: float,
    vicreg_var_weight: float,
    vicreg_cov: float,
    vicreg_cov_weight: float,
) -> dict[str, str]:
    """Runtime loss printout: raw active terms plus weighted contribution."""
    terms = {
        str(total_label): _fmt_metric(total),
        "pred": _fmt_metric(prediction),
        "wpred": _fmt_metric(prediction_weight * prediction),
    }
    if abs(spread_weight) > 1e-12:
        spread_label = "0.0000(off)" if abs(spread) <= 1e-8 else f"{_fmt_metric(spread)}(active)"
        terms["spread"] = spread_label
        terms["wspread"] = _fmt_metric(spread_weight * spread)
    if abs(vicreg_var_weight) > 1e-12:
        terms["vicvar"] = _fmt_metric(vicreg_var)
        terms["wvicvar"] = _fmt_metric(vicreg_var_weight * vicreg_var)
    if abs(vicreg_cov_weight) > 1e-12:
        terms["viccov"] = _fmt_metric(vicreg_cov)
        terms["wviccov"] = _fmt_metric(vicreg_cov_weight * vicreg_cov)
    if abs(symmetry_weight) > 1e-12:
        terms["sym"] = _fmt_metric(symmetry)
        terms["wsym"] = _fmt_metric(symmetry_weight * symmetry)
    return terms


def _flush_csv_rows(path: str, rows: list[list]) -> None:
    if not rows:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    rows.clear()


def _collate_pad_spatial(batch: list[torch.Tensor]) -> torch.Tensor:
    if len(batch) == 0:
        raise ValueError("Empty batch is not supported")
    max_h = max(int(x.shape[-2]) for x in batch)
    max_w = max(int(x.shape[-1]) for x in batch)
    out = []
    for x in batch:
        dh = max_h - int(x.shape[-2])
        dw = max_w - int(x.shape[-1])
        if dh > 0 or dw > 0:
            # Mark padded pixels as invalid so downstream target sampling can reject them.
            x = F.pad(x, (0, dw, 0, dh), mode="constant", value=float("nan"))
        out.append(x)
    return torch.stack(out, dim=0)


def _collate_for_inference(batch):
    """Collate inference batches — handles both (cdd_orig, x_clean) tuples and plain tensors."""
    if len(batch) == 0:
        raise ValueError("Empty batch is not supported")
    if isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2:
        cdd_list = [item[0] for item in batch]
        x_clean_list = [item[1] for item in batch]
        return _collate_pad_spatial(cdd_list), _collate_pad_spatial(x_clean_list)
    return _collate_pad_spatial(batch), None


def _summarize_data_array(arr: np.ndarray) -> dict:
    raw = np.asarray(arr)
    finite = np.isfinite(raw)
    finite_values = raw[finite]
    clean = np.nan_to_num(raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    amin = float(clean.min()) if clean.size > 0 else 0.0
    amax = float(clean.max()) if clean.size > 0 else 0.0
    normalized = (clean - amin) / (amax - amin) if amax - amin > 1e-20 else np.zeros_like(clean)
    nonzero_coords = np.where(normalized > 0.0)
    bbox = None
    if len(nonzero_coords) > 0 and nonzero_coords[0].size > 0:
        bbox = [[int(axis.min()), int(axis.max())] for axis in nonzero_coords]
    quantiles = {}
    if finite_values.size > 0:
        quantiles = {
            str(q): float(np.quantile(finite_values, q))
            for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        }
    return {
        "shape": [int(v) for v in raw.shape],
        "ndim": int(raw.ndim),
        "dtype": str(raw.dtype),
        "size": int(raw.size),
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(raw).sum()),
        "posinf_count": int(np.isposinf(raw).sum()),
        "neginf_count": int(np.isneginf(raw).sum()),
        "raw_finite_zero_count": int(np.count_nonzero(finite_values == 0.0)),
        "raw_finite_zero_fraction": float(np.mean(finite_values == 0.0)) if finite_values.size > 0 else 0.0,
        "raw_quantiles": quantiles,
        "normalized_zero_count": int(np.count_nonzero(normalized == 0.0)),
        "normalized_zero_fraction": float(np.mean(normalized == 0.0)) if normalized.size > 0 else 0.0,
        "normalized_positive_count": int(np.count_nonzero(normalized > 0.0)),
        "normalized_nonzero_bbox": bbox,
        "aspect_ratio_h_over_w": (
            float(raw.shape[-2]) / float(raw.shape[-1])
            if raw.ndim >= 2 and int(raw.shape[-1]) > 0
            else None
        ),
    }


def _write_data_profile(*, data_cfg: dict, session_dir: str, config_name: str) -> None:
    data_root = data_cfg.get("data_root", "data")
    npy_pattern = data_cfg.get("npy_pattern", "*.npy")
    pattern = os.path.join(data_root, npy_pattern)
    files = resolve_input_files(data_root=data_root, npy_pattern=npy_pattern, input_files=data_cfg.get("input_files"))
    profile = {
        "pattern": pattern,
        "input_files": data_cfg.get("input_files"),
        "crop_mode": str(data_cfg.get("crop_mode", "none")),
        "crop_size": data_cfg.get("crop_size"),
        "files": [],
    }
    for path in files:
        item = {"path": path}
        if path.endswith(".fits"):
            try:
                from astropy.io import fits
                arr = fits.getdata(path, memmap=True)
                item.update(_summarize_data_array(np.asarray(arr, dtype=np.float32)))
            except ImportError:
                item.update({"shape": "unknown (astropy not installed)", "nan_count": -1, "normalized_zero_fraction": -1, "aspect_ratio_h_over_w": -1})
        elif path.endswith(".h5"):
            try:
                import h5py
                with h5py.File(path, "r") as h5:
                    arr = h5["data"][:]
                item.update(_summarize_data_array(np.asarray(arr, dtype=np.float32)))
            except ImportError:
                item.update({"shape": "unknown (h5py not installed)", "nan_count": -1, "normalized_zero_fraction": -1, "aspect_ratio_h_over_w": -1})
        else:
            item.update(_summarize_data_array(_safe_load_npy(path, mmap_mode="r")))
        profile["files"].append(item)
        shape_str = f"shape={tuple(item.get('shape', '?'))}" if 'shape' in item else ""
        log_info(
            f"[{config_name}] Data profile: path={path} {shape_str} "
            f"nan={item.get('nan_count', '?')} normalized_zero_fraction={item.get('normalized_zero_fraction', -1):.4f} "
            f"aspect_h_over_w={item.get('aspect_ratio_h_over_w', -1)}"
        )
    with open(os.path.join(session_dir, "data_profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
        f.write("\n")


def _write_cdd_cache_profile(*, cdd_cache: dict | None, session_dir: str, config_name: str) -> None:
    entries = []
    for (path, slice_idx), value in sorted((cdd_cache or {}).items()):
        variants = ("untransformed", "transformed") if isinstance(value, dict) else (None,)
        for variant in variants:
            arr = np.asarray(value[variant] if variant else value)
            finite = np.isfinite(arr)
            item = {
                "path": path,
                "slice_idx": slice_idx,
                "variant": variant,
                "shape": [int(v) for v in arr.shape],
                "finite_count": int(finite.sum()),
                "nan_count": int(np.isnan(arr).sum()),
                "zero_count": int(np.count_nonzero(arr == 0.0)),
                "zero_fraction": float(np.mean(arr == 0.0)) if arr.size > 0 else 0.0,
                "positive_count": int(np.count_nonzero(arr > 0.0)),
                "min": float(arr[finite].min()) if finite.any() else None,
                "max": float(arr[finite].max()) if finite.any() else None,
            }
            entries.append(item)
            variant_tag = f" variant={variant}" if variant else ""
            log_info(
                f"[{config_name}] CDD cache profile: path={path}{variant_tag} shape={tuple(item['shape'])} "
                f"nan={item['nan_count']} zero_fraction={item['zero_fraction']:.4f} max={item['max']}"
            )
    with open(os.path.join(session_dir, "cdd_cache_profile.json"), "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, indent=2)
        f.write("\n")


def _expected_hw_from_data_profile(session_dir: str) -> tuple[int, int] | None:
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


def _full_frame_inference_shape_mismatch(
    *,
    session_dir: str,
    train_cfg: dict,
    inference_outputs_path: str,
) -> str | None:
    if not os.path.exists(inference_outputs_path) or os.path.getsize(inference_outputs_path) <= 0:
        return None
    max_diag = train_cfg.get("inference_max_diagnostic_size")
    try:
        full_frame_requested = max_diag is None or int(max_diag) <= 0
    except (TypeError, ValueError):
        full_frame_requested = str(max_diag).strip().lower() in ("", "none", "false", "full")
    if not full_frame_requested:
        return None
    expected_hw = _expected_hw_from_data_profile(session_dir)
    if expected_hw is None:
        return None
    try:
        outputs = torch.load(inference_outputs_path, map_location="cpu")
        x_clean = outputs.get("x_clean") if isinstance(outputs, dict) else None
        if x_clean is None or not hasattr(x_clean, "shape") or len(x_clean.shape) < 2:
            return None
        observed_hw = (int(x_clean.shape[-2]), int(x_clean.shape[-1]))
    except Exception as e:
        return f"could not read existing inference_outputs.pt ({type(e).__name__}: {e})"
    if observed_hw == expected_hw:
        return None
    return (
        f"existing inference_outputs.pt shape={observed_hw} but current config requests "
        f"full-frame shape={expected_hw}"
    )


def _cdd_disk_cache_paths(*, cache_dir: str, path: str, meta: dict) -> tuple[str, str]:
    abs_path = os.path.abspath(path)
    stat = os.stat(abs_path)
    key_payload = {
        "path": abs_path,
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
        "meta": meta,
    }
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    stem = os.path.splitext(os.path.basename(path))[0]
    return (
        os.path.join(cache_dir, f"{stem}_{key}.npz"),
        os.path.join(cache_dir, f"{stem}_{key}.json"),
    )


def _load_cdd_disk_cache(data_path: str, meta_path: str, expected_meta: dict) -> dict | None:
    if not (os.path.exists(data_path) and os.path.exists(meta_path)):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            got_meta = json.load(f)
        if got_meta != expected_meta:
            return None
        loaded = np.load(data_path)
        if "untransformed" not in loaded or "transformed" not in loaded:
            return None
        return {
            "untransformed": loaded["untransformed"].astype(np.float32, copy=False),
            "transformed": loaded["transformed"].astype(np.float32, copy=False),
        }
    except Exception as e:
        log_info(f"CDD disk cache ignored: {type(e).__name__}: {e}")
        return None


def _save_cdd_disk_cache(data_path: str, meta_path: str, value: dict, meta: dict) -> None:
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    tmp_data = f"{data_path}.tmp"
    tmp_meta = f"{meta_path}.tmp"
    np.savez_compressed(
        tmp_data,
        untransformed=value["untransformed"].astype(np.float32, copy=False),
        transformed=value["transformed"].astype(np.float32, copy=False),
    )
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(f"{tmp_data}.npz", data_path)
    os.replace(tmp_meta, meta_path)


def _normalize_decomposition_backend(model_cfg: dict, data_cfg: dict) -> str:
    model_key = str(model_cfg.get("model_key", model_cfg.get("encoder_type", ""))).strip().lower()
    backend_value = model_cfg.get(
        "decomposition_backend",
        model_cfg.get("pyramid_backend", data_cfg.get("decomposition_backend", "cdd")),
    )
    backend_key = str(backend_value).strip().lower().replace("-", "_")
    if backend_key in ("ring_weighted", "ring_conv_weighted", "weighted_ring", "weighted_ring_conv"):
        return "ring_conv_weighted"
    if model_key.startswith("ring_") or "ring_conv" in model_key:
        return "ring_conv"
    key = backend_key
    if key in ("ring", "ring_conv", "multiscale_ring", "multiscale_ring_conv"):
        return "ring_conv"
    if key in ("cdd", "constrained_diffusion", "constrained_diffusion_decomposition"):
        return "cdd"
    raise ValueError(
        f"Unsupported decomposition backend={backend_value!r}; "
        "expected 'cdd', 'ring_conv', or 'ring_conv_weighted'."
    )


def _resolve_ring_radii(model_cfg: dict) -> tuple[float, ...]:
    radii = model_cfg.get("ring_radii", model_cfg.get("ring_conv_radii"))
    if radii is None:
        sigmas = tuple(float(s) for s in model_cfg.get("sigmas", [2, 4, 8]))
        radii = (0.0,) + sigmas
    vals = tuple(float(v) for v in radii)
    if len(vals) < 2:
        raise ValueError(f"ring_radii must contain at least two boundaries, got {vals!r}")
    if vals[0] != 0.0:
        vals = (0.0,) + vals
    for lo, hi in zip(vals[:-1], vals[1:]):
        if hi <= lo:
            raise ValueError(f"ring_radii must be strictly increasing, got {vals!r}")
    return vals


def _ring_conv2d(
    input_arr: np.ndarray,
    radii: tuple[float, ...],
    *,
    device: torch.device,
    weighted: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return normalized disk/annulus averages as (S, *spatial)."""
    arr = np.asarray(input_arr, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(f"ring_conv expects a 2D image or 3D stack/volume, got shape={arr.shape}")
    max_radius = float(radii[-1])
    pad = int(math.ceil(max_radius))
    yy, xx = np.mgrid[-pad : pad + 1, -pad : pad + 1]
    dist = np.sqrt((yy.astype(np.float32) ** 2) + (xx.astype(np.float32) ** 2))
    kernels = []
    for i, (lo, hi) in enumerate(zip(radii[:-1], radii[1:])):
        if i == 0 and float(lo) <= 0.0:
            mask = dist < float(hi)
        else:
            mask = (dist >= float(lo)) & (dist < float(hi))
        count = int(np.count_nonzero(mask))
        if count <= 0:
            raise ValueError(f"Empty ring kernel for radius range [{lo}, {hi})")
        kernel = mask.astype(np.float32) / float(count)
        kernels.append(kernel)

    conv_device = device if device.type in ("cuda", "mps") else torch.device("cpu")
    x_np = arr[None, None] if arr.ndim == 2 else arr[:, None]
    x = torch.from_numpy(x_np).to(device=conv_device, dtype=torch.float32)
    weight = torch.from_numpy(np.stack(kernels, axis=0)[:, None]).to(device=conv_device, dtype=torch.float32)
    if pad > 0:
        pad_mode = "reflect" if min(int(arr.shape[-2]), int(arr.shape[-1])) > pad else "replicate"
        x = F.pad(x, (pad, pad, pad, pad), mode=pad_mode)
    y = F.conv2d(x, weight)
    if arr.ndim == 2:
        out = y[0].detach().cpu().numpy()
    else:
        out = y.permute(1, 0, 2, 3).detach().cpu().numpy()
    out = np.clip(np.asarray(out, dtype=np.float32), a_min=0.0, a_max=None)
    if not bool(weighted):
        return out
    total = np.sum(out, axis=0, keepdims=True, dtype=np.float32)
    original = np.asarray(arr, dtype=np.float32)[None, ...]
    weights = np.divide(out, total, out=np.zeros_like(out, dtype=np.float32), where=total > float(eps))
    return np.asarray(weights * original, dtype=np.float32)


def _precompute_cdd_cache(
    *,
    data_cfg: dict,
    model_cfg: dict,
    device: torch.device,
    config_name: str,
    session_dir: str = "",
    cache_replicas: int = 1,
) -> dict:
    """Pre-compute a bounded CDD decomposition cache on GPU, store in CPU RAM."""
    decomposition_backend = _normalize_decomposition_backend(model_cfg, data_cfg)
    enabled = bool(data_cfg.get("cdd_precompute", True))
    if not enabled:
        log_info(
            f"[{config_name}] CDD precompute: disabled by data.cdd_precompute=false "
            f"(backend={decomposition_backend})"
        )
        return {}
    data_root = data_cfg.get("data_root", "data")
    npy_pattern = data_cfg.get("npy_pattern", "*.npy")
    cdd_mode = str(model_cfg.get("cdd_mode", data_cfg.get("cdd_mode", "log")))
    cdd_constrained = bool(model_cfg.get("cdd_constrained", data_cfg.get("cdd_constrained", True)))
    cdd_sm_mode = str(model_cfg.get("cdd_sm_mode", data_cfg.get("cdd_sm_mode", "reflect")))
    cdd_append_last_residual = bool(model_cfg.get("cdd_append_last_residual", True))
    cdd_pre_log_transform = bool(model_cfg.get("cdd_pre_log_transform", False))
    cdd_gaussian_backend = str(
        model_cfg.get("cdd_gaussian_backend", data_cfg.get("cdd_gaussian_backend", "cuda"))
    )
    cdd_effective_use_gpu = bool(device.type == "cuda")
    sigmas = tuple(model_cfg.get("sigmas", [2, 4, 8, 16]))
    cdd_num_channels = int(model_cfg.get("cdd_num_channels", len(sigmas)))
    cdd_request_num_channels = model_cfg.get("cdd_request_num_channels", None)
    ring_radii = _resolve_ring_radii(model_cfg) if decomposition_backend in ("ring_conv", "ring_conv_weighted") else None
    if ring_radii is not None:
        cdd_num_channels = len(ring_radii) - 1
        cdd_request_num_channels = None
    expected_default_scales = cdd_num_channels
    model_num_scales = int(model_cfg.get("num_scales", expected_default_scales))
    if device.type not in ("cuda", "mps"):
        log_info(f"[{config_name}] CDD precompute: using CPU backend because device={device.type}")
    npy_files = [
        p for p in resolve_input_files(data_root=data_root, npy_pattern=npy_pattern, input_files=data_cfg.get("input_files"))
        if p.endswith((".npy", ".fits"))
    ]
    if not npy_files:
        log_info(f"[{config_name}] CDD precompute: no files found for pattern, skipping")
        return {}
    disk_cache_enabled = bool(data_cfg.get("cdd_disk_cache", True))
    if "cdd_cache_dir" in data_cfg:
        disk_cache_dir = str(data_cfg["cdd_cache_dir"])
    else:
        disk_cache_dir = os.path.join(session_dir, "cdd_cache")
    cdd_meta = {
        "version": 3,
        "decomposition_backend": decomposition_backend,
        "cdd_mode": cdd_mode,
        "cdd_constrained": cdd_constrained,
        "cdd_sm_mode": cdd_sm_mode,
        "cdd_append_last_residual": cdd_append_last_residual,
        "cdd_pre_log_transform": cdd_pre_log_transform,
        "cdd_gaussian_backend": cdd_gaussian_backend,
        "cdd_effective_use_gpu": cdd_effective_use_gpu,
        "log_eps": float(model_cfg.get("log_eps", 1.0)),
        "cdd_log_std_floor_mult": float(model_cfg.get("cdd_log_std_floor_mult", 0.05)),
        "cdd_min_scale": float(min(float(s) for s in sigmas)),
        "cdd_max_scale": float(max(float(s) for s in sigmas)),
        "sigmas": [float(s) for s in sigmas],
        "ring_radii": None if ring_radii is None else [float(v) for v in ring_radii],
        "cdd_num_channels": cdd_num_channels,
        "cdd_request_num_channels": None if cdd_request_num_channels is None else int(cdd_request_num_channels),
        "model_num_scales": model_num_scales,
    }
    max_files = int(data_cfg.get("cdd_precompute_max_files", 4096))
    if max_files > 0 and len(npy_files) > max_files:
        raise RuntimeError(
            f"[{config_name}] CDD precompute: {len(npy_files)} files exceeds "
            f"data.cdd_precompute_max_files={max_files}. Bump the limit."
        )
    max_gb = float(data_cfg.get("cdd_precompute_max_gb", 8.0))
    if max_gb > 0:
        sample_path = npy_files[0]
        if sample_path.endswith(".fits"):
            from astropy.io import fits as _fits
            sample_shape = _fits.getdata(sample_path, memmap=True).shape
        else:
            sample_shape = _safe_load_npy(sample_path, mmap_mode="r").shape
        n_channels = int(model_num_scales)
        est_bytes_per = int(n_channels) * int(np.prod(sample_shape)) * np.dtype(np.float32).itemsize
        # 2× for untransformed + transformed variants
        est_process_gb = (2 * est_bytes_per * len(npy_files)) / float(1024 ** 3)
        est_node_gb = est_process_gb * max(1, int(cache_replicas))
        if est_node_gb > max_gb:
            raise RuntimeError(
                f"[{config_name}] CDD precompute: estimated cache {est_process_gb:.2f} GiB per process "
                f"x {max(1, int(cache_replicas))} local replica(s) = {est_node_gb:.2f} GiB exceeds "
                f"data.cdd_precompute_max_gb={max_gb:.2f}. Bump the limit, reduce dataset size, "
                "or disable RAM precompute for DDP."
            )
    log_info(
        f"[{config_name}] CDD precompute: {len(npy_files)} file(s), backend={decomposition_backend}, "
        f"disk_cache={'on' if disk_cache_enabled else 'off'}"
        + (f" dir={disk_cache_dir}" if disk_cache_enabled else "")
    )
    cache = {}
    disk_hits = 0
    disk_writes = 0
    cdd = None
    for path in npy_files:
        disk_data_path = disk_meta_path = None
        if disk_cache_enabled:
            disk_data_path, disk_meta_path = _cdd_disk_cache_paths(
                cache_dir=disk_cache_dir,
                path=path,
                meta=cdd_meta,
            )
            cached = _load_cdd_disk_cache(disk_data_path, disk_meta_path, cdd_meta)
            if cached is not None:
                for variant in ("untransformed", "transformed"):
                    if cached[variant].shape[0] != model_num_scales:
                        raise RuntimeError(
                            f"[{config_name}] CDD disk cache channel mismatch for {path} "
                            f"variant={variant}: cached_channels={cached[variant].shape[0]}, "
                            f"model_expected={model_num_scales}, "
                            f"cache_file={disk_data_path}"
                        )
                cache[(path, None)] = cached
                disk_hits += 1
                log_info(
                    f"[{config_name}] CDD disk cache hit: path={path} cache={disk_data_path} "
                    f"untransformed_shape={tuple(cached['untransformed'].shape)} "
                    f"transformed_shape={tuple(cached['transformed'].shape)}"
                )
                continue
        if decomposition_backend == "cdd" and cdd is None:
            allow_monai = cdd_gaussian_backend == "monai"
            cdd = import_constrained_diffusion(session_dir=session_dir, allow_monai=allow_monai)
            log_info(
                f"[{config_name}] CDD import: constrained_diffusion "
                f"monai={'on' if allow_monai else 'blocked'}"
            )
        log_info(f"[{config_name}] CDD GPU compute: path={path} backend={decomposition_backend}")
        if path.endswith(".fits"):
            from astropy.io import fits as _fits
            arr = np.asarray(_fits.getdata(path, memmap=True), dtype=np.float32)
        else:
            arr = _safe_load_npy(path, mmap_mode="r").astype(np.float32)
        # Normalize01 (same order as JEPADataset._preprocess_arr2d).
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        amin, amax = float(arr.min()), float(arr.max())
        if amax - amin > 1e-20:
            arr = (arr - amin) / (amax - amin)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        if arr.ndim not in (2, 3):
            raise ValueError(f"Unexpected ndim={arr.ndim} for {path}")

        # Shared CDD kwargs (untransformed and transformed share scales/config).
        cdd_kwargs = dict(
            max_scale=max(float(s) for s in sigmas),
            min_scale=min(float(s) for s in sigmas),
            mode=cdd_mode,
            constrained=cdd_constrained,
            sm_mode=cdd_sm_mode,
            return_scales=True,
            verbose=False,
            use_gpu=cdd_effective_use_gpu,
            gaussian_backend=cdd_gaussian_backend,
        )
        if cdd_request_num_channels is not None:
            cdd_kwargs["num_channels"] = int(cdd_request_num_channels)

        def _compute_cdd_variant(input_arr: np.ndarray, label: str) -> np.ndarray:
            """Run CDD on *input_arr* and return (S, *spatial) channels."""
            if decomposition_backend in ("ring_conv", "ring_conv_weighted"):
                result = _ring_conv2d(
                    input_arr,
                    ring_radii,
                    device=device,
                    weighted=decomposition_backend == "ring_conv_weighted",
                )
                if result.ndim != input_arr.ndim + 1 or result.shape[1:] != input_arr.shape:
                    raise ValueError(
                        f"ring_conv output for {path} variant={label} must have one leading scale axis "
                        f"over input shape {input_arr.shape}, got {result.shape}"
                    )
                if int(result.shape[0]) != model_num_scales:
                    raise RuntimeError(
                        f"[{config_name}] ring_conv/model scale mismatch for {path} variant={label}: "
                        f"rings={list(zip(ring_radii[:-1], ring_radii[1:]))}, "
                        f"cached_channels={int(result.shape[0])}, model_expected={model_num_scales}."
                    )
                log_info(
                    f"[{config_name}] ring_conv channels variant={label}: "
                    f"rings={list(zip(ring_radii[:-1], ring_radii[1:]))} "
                    f"cached_channels={int(result.shape[0])} model_expected={model_num_scales} "
                    f"input_shape={tuple(input_arr.shape)}"
                )
                return result.astype(np.float32, copy=False)

            channels_arr, residual, scales_used = safe_constrained_diffusion_decomposition(
                cdd, input_arr, **cdd_kwargs,
            )
            actual_ch = int(len(channels_arr))
            if actual_ch < cdd_num_channels:
                raise RuntimeError(
                    f"[{config_name}] CDD returned too few result bands for {path} "
                    f"variant={label}: "
                    f"requested_keep={cdd_num_channels}, returned_bands={actual_ch}, "
                    f"residual_channel={int(cdd_append_last_residual)}, "
                    f"scales_used={np.asarray(scales_used).tolist()}, input_shape={tuple(input_arr.shape)}. "
                    "The vanilla CDD contract is results + residual; the model can ignore extra results, "
                    "but it cannot invent missing configured result channels."
                )
            selected = list(channels_arr[:cdd_num_channels])
            result = np.clip(np.stack(selected, axis=0).astype(np.float32), a_min=0.0, a_max=None)
            if result.ndim != input_arr.ndim + 1 or result.shape[1:] != input_arr.shape:
                raise ValueError(
                    f"CDD output for {path} variant={label} must have one leading scale axis "
                    f"over input shape {input_arr.shape}, got {result.shape}"
                )
            if cdd_append_last_residual:
                kept_sum = np.sum(result, axis=0, dtype=np.float32)
                recomputed_residual = (input_arr - kept_sum).astype(np.float32)
                result[-1] = result[-1] + np.clip(recomputed_residual, a_min=0.0, a_max=None)
            cached_ch = int(result.shape[0])
            ignored_ch = max(0, actual_ch - cdd_num_channels)
            log_info(
                f"[{config_name}] CDD channels variant={label}: "
                f"request_arg={'auto' if cdd_request_num_channels is None else str(int(cdd_request_num_channels))} "
                f"keep_results={cdd_num_channels} returned_bands={actual_ch} ignored_bands={ignored_ch} "
                f"residual_added_to_last={int(cdd_append_last_residual)} "
                f"cached_channels={cached_ch} model_expected={model_num_scales} "
                f"scales_used={np.asarray(scales_used).tolist()} input_shape={tuple(input_arr.shape)}"
            )
            if cached_ch != model_num_scales:
                raise RuntimeError(
                    f"[{config_name}] CDD/model scale mismatch for {path} variant={label}: "
                    f"keep_results={cdd_num_channels}, returned_bands={actual_ch}, "
                    f"residual_added_to_last={int(cdd_append_last_residual)}, "
                    f"cached_channels={cached_ch}, model_expected={model_num_scales}. "
                    "CDD results and residual must align with the encoder input channel count."
                )
            return result

        # ---- untransformed variant (CDD on raw normalized data) ----
        cdd_untransformed = _compute_cdd_variant(arr.copy(), "untransformed")

        # ---- transformed variant (CDD on log-transformed data) ----
        log_eps_val = float(model_cfg.get("log_eps", 1.0))
        log_floor_mult = float(model_cfg.get("cdd_log_std_floor_mult", 0.05))
        eps_f = max(1e-6, log_eps_val)
        arr_clamp = np.clip(arr, 0.0, None)
        arr_std = float(np.std(arr_clamp))
        log_floor = max(eps_f, arr_std * log_floor_mult)
        arr_log = np.log(arr_clamp + log_floor).astype(np.float32)
        cdd_transformed = _compute_cdd_variant(arr_log, "transformed")

        cache_entry = {
            "untransformed": cdd_untransformed.astype(np.float32, copy=False),
            "transformed": cdd_transformed.astype(np.float32, copy=False),
        }
        cache[(path, None)] = cache_entry
        if disk_cache_enabled and disk_data_path is not None and disk_meta_path is not None:
            _save_cdd_disk_cache(disk_data_path, disk_meta_path, cache_entry, cdd_meta)
            disk_writes += 1
            log_info(f"[{config_name}] CDD disk cache saved: {disk_data_path}")
    # Free GPU memory used by CDD.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log_info(
        f"[{config_name}] CDD precompute: {len(cache)} entries cached, "
        f"disk_hits={disk_hits}, disk_writes={disk_writes}, GPU freed"
    )
    return cache


def _move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move_to_device(x, device) for x in value)
    if isinstance(value, list):
        return [_move_to_device(x, device) for x in value]
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    return value


def _sample_locality_refinement_context(
    context_data: tuple,
    *,
    allow_partial_overlap: float = 0.0,
    fixed_n_targets: int | None = None,
    knn_space: str = "spatial",
    candidate_latent_map: torch.Tensor | None = None,
    spatial_radius_px: float | None = None,
    one_target_per_pass: bool = False,
) -> tuple[tuple | None, list[tuple]]:
    """Sample one exactly-sized follow-up target set.

    ``random`` draws an independent vanilla target batch. ``spatial`` draws
    from candidates no farther than ``spatial_radius_px`` from one random
    anchor. ``latent`` directly takes the exact nearest neighbors of one
    random anchor in the clean projected target latent map.
    """
    if len(context_data) < 5 or not isinstance(context_data[4], dict):
        return None, []
    x_context, macro_locations, macro_scales, macro_valid, source_debug = context_data[:5]
    candidate_locations = source_debug.get("candidate_locations")
    candidate_box_sizes = source_debug.get("candidate_box_sizes")
    candidate_valid = source_debug.get("candidate_valid")
    target_box_sizes = source_debug.get("target_box_sizes")
    if any(value is None for value in (candidate_locations, candidate_box_sizes, candidate_valid, target_box_sizes)):
        return None, []
    if candidate_locations.ndim != 3 or candidate_locations.shape[-1] != 2:
        raise RuntimeError("candidate_locations must have shape BxMx2")
    if candidate_valid.shape != candidate_locations.shape[:2]:
        raise RuntimeError("candidate_valid must match candidate_locations[:2]")
    if candidate_box_sizes.shape != candidate_locations.shape[:2]:
        raise RuntimeError("candidate_box_sizes must match candidate_locations[:2]")
    knn_space = str(knn_space).strip().lower()
    if knn_space not in {"random", "spatial", "latent"}:
        raise ValueError("knn_space must be 'random', 'spatial', or 'latent'")
    if knn_space == "spatial" and (spatial_radius_px is None or spatial_radius_px <= 0):
        raise ValueError("spatial sampling requires a positive spatial_radius_px")
    if knn_space == "latent":
        if candidate_latent_map is None or candidate_latent_map.ndim != 4:
            raise RuntimeError("latent KNN requires candidate_latent_map with shape BxCxHxW")
        if int(candidate_latent_map.shape[0]) != int(candidate_locations.shape[0]):
            raise RuntimeError("candidate_latent_map batch size must match candidate_locations")

    bsz, target_slots = macro_valid.shape
    micro_locations = torch.zeros_like(macro_locations)
    micro_scales = torch.zeros_like(macro_scales)
    micro_valid = torch.zeros_like(macro_valid)
    micro_box_sizes = torch.zeros_like(target_box_sizes)
    review_rows: list[tuple] = []
    for bi in range(int(bsz)):
        candidate_indices = torch.nonzero(candidate_valid[bi], as_tuple=False).flatten()
        requested = (
            int(fixed_n_targets)
            if fixed_n_targets is not None
            else int(macro_valid[bi].sum().item())
        )
        if fixed_n_targets is not None and int(candidate_indices.numel()) < requested:
            raise RuntimeError(
                "Fixed locality target count cannot be satisfied: "
                f"sample={bi} requested={requested} candidates={int(candidate_indices.numel())}"
            )
        n_targets = min(requested, int(target_slots), int(candidate_indices.numel()))
        if fixed_n_targets is not None and n_targets != requested:
            raise RuntimeError(
                "Fixed locality target count exceeds the packed target slots: "
                f"sample={bi} requested={requested} slots={int(target_slots)}"
            )
        if n_targets <= 0:
            continue

        pool_locations = candidate_locations[bi, candidate_indices]
        anchor_pool_index = int(
            torch.randint(0, int(candidate_indices.numel()), (), device=pool_locations.device).item()
        )
        anchor = pool_locations[anchor_pool_index]
        offsets = pool_locations.float() - anchor.float().unsqueeze(0)
        spatial_distances_sq = offsets.square().sum(dim=1)
        latent_distances_sq = None
        if candidate_latent_map is not None:
            latent_height, latent_width = candidate_latent_map.shape[-2:]
            pool_y = pool_locations[:, 0].long().clamp(0, int(latent_height) - 1)
            pool_x = pool_locations[:, 1].long().clamp(0, int(latent_width) - 1)
            pool_latents = candidate_latent_map[bi, :, pool_y, pool_x].transpose(0, 1).float()
            anchor_latent = pool_latents[anchor_pool_index]
            latent_distances_sq = (pool_latents - anchor_latent.unsqueeze(0)).square().sum(dim=1)
        if knn_space == "latent":
            assert latent_distances_sq is not None
            selected_pool_indices = torch.topk(
                latent_distances_sq,
                k=n_targets,
                largest=False,
                sorted=True,
            ).indices
            selected_ranks = torch.arange(n_targets, device=pool_locations.device)
            candidate_pool_size = n_targets
        elif knn_space == "spatial":
            radius_sq = float(spatial_radius_px) ** 2
            local_pool = torch.nonzero(
                spatial_distances_sq <= radius_sq,
                as_tuple=False,
            ).flatten()
            if int(local_pool.numel()) < n_targets:
                raise RuntimeError(
                    "Spatial FOV neighborhood cannot satisfy fixed target count: "
                    f"sample={bi} radius_px={float(spatial_radius_px):.3f} "
                    f"requested={n_targets} candidates={int(local_pool.numel())}"
                )
            chosen_in_pool = torch.randperm(
                int(local_pool.numel()), device=pool_locations.device
            )[:n_targets]
            selected_pool_indices = local_pool[chosen_in_pool]
            spatial_order = torch.argsort(spatial_distances_sq)
            spatial_rank = torch.empty_like(spatial_order)
            spatial_rank[spatial_order] = torch.arange(
                int(spatial_order.numel()), device=spatial_order.device
            )
            selected_ranks = spatial_rank[selected_pool_indices]
            candidate_pool_size = int(local_pool.numel())
        else:
            selected_pool_indices = torch.randperm(
                int(candidate_indices.numel()), device=pool_locations.device
            )[:n_targets]
            selected_ranks = torch.arange(n_targets, device=pool_locations.device)
            candidate_pool_size = int(candidate_indices.numel())
        selected_candidate_indices = candidate_indices[selected_pool_indices]

        micro_locations[bi, :n_targets] = candidate_locations[bi, selected_candidate_indices]
        micro_box_sizes[bi, :n_targets] = candidate_box_sizes[bi, selected_candidate_indices]
        micro_valid[bi, :n_targets] = True
        macro_scale_values = macro_scales[bi, macro_valid[bi]]
        if macro_scale_values.numel() > 0:
            scale_order = torch.randperm(
                int(macro_scale_values.numel()),
                device=macro_scale_values.device,
            )
            micro_scales[bi, :n_targets] = macro_scale_values[scale_order[:n_targets]]

        selected_locations_cpu = micro_locations[bi, :n_targets].detach().cpu()
        selected_ranks_cpu = selected_ranks.detach().cpu()
        selected_spatial_distances_cpu = torch.sqrt(
            spatial_distances_sq[selected_pool_indices]
        ).detach().cpu()
        selected_latent_distances_cpu = (
            torch.sqrt(latent_distances_sq[selected_pool_indices]).detach().cpu()
            if latent_distances_sq is not None
            else None
        )
        anchor_cpu = anchor.detach().cpu()
        for target_index in range(n_targets):
            review_rows.append(
                (
                    int(bi),
                    int(anchor_cpu[0]),
                    int(anchor_cpu[1]),
                    int(target_index),
                    int(selected_locations_cpu[target_index, 0]),
                    int(selected_locations_cpu[target_index, 1]),
                    int(selected_ranks_cpu[target_index]),
                    float(selected_spatial_distances_cpu[target_index]),
                    (
                        float(selected_latent_distances_cpu[target_index])
                        if selected_latent_distances_cpu is not None
                        else float("nan")
                    ),
                    int(n_targets),
                    int(candidate_pool_size),
                )
            )

    if not bool(micro_valid.any().item()):
        return None, []

    debug = dict(source_debug)
    debug["target_box_sizes"] = micro_box_sizes
    debug["otf_mask_pass_ids"] = pack_target_mask_passes(
        micro_locations,
        micro_valid,
        micro_box_sizes,
        height=int(x_context.shape[-2]),
        width=int(x_context.shape[-1]),
        allow_partial_overlap=float(allow_partial_overlap),
        one_target_per_pass=bool(one_target_per_pass),
    )
    debug["otf_masking_enabled"] = True
    return (x_context, micro_locations, micro_scales, micro_valid, debug), review_rows


def _locality_context_embeddings(outputs: dict, spatial_mode: str) -> torch.Tensor:
    if str(spatial_mode).lower() == "dense":
        return extract_valid_dense_embeddings(outputs, key="context_patches")
    return extract_valid_pooled_embeddings(outputs, key="context_patches")


def _target_mask_from_data_threshold(data_cfg: dict, threshold: float, config_name: str) -> torch.Tensor | None:
    """Build a full-frame valid-target mask from the configured input array."""
    data_root = data_cfg.get("data_root", "data")
    npy_pattern = data_cfg.get("npy_pattern", "*.npy")
    pattern = os.path.join(data_root, npy_pattern)
    paths = resolve_input_files(data_root=data_root, npy_pattern=npy_pattern, input_files=data_cfg.get("input_files"))
    if not paths:
        log_info(f"[{config_name}] target_threshold mask skipped: no files for {pattern}")
        return None
    try:
        arr = np.asarray(_safe_load_npy(paths[0]), dtype=np.float32)
    except Exception as exc:
        log_info(f"[{config_name}] target_threshold mask skipped: failed to load {paths[0]} ({type(exc).__name__}: {exc})")
        return None
    valid = np.isfinite(arr) & (arr > float(threshold))
    if valid.ndim > 2:
        valid = np.any(valid, axis=tuple(range(valid.ndim - 2)))
    if valid.ndim != 2:
        log_info(f"[{config_name}] target_threshold mask skipped: unsupported shape {tuple(arr.shape)}")
        return None
    frac = float(np.mean(valid.astype(np.float32)))
    mask = torch.from_numpy(valid.astype(np.float32))
    log_info(
        f"[{config_name}] target_threshold mask materialized: "
        f"path={paths[0]} threshold={threshold:g} shape={tuple(mask.shape)} fraction={frac:.4f}"
    )
    return mask


def _clear_stale_dashboard_artifacts(session_dir: str) -> None:
    """Remove derived plot/embedding artifacts before forced re-inference."""
    for name in (
        "dash_data.npz",
        "dashboard.html",
        "dashboard_with_masking_demo.html",
        "volumetric_umap_meta.json",
        "rank_diagnostics.json",
        "effective_rank.json",
        "effective_rank.txt",
    ):
        path = os.path.join(session_dir, name)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    results_dir = os.path.join(session_dir, "results")
    if not os.path.isdir(results_dir):
        return
    for name in os.listdir(results_dir):
        if (
            name.endswith("_latent_vectors_full.npy")
            or name.endswith("_pca_xyz.npy")
            or name.endswith("_umap_xyz.npy")
            or name.endswith("_spatial_shape.npy")
            or name in {
                "latent_vectors_full.npy",
                "pca_xyz.npy",
                "umap_xyz.npy",
                "volumetric_umap_indices.npy",
                "volumetric_umap_latents.npy",
                "volumetric_umap_xyz.npy",
                "volumetric_pca_xyz.npy",
            }
        ):
            try:
                os.remove(os.path.join(results_dir, name))
            except OSError:
                pass


def _input_inference_dir(session_dir: str, ordinal: int, sample_key) -> str:
    path = sample_key[0] if isinstance(sample_key, (tuple, list)) and sample_key else str(sample_key)
    stem = os.path.splitext(os.path.basename(str(path)))[0] or "input"
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return os.path.join(session_dir, "inference_inputs", f"{ordinal:03d}_{safe_stem}")


class _MaskingCollator:
    def __init__(
        self,
        model: PyramidGridJEPA,
        return_debug: bool = False,
        require_precomputed_cdd: bool = False,
        target_mask: Optional[torch.Tensor] = None,
        target_threshold: Optional[float] = None,
        locality_refinement: bool = False,
        locality_refinement_n_target: int | None = None,
    ):
        enc_type = str(getattr(model, "encoder_type", "")).lower()
        self.use_cdd = bool(enc_type in CDD_CUBE_ENCODER_TYPES)
        self.otf_masking = bool(getattr(model, "otf_masking", True))
        self.require_precomputed_cdd = bool(require_precomputed_cdd)
        self.return_debug = bool(
            return_debug
            or locality_refinement
            or enc_type in CDD_DEBUG_ENCODER_TYPES
            or enc_type in MASK_MAP_ENCODER_TYPES
        )
        self.mask_scale = float(model.mask_scale)
        self.mask_scale_range = model.mask_scale_range
        self.mask_box_size = int(model.mask_box_size)
        self.mask_box_size_range = model.mask_box_size_range
        self.random_mask_box_per_target = bool(getattr(model, "random_mask_box_per_target", False))
        self.manual_mask_box_sizes = model.manual_mask_box_sizes
        self.encoder_border_margin = int(model.encoder_receptive_field()) // 2 if hasattr(model, "encoder_receptive_field") else 0
        self.invalid_support_border_margin = int(
            model.invalid_support_border_px() if hasattr(model, "invalid_support_border_px") else self.encoder_border_margin
        )
        self.target_mask = target_mask
        self.target_threshold = target_threshold
        self.locality_refinement = bool(locality_refinement)
        self.locality_refinement_n_target = (
            None
            if locality_refinement_n_target is None
            else int(locality_refinement_n_target)
        )
        self.target_allow_partial_overlap = float(getattr(model, "target_allow_partial_overlap", 0.0))
        self.context_kwargs = {
            "sigmas": model.sigmas,
            "mask_fraction": model.mask_fraction,
            "spacing_scale": model.spacing_scale,
            "global_shift": model.global_shift,
            "align_scales": model.align_scales,
            "patch_size": model.patch_size,
            "random_mask_box_per_target": self.random_mask_box_per_target,
            "manual_mask_box_sizes": self.manual_mask_box_sizes,
            "return_debug": self.return_debug,
            "target_invalid_region_skip": model.target_invalid_region_skip,
            "target_invalid_region_values": model.target_invalid_region_values,
            "target_sampling_mode": model.target_sampling_mode,
            "priority_top_percent": model.priority_top_percent,
            "priority_n_target": model.priority_n_target,
            "priority_min_targets_per_map": model.priority_min_targets_per_map,
            "priority_dithering_pixels": model.priority_dithering_pixels,
            "priority_candidate_oversample": model.priority_candidate_oversample,
            # Packed OTF execution schedules overlaps into later passes, so the
            # sampler must retain them instead of rejecting them up front.
            "target_nonoverlap": False if self.otf_masking else getattr(model, "target_nonoverlap", False),
            "target_allow_partial_overlap": getattr(model, "target_allow_partial_overlap", 0.0),
            "mask_box_hardcap": getattr(model, "mask_box_hardcap", None),
            "use_cdd": self.use_cdd,
            "return_candidate_pool": self.locality_refinement,
        }
        if self.locality_refinement_n_target is not None:
            # OTF packing handles spatial overlap in separate encoder passes,
            # so both macro and locality batches can retain this exact count.
            self.context_kwargs["priority_n_target"] = self.locality_refinement_n_target
            self.context_kwargs["priority_min_targets_per_map"] = self.locality_refinement_n_target

    def _sample_mask_params(self) -> tuple[float, int]:
        mask_scale = self.mask_scale
        if self.mask_scale_range is not None:
            lo, hi = self.mask_scale_range
            mask_scale = lo + (hi - lo) * float(torch.rand(()).item()) if hi > lo else lo

        mask_box_size = self.mask_box_size
        if self.mask_box_size_range is not None and not self.random_mask_box_per_target:
            lo, hi = self.mask_box_size_range
            mask_box_size = int(torch.randint(lo, hi + 1, ()).item()) if hi > lo else lo
        return float(mask_scale), int(mask_box_size)

    def __call__(self, batch):
        metadata = None
        if isinstance(batch[0], (tuple, list)) and len(batch[0]) >= 2 and isinstance(batch[0][-1], dict):
            metadata = [item[-1] for item in batch]
            batch = [item[0] if len(item) == 2 else item[:-1] for item in batch]
        use_cdd = isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2
        if self.use_cdd and self.require_precomputed_cdd and not use_cdd:
            raise RuntimeError(
                "CDD precompute cache was built, but the dataloader batch did not include cached CDD. "
                "Refusing to fall back to per-batch constrained_diffusion_decomposition."
            )
        if use_cdd:
            cdd_list = [item[0] for item in batch]
            x_clean_list = [item[1] for item in batch]
            cdd_orig_in = _collate_pad_spatial(cdd_list)
            x_clean = _collate_pad_spatial(x_clean_list)
        else:
            cdd_orig_in = None
            x_clean = _collate_pad_spatial(batch)
        mask_scale, mask_box_size = self._sample_mask_params()
        invalid_pixel_mask = ~torch.isfinite(x_clean)
        border = int(max(0, min(self.encoder_border_margin, int(x_clean.shape[-2]) // 2, int(x_clean.shape[-1]) // 2)))
        # NaN border dilation: expand invalid regions so targets can't be placed
        # where the encoder FOV / CDD support would reach into NaN/no-data regions.
        # Mirrors _apply_encoder_border_invalid_mask in build_jepa.py.
        nan_border = int(max(0, min(self.invalid_support_border_margin, int(x_clean.shape[-2]) // 2, int(x_clean.shape[-1]) // 2)))
        native_border_preapplied = bool(metadata) and all(
            int(item.get("native_invalid_border_px", 0) or 0) >= nan_border
            for item in metadata
        )
        if border > 0 or nan_border > 0:
            # Dilate only native no-data. Crop-edge rejection is a separate
            # encoder-FOV margin; dilating that edge again over-crops every
            # training cutout by border + nan_border.
            native_invalid = invalid_pixel_mask.clone()
            if nan_border > 0 and native_invalid.any() and not native_border_preapplied:
                k = 2 * nan_border + 1
                invalid_float = native_invalid.float()
                dilated = F.max_pool2d(invalid_float, kernel_size=k, stride=1, padding=nan_border)
                invalid_pixel_mask = dilated > 0.0
            else:
                invalid_pixel_mask = native_invalid
            if border > 0:
                invalid_pixel_mask[:, :, :border, :] = True
                invalid_pixel_mask[:, :, int(x_clean.shape[-2]) - border :, :] = True
                invalid_pixel_mask[:, :, :, :border] = True
                invalid_pixel_mask[:, :, :, int(x_clean.shape[-1]) - border :] = True
        batch_target_mask = self.target_mask
        if self.target_threshold is not None and batch_target_mask is None:
            # Auto-generate from raw data: pixels > threshold are valid targets
            raw = x_clean[:, 0] if x_clean.dim() == 4 else x_clean
            batch_target_mask = (raw > self.target_threshold).to(torch.bool)
        if batch_target_mask is not None:
            tgt_h, tgt_w = int(x_clean.shape[-2]), int(x_clean.shape[-1])
            if batch_target_mask.dim() == 2:
                batch_target_mask = batch_target_mask.unsqueeze(0).unsqueeze(0)
            elif batch_target_mask.dim() == 3:
                batch_target_mask = batch_target_mask.unsqueeze(1)
            if batch_target_mask.shape[-2:] != (tgt_h, tgt_w):
                batch_target_mask = F.interpolate(
                    batch_target_mask.float(),
                    size=(tgt_h, tgt_w),
                    mode="nearest",
                ).bool()
        context_data = prepare_context_batch(
            x_clean=x_clean,
            mask_scale=mask_scale,
            mask_box_size=mask_box_size,
            mask_box_size_range=self.mask_box_size_range,
            cdd_orig_in=cdd_orig_in,
            invalid_pixel_mask_in=invalid_pixel_mask,
            target_mask=batch_target_mask,
            **self.context_kwargs,
        )
        if self.locality_refinement_n_target is not None:
            # Some stochastic samplers can return fewer than the requested
            # number after their final validity checks. Refill empty slots from
            # the complete valid candidate catalogue before OTF pass packing.
            if len(context_data) >= 5 and isinstance(context_data[4], dict):
                x_context, target_locations, target_scales, target_valid, source_debug = context_data[:5]
                debug = dict(source_debug)
                candidate_locations = debug.get("candidate_locations")
                candidate_box_sizes = debug.get("candidate_box_sizes")
                candidate_valid = debug.get("candidate_valid")
                target_box_sizes = debug.get("target_box_sizes")
                if all(
                    value is not None
                    for value in (
                        candidate_locations,
                        candidate_box_sizes,
                        candidate_valid,
                        target_box_sizes,
                    )
                ):
                    target_locations = target_locations.clone()
                    target_scales = target_scales.clone()
                    target_valid = target_valid.clone()
                    target_box_sizes = target_box_sizes.clone()
                    expected = int(self.locality_refinement_n_target)
                    current_slots = int(target_valid.shape[1])
                    if current_slots < expected:
                        extra_slots = expected - current_slots
                        target_locations = torch.cat(
                            [
                                target_locations,
                                torch.zeros(
                                    (int(target_locations.shape[0]), extra_slots, int(target_locations.shape[2])),
                                    device=target_locations.device,
                                    dtype=target_locations.dtype,
                                ),
                            ],
                            dim=1,
                        )
                        target_scales = torch.cat(
                            [
                                target_scales,
                                torch.zeros(
                                    (int(target_scales.shape[0]), extra_slots),
                                    device=target_scales.device,
                                    dtype=target_scales.dtype,
                                ),
                            ],
                            dim=1,
                        )
                        target_valid = torch.cat(
                            [
                                target_valid,
                                torch.zeros(
                                    (int(target_valid.shape[0]), extra_slots),
                                    device=target_valid.device,
                                    dtype=torch.bool,
                                ),
                            ],
                            dim=1,
                        )
                        target_box_sizes = torch.cat(
                            [
                                target_box_sizes,
                                torch.zeros(
                                    (int(target_box_sizes.shape[0]), extra_slots),
                                    device=target_box_sizes.device,
                                    dtype=target_box_sizes.dtype,
                                ),
                            ],
                            dim=1,
                        )
                    for sample_index in range(int(target_valid.shape[0])):
                        missing_slots = torch.nonzero(
                            ~target_valid[sample_index], as_tuple=False
                        ).flatten()[: max(0, expected - int(target_valid[sample_index].sum().item()))]
                        if missing_slots.numel() == 0:
                            continue
                        candidate_indices = torch.nonzero(
                            candidate_valid[sample_index], as_tuple=False
                        ).flatten()
                        selected_locations = {
                            tuple(int(v) for v in location.tolist())
                            for location in target_locations[sample_index, target_valid[sample_index]].cpu()
                        }
                        candidate_indices = torch.as_tensor(
                            [
                                int(index)
                                for index in candidate_indices.tolist()
                                if tuple(
                                    int(v)
                                    for v in candidate_locations[sample_index, int(index)].tolist()
                                )
                                not in selected_locations
                            ],
                            device=candidate_indices.device,
                            dtype=torch.long,
                        )
                        if int(candidate_indices.numel()) < int(missing_slots.numel()):
                            continue
                        chosen = candidate_indices[
                            torch.randperm(
                                int(candidate_indices.numel()),
                                device=candidate_indices.device,
                            )[: int(missing_slots.numel())]
                        ]
                        target_locations[sample_index, missing_slots] = candidate_locations[
                            sample_index, chosen
                        ]
                        target_box_sizes[sample_index, missing_slots] = candidate_box_sizes[
                            sample_index, chosen
                        ]
                        valid_scales = target_scales[sample_index, target_valid[sample_index]]
                        if valid_scales.numel() > 0:
                            scale_indices = torch.randint(
                                0,
                                int(valid_scales.numel()),
                                (int(missing_slots.numel()),),
                                device=valid_scales.device,
                            )
                            target_scales[sample_index, missing_slots] = valid_scales[scale_indices]
                        target_valid[sample_index, missing_slots] = True
                    debug["target_box_sizes"] = target_box_sizes
                    context_data = (
                        x_context,
                        target_locations,
                        target_scales,
                        target_valid,
                        debug,
                    )
            target_counts = context_data[3].sum(dim=1)
            expected = int(self.locality_refinement_n_target)
            if bool((target_counts != expected).any().item()):
                raise RuntimeError(
                    "Fixed locality target count could not be produced for every sample: "
                    f"requested={expected} actual={target_counts.tolist()}"
                )
        if self.otf_masking:
            if len(context_data) < 5 or not isinstance(context_data[4], dict):
                raise RuntimeError("Packed OTF masking requires masking debug tensors")
            debug = dict(context_data[4])
            target_locations = context_data[1]
            target_valid = context_data[3]
            target_box_sizes = debug.get("target_box_sizes")
            if target_box_sizes is None:
                raise RuntimeError("Packed OTF masking requires target_box_sizes")
            target_box_sizes = target_box_sizes.clone()
            cdd_box_sizes = debug.get("cdd_box_sizes")
            if cdd_box_sizes is not None and cdd_box_sizes.numel() > 0:
                fallback_boxes = cdd_box_sizes.amax(dim=1, keepdim=True).expand_as(target_box_sizes)
            else:
                fallback_boxes = torch.full_like(target_box_sizes, float(max(1, self.mask_box_size)))
            target_box_sizes = torch.where(
                target_valid & (target_box_sizes <= 0),
                fallback_boxes,
                target_box_sizes,
            )
            debug["target_box_sizes"] = target_box_sizes
            debug["otf_mask_pass_ids"] = pack_target_mask_passes(
                target_locations,
                target_valid,
                target_box_sizes,
                height=int(x_clean.shape[-2]),
                width=int(x_clean.shape[-1]),
                allow_partial_overlap=self.target_allow_partial_overlap,
                one_target_per_pass=self.locality_refinement_n_target is not None,
            )
            debug["otf_masking_enabled"] = True
            context_data = tuple(context_data[:4]) + (debug,)
        if metadata is not None:
            if len(context_data) >= 5 and isinstance(context_data[4], dict):
                debug = dict(context_data[4])
                debug["augment_metadata"] = metadata
                context_data = tuple(context_data[:4]) + (debug,)
            else:
                context_data = tuple(context_data[:4]) + ({"augment_metadata": metadata},)
        x_clean = torch.nan_to_num(x_clean, nan=0.0, posinf=0.0, neginf=0.0)
        return x_clean, context_data


def _prepare_context_from_model(
    model: PyramidGridJEPA,
    x_clean: torch.Tensor,
    return_debug: bool = False,
):
    enc_type = str(getattr(model, "encoder_type", "")).lower()
    need_debug = bool(
        return_debug
        or enc_type in CDD_DEBUG_ENCODER_TYPES
        or enc_type in MASK_MAP_ENCODER_TYPES
    )
    mask_scale, mask_box_size = model.sample_mask_params(device=x_clean.device)
    invalid_pixel_mask = ~torch.isfinite(x_clean)
    border_margin = int(model.encoder_receptive_field()) // 2 if hasattr(model, "encoder_receptive_field") else 0
    border = int(max(0, min(border_margin, int(x_clean.shape[-2]) // 2, int(x_clean.shape[-1]) // 2)))
    # NaN border dilation (mirrors _MaskingCollator and _apply_encoder_border_invalid_mask).
    support_margin = int(model.invalid_support_border_px() if hasattr(model, "invalid_support_border_px") else border)
    nan_border = int(max(0, min(support_margin, int(x_clean.shape[-2]) // 2, int(x_clean.shape[-1]) // 2)))
    if border > 0 or nan_border > 0:
        native_invalid = invalid_pixel_mask.clone()
        if nan_border > 0 and native_invalid.any():
            k = 2 * nan_border + 1
            invalid_float = native_invalid.float()
            dilated = F.max_pool2d(invalid_float, kernel_size=k, stride=1, padding=nan_border)
            invalid_pixel_mask = dilated > 0.0
        else:
            invalid_pixel_mask = native_invalid
        if border > 0:
            invalid_pixel_mask[:, :, :border, :] = True
            invalid_pixel_mask[:, :, int(x_clean.shape[-2]) - border :, :] = True
            invalid_pixel_mask[:, :, :, :border] = True
            invalid_pixel_mask[:, :, :, int(x_clean.shape[-1]) - border :] = True
    return prepare_context_batch(
        x_clean=x_clean,
        sigmas=model.sigmas,
        mask_fraction=model.mask_fraction,
        mask_scale=mask_scale,
        spacing_scale=model.spacing_scale,
        global_shift=model.global_shift,
        align_scales=model.align_scales,
        mask_box_size=mask_box_size,
        mask_box_size_range=model.mask_box_size_range,
        random_mask_box_per_target=getattr(model, "random_mask_box_per_target", False),
        manual_mask_box_sizes=model.manual_mask_box_sizes,
        cdd_mode=model.cdd_mode,
        cdd_constrained=model.cdd_constrained,
        cdd_sm_mode=model.cdd_sm_mode,
        cdd_append_last_residual=model.cdd_append_last_residual,
        cdd_pre_log_transform=model.cdd_pre_log_transform,
        cdd_gaussian_backend=model.cdd_gaussian_backend,
        patch_size=model.patch_size,
        return_debug=need_debug,
        target_invalid_region_skip=model.target_invalid_region_skip,
        target_invalid_region_values=model.target_invalid_region_values,
        target_sampling_mode=model.target_sampling_mode,
        priority_top_percent=model.priority_top_percent,
        priority_n_target=model.priority_n_target,
        priority_min_targets_per_map=model.priority_min_targets_per_map,
        priority_dithering_pixels=model.priority_dithering_pixels,
        priority_candidate_oversample=model.priority_candidate_oversample,
        target_nonoverlap=getattr(model, "target_nonoverlap", False),
        target_allow_partial_overlap=getattr(model, "target_allow_partial_overlap", 0.0),
        mask_box_hardcap=getattr(model, "mask_box_hardcap", None),
        cdd_use_gpu=(x_clean.device.type == "cuda"),
        use_cdd=bool(enc_type in CDD_CUBE_ENCODER_TYPES),
        invalid_pixel_mask_in=invalid_pixel_mask,
    )



@torch.no_grad()
def evaluate_validation(
    model: PyramidGridJEPA,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    vicreg_spatial_mode: str = "pooled",
) -> dict:
    model.eval()
    n = 0
    loss_sum = 0.0
    sim_sum = 0.0
    scale_mse = defaultdict(list)
    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x_clean, context_result = batch
        x_clean = x_clean.to(device, non_blocking=True)
        context_result = _move_to_device(context_result, device)
        x_context, tloc, tscale, tvalid = context_result[:4]
        debug = context_result[4] if len(context_result) == 5 else {}
        context_data = (x_context, tloc, tscale, tvalid, debug)
        outputs = model(x_clean, context_data=context_data)
        loss = model.compute_loss(outputs)
        sim_val, _, _ = compute_sim_var_cov(outputs, spatial_mode=vicreg_spatial_mode)
        ebs = compute_error_by_scale(outputs)
        for s, v in ebs.items():
            scale_mse[s].append(float(v))
        loss_sum += float(loss.item())
        sim_sum += float(sim_val)
        n += 1

    if n == 0:
        val_loss = 0.0
        val_sim = 0.0
    else:
        val_loss = loss_sum / n
        val_sim = sim_sum / n

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        totals = torch.tensor([loss_sum, sim_sum, float(n)], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        global_n = max(1.0, float(totals[2].item()))
        val_loss = float(totals[0].item() / global_n)
        val_sim = float(totals[1].item() / global_n)
        world_size = torch.distributed.get_world_size()
        if world_size > 1:
            gathered = [None] * world_size
            torch.distributed.all_gather_object(gathered, dict(scale_mse))
            merged_scale_mse: dict[float, list[float]] = {}
            for rank_dict in gathered:
                if rank_dict is None:
                    continue
                for s, values in rank_dict.items():
                    merged_scale_mse.setdefault(float(s), []).extend(values)
            scale_mse = merged_scale_mse

    return {
        "val_loss": val_loss,
        "val_sim": val_sim,
        "val_error_by_scale": {float(s): float(np.mean(v)) for s, v in scale_mse.items()},
    }


def reject_removed_config_aliases(cfg: dict) -> None:
    alias_sections = sorted(set(cfg) & {"cdd_scale_space", "masking", "training", "diagnostics"})
    if alias_sections:
        raise ValueError(
            "Removed config alias sections present: "
            f"{alias_sections}. Use canonical data/model/train sections."
        )


def load_config(path: str) -> dict:
    def _deep_merge(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    def _load_with_base(cfg_path: str, seen: set[str]) -> dict:
        abs_path = os.path.abspath(cfg_path)
        if abs_path in seen:
            chain = " -> ".join(list(seen) + [abs_path])
            raise ValueError(f"Cyclic base_config reference detected: {chain}")
        seen.add(abs_path)
        with open(abs_path, "r", encoding="utf-8") as f:
            if abs_path.endswith((".yaml", ".yml")):
                import yaml as _yaml
                cfg = _yaml.safe_load(f)
            else:
                cfg = json.load(f)

        base_ref = cfg.pop("base_config", None)
        if base_ref is not None:
            # Explicit base_config: load and merge
            base_path = base_ref
            if not os.path.isabs(base_path):
                base_path = os.path.join(os.path.dirname(abs_path), base_path)
            base_cfg = _load_with_base(base_path, seen)
            merged = _deep_merge(base_cfg, cfg)
        elif abs_path.endswith("base_pyramid_scaleaware_convnext.yaml"):
            # Loading the base config itself — no merge needed
            merged = cfg
        else:
            # No base_config: auto-merge with project base for safety.
            # Without this, 75+ essential keys silently vanish.
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_path = os.path.join(project_root, "configs", "base_pyramid_scaleaware_convnext.yaml")
            base_cfg = _load_with_base(base_path, seen)
            merged = _deep_merge(base_cfg, cfg)
        seen.remove(abs_path)
        return merged

    cfg = _load_with_base(path, seen=set())
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    reject_removed_config_aliases(cfg)
    removed = {
        "data.log_transform": "model.post_log_transform",
        "data.image_size": "native-resolution data or explicit crop_size",
        "model.log_transform": "model.post_log_transform",
    }
    stale = []
    if "log_transform" in cfg["data"]:
        stale.append("data.log_transform")
    if "image_size" in cfg["data"]:
        stale.append("data.image_size")
    if "log_transform" in cfg["model"]:
        stale.append("model.log_transform")
    if stale:
        replacements = ", ".join(f"{key}->{removed[key]}" for key in stale)
        raise ValueError(f"Removed config keys present: {replacements}. Update the config schema before training.")
    return cfg


def make_session_dir(root: str, config_name: str) -> str:
    path = os.path.join(root, config_name)
    os.makedirs(path, exist_ok=True)
    return path


def _observed_completed_epoch(session_dir: str) -> int:
    """Best-effort completed epoch from durable training logs."""
    epoch_summary_path = os.path.join(session_dir, "epoch_summary.csv")
    best_epoch = 0
    if os.path.exists(epoch_summary_path):
        try:
            with open(epoch_summary_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        best_epoch = max(best_epoch, int(float(row.get("epoch", 0) or 0)))
                    except (TypeError, ValueError):
                        continue
        except OSError:
            pass
    if best_epoch > 0:
        return best_epoch

    metrics_path = os.path.join(session_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        best_epoch = max(best_epoch, int(float(row.get("epoch", 0) or 0)))
                    except (TypeError, ValueError):
                        continue
        except OSError:
            pass
    return best_epoch


def resolve_pipeline_config(model_cfg: dict) -> bool:
    return bool(model_cfg.get("post_log_transform", True))


def resolve_cdd_cache_use_log(model_cfg: dict, data_cfg: dict) -> bool:
    """Select transformed CDD cache only when explicitly requested or legacy default applies."""
    for cfg in (model_cfg, data_cfg):
        if isinstance(cfg, dict) and "cdd_use_log" in cfg:
            return bool(cfg.get("cdd_use_log"))
    return not bool(model_cfg.get("post_log_transform", True))


def resolve_encoder_type_default(model_cfg: dict) -> str:
    """
    Restricted defaults aligned to the supported encoder matrix.
    """
    # Official key: model_key. Keep encoder_type as legacy fallback.
    if "model_key" in model_cfg:
        return str(model_cfg["model_key"])
    if "encoder_type" in model_cfg:
        return str(model_cfg["encoder_type"])
    mode = str(model_cfg.get("mode", "image")).lower()
    if mode == "pyramid":
        return "cdd_scaleaware_convnext"
    return "convnext_dense_masktoken"


def _is_3d_jepa_mode(mode: str) -> bool:
    mode_norm = str(mode).strip().lower().replace(" ", "_")
    return mode_norm == "3d_slab"


def _resolve_3d_crop_depth(
    *,
    data_cfg: dict,
    model_cfg: dict,
    cdd_cache: dict | None,
    default_depth: int,
) -> int:
    value = data_cfg.get("volume_crop_depth", data_cfg.get("crop_depth_3d", None))
    if isinstance(value, str) and value.strip().lower() == "full":
        raise ValueError("3D crop_depth='full' is no longer supported; use an integer slice depth.")
    if value is not None:
        return int(value)
    return int(default_depth)


def _resolve_encoder_alias_2d(name: str) -> str:
    key = str(name).lower()
    alias = {
        # Preferred naming convention (image / image_pyramid prefixes).
        "convnext_image_dense_masked": "convnext_dense_masktoken",
        "cdd_scaleaware_convnext-pyramid-scaleaware": "cdd_scaleaware_convnext",
        "image_pyramid_cdd_scaleaware_convnext": "cdd_scaleaware_convnext",
        "ring_scaleaware_convnext": "cdd_scaleaware_convnext",
        "ring_conv_scaleaware_convnext": "cdd_scaleaware_convnext",
        "multiscale_ring_conv": "cdd_scaleaware_convnext",
        # Supported canonical names.
        "convnext_dense_masktoken": "convnext_dense_masktoken",
        "cdd_scaleaware_convnext": "cdd_scaleaware_convnext",
        "convnext_dense_pyramid": "convnext_dense_pyramid",
        "escnn_c4_pyramid": "escnn_c4_pyramid",
        # Supported aliases.
        "convnext-pyramid-scaleaware": "cdd_scaleaware_convnext",
        "convnext-pyramid": "convnext_dense_pyramid",
        "escnn-c4-pyramid": "escnn_c4_pyramid",
    }
    if key not in alias:
        raise ValueError(
            f"Unsupported 2D model_key/encoder_type={name!r}. "
            f"Allowed aliases: {sorted(alias)}"
        )
    return alias[key]


def _resolve_encoder_alias_3d(name: str) -> str:
    key = str(name).lower()
    alias = {
        "convnext_dense3d": "convnext_dense3d",
        "cdd_scaleaware_convnext3d": "cdd_scaleaware_convnext3d",
    }
    return alias.get(key, str(name))


def _target_invalid_values_from_config(model_cfg: dict, default=("nan",)) -> tuple:
    values = model_cfg.get("target_invalid_region_values", default)
    if values is None:
        return tuple(default)
    if isinstance(values, str):
        return (values,)
    try:
        return tuple(values)
    except TypeError:
        return (values,)


def build_model_from_config(model_cfg: dict, data_cfg: dict, train_cfg: dict, device: torch.device) -> PyramidGridJEPA:
    """Construct a PyramidGridJEPA from config dicts."""
    mask_spacing_scaling = float(model_cfg.get("mask_spacing_scaling", 1.5))
    model_post_log = resolve_pipeline_config(model_cfg=model_cfg)
    resolved_encoder_type = _resolve_encoder_alias_2d(resolve_encoder_type_default(model_cfg))
    resolved_mode = str(model_cfg.get("mode", "image")).lower()
    if resolved_mode == "image":
        allowed_image = {"convnext_dense_masktoken"}
        if resolved_encoder_type not in allowed_image:
            raise ValueError(
                f"Unsupported image-mode encoder_type={resolved_encoder_type}. "
                "Allowed: convnext_dense_masktoken."
            )
    elif resolved_mode == "pyramid":
        allowed_pyramid = {
            "cdd_scaleaware_convnext",
            "convnext_dense_pyramid",
            "escnn_c4_pyramid",
        }
        if resolved_encoder_type not in allowed_pyramid:
            raise ValueError(
                f"Unsupported pyramid-mode encoder_type={resolved_encoder_type}. "
                "Allowed: cdd_scaleaware_convnext, convnext_dense_pyramid, escnn_c4_pyramid."
            )
    else:
        raise ValueError(f"Unsupported mode={resolved_mode}. Allowed: image, pyramid.")

    patch_size = int(model_cfg.get("patch_size", 3))
    if patch_size <= 0:
        raise ValueError(f"model.patch_size must be positive, got {patch_size}.")
    if patch_size % 2 == 0:
        raise ValueError(f"model.patch_size must be odd, got {patch_size}.")

    mask_scale_cfg = model_cfg.get("mask_size_scaling", 1.0)
    mask_box_cfg = model_cfg.get("mask_size", 16)
    manual_mask_box_sizes_cfg = model_cfg.get("mask_size_manual")

    normalize_loss_l2 = bool(model_cfg.get("normalize_loss_l2", model_cfg.get("normalize_loss", False)))
    active_target_fraction = float(model_cfg.get("active_target_fraction", model_cfg.get("mask_fraction", 1.0)))
    return PyramidGridJEPA(
        latent_channels=model_cfg.get("latent_channels", 32),
        predictor_hidden=model_cfg.get("predictor_hidden"),
        patch_size=patch_size,
        sigmas=tuple(model_cfg.get("sigmas", [2, 4, 8, 16])),
        mask_fraction=active_target_fraction,
        mask_scale=mask_scale_cfg,
        mask_scale_range=None,
        spacing_scale=mask_spacing_scaling,
        global_shift=model_cfg.get("global_shift", True),
        align_scales=model_cfg.get("align_scales", True),
        mask_box_size=mask_box_cfg,
        mask_box_size_range=None,
        random_mask_box_per_target=bool(model_cfg.get("random_mask_box_per_target", False)),
        manual_mask_box_sizes=manual_mask_box_sizes_cfg,
        cdd_mode=model_cfg.get("cdd_mode", data_cfg.get("cdd_mode", "log")),
        cdd_constrained=model_cfg.get("cdd_constrained", data_cfg.get("cdd_constrained", True)),
        cdd_sm_mode=model_cfg.get("cdd_sm_mode", data_cfg.get("cdd_sm_mode", "reflect")),
        cdd_append_last_residual=bool(model_cfg.get("cdd_append_last_residual", True)),
        cdd_pre_log_transform=bool(model_cfg.get("cdd_pre_log_transform", False)),
        cdd_gaussian_backend=model_cfg.get("cdd_gaussian_backend", data_cfg.get("cdd_gaussian_backend", "cuda")),
        post_log_transform=model_cfg.get("post_log_transform", model_post_log),
        log_eps=model_cfg.get("log_eps", float(data_cfg.get("log_eps", 1.0))),
        cdd_log_std_floor_mult=model_cfg.get("cdd_log_std_floor_mult", 0.05),
        ema_momentum=model_cfg.get("ema_momentum", train_cfg.get("momentum", 0.996)),
        normalize_loss_l2=normalize_loss_l2,
        predictor_layernorm=model_cfg.get("predictor_layernorm", True),
        predictor_spatial_conv=model_cfg.get("predictor_spatial_conv", False),
        projector_conv=bool(model_cfg.get("projector_conv", True)),
        predictor_residual=model_cfg.get("predictor_residual", False),
        mode=resolved_mode,
        encoder_type=resolved_encoder_type,
        encoder_width=model_cfg.get("encoder_width", model_cfg.get("latent_channels", 32)),
        encoder_depth=model_cfg.get("encoder_depth", 4),
        encoder_kernel_size=model_cfg.get("encoder_kernel_size", 7),
        convnext_layer_dilations=model_cfg.get("convnext_layer_dilations"),
        encoder_norm_type=model_cfg.get("encoder_norm_type"),
        encoder_norm_groups=model_cfg.get("encoder_norm_groups"),
        encoder_norm_eps=model_cfg.get("encoder_norm_eps"),
        scaleaware_feat_channels=int(model_cfg.get("scaleaware_feat_channels", 8)),
        scaleaware_adapter_kernel_size=int(model_cfg.get("scaleaware_adapter_kernel_size", 3)),
        scaleaware_fusion_type=str(model_cfg.get("scaleaware_fusion_type", "concat")),
        scaleaware_norm_per_scale=bool(model_cfg.get("scaleaware_norm_per_scale", False)),
        scaleaware_adapter_norm=bool(model_cfg.get("scaleaware_adapter_norm", True)),
        scaleaware_final_norm=bool(model_cfg.get("scaleaware_final_norm", True)),
        scaleaware_stem_norm=bool(model_cfg.get("scaleaware_stem_norm", True)),
        encoder_final_norm_type=str(model_cfg.get("encoder_final_norm_type", "layernorm")),
        encoder_head_bias=bool(model_cfg.get("encoder_head_bias", True)),
        target_invalid_region_skip=bool(model_cfg.get("target_invalid_region_skip", True)),
        target_invalid_region_values=_target_invalid_values_from_config(model_cfg),
        target_sampling_mode=str(model_cfg.get("target_sampling_mode", "random")),
        priority_top_percent=float(model_cfg.get("priority_top_percent", 5.0)),
        priority_n_target=model_cfg.get("priority_n_target", 20),
        priority_min_targets_per_map=int(model_cfg.get("priority_min_targets_per_map", 0)),
        priority_dithering_pixels=int(model_cfg.get("priority_dithering_pixels", model_cfg.get("target_dithering_pixels", 6))),
        priority_candidate_oversample=float(model_cfg.get("priority_candidate_oversample", 3.0)),
        use_symmetric_feature_loss=bool(model_cfg.get("use_symmetric_feature_loss", False))
        and float(train_cfg.get("symmetry_loss_weight", 0.0)) > 0.0,
        target_nonoverlap=bool(model_cfg.get("target_nonoverlap", True)),
        target_allow_partial_overlap=float(model_cfg.get("target_allow_partial_overlap", 0.0)),
        otf_masking=bool(model_cfg.get("otf_masking", True)),
        mask_box_hardcap=model_cfg.get("mask_box_hardcap"),
        nan_border_sigma_multiplier=float(model_cfg.get("nan_border_sigma_multiplier", data_cfg.get("nan_border_sigma_multiplier", 3.0))),
        invalid_support_border_mode=model_cfg.get(
            "invalid_support_border_mode",
            data_cfg.get("invalid_support_border_mode", "encoder_rf"),
        ),
        use_grn=bool(model_cfg.get("use_grn", True)),
    ).to(device)


def build_model3d_from_config(model_cfg: dict, train_cfg: dict, device: torch.device) -> PyramidGridJEPA3D:
    mode = str(model_cfg.get("mode", "")).strip().lower().replace(" ", "_")
    if mode != "3d_slab":
        raise ValueError(
            f"Unsupported 3D JEPA mode={model_cfg.get('mode')}. "
            "Use model.mode='3d_slab'."
        )
    if "volumetric_mode" in model_cfg:
        raise ValueError("model.volumetric_mode was removed; use model.mode='3d_slab'.")
    enc_type = _resolve_encoder_alias_3d(model_cfg.get("encoder_type", "cdd_scaleaware_convnext3d")).lower()
    allowed_3d = {"convnext_dense3d", "cdd_scaleaware_convnext3d"}
    if enc_type not in allowed_3d:
        raise ValueError(
            f"Unsupported 3D encoder_type={enc_type}. "
            "Allowed: convnext_dense3d, cdd_scaleaware_convnext3d."
        )
    fusion = str(model_cfg.get("scaleaware_fusion_type", "gate"))
    if "dense" in enc_type:
        fusion = "concat"
    normalize_loss_l2 = bool(model_cfg.get("normalize_loss_l2", model_cfg.get("normalize_loss", False)))
    encoder_depth = int(model_cfg.get("encoder_depth", 3))
    encoder_kernel_size = int(model_cfg.get("encoder_kernel_size", 5))
    sigmas = tuple(model_cfg.get("sigmas", [2, 4, 8, 16]))
    cdd_num_channels = int(model_cfg.get("cdd_num_channels", len(sigmas)))
    expected_default_scales = cdd_num_channels
    num_scales = 1 if enc_type == "convnext_dense3d" else int(model_cfg.get("num_scales", expected_default_scales))
    encoder_rf_depth = compute_3d_encoder_receptive_field_depth(
        encoder_depth=encoder_depth,
        encoder_kernel_size=encoder_kernel_size,
    )
    patch_size = int(model_cfg.get("patch_size", 2))
    mask_box_size_3d = _max_effective_mask_box_size(
        sigmas=sigmas,
        mask_scale=float(model_cfg.get("mask_size_scaling", 1.0)),
        mask_box_size=int(model_cfg.get("mask_size", 0)),
        inner_target_size=patch_size,
        hardcap=model_cfg.get("mask_box_hardcap"),
        manual_mask_box_sizes=model_cfg.get("mask_size_manual"),
    )
    return PyramidGridJEPA3D(
        latent_channels=int(model_cfg.get("latent_channels", 16)),
        scale_channels=int(model_cfg.get("scale_channels", model_cfg.get("encoder_width", 8))),
        num_scales=int(num_scales),
        encoder_type=enc_type,
        patch_size=patch_size,
        num_targets=model_cfg.get("num_targets", "auto"),
        encoder_depth=encoder_depth,
        encoder_kernel_size=encoder_kernel_size,
        encoder_stride=int(model_cfg.get("encoder_stride", 1)),
        ema_momentum=float(model_cfg.get("ema_momentum", train_cfg.get("momentum", 0.996))),
        normalize_loss_l2=normalize_loss_l2,
        post_log_transform=bool(model_cfg.get("post_log_transform", True)),
        log_eps=float(model_cfg.get("log_eps", 1e-6)),
        cdd_log_std_floor_mult=float(model_cfg.get("cdd_log_std_floor_mult", 0.05)),
        fusion=fusion,
        mask_box_size=int(mask_box_size_3d),
        num_mask_boxes=int(model_cfg.get("num_mask_boxes", 8)),
        slab_depth=int(model_cfg.get("slab_depth", max(1, patch_size))),
        use_symmetric_feature_loss=bool(model_cfg.get("use_symmetric_feature_loss", False))
        and float(train_cfg.get("symmetry_loss_weight", 0.0)) > 0.0,
        use_film=bool(model_cfg.get("use_film", True)),
        use_per_scale_adapters=bool(model_cfg.get("use_per_scale_adapters", False)),
        priority_candidate_oversample=float(model_cfg.get("priority_candidate_oversample", 3.0)),
        priority_min_targets_per_map=int(model_cfg.get("priority_min_targets_per_map", 0)),
        target_nonoverlap=bool(model_cfg.get("target_nonoverlap", True)),
        target_allow_partial_overlap=float(model_cfg.get("target_allow_partial_overlap", 0.0)),
        encoder_receptive_field_depth=encoder_rf_depth,
        use_grn=bool(model_cfg.get("use_grn", True)),
        stem_norm=bool(model_cfg.get("scaleaware_stem_norm", True)),
        norm_per_scale=bool(model_cfg.get("scaleaware_norm_per_scale", True)),
        adapter_norm=bool(model_cfg.get("scaleaware_adapter_norm", True)),
        final_norm=bool(model_cfg.get("scaleaware_final_norm", True)),
        activation_checkpointing=bool(model_cfg.get("activation_checkpointing", True)),
        target_invalid_region_skip=bool(model_cfg.get("target_invalid_region_skip", True)),
        target_invalid_region_values=_target_invalid_values_from_config(model_cfg),
        encoder_border_margin_xy=int(max(0, encoder_rf_depth // 2)),
        mask_box_hardcap=model_cfg.get("mask_box_hardcap"),
    ).to(device)


def run_training(config: dict, config_name: str, sessions_root: str = "sessions") -> str:
    reject_removed_config_aliases(config)
    _ensure_training_logging()
    # ── DDP: detect torchrun-launched multi-GPU ──
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        from datetime import timedelta
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler

        required_ddp_env = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
        missing_ddp_env = [name for name in required_ddp_env if name not in os.environ]
        if missing_ddp_env:
            raise RuntimeError(
                "LOCAL_RANK is set but this does not look like a complete torchrun launch. "
                f"Missing environment variables: {missing_ddp_env}."
            )
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            ddp_backend = "nccl"
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            ddp_backend = "gloo"
            device = torch.device("cpu")
        dist.init_process_group(backend=ddp_backend, timeout=timedelta(minutes=30))
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    is_main_process = (global_rank == 0)
    cdd_cache_replicas = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1"))) if is_ddp else 1

    if is_main_process:
        log_info(
            f"[{config_name}] Backend discovered: device={device.type}, "
            f"cuda_available={torch.cuda.is_available()}, "
            f"mps_available={device.type == 'mps'}, "
            f"ddp={'on' if is_ddp else 'off'}, "
            f"local_rank={local_rank}, "
            f"global_rank={global_rank}, "
            f"world_size={world_size}"
        )

    train_cfg = config["train"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    configured_locality_refinement = bool(train_cfg.get("locality_refinement", False))
    vanilla_matched_steps = bool(train_cfg.get("vanilla_matched_steps", False))
    if configured_locality_refinement and vanilla_matched_steps:
        raise ValueError("Enable either locality_refinement or vanilla_matched_steps, not both")
    locality_refinement = configured_locality_refinement or vanilla_matched_steps
    locality_refinement_n_step = int(train_cfg.get("locality_refinement_n_step", 3))
    locality_refinement_n_target_raw = train_cfg.get(
        "n_target",
        train_cfg.get("locality_refinement_n_target"),
    )
    locality_refinement_n_target = (
        None
        if locality_refinement_n_target_raw is None
        else int(locality_refinement_n_target_raw)
    )
    locality_refinement_knn_space = (
        "random"
        if vanilla_matched_steps
        else str(train_cfg.get("locality_refinement_knn_space", "spatial")).strip().lower()
    )
    locality_spatial_fov_factor = float(
        train_cfg.get("locality_refinement_spatial_fov_factor", 1.0)
    )
    if locality_refinement and locality_refinement_n_step <= 0:
        raise ValueError("train.locality_refinement_n_step must be positive when locality refinement is enabled")
    if locality_refinement_n_target is not None and locality_refinement_n_target <= 0:
        raise ValueError("train.n_target must be a positive integer or null")
    if locality_refinement_knn_space not in {"random", "spatial", "latent"}:
        raise ValueError("train.locality_refinement_knn_space must be 'spatial' or 'latent'")
    if locality_spatial_fov_factor <= 0.0:
        raise ValueError("train.locality_refinement_spatial_fov_factor must be positive")
    if "num_threads" in train_cfg:
        num_threads = max(1, int(train_cfg["num_threads"]))
        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(num_threads)
        except RuntimeError:
            pass
        if is_main_process:
            log_info(f"[{config_name}] torch_threads={num_threads}")
    seed = int(train_cfg.get("seed", train_cfg.get("split_seed", 42)))
    rank_seed = seed + int(global_rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed % 2**32)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    if is_main_process:
        log_info(f"[{config_name}] global_seed={seed} rank_seed={rank_seed}")
    is_3d_mode = _is_3d_jepa_mode(model_cfg.get("mode", "image"))

    # Optional WandB logging (config-controlled, main process only)
    _use_wandb = False
    if is_main_process and bool(train_cfg.get("wandb_enabled", False)):
        try:
            import wandb

            _use_wandb = True
            wandb.init(
                project=train_cfg.get("wandb_project", "jepa-training"),
                name=config_name,
                config=config,
                dir=os.path.join(sessions_root, "wandb"),
            )
            log_info(f"[{config_name}] wandb initialized project={train_cfg.get('wandb_project', 'jepa-training')}")
        except ImportError:
            log_info(f"[{config_name}] wandb not installed; pip install wandb to enable")

    session_dir = make_session_dir(sessions_root, config_name)
    set_error_log_path(os.path.join(session_dir, "errors.log"))
    os.makedirs(session_dir, exist_ok=True)
    model_ckpt_path = os.path.join(session_dir, "model_last.pt")
    resume_ckpt_path = os.path.join(session_dir, "checkpoint_last.pt")
    resume_from_existing = os.path.exists(model_ckpt_path)

    if is_main_process:
        with open(os.path.join(session_dir, "config_used.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    if is_ddp:
        dist.barrier()

    input_type = str(data_cfg.get("input_type", "image")).lower()
    allowed_input_types = {"image", "cube", "image_batch"}
    if input_type not in allowed_input_types:
        raise ValueError(
            f"Unsupported data.input_type={input_type}. "
            "Allowed: image, cube, image_batch."
        )
    if input_type == "cube" and not is_3d_mode:
        raise ValueError(
            "data.input_type='cube' requires model.mode='3d_slab'."
        )
    if is_3d_mode and input_type != "cube":
        raise ValueError("3D model modes require data.input_type='cube'.")
    image_batch_inference = (input_type == "image_batch")

    if is_main_process:
        _write_data_profile(data_cfg=data_cfg, session_dir=session_dir, config_name=config_name)

    if is_main_process:
        log_info(
            f"[{config_name}] Resolved pipeline: "
            "dataset_preprocess=normalize01, "
            f"model.post_log_transform={model_cfg.get('post_log_transform', True)}, "
            f"cdd_mode={model_cfg.get('cdd_mode', data_cfg.get('cdd_mode', 'log'))}"
        )

    model = build_model3d_from_config(model_cfg, train_cfg, device) if is_3d_mode else build_model_from_config(model_cfg, data_cfg, train_cfg, device)
    locality_spatial_radius_px = None
    if locality_refinement:
        if is_3d_mode:
            raise ValueError("train.locality_refinement currently supports 2D image/pyramid training only")
        if not bool(getattr(model, "otf_masking", True)):
            raise ValueError("train.locality_refinement requires model.otf_masking=true")
        if str(getattr(model, "target_sampling_mode", "random")) not in {
            "random",
            "priority",
            "priority_small_scale",
        }:
            raise ValueError(
                "train.locality_refinement requires random, priority, or priority_small_scale target sampling"
            )
        if locality_refinement_knn_space == "spatial":
            locality_spatial_radius_px = (
                locality_spatial_fov_factor * float(model.encoder_receptive_field())
            )
        mode_label = "vanilla_matched" if vanilla_matched_steps else locality_refinement_knn_space
        log_info(
            f"[{config_name}] matched_target_batches=on mode={mode_label} "
            f"batches_per_outer_step={1 + locality_refinement_n_step} "
            f"targets_per_batch={'macro_count' if locality_refinement_n_target is None else locality_refinement_n_target} "
            f"one_target_per_otf_pass={locality_refinement_n_target is not None} "
            f"spatial_radius_px={locality_spatial_radius_px}"
        )
    if is_main_process and not is_3d_mode:
        log_info(
            f"[{config_name}] masking_execution="
            f"{'packed_otf' if getattr(model, 'otf_masking', True) else 'legacy_single_pass'}"
        )

    # DDP: wrap model for multi-GPU
    ddp_find_unused_parameters = bool(train_cfg.get("ddp_find_unused_parameters", True))
    ddp_kwargs = dict(find_unused_parameters=ddp_find_unused_parameters)
    if device.type == "cuda":
        ddp_kwargs.update(device_ids=[local_rank], output_device=local_rank)

    def _ddp_wrap(m):
        nonlocal model_without_ddp
        model_without_ddp = m
        if is_ddp:
            wrapped = DDP(m, **ddp_kwargs)
            model_without_ddp = wrapped.module
            return wrapped
        return m

    model = _ddp_wrap(model)
    model_without_ddp = model.module if is_ddp else model
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
        resume_state = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in resume_state:
            try:
                missing, unexpected = model.load_state_dict(
                    resume_state["model_state_dict"],
                    strict=not allow_partial_resume,
                )
            except RuntimeError as e:
                # Common during architecture evolution (e.g. channel-count changes).
                log_error("resume_model_load_state_dict", e)
                if not allow_partial_resume:
                    raise RuntimeError(
                        "Checkpoint model-state load failed. "
                        "Set train.allow_partial_resume=true only if this architecture change is intentional."
                    ) from e
                log_info(
                    f"[{config_name}] warning: resume checkpoint model-state load failed; "
                    "skipping checkpoint and starting fresh model/optimizer/scaler."
                )
                resume_state = None
                start_epoch = 0
                model = (
                    build_model3d_from_config(model_cfg, train_cfg, device)
                    if is_3d_mode
                    else build_model_from_config(model_cfg, data_cfg, train_cfg, device)
                )
                model_without_ddp = model
                model = _ddp_wrap(model)
                missing, unexpected = [], []
            log_info(f"[{config_name}] Resume model: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
            if missing:
                log_info(f"[{config_name}] resume_model missing_keys={len(missing)} keys: {missing[:10]}")
            if unexpected:
                log_info(f"[{config_name}] resume_model unexpected_keys={len(unexpected)} keys: {unexpected[:10]}")
            if missing or unexpected:
                error_msg = (
                    f"CRITICAL: Checkpoint architecture mismatch!\n"
                    f"  Missing keys: {len(missing)} (e.g., {missing[:3]})\n"
                    f"  Unexpected keys: {len(unexpected)} (e.g., {unexpected[:3]})"
                )
                if not allow_partial_resume:
                    raise RuntimeError(error_msg + "\nSet train.allow_partial_resume=true if intentional.")
                log_info("=" * 60)
                log_info(f"[WARNING] {error_msg}")
                log_info("[WARNING] Proceeding anyway due to allow_partial_resume=True")
                log_info("=" * 60)
                if resume_mismatch_action == "error":
                    raise RuntimeError(
                        "Checkpoint model-state mismatch detected and allow_partial_resume=False. "
                        "Set train.allow_partial_resume=true to permit partial model resume."
                    )
                log_info(
                    f"[{config_name}] Warning: checkpoint model-state mismatch; "
                    "skipping resume checkpoint and starting fresh model/optimizer/scaler."
                )
                resume_state = None
                start_epoch = 0
                model = (
                    build_model3d_from_config(model_cfg, train_cfg, device)
                    if is_3d_mode
                    else build_model_from_config(model_cfg, data_cfg, train_cfg, device)
                )
                model = _ddp_wrap(model)
                log_info(f"[{config_name}] resume_checkpoint_ignored={resume_ckpt_path}")
        if resume_state is not None:
            start_epoch = int(resume_state.get("epoch", 0))
            log_info(f"resume_checkpoint={resume_ckpt_path} start_epoch={start_epoch}")
    elif resume_from_existing:
        resume_model_ignored = False
        try:
            missing, unexpected = model.load_state_dict(
                torch.load(model_ckpt_path, map_location=device, weights_only=True),
                strict=not allow_partial_resume,
            )
        except RuntimeError as e:
            # Common during architecture evolution (e.g. channel-count changes).
            log_error("resume_model_load_state_dict", e)
            if not allow_partial_resume:
                raise RuntimeError(
                    "Model checkpoint load failed. "
                    "Set train.allow_partial_resume=true only if this architecture change is intentional."
                ) from e
            log_info(
                f"[{config_name}] warning: model checkpoint load failed; "
                "ignoring model_last and starting fresh model/optimizer/scaler."
            )
            model = (
                build_model3d_from_config(model_cfg, train_cfg, device)
                if is_3d_mode
                else build_model_from_config(model_cfg, data_cfg, train_cfg, device)
            )
            model = _ddp_wrap(model)
            missing, unexpected = [], []
            resume_model_ignored = True
        log_info(f"[{config_name}] Resume model: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
        if missing:
            log_info(f"[{config_name}] resume_model missing_keys={len(missing)} keys: {missing[:10]}")
        if unexpected:
            log_info(f"[{config_name}] resume_model unexpected_keys={len(unexpected)} keys: {unexpected[:10]}")
        if missing or unexpected:
            error_msg = (
                f"CRITICAL: Model checkpoint mismatch!\n"
                f"  Missing keys: {len(missing)} (e.g., {missing[:3]})\n"
                f"  Unexpected keys: {len(unexpected)} (e.g., {unexpected[:3]})"
            )
            if not allow_partial_resume:
                raise RuntimeError(error_msg + "\nSet train.allow_partial_resume=true if intentional.")
            log_info("=" * 60)
            log_info(f"[WARNING] {error_msg}")
            log_info("[WARNING] Proceeding anyway due to allow_partial_resume=True")
            log_info("=" * 60)
            log_info(
                f"[{config_name}] warning: model checkpoint mismatch; "
                "ignoring model_last and starting fresh model/optimizer/scaler."
            )
            model = (
                build_model3d_from_config(model_cfg, train_cfg, device)
                if is_3d_mode
                else build_model_from_config(model_cfg, data_cfg, train_cfg, device)
            )
            model = _ddp_wrap(model)
            log_info(f"[{config_name}] resume_model_ignored={model_ckpt_path}")
        else:
            if not resume_model_ignored:
                log_info(f"resume_model={model_ckpt_path}")

    epochs = int(train_cfg.get("epochs", 20))
    force_recompute_inference = bool(train_cfg.get("force_recompute_inference", False))
    compute_effective_rank = bool(train_cfg.get("compute_effective_rank", False))
    rerun_completed = bool(train_cfg.get("rerun_completed", False))
    observed_epoch = _observed_completed_epoch(session_dir)
    inference_outputs_path = os.path.join(session_dir, "inference_outputs.pt")
    has_inference_outputs = (
        os.path.exists(inference_outputs_path)
        and os.path.getsize(inference_outputs_path) > 0
    )
    inference_summary_path = os.path.join(session_dir, "jepa_energy_summary.json")
    inference_summary = {}
    if os.path.exists(inference_summary_path):
        try:
            with open(inference_summary_path, "r", encoding="utf-8") as f:
                inference_summary = json.load(f)
        except Exception:
            inference_summary = {}
    data_profile_path = os.path.join(session_dir, "data_profile.json")
    requires_tiled_2d_inference = False
    tile_size_for_skip = train_cfg.get("inference_tile_size", train_cfg.get("full_volume_spatial_tile_size", 512))
    try:
        tile_size_for_skip = int(tile_size_for_skip)
    except (TypeError, ValueError):
        tile_size_for_skip = 0
    if not is_3d_mode and tile_size_for_skip > 0 and os.path.exists(data_profile_path):
        try:
            with open(data_profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            first_shape = (profile.get("files") or [{}])[0].get("shape") or []
            if len(first_shape) >= 2:
                requires_tiled_2d_inference = (
                    int(first_shape[-2]) > tile_size_for_skip
                    or int(first_shape[-1]) > tile_size_for_skip
                )
        except Exception:
            requires_tiled_2d_inference = False
    has_tiled_2d_inference = bool(inference_summary.get("inference_tiled_dense_2d", False))
    requires_3d_full_xy_slice = is_3d_mode
    has_3d_full_xy_slice = bool(inference_summary.get("inference_3d_full_xy_slice", False))
    stale_full_frame_reason = None
    if not is_3d_mode and has_inference_outputs:
        stale_full_frame_reason = _full_frame_inference_shape_mismatch(
            session_dir=session_dir,
            train_cfg=train_cfg,
            inference_outputs_path=inference_outputs_path,
        )
        if stale_full_frame_reason is not None:
            log_info(
                f"[{config_name}] stale inference ignored: {stale_full_frame_reason}; "
                "forcing post-training inference recompute."
            )
            has_inference_outputs = False
            has_tiled_2d_inference = False
            force_recompute_inference = True
    has_required_inference = has_inference_outputs and (
        has_tiled_2d_inference if requires_tiled_2d_inference else True
    ) and (
        has_3d_full_xy_slice if requires_3d_full_xy_slice else True
    )
    if 0 < observed_epoch < start_epoch:
        log_info(
            f"[{config_name}] checkpoint epoch {start_epoch} exceeds observed completed epoch "
            f"{observed_epoch}; resuming from observed logs instead."
        )
        start_epoch = observed_epoch
    if (
        start_epoch >= epochs
        and observed_epoch >= epochs
        and has_required_inference
        and not force_recompute_inference
        and not compute_effective_rank
        and not rerun_completed
    ):
        log_info(
            f"[{config_name}] checkpoint epoch {start_epoch} and observed epoch {observed_epoch} "
            f"already >= configured epochs {epochs}, required inference artifacts present; "
            "skipping completed session before CDD/dataloader setup "
            "(set train.rerun_completed=true to force rerun)."
        )
        return session_dir
    if start_epoch >= epochs and observed_epoch >= epochs and not has_required_inference:
        missing = (
            "full-XY 3D slice inference"
            if requires_3d_full_xy_slice and has_inference_outputs
            else "tiled 2D inference"
            if requires_tiled_2d_inference and has_inference_outputs
            else "inference_outputs.pt"
        )
        log_info(
            f"[{config_name}] training complete but {missing} missing; "
            "continuing to dataloader/inference setup."
        )

    # --- image_batch pre-selection ---
    image_batch_selected_indices = None
    image_batch_n_sample = data_cfg.get("image_batch_n_sample", None)
    if input_type == "image_batch" and image_batch_n_sample is not None:
        data_root = data_cfg.get("data_root", "data")
        npy_pattern = data_cfg.get("npy_pattern", "*.npy")
        npy_files = [
            p for p in resolve_input_files(data_root=data_root, npy_pattern=npy_pattern, input_files=data_cfg.get("input_files"))
            if p.endswith(".npy")
        ]
        selected = {}
        sel_path = os.path.join(session_dir, "selected_slices.json")
        rng = random.Random(int(train_cfg.get("split_seed", 42)))
        for fpath in npy_files:
            arr_mm = _safe_load_npy(fpath, mmap_mode="r")
            if arr_mm.ndim != 3:
                continue
            n_total = int(arr_mm.shape[0])
            if str(image_batch_n_sample).strip().lower() == "full":
                sel_idx = list(range(n_total))
            else:
                n_sel = int(image_batch_n_sample)
                n_sel = max(1, min(n_sel, n_total))
                sel_idx = sorted(rng.sample(range(n_total), n_sel))
            selected[fpath] = sel_idx
        image_batch_selected_indices = selected
        if is_main_process:
            with open(sel_path, "w", encoding="utf-8") as f:
                json.dump({k: list(v) for k, v in selected.items()}, f, indent=2)
        total_selected = sum(len(v) for v in selected.values())
        if is_main_process:
            log_info(
                f"[{config_name}] image_batch_n_sample={image_batch_n_sample} "
                f"files={len(selected)} total_selected={total_selected} "
                f"saved_to={sel_path}"
            )

    # --- Pre-compute CDD once on GPU, store in CPU RAM ---
    # Only CDD/pyramid encoders require CDD channels. Plain image ConvNeXt
    # mask-token runs build masks directly on the raw image.
    encoder_type_lower = str(getattr(model_without_ddp, "encoder_type", "")).lower()
    uses_cdd_channels = encoder_type_lower in CDD_CUBE_ENCODER_TYPES
    if uses_cdd_channels and not bool(data_cfg.get("cdd_precompute", True)):
        data_cfg["cdd_precompute"] = True
        if is_main_process:
            log_info(
                f"[{config_name}] CDD precompute: forcing on because "
                f"encoder_type={encoder_type_lower} requires cached CDD channels"
            )
    cdd_cache = _precompute_cdd_cache(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        device=device,
        config_name=config_name,
        session_dir=session_dir,
        cache_replicas=cdd_cache_replicas,
    ) if uses_cdd_channels else None
    if uses_cdd_channels and not cdd_cache:
        raise RuntimeError(
            f"[{config_name}] encoder_type={encoder_type_lower} requires a precomputed CDD cache, "
            "but no CDD entries were built. Check data.data_root/npy_pattern, cdd_precompute_max_files, "
            "and cdd_precompute_max_gb."
        )
    if is_main_process:
        _write_cdd_cache_profile(cdd_cache=cdd_cache, session_dir=session_dir, config_name=config_name)

    if is_3d_mode:
        encoder_rf_depth_3d = compute_3d_encoder_receptive_field_depth(
            encoder_depth=int(model_cfg.get("encoder_depth", 3)),
            encoder_kernel_size=int(model_cfg.get("encoder_kernel_size", 5)),
        )
        target_slab_depth_3d = max(
            int(model_cfg.get("patch_size", 2)),
            int(model_cfg.get("slab_depth", max(1, int(model_cfg.get("patch_size", 2))))),
        )
        auto_crop_depth_3d = int(encoder_rf_depth_3d + target_slab_depth_3d - 1)
        crop_depth_3d = _resolve_3d_crop_depth(
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            cdd_cache=cdd_cache,
            default_depth=auto_crop_depth_3d,
        )
        min_crop_depth_3d = auto_crop_depth_3d
        if crop_depth_3d < min_crop_depth_3d:
            raise ValueError(
                "3D crop depth is too small: "
                f"got {crop_depth_3d}, required at least {min_crop_depth_3d} "
                f"(encoder_rf={encoder_rf_depth_3d}, target_slab_depth={target_slab_depth_3d})"
            )
        if is_main_process:
            log_info(
                f"[{config_name}] {model_without_ddp.mode} geometry: spatial_crop="
                f"{int(data_cfg.get('volume_crop_size', data_cfg.get('crop_size_3d', 64)))} "
                f"crop_depth={crop_depth_3d} encoder_rf_depth={encoder_rf_depth_3d} "
                f"target_depth={target_slab_depth_3d}"
            )
        inference_crop_depth_3d = int(train_cfg.get("inference_crop_depth_3d", crop_depth_3d))
        if inference_crop_depth_3d <= 0:
            raise ValueError(f"train.inference_crop_depth_3d must be positive, got {inference_crop_depth_3d}")
        dataset = JEPA3DCropDataset(
            data_root=data_cfg.get("data_root", "data"),
            npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
            input_files=data_cfg.get("input_files"),
            num_samples=int(data_cfg.get("num_samples", 2000)),
            crop_size=int(data_cfg.get("volume_crop_size", data_cfg.get("crop_size_3d", 64))),
            crop_depth=crop_depth_3d,
            slab_depth=crop_depth_3d,
            depth_axis=int(data_cfg.get("volume_depth_axis", data_cfg.get("cube_slice_axis", 0))),
            random_axis=bool(data_cfg.get("volume_random_axis", False)),
            normalize=bool(data_cfg.get("normalize", True)),
            crop_strategy=str(data_cfg.get("crop_strategy", "random")),
            cdd_cache=cdd_cache,
            cdd_use_log=resolve_cdd_cache_use_log(model_cfg, data_cfg),
        )
        val_dataset = None
        train_dataset = dataset
        inference_dataset = JEPA3DCropDataset(
            data_root=data_cfg.get("data_root", "data"),
            npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
            input_files=data_cfg.get("input_files"),
            num_samples=max(1, int(train_cfg.get("inference_num_samples", 8))),
            crop_size=int(data_cfg.get("volume_crop_size", data_cfg.get("crop_size_3d", 64))),
            crop_depth=inference_crop_depth_3d,
            slab_depth=inference_crop_depth_3d,
            depth_axis=int(data_cfg.get("volume_depth_axis", data_cfg.get("cube_slice_axis", 0))),
            random_axis=False,
            normalize=bool(data_cfg.get("normalize", True)),
            crop_strategy="center",
            cdd_cache=cdd_cache,
            cdd_use_log=resolve_cdd_cache_use_log(model_cfg, data_cfg),
        )
        train_idx = []
        val_idx = []
        n_total = len(dataset.npy_files)
        val_fraction = 0.0
    else:
        train_crop_mode = str(data_cfg.get("crop_mode", "none")).lower()
        train_crop_size = data_cfg.get("crop_size")
        val_crop_mode = "center" if train_crop_mode != "none" else "none"
        native_invalid_border_px = 0
        if bool(data_cfg.get("native_invalid_border_rejection", True)):
            native_invalid_base_border_px = int(
                model.invalid_support_border_px() if hasattr(model, "invalid_support_border_px") else model.encoder_receptive_field() // 2
            )
            native_invalid_extra_crop_px = int(additional_crop_from_config(config))
            native_invalid_border_px = native_invalid_base_border_px + native_invalid_extra_crop_px
        else:
            native_invalid_base_border_px = 0
            native_invalid_extra_crop_px = 0
        log_info(
            f"[{config_name}] Native invalid border rejection: "
            f"border_px={native_invalid_border_px} "
            f"(base={native_invalid_base_border_px} additional_crop={native_invalid_extra_crop_px}) "
            "before crop/augmentation"
        )
        dataset = JEPADataset(
            num_samples=data_cfg.get("num_samples", 2000),
            data_root=data_cfg.get("data_root", "data"),
            npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
            input_files=data_cfg.get("input_files"),
            cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
            cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
            cube_slice_index=data_cfg.get("cube_slice_index", 0),
            crop_mode=train_crop_mode,
            crop_size=train_crop_size,
            d4_augment=bool(data_cfg.get("d4_augment", False)),
            input_type=input_type,
            image_batch_selected_indices=image_batch_selected_indices,
            cdd_cache=cdd_cache,
            cdd_use_log=resolve_cdd_cache_use_log(model_cfg, data_cfg),
            return_metadata=True,
            crop_min_valid_fraction=float(data_cfg.get("crop_min_valid_fraction", 0.0)),
            native_invalid_border_px=native_invalid_border_px,
        )
        val_fraction = float(train_cfg.get("val_fraction", 0.1))
        val_fraction = min(max(val_fraction, 0.0), 0.95)
        ordered_total_idx = list(dataset.sample_index)
        total_idx = list(ordered_total_idx)
        split_seed = int(train_cfg.get("split_seed", 42))
        random.Random(split_seed).shuffle(total_idx)
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
                data_root=data_cfg.get("data_root", "data"),
                npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
                input_files=data_cfg.get("input_files"),
                cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
                cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
                cube_slice_index=data_cfg.get("cube_slice_index", 0),
                crop_mode=val_crop_mode,
                crop_size=train_crop_size,
                d4_augment=False,
                input_type=input_type,
                image_batch_selected_indices=image_batch_selected_indices,
                cdd_cache=cdd_cache,
                cdd_use_log=resolve_cdd_cache_use_log(model_cfg, data_cfg),
                crop_min_valid_fraction=float(data_cfg.get("crop_min_valid_fraction", 0.0)),
                native_invalid_border_px=native_invalid_border_px,
            )
            val_dataset.sample_index = val_idx
    if is_3d_mode:
        inference_sample_index = [(path, None) for path in getattr(inference_dataset, "npy_files", [])]
    else:
        inference_sample_index = list(ordered_total_idx)
    log_info(
        f"[{config_name}] Dataset split: total_index={n_total}, train_index={len(train_idx)}, "
        f"val_index={len(val_idx)}, val_fraction={val_fraction:.3f}"
    )
    if (not is_3d_mode) and getattr(dataset, "crop_mode", "none") != "none":
        log_info(
            f"[{config_name}] Training crop: mode={dataset.crop_mode} "
            f"size={dataset.crop_size}; validation=center; inference=native"
        )
    requested_workers = int(train_cfg.get("num_workers", 4))
    # macOS/MPS-safe default: avoid multiprocessing worker hangs unless explicitly set.
    if device.type == "mps" or (sys.platform == "darwin" and device.type != "cuda"):
        num_workers = 0  # macOS spawn/shared-memory transport is fragile outside CUDA workers.
    elif "num_workers" in train_cfg:
        num_workers = requested_workers
    else:
        num_workers = 4 if device.type == "cuda" else 0
    pin_memory = bool(device.type == "cuda")
    persistent_workers = bool(num_workers > 0)
    prefetch_factor = max(1, int(train_cfg.get("prefetch_factor", 2))) if num_workers > 0 else None
    log_info(
        f"[{config_name}] Dataloader setup: num_workers={num_workers}, "
        f"pin_memory={pin_memory}, persistent_workers={persistent_workers}, "
        f"prefetch_factor={prefetch_factor}"
    )
    loader_worker_kwargs = {}
    if num_workers > 0:
        loader_worker_kwargs["prefetch_factor"] = prefetch_factor
        loader_worker_kwargs["worker_init_fn"] = _seed_dataloader_worker

    # DDP: distribute data across GPUs
    train_sampler = DistributedSampler(train_dataset) if is_ddp else None
    val_sampler = None

    target_mask = None
    target_threshold = data_cfg.get("target_threshold")
    if target_threshold is not None:
        target_threshold = float(target_threshold)
    if data_cfg.get("target_mask"):
        mask_path = os.path.join(data_cfg.get("data_root", "data"), data_cfg["target_mask"])
        if os.path.exists(mask_path):
            target_mask = torch.from_numpy(np.load(mask_path).astype(np.float32))
            log_info(f"[{config_name}] target_mask loaded: {mask_path} shape={tuple(target_mask.shape)}")
    if target_mask is None and target_threshold is not None:
        target_mask = _target_mask_from_data_threshold(data_cfg, target_threshold, config_name)

    masking_collate = None if is_3d_mode else _MaskingCollator(
        model,
        return_debug=bool(train_cfg.get("debug_masking_tensors", False)),
        require_precomputed_cdd=uses_cdd_channels,
        target_mask=target_mask,
        target_threshold=target_threshold,
        locality_refinement=locality_refinement,
        locality_refinement_n_target=locality_refinement_n_target,
    )
    if is_main_process and (not is_3d_mode) and str(getattr(model, "encoder_type", "")).lower() in CDD_CUBE_ENCODER_TYPES:
        log_info(
            f"[{config_name}] CDD batch source: "
            "precomputed_cache"
        )
    dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=masking_collate,
        generator=torch.Generator().manual_seed(int(train_cfg.get("split_seed", 42))),
        **loader_worker_kwargs,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 32),
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=masking_collate,
            **loader_worker_kwargs,
        )
    # Inference must use canonical orientation (no D4 augmentation).
    if not is_3d_mode:
        inference_dataset = JEPADataset(
            num_samples=train_dataset.num_samples,
            data_root=data_cfg.get("data_root", "data"),
            npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
            input_files=data_cfg.get("input_files"),
            cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
            cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
            cube_slice_index=data_cfg.get("cube_slice_index", 0),
            crop_mode="none",
            crop_size=None,
            d4_augment=False,
            input_type=input_type,
            image_batch_inference=image_batch_inference,
            image_batch_selected_indices=image_batch_selected_indices,
            cdd_cache=cdd_cache,
            cdd_use_log=resolve_cdd_cache_use_log(model_cfg, data_cfg),
        )
        inference_dataset.sample_index = inference_sample_index[:1] if inference_sample_index else list(train_idx[:1])
        inference_dataset.num_samples = max(1, len(inference_dataset.sample_index))
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=(None if is_3d_mode else _collate_for_inference),
        **loader_worker_kwargs,
    )

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    use_amp = device.type == "cuda"
    autocast_device = "cuda" if use_amp else "cpu"
    scaler = GradScaler("cuda" if use_amp else "cpu", enabled=use_amp)
    if resume_state is not None:
        optimizer_state_loaded = False
        if "optimizer_state_dict" in resume_state:
            try:
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
                optimizer_state_loaded = True
            except ValueError as e:
                # Model parameterization changed (e.g., architecture update): choose explicit behavior.
                log_error("optimizer_state_incompatible", e)
                if optimizer_mismatch_action == "restart_epoch0":
                    start_epoch = 0
                else:
                    log_info(
                        f"[{config_name}] warning: optimizer_state_incompatible, "
                        f"continuing from epoch {start_epoch} with fresh optimizer: {e}"
                    )
        if optimizer_state_loaded and "scaler_state_dict" in resume_state and torch.cuda.is_available():
            try:
                scaler.load_state_dict(resume_state["scaler_state_dict"])
            except Exception as e:
                log_error("scaler_state_incompatible", e)

    log_interval = train_cfg.get("log_interval", 10)
    log_flush_interval = max(1, int(log_interval))
    diagnostic_interval = max(1, int(train_cfg.get("diagnostic_interval", log_flush_interval)))
    inference_mask_passes = int(train_cfg.get("inference_mask_passes", 1))
    dashboard_mask_inference = bool(train_cfg.get("dashboard_mask_inference", True))
    inference_mask_border = model_cfg.get("inference_mask_border", train_cfg.get("inference_mask_border", True))
    umap_cfg = dict(train_cfg.get("umap", {}))
    inference_tta_enabled = bool(train_cfg.get("inference_tta_enabled", False))
    inference_tta_mode = str(train_cfg.get("inference_tta_mode", "flip4"))
    log_info(f"[{config_name}] umap_config={json.dumps(umap_cfg, sort_keys=True)}")
    prediction_loss_weight = float(train_cfg.get("prediction_loss_weight", 100.0))
    normalize_loss_l2_active = bool(model_cfg.get("normalize_loss_l2", model_cfg.get("normalize_loss", False)))
    gradient_accumulation_mode = str(train_cfg.get("gradient_accumulation_mode", "step")).lower()
    if gradient_accumulation_mode not in ("step", "batch"):
        raise ValueError(
            f"Unsupported train.gradient_accumulation_mode={gradient_accumulation_mode!r}. "
            "Use 'step' or 'batch'."
        )
    accum_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    log_info(f"[{config_name}] gradient_accumulation_mode={gradient_accumulation_mode}")
    spread_regularizer = parse_spread_regularizer_config(train_cfg)
    spread_regularizer_weight = float(spread_regularizer["weight"])
    embed_spread_target = float(spread_regularizer["target_std"])
    spread_regularizer_eps = float(spread_regularizer["eps"])
    log_info(f"[{config_name}] spread_regularizer={json.dumps(spread_regularizer, sort_keys=True)}")
    if locality_refinement and spread_regularizer_weight <= 0.0:
        raise ValueError(
            "train.locality_refinement requires train.spread_regularizer.weight > 0 "
            "so every micro JEPA loss includes the anchored hinge term"
        )
    if locality_refinement:
        log_info(
            f"[{config_name}] followup_objective="
            + (
                "prediction_weight*prediction_loss + spread_weight*spread_hinge"
                if vanilla_matched_steps
                else "prediction_weight*prediction_loss + "
                "spread_weight*relu(micro_hinge-macro_initial_hinge)"
            )
        )
    experimental_losses = dict(train_cfg.get("experimental_losses", {}))
    vicreg_var_weight = float(train_cfg.get("vicreg_var_weight", experimental_losses.get("vicreg_var_weight", 0.0)))
    vicreg_cov_weight = float(train_cfg.get("vicreg_cov_weight", experimental_losses.get("vicreg_cov_weight", 0.0)))
    symmetry_loss_weight = float(train_cfg.get("symmetry_loss_weight", 0.0))
    vicreg_spatial_mode = str(train_cfg.get("vicreg_spatial_mode", "dense")).lower()
    if vicreg_spatial_mode not in ("dense", "pooled"):
        raise ValueError(
            f"Unsupported train.vicreg_spatial_mode={vicreg_spatial_mode}. Use 'dense' or 'pooled'."
        )
    ema_base = float(train_cfg.get("ema_momentum_base", model.ema_momentum))
    ema_final = float(train_cfg.get("ema_momentum_final", 1.0))

    base_lr = float(train_cfg.get("lr", 1e-4))
    min_lr = float(train_cfg.get("min_lr", 1e-6))
    warmup_epochs = float(train_cfg.get("warmup_epochs", 1.0))

    # PyTorch native LR scheduler: warmup → cosine decay
    total_steps_sched = max(1, int(epochs) * max(1, len(dataloader)))
    warmup_steps_sched = int(warmup_epochs * max(1, len(dataloader)))
    if total_steps_sched > 1:
        warmup_steps_sched = min(max(1, warmup_steps_sched), total_steps_sched - 1)
    else:
        warmup_steps_sched = 1
    cosine_steps_sched = max(1, total_steps_sched - warmup_steps_sched)
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    warmup_sched = LinearLR(
        optimizer,
        start_factor=min_lr / max(base_lr, 1e-12),
        end_factor=1.0,
        total_iters=warmup_steps_sched,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps_sched,
        eta_min=min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps_sched],
    )

    metrics_path = os.path.join(session_dir, "metrics.csv")
    metrics_header = [
        "epoch",
        "batch",
        "global_step",
        "loss_optimization_total",
        "loss_macro",
        "loss_total",
        "loss_prediction",
        "lr",
        "loss_spread",
        "loss_vicreg_var",
        "loss_vicreg_cov",
        "loss_symmetry",
        "weighted_prediction",
        "weighted_spread",
        "weighted_vicreg_var",
        "weighted_vicreg_cov",
        "weighted_symmetry",
        "ema_momentum",
        "sim",
        "var",
        "cov",
        "raw_mse",
        "norm_err",
        "valid_frac",
        "embed_spread_mean",
        "embed_spread_min",
        "embed_under_spread_frac",
        "dead_channel_count",
        "context_manifold_size",
        "targets_per_image",
        "otf_macro_passes",
        "otf_followup_mean_passes",
        "mask_footprint_mean_px",
        "mask_footprint_min_px",
        "mask_footprint_max_px",
        "mask_scale_factor",
        "locality_micro_mean_loss",
        "locality_loss",
        "locality_prediction",
        "locality_hinge",
        "locality_initial_hinge",
        "locality_micro_hinge",
        "locality_initial_std",
        "locality_micro_std",
        "locality_micro_steps",
    ]
    for micro_step in range(locality_refinement_n_step if locality_refinement else 0):
        step_number = micro_step + 1
        metrics_header.extend(
            [
                f"locality_micro_step_{step_number}_loss",
                f"locality_micro_step_{step_number}_prediction",
                f"locality_micro_step_{step_number}_hinge",
                f"locality_micro_step_{step_number}_hinge_penalty",
                f"locality_micro_step_{step_number}_otf_passes",
            ]
        )
    metrics_header.append("time_sec")
    if is_main_process:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
                existing_header = list(reader.fieldnames or [])
            if existing_header != metrics_header:
                legacy_names = {
                    "loss_optimization_total": "loss_total",
                    "loss_total": "total_loss",
                    "loss_prediction": "loss_mse",
                    "loss_spread": "loss_sigreg",
                    "loss_symmetry": "loss_symmetric",
                    "weighted_prediction": "weighted_mse",
                    "weighted_spread": "weighted_sigreg",
                    "weighted_symmetry": "weighted_symmetric",
                    "embed_spread_mean": "ctx_std_mean",
                    "embed_spread_min": "ctx_std_min",
                    "context_manifold_size": "ctx_rank",
                    "locality_micro_mean_loss": "locality_loss",
                }
                with open(metrics_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=metrics_header)
                    writer.writeheader()
                    for row in existing_rows:
                        writer.writerow({
                            key: row.get(key, row.get(legacy_names.get(key, ""), ""))
                            for key in metrics_header
                        })
        else:
            with open(metrics_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(metrics_header)
    masked_scales_log_path = os.path.join(session_dir, "masked_scales_log.csv")
    if is_main_process and not os.path.exists(masked_scales_log_path):
        with open(masked_scales_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "batch", "scale", "count"])
    epoch_summary_path = os.path.join(session_dir, "epoch_summary.csv")
    if is_main_process and not os.path.exists(epoch_summary_path):
        with open(epoch_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_sim", "val_error_by_scale_json"])
    visited_targets_log_path = os.path.join(session_dir, "visited_target_locations.csv")
    if is_main_process and not os.path.exists(visited_targets_log_path):
        with open(visited_targets_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "batch", "sample_idx", "target_idx", "z", "y", "x", "scale"])
    locality_matches_path = os.path.join(session_dir, "locality_refinement_matches.csv")
    if is_main_process and locality_refinement and not os.path.exists(locality_matches_path):
        with open(locality_matches_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "batch",
                    "micro_step",
                    "sample_idx",
                    "anchor_y",
                    "anchor_x",
                    "target_idx",
                    "target_y",
                    "target_x",
                    "neighbor_rank",
                    "distance_px",
                    "distance_latent",
                    "n_targets",
                    "knn_pool_size",
                ]
            )
    locality_target_locations_path = os.path.join(
        session_dir,
        "locality_refinement_target_locations.csv",
    )
    if is_main_process and locality_refinement and not os.path.exists(locality_target_locations_path):
        with open(locality_target_locations_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "batch",
                    "micro_step",
                    "sample_idx",
                    "target_kind",
                    "target_idx",
                    "target_y",
                    "target_x",
                ]
            )

    loss_weights_path = os.path.join(session_dir, "loss_weights.json")
    if is_main_process and not os.path.exists(loss_weights_path):
        with open(loss_weights_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prediction_loss_weight": prediction_loss_weight,
                    "spread_regularizer": spread_regularizer,
                    "locality_refinement": {
                        "enabled": locality_refinement,
                        "mode": "vanilla" if vanilla_matched_steps else locality_refinement_knn_space,
                        "n_step": locality_refinement_n_step,
                        "n_target": locality_refinement_n_target,
                        "knn_space": locality_refinement_knn_space,
                        "spatial_fov_factor": locality_spatial_fov_factor,
                        "spatial_radius_px": locality_spatial_radius_px,
                        "one_target_per_otf_pass": locality_refinement_n_target is not None,
                        "hinge_reference": "macro_initial_sample_normalized_std_hinge",
                        "hinge_penalty": (
                            "standard_spread_hinge"
                            if vanilla_matched_steps
                            else "relu(micro_hinge-initial_hinge)"
                        ),
                    },
                    "symmetry_loss_weight": symmetry_loss_weight,
                    "experimental_losses": experimental_losses,
                },
                f,
                indent=2,
            )
            f.write("\n")
    if is_ddp:
        dist.barrier()

    model.train()
    start = time.time()
    visit_counts = None
    if start_epoch >= int(epochs):
        log_info(f"[{config_name}] checkpoint epoch {start_epoch} already >= configured epochs {epochs}, skipping training loop")
    for epoch in range(start_epoch, epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        epoch_total = 0.0
        epoch_macro = 0.0
        epoch_locality_micro_mean = 0.0
        epoch_prediction = 0.0
        epoch_sim = 0.0
        epoch_var = 0.0
        epoch_cov = 0.0
        epoch_spread = 0.0
        epoch_symmetric = 0.0
        epoch_valid_frac = 0.0
        epoch_targets_per_image = 0.0
        epoch_target_slots_per_image = 0.0
        epoch_embed_spread_mean = 0.0
        epoch_context_manifold_size = 0.0
        epoch_batches = 0
        metrics_rows = []
        masked_scale_rows = []
        visited_rows = []
        locality_match_rows = []
        locality_target_location_rows = []
        tqdm.write(f"[{config_name}]")
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"E {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=0.1,
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        for batch_idx, batch in pbar:
            if is_3d_mode:
                x_clean = batch.to(device, non_blocking=True)
                x_clean = torch.nan_to_num(x_clean, nan=0.0, posinf=0.0, neginf=0.0)
                context_data = None
            else:
                x_clean, context_result = batch
                x_clean = x_clean.to(device, non_blocking=True)
                context_result = _move_to_device(context_result, device)
                x_context, tloc, tscale, tvalid = context_result[:4]
                debug = context_result[4] if len(context_result) == 5 else {}
                context_data = (x_context, tloc, tscale, tvalid, debug)

            # PyTorch scheduler handles warmup + cosine automatically
            current_step = epoch * max(1, len(dataloader)) + batch_idx

            with autocast(device_type=autocast_device, enabled=use_amp):
                outputs = model(x_clean, context_data=context_data) if not is_3d_mode else model(x_clean)
                macro_otf_passes = int(
                    outputs.get("otf_masking_num_passes", torch.tensor(0)).item()
                )
                if (
                    is_main_process
                    and batch_idx == 0
                    and "otf_masking_num_passes" in outputs
                ):
                    valid_targets = int(outputs["target_valid"].sum().item())
                    otf_passes = int(outputs["otf_masking_num_passes"].item())
                    log_info(
                        f"[{config_name}] epoch={epoch + 1} "
                        f"otf_mask_passes={otf_passes} valid_targets={valid_targets}"
                    )

                zero_loss = outputs["pred_patches"].new_zeros(())
                if abs(vicreg_var_weight) > 1e-12 or abs(vicreg_cov_weight) > 1e-12:
                    _, var_term_t, cov_term_t = compute_sim_var_cov_torch(
                        outputs,
                        spatial_mode=vicreg_spatial_mode,
                    )
                else:
                    var_term_t = zero_loss
                    cov_term_t = zero_loss
                if abs(spread_regularizer_weight) > 1e-12:
                    loss_spread, z_ctx = compute_output_spread_regularizer_loss(
                        outputs,
                        spread_regularizer,
                        include_predictor=False,
                    )
                else:
                    loss_spread = zero_loss
                    z_ctx = outputs["pred_patches"].new_empty((0, int(outputs["pred_patches"].shape[2])))
                if abs(prediction_loss_weight) > 1e-12:
                    loss_prediction = model.compute_loss(outputs)
                else:
                    loss_prediction = zero_loss
                if abs(symmetry_loss_weight) > 1e-12:
                    loss_symmetry = model.compute_symmetric_loss(outputs)
                else:
                    loss_symmetry = zero_loss
                total_loss = (
                    (prediction_loss_weight * loss_prediction)
                    + (vicreg_var_weight * var_term_t)
                    + (vicreg_cov_weight * cov_term_t)
                    + (spread_regularizer_weight * loss_spread)
                    + (symmetry_loss_weight * loss_symmetry)
                )
                locality_initial_std = None
                locality_initial_hinge = None
                if locality_refinement:
                    locality_initial_embeddings = _locality_context_embeddings(
                        outputs,
                        spatial_mode=str(spread_regularizer.get("spatial_mode", "pooled")),
                    )
                    locality_initial_std = embedding_channel_std(
                        locality_initial_embeddings,
                        eps=spread_regularizer_eps,
                    ).detach()
                    locality_initial_hinge = embedding_std_hinge_loss(
                        locality_initial_embeddings,
                        target_std=embed_spread_target,
                        eps=spread_regularizer_eps,
                    ).detach()
            # DDP: sync component losses for accurate logging across all ranks
            if is_ddp:
                components = torch.stack([
                    total_loss.detach(),
                    loss_prediction.detach(),
                    loss_spread.detach(),
                    loss_symmetry.detach(),
                    var_term_t.detach(),
                    cov_term_t.detach(),
                ])
                dist.all_reduce(components, op=dist.ReduceOp.SUM)
                w = float(dist.get_world_size())
                log_loss_val = float((components[0] / w).item())
                loss_prediction = components[1] / w
                loss_spread = components[2] / w
                loss_symmetry = components[3] / w
                var_term_t = components[4] / w
                cov_term_t = components[5] / w
            else:
                log_loss_val = float(total_loss.item())
            macro_loss_value = log_loss_val

            loss_for_backward = total_loss / accum_steps
            scaler.scale(loss_for_backward).backward()

            locality_total_sum = 0.0
            locality_prediction_sum = 0.0
            locality_hinge_penalty_sum = 0.0
            locality_micro_hinge_sum = 0.0
            locality_micro_std_sum = 0.0
            locality_micro_count = 0
            locality_step_loss_values = [float("nan")] * locality_refinement_n_step
            locality_step_prediction_values = [float("nan")] * locality_refinement_n_step
            locality_step_hinge_values = [float("nan")] * locality_refinement_n_step
            locality_step_hinge_penalty_values = [float("nan")] * locality_refinement_n_step
            locality_step_otf_pass_values = [float("nan")] * locality_refinement_n_step
            if locality_refinement:
                assert (
                    context_data is not None
                    and locality_initial_std is not None
                    and locality_initial_hinge is not None
                )
                macro_location_rows = []
                if is_main_process:
                    macro_locations_cpu = context_data[1].detach().cpu()
                    macro_valid_cpu = context_data[3].detach().cpu().bool()
                    for sample_index in range(int(macro_locations_cpu.shape[0])):
                        for target_index in range(int(macro_locations_cpu.shape[1])):
                            if not bool(macro_valid_cpu[sample_index, target_index]):
                                continue
                            macro_location_rows.append(
                                (
                                    int(sample_index),
                                    "macro",
                                    int(target_index),
                                    int(macro_locations_cpu[sample_index, target_index, -2]),
                                    int(macro_locations_cpu[sample_index, target_index, -1]),
                                )
                            )
                for micro_step in range(locality_refinement_n_step):
                    micro_context_data, match_rows = _sample_locality_refinement_context(
                        context_data,
                        allow_partial_overlap=float(getattr(model_without_ddp, "target_allow_partial_overlap", 0.0)),
                        fixed_n_targets=locality_refinement_n_target,
                        knn_space=locality_refinement_knn_space,
                        candidate_latent_map=outputs["gt_map"].detach(),
                        spatial_radius_px=locality_spatial_radius_px,
                        one_target_per_pass=locality_refinement_n_target is not None,
                    )
                    micro_available = micro_context_data is not None
                    if is_ddp:
                        # Every DDP rank must execute the same number of
                        # forwards/backwards. If any rank lacks a valid local
                        # neighborhood, all ranks skip this microstep.
                        availability = torch.tensor(
                            int(micro_available),
                            device=x_clean.device,
                            dtype=torch.int32,
                        )
                        dist.all_reduce(availability, op=dist.ReduceOp.MIN)
                        micro_available = bool(availability.item())
                    if not micro_available:
                        continue
                    assert micro_context_data is not None
                    with autocast(device_type=autocast_device, enabled=use_amp):
                        micro_outputs = model(x_clean, context_data=micro_context_data)
                        micro_prediction = model.compute_loss(micro_outputs)
                        micro_embeddings = _locality_context_embeddings(
                            micro_outputs,
                            spatial_mode=str(spread_regularizer.get("spatial_mode", "pooled")),
                        )
                        micro_raw_hinge = embedding_std_hinge_loss(
                            micro_embeddings,
                            target_std=embed_spread_target,
                            eps=spread_regularizer_eps,
                        )
                        micro_hinge_penalty = anchored_spread_hinge_loss(
                            micro_embeddings,
                            locality_initial_hinge,
                            target_std=embed_spread_target,
                            eps=spread_regularizer_eps,
                        )
                        followup_hinge_loss = (
                            micro_raw_hinge if vanilla_matched_steps else micro_hinge_penalty
                        )
                        micro_loss = (
                            (prediction_loss_weight * micro_prediction)
                            + (spread_regularizer_weight * followup_hinge_loss)
                        )
                    scaler.scale(
                        micro_loss / float(accum_steps * locality_refinement_n_step)
                    ).backward()
                    locality_total_sum += float(micro_loss.detach().item())
                    locality_prediction_sum += float(micro_prediction.detach().item())
                    locality_hinge_penalty_sum += float(followup_hinge_loss.detach().item())
                    locality_micro_hinge_sum += float(micro_raw_hinge.detach().item())
                    locality_micro_std_sum += float(
                        embedding_channel_std(
                            micro_embeddings.detach(),
                            eps=spread_regularizer_eps,
                        ).mean().item()
                    )
                    locality_micro_count += 1
                    locality_step_loss_values[micro_step] = float(micro_loss.detach().item())
                    locality_step_prediction_values[micro_step] = float(micro_prediction.detach().item())
                    locality_step_hinge_values[micro_step] = float(micro_raw_hinge.detach().item())
                    locality_step_hinge_penalty_values[micro_step] = float(
                        followup_hinge_loss.detach().item()
                    )
                    locality_step_otf_pass_values[micro_step] = float(
                        micro_outputs.get("otf_masking_num_passes", torch.tensor(0)).item()
                    )
                    if is_main_process:
                        locality_match_rows.extend(
                            (epoch + 1, batch_idx, micro_step + 1, *row)
                            for row in match_rows
                        )
                        locality_target_location_rows.extend(
                            (epoch + 1, batch_idx, micro_step + 1, *row)
                            for row in macro_location_rows
                        )
                        locality_target_location_rows.extend(
                            (
                                epoch + 1,
                                batch_idx,
                                micro_step + 1,
                                int(row[0]),
                                "micro",
                                int(row[3]),
                                int(row[4]),
                                int(row[5]),
                            )
                            for row in match_rows
                        )

            locality_loss_value = 0.0
            locality_prediction_value = 0.0
            locality_hinge_penalty_value = 0.0
            locality_micro_hinge_value = 0.0
            locality_micro_std_value = 0.0
            locality_initial_hinge_value = (
                float(locality_initial_hinge.item())
                if locality_initial_hinge is not None and locality_initial_hinge.numel() == 1
                else 0.0
            )
            locality_initial_std_value = (
                float(locality_initial_std.mean().item())
                if locality_initial_std is not None and locality_initial_std.numel() > 0
                else 0.0
            )
            locality_followup_mean_otf_passes = (
                float(np.nanmean(locality_step_otf_pass_values))
                if any(math.isfinite(v) for v in locality_step_otf_pass_values)
                else 0.0
            )
            if locality_refinement and is_ddp:
                # Keep every plotted micro-step curve rank-averaged, just like
                # the aggregate macro and locality metrics.
                step_stats = torch.zeros(
                    (locality_refinement_n_step, 5),
                    device=x_clean.device,
                    dtype=torch.float64,
                )
                for step_index in range(locality_refinement_n_step):
                    if math.isfinite(locality_step_loss_values[step_index]):
                        step_stats[step_index] = torch.tensor(
                            [
                                locality_step_loss_values[step_index],
                                locality_step_prediction_values[step_index],
                                locality_step_hinge_values[step_index],
                                locality_step_hinge_penalty_values[step_index],
                                1.0,
                            ],
                            device=x_clean.device,
                            dtype=torch.float64,
                        )
                dist.all_reduce(step_stats, op=dist.ReduceOp.SUM)
                for step_index in range(locality_refinement_n_step):
                    step_count = float(step_stats[step_index, 4].item())
                    if step_count <= 0.0:
                        continue
                    locality_step_loss_values[step_index] = float(step_stats[step_index, 0].item() / step_count)
                    locality_step_prediction_values[step_index] = float(step_stats[step_index, 1].item() / step_count)
                    locality_step_hinge_values[step_index] = float(step_stats[step_index, 2].item() / step_count)
                    locality_step_hinge_penalty_values[step_index] = float(
                        step_stats[step_index, 3].item() / step_count
                    )
            if locality_micro_count > 0:
                locality_stats = torch.tensor(
                    [
                        locality_total_sum,
                        locality_prediction_sum,
                        locality_hinge_penalty_sum,
                        locality_micro_hinge_sum,
                        locality_micro_std_sum,
                        locality_initial_hinge_value * locality_micro_count,
                        locality_initial_std_value * locality_micro_count,
                        float(locality_micro_count),
                    ],
                    device=x_clean.device,
                    dtype=torch.float64,
                )
                if is_ddp:
                    dist.all_reduce(locality_stats, op=dist.ReduceOp.SUM)
                locality_count_global = max(1.0, float(locality_stats[7].item()))
                locality_loss_value = float(locality_stats[0].item() / locality_count_global)
                locality_prediction_value = float(locality_stats[1].item() / locality_count_global)
                locality_hinge_penalty_value = float(locality_stats[2].item() / locality_count_global)
                locality_micro_hinge_value = float(locality_stats[3].item() / locality_count_global)
                locality_micro_std_value = float(locality_stats[4].item() / locality_count_global)
                locality_initial_hinge_value = float(locality_stats[5].item() / locality_count_global)
                locality_initial_std_value = float(locality_stats[6].item() / locality_count_global)
                log_loss_val += locality_loss_value

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler_scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if (not use_amp) or scaler.get_scale() >= scaler_scale_before_step:
                    scheduler.step()
                model_without_ddp.update_target_encoder()
            current_lr = scheduler.get_last_lr()[0]

            total_steps_sched = max(1, int(epochs) * max(1, len(dataloader)))
            progress = min(1.0, max(0.0, float(current_step) / float(total_steps_sched)))

            # Cosine EMA schedule: anneal from ema_base → ema_final over
            # ema_warmup_fraction of training, then hold at ema_final.
            ema_warmup_frac = float(train_cfg.get("ema_warmup_fraction", 1.0))
            ema_warmup_frac = max(0.0, min(1.0, ema_warmup_frac))
            if ema_warmup_frac <= 0.0:
                ema_progress = 0.0
            else:
                ema_progress = min(1.0, progress / ema_warmup_frac)
            new_momentum = float(
                ema_final - 0.5 * (ema_final - ema_base) * (1.0 + math.cos(math.pi * ema_progress))
            )
            model_without_ddp.ema_momentum = new_momentum
            sim_val, var_val, cov_val = compute_sim_var_cov(
                outputs,
                spatial_mode=vicreg_spatial_mode,
            )
            raw_mse_val, norm_err_val = compute_raw_mse_and_norm_err(outputs)
            energy_val = compute_jepa_energy(outputs)
            target_valid_f = outputs["target_valid"].float()
            valid_frac = float(target_valid_f.mean().item())
            ctx_stats = embedding_spread_stats(z_ctx, target_std=embed_spread_target)
            active_targets_by_sample = target_valid_f.sum(dim=1)
            targets_per_image = float(active_targets_by_sample.mean().item())
            target_slots_per_image = int(outputs["target_valid"].shape[1]) if outputs["target_valid"].dim() >= 2 else 0
            target_z_counts = _format_target_z_counts(
                outputs["target_locations"],
                outputs["target_valid"],
                depth=int(outputs["pred_map"].shape[-3]) if outputs["pred_map"].dim() == 5 else None,
            )
            footprint_values = outputs.get("target_box_sizes")
            if footprint_values is not None and footprint_values.numel() > 0:
                footprint_valid = outputs.get("target_valid")
                if footprint_valid is not None and footprint_valid.shape == footprint_values.shape:
                    footprint_values = footprint_values[footprint_valid]
                footprint_values = footprint_values[footprint_values > 0]
            if footprint_values is None or footprint_values.numel() == 0:
                footprint_values = outputs.get("cdd_box_sizes")
            if footprint_values is None or footprint_values.numel() == 0:
                footprint_values = torch.as_tensor(
                    [outputs.get("mask_footprint_px", model.mask_box_size)],
                    device=x_clean.device,
                    dtype=x_clean.dtype,
                )
            mask_footprint_mean_px = float(footprint_values.float().mean().item())
            mask_footprint_min_px = float(footprint_values.float().min().item())
            mask_footprint_max_px = float(footprint_values.float().max().item())
            mask_scale_factor = float(outputs.get("mask_scale_factor", getattr(model, "mask_scale", 1.0)))

            elapsed = time.time() - start
            global_step = epoch * max(1, len(dataloader)) + batch_idx
            metric_row = [
                    epoch + 1,
                    batch_idx,
                    global_step,
                    log_loss_val,
                    macro_loss_value,
                    log_loss_val,
                    float(loss_prediction.item()),
                    float(current_lr),
                    float(loss_spread.item()),
                    float(var_term_t.item()),
                    float(cov_term_t.item()),
                    float(loss_symmetry.item()),
                    float((prediction_loss_weight * loss_prediction).item()),
                    float((spread_regularizer_weight * loss_spread).item()),
                    float((vicreg_var_weight * var_term_t).item()),
                    float((vicreg_cov_weight * cov_term_t).item()),
                    float((symmetry_loss_weight * loss_symmetry).item()),
                    float(new_momentum),
                    float(sim_val),
                    float(var_val),
                    float(cov_val),
                    float(raw_mse_val),
                    float(norm_err_val),
                    float(valid_frac),
                    ctx_stats["embed_spread_mean"],
                    ctx_stats["embed_spread_min"],
                    ctx_stats["embed_under_spread_frac"],
                    ctx_stats["dead_channel_count"],
                    ctx_stats["context_manifold_size"],
                    targets_per_image,
                    macro_otf_passes,
                    locality_followup_mean_otf_passes,
                    mask_footprint_mean_px,
                    mask_footprint_min_px,
                    mask_footprint_max_px,
                    mask_scale_factor,
                    locality_loss_value,
                    locality_loss_value,
                    locality_prediction_value,
                    locality_hinge_penalty_value,
                    locality_initial_hinge_value,
                    locality_micro_hinge_value,
                    locality_initial_std_value,
                    locality_micro_std_value,
                    locality_micro_count,
                ]
            for step_index in range(locality_refinement_n_step if locality_refinement else 0):
                metric_row.extend(
                    [
                        locality_step_loss_values[step_index],
                        locality_step_prediction_values[step_index],
                        locality_step_hinge_values[step_index],
                        locality_step_hinge_penalty_values[step_index],
                        locality_step_otf_pass_values[step_index],
                    ]
                )
            metric_row.append(round(elapsed, 4))
            metrics_rows.append(metric_row)
            should_log_diagnostics = (
                ((batch_idx + 1) % diagnostic_interval == 0)
                or ((batch_idx + 1) == len(dataloader))
            )
            if should_log_diagnostics:
                # Keep the hot path asynchronous: copy diagnostic tensors to host
                # only at the configured sampling interval.
                scales = outputs["target_scales"].detach().cpu().numpy()
                tvalid = outputs["target_valid"].detach().cpu().numpy().astype(bool)
                valid_scales = scales[tvalid]
                if valid_scales.size > 0:
                    uniq, cnt = np.unique(np.round(valid_scales.astype(np.float32), 6), return_counts=True)
                    for s, c in zip(uniq.tolist(), cnt.tolist()):
                        masked_scale_rows.append([epoch + 1, batch_idx, float(s), int(c)])

                tloc = outputs["target_locations"].detach().cpu().numpy()
                if (not is_3d_mode) and visit_counts is None:
                    aug_meta = debug.get("augment_metadata") if isinstance(debug, dict) else None
                    if isinstance(aug_meta, list) and aug_meta:
                        hh = int(aug_meta[0].get("full_h", outputs["x_clean"].shape[-2]))
                        ww = int(aug_meta[0].get("full_w", outputs["x_clean"].shape[-1]))
                    else:
                        hh, ww = int(outputs["x_clean"].shape[-2]), int(outputs["x_clean"].shape[-1])
                    visit_counts = np.zeros((hh, ww), dtype=np.float32)
                ndim_loc = int(tloc.shape[-1])
                loc_y, loc_x = _target_location_yx(tloc)
                aug_meta = debug.get("augment_metadata") if isinstance(debug, dict) else None
                for bi in range(tloc.shape[0]):
                    meta_bi = aug_meta[bi] if isinstance(aug_meta, list) and bi < len(aug_meta) else None
                    for ki in range(tloc.shape[1]):
                        if not bool(tvalid[bi, ki]):
                            continue
                        yy = int(loc_y[bi, ki])
                        xx = int(loc_x[bi, ki])
                        yy_native, xx_native = _inverse_augmented_yx_to_native(yy, xx, meta_bi)
                        zz = int(tloc[bi, ki, 0]) if ndim_loc >= 3 else 0
                        if (
                            (visit_counts is not None)
                            and ndim_loc == 2
                            and 0 <= yy_native < visit_counts.shape[0]
                            and 0 <= xx_native < visit_counts.shape[1]
                        ):
                            visit_counts[yy_native, xx_native] += 1.0
                        visited_rows.append(
                            [
                                epoch + 1,
                                batch_idx,
                                bi,
                                ki,
                                zz,
                                yy_native,
                                xx_native,
                                float(scales[bi, ki]),
                            ]
                        )
            if is_main_process and (batch_idx + 1) % log_flush_interval == 0:
                _flush_csv_rows(masked_scales_log_path, masked_scale_rows)
                _flush_csv_rows(visited_targets_log_path, visited_rows)
                if locality_refinement:
                    _flush_csv_rows(locality_matches_path, locality_match_rows)
                    _flush_csv_rows(locality_target_locations_path, locality_target_location_rows)
            loss_terms = _format_active_loss_terms(
                total=macro_loss_value if locality_refinement else log_loss_val,
                total_label="macro" if locality_refinement else "total",
                prediction=float(loss_prediction.item()),
                prediction_weight=prediction_loss_weight,
                spread=float(loss_spread.item()),
                spread_weight=spread_regularizer_weight,
                symmetry=float(loss_symmetry.item()),
                symmetry_weight=symmetry_loss_weight,
                vicreg_var=float(var_term_t.item()),
                vicreg_var_weight=vicreg_var_weight,
                vicreg_cov=float(cov_term_t.item()),
                vicreg_cov_weight=vicreg_cov_weight,
            )
            if locality_refinement:
                loss_terms = {"overall": _fmt_metric(log_loss_val), **loss_terms}
            batch_diag = {
                "ctx_std": f"{ctx_stats['embed_spread_mean']:.3f}",
                "ctx_effrank": f"{ctx_stats['context_manifold_size']:.2f}",
                "active_tgt": f"{targets_per_image:.1f}/{target_slots_per_image}",
                "valid": f"{valid_frac:.3f}",
            }
            if target_z_counts:
                batch_diag["z_tgt"] = target_z_counts
            if locality_refinement:
                batch_diag["locality"] = (
                    f"{locality_micro_count}x "
                    f"mean={locality_loss_value:.3g} "
                    f"steps=[{','.join(_fmt_metric(v) for v in locality_step_loss_values if math.isfinite(v))}] "
                    f"pred={locality_prediction_value:.3g} "
                    f"hinge={locality_micro_hinge_value:.3g}/{locality_initial_hinge_value:.3g} "
                    f"hinge_penalty={locality_hinge_penalty_value:.3g} "
                    f"std={locality_micro_std_value:.3g}/{locality_initial_std_value:.3g} "
                    f"otf={macro_otf_passes}+[{','.join(_fmt_metric(v) for v in locality_step_otf_pass_values)}]"
                )
            batch_optim = {
                "lr": f"{current_lr:.1e}",
            }
            if is_main_process and (
                (batch_idx + 1) % log_flush_interval == 0 or (batch_idx + 1) == len(dataloader)
            ):
                tqdm.write(
                    f"[{config_name}] E {epoch + 1}/{epochs} B {batch_idx + 1}/{len(dataloader)} "
                    f"{_format_progress_line('[batch]', loss_terms, batch_diag, batch_optim)}"
                )
            if _use_wandb and (batch_idx + 1) % log_flush_interval == 0:
                import wandb

                wandb_metrics = {
                        "train/loss_optimization_total": log_loss_val,
                        "train/loss_macro": macro_loss_value,
                        "train/loss_total": log_loss_val,
                        "train/loss_prediction": loss_prediction.item(),
                        "train/loss_spread": loss_spread.item(),
                        "train/loss_symmetry": loss_symmetry.item(),
                        "train/lr": current_lr,
                        "train/ema_momentum": new_momentum,
                        "train/sim": sim_val,
                        "metrics/valid_fraction": valid_frac,
                        "metrics/manifold_size": ctx_stats["context_manifold_size"],
                        "metrics/embed_spread": ctx_stats["embed_spread_mean"],
                        "metrics/dead_channels": ctx_stats["dead_channel_count"],
                        "metrics/targets_per_image": targets_per_image,
                        "metrics/target_slots_per_image": target_slots_per_image,
                        "metrics/mask_footprint_mean_px": mask_footprint_mean_px,
                        "train/locality_micro_mean_loss": locality_loss_value,
                        "train/locality_loss": locality_loss_value,
                        "train/locality_prediction": locality_prediction_value,
                        "train/locality_hinge": locality_hinge_penalty_value,
                        "metrics/locality_initial_hinge": locality_initial_hinge_value,
                        "metrics/locality_micro_hinge": locality_micro_hinge_value,
                        "metrics/locality_initial_std": locality_initial_std_value,
                        "metrics/locality_micro_std": locality_micro_std_value,
                        "epoch": epoch + 1,
                    }
                for step_index, step_loss in enumerate(locality_step_loss_values, start=1):
                    if math.isfinite(step_loss):
                        wandb_metrics[f"train/locality_micro_step_{step_index}_loss"] = step_loss
                wandb.log(
                    wandb_metrics,
                    step=global_step,
                )
            epoch_total += log_loss_val
            epoch_macro += macro_loss_value
            epoch_locality_micro_mean += locality_loss_value
            epoch_prediction += float(loss_prediction.item())
            epoch_sim += float(sim_val)
            epoch_var += float(var_val)
            epoch_cov += float(cov_val)
            epoch_spread += float(loss_spread.item())
            epoch_symmetric += float(loss_symmetry.item())
            epoch_valid_frac += float(valid_frac)
            epoch_targets_per_image += float(targets_per_image)
            epoch_target_slots_per_image += float(target_slots_per_image)
            epoch_embed_spread_mean += ctx_stats["embed_spread_mean"]
            epoch_context_manifold_size += ctx_stats["context_manifold_size"]
            epoch_batches += 1

        # Only the main process writes files in DDP mode
        if is_main_process:
            if metrics_rows:
                with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(metrics_rows)
            _flush_csv_rows(masked_scales_log_path, masked_scale_rows)
            _flush_csv_rows(visited_targets_log_path, visited_rows)
            if locality_refinement:
                _flush_csv_rows(locality_matches_path, locality_match_rows)
                _flush_csv_rows(locality_target_locations_path, locality_target_location_rows)
            if visit_counts is not None:
                np.save(os.path.join(session_dir, "visited_target_frequency.npy"), visit_counts.astype(np.float32))

        if epoch_batches > 0:
            avg_total = epoch_total / epoch_batches
            avg_prediction = epoch_prediction / epoch_batches
            epoch_terms = _format_active_loss_terms(
                total=(epoch_macro / epoch_batches) if locality_refinement else avg_total,
                total_label="macro" if locality_refinement else "total",
                prediction=avg_prediction,
                prediction_weight=prediction_loss_weight,
                spread=epoch_spread / epoch_batches,
                spread_weight=spread_regularizer_weight,
                symmetry=epoch_symmetric / epoch_batches,
                symmetry_weight=symmetry_loss_weight,
                vicreg_var=epoch_var / epoch_batches,
                vicreg_var_weight=vicreg_var_weight,
                vicreg_cov=epoch_cov / epoch_batches,
                vicreg_cov_weight=vicreg_cov_weight,
            )
            if locality_refinement:
                epoch_terms = {"overall": _fmt_metric(avg_total), **epoch_terms}
            epoch_diag = {
                "ctx_std": _fmt_metric(epoch_embed_spread_mean / epoch_batches),
                "ctx_effrank": _fmt_metric(epoch_context_manifold_size / epoch_batches),
                "active_tgt": f"{epoch_targets_per_image / epoch_batches:.1f}/{epoch_target_slots_per_image / epoch_batches:.1f}",
                "valid": _fmt_metric(epoch_valid_frac / epoch_batches),
            }
            if locality_refinement:
                epoch_diag["locality_micro_mean"] = _fmt_metric(
                    epoch_locality_micro_mean / epoch_batches
                )
            tqdm.write(
                f"[{config_name}] E {epoch + 1}/{epochs} "
                f"{_format_progress_line('[epoch]', epoch_terms, epoch_diag)}"
            )
        val_loss = 0.0
        val_sim = 0.0
        val_error_by_scale = {}
        if is_main_process and val_loader is not None:
            v = evaluate_validation(
                model=model_without_ddp,
                val_loader=val_loader,
                device=device,
                max_batches=train_cfg.get("val_max_batches"),
                vicreg_spatial_mode=vicreg_spatial_mode,
            )
            val_loss = float(v["val_loss"])
            val_sim = float(v["val_sim"])
            val_error_by_scale = dict(v["val_error_by_scale"])
            tqdm.write(
                f"[{config_name}] Epoch {epoch + 1}/{epochs} val "
                f"loss={_fmt_metric(val_loss)} sim={_fmt_metric(val_sim)} "
                f"err_by_scale={json.dumps(val_error_by_scale, sort_keys=True)}"
            )
        if is_main_process:
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
        # Save resumable checkpoint at the end of every epoch (main process only).
        # Atomic write: tmp → rename prevents corruption on crash/OOM.
        if is_main_process:
            tmp_ckpt = resume_ckpt_path + ".tmp"
            torch.save(
                {
                    "epoch": int(epoch + 1),
                    "model_state_dict": model_without_ddp.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config_name": config_name,
                },
                tmp_ckpt,
            )
            os.replace(tmp_ckpt, resume_ckpt_path)
            tmp_model = model_ckpt_path + ".tmp"
            torch.save(model_without_ddp.state_dict(), tmp_model)
            os.replace(tmp_model, model_ckpt_path)
            tqdm.write(f"[{config_name}] ckpt_saved epoch={epoch + 1}")

    if is_main_process:
        tmp_final = os.path.join(session_dir, "model_last.pt.tmp")
        torch.save(model_without_ddp.state_dict(), tmp_final)
        os.replace(tmp_final, os.path.join(session_dir, "model_last.pt"))

    if is_main_process:
        if force_recompute_inference:
            _clear_stale_dashboard_artifacts(session_dir)
            log_info(f"[{config_name}] stale dashboard/embedding artifacts cleared before re-inference")
        if is_3d_mode:
            session_dir = run_post_training_inference_3d(
                model=model_without_ddp,
                dataloader=inference_loader,
                session_dir=session_dir,
                config_name=config_name,
                force_recompute_inference=force_recompute_inference,
                cdd_cache=cdd_cache,
                inference_depth=inference_crop_depth_3d,
                slice_index=train_cfg.get("inference_slice_index_3d"),
                spatial_tile_size=train_cfg.get("inference_spatial_tile_size_3d", data_cfg.get("volume_crop_size", data_cfg.get("crop_size_3d", 64))),
                spatial_overlap=train_cfg.get("inference_spatial_overlap_3d"),
                inference_tta_enabled=inference_tta_enabled,
                inference_tta_mode=inference_tta_mode,
                inference_mask_border=inference_mask_border,
            )
        else:
            session_dir = run_post_training_inference(
                model=model_without_ddp,
                dataloader=inference_loader,
                session_dir=session_dir,
                config_name=config_name,
                visit_counts=visit_counts,
                force_recompute_inference=force_recompute_inference,
                inference_mask_passes=inference_mask_passes,
                mask_inference=dashboard_mask_inference,
                target_mask=target_mask,
                inference_mask_border=inference_mask_border,
                compute_jepa_energy_fn=compute_jepa_energy,
                compute_target_energy_map_fn=compute_target_energy_map,

                inference_tta_enabled=inference_tta_enabled,
                inference_tta_mode=inference_tta_mode,
                max_diagnostic_size=train_cfg.get("inference_max_diagnostic_size"),
                tile_size=train_cfg.get("inference_tile_size", train_cfg.get("full_volume_spatial_tile_size", 512)),
                tile_overlap=train_cfg.get("inference_tile_overlap"),
                mask_predict_mode=train_cfg.get("mask_predict_mode", train_cfg.get("inference_mask_predict_mode")),
                mask_predict_stride=train_cfg.get("mask_predict_stride", train_cfg.get("inference_mask_predict_stride")),
                mask_predict_box_size=train_cfg.get("mask_predict_box_size", train_cfg.get("inference_mask_predict_box_size")),
                mask_predict_chunk_size=train_cfg.get("mask_predict_chunk_size", train_cfg.get("inference_mask_predict_chunk_size")),
                inference_discard_margin=train_cfg.get("inference_discard_margin"),
                additional_crop=additional_crop_from_config(config),
            )
            run_all_input_inference = bool(train_cfg.get("inference_all_inputs", bool(data_cfg.get("input_files"))))
            if run_all_input_inference and len(inference_sample_index) > 1:
                manifest = []
                for ordinal, sample_key in enumerate(inference_sample_index):
                    if ordinal == 0:
                        manifest.append({
                            "index": ordinal,
                            "sample_key": list(sample_key),
                            "output_dir": session_dir,
                            "dashboard_source": True,
                        })
                        continue
                    input_session_dir = _input_inference_dir(session_dir, ordinal, sample_key)
                    os.makedirs(input_session_dir, exist_ok=True)
                    if force_recompute_inference:
                        _clear_stale_dashboard_artifacts(input_session_dir)
                    input_dataset = JEPADataset(
                        num_samples=1,
                        data_root=data_cfg.get("data_root", "data"),
                        npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
                        input_files=data_cfg.get("input_files"),
                        cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
                        cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
                        cube_slice_index=data_cfg.get("cube_slice_index", 0),
                        crop_mode="none",
                        crop_size=None,
                        d4_augment=False,
                        input_type=input_type,
                        image_batch_inference=image_batch_inference,
                        image_batch_selected_indices=image_batch_selected_indices,
                        cdd_cache=cdd_cache,
                        cdd_use_log=resolve_cdd_cache_use_log(model_cfg, data_cfg),
                    )
                    input_dataset.sample_index = [sample_key]
                    input_loader = DataLoader(
                        input_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=pin_memory,
                        persistent_workers=False,
                        collate_fn=_collate_for_inference,
                        **loader_worker_kwargs,
                    )
                    run_post_training_inference(
                        model=model_without_ddp,
                        dataloader=input_loader,
                        session_dir=input_session_dir,
                        config_name=f"{config_name}/input{ordinal:03d}",
                        visit_counts=None,
                        force_recompute_inference=force_recompute_inference,
                        inference_mask_passes=inference_mask_passes,
                        mask_inference=dashboard_mask_inference,
                        target_mask=target_mask,
                        inference_mask_border=inference_mask_border,
                        compute_jepa_energy_fn=compute_jepa_energy,
                        compute_target_energy_map_fn=compute_target_energy_map,
                        inference_tta_enabled=inference_tta_enabled,
                        inference_tta_mode=inference_tta_mode,
                        max_diagnostic_size=train_cfg.get("inference_max_diagnostic_size"),
                        tile_size=train_cfg.get("inference_tile_size", train_cfg.get("full_volume_spatial_tile_size", 512)),
                        tile_overlap=train_cfg.get("inference_tile_overlap"),
                        mask_predict_mode=train_cfg.get("mask_predict_mode", train_cfg.get("inference_mask_predict_mode")),
                        mask_predict_stride=train_cfg.get("mask_predict_stride", train_cfg.get("inference_mask_predict_stride")),
                        mask_predict_box_size=train_cfg.get("mask_predict_box_size", train_cfg.get("inference_mask_predict_box_size")),
                        mask_predict_chunk_size=train_cfg.get("mask_predict_chunk_size", train_cfg.get("inference_mask_predict_chunk_size")),
                        inference_discard_margin=train_cfg.get("inference_discard_margin"),
                        additional_crop=additional_crop_from_config(config),
                    )
                    manifest.append({
                        "index": ordinal,
                        "sample_key": list(sample_key),
                        "output_dir": input_session_dir,
                        "dashboard_source": False,
                    })
                with open(os.path.join(session_dir, "inference_inputs_manifest.json"), "w", encoding="utf-8") as f:
                    json.dump({"inputs": manifest}, f, indent=2)
                    f.write("\n")
    # Save NPY artifacts (PCA/UMAP/latent embeddings) required by session_to_dash.py.
    # No PNG, HTML, or dashboard rendering is performed here.
    inf_path = os.path.join(session_dir, "inference_outputs.pt")
    post_training_artifacts = bool(train_cfg.get("post_training_artifacts", True))
    artifacts_exist = os.path.exists(os.path.join(session_dir, "rank_diagnostics.json")) and any(
        os.path.getsize(p) > 0 for p in glob.glob(os.path.join(session_dir, "results", "*_pca_*.npy"))
    ) if os.path.isdir(os.path.join(session_dir, "results")) else False
    if artifacts_exist and not force_recompute_inference:
        log_info(f"[{config_name}] post-training artifacts already exist; skip PCA/UMAP/rank generation")
    if (
        is_main_process
        and (not is_3d_mode)
        and os.path.exists(inf_path)
        and post_training_artifacts
        and (force_recompute_inference or not artifacts_exist)
    ):
        try:
            outputs = torch.load(inf_path, map_location="cpu", weights_only=False)
            inference_pca = bool(train_cfg.get("inference_pca", True))
            inference_umap = bool(train_cfg.get("inference_umap", True))
            artifacts_dir = export_inference_dashboard_artifacts(session_dir, outputs, umap_cfg=umap_cfg, inference_pca=inference_pca, inference_umap=inference_umap)
            log_info(f"[{config_name}] artifacts_saved={artifacts_dir}")
            effective_rank = ""
            rank_diag = {}
            try:
                rank_diag = rank_dashboard(outputs)
                try:
                    rank_diag["energy"] = float(compute_jepa_energy(outputs))
                except Exception:
                    pass
                with open(os.path.join(session_dir, "rank_diagnostics.json"), "w", encoding="utf-8") as f:
                    json.dump(rank_diag, f, indent=2)
            except Exception as er:
                log_error("rank_diagnostics", er)
            if compute_effective_rank:
                try:
                    # Use target-branch rank as the primary effective-rank signal.
                    # pred.erank can be confounded by predictor weakness/noise.
                    if "gt" in rank_diag and "erank" in rank_diag["gt"]:
                        effective_rank = f"{float(rank_diag['gt']['erank']):.8f}"
                    else:
                        # Fallback: compute effective rank on valid target patches
                        # (not the full dense map — untrained/unpenalized pixels
                        # would dominate the covariance).  Matches VICReg path.
                        pred_patches = outputs.get("pred_patches")
                        if pred_patches is not None:
                            pp = torch.as_tensor(pred_patches)
                            # pp: B x K x C x Ph x Pw → pool spatial dims → B*K x C
                            pp_pooled = pp.mean(dim=(-2, -1))  # B x K x C
                            tvalid = outputs.get("target_valid")
                            if tvalid is not None:
                                mask = torch.as_tensor(tvalid).bool()
                                pp_pooled = pp_pooled[mask]
                            else:
                                pp_pooled = pp_pooled.reshape(-1, pp_pooled.shape[-1])
                            if pp_pooled.shape[0] >= 2:
                                z = pp_pooled.detach().cpu().numpy().astype(np.float64)
                                effective_rank = f"{compute_effective_rank_from_features(z):.8f}"
                except Exception as er:
                    log_error("effective_rank", er)
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
            log_error("artifact_generation", e)
    else:
        if is_main_process and is_3d_mode and os.path.exists(inf_path) and post_training_artifacts:
            try:
                outputs = torch.load(inf_path, map_location="cpu", weights_only=False)
                umap_meta_path = save_volumetric_umap_embeddings(session_dir, outputs, umap_cfg=umap_cfg)
                log_info(f"[{config_name}] volumetric_umap_saved={umap_meta_path}")
            except Exception as e:
                log_error("volumetric_umap", e)
        elif is_main_process and os.path.exists(inf_path) and not post_training_artifacts:
            log_info(f"[{config_name}] post_training_artifacts=false; skip PCA/UMAP artifact generation")
        elif is_main_process:
            log_info(f"[{config_name}] warning: inference_outputs.pt missing; skip artifact generation")

    if is_main_process and not is_3d_mode and bool(train_cfg.get("scale_probe_enabled", False)):
        try:
            from src.utils.scale_probe import (
                build_cdd_reconstructed_image_variants,
                probe_image_response,
                probe_scale_response,
            )

            model_without_ddp.eval()
            with torch.no_grad():
                if model_without_ddp.mode == "pyramid":
                    cdd_channels = None
                    if os.path.exists(inf_path):
                        inf_outputs = torch.load(inf_path, map_location="cpu", weights_only=False)
                        cdd_channels = inf_outputs.get("cdd_channels_orig")
                        if cdd_channels is not None:
                            cdd_channels = cdd_channels[:1]
                    if cdd_channels is None or cdd_channels.ndim != 4:
                        probe_item = next(iter(dataloader))
                        probe_batch = probe_item[0] if isinstance(probe_item, (tuple, list)) else probe_item
                        probe_batch = probe_batch.to(device, non_blocking=True)
                        # Let _prepare_context_from_model handle NaN internally.
                        ctx_result = _prepare_context_from_model(model_without_ddp, probe_batch, return_debug=True)
                        if len(ctx_result) >= 5:
                            debug = ctx_result[4]
                            cdd_channels = debug.get("cdd_channels_orig")
                    if cdd_channels is not None and cdd_channels.ndim == 4:
                        report = probe_scale_response(
                            model_without_ddp,
                            x_pyr=cdd_channels.to(device),
                            scale_names=train_cfg.get("scale_probe_names"),
                            out_dir=session_dir,
                            run_name=config_name,
                        )
                        log_info(f"[{config_name}] scale_probe_report={json.dumps(report['scale_drop_sensitivity_fraction'])}")
                    else:
                        log_info(f"[{config_name}] scale_probe: cdd_channels not available, skipping")
                elif model_without_ddp.mode == "image":
                    reference_input = None
                    source_image = None
                    if os.path.exists(inf_path):
                        inf_outputs = torch.load(inf_path, map_location="cpu", weights_only=False)
                        reference_input = inf_outputs.get("network_target_in")
                        if reference_input is not None:
                            reference_input = reference_input[:1]
                        source_image = inf_outputs.get("x_clean_raw")
                        if source_image is not None:
                            source_image = source_image[:1]
                    if (
                        reference_input is None
                        or reference_input.ndim != 4
                        or source_image is None
                        or source_image.ndim not in (3, 4)
                    ):
                        probe_item = next(iter(dataloader))
                        probe_batch = probe_item[0] if isinstance(probe_item, (tuple, list)) else probe_item
                        probe_batch = torch.nan_to_num(
                            probe_batch.to(device, non_blocking=True),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        probe_outputs = model_without_ddp(
                            probe_batch[:1],
                            return_debug=True,
                            enable_grid_jitter=False,
                            enable_target_dithering=False,
                            mask_inference=False,
                        )
                        reference_input = probe_outputs.get("network_target_in")
                        source_image = probe_outputs.get("x_clean_raw", probe_batch[:1])
                    if reference_input is not None and reference_input.ndim == 4:
                        variant_inputs = None
                        scale_only_inputs = None
                        variant_names = train_cfg.get(
                            "image_scale_probe_names",
                            train_cfg.get("scale_probe_image_names"),
                        )
                        try:
                            variant_inputs, scale_only_inputs, variant_names = build_cdd_reconstructed_image_variants(
                                reference_input.to(device),
                                source_image=None if source_image is None else source_image.to(device),
                                sigmas=tuple(model_cfg.get("sigmas", [2, 4, 8, 16])),
                                cdd_mode=str(model_cfg.get("cdd_mode", data_cfg.get("cdd_mode", "log"))),
                                cdd_constrained=bool(model_cfg.get("cdd_constrained", data_cfg.get("cdd_constrained", True))),
                                cdd_sm_mode=str(model_cfg.get("cdd_sm_mode", data_cfg.get("cdd_sm_mode", "reflect"))),
                                cdd_gaussian_backend=str(
                                    model_cfg.get("cdd_gaussian_backend", data_cfg.get("cdd_gaussian_backend", "cpu"))
                                ),
                                cdd_append_last_residual=bool(model_cfg.get("cdd_append_last_residual", True)),
                                post_log_transform=bool(model_cfg.get("post_log_transform", True)),
                                log_eps=float(model_cfg.get("log_eps", 1.0)),
                                cdd_log_std_floor_mult=float(model_cfg.get("cdd_log_std_floor_mult", 0.05)),
                                perturb_channel=int(train_cfg.get("image_scale_probe_channel", 0)),
                            )
                            log_info(
                                f"[{config_name}] image_scale_probe: using CDD reconstructed drop-scale variants "
                                f"names={list(variant_names)}"
                            )
                        except Exception as cdd_probe_error:
                            log_info(
                                f"[{config_name}] image_scale_probe: CDD reconstructed variants failed "
                                f"({type(cdd_probe_error).__name__}: {cdd_probe_error}); skipping image scale probe"
                            )
                        if variant_inputs is not None and scale_only_inputs is not None:
                            report = probe_image_response(
                                model_without_ddp,
                                reference_input=reference_input.to(device),
                                variant_inputs=variant_inputs,
                                scale_only_inputs=scale_only_inputs,
                                variant_names=variant_names,
                                out_dir=session_dir,
                                run_name=config_name,
                                perturb_channel=int(train_cfg.get("image_scale_probe_channel", 0)),
                            )
                            log_info(f"[{config_name}] image_scale_probe_report={json.dumps(report['scale_drop_sensitivity_fraction'])}")
                    else:
                        log_info(f"[{config_name}] scale_probe: image reference input not available, skipping")
                else:
                    log_info(f"[{config_name}] scale_probe: unsupported mode={model_without_ddp.mode}, skipping")
            model_without_ddp.train()
        except Exception as e:
            log_error("scale_probe", e)

    if is_ddp:
        dist.barrier()

    return session_dir
