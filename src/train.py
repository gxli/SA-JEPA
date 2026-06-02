from __future__ import annotations

import csv
import json
import math

from tqdm import tqdm
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.dataset import JEPADataset
from src.dataset3d import JEPA3DCropDataset
from src.diagnostics import (
    compute_effective_rank_from_features,
    compute_error_by_scale,
    rank_dashboard,
)
from src.inference import run_post_training_inference, run_post_training_inference_3d
from src.losses import (
    compute_jepa_energy,
    compute_raw_mse_and_norm_err,
    compute_sim_var_cov,
    compute_sim_var_cov_torch,
    compute_target_energy_map,
    embedding_spread_stats,
    extract_valid_pooled_embeddings,
    parse_spread_regularizer_config,
    spread_regularizer_loss,
)
from src.models.build_jepa import CDD_DEBUG_ENCODER_TYPES, PyramidGridJEPA
from src.models.build_jepa3d import PyramidGridJEPA3D
from src.models.masking import prepare_context_batch
from src.utils import log_error, set_error_log_path
from src.utils.viz import save_inference_dashboard, save_volumetric_umap_embeddings

def _fmt_metric(v: float) -> str:
    x = float(v)
    ax = abs(x)
    if ax == 0.0:
        return "0.0000"
    if ax < 1e-3 or ax >= 1e3:
        return f"{x:.3e}"
    return f"{x:.4f}"


def _flush_csv_rows(path: str, rows: list[list]) -> None:
    if not rows:
        return
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
    import glob as _glob

    pattern = os.path.join(data_cfg.get("data_root", "data"), data_cfg.get("npy_pattern", "*.npy"))
    files = sorted(_glob.glob(pattern))
    profile = {
        "pattern": pattern,
        "crop_mode": str(data_cfg.get("crop_mode", "none")),
        "crop_size": data_cfg.get("crop_size"),
        "files": [],
    }
    for path in files:
        item = {"path": path}
        item.update(_summarize_data_array(np.load(path, mmap_mode="r")))
        profile["files"].append(item)
        print(
            f"[{config_name}] Data profile: path={path} shape={tuple(item['shape'])} "
            f"nan={item['nan_count']} normalized_zero_fraction={item['normalized_zero_fraction']:.4f} "
            f"aspect_h_over_w={item['aspect_ratio_h_over_w']}"
        )
    with open(os.path.join(session_dir, "data_profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
        f.write("\n")


def _write_cdd_cache_profile(*, cdd_cache: dict | None, session_dir: str, config_name: str) -> None:
    entries = []
    for (path, slice_idx), value in sorted((cdd_cache or {}).items()):
        arr = np.asarray(value)
        finite = np.isfinite(arr)
        item = {
            "path": path,
            "slice_idx": slice_idx,
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
        print(
            f"[{config_name}] CDD cache profile: path={path} shape={tuple(item['shape'])} "
            f"nan={item['nan_count']} zero_fraction={item['zero_fraction']:.4f} max={item['max']}"
        )
    with open(os.path.join(session_dir, "cdd_cache_profile.json"), "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, indent=2)
        f.write("\n")


def _precompute_cdd_cache(
    *,
    data_cfg: dict,
    model_cfg: dict,
    device: torch.device,
    config_name: str,
) -> dict:
    """Pre-compute CDD decomposition for all data files on GPU, store in CPU RAM."""
    import glob as _glob
    import constrained_diffusion as cdd

    data_root = data_cfg.get("data_root", "data")
    npy_pattern = data_cfg.get("npy_pattern", "*.npy")
    cdd_mode = str(model_cfg.get("cdd_mode", data_cfg.get("cdd_mode", "log")))
    cdd_constrained = bool(model_cfg.get("cdd_constrained", data_cfg.get("cdd_constrained", True)))
    cdd_sm_mode = str(model_cfg.get("cdd_sm_mode", data_cfg.get("cdd_sm_mode", "reflect")))
    cdd_append_last_residual = bool(model_cfg.get("cdd_append_last_residual", True))
    sigmas = tuple(model_cfg.get("sigmas", [2, 4, 8, 16]))
    if device.type != "cuda":
        print(f"[{config_name}] CDD precompute: no CUDA, skipping")
        return {}
    npy_files = sorted(_glob.glob(os.path.join(data_root, npy_pattern)))
    if not npy_files:
        print(f"[{config_name}] CDD precompute: no files found, skipping")
        return {}
    print(f"[{config_name}] CDD precompute: {len(npy_files)} file(s) on GPU...")
    cache = {}
    for path in npy_files:
        arr = np.load(path, mmap_mode="r").astype(np.float32)
        # Normalize01 (same order as JEPADataset._preprocess_arr2d).
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        amin, amax = float(arr.min()), float(arr.max())
        if amax - amin > 1e-20:
            arr = (arr - amin) / (amax - amin)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        if arr.ndim not in (2, 3):
            raise ValueError(f"Unexpected ndim={arr.ndim} for {path}")
        # Preserve volumetric context: decompose a cube once, then let the
        # dataset choose a 2D slice from every cached CDD channel together.
        cdd_channels_arr, cdd_residual = cdd.constrained_diffusion_decomposition(
            arr,
            num_channels=len(sigmas),
            max_scale=max(float(s) for s in sigmas),
            mode=cdd_mode,
            constrained=cdd_constrained,
            sm_mode=cdd_sm_mode,
            return_scales=False,
            verbose=False,
            use_gpu=True,
        )
        # CDD always introduces one leading scale axis:
        # image (H,W) -> (S,H,W), cube (D,H,W) -> (S,D,H,W).
        cdd_orig = np.clip(np.stack(cdd_channels_arr, axis=0).astype(np.float32), a_min=0.0, a_max=None)
        if cdd_orig.ndim != arr.ndim + 1 or cdd_orig.shape[1:] != arr.shape:
            raise ValueError(
                f"CDD output for {path} must have one leading scale axis over input shape {arr.shape}, "
                f"got {cdd_orig.shape}"
            )
        if cdd_append_last_residual:
            cdd_residual = np.asarray(cdd_residual, dtype=np.float32)
            if cdd_residual.shape != arr.shape:
                raise ValueError(
                    f"CDD residual for {path} must match input shape {arr.shape}, got {cdd_residual.shape}"
                )
            cdd_orig[-1] = cdd_orig[-1] + cdd_residual
        cache[(path, None)] = cdd_orig.astype(np.float32, copy=False)
    # Free GPU memory used by CDD.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[{config_name}] CDD precompute: {len(cache)} entries cached, GPU freed")
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


class _MaskingCollator:
    def __init__(self, model: PyramidGridJEPA, return_debug: bool = False):
        enc_type = str(getattr(model, "encoder_type", "")).lower()
        self.return_debug = bool(
            return_debug
            or enc_type in CDD_DEBUG_ENCODER_TYPES
        )
        self.mask_scale = float(model.mask_scale)
        self.mask_scale_range = model.mask_scale_range
        self.mask_box_size = int(model.mask_box_size)
        self.mask_box_size_range = model.mask_box_size_range
        self.context_kwargs = {
            "sigmas": model.sigmas,
            "mask_fraction": model.mask_fraction,
            "spacing_scale": model.spacing_scale,
            "global_shift": model.global_shift,
            "align_scales": model.align_scales,
            "patch_size": model.patch_size,
            "return_debug": self.return_debug,
            "target_invalid_region_skip": model.target_invalid_region_skip,
            "target_invalid_region_values": model.target_invalid_region_values,
            "target_sampling_mode": model.target_sampling_mode,
            "priority_top_percent": model.priority_top_percent,
            "priority_n_target": model.priority_n_target,
            "priority_min_targets_per_map": model.priority_min_targets_per_map,
            "priority_dithering_pixels": model.priority_dithering_pixels,
            "target_nonoverlap": getattr(model, "target_nonoverlap", False),
            "target_allow_partial_overlap": getattr(model, "target_allow_partial_overlap", 0.0),
            "mask_box_hardcap": getattr(model, "mask_box_hardcap", None),
        }

    def _sample_mask_params(self) -> tuple[float, int]:
        mask_scale = self.mask_scale
        if self.mask_scale_range is not None:
            lo, hi = self.mask_scale_range
            mask_scale = lo + (hi - lo) * float(torch.rand(()).item()) if hi > lo else lo

        mask_box_size = self.mask_box_size
        if self.mask_box_size_range is not None:
            lo, hi = self.mask_box_size_range
            mask_box_size = int(torch.randint(lo, hi + 1, ()).item()) if hi > lo else lo
        return float(mask_scale), int(mask_box_size)

    def __call__(self, batch):
        use_cdd = isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2
        if use_cdd:
            cdd_list = [item[0] for item in batch]
            x_clean_list = [item[1] for item in batch]
            cdd_orig_in = _collate_pad_spatial(cdd_list)
            x_clean = _collate_pad_spatial(x_clean_list)
        else:
            cdd_orig_in = None
            x_clean = _collate_pad_spatial(batch)
        mask_scale, mask_box_size = self._sample_mask_params()
        context_data = prepare_context_batch(
            x_clean=x_clean,
            mask_scale=mask_scale,
            mask_box_size=mask_box_size,
            cdd_orig_in=cdd_orig_in,
            **self.context_kwargs,
        )
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
    )
    mask_scale, mask_box_size = model.sample_mask_params(device=x_clean.device)
    return prepare_context_batch(
        x_clean=x_clean,
        sigmas=model.sigmas,
        mask_fraction=model.mask_fraction,
        mask_scale=mask_scale,
        spacing_scale=model.spacing_scale,
        global_shift=model.global_shift,
        align_scales=model.align_scales,
        mask_box_size=mask_box_size,
        cdd_mode=model.cdd_mode,
        cdd_constrained=model.cdd_constrained,
        cdd_sm_mode=model.cdd_sm_mode,
        cdd_append_last_residual=model.cdd_append_last_residual,
        patch_size=model.patch_size,
        return_debug=need_debug,
        target_invalid_region_skip=model.target_invalid_region_skip,
        target_invalid_region_values=model.target_invalid_region_values,
        target_sampling_mode=model.target_sampling_mode,
        priority_top_percent=model.priority_top_percent,
        priority_n_target=model.priority_n_target,
        priority_min_targets_per_map=model.priority_min_targets_per_map,
        priority_dithering_pixels=model.priority_dithering_pixels,
        target_nonoverlap=getattr(model, "target_nonoverlap", False),
        target_allow_partial_overlap=getattr(model, "target_allow_partial_overlap", 0.0),
        mask_box_hardcap=getattr(model, "mask_box_hardcap", None),
        cdd_use_gpu=(x_clean.device.type == "cuda"),
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
        if len(context_result) == 5:
            x_context, tloc, tscale, tvalid, debug = context_result
        else:
            x_context, tloc, tscale, tvalid = context_result
            debug = {}
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
        return {"val_loss": 0.0, "val_sim": 0.0, "val_error_by_scale": {}}
    return {
        "val_loss": loss_sum / n,
        "val_sim": sim_sum / n,
        "val_error_by_scale": {float(s): float(np.mean(v)) for s, v in scale_mse.items()},
    }


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
            cfg = json.load(f)

        base_ref = cfg.pop("base_config", None)
        if base_ref is None:
            merged = cfg
        else:
            base_path = base_ref
            if not os.path.isabs(base_path):
                base_path = os.path.join(os.path.dirname(abs_path), base_path)
            base_cfg = _load_with_base(base_path, seen)
            merged = _deep_merge(base_cfg, cfg)
        seen.remove(abs_path)
        return merged

    cfg = _load_with_base(path, seen=set())
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    if "log_transform" in cfg["data"]:
        legacy_value = cfg["data"].pop("log_transform")
        print(
            "[config migration] ignoring removed data.log_transform="
            f"{legacy_value!r}; dataset preprocessing is always linear normalize01. "
            "Use model.post_log_transform."
        )
    if "image_size" in cfg["data"]:
        legacy_value = cfg["data"].pop("image_size")
        print(
            "[config migration] ignoring removed data.image_size="
            f"{legacy_value!r}; JEPADataset always preserves native input resolution."
        )
    if "log_transform" in cfg["model"]:
        legacy_value = cfg["model"].pop("log_transform")
        print(
            "[config migration] ignoring removed model.log_transform="
            f"{legacy_value!r}; use model.post_log_transform."
        )
    return cfg


def make_session_dir(root: str, config_name: str) -> str:
    path = os.path.join(root, config_name)
    os.makedirs(path, exist_ok=True)
    return path


def resolve_pipeline_config(data_cfg: dict, model_cfg: dict) -> bool:
    return bool(model_cfg.get("post_log_transform", True))


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


def _resolve_encoder_alias_2d(name: str) -> str:
    key = str(name).lower()
    alias = {
        # Preferred naming convention (image / image_pyramid prefixes).
        "convnext_image_dense_masked": "convnext_dense_masktoken",
        "cdd_scaleaware_convnext-pyramid-scaleaware": "cdd_scaleaware_convnext",
        "cdd_scaleaware_convnext-pyramid-scaleaware-d4": "cdd_scaleaware_convnext_d4",
        "rescnn-pyramid-scaleaware": "cdd_scaleaware_rescnn",
        "image_pyramid_cdd_scaleaware_convnext": "cdd_scaleaware_convnext",
        "image_pyramid_cdd_scaleaware_convnext_d4": "cdd_scaleaware_convnext_d4",
        "image_pyramid_cdd_scaleaware_rescnn": "cdd_scaleaware_rescnn",
        # Supported canonical names.
        "convnext_dense_masktoken": "convnext_dense_masktoken",
        "cdd_scaleaware_convnext": "cdd_scaleaware_convnext",
        "cdd_scaleaware_convnext_d4": "cdd_scaleaware_convnext_d4",
        "cdd_scaleaware_rescnn": "cdd_scaleaware_rescnn",
        # Supported aliases.
        "convnext-pyramid-scaleaware": "cdd_scaleaware_convnext",
    }
    return alias.get(key, str(name))


def _resolve_encoder_alias_3d(name: str) -> str:
    key = str(name).lower()
    alias = {
        "convnext_dense3d": "convnext_dense3d",
        "cdd_scaleaware_convnext3d": "cdd_scaleaware_convnext3d",
    }
    return alias.get(key, str(name))


def build_model_from_config(model_cfg: dict, data_cfg: dict, train_cfg: dict, device: torch.device) -> PyramidGridJEPA:
    """Construct a PyramidGridJEPA from config dicts."""
    mask_spacing_scaling = float(model_cfg.get("mask_spacing_scaling", 1.5))
    model_post_log = resolve_pipeline_config(data_cfg=data_cfg, model_cfg=model_cfg)
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
            "cdd_scaleaware_convnext_d4",
            "cdd_scaleaware_rescnn",
            "convnext_dense_pyramid",
        }
        if resolved_encoder_type not in allowed_pyramid:
            raise ValueError(
                f"Unsupported pyramid-mode encoder_type={resolved_encoder_type}. "
                "Allowed: cdd_scaleaware_convnext, cdd_scaleaware_convnext_d4, "
                "cdd_scaleaware_rescnn, convnext_dense_pyramid."
            )
    else:
        raise ValueError(f"Unsupported mode={resolved_mode}. Allowed: image, pyramid.")

    patch_size = int(model_cfg.get("patch_size", 3))
    if patch_size <= 0:
        patch_size = 3
    if patch_size % 2 == 0:
        patch_size = patch_size + 1
        print(f"[warning] even patch_size requested; using odd patch_size={patch_size}")

    normalize_loss_l2 = bool(model_cfg.get("normalize_loss_l2", model_cfg.get("normalize_loss", False)))
    active_target_fraction = float(model_cfg.get("active_target_fraction", model_cfg.get("mask_fraction", 1.0)))
    return PyramidGridJEPA(
        latent_channels=model_cfg.get("latent_channels", 32),
        predictor_hidden=model_cfg.get("predictor_hidden"),
        patch_size=patch_size,
        sigmas=tuple(model_cfg.get("sigmas", [2, 4, 8, 16])),
        mask_fraction=active_target_fraction,
        mask_scale=model_cfg.get("mask_scale_factor", 1.0),
        mask_scale_range=model_cfg.get("mask_scale_factor_range"),
        spacing_scale=mask_spacing_scaling,
        global_shift=model_cfg.get("global_shift", True),
        align_scales=model_cfg.get("align_scales", True),
        mask_box_size=model_cfg.get("mask_footprint_px", 16),
        mask_box_size_range=model_cfg.get("mask_footprint_px_range"),
        cdd_mode=model_cfg.get("cdd_mode", data_cfg.get("cdd_mode", "log")),
        cdd_constrained=model_cfg.get("cdd_constrained", data_cfg.get("cdd_constrained", True)),
        cdd_sm_mode=model_cfg.get("cdd_sm_mode", data_cfg.get("cdd_sm_mode", "reflect")),
        cdd_append_last_residual=bool(model_cfg.get("cdd_append_last_residual", True)),
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
        target_invalid_region_skip=bool(model_cfg.get("target_invalid_region_skip", False)),
        target_invalid_region_values=tuple(model_cfg.get("target_invalid_region_values", [0, "nan"])),
        target_sampling_mode=str(model_cfg.get("target_sampling_mode", "grid")),
        priority_top_percent=float(model_cfg.get("priority_top_percent", 5.0)),
        priority_n_target=model_cfg.get("priority_n_target", 20),
        priority_min_targets_per_map=int(model_cfg.get("priority_min_targets_per_map", 0)),
        priority_dithering_pixels=int(model_cfg.get("priority_dithering_pixels", model_cfg.get("target_dithering_pixels", 6))),
        use_symmetric_feature_loss=bool(model_cfg.get("use_symmetric_feature_loss", False)),
        target_nonoverlap=bool(model_cfg.get("target_nonoverlap", False)),
        target_allow_partial_overlap=float(model_cfg.get("target_allow_partial_overlap", 0.0)),
        mask_box_hardcap=model_cfg.get("mask_box_hardcap"),
        use_grn=bool(model_cfg.get("use_grn", True)),
    ).to(device)


def build_model3d_from_config(model_cfg: dict, train_cfg: dict, device: torch.device) -> PyramidGridJEPA3D:
    mode = str(model_cfg.get("mode", "")).strip().lower().replace(" ", "_")
    if mode != "3d_slab":
        raise ValueError(
            f"Unsupported 3D JEPA mode={model_cfg.get('mode')}. "
            "Use model.mode='3d_slab'; full-volume and slice variants were removed."
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
    return PyramidGridJEPA3D(
        latent_channels=int(model_cfg.get("latent_channels", 16)),
        scale_channels=int(model_cfg.get("scale_channels", model_cfg.get("encoder_width", 8))),
        patch_size=int(model_cfg.get("patch_size", 2)),
        num_targets=int(model_cfg.get("num_targets", 32)),
        encoder_depth=int(model_cfg.get("encoder_depth", 3)),
        encoder_kernel_size=int(model_cfg.get("encoder_kernel_size", 5)),
        encoder_stride=int(model_cfg.get("encoder_stride", 1)),
        ema_momentum=float(model_cfg.get("ema_momentum", train_cfg.get("momentum", 0.996))),
        normalize_loss_l2=normalize_loss_l2,
        post_log_transform=bool(model_cfg.get("post_log_transform", True)),
        log_eps=float(model_cfg.get("log_eps", 1e-6)),
        fusion=fusion,
        mask_box_size=int(model_cfg.get("mask_footprint_px", 8)),
        num_mask_boxes=int(model_cfg.get("num_mask_boxes", 8)),
        slab_depth=int(model_cfg.get("slab_depth", max(1, int(model_cfg.get("patch_size", 2))))),
        use_symmetric_feature_loss=bool(model_cfg.get("use_symmetric_feature_loss", False)),
        use_film=bool(model_cfg.get("use_film", True)),
        use_per_scale_adapters=bool(model_cfg.get("use_per_scale_adapters", False)),
    ).to(device)


def run_training(config: dict, config_name: str, sessions_root: str = "sessions") -> str:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    print(
        f"[{config_name}] Backend discovered: device={device.type}, "
        f"cuda_available={torch.cuda.is_available()}, mps_available={mps_available}"
    )

    train_cfg = config["train"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    is_3d_mode = _is_3d_jepa_mode(model_cfg.get("mode", "image"))

    session_dir = make_session_dir(sessions_root, config_name)
    set_error_log_path(os.path.join(session_dir, "errors.log"))
    os.makedirs(session_dir, exist_ok=True)
    model_ckpt_path = os.path.join(session_dir, "model_last.pt")
    resume_ckpt_path = os.path.join(session_dir, "checkpoint_last.pt")
    resume_from_existing = os.path.exists(model_ckpt_path)

    with open(os.path.join(session_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

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
        raise ValueError("model.mode='3d_slab' requires data.input_type='cube'.")
    image_batch_inference = (input_type == "image_batch")

    resolve_pipeline_config(data_cfg=data_cfg, model_cfg=model_cfg)
    _write_data_profile(data_cfg=data_cfg, session_dir=session_dir, config_name=config_name)

    print(
        f"[{config_name}] Resolved pipeline: "
        "dataset_preprocess=normalize01, "
        f"model.post_log_transform={model_cfg.get('post_log_transform', True)}, "
        f"cdd_mode={model_cfg.get('cdd_mode', data_cfg.get('cdd_mode', 'log'))}"
    )

    model = build_model3d_from_config(model_cfg, train_cfg, device) if is_3d_mode else build_model_from_config(model_cfg, data_cfg, train_cfg, device)
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
            try:
                missing, unexpected = model.load_state_dict(resume_state["model_state_dict"], strict=False)
            except RuntimeError as e:
                # Common during architecture evolution (e.g. channel-count changes).
                log_error("resume_model_load_state_dict", e)
                missing, unexpected = ["__load_state_dict_failed__"], []
            print(f"[{config_name}] Resume model: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
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
                print(f"[{config_name}] resume_checkpoint_ignored={resume_ckpt_path}")
        if resume_state is not None:
            start_epoch = int(resume_state.get("epoch", 0))
            print(f"resume_checkpoint={resume_ckpt_path} start_epoch={start_epoch}")
    elif resume_from_existing:
        try:
            missing, unexpected = model.load_state_dict(torch.load(model_ckpt_path, map_location=device), strict=False)
        except RuntimeError as e:
            # Common during architecture evolution (e.g. channel-count changes).
            log_error("resume_model_load_state_dict", e)
            missing, unexpected = ["__load_state_dict_failed__"], []
        print(f"[{config_name}] Resume model: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
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
            model = (
                build_model3d_from_config(model_cfg, train_cfg, device)
                if is_3d_mode
                else build_model_from_config(model_cfg, data_cfg, train_cfg, device)
            )
            print(f"[{config_name}] resume_model_ignored={model_ckpt_path}")
        else:
            print(f"resume_model={model_ckpt_path}")

    scale_max = float(max(model_cfg.get("sigmas", [2, 4, 8, 16])))
    def _param_or_range_max(value_key: str, range_key: str, default: float) -> float:
        values = model_cfg.get(range_key, model_cfg.get(value_key, default))
        if isinstance(values, (list, tuple)):
            if len(values) != 2:
                raise ValueError(f"{value_key} range must contain exactly two values, got {values!r}")
            return float(max(values))
        return float(values)

    _msb = _param_or_range_max("mask_scale_factor", "mask_scale_factor_range", 1.0)
    _mb = int(round(_param_or_range_max("mask_footprint_px", "mask_footprint_px_range", 16)))
    _mss = float(model_cfg.get("mask_spacing_scaling", 1.5))
    max_box = round(scale_max * _msb + _mb)
    auto_roll_max = max(1, int(round(float(max_box) * _mss)))

    # --- image_batch pre-selection ---
    image_batch_selected_indices = None
    image_batch_n_sample = data_cfg.get("image_batch_n_sample", None)
    if input_type == "image_batch" and image_batch_n_sample is not None:
        import glob as _glob
        data_root = data_cfg.get("data_root", "data")
        npy_pattern = data_cfg.get("npy_pattern", "*.npy")
        npy_files = sorted(_glob.glob(os.path.join(data_root, npy_pattern)))
        selected = {}
        sel_path = os.path.join(session_dir, "selected_slices.json")
        rng = random.Random(int(train_cfg.get("split_seed", 42)))
        for fpath in npy_files:
            arr_mm = np.load(fpath, mmap_mode="r")
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
        with open(sel_path, "w", encoding="utf-8") as f:
            json.dump({k: list(v) for k, v in selected.items()}, f, indent=2)
        total_selected = sum(len(v) for v in selected.values())
        print(
            f"[{config_name}] image_batch_n_sample={image_batch_n_sample} "
            f"files={len(selected)} total_selected={total_selected} "
            f"saved_to={sel_path}"
        )

    # --- Pre-compute CDD once on GPU, store in CPU RAM ---
    cdd_cache = _precompute_cdd_cache(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        device=device,
        config_name=config_name,
    ) if not is_3d_mode else None
    _write_cdd_cache_profile(cdd_cache=cdd_cache, session_dir=session_dir, config_name=config_name)

    if is_3d_mode:
        dataset = JEPA3DCropDataset(
            data_root=data_cfg.get("data_root", "data"),
            npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
            num_samples=int(data_cfg.get("num_samples", 2000)),
            crop_size=int(data_cfg.get("volume_crop_size", data_cfg.get("crop_size_3d", 64))),
            normalize=bool(data_cfg.get("normalize", True)),
            crop_strategy=str(data_cfg.get("crop_strategy", "random")),
        )
        val_dataset = None
        train_dataset = dataset
        inference_dataset = JEPA3DCropDataset(
            data_root=data_cfg.get("data_root", "data"),
            npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
            num_samples=max(1, int(train_cfg.get("inference_num_samples", 8))),
            crop_size=int(data_cfg.get("volume_crop_size", data_cfg.get("crop_size_3d", 64))),
            normalize=bool(data_cfg.get("normalize", True)),
            crop_strategy="center",
        )
        train_idx = []
        val_idx = []
        n_total = len(dataset.npy_files)
        val_fraction = 0.0
    else:
        train_crop_mode = str(data_cfg.get("crop_mode", "none")).lower()
        train_crop_size = data_cfg.get("crop_size")
        val_crop_mode = "center" if train_crop_mode != "none" else "none"
        dataset = JEPADataset(
            num_samples=data_cfg.get("num_samples", 2000),
            data_root=data_cfg.get("data_root", "data"),
            npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
            cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
            cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
            cube_slice_index=data_cfg.get("cube_slice_index", 0),
            crop_mode=train_crop_mode,
            crop_size=train_crop_size,
            random_roll_max=int(max(0, data_cfg.get("random_roll_max", auto_roll_max))),
            d4_augment=bool(data_cfg.get("d4_augment", False)),
            input_type=input_type,
            image_batch_selected_indices=image_batch_selected_indices,
            cdd_cache=cdd_cache,
        )
        val_fraction = float(train_cfg.get("val_fraction", 0.1))
        val_fraction = min(max(val_fraction, 0.0), 0.95)
        total_idx = list(dataset.sample_index)
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
                cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
                cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
                cube_slice_index=data_cfg.get("cube_slice_index", 0),
                crop_mode=val_crop_mode,
                crop_size=train_crop_size,
                random_roll_max=int(max(0, data_cfg.get("random_roll_max", auto_roll_max))),
                d4_augment=False,
                input_type=input_type,
                image_batch_selected_indices=image_batch_selected_indices,
                cdd_cache=cdd_cache,
            )
            val_dataset.sample_index = val_idx
    print(
        f"[{config_name}] Dataset split: total_index={n_total}, train_index={len(train_idx)}, "
        f"val_index={len(val_idx)}, val_fraction={val_fraction:.3f}"
    )
    if hasattr(dataset, "random_roll_max"):
        print(
            f"[{config_name}] Data jitter: random_roll_max={dataset.random_roll_max} "
            f"(symmetric inclusive roll in [-max, +max])"
        )
    if (not is_3d_mode) and getattr(dataset, "crop_mode", "none") != "none":
        print(
            f"[{config_name}] Training crop: mode={dataset.crop_mode} "
            f"size={dataset.crop_size}; validation=center; inference=native"
        )
    requested_workers = int(train_cfg.get("num_workers", 4))
    # macOS/MPS-safe default: avoid multiprocessing worker hangs unless explicitly set.
    if "num_workers" in train_cfg:
        num_workers = requested_workers
    else:
        num_workers = 4 if device.type == "cuda" else 0
    pin_memory = bool(device.type == "cuda")
    persistent_workers = bool(num_workers > 0)
    prefetch_factor = max(1, int(train_cfg.get("prefetch_factor", 2))) if num_workers > 0 else None
    print(
        f"[{config_name}] Dataloader setup: num_workers={num_workers}, "
        f"pin_memory={pin_memory}, persistent_workers={persistent_workers}, "
        f"prefetch_factor={prefetch_factor}"
    )
    loader_worker_kwargs = {}
    if num_workers > 0:
        loader_worker_kwargs["prefetch_factor"] = prefetch_factor

    masking_collate = None if is_3d_mode else _MaskingCollator(
        model,
        return_debug=bool(train_cfg.get("debug_masking_tensors", False)),
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=masking_collate,
        **loader_worker_kwargs,
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
            collate_fn=masking_collate,
            **loader_worker_kwargs,
        )
    # Inference must use canonical orientation (no D4 augmentation).
    if not is_3d_mode:
        inference_dataset = JEPADataset(
            num_samples=train_dataset.num_samples,
            data_root=data_cfg.get("data_root", "data"),
            npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
            cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
            cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
            cube_slice_index=data_cfg.get("cube_slice_index", 0),
            crop_mode="none",
            crop_size=None,
            random_roll_max=int(max(0, data_cfg.get("random_roll_max", auto_roll_max))),
            d4_augment=False,
            input_type=input_type,
            image_batch_inference=image_batch_inference,
            image_batch_selected_indices=image_batch_selected_indices,
            cdd_cache=cdd_cache,
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
    scaler = GradScaler("cuda", enabled=use_amp)
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
                    print(
                        f"[{config_name}] warning: optimizer_state_incompatible, "
                        f"continuing from epoch {start_epoch} with fresh optimizer: {e}"
                    )
        if optimizer_state_loaded and "scaler_state_dict" in resume_state and torch.cuda.is_available():
            try:
                scaler.load_state_dict(resume_state["scaler_state_dict"])
            except Exception as e:
                log_error("scaler_state_incompatible", e)

    epochs = train_cfg.get("epochs", 20)
    log_interval = train_cfg.get("log_interval", 10)
    log_flush_interval = max(1, int(log_interval))
    diagnostic_interval = max(1, int(train_cfg.get("diagnostic_interval", log_flush_interval)))
    force_recompute_inference = bool(train_cfg.get("force_recompute_inference", False))
    inference_mask_passes = int(train_cfg.get("inference_mask_passes", 1))
    mask_inference = bool(train_cfg.get("mask_inference", False))
    viz_crop_border = bool(train_cfg.get("viz_crop_border", False))
    viz_crop_border_px = train_cfg.get("viz_crop_border_px")
    umap_cfg = dict(train_cfg.get("umap", {}))
    compute_effective_rank = bool(train_cfg.get("compute_effective_rank", False))
    inference_visit_batches = int(train_cfg.get("inference_visit_batches", 32))
    inference_tta_enabled = bool(train_cfg.get("inference_tta_enabled", False))
    inference_tta_mode = str(train_cfg.get("inference_tta_mode", "flip4"))
    print(f"[{config_name}] umap_config={json.dumps(umap_cfg, sort_keys=True)}")
    prediction_loss_weight = float(train_cfg.get("prediction_loss_weight", 100.0))
    spread_regularizer = parse_spread_regularizer_config(train_cfg)
    spread_regularizer_weight = float(spread_regularizer["weight"])
    embed_spread_target = float(spread_regularizer["target_std"])
    spread_regularizer_eps = float(spread_regularizer["eps"])
    experimental_losses = dict(train_cfg.get("experimental_losses", {}))
    assert "vicreg_var_weight" not in train_cfg
    assert "vicreg_cov_weight" not in train_cfg
    vicreg_var_weight = float(experimental_losses.get("vicreg_var_weight", 0.0))
    vicreg_cov_weight = float(experimental_losses.get("vicreg_cov_weight", 0.0))
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

    metrics_path = os.path.join(session_dir, "metrics.csv")
    metrics_header = [
        "epoch",
        "batch",
        "global_step",
        "loss_total",
        "loss_prediction",
        "lr",
        "loss_spread",
        "loss_symmetry",
        "weighted_prediction",
        "weighted_spread",
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
        "mask_footprint_mean_px",
        "mask_footprint_min_px",
        "mask_footprint_max_px",
        "mask_scale_factor",
        "time_sec",
    ]
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_header = list(reader.fieldnames or [])
        if existing_header != metrics_header:
            legacy_names = {
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

    loss_weights_path = os.path.join(session_dir, "loss_weights.json")
    if not os.path.exists(loss_weights_path):
        with open(loss_weights_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prediction_loss_weight": prediction_loss_weight,
                    "spread_regularizer": spread_regularizer,
                    "symmetry_loss_weight": symmetry_loss_weight,
                    "experimental_losses": experimental_losses,
                },
                f,
                indent=2,
            )
            f.write("\n")

    movie_dump_every_epoch = bool(train_cfg.get("movie_dump_every_epoch", False))
    movie_batch = next(iter(inference_loader)) if movie_dump_every_epoch else None
    movie_context_data = None

    model.train()
    start = time.time()
    visit_counts = None
    if start_epoch >= int(epochs):
        print(f"[{config_name}] checkpoint epoch {start_epoch} already >= configured epochs {epochs}, skipping training loop")
    prev_epochs = []
    for epoch in range(start_epoch, epochs):
        epoch_total = 0.0
        epoch_prediction = 0.0
        epoch_sim = 0.0
        epoch_var = 0.0
        epoch_cov = 0.0
        epoch_spread = 0.0
        epoch_symmetric = 0.0
        epoch_valid_frac = 0.0
        epoch_embed_spread_mean = 0.0
        epoch_context_manifold_size = 0.0
        epoch_batches = 0
        metrics_rows = []
        masked_scale_rows = []
        visited_rows = []
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
        metrics_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=False, dynamic_ncols=True)
        for batch_idx, batch in pbar:
            if is_3d_mode:
                x_clean = batch.to(device, non_blocking=True)
                x_clean = torch.nan_to_num(x_clean, nan=0.0, posinf=0.0, neginf=0.0)
                context_data = None
            else:
                x_clean, context_result = batch
                x_clean = x_clean.to(device, non_blocking=True)
                context_result = _move_to_device(context_result, device)
                if len(context_result) == 5:
                    x_context, tloc, tscale, tvalid, debug = context_result
                else:
                    x_context, tloc, tscale, tvalid = context_result
                    debug = {}
                context_data = (x_context, tloc, tscale, tvalid, debug)

            # Compute LR before forward/backward/step so the current batch
            # actually uses the intended LR.
            current_step = epoch * max(1, len(dataloader)) + batch_idx
            total_steps = max(1, int(epochs) * max(1, len(dataloader)))
            warmup_steps = int(warmup_epochs * max(1, len(dataloader)))
            if current_step < warmup_steps:
                lr = base_lr * float(current_step + 1) / max(1, warmup_steps)
            else:
                progress_lr = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
                progress_lr = min(1.0, max(0.0, float(progress_lr)))
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress_lr))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=autocast_device, enabled=use_amp):
                outputs = model(x_clean, context_data=context_data) if not is_3d_mode else model(x_clean)
                loss_prediction = model.compute_loss(outputs)
                _, var_term_t, cov_term_t = compute_sim_var_cov_torch(
                    outputs,
                    spatial_mode=vicreg_spatial_mode,
                )
                z_ctx_key = "context_patches" if "context_patches" in outputs else "pred_patches"
                z_ctx = extract_valid_pooled_embeddings(outputs, key=z_ctx_key)
                loss_spread = spread_regularizer_loss(
                    z_ctx,
                    target_std=embed_spread_target,
                    eps=spread_regularizer_eps,
                )
                loss_symmetry = model.compute_symmetric_loss(outputs)
                total_loss = (
                    (prediction_loss_weight * loss_prediction)
                    + (vicreg_var_weight * var_term_t)
                    + (vicreg_cov_weight * cov_term_t)
                    + (spread_regularizer_weight * loss_spread)
                    + (symmetry_loss_weight * loss_symmetry)
                )
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress = min(1.0, max(0.0, float(current_step) / float(total_steps)))

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
            model.ema_momentum = new_momentum
            model.update_target_encoder()
            sim_val, var_val, cov_val = compute_sim_var_cov(
                outputs,
                spatial_mode=vicreg_spatial_mode,
            )
            raw_mse_val, norm_err_val = compute_raw_mse_and_norm_err(outputs)
            valid_frac = float(outputs["target_valid"].float().mean().item())
            ctx_stats = embedding_spread_stats(z_ctx, target_std=embed_spread_target)
            targets_per_image = float(outputs["target_valid"].float().sum(dim=1).mean().item())
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
            mask_scale_factor = float(outputs.get("mask_scale_factor", model.mask_scale))

            elapsed = time.time() - start
            metrics_rows.append(
                [
                    epoch + 1,
                    batch_idx,
                    epoch * max(1, len(dataloader)) + batch_idx,
                    float(total_loss.item()),
                    float(loss_prediction.item()),
                    float(lr),
                    float(loss_spread.item()),
                    float(loss_symmetry.item()),
                    float((prediction_loss_weight * loss_prediction).item()),
                    float((spread_regularizer_weight * loss_spread).item()),
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
                    mask_footprint_mean_px,
                    mask_footprint_min_px,
                    mask_footprint_max_px,
                    mask_scale_factor,
                    round(elapsed, 4),
                ]
            )
            should_log_diagnostics = (
                ((batch_idx + 1) % diagnostic_interval == 0)
                or ((batch_idx + 1) == len(dataloader))
            )
            if "cdd_channels_masked" in outputs:
                cube_path = os.path.join(session_dir, "example_masked_channel_cube.npy")
                if not os.path.exists(cube_path):
                    np.save(
                        cube_path,
                        outputs["cdd_channels_masked"][0].detach().cpu().numpy().astype(np.float32),
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
                    hh, ww = int(outputs["x_clean"].shape[-2]), int(outputs["x_clean"].shape[-1])
                    visit_counts = np.zeros((hh, ww), dtype=np.float32)
                for bi in range(tloc.shape[0]):
                    for ki in range(tloc.shape[1]):
                        if not bool(tvalid[bi, ki]):
                            continue
                        yy = int(tloc[bi, ki, 0])
                        xx = int(tloc[bi, ki, 1])
                        if (visit_counts is not None) and 0 <= yy < visit_counts.shape[0] and 0 <= xx < visit_counts.shape[1]:
                            visit_counts[yy, xx] += 1.0
                        visited_rows.append(
                            [
                                epoch + 1,
                                batch_idx,
                                bi,
                                ki,
                                yy,
                                xx,
                                float(scales[bi, ki]),
                            ]
                        )
            if (batch_idx + 1) % log_flush_interval == 0:
                _flush_csv_rows(masked_scales_log_path, masked_scale_rows)
                _flush_csv_rows(visited_targets_log_path, visited_rows)
            metrics_bar.set_description_str(
                f"total={total_loss.item():.4f} prediction={loss_prediction.item():.4f} "
                f"spread={loss_spread.item():.4f} sim={sim_val:.4f} "
                f"embed_spread={ctx_stats['embed_spread_mean']:.3e} "
                f"manifold={ctx_stats['context_manifold_size']:.2f} "
                f"vf={valid_frac:.3f} lr={lr:.1e}",
                refresh=True,
            )
            epoch_total += float(total_loss.item())
            epoch_prediction += float(loss_prediction.item())
            epoch_sim += float(sim_val)
            epoch_var += float(var_val)
            epoch_cov += float(cov_val)
            epoch_spread += float(loss_spread.item())
            epoch_symmetric += float(loss_symmetry.item())
            epoch_valid_frac += float(valid_frac)
            epoch_embed_spread_mean += ctx_stats["embed_spread_mean"]
            epoch_context_manifold_size += ctx_stats["context_manifold_size"]
            epoch_batches += 1

        metrics_bar.close()
        if metrics_rows:
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(metrics_rows)
        _flush_csv_rows(masked_scales_log_path, masked_scale_rows)
        _flush_csv_rows(visited_targets_log_path, visited_rows)
        if visit_counts is not None:
            np.save(os.path.join(session_dir, "visited_target_frequency.npy"), visit_counts.astype(np.float32))

        if epoch_batches > 0:
            avg_total = epoch_total / epoch_batches
            avg_prediction = epoch_prediction / epoch_batches
            prev_epochs.append(avg_total)
            prev_str = " | ".join(
                f"e{e_idx + 1}={prev_epochs[e_idx]:.4f}"
                for e_idx in range(max(0, len(prev_epochs) - 5), len(prev_epochs) - 1)
            )
            if prev_str:
                prev_str = f" [{prev_str}]"
            tqdm.write(
                f"[{config_name}] Epoch {epoch + 1}/{epochs} "
                f"total={avg_total:.4f} prediction={avg_prediction:.4f} "
                f"spread={_fmt_metric(epoch_spread/epoch_batches)} "
                f"sim={_fmt_metric(epoch_sim/epoch_batches)} "
                f"embed_spread={_fmt_metric(epoch_embed_spread_mean/epoch_batches)} "
                f"manifold={_fmt_metric(epoch_context_manifold_size/epoch_batches)} "
                f"vfrac={_fmt_metric(epoch_valid_frac/epoch_batches)}"
                f"{prev_str}"
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
        tqdm.write(f"[{config_name}] ckpt_saved epoch={epoch + 1}")

        # Per-epoch embedding snapshot for movie generation.
        if movie_dump_every_epoch:
            movie_dir = os.path.join(session_dir, "movie_frames")
            os.makedirs(movie_dir, exist_ok=True)
            with torch.no_grad():
                model.eval()
                if isinstance(movie_batch, (tuple, list)):
                    movie_batch = movie_batch[1] if movie_batch[1] is not None else movie_batch[0]
                x_movie = movie_batch.to(device, non_blocking=True)
                # Let _prepare_context_from_model handle NaN internally.
                if is_3d_mode:
                    x_movie = torch.nan_to_num(x_movie, nan=0.0, posinf=0.0, neginf=0.0)
                    out_m = model(x_movie)
                else:
                    if movie_context_data is None:
                        movie_context_data = _prepare_context_from_model(model, x_movie, return_debug=False)
                    out_m = model(x_movie, context_data=movie_context_data)
                frame = {
                    "pred_map": out_m["pred_map"][:1].detach().cpu(),
                    "gt_map": out_m["gt_map"][:1].detach().cpu(),
                    "x_clean": x_movie[:1].detach().cpu(),
                }
                torch.save(frame, os.path.join(movie_dir, f"epoch_{epoch + 1:04d}.pt"))
            model.train()

    torch.save(model.state_dict(), os.path.join(session_dir, "model_last.pt"))

    if is_3d_mode:
        session_dir = run_post_training_inference_3d(
            model=model,
            dataloader=inference_loader,
            session_dir=session_dir,
            config_name=config_name,
            force_recompute_inference=force_recompute_inference,
        )
    else:
        session_dir = run_post_training_inference(
            model=model,
            dataloader=inference_loader,
            session_dir=session_dir,
            config_name=config_name,
            visit_counts=visit_counts,
            force_recompute_inference=force_recompute_inference,
            inference_mask_passes=inference_mask_passes,
            mask_inference=mask_inference,
            viz_crop_border=viz_crop_border,
            viz_crop_border_px=viz_crop_border_px,
            compute_jepa_energy_fn=compute_jepa_energy,
            compute_target_energy_map_fn=compute_target_energy_map,
            inference_visit_batches=inference_visit_batches,
            training_d4_augment=bool(data_cfg.get("d4_augment", False)),
            inference_tta_enabled=inference_tta_enabled,
            inference_tta_mode=inference_tta_mode,
        )
    # Save NPY artifacts (PCA/UMAP/latent embeddings) required by session_to_dash.py.
    # No PNG, HTML, or dashboard rendering is performed here.
    inf_path = os.path.join(session_dir, "inference_outputs.pt")
    if (not is_3d_mode) and os.path.exists(inf_path):
        try:
            outputs = torch.load(inf_path, map_location="cpu")
            artifacts_dir = save_inference_dashboard(session_dir, outputs, umap_cfg=umap_cfg)
            print(f"[{config_name}] artifacts_saved={artifacts_dir}")
            effective_rank = ""
            rank_diag = {}
            try:
                rank_diag = rank_dashboard(outputs)
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
        if is_3d_mode and os.path.exists(inf_path):
            try:
                outputs = torch.load(inf_path, map_location="cpu")
                umap_meta_path = save_volumetric_umap_embeddings(session_dir, outputs, umap_cfg=umap_cfg)
                print(f"[{config_name}] volumetric_umap_saved={umap_meta_path}")
            except Exception as e:
                log_error("volumetric_umap", e)
        else:
            print(f"[{config_name}] warning: inference_outputs.pt missing; skip artifact generation")

    if not is_3d_mode and bool(train_cfg.get("scale_probe_enabled", False)) and model.mode == "pyramid":
        try:
            from src.utils.scale_probe import probe_scale_response

            probe_batch, _ = next(iter(dataloader))
            probe_batch = probe_batch.to(device, non_blocking=True)
            # Let _prepare_context_from_model handle NaN internally.
            model.eval()
            with torch.no_grad():
                ctx_result = _prepare_context_from_model(model, probe_batch, return_debug=True)
                if len(ctx_result) >= 5:
                    debug = ctx_result[4]
                    cdd_channels = debug.get("cdd_channels_orig")
                    if cdd_channels is not None and cdd_channels.ndim == 4:
                        report = probe_scale_response(
                            model,
                            x_pyr=cdd_channels.to(device),
                            scale_names=train_cfg.get("scale_probe_names"),
                            out_dir=session_dir,
                            run_name=config_name,
                        )
                        print(f"[{config_name}] scale_probe_report={json.dumps(report['scale_drop_sensitivity_fraction'])}")
                    else:
                        print(f"[{config_name}] scale_probe: cdd_channels not in debug, skipping")
            model.train()
        except Exception as e:
            log_error("scale_probe", e)

    return session_dir
