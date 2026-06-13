================================================================================
FILE: src/diagnostics.py
================================================================================

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


MAX_RANK_TOKENS = 65536


def _subsample_rows_np(z: np.ndarray, max_rows: int = MAX_RANK_TOKENS) -> np.ndarray:
    if z.shape[0] <= int(max_rows):
        return z
    rng = np.random.default_rng(42)
    idx = rng.choice(z.shape[0], size=int(max_rows), replace=False)
    return z[idx]


def _subsample_rows_torch(z: torch.Tensor, max_rows: int = MAX_RANK_TOKENS) -> torch.Tensor:
    if z.shape[0] <= int(max_rows):
        return z
    idx = torch.randperm(z.shape[0], device=z.device)[:int(max_rows)]
    return z.index_select(0, idx)


def compute_effective_rank_from_features(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 1:
        return 0.0
    z = _subsample_rows_np(z)
    z = z - z.mean(axis=0, keepdims=True)
    denom = max(1, z.shape[0] - 1)
    try:
        svals = np.linalg.svd(z, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        jitter_scale = max(float(np.std(z)), 1e-12) * 1e-5
        z_jitter = z + np.random.normal(0, jitter_scale, z.shape)
        svals = np.linalg.svd(z_jitter, full_matrices=False, compute_uv=False)
    evals = np.square(svals) / denom
    s = float(evals.sum())
    if s <= 0.0:
        return 0.0
    p = evals / s
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    h = float(-np.sum(p * np.log(p)))
    return float(np.exp(h))


def spectral_rank_stats(fmap: torch.Tensor, eps: float = 1e-12, dead_thresh: float = 1e-5) -> dict:
    """
    fmap: C,H,W or B,C,H,W
    Computes covariance spectral stats over spatial tokens.
    """
    if fmap.ndim == 3:
        z = fmap.permute(1, 2, 0).reshape(-1, fmap.shape[0])
    elif fmap.ndim == 4:
        z = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])
    else:
        raise ValueError(f"Expected C,H,W or B,C,H,W, got {tuple(fmap.shape)}")

    z = z.float()
    z = _subsample_rows_torch(z)
    z = z - z.mean(dim=0, keepdim=True)

    ch_std = z.std(dim=0, unbiased=False)
    dead = ch_std < float(dead_thresh)
    dead_frac = dead.float().mean().item()

    denom = max(1, z.shape[0] - 1)
    try:
        eig_raw = torch.linalg.svdvals(z)
    except RuntimeError:
        jitter_scale = z.std(unbiased=False).clamp_min(1e-12) * 1e-5
        z_jitter = z + torch.randn_like(z) * jitter_scale
        eig_raw = torch.linalg.svdvals(z_jitter)
    eig = eig_raw.pow(2).div(float(denom)).clamp_min(0)

    total = eig.sum().clamp_min(eps)
    p = eig / total
    entropy = -(p * torch.log(p.clamp_min(eps))).sum()
    erank = torch.exp(entropy)

    top1 = p[:1].sum()
    top4 = p[:4].sum()
    top8 = p[:8].sum()
    participation = total.pow(2) / eig.pow(2).sum().clamp_min(eps)

    return {
        "erank": float(erank.item()),
        "manifold_size": float(participation.item()),
        "top1_energy": float(top1.item()),
        "top4_energy": float(top4.item()),
        "top8_energy": float(top8.item()),
        "dead_channel_fraction": float(dead_frac),
        "dead_channel_count": int(dead.sum().item()),
        "mean_channel_std": float(ch_std.mean().item()),
        "min_channel_std": float(ch_std.min().item()),
        "max_channel_std": float(ch_std.max().item()),
        "dead_channel_threshold": float(dead_thresh),
    }


def rank_dashboard(outputs: dict) -> dict:
    context = outputs.get("context_map")
    pred = outputs["pred_map"]
    gt = outputs["gt_map"]

    out = {}
    if context is not None:
        out["context"] = spectral_rank_stats(torch.as_tensor(context))
    out["pred"] = spectral_rank_stats(torch.as_tensor(pred))
    out["gt"] = spectral_rank_stats(torch.as_tensor(gt))
    out["rank_match_ratio"] = out["pred"]["erank"] / max(out["gt"]["erank"], 1e-12)
    out["volume_match_ratio"] = out["pred"]["manifold_size"] / max(
        out["gt"]["manifold_size"], 1e-12
    )
    return out


def compute_error_by_scale(outputs: dict) -> dict[float, float]:
    pred = outputs["pred_patches"].detach()  # B,K,C,P,P
    gt = outputs["gt_patches"].detach()  # B,K,C,P,P
    scales = outputs["target_scales"].detach()  # B,K
    valid = outputs["target_valid"].detach()  # B,K

    # Per-target MSE averaged over C,P,P
    reduce_dims = tuple(range(2, pred.dim()))
    mse_bk = torch.mean((pred - gt) ** 2, dim=reduce_dims)  # B,K
    out = defaultdict(list)
    b, k = mse_bk.shape
    for bi in range(b):
        for ki in range(k):
            if not bool(valid[bi, ki].item()):
                continue
            s = round(float(scales[bi, ki].item()), 6)
            out[s].append(float(mse_bk[bi, ki].item()))
    return {float(s): float(np.mean(v)) for s, v in out.items() if len(v) > 0}


================================================================================
FILE: src/__init__.py
================================================================================

"""sajepa — Scale-Aware Joint-Embedding Predictive Architecture for Physical Fields."""

from src.api import ScaleAwareJEPA
from src.utils.memory import OOMSafeTrainer, clear_memory_cache, compute_accumulation_steps

__all__ = ["ScaleAwareJEPA", "OOMSafeTrainer", "clear_memory_cache", "compute_accumulation_steps"]


================================================================================
FILE: src/inference_from_session.py
================================================================================

"""
Inference from a trained JEPA session.

Load a checkpoint from any `sessions/<name>/` directory, run the model on
arbitrary input data with optional crop/tile support for large arrays,
and save results into a new inference-only session.

Usage:
    python -m src.inference_from_session \\
        --session sessions/gen_121_mhd_run_006_ms1p2 \\
        --input data/some_large_file.npy \\
        --crop-size 256 \\
        --mode image \\
        --output-session sessions/inference_gen_121_run_006
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import JEPADataset
from src.dataset3d import JEPA3DCropDataset
from src.train import (
    _collate_pad_spatial,
    _collate_for_inference,
    build_model_from_config,
    build_model3d_from_config,
    load_config,
    make_session_dir,
    resolve_pipeline_config,
)
from src.inference import (
    run_post_training_inference,
    run_post_training_inference_3d,
    _save_npz,
    _forward_tta_streaming_2d,
)
from src.utils.npy import _safe_load_npy, normalize01
from src.utils.viz import save_inference_dashboard


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_session(session_dir: str, device: torch.device, *, strict_load: bool = True):
    """Reconstruct a PyramidGridJEPA model from a session directory.

    Returns (model, config, session_dir).
    The model is set to eval() and moved to *device*.
    """
    config_path = os.path.join(session_dir, "config_used.json")
    if not os.path.exists(config_path):
        # Fall back to the original config name from checkpoint
        ckpt_path = os.path.join(session_dir, "checkpoint_last.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            config_name = ckpt.get("config_name")
            if config_name:
                config_path = os.path.join("configs", f"{config_name}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Cannot locate config_used.json or infer original config name from {session_dir}"
        )

    config = load_config(config_path)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})

    is_3d_mode = str(model_cfg.get("mode", "image")).strip().lower() in ("3d_slab", "3d-slab")
    if is_3d_mode:
        model = build_model3d_from_config(model_cfg, train_cfg, device)
    else:
        model = build_model_from_config(model_cfg, data_cfg, train_cfg, device)

    model_ckpt = os.path.join(session_dir, "model_last.pt")
    if not os.path.exists(model_ckpt):
        raise FileNotFoundError(f"model_last.pt not found in {session_dir}")

    state = torch.load(model_ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=strict_load)
    if missing:
        print(f"[inference] load_model missing_keys={missing}")
    if unexpected:
        print(f"[inference] load_model unexpected_keys={unexpected}")

    model.to(device)
    model.eval()
    return model, config, session_dir


# ---------------------------------------------------------------------------
# Raw data loading (no JEPADataset dependencies)
# ---------------------------------------------------------------------------

def _tile_crops_2d(
    arr2d: np.ndarray,
    crop_size: int,
    crop_mode: str = "center",
) -> list[np.ndarray]:
    """Split a large 2D array into overlapping/non-overlapping tiles."""
    h, w = arr2d.shape
    cs = int(crop_size)
    if h <= cs and w <= cs:
        # No tiling needed — return the whole thing (possibly padded)
        if h < cs or w < cs:
            padded = np.zeros((cs, cs), dtype=np.float32)
            padded[:h, :w] = np.asarray(arr2d, dtype=np.float32)
            return [padded]
        return [np.asarray(arr2d, dtype=np.float32).copy()]

    # Center mode: extract single center crop directly, no tiling needed.
    if crop_mode == "center":
        y0 = max(0, (h - cs) // 2)
        x0 = max(0, (w - cs) // 2)
        tile = np.zeros((cs, cs), dtype=np.float32)
        th = min(cs, h - y0)
        tw = min(cs, w - x0)
        tile[:th, :tw] = np.asarray(arr2d[y0 : y0 + th, x0 : x0 + tw], dtype=np.float32)
        return [tile]

    tiles = []
    stride = max(1, cs // 2)  # 50% overlap
    for y0 in range(0, h, stride):
        y1 = min(y0 + cs, h)
        for x0 in range(0, w, stride):
            x1 = min(x0 + cs, w)
            tile = np.zeros((cs, cs), dtype=np.float32)
            th = y1 - y0
            tw = x1 - x0
            tile[:th, :tw] = np.asarray(arr2d[y0:y1, x0:x1], dtype=np.float32)
            tiles.append(tile)

    return tiles


@dataclass(frozen=True)
class TileLayout2D:
    original_shape: tuple[int, int]
    crop_size: int
    origins: tuple[tuple[int, int], ...]
    valid_shapes: tuple[tuple[int, int], ...]


def _tile_crops_2d_with_layout(
    arr2d: np.ndarray,
    crop_size: int,
    crop_mode: str = "center",
) -> tuple[list[np.ndarray], TileLayout2D | None]:
    h, w = arr2d.shape
    cs = int(crop_size)
    if h <= cs and w <= cs:
        if h < cs or w < cs:
            padded = np.zeros((cs, cs), dtype=np.float32)
            padded[:h, :w] = np.asarray(arr2d, dtype=np.float32)
            return [padded], TileLayout2D((h, w), cs, ((0, 0),), ((h, w),))
        return [np.asarray(arr2d, dtype=np.float32).copy()], None

    if crop_mode == "center":
        y0 = max(0, (h - cs) // 2)
        x0 = max(0, (w - cs) // 2)
        tile = np.zeros((cs, cs), dtype=np.float32)
        th = min(cs, h - y0)
        tw = min(cs, w - x0)
        tile[:th, :tw] = np.asarray(arr2d[y0 : y0 + th, x0 : x0 + tw], dtype=np.float32)
        return [tile], None

    tiles = []
    origins = []
    valid_shapes = []
    stride = max(1, cs // 2)
    for y0 in range(0, h, stride):
        y1 = min(y0 + cs, h)
        for x0 in range(0, w, stride):
            x1 = min(x0 + cs, w)
            tile = np.zeros((cs, cs), dtype=np.float32)
            th = y1 - y0
            tw = x1 - x0
            tile[:th, :tw] = np.asarray(arr2d[y0:y1, x0:x1], dtype=np.float32)
            tiles.append(tile)
            origins.append((y0, x0))
            valid_shapes.append((th, tw))
    return tiles, TileLayout2D((h, w), cs, tuple(origins), tuple(valid_shapes))


def _stitch_tile_tensor(value, layout: TileLayout2D | None):
    if value is None or layout is None:
        return value
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.dim() < 4 or int(tensor.shape[0]) != len(layout.origins):
        return value

    tile_h = int(tensor.shape[-2])
    tile_w = int(tensor.shape[-1])
    scale_y = float(tile_h) / float(layout.crop_size)
    scale_x = float(tile_w) / float(layout.crop_size)
    out_h = max(1, int(np.ceil(float(layout.original_shape[0]) * scale_y)))
    out_w = max(1, int(np.ceil(float(layout.original_shape[1]) * scale_x)))
    out_shape = (1, *tensor.shape[1:-2], out_h, out_w)
    # Accumulate on CPU to avoid GPU OOM on large images (e.g. 10k×10k)
    out = torch.zeros(out_shape, dtype=tensor.dtype, device="cpu")
    counts = torch.zeros((1, *([1] * (tensor.dim() - 3)), out_h, out_w), dtype=tensor.dtype, device="cpu")
    for idx, ((y0, x0), (th, tw)) in enumerate(zip(layout.origins, layout.valid_shapes)):
        oy0 = int(np.floor(float(y0) * scale_y))
        ox0 = int(np.floor(float(x0) * scale_x))
        oy1 = min(out_h, int(np.ceil(float(y0 + th) * scale_y)))
        ox1 = min(out_w, int(np.ceil(float(x0 + tw) * scale_x)))
        vh = max(0, oy1 - oy0)
        vw = max(0, ox1 - ox0)
        if vh <= 0 or vw <= 0:
            continue
        tile = tensor[idx : idx + 1, ..., :vh, :vw].cpu()
        out[..., oy0:oy1, ox0:ox1] += tile
        counts[..., oy0:oy1, ox0:ox1] += 1
    return out / counts.clamp_min(1)


def _make_depth_slabs(
    volume: np.ndarray,
    depth_size: int | None,
    slice_index: int | None = None,
) -> list[np.ndarray]:
    """Return D×H×W slabs from an already depth-first normalized volume."""
    if volume.ndim != 3:
        raise ValueError(f"Expected depth-first 3D volume, got shape {volume.shape}")
    depth = int(volume.shape[0])
    if depth_size is None:
        if slice_index is not None:
            idx = int(slice_index) % depth
            return [volume[idx : idx + 1]]
        return [volume]

    slab_depth = max(1, int(depth_size))
    centers = [int(slice_index) % depth] if slice_index is not None else list(range(depth))
    slabs: list[np.ndarray] = []
    for center in centers:
        start = center - slab_depth // 2
        end = start + slab_depth
        src0 = max(0, start)
        src1 = min(depth, end)
        dst0 = max(0, -start)
        slab = np.zeros((slab_depth, volume.shape[1], volume.shape[2]), dtype=np.float32)
        slab[dst0 : dst0 + (src1 - src0)] = volume[src0:src1]
        slabs.append(slab)
    return slabs


def _stitch_tiled_outputs(outputs: dict, layout: TileLayout2D | None) -> dict:
    if layout is None:
        return outputs
    stitched = dict(outputs)
    for key in ("pred_map", "gt_map", "context_map", "x_clean_raw", "x_context_raw"):
        stitched[key] = _stitch_tile_tensor(stitched.get(key), layout)
    stitched["tile_layout"] = {
        "original_shape": list(layout.original_shape),
        "crop_size": int(layout.crop_size),
        "num_tiles": len(layout.origins),
    }
    return stitched


def load_raw_data(
    input_path: str,
    crop_size: int | None = None,
    crop_mode: str = "center",
    mode: str = "image",
    slice_axis: int = 0,
    slice_index: int | None = None,
    return_layout: bool = False,
    slab_depth: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, TileLayout2D | None]:
    """Load an arbitrary .npy and return a B×1×H×W tensor (or B×1×D×H×W for 3D).

    Args:
        input_path: path to a .npy file (2D or 3D).
        crop_size: if set and data is larger, crop or tile.
        crop_mode: 'center' (single crop) or 'tile' (tiled crops across image).
        mode: 'image' (2D) or '3d_slab' (3D volume).
        slice_axis: for 3D mode, which axis to treat as depth (default 0).
        slice_index: for 3D mode, specific center slice index, or None for all slices.
        slab_depth: for 3D mode, depth window per sample. If None, use the full
            input volume as one sample.

    Returns:
        Tensor B×1×H×W for image mode, B×1×D×H×W for 3D slab mode.
    """
    arr = _safe_load_npy(input_path, mmap_mode="r")
    print(f"[inference] loaded {input_path} shape={arr.shape} dtype={arr.dtype}")

    mode_norm = str(mode).strip().lower()
    if mode_norm in ("3d_slab", "3d-slab"):
        if arr.ndim != 3:
            raise ValueError(f"3D slab mode requires 3D input, got shape {arr.shape}")
        axis = int(slice_axis) % 3
        volume = np.moveaxis(np.asarray(arr, dtype=np.float32), axis, 0)
        volume = normalize01(volume)
        slabs = _make_depth_slabs(volume, slab_depth, slice_index=slice_index)
        processed = []
        for slab in slabs:
            if crop_size and max(slab.shape[-2:]) > crop_size:
                if crop_mode == "tile":
                    raise ValueError("Tiled 3D slab inference is not supported yet; use center crop or smaller inputs.")
                h, w = slab.shape[-2:]
                cs = int(crop_size)
                y0 = max(0, (h - cs) // 2)
                x0 = max(0, (w - cs) // 2)
                cropped = np.zeros((slab.shape[0], cs, cs), dtype=np.float32)
                th = min(cs, h - y0)
                tw = min(cs, w - x0)
                cropped[:, :th, :tw] = slab[:, y0 : y0 + th, x0 : x0 + tw]
                slab = cropped
            processed.append(np.asarray(slab, dtype=np.float32))
        tensor = np.stack(processed, axis=0)  # B×D×H×W
        out = torch.from_numpy(tensor).unsqueeze(1)  # B×1×D×H×W
        return (out, None) if return_layout else out

    # 2D image mode
    if arr.ndim == 3:
        # Try to squeeze a leading singleton dim
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(
                f"Image mode requires 2D data (or squeezable 3D), got shape {arr.shape}. "
                "Use --mode 3d_slab for 3D volumes."
            )

    if crop_size and max(arr.shape) > crop_size:
        arr_norm = normalize01(np.asarray(arr, dtype=np.float32))
        tiles, layout = _tile_crops_2d_with_layout(arr_norm, crop_size, crop_mode)
        tensor = np.stack([np.asarray(t, dtype=np.float32) for t in tiles], axis=0)
    else:
        layout = None
        tensor = normalize01(np.asarray(arr, dtype=np.float32))[np.newaxis, ...]  # 1×H×W

    out = torch.from_numpy(tensor).unsqueeze(1)  # B×1×H×W
    return (out, layout) if return_layout else out


# ---------------------------------------------------------------------------
# CDD pyramid helper (simplified for inference)
# ---------------------------------------------------------------------------

def _build_cdd_pyramid(
    x: torch.Tensor,
    model_cfg: dict,
    data_cfg: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a CDD pyramid from a raw image tensor using the model config.

    Returns (cdd_fields, x_clean) where cdd_fields is B×S×H×W.
    For pure inference this is essentially computing CDD decomposition
    using the same pipeline as training.
    """
    from src.cdd import constrained_diffusion_decomposition

    # x: B×1×H×W → list of B×S×H×W
    sigmas = model_cfg.get("sigmas", [2, 4, 8, 16])
    cdd_mode = str(data_cfg.get("cdd_mode", model_cfg.get("cdd_mode", "log")))
    cdd_constrained = bool(model_cfg.get("cdd_constrained", True))
    cdd_sm_mode = str(data_cfg.get("cdd_sm_mode", model_cfg.get("cdd_sm_mode", "reflect")))

    bsz = x.shape[0]
    x_np = x.squeeze(1).detach().cpu().numpy().astype(np.float32)

    cdd_list = []
    for i in range(bsz):
        cdd_result = constrained_diffusion_decomposition(
            x_np[i],
            scales=list(sigmas),
            mode=cdd_mode,
            constrained=cdd_constrained,
            sm_mode=cdd_sm_mode,
        )
        # cdd_result should be S×H×W
        cdd_t = torch.from_numpy(np.asarray(cdd_result, dtype=np.float32)).to(device)
        if cdd_t.ndim == 3:
            cdd_list.append(cdd_t.unsqueeze(0))
        else:
            cdd_list.append(cdd_t)

    cdd_fields = torch.cat(cdd_list, dim=0)  # B×S×H×W
    x_clean = x.clone()  # Keep original for downstream

    return cdd_fields, x_clean


# ---------------------------------------------------------------------------
# Core inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_on_data(
    model,
    data_loader: DataLoader,
    device: torch.device,
    mask_inference: bool = True,
    inference_tta_enabled: bool = False,
    inference_tta_mode: str = "flip4",
) -> dict:
    """Run inference over a DataLoader, collecting output maps.

    Returns a dict compatible with save_inference_dashboard / downstream analysis.
    """
    model.eval()

    pred_maps = []
    gt_maps = []
    context_maps = []
    x_clean_list = []
    x_context_list = []
    all_target_locs = []
    all_target_scales = []
    all_target_valid = []

    for batch in data_loader:
        if isinstance(batch, (tuple, list)):
            cdd_batch, x_batch = batch
        else:
            cdd_batch = batch
            x_batch = None

        cdd_batch = cdd_batch.to(device, non_blocking=True)
        if x_batch is not None:
            x_batch = x_batch.to(device, non_blocking=True)

        x_in = x_batch if x_batch is not None else cdd_batch
        cdd_fields = cdd_batch if (x_batch is not None and x_batch.dim() == 4 and model.mode == "pyramid") else None

        def _forward_one(xv: torch.Tensor, cdv: torch.Tensor | None):
            if cdv is not None and model.mode == "pyramid":
                return model(xv, mask_inference=bool(mask_inference), context_data=None, cdd_orig=cdv)
            return model(xv, mask_inference=bool(mask_inference))

        if inference_tta_enabled:
            out, _ = _forward_tta_streaming_2d(
                x=x_in,
                mode=inference_tta_mode,
                forward_one=_forward_one,
                cdd=cdd_fields,
            )
        else:
            out = _forward_one(x_in, cdd_fields)

        pred_map = out.get("pred_map")
        gt_map = out.get("gt_map")
        context_map = out.get("context_map")

        if pred_map is not None:
            pred_maps.append(pred_map.cpu())
        if gt_map is not None:
            gt_maps.append(gt_map.cpu())
        if context_map is not None:
            context_maps.append(context_map.cpu())
        if out.get("x_clean_raw") is not None:
            x_clean_list.append(out["x_clean_raw"].cpu())
        elif x_batch is not None:
            x_clean_list.append(x_batch.cpu())
        if out.get("x_context_raw") is not None:
            x_context_list.append(out["x_context_raw"].cpu())
        else:
            x_context_list.append(cdd_batch.cpu())
        if out.get("target_locations") is not None:
            all_target_locs.append(out["target_locations"].cpu())
        if out.get("target_scales") is not None:
            all_target_scales.append(out["target_scales"].cpu())
        if out.get("target_valid") is not None:
            all_target_valid.append(out["target_valid"].cpu())

    # Stack and mean across batches
    def _stack_or_none(lst):
        if not lst:
            return None
        return torch.cat(lst, dim=0)

    outputs = {
        "pred_map": _stack_or_none(pred_maps),
        "gt_map": _stack_or_none(gt_maps),
        "context_map": _stack_or_none(context_maps),
        "x_clean_raw": _stack_or_none(x_clean_list) if x_clean_list else None,
        "x_context_raw": _stack_or_none(x_context_list) if x_context_list else None,
        "target_locations": _stack_or_none(all_target_locs),
        "target_scales": _stack_or_none(all_target_scales),
        "target_valid": _stack_or_none(all_target_valid),
    }
    return outputs


# ---------------------------------------------------------------------------
# Session output saving
# ---------------------------------------------------------------------------

def save_inference_session(
    outputs: dict,
    output_dir: str,
    config: dict,
    input_path: str,
    crop_size: int | None = None,
    mode: str = "image",
    mask_inference: bool = True,
) -> str:
    """Save inference outputs as a new inference-only session.

    Returns the output_dir path.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Mark as inference-only
    config_out = dict(config)
    config_out["_inference"] = {
        "inference_only": True,
        "source_session": config_out.pop("_source_session", None),
        "input_file": os.path.abspath(input_path),
        "crop_size": crop_size,
        "mode": mode,
        "mask_inference": bool(mask_inference),
    }

    with open(os.path.join(output_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2)

    # Save raw tensors
    torch.save(outputs, os.path.join(output_dir, "inference_outputs.pt"))

    # Save compressed NPZ maps and target metadata.
    for key in ("pred_map", "gt_map", "context_map", "target_locations", "target_scales", "target_valid"):
        val = outputs.get(key)
        if val is not None:
            _save_npz(os.path.join(output_dir, f"{key}.npz"), val.cpu().numpy() if hasattr(val, "cpu") else val)

    # Save network inputs
    for key in ("x_clean_raw", "x_context_raw"):
        val = outputs.get(key)
        if val is not None:
            _save_npz(
                os.path.join(output_dir, f"network_input_{'clean' if 'clean' in key else 'context'}.npz"),
                val.cpu().numpy() if hasattr(val, "cpu") else val,
            )

    # JEPA energy summary
    energy_summary = {
        "inference_only": True,
        "input_file": os.path.abspath(input_path),
        "crop_size": crop_size,
        "mode": mode,
        "mask_inference": bool(mask_inference),
        "pred_map_shape": list(outputs["pred_map"].shape) if outputs.get("pred_map") is not None else None,
        "gt_map_shape": list(outputs["gt_map"].shape) if outputs.get("gt_map") is not None else None,
        "target_locations_shape": list(outputs["target_locations"].shape) if outputs.get("target_locations") is not None else None,
    }
    if outputs.get("tile_layout") is not None:
        energy_summary["tile_layout"] = outputs["tile_layout"]
    with open(os.path.join(output_dir, "jepa_energy_summary.json"), "w", encoding="utf-8") as f:
        json.dump(energy_summary, f, indent=2)

    # Dashboard data
    try:
        artifacts_dir = save_inference_dashboard(output_dir, outputs, umap_cfg={})
        print(f"[inference] dashboard_saved={artifacts_dir}")
    except Exception as e:
        print(f"[inference] dashboard generation failed (non-fatal): {e}")

    print(f"[inference] session saved to {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_args(args, config_dict: dict | None) -> argparse.Namespace:
    """Merge config file values with CLI overrides. CLI wins where both are set."""
    if config_dict is None:
        return args

    # Config file keys → argparse attribute names
    key_map = {
        "session": "session",
        "input": "input",
        "crop_size": "crop_size",
        "crop_mode": "crop_mode",
        "mode": "mode",
        "mask_inference": "mask_inference",
        "slice_axis": "slice_axis",
        "slice_index": "slice_index",
        "output_session": "output_session",
        "batch_size": "batch_size",
        "tta": "tta",
        "tta_mode": "tta_mode",
        "device": "device",
        "allow_partial_load": "allow_partial_load",
    }

    cli_defaults = {
        "session": None,
        "input": None,
        "crop_size": None,
        "crop_mode": "center",
        "mode": "image",
        "mask_inference": True,
        "slice_axis": 0,
        "slice_index": None,
        "output_session": None,
        "batch_size": 2,
        "tta": False,
        "tta_mode": "flip4",
        "device": None,
        "allow_partial_load": False,
    }

    for config_key, attr_name in key_map.items():
        cli_val = getattr(args, attr_name, None)
        default_val = cli_defaults.get(attr_name)
        config_val = config_dict.get(config_key)

        # CLI value takes precedence if it differs from the default (i.e. user explicitly set it)
        if cli_val is not None and cli_val != default_val:
            continue
        if config_val is not None:
            setattr(args, attr_name, config_val)

    return args


def main():
    parser = argparse.ArgumentParser(
        description="Run inference from a trained JEPA session on arbitrary data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From a config file
  python -m src.inference_from_session --config configs/inference_chengdu.json

  # Config file with CLI overrides
  python -m src.inference_from_session --config configs/inference_chengdu.json --crop-size 128

  # 2D image inference from CLI flags
  python -m src.inference_from_session \\
      --session sessions/gen_121_mhd_run_006_ms1p2 \\
      --input data/chengdu.npy \\
      --crop-size 256 \\
      --output-session sessions/inference_chengdu

  # 3D slab mode — processes all depth slices
  python -m src.inference_from_session \\
      --session sessions/gen_121_mhd_run_006_ms1p2 \\
      --input data/ngc3627_mom0.npy \\
      --mode 3d_slab \\
      --slice-axis 0 \\
      --output-session sessions/inference_ngc_3d

  # Tiled inference for very large images
  python -m src.inference_from_session \\
      --session sessions/gen_121_mhd_run_006_ms1p2 \\
      --input data/huge_mosaic.npy \\
      --crop-size 256 \\
      --crop-mode tile \\
      --output-session sessions/inference_mosaic
        """,
    )
    parser.add_argument("--config", default=None, help="Path to inference config JSON")
    parser.add_argument("--session", default=None, help="Path to trained session directory")
    parser.add_argument("--input", default=None, help="Path to input .npy file")
    parser.add_argument("--crop-size", type=int, default=None, help="Crop/tile size for large inputs")
    parser.add_argument("--crop-mode", default="center", choices=["center", "tile"], help="Crop mode")
    parser.add_argument("--mode", default="image", choices=["image", "3d_slab"], help="Inference mode")
    parser.add_argument(
        "--no-mask-inference",
        dest="mask_inference",
        action="store_false",
        help="Use clean context features for representation export instead of masked prediction evaluation.",
    )
    parser.set_defaults(mask_inference=True)
    parser.add_argument("--slice-axis", type=int, default=0, help="Depth axis for 3D slab mode")
    parser.add_argument("--slice-index", type=int, default=None, help="Specific slice index for 3D mode")
    parser.add_argument("--output-session", default=None, help="Output session directory")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for inference")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--tta-mode", default="flip4", choices=["flip4", "rot4", "d4"], help="TTA view set")
    parser.add_argument("--device", default=None, help="Override device (cuda, mps, cpu)")
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys instead of failing strict model loading.",
    )
    args = parser.parse_args()

    # Load config file if provided, merge CLI overrides
    config_dict = None
    if args.config:
        if not os.path.exists(args.config):
            print(f"[inference] ERROR: config file not found: {args.config}")
            sys.exit(1)
        with open(args.config, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        print(f"[inference] loaded config from {args.config}")
        args = _resolve_args(args, config_dict)

    if not args.session:
        parser.error("--session is required (via CLI or config file)")
    if not args.input:
        parser.error("--input is required (via CLI or config file)")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[inference] device={device}")

    # Load model
    model, config, source_session = load_model_from_session(
        args.session,
        device,
        strict_load=not bool(args.allow_partial_load),
    )
    config["_source_session"] = source_session
    print(f"[inference] model loaded from {source_session}")

    # Load data
    data_tensor, tile_layout = load_raw_data(
        args.input,
        crop_size=args.crop_size,
        crop_mode=args.crop_mode,
        mode=args.mode,
        slice_axis=args.slice_axis,
        slice_index=args.slice_index,
        return_layout=True,
        slab_depth=(
            getattr(model, "required_input_depth", None)
            if str(args.mode).strip().lower() in ("3d_slab", "3d-slab")
            else None
        ),
    )
    print(f"[inference] data shape={tuple(data_tensor.shape)}")
    if tile_layout is not None:
        print(
            f"[inference] tiled input will be stitched: original_shape={tile_layout.original_shape} "
            f"tiles={len(tile_layout.origins)} crop_size={tile_layout.crop_size}"
        )

    # Build a simple DataLoader
    class _TensorDataset(torch.utils.data.Dataset):
        def __init__(self, t):
            self.t = t

        def __len__(self):
            return self.t.shape[0]

        def __getitem__(self, idx):
            return self.t[idx]

    dataset = _TensorDataset(data_tensor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_pad_spatial,
    )

    # Run inference
    print(f"[inference] running forward pass on {len(dataset)} samples...")
    outputs = run_inference_on_data(
        model,
        loader,
        device,
        mask_inference=args.mask_inference,
        inference_tta_enabled=args.tta,
        inference_tta_mode=args.tta_mode,
    )
    outputs = _stitch_tiled_outputs(outputs, tile_layout)
    print(f"[inference] pred_map shape={tuple(outputs['pred_map'].shape) if outputs.get('pred_map') is not None else None}")

    # Save
    output_dir = args.output_session or f"sessions/inference_{os.path.basename(args.session)}_{os.path.basename(args.input).replace('.npy', '')}"
    save_inference_session(
        outputs,
        output_dir,
        config,
        args.input,
        crop_size=args.crop_size,
        mode=args.mode,
        mask_inference=args.mask_inference,
    )

    print(f"[inference] done → {output_dir}")


if __name__ == "__main__":
    main()


================================================================================
FILE: src/dataset.py
================================================================================

from __future__ import annotations
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.npy import _safe_load_npy, normalize01

try:
    import h5py  # optional — enables chunked HDF5 reading for large 3D arrays
except ImportError:
    h5py = None

try:
    from astropy.io import fits  # optional — FITS file support
except ImportError:
    fits = None


class JEPADataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1000,
        data_root: str = "data",
        npy_pattern: str = "*.npy",
        cube_slice_strategy: str = "auto",
        cube_slice_axis: int = 0,
        cube_slice_index: int = 0,
        crop_mode: str = "none",
        crop_size: int | tuple[int, int] | list[int] | None = None,
        random_roll_max: int = 0,
        d4_augment: bool = False,
        input_type: str = "image",
        image_batch_inference: bool = False,
        image_batch_selected_indices: dict | None = None,
        cdd_cache: dict | None = None,
        crop_min_valid_fraction: float = 0.0,
    ):
        self.input_type = str(input_type).lower()
        allowed_input_types = {"image", "cube", "image_batch"}
        if self.input_type not in allowed_input_types:
            raise ValueError(
                f"Unknown input_type={input_type}. "
                "Use 'image', 'cube', or 'image_batch'."
            )
        self.image_batch_inference = bool(image_batch_inference)
        self.image_batch_selected_indices = image_batch_selected_indices
        self.cube_slice_strategy = str(cube_slice_strategy).lower()
        allowed_strategies = {"auto", "random", "center", "fixed", "all"}
        if self.cube_slice_strategy not in allowed_strategies:
            raise ValueError(
                f"Unknown cube_slice_strategy={cube_slice_strategy}. "
                "Use 'auto', 'random', 'center', 'fixed', or 'all'."
            )
        self.num_samples = num_samples
        self.cube_slice_axis = cube_slice_axis
        self.cube_slice_index = cube_slice_index
        self.crop_mode = str(crop_mode).lower()
        if self.crop_mode not in {"none", "random", "center"}:
            raise ValueError("crop_mode must be one of: none, random, center")
        self.crop_size = self._coerce_crop_size(crop_size)
        if self.crop_mode != "none" and self.crop_size is None:
            raise ValueError("crop_size is required when crop_mode is not 'none'")
        self.random_roll_max = int(random_roll_max)
        self.d4_augment = bool(d4_augment)
        self.cdd_cache = cdd_cache or None
        self.crop_min_valid_fraction = float(crop_min_valid_fraction) if crop_min_valid_fraction is not None else 0.0

        pattern = os.path.join(data_root, npy_pattern)
        self.pattern = pattern
        self.npy_files = sorted(glob.glob(pattern))

        # Also scan for .h5 files (preferred for fast random-access slicing)
        h5_pattern = pattern.replace(".npy", ".h5") if pattern.endswith(".npy") else os.path.join(data_root, "*.h5")
        self.h5_files = sorted(glob.glob(h5_pattern)) if h5py is not None else []
        if self.h5_files:
            print(f"[dataset] Found {len(self.h5_files)} .h5 file(s); using chunked HDF5 for fast I/O")

        self.fits_files = []
        if not self.npy_files and not self.h5_files:
            raise FileNotFoundError(f"No .npy, .h5, or .fits files found with pattern: {pattern}")
        self.sample_index = self._build_sample_index()
        if self.num_samples is None:
            self.num_samples = len(self.sample_index)
    def _preprocess_arr2d(self, arr2d: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr2d, dtype=np.float32)
        finite = np.isfinite(arr)
        out = np.zeros_like(arr, dtype=np.float32)
        if not bool(finite.any()):
            return out
        finite_vals = arr[finite]
        amin = float(finite_vals.min())
        amax = float(finite_vals.max())
        denom = amax - amin
        if denom > 1e-20:
            out[finite] = (arr[finite] - amin) / denom
        return out

    @staticmethod
    def _probe_file_shape(path: str) -> tuple[int, ...]:
        """Read array shape without loading full data. Supports .npy and .h5."""
        if path.endswith(".h5"):
            if h5py is None:
                raise ImportError("h5py is required to read .h5 files; pip install h5py")
            with h5py.File(path, "r") as f:
                return tuple(f["data"].shape)
        if path.endswith(".fits"):
            if fits is None:
                raise ImportError("astropy is required to read .fits files; pip install astropy")
            return fits.getdata(path, memmap=True).shape
        arr = _safe_load_npy(path, mmap_mode="r")
        return arr.shape

    @staticmethod
    def _is_h5(path: str) -> bool:
        return path.endswith(".h5")

    @staticmethod
    def _is_fits(path: str) -> bool:
        return path.endswith(".fits")

    def _build_sample_index(self):
        # Also scan for .fits files
        fits_pattern = self.pattern.replace(".npy", ".fits") if self.pattern.endswith(".npy") else os.path.join(os.path.dirname(self.pattern), "*.fits")
        self.fits_files = sorted(glob.glob(fits_pattern)) if fits is not None else []
        if self.fits_files:
            print(f"[dataset] Found {len(self.fits_files)} .fits file(s)")

        all_files = list(self.npy_files) + list(self.h5_files) + list(self.fits_files)
        index = []
        for path in all_files:
            shape = self._probe_file_shape(path)
            ndim = len(shape)
            if ndim == 2:
                index.append((path, None))
            elif ndim == 3:
                if self.input_type == "image_batch":
                    if self.image_batch_selected_indices is not None and path in self.image_batch_selected_indices:
                        sel = self.image_batch_selected_indices[path]
                        for sidx in sel:
                            index.append((path, int(sidx)))
                    elif self.image_batch_inference:
                        index.append((path, 0))
                    else:
                        index.append((path, None))
                else:
                    axis = self.cube_slice_axis % 3
                    depth = shape[axis]
                    if self.cube_slice_strategy == "all":
                        for sidx in range(depth):
                            index.append((path, sidx))
                    else:
                        index.append((path, None))
            else:
                raise ValueError(f"Expected 2D or 3D array in {path}, got shape {shape}")
        if not index:
            raise ValueError("No usable samples found from npy files.")
        return index

    @property
    def rng(self):
        """Lazily initialize an isolated generator per DataLoader worker."""
        if not hasattr(self, "_rng") or self._rng is None:
            import torch.utils.data

            worker_info = torch.utils.data.get_worker_info()
            seed = worker_info.seed % (2**31 - 1) if worker_info is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())
            self._rng = np.random.default_rng(seed)
        return self._rng

    def _pick_slice_index(self, depth: int) -> int:
        strategy = self.cube_slice_strategy
        if strategy == "auto":
            strategy = "random"
        if strategy == "random":
            return int(self.rng.integers(0, depth))
        if strategy == "center":
            return depth // 2
        if strategy == "fixed":
            return int(np.clip(self.cube_slice_index, 0, depth - 1))
        raise ValueError(
            f"Unknown cube_slice_strategy={strategy}. "
            "Use 'auto', 'random', 'center', 'fixed', or 'all'."
        )

    def _extract_2d_from_array(self, arr: np.ndarray, forced_slice_idx=None) -> tuple[np.ndarray, int | None]:
        if arr.ndim == 2:
            return arr, None
        if self.input_type == "image_batch":
            depth = arr.shape[0]
            if self.image_batch_inference:
                sidx = 0
            elif forced_slice_idx is not None:
                sidx = forced_slice_idx
            else:
                sidx = int(self.rng.integers(0, depth))
            sidx = int(np.clip(sidx, 0, depth - 1))
            return arr[sidx], int(sidx)
        axis = self.cube_slice_axis % 3
        depth = arr.shape[axis]
        sidx = forced_slice_idx
        if sidx is None:
            sidx = self._pick_slice_index(depth)
        slicer = [slice(None), slice(None), slice(None)]
        slicer[axis] = int(np.clip(sidx, 0, depth - 1))
        return arr[tuple(slicer)], int(sidx)

    def _extract_2d_from_cdd(self, cdd: np.ndarray, forced_slice_idx=None) -> np.ndarray:
        if cdd.ndim == 3:
            return cdd
        if cdd.ndim != 4:
            raise ValueError(f"Expected cached CDD shape (S,H,W) or (S,D,H,W), got {cdd.shape}")
        if self.input_type == "image_batch":
            axis = 0
            depth = cdd.shape[axis + 1]
            if self.image_batch_inference:
                sidx = 0
            elif forced_slice_idx is not None:
                sidx = forced_slice_idx
            else:
                sidx = int(self.rng.integers(0, depth))
        else:
            axis = self.cube_slice_axis % 3
            depth = cdd.shape[axis + 1]
            sidx = forced_slice_idx
            if sidx is None:
                sidx = self._pick_slice_index(depth)
        slicer = [slice(None), slice(None), slice(None), slice(None)]
        slicer[axis + 1] = int(np.clip(sidx, 0, depth - 1))
        return cdd[tuple(slicer)]

    def _load_sample(self, path: str, forced_slice_idx=None) -> torch.Tensor:
        if self._is_h5(path):
            if h5py is None:
                raise ImportError("h5py is required to read .h5 files; pip install h5py")
            with h5py.File(path, "r") as h5_file:
                ds = h5_file["data"]
                arr2d, _ = self._extract_2d_from_array(ds, forced_slice_idx=forced_slice_idx)
                arr2d = np.asarray(arr2d, dtype=np.float32)
        elif self._is_fits(path):
            if fits is None:
                raise ImportError("astropy is required to read .fits files; pip install astropy")
            arr_mm = fits.getdata(path, memmap=True)
            arr2d, _ = self._extract_2d_from_array(arr_mm, forced_slice_idx=forced_slice_idx)
        else:
            arr_mm = _safe_load_npy(path, mmap_mode="r")
            arr2d, _ = self._extract_2d_from_array(arr_mm, forced_slice_idx=forced_slice_idx)
        arr = self._preprocess_arr2d(arr2d)

        # Keep native resolution (including non-square fields).
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)  # 1 x H x W

    @staticmethod
    def _coerce_crop_size(crop_size) -> tuple[int, int] | None:
        if crop_size is None:
            return None
        if isinstance(crop_size, (list, tuple)):
            if len(crop_size) != 2:
                raise ValueError(f"crop_size must be an int or [height, width], got {crop_size!r}")
            crop_h, crop_w = int(crop_size[0]), int(crop_size[1])
        else:
            crop_size_int = int(crop_size)
            if crop_size_int <= 0:
                raise ValueError(f"crop_size must be positive, got {crop_size!r}")
            return crop_size_int, crop_size_int
        if crop_h <= 0 or crop_w <= 0:
            raise ValueError(f"crop_size must be positive, got {crop_size!r}")
        return crop_h, crop_w

    def _crop_slices(self, h: int, w: int) -> tuple[slice, slice] | None:
        if self.crop_mode == "none" or self.crop_size is None:
            return None
        crop_h, crop_w = self.crop_size
        if crop_h > h or crop_w > w:
            raise ValueError(f"crop_size={self.crop_size} exceeds image shape={(h, w)}")
        if self.crop_mode == "center":
            y0 = (h - crop_h) // 2
            x0 = (w - crop_w) // 2
        else:
            # Crop origin stays inside the margin implied by the crop size.
            y0 = int(self.rng.integers(0, h - crop_h + 1))
            x0 = int(self.rng.integers(0, w - crop_w + 1))
        return slice(y0, y0 + crop_h), slice(x0, x0 + crop_w)

    def _crop_tensor(self, x: torch.Tensor) -> torch.Tensor:
        crop = self._crop_slices(int(x.shape[-2]), int(x.shape[-1]))
        if crop is None:
            return x
        crop_y, crop_x = crop
        return x[..., crop_y, crop_x]

    @staticmethod
    def _normalize01(arr: np.ndarray) -> np.ndarray:
        return normalize01(arr)

    def _apply_augmentations(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply d4 flips + random_roll to all tensors identically. Shared by both data paths."""
        if not tensors:
            return tensors
        if self.d4_augment:
            h, w = tensors[0].shape[-2], tensors[0].shape[-1]
            if h == w:
                k = int(self.rng.integers(0, 4))
                if k:
                    tensors = tuple(torch.rot90(t, k=k, dims=(-2, -1)) for t in tensors)
                if bool(self.rng.integers(0, 2)):
                    tensors = tuple(torch.flip(t, dims=(-1,)) for t in tensors)
            elif bool(self.rng.integers(0, 2)):
                tensors = tuple(torch.flip(t, dims=(-2,)) for t in tensors)
            if h != w and bool(self.rng.integers(0, 2)):
                tensors = tuple(torch.flip(t, dims=(-1,)) for t in tensors)
        if self.random_roll_max > 0:
            h, w = tensors[0].shape[-2], tensors[0].shape[-1]
            pad_val = int(min(self.random_roll_max, max(0, h - 1), max(0, w - 1)))
            if pad_val <= 0:
                return tensors
            dy = int(self.rng.integers(-pad_val, pad_val + 1))
            dx = int(self.rng.integers(-pad_val, pad_val + 1))
            padded = tuple(
                torch.nn.functional.pad(t, (pad_val, pad_val, pad_val, pad_val), mode='reflect')
                for t in tensors
            )
            y0, x0 = pad_val - dy, pad_val - dx
            tensors = tuple(p[..., y0:y0 + h, x0:x0 + w] for p in padded)
        return tensors

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        key = self.sample_index[idx % len(self.sample_index)]

        if self.cdd_cache is not None:
            path, forced_slice_idx = key
            # New caches retain full cubes at (path, None). Fall back to an
            # exact key so existing 2D/per-slice caches remain usable.
            cdd_np = self.cdd_cache.get((path, None))
            if cdd_np is None:
                cdd_np = self.cdd_cache.get(key)
            if cdd_np is None:
                raise KeyError(f"CDD cache miss for key={key}")
            cdd_np = self._extract_2d_from_cdd(np.asarray(cdd_np), forced_slice_idx=forced_slice_idx)
            # cdd_np is now (S, H, W) float32
            cdd_orig = torch.from_numpy(cdd_np.astype(np.float32))
            x_clean_full = cdd_orig.sum(dim=0, keepdim=True)  # 1 x H x W
            max_retries = 100
            for attempt in range(max_retries):
                crop = self._crop_slices(int(cdd_orig.shape[-2]), int(cdd_orig.shape[-1]))
                if crop is not None:
                    crop_y, crop_x = crop
                    cdd_cropped = cdd_orig[..., crop_y, crop_x]
                    x_clean = x_clean_full[..., crop_y, crop_x]
                else:
                    cdd_cropped = cdd_orig
                    x_clean = x_clean_full
                if self.crop_min_valid_fraction > 0.0 and self.crop_mode == "random" and crop is not None:
                    arr = x_clean.squeeze(0).numpy()
                    finite_nonzero = np.isfinite(arr) & (arr > 1e-8)
                    if finite_nonzero.mean() >= self.crop_min_valid_fraction:
                        break
                else:
                    break
                if attempt == max_retries - 1:
                    break
            cdd_orig, x_clean = self._apply_augmentations(cdd_cropped, x_clean)
            return cdd_orig, x_clean

        path, forced_slice_idx = key
        max_retries = 100
        for attempt in range(max_retries):
            sample = self._load_sample(path, forced_slice_idx=forced_slice_idx).clone()  # 1 x H x W
            sample = self._crop_tensor(sample)
            if self.crop_min_valid_fraction > 0.0 and self.crop_mode == "random":
                arr = sample.squeeze(0).numpy()
                finite_nonzero = np.isfinite(arr) & (arr > 1e-8)
                if finite_nonzero.mean() >= self.crop_min_valid_fraction:
                    break
            else:
                break
            if attempt == max_retries - 1:
                break  # accept anyway after max retries
        (sample,) = self._apply_augmentations(sample)
        return sample


================================================================================
FILE: src/api.py
================================================================================

"""Public API for sajepa — Scale-Aware JEPA pipeline."""

from __future__ import annotations

import copy
import os
import tempfile
from typing import Optional

import numpy as np
import torch
import yaml

from src.train import load_config, run_training
from src.utils.memory import OOMSafeTrainer, clear_memory_cache


class ScaleAwareJEPA:
    """Scale-Aware Joint-Embedding Predictive Architecture for physical fields.

    Usage:
        model = ScaleAwareJEPA(config="configs/mhd_turbulence.yaml")
        model.fit(field, epochs=10)
        latent = model.extract(field)
        model.save_session("sessions/my_run")
        # Later:
        model2 = ScaleAwareJEPA.load_session("sessions/my_run")
        latent2 = model2.extract(new_field)
    """

    def __init__(self, config: Optional[dict | str] = None):
        self._config = self._parse_config(config)
        self._session_dir: Optional[str] = None
        self._is_trained: bool = False

    # ── training ────────────────────────────────────────────────

    def fit(self, field: torch.Tensor, epochs: Optional[int] = None) -> "ScaleAwareJEPA":
        """Train on a raw physical field.  Returns self for chaining."""
        cfg = copy.deepcopy(self._config)
        if epochs is not None:
            cfg.setdefault("training", cfg.setdefault("train", {}))["epochs"] = int(epochs)

        sessions_dir = tempfile.mkdtemp(prefix="sajepa_")
        data_dir = os.path.join(sessions_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, "_input.npy")
        arr = field.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        np.save(data_path, arr)
        cfg.setdefault("data", {})["npy_pattern"] = "_input.npy"

        # --- auto-batch OOM handling ---
        train_cfg = cfg.setdefault("train", cfg.setdefault("training", {}))
        trainer = OOMSafeTrainer(
            initial_batch=int(train_cfg.get("batch_size", 4)),
            target_batch=int(train_cfg.get("target_batch_size", train_cfg.get("target_batch", 32))),
            scale_mode=str(train_cfg.get("auto_scale_batch_size", "power_of_two")),
            max_retries=int(train_cfg.get("oom_max_retries", 5)),
        )
        train_cfg["batch_size"] = trainer.batch_size
        train_cfg["gradient_accumulation_steps"] = trainer.accumulation_steps

        if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if not torch.cuda.is_available():
            train_cfg["num_workers"] = 0

        print(f"[sajepa] batch={trainer.batch_size} accum={trainer.accumulation_steps}")

        old_cwd = os.getcwd()
        os.chdir(sessions_dir)
        session_dir = None
        try:
            while True:
                try:
                    session_dir = run_training(cfg, config_name="sajepa", sessions_root=".")
                    break
                except RuntimeError as e:
                    if not trainer.handle_oom(e):
                        raise
                    train_cfg["batch_size"] = trainer.batch_size
                    train_cfg["gradient_accumulation_steps"] = trainer.accumulation_steps
                    clear_memory_cache()
        finally:
            self._session_dir = os.path.abspath(session_dir) if session_dir else None
            os.chdir(old_cwd)

        if self._session_dir is None:
            raise RuntimeError("Training failed after all OOM retries.")
        self._is_trained = True
        _print_metrics_summary(self._session_dir)
        return self

    # ── inference ───────────────────────────────────────────────

    @torch.no_grad()
    def project(self, field: torch.Tensor, method: str = "umap") -> dict:
        """Extract latents and project to 2D with fallback.

        Always returns a dict with at least ``"pca"``; ``"umap"`` may be None
        if GPU/torchdr is unavailable or fails.
        """
        latent = self.extract(field)
        C, H, W = latent.shape
        flat = latent.permute(1, 2, 0).reshape(H * W, C).float()

        results: dict[str, Optional[np.ndarray]] = {"pca": None, "umap": None}

        # PCA — always works, always runs
        try:
            _, _, V = torch.pca_lowrank(flat, q=min(2, C))
            results["pca"] = torch.matmul(flat, V[:, :2]).cpu().numpy()
        except Exception as e:
            print(f"[sajepa] PCA fallback failed: {e}")

        # UMAP — best-effort
        if method.lower() == "umap":
            try:
                import torchdr
                reducer = torchdr.UMAP(
                    n_neighbors=self._config.get("diagnostics", {}).get("umap", {}).get("n_neighbors", 50),
                    min_dist=self._config.get("diagnostics", {}).get("umap", {}).get("min_dist", 0.2),
                    device=str(self._get_device()),
                )
                emb = reducer.fit_transform(flat.to(self._get_device()))
                results["umap"] = emb.cpu().numpy()
            except Exception as e:
                print(f"[sajepa] UMAP unavailable ({type(e).__name__}), PCA only.")

        return results

    @torch.no_grad()
    def extract(self, field: torch.Tensor) -> torch.Tensor:
        """Return pixel-registered latent atlas for the given field.  No training."""
        if self._session_dir is None:
            raise RuntimeError("No session available. Call fit() first or load_session().")
        inf_path = os.path.join(self._session_dir, "inference_outputs.pt")
        if not os.path.exists(inf_path):
            # Run inference if not already done
            if not self._is_trained:
                raise RuntimeError("Model not trained. Call fit() before extract().")
            raise RuntimeError("No inference outputs found — call fit() first.")
        outputs = torch.load(inf_path, map_location="cpu", weights_only=False)
        ctx = outputs.get("context_map")
        if ctx is None:
            ctx = outputs.get("pred_map")
        if ctx is None:
            raise RuntimeError("No latent map in inference outputs.")
        return ctx.squeeze(0).cpu()

    # ── persistence ─────────────────────────────────────────────

    def save_session(self, path: str):
        """Save model weights, config, and session artifacts to *path*."""
        if self._session_dir is None:
            raise RuntimeError("No session to save. Call fit() first.")
        import shutil
        os.makedirs(path, exist_ok=True)
        for name in ("config_used.json", "model_last.pt", "metrics.csv",
                     "inference_outputs.pt", "dashboard.html"):
            src = os.path.join(self._session_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(path, name))
        with open(os.path.join(path, "config.yaml"), "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)
        print(f"[sajepa] session saved to {path}")

    @classmethod
    def load_session(cls, path: str) -> "ScaleAwareJEPA":
        """Restore a model from a saved session directory."""
        cfg_path = os.path.join(path, "config.yaml")
        if os.path.exists(cfg_path):
            instance = cls(config=cfg_path)
        else:
            instance = cls(config=os.path.join(path, "config_used.json"))
        instance._session_dir = os.path.abspath(path)
        instance._is_trained = os.path.exists(os.path.join(path, "inference_outputs.pt"))
        return instance

    # ── diagnostics ────────────────────────────────────────────

    def analyze_rank(self) -> dict:
        """Return effective-rank diagnostics for the current session."""
        if self._session_dir is None:
            raise RuntimeError("No session. Call fit() or load_session() first.")
        from scripts.print_effective_rank import rank_summary
        rows = rank_summary([self._session_dir])
        if not rows:
            return {}
        cols = [
            "session", "mode", "mask_scale", "mask_box", "sampling",
            "l2_norm", "psnorm", "final_norm", "sig_type", "sig_w", "sig_t",
            "sym_loss", "depth", "dilations", "hardcap", "energy", "sim_r", "hinge_r", "sig_r",
            "erank", "context_erank", "predictor_erank", "target_erank",
            "top1", "pred_part", "target_part", "part_ratio", "dead_frac", "dead_ch",
        ]
        return dict(zip(cols, rows[0]))

    # ── dashboard ───────────────────────────────────────────────

    def generate_dashboard(self, output_path: Optional[str] = None):
        """Generate interactive HTML dashboard from the current session.

        If session already has dash artifacts (from post-training inference),
        uses those.  Otherwise falls back to session_to_dash.py.
        """
        if self._session_dir is None:
            raise RuntimeError("No session. Call fit() or load_session() first.")
        try:
            from scripts.session_to_dash import compute_dash_data, plot_dash
            compute_dash_data(self._session_dir, overwrite=False)
            plot_dash(self._session_dir, overwrite=False)
            dash = os.path.join(self._session_dir, "dashboard.html")
            if output_path and os.path.exists(dash):
                import shutil
                shutil.copy2(dash, output_path)
            print(f"[sajepa] dashboard: {output_path or dash}")
        except Exception as e:
            print(f"[sajepa] dashboard failed ({type(e).__name__}), generating minimal dashboard...")
            _generate_minimal_dashboard(self._session_dir, output_path)

    # ── internals ───────────────────────────────────────────────

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _parse_config(config: Optional[dict | str]) -> dict:
        if config is None:
            return load_config(os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "configs", "base_pyramid_scaleaware_convnext.yaml"))
        if isinstance(config, str):
            return load_config(config)
        base = load_config(os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs", "base_pyramid_scaleaware_convnext.yaml"))
        return _deep_merge(base, config)


def _print_metrics_summary(session_dir: str) -> None:
    import csv as _csv
    path = os.path.join(session_dir, "metrics.csv")
    if not os.path.exists(path):
        return
    try:
        epochs: dict[int, dict[str, list[float]]] = {}
        with open(path, "r") as f:
            for row in _csv.DictReader(f):
                ep = int(row.get("epoch", -1))
                if ep < 0:
                    continue
                if ep not in epochs:
                    epochs[ep] = {}
                for k in ("loss_total", "loss_prediction", "loss_spread", "sim", "var", "cov", "lr"):
                    v = row.get(k, "")
                    if v:
                        epochs[ep].setdefault(k, []).append(float(v))
        if not epochs:
            return
        last_ep, first_ep = max(epochs.keys()), min(epochs.keys())
        print(f"\n{'='*60}")
        print(f"Training Metrics (epoch {first_ep} → {last_ep})")
        print(f"{'='*60}")
        keys = [
            ("loss_total", "L(total)     "), ("loss_prediction", "MSE(pred,gt) "),
            ("loss_spread", "sig=relu(1-std)"), ("sim", "cos(pred,gt) "),
            ("var", "var_term    "), ("cov", "cov_term    "), ("lr", "lr          "),
        ]
        for key, label in keys:
            vals_f = epochs[first_ep].get(key, [])
            vals_l = epochs[last_ep].get(key, [])
            first = sum(vals_f) / len(vals_f) if vals_f else 0.0
            last = sum(vals_l) / len(vals_l) if vals_l else 0.0
            ratio = last / first if first > 1e-20 else 1.0
            print(f"  {label}: {first:>8.4f} → {last:>8.4f}  (ratio={ratio:.3f})")
        print(f"{'='*60}")
    except Exception:
        pass


def _generate_minimal_dashboard(session_dir: str, output_path: Optional[str] = None) -> None:
    """Fallback dashboard: reads whatever artifacts exist and renders a simple HTML."""
    dash_path = output_path or os.path.join(session_dir, "dashboard.html")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as _np

        # Try loading what's available
        inf_path = os.path.join(session_dir, "inference_outputs.pt")
        has_inference = os.path.exists(inf_path)
        outputs = torch.load(inf_path, map_location="cpu", weights_only=False) if has_inference else {}

        fig = make_subplots(rows=1, cols=1, subplot_titles=["sajepa Session"])
        if has_inference:
            ctx = outputs.get("context_map")
            if ctx is not None:
                img = ctx.squeeze().cpu().numpy()
                if img.ndim == 3:
                    img = img.mean(0)
                fig.add_trace(go.Heatmap(z=img, colorscale="Viridis"), row=1, col=1)
        fig.write_html(dash_path)
        print(f"[sajepa] minimal dashboard saved: {dash_path}")
    except Exception as e:
        print(f"[sajepa] minimal dashboard failed: {e}")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


================================================================================
FILE: src/train.py
================================================================================

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
from src.inference import run_post_training_inference, run_post_training_inference_3d, run_full_volume_inference_3d
from src.losses import (
    compute_jepa_energy,
    compute_output_spread_regularizer_loss,
    compute_raw_mse_and_norm_err,
    compute_sim_var_cov,
    compute_sim_var_cov_torch,
    compute_target_energy_map,
    embedding_spread_stats,
    parse_spread_regularizer_config,
)
from src.models.build_jepa import CDD_CUBE_ENCODER_TYPES, CDD_DEBUG_ENCODER_TYPES, MASK_MAP_ENCODER_TYPES, PyramidGridJEPA
from src.models.build_jepa3d import PyramidGridJEPA3D, compute_3d_encoder_receptive_field_depth
from src.models.masking import prepare_context_batch
from src.utils import log_error, set_error_log_path
from src.utils.npy import _safe_load_npy
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
        print(
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
    cache_replicas: int = 1,
) -> dict:
    """Pre-compute a bounded CDD decomposition cache on GPU, store in CPU RAM."""
    import glob as _glob
    import constrained_diffusion as cdd

    data_root = data_cfg.get("data_root", "data")
    npy_pattern = data_cfg.get("npy_pattern", "*.npy")
    cdd_mode = str(model_cfg.get("cdd_mode", data_cfg.get("cdd_mode", "log")))
    cdd_constrained = bool(model_cfg.get("cdd_constrained", data_cfg.get("cdd_constrained", True)))
    cdd_sm_mode = str(model_cfg.get("cdd_sm_mode", data_cfg.get("cdd_sm_mode", "reflect")))
    cdd_append_last_residual = bool(model_cfg.get("cdd_append_last_residual", True))
    cdd_pre_log_transform = bool(model_cfg.get("cdd_pre_log_transform", False))
    sigmas = tuple(model_cfg.get("sigmas", [2, 4, 8, 16]))
    if device.type not in ("cuda", "mps"):
        raise RuntimeError(f"[{config_name}] CDD precompute requires CUDA or MPS. Got device={device.type}.")
    npy_files = sorted(_glob.glob(os.path.join(data_root, npy_pattern)))
    if not npy_files:
        print(f"[{config_name}] CDD precompute: no files found for pattern, skipping")
        return {}
    enabled = bool(data_cfg.get("cdd_precompute", True))
    if not enabled:
        print(f"[{config_name}] CDD precompute: disabled by data.cdd_precompute=false")
        return {}
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
        n_channels = len(sigmas) + (1 if cdd_append_last_residual else 0)
        est_bytes_per = int(n_channels) * int(np.prod(sample_shape)) * np.dtype(np.float32).itemsize
        est_process_gb = (est_bytes_per * len(npy_files)) / float(1024 ** 3)
        est_node_gb = est_process_gb * max(1, int(cache_replicas))
        if est_node_gb > max_gb:
            raise RuntimeError(
                f"[{config_name}] CDD precompute: estimated cache {est_process_gb:.2f} GiB per process "
                f"x {max(1, int(cache_replicas))} local replica(s) = {est_node_gb:.2f} GiB exceeds "
                f"data.cdd_precompute_max_gb={max_gb:.2f}. Bump the limit, reduce dataset size, "
                "or disable RAM precompute for DDP."
            )
    print(f"[{config_name}] CDD precompute: {len(npy_files)} file(s) on GPU...")
    cache = {}
    for path in npy_files:
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
        # Preserve volumetric context: decompose a cube once, then let the
        # dataset choose a 2D slice from every cached CDD channel together.
        if cdd_pre_log_transform:
            log_eps_val = float(model_cfg.get("log_eps", 1.0))
            log_floor_mult = float(model_cfg.get("cdd_log_std_floor_mult", 0.05))
            eps_f = max(1e-6, log_eps_val)
            arr_clamp = np.clip(arr, 0.0, None)
            arr_std = float(np.std(arr_clamp))
            log_floor = max(eps_f, arr_std * log_floor_mult)
            arr = np.log(arr_clamp + log_floor).astype(np.float32)
        cdd_channels_arr, cdd_residual = cdd.constrained_diffusion_decomposition(
            arr,
            num_channels=len(sigmas),
            max_scale=max(float(s) for s in sigmas),
            mode=cdd_mode,
            constrained=cdd_constrained,
            sm_mode=cdd_sm_mode,
            return_scales=False,
            verbose=False,
            use_gpu=(device.type == "cuda"),
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
        self.use_cdd = bool(enc_type in CDD_CUBE_ENCODER_TYPES)
        self.return_debug = bool(
            return_debug
            or enc_type in CDD_DEBUG_ENCODER_TYPES
            or enc_type in MASK_MAP_ENCODER_TYPES
        )
        self.mask_scale = float(model.mask_scale)
        self.mask_scale_range = model.mask_scale_range
        self.mask_box_size = int(model.mask_box_size)
        self.mask_box_size_range = model.mask_box_size_range
        self.random_mask_box_per_target = bool(getattr(model, "random_mask_box_per_target", False))
        self.manual_mask_box_sizes = model.manual_mask_box_sizes
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
            "target_nonoverlap": getattr(model, "target_nonoverlap", False),
            "target_allow_partial_overlap": getattr(model, "target_allow_partial_overlap", 0.0),
            "mask_box_hardcap": getattr(model, "mask_box_hardcap", None),
            "use_cdd": self.use_cdd,
        }

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
            mask_box_size_range=self.mask_box_size_range,
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
        or enc_type in MASK_MAP_ENCODER_TYPES
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
        mask_box_size_range=model.mask_box_size_range,
        random_mask_box_per_target=getattr(model, "random_mask_box_per_target", False),
        manual_mask_box_sizes=model.manual_mask_box_sizes,
        cdd_mode=model.cdd_mode,
        cdd_constrained=model.cdd_constrained,
        cdd_sm_mode=model.cdd_sm_mode,
        cdd_append_last_residual=model.cdd_append_last_residual,
        cdd_pre_log_transform=model.cdd_pre_log_transform,
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

    return {
        "val_loss": val_loss,
        "val_sim": val_sim,
        "val_error_by_scale": {float(s): float(np.mean(v)) for s, v in scale_mse.items()},
    }


def _flatten_structured_config(cfg: dict) -> dict:
    """Convert the clean YAML config structure to the legacy flat format.
    
    Passes through ALL keys from each section, so any key valid in the
    legacy flat format works unchanged in the structured YAML.
    """
    # If already flat, return as-is.
    if "cdd_scale_space" not in cfg and "masking" not in cfg and "diagnostics" not in cfg:
        return cfg

    out: dict = {}
    out["data"] = dict(cfg.get("data", {}))
    out["model"] = dict(cfg.get("model", {}))
    out["train"] = dict(cfg.get("training", {}))

    # Merge cdd_scale_space → model (all keys pass through)
    out["model"].update(cfg.get("cdd_scale_space", {}))

    # Merge masking → model (all keys pass through)
    out["model"].update(cfg.get("masking", {}))

    # Merge diagnostics → train (all keys pass through)
    out["train"].update(cfg.get("diagnostics", {}))

    return out


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
    cfg = _flatten_structured_config(cfg)  # flatten after base merge
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
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
    return mode_norm in {"3d_slab", "3d_full_volume"}


def _is_3d_full_volume_mode(mode: str) -> bool:
    return str(mode).strip().lower().replace(" ", "_") == "3d_full_volume"


def _infer_full_volume_depth_3d(data_cfg: dict, cdd_cache: dict | None) -> int:
    if cdd_cache:
        depths = [int(np.asarray(v).shape[1]) for v in cdd_cache.values() if np.asarray(v).ndim == 4]
        if depths:
            return max(depths)

    import glob as _glob

    data_root = data_cfg.get("data_root", "data")
    npy_pattern = data_cfg.get("npy_pattern", "*.npy")
    paths = sorted(_glob.glob(os.path.join(data_root, npy_pattern)))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {data_root}/{npy_pattern}")
    axis = int(data_cfg.get("volume_depth_axis", data_cfg.get("cube_slice_axis", 0))) % 3
    depths = []
    for path in paths:
        shape = tuple(_safe_load_npy(path, mmap_mode="r").shape)
        if len(shape) != 3:
            raise ValueError(f"3D full-volume mode expects cube arrays, got shape={shape} in {path}")
        depths.append(int(shape[axis]))
    return max(depths)


def _resolve_3d_crop_depth(
    *,
    data_cfg: dict,
    model_cfg: dict,
    cdd_cache: dict | None,
    default_depth: int,
    full_volume_mode: bool,
) -> int:
    value = data_cfg.get("volume_crop_depth", data_cfg.get("crop_depth_3d", None))
    if isinstance(value, str) and value.strip().lower() == "full":
        return _infer_full_volume_depth_3d(data_cfg, cdd_cache)
    if value is not None:
        return int(value)
    if full_volume_mode:
        return _infer_full_volume_depth_3d(data_cfg, cdd_cache)
    return int(default_depth)


def _resolve_encoder_alias_2d(name: str) -> str:
    key = str(name).lower()
    alias = {
        # Preferred naming convention (image / image_pyramid prefixes).
        "convnext_image_dense_masked": "convnext_dense_masktoken",
        "cdd_scaleaware_convnext-pyramid-scaleaware": "cdd_scaleaware_convnext",
        "image_pyramid_cdd_scaleaware_convnext": "cdd_scaleaware_convnext",
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
        target_invalid_region_values=tuple(model_cfg.get("target_invalid_region_values", [0, "nan"])),
        target_sampling_mode=str(model_cfg.get("target_sampling_mode", "random")),
        priority_top_percent=float(model_cfg.get("priority_top_percent", 5.0)),
        priority_n_target=model_cfg.get("priority_n_target", 20),
        priority_min_targets_per_map=int(model_cfg.get("priority_min_targets_per_map", 0)),
        priority_dithering_pixels=int(model_cfg.get("priority_dithering_pixels", model_cfg.get("target_dithering_pixels", 6))),
        priority_candidate_oversample=float(model_cfg.get("priority_candidate_oversample", 3.0)),
        use_symmetric_feature_loss=bool(model_cfg.get("use_symmetric_feature_loss", False))
        and float(train_cfg.get("symmetry_loss_weight", train_cfg.get("symmetric_feature_loss_weight", 0.0))) > 0.0,
        target_nonoverlap=bool(model_cfg.get("target_nonoverlap", True)),
        target_allow_partial_overlap=float(model_cfg.get("target_allow_partial_overlap", 0.0)),
        mask_box_hardcap=model_cfg.get("mask_box_hardcap"),
        use_grn=bool(model_cfg.get("use_grn", True)),
    ).to(device)


def build_model3d_from_config(model_cfg: dict, train_cfg: dict, device: torch.device) -> PyramidGridJEPA3D:
    mode = str(model_cfg.get("mode", "")).strip().lower().replace(" ", "_")
    if mode not in {"3d_slab", "3d_full_volume"}:
        raise ValueError(
            f"Unsupported 3D JEPA mode={model_cfg.get('mode')}. "
            "Use model.mode='3d_slab' or model.mode='3d_full_volume'."
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
    cdd_append_last_residual = bool(model_cfg.get("cdd_append_last_residual", True))
    num_scales = 1 if enc_type == "convnext_dense3d" else len(sigmas) + (1 if cdd_append_last_residual else 0)
    encoder_rf_depth = compute_3d_encoder_receptive_field_depth(
        encoder_depth=encoder_depth,
        encoder_kernel_size=encoder_kernel_size,
    )
    return PyramidGridJEPA3D(
        latent_channels=int(model_cfg.get("latent_channels", 16)),
        scale_channels=int(model_cfg.get("scale_channels", model_cfg.get("encoder_width", 8))),
        num_scales=int(num_scales),
        encoder_type=enc_type,
        patch_size=int(model_cfg.get("patch_size", 2)),
        num_targets=int(model_cfg.get("num_targets", 32)),
        encoder_depth=encoder_depth,
        encoder_kernel_size=encoder_kernel_size,
        encoder_stride=int(model_cfg.get("encoder_stride", 1)),
        ema_momentum=float(model_cfg.get("ema_momentum", train_cfg.get("momentum", 0.996))),
        normalize_loss_l2=normalize_loss_l2,
        post_log_transform=bool(model_cfg.get("post_log_transform", True)),
        log_eps=float(model_cfg.get("log_eps", 1e-6)),
        cdd_log_std_floor_mult=float(model_cfg.get("cdd_log_std_floor_mult", 0.05)),
        fusion=fusion,
        mask_box_size=int(model_cfg.get("mask_size", 8)),
        num_mask_boxes=int(model_cfg.get("num_mask_boxes", 8)),
        slab_depth=int(model_cfg.get("slab_depth", max(1, int(model_cfg.get("patch_size", 2))))),
        use_symmetric_feature_loss=bool(model_cfg.get("use_symmetric_feature_loss", False))
        and float(train_cfg.get("symmetry_loss_weight", train_cfg.get("symmetric_feature_loss_weight", 0.0))) > 0.0,
        use_film=bool(model_cfg.get("use_film", True)),
        use_per_scale_adapters=bool(model_cfg.get("use_per_scale_adapters", False)),
        priority_candidate_oversample=float(model_cfg.get("priority_candidate_oversample", 3.0)),
        encoder_receptive_field_depth=encoder_rf_depth,
        use_grn=bool(model_cfg.get("use_grn", True)),
        stem_norm=bool(model_cfg.get("scaleaware_stem_norm", True)),
        norm_per_scale=bool(model_cfg.get("scaleaware_norm_per_scale", True)),
        adapter_norm=bool(model_cfg.get("scaleaware_adapter_norm", True)),
        final_norm=bool(model_cfg.get("scaleaware_final_norm", True)),
        full_volume_training=(mode == "3d_full_volume"),
    ).to(device)


def _dump_movie_frame(
    model,
    movie_batch,
    session_dir,
    epoch,
    batch_idx,
    is_3d_mode,
    device,
    movie_context_data=None,
    fixed_targets: bool = False,
):
    """Save a single movie frame (pred_map, gt_map, x_clean) for later rendering.

    Returns updated movie_context_data.  When fixed_targets is false, the 2D
    movie probe keeps the same image batch but resamples target masks per frame.
    """
    movie_dir = os.path.join(session_dir, "movie_frames")
    os.makedirs(movie_dir, exist_ok=True)
    frame_idx = len([f for f in os.listdir(movie_dir) if f.endswith(".pt")])
    with torch.no_grad():
        model.eval()
        if isinstance(movie_batch, (tuple, list)):
            mb = movie_batch[1] if movie_batch[1] is not None else movie_batch[0]
        else:
            mb = movie_batch
        x_movie = mb.to(device, non_blocking=True)
        if is_3d_mode:
            x_movie = torch.nan_to_num(x_movie, nan=0.0, posinf=0.0, neginf=0.0)
            out_m = model(x_movie)
        else:
            if (not bool(fixed_targets)) or movie_context_data is None:
                movie_context_data = _prepare_context_from_model(model, x_movie, return_debug=False)
            out_m = model(x_movie, context_data=movie_context_data)
        frame = {
            "pred_map": out_m["pred_map"][:1].detach().cpu(),
            "gt_map": out_m["gt_map"][:1].detach().cpu(),
            "x_clean": x_movie[:1].detach().cpu(),
            "target_locations": out_m["target_locations"][:1].detach().cpu(),
            "target_valid": out_m["target_valid"][:1].detach().cpu(),
            "target_scales": out_m["target_scales"][:1].detach().cpu(),
            "epoch": epoch,
            "batch": batch_idx,
            "movie_fixed_targets": bool(fixed_targets),
        }
        ctx = out_m.get("context_map")
        if ctx is not None:
            frame["context_map"] = ctx[:1].detach().cpu()
        torch.save(frame, os.path.join(movie_dir, f"frame_{frame_idx:05d}.pt"))
    return movie_context_data


def run_training(config: dict, config_name: str, sessions_root: str = "sessions") -> str:
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
        print(
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
    if is_ddp and bool(data_cfg.get("cdd_precompute", True)):
        data_cfg["cdd_precompute"] = False
        if is_main_process:
            print(f"[{config_name}] DDP detected: disabling in-process CDD RAM precompute")
    seed = int(train_cfg.get("seed", train_cfg.get("split_seed", 42)))
    rank_seed = seed + int(global_rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed % 2**32)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    if is_main_process:
        print(f"[{config_name}] global_seed={seed} rank_seed={rank_seed}")
    is_3d_mode = _is_3d_jepa_mode(model_cfg.get("mode", "image"))
    is_3d_full_volume_mode = _is_3d_full_volume_mode(model_cfg.get("mode", "image"))

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
            print(f"[{config_name}] wandb initialized project={train_cfg.get('wandb_project', 'jepa-training')}")
        except ImportError:
            print(f"[{config_name}] wandb not installed; pip install wandb to enable")

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
        raise ValueError("model.mode='3d_slab' requires data.input_type='cube'.")
    image_batch_inference = (input_type == "image_batch")

    resolve_pipeline_config(data_cfg=data_cfg, model_cfg=model_cfg)
    if is_main_process:
        _write_data_profile(data_cfg=data_cfg, session_dir=session_dir, config_name=config_name)

    if is_main_process:
        print(
            f"[{config_name}] Resolved pipeline: "
            "dataset_preprocess=normalize01, "
            f"model.post_log_transform={model_cfg.get('post_log_transform', True)}, "
            f"cdd_mode={model_cfg.get('cdd_mode', data_cfg.get('cdd_mode', 'log'))}"
        )

    model = build_model3d_from_config(model_cfg, train_cfg, device) if is_3d_mode else build_model_from_config(model_cfg, data_cfg, train_cfg, device)

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
                print(
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
            print(f"[{config_name}] Resume model: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
            if missing:
                print(f"[{config_name}] resume_model missing_keys={len(missing)} keys: {missing[:10]}")
            if unexpected:
                print(f"[{config_name}] resume_model unexpected_keys={len(unexpected)} keys: {unexpected[:10]}")
            if missing or unexpected:
                error_msg = (
                    f"CRITICAL: Checkpoint architecture mismatch!\n"
                    f"  Missing keys: {len(missing)} (e.g., {missing[:3]})\n"
                    f"  Unexpected keys: {len(unexpected)} (e.g., {unexpected[:3]})"
                )
                if not allow_partial_resume:
                    raise RuntimeError(error_msg + "\nSet train.allow_partial_resume=true if intentional.")
                print("=" * 60)
                print(f"[WARNING] {error_msg}")
                print("[WARNING] Proceeding anyway due to allow_partial_resume=True")
                print("=" * 60)
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
                model = _ddp_wrap(model)
                print(f"[{config_name}] resume_checkpoint_ignored={resume_ckpt_path}")
        if resume_state is not None:
            start_epoch = int(resume_state.get("epoch", 0))
            print(f"resume_checkpoint={resume_ckpt_path} start_epoch={start_epoch}")
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
            print(
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
        print(f"[{config_name}] Resume model: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
        if missing:
            print(f"[{config_name}] resume_model missing_keys={len(missing)} keys: {missing[:10]}")
        if unexpected:
            print(f"[{config_name}] resume_model unexpected_keys={len(unexpected)} keys: {unexpected[:10]}")
        if missing or unexpected:
            error_msg = (
                f"CRITICAL: Model checkpoint mismatch!\n"
                f"  Missing keys: {len(missing)} (e.g., {missing[:3]})\n"
                f"  Unexpected keys: {len(unexpected)} (e.g., {unexpected[:3]})"
            )
            if not allow_partial_resume:
                raise RuntimeError(error_msg + "\nSet train.allow_partial_resume=true if intentional.")
            print("=" * 60)
            print(f"[WARNING] {error_msg}")
            print("[WARNING] Proceeding anyway due to allow_partial_resume=True")
            print("=" * 60)
            print(
                f"[{config_name}] warning: model checkpoint mismatch; "
                "ignoring model_last and starting fresh model/optimizer/scaler."
            )
            model = (
                build_model3d_from_config(model_cfg, train_cfg, device)
                if is_3d_mode
                else build_model_from_config(model_cfg, data_cfg, train_cfg, device)
            )
            model = _ddp_wrap(model)
            print(f"[{config_name}] resume_model_ignored={model_ckpt_path}")
        else:
            if not resume_model_ignored:
                print(f"resume_model={model_ckpt_path}")

    scale_max = float(max(model_cfg.get("sigmas", [2, 4, 8, 16])))
    def _param_max(value_key: str, default: float) -> float:
        values = model_cfg.get(value_key, default)
        if isinstance(values, (list, tuple)):
            if len(values) != 2:
                raise ValueError(f"{value_key} range must contain exactly two values, got {values!r}")
            return float(max(values))
        return float(values)

    _msb = _param_max("mask_size_scaling", 1.0)
    _mb = int(round(_param_max("mask_size", 16)))
    _manual_mask_sizes = model_cfg.get("mask_size_manual")
    if _manual_mask_sizes is not None:
        if isinstance(_manual_mask_sizes, str):
            _manual_items = [v.strip() for v in _manual_mask_sizes.split(",") if v.strip()]
        else:
            try:
                _manual_items = list(_manual_mask_sizes)
            except TypeError:
                _manual_items = [_manual_mask_sizes]
        max_box = max(int(round(float(v))) for v in _manual_items)
    else:
        max_box = round(scale_max * _msb + _mb)
    _mss = float(model_cfg.get("mask_spacing_scaling", 1.5))
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
            print(
                f"[{config_name}] image_batch_n_sample={image_batch_n_sample} "
                f"files={len(selected)} total_selected={total_selected} "
                f"saved_to={sel_path}"
            )

    # --- Pre-compute CDD once on GPU, store in CPU RAM ---
    # Only CDD/pyramid encoders require CDD channels. Plain image ConvNeXt
    # mask-token runs build masks directly on the raw image.
    encoder_type_lower = str(getattr(model_without_ddp, "encoder_type", "")).lower()
    uses_cdd_channels = encoder_type_lower in CDD_CUBE_ENCODER_TYPES
    cdd_cache = _precompute_cdd_cache(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        device=device,
        config_name=config_name,
        cache_replicas=cdd_cache_replicas,
    ) if uses_cdd_channels else None
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
            full_volume_mode=is_3d_full_volume_mode,
        )
        min_crop_depth_3d = target_slab_depth_3d if is_3d_full_volume_mode else auto_crop_depth_3d
        if crop_depth_3d < min_crop_depth_3d:
            raise ValueError(
                "3D crop depth is too small: "
                f"got {crop_depth_3d}, required at least {min_crop_depth_3d} "
                f"(encoder_rf={encoder_rf_depth_3d}, target_slab_depth={target_slab_depth_3d})"
            )
        if is_main_process:
            print(
                f"[{config_name}] {model_without_ddp.mode} geometry: spatial_crop="
                f"{int(data_cfg.get('volume_crop_size', data_cfg.get('crop_size_3d', 64)))} "
                f"crop_depth={crop_depth_3d} encoder_rf_depth={encoder_rf_depth_3d} "
                f"target_depth={'full' if is_3d_full_volume_mode else target_slab_depth_3d}"
            )
        dataset = JEPA3DCropDataset(
            data_root=data_cfg.get("data_root", "data"),
            npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
            num_samples=int(data_cfg.get("num_samples", 2000)),
            crop_size=int(data_cfg.get("volume_crop_size", data_cfg.get("crop_size_3d", 64))),
            crop_depth=crop_depth_3d,
            slab_depth=crop_depth_3d,
            depth_axis=int(data_cfg.get("volume_depth_axis", data_cfg.get("cube_slice_axis", 0))),
            random_axis=bool(data_cfg.get("volume_random_axis", False)),
            normalize=bool(data_cfg.get("normalize", True)),
            crop_strategy=str(data_cfg.get("crop_strategy", "random")),
            cdd_cache=cdd_cache,
        )
        val_dataset = None
        train_dataset = dataset
        inference_dataset = JEPA3DCropDataset(
            data_root=data_cfg.get("data_root", "data"),
            npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
            num_samples=max(1, int(train_cfg.get("inference_num_samples", 8))),
            crop_size=int(data_cfg.get("volume_crop_size", data_cfg.get("crop_size_3d", 64))),
            crop_depth=crop_depth_3d,
            slab_depth=crop_depth_3d,
            depth_axis=int(data_cfg.get("volume_depth_axis", data_cfg.get("cube_slice_axis", 0))),
            random_axis=False,
            normalize=bool(data_cfg.get("normalize", True)),
            crop_strategy="center",
            cdd_cache=cdd_cache,
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
    if device.type == "mps":
        num_workers = 0  # macOS spawn can't pickle closures
    elif "num_workers" in train_cfg:
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
    def _worker_init_fn(worker_id: int) -> None:
        """Ensure each DataLoader worker has a unique NumPy and Python random seed."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader_worker_kwargs = {}
    if num_workers > 0:
        loader_worker_kwargs["prefetch_factor"] = prefetch_factor
        loader_worker_kwargs["worker_init_fn"] = _worker_init_fn

    # DDP: distribute data across GPUs
    train_sampler = DistributedSampler(train_dataset) if is_ddp else None
    val_sampler = None

    masking_collate = None if is_3d_mode else _MaskingCollator(
        model,
        return_debug=bool(train_cfg.get("debug_masking_tensors", False)),
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
    prediction_loss_weight = float(train_cfg.get("prediction_loss_weight", train_cfg.get("mse_loss_weight", 100.0)))
    normalize_loss_l2_active = bool(model_cfg.get("normalize_loss_l2", model_cfg.get("normalize_loss", False)))
    spread_regularizer = parse_spread_regularizer_config(train_cfg)
    spread_regularizer_weight = float(spread_regularizer["weight"])
    embed_spread_target = float(spread_regularizer["target_std"])
    spread_regularizer_eps = float(spread_regularizer["eps"])
    print(f"[{config_name}] spread_regularizer={json.dumps(spread_regularizer, sort_keys=True)}")
    experimental_losses = dict(train_cfg.get("experimental_losses", {}))
    vicreg_var_weight = float(train_cfg.get("vicreg_var_weight", experimental_losses.get("vicreg_var_weight", 0.0)))
    vicreg_cov_weight = float(train_cfg.get("vicreg_cov_weight", experimental_losses.get("vicreg_cov_weight", 0.0)))
    symmetry_loss_weight = float(train_cfg.get("symmetry_loss_weight", train_cfg.get("symmetric_feature_loss_weight", 0.0)))
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
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    warmup_sched = LinearLR(
        optimizer,
        start_factor=min_lr / max(base_lr, 1e-12),
        end_factor=1.0,
        total_iters=warmup_steps_sched,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=total_steps_sched - warmup_steps_sched,
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
    if is_main_process:
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

    loss_weights_path = os.path.join(session_dir, "loss_weights.json")
    if is_main_process and not os.path.exists(loss_weights_path):
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
    if is_ddp:
        dist.barrier()

    movie_dump_every_epoch = bool(train_cfg.get("movie_dump_every_epoch", False))
    movie_dump_every_n_batches = int(train_cfg.get("movie_dump_every_n_batches", 0))
    movie_fixed_targets = bool(train_cfg.get("movie_fixed_targets", False))
    movie_batch = next(iter(inference_loader)) if (movie_dump_every_epoch or movie_dump_every_n_batches > 0) else None
    movie_context_data = None

    model.train()
    start = time.time()
    visit_counts = None
    if start_epoch >= int(epochs):
        print(f"[{config_name}] checkpoint epoch {start_epoch} already >= configured epochs {epochs}, skipping training loop")
    prev_epochs = []
    for epoch in range(start_epoch, epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)
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
                x_context, tloc, tscale, tvalid = context_result[:4]
                debug = context_result[4] if len(context_result) == 5 else {}
                context_data = (x_context, tloc, tscale, tvalid, debug)

            # PyTorch scheduler handles warmup + cosine automatically
            current_step = epoch * max(1, len(dataloader)) + batch_idx

            with autocast(device_type=autocast_device, enabled=use_amp):
                outputs = model(x_clean, context_data=context_data) if not is_3d_mode else model(x_clean)
                _, var_term_t, cov_term_t = compute_sim_var_cov_torch(
                    outputs,
                    spatial_mode=vicreg_spatial_mode,
                )
                loss_spread, z_ctx = compute_output_spread_regularizer_loss(
                    outputs,
                    spread_regularizer,
                    include_predictor=False,
                )
                loss_prediction = model.compute_loss(outputs)
                loss_symmetry = model.compute_symmetric_loss(outputs)
                total_loss = (
                    (prediction_loss_weight * loss_prediction)
                    + (vicreg_var_weight * var_term_t)
                    + (vicreg_cov_weight * cov_term_t)
                    + (spread_regularizer_weight * loss_spread)
                    + (symmetry_loss_weight * loss_symmetry)
                )
            # DDP: sync a DETACHED clone for logging only (autograd must stay local)
            if is_ddp:
                log_loss = total_loss.detach().clone()
                dist.all_reduce(log_loss, op=dist.ReduceOp.SUM)
                log_loss_val = float((log_loss / dist.get_world_size()).item())
            else:
                log_loss_val = float(total_loss.item())

            accum_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
            scaler.scale(total_loss / accum_steps).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler_scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if (not use_amp) or scaler.get_scale() >= scaler_scale_before_step:
                    scheduler.step()
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
            model_without_ddp.update_target_encoder()
            sim_val, var_val, cov_val = compute_sim_var_cov(
                outputs,
                spatial_mode=vicreg_spatial_mode,
            )
            raw_mse_val, norm_err_val = compute_raw_mse_and_norm_err(outputs)
            energy_val = compute_jepa_energy(outputs)
            valid_frac = float(outputs["target_valid"].float().mean().item())
            ctx_stats = embedding_spread_stats(z_ctx, target_std=embed_spread_target)
            targets_per_image = float(outputs["target_valid"].float().sum(dim=1).mean().item())
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
            metrics_rows.append(
                [
                    epoch + 1,
                    batch_idx,
                    global_step,
                    log_loss_val,
                    float(loss_prediction.item()),
                    float(current_lr),
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
            if is_main_process and "cdd_channels_masked" in outputs:
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
                ndim_loc = int(tloc.shape[-1])
                for bi in range(tloc.shape[0]):
                    for ki in range(tloc.shape[1]):
                        if not bool(tvalid[bi, ki]):
                            continue
                        yy = int(tloc[bi, ki, 0])
                        xx = int(tloc[bi, ki, 1])
                        zz = int(tloc[bi, ki, 2]) if ndim_loc >= 3 else 0
                        if (visit_counts is not None) and ndim_loc == 2 and 0 <= yy < visit_counts.shape[0] and 0 <= xx < visit_counts.shape[1]:
                            visit_counts[yy, xx] += 1.0
                        visited_rows.append(
                            [
                                epoch + 1,
                                batch_idx,
                                bi,
                                ki,
                                zz,
                                yy,
                                xx,
                                float(scales[bi, ki]),
                            ]
                        )
            if is_main_process and (batch_idx + 1) % log_flush_interval == 0:
                _flush_csv_rows(masked_scales_log_path, masked_scale_rows)
                _flush_csv_rows(visited_targets_log_path, visited_rows)
            metrics_bar.set_description_str(
                f"L={log_loss_val:.4f} "
                f"MSE(pred,gt)={loss_prediction.item():.4f} "
                f"E={energy_val:.4f} "
                f"sig=relu(1.0-std)={loss_spread.item():.4f} "
                f"cos(pred,gt)={sim_val:.4f} "
                f"std(ch)={ctx_stats['embed_spread_mean']:.3f} "
                f"rank=exp(H(p))={ctx_stats['context_manifold_size']:.2f} "
                f"v={valid_frac:.3f} lr={current_lr:.1e}",
                refresh=True,
            )
            if _use_wandb and (batch_idx + 1) % log_flush_interval == 0:
                import wandb

                wandb.log(
                    {
                        "train/loss_total": total_loss.item(),
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
                        "metrics/mask_footprint_mean_px": mask_footprint_mean_px,
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )
            epoch_total += log_loss_val
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

            # Per-N-batches movie frame dump (captures rapid early learning)
            if is_main_process and movie_dump_every_n_batches > 0 and (batch_idx + 1) % movie_dump_every_n_batches == 0:
                movie_context_data = _dump_movie_frame(
                    model_without_ddp,
                    movie_batch,
                    session_dir,
                    epoch + 1,
                    batch_idx + 1,
                    is_3d_mode,
                    device,
                    movie_context_data,
                    fixed_targets=movie_fixed_targets,
                )
                model.train()

        metrics_bar.close()
        # Only the main process writes files in DDP mode
        if is_main_process:
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
                f"[{config_name}] E {epoch + 1}/{epochs} "
                f"L={avg_total:.4f} mse={avg_prediction:.4f} sig={_fmt_metric(epoch_spread/epoch_batches)} "
                f"cos={_fmt_metric(epoch_sim/epoch_batches)} "
                f"std={_fmt_metric(epoch_embed_spread_mean/epoch_batches)} "
                f"rank={_fmt_metric(epoch_context_manifold_size/epoch_batches)} "
                f"v={_fmt_metric(epoch_valid_frac/epoch_batches)}"
                f"{prev_str}"
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

        # Per-epoch embedding snapshot for movie generation.
        if is_main_process and movie_dump_every_epoch:
            movie_context_data = _dump_movie_frame(
                model_without_ddp,
                movie_batch,
                session_dir,
                epoch + 1,
                0,
                is_3d_mode,
                device,
                movie_context_data,
                fixed_targets=movie_fixed_targets,
            )
            model.train()

    if is_main_process:
        tmp_final = os.path.join(session_dir, "model_last.pt.tmp")
        torch.save(model_without_ddp.state_dict(), tmp_final)
        os.replace(tmp_final, os.path.join(session_dir, "model_last.pt"))

    if is_main_process:
        if is_3d_mode:
            session_dir = run_post_training_inference_3d(
                model=model_without_ddp,
                dataloader=inference_loader,
                session_dir=session_dir,
                config_name=config_name,
                force_recompute_inference=force_recompute_inference,
            )
            if cdd_cache:
                target_slab_depth_3d = max(
                    int(model_cfg.get("patch_size", 2)),
                    int(model_cfg.get("slab_depth", max(1, int(model_cfg.get("patch_size", 2))))),
                )
                run_full_volume_inference_3d(
                    model=model_without_ddp,
                    cdd_cache=cdd_cache,
                    session_dir=session_dir,
                    config_name=config_name,
                    device=device,
                    slab_depth=int(crop_depth_3d if is_3d_full_volume_mode else getattr(model_without_ddp, "required_input_depth", target_slab_depth_3d)),
                    post_log_transform=bool(model_cfg.get("post_log_transform", True)),
                    log_eps=float(model_cfg.get("log_eps", 1.0)),
                    cdd_log_std_floor_mult=float(model_cfg.get("cdd_log_std_floor_mult", 0.05)),
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
                mask_inference=mask_inference,
                viz_crop_border=viz_crop_border,
                viz_crop_border_px=viz_crop_border_px,
                compute_jepa_energy_fn=compute_jepa_energy,
                compute_target_energy_map_fn=compute_target_energy_map,
                inference_visit_batches=inference_visit_batches,
                training_d4_augment=bool(data_cfg.get("d4_augment", False)),
                inference_tta_enabled=inference_tta_enabled,
                inference_tta_mode=inference_tta_mode,
                max_diagnostic_size=int(train_cfg.get("max_diagnostic_size", 768)),
            )
    # Save NPY artifacts (PCA/UMAP/latent embeddings) required by session_to_dash.py.
    # No PNG, HTML, or dashboard rendering is performed here.
    inf_path = os.path.join(session_dir, "inference_outputs.pt")
    if is_main_process and (not is_3d_mode) and os.path.exists(inf_path):
        try:
            outputs = torch.load(inf_path, map_location="cpu", weights_only=False)
            artifacts_dir = save_inference_dashboard(session_dir, outputs, umap_cfg=umap_cfg)
            print(f"[{config_name}] artifacts_saved={artifacts_dir}")
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
        if is_main_process and is_3d_mode and os.path.exists(inf_path):
            try:
                outputs = torch.load(inf_path, map_location="cpu", weights_only=False)
                umap_meta_path = save_volumetric_umap_embeddings(session_dir, outputs, umap_cfg=umap_cfg)
                print(f"[{config_name}] volumetric_umap_saved={umap_meta_path}")
            except Exception as e:
                log_error("volumetric_umap", e)
        elif is_main_process:
            print(f"[{config_name}] warning: inference_outputs.pt missing; skip artifact generation")

    if is_main_process and not is_3d_mode and bool(train_cfg.get("scale_probe_enabled", False)) and model_without_ddp.mode == "pyramid":
        try:
            from src.utils.scale_probe import probe_scale_response

            model_without_ddp.eval()
            with torch.no_grad():
                cdd_channels = None
                if os.path.exists(inf_path):
                    inf_outputs = torch.load(inf_path, map_location="cpu", weights_only=False)
                    cdd_channels = inf_outputs.get("cdd_channels_orig")
                    if cdd_channels is not None:
                        cdd_channels = cdd_channels[:1]
                if cdd_channels is None or cdd_channels.ndim != 4:
                    probe_batch, _ = next(iter(dataloader))
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
                    print(f"[{config_name}] scale_probe_report={json.dumps(report['scale_drop_sensitivity_fraction'])}")
                else:
                    print(f"[{config_name}] scale_probe: cdd_channels not available, skipping")
            model_without_ddp.train()
        except Exception as e:
            log_error("scale_probe", e)

    if is_ddp:
        dist.barrier()

    return session_dir


================================================================================
FILE: src/inference.py
================================================================================

from __future__ import annotations

import json
import os
import csv
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from src.losses import representation_dense_energy
from src.models.masking import _max_effective_mask_box_size


def _save_npz(path: str, arr: np.ndarray) -> None:
    """Save a single array as compressed .npz (zip archive, key='arr').

    Replaces np.save(…, .npy) to drastically reduce disk footprint for spatial
    maps that are sparse, zero-heavy, or contain structured data.
    """
    np.savez_compressed(path, arr=arr)


def _iter_tta_views_2d(x: torch.Tensor, mode: str):
    m = str(mode).lower().strip()
    if m in ("none", "", "off"):
        yield "id", x
        return
    if m in ("flip4", "4fold", "4-fold"):
        yield "id", x
        yield "fx", torch.flip(x, dims=(-1,))
        yield "fy", torch.flip(x, dims=(-2,))
        yield "fxy", torch.flip(x, dims=(-2, -1))
        return
    if m in ("d4", "dihedral8", "8fold", "8-fold", "rotflip8"):
        for k in range(4):
            xr = torch.rot90(x, k=k, dims=(-2, -1))
            yield f"r{k}", xr
            yield f"r{k}fx", torch.flip(xr, dims=(-1,))
        return
    if m in ("rot4", "rot", "rot90"):
        for k in range(4):
            yield f"r{k}", torch.rot90(x, k=k, dims=(-2, -1))
        return
    raise ValueError(f"Unsupported inference TTA mode: {mode}")


def _apply_tta_2d(name: str, z: torch.Tensor) -> torch.Tensor:
    """Invert a TTA view transform produced by _iter_tta_views_2d."""
    if name == "id":
        return z
    if name == "fx":
        return torch.flip(z, dims=(-1,))
    if name == "fy":
        return torch.flip(z, dims=(-2,))
    if name == "fxy":
        return torch.flip(z, dims=(-2, -1))
    if name.startswith("r") and name[1:2].isdigit():
        rest = name[1:]
        if rest.endswith("fx"):
            k = int(rest[:-2])
            return torch.rot90(torch.flip(z, dims=(-1,)), k=-k, dims=(-2, -1))
        k = int(rest)
        return torch.rot90(z, k=-k, dims=(-2, -1))
    raise ValueError(name)


def _forward_tta_streaming_2d(
    *,
    x: torch.Tensor,
    mode: str,
    forward_one: Callable[[torch.Tensor, torch.Tensor | None], dict],
    cdd: torch.Tensor | None = None,
    keys=("pred_map", "gt_map", "context_map"),
) -> tuple[dict, int]:
    """Run TTA one view at a time and average aligned maps without stacking views."""
    first_out = None
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    n_views = 0
    align_keys = set(keys) | {
        "x_clean_raw",
        "x_context_raw",
        "x_clean",
        "x_context",
        "network_context_in",
        "network_target_in",
        "target_mask_map",
        "cdd_channels_orig",
        "cdd_channels_masked",
        "dip_field_per_channel",
        "pyramid_mask_token",
    }
    cdd_iter = _iter_tta_views_2d(cdd, mode) if cdd is not None else None
    for name, xv in _iter_tta_views_2d(x, mode):
        cdv = next(cdd_iter)[1] if cdd_iter is not None else None
        out_v = forward_one(xv, cdv)
        n_views += 1
        for key in align_keys:
            if out_v.get(key) is not None:
                out_v[key] = _apply_tta_2d(name, out_v[key])
        if first_out is None:
            first_out = out_v
        for key in keys:
            value = out_v.get(key)
            if value is None:
                continue
            if key not in sums:
                sums[key] = value
                counts[key] = 1
            else:
                sums[key].add_(value)
                counts[key] += 1
        if out_v is not first_out:
            del out_v
    if first_out is None:
        raise ValueError("TTA produced no views")
    for key, value in sums.items():
        first_out[key] = value.div(float(max(1, counts[key])))
    return first_out, n_views


def _mask_invalid_targets_from_input(
    *,
    outputs: dict,
    x_input: torch.Tensor,
    patch_size: int,
    invalid_values=(0.0, "nan"),
) -> int:
    loc = outputs["target_locations"]
    valid = outputs["target_valid"]
    if x_input.dim() != 4 or x_input.shape[1] < 1:
        return 0
    bsz, ksz, _ = loc.shape
    h, w = int(x_input.shape[-2]), int(x_input.shape[-1])
    half_lo = int(patch_size) // 2
    half_hi = int(patch_size) - half_lo
    invalid_specs = tuple(invalid_values) if invalid_values is not None else tuple()
    updated = valid.clone()
    numeric_specs = []
    check_nan = False
    for spec in invalid_specs:
        if isinstance(spec, str) and spec.lower() == "nan":
            check_nan = True
        else:
            try:
                numeric_specs.append(float(spec))
            except (TypeError, ValueError):
                continue
    loc_y = loc[..., 0].long()
    loc_x = loc[..., 1].long()
    y0 = loc_y - half_lo
    x0 = loc_x - half_lo
    y1 = loc_y + half_hi
    x1 = loc_x + half_hi
    in_bounds = (y0 >= 0) & (x0 >= 0) & (y1 <= h) & (x1 <= w)
    invalid_target = valid & ~in_bounds
    if int(patch_size) > 0 and h >= int(patch_size) and w >= int(patch_size):
        patches = F.unfold(x_input[:, :1], kernel_size=int(patch_size)).transpose(1, 2)
        linear = (y0.clamp(0, h - int(patch_size)) * (w - int(patch_size) + 1) + x0.clamp(0, w - int(patch_size))).clamp_min(0)
        gather_idx = linear.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        target_patches = patches.gather(1, gather_idx)
        invalid_mask = torch.zeros_like(target_patches, dtype=torch.bool)
        if check_nan:
            invalid_mask |= ~torch.isfinite(target_patches)
        for spec in numeric_specs:
            invalid_mask |= torch.isclose(
                target_patches,
                torch.tensor(spec, device=target_patches.device, dtype=target_patches.dtype),
            )
        invalid_target |= valid & in_bounds & invalid_mask.all(dim=-1)
    else:
        invalid_target |= valid
    updated[invalid_target] = False
    outputs["target_valid"] = updated
    return int(invalid_target.sum().item())


def _accumulate_point_energy(
    *,
    outputs: dict,
    energy_sum: torch.Tensor,
    count_map: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[float, int]:
    pred = outputs["pred_patches"]
    gt = outputs["gt_patches"].detach()
    loc = outputs["target_locations"]
    valid = outputs["target_valid"]
    h, w = int(image_size[0]), int(image_size[1])
    reduce_dims = tuple(range(2, pred.dim()))
    err = (pred - gt).pow(2).mean(dim=reduce_dims)  # B,K
    cy = loc[..., 0].long()
    cx = loc[..., 1].long()
    in_bounds = valid & (cy >= 0) & (cy < h) & (cx >= 0) & (cx < w)
    if not bool(in_bounds.any().item()):
        return 0.0, 0
    b_idx = torch.arange(err.shape[0], device=err.device).unsqueeze(1).expand_as(err)
    flat_idx = (b_idx * h * w + cy.clamp(0, h - 1) * w + cx.clamp(0, w - 1))[in_bounds]
    values = err[in_bounds].to(dtype=energy_sum.dtype)
    energy_sum.view(-1).scatter_add_(0, flat_idx.reshape(-1), values.reshape(-1))
    count_map.view(-1).scatter_add_(0, flat_idx.reshape(-1), torch.ones_like(values, dtype=count_map.dtype).reshape(-1))
    return float(values.sum().item()), int(values.numel())


def _apply_nan_boundary_frame(x: torch.Tensor, border_px: int) -> torch.Tensor:
    if border_px <= 0:
        return x
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(x.shape)}")
    out = x.clone()
    _, _, h, w = out.shape
    b = int(max(0, min(border_px, h // 2, w // 2)))
    if b <= 0:
        return out
    out[:, :, :b, :] = float("nan")
    out[:, :, h - b :, :] = float("nan")
    out[:, :, :, :b] = float("nan")
    out[:, :, :, w - b :] = float("nan")
    return out


def _first_full_resolution_batch(dataloader, max_diagnostic_size: int = 768):
    """Fetch a single batch for diagnostic visualization.

    Enforces a max size to prevent GPU OOM and 100MB+ HTML crashes
    on very large images (e.g. 2000×2000+).
    """
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None or not hasattr(dataset, "__getitem__"):
        return None
    old = {}

    # Check original image size to see if we need to enforce a crop
    needs_safety_crop = False
    try:
        sample_path = dataset.sample_index[0][0]
        shape = dataset._probe_file_shape(sample_path)
        if shape[-2] > max_diagnostic_size or shape[-1] > max_diagnostic_size:
            needs_safety_crop = True
    except Exception:
        pass

    target_crop_mode = "center" if needs_safety_crop else "none"
    target_crop_size = max_diagnostic_size if needs_safety_crop else None

    for name, value in (
        ("crop_mode", target_crop_mode),
        ("crop_size", target_crop_size),
        ("d4_augment", False),
        ("random_roll_max", 0),
        ("crop_min_valid_fraction", 0.0),
    ):
        if hasattr(dataset, name):
            old[name] = getattr(dataset, name)
            setattr(dataset, name, value)
    try:
        sample = dataset[0]
    finally:
        for name, value in old.items():
            setattr(dataset, name, value)
    if isinstance(sample, (tuple, list)):
        return tuple(x.unsqueeze(0) if torch.is_tensor(x) and x.dim() in (3, 4) else x for x in sample)
    if torch.is_tensor(sample):
        return sample.unsqueeze(0) if sample.dim() in (3, 4) else sample
    return None


def run_post_training_inference(
    *,
    model,
    dataloader,
    session_dir: str,
    config_name: str,
    visit_counts,
    force_recompute_inference: bool,
    inference_mask_passes: int,
    mask_inference: bool,
    viz_crop_border: bool,
    viz_crop_border_px: int | None,
    compute_jepa_energy_fn: Callable,
    compute_target_energy_map_fn: Callable,
    inference_visit_batches: int = 32,
    training_d4_augment: bool = False,
    inference_tta_enabled: bool = False,
    inference_tta_mode: str = "flip4",
    max_diagnostic_size: int = 768,
) -> str:
    inference_outputs_path = os.path.join(session_dir, "inference_outputs.pt")
    if (not force_recompute_inference) and os.path.exists(inference_outputs_path):
        print(
            f"[{config_name}] inference_outputs.pt already exists; "
            "skipping post-training dashboard-sample inference "
            "(set train.force_recompute_inference=true to recompute)"
        )
        return session_dir

    inference_required = [
        inference_outputs_path,
        os.path.join(session_dir, "network_input_clean.npz"),
        os.path.join(session_dir, "network_input_context.npz"),
        os.path.join(session_dir, "pred_map.npz"),
        os.path.join(session_dir, "gt_map.npz"),
        os.path.join(session_dir, "target_energy_map.npz"),
        os.path.join(session_dir, "jepa_energy_summary.json"),
    ]
    if (not force_recompute_inference) and all(os.path.exists(p) for p in inference_required):
        print(
            f"[{config_name}] inference artifacts already exist; "
            "skipping post-training dashboard-sample inference "
            "(set train.force_recompute_inference=true to recompute)"
        )
        return session_dir

    try:
        dataloader_len = len(dataloader)
    except TypeError:
        dataloader_len = None
    if dataloader_len is not None and dataloader_len > 1:
        print(
            f"[{config_name}] post_training_inference scope=dashboard_sample_batch "
            f"using first batch only out of {dataloader_len}; use src.inference_from_session "
            "for explicit dataset inference"
        )
    else:
        print(f"[{config_name}] post_training_inference scope=dashboard_sample_batch")
    model.eval()
    with torch.no_grad():
        print(f"[{config_name}] post_training_inference loading sample batch")
        raw_batch = _first_full_resolution_batch(dataloader, max_diagnostic_size=max_diagnostic_size)
        if raw_batch is None:
            raw_batch = next(iter(dataloader))
        else:
            print(f"[{config_name}] post_training_inference using uncropped first image for dashboard")
        if isinstance(raw_batch, (tuple, list)) and len(raw_batch) == 2 and raw_batch[1] is not None:
            cdd_raw, x_raw = raw_batch
            cdd_raw = cdd_raw.to(next(model.parameters()).device)
        else:
            cdd_raw = None
            x_raw = raw_batch if not isinstance(raw_batch, (tuple, list)) else raw_batch[0]
        x_raw = x_raw.to(next(model.parameters()).device)
        # Deterministic lattice sweep is only meaningful when mask inference is enabled.
        mask_scale = float(getattr(model, "mask_scale", 1.0))
        mask_box_size = int(getattr(model, "mask_box_size", 16))
        max_box = _max_effective_mask_box_size(
            sigmas=tuple(float(s) for s in getattr(model, "sigmas", (16.0,))),
            mask_scale=mask_scale,
            mask_box_size=mask_box_size,
            inner_target_size=int(getattr(model, "patch_size", 3)),
            hardcap=getattr(model, "mask_box_hardcap", None),
            manual_mask_box_sizes=getattr(model, "manual_mask_box_sizes", None),
        )
        spacing = int(
            max(
                1,
                round(float(max_box) * float(getattr(model, "spacing_scale", 1.5))),
            )
        )
        if bool(mask_inference):
            # Lattice sweep provides a dense inference map from discrete block masks.
            import warnings
            warnings.warn(
                "Lattice sweep inference is deprecated and will be removed. "
                "Switch to block masking for single-pass dense energy maps.",
                FutureWarning,
                stacklevel=2,
            )
            all_shifts = [(dy, dx) for dy in range(spacing) for dx in range(spacing)]
            n_passes = max(1, int(inference_mask_passes))
            shifts = all_shifts if n_passes <= 0 else all_shifts[: min(len(all_shifts), n_passes)]
        else:
            shifts = [(0, 0)]
        print(f"[{config_name}] post_training_inference model forward deterministic_shifts={len(shifts)} spacing={spacing}")
        outputs = None
        energy_sum = None
        count_map = None
        total_energy = 0.0
        total_valid = 0
        invalid_region_skip = bool(getattr(model, "target_invalid_region_skip", False))
        invalid_region_values = tuple(getattr(model, "target_invalid_region_values", (0.0, "nan")))
        patch_size = int(getattr(model, "patch_size", 2))
        tta_view_count = 1
        shift_sums: dict[str, torch.Tensor] = {}
        shift_counts: dict[str, int] = {}
        for pi, shift in enumerate(shifts):
            return_debug_next = pi == 0

            def _forward_one_tta(xv: torch.Tensor, cdv: torch.Tensor | None) -> dict:
                nonlocal return_debug_next
                return_debug = return_debug_next
                return_debug_next = False
                return model(
                    xv,
                    return_debug=return_debug,
                    enable_grid_jitter=False,
                    enable_target_dithering=False,
                    lattice_shift_override=shift,
                    mask_inference=bool(mask_inference),
                    cdd_orig=cdv,
                )

            if bool(inference_tta_enabled):
                out_i, tta_view_count = _forward_tta_streaming_2d(
                    x=x_raw,
                    mode=inference_tta_mode,
                    forward_one=_forward_one_tta,
                    cdd=cdd_raw,
                )
            else:
                out_i = _forward_one_tta(x_raw, cdd_raw)
            for key in ("pred_map", "gt_map", "context_map"):
                value = out_i.get(key)
                if value is None:
                    continue
                if key not in shift_sums:
                    shift_sums[key] = value
                    shift_counts[key] = 1
                else:
                    shift_sums[key].add_(value)
                    shift_counts[key] += 1
            if invalid_region_skip:
                _mask_invalid_targets_from_input(
                    outputs=out_i,
                    x_input=x_raw,
                    patch_size=patch_size,
                    invalid_values=invalid_region_values,
                )
            if outputs is None:
                outputs = out_i
                h, w = outputs["x_clean"].shape[-2:]
                bsz = outputs["x_clean"].shape[0]
                energy_sum = torch.zeros((bsz, 1, h, w), device=outputs["x_clean"].device, dtype=outputs["x_clean"].dtype)
                count_map = torch.zeros_like(energy_sum)
            e_tot_i, n_val_i = _accumulate_point_energy(
                outputs=out_i,
                energy_sum=energy_sum,
                count_map=count_map,
                image_size=(h, w),
            )
            total_energy += float(e_tot_i)
            total_valid += int(n_val_i)
        assert outputs is not None and energy_sum is not None and count_map is not None
        # Average aligned maps across deterministic shifts to stabilize dashboard embeddings.
        if len(shifts) > 1:
            for key, value in shift_sums.items():
                outputs[key] = value.div(float(max(1, shift_counts[key])))

    inference_outputs = {
        "inference_scope": "dashboard_sample_batch",
        "inference_num_dataloader_batches_seen": 1,
        "inference_num_dataloader_batches_available": dataloader_len if dataloader_len is not None else -1,
        "x_clean_raw": outputs.get("x_clean_raw", outputs["x_clean"])[:8].detach().cpu(),
        "x_context_raw": outputs.get("x_context_raw", outputs["x_context"])[:8].detach().cpu(),
        "x_clean": outputs["x_clean"][:8].detach().cpu(),
        "x_context": outputs["x_context"][:8].detach().cpu(),
        "target_locations": outputs["target_locations"][:8].detach().cpu(),
        "target_scales": outputs["target_scales"][:8].detach().cpu(),
        "target_valid": outputs["target_valid"][:8].detach().cpu(),
        "pred_map": outputs["pred_map"][:2].detach().cpu(),
        "gt_map": outputs["gt_map"][:2].detach().cpu(),
        "context_map": outputs.get("context_map", outputs["pred_map"])[:2].detach().cpu(),
        "pred_patches": outputs["pred_patches"][:2].detach().cpu(),
        "gt_patches": outputs["gt_patches"][:2].detach().cpu(),
    }
    if "network_context_in" in outputs:
        inference_outputs["network_context_in"] = outputs["network_context_in"][:8].detach().cpu()
    if "network_target_in" in outputs:
        inference_outputs["network_target_in"] = outputs["network_target_in"][:8].detach().cpu()
    if "target_mask_map" in outputs:
        inference_outputs["target_mask_map"] = outputs["target_mask_map"][:8].detach().cpu()
    if "cdd_channels_orig" in outputs:
        inference_outputs["cdd_channels_orig"] = outputs["cdd_channels_orig"][:8].detach().cpu()
    if "cdd_channels_masked" in outputs:
        inference_outputs["cdd_channels_masked"] = outputs["cdd_channels_masked"][:8].detach().cpu()
    if "dip_field_per_channel" in outputs:
        inference_outputs["dip_field_per_channel"] = outputs["dip_field_per_channel"][:8].detach().cpu()
    if "pyramid_mask_token" in outputs:
        inference_outputs["pyramid_mask_token"] = outputs["pyramid_mask_token"][:8].detach().cpu()
    for k in (
        "priority_good_candidates",
        "priority_nonzero_mean",
        "priority_auto_base_targets",
        "priority_effective_targets",
    ):
        if k in outputs:
            inference_outputs[k] = outputs[k][:8].detach().cpu()
    energy_scalar = float(total_energy / max(1, total_valid))
    energy_scalar_norm = compute_jepa_energy_fn(outputs, normalize=True)
    # Dense full-image energy from lattice prediction/target maps.
    e_map_dense = compute_target_energy_map_fn(
        outputs,
        image_size=(int(outputs["x_clean"].shape[-2]), int(outputs["x_clean"].shape[-1])),
    )
    # Keep point-sampled target energy as a secondary diagnostic.
    e_map_points = energy_sum / count_map.clamp_min(1.0)
    inference_outputs["jepa_energy"] = torch.tensor(energy_scalar, dtype=torch.float32)
    inference_outputs["jepa_energy_normalized"] = torch.tensor(energy_scalar_norm, dtype=torch.float32)
    inference_outputs["target_energy_map"] = e_map_dense["energy_rel_sym"][:8].detach().cpu()
    inference_outputs["target_energy_raw_map"] = e_map_dense["energy_raw"][:8].detach().cpu()
    inference_outputs["target_energy_rel_gt_map"] = e_map_dense["energy_rel_gt"][:8].detach().cpu()
    inference_outputs["target_energy_cosine_map"] = e_map_dense["energy_cosine"][:8].detach().cpu()
    inference_outputs["target_energy_point_map"] = e_map_points[:8].detach().cpu()
    inference_outputs["target_energy_count_map"] = count_map[:8].detach().cpu()

    if bool(viz_crop_border):
        if viz_crop_border_px is None:
            auto_border = int(max(getattr(model, "sigmas", (16.0,))))
        else:
            auto_border = int(max(0, viz_crop_border_px))
        inference_outputs["target_energy_map"] = _apply_nan_boundary_frame(
            inference_outputs["target_energy_map"], auto_border
        )
        inference_outputs["target_energy_raw_map"] = _apply_nan_boundary_frame(
            inference_outputs["target_energy_raw_map"], auto_border
        )
        inference_outputs["target_energy_rel_gt_map"] = _apply_nan_boundary_frame(
            inference_outputs["target_energy_rel_gt_map"], auto_border
        )
        inference_outputs["target_energy_cosine_map"] = _apply_nan_boundary_frame(
            inference_outputs["target_energy_cosine_map"], auto_border
        )
        inference_outputs["target_energy_point_map"] = _apply_nan_boundary_frame(
            inference_outputs["target_energy_point_map"], auto_border
        )

    if "target_mask_map" in inference_outputs:
        tmap = inference_outputs["target_mask_map"]
    else:
        # Fallback map should represent target centers as points, not squares.
        tloc = inference_outputs["target_locations"]
        tvalid = inference_outputs["target_valid"]
        bsz, _, _ = tloc.shape
        h, w = inference_outputs["x_clean"].shape[-2:]
        tmap = torch.zeros((bsz, 1, h, w), dtype=inference_outputs["x_clean"].dtype)
        for bi in range(bsz):
            for ki in range(tloc.shape[1]):
                if not bool(tvalid[bi, ki].item()):
                    continue
                cy = int(tloc[bi, ki, 0].item())
                cx = int(tloc[bi, ki, 1].item())
                if 0 <= cy < h and 0 <= cx < w:
                    # 3×3 cross so targets are visible when downscaled
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = cy + dy, cx + dx
                            if 0 <= yy < h and 0 <= xx < w:
                                tmap[bi, 0, yy, xx] = 1.0
    inference_outputs["target_map"] = tmap
    torch.save(inference_outputs, inference_outputs_path)
    print(f"[{config_name}] saved inference_outputs.pt")

    _save_npz(os.path.join(session_dir, "network_input_clean.npz"), inference_outputs["x_clean"].numpy())
    _save_npz(os.path.join(session_dir, "network_input_context.npz"), inference_outputs["x_context"].numpy())
    _save_npz(os.path.join(session_dir, "network_input_clean_raw.npz"), inference_outputs["x_clean_raw"].numpy())
    _save_npz(os.path.join(session_dir, "network_input_context_raw.npz"), inference_outputs["x_context_raw"].numpy())
    if "network_context_in" in inference_outputs:
        _save_npz(
            os.path.join(session_dir, "network_context_in.npz"),
            inference_outputs["network_context_in"].numpy(),
        )
    if "network_target_in" in inference_outputs:
        _save_npz(
            os.path.join(session_dir, "network_target_in.npz"),
            inference_outputs["network_target_in"].numpy(),
        )
    _save_npz(os.path.join(session_dir, "target_valid.npz"), inference_outputs["target_valid"].numpy())
    if "target_mask_map" in inference_outputs:
        _save_npz(os.path.join(session_dir, "target_mask_map.npz"), inference_outputs["target_mask_map"].numpy())
    if "cdd_channels_orig" in inference_outputs:
        _save_npz(os.path.join(session_dir, "cdd_channels_orig.npz"), inference_outputs["cdd_channels_orig"].numpy())
    if "cdd_channels_masked" in inference_outputs:
        _save_npz(os.path.join(session_dir, "cdd_channels_masked.npz"), inference_outputs["cdd_channels_masked"].numpy())
        # Requested artifact: one example masked channel cube for quick inspection.
        _save_npz(
            os.path.join(session_dir, "example_masked_channel_cube.npz"),
            inference_outputs["cdd_channels_masked"][0].numpy().astype(np.float32),
        )
    if "dip_field_per_channel" in inference_outputs:
        _save_npz(
            os.path.join(session_dir, "dip_field_per_channel.npz"),
            inference_outputs["dip_field_per_channel"].numpy(),
        )
        # Backward-compatible dashboard artifact name.
        _save_npz(
            os.path.join(session_dir, "pyramid_mask_token.npz"),
            inference_outputs["dip_field_per_channel"].numpy(),
        )
    if "pyramid_mask_token" in inference_outputs:
        _save_npz(os.path.join(session_dir, "pyramid_mask_token.npz"), inference_outputs["pyramid_mask_token"].numpy())
    if visit_counts is not None:
        _save_npz(os.path.join(session_dir, "visited_target_frequency.npz"), visit_counts.astype(np.float32))
    _save_npz(os.path.join(session_dir, "target_energy_map.npz"), inference_outputs["target_energy_map"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_raw_map.npz"), inference_outputs["target_energy_raw_map"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_rel_gt_map.npz"), inference_outputs["target_energy_rel_gt_map"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_cosine_map.npz"), inference_outputs["target_energy_cosine_map"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_point_map.npz"), inference_outputs["target_energy_point_map"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_count_map.npz"), inference_outputs["target_energy_count_map"].numpy())
    with open(os.path.join(session_dir, "jepa_energy_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "jepa_energy": float(energy_scalar),
                "jepa_energy_normalized": float(energy_scalar_norm),
                "inference_mask_passes": int(len(shifts)),
                "inference_grid_spacing": int(spacing),
                "mask_inference": bool(mask_inference),
                "inference_tta_enabled": bool(inference_tta_enabled),
                "inference_tta_mode": str(inference_tta_mode),
                "inference_tta_views": int(tta_view_count),
            },
            f,
            indent=2,
        )
    if "priority_effective_targets" in inference_outputs:
        active_target_fraction = float(getattr(model, "mask_fraction", 1.0))
        priority_n_target_cfg = getattr(model, "priority_n_target", 20)
        summary = {
            "active_target_fraction": float(active_target_fraction),
            "priority_n_target_config": priority_n_target_cfg,
            "priority_min_targets_per_map_config": int(getattr(model, "priority_min_targets_per_map", 0)),
            "priority_good_candidates_mean": float(inference_outputs.get("priority_good_candidates", torch.tensor([])).float().mean().item())
            if inference_outputs.get("priority_good_candidates", None) is not None and inference_outputs["priority_good_candidates"].numel() > 0
            else 0.0,
            "priority_nonzero_mean_mean": float(inference_outputs.get("priority_nonzero_mean", torch.tensor([])).float().mean().item())
            if inference_outputs.get("priority_nonzero_mean", None) is not None and inference_outputs["priority_nonzero_mean"].numel() > 0
            else 1.0,
            "priority_auto_base_targets_mean": float(inference_outputs.get("priority_auto_base_targets", torch.tensor([])).float().mean().item())
            if inference_outputs.get("priority_auto_base_targets", None) is not None and inference_outputs["priority_auto_base_targets"].numel() > 0
            else 0.0,
            "priority_effective_targets_mean": float(inference_outputs.get("priority_effective_targets", torch.tensor([])).float().mean().item())
            if inference_outputs.get("priority_effective_targets", None) is not None and inference_outputs["priority_effective_targets"].numel() > 0
            else 0.0,
        }
        with open(os.path.join(session_dir, "target_selection_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
    # Canonical target-visit heatmap from inference loader (d4_augment is forced off there).
    # This avoids mirrored-symmetry artefacts from training-time augmentation logs.
    visit_h = int(inference_outputs["x_clean"].shape[-2])
    visit_w = int(inference_outputs["x_clean"].shape[-1])
    canonical_visit_counts = np.zeros((visit_h, visit_w), dtype=np.float32)
    canonical_rows = []
    max_visit_batches = int(inference_visit_batches)
    if max_visit_batches < 0:
        max_visit_batches = 0
    dev = next(model.parameters()).device
    with torch.no_grad():
        for ib, batch in enumerate(dataloader):
            if max_visit_batches > 0 and ib >= max_visit_batches:
                break
            if isinstance(batch, (tuple, list)) and len(batch) == 2 and batch[1] is not None:
                cdd_b, xb = batch
                cdd_b = cdd_b.to(dev)
            else:
                cdd_b = None
                xb = batch if not isinstance(batch, (tuple, list)) else batch[0]
            xb = xb.to(dev)
            outb = model(
                xb,
                return_debug=False,
                enable_grid_jitter=False,
                mask_inference=bool(mask_inference),
                cdd_orig=cdd_b,
            )
            tloc_b = outb["target_locations"].detach().cpu().numpy()
            tvalid_b = outb["target_valid"].detach().cpu().numpy().astype(bool)
            tscale_b = outb["target_scales"].detach().cpu().numpy()
            for bi in range(tloc_b.shape[0]):
                for ki in range(tloc_b.shape[1]):
                    if not bool(tvalid_b[bi, ki]):
                        continue
                    yy = int(tloc_b[bi, ki, 0])
                    xx = int(tloc_b[bi, ki, 1])
                    if 0 <= yy < visit_h and 0 <= xx < visit_w:
                        canonical_visit_counts[yy, xx] += 1.0
                        canonical_rows.append(
                            [int(ib), int(bi), int(ki), int(yy), int(xx), float(tscale_b[bi, ki])]
                        )
    _save_npz(
        os.path.join(session_dir, "visited_target_frequency_canonical.npz"),
        canonical_visit_counts.astype(np.float32),
    )
    with open(
        os.path.join(session_dir, "visited_target_locations_canonical.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(["inference_batch", "sample_idx", "target_idx", "y", "x", "scale"])
        if canonical_rows:
            w.writerows(canonical_rows)
    if bool(training_d4_augment):
        print(
            f"[{config_name}] canonical_visit_map_saved "
            "(built from d4_augment=false inference loader)"
        )
    _save_npz(os.path.join(session_dir, "pred_map.npz"), inference_outputs["pred_map"].numpy())
    _save_npz(os.path.join(session_dir, "gt_map.npz"), inference_outputs["gt_map"].numpy())
    pred_norm = inference_outputs["pred_map"].norm(dim=1).numpy()
    gt_norm = inference_outputs["gt_map"].norm(dim=1).numpy()
    err_norm = (inference_outputs["pred_map"] - inference_outputs["gt_map"]).norm(dim=1).numpy()
    _save_npz(os.path.join(session_dir, "pred_latent_norm.npz"), pred_norm)
    _save_npz(os.path.join(session_dir, "gt_latent_norm.npz"), gt_norm)
    _save_npz(os.path.join(session_dir, "pred_gt_latent_error_norm.npz"), err_norm)
    print(
        f"[{config_name}] post_training_artifacts_saved session_dir={session_dir} "
        f"(run scripts/session_to_dash.py to generate plots/dashboards)"
    )
    return session_dir


def run_post_training_inference_3d(
    *,
    model,
    dataloader,
    session_dir: str,
    config_name: str,
    force_recompute_inference: bool,
) -> str:
    inference_outputs_path = os.path.join(session_dir, "inference_outputs.pt")
    if (not force_recompute_inference) and os.path.exists(inference_outputs_path):
        return session_dir

    model.eval()
    with torch.no_grad():
        x = next(iter(dataloader))
        x = x.to(next(model.parameters()).device)
        outputs = model(x)

    pred_map = outputs["pred_map"][:1].detach().cpu()
    gt_map = outputs["gt_map"][:1].detach().cpu()
    context_map = outputs["context_map"][:1].detach().cpu()
    x_clean = outputs["x_clean"][:1].detach().cpu()
    x_context = outputs["x_context"][:1].detach().cpu()
    mask_cube = outputs["mask_cube"][:1].detach().cpu()
    if pred_map.dim() != 5:
        raise ValueError(f"Expected center-slab pred_map BxCxDxHxW, got {tuple(pred_map.shape)}")
    center_slab_middle_index = int(pred_map.shape[2] // 2)
    slab_energy = representation_dense_energy(pred_map, gt_map)

    inference_outputs = {
        "x_clean": x_clean,
        "x_context": x_context,
        "mask_cube": mask_cube,
        "pred_map": pred_map,
        "gt_map": gt_map,
        "context_map": context_map,
        "target_locations": outputs["target_locations"][:1].detach().cpu(),
        "target_valid": outputs["target_valid"][:1].detach().cpu(),
        "target_scales": outputs.get("target_scales", torch.ones_like(outputs["target_valid"], dtype=x_clean.dtype))[:1].detach().cpu(),
        "pred_patches": outputs["pred_patches"][:1].detach().cpu(),
        "gt_patches": outputs["gt_patches"][:1].detach().cpu(),
        "target_energy_map": slab_energy["energy_rel_sym"],
        "target_energy_raw_map": slab_energy["energy_raw"],
        "target_energy_rel_gt_map": slab_energy["energy_rel_gt"],
        "target_energy_cosine_map": slab_energy["energy_cosine"],
        "center_slab_middle_index": torch.tensor(center_slab_middle_index, dtype=torch.int64),
    }
    inference_outputs["selected_slab_start_index"] = outputs["selected_slab_start_index"][:1].detach().cpu()
    inference_outputs["selected_slab_depth"] = outputs["selected_slab_depth"][:1].detach().cpu()
    torch.save(inference_outputs, inference_outputs_path)

    _save_npz(os.path.join(session_dir, "network_input_clean_3d.npz"), x_clean.numpy())
    _save_npz(os.path.join(session_dir, "network_input_context_3d.npz"), x_context.numpy())
    _save_npz(os.path.join(session_dir, "mask_cube_3d.npz"), mask_cube.numpy())
    _save_npz(os.path.join(session_dir, "pred_map_3d.npz"), pred_map.numpy())
    _save_npz(os.path.join(session_dir, "gt_map_3d.npz"), gt_map.numpy())
    _save_npz(os.path.join(session_dir, "context_map_3d.npz"), context_map.numpy())
    _save_npz(os.path.join(session_dir, "target_energy_map_slab.npz"), slab_energy["energy_rel_sym"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_raw_map_slab.npz"), slab_energy["energy_raw"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_rel_gt_map_slab.npz"), slab_energy["energy_rel_gt"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_cosine_map_slab.npz"), slab_energy["energy_cosine"].numpy())
    print(f"[{config_name}] saved 3D inference artifacts")
    return session_dir


def run_full_volume_inference_3d(
    *,
    model,
    cdd_cache: dict,
    session_dir: str,
    config_name: str,
    device: torch.device,
    slab_depth: int = 8,
    overlap: int = 4,
    post_log_transform: bool = True,
    log_eps: float = 1.0,
    cdd_log_std_floor_mult: float = 0.05,
) -> str:
    """Run encoder on the full precomputed CDD volume and save context map.

    Slides the encoder over the depth axis in overlapping slabs, stitches
    results by averaging in the overlap region. Saves the full (C, D, H, W)
    context_map_3d as .npz.
    """
    if not cdd_cache:
        print(f"[{config_name}] full-volume inference: no CDD cache, skipping")
        return session_dir

    model.eval()
    with torch.no_grad():
        for (path, _), cdd_vol in cdd_cache.items():
            # cdd_vol: (S, D, H, W) numpy float32
            s, d, h, w = cdd_vol.shape
            vol_t = torch.from_numpy(cdd_vol).unsqueeze(0).to(device)  # (1, S, D, H, W)

            # Apply post_log_transform to match training input distribution
            if post_log_transform:
                eps_f = max(1e-6, float(log_eps))
                vol_clamp = vol_t.clamp(min=0.0)
                base_std = torch.std(vol_clamp, dim=(-3, -2, -1), keepdim=True)
                log_floor = torch.clamp(base_std * float(cdd_log_std_floor_mult), min=eps_f)
                vol_t = torch.log(vol_clamp + log_floor)

            full_volume_training = bool(getattr(model, "full_volume_training", False))
            target_depth = d if full_volume_training else max(1, min(int(getattr(model, "slab_depth", slab_depth)), slab_depth))
            context_margin = 0 if full_volume_training else max(0, (slab_depth - target_depth) // 2)
            target_overlap = min(max(0, int(overlap)), max(0, target_depth - 1))
            step = max(1, target_depth - target_overlap)
            starts = list(range(0, max(1, d - slab_depth + 1), step))
            tail_start = max(0, d - slab_depth)
            if not starts or starts[-1] != tail_start:
                starts.append(tail_start)

            ctx_sum = None
            ctx_weight = torch.zeros((1, 1, d, 1, 1), device=device)

            for z0 in starts:
                ze = min(z0 + slab_depth, d)
                slab = vol_t[:, :, z0:ze]  # (1, S, slab, H, W)
                valid_depth = int(ze - z0)
                if slab.shape[2] < slab_depth:
                    pad_needed = slab_depth - slab.shape[2]
                    pad_mode = "reflect" if slab.shape[2] > 1 else "replicate"
                    slab = F.pad(slab, (0, 0, 0, 0, 0, pad_needed), mode=pad_mode)

                mask_tokens = torch.zeros_like(slab)
                ctx = model.context_encoder(slab, mask_tokens=mask_tokens)[:, :, :valid_depth]
                if ctx_sum is None:
                    ctx_sum = torch.zeros((1, int(ctx.shape[1]), d, int(ctx.shape[3]), int(ctx.shape[4])), device=device)
                write_rel0 = 0 if z0 == 0 else min(context_margin, valid_depth)
                write_rel1 = valid_depth if ze == d else max(write_rel0, min(valid_depth, slab_depth - context_margin))
                if write_rel1 <= write_rel0:
                    continue
                out_z0 = z0 + write_rel0
                out_z1 = z0 + write_rel1
                ctx_sum[:, :, out_z0:out_z1] += ctx[:, :, write_rel0:write_rel1]
                ctx_weight[:, :, out_z0:out_z1] += 1.0

            if ctx_sum is None:
                raise RuntimeError(f"[{config_name}] full-volume inference produced no slabs for {path}")
            ctx_avg = ctx_sum / ctx_weight.clamp_min(1.0)

            name = os.path.basename(path).replace('.npy', '')
            out_path = os.path.join(session_dir, f"{name}_context_map_3d.npz")
            _save_npz(out_path, ctx_avg.squeeze(0).cpu().numpy())
            print(f"[{config_name}] full-volume context map saved: {out_path}")

    return session_dir


================================================================================
FILE: src/losses.py
================================================================================

from __future__ import annotations

import torch
import torch.nn.functional as F


def l2_normalize_patches(tensor: torch.Tensor) -> torch.Tensor:
    """L2-normalize per-target patch/cube tensors across all non-B/K dimensions."""
    if tensor.dim() < 3:
        raise ValueError(f"Expected tensor with B,K,... shape, got {tuple(tensor.shape)}")
    b, k = tensor.shape[:2]
    patch_shape = tensor.shape[2:]
    return F.normalize(tensor.reshape(b, k, -1), dim=2).reshape(b, k, *patch_shape)


def parse_spread_regularizer_config(train_cfg: dict) -> dict[str, float | str]:
    removed_flat_keys = {
        "spread_regularizer_weight",
        "spread_sketch_dim",
        "embed_spread_target",
        "spread_regularizer_eps",
        "spread_regularizer_noise_std",
    }
    stale_keys = removed_flat_keys & set(train_cfg)
    assert not stale_keys, f"Use train.spread_regularizer instead of flat keys: {sorted(stale_keys)}"
    cfg = dict(train_cfg.get("spread_regularizer", {}))
    expected_keys = {"type", "target", "weight", "target_std", "eps", "sketch_dim", "sketch_seed"}
    extra_keys = set(cfg) - expected_keys
    assert not extra_keys, f"Unsupported train.spread_regularizer keys: {sorted(extra_keys)}"
    sigreg_type = str(cfg.get("type", "std_hinge"))
    sigreg_target = str(cfg.get("target", "context"))
    sigreg_weight = float(cfg.get("weight", 0.0))
    assert sigreg_type in {"std_hinge", "weak_sigreg", "sketched_sigreg"}
    assert sigreg_target == "context"
    assert sigreg_weight >= 0
    target_std = float(cfg.get("target_std", 1.0))
    eps = float(cfg.get("eps", 1e-4))
    sketch_dim = int(cfg.get("sketch_dim", 64))
    sketch_seed = int(cfg.get("sketch_seed", 0))
    assert target_std >= 0
    assert eps > 0
    assert sketch_dim > 0
    assert sketch_seed >= 0
    return {
        "type": sigreg_type,
        "target": sigreg_target,
        "weight": sigreg_weight,
        "target_std": target_std,
        "eps": eps,
        "sketch_dim": sketch_dim,
        "sketch_seed": sketch_seed,
    }


def _offdiag(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("offdiag expects square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _reshape_patch_pairs(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    spatial_mode: str = "dense",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten B×K×C×P×P (or ×P×P×P) to (N, C) tokens, filter by valid."""
    if pred.dim() not in (5, 6):
        raise ValueError(f"Expected pred/gt rank 5 or 6, got {pred.dim()}")
    b, k, c = pred.shape[:3]
    spatial_shape = pred.shape[3:]
    spatial_n = 1
    for s in spatial_shape:
        spatial_n *= int(s)
    mode = str(spatial_mode).lower()
    if mode not in ("dense", "pooled"):
        raise ValueError(f"Unsupported spatial_mode={spatial_mode}. Use 'dense' or 'pooled'.")
    if mode == "pooled":
        reduce_dims = tuple(range(3, pred.dim()))
        pred_v = pred.mean(dim=reduce_dims)
        gt_v = gt.mean(dim=reduce_dims)
        vm = valid.reshape(-1)
        z1 = pred_v.reshape(-1, c)[vm]
        z2 = gt_v.reshape(-1, c)[vm]
        return z1, z2
    # dense mode: keep spatial tokens
    if pred.dim() == 5:
        pred_v = pred.permute(0, 1, 3, 4, 2).reshape(b, k, spatial_n, c)
        gt_v = gt.permute(0, 1, 3, 4, 2).reshape(b, k, spatial_n, c)
    else:
        pred_v = pred.permute(0, 1, 3, 4, 5, 2).reshape(b, k, spatial_n, c)
        gt_v = gt.permute(0, 1, 3, 4, 5, 2).reshape(b, k, spatial_n, c)
    vm = valid.unsqueeze(-1).unsqueeze(-1).expand(b, k, spatial_n, 1).reshape(-1)
    z1 = pred_v.reshape(-1, c)[vm]
    z2 = gt_v.reshape(-1, c)[vm]
    return z1, z2


def extract_valid_pooled_embeddings(outputs: dict, key: str = "context_patches") -> torch.Tensor:
    patches = outputs[key]  # Pre-predictor context embeddings by default: B,K,C,...
    valid = outputs["target_valid"]  # B,K
    _, _, c = patches.shape[:3]
    pooled = patches.mean(dim=tuple(range(3, patches.dim())))  # B,K,C
    vm = valid.reshape(-1)
    z = pooled.reshape(-1, c)[vm]
    return z


def extract_valid_dense_embeddings(outputs: dict, key: str = "context_patches") -> torch.Tensor:
    patches = outputs[key]  # B,K,C,... spatial patch tokens
    valid = outputs["target_valid"]  # B,K
    if patches.dim() not in (5, 6):
        raise ValueError(f"Expected {key} rank 5 or 6, got {patches.dim()}")
    b, k, c = patches.shape[:3]
    spatial_shape = patches.shape[3:]
    spatial_n = 1
    for size in spatial_shape:
        spatial_n *= int(size)
    if patches.dim() == 5:
        z = patches.permute(0, 1, 3, 4, 2).reshape(b, k, spatial_n, c)
    else:
        z = patches.permute(0, 1, 3, 4, 5, 2).reshape(b, k, spatial_n, c)
    vm = valid.unsqueeze(-1).unsqueeze(-1).expand(b, k, spatial_n, 1).reshape(-1)
    return z.reshape(-1, c)[vm]


def spread_regularizer_loss(
    z: torch.Tensor,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Standard-deviation hinge on context embeddings.
    Gradients remain useful close to collapse without hidden projection modes.
    """
    z = z.float()  # cast to fp32 to avoid underflow in fp16
    if z.numel() == 0 or z.shape[0] < 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    n = z.shape[0]
    z = z - z.mean(dim=0, keepdim=True)
    # Escape jitter: tiny noise prevents zero-gradient trap at perfect collapse.
    escape_jitter = max(float(eps) ** 0.5, 1e-3)
    z_escape = z + escape_jitter * torch.randn_like(z)
    std = torch.sqrt(z_escape.var(dim=0, unbiased=False) + float(eps))
    loss = torch.relu(float(target_std) - std).mean()

    effective_n = 256.0
    if n > effective_n:
        grad_scale = float(n) / effective_n
        loss = loss * grad_scale - loss.detach() * (grad_scale - 1.0)

    return loss


def weak_sigreg_loss(
    z: torch.Tensor,
    target_std: float = 1.0,
    sketch_dim: int = 64,
    eps: float = 1e-4,
    sketch_seed: int = 0,
) -> torch.Tensor:
    """Sketched isotropic Gaussian regularization on context embeddings.

    Applies the variance hinge on the full embedding dimension for consistent
    anti-collapse gradients, then uses a rotating sketch only for the cheaper
    off-diagonal covariance penalty.
    """
    del sketch_seed  # Kept as a config key for reproducibility metadata/backward compatibility.
    z = z.float()
    if z.numel() == 0 or z.shape[0] < 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    n, c = z.shape
    z = z - z.mean(dim=0, keepdim=True)

    # Escape jitter: tiny noise prevents zero-gradient trap at perfect collapse.
    escape_jitter = max(float(eps) ** 0.5, 1e-3)
    z_escape = z + escape_jitter * torch.randn_like(z)
    std_full = torch.sqrt(z_escape.var(dim=0, unbiased=False) + float(eps))
    var_loss = torch.relu(float(target_std) - std_full).mean()

    k = min(int(sketch_dim), int(c))
    if c > k:
        sketch = torch.randn(c, k, device=z.device, dtype=z.dtype)
        sketch, _ = torch.linalg.qr(sketch, mode="reduced")
        z_proj = z @ sketch
    else:
        z_proj = z

    cov = (z_proj.t() @ z_proj) / (float(n - 1) + float(eps))
    cov = cov - torch.diag(cov.diag())
    cov_loss = cov.pow(2).sum() / float(k)
    loss = var_loss + cov_loss

    effective_n = 256.0
    if n > effective_n:
        grad_scale = float(n) / effective_n
        loss = loss * grad_scale - loss.detach() * (grad_scale - 1.0)

    return loss


def sketched_sigreg_loss(
    z: torch.Tensor,
    target_std: float = 1.0,
    sketch_dim: int = 64,
    eps: float = 1e-6,
    sketch_seed: int = 0,
) -> torch.Tensor:
    del sketch_seed  # Preserve config shape; projection is intentionally stochastic.
    z = z.float()
    if z.numel() == 0 or z.shape[0] < 2:
        return z.sum() * 0.0

    z = z - z.mean(dim=0, keepdim=True)
    c = z.shape[1]
    sketch_dim = int(max(1, sketch_dim))
    a = torch.randn((c, sketch_dim), device=z.device, dtype=z.dtype)
    a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-6)
    y = z @ a
    projected_var_loss = (y.var(dim=0, unbiased=False) - float(target_std) ** 2).pow(2).mean()

    # The historical projected variance loss has zero gradient at exact
    # collapse because d(var(z @ A))/dz is proportional to z. Keep the old
    # isotropic pressure, but add a direct std hinge so near-zero embeddings
    # have an escape gradient instead of sitting at loss ~= 1 forever.
    # If all target embeddings are exactly identical, std(z) is high-loss but
    # zero-gradient. Add tiny loss-local jitter so collapsed dimensions get a
    # deterministic escape direction through the std hinge; the jitter is not
    # applied to the model outputs or diagnostics.
    escape_jitter = max(float(eps) ** 0.5, 1e-3)
    z_escape = z + escape_jitter * torch.randn_like(z)
    std_full = torch.sqrt(z_escape.var(dim=0, unbiased=False) + float(eps))
    escape_loss = torch.relu(float(target_std) - std_full).mean()

    std_corr = torch.sqrt(z.var(dim=0, unbiased=False) + float(eps))
    z_corr = z / std_corr.clamp_min(float(eps)).unsqueeze(0)
    corr = (z_corr.t() @ z_corr) / float(max(1, z.shape[0] - 1))
    corr = corr - torch.diag(corr.diag())
    if c > 1:
        decorrelation_loss = corr.pow(2).sum() / float(c * (c - 1))
    else:
        decorrelation_loss = corr.sum() * 0.0
    return projected_var_loss + escape_loss + decorrelation_loss


def compute_spread_regularizer_loss(z: torch.Tensor, cfg: dict[str, float | str]) -> torch.Tensor:
    sigreg_type = str(cfg.get("type", "std_hinge"))
    if sigreg_type == "std_hinge":
        return spread_regularizer_loss(
            z,
            target_std=float(cfg.get("target_std", 1.0)),
            eps=float(cfg.get("eps", 1e-4)),
        )
    if sigreg_type == "weak_sigreg":
        return weak_sigreg_loss(
            z,
            target_std=float(cfg.get("target_std", 1.0)),
            sketch_dim=int(cfg.get("sketch_dim", 64)),
            eps=float(cfg.get("eps", 1e-4)),
            sketch_seed=int(cfg.get("sketch_seed", 0)),
        )
    if sigreg_type == "sketched_sigreg":
        return sketched_sigreg_loss(
            z,
            target_std=float(cfg.get("target_std", 1.0)),
            sketch_dim=int(cfg.get("sketch_dim", 64)),
            eps=float(cfg.get("eps", 1e-6)),
            sketch_seed=int(cfg.get("sketch_seed", 0)),
        )
    raise ValueError(f"Unsupported spread regularizer type: {sigreg_type}")


def compute_output_spread_regularizer_loss(
    outputs: dict,
    cfg: dict[str, float | str],
    *,
    include_predictor: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "context_patches" not in outputs:
        raise KeyError("SIGReg requires outputs['context_patches']; refusing to regularize predictor-only outputs")
    z_ctx = extract_valid_pooled_embeddings(outputs, key="context_patches")
    loss = compute_spread_regularizer_loss(z_ctx, cfg)
    if include_predictor:
        z_pred = extract_valid_pooled_embeddings(outputs, key="pred_patches")
        loss = loss + compute_spread_regularizer_loss(z_pred, cfg)
    return loss, z_ctx


@torch.no_grad()
def embedding_spread_stats(
    z: torch.Tensor,
    target_std: float = 1.0,
    dead_channel_threshold: float = 1e-5,
) -> dict[str, float]:
    """Compact collapse diagnostics for pooled context embeddings."""
    z = z.detach().float()
    if z.numel() == 0:
        return {
            "embed_spread_mean": 0.0,
            "embed_spread_min": 0.0,
            "embed_under_spread_frac": 1.0,
            "dead_channel_count": 0,
            "context_manifold_size": 0.0,
        }
    z = z - z.mean(dim=0, keepdim=True)
    var = z.var(dim=0, unbiased=False)
    cov = (z.T @ z) / max(1, int(z.shape[0]))
    eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    eig_sum = eig.sum()
    rank = 0.0
    if float(eig_sum.item()) > 1e-20:
        p = eig / eig_sum
        p = p[p > 1e-20]
        if p.numel() > 0:
            rank = float(torch.exp(-(p * p.log()).sum()).item())
    std = torch.sqrt(var + 1e-12)
    return {
        "embed_spread_mean": float(std.mean().item()),
        "embed_spread_min": float(std.min().item()),
        "embed_under_spread_frac": float((std < float(target_std)).float().mean().item()),
        "dead_channel_count": int((std < float(dead_channel_threshold)).sum().item()),
        "context_manifold_size": rank,
    }


def _compute_sim_var_cov_tensors(
    outputs: dict,
    spatial_mode: str = "dense",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = outputs["pred_patches"].float()  # keep graph, cast to fp32 for safe norm
    ctx = outputs.get("context_patches", outputs["pred_patches"]).float()  # regularize before predictor
    gt = outputs["gt_patches"].float()  # keep graph (target branch already no-grad in forward)
    valid = outputs["target_valid"]

    z_pred, z_gt = _reshape_patch_pairs(pred, gt, valid, spatial_mode=spatial_mode)
    z_ctx, _ = _reshape_patch_pairs(ctx, gt, valid, spatial_mode=spatial_mode)
    if z_pred.numel() == 0 or z_gt.numel() == 0:
        z = pred.sum() * 0.0
        return z, z, z

    sim = torch.nn.functional.cosine_similarity(z_pred, z_gt, dim=1).mean()
    if z_pred.shape[0] < 2:
        z = sim * 0.0
        return sim, z, z

    # Escape jitter: tiny noise prevents zero-gradient trap at perfect collapse.
    escape_jitter = 1e-2
    z_ctx_esc = z_ctx + escape_jitter * torch.randn_like(z_ctx)
    z_gt_esc = z_gt + escape_jitter * torch.randn_like(z_gt)
    std_ctx = torch.sqrt(z_ctx_esc.var(dim=0, unbiased=False) + 1e-4)
    std_gt = torch.sqrt(z_gt_esc.var(dim=0, unbiased=False) + 1e-4)
    var_term = 0.5 * (torch.relu(1.0 - std_ctx).mean() + torch.relu(1.0 - std_gt).mean())

    z1c = z_ctx - z_ctx.mean(dim=0, keepdim=True)
    z2c = z_gt - z_gt.mean(dim=0, keepdim=True)
    cov_z1 = (z1c.T @ z1c) / max(1, z1c.shape[0] - 1)
    cov_z2 = (z2c.T @ z2c) / max(1, z2c.shape[0] - 1)
    dim = max(1, int(cov_z1.shape[0]))
    cov_term = 0.5 * ((_offdiag(cov_z1).pow(2).sum() / dim) + (_offdiag(cov_z2).pow(2).sum() / dim))
    return sim, var_term, cov_term


def compute_sim_var_cov(outputs: dict, spatial_mode: str = "dense") -> tuple[float, float, float]:
    values = _compute_sim_var_cov_tensors(outputs, spatial_mode=spatial_mode)
    return tuple(float(value.detach().item()) for value in values)


def compute_sim_var_cov_torch(outputs: dict, spatial_mode: str = "dense") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _compute_sim_var_cov_tensors(outputs, spatial_mode=spatial_mode)


def compute_raw_mse_and_norm_err(outputs: dict) -> tuple[float, float]:
    pred = outputs["pred_patches"].detach().float()
    gt = outputs["gt_patches"].detach().float()
    valid = outputs["target_valid"].detach()
    z1, z2 = _reshape_patch_pairs(pred, gt, valid, spatial_mode="dense")
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
        b, k = pred.shape[:2]
        patch_shape = pred.shape[2:]
        pred = torch.nn.functional.normalize(pred.reshape(b, k, -1), dim=2).reshape(b, k, *patch_shape)
        gt = torch.nn.functional.normalize(gt.reshape(b, k, -1), dim=2).reshape(b, k, *patch_shape)
    reduce_dims = tuple(range(2, pred.dim()))
    energy_per_target = (pred - gt).pow(2).mean(dim=reduce_dims)
    if bool(valid.any()):
        return float(energy_per_target[valid].mean().item())
    return 0.0


def representation_dense_energy(pred_map: torch.Tensor, gt_map: torch.Tensor, eps: float = 1e-8) -> dict[str, torch.Tensor]:
    diff = pred_map - gt_map
    diff2 = diff.pow(2)

    raw = diff2.mean(dim=1, keepdim=True)

    gt_norm2 = gt_map.pow(2).sum(dim=1, keepdim=True)
    pred_norm2 = pred_map.pow(2).sum(dim=1, keepdim=True)

    rel_gt = diff2.sum(dim=1, keepdim=True) / gt_norm2.clamp_min(eps)
    rel_sym = diff2.sum(dim=1, keepdim=True) / (0.5 * (gt_norm2 + pred_norm2)).clamp_min(eps)

    cos = (1.0 - F.cosine_similarity(pred_map, gt_map, dim=1, eps=eps).unsqueeze(1)).clamp_min(0.0)

    return {
        "energy_raw": raw,
        "energy_rel_gt": rel_gt,
        "energy_rel_sym": rel_sym,
        "energy_cosine": cos,
    }



def compute_target_energy_map(outputs: dict, image_size: tuple[int, int]) -> dict[str, torch.Tensor]:
    pred_map = outputs["pred_map"]
    gt_map = outputs["gt_map"].detach()
    h, w = int(image_size[0]), int(image_size[1])

    result = representation_dense_energy(pred_map, gt_map)

    for key in result:
        if result[key].shape[-2:] != (h, w):
            result[key] = F.interpolate(result[key], size=(h, w), mode="bilinear", align_corners=False)
    return result


================================================================================
FILE: src/dataset3d.py
================================================================================

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.npy import _safe_load_npy, normalize01


class JEPA3DCropDataset(Dataset):
    """3D dataset that consumes precomputed CDD cache entries.

    Each cache entry is (S, D, H, W) — S CDD scale channels over a full volume.
    The dataset randomly selects a slice axis (X/Y/Z), rotates the CDD data so
    that axis becomes the depth dimension, then extracts a context crop of size
    (S, crop_depth, crop_size, crop_size).

    When no CDD cache is provided (raw mode), falls back to loading raw .npy
    volumes, normalizing to [0,1], and treating the single channel as S=1.
    """

    def __init__(
        self,
        data_root: str = "data",
        npy_pattern: str = "*.npy",
        num_samples: int = 1000,
        crop_size: int = 64,
        crop_depth: int | None = None,
        slab_depth: int | None = None,
        depth_axis: int = 0,
        random_axis: bool = False,
        normalize: bool = True,
        crop_strategy: str = "random",
        cdd_cache: dict | None = None,
    ):
        import glob
        self.npy_files = sorted(glob.glob(os.path.join(data_root, npy_pattern)))
        if not self.npy_files:
            raise FileNotFoundError(f"No .npy files found in {data_root}/{npy_pattern}")

        self.num_samples = int(num_samples)
        self.crop_size = int(crop_size)
        self.slab_depth = int(slab_depth) if slab_depth is not None else int(crop_size)
        self.crop_depth = int(crop_depth) if crop_depth is not None else self.slab_depth
        self.depth_axis = int(depth_axis) % 3
        self.random_axis = bool(random_axis)
        self.normalize = bool(normalize)
        self.crop_strategy = str(crop_strategy).lower()
        if self.crop_strategy not in ("random", "center", "mixed"):
            raise ValueError("crop_strategy must be one of: random, center, mixed")
        self.cdd_cache = cdd_cache

    def __len__(self):
        return self.num_samples

    @staticmethod
    def _normalize01(x: np.ndarray) -> np.ndarray:
        return normalize01(x)

    def _choose_axis(self) -> int:
        if self.random_axis:
            return int(np.random.randint(0, 3))
        return self.depth_axis

    def _orient_to_axis(self, arr: np.ndarray, axis: int) -> np.ndarray:
        """Move the chosen axis to position 1 (after scale dim).

        Input: (S, X, Y, Z) where S=scale channels.
        axis 0: X-depth → (S, X, Y, Z)  [D=Y, H=W=side]
        axis 1: Y-depth → (S, Y, X, Z)  [swap X↔Y]
        axis 2: Z-depth → (S, Z, X, Y)  [move Z to front after S]
        """
        if axis == 0:
            return arr  # already S,X,Y,Z
        elif axis == 1:
            return arr.transpose(0, 2, 1, 3)  # S,Y,X,Z
        else:
            return arr.transpose(0, 3, 1, 2)  # S,Z,X,Y

    def _pad_to_crop_shape(self, arr: np.ndarray) -> np.ndarray:
        cd = int(self.crop_depth)
        cs = int(self.crop_size)
        pads = []
        for size, target in zip(arr.shape, (arr.shape[0], cd, cs, cs)):
            missing = max(0, int(target) - int(size))
            before = missing // 2
            pads.append((before, missing - before))
        if not any(before or after for before, after in pads):
            return arr
        mode = "reflect" if min(arr.shape[1:]) > 1 else "edge"
        return np.pad(arr, tuple(pads), mode=mode)

    def _crop_context(self, arr: np.ndarray) -> np.ndarray:
        """Extract a context crop along depth, height, and width.

        Input: (S, D, H, W). Returns: (S, crop_depth, crop_size, crop_size).
        """
        arr = self._pad_to_crop_shape(arr)
        cd = int(self.crop_depth)
        cs = int(self.crop_size)
        _, d, h, w = arr.shape
        if self.crop_strategy == "center" or (self.crop_strategy == "mixed" and np.random.rand() < 0.5):
            z0 = max(0, (d - cd) // 2)
            y0 = max(0, (h - cs) // 2)
            x0 = max(0, (w - cs) // 2)
        else:
            z0 = np.random.randint(0, d - cd + 1)
            y0 = np.random.randint(0, h - cs + 1)
            x0 = np.random.randint(0, w - cs + 1)
        return arr[:, z0:z0 + cd, y0:y0 + cs, x0:x0 + cs]

    def _get_raw_volume(self, idx: int) -> np.ndarray:
        """Fallback: load raw .npy, normalize, return as (1, D, H, W)."""
        path = self.npy_files[idx % len(self.npy_files)]
        arr = _safe_load_npy(path, mmap_mode="r")
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array D,H,W, got shape={arr.shape} in {path}")
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if self.normalize:
            arr = self._normalize01(arr)
        return arr[np.newaxis, ...]  # (1, D, H, W)

    def _get_cdd_volume(self, idx: int) -> np.ndarray:
        """Load CDD precomputed volume from cache. Returns (S, D, H, W)."""
        path = self.npy_files[idx % len(self.npy_files)]
        key = (path, None)
        if key in self.cdd_cache:
            return self.cdd_cache[key].copy()  # (S, D, H, W)
        # Fallback
        return self._get_raw_volume(idx)

    def __getitem__(self, idx):
        if self.cdd_cache is not None:
            vol = self._get_cdd_volume(idx)
        else:
            vol = self._get_raw_volume(idx)

        axis = self._choose_axis()
        vol = self._orient_to_axis(vol, axis)  # (S, D, H, W) with chosen axis as D
        slab = self._crop_context(vol)  # (S, crop_depth, crop_size, crop_size)

        return torch.from_numpy(slab.astype(np.float32))


================================================================================
FILE: src/utils/scale_probe.py
================================================================================

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _get_context_encoder(model):
    if hasattr(model, "context_encoder"):
        return model.context_encoder
    if hasattr(model, "encoder"):
        return model.encoder
    raise AttributeError("Model has neither context_encoder nor encoder")


def _encode_context(model, x_pyr, mask_tokens=None):
    encoder = _get_context_encoder(model)
    if mask_tokens is None:
        try:
            return encoder(x_pyr)
        except TypeError:
            return encoder(x_pyr, mask_tokens=torch.zeros_like(x_pyr))
    try:
        return encoder(x_pyr, mask_tokens=mask_tokens)
    except TypeError:
        try:
            return encoder(x_pyr, mask_tokens)
        except TypeError:
            return encoder(torch.cat([x_pyr, mask_tokens], dim=1))


@torch.no_grad()
def _normalize_channel_map(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return z / (z.norm(dim=1, keepdim=True) + eps)


@torch.no_grad()
def _effective_rank(z: torch.Tensor, max_points: int = 50000, eps: float = 1e-12) -> float:
    if z.ndim < 4:
        raise ValueError(f"Expected dense feature map B,C,... got {tuple(z.shape)}")
    c = z.shape[1]
    x = z.detach().float().permute(0, *range(2, z.ndim), 1).reshape(-1, c)
    if x.shape[0] > max_points:
        idx = torch.randperm(x.shape[0], device=x.device)[:max_points]
        x = x[idx]
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    evals = torch.linalg.eigvalsh(cov).clamp_min(0)
    p = evals / evals.sum().clamp_min(eps)
    entropy = -(p * (p + eps).log()).sum()
    return float(torch.exp(entropy).item())


@torch.no_grad()
def probe_scale_response(
    model,
    x_pyr: torch.Tensor,
    mask_tokens: Optional[torch.Tensor] = None,
    scale_names: Optional[Sequence[str]] = None,
    out_dir: str | Path = "scale_response_report",
    run_name: str = "probe",
    include_predictor: bool = True,
    max_rank_points: int = 50000,
) -> Dict:
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if x_pyr.ndim != 4:
        raise ValueError(f"Expected x_pyr B,S,H,W, got {tuple(x_pyr.shape)}")

    B, S, H, W = x_pyr.shape
    if scale_names is None:
        scale_names = [f"scale_{i}" for i in range(S)]
    if len(scale_names) != S:
        raise ValueError(f"scale_names length {len(scale_names)} != S={S}")

    if mask_tokens is None:
        mask_tokens = torch.zeros_like(x_pyr)

    z_full = _encode_context(model, x_pyr, mask_tokens=mask_tokens)
    z_full_n = _normalize_channel_map(z_full)

    pred_full = None
    pred_full_n = None
    if include_predictor and hasattr(model, "predictor"):
        try:
            z_in_full = model.projector(z_full) if hasattr(model, "projector") else z_full
            pred_full = model.predictor(z_in_full)
            pred_full_n = _normalize_channel_map(pred_full)
        except Exception:
            pred_full = None
            pred_full_n = None

    sensitivity_maps = []
    scale_only_sim_maps = []
    pred_sensitivity_maps = []

    def _restore_input_hw(map_bhw: torch.Tensor) -> torch.Tensor:
        if map_bhw.shape[-2:] == (H, W):
            return map_bhw
        return F.interpolate(
            map_bhw.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    for s in range(S):
        x_drop = x_pyr.clone()
        x_drop[:, s] = 0.0
        z_drop = _encode_context(model, x_drop, mask_tokens=mask_tokens)
        z_drop_n = _normalize_channel_map(z_drop)

        diff = (z_full_n - z_drop_n).pow(2).sum(dim=1).sqrt()
        sensitivity_maps.append(_restore_input_hw(diff))

        if pred_full_n is not None:
            z_in_drop = model.projector(z_drop) if hasattr(model, "projector") else z_drop
            pred_drop = model.predictor(z_in_drop)
            pred_drop_n = _normalize_channel_map(pred_drop)
            pred_diff = (pred_full_n - pred_drop_n).pow(2).sum(dim=1).sqrt()
            pred_sensitivity_maps.append(_restore_input_hw(pred_diff))

        x_one = torch.zeros_like(x_pyr)
        x_one[:, s] = x_pyr[:, s]
        z_one = _encode_context(model, x_one, mask_tokens=mask_tokens)
        z_one_n = _normalize_channel_map(z_one)
        sim = (z_full_n * z_one_n).sum(dim=1)
        scale_only_sim_maps.append(_restore_input_hw(sim))

    sensitivity_maps = torch.stack(sensitivity_maps, dim=1)  # B,S,H,W
    scale_only_sim_maps = torch.stack(scale_only_sim_maps, dim=1)

    if pred_sensitivity_maps:
        pred_sensitivity_maps = torch.stack(pred_sensitivity_maps, dim=1)
    else:
        pred_sensitivity_maps = None

    sens_global = sensitivity_maps.mean(dim=(0, 2, 3))
    sim_global = scale_only_sim_maps.mean(dim=(0, 2, 3))
    pred_sens_global = None
    if pred_sensitivity_maps is not None:
        pred_sens_global = pred_sensitivity_maps.mean(dim=(0, 2, 3))

    sens_frac = sens_global / sens_global.sum().clamp_min(1e-12)
    if pred_sens_global is not None:
        pred_sens_frac = pred_sens_global / pred_sens_global.sum().clamp_min(1e-12)
    else:
        pred_sens_frac = None

    winner_map = sensitivity_maps[0].argmax(dim=0)  # H,W — per-location dominant scale

    report = {
        "run_name": run_name,
        "input_shape": list(x_pyr.shape),
        "feature_shape": list(z_full.shape),
        "scale_names": list(scale_names),
        "context_effective_rank": _effective_rank(z_full, max_points=max_rank_points),
        "scale_drop_sensitivity": {
            name: float(sens_global[i].item()) for i, name in enumerate(scale_names)
        },
        "scale_drop_sensitivity_fraction": {
            name: float(sens_frac[i].item()) for i, name in enumerate(scale_names)
        },
        "scale_only_similarity_to_full": {
            name: float(sim_global[i].item()) for i, name in enumerate(scale_names)
        },
        "dominant_context_scale": scale_names[int(torch.argmax(sens_global).item())],
    }

    if pred_full is not None:
        report["predictor_effective_rank"] = _effective_rank(pred_full, max_points=max_rank_points)
    if pred_sens_global is not None:
        report["pred_scale_drop_sensitivity"] = {
            name: float(pred_sens_global[i].item()) for i, name in enumerate(scale_names)
        }
        report["pred_scale_drop_sensitivity_fraction"] = {
            name: float(pred_sens_frac[i].item()) for i, name in enumerate(scale_names)
        }
        report["dominant_pred_scale"] = scale_names[int(torch.argmax(pred_sens_global).item())]

    save_obj = {
        "sensitivity_maps": sensitivity_maps.detach().cpu(),
        "scale_only_sim_maps": scale_only_sim_maps.detach().cpu(),
        "winner_map": winner_map.detach().cpu(),
        "input_map": x_pyr[0].sum(dim=0).detach().cpu(),
        "z_full": z_full.detach().cpu(),
    }
    if pred_sensitivity_maps is not None:
        save_obj["pred_sensitivity_maps"] = pred_sensitivity_maps.detach().cpu()
    if pred_full is not None:
        save_obj["pred_full"] = pred_full.detach().cpu()

    torch.save(save_obj, out_dir / f"{run_name}_scale_response.pt")

    with open(out_dir / f"{run_name}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    lines = []
    lines.append(f"Scale-response report: {run_name}")
    lines.append(f"input shape:   {tuple(x_pyr.shape)}")
    lines.append(f"feature shape: {tuple(z_full.shape)}")
    lines.append(f"context effective rank: {report['context_effective_rank']:.4f}")
    if "predictor_effective_rank" in report:
        lines.append(f"predictor effective rank: {report['predictor_effective_rank']:.4f}")
    lines.append("")
    lines.append("Context scale-drop sensitivity:")
    for name in scale_names:
        val = report["scale_drop_sensitivity"][name]
        frac = report["scale_drop_sensitivity_fraction"][name]
        simv = report["scale_only_similarity_to_full"][name]
        lines.append(f"  {name:>12s}: drop={val:.6f}  frac={frac:.3f}  only_sim={simv:.3f}")
    lines.append(f"dominant context scale: {report['dominant_context_scale']}")
    if "pred_scale_drop_sensitivity" in report:
        lines.append("")
        lines.append("Predictor scale-drop sensitivity:")
        for name in scale_names:
            val = report["pred_scale_drop_sensitivity"][name]
            frac = report["pred_scale_drop_sensitivity_fraction"][name]
            lines.append(f"  {name:>12s}: drop={val:.6f}  frac={frac:.3f}")
        lines.append(f"dominant pred scale: {report['dominant_pred_scale']}")

    with open(out_dir / f"{run_name}_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    return report


================================================================================
FILE: src/utils/memory.py
================================================================================

"""Auto-batch memory management for sajepa — OOM-safe training with dynamic batch scaling."""

from __future__ import annotations

import gc
import math
from typing import Optional

import torch


def clear_memory_cache():
    """Flush accumulated tensors from system cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _next_power_of_two_batch(current: int) -> int:
    """Return the next lower power-of-two batch size, clamped to minimum 1."""
    if current <= 1:
        return 1
    return max(1, 2 ** int(math.log2(current - 1)))


def auto_batch_size(
    initial_batch: int = 4,
    target_batch: int = 32,
    scale_mode: str = "power_of_two",
    max_retries: int = 5,
) -> int:
    """Determine a safe batch size, retrying with smaller sizes on OOM.

    Args:
        initial_batch: Starting batch size to try.
        target_batch: Desired effective batch size (for accumulation).
        scale_mode: "power_of_two" halves each retry; "linear" subtracts 1.
        max_retries: Maximum number of OOM retries before giving up.

    Returns:
        Safe batch size (<= initial_batch).
    """
    return int(initial_batch)


def compute_accumulation_steps(
    batch_size: int,
    target_batch: int = 32,
) -> int:
    """Compute gradient accumulation steps to match target effective batch size.

    Args:
        batch_size: Actual per-step batch size (after OOM scaling).
        target_batch: Desired effective batch size.

    Returns:
        Number of accumulation steps (>= 1).
    """
    if batch_size <= 0:
        return 1
    steps = max(1, int(math.ceil(target_batch / batch_size)))
    return steps


class OOMSafeTrainer:
    """Wrapper that retries training with progressively smaller batch sizes on OOM.

    Usage:
        trainer = OOMSafeTrainer(initial_batch=4, target_batch=32)
        for attempt in trainer:
            try:
                run_training_loop(model, loader, optimizer, batch_size=trainer.batch_size)
                break  # success
            except RuntimeError as e:
                if not trainer.handle_oom(e):
                    raise
    """

    def __init__(
        self,
        initial_batch: int = 4,
        target_batch: int = 32,
        scale_mode: str = "power_of_two",
        max_retries: int = 5,
    ):
        self.initial_batch = int(initial_batch)
        self.target_batch = int(target_batch)
        self.scale_mode = str(scale_mode)
        self.max_retries = int(max_retries)
        self.batch_size = self.initial_batch
        self.accumulation_steps = compute_accumulation_steps(self.batch_size, self.target_batch)
        self._attempt = 0
        self._done = False

    def handle_oom(self, error: RuntimeError) -> bool:
        """Handle OOM error. Returns True if retry is possible, False if exhausted."""
        if "out of memory" not in str(error).lower():
            return False
        self._attempt += 1
        if self._attempt >= self.max_retries:
            return False
        clear_memory_cache()
        prev_batch = self.batch_size
        if self.scale_mode == "power_of_two":
            self.batch_size = _next_power_of_two_batch(self.batch_size)
        else:
            self.batch_size = max(1, self.batch_size - 1)
        self.accumulation_steps = compute_accumulation_steps(self.batch_size, self.target_batch)
        print(
            f"[sajepa] OOM at batch={prev_batch} → retrying batch={self.batch_size} "
            f"(accum={self.accumulation_steps}, attempt {self._attempt}/{self.max_retries})"
        )
        return True

    def __iter__(self):
        self._attempt = 0
        self.batch_size = self.initial_batch
        self.accumulation_steps = compute_accumulation_steps(self.batch_size, self.target_batch)
        self._done = False
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        self._done = True
        return self


================================================================================
FILE: src/utils/__init__.py
================================================================================

"""Shared utilities for the JEPA training pipeline."""

import traceback
import datetime
import os
from typing import Optional

_SESSION_ERROR_LOG_PATH: Optional[str] = None


def set_error_log_path(path: str) -> None:
    """Set the session-specific error log path for the current training run."""
    global _SESSION_ERROR_LOG_PATH
    _SESSION_ERROR_LOG_PATH = path


def log_error(tag: str, exc: Optional[Exception] = None) -> None:
    """Log an error with full traceback to the session error log and stdout.

    Args:
        tag: Short description of where the error occurred (e.g. 'effective_rank').
        exc: The exception, if any. If None, the current exception is captured.
    """
    global _SESSION_ERROR_LOG_PATH
    msg = _format_error(tag, exc)
    print(msg, end="")
    if _SESSION_ERROR_LOG_PATH is not None:
        try:
            os.makedirs(os.path.dirname(_SESSION_ERROR_LOG_PATH), exist_ok=True)
            with open(_SESSION_ERROR_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(msg)
        except Exception:
            pass  # can't log if the log itself fails


def _format_error(tag: str, exc: Optional[Exception] = None) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if exc is None:
        tb = traceback.format_exc().strip()
    else:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
    return f"[{ts}] [{tag}]\n{tb}\n\n"


================================================================================
FILE: src/utils/viz.py
================================================================================

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
            fit_jitter = fit_centered.astype(np.float64) + np.eye(fit_centered.shape[0], fit_centered.shape[1]) * 1e-5
            _, _, vt = np.linalg.svd(fit_jitter, full_matrices=False)
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

    if x.shape[0] > fit_max_tokens:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(x.shape[0], size=fit_max_tokens, replace=False)
        fit_x = x[idx]
        needs_fit_transform = True
    else:
        needs_fit_transform = False

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

    if n_components == 3:
        return _compute_pca_3d(x, fit_max_tokens=fit_max_tokens)
    return _compute_pca_2d(x, fit_max_tokens=fit_max_tokens)


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
    fit_max_tokens = int(umap_cfg.get("fit_max_tokens", 65536))
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
        latent_map = fmap[0].detach().cpu().numpy().astype(np.float32)
        z = np.transpose(latent_map, (1, 2, 0)).reshape(-1, fmap.shape[1]).astype(np.float32)

        # Filter invalid-region latents from PCA/UMAP, keep NaN sentinels.
        pca3 = _filtered_embedding(
            z, valid_mask_flat,
            lambda arr: _compute_pca_3d(arr, fit_max_tokens=fit_max_tokens),
        ).astype(np.float32)

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
                fit_max_tokens=fit_max_tokens,
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
                    fit_max_tokens=fit_max_tokens,
                ).astype(np.float32)
                umap3 = np.full((z.shape[0], 3), np.nan, dtype=np.float32)
                umap3[valid_mask_flat] = umap3_valid

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
    np.save(os.path.join(results_dir, "latent_vectors_full.npy"), default_latent_map)
    np.save(os.path.join(results_dir, "pca_xyz.npy"), default_pca_map)
    np.save(os.path.join(results_dir, "umap_xyz.npy"), default_umap_map)
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


================================================================================
FILE: src/utils/npy.py
================================================================================

from __future__ import annotations

import numpy as np


def _safe_load_npy(path: str, *, mmap_mode: str | None = "r") -> np.ndarray:
    """Load .npy data, falling back for legacy object/pickled arrays.

    Modern NumPy refuses to mmap pickled object arrays. These are local training
    artifacts, so fall back only for that specific failure and normalize common
    wrapper shapes back into an ndarray.
    """
    try:
        return np.load(path, mmap_mode=mmap_mode)
    except ValueError as exc:
        msg = str(exc).lower()
        if "pickle" not in msg and "object" not in msg:
            raise

    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object:
        if arr.shape == ():
            arr = arr.item()
        elif arr.size == 1:
            arr = arr.reshape(-1)[0]
        else:
            arr = arr.tolist()
    return np.asarray(arr)


def normalize01(x: np.ndarray) -> np.ndarray:
    """Normalize finite array values to [0, 1], replacing non-finite values safely."""
    arr = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(arr)
    if not bool(finite.any()):
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    arr = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
    if hi > lo + 1e-20:
        return ((arr - lo) / (hi - lo)).astype(np.float32)
    return np.zeros_like(arr, dtype=np.float32)


================================================================================
FILE: src/models/predictor.py
================================================================================

import torch.nn as nn

from .encoders import LayerNorm2d


class FullResPredictor(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        hidden: int = 64,
        use_layernorm: bool = False,
        kernel_size: int = 3,
        spatial_conv: bool = True,
        residual: bool = False,
    ):
        super().__init__()
        self.residual = bool(residual)
        if not spatial_conv:
            # Channel-only: 1x1 -> LayerNorm -> GELU -> 1x1, zero-init last conv, residual.
            norm = LayerNorm2d(hidden) if use_layernorm else nn.Identity()
            self.net = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1),
                norm,
                nn.GELU(),
                nn.Conv2d(hidden, channels, kernel_size=1),
            )
            nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.net[-1].bias)
            return

        k = int(kernel_size)
        if k <= 0 or (k % 2) == 0:
            raise ValueError(f"FullResPredictor kernel_size must be a positive odd integer, got {kernel_size}")
        pad = k // 2
        norm1 = LayerNorm2d(hidden) if use_layernorm else nn.Identity()
        norm2 = LayerNorm2d(hidden) if use_layernorm else nn.Identity()
        mid_conv_kwargs = {"kernel_size": k, "padding": pad}
        if k > 1:
            mid_conv_kwargs["padding_mode"] = "reflect"
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            norm1,
            nn.GELU(),
            nn.Conv2d(hidden, hidden, **mid_conv_kwargs),
            norm2,
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x):
        y = self.net(x)
        return x + y if self.residual else y


================================================================================
FILE: src/models/predictor3d.py
================================================================================

from __future__ import annotations

import torch.nn as nn


class FullResPredictor3D(nn.Module):
    def __init__(self, channels: int = 16, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1),
        )

    def forward(self, x):
        return self.net(x)


================================================================================
FILE: src/models/symmetry.py
================================================================================

from __future__ import annotations

import itertools

import torch
import torch.nn as nn


def _pad_to_square(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad spatial dims (H,W) to square max(H,W). Returns (padded, (orig_h, orig_w))."""
    B, C, H, W = x.shape
    if H == W:
        return x, (H, W)
    S = max(H, W)
    pad_h = S - H
    pad_w = S - W
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return torch.nn.functional.pad(
        x,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0.0,
    ), (H, W)


def _crop_from_square(x: torch.Tensor, orig_shape: tuple[int, int]) -> torch.Tensor:
    """Crop back from square to original (H, W)."""
    H, W = orig_shape
    cur_h, cur_w = x.shape[-2:]
    top = max(0, (cur_h - H) // 2)
    left = max(0, (cur_w - W) // 2)
    return x[..., top:top + H, left:left + W]


def _stack_spatial_kwargs(kwargs: dict, spatial_shape: tuple[int, ...], view_fns: list, pad_2d: bool = False) -> list[dict]:
    per_view = [dict() for _ in view_fns]
    for name, val in kwargs.items():
        if torch.is_tensor(val) and val.ndim >= len(spatial_shape) and val.shape[-len(spatial_shape):] == spatial_shape:
            if pad_2d:
                val, _ = _pad_to_square(val)
            for i, fn in enumerate(view_fns):
                per_view[i][name] = fn(val)
        else:
            for i in range(len(view_fns)):
                per_view[i][name] = val
    return per_view


def _encode_view_chunks(
    encoder: nn.Module,
    view_inputs: list[torch.Tensor],
    view_kwargs: list[dict],
    inverse_fns: list,
    max_views_per_forward: int,
) -> torch.Tensor:
    max_views = max(1, int(max_views_per_forward))
    aligned = []
    for start in range(0, len(view_inputs), max_views):
        xs = view_inputs[start:start + max_views]
        kws = view_kwargs[start:start + max_views]
        x_batch = torch.cat(xs, dim=0)
        kw_batch = {}
        for name in kws[0].keys():
            vals = [kw[name] for kw in kws]
            if torch.is_tensor(vals[0]) and all(torch.is_tensor(v) and v.shape == vals[0].shape for v in vals):
                kw_batch[name] = torch.cat(vals, dim=0)
            else:
                kw_batch[name] = vals[0]
        feat_batch = encoder(x_batch, **kw_batch)
        for local_i, feat in enumerate(torch.chunk(feat_batch, chunks=len(xs), dim=0)):
            aligned.append(inverse_fns[start + local_i](feat))
    return torch.stack(aligned, dim=0)


def symmetric_forward_2d(
    encoder: nn.Module,
    x: torch.Tensor,
    return_var: bool = False,
    max_views_per_forward: int = 1,
    **kwargs,
):
    """
    Four-way rotational group average for dense 2D spatial features.

    Pads to square, evaluates the four rotations in view chunks, unrotates,
    crops back, and averages. Handles non-square (H != W) inputs transparently.

    When return_var=True, also returns the per-pixel variance across the 4 rotation
    views as a regularisation signal (shape matches the averaged output).
    """
    if x.ndim < 4:
        raise ValueError(f"symmetric_forward_2d expects at least 4D input, got {tuple(x.shape)}")

    B, C, H, W = x.shape

    x_sq, orig_shape = _pad_to_square(x)
    view_fns = [lambda t, k=k: torch.rot90(t, k=k, dims=(-2, -1)) for k in range(4)]
    inverse_fns = [
        lambda t, k=k: _crop_from_square(torch.rot90(t, k=-k, dims=(-2, -1)), orig_shape)
        for k in range(4)
    ]
    view_inputs = [fn(x_sq) for fn in view_fns]
    view_kwargs = _stack_spatial_kwargs(kwargs, (H, W), view_fns, pad_2d=True)
    feats_stacked_inv = _encode_view_chunks(
        encoder,
        view_inputs,
        view_kwargs,
        inverse_fns,
        max_views_per_forward=max_views_per_forward,
    )

    avg = feats_stacked_inv.mean(dim=0)
    if return_var:
        var = feats_stacked_inv.var(dim=0, unbiased=False).clamp(min=0.0)
        return avg, var
    return avg


def symmetric_forward_3d(
    encoder: nn.Module,
    x: torch.Tensor,
    return_var: bool = False,
    max_views_per_forward: int = 1,
    **kwargs,
):
    """
    Eight-way flip group average for dense 3D spatial features.

    Evaluates the 8 flip configurations in view chunks, then unflips and averages.
    Flips are involutions, so the same flip axes align encoder outputs back to
    the original D/H/W layout.
    """
    if x.ndim < 5:
        raise ValueError(f"symmetric_forward_3d expects at least 5D input, got {tuple(x.shape)}")

    B, C, D, H, W = x.shape
    spatial_dims = (-3, -2, -1)

    # Build all 8 flip configurations (2^3 = 8)
    all_dims_list = []
    for r in range(len(spatial_dims) + 1):
        for dims in itertools.combinations(spatial_dims, r):
            all_dims_list.append(dims)

    view_fns = [
        (lambda t, dims=dims: torch.flip(t, dims=dims) if dims else t)
        for dims in all_dims_list
    ]
    inverse_fns = view_fns
    view_inputs = [fn(x) for fn in view_fns]
    view_kwargs = _stack_spatial_kwargs(kwargs, (D, H, W), view_fns, pad_2d=False)
    feats_stacked_inv = _encode_view_chunks(
        encoder,
        view_inputs,
        view_kwargs,
        inverse_fns,
        max_views_per_forward=max_views_per_forward,
    )

    avg = feats_stacked_inv.mean(dim=0)
    if return_var:
        var = feats_stacked_inv.var(dim=0, unbiased=False).clamp(min=0.0)
        return avg, var
    return avg


================================================================================
FILE: src/models/encoders3d.py
================================================================================

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Norm layers
# ---------------------------------------------------------------------------

class LayerNorm3d(nn.Module):
    """LayerNorm over channels for B,C,D,H,W tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)  # B,D,H,W,C
        x = self.norm(x)
        return x.permute(0, 4, 1, 2, 3)  # B,C,D,H,W


class GRN3D(nn.Module):
    """Global Response Normalization for 3D (ConvNeXt V2 style).

    Operates in channels-last layout (B,D,H,W,C).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, int(dim)))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, int(dim)))
        self.eps = float(eps)

    def forward(self, x):
        # x: B,D,H,W,C
        gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


# ---------------------------------------------------------------------------
# ConvNeXt3D Block
# ---------------------------------------------------------------------------

class ConvNeXtBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        mlp_ratio: float = 4.0,
        use_grn: bool = True,
    ):
        super().__init__()
        pad = kernel_size // 2
        hidden = int(channels * mlp_ratio)
        self.use_grn = bool(use_grn)

        self.dw = nn.Conv3d(channels, channels, kernel_size, padding=pad, groups=channels, padding_mode="replicate")
        self.norm = LayerNorm3d(channels)
        self.pw1 = nn.Conv3d(channels, hidden, 1)
        self.act = nn.GELU()
        # GRN operates in channels-last: B,D,H,W,C
        self.grn = GRN3D(hidden) if self.use_grn else nn.Identity()
        self.pw2 = nn.Conv3d(hidden, channels, 1)

        nn.init.zeros_(self.pw2.weight)
        if self.pw2.bias is not None:
            nn.init.zeros_(self.pw2.bias)

    def forward(self, x):
        y = self.dw(x)                     # B,C,D,H,W
        y = self.norm(y)                   # LayerNorm3d (permute → norm → permute)
        y = self.pw1(y)
        y = self.act(y)
        y = y.permute(0, 2, 3, 4, 1)      # B,D,H,W,C for GRN
        y = self.grn(y)
        y = y.permute(0, 4, 1, 2, 3)      # B,C,D,H,W
        y = self.pw2(y)
        return x + y


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

class ScaleGateMixer3D(nn.Module):
    def __init__(self, channels: int, num_scales: int, hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(channels * num_scales, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, num_scales, 1),
        )

    def forward(self, x):
        b, s, c, d, h, w = x.shape
        flat = x.reshape(b, s * c, d, h, w)
        weights = torch.softmax(self.gate(flat), dim=1)
        y = (x * weights[:, :, None]).sum(dim=1)
        return y


# ---------------------------------------------------------------------------
# ScaleAware ConvNeXt3D Encoder  (WITH norm flags — mirrors 2D)
# ---------------------------------------------------------------------------

class ScaleAwareConvNeXt3DEncoder(nn.Module):
    def __init__(
        self,
        num_scales: int,
        out_channels: int = 16,
        scale_channels: int = 8,
        depth: int = 3,
        kernel_size: int = 5,
        fusion: str = "gate",
        stride: int = 1,
        *,
        use_grn: bool = True,
        stem_norm: bool = True,
        norm_per_scale: bool = True,
        adapter_norm: bool = True,
        final_norm: bool = True,
    ):
        super().__init__()
        self.num_scales = int(num_scales)
        self.fusion = str(fusion)
        self.norm_per_scale = bool(norm_per_scale)
        self.adapter_norm = bool(adapter_norm)

        # Stem: [field, mask_token] → scale_channels
        stem_layers = [
            nn.Conv3d(2, scale_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(scale_channels, scale_channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if stem_norm:
            stem_layers.append(LayerNorm3d(scale_channels))
        self.scale_stem = nn.Sequential(*stem_layers)
        self.per_scale_norm = LayerNorm3d(scale_channels) if self.norm_per_scale else nn.Identity()

        # Fusion
        if self.fusion == "gate":
            self.fuse = ScaleGateMixer3D(scale_channels, self.num_scales)
            fused_channels = scale_channels
        elif self.fusion == "concat":
            self.fuse = None
            fused_channels = scale_channels * self.num_scales
        else:
            raise ValueError(f"Unknown fusion={fusion}")

        # Proj
        self.proj = nn.Conv3d(fused_channels, out_channels, kernel_size=1)

        # Blocks
        blocks = []
        for _ in range(int(depth)):
            blocks.append(ConvNeXtBlock3D(out_channels, kernel_size=int(kernel_size), use_grn=use_grn))
        self.blocks = nn.Sequential(*blocks)

        # Final norm
        self.final_norm = LayerNorm3d(out_channels) if final_norm else nn.Identity()

    def forward(self, fields, mask_tokens=None):
        if mask_tokens is None:
            mask_tokens = torch.zeros_like(fields)

        b, s, d, h, w = fields.shape
        if s != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} scales, got {s}")

        x = torch.stack([fields, mask_tokens], dim=2)  # B, S, 2, D, H, W
        x = x.reshape(b * s, 2, d, h, w)
        x = self.scale_stem(x)
        cs = x.shape[1]
        x = x.reshape(b, s, cs, d, h, w)  # B, S, C, D, H, W

        # Per-scale norm before fusion
        if self.norm_per_scale:
            x = self.per_scale_norm(x.reshape(b * s, cs, d, h, w)).reshape(b, s, cs, d, h, w)

        # Fusion
        if self.fusion == "gate":
            x = self.fuse(x)
        else:
            x = x.reshape(b, s * cs, d, h, w)

        x = self.proj(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        return x


# ---------------------------------------------------------------------------
# FiLM + Shared ConvNeXt3D Encoder  (WITH norm flags)
# ---------------------------------------------------------------------------

class ScaleFiLM3d(nn.Module):
    """Per-scale affine modulation for B,S,C,D,H,W tensors."""

    def __init__(self, num_scales: int, channels: int):
        super().__init__()
        self.emb = nn.Embedding(int(num_scales), int(channels) * 2)
        nn.init.zeros_(self.emb.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, D, H, W = x.shape
        idx = torch.arange(S, device=x.device)
        gb = self.emb(idx)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.view(1, S, C, 1, 1, 1)
        beta = beta.view(1, S, C, 1, 1, 1)
        return x * (1.0 + gamma) + beta


class SharedScaleConvNeXtStage3d(nn.Module):
    """Apply a single shared ConvNeXt3D block to every scale, with optional FiLM."""

    def __init__(self, block: nn.Module, num_scales: int, channels: int, use_film: bool = True):
        super().__init__()
        self.block = block
        self.film = ScaleFiLM3d(num_scales, channels) if use_film else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, D, H, W = x.shape
        x = self.film(x)
        y = x.reshape(B * S, C, D, H, W)
        y = self.block(y)
        y = y.reshape(B, S, C, D, H, W)
        return y


class PerScaleAdapter3d(nn.Module):
    """Tiny per-scale 1x1x1 conv adapter (residual, zero-init)."""

    def __init__(self, num_scales: int, channels: int, use_norm: bool = True):
        super().__init__()
        self.use_norm = bool(use_norm)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, 1),
                nn.GELU(),
                nn.Conv3d(channels, channels, 1),
            )
            for _ in range(int(num_scales))
        ])
        self.norms = nn.ModuleList([LayerNorm3d(channels) for _ in range(int(num_scales))])
        for a in self.adapters:
            nn.init.zeros_(a[-1].weight)
            if a[-1].bias is not None:
                nn.init.zeros_(a[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, D, H, W = x.shape
        outs = []
        for s in range(S):
            xs = x[:, s]
            adapt = self.adapters[s](xs)
            if self.use_norm:
                adapt = self.norms[s](adapt)
            outs.append(xs + adapt)
        return torch.stack(outs, dim=1)


class ScaleFiLMConvNeXt3DEncoder(nn.Module):
    """3D scale-aware encoder: shared ConvNeXt3D blocks + scale FiLM + norms.

    Input:
      fields:      B x S x D x H x W
      mask_tokens: B x S x D x H x W

    Norm flags mirror the 2D CDDScaleAwareConvNeXtEncoder:
      stem_norm, norm_per_scale, adapter_norm, final_norm, use_grn
    """

    def __init__(
        self,
        num_scales: int,
        out_channels: int = 16,
        scale_channels: int = 8,
        depth: int = 3,
        kernel_size: int = 5,
        fusion: str = "gate",
        use_film: bool = True,
        use_per_scale_adapters: bool = False,
        stride: int = 1,
        *,
        use_grn: bool = True,
        stem_norm: bool = True,
        norm_per_scale: bool = True,
        adapter_norm: bool = True,
        final_norm: bool = True,
    ):
        super().__init__()
        self.num_scales = int(num_scales)
        self.fusion = str(fusion)
        self.norm_per_scale = bool(norm_per_scale)

        # Per-scale stem  [field, mask] → scale_channels
        stem_layers = [
            nn.Conv3d(2, scale_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(scale_channels, scale_channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if stem_norm:
            stem_layers.append(LayerNorm3d(scale_channels))
        self.scale_stem = nn.Sequential(*stem_layers)
        self.per_scale_norm = LayerNorm3d(scale_channels) if self.norm_per_scale else nn.Identity()

        # Shared ConvNeXt3D blocks with FiLM
        self.blocks = nn.ModuleList()
        for _ in range(int(depth)):
            blk = ConvNeXtBlock3D(scale_channels, kernel_size=int(kernel_size), use_grn=use_grn)
            stage = SharedScaleConvNeXtStage3d(
                block=blk,
                num_scales=self.num_scales,
                channels=scale_channels,
                use_film=use_film,
            )
            self.blocks.append(stage)
            if use_per_scale_adapters:
                self.blocks.append(
                    PerScaleAdapter3d(self.num_scales, scale_channels, use_norm=adapter_norm)
                )

        # Fusion
        if self.fusion == "gate":
            self.fuse = ScaleGateMixer3D(scale_channels, self.num_scales)
            fused_channels = scale_channels
        elif self.fusion == "concat":
            self.fuse = None
            fused_channels = scale_channels * self.num_scales
        else:
            raise ValueError(f"Unknown fusion={fusion}")

        # Head
        self.head = nn.Conv3d(fused_channels, out_channels, kernel_size=1)

        # Final norm
        self.final_norm = LayerNorm3d(out_channels) if final_norm else nn.Identity()

    def forward(self, fields, mask_tokens=None):
        if mask_tokens is None:
            mask_tokens = torch.zeros_like(fields)

        b, s, d, h, w = fields.shape
        if s != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} scales, got {s}")

        # Per-scale stem
        x = torch.stack([fields, mask_tokens], dim=2)  # B, S, 2, D, H, W
        x = x.reshape(b * s, 2, d, h, w)
        x = self.scale_stem(x)
        cs = x.shape[1]
        x = x.reshape(b, s, cs, d, h, w)  # B, S, C, D, H, W

        # Per-scale norm
        if self.norm_per_scale:
            x = self.per_scale_norm(x.reshape(b * s, cs, d, h, w)).reshape(b, s, cs, d, h, w)

        # Shared ConvNeXt3D blocks + adapters
        for blk in self.blocks:
            x = blk(x)

        # Fusion
        if self.fusion == "gate":
            x = self.fuse(x)  # B, C, D, H, W
        else:
            x = x.reshape(b, s * cs, d, h, w)

        # Head + final norm
        x = self.head(x)
        x = self.final_norm(x)
        return x


================================================================================
FILE: src/models/__init__.py
================================================================================

from .build_jepa import PyramidGridJEPA
from .build_jepa3d import PyramidGridJEPA3D

__all__ = ["PyramidGridJEPA", "PyramidGridJEPA3D"]


================================================================================
FILE: src/models/encoders.py
================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F



class LayerNorm2d(nn.Module):
    """LayerNorm over channels for BCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, int(dim)))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, int(dim)))
        self.eps = float(eps)

    def forward(self, x):
        # x: B,H,W,C
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


def _valid_groups(channels: int, groups: int) -> int:
    g = max(1, int(groups))
    while channels % g != 0 and g > 1:
        g -= 1
    return g


def make_norm2d(channels: int, norm_type: str = "layernorm", norm_groups: int = 8, norm_eps: float = 1e-6) -> nn.Module:
    kind = str(norm_type).lower()
    if kind == "layernorm":
        return LayerNorm2d(channels, eps=float(norm_eps))
    if kind == "groupnorm":
        return nn.GroupNorm(_valid_groups(channels, norm_groups), channels, eps=float(norm_eps))
    raise ValueError(f"Unsupported norm_type={norm_type}. Use 'layernorm' or 'groupnorm'.")


def _normalize_convnext_dilations(dilations, depth: int) -> list[int]:
    depth = int(depth)
    if dilations is None:
        return [1] * depth
    values = [int(d) for d in dilations]
    if not values:
        raise ValueError("ConvNeXt dilations must contain at least one value.")
    if any(d <= 0 for d in values):
        raise ValueError(f"ConvNeXt dilations must be positive integers, got {values}.")
    if len(values) < depth:
        reps = (depth + len(values) - 1) // len(values)
        values = (values * reps)[:depth]
    elif len(values) > depth:
        values = values[:depth]
    return values


class ConvNeXtDenseBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        kernel_size: int = 7,
        dilation: int = 1,
        layer_scale_init: float = 1e-6,
        use_reflect_padding: bool = True,
        use_grn: bool = True,
    ):
        super().__init__()
        self.use_grn = bool(use_grn)
        self.dilation = int(dilation)
        pad = (int(kernel_size) // 2) * self.dilation
        if use_reflect_padding:
            self.dwconv = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=0,
                    dilation=self.dilation,
                    groups=channels,
                ),
            )
        else:
            self.dwconv = nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=pad,
                dilation=self.dilation,
                groups=channels,
            )
        self.norm = nn.LayerNorm(channels)
        self.pw1 = nn.Linear(channels, expansion * channels)
        self.act = nn.GELU()
        self.grn = GRN(expansion * channels) if self.use_grn else nn.Identity()
        self.pw2 = nn.Linear(expansion * channels, channels)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels))

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # B,H,W,C
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class ConvNeXtDenseEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        latent_channels: int = 32,
        depth: int = 4,
        kernel_size: int = 7,
        expansion: int = 4,
        use_reflect_padding: bool = True,
        final_norm: bool = True,
        final_norm_type: str = "layernorm",
        head_bias: bool = True,
        dilations=None,
        use_grn: bool = True,
        stem_norm: bool = True,
    ):
        super().__init__()
        depth = int(depth)
        dilations = _normalize_convnext_dilations(dilations, depth)
        self.dilations = tuple(dilations)

        self.stem = nn.Sequential(
            nn.ReflectionPad2d(1) if use_reflect_padding else nn.Identity(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=0 if use_reflect_padding else 1),
            LayerNorm2d(hidden_channels) if stem_norm else nn.Identity(),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                ConvNeXtDenseBlock(
                    channels=hidden_channels,
                    expansion=expansion,
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    use_reflect_padding=use_reflect_padding,
                    use_grn=use_grn,
                )
                for i in range(depth)
            ]
        )
        self.head = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1, bias=head_bias)
        if not final_norm:
            self.final_norm = nn.Identity()
        else:
            ntype = str(final_norm_type).lower()
            if ntype == "batchnorm":
                self.final_norm = nn.BatchNorm2d(latent_channels, track_running_stats=False)
            elif ntype in ("layernorm", ""):
                self.final_norm = LayerNorm2d(latent_channels)
            else:
                raise ValueError(
                    f"Unsupported final_norm_type={final_norm_type}. "
                    "Use 'layernorm' or 'batchnorm'."
                )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.final_norm(x)
        return x


class EscnnC4PyramidEncoder(nn.Module):
    """
    C4 rotation-equivariant pyramid encoder using escnn.

    Input is a normal BCHW tensor with the same channel contract as
    convnext_dense_pyramid: per-scale CDD channels concatenated with per-scale
    mask-token channels. escnn lifts those trivial input fields into regular
    C4 fields, applies equivariant R2Conv blocks, then group-pools to return a
    standard invariant BCHW tensor.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        latent_channels: int = 32,
        depth: int = 4,
        kernel_size: int = 7,
        final_norm: bool = True,
        final_norm_type: str = "layernorm",
    ):
        super().__init__()
        try:
            from escnn import gspaces
            from escnn import nn as enn
        except ImportError as exc:
            raise ImportError(
                "escnn_c4_pyramid requires the optional dependency 'escnn'. "
                "Install it with: pip install escnn"
            ) from exc

        depth = int(depth)
        hidden_channels = int(hidden_channels)
        latent_channels = int(latent_channels)
        kernel_size = int(kernel_size)
        padding = kernel_size // 2

        self.enn = enn
        self.r2_act = gspaces.rot2dOnR2(N=4)
        self.in_type = enn.FieldType(self.r2_act, int(in_channels) * [self.r2_act.trivial_repr])
        self.hidden_type = enn.FieldType(self.r2_act, hidden_channels * [self.r2_act.regular_repr])
        self.out_type = enn.FieldType(self.r2_act, latent_channels * [self.r2_act.regular_repr])

        self.lift = enn.SequentialModule(
            enn.R2Conv(self.in_type, self.hidden_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(self.hidden_type),
            enn.ReLU(self.hidden_type, inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                enn.SequentialModule(
                    enn.R2Conv(self.hidden_type, self.hidden_type, kernel_size=kernel_size, padding=padding, bias=False),
                    enn.InnerBatchNorm(self.hidden_type),
                    enn.ReLU(self.hidden_type, inplace=True),
                )
                for _ in range(depth)
            ]
        )
        self.head = enn.R2Conv(self.hidden_type, self.out_type, kernel_size=1, padding=0, bias=True)
        self.gpool = enn.GroupPooling(self.out_type)
        if not final_norm:
            self.final_norm = nn.Identity()
        else:
            ntype = str(final_norm_type).lower()
            if ntype == "batchnorm":
                self.final_norm = nn.BatchNorm2d(latent_channels, track_running_stats=False)
            elif ntype in ("layernorm", ""):
                self.final_norm = LayerNorm2d(latent_channels)
            else:
                raise ValueError(
                    f"Unsupported final_norm_type={final_norm_type}. "
                    "Use 'layernorm' or 'batchnorm'."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.enn.GeometricTensor(x, self.in_type)
        gx = self.lift(gx)
        for block in self.blocks:
            gx = gx + block(gx)
        gx = self.head(gx)
        gx = self.gpool(gx)
        return self.final_norm(gx.tensor)


class CDDScaleAwareConvNeXtEncoder(nn.Module):
    """
    Scale-aware CDD pyramid encoder.

    Input:
      fields:      B x S x H x W
      mask_tokens: B x S x H x W

    Per scale:
      [field_s, mask_s, normalized_log_sigma_s] -> shared adapter.
    Then concatenate scale features and feed a dense ConvNeXt encoder.
    """

    def __init__(
        self,
        scales,
        hidden_channels: int,
        latent_channels: int,
        depth: int = 4,
        kernel_size: int = 7,
        expansion: int = 4,
        scale_feat_channels: int = 8,
        adapter_kernel_size: int = 3,
        fusion_type: str = "concat",
        use_reflect_padding: bool = True,
        final_norm: bool = True,
        final_norm_type: str = "layernorm",
        head_bias: bool = True,
        cdd_append_last_residual: bool = True,
        adapter_norm: bool = True,
        use_grn: bool = True,
        stem_norm: bool = True,
        dilations=None,
    ):
        super().__init__()
        self.scales = tuple(float(s) for s in scales)
        self.num_scales = len(self.scales)
        self.scale_feat_channels = int(scale_feat_channels)
        self.fusion_type = str(fusion_type).lower()
        self.cdd_append_last_residual = bool(cdd_append_last_residual)
        if self.fusion_type not in ("concat", "topdown"):
            raise ValueError(f"Unsupported fusion_type={fusion_type}. Use 'concat' or 'topdown'.")

        logs = torch.log(torch.tensor(self.scales, dtype=torch.float32))
        if logs.numel() > 1:
            logs = (logs - logs.mean()) / logs.std(unbiased=False).clamp_min(1e-6)
        else:
            logs = logs * 0.0
        self.register_buffer("scale_codes", logs.view(1, self.num_scales, 1, 1), persistent=False)

        pad = int(adapter_kernel_size) // 2
        if use_reflect_padding and pad > 0:
            adapter_layers = [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(3, self.scale_feat_channels, kernel_size=int(adapter_kernel_size), padding=0),
            ]
        else:
            adapter_layers = [
                nn.Conv2d(3, self.scale_feat_channels, kernel_size=int(adapter_kernel_size), padding=pad),
            ]
        if adapter_norm:
            adapter_layers.append(LayerNorm2d(self.scale_feat_channels))
        adapter_layers += [
            nn.GELU(),
            nn.Conv2d(self.scale_feat_channels, self.scale_feat_channels, kernel_size=1),
        ]
        if adapter_norm:
            adapter_layers.append(LayerNorm2d(self.scale_feat_channels))
        adapter_layers.append(nn.GELU())
        self.adapter = nn.Sequential(*adapter_layers)

        print(
            f"[CDDScaleAwareConvNeXt] depth={depth}, dilations={dilations}, "
            f"stem_norm={stem_norm}, adapter_norm={adapter_norm}, "
            f"final_norm={final_norm}({final_norm_type}), grn={use_grn}"
        )
        self.convnext = ConvNeXtDenseEncoder(
            in_channels=self.num_scales * self.scale_feat_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            depth=depth,
            kernel_size=kernel_size,
            expansion=expansion,
            use_reflect_padding=use_reflect_padding,
            final_norm=final_norm,
            final_norm_type=final_norm_type,
            head_bias=head_bias,
            use_grn=use_grn,
            stem_norm=stem_norm,
            dilations=dilations,
        )
        if self.fusion_type == "topdown":
            self.fusion_proj = nn.ModuleList(
                [
                    nn.Conv2d(self.scale_feat_channels, self.scale_feat_channels, kernel_size=1)
                    for _ in range(self.num_scales)
                ]
            )

    def forward(self, fields: torch.Tensor, mask_tokens=None) -> torch.Tensor:
        if fields.ndim != 4:
            raise ValueError(f"Expected fields B,S,H,W, got {tuple(fields.shape)}")
        b, s, h, w = fields.shape
        if mask_tokens is None:
            mask_tokens = torch.zeros_like(fields)

        if s != self.num_scales:
            if s > self.num_scales:
                n_extra = s - self.num_scales
                if self.cdd_append_last_residual:
                    base = fields[:, : self.num_scales, :, :]
                    extra = fields[:, self.num_scales :, :, :]
                    last = base[:, self.num_scales - 1 : self.num_scales, :, :] + extra.sum(dim=1, keepdim=True)
                    fields = torch.cat([base[:, : self.num_scales - 1, :, :], last], dim=1)
                else:
                    fields = fields[:, : self.num_scales, :, :]
                print(
                    f"[{self.__class__.__name__}] WARNING: Truncated {n_extra} extra channel(s) "
                    f"(append_last_residual={self.cdd_append_last_residual}). Check model.sigmas and encoder scale count."
                )
            else:
                n_missing = self.num_scales - s
                if self.cdd_append_last_residual:
                    residual = fields[:, -1:, :, :]
                    res_mask = mask_tokens[:, -1:, :, :]
                    n_split = n_missing + 1
                    split = residual / float(n_split)
                    fields = torch.cat([fields[:, :-1, :, :], split.expand(-1, n_split, -1, -1)], dim=1)
                    mask_tokens = torch.cat([mask_tokens[:, :-1, :, :], res_mask.expand(-1, n_split, -1, -1)], dim=1)
                else:
                    zeros = torch.zeros(b, n_missing, h, w, dtype=fields.dtype, device=fields.device)
                    fields = torch.cat([fields, zeros], dim=1)
                    mask_tokens = torch.cat([mask_tokens, zeros], dim=1)
                print(
                    f"[{self.__class__.__name__}] WARNING: Padded {n_missing} missing channel(s) "
                    f"(append_last_residual={self.cdd_append_last_residual}). Check model.sigmas and encoder scale count."
                )
            s = self.num_scales

        mask_tokens = mask_tokens[:, :s, :, :]
        if mask_tokens.shape != fields.shape:
            raise ValueError(
                f"mask_tokens shape must match fields shape. fields={tuple(fields.shape)} mask={tuple(mask_tokens.shape)}"
            )

        scale_maps = self.scale_codes.to(dtype=fields.dtype, device=fields.device).expand(b, s, h, w)
        feats = []
        for i in range(s):
            xi = torch.stack([fields[:, i], mask_tokens[:, i], scale_maps[:, i]], dim=1)
            feats.append(self.adapter(xi))
        if self.fusion_type == "topdown":
            fused = [None] * s
            running = None
            for rev_i, feat in enumerate(reversed(feats)):
                idx = s - 1 - rev_i
                if running is None:
                    running = feat
                else:
                    if running.shape[-2:] != feat.shape[-2:]:
                        running = F.interpolate(running, size=feat.shape[-2:], mode="bilinear", align_corners=False)
                    running = feat + running
                fused[idx] = self.fusion_proj[idx](running)
            feats = fused
        x = torch.cat(feats, dim=1)
        return self.convnext(x)


================================================================================
FILE: src/models/masking.py
================================================================================

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch


PRIMARY_TARGET_SAMPLING_MODES = ("random", "priority", "priority_small_scale")
LEGACY_TARGET_SAMPLING_MODES = ("lattice",)
LEGACY_TARGET_SAMPLING_ALIASES = {
    "grid": "lattice",
    "priority_sampling": "priority",
    "priority_small_scale": "priority_small_scale",
}
ALLOWED_TARGET_SAMPLING_MODES = PRIMARY_TARGET_SAMPLING_MODES + LEGACY_TARGET_SAMPLING_MODES


def normalize_target_sampling_mode(mode: str) -> str:
    sampling_mode = str(mode).strip().lower()
    sampling_mode = LEGACY_TARGET_SAMPLING_ALIASES.get(sampling_mode, sampling_mode)
    if sampling_mode not in ALLOWED_TARGET_SAMPLING_MODES:
        allowed = ", ".join(ALLOWED_TARGET_SAMPLING_MODES)
        raise ValueError(f"target_sampling_mode must be one of: {allowed}; got {mode!r}")
    return sampling_mode


def norm_per_sample_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize each [H,W] map independently per sample+channel.
    x: B x S x H x W
    """
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    return (x - mean) / std


def _shared_grid_centers(
    h: int,
    w: int,
    base_margin: int,
    spacing_px: int,
    global_shift: bool,
    device: torch.device,
    enable_grid_jitter: bool = True,
    lattice_shift_override: Optional[Tuple[int, int]] = None,
):
    """Generate one globally-shifted full-image lattice, then boundary-mask it."""
    spacing_px = int(max(1, spacing_px))
    if lattice_shift_override is not None:
        shift_y = int(lattice_shift_override[0]) % spacing_px
        shift_x = int(lattice_shift_override[1]) % spacing_px
    elif global_shift:
        shift_y = int(torch.randint(0, spacing_px, (1,), device=device).item())
        shift_x = int(torch.randint(0, spacing_px, (1,), device=device).item())
    else:
        shift_y = 0
        shift_x = 0

    y_centers = list(range(shift_y % spacing_px, h, spacing_px))
    x_centers = list(range(shift_x % spacing_px, w, spacing_px))
    if len(y_centers) == 0:
        y_centers = [h // 2]
    if len(x_centers) == 0:
        x_centers = [w // 2]

    raw_centers = [(cy, cx) for cy in y_centers for cx in x_centers]
    if enable_grid_jitter:
        grid_dy = (float(torch.rand(1, device=device).item()) - 0.5) * float(spacing_px)
        grid_dx = (float(torch.rand(1, device=device).item()) - 0.5) * float(spacing_px)
    else:
        grid_dy = 0.0
        grid_dx = 0.0

    shared_centers = []
    for cy, cx in raw_centers:
        jy = int(round(float(cy) + grid_dy))
        jx = int(round(float(cx) + grid_dx))
        jy = int(min(h - 1, max(0, jy)))
        jx = int(min(w - 1, max(0, jx)))
        edge_dist = min(jy, (h - 1) - jy, jx, (w - 1) - jx)
        if edge_dist < int(base_margin):
            continue
        shared_centers.append((jy, jx))

    if len(shared_centers) == 0:
        y_min = int(base_margin)
        y_max = int(max(y_min, h - 1 - int(base_margin)))
        x_min = int(base_margin)
        x_max = int(max(x_min, w - 1 - int(base_margin)))
        for cy, cx in raw_centers:
            iy = int(min(y_max, max(y_min, int(cy))))
            ix = int(min(x_max, max(x_min, int(cx))))
            shared_centers.append((iy, ix))

    return shared_centers


def _build_priority_catalogue_from_cdd_ratio(
    cdd_orig: np.ndarray,
    top_percent: float,
    patch_size: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    """Rank pixels by (sum of two smallest scales) / (total flux)."""
    if cdd_orig.ndim != 3 or cdd_orig.shape[0] <= 0:
        return []
    if cdd_orig.shape[0] == 1:
        numerator = cdd_orig[0]
    else:
        numerator = cdd_orig[0] + cdd_orig[1]
    total_flux = np.sum(cdd_orig, axis=0)
    denom = np.maximum(total_flux.astype(np.float64, copy=False), 1e-6)
    numerator64 = numerator.astype(np.float64, copy=False)
    ratio = np.divide(numerator64, denom, out=np.zeros_like(numerator64), where=denom > 0.0)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

    half_lo = int(patch_size) // 2
    half_hi = int(patch_size) - half_lo
    valid = np.ones((h, w), dtype=bool)
    # Empty background pixels have tied zero ratios and should never enter a
    # priority catalogue merely because top_percent exceeds the signal area.
    valid &= total_flux > 1e-8
    if half_lo > 0:
        valid[:half_lo, :] = False
        valid[:, :half_lo] = False
    if half_hi > 0:
        valid[h - half_hi + 1 :, :] = False
        valid[:, w - half_hi + 1 :] = False
    valid_idx = np.flatnonzero(valid.reshape(-1))
    if valid_idx.size == 0:
        return []

    pct = float(np.clip(top_percent, 0.0, 100.0))
    k = int(math.ceil((pct / 100.0) * float(valid_idx.size)))
    k = max(1, min(k, int(valid_idx.size)))
    valid_ratio = ratio.reshape(-1)[valid_idx]
    top_local = np.argpartition(valid_ratio, -k)[-k:]
    top_local = top_local[np.argsort(valid_ratio[top_local])[::-1]]
    selected = valid_idx[top_local]
    ys = (selected // w).astype(np.int64)
    xs = (selected % w).astype(np.int64)
    return [(int(y), int(x)) for y, x in zip(ys, xs)]


def _build_random_catalogue_from_cdd(
    cdd_orig: np.ndarray,
    patch_size: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    """All valid signal pixels, shuffled later by torch for random sampling."""
    if cdd_orig.ndim != 3 or cdd_orig.shape[0] <= 0:
        return []
    total_flux = np.sum(cdd_orig, axis=0)
    half_lo = int(patch_size) // 2
    half_hi = int(patch_size) - half_lo
    valid = total_flux > 1e-8
    if half_lo > 0:
        valid[:half_lo, :] = False
        valid[:, :half_lo] = False
    if half_hi > 0:
        valid[h - half_hi + 1 :, :] = False
        valid[:, w - half_hi + 1 :] = False
    valid_idx = np.flatnonzero(valid.reshape(-1))
    if valid_idx.size == 0:
        return []
    ys = (valid_idx // w).astype(np.int64)
    xs = (valid_idx % w).astype(np.int64)
    return [(int(y), int(x)) for y, x in zip(ys, xs)]


def _build_random_catalogue_from_array(
    arr: np.ndarray,
    patch_size: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    """All valid nonzero image pixels, shuffled later by torch."""
    arr2d = np.asarray(arr)
    if arr2d.ndim != 2:
        return []
    half_lo = int(patch_size) // 2
    half_hi = int(patch_size) - half_lo
    valid = np.isfinite(arr2d) & (arr2d > 1e-8)
    if half_lo > 0:
        valid[:half_lo, :] = False
        valid[:, :half_lo] = False
    if half_hi > 0:
        valid[h - half_hi + 1 :, :] = False
        valid[:, w - half_hi + 1 :] = False
    valid_idx = np.flatnonzero(valid.reshape(-1))
    if valid_idx.size == 0:
        return []
    ys = (valid_idx // w).astype(np.int64)
    xs = (valid_idx % w).astype(np.int64)
    return [(int(y), int(x)) for y, x in zip(ys, xs)]


def _fractional_spatial_target_budget(
    height: int,
    width: int,
    box_size: int,
    oversample: float,
    device: torch.device,
    minimum: int = 0,
) -> int | None:
    """Budget candidates as f * image_area / box_area with stochastic rounding."""
    f = float(oversample)
    if f <= 0.0:
        return None
    box = max(1, int(box_size))
    desired = f * float(max(1, int(height)) * max(1, int(width))) / float(box * box)
    base = int(math.floor(desired))
    frac = float(desired - base)
    extra = int(torch.rand(1, device=device).item() < frac)
    return max(int(minimum), base + extra)


def _dither_target_center(
    cy: int,
    cx: int,
    h: int,
    w: int,
    half_lo: int,
    half_hi: int,
    dithering_pixels: int,
    device: torch.device,
) -> tuple[int, int]:
    """Jitter a center within a local square and keep target patch in-bounds."""
    d = int(max(0, dithering_pixels))
    if d <= 1:
        return int(cy), int(cx)
    max_off = d // 2
    dy = int(torch.randint(-max_off, max_off + 1, (1,), device=device).item())
    dx = int(torch.randint(-max_off, max_off + 1, (1,), device=device).item())
    cy2 = int(cy) + dy
    cx2 = int(cx) + dx
    y_min = int(half_lo)
    y_max = int(max(y_min, h - half_hi))
    x_min = int(half_lo)
    x_max = int(max(x_min, w - half_hi))
    cy2 = int(min(y_max, max(y_min, cy2)))
    cx2 = int(min(x_max, max(x_min, cx2)))
    return cy2, cx2


def _odd_box(v: int, minimum: int = 3, bump_up: bool = True) -> int:
    x = int(max(minimum, v))
    if x % 2 == 0:
        x += 1 if bump_up else -1
    x = max(x, minimum)
    if x % 2 == 0:
        x += 1
    return x


def _rejection_sample_targets(
    candidates: list[tuple[int, int]],
    num_targets: int,
    h: int,
    w: int,
    exclusion_box: int,
    device: torch.device,
    max_tries: int = 4096,
    allow_partial_overlap: float = 0.0,
) -> list[tuple[int, int]]:
    """Select non-overlapping targets via occupancy-map rejection sampling.

    The exclusion footprint is a square of size exclusion_box centered on
    each accepted target.  This footprint protects the *mask* from
    overlapping, not just the inner target patch.
    """
    if exclusion_box <= 0 or len(candidates) == 0:
        return candidates[:num_targets]

    half = exclusion_box // 2
    occ = torch.zeros((h, w), dtype=torch.bool, device="cpu")
    accepted: list[tuple[int, int]] = []
    tries = 0

    # Shuffle candidates for unbiased selection.
    perm = torch.randperm(len(candidates), device="cpu").tolist()
    idx = 0
    while len(accepted) < num_targets and tries < max_tries and idx < len(candidates):
        tries += 1
        cy, cx = candidates[int(perm[idx])]
        idx += 1

        y0 = max(0, int(cy) - half)
        y1 = min(h, int(cy) + exclusion_box - half)
        x0 = max(0, int(cx) - half)
        x1 = min(w, int(cx) + exclusion_box - half)
        if y1 <= y0 or x1 <= x0:
            continue

        footprint = occ[y0:y1, x0:x1]
        overlap_frac = float(footprint.sum().item()) / float(max(1, footprint.numel()))
        if overlap_frac <= float(allow_partial_overlap):
            accepted.append((int(cy), int(cx)))
            occ[y0:y1, x0:x1] = True

    return accepted


def _rejection_sample_targets_with_boxes(
    candidates: list[tuple[int, int]],
    candidate_boxes: list[int],
    num_targets: int,
    h: int,
    w: int,
    device: torch.device,
    max_tries: int = 4096,
    allow_partial_overlap: float = 0.0,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Select targets using each candidate's pre-designated square footprint."""
    if len(candidates) == 0:
        return [], []
    if len(candidate_boxes) != len(candidates):
        raise ValueError("candidate_boxes must have the same length as candidates")

    occ = torch.zeros((h, w), dtype=torch.bool, device="cpu")
    accepted: list[tuple[int, int]] = []
    accepted_boxes: list[int] = []
    tries = 0

    perm = torch.randperm(len(candidates), device="cpu").tolist()
    idx = 0
    while len(accepted) < num_targets and tries < max_tries and idx < len(candidates):
        tries += 1
        src_idx = int(perm[idx])
        idx += 1
        cy, cx = candidates[src_idx]
        box = int(candidate_boxes[src_idx])
        half = box // 2

        y0 = max(0, int(cy) - half)
        y1 = min(h, int(cy) + box - half)
        x0 = max(0, int(cx) - half)
        x1 = min(w, int(cx) + box - half)
        if y1 <= y0 or x1 <= x0:
            continue

        footprint = occ[y0:y1, x0:x1]
        overlap_frac = float(footprint.sum().item()) / float(max(1, footprint.numel()))
        if overlap_frac <= float(allow_partial_overlap):
            accepted.append((int(cy), int(cx)))
            accepted_boxes.append(int(box))
            occ[y0:y1, x0:x1] = True

    return accepted, accepted_boxes


def _effective_mask_box_size(
    sigma: float,
    mask_scale: float,
    mask_box_size: int,
    inner_target_size: int,
    hardcap: int | None = None,
) -> int:
    """Compute mask box size: round(sigma * mask_scale + mask_box_size).

    mask_scale=0  -> constant box (mask_box_size across all scales).
    mask_box_size=0 -> pure pyramid (box proportional to sigma).
    hardcap clamps the result to a maximum (None or 0 = no cap).
    """
    if hardcap is not None and hardcap > 0 and int(hardcap) < int(inner_target_size):
        raise ValueError(
            f"mask_box_hardcap={int(hardcap)} cannot be smaller than "
            f"inner_target_size={int(inner_target_size)}"
        )
    base_box = round(float(sigma) * float(mask_scale) + int(mask_box_size))
    box = max(base_box, int(inner_target_size))
    capped = False
    if hardcap is not None and hardcap > 0:
        if box > int(hardcap):
            box = int(hardcap)
            capped = True
    # bump_up=False when capped: round DOWN to the nearest odd so we never
    # silently exceed the hardcap (e.g. hardcap=16 → _odd_box(16, bump_up=False)=15).
    return _odd_box(box, bump_up=not capped)


def _manual_mask_box_size(
    manual_mask_box_sizes: Sequence[int] | None,
    index: int,
    inner_target_size: int,
    hardcap: int | None = None,
) -> int | None:
    if manual_mask_box_sizes is None:
        return None
    if len(manual_mask_box_sizes) <= 0:
        return None
    raw = manual_mask_box_sizes[min(int(index), len(manual_mask_box_sizes) - 1)]
    box = max(int(round(float(raw))), int(inner_target_size))
    capped = False
    if hardcap is not None and hardcap > 0 and box > int(hardcap):
        box = int(hardcap)
        capped = True
    return _odd_box(box, bump_up=not capped)


def _effective_mask_box_size_for_index(
    index: int,
    sigma: float,
    mask_scale: float,
    mask_box_size: int,
    inner_target_size: int,
    hardcap: int | None = None,
    manual_mask_box_sizes: Sequence[int] | None = None,
) -> int:
    manual = _manual_mask_box_size(
        manual_mask_box_sizes=manual_mask_box_sizes,
        index=index,
        inner_target_size=inner_target_size,
        hardcap=hardcap,
    )
    if manual is not None:
        return manual
    return _effective_mask_box_size(
        sigma=sigma,
        mask_scale=mask_scale,
        mask_box_size=mask_box_size,
        inner_target_size=inner_target_size,
        hardcap=hardcap,
    )


def _max_effective_mask_box_size(
    sigmas: Sequence[float],
    mask_scale: float,
    mask_box_size: int,
    inner_target_size: int,
    hardcap: int | None = None,
    manual_mask_box_sizes: Sequence[int] | None = None,
) -> int:
    boxes = [
        _effective_mask_box_size_for_index(
            index=i,
            sigma=float(s),
            mask_scale=mask_scale,
            mask_box_size=mask_box_size,
            inner_target_size=inner_target_size,
            hardcap=hardcap,
            manual_mask_box_sizes=manual_mask_box_sizes,
        )
        for i, s in enumerate(sigmas)
    ]
    return max(boxes) if boxes else _effective_mask_box_size(
        sigma=1.0,
        mask_scale=mask_scale,
        mask_box_size=mask_box_size,
        inner_target_size=inner_target_size,
        hardcap=hardcap,
    )


def _sample_random_mask_box_sizes(
    n: int,
    mask_box_size_range: tuple[int, int] | None,
    inner_target_size: int,
    hardcap: int | None,
    device: torch.device,
) -> list[int]:
    if n <= 0:
        return []
    if mask_box_size_range is None:
        return []
    lo, hi = sorted((int(mask_box_size_range[0]), int(mask_box_size_range[1])))
    if hi < lo:
        hi = lo
    raw = torch.randint(lo, hi + 1, (int(n),), device=device) if hi > lo else torch.full((int(n),), lo, device=device)
    boxes: list[int] = []
    for v in raw.detach().cpu().tolist():
        box = max(int(round(float(v))), int(inner_target_size))
        capped = False
        if hardcap is not None and hardcap > 0 and box > int(hardcap):
            box = int(hardcap)
            capped = True
        boxes.append(_odd_box(box, bump_up=not capped))
    return boxes


def _ensure_target_patches_masked(
    sample_locations: list[tuple[int, int]],
    mask_map: np.ndarray,
    inner_target_size: int,
) -> None:
    patch_half_lo = int(inner_target_size) // 2
    patch_half_hi = int(inner_target_size) - patch_half_lo
    h, w = mask_map.shape
    for cy, cx in sample_locations:
        y0 = max(0, int(cy) - patch_half_lo)
        y1 = min(h, int(cy) + patch_half_hi)
        x0 = max(0, int(cx) - patch_half_lo)
        x1 = min(w, int(cx) + patch_half_hi)
        if y1 <= y0 or x1 <= x0:
            continue
        if not np.all(mask_map[y0:y1, x0:x1] > 0):
            raise RuntimeError(
                "Target patch is not fully covered by the mask footprint; "
                f"center={(int(cy), int(cx))}, patch_size={int(inner_target_size)}"
            )


def _target_patch_has_valid_input(
    arr: np.ndarray,
    sample_invalid_mask: np.ndarray | None,
    invalid_value_specs: Sequence[object],
    cy: int,
    cx: int,
    inner_target_size: int,
    nan_mask: np.ndarray | None = None,
) -> bool:
    patch_half_lo = int(inner_target_size) // 2
    patch_half_hi = int(inner_target_size) - patch_half_lo
    py0 = int(cy) - patch_half_lo
    py1 = int(cy) + patch_half_hi
    px0 = int(cx) - patch_half_lo
    px1 = int(cx) + patch_half_hi
    h, w = arr.shape
    if py0 < 0 or px0 < 0 or py1 > h or px1 > w:
        return False
    patch = arr[py0:py1, px0:px1]
    if patch.size == 0:
        return False
    invalid_mask = np.zeros_like(patch, dtype=bool)
    if sample_invalid_mask is not None:
        invalid_mask |= sample_invalid_mask[py0:py1, px0:px1]
    # Precomputed NaN mask avoids per-patch isnan() calls
    if nan_mask is not None and "nan" in (str(s).lower() for s in invalid_value_specs if isinstance(s, str)):
        invalid_mask |= nan_mask[py0:py1, px0:px1]
    for spec in invalid_value_specs:
        if isinstance(spec, str) and spec.lower() == "nan":
            if nan_mask is None:
                invalid_mask |= np.isnan(patch)
        else:
            try:
                invalid_mask |= np.isclose(patch, float(spec), equal_nan=False)
            except (TypeError, ValueError):
                continue
    return bool(np.any(~invalid_mask))


def make_pyramid_grid_context(
    x_clean: torch.Tensor,
    sigmas=(2, 4, 8, 16),
    mask_fraction: float = 1.0,
    mask_scale: float = 1.0,
    spacing_scale: float = 1.5,
    global_shift: bool = True,
    align_scales: bool = True,
    mask_box_size: int = 16,
    mask_box_size_range: Optional[Tuple[int, int]] = None,
    random_mask_box_per_target: bool = False,
    manual_mask_box_sizes: Optional[Sequence[int]] = None,
    cdd_mode: str = "log",
    cdd_constrained: bool = True,
    cdd_sm_mode: str = "reflect",
    cdd_append_last_residual: bool = True,
    cdd_pre_log_transform: bool = False,
    cdd_log_eps: float = 1.0,
    cdd_log_std_floor_mult: float = 0.05,
    inner_target_size: int = 2,
    return_debug: bool = False,
    enable_grid_jitter: bool = True,
    enable_target_dithering: bool = True,
    lattice_shift_override: Optional[Tuple[int, int]] = None,
    target_invalid_region_skip: bool = True,
    target_invalid_region_values=(0.0, "nan"),
    invalid_pixel_mask: Optional[torch.Tensor] = None,
    target_sampling_mode: str = "random",
    priority_top_percent: float = 5.0,
    priority_n_target: int | str = 20,
    priority_min_targets_per_map: int = 0,
    priority_dithering_pixels: Optional[int] = None,
    priority_candidate_oversample: float = 3.0,
    target_nonoverlap: bool = True,
    target_allow_partial_overlap: float = 0.0,
    mask_box_hardcap: int | None = None,
    cdd_use_gpu: bool = False,
    cdd_orig_in: Optional[torch.Tensor] = None,
    use_cdd: bool = True,
):
    """
    x_clean: B x 1 x H x W

    Returns:
        x_context: B x 1 x H x W
        target_locations: B x K x 2, storing y,x centers
        target_scales: B x K, storing sigma per location
    """
    if x_clean.dim() != 4:
        raise ValueError(f"Expected BxCxHxW, got {tuple(x_clean.shape)}")
    # Targets must be odd-sized so the target location is the true center pixel.
    inner_target_size = int(inner_target_size)
    if inner_target_size <= 0:
        inner_target_size = 3
    if inner_target_size % 2 == 0:
        inner_target_size = inner_target_size + 1
    if x_clean.shape[1] != 1:
        raise ValueError(f"Expected grayscale input with 1 channel, got {x_clean.shape[1]}")
    sampling_mode = normalize_target_sampling_mode(target_sampling_mode)
    sampled_mode = sampling_mode in ("random", "priority", "priority_small_scale")
    random_sampling_mode = sampling_mode == "random"
    priority_sampling_mode = sampling_mode in ("priority", "priority_small_scale")
    _use_linear_catalogue = sampling_mode == "priority_small_scale"
    if priority_dithering_pixels is None or priority_dithering_pixels <= 0:
        priority_dithering_pixels = inner_target_size
    else:
        priority_dithering_pixels = int(priority_dithering_pixels)
    # Safeguard: global_shift is a lattice/grid concept only.
    # Priority sampling selects targets from ranked pixels, so disable it.
    effective_global_shift = bool(global_shift) if not sampled_mode else False

    b, _, h, w = x_clean.shape
    active_sigmas = tuple(float(s) for s in sigmas)
    if not active_sigmas:
        raise ValueError("sigmas must contain at least one CDD scale")
    n_sigmas = len(active_sigmas)
    total_fraction = max(0.0, float(mask_fraction))
    per_scale_fraction = total_fraction / max(1, n_sigmas)

    x_context = x_clean.clone()
    invalid_value_specs = tuple(target_invalid_region_values) if target_invalid_region_values is not None else tuple()

    all_locations = []
    all_scales = []
    all_valid = []
    all_mask_maps = []
    all_unique_centers = []
    all_cdd_orig = []
    all_cdd_masked = []
    all_dip_fields = []
    all_dip_fields_per_channel = []
    all_dip_proto_per_channel = []
    all_cdd_box_sizes = []
    all_cdd_blur_sigmas = []
    all_priority_good_candidates = []
    all_priority_nonzero_mean = []
    all_priority_prescreen_candidates = []
    all_priority_auto_base_targets = []
    all_priority_effective_targets = []
    all_target_box_sizes = []

    for bi in range(b):
        arr = x_clean[bi, 0].cpu().numpy().copy()
        sample_invalid_mask = invalid_pixel_mask[bi, 0].cpu().numpy() if invalid_pixel_mask is not None else None
        nan_mask = np.isnan(arr)
        priority_good_candidates_bi = 0.0
        priority_nonzero_mean_bi = 1.0
        priority_prescreen_candidates_bi = 0.0
        priority_auto_base_targets_bi = 0.0
        priority_effective_targets_bi = 0.0
        priority_center_boxes: list[int] = []

        applied_locations = []
        applied_scales = []
        applied_boxes = []
        applied_mask_hard = torch.zeros((h, w), dtype=torch.uint8, device=x_clean.device)

        # Compute shared grid centers for scale alignment
        base_box = _max_effective_mask_box_size(
            sigmas=active_sigmas,
            mask_scale=mask_scale,
            mask_box_size=mask_box_size,
            inner_target_size=inner_target_size,
            hardcap=mask_box_hardcap,
            manual_mask_box_sizes=manual_mask_box_sizes,
        )
        base_margin = base_box // 2 + 1
        spacing_px = int(max(1, round(float(base_box) * float(spacing_scale))))
        shared_centers = _shared_grid_centers(
            h=h,
            w=w,
            base_margin=base_margin,
            spacing_px=spacing_px,
            global_shift=effective_global_shift,
            device=x_clean.device,
            enable_grid_jitter=bool(enable_grid_jitter),
            lattice_shift_override=lattice_shift_override,
        )
        shared_centers_dithered = None
        if align_scales and not sampled_mode and len(shared_centers) > 0:
            # Dither once and reuse across scales to keep target/mask centers aligned.
            max_box = _max_effective_mask_box_size(
                sigmas=active_sigmas,
                mask_scale=mask_scale,
                mask_box_size=mask_box_size,
                inner_target_size=inner_target_size,
                hardcap=mask_box_hardcap,
                manual_mask_box_sizes=manual_mask_box_sizes,
            )
            max_half_lo = max_box // 2
            max_half_hi = max_box - max_half_lo
            if enable_target_dithering:
                shared_centers_dithered = []
                for cy0, cx0 in shared_centers:
                    cy1, cx1 = _dither_target_center(
                        cy=int(cy0),
                        cx=int(cx0),
                        h=h,
                        w=w,
                        half_lo=max_half_lo,
                        half_hi=max_half_hi,
                        # Grid/lattice mode always dithers by lattice spacing.
                        dithering_pixels=spacing_px,
                        device=x_clean.device,
                    )
                    shared_centers_dithered.append((int(cy1), int(cx1)))
        if total_fraction <= 0.0:
            shared_centers = []
            shared_centers_dithered = []
        # Apply count budget from mask_fraction for both full-grid and sampled-grid paths.
        if len(shared_centers) > 0:
            base_budget = per_scale_fraction * float(h * w)
            base_desired = base_budget / max(1.0, float(base_box * base_box))
            base_count = int(math.floor(base_desired))
            base_extra = int(torch.rand(1, device=x_clean.device).item() < float(base_desired - base_count))
            base_max_count = max(0, base_count + base_extra)
            if len(shared_centers) > base_max_count:
                idx = torch.randperm(len(shared_centers), device=x_clean.device)[:base_max_count]
                shared_centers = [shared_centers[int(i)] for i in idx]
                if shared_centers_dithered is not None:
                    shared_centers_dithered = [shared_centers_dithered[int(i)] for i in idx]

        use_cdd_for_sample = bool(use_cdd)
        if active_sigmas:
            if cdd_orig_in is not None:
                # Keep everything on GPU — no CPU transfer for CDD data.
                cdd_orig_t = cdd_orig_in[bi].to(device=x_clean.device, dtype=x_clean.dtype)
                cdd_residual_t = None
                use_cdd_for_sample = True
            elif use_cdd_for_sample:
                import constrained_diffusion as cdd

                cdd_kwargs = dict(
                    mode=cdd_mode,
                    constrained=bool(cdd_constrained),
                    sm_mode=cdd_sm_mode,
                    return_scales=False,
                    verbose=False,
                    use_gpu=bool(cdd_use_gpu),
                )
                if cdd_pre_log_transform:
                    eps = max(1e-6, float(cdd_log_eps))
                    arr_clamp = np.clip(arr, 0.0, None)
                    arr_std = float(np.std(arr_clamp))
                    log_floor = max(eps, arr_std * float(cdd_log_std_floor_mult))
                    arr_log = np.log(arr_clamp + log_floor).astype(np.float32)
                else:
                    arr_log = arr.astype(np.float32)
                cdd_channels_arr, cdd_residual = cdd.constrained_diffusion_decomposition(
                    arr_log,
                    num_channels=len(active_sigmas),
                    max_scale=max(active_sigmas),
                    **cdd_kwargs,
                )

                cdd_orig_np = np.clip(np.asarray(cdd_channels_arr, dtype=np.float32), a_min=0.0, a_max=None)
                cdd_orig_t = torch.from_numpy(cdd_orig_np).to(device=x_clean.device, dtype=x_clean.dtype)
                cdd_residual_t = torch.from_numpy(np.asarray(cdd_residual, dtype=np.float32)).to(device=x_clean.device, dtype=x_clean.dtype) if cdd_residual is not None else None

                if cdd_append_last_residual and cdd_residual_t is not None:
                    cdd_orig_t[-1] = cdd_orig_t[-1] + cdd_residual_t
            else:
                cdd_orig_t = x_clean[bi].to(device=x_clean.device, dtype=x_clean.dtype).clone()
                cdd_residual_t = None

            # Mutable mask target stays on GPU
            cdd_mod_t = cdd_orig_t.clone()
            cdd_orig = cdd_orig_t.cpu().numpy()  # CPU copy strictly for catalogue sorting

            if use_cdd_for_sample:
                all_cdd_orig.append(cdd_orig_t.clone().detach())

            priority_catalogue = []
            if sampled_mode:
                if priority_sampling_mode and use_cdd_for_sample:
                    _cdd_for_catalogue = np.expm1(cdd_orig) if _use_linear_catalogue else cdd_orig
                    priority_catalogue = _build_priority_catalogue_from_cdd_ratio(
                        cdd_orig=_cdd_for_catalogue,
                        top_percent=float(priority_top_percent),
                        patch_size=int(inner_target_size),
                        h=h,
                        w=w,
                    )
                elif use_cdd_for_sample:
                    priority_catalogue = _build_random_catalogue_from_cdd(
                        cdd_orig=cdd_orig,
                        patch_size=int(inner_target_size),
                        h=h,
                        w=w,
                    )
                else:
                    priority_catalogue = _build_random_catalogue_from_array(
                        arr=arr,
                        patch_size=int(inner_target_size),
                        h=h,
                        w=w,
                    )
                if len(priority_catalogue) > 0:
                    # Reject candidates too close to boundary for the largest
                    # possible mask footprint across pyramid scales.
                    max_box = _max_effective_mask_box_size(
                        sigmas=active_sigmas,
                        mask_scale=mask_scale,
                        mask_box_size=mask_box_size,
                        inner_target_size=inner_target_size,
                        hardcap=mask_box_hardcap,
                        manual_mask_box_sizes=manual_mask_box_sizes,
                    )
                    candidate_boxes = _sample_random_mask_box_sizes(
                        n=len(priority_catalogue),
                        mask_box_size_range=mask_box_size_range,
                        inner_target_size=inner_target_size,
                        hardcap=mask_box_hardcap,
                        device=x_clean.device,
                    ) if bool(random_mask_box_per_target) and mask_box_size_range is not None else [int(max_box)] * len(priority_catalogue)
                    good_candidates = []
                    good_candidate_boxes = []
                    for (cy, cx), cand_box in zip(priority_catalogue, candidate_boxes):
                        cand_half_lo = int(cand_box) // 2
                        cand_half_hi = int(cand_box) - cand_half_lo
                        y0 = int(cy) - int(cand_half_lo)
                        y1 = int(cy) + int(cand_half_hi)
                        x0 = int(cx) - int(cand_half_lo)
                        x1 = int(cx) + int(cand_half_hi)
                        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                            continue
                        if not _target_patch_has_valid_input(
                            arr=arr,
                            sample_invalid_mask=sample_invalid_mask,
                            invalid_value_specs=invalid_value_specs,
                            cy=int(cy), cx=int(cx),
                            inner_target_size=inner_target_size,
                            nan_mask=nan_mask,
                        ):
                            continue
                        good_candidates.append((int(cy), int(cx)))
                        good_candidate_boxes.append(int(cand_box))
                    priority_catalogue = good_candidates
                    candidate_boxes = good_candidate_boxes
                    if priority_sampling_mode:
                        budget_box = int(round(float(np.mean(candidate_boxes)))) if candidate_boxes else int(max_box)
                        prescreen_count = _fractional_spatial_target_budget(
                            height=h,
                            width=w,
                            box_size=budget_box,
                            oversample=float(priority_candidate_oversample),
                            device=x_clean.device,
                            minimum=int(priority_min_targets_per_map),
                        )
                        if prescreen_count is not None and len(priority_catalogue) > prescreen_count:
                            perm = torch.randperm(len(priority_catalogue), device=x_clean.device)[:prescreen_count]
                            priority_catalogue = [priority_catalogue[int(i)] for i in perm]
                            candidate_boxes = [candidate_boxes[int(i)] for i in perm]
                    priority_prescreen_candidates_bi = float(len(priority_catalogue))

                    # Rule of thumb: start from image_area / max_mask_area.
                    # The candidate list decides where targets may land; auto
                    # target count should not depend on rank density artifacts.
                    nonzero_mean = 1.0
                    budget_box = int(round(float(np.mean(candidate_boxes)))) if candidate_boxes else int(max_box)
                    auto_base = _fractional_spatial_target_budget(
                        height=h,
                        width=w,
                        box_size=budget_box,
                        oversample=1.0,
                        device=x_clean.device,
                        minimum=0,
                    ) or 0

                    priority_n_raw = priority_n_target
                    if isinstance(priority_n_raw, str) and priority_n_raw.strip().lower() == "auto":
                        base_targets_unscaled = auto_base
                    else:
                        try:
                            base_targets_unscaled = int(round(float(priority_n_raw)))
                        except (TypeError, ValueError):
                            base_targets_unscaled = 0
                    min_targets = max(0, int(priority_min_targets_per_map))
                    base_targets_scaled = max(0, int(round(float(base_targets_unscaled) * float(total_fraction))))
                    base_targets = max(min_targets, base_targets_scaled)
                    k_sel = min(base_targets, len(priority_catalogue))
                    # Shuffle first, then keep extra seeds because explicit
                    # overlap rejection after dithering may discard some.
                    perm = torch.randperm(len(priority_catalogue), device=x_clean.device)
                    preselect_mult = 8 if bool(target_nonoverlap) else 1
                    selected_idx = perm[:min(k_sel * preselect_mult, len(priority_catalogue))]
                    priority_catalogue = [
                        priority_catalogue[int(i)]
                        for i in selected_idx
                    ]
                    candidate_boxes = [candidate_boxes[int(i)] for i in selected_idx]
                    priority_center_boxes = list(candidate_boxes)
                    priority_good_candidates_bi = float(len(good_candidates))
                    priority_nonzero_mean_bi = float(nonzero_mean)
                    priority_auto_base_targets_bi = float(auto_base)
                    priority_effective_targets_bi = float(k_sel)  # updated below after non-overlap
            # Dither once per selected priority seed and reuse across scales.
            # This avoids per-scale micro-clusters around the same logical target.
            priority_centers_dithered: list[tuple[int, int]] = []
            if sampled_mode and len(priority_catalogue) > 0:
                patch_half_lo = int(inner_target_size) // 2
                patch_half_hi = int(inner_target_size) - patch_half_lo
                dithered_boxes: list[int] = []
                for idx0, (cy0, cx0) in enumerate(priority_catalogue):
                    cand_box = int(priority_center_boxes[idx0]) if idx0 < len(priority_center_boxes) else max_box
                    cand_half_lo = cand_box // 2
                    cand_half_hi = cand_box - cand_half_lo
                    cy1, cx1 = _dither_target_center(
                        cy=int(cy0),
                        cx=int(cx0),
                        h=h,
                        w=w,
                        half_lo=cand_half_lo,
                        half_hi=cand_half_hi,
                        dithering_pixels=priority_dithering_pixels,
                        device=x_clean.device,
                    )
                    priority_centers_dithered.append((int(cy1), int(cx1)))
                    dithered_boxes.append(int(cand_box))

                # Non-overlap enforcement on the *dithered* centers so that
                # dithering cannot undo the protection.
                if bool(target_nonoverlap) and len(priority_centers_dithered) > 1:
                    if bool(random_mask_box_per_target):
                        priority_centers_dithered, dithered_boxes = _rejection_sample_targets_with_boxes(
                            candidates=priority_centers_dithered,
                            candidate_boxes=dithered_boxes,
                            num_targets=k_sel,
                            h=h,
                            w=w,
                            device=x_clean.device,
                            allow_partial_overlap=float(target_allow_partial_overlap),
                        )
                    else:
                        priority_centers_dithered = _rejection_sample_targets(
                            candidates=priority_centers_dithered,
                            num_targets=k_sel,
                            h=h,
                            w=w,
                            exclusion_box=max_box,
                            device=x_clean.device,
                            allow_partial_overlap=float(target_allow_partial_overlap),
                        )
                        dithered_boxes = dithered_boxes[:len(priority_centers_dithered)]
                    priority_effective_targets_bi = float(len(priority_centers_dithered))
                priority_center_boxes = dithered_boxes

            num_cdd_ch = cdd_mod_t.shape[0]
            dip_field_t = torch.zeros((h, w), dtype=x_clean.dtype, device=x_clean.device)
            dip_field_ch_t = torch.zeros((num_cdd_ch, h, w), dtype=x_clean.dtype, device=x_clean.device)
            dip_proto_ch_t = torch.zeros((num_cdd_ch, h, w), dtype=x_clean.dtype, device=x_clean.device)
            dip_proto_written = torch.zeros(num_cdd_ch, dtype=torch.int32, device=x_clean.device)
            cdd_box_sizes = []
            cdd_blur_sigmas = []

            for si, sigma in enumerate(active_sigmas):
                box = _effective_mask_box_size_for_index(
                    index=si,
                    sigma=float(sigma),
                    mask_scale=mask_scale,
                    mask_box_size=mask_box_size,
                    inner_target_size=inner_target_size,
                    hardcap=mask_box_hardcap,
                    manual_mask_box_sizes=manual_mask_box_sizes,
                )
                ch = min(si, cdd_mod_t.shape[0] - 1)
                half_lo = box // 2
                half_hi = box - half_lo
                cdd_box_sizes.append(float(box))
                cdd_blur_sigmas.append(0.0)
                if sampled_mode and len(priority_centers_dithered) > 0:
                    centers = priority_centers_dithered
                elif align_scales:
                    centers = shared_centers_dithered if shared_centers_dithered is not None else shared_centers
                else:
                    spacing = int(max(1, round(float(box) * float(spacing_scale))))
                    margin = max(half_lo, half_hi) + 1
                    area_budget = per_scale_fraction * float(h * w)
                    desired_count = area_budget / max(1.0, float(box * box))
                    base_count = int(math.floor(desired_count))
                    frac = float(desired_count - base_count)
                    extra = int(torch.rand(1, device=x_clean.device).item() < frac)
                    max_count = max(0, base_count + extra)
                    if max_count <= 0:
                        continue
                    if lattice_shift_override is not None:
                        shift_y = int(lattice_shift_override[0]) % spacing
                        shift_x = int(lattice_shift_override[1]) % spacing
                    else:
                        shift_y = int(torch.randint(0, max(1, spacing), (1,), device=x_clean.device).item())
                        shift_x = int(torch.randint(0, max(1, spacing), (1,), device=x_clean.device).item())
                    y_start = margin + shift_y
                    x_start = margin + shift_x
                    y_centers = list(range(y_start, max(y_start + 1, h - margin), spacing))
                    x_centers = list(range(x_start, max(x_start + 1, w - margin), spacing))
                    centers = [(cy, cx) for cy in y_centers for cx in x_centers]
                    if len(centers) > max_count:
                        idx = torch.randperm(len(centers), device=x_clean.device)[:max_count]
                        centers = [centers[int(i)] for i in idx]

                for center_idx, (cy, cx) in enumerate(centers):
                    applied_box = int(box)
                    if sampled_mode and bool(random_mask_box_per_target) and center_idx < len(priority_center_boxes):
                        applied_box = int(priority_center_boxes[center_idx])
                        half_lo = applied_box // 2
                        half_hi = applied_box - half_lo
                    # Priority/random modes dither centers once above.
                    if enable_target_dithering and not (
                        (sampled_mode and len(priority_centers_dithered) > 0)
                        or align_scales
                    ):
                        cy, cx = _dither_target_center(
                            cy=int(cy),
                            cx=int(cx),
                            h=h,
                            w=w,
                            half_lo=half_lo,
                            half_hi=half_hi,
                            # Grid/lattice mode always dithers by lattice spacing.
                            dithering_pixels=spacing,
                            device=x_clean.device,
                        )
                    y0 = max(0, cy - half_lo)
                    y1 = min(h, cy + half_hi)
                    x0 = max(0, cx - half_lo)
                    x1 = min(w, cx + half_hi)
                    if y1 <= y0 or x1 <= x0:
                        continue
                    cdd_mod_t[ch, y0:y1, x0:x1] = 0.0
                    dip_field_t[y0:y1, x0:x1] = 1.0
                    dip_field_ch_t[ch, y0:y1, x0:x1] = 1.0
                    if dip_proto_written[ch] == 0:
                        dip_proto_ch_t[ch, y0:y1, x0:x1] = 1.0
                        dip_proto_written[ch] = 1
                    applied_mask_hard[y0:y1, x0:x1] = 1
                    applied_locations.append((cy, cx))
                    applied_scales.append(float(sigma))
                    applied_boxes.append(float(applied_box))

            # Reconstruct entirely on GPU
            recon_t = torch.sum(cdd_mod_t, dim=0)
            if cdd_residual_t is not None and not cdd_append_last_residual:
                recon_t = recon_t + cdd_residual_t
            x_context[bi, 0] = torch.clamp(recon_t, min=0.0)

            # IMPORTANT:
            # In priority mode we still must keep the *dithered* centers.
            # applied_locations/applied_scales are populated after dithering,
            # while priority_catalogue holds the pre-dither seed centers.
            if sampled_mode and len(priority_centers_dithered) > 0:
                unique_loc_to_scale = {(int(cy), int(cx)): float(active_sigmas[0]) for cy, cx in priority_centers_dithered}
            else:
                sample_locations = list(applied_locations)
                sample_scales = list(applied_scales)
                unique_loc_to_scale = {}
                for (cy, cx), s in zip(sample_locations, sample_scales):
                    key = (int(cy), int(cx))
                    if key not in unique_loc_to_scale:
                        unique_loc_to_scale[key] = float(s)
            sample_locations = []
            sample_scales = []
            patch_half_lo = int(inner_target_size) // 2
            patch_half_hi = int(inner_target_size) - patch_half_lo
            for cy, cx in unique_loc_to_scale.keys():
                iy = int(cy)
                ix = int(cx)
                if iy - patch_half_lo < 0 or ix - patch_half_lo < 0:
                    continue
                if iy + patch_half_hi > h or ix + patch_half_hi > w:
                    continue
                if bool(target_invalid_region_skip) or sampled_mode:
                    if not _target_patch_has_valid_input(
                        arr=arr,
                        sample_invalid_mask=sample_invalid_mask,
                        invalid_value_specs=invalid_value_specs,
                        cy=iy, cx=ix,
                        inner_target_size=inner_target_size,
                        nan_mask=nan_mask,
                    ):
                        continue
                sample_locations.append((iy, ix))
                sample_scales.append(float(unique_loc_to_scale[(cy, cx)]))
            sample_valid = [1] * len(sample_locations)
            _ensure_target_patches_masked(sample_locations, applied_mask_hard.cpu().numpy(), inner_target_size)

            all_locations.append(sample_locations)
            all_scales.append(sample_scales)
            all_valid.append(sample_valid)
            if return_debug:
                uniq = []
                seen = set()
                for cy, cx in applied_locations:
                    key = (int(cy), int(cx))
                    if key not in seen:
                        seen.add(key)
                        uniq.append((cy, cx))
                all_unique_centers.append(torch.tensor(uniq, dtype=torch.long))
            all_mask_maps.append(applied_mask_hard.cpu().clone())
            all_cdd_masked.append(cdd_mod_t.cpu().clone())
            all_dip_fields.append(dip_field_t.cpu())
            all_dip_fields_per_channel.append(dip_field_ch_t.cpu())
            all_dip_proto_per_channel.append(dip_proto_ch_t.cpu())
            all_cdd_box_sizes.append(torch.tensor(cdd_box_sizes, dtype=torch.float32))
            all_cdd_blur_sigmas.append(torch.tensor(cdd_blur_sigmas, dtype=torch.float32))
            all_target_box_sizes.append(list(applied_boxes))
            all_priority_good_candidates.append(priority_good_candidates_bi)
            all_priority_nonzero_mean.append(priority_nonzero_mean_bi)
            all_priority_prescreen_candidates.append(priority_prescreen_candidates_bi)
            all_priority_auto_base_targets.append(priority_auto_base_targets_bi)
            all_priority_effective_targets.append(priority_effective_targets_bi)

            continue

    # Pack variable-length targets to fixed K so batching is always valid.
    k_fixed = max((len(v) for v in all_locations), default=0)
    k_fixed = max(1, k_fixed)

    loc_np = np.zeros((b, k_fixed, 2), dtype=np.int64)
    sca_np = np.zeros((b, k_fixed), dtype=np.float32)
    val_np = np.zeros((b, k_fixed), dtype=np.bool_)
    box_np = np.zeros((b, k_fixed), dtype=np.float32)

    for bi in range(b):
        n_total = len(all_locations[bi])
        n = min(n_total, k_fixed)
        if n <= 0:
            continue
        loc_np[bi, :n, :] = np.asarray(all_locations[bi][:n], dtype=np.int64)
        sca_np[bi, :n] = np.asarray(all_scales[bi][:n], dtype=np.float32)
        val_np[bi, :n] = True
        if bi < len(all_target_box_sizes) and len(all_target_box_sizes[bi]) > 0:
            box_np[bi, :n] = np.asarray(all_target_box_sizes[bi][:n], dtype=np.float32)

    target_locations = torch.from_numpy(loc_np).to(device=x_clean.device, dtype=torch.long)
    target_scales = torch.from_numpy(sca_np).to(device=x_clean.device, dtype=x_clean.dtype)
    target_valid = torch.from_numpy(val_np).to(device=x_clean.device, dtype=torch.bool)
    target_box_sizes = torch.from_numpy(box_np).to(device=x_clean.device, dtype=x_clean.dtype)

    if not return_debug:
        return x_context, target_locations, target_scales, target_valid

    max_centers = max((int(t.shape[0]) for t in all_unique_centers), default=0)
    centers_pad = torch.full((b, max_centers, 2), -1, dtype=torch.long, device=x_clean.device)
    for bi, t in enumerate(all_unique_centers):
        if t.numel() > 0:
            centers_pad[bi, :t.shape[0]] = t.to(device=x_clean.device)

    def _safe_stack(tensor_list):
        if not tensor_list:
            return torch.empty(0, device=x_clean.device, dtype=x_clean.dtype)
        return torch.stack([t.to(device=x_clean.device, dtype=x_clean.dtype) for t in tensor_list], dim=0)

    debug = {
        "mask_map": torch.stack([m.to(device=x_clean.device) for m in all_mask_maps], dim=0),
        "unique_centers": centers_pad,
        "cdd_channels_orig": _safe_stack(all_cdd_orig),
        "cdd_channels_masked": _safe_stack(all_cdd_masked),
        "dip_field": _safe_stack(all_dip_fields),
        "dip_field_per_channel": _safe_stack(all_dip_fields_per_channel),
        "dip_proto_per_channel": _safe_stack(all_dip_proto_per_channel),
        "cdd_box_sizes": _safe_stack(all_cdd_box_sizes),
        "cdd_blur_sigmas": _safe_stack(all_cdd_blur_sigmas),
        "target_box_sizes": target_box_sizes,
        "priority_good_candidates": torch.tensor(all_priority_good_candidates, dtype=x_clean.dtype, device=x_clean.device),
        "priority_nonzero_mean": torch.tensor(all_priority_nonzero_mean, dtype=x_clean.dtype, device=x_clean.device),
        "priority_prescreen_candidates": torch.tensor(all_priority_prescreen_candidates, dtype=x_clean.dtype, device=x_clean.device),
        "priority_auto_base_targets": torch.tensor(all_priority_auto_base_targets, dtype=x_clean.dtype, device=x_clean.device),
        "priority_effective_targets": torch.tensor(all_priority_effective_targets, dtype=x_clean.dtype, device=x_clean.device),
        "mask_scale_factor": torch.tensor(float(mask_scale), dtype=x_clean.dtype, device=x_clean.device),
        "mask_footprint_px": torch.tensor(float(mask_box_size), dtype=x_clean.dtype, device=x_clean.device),
        "random_mask_box_per_target": torch.tensor(float(bool(random_mask_box_per_target)), dtype=x_clean.dtype, device=x_clean.device),
    }
    return x_context, target_locations, target_scales, target_valid, debug


def prepare_context_batch(
    x_clean: torch.Tensor,
    *,
    sigmas,
    mask_fraction: float = 1.0,
    mask_scale: float = 1.0,
    spacing_scale: float = 1.5,
    global_shift: bool = True,
    align_scales: bool = True,
    mask_box_size: int = 16,
    mask_box_size_range: Optional[Tuple[int, int]] = None,
    random_mask_box_per_target: bool = False,
    manual_mask_box_sizes: Optional[Sequence[int]] = None,
    cdd_mode: str = "log",
    cdd_constrained: bool = True,
    cdd_sm_mode: str = "reflect",
    cdd_append_last_residual: bool = True,
    cdd_pre_log_transform: bool = False,
    cdd_log_eps: float = 1.0,
    cdd_log_std_floor_mult: float = 0.05,
    patch_size: int = 3,
    return_debug: bool = False,
    enable_grid_jitter: bool = True,
    enable_target_dithering: bool = True,
    lattice_shift_override: Optional[Tuple[int, int]] = None,
    target_invalid_region_skip: bool = True,
    target_invalid_region_values=(0.0, "nan"),
    target_sampling_mode: str = "random",
    priority_top_percent: float = 5.0,
    priority_n_target: int | str = 20,
    priority_min_targets_per_map: int = 0,
    priority_dithering_pixels: Optional[int] = None,
    priority_candidate_oversample: float = 3.0,
    target_nonoverlap: bool = True,
    target_allow_partial_overlap: float = 0.0,
    mask_box_hardcap: int | None = None,
    cdd_use_gpu: bool = False,
    cdd_orig_in: Optional[torch.Tensor] = None,
    use_cdd: bool = True,
):
    """Prepare context tensors from a clean batch.

    Handles NaN detection + scrubbing before masking so the downstream network
    receives pre-computed context.  Safe to call from DataLoader collate workers
    or the main process (no CUDA requirement).
    """
    invalid_pixel_mask = ~torch.isfinite(x_clean)
    if invalid_pixel_mask.any():
        x_clean = torch.nan_to_num(x_clean, nan=0.0, posinf=0.0, neginf=0.0)

    return make_pyramid_grid_context(
        x_clean=x_clean,
        sigmas=sigmas,
        mask_fraction=mask_fraction,
        mask_scale=mask_scale,
        spacing_scale=spacing_scale,
        global_shift=global_shift,
        align_scales=align_scales,
        mask_box_size=mask_box_size,
        mask_box_size_range=mask_box_size_range,
        random_mask_box_per_target=random_mask_box_per_target,
        manual_mask_box_sizes=manual_mask_box_sizes,
        cdd_mode=cdd_mode,
        cdd_constrained=cdd_constrained,
        cdd_sm_mode=cdd_sm_mode,
        cdd_append_last_residual=cdd_append_last_residual,
        cdd_pre_log_transform=cdd_pre_log_transform,
        cdd_log_eps=cdd_log_eps,
        cdd_log_std_floor_mult=cdd_log_std_floor_mult,
        inner_target_size=patch_size,
        return_debug=return_debug,
        enable_grid_jitter=enable_grid_jitter,
        enable_target_dithering=enable_target_dithering,
        lattice_shift_override=lattice_shift_override,
        target_invalid_region_skip=target_invalid_region_skip,
        target_invalid_region_values=target_invalid_region_values,
        invalid_pixel_mask=invalid_pixel_mask,
        target_sampling_mode=target_sampling_mode,
        priority_top_percent=priority_top_percent,
        priority_n_target=priority_n_target,
        priority_min_targets_per_map=priority_min_targets_per_map,
        priority_dithering_pixels=priority_dithering_pixels,
        priority_candidate_oversample=priority_candidate_oversample,
        target_nonoverlap=target_nonoverlap,
        target_allow_partial_overlap=target_allow_partial_overlap,
        mask_box_hardcap=mask_box_hardcap,
        cdd_use_gpu=cdd_use_gpu,
        cdd_orig_in=cdd_orig_in,
        use_cdd=use_cdd,
    )


def extract_location_patches(
    z: torch.Tensor,
    locations: torch.Tensor,
    patch_size: int,
):
    """
    z:         B x C x H x W
    locations: B x K x 2, y/x centers

    Returns:
        patches: B x K x C x patch_size x patch_size
    """
    b, c, h, w = z.shape
    _, k, _ = locations.shape

    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if patch_size > h or patch_size > w:
        raise ValueError(f"patch_size={patch_size} exceeds feature map size {(h, w)}")

    half = patch_size // 2
    half_hi = patch_size - half

    y0 = locations[:, :, 0] - half  # B x K
    x0 = locations[:, :, 1] - half  # B x K

    valid = (y0 >= 0) & (x0 >= 0) & (y0 + patch_size <= h) & (x0 + patch_size <= w)  # B x K

    dy = torch.arange(patch_size, device=z.device)  # P
    dx = torch.arange(patch_size, device=z.device)  # P

    y_idx = y0.view(b, k, 1, 1) + dy.view(1, 1, patch_size, 1)    # B x K x P x 1
    x_idx = x0.view(b, k, 1, 1) + dx.view(1, 1, 1, patch_size)    # B x K x 1 x P

    y_idx = y_idx.clamp(0, h - 1)
    x_idx = x_idx.clamp(0, w - 1)

    # Use broadcasting in advanced indexing to avoid materializing large
    # expanded integer index tensors.
    b_idx = torch.arange(b, device=z.device).view(b, 1, 1, 1, 1)
    c_idx = torch.arange(c, device=z.device).view(1, 1, c, 1, 1)
    y_idx = y_idx.unsqueeze(2)  # Actual shape: B x K x 1 x P x 1
    x_idx = x_idx.unsqueeze(2)  # Actual shape: B x K x 1 x 1 x P

    patches = z[b_idx, c_idx, y_idx, x_idx]  # B x K x C x P x P
    valid_mask = valid.view(b, k, 1, 1, 1)
    patches = torch.where(valid_mask, patches, torch.zeros_like(patches))

    return patches


================================================================================
FILE: src/models/build_jepa.py
================================================================================

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import (
    CDDScaleAwareConvNeXtEncoder,
    ConvNeXtDenseEncoder,
    LayerNorm2d,
)
from .masking import (
    extract_location_patches,
    make_pyramid_grid_context,
    norm_per_sample_channel,
    normalize_target_sampling_mode,
    prepare_context_batch,
)
from .predictor import FullResPredictor
from .symmetry import symmetric_forward_2d
from src.losses import l2_normalize_patches

# Shared encoder-type sets used by both build_jepa.py and train.py.
CDD_CUBE_ENCODER_TYPES = frozenset({
    "cdd_scaleaware_convnext",
    "cdd_scaleaware_convnext3d",
    "convnext_dense_pyramid",
    "escnn_c4_pyramid",
})

CDD_DEBUG_ENCODER_TYPES = frozenset(CDD_CUBE_ENCODER_TYPES)
MASK_MAP_ENCODER_TYPES = frozenset({"convnext_dense_masktoken"})


class PyramidGridJEPA(nn.Module):
    def __init__(
        self,
        latent_channels: int = 32,
        predictor_hidden: int = None,
        patch_size: int = 3,
        sigmas=(2, 4, 8, 16),
        mask_fraction: float = 1.0,
        mask_scale: float = 1.0,
        mask_scale_range=None,
        spacing_scale: float = 1.5,
        global_shift: bool = True,
        align_scales: bool = True,
        mask_box_size: int = 16,
        mask_box_size_range=None,
        random_mask_box_per_target: bool = False,
        manual_mask_box_sizes=None,
        cdd_mode: str = "log",
        cdd_constrained: bool = True,
        cdd_sm_mode: str = "reflect",
        cdd_append_last_residual: bool = True,
        cdd_pre_log_transform: bool = False,
        post_log_transform: bool = True,
        log_eps: float = 1.0,
        cdd_log_std_floor_mult: float = 0.05,
        ema_momentum: float = 0.996,
        normalize_loss_l2: bool = False,
        predictor_layernorm: bool = True,
        predictor_spatial_conv: bool = False,
        projector_conv: bool = True,
        predictor_residual: bool = False,
        mode: str = "image",
        encoder_type: str = "convnext_dense_masktoken",
        encoder_width: int = 32,
        encoder_depth: int = 4,
        encoder_kernel_size: int = 7,
        convnext_layer_dilations=None,
        encoder_norm_type: Optional[str] = None,
        encoder_norm_groups: Optional[int] = None,
        encoder_norm_eps: Optional[float] = None,
        scaleaware_feat_channels: int = 8,
        scaleaware_adapter_kernel_size: int = 3,
        scaleaware_fusion_type: str = "concat",
        scaleaware_norm_per_scale: bool = False,
        scaleaware_adapter_norm: bool = True,
        scaleaware_final_norm: bool = True,
        scaleaware_stem_norm: bool = True,
        encoder_final_norm_type: str = "layernorm",
        encoder_head_bias: bool = True,
        target_invalid_region_skip: bool = True,
        target_invalid_region_values=(0.0, "nan"),
        target_sampling_mode: str = "random",
        priority_top_percent: float = 5.0,
        priority_n_target: int | str = 20,
        priority_min_targets_per_map: int = 0,
        priority_dithering_pixels: int = 6,
        priority_candidate_oversample: float = 3.0,
        use_symmetric_feature_loss: bool = False,
        target_nonoverlap: bool = True,
        target_allow_partial_overlap: float = 0.0,
        mask_box_hardcap: int | None = None,
        use_grn: bool = True,
    ):
        super().__init__()

        p = int(patch_size)
        if p <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size!r}.")
        if p % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {patch_size!r}.")
        self.patch_size = p
        self.sigmas = tuple(sigmas)
        self.mask_fraction = float(mask_fraction)
        mask_scale_value, inline_mask_scale_range = self._split_float_param(mask_scale, 1.0, "mask_scale")
        if mask_scale_range is not None and inline_mask_scale_range is not None:
            raise ValueError("Specify either mask_scale as a range or mask_scale_range, not both.")
        self.mask_scale = mask_scale_value
        self.mask_scale_range = self._coerce_float_range(
            mask_scale_range if mask_scale_range is not None else inline_mask_scale_range,
            "mask_scale_range",
        )
        self.spacing_scale = float(spacing_scale)
        self.global_shift = bool(global_shift)
        self.align_scales = bool(align_scales)
        mask_box_size_value, inline_mask_box_size_range = self._split_int_param(
            mask_box_size,
            16,
            "mask_box_size",
        )
        if mask_box_size_range is not None and inline_mask_box_size_range is not None:
            raise ValueError("Specify either mask_box_size as a range or mask_box_size_range, not both.")
        self.mask_box_size = mask_box_size_value
        self.mask_box_size_range = self._coerce_int_range(
            mask_box_size_range if mask_box_size_range is not None else inline_mask_box_size_range,
            "mask_box_size_range",
        )
        self.random_mask_box_per_target = bool(random_mask_box_per_target)
        self.manual_mask_box_sizes = self._coerce_manual_mask_box_sizes(manual_mask_box_sizes)
        if self.manual_mask_box_sizes is not None:
            if len(self.manual_mask_box_sizes) < len(self.sigmas):
                print(
                    "[warning] manual_mask_box_sizes shorter than sigmas/CDD channels; "
                    f"reusing last size for remaining channels: {self.manual_mask_box_sizes}"
                )
            elif len(self.manual_mask_box_sizes) > len(self.sigmas):
                print(
                    "[warning] manual_mask_box_sizes longer than sigmas/CDD channels; "
                    f"extra sizes will be ignored: {self.manual_mask_box_sizes}"
                )
        self.cdd_mode = str(cdd_mode)
        self.cdd_constrained = bool(cdd_constrained)
        self.cdd_sm_mode = str(cdd_sm_mode)
        self.cdd_append_last_residual = bool(cdd_append_last_residual)
        self.cdd_pre_log_transform = bool(cdd_pre_log_transform)
        self.post_log_transform = bool(post_log_transform)
        self.log_eps = float(log_eps)
        self.cdd_log_std_floor_mult = float(cdd_log_std_floor_mult)
        self.ema_momentum = float(ema_momentum)
        self.normalize_loss_l2 = bool(normalize_loss_l2)
        self.predictor_layernorm = bool(predictor_layernorm)
        self.predictor_spatial_conv = bool(predictor_spatial_conv)
        self.predictor_residual = bool(predictor_residual)
        self.mode = str(mode)
        self.encoder_type = str(encoder_type)
        self.encoder_width = int(encoder_width)
        self.encoder_depth = int(encoder_depth)
        self.encoder_kernel_size = int(encoder_kernel_size)
        self.convnext_layer_dilations = (
            None if convnext_layer_dilations is None else tuple(int(d) for d in convnext_layer_dilations)
        )
        self.encoder_norm_type = None if encoder_norm_type is None else str(encoder_norm_type).lower()
        self.encoder_norm_groups = None if encoder_norm_groups is None else int(encoder_norm_groups)
        self.encoder_norm_eps = None if encoder_norm_eps is None else float(encoder_norm_eps)
        self.scaleaware_feat_channels = int(scaleaware_feat_channels)
        self.scaleaware_adapter_kernel_size = int(scaleaware_adapter_kernel_size)
        self.scaleaware_fusion_type = str(scaleaware_fusion_type)
        self.scaleaware_norm_per_scale = bool(scaleaware_norm_per_scale)
        self.scaleaware_adapter_norm = bool(scaleaware_adapter_norm)
        self.scaleaware_final_norm = bool(scaleaware_final_norm)
        self.scaleaware_stem_norm = bool(scaleaware_stem_norm)
        self.encoder_final_norm_type = str(encoder_final_norm_type).lower()
        self.encoder_head_bias = bool(encoder_head_bias)
        self.use_grn = bool(use_grn)
        self.target_invalid_region_skip = bool(target_invalid_region_skip)
        if target_invalid_region_values is None:
            self.target_invalid_region_values = (0.0, "nan")
        else:
            self.target_invalid_region_values = tuple(target_invalid_region_values)
        self.target_sampling_mode = normalize_target_sampling_mode(str(target_sampling_mode))
        self.priority_top_percent = float(priority_top_percent)
        # Keep raw value to support non-numeric modes such as "auto".
        self.priority_n_target = priority_n_target
        self.priority_min_targets_per_map = int(priority_min_targets_per_map)
        self.priority_dithering_pixels = int(priority_dithering_pixels)
        self.priority_candidate_oversample = float(priority_candidate_oversample)
        self.use_symmetric_feature_loss = bool(use_symmetric_feature_loss)
        self.target_nonoverlap = bool(target_nonoverlap)
        self.target_allow_partial_overlap = float(target_allow_partial_overlap)
        self.mask_box_hardcap = None if mask_box_hardcap is None else int(mask_box_hardcap)
        self.projector_conv = bool(projector_conv)
        if self.mode not in ("image", "pyramid"):
            raise ValueError(f"Unknown mode={self.mode}; expected 'image' or 'pyramid'")
        if self.encoder_type == "convnext_dense_masktoken":
            if self.mode != "image":
                raise ValueError(f"{self.encoder_type} requires mode='image'.")
        if self.encoder_type == "cdd_scaleaware_convnext":
            if self.mode != "pyramid":
                raise ValueError("cdd_scaleaware_convnext requires mode='pyramid'.")
            self.context_encoder = CDDScaleAwareConvNeXtEncoder(
                scales=tuple(float(s) for s in self.sigmas),
                hidden_channels=self.encoder_width,
                latent_channels=latent_channels,
                depth=self.encoder_depth,
                kernel_size=self.encoder_kernel_size,
                expansion=4,
                scale_feat_channels=self.scaleaware_feat_channels,
                adapter_kernel_size=self.scaleaware_adapter_kernel_size,
                fusion_type=self.scaleaware_fusion_type,
                use_reflect_padding=True,
                final_norm=self.scaleaware_final_norm,
                final_norm_type=self.encoder_final_norm_type,
                head_bias=self.encoder_head_bias,
                cdd_append_last_residual=self.cdd_append_last_residual,
                adapter_norm=self.scaleaware_adapter_norm,
                use_grn=self.use_grn,
                stem_norm=self.scaleaware_stem_norm,
                dilations=self.convnext_layer_dilations,
            )
        elif self.encoder_type == "convnext_dense_pyramid":
            if self.mode != "pyramid":
                raise ValueError("convnext_dense_pyramid requires mode='pyramid'.")
            pyr_in_channels = 2 * max(1, len(self.sigmas))
            self.context_encoder = ConvNeXtDenseEncoder(
                in_channels=pyr_in_channels,
                hidden_channels=self.encoder_width,
                latent_channels=latent_channels,
                depth=self.encoder_depth,
                kernel_size=self.encoder_kernel_size,
                expansion=4,
                use_reflect_padding=True,
                final_norm=True,
                use_grn=self.use_grn,
                dilations=self.convnext_layer_dilations,
            )
        elif self.encoder_type == "escnn_c4_pyramid":
            if self.mode != "pyramid":
                raise ValueError(f"{self.encoder_type} requires mode='pyramid'.")
            pyr_in_channels = 2 * max(1, len(self.sigmas))
            self.context_encoder = EscnnC4PyramidEncoder(
                in_channels=pyr_in_channels,
                hidden_channels=self.encoder_width,
                latent_channels=latent_channels,
                depth=self.encoder_depth,
                kernel_size=self.encoder_kernel_size,
                final_norm=self.scaleaware_final_norm,
                final_norm_type=self.encoder_final_norm_type,
            )
        elif self.encoder_type == "convnext_dense_masktoken":
            # 2D ConvNeXt image mode with explicit hard-mask token channel.
            self.context_encoder = ConvNeXtDenseEncoder(
                in_channels=2,
                hidden_channels=self.encoder_width,
                latent_channels=latent_channels,
                depth=self.encoder_depth,
                kernel_size=self.encoder_kernel_size,
                expansion=4,
                use_reflect_padding=True,
                final_norm=True,
                use_grn=self.use_grn,
                dilations=self.convnext_layer_dilations,
            )
        else:
            raise ValueError(f"Unknown encoder_type={self.encoder_type}")

        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        if predictor_hidden is None:
            predictor_hidden = latent_channels * 2
        if self.projector_conv:
            self.projector = nn.Sequential(
                nn.Conv2d(latent_channels, int(predictor_hidden), kernel_size=1),
                LayerNorm2d(int(predictor_hidden)) if self.predictor_layernorm else nn.Identity(),
                nn.GELU(),
                nn.Conv2d(int(predictor_hidden), latent_channels, kernel_size=1),
            )
        else:
            self.projector = nn.Identity()
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False
        # For D4 encoders, keep predictor point-wise to avoid reintroducing
        # post-encoder directional spatial derivatives.
        pred_ks = 1 if "_d4" in self.encoder_type else 3
        self.predictor = FullResPredictor(
            channels=latent_channels,
            hidden=int(predictor_hidden),
            use_layernorm=self.predictor_layernorm,
            spatial_conv=self.predictor_spatial_conv,
            residual=self.predictor_residual,
            kernel_size=pred_ks,
        )

    @staticmethod
    def _coerce_float_range(value, name: str):
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must be a list/tuple of exactly two values, got {value!r}")
        lo, hi = sorted((float(value[0]), float(value[1])))
        return lo, hi

    @classmethod
    def _split_float_param(cls, value, default: float, name: str):
        if value is None:
            return float(default), None
        if isinstance(value, (list, tuple)):
            lo, hi = cls._coerce_float_range(value, name)
            return float((lo + hi) / 2.0), (lo, hi)
        return float(value), None

    @staticmethod
    def _coerce_int_range(value, name: str):
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must be a list/tuple of exactly two values, got {value!r}")
        lo, hi = sorted((int(round(float(value[0]))), int(round(float(value[1])))))
        if lo < 1:
            raise ValueError(f"{name} must be >= 1, got {value!r}")
        return lo, hi

    @classmethod
    def _split_int_param(cls, value, default: int, name: str):
        if value is None:
            return int(default), None
        if isinstance(value, (list, tuple)):
            lo, hi = cls._coerce_int_range(value, name)
            return int(round((lo + hi) / 2.0)), (lo, hi)
        return int(round(float(value))), None

    def sample_mask_params(self, device=None) -> tuple[float, int]:
        """Return effective mask scale and box size for this masking call."""
        rand_device = device if device is not None else torch.device("cpu")
        mask_scale = self.mask_scale
        if self.mask_scale_range is not None:
            lo, hi = self.mask_scale_range
            if hi > lo:
                mask_scale = lo + (hi - lo) * float(torch.rand((), device=rand_device).item())
            else:
                mask_scale = lo

        mask_box_size = self.mask_box_size
        if self.mask_box_size_range is not None and not self.random_mask_box_per_target:
            lo, hi = self.mask_box_size_range
            if hi > lo:
                mask_box_size = int(torch.randint(lo, hi + 1, (), device=rand_device).item())
            else:
                mask_box_size = lo

        return float(mask_scale), int(mask_box_size)

    @staticmethod
    def _coerce_manual_mask_box_sizes(value) -> tuple[int, ...] | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            items = [v.strip() for v in stripped.split(",") if v.strip()]
        else:
            try:
                items = list(value)
            except TypeError:
                items = [value]
        if not items:
            return None
        sizes = tuple(int(round(float(v))) for v in items)
        if any(v <= 0 for v in sizes):
            raise ValueError(f"manual_mask_box_sizes must contain positive sizes, got {sizes}")
        return sizes

    def forward(
        self,
        x_clean,
        return_debug: bool = False,
        enable_grid_jitter: bool = True,
        enable_target_dithering: bool = True,
        lattice_shift_override=None,
        mask_inference: bool = True,
        context_data=None,
        cdd_orig: torch.Tensor | None = None,
    ):
        """
        x_clean: B x 1 x H x W

        context_data (optional): tuple of (x_context, target_locations,
            target_scales, target_valid [, debug]) pre-computed by
            prepare_context_batch / make_pyramid_grid_context.  When
            provided the masking step is skipped entirely — this must be
            called *outside* autocast in training loops.
        """
        if x_clean.dim() != 4:
            raise ValueError(f"Expected BxCxHxW, got {tuple(x_clean.shape)}")

        if x_clean.shape[1] != 1:
            raise ValueError(f"Expected grayscale input, got {x_clean.shape[1]} channels")

        if context_data is not None:
            x_context = context_data[0].to(device=x_clean.device)
            target_locations = context_data[1].to(device=x_clean.device)
            target_scales = context_data[2].to(device=x_clean.device)
            target_valid = context_data[3].to(device=x_clean.device)
            debug = context_data[4] if len(context_data) > 4 else {}
        else:
            invalid_pixel_mask = ~torch.isfinite(x_clean)
            if invalid_pixel_mask.any():
                x_clean = torch.nan_to_num(x_clean, nan=0.0, posinf=0.0, neginf=0.0)

            debug_encoder_types = CDD_DEBUG_ENCODER_TYPES | MASK_MAP_ENCODER_TYPES
            need_debug_tensors = bool(
                return_debug
                or self.encoder_type in debug_encoder_types
            )
            effective_mask_scale, effective_mask_box_size = self.sample_mask_params(device=x_clean.device)
            if need_debug_tensors:
                x_context, target_locations, target_scales, target_valid, debug = make_pyramid_grid_context(
                    x_clean=x_clean,
                    sigmas=self.sigmas,
                    mask_fraction=self.mask_fraction,
                    mask_scale=effective_mask_scale,
                    spacing_scale=self.spacing_scale,
                    global_shift=self.global_shift,
                    align_scales=self.align_scales,
                    mask_box_size=effective_mask_box_size,
                    mask_box_size_range=self.mask_box_size_range,
                    random_mask_box_per_target=self.random_mask_box_per_target,
                    manual_mask_box_sizes=self.manual_mask_box_sizes,
                    cdd_mode=self.cdd_mode,
                    cdd_constrained=self.cdd_constrained,
                    cdd_sm_mode=self.cdd_sm_mode,
                    cdd_append_last_residual=self.cdd_append_last_residual,
                    cdd_pre_log_transform=self.cdd_pre_log_transform,
                    inner_target_size=self.patch_size,
                    return_debug=True,
                    enable_grid_jitter=enable_grid_jitter,
                    enable_target_dithering=enable_target_dithering,
                    lattice_shift_override=lattice_shift_override,
                    target_invalid_region_skip=self.target_invalid_region_skip,
                    target_invalid_region_values=self.target_invalid_region_values,
                    invalid_pixel_mask=invalid_pixel_mask,
                    target_sampling_mode=self.target_sampling_mode,
                    priority_top_percent=self.priority_top_percent,
                    priority_n_target=self.priority_n_target,
                    priority_min_targets_per_map=self.priority_min_targets_per_map,
                    priority_dithering_pixels=self.priority_dithering_pixels,
                    priority_candidate_oversample=self.priority_candidate_oversample,
                    target_nonoverlap=self.target_nonoverlap,
                    target_allow_partial_overlap=self.target_allow_partial_overlap,
                    mask_box_hardcap=self.mask_box_hardcap,
                    cdd_orig_in=cdd_orig,
                    use_cdd=self.encoder_type in CDD_CUBE_ENCODER_TYPES,
                )
            else:
                x_context, target_locations, target_scales, target_valid = make_pyramid_grid_context(
                    x_clean=x_clean,
                    sigmas=self.sigmas,
                    mask_fraction=self.mask_fraction,
                    mask_scale=effective_mask_scale,
                    spacing_scale=self.spacing_scale,
                    global_shift=self.global_shift,
                    align_scales=self.align_scales,
                    mask_box_size=effective_mask_box_size,
                    mask_box_size_range=self.mask_box_size_range,
                    random_mask_box_per_target=self.random_mask_box_per_target,
                    manual_mask_box_sizes=self.manual_mask_box_sizes,
                    cdd_mode=self.cdd_mode,
                    cdd_constrained=self.cdd_constrained,
                    cdd_sm_mode=self.cdd_sm_mode,
                    cdd_append_last_residual=self.cdd_append_last_residual,
                    cdd_pre_log_transform=self.cdd_pre_log_transform,
                    inner_target_size=self.patch_size,
                    enable_grid_jitter=enable_grid_jitter,
                    enable_target_dithering=enable_target_dithering,
                    lattice_shift_override=lattice_shift_override,
                    target_invalid_region_skip=self.target_invalid_region_skip,
                    target_invalid_region_values=self.target_invalid_region_values,
                    invalid_pixel_mask=invalid_pixel_mask,
                    target_sampling_mode=self.target_sampling_mode,
                    priority_top_percent=self.priority_top_percent,
                    priority_n_target=self.priority_n_target,
                    priority_min_targets_per_map=self.priority_min_targets_per_map,
                    priority_dithering_pixels=self.priority_dithering_pixels,
                    priority_candidate_oversample=self.priority_candidate_oversample,
                    target_nonoverlap=self.target_nonoverlap,
                    target_allow_partial_overlap=self.target_allow_partial_overlap,
                    mask_box_hardcap=self.mask_box_hardcap,
                    cdd_orig_in=cdd_orig,
                )

        x_clean_enc = x_clean
        x_context_enc = x_context
        if self.post_log_transform:
            eps = max(1e-6, float(self.log_eps))
            # Shared floor keeps clean and masked CDD reconstructions on one scale.
            base = torch.clamp(x_clean, min=0.0)
            base_std = torch.std(base, dim=(-2, -1), keepdim=True)
            log_floor = torch.clamp(base_std * float(self.cdd_log_std_floor_mult), min=eps)
            x_clean_enc = torch.log(torch.clamp(x_clean, min=0.0) + log_floor)
            x_context_enc = torch.log(torch.clamp(x_context, min=0.0) + log_floor)

        # Optional multiscale CDD path: encode channel cubes directly.
        # Keep x_clean/x_context image outputs for backward-compatible diagnostics.
        enc_target = x_clean_enc
        enc_context = x_context_enc
        actual_context_in = None
        actual_target_in = None
        cdd_orig = None
        cdd_masked = None
        dip_per_ch = None
        cdd_orig_enc = None
        cdd_masked_enc = None
        needs_cdd_cube = self.encoder_type in CDD_CUBE_ENCODER_TYPES
        if needs_cdd_cube:
            cdd_orig = debug["cdd_channels_orig"].to(device=x_clean.device, dtype=x_clean.dtype)
            cdd_masked = debug["cdd_channels_masked"].to(device=x_clean.device, dtype=x_clean.dtype)
            dip_per_ch = debug["dip_field_per_channel"].to(device=x_clean.device, dtype=x_clean.dtype)
            # Global CDD-cube stabilization for pyramid encoders that consume
            # concatenated channel cubes directly (non-CDDOpNet paths).
            if self.post_log_transform:
                eps = max(1e-6, float(self.log_eps))
                base = torch.clamp(x_clean, min=0.0)
                base_std = torch.std(base, dim=(-2, -1), keepdim=True)
                log_floor = torch.clamp(base_std * float(self.cdd_log_std_floor_mult), min=eps)
                cdd_orig_enc = torch.log(torch.clamp(cdd_orig, min=0.0) + log_floor)
                cdd_masked_enc = torch.log(torch.clamp(cdd_masked, min=0.0) + log_floor)
            else:
                cdd_orig_enc = cdd_orig
                cdd_masked_enc = cdd_masked
            zero_token = torch.zeros_like(dip_per_ch)
            # target: original per-scale channels + zero token maps
            enc_target = torch.cat([cdd_orig_enc, zero_token], dim=1)
            # context: masked per-scale channels + mask token maps
            enc_context = torch.cat([cdd_masked_enc, dip_per_ch], dim=1)
        if not bool(mask_inference):
            # In mask-free inference, predictor branch should consume clean features.
            enc_context = enc_target
        symmetric_var = None  # trainable context-encoder rotation-view variance
        target_symmetric_var = None  # detached EMA diagnostic only
        if self.encoder_type == "cdd_scaleaware_convnext":
            if self.mode != "pyramid":
                raise ValueError("cdd_scaleaware_convnext requires mode='pyramid'.")
            mask_tokens = dip_per_ch
            cdd_orig_scaleaware = cdd_orig_enc
            cdd_masked_scaleaware = cdd_masked_enc
            if self.scaleaware_norm_per_scale:
                cdd_orig_scaleaware = norm_per_sample_channel(cdd_orig_scaleaware)
                cdd_masked_scaleaware = norm_per_sample_channel(cdd_masked_scaleaware)
            zero_mask_tokens = torch.zeros_like(mask_tokens)
            if bool(mask_inference):
                if self.use_symmetric_feature_loss:
                    context_map, ctx_var = symmetric_forward_2d(
                        self.context_encoder,
                        cdd_masked_scaleaware,
                        mask_tokens=mask_tokens,
                        return_var=True,
                    )
                    symmetric_var = ctx_var if symmetric_var is None else symmetric_var + ctx_var
                else:
                    context_map = self.context_encoder(cdd_masked_scaleaware, mask_tokens=mask_tokens)
            else:
                if self.use_symmetric_feature_loss:
                    context_map, ctx_var = symmetric_forward_2d(
                        self.context_encoder,
                        cdd_orig_scaleaware,
                        mask_tokens=zero_mask_tokens,
                        return_var=True,
                    )
                    symmetric_var = ctx_var if symmetric_var is None else symmetric_var + ctx_var
                else:
                    context_map = self.context_encoder(cdd_orig_scaleaware, mask_tokens=zero_mask_tokens)
            with torch.no_grad():
                if self.use_symmetric_feature_loss:
                    gt_map, gt_var = symmetric_forward_2d(
                        self.target_encoder,
                        cdd_orig_scaleaware,
                        mask_tokens=zero_mask_tokens,
                        return_var=True,
                    )
                    target_symmetric_var = gt_var if target_symmetric_var is None else target_symmetric_var + gt_var
                else:
                    gt_map = self.target_encoder(cdd_orig_scaleaware, mask_tokens=zero_mask_tokens)
        elif self.encoder_type in ("convnext_dense_pyramid", "escnn_c4_pyramid"):
            if self.mode != "pyramid":
                raise ValueError(f"{self.encoder_type} requires mode='pyramid'.")
            mask_tokens = dip_per_ch
            if bool(mask_inference):
                enc_context = torch.cat([cdd_masked_enc, mask_tokens], dim=1)
            else:
                enc_context = torch.cat([cdd_orig_enc, torch.zeros_like(mask_tokens)], dim=1)
            enc_target = torch.cat([cdd_orig_enc, torch.zeros_like(mask_tokens)], dim=1)
            with torch.no_grad():
                if self.use_symmetric_feature_loss:
                    gt_map, gt_var = symmetric_forward_2d(self.target_encoder, enc_target, return_var=True)
                    target_symmetric_var = gt_var if target_symmetric_var is None else target_symmetric_var + gt_var
                else:
                    gt_map = self.target_encoder(enc_target)
            if self.use_symmetric_feature_loss:
                context_map, ctx_var = symmetric_forward_2d(self.context_encoder, enc_context, return_var=True)
                symmetric_var = ctx_var if symmetric_var is None else symmetric_var + ctx_var
            else:
                context_map = self.context_encoder(enc_context)
        elif self.encoder_type == "convnext_dense_masktoken":
            if self.mode != "image":
                raise ValueError(f"{self.encoder_type} requires mode='image'.")
            if "mask_map" not in debug:
                raise RuntimeError(
                    f"{self.encoder_type} requires debug['mask_map']; "
                    "call make_pyramid_grid_context with return_debug=True."
                )
            mask_token = debug["mask_map"].to(device=x_clean_enc.device, dtype=x_clean_enc.dtype)
            if mask_token.ndim == 3:
                mask_token = mask_token.unsqueeze(1)
            if mask_token.ndim != 4:
                raise RuntimeError(f"Expected mask_map Bx1xHxW or BxHxW, got {tuple(mask_token.shape)}")
            if mask_token.shape[1] != 1:
                mask_token = mask_token[:, :1]
            mask_token = mask_token.clamp(0.0, 1.0)
            zero_token = torch.zeros_like(mask_token)

            # Fixed image ConvNeXt contract:
            # context  = [zero-filled masked image, binary mask map]
            # target   = [clean image, zero mask map]
            clean_image = x_clean_enc
            masked_image = clean_image * (1.0 - mask_token)
            if bool(mask_inference):
                context_in = torch.cat([masked_image, mask_token], dim=1)
            else:
                context_in = torch.cat([clean_image, zero_token], dim=1)
            target_in = torch.cat([clean_image, zero_token], dim=1)

            actual_context_in = context_in
            actual_target_in = target_in

            with torch.no_grad():
                if self.use_symmetric_feature_loss:
                    gt_map, gt_var = symmetric_forward_2d(self.target_encoder, target_in, return_var=True)
                    target_symmetric_var = gt_var if target_symmetric_var is None else target_symmetric_var + gt_var
                else:
                    gt_map = self.target_encoder(target_in)
            if self.use_symmetric_feature_loss:
                context_map, ctx_var = symmetric_forward_2d(self.context_encoder, context_in, return_var=True)
                symmetric_var = ctx_var if symmetric_var is None else symmetric_var + ctx_var
            else:
                context_map = self.context_encoder(context_in)
        else:
            with torch.no_grad():
                gt_map = self.target_encoder(enc_target)
            context_map = self.context_encoder(enc_context)
        context_base = context_map
        gt_base = gt_map
        context_proj = self.projector(context_base)
        pred_map = self.predictor(context_proj)
        with torch.no_grad():
            gt_map = self.target_projector(gt_base)

        pred_patches = extract_location_patches(pred_map, target_locations, patch_size=self.patch_size)
        gt_patches = extract_location_patches(gt_map, target_locations, patch_size=self.patch_size)
        context_patches = extract_location_patches(context_proj, target_locations, patch_size=self.patch_size)

        out = {
            "pred_patches": pred_patches,
            "gt_patches": gt_patches,
            "context_patches": context_patches,
            # Raw pre-encoder tensors (for diagnostics/visualization).
            "x_clean_raw": x_clean,
            "x_context_raw": x_context,
            # Actual network inputs after shared post-mask transform.
            "x_clean": x_clean_enc,
            "x_context": x_context_enc,
            "target_locations": target_locations,
            "target_scales": target_scales,
            "target_valid": target_valid,
            "context_map": context_base,
            "pred_map": pred_map,
            "gt_map": gt_map,
        }
        if symmetric_var is not None:
            out["symmetric_var"] = symmetric_var
        if target_symmetric_var is not None:
            out["target_symmetric_var"] = target_symmetric_var
        if actual_context_in is not None:
            out["network_context_in"] = actual_context_in
            out["network_target_in"] = actual_target_in
        for key in ("mask_scale_factor", "mask_footprint_px", "cdd_box_sizes", "target_box_sizes", "random_mask_box_per_target"):
            if key in debug:
                out[key] = debug[key].to(device=x_clean.device, dtype=x_clean.dtype)
        if return_debug:
            # Exact applied hard mask footprint from make_pyramid_grid_context.
            out["target_mask_map"] = debug["mask_map"].unsqueeze(1).to(device=x_clean.device, dtype=x_clean.dtype)
            for k in (
                "priority_good_candidates",
                "priority_nonzero_mean",
                "priority_prescreen_candidates",
                "priority_auto_base_targets",
                "priority_effective_targets",
            ):
                if k in debug:
                    out[k] = debug[k].to(device=x_clean.device, dtype=x_clean.dtype)
        if needs_cdd_cube:
            out["cdd_channels_orig"] = debug["cdd_channels_orig"].to(device=x_clean.device, dtype=x_clean.dtype)
            out["cdd_channels_masked"] = debug["cdd_channels_masked"].to(device=x_clean.device, dtype=x_clean.dtype)
            out["dip_field_per_channel"] = debug["dip_field_per_channel"].to(device=x_clean.device, dtype=x_clean.dtype)
        return out

    def compute_symmetric_loss(self, outputs):
        """Context-encoder view variance, averaged over spatial and channel dims."""
        var = outputs.get("symmetric_var")
        if var is None:
            return torch.tensor(0.0, device=outputs["pred_patches"].device)
        return var.mean()

    def compute_loss(self, outputs):
        # Keep reductions in fp32: patch sums can overflow under AMP.
        pred = outputs["pred_patches"].float()
        gt = outputs["gt_patches"].detach().float()

        valid = outputs["target_valid"]  # B x K (bool)

        if self.normalize_loss_l2:
            # Normalize the full patch vector so spatial contrast is preserved.
            pred = l2_normalize_patches(pred)
            gt = l2_normalize_patches(gt)
            outputs["pred_patches"] = pred
            outputs["gt_patches"] = gt
        loss_map = F.mse_loss(pred, gt, reduction="none")  # B x K x C x P x P
        w = valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(loss_map.dtype)
        if not bool(valid.any().item()):
            # No valid targets in this batch: return graph-connected zero loss.
            return loss_map.sum() * 0.0
        denom = torch.clamp(w.sum() * loss_map.shape[2] * loss_map.shape[3] * loss_map.shape[4], min=1.0)
        return (loss_map * w).sum() / denom

    @torch.no_grad()
    def update_target_encoder(self):
        # Use base_encoder directly when a D4 / other wrapper is present to avoid
        # coupling the EMA to wrapper parameters that may appear in the future.
        ctx_enc = getattr(self.context_encoder, "base_encoder", self.context_encoder)
        tgt_enc = getattr(self.target_encoder, "base_encoder", self.target_encoder)
        for p_context, p_target in zip(ctx_enc.parameters(), tgt_enc.parameters()):
            p_target.mul_(self.ema_momentum).add_(p_context.detach(), alpha=1.0 - self.ema_momentum)
        if self.projector_conv:
            for p_proj, p_target_proj in zip(self.projector.parameters(), self.target_projector.parameters()):
                p_target_proj.mul_(self.ema_momentum).add_(p_proj.detach(), alpha=1.0 - self.ema_momentum)


================================================================================
FILE: src/models/masking3d.py
================================================================================

from __future__ import annotations

import torch


def sample_target_locations_3d(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    num_targets: int,
    patch_size: int,
    device,
):
    half = patch_size // 2
    lo_z = half
    hi_z = depth - (patch_size - half)
    lo_y = half
    hi_y = height - (patch_size - half)
    lo_x = half
    hi_x = width - (patch_size - half)

    if hi_z < lo_z or hi_y < lo_y or hi_x < lo_x:
        raise ValueError("Patch too large for volume")

    z = torch.randint(lo_z, hi_z + 1, (batch_size, num_targets), device=device)
    y = torch.randint(lo_y, hi_y + 1, (batch_size, num_targets), device=device)
    x = torch.randint(lo_x, hi_x + 1, (batch_size, num_targets), device=device)

    loc = torch.stack([z, y, x], dim=-1)
    valid = torch.ones((batch_size, num_targets), dtype=torch.bool, device=device)
    return loc, valid


def extract_location_cubes(z: torch.Tensor, locations: torch.Tensor, patch_size: int):
    if z.ndim != 5:
        raise ValueError(f"Expected z B,C,D,H,W, got {tuple(z.shape)}")

    b, c, d, h, w = z.shape
    _, k, ndim = locations.shape
    if ndim != 3:
        raise ValueError(f"Expected locations B,K,3, got {tuple(locations.shape)}")

    p = int(patch_size)
    if p <= 0:
        raise ValueError(f"patch_size must be positive, got {p}")
    if p > d or p > h or p > w:
        raise ValueError(f"patch_size={p} exceeds feature map size {(d, h, w)}")

    half = p // 2
    z0 = locations[:, :, 0] - half
    y0 = locations[:, :, 1] - half
    x0 = locations[:, :, 2] - half

    valid = (
        (z0 >= 0)
        & (y0 >= 0)
        & (x0 >= 0)
        & (z0 + p <= d)
        & (y0 + p <= h)
        & (x0 + p <= w)
    )

    dz = torch.arange(p, device=z.device)
    dy = torch.arange(p, device=z.device)
    dx = torch.arange(p, device=z.device)

    zz = z0.view(b, k, 1, 1, 1) + dz.view(1, 1, p, 1, 1)
    yy = y0.view(b, k, 1, 1, 1) + dy.view(1, 1, 1, p, 1)
    xx = x0.view(b, k, 1, 1, 1) + dx.view(1, 1, 1, 1, p)

    zz = zz.clamp(0, d - 1)
    yy = yy.clamp(0, h - 1)
    xx = xx.clamp(0, w - 1)

    b_idx = torch.arange(b, device=z.device).view(b, 1, 1, 1, 1, 1)
    c_idx = torch.arange(c, device=z.device).view(1, 1, c, 1, 1, 1)

    zz = zz.unsqueeze(2)
    yy = yy.unsqueeze(2)
    xx = xx.unsqueeze(2)

    cubes = z[b_idx, c_idx, zz, yy, xx]
    valid_mask = valid.view(b, k, 1, 1, 1, 1)
    cubes = torch.where(valid_mask, cubes, torch.zeros_like(cubes))
    return cubes


================================================================================
FILE: src/models/build_jepa3d.py
================================================================================

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders3d import ScaleAwareConvNeXt3DEncoder, ScaleFiLMConvNeXt3DEncoder
from .masking import _fractional_spatial_target_budget
from .masking3d import extract_location_cubes, sample_target_locations_3d
from .predictor3d import FullResPredictor3D
from .symmetry import symmetric_forward_3d
from src.losses import l2_normalize_patches


def compute_3d_encoder_receptive_field_depth(encoder_depth: int = 3, encoder_kernel_size: int = 5) -> int:
    """Depth receptive field for the same-padded 3D encoder path.

    The 3D encoders use two 3x3x3 stem convolutions, then `encoder_depth`
    ConvNeXt depthwise convolutions with `encoder_kernel_size`.
    """
    return 1 + 2 * (3 - 1) + max(0, int(encoder_depth)) * (int(encoder_kernel_size) - 1)


class PyramidGridJEPA3D(nn.Module):
    def __init__(
        self,
        latent_channels=16,
        scale_channels=8,
        num_scales: int = 1,
        encoder_type: str = "cdd_scaleaware_convnext3d",
        patch_size=2,
        num_targets=32,
        encoder_depth=3,
        encoder_kernel_size=5,
        encoder_stride=1,
        ema_momentum=0.996,
        normalize_loss_l2=False,
        post_log_transform: bool = True,
        log_eps: float = 1e-6,
        cdd_log_std_floor_mult: float = 0.05,
        fusion="gate",
        mask_box_size: int = 8,
        num_mask_boxes: int = 8,
        slab_depth: int = 3,
        use_symmetric_feature_loss: bool = False,
        use_film: bool = True,
        use_per_scale_adapters: bool = False,
        priority_candidate_oversample: float = 3.0,
        encoder_receptive_field_depth: int | None = None,
        use_grn: bool = True,
        stem_norm: bool = True,
        norm_per_scale: bool = True,
        adapter_norm: bool = True,
        final_norm: bool = True,
        full_volume_training: bool = False,
    ):
        super().__init__()
        self.num_scales = int(num_scales)
        self.encoder_type = str(encoder_type).lower()
        self.patch_size = int(patch_size)
        self.num_targets = int(num_targets)
        self.ema_momentum = float(ema_momentum)
        self.normalize_loss_l2 = bool(normalize_loss_l2)
        self.post_log_transform = bool(post_log_transform)
        self.log_eps = float(log_eps)
        self.cdd_log_std_floor_mult = float(cdd_log_std_floor_mult)
        self.mask_box_size = int(mask_box_size)
        self.num_mask_boxes = int(num_mask_boxes)
        self.mode = "3d_full_volume" if bool(full_volume_training) else "3d_slab"
        self.full_volume_training = bool(full_volume_training)
        self.slab_depth = max(self.patch_size, int(slab_depth))
        self.encoder_receptive_field_depth = int(
            encoder_receptive_field_depth
            if encoder_receptive_field_depth is not None
            else compute_3d_encoder_receptive_field_depth(encoder_depth, encoder_kernel_size)
        )
        self.required_input_depth = int(
            self.slab_depth
            if self.full_volume_training
            else self.encoder_receptive_field_depth + self.slab_depth - 1
        )
        self.use_symmetric_feature_loss = bool(use_symmetric_feature_loss)
        self.use_film = bool(use_film)
        self.use_per_scale_adapters = bool(use_per_scale_adapters)
        self.priority_candidate_oversample = float(priority_candidate_oversample)

        if self.use_film or self.use_per_scale_adapters:
            self.context_encoder = ScaleFiLMConvNeXt3DEncoder(
                num_scales=self.num_scales,
                out_channels=int(latent_channels),
                scale_channels=int(scale_channels),
                depth=int(encoder_depth),
                kernel_size=int(encoder_kernel_size),
                stride=int(encoder_stride),
                fusion=str(fusion),
                use_film=self.use_film,
                use_per_scale_adapters=self.use_per_scale_adapters,
                use_grn=bool(use_grn),
                stem_norm=bool(stem_norm),
                norm_per_scale=bool(norm_per_scale),
                adapter_norm=bool(adapter_norm),
                final_norm=bool(final_norm),
            )
        else:
            self.context_encoder = ScaleAwareConvNeXt3DEncoder(
                num_scales=self.num_scales,
                out_channels=int(latent_channels),
                scale_channels=int(scale_channels),
                depth=int(encoder_depth),
                kernel_size=int(encoder_kernel_size),
                stride=int(encoder_stride),
                fusion=str(fusion),
                use_grn=bool(use_grn),
                stem_norm=bool(stem_norm),
                norm_per_scale=bool(norm_per_scale),
                adapter_norm=bool(adapter_norm),
                final_norm=bool(final_norm),
            )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.projector = nn.Identity()
        self.predictor3d = FullResPredictor3D(
            channels=int(latent_channels),
            hidden=max(2 * int(latent_channels), 32),
        )
    def make_fields(self, x):
        # The input channel is the single direct 3D field axis consumed by the encoder.
        return x

    def _make_random_box_mask3d(
        self,
        batch_size: int,
        depth: int,
        height: int,
        width: int,
        device,
        focus_slab_start_idx: torch.Tensor,
    ):
        box = max(1, int(self.mask_box_size))
        n_box = max(1, int(self.num_mask_boxes))
        mask = torch.zeros((batch_size, 1, depth, height, width), device=device)
        z_lim = max(1, depth - box + 1)
        y_lim = max(1, height - box + 1)
        x_lim = max(1, width - box + 1)
        z0 = torch.empty((batch_size, n_box), device=device, dtype=torch.long)

        slab_start_cpu = focus_slab_start_idx.detach().to("cpu").numpy()
        slab_depth = max(1, min(int(self.slab_depth), depth))
        for b in range(batch_size):
            slab_start = int(slab_start_cpu[b])
            slab_end = slab_start + slab_depth
            lo = max(0, slab_start - box + 1)
            hi = min(z_lim, slab_end)
            if hi <= lo:
                lo, hi = 0, z_lim
            z0[b] = torch.randint(lo, hi, (n_box,), device=device)

        y0 = torch.randint(0, y_lim, (batch_size, n_box), device=device)
        x0 = torch.randint(0, x_lim, (batch_size, n_box), device=device)
        # Build mask on CPU to avoid per-slice CUDA kernel launches
        z0_cpu = z0.detach().to("cpu").numpy()
        y0_cpu = y0.detach().to("cpu").numpy()
        x0_cpu = x0.detach().to("cpu").numpy()
        mask_cpu = mask.detach().to("cpu").numpy()
        for b in range(batch_size):
            for j in range(n_box):
                zz = int(z0_cpu[b, j])
                yy = int(y0_cpu[b, j])
                xx = int(x0_cpu[b, j])
                mask_cpu[b, 0, zz : zz + box, yy : yy + box, xx : xx + box] = 1.0
        mask.copy_(torch.from_numpy(mask_cpu).to(device=mask.device))
        return mask

    def _center_slab_start_index(self, batch_size: int, depth: int, device) -> tuple[torch.Tensor, int]:
        if self.full_volume_training:
            starts = torch.zeros((batch_size,), dtype=torch.long, device=device)
            return starts, int(depth)
        slab_depth = max(1, min(int(self.slab_depth), int(depth)))
        start = max(0, (int(depth) - slab_depth) // 2)
        starts = torch.full((batch_size,), start, dtype=torch.long, device=device)
        return starts, slab_depth

    @staticmethod
    def _gather_slabs(z: torch.Tensor, slab_starts: torch.Tensor, slab_depth: int) -> torch.Tensor:
        b, c, d, h, w = z.shape
        offsets = torch.arange(slab_depth, device=z.device).view(1, slab_depth)
        slab_idx = (slab_starts.view(b, 1) + offsets).clamp(0, d - 1)
        gather_idx = slab_idx.view(b, 1, slab_depth, 1, 1).expand(b, c, slab_depth, h, w)
        return z.gather(dim=2, index=gather_idx)

    def forward(self, x_clean, **kwargs):
        if x_clean.dim() != 5:
            raise ValueError(f"Expected BxSxDxHxW, got {tuple(x_clean.shape)}")

        b, s, _, _, _ = x_clean.shape
        if s == 1 and s != self.num_scales:
            # On-the-fly 3D CDD decomposition (DDP fallback — no precomputed cache).
            import constrained_diffusion as cdd
            import numpy as np
            x_np = x_clean[:, 0].cpu().numpy()  # (B, D, H, W)
            decomposed = []
            for bi in range(b):
                arr = x_np[bi].astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-20)
                channels_arr, _residual = cdd.constrained_diffusion_decomposition(
                    arr,
                    num_channels=self.num_scales,
                    max_scale=16.0,
                    mode="log",
                    constrained=True,
                    sm_mode="reflect",
                    return_scales=False,
                    verbose=False,
                    use_gpu=False,
                )
                ch = np.clip(np.stack(channels_arr, axis=0), 0.0, None).astype(np.float32)
                decomposed.append(torch.from_numpy(ch).to(x_clean.device))
            x_clean = torch.stack(decomposed, dim=0)  # (B, S, D, H, W)
            s = self.num_scales
        elif s != self.num_scales:
            raise ValueError(f"Expected Bx{self.num_scales}xDxHxW, got {tuple(x_clean.shape)}")
        fields = self.make_fields(x_clean)
        _, _, d, h, w = fields.shape
        if d < int(self.required_input_depth):
            raise ValueError(
                "3d_slab input depth is too small for the configured encoder/target geometry: "
                f"got {d}, required at least {self.required_input_depth} "
                f"(encoder_rf={self.encoder_receptive_field_depth}, target_slab_depth={self.slab_depth})"
            )

        slab_starts, slab_depth = self._center_slab_start_index(batch_size=b, depth=d, device=x_clean.device)

        box_mask = self._make_random_box_mask3d(
            b,
            d,
            h,
            w,
            x_clean.device,
            focus_slab_start_idx=slab_starts,
        )
        fields_context = fields * (1.0 - box_mask)
        mask_tokens = box_mask.expand(-1, fields.shape[1], -1, -1, -1)
        if self.post_log_transform:
            eps = max(1e-6, float(self.log_eps))
            base = torch.clamp(fields, min=0.0)
            base_std = torch.std(base, dim=(-3, -2, -1), keepdim=True)
            log_floor = torch.clamp(base_std * float(self.cdd_log_std_floor_mult), min=eps)
            fields = torch.log(base + log_floor)
            base_ctx = torch.clamp(fields_context, min=0.0)
            fields_context = torch.log(base_ctx + log_floor)

        if self.use_symmetric_feature_loss:
            context_map_3d, symmetric_var = symmetric_forward_3d(
                self.context_encoder,
                fields_context,
                mask_tokens=mask_tokens,
                return_var=True,
            )
        else:
            context_map_3d = self.context_encoder(fields_context, mask_tokens=mask_tokens)
            symmetric_var = None
        with torch.no_grad():
            zero_mask_tokens = torch.zeros_like(fields)
            if self.use_symmetric_feature_loss:
                gt_map_3d, target_symmetric_var = symmetric_forward_3d(
                    self.target_encoder,
                    fields,
                    mask_tokens=zero_mask_tokens,
                    return_var=True,
                )
            else:
                gt_map_3d = self.target_encoder(fields, mask_tokens=zero_mask_tokens)
                target_symmetric_var = None

        context_map = self._gather_slabs(context_map_3d, slab_starts, slab_depth)
        gt_map = self._gather_slabs(gt_map_3d, slab_starts, slab_depth)
        pred_map = self.predictor3d(context_map)
        _, _, dz, hy, wx = pred_map.shape
        target_budget = _fractional_spatial_target_budget(
            height=hy,
            width=wx,
            box_size=max(1, int(self.mask_box_size)),
            oversample=self.priority_candidate_oversample,
            device=x_clean.device,
            minimum=1,
        )
        num_targets = max(1, int(self.num_targets))
        if target_budget is not None:
            num_targets = max(1, min(num_targets, int(target_budget)))
        target_locations, target_valid = sample_target_locations_3d(
            batch_size=b,
            depth=dz,
            height=hy,
            width=wx,
            num_targets=num_targets,
            patch_size=self.patch_size,
            device=x_clean.device,
        )
        pred_patches = extract_location_cubes(pred_map, target_locations, self.patch_size)
        gt_patches = extract_location_cubes(gt_map, target_locations, self.patch_size)
        context_patches = extract_location_cubes(context_map, target_locations, self.patch_size)

        out = {
            "pred_patches": pred_patches,
            "gt_patches": gt_patches,
            "context_patches": context_patches,
            "target_locations": target_locations,
            "target_valid": target_valid,
            "target_scales": torch.ones((b, num_targets), device=x_clean.device, dtype=x_clean.dtype),
            "context_map": context_map,
            "context_map_3d": context_map_3d,
            "pred_map": pred_map,
            "gt_map": gt_map,
            "gt_map_3d": gt_map_3d,
            "x_clean": self._gather_slabs(x_clean, slab_starts, slab_depth),
            "x_clean_full": x_clean,
            "x_context": self._gather_slabs(fields_context, slab_starts, slab_depth),
            "x_context_full": fields_context,
            "mask_cube": box_mask,
            "selected_slab_start_index": slab_starts,
            "selected_slab_depth": torch.full((b,), int(slab_depth), device=x_clean.device, dtype=torch.long),
            "encoder_receptive_field_depth": torch.full((b,), int(self.encoder_receptive_field_depth), device=x_clean.device, dtype=torch.long),
            "required_input_depth": torch.full((b,), int(self.required_input_depth), device=x_clean.device, dtype=torch.long),
            "mask_footprint_px": torch.tensor(float(self.mask_box_size), device=x_clean.device, dtype=x_clean.dtype),
            "mask_scale_factor": torch.tensor(1.0, device=x_clean.device, dtype=x_clean.dtype),
        }
        if symmetric_var is not None:
            out["symmetric_var"] = symmetric_var
        if target_symmetric_var is not None:
            out["target_symmetric_var"] = target_symmetric_var
        return out

    def compute_symmetric_loss(self, outputs):
        """Context-encoder view variance, averaged over spatial and channel dims."""
        var = outputs.get("symmetric_var")
        if var is None:
            return torch.tensor(0.0, device=outputs["pred_patches"].device)
        return var.mean()

    def compute_loss(self, outputs):
        # Keep reductions in fp32: cube sums can overflow under AMP.
        pred = outputs["pred_patches"].float()
        gt = outputs["gt_patches"].detach().float()
        valid = outputs["target_valid"]

        if self.normalize_loss_l2:
            # Normalize the full cube vector so spatial contrast is preserved.
            pred = l2_normalize_patches(pred)
            gt = l2_normalize_patches(gt)
            outputs["pred_patches"] = pred
            outputs["gt_patches"] = gt

        loss_map = F.mse_loss(pred, gt, reduction="none")
        view_shape = [valid.shape[0], valid.shape[1]] + [1] * (loss_map.dim() - 2)
        w = valid.view(*view_shape).to(loss_map.dtype)

        if not bool(valid.any().item()):
            return loss_map.sum() * 0.0

        denom = torch.clamp(w.sum() * math.prod(loss_map.shape[2:]), min=1.0)
        return (loss_map * w).sum() / denom

    @torch.no_grad()
    def update_target_encoder(self):
        for p_context, p_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_target.mul_(self.ema_momentum).add_(p_context.detach(), alpha=1.0 - self.ema_momentum)


c: No such file or directory
- is a terminal (use -f to open it)
