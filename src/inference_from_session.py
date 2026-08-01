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
from src.utils.cdd_import import import_constrained_diffusion, safe_constrained_diffusion_decomposition
from src.utils.support import invalid_support_border_from_config
from src.utils.viz import export_inference_dashboard_artifacts


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


@dataclass
class TileLayout2D:
    original_shape: tuple[int, int]
    crop_size: int
    origins: tuple[tuple[int, int], ...]
    valid_shapes: tuple[tuple[int, int], ...]
    visit_map: np.ndarray | None = None  # H×W count of tile coverage per pixel
    valid_mask: np.ndarray | None = None  # H×W finite input pixels; invalid pixels stay NaN after stitching
    min_valid_fraction: float = 0.5
    edge_halo_px: int = 0  # Hard-rejected support radius at internal tile edges.


def _tile_starts(length: int, crop_size: int, stride: int) -> list[int]:
    if length <= crop_size:
        return [0]
    starts = list(range(0, max(1, length - crop_size + 1), stride))
    last = length - crop_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _valid_pixel_mask(arr: np.ndarray) -> np.ndarray:
    """Default inference validity: finite pixels are valid.

    Some science maps use zero as a physical value, while NaN marks no-data.
    Treating zero as invalid creates square holes and tile-selection bias.
    """
    a = np.asarray(arr)
    return np.isfinite(a)


def _tile_crops_2d_with_layout(
    arr2d: np.ndarray,
    crop_size: int,
    crop_mode: str = "center",
    crop_min_valid_fraction: float = 0.5,
    valid_mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], TileLayout2D | None]:
    h, w = arr2d.shape
    cs = int(crop_size)
    valid_mask_arr = _valid_pixel_mask(arr2d) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid_mask_arr.shape != (h, w):
        raise ValueError(f"valid_mask shape {valid_mask_arr.shape} must match input shape {(h, w)}")
    if h <= cs and w <= cs:
        valid_fraction = float(valid_mask_arr.sum()) / float(h * w)
        if valid_fraction <= float(crop_min_valid_fraction):
            raise ValueError(
                f"No valid inference tile: only {valid_fraction:.3f} valid pixels in padded "
                f"{cs}x{cs} cutout, below crop_min_valid_fraction={crop_min_valid_fraction:.3f}."
            )
        if h < cs or w < cs:
            padded = np.zeros((cs, cs), dtype=np.float32)
            padded[:h, :w] = np.asarray(arr2d, dtype=np.float32)
            visit_map = np.ones((h, w), dtype=np.int32)
            return [padded], TileLayout2D((h, w), cs, ((0, 0),), ((h, w),), visit_map, valid_mask_arr.copy(), float(crop_min_valid_fraction))
        if bool((~valid_mask_arr).any()):
            visit_map = np.ones((h, w), dtype=np.int32)
            return [np.asarray(arr2d, dtype=np.float32).copy()], TileLayout2D(
                (h, w), cs, ((0, 0),), ((h, w),), visit_map, valid_mask_arr.copy(), float(crop_min_valid_fraction)
            )
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
    y_starts = _tile_starts(h, cs, stride)
    x_starts = _tile_starts(w, cs, stride)
    min_valid = float(crop_min_valid_fraction)
    for y0 in y_starts:
        y1 = min(y0 + cs, h)
        for x0 in x_starts:
            x1 = min(x0 + cs, w)
            th = y1 - y0
            tw = x1 - x0
            region = np.asarray(arr2d[y0:y1, x0:x1], dtype=np.float32)
            valid_fraction = float(valid_mask_arr[y0:y1, x0:x1].sum()) / float(max(1, th * tw))
            if valid_fraction <= min_valid:
                continue
            tile = np.zeros((cs, cs), dtype=np.float32)
            tile[:th, :tw] = region
            tiles.append(tile)
            origins.append((y0, x0))
            valid_shapes.append((th, tw))
    if not tiles:
        raise ValueError(
            f"No inference tiles passed crop_min_valid_fraction={min_valid:.3f} "
            f"for input shape {(h, w)} and crop_size={cs}."
        )
    visit_map = np.zeros((h, w), dtype=np.int32)
    for (y0, x0), (th, tw) in zip(origins, valid_shapes):
        visit_map[y0:y0+th, x0:x0+tw] += 1
    return tiles, TileLayout2D(
        (h, w),
        cs,
        tuple(origins),
        tuple(valid_shapes),
        visit_map=visit_map,
        valid_mask=valid_mask_arr.copy(),
        min_valid_fraction=min_valid,
    )


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
    valid_mask = layout.valid_mask
    edge_halo_y = max(0, int(round(float(layout.edge_halo_px) * scale_y)))
    edge_halo_x = max(0, int(round(float(layout.edge_halo_px) * scale_x)))
    blend_y = max(1, int(round(float(layout.crop_size) * scale_y * 0.25)))
    blend_x = max(1, int(round(float(layout.crop_size) * scale_x * 0.25)))

    def _edge_blend(length: int, blend: int, has_before: bool, has_after: bool) -> np.ndarray:
        weights = np.ones(int(length), dtype=np.float32)
        n = int(min(max(1, blend), max(1, length)))
        floor = 0.0
        if has_before and n > 1:
            weights[:n] *= np.linspace(floor, 1.0, n, dtype=np.float32)
        if has_after and n > 1:
            weights[-n:] *= np.linspace(1.0, floor, n, dtype=np.float32)
        return weights

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
        wy = _edge_blend(vh, blend_y, y0 > 0, (y0 + th) < layout.original_shape[0])
        wx = _edge_blend(vw, blend_x, x0 > 0, (x0 + tw) < layout.original_shape[1])
        blend_weight = wy[:, None] * wx[None, :]
        # Tile padding contaminates predictions inside the encoder/CDD support
        # radius. Never blend that halo into the mosaic: latent derivatives
        # amplify even a small contribution into visible seam bands.
        if y0 > 0 and edge_halo_y > 0:
            blend_weight[: min(vh, edge_halo_y), :] = 0.0
        if (y0 + th) < layout.original_shape[0] and edge_halo_y > 0:
            blend_weight[max(0, vh - edge_halo_y) :, :] = 0.0
        if x0 > 0 and edge_halo_x > 0:
            blend_weight[:, : min(vw, edge_halo_x)] = 0.0
        if (x0 + tw) < layout.original_shape[1] and edge_halo_x > 0:
            blend_weight[:, max(0, vw - edge_halo_x) :] = 0.0
        weight = torch.from_numpy(blend_weight)[None, None].to(dtype=tensor.dtype)
        while weight.dim() < tensor.dim():
            weight = weight.unsqueeze(1)
        if valid_mask is not None:
            mask_crop = np.asarray(valid_mask[y0 : y0 + th, x0 : x0 + tw], dtype=bool)
            if mask_crop.shape != (vh, vw):
                yy = np.clip(np.floor(np.arange(vh, dtype=np.float32) / max(scale_y, 1e-12)).astype(np.int64), 0, th - 1)
                xx = np.clip(np.floor(np.arange(vw, dtype=np.float32) / max(scale_x, 1e-12)).astype(np.int64), 0, tw - 1)
                mask_crop = mask_crop[np.ix_(yy, xx)]
            mask_weight = torch.from_numpy(mask_crop.astype(np.float32))[None, None].to(dtype=tensor.dtype)
            while mask_weight.dim() < tensor.dim():
                mask_weight = mask_weight.unsqueeze(1)
            weight = weight * mask_weight
        if torch.is_floating_point(tile):
            finite_weight = torch.isfinite(tile).all(dim=tuple(range(1, tile.dim() - 2)), keepdim=True).to(dtype=tensor.dtype)
            weight = weight * finite_weight
            tile = torch.nan_to_num(tile, nan=0.0, posinf=0.0, neginf=0.0)
        out[..., oy0:oy1, ox0:ox1] += tile * weight
        counts[..., oy0:oy1, ox0:ox1] += weight
    stitched = out / counts.clamp_min(1)
    if torch.is_floating_point(stitched):
        stitched = torch.where(counts > 0, stitched, torch.full_like(stitched, float("nan")))
    return stitched


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
    for key in (
        "pred_map",
        "mask_pred_map",
        "masked_pred_map",
        "masked_target_pred_map",
        "gt_map",
        "context_map",
        "x_clean_raw",
        "x_context_raw",
        "target_energy_map",
        "x_clean",
        "x_context",
    ):
        stitched[key] = _stitch_tile_tensor(stitched.get(key), layout)
    if stitched.get("x_clean_raw") is not None:
        stitched["x_clean"] = stitched["x_clean_raw"]
    if stitched.get("x_context_raw") is not None:
        stitched["x_context"] = stitched["x_context_raw"]
    stitched["tile_layout"] = {
        "original_shape": list(layout.original_shape),
        "crop_size": int(layout.crop_size),
        "num_tiles": len(layout.origins),
        "edge_halo_px": int(layout.edge_halo_px),
    }
    return stitched


def _tile_channel_field_with_layout(field: torch.Tensor, layout: TileLayout2D) -> torch.Tensor:
    """Tile a full-frame C×H×W tensor using an existing 2D tile layout."""
    field_cpu = field.detach().cpu()
    if field_cpu.dim() == 4:
        if int(field_cpu.shape[0]) != 1:
            raise ValueError(f"Expected one full-frame field, got shape={tuple(field_cpu.shape)}")
        field_cpu = field_cpu[0]
    if field_cpu.dim() != 3:
        raise ValueError(f"Expected CxHxW field, got shape={tuple(field_cpu.shape)}")
    tiles = []
    cs = int(layout.crop_size)
    for (y0, x0), (th, tw) in zip(layout.origins, layout.valid_shapes):
        tile = torch.zeros((int(field_cpu.shape[0]), cs, cs), dtype=field_cpu.dtype)
        tile[:, :th, :tw] = field_cpu[:, y0 : y0 + th, x0 : x0 + tw]
        tiles.append(tile)
    return torch.stack(tiles, dim=0)


def _erode_valid_mask(
    valid_mask: np.ndarray,
    border_px: int,
    *,
    reject_outer_border: bool = True,
) -> np.ndarray:
    """Reject pixels near no-data and, by default, the outer image border."""
    valid = np.asarray(valid_mask, dtype=bool)
    b = int(max(0, border_px))
    if b <= 0:
        return valid.copy()
    invalid = torch.from_numpy((~valid).astype(np.float32))[None, None]
    k = 2 * b + 1
    dilated_invalid = F.max_pool2d(invalid, kernel_size=k, stride=1, padding=b)[0, 0].numpy() > 0.0
    accepted = valid & (~dilated_invalid)
    if reject_outer_border:
        accepted[:b, :] = False
        accepted[-b:, :] = False
        accepted[:, :b] = False
        accepted[:, -b:] = False
    return accepted


def _assert_valid_output_coverage(outputs: dict, valid_mask: np.ndarray | None) -> None:
    """Fail instead of silently leaving holes in the accepted output area."""
    if valid_mask is None or outputs.get("pred_map") is None:
        return
    pred = outputs["pred_map"]
    tensor = pred if torch.is_tensor(pred) else torch.as_tensor(pred)
    if tensor.dim() < 4 or tuple(int(v) for v in tensor.shape[-2:]) != tuple(valid_mask.shape):
        return
    finite = torch.isfinite(tensor).all(dim=tuple(range(tensor.dim() - 2)))
    expected = torch.from_numpy(np.asarray(valid_mask, dtype=bool)).to(finite.device)
    missing = expected & (~finite)
    if bool(missing.any()):
        raise RuntimeError(
            f"Tiled inference left {int(missing.sum())} accepted pixels uncovered after "
            "tile-edge halo rejection. Increase tile overlap or reduce the halo."
        )


def _configured_nan_border_px(model, config: dict, override: int | None = None) -> int:
    if override is not None and int(override) >= 0:
        return int(override)
    if hasattr(model, "invalid_support_border_px"):
        try:
            return int(max(0, model.invalid_support_border_px()))
        except Exception:
            pass
    return int(invalid_support_border_from_config(config))


def _apply_output_valid_mask(outputs: dict, valid_mask: np.ndarray | None) -> dict:
    if valid_mask is None:
        return outputs
    mask_np = np.asarray(valid_mask, dtype=bool)
    out = dict(outputs)
    for key in (
        "pred_map",
        "mask_pred_map",
        "masked_pred_map",
        "masked_target_pred_map",
        "gt_map",
        "context_map",
        "target_energy_map",
    ):
        value = out.get(key)
        if value is None:
            continue
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        if tensor.dim() < 4 or tuple(int(v) for v in tensor.shape[-2:]) != mask_np.shape:
            continue
        mask = torch.from_numpy(mask_np).to(device=tensor.device, dtype=torch.bool)
        while mask.dim() < tensor.dim():
            mask = mask.unsqueeze(0)
        if torch.is_floating_point(tensor):
            out[key] = torch.where(mask, tensor, torch.full_like(tensor, float("nan")))
    return out


def load_raw_data(
    input_path: str,
    crop_size: int | None = None,
    crop_mode: str = "center",
    mode: str = "image",
    slice_axis: int = 0,
    slice_index: int | None = None,
    return_layout: bool = False,
    slab_depth: int | None = None,
    crop_min_valid_fraction: float = 0.5,
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
        arr_raw = np.asarray(arr, dtype=np.float32)
        raw_valid_mask = _valid_pixel_mask(arr_raw)
        arr_norm = normalize01(arr_raw)
        tiles, layout = _tile_crops_2d_with_layout(
            arr_norm,
            crop_size,
            crop_mode,
            crop_min_valid_fraction,
            valid_mask=raw_valid_mask,
        )
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
    sigmas_override: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a CDD pyramid from a raw image tensor using the model config.

    Returns (cdd_fields, x_clean) where cdd_fields is B×S×H×W.
    For pure inference this is essentially computing CDD decomposition
    using the same pipeline as training.
    """
    # x: B×1×H×W → list of B×S×H×W
    sigmas = sigmas_override if sigmas_override is not None else model_cfg.get("sigmas", [2, 4, 8, 16])
    cdd_mode = str(data_cfg.get("cdd_mode", model_cfg.get("cdd_mode", "log")))
    cdd_constrained = bool(model_cfg.get("cdd_constrained", True))
    cdd_sm_mode = str(data_cfg.get("cdd_sm_mode", model_cfg.get("cdd_sm_mode", "reflect")))
    cdd_gaussian_backend = str(model_cfg.get("cdd_gaussian_backend", data_cfg.get("cdd_gaussian_backend", "cuda")))
    cdd = import_constrained_diffusion(allow_monai=cdd_gaussian_backend == "monai")

    bsz = x.shape[0]
    x_np = x.squeeze(1).detach().cpu().numpy().astype(np.float32)

    cdd_num_channels = int(model_cfg.get("cdd_num_channels", len(sigmas)))
    cdd_request_num_channels = model_cfg.get("cdd_request_num_channels")
    append_residual = bool(model_cfg.get("cdd_append_last_residual", True))
    cdd_list = []
    for i in range(bsz):
        cdd_kwargs = dict(
            min_scale=min(float(s) for s in sigmas),
            max_scale=max(float(s) for s in sigmas),
            mode=cdd_mode,
            constrained=cdd_constrained,
            sm_mode=cdd_sm_mode,
            return_scales=True,
            verbose=False,
            use_gpu=device.type == "cuda",
            gaussian_backend=cdd_gaussian_backend,
        )
        if cdd_request_num_channels is not None:
            cdd_kwargs["num_channels"] = int(cdd_request_num_channels)
        cdd_result = safe_constrained_diffusion_decomposition(cdd, x_np[i], **cdd_kwargs)
        if not isinstance(cdd_result, tuple) or len(cdd_result) < 2:
            raise RuntimeError("CDD inference must return (bands, residual[, scales]).")
        bands, _residual = cdd_result[:2]
        bands_arr = np.asarray(bands, dtype=np.float32)
        if int(bands_arr.shape[0]) < cdd_num_channels:
            raise RuntimeError(
                f"CDD returned {bands_arr.shape[0]} bands, but the trained model requires "
                f"{cdd_num_channels}. Do not override the checkpoint's scale contract."
            )
        selected = np.clip(bands_arr[:cdd_num_channels], a_min=0.0, a_max=None)
        if append_residual:
            recomputed_residual = x_np[i] - np.sum(selected, axis=0, dtype=np.float32)
            selected[-1] += np.clip(recomputed_residual, a_min=0.0, a_max=None)
        cdd_t = torch.from_numpy(selected.astype(np.float32, copy=False)).to(device)
        if cdd_t.ndim == 3:
            cdd_list.append(cdd_t.unsqueeze(0))
        else:
            cdd_list.append(cdd_t)

    cdd_fields = torch.cat(cdd_list, dim=0)  # B×S×H×W
    x_clean = x.clone()  # Keep original for downstream

    return cdd_fields, x_clean


def _parse_sigmas_arg(value: str | None) -> list[float] | None:
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        return None
    return [float(p) for p in parts]


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

    Returns a dict compatible with dashboard artifact export / downstream analysis.
    """
    model.eval()

    pred_maps = []
    mask_pred_maps = []
    masked_pred_maps = []
    masked_target_pred_maps = []
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

        def _forward_clean(xv: torch.Tensor, cdv: torch.Tensor | None):
            if cdv is not None and model.mode == "pyramid":
                return model(
                    xv,
                    mask_inference=False,
                    context_data=None,
                    cdd_orig=cdv,
                    enable_grid_jitter=False,
                    enable_target_dithering=False,
                )
            return model(
                xv,
                mask_inference=False,
                enable_grid_jitter=False,
                enable_target_dithering=False,
            )

        if inference_tta_enabled:
            out, _ = _forward_tta_streaming_2d(
                x=x_in,
                mode=inference_tta_mode,
                forward_one=_forward_clean,
                cdd=cdd_fields,
            )
        else:
            out = _forward_clean(x_in, cdd_fields)

        pred_map = out.get("pred_map")
        # Inference-session latent exports must be dense.  Do not expose the
        # model's training-style target-masked branch here; it creates scattered
        # target artifacts in latent maps.  A true masked branch should be
        # computed by the dense sliding/composed mask inference path instead.
        mask_pred_map = pred_map if bool(mask_inference) else None
        masked_pred_map = mask_pred_map
        masked_target_pred_map = None
        gt_map = out.get("gt_map")
        context_map = out.get("context_map")

        if pred_map is not None:
            pred_maps.append(pred_map.cpu())
        if mask_pred_map is not None:
            mask_pred_maps.append(mask_pred_map.cpu())
        if masked_pred_map is not None:
            masked_pred_maps.append(masked_pred_map.cpu())
        if masked_target_pred_map is not None:
            masked_target_pred_maps.append(masked_target_pred_map.cpu())
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

    # Tiles may produce different N_targets → pad to max before stacking.
    def _pad_and_stack(lst, dim=1, pad_val=-1):
        if not lst:
            return None
        if len(lst) == 1:
            return lst[0]
        shapes = [t.shape[dim] for t in lst]
        if len(set(shapes)) == 1:
            return torch.cat(lst, dim=0)
        max_n = max(shapes)
        padded = []
        for t in lst:
            if t.shape[dim] < max_n:
                pad_shape = list(t.shape)
                pad_shape[dim] = max_n - t.shape[dim]
                pad = torch.full(pad_shape, pad_val, dtype=t.dtype, device=t.device)
                t = torch.cat([t, pad], dim=dim)
            padded.append(t)
        return torch.cat(padded, dim=0)

    outputs = {
        "pred_map": _stack_or_none(pred_maps),
        "mask_pred_map": _stack_or_none(mask_pred_maps),
        "masked_pred_map": _stack_or_none(masked_pred_maps),
        "masked_target_pred_map": _stack_or_none(masked_target_pred_maps),
        "gt_map": _stack_or_none(gt_maps),
        "context_map": _stack_or_none(context_maps),
        "x_clean_raw": _stack_or_none(x_clean_list) if x_clean_list else None,
        "x_context_raw": _stack_or_none(x_context_list) if x_context_list else None,
        "target_locations": _pad_and_stack(all_target_locs, dim=1, pad_val=-1),
        "target_scales": _pad_and_stack(all_target_scales, dim=1, pad_val=-1),
        "target_valid": _pad_and_stack(all_target_valid, dim=1, pad_val=0),
    }
    # Keep inference-only sessions compatible with the training dashboard path.
    outputs["x_clean"] = outputs["x_clean_raw"]
    outputs["x_context"] = outputs["x_context_raw"]
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
    make_dashboard: bool = True,
    umap_cfg: dict | None = None,
    tile_layout: TileLayout2D | None = None,
    crop_min_valid_fraction: float | None = None,
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
        "crop_min_valid_fraction": crop_min_valid_fraction,
        "mode": mode,
        "mask_inference": bool(mask_inference),
    }
    if tile_layout is not None:
        config_out["_inference"]["tile_layout"] = {
            "original_shape": list(tile_layout.original_shape),
            "crop_size": int(tile_layout.crop_size),
            "num_tiles": int(len(tile_layout.origins)),
            "min_valid_fraction": float(tile_layout.min_valid_fraction),
            "edge_halo_px": int(tile_layout.edge_halo_px),
        }

    with open(os.path.join(output_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2)

    # Save raw tensors
    outputs = dict(outputs)
    if outputs.get("x_clean") is None and outputs.get("x_clean_raw") is not None:
        outputs["x_clean"] = outputs["x_clean_raw"]
    if outputs.get("x_context") is None and outputs.get("x_context_raw") is not None:
        outputs["x_context"] = outputs["x_context_raw"]
    if outputs.get("target_energy_map") is None:
        pred = outputs.get("masked_pred_map", outputs.get("pred_map"))
        target = outputs.get("gt_map")
        if pred is not None and target is not None:
            pred_t = torch.as_tensor(pred).detach().float().cpu()
            target_t = torch.as_tensor(target).detach().float().cpu()
            if pred_t.dim() == 5:
                pred_t = pred_t[:, :, pred_t.shape[2] // 2]
            if target_t.dim() == 5:
                target_t = target_t[:, :, target_t.shape[2] // 2]
            if pred_t.dim() == 4 and target_t.dim() == 4 and pred_t.shape == target_t.shape:
                outputs["target_energy_map"] = (pred_t - target_t).pow(2).mean(dim=1, keepdim=True)
    torch.save(outputs, os.path.join(output_dir, "inference_outputs.pt"))
    # Save tile visit heatmap for tiled inference
    if tile_layout is not None and tile_layout.visit_map is not None:
        np.save(os.path.join(output_dir, "tile_visit_map.npy"), tile_layout.visit_map.astype(np.int32))

    # Save compressed NPZ maps and target metadata.
    for key in ("pred_map", "mask_pred_map", "masked_pred_map", "masked_target_pred_map", "gt_map", "context_map", "target_locations", "target_scales", "target_valid", "target_energy_map"):
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

    # Dashboard/UMAP data can be CPU-heavy; keep it opt-in for API smoke paths.
    if make_dashboard:
        try:
            artifacts_dir = export_inference_dashboard_artifacts(output_dir, outputs, umap_cfg=umap_cfg or {})
            print(f"[inference] dashboard_artifacts_saved={artifacts_dir}")
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
        "max_crop": "crop_size",
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
        "nan_border_px": "nan_border_px",
        "inference_sigmas": "inference_sigmas",
    }

    cli_defaults = {
        "session": None,
        "input": None,
        "crop_size": None,
        "crop_mode": "tile",
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
        "nan_border_px": None,
        "inference_sigmas": None,
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
    parser.add_argument("--crop-size", "--max-crop", dest="crop_size", type=int, default=None, help="Crop/tile size for large inputs")
    parser.add_argument("--crop-mode", default="tile", choices=["center", "tile"], help="Crop mode")
    parser.add_argument(
        "--crop-min-valid-fraction",
        type=float,
        default=0.8,
        help="Keep only tiled cutouts with more than this fraction of finite pixels",
    )
    parser.add_argument(
        "--nan-border-px",
        type=int,
        default=None,
        help="Set output pixels within this many px of NaN/no-data to NaN. Default auto uses max configured inference scale.",
    )
    parser.add_argument(
        "--inference-sigmas",
        default=None,
        help="Optional comma-separated CDD sigmas for inference-time CDD construction, e.g. 2,4,8,16.",
    )
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
    inference_sigmas = _parse_sigmas_arg(args.inference_sigmas)
    if inference_sigmas is not None:
        config.setdefault("_inference_overrides", {})["sigmas"] = inference_sigmas
        print(f"[inference] inference_sigmas_override={inference_sigmas}")
    print(f"[inference] model loaded from {source_session}")

    # Load data
    data_tensor, tile_layout = load_raw_data(
        args.input,
        crop_size=args.crop_size,
        crop_mode=args.crop_mode,
        crop_min_valid_fraction=args.crop_min_valid_fraction,
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
    raw_output_valid_mask = None
    if str(args.mode).strip().lower() == "image":
        raw_arr_for_mask = np.asarray(np.squeeze(_safe_load_npy(args.input, mmap_mode="r")), dtype=np.float32)
        if raw_arr_for_mask.ndim == 2:
            raw_valid = _valid_pixel_mask(raw_arr_for_mask)
            nan_border_px = _configured_nan_border_px(model, config, args.nan_border_px)
            raw_output_valid_mask = _erode_valid_mask(raw_valid, nan_border_px)
            rejected = int(raw_valid.sum() - raw_output_valid_mask.sum())
            print(
                f"[inference] output NaN boundary rejection: border_px={nan_border_px} "
                f"valid_before={int(raw_valid.sum())}/{raw_valid.size} "
                f"valid_after={int(raw_output_valid_mask.sum())}/{raw_output_valid_mask.size} "
                f"rejected_near_nan={rejected}"
            )
            if tile_layout is not None:
                tile_layout.valid_mask = raw_output_valid_mask.copy()
                tile_layout.edge_halo_px = int(nan_border_px)
                print(
                    f"[inference] tile-edge rejection: halo_px={tile_layout.edge_halo_px} "
                    "(internal contaminated borders receive zero stitch weight)"
                )
    cdd_tensor = None
    if (
        str(args.mode).strip().lower() == "image"
        and str(getattr(model, "mode", "")).strip().lower() == "pyramid"
    ):
        arr_raw = _safe_load_npy(args.input, mmap_mode="r")
        arr_raw = np.asarray(np.squeeze(arr_raw), dtype=np.float32)
        if arr_raw.ndim != 2:
            raise ValueError(f"Global tiled CDD path expects 2D image input, got shape={arr_raw.shape}")
        arr_norm = normalize01(arr_raw)
        x_full = torch.from_numpy(arr_norm).view(1, 1, *arr_norm.shape)
        expected_scales = int(getattr(getattr(model, "context_encoder", None), "num_scales", len(model.sigmas)))
        requested_scales = inference_sigmas if inference_sigmas is not None else list(model.sigmas)
        if len(requested_scales) != expected_scales:
            raise ValueError(
                f"Inference CDD scale count {len(requested_scales)} does not match the trained encoder's "
                f"{expected_scales} channels. Use the checkpoint scales or a matching checkpoint."
            )
        print("[inference] building one training-contract full-frame CDD before encoding")
        cdd_full, _ = _build_cdd_pyramid(
            x_full,
            config.get("model", {}),
            config.get("data", {}),
            device,
            sigmas_override=inference_sigmas,
        )
        if tile_layout is None:
            cdd_tensor = cdd_full.detach().cpu()
        else:
            cdd_tensor = _tile_channel_field_with_layout(cdd_full.detach().cpu(), tile_layout)
        if int(cdd_tensor.shape[0]) != int(data_tensor.shape[0]):
            raise RuntimeError(
                f"CDD tile count {cdd_tensor.shape[0]} != image tile count {data_tensor.shape[0]}"
            )
        print(f"[inference] global CDD tiled shape={tuple(cdd_tensor.shape)}")

    # Build a simple DataLoader
    class _TensorDataset(torch.utils.data.Dataset):
        def __init__(self, t, cdd=None):
            self.t = t
            self.cdd = cdd

        def __len__(self):
            return self.t.shape[0]

        def __getitem__(self, idx):
            if self.cdd is not None:
                return self.cdd[idx], self.t[idx]
            return self.t[idx]

    dataset = _TensorDataset(data_tensor, cdd_tensor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_for_inference,
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
    outputs = _apply_output_valid_mask(outputs, raw_output_valid_mask)
    if raw_output_valid_mask is not None:
        allowed = torch.from_numpy(raw_output_valid_mask.astype(np.float32))[None, None]
        outputs["target_allowed_mask_map"] = allowed
        outputs["output_valid_mask"] = allowed.to(dtype=torch.bool)
    _assert_valid_output_coverage(outputs, raw_output_valid_mask)
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
        tile_layout=tile_layout,
        crop_min_valid_fraction=args.crop_min_valid_fraction,
    )

    print(f"[inference] done → {output_dir}")


if __name__ == "__main__":
    main()
