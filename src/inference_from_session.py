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
from src.utils.npy import _safe_load_npy
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

def _normalize01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    amin, amax = float(a.min()), float(a.max())
    if amax - amin > 1e-20:
        return ((a - amin) / (amax - amin)).astype(np.float32)
    return np.zeros_like(a, dtype=np.float32)


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

    # If only one tile requested (center mode)
    if crop_mode == "center" and len(tiles) > 1:
        y0 = max(0, (h - cs) // 2)
        x0 = max(0, (w - cs) // 2)
        tile = np.zeros((cs, cs), dtype=np.float32)
        th = min(cs, h - y0)
        tw = min(cs, w - x0)
        tile[:th, :tw] = np.asarray(arr2d[y0 : y0 + th, x0 : x0 + tw], dtype=np.float32)
        return [tile]

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
    h, w = layout.original_shape
    out_shape = (1, *tensor.shape[1:-2], h, w)
    # Accumulate on CPU to avoid GPU OOM on large images (e.g. 10k×10k)
    out = torch.zeros(out_shape, dtype=tensor.dtype, device="cpu")
    counts = torch.zeros((1, *([1] * (tensor.dim() - 3)), h, w), dtype=tensor.dtype, device="cpu")
    for idx, ((y0, x0), (th, tw)) in enumerate(zip(layout.origins, layout.valid_shapes)):
        tile = tensor[idx : idx + 1, ..., :th, :tw].cpu()
        out[..., y0 : y0 + th, x0 : x0 + tw] += tile
        counts[..., y0 : y0 + th, x0 : x0 + tw] += 1
    return out / counts.clamp_min(1)


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
) -> torch.Tensor | tuple[torch.Tensor, TileLayout2D | None]:
    """Load an arbitrary .npy and return a B×1×H×W tensor (or B×1×D×H×W for 3D).

    Args:
        input_path: path to a .npy file (2D or 3D).
        crop_size: if set and data is larger, crop or tile.
        crop_mode: 'center' (single crop) or 'tile' (tiled crops across image).
        mode: 'image' (2D) or '3d_slab' (3D volume).
        slice_axis: for 3D mode, which axis to treat as depth (default 0).
        slice_index: for 3D mode, specific slice index, or None for all slices.

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
        depth = arr.shape[axis]

        if slice_index is not None:
            slices = [int(slice_index) % depth]
        else:
            slices = list(range(depth))

        slabs = []
        for idx in slices:
            slab = np.asarray(np.take(arr, idx, axis=axis), dtype=np.float32)
            if crop_size and max(slab.shape) > crop_size:
                tiles = _tile_crops_2d(slab, crop_size, crop_mode)
                for tile in tiles:
                    slabs.append(_normalize01(tile))
            else:
                slabs.append(_normalize01(slab))
        tensor = np.stack(slabs, axis=0)  # B×H×W
        out = torch.from_numpy(tensor).unsqueeze(1)  # B×1×H×W
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
        tiles, layout = _tile_crops_2d_with_layout(arr, crop_size, crop_mode)
        tensor = np.stack([_normalize01(t) for t in tiles], axis=0)
    else:
        layout = None
        tensor = _normalize01(np.asarray(arr, dtype=np.float32))[np.newaxis, ...]  # 1×H×W

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
                return model(xv, mask_inference=False, context_data=None, cdd_orig=cdv)
            return model(xv, mask_inference=False)

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
        if x_batch is not None:
            x_clean_list.append(x_batch.cpu())
        x_context_list.append(cdd_batch.cpu())

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
    }

    with open(os.path.join(output_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2)

    # Save raw tensors
    torch.save(outputs, os.path.join(output_dir, "inference_outputs.pt"))

    # Save compressed NPZ maps
    for key in ("pred_map", "gt_map", "context_map"):
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
        "pred_map_shape": list(outputs["pred_map"].shape) if outputs.get("pred_map") is not None else None,
        "gt_map_shape": list(outputs["gt_map"].shape) if outputs.get("gt_map") is not None else None,
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
      --input data/ngc3627_12m+7m+tp_co21_strict_mom0.npy_sm.npy \\
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
    )

    print(f"[inference] done → {output_dir}")


if __name__ == "__main__":
    main()
