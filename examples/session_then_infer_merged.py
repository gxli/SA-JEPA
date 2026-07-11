#!/usr/bin/env python3
"""Load a trained session, run inference on a pixel-merged field, and save a new session.

This example does not train. Run ``examples/example_config_driven.py`` first, or
pass ``--source-session`` pointing at an existing trained session that contains
``model_last.pt`` and saved ``umap_weights_predict.pkl``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("SAJEPA_ENABLE_CPU_UMAP", "1")
os.environ.setdefault("SAJEPA_ENABLE_TORCHDR_UMAP", "0")
os.environ.setdefault("SAJEPA_STRICT_UMAP", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
sys.path.insert(0, str(ROOT))

import numpy as np

from sajepa import ScaleAwareJEPA
from src.train import load_config


def _input_from_config(config_path: Path) -> Path:
    cfg = load_config(str(config_path))
    data_cfg = cfg.get("data", {})
    data_root = Path(data_cfg.get("data_root", ""))
    npy_pattern = data_cfg.get("npy_pattern")
    if not npy_pattern:
        raise ValueError(f"{config_path} has no data.npy_pattern")
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    return data_root / str(npy_pattern)


def _as_float_field(path: Path, zero_to_nan: bool) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float32)
    if zero_to_nan:
        arr[arr == 0] = np.nan
    if arr.ndim < 2:
        raise ValueError(f"Expected at least a 2D field, got shape={arr.shape}")
    return arr


def _merge_pixels_last2(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Downsample the last two axes by averaging all pixels inside output bins."""
    h, w = arr.shape[-2:]
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"merged size must be positive, got {out_h}x{out_w}")
    if h < out_h or w < out_w:
        raise ValueError(f"Cannot merge input shape {h}x{w} into larger shape {out_h}x{out_w}")

    y_edges = np.linspace(0, h, out_h + 1, dtype=np.int64)
    x_edges = np.linspace(0, w, out_w + 1, dtype=np.int64)
    merged = np.empty((*arr.shape[:-2], out_h, out_w), dtype=np.float32)
    for yi in range(out_h):
        y0, y1 = int(y_edges[yi]), int(y_edges[yi + 1])
        for xi in range(out_w):
            x0, x1 = int(x_edges[xi]), int(x_edges[xi + 1])
            with np.errstate(invalid="ignore", divide="ignore"):
                merged[..., yi, xi] = np.nanmean(arr[..., y0:y1, x0:x1], axis=(-2, -1))
    return merged


def _set_umap_reuse_defaults(model: ScaleAwareJEPA, fit_max_tokens: int, transform_batch: int) -> None:
    train_cfg = model._config.setdefault("train", {})
    umap_cfg = train_cfg.setdefault("umap", {})
    umap_cfg["save_umap_weights"] = True
    umap_cfg["reuse_umap_weights"] = True
    umap_cfg["fit_max_tokens"] = int(fit_max_tokens)
    umap_cfg["transform_batch"] = int(transform_batch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-session",
        type=Path,
        default=ROOT / "sessions" / "example_config_driven",
        help="Trained session to load. Defaults to the config-driven MHD example session.",
    )
    parser.add_argument(
        "--input-npy",
        type=Path,
        default=None,
        help="Input .npy to merge and infer. Defaults to configs/examples/mhd_example.yaml input.",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "examples" / "mhd_example.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--merged-size", type=int, default=None, help="Square merged output size. Defaults to half resolution.")
    parser.add_argument("--zero-to-nan", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--umap-fit-max-tokens", type=int, default=2048)
    parser.add_argument("--umap-transform-batch", type=int, default=2048)
    args = parser.parse_args()

    source_session = args.source_session if args.source_session.is_absolute() else ROOT / args.source_session
    input_npy = args.input_npy or _input_from_config(args.config)
    input_npy = input_npy if input_npy.is_absolute() else ROOT / input_npy
    if not source_session.exists():
        raise FileNotFoundError(
            f"source session not found: {source_session}\n"
            "Train one first, for example: python examples/example_config_driven.py"
        )

    field = _as_float_field(input_npy, zero_to_nan=args.zero_to_nan)
    h, w = field.shape[-2:]
    merged_size = int(args.merged_size) if args.merged_size is not None else max(1, min(h, w) // 2)
    merged = _merge_pixels_last2(field, merged_size, merged_size)

    output_root = source_session.parent
    merged_path = output_root / f"{input_npy.stem}_merged_{merged_size}x{merged_size}.npy"
    np.save(merged_path, merged.astype(np.float32, copy=False))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = output_root / f"{source_session.name}_infer_merged_{merged_size}x{merged_size}"
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir

    model = ScaleAwareJEPA.load_session(str(source_session))
    _set_umap_reuse_defaults(
        model,
        fit_max_tokens=args.umap_fit_max_tokens,
        transform_batch=args.umap_transform_batch,
    )
    inference_session = model.infer_npy(
        str(merged_path),
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        make_dashboard=True,
    )

    results = Path(inference_session) / "results"
    dashboard = Path(inference_session) / "dashboard.html"
    interactive = model.save_interactive_umap(
        str(results / "predict_umap_xyz.npy"),
        str(results / "interactive_umap_display_full_latent_similarity_predict.html"),
        similarity_npy=str(results / "predict_latent_vectors_full.npy"),
        display_label="umap",
        similarity_label="full_latent",
    )
    print(
        "\nDone."
        f"\n  source_session:    {source_session}"
        f"\n  input_npy:         {input_npy}"
        f"\n  merged_input:      {merged_path}"
        f"\n  inference_session: {inference_session}"
        f"\n  dashboard:         {dashboard}"
        f"\n  interactive_umap:  {interactive}"
    )


if __name__ == "__main__":
    main()
