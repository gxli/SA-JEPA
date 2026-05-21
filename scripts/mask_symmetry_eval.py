#!/usr/bin/env python3
import argparse
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import plotly.graph_objects as go
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import JEPADataset
from src.models.build_jepa import make_pyramid_grid_context


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(data_cfg: dict, for_cdd_masking: bool = False) -> JEPADataset:
    ds_log_transform = bool(data_cfg.get("log_transform", True))
    ds_apply_cdd = True
    if for_cdd_masking:
        ds_log_transform = False
        ds_apply_cdd = False
    return JEPADataset(
        num_samples=max(1, int(data_cfg.get("num_samples", 1))),
        image_size=int(data_cfg.get("image_size", 256)),
        data_root=data_cfg.get("data_root", "data"),
        npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
        log_transform=ds_log_transform,
        log_eps=float(data_cfg.get("log_eps", 1.0)),
        cdd_scales=data_cfg.get("cdd_scales", [2, 4, 8]),
        cdd_strength=float(data_cfg.get("cdd_strength", 1.0)),
        cdd_clip=bool(data_cfg.get("cdd_clip", True)),
        norm_before_cdd=bool(data_cfg.get("norm_before_cdd", True)),
        cdd_mode=data_cfg.get("cdd_mode", "log"),
        cdd_constrained=bool(data_cfg.get("cdd_constrained", True)),
        cdd_sm_mode=data_cfg.get("cdd_sm_mode", "reflect"),
        apply_cdd=ds_apply_cdd,
        cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
        cube_slice_axis=int(data_cfg.get("cube_slice_axis", 0)),
        cube_slice_index=int(data_cfg.get("cube_slice_index", 0)),
    )


def make_context_and_debug(x: torch.Tensor, model_cfg: dict, seed: int):
    torch.manual_seed(seed)
    return make_pyramid_grid_context(
        x_clean=x,
        sigmas=tuple(model_cfg.get("sigmas", [2, 4, 8, 16])),
        cell_sizes=tuple(model_cfg.get("cell_sizes", [16, 32, 64, 128])),
        mask_fraction=float(model_cfg.get("mask_fraction", 1.0)),
        box_sigma_mult=float(model_cfg.get("box_sigma_mult", 4.0)),
        mask_scale=float(model_cfg.get("mask_scale", 1.0)),
        spacing_scale=float(model_cfg.get("spacing_scale", 1.5)),
        full_grid=bool(model_cfg.get("full_grid", True)),
        global_shift=bool(model_cfg.get("global_shift", True)),
        align_scales=bool(model_cfg.get("align_scales", True)),
        constant_mask_box=bool(model_cfg.get("constant_mask_box", True)),
        mask_box_size=int(model_cfg.get("mask_box_size", 16)),
        blur_mode=model_cfg.get("blur_mode", "cdd"),
        cdd_mode=model_cfg.get("cdd_mode", "log"),
        cdd_constrained=bool(model_cfg.get("cdd_constrained", True)),
        cdd_sm_mode=model_cfg.get("cdd_sm_mode", "reflect"),
        mask_fill_mode=model_cfg.get("mask_fill_mode", "zero"),
        dip_sigma_mult=float(model_cfg.get("dip_sigma_mult", 1.0)),
        constant_gaussian_sigma=float(model_cfg.get("constant_gaussian_sigma", 1.0)),
        return_debug=True,
    )


def evaluate_mask_symmetry(
    ds: JEPADataset,
    model_cfg_run: dict,
    n_samples: int,
    base_seed: int,
    visit_source: str = "hard",
) -> tuple[np.ndarray, dict]:
    n = int(max(1, n_samples))
    x0 = ds[0][0].numpy().astype(np.float32)
    h, w = int(x0.shape[0]), int(x0.shape[1])
    acc = np.zeros((h, w), dtype=np.float64)
    for i in range(n):
        x = ds[i % len(ds)][0].numpy().astype(np.float32)
        x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
        _, _, _, _, debug = make_context_and_debug(x_t, model_cfg_run, int(base_seed + i))
        if visit_source == "gaussian":
            dip_t = debug.get("dip_field")
            if dip_t is None or dip_t.numel() == 0:
                mask = np.clip(debug["mask_map"][0].cpu().numpy().astype(np.float32), 0.0, 1.0)
            else:
                mask = np.clip(dip_t[0].cpu().numpy().astype(np.float32), 0.0, 1.0)
        elif visit_source == "centers":
            mask = np.zeros((h, w), dtype=np.float32)
            ctr = debug.get("unique_centers")
            if ctr is not None and ctr.numel() > 0:
                pts = ctr[0].cpu().numpy()
                for cy, cx in pts.tolist():
                    cy = int(cy)
                    cx = int(cx)
                    if cy >= 0 and cx >= 0 and cy < h and cx < w:
                        mask[cy, cx] = 1.0
        else:
            # "hard": true applied footprint used by masking logic.
            mask = np.clip(debug["mask_map"][0].cpu().numpy().astype(np.float32), 0.0, 1.0)
        acc += mask.astype(np.float64)
    heat = (acc / float(n)).astype(np.float32)
    lr_flip = heat[:, ::-1]
    tb_flip = heat[::-1, :]
    metrics = {
        "samples": int(n),
        "mean_mask_value": float(np.mean(heat)),
        "lr_symmetry_mae": float(np.mean(np.abs(heat - lr_flip))),
        "tb_symmetry_mae": float(np.mean(np.abs(heat - tb_flip))),
        "top_minus_bottom": float(np.mean(heat[: (h // 2), :]) - np.mean(heat[(h // 2) :, :])),
        "left_minus_right": float(np.mean(heat[:, : (w // 2)]) - np.mean(heat[:, (w // 2) :])),
    }
    return heat, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate mask boundary symmetry via aggregate heatmap")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sessions-dir", type=str, default="sessions")
    parser.add_argument("--mask-mode", type=str, choices=["config", "zero", "gaussian_dip"], default="config")
    parser.add_argument("--force-blur-mode", type=str, choices=["gaussian", "cdd"], default=None)
    parser.add_argument("--rigid-mask-box", action="store_true")
    parser.add_argument("--no-align-scales", action="store_true", help="Disable shared cross-scale aligned grid centers")
    parser.add_argument("--no-full-grid", action="store_true", help="Sample a stochastic subset instead of full lattice")
    parser.add_argument("--no-global-shift", action="store_true", help="Disable per-sample global lattice shift")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--visit-source",
        type=str,
        choices=["hard", "gaussian", "centers"],
        default="hard",
        help="What to accumulate: hard mask footprint, gaussian dip field, or centers only",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = dict(cfg.get("model", {}))
    data_cfg = cfg.get("data", {})
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    session_dir = os.path.join(args.sessions_dir, config_name)
    os.makedirs(session_dir, exist_ok=True)

    mode = args.mask_mode
    if mode == "config":
        mode = str(model_cfg.get("mask_fill_mode", "zero"))
    model_cfg["mask_fill_mode"] = mode
    if args.force_blur_mode is not None:
        model_cfg["blur_mode"] = args.force_blur_mode
    model_cfg["constant_mask_box"] = bool(model_cfg.get("constant_mask_box", True))
    if not args.rigid_mask_box:
        model_cfg["constant_mask_box"] = False
    if args.no_align_scales:
        model_cfg["align_scales"] = False
    if args.no_full_grid:
        model_cfg["full_grid"] = False
    if args.no_global_shift:
        model_cfg["global_shift"] = False

    ds = build_dataset(data_cfg, for_cdd_masking=(model_cfg.get("blur_mode", "cdd") == "cdd"))
    heat, metrics = evaluate_mask_symmetry(
        ds,
        model_cfg,
        int(args.samples),
        int(args.seed),
        visit_source=str(args.visit_source),
    )

    html_path = os.path.join(session_dir, f"mask_symmetry_eval_{mode}_{int(args.samples)}.html")
    npy_path = os.path.join(session_dir, f"mask_symmetry_eval_{mode}_{int(args.samples)}.npy")
    json_path = os.path.join(session_dir, f"mask_symmetry_eval_{mode}_{int(args.samples)}.json")

    vmax = float(max(1e-6, np.max(heat)))
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=heat,
                colorscale="Magma",
                zmin=0.0,
                zmax=vmax,
                colorbar=dict(title="freq"),
            )
        ]
    )
    fig.update_layout(
        title=f"Mask Frequency Heatmap ({mode}, source={args.visit_source}, n={int(args.samples)})",
        template="plotly_white",
        width=820,
        height=720,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showticklabels=False, constrain="domain")
    fig.update_yaxes(showticklabels=False, scaleanchor="x", scaleratio=1, constrain="domain")
    fig.write_html(html_path, include_plotlyjs="cdn")

    np.save(npy_path, heat)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "mode": mode,
                "samples": int(args.samples),
                "seed": int(args.seed),
                "visit_source": str(args.visit_source),
                "metrics": metrics,
                "heatmap_npy": npy_path,
                "heatmap_html": html_path,
            },
            f,
            indent=2,
        )

    print(f"saved_html={html_path}")
    print(f"saved_npy={npy_path}")
    print(f"saved_json={json_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
