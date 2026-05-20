#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import JEPADataset
from src.models.build_jepa import PyramidGridJEPA


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_pca_2d(x: np.ndarray) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA

        return PCA(n_components=2).fit_transform(x)
    except Exception:
        x_t = torch.from_numpy(x.astype(np.float32))
        x_t = x_t - x_t.mean(dim=0, keepdim=True)
        u, s, _ = torch.pca_lowrank(x_t, q=2)
        return (u[:, :2] * s[:2]).cpu().numpy()


def _compute_umap_2d(x: np.ndarray) -> np.ndarray:
    try:
        from cuml.manifold import UMAP as CuMLUMAP

        return CuMLUMAP(n_components=2, random_state=42).fit_transform(x)
    except Exception:
        pass
    try:
        import umap

        return umap.UMAP(n_components=2, random_state=42).fit_transform(x)
    except Exception:
        pass
    return _compute_pca_2d(x)


def build_model(model_cfg: dict, data_cfg: dict, device: torch.device) -> PyramidGridJEPA:
    blur_mode = model_cfg.get("blur_mode", "gaussian")
    if blur_mode == "cdd":
        model_post_log = bool(data_cfg.get("log_transform", True))
    else:
        model_post_log = bool(model_cfg.get("post_log_transform", data_cfg.get("log_transform", True)))

    model = PyramidGridJEPA(
        latent_channels=model_cfg.get("latent_channels", 32),
        predictor_hidden=model_cfg.get("predictor_hidden"),
        patch_size=model_cfg.get("patch_size", 2),
        sigmas=tuple(model_cfg.get("sigmas", [2, 4, 8, 16])),
        cell_sizes=tuple(model_cfg.get("cell_sizes", [16, 32, 64, 128])),
        max_targets_per_image=model_cfg.get("max_targets_per_image", 16),
        mask_fraction=model_cfg.get("mask_fraction", 0.20),
        spacing_mult=model_cfg.get("spacing_mult", 1.5),
        box_sigma_mult=model_cfg.get("box_sigma_mult", 4.0),
        mask_scale=model_cfg.get("mask_scale", 1.0),
        min_mask_scale=model_cfg.get("min_mask_scale", 0.0),
        spacing_scale=model_cfg.get("spacing_scale", 2.0),
        full_grid=model_cfg.get("full_grid", True),
        global_shift=model_cfg.get("global_shift", True),
        align_scales=model_cfg.get("align_scales", True),
        constant_mask_box=model_cfg.get("constant_mask_box", True),
        mask_box_size=model_cfg.get("mask_box_size", 16),
        blur_mode=blur_mode,
        cdd_mode=model_cfg.get("cdd_mode", "log"),
        cdd_constrained=model_cfg.get("cdd_constrained", True),
        cdd_sm_mode=model_cfg.get("cdd_sm_mode", "reflect"),
        mask_fill_mode=model_cfg.get("mask_fill_mode", "zero"),
        dip_sigma_mult=model_cfg.get("dip_sigma_mult", 1.0),
        post_log_transform=model_post_log,
        log_eps=model_cfg.get("log_eps", float(data_cfg.get("log_eps", 1.0))),
        cdd_log_std_floor_mult=model_cfg.get("cdd_log_std_floor_mult", 0.05),
        ema_momentum=model_cfg.get("ema_momentum", 0.996),
        normalize_loss=model_cfg.get("normalize_loss", True),
    ).to(device)
    return model


def _cube_depth(arr: np.ndarray, axis: int) -> int:
    if arr.ndim == 2:
        return 1
    if arr.ndim == 3:
        return arr.shape[axis % 3]
    raise ValueError(f"Expected 2D or 3D array, got {arr.shape}")


def main():
    parser = argparse.ArgumentParser(description="Full-volume inference over all cube slices")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sessions-dir", type=str, default="sessions")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-embed-tokens", type=int, default=200000)
    parser.add_argument("--file-index", type=int, default=None, help="Only run one matched file by index")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    session_dir = os.path.join(args.sessions_dir, config_name)
    os.makedirs(session_dir, exist_ok=True)

    data_root = data_cfg.get("data_root", "data")
    npy_pattern = data_cfg.get("npy_pattern", "*.npy")
    files = sorted(glob.glob(os.path.join(data_root, npy_pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched {os.path.join(data_root, npy_pattern)}")
    if args.file_index is not None:
        files = [files[int(args.file_index)]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg, data_cfg, device)

    ckpt = args.checkpoint or os.path.join(session_dir, "model_last.pt")
    loaded_ckpt = None
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        loaded_ckpt = ckpt
    model.eval()

    # Use dataset preprocessing exactly as training policy.
    ds = JEPADataset(
        num_samples=1,
        image_size=data_cfg.get("image_size", 256),
        data_root=data_root,
        npy_pattern=npy_pattern,
        log_transform=bool(data_cfg.get("log_transform", True)),
        log_eps=data_cfg.get("log_eps", 1.0),
        cdd_scales=data_cfg.get("cdd_scales", [2, 4, 8, 16]),
        cdd_strength=data_cfg.get("cdd_strength", 1.0),
        cdd_clip=data_cfg.get("cdd_clip", True),
        norm_before_cdd=data_cfg.get("norm_before_cdd", True),
        cdd_mode=data_cfg.get("cdd_mode", "log"),
        cdd_constrained=data_cfg.get("cdd_constrained", True),
        cdd_sm_mode=data_cfg.get("cdd_sm_mode", "reflect"),
        apply_cdd=(model_cfg.get("blur_mode", "gaussian") != "cdd"),
        cube_slice_strategy="fixed",
        cube_slice_axis=int(data_cfg.get("cube_slice_axis", 0)),
        cube_slice_index=0,
    )

    embed_blocks = []
    token_count = 0
    axis = int(data_cfg.get("cube_slice_axis", 0))
    per_file_meta = []

    with torch.no_grad():
        for fidx, cube_path in enumerate(files):
            arr = np.load(cube_path, mmap_mode="r")
            depth = _cube_depth(arr, axis)
            pred_norm_slices = []
            gt_norm_slices = []
            for sidx in range(depth):
                x = ds._load_sample(cube_path, forced_slice_idx=sidx).unsqueeze(0).to(device)
                out = model(x)
                pred = out["pred_map"][0]
                gt = out["gt_map"][0]
                pred_norm = pred.detach().cpu().norm(dim=0).numpy().astype(np.float32)
                gt_norm = gt.detach().cpu().norm(dim=0).numpy().astype(np.float32)
                pred_norm_slices.append(pred_norm)
                gt_norm_slices.append(gt_norm)

                if token_count < int(args.max_embed_tokens):
                    pv = pred.detach().cpu().permute(1, 2, 0).reshape(-1, pred.shape[0]).numpy()
                    gv = gt.detach().cpu().permute(1, 2, 0).reshape(-1, gt.shape[0]).numpy()
                    block = np.concatenate([pv, gv], axis=0)
                    left = int(args.max_embed_tokens) - token_count
                    if block.shape[0] > left:
                        idx = np.linspace(0, block.shape[0] - 1, num=max(1, left), dtype=np.int64)
                        block = block[idx]
                    embed_blocks.append(block)
                    token_count += block.shape[0]

            pred_vol = np.stack(pred_norm_slices, axis=0)
            gt_vol = np.stack(gt_norm_slices, axis=0)
            stem = os.path.splitext(os.path.basename(cube_path))[0]
            pred_path = os.path.join(session_dir, f"full_volume_pred_norm__{fidx:03d}__{stem}.npy")
            gt_path = os.path.join(session_dir, f"full_volume_gt_norm__{fidx:03d}__{stem}.npy")
            np.save(pred_path, pred_vol)
            np.save(gt_path, gt_vol)
            per_file_meta.append(
                {
                    "file_index": int(fidx),
                    "cube_path": cube_path,
                    "depth": int(depth),
                    "pred_norm_path": pred_path,
                    "gt_norm_path": gt_path,
                }
            )
            print(f"file_done={fidx} depth={depth} path={cube_path}")

    embeds = np.concatenate(embed_blocks, axis=0) if embed_blocks else np.zeros((1, 2), dtype=np.float32)
    pca_2d = _compute_pca_2d(embeds)
    umap_2d = _compute_umap_2d(embeds)
    np.save(os.path.join(session_dir, "full_volume_pca_embeddings.npy"), pca_2d)
    np.save(os.path.join(session_dir, "full_volume_umap_embeddings.npy"), umap_2d)

    meta = {
        "config": args.config,
        "slice_axis": axis,
        "checkpoint_loaded": loaded_ckpt,
        "n_files": int(len(files)),
        "files": per_file_meta,
        "pca_path": os.path.join(session_dir, "full_volume_pca_embeddings.npy"),
        "umap_path": os.path.join(session_dir, "full_volume_umap_embeddings.npy"),
        "embed_tokens_used": int(embeds.shape[0]),
    }
    with open(os.path.join(session_dir, "full_volume_inference_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"session_saved={session_dir}")
    print(f"n_files={len(files)}")
    print(f"checkpoint_loaded={loaded_ckpt}")
    print(f"saved={meta['pca_path']}")
    print(f"saved={meta['umap_path']}")


if __name__ == "__main__":
    main()
