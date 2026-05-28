#!/usr/bin/env python3
"""Convert per-epoch embedding snapshots to movie frames (PCA + UMAP)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _fit_pca_2d(all_pred: np.ndarray, all_gt: np.ndarray):
    """Fit one PCA on concatenated pred+gt from all frames."""
    x = np.concatenate([all_pred, all_gt], axis=0)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(x)
    return pca


def _fit_umap_2d(all_pred: np.ndarray, all_gt: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.15):
    """Fit one UMAP on concatenated pred+gt from all frames."""
    try:
        import umap
    except ImportError:
        print("[session_to_movie] umap-learn not installed; skipping UMAP, will use PCA only")
        return None
    x = np.concatenate([all_pred, all_gt], axis=0)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=42,
        init="spectral",
    )
    reducer.fit(x)
    return reducer


def _render_frame(
    pred_map: np.ndarray,   # [C, H, W] or [H*W, C]
    gt_map: np.ndarray,
    pca,
    umap_reducer,
    epoch: int,
    out_path: str,
):
    """Render a 2x2 frame: pred PCA | gt PCA / pred UMAP | gt UMAP."""
    H, W = pred_map.shape[-2], pred_map.shape[-1]
    C = pred_map.shape[0] if pred_map.ndim == 3 else pred_map.shape[-1]

    # Flatten to [N, C]
    if pred_map.ndim == 3:
        pred_flat = pred_map.reshape(C, -1).T  # [H*W, C]
        gt_flat = gt_map.reshape(C, -1).T
    else:
        pred_flat = pred_map
        gt_flat = gt_map

    # Sub-sample for speed (max 5000 points)
    n_max = 5000
    if pred_flat.shape[0] > n_max:
        rng = np.random.RandomState(epoch)
        idx = rng.choice(pred_flat.shape[0], n_max, replace=False)
        pred_sub = pred_flat[idx]
        gt_sub = gt_flat[idx]
    else:
        pred_sub = pred_flat
        gt_sub = gt_flat

    pred_pca = pca.transform(pred_sub)
    gt_pca = pca.transform(gt_sub)

    if umap_reducer is not None:
        pred_umap = umap_reducer.transform(pred_sub)
        gt_umap = umap_reducer.transform(gt_sub)
    else:
        pred_umap = pred_pca
        gt_umap = gt_pca

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(pred_pca[:, 0], pred_pca[:, 1], s=2, c="C0", alpha=0.6)
    ax1.set_title("Pred PCA", fontsize=10)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(gt_pca[:, 0], gt_pca[:, 1], s=2, c="C1", alpha=0.6)
    ax2.set_title("GT PCA", fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(pred_umap[:, 0], pred_umap[:, 1], s=2, c="C0", alpha=0.6)
    ax3.set_title("Pred UMAP", fontsize=10)
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(gt_umap[:, 0], gt_umap[:, 1], s=2, c="C1", alpha=0.6)
    ax4.set_title("GT UMAP", fontsize=10)
    ax4.set_xticks([])
    ax4.set_yticks([])

    fig.suptitle(f"Epoch {epoch}", fontsize=13, y=0.98)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def collect_all_frames(movie_dir: str) -> tuple[list[int], list[str], np.ndarray, np.ndarray]:
    """Load all frames and return aggregated pred/gt arrays for fitting."""
    import glob
    pattern = os.path.join(movie_dir, "epoch_*.pt")
    paths = sorted(glob.glob(pattern))
    epochs = []
    all_pred_chunks = []
    all_gt_chunks = []
    for p in paths:
        fname = os.path.basename(p)
        ep = int(fname.replace("epoch_", "").replace(".pt", ""))
        frame = torch.load(p, map_location="cpu", weights_only=False)
        pm = frame["pred_map"][0].numpy()  # [C, H, W]
        gm = frame["gt_map"][0].numpy()
        C, H, W = pm.shape
        pred_flat = pm.reshape(C, -1).T  # [H*W, C]
        gt_flat = gm.reshape(C, -1).T
        # Subsample for PCA/UMAP fitting
        n_max_fit = 2000
        if pred_flat.shape[0] > n_max_fit:
            rng = np.random.RandomState(42)
            idx = rng.choice(pred_flat.shape[0], n_max_fit, replace=False)
            pred_flat = pred_flat[idx]
            gt_flat = gt_flat[idx]
        all_pred_chunks.append(pred_flat)
        all_gt_chunks.append(gt_flat)
        epochs.append(ep)
    all_pred = np.concatenate(all_pred_chunks, axis=0)
    all_gt = np.concatenate(all_gt_chunks, axis=0)
    return epochs, paths, all_pred, all_gt


def main():
    parser = argparse.ArgumentParser(description="Convert session movie frames to PNGs")
    parser.add_argument("session_dir", help="Path to session directory containing movie_frames/")
    parser.add_argument("--out-dir", default=None, help="Output directory for PNG frames")
    parser.add_argument("--make-mp4", action="store_true", help="Generate MP4 with ffmpeg")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second for MP4")
    args = parser.parse_args()

    session_dir = os.path.abspath(args.session_dir)
    movie_dir = os.path.join(session_dir, "movie_frames")
    if not os.path.isdir(movie_dir):
        print(f"[session_to_movie] movie_frames/ not found in {session_dir}")
        sys.exit(1)

    session_name = os.path.basename(session_dir.rstrip("/"))
    out_root = args.out_dir or os.path.join(ROOT, "results", "movie_pngs")
    out_dir = os.path.join(out_root, session_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[session_to_movie] loading frames from {movie_dir}")
    epochs, frame_paths, all_pred, all_gt = collect_all_frames(movie_dir)
    print(f"[session_to_movie] loaded {len(epochs)} frames, fitting PCA...")

    pca = _fit_pca_2d(all_pred, all_gt)
    print(f"[session_to_movie] PCA fit done, fitting UMAP...")
    umap_reducer = _fit_umap_2d(all_pred, all_gt)
    print(f"[session_to_movie] UMAP {'fit done' if umap_reducer else 'skipped'}")

    # Load loss weights for annotation
    lw_path = os.path.join(session_dir, "loss_weights.json")
    loss_weights = {}
    if os.path.exists(lw_path):
        with open(lw_path) as f:
            loss_weights = json.load(f)

    for i, (ep, fp) in enumerate(zip(epochs, frame_paths)):
        frame = torch.load(fp, map_location="cpu", weights_only=False)
        pm = frame["pred_map"][0].numpy()
        gm = frame["gt_map"][0].numpy()
        out_path = os.path.join(out_dir, f"frame_{i:04d}.png")
        _render_frame(pm, gm, pca, umap_reducer, ep, out_path)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[session_to_movie] rendered {i + 1}/{len(epochs)} frames")

    print(f"[session_to_movie] all {len(epochs)} frames saved to {out_dir}")

    if args.make_mp4:
        mp4_path = os.path.join(out_dir, f"{session_name}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", os.path.join(out_dir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            mp4_path,
        ]
        import subprocess
        subprocess.run(cmd, check=True)
        print(f"[session_to_movie] mp4 saved to {mp4_path}")


if __name__ == "__main__":
    main()
