#!/usr/bin/env python3
"""Convert per-epoch embedding snapshots to movie frames (PCA + UMAP)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")  # headless — no display required

import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.viz import _preprocess_latents_for_umap

FIT_MODES = ("per_frame", "final", "all")
# "separate" = fit PCA+UMAP on pred and target independently (matches dashboard)
# "union"    = fit on pred+target concatenated (old behavior)
UMAP_MERGE_MODE = "separate"


def _fit_pca_3d(all_pred: np.ndarray, all_gt: np.ndarray, random_state: int = 42):
    """Fit one PCA (3 components) on concatenated pred+gt from all frames."""
    x = np.concatenate([all_pred, all_gt], axis=0)
    pca = PCA(n_components=3, random_state=int(random_state))
    pca.fit(x)
    return pca


class _MovieUMAP:
    """Small wrapper that applies the same preprocessing at fit and transform."""

    def __init__(self, reducer, *, l2_normalize: bool, standardize: bool):
        self.reducer = reducer
        self.l2_normalize = bool(l2_normalize)
        self.standardize = bool(standardize)

    def _prep(self, x: np.ndarray) -> np.ndarray:
        return _preprocess_latents_for_umap(
            x,
            l2_normalize=self.l2_normalize,
            standardize=self.standardize,
        ).astype(np.float32, copy=False)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_prep = self._prep(x)
        try:
            import torch

            if self.reducer.__class__.__module__.startswith("torchdr"):
                z = self.reducer.transform(torch.from_numpy(x_prep))
                return z.cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)
        except Exception:
            pass
        return self.reducer.transform(x_prep)


def _fit_umap_3d(
    x: np.ndarray,
    n_neighbors: int = 50,
    min_dist: float = 0.2,
    metric: str = "euclidean",
    *,
    random_state: int = 42,
    init: str = "spectral",
    fit_max_tokens: int = 65536,
    l2_normalize: bool = False,
    standardize: bool = False,
):
    """Fit one 3D UMAP on *x* using the same preprocessing contract as dashboard."""
    x = _preprocess_latents_for_umap(
        x,
        l2_normalize=bool(l2_normalize),
        standardize=bool(standardize),
    ).astype(np.float32, copy=False)
    n_total = x.shape[0]

    if n_total > fit_max_tokens:
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(n_total, size=int(fit_max_tokens), replace=False)
        x_fit = x[idx]
        print(f"[session_to_movie] UMAP subsampled {n_total} → {fit_max_tokens} points for fitting")
    else:
        x_fit = x

    init_mode = str(init).lower()
    if init_mode not in ("spectral", "random"):
        init_mode = "spectral"

    try:
        from cuml.manifold import UMAP as CuMLUMAP
        reducer = CuMLUMAP(
            n_components=3,
            n_neighbors=int(n_neighbors),
            min_dist=min_dist,
            metric=metric,
            random_state=int(random_state),
            init=init_mode,
        )
        reducer.fit(x_fit)
        print("[session_to_movie] UMAP using cuML (GPU)")
        return _MovieUMAP(reducer, l2_normalize=l2_normalize, standardize=standardize)
    except Exception as e:
        print(f"[session_to_movie] cuML UMAP unavailable ({type(e).__name__}: {e})")

    try:
        import torch
        import torchdr

        if hasattr(torchdr, "UMAP"):
            reducer = torchdr.UMAP(
                n_components=3,
                n_neighbors=int(n_neighbors),
                min_dist=float(min_dist),
            )
            reducer.fit(torch.from_numpy(x_fit.astype(np.float32)))
            print("[session_to_movie] UMAP using torchdr")
            return _MovieUMAP(reducer, l2_normalize=l2_normalize, standardize=standardize)
    except Exception as e:
        print(f"[session_to_movie] torchdr UMAP unavailable ({type(e).__name__}: {e})")

    try:
        import umap
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=int(random_state),
            init=init_mode,
        )
        reducer.fit(x_fit)
        print("[session_to_movie] UMAP using umap-learn (CPU)")
        return _MovieUMAP(reducer, l2_normalize=l2_normalize, standardize=standardize)
    except Exception as e:
        print(f"[session_to_movie] UMAP unavailable ({type(e).__name__}: {e}); falling back to PCA")
        return None


def _render_frame(
    pred_map: np.ndarray,   # [C, H, W] or [H*W, C]
    gt_map: np.ndarray,
    pca,
    umap_reducer,
    epoch: int,
    out_path: str,
    dpi: int = 100,
    figsize: tuple = (12, 10),
):
    """Render a 2x2 frame: pred PCA | gt PCA / pred UMAP | gt UMAP."""
    from matplotlib import pyplot as plt
    from matplotlib.gridspec import GridSpec

    H, W = pred_map.shape[-2], pred_map.shape[-1]
    C = pred_map.shape[0] if pred_map.ndim == 3 else pred_map.shape[-1]

    # Flatten to [N, C]
    if pred_map.ndim == 3:
        pred_flat = pred_map.reshape(C, -1).T  # [H*W, C]
        gt_flat = gt_map.reshape(C, -1).T
    else:
        pred_flat = pred_map
        gt_flat = gt_map

    pred_pca = pca.transform(pred_flat)
    gt_pca = pca.transform(gt_flat)

    if umap_reducer is not None:
        pred_umap = umap_reducer.transform(pred_flat)
        gt_umap = umap_reducer.transform(gt_flat)
    else:
        pred_umap = pred_pca
        gt_umap = gt_pca

    fig = plt.figure(figsize=figsize)
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
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _render_frame_8panel(
    pred_pca: np.ndarray,   # [H*W, 3]
    gt_pca: np.ndarray,
    pred_umap: np.ndarray,
    gt_umap: np.ndarray,
    H: int,
    W: int,
    epoch: int,
    out_path: str,
    dpi: int = 100,
    figsize: tuple = (16, 10),
):
    """Render 4x2 frame with dashboard-like 3D RGB maps + 3D scatter panels."""
    from matplotlib import pyplot as plt
    from matplotlib.gridspec import GridSpec

    def _rgb_from_xyz(pts_3d):
        pts = np.asarray(pts_3d, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            padded = np.zeros((max(int(pts.shape[0]) if pts.ndim > 0 else 0, 1), 3), dtype=np.float32)
            if pts.ndim == 2 and pts.shape[0] > 0:
                padded[:, :pts.shape[1]] = pts[:, :min(pts.shape[1], 3)]
            pts = padded
        else:
            pts = pts[:, :3]
        fin = np.isfinite(pts).all(axis=1)
        if fin.any():
            lo = np.percentile(pts[fin], 1.0, axis=0)
            hi = np.percentile(pts[fin], 99.0, axis=0)
        else:
            lo = np.zeros(3, dtype=np.float32)
            hi = np.ones(3, dtype=np.float32)
        den = np.clip(hi - lo, 1e-8, None)
        clipped = np.clip((pts - lo) / den, 0.0, 1.0)
        clipped[~fin] = 0.0
        rgb_flat = clipped.astype(np.float32)
        rgb = rgb_flat.reshape(H, W, 3)[::-1, :, :]
        return rgb, rgb_flat

    def _style_3d(ax, title: str):
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("dim-1", fontsize=6, labelpad=-8)
        ax.set_ylabel("dim-2", fontsize=6, labelpad=-8)
        ax.set_zlabel("dim-3", fontsize=6, labelpad=-8)
        try:
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=18, azim=-58)
        except Exception:
            pass

    def _scatter3d(ax, pts_3d, rgb_flat, title: str):
        pts = np.asarray(pts_3d, dtype=np.float32)
        rgb = np.asarray(rgb_flat, dtype=np.float32)
        n = min(int(pts.shape[0]), int(rgb.shape[0]))
        pts = pts[:n]
        rgb = rgb[:n]
        if n > 65536:
            step = int(np.ceil(n / 65536.0))
            pts = pts[::step]
            rgb = rgb[::step]
        fin = np.isfinite(pts).all(axis=1)
        pts = pts[fin]
        rgb = rgb[fin]
        if pts.size > 0:
            ax.scatter(
                pts[:, 0],
                -pts[:, 1],
                pts[:, 2],
                s=1,
                c=np.clip(rgb, 0.0, 1.0),
                alpha=0.82,
                linewidths=0,
                depthshade=False,
            )
        _style_3d(ax, title)

    pred_pca_rgb, pred_pca_rgb_flat = _rgb_from_xyz(pred_pca)
    gt_pca_rgb, gt_pca_rgb_flat = _rgb_from_xyz(gt_pca)
    pred_umap_rgb, pred_umap_rgb_flat = _rgb_from_xyz(pred_umap)
    gt_umap_rgb, gt_umap_rgb_flat = _rgb_from_xyz(gt_umap)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig, wspace=0.3, hspace=0.35)

    # Row 1: PCA
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(pred_pca_rgb)
    ax.set_title("Pred PCA RGB", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1], projection="3d")
    _scatter3d(ax, pred_pca, pred_pca_rgb_flat, "Pred PCA 3D")

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(gt_pca_rgb)
    ax.set_title("GT PCA RGB", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3], projection="3d")
    _scatter3d(ax, gt_pca, gt_pca_rgb_flat, "GT PCA 3D")

    # Row 2: UMAP
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(pred_umap_rgb)
    ax.set_title("Pred UMAP RGB", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1], projection="3d")
    _scatter3d(ax, pred_umap, pred_umap_rgb_flat, "Pred UMAP 3D")

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(gt_umap_rgb)
    ax.set_title("GT UMAP RGB", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 3], projection="3d")
    _scatter3d(ax, gt_umap, gt_umap_rgb_flat, "GT UMAP 3D")

    fig.suptitle(f"Epoch {epoch}", fontsize=13, y=0.99)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.92, wspace=0.24, hspace=0.32)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_embedding_dump(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    pca,
    umap_reducer,
    epoch: int,
    out_path: str,
) -> None:
    """Save whole-map PCA/UMAP arrays for rendering or analysis downstream."""
    channels = pred_map.shape[0]
    pred_flat = pred_map.reshape(channels, -1).T
    gt_flat = gt_map.reshape(channels, -1).T
    pred_pca = pca.transform(pred_flat).astype(np.float32)
    gt_pca = pca.transform(gt_flat).astype(np.float32)
    if umap_reducer is None:
        pred_umap = pred_pca
        gt_umap = gt_pca
    else:
        pred_umap = umap_reducer.transform(pred_flat).astype(np.float32)
        gt_umap = umap_reducer.transform(gt_flat).astype(np.float32)
    np.savez_compressed(
        out_path,
        epoch=np.asarray(epoch, dtype=np.int64),
        spatial_shape=np.asarray(pred_map.shape[-2:], dtype=np.int64),
        pred_pca=pred_pca,
        gt_pca=gt_pca,
        pred_umap=pred_umap,
        gt_umap=gt_umap,
    )


def collect_all_frames(
    movie_dir: str,
) -> tuple[
    list[int],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    """Load all frames, return aggregated pred+gt+ctx arrays. ctx may be None."""
    import glob
    pattern = os.path.join(movie_dir, "frame_*.pt")
    if not glob.glob(pattern):
        pattern = os.path.join(movie_dir, "epoch_*.pt")
    paths = sorted(glob.glob(pattern))
    epochs = []
    all_pred_chunks, all_gt_chunks, all_ctx_chunks = [], [], []
    has_ctx = False
    for p in paths:
        frame = torch.load(p, map_location="cpu", weights_only=False)
        ep = int(frame["epoch"]) if "epoch" in frame else int(os.path.basename(p).replace("epoch_", "").replace("frame_", "").replace(".pt", ""))
        pm = frame["pred_map"][0].numpy()
        gm = frame["gt_map"][0].numpy()
        C = pm.shape[0]
        all_pred_chunks.append(pm.reshape(C, -1).T)
        all_gt_chunks.append(gm.reshape(C, -1).T)
        cm = frame.get("context_map")
        if cm is not None:
            has_ctx = True
            all_ctx_chunks.append(cm[0].numpy().reshape(C, -1).T)
        epochs.append(ep)
    all_pred = np.concatenate(all_pred_chunks, axis=0)
    all_gt = np.concatenate(all_gt_chunks, axis=0)
    all_ctx = np.concatenate(all_ctx_chunks, axis=0) if has_ctx else None
    return epochs, paths, all_pred, all_gt, all_ctx, all_pred_chunks, all_gt_chunks, all_ctx_chunks


def main():
    parser = argparse.ArgumentParser(description="Convert session movie frames to PNGs")
    parser.add_argument("session_dir", help="Path to session directory containing movie_frames/")
    parser.add_argument("--config", default=None, help="Path to movie config JSON (overrides defaults)")
    parser.add_argument("--out-dir", default=None, help="Output directory for PNG frames")
    parser.add_argument("--dump-only", action="store_true", help="Save PCA/UMAP NPZ dumps without rendering PNGs")
    parser.add_argument("--make-mp4", action="store_true", help="Generate MP4 with ffmpeg")
    parser.add_argument("--force", action="store_true", help="Remove existing movie PNG/MP4/cache outputs before rendering")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second for MP4")
    parser.add_argument("--seed", type=int, default=42, help="UMAP random seed")
    parser.add_argument("--n-neighbors", type=int, default=50, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.2, help="UMAP min_dist")
    parser.add_argument("--umap-metric", default="euclidean", help="UMAP distance metric")
    parser.add_argument("--umap-init", default=None, help="UMAP init mode; defaults to session train.umap.init or spectral")
    parser.add_argument("--umap-standardize", action="store_true", help="Force UMAP channel standardization")
    parser.add_argument("--umap-l2-normalize", action="store_true", help="Force UMAP row L2 normalization")
    parser.add_argument("--fit-max-tokens", type=int, default=None, help="Maximum tokens used to fit UMAP")
    parser.add_argument(
        "--pca-fit-mode",
        choices=FIT_MODES,
        default="per_frame",
        help="PCA fit strategy: per_frame, final, or all",
    )
    parser.add_argument(
        "--umap-fit-mode",
        choices=FIT_MODES,
        default="per_frame",
        help="UMAP fit strategy: per_frame, final, or all",
    )
    parser.add_argument("--branches", default="pred,gt", help="Branches to render: pred,gt,ctx (comma-separated)")
    parser.add_argument("--dpi", type=int, default=100, help="PNG DPI")
    parser.add_argument("--figsize", nargs=2, type=float, default=[12, 10], help="Figure size in inches")
    args = parser.parse_args()

    # Load config file if provided (CLI flags override config values)
    if args.config:
        if not os.path.exists(args.config):
            print(f"[session_to_movie] config not found: {args.config}")
            sys.exit(1)
        with open(args.config) as f:
            cfg = json.load(f)
        # Config sets defaults; CLI args override
        for key in ("seed", "n_neighbors", "min_dist", "dpi", "fps", "umap_metric"):
            cli_val = getattr(args, key.replace("-", "_"))
            cfg_key = f"umap_{key}" if key not in ("fps", "dpi", "seed") else key
            cfg_val = cfg.get(cfg_key)
            default_map = {"seed": 42, "n_neighbors": 50, "min_dist": 0.2, "dpi": 100, "fps": 5, "umap_metric": "euclidean"}
            if cli_val == default_map.get(key) and cfg_val is not None:
                setattr(args, key.replace("-", "_"), cfg_val)
        if args.out_dir is None and "output_dir" in cfg:
            args.out_dir = cfg["output_dir"]
        if args.fps == 5 and "fps" in cfg:
            args.fps = cfg["fps"]
        if "figsize" in cfg:
            args.figsize = cfg["figsize"]
    if args.dump_only and args.make_mp4:
        parser.error("--dump-only cannot be combined with --make-mp4")

    session_dir = os.path.abspath(args.session_dir)
    movie_dir = os.path.join(session_dir, "movie_frames")
    if not os.path.isdir(movie_dir):
        print(f"[session_to_movie] movie_frames/ not found in {session_dir}")
        sys.exit(1)

    session_name = os.path.basename(session_dir.rstrip("/"))
    out_root = args.out_dir or os.path.join(ROOT, "results", "movie_pngs")
    out_dir = os.path.join(out_root, session_name)
    save_pngs = not args.dump_only and not args.make_mp4
    print(f"[session_to_movie] save_pngs={save_pngs}  out_dir={out_dir}")
    if not args.dump_only:
        os.makedirs(out_dir, exist_ok=True)
    dump_dir = os.path.join(session_dir, "movie_embedding_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    if args.force:
        import glob

        removed = 0
        cleanup_patterns = (
            os.path.join(out_dir, "frame_*.png"),
            os.path.join(out_dir, f"{session_name}.mp4"),
            os.path.join(out_dir, "movie_pca_embeddings.npz"),
            os.path.join(out_dir, "movie_umap_embeddings.npz"),
            os.path.join(out_dir, "movie_umap_embeddings_*.npz"),
        )
        for pattern in cleanup_patterns:
            for stale_path in glob.glob(pattern):
                try:
                    os.remove(stale_path)
                    removed += 1
                except FileNotFoundError:
                    pass
        if removed > 0:
            print(f"[session_to_movie] force removed {removed} stale movie output(s)")

    print(f"[session_to_movie] loading frames from {movie_dir}")
    epochs, frame_paths, all_pred, all_gt, all_ctx, all_pred_chunks, all_gt_chunks, all_ctx_chunks = collect_all_frames(movie_dir)
    branches = [b.strip() for b in args.branches.split(",")]
    has_ctx = all_ctx is not None and "ctx" in branches
    print(f"[session_to_movie] loaded {len(epochs)} frames")

    # ── Read UMAP params: session config → movie config → defaults ──
    session_umap_cfg = {}
    session_cfg_path = os.path.join(session_dir, "config_used.json")
    if os.path.exists(session_cfg_path):
        with open(session_cfg_path) as f:
            session_umap_cfg = json.load(f).get("train", {}).get("umap", {})
    # Fallback: movie config file (if loaded)
    movie_umap_cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            movie_umap_cfg = json.load(f)
    _umap_n = int(session_umap_cfg.get("n_neighbors") or movie_umap_cfg.get("umap_n_neighbors") or 15)
    _umap_d = float(session_umap_cfg.get("min_dist") or movie_umap_cfg.get("umap_min_dist") or 0.05)
    _umap_m = str(session_umap_cfg.get("metric") or movie_umap_cfg.get("umap_metric") or "cosine")
    _umap_seed = int(session_umap_cfg.get("random_state") or movie_umap_cfg.get("umap_random_state") or args.seed)
    _umap_init = str(args.umap_init or session_umap_cfg.get("init") or movie_umap_cfg.get("umap_init") or "spectral")
    _umap_standardize = bool(args.umap_standardize or session_umap_cfg.get("standardize", movie_umap_cfg.get("umap_standardize", False)))
    _umap_l2 = bool(args.umap_l2_normalize or session_umap_cfg.get("l2_normalize", movie_umap_cfg.get("umap_l2_normalize", False)))
    _umap_fit_max = int(args.fit_max_tokens or session_umap_cfg.get("fit_max_tokens") or movie_umap_cfg.get("umap_fit_max_tokens") or 65536)

    # Cache key includes UMAP params so changing them invalidates cache
    import hashlib
    _cache_key = hashlib.md5(
        (
            f"{_umap_n}_{_umap_d}_{_umap_m}_{_umap_seed}_{_umap_init}_"
            f"std{int(_umap_standardize)}_l2{int(_umap_l2)}_max{_umap_fit_max}_"
            f"{args.pca_fit_mode}_{args.umap_fit_mode}_{UMAP_MERGE_MODE}"
        ).encode()
    ).hexdigest()[:8]
    cache_dir = out_dir
    os.makedirs(cache_dir, exist_ok=True)
    pca_cache = os.path.join(cache_dir, "movie_pca_embeddings.npz")
    umap_cache = os.path.join(cache_dir, f"movie_umap_embeddings_{_cache_key}.npz")

    def _cache_has_3d_embeddings() -> bool:
        if not (os.path.exists(pca_cache) and os.path.exists(umap_cache)):
            return False
        try:
            with np.load(pca_cache) as pca_data, np.load(umap_cache) as umap_data:
                arrays = (
                    pca_data["pred_pca"],
                    pca_data["gt_pca"],
                    umap_data["pred_umap"],
                    umap_data["gt_umap"],
                )
                return all(arr.ndim == 3 and arr.shape[-1] >= 3 for arr in arrays)
        except Exception:
            return False

    if (os.path.exists(pca_cache) or os.path.exists(umap_cache)) and not _cache_has_3d_embeddings():
        for stale in (pca_cache, umap_cache):
            if os.path.exists(stale):
                os.remove(stale)
        print("[session_to_movie] removed stale non-3D PCA/UMAP cache")

    # Step 1: fit or load cached PCA/UMAP
    print(f"[session_to_movie] PCA={args.pca_fit_mode}  UMAP={args.umap_fit_mode}  merge={UMAP_MERGE_MODE}")
    if _cache_has_3d_embeddings():
        print("[session_to_movie] loading cached PCA/UMAP, skipping fit...")
    else:
        # Detect UMAP backend
        _umap_backend = "CPU (umap-learn)"
        try:
            from cuml.manifold import UMAP  # noqa: F401
            _umap_backend = "GPU (cuml)"
        except ImportError:
            pass
        print(f"[session_to_movie] UMAP backend: {_umap_backend}")

        # ── Build global PCA (for "final" / "all" modes) ──
        if args.pca_fit_mode == "final":
            _fit_data = np.concatenate([all_pred_chunks[-1], all_gt_chunks[-1]] + ([all_ctx_chunks[-1]] if has_ctx else []), axis=0)
            print(f"[session_to_movie] fitting PCA on final frame (n_samples={_fit_data.shape[0]})...")
            pca_global = PCA(n_components=3, random_state=_umap_seed).fit(_fit_data)
        elif args.pca_fit_mode == "all":
            _fit_data = np.concatenate([all_pred, all_gt] + ([all_ctx] if has_ctx else []), axis=0)
            print(f"[session_to_movie] fitting PCA on all frames (n_samples={_fit_data.shape[0]})...")
            pca_global = PCA(n_components=3, random_state=_umap_seed).fit(_fit_data)
        else:
            pca_global = None  # per_frame — fit inside loop

        # ── Build global UMAP (for "final" / "all" modes) ──
        umap_global = None
        umap_global_pred = None
        umap_global_gt = None
        if args.umap_fit_mode == "final":
            if UMAP_MERGE_MODE == "separate":
                print(f"[session_to_movie] fitting UMAP on final frame (separate pred + target)...")
                umap_global_pred = _fit_umap_3d(
                    all_pred_chunks[-1],
                    n_neighbors=_umap_n,
                    min_dist=_umap_d,
                    metric=_umap_m,
                    random_state=_umap_seed,
                    init=_umap_init,
                    fit_max_tokens=_umap_fit_max,
                    l2_normalize=_umap_l2,
                    standardize=_umap_standardize,
                )
                umap_global_gt = _fit_umap_3d(
                    all_gt_chunks[-1],
                    n_neighbors=_umap_n,
                    min_dist=_umap_d,
                    metric=_umap_m,
                    random_state=_umap_seed,
                    init=_umap_init,
                    fit_max_tokens=_umap_fit_max,
                    l2_normalize=_umap_l2,
                    standardize=_umap_standardize,
                )
            else:
                _umap_data = np.concatenate([all_pred_chunks[-1], all_gt_chunks[-1]] + ([all_ctx_chunks[-1]] if has_ctx else []), axis=0)
                print(f"[session_to_movie] fitting UMAP on final frame (n_samples={_umap_data.shape[0]})...")
                umap_global = _fit_umap_3d(
                    _umap_data,
                    n_neighbors=_umap_n,
                    min_dist=_umap_d,
                    metric=_umap_m,
                    random_state=_umap_seed,
                    init=_umap_init,
                    fit_max_tokens=_umap_fit_max,
                    l2_normalize=_umap_l2,
                    standardize=_umap_standardize,
                )
        elif args.umap_fit_mode == "all":
            if UMAP_MERGE_MODE == "separate":
                print(f"[session_to_movie] fitting UMAP on all frames (separate pred + target)...")
                umap_global_pred = _fit_umap_3d(
                    all_pred,
                    n_neighbors=_umap_n,
                    min_dist=_umap_d,
                    metric=_umap_m,
                    random_state=_umap_seed,
                    init=_umap_init,
                    fit_max_tokens=_umap_fit_max,
                    l2_normalize=_umap_l2,
                    standardize=_umap_standardize,
                )
                umap_global_gt = _fit_umap_3d(
                    all_gt,
                    n_neighbors=_umap_n,
                    min_dist=_umap_d,
                    metric=_umap_m,
                    random_state=_umap_seed,
                    init=_umap_init,
                    fit_max_tokens=_umap_fit_max,
                    l2_normalize=_umap_l2,
                    standardize=_umap_standardize,
                )
            else:
                _umap_data = np.concatenate([all_pred, all_gt] + ([all_ctx] if has_ctx else []), axis=0)
                print(f"[session_to_movie] fitting UMAP on all frames (n_samples={_umap_data.shape[0]})...")
                umap_global = _fit_umap_3d(
                    _umap_data,
                    n_neighbors=_umap_n,
                    min_dist=_umap_d,
                    metric=_umap_m,
                    random_state=_umap_seed,
                    init=_umap_init,
                    fit_max_tokens=_umap_fit_max,
                    l2_normalize=_umap_l2,
                    standardize=_umap_standardize,
                )

    # Load loss weights for annotation
    lw_path = os.path.join(session_dir, "loss_weights.json")
    loss_weights = {}
    if os.path.exists(lw_path):
        with open(lw_path) as f:
            loss_weights = json.load(f)

    if not _cache_has_3d_embeddings():
        mode_label = f"PCA={args.pca_fit_mode} UMAP={args.umap_fit_mode} merge={UMAP_MERGE_MODE}"
        print(f"[session_to_movie] precomputing frames [{mode_label}]...")
        frame0 = torch.load(frame_paths[0], map_location="cpu", weights_only=False)
        H0, W0 = int(frame0["pred_map"].shape[-2]), int(frame0["pred_map"].shape[-1])
        all_pred_pca, all_gt_pca = [], []
        all_pred_umap, all_gt_umap = [], []
        for i, (ep, fp) in enumerate(zip(epochs, frame_paths)):
            frame = torch.load(fp, map_location="cpu", weights_only=False)
            pm = frame["pred_map"][0].numpy()
            gm = frame["gt_map"][0].numpy()
            C = pm.shape[0]
            pred_flat = pm.reshape(C, -1).T
            gt_flat = gm.reshape(C, -1).T

            # PCA
            if args.pca_fit_mode == "per_frame":
                if UMAP_MERGE_MODE == "separate":
                    pca_pred = PCA(n_components=3, random_state=_umap_seed).fit(pred_flat)
                    pca_gt = PCA(n_components=3, random_state=_umap_seed).fit(gt_flat)
                else:
                    pca_frame = PCA(n_components=3, random_state=_umap_seed).fit(np.concatenate([pred_flat, gt_flat], axis=0))
                    pca_pred = pca_gt = pca_frame
                pred_pca = pca_pred.transform(pred_flat).astype(np.float32)
                gt_pca = pca_gt.transform(gt_flat).astype(np.float32)
            else:
                pred_pca = pca_global.transform(pred_flat).astype(np.float32)
                gt_pca = pca_global.transform(gt_flat).astype(np.float32)

            # UMAP
            if args.umap_fit_mode == "per_frame":
                if UMAP_MERGE_MODE == "separate":
                    umap_pred = _fit_umap_3d(
                        pred_flat,
                        n_neighbors=_umap_n,
                        min_dist=_umap_d,
                        metric=_umap_m,
                        random_state=_umap_seed,
                        init=_umap_init,
                        fit_max_tokens=_umap_fit_max,
                        l2_normalize=_umap_l2,
                        standardize=_umap_standardize,
                    )
                    umap_gt = _fit_umap_3d(
                        gt_flat,
                        n_neighbors=_umap_n,
                        min_dist=_umap_d,
                        metric=_umap_m,
                        random_state=_umap_seed,
                        init=_umap_init,
                        fit_max_tokens=_umap_fit_max,
                        l2_normalize=_umap_l2,
                        standardize=_umap_standardize,
                    )
                else:
                    combined = np.concatenate([pred_flat, gt_flat], axis=0)
                    frame_umap = _fit_umap_3d(
                        combined,
                        n_neighbors=_umap_n,
                        min_dist=_umap_d,
                        metric=_umap_m,
                        random_state=_umap_seed,
                        init=_umap_init,
                        fit_max_tokens=_umap_fit_max,
                        l2_normalize=_umap_l2,
                        standardize=_umap_standardize,
                    )
                    umap_pred = umap_gt = frame_umap
            else:
                if UMAP_MERGE_MODE == "separate":
                    umap_pred = umap_global_pred
                    umap_gt = umap_global_gt
                else:
                    umap_pred = umap_gt = umap_global
            if umap_pred is not None and umap_gt is not None:
                pred_umap = umap_pred.transform(pred_flat).astype(np.float32)
                gt_umap = umap_gt.transform(gt_flat).astype(np.float32)
            else:
                pred_umap, gt_umap = pred_pca, gt_pca

            all_pred_pca.append(pred_pca)
            all_gt_pca.append(gt_pca)
            all_pred_umap.append(pred_umap)
            all_gt_umap.append(gt_umap)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[session_to_movie] precomputed {i + 1}/{len(epochs)} frames")
        np.savez_compressed(pca_cache, pred_pca=np.stack(all_pred_pca, axis=0), gt_pca=np.stack(all_gt_pca, axis=0), spatial_shape=np.array([H0, W0], dtype=np.int64), epochs=np.array(epochs, dtype=np.int64))
        np.savez_compressed(umap_cache, pred_umap=np.stack(all_pred_umap, axis=0), gt_umap=np.stack(all_gt_umap, axis=0))
        print(f"[session_to_movie] PCA/UMAP cache saved to {cache_dir}")

    # Step 2: render transient PNGs for ffmpeg only; do not leave PNGs in outputs.
    cached_pca = np.load(pca_cache)
    cached_umap = np.load(umap_cache)
    pred_pca_stack = cached_pca["pred_pca"]
    gt_pca_stack = cached_pca["gt_pca"]
    pred_umap_stack = cached_umap["pred_umap"]
    gt_umap_stack = cached_umap["gt_umap"]
    H0, W0 = int(cached_pca["spatial_shape"][0]), int(cached_pca["spatial_shape"][1])

    frame_tmp = None if args.dump_only else tempfile.TemporaryDirectory(prefix="jepa_movie_frames_")
    frame_dir = frame_tmp.name if frame_tmp is not None else None
    try:
        for i, (ep, fp) in enumerate(zip(epochs, frame_paths)):
            pred_pca = pred_pca_stack[i]  # [H*W, 3]
            gt_pca = gt_pca_stack[i]
            pred_umap = pred_umap_stack[i]
            gt_umap = gt_umap_stack[i]
            if args.dump_only:
                dumpp = os.path.join(dump_dir, f"epoch_{ep:04d}.npz")
                np.savez_compressed(dumpp, epoch=np.asarray(ep, dtype=np.int64), spatial_shape=np.asarray([H0, W0], dtype=np.int64), pred_pca=pred_pca, gt_pca=gt_pca, pred_umap=pred_umap, gt_umap=gt_umap)
                continue
            out_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
            _render_frame_8panel(pred_pca, gt_pca, pred_umap, gt_umap, H0, W0, ep, out_path, dpi=args.dpi, figsize=tuple(args.figsize))
            if (i + 1) % 20 == 0 or i == 0:
                print(f"[session_to_movie] rendered {i + 1}/{len(epochs)} transient frames")

        print(f"[session_to_movie] whole-map embedding dumps saved to {dump_dir}")
        if args.dump_only:
            return
        if not args.make_mp4:
            import shutil
            import glob as _glob
            for src in sorted(_glob.glob(os.path.join(frame_dir, "frame_*.png"))):
                dst = os.path.join(out_dir, os.path.basename(src))
                shutil.copy2(src, dst)
            n_saved = len(os.listdir(out_dir))
            frame_tmp.cleanup()
            frame_tmp = None
            print(f"[session_to_movie] saved {n_saved} PNG frames to {out_dir}")
            return

        mp4_path = os.path.join(out_dir, f"{session_name}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", os.path.join(frame_dir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            mp4_path,
        ]
        import subprocess
        subprocess.run(cmd, check=True)
        print(f"[session_to_movie] mp4 saved to {mp4_path}")
    finally:
        if frame_tmp is not None:
            frame_tmp.cleanup()


if __name__ == "__main__":
    main()
