#!/usr/bin/env python3
"""Generate simple TorchDR UMAP movie frames from normal saved session results."""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


BRANCHES = ("context", "predict", "masked_predict", "target")


def _load_latent_map(session_dir: str, branch: str) -> np.ndarray | None:
    results_dir = os.path.join(session_dir, "results")
    path = os.path.join(results_dir, f"{branch}_latent_vectors_full.npy")
    if os.path.exists(path):
        arr = np.load(path)
        return arr.astype(np.float32, copy=False)

    inf_path = os.path.join(session_dir, "inference_outputs.pt")
    if not os.path.exists(inf_path):
        return None
    key = {
        "context": "context_map",
        "predict": "pred_map",
        "masked_predict": "masked_pred_map",
        "target": "gt_map",
    }[branch]
    outputs = torch.load(inf_path, map_location="cpu", weights_only=False)
    value = outputs.get(key)
    if value is None and branch == "masked_predict":
        value = outputs.get("pred_map")
    if value is None:
        return None
    tensor = torch.as_tensor(value)
    if tensor.dim() == 4:
        tensor = tensor[0]
    elif tensor.dim() == 5:
        tensor = tensor[0, :, tensor.shape[2] // 2]
    if tensor.dim() != 3:
        raise RuntimeError(f"{branch} latent map must be CxHxW, got {tuple(tensor.shape)}")
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _valid_rows(latent_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, h, w = latent_map.shape
    rows = np.transpose(latent_map, (1, 2, 0)).reshape(h * w, c)
    valid = np.isfinite(rows).all(axis=1)
    return rows, valid


def _shared_torchdr_umap_points(
    rows_by_branch: dict[str, np.ndarray],
    valid_by_branch: dict[str, np.ndarray],
    *,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    fit_max_tokens: int,
    seed: int,
    device: str,
) -> dict[str, dict[str, np.ndarray]]:
    fit_rows = []
    labels = []
    indices = []
    rng = np.random.default_rng(seed)
    for branch, rows in rows_by_branch.items():
        valid_idx = np.flatnonzero(valid_by_branch[branch])
        if valid_idx.size == 0:
            continue
        if valid_idx.size > fit_max_tokens:
            valid_idx = rng.choice(valid_idx, size=fit_max_tokens, replace=False)
        fit_rows.append(rows[valid_idx])
        labels.extend([branch] * int(valid_idx.size))
        indices.extend(int(i) for i in valid_idx)
    if not fit_rows:
        raise RuntimeError("No finite latent rows available for TorchDR UMAP.")

    import torchdr

    fit_x = np.concatenate(fit_rows, axis=0).astype(np.float32, copy=False)
    reducer = torchdr.UMAP(
        n_components=int(n_components),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        backend=None,
    )
    with torch.no_grad():
        emb = reducer.fit_transform(torch.from_numpy(fit_x).to(device))
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    emb = np.asarray(emb, dtype=np.float32)

    out: dict[str, dict[str, np.ndarray]] = {}
    labels_arr = np.asarray(labels)
    indices_arr = np.asarray(indices, dtype=np.int64)
    for branch in rows_by_branch:
        mask = labels_arr == branch
        out[branch] = {
            "indices": indices_arr[mask],
            "embedding": emb[mask],
        }
    return out


def _render_frame(points: np.ndarray, title: str, path: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    plt.figure(figsize=(6, 6))
    color = points[:, 2] if points.shape[1] >= 3 else points[:, 0]
    plt.scatter(points[:, 0], points[:, 1], c=color, s=4, cmap="viridis", linewidths=0, alpha=0.8)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout(pad=0.2)
    plt.savefig(path, dpi=160)
    plt.close()


def build_movie_frames(
    session_dir: str,
    branches: Iterable[str],
    *,
    n_neighbors: int,
    min_dist: float,
    fit_max_tokens: int,
    seed: int,
    device: str,
) -> str:
    session_dir = os.path.abspath(session_dir)
    out_dir = os.path.join(session_dir, "results", "movie")
    os.makedirs(out_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(out_dir, "frame_*.png")) + glob.glob(os.path.join(out_dir, "*_torchdr_umap_points.npz")):
        os.remove(stale)

    latent_by_branch = {}
    rows_by_branch = {}
    valid_by_branch = {}
    shapes = {}
    for branch in branches:
        latent = _load_latent_map(session_dir, branch)
        if latent is None:
            continue
        rows, valid = _valid_rows(latent)
        latent_by_branch[branch] = latent
        rows_by_branch[branch] = rows
        valid_by_branch[branch] = valid
        shapes[branch] = latent.shape[-2:]

    if not rows_by_branch:
        raise RuntimeError(f"No branch latent maps found in {session_dir}")

    print("[session_to_movie] fitting TorchDR UMAP")
    embeddings = _shared_torchdr_umap_points(
        rows_by_branch,
        valid_by_branch,
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        fit_max_tokens=fit_max_tokens,
        seed=seed,
        device=device,
    )

    all_points = np.concatenate([item["embedding"] for item in embeddings.values() if item["embedding"].size], axis=0)
    x_pad = max(1e-6, 0.05 * float(np.nanmax(all_points[:, 0]) - np.nanmin(all_points[:, 0])))
    y_pad = max(1e-6, 0.05 * float(np.nanmax(all_points[:, 1]) - np.nanmin(all_points[:, 1])))
    xlim = (float(np.nanmin(all_points[:, 0]) - x_pad), float(np.nanmax(all_points[:, 0]) + x_pad))
    ylim = (float(np.nanmin(all_points[:, 1]) - y_pad), float(np.nanmax(all_points[:, 1]) + y_pad))

    for frame_idx, branch in enumerate(rows_by_branch):
        payload = embeddings[branch]
        if payload["embedding"].size == 0:
            continue
        h, w = shapes[branch]
        np.savez_compressed(
            os.path.join(out_dir, f"{branch}_torchdr_umap_points.npz"),
            indices=payload["indices"],
            embedding=payload["embedding"],
            spatial_shape=np.asarray([h, w], dtype=np.int64),
        )
        frame_path = os.path.join(out_dir, f"frame_{frame_idx:04d}_{branch}.png")
        _render_frame(payload["embedding"], f"{branch} TorchDR UMAP", frame_path, xlim, ylim)
        print(f"[session_to_movie] saved {frame_path}")

    print(f"[session_to_movie] movie artifacts saved to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple TorchDR UMAP movie frames from a saved session.")
    parser.add_argument("session_dir")
    parser.add_argument("--branches", nargs="+", default=list(BRANCHES), choices=list(BRANCHES))
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--min-dist", type=float, default=0.2)
    parser.add_argument("--fit-max-tokens", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    build_movie_frames(
        args.session_dir,
        args.branches,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        fit_max_tokens=args.fit_max_tokens,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
