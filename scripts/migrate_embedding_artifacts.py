#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _load_shape(results_dir: Path, prefix: str, n: int) -> tuple[int, int]:
    shape_path = results_dir / f"{prefix}_spatial_shape.npy" if prefix else results_dir / "spatial_shape.npy"
    if shape_path.exists():
        arr = np.asarray(np.load(shape_path), dtype=np.int64).reshape(-1)
        if arr.size >= 2 and int(arr[0]) > 0 and int(arr[1]) > 0 and int(arr[0]) * int(arr[1]) == n:
            return int(arr[0]), int(arr[1])
    raise ValueError(f"missing compatible spatial_shape for prefix={prefix!r}, n={n}; refusing guessed reshape")


def _save(path: Path, arr: np.ndarray, *, dry_run: bool) -> None:
    print(f"{'would_write' if dry_run else 'wrote'} {path} shape={tuple(arr.shape)}")
    if not dry_run:
        np.save(path, arr.astype(np.float32, copy=False))


def _migrate_latents(results_dir: Path, prefix: str, *, dry_run: bool) -> None:
    path = results_dir / f"{prefix}_latent_vectors_full.npy" if prefix else results_dir / "latent_vectors_full.npy"
    if not path.exists():
        return
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 3:
        print(f"ok {path} shape={tuple(arr.shape)}")
        return
    if arr.ndim != 2:
        print(f"skip {path} unsupported_shape={tuple(arr.shape)}")
        return
    n, c = int(arr.shape[0]), int(arr.shape[1])
    h, w = _load_shape(results_dir, prefix, n)
    _save(path, arr.reshape(h, w, c).transpose(2, 0, 1), dry_run=dry_run)


def _migrate_xyz(results_dir: Path, prefix: str, kind: str, *, dry_run: bool) -> None:
    xyz_path = results_dir / f"{prefix}_{kind}_xyz.npy" if prefix else results_dir / f"{kind}_xyz.npy"
    component_paths = [
        results_dir / f"{prefix}_{kind}_{axis}.npy" if prefix else results_dir / f"{kind}_{axis}.npy"
        for axis in ("x", "y", "z")
    ]

    xyz = None
    if xyz_path.exists():
        arr = np.asarray(np.load(xyz_path), dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 3:
            xyz = arr
            print(f"ok {xyz_path} shape={tuple(arr.shape)}")
        elif arr.ndim == 2 and arr.shape[1] >= 3:
            n = int(arr.shape[0])
            h, w = _load_shape(results_dir, prefix, n)
            xyz = arr[:, :3].reshape(h, w, 3).transpose(2, 0, 1)
            _save(xyz_path, xyz, dry_run=dry_run)
        else:
            print(f"skip {xyz_path} unsupported_shape={tuple(arr.shape)}")

    if xyz is None and all(p.exists() for p in component_paths):
        comps = [np.asarray(np.load(p), dtype=np.float32) for p in component_paths]
        flat = [c.reshape(-1) for c in comps]
        sizes = [int(v.size) for v in flat]
        if len(set(sizes)) != 1:
            raise ValueError(f"component size mismatch for prefix={prefix!r} kind={kind}: {sizes}")
        n = sizes[0]
        h, w = _load_shape(results_dir, prefix, n)
        xyz = np.stack([v.reshape(h, w) for v in flat], axis=0).astype(np.float32)
        _save(xyz_path, xyz, dry_run=dry_run)

    if xyz is not None:
        for i, path in enumerate(component_paths):
            if not path.exists():
                continue
            comp = np.asarray(np.load(path), dtype=np.float32)
            if comp.ndim == 2 and comp.shape == tuple(xyz.shape[1:]):
                print(f"ok {path} shape={tuple(comp.shape)}")
                continue
            _save(path, xyz[i], dry_run=dry_run)


def migrate_results(results_dir: Path, *, dry_run: bool = False) -> None:
    prefixes = ["", "predict", "target", "context"]
    for prefix in prefixes:
        _migrate_latents(results_dir, prefix, dry_run=dry_run)
        _migrate_xyz(results_dir, prefix, "pca", dry_run=dry_run)
        _migrate_xyz(results_dir, prefix, "umap", dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate flat embedding artifacts to image-shaped maps.")
    parser.add_argument("path", help="Session directory or its results/ directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = Path(args.path)
    results_dir = path if path.name == "results" else path / "results"
    if not results_dir.is_dir():
        raise FileNotFoundError(f"missing results directory: {results_dir}")
    migrate_results(results_dir, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
