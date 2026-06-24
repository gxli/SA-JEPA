from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.npy import _safe_load_npy, normalize01


class JEPA3DCropDataset(Dataset):
    """3D dataset that consumes precomputed CDD cache entries.

    Each cache entry is (S, D, H, W) — S CDD scale channels over a cube.
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
