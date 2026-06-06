from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class JEPA3DCropDataset(Dataset):
    def __init__(
        self,
        data_root: str = "data",
        npy_pattern: str = "*.npy",
        num_samples: int = 1000,
        crop_size: int = 64,
        crop_depth: int | None = None,
        depth_axis: int = 0,
        random_axis: bool = False,
        normalize: bool = True,
        crop_strategy: str = "random",
    ):
        self.npy_files = sorted(glob.glob(os.path.join(data_root, npy_pattern)))
        if not self.npy_files:
            raise FileNotFoundError(f"No .npy files found in {data_root}/{npy_pattern}")

        self.num_samples = int(num_samples)
        self.crop_size = int(crop_size)
        self.crop_depth = int(crop_depth) if crop_depth is not None else int(crop_size)
        self.depth_axis = int(depth_axis) % 3
        self.random_axis = bool(random_axis)
        self.normalize = bool(normalize)
        self.crop_strategy = str(crop_strategy).lower()
        if self.crop_strategy not in ("random", "center", "mixed"):
            raise ValueError("crop_strategy must be one of: random, center, mixed")

    def __len__(self):
        return self.num_samples

    @staticmethod
    def _normalize01(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        lo = float(x.min())
        hi = float(x.max())
        if hi > lo + 1e-20:
            return (x - lo) / (hi - lo)
        return np.zeros_like(x, dtype=np.float32)

    def _choose_depth_axis(self) -> int:
        if self.random_axis:
            return int(np.random.randint(0, 3))
        return self.depth_axis

    def _orient_depth_first(self, arr: np.ndarray) -> np.ndarray:
        axis = self._choose_depth_axis()
        if axis == 0:
            return arr
        return np.moveaxis(arr, axis, 0)

    def _check_shape(self, arr: np.ndarray) -> tuple[int, int, int]:
        cs = self.crop_size
        cd = self.crop_depth
        d, h, w = arr.shape
        if d < cd or h < cs or w < cs:
            raise ValueError(f"crop_depth={cd}, crop_size={cs} exceeds volume shape={arr.shape}")
        return d, h, w

    def _random_crop(self, arr: np.ndarray) -> np.ndarray:
        cs = self.crop_size
        cd = self.crop_depth
        d, h, w = self._check_shape(arr)

        # Valid box-origin ranges are [0, dim - crop_dim]. The target slab sits
        # at this crop's center, so random z0 randomly places the target slab
        # center while preserving the encoder context margin.
        z0 = np.random.randint(0, d - cd + 1)
        y0 = np.random.randint(0, h - cs + 1)
        x0 = np.random.randint(0, w - cs + 1)
        return arr[z0 : z0 + cd, y0 : y0 + cs, x0 : x0 + cs]

    def _center_crop(self, arr: np.ndarray) -> np.ndarray:
        cs = self.crop_size
        cd = self.crop_depth
        d, h, w = self._check_shape(arr)
        z0 = max(0, (d - cd) // 2)
        y0 = max(0, (h - cs) // 2)
        x0 = max(0, (w - cs) // 2)
        return arr[z0 : z0 + cd, y0 : y0 + cs, x0 : x0 + cs]

    def _crop(self, arr: np.ndarray) -> np.ndarray:
        if self.crop_strategy == "center":
            return self._center_crop(arr)
        if self.crop_strategy == "random":
            return self._random_crop(arr)
        if np.random.rand() < 0.5:
            return self._center_crop(arr)
        return self._random_crop(arr)

    def __getitem__(self, idx):
        path = self.npy_files[idx % len(self.npy_files)]
        arr = np.load(path, mmap_mode="r")

        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array D,H,W, got shape={arr.shape} in {path}")

        arr = self._orient_depth_first(arr)
        crop = self._crop(arr)
        crop = np.asarray(crop, dtype=np.float32)
        crop = np.nan_to_num(crop, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize:
            crop = self._normalize01(crop)

        return torch.from_numpy(crop.astype(np.float32)).unsqueeze(0)
