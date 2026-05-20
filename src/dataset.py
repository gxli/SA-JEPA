import glob
import hashlib
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class JEPADataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 224,
        data_root: str = "data",
        npy_pattern: str = "*.npy",
        log_transform: bool = True,
        log_eps: float = 1.0,
        cdd_scales=None,
        cdd_strength: float = 1.0,
        cdd_clip: bool = True,
        norm_before_cdd: bool = True,
        cdd_mode: str = "log",
        cdd_constrained: bool = True,
        cdd_sm_mode: str = "reflect",
        apply_cdd: bool = True,
        cube_slice_strategy: str = "random",
        cube_slice_axis: int = 0,
        cube_slice_index: int = 0,
        random_roll_max: int = 0,
        cache_cdd: bool = True,
        cdd_cache_dir: str | None = None,
        cdd_mem_cache_max: int = 64,
        cache_random_slices: bool = False,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.log_transform = log_transform
        self.log_eps = log_eps
        self.cdd_scales = cdd_scales if cdd_scales is not None else [2, 4, 8, 16]
        self.cdd_strength = cdd_strength
        self.cdd_clip = cdd_clip
        self.norm_before_cdd = norm_before_cdd
        self.cdd_mode = cdd_mode
        self.cdd_constrained = cdd_constrained
        self.cdd_sm_mode = cdd_sm_mode
        self.apply_cdd = apply_cdd
        self.cube_slice_strategy = cube_slice_strategy
        self.cube_slice_axis = cube_slice_axis
        self.cube_slice_index = cube_slice_index
        self.random_roll_max = int(random_roll_max)
        self.cache_cdd = bool(cache_cdd)
        self.cdd_cache_dir = cdd_cache_dir
        self.cdd_mem_cache_max = int(cdd_mem_cache_max)
        self.cache_random_slices = bool(cache_random_slices)
        self._sample_cache = OrderedDict()
        if self.cache_cdd and self.cdd_cache_dir:
            os.makedirs(self.cdd_cache_dir, exist_ok=True)
        if self.apply_cdd and self.cache_cdd and self.cube_slice_strategy.lower() == "random" and not self.cache_random_slices:
            # Random-slice cache is disabled by default for reproducibility/cost control.
            print("[JEPADataset] cache_random_slices=False: skipping CDD cache for random 3D slices.")

        pattern = os.path.join(data_root, npy_pattern)
        self.npy_files = sorted(glob.glob(pattern))

        if not self.npy_files:
            raise FileNotFoundError(f"No .npy files found with pattern: {pattern}")
        self.sample_index = self._build_sample_index()
        if self.num_samples is None:
            self.num_samples = len(self.sample_index)

    def _build_sample_index(self):
        index = []
        for path in self.npy_files:
            arr = np.load(path, mmap_mode="r")
            ndim = arr.ndim
            shape = arr.shape
            if ndim == 2:
                index.append((path, None))
            elif ndim == 3:
                axis = self.cube_slice_axis % 3
                depth = shape[axis]
                if self.cube_slice_strategy == "all":
                    for sidx in range(depth):
                        index.append((path, sidx))
                else:
                    # slice will be selected dynamically in __getitem__
                    index.append((path, None))
            else:
                raise ValueError(f"Expected 2D or 3D array in {path}, got shape {shape}")
        if not index:
            raise ValueError("No usable samples found from npy files.")
        return index

    def _pick_slice_index(self, depth: int) -> int:
        strategy = self.cube_slice_strategy.lower()
        if strategy == "random":
            return int(np.random.randint(0, depth))
        if strategy == "center":
            return depth // 2
        if strategy == "fixed":
            return int(np.clip(self.cube_slice_index, 0, depth - 1))
        # Fallback for unknown values: center.
        return depth // 2

    def _extract_2d_from_array(self, arr: np.ndarray, forced_slice_idx=None) -> tuple[np.ndarray, int | None]:
        if arr.ndim == 2:
            return arr, None
        axis = self.cube_slice_axis % 3
        depth = arr.shape[axis]
        sidx = forced_slice_idx
        if sidx is None:
            sidx = self._pick_slice_index(depth)
        slicer = [slice(None), slice(None), slice(None)]
        slicer[axis] = int(np.clip(sidx, 0, depth - 1))
        return arr[tuple(slicer)], int(sidx)

    def _cache_file_path(self, path: str, slice_idx: int | None) -> str:
        tag = f"{os.path.abspath(path)}|axis={self.cube_slice_axis}|slice={slice_idx}"
        digest = hashlib.md5(tag.encode("utf-8")).hexdigest()[:16]
        stem = os.path.splitext(os.path.basename(path))[0]
        return os.path.join(self.cdd_cache_dir, f"{stem}__{digest}.npy")

    def _load_sample(self, path: str, forced_slice_idx=None) -> torch.Tensor:
        arr_mm = np.load(path, mmap_mode="r")
        arr2d, sidx = self._extract_2d_from_array(arr_mm, forced_slice_idx=forced_slice_idx)
        cache_key = (os.path.abspath(path), sidx)

        is_random_slice = (self.cube_slice_strategy.lower() == "random")
        allow_random_cache = bool(self.cache_random_slices)
        use_cache = self.apply_cdd and self.cache_cdd and (allow_random_cache or not is_random_slice)
        arr = None
        if use_cache and cache_key in self._sample_cache:
            arr = self._sample_cache.pop(cache_key).copy()
            # LRU touch.
            self._sample_cache[cache_key] = arr.copy()
        elif use_cache and self.cdd_cache_dir:
            cpath = self._cache_file_path(path, sidx)
            if os.path.exists(cpath):
                arr = np.load(cpath).astype(np.float32, copy=False)

        if arr is None:
            arr = np.asarray(arr2d, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize once before CDD.
            arr = self._normalize01(arr)

            if self.apply_cdd:
                arr = self._apply_cdd(arr)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                if use_cache:
                    self._sample_cache[cache_key] = arr.astype(np.float32, copy=True)
                    # Enforce LRU bound to avoid unbounded memory growth.
                    while self.cdd_mem_cache_max >= 0 and len(self._sample_cache) > self.cdd_mem_cache_max:
                        self._sample_cache.popitem(last=False)
                    if self.cdd_cache_dir:
                        cpath = self._cache_file_path(path, sidx)
                        if not os.path.exists(cpath):
                            np.save(cpath, arr.astype(np.float32, copy=False))

        if self.log_transform:
            eps = self._choose_log_eps(arr, self.log_eps)
            arr = np.log(np.clip(arr, a_min=0.0, a_max=None) + eps)

        x = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)  # 1 x H x W
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return x

    @staticmethod
    def _normalize01(arr: np.ndarray) -> np.ndarray:
        amin = float(arr.min())
        amax = float(arr.max())
        denom = amax - amin
        if denom > 1e-20:
            return (arr - amin) / denom
        return np.zeros_like(arr, dtype=np.float32)

    def _apply_cdd(self, arr01: np.ndarray) -> np.ndarray:
        import constrained_diffusion as cdd

        arr_in = arr01.astype(np.float32, copy=True)
        # Policy: linear normalize only to keep values in a reasonable range before CDD.
        if self.norm_before_cdd:
            arr_in = self._normalize01(arr_in)

        cdd_kwargs = dict(
            mode=self.cdd_mode,
            constrained=bool(self.cdd_constrained),
            sm_mode=self.cdd_sm_mode,
            return_scales=False,
            verbose=False,
            use_gpu=False,
        )
        try:
            result, residual = cdd.constrained_diffusion_decomposition(
                arr_in,
                scales=tuple(float(s) for s in self.cdd_scales),
                **cdd_kwargs,
            )
        except TypeError:
            result, residual = cdd.constrained_diffusion_decomposition(
                arr_in,
                num_channels=max(1, len(self.cdd_scales)),
                **cdd_kwargs,
            )

        result = np.asarray(result, dtype=np.float32)
        residual = np.asarray(residual, dtype=np.float32)
        # CDD channels are treated as non-negative components.
        result = np.clip(result, a_min=0.0, a_max=None)
        recon = np.sum(result, axis=0) + residual

        # Strength=1.0 keeps CDD reconstruction; other values blend relative to input.
        out = arr_in + float(self.cdd_strength) * (recon - arr_in)
        # By definition we keep CDD output non-negative before any log transform.
        out = np.clip(out, a_min=0.0, a_max=None)
        if self.cdd_clip:
            lo = float(np.percentile(out, 0.5))
            hi = float(np.percentile(out, 99.5))
            if hi > lo + 1e-12:
                out = np.clip(out, lo, hi)
        return out.astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        path, forced_slice_idx = self.sample_index[idx % len(self.sample_index)]
        sample = self._load_sample(path, forced_slice_idx=forced_slice_idx).clone()  # 1 x H x W
        if self.random_roll_max > 0:
            # Inclusive, symmetric dithering in [-random_roll_max, random_roll_max].
            dy = int(np.random.randint(-self.random_roll_max, self.random_roll_max + 1))
            dx = int(np.random.randint(-self.random_roll_max, self.random_roll_max + 1))
            sample = torch.roll(sample, shifts=(dy, dx), dims=(-2, -1))
        return sample

    @staticmethod
    def _choose_log_eps(arr: np.ndarray, cfg_eps: float = 1.0) -> float:
        pos = arr[arr > 0]
        if pos.size == 0:
            return 1e-30
        p10 = float(np.percentile(pos, 10))
        auto_eps = max(1e-30, p10 * 1e-3)
        # Safety cap to avoid flattening tiny-valued fields.
        return min(float(cfg_eps), max(1e-30, p10 * 1e-2), auto_eps * 10.0)
