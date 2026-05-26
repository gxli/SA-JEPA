from __future__ import annotations

import glob
import hashlib
import os
import tempfile
import numpy as np
import torch
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
        cube_slice_strategy: str = "auto",
        cube_slice_axis: int = 0,
        cube_slice_index: int = 0,
        random_roll_max: int = 0,
        d4_augment: bool = False,
        cache_cdd: bool = True,
        cdd_cache_dir: str | None = None,
        cache_random_slices: bool = False,
        precompute_cdd_cache_all_slices: bool = False,
    ):
        self.cube_slice_strategy = str(cube_slice_strategy).lower()
        allowed_strategies = {"auto", "random", "center", "fixed", "all"}
        if self.cube_slice_strategy not in allowed_strategies:
            raise ValueError(
                f"Unknown cube_slice_strategy={cube_slice_strategy}. "
                "Use 'auto', 'random', 'center', 'fixed', or 'all'."
            )
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
        self.cube_slice_axis = cube_slice_axis
        self.cube_slice_index = cube_slice_index
        self.random_roll_max = int(random_roll_max)
        self.d4_augment = bool(d4_augment)
        self.cache_cdd = bool(cache_cdd)
        self.cdd_cache_dir = cdd_cache_dir
        self.cache_random_slices = bool(cache_random_slices)
        self.precompute_cdd_cache_all_slices = bool(precompute_cdd_cache_all_slices)
        if self.cache_cdd and self.cdd_cache_dir:
            os.makedirs(self.cdd_cache_dir, exist_ok=True)
        if self.precompute_cdd_cache_all_slices and not self.cache_random_slices:
            # Precompute is only useful when random-slice reads are allowed to use cache.
            self.cache_random_slices = True
            print("[JEPADataset] enabling cache_random_slices=True because precompute_cdd_cache_all_slices=True")
        if self.apply_cdd and self.cache_cdd and self.cube_slice_strategy in ("random", "auto") and not self.cache_random_slices:
            # Random-slice cache is disabled by default for reproducibility/cost control.
            print("[JEPADataset] cache_random_slices=False: skipping CDD cache for random 3D slices.")

        pattern = os.path.join(data_root, npy_pattern)
        self.npy_files = sorted(glob.glob(pattern))

        if not self.npy_files:
            raise FileNotFoundError(f"No .npy files found with pattern: {pattern}")
        self.sample_index = self._build_sample_index()
        if self.num_samples is None:
            self.num_samples = len(self.sample_index)
        if self.precompute_cdd_cache_all_slices:
            self._precompute_cdd_cache_all_slices()

    def _preprocess_arr2d(self, arr2d: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr2d, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = self._normalize01(arr)
        if self.apply_cdd:
            arr = self._apply_cdd(arr)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(np.float32, copy=False)

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
        strategy = self.cube_slice_strategy
        if strategy == "auto":
            strategy = "random"
        if strategy == "random":
            return int(np.random.randint(0, depth))
        if strategy == "center":
            return depth // 2
        if strategy == "fixed":
            return int(np.clip(self.cube_slice_index, 0, depth - 1))
        raise ValueError(
            f"Unknown cube_slice_strategy={strategy}. "
            "Use 'auto', 'random', 'center', 'fixed', or 'all'."
        )

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

    @staticmethod
    def _atomic_save_npy(path: str, arr: np.ndarray) -> None:
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "wb") as f:
            np.save(f, arr)
        os.replace(tmp_path, path)

    def _precompute_cdd_cache_all_slices(self) -> None:
        if not self.apply_cdd or not self.cache_cdd or not self.cdd_cache_dir:
            print("[JEPADataset] precompute_cdd_cache_all_slices skipped (requires apply_cdd=true, cache_cdd=true, cdd_cache_dir set)")
            return
        n_files_2d = 0
        n_files_3d = 0
        n_entries_2d = 0
        n_slices_total = 0
        n_written = 0
        axis = self.cube_slice_axis % 3
        for path in self.npy_files:
            arr_mm = np.load(path, mmap_mode="r")
            if arr_mm.ndim == 2:
                n_files_2d += 1
                n_entries_2d += 1
                cpath = self._cache_file_path(path, None)
                if not os.path.exists(cpath):
                    arr = np.asarray(arr_mm, dtype=np.float32)
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    arr = self._normalize01(arr)
                    arr = self._apply_cdd(arr)
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    self._atomic_save_npy(cpath, arr.astype(np.float32, copy=False))
                    n_written += 1
                continue
            if arr_mm.ndim != 3:
                continue
            n_files_3d += 1
            depth = int(arr_mm.shape[axis])
            for sidx in range(depth):
                n_slices_total += 1
                cpath = self._cache_file_path(path, sidx)
                if os.path.exists(cpath):
                    continue
                arr2d, _ = self._extract_2d_from_array(arr_mm, forced_slice_idx=sidx)
                arr = np.asarray(arr2d, dtype=np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                arr = self._normalize01(arr)
                arr = self._apply_cdd(arr)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                self._atomic_save_npy(cpath, arr.astype(np.float32, copy=False))
                n_written += 1
        print(
            f"[JEPADataset] precompute_cdd_cache_all_slices done "
            f"files_2d={n_files_2d} entries_2d={n_entries_2d} "
            f"files_3d={n_files_3d} slices_total={n_slices_total} slices_written={n_written}"
        )

    def _load_sample(self, path: str, forced_slice_idx=None) -> torch.Tensor:
        with open(path, "rb") as f:
            arr_mm = np.load(f)
        arr2d, sidx = self._extract_2d_from_array(arr_mm, forced_slice_idx=forced_slice_idx)

        is_3d_random_slice = (arr_mm.ndim == 3) and (self.cube_slice_strategy in ("random", "auto"))
        arr = None
        if self.apply_cdd and self.cache_cdd and (self.cache_random_slices or not is_3d_random_slice):
            if self.cdd_cache_dir:
                cpath = self._cache_file_path(path, sidx)
                if os.path.exists(cpath):
                    arr = np.load(cpath).astype(np.float32, copy=False)

        if arr is None:
            arr = self._preprocess_arr2d(arr2d)
            if self.apply_cdd and self.cache_cdd and self.cdd_cache_dir and (self.cache_random_slices or not is_3d_random_slice):
                cpath = self._cache_file_path(path, sidx)
                if not os.path.exists(cpath):
                    self._atomic_save_npy(cpath, arr.astype(np.float32, copy=False))

        if self.log_transform:
            eps = self._choose_log_eps(arr, self.log_eps)
            arr = np.log(np.clip(arr, a_min=0.0, a_max=None) + eps)

        # Keep native resolution (including non-square fields).
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)  # 1 x H x W

    @staticmethod
    def _normalize01(arr: np.ndarray) -> np.ndarray:
        amin = float(arr.min())
        amax = float(arr.max())
        denom = amax - amin
        if denom > 1e-20:
            return (arr - amin) / denom
        return np.zeros_like(arr, dtype=np.float32)

    def _apply_cdd(self, arr01: np.ndarray) -> np.ndarray:
        if "MPLCONFIGDIR" not in os.environ:
            os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mplconfig")
        os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
        import constrained_diffusion as cdd

        arr_in = arr01.astype(np.float32, copy=True)
        # Policy: Linearly normalize only to keep values in a reasonable range before CDD.
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
        if self.d4_augment:
            # Shape-safe augmentation for non-square inputs: random H/V flips only.
            if bool(np.random.randint(0, 2)):
                sample = torch.flip(sample, dims=(-1,))  # horizontal
            if bool(np.random.randint(0, 2)):
                sample = torch.flip(sample, dims=(-2,))  # vertical
        if self.random_roll_max > 0:
            dy = int(np.random.randint(-self.random_roll_max, self.random_roll_max + 1))
            dx = int(np.random.randint(-self.random_roll_max, self.random_roll_max + 1))
            pad_val = self.random_roll_max
            padded = torch.nn.functional.pad(sample, (pad_val, pad_val, pad_val, pad_val), mode='reflect')
            h, w = sample.shape[-2], sample.shape[-1]
            y0 = pad_val - dy
            x0 = pad_val - dx
            sample = padded[..., y0:y0+h, x0:x0+w]
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
