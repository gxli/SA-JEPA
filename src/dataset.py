from __future__ import annotations
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.npy import _safe_load_npy, normalize01

try:
    import h5py  # optional — enables chunked HDF5 reading for large 3D arrays
except ImportError:
    h5py = None

try:
    from astropy.io import fits  # optional — FITS file support
except ImportError:
    fits = None


class JEPADataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1000,
        data_root: str = "data",
        npy_pattern: str = "*.npy",
        cube_slice_strategy: str = "auto",
        cube_slice_axis: int = 0,
        cube_slice_index: int = 0,
        crop_mode: str = "none",
        crop_size: int | tuple[int, int] | list[int] | None = None,
        d4_augment: bool = False,
        input_type: str = "image",
        image_batch_inference: bool = False,
        image_batch_selected_indices: dict | None = None,
        cdd_cache: dict | None = None,
        crop_min_valid_fraction: float = 0.0,
        cdd_use_log: bool = False,
    ):
        self.input_type = str(input_type).lower()
        allowed_input_types = {"image", "cube", "image_batch"}
        if self.input_type not in allowed_input_types:
            raise ValueError(
                f"Unknown input_type={input_type}. "
                "Use 'image', 'cube', or 'image_batch'."
            )
        self.image_batch_inference = bool(image_batch_inference)
        self.image_batch_selected_indices = image_batch_selected_indices
        self.cube_slice_strategy = str(cube_slice_strategy).lower()
        allowed_strategies = {"auto", "random", "center", "fixed", "all"}
        if self.cube_slice_strategy not in allowed_strategies:
            raise ValueError(
                f"Unknown cube_slice_strategy={cube_slice_strategy}. "
                "Use 'auto', 'random', 'center', 'fixed', or 'all'."
            )
        self.num_samples = num_samples
        self.cube_slice_axis = cube_slice_axis
        self.cube_slice_index = cube_slice_index
        self.crop_mode = str(crop_mode).lower()
        if self.crop_mode not in {"none", "random", "center"}:
            raise ValueError("crop_mode must be one of: none, random, center")
        self.crop_size = self._coerce_crop_size(crop_size)
        if self.crop_mode != "none" and self.crop_size is None:
            raise ValueError("crop_size is required when crop_mode is not 'none'")
        self.d4_augment = bool(d4_augment)
        self.cdd_cache = cdd_cache or None
        self.cdd_use_log = bool(cdd_use_log)
        self.crop_min_valid_fraction = float(crop_min_valid_fraction) if crop_min_valid_fraction is not None else 0.0

        pattern = os.path.join(data_root, npy_pattern)
        self.pattern = pattern
        self.npy_files = sorted(glob.glob(pattern))

        # Also scan for .h5 files (preferred for fast random-access slicing)
        h5_pattern = pattern.replace(".npy", ".h5") if pattern.endswith(".npy") else os.path.join(data_root, "*.h5")
        self.h5_files = sorted(glob.glob(h5_pattern)) if h5py is not None else []
        if self.h5_files:
            print(f"[dataset] Found {len(self.h5_files)} .h5 file(s); using chunked HDF5 for fast I/O")

        self.fits_files = []
        if not self.npy_files and not self.h5_files:
            raise FileNotFoundError(f"No .npy, .h5, or .fits files found with pattern: {pattern}")
        self.sample_index = self._build_sample_index()
        if self.num_samples is None:
            self.num_samples = len(self.sample_index)
    def _preprocess_arr2d(self, arr2d: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr2d, dtype=np.float32)
        finite = np.isfinite(arr)
        out = np.zeros_like(arr, dtype=np.float32)
        if not bool(finite.any()):
            return out
        finite_vals = arr[finite]
        amin = float(finite_vals.min())
        amax = float(finite_vals.max())
        denom = amax - amin
        if denom > 1e-20:
            out[finite] = (arr[finite] - amin) / denom
        return out

    @staticmethod
    def _probe_file_shape(path: str) -> tuple[int, ...]:
        """Read array shape without loading full data. Supports .npy and .h5."""
        if path.endswith(".h5"):
            if h5py is None:
                raise ImportError("h5py is required to read .h5 files; pip install h5py")
            with h5py.File(path, "r") as f:
                return tuple(f["data"].shape)
        if path.endswith(".fits"):
            if fits is None:
                raise ImportError("astropy is required to read .fits files; pip install astropy")
            return fits.getdata(path, memmap=True).shape
        arr = _safe_load_npy(path, mmap_mode="r")
        return arr.shape

    @staticmethod
    def _is_h5(path: str) -> bool:
        return path.endswith(".h5")

    @staticmethod
    def _is_fits(path: str) -> bool:
        return path.endswith(".fits")

    def _build_sample_index(self):
        all_files = list(self.npy_files) + list(self.h5_files)
        index = []
        for path in all_files:
            shape = self._probe_file_shape(path)
            ndim = len(shape)
            if ndim == 2:
                index.append((path, None))
            elif ndim == 3:
                if self.input_type == "image_batch":
                    if self.image_batch_selected_indices is not None and path in self.image_batch_selected_indices:
                        sel = self.image_batch_selected_indices[path]
                        for sidx in sel:
                            index.append((path, int(sidx)))
                    elif self.image_batch_inference:
                        index.append((path, 0))
                    else:
                        index.append((path, None))
                else:
                    axis = self.cube_slice_axis % 3
                    depth = shape[axis]
                    if self.cube_slice_strategy == "all":
                        for sidx in range(depth):
                            index.append((path, sidx))
                    else:
                        index.append((path, None))
            else:
                raise ValueError(f"Expected 2D or 3D array in {path}, got shape {shape}")
        if not index:
            raise ValueError("No usable samples found from npy files.")
        return index

    @property
    def rng(self):
        """Lazily initialize an isolated generator per DataLoader worker."""
        if not hasattr(self, "_rng") or self._rng is None:
            import torch.utils.data

            worker_info = torch.utils.data.get_worker_info()
            seed = worker_info.seed % (2**31 - 1) if worker_info is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())
            self._rng = np.random.default_rng(seed)
        return self._rng

    def _pick_slice_index(self, depth: int) -> int:
        strategy = self.cube_slice_strategy
        if strategy == "auto":
            strategy = "random"
        if strategy == "random":
            return int(self.rng.integers(0, depth))
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
        if self.input_type == "image_batch":
            depth = arr.shape[0]
            if self.image_batch_inference:
                sidx = 0
            elif forced_slice_idx is not None:
                sidx = forced_slice_idx
            else:
                sidx = int(self.rng.integers(0, depth))
            sidx = int(np.clip(sidx, 0, depth - 1))
            return arr[sidx], int(sidx)
        axis = self.cube_slice_axis % 3
        depth = arr.shape[axis]
        sidx = forced_slice_idx
        if sidx is None:
            sidx = self._pick_slice_index(depth)
        slicer = [slice(None), slice(None), slice(None)]
        slicer[axis] = int(np.clip(sidx, 0, depth - 1))
        return arr[tuple(slicer)], int(sidx)

    def _extract_2d_from_cdd(self, cdd: np.ndarray, forced_slice_idx=None) -> np.ndarray:
        if cdd.ndim == 3:
            return cdd
        if cdd.ndim != 4:
            raise ValueError(f"Expected cached CDD shape (S,H,W) or (S,D,H,W), got {cdd.shape}")
        if self.input_type == "image_batch":
            axis = 0
            depth = cdd.shape[axis + 1]
            if self.image_batch_inference:
                sidx = 0
            elif forced_slice_idx is not None:
                sidx = forced_slice_idx
            else:
                sidx = int(self.rng.integers(0, depth))
        else:
            axis = self.cube_slice_axis % 3
            depth = cdd.shape[axis + 1]
            sidx = forced_slice_idx
            if sidx is None:
                sidx = self._pick_slice_index(depth)
        slicer = [slice(None), slice(None), slice(None), slice(None)]
        slicer[axis + 1] = int(np.clip(sidx, 0, depth - 1))
        return cdd[tuple(slicer)]

    def _load_sample(self, path: str, forced_slice_idx=None) -> torch.Tensor:
        if self._is_h5(path):
            if h5py is None:
                raise ImportError("h5py is required to read .h5 files; pip install h5py")
            with h5py.File(path, "r") as h5_file:
                ds = h5_file["data"]
                arr2d, _ = self._extract_2d_from_array(ds, forced_slice_idx=forced_slice_idx)
                arr2d = np.asarray(arr2d, dtype=np.float32)
        elif self._is_fits(path):
            if fits is None:
                raise ImportError("astropy is required to read .fits files; pip install astropy")
            arr_mm = fits.getdata(path, memmap=True)
            arr2d, _ = self._extract_2d_from_array(arr_mm, forced_slice_idx=forced_slice_idx)
        else:
            arr_mm = _safe_load_npy(path, mmap_mode="r")
            arr2d, _ = self._extract_2d_from_array(arr_mm, forced_slice_idx=forced_slice_idx)
        arr = self._preprocess_arr2d(arr2d)

        # Keep native resolution (including non-square fields).
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)  # 1 x H x W

    @staticmethod
    def _coerce_crop_size(crop_size) -> tuple[int, int] | None:
        if crop_size is None:
            return None
        if isinstance(crop_size, (list, tuple)):
            if len(crop_size) != 2:
                raise ValueError(f"crop_size must be an int or [height, width], got {crop_size!r}")
            crop_h, crop_w = int(crop_size[0]), int(crop_size[1])
        else:
            crop_size_int = int(crop_size)
            if crop_size_int <= 0:
                raise ValueError(f"crop_size must be positive, got {crop_size!r}")
            return crop_size_int, crop_size_int
        if crop_h <= 0 or crop_w <= 0:
            raise ValueError(f"crop_size must be positive, got {crop_size!r}")
        return crop_h, crop_w

    def _crop_slices(self, h: int, w: int) -> tuple[slice, slice] | None:
        if self.crop_mode == "none" or self.crop_size is None:
            return None
        crop_h, crop_w = self._coerce_crop_size(self.crop_size)
        if crop_h > h or crop_w > w:
            raise ValueError(f"crop_size={self.crop_size} exceeds image shape={(h, w)}")
        if self.crop_mode == "center":
            y0 = (h - crop_h) // 2
            x0 = (w - crop_w) // 2
        else:
            # Crop origin stays inside the margin implied by the crop size.
            y0 = int(self.rng.integers(0, h - crop_h + 1))
            x0 = int(self.rng.integers(0, w - crop_w + 1))
        return slice(y0, y0 + crop_h), slice(x0, x0 + crop_w)

    def _crop_tensor(self, x: torch.Tensor) -> torch.Tensor:
        crop = self._crop_slices(int(x.shape[-2]), int(x.shape[-1]))
        if crop is None:
            return x
        crop_y, crop_x = crop
        return x[..., crop_y, crop_x]

    @staticmethod
    def _normalize01(arr: np.ndarray) -> np.ndarray:
        return normalize01(arr)

    def _apply_augmentations(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply d4 flips to all tensors identically. Shared by both data paths."""
        if not tensors:
            return tensors
        if self.d4_augment:
            h, w = tensors[0].shape[-2], tensors[0].shape[-1]
            if h == w:
                k = int(self.rng.integers(0, 4))
                if k:
                    tensors = tuple(torch.rot90(t, k=k, dims=(-2, -1)) for t in tensors)
                if bool(self.rng.integers(0, 2)):
                    tensors = tuple(torch.flip(t, dims=(-1,)) for t in tensors)
            elif bool(self.rng.integers(0, 2)):
                tensors = tuple(torch.flip(t, dims=(-2,)) for t in tensors)
            if h != w and bool(self.rng.integers(0, 2)):
                tensors = tuple(torch.flip(t, dims=(-1,)) for t in tensors)
        return tensors

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        key = self.sample_index[idx % len(self.sample_index)]

        if self.cdd_cache is not None:
            path, forced_slice_idx = key
            # New caches store a dict {"untransformed": ..., "transformed": ...}.
            # Pick the right variant based on whether the model expects log-space CDD.
            cache_val = self.cdd_cache.get((path, None))
            if cache_val is None:
                cache_val = self.cdd_cache.get(key)
            if cache_val is None:
                raise KeyError(f"CDD cache miss for key={key}")
            if isinstance(cache_val, dict):
                cdd_np = cache_val["transformed" if self.cdd_use_log else "untransformed"]
            else:
                cdd_np = cache_val
            cdd_np = self._extract_2d_from_cdd(np.asarray(cdd_np), forced_slice_idx=forced_slice_idx)
            # cdd_np is now (S, H, W) float32
            cdd_orig = torch.from_numpy(cdd_np.astype(np.float32))
            x_clean_full = cdd_orig.sum(dim=0, keepdim=True)  # 1 x H x W
            max_retries = 100
            for attempt in range(max_retries):
                crop = self._crop_slices(int(cdd_orig.shape[-2]), int(cdd_orig.shape[-1]))
                if crop is not None:
                    crop_y, crop_x = crop
                    cdd_cropped = cdd_orig[..., crop_y, crop_x]
                    x_clean = x_clean_full[..., crop_y, crop_x]
                else:
                    cdd_cropped = cdd_orig
                    x_clean = x_clean_full
                if self.crop_min_valid_fraction > 0.0 and self.crop_mode == "random" and crop is not None:
                    arr = x_clean.squeeze(0).numpy()
                    finite_nonzero = np.isfinite(arr) & (arr > 1e-8)
                    if finite_nonzero.mean() >= self.crop_min_valid_fraction:
                        break
                else:
                    break
                if attempt == max_retries - 1:
                    break
            cdd_orig, x_clean = self._apply_augmentations(cdd_cropped, x_clean)
            return cdd_orig, x_clean

        path, forced_slice_idx = key
        max_retries = 100
        for attempt in range(max_retries):
            sample = self._load_sample(path, forced_slice_idx=forced_slice_idx).clone()  # 1 x H x W
            sample = self._crop_tensor(sample)
            if self.crop_min_valid_fraction > 0.0 and self.crop_mode == "random":
                arr = sample.squeeze(0).numpy()
                finite_nonzero = np.isfinite(arr) & (arr > 1e-8)
                if finite_nonzero.mean() >= self.crop_min_valid_fraction:
                    break
            else:
                break
            if attempt == max_retries - 1:
                break  # accept anyway after max retries
        (sample,) = self._apply_augmentations(sample)
        return sample
