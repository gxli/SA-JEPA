from __future__ import annotations
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class JEPADataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1000,
        data_root: str = "data",
        npy_pattern: str = "*.npy",
        cube_slice_strategy: str = "auto",
        cube_slice_axis: int = 0,
        cube_slice_index: int = 0,
        random_roll_max: int = 0,
        d4_augment: bool = False,
        input_type: str = "image",
        image_batch_inference: bool = False,
        image_batch_selected_indices: dict | None = None,
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
        self.random_roll_max = int(random_roll_max)
        self.d4_augment = bool(d4_augment)

        pattern = os.path.join(data_root, npy_pattern)
        self.npy_files = sorted(glob.glob(pattern))

        if not self.npy_files:
            raise FileNotFoundError(f"No .npy files found with pattern: {pattern}")
        self.sample_index = self._build_sample_index()
        if self.num_samples is None:
            self.num_samples = len(self.sample_index)
    def _preprocess_arr2d(self, arr2d: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr2d, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = self._normalize01(arr)
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
                if self.input_type == "image_batch":
                    if self.image_batch_selected_indices is not None and path in self.image_batch_selected_indices:
                        sel = self.image_batch_selected_indices[path]
                        for sidx in sel:
                            index.append((path, int(sidx)))
                    elif self.image_batch_inference:
                        index.append((path, 0))
                    else:
                        # Dynamic random selection each __getitem__
                        index.append((path, None))
                else:
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
        if self.input_type == "image_batch":
            depth = arr.shape[0]
            if self.image_batch_inference:
                sidx = 0
            elif forced_slice_idx is not None:
                sidx = forced_slice_idx
            else:
                sidx = int(np.random.randint(0, depth))
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

    def _load_sample(self, path: str, forced_slice_idx=None) -> torch.Tensor:
        arr_mm = np.load(path, mmap_mode="r")
        arr2d, _ = self._extract_2d_from_array(arr_mm, forced_slice_idx=forced_slice_idx)
        arr = self._preprocess_arr2d(arr2d)

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
