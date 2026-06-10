from __future__ import annotations

import numpy as np


def _safe_load_npy(path: str, *, mmap_mode: str | None = "r") -> np.ndarray:
    """Load .npy data, falling back for legacy object/pickled arrays.

    Modern NumPy refuses to mmap pickled object arrays. These are local training
    artifacts, so fall back only for that specific failure and normalize common
    wrapper shapes back into an ndarray.
    """
    try:
        return np.load(path, mmap_mode=mmap_mode)
    except ValueError as exc:
        msg = str(exc).lower()
        if "pickle" not in msg and "object" not in msg:
            raise

    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object:
        if arr.shape == ():
            arr = arr.item()
        elif arr.size == 1:
            arr = arr.reshape(-1)[0]
        else:
            arr = arr.tolist()
    return np.asarray(arr)


def normalize01(x: np.ndarray) -> np.ndarray:
    """Normalize finite array values to [0, 1], replacing non-finite values safely."""
    arr = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(arr)
    if not bool(finite.any()):
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    arr = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
    if hi > lo + 1e-20:
        return ((arr - lo) / (hi - lo)).astype(np.float32)
    return np.zeros_like(arr, dtype=np.float32)
