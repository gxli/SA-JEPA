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
