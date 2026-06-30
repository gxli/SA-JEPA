from __future__ import annotations

import builtins
import os
import tempfile
import warnings
from types import ModuleType


def ensure_writable_mpl_config(session_dir: str = "") -> str:
    """Keep optional CDD imports away from unwritable matplotlib home caches."""
    existing = os.environ.get("MPLCONFIGDIR")
    if existing:
        return existing
    candidates = []
    if session_dir:
        candidates.append(os.path.join(session_dir, ".matplotlib"))
    candidates.append(os.path.join(tempfile.gettempdir(), "sajepa_mplconfig"))
    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = path
            return path
        except OSError:
            continue
    raise RuntimeError("Could not create a writable MPLCONFIGDIR for CDD import")


def import_constrained_diffusion(*, session_dir: str = "", allow_monai: bool = False) -> ModuleType:
    """Import constrained_diffusion without pulling MONAI into non-MONAI paths."""
    ensure_writable_mpl_config(session_dir)
    if allow_monai:
        import constrained_diffusion as cdd

        return cdd

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "monai" or name.startswith("monai."):
            raise ImportError("MONAI disabled for this constrained_diffusion import")
        return original_import(name, globals, locals, fromlist, level)

    try:
        builtins.__import__ = guarded_import
        import constrained_diffusion as cdd
    finally:
        builtins.__import__ = original_import
    return cdd


def safe_constrained_diffusion_decomposition(cdd: ModuleType, *args, **kwargs):
    """Call CDD with compatibility fallbacks for older/GPU-fragile installs."""
    try:
        return cdd.constrained_diffusion_decomposition(*args, **kwargs)
    except TypeError as exc:
        msg = str(exc)
        if "gaussian_backend" not in msg and "unexpected keyword" not in msg:
            raise
        if "gaussian_backend" not in kwargs:
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("gaussian_backend", None)
        retry_kwargs["use_gpu"] = False
        warnings.warn(
            "CDD backend does not accept gaussian_backend; retrying CDD on CPU without that argument.",
            RuntimeWarning,
            stacklevel=2,
        )
        return cdd.constrained_diffusion_decomposition(*args, **retry_kwargs)
    except Exception:
        requested_gpu = bool(kwargs.get("use_gpu", False))
        requested_backend = kwargs.get("gaussian_backend")
        if not requested_gpu and requested_backend in (None, "", "cpu"):
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("gaussian_backend", None)
        retry_kwargs["use_gpu"] = False
        warnings.warn(
            "CDD failed with the requested Gaussian/GPU backend; retrying CDD on CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return cdd.constrained_diffusion_decomposition(*args, **retry_kwargs)
