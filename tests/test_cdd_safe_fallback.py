from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.utils.cdd_import import safe_constrained_diffusion_decomposition


def test_safe_cdd_drops_gaussian_backend_for_old_api():
    calls = []

    def fake_cdd(arr, **kwargs):
        calls.append(dict(kwargs))
        if "gaussian_backend" in kwargs:
            raise TypeError("unexpected keyword argument 'gaussian_backend'")
        return "ok"

    cdd = SimpleNamespace(constrained_diffusion_decomposition=fake_cdd)

    with pytest.warns(RuntimeWarning, match="does not accept gaussian_backend"):
        out = safe_constrained_diffusion_decomposition(
            cdd,
            "arr",
            use_gpu=True,
            gaussian_backend="cuda",
        )

    assert out == "ok"
    assert calls == [
        {"use_gpu": True, "gaussian_backend": "cuda"},
        {"use_gpu": False},
    ]


def test_safe_cdd_falls_back_to_cpu_after_backend_failure():
    calls = []

    def fake_cdd(arr, **kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("use_gpu"):
            raise RuntimeError("CUDA backend failed")
        return "cpu-ok"

    cdd = SimpleNamespace(constrained_diffusion_decomposition=fake_cdd)

    with pytest.warns(RuntimeWarning, match="retrying CDD on CPU"):
        out = safe_constrained_diffusion_decomposition(
            cdd,
            "arr",
            use_gpu=True,
            gaussian_backend="cuda",
        )

    assert out == "cpu-ok"
    assert calls == [
        {"use_gpu": True, "gaussian_backend": "cuda"},
        {"use_gpu": False},
    ]
