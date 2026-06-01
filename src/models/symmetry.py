from __future__ import annotations

import itertools
from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def _symmetric_cache_pass(module: nn.Module, enabled: bool):
    sentinel = object()
    old = getattr(module, "_symmetric_cache_pass", sentinel)
    setattr(module, "_symmetric_cache_pass", bool(enabled))
    try:
        yield
    finally:
        if old is sentinel:
            delattr(module, "_symmetric_cache_pass")
        else:
            setattr(module, "_symmetric_cache_pass", old)


def _transform_tensor_kwargs(
    kwargs: dict,
    spatial_shape: tuple[int, ...],
    transform,
) -> dict:
    out = {}
    for name, val in kwargs.items():
        if torch.is_tensor(val) and val.ndim >= len(spatial_shape) and tuple(val.shape[-len(spatial_shape) :]) == spatial_shape:
            out[name] = transform(val)
        else:
            out[name] = val
    return out


def symmetric_forward_2d(encoder: nn.Module, x: torch.Tensor, return_var: bool = False, **kwargs):
    """
    Four-way rotational group average for dense 2D spatial features.

    Inputs and tensor kwargs with matching HxW spatial shape are rotated together.
    Encoder outputs are inverse-rotated before averaging, preserving the original
    feature-map layout.

    When return_var=True, also returns the per-pixel variance across the 4 rotation
    views as a regularisation signal (shape matches the averaged output).
    """
    if x.ndim < 4:
        raise ValueError(f"symmetric_forward_2d expects at least 4D input, got {tuple(x.shape)}")

    spatial_shape = tuple(x.shape[-2:])
    accum = None
    sq_accum = None
    for k in range(4):
        rot = lambda t, kk=k: torch.rot90(t, k=kk, dims=(-2, -1))
        x_rot = rot(x)
        kw = _transform_tensor_kwargs(kwargs, spatial_shape, rot)
        with _symmetric_cache_pass(encoder, enabled=(k == 0)):
            feat = encoder(x_rot, **kw)
        feat = torch.rot90(feat, k=-k, dims=(-2, -1))
        accum = feat if accum is None else accum + feat
        if return_var:
            sq_accum = feat.pow(2) if sq_accum is None else sq_accum + feat.pow(2)
    avg = accum / 4.0
    if return_var:
        var = (sq_accum / 4.0) - avg.pow(2)
        var = var.clamp(min=0.0)
        return avg, var
    return avg


def symmetric_forward_3d(encoder: nn.Module, x: torch.Tensor, return_var: bool = False, **kwargs):
    """
    Eight-way flip group average for dense 3D spatial features.

    Flips are involutions, so the same flip axes align encoder outputs back to
    the original D/H/W layout.
    """
    if x.ndim < 5:
        raise ValueError(f"symmetric_forward_3d expects at least 5D input, got {tuple(x.shape)}")

    spatial_shape = tuple(x.shape[-3:])
    spatial_dims = (-3, -2, -1)
    accum = None
    sq_accum = None
    count = 0
    for r in range(len(spatial_dims) + 1):
        for dims in itertools.combinations(spatial_dims, r):
            if dims:
                flip = lambda t, dd=dims: torch.flip(t, dims=dd)
                x_flip = flip(x)
                kw = _transform_tensor_kwargs(kwargs, spatial_shape, flip)
                feat = encoder(x_flip, **kw)
                feat = flip(feat)
            else:
                feat = encoder(x, **kwargs)
            accum = feat if accum is None else accum + feat
            if return_var:
                sq_accum = feat.pow(2) if sq_accum is None else sq_accum + feat.pow(2)
            count += 1
    avg = accum / float(count)
    if return_var:
        var = (sq_accum / float(count)) - avg.pow(2)
        return avg, var.clamp(min=0.0)
    return avg
