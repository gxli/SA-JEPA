from __future__ import annotations

import torch


def sample_target_locations_3d(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    num_targets: int,
    patch_size: int,
    device,
):
    half = patch_size // 2
    lo_z = half
    hi_z = depth - (patch_size - half)
    lo_y = half
    hi_y = height - (patch_size - half)
    lo_x = half
    hi_x = width - (patch_size - half)

    if hi_z < lo_z or hi_y < lo_y or hi_x < lo_x:
        raise ValueError("Patch too large for volume")

    z = torch.randint(lo_z, hi_z + 1, (batch_size, num_targets), device=device)
    y = torch.randint(lo_y, hi_y + 1, (batch_size, num_targets), device=device)
    x = torch.randint(lo_x, hi_x + 1, (batch_size, num_targets), device=device)

    loc = torch.stack([z, y, x], dim=-1)
    valid = torch.ones((batch_size, num_targets), dtype=torch.bool, device=device)
    return loc, valid


def extract_location_cubes(z: torch.Tensor, locations: torch.Tensor, patch_size: int):
    if z.ndim != 5:
        raise ValueError(f"Expected z B,C,D,H,W, got {tuple(z.shape)}")

    b, c, d, h, w = z.shape
    _, k, ndim = locations.shape
    if ndim != 3:
        raise ValueError(f"Expected locations B,K,3, got {tuple(locations.shape)}")

    p = int(patch_size)
    if p <= 0:
        raise ValueError(f"patch_size must be positive, got {p}")
    if p > d or p > h or p > w:
        raise ValueError(f"patch_size={p} exceeds feature map size {(d, h, w)}")

    half = p // 2
    z0 = locations[:, :, 0] - half
    y0 = locations[:, :, 1] - half
    x0 = locations[:, :, 2] - half

    valid = (
        (z0 >= 0)
        & (y0 >= 0)
        & (x0 >= 0)
        & (z0 + p <= d)
        & (y0 + p <= h)
        & (x0 + p <= w)
    )

    dz = torch.arange(p, device=z.device)
    dy = torch.arange(p, device=z.device)
    dx = torch.arange(p, device=z.device)

    zz = z0.view(b, k, 1, 1, 1) + dz.view(1, 1, p, 1, 1)
    yy = y0.view(b, k, 1, 1, 1) + dy.view(1, 1, 1, p, 1)
    xx = x0.view(b, k, 1, 1, 1) + dx.view(1, 1, 1, 1, p)

    zz = zz.clamp(0, d - 1)
    yy = yy.clamp(0, h - 1)
    xx = xx.clamp(0, w - 1)

    b_idx = torch.arange(b, device=z.device).view(b, 1, 1, 1, 1, 1)
    c_idx = torch.arange(c, device=z.device).view(1, 1, c, 1, 1, 1)

    zz = zz.unsqueeze(2)
    yy = yy.unsqueeze(2)
    xx = xx.unsqueeze(2)

    cubes = z[b_idx, c_idx, zz, yy, xx]
    valid_mask = valid.view(b, k, 1, 1, 1, 1)
    cubes = torch.where(valid_mask, cubes, torch.zeros_like(cubes))
    return cubes
