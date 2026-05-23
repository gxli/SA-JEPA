from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_blur_batch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    x: B x 1 x H x W
    """
    radius = max(1, int(round(3.0 * sigma)))
    size = 2 * radius + 1

    coords = torch.arange(size, dtype=x.dtype, device=x.device) - radius
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g = g / g.sum()

    kernel = torch.outer(g, g)
    kernel = kernel / kernel.sum()

    channels = x.shape[1]
    weight = kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)

    return F.conv2d(x, weight, padding=radius, groups=channels)


def norm_per_sample_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize each [H,W] map independently per sample+channel.
    x: B x S x H x W
    """
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    return (x - mean) / std


def _shared_grid_centers(
    h: int,
    w: int,
    base_margin: int,
    spacing_px: int,
    global_shift: bool,
    device: torch.device,
    forced_shift_y: Optional[int] = None,
    forced_shift_x: Optional[int] = None,
    enable_grid_jitter: bool = True,
):
    """Generate one globally-shifted full-image lattice, then boundary-mask it."""
    spacing_px = int(max(1, spacing_px))
    if forced_shift_y is not None and forced_shift_x is not None:
        shift_y = int(forced_shift_y)
        shift_x = int(forced_shift_x)
    elif global_shift:
        shift_y = int(torch.randint(0, spacing_px, (1,), device=device).item())
        shift_x = int(torch.randint(0, spacing_px, (1,), device=device).item())
    else:
        shift_y = 0
        shift_x = 0

    y_centers = list(range(shift_y % spacing_px, h, spacing_px))
    x_centers = list(range(shift_x % spacing_px, w, spacing_px))
    if len(y_centers) == 0:
        y_centers = [h // 2]
    if len(x_centers) == 0:
        x_centers = [w // 2]

    raw_centers = [(cy, cx) for cy in y_centers for cx in x_centers]
    if enable_grid_jitter:
        grid_dy = (float(torch.rand(1, device=device).item()) - 0.5) * float(spacing_px)
        grid_dx = (float(torch.rand(1, device=device).item()) - 0.5) * float(spacing_px)
    else:
        grid_dy = 0.0
        grid_dx = 0.0

    shared_centers = []
    for cy, cx in raw_centers:
        jy = int(round(float(cy) + grid_dy))
        jx = int(round(float(cx) + grid_dx))
        jy = int(min(h - 1, max(0, jy)))
        jx = int(min(w - 1, max(0, jx)))
        edge_dist = min(jy, (h - 1) - jy, jx, (w - 1) - jx)
        if edge_dist < int(base_margin):
            continue
        shared_centers.append((jy, jx))

    if len(shared_centers) == 0:
        y_min = int(base_margin)
        y_max = int(max(y_min, h - 1 - int(base_margin)))
        x_min = int(base_margin)
        x_max = int(max(x_min, w - 1 - int(base_margin)))
        for cy, cx in raw_centers:
            iy = int(min(y_max, max(y_min, int(cy))))
            ix = int(min(x_max, max(x_min, int(cx))))
            shared_centers.append((iy, ix))

    return shared_centers


def make_pyramid_grid_context(
    x_clean: torch.Tensor,
    sigmas=(2, 4, 8, 16),
    cell_sizes=(16, 32, 64, 128),
    mask_fraction: float = 1.0,
    box_sigma_mult: float = 4.0,
    mask_scale: float = 1.0,
    min_mask_scale: float = 0.0,
    spacing_scale: float = 1.5,
    mask_size: float = 0.0,
    full_grid: bool = True,
    global_shift: bool = True,
    align_scales: bool = True,
    constant_mask_box: bool = True,
    mask_box_size: int = 16,
    blur_mode: str = "gaussian",
    cdd_mode: str = "log",
    cdd_constrained: bool = True,
    cdd_sm_mode: str = "reflect",
    mask_fill_mode: str = "zero",
    dip_sigma_mult: float = 1.0,
    constant_gaussian_sigma: float = 1.0,
    scaleaware_gaussian_ratios=None,
    cdd_append_last_residual: bool = True,
    inner_target_size: int = 2,
    return_debug: bool = False,
    forced_grid_shift: Optional[Tuple[int, int]] = None,
    enable_grid_jitter: bool = True,
    target_invalid_region_skip: bool = False,
    target_invalid_region_values=(0.0, "nan"),
    invalid_pixel_mask: Optional[torch.Tensor] = None,
):
    """
    x_clean: B x 1 x H x W

    Returns:
        x_context: B x 1 x H x W
        target_locations: B x K x 2, storing y,x centers
        target_scales: B x K, storing sigma per location
    """
    if x_clean.dim() != 4:
        raise ValueError(f"Expected BxCxHxW, got {tuple(x_clean.shape)}")

    if x_clean.shape[1] != 1:
        raise ValueError(f"Expected grayscale input with 1 channel, got {x_clean.shape[1]}")
    if blur_mode not in ("gaussian", "cdd"):
        raise ValueError(f"Unsupported blur_mode: {blur_mode}")
    if mask_fill_mode not in ("zero", "gaussian_dip", "constant_gaussian", "gaussian_scaleaware"):
        raise ValueError(
            f"Unsupported mask_fill_mode: {mask_fill_mode}. "
            "Use 'zero', 'gaussian_dip', 'constant_gaussian', or 'gaussian_scaleaware'."
        )

    b, _, h, w = x_clean.shape
    active_sigmas = tuple(float(s) for s in sigmas)
    active_cells = tuple(int(c) for c in cell_sizes)
    if len(active_cells) < len(active_sigmas):
        reps = (len(active_sigmas) + len(active_cells) - 1) // len(active_cells)
        active_cells = (active_cells * reps)[:len(active_sigmas)]
    elif len(active_cells) > len(active_sigmas):
        active_cells = active_cells[:len(active_sigmas)]

    n_sigmas = len(active_sigmas)
    per_scale_fraction = float(mask_fraction) / max(1, n_sigmas)

    x_context = x_clean.clone()
    x_context_np = x_context.cpu().numpy()
    invalid_value_specs = tuple(target_invalid_region_values) if target_invalid_region_values is not None else tuple()

    all_locations = []
    all_scales = []
    all_valid = []
    all_mask_maps = []
    all_unique_centers = []
    all_cdd_orig = []
    all_cdd_masked = []
    all_dip_fields = []
    all_dip_fields_per_channel = []
    all_dip_proto_per_channel = []

    for bi in range(b):
        arr = x_context_np[bi, 0].copy()
        sample_invalid_mask = invalid_pixel_mask[bi, 0].cpu().numpy() if invalid_pixel_mask is not None else None

        applied_locations = []
        applied_scales = []
        applied_mask_hard = np.zeros((h, w), dtype=np.uint8)

        # Compute shared grid centers for scale alignment
        base_margin = int(max(1.0, max(sigmas) * mask_scale * spacing_scale))
        base_box = int(max(2.0, max(sigmas) * mask_scale))
        spacing_px = int(max(1, round(float(base_box) * float(spacing_scale))))
        shared_centers = _shared_grid_centers(
            h=h,
            w=w,
            base_margin=base_margin,
            spacing_px=spacing_px,
            global_shift=global_shift,
            device=x_clean.device,
            forced_shift_y=(None if forced_grid_shift is None else int(forced_grid_shift[0])),
            forced_shift_x=(None if forced_grid_shift is None else int(forced_grid_shift[1])),
            enable_grid_jitter=bool(enable_grid_jitter),
        )
        if not full_grid:
            base_budget = per_scale_fraction * float(h * w)
            base_desired = base_budget / max(1.0, float(base_box * base_box))
            base_count = int(math.floor(base_desired))
            base_extra = int(torch.rand(1, device=x_clean.device).item() < float(base_desired - base_count))
            base_max_count = max(0, base_count + base_extra)
            if len(shared_centers) > base_max_count:
                idx = torch.randperm(len(shared_centers), device=x_clean.device)[:base_max_count]
                shared_centers = [shared_centers[int(i)] for i in idx]

        for si, sigma in enumerate(active_sigmas):
            eff_sigma = max(float(sigma), float(min_mask_scale))
            # Ensure masked region always includes the target patch:
            min_box = int(max(2, round(float(sigma) * float(mask_fraction)), int(inner_target_size)))
            if constant_mask_box:
                base_box = int(mask_box_size)
            else:
                base_box = int(max(2, round(float(sigma) * float(mask_fraction) + float(mask_size))))
            box = int(max(base_box, min_box))
            half_lo = box // 2
            half_hi = box - half_lo
            if align_scales:
                centers = shared_centers
            else:
                spacing = int(max(1, round(float(box) * float(spacing_scale))))
                margin = max(half_lo, half_hi) + 1
                area_budget = per_scale_fraction * float(h * w)
                desired_count = area_budget / max(1.0, float(box * box))
                base_count = int(math.floor(desired_count))
                frac = float(desired_count - base_count)
                extra = int(torch.rand(1, device=x_clean.device).item() < frac)
                max_count = max(0, base_count + extra)
                if max_count <= 0:
                    continue
                if forced_grid_shift is not None:
                    shift_y = int(forced_grid_shift[0]) % max(1, spacing)
                    shift_x = int(forced_grid_shift[1]) % max(1, spacing)
                else:
                    shift_y = int(torch.randint(0, max(1, spacing), (1,), device=x_clean.device).item())
                    shift_x = int(torch.randint(0, max(1, spacing), (1,), device=x_clean.device).item())
                y_start = margin + shift_y
                x_start = margin + shift_x
                y_centers = list(range(y_start, max(y_start + 1, h - margin), spacing))
                x_centers = list(range(x_start, max(x_start + 1, w - margin), spacing))
                centers = [(cy, cx) for cy in y_centers for cx in x_centers]
                if len(centers) > max_count:
                    idx = torch.randperm(len(centers), device=x_clean.device)[:max_count]
                    centers = [centers[int(i)] for i in idx]

            for cy, cx in centers:
                y0 = max(0, cy - half_lo)
                y1 = min(h, cy + half_hi)
                x0 = max(0, cx - half_lo)
                x1 = min(w, cx + half_hi)
                if y1 <= y0 or x1 <= x0:
                    continue
                if mask_fill_mode in ("gaussian_dip", "constant_gaussian", "gaussian_scaleaware"):
                    if mask_fill_mode == "constant_gaussian":
                        blur_sigma = max(1e-6, float(constant_gaussian_sigma))
                    elif mask_fill_mode == "gaussian_scaleaware":
                        if scaleaware_gaussian_ratios is None or len(scaleaware_gaussian_ratios) == 0:
                            ratio = 1.0
                        else:
                            ratio = float(scaleaware_gaussian_ratios[min(si, len(scaleaware_gaussian_ratios) - 1)])
                        blur_sigma = max(1e-6, float(sigma) * float(mask_fraction) * float(dip_sigma_mult) * ratio)
                    else:
                        blur_sigma = max(1e-6, float(sigma) * float(mask_fraction) * float(dip_sigma_mult))
                    yy = np.arange(y0, y1, dtype=np.float32).reshape(-1, 1)
                    xx = np.arange(x0, x1, dtype=np.float32).reshape(1, -1)
                    dist2 = ((yy - float(cy)) / blur_sigma) ** 2 + ((xx - float(cx)) / blur_sigma) ** 2
                    g = np.exp(-0.5 * dist2).astype(np.float32)
                    arr[y0:y1, x0:x1] = arr[y0:y1, x0:x1] * (1.0 - g)
                else:
                    arr[y0:y1, x0:x1] = 0.0
                applied_mask_hard[y0:y1, x0:x1] = 1
                applied_locations.append((cy, cx))
                applied_scales.append(float(sigma))

        # --- CDD mode branch ---
        if blur_mode == "cdd":
            import constrained_diffusion as cdd

            cdd_kwargs = dict(
                mode=cdd_mode,
                constrained=bool(cdd_constrained),
                sm_mode=cdd_sm_mode,
                return_scales=False,
                verbose=False,
                use_gpu=False,
            )
            cdd_channels_arr, cdd_residual = cdd.constrained_diffusion_decomposition(
                arr.astype(np.float32),
                scales=tuple(float(s) for s in active_sigmas),
                **cdd_kwargs,
            )
            cdd_channels_arr = np.asarray(cdd_channels_arr, dtype=np.float32)
            cdd_residual = np.asarray(cdd_residual, dtype=np.float32)

            cdd_orig = np.clip(np.asarray(cdd_channels_arr, dtype=np.float32), a_min=0.0, a_max=None)

            cdd_mod = cdd_channels_arr.copy()

            if cdd_append_last_residual:
                cdd_orig[-1] = cdd_orig[-1] + cdd_residual
                cdd_mod[-1] = cdd_mod[-1] + cdd_residual

            all_cdd_orig.append(torch.from_numpy(cdd_orig.copy()))

            num_cdd_ch = cdd_mod.shape[0]
            dip_field = np.zeros((h, w), dtype=np.float32)
            dip_field_ch = np.zeros((num_cdd_ch, h, w), dtype=np.float32)
            dip_proto_ch = np.zeros((num_cdd_ch, h, w), dtype=np.float32)
            dip_proto_written = np.zeros(num_cdd_ch, dtype=np.int32)

            for si, sigma in enumerate(active_sigmas):
                eff_sigma = max(float(sigma), float(min_mask_scale))
                base_box = int(max(2, round(float(eff_sigma) * float(mask_fraction))))
                box = int(max(2, base_box))
                ch = min(si, cdd_mod.shape[0] - 1)
                half_lo = box // 2
                half_hi = box - half_lo
                if align_scales:
                    centers = shared_centers
                else:
                    spacing = int(max(1, round(float(box) * float(spacing_scale))))
                    margin = max(half_lo, half_hi) + 1
                    area_budget = per_scale_fraction * float(h * w)
                    desired_count = area_budget / max(1.0, float(box * box))
                    base_count = int(math.floor(desired_count))
                    frac = float(desired_count - base_count)
                    extra = int(torch.rand(1, device=x_clean.device).item() < frac)
                    max_count = max(0, base_count + extra)
                    if max_count <= 0:
                        continue
                    if forced_grid_shift is not None:
                        shift_y = int(forced_grid_shift[0]) % max(1, spacing)
                        shift_x = int(forced_grid_shift[1]) % max(1, spacing)
                    else:
                        shift_y = int(torch.randint(0, max(1, spacing), (1,), device=x_clean.device).item())
                        shift_x = int(torch.randint(0, max(1, spacing), (1,), device=x_clean.device).item())
                    y_start = margin + shift_y
                    x_start = margin + shift_x
                    y_centers = list(range(y_start, max(y_start + 1, h - margin), spacing))
                    x_centers = list(range(x_start, max(x_start + 1, w - margin), spacing))
                    centers = [(cy, cx) for cy in y_centers for cx in x_centers]
                    if len(centers) > max_count:
                        idx = torch.randperm(len(centers), device=x_clean.device)[:max_count]
                        centers = [centers[int(i)] for i in idx]

                for cy, cx in centers:
                    y0 = max(0, cy - half_lo)
                    y1 = min(h, cy + half_hi)
                    x0 = max(0, cx - half_lo)
                    x1 = min(w, cx + half_hi)
                    if y1 <= y0 or x1 <= x0:
                        continue
                    cdd_mod[ch, y0:y1, x0:x1] = 0.0
                    dip_field[y0:y1, x0:x1] = 1.0
                    dip_field_ch[ch, y0:y1, x0:x1] = 1.0
                    if dip_proto_written[ch] == 0:
                        dip_proto_ch[ch, y0:y1, x0:x1] = 1.0
                        dip_proto_written[ch] = 1
                    applied_mask_hard[y0:y1, x0:x1] = 1
                    applied_locations.append((cy, cx))
                    applied_scales.append(float(sigma))

            recon = np.sum(cdd_mod, axis=0) + cdd_residual
            recon = np.clip(recon, a_min=0.0, a_max=None)
            x_context[bi, 0] = torch.from_numpy(recon).to(device=x_clean.device, dtype=x_clean.dtype)

            sample_locations = list(applied_locations)
            sample_scales = list(applied_scales)
            unique_loc_to_scale = {}
            for (cy, cx), s in zip(sample_locations, sample_scales):
                key = (int(cy), int(cx))
                if key not in unique_loc_to_scale:
                    unique_loc_to_scale[key] = float(s)
            sample_locations = []
            sample_scales = []
            patch_half_lo = int(inner_target_size) // 2
            patch_half_hi = int(inner_target_size) - patch_half_lo
            for cy, cx in unique_loc_to_scale.keys():
                iy = int(cy)
                ix = int(cx)
                if iy - patch_half_lo < 0 or ix - patch_half_lo < 0:
                    continue
                if iy + patch_half_hi > h or ix + patch_half_hi > w:
                    continue
                if bool(target_invalid_region_skip):
                    py0 = iy - patch_half_lo
                    py1 = iy + patch_half_hi
                    px0 = ix - patch_half_lo
                    px1 = ix + patch_half_hi
                    patch = arr[py0:py1, px0:px1]
                    if patch.size == 0:
                        continue
                    invalid_mask = np.zeros_like(patch, dtype=bool)
                    if sample_invalid_mask is not None:
                        invalid_mask |= sample_invalid_mask[py0:py1, px0:px1]
                    for spec in invalid_value_specs:
                        if isinstance(spec, str) and spec.lower() == "nan":
                            invalid_mask |= np.isnan(patch)
                        else:
                            try:
                                invalid_mask |= np.isclose(patch, float(spec), equal_nan=False)
                            except (TypeError, ValueError):
                                continue
                    if np.all(invalid_mask):
                        continue
                sample_locations.append((iy, ix))
                sample_scales.append(float(unique_loc_to_scale[(cy, cx)]))
            sample_valid = [1] * len(sample_locations)

            all_locations.append(sample_locations)
            all_scales.append(sample_scales)
            all_valid.append(sample_valid)
            if return_debug:
                uniq = []
                seen = set()
                for cy, cx in applied_locations:
                    key = (int(cy), int(cx))
                    if key not in seen:
                        seen.add(key)
                        uniq.append((cy, cx))
                all_unique_centers.append(torch.tensor(uniq, dtype=torch.long))
            all_mask_maps.append(torch.from_numpy(applied_mask_hard.copy()))
            all_cdd_masked.append(torch.from_numpy(np.clip(cdd_mod, a_min=0.0, a_max=None)))
            all_dip_fields.append(torch.from_numpy(dip_field))
            all_dip_fields_per_channel.append(torch.from_numpy(dip_field_ch))
            all_dip_proto_per_channel.append(torch.from_numpy(dip_proto_ch))

            continue

        # --- Gaussian mode target sampling ---
        sample_locations = list(applied_locations)
        sample_scales = list(applied_scales)
        unique_loc_to_scale = {}
        for (cy, cx), s in zip(sample_locations, sample_scales):
            key = (int(cy), int(cx))
            if key not in unique_loc_to_scale:
                unique_loc_to_scale[key] = float(s)
        sample_locations = []
        sample_scales = []
        patch_half_lo = int(inner_target_size) // 2
        patch_half_hi = int(inner_target_size) - patch_half_lo
        for cy, cx in unique_loc_to_scale.keys():
            iy = int(cy)
            ix = int(cx)
            if iy - patch_half_lo < 0 or ix - patch_half_lo < 0:
                continue
            if iy + patch_half_hi > h or ix + patch_half_hi > w:
                continue
            if bool(target_invalid_region_skip):
                py0 = iy - patch_half_lo
                py1 = iy + patch_half_hi
                px0 = ix - patch_half_lo
                px1 = ix + patch_half_hi
                patch = arr[py0:py1, px0:px1]
                if patch.size == 0:
                    continue
                invalid_mask = np.zeros_like(patch, dtype=bool)
                if sample_invalid_mask is not None:
                    invalid_mask |= sample_invalid_mask[py0:py1, px0:px1]
                for spec in invalid_value_specs:
                    if isinstance(spec, str) and spec.lower() == "nan":
                        invalid_mask |= np.isnan(patch)
                    else:
                        try:
                            invalid_mask |= np.isclose(patch, float(spec), equal_nan=False)
                        except (TypeError, ValueError):
                            continue
                if np.all(invalid_mask):
                    continue
            sample_locations.append((iy, ix))
            sample_scales.append(float(unique_loc_to_scale[(cy, cx)]))
        sample_valid = [1] * len(sample_locations)

        # Write masked NumPy array back to the context tensor.
        x_context[bi, 0] = torch.from_numpy(arr).to(device=x_context.device, dtype=x_context.dtype)

        all_locations.append(sample_locations)
        all_scales.append(sample_scales)
        all_valid.append(sample_valid)
        if return_debug:
            uniq = []
            seen = set()
            for cy, cx in applied_locations:
                key = (int(cy), int(cx))
                if key not in seen:
                    seen.add(key)
                    uniq.append((cy, cx))
            all_unique_centers.append(torch.tensor(uniq, dtype=torch.long))
        all_mask_maps.append(torch.from_numpy(applied_mask_hard.copy()))

    # Pack variable-length targets to fixed K so batching is always valid.
    k_fixed = max((len(v) for v in all_locations), default=0)
    k_fixed = max(1, k_fixed)

    loc_np = np.zeros((b, k_fixed, 2), dtype=np.int64)
    sca_np = np.zeros((b, k_fixed), dtype=np.float32)
    val_np = np.zeros((b, k_fixed), dtype=np.bool_)

    for bi in range(b):
        n_total = len(all_locations[bi])
        n = min(n_total, k_fixed)
        if n <= 0:
            continue
        loc_np[bi, :n, :] = np.asarray(all_locations[bi][:n], dtype=np.int64)
        sca_np[bi, :n] = np.asarray(all_scales[bi][:n], dtype=np.float32)
        val_np[bi, :n] = True

    target_locations = torch.from_numpy(loc_np).to(device=x_clean.device, dtype=torch.long)
    target_scales = torch.from_numpy(sca_np).to(device=x_clean.device, dtype=x_clean.dtype)
    target_valid = torch.from_numpy(val_np).to(device=x_clean.device, dtype=torch.bool)

    if not return_debug:
        return x_context, target_locations, target_scales, target_valid

    max_centers = max((int(t.shape[0]) for t in all_unique_centers), default=0)
    centers_pad = torch.full((b, max_centers, 2), -1, dtype=torch.long, device=x_clean.device)
    for bi, t in enumerate(all_unique_centers):
        if t.numel() > 0:
            centers_pad[bi, :t.shape[0]] = t.to(device=x_clean.device)

    def _safe_stack(tensor_list):
        if not tensor_list:
            return torch.empty(0, device=x_clean.device, dtype=x_clean.dtype)
        return torch.stack([t.to(device=x_clean.device, dtype=x_clean.dtype) for t in tensor_list], dim=0)

    debug = {
        "mask_map": torch.stack([m.to(device=x_clean.device) for m in all_mask_maps], dim=0),
        "unique_centers": centers_pad,
        "cdd_channels_orig": _safe_stack(all_cdd_orig),
        "cdd_channels_masked": _safe_stack(all_cdd_masked),
        "dip_field": _safe_stack(all_dip_fields),
        "dip_field_per_channel": _safe_stack(all_dip_fields_per_channel),
        "dip_proto_per_channel": _safe_stack(all_dip_proto_per_channel),
    }
    return x_context, target_locations, target_scales, target_valid, debug


def extract_location_patches(
    z: torch.Tensor,
    locations: torch.Tensor,
    patch_size: int,
):
    """
    z:         B x C x H x W
    locations: B x K x 2, y/x centers

    Returns:
        patches: B x K x C x patch_size x patch_size
    """
    b, _, h, w = z.shape
    _, k, _ = locations.shape

    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if patch_size > h or patch_size > w:
        raise ValueError(f"patch_size={patch_size} exceeds feature map size {(h, w)}")

    half = patch_size // 2
    half_hi = patch_size - half
    patches = []

    for bi in range(b):
        sample_patches = []

        for ki in range(k):
            cy = int(locations[bi, ki, 0].item())
            cx = int(locations[bi, ki, 1].item())

            y0 = cy - half
            y1 = cy + half_hi
            x0 = cx - half
            x1 = cx + half_hi
            if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                sample_patches.append(z.new_zeros((1, z.shape[1], patch_size, patch_size)))
            else:
                sample_patches.append(z[bi : bi + 1, :, y0:y1, x0:x1])

        sample_patches = torch.cat(sample_patches, dim=0)
        patches.append(sample_patches.unsqueeze(0))

    return torch.cat(patches, dim=0)
