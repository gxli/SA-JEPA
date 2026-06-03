from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch


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
    enable_grid_jitter: bool = True,
    lattice_shift_override: Optional[Tuple[int, int]] = None,
):
    """Generate one globally-shifted full-image lattice, then boundary-mask it."""
    spacing_px = int(max(1, spacing_px))
    if lattice_shift_override is not None:
        shift_y = int(lattice_shift_override[0]) % spacing_px
        shift_x = int(lattice_shift_override[1]) % spacing_px
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


def _build_priority_catalogue_from_cdd_ratio(
    cdd_orig: np.ndarray,
    top_percent: float,
    patch_size: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    """Rank pixels by (sum of two smallest scales) / (total flux)."""
    if cdd_orig.ndim != 3 or cdd_orig.shape[0] <= 0:
        return []
    if cdd_orig.shape[0] == 1:
        numerator = cdd_orig[0]
    else:
        numerator = cdd_orig[0] + cdd_orig[1]
    total_flux = np.sum(cdd_orig, axis=0)
    denom = np.maximum(total_flux, 1e-8)
    ratio = np.nan_to_num(numerator / denom, nan=0.0, posinf=0.0, neginf=0.0)

    half_lo = int(patch_size) // 2
    half_hi = int(patch_size) - half_lo
    valid = np.ones((h, w), dtype=bool)
    # Empty background pixels have tied zero ratios and should never enter a
    # priority catalogue merely because top_percent exceeds the signal area.
    valid &= total_flux > 1e-8
    if half_lo > 0:
        valid[:half_lo, :] = False
        valid[:, :half_lo] = False
    if half_hi > 0:
        valid[h - half_hi :, :] = False
        valid[:, w - half_hi :] = False
    valid_idx = np.flatnonzero(valid.reshape(-1))
    if valid_idx.size == 0:
        return []

    pct = float(np.clip(top_percent, 0.0, 100.0))
    k = int(math.ceil((pct / 100.0) * float(valid_idx.size)))
    k = max(1, min(k, int(valid_idx.size)))
    valid_ratio = ratio.reshape(-1)[valid_idx]
    top_local = np.argpartition(valid_ratio, -k)[-k:]
    top_local = top_local[np.argsort(valid_ratio[top_local])[::-1]]
    selected = valid_idx[top_local]
    ys = (selected // w).astype(np.int64)
    xs = (selected % w).astype(np.int64)
    return [(int(y), int(x)) for y, x in zip(ys, xs)]


def _fractional_spatial_target_budget(
    height: int,
    width: int,
    box_size: int,
    oversample: float,
    device: torch.device,
    minimum: int = 0,
) -> int | None:
    """Budget candidates as f * image_area / box_area with stochastic rounding."""
    f = float(oversample)
    if f <= 0.0:
        return None
    box = max(1, int(box_size))
    desired = f * float(max(1, int(height)) * max(1, int(width))) / float(box * box)
    base = int(math.floor(desired))
    frac = float(desired - base)
    extra = int(torch.rand(1, device=device).item() < frac)
    return max(int(minimum), base + extra)


def _dither_target_center(
    cy: int,
    cx: int,
    h: int,
    w: int,
    half_lo: int,
    half_hi: int,
    dithering_pixels: int,
    device: torch.device,
) -> tuple[int, int]:
    """Jitter a center within a local square and keep target patch in-bounds."""
    d = int(max(0, dithering_pixels))
    if d <= 1:
        return int(cy), int(cx)
    max_off = d // 2
    dy = int(torch.randint(-max_off, max_off + 1, (1,), device=device).item())
    dx = int(torch.randint(-max_off, max_off + 1, (1,), device=device).item())
    cy2 = int(cy) + dy
    cx2 = int(cx) + dx
    y_min = int(half_lo)
    y_max = int(max(y_min, h - half_hi))
    x_min = int(half_lo)
    x_max = int(max(x_min, w - half_hi))
    cy2 = int(min(y_max, max(y_min, cy2)))
    cx2 = int(min(x_max, max(x_min, cx2)))
    return cy2, cx2


def _odd_box(v: int, minimum: int = 3, bump_up: bool = True) -> int:
    x = int(max(minimum, v))
    if x % 2 == 0:
        x += 1 if bump_up else -1
    x = max(x, minimum)
    if x % 2 == 0:
        x += 1
    return x


def _rejection_sample_targets(
    candidates: list[tuple[int, int]],
    num_targets: int,
    h: int,
    w: int,
    exclusion_box: int,
    device: torch.device,
    max_tries: int = 4096,
    allow_partial_overlap: float = 0.0,
) -> list[tuple[int, int]]:
    """Select non-overlapping targets via occupancy-map rejection sampling.

    The exclusion footprint is a square of size exclusion_box centered on
    each accepted target.  This footprint protects the *mask* from
    overlapping, not just the inner target patch.
    """
    if exclusion_box <= 0 or len(candidates) == 0:
        return candidates[:num_targets]

    half = exclusion_box // 2
    occ = torch.zeros((h, w), dtype=torch.bool, device=device)
    accepted: list[tuple[int, int]] = []
    tries = 0

    # Shuffle candidates for unbiased selection.
    perm = torch.randperm(len(candidates), device=device)
    idx = 0
    while len(accepted) < num_targets and tries < max_tries and idx < len(candidates):
        tries += 1
        cy, cx = candidates[int(perm[idx])]
        idx += 1

        y0 = max(0, int(cy) - half)
        y1 = min(h, int(cy) + exclusion_box - half)
        x0 = max(0, int(cx) - half)
        x1 = min(w, int(cx) + exclusion_box - half)
        if y1 <= y0 or x1 <= x0:
            continue

        footprint = occ[y0:y1, x0:x1]
        overlap_frac = float(footprint.float().mean().item())
        if overlap_frac <= float(allow_partial_overlap):
            accepted.append((int(cy), int(cx)))
            occ[y0:y1, x0:x1] = True

    return accepted


def _effective_mask_box_size(
    sigma: float,
    mask_scale: float,
    mask_box_size: int,
    inner_target_size: int,
    hardcap: int | None = None,
) -> int:
    """Compute mask box size: round(sigma * mask_scale + mask_box_size).

    mask_scale=0  -> constant box (mask_box_size across all scales).
    mask_box_size=0 -> pure pyramid (box proportional to sigma).
    hardcap clamps the result to a maximum (None or 0 = no cap).
    """
    if hardcap is not None and hardcap > 0 and int(hardcap) < int(inner_target_size):
        raise ValueError(
            f"mask_box_hardcap={int(hardcap)} cannot be smaller than "
            f"inner_target_size={int(inner_target_size)}"
        )
    base_box = round(float(sigma) * float(mask_scale) + int(mask_box_size))
    box = max(base_box, int(inner_target_size))
    capped = False
    if hardcap is not None and hardcap > 0:
        if box > int(hardcap):
            box = int(hardcap)
            capped = True
    # bump_up=False when capped: round DOWN to the nearest odd so we never
    # silently exceed the hardcap (e.g. hardcap=16 → _odd_box(16, bump_up=False)=15).
    return _odd_box(box, bump_up=not capped)


def _ensure_target_patches_masked(
    sample_locations: list[tuple[int, int]],
    mask_map: np.ndarray,
    inner_target_size: int,
) -> None:
    patch_half_lo = int(inner_target_size) // 2
    patch_half_hi = int(inner_target_size) - patch_half_lo
    h, w = mask_map.shape
    for cy, cx in sample_locations:
        y0 = max(0, int(cy) - patch_half_lo)
        y1 = min(h, int(cy) + patch_half_hi)
        x0 = max(0, int(cx) - patch_half_lo)
        x1 = min(w, int(cx) + patch_half_hi)
        if y1 <= y0 or x1 <= x0:
            continue
        if not np.all(mask_map[y0:y1, x0:x1] > 0):
            raise RuntimeError(
                "Target patch is not fully covered by the mask footprint; "
                f"center={(int(cy), int(cx))}, patch_size={int(inner_target_size)}"
            )


def make_pyramid_grid_context(
    x_clean: torch.Tensor,
    sigmas=(2, 4, 8, 16),
    mask_fraction: float = 1.0,
    mask_scale: float = 1.0,
    spacing_scale: float = 1.5,
    global_shift: bool = True,
    align_scales: bool = True,
    mask_box_size: int = 16,
    cdd_mode: str = "log",
    cdd_constrained: bool = True,
    cdd_sm_mode: str = "reflect",
    cdd_append_last_residual: bool = True,
    inner_target_size: int = 2,
    return_debug: bool = False,
    enable_grid_jitter: bool = True,
    enable_target_dithering: bool = True,
    lattice_shift_override: Optional[Tuple[int, int]] = None,
    target_invalid_region_skip: bool = False,
    target_invalid_region_values=(0.0, "nan"),
    invalid_pixel_mask: Optional[torch.Tensor] = None,
    target_sampling_mode: str = "grid",
    priority_top_percent: float = 5.0,
    priority_n_target: int | str = 20,
    priority_min_targets_per_map: int = 0,
    priority_dithering_pixels: Optional[int] = None,
    priority_candidate_oversample: float = 3.0,
    target_nonoverlap: bool = False,
    target_allow_partial_overlap: float = 0.0,
    mask_box_hardcap: int | None = None,
    cdd_use_gpu: bool = False,
    cdd_orig_in: Optional[torch.Tensor] = None,
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
    # Targets must be odd-sized so the target location is the true center pixel.
    inner_target_size = int(inner_target_size)
    if inner_target_size <= 0:
        inner_target_size = 3
    if inner_target_size % 2 == 0:
        inner_target_size = inner_target_size + 1
    if x_clean.shape[1] != 1:
        raise ValueError(f"Expected grayscale input with 1 channel, got {x_clean.shape[1]}")
    sampling_mode = str(target_sampling_mode).strip().lower()
    if priority_dithering_pixels is None or priority_dithering_pixels <= 0:
        priority_dithering_pixels = inner_target_size
    else:
        priority_dithering_pixels = int(priority_dithering_pixels)
    # Safeguard: global_shift is a lattice/grid concept only.
    # Priority sampling selects targets from ranked pixels, so disable it.
    effective_global_shift = bool(global_shift) if sampling_mode != "priority_sampling" else False

    b, _, h, w = x_clean.shape
    active_sigmas = tuple(float(s) for s in sigmas)
    if not active_sigmas:
        raise ValueError("sigmas must contain at least one CDD scale")
    n_sigmas = len(active_sigmas)
    total_fraction = max(0.0, float(mask_fraction))
    per_scale_fraction = total_fraction / max(1, n_sigmas)

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
    all_cdd_box_sizes = []
    all_cdd_blur_sigmas = []
    all_priority_good_candidates = []
    all_priority_nonzero_mean = []
    all_priority_prescreen_candidates = []
    all_priority_auto_base_targets = []
    all_priority_effective_targets = []

    for bi in range(b):
        arr = x_context_np[bi, 0].copy()
        sample_invalid_mask = invalid_pixel_mask[bi, 0].cpu().numpy() if invalid_pixel_mask is not None else None
        priority_good_candidates_bi = 0.0
        priority_nonzero_mean_bi = 1.0
        priority_prescreen_candidates_bi = 0.0
        priority_auto_base_targets_bi = 0.0
        priority_effective_targets_bi = 0.0

        applied_locations = []
        applied_scales = []
        applied_mask_hard = np.zeros((h, w), dtype=np.uint8)

        # Compute shared grid centers for scale alignment
        max_sigma = max(float(s) for s in sigmas)
        base_box = _effective_mask_box_size(
            sigma=max_sigma,
            mask_scale=mask_scale,
            mask_box_size=mask_box_size,
            inner_target_size=inner_target_size,
            hardcap=mask_box_hardcap,
        )
        base_margin = base_box // 2 + 1
        spacing_px = int(max(1, round(float(base_box) * float(spacing_scale))))
        shared_centers = _shared_grid_centers(
            h=h,
            w=w,
            base_margin=base_margin,
            spacing_px=spacing_px,
            global_shift=effective_global_shift,
            device=x_clean.device,
            enable_grid_jitter=bool(enable_grid_jitter),
            lattice_shift_override=lattice_shift_override,
        )
        shared_centers_dithered = None
        if align_scales and sampling_mode != "priority_sampling" and len(shared_centers) > 0:
            # Dither once and reuse across scales to keep target/mask centers aligned.
            max_sigma = max(float(s) for s in active_sigmas)
            max_box = _effective_mask_box_size(
                sigma=max_sigma,
                mask_scale=mask_scale,
                mask_box_size=mask_box_size,
                inner_target_size=inner_target_size,
                hardcap=mask_box_hardcap,
            )
            max_half_lo = max_box // 2
            max_half_hi = max_box - max_half_lo
            if enable_target_dithering:
                shared_centers_dithered = []
                for cy0, cx0 in shared_centers:
                    cy1, cx1 = _dither_target_center(
                        cy=int(cy0),
                        cx=int(cx0),
                        h=h,
                        w=w,
                        half_lo=max_half_lo,
                        half_hi=max_half_hi,
                        # Grid/lattice mode always dithers by lattice spacing.
                        dithering_pixels=spacing_px,
                        device=x_clean.device,
                    )
                    shared_centers_dithered.append((int(cy1), int(cx1)))
        if total_fraction <= 0.0:
            shared_centers = []
            shared_centers_dithered = []
        # Apply count budget from mask_fraction for both full-grid and sampled-grid paths.
        if len(shared_centers) > 0:
            base_budget = per_scale_fraction * float(h * w)
            base_desired = base_budget / max(1.0, float(base_box * base_box))
            base_count = int(math.floor(base_desired))
            base_extra = int(torch.rand(1, device=x_clean.device).item() < float(base_desired - base_count))
            base_max_count = max(0, base_count + base_extra)
            if len(shared_centers) > base_max_count:
                idx = torch.randperm(len(shared_centers), device=x_clean.device)[:base_max_count]
                shared_centers = [shared_centers[int(i)] for i in idx]
                if shared_centers_dithered is not None:
                    shared_centers_dithered = [shared_centers_dithered[int(i)] for i in idx]

        # Masking is applied only after CDD decomposition.
        if active_sigmas:
            if cdd_orig_in is not None:
                # Use pre-computed CDD channels (residual already baked in).
                cdd_orig = cdd_orig_in[bi].cpu().numpy().astype(np.float32, copy=False)
                cdd_mod = cdd_orig.copy()
                cdd_residual = None  # unused — residual is in last channel
            else:
                import constrained_diffusion as cdd

                cdd_kwargs = dict(
                    mode=cdd_mode,
                    constrained=bool(cdd_constrained),
                    sm_mode=cdd_sm_mode,
                    return_scales=False,
                    verbose=False,
                    use_gpu=bool(cdd_use_gpu),
                )
                cdd_channels_arr, cdd_residual = cdd.constrained_diffusion_decomposition(
                    arr.astype(np.float32),
                    num_channels=len(active_sigmas),
                    max_scale=max(active_sigmas),
                    **cdd_kwargs,
                )
                cdd_channels_arr = np.asarray(cdd_channels_arr, dtype=np.float32)
                cdd_residual = np.asarray(cdd_residual, dtype=np.float32)

                cdd_orig = np.clip(np.asarray(cdd_channels_arr, dtype=np.float32), a_min=0.0, a_max=None)

                if cdd_append_last_residual:
                    cdd_orig[-1] = cdd_orig[-1] + cdd_residual

                # Context branch starts from the exact same clipped+residual base as target.
                cdd_mod = cdd_orig.copy()

            all_cdd_orig.append(torch.from_numpy(cdd_orig.copy()))

            priority_catalogue = []
            if sampling_mode == "priority_sampling":
                priority_catalogue = _build_priority_catalogue_from_cdd_ratio(
                    cdd_orig=cdd_orig,
                    top_percent=float(priority_top_percent),
                    patch_size=int(inner_target_size),
                    h=h,
                    w=w,
                )
                if len(priority_catalogue) > 0:
                    # Reject candidates too close to boundary for the largest
                    # possible mask footprint across pyramid scales.
                    max_box = _effective_mask_box_size(
                        sigma=max(float(s) for s in active_sigmas),
                        mask_scale=mask_scale,
                        mask_box_size=mask_box_size,
                        inner_target_size=inner_target_size,
                        hardcap=mask_box_hardcap,
                    )
                    max_half_lo = max_box // 2
                    max_half_hi = max_box - max_half_lo
                    good_candidates = []
                    for cy, cx in priority_catalogue:
                        y0 = int(cy) - int(max_half_lo)
                        y1 = int(cy) + int(max_half_hi)
                        x0 = int(cx) - int(max_half_lo)
                        x1 = int(cx) + int(max_half_hi)
                        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                            continue
                        good_candidates.append((int(cy), int(cx)))
                    priority_catalogue = good_candidates
                    prescreen_count = _fractional_spatial_target_budget(
                        height=h,
                        width=w,
                        box_size=max_box,
                        oversample=float(priority_candidate_oversample),
                        device=x_clean.device,
                        minimum=int(priority_min_targets_per_map),
                    )
                    if prescreen_count is not None and len(priority_catalogue) > prescreen_count:
                        priority_catalogue = priority_catalogue[:prescreen_count]
                    priority_prescreen_candidates_bi = float(len(priority_catalogue))

                    # Auto target-count estimate from overlap density:
                    # N_auto = (#good_candidates) / mean(nonzero(dummy_map)).
                    dummy = np.zeros((h, w), dtype=np.float32)
                    for cy, cx in priority_catalogue:
                        y0 = max(0, int(cy) - int(max_half_lo))
                        y1 = min(h, int(cy) + int(max_half_hi))
                        x0 = max(0, int(cx) - int(max_half_lo))
                        x1 = min(w, int(cx) + int(max_half_hi))
                        if y1 <= y0 or x1 <= x0:
                            continue
                        dummy[y0:y1, x0:x1] += 1.0
                    nonzero = dummy[dummy > 0]
                    nonzero_mean = float(nonzero.mean()) if nonzero.size > 0 else 1.0
                    auto_base = (
                        int(round(float(len(priority_catalogue)) / max(nonzero_mean, 1e-6)))
                        if len(priority_catalogue) > 0
                        else 0
                    )

                    priority_n_raw = priority_n_target
                    if isinstance(priority_n_raw, str) and priority_n_raw.strip().lower() == "auto":
                        base_targets_unscaled = auto_base
                    else:
                        try:
                            base_targets_unscaled = int(round(float(priority_n_raw)))
                        except (TypeError, ValueError):
                            base_targets_unscaled = 0
                    min_targets = max(0, int(priority_min_targets_per_map))
                    base_targets_scaled = max(0, int(round(float(base_targets_unscaled) * float(total_fraction))))
                    base_targets = max(min_targets, base_targets_scaled)
                    k_sel = min(base_targets, len(priority_catalogue))
                    if target_nonoverlap:
                        # Shuffle first, then non-overlap filter on undithered
                        # centres to get a reasonable initial set.  The real
                        # non-overlap enforcement happens *after* dithering below.
                        perm = torch.randperm(len(priority_catalogue), device=x_clean.device)
                        priority_catalogue = [priority_catalogue[int(i)] for i in perm[:min(k_sel * 2, len(priority_catalogue))]]
                    else:
                        perm = torch.randperm(len(priority_catalogue), device=x_clean.device)
                        priority_catalogue = [priority_catalogue[int(i)] for i in perm[:k_sel]]
                    priority_good_candidates_bi = float(len(good_candidates))
                    priority_nonzero_mean_bi = float(nonzero_mean)
                    priority_auto_base_targets_bi = float(auto_base)
                    priority_effective_targets_bi = float(k_sel)  # updated below after non-overlap
            # Dither once per selected priority seed and reuse across scales.
            # This avoids per-scale micro-clusters around the same logical target.
            priority_centers_dithered: list[tuple[int, int]] = []
            if sampling_mode == "priority_sampling" and len(priority_catalogue) > 0:
                patch_half_lo = int(inner_target_size) // 2
                patch_half_hi = int(inner_target_size) - patch_half_lo
                for cy0, cx0 in priority_catalogue:
                    cy1, cx1 = _dither_target_center(
                        cy=int(cy0),
                        cx=int(cx0),
                        h=h,
                        w=w,
                        half_lo=patch_half_lo,
                        half_hi=patch_half_hi,
                        dithering_pixels=priority_dithering_pixels,
                        device=x_clean.device,
                    )
                    priority_centers_dithered.append((int(cy1), int(cx1)))

                # Non-overlap enforcement on the *dithered* centres so that
                # dithering cannot undo the protection.
                if target_nonoverlap and len(priority_centers_dithered) > 1:
                    priority_centers_dithered = _rejection_sample_targets(
                        candidates=priority_centers_dithered,
                        num_targets=k_sel,
                        h=h,
                        w=w,
                        exclusion_box=max_box,
                        device=x_clean.device,
                        allow_partial_overlap=float(target_allow_partial_overlap),
                    )
                    priority_effective_targets_bi = float(len(priority_centers_dithered))

            num_cdd_ch = cdd_mod.shape[0]
            dip_field = np.zeros((h, w), dtype=np.float32)
            dip_field_ch = np.zeros((num_cdd_ch, h, w), dtype=np.float32)
            dip_proto_ch = np.zeros((num_cdd_ch, h, w), dtype=np.float32)
            dip_proto_written = np.zeros(num_cdd_ch, dtype=np.int32)
            cdd_box_sizes = []
            cdd_blur_sigmas = []

            for si, sigma in enumerate(active_sigmas):
                box = _effective_mask_box_size(
                    sigma=float(sigma),
                    mask_scale=mask_scale,
                    mask_box_size=mask_box_size,
                    inner_target_size=inner_target_size,
                    hardcap=mask_box_hardcap,
                )
                ch = min(si, cdd_mod.shape[0] - 1)
                half_lo = box // 2
                half_hi = box - half_lo
                cdd_box_sizes.append(float(box))
                cdd_blur_sigmas.append(0.0)
                if sampling_mode == "priority_sampling" and len(priority_centers_dithered) > 0:
                    centers = priority_centers_dithered
                elif align_scales:
                    centers = shared_centers_dithered if shared_centers_dithered is not None else shared_centers
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
                    if lattice_shift_override is not None:
                        shift_y = int(lattice_shift_override[0]) % spacing
                        shift_x = int(lattice_shift_override[1]) % spacing
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
                    # In priority mode, centers were already dithered once above.
                    if enable_target_dithering and not (
                        (sampling_mode == "priority_sampling" and len(priority_centers_dithered) > 0)
                        or align_scales
                    ):
                        cy, cx = _dither_target_center(
                            cy=int(cy),
                            cx=int(cx),
                            h=h,
                            w=w,
                            half_lo=half_lo,
                            half_hi=half_hi,
                            # Grid/lattice mode always dithers by lattice spacing.
                            dithering_pixels=spacing,
                            device=x_clean.device,
                        )
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

            # If residual is already baked into cdd_mod[-1], don't double-add it.
            # Pre-computed CDD always has residual in the last channel.
            if cdd_residual is None or cdd_append_last_residual:
                recon = np.sum(cdd_mod, axis=0)
            else:
                recon = np.sum(cdd_mod, axis=0) + cdd_residual
            recon = np.clip(recon, a_min=0.0, a_max=None)
            x_context[bi, 0] = torch.from_numpy(recon).to(device=x_clean.device, dtype=x_clean.dtype)

            # IMPORTANT:
            # In priority mode we still must keep the *dithered* centers.
            # applied_locations/applied_scales are populated after dithering,
            # while priority_catalogue holds the pre-dither seed centers.
            if sampling_mode == "priority_sampling" and len(priority_centers_dithered) > 0:
                unique_loc_to_scale = {(int(cy), int(cx)): float(active_sigmas[0]) for cy, cx in priority_centers_dithered}
            else:
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
            _ensure_target_patches_masked(sample_locations, applied_mask_hard, inner_target_size)

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
            all_cdd_masked.append(torch.from_numpy(cdd_mod.copy()))
            all_dip_fields.append(torch.from_numpy(dip_field))
            all_dip_fields_per_channel.append(torch.from_numpy(dip_field_ch))
            all_dip_proto_per_channel.append(torch.from_numpy(dip_proto_ch))
            all_cdd_box_sizes.append(torch.tensor(cdd_box_sizes, dtype=torch.float32))
            all_cdd_blur_sigmas.append(torch.tensor(cdd_blur_sigmas, dtype=torch.float32))
            all_priority_good_candidates.append(priority_good_candidates_bi)
            all_priority_nonzero_mean.append(priority_nonzero_mean_bi)
            all_priority_prescreen_candidates.append(priority_prescreen_candidates_bi)
            all_priority_auto_base_targets.append(priority_auto_base_targets_bi)
            all_priority_effective_targets.append(priority_effective_targets_bi)

            continue

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
        "cdd_box_sizes": _safe_stack(all_cdd_box_sizes),
        "cdd_blur_sigmas": _safe_stack(all_cdd_blur_sigmas),
        "priority_good_candidates": torch.tensor(all_priority_good_candidates, dtype=x_clean.dtype, device=x_clean.device),
        "priority_nonzero_mean": torch.tensor(all_priority_nonzero_mean, dtype=x_clean.dtype, device=x_clean.device),
        "priority_prescreen_candidates": torch.tensor(all_priority_prescreen_candidates, dtype=x_clean.dtype, device=x_clean.device),
        "priority_auto_base_targets": torch.tensor(all_priority_auto_base_targets, dtype=x_clean.dtype, device=x_clean.device),
        "priority_effective_targets": torch.tensor(all_priority_effective_targets, dtype=x_clean.dtype, device=x_clean.device),
        "mask_scale_factor": torch.tensor(float(mask_scale), dtype=x_clean.dtype, device=x_clean.device),
        "mask_footprint_px": torch.tensor(float(mask_box_size), dtype=x_clean.dtype, device=x_clean.device),
    }
    return x_context, target_locations, target_scales, target_valid, debug


def prepare_context_batch(
    x_clean: torch.Tensor,
    *,
    sigmas,
    mask_fraction: float = 1.0,
    mask_scale: float = 1.0,
    spacing_scale: float = 1.5,
    global_shift: bool = True,
    align_scales: bool = True,
    mask_box_size: int = 16,
    cdd_mode: str = "log",
    cdd_constrained: bool = True,
    cdd_sm_mode: str = "reflect",
    cdd_append_last_residual: bool = True,
    patch_size: int = 2,
    return_debug: bool = False,
    enable_grid_jitter: bool = True,
    enable_target_dithering: bool = True,
    lattice_shift_override: Optional[Tuple[int, int]] = None,
    target_invalid_region_skip: bool = False,
    target_invalid_region_values=(0.0, "nan"),
    target_sampling_mode: str = "grid",
    priority_top_percent: float = 5.0,
    priority_n_target: int | str = 20,
    priority_min_targets_per_map: int = 0,
    priority_dithering_pixels: Optional[int] = None,
    priority_candidate_oversample: float = 3.0,
    target_nonoverlap: bool = False,
    target_allow_partial_overlap: float = 0.0,
    mask_box_hardcap: int | None = None,
    cdd_use_gpu: bool = False,
    cdd_orig_in: Optional[torch.Tensor] = None,
):
    """Prepare context tensors from a clean batch.

    Handles NaN detection + scrubbing before masking so the downstream network
    receives pre-computed context.  Safe to call from DataLoader collate workers
    or the main process (no CUDA requirement).
    """
    invalid_pixel_mask = ~torch.isfinite(x_clean)
    if invalid_pixel_mask.any():
        x_clean = torch.nan_to_num(x_clean, nan=0.0, posinf=0.0, neginf=0.0)

    return make_pyramid_grid_context(
        x_clean=x_clean,
        sigmas=sigmas,
        mask_fraction=mask_fraction,
        mask_scale=mask_scale,
        spacing_scale=spacing_scale,
        global_shift=global_shift,
        align_scales=align_scales,
        mask_box_size=mask_box_size,
        cdd_mode=cdd_mode,
        cdd_constrained=cdd_constrained,
        cdd_sm_mode=cdd_sm_mode,
        cdd_append_last_residual=cdd_append_last_residual,
        inner_target_size=patch_size,
        return_debug=return_debug,
        enable_grid_jitter=enable_grid_jitter,
        enable_target_dithering=enable_target_dithering,
        lattice_shift_override=lattice_shift_override,
        target_invalid_region_skip=target_invalid_region_skip,
        target_invalid_region_values=target_invalid_region_values,
        invalid_pixel_mask=invalid_pixel_mask,
        target_sampling_mode=target_sampling_mode,
        priority_top_percent=priority_top_percent,
        priority_n_target=priority_n_target,
        priority_min_targets_per_map=priority_min_targets_per_map,
        priority_dithering_pixels=priority_dithering_pixels,
        priority_candidate_oversample=priority_candidate_oversample,
        target_nonoverlap=target_nonoverlap,
        target_allow_partial_overlap=target_allow_partial_overlap,
        mask_box_hardcap=mask_box_hardcap,
        cdd_use_gpu=cdd_use_gpu,
        cdd_orig_in=cdd_orig_in,
    )


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
    b, c, h, w = z.shape
    _, k, _ = locations.shape

    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if patch_size > h or patch_size > w:
        raise ValueError(f"patch_size={patch_size} exceeds feature map size {(h, w)}")

    half = patch_size // 2
    half_hi = patch_size - half

    y0 = locations[:, :, 0] - half  # B x K
    x0 = locations[:, :, 1] - half  # B x K

    valid = (y0 >= 0) & (x0 >= 0) & (y0 + patch_size <= h) & (x0 + patch_size <= w)  # B x K

    dy = torch.arange(patch_size, device=z.device)  # P
    dx = torch.arange(patch_size, device=z.device)  # P

    y_idx = y0.view(b, k, 1, 1) + dy.view(1, 1, patch_size, 1)    # B x K x P x 1
    x_idx = x0.view(b, k, 1, 1) + dx.view(1, 1, 1, patch_size)    # B x K x 1 x P

    y_idx = y_idx.clamp(0, h - 1)
    x_idx = x_idx.clamp(0, w - 1)

    # Use broadcasting in advanced indexing to avoid materializing large
    # expanded integer index tensors.
    b_idx = torch.arange(b, device=z.device).view(b, 1, 1, 1, 1)
    c_idx = torch.arange(c, device=z.device).view(1, 1, c, 1, 1)
    y_idx = y_idx.unsqueeze(2)  # Actual shape: B x K x 1 x P x 1
    x_idx = x_idx.unsqueeze(2)  # Actual shape: B x K x 1 x 1 x P

    patches = z[b_idx, c_idx, y_idx, x_idx]  # B x K x C x P x P
    valid_mask = valid.view(b, k, 1, 1, 1)
    patches = torch.where(valid_mask, patches, torch.zeros_like(patches))

    return patches
