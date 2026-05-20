import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import FullResEncoder
from .predictor import FullResPredictor

_WARNED_CDD_FORWARD_CPU = False

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


def make_pyramid_grid_context(
    x_clean: torch.Tensor,
    sigmas=(2, 4, 8, 16),
    cell_sizes=(16, 32, 64, 128),
    max_targets_per_image: int = 16,
    mask_fraction: float = 0.20,
    spacing_mult: float = 1.5,
    box_sigma_mult: float = 4.0,
    mask_scale: float = 1.0,
    min_mask_scale: float = 0.0,
    spacing_scale: float = 2.0,
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
    inner_target_size: int = 2,
    return_debug: bool = False,
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
    if blur_mode != "gaussian":
        raise ValueError(
            f"Unsupported blur_mode: {blur_mode}. "
            "Allowed blur_mode is only 'gaussian' (use mask_fill_mode=zero or gaussian_dip)."
        )
    if mask_fill_mode not in ("zero", "gaussian_dip"):
        raise ValueError(f"Unsupported mask_fill_mode: {mask_fill_mode}")

    b, _, h, w = x_clean.shape
    x_context = x_clean.clone()
    global _WARNED_CDD_FORWARD_CPU

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
    active_sigmas = tuple(float(s) for s in sigmas)
    if len(active_sigmas) == 0:
        raise ValueError("sigmas must not be empty")
    if cell_sizes is None or len(cell_sizes) == 0:
        active_cells = tuple(0 for _ in active_sigmas)
    else:
        # Match cell-size list to sigma list length.
        cs = [int(v) for v in cell_sizes]
        if len(cs) < len(active_sigmas):
            cs.extend([cs[-1]] * (len(active_sigmas) - len(cs)))
        active_cells = tuple(cs[: len(active_sigmas)])
    per_scale_fraction = float(mask_fraction) / float(len(active_sigmas))

    for bi in range(b):
        sample_locations = []
        sample_scales = []
        applied_locations = []
        applied_scales = []
        yy_full_t = torch.arange(h, device=x_clean.device, dtype=x_clean.dtype).view(h, 1)
        xx_full_t = torch.arange(w, device=x_clean.device, dtype=x_clean.dtype).view(1, w)
        yy_full_np = np.arange(h, dtype=np.float32).reshape(h, 1)
        xx_full_np = np.arange(w, dtype=np.float32).reshape(1, w)
        cdd_result = None
        cdd_residual = None
        dip_field = np.zeros((h, w), dtype=np.float32)
        dip_field_ch = np.zeros((max(1, len(active_sigmas)), h, w), dtype=np.float32)
        dip_proto_ch = np.zeros((max(1, len(active_sigmas)), h, w), dtype=np.float32)
        dip_proto_written = np.zeros((max(1, len(active_sigmas)),), dtype=np.int32)
        if blur_mode == "cdd":
            if x_clean.device.type in ("cuda", "mps") and not _WARNED_CDD_FORWARD_CPU:
                print(
                    "[PyramidGridJEPA] warning: blur_mode='cdd' runs CPU NumPy CDD in forward "
                    "(device bounce CPU<->GPU). Consider dataset-side CDD or torch-native implementation."
                )
                _WARNED_CDD_FORWARD_CPU = True
            import constrained_diffusion as cdd

            arr = x_clean[bi, 0].detach().cpu().numpy().astype(np.float32)
            cdd_kwargs = dict(
                mode=cdd_mode,
                constrained=bool(cdd_constrained),
                sm_mode=cdd_sm_mode,
                return_scales=True,
                verbose=False,
                use_gpu=False,
            )
            try:
                cdd_result, cdd_residual, _ = cdd.constrained_diffusion_decomposition(
                    arr,
                    scales=active_sigmas,
                    **cdd_kwargs,
                )
            except TypeError:
                cdd_result, cdd_residual, _ = cdd.constrained_diffusion_decomposition(
                    arr,
                    num_channels=max(1, len(active_sigmas)),
                    **cdd_kwargs,
                )
            cdd_result = np.asarray(cdd_result, dtype=np.float32)
            cdd_residual = np.asarray(cdd_residual, dtype=np.float32)
            # Treat CDD channels as non-negative components.
            cdd_result = np.clip(cdd_result, a_min=0.0, a_max=None)
            cdd_mod = cdd_result.copy()

        shared_centers = None
        if align_scales:
            base_sigma = active_sigmas[0]
            eff_base_sigma = max(float(base_sigma), float(min_mask_scale))
            base_cell = int(max(0, active_cells[0]))
            if constant_mask_box:
                base_box = int(mask_box_size)
            else:
                # Use both sigma and configured cell size to avoid misleading knobs.
                box_sigma = int(max(2, round(eff_base_sigma * float(box_sigma_mult) * float(mask_scale))))
                box_cell = int(max(2, round(max(1, base_cell) * float(mask_fraction)))) if base_cell > 0 else 2
                base_box = int(max(box_sigma, box_cell))
            largest_sigma = float(max(active_sigmas))
            eff_largest_sigma = max(float(largest_sigma), float(min_mask_scale))
            largest_cell = int(max(active_cells))
            if constant_mask_box:
                largest_box = int(mask_box_size)
            else:
                box_sigma = int(max(2, round(eff_largest_sigma * float(box_sigma_mult) * float(mask_scale))))
                box_cell = int(max(2, round(max(1, largest_cell) * float(mask_fraction)))) if largest_cell > 0 else 2
                largest_box = int(max(box_sigma, box_cell))
            # Align-grid spacing: prefer configured cell size when provided, else sigma rule.
            sigma_spacing = int(max(largest_box + 1, round(eff_largest_sigma * float(mask_scale) * float(spacing_scale))))
            cell_spacing = int(max(largest_box + 1, largest_cell)) if largest_cell > 0 else 0
            base_spacing = int(max(sigma_spacing, cell_spacing))
            base_margin = largest_box // 2 + 1
            if global_shift:
                base_shift_y = int(torch.randint(0, max(1, base_spacing), (1,), device=x_clean.device).item())
                base_shift_x = int(torch.randint(0, max(1, base_spacing), (1,), device=x_clean.device).item())
            else:
                base_shift_y = 0
                base_shift_x = 0
            y_start = base_margin + base_shift_y
            x_start = base_margin + base_shift_x
            y_centers = list(range(y_start, max(y_start + 1, h - base_margin), base_spacing))
            x_centers = list(range(x_start, max(x_start + 1, w - base_margin), base_spacing))
            shared_centers = [(cy, cx) for cy in y_centers for cx in x_centers]
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
            cell_size = int(max(0, active_cells[si]))
            # Ensure masked region always includes the target patch:
            # min box = max(cdd_scale * mask_fraction, inner_target_size)
            min_box = int(max(2, round(float(sigma) * float(mask_fraction)), int(inner_target_size)))
            if constant_mask_box:
                base_box = int(mask_box_size)
            else:
                box_sigma = int(max(2, round(eff_sigma * float(box_sigma_mult) * float(mask_scale))))
                box_cell = int(max(2, round(max(1, cell_size) * float(mask_fraction)))) if cell_size > 0 else 2
                base_box = int(max(box_sigma, box_cell))
            box = int(max(base_box, min_box))
            half = box // 2
            if align_scales:
                centers = shared_centers
            else:
                spacing_sigma = int(max(box + 1, round(box * float(spacing_mult))))
                spacing_cell = int(max(box + 1, cell_size)) if cell_size > 0 else 0
                spacing = int(max(spacing_sigma, spacing_cell))
                margin = half + 1
                area_budget = per_scale_fraction * float(h * w)
                desired_count = area_budget / max(1.0, float(box * box))
                base_count = int(math.floor(desired_count))
                frac = float(desired_count - base_count)
                extra = int(torch.rand(1, device=x_clean.device).item() < frac)
                max_count = max(0, base_count + extra)
                if max_count <= 0:
                    continue
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
                y0 = max(0, cy - half)
                y1 = min(h, cy + half)
                x0 = max(0, cx - half)
                x1 = min(w, cx + half)
                if blur_mode == "gaussian":
                    if mask_fill_mode == "zero":
                        x_context[bi : bi + 1, :, y0:y1, x0:x1] = 0.0
                    elif mask_fill_mode == "gaussian_dip":
                        s = max(1e-6, float(eff_sigma) * float(dip_sigma_mult))
                        g = torch.exp(-(((yy_full_t - float(cy)) ** 2 + (xx_full_t - float(cx)) ** 2) / (2.0 * s * s)))
                        x_context[bi : bi + 1] *= (1.0 - g).view(1, 1, h, w)
                        g_np = g.detach().cpu().numpy().astype(np.float32)
                        dip_field = np.maximum(dip_field, g_np)
                        chp = min(si, dip_field_ch.shape[0] - 1)
                        dip_field_ch[chp] = np.maximum(dip_field_ch[chp], g_np)
                        if dip_proto_written[chp] == 0:
                            dip_proto_ch[chp] = g_np
                            dip_proto_written[chp] = 1
                    else:
                        # No legacy replacement blur branch: gaussian mode is zero or gaussian_dip only.
                        pass
                else:
                    ch = min(si, cdd_mod.shape[0] - 1)
                    if mask_fill_mode == "gaussian_dip":
                        s = max(1e-6, float(eff_sigma) * float(dip_sigma_mult))
                        g = np.exp(-(((yy_full_np - float(cy)) ** 2 + (xx_full_np - float(cx)) ** 2) / (2.0 * s * s))).astype(
                            np.float32
                        )
                        cdd_mod[ch] *= (1.0 - g)
                        dip_field = np.maximum(dip_field, g)
                        dip_field_ch[ch] = np.maximum(dip_field_ch[ch], g)
                        if dip_proto_written[ch] == 0:
                            dip_proto_ch[ch] = g
                            dip_proto_written[ch] = 1
                    else:
                        cdd_mod[ch, y0:y1, x0:x1] = 0.0
                applied_locations.append((cy, cx))
                applied_scales.append(float(sigma))
                # Training targets are grid centers (deduplicated), not per-scale duplicates.
                sample_locations.append((cy, cx))
                sample_scales.append(float(sigma))

        if blur_mode == "cdd":
            recon = np.sum(cdd_mod, axis=0) + cdd_residual
            # Keep CDD reconstruction non-negative before shared log transform.
            recon = np.clip(recon, a_min=0.0, a_max=None)
            x_context[bi, 0] = torch.from_numpy(recon).to(device=x_clean.device, dtype=x_clean.dtype)

        # Keep a regular grid of unique centers to avoid overlap-heavy duplicate targets.
        unique_loc_to_scale = {}
        for (cy, cx), s in zip(sample_locations, sample_scales):
            key = (int(cy), int(cx))
            if key not in unique_loc_to_scale:
                unique_loc_to_scale[key] = float(s)
        sample_locations = sorted(unique_loc_to_scale.keys())
        sample_scales = [float(unique_loc_to_scale[k]) for k in sample_locations]

        if len(sample_locations) > max_targets_per_image:
            # Deterministic, evenly spaced downsample keeps target layout regular.
            idx = torch.linspace(0, len(sample_locations) - 1, steps=max_targets_per_image, device=x_clean.device)
            keep = [int(round(v)) for v in idx.detach().cpu().numpy().tolist()]
            sample_locations = [sample_locations[i] for i in keep]
            sample_scales = [sample_scales[i] for i in keep]
        sample_valid = [1] * len(sample_locations)

        while len(sample_locations) < max_targets_per_image:
            sample_locations.append((h // 2, w // 2))
            sample_scales.append(0.0)
            sample_valid.append(0)

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
                    uniq.append(key)
            if mask_fill_mode == "gaussian_dip":
                m_soft = np.zeros((h, w), dtype=np.float32)
                for (cy, cx), s in zip(applied_locations, applied_scales):
                    eff_s = max(float(s), float(min_mask_scale))
                    ss = max(1e-6, eff_s * float(dip_sigma_mult))
                    g = np.exp(-(((yy_full_np - float(cy)) ** 2 + (xx_full_np - float(cx)) ** 2) / (2.0 * ss * ss))).astype(
                        np.float32
                    )
                    m_soft = np.maximum(m_soft, g)
                m = (m_soft > 1e-3).astype(np.uint8)
            else:
                m = np.zeros((h, w), dtype=np.uint8)
                for (cy, cx), s in zip(applied_locations, applied_scales):
                    eff_s = max(float(s), float(min_mask_scale))
                    box = int(mask_box_size) if constant_mask_box else int(max(2, round(eff_s * float(mask_scale))))
                    half = box // 2
                    y0 = max(0, int(cy) - half)
                    y1 = min(h, int(cy) + half)
                    x0 = max(0, int(cx) - half)
                    x1 = min(w, int(cx) + half)
                    m[y0:y1, x0:x1] = 1
            all_mask_maps.append(torch.from_numpy(m))
            all_unique_centers.append(torch.tensor(uniq, dtype=torch.long))
            if blur_mode == "cdd":
                all_cdd_orig.append(torch.from_numpy(cdd_result))
                all_cdd_masked.append(torch.from_numpy(cdd_mod))
            else:
                all_cdd_orig.append(torch.empty(0))
                all_cdd_masked.append(torch.empty(0))
            all_dip_fields.append(torch.from_numpy(dip_field))
            all_dip_fields_per_channel.append(torch.from_numpy(dip_field_ch))
            all_dip_proto_per_channel.append(torch.from_numpy(dip_proto_ch))

    target_locations = torch.tensor(all_locations, dtype=torch.long, device=x_clean.device)
    target_scales = torch.tensor(all_scales, dtype=x_clean.dtype, device=x_clean.device)
    target_valid = torch.tensor(all_valid, dtype=torch.bool, device=x_clean.device)

    if not return_debug:
        return x_context, target_locations, target_scales, target_valid

    max_centers = max((int(t.shape[0]) for t in all_unique_centers), default=0)
    centers_pad = torch.full((b, max_centers, 2), -1, dtype=torch.long, device=x_clean.device)
    for bi, t in enumerate(all_unique_centers):
        if t.numel() > 0:
            centers_pad[bi, : t.shape[0]] = t.to(device=x_clean.device)

    debug = {
        "mask_map": torch.stack([m.to(device=x_clean.device) for m in all_mask_maps], dim=0),
        "unique_centers": centers_pad,
        "cdd_channels_orig": torch.stack([t.to(device=x_clean.device, dtype=x_clean.dtype) for t in all_cdd_orig], dim=0)
        if blur_mode == "cdd"
        else torch.empty(0, device=x_clean.device, dtype=x_clean.dtype),
        "cdd_channels_masked": torch.stack([t.to(device=x_clean.device, dtype=x_clean.dtype) for t in all_cdd_masked], dim=0)
        if blur_mode == "cdd"
        else torch.empty(0, device=x_clean.device, dtype=x_clean.dtype),
        "dip_field": torch.stack([t.to(device=x_clean.device, dtype=x_clean.dtype) for t in all_dip_fields], dim=0),
        "dip_field_per_channel": torch.stack(
            [t.to(device=x_clean.device, dtype=x_clean.dtype) for t in all_dip_fields_per_channel], dim=0
        ),
        "dip_proto_per_channel": torch.stack(
            [t.to(device=x_clean.device, dtype=x_clean.dtype) for t in all_dip_proto_per_channel], dim=0
        ),
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
    patches = []

    for bi in range(b):
        sample_patches = []

        for ki in range(k):
            cy = int(locations[bi, ki, 0].item())
            cx = int(locations[bi, ki, 1].item())

            y0 = cy - half
            y1 = cy + half
            x0 = cx - half
            x1 = cx + half

            if y0 < 0:
                y0 = 0
                y1 = patch_size
            if x0 < 0:
                x0 = 0
                x1 = patch_size
            if y1 > h:
                y1 = h
                y0 = h - patch_size
            if x1 > w:
                x1 = w
                x0 = w - patch_size

            sample_patches.append(z[bi : bi + 1, :, y0:y1, x0:x1])

        sample_patches = torch.cat(sample_patches, dim=0)
        patches.append(sample_patches.unsqueeze(0))

    return torch.cat(patches, dim=0)


class PyramidGridJEPA(nn.Module):
    def __init__(
        self,
        latent_channels: int = 32,
        predictor_hidden: int = None,
        patch_size: int = 2,
        sigmas=(2, 4, 8, 16),
        cell_sizes=(16, 32, 64, 128),
        max_targets_per_image: int = 16,
        mask_fraction: float = 0.20,
        spacing_mult: float = 1.5,
        box_sigma_mult: float = 4.0,
        mask_scale: float = 1.0,
        min_mask_scale: float = 0.0,
        spacing_scale: float = 2.0,
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
        post_log_transform: bool = True,
        log_eps: float = 1.0,
        cdd_log_std_floor_mult: float = 0.05,
        ema_momentum: float = 0.996,
        normalize_loss: bool = True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.sigmas = tuple(sigmas)
        self.cell_sizes = tuple(cell_sizes)
        self.max_targets_per_image = int(max_targets_per_image)
        self.mask_fraction = float(mask_fraction)
        self.spacing_mult = float(spacing_mult)
        self.box_sigma_mult = float(box_sigma_mult)
        self.mask_scale = float(mask_scale)
        self.min_mask_scale = float(min_mask_scale)
        self.spacing_scale = float(spacing_scale)
        self.full_grid = bool(full_grid)
        self.global_shift = bool(global_shift)
        self.align_scales = bool(align_scales)
        self.constant_mask_box = bool(constant_mask_box)
        self.mask_box_size = int(mask_box_size)
        self.blur_mode = str(blur_mode)
        if self.blur_mode != "gaussian":
            raise ValueError(
                f"Unsupported blur_mode: {self.blur_mode}. "
                "Allowed blur_mode is only 'gaussian' (use mask_fill_mode=zero or gaussian_dip)."
            )
        self.cdd_mode = str(cdd_mode)
        self.cdd_constrained = bool(cdd_constrained)
        self.cdd_sm_mode = str(cdd_sm_mode)
        self.mask_fill_mode = str(mask_fill_mode)
        self.dip_sigma_mult = float(dip_sigma_mult)
        self.post_log_transform = bool(post_log_transform)
        self.log_eps = float(log_eps)
        self.cdd_log_std_floor_mult = float(cdd_log_std_floor_mult)
        self.ema_momentum = float(ema_momentum)
        self.normalize_loss = bool(normalize_loss)

        self.context_encoder = FullResEncoder(in_channels=1, latent_channels=latent_channels)

        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        if predictor_hidden is None:
            predictor_hidden = latent_channels * 2
        self.predictor = FullResPredictor(channels=latent_channels, hidden=int(predictor_hidden))

    def forward(self, x_clean):
        """
        x_clean: B x 1 x H x W
        """
        if x_clean.dim() != 4:
            raise ValueError(f"Expected BxCxHxW, got {tuple(x_clean.shape)}")

        if x_clean.shape[1] != 1:
            raise ValueError(f"Expected grayscale input, got {x_clean.shape[1]} channels")

        x_context, target_locations, target_scales, target_valid = make_pyramid_grid_context(
            x_clean=x_clean,
            sigmas=self.sigmas,
            cell_sizes=self.cell_sizes,
            max_targets_per_image=self.max_targets_per_image,
            mask_fraction=self.mask_fraction,
            spacing_mult=self.spacing_mult,
            box_sigma_mult=self.box_sigma_mult,
            mask_scale=self.mask_scale,
            min_mask_scale=self.min_mask_scale,
            spacing_scale=self.spacing_scale,
            full_grid=self.full_grid,
            global_shift=self.global_shift,
            align_scales=self.align_scales,
            constant_mask_box=self.constant_mask_box,
            mask_box_size=self.mask_box_size,
            blur_mode=self.blur_mode,
            cdd_mode=self.cdd_mode,
            cdd_constrained=self.cdd_constrained,
            cdd_sm_mode=self.cdd_sm_mode,
            mask_fill_mode=self.mask_fill_mode,
            dip_sigma_mult=self.dip_sigma_mult,
            inner_target_size=self.patch_size,
        )

        x_clean_enc = x_clean
        x_context_enc = x_context
        if self.post_log_transform:
            eps = max(1e-30, float(self.log_eps))
            if self.blur_mode == "cdd":
                # CDD stabilization: shared per-sample std floor from clean branch.
                base = torch.clamp(x_clean, min=0.0)
                base_std = torch.std(base, dim=(-2, -1), keepdim=True)
                log_floor = torch.clamp(base_std * float(self.cdd_log_std_floor_mult), min=eps)
                x_clean_enc = torch.log(torch.clamp(x_clean, min=0.0) + log_floor)
                x_context_enc = torch.log(torch.clamp(x_context, min=0.0) + log_floor)
            else:
                x_clean_enc = torch.log(torch.clamp(x_clean, min=0.0) + eps)
                x_context_enc = torch.log(torch.clamp(x_context, min=0.0) + eps)

        with torch.no_grad():
            gt_map = self.target_encoder(x_clean_enc)

        context_map = self.context_encoder(x_context_enc)
        pred_map = self.predictor(context_map)

        pred_patches = extract_location_patches(pred_map, target_locations, patch_size=self.patch_size)
        gt_patches = extract_location_patches(gt_map, target_locations, patch_size=self.patch_size)

        return {
            "pred_patches": pred_patches,
            "gt_patches": gt_patches,
            # Raw pre-encoder tensors (for diagnostics/visualization).
            "x_clean_raw": x_clean,
            "x_context_raw": x_context,
            # Actual network inputs after shared post-mask transform.
            "x_clean": x_clean_enc,
            "x_context": x_context_enc,
            "target_locations": target_locations,
            "target_scales": target_scales,
            "target_valid": target_valid,
            "context_map": context_map,
            "pred_map": pred_map,
            "gt_map": gt_map,
        }

    def compute_loss(self, outputs):
        pred = outputs["pred_patches"]
        gt = outputs["gt_patches"].detach()

        valid = outputs["target_valid"]  # B x K (bool)

        if self.normalize_loss:
            pred = F.normalize(pred, dim=2)
            gt = F.normalize(gt, dim=2)
        loss_map = F.mse_loss(pred, gt, reduction="none")  # B x K x C x P x P
        w = valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(loss_map.dtype)
        denom = torch.clamp(w.sum() * loss_map.shape[2] * loss_map.shape[3] * loss_map.shape[4], min=1.0)
        return (loss_map * w).sum() / denom

    @torch.no_grad()
    def update_target_encoder(self):
        for p_context, p_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(self.ema_momentum).add_((1.0 - self.ema_momentum) * p_context.detach().data)
