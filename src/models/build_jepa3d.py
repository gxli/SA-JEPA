from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders3d import ScaleAwareConvNeXt3DEncoder, ScaleFiLMConvNeXt3DEncoder
from .masking import _fractional_spatial_target_budget
from .masking3d import extract_location_cubes
from .predictor3d import FullResPredictor3D
from .symmetry import symmetric_forward_3d
from src.losses import l2_normalize_patches


def compute_3d_encoder_receptive_field_depth(encoder_depth: int = 3, encoder_kernel_size: int = 5) -> int:
    """Depth receptive field for the same-padded 3D encoder path.

    The 3D encoders use two 3x3x3 stem convolutions, then `encoder_depth`
    ConvNeXt depthwise convolutions with `encoder_kernel_size`.
    """
    return 1 + 2 * (3 - 1) + max(0, int(encoder_depth)) * (int(encoder_kernel_size) - 1)


class PyramidGridJEPA3D(nn.Module):
    def __init__(
        self,
        latent_channels=16,
        scale_channels=8,
        num_scales: int = 1,
        encoder_type: str = "cdd_scaleaware_convnext3d",
        patch_size=2,
        num_targets: int | str = "auto",
        encoder_depth=3,
        encoder_kernel_size=5,
        encoder_stride=1,
        ema_momentum=0.996,
        normalize_loss_l2=False,
        post_log_transform: bool = True,
        log_eps: float = 1e-6,
        cdd_log_std_floor_mult: float = 0.05,
        fusion="gate",
        mask_box_size: int = 8,
        num_mask_boxes: int = 8,
        slab_depth: int = 3,
        use_symmetric_feature_loss: bool = False,
        use_film: bool = True,
        use_per_scale_adapters: bool = False,
        priority_candidate_oversample: float = 3.0,
        priority_min_targets_per_map: int = 0,
        target_nonoverlap: bool = True,
        target_allow_partial_overlap: float = 0.0,
        encoder_receptive_field_depth: int | None = None,
        use_grn: bool = True,
        stem_norm: bool = True,
        norm_per_scale: bool = True,
        adapter_norm: bool = True,
        final_norm: bool = True,
    ):
        super().__init__()
        self.num_scales = int(num_scales)
        self.encoder_type = str(encoder_type).lower()
        self.patch_size = int(patch_size)
        self.num_targets = num_targets
        self.ema_momentum = float(ema_momentum)
        self.normalize_loss_l2 = bool(normalize_loss_l2)
        self.post_log_transform = bool(post_log_transform)
        self.log_eps = float(log_eps)
        self.cdd_log_std_floor_mult = float(cdd_log_std_floor_mult)
        self.mask_box_size = int(mask_box_size)
        self.num_mask_boxes = int(num_mask_boxes)
        self.mode = "3d_slab"
        self.slab_depth = max(self.patch_size, int(slab_depth))
        self.encoder_receptive_field_depth = int(
            encoder_receptive_field_depth
            if encoder_receptive_field_depth is not None
            else compute_3d_encoder_receptive_field_depth(encoder_depth, encoder_kernel_size)
        )
        self.required_input_depth = int(self.encoder_receptive_field_depth + self.slab_depth - 1)
        self.use_symmetric_feature_loss = bool(use_symmetric_feature_loss)
        self.use_film = bool(use_film)
        self.use_per_scale_adapters = bool(use_per_scale_adapters)
        self.priority_candidate_oversample = float(priority_candidate_oversample)
        self.priority_min_targets_per_map = int(priority_min_targets_per_map)
        self.target_nonoverlap = bool(target_nonoverlap)
        self.target_allow_partial_overlap = float(target_allow_partial_overlap)

        if self.use_film or self.use_per_scale_adapters:
            self.context_encoder = ScaleFiLMConvNeXt3DEncoder(
                num_scales=self.num_scales,
                out_channels=int(latent_channels),
                scale_channels=int(scale_channels),
                depth=int(encoder_depth),
                kernel_size=int(encoder_kernel_size),
                stride=int(encoder_stride),
                fusion=str(fusion),
                use_film=self.use_film,
                use_per_scale_adapters=self.use_per_scale_adapters,
                use_grn=bool(use_grn),
                stem_norm=bool(stem_norm),
                norm_per_scale=bool(norm_per_scale),
                adapter_norm=bool(adapter_norm),
                final_norm=bool(final_norm),
            )
        else:
            self.context_encoder = ScaleAwareConvNeXt3DEncoder(
                num_scales=self.num_scales,
                out_channels=int(latent_channels),
                scale_channels=int(scale_channels),
                depth=int(encoder_depth),
                kernel_size=int(encoder_kernel_size),
                stride=int(encoder_stride),
                fusion=str(fusion),
                use_grn=bool(use_grn),
                stem_norm=bool(stem_norm),
                norm_per_scale=bool(norm_per_scale),
                adapter_norm=bool(adapter_norm),
                final_norm=bool(final_norm),
            )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.projector = nn.Identity()
        self.predictor3d = FullResPredictor3D(
            channels=int(latent_channels),
            hidden=max(2 * int(latent_channels), 32),
        )
    def make_fields(self, x):
        # The input channel is the single direct 3D field axis consumed by the encoder.
        return x

    def _make_random_box_mask3d(
        self,
        batch_size: int,
        depth: int,
        height: int,
        width: int,
        device,
        focus_slab_start_idx: torch.Tensor,
    ):
        box = max(1, int(self.mask_box_size))
        n_box = max(1, int(self.num_mask_boxes), int(self._target_budget(height=height, width=width, device=device)))
        mask = torch.zeros((batch_size, 1, depth, height, width), device=device)
        z_lim = max(1, depth - box + 1)
        y_lim = max(1, height - box + 1)
        x_lim = max(1, width - box + 1)
        z0 = torch.empty((batch_size, n_box), device=device, dtype=torch.long)

        slab_start_cpu = focus_slab_start_idx.detach().to("cpu").numpy()
        slab_depth = max(1, min(int(self.slab_depth), depth))
        for b in range(batch_size):
            slab_start = int(slab_start_cpu[b])
            slab_end = slab_start + slab_depth
            lo = max(0, slab_start - box + 1)
            hi = min(z_lim, slab_end)
            if hi <= lo:
                lo, hi = 0, z_lim
            z0[b] = torch.randint(lo, hi, (n_box,), device=device)

        y0 = torch.randint(0, y_lim, (batch_size, n_box), device=device)
        x0 = torch.randint(0, x_lim, (batch_size, n_box), device=device)
        # Build mask on CPU to avoid per-slice CUDA kernel launches
        z0_cpu = z0.detach().to("cpu").numpy()
        y0_cpu = y0.detach().to("cpu").numpy()
        x0_cpu = x0.detach().to("cpu").numpy()
        mask_cpu = mask.detach().to("cpu").numpy()
        for b in range(batch_size):
            for j in range(n_box):
                zz = int(z0_cpu[b, j])
                yy = int(y0_cpu[b, j])
                xx = int(x0_cpu[b, j])
                mask_cpu[b, 0, zz : zz + box, yy : yy + box, xx : xx + box] = 1.0
        mask.copy_(torch.from_numpy(mask_cpu).to(device=mask.device))
        return mask

    def _target_budget(self, height: int, width: int, device) -> int:
        raw = self.num_targets
        if isinstance(raw, str) and raw.strip().lower() == "auto":
            budget_oversample = float(self.priority_candidate_oversample)
            if budget_oversample <= 0.0:
                budget_oversample = 1.0
            return int(
                _fractional_spatial_target_budget(
                    height=int(height),
                    width=int(width),
                    box_size=max(1, int(self.mask_box_size)),
                    oversample=budget_oversample,
                    device=device,
                    minimum=int(self.priority_min_targets_per_map),
                    overlap_fraction=float(self.target_allow_partial_overlap),
                )
                or 0
            )
        return max(0, int(round(float(raw))))

    def _sample_targets_from_masked_slab(
        self,
        mask_slab: torch.Tensor,
        num_targets: int,
        patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask_slab.dim() != 5:
            raise ValueError(f"Expected mask_slab Bx1xDxHxW, got {tuple(mask_slab.shape)}")
        b, _, d, h, w = mask_slab.shape
        device = mask_slab.device
        p = int(patch_size)
        half = p // 2
        # mask_slab is already the gathered target slab after the encoder has
        # seen its required depth context. Do not apply the full input-depth RF
        # margin to this small target slab, or depth-3 targets become impossible.
        spatial_fov_half = max(half, self.encoder_receptive_field_depth // 2)
        lo_z = half
        hi_z = d - (p - half)
        lo_y = spatial_fov_half
        hi_y = h - spatial_fov_half
        lo_x = spatial_fov_half
        hi_x = w - spatial_fov_half
        target_budget = max(1, int(num_targets))
        loc_out = torch.zeros((b, target_budget, 3), dtype=torch.long, device=device)
        valid_out = torch.zeros((b, target_budget), dtype=torch.bool, device=device)
        if hi_z < lo_z or hi_y < lo_y or hi_x < lo_x:
            return loc_out, valid_out

        exclusion = max(1, int(self.mask_box_size))
        allow_partial = float(self.target_allow_partial_overlap)
        for bi in range(b):
            candidates = torch.nonzero(mask_slab[bi, 0] > 0, as_tuple=False)
            if candidates.numel() == 0:
                continue
            inside = (
                (candidates[:, 0] >= lo_z)
                & (candidates[:, 0] <= hi_z)
                & (candidates[:, 1] >= lo_y)
                & (candidates[:, 1] <= hi_y)
                & (candidates[:, 2] >= lo_x)
                & (candidates[:, 2] <= hi_x)
            )
            candidates = candidates[inside]
            if candidates.numel() == 0:
                continue
            perm = torch.randperm(candidates.shape[0], device=device)
            candidates = candidates[perm]
            selected = []
            for cand in candidates:
                if len(selected) >= target_budget:
                    break
                if bool(self.target_nonoverlap) and selected:
                    yy = int(cand[1].item())
                    xx = int(cand[2].item())
                    ok = True
                    min_sep = max(0.0, float(exclusion) * (1.0 - allow_partial))
                    for prev in selected:
                        if max(abs(yy - int(prev[1])), abs(xx - int(prev[2]))) < min_sep:
                            ok = False
                            break
                    if not ok:
                        continue
                selected.append(cand)
            if selected:
                stacked = torch.stack(selected, dim=0)
                n = min(target_budget, stacked.shape[0])
                loc_out[bi, :n] = stacked[:n]
                valid_out[bi, :n] = True
        return loc_out, valid_out

    def _center_slab_start_index(self, batch_size: int, depth: int, device) -> tuple[torch.Tensor, int]:
        slab_depth = max(1, min(int(self.slab_depth), int(depth)))
        start = max(0, (int(depth) - slab_depth) // 2)
        starts = torch.full((batch_size,), start, dtype=torch.long, device=device)
        return starts, slab_depth

    @staticmethod
    def _gather_slabs(z: torch.Tensor, slab_starts: torch.Tensor, slab_depth: int) -> torch.Tensor:
        b, c, d, h, w = z.shape
        offsets = torch.arange(slab_depth, device=z.device).view(1, slab_depth)
        slab_idx = (slab_starts.view(b, 1) + offsets).clamp(0, d - 1)
        gather_idx = slab_idx.view(b, 1, slab_depth, 1, 1).expand(b, c, slab_depth, h, w)
        return z.gather(dim=2, index=gather_idx)

    def forward(self, x_clean, **kwargs):
        if x_clean.dim() != 5:
            raise ValueError(f"Expected BxSxDxHxW, got {tuple(x_clean.shape)}")

        b, s, _, _, _ = x_clean.shape
        if s == 1 and s != self.num_scales:
            # On-the-fly 3D CDD decomposition (DDP fallback — no precomputed cache).
            import constrained_diffusion as cdd
            import numpy as np
            x_np = x_clean[:, 0].cpu().numpy()  # (B, D, H, W)
            decomposed = []
            for bi in range(b):
                arr = x_np[bi].astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-20)
                channels_arr, _residual = cdd.constrained_diffusion_decomposition(
                    arr,
                    num_channels=self.num_scales,
                    max_scale=16.0,
                    mode="log",
                    constrained=True,
                    sm_mode="reflect",
                    return_scales=False,
                    verbose=False,
                    use_gpu=False,
                    gaussian_backend="cuda",
                )
                ch = np.clip(np.stack(channels_arr, axis=0), 0.0, None).astype(np.float32)
                decomposed.append(torch.from_numpy(ch).to(x_clean.device))
            x_clean = torch.stack(decomposed, dim=0)  # (B, S, D, H, W)
            s = self.num_scales
        elif s != self.num_scales:
            raise ValueError(f"Expected Bx{self.num_scales}xDxHxW, got {tuple(x_clean.shape)}")
        fields = self.make_fields(x_clean)
        _, _, d, h, w = fields.shape
        if d < int(self.required_input_depth):
            raise ValueError(
                "3d_slab input depth is too small for the configured encoder/target geometry: "
                f"got {d}, required at least {self.required_input_depth} "
                f"(encoder_rf={self.encoder_receptive_field_depth}, target_slab_depth={self.slab_depth})"
            )
        if d > int(self.required_input_depth):
            input_start = max(0, (int(d) - int(self.required_input_depth)) // 2)
            input_end = input_start + int(self.required_input_depth)
            fields = fields[:, :, input_start:input_end]
            x_clean = x_clean[:, :, input_start:input_end]
            d = int(fields.shape[2])

        slab_starts, slab_depth = self._center_slab_start_index(batch_size=b, depth=d, device=x_clean.device)

        box_mask = self._make_random_box_mask3d(
            b,
            d,
            h,
            w,
            x_clean.device,
            focus_slab_start_idx=slab_starts,
        )
        fields_context = fields * (1.0 - box_mask)
        mask_tokens = box_mask.expand(-1, fields.shape[1], -1, -1, -1)
        if self.post_log_transform:
            eps = max(1e-6, float(self.log_eps))
            base = torch.clamp(fields, min=0.0)
            base_std = torch.std(base, dim=(-3, -2, -1), keepdim=True)
            log_floor = torch.clamp(base_std * float(self.cdd_log_std_floor_mult), min=eps)
            fields = torch.log(base + log_floor)
            base_ctx = torch.clamp(fields_context, min=0.0)
            fields_context = torch.log(base_ctx + log_floor)

        return_full_3d_maps = bool(kwargs.get("return_full_3d_maps", False))
        with torch.no_grad():
            zero_mask_tokens = torch.zeros_like(fields)
            if self.use_symmetric_feature_loss:
                gt_map_3d, target_symmetric_var = symmetric_forward_3d(
                    self.target_encoder,
                    fields,
                    mask_tokens=zero_mask_tokens,
                    return_var=True,
                )
            else:
                gt_map_3d = self.target_encoder(fields, mask_tokens=zero_mask_tokens)
                target_symmetric_var = None
            gt_map = self._gather_slabs(gt_map_3d, slab_starts, slab_depth)
            gt_map_full = gt_map_3d if return_full_3d_maps else None
            if not return_full_3d_maps:
                del gt_map_3d
            del zero_mask_tokens

        if self.use_symmetric_feature_loss:
            context_map_3d, symmetric_var = symmetric_forward_3d(
                self.context_encoder,
                fields_context,
                mask_tokens=mask_tokens,
                return_var=True,
            )
        else:
            context_map_3d = self.context_encoder(fields_context, mask_tokens=mask_tokens)
            symmetric_var = None
        context_map = self._gather_slabs(context_map_3d, slab_starts, slab_depth)
        context_map_full = context_map_3d if return_full_3d_maps else None
        if not return_full_3d_maps:
            del context_map_3d
        pred_map = self.predictor3d(context_map)
        _, _, dz, hy, wx = pred_map.shape
        num_targets = max(1, self._target_budget(height=hy, width=wx, device=x_clean.device))
        mask_slab = self._gather_slabs(box_mask, slab_starts, slab_depth)
        target_locations, target_valid = self._sample_targets_from_masked_slab(
            mask_slab=mask_slab,
            num_targets=num_targets,
            patch_size=self.patch_size,
        )
        pred_patches = extract_location_cubes(pred_map, target_locations, self.patch_size)
        gt_patches = extract_location_cubes(gt_map, target_locations, self.patch_size)
        context_patches = extract_location_cubes(context_map, target_locations, self.patch_size)

        out = {
            "pred_patches": pred_patches,
            "gt_patches": gt_patches,
            "context_patches": context_patches,
            "target_locations": target_locations,
            "target_valid": target_valid,
            "target_scales": torch.ones((b, num_targets), device=x_clean.device, dtype=x_clean.dtype),
            "context_map": context_map,
            "pred_map": pred_map,
            "gt_map": gt_map,
            "x_clean": self._gather_slabs(x_clean, slab_starts, slab_depth),
            "x_clean_full": x_clean,
            "x_context": self._gather_slabs(fields_context, slab_starts, slab_depth),
            "x_context_full": fields_context,
            "mask_cube": box_mask,
            "selected_slab_start_index": slab_starts,
            "selected_slab_depth": torch.full((b,), int(slab_depth), device=x_clean.device, dtype=torch.long),
            "encoder_receptive_field_depth": torch.full((b,), int(self.encoder_receptive_field_depth), device=x_clean.device, dtype=torch.long),
            "required_input_depth": torch.full((b,), int(self.required_input_depth), device=x_clean.device, dtype=torch.long),
            "mask_footprint_px": torch.tensor(float(self.mask_box_size), device=x_clean.device, dtype=x_clean.dtype),
            "mask_scale_factor": torch.tensor(1.0, device=x_clean.device, dtype=x_clean.dtype),
        }
        if return_full_3d_maps:
            out["context_map_3d"] = context_map_full
            out["gt_map_3d"] = gt_map_full
        if symmetric_var is not None:
            out["symmetric_var"] = symmetric_var
        if target_symmetric_var is not None:
            out["target_symmetric_var"] = target_symmetric_var
        return out

    def compute_symmetric_loss(self, outputs):
        """Context-encoder view variance, averaged over spatial and channel dims."""
        var = outputs.get("symmetric_var")
        if var is None:
            return torch.tensor(0.0, device=outputs["pred_patches"].device)
        return var.mean()

    def compute_loss(self, outputs):
        # Keep reductions in fp32: cube sums can overflow under AMP.
        pred = outputs["pred_patches"].float()
        gt = outputs["gt_patches"].detach().float()
        valid = outputs["target_valid"]

        if self.normalize_loss_l2:
            # Normalize the full cube vector so spatial contrast is preserved.
            pred = l2_normalize_patches(pred)
            gt = l2_normalize_patches(gt)
            outputs["pred_patches"] = pred
            outputs["gt_patches"] = gt

        loss_map = F.mse_loss(pred, gt, reduction="none")
        view_shape = [valid.shape[0], valid.shape[1]] + [1] * (loss_map.dim() - 2)
        w = valid.view(*view_shape).to(loss_map.dtype)

        if not bool(valid.any().item()):
            return loss_map.sum() * 0.0

        denom = torch.clamp(w.sum() * math.prod(loss_map.shape[2:]), min=1.0)
        return (loss_map * w).sum() / denom

    @torch.no_grad()
    def update_target_encoder(self):
        for p_context, p_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_target.mul_(self.ema_momentum).add_(p_context.detach(), alpha=1.0 - self.ema_momentum)
