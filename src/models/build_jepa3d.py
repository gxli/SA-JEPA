from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders3d import ScaleAwareConvNeXt3DEncoder
from .masking import extract_location_patches
from .masking3d import extract_location_cubes, make_gaussian_pyramid3d, sample_target_locations_3d
from .predictor import FullResPredictor
from .predictor3d import FullResPredictor3D


class PyramidGridJEPA3D(nn.Module):
    def __init__(
        self,
        latent_channels=16,
        scale_channels=8,
        sigmas=(2, 4, 8, 16),
        patch_size=2,
        num_targets=32,
        encoder_depth=3,
        encoder_kernel_size=5,
        encoder_stride=1,
        ema_momentum=0.996,
        normalize_loss=False,
        fusion="gate",
        constant_mask_box: bool = False,
        mask_box_size: int = 8,
        num_mask_boxes: int = 8,
        mode: str = "2d",
        slab_depth: int = 3,
        slab_boundary_margin: int = 0,
    ):
        super().__init__()
        self.sigmas = tuple(sigmas)
        self.patch_size = int(patch_size)
        self.num_targets = int(num_targets)
        self.ema_momentum = float(ema_momentum)
        self.normalize_loss = bool(normalize_loss)
        self.constant_mask_box = bool(constant_mask_box)
        self.mask_box_size = int(mask_box_size)
        self.num_mask_boxes = int(num_mask_boxes)
        self.mode = self._normalize_mode(mode)
        self.slab_depth = max(1, int(slab_depth))
        self.slab_boundary_margin = max(0, int(slab_boundary_margin))

        self.context_encoder = ScaleAwareConvNeXt3DEncoder(
            num_scales=len(self.sigmas),
            out_channels=int(latent_channels),
            scale_channels=int(scale_channels),
            depth=int(encoder_depth),
            kernel_size=int(encoder_kernel_size),
            stride=int(encoder_stride),
            fusion=str(fusion),
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.projector = nn.Identity()
        self.predictor3d = FullResPredictor3D(
            channels=int(latent_channels),
            hidden=max(2 * int(latent_channels), 32),
        )
        self.predictor2d = FullResPredictor(
            channels=int(latent_channels),
            hidden=max(2 * int(latent_channels), 32),
            kernel_size=1,
        )

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        mode_norm = str(mode).strip().lower().replace(" ", "_")
        aliases = {
            "pyramid3d": "3d",
            "full3d": "3d",
            "3d": "3d",
            "2d": "2d",
            "slice": "3d_slice",
            "3d_slice": "3d_slice",
            "slab": "3d_slab",
            "3d_slab": "3d_slab",
        }
        if mode_norm not in aliases:
            raise ValueError(
                f"Unsupported 3D JEPA mode={mode}. "
                "Allowed: 3d, 2d, 3d_slice, 3d_slab."
            )
        return aliases[mode_norm]

    def make_pyramid(self, x):
        return make_gaussian_pyramid3d(x, sigmas=self.sigmas)

    def _make_random_box_mask3d(
        self,
        batch_size: int,
        depth: int,
        height: int,
        width: int,
        device,
        focus_slice_idx: torch.Tensor | None = None,
        focus_slab_start_idx: torch.Tensor | None = None,
    ):
        box = max(1, int(self.mask_box_size))
        n_box = max(1, int(self.num_mask_boxes))
        mask = torch.zeros((batch_size, 1, depth, height, width), device=device)
        z_lim = max(1, depth - box + 1)
        y_lim = max(1, height - box + 1)
        x_lim = max(1, width - box + 1)
        z0 = torch.empty((batch_size, n_box), device=device, dtype=torch.long)

        if focus_slice_idx is not None:
            focus_slice_cpu = focus_slice_idx.detach().to("cpu").numpy()
            for b in range(batch_size):
                center = int(focus_slice_cpu[b])
                lo = max(0, center - box + 1)
                hi = min(z_lim, center + 1)
                if hi <= lo:
                    lo, hi = 0, z_lim
                z0[b] = torch.randint(lo, hi, (n_box,), device=device)
        elif focus_slab_start_idx is not None:
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
        else:
            z0 = torch.randint(0, z_lim, (batch_size, n_box), device=device)

        y0 = torch.randint(0, y_lim, (batch_size, n_box), device=device)
        x0 = torch.randint(0, x_lim, (batch_size, n_box), device=device)
        # Move once to host to avoid repeated GPU->CPU scalar sync in Python loops.
        z0_cpu = z0.detach().to("cpu").numpy()
        y0_cpu = y0.detach().to("cpu").numpy()
        x0_cpu = x0.detach().to("cpu").numpy()
        for b in range(batch_size):
            for j in range(n_box):
                zz = int(z0_cpu[b, j])
                yy = int(y0_cpu[b, j])
                xx = int(x0_cpu[b, j])
                mask[b, 0, zz : zz + box, yy : yy + box, xx : xx + box] = 1.0
        return mask

    def _sample_slice_index(self, batch_size: int, depth: int, device) -> torch.Tensor:
        if self.training:
            return torch.randint(0, max(1, depth), (batch_size,), device=device)
        return torch.full((batch_size,), int(depth // 2), dtype=torch.long, device=device)

    def _sample_slab_start_index(self, batch_size: int, depth: int, device) -> tuple[torch.Tensor, int]:
        slab_depth = max(1, min(int(self.slab_depth), int(depth)))
        max_start = max(0, depth - slab_depth)
        if max_start == 0:
            return torch.zeros((batch_size,), dtype=torch.long, device=device), slab_depth

        edge = max(0, int(self.slab_boundary_margin))
        lo = min(max_start, edge)
        hi_inclusive = max(lo, max_start - edge)
        if self.training:
            starts = torch.randint(lo, hi_inclusive + 1, (batch_size,), device=device)
        else:
            starts = torch.full(
                (batch_size,),
                int((lo + hi_inclusive) // 2),
                dtype=torch.long,
                device=device,
            )
        return starts, slab_depth

    @staticmethod
    def _gather_slices(z: torch.Tensor, slice_idx: torch.Tensor) -> torch.Tensor:
        b, c, _, h, w = z.shape
        gather_idx = slice_idx.view(b, 1, 1, 1, 1).expand(b, c, 1, h, w)
        return z.gather(dim=2, index=gather_idx).squeeze(2)

    @staticmethod
    def _gather_slabs(z: torch.Tensor, slab_starts: torch.Tensor, slab_depth: int) -> torch.Tensor:
        b, c, d, h, w = z.shape
        offsets = torch.arange(slab_depth, device=z.device).view(1, slab_depth)
        slab_idx = (slab_starts.view(b, 1) + offsets).clamp(0, d - 1)
        gather_idx = slab_idx.view(b, 1, slab_depth, 1, 1).expand(b, c, slab_depth, h, w)
        return z.gather(dim=2, index=gather_idx)

    @staticmethod
    def _sample_target_locations_2d(
        batch_size: int,
        height: int,
        width: int,
        num_targets: int,
        patch_size: int,
        device,
    ):
        half = patch_size // 2
        lo_y = half
        hi_y = height - (patch_size - half)
        lo_x = half
        hi_x = width - (patch_size - half)
        if hi_y <= lo_y or hi_x <= lo_x:
            raise ValueError("Patch too large for 2D feature map")

        y0 = torch.randint(lo_y, hi_y + 1, (batch_size, num_targets), device=device)
        x0 = torch.randint(lo_x, hi_x + 1, (batch_size, num_targets), device=device)
        loc = torch.stack([y0, x0], dim=-1)
        valid = torch.ones((batch_size, num_targets), dtype=torch.bool, device=device)
        return loc, valid

    def forward(self, x_clean, **kwargs):
        if x_clean.dim() != 5:
            raise ValueError(f"Expected Bx1xDxHxW, got {tuple(x_clean.shape)}")

        b, _, _, _, _ = x_clean.shape
        fields = self.make_pyramid(x_clean)
        _, _, d, h, w = fields.shape

        slice_idx = None
        slab_starts = None
        slab_depth = None
        if self.mode in {"2d", "3d_slice"}:
            slice_idx = self._sample_slice_index(batch_size=b, depth=d, device=x_clean.device)
        elif self.mode == "3d_slab":
            slab_starts, slab_depth = self._sample_slab_start_index(batch_size=b, depth=d, device=x_clean.device)

        fields_context = fields
        mask_tokens = torch.zeros_like(fields)
        if self.constant_mask_box:
            box_mask = self._make_random_box_mask3d(
                b,
                d,
                h,
                w,
                x_clean.device,
                focus_slice_idx=slice_idx,
                focus_slab_start_idx=slab_starts,
            )
            fields_context = fields * (1.0 - box_mask)
            mask_tokens = box_mask.expand(-1, fields.shape[1], -1, -1, -1)

        context_map_3d = self.context_encoder(fields_context, mask_tokens=mask_tokens)
        with torch.no_grad():
            gt_map_3d = self.target_encoder(fields, mask_tokens=torch.zeros_like(fields))

        if self.mode == "3d":
            context_map = context_map_3d
            gt_map = gt_map_3d
            pred_map = self.predictor3d(context_map)
            _, _, dz, hy, wx = pred_map.shape
            target_locations, target_valid = sample_target_locations_3d(
                batch_size=b,
                depth=dz,
                height=hy,
                width=wx,
                num_targets=self.num_targets,
                patch_size=self.patch_size,
                device=x_clean.device,
            )
            pred_patches = extract_location_cubes(pred_map, target_locations, self.patch_size)
            gt_patches = extract_location_cubes(gt_map, target_locations, self.patch_size)
        elif self.mode in {"2d", "3d_slice"}:
            context_map = self._gather_slices(context_map_3d, slice_idx)
            gt_map = self._gather_slices(gt_map_3d, slice_idx)
            pred_map = self.predictor2d(context_map)
            _, _, hy, wx = pred_map.shape
            target_locations, target_valid = self._sample_target_locations_2d(
                batch_size=b,
                height=hy,
                width=wx,
                num_targets=self.num_targets,
                patch_size=self.patch_size,
                device=x_clean.device,
            )
            pred_patches = extract_location_patches(pred_map, target_locations, self.patch_size)
            gt_patches = extract_location_patches(gt_map, target_locations, self.patch_size)
        elif self.mode == "3d_slab":
            context_map = self._gather_slabs(context_map_3d, slab_starts, slab_depth)
            gt_map = self._gather_slabs(gt_map_3d, slab_starts, slab_depth)
            pred_map = self.predictor3d(context_map)
            _, _, dz, hy, wx = pred_map.shape
            target_locations, target_valid = sample_target_locations_3d(
                batch_size=b,
                depth=dz,
                height=hy,
                width=wx,
                num_targets=self.num_targets,
                patch_size=self.patch_size,
                device=x_clean.device,
            )
            pred_patches = extract_location_cubes(pred_map, target_locations, self.patch_size)
            gt_patches = extract_location_cubes(gt_map, target_locations, self.patch_size)
        else:
            raise RuntimeError(f"Unsupported mode at runtime: {self.mode}")

        extra_outputs = {}
        if slice_idx is not None:
            extra_outputs["selected_slice_index"] = slice_idx
        if slab_starts is not None:
            extra_outputs["selected_slab_start_index"] = slab_starts
            extra_outputs["selected_slab_depth"] = torch.full(
                (b,),
                int(slab_depth),
                device=x_clean.device,
                dtype=torch.long,
            )

        return {
            "pred_patches": pred_patches,
            "gt_patches": gt_patches,
            "target_locations": target_locations,
            "target_valid": target_valid,
            "target_scales": torch.ones((b, self.num_targets), device=x_clean.device, dtype=x_clean.dtype),
            "context_map": context_map,
            "context_map_3d": context_map_3d,
            "pred_map": pred_map,
            "gt_map": gt_map,
            "gt_map_3d": gt_map_3d,
            "x_clean": x_clean,
            "x_context": x_clean,
            **extra_outputs,
        }

    def compute_loss(self, outputs):
        pred = outputs["pred_patches"]
        gt = outputs["gt_patches"].detach()
        valid = outputs["target_valid"]

        if self.normalize_loss:
            pred = F.normalize(pred, dim=2)
            gt = F.normalize(gt, dim=2)

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
