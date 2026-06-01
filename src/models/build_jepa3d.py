from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders3d import ScaleAwareConvNeXt3DEncoder, ScaleFiLMConvNeXt3DEncoder
from .masking3d import extract_location_cubes, sample_target_locations_3d
from .predictor3d import FullResPredictor3D
from .symmetry import symmetric_forward_3d


class PyramidGridJEPA3D(nn.Module):
    def __init__(
        self,
        latent_channels=16,
        scale_channels=8,
        patch_size=2,
        num_targets=32,
        encoder_depth=3,
        encoder_kernel_size=5,
        encoder_stride=1,
        ema_momentum=0.996,
        normalize_loss_l2=False,
        post_log_transform: bool = True,
        log_eps: float = 1e-6,
        fusion="gate",
        mask_box_size: int = 8,
        num_mask_boxes: int = 8,
        slab_depth: int = 3,
        use_symmetric_feature_loss: bool = False,
        use_film: bool = True,
        use_per_scale_adapters: bool = False,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.num_targets = int(num_targets)
        self.ema_momentum = float(ema_momentum)
        self.normalize_loss_l2 = bool(normalize_loss_l2)
        self.post_log_transform = bool(post_log_transform)
        self.log_eps = float(log_eps)
        self.mask_box_size = int(mask_box_size)
        self.num_mask_boxes = int(num_mask_boxes)
        self.mode = "3d_slab"
        self.slab_depth = max(self.patch_size, int(slab_depth))
        self.use_symmetric_feature_loss = bool(use_symmetric_feature_loss)
        self.use_film = bool(use_film)
        self.use_per_scale_adapters = bool(use_per_scale_adapters)

        if self.use_film or self.use_per_scale_adapters:
            self.context_encoder = ScaleFiLMConvNeXt3DEncoder(
                num_scales=1,
                out_channels=int(latent_channels),
                scale_channels=int(scale_channels),
                depth=int(encoder_depth),
                kernel_size=int(encoder_kernel_size),
                stride=int(encoder_stride),
                fusion=str(fusion),
                use_film=self.use_film,
                use_per_scale_adapters=self.use_per_scale_adapters,
            )
        else:
            self.context_encoder = ScaleAwareConvNeXt3DEncoder(
                num_scales=1,
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
        n_box = max(1, int(self.num_mask_boxes))
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
            raise ValueError(f"Expected Bx1xDxHxW, got {tuple(x_clean.shape)}")

        b, _, _, _, _ = x_clean.shape
        fields = self.make_fields(x_clean)
        _, _, d, h, w = fields.shape

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
            eps = max(1e-30, float(self.log_eps))
            fields = torch.log(torch.clamp(fields, min=0.0) + eps)
            fields_context = torch.log(torch.clamp(fields_context, min=0.0) + eps)

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
        context_patches = extract_location_cubes(context_map, target_locations, self.patch_size)

        out = {
            "pred_patches": pred_patches,
            "gt_patches": gt_patches,
            "context_patches": context_patches,
            "target_locations": target_locations,
            "target_valid": target_valid,
            "target_scales": torch.ones((b, self.num_targets), device=x_clean.device, dtype=x_clean.dtype),
            "context_map": context_map,
            "context_map_3d": context_map_3d,
            "pred_map": pred_map,
            "gt_map": gt_map,
            "gt_map_3d": gt_map_3d,
            "x_clean": x_clean,
            "x_context": fields_context,
            "mask_cube": box_mask,
            "selected_slab_start_index": slab_starts,
            "selected_slab_depth": torch.full((b,), int(slab_depth), device=x_clean.device, dtype=torch.long),
        }
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
            b, k = pred.shape[:2]
            cube_shape = pred.shape[2:]
            pred = F.normalize(pred.reshape(b, k, -1), dim=2).reshape(b, k, *cube_shape)
            gt = F.normalize(gt.reshape(b, k, -1), dim=2).reshape(b, k, *cube_shape)
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
