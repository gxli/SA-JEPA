from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders3d import ScaleAwareConvNeXt3DEncoder
from .masking3d import extract_location_cubes, make_gaussian_pyramid3d, sample_target_locations_3d
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
        self.predictor = FullResPredictor3D(
            channels=int(latent_channels),
            hidden=max(2 * int(latent_channels), 32),
        )

    def make_pyramid(self, x):
        return make_gaussian_pyramid3d(x, sigmas=self.sigmas)

    def _make_random_box_mask3d(self, batch_size: int, depth: int, height: int, width: int, device):
        box = max(1, int(self.mask_box_size))
        n_box = max(1, int(self.num_mask_boxes))
        mask = torch.zeros((batch_size, 1, depth, height, width), device=device)
        z_lim = max(1, depth - box + 1)
        y_lim = max(1, height - box + 1)
        x_lim = max(1, width - box + 1)
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

    def forward(self, x_clean, **kwargs):
        if x_clean.dim() != 5:
            raise ValueError(f"Expected Bx1xDxHxW, got {tuple(x_clean.shape)}")

        b, _, _, _, _ = x_clean.shape
        fields = self.make_pyramid(x_clean)
        fields_context = fields
        mask_tokens = torch.zeros_like(fields)
        if self.constant_mask_box:
            _, _, d, h, w = fields.shape
            box_mask = self._make_random_box_mask3d(b, d, h, w, x_clean.device)
            fields_context = fields * (1.0 - box_mask)
            mask_tokens = box_mask.expand(-1, fields.shape[1], -1, -1, -1)

        context_map = self.context_encoder(fields_context, mask_tokens=mask_tokens)
        with torch.no_grad():
            gt_map = self.target_encoder(fields, mask_tokens=torch.zeros_like(fields))

        pred_map = self.predictor(context_map)
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

        pred_cubes = extract_location_cubes(pred_map, target_locations, self.patch_size)
        gt_cubes = extract_location_cubes(gt_map, target_locations, self.patch_size)

        return {
            "pred_patches": pred_cubes,
            "gt_patches": gt_cubes,
            "target_locations": target_locations,
            "target_valid": target_valid,
            "target_scales": torch.ones((b, self.num_targets), device=x_clean.device, dtype=x_clean.dtype),
            "context_map": context_map,
            "pred_map": pred_map,
            "gt_map": gt_map,
            "x_clean": x_clean,
            "x_context": x_clean,
        }

    def compute_loss(self, outputs):
        pred = outputs["pred_patches"]
        gt = outputs["gt_patches"].detach()
        valid = outputs["target_valid"]

        if self.normalize_loss:
            pred = F.normalize(pred, dim=2)
            gt = F.normalize(gt, dim=2)

        loss_map = F.mse_loss(pred, gt, reduction="none")
        w = valid.view(valid.shape[0], valid.shape[1], 1, 1, 1, 1).to(loss_map.dtype)

        if not bool(valid.any().item()):
            return loss_map.sum() * 0.0

        denom = torch.clamp(
            w.sum()
            * loss_map.shape[2]
            * loss_map.shape[3]
            * loss_map.shape[4]
            * loss_map.shape[5],
            min=1.0,
        )
        return (loss_map * w).sum() / denom

    @torch.no_grad()
    def update_target_encoder(self):
        for p_context, p_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_target.mul_(self.ema_momentum).add_(p_context.detach(), alpha=1.0 - self.ema_momentum)
