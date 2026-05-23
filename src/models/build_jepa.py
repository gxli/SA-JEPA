import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dense_unet import DenseUNetSmallEncoder
from .cdd_opnet import CDDOpNetEncoder
from .encoders import (
    CDDScaleAwareConvNeXtEncoder,
    ConvNeXtDenseEncoder,
    FullResEncoder,
    PyramidConvNeXtDilatedEncoder,
    PyramidResDilatedEncoder,
    ResCNNDenseEncoder,
)
from .masking import (
    extract_location_patches,
    make_pyramid_grid_context,
    norm_per_sample_channel,
)
from .mfae_convnext import MFAEConvNeXtDenseEncoder
from .predictor import FullResPredictor


class PyramidGridJEPA(nn.Module):
    def __init__(
        self,
        latent_channels: int = 32,
        predictor_hidden: int = None,
        patch_size: int = 2,
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
        scaleaware_gaussian_ratios=(0.25, 0.5, 1.0, 2.0),
        cdd_append_last_residual: bool = True,
        post_log_transform: bool = True,
        log_eps: float = 1.0,
        cdd_log_std_floor_mult: float = 0.05,
        ema_momentum: float = 0.996,
        normalize_loss: bool = True,
        predictor_layernorm: bool = False,
        mode: str = "image",
        encoder_type: str = "fullres",
        encoder_width: int = 32,
        encoder_depth: int = 4,
        encoder_kernel_size: int = 7,
        encoder_norm_type: Optional[str] = None,
        encoder_norm_groups: Optional[int] = None,
        encoder_norm_eps: Optional[float] = None,
        scaleaware_feat_channels: int = 8,
        scaleaware_adapter_kernel_size: int = 3,
        scaleaware_fusion_type: str = "concat",
        scaleaware_norm_per_scale: bool = False,
        mfae_scales=(1, 2, 4),
        mfae_features=("x", "gradmag", "abslap", "local_std"),
        mfae_normalize_attributes: bool = False,
        mfae_include_mask_tokens: bool = True,
        opnet_dilation_mode: str = "half_cdd_scale",
        opnet_dilations=None,
        opnet_max_dilation: int = 16,
        opnet_channel_mode: str = "multi",
        op_smoothing_mode: str = "sqrt_scale",
        op_smoothing_mult: float = 1.0,
        op_smoothing_padding_mode: str = "reflect",
        opnet_cache_primitives: bool = True,
        opnet_cache_detach: bool = True,
        target_invalid_region_skip: bool = False,
        target_invalid_region_values=(0.0, "nan"),
    ):
        super().__init__()

        self.patch_size = patch_size
        self.sigmas = tuple(sigmas)
        self.cell_sizes = tuple(cell_sizes)
        self.mask_fraction = float(mask_fraction)
        self.box_sigma_mult = float(box_sigma_mult)
        self.mask_scale = float(mask_scale)
        self.min_mask_scale = float(min_mask_scale)
        self.spacing_scale = float(spacing_scale)
        self.mask_size = float(mask_size)
        self.full_grid = bool(full_grid)
        self.global_shift = bool(global_shift)
        self.align_scales = bool(align_scales)
        self.constant_mask_box = bool(constant_mask_box)
        self.mask_box_size = int(mask_box_size)
        self.blur_mode = str(blur_mode)
        self.cdd_mode = str(cdd_mode)
        self.cdd_constrained = bool(cdd_constrained)
        self.cdd_sm_mode = str(cdd_sm_mode)
        self.mask_fill_mode = str(mask_fill_mode)
        self.dip_sigma_mult = float(dip_sigma_mult)
        self.constant_gaussian_sigma = float(constant_gaussian_sigma)
        if scaleaware_gaussian_ratios is None:
            self.scaleaware_gaussian_ratios = (0.25, 0.5, 1.0, 2.0)
        else:
            self.scaleaware_gaussian_ratios = tuple(float(v) for v in scaleaware_gaussian_ratios)
        self.cdd_append_last_residual = bool(cdd_append_last_residual)
        self.post_log_transform = bool(post_log_transform)
        self.log_eps = float(log_eps)
        self.cdd_log_std_floor_mult = float(cdd_log_std_floor_mult)
        self.ema_momentum = float(ema_momentum)
        self.normalize_loss = bool(normalize_loss)
        self.predictor_layernorm = bool(predictor_layernorm)
        self.mode = str(mode)
        self.encoder_type = str(encoder_type)
        self.encoder_width = int(encoder_width)
        self.encoder_depth = int(encoder_depth)
        self.encoder_kernel_size = int(encoder_kernel_size)
        self.encoder_norm_type = None if encoder_norm_type is None else str(encoder_norm_type).lower()
        self.encoder_norm_groups = None if encoder_norm_groups is None else int(encoder_norm_groups)
        self.encoder_norm_eps = None if encoder_norm_eps is None else float(encoder_norm_eps)
        self.mfae_scales = tuple(mfae_scales)
        self.mfae_features = tuple(mfae_features)
        self.mfae_normalize_attributes = bool(mfae_normalize_attributes)
        self.mfae_include_mask_tokens = bool(mfae_include_mask_tokens)
        self.scaleaware_feat_channels = int(scaleaware_feat_channels)
        self.scaleaware_adapter_kernel_size = int(scaleaware_adapter_kernel_size)
        self.scaleaware_fusion_type = str(scaleaware_fusion_type).lower()
        self.scaleaware_norm_per_scale = bool(scaleaware_norm_per_scale)
        self.opnet_dilation_mode = str(opnet_dilation_mode)
        self.opnet_dilations = opnet_dilations
        self.opnet_max_dilation = int(opnet_max_dilation)
        self.opnet_channel_mode = str(opnet_channel_mode)
        self.op_smoothing_mode = str(op_smoothing_mode)
        self.op_smoothing_mult = float(op_smoothing_mult)
        self.op_smoothing_padding_mode = str(op_smoothing_padding_mode)
        self.target_invalid_region_skip = bool(target_invalid_region_skip)
        if target_invalid_region_values is None:
            self.target_invalid_region_values = (0.0, "nan")
        else:
            self.target_invalid_region_values = tuple(target_invalid_region_values)
        self.opnet_cache_primitives = bool(opnet_cache_primitives)
        self.opnet_cache_detach = bool(opnet_cache_detach)
        if self.mode not in ("image", "pyramid"):
            raise ValueError(f"Unknown mode={self.mode}; expected 'image' or 'pyramid'")
        if self.encoder_type == "pyramid_cnn_res_dilated":
            norm_type = self.encoder_norm_type if self.encoder_norm_type is not None else "layernorm"
            norm_groups = self.encoder_norm_groups if self.encoder_norm_groups is not None else 8
            norm_eps = self.encoder_norm_eps if self.encoder_norm_eps is not None else 1e-6
            # Per-scale map + per-scale masked-token map.
            pyr_in_channels = 2 * max(1, len(self.sigmas))
            self.context_encoder = PyramidResDilatedEncoder(
                in_channels=pyr_in_channels,
                hidden_channels=self.encoder_width,
                latent_channels=latent_channels,
                depth=self.encoder_depth,
                final_norm=True,
                norm_type=norm_type,
                norm_groups=norm_groups,
                norm_eps=norm_eps,
            )
        elif self.encoder_type == "pyramid_convnext_dilated":
            norm_type = self.encoder_norm_type if self.encoder_norm_type is not None else "layernorm"
            norm_groups = self.encoder_norm_groups if self.encoder_norm_groups is not None else 8
            norm_eps = self.encoder_norm_eps if self.encoder_norm_eps is not None else 1e-6
            # Per-scale map + per-scale masked-token map.
            pyr_in_channels = 2 * max(1, len(self.sigmas))
            self.context_encoder = PyramidConvNeXtDilatedEncoder(
                in_channels=pyr_in_channels,
                hidden_channels=self.encoder_width,
                latent_channels=latent_channels,
                depth=max(10, self.encoder_depth),
                final_norm=True,
                norm_type=norm_type,
                norm_groups=norm_groups,
                norm_eps=norm_eps,
            )
        elif self.encoder_type == "convnext_dense_pyramid":
            # Pyramid input = per-scale CDD channels + per-scale mask/indicator channels.
            pyr_in_channels = 2 * max(1, len(self.sigmas))
            self.context_encoder = ConvNeXtDenseEncoder(
                in_channels=pyr_in_channels,
                hidden_channels=self.encoder_width,
                latent_channels=latent_channels,
                depth=self.encoder_depth,
                kernel_size=self.encoder_kernel_size,
                expansion=4,
                use_reflect_padding=True,
                final_norm=True,
            )
        elif self.encoder_type == "rescnn_dense_pyramid":
            # Pyramid input = per-scale CDD channels + per-scale mask/indicator channels.
            pyr_in_channels = 2 * max(1, len(self.sigmas))
            norm_type = self.encoder_norm_type if self.encoder_norm_type is not None else "groupnorm"
            norm_groups = self.encoder_norm_groups if self.encoder_norm_groups is not None else 1
            norm_eps = self.encoder_norm_eps if self.encoder_norm_eps is not None else 1e-5
            self.context_encoder = ResCNNDenseEncoder(
                in_channels=pyr_in_channels,
                hidden_channels=self.encoder_width,
                latent_channels=latent_channels,
                depth=self.encoder_depth,
                final_norm=True,
                norm_type=norm_type,
                norm_groups=norm_groups,
                norm_eps=norm_eps,
            )
        elif self.encoder_type == "cdd_opnet":
            self.context_encoder = CDDOpNetEncoder(
                field_channels=max(1, len(self.sigmas)),
                scales=tuple(float(s) for s in self.sigmas),
                latent_channels=latent_channels,
                hidden_channels=self.encoder_width,
                depth=self.encoder_depth,
                kernel_size=self.encoder_kernel_size,
                expansion=4,
                use_reflect_padding=True,
                final_norm=True,
                include_mask_tokens=True,
                log_eps=self.log_eps,
                log_std_floor_mult=self.cdd_log_std_floor_mult,
                opnet_dilation_mode=self.opnet_dilation_mode,
                opnet_dilations=self.opnet_dilations,
                opnet_max_dilation=self.opnet_max_dilation,
                opnet_channel_mode=self.opnet_channel_mode,
                op_smoothing_mode=self.op_smoothing_mode,
                op_smoothing_mult=self.op_smoothing_mult,
                op_smoothing_padding_mode=self.op_smoothing_padding_mode,
                cache_primitives=self.opnet_cache_primitives,
                cache_detach=self.opnet_cache_detach,
            )
        elif self.encoder_type == "cdd_scaleaware_convnext":
            if self.mode != "pyramid":
                raise ValueError("cdd_scaleaware_convnext requires mode='pyramid'.")
            self.context_encoder = CDDScaleAwareConvNeXtEncoder(
                scales=tuple(float(s) for s in self.sigmas),
                hidden_channels=self.encoder_width,
                latent_channels=latent_channels,
                depth=self.encoder_depth,
                kernel_size=self.encoder_kernel_size,
                expansion=4,
                scale_feat_channels=self.scaleaware_feat_channels,
                adapter_kernel_size=self.scaleaware_adapter_kernel_size,
                fusion_type=self.scaleaware_fusion_type,
                use_reflect_padding=True,
                final_norm=True,
                cdd_append_last_residual=self.cdd_append_last_residual,
            )
        elif self.encoder_type == "fullres":
            self.context_encoder = FullResEncoder(in_channels=1, latent_channels=latent_channels)
        elif self.encoder_type == "convnext_dense":
            self.context_encoder = ConvNeXtDenseEncoder(
                in_channels=1,
                hidden_channels=self.encoder_width,
                latent_channels=latent_channels,
                depth=self.encoder_depth,
                kernel_size=self.encoder_kernel_size,
                expansion=4,
                use_reflect_padding=True,
                final_norm=True,
            )
        elif self.encoder_type == "mfae_convnext":
            if self.mode == "pyramid":
                field_channels = len(self.sigmas)
                include_mask_tokens = bool(self.mfae_include_mask_tokens)
            elif self.mode == "image":
                field_channels = 1
                include_mask_tokens = False
            else:
                raise ValueError(
                    f"mfae_convnext supports mode='image' or mode='pyramid', got mode={self.mode}"
                )
            self.context_encoder = MFAEConvNeXtDenseEncoder(
                field_channels=field_channels,
                hidden_channels=self.encoder_width,
                latent_channels=latent_channels,
                depth=self.encoder_depth,
                kernel_size=self.encoder_kernel_size,
                expansion=4,
                use_reflect_padding=True,
                final_norm=True,
                mfae_scales=self.mfae_scales,
                mfae_features=self.mfae_features,
                mfae_normalize_attributes=self.mfae_normalize_attributes,
                include_mask_tokens=include_mask_tokens,
            )
        elif self.encoder_type == "dense_unet_small":
            self.context_encoder = DenseUNetSmallEncoder(
                in_channels=1,
                width=self.encoder_width,
                latent_channels=latent_channels,
                groups=8,
                final_norm=True,
            )
        elif self.encoder_type == "rescnn_dense":
            norm_type = self.encoder_norm_type if self.encoder_norm_type is not None else "groupnorm"
            norm_groups = self.encoder_norm_groups if self.encoder_norm_groups is not None else 1
            norm_eps = self.encoder_norm_eps if self.encoder_norm_eps is not None else 1e-5
            self.context_encoder = ResCNNDenseEncoder(
                in_channels=1,
                hidden_channels=self.encoder_width,
                latent_channels=latent_channels,
                depth=self.encoder_depth,
                final_norm=True,
                norm_type=norm_type,
                norm_groups=norm_groups,
                norm_eps=norm_eps,
            )
        else:
            raise ValueError(f"Unknown encoder_type={self.encoder_type}")

        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        if predictor_hidden is None:
            predictor_hidden = latent_channels * 2
        self.predictor = FullResPredictor(
            channels=latent_channels,
            hidden=int(predictor_hidden),
            use_layernorm=self.predictor_layernorm,
        )

    def forward(
        self,
        x_clean,
        return_debug: bool = False,
        forced_grid_shift: Optional[Tuple[int, int]] = None,
        enable_grid_jitter: bool = True,
        mask_inference: bool = True,
    ):
        """
        x_clean: B x 1 x H x W
        """
        if x_clean.dim() != 4:
            raise ValueError(f"Expected BxCxHxW, got {tuple(x_clean.shape)}")

        if x_clean.shape[1] != 1:
            raise ValueError(f"Expected grayscale input, got {x_clean.shape[1]} channels")
        invalid_pixel_mask = ~torch.isfinite(x_clean)
        if invalid_pixel_mask.any():
            x_clean = torch.nan_to_num(x_clean, nan=0.0, posinf=0.0, neginf=0.0)

        need_debug_tensors = bool(return_debug or self.mode == "pyramid")
        if need_debug_tensors:
            x_context, target_locations, target_scales, target_valid, debug = make_pyramid_grid_context(
                x_clean=x_clean,
                sigmas=self.sigmas,
                cell_sizes=self.cell_sizes,
                mask_fraction=self.mask_fraction,
                box_sigma_mult=self.box_sigma_mult,
                mask_scale=self.mask_scale,
                min_mask_scale=self.min_mask_scale,
                spacing_scale=self.spacing_scale,
                mask_size=self.mask_size,
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
                constant_gaussian_sigma=self.constant_gaussian_sigma,
                scaleaware_gaussian_ratios=self.scaleaware_gaussian_ratios,
                cdd_append_last_residual=self.cdd_append_last_residual,
                inner_target_size=self.patch_size,
                return_debug=True,
                forced_grid_shift=forced_grid_shift,
                enable_grid_jitter=enable_grid_jitter,
                target_invalid_region_skip=self.target_invalid_region_skip,
                target_invalid_region_values=self.target_invalid_region_values,
                invalid_pixel_mask=invalid_pixel_mask,
            )
        else:
            x_context, target_locations, target_scales, target_valid = make_pyramid_grid_context(
                x_clean=x_clean,
                sigmas=self.sigmas,
                cell_sizes=self.cell_sizes,
                mask_fraction=self.mask_fraction,
                box_sigma_mult=self.box_sigma_mult,
                mask_scale=self.mask_scale,
                min_mask_scale=self.min_mask_scale,
                spacing_scale=self.spacing_scale,
                mask_size=self.mask_size,
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
                constant_gaussian_sigma=self.constant_gaussian_sigma,
                scaleaware_gaussian_ratios=self.scaleaware_gaussian_ratios,
                cdd_append_last_residual=self.cdd_append_last_residual,
                inner_target_size=self.patch_size,
                forced_grid_shift=forced_grid_shift,
                enable_grid_jitter=enable_grid_jitter,
                target_invalid_region_skip=self.target_invalid_region_skip,
                target_invalid_region_values=self.target_invalid_region_values,
                invalid_pixel_mask=invalid_pixel_mask,
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

        # Optional pyramid-mode path: encode multiscale channel cubes directly.
        # Keep x_clean/x_context image outputs for backward-compatible diagnostics.
        enc_target = x_clean_enc
        enc_context = x_context_enc
        if self.mode == "pyramid":
            cdd_orig = debug["cdd_channels_orig"].to(dtype=x_clean.dtype)
            cdd_masked = debug["cdd_channels_masked"].to(dtype=x_clean.dtype)
            dip_per_ch = debug["dip_field_per_channel"].to(dtype=x_clean.dtype)
            zero_token = torch.zeros_like(dip_per_ch)
            # target: original per-scale channels + zero token maps
            enc_target = torch.cat([cdd_orig, zero_token], dim=1)
            # context: masked per-scale channels + mask token maps
            enc_context = torch.cat([cdd_masked, dip_per_ch], dim=1)
            if not bool(mask_inference):
                # In mask-free inference, predictor branch should consume clean features.
                enc_context = enc_target
        if self.encoder_type == "mfae_convnext":
            if self.mode == "image":
                context_input = x_context_enc if bool(mask_inference) else x_clean_enc
                context_map = self.context_encoder(context_input)
                with torch.no_grad():
                    gt_map = self.target_encoder(x_clean_enc)
            elif self.mode == "pyramid":
                cdd_orig = debug["cdd_channels_orig"].to(dtype=x_clean.dtype)
                cdd_masked = debug["cdd_channels_masked"].to(dtype=x_clean.dtype)
                mask_tokens = debug["dip_field_per_channel"].to(dtype=x_clean.dtype)
                if bool(mask_inference):
                    context_map = self.context_encoder(cdd_masked, mask_tokens=mask_tokens)
                else:
                    context_map = self.context_encoder(cdd_orig, mask_tokens=torch.zeros_like(mask_tokens))
                with torch.no_grad():
                    gt_map = self.target_encoder(cdd_orig, mask_tokens=torch.zeros_like(mask_tokens))
            else:
                raise ValueError(
                    f"mfae_convnext supports mode='image' or mode='pyramid', got mode={self.mode}"
                )
        elif self.encoder_type == "cdd_opnet":
            if self.mode != "pyramid":
                raise ValueError("cdd_opnet requires mode='pyramid'.")
            cdd_orig = debug["cdd_channels_orig"].to(dtype=x_clean.dtype)
            cdd_masked = debug["cdd_channels_masked"].to(dtype=x_clean.dtype)
            mask_tokens = debug["dip_field_per_channel"].to(dtype=x_clean.dtype)
            if bool(mask_inference):
                context_map = self.context_encoder(cdd_masked, mask_tokens=mask_tokens, floor_source=cdd_orig)
            else:
                context_map = self.context_encoder(
                    cdd_orig,
                    mask_tokens=torch.zeros_like(mask_tokens),
                    floor_source=cdd_orig,
                )
            with torch.no_grad():
                gt_map = self.target_encoder(
                    cdd_orig,
                    mask_tokens=torch.zeros_like(mask_tokens),
                    floor_source=cdd_orig,
                )
        elif self.encoder_type == "cdd_scaleaware_convnext":
            if self.mode != "pyramid":
                raise ValueError("cdd_scaleaware_convnext requires mode='pyramid'.")
            cdd_orig = debug["cdd_channels_orig"].to(dtype=x_clean.dtype)
            cdd_masked = debug["cdd_channels_masked"].to(dtype=x_clean.dtype)
            mask_tokens = debug["dip_field_per_channel"].to(dtype=x_clean.dtype)
            if self.scaleaware_norm_per_scale:
                cdd_orig = norm_per_sample_channel(cdd_orig)
                cdd_masked = norm_per_sample_channel(cdd_masked)
            if bool(mask_inference):
                context_map = self.context_encoder(cdd_masked, mask_tokens=mask_tokens)
            else:
                context_map = self.context_encoder(cdd_orig, mask_tokens=torch.zeros_like(mask_tokens))
            with torch.no_grad():
                gt_map = self.target_encoder(cdd_orig, mask_tokens=torch.zeros_like(mask_tokens))
        else:
            with torch.no_grad():
                gt_map = self.target_encoder(enc_target)
            context_map = self.context_encoder(enc_context)
        pred_map = self.predictor(context_map)

        pred_patches = extract_location_patches(pred_map, target_locations, patch_size=self.patch_size)
        gt_patches = extract_location_patches(gt_map, target_locations, patch_size=self.patch_size)

        out = {
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
        if return_debug or self.mode == "pyramid":
            # Exact applied hard mask footprint from make_pyramid_grid_context.
            if return_debug:
                out["target_mask_map"] = debug["mask_map"].unsqueeze(1).to(dtype=x_clean.dtype)
            out["cdd_channels_orig"] = debug["cdd_channels_orig"].to(dtype=x_clean.dtype)
            out["cdd_channels_masked"] = debug["cdd_channels_masked"].to(dtype=x_clean.dtype)
            out["dip_field_per_channel"] = debug["dip_field_per_channel"].to(dtype=x_clean.dtype)
            out["pyramid_mask_token"] = debug["dip_field_per_channel"].to(dtype=x_clean.dtype)
        return out

    def compute_loss(self, outputs):
        pred = outputs["pred_patches"]
        gt = outputs["gt_patches"].detach()

        valid = outputs["target_valid"]  # B x K (bool)

        if self.normalize_loss:
            pred = F.normalize(pred, dim=2)
            gt = F.normalize(gt, dim=2)
        loss_map = F.mse_loss(pred, gt, reduction="none")  # B x K x C x P x P
        w = valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(loss_map.dtype)
        if not bool(valid.any().item()):
            # No valid targets in this batch: return graph-connected zero loss.
            return loss_map.sum() * 0.0
        denom = torch.clamp(w.sum() * loss_map.shape[2] * loss_map.shape[3] * loss_map.shape[4], min=1.0)
        return (loss_map * w).sum() / denom

    @torch.no_grad()
    def update_target_encoder(self):
        for p_context, p_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(self.ema_momentum).add_((1.0 - self.ema_momentum) * p_context.detach().data)
