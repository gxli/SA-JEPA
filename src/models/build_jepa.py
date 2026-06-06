from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import (
    CDDScaleAwareConvNeXtEncoder,
    ConvNeXtDenseEncoder,
    LayerNorm2d,
)
from .masking import (
    extract_location_patches,
    make_pyramid_grid_context,
    norm_per_sample_channel,
    normalize_target_sampling_mode,
    prepare_context_batch,
)
from .predictor import FullResPredictor
from .symmetry import symmetric_forward_2d

# Shared encoder-type sets used by both build_jepa.py and train.py.
CDD_CUBE_ENCODER_TYPES = frozenset({
    "cdd_scaleaware_convnext",
    "convnext_dense_pyramid",
    "escnn_c4_pyramid",
})

CDD_DEBUG_ENCODER_TYPES = frozenset(CDD_CUBE_ENCODER_TYPES | {
    "convnext_dense_masktoken",
})


class PyramidGridJEPA(nn.Module):
    def __init__(
        self,
        latent_channels: int = 32,
        predictor_hidden: int = None,
        patch_size: int = 3,
        sigmas=(2, 4, 8, 16),
        mask_fraction: float = 1.0,
        mask_scale: float = 1.0,
        mask_scale_range=None,
        spacing_scale: float = 1.5,
        global_shift: bool = True,
        align_scales: bool = True,
        mask_box_size: int = 16,
        mask_box_size_range=None,
        random_mask_box_per_target: bool = False,
        manual_mask_box_sizes=None,
        cdd_mode: str = "log",
        cdd_constrained: bool = True,
        cdd_sm_mode: str = "reflect",
        cdd_append_last_residual: bool = True,
        post_log_transform: bool = True,
        log_eps: float = 1.0,
        cdd_log_std_floor_mult: float = 0.05,
        ema_momentum: float = 0.996,
        normalize_loss_l2: bool = False,
        predictor_layernorm: bool = True,
        predictor_spatial_conv: bool = False,
        projector_conv: bool = True,
        predictor_residual: bool = False,
        mode: str = "image",
        encoder_type: str = "convnext_dense_masktoken",
        encoder_width: int = 32,
        encoder_depth: int = 4,
        encoder_kernel_size: int = 7,
        convnext_layer_dilations=None,
        encoder_norm_type: Optional[str] = None,
        encoder_norm_groups: Optional[int] = None,
        encoder_norm_eps: Optional[float] = None,
        scaleaware_feat_channels: int = 8,
        scaleaware_adapter_kernel_size: int = 3,
        scaleaware_fusion_type: str = "concat",
        scaleaware_norm_per_scale: bool = False,
        scaleaware_adapter_norm: bool = True,
        scaleaware_final_norm: bool = True,
        scaleaware_stem_norm: bool = True,
        encoder_final_norm_type: str = "layernorm",
        encoder_head_bias: bool = True,
        target_invalid_region_skip: bool = True,
        target_invalid_region_values=(0.0, "nan"),
        target_sampling_mode: str = "random",
        priority_top_percent: float = 5.0,
        priority_n_target: int | str = 20,
        priority_min_targets_per_map: int = 0,
        priority_dithering_pixels: int = 6,
        priority_candidate_oversample: float = 3.0,
        use_symmetric_feature_loss: bool = False,
        target_nonoverlap: bool = True,
        target_allow_partial_overlap: float = 0.0,
        mask_box_hardcap: int | None = None,
        use_grn: bool = True,
    ):
        super().__init__()

        p = int(patch_size)
        if p <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size!r}.")
        if p % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {patch_size!r}.")
        self.patch_size = p
        self.sigmas = tuple(sigmas)
        self.mask_fraction = float(mask_fraction)
        mask_scale_value, inline_mask_scale_range = self._split_float_param(mask_scale, 1.0, "mask_scale")
        if mask_scale_range is not None and inline_mask_scale_range is not None:
            raise ValueError("Specify either mask_scale as a range or mask_scale_range, not both.")
        self.mask_scale = mask_scale_value
        self.mask_scale_range = self._coerce_float_range(
            mask_scale_range if mask_scale_range is not None else inline_mask_scale_range,
            "mask_scale_range",
        )
        self.spacing_scale = float(spacing_scale)
        self.global_shift = bool(global_shift)
        self.align_scales = bool(align_scales)
        mask_box_size_value, inline_mask_box_size_range = self._split_int_param(
            mask_box_size,
            16,
            "mask_box_size",
        )
        if mask_box_size_range is not None and inline_mask_box_size_range is not None:
            raise ValueError("Specify either mask_box_size as a range or mask_box_size_range, not both.")
        self.mask_box_size = mask_box_size_value
        self.mask_box_size_range = self._coerce_int_range(
            mask_box_size_range if mask_box_size_range is not None else inline_mask_box_size_range,
            "mask_box_size_range",
        )
        self.random_mask_box_per_target = bool(random_mask_box_per_target)
        self.manual_mask_box_sizes = self._coerce_manual_mask_box_sizes(manual_mask_box_sizes)
        if self.manual_mask_box_sizes is not None:
            if len(self.manual_mask_box_sizes) < len(self.sigmas):
                print(
                    "[warning] manual_mask_box_sizes shorter than sigmas/CDD channels; "
                    f"reusing last size for remaining channels: {self.manual_mask_box_sizes}"
                )
            elif len(self.manual_mask_box_sizes) > len(self.sigmas):
                print(
                    "[warning] manual_mask_box_sizes longer than sigmas/CDD channels; "
                    f"extra sizes will be ignored: {self.manual_mask_box_sizes}"
                )
        self.cdd_mode = str(cdd_mode)
        self.cdd_constrained = bool(cdd_constrained)
        self.cdd_sm_mode = str(cdd_sm_mode)
        self.cdd_append_last_residual = bool(cdd_append_last_residual)
        self.post_log_transform = bool(post_log_transform)
        self.log_eps = float(log_eps)
        self.cdd_log_std_floor_mult = float(cdd_log_std_floor_mult)
        self.ema_momentum = float(ema_momentum)
        self.normalize_loss_l2 = bool(normalize_loss_l2)
        self.predictor_layernorm = bool(predictor_layernorm)
        self.predictor_spatial_conv = bool(predictor_spatial_conv)
        self.predictor_residual = bool(predictor_residual)
        self.mode = str(mode)
        self.encoder_type = str(encoder_type)
        self.encoder_width = int(encoder_width)
        self.encoder_depth = int(encoder_depth)
        self.encoder_kernel_size = int(encoder_kernel_size)
        self.convnext_layer_dilations = (
            None if convnext_layer_dilations is None else tuple(int(d) for d in convnext_layer_dilations)
        )
        self.encoder_norm_type = None if encoder_norm_type is None else str(encoder_norm_type).lower()
        self.encoder_norm_groups = None if encoder_norm_groups is None else int(encoder_norm_groups)
        self.encoder_norm_eps = None if encoder_norm_eps is None else float(encoder_norm_eps)
        self.scaleaware_feat_channels = int(scaleaware_feat_channels)
        self.scaleaware_adapter_kernel_size = int(scaleaware_adapter_kernel_size)
        self.scaleaware_fusion_type = str(scaleaware_fusion_type)
        self.scaleaware_norm_per_scale = bool(scaleaware_norm_per_scale)
        self.scaleaware_adapter_norm = bool(scaleaware_adapter_norm)
        self.scaleaware_final_norm = bool(scaleaware_final_norm)
        self.scaleaware_stem_norm = bool(scaleaware_stem_norm)
        self.encoder_final_norm_type = str(encoder_final_norm_type).lower()
        self.encoder_head_bias = bool(encoder_head_bias)
        self.use_grn = bool(use_grn)
        self.target_invalid_region_skip = bool(target_invalid_region_skip)
        if target_invalid_region_values is None:
            self.target_invalid_region_values = (0.0, "nan")
        else:
            self.target_invalid_region_values = tuple(target_invalid_region_values)
        self.target_sampling_mode = normalize_target_sampling_mode(str(target_sampling_mode))
        self.priority_top_percent = float(priority_top_percent)
        # Keep raw value to support non-numeric modes such as "auto".
        self.priority_n_target = priority_n_target
        self.priority_min_targets_per_map = int(priority_min_targets_per_map)
        self.priority_dithering_pixels = int(priority_dithering_pixels)
        self.priority_candidate_oversample = float(priority_candidate_oversample)
        self.use_symmetric_feature_loss = bool(use_symmetric_feature_loss)
        self.target_nonoverlap = bool(target_nonoverlap)
        self.target_allow_partial_overlap = float(target_allow_partial_overlap)
        self.mask_box_hardcap = None if mask_box_hardcap is None else int(mask_box_hardcap)
        self.projector_conv = bool(projector_conv)
        if self.mode not in ("image", "pyramid"):
            raise ValueError(f"Unknown mode={self.mode}; expected 'image' or 'pyramid'")
        if self.encoder_type == "convnext_dense_masktoken":
            if self.mode != "image":
                raise ValueError(f"{self.encoder_type} requires mode='image'.")
        if self.encoder_type == "cdd_scaleaware_convnext":
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
                final_norm=self.scaleaware_final_norm,
                final_norm_type=self.encoder_final_norm_type,
                head_bias=self.encoder_head_bias,
                cdd_append_last_residual=self.cdd_append_last_residual,
                adapter_norm=self.scaleaware_adapter_norm,
                use_grn=self.use_grn,
                stem_norm=self.scaleaware_stem_norm,
                dilations=self.convnext_layer_dilations,
            )
        elif self.encoder_type == "convnext_dense_pyramid":
            if self.mode != "pyramid":
                raise ValueError("convnext_dense_pyramid requires mode='pyramid'.")
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
                use_grn=self.use_grn,
                dilations=self.convnext_layer_dilations,
            )
        elif self.encoder_type == "escnn_c4_pyramid":
            if self.mode != "pyramid":
                raise ValueError(f"{self.encoder_type} requires mode='pyramid'.")
            pyr_in_channels = 2 * max(1, len(self.sigmas))
            self.context_encoder = EscnnC4PyramidEncoder(
                in_channels=pyr_in_channels,
                hidden_channels=self.encoder_width,
                latent_channels=latent_channels,
                depth=self.encoder_depth,
                kernel_size=self.encoder_kernel_size,
                final_norm=self.scaleaware_final_norm,
                final_norm_type=self.encoder_final_norm_type,
            )
        elif self.encoder_type == "convnext_dense_masktoken":
            # 2D ConvNeXt image mode with explicit hard-mask token channel.
            self.context_encoder = ConvNeXtDenseEncoder(
                in_channels=2,
                hidden_channels=self.encoder_width,
                latent_channels=latent_channels,
                depth=self.encoder_depth,
                kernel_size=self.encoder_kernel_size,
                expansion=4,
                use_reflect_padding=True,
                final_norm=True,
                use_grn=self.use_grn,
                dilations=self.convnext_layer_dilations,
            )
        else:
            raise ValueError(f"Unknown encoder_type={self.encoder_type}")

        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        if predictor_hidden is None:
            predictor_hidden = latent_channels * 2
        if self.projector_conv:
            self.projector = nn.Sequential(
                nn.Conv2d(latent_channels, int(predictor_hidden), kernel_size=1),
                LayerNorm2d(int(predictor_hidden)) if self.predictor_layernorm else nn.Identity(),
                nn.GELU(),
                nn.Conv2d(int(predictor_hidden), latent_channels, kernel_size=1),
            )
        else:
            self.projector = nn.Identity()
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False
        # For D4 encoders, keep predictor point-wise to avoid reintroducing
        # post-encoder directional spatial derivatives.
        pred_ks = 1 if "_d4" in self.encoder_type else 3
        self.predictor = FullResPredictor(
            channels=latent_channels,
            hidden=int(predictor_hidden),
            use_layernorm=self.predictor_layernorm,
            spatial_conv=self.predictor_spatial_conv,
            residual=self.predictor_residual,
            kernel_size=pred_ks,
        )

    @staticmethod
    def _coerce_float_range(value, name: str):
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must be a list/tuple of exactly two values, got {value!r}")
        lo, hi = sorted((float(value[0]), float(value[1])))
        return lo, hi

    @classmethod
    def _split_float_param(cls, value, default: float, name: str):
        if value is None:
            return float(default), None
        if isinstance(value, (list, tuple)):
            lo, hi = cls._coerce_float_range(value, name)
            return float((lo + hi) / 2.0), (lo, hi)
        return float(value), None

    @staticmethod
    def _coerce_int_range(value, name: str):
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must be a list/tuple of exactly two values, got {value!r}")
        lo, hi = sorted((int(round(float(value[0]))), int(round(float(value[1])))))
        if lo < 1:
            raise ValueError(f"{name} must be >= 1, got {value!r}")
        return lo, hi

    @classmethod
    def _split_int_param(cls, value, default: int, name: str):
        if value is None:
            return int(default), None
        if isinstance(value, (list, tuple)):
            lo, hi = cls._coerce_int_range(value, name)
            return int(round((lo + hi) / 2.0)), (lo, hi)
        return int(round(float(value))), None

    def sample_mask_params(self, device=None) -> tuple[float, int]:
        """Return effective mask scale and box size for this masking call."""
        rand_device = device if device is not None else torch.device("cpu")
        mask_scale = self.mask_scale
        if self.mask_scale_range is not None:
            lo, hi = self.mask_scale_range
            if hi > lo:
                mask_scale = lo + (hi - lo) * float(torch.rand((), device=rand_device).item())
            else:
                mask_scale = lo

        mask_box_size = self.mask_box_size
        if self.mask_box_size_range is not None and not self.random_mask_box_per_target:
            lo, hi = self.mask_box_size_range
            if hi > lo:
                mask_box_size = int(torch.randint(lo, hi + 1, (), device=rand_device).item())
            else:
                mask_box_size = lo

        return float(mask_scale), int(mask_box_size)

    @staticmethod
    def _coerce_manual_mask_box_sizes(value) -> tuple[int, ...] | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            items = [v.strip() for v in stripped.split(",") if v.strip()]
        else:
            try:
                items = list(value)
            except TypeError:
                items = [value]
        if not items:
            return None
        sizes = tuple(int(round(float(v))) for v in items)
        if any(v <= 0 for v in sizes):
            raise ValueError(f"manual_mask_box_sizes must contain positive sizes, got {sizes}")
        return sizes

    def forward(
        self,
        x_clean,
        return_debug: bool = False,
        enable_grid_jitter: bool = True,
        enable_target_dithering: bool = True,
        lattice_shift_override=None,
        mask_inference: bool = True,
        context_data=None,
        cdd_orig: torch.Tensor | None = None,
    ):
        """
        x_clean: B x 1 x H x W

        context_data (optional): tuple of (x_context, target_locations,
            target_scales, target_valid [, debug]) pre-computed by
            prepare_context_batch / make_pyramid_grid_context.  When
            provided the masking step is skipped entirely — this must be
            called *outside* autocast in training loops.
        """
        if x_clean.dim() != 4:
            raise ValueError(f"Expected BxCxHxW, got {tuple(x_clean.shape)}")

        if x_clean.shape[1] != 1:
            raise ValueError(f"Expected grayscale input, got {x_clean.shape[1]} channels")

        if context_data is not None:
            x_context = context_data[0].to(device=x_clean.device)
            target_locations = context_data[1].to(device=x_clean.device)
            target_scales = context_data[2].to(device=x_clean.device)
            target_valid = context_data[3].to(device=x_clean.device)
            debug = context_data[4] if len(context_data) > 4 else {}
        else:
            invalid_pixel_mask = ~torch.isfinite(x_clean)
            if invalid_pixel_mask.any():
                x_clean = torch.nan_to_num(x_clean, nan=0.0, posinf=0.0, neginf=0.0)

            debug_encoder_types = CDD_DEBUG_ENCODER_TYPES
            need_debug_tensors = bool(
                return_debug
                or self.encoder_type in debug_encoder_types
            )
            effective_mask_scale, effective_mask_box_size = self.sample_mask_params(device=x_clean.device)
            if need_debug_tensors:
                x_context, target_locations, target_scales, target_valid, debug = make_pyramid_grid_context(
                    x_clean=x_clean,
                    sigmas=self.sigmas,
                    mask_fraction=self.mask_fraction,
                    mask_scale=effective_mask_scale,
                    spacing_scale=self.spacing_scale,
                    global_shift=self.global_shift,
                    align_scales=self.align_scales,
                    mask_box_size=effective_mask_box_size,
                    mask_box_size_range=self.mask_box_size_range,
                    random_mask_box_per_target=self.random_mask_box_per_target,
                    manual_mask_box_sizes=self.manual_mask_box_sizes,
                    cdd_mode=self.cdd_mode,
                    cdd_constrained=self.cdd_constrained,
                    cdd_sm_mode=self.cdd_sm_mode,
                    cdd_append_last_residual=self.cdd_append_last_residual,
                    inner_target_size=self.patch_size,
                    return_debug=True,
                    enable_grid_jitter=enable_grid_jitter,
                    enable_target_dithering=enable_target_dithering,
                    lattice_shift_override=lattice_shift_override,
                    target_invalid_region_skip=self.target_invalid_region_skip,
                    target_invalid_region_values=self.target_invalid_region_values,
                    invalid_pixel_mask=invalid_pixel_mask,
                    target_sampling_mode=self.target_sampling_mode,
                    priority_top_percent=self.priority_top_percent,
                    priority_n_target=self.priority_n_target,
                    priority_min_targets_per_map=self.priority_min_targets_per_map,
                    priority_dithering_pixels=self.priority_dithering_pixels,
                    priority_candidate_oversample=self.priority_candidate_oversample,
                    target_nonoverlap=self.target_nonoverlap,
                    target_allow_partial_overlap=self.target_allow_partial_overlap,
                    mask_box_hardcap=self.mask_box_hardcap,
                    cdd_orig_in=cdd_orig,
                )
            else:
                x_context, target_locations, target_scales, target_valid = make_pyramid_grid_context(
                    x_clean=x_clean,
                    sigmas=self.sigmas,
                    mask_fraction=self.mask_fraction,
                    mask_scale=effective_mask_scale,
                    spacing_scale=self.spacing_scale,
                    global_shift=self.global_shift,
                    align_scales=self.align_scales,
                    mask_box_size=effective_mask_box_size,
                    mask_box_size_range=self.mask_box_size_range,
                    random_mask_box_per_target=self.random_mask_box_per_target,
                    manual_mask_box_sizes=self.manual_mask_box_sizes,
                    cdd_mode=self.cdd_mode,
                    cdd_constrained=self.cdd_constrained,
                    cdd_sm_mode=self.cdd_sm_mode,
                    cdd_append_last_residual=self.cdd_append_last_residual,
                    inner_target_size=self.patch_size,
                    enable_grid_jitter=enable_grid_jitter,
                    enable_target_dithering=enable_target_dithering,
                    lattice_shift_override=lattice_shift_override,
                    target_invalid_region_skip=self.target_invalid_region_skip,
                    target_invalid_region_values=self.target_invalid_region_values,
                    invalid_pixel_mask=invalid_pixel_mask,
                    target_sampling_mode=self.target_sampling_mode,
                    priority_top_percent=self.priority_top_percent,
                    priority_n_target=self.priority_n_target,
                    priority_min_targets_per_map=self.priority_min_targets_per_map,
                    priority_dithering_pixels=self.priority_dithering_pixels,
                    priority_candidate_oversample=self.priority_candidate_oversample,
                    target_nonoverlap=self.target_nonoverlap,
                    target_allow_partial_overlap=self.target_allow_partial_overlap,
                    mask_box_hardcap=self.mask_box_hardcap,
                    cdd_orig_in=cdd_orig,
                )

        x_clean_enc = x_clean
        x_context_enc = x_context
        if self.post_log_transform:
            eps = max(1e-6, float(self.log_eps))
            # Shared floor keeps clean and masked CDD reconstructions on one scale.
            base = torch.clamp(x_clean, min=0.0)
            base_std = torch.std(base, dim=(-2, -1), keepdim=True)
            log_floor = torch.clamp(base_std * float(self.cdd_log_std_floor_mult), min=eps)
            x_clean_enc = torch.log(torch.clamp(x_clean, min=0.0) + log_floor)
            x_context_enc = torch.log(torch.clamp(x_context, min=0.0) + log_floor)

        # Optional multiscale CDD path: encode channel cubes directly.
        # Keep x_clean/x_context image outputs for backward-compatible diagnostics.
        enc_target = x_clean_enc
        enc_context = x_context_enc
        actual_context_in = None
        actual_target_in = None
        cdd_orig = None
        cdd_masked = None
        dip_per_ch = None
        cdd_orig_enc = None
        cdd_masked_enc = None
        needs_cdd_cube = self.encoder_type in CDD_CUBE_ENCODER_TYPES
        if needs_cdd_cube:
            cdd_orig = debug["cdd_channels_orig"].to(device=x_clean.device, dtype=x_clean.dtype)
            cdd_masked = debug["cdd_channels_masked"].to(device=x_clean.device, dtype=x_clean.dtype)
            dip_per_ch = debug["dip_field_per_channel"].to(device=x_clean.device, dtype=x_clean.dtype)
            # Global CDD-cube stabilization for pyramid encoders that consume
            # concatenated channel cubes directly (non-CDDOpNet paths).
            if self.post_log_transform:
                eps = max(1e-6, float(self.log_eps))
                base = torch.clamp(x_clean, min=0.0)
                base_std = torch.std(base, dim=(-2, -1), keepdim=True)
                log_floor = torch.clamp(base_std * float(self.cdd_log_std_floor_mult), min=eps)
                cdd_orig_enc = torch.log(torch.clamp(cdd_orig, min=0.0) + log_floor)
                cdd_masked_enc = torch.log(torch.clamp(cdd_masked, min=0.0) + log_floor)
            else:
                cdd_orig_enc = cdd_orig
                cdd_masked_enc = cdd_masked
            zero_token = torch.zeros_like(dip_per_ch)
            # target: original per-scale channels + zero token maps
            enc_target = torch.cat([cdd_orig_enc, zero_token], dim=1)
            # context: masked per-scale channels + mask token maps
            enc_context = torch.cat([cdd_masked_enc, dip_per_ch], dim=1)
        if not bool(mask_inference):
            # In mask-free inference, predictor branch should consume clean features.
            enc_context = enc_target
        symmetric_var = None  # trainable context-encoder rotation-view variance
        target_symmetric_var = None  # detached EMA diagnostic only
        if self.encoder_type == "cdd_scaleaware_convnext":
            if self.mode != "pyramid":
                raise ValueError("cdd_scaleaware_convnext requires mode='pyramid'.")
            mask_tokens = dip_per_ch
            cdd_orig_scaleaware = cdd_orig_enc
            cdd_masked_scaleaware = cdd_masked_enc
            if self.scaleaware_norm_per_scale:
                cdd_orig_scaleaware = norm_per_sample_channel(cdd_orig_scaleaware)
                cdd_masked_scaleaware = norm_per_sample_channel(cdd_masked_scaleaware)
            zero_mask_tokens = torch.zeros_like(mask_tokens)
            if bool(mask_inference):
                if self.use_symmetric_feature_loss:
                    context_map, ctx_var = symmetric_forward_2d(
                        self.context_encoder,
                        cdd_masked_scaleaware,
                        mask_tokens=mask_tokens,
                        return_var=True,
                    )
                    symmetric_var = ctx_var if symmetric_var is None else symmetric_var + ctx_var
                else:
                    context_map = self.context_encoder(cdd_masked_scaleaware, mask_tokens=mask_tokens)
            else:
                if self.use_symmetric_feature_loss:
                    context_map, ctx_var = symmetric_forward_2d(
                        self.context_encoder,
                        cdd_orig_scaleaware,
                        mask_tokens=zero_mask_tokens,
                        return_var=True,
                    )
                    symmetric_var = ctx_var if symmetric_var is None else symmetric_var + ctx_var
                else:
                    context_map = self.context_encoder(cdd_orig_scaleaware, mask_tokens=zero_mask_tokens)
            with torch.no_grad():
                if self.use_symmetric_feature_loss:
                    gt_map, gt_var = symmetric_forward_2d(
                        self.target_encoder,
                        cdd_orig_scaleaware,
                        mask_tokens=zero_mask_tokens,
                        return_var=True,
                    )
                    target_symmetric_var = gt_var if target_symmetric_var is None else target_symmetric_var + gt_var
                else:
                    gt_map = self.target_encoder(cdd_orig_scaleaware, mask_tokens=zero_mask_tokens)
        elif self.encoder_type in ("convnext_dense_pyramid", "escnn_c4_pyramid"):
            if self.mode != "pyramid":
                raise ValueError(f"{self.encoder_type} requires mode='pyramid'.")
            mask_tokens = dip_per_ch
            if bool(mask_inference):
                enc_context = torch.cat([cdd_masked_enc, mask_tokens], dim=1)
            else:
                enc_context = torch.cat([cdd_orig_enc, torch.zeros_like(mask_tokens)], dim=1)
            enc_target = torch.cat([cdd_orig_enc, torch.zeros_like(mask_tokens)], dim=1)
            with torch.no_grad():
                if self.use_symmetric_feature_loss:
                    gt_map, gt_var = symmetric_forward_2d(self.target_encoder, enc_target, return_var=True)
                    target_symmetric_var = gt_var if target_symmetric_var is None else target_symmetric_var + gt_var
                else:
                    gt_map = self.target_encoder(enc_target)
            if self.use_symmetric_feature_loss:
                context_map, ctx_var = symmetric_forward_2d(self.context_encoder, enc_context, return_var=True)
                symmetric_var = ctx_var if symmetric_var is None else symmetric_var + ctx_var
            else:
                context_map = self.context_encoder(enc_context)
        elif self.encoder_type == "convnext_dense_masktoken":
            if self.mode != "image":
                raise ValueError(f"{self.encoder_type} requires mode='image'.")
            if "mask_map" not in debug:
                raise RuntimeError(
                    f"{self.encoder_type} requires debug['mask_map']; "
                    "call make_pyramid_grid_context with return_debug=True."
                )
            mask_token = debug["mask_map"].to(device=x_clean_enc.device, dtype=x_clean_enc.dtype)
            if mask_token.ndim == 3:
                mask_token = mask_token.unsqueeze(1)
            if mask_token.ndim != 4:
                raise RuntimeError(f"Expected mask_map Bx1xHxW or BxHxW, got {tuple(mask_token.shape)}")
            if mask_token.shape[1] != 1:
                mask_token = mask_token[:, :1]
            mask_token = mask_token.clamp(0.0, 1.0)
            zero_token = torch.zeros_like(mask_token)

            # Fixed image ConvNeXt contract:
            # context  = [zero-filled masked image, binary mask map]
            # target   = [clean image, zero mask map]
            clean_image = x_clean_enc
            masked_image = clean_image * (1.0 - mask_token)
            if bool(mask_inference):
                context_in = torch.cat([masked_image, mask_token], dim=1)
            else:
                context_in = torch.cat([clean_image, zero_token], dim=1)
            target_in = torch.cat([clean_image, zero_token], dim=1)

            actual_context_in = context_in
            actual_target_in = target_in

            with torch.no_grad():
                if self.use_symmetric_feature_loss:
                    gt_map, gt_var = symmetric_forward_2d(self.target_encoder, target_in, return_var=True)
                    target_symmetric_var = gt_var if target_symmetric_var is None else target_symmetric_var + gt_var
                else:
                    gt_map = self.target_encoder(target_in)
            if self.use_symmetric_feature_loss:
                context_map, ctx_var = symmetric_forward_2d(self.context_encoder, context_in, return_var=True)
                symmetric_var = ctx_var if symmetric_var is None else symmetric_var + ctx_var
            else:
                context_map = self.context_encoder(context_in)
        else:
            with torch.no_grad():
                gt_map = self.target_encoder(enc_target)
            context_map = self.context_encoder(enc_context)
        context_base = context_map
        gt_base = gt_map
        context_proj = self.projector(context_base)
        pred_map = self.predictor(context_proj)
        with torch.no_grad():
            gt_map = self.target_projector(gt_base)

        pred_patches = extract_location_patches(pred_map, target_locations, patch_size=self.patch_size)
        gt_patches = extract_location_patches(gt_map, target_locations, patch_size=self.patch_size)
        context_patches = extract_location_patches(context_proj, target_locations, patch_size=self.patch_size)

        out = {
            "pred_patches": pred_patches,
            "gt_patches": gt_patches,
            "context_patches": context_patches,
            # Raw pre-encoder tensors (for diagnostics/visualization).
            "x_clean_raw": x_clean,
            "x_context_raw": x_context,
            # Actual network inputs after shared post-mask transform.
            "x_clean": x_clean_enc,
            "x_context": x_context_enc,
            "target_locations": target_locations,
            "target_scales": target_scales,
            "target_valid": target_valid,
            "context_map": context_base,
            "pred_map": pred_map,
            "gt_map": gt_map,
        }
        if symmetric_var is not None:
            out["symmetric_var"] = symmetric_var
        if target_symmetric_var is not None:
            out["target_symmetric_var"] = target_symmetric_var
        if actual_context_in is not None:
            out["network_context_in"] = actual_context_in
            out["network_target_in"] = actual_target_in
        for key in ("mask_scale_factor", "mask_footprint_px", "cdd_box_sizes", "target_box_sizes", "random_mask_box_per_target"):
            if key in debug:
                out[key] = debug[key].to(device=x_clean.device, dtype=x_clean.dtype)
        if return_debug or needs_cdd_cube:
            # Exact applied hard mask footprint from make_pyramid_grid_context.
            if return_debug:
                out["target_mask_map"] = debug["mask_map"].unsqueeze(1).to(device=x_clean.device, dtype=x_clean.dtype)
                for k in (
                    "priority_good_candidates",
                    "priority_nonzero_mean",
                    "priority_prescreen_candidates",
                    "priority_auto_base_targets",
                    "priority_effective_targets",
                ):
                    if k in debug:
                        out[k] = debug[k].to(device=x_clean.device, dtype=x_clean.dtype)
            out["cdd_channels_orig"] = debug["cdd_channels_orig"].to(device=x_clean.device, dtype=x_clean.dtype)
            out["cdd_channels_masked"] = debug["cdd_channels_masked"].to(device=x_clean.device, dtype=x_clean.dtype)
            out["dip_field_per_channel"] = debug["dip_field_per_channel"].to(device=x_clean.device, dtype=x_clean.dtype)
        return out

    def compute_symmetric_loss(self, outputs):
        """Context-encoder view variance, averaged over spatial and channel dims."""
        var = outputs.get("symmetric_var")
        if var is None:
            return torch.tensor(0.0, device=outputs["pred_patches"].device)
        return var.mean()

    def compute_loss(self, outputs):
        # Keep reductions in fp32: patch sums can overflow under AMP.
        pred = outputs["pred_patches"].float()
        gt = outputs["gt_patches"].detach().float()

        valid = outputs["target_valid"]  # B x K (bool)

        if self.normalize_loss_l2:
            # Normalize the full patch vector so spatial contrast is preserved.
            b, k, c, p1, p2 = pred.shape
            pred = F.normalize(pred.reshape(b, k, -1), dim=2).reshape(b, k, c, p1, p2)
            gt = F.normalize(gt.reshape(b, k, -1), dim=2).reshape(b, k, c, p1, p2)
            outputs["pred_patches"] = pred
            outputs["gt_patches"] = gt
        loss_map = F.mse_loss(pred, gt, reduction="none")  # B x K x C x P x P
        w = valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(loss_map.dtype)
        if not bool(valid.any().item()):
            # No valid targets in this batch: return graph-connected zero loss.
            return loss_map.sum() * 0.0
        denom = torch.clamp(w.sum() * loss_map.shape[2] * loss_map.shape[3] * loss_map.shape[4], min=1.0)
        return (loss_map * w).sum() / denom

    @torch.no_grad()
    def update_target_encoder(self):
        # Use base_encoder directly when a D4 / other wrapper is present to avoid
        # coupling the EMA to wrapper parameters that may appear in the future.
        ctx_enc = getattr(self.context_encoder, "base_encoder", self.context_encoder)
        tgt_enc = getattr(self.target_encoder, "base_encoder", self.target_encoder)
        for p_context, p_target in zip(ctx_enc.parameters(), tgt_enc.parameters()):
            p_target.mul_(self.ema_momentum).add_(p_context.detach(), alpha=1.0 - self.ema_momentum)
        if self.projector_conv:
            for p_proj, p_target_proj in zip(self.projector.parameters(), self.target_projector.parameters()):
                p_target_proj.mul_(self.ema_momentum).add_(p_proj.detach(), alpha=1.0 - self.ema_momentum)
