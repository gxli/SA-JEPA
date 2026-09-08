from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
from src.losses import l2_normalize_patches
from src.utils.support import (
    cdd_support_border_from_model,
    constrain_hardcap_to_encoder_footprint,
    encoder_border_from_model,
    encoder_receptive_field_from_model,
    invalid_support_border_from_model,
    normalize_invalid_support_border_mode,
)

# Shared encoder-type sets used by both build_jepa.py and train.py.
CDD_CUBE_ENCODER_TYPES = frozenset({
    "cdd_scaleaware_convnext",
    "cdd_scaleaware_convnext3d",
    "convnext_dense_pyramid",
    "escnn_c4_pyramid",
})

CDD_DEBUG_ENCODER_TYPES = frozenset(CDD_CUBE_ENCODER_TYPES)
MASK_MAP_ENCODER_TYPES = frozenset({"convnext_dense_masktoken"})


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
        cdd_pre_log_transform: bool = False,
        cdd_gaussian_backend: str = "cuda",
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
        target_invalid_region_values=("nan",),
        target_sampling_mode: str = "random",
        priority_top_percent: float = 5.0,
        priority_n_target: int | str = 20,
        priority_min_targets_per_map: int = 0,
        priority_dithering_pixels: int = 6,
        priority_candidate_oversample: float = 3.0,
        use_symmetric_feature_loss: bool = False,
        target_nonoverlap: bool = True,
        target_allow_partial_overlap: float = 0.0,
        otf_masking: bool = True,
        mask_box_hardcap: int | None = None,
        nan_border_sigma_multiplier: float = 3.0,
        invalid_support_border_mode: str = "encoder_rf",
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
        self.cdd_pre_log_transform = bool(cdd_pre_log_transform)
        self.cdd_gaussian_backend = str(cdd_gaussian_backend)
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
            self.target_invalid_region_values = ("nan",)
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
        self.otf_masking = bool(otf_masking)
        self.requested_mask_box_hardcap = None if mask_box_hardcap is None else int(mask_box_hardcap)
        self.mask_box_hardcap = constrain_hardcap_to_encoder_footprint(
            self.requested_mask_box_hardcap,
            self.encoder_receptive_field(),
        )
        if (
            self.requested_mask_box_hardcap is not None
            and self.mask_box_hardcap is not None
            and int(self.mask_box_hardcap) < int(self.requested_mask_box_hardcap)
        ):
            print(
                "[support] mask_box_hardcap constrained to encoder footprint: "
                f"requested={self.requested_mask_box_hardcap} effective={self.mask_box_hardcap} "
                f"encoder_rf={self.encoder_receptive_field()} margin=20%"
            )
        self.nan_border_sigma_multiplier = float(nan_border_sigma_multiplier)
        self.invalid_support_border_mode = normalize_invalid_support_border_mode(invalid_support_border_mode)
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

    def encoder_receptive_field(self) -> int:
        return int(encoder_receptive_field_from_model(self))

    def invalid_support_border_px(self, mask_scale: float | None = None) -> int:
        """Pixel radius that must stay finite around a target/inference pixel."""
        return int(invalid_support_border_from_model(self, mask_scale=mask_scale))

    def encoder_receptive_field_border_px(self) -> int:
        """Pixel radius implied by the encoder spatial receptive field."""
        return int(encoder_border_from_model(self))

    def cdd_support_border_px(self, mask_scale: float | None = None) -> int:
        """Conservative CDD/mask support radius, separate from encoder border rejection."""
        return int(cdd_support_border_from_model(self, mask_scale=mask_scale))

    def _apply_encoder_border_invalid_mask(self, invalid_pixel_mask: torch.Tensor) -> torch.Tensor:
        if invalid_pixel_mask.dim() != 4:
            return invalid_pixel_mask
        _, _, h, w = invalid_pixel_mask.shape
        encoder_border = int(max(0, min(self.encoder_receptive_field() // 2, h // 2, w // 2)))
        nan_border = int(max(0, min(self.invalid_support_border_px(), h // 2, w // 2)))

        if encoder_border <= 0 and nan_border <= 0:
            return invalid_pixel_mask

        # Dilate native no-data first. The crop edge is rejected independently
        # by the encoder FOV and must not be dilated by the CDD support radius.
        out = invalid_pixel_mask.clone()
        if nan_border > 0 and out.any():
            k = 2 * nan_border + 1
            invalid_float = out.float()
            dilated = F.max_pool2d(invalid_float, kernel_size=k, stride=1, padding=nan_border)
            out = dilated > 0.0

        if encoder_border > 0:
            out[:, :, :encoder_border, :] = True
            out[:, :, h - encoder_border :, :] = True
            out[:, :, :, :encoder_border] = True
            out[:, :, :, w - encoder_border :] = True

        return out

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

    def _make_otf_mask_tokens(
        self,
        *,
        pass_index: int,
        pass_ids_cpu: torch.Tensor,
        target_locations_cpu: torch.Tensor,
        target_scales_cpu: torch.Tensor,
        target_valid_cpu: torch.Tensor,
        target_box_sizes_cpu: torch.Tensor,
        cdd_box_sizes_cpu: torch.Tensor | None,
        channels: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build one packed, per-scale mask without rejecting any target."""
        bsz, target_slots = target_valid_cpu.shape
        tokens = torch.zeros((bsz, channels, height, width), device=device, dtype=dtype)
        sampled_mode = self.target_sampling_mode in ("random", "priority", "priority_small_scale")
        mask_all_scales = bool(self.align_scales or sampled_mode)

        for bi in range(int(bsz)):
            for ki in range(int(target_slots)):
                if not bool(target_valid_cpu[bi, ki]) or int(pass_ids_cpu[bi, ki]) != int(pass_index):
                    continue
                cy = int(target_locations_cpu[bi, ki, 0])
                cx = int(target_locations_cpu[bi, ki, 1])
                target_box = max(self.patch_size, int(round(float(target_box_sizes_cpu[bi, ki]))))

                if channels == 1:
                    channel_boxes = ((0, target_box),)
                elif self.random_mask_box_per_target:
                    channel_boxes = tuple((ci, target_box) for ci in range(channels))
                elif mask_all_scales:
                    channel_boxes = tuple(
                        (
                            ci,
                            max(
                                self.patch_size,
                                int(round(float(cdd_box_sizes_cpu[bi, min(ci, cdd_box_sizes_cpu.shape[1] - 1)])))
                                if cdd_box_sizes_cpu is not None and cdd_box_sizes_cpu.numel() > 0
                                else target_box,
                            ),
                        )
                        for ci in range(channels)
                    )
                else:
                    scale = float(target_scales_cpu[bi, ki])
                    ci = min(range(channels), key=lambda idx: abs(float(self.sigmas[min(idx, len(self.sigmas) - 1)]) - scale))
                    channel_boxes = ((ci, target_box),)

                for ci, box in channel_boxes:
                    half_lo = int(box) // 2
                    half_hi = int(box) - half_lo
                    y0 = max(0, cy - half_lo)
                    y1 = min(height, cy + half_hi)
                    x0 = max(0, cx - half_lo)
                    x1 = min(width, cx + half_hi)
                    if y1 > y0 and x1 > x0:
                        tokens[bi, ci, y0:y1, x0:x1] = 1.0
        return tokens

    def _forward_packed_otf(
        self,
        *,
        x_clean_enc: torch.Tensor,
        cdd_orig: torch.Tensor | None,
        cdd_orig_enc: torch.Tensor | None,
        effective_invalid: torch.Tensor,
        log_floor: torch.Tensor | None,
        target_locations: torch.Tensor,
        target_scales: torch.Tensor,
        target_valid: torch.Tensor,
        debug: dict,
    ) -> dict:
        """Run sequential OTF context passes and assemble one result per target.

        Training checkpoints each pass so encoder activations are recomputed
        during backward instead of retaining every pass in GPU memory.
        """
        pass_ids = debug["otf_mask_pass_ids"].to(device=x_clean_enc.device, dtype=torch.long)
        if pass_ids.shape != target_valid.shape:
            raise RuntimeError(
                "otf_mask_pass_ids must match target_valid, "
                f"got {tuple(pass_ids.shape)} vs {tuple(target_valid.shape)}"
            )
        num_passes = max(1, int(pass_ids.max().item()) + 1)
        target_box_sizes = debug["target_box_sizes"].to(device=x_clean_enc.device, dtype=x_clean_enc.dtype)
        cdd_box_sizes = debug.get("cdd_box_sizes")

        # Mask construction is discrete. Copy its compact metadata once instead
        # of synchronizing the accelerator for every target and every pass.
        pass_ids_cpu = pass_ids.detach().cpu()
        target_locations_cpu = target_locations.detach().cpu()
        target_scales_cpu = target_scales.detach().cpu()
        target_valid_cpu = target_valid.detach().cpu()
        target_box_sizes_cpu = target_box_sizes.detach().cpu()
        cdd_box_sizes_cpu = None if cdd_box_sizes is None else cdd_box_sizes.detach().cpu()

        actual_target_in = None
        target_symmetric_var = None
        if self.encoder_type == "cdd_scaleaware_convnext":
            if cdd_orig_enc is None:
                raise RuntimeError("Packed OTF CDD masking requires clean CDD channels")
            target_fields = norm_per_sample_channel(cdd_orig_enc) if self.scaleaware_norm_per_scale else cdd_orig_enc
            zero_tokens = torch.zeros_like(target_fields)
            with torch.no_grad():
                if self.use_symmetric_feature_loss:
                    gt_base, target_symmetric_var = symmetric_forward_2d(
                        self.target_encoder,
                        target_fields,
                        mask_tokens=zero_tokens,
                        return_var=True,
                    )
                else:
                    gt_base = self.target_encoder(target_fields, mask_tokens=zero_tokens)
        elif self.encoder_type in ("convnext_dense_pyramid", "escnn_c4_pyramid"):
            if cdd_orig_enc is None:
                raise RuntimeError("Packed OTF pyramid masking requires clean CDD channels")
            zero_tokens = torch.zeros_like(cdd_orig_enc)
            target_input = torch.cat([cdd_orig_enc, zero_tokens], dim=1)
            with torch.no_grad():
                if self.use_symmetric_feature_loss:
                    gt_base, target_symmetric_var = symmetric_forward_2d(
                        self.target_encoder, target_input, return_var=True
                    )
                else:
                    gt_base = self.target_encoder(target_input)
        elif self.encoder_type == "convnext_dense_masktoken":
            zero_tokens = torch.zeros_like(x_clean_enc[:, :1])
            target_input = torch.cat([x_clean_enc, zero_tokens], dim=1)
            actual_target_in = target_input
            with torch.no_grad():
                if self.use_symmetric_feature_loss:
                    gt_base, target_symmetric_var = symmetric_forward_2d(
                        self.target_encoder, target_input, return_var=True
                    )
                else:
                    gt_base = self.target_encoder(target_input)
        else:
            raise RuntimeError(f"Packed OTF masking is unsupported for encoder_type={self.encoder_type}")

        with torch.no_grad():
            gt_map = self.target_projector(gt_base)
            gt_patches = extract_location_patches(gt_map, target_locations, patch_size=self.patch_size)

        pred_patches = None
        context_patches = None
        representative_context = None
        representative_pred = None
        actual_context_in = None
        symmetric_vars = []
        bsz, _, height, width = x_clean_enc.shape

        def _run_context(function, *args):
            if self.training and torch.is_grad_enabled():
                return checkpoint(function, *args, use_reentrant=False)
            return function(*args)

        for pass_index in range(num_passes):
            channels = int(cdd_orig.shape[1]) if cdd_orig is not None else 1
            mask_tokens = self._make_otf_mask_tokens(
                pass_index=pass_index,
                pass_ids_cpu=pass_ids_cpu,
                target_locations_cpu=target_locations_cpu,
                target_scales_cpu=target_scales_cpu,
                target_valid_cpu=target_valid_cpu,
                target_box_sizes_cpu=target_box_sizes_cpu,
                cdd_box_sizes_cpu=cdd_box_sizes_cpu,
                channels=channels,
                height=height,
                width=width,
                device=x_clean_enc.device,
                dtype=x_clean_enc.dtype,
            )

            if self.encoder_type == "cdd_scaleaware_convnext":
                assert cdd_orig is not None
                masked_raw = cdd_orig * (1.0 - mask_tokens)
                if self.post_log_transform:
                    assert log_floor is not None
                    context_fields = torch.log(torch.clamp(masked_raw, min=0.0) + log_floor)
                else:
                    context_fields = masked_raw
                if effective_invalid.any():
                    invalid_expanded = effective_invalid.expand(-1, context_fields.shape[1], -1, -1)
                    context_fields = context_fields.masked_fill(invalid_expanded, 0.0)
                    mask_tokens = mask_tokens.masked_fill(invalid_expanded, 1.0)
                if self.scaleaware_norm_per_scale:
                    context_fields = norm_per_sample_channel(context_fields)
                if self.use_symmetric_feature_loss:
                    context_base, context_var = _run_context(
                        lambda fields, tokens: symmetric_forward_2d(
                            self.context_encoder,
                            fields,
                            mask_tokens=tokens,
                            return_var=True,
                        ),
                        context_fields,
                        mask_tokens,
                    )
                    symmetric_vars.append(context_var)
                else:
                    context_base = _run_context(
                        lambda fields, tokens: self.context_encoder(fields, mask_tokens=tokens),
                        context_fields,
                        mask_tokens,
                    )
            elif self.encoder_type in ("convnext_dense_pyramid", "escnn_c4_pyramid"):
                assert cdd_orig is not None
                masked_raw = cdd_orig * (1.0 - mask_tokens)
                if self.post_log_transform:
                    assert log_floor is not None
                    context_fields = torch.log(torch.clamp(masked_raw, min=0.0) + log_floor)
                else:
                    context_fields = masked_raw
                if effective_invalid.any():
                    invalid_expanded = effective_invalid.expand(-1, context_fields.shape[1], -1, -1)
                    context_fields = context_fields.masked_fill(invalid_expanded, 0.0)
                    mask_tokens = mask_tokens.masked_fill(invalid_expanded, 1.0)
                context_input = torch.cat([context_fields, mask_tokens], dim=1)
                if self.use_symmetric_feature_loss:
                    context_base, context_var = _run_context(
                        lambda fields: symmetric_forward_2d(
                            self.context_encoder, fields, return_var=True
                        ),
                        context_input,
                    )
                    symmetric_vars.append(context_var)
                else:
                    context_base = _run_context(self.context_encoder, context_input)
            else:
                masked_image = x_clean_enc * (1.0 - mask_tokens)
                context_input = torch.cat([masked_image, mask_tokens], dim=1)
                if actual_context_in is None:
                    actual_context_in = context_input
                if self.use_symmetric_feature_loss:
                    context_base, context_var = _run_context(
                        lambda fields: symmetric_forward_2d(
                            self.context_encoder, fields, return_var=True
                        ),
                        context_input,
                    )
                    symmetric_vars.append(context_var)
                else:
                    context_base = _run_context(self.context_encoder, context_input)

            context_proj = _run_context(self.projector, context_base)
            pred_map = _run_context(self.predictor, context_proj)
            pred_for_pass = extract_location_patches(pred_map, target_locations, patch_size=self.patch_size)
            context_for_pass = extract_location_patches(context_proj, target_locations, patch_size=self.patch_size)
            selector = ((pass_ids == pass_index) & target_valid).view(
                target_valid.shape[0], target_valid.shape[1], 1, 1, 1
            )
            selected_pred = torch.where(selector, pred_for_pass, torch.zeros_like(pred_for_pass))
            selected_context = torch.where(selector, context_for_pass, torch.zeros_like(context_for_pass))
            pred_patches = selected_pred if pred_patches is None else pred_patches + selected_pred
            context_patches = selected_context if context_patches is None else context_patches + selected_context
            if representative_context is None:
                representative_context = context_base
                representative_pred = pred_map

        assert pred_patches is not None and context_patches is not None
        assert representative_context is not None and representative_pred is not None
        symmetric_var = torch.stack(symmetric_vars, dim=0).mean(dim=0) if symmetric_vars else None
        return {
            "context_map": representative_context,
            "pred_map": representative_pred,
            "gt_map": gt_map,
            "pred_patches": pred_patches,
            "gt_patches": gt_patches,
            "context_patches": context_patches,
            "symmetric_var": symmetric_var,
            "target_symmetric_var": target_symmetric_var,
            "actual_context_in": actual_context_in,
            "actual_target_in": actual_target_in,
            "num_passes": num_passes,
            "pass_ids": pass_ids,
        }

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

        # Compute NaN/invalid mask from the raw image BEFORE nan_to_num.
        # NaN regions serve as a natural mask — targets + encoder input
        # must both be rejected from these regions and their dilated surroundings.
        invalid_pixel_mask = ~torch.isfinite(x_clean)
        # Replace NaN with 0 in x_clean so downstream ops (std, log, etc.)
        # don't propagate NaN.  The invalid_pixel_mask preserves the original
        # NaN locations so the encoder can still reject those regions.
        if invalid_pixel_mask.any():
            x_clean = torch.nan_to_num(x_clean, nan=0.0, posinf=0.0, neginf=0.0)
        invalid_pixel_mask = self._apply_encoder_border_invalid_mask(invalid_pixel_mask)

        if context_data is not None:
            x_context = context_data[0].to(device=x_clean.device)
            target_locations = context_data[1].to(device=x_clean.device)
            target_scales = context_data[2].to(device=x_clean.device)
            target_valid = context_data[3].to(device=x_clean.device)
            debug = context_data[4] if len(context_data) > 4 else {}
        else:

            debug_encoder_types = CDD_DEBUG_ENCODER_TYPES | MASK_MAP_ENCODER_TYPES
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
                    cdd_pre_log_transform=self.cdd_pre_log_transform,
                    cdd_gaussian_backend=self.cdd_gaussian_backend,
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
                    use_cdd=self.encoder_type in CDD_CUBE_ENCODER_TYPES,
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
                    cdd_pre_log_transform=self.cdd_pre_log_transform,
                    cdd_gaussian_backend=self.cdd_gaussian_backend,
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
        log_floor = None
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
        effective_invalid = invalid_pixel_mask
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
            # Apply NaN border mask to CDD features + mask tokens.
            # invalid_pixel_mask computed from x_clean may be all-False when
            # context_data is pre-computed (training path).  In that case the
            # merged mask is stored in the debug dict by prepare_context_batch.
            if not effective_invalid.any() and "_invalid_pixel_mask" in debug:
                effective_invalid = debug["_invalid_pixel_mask"].to(
                    device=x_clean.device, dtype=torch.bool,
                )
            if effective_invalid.any():
                s = cdd_orig_enc.shape[1]
                inv_expanded = effective_invalid.expand(-1, s, -1, -1)
                # Zero out CDD features at invalid regions so the encoder
                # sees neutral input instead of fake cold (NaN→0) pixels.
                cdd_orig_enc = cdd_orig_enc.masked_fill(inv_expanded, 0.0)
                cdd_masked_enc = cdd_masked_enc.masked_fill(inv_expanded, 0.0)
                # Set mask tokens at invalid regions so the per-scale adapter
                # learns to ignore those positions entirely.
                dip_per_ch = dip_per_ch.masked_fill(inv_expanded, 1.0)
                zero_token = zero_token.masked_fill(inv_expanded, 1.0)
            # target: original per-scale channels + zero token maps
            enc_target = torch.cat([cdd_orig_enc, zero_token], dim=1)
            # context: masked per-scale channels + mask token maps
            enc_context = torch.cat([cdd_masked_enc, dip_per_ch], dim=1)
        if not bool(mask_inference):
            # In mask-free inference, predictor branch should consume clean features.
            enc_context = enc_target
        symmetric_var = None  # trainable context-encoder symmetry-view variance
        target_symmetric_var = None  # detached EMA diagnostic only
        packed_otf = None
        use_packed_otf = bool(
            self.otf_masking
            and mask_inference
            and isinstance(debug, dict)
            and debug.get("otf_masking_enabled", False)
            and "otf_mask_pass_ids" in debug
            and "target_box_sizes" in debug
        )
        if use_packed_otf:
            packed_otf = self._forward_packed_otf(
                x_clean_enc=x_clean_enc,
                cdd_orig=cdd_orig,
                cdd_orig_enc=cdd_orig_enc,
                effective_invalid=effective_invalid,
                log_floor=log_floor,
                target_locations=target_locations,
                target_scales=target_scales,
                target_valid=target_valid,
                debug=debug,
            )
            context_map = packed_otf["context_map"]
            pred_map = packed_otf["pred_map"]
            gt_map = packed_otf["gt_map"]
            pred_patches = packed_otf["pred_patches"]
            gt_patches = packed_otf["gt_patches"]
            context_patches = packed_otf["context_patches"]
            symmetric_var = packed_otf["symmetric_var"]
            target_symmetric_var = packed_otf["target_symmetric_var"]
            actual_context_in = packed_otf["actual_context_in"]
            actual_target_in = packed_otf["actual_target_in"]
        elif self.encoder_type == "cdd_scaleaware_convnext":
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
        if packed_otf is None:
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
        if packed_otf is not None:
            out["otf_masking_num_passes"] = torch.tensor(
                float(packed_otf["num_passes"]), device=x_clean.device, dtype=x_clean.dtype
            )
            out["otf_mask_pass_ids"] = packed_otf["pass_ids"]
        for key in ("mask_scale_factor", "mask_footprint_px", "cdd_box_sizes", "target_box_sizes", "random_mask_box_per_target"):
            if key in debug:
                out[key] = debug[key].to(device=x_clean.device, dtype=x_clean.dtype)
        if return_debug:
            # Exact applied hard mask footprint from make_pyramid_grid_context.
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
        if needs_cdd_cube:
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
            pred = l2_normalize_patches(pred)
            gt = l2_normalize_patches(gt)
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
