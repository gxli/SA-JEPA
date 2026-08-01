from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SupportGeometry:
    """Shared pixel-support geometry for JEPA masking/inference/dashboard paths."""

    encoder_rf_px: int
    encoder_border_px: int
    cdd_support_border_px: int
    invalid_support_border_px: int
    hardcap_px: int | None
    requested_hardcap_px: int | None


DEFAULT_HARDCAP_FOOTPRINT_MARGIN_FRACTION = 0.20


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_dilations(dilations: Iterable[Any] | None, depth: int) -> list[int]:
    depth = max(0, int(depth))
    if dilations is None:
        return [1] * depth
    try:
        values = [max(1, int(v)) for v in dilations]
    except TypeError:
        values = [1] * depth
    if not values:
        return [1] * depth
    if len(values) < depth:
        reps = (depth + len(values) - 1) // len(values)
        values = (values * reps)[:depth]
    else:
        values = values[:depth]
    return values


def convnext_encoder_receptive_field_px(
    *,
    depth: int,
    kernel_size: int,
    dilations: Iterable[Any] | None = None,
) -> int:
    """Dense ConvNeXt encoder receptive field in input pixels.

    The local dense encoders use a 3x3 stem plus a 3x3 projection before the
    ConvNeXt blocks, hence the fixed ``1 + 2 + 2`` base footprint.
    """

    k = max(1, int(kernel_size))
    rf = 1 + 2 + 2
    for dilation in normalize_dilations(dilations, int(depth)):
        rf += max(0, k - 1) * max(1, int(dilation))
    return max(1, int(rf))


def encoder_border_from_rf(rf_px: int) -> int:
    return int(max(0, int(rf_px) // 2))


def max_hardcap_for_encoder_footprint(
    encoder_rf_px: int,
    *,
    margin_fraction: float = DEFAULT_HARDCAP_FOOTPRINT_MARGIN_FRACTION,
) -> int:
    """Largest mask hardcap that fits inside the encoder footprint.

    A 20% margin means the mask box may occupy at most 80% of the encoder
    receptive field, leaving context around the masked target.
    """

    rf = max(1, int(encoder_rf_px))
    margin = min(max(float(margin_fraction), 0.0), 0.95)
    return max(1, int(math.floor(float(rf) * (1.0 - margin))))


def constrain_hardcap_to_encoder_footprint(
    hardcap: Any,
    encoder_rf_px: int,
    *,
    margin_fraction: float = DEFAULT_HARDCAP_FOOTPRINT_MARGIN_FRACTION,
) -> int | None:
    """Clamp an optional mask_box_hardcap so it cannot exceed encoder support."""

    if hardcap is None:
        return None
    value = _as_int(hardcap, 0)
    if value <= 0:
        return None
    return int(min(value, max_hardcap_for_encoder_footprint(encoder_rf_px, margin_fraction=margin_fraction)))


def normalize_invalid_support_border_mode(mode: Any) -> str:
    value = str(mode or "cdd_support").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cdd": "cdd_support",
        "cdd_support_border": "cdd_support",
        "encoder_width": "encoder_width_half_plus_one",
        "width": "encoder_width_half_plus_one",
        "width_half": "encoder_width_half_plus_one",
        "width_half_plus_one": "encoder_width_half_plus_one",
        "half_encoder_width_plus_one": "encoder_width_half_plus_one",
    }
    return aliases.get(value, value)


def encoder_width_half_plus_one_border_px(encoder_width: Any) -> int:
    return int(max(0, _as_int(encoder_width, 32) // 2 + 1))


def encoder_receptive_field_from_model(model: Any) -> int:
    depth = _as_int(getattr(model, "encoder_depth", 4), 4)
    kernel = _as_int(getattr(model, "encoder_kernel_size", 7), 7)
    dilations = getattr(model, "convnext_layer_dilations", None)
    return convnext_encoder_receptive_field_px(depth=depth, kernel_size=kernel, dilations=dilations)


def encoder_border_from_model(model: Any) -> int:
    return encoder_border_from_rf(encoder_receptive_field_from_model(model))


def _model_cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    cfg = config.get("model", {})
    return cfg if isinstance(cfg, dict) else {}


def encoder_receptive_field_from_config(config: dict[str, Any] | None) -> int:
    cfg = _model_cfg(config)
    depth = _as_int(cfg.get("encoder_depth", 4), 4)
    kernel = _as_int(cfg.get("encoder_kernel_size", 7), 7)
    return convnext_encoder_receptive_field_px(
        depth=depth,
        kernel_size=kernel,
        dilations=cfg.get("convnext_layer_dilations"),
    )


def encoder_border_from_config(config: dict[str, Any] | None) -> int:
    return encoder_border_from_rf(encoder_receptive_field_from_config(config))


def cdd_support_border_px(
    *,
    sigmas: Iterable[Any] | None,
    mask_scale: Any = 1.0,
    support_multiplier: Any = 3.0,
    hardcap: Any = None,
    encoder_rf_px: Any = None,
    hardcap_margin_fraction: float = DEFAULT_HARDCAP_FOOTPRINT_MARGIN_FRACTION,
) -> int:
    values = [float(s) for s in (sigmas or [])]
    max_sigma = max(values) if values else 16.0
    mask_border = int(math.ceil(max_sigma * max(1.0, _as_float(mask_scale, 1.0))))
    support_border = int(math.ceil(max_sigma * max(1.0, _as_float(support_multiplier, 3.0))))
    if encoder_rf_px is not None:
        hardcap = constrain_hardcap_to_encoder_footprint(
            hardcap,
            _as_int(encoder_rf_px, 1),
            margin_fraction=hardcap_margin_fraction,
        )
    hardcap_border = int(max(0, _as_int(hardcap, 0) if hardcap is not None else 0))
    return int(max(mask_border, support_border, hardcap_border))


def cdd_support_border_from_model(model: Any, mask_scale: Any | None = None) -> int:
    rf = encoder_receptive_field_from_model(model)
    return cdd_support_border_px(
        sigmas=getattr(model, "sigmas", [2, 4, 8, 16]),
        mask_scale=getattr(model, "mask_scale", 1.0) if mask_scale is None else mask_scale,
        support_multiplier=getattr(model, "nan_border_sigma_multiplier", 3.0),
        hardcap=getattr(model, "mask_box_hardcap", None),
        encoder_rf_px=rf,
    )


def cdd_support_border_from_config(config: dict[str, Any] | None) -> int:
    cfg = _model_cfg(config)
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    data_cfg = data_cfg if isinstance(data_cfg, dict) else {}
    override_cfg = config.get("_inference_overrides", {}) if isinstance(config, dict) else {}
    override_cfg = override_cfg if isinstance(override_cfg, dict) else {}
    sigmas = override_cfg.get("sigmas", cfg.get("sigmas", [2, 4, 8, 16]))
    support_multiplier = cfg.get(
        "nan_border_sigma_multiplier",
        data_cfg.get("nan_border_sigma_multiplier", cfg.get("cdd_support_sigma_multiplier", 3.0)),
    )
    return cdd_support_border_px(
        sigmas=sigmas,
        mask_scale=cfg.get("mask_size_scaling", cfg.get("mask_scale", 1.0)),
        support_multiplier=support_multiplier,
        hardcap=cfg.get("mask_box_hardcap"),
        encoder_rf_px=encoder_receptive_field_from_config(config),
    )


def invalid_support_border_from_model(model: Any, mask_scale: Any | None = None) -> int:
    mode = normalize_invalid_support_border_mode(getattr(model, "invalid_support_border_mode", "cdd_support"))
    if mode == "encoder_width_half_plus_one":
        return encoder_width_half_plus_one_border_px(getattr(model, "encoder_width", 32))
    if mode == "encoder_rf":
        return encoder_border_from_model(model)
    return cdd_support_border_from_model(model, mask_scale=mask_scale)


def invalid_support_border_from_config(config: dict[str, Any] | None) -> int:
    cfg = _model_cfg(config)
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    data_cfg = data_cfg if isinstance(data_cfg, dict) else {}
    mode = normalize_invalid_support_border_mode(
        cfg.get("invalid_support_border_mode", data_cfg.get("invalid_support_border_mode", "cdd_support"))
    )
    if mode == "encoder_width_half_plus_one":
        return encoder_width_half_plus_one_border_px(cfg.get("encoder_width", cfg.get("latent_channels", 32)))
    if mode == "encoder_rf":
        return encoder_border_from_config(config)
    return cdd_support_border_from_config(config)


def support_geometry_from_model(model: Any) -> SupportGeometry:
    rf = encoder_receptive_field_from_model(model)
    requested_hardcap = getattr(model, "requested_mask_box_hardcap", getattr(model, "mask_box_hardcap", None))
    return SupportGeometry(
        encoder_rf_px=int(rf),
        encoder_border_px=encoder_border_from_rf(rf),
        cdd_support_border_px=cdd_support_border_from_model(model),
        invalid_support_border_px=invalid_support_border_from_model(model),
        hardcap_px=constrain_hardcap_to_encoder_footprint(getattr(model, "mask_box_hardcap", None), rf),
        requested_hardcap_px=None if requested_hardcap is None else _as_int(requested_hardcap, 0),
    )


def support_geometry_from_config(config: dict[str, Any] | None) -> SupportGeometry:
    rf = encoder_receptive_field_from_config(config)
    requested_hardcap = _model_cfg(config).get("mask_box_hardcap")
    return SupportGeometry(
        encoder_rf_px=int(rf),
        encoder_border_px=encoder_border_from_rf(rf),
        cdd_support_border_px=cdd_support_border_from_config(config),
        invalid_support_border_px=invalid_support_border_from_config(config),
        hardcap_px=constrain_hardcap_to_encoder_footprint(requested_hardcap, rf),
        requested_hardcap_px=None if requested_hardcap is None else _as_int(requested_hardcap, 0),
    )
