from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.dataset import JEPADataset
from src.diagnostics import (
    compute_effective_rank_from_features,
    compute_error_by_scale,
    rank_dashboard,
)
from src.inference import run_post_training_inference
from src.losses import (
    compute_jepa_energy,
    hard_negative_jepa_contrast_loss,
    compute_raw_mse_and_norm_err,
    compute_sim_var_cov,
    compute_sim_var_cov_torch,
    compute_target_energy_map,
    extract_valid_pooled_embeddings,
    sketched_sigreg_loss,
)
from src.models.build_jepa import PyramidGridJEPA
from src.models.masking import prepare_context_batch
from src.viz import save_inference_dashboard, save_loss_curve

def _fmt_metric(v: float) -> str:
    x = float(v)
    ax = abs(x)
    if ax == 0.0:
        return "0.0000"
    if ax < 1e-3 or ax >= 1e3:
        return f"{x:.3e}"
    return f"{x:.4f}"


def _collate_pad_hw(batch: list[torch.Tensor]) -> torch.Tensor:
    if len(batch) == 0:
        raise ValueError("Empty batch is not supported")
    max_h = max(int(x.shape[-2]) for x in batch)
    max_w = max(int(x.shape[-1]) for x in batch)
    out = []
    for x in batch:
        dh = max_h - int(x.shape[-2])
        dw = max_w - int(x.shape[-1])
        if dh > 0 or dw > 0:
            # Mark padded pixels as invalid so downstream target sampling can reject them.
            x = F.pad(x, (0, dw, 0, dh), mode="constant", value=float("nan"))
        out.append(x)
    return torch.stack(out, dim=0)


def _make_masking_collate(model: PyramidGridJEPA):
    """Return a collate function that runs masking in the DataLoader worker.

    Extracts scalar/lightweight masking params from *model* so the closure
    does not capture the full nn.Module (avoids expensive pickling).
    """
    params = {
        "sigmas": model.sigmas,
        "cell_sizes": model.cell_sizes,
        "mask_fraction": model.mask_fraction,
        "box_sigma_mult": model.box_sigma_mult,
        "mask_scale": model.mask_scale,
        "min_mask_scale": model.min_mask_scale,
        "spacing_scale": model.spacing_scale,
        "mask_size": model.mask_size,
        "full_grid": model.full_grid,
        "global_shift": model.global_shift,
        "align_scales": model.align_scales,
        "constant_mask_box": model.constant_mask_box,
        "mask_box_size": model.mask_box_size,
        "blur_mode": model.blur_mode,
        "cdd_mode": model.cdd_mode,
        "cdd_constrained": model.cdd_constrained,
        "cdd_sm_mode": model.cdd_sm_mode,
        "mask_fill_mode": model.mask_fill_mode,
        "dip_sigma_mult": model.dip_sigma_mult,
        "constant_gaussian_sigma": model.constant_gaussian_sigma,
        "scaleaware_gaussian_ratios": model.scaleaware_gaussian_ratios,
        "cdd_append_last_residual": model.cdd_append_last_residual,
        "patch_size": model.patch_size,
        "need_debug": (model.mode == "pyramid"),
        "target_invalid_region_skip": model.target_invalid_region_skip,
        "target_invalid_region_values": model.target_invalid_region_values,
        "target_sampling_mode": model.target_sampling_mode,
        "priority_top_percent": model.priority_top_percent,
        "priority_n_target": model.priority_n_target,
        "target_dithering_pixels": model.target_dithering_pixels,
    }

    def collate_fn(batch):
        x_clean = _collate_pad_hw(batch)
        result = prepare_context_batch(
            x_clean=x_clean,
            sigmas=params["sigmas"],
            cell_sizes=params["cell_sizes"],
            mask_fraction=params["mask_fraction"],
            box_sigma_mult=params["box_sigma_mult"],
            mask_scale=params["mask_scale"],
            min_mask_scale=params["min_mask_scale"],
            spacing_scale=params["spacing_scale"],
            mask_size=params["mask_size"],
            full_grid=params["full_grid"],
            global_shift=params["global_shift"],
            align_scales=params["align_scales"],
            constant_mask_box=params["constant_mask_box"],
            mask_box_size=params["mask_box_size"],
            blur_mode=params["blur_mode"],
            cdd_mode=params["cdd_mode"],
            cdd_constrained=params["cdd_constrained"],
            cdd_sm_mode=params["cdd_sm_mode"],
            mask_fill_mode=params["mask_fill_mode"],
            dip_sigma_mult=params["dip_sigma_mult"],
            constant_gaussian_sigma=params["constant_gaussian_sigma"],
            scaleaware_gaussian_ratios=params["scaleaware_gaussian_ratios"],
            cdd_append_last_residual=params["cdd_append_last_residual"],
            patch_size=params["patch_size"],
            return_debug=params["need_debug"],
            target_invalid_region_skip=params["target_invalid_region_skip"],
            target_invalid_region_values=params["target_invalid_region_values"],
            target_sampling_mode=params["target_sampling_mode"],
            priority_top_percent=params["priority_top_percent"],
            priority_n_target=params["priority_n_target"],
            target_dithering_pixels=params["target_dithering_pixels"],
        )
        if params["need_debug"]:
            x_context, tloc, tscale, tvalid, debug_tensors = result
            return x_clean, x_context, tloc, tscale, tvalid, debug_tensors
        else:
            x_context, tloc, tscale, tvalid = result
            return x_clean, x_context, tloc, tscale, tvalid, {}

    return collate_fn



@torch.no_grad()
def evaluate_validation(
    model: PyramidGridJEPA,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    vicreg_spatial_mode: str = "dense",
) -> dict:
    model.eval()
    n = 0
    loss_sum = 0.0
    sim_sum = 0.0
    scale_mse = defaultdict(list)
    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x_clean, x_context, tloc, tscale, tvalid, debug = batch
        x_clean = x_clean.to(device, non_blocking=True)
        x_context = x_context.to(device, non_blocking=True)
        tloc = tloc.to(device, non_blocking=True)
        tscale = tscale.to(device, non_blocking=True)
        tvalid = tvalid.to(device, non_blocking=True)
        if debug:
            debug = {k: v.to(device, non_blocking=True) for k, v in debug.items()}
        context_data = (x_context, tloc, tscale, tvalid, debug)
        outputs = model(x_clean, context_data=context_data)
        loss = model.compute_loss(outputs)
        sim_val, _, _ = compute_sim_var_cov(outputs, spatial_mode=vicreg_spatial_mode)
        ebs = compute_error_by_scale(outputs)
        for s, v in ebs.items():
            scale_mse[s].append(float(v))
        loss_sum += float(loss.item())
        sim_sum += float(sim_val)
        n += 1

    if n == 0:
        return {"val_loss": 0.0, "val_sim": 0.0, "val_error_by_scale": {}}
    return {
        "val_loss": loss_sum / n,
        "val_sim": sim_sum / n,
        "val_error_by_scale": {float(s): float(np.mean(v)) for s, v in scale_mse.items()},
    }


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_session_dir(root: str, config_name: str) -> str:
    path = os.path.join(root, config_name)
    os.makedirs(path, exist_ok=True)
    return path


def resolve_pipeline_config(data_cfg: dict, model_cfg: dict) -> tuple[bool, bool, bool]:
    blur_mode = str(model_cfg.get("blur_mode", "gaussian"))
    if blur_mode not in ("gaussian", "cdd"):
        raise ValueError(
            f"Unsupported blur_mode={blur_mode}. "
            "Allowed blur_mode values are 'gaussian' and 'cdd'."
        )
    # Policy:
    # - gaussian mode: dataset runs CDD, no pre-log, model may apply post-log.
    # - cdd mode: dataset skips CDD and pre-log, model performs CDD masking.
    dataset_apply_cdd = (blur_mode == "gaussian")
    dataset_log_transform = False
    model_post_log = bool(model_cfg.get("post_log_transform", data_cfg.get("log_transform", True)))
    return dataset_apply_cdd, dataset_log_transform, model_post_log


def resolve_encoder_type_default(model_cfg: dict) -> str:
    """
    Keep old defaults for image mode, but switch pyramid-mode default
    to the new isolated CDD encoder when encoder_type is not specified.
    """
    if "encoder_type" in model_cfg:
        return str(model_cfg["encoder_type"])
    mode = str(model_cfg.get("mode", "image"))
    if mode == "pyramid":
        return "cdd_opnet"
    return "fullres"


def build_model_from_config(model_cfg: dict, data_cfg: dict, train_cfg: dict, device: torch.device) -> PyramidGridJEPA:
    """Construct a PyramidGridJEPA from config dicts, with backward-compatible param aliases."""
    blur_mode = str(model_cfg.get("blur_mode", "gaussian"))
    mask_scaling_box = float(model_cfg.get("mask_scaling_box", model_cfg.get("mask_scale", 1.0)))
    # gscale is deprecated: gaussian width is now controlled by mask_fraction only.
    mask_scaling_gaussian = 1.0
    mask_spacing_scaling = float(model_cfg.get("mask_spacing_scaling", model_cfg.get("spacing_scale", 1.5)))
    mask_size = float(model_cfg.get("mask_size", 0.0))
    _, _, model_post_log = resolve_pipeline_config(data_cfg=data_cfg, model_cfg=model_cfg)
    resolved_encoder_type = resolve_encoder_type_default(model_cfg)

    return PyramidGridJEPA(
        latent_channels=model_cfg.get("latent_channels", 32),
        predictor_hidden=model_cfg.get("predictor_hidden"),
        patch_size=model_cfg.get("patch_size", 2),
        sigmas=tuple(model_cfg.get("sigmas", [2, 4, 8, 16])),
        cell_sizes=tuple(model_cfg.get("cell_sizes", [16, 32, 64, 128])),
        mask_fraction=model_cfg.get("mask_fraction", 1.0),
        box_sigma_mult=model_cfg.get("box_sigma_mult", 4.0),
        mask_scale=mask_scaling_box,
        min_mask_scale=model_cfg.get("min_mask_scale", 0.0),
        spacing_scale=mask_spacing_scaling,
        mask_size=mask_size,
        full_grid=model_cfg.get("full_grid", True),
        global_shift=model_cfg.get("global_shift", True),
        align_scales=model_cfg.get("align_scales", True),
        constant_mask_box=model_cfg.get("constant_mask_box", True),
        mask_box_size=model_cfg.get("mask_box_size", 16),
        blur_mode=blur_mode,
        cdd_mode=model_cfg.get("cdd_mode", "log"),
        cdd_constrained=model_cfg.get("cdd_constrained", True),
        cdd_sm_mode=model_cfg.get("cdd_sm_mode", "reflect"),
        mask_fill_mode=model_cfg.get("mask_fill_mode", "zero"),
        dip_sigma_mult=mask_scaling_gaussian,
        constant_gaussian_sigma=model_cfg.get("constant_gaussian_sigma", 1.0),
        cdd_append_last_residual=bool(model_cfg.get("cdd_append_last_residual", True)),
        post_log_transform=model_cfg.get("post_log_transform", model_post_log),
        log_eps=model_cfg.get("log_eps", float(data_cfg.get("log_eps", 1.0))),
        cdd_log_std_floor_mult=model_cfg.get("cdd_log_std_floor_mult", 0.05),
        ema_momentum=model_cfg.get("ema_momentum", train_cfg.get("momentum", 0.996)),
        normalize_loss=model_cfg.get("normalize_loss", True),
        predictor_layernorm=model_cfg.get("predictor_layernorm", False),
        mode=model_cfg.get("mode", "image"),
        encoder_type=resolved_encoder_type,
        encoder_width=model_cfg.get("encoder_width", model_cfg.get("latent_channels", 32)),
        encoder_depth=model_cfg.get("encoder_depth", 4),
        encoder_kernel_size=model_cfg.get("encoder_kernel_size", 7),
        encoder_norm_type=model_cfg.get("encoder_norm_type"),
        encoder_norm_groups=model_cfg.get("encoder_norm_groups"),
        encoder_norm_eps=model_cfg.get("encoder_norm_eps"),
        scaleaware_feat_channels=int(model_cfg.get("scaleaware_feat_channels", 8)),
        scaleaware_adapter_kernel_size=int(model_cfg.get("scaleaware_adapter_kernel_size", 3)),
        scaleaware_fusion_type=str(model_cfg.get("scaleaware_fusion_type", "concat")),
        scaleaware_norm_per_scale=bool(model_cfg.get("scaleaware_norm_per_scale", False)),
        mfae_scales=tuple(model_cfg.get("mfae_scales", [1, 2, 4])),
        mfae_features=tuple(model_cfg.get("mfae_features", ["x", "gradmag", "abslap", "local_std"])),
        mfae_normalize_attributes=bool(model_cfg.get("mfae_normalize_attributes", False)),
        mfae_include_mask_tokens=bool(model_cfg.get("mfae_include_mask_tokens", True)),
        scaleaware_gaussian_ratios=tuple(model_cfg.get("scaleaware_gaussian_ratios", [0.25, 0.5, 1.0, 2.0])),
        opnet_dilation_mode=model_cfg.get("opnet_dilation_mode", "half_cdd_scale"),
        opnet_dilations=model_cfg.get("opnet_dilations"),
        opnet_max_dilation=int(model_cfg.get("opnet_max_dilation", 16)),
        opnet_channel_mode=model_cfg.get("opnet_channel_mode", "multi"),
        op_smoothing_mode=model_cfg.get("op_smoothing_mode", "sqrt_scale"),
        op_smoothing_mult=float(model_cfg.get("op_smoothing_mult", 1.0)),
        op_smoothing_padding_mode=model_cfg.get("op_smoothing_padding_mode", "reflect"),
        opnet_cache_primitives=bool(model_cfg.get("opnet_cache_primitives", True)),
        opnet_cache_detach=bool(model_cfg.get("opnet_cache_detach", True)),
        target_invalid_region_skip=bool(model_cfg.get("target_invalid_region_skip", False)),
        target_invalid_region_values=tuple(model_cfg.get("target_invalid_region_values", [0, "nan"])),
        target_sampling_mode=str(model_cfg.get("target_sampling_mode", "grid")),
        priority_top_percent=float(model_cfg.get("priority_top_percent", 5.0)),
        priority_n_target=int(model_cfg.get("priority_n_target", 20)),
        target_dithering_pixels=int(model_cfg.get("target_dithering_pixels", 6)),
    ).to(device)


def run_training(config: dict, config_name: str, sessions_root: str = "sessions") -> str:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    print(
        f"[{config_name}] backend_discovered device={device.type} "
        f"cuda_available={torch.cuda.is_available()} mps_available={mps_available}"
    )

    train_cfg = config["train"]
    model_cfg = config["model"]
    data_cfg = config["data"]

    session_dir = make_session_dir(sessions_root, config_name)
    os.makedirs(session_dir, exist_ok=True)
    model_ckpt_path = os.path.join(session_dir, "model_last.pt")
    resume_ckpt_path = os.path.join(session_dir, "checkpoint_last.pt")
    resume_from_existing = os.path.exists(model_ckpt_path)

    with open(os.path.join(session_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    dataset_apply_cdd, dataset_log_transform, _ = resolve_pipeline_config(data_cfg=data_cfg, model_cfg=model_cfg)

    print(
        f"[{config_name}] resolved_pipeline "
        f"dataset_apply_cdd={dataset_apply_cdd} "
        f"dataset_log_transform={dataset_log_transform} "
        f"model.post_log_transform={model_cfg.get('post_log_transform', '<unset>')} "
        f"data.log_transform={data_cfg.get('log_transform', True)} "
        f"data.cdd_mode={data_cfg.get('cdd_mode', 'log')} "
        f"model.cdd_mode={model_cfg.get('cdd_mode', 'log')}"
    )

    model = build_model_from_config(model_cfg, data_cfg, train_cfg, device)
    allow_partial_resume = bool(train_cfg.get("allow_partial_resume", False))
    resume_mismatch_action = str(train_cfg.get("resume_mismatch_action", "skip")).lower()
    if resume_mismatch_action not in ("skip", "error"):
        raise ValueError(
            f"Unsupported resume_mismatch_action={resume_mismatch_action}. "
            "Use 'skip' or 'error'."
        )
    optimizer_mismatch_action = str(train_cfg.get("optimizer_mismatch_action", "continue_fresh_optimizer")).lower()
    if optimizer_mismatch_action not in ("continue_fresh_optimizer", "restart_epoch0"):
        raise ValueError(
            f"Unsupported optimizer_mismatch_action={optimizer_mismatch_action}. "
            "Use 'continue_fresh_optimizer' or 'restart_epoch0'."
        )

    start_epoch = 0
    resume_state = None
    if os.path.exists(resume_ckpt_path):
        resume_state = torch.load(resume_ckpt_path, map_location=device)
        if "model_state_dict" in resume_state:
            try:
                missing, unexpected = model.load_state_dict(resume_state["model_state_dict"], strict=False)
            except RuntimeError as e:
                # Common during architecture evolution (e.g. channel-count changes).
                print(f"[{config_name}] resume_model load_state_dict_failed: {e}")
                missing, unexpected = ["__load_state_dict_failed__"], []
            print(f"[{config_name}] resume_model missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
            if missing:
                print(f"[{config_name}] resume_model missing_keys_list={missing}")
            if unexpected:
                print(f"[{config_name}] resume_model unexpected_keys_list={unexpected}")
            if (missing or unexpected) and not allow_partial_resume:
                if resume_mismatch_action == "error":
                    raise RuntimeError(
                        "Checkpoint model-state mismatch detected and allow_partial_resume=False. "
                        "Set train.allow_partial_resume=true to permit partial model resume."
                    )
                print(
                    f"[{config_name}] warning: checkpoint model-state mismatch; "
                    "skipping resume checkpoint and starting fresh model/optimizer/scaler."
                )
                resume_state = None
                start_epoch = 0
                model = build_model_from_config(model_cfg, data_cfg, train_cfg, device)
                print(f"[{config_name}] resume_checkpoint_ignored={resume_ckpt_path}")
        if resume_state is not None:
            start_epoch = int(resume_state.get("epoch", 0))
            print(f"resume_checkpoint={resume_ckpt_path} start_epoch={start_epoch}")
    elif resume_from_existing:
        try:
            missing, unexpected = model.load_state_dict(torch.load(model_ckpt_path, map_location=device), strict=False)
        except RuntimeError as e:
            # Common during architecture evolution (e.g. channel-count changes).
            print(f"[{config_name}] resume_model load_state_dict_failed: {e}")
            missing, unexpected = ["__load_state_dict_failed__"], []
        print(f"[{config_name}] resume_model missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        if missing:
            print(f"[{config_name}] resume_model missing_keys_list={missing}")
        if unexpected:
            print(f"[{config_name}] resume_model unexpected_keys_list={unexpected}")
        if (missing or unexpected) and not allow_partial_resume:
            if resume_mismatch_action == "error":
                raise RuntimeError(
                    "Model checkpoint mismatch detected and allow_partial_resume=False. "
                    "Set train.allow_partial_resume=true to permit partial model resume."
                )
            print(
                f"[{config_name}] warning: model checkpoint mismatch; "
                "ignoring model_last and starting fresh model/optimizer/scaler."
            )
            model = build_model_from_config(model_cfg, data_cfg, train_cfg, device)
            print(f"[{config_name}] resume_model_ignored={model_ckpt_path}")
        else:
            print(f"resume_model={model_ckpt_path}")

    scale_max = float(max(model_cfg.get("sigmas", [2, 4, 8, 16])))
    _msb = float(model_cfg.get("mask_scaling_box", model_cfg.get("mask_scale", 1.0)))
    _mss = float(model_cfg.get("mask_spacing_scaling", model_cfg.get("spacing_scale", 1.5)))
    auto_roll_max = max(1, int(round(scale_max * _msb * _mss)))

    dataset = JEPADataset(
        num_samples=data_cfg.get("num_samples", 2000),
        image_size=data_cfg.get("image_size", 256),
        data_root=data_cfg.get("data_root", "data"),
        npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
        log_transform=dataset_log_transform,
        log_eps=data_cfg.get("log_eps", 1.0),
        cdd_scales=data_cfg.get("cdd_scales", [2, 4, 8, 16]),
        cdd_strength=data_cfg.get("cdd_strength", 1.0),
        cdd_clip=data_cfg.get("cdd_clip", True),
        norm_before_cdd=data_cfg.get("norm_before_cdd", True),
        cdd_mode=data_cfg.get("cdd_mode", "log"),
        cdd_constrained=data_cfg.get("cdd_constrained", True),
        cdd_sm_mode=data_cfg.get("cdd_sm_mode", "reflect"),
        apply_cdd=dataset_apply_cdd,
        cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
        cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
        cube_slice_index=data_cfg.get("cube_slice_index", 0),
        random_roll_max=int(max(0, data_cfg.get("random_roll_max", auto_roll_max))),
        d4_augment=bool(data_cfg.get("d4_augment", False)),
        cache_cdd=bool(data_cfg.get("cache_cdd", True)),
        cdd_cache_dir=data_cfg.get("cdd_cache_dir"),
        cache_random_slices=bool(data_cfg.get("cache_random_slices", False)),
        precompute_cdd_cache_all_slices=bool(data_cfg.get("precompute_cdd_cache_all_slices", False)),
    )
    val_fraction = float(train_cfg.get("val_fraction", 0.1))
    val_fraction = min(max(val_fraction, 0.0), 0.95)
    total_idx = list(dataset.sample_index)
    n_total = len(total_idx)
    n_val_idx = int(round(n_total * val_fraction)) if n_total > 1 else 0
    if val_fraction > 0.0 and n_val_idx == 0 and n_total > 1:
        n_val_idx = 1
    n_train_idx = max(1, n_total - n_val_idx)
    train_idx = total_idx[:n_train_idx]
    val_idx = total_idx[n_train_idx:] if n_val_idx > 0 else []

    train_dataset = dataset
    train_dataset.sample_index = train_idx
    train_dataset.num_samples = int(train_cfg.get("num_samples", data_cfg.get("num_samples", 2000)))

    val_dataset = None
    if len(val_idx) > 0:
        val_dataset = JEPADataset(
            num_samples=max(1, int(train_cfg.get("val_num_samples", max(16, int(0.25 * train_dataset.num_samples))))),
            image_size=data_cfg.get("image_size", 256),
            data_root=data_cfg.get("data_root", "data"),
            npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
            log_transform=dataset_log_transform,
            log_eps=data_cfg.get("log_eps", 1.0),
            cdd_scales=data_cfg.get("cdd_scales", [2, 4, 8, 16]),
            cdd_strength=data_cfg.get("cdd_strength", 1.0),
            cdd_clip=data_cfg.get("cdd_clip", True),
            norm_before_cdd=data_cfg.get("norm_before_cdd", True),
            cdd_mode=data_cfg.get("cdd_mode", "log"),
            cdd_constrained=data_cfg.get("cdd_constrained", True),
            cdd_sm_mode=data_cfg.get("cdd_sm_mode", "reflect"),
            apply_cdd=dataset_apply_cdd,
            cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
            cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
            cube_slice_index=data_cfg.get("cube_slice_index", 0),
            random_roll_max=int(max(0, data_cfg.get("random_roll_max", auto_roll_max))),
            d4_augment=False,
            cache_cdd=bool(data_cfg.get("cache_cdd", True)),
            cdd_cache_dir=data_cfg.get("cdd_cache_dir"),
            cache_random_slices=bool(data_cfg.get("cache_random_slices", False)),
            precompute_cdd_cache_all_slices=bool(data_cfg.get("precompute_cdd_cache_all_slices", False)),
        )
        val_dataset.sample_index = val_idx
    print(
        f"[{config_name}] dataset_split total_index={n_total} train_index={len(train_idx)} "
        f"val_index={len(val_idx)} val_fraction={val_fraction:.3f}"
    )
    print(
        f"[{config_name}] data_jitter random_roll_max={dataset.random_roll_max} "
        f"(symmetric inclusive roll in [-max,+max])"
    )
    requested_workers = int(train_cfg.get("num_workers", 4))
    # macOS/MPS-safe default: avoid multiprocessing worker hangs unless explicitly set.
    if "num_workers" in train_cfg:
        num_workers = requested_workers
    else:
        num_workers = 4 if device.type == "cuda" else 0
    pin_memory = bool(device.type == "cuda")
    persistent_workers = bool(num_workers > 0)
    print(
        f"[{config_name}] dataloader_setup num_workers={num_workers} "
        f"pin_memory={pin_memory} persistent_workers={persistent_workers}"
    )

    masking_collate = _make_masking_collate(model)
    dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=masking_collate,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 32),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=masking_collate,
        )
    # Inference must use canonical orientation (no D4 augmentation).
    inference_dataset = JEPADataset(
        num_samples=train_dataset.num_samples,
        image_size=data_cfg.get("image_size", 256),
        data_root=data_cfg.get("data_root", "data"),
        npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
        log_transform=dataset_log_transform,
        log_eps=data_cfg.get("log_eps", 1.0),
        cdd_scales=data_cfg.get("cdd_scales", [2, 4, 8, 16]),
        cdd_strength=data_cfg.get("cdd_strength", 1.0),
        cdd_clip=data_cfg.get("cdd_clip", True),
        norm_before_cdd=data_cfg.get("norm_before_cdd", True),
        cdd_mode=data_cfg.get("cdd_mode", "log"),
        cdd_constrained=data_cfg.get("cdd_constrained", True),
        cdd_sm_mode=data_cfg.get("cdd_sm_mode", "reflect"),
        apply_cdd=dataset_apply_cdd,
        cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
        cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
        cube_slice_index=data_cfg.get("cube_slice_index", 0),
        random_roll_max=int(max(0, data_cfg.get("random_roll_max", auto_roll_max))),
        d4_augment=False,
        cache_cdd=bool(data_cfg.get("cache_cdd", True)),
        cdd_cache_dir=data_cfg.get("cdd_cache_dir"),
        cache_random_slices=bool(data_cfg.get("cache_random_slices", False)),
        precompute_cdd_cache_all_slices=bool(data_cfg.get("precompute_cdd_cache_all_slices", False)),
    )
    inference_dataset.sample_index = list(train_idx)
    inference_dataset.num_samples = train_dataset.num_samples
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=_collate_pad_hw,
    )

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    if resume_state is not None:
        optimizer_state_loaded = False
        if "optimizer_state_dict" in resume_state:
            try:
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
                optimizer_state_loaded = True
            except ValueError as e:
                # Model parameterization changed (e.g., architecture update): choose explicit behavior.
                if optimizer_mismatch_action == "restart_epoch0":
                    print(f"[{config_name}] warning: optimizer_state_incompatible, restarting epoch counter at 0: {e}")
                    start_epoch = 0
                else:
                    print(
                        f"[{config_name}] warning: optimizer_state_incompatible, "
                        f"continuing from epoch {start_epoch} with fresh optimizer: {e}"
                    )
        if optimizer_state_loaded and "scaler_state_dict" in resume_state and torch.cuda.is_available():
            try:
                scaler.load_state_dict(resume_state["scaler_state_dict"])
            except Exception as e:
                print(f"[{config_name}] warning: scaler_state_incompatible, starting scaler fresh: {e}")

    epochs = train_cfg.get("epochs", 20)
    log_interval = train_cfg.get("log_interval", 10)
    force_recompute_inference = bool(train_cfg.get("force_recompute_inference", False))
    inference_mask_passes = int(train_cfg.get("inference_mask_passes", 1))
    mask_inference = bool(train_cfg.get("mask_inference", False))
    viz_crop_border = bool(train_cfg.get("viz_crop_border", False))
    viz_crop_border_px = train_cfg.get("viz_crop_border_px")
    umap_cfg = dict(train_cfg.get("umap", {}))
    compute_effective_rank = bool(train_cfg.get("compute_effective_rank", False))
    inference_visit_batches = int(train_cfg.get("inference_visit_batches", 32))
    print(f"[{config_name}] umap_config={json.dumps(umap_cfg, sort_keys=True)}")
    jepa_loss_weight = float(train_cfg.get("jepa_loss_weight", 100.0))
    vicreg_var_weight = float(train_cfg.get("vicreg_var_weight", 1.0))
    vicreg_cov_weight = float(train_cfg.get("vicreg_cov_weight", 0.1))
    sigreg_weight = float(train_cfg.get("sigreg_weight", 0.0))
    sigreg_sketch_dim = int(train_cfg.get("sigreg_sketch_dim", 64))
    sharp_contrast_weight = float(train_cfg.get("sharp_contrast_weight", 0.0))
    sharp_contrast_temperature = float(train_cfg.get("sharp_contrast_temperature", 0.10))
    sharp_contrast_same_scale = bool(train_cfg.get("sharp_contrast_same_scale", True))
    sharp_contrast_same_sample = bool(train_cfg.get("sharp_contrast_same_sample", True))
    vicreg_spatial_mode = str(train_cfg.get("vicreg_spatial_mode", "dense")).lower()
    if vicreg_spatial_mode not in ("dense", "pooled"):
        raise ValueError(
            f"Unsupported train.vicreg_spatial_mode={vicreg_spatial_mode}. Use 'dense' or 'pooled'."
        )
    ema_base = float(train_cfg.get("ema_momentum_base", model.ema_momentum))
    ema_final = float(train_cfg.get("ema_momentum_final", 1.0))

    metrics_path = os.path.join(session_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "batch",
                    "global_step",
                    "total_loss",
                    "loss_jepa",
                    "loss_pixel",
                    "loss_sigreg",
                    "loss_sharp_contrast",
                    "loss_var",
                    "loss_cov",
                    "weighted_jepa",
                    "weighted_sigreg",
                    "weighted_sharp_contrast",
                    "weighted_var",
                    "weighted_cov",
                    "ema_momentum",
                    "sim",
                    "var",
                    "cov",
                    "raw_mse",
                    "norm_err",
                    "valid_frac",
                    "time_sec",
                ]
            )
    masked_scales_log_path = os.path.join(session_dir, "masked_scales_log.csv")
    if not os.path.exists(masked_scales_log_path):
        with open(masked_scales_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "batch", "scale", "count"])
    epoch_summary_path = os.path.join(session_dir, "epoch_summary.csv")
    if not os.path.exists(epoch_summary_path):
        with open(epoch_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_sim", "val_error_by_scale_json"])
    visited_targets_log_path = os.path.join(session_dir, "visited_target_locations.csv")
    if not os.path.exists(visited_targets_log_path):
        with open(visited_targets_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "batch", "sample_idx", "target_idx", "y", "x", "scale"])

    model.train()
    start = time.time()
    visit_counts = None
    if start_epoch >= int(epochs):
        print(f"[{config_name}] checkpoint epoch {start_epoch} already >= configured epochs {epochs}, skipping training loop")
    for epoch in range(start_epoch, epochs):
        epoch_total = 0.0
        epoch_jepa = 0.0
        epoch_pixel = 0.0
        epoch_sim = 0.0
        epoch_var = 0.0
        epoch_cov = 0.0
        epoch_sigreg = 0.0
        epoch_sharp = 0.0
        epoch_valid_frac = 0.0
        epoch_batches = 0
        metrics_rows = []
        masked_scale_rows = []
        visited_rows = []
        for batch_idx, batch in enumerate(dataloader):
            x_clean, x_context, tloc, tscale, tvalid, debug = batch
            x_clean = x_clean.to(device, non_blocking=True)
            x_context = x_context.to(device, non_blocking=True)
            tloc = tloc.to(device, non_blocking=True)
            tscale = tscale.to(device, non_blocking=True)
            tvalid = tvalid.to(device, non_blocking=True)
            if debug:
                debug = {k: v.to(device, non_blocking=True) for k, v in debug.items()}
            context_data = (x_context, tloc, tscale, tvalid, debug)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(x_clean, context_data=context_data)
                loss_jepa = model.compute_loss(outputs)
                _, var_term_t, cov_term_t = compute_sim_var_cov_torch(
                    outputs,
                    spatial_mode=vicreg_spatial_mode,
                )
                z_pred = extract_valid_pooled_embeddings(outputs, key="pred_patches")
                loss_sigreg = sketched_sigreg_loss(z_pred, sketch_dim=sigreg_sketch_dim)
                loss_sharp = hard_negative_jepa_contrast_loss(
                    outputs,
                    temperature=sharp_contrast_temperature,
                    same_scale=sharp_contrast_same_scale,
                    same_sample=sharp_contrast_same_sample,
                )
                total_loss = (
                    (jepa_loss_weight * loss_jepa)
                    + (sharp_contrast_weight * loss_sharp)
                    + (vicreg_var_weight * var_term_t)
                    + (vicreg_cov_weight * cov_term_t)
                    + (sigreg_weight * loss_sigreg)
                )
                loss_pixel_val = 0.0

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            current_step = epoch * max(1, len(dataloader)) + batch_idx
            total_steps = max(1, int(epochs) * max(1, len(dataloader)))
            progress = min(1.0, max(0.0, float(current_step) / float(total_steps)))
            # Cosine EMA schedule from base momentum toward final momentum.
            new_momentum = float(
                ema_final - 0.5 * (ema_final - ema_base) * (1.0 + math.cos(math.pi * progress))
            )
            model.ema_momentum = new_momentum
            model.update_target_encoder()
            sim_val, var_val, cov_val = compute_sim_var_cov(
                outputs,
                spatial_mode=vicreg_spatial_mode,
            )
            raw_mse_val, norm_err_val = compute_raw_mse_and_norm_err(outputs)
            valid_frac = float(outputs["target_valid"].float().mean().item())

            elapsed = time.time() - start
            metrics_rows.append(
                [
                    epoch + 1,
                    batch_idx,
                    epoch * max(1, len(dataloader)) + batch_idx,
                    float(total_loss.item()),
                    float(loss_jepa.item()),
                    float(loss_pixel_val),
                    float(loss_sigreg.item()),
                    float(loss_sharp.item()),
                    float(var_term_t.item()),
                    float(cov_term_t.item()),
                    float((jepa_loss_weight * loss_jepa).item()),
                    float((sigreg_weight * loss_sigreg).item()),
                    float((sharp_contrast_weight * loss_sharp).item()),
                    float((vicreg_var_weight * var_term_t).item()),
                    float((vicreg_cov_weight * cov_term_t).item()),
                    float(new_momentum),
                    float(sim_val),
                    float(var_val),
                    float(cov_val),
                    float(raw_mse_val),
                    float(norm_err_val),
                    float(valid_frac),
                    round(elapsed, 4),
                ]
            )
            # Save masked-scale usage as training log in session dir.
            scales = outputs["target_scales"].detach().cpu().numpy()
            valid = outputs["target_valid"].detach().cpu().numpy().astype(bool)
            valid_scales = scales[valid]
            if "cdd_channels_masked" in outputs:
                cube_path = os.path.join(session_dir, "example_masked_channel_cube.npy")
                if not os.path.exists(cube_path):
                    np.save(
                        cube_path,
                        outputs["cdd_channels_masked"][0].detach().cpu().numpy().astype(np.float32),
                    )
            if valid_scales.size > 0:
                uniq, cnt = np.unique(np.round(valid_scales.astype(np.float32), 6), return_counts=True)
                for s, c in zip(uniq.tolist(), cnt.tolist()):
                    masked_scale_rows.append([epoch + 1, batch_idx, float(s), int(c)])
            # Save visited target locations for full-session diagnostics.
            tloc = outputs["target_locations"].detach().cpu().numpy()
            tvalid = outputs["target_valid"].detach().cpu().numpy().astype(bool)
            tscale = outputs["target_scales"].detach().cpu().numpy()
            if visit_counts is None:
                hh, ww = int(outputs["x_clean"].shape[-2]), int(outputs["x_clean"].shape[-1])
                visit_counts = np.zeros((hh, ww), dtype=np.float32)
            for bi in range(tloc.shape[0]):
                for ki in range(tloc.shape[1]):
                    if not bool(tvalid[bi, ki]):
                        continue
                    yy = int(tloc[bi, ki, 0])
                    xx = int(tloc[bi, ki, 1])
                    if 0 <= yy < visit_counts.shape[0] and 0 <= xx < visit_counts.shape[1]:
                        visit_counts[yy, xx] += 1.0
            bsz = tloc.shape[0]
            ksz = tloc.shape[1]
            for bi in range(bsz):
                for ki in range(ksz):
                    if not bool(tvalid[bi, ki]):
                        continue
                    visited_rows.append(
                        [
                            epoch + 1,
                            batch_idx,
                            bi,
                            ki,
                            int(tloc[bi, ki, 0]),
                            int(tloc[bi, ki, 1]),
                            float(tscale[bi, ki]),
                        ]
                    )
            if batch_idx % log_interval == 0:
                print(
                    f"[{config_name}] Epoch {epoch + 1}/{epochs} Batch {batch_idx}/{len(dataloader)} "
                    f"total={_fmt_metric(total_loss.item())} jepa={_fmt_metric(loss_jepa.item())} pixel={_fmt_metric(loss_pixel_val)} "
                    f"sigreg={_fmt_metric(loss_sigreg.item())} "
                    f"sharp={_fmt_metric(loss_sharp.item())} "
                    f"sim={_fmt_metric(sim_val)} var={_fmt_metric(var_val)} cov={_fmt_metric(cov_val)} "
                    f"raw_mse={_fmt_metric(raw_mse_val)} norm_err={_fmt_metric(norm_err_val)} "
                    f"valid_frac={_fmt_metric(valid_frac)}"
                )
            epoch_total += float(total_loss.item())
            epoch_jepa += float(loss_jepa.item())
            epoch_pixel += float(loss_pixel_val)
            epoch_sim += float(sim_val)
            epoch_var += float(var_val)
            epoch_cov += float(cov_val)
            epoch_sigreg += float(loss_sigreg.item())
            epoch_sharp += float(loss_sharp.item())
            epoch_valid_frac += float(valid_frac)
            epoch_batches += 1

        if metrics_rows:
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(metrics_rows)
        if masked_scale_rows:
            with open(masked_scales_log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(masked_scale_rows)
        if visited_rows:
            with open(visited_targets_log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(visited_rows)
        if visit_counts is not None:
            np.save(os.path.join(session_dir, "visited_target_frequency.npy"), visit_counts.astype(np.float32))

        if epoch_batches > 0:
            print(
                f"[{config_name}] Epoch {epoch + 1}/{epochs} summary "
                f"avg_total={_fmt_metric(epoch_total/epoch_batches)} "
                f"avg_jepa={_fmt_metric(epoch_jepa/epoch_batches)} "
                f"avg_pixel={_fmt_metric(epoch_pixel/epoch_batches)} "
                f"avg_sigreg={_fmt_metric(epoch_sigreg/epoch_batches)} "
                f"avg_sharp={_fmt_metric(epoch_sharp/epoch_batches)} "
                f"avg_sim={_fmt_metric(epoch_sim/epoch_batches)} "
                f"avg_var={_fmt_metric(epoch_var/epoch_batches)} "
                f"avg_cov={_fmt_metric(epoch_cov/epoch_batches)} "
                f"avg_valid_frac={_fmt_metric(epoch_valid_frac/epoch_batches)}"
            )
        val_loss = 0.0
        val_sim = 0.0
        val_error_by_scale = {}
        if val_loader is not None:
            v = evaluate_validation(
                model=model,
                val_loader=val_loader,
                device=device,
                max_batches=train_cfg.get("val_max_batches"),
                vicreg_spatial_mode=vicreg_spatial_mode,
            )
            val_loss = float(v["val_loss"])
            val_sim = float(v["val_sim"])
            val_error_by_scale = dict(v["val_error_by_scale"])
            print(
                f"[{config_name}] Epoch {epoch + 1}/{epochs} validation "
                f"val_loss={_fmt_metric(val_loss)} val_sim={_fmt_metric(val_sim)} "
                f"val_error_by_scale={json.dumps(val_error_by_scale, sort_keys=True)}"
            )
        with open(epoch_summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    round(epoch_total / max(1, epoch_batches), 8),
                    round(val_loss, 8),
                    round(val_sim, 8),
                    json.dumps(val_error_by_scale, sort_keys=True),
                ]
            )
        model.train()
        # Save resumable checkpoint at the end of every epoch.
        torch.save(
            {
                "epoch": int(epoch + 1),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config_name": config_name,
            },
            resume_ckpt_path,
        )
        # Keep model_last in sync for inference-only resume paths.
        torch.save(model.state_dict(), model_ckpt_path)
        print(f"[{config_name}] checkpoint_saved={resume_ckpt_path} epoch={epoch + 1}")

    torch.save(model.state_dict(), os.path.join(session_dir, "model_last.pt"))

    session_dir = run_post_training_inference(
        model=model,
        dataloader=inference_loader,
        session_dir=session_dir,
        config_name=config_name,
        visit_counts=visit_counts,
        force_recompute_inference=force_recompute_inference,
        inference_mask_passes=inference_mask_passes,
        mask_inference=mask_inference,
        viz_crop_border=viz_crop_border,
        viz_crop_border_px=viz_crop_border_px,
        compute_jepa_energy_fn=compute_jepa_energy,
        compute_target_energy_map_fn=compute_target_energy_map,
        inference_visit_batches=inference_visit_batches,
        training_d4_augment=bool(data_cfg.get("d4_augment", False)),
    )
    # Keep dashboard artifacts in sync with inference outputs for all runs.
    # This writes session/results/* embedding files required by session_to_dash.py.
    inf_path = os.path.join(session_dir, "inference_outputs.pt")
    if os.path.exists(inf_path):
        try:
            outputs = torch.load(inf_path, map_location="cpu")
            dash_path = save_inference_dashboard(session_dir, outputs, umap_cfg=umap_cfg)
            print(f"[{config_name}] dashboard_saved={dash_path}")
            effective_rank = ""
            rank_diag = {}
            try:
                rank_diag = rank_dashboard(outputs)
                with open(os.path.join(session_dir, "rank_diagnostics.json"), "w", encoding="utf-8") as f:
                    json.dump(rank_diag, f, indent=2)
            except Exception as er:
                print(f"[{config_name}] warning: rank_diagnostics_failed: {type(er).__name__}: {er}")
            if compute_effective_rank:
                try:
                    if "pred" in rank_diag and "erank" in rank_diag["pred"]:
                        effective_rank = f"{float(rank_diag['pred']['erank']):.8f}"
                    else:
                        pred_map = outputs.get("pred_map")
                        if pred_map is not None:
                            pm = torch.as_tensor(pred_map)
                            z = pm[0].detach().cpu().permute(1, 2, 0).reshape(-1, int(pm.shape[1])).numpy()
                            effective_rank = f"{compute_effective_rank_from_features(z):.8f}"
                except Exception as er:
                    print(f"[{config_name}] warning: effective_rank_failed: {type(er).__name__}: {er}")
            # Dedicated artifact for simple downstream collection.
            # Empty string means rank was not computed for this run.
            with open(os.path.join(session_dir, "effective_rank.txt"), "w", encoding="utf-8") as f:
                f.write(f"{effective_rank}\n")
            with open(os.path.join(session_dir, "effective_rank.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": int(time.time()),
                        "config_name": config_name,
                        "compute_effective_rank": bool(compute_effective_rank),
                        "effective_rank": (None if effective_rank == "" else float(effective_rank)),
                    },
                    f,
                    indent=2,
                )
            run_results_path = os.path.join(session_dir, "run_results.csv")
            if not os.path.exists(run_results_path):
                with open(run_results_path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(["timestamp", "config_name", "compute_effective_rank", "effective_rank"])
            with open(run_results_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([int(time.time()), config_name, int(compute_effective_rank), effective_rank])
        except Exception as e:
            print(f"[{config_name}] warning: dashboard generation failed: {type(e).__name__}: {e}")
    else:
        print(f"[{config_name}] warning: inference_outputs.pt missing; skip dashboard generation")
    return session_dir
