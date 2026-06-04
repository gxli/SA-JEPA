# Gen 124 — Training Defaults Reference

## Architecture (fixed)

| Knob | Value | Notes |
|------|-------|-------|
| `model_key` | `cdd_scaleaware_convnext` | ConvNeXt scale-aware pyramid encoder |
| `mode` | `pyramid` | CDD multi-scale decomposition |
| `sigmas` | `[2, 4, 8, 16]` | CDD diffusion scales (from base config) |
| `encoder_depth` | `4` | ConvNeXt blocks (4 = ~2M params) |
| `encoder_width` | `64` | Hidden channels |
| `latent_channels` | `32` | Output embedding dim |
| `encoder_kernel_size` | `7` | Depthwise conv kernel |

## Scale-aware encoder

| Knob | Value | Notes |
|------|-------|-------|
| `scaleaware_feat_channels` | `8` | Per-scale adapter output channels |
| `scaleaware_fusion_type` | `topdown` | Coarse→fine residual fusion |
| `scaleaware_norm_per_scale` | `true` | LayerNorm per scale in adapter |
| `scaleaware_adapter_norm` | `true` | Norm in shared adapter |
| `scaleaware_stem_norm` | `true` | Norm in ConvNeXt stem |
| `scaleaware_final_norm` | `true` | Norm after encoder head |
| `scaleaware_adapter_kernel_size` | `3` | Adapter conv kernel |

## Masking — the two key knobs to sweep

| Knob | Effect | Range |
|------|--------|-------|
| `mask_scale_factor` | Scale multiplier: `box = σ × factor + footprint` | `0` = pure fixed-box, `0.4–2.0` = pyramid |
| `mask_footprint_px` | Fixed additive box size (px) | `0` = pure pyramid, `3–15` = fixed-box |
| `mask_spacing_scaling` | Grid spacing = `box × spacing` | `2.0` (default) |

**Key:** `mask_scale_factor=0` + `mask_footprint_px=N` → pure fixed-box. `mask_scale_factor=N` + `mask_footprint_px=0` → pure scale-tied pyramid.

## Target sampling

| Knob | Value | Notes |
|------|-------|-------|
| `target_sampling_mode` | `random` | `random` or `priority` (flux-ranked) |
| `priority_top_percent` | `100` | Full catalogue (no filtering) |
| `priority_n_target` | `auto` | Auto-compute from image area |
| `target_nonoverlap` | `false` | Allow overlapping target patches |

## Regularization

| Knob | Value | Effect |
|------|-------|--------|
| `normalize_loss_l2` | `true` | L2-normalize patch embeddings before MSE |
| `predictor_spatial_conv` | `true` | 3×3 spatial conv in predictor (`false` = 1×1) |
| `spread_regularizer.weight` | `2.0` | Hinge loss pushing embedding std toward 1.0 |
| `symmetric_feature_loss_weight` | `0.003` | 4-way rotational invariance penalty |
| `use_grn` | `true` | Global Response Normalization in ConvNeXt blocks |
| `post_log_transform` | `true` | log(1 + x) transform after CDD |
| `predictor_hidden` | `96` | Predictor MLP hidden dim |
| `predictor_layernorm` | `false` | No LayerNorm in predictor |

## Training

| Knob | Value | Notes |
|------|-------|-------|
| `epochs` | `10` | Full passes |
| `batch_size` | `4` | Per GPU (256×256 fits on 24 GB) |
| `lr` | `1e-4` | AdamW base learning rate |
| `min_lr` | `1e-6` | Cosine floor |
| `warmup_epochs` | `1.0` | Linear warmup |
| `weight_decay` | `1e-5` | AdamW regularization |
| `ema_momentum_base` | `0.99` | Initial EMA |
| `ema_momentum_final` | `0.9999` | Final EMA |
| `ema_warmup_fraction` | `0.25` | Fraction of training to anneal EMA |

## Loss weights

| Knob | Value | Notes |
|------|-------|-------|
| `mse_loss_weight` | `50` | Primary JEPA prediction loss |
| `vicreg_var_weight` | `0` | Variance term (disabled) |
| `vicreg_cov_weight` | `0` | Covariance term (disabled) |
| `vicreg_spatial_mode` | `pooled` | Per-patch pooling |

## Inference

| Knob | Value | Notes |
|------|-------|-------|
| `inference_tta_enabled` | `true` | Test-time 4-way flip augmentation |
| `inference_tta_mode` | `flip4` | 4-fold rotational TTA |
| `scale_probe_enabled` | `true` | Per-scale response analysis |
| `compute_effective_rank` | `true` | Embedding rank diagnostic |

## Data

| Knob | Value | Notes |
|------|-------|-------|
| `npy_pattern` | dataset-specific | MHD: `C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy` |
| `cdd_mode` | `log` | Logarithmic CDD decomposition |
| `cdd_constrained` | `true` | Non-negative CDD |
| `d4_augment` | `true` | Random 90° rotations during training |
