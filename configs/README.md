# ⚙️ Configuration Knobs Dictionary

Every key in `base_pyramid_scaleaware_convnext.yaml` (merged automatically into
all configs — see [load_config doc](../README.md#-warning-always-inherit-from-the-base-config)).

---

## 1. Data & Preprocessing Pipeline

| Key | Default | Description |
|:----|:-------:|:------------|
| `data.data_root` | `data` | Root directory for input `.npy` files. |
| `data.npy_pattern` | `*.npy` | Glob pattern matching input files. |
| `data.input_type` | `image` | `image` (2D) or `cube` (3D volume). |
| `data.num_samples` | `200` | Virtual dataset size (augmented views per epoch). |
| `data.d4_augment` | `true` | Dihedral D4 augmentation (rotations + flips). |
| `data.crop_mode` | `none` | `none`, `center` (single window), or `tile` (sliding). |
| `data.crop_size` | — | Tile size in pixels for `crop_mode`. |
| `data.crop_min_valid_fraction` | `0.5` | Minimum valid fraction to accept a tile. |
| `data.cube_slice_strategy` | `center` | How 2D slices are extracted from 3D cubes. |
| `data.cube_slice_axis` | `0` | Depth axis for cube slicing. |
| `data.cube_slice_index` | `0` | Fixed slice index when strategy is `fixed`. |
| `data.target_mask` | — | Optional `.npy` binary mask: 1 = valid target pixel, 0 = skip. |
| `data.target_threshold` | — | Auto-generate target mask from raw data: pixels above threshold are valid targets (e.g. `1e-4`). |

## 2. CDD (Constrained Diffusion Decomposition)

| Key | Default | Description |
|:----|:-------:|:------------|
| `model.cdd_mode` | `log` | CDD decomposition mode (`log` recommended). |
| `model.cdd_constrained` | `true` | Enforce `sum(scales) + residual = input`. |
| `model.cdd_sm_mode` | `reflect` | Boundary smoothing mode for CDD. |
| `model.cdd_append_last_residual` | `true` | Fold residual into last scale channel. |
| `model.cdd_precompute` | `true` | GPU-precompute CDD cache; stored per-session. |
| `model.cdd_log_std_floor_mult` | `0.05` | Log-transform floor = `max(eps, std × floor_mult)`. |
| `model.sigmas` | `[2,4,8,16]` | CDD scale hierarchy; 5 scales recommended `[2,4,8,16,32]`. |
| `model.align_scales` | `true` | Align mask centers across scale levels. |
| `data.log_eps` | `1e-6` | Floor constant for log-preprocessing. |

## 3. Scale-Aware Masking Geometry

| Key | Default | Description |
|:----|:-------:|:------------|
| `model.mask_size_scaling` | `1.0` | Box size = `round(sigma × scaling + mask_size)`. Set to `0` for fixed boxes. |
| `model.mask_size` | `0` | Additive constant to mask box size in pixels. |
| `model.mask_spacing_scaling` | `2.0` | Grid spacing = `box_size × spacing_scaling`. |
| `model.mask_box_hardcap` | — | Hard maximum on mask box size (px). |
| `model.patch_size` | `3` | Target patch size for JEPA prediction. |
| `model.target_sampling_mode` | `random` | `random`, `priority`, or `priority_small_scale`. |
| `model.target_nonoverlap` | `true` | Prevent target patches from overlapping. |
| `model.target_allow_partial_overlap` | `0.0` | Tolerance for partial overlap (0 = strict). |
| `model.target_invalid_region_skip` | `true` | Skip targets in NaN/FOV border regions. |
| `model.active_target_fraction` | `1.0` | Fraction of candidate grid cells that are eligible. |
| `model.priority_top_percent` | `100` | Priority sampling: top-% of high-gradient cells. |
| `model.priority_n_target` | `auto` | Priority sampling: number of target candidates. |
| `model.priority_min_targets_per_map` | `10` | Fallback minimum targets when priority is scarce. |
| `model.priority_dithering_pixels` | `6` | Jitter radius (px) for priority-selected targets. |
| `model.priority_candidate_oversample` | `0` | Oversampling factor for candidate pool. |
| `model.global_shift` | `false` | Global lattice shift for target grid. |

## 4. ConvNeXt Backbone Architecture

| Key | Default | Description |
|:----|:-------:|:------------|
| `model.mode` | `pyramid` | `pyramid` (2D), `3d_slab` (3D), `3d_full_volume` (3D infer). |
| `model.model_key` | `cdd_scaleaware_convnext` | Encoder variant selector. |
| `model.encoder_width` | `64` | Base channel width for ConvNeXt blocks. |
| `model.encoder_depth` | `4` | Number of ConvNeXt blocks. |
| `model.encoder_kernel_size` | `7` | Depthwise convolution kernel size. |
| `model.latent_channels` | `32` | Output channel count (dense latent atlas). |
| `model.predictor_hidden` | `96` | JEPA predictor internal hidden channels. |
| `model.predictor_layernorm` | `true` | LayerNorm in predictor MLP. |
| `model.predictor_spatial_conv` | `true` | Spatial convolution in predictor. |
| `model.predictor_residual` | `false` | Residual connection in predictor blocks. |
| `model.use_grn` | `true` | Global Response Normalization (ConvNeXt V2). |
| `model.normalize_loss_l2` | `false` | L2-normalize latent patches before loss. |
| `model.post_log_transform` | `true` | Log-transform input before feeding to encoder. |

**Scale-Aware Adapter (FiLM / Per-Scale Norms)**

| Key | Default | Description |
|:----|:-------:|:------------|
| `model.scaleaware_feat_channels` | `8` | Scale-conditioning embedding width. |
| `model.scaleaware_adapter_kernel_size` | `3` | Adapter convolution kernel. |
| `model.scaleaware_fusion_type` | `topdown` | How scale embeddings are fused into features. |
| `model.scaleaware_norm_per_scale` | `true` | Per-scale normalization in stem. |
| `model.scaleaware_final_norm` | `true` | LayerNorm after feature fusion. |
| `model.scaleaware_stem_norm` | `true` | Normalization in input stem. |
| `model.scaleaware_adapter_norm` | `true` | Normalization in scale adapters. |

**Dilation Presets (2D)**

| Preset | Dilations | Receptive Field (k=7) |
|:-------|:----------|:----------------------:|
| Standard | `[1, 1, 1, 1]` | 29 px (~15 px usable) |
| Wide-field | `[1, 1, 2, 4]` | 49 px (~25 px usable) |

Set via `model.convnext_layer_dilations: [1, 1, 2, 4]`.

## 5. JEPA Predictor & EMA Target Schedules

| Key | Default | Description |
|:----|:-------:|:------------|
| `model.ema_momentum` | `0.996` | Base EMA momentum for target encoder update. |
| `train.ema_momentum_base` | `0.99` | Initial EMA momentum (epoch 1). |
| `train.ema_momentum_final` | `0.9999` | Asymptotic EMA momentum. |
| `model.use_symmetric_feature_loss` | `false` | Enable flip-symmetry loss (2D only). |
| `train.symmetry_loss_weight` | `0.0` | Weight for symmetric feature loss (set to `0.003` to enable). |

## 6. Loss Optimization & Entropy Regularization

| Key | Default | Description |
|:----|:-------:|:------------|
| `train.prediction_loss_weight` | `50` | JEPA latent prediction MSE multiplier. |
| `train.spread_regularizer.type` | `std_hinge` | Anti-collapse regularizer: `std_hinge`, `weak_sigreg`, `sketched_sigreg`. |
| `train.spread_regularizer.target` | `context` | Which encoder to regularize (`context` or `target`). |
| `train.spread_regularizer.spatial_mode` | `pooled` | `pooled` (per-map) or `per_patch`. |
| `train.spread_regularizer.weight` | `2` | Regularizer multiplier (recommend `5` for production). |
| `train.spread_regularizer.target_std` | `1.0` | Target standard deviation for std_hinge. |
| `train.spread_regularizer.eps` | `0.0001` | Numerical stability epsilon. |
| `train.vicreg_spatial_mode` | `pooled` | VICReg spatial aggregation mode. |

## 7. Training Loop & Hardware

| Key | Default | Description |
|:----|:-------:|:------------|
| `train.epochs` | `10` | Number of training epochs. |
| `train.batch_size` | `4` | Per-GPU batch size. |
| `train.gradient_accumulation_steps` | `1` | Effective batch = `batch_size × grad_accum`. |
| `train.lr` | `0.0001` | Base learning rate (AdamW). |
| `train.weight_decay` | `1e-5` | AdamW weight decay. |
| `train.num_workers` | `8` | DataLoader worker processes. |
| `train.inference_tta_enabled` | `false` | Test-time augmentation during inference. |
| `train.inference_tta_mode` | `flip4` | TTA view set: `flip4`, `rot4`, `d4`. |
| `train.inference_discard_margin` | FOV/2 | Border pixels discarded during inference. Set to `0` for full-image presentation. |
| `train.force_recompute_inference` | `false` | Re-run inference even if `inference_outputs.pt` exists. |
| `train.post_training_artifacts` | `true` | Generate PCA/UMAP embeddings after training. |

## 8. Diagnostics & Visualization

| Key | Default | Description |
|:----|:-------:|:------------|
| `train.compute_effective_rank` | `true` | Compute effective manifold rank per epoch. |
| `train.scale_probe_enabled` | `true` | Run scale-response probe after training. |
| `train.viz_crop_border` | `true` | Crop encoder-FOV border in visualizations. |
| `train.umap.metric` | `euclidean` | UMAP distance metric. |
| `train.umap.standardize` | `true` | Standardize latent features before UMAP. |
| `train.umap.l2_normalize` | `false` | L2-normalize latent features before UMAP. |
| `train.umap.n_neighbors` | `50` | UMAP neighborhood size. |
| `train.umap.min_dist` | `0.2` | UMAP minimum embedding distance. |
