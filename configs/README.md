# Config Layout (Base + Overrides)

Use a shared base config and tiny experiment overrides.

## Files

- `base_pyramid_scaleaware_convnext.json`: production baseline.
- `experiments/*.json`: per-run overrides using `base_config`.

## Base vs Session Split

- Put stable defaults in `base_*`:
- data path/pattern, CDD/log policy, common UMAP, optimizer defaults.
- Put run/session knobs in `experiments/*`:
- model family/arch choice (`mode`, `model_key`, encoder depth/width/kernel),
- masking behavior (`constant_mask_box`, `mask_footprint_px`, `mask_scale_factor`, `mask_spacing_scaling`),
- target sampling policy (`target_sampling_mode`, priority knobs),
- schedule/loss logging (`epochs`, `log_interval`, VICReg/spread/loss weights).

Session configs always override base keys after merge.

## How it works

`src.train.load_config()` supports:

- `base_config`: relative or absolute path to another JSON config.
- Recursive deep merge (child overrides parent keys).
- Cycle detection for bad base chains.

## Canonical masking keys

Use only:

- `model.mask_scale_factor`
- `model.mask_spacing_scaling`

Do not use legacy keys:

- `model.mask_scale`
- `model.spacing_scale`
- `model.mask_scaling_box`

## Model-Side CDD

CDD decomposition is performed only by model-side masking. The dataset loads
and normalizes images without running CDD or maintaining a CDD cache.

Use `model.sigmas` as the single source of truth for CDD decomposition scales.
There is no standalone Gaussian masking mode or `model.blur_mode` selector.

## 3D Modes

Supported 3D modes require `data.input_type: "cube"`.

- `model.mode: "3d_slab"` consumes 3D crops, applies box masks that intersect
  a thin center slab, and computes 3D patch loss inside that slab. Use
  `model.slab_depth` to set its thickness.
- `model.mode: "3d_full_volume"` uses the same 3D encoder path but computes
  targets across the full crop depth. Set `data.volume_crop_depth: full` to
  train on the full cube depth, or provide an integer crop depth.

## Shared CDD/log knobs

Keep these in `data` only in config files:

- `cdd_mode`
- `cdd_constrained`
- `cdd_sm_mode`
- `log_eps`

The loader mirrors them into `model` only when missing, so files stay DRY.

Dataset preprocessing is always linear `normalize01`. Do not use
`data.log_transform`; `model.post_log_transform` is the only runtime log
switch. The model applies it after masking.

`JEPADataset` preserves native input resolution by default. Do not use the
removed `data.image_size` key; there is no implicit resize step. To train large
2D images on random crops, set:

- `data.crop_mode`: `"random"`
- `data.crop_size`: an integer for square crops or `[height, width]`

Validation uses a deterministic center crop of the same size. Post-training
inference keeps native resolution so exported maps cover the full image.
CDD always adds one leading scale dimension: `(H, W) -> (S, H, W)` and
`(D, H, W) -> (S, D, H, W)`. For a 3D array consumed as 2D slices, CDD is
cached on the full cube first. The dataset then selects one aligned CDD slice
and finally applies the 2D crop.

The spread regularizer uses predictor patch embeddings by default, matching the
JEPA prediction manifold. Configure only the explicit standard-deviation hinge:

```yaml
spread_regularizer:
  type: std_hinge
  target: predictor
  weight: 2
  target_std: 1.0
  eps: 1.0e-4
```

Variance and covariance regularizers are experimental. To enable them for an
ablation, place `vicreg_var_weight` and `vicreg_cov_weight` under
`train.experimental_losses`; they are not production loss terms.
