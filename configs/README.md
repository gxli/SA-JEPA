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
- masking behavior (`constant_mask_box`, `mask_box_size`, `mask_size_scaling`, `mask_spacing_scaling`),
- target sampling policy (`target_sampling_mode`, priority knobs),
- schedule/loss logging (`epochs`, `log_interval`, vicreg/sigreg/loss weights).

Session configs always override base keys after merge.

## How it works

`src.train.load_config()` supports:

- `base_config`: relative or absolute path to another JSON config.
- Recursive deep merge (child overrides parent keys).
- Cycle detection for bad base chains.

## Canonical masking keys

Use only:

- `model.mask_size_scaling`
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

## 3D Slab Mode

The only supported 3D mode is `model.mode: "3d_slab"` with
`data.input_type: "cube"`. The model consumes the raw 3D field directly,
applies box masks that intersect a thin center slab, and computes 3D patch loss
inside that slab. Use `model.slab_depth` to set its thickness.

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

`JEPADataset` always preserves native input resolution. Do not use the removed
`data.image_size` key; there is no implicit resize step.

SigReg always uses pre-predictor context patch embeddings when
`sigreg_weight > 0`. Do not use the removed `sigreg_on_pred` selector.
