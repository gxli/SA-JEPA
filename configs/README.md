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

## Shared CDD/log knobs

Keep these in `data` only in config files:

- `cdd_mode`
- `cdd_constrained`
- `cdd_sm_mode`
- `log_eps`

The loader mirrors them into `model` only when missing, so files stay DRY.
