# Dual-Encoder Target-Pair ConvNeXt-JEPA

Config-driven training pipeline for a dual-encoder ConvNeXt-Tiny JEPA with EMA teacher updates and pixel-level decoder supervision.

## Project Layout

```text
convnext_jepa/
├── configs/                 # Experiment configs (*.json)
├── data/                    # Your datasets
├── results/                 # Generated plots/reports
├── sessions/                # Per-run artifacts
├── scripts/
│   └── session_to_plots.py  # Convert session metrics -> plots
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── models/
│   └── utils/
├── main.py                  # Single-config entrypoint
├── run.sh                   # Loop all configs
└── requirements.txt
```

## Setup

```bash
cd /Users/gxli/proj/ml/multiscale_conv_jepa/convnext_jepa
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Run Training

Run all configs in `configs/`:

```bash
./run.sh
```

Run a single config:

```bash
python3 main.py --config configs/base_3090.json --sessions-dir sessions
```

## Session Artifacts

Each run creates/updates `sessions/<config_name>/` with:

- `config_used.json`: exact config snapshot
- `metrics.csv`: training metrics by batch
- `model_last.pt`: final model checkpoint
- `inference_outputs.pt`: saved inference tensors

## Blur Demo / Channel Maps

Generate the blur demo outputs for one config:

```bash
python3 scripts/blur_demo.py --config configs/test_run_data.json --sessions-dir sessions
```

Outputs in `sessions/<config_name>/`:

- `blur_demo.png`: 4-panel plot
  - Original
  - Center Masked
  - Ratio `(I2 / I1)`
  - Fractional change `(I2 - I1) / (I2 + I1)`
- `blur_demo_channels.png`: per-channel maps (`Original`, `Masked`, `Delta`) when enabled
- `cdd_result.npy`: CDD component channels (all channels returned by CDD)
- `cdd_residual.npy`: CDD residual
- `blur_demo_meta.json`: run metadata (scales, spacing, center count, etc.)

Enable/disable channel maps in config:

```json
"blur_demo": {
  "make_channel_plot": true
}
```

Key spacing controls in config:

```json
"blur_demo": {
  "mask_scale": 2.0,
  "pyramid_spacing_mult": 2.0
}
```

Current spacing rule:

- `spacing_px = largest_scale * mask_scale * pyramid_spacing_mult`
- center count is auto-determined from available area when `num_random_centers` is `"auto"`.

Note on macOS/MPS:

- The message `No supported GPU was found.` is printed by the CDD package's GPU check (CUDA-oriented).  
  It does not mean your Apple MPS runtime is unavailable for PyTorch generally.

## Masking Demo (JEPA Context Mask)

Generate a masking demo that uses the current config and current dataset pipeline:

```bash
python3 scripts/masking_demo.py --config configs/test_run_data.json --sessions-dir sessions
```

Optional reproducibility controls:

```bash
python3 scripts/masking_demo.py \
  --config configs/test_run_data.json \
  --sessions-dir sessions \
  --sample-index 0 \
  --seed 42
```

Outputs in `sessions/<config_name>/`:

- `masking_demo.png`: 4-panel visualization
  - Original
  - Masked/Context
  - Mask overlay (changed regions in red)
  - `|Context - Original|` magnitude map
- `masking_demo_meta.json`: run metadata (config, sample, sigmas, cell sizes, target count)

How it matches training:

- Loads data through `JEPADataset` with your config values (`data_root`, `npy_pattern`, slice strategy, `image_size`, log settings, etc.).
- Applies context masking via the same function used in model forward pass: `make_pyramid_grid_context(...)`.
- Uses `model.sigmas`, `model.cell_sizes`, and `model.sigmas` and `model.cell_sizes` from the active config.

## Generate Plots

```bash
python3 scripts/session_to_plots.py --sessions-dir sessions --results-dir results
```

Outputs loss curves as PNG files in `results/` (one per session).

## Config Schema

Example: `configs/base_3090.json`

- `model.pretrained` (bool)
- `data.num_samples` (int)
- `data.image_size` (int)
- `train.epochs` (int)
- `train.batch_size` (int)
- `train.num_workers` (int)
- `train.lr` (float)
- `train.weight_decay` (float)
- `train.momentum` (float, EMA)
- `train.log_interval` (int)
- `loss.weight_jepa` (float)
- `loss.weight_pixel` (float)

## Notes

- Mixed precision (`torch.amp`) is enabled automatically when CUDA is available.
- The dataset is currently a template with synthetic tensors; replace `src/dataset.py` with real data loading.
