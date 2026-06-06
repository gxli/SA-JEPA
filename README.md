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

## Inference from a Trained Session

Load any trained session checkpoint and run inference on arbitrary `.npy` data — supports large files via crop/tile and 3D volumes via slab mode. Two interfaces: config file or CLI flags.

### Config-driven (recommended)

```bash
python -m src.inference_from_session --config configs/inference/chengdu.json
```

CLI flags override config values:

```bash
python -m src.inference_from_session --config configs/inference/chengdu.json --crop-size 128
```

### CLI flags

```bash
python -m src.inference_from_session \
    --session sessions/gen_121_mhd_run_006_ms1p2 \
    --input data/chengdu.npy \
    --output-session sessions/inference_chengdu
```

### Inference config schema

```json
{
  "session": "sessions/gen_121_mhd_run_006_ms1p2",
  "input": "data/chengdu.npy",
  "output_session": "sessions/inference_chengdu",
  "crop_size": null,
  "crop_mode": "center",
  "mode": "image",
  "slice_axis": 0,
  "slice_index": null,
  "batch_size": 2,
  "tta": false,
  "tta_mode": "d4",
  "device": null
}
```

### Crop / Tile for large data

When input exceeds GPU memory, crop to a fixed size:

```bash
# Center crop (single patch)
python -m src.inference_from_session \
    --session sessions/gen_121_mhd_run_006_ms1p2 \
    --input data/huge_mosaic.npy \
    --crop-size 256 \
    --crop-mode center \
    --output-session sessions/inference_mosaic_center

# Tiled (sliding window, 50% overlap, all tiles processed)
python -m src.inference_from_session \
    --session sessions/gen_121_mhd_run_006_ms1p2 \
    --input data/huge_mosaic.npy \
    --crop-size 256 \
    --crop-mode tile \
    --output-session sessions/inference_mosaic_tiled
```

### 3D slab mode

For 3D volumes (e.g., NGC data), each depth slice is processed independently:

```bash
# All slices
python -m src.inference_from_session \
    --session sessions/gen_121_mhd_run_006_ms1p2 \
    --input data/ngc3627_12m+7m+tp_co21_strict_mom0.npy_sm.npy \
    --mode 3d_slab \
    --slice-axis 0 \
    --output-session sessions/inference_ngc_3d

# Single slice
python -m src.inference_from_session \
    --session sessions/gen_121_mhd_run_006_ms1p2 \
    --input data/ngc3627_12m+7m+tp_co21_strict_mom0.npy_sm.npy \
    --mode 3d_slab \
    --slice-axis 0 \
    --slice-index 42 \
    --output-session sessions/inference_ngc_slice42
```

### TTA (Test-Time Augmentation)

Enable D4 augmentation during inference to average rotations and flips:

```bash
python -m src.inference_from_session \
    --session sessions/gen_121_mhd_run_006_ms1p2 \
    --input data/chengdu.npy \
    --tta \
    --tta-mode d4 \
    --output-session sessions/inference_chengdu_tta
```

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--session` | (required) | Path to trained session directory |
| `--input` | (required) | Path to input `.npy` file |
| `--output-session` | auto | Output session directory (auto-generates if omitted) |
| `--crop-size` | `None` | Crop/tile size for large inputs |
| `--crop-mode` | `center` | `center` (single patch) or `tile` (sliding window) |
| `--mode` | `image` | `image` (2D) or `3d_slab` (3D volume) |
| `--slice-axis` | `0` | Depth axis for 3D slab mode |
| `--slice-index` | `None` | Specific slice for 3D mode (omitted = all) |
| `--batch-size` | `2` | Batch size for inference |
| `--tta` | disabled | Enable test-time augmentation |
| `--tta-mode` | `flip4` | TTA view set: `flip4`, `rot4`, or `d4` |
| `--device` | auto | Override device (`cuda`, `mps`, `cpu`) |

### Inference session structure

The output session at `sessions/<name>/` contains:

- `config_used.json` — frozen config with `_inference.inference_only: true`
- `inference_outputs.pt` — full output dict with `pred_map`, `gt_map`, `context_map`
- `pred_map.npz`, `gt_map.npz`, `context_map.npz` — compressed latent maps
- `network_input_clean.npz`, `network_input_context.npz` — input snapshots
- `jepa_energy_summary.json` — scalar energy + metadata
- `dash_data.npz` — dashboard-compatible visualization data

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

- Loads data through `JEPADataset` with your config values (`data_root`, `npy_pattern`, slice strategy, crop settings, etc.).
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
- `data.crop_mode` / `data.crop_size` (optional crop training)
- `train.epochs` (int)
- `train.batch_size` (int)
- `train.num_workers` (int)
- `train.lr` (float)
- `train.weight_decay` (float)
- `train.momentum` (float, EMA)
- `train.log_interval` (int)
- `loss.weight_jepa` (float)
- `loss.weight_pixel` (float)

## Inference Mode

Run a trained model on arbitrary `.npy` or `.fits` data without a config file:

```bash
python -m src.inference_from_session \
    --session sessions/gen_127_mhd_run_001_ms1p2 \
    --input data/chengdu.npy \
    --output-session sessions/inference_chengdu
```

Or via config:

```bash
python -m src.inference_from_session --config configs/inference/chengdu.json
```

### Large images — tiled inference

For images that exceed GPU memory, use `--crop-mode tile` with `--crop-size`:

```bash
python -m src.inference_from_session \
    --session sessions/gen_127_mhd_run_001_ms1p2 \
    --input data/huge_mosaic.npy \
    --crop-size 256 --crop-mode tile \
    --output-session sessions/inference_mosaic
```

Tiles are stitched on CPU to avoid GPU OOM. 50% overlap, simple averaging.

### Test-time augmentation

```bash
python -m src.inference_from_session \
    --session ... --input ... \
    --tta --tta-mode flip4
```

### 3D slab mode

```bash
python -m src.inference_from_session \
    --session ... --input data/volume.npy \
    --mode 3d_slab --slice-axis 0
```

### Output

Creates `sessions/<name>/` with:
- `inference_outputs.pt` — full tensors
- `pred_map.npz`, `gt_map.npz`, `context_map.npz`
- `config_used.json` — marked `inference_only: true`
- Dashboard artifacts (PCA/UMAP arrays for `session_to_dash.py`)

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--session` | required | Path to trained session |
| `--input` | required | Path to `.npy` or `.fits` file |
| `--crop-size` | None | Crop size for large inputs |
| `--crop-mode` | center | `center` or `tile` |
| `--mode` | image | `image` or `3d_slab` |
| `--output-session` | auto | Output directory |
| `--batch-size` | 2 | Per-GPU batch size |
| `--tta` | off | Enable test-time augmentation |
| `--tta-mode` | flip4 | `flip4`, `rot4`, or `d4` |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--allow-partial-load` | off | Skip strict checkpoint matching |

## Notes

- Mixed precision (`torch.amp`) is enabled automatically when CUDA is available.
- The dataset is currently a template with synthetic tensors; replace `src/dataset.py` with real data loading.
