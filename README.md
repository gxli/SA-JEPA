# `sajepa` — ScaleAware JEPA for Continuous Physical Fields

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

`sajepa` is a PyTorch implementation of **ScaleAware JEPA**, a non-generative,
self-supervised method for constructing dense, label-free, spatially
back-mappable latent coordinates for continuous two-dimensional scalar fields.

---

## 🏛️ Design & Features

ScaleAware JEPA replaces pixel-reconstruction architectures with dual-branch
joint-embedding predictions optimized for continuous structures:

- **ScaleAware Processing:** Constrained Diffusion Decomposition (CDD)
  decomposes an input field into aligned fine-to-coarse components.  The
  encoder receives these components alongside scale-aware masks, meaning
  the context available for prediction dynamically scales with the hidden
  target structure.
- **JEPA Latent Prediction:** An online encoder processes a masked field
  and predicts the latent representation produced by an Exponential Moving
  Average (EMA) target encoder from the corresponding unmasked field.
  Training matches latent representations at hidden target locations instead
  of reconstructing noisy pixels.
- **Dense Latent Exploration:** Every pixel receives a distinct latent
  coordinate that maps back directly to the original field coordinate space.

---

## 🎯 Application Scenarios

`sajepa` is explicitly engineered for label-free representation learning of
continuous 2D scalar fields where distinct objects, semantic boundaries, or
segmentation rules are not known in advance.

**Standard Discovery Workflow**

1. **Train** on raw scalar fields without human annotations or discrete tokens.
2. **Extract** the dense, pixel-registered latent representation space.
3. **Inspect** latent cluster neighborhoods globally using integrated PCA or UMAP tools.
4. **Back-map** selected latent neighborhoods to the original raw arrays to examine physical morphology.

**Ideal Use Cases:** Continuous scalar fields — MHD turbulence, astronomical
intensity maps, nighttime-light rasters, and other physical distributions.

---

## ⚡ Installation

```bash
git clone https://github.com/gxli/SA-JEPA.git
cd SA-JEPA
pip install -e .
```

> **Apple Silicon (MPS) Note:** If you encounter missing operations on Apple
> GPUs (such as a CDD operation utilizing `avg_pool3d`), configure your
> environment to use the native PyTorch fallback engine:
>
> ```bash
> export PYTORCH_ENABLE_MPS_FALLBACK=1
> ```

---

## 🚀 Quick Start

Load an arbitrary 2D scalar array, fit a baseline model, and extract its dense
latent atlas. The bundled examples use
`configs/examples/mhd_example.yaml` as the canonical prototype: MHD data, `mask_size_scaling=1.2`, pooled context `std_hinge`,
and spread weight `5`.

### 📊 Dashboard Output

After any run, two self-contained HTML files land in the session directory:

| File | Purpose |
|:---|:---|
| `sessions/<name>/dashboard.html` | Plotly diagnostic dashboard (loss curves, latent projections, rank metrics) |
| `sessions/<name>/results/interactive_umap_predict.html` | Click-to-similarity interactive UMAP browser |

Reopen later:
```python
model = ScaleAwareJEPA.load_session("sessions/my_run")
model.open_dashboard()           # opens dashboard.html
model.open_interactive_umap()    # opens interactive UMAP
```

### 🐍 Python API

```python
import numpy as np
import torch
from sajepa import ScaleAwareJEPA

# 1. Load any 2D scalar field (H, W)
field = torch.from_numpy(np.load("path/to/your_field.npy"))

# 2. Train a default scale-aware model
model = ScaleAwareJEPA()
model.fit(field, epochs=10, session_dir="sessions/my_run")

# 3. Extract the dense latent atlas → shape (C_latent, H, W)
latent = model.extract(field)

# 4. Generate dashboards in the session directory
model.generate_dashboard()                # → sessions/my_run/dashboard.html
umap_html = model.open_interactive_umap() # → sessions/my_run/results/interactive_umap_predict.html

# 5. Save everything
model.save_session("sessions/my_run")

print(f"Dashboard:     sessions/my_run/dashboard.html")
print(f"Interactive:   {umap_html}")
```

### ⚙️ Config-driven

```python
from sajepa import ScaleAwareJEPA

model = ScaleAwareJEPA(config="configs/examples/mhd_example.yaml")
model.train(config_name="mhd_example", sessions_dir="sessions", dashboard=True)

print(f"Dashboard:     {model.session_dir}/dashboard.html")
```

### ⌨️ Command Line

```bash
sajepa-train --config configs/examples/mhd_example.yaml --sessions-dir sessions
```

Output lands in `sessions/mhd_example/` — open `sessions/mhd_example/dashboard.html`.

---

**Reloading & Continuing Workspaces**

```python
# Restore an existing saved model session
model = ScaleAwareJEPA.load_session("sessions/my_run")
latent = model.extract(field)

# WEIGHTS-ONLY SEED: Warm-start on new data using prior weights (resets optimizers & epochs)
model.fit(
    new_field,
    epochs=10,
    session_dir="sessions/refine_new_data",
    base_session="sessions/my_run",
    base_session_mode="weights",
)

# FULL CONTINUATION: Resume training matching optimizer, scaler states, and historical epoch values
model.fit(
    new_field,
    epochs=30,
    session_dir="sessions/continue_old_run",
    base_session="sessions/my_run",
    base_session_mode="resume",
)
```

---

## 📂 Run Session Output Structure

Every executed training directory produces a self-contained folder structure.
After post-training inference and optional dashboard generation, common artifacts
include:

| File Path | Artifact Contents |
|:---|:---|
| `sessions/my_run/model_last.pt` | Latest saved system model weights file. |
| `sessions/my_run/checkpoint_last.pt` | Optimization states for complete session training recovery. |
| `sessions/my_run/metrics.csv` | Comprehensive training logging metrics (loss histories, LR schedules, rank properties). |
| `sessions/my_run/dashboard.html` | Self-contained interactive Plotly diagnostic dashboard, generated by `model.generate_dashboard()` or `scripts/session_to_dash.py`. |
| `sessions/my_run/results/predict_latent_vectors_full.npy` | Dense computed coordinate latent atlas map array `(C, H, W)`. |
| `sessions/my_run/results/predict_pca_xyz.npy` / `predict_umap_xyz.npy` | PCA and UMAP coordinate maps registered to physical field coordinates. |

---

## ⚙️ Hyperparameter Knobs

The default `ScaleAwareJEPA()` pipeline tracks these baseline starting targets.
The public examples intentionally override the spread weight to `5` to reproduce
the gen_186 MHD run 2 prototype.

**Training Settings**

- `epochs`: `10`
- `batch_size`: `4`
- `gradient_accumulation_steps`: `1` in the base YAML and CLI path (effective optimizer batch = `batch_size × accumulation_steps`).
- The API `fit()` helper wraps this with OOM-safe auto-scaling and defaults to a target effective batch of `32`, so it may set `gradient_accumulation_steps` automatically.
- The per-step sample count for the JEPA loss and pooled spread regularizer is `batch_size × N_targets`, where `N_targets` is determined automatically from the image size and mask geometry.
- **Optimization Details:** AdamW optimizer engine, starting base learning rate `1e-4` (min `1e-6`), 1-epoch warm-up phase, weight decay penalty evaluated at `1e-5`.
- **EMA Schedules:** Targets update along a momentum gradient tracking scale of `0.99 → 0.9999`.

**Loss Components**

- `prediction_loss_weight`: `50` (primary JEPA latent prediction MSE multiplier).
- `spread_regularizer`: configured as `std_hinge`, with a scaling `weight: 2`, mapping against `target: context` inside a `"pooled"` spatial mode.
- `symmetry_loss_weight`: `0` (off by default; set to `0.003` for weak four-view flip consistency).
- `normalize_loss_l2`: `false` (preserves exact latent spatial amplitude calculations).

**Modeling Dimensions**

- Latent width: `32` total dense channels.
- Encoder backbones: 4 sequenced ConvNeXt blocks using an internal base width configuration of `64`.
- Projector blocks: maps projections through a `32 → 96 → 32` bottleneck transformation.
- Predictor blocks: latent spatial translation layer tracking a default width of `96`.
- **Dilations** — `[1, 1, 1, 1]` standard; `[1, 1, 2, 4]` for larger fields.

  **Receptive Field Formula:**
  
  `1 + Σ (kernel−1) × d_i`

  *For kernel=7:* `25 px` with `[1,1,1,1]` and `49 px` with `[1,1,2,4]`.
  The full encoder adds a 3×3 adapter and 3×3 stem (≈ +4 px additional);
  with GRN enabled the strict dependency is global across the feature map.

> OOM-safe auto-scaling (`fit()` only): on CUDA OOM, `fit()` halves
> the batch size and increases gradient accumulation to preserve the
> target effective batch.  The `train()` method and CLI do not yet
> support this.

---

## 💻 API Reference

| Python Method Interface | Returns | Purpose Description |
|:---|:---|:---|
| `ScaleAwareJEPA(config=None)` | model | Initializes a pipeline from default settings, a dictionary, or configuration paths. |
| `model.fit(field, epochs, ...)` | self | Executes training pipelines over a designated 2D physical target array. |
| `model.train(configs=None, ...)` | self | Orchestrates structured, YAML config-driven baseline training scenarios. |
| `model.extract(field)` | `(C, H, W)` | Returns the dense latent map from the current session's inference artifacts. |
| `model.project(field, method="umap")` | dict | Projects the current session's latent map with PCA and best-effort UMAP, falling back to PCA-only if UMAP is unavailable. |
| `model.infer_npy(path, **kwargs)` | string | Runs direct automated forward passes on a target `.npy` layout file path. |
| `model.analyze_rank()` | dict | Evaluates structural properties (effective manifold rank, dead channel screens). |
| `model.save_session(path)` | — | Copies the core saved session files and writes the active API config. |
| `model.generate_dashboard(path=None)` | — | Compiles an interactive visualization HTML file. |
| `model.open_dashboard()` | string | Launches the active session's tracking dashboard directly inside a system browser. |
| `model.open_interactive_umap()` | string | Launches an interactive click-to-similarity diagnostic UMAP session web tool. |

---

## ⌨️ Command Line Utility & Diagnostics

```bash
# Execute structured multi-scale model training pipelines via terminal CLI profiles
sajepa-train --config configs/base_pyramid_scaleaware_convnext.yaml --sessions-dir sessions

# Audit active latent spaces to search for systemic channel collapse and calculate manifold summaries
python scripts/print_session_summary.py sessions/gen_*

# Launch your structural interactive Plotly analytics dashboard
python scripts/session_to_dash.py --sessions-dir sessions --stage all --export-dir results/dashboard

# Execute a sliding-window tiled inference workflow on very large out-of-core fields
python -m src.inference_from_session \
  --session sessions/mhd_run_01 \
  --input data/large_field.npy \
  --crop-size 256 \
  --crop-mode tile

# Optional flip TTA can be enabled explicitly when desired:
#   --tta --tta-mode flip4
```

---

## 📁 Repository Layout

<details><summary>Click to expand file tree</summary>

```text
.
├── configs/
│   └── base_pyramid_scaleaware_convnext.yaml   # Canonical training configuration profile
├── examples/
│   ├── quickstart.py                            # Basic programmatic entry validation script
│   ├── example_mhd_inline.py                    # Annotated script using inline variables
│   ├── example_config_driven.py                 # Config-override API training example
│   └── example_cli.sh                           # Shell example for CLI-driven runs
├── scripts/
│   ├── train.py                                 # Core training terminal application interface
│   ├── print_session_summary.py                 # Post-run evaluation summary calculator
│   ├── session_to_dash.py                       # Exporter script managing Plotly layouts
│   ├── session_to_movie.py                      # Converts saved movie frames into latent-space movies
│   └── session_to_plots.py                      # Exports static publication-ready vector figures
├── src/
│   ├── api.py                                   # Main developer ScaleAwareJEPA interface endpoint
│   ├── losses.py                                # Objective-loss configurations and spread metrics
│   ├── models/
│   │   ├── encoders.py                          # Scale-aware ConvNeXt structural backbones
│   │   ├── masking.py                           # Scale-informed matrix mask builders
│   │   └── predictor.py                         # Joint-Embedding spatial prediction layers
│   └── utils/
│       └── memory.py                            # Resilient auto-scaling CUDA OOM handlers
└── tests/
```

</details>

---

## 📜 Citations & References

If you apply this Multi-Scale Constrained Diffusion Decomposition engine
within formal academic research pipelines, please attribute credit via the
citation record provided below:

```bibtex
@article{li2022constrained,
  author  = {Li, Guang-Xing},
  title   = {Multiscale decomposition of astronomical maps: A constrained diffusion method},
  journal = {The Astrophysical Journal Supplement Series},
  volume  = {259},
  number  = {2},
  pages   = {59},
  year    = {2022},
  doi     = {10.3847/1538-4365/ac4bc4}
}
```

---

## 📜 License

This package is open-source software distributed under the terms of the
MIT License.
