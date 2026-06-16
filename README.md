# `sajepa` — Scale-Aware JEPA for Continuous Physical Fields

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

`sajepa` is a non-generative, self-supervised framework factory designed to
extract **pixel-registered latent coordinates** from continuous physical
fields (magnetohydrodynamic turbulence, fluid continuums, multi-wavelength
astronomical data).

By coupling **Constrained Diffusion Decomposition (CDD)** with a
**Joint-Embedding Predictive Architecture (JEPA)**, the network treats
spatial masking as a sequence of scale-specific structural interventions.
This forces the predictor to learn invariant physical transport operators
from scratch, without representation collapse or pixel reconstruction
artifacts.

---

## 🏛️ Core Architecture

Unlike traditional vision models that utilize discrete, object-centric
bounding boxes or fixed-scale token masking, `sajepa` operates under a
scale-space paradigm optimized for continuums where clear boundaries do
not exist:

* **No Object Slots** — Replaces discrete entity tracking with continuous scale hierarchies.
* **Scale-Informed Masking** — Synthesizes masking footprints directly from the local diffusion scales of the field.
* **Glass-Box Constraints** — CDD processes scalar inputs as explicit, scale-separated components while retaining absolute pixel registration.

---

## ⚡ Installation

```bash
git clone https://github.com/yourusername/sajepa.git
cd sajepa
pip install -e .
```

macOS Apple Silicon (MPS) is supported. Set `PYTORCH_ENABLE_MPS_FALLBACK=1`
if you encounter missing MPS ops (e.g. `avg_pool3d` in CDD).

---

## 🚀 Quick Start

### Python API

`sajepa` trains from scratch on arbitrary 2D scalar fields — no pre-trained
weights required.

```python
import torch
from sajepa import ScaleAwareJEPA

field = torch.load("data/ngc3627_co_emission.pt")  # (H, W)

# 1. Use the default scale-aware ConvNeXt baseline
model = ScaleAwareJEPA()

# 2. Train (resumes from checkpoint if session_dir exists)
model.fit(field, epochs=10)
# or with explicit session dir:
model.fit(field, epochs=10, session_dir="sessions/my_run")

# 3. Continue training (auto-resume)
model.fit(field, epochs=20, session_dir="sessions/my_run")

# 4. Extract pixel-registered latent atlas
latent = model.extract(field)                # → (C_latent, H, W)

# 5. Project to 2D (PCA + UMAP with GPU fallback)
proj = model.project(field, method="umap")   # → {"pca": np.array, "umap": np.array|None}

# 6. Diagnostics
metrics = model.analyze_rank()               # → dict: erank, energy, sim_r, hinge_r, ...

# 7. Persist and visualize
model.save_session("sessions/my_run")
model.generate_dashboard("results/api_dashboard.html")

# 8. Restore later
model2 = ScaleAwareJEPA.load_session("sessions/my_run")
latent2 = model2.extract(new_field)
```

| method | returns | purpose |
|--------|---------|---------|
| `ScaleAwareJEPA(config?)` | model | init from dict, YAML/JSON path, or defaults |
| `model.fit(field, epochs, *, session_dir?)` | self | train; resumes if *session_dir* exists |
| `model.train(configs?, *, config_name?, sessions_dir?, dashboard?)` | self | train from config with full pipeline |
| `model.extract(field)` | `(C,H,W)` tensor | pixel-registered latent atlas |
| `model.project(field, method="umap")` | `{"pca":..., "umap":...}` | 2D projection (PCA + UMAP) |
| `model.infer_npy(path, output_dir?, **kwargs)` | str | run inference on a new `.npy` file |
| `model.analyze_rank()` | dict | effective rank, sim, hinge, dead channels |
| `model.save_session(path)` | — | persist config, weights, inference, dashboard |
| `model.generate_dashboard(path?)` | — | interactive Plotly HTML dashboard |
| `ScaleAwareJEPA.load_session(path)` | model | restore from saved session |
| `ScaleAwareJEPA.infer_from_session(session, in, out)` | str | classmethod: inference without loading |

### Configurable Knobs

All knobs sit under `model.*` or `training.*` in config files. The default
`ScaleAwareJEPA()` baseline uses the values shown below.

| Section | Key | Default | Purpose |
|---|---|---|---|
| `training` | `epochs` | 10 | training duration |
| `training` | `batch_size` | 4 | per-step samples |
| `training` | `gradient_accumulation_steps` | 1 | effective batch = bs × accum |
| `model` | `convnext_layer_dilations` | `null` | dilation per ConvNeXt block; `[1,1,2,4]` for large FOV on big images |
| `model` | `mask_box_hardcap` | `null` | ceiling on mask box pixels; 48 for NGC-style 686×398 fields |
| `training.spread_regularizer` | `weight` | 2 | anti-collapse hinge strength |
| `training.spread_regularizer` | `spatial_mode` | `pooled` | `pooled` (per-patch mean) or `dense` (per-token) |
| `training.spread_regularizer` | `target` | `context` | which branch to regularise |
| `training` | `prediction_loss_weight` | 50 | MSE weight |
| `training` | `symmetry_loss_weight` | 0.003 | MHD D₄ equivariance; 0 for NGC/Chengdu (OOM-prone) |
| `model` | `normalize_loss_l2` | false | angular vs amplitude MSE |
| `model` | `use_symmetric_feature_loss` | true | disable on large non-square images |

The program prints the effective receptive field at startup when custom
dilations are set.  NGC-style runs should use `convnext_layer_dilations:
[1,1,2,4]` plus `mask_box_hardcap: 48` and `symmetry_loss_weight: 0`.

### CLI

```bash
sajepa-train --config configs/examples/mhd_2d_ms1p2.json --sessions-dir sessions
```

### Built-in Examples

```bash
python examples/api_dashboard_smoke.py    # Minimal public API + dashboard smoke
python examples/quickstart.py             # Synthetic 128×128, 3 epochs
python examples/test_mhd.py               # C12 MHD data, gen_139 config, 10 epochs
python examples/inspect_cdd_masking.py    # CDD channel + mask consistency check
python examples/test_cdd.py               # CDD standalone MPS debug
```

---

## 🛠️ Auto-Batch OOM Safeguards

On CUDA OOM, the training engine catches the error, halves batch size, and
scales gradient accumulation to maintain the target effective batch:

```yaml
# configs/mhd_turbulence.yaml
train:
  batch_size: 4
  target_batch_size: 32
  auto_scale_batch_size: "power_of_two"   # OOM → 4 → 2 → 1
  precision: "bf16"
```

---

## 📁 Repository Layout

```
.
├── configs/
│   ├── base_pyramid_scaleaware_convnext.json   # Default encoder backbone
│   └── examples/mhd_2d_ms1p2.json              # Example 2D MHD run config
├── examples/
│   ├── quickstart.py            # Minimal training + extraction
│   ├── test_mhd.py              # C12 MHD data training run
│   ├── inspect_cdd_masking.py  # CDD channel-mask scale check
│   └── test_cdd.py             # CDD standalone MPS debug
├── scripts/
│   ├── train.py                 # CLI entry point
│   ├── print_effective_rank.py  # Collapse + manifold diagnostics
│   ├── session_to_dash.py       # Interactive HTML dashboards
│   ├── session_to_movie.py      # Latent space optimization movies
│   └── session_to_plots.py      # Static PCA/UMAP plots
├── src/
│   ├── api.py                   # ScaleAwareJEPA interface
│   ├── losses.py                # JEPA loss + spread regularizers
│   ├── train.py                 # Training loop
│   ├── models/
│   │   ├── encoders.py          # Scale-aware ConvNeXt extractors
│   │   ├── masking.py           # CDD-footprint mask generation
│   │   └── predictor.py         # Spatial predictor
│   └── utils/
│       └── memory.py            # OOM-safe auto-batch handler
└── tests/
```

---

## 📊 Diagnostics & Latent Space Auditing

```bash
# Effective rank, hinge-loss ratios, dead channels
python scripts/print_effective_rank.py sessions/gen_*

# Interactive Plotly dashboard with UMAP embeddings
python scripts/session_to_dash.py --session sessions/mhd_run_01

# Static PCA/UMAP plots
python scripts/session_to_plots.py --session sessions/mhd_run_01 --results-dir results
```

UMAP runs GPU-native via `torchdr` when available (MPS/CUDA), falling back
to `umap-learn` on CPU.

---

## 🔍 Large-Field Tiled Inference

```bash
python -m src.inference_from_session \
    --session sessions/mhd_run_01 \
    --input data/large_field.npy \
    --crop-size 256 --crop-mode tile \
    --tta --tta-mode flip4
```

---

## 📜 References

```bibtex
@misc{li2026multiscalejepa,
  title  = {Scale-Informed Masking for Self-Supervised Structure Discovery in Physical Fields},
  author = {Li, Guang-Xing},
  year   = {2026},
}

@article{li2022constrained,
  title   = {Constrained Diffusion Decomposition},
  author  = {Li, Guang-Xing},
  journal = {ApJS},
  volume  = {258},
  pages   = {44},
  year    = {2022},
  doi     = {10.3847/1538-4365/ac4bc4}
}
```

---

## License

MIT
