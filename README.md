````markdown
# Multiscale-JEPA: Scale-Informed Masking for Self-Supervised Structure Discovery in Physical Fields

**Multiscale-JEPA** is a self-supervised framework for learning dense latent
coordinates of continuous physical fields. The goal is not classification or
pixel reconstruction, but **structure discovery**: learning an embedding whose
neighborhoods can be mapped back to coherent spatial morphology in the original
field.

The central idea is that masking geometry in physical fields should be tied to
the physical scale hierarchy of the data, rather than chosen as an arbitrary
fixed box in image space. Multiscale-JEPA implements this by combining:

1. **Constrained Diffusion Decomposition (CDD)**  
   Converts a scalar field into pixel-registered, scale-separated components.

2. **Scale-aware masking**  
   Masks each CDD component with a footprint proportional to its diffusion
   scale, so the prediction task is posed separately at fine, intermediate, and
   coarse levels.

3. **Joint-Embedding Predictive Architecture (JEPA)**  
   Predicts projected target representations from projected context
   representations, without reconstructing pixels.

4. **Dense latent analysis**  
   Applies the frozen target branch densely to the full field, then visualizes
   the resulting latent map with PCA/UMAP and maps latent neighborhoods back to
   spatial locations.

---

## Science Motivation

Physical fields are intrinsically multiscale. Turbulent density fields contain
diffuse regions, filaments, shocks, and compact structures; molecular-gas maps
contain spiral arms, interarm regions, dense complexes, and extended envelopes;
nighttime-light maps contain compact bright cores, corridors, and diffuse urban
structure.

A fixed image-space mask probes only one arbitrary pixel scale. Multiscale-JEPA
instead ties the mask footprint to a physical scale coordinate supplied by CDD.
This makes the context--target prediction task scale-aware: the model must infer
hidden structure at the scale represented by each CDD component.

The learned embedding is therefore used as an exploratory coordinate system for
the field. Latent neighborhoods are interpreted as **structural hypotheses**,
not as supervised classes or validated physical phases.

---

## Method Overview

<!-- TODO: replace with paper figure -->
![Multiscale-JEPA architecture placeholder](docs/figures/framework_placeholder.png)

**Figure placeholder:** CDD-scale-aware JEPA architecture.  
Suggested file: `docs/figures/framework.png`

The training path is:

```text
target field
    → target encoder
    → EMA target projector
    → projected target map q_t

masked context field
    → context encoder
    → online projector
    → projected context map q_c
    → predictor
    → predicted target map q_hat_t

loss = MSE(q_hat_t, q_t) at hidden target patches
     + weak spread regularizer on projected context patches
     + optional weak symmetry loss for MHD
````

The spread regularizer is applied **after the online projector and before the
predictor**, so it constrains the projected representation used as predictor
input, not the predictor output.

At inference, masking is disabled and the frozen target branch is applied
densely to the full field. The resulting projected latent map is used for
PCA/UMAP visualization and spatial back-mapping.

---

## Scale-Informed Masking

<!-- TODO: replace with masking demo from paper -->

![CDD pyramid masking placeholder](docs/figures/masking_demo_placeholder.png)

**Figure placeholder:** CDD decomposition and pyramid masking.
Suggested file: `docs/figures/masking_demo.png`

For CDD channel `s`, the nominal mask footprint is

```text
B_s_nom = sigma_s * f_mask + B_0
```

where:

* `sigma_s` is the CDD diffusion scale,
* `f_mask` is the scale multiplier,
* `B_0` is a fixed image-space offset.

The actual footprint is lower-bounded by the central target patch:

```text
B_s = oddceil(max(3, B_s_nom))
```

so the context mask always covers the central `3 × 3` prediction target.

Setting `B_0 = 0` gives a pure scale-tied pyramid mask. Setting
`f_mask = 0` gives a fixed-box mask. Fixed-box and randomized masks are used as
controls in the masking sweeps.

---

## CDD Frontend

<!-- TODO: replace with CDD vs wavelet figure -->

![CDD versus wavelet placeholder](docs/figures/cdd_vs_wavelet_placeholder.png)

**Figure placeholder:** CDD versus matched wavelet decomposition.
Suggested file: `docs/figures/cdd_vs_wavelet.png`

The selected runs use four CDD diffusion scales:

```text
[2, 4, 8, 16]
```

The unresolved residual is folded into the last scale channel, so the encoder
receives four CDD channels total. The last channel contains the coarsest
component plus residual structure not represented by the preceding scale bands.

CDD is used because it provides localized, pixel-registered scale components
that align naturally with scale-tied masking.

---

## Example Results

### MHD turbulence

<!-- TODO: replace with MHD latent map -->

![MHD dense latent map placeholder](docs/figures/mhd_latent_placeholder.png)

**Figure placeholder:** PCA/UMAP dense latent maps for the MHD density field.
Suggested file: `docs/figures/jepa_mhd.png`

<!-- TODO: replace with MHD back-mapping -->

![MHD back-mapping placeholder](docs/figures/mhd_backmapping_placeholder.png)

**Figure placeholder:** latent-neighborhood back-mapping for MHD.
Suggested file: `docs/figures/mhd_backmapping.png`

### Nighttime lights

<!-- TODO: replace with Chengdu latent map -->

![Chengdu dense latent map placeholder](docs/figures/chengdu_latent_placeholder.png)

**Figure placeholder:** dense latent map for Chengdu nighttime lights.
Suggested file: `docs/figures/jepa_chengdu.png`

### Molecular gas

<!-- TODO: replace with NGC latent map -->

![NGC dense latent map placeholder](docs/figures/ngc_latent_placeholder.png)

**Figure placeholder:** dense latent map for NGC 3627 molecular gas.
Suggested file: `docs/figures/jepa_ngc.png`

---

## Repository Layout

```text
convnext_jepa/
├── configs/                  # Training and inference configs
├── data/                     # Local input data; not tracked by git
├── docs/
│   └── figures/              # README and paper figures
├── results/                  # Generated plots and reports
├── sessions/                 # Per-run checkpoints and artifacts
├── scripts/                  # Plotting, diagnostics, and demo scripts
├── src/
│   ├── dataset.py            # Dataset and CDD-aware loading
│   ├── train.py              # Training loop
│   ├── inference_from_session.py
│   ├── models/               # Scale-aware ConvNeXt and JEPA modules
│   └── utils/
├── main.py                   # Single-config training entry point
├── run.sh                    # Batch launcher over configs
└── requirements.txt
```

---

## Installation

```bash
git clone <repo-url>
cd convnext_jepa

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

CUDA is recommended for training. CPU and Apple MPS may work for smaller
inference runs, but large CDD precomputation and training are GPU-oriented.

---

## Training

Run a single configuration:

```bash
python main.py \
    --config configs/example_mhd_ms1p2.json \
    --sessions-dir sessions
```

Run all configs in `configs/`:

```bash
./run.sh
```

Each run writes a session directory:

```text
sessions/<session_name>/
├── config_used.json
├── metrics.csv
├── model_last.pt
├── inference_outputs.pt
├── pred_map.npz
├── gt_map.npz
├── context_map.npz
└── dashboard / diagnostic artifacts
```

---

## Inference

Run inference from a trained session on a new `.npy` field:

```bash
python -m src.inference_from_session \
    --session sessions/<trained_session> \
    --input data/example_field.npy \
    --output-session sessions/inference_example \
    --tta \
    --tta-mode flip4
```

For large images, use tiled inference:

```bash
python -m src.inference_from_session \
    --session sessions/<trained_session> \
    --input data/large_field.npy \
    --crop-size 256 \
    --crop-mode tile \
    --output-session sessions/inference_large_tiled
```

For 3D volumes, process slices independently:

```bash
python -m src.inference_from_session \
    --session sessions/<trained_session> \
    --input data/volume.npy \
    --mode 3d_slab \
    --slice-axis 0 \
    --output-session sessions/inference_volume
```

---

## Plotting and Diagnostics

Generate loss curves and session plots:

```bash
python scripts/session_to_plots.py \
    --sessions-dir sessions \
    --results-dir results
```

Print effective-rank diagnostics for a group of sessions:

```bash
python scripts/print_effective_rank.py sessions/gen_*
```

Typical reported diagnostics include:

* effective rank,
* context / predictor / target rank,
* target participation,
* top eigenvalue fraction,
* dead-channel count,
* hinge-loss ratio,
* final loss terms.

These are used as **screening diagnostics** for collapse and latent-space usage.
Final selection of visualizations is based on dense latent morphology and
spatial back-mapping, not scalar diagnostics alone.

---

## Masking Demo

Generate a CDD-aware masking demo using the same masking pipeline as training:

```bash
python scripts/masking_demo.py \
    --config configs/example_mhd_ms1p2.json \
    --sessions-dir sessions \
    --sample-index 0 \
    --seed 42
```

Outputs:

```text
sessions/<config_name>/
├── masking_demo.png
└── masking_demo_meta.json
```

---

## CDD / Decomposition Demo

Generate decomposition and channel maps:

```bash
python scripts/blur_demo.py \
    --config configs/example_mhd_ms1p2.json \
    --sessions-dir sessions
```

Outputs may include:

```text
sessions/<config_name>/
├── blur_demo.png
├── blur_demo_channels.png
├── cdd_result.npy
├── cdd_residual.npy
└── blur_demo_meta.json
```

---

## Reproducing the Paper Runs

The paper uses selected pyramid-mask runs with scale multiplier:

```text
f_mask = 1.2
```

for the main MHD, Chengdu, and NGC visualizations.

Default selected settings include:

```text
CDD scales:              [2, 4, 8, 16]
residual handling:       folded into last CDD channel
encoder:                 dense scale-aware ConvNeXt
encoder depth:           4
ConvNeXt dilations:      [1, 1, 2, 4]
latent channels:         32
projected channels:      96
predictor hidden width:  96
mask hard cap:           48 px
prediction loss:         unnormalized MSE in projected latent space
spread regularizer:      std-hinge on projected context patches
target patch:            central 3 × 3
TTA:                     flip4
```

Example run pattern:

```bash
python main.py \
    --config configs/paper/mhd_ms1p2.json \
    --sessions-dir sessions

python -m src.inference_from_session \
    --config configs/inference/mhd_ms1p2.json
```

---

## Data

This repository does not include large scientific datasets by default.

Expected input format:

* `.npy` scalar fields for 2D images,
* `.npy` volumes for 3D slab inference,
* optionally `.fits` files if FITS support is enabled in the local environment.

Place local data under:

```text
data/
```

and update the corresponding config entries.

---

## Citation

If you use this code, please cite the associated paper:

```bibtex
@misc{li2026multiscalejepa,
  title  = {Scale-Informed Masking for Self-Supervised Structure Discovery in Physical Fields},
  author = {Li, Guang-Xing},
  year   = {2026},
  note   = {Preprint}
}
```

CDD is described in:

```bibtex
@article{li2022constrained,
  title={Multi-scale Decomposition of Astronomical Maps: A Constrained Diffusion Method},
  author={Li, Guang-Xing},
  journal={The Astrophysical Journal Supplement Series},
  volume={258},
  number={2},
  pages={44},
  year={2022},
  doi={10.3847/1538-4365/ac4bc4}
}
```

---

## License

Add license information here.

Recommended:

```text
MIT License for code.
Separate data licenses for external scientific datasets.
```

---

## Contact

For questions, open an issue or contact the maintainer.

```
```
