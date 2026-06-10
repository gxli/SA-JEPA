# Reasonix project memory

Notes the user pinned via the `#` prompt prefix. The whole file is
loaded into the immutable system prefix every session — keep it terse.

- `sajepa` — Scale-Aware JEPA for Continuous Physical Fields

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

`sajepa` is a non-generative, self-supervised framework factory designed to extract **pixel-registered latent coordinates** from continuous physical fields (e.g., magnetohydrodynamic turbulence, fluid continuums, and multi-wavelength astronomical data). 

By coupling **Constrained Diffusion Decomposition (CDD)** with a **Joint-Embedding Predictive Architecture (JEPA)**, the network treats spatial masking as a sequence of scale-specific structural interventions. This forces the predictor to learn invariant physical transport operators natively from scratch without representation collapse or pixel reconstruction artifacts.

---

## 🏛️ Core Architecture

Unlike traditional vision world models that utilize discrete, object-centric bounding boxes or fixed-scale token masking, `sajepa` operates under a scale-space paradigm optimized for continuums where clear boundaries do not exist:

* **No Object Slots:** Replaces discrete entity tracking with continuous scale hierarchies.
* **Scale-Informed Masking:** Synthesizes masking footprints directly from the local diffusion scales of the field.
* **Glass-Box Constraints:** Leverages CDD to process scalar inputs as explicit, scale-separated components while retaining absolute pixel registration.

---

## ⚡ Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/sajepa.git](https://github.com/yourusername/sajepa.git)
cd sajepa

# Install in editable mode for local development
pip install -e ., those kept?
