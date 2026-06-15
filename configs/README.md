# Configuration Baselines

Use `base_pyramid_scaleaware_convnext.yaml` as the canonical 2D MHD baseline.
It keeps the production defaults intentionally small:

```yaml
model:
  normalize_loss_l2: false

training:
  prediction_loss_weight: 50
  symmetry_loss_weight: 0.003
  spread_regularizer:
    type: std_hinge
    target: context
    spatial_mode: dense
    weight: 2
    target_std: 1.0
    eps: 0.0001

data:
  d4_augment: true
```

Important conventions:

- `spread_regularizer.target` defaults to `context`. Use `predictor` or `both`
  only for explicit ablations.
- `spread_regularizer.spatial_mode` defaults to `dense`, matching the
  dense-token hinge behavior used in the selected MHD runs. Set `pooled` only
  for explicit ablations.
- `std_hinge` is the default anti-collapse loss. `weak_sigreg` and
  `sketched_sigreg` are ablation regularizers, not the main baseline.
- `prediction_loss_weight` and `symmetry_loss_weight` are the canonical training
  weight names.
- Reflect/circular random image shifting is not part of the baseline.
- VICReg weights are optional training terms. When they are zero, VICReg values
  are diagnostics only.

Keep `configs/experiments/` for current runnable sweeps. Move historical or
one-off generations to `configs_bk/`.
