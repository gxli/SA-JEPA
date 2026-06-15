"""Minimal public-API smoke: train from defaults and write a dashboard."""

from __future__ import annotations

import torch

from sajepa import ScaleAwareJEPA


field = torch.rand(64, 64)
model = ScaleAwareJEPA()
model.fit(field, epochs=1)
model.generate_dashboard("results/api_dashboard_smoke.html")
