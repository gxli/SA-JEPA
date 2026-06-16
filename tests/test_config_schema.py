from __future__ import annotations

import pytest

from src.train import load_config


def test_removed_alias_sections_are_rejected(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "\n".join(
            [
                "data: {}",
                "model: {}",
                "train: {}",
                "masking:",
                "  mask_size_scaling: 1.2",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Removed config alias sections"):
        load_config(str(cfg))


def test_base_then_override_keeps_explicit_model_value(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "\n".join(
            [
                "data: {}",
                "model:",
                "  cdd_constrained: true",
                "  cdd_mode: log",
                "train: {}",
            ]
        ),
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "\n".join(
            [
                f"base_config: {base}",
                "model:",
                "  cdd_constrained: false",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(child))

    assert cfg["model"]["cdd_constrained"] is False
    assert cfg["model"]["cdd_mode"] == "log"
