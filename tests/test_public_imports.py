from __future__ import annotations

import json


def test_public_and_source_import_routes_match():
    from sajepa import ScaleAwareJEPA as PublicScaleAwareJEPA
    from src.api import ScaleAwareJEPA as SourceScaleAwareJEPA

    assert PublicScaleAwareJEPA is SourceScaleAwareJEPA


def test_api_file_config_is_default_plus_override(tmp_path):
    from sajepa import ScaleAwareJEPA

    cfg_path = tmp_path / "override.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "model:",
                "  mask_size_scaling: 1.2",
                "train:",
                "  epochs: 3",
            ]
        ),
        encoding="utf-8",
    )

    model = ScaleAwareJEPA(config=str(cfg_path))

    assert model._config["model"]["mask_size_scaling"] == 1.2
    assert model._config["train"]["epochs"] == 3
    assert "sigmas" in model._config["model"]
    assert "data_root" in model._config["data"]


def test_load_session_uses_saved_config_exactly(tmp_path):
    from sajepa import ScaleAwareJEPA

    session = tmp_path / "session"
    session.mkdir()
    (session / "config_used.json").write_text(
        json.dumps({"data": {}, "model": {"mask_size_scaling": 1.2}, "train": {}}),
        encoding="utf-8",
    )

    model = ScaleAwareJEPA.load_session(str(session))

    assert model._config["model"] == {"mask_size_scaling": 1.2}


def test_api_train_wrapper_uses_default_plus_overrides(monkeypatch, tmp_path):
    import src.api as api
    from sajepa import ScaleAwareJEPA

    captured = {}

    def fake_run_training(cfg, config_name, sessions_root):
        captured["cfg"] = cfg
        captured["config_name"] = config_name
        captured["sessions_root"] = sessions_root
        session = tmp_path / config_name
        session.mkdir()
        return str(session)

    monkeypatch.setattr(api, "run_training", fake_run_training)

    model = ScaleAwareJEPA().train(
        configs={"model": {"mask_size_scaling": 1.2}, "train": {"epochs": 3}},
        config_name="demo",
        sessions_dir=str(tmp_path),
    )

    assert model.session_dir == str((tmp_path / "demo").resolve())
    assert captured["cfg"]["model"]["mask_size_scaling"] == 1.2
    assert captured["cfg"]["train"]["epochs"] == 3
    assert "sigmas" in captured["cfg"]["model"]


def test_api_train_base_session_defaults_to_weights_only(monkeypatch, tmp_path):
    import src.api as api
    from sajepa import ScaleAwareJEPA

    base = tmp_path / "base"
    base.mkdir()
    (base / "model_last.pt").write_bytes(b"model")
    (base / "checkpoint_last.pt").write_bytes(b"checkpoint")

    def fake_run_training(cfg, config_name, sessions_root):
        session = tmp_path / "new"
        assert (session / "model_last.pt").read_bytes() == b"model"
        assert not (session / "checkpoint_last.pt").exists()
        return str(session)

    monkeypatch.setattr(api, "run_training", fake_run_training)

    ScaleAwareJEPA().train(
        config_name="new",
        sessions_dir=str(tmp_path),
        base_session=str(base),
    )


def test_api_train_base_session_resume_mode_copies_full_checkpoint(monkeypatch, tmp_path):
    import src.api as api
    from sajepa import ScaleAwareJEPA

    base = tmp_path / "base"
    base.mkdir()
    (base / "model_last.pt").write_bytes(b"model")
    (base / "checkpoint_last.pt").write_bytes(b"checkpoint")

    def fake_run_training(cfg, config_name, sessions_root):
        session = tmp_path / "new"
        assert (session / "model_last.pt").read_bytes() == b"model"
        assert (session / "checkpoint_last.pt").read_bytes() == b"checkpoint"
        return str(session)

    monkeypatch.setattr(api, "run_training", fake_run_training)

    ScaleAwareJEPA().train(
        config_name="new",
        sessions_dir=str(tmp_path),
        base_session=str(base),
        base_session_mode="resume",
    )


def test_api_dict_config_rejects_removed_aliases():
    import pytest
    from sajepa import ScaleAwareJEPA

    with pytest.raises(ValueError, match="Removed config alias sections"):
        ScaleAwareJEPA(config={"masking": {"mask_size_scaling": 1.2}})


def test_api_base_session_requires_checkpoint(tmp_path):
    import pytest
    from sajepa import ScaleAwareJEPA

    base = tmp_path / "empty"
    base.mkdir()

    with pytest.raises(FileNotFoundError, match="no model_last.pt"):
        ScaleAwareJEPA().train(
            config_name="new",
            sessions_dir=str(tmp_path),
            base_session=str(base),
        )


def test_api_base_session_mode_is_validated(tmp_path):
    import pytest
    from sajepa import ScaleAwareJEPA

    base = tmp_path / "base"
    base.mkdir()
    (base / "model_last.pt").write_bytes(b"model")

    with pytest.raises(ValueError, match="base_session_mode"):
        ScaleAwareJEPA().train(
            config_name="new",
            sessions_dir=str(tmp_path),
            base_session=str(base),
            base_session_mode="bad",
        )
