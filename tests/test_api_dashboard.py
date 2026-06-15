from __future__ import annotations

import torch


def test_public_api_generates_dashboard_from_import_route(tmp_path):
    from sajepa import ScaleAwareJEPA

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    torch.save({"context_map": torch.randn(1, 4, 8, 8)}, session_dir / "inference_outputs.pt")

    model = ScaleAwareJEPA()
    model._session_dir = str(session_dir)
    model._is_trained = True

    out_html = tmp_path / "dashboard.html"
    model.generate_dashboard(str(out_html))

    assert out_html.exists()
    assert "<html" in out_html.read_text(encoding="utf-8").lower()
