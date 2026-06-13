from __future__ import annotations


def test_public_and_source_import_routes_match():
    from sajepa import ScaleAwareJEPA as PublicScaleAwareJEPA
    from src.api import ScaleAwareJEPA as SourceScaleAwareJEPA

    assert PublicScaleAwareJEPA is SourceScaleAwareJEPA
