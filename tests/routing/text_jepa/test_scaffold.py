"""Scaffolding smoke test — package is importable."""
from __future__ import annotations


def test_package_imports():
    import src.routing.text_jepa as tj  # noqa: F401
    assert hasattr(tj, "__all__")
