"""Integration-test gating: skip unless on Apple Silicon macOS (MLX required)."""
from __future__ import annotations

import platform

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip integration tests on non-MLX hosts (Linux CI, Intel macOS)."""
    if "integration" in str(item.fspath):
        if platform.machine() != "arm64" or platform.system() != "Darwin":
            pytest.skip("integration tests require Apple Silicon macOS (MLX)")
