"""Tests for collapse monitor."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_embedding_std_zero_on_constant():
    from src.routing.text_jepa.collapse import embedding_std

    x = torch.ones(4, 8, 128)
    assert embedding_std(x) == pytest.approx(0.0)


def test_embedding_std_positive_on_random():
    from src.routing.text_jepa.collapse import embedding_std

    x = torch.randn(4, 8, 128)
    assert embedding_std(x) > 0.5


def test_collapse_monitor_triggers_after_threshold():
    from src.routing.text_jepa.collapse import CollapseMonitor

    mon = CollapseMonitor(floor=0.01, patience=2)
    assert mon.check(std=0.5) is False
    assert mon.check(std=0.005) is False  # 1 strike
    assert mon.check(std=0.003) is True  # 2 strikes → collapse


def test_collapse_monitor_resets_on_recovery():
    from src.routing.text_jepa.collapse import CollapseMonitor

    mon = CollapseMonitor(floor=0.01, patience=2)
    mon.check(std=0.005)  # 1 strike
    mon.check(std=0.5)  # recover
    assert mon.check(std=0.005) is False  # 1 strike again, not 2
