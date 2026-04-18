"""Tests for the per-module forgetting gate (post-pivot sweep feedback).

Unit tests cover :func:`src.eval.forgetting.apply_per_module_gate`; a
live-matrix test applies the gate against ``results/forgetting-matrix.json``
and asserts the empirical invariant: with ``mlp.shared_expert_gate``
ignored every pair passes, and without the ignore at least one pair
flags the rank-1 canary.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.eval.forgetting import (
    DEFAULT_PER_MODULE_IGNORE,
    PerModuleGateDecision,
    apply_per_module_gate,
)


# ---------------------------------------------------------------------------
# Unit behaviour
# ---------------------------------------------------------------------------


def test_per_module_gate_all_above() -> None:
    """Every module comfortably above 30° → pass, no offenders."""
    decision = apply_per_module_gate(
        {"mod1": 45.0, "mod2": 60.0},
        winrate_drop=0.1,
        ignore_modules=set(),
    )
    assert isinstance(decision, PerModuleGateDecision)
    assert decision.failed is False
    assert decision.offending_modules == []
    assert decision.min_angle_module == "mod1"
    assert decision.min_angle_value == pytest.approx(45.0)


def test_per_module_gate_one_below() -> None:
    """One module below 30° + drop above threshold → fail, offenders listed."""
    decision = apply_per_module_gate(
        {"mod1": 20.0, "mod2": 60.0},
        winrate_drop=0.1,
        ignore_modules=set(),
    )
    assert decision.failed is True
    assert decision.offending_modules == ["mod1"]
    assert decision.min_angle_module == "mod1"
    assert decision.min_angle_value == pytest.approx(20.0)


def test_per_module_gate_ignored_module() -> None:
    """Ignored canary below threshold is excluded from the decision.

    ``ignored_canary`` would fail alone (10° < 30°), but with it in
    ``ignore_modules`` only ``mod1`` is considered. ``mod1=45°`` is
    safe, so the gate passes and no offenders are flagged.
    """
    decision = apply_per_module_gate(
        {"mod1": 45.0, "ignored_canary": 10.0},
        winrate_drop=0.1,
        ignore_modules={"ignored_canary"},
    )
    assert decision.failed is False
    assert decision.offending_modules == []
    assert decision.min_angle_module == "mod1"
    assert decision.min_angle_value == pytest.approx(45.0)


def test_per_module_gate_ignored_hides_low_module_but_real_offender_still_fails() -> None:
    """Ignoring the canary does not rescue a genuine per-module offender."""
    decision = apply_per_module_gate(
        {"mod1": 20.0, "ignored_canary": 10.0, "mod2": 55.0},
        winrate_drop=0.1,
        ignore_modules={"ignored_canary"},
    )
    assert decision.failed is True
    assert decision.offending_modules == ["mod1"]
    # ignored_canary MUST NOT appear as min module.
    assert decision.min_angle_module == "mod1"
    assert decision.min_angle_value == pytest.approx(20.0)


def test_per_module_gate_no_winrate() -> None:
    """Partial / angle-only mode: never fail, but still list offenders."""
    decision = apply_per_module_gate(
        {"mod1": 20.0, "mod2": 55.0},
        winrate_drop=None,
        ignore_modules=set(),
    )
    assert decision.failed is False
    # Offenders still surfaced informationally.
    assert decision.offending_modules == ["mod1"]
    assert decision.min_angle_module == "mod1"
    assert decision.min_angle_value == pytest.approx(20.0)


def test_per_module_gate_angle_below_but_drop_safe() -> None:
    """AND-logic: low angle alone is not enough — needs drop too."""
    decision = apply_per_module_gate(
        {"mod1": 15.0, "mod2": 55.0},
        winrate_drop=0.01,  # below 0.03 threshold
        ignore_modules=set(),
    )
    assert decision.failed is False
    # Offender still listed — the gate only passes because of the drop.
    assert decision.offending_modules == ["mod1"]


def test_per_module_gate_default_ignores_shared_expert_gate() -> None:
    """Default ``ignore_modules`` masks ``mlp.shared_expert_gate``."""
    decision = apply_per_module_gate(
        {"mlp.shared_expert_gate": 12.0, "self_attn.q_proj": 80.0},
        winrate_drop=0.5,
    )
    assert decision.failed is False
    assert decision.offending_modules == []
    assert decision.min_angle_module == "self_attn.q_proj"
    assert "mlp.shared_expert_gate" in DEFAULT_PER_MODULE_IGNORE


def test_per_module_gate_empty() -> None:
    """Empty input: never fail, min_angle_value is NaN."""
    decision = apply_per_module_gate({}, winrate_drop=0.5, ignore_modules=set())
    assert decision.failed is False
    assert decision.offending_modules == []
    assert decision.min_angle_module == ""
    assert math.isnan(decision.min_angle_value)


def test_per_module_gate_all_ignored() -> None:
    """All input modules ignored: behaves like empty input."""
    decision = apply_per_module_gate(
        {"a": 10.0, "b": 20.0},
        winrate_drop=0.5,
        ignore_modules={"a", "b"},
    )
    assert decision.failed is False
    assert decision.offending_modules == []
    assert decision.min_angle_module == ""
    assert math.isnan(decision.min_angle_value)


def test_per_module_gate_thresholds_overridable() -> None:
    """Caller can tighten the gate by supplying custom thresholds."""
    decision = apply_per_module_gate(
        {"mod1": 40.0},
        winrate_drop=0.05,
        angle_threshold=50.0,  # tighter than default 30
        winrate_drop_threshold=0.03,
        ignore_modules=set(),
    )
    assert decision.failed is True
    assert decision.offending_modules == ["mod1"]


# ---------------------------------------------------------------------------
# Live matrix: apply the gate to the committed sweep output
# ---------------------------------------------------------------------------


_MATRIX_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "forgetting-matrix.json"
)


def _load_matrix() -> dict:
    if not _MATRIX_PATH.is_file():
        pytest.skip(f"forgetting matrix not found at {_MATRIX_PATH}")
    return json.loads(_MATRIX_PATH.read_text(encoding="utf-8"))


def test_live_matrix_passes_per_module_with_shared_expert_gate_ignored() -> None:
    """Post-pivot sweep: every pair passes the per-module gate once the
    rank-1 ``mlp.shared_expert_gate`` canary is excluded.

    This is the structural invariant the ignore list is designed to
    enforce — if this assertion starts failing, a NEW per-module
    regression has been introduced.
    """
    matrix = _load_matrix()
    pairs = matrix["pairs"]
    assert len(pairs) >= 2, "matrix must contain at least one pair"

    for pair in pairs:
        angles = pair["angle_degrees_per_module"]
        # Use the angle-only partial mode (drop=None): the assertion is
        # that no non-ignored module drops below 30°, independent of
        # win-rate (we don't have win-rate for these sweeps).
        decision = apply_per_module_gate(angles, winrate_drop=None)
        # With shared_expert_gate ignored every pair should have an
        # empty offender list.
        assert decision.offending_modules == [], (
            f"pair {pair['prior']} -> {pair['new']} surfaces "
            f"per-module offenders even with shared_expert_gate ignored: "
            f"{decision.offending_modules} (min "
            f"{decision.min_angle_module}={decision.min_angle_value:.2f}°)"
        )
        # And the non-ignored minimum should stay above the threshold.
        assert decision.min_angle_value >= 30.0


def test_live_matrix_flags_shared_expert_gate_without_ignore() -> None:
    """Post-pivot sweep: at least one pair surfaces
    ``mlp.shared_expert_gate`` as an offender when nothing is ignored.

    Documents the empirical finding (python↔typescript 17.3°) and
    guards against the ignore list silently masking a real regression
    elsewhere.
    """
    matrix = _load_matrix()
    pairs = matrix["pairs"]

    flagged = 0
    for pair in pairs:
        angles = pair["angle_degrees_per_module"]
        decision = apply_per_module_gate(
            angles,
            winrate_drop=None,
            ignore_modules=set(),
        )
        if "mlp.shared_expert_gate" in decision.offending_modules:
            flagged += 1
    assert flagged >= 1, (
        "expected >=1 pair to surface mlp.shared_expert_gate as an "
        "offender when ignore_modules is empty; sweep found none — "
        "either the matrix has changed or the canary constant is stale."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
