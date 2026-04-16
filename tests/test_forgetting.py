"""Tests for src.eval.forgetting — forgetting check framework (Story-16).

All tests are mocked: no real model, no GPU required.
Tests that need torch are marked and skipped if torch is unavailable.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.eval.forgetting import (
    ForgettingEvaluator,
    ForgettingReport,
    GradientSubspaceAnalyzer,
    check_forgetting,
    save_forgetting_report,
    ForgettingCheckResult,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# -----------------------------------------------------------------------
# GradientSubspaceAnalyzer.compute_angle (requires torch)
# -----------------------------------------------------------------------


@needs_torch
class TestComputeAngle:
    """Test SVD-based gradient subspace angle measurement."""

    def test_orthogonal_vectors_give_90_degrees(self):
        """Two orthogonal subspaces should yield ~90°."""
        a = torch.eye(768, 16)
        b = torch.zeros(768, 16)
        b[16:32, :] = torch.eye(16)
        analyzer = GradientSubspaceAnalyzer()
        angle = analyzer.compute_angle(a, b)
        assert angle > 80.0, f"Expected ~90°, got {angle:.1f}°"

    def test_parallel_vectors_give_0_degrees(self):
        """Identical subspaces should yield ~0°."""
        a = torch.randn(768, 16)
        analyzer = GradientSubspaceAnalyzer()
        angle = analyzer.compute_angle(a, a)
        assert angle < 1.0, f"Expected ~0°, got {angle:.1f}°"

    def test_45_degree_case(self):
        """A known 45° rotation in 2D embedded in higher dims."""
        a = torch.zeros(100, 1)
        a[0] = 1.0

        b = torch.zeros(100, 1)
        b[0] = math.cos(math.radians(45))
        b[1] = math.sin(math.radians(45))

        analyzer = GradientSubspaceAnalyzer()
        angle = analyzer.compute_angle(a, b)
        assert 40.0 < angle < 50.0, f"Expected ~45°, got {angle:.1f}°"


@needs_torch
class TestLegacyComputeSubspaceAngle:
    def test_orthogonal_subspaces(self):
        from src.eval.forgetting import compute_subspace_angle
        a = torch.eye(768, 16)
        b = torch.zeros(768, 16)
        b[16:32, :] = torch.eye(16)
        angle = compute_subspace_angle(a, b)
        assert angle > 80.0

    def test_identical_subspaces(self):
        from src.eval.forgetting import compute_subspace_angle
        a = torch.randn(768, 16)
        angle = compute_subspace_angle(a, a)
        assert angle < 1.0


# -----------------------------------------------------------------------
# ForgettingReport
# -----------------------------------------------------------------------


class TestForgettingReport:
    """Test the immutable ForgettingReport dataclass."""

    def test_is_frozen(self):
        report = ForgettingReport(
            stack_id="stack-01",
            new_stack_id="stack-02",
            angle=45.0,
            winrate_base=0.80,
            winrate_adapted=0.79,
            winrate_drop=0.01,
            passed=True,
            should_rollback=False,
        )
        with pytest.raises(AttributeError):
            report.angle = 99.0  # type: ignore[misc]

    def test_rollback_true_when_both_conditions_met(self):
        """angle=25 (<30) AND winrate_drop=0.06 (>0.03) -> should_rollback=True."""
        report = ForgettingReport(
            stack_id="s01", new_stack_id="s02",
            angle=25.0, winrate_base=0.80, winrate_adapted=0.74,
            winrate_drop=0.06, passed=False, should_rollback=True,
        )
        assert report.should_rollback is True
        assert report.passed is False

    def test_no_rollback_when_only_angle_bad(self):
        """Low angle but acceptable win-rate -> should_rollback=False."""
        report = ForgettingReport(
            stack_id="s01", new_stack_id="s02",
            angle=25.0, winrate_base=0.80, winrate_adapted=0.79,
            winrate_drop=0.01, passed=True, should_rollback=False,
        )
        assert report.should_rollback is False

    def test_no_rollback_when_only_winrate_bad(self):
        """High win-rate drop but safe angle -> should_rollback=False."""
        report = ForgettingReport(
            stack_id="s01", new_stack_id="s02",
            angle=45.0, winrate_base=0.80, winrate_adapted=0.74,
            winrate_drop=0.06, passed=True, should_rollback=False,
        )
        assert report.should_rollback is False


# -----------------------------------------------------------------------
# ForgettingEvaluator.check_stack
# -----------------------------------------------------------------------


class TestCheckStack:
    """Test single-stack forgetting check with mocked evaluator + analyzer."""

    def setup_method(self):
        self.mock_evaluator = MagicMock()
        self.mock_analyzer = GradientSubspaceAnalyzer()
        self.fe = ForgettingEvaluator(
            stack_evaluator=self.mock_evaluator,
            analyzer=self.mock_analyzer,
        )

    def test_rollback_triggered(self):
        """angle=25, wr_base=0.80, wr_adapted=0.74 -> should_rollback=True."""
        report = self.fe.check_stack(
            stack_id="stack-01",
            new_stack_id="stack-03",
            eval_data_path=Path("data/eval/stack-01.jsonl"),
            angle=25.0,
            winrate_base=0.80,
            winrate_adapted=0.74,
        )
        assert report.should_rollback is True
        assert report.passed is False
        assert report.winrate_drop == pytest.approx(0.06)
        assert report.angle == 25.0

    def test_pass_case(self):
        """angle=45, wr_base=0.80, wr_adapted=0.79 -> should_rollback=False."""
        report = self.fe.check_stack(
            stack_id="stack-01",
            new_stack_id="stack-03",
            eval_data_path=Path("data/eval/stack-01.jsonl"),
            angle=45.0,
            winrate_base=0.80,
            winrate_adapted=0.79,
        )
        assert report.should_rollback is False
        assert report.passed is True
        assert report.winrate_drop == pytest.approx(0.01)

    def test_low_angle_acceptable_winrate_no_rollback(self):
        """Edge case: low angle but acceptable win-rate -> should_rollback=False."""
        report = self.fe.check_stack(
            stack_id="stack-02",
            new_stack_id="stack-05",
            eval_data_path=Path("data/eval/stack-02.jsonl"),
            angle=20.0,
            winrate_base=0.82,
            winrate_adapted=0.80,
        )
        assert report.should_rollback is False
        assert report.passed is True

    def test_high_winrate_drop_safe_angle_no_rollback(self):
        """High win-rate drop but safe angle -> no rollback."""
        report = self.fe.check_stack(
            stack_id="stack-01",
            new_stack_id="stack-04",
            eval_data_path=Path("data/eval/stack-01.jsonl"),
            angle=60.0,
            winrate_base=0.85,
            winrate_adapted=0.75,
        )
        assert report.should_rollback is False
        assert report.passed is True

    def test_report_fields_complete(self):
        """Verify all ForgettingReport fields are set correctly."""
        report = self.fe.check_stack(
            stack_id="stack-01",
            new_stack_id="stack-02",
            eval_data_path=Path("data/eval/stack-01.jsonl"),
            angle=40.0,
            winrate_base=0.90,
            winrate_adapted=0.88,
        )
        assert report.stack_id == "stack-01"
        assert report.new_stack_id == "stack-02"
        assert report.angle == 40.0
        assert report.winrate_base == 0.90
        assert report.winrate_adapted == 0.88
        assert report.winrate_drop == pytest.approx(0.02)


# -----------------------------------------------------------------------
# ForgettingEvaluator.check_all_previous
# -----------------------------------------------------------------------


class TestCheckAllPrevious:
    """Test batch forgetting check across all previous stacks."""

    def setup_method(self):
        self.fe = ForgettingEvaluator(
            stack_evaluator=MagicMock(),
            analyzer=GradientSubspaceAnalyzer(),
        )

    def test_runs_for_each_previous_stack(self):
        """Should return one report per prior stack."""
        results = [
            {"stack_id": "stack-01", "angle": 50.0, "winrate_base": 0.80, "winrate_adapted": 0.79},
            {"stack_id": "stack-02", "angle": 55.0, "winrate_base": 0.82, "winrate_adapted": 0.81},
            {"stack_id": "stack-03", "angle": 48.0, "winrate_base": 0.78, "winrate_adapted": 0.77},
        ]
        reports = self.fe.check_all_previous(
            trained_stacks=["stack-01", "stack-02", "stack-03"],
            new_stack_id="stack-04",
            results=results,
        )
        assert len(reports) == 3
        assert all(r.new_stack_id == "stack-04" for r in reports)
        assert [r.stack_id for r in reports] == ["stack-01", "stack-02", "stack-03"]

    def test_all_pass(self):
        """All stacks pass -> all reports have passed=True."""
        results = [
            {"stack_id": "stack-01", "angle": 50.0, "winrate_base": 0.80, "winrate_adapted": 0.79},
            {"stack_id": "stack-02", "angle": 55.0, "winrate_base": 0.82, "winrate_adapted": 0.81},
        ]
        reports = self.fe.check_all_previous(
            trained_stacks=["stack-01", "stack-02"],
            new_stack_id="stack-03",
            results=results,
        )
        assert all(r.passed for r in reports)
        assert not any(r.should_rollback for r in reports)

    def test_one_rollback_among_many(self):
        """One stack triggers rollback among several passing stacks."""
        results = [
            {"stack_id": "stack-01", "angle": 50.0, "winrate_base": 0.80, "winrate_adapted": 0.79},
            {"stack_id": "stack-02", "angle": 25.0, "winrate_base": 0.82, "winrate_adapted": 0.74},
            {"stack_id": "stack-03", "angle": 48.0, "winrate_base": 0.78, "winrate_adapted": 0.77},
        ]
        reports = self.fe.check_all_previous(
            trained_stacks=["stack-01", "stack-02", "stack-03"],
            new_stack_id="stack-04",
            results=results,
        )
        rollbacks = [r for r in reports if r.should_rollback]
        assert len(rollbacks) == 1
        assert rollbacks[0].stack_id == "stack-02"


# -----------------------------------------------------------------------
# Legacy function tests (backward compat)
# -----------------------------------------------------------------------


class TestLegacyCheckForgetting:
    def test_passes_when_both_safe(self):
        r = check_forgetting(0.01, 45.0, "stack-01", "stack-02")
        assert r.passed is True

    def test_fails_when_both_bad(self):
        r = check_forgetting(0.05, 20.0, "stack-01", "stack-02")
        assert r.passed is False

    def test_passes_when_only_delta_bad(self):
        r = check_forgetting(0.05, 45.0, "stack-01", "stack-02")
        assert r.passed is True

    def test_passes_when_only_angle_bad(self):
        r = check_forgetting(0.01, 20.0, "stack-01", "stack-02")
        assert r.passed is True


class TestLegacySaveReport:
    def test_saves_json(self, tmp_path):
        results = [
            check_forgetting(0.01, 45.0, "stack-01", "stack-03"),
            check_forgetting(0.02, 50.0, "stack-02", "stack-03"),
        ]
        path = save_forgetting_report(results, "stack-03", str(tmp_path))
        data = json.loads(path.read_text())
        assert data["all_passed"] is True
        assert len(data["checks"]) == 2
