from __future__ import annotations

import json
import pytest
from pathlib import Path
from src.ralph.forgetting_auto import ForgettingChecker, ForgettingResult


@pytest.fixture
def checker(tmp_path):
    return ForgettingChecker(
        evals_dir=tmp_path / "evals",
        angle_threshold=30.0,
        winrate_drop_threshold=0.03,
    )


class TestForgettingChecker:
    def test_pass_when_angle_high_and_winrate_stable(self, checker):
        result = checker.evaluate(angle=45.0, winrate_base=0.75, winrate_adapted=0.74)
        assert result.passed is True
        assert result.should_rollback is False

    def test_fail_when_angle_low_and_winrate_drops(self, checker):
        result = checker.evaluate(angle=25.0, winrate_base=0.75, winrate_adapted=0.70)
        assert result.passed is False
        assert result.should_rollback is True

    def test_pass_when_angle_low_but_winrate_stable(self, checker):
        result = checker.evaluate(angle=25.0, winrate_base=0.75, winrate_adapted=0.73)
        assert result.passed is True
        assert result.should_rollback is False

    def test_saves_eval_json(self, checker, tmp_path):
        result = checker.evaluate(angle=45.0, winrate_base=0.75, winrate_adapted=0.74)
        path = checker.save_result("stack-05", result)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["stack_id"] == "stack-05"
        assert data["passed"] is True
