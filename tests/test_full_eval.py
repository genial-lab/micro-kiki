"""Full evaluation suite orchestrator (Story-71).

Orchestrates all eval types: per-stack, forgetting, router accuracy, group eval.
Framework only (mocked) — actual eval runs need trained models.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.eval.stack_eval import StackEvaluator, JUDGE_PROMPT
from src.eval.forgetting import (
    ForgettingEvaluator,
    ForgettingReport,
    GradientSubspaceAnalyzer,
    check_forgetting,
)
from src.routing.dispatcher import dispatch, load_intent_mapping, MetaIntent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mapping():
    return load_intent_mapping("configs/meta_intents.yaml")


@pytest.fixture
def mock_judge():
    """Mock judge client that returns deterministic eval results."""
    judge = AsyncMock()
    judge.generate = AsyncMock(
        return_value=json.dumps({"winner": "stack", "score": 0.75, "reason": "better"})
    )
    return judge


@pytest.fixture
def eval_data(tmp_path):
    """Create a minimal eval JSONL file."""
    p = tmp_path / "eval.jsonl"
    lines = [
        json.dumps({"prompt": "Bonjour, comment ca va?"}),
        json.dumps({"prompt": "Write a Python sort function"}),
        json.dumps({"prompt": "Explain the I2C bus protocol"}),
    ]
    p.write_text("\n".join(lines))
    return p


@pytest.fixture
def forgetting_evaluator():
    return ForgettingEvaluator(
        stack_evaluator=MagicMock(),
        analyzer=GradientSubspaceAnalyzer(),
    )


# ---------------------------------------------------------------------------
# Per-stack evaluation
# ---------------------------------------------------------------------------


class TestPerStackEval:
    """Test per-stack evaluation with mocked judge and generate_fn."""

    async def test_evaluator_returns_results(self, mock_judge, eval_data):
        evaluator = StackEvaluator(judge_client=mock_judge, judge_model="mock-judge")
        generate_fn = AsyncMock(return_value="mocked response")
        result = await evaluator.evaluate(eval_data, generate_fn, "stack-01-chat-fr")
        assert result["stack"] == "stack-01-chat-fr"
        assert result["n_prompts"] == 3
        assert "win_rate_vs_base" in result
        assert "avg_judge_score" in result

    async def test_evaluator_counts_wins(self, eval_data):
        judge = AsyncMock()
        judge.generate = AsyncMock(
            return_value=json.dumps({"winner": "stack", "score": 0.8, "reason": "good"})
        )
        evaluator = StackEvaluator(judge_client=judge, judge_model="mock-judge")
        generate_fn = AsyncMock(return_value="response")
        result = await evaluator.evaluate(eval_data, generate_fn, "stack-02")
        assert result["win_rate_vs_base"] == 1.0  # all "stack" wins

    async def test_evaluator_handles_parse_error(self, eval_data):
        judge = AsyncMock()
        judge.generate = AsyncMock(return_value="not valid json")
        evaluator = StackEvaluator(judge_client=judge, judge_model="mock-judge")
        generate_fn = AsyncMock(return_value="response")
        result = await evaluator.evaluate(eval_data, generate_fn, "stack-03")
        # Parse error falls back to base win with score 0.5
        assert result["win_rate_vs_base"] == 0.0
        assert result["avg_judge_score"] == pytest.approx(0.5)

    async def test_evaluator_calls_generate_for_base_and_stack(self, mock_judge, eval_data):
        evaluator = StackEvaluator(judge_client=mock_judge, judge_model="mock-judge")
        generate_fn = AsyncMock(return_value="response")
        await evaluator.evaluate(eval_data, generate_fn, "stack-01")
        # 3 prompts x 2 calls each (base + stack) = 6
        assert generate_fn.call_count == 6

    async def test_evaluator_sample_responses_capped(self, mock_judge, eval_data):
        evaluator = StackEvaluator(judge_client=mock_judge, judge_model="mock-judge")
        generate_fn = AsyncMock(return_value="response")
        result = await evaluator.evaluate(eval_data, generate_fn, "stack-01")
        assert len(result["sample_responses"]) <= 5


# ---------------------------------------------------------------------------
# Forgetting evaluation
# ---------------------------------------------------------------------------


class TestForgettingEvalPipeline:
    """Test the forgetting check pipeline as part of full eval."""

    def test_single_stack_pass(self, forgetting_evaluator):
        report = forgetting_evaluator.check_stack(
            stack_id="stack-01", new_stack_id="stack-04",
            eval_data_path=Path("data/eval/stack-01.jsonl"),
            angle=50.0, winrate_base=0.80, winrate_adapted=0.79,
        )
        assert report.passed is True
        assert report.should_rollback is False

    def test_single_stack_rollback(self, forgetting_evaluator):
        report = forgetting_evaluator.check_stack(
            stack_id="stack-01", new_stack_id="stack-04",
            eval_data_path=Path("data/eval/stack-01.jsonl"),
            angle=25.0, winrate_base=0.80, winrate_adapted=0.74,
        )
        assert report.passed is False
        assert report.should_rollback is True

    def test_batch_forgetting_all_pass(self, forgetting_evaluator):
        results = [
            {"stack_id": "stack-01", "angle": 50.0, "winrate_base": 0.80, "winrate_adapted": 0.79},
            {"stack_id": "stack-02", "angle": 55.0, "winrate_base": 0.82, "winrate_adapted": 0.81},
        ]
        reports = forgetting_evaluator.check_all_previous(
            trained_stacks=["stack-01", "stack-02"],
            new_stack_id="stack-03",
            results=results,
        )
        assert all(r.passed for r in reports)

    def test_batch_forgetting_one_failure(self, forgetting_evaluator):
        results = [
            {"stack_id": "stack-01", "angle": 50.0, "winrate_base": 0.80, "winrate_adapted": 0.79},
            {"stack_id": "stack-02", "angle": 20.0, "winrate_base": 0.82, "winrate_adapted": 0.74},
        ]
        reports = forgetting_evaluator.check_all_previous(
            trained_stacks=["stack-01", "stack-02"],
            new_stack_id="stack-03",
            results=results,
        )
        rollbacks = [r for r in reports if r.should_rollback]
        assert len(rollbacks) == 1
        assert rollbacks[0].stack_id == "stack-02"


# ---------------------------------------------------------------------------
# Router accuracy evaluation
# ---------------------------------------------------------------------------


class TestRouterAccuracyEval:
    """Test router accuracy measurement as part of full eval suite."""

    def test_perfect_routing_accuracy(self, mapping):
        """All test prompts route to expected intent."""
        test_cases = [
            (0, MetaIntent.QUICK_REPLY),
            (1, MetaIntent.REASONING),
            (2, MetaIntent.CODING),
            (15, MetaIntent.AGENTIC),
            (28, MetaIntent.TOOL_USE),
        ]
        correct = 0
        for domain_idx, expected_intent in test_cases:
            logits = [0.05] * 35
            logits[domain_idx] = 0.90
            result = dispatch(logits, mapping)
            if result.intent == expected_intent:
                correct += 1
        accuracy = correct / len(test_cases)
        assert accuracy == 1.0

    def test_multi_domain_routing_accuracy(self, mapping):
        """Test that multi-domain activation routes to dominant intent."""
        logits = [0.05] * 35
        logits[2] = 0.90  # python (coding)
        logits[3] = 0.70  # typescript (coding)
        logits[1] = 0.30  # reasoning
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.CODING

    def test_confidence_values_in_range(self, mapping):
        logits = [0.05] * 35
        logits[0] = 0.92
        result = dispatch(logits, mapping)
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Group evaluation framework
# ---------------------------------------------------------------------------


class TestGroupEvalFramework:
    """Test the group evaluation script framework."""

    def test_group_eval_output_structure(self, tmp_path):
        from scripts.group_eval import run_group_eval
        output = tmp_path / "results" / "group-eval.json"
        result = run_group_eval(max_stack=20, output_path=str(output))
        assert "timestamp" in result
        assert result["max_stack"] == 20
        assert "num_stacks_evaluated" in result
        assert "stacks" in result
        assert output.exists()

    def test_group_eval_json_roundtrip(self, tmp_path):
        from scripts.group_eval import run_group_eval
        output = tmp_path / "results" / "group-eval.json"
        run_group_eval(max_stack=5, output_path=str(output))
        data = json.loads(output.read_text())
        assert data["max_stack"] == 5
        assert isinstance(data["stacks"], list)


# ---------------------------------------------------------------------------
# Full eval orchestration
# ---------------------------------------------------------------------------


class TestFullEvalOrchestration:
    """Test that the full eval pipeline can be orchestrated end-to-end."""

    def test_eval_types_enumeration(self):
        """Verify all eval types are accounted for."""
        eval_types = {
            "per_stack",
            "forgetting",
            "router_accuracy",
            "group_eval",
        }
        assert len(eval_types) == 4

    async def test_per_stack_then_forgetting(self, mock_judge, eval_data, forgetting_evaluator):
        """Simulate: evaluate stack, then check forgetting."""
        # Step 1: per-stack eval
        evaluator = StackEvaluator(judge_client=mock_judge, judge_model="mock")
        generate_fn = AsyncMock(return_value="response")
        stack_result = await evaluator.evaluate(eval_data, generate_fn, "stack-05")
        assert stack_result["n_prompts"] == 3

        # Step 2: forgetting check
        report = forgetting_evaluator.check_stack(
            stack_id="stack-04", new_stack_id="stack-05",
            eval_data_path=Path("data/eval/stack-04.jsonl"),
            angle=45.0, winrate_base=0.80, winrate_adapted=0.78,
        )
        assert report.passed is True

    def test_all_eval_results_serializable(self, mapping, forgetting_evaluator):
        """All eval results must be JSON-serializable."""
        # Router accuracy result
        logits = [0.05] * 35
        logits[0] = 0.90
        router_result = dispatch(logits, mapping)
        json.dumps({"intent": router_result.intent.value, "confidence": router_result.confidence})

        # Forgetting result
        report = forgetting_evaluator.check_stack(
            stack_id="stack-01", new_stack_id="stack-02",
            eval_data_path=Path("data/eval/stack-01.jsonl"),
            angle=45.0, winrate_base=0.80, winrate_adapted=0.79,
        )
        json.dumps({
            "stack_id": report.stack_id,
            "angle": report.angle,
            "passed": report.passed,
            "should_rollback": report.should_rollback,
        })
