"""Unit tests for src.eval.scorers."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from src.eval.scorers import JudgeScorer, containment_score


# ---------------------------------------------------------------------------
# Containment scorer
# ---------------------------------------------------------------------------


def test_containment_score_match() -> None:
    """Reference fully contained in response → score 1.0."""
    score = asyncio.run(
        containment_score(
            prompt="Qu'est-ce que Python?",
            reference="Python est un langage",
            response="La réponse est: Python est un langage de programmation.",
        )
    )
    assert score == 1.0


def test_containment_score_miss() -> None:
    """Reference tokens absent from response → score 0.0."""
    score = asyncio.run(
        containment_score(
            prompt="Qu'est-ce que Python?",
            reference="xyzzy fnord",
            response="Completely unrelated content.",
        )
    )
    assert score == 0.0


# ---------------------------------------------------------------------------
# Judge scorer
# ---------------------------------------------------------------------------


class _FakeJudgeClient:
    """Minimal judge client stub — records calls and returns a preset payload."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def generate(self, prompt: str, model: str) -> str:
        self.calls.append({"prompt": prompt, "model": model})
        return self.payload


def test_judge_scorer_happy_path() -> None:
    """Judge returning ``{"score": 0.8}`` → scorer returns 0.8."""
    client = _FakeJudgeClient(json.dumps({"winner": "stack", "score": 0.8, "reason": "ok"}))
    scorer = JudgeScorer(client, judge_model="mistral-large")

    score = asyncio.run(scorer("what is 2+2?", "4", "The answer is 4."))
    assert score == pytest.approx(0.8)

    # Verifies the canonical JUDGE_PROMPT was used (stack_eval template).
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "mistral-large"
    assert "Compare these two responses" in call["prompt"]
    assert "what is 2+2?" in call["prompt"]
    assert "The answer is 4." in call["prompt"]


def test_judge_scorer_bad_json(caplog: pytest.LogCaptureFixture) -> None:
    """Unparseable judge output → 0.0 with a warning log."""
    client = _FakeJudgeClient("not valid json at all")
    scorer = JudgeScorer(client)

    with caplog.at_level(logging.WARNING, logger="src.eval.scorers"):
        score = asyncio.run(scorer("p", "r", "resp"))

    assert score == 0.0
    assert any("bad JSON" in rec.message for rec in caplog.records), caplog.records


def test_judge_scorer_score_normalization() -> None:
    """Out-of-range scores are clipped into ``[0, 1]``."""
    # High: score = 5 → clipped to 1.0
    high_client = _FakeJudgeClient(json.dumps({"winner": "stack", "score": 5}))
    assert asyncio.run(JudgeScorer(high_client)("p", "r", "resp")) == 1.0

    # Low: score = -0.2 → clipped to 0.0
    low_client = _FakeJudgeClient(json.dumps({"winner": "base", "score": -0.2}))
    assert asyncio.run(JudgeScorer(low_client)("p", "r", "resp")) == 0.0

    # Mid: score = 0.42 → preserved
    mid_client = _FakeJudgeClient(json.dumps({"winner": "stack", "score": 0.42}))
    assert asyncio.run(JudgeScorer(mid_client)("p", "r", "resp")) == pytest.approx(0.42)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
