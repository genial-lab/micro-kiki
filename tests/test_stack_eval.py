from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock
from src.eval.stack_eval import StackEvaluator


@pytest.fixture
def evaluator():
    judge = AsyncMock()
    judge.generate.return_value = json.dumps({"winner": "stack", "score": 0.8, "reason": "Better"})
    return StackEvaluator(judge_client=judge)


class TestStackEvaluator:
    @pytest.mark.asyncio
    async def test_evaluate_returns_results(self, evaluator, tmp_path):
        ep = tmp_path / "eval.jsonl"
        ep.write_text("\n".join(json.dumps({"prompt": f"Q{i}"}) for i in range(3)))

        async def gen(prompt, adapter=None):
            return f"Response to {prompt[:10]}"

        r = await evaluator.evaluate(ep, gen, "stack-01")
        assert r["n_prompts"] == 3
        assert 0 <= r["win_rate_vs_base"] <= 1

    @pytest.mark.asyncio
    async def test_win_rate_range(self, evaluator, tmp_path):
        ep = tmp_path / "eval.jsonl"
        ep.write_text(json.dumps({"prompt": "test"}))

        async def gen(prompt, adapter=None):
            return "resp"

        r = await evaluator.evaluate(ep, gen, "test")
        assert 0 <= r["win_rate_vs_base"] <= 1
