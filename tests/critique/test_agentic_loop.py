from __future__ import annotations

import json
import pytest
from src.critique.agentic_loop import AgenticLoop


class TestAgenticLoop:
    @pytest.mark.asyncio
    async def test_completes_in_single_iteration(self):
        plan = json.dumps([
            {"step": 1, "action": "answer directly", "tool": None, "expected_output": "answer"}
        ])
        eval_ok = json.dumps({
            "meets_expectations": True, "issues": [],
            "should_retry": False, "next_action": "proceed",
        })
        call_count = 0

        async def mock_generate(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return plan
            if call_count == 2:
                return "The answer is 42"
            return eval_ok

        loop = AgenticLoop(generate_fn=mock_generate, max_iterations=5)
        result = await loop.run(task="What is the answer?", tools=[])
        assert result.completed is True
        assert result.iterations == 1
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_respects_max_iterations(self):
        plan = json.dumps([
            {"step": 1, "action": "try something", "tool": None, "expected_output": "result"}
        ])
        eval_retry = json.dumps({
            "meets_expectations": False, "issues": ["wrong"],
            "should_retry": True, "next_action": "retry",
        })

        async def mock_generate(prompt: str) -> str:
            if "Break this task" in prompt:
                return plan
            if "Evaluate" in prompt:
                return eval_retry
            return "attempt"

        loop = AgenticLoop(generate_fn=mock_generate, max_iterations=3)
        result = await loop.run(task="impossible task", tools=[])
        assert result.completed is False
        assert result.iterations == 3

    @pytest.mark.asyncio
    async def test_abort_stops_loop(self):
        plan = json.dumps([
            {"step": 1, "action": "fail", "tool": None, "expected_output": "x"}
        ])
        eval_abort = json.dumps({
            "meets_expectations": False, "issues": ["fatal"],
            "should_retry": False, "next_action": "abort",
        })

        async def mock_generate(prompt: str) -> str:
            if "Break this task" in prompt:
                return plan
            if "Evaluate" in prompt:
                return eval_abort
            return "failed attempt"

        loop = AgenticLoop(generate_fn=mock_generate, max_iterations=5)
        result = await loop.run(task="bad task", tools=[])
        assert result.completed is False
        assert result.iterations == 1
