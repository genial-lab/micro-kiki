from __future__ import annotations

import json
import pytest
from src.critique.self_refine import SelfRefine, CritiqueResult


class TestSelfRefine:
    @pytest.mark.asyncio
    async def test_no_correction_when_critique_clean(self):
        critique_json = json.dumps({
            "factual_errors": [], "missing_info": [], "clarity_issues": [],
            "confidence": 0.95, "needs_correction": False,
            "summary": "Response is accurate and complete.",
        })

        async def mock_generate(prompt: str) -> str:
            return critique_json

        refine = SelfRefine(generate_fn=mock_generate)
        result = await refine.run(query="What is MoE?", response="MoE explanation...")
        assert result.corrected is False
        assert result.final_response == "MoE explanation..."

    @pytest.mark.asyncio
    async def test_correction_when_critique_finds_issues(self):
        call_count = 0
        critique_json = json.dumps({
            "factual_errors": ["Wrong parameter count"], "missing_info": [],
            "clarity_issues": [], "confidence": 0.4,
            "needs_correction": True, "summary": "Factual error.",
        })

        async def mock_generate(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return critique_json
            return "Corrected MoE explanation with right params"

        refine = SelfRefine(generate_fn=mock_generate)
        result = await refine.run(query="What is MoE?", response="Wrong MoE explanation")
        assert result.corrected is True
        assert "Corrected" in result.final_response
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_critique_result_contains_feedback(self):
        critique_json = json.dumps({
            "factual_errors": ["Error A"], "missing_info": ["Info B"],
            "clarity_issues": [], "confidence": 0.3,
            "needs_correction": True, "summary": "Needs work.",
        })

        async def mock_generate(prompt: str) -> str:
            if "critical reviewer" in prompt.lower() or "Critique" not in prompt:
                return critique_json
            return "Fixed"

        refine = SelfRefine(generate_fn=mock_generate)
        result = await refine.run(query="q", response="r")
        assert result.critique.factual_errors == ["Error A"]
        assert result.critique.confidence == 0.3
