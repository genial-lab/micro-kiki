"""Tests for AntiBiasPipeline orchestrator (Story-93)."""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock

from src.cognitive.rbd import ReasoningBiasDetector, BiasDetection
from src.cognitive.antibias import (
    AntiBiasPipeline,
    AntiBiasResult,
    PipelineConfig,
    PipelineDecision,
)


def _make_detector(biased: bool, bias_type: str | None = None, confidence: float = 0.8) -> ReasoningBiasDetector:
    """Create a detector with a mocked generate_fn returning fixed detection."""
    result = json.dumps({
        "biased": biased,
        "bias_type": bias_type,
        "explanation": "test explanation",
        "confidence": confidence,
    })

    async def mock_gen(prompt: str) -> str:
        return result

    return ReasoningBiasDetector(generate_fn=mock_gen)


class TestAntiBiasPipelineClean:
    """Test pipeline with unbiased responses."""

    @pytest.mark.asyncio
    async def test_clean_response_passes_through(self):
        detector = _make_detector(biased=False, confidence=0.1)
        pipeline = AntiBiasPipeline(detector)

        result = await pipeline.process("What is Python?", "Python is a language.")
        assert not result.bias_detected
        assert not result.rewritten
        assert result.final_response == "Python is a language."

    @pytest.mark.asyncio
    async def test_clean_response_logged(self):
        detector = _make_detector(biased=False, confidence=0.1)
        pipeline = AntiBiasPipeline(detector)

        await pipeline.process("prompt", "response")
        assert len(pipeline.decisions) == 1
        assert not pipeline.decisions[0].bias_detected


class TestAntiBiasPipelineBiased:
    """Test pipeline with biased responses triggering DeFrame."""

    @pytest.mark.asyncio
    async def test_biased_triggers_rewrite(self):
        detector = _make_detector(biased=True, bias_type="stereotyping", confidence=0.9)

        async def mock_generate(prompt: str) -> str:
            return "Rewritten fair response."

        pipeline = AntiBiasPipeline(detector, generate_fn=mock_generate)
        result = await pipeline.process("biased prompt", "biased response")

        assert result.bias_detected
        assert result.rewritten
        assert result.final_response == "Rewritten fair response."
        assert result.original_response == "biased response"

    @pytest.mark.asyncio
    async def test_biased_without_generator_no_rewrite(self):
        detector = _make_detector(biased=True, bias_type="framing", confidence=0.7)
        pipeline = AntiBiasPipeline(detector, generate_fn=None)

        result = await pipeline.process("prompt", "framed response")
        assert result.bias_detected
        assert not result.rewritten
        assert result.final_response == "framed response"

    @pytest.mark.asyncio
    async def test_low_confidence_not_flagged(self):
        detector = _make_detector(biased=True, bias_type="anchoring", confidence=0.3)
        pipeline = AntiBiasPipeline(detector)

        result = await pipeline.process("prompt", "response")
        assert not result.bias_detected


class TestPipelineDecisionLog:
    """Test structured JSON logging of decisions."""

    @pytest.mark.asyncio
    async def test_multiple_decisions_logged(self):
        detector = _make_detector(biased=False, confidence=0.1)
        pipeline = AntiBiasPipeline(detector)

        await pipeline.process("prompt 1", "response 1")
        await pipeline.process("prompt 2", "response 2")
        await pipeline.process("prompt 3", "response 3")

        assert len(pipeline.decisions) == 3
        for d in pipeline.decisions:
            assert isinstance(d, PipelineDecision)
            assert d.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_flush_log_writes_json(self, tmp_path):
        detector = _make_detector(biased=True, bias_type="confirmation", confidence=0.8)

        async def mock_gen(prompt: str) -> str:
            return "fixed"

        pipeline = AntiBiasPipeline(
            detector,
            generate_fn=mock_gen,
            config=PipelineConfig(log_path=str(tmp_path / "decisions.json")),
        )

        await pipeline.process("test prompt", "biased text")
        log_path = pipeline.flush_log()

        assert log_path.exists()
        data = json.loads(log_path.read_text())
        assert len(data) == 1
        assert data[0]["bias_detected"] is True
        assert data[0]["bias_type"] == "confirmation"
        assert "timestamp" in data[0]

    @pytest.mark.asyncio
    async def test_clear_resets_decisions(self):
        detector = _make_detector(biased=False, confidence=0.0)
        pipeline = AntiBiasPipeline(detector)

        await pipeline.process("p", "r")
        assert len(pipeline.decisions) == 1

        pipeline.clear()
        assert len(pipeline.decisions) == 0


class TestPipelineStats:
    """Test summary statistics."""

    @pytest.mark.asyncio
    async def test_stats_counts(self):
        biased_detector = _make_detector(biased=True, bias_type="stereotyping", confidence=0.9)

        async def mock_gen(prompt: str) -> str:
            return "fixed"

        pipeline = AntiBiasPipeline(biased_detector, generate_fn=mock_gen)

        await pipeline.process("p1", "biased")
        await pipeline.process("p2", "biased")

        stats = pipeline.stats
        assert stats["total_processed"] == 2
        assert stats["flagged"] == 2
        assert stats["rewritten"] == 2
        assert stats["flag_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_stats_empty(self):
        detector = _make_detector(biased=False)
        pipeline = AntiBiasPipeline(detector)

        stats = pipeline.stats
        assert stats["total_processed"] == 0
        assert stats["flag_rate"] == 0.0
