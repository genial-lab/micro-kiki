"""Integration tests for the full anti-bias pipeline: RBD detection + DeFrame rewrite."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.cognitive.rbd import ReasoningBiasDetector, BiasDetection
from src.cognitive.antibias import (
    AntiBiasOrchestrator,
    AntiBiasPipeline,
    AntiBiasResult,
    PipelineConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rbd_response(biased: bool, bias_type: str | None = None,
                       explanation: str = "", confidence: float = 0.0) -> str:
    return json.dumps({
        "biased": biased,
        "bias_type": bias_type,
        "explanation": explanation,
        "confidence": confidence,
    })


@pytest.fixture
def clean_generate() -> AsyncMock:
    """Generate function that always reports no bias."""
    gen = AsyncMock()
    gen.return_value = _make_rbd_response(
        biased=False, explanation="No bias detected", confidence=0.1,
    )
    return gen


@pytest.fixture
def biased_generate() -> AsyncMock:
    """Generate function that detects bias on first call, rewrites on second."""
    gen = AsyncMock()
    gen.side_effect = [
        _make_rbd_response(
            biased=True, bias_type="stereotyping",
            explanation="Gender stereotyping detected", confidence=0.92,
        ),
        "A fair and balanced rewrite of the original response.",
    ]
    return gen


@pytest.fixture
def pipeline_clean(clean_generate: AsyncMock) -> AntiBiasPipeline:
    rbd = ReasoningBiasDetector(generate_fn=clean_generate)
    return AntiBiasPipeline(detector=rbd, generate_fn=clean_generate)


@pytest.fixture
def pipeline_biased(biased_generate: AsyncMock) -> AntiBiasPipeline:
    rbd = ReasoningBiasDetector(generate_fn=biased_generate)
    return AntiBiasPipeline(detector=rbd, generate_fn=biased_generate)


# ---------------------------------------------------------------------------
# Full pipeline: biased input → detect → rewrite
# ---------------------------------------------------------------------------

class TestAntiBiasPipelineBiasedInput:
    @pytest.mark.asyncio
    async def test_detects_bias(self, pipeline_biased: AntiBiasPipeline):
        # Arrange
        prompt = "Describe a typical nurse"
        response = "Nurses are usually women who are caring and gentle."

        # Act
        result = await pipeline_biased.process(prompt, response)

        # Assert
        assert result.bias_detected is True
        assert result.detection.bias_type == "stereotyping"
        assert result.detection.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_rewrites_biased_response(self, pipeline_biased: AntiBiasPipeline):
        # Arrange
        prompt = "Describe a typical nurse"
        response = "Nurses are usually women who are caring and gentle."

        # Act
        result = await pipeline_biased.process(prompt, response)

        # Assert
        assert result.rewritten is True
        assert result.final_response != result.original_response
        assert "fair and balanced" in result.final_response

    @pytest.mark.asyncio
    async def test_preserves_original_in_result(self, pipeline_biased: AntiBiasPipeline):
        prompt = "Describe a typical nurse"
        response = "Nurses are usually women who are caring and gentle."

        result = await pipeline_biased.process(prompt, response)

        assert result.original_response == response


# ---------------------------------------------------------------------------
# Full pipeline: clean input → detect → pass through
# ---------------------------------------------------------------------------

class TestAntiBiasPipelineCleanInput:
    @pytest.mark.asyncio
    async def test_no_bias_detected(self, pipeline_clean: AntiBiasPipeline):
        prompt = "What is the boiling point of water?"
        response = "Water boils at 100 degrees Celsius at standard pressure."

        result = await pipeline_clean.process(prompt, response)

        assert result.bias_detected is False

    @pytest.mark.asyncio
    async def test_response_unchanged(self, pipeline_clean: AntiBiasPipeline):
        prompt = "What is the boiling point of water?"
        response = "Water boils at 100 degrees Celsius at standard pressure."

        result = await pipeline_clean.process(prompt, response)

        assert result.rewritten is False
        assert result.final_response == response

    @pytest.mark.asyncio
    async def test_detection_confidence_below_threshold(self, pipeline_clean: AntiBiasPipeline):
        prompt = "Explain photosynthesis"
        response = "Plants convert sunlight into energy."

        result = await pipeline_clean.process(prompt, response)

        assert result.detection.confidence < 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestAntiBiasPipelineEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_input(self, pipeline_clean: AntiBiasPipeline):
        result = await pipeline_clean.process("", "")

        assert result.bias_detected is False
        assert result.final_response == ""

    @pytest.mark.asyncio
    async def test_very_short_input(self, pipeline_clean: AntiBiasPipeline):
        result = await pipeline_clean.process("Hi", "OK")

        assert result.bias_detected is False
        assert result.final_response == "OK"

    @pytest.mark.asyncio
    async def test_no_generate_fn_skips_rewrite(self):
        """When no generate_fn is provided, bias is detected but not rewritten."""
        detect_gen = AsyncMock(return_value=_make_rbd_response(
            biased=True, bias_type="framing",
            explanation="Framing bias", confidence=0.85,
        ))
        rbd = ReasoningBiasDetector(generate_fn=detect_gen)
        pipeline = AntiBiasPipeline(detector=rbd, generate_fn=None)

        result = await pipeline.process("prompt", "biased response")

        assert result.bias_detected is True
        assert result.rewritten is False
        assert result.final_response == "biased response"

    @pytest.mark.asyncio
    async def test_low_confidence_bias_not_rewritten(self):
        """Bias flagged with confidence < 0.5 should not trigger rewrite."""
        gen = AsyncMock(return_value=_make_rbd_response(
            biased=True, bias_type="anchoring",
            explanation="Mild anchoring", confidence=0.3,
        ))
        rbd = ReasoningBiasDetector(generate_fn=gen)
        pipeline = AntiBiasPipeline(detector=rbd, generate_fn=gen)

        result = await pipeline.process("prompt", "response")

        assert result.bias_detected is False
        assert result.rewritten is False

    @pytest.mark.asyncio
    async def test_malformed_rbd_json_treated_as_clean(self):
        """If the RBD returns invalid JSON, treat as no bias."""
        gen = AsyncMock(return_value="NOT VALID JSON {{{")
        rbd = ReasoningBiasDetector(generate_fn=gen)
        pipeline = AntiBiasPipeline(detector=rbd, generate_fn=gen)

        result = await pipeline.process("prompt", "response")

        assert result.bias_detected is False
        assert result.rewritten is False


# ---------------------------------------------------------------------------
# Pipeline bookkeeping: decisions log and stats
# ---------------------------------------------------------------------------

class TestAntiBiasPipelineBookkeeping:
    @pytest.mark.asyncio
    async def test_decisions_logged(self, pipeline_clean: AntiBiasPipeline):
        await pipeline_clean.process("p1", "r1")
        await pipeline_clean.process("p2", "r2")

        assert len(pipeline_clean.decisions) == 2
        assert pipeline_clean.decisions[0].prompt_preview == "p1"
        assert pipeline_clean.decisions[1].prompt_preview == "p2"

    @pytest.mark.asyncio
    async def test_stats_clean(self, pipeline_clean: AntiBiasPipeline):
        await pipeline_clean.process("p", "r")

        stats = pipeline_clean.stats
        assert stats["total_processed"] == 1
        assert stats["flagged"] == 0
        assert stats["rewritten"] == 0
        assert stats["flag_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_stats_biased(self, pipeline_biased: AntiBiasPipeline):
        await pipeline_biased.process("p", "r")

        stats = pipeline_biased.stats
        assert stats["total_processed"] == 1
        assert stats["flagged"] == 1
        assert stats["rewritten"] == 1

    @pytest.mark.asyncio
    async def test_flush_log_creates_file(self, pipeline_clean: AntiBiasPipeline):
        await pipeline_clean.process("p", "r")

        with tempfile.TemporaryDirectory() as tmp:
            path = pipeline_clean.flush_log(f"{tmp}/decisions.json")
            assert path.exists()
            data = json.loads(path.read_text())
            assert len(data) == 1
            assert data[0]["prompt_preview"] == "p"

    @pytest.mark.asyncio
    async def test_clear_resets_state(self, pipeline_clean: AntiBiasPipeline):
        await pipeline_clean.process("p", "r")
        pipeline_clean.clear()

        assert len(pipeline_clean.decisions) == 0
        assert pipeline_clean.stats["total_processed"] == 0

    @pytest.mark.asyncio
    async def test_latency_recorded(self, pipeline_clean: AntiBiasPipeline):
        await pipeline_clean.process("p", "r")

        decision = pipeline_clean.decisions[0]
        assert decision.latency_ms >= 0
