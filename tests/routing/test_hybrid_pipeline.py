"""Tests for the triple-hybrid routing pipeline."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.routing.hybrid_pipeline import (
    HybridPipeline,
    HybridPipelineConfig,
    PipelineResult,
    _extract_confidence,
    _count_memory_lines,
)
from src.routing.model_router import RouteDecision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_quantum_router(domain: str = "embedded", confidence: float = 0.85) -> MagicMock:
    """Return a mock QuantumRouter whose .route() returns a RouteDecision."""
    qr = MagicMock()
    qr.route.return_value = RouteDecision(
        model_id="qwen35b",
        adapter=f"stack-{domain}",
        reason=f"quantum-vqc: {domain} (conf={confidence:.3f})",
    )
    return qr


def _make_model_router(model_id: str = "qwen35b", adapter: str | None = None) -> MagicMock:
    mr = MagicMock()
    mr.select.return_value = RouteDecision(
        model_id=model_id,
        adapter=adapter,
        reason="classical fallback",
    )
    return mr


def _make_aeon_hook(memories: int = 3) -> MagicMock:
    hook = MagicMock()
    memory_lines = "\n".join(f"[Memory] fact {i}" for i in range(memories))
    hook.pre_inference.side_effect = lambda prompt, **_: memory_lines + "\n" + prompt
    hook.post_inference.return_value = None
    return hook


def _make_negotiator(winner: str = "negotiated response") -> MagicMock:
    from src.cognitive.negotiator import NegotiationResult

    neg = MagicMock()
    neg.negotiate = AsyncMock(
        return_value=NegotiationResult(
            winner_response=winner,
            winner_idx=0,
            judge_result=MagicMock(),
            catfish_result=None,
            num_candidates=2,
        )
    )
    return neg


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


def test_extract_confidence_present() -> None:
    reason = "quantum-vqc: embedded (conf=0.823)"
    assert abs(_extract_confidence(reason) - 0.823) < 1e-3


def test_extract_confidence_absent() -> None:
    assert _extract_confidence("classical fallback") == 0.0


def test_count_memory_lines() -> None:
    original = "hello world"
    augmented = "[Memory] fact 0\n[Memory] fact 1\nhello world"
    assert _count_memory_lines(augmented, original) == 2


def test_count_memory_lines_no_memory() -> None:
    prompt = "no memories here"
    assert _count_memory_lines(prompt, prompt) == 0


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_all_components() -> None:
    """Pipeline with quantum (high confidence), memory, and negotiator all active."""
    config = HybridPipelineConfig(
        use_quantum=True,
        use_memory=True,
        use_negotiator=True,
        quantum_confidence_threshold=0.7,
    )
    qr = _make_quantum_router(domain="kicad-dsl", confidence=0.90)
    mr = _make_model_router()
    aeon = _make_aeon_hook(memories=2)
    negotiator = _make_negotiator(winner="best answer")

    pipeline = HybridPipeline(config, quantum_router=qr, model_router=mr,
                              aeon_hook=aeon, negotiator=negotiator)
    result = await pipeline.route_and_infer("design a KiCad schematic")

    assert isinstance(result, PipelineResult)
    assert result.quantum_used is True
    assert result.quantum_confidence >= 0.7
    assert result.memories_injected == 2
    assert result.negotiator_used is True
    assert result.response == "best answer"
    assert result.latency_ms >= 0.0
    assert result.route.adapter == "stack-kicad-dsl"


@pytest.mark.asyncio
async def test_quantum_fallback_to_classical_when_low_confidence() -> None:
    """Quantum confidence below threshold → classical router is used instead."""
    config = HybridPipelineConfig(
        use_quantum=True,
        use_memory=False,
        use_negotiator=False,
        quantum_confidence_threshold=0.7,
    )
    # Quantum returns low confidence
    qr = _make_quantum_router(domain="embedded", confidence=0.45)
    mr = _make_model_router(model_id="devstral", adapter=None)

    pipeline = HybridPipeline(config, quantum_router=qr, model_router=mr)
    result = await pipeline.route_and_infer("write some python code")

    # Quantum ran but its decision was discarded
    assert result.quantum_used is False
    assert result.quantum_confidence == pytest.approx(0.45, abs=1e-3)
    # Classical router result was used
    assert result.route.model_id == "devstral"
    qr.route.assert_called_once()
    mr.select.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_without_quantum() -> None:
    """use_quantum=False → quantum router is never called."""
    config = HybridPipelineConfig(use_quantum=False, use_memory=False, use_negotiator=False)
    qr = _make_quantum_router()
    mr = _make_model_router(model_id="qwen35b", adapter="stack-spice")

    pipeline = HybridPipeline(config, quantum_router=qr, model_router=mr)
    result = await pipeline.route_and_infer("simulate a SPICE circuit")

    assert result.quantum_used is False
    assert result.quantum_confidence == 0.0
    qr.route.assert_not_called()
    mr.select.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_without_memory() -> None:
    """use_memory=False → AeonServingHook is never called."""
    config = HybridPipelineConfig(use_quantum=False, use_memory=False, use_negotiator=False)
    aeon = _make_aeon_hook(memories=5)
    mr = _make_model_router()

    pipeline = HybridPipeline(config, model_router=mr, aeon_hook=aeon)
    result = await pipeline.route_and_infer("explain embedded systems")

    assert result.memories_injected == 0
    aeon.pre_inference.assert_not_called()
    aeon.post_inference.assert_not_called()


def test_route_only_returns_route_decision() -> None:
    """route_only must return a RouteDecision without raising."""
    config = HybridPipelineConfig(use_quantum=False)
    mr = _make_model_router(model_id="devstral")

    pipeline = HybridPipeline(config, model_router=mr)
    decision = pipeline.route_only("write a firmware driver")

    assert isinstance(decision, RouteDecision)
    assert decision.model_id == "devstral"


def test_route_only_uses_quantum_when_high_confidence() -> None:
    """route_only falls through to quantum when confidence is sufficient."""
    config = HybridPipelineConfig(
        use_quantum=True,
        quantum_confidence_threshold=0.7,
    )
    qr = _make_quantum_router(domain="stm32", confidence=0.92)
    mr = _make_model_router()

    pipeline = HybridPipeline(config, quantum_router=qr, model_router=mr)
    decision = pipeline.route_only("configure STM32 timer")

    assert decision.adapter == "stack-stm32"
    qr.route.assert_called_once()
    mr.select.assert_not_called()


@pytest.mark.asyncio
async def test_negotiator_disabled_by_default() -> None:
    """Default config disables negotiator — it should not be called."""
    config = HybridPipelineConfig()  # use_negotiator=False by default
    qr = _make_quantum_router(confidence=0.95)
    negotiator = _make_negotiator()

    pipeline = HybridPipeline(config, quantum_router=qr, negotiator=negotiator)
    result = await pipeline.route_and_infer("what is the answer?")

    assert result.negotiator_used is False
    negotiator.negotiate.assert_not_called()
