"""Triple-hybrid routing pipeline: Quantum → SNN → Classical.

Orchestrates:
1. Quantum VQC Router — domain classification (PennyLane simulator)
2. Classical Model Router — model + adapter selection
3. Aeon Memory — pre-inference context injection
4. Inference dispatch — to selected model
5. Negotiator — post-inference quality arbitration (if multi-candidate)
6. Aeon Memory — post-inference persistence
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.routing.model_router import ModelRouter, RouteDecision

if TYPE_CHECKING:
    from src.routing.quantum_router import QuantumRouter
    from src.serving.aeon_hook import AeonServingHook
    from src.cognitive.negotiator import Negotiator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridPipelineConfig:
    """Frozen configuration for the triple-hybrid routing pipeline.

    Args:
        use_quantum: Enable VQC router. Falls back to classical if False or
            PennyLane is unavailable.
        use_memory: Enable Aeon pre/post-inference memory.
        use_negotiator: Enable CAMP negotiator for multi-candidate arbitration.
            Disabled by default (adds latency).
        quantum_confidence_threshold: If quantum confidence is below this value
            the classical ModelRouter is also run as a backup and its result
            takes precedence.
    """

    use_quantum: bool = True
    use_memory: bool = True
    use_negotiator: bool = False
    quantum_confidence_threshold: float = 0.7


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Full result returned by :meth:`HybridPipeline.route_and_infer`.

    Args:
        response: Final model response (stub in inference-less deployments).
        route: Selected routing decision.
        quantum_used: Whether the VQC router contributed to the decision.
        quantum_confidence: Max softmax probability from the VQC circuit.
        memories_injected: Number of memory lines prepended to the prompt.
        negotiator_used: Whether the Negotiator ran arbitration this turn.
        latency_ms: Wall-clock time for the full pipeline in milliseconds.
    """

    response: str
    route: RouteDecision
    quantum_used: bool
    quantum_confidence: float
    memories_injected: int
    negotiator_used: bool
    latency_ms: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_confidence(reason: str) -> float:
    """Parse the confidence value embedded by QuantumRouter in reason strings.

    QuantumRouter encodes confidence as ``conf=0.xxx`` in the reason field.
    Returns 0.0 if the pattern is absent.
    """
    marker = "conf="
    idx = reason.find(marker)
    if idx == -1:
        return 0.0
    try:
        return float(reason[idx + len(marker):idx + len(marker) + 5])
    except ValueError:
        return 0.0


def _count_memory_lines(augmented: str, original: str) -> int:
    """Count [Memory] lines prepended by AeonServingHook.pre_inference."""
    if augmented == original:
        return 0
    prefix = augmented[: len(augmented) - len(original)]
    return prefix.count("[Memory]")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class HybridPipeline:
    """Triple-hybrid routing pipeline.

    Wires together:
    - QuantumRouter (VQC domain classification)
    - ModelRouter  (model + adapter selection)
    - AeonServingHook (pre/post-inference memory)
    - Negotiator (optional CAMP arbitration)

    All components are injected — pass ``None`` to disable individually or
    set the corresponding flag in :class:`HybridPipelineConfig`.
    """

    def __init__(
        self,
        config: HybridPipelineConfig | None = None,
        quantum_router: QuantumRouter | None = None,
        model_router: ModelRouter | None = None,
        aeon_hook: AeonServingHook | None = None,
        negotiator: Negotiator | None = None,
    ) -> None:
        self.config = config or HybridPipelineConfig()
        self._quantum = quantum_router
        self._model_router = model_router or ModelRouter()
        self._aeon = aeon_hook
        self._negotiator = negotiator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_only(self, query: str) -> RouteDecision:
        """Run routing only (no memory, no inference, no negotiation).

        Uses quantum router when available and configured, otherwise falls
        back to the classical ModelRouter.

        Args:
            query: Raw user query string.

        Returns:
            RouteDecision with model_id, adapter, and reason.
        """
        if self.config.use_quantum and self._quantum is not None:
            embedding = _query_to_embedding(query)
            decision = self._quantum.route(embedding)
            confidence = _extract_confidence(decision.reason)
            if confidence >= self.config.quantum_confidence_threshold:
                logger.debug(
                    "route_only: quantum decision (conf=%.3f) → %s",
                    confidence,
                    decision.model_id,
                )
                return decision
            # Low confidence — fall through to classical
            logger.debug(
                "route_only: quantum confidence %.3f < threshold %.3f, using classical",
                confidence,
                self.config.quantum_confidence_threshold,
            )

        domain_hint = _domain_hint_from_query(query)
        return self._model_router.select(query, domain_hint=domain_hint)

    async def route_and_infer(
        self,
        query: str,
        context: dict | None = None,
    ) -> PipelineResult:
        """Full pipeline: route → memory → infer → negotiate → persist.

        Steps:
        1. Quantum router classifies domain (if available and enabled).
        2. If quantum confidence < threshold, run classical router as backup.
        3. ModelRouter selects final model + adapter.
        4. Aeon pre-inference: inject memories into prompt.
        5. Inference (stubbed — returns placeholder).
        6. Negotiator arbitration (if enabled and multiple candidates).
        7. Aeon post-inference: persist turn.

        Args:
            query: Raw user query string.
            context: Optional extra context dict (reserved for future use).

        Returns:
            PipelineResult with response, routing metadata, and timing.
        """
        t0 = time.monotonic()
        turn_id = str(uuid.uuid4())

        # ------------------------------------------------------------------
        # Steps 1-3: routing
        # ------------------------------------------------------------------
        quantum_used = False
        quantum_confidence = 0.0
        route: RouteDecision | None = None

        if self.config.use_quantum and self._quantum is not None:
            embedding = _query_to_embedding(query)
            q_decision = self._quantum.route(embedding)
            quantum_confidence = _extract_confidence(q_decision.reason)

            if quantum_confidence >= self.config.quantum_confidence_threshold:
                route = q_decision
                quantum_used = True
                logger.debug(
                    "quantum router selected (conf=%.3f) → %s",
                    quantum_confidence,
                    route.model_id,
                )
            else:
                logger.debug(
                    "quantum confidence %.3f below threshold %.3f — running classical fallback",
                    quantum_confidence,
                    self.config.quantum_confidence_threshold,
                )

        if route is None:
            # Classical fallback (Steps 2-3)
            domain_hint = _domain_hint_from_query(query)
            route = self._model_router.select(query, domain_hint=domain_hint)
            logger.debug("classical router selected → %s (adapter=%s)", route.model_id, route.adapter)

        # ------------------------------------------------------------------
        # Step 4: Aeon pre-inference
        # ------------------------------------------------------------------
        prompt = query
        memories_injected = 0

        if self.config.use_memory and self._aeon is not None:
            augmented = self._aeon.pre_inference(prompt)
            memories_injected = _count_memory_lines(augmented, prompt)
            prompt = augmented
            logger.debug("aeon pre-inference: %d memories injected", memories_injected)

        # ------------------------------------------------------------------
        # Step 5: Inference (stub)
        # ------------------------------------------------------------------
        response = await _stub_infer(prompt, route, context)

        # ------------------------------------------------------------------
        # Step 6: Negotiator (optional)
        # ------------------------------------------------------------------
        negotiator_used = False

        if self.config.use_negotiator and self._negotiator is not None:
            candidates = [response]
            # Multi-candidate: generate a second stub variant for arbitration
            alt_response = await _stub_infer(prompt, route, context, variant=1)
            candidates.append(alt_response)

            try:
                neg_result = await self._negotiator.negotiate(prompt, candidates)
                response = neg_result.winner_response
                negotiator_used = True
                logger.debug(
                    "negotiator arbitrated %d candidates → winner idx %d",
                    neg_result.num_candidates,
                    neg_result.winner_idx,
                )
            except Exception:
                logger.warning("Negotiator failed, keeping original response", exc_info=True)

        # ------------------------------------------------------------------
        # Step 7: Aeon post-inference
        # ------------------------------------------------------------------
        if self.config.use_memory and self._aeon is not None:
            domain = route.adapter.replace("stack-", "") if route.adapter else "base"
            try:
                self._aeon.post_inference(
                    prompt=query,
                    response=response,
                    domain=domain,
                    turn_id=turn_id,
                )
            except Exception:
                logger.warning("Aeon post-inference failed for turn %s", turn_id, exc_info=True)

        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            "pipeline done: model=%s adapter=%s quantum=%s memories=%d latency=%.1fms",
            route.model_id,
            route.adapter,
            quantum_used,
            memories_injected,
            latency_ms,
        )

        return PipelineResult(
            response=response,
            route=route,
            quantum_used=quantum_used,
            quantum_confidence=quantum_confidence,
            memories_injected=memories_injected,
            negotiator_used=negotiator_used,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _query_to_embedding(query: str) -> np.ndarray:
    """Minimal text → float embedding for routing (bag-of-chars hashing).

    This is intentionally lightweight — just enough to feed the VQC circuit
    during integration.  Replace with a real sentence encoder for production.

    Returns:
        Float32 array of length 64.
    """
    dim = 64
    vec = np.zeros(dim, dtype=np.float32)
    for i, ch in enumerate(query):
        vec[i % dim] += float(ord(ch))
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _domain_hint_from_query(query: str) -> str | None:
    """Cheap keyword scan to extract a domain hint for the classical router."""
    from src.routing.router import NICHE_DOMAINS

    q_lower = query.lower()
    for domain in NICHE_DOMAINS:
        if domain in q_lower:
            return domain
    return None


async def _stub_infer(
    prompt: str,
    route: RouteDecision,
    context: dict | None,
    variant: int = 0,
) -> str:
    """Stub inference — returns a placeholder response.

    Replace with real dispatch (MLX / vLLM) in production.
    """
    adapter_tag = f" [{route.adapter}]" if route.adapter else ""
    suffix = f" (v{variant})" if variant else ""
    return f"[stub:{route.model_id}{adapter_tag}]{suffix} {prompt[:80]}"
