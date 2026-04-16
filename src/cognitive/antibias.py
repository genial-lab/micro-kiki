"""Anti-bias orchestrator: RBD check + DeFrame re-generation + pipeline."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.cognitive.rbd import ReasoningBiasDetector, BiasDetection

logger = logging.getLogger(__name__)

DEFRAME_PROMPT = """The following response was flagged for potential {bias_type} bias.
Rewrite it to be fair, balanced, and free of the identified bias while preserving accuracy.

Original prompt: {prompt}
Original response: {response}
Bias detected: {explanation}

Rewrite:"""


@dataclass(frozen=True)
class AntiBiasResult:
    original_response: str
    final_response: str
    bias_detected: bool
    detection: BiasDetection
    rewritten: bool


@dataclass(frozen=True)
class PipelineDecision:
    """A single logged decision from the AntiBiasPipeline."""
    timestamp: str
    prompt_preview: str
    bias_detected: bool
    bias_type: str | None
    confidence: float
    rewritten: bool
    latency_ms: float


class AntiBiasOrchestrator:
    def __init__(self, detector: ReasoningBiasDetector, generate_fn=None) -> None:
        self._detector = detector
        self._generate = generate_fn

    async def check_and_fix(self, prompt: str, response: str) -> AntiBiasResult:
        detection = await self._detector.detect(prompt, response)

        if not detection.biased or detection.confidence < 0.5:
            return AntiBiasResult(
                original_response=response, final_response=response,
                bias_detected=False, detection=detection, rewritten=False,
            )

        if self._generate is None:
            return AntiBiasResult(
                original_response=response, final_response=response,
                bias_detected=True, detection=detection, rewritten=False,
            )

        rewrite = await self._generate(DEFRAME_PROMPT.format(
            bias_type=detection.bias_type or "unknown",
            prompt=prompt, response=response, explanation=detection.explanation,
        ))

        logger.info("DeFrame rewrite triggered for %s bias", detection.bias_type)
        return AntiBiasResult(
            original_response=response, final_response=rewrite,
            bias_detected=True, detection=detection, rewritten=True,
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the AntiBiasPipeline."""
    confidence_threshold: float = 0.5
    max_retries: int = 1
    log_path: str = "results/antibias-decisions.json"
    preview_length: int = 80


class AntiBiasPipeline:
    """Orchestrator that runs RBD on every response and triggers DeFrame if flagged.

    Logs all decisions to structured JSON for audit and analysis.
    """

    def __init__(
        self,
        detector: ReasoningBiasDetector,
        generate_fn: Any = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._orchestrator = AntiBiasOrchestrator(detector, generate_fn)
        self._decisions: list[PipelineDecision] = []

    @property
    def decisions(self) -> list[PipelineDecision]:
        """Return all logged decisions."""
        return list(self._decisions)

    @property
    def stats(self) -> dict[str, Any]:
        """Summary statistics of pipeline decisions."""
        total = len(self._decisions)
        flagged = sum(1 for d in self._decisions if d.bias_detected)
        rewritten = sum(1 for d in self._decisions if d.rewritten)
        return {
            "total_processed": total,
            "flagged": flagged,
            "rewritten": rewritten,
            "flag_rate": flagged / total if total else 0.0,
            "rewrite_rate": rewritten / total if total else 0.0,
        }

    async def process(self, prompt: str, response: str) -> AntiBiasResult:
        """Run RBD check on a response, trigger DeFrame if biased, log decision."""
        t0 = time.perf_counter()
        result = await self._orchestrator.check_and_fix(prompt, response)
        latency_ms = (time.perf_counter() - t0) * 1000

        preview = prompt[:self._config.preview_length]
        decision = PipelineDecision(
            timestamp=datetime.now().isoformat(),
            prompt_preview=preview,
            bias_detected=result.bias_detected,
            bias_type=result.detection.bias_type,
            confidence=result.detection.confidence,
            rewritten=result.rewritten,
            latency_ms=round(latency_ms, 2),
        )
        self._decisions.append(decision)

        if result.bias_detected:
            logger.warning(
                "Bias detected (%s, conf=%.2f) in prompt: %s...",
                result.detection.bias_type, result.detection.confidence, preview,
            )
        return result

    def flush_log(self, path: str | None = None) -> Path:
        """Write all decisions to JSON file."""
        out = Path(path or self._config.log_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        records = [
            {
                "timestamp": d.timestamp,
                "prompt_preview": d.prompt_preview,
                "bias_detected": d.bias_detected,
                "bias_type": d.bias_type,
                "confidence": d.confidence,
                "rewritten": d.rewritten,
                "latency_ms": d.latency_ms,
            }
            for d in self._decisions
        ]
        out.write_text(json.dumps(records, indent=2))
        logger.info("Flushed %d decisions to %s", len(records), out)
        return out

    def clear(self) -> None:
        """Clear all logged decisions."""
        self._decisions.clear()
