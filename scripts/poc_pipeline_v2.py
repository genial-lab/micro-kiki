#!/usr/bin/env python3
"""micro-kiki POC v2 — VQC Router + Negotiator + 10-Domain Benchmark.

Extends poc_pipeline.py with:
1. Quantum VQC Router (PennyLane) for domain classification,
   with graceful fallback to keyword-based detection.
2. Negotiator quality arbitration: if confidence < 0.8,
   generate a second response and let Negotiator pick the best.
3. 10-domain benchmark scenario (one prompt per niche domain).
4. Multi-turn scenario (4-turn buck converter conversation).
5. Stats summary: per-domain routing, memory recall, average latency.

Usage:
    python3 scripts/poc_pipeline_v2.py --scenario basic
    python3 scripts/poc_pipeline_v2.py --scenario multi-turn
    python3 scripts/poc_pipeline_v2.py --scenario all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean

# ---------------------------------------------------------------------------
# Working directory (same convention as poc_pipeline.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Metal memory limits (Mac Studio M3 Ultra 512 GB)
# ---------------------------------------------------------------------------
import mlx.core as mx
mx.set_memory_limit(460 * 1024**3)
mx.set_cache_limit(32 * 1024**3)

from mlx_lm import load, generate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("poc_v2")

MODEL = "models/qwen3.5-35b-a3b"

# ---------------------------------------------------------------------------
# micro-kiki imports
# ---------------------------------------------------------------------------
from src.routing.router import NICHE_DOMAINS
from src.routing.model_router import ModelRouter, RouteDecision
from src.memory.aeon import AeonPalace
from src.memory.trace import Episode

# ---------------------------------------------------------------------------
# 10-domain test prompts (one per niche, covers all NICHE_DOMAINS)
# ---------------------------------------------------------------------------
DOMAIN_TESTS: dict[str, str] = {
    "kicad-dsl":   "Create a KiCad S-expression for a TQFP-48 footprint with thermal pad.",
    "spice":       "Write a SPICE netlist for a current-mode buck converter at 500kHz.",
    "emc":         "Design an EMI filter for USB 3.0 to meet CISPR 32 Class B.",
    "stm32":       "Write STM32 HAL code for DMA-based ADC on 4 channels.",
    "embedded":    "Implement a circular buffer in C for UART RX interrupt handler.",
    "power":       "Design a 48V to 12V synchronous buck converter at 5A.",
    "dsp":         "Implement a 256-point FFT in fixed-point Q15 for Cortex-M4.",
    "electronics": "Design an instrumentation amplifier with gain=100 using AD620.",
    "freecad":     "Write a FreeCAD macro for a parametric heatsink with fins.",
    "platformio":  "Write platformio.ini for ESP32-S3 + STM32F407 multi-env build.",
}


# ---------------------------------------------------------------------------
# Result dataclass (extended from v1)
# ---------------------------------------------------------------------------

@dataclass
class PipelineResultV2:
    query: str
    route: RouteDecision
    memories_injected: int
    response: str
    latency_ms: float
    domain_detected: str
    expected_domain: str | None        # set during benchmark scenario
    quantum_used: bool = False
    quantum_confidence: float = 0.0
    negotiator_used: bool = False
    negotiator_candidates: int = 0


# ---------------------------------------------------------------------------
# Base pipeline (copy of MicroKikiPipeline — unchanged for backward compat)
# ---------------------------------------------------------------------------

class MicroKikiPipeline:
    """Full micro-kiki pipeline demo (v1 — unchanged)."""

    def __init__(self) -> None:
        logger.info("=" * 60)
        logger.info("micro-kiki POC — Triple-Hybrid Pipeline")
        logger.info("=" * 60)

        logger.info("\n[1/4] Initializing Model Router...")
        self.router = ModelRouter()
        logger.info("  Router: 11 domains (10 niches + base)")

        logger.info("[2/4] Initializing Aeon Memory Palace...")
        self.memory = AeonPalace()
        logger.info("  Memory: Atlas vector + Trace episodic graph")

        logger.info("[3/4] Loading Qwen3.5-35B-A3B...")
        t0 = time.time()
        self.model, self.tokenizer = load(MODEL)
        logger.info("  Model loaded in %.1fs", time.time() - t0)

        self.turn_count = 0
        logger.info("[4/4] Pipeline ready!\n")

    def _detect_domain(self, query: str) -> str:
        """Score-based domain detection (v2 — highest score wins).

        Each keyword match adds weight. High-priority keywords (unique to a
        domain) get weight 3, regular keywords get weight 1.  This fixes the
        DSP/stm32 and PlatformIO/stm32 conflicts from v1.
        """
        query_lower = query.lower()

        # (keyword, weight) — weight 3 = domain-defining, 1 = supportive
        domain_keywords: dict[str, list[tuple[str, int]]] = {
            "platformio":  [("platformio", 3), ("platformio.ini", 3), ("pio", 2),
                            ("lib_deps", 2), ("build_flags", 2), ("board", 1),
                            ("multi-env", 2)],
            "kicad-dsl":   [("kicad", 3), ("s-expression", 3), ("footprint", 2),
                            ("schematic", 1), ("pcb layout", 2), ("symbol", 1)],
            "freecad":     [("freecad", 3), ("macro", 1), ("parametric", 1),
                            ("3d model", 1), ("part design", 2), ("heatsink", 1)],
            "dsp":         [("fft", 3), ("fir", 2), ("iir", 2), ("dsp", 3),
                            ("filter design", 2), ("convolution", 2),
                            ("fixed-point", 2), ("q15", 3)],
            "spice":       [("spice", 3), ("netlist", 2), ("ngspice", 3),
                            ("ltspice", 3), (".subckt", 3), (".model", 1),
                            ("transient", 1), ("simulation", 1)],
            "emc":         [("emc", 3), ("emi", 3), ("shielding", 2), ("cispr", 3),
                            ("grounding", 1), ("esd", 2), ("compliance", 1),
                            ("radiated", 2)],
            "stm32":       [("stm32", 3), ("hal_", 3), ("cubemx", 3),
                            ("cortex-m", 1), ("stm32f", 3), ("stm32h", 3)],
            "embedded":    [("rtos", 3), ("freertos", 3), ("interrupt", 1),
                            ("dma", 1), ("bare-metal", 2), ("firmware", 1),
                            ("isr", 2), ("circular buffer", 2), ("uart", 1)],
            "power":       [("buck", 2), ("boost", 2), ("smps", 3),
                            ("converter", 1), ("mosfet driver", 2),
                            ("inductor selection", 2), ("synchronous", 1)],
            "electronics": [("op-amp", 3), ("amplifier", 2), ("transistor", 2),
                            ("bias point", 2), ("gain", 1), ("bandwidth", 1),
                            ("instrumentation", 2)],
        }

        scores: dict[str, int] = {}
        for domain, kw_list in domain_keywords.items():
            score = sum(w for kw, w in kw_list if kw in query_lower)
            if score > 0:
                scores[domain] = score

        if not scores:
            return "base"
        return max(scores, key=scores.__getitem__)

    def _infer(self, augmented_query: str, route: RouteDecision) -> str:
        """Run MLX inference and return the raw response string."""
        chat = [{"role": "user", "content": augmented_query}]
        formatted = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
        logger.info("  [INFERENCE] Generating (model: %s)...", route.model_id)
        response = generate(self.model, self.tokenizer, prompt=formatted,
                            max_tokens=500, verbose=False)
        logger.info("  [INFERENCE] %d chars generated", len(response))
        return response

    def process(self, query: str) -> PipelineResultV2:
        """v1-compatible full pipeline: route → memory → infer → persist."""
        self.turn_count += 1
        t0 = time.time()

        logger.info("─" * 60)
        logger.info("Turn %d", self.turn_count)
        logger.info("─" * 60)
        logger.info("User: %s", query[:100])

        domain = self._detect_domain(query)
        route = self.router.select(query, domain_hint=domain if domain != "base" else None)
        logger.info("  [ROUTER] Domain: %s → Model: %s, Adapter: %s",
                    domain, route.model_id, route.adapter or "none")

        memories = self.memory.recall(query, top_k=3)
        memory_context = ""
        if memories:
            memory_lines = [f"[Memory] {ep.content[:100]}" for ep in memories]
            memory_context = "\n".join(memory_lines) + "\n\n"
            logger.info("  [MEMORY] Recalled %d episodes", len(memories))
        else:
            logger.info("  [MEMORY] No relevant memories found")

        augmented_query = memory_context + query if memory_context else query
        response = self._infer(augmented_query, route)

        episode_id = self.memory.write(
            content=f"Q: {query[:300]}\nA: {response[:600]}",
            domain=domain,
            timestamp=datetime.now(),
            source="poc-pipeline-v2",
        )
        logger.info("  [MEMORY WRITE] Persisted as episode %s", episode_id[:8])

        latency = (time.time() - t0) * 1000
        logger.info("\n  Assistant: %s", response[:200])
        logger.info("  [STATS] Domain: %s | Memories: %d | Latency: %.0f ms | Episode: %s",
                    domain, len(memories), latency, episode_id[:8])

        return PipelineResultV2(
            query=query, route=route, memories_injected=len(memories),
            response=response, latency_ms=latency, domain_detected=domain,
            expected_domain=None,
        )


# ---------------------------------------------------------------------------
# Extended pipeline v2
# ---------------------------------------------------------------------------

class MicroKikiPipelineV2(MicroKikiPipeline):
    """Extended pipeline with VQC router + Negotiator."""

    # Confidence threshold: below this, trigger Negotiator arbitration
    NEGOTIATOR_CONFIDENCE_THRESHOLD = 0.8

    def __init__(self) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Quantum VQC Router (optional — PennyLane required)
        # ------------------------------------------------------------------
        self.quantum_router = None
        try:
            from src.routing.quantum_router import QuantumRouter
            self.quantum_router = QuantumRouter()
            logger.info("[V2] QuantumRouter loaded (PennyLane available)")
        except ImportError as exc:
            logger.warning("[V2] QuantumRouter disabled: %s", exc)

        # ------------------------------------------------------------------
        # Negotiator (required sub-components accept None clients — graceful)
        # ------------------------------------------------------------------
        self.negotiator = None
        try:
            from src.cognitive.argument_extractor import ArgumentExtractor
            from src.cognitive.judge import AdaptiveJudge
            from src.cognitive.catfish import CatfishModule
            from src.cognitive.negotiator import Negotiator

            extractor = ArgumentExtractor(generate_fn=None)   # heuristic mode
            judge = AdaptiveJudge(fast_client=None, deep_client=None)
            catfish = CatfishModule(generate_fn=None)
            self.negotiator = Negotiator(extractor=extractor, judge=judge, catfish=catfish)
            logger.info("[V2] Negotiator loaded (heuristic mode — no external judge client)")
        except ImportError as exc:
            logger.warning("[V2] Negotiator disabled: %s", exc)

        logger.info("[V2] Pipeline v2 ready!\n")

    # ------------------------------------------------------------------
    # Quantum routing helper
    # ------------------------------------------------------------------

    def _quantum_detect_domain(self, query: str) -> tuple[str, float, bool]:
        """Classify domain via VQC router; fall back to keyword detection.

        Returns:
            (domain, confidence, quantum_used)
        """
        if self.quantum_router is not None:
            try:
                from src.routing.hybrid_pipeline import _query_to_embedding
                embedding = _query_to_embedding(query)
                decision = self.quantum_router.route(embedding)

                # Parse confidence from reason string: "quantum-vqc: <domain> (conf=X.XXX)"
                confidence = 0.0
                marker = "conf="
                idx = decision.reason.find(marker)
                if idx != -1:
                    try:
                        confidence = float(decision.reason[idx + len(marker): idx + len(marker) + 5])
                    except ValueError:
                        pass

                # Extract domain from adapter name ("stack-<domain>") or "base"
                if decision.adapter:
                    domain = decision.adapter.replace("stack-", "")
                else:
                    domain = "base"

                logger.info("  [QUANTUM] Domain: %s (conf=%.3f)", domain, confidence)
                return domain, confidence, True

            except Exception as exc:
                logger.warning("  [QUANTUM] Failed (%s) — falling back to keyword", exc)

        # Fallback: keyword-based
        domain = self._detect_domain(query)
        return domain, 0.0, False

    # ------------------------------------------------------------------
    # Negotiator helper (sync wrapper around async negotiate)
    # ------------------------------------------------------------------

    def _run_negotiator(
        self,
        query: str,
        response1: str,
        response2: str,
    ) -> tuple[str, bool, int]:
        """Run async Negotiator in a sync context.

        Returns:
            (winning_response, negotiator_used, num_candidates)
        """
        if self.negotiator is None:
            logger.info("  [NEGOTIATOR] Disabled — keeping first response")
            return response1, False, 0

        async def _negotiate() -> str:
            result = await self.negotiator.negotiate(query, [response1, response2])
            logger.info(
                "  [NEGOTIATOR] %d candidates → winner idx %d (judge: %s)",
                result.num_candidates,
                result.winner_idx,
                result.judge_result.backend_used,
            )
            return result.winner_response

        try:
            winner = asyncio.run(_negotiate())
            return winner, True, 2
        except Exception as exc:
            logger.warning("  [NEGOTIATOR] Arbitration failed (%s) — keeping first response", exc)
            return response1, False, 0

    # ------------------------------------------------------------------
    # Main process override
    # ------------------------------------------------------------------

    def process(self, query: str, expected_domain: str | None = None) -> PipelineResultV2:
        """v2 full pipeline: quantum route → memory → infer → negotiate → persist."""
        self.turn_count += 1
        t0 = time.time()

        logger.info("─" * 60)
        logger.info("Turn %d", self.turn_count)
        logger.info("─" * 60)
        logger.info("User: %s", query[:100])

        # Step 1: Domain classification (quantum or keyword fallback)
        domain, q_confidence, q_used = self._quantum_detect_domain(query)
        route = self.router.select(query, domain_hint=domain if domain != "base" else None)
        logger.info("  [ROUTER] Domain: %s → Model: %s, Adapter: %s (quantum=%s)",
                    domain, route.model_id, route.adapter or "none", q_used)

        # Step 2: Memory recall
        memories = self.memory.recall(query, top_k=3)
        memory_context = ""
        if memories:
            memory_lines = [f"[Memory] {ep.content[:100]}" for ep in memories]
            memory_context = "\n".join(memory_lines) + "\n\n"
            logger.info("  [MEMORY] Recalled %d episodes", len(memories))
            for ep in memories:
                logger.info("    → %s...", ep.content[:60])
        else:
            logger.info("  [MEMORY] No relevant memories found")

        augmented_query = memory_context + query if memory_context else query

        # Step 3: First inference
        response = self._infer(augmented_query, route)

        # Step 4: Negotiator arbitration (if confidence low or quantum not used)
        negotiator_used = False
        negotiator_candidates = 0
        trigger_negotiator = (
            self.negotiator is not None
            and (not q_used or q_confidence < self.NEGOTIATOR_CONFIDENCE_THRESHOLD)
        )

        if trigger_negotiator:
            logger.info(
                "  [NEGOTIATOR] Triggering arbitration (q_used=%s, conf=%.3f < %.2f)",
                q_used, q_confidence, self.NEGOTIATOR_CONFIDENCE_THRESHOLD,
            )
            # Generate a second candidate with a variation in the prompt
            alt_prompt = augmented_query + "\n[Respond with an alternative formulation]"
            response2 = self._infer(alt_prompt, route)
            response, negotiator_used, negotiator_candidates = self._run_negotiator(
                query, response, response2,
            )
        else:
            logger.info(
                "  [NEGOTIATOR] Skipped (q_used=%s, conf=%.3f >= %.2f)",
                q_used, q_confidence, self.NEGOTIATOR_CONFIDENCE_THRESHOLD,
            )

        # Step 5: Memory write
        episode_id = self.memory.write(
            content=f"Q: {query[:300]}\nA: {response[:600]}",
            domain=domain,
            timestamp=datetime.now(),
            source="poc-pipeline-v2",
        )
        logger.info("  [MEMORY WRITE] Persisted as episode %s", episode_id[:8])

        latency = (time.time() - t0) * 1000

        logger.info("\n  Assistant: %s", response[:200])
        logger.info(
            "  [STATS] Domain: %s | Memories: %d | Latency: %.0f ms | "
            "Quantum: %s (conf=%.3f) | Negotiator: %s",
            domain, len(memories), latency,
            q_used, q_confidence,
            negotiator_used,
        )

        return PipelineResultV2(
            query=query,
            route=route,
            memories_injected=len(memories),
            response=response,
            latency_ms=latency,
            domain_detected=domain,
            expected_domain=expected_domain,
            quantum_used=q_used,
            quantum_confidence=q_confidence,
            negotiator_used=negotiator_used,
            negotiator_candidates=negotiator_candidates,
        )


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------

def run_scenario_basic(pipeline: MicroKikiPipelineV2) -> list[PipelineResultV2]:
    """10-domain benchmark: one prompt per niche domain."""
    logger.info("\n" + "=" * 60)
    logger.info("SCENARIO: 10-Domain Benchmark")
    logger.info("=" * 60)
    results: list[PipelineResultV2] = []
    for expected_domain, prompt in DOMAIN_TESTS.items():
        result = pipeline.process(prompt, expected_domain=expected_domain)
        results.append(result)
    return results


def run_scenario_multiturn(pipeline: MicroKikiPipelineV2) -> list[PipelineResultV2]:
    """Multi-turn: memory should recall previous context (buck converter design)."""
    logger.info("\n" + "=" * 60)
    logger.info("SCENARIO: Multi-Turn Buck Converter")
    logger.info("=" * 60)
    prompts = [
        "I'm designing a buck converter. The input is 24V, output 5V at 3A. "
        "What switching frequency should I use?",
        "For the buck converter we discussed, write the SPICE netlist with the values "
        "you suggested.",
        "Now add a second output stage: 3.3V at 1A from the same 24V input. "
        "Show the complete netlist.",
        "What were the inductor values we used for both converters?",
    ]
    results: list[PipelineResultV2] = []
    for p in prompts:
        results.append(pipeline.process(p))
    return results


# ---------------------------------------------------------------------------
# Stats summary
# ---------------------------------------------------------------------------

def print_stats(results: list[PipelineResultV2]) -> dict:
    """Print per-domain routing accuracy, memory recall, and latency stats."""
    logger.info("\n" + "=" * 60)
    logger.info("STATS SUMMARY — %d turns", len(results))
    logger.info("=" * 60)

    # Per-domain routing accuracy (only for results that have expected_domain set)
    benchmark_results = [r for r in results if r.expected_domain is not None]
    if benchmark_results:
        correct = sum(
            1 for r in benchmark_results
            if r.domain_detected == r.expected_domain
        )
        accuracy = correct / len(benchmark_results) * 100
        logger.info("\nDomain routing accuracy: %d/%d (%.0f%%)",
                    correct, len(benchmark_results), accuracy)
        logger.info("%-15s %-15s %-15s %s",
                    "Expected", "Detected", "Correct?", "Query[:40]")
        logger.info("─" * 70)
        for r in benchmark_results:
            ok = "OK" if r.domain_detected == r.expected_domain else "MISS"
            logger.info("%-15s %-15s %-15s %s",
                        r.expected_domain, r.domain_detected, ok, r.query[:40])

    # Memory recall
    total_memories = sum(r.memories_injected for r in results)
    turns_with_memory = sum(1 for r in results if r.memories_injected > 0)
    logger.info(
        "\nMemory recall: %d total injections across %d/%d turns",
        total_memories, turns_with_memory, len(results),
    )

    # Latency
    latencies = [r.latency_ms for r in results]
    logger.info(
        "Latency: avg=%.0f ms, min=%.0f ms, max=%.0f ms",
        mean(latencies), min(latencies), max(latencies),
    )

    # Quantum + Negotiator usage
    q_used = sum(1 for r in results if r.quantum_used)
    neg_used = sum(1 for r in results if r.negotiator_used)
    logger.info("Quantum router used: %d/%d turns", q_used, len(results))
    logger.info("Negotiator triggered: %d/%d turns", neg_used, len(results))

    stats = {
        "total_turns": len(results),
        "routing_accuracy_pct": (
            correct / len(benchmark_results) * 100 if benchmark_results else None
        ),
        "total_memory_injections": total_memories,
        "turns_with_memory": turns_with_memory,
        "avg_latency_ms": mean(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "quantum_used_turns": q_used,
        "negotiator_triggered_turns": neg_used,
    }
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="micro-kiki POC Pipeline v2")
    parser.add_argument(
        "--scenario",
        default="basic",
        choices=["basic", "multi-turn", "all"],
        help="basic = 10-domain benchmark | multi-turn = 4-turn buck converter | "
             "all = both scenarios",
    )
    args = parser.parse_args()

    pipeline = MicroKikiPipelineV2()
    all_results: list[PipelineResultV2] = []

    if args.scenario in ("basic", "all"):
        all_results.extend(run_scenario_basic(pipeline))

    if args.scenario in ("multi-turn", "all"):
        all_results.extend(run_scenario_multiturn(pipeline))

    # Stats summary
    stats = print_stats(all_results)

    # Save results
    out = Path("results/poc-pipeline-v2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {
            "stats": stats,
            "turns": [
                {
                    "query": r.query,
                    "expected_domain": r.expected_domain,
                    "domain": r.domain_detected,
                    "route": r.route.model_id,
                    "adapter": r.route.adapter,
                    "memories": r.memories_injected,
                    "response_length": len(r.response),
                    "latency_ms": r.latency_ms,
                    "quantum_used": r.quantum_used,
                    "quantum_confidence": r.quantum_confidence,
                    "negotiator_used": r.negotiator_used,
                    "negotiator_candidates": r.negotiator_candidates,
                    "response": r.response[:500],
                }
                for r in all_results
            ],
        },
        indent=2,
        ensure_ascii=False,
    ))

    logger.info("\nResults saved: %s", out)


if __name__ == "__main__":
    main()
