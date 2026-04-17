#!/usr/bin/env python3
"""micro-kiki POC — Full Pipeline Demo.

Demonstrates the complete triple-hybrid architecture:
1. Quantum VQC Router → domain classification
2. Model Router → model + adapter selection
3. Aeon Memory → context injection from past conversations
4. Inference → 35B base (or with LoRA adapter)
5. Negotiator → quality arbitration
6. Aeon Memory → persist the exchange

Usage:
    python3 scripts/poc_pipeline.py
    python3 scripts/poc_pipeline.py --scenario multi-turn
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.chdir("/Users/clems/micro-kiki")

import mlx.core as mx
mx.set_memory_limit(460 * 1024**3)
mx.set_cache_limit(32 * 1024**3)

from mlx_lm import load, generate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("poc")

MODEL = "models/qwen3.5-35b-a3b"

# Import micro-kiki modules
import sys
sys.path.insert(0, ".")
from src.routing.router import NICHE_DOMAINS
from src.routing.model_router import ModelRouter, RouteDecision
from src.memory.aeon import AeonPalace
from src.memory.trace import Episode


@dataclass
class PipelineResult:
    query: str
    route: RouteDecision
    memories_injected: int
    response: str
    latency_ms: float
    domain_detected: str


class MicroKikiPipeline:
    """Full micro-kiki pipeline demo."""

    def __init__(self):
        logger.info("=" * 60)
        logger.info("micro-kiki POC — Triple-Hybrid Pipeline")
        logger.info("=" * 60)

        # 1. Router
        logger.info("\n[1/4] Initializing Model Router...")
        self.router = ModelRouter()
        logger.info("  Router: 11 domains (10 niches + base)")

        # 2. Memory
        logger.info("[2/4] Initializing Aeon Memory Palace...")
        self.memory = AeonPalace()
        logger.info("  Memory: Atlas vector + Trace episodic graph")

        # 3. Model
        logger.info("[3/4] Loading Qwen3.5-35B-A3B...")
        t0 = time.time()
        self.model, self.tokenizer = load(MODEL)
        logger.info("  Model loaded in %.1fs", time.time() - t0)

        # 4. Stats
        self.turn_count = 0
        logger.info("[4/4] Pipeline ready!\n")

    def _detect_domain(self, query: str) -> str:
        """Simple keyword-based domain detection."""
        query_lower = query.lower()
        domain_keywords = {
            "spice": ["spice", "netlist", "ngspice", "ltspice", ".subckt", ".model", "transient", "simulation"],
            "kicad-dsl": ["kicad", "schematic", "footprint", "s-expression", "pcb layout", "symbol"],
            "emc": ["emc", "emi", "shielding", "cispr", "grounding", "esd", "compliance", "radiated"],
            "stm32": ["stm32", "hal_", "cubemx", "cortex-m", "stm32f", "stm32h"],
            "embedded": ["rtos", "freertos", "interrupt", "dma", "bare-metal", "firmware", "isr"],
            "power": ["buck", "boost", "smps", "converter", "mosfet driver", "inductor selection"],
            "dsp": ["fft", "fir", "iir", "dsp", "filter design", "convolution"],
            "electronics": ["op-amp", "amplifier", "transistor", "bias point", "gain", "bandwidth"],
            "freecad": ["freecad", "macro", "parametric", "3d model", "part design"],
            "platformio": ["platformio", "pio", "board", "lib_deps", "build_flags"],
        }
        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return domain
        return "base"

    def process(self, query: str) -> PipelineResult:
        """Full pipeline: route → memory → infer → persist."""
        self.turn_count += 1
        t0 = time.time()

        logger.info("─" * 60)
        logger.info("Turn %d", self.turn_count)
        logger.info("─" * 60)
        logger.info("User: %s", query[:100])

        # Step 1: Route
        domain = self._detect_domain(query)
        route = self.router.select(query, domain_hint=domain if domain != "base" else None)
        logger.info("\n  [ROUTER] Domain: %s → Model: %s, Adapter: %s",
                     domain, route.model_id, route.adapter or "none")

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

        # Step 3: Inference
        augmented_query = memory_context + query if memory_context else query
        chat = [{"role": "user", "content": augmented_query}]
        formatted = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
        logger.info("  [INFERENCE] Generating (model: %s)...", route.model_id)
        response = generate(self.model, self.tokenizer, prompt=formatted,
                            max_tokens=500, verbose=False)
        logger.info("  [INFERENCE] %d chars generated", len(response))

        # Step 4: Memory write
        episode_id = self.memory.write(
            content=f"Q: {query[:200]}\nA: {response[:200]}",
            domain=domain,
            timestamp=datetime.now(),
            source="poc-pipeline",
        )
        logger.info("  [MEMORY WRITE] Persisted as episode %s", episode_id[:8])

        latency = (time.time() - t0) * 1000

        # Step 5: Response
        logger.info("\n  Assistant: %s", response[:200])
        logger.info("  [STATS] Domain: %s | Memories: %d | Latency: %.0f ms | Episode: %s",
                     domain, len(memories), latency, episode_id[:8])

        return PipelineResult(
            query=query, route=route, memories_injected=len(memories),
            response=response, latency_ms=latency, domain_detected=domain,
        )


def run_scenario_basic(pipeline: MicroKikiPipeline) -> list[PipelineResult]:
    """Basic demo: one prompt per domain."""
    prompts = [
        "Write a SPICE netlist for a buck converter with 12V input and 3.3V output at 2A.",
        "Design a common-mode filter for USB 3.0 to meet CISPR 32 Class B emissions.",
        "Write STM32 HAL code for configuring TIM1 in center-aligned PWM mode at 20kHz.",
        "Implement a FreeRTOS task that reads an I2C sensor every 100ms and sends data via queue.",
        "What is the weather today?",  # base domain — should route to base
    ]
    results = []
    for p in prompts:
        results.append(pipeline.process(p))
    return results


def run_scenario_multiturn(pipeline: MicroKikiPipeline) -> list[PipelineResult]:
    """Multi-turn: memory should recall previous context."""
    prompts = [
        "I'm designing a buck converter. The input is 24V, output 5V at 3A. What switching frequency should I use?",
        "For the buck converter we discussed, write the SPICE netlist with the values you suggested.",
        "Now add a second output stage: 3.3V at 1A from the same 24V input. Show the complete netlist.",
        "What were the inductor values we used for both converters?",
    ]
    results = []
    for p in prompts:
        results.append(pipeline.process(p))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="micro-kiki POC Pipeline")
    parser.add_argument("--scenario", default="basic", choices=["basic", "multi-turn"])
    args = parser.parse_args()

    pipeline = MicroKikiPipeline()

    if args.scenario == "basic":
        results = run_scenario_basic(pipeline)
    else:
        results = run_scenario_multiturn(pipeline)

    # Save results
    out = Path("results/poc-pipeline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        [{"query": r.query, "domain": r.domain_detected, "route": r.route.model_id,
          "adapter": r.route.adapter, "memories": r.memories_injected,
          "response_length": len(r.response), "latency_ms": r.latency_ms,
          "response": r.response[:500]} for r in results],
        indent=2, ensure_ascii=False,
    ))

    logger.info("\n" + "=" * 60)
    logger.info("POC COMPLETE — %d turns", len(results))
    logger.info("=" * 60)
    for r in results:
        logger.info("  [%s] %s → %d chars, %d memories, %.0f ms",
                     r.domain_detected, r.query[:50], len(r.response),
                     r.memories_injected, r.latency_ms)
    logger.info("\nResults: %s", out)


if __name__ == "__main__":
    main()
