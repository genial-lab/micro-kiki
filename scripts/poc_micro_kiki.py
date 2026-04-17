#!/usr/bin/env python3
"""micro-kiki POC — Base 35B vs Niche LoRA comparison.

Loads the base model, runs prompts through base AND niche adapter,
shows side-by-side quality difference.

Usage:
    python3 scripts/poc_micro_kiki.py
    python3 scripts/poc_micro_kiki.py --domain spice --prompts 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.chdir("/Users/clems/micro-kiki")

# Metal limits
import mlx.core as mx
mx.set_memory_limit(460 * 1024**3)
mx.set_cache_limit(32 * 1024**3)

from mlx_lm import load, generate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "models/qwen3.5-35b-a3b"
ADAPTERS = {
    "spice": "outputs/stacks/stack-spice/adapters.safetensors",
    "emc": "outputs/stacks/stack-emc/adapters.safetensors",
    "stm32": "outputs/stacks/stack-stm32/adapters.safetensors",
    "embedded": "outputs/stacks/stack-embedded/adapters.safetensors",
}

POC_PROMPTS = {
    "spice": [
        "Write a SPICE netlist for a Sallen-Key 2nd order Butterworth low-pass filter at 10kHz with unity gain.",
        "Create an ngspice Monte Carlo analysis for a voltage divider with 5% tolerance resistors.",
        "Design a SPICE subcircuit for a MOSFET H-bridge with dead-time control.",
        "Write a SPICE transient simulation for a synchronous buck converter at 500kHz.",
        "Create a SPICE model for a flyback transformer with leakage inductance k=0.95.",
    ],
    "emc": [
        "Design a common-mode choke filter for USB 3.0 that meets CISPR 32 Class B.",
        "Calculate the shielding effectiveness of a 1mm aluminum enclosure at 300MHz.",
        "Write the PCB layout rules for minimizing radiated emissions from a 100MHz differential clock.",
        "Design a Pi-filter for DC power input to pass MIL-STD-461G CE102.",
        "Calculate return current path impact when a high-speed signal crosses a split plane.",
    ],
    "stm32": [
        "Write STM32 HAL code for configuring TIM1 in PWM mode on channel 1 at 20kHz.",
        "Create STM32 initialization code for SPI1 in full-duplex master mode at 10MHz.",
        "Write STM32 HAL code for DMA-based ADC conversion on 4 channels with circular buffer.",
        "Design STM32 clock configuration for STM32H743 at 480MHz with PLL cascade.",
        "Write STM32 HAL code for I2C communication with BME280 temperature sensor.",
    ],
    "embedded": [
        "Write a bare-metal SPI driver for MAX31855 thermocouple on ARM Cortex-M4.",
        "Design a circular buffer implementation in C for UART receive interrupt handler.",
        "Write a FreeRTOS task synchronization using semaphores for producer-consumer pattern.",
        "Create an interrupt-safe ring buffer for passing data between ISR and main loop.",
        "Write a PID controller in fixed-point Q15 arithmetic for motor speed control.",
    ],
}


def run_prompt(model, tokenizer, prompt: str, max_tokens: int = 512) -> tuple[str, float]:
    """Generate response and return (text, latency_ms)."""
    chat = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    t0 = time.time()
    response = generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False)
    latency = (time.time() - t0) * 1000
    return response, latency


def main() -> None:
    parser = argparse.ArgumentParser(description="micro-kiki POC: base vs niche LoRA")
    parser.add_argument("--domain", default="spice", choices=list(ADAPTERS.keys()))
    parser.add_argument("--prompts", type=int, default=3, help="Number of prompts to test")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--all-domains", action="store_true", help="Test all 4 domains")
    args = parser.parse_args()

    domains = list(ADAPTERS.keys()) if args.all_domains else [args.domain]

    # Load base model
    logger.info("Loading base model: %s", MODEL)
    t0 = time.time()
    model, tokenizer = load(MODEL)
    logger.info("Base model loaded in %.1fs", time.time() - t0)

    results = {}

    for domain in domains:
        adapter_path = ADAPTERS[domain]
        if not Path(adapter_path).exists():
            logger.warning("SKIP %s: adapter not found at %s", domain, adapter_path)
            continue

        prompts = POC_PROMPTS.get(domain, [])[:args.prompts]
        logger.info("\n" + "=" * 60)
        logger.info("DOMAIN: %s (%d prompts)", domain, len(prompts))
        logger.info("=" * 60)

        domain_results = []

        for i, prompt in enumerate(prompts):
            logger.info("\n--- Prompt %d/%d ---", i + 1, len(prompts))
            logger.info("Q: %s", prompt[:80])

            # Base response
            logger.info("\n[BASE 35B]")
            base_resp, base_lat = run_prompt(model, tokenizer, prompt, args.max_tokens)
            logger.info("  Latency: %.0f ms, Length: %d chars", base_lat, len(base_resp))
            logger.info("  Response: %s...", base_resp[:200])

            # Load adapter
            logger.info("\n[NICHE LoRA: %s]", domain)
            model_lora, _ = load(MODEL, adapter_path=str(Path(adapter_path).parent))
            lora_resp, lora_lat = run_prompt(model_lora, tokenizer, prompt, args.max_tokens)
            logger.info("  Latency: %.0f ms, Length: %d chars", lora_lat, len(lora_resp))
            logger.info("  Response: %s...", lora_resp[:200])

            # Compare
            base_len = len(base_resp)
            lora_len = len(lora_resp)
            logger.info("\n  COMPARISON:")
            logger.info("    Base: %d chars, %.0f ms", base_len, base_lat)
            logger.info("    LoRA: %d chars, %.0f ms", lora_len, lora_lat)
            logger.info("    Delta: %+d chars, %+.0f ms", lora_len - base_len, lora_lat - base_lat)

            domain_results.append({
                "prompt": prompt,
                "base_length": base_len,
                "lora_length": lora_len,
                "base_latency_ms": base_lat,
                "lora_latency_ms": lora_lat,
                "base_response": base_resp[:500],
                "lora_response": lora_resp[:500],
            })

            # Free adapter
            del model_lora

        results[domain] = domain_results

    # Save results
    out = Path("results/poc-micro-kiki.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("\nResults saved to %s", out)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("POC SUMMARY")
    logger.info("=" * 60)
    for domain, dr in results.items():
        avg_base = sum(r["base_length"] for r in dr) / len(dr)
        avg_lora = sum(r["lora_length"] for r in dr) / len(dr)
        logger.info("  %s: base avg %.0f chars, lora avg %.0f chars (%+.0f%%)",
                     domain, avg_base, avg_lora, (avg_lora - avg_base) / max(1, avg_base) * 100)


if __name__ == "__main__":
    main()
