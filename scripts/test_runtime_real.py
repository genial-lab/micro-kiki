#!/usr/bin/env python3
"""Test MoE-LoRA runtime on real Qwen3.5-4B model with trained adapters.

Loads base model, patches with domain adapters, generates text, and measures
logit deltas to confirm per-token routing produces visible differences.

Usage (on Studio):
    /opt/homebrew/bin/python3.12 scripts/test_runtime_real.py

Expects:
    - Base model: /Users/clems/KIKI-Mac_tunner/models/Qwen3.5-4B
    - Adapters: /Users/clems/KIKI-Mac_tunner/output/micro-kiki/stacks-v3-r16/<domain>/
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate

from src.serving.moe_lora_runtime import MoELoRARuntime, MoELoRAConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("test_runtime_real")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_MODEL = "/Users/clems/KIKI-Mac_tunner/models/Qwen3.5-4B"
ADAPTER_ROOT = "/Users/clems/KIKI-Mac_tunner/output/micro-kiki/stacks-v3-r16"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = [
    {
        "id": "python_merge",
        "text": "Write a Python function to merge two sorted lists into one sorted list.",
        "expected_domain": "python",
    },
    {
        "id": "esp32_uart",
        "text": "Explain UART configuration on ESP32, including baud rate and pin mapping.",
        "expected_domain": "embedded",
    },
    {
        "id": "ams1117_pinout",
        "text": "What is the pinout of the AMS1117-3.3 voltage regulator?",
        "expected_domain": "electronics",
    },
]


def check_adapter_health(runtime: MoELoRARuntime) -> dict:
    """Check adapter weight magnitudes to detect untrained adapters."""
    zero_b = 0
    nonzero_b = 0
    total_a_mag = 0.0
    total_b_mag = 0.0
    count = 0

    for key, proj in runtime._projections.items():
        for i in range(proj.num_experts):
            a = proj.lora_a[i]
            b = proj.lora_b[i]
            mx.eval(a, b)
            a_mag = mx.max(mx.abs(a)).item()
            b_mag = mx.max(mx.abs(b)).item()
            total_a_mag += a_mag
            total_b_mag += b_mag
            count += 1
            if b_mag == 0:
                zero_b += 1
            else:
                nonzero_b += 1

    return {
        "total_experts": count,
        "zero_lora_b": zero_b,
        "nonzero_lora_b": nonzero_b,
        "avg_lora_a_max": total_a_mag / max(count, 1),
        "avg_lora_b_max": total_b_mag / max(count, 1),
        "is_trained": nonzero_b > 0,
    }


def format_prompt(text: str, tokenizer: object) -> str:
    """Format prompt using chat template with thinking disabled."""
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def compute_logit_stats(
    model: object,
    tokenizer: object,
    prompt_text: str,
) -> dict:
    """Compute logit statistics for a prompt (mean, std, top-5 token probs).

    Returns dict with logit stats for the first predicted token.
    """
    formatted = format_prompt(prompt_text, tokenizer)
    tokens = tokenizer.encode(formatted)
    x = mx.array([tokens])

    # Forward pass to get logits
    logits = model(x)  # (1, seq_len, vocab_size)
    mx.eval(logits)

    # Take logits for the last position (next token prediction)
    last_logits = logits[0, -1, :]  # (vocab_size,)

    # Compute stats
    mean_val = mx.mean(last_logits).item()
    std_val = mx.sqrt(mx.mean((last_logits - mean_val) ** 2)).item()
    max_val = mx.max(last_logits).item()

    # Top-5 tokens
    top5_idx = mx.argsort(last_logits)[-5:]
    top5_logits = last_logits[top5_idx]
    mx.eval(top5_idx, top5_logits)

    top5_tokens = []
    for i in range(5):
        idx = top5_idx[4 - i].item()
        logit = top5_logits[4 - i].item()
        token_str = tokenizer.decode([idx])
        top5_tokens.append((token_str, idx, logit))

    return {
        "mean": mean_val,
        "std": std_val,
        "max": max_val,
        "top5": top5_tokens,
    }


def generate_response(
    runtime: MoELoRARuntime,
    prompt_text: str,
    max_tokens: int = 200,
) -> tuple[str, float]:
    """Generate text and return (response, elapsed_seconds)."""
    formatted = format_prompt(prompt_text, runtime.tokenizer)
    t0 = time.monotonic()
    response = mlx_generate(
        runtime.model,
        runtime.tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
    )
    elapsed = time.monotonic() - t0
    return response, elapsed


def print_separator(title: str) -> None:
    """Print a visual separator."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_logit_comparison(
    label: str,
    base_stats: dict,
    patched_stats: dict,
) -> None:
    """Print logit delta comparison."""
    delta_mean = patched_stats["mean"] - base_stats["mean"]
    delta_std = patched_stats["std"] - base_stats["std"]
    delta_max = patched_stats["max"] - base_stats["max"]

    print(f"\n  Logit deltas ({label}):")
    print(f"    mean:  {base_stats['mean']:+.4f} -> {patched_stats['mean']:+.4f}  (delta: {delta_mean:+.4f})")
    print(f"    std:   {base_stats['std']:.4f} -> {patched_stats['std']:.4f}  (delta: {delta_std:+.4f})")
    print(f"    max:   {base_stats['max']:+.4f} -> {patched_stats['max']:+.4f}  (delta: {delta_max:+.4f})")

    print(f"    Base top-5:    ", end="")
    for tok, idx, logit in base_stats["top5"]:
        print(f"'{tok}'({logit:.2f}) ", end="")
    print()

    print(f"    Patched top-5: ", end="")
    for tok, idx, logit in patched_stats["top5"]:
        print(f"'{tok}'({logit:.2f}) ", end="")
    print()


def main() -> None:
    """Run the real-model MoE-LoRA runtime test."""

    print_separator("MoE-LoRA Runtime — Real Model Test")
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  Adapter root: {ADAPTER_ROOT}")
    print(f"  Backend: MLX")

    # ------------------------------------------------------------------
    # 1. Load base model
    # ------------------------------------------------------------------
    print_separator("Step 1: Loading base model")
    config = MoELoRAConfig(
        rank=16,
        alpha=32.0,
        top_k=2,
        num_experts=4,
        router_hidden=64,
    )
    runtime = MoELoRARuntime(config=config)

    t0 = time.monotonic()
    runtime.load_base_model(BASE_MODEL)
    load_time = time.monotonic() - t0
    print(f"  Model loaded in {load_time:.1f}s")
    print(f"  Runtime info: {runtime.info()}")

    # ------------------------------------------------------------------
    # 2. Baseline: generate without adapter
    # ------------------------------------------------------------------
    print_separator("Step 2: Baseline generation (no adapter)")
    baseline_stats = {}

    for prompt in PROMPTS:
        print(f"\n  --- Prompt: {prompt['id']} ---")
        print(f"  {prompt['text']}")

        # Generate
        response, elapsed = generate_response(runtime, prompt["text"])
        print(f"  Response ({elapsed:.1f}s):")
        print(f"    {response[:300]}{'...' if len(response) > 300 else ''}")

        # Logit stats
        stats = compute_logit_stats(runtime.model, runtime.tokenizer, prompt["text"])
        baseline_stats[prompt["id"]] = stats

    # ------------------------------------------------------------------
    # 3. Load python adapter, re-run all prompts
    # ------------------------------------------------------------------
    print_separator("Step 3: Python adapter loaded")
    python_adapter = Path(ADAPTER_ROOT) / "python"
    t0 = time.monotonic()
    patched = runtime.load_adapter(python_adapter)
    patch_time = time.monotonic() - t0
    print(f"  Patched {patched} projections in {patch_time*1000:.0f}ms")
    print(f"  Adapter: {runtime.current_adapter}")

    # Health check
    health = check_adapter_health(runtime)
    print(f"  Adapter health: {health}")
    if not health["is_trained"]:
        print(f"\n  WARNING: All lora_b tensors are zero!")
        print(f"  This adapter was never trained (B=0 init, never updated).")
        print(f"  The MoE-LoRA runtime is working correctly but the adapter")
        print(f"  weights produce zero delta. Training must be re-run.")
        print(f"  (lora_a avg max: {health['avg_lora_a_max']:.6f}, "
              f"lora_b avg max: {health['avg_lora_b_max']:.6f})")

    python_stats = {}
    for prompt in PROMPTS:
        print(f"\n  --- Prompt: {prompt['id']} (python adapter) ---")

        # Generate
        response, elapsed = generate_response(runtime, prompt["text"])
        print(f"  Response ({elapsed:.1f}s):")
        print(f"    {response[:300]}{'...' if len(response) > 300 else ''}")

        # Logit stats
        stats = compute_logit_stats(runtime.model, runtime.tokenizer, prompt["text"])
        python_stats[prompt["id"]] = stats

        # Compare
        print_logit_comparison(
            f"base vs python ({prompt['id']})",
            baseline_stats[prompt["id"]],
            stats,
        )

    # ------------------------------------------------------------------
    # 4. Hot-swap to embedded adapter, re-run ESP32 prompt
    # ------------------------------------------------------------------
    print_separator("Step 4: Hot-swap to embedded adapter")
    embedded_adapter = Path(ADAPTER_ROOT) / "embedded"
    t0 = time.monotonic()
    patched = runtime.load_adapter(embedded_adapter)
    swap_time = time.monotonic() - t0
    print(f"  Hot-swapped: {patched} projections in {swap_time*1000:.0f}ms")
    print(f"  Adapter: {runtime.current_adapter}")

    health = check_adapter_health(runtime)
    print(f"  Adapter health: {health}")
    if not health["is_trained"]:
        print(f"  WARNING: Embedded adapter also untrained (all lora_b = 0)")

    esp32_prompt = PROMPTS[1]  # ESP32 UART prompt
    print(f"\n  --- Prompt: {esp32_prompt['id']} (embedded adapter) ---")

    response, elapsed = generate_response(runtime, esp32_prompt["text"])
    print(f"  Response ({elapsed:.1f}s):")
    print(f"    {response[:300]}{'...' if len(response) > 300 else ''}")

    embedded_stats = compute_logit_stats(
        runtime.model, runtime.tokenizer, esp32_prompt["text"]
    )

    print_logit_comparison(
        "base vs embedded (esp32_uart)",
        baseline_stats["esp32_uart"],
        embedded_stats,
    )
    print_logit_comparison(
        "python vs embedded (esp32_uart)",
        python_stats["esp32_uart"],
        embedded_stats,
    )

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print_separator("Summary: Logit Deltas (mean absolute)")

    print(f"\n  {'Prompt':<20} {'Base mean':>10} {'Python':>10} {'Embedded':>10} {'Py delta':>10} {'Em delta':>10}")
    print(f"  {'-'*70}")

    for prompt in PROMPTS:
        pid = prompt["id"]
        b = baseline_stats[pid]["mean"]
        p = python_stats[pid]["mean"]
        e_val = embedded_stats.get("mean", 0) if pid == "esp32_uart" else None

        py_delta = p - b
        em_delta = (embedded_stats["mean"] - b) if pid == "esp32_uart" else None

        em_str = f"{e_val:+.4f}" if e_val is not None else "n/a"
        em_d_str = f"{em_delta:+.4f}" if em_delta is not None else "n/a"

        print(f"  {pid:<20} {b:+10.4f} {p:+10.4f} {em_str:>10} {py_delta:+10.4f} {em_d_str:>10}")

    print(f"\n  Expected: python adapter -> large delta on python_merge, small on others")
    print(f"  Expected: embedded adapter -> large delta on esp32_uart")

    # Verdict
    py_merge_delta = abs(python_stats["python_merge"]["mean"] - baseline_stats["python_merge"]["mean"])
    py_esp32_delta = abs(python_stats["esp32_uart"]["mean"] - baseline_stats["esp32_uart"]["mean"])
    em_esp32_delta = abs(embedded_stats["mean"] - baseline_stats["esp32_uart"]["mean"])

    print(f"\n  Python adapter delta on python_merge: {py_merge_delta:.4f}")
    print(f"  Python adapter delta on esp32_uart:   {py_esp32_delta:.4f}")
    print(f"  Embedded adapter delta on esp32_uart:  {em_esp32_delta:.4f}")

    if py_merge_delta > 0.001 or em_esp32_delta > 0.001:
        print("\n  PASS: MoE-LoRA routing produces measurable logit changes")
    else:
        print("\n  WARNING: Logit deltas are very small — adapter may not be effective")

    print(f"\n{'='*70}")
    print("  Test complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
