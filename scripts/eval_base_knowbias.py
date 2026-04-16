#!/usr/bin/env python3
"""Evaluate base model perplexity before/after KnowBias debiasing.

Framework script: runs with mocked models when no real model available.

Usage:
    uv run python scripts/eval_base_knowbias.py --base-model models/qwen3.5-35b-a3b --debiased-model models/qwen3.5-35b-a3b-debiased
    uv run python scripts/eval_base_knowbias.py --help
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

HELD_OUT_SAMPLES = [
    "The capital of France is Paris, known for the Eiffel Tower.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The Pythagorean theorem states that a^2 + b^2 = c^2.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "Shakespeare wrote Romeo and Juliet in the late 16th century.",
    "The speed of light in vacuum is approximately 3 x 10^8 m/s.",
    "DNA carries genetic information in all living organisms.",
    "The Industrial Revolution began in Britain in the late 18th century.",
    "Ohm's law defines the relationship between voltage, current, and resistance.",
    "Machine learning models learn patterns from data without explicit programming.",
]


@dataclass(frozen=True)
class EvalConfig:
    base_model: str
    debiased_model: str
    output: str = "results/knowbias-eval.json"
    max_samples: int = 0  # 0 = all


@dataclass(frozen=True)
class PerplexityResult:
    model_path: str
    perplexity: float
    num_samples: int
    status: str


def _model_exists(path: str) -> bool:
    """Check if model path exists locally or looks like a HF repo ID."""
    if Path(path).exists():
        return True
    if "/" in path and not path.startswith("/"):
        return True  # HF-style repo ID
    return False


def compute_perplexity_mock(model_path: str, samples: list[str]) -> PerplexityResult:
    """Mock perplexity computation for framework validation.

    Returns deterministic fake values based on model path hash.
    """
    seed = sum(ord(c) for c in model_path) % 1000
    base_ppl = 5.0 + (seed % 50) / 10.0
    logger.info(
        "Mock perplexity for %s: %.2f (model not loaded, framework mode)",
        model_path, base_ppl,
    )
    return PerplexityResult(
        model_path=model_path,
        perplexity=base_ppl,
        num_samples=len(samples),
        status="mock",
    )


def compute_perplexity_real(model_path: str, samples: list[str]) -> PerplexityResult:
    """Compute real perplexity using torch + transformers.

    Requires: torch, transformers installed and model downloaded.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.warning("torch/transformers not available, falling back to mock")
        return compute_perplexity_mock(model_path, samples)

    logger.info("Loading model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for sample in samples:
        inputs = tokenizer(sample, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss)

    return PerplexityResult(
        model_path=model_path,
        perplexity=ppl,
        num_samples=len(samples),
        status="real",
    )


def run_eval(config: EvalConfig) -> dict:
    """Run perplexity comparison between base and debiased models."""
    samples = HELD_OUT_SAMPLES
    if config.max_samples > 0:
        samples = samples[: config.max_samples]

    base_available = _model_exists(config.base_model)
    debiased_available = _model_exists(config.debiased_model)

    if base_available and debiased_available:
        logger.info("Both models found, running real evaluation")
        compute = compute_perplexity_real
    else:
        missing = []
        if not base_available:
            missing.append(config.base_model)
        if not debiased_available:
            missing.append(config.debiased_model)
        logger.warning("Model(s) not found: %s — using mock mode", missing)
        compute = compute_perplexity_mock

    base_result = compute(config.base_model, samples)
    debiased_result = compute(config.debiased_model, samples)

    delta = debiased_result.perplexity - base_result.perplexity
    delta_pct = (delta / base_result.perplexity * 100) if base_result.perplexity > 0 else 0.0

    report = {
        "timestamp": datetime.now().isoformat(),
        "base": {
            "model": base_result.model_path,
            "perplexity": round(base_result.perplexity, 4),
            "num_samples": base_result.num_samples,
            "status": base_result.status,
        },
        "debiased": {
            "model": debiased_result.model_path,
            "perplexity": round(debiased_result.perplexity, 4),
            "num_samples": debiased_result.num_samples,
            "status": debiased_result.status,
        },
        "delta": round(delta, 4),
        "delta_pct": round(delta_pct, 2),
        "capacity_preserved": abs(delta_pct) < 5.0,
    }

    logger.info(
        "Base PPL=%.4f  Debiased PPL=%.4f  Delta=%.4f (%.2f%%)",
        base_result.perplexity, debiased_result.perplexity, delta, delta_pct,
    )
    if report["capacity_preserved"]:
        logger.info("PASS: Capacity preserved (delta < 5%%)")
    else:
        logger.warning("WARN: Capacity degradation detected (delta >= 5%%)")

    return report


def save_results(report: dict, output: str) -> Path:
    """Save eval report as JSON."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    logger.info("Results saved to %s", path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base model perplexity before/after KnowBias debiasing",
    )
    parser.add_argument(
        "--base-model", required=True,
        help="Path or HF repo ID for the base model",
    )
    parser.add_argument(
        "--debiased-model", required=True,
        help="Path or HF repo ID for the debiased model",
    )
    parser.add_argument(
        "--output", default="results/knowbias-eval.json",
        help="Output JSON path (default: results/knowbias-eval.json)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Limit held-out samples (0 = all, default: 0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()
    config = EvalConfig(
        base_model=args.base_model,
        debiased_model=args.debiased_model,
        output=args.output,
        max_samples=args.max_samples,
    )
    report = run_eval(config)
    save_results(report, config.output)
