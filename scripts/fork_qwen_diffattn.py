"""Fork Qwen3.5-4B with DiffAttn on full-attention layers.

Loads the base model, identifies the full-attention layers (those with
standard Q/K/V projections and no GatedDeltaNet gate), patches them
with DiffAttn via :func:`apply_diff_attention`, then saves the forked
model.  Includes a perplexity rollback check.

Usage::

    uv run python scripts/fork_qwen_diffattn.py
    uv run python scripts/fork_qwen_diffattn.py --base-dir models/qwen3.5-4b/bf16
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PERPLEXITY_THRESHOLD = 0.03
OUTLIER_REDUCTION_MIN = 0.30
DEFAULT_BASE_DIR = "models/qwen3.5-4b/bf16"
HF_FALLBACK = "Qwen/Qwen3.5-4B"
DEFAULT_OUTPUT_DIR = "models/qwen3.5-4b-diffattn"
CALIBRATION_TEXT = (
    "The differential attention mechanism computes two independent "
    "softmax attention maps and subtracts the second from the first, "
    "scaled by a learnable lambda parameter. This cancels shared noise "
    "across attention heads and reduces activation outliers, which is "
    "particularly beneficial for quantized inference at Q4 precision."
)


def identify_full_attention_layers(model: object) -> list[int]:
    """Return indices of full-attention layers (not GatedDeltaNet).

    Full-attention layers have ``q_proj`` on their ``self_attn`` and
    do NOT have a ``gate`` attribute (which marks linear/GatedDeltaNet).
    """
    indices: list[int] = []
    layers = model.model.layers  # type: ignore[attr-defined]
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        has_qkv = hasattr(attn, "q_proj") and hasattr(attn, "k_proj")
        is_gated = hasattr(attn, "gate")
        if has_qkv and not is_gated:
            indices.append(i)
    return indices


def compute_perplexity(
    model: object,
    tokenizer: object,
    text: str,
) -> float:
    """Compute perplexity on a short text for sanity checking."""
    import torch

    inputs = tokenizer(text, return_tensors="pt")  # type: ignore[operator]
    input_ids = inputs["input_ids"].to(
        next(model.parameters()).device  # type: ignore[attr-defined]
    )
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)  # type: ignore[operator]
    return float(torch.exp(outputs.loss).item())


def check_rollback(metrics: dict[str, object]) -> bool:
    """Return True if the fork should be rolled back."""
    ppl_delta = float(metrics.get("perplexity_delta", 0.0))  # type: ignore[arg-type]
    if ppl_delta > PERPLEXITY_THRESHOLD:
        logger.warning(
            "Perplexity delta %.4f exceeds threshold %.4f — rollback recommended",
            ppl_delta,
            PERPLEXITY_THRESHOLD,
        )
        return True
    outlier_reduction = float(metrics.get("outlier_reduction", 1.0))  # type: ignore[arg-type]
    if outlier_reduction < OUTLIER_REDUCTION_MIN:
        logger.warning(
            "Outlier reduction %.4f below minimum %.4f — rollback recommended",
            outlier_reduction,
            OUTLIER_REDUCTION_MIN,
        )
        return True
    return False


def fork_with_diffattn(
    base_dir: str,
    output_dir: str,
    calibration_text: str = CALIBRATION_TEXT,
) -> dict[str, object]:
    """Fork the base model with DiffAttn on full-attention layers.

    Args:
        base_dir: Path to local model or HuggingFace model ID.
        output_dir: Where to save the forked model.
        calibration_text: Short text for perplexity sanity check.

    Returns:
        Metrics dict with layer info, perplexity data, and status.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.base.diff_attention import DiffAttentionConfig, apply_diff_attention

    # Resolve model source
    model_source = base_dir
    if not Path(base_dir).exists():
        logger.info("Local path %s not found, falling back to HuggingFace: %s", base_dir, HF_FALLBACK)
        model_source = HF_FALLBACK

    logger.info("Loading base model from %s", model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Identify full-attention layers
    full_attn_indices = identify_full_attention_layers(model)
    logger.info("Found %d full-attention layers: %s", len(full_attn_indices), full_attn_indices)

    if not full_attn_indices:
        logger.warning("No full-attention layers found — saving unmodified model")
        metrics: dict[str, object] = {
            "base_dir": base_dir,
            "output_dir": output_dir,
            "full_attn_layers": [],
            "full_attn_count": 0,
            "status": "no_layers_found",
        }
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "fork_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
        return metrics

    # Compute baseline perplexity
    logger.info("Computing baseline perplexity...")
    baseline_ppl = compute_perplexity(model, tokenizer, calibration_text)
    logger.info("Baseline perplexity: %.4f", baseline_ppl)

    # Apply DiffAttn
    config = DiffAttentionConfig(
        d_model=model.config.hidden_size,
        num_heads=model.config.num_attention_heads,
        num_layers=len(full_attn_indices),
    )
    patched = apply_diff_attention(model, full_attn_indices, config)

    # Compute post-patch perplexity
    logger.info("Computing post-patch perplexity...")
    patched_ppl = compute_perplexity(model, tokenizer, calibration_text)
    logger.info("Post-patch perplexity: %.4f", patched_ppl)

    ppl_delta = (patched_ppl - baseline_ppl) / baseline_ppl if baseline_ppl > 0 else 0.0

    metrics = {
        "base_dir": base_dir,
        "output_dir": output_dir,
        "model_source": model_source,
        "full_attn_layers": full_attn_indices,
        "full_attn_count": len(full_attn_indices),
        "patched_layers": patched,
        "baseline_perplexity": baseline_ppl,
        "patched_perplexity": patched_ppl,
        "perplexity_delta": ppl_delta,
        "status": "ok",
    }

    # Check rollback
    if check_rollback(metrics):
        metrics["status"] = "rollback_recommended"
        logger.warning("Rollback recommended — saving metrics but NOT the model")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "fork_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
        return metrics

    # Save forked model
    logger.info("Saving forked model to %s", output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (Path(output_dir) / "fork_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    logger.info("Fork complete — %d layers patched", len(patched))
    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Fork Qwen3.5-4B with DiffAttn")
    parser.add_argument(
        "--base-dir",
        default=os.environ.get("MICRO_KIKI_BASE_DIR", DEFAULT_BASE_DIR),
        help="Path to base model (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("MICRO_KIKI_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        help="Output directory (default: %(default)s)",
    )
    args = parser.parse_args()
    result = fork_with_diffattn(args.base_dir, args.output_dir)
    logger.info("Result: %s", json.dumps(result, indent=2))
