#!/usr/bin/env python3
"""Evaluate Brainstacks: per-domain loss, cross-domain forgetting delta.

Usage:
    # Evaluate a single domain
    python scripts/micro_kiki/eval_stack.py \\
        --config configs/micro_kiki/brainstacks.yaml \\
        --domain python

    # Evaluate all trained domains (forgetting matrix)
    python scripts/micro_kiki/eval_stack.py \\
        --config configs/micro_kiki/brainstacks.yaml \\
        --all
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import yaml
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Archived 2026-04-19 — see docs/research/2026-04-19-moe-lora-root-cause.md
from legacy.moe_lora import apply_moe_lora  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Config loader (standalone, no train_stack dependency)
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate a brainstacks YAML config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)

    required_keys = ["model", "moe_lora", "training", "data", "output", "curriculum", "forgetting"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    return config


# ---------------------------------------------------------------------------
# Dataset loading (standalone, chat-template JSONL)
# ---------------------------------------------------------------------------

def load_domain_dataset(
    domain_dir: str,
    split: str = "valid",
) -> list[dict]:
    """Load a JSONL dataset from domain_dir/{split}.jsonl.

    Each line is a JSON object with a 'messages' key (chat format).
    Returns a list of parsed JSON objects.
    """
    jsonl_path = Path(domain_dir) / f"{split}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

    examples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _tokenize_chat(
    tokenizer,
    messages: list[dict],
    max_seq_length: int,
) -> mx.array:
    """Apply chat template and tokenize, returning token IDs truncated to max_seq_length."""
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        # Fallback: concatenate role: content
        parts = []
        for msg in messages:
            parts.append(f"{msg['role']}: {msg['content']}")
        text = "\n".join(parts)

    tokens = tokenizer.encode(text)
    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]
    return mx.array(tokens)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def _compute_loss_on_tokens(
    model,
    tokens: mx.array,
) -> float:
    """Compute cross-entropy loss on a single token sequence.

    Args:
        model: the language model (base + MoE-LoRA attached)
        tokens: 1-D array of token IDs

    Returns:
        Average cross-entropy loss (float).
    """
    if tokens.shape[0] < 2:
        return 0.0

    # (1, seq_len)
    input_ids = tokens[:-1].reshape(1, -1)
    targets = tokens[1:].reshape(-1)

    logits = model(input_ids)  # (1, seq_len-1, vocab)
    logits = logits.reshape(-1, logits.shape[-1])  # (seq_len-1, vocab)

    # Cross-entropy
    loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
    return loss.item()


def evaluate_domain(
    model: nn.Module,
    tokenizer,
    domain_dir: str,
    max_seq_length: int = 2048,
    val_batches: int = 10,
) -> float:
    """Evaluate model on a domain's validation set.

    Returns average validation loss across up to val_batches examples.
    """
    examples = load_domain_dataset(domain_dir, split="valid")
    if not examples:
        print(f"  WARNING: No validation examples in {domain_dir}")
        return float("inf")

    n_eval = min(len(examples), val_batches)
    total_loss = 0.0
    valid_count = 0

    for ex in examples[:n_eval]:
        messages = ex.get("messages", [])
        if not messages:
            continue
        tokens = _tokenize_chat(tokenizer, messages, max_seq_length)
        if tokens.shape[0] < 2:
            continue
        loss = _compute_loss_on_tokens(model, tokens)
        mx.eval(mx.array(0.0))  # sync
        total_loss += loss
        valid_count += 1

    if valid_count == 0:
        return float("inf")
    return total_loss / valid_count


# ---------------------------------------------------------------------------
# Stack weight loading
# ---------------------------------------------------------------------------

def load_stack_weights(model: nn.Module, stack_dir: str) -> None:
    """Load a frozen stack's MoE-LoRA weights into the model.

    Reads adapters.safetensors from the stack directory and loads
    the weights with strict=False (only MoE-LoRA keys will match).
    """
    adapter_path = Path(stack_dir) / "adapters.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(f"No adapter found at {adapter_path}")
    model.load_weights(str(adapter_path), strict=False)


# ---------------------------------------------------------------------------
# Single-domain evaluation
# ---------------------------------------------------------------------------

def evaluate_single_domain(
    config_path: str,
    domain: str,
) -> dict:
    """Evaluate a single domain stack.

    Loads the base model, attaches MoE-LoRA, loads the domain's frozen
    adapter weights, and computes validation loss + perplexity.

    Returns dict with domain, val_loss, perplexity.
    """
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent.parent
    model_cfg = config["model"]
    moe_cfg = config["moe_lora"]
    output_cfg = config["output"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    # Load base model
    model_path = str(project_root / model_cfg["path"])
    from mlx_lm import load as mlx_load

    print(f"Loading base model from {model_path}...")
    model, tokenizer = mlx_load(model_path)
    model.freeze()

    # Attach MoE-LoRA
    n_attached = apply_moe_lora(
        model,
        target_modules=model_cfg["target_modules"],
        num_experts=moe_cfg["num_experts"],
        rank=moe_cfg["rank"],
        alpha=moe_cfg["alpha"],
        top_k=moe_cfg["top_k"],
        router_hidden=moe_cfg["router_hidden"],
    )
    print(f"Attached {n_attached} MoE-LoRA layers")

    # Load the domain stack weights
    stack_dir = str(project_root / output_cfg["base_dir"] / domain)
    print(f"Loading stack weights from {stack_dir}...")
    load_stack_weights(model, stack_dir)

    # Evaluate on own domain
    domain_dir = str(project_root / data_cfg["base_dir"] / domain)
    t0 = time.time()
    val_loss = evaluate_domain(
        model,
        tokenizer,
        domain_dir,
        max_seq_length=train_cfg["max_seq_length"],
        val_batches=train_cfg.get("val_batches", 10),
    )
    elapsed = time.time() - t0
    perplexity = math.exp(val_loss) if val_loss < 20.0 else float("inf")

    # Compare with stack_meta.json if available
    meta_path = Path(stack_dir) / "stack_meta.json"
    original_loss = None
    delta = None
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        original_loss = meta.get("val_loss")
        if original_loss is not None:
            delta = val_loss - original_loss

    result = {
        "domain": domain,
        "val_loss": round(val_loss, 6),
        "perplexity": round(perplexity, 4),
        "eval_time_s": round(elapsed, 2),
    }
    if original_loss is not None:
        result["original_loss"] = round(original_loss, 6)
        result["delta"] = round(delta, 6)

    print(f"\n  {domain}:")
    print(f"    val_loss   = {val_loss:.4f}")
    print(f"    perplexity = {perplexity:.4f}")
    if delta is not None:
        max_delta = config["forgetting"]["max_delta"]
        status = "OK" if delta < max_delta else "FAIL"
        print(f"    delta      = {delta:+.4f} [{status}]")
    print(f"    time       = {elapsed:.1f}s")

    return result


# ---------------------------------------------------------------------------
# Full forgetting matrix
# ---------------------------------------------------------------------------

def evaluate_all_domains(config_path: str) -> dict[str, dict[str, float]]:
    """Evaluate all trained domains and print forgetting matrix.

    For each trained stack, loads its weights and evaluates on every
    trained domain's validation set. Then compares against original
    losses from stack_meta.json and flags regressions.

    Returns nested dict: results[active_domain][eval_domain] = val_loss.
    """
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent.parent
    output_cfg = config["output"]
    data_cfg = config["data"]
    curriculum = config["curriculum"]

    # Find which domains have been trained
    trained = []
    for domain in curriculum:
        stack_dir = project_root / output_cfg["base_dir"] / domain
        if (stack_dir / "adapters.safetensors").exists():
            trained.append(domain)

    if not trained:
        print("No trained stacks found.")
        return {}

    print(f"Found {len(trained)} trained stacks: {', '.join(trained)}")
    print()

    # Load base model once
    model_cfg = config["model"]
    moe_cfg = config["moe_lora"]
    train_cfg = config["training"]
    model_path = str(project_root / model_cfg["path"])

    from mlx_lm import load as mlx_load

    print(f"Loading base model from {model_path}...")
    model, tokenizer = mlx_load(model_path)
    model.freeze()

    # For each trained domain, load its stack and eval on all domains
    results: dict[str, dict[str, float]] = {}
    total_t0 = time.time()

    for active_domain in trained:
        print(f"\n{'=' * 50}")
        print(f"Stack: {active_domain}")
        print(f"{'=' * 50}")

        # Re-attach fresh MoE-LoRA (overwrites previous adapter state)
        apply_moe_lora(
            model,
            target_modules=model_cfg["target_modules"],
            num_experts=moe_cfg["num_experts"],
            rank=moe_cfg["rank"],
            alpha=moe_cfg["alpha"],
            top_k=moe_cfg["top_k"],
            router_hidden=moe_cfg["router_hidden"],
        )

        # Load this stack's weights
        stack_dir = str(project_root / output_cfg["base_dir"] / active_domain)
        load_stack_weights(model, stack_dir)

        # Eval on all trained domains
        domain_results: dict[str, float] = {}
        for eval_domain in trained:
            domain_dir = str(project_root / data_cfg["base_dir"] / eval_domain)
            try:
                val_loss = evaluate_domain(
                    model,
                    tokenizer,
                    domain_dir,
                    max_seq_length=train_cfg["max_seq_length"],
                    val_batches=5,
                )
            except FileNotFoundError:
                val_loss = float("inf")

            domain_results[eval_domain] = val_loss
            is_own = eval_domain == active_domain
            marker = " <-- own" if is_own else ""
            ppl = math.exp(val_loss) if val_loss < 20.0 else float("inf")
            print(f"  {eval_domain:20s}: loss={val_loss:.4f}  ppl={ppl:.2f}{marker}")

        results[active_domain] = domain_results

    total_elapsed = time.time() - total_t0

    # -----------------------------------------------------------------------
    # Print forgetting summary
    # -----------------------------------------------------------------------
    max_delta = config["forgetting"]["max_delta"]

    print(f"\n{'=' * 60}")
    print("FORGETTING SUMMARY")
    print(f"{'=' * 60}")
    print(f"(max_delta threshold: {max_delta})")
    print()

    any_fail = False
    for domain in trained:
        stack_dir = project_root / output_cfg["base_dir"] / domain
        meta_path = stack_dir / "stack_meta.json"
        if not meta_path.exists():
            print(f"  {domain}: no stack_meta.json, skipping")
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        original_loss = meta.get("val_loss")
        if original_loss is None:
            print(f"  {domain}: no val_loss in meta, skipping")
            continue

        # Check how later stacks affected this domain
        domain_idx = trained.index(domain)
        has_later = False
        for later_domain in trained[domain_idx + 1:]:
            if domain in results.get(later_domain, {}):
                has_later = True
                current_loss = results[later_domain][domain]
                delta = current_loss - original_loss
                status = "OK" if delta < max_delta else "FAIL"
                if status == "FAIL":
                    any_fail = True
                print(
                    f"  {domain:20s} after {later_domain:20s}: "
                    f"orig={original_loss:.4f}  now={current_loss:.4f}  "
                    f"delta={delta:+.4f} [{status}]"
                )
        if not has_later:
            print(f"  {domain:20s}: last trained stack, no later stacks to check")

    print()
    if any_fail:
        print("RESULT: FORGETTING DETECTED -- some deltas exceed threshold")
    else:
        print("RESULT: ALL OK -- no catastrophic forgetting detected")
    print(f"Total eval time: {total_elapsed:.1f}s")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Brainstacks -- Evaluate domain stacks and compute forgetting matrix",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/micro_kiki/brainstacks.yaml",
        help="Path to brainstacks YAML config",
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Evaluate a single domain stack",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all trained domains (forgetting matrix)",
    )
    args = parser.parse_args()

    if args.all:
        evaluate_all_domains(args.config)
    elif args.domain:
        result = evaluate_single_domain(args.config, args.domain)
        print(f"\nJSON: {json.dumps(result, indent=2)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
