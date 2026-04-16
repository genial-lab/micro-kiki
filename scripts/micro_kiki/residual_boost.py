#!/usr/bin/env python3
"""Residual boosting for Brainstacks training.

After the main SFT pass on a domain, residual boosting identifies "hard"
examples (top 25% by loss) and retrains the stack with boosted weights
on those examples. This squeezes extra performance on the tail of the
distribution without overfitting the easy examples.

Paper: Brainstacks (arXiv:2604.01152), Section 3.3
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def compute_per_example_loss(
    batch_logits: list[mx.array],
    batch_labels: list[mx.array],
    lengths: list[int],
) -> list[float]:
    """Compute cross-entropy loss for each example individually.

    Args:
        batch_logits: list of (1, seq_len, vocab_size) logit tensors
        batch_labels: list of (1, seq_len) label tensors
        lengths: list of actual sequence lengths (before padding)

    Returns:
        List of float losses, one per example
    """
    losses = []
    for logits, labels, length in zip(batch_logits, batch_labels, lengths):
        # Shift: predict next token
        shift_logits = logits[:, :-1, :]  # (1, T-1, V)
        shift_labels = labels[:, 1:]       # (1, T-1)

        # Truncate to actual length
        eff_len = min(length - 1, shift_logits.shape[1])
        if eff_len <= 0:
            losses.append(0.0)
            continue

        sl = shift_logits[:, :eff_len, :]  # (1, eff_len, V)
        tl = shift_labels[:, :eff_len]     # (1, eff_len)

        # Cross entropy
        log_probs = nn.log_softmax(sl, axis=-1)           # (1, eff_len, V)
        tl_expanded = mx.expand_dims(tl, axis=-1)         # (1, eff_len, 1)
        token_losses = -mx.take_along_axis(log_probs, tl_expanded, axis=-1)  # (1, eff_len, 1)
        token_losses = mx.squeeze(token_losses, axis=-1)  # (1, eff_len)

        mean_loss = mx.mean(token_losses).item()
        losses.append(float(mean_loss))

    return losses


def select_hard_examples(
    losses: list[float],
    quantile: float = 0.75,
) -> list[int]:
    """Select hard examples based on loss quantile.

    Args:
        losses: per-example losses
        quantile: threshold quantile (0.75 = top 25% hardest)

    Returns:
        List of indices into the original dataset for hard examples
    """
    if len(losses) == 0:
        return []

    threshold = float(np.quantile(losses, quantile))
    hard_indices = [i for i, loss in enumerate(losses) if loss >= threshold]

    # Always select at least 1 example
    if len(hard_indices) == 0 and len(losses) > 0:
        hard_indices = [int(np.argmax(losses))]

    return hard_indices


def build_boosted_weights(
    num_examples: int,
    hard_indices: list[int],
    boost_weight: float = 2.0,
) -> list[float]:
    """Build per-example loss weights for boosted training.

    Hard examples get boost_weight, others get 1.0.

    Args:
        num_examples: total number of examples
        hard_indices: indices of hard examples
        boost_weight: multiplier for hard examples

    Returns:
        List of float weights, one per example
    """
    hard_set = set(hard_indices)
    return [boost_weight if i in hard_set else 1.0 for i in range(num_examples)]


def run_residual_boost_round(
    model: nn.Module,
    tokenizer,
    dataset: list[dict],
    optimizer,
    projectors: dict,
    config: dict,
    round_num: int,
) -> float:
    """Run one round of residual boosting.

    1. Evaluate per-example loss on the full dataset
    2. Select hard examples (top quantile)
    3. Retrain for boost_steps with boosted weights
    4. Return the average loss after boosting

    Args:
        model: model with MoE-LoRA attached
        tokenizer: tokenizer for encoding
        dataset: list of {"messages": [...]} dicts
        optimizer: MLX optimizer (will be reset with lower LR)
        projectors: null-space projectors dict (layer_name -> projector)
        config: residual_boost section from brainstacks.yaml
        round_num: 1-indexed round number (for logging)

    Returns:
        Average loss after this boost round
    """
    from micro_kiki.moe_lora import collect_moe_lora_layers

    boost_steps = config.get("boost_steps", 100)
    boost_weight = config.get("boost_weight", 2.0)
    quantile = config.get("hard_example_quantile", 0.75)
    max_seq_length = config.get("max_seq_length", 2048)

    print(f"\n  [Boost round {round_num}] Evaluating per-example loss...")

    # Step 1: Collect per-example losses
    batch_logits = []
    batch_labels = []
    lengths = []

    for example in dataset:
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer.encode(text)[:max_seq_length]
        length = len(tokens)

        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        batch_logits.append(logits)
        batch_labels.append(input_ids)
        lengths.append(length)

    losses = compute_per_example_loss(batch_logits, batch_labels, lengths)
    avg_loss_before = sum(losses) / len(losses) if losses else 0.0
    print(f"  [Boost round {round_num}] Avg loss before: {avg_loss_before:.4f}")

    # Step 2: Select hard examples
    hard_indices = select_hard_examples(losses, quantile=quantile)
    print(f"  [Boost round {round_num}] Hard examples: {len(hard_indices)}/{len(dataset)}")

    if len(hard_indices) == 0:
        return avg_loss_before

    # Step 3: Build boosted dataset
    weights = build_boosted_weights(len(dataset), hard_indices, boost_weight)

    # Step 4: Retrain on weighted dataset for boost_steps
    step = 0
    total_loss = 0.0

    while step < boost_steps:
        for idx in hard_indices:
            if step >= boost_steps:
                break

            example = dataset[idx]
            w = weights[idx]
            messages = example["messages"]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            tokens = tokenizer.encode(text)[:max_seq_length]
            input_ids = mx.array([tokens])
            labels = input_ids

            def loss_fn(model_params):
                model.update(model_params)
                logits = model(input_ids)
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                ce = nn.losses.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    reduction="mean",
                )
                return ce * w  # apply boost weight

            loss, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
            mx.eval(loss)

            # Project gradients into null-space (if projectors exist)
            if projectors:
                grads = _project_all_grads(grads, projectors)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            step += 1

    avg_loss_after = total_loss / max(step, 1)
    print(f"  [Boost round {round_num}] Avg loss after {step} steps: {avg_loss_after:.4f}")

    return avg_loss_after


def _project_all_grads(grads, projectors: dict):
    """Apply null-space projection to all MoE-LoRA gradients.

    This is a recursive walk matching gradient keys to projector keys.
    Only MoE-LoRA weights (keys containing '_moe_lora') are projected.
    """
    from micro_kiki.null_space import project_gradient

    def _walk(g, prefix=""):
        if isinstance(g, dict):
            return {k: _walk(v, f"{prefix}.{k}" if prefix else k) for k, v in g.items()}
        elif isinstance(g, list):
            return [_walk(v, f"{prefix}.{i}") for i, v in enumerate(g)]
        elif isinstance(g, mx.array):
            # Check if this gradient belongs to a MoE-LoRA layer
            for layer_name, P in projectors.items():
                if layer_name in prefix:
                    return project_gradient(g, P)
            return g
        return g

    return _walk(grads)
