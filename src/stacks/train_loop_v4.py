"""Custom MLX training loop with Brainstacks null-space gradient projection.

Wraps a standard MLX LoRA training step with a post-grad hook that
projects each LoRA parameter's gradient into the null-space of previously
frozen adapter stacks.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from src.stacks.null_space_v4 import MODULE_KINDS, NullSpaceRegistry

logger = logging.getLogger(__name__)


def _parse_param_key(key: str) -> tuple[int, str] | None:
    """Extract (layer_idx, module_kind) from a LoRA parameter key.

    Handles dot-separated nested key paths.

    Example:
        'layers.10.self_attn.q_proj.lora_a' -> (10, 'self_attn.q_proj')
        'layers.0.mlp.switch_mlp.gate_proj.lora_b' -> (0, 'mlp.switch_mlp.gate_proj')

    Returns None if the key is not a recognized LoRA parameter.
    """
    parts = key.split(".")

    # Must end in lora_a or lora_b
    if not parts or parts[-1] not in ("lora_a", "lora_b"):
        return None

    # Find "layers" segment
    try:
        layer_pos = parts.index("layers")
        layer_idx = int(parts[layer_pos + 1])
    except (ValueError, IndexError):
        return None

    # Module kind = everything between the layer index and lora_a/b suffix
    mod_parts = parts[layer_pos + 2 : -1]
    if not mod_parts:
        return None

    module = ".".join(mod_parts)
    if module in MODULE_KINDS:
        return (layer_idx, module)
    return None


def project_grad_tree(
    grads: dict,
    registry: NullSpaceRegistry,
    prefix: str = "",
) -> dict:
    """Recursively walk the MLX gradient tree and project LoRA gradients.

    The MLX gradient tree is a nested dict matching the model parameter
    tree structure.  Example path built during traversal:
        layers -> 0 -> self_attn -> q_proj -> lora_a

    For each leaf mx.array whose key path matches a LoRA parameter:
      1. Convert to numpy float32
      2. Flatten to 1-D
      3. Project via registry (null-space orthogonal projection)
      4. Reshape back and convert to mx.array

    Non-LoRA leaves pass through unchanged.
    """
    projected: dict = {}
    for key, value in grads.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            projected[key] = project_grad_tree(value, registry, full_key)
        elif isinstance(value, mx.array):
            parsed = _parse_param_key(full_key)
            if parsed is not None:
                layer_idx, module = parsed
                grad_np = np.array(value, dtype=np.float32)
                original_shape = grad_np.shape
                proj_flat = registry.project(layer_idx, module, grad_np.flatten())
                projected[key] = mx.array(proj_flat.reshape(original_shape))
            else:
                projected[key] = value
        else:
            projected[key] = value
    return projected


def train_stack(
    model: nn.Module,
    tokenizer,
    train_data: list[dict],
    val_data: list[dict],
    frozen_adapter_paths: list[str],
    *,
    iters: int = 200,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    max_seq_length: int = 1024,
    null_space_top_k: int = 32,
    output_dir: str = "output/stack",
) -> dict:
    """Train a single LoRA stack with null-space gradient projection.

    Args:
        model: MLX model with LoRA adapters already applied.
        tokenizer: Tokenizer with an ``encode`` method.
        train_data: List of dicts with a ``"text"`` key.
        val_data: List of dicts with a ``"text"`` key (currently unused,
            reserved for future validation steps).
        frozen_adapter_paths: Paths to previously trained adapter dirs.
            Each dir must contain ``adapters.safetensors``.
        iters: Number of training steps.
        batch_size: Tokens per step (currently 1 sequence at a time).
        learning_rate: Adam learning rate.
        max_seq_length: Truncate sequences to this length.
        null_space_top_k: Number of null-space directions to protect.
        output_dir: Where to save the final adapter.

    Returns:
        dict with key ``"train_losses"`` (list of per-step loss values).
    """
    # Build null-space projectors from frozen adapters
    logger.info(
        "Building null-space projectors from %d frozen adapters...",
        len(frozen_adapter_paths),
    )
    t0 = time.time()
    registry = NullSpaceRegistry.from_frozen_adapters(
        adapter_paths=frozen_adapter_paths,
        layers=range(32),
        modules=MODULE_KINDS,
        top_k=null_space_top_k,
    )
    logger.info("Projectors built in %.1fs", time.time() - t0)

    optimizer = optim.Adam(learning_rate=learning_rate)

    def loss_fn(model: nn.Module, tokens: list[int]) -> mx.array:
        x = mx.array(tokens[:-1])[None]  # (1, seq-1)
        y = mx.array(tokens[1:])[None]   # (1, seq-1)
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    metrics: dict = {"train_losses": []}

    for step in range(iters):
        example = train_data[step % len(train_data)]
        tokens = tokenizer.encode(example.get("text", ""))[:max_seq_length]
        if len(tokens) < 4:
            continue

        loss, grads = loss_and_grad(model, tokens)
        mx.eval(loss)

        if frozen_adapter_paths:
            grads = project_grad_tree(grads, registry)

        optimizer.update(model, grads)
        mx.eval(model.parameters())

        loss_val = loss.item()
        metrics["train_losses"].append(loss_val)

        if (step + 1) % 50 == 0:
            logger.info(
                "step %d/%d: train_loss=%.4f", step + 1, iters, loss_val
            )

    # Save LoRA parameters only
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lora_params: dict[str, mx.array] = {
        k: v
        for k, v in dict(model.named_parameters()).items()
        if "lora_a" in k or "lora_b" in k
    }
    mx.save_safetensors(str(output_path / "adapters.safetensors"), lora_params)
    logger.info(
        "Stack saved to %s (%d LoRA tensors)", output_path, len(lora_params)
    )
    return metrics
