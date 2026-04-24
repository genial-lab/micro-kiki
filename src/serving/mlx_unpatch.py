"""In-place LoRA unpatching for fast adapter swap on a live MLX model.

``mlx_lm.tuner.utils.remove_lora_layers`` (as of mlx_lm 0.31.2) only handles
``LoRALinear``. When the base model is a MoE with ``SwitchLinear`` experts
(Qwen3.6-35B-A3B) the tuner wraps them as ``LoRASwitchLinear``, which the
stock helper leaves patched — forcing adapter swap to reload the full base
(~19 GB for 4-bit, ~1-2 s even with a hot page cache).

This module walks the model tree and restores the stored ``.linear`` /
``.embedding`` base layer on every LoRA + DoRA wrapper type. Use it
before applying a *different* adapter on the same model to avoid the
reload cost.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.nn as nn  # pragma: no cover


def unpatch_all_lora(model: "nn.Module") -> "nn.Module":
    """Restore base layers in place on a LoRA-patched MLX model.

    Walks ``model.named_modules()`` and, for every ``LoRALinear`` /
    ``LoRASwitchLinear`` / ``DoRALinear`` / ``LoRAEmbedding`` /
    ``DoRAEmbedding`` encountered, records the substitution
    ``(path, wrapper.linear | wrapper.embedding)``. The substitutions
    are applied via a single ``model.update_modules(tree_unflatten(...))``
    call so the tree pointers are rewritten atomically.

    Returns the same model (mutated in place). No-op if the model has
    no LoRA wrappers.

    Raises
    ------
    RuntimeError
        If ``mlx_lm`` LoRA/DoRA modules cannot be imported — caller
        should catch this and fall back to a full base reload.
    """
    try:
        from mlx.utils import tree_unflatten
        from mlx_lm.tuner.lora import (
            LoRAEmbedding,
            LoRALinear,
            LoRASwitchLinear,
        )
        from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(f"mlx_lm unpatch deps missing: {exc}") from exc

    linear_types = (LoRALinear, LoRASwitchLinear, DoRALinear)
    embedding_types = (LoRAEmbedding, DoRAEmbedding)

    reset: list[tuple[str, object]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, linear_types):
            reset.append((name, mod.linear))
        elif isinstance(mod, embedding_types):
            reset.append((name, mod.embedding))

    if reset:
        model.update_modules(tree_unflatten(reset))
    return model
