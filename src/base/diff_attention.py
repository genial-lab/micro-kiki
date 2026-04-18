"""Differential Attention (arxiv 2410.05258, ICLR 2025).

Applied only to the full_attention layers of Qwen3.5-35B-A3B (not the
GatedDeltaNet/linear layers).  DiffAttn cancels noise via:

    scores = softmax(Q1*K1) - lambda * softmax(Q2*K2)

This module provides:
- :class:`DiffAttentionConfig` — immutable config for the mechanism.
- :class:`DifferentialAttention` — standalone DiffAttn module.
- :class:`DiffAttentionWrapper` — wraps a standard attention layer.
- :func:`apply_diff_attention` — patches specified layers in-place.

Heavy dependencies (torch) are imported lazily so that the module can
be imported for introspection without torch installed.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Resolve nn.Module base class (stub if torch absent)
# -------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _TORCH_AVAILABLE = False

    class _ModuleStub:
        """Placeholder base when torch is not installed."""

        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)

    class _nn_stub:  # noqa: N801
        Module = _ModuleStub
        Linear = None
        Parameter = None

    nn = _nn_stub  # type: ignore[assignment]


# -------------------------------------------------------------------
# Public, torch-free API
# -------------------------------------------------------------------


@dataclass(frozen=True)
class DiffAttentionConfig:
    """Immutable configuration for the differential attention mechanism."""

    d_model: int
    num_heads: int
    num_layers: int = 13
    reinit_lambda: float = 0.8


def init_lambda(layer_idx: int, num_layers: int, reinit_lambda: float = 0.8) -> float:
    """Compute the per-layer lambda initialisation value.

    Scales linearly with depth so that deeper layers start with a
    stronger differential signal.
    """
    return reinit_lambda * ((layer_idx + 1) / num_layers)


# -------------------------------------------------------------------
# Torch-dependent classes
# -------------------------------------------------------------------


class DifferentialAttention(nn.Module):  # type: ignore[misc]
    """Core differential attention mechanism.

    Splits Q and K into two halves, computes two independent softmax
    attention maps, and subtracts the second (scaled by learnable
    lambda) from the first.  This cancels shared noise across heads
    and reduces activation outliers.

    Args:
        d_model: Hidden size of the model.
        num_heads: Number of attention heads.
        layer_idx: Index of this layer (for lambda init scaling).
        num_layers: Total number of DiffAttn layers (for lambda init).
        reinit_lambda: Base lambda value before depth scaling.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        layer_idx: int = 0,
        num_layers: int = 13,
        reinit_lambda: float = 0.8,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("DifferentialAttention requires torch")
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q1_proj = nn.Linear(d_model, d_model, bias=False)
        self.k1_proj = nn.Linear(d_model, d_model, bias=False)
        self.q2_proj = nn.Linear(d_model, d_model, bias=False)
        self.k2_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        lam_init = init_lambda(layer_idx, num_layers, reinit_lambda)
        self.lambda_param = nn.Parameter(
            torch.full((num_heads,), lam_init)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute differential attention.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            mask: Optional attention mask (not applied in current impl).

        Returns:
            Output tensor of same shape as *x*.
        """
        batch, seq_len, _ = x.shape

        def reshape(t: torch.Tensor) -> torch.Tensor:
            # (batch, seq, d_model) -> (batch, heads, seq, head_dim)
            return t.view(
                batch, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)

        q1, k1 = reshape(self.q1_proj(x)), reshape(self.k1_proj(x))
        q2, k2 = reshape(self.q2_proj(x)), reshape(self.k2_proj(x))
        v = reshape(self.v_proj(x))
        scale = math.sqrt(self.head_dim)

        attn1 = F.softmax(
            torch.matmul(q1, k1.transpose(-2, -1)) / scale, dim=-1
        )
        attn2 = F.softmax(
            torch.matmul(q2, k2.transpose(-2, -1)) / scale, dim=-1
        )

        # lambda: (1, num_heads, 1, 1) for broadcasting
        lam = self.lambda_param.view(1, self.num_heads, 1, 1)
        out = torch.matmul(attn1 - lam * attn2, v)
        return self.o_proj(
            out.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )


class DiffAttentionWrapper(nn.Module):  # type: ignore[misc]
    """Wraps a standard HF attention layer with DiffAttn.

    Preserves output dtype and shape contract.  The original module is
    kept as ``self.original_attn`` for rollback.

    Args:
        original_attn: The original ``self_attn`` module from a
            transformer layer.
        d_model: Hidden size.
        num_heads: Number of attention heads.
        layer_idx: Layer index for lambda initialisation.
        config: Optional :class:`DiffAttentionConfig` overrides.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        d_model: int,
        num_heads: int,
        layer_idx: int = 0,
        config: DiffAttentionConfig | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("DiffAttentionWrapper requires torch")
        super().__init__()
        self.original_attn = original_attn
        cfg = config or DiffAttentionConfig(
            d_model=d_model, num_heads=num_heads
        )
        self.diff_attn = DifferentialAttention(
            d_model=d_model,
            num_heads=num_heads,
            layer_idx=layer_idx,
            num_layers=cfg.num_layers,
            reinit_lambda=cfg.reinit_lambda,
        )

    def _warm_start_projections(self) -> None:
        """Copy Q/K/V/O from original attn; perturb Q2/K2 (std=0.01)."""
        orig = self.original_attn
        da = self.diff_attn
        if hasattr(orig, "q_proj"):
            da.q1_proj.weight.data.copy_(orig.q_proj.weight.data)
            da.q2_proj.weight.data.copy_(orig.q_proj.weight.data)
            da.q2_proj.weight.data.add_(
                torch.randn_like(da.q2_proj.weight.data) * 0.01
            )
        if hasattr(orig, "k_proj"):
            da.k1_proj.weight.data.copy_(orig.k_proj.weight.data)
            da.k2_proj.weight.data.copy_(orig.k_proj.weight.data)
            da.k2_proj.weight.data.add_(
                torch.randn_like(da.k2_proj.weight.data) * 0.01
            )
        if hasattr(orig, "v_proj"):
            da.v_proj.weight.data.copy_(orig.v_proj.weight.data)
        if hasattr(orig, "o_proj"):
            da.o_proj.weight.data.copy_(orig.o_proj.weight.data)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...]:
        """Run DiffAttn, preserving input dtype (BF16 safe)."""
        input_dtype = hidden_states.dtype
        out = self.diff_attn(hidden_states.float())
        out = out.to(input_dtype)
        return (out,)


def apply_diff_attention(
    model: nn.Module,
    layer_indices: list[int],
    config: DiffAttentionConfig | None = None,
) -> list[int]:
    """Patch specified transformer layers with DiffAttn in-place.

    Replaces ``model.model.layers[i].self_attn`` with a
    :class:`DiffAttentionWrapper` for each index in *layer_indices*.

    Args:
        model: A HuggingFace-style causal LM with ``model.model.layers``.
        layer_indices: Which layer indices to patch.
        config: Optional config overrides.

    Returns:
        List of layer indices that were successfully patched.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("apply_diff_attention requires torch")

    patched: list[int] = []
    layers = model.model.layers  # type: ignore[attr-defined]

    for idx in layer_indices:
        if idx < 0 or idx >= len(layers):
            logger.warning(
                "Layer index %d out of range (0..%d), skipping",
                idx,
                len(layers) - 1,
            )
            continue

        layer = layers[idx]
        original_attn = layer.self_attn

        # Extract dimensions from the original attention module
        if hasattr(original_attn, "q_proj"):
            d_model = original_attn.q_proj.in_features
        elif hasattr(original_attn, "hidden_size"):
            d_model = original_attn.hidden_size
        else:
            logger.warning(
                "Cannot determine d_model for layer %d, skipping", idx
            )
            continue

        num_heads = getattr(original_attn, "num_heads", d_model // 64)

        wrapper = DiffAttentionWrapper(
            original_attn=original_attn,
            d_model=d_model,
            num_heads=num_heads,
            layer_idx=idx,
            config=config,
        )
        wrapper._warm_start_projections()

        # Cast wrapper to match original dtype
        param = next(original_attn.parameters(), None)
        if param is not None:
            wrapper = wrapper.to(dtype=param.dtype, device=param.device)

        layer.self_attn = wrapper
        patched.append(idx)
        logger.info("Patched layer %d with DiffAttentionWrapper", idx)

    logger.info(
        "Patched %d / %d requested layers", len(patched), len(layer_indices)
    )
    return patched
