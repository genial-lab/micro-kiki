#!/usr/bin/env python3
"""MoE-LoRA module for Brainstacks on Qwen3.5-4B.

Each MoE-LoRA layer replaces a single nn.Linear with:
  - N LoRA experts (A_i, B_i) with rank r
  - A learned router MLP that selects top-k experts per token
  - The base weight W is frozen; only LoRA deltas are trainable

Architecture per projection:
  y = W @ x + sum_topk( gate_i * (B_i @ A_i @ x) * scale )

Paper: Brainstacks (arXiv:2604.01152)
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class LoRAExpert(nn.Module):
    """Single LoRA expert: low-rank delta W = B @ A scaled by alpha/rank.

    Uses rsLoRA scaling: scale = alpha / sqrt(rank) instead of alpha / rank
    for rank-stabilized training (Kalajdzievski 2023).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        use_rs_lora: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        # rsLoRA: scale by 1/sqrt(r) instead of 1/r for stable high-rank training
        self.scale = alpha / math.sqrt(rank) if use_rs_lora else alpha / rank
        self.dropout = dropout

        # A: (in_features, rank) — Kaiming uniform init
        self.lora_a = mx.random.normal((in_features, rank)) * (1.0 / math.sqrt(in_features))
        # B: (rank, out_features) — zero init so expert starts as no-op
        self.lora_b = mx.zeros((rank, out_features))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (..., in_features) -> (..., out_features)
        if self.dropout > 0 and self.training:
            mask = mx.random.bernoulli(1.0 - self.dropout, x.shape)
            x = x * mask / (1.0 - self.dropout)
        # (..., in) @ (in, r) -> (..., r)
        h = x @ self.lora_a
        # (..., r) @ (r, out) -> (..., out)
        out = h @ self.lora_b
        return out * self.scale

    def flat_weights(self) -> mx.array:
        """Return concatenated flattened weights for null-space projection."""
        return mx.concatenate([
            mx.reshape(self.lora_a, (-1,)),
            mx.reshape(self.lora_b, (-1,)),
        ])

    @property
    def num_params(self) -> int:
        return self.in_features * self.rank + self.rank * self.out_features


class MoELoRALayer(nn.Module):
    """Mixture-of-Experts LoRA layer.

    Replaces a single frozen linear projection with N LoRA experts
    and a learned top-k router.

    Forward:
      router_logits = Router(x)             # (batch, seq, num_experts)
      topk_weights, topk_indices = topk(router_logits, k)
      topk_weights = softmax(topk_weights)  # normalize over selected experts
      y = sum_i( topk_weights[i] * Expert_i(x) )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 4,
        rank: int = 16,
        alpha: float = 32.0,
        top_k: int = 2,
        dropout: float = 0.0,
        router_hidden: int = 64,
        use_rs_lora: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert pool
        self.experts = [
            LoRAExpert(
                in_features=in_features,
                out_features=out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                use_rs_lora=use_rs_lora,
            )
            for _ in range(num_experts)
        ]

        # Router: small MLP mapping input hidden -> expert logits
        self.router_w1 = nn.Linear(in_features, router_hidden)
        self.router_w2 = nn.Linear(router_hidden, num_experts)

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Compute routing weights and expert indices.

        Args:
            x: (batch, seq, in_features)

        Returns:
            weights: (batch, seq, top_k) — normalized weights
            indices: (batch, seq, top_k) — expert indices
        """
        # Router forward
        h = nn.gelu(self.router_w1(x))           # (B, T, router_hidden)
        logits = self.router_w2(h)                 # (B, T, num_experts)

        # Top-k selection — mx.topk returns values only (no indices)
        # Use argsort to get indices, then gather the logits
        sorted_indices = mx.argsort(logits, axis=-1)
        # Take the last top_k (highest logits)
        indices = sorted_indices[..., -self.top_k:]
        top_k_logits = mx.take_along_axis(logits, indices, axis=-1)
        weights = mx.softmax(top_k_logits, axis=-1)  # normalize over selected
        return weights, indices

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: soft-weighted sum over ALL experts.

        Uses softmax over all experts (fully differentiable) instead of
        top-k hard routing (which uses argsort, non-differentiable in MLX).
        The router still learns to specialize experts via the softmax weights.

        Args:
            x: (..., in_features) — supports any batch dimensions

        Returns:
            delta: (..., out_features) — additive LoRA delta
        """
        # Ensure 3D for router
        orig_shape = x.shape
        if x.ndim == 2:
            x = mx.expand_dims(x, axis=0)  # (1, T, in)

        # Router: compute soft weights over ALL experts
        h = nn.gelu(self.router_w1(x))     # (..., router_hidden)
        logits = self.router_w2(h)          # (..., num_experts)
        weights = mx.softmax(logits, axis=-1)  # (..., num_experts)

        # Compute all expert outputs
        expert_outputs = mx.stack(
            [expert(x) for expert in self.experts], axis=-2
        )  # (..., num_experts, out_features)

        # Weighted sum over ALL experts (fully differentiable)
        w = mx.expand_dims(weights, axis=-1)  # (..., num_experts, 1)
        delta = mx.sum(expert_outputs * w, axis=-2)  # (..., out_features)

        # Restore original batch dims
        if len(orig_shape) == 2:
            delta = mx.squeeze(delta, axis=0)

        return delta

    def all_expert_weights_flat(self) -> list[mx.array]:
        """Return flat weight vectors for each expert (for null-space)."""
        return [expert.flat_weights() for expert in self.experts]



class MoELoRALinear(nn.Module):
    """Linear layer wrapped with MoE-LoRA: output = base(x) + moe_lora(x).

    This replaces the original nn.Linear in the model so that MoE-LoRA
    participates in the forward pass and receives gradients during training.
    """

    def __init__(self, base_linear: nn.Linear, moe_lora: "MoELoRALayer"):
        super().__init__()
        self.base = base_linear
        self.moe_lora = moe_lora

    def __call__(self, x):
        base_out = self.base(x)
        lora_out = self.moe_lora(x)
        return base_out + lora_out

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return getattr(self.base, "bias", None)


def apply_moe_lora(
    model: nn.Module,
    target_modules: list[str],
    num_experts: int = 4,
    rank: int = 16,
    alpha: float = 32.0,
    top_k: int = 2,
    dropout: float = 0.0,
    router_hidden: int = 64,
    use_rs_lora: bool = True,
) -> int:
    """Attach MoE-LoRA layers to all matching linear projections in the model.

    Walks model.model.layers[i].{self_attn,mlp}.{target_module} and stores
    a MoE-LoRA adapter as a sibling attribute.

    The base Linear is NOT removed — the forward pass becomes:
        y = base_linear(x) + moe_lora(x)

    We store the MoE-LoRA as an attribute named `{target}_moe_lora` on the
    parent submodule.

    Returns:
        count: number of MoE-LoRA layers attached
    """
    count = 0
    layers = model.model.layers if hasattr(model, "model") else model.layers

    for layer_idx, layer in enumerate(layers):
        for sub_name in ["self_attn", "mlp"]:
            sub_module = getattr(layer, sub_name, None)
            if sub_module is None:
                continue
            for target in target_modules:
                linear = getattr(sub_module, target, None)
                if linear is None or not isinstance(linear, nn.Linear):
                    continue
                in_f = linear.weight.shape[1]
                out_f = linear.weight.shape[0]
                moe = MoELoRALayer(
                    in_features=in_f,
                    out_features=out_f,
                    num_experts=num_experts,
                    rank=rank,
                    alpha=alpha,
                    top_k=top_k,
                    dropout=dropout,
                    router_hidden=router_hidden,
                    use_rs_lora=use_rs_lora,
                )
                # Replace base Linear with MoELoRALinear wrapper
                # This makes MoE-LoRA part of the forward pass graph
                # so gradients flow to lora_a/lora_b during training
                wrapped = MoELoRALinear(linear, moe)
                setattr(sub_module, target, wrapped)
                # Also store sibling ref for null-space extraction
                attr_name = f"{target}_moe_lora"
                setattr(sub_module, attr_name, moe)
                count += 1

    return count


def collect_moe_lora_layers(model: nn.Module) -> list[MoELoRALayer]:
    """Walk the model and collect all attached MoE-LoRA layers."""
    moe_layers = []
    layers = model.model.layers if hasattr(model, "model") else model.layers
    for layer in layers:
        for sub_name in ["self_attn", "mlp"]:
            sub_module = getattr(layer, sub_name, None)
            if sub_module is None:
                continue
            for attr_name in dir(sub_module):
                if attr_name.endswith("_moe_lora"):
                    moe = getattr(sub_module, attr_name)
                    if isinstance(moe, MoELoRALayer):
                        moe_layers.append(moe)
    return moe_layers


def moe_lora_forward_hook(base_linear: nn.Linear, moe_lora: MoELoRALayer, x: mx.array) -> mx.array:
    """Combined forward: frozen base + MoE-LoRA delta.

    This is called by the patched forward in train_stack.py.
    """
    base_out = base_linear(x)
    delta = moe_lora(x)
    return base_out + delta
