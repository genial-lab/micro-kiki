"""MoLoRA: Mixture-of-Experts LoRA (arxiv 2603.15965)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoLoRAConfig:
    rank: int = 16
    num_experts: int = 4
    top_k: int = 2
    alpha: int = 32
    dropout: float = 0.0


def _get_layer_class():
    """Lazy import to avoid requiring torch at module load time."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MoLoRALayer(nn.Module):
        def __init__(self, in_features: int, out_features: int, config: MoLoRAConfig) -> None:
            super().__init__()
            self.config = config
            self.scaling = config.alpha / config.rank
            self.gate = nn.Linear(in_features, config.num_experts, bias=False)
            self.lora_a = nn.ParameterList([
                nn.Parameter(torch.randn(in_features, config.rank) * 0.01)
                for _ in range(config.num_experts)
            ])
            self.lora_b = nn.ParameterList([
                nn.Parameter(torch.zeros(config.rank, out_features))
                for _ in range(config.num_experts)
            ])
            self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, seq_len, dim = x.shape
            x_flat = x.view(-1, dim)
            gate_logits = self.gate(x_flat)
            topk_vals, topk_ids = gate_logits.topk(self.config.top_k, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)

            output = torch.zeros(x_flat.shape[0], self.lora_b[0].shape[1], device=x.device, dtype=x.dtype)
            for k in range(self.config.top_k):
                expert_ids = topk_ids[:, k]
                weights = topk_weights[:, k].unsqueeze(-1)
                for eidx in range(self.config.num_experts):
                    mask = expert_ids == eidx
                    if not mask.any():
                        continue
                    h = self.dropout(x_flat[mask]) @ self.lora_a[eidx]
                    output[mask] += weights[mask] * (h @ self.lora_b[eidx]) * self.scaling

            return output.view(batch, seq_len, -1)

    return MoLoRALayer


def MoLoRALayer(in_features: int, out_features: int, config: MoLoRAConfig):
    """Factory that lazily imports torch."""
    cls = _get_layer_class()
    return cls(in_features, out_features, config)
