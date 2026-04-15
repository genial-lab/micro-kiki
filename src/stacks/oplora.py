"""Orthogonal Projection LoRA (arxiv 2510.13003)."""
from __future__ import annotations

import torch


def orthogonal_projection(prior_subspace: torch.Tensor, dim: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(prior_subspace.float())
    identity = torch.eye(dim, device=prior_subspace.device, dtype=torch.float32)
    return (identity - q @ q.T).to(prior_subspace.dtype)


def init_oplora_experts(
    in_features: int, rank: int, num_experts: int,
    prior_subspace: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    experts = []
    for _ in range(num_experts):
        a = torch.randn(in_features, rank) * 0.01
        if prior_subspace is not None:
            a = orthogonal_projection(prior_subspace, in_features) @ a
        experts.append(a)
    return experts
