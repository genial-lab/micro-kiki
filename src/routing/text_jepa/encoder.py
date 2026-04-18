"""Student encoder — trainable MLP projector on top of frozen MiniLM embeddings."""
from __future__ import annotations

import torch
from torch import nn


class StudentEncoder(nn.Module):
    """2-layer MLP: input_dim -> hidden_dim -> output_dim with GELU."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., input_dim) → (..., output_dim)."""
        return self.net(x)
