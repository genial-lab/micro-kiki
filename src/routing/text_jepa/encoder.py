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


class TeacherEncoder(nn.Module):
    """EMA copy of a StudentEncoder. All params frozen; forward runs under no_grad."""

    def __init__(self, student: StudentEncoder) -> None:
        super().__init__()
        import copy

        self.net = copy.deepcopy(student.net)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, student: StudentEncoder, momentum: float) -> None:
        """θ_teacher ← m·θ_teacher + (1-m)·θ_student."""
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        for pt, ps in zip(self.parameters(), student.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
