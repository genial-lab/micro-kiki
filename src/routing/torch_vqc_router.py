"""Torch-native VQC router: autograd training + batched inference.

Replaces PennyLane's parameter-shift gradient computation with torch autograd
backprop through the explicit state-vector simulation in `torch_vqc.py`.

On 6-qubit, 6-layer StronglyEntanglingLayers, this yields 2-3 orders of
magnitude speedup over `quantum_router.QuantumRouter.train` (per-sample SGD
with 108 parameter-shift evaluations per gradient step).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from src.routing.torch_vqc import torch_vqc_forward


class TorchVQCRouter(nn.Module):
    """VQC + classical head with autograd-based training.

    Parameters:
        vqc_weights: (n_layers, n_qubits, 3) rotation angles
        linear_w: (n_qubits, n_classes) classical head weights
        linear_b: (n_classes,) classical head biases
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 6,
        n_classes: int = 35,
        lr: float = 0.01,
        seed: int = 42,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Match QuantumRouter init: vqc weights uniform on [0, 2π)
        rng = np.random.default_rng(seed)
        w_init = rng.uniform(0.0, 2 * math.pi, size=(n_layers, n_qubits, 3))
        self.vqc_weights = nn.Parameter(torch.from_numpy(w_init).double())

        # Optional projection: linear (input_dim → n_qubits) OR MLP (input_dim → hidden_dim → n_qubits)
        # Without: circuit sees features[:n_qubits] only — severe bottleneck.
        # Linear: layer learns which n_qubits-D subspace carries class-discriminative signal.
        # MLP (hidden_dim set): non-linear compression for richer feature extraction.
        self.projection: nn.Module | None
        if input_dim is not None:
            if hidden_dim is not None:
                self.projection = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, n_qubits),
                ).double()
            else:
                self.projection = nn.Linear(input_dim, n_qubits).double()
        else:
            self.projection = None

        # Classical head: normal(0, 0.1) init, zero bias (match QuantumRouter)
        rng2 = np.random.default_rng(seed + 1)
        lw_init = rng2.standard_normal((n_qubits, n_classes)) * 0.1
        self.linear_w = nn.Parameter(torch.from_numpy(lw_init).double())
        self.linear_b = nn.Parameter(torch.zeros(n_classes, dtype=torch.float64))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute logits for a (B, feat_dim) batch of embeddings."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if self.projection is not None:
            # Project input → n_qubits, bound to [-π, π] for rotations
            features = math.pi * torch.tanh(self.projection(features))
        qubits = torch_vqc_forward(
            features, self.vqc_weights, n_qubits=self.n_qubits, n_layers=self.n_layers
        )
        return qubits @ self.linear_w + self.linear_b

    def train_batched(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 10,
    ) -> list[float]:
        """Full-batch SGD over `epochs` passes. Returns per-epoch loss."""
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        losses: list[float] = []
        for _ in range(epochs):
            opt.zero_grad()
            logits = self.forward(embeddings)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    @torch.no_grad()
    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Argmax class prediction for a (B, feat_dim) batch."""
        logits = self.forward(embeddings)
        return logits.argmax(dim=-1)
