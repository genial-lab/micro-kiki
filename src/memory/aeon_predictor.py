"""AeonPredictor — JEPA-inspired latent predictor on top of AeonSleep.

Adds a small numpy MLP that learns h_t -> h_{t+1} from the temporal
edges of TraceGraph. No torch, no sklearn — same pattern as
ForgettingGate so this runs on GrosMac M5 / 16 GB and on CI.

Public API:
    AeonPredictor(palace, config)
        .ingest_latent(turn_id, h, ts, stack_id=None)
        .predict_next(h_t, horizon=1, stack_id=None) -> np.ndarray
        .recall(query_vec, top_k=10)          # delegates to palace
        .fit_on_buffer(lr=1e-3, epochs=1, batch_size=32)
        .ready -> bool

    LatentMLP(dim, hidden, n_stacks)
        .forward(x, stack_onehot) -> h_hat
        .backward_cosine(x, stack_onehot, target) -> float  # returns loss

    PredictorConfig(frozen dataclass)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.memory.aeonsleep import AeonSleep


@dataclass(frozen=True)
class PredictorConfig:
    """Immutable predictor config."""

    dim: int
    hidden: int = 256
    horizon: int = 1
    n_stacks: int = 16
    cold_start_threshold: int = 500
    seed: int = 0


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class LatentMLP:
    """2-layer numpy MLP with skip connection.

    Input: concat(x[dim], stack_onehot[n_stacks]) of size dim+n_stacks
    Hidden: linear(hidden) -> ReLU -> linear(hidden) -> ReLU
    Output: linear(dim) + x   (residual / skip on the embedding path)
    """

    def __init__(self, dim: int, hidden: int, n_stacks: int, seed: int = 0) -> None:
        self.dim = dim
        self.hidden = hidden
        self.n_stacks = n_stacks
        rng = np.random.default_rng(seed)
        in_dim = dim + n_stacks
        scale1 = np.sqrt(2.0 / in_dim)
        scale2 = np.sqrt(2.0 / hidden)
        scale3 = np.sqrt(2.0 / hidden) * 0.1  # small init so skip dominates at t=0
        self.w1 = (rng.standard_normal((in_dim, hidden)) * scale1).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = (rng.standard_normal((hidden, hidden)) * scale2).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.w3 = (rng.standard_normal((hidden, dim)) * scale3).astype(np.float32)
        self.b3 = np.zeros(dim, dtype=np.float32)

    def forward(self, x: np.ndarray, stack_onehot: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must be (batch, {self.dim}), got {x.shape}")
        if stack_onehot.shape != (x.shape[0], self.n_stacks):
            raise ValueError(
                f"stack_onehot must be (batch, {self.n_stacks}), got {stack_onehot.shape}"
            )
        inp = np.concatenate([x, stack_onehot], axis=1).astype(np.float32)
        z1 = np.clip(inp @ self.w1 + self.b1, -30.0, 30.0)
        h1 = _relu(z1)
        z2 = np.clip(h1 @ self.w2 + self.b2, -30.0, 30.0)
        h2 = _relu(z2)
        delta = h2 @ self.w3 + self.b3
        out = (x + delta).astype(np.float32)
        # Cache for backward.
        self._cache = {"inp": inp, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "x": x}
        return out


class AeonPredictor:
    """Facade wrapping AeonSleep with a latent predictor."""

    def __init__(self, palace: "AeonSleep", config: PredictorConfig) -> None:
        raise NotImplementedError("Task 5")
