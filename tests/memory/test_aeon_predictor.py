"""Tests for AeonPredictor — JEPA-inspired latent predictor on top of AeonSleep."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.memory.aeon_predictor import (
    AeonPredictor,
    LatentMLP,
    PredictorConfig,
)
from src.memory.aeonsleep import AeonSleep, Episode


def _mock_embed(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_imports_exist():
    assert AeonPredictor is not None
    assert LatentMLP is not None
    assert PredictorConfig is not None


def test_config_defaults():
    cfg = PredictorConfig(dim=384)
    assert cfg.dim == 384
    assert cfg.hidden == 256
    assert cfg.horizon == 1
    assert cfg.n_stacks == 16
    assert cfg.cold_start_threshold == 500


class TestLatentMLPForward:
    def test_forward_shape(self):
        mlp = LatentMLP(dim=384, hidden=256, n_stacks=16, seed=0)
        x = _mock_embed(384, seed=1).reshape(1, -1)
        stack = np.zeros((1, 16), dtype=np.float32)
        stack[0, 3] = 1.0
        h_hat = mlp.forward(x, stack)
        assert h_hat.shape == (1, 384)
        assert h_hat.dtype == np.float32

    def test_forward_batch(self):
        mlp = LatentMLP(dim=64, hidden=32, n_stacks=8, seed=0)
        x = np.stack([_mock_embed(64, seed=i) for i in range(5)])
        stack = np.zeros((5, 8), dtype=np.float32)
        stack[np.arange(5), np.arange(5) % 8] = 1.0
        h_hat = mlp.forward(x, stack)
        assert h_hat.shape == (5, 64)
        # Skip connection means output is not trivially zero at init.
        assert not np.allclose(h_hat, 0.0, atol=1e-6)

    def test_forward_skip_dominates_at_init(self):
        # With small init weights, forward should be close to x (skip path).
        mlp = LatentMLP(dim=32, hidden=16, n_stacks=4, seed=0)
        x = _mock_embed(32, seed=42).reshape(1, -1)
        stack = np.zeros((1, 4), dtype=np.float32)
        stack[0, 0] = 1.0
        h_hat = mlp.forward(x, stack)
        cos = float(
            (h_hat[0] @ x[0])
            / ((np.linalg.norm(h_hat[0]) * np.linalg.norm(x[0])) + 1e-8)
        )
        assert cos > 0.5


class TestLatentMLPBackward:
    def test_backward_reduces_cosine_loss(self):
        rng = np.random.default_rng(0)
        dim, n_stacks = 32, 4
        mlp = LatentMLP(dim=dim, hidden=16, n_stacks=n_stacks, seed=0)
        # Build a trivial pair: target = rotated x, same stack.
        x = rng.standard_normal((8, dim)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        target = np.roll(x, 1, axis=1)  # deterministic "next" latent
        target /= np.linalg.norm(target, axis=1, keepdims=True) + 1e-8
        stack = np.zeros((8, n_stacks), dtype=np.float32)
        stack[:, 0] = 1.0

        losses = []
        for _ in range(200):
            _ = mlp.forward(x, stack)
            loss = mlp.backward_cosine(target, lr=0.05)
            losses.append(loss)

        assert losses[-1] < losses[0] - 0.1, (
            f"loss did not drop enough: start={losses[0]:.3f} end={losses[-1]:.3f}"
        )

    def test_backward_returns_scalar(self):
        mlp = LatentMLP(dim=16, hidden=8, n_stacks=2, seed=0)
        x = np.ones((2, 16), dtype=np.float32) / 4.0
        stack = np.zeros((2, 2), dtype=np.float32)
        stack[:, 0] = 1.0
        _ = mlp.forward(x, stack)
        loss = mlp.backward_cosine(x, lr=0.01)
        assert isinstance(loss, float)
        assert 0.0 <= loss <= 2.0  # cosine loss range
