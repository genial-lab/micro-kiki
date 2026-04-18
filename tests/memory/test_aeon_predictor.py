"""Tests for AeonPredictor — JEPA-inspired latent predictor on top of AeonSleep."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.memory.aeon_predictor import (
    AeonPredictor,
    LatentMLP,
    PredictorConfig,
    detect_collapse,
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


class TestCollapseDetector:
    def test_flags_collapsed_predictions(self):
        rng = np.random.default_rng(0)
        h_t = rng.standard_normal((100, 64)).astype(np.float32)
        h_hat = np.ones_like(h_t) * 0.01  # near-constant == collapse
        flagged, ratio = detect_collapse(h_t, h_hat)
        assert flagged is True
        assert ratio < 0.1

    def test_accepts_healthy_predictions(self):
        rng = np.random.default_rng(1)
        h_t = rng.standard_normal((100, 64)).astype(np.float32)
        h_hat = h_t + rng.standard_normal(h_t.shape).astype(np.float32) * 0.05
        flagged, ratio = detect_collapse(h_t, h_hat)
        assert flagged is False
        assert ratio > 0.5

    def test_boundary_exactly_at_threshold(self):
        # Ratio exactly 0.1 -> NOT flagged (strict <).
        h_t = np.ones((10, 4), dtype=np.float32)
        h_t[0, 0] = 11.0  # std ~ 3.0
        h_hat = np.ones((10, 4), dtype=np.float32)
        h_hat[0, 0] = 2.0  # std ~ 0.3 -> ratio exactly 0.1
        flagged, ratio = detect_collapse(h_t, h_hat, threshold=0.1)
        assert flagged is False or ratio == pytest.approx(0.1, abs=1e-6)


class TestAeonPredictorIngest:
    def _mk_palace(self, dim: int = 32) -> AeonSleep:
        return AeonSleep(dim=dim)

    def test_ingest_appends_to_buffer(self):
        palace = self._mk_palace(dim=32)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=32))
        h = _mock_embed(32, seed=1)
        pred.ingest_latent("t0", h, ts=datetime(2026, 4, 17, 10, 0))
        assert pred.buffer_size() == 1

    def test_ingest_builds_temporal_pairs(self):
        palace = self._mk_palace(dim=32)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=32))
        t0 = datetime(2026, 4, 17, 10, 0)
        pred.ingest_latent("t0", _mock_embed(32, seed=1), ts=t0, stack_id=3)
        pred.ingest_latent(
            "t1", _mock_embed(32, seed=2), ts=t0 + timedelta(minutes=1), stack_id=3
        )
        pairs = pred.pairs_for_training()
        assert len(pairs) == 1
        h_t, h_next, stack_id = pairs[0]
        assert h_t.shape == (32,)
        assert h_next.shape == (32,)
        assert stack_id == 3

    def test_ingest_rejects_wrong_dim(self):
        palace = self._mk_palace(dim=32)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=32))
        with pytest.raises(ValueError, match="dim"):
            pred.ingest_latent("t0", np.zeros(16, dtype=np.float32), ts=datetime.utcnow())

    def test_ingest_writes_to_palace_and_adds_temporal_edge(self):
        palace = self._mk_palace(dim=32)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=32))
        t0 = datetime(2026, 4, 17, 10, 0)
        pred.ingest_latent("t0", _mock_embed(32, seed=1), ts=t0)
        pred.ingest_latent("t1", _mock_embed(32, seed=2), ts=t0 + timedelta(minutes=1))
        temporal_edges = [e for e in palace.graph.edges(kind="temporal")]
        # Both nodes present, and a temporal edge t0 -> t1 exists.
        assert palace.graph.has_node("t0")
        assert palace.graph.has_node("t1")
        assert any(e.src == "t0" and e.dst == "t1" for e in temporal_edges)


class TestAeonPredictorPredict:
    def test_predict_cold_start_returns_input(self):
        palace = AeonSleep(dim=32)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=32))
        h = _mock_embed(32, seed=1)
        out = pred.predict_next(h, horizon=1)
        assert out.shape == (32,)
        np.testing.assert_allclose(out, h, atol=1e-6)  # fallback == identity

    def test_predict_after_ready_differs_from_input(self):
        palace = AeonSleep(dim=16)
        cfg = PredictorConfig(dim=16, hidden=8, n_stacks=2, cold_start_threshold=4)
        pred = AeonPredictor(palace=palace, config=cfg)
        t0 = datetime(2026, 4, 17, 10, 0)
        for i in range(6):
            pred.ingest_latent(
                f"t{i}", _mock_embed(16, seed=i), ts=t0 + timedelta(minutes=i)
            )
        # Force trained flag by calling fit_on_buffer (added in this task).
        pred.fit_on_buffer(lr=0.05, epochs=50, batch_size=4)
        assert pred.ready is True
        h = _mock_embed(16, seed=99)
        out = pred.predict_next(h, horizon=1)
        assert out.shape == (16,)
        # After training it should NOT be exactly h (residual has moved).
        assert not np.allclose(out, h, atol=1e-4)

    def test_predict_horizon_k_iterates(self):
        palace = AeonSleep(dim=16)
        cfg = PredictorConfig(dim=16, hidden=8, n_stacks=2, cold_start_threshold=4)
        pred = AeonPredictor(palace=palace, config=cfg)
        t0 = datetime(2026, 4, 17, 10, 0)
        for i in range(6):
            pred.ingest_latent(
                f"t{i}", _mock_embed(16, seed=i), ts=t0 + timedelta(minutes=i)
            )
        pred.fit_on_buffer(lr=0.05, epochs=30, batch_size=4)
        h = _mock_embed(16, seed=99)
        out1 = pred.predict_next(h, horizon=1)
        out3 = pred.predict_next(h, horizon=3)
        # horizon=3 must differ from horizon=1 (3 applied rollouts).
        assert not np.allclose(out1, out3, atol=1e-4)

    def test_predict_rejects_bad_horizon(self):
        palace = AeonSleep(dim=16)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=16))
        with pytest.raises(ValueError, match="horizon"):
            pred.predict_next(_mock_embed(16, seed=0), horizon=0)


class TestAeonPredictorRecall:
    def test_recall_delegates_to_palace(self):
        palace = AeonSleep(dim=16)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=16))
        t0 = datetime(2026, 4, 17, 10, 0)
        pred.ingest_latent("t0", _mock_embed(16, seed=1), ts=t0)
        pred.ingest_latent(
            "t1", _mock_embed(16, seed=2), ts=t0 + timedelta(minutes=1)
        )
        hits = pred.recall(_mock_embed(16, seed=1), top_k=2)
        assert len(hits) >= 1
        # Top hit should be t0 or t1 — both live in the atlas now.
        assert hits[0].episode_id in {"t0", "t1"}

    def test_recall_top_k_respected(self):
        palace = AeonSleep(dim=16)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=16))
        t0 = datetime(2026, 4, 17, 10, 0)
        for i in range(5):
            pred.ingest_latent(
                f"t{i}", _mock_embed(16, seed=i), ts=t0 + timedelta(minutes=i)
            )
        hits = pred.recall(_mock_embed(16, seed=0), top_k=3)
        assert len(hits) == 3


class TestStackConditioning:
    def test_two_stacks_learn_different_targets(self):
        """Same h_t but stacks 0 and 1 learn opposite shifts."""
        palace = AeonSleep(dim=16)
        cfg = PredictorConfig(
            dim=16, hidden=16, n_stacks=4, cold_start_threshold=4
        )
        pred = AeonPredictor(palace=palace, config=cfg)
        t0 = datetime(2026, 4, 17, 10, 0)

        rng = np.random.default_rng(0)
        # Alternate stack 0 (shift-right) and stack 1 (shift-left) pairs.
        for i in range(40):
            h = rng.standard_normal(16).astype(np.float32)
            h /= np.linalg.norm(h) + 1e-8
            pred.ingest_latent(
                f"t{i}", h, ts=t0 + timedelta(minutes=i), stack_id=i % 2
            )

        # Train long enough that the two stacks separate.
        pred.fit_on_buffer(lr=0.05, epochs=200, batch_size=8)

        probe = rng.standard_normal(16).astype(np.float32)
        probe /= np.linalg.norm(probe) + 1e-8
        out_s0 = pred.predict_next(probe, horizon=1, stack_id=0)
        out_s1 = pred.predict_next(probe, horizon=1, stack_id=1)
        assert not np.allclose(out_s0, out_s1, atol=1e-3), (
            "stack 0 and stack 1 should produce distinguishable outputs"
        )

    def test_unknown_stack_uses_null_condition(self):
        palace = AeonSleep(dim=16)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=16, cold_start_threshold=2))
        t0 = datetime(2026, 4, 17, 10, 0)
        for i in range(4):
            pred.ingest_latent(
                f"t{i}", _mock_embed(16, seed=i),
                ts=t0 + timedelta(minutes=i), stack_id=None,
            )
        pred.fit_on_buffer(epochs=10, batch_size=4)
        # stack_id=None should not raise and should not produce NaN.
        out = pred.predict_next(_mock_embed(16, seed=99), stack_id=None)
        assert not np.any(np.isnan(out))


class TestCentering:
    def test_centering_disabled_by_default(self):
        mlp = LatentMLP(dim=16, hidden=8, n_stacks=2, seed=0)
        assert mlp.use_centering is False

    def test_centering_enabled_shifts_running_mean(self):
        mlp = LatentMLP(
            dim=16, hidden=8, n_stacks=2, seed=0,
            use_centering=True, centering_momentum=0.5,
        )
        # Initially zero.
        assert np.allclose(mlp._running_mean, 0.0)
        x = np.ones((4, 16), dtype=np.float32) * 0.5
        stack = np.zeros((4, 2), dtype=np.float32)
        stack[:, 0] = 1.0
        _ = mlp.forward(x, stack)
        # After one forward, running_mean should be non-zero.
        assert not np.allclose(mlp._running_mean, 0.0)

    def test_predictor_config_propagates_centering(self):
        palace = AeonSleep(dim=16)
        cfg = PredictorConfig(dim=16, use_centering=True, centering_momentum=0.8)
        pred = AeonPredictor(palace=palace, config=cfg)
        assert pred.mlp.use_centering is True
        assert pred.mlp.centering_momentum == pytest.approx(0.8)

    def test_centering_does_not_break_training(self):
        """Regression test: training still converges with centering on."""
        palace = AeonSleep(dim=16)
        cfg = PredictorConfig(
            dim=16, hidden=8, n_stacks=2,
            cold_start_threshold=2, use_centering=True,
        )
        pred = AeonPredictor(palace=palace, config=cfg)
        t0 = datetime(2026, 4, 17, 10, 0)
        for i in range(8):
            pred.ingest_latent(
                f"t{i}", _mock_embed(16, seed=i),
                ts=t0 + timedelta(minutes=i), stack_id=i % 2,
            )
        history = pred.fit_on_buffer(lr=0.05, epochs=20, batch_size=4)
        assert len(history) == 20
        # Loss should not be NaN or exploded.
        assert all(not np.isnan(h) and h < 5.0 for h in history)


class TestPerStackCentering:
    def test_per_stack_centering_disabled_by_default(self):
        mlp = LatentMLP(dim=16, hidden=8, n_stacks=2, seed=0,
                       use_centering=True, per_stack_centering=False)
        assert mlp.per_stack_centering is False

    def test_per_stack_centering_maintains_separate_means(self):
        mlp = LatentMLP(dim=8, hidden=8, n_stacks=2, seed=0,
                       use_centering=True, per_stack_centering=True, centering_momentum=0.5)
        # First stack (sid=0)
        x0 = np.ones((3, 8), dtype=np.float32) * 0.5
        stack0 = np.zeros((3, 2), dtype=np.float32); stack0[:, 0] = 1.0
        sids0 = np.array([0, 0, 0], dtype=np.int64)
        _ = mlp.forward(x0, stack0, stack_ids=sids0)
        mean_stack0 = mlp._running_means_per_stack[1].copy()

        # Second stack (sid=1), very different input
        x1 = np.ones((3, 8), dtype=np.float32) * 5.0
        stack1 = np.zeros((3, 2), dtype=np.float32); stack1[:, 1] = 1.0
        sids1 = np.array([1, 1, 1], dtype=np.int64)
        _ = mlp.forward(x1, stack1, stack_ids=sids1)
        mean_stack1 = mlp._running_means_per_stack[2].copy()

        # The two per-stack running means must differ
        assert not np.allclose(mean_stack0, mean_stack1)

    def test_predictor_config_has_per_stack_flag(self):
        cfg = PredictorConfig(dim=16, use_centering=True, per_stack_centering=True)
        assert cfg.per_stack_centering is True

    def test_per_stack_centering_propagates_via_config(self):
        palace = AeonSleep(dim=16)
        cfg = PredictorConfig(dim=16, use_centering=True, per_stack_centering=True)
        pred = AeonPredictor(palace=palace, config=cfg)
        assert pred.mlp.per_stack_centering is True


class TestLayerNormDelta:
    def test_layernorm_disabled_by_default(self):
        mlp = LatentMLP(dim=16, hidden=8, n_stacks=2, seed=0)
        assert mlp.use_layernorm_delta is False

    def test_layernorm_preserves_finite_output(self):
        mlp = LatentMLP(dim=16, hidden=8, n_stacks=2, seed=0, use_layernorm_delta=True)
        x = np.random.default_rng(1).standard_normal((4, 16)).astype(np.float32)
        stack = np.zeros((4, 2), dtype=np.float32)
        stack[np.arange(4), np.arange(4) % 2] = 1.0
        out = mlp.forward(x, stack)
        assert np.all(np.isfinite(out))

    def test_layernorm_config_propagates(self):
        palace = AeonSleep(dim=16)
        cfg = PredictorConfig(dim=16, use_layernorm_delta=True)
        pred = AeonPredictor(palace=palace, config=cfg)
        assert pred.mlp.use_layernorm_delta is True
