"""Tests for the Spikingformer adapter (story-30).

All tests use a mock backend since spikingjelly is unlikely to be
installed in the test environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from src.spiking.spikingformer_adapter import (
    ConversionBackend,
    SpikingformerAdapter,
    SpikingformerConfig,
    has_spikingjelly,
)


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


@dataclass
class MockSpikingLayer:
    """Mock spiking layer that does a simple matmul + ReLU."""

    weight: np.ndarray
    bias: np.ndarray | None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.weight.T
        if self.bias is not None:
            z = z + self.bias
        return np.maximum(z, 0.0)


class MockBackend:
    """Mock conversion backend for testing without spikingjelly."""

    def convert_linear(
        self, weight: np.ndarray, bias: np.ndarray | None
    ) -> MockSpikingLayer:
        return MockSpikingLayer(weight=weight, bias=bias)

    def convert_attention(
        self,
        q_weight: np.ndarray,
        k_weight: np.ndarray,
        v_weight: np.ndarray,
    ) -> dict[str, MockSpikingLayer]:
        return {
            "q": MockSpikingLayer(q_weight, None),
            "k": MockSpikingLayer(k_weight, None),
            "v": MockSpikingLayer(v_weight, None),
        }

    def forward(self, model: Any, x: np.ndarray) -> np.ndarray:
        out = x
        if isinstance(model, list):
            for layer in model:
                out = layer(out)
        return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpikingformerAdapter:
    """Story-30 tests with mocked spikingjelly."""

    def test_config_defaults(self) -> None:
        cfg = SpikingformerConfig()
        assert cfg.timesteps == 4
        assert cfg.spike_mode == "rate"
        assert cfg.threshold == 1.0
        assert cfg.backend == "torch"

    def test_has_spikingjelly_returns_bool(self) -> None:
        result = has_spikingjelly()
        assert isinstance(result, bool)

    def test_convert_raises_without_backend_or_spikingjelly(self) -> None:
        """Without spikingjelly and no mock, convert() raises RuntimeError."""
        if has_spikingjelly():
            pytest.skip("spikingjelly is installed")
        adapter = SpikingformerAdapter()
        with pytest.raises(RuntimeError, match="spikingjelly is not installed"):
            adapter.convert([{"weight": np.eye(4), "bias": None}])

    def test_convert_with_mock_backend(self) -> None:
        """Mock backend converts a list of weight dicts."""
        backend = MockBackend()
        adapter = SpikingformerAdapter(backend=backend)

        rng = np.random.default_rng(0)
        layers = [
            {"weight": rng.standard_normal((8, 4)), "bias": np.zeros(8)},
            {"weight": rng.standard_normal((2, 8)), "bias": np.zeros(2)},
        ]
        snn = adapter.convert(layers)
        assert len(snn) == 2
        assert isinstance(snn[0], MockSpikingLayer)

    def test_forward_with_mock_backend(self) -> None:
        """Mock forward produces correct output shape."""
        backend = MockBackend()
        adapter = SpikingformerAdapter(backend=backend)

        rng = np.random.default_rng(1)
        layers = [
            {"weight": rng.standard_normal((8, 4)), "bias": np.zeros(8)},
            {"weight": rng.standard_normal((2, 8)), "bias": np.zeros(2)},
        ]
        snn = adapter.convert(layers)
        x = rng.standard_normal((3, 4))
        out = adapter.forward(snn, x)
        assert out.shape == (3, 2)

    def test_info_reports_status(self) -> None:
        adapter = SpikingformerAdapter(
            config=SpikingformerConfig(timesteps=8),
            backend=MockBackend(),
        )
        info = adapter.info()
        assert info["backend_injected"] is True
        assert info["config"]["timesteps"] == 8
        assert isinstance(info["spikingjelly_installed"], bool)
        assert isinstance(info["torch_installed"], bool)

    def test_available_with_mock(self) -> None:
        adapter = SpikingformerAdapter(backend=MockBackend())
        assert adapter.available is True

    def test_available_without_anything(self) -> None:
        if has_spikingjelly():
            pytest.skip("spikingjelly is installed")
        adapter = SpikingformerAdapter()
        assert adapter.available is False

    def test_protocol_compliance(self) -> None:
        """MockBackend satisfies the ConversionBackend protocol."""
        backend = MockBackend()
        assert isinstance(backend, ConversionBackend)

    def test_mock_attention_conversion(self) -> None:
        """Mock backend converts attention projections."""
        backend = MockBackend()
        adapter = SpikingformerAdapter(backend=backend)

        rng = np.random.default_rng(2)
        q_w = rng.standard_normal((16, 8))
        k_w = rng.standard_normal((16, 8))
        v_w = rng.standard_normal((16, 8))
        result = backend.convert_attention(q_w, k_w, v_w)
        assert "q" in result and "k" in result and "v" in result


# ---------------------------------------------------------------------------
# Spike-attention + SpikingformerConverter tests (story-30 core)
# ---------------------------------------------------------------------------

from src.spiking.spikingformer import (
    spike_attention,
    SpikingTransformerLayer,
    SpikingformerConverter,
)


class TestSpikeAttention:
    """Tests for the spike-driven attention primitive."""

    def test_spike_attention_2d_shape(self) -> None:
        rng = np.random.default_rng(10)
        seq, dk, dv = 5, 8, 8
        Q = rng.standard_normal((seq, dk))
        K = rng.standard_normal((seq, dk))
        V = rng.standard_normal((seq, dv))
        out = spike_attention(Q, K, V, threshold=0.0)
        assert out.shape == (seq, dv)

    def test_spike_attention_3d_batched(self) -> None:
        rng = np.random.default_rng(11)
        batch, seq, dk, dv = 2, 4, 6, 6
        Q = rng.standard_normal((batch, seq, dk))
        K = rng.standard_normal((batch, seq, dk))
        V = rng.standard_normal((batch, seq, dv))
        out = spike_attention(Q, K, V, threshold=0.0)
        assert out.shape == (batch, seq, dv)

    def test_spike_attention_high_threshold_zeros(self) -> None:
        """Very high threshold => no spikes => zero output."""
        rng = np.random.default_rng(12)
        Q = rng.standard_normal((3, 4)) * 0.1
        K = rng.standard_normal((3, 4)) * 0.1
        V = rng.standard_normal((3, 4))
        out = spike_attention(Q, K, V, threshold=100.0)
        np.testing.assert_array_equal(out, 0.0)

    def test_spike_attention_produces_finite(self) -> None:
        rng = np.random.default_rng(13)
        Q = rng.standard_normal((4, 8))
        K = rng.standard_normal((4, 8))
        V = rng.standard_normal((4, 8))
        out = spike_attention(Q, K, V, threshold=1.0)
        assert np.all(np.isfinite(out))


class TestSpikingTransformerLayer:
    """Tests for a single spiking transformer layer."""

    @staticmethod
    def _make_layer(
        d_model: int = 16,
        d_k: int = 16,
        d_ff: int = 32,
        seed: int = 42,
    ) -> SpikingTransformerLayer:
        rng = np.random.default_rng(seed)
        scale = 0.1
        return SpikingTransformerLayer(
            w_q=rng.standard_normal((d_model, d_k)) * scale,
            w_k=rng.standard_normal((d_model, d_k)) * scale,
            w_v=rng.standard_normal((d_model, d_k)) * scale,
            w_out=rng.standard_normal((d_k, d_model)) * scale,
            w_mlp1=rng.standard_normal((d_model, d_ff)) * scale,
            w_mlp2=rng.standard_normal((d_ff, d_model)) * scale,
            threshold=0.5,
        )

    def test_forward_2d(self) -> None:
        layer = self._make_layer()
        x = np.random.default_rng(0).standard_normal((4, 16))
        out = layer(x)
        assert out.shape == (4, 16)
        assert np.all(np.isfinite(out))

    def test_forward_3d_batched(self) -> None:
        layer = self._make_layer()
        x = np.random.default_rng(1).standard_normal((2, 4, 16))
        out = layer(x)
        assert out.shape == (2, 4, 16)

    def test_param_count(self) -> None:
        layer = self._make_layer(d_model=16, d_k=16, d_ff=32)
        # 4 * 16*16 + 2 * 16*32 = 1024 + 1024 = 2048
        assert layer.param_count == 2048


class TestSpikingformerConverter:
    """Tests for the converter on a tiny 2-layer transformer."""

    @staticmethod
    def _make_ann_layers(
        d_model: int = 16,
        d_ff: int = 32,
        n_layers: int = 2,
        seed: int = 99,
    ) -> list[dict[str, np.ndarray]]:
        rng = np.random.default_rng(seed)
        scale = 0.1
        layers = []
        for _ in range(n_layers):
            layers.append({
                "w_q": rng.standard_normal((d_model, d_model)) * scale,
                "w_k": rng.standard_normal((d_model, d_model)) * scale,
                "w_v": rng.standard_normal((d_model, d_model)) * scale,
                "w_out": rng.standard_normal((d_model, d_model)) * scale,
                "w_mlp1": rng.standard_normal((d_model, d_ff)) * scale,
                "w_mlp2": rng.standard_normal((d_ff, d_model)) * scale,
            })
        return layers

    def test_convert_and_forward(self) -> None:
        converter = SpikingformerConverter(threshold=0.5)
        ann_layers = self._make_ann_layers()
        snn_layers = converter.convert_model(ann_layers)
        assert len(snn_layers) == 2

        x = np.random.default_rng(0).standard_normal((3, 16))
        out = converter.forward_model(snn_layers, x)
        assert out.shape == (3, 16)
        assert np.all(np.isfinite(out))

    def test_sparsity_computed(self) -> None:
        converter = SpikingformerConverter(threshold=0.5)
        ann_layers = self._make_ann_layers()
        snn_layers = converter.convert_model(ann_layers)
        x = np.random.default_rng(1).standard_normal((3, 16))
        stats = converter.compute_sparsity(snn_layers, x)
        assert "per_layer" in stats
        assert len(stats["per_layer"]) == 2
        assert 0.0 <= stats["avg_sparsity"] <= 1.0
