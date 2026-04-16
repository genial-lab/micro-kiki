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
