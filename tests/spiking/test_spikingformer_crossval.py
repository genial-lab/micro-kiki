"""Story-32: Spikingformer cross-validation tests.

Tests SpikingFormer adapter with multiple timestep configurations
(T=16, T=64, T=128), verifies spike rate consistency, and
cross-validates with the LIF neuron primitive.

All numpy-only — no torch dependency.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.spiking.lif_neuron import LIFNeuron, rate_encode
from src.spiking.spikingformer_adapter import (
    ConversionBackend,
    SpikingformerAdapter,
    SpikingformerConfig,
)


# ---------------------------------------------------------------------------
# Mock backend (numpy-only, implements ConversionBackend protocol)
# ---------------------------------------------------------------------------


class NumpySpikingBackend:
    """Pure-numpy mock backend for SpikingformerAdapter testing.

    Converts linear layers to spiking equivalents using LIF neurons.
    Forward pass simulates rate-coded spiking inference.
    """

    def __init__(self, timesteps: int, threshold: float = 1.0, tau: float = 1.0) -> None:
        self.timesteps = timesteps
        self.threshold = threshold
        self.tau = tau

    def convert_linear(self, weight: np.ndarray, bias: np.ndarray | None) -> dict:
        """Store weight/bias for spiking simulation."""
        return {"weight": weight, "bias": bias, "type": "spiking_linear"}

    def convert_attention(
        self,
        q_weight: np.ndarray,
        k_weight: np.ndarray,
        v_weight: np.ndarray,
    ) -> dict:
        """Store Q/K/V weights for spiking attention."""
        return {
            "q": q_weight,
            "k": k_weight,
            "v": v_weight,
            "type": "spiking_attention",
        }

    def forward(self, model: list, x: np.ndarray) -> np.ndarray:
        """Run spiking forward pass through converted layers.

        Each layer: x -> linear -> rate_encode -> LIF -> spike_count * threshold.
        """
        out = x.copy()
        for layer in model:
            if layer["type"] == "spiking_linear":
                # Linear transform
                z = out @ layer["weight"].T
                if layer["bias"] is not None:
                    z += layer["bias"]
                # Rate-encode and run through LIF
                z_clipped = np.clip(z, 0, None)  # ReLU before encoding
                max_rate = float(np.max(z_clipped)) + 1e-8
                thr = max_rate / self.timesteps
                neuron = LIFNeuron(threshold=thr, tau=self.tau)
                # Process each element
                result = np.zeros_like(z)
                for idx in np.ndindex(z.shape):
                    val = float(z_clipped[idx])
                    current = rate_encode(np.array([val]), timesteps=self.timesteps, max_rate=max_rate)
                    spikes, _ = neuron.simulate(current)
                    result[idx] = spikes.sum() * thr
                out = result
        return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG."""
    return np.random.default_rng(42)


@pytest.fixture(params=[16, 64, 128], ids=["T=16", "T=64", "T=128"])
def timesteps(request) -> int:
    """Parametrized timestep values."""
    return request.param


@pytest.fixture
def backend(timesteps: int) -> NumpySpikingBackend:
    """Create a backend with the given timesteps."""
    return NumpySpikingBackend(timesteps=timesteps)


@pytest.fixture
def adapter(timesteps: int, backend: NumpySpikingBackend) -> SpikingformerAdapter:
    """Create a SpikingformerAdapter with mock backend."""
    config = SpikingformerConfig(timesteps=timesteps, spike_mode="rate", threshold=1.0)
    return SpikingformerAdapter(config=config, backend=backend)


@pytest.fixture
def simple_model(rng: np.random.Generator) -> list[dict]:
    """A simple 2-layer model as a list of weight dicts."""
    return [
        {"weight": rng.standard_normal((8, 4)).astype(np.float32), "bias": np.zeros(8, dtype=np.float32)},
        {"weight": rng.standard_normal((4, 8)).astype(np.float32), "bias": np.zeros(4, dtype=np.float32)},
    ]


# ---------------------------------------------------------------------------
# Tests: Adapter creation and configuration
# ---------------------------------------------------------------------------


class TestSpikingformerConfig:
    """Test adapter configuration across timestep values."""

    def test_config_timesteps(self, timesteps: int) -> None:
        config = SpikingformerConfig(timesteps=timesteps)
        assert config.timesteps == timesteps

    def test_adapter_available_with_backend(self, adapter: SpikingformerAdapter) -> None:
        assert adapter.available is True

    def test_adapter_info(self, adapter: SpikingformerAdapter, timesteps: int) -> None:
        info = adapter.info()
        assert info["config"]["timesteps"] == timesteps
        assert info["backend_injected"] is True


# ---------------------------------------------------------------------------
# Tests: Conversion
# ---------------------------------------------------------------------------


class TestConversion:
    """Test model conversion across timesteps."""

    def test_convert_list_model(self, adapter: SpikingformerAdapter, simple_model: list[dict]) -> None:
        converted = adapter.convert(simple_model)
        assert isinstance(converted, list)
        assert len(converted) == 2
        for layer in converted:
            assert layer["type"] == "spiking_linear"

    def test_convert_preserves_shapes(self, adapter: SpikingformerAdapter, simple_model: list[dict]) -> None:
        converted = adapter.convert(simple_model)
        assert converted[0]["weight"].shape == (8, 4)
        assert converted[1]["weight"].shape == (4, 8)

    def test_convert_without_backend_raises(self, timesteps: int) -> None:
        adapter = SpikingformerAdapter(config=SpikingformerConfig(timesteps=timesteps))
        with pytest.raises(RuntimeError, match="spikingjelly is not installed"):
            adapter.convert([{"weight": np.eye(4)}])


# ---------------------------------------------------------------------------
# Tests: Spike rate consistency across timesteps
# ---------------------------------------------------------------------------


class TestSpikeRateConsistency:
    """Verify that spike rates converge as timesteps increase.

    Higher T should give more precise rate coding. The reconstruction
    error (|spike_rate - activation|) should decrease with T.
    """

    def test_spike_rate_reconstruction_error_bounded(self, timesteps: int) -> None:
        """Reconstruction error must be bounded by 1/T."""
        max_rate = 1.0
        thr = max_rate / timesteps
        neuron = LIFNeuron(threshold=thr, tau=1.0)

        activations = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        for a in activations:
            current = rate_encode(np.array([a]), timesteps=timesteps, max_rate=max_rate)
            spikes, _ = neuron.simulate(current)
            reconstructed = float(spikes.sum() * thr)
            error = abs(reconstructed - a)
            # Error bounded by threshold = max_rate / T
            assert error <= thr + 1e-9, f"T={timesteps}, a={a}, recon={reconstructed}, err={error}"

    def test_spike_rate_monotonic_with_activation(self, timesteps: int) -> None:
        """Higher activations must produce more spikes."""
        max_rate = 1.0
        thr = max_rate / timesteps
        neuron = LIFNeuron(threshold=thr, tau=1.0)

        activations = [0.1, 0.3, 0.5, 0.7, 0.9]
        spike_counts = []
        for a in activations:
            current = rate_encode(np.array([a]), timesteps=timesteps, max_rate=max_rate)
            spikes, _ = neuron.simulate(current)
            spike_counts.append(float(spikes.sum()))

        for i in range(len(spike_counts) - 1):
            assert spike_counts[i] <= spike_counts[i + 1], (
                f"T={timesteps}: spike count not monotonic at "
                f"a={activations[i]} ({spike_counts[i]}) -> a={activations[i+1]} ({spike_counts[i+1]})"
            )


class TestSpikeRateConvergence:
    """Test that higher T gives better precision (cross-T comparison)."""

    def test_higher_T_gives_smaller_max_error(self) -> None:
        """Max reconstruction error should decrease with T."""
        max_rate = 1.0
        activations = np.linspace(0.05, 0.95, 20)
        timestep_values = [16, 64, 128]
        max_errors = []

        for T in timestep_values:
            thr = max_rate / T
            neuron = LIFNeuron(threshold=thr, tau=1.0)
            errors = []
            for a in activations:
                current = rate_encode(np.array([a]), timesteps=T, max_rate=max_rate)
                spikes, _ = neuron.simulate(current)
                reconstructed = float(spikes.sum() * thr)
                errors.append(abs(reconstructed - a))
            max_errors.append(max(errors))

        # Each T should have lower or equal max error than the previous
        for i in range(len(max_errors) - 1):
            assert max_errors[i] >= max_errors[i + 1] - 1e-9, (
                f"T={timestep_values[i]} max_err={max_errors[i]:.6f} > "
                f"T={timestep_values[i+1]} max_err={max_errors[i+1]:.6f}"
            )


# ---------------------------------------------------------------------------
# Tests: Cross-validation with LIF neuron
# ---------------------------------------------------------------------------


class TestCrossValidationWithLIF:
    """Cross-validate SpikingformerAdapter output against raw LIF neuron."""

    def test_adapter_forward_matches_lif_output(
        self, timesteps: int, rng: np.random.Generator
    ) -> None:
        """Single-layer adapter forward must match direct LIF simulation."""
        dim = 1
        weight = np.eye(dim, dtype=np.float32)  # 1x1 identity
        model_def = [{"weight": weight, "bias": None}]

        backend = NumpySpikingBackend(timesteps=timesteps)
        config = SpikingformerConfig(timesteps=timesteps)
        adapter = SpikingformerAdapter(config=config, backend=backend)

        converted = adapter.convert(model_def)
        x = np.array([[0.5]], dtype=np.float32)  # (1, 1) input
        adapter_out = adapter.forward(converted, x)

        # Direct LIF: same activation through rate_encode + LIF
        max_rate = float(np.clip(x, 0, None).max()) + 1e-8
        thr = max_rate / timesteps
        neuron = LIFNeuron(threshold=thr, tau=1.0)
        current = rate_encode(np.array([0.5]), timesteps=timesteps, max_rate=max_rate)
        spikes, _ = neuron.simulate(current)
        lif_out = float(spikes.sum() * thr)

        # The adapter's output for a single-element identity layer should match
        np.testing.assert_allclose(adapter_out.flatten()[0], lif_out, atol=1e-6)

    def test_lif_neuron_reset_consistency(self, timesteps: int) -> None:
        """After a full run, a fresh neuron should produce identical output."""
        max_rate = 1.0
        thr = max_rate / timesteps
        neuron1 = LIFNeuron(threshold=thr, tau=1.0)
        neuron2 = LIFNeuron(threshold=thr, tau=1.0)

        a = 0.6
        current = rate_encode(np.array([a]), timesteps=timesteps, max_rate=max_rate)

        spikes1, v1 = neuron1.simulate(current)
        spikes2, v2 = neuron2.simulate(current)

        np.testing.assert_array_equal(spikes1, spikes2)
        np.testing.assert_array_equal(v1, v2)

    def test_zero_input_no_spikes(self, timesteps: int) -> None:
        """Zero input should produce zero spikes regardless of T."""
        max_rate = 1.0
        thr = max_rate / timesteps
        neuron = LIFNeuron(threshold=thr, tau=1.0)

        current = rate_encode(np.array([0.0]), timesteps=timesteps, max_rate=max_rate)
        spikes, _ = neuron.simulate(current)
        assert spikes.sum() == 0.0, f"T={timesteps}: zero input produced {spikes.sum()} spikes"

    def test_max_input_saturates(self, timesteps: int) -> None:
        """Max rate input should produce T spikes (one per timestep)."""
        max_rate = 1.0
        thr = max_rate / timesteps
        neuron = LIFNeuron(threshold=thr, tau=1.0)

        current = rate_encode(np.array([max_rate]), timesteps=timesteps, max_rate=max_rate)
        spikes, _ = neuron.simulate(current)
        assert spikes.sum() == pytest.approx(timesteps, abs=1), (
            f"T={timesteps}: max input produced {spikes.sum()} spikes, expected ~{timesteps}"
        )


# ---------------------------------------------------------------------------
# Tests: Multi-dimensional inputs
# ---------------------------------------------------------------------------


class TestMultiDimensional:
    """Test adapter with multi-dimensional inputs."""

    def test_batch_input(self, adapter: SpikingformerAdapter, rng: np.random.Generator) -> None:
        """Adapter should handle batch inputs correctly."""
        weight = rng.standard_normal((4, 4)).astype(np.float32) * 0.1
        model_def = [{"weight": weight, "bias": np.zeros(4, dtype=np.float32)}]
        converted = adapter.convert(model_def)

        # Batch of 3 inputs
        x = np.abs(rng.standard_normal((3, 4)).astype(np.float32)) * 0.5
        out = adapter.forward(converted, x)

        assert out.shape == (3, 4), f"Expected shape (3, 4), got {out.shape}"
        # All outputs should be non-negative (rate-coded)
        assert np.all(out >= -1e-9), "Spiking output should be non-negative"

    def test_output_varies_with_input(self, adapter: SpikingformerAdapter) -> None:
        """Different inputs should generally produce different outputs."""
        weight = np.eye(4, dtype=np.float32)
        model_def = [{"weight": weight, "bias": None}]
        converted = adapter.convert(model_def)

        x1 = np.array([[0.2, 0.4, 0.6, 0.8]], dtype=np.float32)
        x2 = np.array([[0.8, 0.6, 0.4, 0.2]], dtype=np.float32)

        out1 = adapter.forward(converted, x1)
        out2 = adapter.forward(converted, x2)

        # With identity weights, outputs should differ
        assert not np.allclose(out1, out2), "Different inputs should produce different outputs"
