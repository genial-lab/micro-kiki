"""Story-25 tests for LIF neuron primitives.

Verifies spike rate / ReLU equivalence, soft reset behaviour,
zero-input silence, and rate_encode correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.spiking.lif_neuron import LIFNeuron, rate_encode


# -----------------------------------------------------------------------
# LIF neuron: spike rate matches ReLU activation for constant input
# -----------------------------------------------------------------------


class TestLIFRateReLUEquivalence:
    """Spike count * threshold recovers relu(a) for constant input over T steps."""

    @pytest.mark.parametrize("activation", [0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
    def test_rate_matches_relu_T16(self, activation: float) -> None:
        """With T=16, spike_count * threshold ~ relu(activation)."""
        # Arrange
        T = 16
        max_rate = 1.0
        threshold = max_rate / T
        neuron = LIFNeuron(threshold=threshold, tau=1.0)
        current = rate_encode(np.array([activation]), timesteps=T, max_rate=max_rate)

        # Act
        spikes, _ = neuron.simulate(current)
        reconstructed = float(spikes.sum() * threshold)

        # Assert — quantisation error bounded by threshold = 1/T
        assert abs(reconstructed - activation) <= threshold + 1e-9

    @pytest.mark.parametrize("activation", [0.1, 0.3, 0.7, 0.9])
    def test_rate_matches_relu_T64(self, activation: float) -> None:
        """Higher T=64 gives tighter reconstruction."""
        # Arrange
        T = 64
        max_rate = 1.0
        threshold = max_rate / T
        neuron = LIFNeuron(threshold=threshold, tau=1.0)
        current = rate_encode(np.array([activation]), timesteps=T, max_rate=max_rate)

        # Act
        spikes, _ = neuron.simulate(current)
        reconstructed = float(spikes.sum() * threshold)

        # Assert — error <= 1/64 ~ 0.016
        assert abs(reconstructed - activation) <= threshold + 1e-9

    def test_rate_matches_relu_vector(self, rng: np.random.Generator) -> None:
        """Vector of activations all reconstruct within tolerance."""
        # Arrange
        T = 32
        max_rate = 1.0
        threshold = max_rate / T
        neuron = LIFNeuron(threshold=threshold, tau=1.0)
        activations = np.array([0.0, 0.1, 0.3, 0.5, 0.8, 1.0])
        current = rate_encode(activations, timesteps=T, max_rate=max_rate)

        # Act
        spikes, _ = neuron.simulate(current)
        reconstructed = spikes.sum(axis=0) * threshold

        # Assert
        np.testing.assert_allclose(reconstructed, activations, atol=threshold + 1e-9)

    def test_negative_input_treated_as_zero(self) -> None:
        """Negative activations are clipped to 0 by rate_encode (ReLU semantics)."""
        # Arrange
        T = 16
        max_rate = 1.0
        threshold = max_rate / T
        neuron = LIFNeuron(threshold=threshold, tau=1.0)
        current = rate_encode(np.array([-0.5, -1.0, -2.0]), timesteps=T, max_rate=max_rate)

        # Act
        spikes, _ = neuron.simulate(current)

        # Assert — no spikes for negative inputs
        assert spikes.sum() == 0.0


# -----------------------------------------------------------------------
# LIF neuron: soft reset preserves residual voltage
# -----------------------------------------------------------------------


class TestLIFSoftReset:
    """Verify that soft reset subtracts threshold (not resets to zero)."""

    def test_residual_voltage_preserved(self) -> None:
        """After a spike, residual V = V_pre - threshold is kept."""
        # Arrange — feed current > threshold in one step so residual is nonzero
        neuron = LIFNeuron(threshold=1.0, tau=1.0)
        # Step 1: current=1.5 => V=1.5 >= 1.0 => spike, V_after = 1.5-1.0 = 0.5
        # Step 2: current=0.0 => V=0.5 + 0.0 = 0.5 (no spike)
        currents = np.array([[1.5], [0.0]])

        # Act
        spikes, v_final = neuron.simulate(currents)

        # Assert
        assert spikes[0, 0] == 1.0, "should spike on step 0"
        assert spikes[1, 0] == 0.0, "should NOT spike on step 1"
        assert v_final[0] == pytest.approx(0.5), "residual voltage must be 0.5"

    def test_double_spike_with_high_current(self) -> None:
        """Current 2.5 * threshold over 2 steps accumulates correctly."""
        # Arrange
        neuron = LIFNeuron(threshold=1.0, tau=1.0)
        # Step 0: V=2.5, spike, V=1.5, spike again? No — only one spike per step
        # Actually the code fires once per step. So:
        # Step 0: V=0+2.5=2.5 >= 1.0 => spike, V=2.5-1.0=1.5
        # Step 1: V=1.5+0.0=1.5 >= 1.0 => spike, V=1.5-1.0=0.5
        currents = np.array([[2.5], [0.0]])

        # Act
        spikes, v_final = neuron.simulate(currents)

        # Assert — one spike per timestep, soft reset carries residual forward
        assert spikes[0, 0] == 1.0
        assert spikes[1, 0] == 1.0  # residual 1.5 >= threshold
        assert v_final[0] == pytest.approx(0.5)

    def test_soft_reset_vs_hard_reset(self) -> None:
        """With soft reset, total spikes * threshold + final V = total input."""
        # Arrange
        neuron = LIFNeuron(threshold=0.3, tau=1.0)
        total_input = 1.7
        T = 5
        currents = np.full((T, 1), total_input / T)

        # Act
        spikes, v_final = neuron.simulate(currents)
        reconstructed = spikes.sum() * neuron.threshold + v_final[0]

        # Assert — energy conservation: spikes*thr + residual = total input
        assert reconstructed == pytest.approx(total_input, abs=1e-9)


# -----------------------------------------------------------------------
# LIF neuron: zero input produces zero spikes
# -----------------------------------------------------------------------


class TestLIFZeroInput:
    """Zero current must produce zero spikes across any number of timesteps."""

    @pytest.mark.parametrize("T", [1, 10, 100])
    def test_zero_input_zero_spikes_scalar(self, T: int) -> None:
        # Arrange
        neuron = LIFNeuron(threshold=1.0, tau=1.0)
        currents = np.zeros((T, 1))

        # Act
        spikes, v_final = neuron.simulate(currents)

        # Assert
        assert spikes.sum() == 0.0
        assert v_final[0] == pytest.approx(0.0)

    def test_zero_input_zero_spikes_vector(self) -> None:
        # Arrange
        neuron = LIFNeuron(threshold=0.5, tau=1.0)
        currents = np.zeros((50, 8))

        # Act
        spikes, v_final = neuron.simulate(currents)

        # Assert
        assert spikes.sum() == 0.0
        np.testing.assert_array_equal(v_final, np.zeros(8))

    def test_zero_input_with_leaky_tau(self) -> None:
        """Even with leak < 1, zero input still gives zero spikes."""
        # Arrange
        neuron = LIFNeuron(threshold=1.0, tau=0.9)
        currents = np.zeros((20, 4))

        # Act
        spikes, _ = neuron.simulate(currents)

        # Assert
        assert spikes.sum() == 0.0


# -----------------------------------------------------------------------
# rate_encode function
# -----------------------------------------------------------------------


class TestRateEncode:
    """Verify rate_encode produces correct spike trains."""

    def test_shape(self) -> None:
        """Output shape is (T,) + activations.shape."""
        # Arrange
        activations = np.array([0.5, 0.3, 0.7])

        # Act
        current = rate_encode(activations, timesteps=10, max_rate=1.0)

        # Assert
        assert current.shape == (10, 3)

    def test_constant_current_value(self) -> None:
        """Each timestep gets activation / T (after clipping)."""
        # Arrange
        a = 0.8
        T = 20

        # Act
        current = rate_encode(np.array([a]), timesteps=T, max_rate=1.0)

        # Assert — all timesteps identical
        expected = a / T
        np.testing.assert_allclose(current, expected)

    def test_clips_above_max_rate(self) -> None:
        """Activations above max_rate are clipped."""
        # Arrange
        a = 5.0
        T = 10
        max_rate = 2.0

        # Act
        current = rate_encode(np.array([a]), timesteps=T, max_rate=max_rate)

        # Assert — clipped to max_rate/T
        np.testing.assert_allclose(current, max_rate / T)

    def test_clips_negative_to_zero(self) -> None:
        """Negative activations are clipped to zero."""
        # Arrange & Act
        current = rate_encode(np.array([-1.0, -0.5]), timesteps=8, max_rate=1.0)

        # Assert
        np.testing.assert_array_equal(current, 0.0)

    def test_multidimensional_input(self) -> None:
        """Works with 2D activation arrays."""
        # Arrange
        activations = np.array([[0.2, 0.4], [0.6, 0.8]])
        T = 5

        # Act
        current = rate_encode(activations, timesteps=T, max_rate=1.0)

        # Assert
        assert current.shape == (5, 2, 2)
        np.testing.assert_allclose(current[0], activations / T)
        # All timesteps identical
        for t in range(T):
            np.testing.assert_array_equal(current[t], current[0])

    def test_validation_errors(self) -> None:
        """Invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            rate_encode(np.array([1.0]), timesteps=0)
        with pytest.raises(ValueError):
            rate_encode(np.array([1.0]), timesteps=10, max_rate=-1.0)
