"""Story-25 tests for LAS converter.

Verifies SpikingLinear forward pass, convert_layer, and
verify_equivalence on numpy-only paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.spiking.las_converter import (
    LASConverter,
    SpikingLinear,
    convert_linear,
    verify_equivalence,
)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _ann_linear(w: np.ndarray, b: np.ndarray | None, x: np.ndarray) -> np.ndarray:
    z = x @ w.T
    if b is not None:
        z = z + b
    return _relu(z)


# -----------------------------------------------------------------------
# SpikingLinear: forward pass matches nn.Linear for positive inputs
# -----------------------------------------------------------------------


class TestSpikingLinearForward:
    """SpikingLinear forward matches ANN linear + ReLU within quantisation error."""

    def test_positive_inputs_match_ann(self, rng: np.random.Generator) -> None:
        """For positive inputs, spiking output ~ relu(x @ W.T + b)."""
        # Arrange
        w = rng.standard_normal((6, 4)).astype(np.float64) * 0.2
        b = rng.standard_normal(6).astype(np.float64) * 0.05
        T = 256  # high T for tight bound
        snn = SpikingLinear(weight=w, bias=b, timesteps=T, max_rate=1.0)
        # Positive inputs only (within [0, 0.5])
        x = rng.uniform(0.0, 0.5, size=(3, 4))

        # Act
        snn_out = snn.forward(x)
        ann_out = _ann_linear(w, b, x)

        # Assert — relative error within 5%
        rel_err = np.linalg.norm(snn_out - ann_out) / (np.linalg.norm(ann_out) + 1e-12)
        assert rel_err < 0.05, f"rel_err={rel_err:.4f}"

    def test_identity_activation_skips_relu(self, rng: np.random.Generator) -> None:
        """With activation='identity', negative pre-activations pass through."""
        # Arrange
        w = np.eye(4)  # identity weights
        b = None
        T = 64
        snn = SpikingLinear(weight=w, bias=b, timesteps=T, max_rate=1.0, activation="identity")
        x = np.array([[0.5, 0.3, 0.8, 0.1]])  # all positive for rate-code

        # Act
        snn_out = snn.forward(x)

        # Assert — should approximate x (identity weight, no bias)
        threshold = 1.0 / T
        np.testing.assert_allclose(snn_out, x, atol=threshold + 1e-9)

    def test_output_shape_matches(self) -> None:
        """Output shape is (..., out_features)."""
        # Arrange
        w = np.zeros((5, 3))
        snn = SpikingLinear(weight=w, bias=None, timesteps=8)

        # Act
        out = snn.forward(np.zeros((2, 3)))

        # Assert
        assert out.shape == (2, 5)

    def test_single_sample(self, rng: np.random.Generator) -> None:
        """Works with 1D input (single sample)."""
        # Arrange
        w = rng.standard_normal((4, 3)) * 0.2
        snn = SpikingLinear(weight=w, bias=None, timesteps=32)
        x = np.array([0.1, 0.2, 0.3])

        # Act
        out = snn.forward(x)

        # Assert
        assert out.shape == (4,)


# -----------------------------------------------------------------------
# LASConverter.convert_layer: converts weight matrix correctly
# -----------------------------------------------------------------------


class TestConvertLayer:
    """LASConverter.convert_layer produces a correct SpikingLinear."""

    def test_convert_dict_layer(self, rng: np.random.Generator) -> None:
        """Converting a dict {weight, bias} creates SpikingLinear with matching dims."""
        # Arrange
        w = rng.standard_normal((8, 4)).astype(np.float64) * 0.3
        b = rng.standard_normal(8).astype(np.float64) * 0.05
        converter = LASConverter(timesteps=32, max_rate=1.0)

        # Act
        snn_layer = converter.convert_layer({"weight": w, "bias": b})

        # Assert
        assert isinstance(snn_layer, SpikingLinear)
        assert snn_layer.in_features == 4
        assert snn_layer.out_features == 8
        np.testing.assert_array_equal(snn_layer.weight, w)
        np.testing.assert_array_equal(snn_layer.bias, b)
        assert snn_layer.timesteps == 32

    def test_convert_tuple_layer(self) -> None:
        """Converting a (weight, bias) tuple works."""
        # Arrange
        w = np.eye(3)
        b = np.zeros(3)
        converter = LASConverter(timesteps=16)

        # Act
        snn_layer = converter.convert_layer((w, b))

        # Assert
        assert isinstance(snn_layer, SpikingLinear)
        np.testing.assert_array_equal(snn_layer.weight, w)

    def test_convert_no_bias(self) -> None:
        """Converting a layer with no bias sets bias to None."""
        # Arrange
        w = np.ones((2, 3))
        converter = LASConverter(timesteps=8)

        # Act
        snn_layer = converter.convert_layer({"weight": w})

        # Assert
        assert snn_layer.bias is None

    def test_converted_layer_forward_matches_ann(self, rng: np.random.Generator) -> None:
        """Forward pass through converted layer matches ANN within tolerance."""
        # Arrange
        w = rng.standard_normal((6, 4)).astype(np.float64) * 0.2
        b = np.zeros(6)
        converter = LASConverter(timesteps=256, max_rate=1.0)
        snn_layer = converter.convert_layer({"weight": w, "bias": b})
        x = np.clip(rng.standard_normal((3, 4)) * 0.2, -0.3, 0.3)

        # Act
        snn_out = snn_layer.forward(x)
        ann_out = _ann_linear(w, b, x)

        # Assert
        rel_err = np.linalg.norm(snn_out - ann_out) / (np.linalg.norm(ann_out) + 1e-12)
        assert rel_err < 0.05, f"rel_err={rel_err:.4f}"

    def test_converter_timesteps_propagate(self) -> None:
        """Converter's timesteps are used by converted layers."""
        # Arrange
        converter = LASConverter(timesteps=42, max_rate=2.0)
        w = np.eye(3)

        # Act
        snn_layer = converter.convert_layer({"weight": w})

        # Assert
        assert snn_layer.timesteps == 42
        assert snn_layer.max_rate == 2.0


# -----------------------------------------------------------------------
# LASConverter.verify_equivalence
# -----------------------------------------------------------------------


class TestVerifyEquivalence:
    """verify_equivalence returns True for correctly converted layers."""

    def test_returns_true_for_correct_conversion(self, rng: np.random.Generator) -> None:
        """A properly converted layer passes equivalence check."""
        # Arrange
        w = rng.standard_normal((6, 4)).astype(np.float64) * 0.2
        b = np.zeros(6)
        converter = LASConverter(timesteps=256, max_rate=1.0)
        snn_layer = converter.convert_layer({"weight": w, "bias": b})

        def ann(x: np.ndarray) -> np.ndarray:
            return _ann_linear(w, b, x)

        sample = np.clip(rng.standard_normal((5, 4)) * 0.2, -0.3, 0.3)

        # Act
        result = converter.verify_equivalence(ann, snn_layer, sample, tol=0.05)

        # Assert
        assert result is True

    def test_returns_false_for_wrong_weights(self, rng: np.random.Generator) -> None:
        """Mismatched weights cause verify_equivalence to return False."""
        # Arrange
        w_snn = rng.standard_normal((6, 4)).astype(np.float64) * 0.3
        w_ann = rng.standard_normal((6, 4)).astype(np.float64) * 0.3
        b = np.zeros(6)
        converter = LASConverter(timesteps=64, max_rate=1.0)
        snn_layer = converter.convert_layer({"weight": w_snn, "bias": b})

        def ann(x: np.ndarray) -> np.ndarray:
            return _ann_linear(w_ann, b, x)

        sample = np.clip(rng.standard_normal((3, 4)) * 0.25, -0.5, 0.5)

        # Act
        result = converter.verify_equivalence(ann, snn_layer, sample, tol=1e-3)

        # Assert
        assert result is False

    def test_module_level_convenience_function(self, rng: np.random.Generator) -> None:
        """The module-level verify_equivalence() works identically."""
        # Arrange
        w = rng.standard_normal((4, 3)).astype(np.float64) * 0.2
        b = np.zeros(4)
        snn = convert_linear({"weight": w, "bias": b}, timesteps=256)

        def ann(x: np.ndarray) -> np.ndarray:
            return _ann_linear(w, b, x)

        sample = np.clip(rng.standard_normal((2, 3)) * 0.2, -0.3, 0.3)

        # Act & Assert
        assert verify_equivalence(ann, snn, sample, tol=0.05)

    def test_zero_ann_output_handling(self) -> None:
        """When ANN output is all zeros, verify still works (eps in denominator)."""
        # Arrange
        w = np.zeros((3, 2))
        b = np.zeros(3)
        converter = LASConverter(timesteps=16)
        snn = converter.convert_layer({"weight": w, "bias": b})

        def ann(x: np.ndarray) -> np.ndarray:
            return _ann_linear(w, b, x)

        sample = np.ones((1, 2))

        # Act — should not divide by zero
        result = converter.verify_equivalence(ann, snn, sample, tol=0.05)

        # Assert — both outputs are zero, so rel_err ~ 0
        assert result is True
