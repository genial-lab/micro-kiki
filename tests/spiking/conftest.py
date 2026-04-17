"""Shared fixtures for spiking tests (story-25)."""

from __future__ import annotations

import numpy as np
import pytest

from src.spiking.lif_neuron import LIFNeuron


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def default_neuron() -> LIFNeuron:
    """LIF neuron with default parameters (threshold=1.0, tau=1.0)."""
    return LIFNeuron(threshold=1.0, tau=1.0)
