"""Verify torch VQC forward matches PennyLane reference numerically.

TDD: this test must fail before torch_vqc.py exists, pass after.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
import torch


N_QUBITS = 6
N_LAYERS = 6


def _pennylane_reference(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Run the identical circuit through PennyLane — ground truth."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def circuit(w, f):
        qml.AngleEmbedding(f[:N_QUBITS], wires=range(N_QUBITS))
        qml.StronglyEntanglingLayers(w, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return np.array(circuit(weights, features))


def test_forward_matches_pennylane_single_sample():
    from src.routing.torch_vqc import torch_vqc_forward

    rng = np.random.default_rng(0)
    features = rng.uniform(0.0, 2 * np.pi, size=10).astype(np.float64)
    weights = rng.uniform(0.0, 2 * np.pi, size=(N_LAYERS, N_QUBITS, 3)).astype(np.float64)

    ref = _pennylane_reference(features, weights)  # shape (N_QUBITS,)

    f_t = torch.from_numpy(features).double()
    w_t = torch.from_numpy(weights).double()
    out = torch_vqc_forward(f_t, w_t, n_qubits=N_QUBITS, n_layers=N_LAYERS)

    assert out.shape == (N_QUBITS,), f"expected ({N_QUBITS},), got {tuple(out.shape)}"
    np.testing.assert_allclose(
        out.detach().cpu().numpy(),
        ref,
        atol=1e-5,
        err_msg=f"torch={out}, pennylane={ref}",
    )


def test_forward_matches_pennylane_batched():
    from src.routing.torch_vqc import torch_vqc_forward

    rng = np.random.default_rng(1)
    B = 3
    features = rng.uniform(0.0, 2 * np.pi, size=(B, 10)).astype(np.float64)
    weights = rng.uniform(0.0, 2 * np.pi, size=(N_LAYERS, N_QUBITS, 3)).astype(np.float64)

    refs = np.stack([_pennylane_reference(features[b], weights) for b in range(B)], axis=0)

    f_t = torch.from_numpy(features).double()
    w_t = torch.from_numpy(weights).double()
    out = torch_vqc_forward(f_t, w_t, n_qubits=N_QUBITS, n_layers=N_LAYERS)

    assert out.shape == (B, N_QUBITS), f"expected ({B}, {N_QUBITS}), got {tuple(out.shape)}"
    np.testing.assert_allclose(
        out.detach().cpu().numpy(),
        refs,
        atol=1e-5,
        err_msg=f"batched mismatch: torch={out}, pennylane={refs}",
    )


def test_forward_different_seeds_different_outputs():
    """Sanity check: distinct weights → distinct outputs (not constant function)."""
    from src.routing.torch_vqc import torch_vqc_forward

    rng = np.random.default_rng(42)
    features = rng.uniform(0.0, 2 * np.pi, size=10).astype(np.float64)
    w1 = rng.uniform(0.0, 2 * np.pi, size=(N_LAYERS, N_QUBITS, 3)).astype(np.float64)
    w2 = rng.uniform(0.0, 2 * np.pi, size=(N_LAYERS, N_QUBITS, 3)).astype(np.float64)

    f_t = torch.from_numpy(features).double()
    o1 = torch_vqc_forward(f_t, torch.from_numpy(w1).double(), n_qubits=N_QUBITS, n_layers=N_LAYERS)
    o2 = torch_vqc_forward(f_t, torch.from_numpy(w2).double(), n_qubits=N_QUBITS, n_layers=N_LAYERS)

    assert not torch.allclose(o1, o2, atol=1e-3), "different weights should produce different outputs"
