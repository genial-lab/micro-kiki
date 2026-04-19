"""Training tests: autograd loop reduces loss and reaches non-trivial accuracy.

TDD: verify that TorchVQCRouter learns on a small synthetic task, replacing
PennyLane's parameter-shift training with torch autograd.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


def _make_synthetic_task(n_classes: int = 4, samples_per_class: int = 20, seed: int = 0):
    """Each class has embeddings clustered around a distinct mean in first 6 dims.

    Classifiable by the first 6 features alone — what the VQC sees.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        center = rng.uniform(0.5, 2 * np.pi - 0.5, size=6)  # stay away from 0/2π
        for _ in range(samples_per_class):
            noise = rng.normal(0, 0.15, size=10)
            x = np.zeros(10)
            x[:6] = center + noise[:6]
            x[6:] = noise[6:]
            X.append(x)
            y.append(c)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def test_training_reduces_loss():
    from src.routing.torch_vqc_router import TorchVQCRouter

    X, y = _make_synthetic_task(n_classes=4, samples_per_class=10, seed=0)
    model = TorchVQCRouter(n_qubits=6, n_layers=6, n_classes=4, lr=0.1, seed=42)

    X_t = torch.from_numpy(X).double()
    y_t = torch.from_numpy(y)

    losses = model.train_batched(X_t, y_t, epochs=30)

    assert len(losses) == 30
    assert losses[-1] < losses[0] - 0.05, (
        f"expected loss decrease >0.05, got {losses[0]:.4f} → {losses[-1]:.4f}"
    )


def test_training_reaches_nontrivial_accuracy():
    """After training, train-set accuracy should clearly beat chance (25% for 4 classes)."""
    from src.routing.torch_vqc_router import TorchVQCRouter

    X, y = _make_synthetic_task(n_classes=4, samples_per_class=15, seed=1)
    model = TorchVQCRouter(n_qubits=6, n_layers=6, n_classes=4, lr=0.1, seed=7)

    X_t = torch.from_numpy(X).double()
    y_t = torch.from_numpy(y)
    model.train_batched(X_t, y_t, epochs=50)

    with torch.no_grad():
        logits = model.forward(X_t)
        preds = logits.argmax(dim=-1)
        acc = (preds == y_t).float().mean().item()

    assert acc > 0.50, f"expected >50% (well above 25% chance), got {acc:.3f}"


def test_gradients_flow_through_vqc_weights():
    """Sanity: VQC weights must receive non-zero gradient."""
    from src.routing.torch_vqc_router import TorchVQCRouter

    X, y = _make_synthetic_task(n_classes=3, samples_per_class=5, seed=2)
    model = TorchVQCRouter(n_qubits=6, n_layers=6, n_classes=3, lr=0.05, seed=3)
    X_t = torch.from_numpy(X).double()
    y_t = torch.from_numpy(y)

    # One forward + backward
    logits = model.forward(X_t)
    loss = torch.nn.functional.cross_entropy(logits, y_t)
    loss.backward()

    assert model.vqc_weights.grad is not None
    assert model.vqc_weights.grad.abs().max().item() > 1e-6, (
        "VQC weight gradient is zero — autograd not flowing through circuit"
    )
    assert model.linear_w.grad is not None
    assert model.linear_w.grad.abs().max().item() > 1e-6


def test_training_faster_than_pennylane_baseline():
    """Benchmark: torch autograd training must be >=20× faster than PennyLane parameter-shift."""
    import time

    from src.routing.torch_vqc_router import TorchVQCRouter
    from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig

    X, y = _make_synthetic_task(n_classes=4, samples_per_class=10, seed=4)
    epochs = 3

    # Torch autograd (full batch)
    model = TorchVQCRouter(n_qubits=6, n_layers=6, n_classes=4, lr=0.05, seed=0)
    X_t = torch.from_numpy(X).double()
    y_t = torch.from_numpy(y)
    _ = model.train_batched(X_t[:5], y_t[:5], epochs=1)  # warmup
    t0 = time.perf_counter()
    model.train_batched(X_t, y_t, epochs=epochs)
    torch_time = time.perf_counter() - t0

    # PennyLane reference
    cfg = QuantumRouterConfig(n_qubits=6, n_layers=6, n_classes=4, learning_rate=0.05)
    vqc = QuantumRouter(cfg)
    t0 = time.perf_counter()
    vqc.train(X, y, epochs=epochs)
    pl_time = time.perf_counter() - t0

    speedup = pl_time / torch_time
    print(f"\nSpeedup: {speedup:.1f}× (torch {torch_time*1000:.0f}ms vs PL {pl_time*1000:.0f}ms)")
    assert speedup >= 20.0, f"expected ≥20× speedup, got {speedup:.1f}×"
