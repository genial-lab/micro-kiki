"""Tests for src/routing/classical_baselines.py run_classical_baseline dispatcher."""
from __future__ import annotations

import numpy as np
import pytest


def _make_separable_task(n_classes: int = 4, per_class: int = 40, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-2.0, 2.0, size=(n_classes, dim))
    X, y = [], []
    for c in range(n_classes):
        for _ in range(per_class):
            X.append(centers[c] + rng.normal(0, 0.2, size=dim))
            y.append(c)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def test_run_baseline_stratified_returns_expected_shape():
    from src.routing.classical_baselines import run_classical_baseline

    X, y = _make_separable_task(n_classes=4, per_class=20, seed=0)
    n_tr = int(0.8 * len(X))
    out = run_classical_baseline("stratified", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], seed=0)
    assert set(out.keys()) == {"name", "accuracy", "macro_f1", "train_time_s", "n_params"}
    assert out["name"] == "stratified"
    assert 0.0 <= out["accuracy"] <= 1.0
    assert 0.0 <= out["macro_f1"] <= 1.0
    assert out["train_time_s"] >= 0.0
    assert out["n_params"] >= 0


def test_run_baseline_logreg_beats_chance_on_separable_task():
    from src.routing.classical_baselines import run_classical_baseline

    X, y = _make_separable_task(n_classes=4, per_class=30, seed=1)
    n_tr = int(0.8 * len(X))
    out = run_classical_baseline("logreg", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], seed=0)
    assert out["accuracy"] > 0.85, f"LogReg should ace separable task, got {out['accuracy']:.3f}"


def test_run_baseline_logreg_pca_matches_logreg_when_pca_dim_eq_input_dim():
    from src.routing.classical_baselines import run_classical_baseline

    X, y = _make_separable_task(n_classes=3, per_class=20, dim=8, seed=2)
    n_tr = int(0.8 * len(X))
    a = run_classical_baseline("logreg", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], seed=0)
    b = run_classical_baseline("logreg_pca", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], seed=0, pca_dim=8)
    # PCA with pca_dim == input_dim is lossless (up to rotation), LogReg should match
    assert abs(a["accuracy"] - b["accuracy"]) < 0.05


def test_run_baseline_mlp_returns_param_count():
    from src.routing.classical_baselines import run_classical_baseline

    X, y = _make_separable_task(n_classes=4, per_class=20, dim=16, seed=3)
    n_tr = int(0.8 * len(X))
    out = run_classical_baseline("mlp", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], seed=0, hidden_dim=8)
    # MLP: (16*8 + 8) + (8*4 + 4) = 172
    expected = 16 * 8 + 8 + 8 * 4 + 4
    assert out["n_params"] == expected, f"expected {expected}, got {out['n_params']}"


def test_run_baseline_torch_vqc_returns_result():
    from src.routing.classical_baselines import run_classical_baseline

    X, y = _make_separable_task(n_classes=3, per_class=20, dim=32, seed=4)
    n_tr = int(0.8 * len(X))
    out = run_classical_baseline(
        "torch_vqc", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:],
        seed=0, n_qubits=4, n_layers=6, epochs=30,
    )
    assert 0.0 <= out["accuracy"] <= 1.0
    assert out["n_params"] > 0


def test_run_baseline_unknown_name_raises():
    from src.routing.classical_baselines import run_classical_baseline

    X, y = _make_separable_task(n_classes=2, per_class=10, seed=5)
    n_tr = int(0.8 * len(X))
    with pytest.raises(ValueError, match="unknown baseline"):
        run_classical_baseline("bogus", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:])
