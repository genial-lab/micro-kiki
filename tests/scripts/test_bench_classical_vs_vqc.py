"""Integration test: bench_classical_vs_vqc.py produces a valid JSON report."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_synthetic_embeddings(tmp_path):
    """Simulate what the real bench would see: a (n_samples, 384) embedding matrix
    + integer labels, saved as .npz so the bench can load without SentenceTransformer.
    """
    rng = np.random.default_rng(0)
    n_classes = 4
    per_class = 25
    centers = rng.uniform(-1.5, 1.5, size=(n_classes, 384))
    X, y = [], []
    for c in range(n_classes):
        for _ in range(per_class):
            X.append(centers[c] + rng.normal(0, 0.3, size=384))
            y.append(c)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)

    cache = tmp_path / "embs.npz"
    np.savez(cache, embeddings=X, labels=y)
    return cache


def test_bench_produces_valid_json(tmp_path, tmp_synthetic_embeddings):
    output = tmp_path / "c1-out.json"
    cmd = [
        sys.executable,
        "scripts/bench_classical_vs_vqc.py",
        "--embeddings-npz", str(tmp_synthetic_embeddings),
        "--output", str(output),
        "--seeds", "0,1",
        "--epochs", "30",          # tiny for CI speed
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"stderr: {r.stderr}\nstdout: {r.stdout}"
    assert output.exists()

    data = json.loads(output.read_text())
    # Expected: one entry per baseline × seed + an aggregated summary
    assert "runs" in data and len(data["runs"]) == 5 * 2  # 5 baselines × 2 seeds
    assert "aggregated" in data
    assert set(data["aggregated"].keys()) == {
        "stratified", "logreg", "logreg_pca", "mlp", "torch_vqc"
    }
    for name, agg in data["aggregated"].items():
        assert "accuracy_mean" in agg
        assert "accuracy_std" in agg
        assert 0.0 <= agg["accuracy_mean"] <= 1.0
