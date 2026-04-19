"""Tests for src/data/corpus_validator.py — cluster vs existing taxonomy matching."""
from __future__ import annotations

import numpy as np
import pytest


def test_hungarian_match_perfect_overlap():
    """When K clusters exactly match K domains (after permutation), overlap = 1.0."""
    from src.data.corpus_validator import match_clusters_to_domains

    # 3 domains, 3 clusters, perfectly separated after relabelling
    true_domain = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    cluster_id  = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])  # permuted
    m = match_clusters_to_domains(true_domain, cluster_id, n_domains=3)
    assert m["mean_overlap"] == pytest.approx(1.0, abs=1e-9)
    assert m["assignment"] == {2: 0, 0: 1, 1: 2}


def test_hungarian_match_zero_overlap():
    """Entirely mismatched clusters give mean_overlap ~ 1 / n_domains (chance)."""
    from src.data.corpus_validator import match_clusters_to_domains

    rng = np.random.default_rng(0)
    true_domain = rng.integers(0, 5, size=200)
    cluster_id = rng.integers(0, 5, size=200)
    m = match_clusters_to_domains(true_domain, cluster_id, n_domains=5)
    assert 0.1 < m["mean_overlap"] < 0.35


def test_hungarian_match_unequal_k_and_n_domains():
    """More clusters than domains: only n_domains clusters get assigned."""
    from src.data.corpus_validator import match_clusters_to_domains

    true_domain = np.array([0, 0, 0, 1, 1, 1])
    cluster_id  = np.array([0, 0, 1, 2, 2, 3])  # 4 clusters, 2 domains
    m = match_clusters_to_domains(true_domain, cluster_id, n_domains=2)
    assert len(m["assignment"]) == 2  # only 2 clusters mapped to domains
    assert m["mean_overlap"] >= 0.5  # better than chance


def test_cluster_embeddings_returns_labels():
    from src.data.corpus_validator import cluster_embeddings_hdbscan

    rng = np.random.default_rng(0)
    # 3 well-separated clusters
    centers = rng.uniform(-5, 5, size=(3, 16))
    X = np.vstack([centers[i] + rng.normal(0, 0.3, size=(30, 16)) for i in range(3)])
    labels = cluster_embeddings_hdbscan(X, min_cluster_size=10)
    assert len(labels) == 90
    uniq = set(labels) - {-1}
    assert 2 <= len(uniq) <= 4, f"expected ~3 clusters, got {len(uniq)}"
