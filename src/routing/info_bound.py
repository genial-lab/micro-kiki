"""Information-theoretic upper bound on VQC test accuracy.

Combines Holevo capacity (N qubits → max N bits of classical info extractable)
with Fano's inequality (classifier error floor given I(M; Y)) to produce a
single function `acc_upper_bound(n_qubits, n_classes, mi_estimate_bits)`.

The MI estimator uses sklearn's `mutual_info_classif` (k-NN based,
Kraskov-Stögbauer-Grassberger estimator).
"""
from __future__ import annotations

import math

import numpy as np


def holevo_capacity_bits(n_qubits: int) -> float:
    """Holevo bound: max classical info extractable from N qubits = N bits."""
    return float(n_qubits)


def fano_error_lower_bound(mi_bits: float, n_classes: int, h_y: float | None = None) -> float:
    """Fano inequality lower bound on classification error.

    P_err >= (H(Y) - I(M; Y) - 1) / log2(K - 1)

    Args:
        mi_bits: estimated mutual information I(M; Y) in bits.
        n_classes: K.
        h_y: entropy of the label distribution (defaults to uniform = log2(K)).

    Returns:
        Lower bound on P_err, clipped to [0, 1].
    """
    if h_y is None:
        h_y = math.log2(n_classes)
    if n_classes <= 1:
        return 0.0
    bound = (h_y - mi_bits - 1.0) / math.log2(n_classes - 1)
    return float(max(0.0, min(1.0, bound)))


def acc_upper_bound(n_qubits: int, n_classes: int, mi_estimate_bits: float,
                    h_y: float | None = None) -> float:
    """Upper bound on test accuracy: 1 - Fano_error(min(MI, Holevo)).

    Caps the MI at the Holevo capacity — the VQC cannot extract more than
    N bits of class information from N qubits, regardless of how informative
    the embedding is.
    """
    effective_mi = min(mi_estimate_bits, holevo_capacity_bits(n_qubits))
    err_floor = fano_error_lower_bound(effective_mi, n_classes, h_y=h_y)
    return 1.0 - err_floor


def estimate_mi_bits(X: np.ndarray, y: np.ndarray, *, n_neighbors: int = 3,
                     random_state: int = 0) -> float:
    """Estimate I(X; Y) in bits using sklearn's k-NN-based estimator.

    Sums per-feature MI (overestimates for correlated features, but tight enough
    for our Holevo-comparison use case on 4-to-10 dim post-projection features).
    """
    from sklearn.feature_selection import mutual_info_classif

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    # mutual_info_classif returns MI in nats by default
    mi_nats = mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=random_state)
    # Convert nats → bits
    return float(mi_nats.sum() / math.log(2))
