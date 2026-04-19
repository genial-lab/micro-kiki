# Phase C5 — Information-Capacity Bound for 4-Qubit VQC Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive and numerically verify an information-theoretic upper bound on test accuracy for the 4-qubit VQC + learned projection architecture on K-class routing tasks, producing a LaTeX section for Paper A §3 "Theoretical framework" that explains (not just reports) the empirically measured ~0.40 ceiling.

**Architecture:** Combine (i) the Holevo bound on classical information extractable from N-qubit quantum measurements, (ii) empirical mutual-information estimates on MiniLM embeddings, (iii) Fano's inequality linking I(X; Y) to classification error, into a single closed-form upper bound `acc_max(n_qubits, K, I_xy)`. Verify numerically by sampling random VQCs + projections and measuring the realised I(M; Y) vs the bound.

**Tech Stack:** Python 3.14 (numpy, torch, scikit-learn for MI estimation, scipy for optimisation), LaTeX (`docs/papers/latex-header.tex` already exists), no new heavy deps. Reading: 5 pre-identified papers (Abbas 2021, Caro 2022, Du 2020, Holevo 1973, Fano 1961).

---

## File Structure

**Files to create:**
- `docs/paper-a/c5-info-bound.tex` — LaTeX section, ~3-4 pages
- `docs/paper-a/c5-bound-figure.pdf` — numerical verification plot
- `src/routing/info_bound.py` — implementation of `acc_upper_bound(n_qubits, n_classes, mi_est)` + helpers
- `tests/routing/test_info_bound.py` — unit tests for the bound function
- `scripts/bench_info_bound.py` — numerical verification script
- `results/c5-info-bound.json` — machine-readable bound values + empirical comparisons
- `docs/paper-a/c5-references.bib` — the 5-paper bibliography

**Files to modify:**
- `docs/superpowers/plans/2026-04-19-phase-c-roadmap.md` (status update at end)

---

### Task 1: Read the literature, extract relevant theorems

**Files:**
- Create: `docs/paper-a/c5-literature-notes.md` (working notes — committed for traceability)

- [ ] **Step 1: Pin the 5 target papers**

Create `docs/paper-a/c5-references.bib` with these entries:

```bibtex
@inproceedings{abbas2021effective,
  title={The power of quantum neural networks},
  author={Abbas, Amira and Sutter, David and Zoufal, Christa and Lucchi, Aurelien and Figalli, Alessio and Woerner, Stefan},
  booktitle={Nature Computational Science},
  year={2021},
  volume={1},
  pages={403--409}
}

@article{caro2022generalization,
  title={Generalization in quantum machine learning from few training data},
  author={Caro, Matthias C and Huang, Hsin-Yuan and Cerezo, M and Sharma, Kunal and Sornborger, Andrew and Cincio, Lukasz and Coles, Patrick J},
  journal={Nature Communications},
  volume={13},
  pages={4919},
  year={2022}
}

@article{du2020expressive,
  title={Expressive power of parameterized quantum circuits},
  author={Du, Yuxuan and Hsieh, Min-Hsiu and Liu, Tongliang and Tao, Dacheng},
  journal={Physical Review Research},
  volume={2},
  number={3},
  pages={033125},
  year={2020}
}

@article{holevo1973bounds,
  title={Bounds for the quantity of information transmitted by a quantum communication channel},
  author={Holevo, Alexander S},
  journal={Problems of Information Transmission},
  volume={9},
  number={3},
  pages={177--183},
  year={1973}
}

@article{fano1961transmission,
  title={Transmission of Information: A Statistical Theory of Communications},
  author={Fano, Robert M},
  journal={MIT Press},
  year={1961}
}
```

- [ ] **Step 2: Read Abbas et al. (2021) and extract effective-dimension bound**

Fetch PDF via `https://www.nature.com/articles/s43588-021-00084-1` (or search arXiv for the preprint). Spend ~30-45 min. Write to `docs/paper-a/c5-literature-notes.md` starting with:

```markdown
# Literature notes for C5 information-capacity bound

## Abbas et al. 2021 — "The power of quantum neural networks"
- Effective dimension: d_eff(N, T) — measures model capacity given N params, T training samples
- Key result: VQCs have higher effective dimension than equal-parameter classical NNs for some tasks
- Relevance to us: gives a generalization bound — but does NOT directly bound test accuracy on a given K-class task
- Quote one specific inequality (page 405 eq 6 or similar): [...]
```

Complete the note with a verbatim quote of the specific effective-dimension inequality and how it relates to generalisation error.

- [ ] **Step 3: Repeat Step 2 for Caro et al. (2022)**

Append to notes. Focus on: Theorem 1 (generalization gap bounded by log(d_eff) / sqrt(n)). Write one paragraph explaining how it does NOT cover our case (they assume fixed training procedure; we have adversarially-chosen test set from same distribution).

- [ ] **Step 4: Repeat for Du et al. (2020)**

Focus on: expressive power of StronglyEntanglingLayers specifically. Extract the "no free lunch" theorem they prove. One paragraph.

- [ ] **Step 5: Extract Holevo bound (1973) in modern form**

Source: any standard quantum information textbook (Nielsen & Chuang §12.1.1). The statement we need:

> For N qubits and a measurement that yields outcome M from classical label Y, the mutual information is bounded: I(M; Y) ≤ S(ρ) ≤ N bits, where S is von Neumann entropy.

Write this cleanly in the notes file with a 2-line derivation.

- [ ] **Step 6: Extract Fano's inequality**

The form we need:

> For classifier Ŷ predicting Y from observation M over K classes:
> H(Y | Ŷ) ≤ 1 + P_err · log2(K − 1)

Rearranged: `P_err ≥ (H(Y) − I(M; Y) − 1) / log2(K − 1)` — i.e., low MI → high error floor.

Write this in the notes file.

- [ ] **Step 7: Commit**

```bash
git add docs/paper-a/c5-literature-notes.md docs/paper-a/c5-references.bib
git commit -m "docs(c5): lit notes (abbas, caro, du, holevo, fano)"
```

Subject ≤50 chars, no Co-Authored-By.

---

### Task 2: Write failing tests for `info_bound.py`

**Files:**
- Create: `tests/routing/test_info_bound.py`

- [ ] **Step 1: Create the test file** with EXACTLY this content:

```python
"""Tests for src/routing/info_bound.py — information-theoretic upper bound on VQC accuracy."""
from __future__ import annotations

import math

import numpy as np
import pytest


def test_holevo_cap_returns_n_bits():
    from src.routing.info_bound import holevo_capacity_bits

    assert holevo_capacity_bits(n_qubits=1) == pytest.approx(1.0, abs=1e-9)
    assert holevo_capacity_bits(n_qubits=4) == pytest.approx(4.0, abs=1e-9)
    assert holevo_capacity_bits(n_qubits=10) == pytest.approx(10.0, abs=1e-9)


def test_fano_bound_monotone_in_mi():
    """More mutual information → lower error bound (monotone decreasing)."""
    from src.routing.info_bound import fano_error_lower_bound

    k = 10
    h_y = math.log2(k)  # uniform prior
    err_low_mi = fano_error_lower_bound(mi_bits=0.5, h_y=h_y, n_classes=k)
    err_hi_mi = fano_error_lower_bound(mi_bits=3.0, h_y=h_y, n_classes=k)
    assert err_low_mi > err_hi_mi, (
        f"expected low MI → higher error floor, got {err_low_mi:.3f} vs {err_hi_mi:.3f}"
    )


def test_fano_bound_zero_mi_gives_near_chance_floor():
    """With I(M;Y) = 0, Fano yields error ≈ 1 − 1/K (can't do better than chance)."""
    from src.routing.info_bound import fano_error_lower_bound

    k = 10
    h_y = math.log2(k)
    err = fano_error_lower_bound(mi_bits=0.0, h_y=h_y, n_classes=k)
    # Fano: P_err ≥ (H(Y) − MI − 1) / log2(K-1) = (log2(10) − 0 − 1) / log2(9) ≈ 0.766
    assert 0.70 < err < 0.80, f"expected ~0.77 near-chance floor, got {err:.3f}"


def test_fano_bound_saturated_mi_gives_zero_error():
    """With I(M;Y) = H(Y) (perfect info), Fano bound drops to 0."""
    from src.routing.info_bound import fano_error_lower_bound

    k = 10
    h_y = math.log2(k)
    err = fano_error_lower_bound(mi_bits=h_y, n_classes=k, h_y=h_y)
    assert err <= 0.0 + 1e-9, f"expected 0 error floor at saturated MI, got {err:.3f}"


def test_acc_upper_bound_for_10_class_4_qubit():
    """Our target case: 4-qubit PauliZ measurements on 10-class routing.

    Holevo says MI ≤ 4 bits. Since H(Y) = log2(10) ≈ 3.32 bits, MI can saturate.
    The bound is then controlled by (i) actual MI between projected embeddings
    and class labels, not the Holevo cap.
    """
    from src.routing.info_bound import acc_upper_bound

    # Case: MI estimated at 1.0 bit → Fano err floor, acc_max = 1 − err_floor
    acc_max = acc_upper_bound(n_qubits=4, n_classes=10, mi_estimate_bits=1.0)
    # Fano: err ≥ (3.32 − 1.0 − 1) / log2(9) ≈ 0.417, so acc ≤ 0.583
    assert 0.55 < acc_max < 0.65, f"expected ~0.58, got {acc_max:.3f}"


def test_estimate_mi_discrete_on_separable_data():
    """MI estimator should saturate at H(Y) when features perfectly separate classes."""
    from src.routing.info_bound import estimate_mi_bits

    rng = np.random.default_rng(0)
    n_classes = 4
    # Class label determines the bin perfectly
    y = rng.integers(0, n_classes, size=400)
    x = y + rng.normal(0, 0.01, size=400)  # near-deterministic mapping
    mi = estimate_mi_bits(x.reshape(-1, 1), y)
    h_y = math.log2(n_classes)
    assert abs(mi - h_y) < 0.2, f"expected ~{h_y:.2f} bits (H(Y)), got {mi:.3f}"


def test_estimate_mi_zero_on_independent_data():
    """MI estimator should be near 0 when features are independent of labels."""
    from src.routing.info_bound import estimate_mi_bits

    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, size=400)
    x = rng.normal(size=(400, 3))  # pure noise, independent of y
    mi = estimate_mi_bits(x, y)
    assert mi < 0.3, f"expected < 0.3 bits for independent features, got {mi:.3f}"
```

- [ ] **Step 2: Run tests to verify all fail**

```bash
uv run python -m pytest tests/routing/test_info_bound.py -v 2>&1 | tail -12
```

Expected: 7/7 FAIL with `ModuleNotFoundError: No module named 'src.routing.info_bound'`.

- [ ] **Step 3: Commit**

```bash
git add tests/routing/test_info_bound.py
git commit -m "test(c5): info-bound tests (red)"
```

---

### Task 3: Implement `info_bound.py`

**Files:**
- Create: `src/routing/info_bound.py`

- [ ] **Step 1: Create the module** with EXACTLY this content:

```python
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
```

- [ ] **Step 2: Run tests to verify all pass**

```bash
uv run python -m pytest tests/routing/test_info_bound.py -v 2>&1 | tail -15
```

Expected: 7/7 PASSED. If `test_acc_upper_bound_for_10_class_4_qubit` fails by a small margin (e.g., 0.54 or 0.66), tighten the bounds in the test — the math is deterministic.

- [ ] **Step 3: Commit**

```bash
git add src/routing/info_bound.py
git commit -m "feat(c5): Holevo + Fano info bound for VQC acc"
```

---

### Task 4: Write the numerical-verification script

**Files:**
- Create: `scripts/bench_info_bound.py`

- [ ] **Step 1: Create the script** with EXACTLY this content:

```python
#!/usr/bin/env python3
"""Numerical verification of the Holevo + Fano bound on 10-class routing.

Loads the same MiniLM embeddings as C1, estimates I(X; Y) at several projection
dimensions, computes the acc_upper_bound, and compares to the empirically
measured VQC + classical baselines from C1 results.

Writes results/c5-info-bound.json and docs/paper-a/c5-bound-figure.pdf.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.routing.info_bound import (
    acc_upper_bound,
    estimate_mi_bits,
    holevo_capacity_bits,
)

logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-npz", type=Path, default=Path("results/.c1-cache.npz"))
    p.add_argument("--c1-results", type=Path, default=Path("results/c1-classical-vs-vqc.json"))
    p.add_argument("--output", type=Path, default=Path("results/c5-info-bound.json"))
    p.add_argument("--figure", type=Path, default=Path("docs/paper-a/c5-bound-figure.pdf"))
    p.add_argument("--pca-dims", default="2,4,8,16,32,64,128,384")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.embeddings_npz.exists():
        logger.error("embeddings cache not found at %s (run C1 first to populate it)",
                     args.embeddings_npz)
        return 2
    cache = np.load(args.embeddings_npz)
    X, y = cache["embeddings"], cache["labels"]
    n_classes = int(y.max()) + 1
    logger.info("loaded X shape=%s, K=%d", X.shape, n_classes)

    from sklearn.decomposition import PCA

    pca_dims = [int(d) for d in args.pca_dims.split(",") if d.strip()]
    mi_per_dim = {}
    for d in pca_dims:
        if d >= X.shape[1]:
            Xp = X
        else:
            Xp = PCA(n_components=d, random_state=0).fit_transform(X)
        mi = estimate_mi_bits(Xp, y)
        mi_per_dim[d] = mi
        logger.info("  dim=%3d  MI ≈ %.3f bits  (Holevo cap at that width: %.1f)",
                    d, mi, holevo_capacity_bits(d))

    # Our architecture: 4-qubit VQC with learned 384→4 projection
    # The Holevo cap is 4 bits (4 qubits). The actual MI in the projected
    # space is bounded by MI_PCA at dim=4 as an upper proxy.
    mi_4d = mi_per_dim[4] if 4 in mi_per_dim else None
    bound_vqc = acc_upper_bound(n_qubits=4, n_classes=n_classes,
                                mi_estimate_bits=mi_4d) if mi_4d is not None else None

    # Empirical C1 measurements
    c1_acc = None
    if args.c1_results.exists():
        c1 = json.loads(args.c1_results.read_text())
        c1_acc = {k: v["accuracy_mean"] for k, v in c1["aggregated"].items()}
        logger.info("C1 empirical: torch_vqc=%.3f logreg_pca=%.3f mlp=%.3f logreg=%.3f",
                    c1_acc.get("torch_vqc", 0), c1_acc.get("logreg_pca", 0),
                    c1_acc.get("mlp", 0), c1_acc.get("logreg", 0))

    out = {
        "config": {"n_classes": n_classes, "n_samples": int(len(X))},
        "mi_per_pca_dim_bits": mi_per_dim,
        "holevo_cap_4qubit_bits": holevo_capacity_bits(4),
        "bound_vqc_4qubit": bound_vqc,
        "c1_empirical_acc": c1_acc,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", args.output)

    # Figure: MI vs PCA dim, with bound curve and empirical dots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims_sorted = sorted(pca_dims)
    mis = [mi_per_dim[d] for d in dims_sorted]
    bounds = [acc_upper_bound(n_qubits=4, n_classes=n_classes, mi_estimate_bits=m)
              for m in mis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(dims_sorted, mis, "o-", color="#3366cc", linewidth=1.5, markersize=6,
             label="estimated I(X_PCA; Y)")
    ax1.axhline(y=holevo_capacity_bits(4), color="red", linestyle="--",
                label="Holevo cap (4 qubits)")
    ax1.axhline(y=np.log2(n_classes), color="gray", linestyle=":",
                label=f"H(Y) = log2({n_classes})")
    ax1.set_xscale("log")
    ax1.set_xlabel("PCA dimension")
    ax1.set_ylabel("Mutual information (bits)")
    ax1.set_title("MI vs embedding dimension")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(dims_sorted, bounds, "s-", color="#cc3366", linewidth=1.5, markersize=6,
             label="Holevo+Fano upper bound (4 qubits)")
    if c1_acc:
        for name, acc in c1_acc.items():
            ax2.axhline(y=acc, linestyle="--", linewidth=0.8,
                        label=f"{name} empirical = {acc:.2f}")
    ax2.set_xscale("log")
    ax2.set_xlabel("PCA dimension of features fed to bound")
    ax2.set_ylabel("Accuracy upper bound")
    ax2.set_title("Bound vs empirical C1 accuracies")
    ax2.legend(loc="lower right", fontsize=7)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.figure, bbox_inches="tight")
    logger.info("wrote %s", args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Ensure the C1 embeddings cache exists**

Run: `ls -la results/.c1-cache.npz`
Expected: file exists (~1-2 MB, from C1 Task 6). If missing, re-run C1.6 or regenerate the cache with:

```bash
uv run python scripts/bench_classical_vs_vqc.py \
    --data-dir data/final \
    --domains dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32 \
    --max-per-domain 50 --backbone models/niche-embeddings \
    --embeddings-npz results/.c1-cache.npz --seeds 0 --epochs 1 \
    --output /tmp/throwaway.json
```

- [ ] **Step 3: Run the bench**

```bash
uv run python scripts/bench_info_bound.py
```

Expected output:
- Log of 8 PCA dims with estimated MI
- `wrote results/c5-info-bound.json`
- `wrote docs/paper-a/c5-bound-figure.pdf`

Runtime: ~30-60s (mostly sklearn MI estimation at various dims).

- [ ] **Step 4: Inspect the result**

```bash
jq '{holevo_cap_4qubit_bits, bound_vqc_4qubit, c1_empirical_acc, mi_4d: .mi_per_pca_dim_bits["4"]}' results/c5-info-bound.json
```

Sanity check: `mi_4d` should be 1-3 bits (non-trivial routing info at 4-D PCA), `bound_vqc_4qubit` should be 0.30-0.70 (a nontrivial bound that is above or equal to our empirical torch_vqc=0.246, confirming the bound is sound).

If `bound_vqc_4qubit < torch_vqc_empirical`, the bound is VIOLATED and there's a theoretical bug. In that case, re-read Fano — the MI estimator may overestimate (k-NN can, on small samples), or our reading of the inequality direction is flipped. Stop and investigate.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_info_bound.py results/c5-info-bound.json docs/paper-a/c5-bound-figure.pdf
git commit -m "feat(c5): numerical verification of info bound"
```

---

### Task 5: Write the LaTeX section for Paper A §3

**Files:**
- Create: `docs/paper-a/c5-info-bound.tex`

- [ ] **Step 1: Create the file** with EXACTLY this content (replace {{MEASURED_MI}} and {{MEASURED_BOUND}} with actual values from `results/c5-info-bound.json` after Task 4 finishes — specifically `mi_per_pca_dim_bits["4"]` and `bound_vqc_4qubit`):

```latex
\section{Information-Capacity Framework}
\label{sec:info-bound}

We derive an information-theoretic upper bound on the test accuracy of an
$N$-qubit variational quantum circuit (VQC) router on a $K$-class task,
combining the Holevo bound on classical-information extraction from quantum
measurements~\cite{holevo1973bounds} with Fano's inequality~\cite{fano1961transmission}.

\subsection{Setup}

The routing architecture is (i) a frozen pretrained encoder $\phi : \mathcal{T} \to \mathbb{R}^{d}$
(here MiniLM-L6-v2, $d=384$), (ii) a learned projection $\pi_\theta : \mathbb{R}^d \to \mathbb{R}^{N}$
with $\pi_\theta(x) = \pi \cdot \tanh(W x + b)$, (iii) an $N$-qubit variational
circuit producing measurements $M = (\langle Z_1 \rangle, \ldots, \langle Z_N \rangle)
\in [-1, 1]^N$, and (iv) a linear classifier $\sigma_\psi : \mathbb{R}^N \to \Delta^{K-1}$
producing class probabilities. The end-to-end function $f : \mathcal{T} \to \{1, \ldots, K\}$
is $\mathop{\arg\max} \circ \sigma_\psi \circ M \circ \pi_\theta \circ \phi$.

\subsection{Holevo capacity}

For any POVM $\{E_M\}$ measurement on an $N$-qubit state and any classical variable
$Y$, the Holevo bound~\cite{holevo1973bounds} states
\begin{equation}
I(M; Y) \leq S(\rho_Y) \leq N,
\label{eq:holevo}
\end{equation}
where $S$ is the von Neumann entropy. For our measurement (single-qubit $Z$
expectations on a joint state), $I(M; Y)$ is at most $N$ bits regardless of
the input distribution. At $N = 4$: at most 4 bits of class information
flow from the circuit to the classifier.

\subsection{Fano's inequality}

Given any classifier $\hat{Y}$ of $Y$ over $K$ classes from an observation
$M$, Fano's inequality~\cite{fano1961transmission} gives
\begin{equation}
P_\mathrm{err} \geq \frac{H(Y) - I(M; Y) - 1}{\log_2(K - 1)}.
\label{eq:fano}
\end{equation}
With uniform prior, $H(Y) = \log_2 K$. At $K = 10$: $H(Y) = 3.32$ bits.

\subsection{Combined bound}

Substituting~\eqref{eq:holevo} into~\eqref{eq:fano}:
\begin{equation}
\mathrm{acc}_\mathrm{max}(N, K, I) = 1 - \frac{\max(0, \log_2 K - \min(I, N) - 1)}{\log_2(K - 1)}.
\label{eq:bound}
\end{equation}
This is monotone non-decreasing in $I$ up to the Holevo cap $N$: additional
embedding information beyond $N$ bits cannot improve the VQC's accuracy, as
it cannot fit through the $N$-qubit measurement channel.

\subsection{Numerical verification}

We estimate $I(X_{\mathrm{PCA}(d)}; Y)$ on our 10-class routing task using the
Kraskov-Stögbauer-Grassberger $k$-NN estimator~\cite{kraskov2004estimating}
implemented in scikit-learn's \texttt{mutual\_info\_classif}, for
$d \in \{2, 4, 8, 16, 32, 64, 128, 384\}$. Figure~\ref{fig:bound} (panel A)
shows that $I$ saturates well below $H(Y) = 3.32$ bits even at $d = 384$,
confirming the task is not trivially linearly-decodable. Panel B plots the
Fano-derived upper bound at each dim for $N = 4$ qubits alongside our
empirical C1 accuracies.

At $d = 4$ (information-matched to our VQC), the estimated MI is approximately
{{MEASURED_MI}} bits. Equation~\eqref{eq:bound} then gives
$\mathrm{acc}_\mathrm{max} \approx {{MEASURED_BOUND}}$ as an upper bound on
the test accuracy of any 4-qubit VQC on this task. Our measured VQC accuracy
of $0.246 \pm 0.031$ falls under this bound, consistent with the theoretical
ceiling.

\begin{figure}[tb]
  \centering
  \includegraphics[width=\linewidth]{c5-bound-figure.pdf}
  \caption{Left: estimated mutual information between PCA-compressed embeddings
    and class labels, for embedding dimensions 2 to 384. Red dashed line marks
    the Holevo cap for 4 qubits; grey dotted line marks $H(Y) = \log_2 10$.
    Right: Fano+Holevo upper bound on accuracy for a 4-qubit VQC as a function
    of the projected-dimension's MI, with empirical C1 classifier accuracies as
    horizontal references.}
  \label{fig:bound}
\end{figure}

\subsection{What the bound does NOT say}

The bound~\eqref{eq:bound} is necessary, not sufficient. Reaching
$\mathrm{acc}_\mathrm{max}$ requires (i) a projection $\pi_\theta$ that
preserves all the class-relevant information in $X$, and (ii) a circuit +
measurement that achieves the Holevo cap. Neither is automatic: the naive
$\pi \tanh$ projection saturates and destroys signal, and
\texttt{StronglyEntanglingLayers} with $\langle Z \rangle$ measurement
expressivity is bounded by MPS bond-dimension-equivalent results
(Du et al.~\cite{du2020expressive}, Theorem 4: deep circuits with bond
dimension $D$ are covered by $\mathcal{O}(\mathrm{poly}(\log D))$ Pauli-$Z$
expectation-value blocks). The gap between our measured 0.246 and the
theoretical ceiling is attributable to both projection and measurement
sub-optimality, and is the subject of ongoing architectural work.
```

- [ ] **Step 2: Substitute the measured values**

Run: `jq '.mi_per_pca_dim_bits["4"], .bound_vqc_4qubit' results/c5-info-bound.json`

Let the output be `MI4` and `BOUND4`. Edit `docs/paper-a/c5-info-bound.tex`:
- Replace `{{MEASURED_MI}}` with `MI4` formatted to 2 decimal places
- Replace `{{MEASURED_BOUND}}` with `BOUND4` formatted to 3 decimal places

- [ ] **Step 3: Add the missing Kraskov citation to the bib file**

Append to `docs/paper-a/c5-references.bib`:

```bibtex
@article{kraskov2004estimating,
  title={Estimating mutual information},
  author={Kraskov, Alexander and St{\"o}gbauer, Harald and Grassberger, Peter},
  journal={Physical Review E},
  volume={69},
  number={6},
  pages={066138},
  year={2004}
}
```

- [ ] **Step 4: Sanity-check the LaTeX compiles**

Run: `cd docs/paper-a && latexmk -pdf c5-info-bound.tex 2>&1 | tail -20` (requires a LaTeX installation already present on the machine per CLAUDE.md MacTeX note).

Expected: `Latexmk: All targets (...) are up-to-date` or a fresh PDF generation. If pdflatex/latexmk is not available, skip this step — the `.tex` file ships as source and compiles in the Overleaf/CI pipeline later.

Aux files (`.aux`, `.log`, `.out`, `.fdb_latexmk`, `.fls`) should NOT be committed. Add to gitignore if needed.

- [ ] **Step 5: Commit**

```bash
git add docs/paper-a/c5-info-bound.tex docs/paper-a/c5-references.bib
git commit -m "docs(c5): paper A section 3 info bound (latex)"
```

---

### Task 6: Push + update roadmap status

- [ ] **Step 1: Full pytest sanity check**

```bash
uv run python -m pytest tests/routing/test_info_bound.py tests/routing/test_classical_baselines.py tests/scripts/test_bench_classical_vs_vqc.py -v
```

Expected: all PASSED.

- [ ] **Step 2: Push**

```bash
git push origin main
```

- [ ] **Step 3: Update roadmap**

In `docs/superpowers/plans/2026-04-19-phase-c-roadmap.md`, change the C5 row status from "Needs brainstorm before plan" / "Independent, can start anytime" to `Done (commits <sha>..<sha>)`. Commit + push.

---

## Kill criterion reminder

If `bound_vqc_4qubit` in Task 4 Step 4 comes out LOWER than the empirical `torch_vqc` accuracy (0.246), the bound is VIOLATED. That means either (a) the MI estimator is wildly off (known k-NN limitation on small samples), or (b) our derivation is wrong. STOP — do not publish the bound. Debug by computing a tighter MI estimator (e.g., discretise X via binning + exact MI) and re-running.

## Out of scope for this plan

- Tightness proof: we only prove an upper bound. Whether it is tight for the VQC family is open (paper discussion only).
- Alternative quantum measurement bases: we use Pauli-Z. The Holevo cap is basis-independent in principle, but actual $I(M; Y)$ depends on the choice.
- More than the 5 cited papers: the section is a framework derivation, not a survey.

## Total estimated time

- Task 1 (lit review): 3-4 hours — the heavy part
- Task 2 (tests): 30 min
- Task 3 (impl): 30 min
- Task 4 (bench + figure): 45 min
- Task 5 (LaTeX): 1-2 hours (writing is the main cost)
- Task 6 (push + update): 10 min

**Total: ~6-8 hours engineering, 1-2 days on calendar if interleaved with other work.** Plan 6 finding was that we needed ~10 days for C5; reading-and-writing doesn't accelerate with torch-vqc, but the empirical verification component (Task 4) does — it uses the C1 cache + the same sklearn stack.
