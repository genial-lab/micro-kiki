#!/usr/bin/env python3
"""Benchmark quantum VQC router vs classical sigmoid router.

Generates synthetic embeddings for 11-domain subset (dsp, electronics, emc, ..., base),
trains both routers, then compares accuracy, per-classification latency, and parameter count.
Note: This benchmark intentionally uses a reduced 11-domain subset for fast iteration;
the production VQC router handles all 35 domains.

Results are written to results/quantum-router-benchmark.json.

Usage:
    uv run python scripts/benchmark_quantum_router.py --help
    uv run python scripts/benchmark_quantum_router.py
    uv run python scripts/benchmark_quantum_router.py --n-samples 200 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESULTS_DIR = Path(__file__).parent.parent / "results"

# 11-domain synthetic subset for benchmarking (does not match full router._ALL_DOMAINS which has 35)
# The production router supports all 35 domains; this benchmark uses a reduced subset for fast local testing.
_ALL_DOMAINS = [
    "dsp",
    "electronics",
    "emc",
    "embedded",
    "freecad",
    "kicad-dsl",
    "platformio",
    "power",
    "spice",
    "stm32",
    "base",
]
N_CLASSES = len(_ALL_DOMAINS)  # 11 (synthetic subset, not production scale)


@dataclass
class RouterMetrics:
    """Results for one router variant."""

    name: str
    accuracy: float
    latency_ms_mean: float
    latency_ms_p95: float
    n_parameters: int
    train_loss_first: float
    train_loss_last: float


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def make_synthetic_data(
    n_samples: int,
    embedding_dim: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate linearly separable embeddings per class.

    Each class is a Gaussian cluster centred at a distinct location in
    ``embedding_dim``-space. This guarantees trainability on synthetic data
    for both classical and quantum routers.

    Returns:
        embeddings: Shape (n_samples, embedding_dim).
        labels: Integer class indices, shape (n_samples,).
    """
    rng = np.random.default_rng(seed)
    per_class = max(n_samples // N_CLASSES, 1)
    embeddings_list = []
    labels_list = []

    for class_idx in range(N_CLASSES):
        # Cluster centre: sparse unit vector scaled so clusters are separable
        centre = np.zeros(embedding_dim)
        # Deterministically place cluster along a rotating combination of dims
        dim_a = (class_idx * 3) % embedding_dim
        dim_b = (class_idx * 7 + 1) % embedding_dim
        centre[dim_a] = 2.0
        centre[dim_b] = 1.5
        samples = rng.standard_normal((per_class, embedding_dim)) * 0.3 + centre
        embeddings_list.append(samples)
        labels_list.extend([class_idx] * per_class)

    embeddings = np.vstack(embeddings_list).astype(np.float64)
    labels = np.array(labels_list, dtype=np.int64)

    # Shuffle
    perm = rng.permutation(len(labels))
    return embeddings[perm], labels[perm]


def train_test_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    perm = rng.permutation(n)
    split = max(1, int(n * (1 - test_ratio)))
    train_idx, test_idx = perm[:split], perm[split:]
    return embeddings[train_idx], labels[train_idx], embeddings[test_idx], labels[test_idx]


# ---------------------------------------------------------------------------
# Classical router (sigmoid linear classifier — mirrors MetaRouter logic)
# ---------------------------------------------------------------------------


class ClassicalRouter:
    """Simple linear softmax classifier — classical baseline."""

    def __init__(self, embedding_dim: int = 64, n_classes: int = N_CLASSES) -> None:
        rng = np.random.default_rng(1)
        self.W = rng.standard_normal((embedding_dim, n_classes)) * 0.01
        self.b = np.zeros(n_classes)
        self.n_classes = n_classes

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def predict(self, embedding: np.ndarray) -> int:
        logits = embedding @ self.W + self.b
        return int(np.argmax(self._softmax(logits)))

    def train(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50,
        lr: float = 0.01,
    ) -> list[float]:
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for emb, label in zip(embeddings, labels):
                logits = emb @ self.W + self.b
                probs = self._softmax(logits)
                epoch_loss += -float(np.log(probs[int(label)] + 1e-12))
                d = probs.copy()
                d[int(label)] -= 1.0
                self.W -= lr * np.outer(emb, d)
                self.b -= lr * d
            losses.append(epoch_loss / max(len(embeddings), 1))
        return losses

    def n_parameters(self) -> int:
        return self.W.size + self.b.size


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_classical(
    router: ClassicalRouter,
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, list[float]]:
    """Returns (accuracy, list_of_latencies_ms)."""
    correct = 0
    latencies = []
    for emb, label in zip(embeddings, labels):
        t0 = time.perf_counter()
        pred = router.predict(emb)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if pred == int(label):
            correct += 1
    accuracy = correct / max(len(labels), 1)
    return accuracy, latencies


def evaluate_quantum(
    router: "QuantumRouter",  # type: ignore[name-defined]
    embeddings: np.ndarray,
    labels: np.ndarray,
    all_domains: list[str],
) -> tuple[float, list[float]]:
    """Returns (accuracy, list_of_latencies_ms)."""
    correct = 0
    latencies = []
    for emb, label in zip(embeddings, labels):
        t0 = time.perf_counter()
        decision = router.route(emb)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        # Decode domain from decision
        if decision.adapter is not None:
            pred_domain = decision.adapter.removeprefix("stack-")
        else:
            pred_domain = "base"
        pred_idx = all_domains.index(pred_domain) if pred_domain in all_domains else -1
        if pred_idx == int(label):
            correct += 1
    accuracy = correct / max(len(labels), 1)
    return accuracy, latencies


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(n_samples: int, epochs: int) -> dict:
    """Execute full benchmark and return results dict."""
    logger.info("Generating %d synthetic samples per %d classes …", n_samples, N_CLASSES)
    embeddings, labels = make_synthetic_data(n_samples=n_samples, embedding_dim=64)
    X_train, y_train, X_test, y_test = train_test_split(embeddings, labels)
    logger.info(
        "Train=%d  Test=%d  epochs=%d", len(y_train), len(y_test), epochs
    )

    results: dict = {
        "n_samples": n_samples,
        "n_classes": N_CLASSES,
        "epochs": epochs,
        "embedding_dim": 64,
        "domains": _ALL_DOMAINS,
    }

    # ------------------------------------------------------------------
    # Classical router
    # ------------------------------------------------------------------
    logger.info("Training classical router …")
    classical = ClassicalRouter(embedding_dim=64)
    classical_losses = classical.train(X_train, y_train, epochs=epochs, lr=0.05)

    classical_acc, classical_lats = evaluate_classical(classical, X_test, y_test)
    classical_metrics = RouterMetrics(
        name="classical-linear",
        accuracy=round(classical_acc, 4),
        latency_ms_mean=round(float(np.mean(classical_lats)), 4),
        latency_ms_p95=round(float(np.percentile(classical_lats, 95)), 4),
        n_parameters=classical.n_parameters(),
        train_loss_first=round(classical_losses[0], 6),
        train_loss_last=round(classical_losses[-1], 6),
    )
    logger.info(
        "Classical → acc=%.3f  latency=%.3fms  params=%d",
        classical_acc,
        classical_metrics.latency_ms_mean,
        classical_metrics.n_parameters,
    )

    results["classical"] = asdict(classical_metrics)

    # ------------------------------------------------------------------
    # Quantum VQC router (requires PennyLane)
    # ------------------------------------------------------------------
    try:
        from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig

        logger.info("Training quantum VQC router …")
        qcfg = QuantumRouterConfig(n_qubits=4, n_layers=3, learning_rate=0.05)
        qr = QuantumRouter(qcfg)

        # Parameter count: VQC weights + linear head
        q_params = qr.weights.size + qr.linear_w.size + qr.linear_b.size

        quantum_losses = qr.train(X_train, y_train, epochs=epochs)
        quantum_acc, quantum_lats = evaluate_quantum(qr, X_test, y_test, _ALL_DOMAINS)

        quantum_metrics = RouterMetrics(
            name="quantum-vqc",
            accuracy=round(quantum_acc, 4),
            latency_ms_mean=round(float(np.mean(quantum_lats)), 4),
            latency_ms_p95=round(float(np.percentile(quantum_lats, 95)), 4),
            n_parameters=q_params,
            train_loss_first=round(quantum_losses[0], 6),
            train_loss_last=round(quantum_losses[-1], 6),
        )
        logger.info(
            "Quantum VQC → acc=%.3f  latency=%.3fms  params=%d",
            quantum_acc,
            quantum_metrics.latency_ms_mean,
            q_params,
        )
        results["quantum"] = asdict(quantum_metrics)

        # Comparison summary
        results["comparison"] = {
            "accuracy_delta": round(quantum_acc - classical_acc, 4),
            "latency_ratio": round(
                quantum_metrics.latency_ms_mean / max(classical_metrics.latency_ms_mean, 1e-9), 2
            ),
            "param_ratio": round(q_params / max(classical.n_parameters(), 1), 4),
            "quantum_faster": quantum_metrics.latency_ms_mean < classical_metrics.latency_ms_mean,
        }

    except ImportError as exc:
        logger.warning("PennyLane not available — quantum benchmark skipped: %s", exc)
        results["quantum"] = {"error": "PennyLane not installed", "skipped": True}
        results["comparison"] = None

    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark quantum VQC router vs classical sigmoid router."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Total synthetic samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for both routers (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_RESULTS_DIR / "quantum-router-benchmark.json"),
        help="Output JSON path (default: results/quantum-router-benchmark.json)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    results = run_benchmark(n_samples=args.n_samples, epochs=args.epochs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", output_path)

    # Print summary to stdout
    print("\n=== Quantum Router Benchmark ===")
    print(f"  n_samples : {results['n_samples']}")
    print(f"  epochs    : {results['epochs']}")
    if "classical" in results:
        c = results["classical"]
        print(f"\n  Classical  acc={c['accuracy']:.3f}  "
              f"lat={c['latency_ms_mean']:.3f}ms  params={c['n_parameters']}")
    if "quantum" in results and not results["quantum"].get("skipped"):
        q = results["quantum"]
        print(f"  Quantum    acc={q['accuracy']:.3f}  "
              f"lat={q['latency_ms_mean']:.3f}ms  params={q['n_parameters']}")
    if results.get("comparison"):
        cmp = results["comparison"]
        print(f"\n  accuracy Δ : {cmp['accuracy_delta']:+.4f}")
        print(f"  latency ×  : {cmp['latency_ratio']:.1f}x  "
              f"({'quantum faster' if cmp['quantum_faster'] else 'classical faster'})")
        print(f"  params  ×  : {cmp['param_ratio']:.4f}x")
    print()


if __name__ == "__main__":
    main()
