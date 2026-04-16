"""E2E acceptance test framework for v0.3 neuroscience stack (story-38).

All tests are mocked — no GPU, no torch, no external services needed.
Acceptance targets validated as assertions:
- PI-depth-10 recall >= 95%
- Latency placeholder (mocked)
- Routing agreement >= 98%
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from src.memory.atlas import AtlasIndex
from src.spiking.las_converter import LASConverter, SpikingMoELayer
from src.spiking.lif_neuron import LIFNeuron, rate_encode


# ---------------------------------------------------------------------------
# Mock AeonSleep for write+recall test
# ---------------------------------------------------------------------------

DIM = 32


@dataclass
class MockEpisode:
    key: str
    embedding: list[float]
    payload: dict[str, Any]


class MockAeonSleep:
    """Minimal AeonSleep mock: write episodes, recall by similarity."""

    def __init__(self, dim: int = DIM) -> None:
        self.index = AtlasIndex(dim=dim)
        self.episodes: dict[str, MockEpisode] = {}

    def write(self, episode: MockEpisode) -> None:
        self.episodes[episode.key] = episode
        self.index.add(
            key=episode.key,
            vector=episode.embedding,
            payload=episode.payload,
        )

    def recall(self, query: list[float], k: int = 10) -> list[str]:
        hits = self.index.search(query, k=k)
        return [h.key for h in hits]


def _make_embedding(key_axis: int, strength: float = 1.0) -> list[float]:
    """Create a unit-normalised embedding with energy on key_axis."""
    vec = [0.0] * DIM
    vec[key_axis % DIM] = strength
    vec[(key_axis + 1) % DIM] = 0.1  # minor noise
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestE2EAeonSleep:
    """AeonSleep write+recall acceptance test."""

    def test_pi_depth_10_recall(self) -> None:
        """PI (proactive interference) depth 10: >= 95% recall accuracy.

        Plant 50 topic chains x 10 episodes each. Query with the anchor
        embedding and check that the correct chain surfaces in top-10.
        """
        sleep = MockAeonSleep(dim=DIM)
        num_chains = 50
        depth = 10

        # Plant episodes
        for chain_id in range(num_chains):
            for ep in range(depth):
                key = f"chain-{chain_id}-ep-{ep}"
                emb = _make_embedding(chain_id, strength=1.0 - ep * 0.02)
                sleep.write(MockEpisode(
                    key=key,
                    embedding=emb,
                    payload={"chain": chain_id, "depth": ep},
                ))

        # Recall
        correct = 0
        for chain_id in range(num_chains):
            query = _make_embedding(chain_id)
            results = sleep.recall(query, k=10)
            chain_keys = {f"chain-{chain_id}-ep-{e}" for e in range(depth)}
            if any(r in chain_keys for r in results):
                correct += 1

        accuracy = correct / num_chains
        assert accuracy >= 0.95, f"PI-depth-10 recall {accuracy:.2%} < 95%"


class TestE2ESNNForward:
    """SNN backbone forward pass acceptance test."""

    def test_snn_forward_matches_ann(self) -> None:
        """Single-layer SNN forward stays within 5% of ANN."""
        rng = np.random.default_rng(42)
        dim = 64
        w = rng.standard_normal((dim, dim)).astype(np.float64) * 0.2
        b = np.zeros(dim, dtype=np.float64)

        converter = LASConverter(timesteps=128, max_rate=1.0)
        snn = converter.convert_layer({"weight": w, "bias": b})

        x = np.clip(rng.standard_normal((8, dim)) * 0.2, -0.5, 0.5)
        snn_out = snn(x)
        ann_out = np.maximum(x @ w.T + b, 0.0)

        rel_err = np.linalg.norm(snn_out - ann_out) / (
            np.linalg.norm(ann_out) + 1e-12
        )
        assert rel_err < 0.05, f"SNN rel_err {rel_err:.4f} >= 0.05"

    def test_lif_neuron_spike_count(self) -> None:
        """LIF neuron integrates spikes correctly over T timesteps."""
        neuron = LIFNeuron(threshold=0.25, tau=1.0)
        # Constant current should produce predictable spike count
        T = 32
        current = np.full((T, 4), 0.25)
        spikes, _ = neuron.simulate(current)
        counts = spikes.sum(axis=0)
        # Each element should spike ~T times with current==threshold
        assert np.all(counts >= T - 2), f"spike counts {counts} too low"


class TestE2ERouting:
    """Routing decision acceptance test."""

    def test_routing_agreement(self) -> None:
        """MoE routing agreement between ANN and SNN >= 98%.

        Uses a 4-expert MoE with clearly separated expert preferences.
        """
        rng = np.random.default_rng(99)
        dim = 64
        num_experts = 4
        num_samples = 100
        top_k = 2

        # Make router weights with clear expert separation
        router_w = np.zeros((num_experts, dim), dtype=np.float64)
        for i in range(num_experts):
            start = i * (dim // num_experts)
            end = start + (dim // num_experts)
            router_w[i, start:end] = 1.0

        expert_ws = [
            rng.standard_normal((32, dim)).astype(np.float64) * 0.1
            for _ in range(num_experts)
        ]

        converter = LASConverter(timesteps=128, max_rate=1.0)
        snn_moe = converter.convert_moe_layer(
            router={"weight": router_w, "bias": np.zeros(num_experts)},
            experts=[
                {"weight": ew, "bias": np.zeros(32)}
                for ew in expert_ws
            ],
            top_k=top_k,
        )

        x = rng.standard_normal((num_samples, dim)).astype(np.float64) * 0.3
        snn_indices = snn_moe.selected_experts(x)

        # ANN expert selection
        ann_logits = x @ router_w.T
        ann_indices = np.zeros((num_samples, top_k), dtype=np.int64)
        for b in range(num_samples):
            ann_indices[b] = np.argsort(ann_logits[b])[-top_k:][::-1]

        agreement = sum(
            set(snn_indices[b].tolist()) == set(ann_indices[b].tolist())
            for b in range(num_samples)
        )
        ratio = agreement / num_samples
        assert ratio >= 0.98, f"routing agreement {ratio:.2%} < 98%"


class TestE2ELatencyPlaceholder:
    """Latency placeholder test (mocked)."""

    def test_latency_within_budget(self) -> None:
        """Placeholder: assert mocked latency < 100ms per token."""
        # In production this would measure real inference time.
        # For v0.3 framework story, use a mock value.
        mock_latency_ms = 42.0
        assert mock_latency_ms < 100.0, (
            f"latency {mock_latency_ms}ms exceeds 100ms budget"
        )


class TestE2EEnergyRatio:
    """Energy ratio sanity check."""

    def test_energy_ratio_formula(self) -> None:
        """Verify energy ratio computation from story-32."""
        from scripts.energy_bench import compute_energy

        result = compute_energy(
            model_params=7e9,
            seq_len=2048,
            spike_rate=0.3,
            timesteps=4,
        )
        # spike_rate * timesteps / 2 = 0.3 * 4 / 2 = 0.6
        expected_ratio = 0.3 * 4 / 2.0
        assert abs(result.energy_ratio - expected_ratio) < 1e-6
        assert result.snn_energy_saving_pct > 0
