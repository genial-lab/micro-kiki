"""Tests for LAS MoE-aware conversion (story-21).

Acceptance criteria:
- 4-expert micro-MoE (128-d, top-2)
- Expert selection agreement >= 99%
- Output MSE <= 1e-3
"""

from __future__ import annotations

import numpy as np
import pytest

from src.spiking.las_converter import LASConverter, SpikingMoELayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_moe_weights(
    rng: np.random.Generator,
    dim: int = 128,
    num_experts: int = 4,
    out_dim: int = 64,
    scale: float = 0.1,
) -> tuple[dict, list[dict]]:
    """Build router + expert weight dicts for a micro-MoE layer."""
    router = {
        "weight": rng.standard_normal((num_experts, dim)).astype(np.float64) * scale,
        "bias": np.zeros(num_experts, dtype=np.float64),
    }
    experts = []
    for _ in range(num_experts):
        experts.append(
            {
                "weight": rng.standard_normal((out_dim, dim)).astype(np.float64) * scale,
                "bias": np.zeros(out_dim, dtype=np.float64),
            }
        )
    return router, experts


def _ann_moe_forward(
    x: np.ndarray,
    router_w: np.ndarray,
    router_b: np.ndarray,
    expert_ws: list[np.ndarray],
    expert_bs: list[np.ndarray],
    top_k: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """ANN MoE forward: returns (output, expert_indices)."""
    squeeze = x.ndim == 1
    if squeeze:
        x = x[np.newaxis, :]

    logits = x @ router_w.T + router_b
    batch_size = x.shape[0]
    out_dim = expert_ws[0].shape[0]
    output = np.zeros((batch_size, out_dim), dtype=np.float64)
    indices = np.zeros((batch_size, top_k), dtype=np.int64)

    for b in range(batch_size):
        row_logits = logits[b]
        top_idx = np.argsort(row_logits)[-top_k:][::-1]
        indices[b] = top_idx
        sel_logits = row_logits[top_idx] - row_logits[top_idx].max()
        exp_l = np.exp(sel_logits)
        weights = exp_l / (exp_l.sum() + 1e-12)
        for i, eidx in enumerate(top_idx):
            expert_out = np.maximum(x[b] @ expert_ws[eidx].T + expert_bs[eidx], 0.0)
            output[b] += weights[i] * expert_out

    if squeeze:
        return output[0], indices[0]
    return output, indices


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLASMoE:
    """Story-21 acceptance tests for MoE-aware LAS conversion."""

    @pytest.fixture()
    def moe_setup(self):
        rng = np.random.default_rng(42)
        dim = 128
        num_experts = 4
        out_dim = 64
        top_k = 2

        router, experts = _build_moe_weights(
            rng, dim=dim, num_experts=num_experts, out_dim=out_dim
        )
        converter = LASConverter(timesteps=256, max_rate=1.0)
        snn_moe = converter.convert_moe_layer(
            router=router, experts=experts, top_k=top_k
        )
        x = rng.standard_normal((32, dim)).astype(np.float64) * 0.3
        return {
            "snn_moe": snn_moe,
            "router": router,
            "experts": experts,
            "x": x,
            "top_k": top_k,
        }

    def test_expert_selection_agreement(self, moe_setup: dict) -> None:
        """Expert selection agreement must be >= 99%."""
        s = moe_setup
        snn_indices = s["snn_moe"].selected_experts(s["x"])
        _, ann_indices = _ann_moe_forward(
            s["x"],
            s["router"]["weight"],
            s["router"]["bias"],
            [e["weight"] for e in s["experts"]],
            [e["bias"] for e in s["experts"]],
            top_k=s["top_k"],
        )
        # Compare sets of selected experts per sample
        agreement = 0
        total = snn_indices.shape[0]
        for b in range(total):
            if set(snn_indices[b].tolist()) == set(ann_indices[b].tolist()):
                agreement += 1
        ratio = agreement / total
        assert ratio >= 0.99, f"expert selection agreement {ratio:.3f} < 0.99"

    def test_output_mse(self, moe_setup: dict) -> None:
        """Output MSE between ANN and SNN MoE must be <= 1e-3."""
        s = moe_setup
        snn_out = s["snn_moe"].forward(s["x"])
        ann_out, _ = _ann_moe_forward(
            s["x"],
            s["router"]["weight"],
            s["router"]["bias"],
            [e["weight"] for e in s["experts"]],
            [e["bias"] for e in s["experts"]],
            top_k=s["top_k"],
        )
        mse = float(np.mean((snn_out - ann_out) ** 2))
        assert mse <= 1e-3, f"output MSE {mse:.6f} > 1e-3"

    def test_moe_shape_and_attributes(self, moe_setup: dict) -> None:
        """SpikingMoELayer has correct attributes."""
        snn_moe: SpikingMoELayer = moe_setup["snn_moe"]
        assert snn_moe.num_experts == 4
        assert snn_moe.top_k == 2
        assert len(snn_moe.experts) == 4
        assert snn_moe.router.in_features == 128
        assert snn_moe.router.out_features == 4

    def test_single_sample_forward(self, moe_setup: dict) -> None:
        """MoE forward works on a single (non-batched) sample."""
        s = moe_setup
        single = s["x"][0]
        out = s["snn_moe"].forward(single)
        assert out.shape == (64,), f"expected (64,), got {out.shape}"
