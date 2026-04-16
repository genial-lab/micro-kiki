"""Spikingformer — spike-driven self-attention ANN-to-SNN conversion.

Story-30 implementation. Simplified Spikingformer (AAAI 2026) that
replaces softmax attention with spike-gated similarity using LIF
neurons. This is a *training-free* conversion alternative to LAS.

Key difference from LAS:
- LAS does lossless activation scaling (rate-coded LIF per layer)
- Spikingformer replaces softmax with binary spike gates in attention

Public surface:

- :func:`spike_attention` — core spike-driven attention primitive
- :class:`SpikingTransformerLayer` — single transformer layer with
  spike attention + spiking MLP
- :class:`SpikingformerConverter` — converts a toy ANN transformer
  to its spiking equivalent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.spiking.lif_neuron import LIFNeuron

__all__ = [
    "spike_attention",
    "SpikingTransformerLayer",
    "SpikingformerConverter",
]


def spike_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    threshold: float = 1.0,
) -> np.ndarray:
    """Spike-driven attention: replace softmax with LIF-gated similarity.

    Parameters
    ----------
    Q : np.ndarray
        Query matrix, shape ``(seq_len, d_k)`` or ``(batch, seq, d_k)``.
    K : np.ndarray
        Key matrix, same shape as Q.
    V : np.ndarray
        Value matrix, shape ``(seq_len, d_v)`` or ``(batch, seq, d_v)``.
    threshold : float
        Firing threshold for the spike gate. Scores above this
        produce a binary 1; below produce 0.

    Returns
    -------
    np.ndarray
        Spike-weighted output, same leading dims as Q, last dim = d_v.
    """
    if Q.ndim == 2:
        scores = Q @ K.T
        spikes = (scores > threshold).astype(np.float64)
        return spikes @ V
    elif Q.ndim == 3:
        # batched: (batch, seq, d_k) @ (batch, d_k, seq) -> (batch, seq, seq)
        scores = np.einsum("bik,bjk->bij", Q, K)
        spikes = (scores > threshold).astype(np.float64)
        return np.einsum("bij,bjd->bid", spikes, V)
    else:
        raise ValueError(f"Q must be 2-D or 3-D, got {Q.ndim}-D")


@dataclass
class SpikingTransformerLayer:
    """Single transformer layer with spike-driven attention + spiking MLP.

    Architecture:
    1. Linear projections Q, K, V
    2. Spike attention (replace softmax with binary spike gate)
    3. Output projection
    4. Residual connection
    5. Spiking MLP (linear + LIF-gated ReLU + linear)
    6. Residual connection

    All projections use ANN-equivalent matmuls (no rate coding).
    The spike gate in attention is the key Spikingformer contribution.
    """

    w_q: np.ndarray  # (d_model, d_k)
    w_k: np.ndarray  # (d_model, d_k)
    w_v: np.ndarray  # (d_model, d_v)
    w_out: np.ndarray  # (d_v, d_model)
    w_mlp1: np.ndarray  # (d_model, d_ff)
    w_mlp2: np.ndarray  # (d_ff, d_model)
    threshold: float = 1.0
    mlp_threshold: float = 0.0  # ReLU-like: pass if > 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through one spiking transformer layer.

        Parameters
        ----------
        x : np.ndarray
            Input of shape ``(seq_len, d_model)`` or
            ``(batch, seq_len, d_model)``.

        Returns
        -------
        np.ndarray
            Output of same shape as x.
        """
        squeeze = x.ndim == 2
        if squeeze:
            x = x[np.newaxis, :]

        # --- Spike attention ---
        Q = np.einsum("...i,ij->...j", x, self.w_q)
        K = np.einsum("...i,ij->...j", x, self.w_k)
        V = np.einsum("...i,ij->...j", x, self.w_v)
        attn_out = spike_attention(Q, K, V, threshold=self.threshold)
        projected = np.einsum("...i,ij->...j", attn_out, self.w_out)
        h = x + projected  # residual

        # --- Spiking MLP ---
        ff = np.einsum("...i,ij->...j", h, self.w_mlp1)
        # Spike gate: binary activation replacing ReLU
        ff_spikes = (ff > self.mlp_threshold).astype(np.float64) * ff
        ff_out = np.einsum("...i,ij->...j", ff_spikes, self.w_mlp2)
        out = h + ff_out  # residual

        if squeeze:
            return out[0]
        return out

    __call__ = forward

    @property
    def param_count(self) -> int:
        """Total number of parameters."""
        total = (
            self.w_q.size + self.w_k.size + self.w_v.size
            + self.w_out.size + self.w_mlp1.size + self.w_mlp2.size
        )
        return int(total)


@dataclass
class SpikingformerConverter:
    """Convert a toy ANN transformer into a spiking equivalent.

    This converter implements the Spikingformer approach: replace
    softmax attention with spike-gated similarity. Linear projections
    are kept as-is (ANN matmuls). The key innovation is in the
    attention mechanism, not in per-layer activation encoding (that
    is what LAS does differently).

    Parameters
    ----------
    threshold : float
        Spike threshold for attention gates.
    mlp_threshold : float
        Threshold for MLP spike gates (0.0 = ReLU-like).
    """

    threshold: float = 1.0
    mlp_threshold: float = 0.0

    def convert_layer(
        self,
        w_q: np.ndarray,
        w_k: np.ndarray,
        w_v: np.ndarray,
        w_out: np.ndarray,
        w_mlp1: np.ndarray,
        w_mlp2: np.ndarray,
    ) -> SpikingTransformerLayer:
        """Convert one transformer layer's weights to a spiking layer."""
        return SpikingTransformerLayer(
            w_q=w_q.copy(),
            w_k=w_k.copy(),
            w_v=w_v.copy(),
            w_out=w_out.copy(),
            w_mlp1=w_mlp1.copy(),
            w_mlp2=w_mlp2.copy(),
            threshold=self.threshold,
            mlp_threshold=self.mlp_threshold,
        )

    def convert_model(
        self,
        layers: list[dict[str, np.ndarray]],
    ) -> list[SpikingTransformerLayer]:
        """Convert a list of layer weight dicts to spiking layers.

        Each dict must have keys: ``w_q``, ``w_k``, ``w_v``, ``w_out``,
        ``w_mlp1``, ``w_mlp2``.
        """
        result = []
        for layer_dict in layers:
            result.append(self.convert_layer(
                w_q=layer_dict["w_q"],
                w_k=layer_dict["w_k"],
                w_v=layer_dict["w_v"],
                w_out=layer_dict["w_out"],
                w_mlp1=layer_dict["w_mlp1"],
                w_mlp2=layer_dict["w_mlp2"],
            ))
        return result

    def forward_model(
        self,
        layers: list[SpikingTransformerLayer],
        x: np.ndarray,
    ) -> np.ndarray:
        """Run forward through a stack of spiking transformer layers."""
        out = x
        for layer in layers:
            out = layer.forward(out)
        return out

    def compute_sparsity(
        self,
        layers: list[SpikingTransformerLayer],
        x: np.ndarray,
    ) -> dict[str, Any]:
        """Measure activation sparsity through the spiking model.

        Returns per-layer spike rates and overall sparsity.
        """
        layer_stats = []
        h = x.copy()
        squeeze = h.ndim == 2
        if squeeze:
            h = h[np.newaxis, :]

        for i, layer in enumerate(layers):
            Q = np.einsum("...i,ij->...j", h, layer.w_q)
            K = np.einsum("...i,ij->...j", h, layer.w_k)
            scores = np.einsum("bik,bjk->bij", Q, K)
            attn_spikes = (scores > layer.threshold).astype(np.float64)
            attn_rate = float(attn_spikes.mean())

            ff = np.einsum("...i,ij->...j", h, layer.w_mlp1)
            mlp_active = (ff > layer.mlp_threshold).astype(np.float64)
            mlp_rate = float(mlp_active.mean())

            layer_stats.append({
                "layer": i,
                "attn_spike_rate": attn_rate,
                "mlp_active_rate": mlp_rate,
                "overall_sparsity": 1.0 - (attn_rate + mlp_rate) / 2.0,
            })

            h = layer.forward(h if not squeeze else h[0])
            if h.ndim == 2:
                h = h[np.newaxis, :]

        avg_sparsity = float(np.mean([s["overall_sparsity"]
                                       for s in layer_stats]))
        return {
            "per_layer": layer_stats,
            "avg_sparsity": avg_sparsity,
        }
