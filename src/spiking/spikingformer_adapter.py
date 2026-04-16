"""Spikingformer adapter — wraps Spikingformer (AAAI 2026) as an
alternative SNN conversion backend to LAS.

Story-30 implementation. Uses graceful import fallback for
``spikingjelly`` following the DiffAttn pattern used elsewhere in the
codebase: the adapter is always importable, but ``convert()`` raises
``RuntimeError`` if spikingjelly is not installed (unless a mock
backend is provided).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful spikingjelly import
# ---------------------------------------------------------------------------

try:
    import spikingjelly  # type: ignore[import-untyped]
    from spikingjelly.activation_based import neuron as sj_neuron  # type: ignore[import-untyped]
    from spikingjelly.activation_based import layer as sj_layer  # type: ignore[import-untyped]

    _HAS_SPIKINGJELLY = True
except ImportError:
    spikingjelly = None  # type: ignore[assignment]
    sj_neuron = None  # type: ignore[assignment]
    sj_layer = None  # type: ignore[assignment]
    _HAS_SPIKINGJELLY = False

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


__all__ = [
    "SpikingformerAdapter",
    "SpikingformerConfig",
    "has_spikingjelly",
]


def has_spikingjelly() -> bool:
    """Return True if spikingjelly is importable."""
    return _HAS_SPIKINGJELLY


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SpikingformerConfig:
    """Configuration for Spikingformer conversion.

    Parameters
    ----------
    timesteps : int
        Number of spiking timesteps T (default 4, paper recommendation).
    spike_mode : str
        Spike encoding mode: ``"rate"`` (default) or ``"temporal"``.
    threshold : float
        LIF neuron firing threshold (default 1.0).
    backend : str
        SpikingJelly backend: ``"torch"`` or ``"cupy"`` (default torch).
    """

    timesteps: int = 4
    spike_mode: str = "rate"
    threshold: float = 1.0
    backend: str = "torch"


# ---------------------------------------------------------------------------
# Protocol for pluggable conversion backend (used in tests with mocks)
# ---------------------------------------------------------------------------


@runtime_checkable
class ConversionBackend(Protocol):
    """Protocol for a pluggable SNN conversion backend."""

    def convert_linear(self, weight: np.ndarray, bias: np.ndarray | None) -> Any:
        """Convert a linear layer to its spiking equivalent."""
        ...

    def convert_attention(
        self,
        q_weight: np.ndarray,
        k_weight: np.ndarray,
        v_weight: np.ndarray,
    ) -> Any:
        """Convert an attention block to spiking spike-driven attention."""
        ...

    def forward(self, model: Any, x: np.ndarray) -> np.ndarray:
        """Run spiking forward pass."""
        ...


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class SpikingformerAdapter:
    """Adapter wrapping Spikingformer as alternative to LAS.

    Usage::

        adapter = SpikingformerAdapter(config=SpikingformerConfig(timesteps=4))
        snn_model = adapter.convert(ann_model)
        output = adapter.forward(snn_model, input_tensor)

    If spikingjelly is not installed, ``convert()`` raises
    ``RuntimeError`` unless a mock ``backend`` is injected.
    """

    config: SpikingformerConfig = field(default_factory=SpikingformerConfig)
    backend: ConversionBackend | None = None

    def convert(self, model: Any) -> Any:
        """Convert an ANN model to Spikingformer SNN.

        Parameters
        ----------
        model:
            A torch ``nn.Module`` or a list of weight dicts (for testing).

        Returns
        -------
        snn_model:
            The converted spiking model. Type depends on the backend.

        Raises
        ------
        RuntimeError
            If spikingjelly is not installed and no mock backend provided.
        """
        if self.backend is not None:
            return self._convert_with_backend(model)

        if not _HAS_SPIKINGJELLY:
            raise RuntimeError(
                "spikingjelly is not installed. Install with: "
                "pip install spikingjelly>=0.0.0.0.14 "
                "or inject a mock backend for testing."
            )

        return self._convert_spikingjelly(model)

    def forward(self, snn_model: Any, x: Any) -> Any:
        """Run forward pass on a converted model.

        Parameters
        ----------
        snn_model:
            Model returned by ``convert()``.
        x:
            Input tensor (torch or numpy).

        Returns
        -------
        Output tensor.
        """
        if self.backend is not None:
            return self.backend.forward(snn_model, x)

        if not _HAS_SPIKINGJELLY or not _HAS_TORCH:
            raise RuntimeError(
                "spikingjelly/torch not available. Use a mock backend."
            )

        return self._forward_spikingjelly(snn_model, x)

    # ---- internal: spikingjelly path -------------------------------------

    def _convert_spikingjelly(self, model: Any) -> Any:
        """Real Spikingformer conversion via spikingjelly."""
        assert _HAS_SPIKINGJELLY and _HAS_TORCH
        logger.info(
            "Converting model via Spikingformer (T=%d, mode=%s)",
            self.config.timesteps,
            self.config.spike_mode,
        )
        # Placeholder for real spikingjelly conversion pipeline.
        # In production this would walk the model graph and replace
        # nn.Linear with sj_layer.Linear + sj_neuron.LIFNode.
        raise NotImplementedError(
            "Full Spikingformer conversion requires model-specific "
            "graph walking. Use convert_moe_layer/convert_mistral_block "
            "from LASConverter for framework-level tests."
        )

    def _forward_spikingjelly(self, snn_model: Any, x: Any) -> Any:
        """Real spikingjelly forward with multi-step simulation."""
        assert _HAS_TORCH
        # Multi-step forward: repeat input T times and accumulate spikes
        raise NotImplementedError("Requires converted spikingjelly model.")

    # ---- internal: pluggable backend path --------------------------------

    def _convert_with_backend(self, model: Any) -> Any:
        """Convert using the injected backend."""
        assert self.backend is not None
        logger.info("Converting model via injected backend")

        if isinstance(model, (list, tuple)):
            converted_layers = []
            for layer in model:
                if isinstance(layer, dict):
                    w = layer["weight"]
                    b = layer.get("bias")
                    converted_layers.append(
                        self.backend.convert_linear(w, b)
                    )
            return converted_layers

        # For torch-like models, extract linears
        children = getattr(model, "children", None)
        if callable(children):
            converted_layers = []
            for child in children():
                w = getattr(child, "weight", None)
                if w is not None and getattr(w, "ndim", 0) == 2:
                    w_np = np.asarray(w.detach().cpu().numpy()) if hasattr(w, "detach") else np.asarray(w)
                    b_attr = getattr(child, "bias", None)
                    b_np = np.asarray(b_attr.detach().cpu().numpy()) if b_attr is not None and hasattr(b_attr, "detach") else None
                    converted_layers.append(
                        self.backend.convert_linear(w_np, b_np)
                    )
            return converted_layers

        raise TypeError(f"unsupported model type: {type(model).__name__}")

    # ---- introspection ---------------------------------------------------

    @property
    def available(self) -> bool:
        """True if a real or mock backend is usable."""
        return self.backend is not None or _HAS_SPIKINGJELLY

    def info(self) -> dict[str, Any]:
        """Return adapter status dict."""
        return {
            "spikingjelly_installed": _HAS_SPIKINGJELLY,
            "torch_installed": _HAS_TORCH,
            "backend_injected": self.backend is not None,
            "config": {
                "timesteps": self.config.timesteps,
                "spike_mode": self.config.spike_mode,
                "threshold": self.config.threshold,
                "backend": self.config.backend,
            },
        }
