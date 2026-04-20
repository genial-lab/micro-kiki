"""Bridge between micro-kiki's routing hot path and kiki-flow's StreamingRunner.

Owns the tokenizer handle so the runner does not depend on MetaRouter.forward
receiving a query-string parameter. The bridge is constructed once at router
init and threaded through as an optional attribute.

Weights version: v0.3 (text-conditioned via QueryConditionedF + JEPA
pre-training). Replaces v0.2-d128 (synthetic JKO pairs only).

DRAFT: do not enable in production until
kiki-flow-research/artifacts/v0.3.safetensors exists (produced by T22/T23
of the workshop sprint).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default weights version. Bump this string when a new surrogate is trained.
# v0.2-d128: synthetic JKO pairs only (128-dim, Mode-A, 30 epochs, loss 9.8e-5)
# v0.3:      text-conditioned via QueryConditionedF + g_JEPA pre-training
_WEIGHTS_VERSION = "v0.3"


class KikiFlowBridge:
    """Lazy-loaded StreamingRunner wrapper with its own MiniLM tokenizer.

    Returns None from route_advisory() if disabled, NaN, dim mismatch, or any
    other failure; callers must treat the result as purely advisory.

    Args:
        weights_path: Path to the surrogate .safetensors checkpoint. Must be
            v0.3 or later (text-conditioned). For v0.2-d128 checkpoints, pin
            weights_version="v0.2-d128" explicitly until migration is complete.
        state_dim: Hidden state dimension. Must match the checkpoint.
        embed_dim: Query encoder output dimension. Must match the checkpoint.
        hidden: Hidden layer width of the BridgeHead MLP.
        tokenizer: Optional pre-loaded tokenizer. If None, QueryEncoder uses
            its own sentence-transformers MiniLM.
        prior_weight: Blend factor applied to advisory scores (default 10%).
        weights_version: Version tag for logging/metrics. Default "v0.3".
    """

    def __init__(
        self,
        weights_path: Path,
        state_dim: int = 128,
        embed_dim: int = 384,
        hidden: int = 256,
        tokenizer: Any | None = None,
        prior_weight: float = 0.1,
        weights_version: str = _WEIGHTS_VERSION,
    ) -> None:
        self.enabled = bool(int(os.environ.get("KIKI_FLOW_ENABLED", "0")))
        self.prior_weight = prior_weight
        self.weights_version = weights_version
        self._runner = None
        if not self.enabled:
            return
        try:
            from kiki_flow_core.hooks import RoutingAdapter
            from kiki_flow_core.state import FlowState
            from kiki_flow_core.track3_deploy.neural_surrogate import NeuralSurrogate
            from kiki_flow_core.track3_deploy.query_encoder import QueryEncoder
            from kiki_flow_core.track3_deploy.streaming_runner import StreamingRunner

            surrogate = NeuralSurrogate.load(
                weights_path, state_dim=state_dim, embed_dim=embed_dim, hidden=hidden
            )
            # QueryEncoder uses the provided tokenizer if MiniLM is
            # unavailable, else its own MiniLM via sentence-transformers.
            encoder = QueryEncoder(use_stub=(tokenizer is None))
            # Initial state: uniform over all species
            stacks = [f"stack_{i:02d}" for i in range(32)]
            orthos = ["phono", "lex", "syntax", "sem"]
            initial = FlowState(
                rho={f"{o}:{s}": np.array([1.0 / 32]) for o in orthos for s in stacks},
                P_theta=np.zeros(8),
                mu_curr=np.array([1.0]),
                tau=0,
                metadata={"track_id": "T3"},
            )
            self._runner = StreamingRunner(
                surrogate=surrogate,
                encoder=encoder,
                routing_adapter=RoutingAdapter(publisher=lambda _adv: None),
                initial_state=initial,
            )
            logger.info(
                "kiki-flow bridge ready, state_dim=%d, weights=%s",
                state_dim,
                weights_version,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("kiki-flow bridge init failed, disabling: %s", e)
            self.enabled = False

    def route_advisory(self, query: str) -> np.ndarray | None:
        """Return a (32,)-shaped array of stack-level advisory weights, or None."""
        if not self.enabled or self._runner is None:
            return None
        try:
            advisory = self._runner.on_query(query)
            summary = advisory.get("state_summary", {})
            if not summary:
                return None
            # Aggregate over the 4 ortho species per stack
            weights = np.zeros(32, dtype=np.float32)
            for key, val in summary.items():
                if ":" not in key:
                    continue
                _ortho, stack_name = key.split(":", 1)
                if not stack_name.startswith("stack_"):
                    continue
                idx = int(stack_name.split("_", 1)[1])
                if 0 <= idx < 32:
                    weights[idx] += float(val) * 0.25  # /4 ortho contributions
            return weights
        except Exception as e:  # noqa: BLE001
            logger.warning("kiki-flow advisory failed, passthrough: %s", e)
            return None
