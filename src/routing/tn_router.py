"""Tensor-network router: MPS-based gating via interference amplitudes."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TNRouterConfig:
    input_dim: int = 3072
    num_domains: int = 35
    bond_dim: int = 16
    num_capabilities: int = 5


def estimate_tn_router_params(config: TNRouterConfig) -> int:
    """Estimate parameter count for TN router."""
    # MPS chain: input_dim -> bond_dim chain -> output
    total_outputs = config.num_domains + config.num_capabilities
    return config.input_dim * config.bond_dim + config.bond_dim * total_outputs
