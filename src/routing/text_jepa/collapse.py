"""Representation-collapse watchdog — abort training if student std floors."""
from __future__ import annotations

import torch


def embedding_std(x: torch.Tensor) -> float:
    """Mean over feature dims of token-wise std. Collapse detector signal."""
    # flatten batch+seq, keep features
    flat = x.reshape(-1, x.size(-1))
    return float(flat.std(dim=0).mean().item())


class CollapseMonitor:
    """Flag collapse when std stays below `floor` for `patience` consecutive checks."""

    def __init__(self, floor: float = 0.01, patience: int = 2) -> None:
        self.floor = floor
        self.patience = patience
        self._strikes = 0

    def check(self, std: float) -> bool:
        """Returns True if collapse detected (caller should abort training)."""
        if std < self.floor:
            self._strikes += 1
        else:
            self._strikes = 0
        return self._strikes >= self.patience
