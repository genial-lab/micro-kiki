"""L1 loss on masked token positions — V-JEPA 2 recipe."""
from __future__ import annotations

import torch


def masked_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """L1 loss averaged over masked positions only.

    Args:
        pred: (batch, seq_len, dim).
        target: (batch, seq_len, dim) — must be detached (teacher stop-gradient).
        mask: (batch, seq_len) bool, True where position is masked.

    Returns:
        Scalar loss.
    """
    if mask.sum() == 0:
        raise ValueError("mask has zero masked positions")

    # Apply mask along seq dim, broadcast over feature dim
    diff = (pred - target).abs()  # (batch, seq, dim)
    mask_f = mask.unsqueeze(-1).to(diff.dtype)
    masked_sum = (diff * mask_f).sum()
    n_masked_elems = mask_f.sum() * diff.size(-1)
    return masked_sum / n_masked_elems
