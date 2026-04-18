"""Tests for masked L1 loss — compare with known inputs."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_loss_ignores_unmasked_positions():
    from src.routing.text_jepa.loss import masked_l1_loss

    pred = torch.zeros(1, 4, 2)
    target = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]])
    mask = torch.tensor([[False, True, False, False]])  # only pos 1

    loss = masked_l1_loss(pred, target, mask)
    # mean(|0-2|, |0-2|) over masked = 2.0
    assert loss.item() == pytest.approx(2.0)


def test_loss_zero_when_pred_equals_target():
    from src.routing.text_jepa.loss import masked_l1_loss

    pred = torch.randn(2, 8, 16)
    target = pred.clone()
    mask = torch.ones(2, 8, dtype=torch.bool)

    loss = masked_l1_loss(pred, target, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_raises_when_no_masked_positions():
    from src.routing.text_jepa.loss import masked_l1_loss

    pred = torch.zeros(1, 4, 2)
    target = torch.zeros(1, 4, 2)
    mask = torch.zeros(1, 4, dtype=torch.bool)

    with pytest.raises(ValueError):
        masked_l1_loss(pred, target, mask)


def test_loss_differentiable_wrt_pred():
    from src.routing.text_jepa.loss import masked_l1_loss

    pred = torch.randn(1, 4, 2, requires_grad=True)
    target = torch.randn(1, 4, 2)
    mask = torch.tensor([[True, False, True, False]])

    loss = masked_l1_loss(pred, target, mask)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
