"""Trainer tests — single step on synthetic data should reduce loss."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_trainer_single_step_runs():
    from src.routing.text_jepa.trainer import TextJEPATrainer

    trainer = TextJEPATrainer(
        input_dim=384,
        latent_dim=128,
        hidden_dim=256,
        predictor_hidden=32,
        lr=1e-3,
        ema_momentum=0.99,
        mask_ratio=0.4,
        min_span=3,
        max_span=5,
        seed=0,
    )
    # Batch of 2 sequences, 16 tokens, 384-d
    tokens = torch.randn(2, 16, 384)
    loss = trainer.step(tokens)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_trainer_reduces_loss_on_fixed_batch():
    """Overfitting on a single batch must reduce loss monotonically-ish."""
    from src.routing.text_jepa.trainer import TextJEPATrainer

    trainer = TextJEPATrainer(
        input_dim=64,
        latent_dim=32,
        hidden_dim=64,
        predictor_hidden=16,
        lr=1e-2,
        ema_momentum=0.99,
        mask_ratio=0.4,
        min_span=3,
        max_span=5,
        seed=0,
    )
    torch.manual_seed(0)
    tokens = torch.randn(2, 16, 64)
    first = trainer.step(tokens).item()
    for _ in range(50):
        trainer.step(tokens)
    last = trainer.step(tokens).item()
    assert last < first, f"loss did not decrease: first={first} last={last}"


def test_trainer_ema_updates_teacher():
    from src.routing.text_jepa.trainer import TextJEPATrainer

    trainer = TextJEPATrainer(
        input_dim=64, latent_dim=32, hidden_dim=64, predictor_hidden=16,
        lr=1e-3, ema_momentum=0.9, mask_ratio=0.4, min_span=3, max_span=5, seed=0,
    )
    before = [p.detach().clone() for p in trainer.teacher.parameters()]
    tokens = torch.randn(2, 16, 64)
    for _ in range(5):
        trainer.step(tokens)
    after = list(trainer.teacher.parameters())
    moved = any(
        not torch.allclose(b, a.data) for b, a in zip(before, after)
    )
    assert moved, "teacher did not update via EMA"


def test_trainer_detects_collapse():
    """If student output std floors, collapse flag should raise."""
    from src.routing.text_jepa.trainer import TextJEPATrainer

    trainer = TextJEPATrainer(
        input_dim=64, latent_dim=32, hidden_dim=64, predictor_hidden=16,
        lr=1e-3, ema_momentum=0.99, mask_ratio=0.4, min_span=3, max_span=5, seed=0,
        collapse_floor=10.0,  # absurd floor to force collapse
        collapse_patience=1,
    )
    tokens = torch.randn(2, 16, 64)
    trainer.step(tokens)
    assert trainer.collapsed is True
