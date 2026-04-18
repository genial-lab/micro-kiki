"""Integration test: sleep_cycle step 5 trains the predictor."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.memory.aeon_predictor import (
    AeonPredictor,
    PredictorConfig,
    detect_collapse,
)
from src.memory.aeonsleep import AeonSleep


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-8)


@pytest.fixture
def palace_with_pairs():
    from src.memory.aeonsleep import Episode

    palace = AeonSleep(dim=16, keep_threshold=1.0)
    cfg = PredictorConfig(
        dim=16, hidden=16, n_stacks=4, cold_start_threshold=4
    )
    pred = AeonPredictor(palace=palace, config=cfg)
    t0 = datetime(2026, 4, 17, 10, 0)
    rng = np.random.default_rng(0)
    # Ingest latent pairs into predictor.
    for i in range(12):
        h = _unit(rng.standard_normal(16).astype(np.float32))
        pred.ingest_latent(
            f"t{i}", h, ts=t0 + timedelta(minutes=i), stack_id=i % 4
        )
    # Write episodes to palace so sleep_cycle can access them.
    for i in range(12):
        emb = _unit(rng.standard_normal(16).astype(np.float32))
        palace.write(
            Episode(
                id=f"ep_{i}",
                text=f"episode {i}",
                embedding=emb,
                ts=t0 + timedelta(minutes=i),
                topic="test",
            )
        )
    palace.attach_predictor(pred)
    return palace, pred


def test_sleep_cycle_runs_predictor_training(palace_with_pairs):
    palace, pred = palace_with_pairs
    assert pred._trained_once is False
    report = palace.sleep_cycle()
    assert pred._trained_once is True
    # The report carries predictor stats.
    assert report.predictor_epochs == 1
    assert report.predictor_loss is not None
    assert 0.0 <= report.predictor_loss <= 2.0


def test_sleep_cycle_rollback_on_collapse(palace_with_pairs):
    palace, pred = palace_with_pairs
    # Trigger collapse by mocking the detect_collapse result.
    # Patch detect_collapse to return True, simulating a collapse.
    from unittest.mock import patch

    pre_w1 = pred.mlp.w1.copy()
    pre_b1 = pred.mlp.b1.copy()

    with patch(
        "src.memory.aeon_predictor.detect_collapse", return_value=(True, 0.05)
    ):
        report = palace.sleep_cycle()

    # Collapse detector should have tripped and weights rolled back.
    assert report.predictor_collapsed is True
    # After rollback, w1 and b1 should match the pre-fit snapshot.
    np.testing.assert_allclose(pred.mlp.w1, pre_w1, atol=1e-6)
    np.testing.assert_allclose(pred.mlp.b1, pre_b1, atol=1e-6)


def test_sleep_cycle_no_predictor_attached():
    palace = AeonSleep(dim=16)
    report = palace.sleep_cycle()
    assert report.predictor_epochs == 0
    assert report.predictor_loss is None
    assert report.predictor_collapsed is False
