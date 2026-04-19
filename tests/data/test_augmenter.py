"""Tests for src/data/augmenter.py — teacher-based synthetic completion."""
from __future__ import annotations

from unittest.mock import MagicMock


def test_augment_domain_uses_seeds_from_existing_samples():
    from src.data.augmenter import augment_domain_via_teacher

    seeds = [
        "How do I configure a Schmitt trigger input?",
        "What is the difference between BJT and MOSFET biasing?",
    ]
    teacher = MagicMock(return_value="A synthetic electronics question about ...")

    augmented = augment_domain_via_teacher(
        domain="electronics",
        seeds=seeds,
        n_to_generate=5,
        teacher_fn=teacher,
    )
    assert len(augmented) == 5
    assert all(isinstance(x, str) for x in augmented)
    assert teacher.call_count == 5
    for call in teacher.call_args_list:
        prompt = call.args[0] if call.args else call.kwargs.get("prompt", "")
        has_seed = any(s in prompt for s in seeds)
        assert has_seed, f"teacher prompt missing seed: {prompt!r}"


def test_augment_domain_zero_n_returns_empty():
    from src.data.augmenter import augment_domain_via_teacher

    out = augment_domain_via_teacher(
        domain="x", seeds=["s"], n_to_generate=0, teacher_fn=MagicMock()
    )
    assert out == []
