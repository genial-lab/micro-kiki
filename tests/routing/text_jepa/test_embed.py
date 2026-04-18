"""Tests for TextJEPAEmbedder — inference-time embedding helper."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_embed_shape_mean_pooled():
    """Simulated backbone: return a shape matching (seq_len, 384)."""
    from src.routing.text_jepa.embed import TextJEPAEmbedder
    from src.routing.text_jepa.encoder import StudentEncoder

    student = StudentEncoder(input_dim=16, hidden_dim=32, output_dim=8)

    def fake_token_embed(text: str) -> torch.Tensor:
        # deterministic 5-token "encoding"
        return torch.randn(5, 16)

    embedder = TextJEPAEmbedder(
        student=student, token_embed_fn=fake_token_embed, latent_dim=8
    )
    vec = embedder.embed("hello world")
    assert vec.shape == (8,)


def test_embed_returns_numpy():
    import numpy as np

    from src.routing.text_jepa.embed import TextJEPAEmbedder
    from src.routing.text_jepa.encoder import StudentEncoder

    student = StudentEncoder(input_dim=16, hidden_dim=32, output_dim=8)
    embedder = TextJEPAEmbedder(
        student=student,
        token_embed_fn=lambda text: torch.randn(3, 16),
        latent_dim=8,
    )
    vec = embedder.embed("x")
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float64 or vec.dtype == np.float32


def test_embed_eval_mode_no_grad():
    """Calling embed must not accumulate gradients on student."""
    from src.routing.text_jepa.embed import TextJEPAEmbedder
    from src.routing.text_jepa.encoder import StudentEncoder

    student = StudentEncoder(input_dim=16, hidden_dim=32, output_dim=8)
    embedder = TextJEPAEmbedder(
        student=student,
        token_embed_fn=lambda text: torch.randn(3, 16),
        latent_dim=8,
    )
    embedder.embed("x")
    for p in student.parameters():
        assert p.grad is None
