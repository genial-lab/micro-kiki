"""Tests for StudentEncoder and TeacherEncoder."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_student_forward_shape():
    from src.routing.text_jepa.encoder import StudentEncoder

    model = StudentEncoder(input_dim=384, hidden_dim=256, output_dim=128)
    x = torch.randn(4, 16, 384)  # (batch, seq_len, input_dim)
    out = model(x)
    assert out.shape == (4, 16, 128)


def test_student_has_trainable_params():
    from src.routing.text_jepa.encoder import StudentEncoder

    model = StudentEncoder(input_dim=384, hidden_dim=256, output_dim=128)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0


def test_student_output_is_finite():
    from src.routing.text_jepa.encoder import StudentEncoder

    model = StudentEncoder(input_dim=384, hidden_dim=256, output_dim=128)
    x = torch.randn(2, 8, 384)
    out = model(x)
    assert torch.isfinite(out).all()
