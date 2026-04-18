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


def test_teacher_initialised_from_student():
    from src.routing.text_jepa.encoder import StudentEncoder, TeacherEncoder

    student = StudentEncoder(input_dim=384, hidden_dim=256, output_dim=128)
    teacher = TeacherEncoder(student)

    for ps, pt in zip(student.parameters(), teacher.parameters()):
        torch.testing.assert_close(ps.data, pt.data)


def test_teacher_params_frozen():
    from src.routing.text_jepa.encoder import StudentEncoder, TeacherEncoder

    student = StudentEncoder()
    teacher = TeacherEncoder(student)
    for p in teacher.parameters():
        assert p.requires_grad is False


def test_teacher_ema_update_mixes_params():
    from src.routing.text_jepa.encoder import StudentEncoder, TeacherEncoder

    student = StudentEncoder()
    teacher = TeacherEncoder(student)

    # Perturb student
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.ones_like(p))

    teacher.update(student, momentum=0.9)

    # Teacher moved 10% of the way toward student
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        diff = (ps.data - pt.data).abs().mean().item()
        # student moved by ~1, teacher by ~0.1 (with m=0.9)
        assert 0.05 < diff < 0.95


def test_teacher_forward_stop_gradient():
    from src.routing.text_jepa.encoder import StudentEncoder, TeacherEncoder

    student = StudentEncoder()
    teacher = TeacherEncoder(student)
    x = torch.randn(2, 8, 384, requires_grad=True)

    out = teacher(x)
    assert out.requires_grad is False
