"""Integration: HybridPipelineConfig carries use_text_jepa; build_embedder routes correctly."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_config_accepts_use_text_jepa():
    from src.routing.hybrid_pipeline import HybridPipelineConfig

    cfg = HybridPipelineConfig(use_text_jepa=True, text_jepa_checkpoint="dummy.pt")
    assert cfg.use_text_jepa is True
    assert cfg.text_jepa_checkpoint == "dummy.pt"


def test_config_defaults_use_text_jepa_false():
    from src.routing.hybrid_pipeline import HybridPipelineConfig

    cfg = HybridPipelineConfig()
    assert cfg.use_text_jepa is False


def test_build_text_jepa_embedder_roundtrip(tmp_path):
    """Train a tiny student, save it, load it via build_text_jepa_embedder, embed."""
    from src.routing.hybrid_pipeline import build_text_jepa_embedder
    from src.routing.text_jepa.encoder import StudentEncoder

    student = StudentEncoder(input_dim=32, hidden_dim=16, output_dim=8)
    ckpt_path = tmp_path / "student.pt"
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "predictor_state_dict": {},
            "config": {
                "input_dim": 32,
                "latent_dim": 8,
                "hidden_dim": 16,
                "seq_len": 4,
                "backbone": "random",
            },
        },
        ckpt_path,
    )

    embedder = build_text_jepa_embedder(ckpt_path)
    vec = embedder.embed("hello")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (8,)
