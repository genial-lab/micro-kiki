from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from src.stacks.oplora import orthogonal_projection, init_oplora_experts


class TestOPLoRA:
    def test_projection_is_orthogonal(self):
        prior = torch.randn(768, 32)
        proj = orthogonal_projection(prior, 768)
        assert (proj @ prior).abs().max() < 0.01

    def test_init_produces_low_cosine(self):
        prior = torch.randn(768, 16)
        experts = init_oplora_experts(768, 16, 4, prior)
        for a in experts:
            cos = F.cosine_similarity(a.flatten().unsqueeze(0), prior.flatten().unsqueeze(0))
            assert cos.abs().item() < 0.2

    def test_init_without_prior(self):
        experts = init_oplora_experts(768, 16, 4, None)
        assert len(experts) == 4 and experts[0].shape == (768, 16)
