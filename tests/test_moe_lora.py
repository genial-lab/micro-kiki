from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.stacks.moe_lora import MoLoRAConfig, MoLoRALayer, _get_layer_class


class TestMoLoRAConfig:
    def test_defaults(self):
        c = MoLoRAConfig()
        assert c.rank == 16 and c.num_experts == 4 and c.top_k == 2

    def test_custom(self):
        c = MoLoRAConfig(rank=8, num_experts=8, top_k=3, alpha=16)
        assert c.rank == 8 and c.num_experts == 8


class TestMoLoRALayer:
    def test_forward_shape(self):
        layer = MoLoRALayer(768, 768, MoLoRAConfig())
        assert layer(torch.randn(2, 16, 768)).shape == (2, 16, 768)

    def test_output_shape_small(self):
        """MoLoRA output matches input shape (lora_b is zero-init, so output is zero at init)."""
        layer = MoLoRALayer(256, 256, MoLoRAConfig(rank=16, num_experts=4, top_k=2))
        out = layer(torch.randn(1, 4, 256))
        assert out.shape == (1, 4, 256)
        # lora_b is zero-initialized (standard LoRA convention), so initial output is zero
        assert out.abs().sum() == 0

    def test_parameter_count(self):
        layer = MoLoRALayer(768, 768, MoLoRAConfig())
        total = sum(p.numel() for p in layer.parameters())
        assert 90_000 < total < 200_000
