from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")
from src.routing.router import MetaRouter


@pytest.fixture
def router():
    return MetaRouter(input_dim=768, num_domains=35, num_capabilities=5)


class TestMetaRouter37:
    def test_output_shape(self, router):
        x = torch.randn(1, 768)
        output = router(x)
        assert output.shape == (1, 40)

    def test_outputs_are_sigmoid(self, router):
        x = torch.randn(4, 768)
        output = router(x)
        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_domain_and_capability_split(self, router):
        x = torch.randn(1, 768)
        output = router(x)
        domains = router.get_domains(output)
        capabilities = router.get_capabilities(output)
        assert domains.shape == (1, 35)
        assert capabilities.shape == (1, 5)

    def test_active_stacks_respects_max(self, router):
        x = torch.randn(1, 768)
        output = router(x)
        active = router.get_active_domains(output, threshold=0.0, max_active=4)
        assert len(active[0]) <= 4

    def test_active_capabilities_with_thresholds(self, router):
        thresholds = {
            "web_search": 0.15,
            "self_critique_token": 0.10,
            "self_critique_response": 0.20,
            "self_critique_task": 0.35,
            "deep_eval": 0.25,
        }
        x = torch.randn(1, 768)
        output = router(x)
        active_caps = router.get_active_capabilities(output, thresholds)
        assert isinstance(active_caps, dict)
        assert all(isinstance(v, bool) for v in active_caps.values())
