"""End-to-end smoke test for 3-stack pipeline (Story-22).

Tests the PLUMBING: router -> dispatcher -> switchable -> response.
All mocked — no real model, no GPU required.
"""
from __future__ import annotations

import logging

import pytest
from unittest.mock import MagicMock, patch

from src.routing.dispatcher import dispatch, load_intent_mapping, MetaIntent, DispatchResult
from src.serving.switchable import SwitchableModel, MAX_ACTIVE_STACKS

logger = logging.getLogger(__name__)

try:
    import torch
    from src.routing.router import MetaRouter
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


@pytest.fixture
def mapping():
    return load_intent_mapping("configs/meta_intents.yaml")


@pytest.fixture
def stacks_dir(tmp_path):
    """Create a fake stacks directory with 3 adapters."""
    d = tmp_path / "stacks"
    d.mkdir()
    for name in ["stack-01-chat-fr", "stack-02-reasoning", "stack-03-python"]:
        (d / name).mkdir()
    return d


class TestRouteToCorrectStack:
    """Verify that router sigmoid outputs dispatch to the right meta-intent."""

    def test_chat_fr_routes_to_quick_reply(self, mapping):
        logits = [0.05] * 35
        logits[0] = 0.92  # chat-fr dominant
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.QUICK_REPLY
        assert 0 in result.active_domains

    def test_python_routes_to_coding(self, mapping):
        logits = [0.05] * 35
        logits[2] = 0.88  # python dominant
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.CODING
        assert 2 in result.active_domains

    def test_reasoning_routes_correctly(self, mapping):
        logits = [0.05] * 35
        logits[1] = 0.85  # reasoning dominant
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.REASONING

    def test_embedded_routes_to_research(self, mapping):
        logits = [0.05] * 35
        logits[14] = 0.80  # embedded-c
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.RESEARCH

    def test_git_routes_to_tool_use(self, mapping):
        logits = [0.05] * 35
        logits[28] = 0.75  # git-workflow
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.TOOL_USE


class TestMultiStackActivation:
    """Verify that multiple domains can be active simultaneously."""

    def test_two_coding_domains_active(self, mapping):
        logits = [0.05] * 35
        logits[2] = 0.80  # python
        logits[3] = 0.70  # typescript
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.CODING
        assert 2 in result.active_domains
        assert 3 in result.active_domains

    def test_switchable_loads_multiple_stacks(self, stacks_dir):
        model = SwitchableModel(stacks_dir=str(stacks_dir))
        model.apply_stacks(["stack-01-chat-fr", "stack-02-reasoning"])
        assert len(model.active_stacks) == 2
        assert "stack-01-chat-fr" in model.active_stacks

    def test_switchable_max_stacks_enforced(self, stacks_dir):
        model = SwitchableModel(stacks_dir=str(stacks_dir))
        with pytest.raises(ValueError, match="Cannot activate"):
            model.apply_stacks(["a", "b", "c", "d", "e"])


class TestFallbackToChat:
    """When no domain scores above threshold, should still dispatch."""

    def test_low_scores_still_dispatch(self, mapping):
        logits = [0.10] * 35  # all below threshold 0.12
        result = dispatch(logits, mapping)
        # Should still return a result (best-effort)
        assert isinstance(result, DispatchResult)
        assert isinstance(result.intent, MetaIntent)

    def test_all_zero_dispatches(self, mapping):
        logits = [0.0] * 35
        result = dispatch(logits, mapping)
        assert isinstance(result, DispatchResult)
        assert result.active_domains == []


class TestDomainSpecificRouting:
    """Test domain-specific routing for various meta-intents."""

    def test_kicad_routes_to_research(self, mapping):
        logits = [0.05] * 35
        logits[11] = 0.90  # kicad-dsl
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.RESEARCH

    def test_rtos_routes_to_agentic(self, mapping):
        logits = [0.05] * 35
        logits[15] = 0.88  # rtos
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.AGENTIC

    def test_doc_writing_routes_to_creative(self, mapping):
        logits = [0.05] * 35
        logits[29] = 0.85  # doc-writing
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.CREATIVE


class TestNoStackCase:
    """Verify behavior when switchable model has no stacks loaded."""

    def test_empty_stacks_dir(self, tmp_path):
        d = tmp_path / "empty_stacks"
        d.mkdir()
        model = SwitchableModel(stacks_dir=str(d))
        assert model.list_available() == []
        assert model.active_stacks == []

    def test_nonexistent_stacks_dir(self, tmp_path):
        model = SwitchableModel(stacks_dir=str(tmp_path / "nonexistent"))
        assert model.list_available() == []

    def test_clear_stacks_idempotent(self, stacks_dir):
        model = SwitchableModel(stacks_dir=str(stacks_dir))
        model.clear_stacks()
        model.clear_stacks()  # should not raise
        assert model.active_stacks == []

    def test_apply_empty_list(self, stacks_dir):
        model = SwitchableModel(stacks_dir=str(stacks_dir))
        model.apply_stacks([])
        assert model.active_stacks == []


class TestEndToEndPipeline:
    """Full pipeline: mock router output -> dispatch -> switchable apply."""

    def test_full_pipeline_chat(self, mapping, stacks_dir):
        # 1. Simulate router output
        logits = [0.05] * 35
        logits[0] = 0.92  # chat-fr

        # 2. Dispatch to meta-intent
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.QUICK_REPLY

        # 3. Apply stacks based on active domains
        model = SwitchableModel(stacks_dir=str(stacks_dir))
        if 0 in result.active_domains:
            model.apply_stacks(["stack-01-chat-fr"])
        assert "stack-01-chat-fr" in model.active_stacks

    def test_full_pipeline_coding(self, mapping, stacks_dir):
        logits = [0.05] * 35
        logits[2] = 0.88  # python

        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.CODING

        model = SwitchableModel(stacks_dir=str(stacks_dir))
        model.apply_stacks(["stack-03-python"])
        assert "stack-03-python" in model.active_stacks

    def test_cache_hit_on_repeated_apply(self, stacks_dir):
        """Applying same stacks twice should hit cache."""
        model = SwitchableModel(stacks_dir=str(stacks_dir))
        model.apply_stacks(["stack-01-chat-fr"])
        # Second apply with same set should be a no-op (cache key match)
        model.apply_stacks(["stack-01-chat-fr"])
        assert model.active_stacks == ["stack-01-chat-fr"]


@needs_torch
class TestMetaRouterIntegration:
    """Integration test with the actual MetaRouter nn.Module."""

    def test_router_output_shape(self):
        router = MetaRouter(input_dim=768, num_domains=35, num_capabilities=5)
        x = torch.randn(1, 768)
        out = router(x)
        assert out.shape == (1, 40)

    def test_router_outputs_are_sigmoid(self):
        router = MetaRouter(input_dim=768, num_domains=35, num_capabilities=5)
        x = torch.randn(1, 768)
        out = router(x)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_router_to_dispatch_pipeline(self):
        mapping = load_intent_mapping("configs/meta_intents.yaml")
        router = MetaRouter(input_dim=768, num_domains=35, num_capabilities=5)
        x = torch.randn(1, 768)
        out = router(x)
        domains = router.get_domains(out)
        result = dispatch(domains, mapping)
        assert isinstance(result.intent, MetaIntent)
