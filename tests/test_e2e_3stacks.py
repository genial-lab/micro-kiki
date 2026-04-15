"""E2E smoke test for 3-stack pipeline (mock-based, no real model)."""
from __future__ import annotations

import pytest
from src.routing.dispatcher import dispatch, load_intent_mapping, MetaIntent
from src.serving.switchable import SwitchableModel


@pytest.fixture
def mapping():
    return load_intent_mapping("configs/meta_intents.yaml")


class TestE2E3Stacks:
    def test_chat_fr_routed_correctly(self, mapping):
        # Simulate router output where chat-fr (idx 0) is dominant
        logits = [0.05] * 32
        logits[0] = 0.92
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.QUICK_REPLY
        assert 0 in result.active_domains

    def test_reasoning_routed_correctly(self, mapping):
        logits = [0.05] * 32
        logits[1] = 0.88
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.REASONING

    def test_python_routed_correctly(self, mapping):
        logits = [0.05] * 32
        logits[2] = 0.85
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.CODING

    def test_switchable_model_no_crash(self, tmp_path):
        stacks = tmp_path / "stacks"
        stacks.mkdir()
        (stacks / "stack-01-chat-fr").mkdir()
        model = SwitchableModel(stacks_dir=str(stacks))
        available = model.list_available()
        assert "stack-01-chat-fr" in available
        model.apply_stacks([])
        model.clear_stacks()
