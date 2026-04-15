from __future__ import annotations

import pytest
from src.routing.dispatcher import dispatch, validate_mapping, MetaIntent, load_intent_mapping


@pytest.fixture
def mapping():
    return load_intent_mapping("configs/meta_intents.yaml")


class TestDispatcher:
    def test_mapping_validates(self, mapping):
        assert validate_mapping(mapping, num_domains=32)

    def test_chat_fr_dominant(self, mapping):
        logits = [0.0] * 32
        logits[0] = 0.9  # chat-fr
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.QUICK_REPLY

    def test_python_dominant(self, mapping):
        logits = [0.0] * 32
        logits[2] = 0.85  # python
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.CODING

    def test_reasoning_dominant(self, mapping):
        logits = [0.0] * 32
        logits[1] = 0.9  # reasoning
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.REASONING

    def test_embedded_research(self, mapping):
        logits = [0.0] * 32
        logits[14] = 0.8  # embedded-c
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.RESEARCH

    def test_active_domains_filtered(self, mapping):
        logits = [0.05] * 32
        logits[0] = 0.9
        logits[2] = 0.5
        result = dispatch(logits, mapping)
        assert 0 in result.active_domains
        assert 2 in result.active_domains

    def test_multi_domain_picks_highest(self, mapping):
        logits = [0.0] * 32
        logits[2] = 0.7   # python (coding)
        logits[1] = 0.8   # reasoning
        result = dispatch(logits, mapping)
        assert result.intent == MetaIntent.REASONING

    def test_all_intents_covered(self, mapping):
        intents = set(mapping.keys())
        expected = {"quick-reply", "coding", "reasoning", "creative", "research", "agentic", "tool-use"}
        assert intents == expected

    def test_confidence_positive(self, mapping):
        logits = [0.1] * 32
        result = dispatch(logits, mapping)
        assert result.confidence > 0

    def test_dispatch_result_dataclass(self, mapping):
        logits = [0.5] * 32
        result = dispatch(logits, mapping)
        assert hasattr(result, "intent")
        assert hasattr(result, "confidence")
        assert hasattr(result, "active_domains")
