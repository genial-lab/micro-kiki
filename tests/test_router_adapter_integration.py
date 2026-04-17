"""Story-18: Integration test router -> adapter -> response.

Tests the full routing chain through ModelRouter.select(), verifying:
- Each of the 10 niche domains routes to the correct stack-<domain> adapter
- Base queries (no domain keywords) return adapter=None
- domain_hint overrides keyword detection
- Every RouteDecision contains a valid model_id
"""

from __future__ import annotations

import pytest

from src.routing.model_router import ModelRouter, RouteDecision, SUBCATEGORY_MAP
from src.routing.router import NICHE_DOMAINS

VALID_MODEL_IDS = {"qwen35b", "qwen480b", "devstral"}


@pytest.fixture
def router() -> ModelRouter:
    return ModelRouter()


# -----------------------------------------------------------------------
# 1. Each niche domain -> correct adapter name (stack-<domain>)
# -----------------------------------------------------------------------

class TestNicheDomainRouting:
    """For each of the 10 niche domains, ModelRouter.select() returns
    the correct adapter name stack-<domain> (or parent via SUBCATEGORY_MAP)."""

    @pytest.mark.parametrize("domain", sorted(NICHE_DOMAINS))
    def test_domain_hint_returns_correct_adapter(
        self, router: ModelRouter, domain: str
    ) -> None:
        decision = router.select("some query", domain_hint=domain)

        # Sub-category domains route to their parent adapter
        parent = SUBCATEGORY_MAP.get(domain)
        if parent and parent in NICHE_DOMAINS:
            expected_adapter = f"stack-{parent}"
        else:
            expected_adapter = f"stack-{domain}"

        assert decision.adapter == expected_adapter
        assert decision.model_id == "qwen35b"
        assert isinstance(decision, RouteDecision)

    def test_all_10_niche_domains_exist(self) -> None:
        assert len(NICHE_DOMAINS) == 10


# -----------------------------------------------------------------------
# 2. Base query (no domain keywords) -> adapter=None
# -----------------------------------------------------------------------

class TestBaseQueryRouting:
    """A query with no domain keywords and no hints returns adapter=None."""

    @pytest.mark.parametrize("query", [
        "What is the meaning of life?",
        "Tell me a joke about cats.",
        "Summarize the French revolution in three sentences.",
        "How does photosynthesis work?",
    ])
    def test_base_query_returns_no_adapter(
        self, router: ModelRouter, query: str
    ) -> None:
        decision = router.select(query)
        assert decision.adapter is None
        assert decision.model_id == "qwen35b"
        assert "default" in decision.reason.lower() or "base" in decision.reason.lower()


# -----------------------------------------------------------------------
# 3. domain_hint overrides keyword detection
# -----------------------------------------------------------------------

class TestDomainHintOverride:
    """domain_hint takes precedence over code keywords in the query."""

    def test_hint_overrides_code_keywords(self, router: ModelRouter) -> None:
        # Query contains "python" which would normally route to devstral
        decision = router.select(
            "Write a Python script for GPIO", domain_hint="embedded"
        )
        assert decision.adapter == "stack-embedded"
        assert decision.model_id == "qwen35b"

    def test_hint_overrides_debug_keyword(self, router: ModelRouter) -> None:
        decision = router.select(
            "Debug this firmware crash", domain_hint="dsp"
        )
        assert decision.adapter == "stack-dsp"
        assert decision.model_id == "qwen35b"

    def test_code_keyword_without_hint_routes_devstral(
        self, router: ModelRouter
    ) -> None:
        decision = router.select("Debug this segfault in my code")
        assert decision.adapter is None
        assert decision.model_id == "devstral"

    def test_hint_not_in_niche_falls_through(self, router: ModelRouter) -> None:
        # A hint that is NOT in NICHE_DOMAINS should not produce an adapter
        decision = router.select("some query", domain_hint="nonexistent-domain")
        assert decision.adapter is None


# -----------------------------------------------------------------------
# 4. Every RouteDecision has a valid model_id
# -----------------------------------------------------------------------

class TestValidModelId:
    """All routing paths produce a model_id in {qwen35b, qwen480b, devstral}."""

    @pytest.mark.parametrize("domain", sorted(NICHE_DOMAINS))
    def test_niche_model_id_valid(
        self, router: ModelRouter, domain: str
    ) -> None:
        decision = router.select("query", domain_hint=domain)
        assert decision.model_id in VALID_MODEL_IDS

    def test_base_model_id_valid(self, router: ModelRouter) -> None:
        decision = router.select("Hello, how are you?")
        assert decision.model_id in VALID_MODEL_IDS

    def test_code_model_id_valid(self, router: ModelRouter) -> None:
        decision = router.select("Write a Python function")
        assert decision.model_id in VALID_MODEL_IDS

    def test_deep_reasoning_model_id_valid(self, router: ModelRouter) -> None:
        decision = router.select("Explain quantum entanglement", require_deep=True)
        assert decision.model_id == "qwen480b"
        assert decision.model_id in VALID_MODEL_IDS
        assert decision.adapter is None

    def test_reason_field_is_nonempty(self, router: ModelRouter) -> None:
        for domain in NICHE_DOMAINS:
            decision = router.select("q", domain_hint=domain)
            assert decision.reason, f"Empty reason for domain={domain}"
        # Also check base + code + deep paths
        assert router.select("hello").reason
        assert router.select("python code").reason
        assert router.select("x", require_deep=True).reason
