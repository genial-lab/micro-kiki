"""Story-19: E2E smoke test for 10 domains + base fallback.

Verifies that:
1. ModelRouter.select() routes each of the 10 niche domain prompts correctly.
2. AeonPalace write + recall round-trip works for every domain.
3. All 10 niche domains get adapter != None.
4. The base query ("What is the weather?") gets adapter == None.
"""
from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
import pytest

from src.routing.model_router import ModelRouter, RouteDecision
from src.routing.router import NICHE_DOMAINS
from src.memory.aeon import AeonPalace

# ---------------------------------------------------------------------------
# 10-domain test prompts (mirrors DOMAIN_TESTS from poc_pipeline_v2.py)
# ---------------------------------------------------------------------------
DOMAIN_TESTS: dict[str, str] = {
    "kicad-dsl":   "Create a KiCad S-expression for a TQFP-48 footprint with thermal pad.",
    "spice":       "Write a SPICE netlist for a current-mode buck converter at 500kHz.",
    "emc":         "Design an EMI filter for USB 3.0 to meet CISPR 32 Class B.",
    "stm32":       "Write STM32 HAL code for DMA-based ADC on 4 channels.",
    "embedded":    "Implement a circular buffer in C for UART RX interrupt handler.",
    "power":       "Design a 48V to 12V synchronous buck converter at 5A.",
    "dsp":         "Implement a 256-point FFT in fixed-point Q15 for Cortex-M4.",
    "electronics": "Design an instrumentation amplifier with gain=100 using AD620.",
    "freecad":     "Write a FreeCAD macro for a parametric heatsink with fins.",
    "platformio":  "Write platformio.ini for ESP32-S3 + STM32F407 multi-env build.",
}

BASE_QUERY = "What is the weather?"


# ---------------------------------------------------------------------------
# Mock embed helper (deterministic, hash-based)
# ---------------------------------------------------------------------------
def _mock_embed(dim: int = 64):
    def fn(text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)
    return fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def router() -> ModelRouter:
    return ModelRouter()


@pytest.fixture
def palace() -> AeonPalace:
    return AeonPalace(dim=64, embed_fn=_mock_embed(64))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDomainRouting:
    """Verify ModelRouter routes each niche domain prompt to an adapter."""

    @pytest.mark.parametrize("domain,prompt", list(DOMAIN_TESTS.items()))
    def test_niche_domain_gets_adapter(self, router: ModelRouter, domain: str, prompt: str):
        """Each niche domain prompt should route to a non-None adapter."""
        route = router.select(prompt, domain_hint=domain)
        assert route.adapter is not None, (
            f"Domain '{domain}' should have an adapter, got None. "
            f"Route: {route}"
        )

    @pytest.mark.parametrize("domain,prompt", list(DOMAIN_TESTS.items()))
    def test_niche_domain_adapter_name(self, router: ModelRouter, domain: str, prompt: str):
        """Adapter name should contain 'stack-' prefix."""
        route = router.select(prompt, domain_hint=domain)
        assert route.adapter is not None
        assert route.adapter.startswith("stack-"), (
            f"Adapter for '{domain}' should start with 'stack-', got '{route.adapter}'"
        )

    @pytest.mark.parametrize("domain,prompt", list(DOMAIN_TESTS.items()))
    def test_niche_domain_model_id(self, router: ModelRouter, domain: str, prompt: str):
        """All niche domains should route to qwen35b."""
        route = router.select(prompt, domain_hint=domain)
        assert route.model_id == "qwen35b", (
            f"Domain '{domain}' should use qwen35b, got '{route.model_id}'"
        )

    def test_base_query_no_adapter(self, router: ModelRouter):
        """Base (non-domain) query should get adapter == None."""
        route = router.select(BASE_QUERY, domain_hint=None)
        assert route.adapter is None, (
            f"Base query should have adapter=None, got '{route.adapter}'"
        )

    def test_all_10_domains_covered(self):
        """Ensure DOMAIN_TESTS covers all 10 NICHE_DOMAINS."""
        assert set(DOMAIN_TESTS.keys()) == set(NICHE_DOMAINS), (
            f"DOMAIN_TESTS keys {set(DOMAIN_TESTS.keys())} != "
            f"NICHE_DOMAINS {set(NICHE_DOMAINS)}"
        )


class TestMemoryRoundTrip:
    """Verify AeonPalace write + recall cycle for every domain."""

    @pytest.mark.parametrize("domain,prompt", list(DOMAIN_TESTS.items()))
    def test_write_recall_cycle(self, palace: AeonPalace, domain: str, prompt: str):
        """Write an episode and recall it back by the same query text."""
        content = f"User: {prompt}\nAssistant: [mock response for {domain}]"
        episode_id = palace.write(
            content=content,
            domain=domain,
            timestamp=datetime.now(),
            source="test-e2e-smoke",
        )
        assert episode_id, "write() should return a non-empty episode ID"

        recalled = palace.recall(prompt, top_k=3)
        assert len(recalled) >= 1, (
            f"recall() for domain '{domain}' returned no episodes after write"
        )
        # The written episode should be among the recalled ones
        recalled_ids = [ep.id for ep in recalled]
        assert episode_id in recalled_ids, (
            f"Written episode {episode_id} not found in recalled IDs {recalled_ids}"
        )

    def test_base_query_write_recall(self, palace: AeonPalace):
        """Base (non-domain) query also round-trips through memory."""
        content = f"User: {BASE_QUERY}\nAssistant: I don't have weather data."
        episode_id = palace.write(
            content=content,
            domain="base",
            timestamp=datetime.now(),
            source="test-e2e-smoke",
        )
        recalled = palace.recall(BASE_QUERY, top_k=3)
        assert len(recalled) >= 1
        assert episode_id in [ep.id for ep in recalled]


class TestFullE2ESmoke:
    """Combined routing + memory smoke test for all 11 queries."""

    def test_all_domains_e2e(self, router: ModelRouter, palace: AeonPalace):
        """Run through all 10 domains: route, write, recall."""
        for domain, prompt in DOMAIN_TESTS.items():
            # Route
            route = router.select(prompt, domain_hint=domain)
            assert route.adapter is not None, f"No adapter for {domain}"

            # Write
            content = f"User: {prompt}\nAssistant: [response for {domain}]"
            ep_id = palace.write(
                content=content,
                domain=domain,
                timestamp=datetime.now(),
                source="test-e2e-smoke",
            )

            # Recall — use top_k=20 because multiple episodes compete
            recalled = palace.recall(prompt, top_k=20)
            assert any(ep.id == ep_id for ep in recalled), (
                f"Episode {ep_id} for {domain} not recalled"
            )

    def test_base_fallback_e2e(self, router: ModelRouter, palace: AeonPalace):
        """Base query: adapter=None, memory round-trip works."""
        route = router.select(BASE_QUERY, domain_hint=None)
        assert route.adapter is None

        ep_id = palace.write(
            content=f"User: {BASE_QUERY}\nAssistant: No weather data.",
            domain="base",
            timestamp=datetime.now(),
            source="test-e2e-smoke",
        )
        recalled = palace.recall(BASE_QUERY, top_k=3)
        assert any(ep.id == ep_id for ep in recalled)

    def test_adapter_none_only_for_base(self, router: ModelRouter):
        """Across all 11 queries, only the base query should get adapter=None."""
        adapters = {}
        for domain, prompt in DOMAIN_TESTS.items():
            route = router.select(prompt, domain_hint=domain)
            adapters[domain] = route.adapter

        base_route = router.select(BASE_QUERY, domain_hint=None)
        adapters["base"] = base_route.adapter

        # All 10 niches should have an adapter
        for domain in DOMAIN_TESTS:
            assert adapters[domain] is not None, f"{domain} should have adapter"

        # Base should not
        assert adapters["base"] is None, "base should have adapter=None"
