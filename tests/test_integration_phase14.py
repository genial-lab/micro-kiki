from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")
from unittest.mock import AsyncMock, MagicMock

from src.routing.router import MetaRouter
from src.search.cache import SearchCache
from src.critique.best_of_n import BestOfN
from src.orchestrator.engine import OrchestrationEngine


class TestPhase14Integration:
    def test_router_37_outputs_feed_engine(self):
        router = MetaRouter(input_dim=768, num_domains=35, num_capabilities=5)
        x = torch.randn(1, 768)
        output = router(x)

        thresholds = {
            "web_search": 0.15, "self_critique_token": 0.10,
            "self_critique_response": 0.20, "self_critique_task": 0.35,
            "deep_eval": 0.25,
        }
        active_caps = router.get_active_capabilities(output, thresholds)
        active_domains = router.get_active_domains(output, threshold=0.12, max_active=4)

        assert isinstance(active_caps, dict)
        assert len(active_caps) == 5
        assert len(active_domains[0]) <= 4

    def test_cache_integrates_with_backends(self, tmp_path):
        cache = SearchCache(db_path=str(tmp_path / "cache.sqlite"))
        cache.store(
            backend="exa", query="test",
            results=[{"title": "Result", "url": "https://a.com"}],
            ttl_seconds=3600,
        )
        hit = cache.lookup(backend="exa", query="test")
        assert hit is not None
        cache.close()

    @pytest.mark.asyncio
    async def test_engine_full_flow_mock(self):
        engine = OrchestrationEngine(
            capabilities_config={
                "web_search": {"threshold": 0.15},
                "self_critique_token": {"threshold": 0.10},
                "self_critique_response": {"threshold": 0.20},
                "self_critique_task": {"threshold": 0.35},
                "deep_eval": {"threshold": 0.25},
            },
            best_of_n_config={
                "high_threshold": 0.8, "mid_threshold": 0.5,
                "mid_n": 3, "low_n": 5,
            },
        )
        active_caps = {
            "web_search": False, "self_critique_token": True,
            "self_critique_response": False, "self_critique_task": False,
            "deep_eval": False,
        }
        result = await engine.process(
            query="Explain MoE-LoRA",
            active_capabilities=active_caps,
            generate_fn=AsyncMock(return_value=("MoE-LoRA is...", -1.0)),
            router_confidence=0.6,
        )
        assert "MoE-LoRA" in result.response
