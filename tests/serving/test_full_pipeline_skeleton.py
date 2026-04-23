"""Skeleton smoke tests for the 7-stage full pipeline server (PB-T2).

Heavy components (MLX/torch) are never imported here — every real factory
is monkeypatched with a lightweight fake. This test only exercises the
FastAPI wiring, ``/health``, and ``/v1/models``.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_and_models(monkeypatch):
    """Skeleton smoke: app boots with heavy components mocked."""
    import src.serving.full_pipeline_server as fps

    class _FakeRuntime:
        base_loaded = True

    class _FakeMeta:
        def route(self, query):
            return []

    class _FakeAeon:
        def recall(self, q, k=3):
            return []

        def write(self, ep):
            pass

    class _FakeNeg:
        async def arbitrate(self, cs):
            return cs[0], {}

    class _FakeAB:
        async def check(self, t, ctx=None):
            return t, {}

    monkeypatch.setattr(fps, "_build_runtime", lambda cfg, **_k: _FakeRuntime())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _FakeMeta())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _FakeAeon())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _FakeNeg())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _FakeAB())

    cfg = fps.FullPipelineConfig.defaults()
    app = fps.make_app(cfg)
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["components"]["runtime"] is True
    assert body["components"]["meta_router"] is True
    assert body["components"]["aeon"] is True
    assert body["components"]["negotiator"] is True
    assert body["components"]["antibias"] is True

    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 42
    # The 7 real meta intents are: quick-reply, reasoning, coding,
    # creative, research, agentic, tool-use.
    ids = {m["id"] for m in body["data"]}
    for intent in (
        "quick-reply",
        "reasoning",
        "coding",
        "creative",
        "research",
        "agentic",
        "tool-use",
    ):
        assert f"kiki-meta-{intent}" in ids
    # Spot-check niche
    assert "kiki-niche-stm32" in ids
    assert "kiki-niche-chat-fr" in ids
