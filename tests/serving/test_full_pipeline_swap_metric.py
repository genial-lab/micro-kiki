"""The registry exposes a histogram for adapter-swap latency with a
`method` label that matches what the runtime emits (unpatch|reload)."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_registry_has_adapter_swap_metric() -> None:
    import src.serving.full_pipeline_server as fps

    _reg, metrics = fps._build_registry()
    assert "adapter_swap" in metrics, "registry missing adapter_swap entry"


def test_metrics_endpoint_exposes_swap_histogram(monkeypatch) -> None:
    """The /metrics endpoint must include kiki_adapter_swap_seconds in its
    scrape output (Prometheus HELP/TYPE lines are emitted even before the
    first observation)."""
    import src.serving.full_pipeline_server as fps

    class _Fake:
        def __init__(self, *_a, **_k) -> None:
            pass

        def apply(self, _a) -> None:
            pass

        def generate(self, *_a, **_k) -> str:
            return "ok"

        def route(self, _q):
            return [("coding", 0.9)]

        def recall(self, *_a, **_k):
            return []

        def write(self, *_a, **_k):
            pass

        async def arbitrate(self, cs):
            return cs[0], {}

        async def check(self, t, ctx=None):
            return t, {}

    monkeypatch.setattr(fps, "_build_runtime", lambda cfg, **_k: _Fake())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _Fake())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _Fake())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _Fake())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _Fake())

    client = TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "kiki_adapter_swap_seconds" in body
