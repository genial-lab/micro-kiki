"""PB-T10 — queue depth guard (429) + per-stage Prometheus metrics.

These tests assert two independent behaviors added in PB-T10:

1. ``/v1/chat/completions`` returns ``429 queue_full`` when
   ``state.queue_depth >= cfg.max_queue_depth`` at request entry.
   The depth counter is incremented on entry and decremented in
   ``finally`` so every exit path (200, 503, 500, 404, exception)
   releases the slot.

2. ``GET /metrics`` exposes four Prometheus metric families using the
   ``prometheus_client`` text format:

   * ``kiki_requests_total{model,status}`` — counter, per terminal path
   * ``kiki_stage_latency_seconds{stage}`` — histogram, per pipeline stage
   * ``kiki_queue_depth`` — gauge sampled at scrape time
   * ``kiki_rejections_total{reason}`` — counter bumped on 429 paths
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def _happy_wire(monkeypatch):
    """Monkeypatch the 5 factories to a trivially-succeeding pipeline."""
    import src.serving.full_pipeline_server as fps

    class _Aeon:
        def recall(self, q, k=3):
            return []

        def write(self, ep):
            pass

    class _Runtime:
        def apply(self, a):
            pass

        def generate(self, p, **kw):
            return "ok"

    class _Meta:
        def route(self, q):
            return [("coding", 0.9)]

    class _Neg:
        async def arbitrate(self, cs):
            return cs[0], {}

    class _AB:
        async def check(self, t, ctx=None):
            return t, {}

    for name, fac in (
        ("_build_runtime", lambda cfg, **_k: _Runtime()),
        ("_build_meta_router", lambda cfg: _Meta()),
        ("_build_aeon", lambda cfg: _Aeon()),
        ("_build_negotiator", lambda cfg: _Neg()),
        ("_build_antibias", lambda cfg: _AB()),
    ):
        monkeypatch.setattr(fps, name, fac)
    return fps


def test_queue_full_returns_429(monkeypatch):
    fps = _happy_wire(monkeypatch)
    cfg = fps.FullPipelineConfig.defaults()
    cfg.max_queue_depth = 0  # always full
    client = TestClient(fps.make_app(cfg))
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "kiki-meta-coding",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 429
    body = r.json()
    assert body["error"]["type"] == "queue_full"


def test_queue_depth_respected_one(monkeypatch):
    """With max_queue_depth=1 and serial tests, a single request succeeds."""
    fps = _happy_wire(monkeypatch)
    cfg = fps.FullPipelineConfig.defaults()
    cfg.max_queue_depth = 1
    client = TestClient(fps.make_app(cfg))
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "kiki-meta-coding",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 200


def test_metrics_endpoint_includes_per_stage_latency(monkeypatch):
    fps = _happy_wire(monkeypatch)
    client = TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))
    client.post(
        "/v1/chat/completions",
        json={
            "model": "kiki-meta-coding",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    # Key metric families
    assert "kiki_requests_total" in body
    assert "kiki_stage_latency_seconds" in body
    assert "kiki_queue_depth" in body
    # Per-stage labels
    for stage in (
        "recall",
        "router",
        "inference",
        "negotiator",
        "antibias",
        "memory_write",
    ):
        assert (
            f'stage="{stage}"' in body or f"stage=\"{stage}\"" in body
        ), f"missing stage label {stage}"
    # Model label visible on requests_total
    assert 'model="kiki-meta-coding"' in body or "kiki-meta-coding" in body


def test_rejections_counter_increments_on_429(monkeypatch):
    fps = _happy_wire(monkeypatch)
    cfg = fps.FullPipelineConfig.defaults()
    cfg.max_queue_depth = 0
    app = fps.make_app(cfg)
    client = TestClient(app)
    client.post(
        "/v1/chat/completions",
        json={
            "model": "kiki-meta-coding",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "kiki_rejections_total" in r.text
    assert 'reason="queue_full"' in r.text or "reason=\"queue_full\"" in r.text
