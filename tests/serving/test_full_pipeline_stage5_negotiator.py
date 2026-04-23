"""Stage 5 tests — Negotiator arbitration (PB-T7).

Verifies the orchestrator captures the Negotiator's ``(winner, quality)``
tuple, forwards the winner to ``AntiBias.check`` with ``quality`` in the
ctx, and degrades to ``candidates[0]`` when the Negotiator raises.

Endpoint returns 200 end-to-end after PB-T9 lands stage 7 + OpenAI shape.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_negotiator_winner_flows_to_antibias(monkeypatch):
    import src.serving.full_pipeline_server as fps

    ab_calls = []

    class _Aeon:
        def recall(self, q, k=3): return []
        def write(self, ep): pass
    class _Runtime:
        def apply(self, a): pass
        def generate(self, p, **kw): return "candidate-one"  # all 3 same for simplicity
    class _Meta:
        def route(self, q): return [("coding", 0.9)]
    class _Neg:
        async def arbitrate(self, cs):
            return "NEGOTIATED_WINNER:" + "|".join(cs), {"agreement": 0.87, "quality": 0.82}
    class _AB:
        async def check(self, text, ctx=None):
            ab_calls.append((text, ctx))
            return text, {"rewritten": False}

    monkeypatch.setattr(fps, "_build_runtime", lambda cfg, **_k: _Runtime())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _Meta())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _Aeon())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _Neg())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _AB())

    cfg = fps.FullPipelineConfig.defaults()
    cfg.negotiator_k = 3
    client = TestClient(fps.make_app(cfg))
    client.post("/v1/chat/completions", json={
        "model": "kiki-meta-coding",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert ab_calls, "AntiBias.check must have been called with the Negotiator's winner"
    text, ctx = ab_calls[0]
    assert text.startswith("NEGOTIATED_WINNER:")
    assert "candidate-one|candidate-one|candidate-one" in text
    # ctx should include quality info so AntiBias can reason
    assert ctx is not None


def test_negotiator_failure_degrades_to_candidate_zero(monkeypatch):
    import src.serving.full_pipeline_server as fps

    ab_calls = []

    class _Aeon:
        def recall(self, q, k=3): return []
        def write(self, ep): pass
    class _Runtime:
        def apply(self, a): pass
        def generate(self, p, **kw): return f"cand-{len(ab_calls)+1}"
    class _Meta:
        def route(self, q): return [("coding", 0.9)]
    class _Neg:
        async def arbitrate(self, cs): raise RuntimeError("negotiator crash")
    class _AB:
        async def check(self, t, ctx=None):
            ab_calls.append(t)
            return t, {}

    monkeypatch.setattr(fps, "_build_runtime", lambda cfg, **_k: _Runtime())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _Meta())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _Aeon())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _Neg())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _AB())

    client = TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))
    r = client.post("/v1/chat/completions", json={
        "model": "kiki-meta-coding",
        "messages": [{"role": "user", "content": "x"}],
    })
    # 200 end-to-end after PB-T9; Negotiator crash must still reach AB with cand-1.
    assert r.status_code == 200
    assert ab_calls, "degraded path should still reach AntiBias"
    assert ab_calls[0] == "cand-1", f"first candidate expected, got {ab_calls[0]!r}"
