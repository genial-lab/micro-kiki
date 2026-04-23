"""Stage 4 tests — K-candidate MLX inference (PB-T6).

Verifies the orchestrator builds a deterministic prompt from
``req.messages`` + recalled episodes, calls ``state.runtime.generate``
exactly ``cfg.negotiator_k`` times, forwards ``max_tokens`` (defaulting
to 512), and surfaces inference failures as 500 ``inference_error``.

Heavy deps (MLX/torch) are never imported; factories are monkeypatched.
Endpoint returns 200 end-to-end after PB-T9 lands stage 7 + OpenAI shape.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_inference_k_candidates_passed_to_negotiator(monkeypatch):
    import src.serving.full_pipeline_server as fps

    gen_calls = []

    class _Aeon:
        def recall(self, q, k=3): return [{"text": "prior episode A"}]
        def write(self, ep): pass
    class _Runtime:
        def apply(self, adapters): pass
        def generate(self, prompt, **kw):
            gen_calls.append((prompt, kw.get("max_tokens")))
            return f"cand-{len(gen_calls)}"
    class _Meta:
        def route(self, q): return [("coding", 0.9)]

    class _Neg:
        def __init__(self): self.seen: list[str] | None = None
        async def arbitrate(self, cs):
            self.seen = list(cs)
            return cs[0], {"agreement": 0.9}
    class _AB:
        async def check(self, t, ctx=None): return t, {}

    neg = _Neg()
    monkeypatch.setattr(fps, "_build_runtime", lambda cfg, **_k: _Runtime())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _Meta())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _Aeon())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: neg)
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _AB())

    cfg = fps.FullPipelineConfig.defaults()
    cfg.negotiator_k = 3
    client = TestClient(fps.make_app(cfg))
    client.post("/v1/chat/completions", json={
        "model": "kiki-meta-coding",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 77,
    })
    # runtime.generate called K=3 times
    assert len(gen_calls) == 3
    # max_tokens forwarded
    assert all(kw == 77 for _prompt, kw in gen_calls)
    # same prompt each call (deterministic prompt assembly)
    prompts = {p for p, _ in gen_calls}
    assert len(prompts) == 1
    prompt = prompts.pop()
    # prompt includes the user text and the recalled episode
    assert "hi" in prompt
    assert "prior episode A" in prompt
    # Negotiator saw exactly the 3 candidates in order
    assert neg.seen == ["cand-1", "cand-2", "cand-3"]


def test_inference_failure_returns_500(monkeypatch):
    import src.serving.full_pipeline_server as fps

    class _Aeon:
        def recall(self, q, k=3): return []
        def write(self, ep): pass
    class _Runtime:
        def apply(self, adapters): pass
        def generate(self, p, **kw): raise RuntimeError("MLX OOM")
    class _Meta:
        def route(self, q): return [("coding", 0.9)]
    class _Neg:
        async def arbitrate(self, cs): return cs[0], {}
    class _AB:
        async def check(self, t, ctx=None): return t, {}

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
    assert r.status_code == 500
    assert r.json()["error"]["type"] == "inference_error"


def test_max_tokens_defaults_to_512(monkeypatch):
    import src.serving.full_pipeline_server as fps

    gen_calls = []

    class _Aeon:
        def recall(self, q, k=3): return []
        def write(self, ep): pass
    class _Runtime:
        def apply(self, adapters): pass
        def generate(self, p, **kw):
            gen_calls.append(kw.get("max_tokens"))
            return "x"
    class _Meta:
        def route(self, q): return [("coding", 0.9)]
    class _Neg:
        async def arbitrate(self, cs): return cs[0], {}
    class _AB:
        async def check(self, t, ctx=None): return t, {}

    monkeypatch.setattr(fps, "_build_runtime", lambda cfg, **_k: _Runtime())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _Meta())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _Aeon())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _Neg())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _AB())

    client = TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))
    client.post("/v1/chat/completions", json={
        "model": "kiki-meta-coding",
        "messages": [{"role": "user", "content": "x"}],
        # no max_tokens in request
    })
    assert all(mt == 512 for mt in gen_calls), (
        f"expected 512 default, got {gen_calls}"
    )
    assert gen_calls, "expected at least one generate call"
