"""Parse-layer tests for POST /v1/chat/completions (PB-T3).

Validates request schema + OpenAI-style error shapes:
- 404 ``model_not_found`` for unknown aliases
- 422 FastAPI default for malformed bodies (missing ``messages``)
- 200 OpenAI ``chat.completion`` on happy path — full pipeline landed
  in PB-T9.

Heavy deps (MLX/torch/mlx-lm) never imported; factories are monkeypatched
with lightweight fakes. Mirrors the skeleton test pattern.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def _client(monkeypatch):
    import src.serving.full_pipeline_server as fps

    class _FakeRuntime:
        base_loaded = True

        def apply(self, adapters):
            pass

        def generate(self, prompt, **kw):
            return "stub"

    class _FakeMeta:
        def route(self, q):
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
    return TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))


def test_chat_unknown_model_404(monkeypatch):
    client = _client(monkeypatch)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "kiki-niche-absolutely-unknown",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["type"] == "model_not_found"
    assert body["error"]["param"] == "model"


def test_chat_malformed_request_422(monkeypatch):
    client = _client(monkeypatch)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "kiki-meta-coding"},  # no messages
    )
    assert r.status_code == 422


def test_chat_happy_path_200(monkeypatch):
    client = _client(monkeypatch)
    # Use a REAL meta alias (per T1 discovery the intents are:
    # quick-reply, reasoning, coding, creative, research, agentic, tool-use)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "kiki-meta-coding",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    # PB-T9 closed the end-to-end path — happy path returns OpenAI shape.
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "kiki-meta-coding"
    assert body["choices"][0]["message"]["role"] == "assistant"
