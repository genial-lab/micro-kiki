"""Stage 7 — Aeon write + OpenAI chat.completion final response.

These tests assert the end-to-end 200 path of ``/v1/chat/completions``
once PB-T9 lands stage 7. The orchestrator must:

1. Call ``state.aeon.write(episode_dict)`` best-effort (failure must
   never block the 200 response).
2. Return a proper OpenAI ``chat.completion`` body with the final_text
   as the assistant message.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_full_pipeline_returns_openai_chat_completion(monkeypatch):
    import src.serving.full_pipeline_server as fps

    write_calls = []

    class _Aeon:
        def recall(self, q, k=3):
            return [{"text": "earlier episode"}]

        def write(self, ep):
            write_calls.append(ep)

    class _Runtime:
        def apply(self, a):
            pass

        def generate(self, p, **kw):
            return "raw-candidate"

    class _Meta:
        def route(self, q):
            return [("coding", 0.9)]

    class _Neg:
        async def arbitrate(self, cs):
            return "WINNER", {"agreement": 0.9, "quality": 0.8}

    class _AB:
        async def check(self, t, ctx=None):
            return t, {"rewritten": False}

    monkeypatch.setattr(fps, "_build_runtime", lambda cfg: _Runtime())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _Meta())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _Aeon())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _Neg())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _AB())

    client = TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "kiki-meta-coding",
            "messages": [{"role": "user", "content": "Hi there"}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # OpenAI shape
    assert body["object"] == "chat.completion"
    assert body["model"] == "kiki-meta-coding"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "WINNER"
    assert choice["finish_reason"] == "stop"
    assert "usage" in body and body["usage"]["total_tokens"] > 0

    # Aeon.write called exactly once with a full episode dict
    assert len(write_calls) == 1
    ep = write_calls[0]
    assert ep["query"] == "Hi there"
    assert ep["response"] == "WINNER"
    assert ep["adapters"] == ["coding"]
    assert "recalled" in ep
    assert (
        "earlier episode" in ep["recalled"][0]
        if isinstance(ep["recalled"][0], str)
        else True
    )


def test_niche_mode_full_pipeline_200(monkeypatch):
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
            return "niche-output"

    class _Meta:
        def route(self, q):
            raise AssertionError("MetaRouter must not fire in niche mode")

    class _Neg:
        async def arbitrate(self, cs):
            return cs[0], {"agreement": 0.9}

    class _AB:
        async def check(self, t, ctx=None):
            return t, {}

    monkeypatch.setattr(fps, "_build_runtime", lambda cfg: _Runtime())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _Meta())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _Aeon())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _Neg())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _AB())

    client = TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "kiki-niche-stm32",
            "messages": [{"role": "user", "content": "UART"}],
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "niche-output"


def test_aeon_write_failure_does_not_block_200(monkeypatch):
    import src.serving.full_pipeline_server as fps

    class _Aeon:
        def recall(self, q, k=3):
            return []

        def write(self, ep):
            raise RuntimeError("disk full")

    class _Runtime:
        def apply(self, a):
            pass

        def generate(self, p, **kw):
            return "x"

    class _Meta:
        def route(self, q):
            return [("coding", 0.9)]

    class _Neg:
        async def arbitrate(self, cs):
            return "winner", {}

    class _AB:
        async def check(self, t, ctx=None):
            return t, {}

    monkeypatch.setattr(fps, "_build_runtime", lambda cfg: _Runtime())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _Meta())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _Aeon())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _Neg())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _AB())

    client = TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "kiki-meta-coding",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    # Aeon write failing must not block the user's response
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "winner"
