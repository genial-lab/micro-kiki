"""Opt-in <think>...</think> stripping behaviour."""
from fastapi.testclient import TestClient


def _wire(monkeypatch, raw_winner: str):
    import src.serving.full_pipeline_server as fps

    class _Aeon:
        def recall(self, q, k=3): return []
        def write(self, ep): pass

    class _Runtime:
        def apply(self, a): pass
        def generate(self, p, **kw): return raw_winner

    class _Meta:
        def route(self, q): return [("coding", 0.9)]

    class _Neg:
        async def arbitrate(self, cs): return cs[0], {}

    class _AB:
        async def check(self, t, ctx=None): return t, {}

    for name, fac in (
        ("_build_runtime", lambda cfg: _Runtime()),
        ("_build_meta_router", lambda cfg: _Meta()),
        ("_build_aeon", lambda cfg: _Aeon()),
        ("_build_negotiator", lambda cfg: _Neg()),
        ("_build_antibias", lambda cfg: _AB()),
    ):
        monkeypatch.setattr(fps, name, fac)
    return TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))


def test_strip_thinking_default_false_keeps_cot(monkeypatch):
    """Default: <think> block is preserved in the response."""
    raw = "<think>\n\nLet me ponder...\n\n</think>\n\nBonjour"
    client = _wire(monkeypatch, raw)
    r = client.post("/v1/chat/completions", json={
        "model": "kiki-meta-reasoning",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert "<think>" in content
    assert "Bonjour" in content


def test_strip_thinking_true_removes_cot(monkeypatch):
    raw = "<think>\n\nLet me ponder...\n\n</think>\n\nBonjour"
    client = _wire(monkeypatch, raw)
    r = client.post("/v1/chat/completions", json={
        "model": "kiki-niche-chat-fr",
        "messages": [{"role": "user", "content": "hi"}],
        "strip_thinking": True,
    })
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert "<think>" not in content
    assert content == "Bonjour"


def test_strip_thinking_no_cot_is_noop(monkeypatch):
    """When model does not emit <think>, strip_thinking leaves text intact."""
    raw = "Just the answer."
    client = _wire(monkeypatch, raw)
    r = client.post("/v1/chat/completions", json={
        "model": "kiki-niche-chat-fr",
        "messages": [{"role": "user", "content": "hi"}],
        "strip_thinking": True,
    })
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Just the answer."


def test_strip_thinking_multiline_think(monkeypatch):
    raw = "<think>step1\nstep2\nstep3</think>\n\nDone."
    client = _wire(monkeypatch, raw)
    r = client.post("/v1/chat/completions", json={
        "model": "kiki-niche-stm32",
        "messages": [{"role": "user", "content": "hi"}],
        "strip_thinking": True,
    })
    content = r.json()["choices"][0]["message"]["content"]
    assert "step1" not in content
    assert content == "Done."
