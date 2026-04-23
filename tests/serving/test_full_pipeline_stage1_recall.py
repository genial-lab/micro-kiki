from fastapi.testclient import TestClient


def test_recall_invoked_with_user_text(monkeypatch):
    import src.serving.full_pipeline_server as fps

    recall_calls = []

    class _Aeon:
        def recall(self, query, k=3):
            recall_calls.append((query, k))
            return [{"text": "past episode"}]
        def write(self, ep): pass

    class _Runtime:
        def apply(self, adapters): pass
        def generate(self, prompt, **kw): return "out"
    class _Meta:
        def route(self, q): return []
    class _Neg:
        async def arbitrate(self, cs): return cs[0], {}
    class _AB:
        async def check(self, t, ctx=None): return t, {}

    monkeypatch.setattr(fps, "_build_runtime", lambda cfg, **_k: _Runtime())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _Meta())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _Aeon())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _Neg())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _AB())

    app = fps.make_app(fps.FullPipelineConfig.defaults())
    client = TestClient(app)
    client.post("/v1/chat/completions", json={
        "model": "kiki-meta-coding",
        "messages": [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "Write a hello world"},
        ],
    })
    # Endpoint is 200 end-to-end after PB-T9. Regardless, recall must
    # already have been invoked with the latest user message.
    assert recall_calls, "Aeon.recall must have been called"
    query, k = recall_calls[0]
    assert query == "Write a hello world"  # last USER message drives recall
    assert k == 3


def test_recall_failure_is_non_blocking(monkeypatch):
    """If Aeon.recall raises, the request still progresses."""
    import src.serving.full_pipeline_server as fps

    class _AeonExploding:
        def recall(self, query, k=3): raise RuntimeError("aeon disk full")
        def write(self, ep): pass

    class _Runtime:
        def apply(self, adapters): pass
        def generate(self, p, **kw): return "x"
    class _Meta:
        def route(self, q): return []
    class _Neg:
        async def arbitrate(self, cs): return cs[0], {}
    class _AB:
        async def check(self, t, ctx=None): return t, {}

    monkeypatch.setattr(fps, "_build_runtime", lambda cfg, **_k: _Runtime())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _Meta())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _AeonExploding())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _Neg())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _AB())

    client = TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))
    r = client.post("/v1/chat/completions", json={
        "model": "kiki-meta-coding",
        "messages": [{"role": "user", "content": "hi"}],
    })
    # 200 end-to-end after PB-T9 — Aeon failure must NOT propagate as 500.
    assert r.status_code == 200, f"Aeon failure must not propagate as 500; got {r.status_code}: {r.text}"
