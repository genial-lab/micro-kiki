from fastapi.testclient import TestClient


def test_meta_mode_uses_router(monkeypatch):
    import src.serving.full_pipeline_server as fps

    apply_calls = []

    class _Aeon:
        def recall(self, q, k=3): return []
        def write(self, ep): pass

    class _Runtime:
        def apply(self, adapters): apply_calls.append(list(adapters))
        def generate(self, p, **kw): return "r"

    class _Meta:
        def __init__(self): self.calls = []
        def route(self, q):
            self.calls.append(q)
            return [("coding", 0.9), ("python", 0.7)]

    class _Neg:
        async def arbitrate(self, cs): return cs[0], {}
    class _AB:
        async def check(self, t, ctx=None): return t, {}

    m = _Meta()
    monkeypatch.setattr(fps, "_build_runtime", lambda cfg: _Runtime())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: m)
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _Aeon())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _Neg())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _AB())

    client = TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))
    client.post("/v1/chat/completions", json={
        "model": "kiki-meta-coding",
        "messages": [{"role": "user", "content": "Write a pytest fixture"}],
    })
    assert m.calls == ["Write a pytest fixture"]
    assert apply_calls == [["coding", "python"]]


def test_niche_mode_bypasses_router(monkeypatch):
    import src.serving.full_pipeline_server as fps

    apply_calls = []
    meta_calls = []

    class _Aeon:
        def recall(self, q, k=3): return []
        def write(self, ep): pass
    class _Runtime:
        def apply(self, adapters): apply_calls.append(list(adapters))
        def generate(self, p, **kw): return "r"
    class _Meta:
        def route(self, q):
            meta_calls.append(q)
            return [("should_not_be_used", 1.0)]
    class _Neg:
        async def arbitrate(self, cs): return cs[0], {}
    class _AB:
        async def check(self, t, ctx=None): return t, {}

    monkeypatch.setattr(fps, "_build_runtime", lambda cfg: _Runtime())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _Meta())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _Aeon())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _Neg())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _AB())

    client = TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))
    client.post("/v1/chat/completions", json={
        "model": "kiki-niche-stm32",
        "messages": [{"role": "user", "content": "SPI on STM32F4"}],
    })
    assert meta_calls == [], "MetaRouter must NOT be called in niche mode"
    assert apply_calls == [["stm32"]]


def test_apply_failure_returns_503(monkeypatch):
    import src.serving.full_pipeline_server as fps

    class _Aeon:
        def recall(self, q, k=3): return []
        def write(self, ep): pass
    class _Runtime:
        def apply(self, adapters): raise FileNotFoundError("adapter missing")
        def generate(self, p, **kw): return "r"
    class _Meta:
        def route(self, q): return [("coding", 0.9)]
    class _Neg:
        async def arbitrate(self, cs): return cs[0], {}
    class _AB:
        async def check(self, t, ctx=None): return t, {}

    monkeypatch.setattr(fps, "_build_runtime", lambda cfg: _Runtime())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _Meta())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _Aeon())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _Neg())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _AB())

    client = TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))
    r = client.post("/v1/chat/completions", json={
        "model": "kiki-meta-coding",
        "messages": [{"role": "user", "content": "x"}],
    })
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "adapter_apply_failed"


def test_meta_router_failure_falls_back_to_base(monkeypatch):
    """If MetaRouter raises, selection degrades to base (empty adapter list) and continues."""
    import src.serving.full_pipeline_server as fps

    apply_calls = []

    class _Aeon:
        def recall(self, q, k=3): return []
        def write(self, ep): pass
    class _Runtime:
        def apply(self, adapters): apply_calls.append(list(adapters))
        def generate(self, p, **kw): return "r"
    class _Meta:
        def route(self, q): raise RuntimeError("router weights corrupt")
    class _Neg:
        async def arbitrate(self, cs): return cs[0], {}
    class _AB:
        async def check(self, t, ctx=None): return t, {}

    monkeypatch.setattr(fps, "_build_runtime", lambda cfg: _Runtime())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _Meta())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _Aeon())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _Neg())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _AB())

    client = TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))
    r = client.post("/v1/chat/completions", json={
        "model": "kiki-meta-coding",
        "messages": [{"role": "user", "content": "x"}],
    })
    # 200 end-to-end after PB-T9 — MetaRouter failure must degrade to base, not 500.
    assert r.status_code == 200
    assert apply_calls == [[]], "base-only fallback should apply empty adapter list"
