"""Unit tests for :mod:`src.serving.app`.

Exercise the T6 scope : ``create_app`` + ``/health``,
``/v1/models``, ``/metrics``, and the T6 scaffold of
``/v1/chat/completions`` (non-streaming only — T7/T8 will add
the full behaviour).

All tests use :class:`FakeMLXRuntime` — no MLX, no Studio. Uses
``fastapi.testclient.TestClient`` (sync) so we don't need an
event loop wrapper around a streaming endpoint (T8 will
introduce ``httpx.AsyncClient`` for SSE).
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.serving import schemas as s
from src.serving.app import Metrics, create_app
from src.serving.runtime import FakeMLXRuntime


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _app(
    runtime: FakeMLXRuntime | None = None,
    **kwargs,
):
    rt = runtime or FakeMLXRuntime(
        adapters=["python", "cpp", "chat-fr"],
    )
    return create_app(rt, **kwargs)


# ---------------------------------------------------------------------------
# /health.
# ---------------------------------------------------------------------------


def test_health_returns_ok_and_runtime_payload() -> None:
    app = _app()
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "uptime_s" in body
    assert body["runtime"]["runtime"] == "fake"
    assert body["adapters_count"] == 3


# ---------------------------------------------------------------------------
# /v1/models.
# ---------------------------------------------------------------------------


def test_v1_models_lists_adapters_with_namespace_and_aliases() -> None:
    app = _app(
        model_aliases={"code": "qwen3.6-35b-python"},
    )
    with TestClient(app) as client:
        resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = {entry["id"] for entry in body["data"]}
    # All 3 adapters namespaced.
    assert "qwen3.6-35b-python" in ids
    assert "qwen3.6-35b-cpp" in ids
    assert "qwen3.6-35b-chat-fr" in ids
    # Auto alias always present.
    assert "qwen3.6-35b-auto" in ids
    # User alias shown verbatim.
    assert "code" in ids
    # Shape strict.
    for entry in body["data"]:
        assert entry["object"] == "model"
        assert entry["owned_by"] == "factory4life"


# ---------------------------------------------------------------------------
# /metrics.
# ---------------------------------------------------------------------------


def test_metrics_is_prometheus_text_format() -> None:
    app = _app()
    with TestClient(app) as client:
        # Prime some traffic.
        client.get("/health")
        client.get("/v1/models")
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    body = resp.text
    # Standard Prometheus TYPE lines.
    assert "# TYPE mlx_runtime_info gauge" in body
    assert "# TYPE mlx_requests_total counter" in body
    assert "# TYPE mlx_request_seconds histogram" in body
    # At least the two endpoints we hit are accounted for.
    assert 'endpoint="/health"' in body
    assert 'endpoint="/v1/models"' in body


def test_metrics_records_adapter_selection() -> None:
    app = _app()
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        resp = client.get("/metrics")
    assert 'adapter="qwen3.6-35b-python"' in resp.text


def test_metrics_latency_histogram_buckets() -> None:
    """Every bucket ≥ elapsed must be incremented for each
    request. FakeMLXRuntime is instant so all requests land
    in the 0.1s bucket."""
    app = _app()
    with TestClient(app) as client:
        for _ in range(3):
            client.get("/health")
        resp = client.get("/metrics")
    body = resp.text
    # 3 health requests, all instant → bucket 0.1 has ≥ 3.
    line_010 = next(
        ln for ln in body.splitlines()
        if 'endpoint="/health"' in ln
        and 'le="0.1"' in ln
        and ln.startswith("mlx_request_seconds_bucket")
    )
    count = int(line_010.rsplit(" ", 1)[-1])
    assert count >= 3


# ---------------------------------------------------------------------------
# /v1/chat/completions (T6 scaffold).
# ---------------------------------------------------------------------------


def test_chat_completions_non_streaming_basic() -> None:
    app = _app(
        runtime=FakeMLXRuntime(
            scripted_responses={"hello": "hi there"},
        )
    )
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "hello world"}],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["content"] == "hi there"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] > 0


def test_chat_completions_rejects_stream_until_t8() -> None:
    """T6 ships a non-streaming scaffold ; streaming goes out
    with T8. Until then, ``stream=true`` returns a clean 400
    with ``type=not_implemented`` so clients discover the
    limitation rather than get silent buffering."""
    app = _app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["error"]["type"] == "not_implemented"


def test_chat_completions_validates_schema() -> None:
    """Invalid request bodies should 422 via Pydantic's default
    handler — the middleware must not eat the validation error."""
    app = _app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                # Missing 'messages' → 422.
            },
        )
    assert resp.status_code == 422


def test_chat_completions_accepts_litellm_extra_body() -> None:
    """LiteLLM-forwarded provider-specific fields must not
    cause 400/422. The schema drops them silently."""
    app = _app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "custom_openrouter_header": "foo",
                "anthropic_metadata": {"tenant": "a"},
            },
        )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Metrics class — direct unit test.
# ---------------------------------------------------------------------------


def test_metrics_prometheus_text_without_traffic_is_still_valid() -> None:
    """An empty ``/metrics`` response must still be valid
    Prometheus exposition format (headers + types only)."""
    m = Metrics()
    out = m.to_prometheus_text("TestRuntime")
    assert out.endswith("\n")
    assert "# TYPE mlx_runtime_info gauge" in out
    assert 'mlx_runtime_info{runtime="TestRuntime"}' in out


def test_metrics_records_and_renders() -> None:
    m = Metrics()
    m.record_request("/v1/chat/completions", 200, 0.35)
    m.record_request("/v1/chat/completions", 200, 1.2)
    m.record_request("/v1/chat/completions", 500, 0.05)
    m.record_adapter("python")
    m.record_adapter("python")
    m.record_adapter(None)  # → "base"
    out = m.to_prometheus_text("FakeMLXRuntime")
    # 2 distinct (endpoint, status) tuples in the counter.
    assert out.count("mlx_requests_total{") == 2
    # Two adapters tracked : python + base.
    assert 'adapter="python"} 2' in out
    assert 'adapter="base"} 1' in out
    # Histogram sums are reasonable.
    assert "mlx_request_seconds_sum{" in out
    assert "mlx_request_seconds_count{" in out


# ---------------------------------------------------------------------------
# x-request-id header round-trip.
# ---------------------------------------------------------------------------


def test_request_id_header_roundtrip() -> None:
    """Client-supplied ``x-request-id`` must be echoed back for
    correlation in agent chain logs. When absent, a fresh one
    is generated."""
    app = _app()
    with TestClient(app) as client:
        resp1 = client.get(
            "/health", headers={"x-request-id": "test-123"},
        )
        resp2 = client.get("/health")
    assert resp1.headers["x-request-id"] == "test-123"
    # Generated one exists and looks hex-ish.
    assert "x-request-id" in resp2.headers
    generated = resp2.headers["x-request-id"]
    assert len(generated) >= 16
