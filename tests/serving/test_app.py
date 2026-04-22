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

import json

from fastapi.testclient import TestClient

from src.serving import schemas as s
from src.serving.app import Metrics, create_app
from src.serving.runtime import (
    AdapterNotFound,
    FakeMLXRuntime,
    JSONSchemaValidationError,
)


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


def test_chat_completions_stream_returns_sse_with_done_sentinel() -> None:
    """T8 — ``stream=true`` returns SSE with ``data:`` frames and
    ends with ``data: [DONE]``."""
    rt = FakeMLXRuntime(scripted_responses={"x": "a b c"})
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert body.endswith("data: [DONE]\n\n")
    # Every non-DONE data line must be parseable JSON.
    for line in body.splitlines():
        if line.startswith("data: ") and not line.endswith("[DONE]"):
            payload = line[len("data: "):]
            obj = json.loads(payload)
            assert obj.get("object") == "chat.completion.chunk"


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


# ---------------------------------------------------------------------------
# T7 — tool calling.
# ---------------------------------------------------------------------------


def test_chat_completions_with_tools_returns_tool_call() -> None:
    """When the runtime decides to call a tool, the response
    must have ``content=null`` and ``finish_reason=tool_calls``."""
    forced = s.ToolCall(
        function=s.FunctionCall(
            name="get_weather",
            arguments=json.dumps({"city": "Paris"}),
        )
    )
    runtime = FakeMLXRuntime(force_tool_call=forced)
    app = _app(runtime=runtime)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "weather ?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    msg = body["choices"][0]["message"]
    # Silent-killer #1 : content is None, not "".
    assert msg["content"] is None
    assert msg["tool_calls"] is not None
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
    # Arguments travel as a JSON string (not dict).
    args = msg["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    assert json.loads(args) == {"city": "Paris"}
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_choice_required_without_tools_rejected() -> None:
    """``tool_choice='required'`` with empty tools → 400."""
    app = _app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "tool_choice": "required",
            },
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["error"]["type"] == "invalid_request_error"
    assert body["detail"]["error"]["param"] == "tool_choice"


def test_tool_choice_structured_name_must_match_declared_tool() -> None:
    app = _app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "existing_fn"},
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "missing_fn"},
                },
            },
        )
    assert resp.status_code == 400
    assert "not found in tools" in resp.json()["detail"]["error"]["message"]


def test_tool_choice_structured_with_empty_tools_rejected() -> None:
    """Structured tool_choice with empty tools is a logical
    contradiction — return 400 rather than silently ignore."""
    app = _app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "f"},
                },
            },
        )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# T7 — response_format (JSON mode).
# ---------------------------------------------------------------------------


def test_response_format_json_schema_without_payload_rejected() -> None:
    """Silent killer : ``{"type": "json_schema"}`` without
    ``json_schema`` payload — the runtime has nothing to
    enforce, must 400 up front."""
    app = _app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "response_format": {"type": "json_schema"},
            },
        )
    assert resp.status_code == 400
    assert (
        resp.json()["detail"]["error"]["param"] == "response_format"
    )


def test_response_format_json_object_passes_through() -> None:
    """json_object mode is looser — no schema needed. The
    runtime is expected to nudge the model via prompt
    instruction. Just verify the request reaches it."""
    app = _app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "response_format": {"type": "json_object"},
            },
        )
    assert resp.status_code == 200


def test_response_format_json_schema_with_payload_passes_through() -> None:
    app = _app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": {"type": "object"},
                        "strict": True,
                    },
                },
            },
        )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# T7 — runtime exception → OpenAI-shaped error mapping.
# ---------------------------------------------------------------------------


class _ErroringRuntime(FakeMLXRuntime):
    """Raises a user-supplied ``RuntimeError_`` subclass."""

    def __init__(self, exc_cls, message: str) -> None:
        super().__init__()
        self._exc_cls = exc_cls
        self._message = message

    async def generate(self, request):  # type: ignore[override]
        raise self._exc_cls(self._message)


def test_adapter_not_found_maps_to_404() -> None:
    rt = _ErroringRuntime(AdapterNotFound, "adapter 'xxx' missing")
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-xxx",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["type"] == "adapter_not_found"
    assert "xxx" in body["error"]["message"]


def test_json_schema_validation_error_maps_to_400_for_instructor() -> None:
    """Instructor retries on exactly ``type='json_schema_validation'``.
    Return 400 (not 500) so LiteLLM doesn't self-DDoS by retrying
    on a 5xx."""
    rt = _ErroringRuntime(
        JSONSchemaValidationError, "guided decoding unavailable",
    )
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": {"type": "object"},
                    },
                },
            },
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["type"] == "json_schema_validation"


# ---------------------------------------------------------------------------
# T8 — SSE streaming in depth.
# ---------------------------------------------------------------------------


def _parse_sse(body: str) -> list[dict]:
    """Split an SSE response body into the list of non-DONE
    JSON payloads."""
    out: list[dict] = []
    for line in body.splitlines():
        if line.startswith("data: ") and not line.endswith("[DONE]"):
            out.append(json.loads(line[len("data: "):]))
    return out


def test_stream_first_chunk_carries_role_assistant() -> None:
    """OpenAI streaming contract : the very first chunk delta
    carries ``role=assistant``, no content. Subsequent chunks
    omit role."""
    rt = FakeMLXRuntime(scripted_responses={"x": "hello"})
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
        )
    chunks = _parse_sse(resp.text)
    first = chunks[0]
    assert first["choices"][0]["delta"].get("role") == "assistant"
    assert first["choices"][0]["delta"].get("content") is None


def test_stream_final_chunk_carries_finish_reason_stop() -> None:
    """Only the final chunk may have non-null finish_reason
    (silent killer for the OpenAI SDK's streaming parser)."""
    rt = FakeMLXRuntime(scripted_responses={"x": "one two"})
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
        )
    chunks = _parse_sse(resp.text)
    # All intermediate chunks must have finish_reason=None.
    for c in chunks[:-1]:
        assert c["choices"][0].get("finish_reason") in (None,)
    # The last content-bearing chunk carries the finish_reason.
    finish_chunks = [
        c
        for c in chunks
        if c["choices"]
        and c["choices"][0].get("finish_reason") is not None
    ]
    assert len(finish_chunks) == 1
    assert finish_chunks[0]["choices"][0]["finish_reason"] == "stop"


def test_stream_content_deltas_recompose_to_original_text() -> None:
    """Concatenate all delta.content values — should match the
    scripted response (plus token-break whitespace that
    FakeMLXRuntime inserts)."""
    rt = FakeMLXRuntime(
        scripted_responses={"hello": "one two three four"},
    )
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
    chunks = _parse_sse(resp.text)
    recomposed = "".join(
        c["choices"][0]["delta"].get("content") or ""
        for c in chunks
        if c["choices"]
    )
    assert recomposed.strip() == "one two three four"


def test_stream_tool_call_deltas_concatenate_correctly() -> None:
    """LangChain accumulates tool_calls[index].function.arguments
    deltas — the sum must equal the original JSON arguments."""
    forced = s.ToolCall(
        function=s.FunctionCall(
            name="emit_json",
            arguments=json.dumps({"city": "Paris", "units": "C"}),
        )
    )
    rt = FakeMLXRuntime(force_tool_call=forced)
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "emit"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "emit_json",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "stream": True,
            },
        )
    chunks = _parse_sse(resp.text)
    recomposed = "".join(
        tc.get("function", {}).get("arguments") or ""
        for c in chunks
        for tc in (c["choices"][0]["delta"].get("tool_calls") or [])
        if c["choices"]
    )
    assert json.loads(recomposed) == {"city": "Paris", "units": "C"}
    # Final chunk has finish_reason=tool_calls.
    finish_chunks = [
        c
        for c in chunks
        if c["choices"]
        and c["choices"][0].get("finish_reason") is not None
    ]
    assert len(finish_chunks) == 1
    assert finish_chunks[0]["choices"][0]["finish_reason"] == "tool_calls"


def test_stream_include_usage_emits_final_usage_chunk() -> None:
    """LangChain's callback manager requires a terminal chunk
    with ``choices=[]`` + ``usage`` populated when
    ``stream_options.include_usage=True``. Without it, token
    counts are zero in downstream observability."""
    rt = FakeMLXRuntime(scripted_responses={"x": "a b c"})
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    chunks = _parse_sse(resp.text)
    usage_chunks = [c for c in chunks if c.get("usage") is not None]
    assert len(usage_chunks) == 1
    usage = usage_chunks[0]
    assert usage["choices"] == []
    assert usage["usage"]["total_tokens"] > 0


def test_stream_headers_disable_proxy_buffering() -> None:
    """The response must carry ``X-Accel-Buffering: no`` so
    intermediate nginx / reverse proxies don't batch SSE frames
    — that would shred per-token latency for agent clients."""
    rt = FakeMLXRuntime()
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
        )
    assert resp.headers.get("x-accel-buffering") == "no"
    assert resp.headers.get("cache-control") == "no-cache"


# ---------------------------------------------------------------------------
# T9 — /v1/internal/kv-status + long context.
# ---------------------------------------------------------------------------


def test_kv_status_endpoint_returns_runtime_snapshot() -> None:
    """The KV-status endpoint dimensions the fleet ; minimal
    shape contract : ``runtime``, ``max_context_tokens``,
    session + cache counters."""
    rt = FakeMLXRuntime()
    rt._sessions_active = 3
    rt._kv_bytes_used = 5 * 1024**3  # 5 GB
    rt._prefix_cache_entries = 7
    rt._prefix_cache_hits = 18
    rt._prefix_cache_lookups = 20
    rt._context_tokens_observed = [2048, 4096, 8192, 16384, 32768]
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.get("/v1/internal/kv-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["runtime"] == "fake"
    # 1 M native ceiling with YaRN — Qwen3.6-35B-A3B can go that
    # far and the fake reports whatever we set.
    assert body["max_context_tokens"] == 1_048_576
    assert body["sessions_active"] == 3
    assert body["kv_bytes_used"] == 5 * 1024**3
    assert body["kv_bytes_free"] >= 0
    assert body["prefix_cache_entries"] == 7
    assert body["prefix_cache_hit_rate"] == 0.9  # 18/20
    assert body["context_tokens_p50"] > 0


def test_kv_status_endpoint_graceful_when_runtime_lacks_method() -> None:
    """Older backends may not implement ``kv_stats`` — the
    endpoint must never 500. It falls back to a synthesised
    payload built from ``health()`` + a flag."""

    class BareRuntime(FakeMLXRuntime):
        def kv_stats(self):
            raise NotImplementedError

    app = _app(runtime=BareRuntime())
    with TestClient(app) as client:
        resp = client.get("/v1/internal/kv-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["kv_stats_supported"] is False


def test_health_surfaces_max_context_tokens_at_top_level() -> None:
    """Autoscalers don't parse nested runtime dicts — the
    context ceiling must be at the top level of ``/health``."""
    app = _app()
    with TestClient(app) as client:
        resp = client.get("/health")
    body = resp.json()
    assert body["max_context_tokens"] == 1_048_576


def test_kv_status_custom_budget_reflected() -> None:
    """Operators can bump the KV pool budget at runtime ; the
    endpoint must report the new number without re-init."""
    rt = FakeMLXRuntime()
    rt._kv_bytes_budget = 160 * 1024**3  # 160 GB
    rt._kv_bytes_used = 40 * 1024**3
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.get("/v1/internal/kv-status")
    body = resp.json()
    assert body["kv_bytes_budget"] == 160 * 1024**3
    assert body["kv_bytes_free"] == 120 * 1024**3


def test_dynamic_context_ceiling_header_on_success() -> None:
    """Every chat completion (including non-streaming) carries
    ``X-Max-Context-Available`` so clients can self-adjust before
    hitting a 413 on a later request."""
    app = _app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
    assert resp.status_code == 200
    available = int(resp.headers["x-max-context-available"])
    # With a fresh FakeMLXRuntime (60 GB budget, 40 KB/token),
    # ceiling is capped at max_context_tokens = 1 M.
    assert available == 1_048_576


def test_dynamic_context_shrinks_as_kv_fills() -> None:
    """As the KV pool fills, the ceiling must shrink linearly —
    this is the whole point of dynamic context : operators who
    want more concurrency cap the ceiling, those who want longer
    context keep concurrency low."""
    rt = FakeMLXRuntime()
    rt._kv_bytes_budget = 1 * 1024**3  # 1 GB
    # Use 512 MB of the 1 GB budget → remaining 512 MB ÷ 40 KB =
    # 13 107 tokens.
    rt._kv_bytes_used = 512 * 1024**2
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
    available = int(resp.headers["x-max-context-available"])
    # 512 MB / 40 KB/token ≈ 13 107 tokens. Accept ±5 %.
    assert 12_000 <= available <= 14_000


def test_explicit_max_tokens_over_ceiling_returns_413() -> None:
    """Explicit ``max_completion_tokens`` > current ceiling → 413
    ``context_length_exceeded`` with the ceiling echoed in the
    body + ``X-Max-Context-Available`` header. The client can
    retry smaller."""
    rt = FakeMLXRuntime()
    rt._kv_bytes_budget = 100 * 1024**2  # tiny budget
    rt._kv_bytes_used = 50 * 1024**2
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "max_completion_tokens": 100_000,
            },
        )
    assert resp.status_code == 413
    body = resp.json()
    assert body["detail"]["error"]["type"] == "context_length_exceeded"
    assert body["detail"]["error"]["code"] == "dynamic_ceiling"
    assert "x-max-context-available" in resp.headers


def test_max_tokens_within_ceiling_honored_verbatim() -> None:
    """Explicit ``max_completion_tokens`` within the ceiling is
    honored — we don't silently downgrade to the ceiling."""
    rt = FakeMLXRuntime()
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "max_completion_tokens": 512,
            },
        )
    assert resp.status_code == 200
    # Header echoes the ceiling, not the request value — clients
    # get the accurate "you can still go this high" signal.
    assert int(resp.headers["x-max-context-available"]) >= 512


def test_legacy_max_tokens_field_also_admitted() -> None:
    """OpenAI renamed ``max_tokens`` → ``max_completion_tokens``
    in 2024 ; the legacy field must still be honored so old
    LangChain / Instructor clients work."""
    rt = FakeMLXRuntime()
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 256,
            },
        )
    assert resp.status_code == 200


def test_stream_response_carries_admission_header() -> None:
    """Streaming responses also need ``X-Max-Context-Available``
    alongside the other SSE headers — LangChain observes them
    on the initial response."""
    rt = FakeMLXRuntime(scripted_responses={"x": "ok"})
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-python",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
        )
    assert resp.status_code == 200
    assert "x-max-context-available" in resp.headers


def test_kv_bytes_per_token_matches_qwen36_hybrid_math() -> None:
    """40 KB/token is the Qwen3.6-35B-A3B hybrid-attention KV
    cost (10 full-attn layers × 8 KV heads × 128 head_dim × 2
    (K+V) × 2 bytes BF16 = 40 KB). Canonical value — regressions
    here mean we broke the RAM sizing model."""
    rt = FakeMLXRuntime()
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.get("/v1/internal/kv-status")
    body = resp.json()
    assert body["kv_bytes_per_token"] == 40 * 1024


def test_stream_runtime_error_serialised_as_final_data_frame() -> None:
    """Mid-stream errors come through as a final ``data: {error}``
    frame before ``[DONE]``. LangChain / OpenAI SDK see a valid
    terminator either way — no parse crash."""
    rt = _ErroringRuntime(
        AdapterNotFound, "adapter 'ghost' missing",
    )

    async def _empty_stream(request):  # noqa: ANN001
        raise AdapterNotFound("adapter 'ghost' missing")
        yield  # pragma: no cover — satisfy generator typing

    rt.generate_stream = _empty_stream  # type: ignore[method-assign]
    app = _app(runtime=rt)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-ghost",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
        )
    assert resp.status_code == 200  # SSE itself always 200 ; error in-band.
    lines = [
        line[len("data: "):]
        for line in resp.text.splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    assert lines, "expected at least an error data-frame before [DONE]"
    last = json.loads(lines[-1])
    assert last["error"]["type"] == "adapter_not_found"
    assert resp.text.endswith("data: [DONE]\n\n")
