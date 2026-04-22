"""FastAPI app factory for the factory4life OpenAI-compat serving.

The app is **backend-agnostic** — it depends only on the
:class:`MLXRuntime` Protocol from :mod:`runtime`. Swapping the
backend (vanilla LoRA / MoE-native / Fake) is done at
``create_app`` time ; the routes never know which implementation
they are talking to.

Routes (T6 scope) :

- ``GET /health``                  — uptime + runtime health dict
- ``GET /v1/models``               — list of adapter ids
- ``GET /metrics``                 — Prometheus exposition format

T7 adds ``POST /v1/chat/completions`` (non-streaming + tool
calling + JSON mode). T8 adds SSE streaming on the same route.

The app intentionally **does not** own the runtime lifecycle. The
runtime is passed in already loaded (or mocked). This keeps the
app importable without MLX on CI and makes tests trivial — a
``FakeMLXRuntime()`` instance is all that's needed.

Metrics philosophy
------------------

Minimal Prometheus exposition — 4 series that matter for agent
workloads :

- ``mlx_requests_total{endpoint, status}`` — request count.
- ``mlx_request_seconds{endpoint}`` — latency histogram.
- ``mlx_adapter_requests_total{adapter}`` — per-adapter
  selection frequency (rate limit / hotspot detection).
- ``mlx_runtime_info{runtime}`` — backend identifier + uptime.

No external ``prometheus_client`` dep ; we hand-format the text
exposition. For more serious observability, swap the ``/metrics``
handler for ``prometheus_fastapi_instrumentator`` at deploy time.
"""
from __future__ import annotations

import contextlib
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from . import schemas as s
from .runtime import MLXRuntime, RuntimeError_

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-process metrics accumulator (no external dep).
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    """Tiny in-memory counters / histograms. Not thread-safe for
    heavy concurrency — acceptable for ~100 req/s on a single
    serving process. Swap for ``prometheus_client`` at scale.
    """

    started_at: float = field(default_factory=time.time)
    requests_total: dict[tuple[str, int], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    adapter_requests_total: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    # Latency histogram : coarse buckets geared for LLM serving.
    _buckets_s: tuple[float, ...] = (
        0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf"),
    )
    latency_bucket_counts: dict[tuple[str, float], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    latency_sum: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    latency_count: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record_request(
        self, endpoint: str, status: int, elapsed_s: float,
    ) -> None:
        self.requests_total[(endpoint, status)] += 1
        self.latency_sum[endpoint] += elapsed_s
        self.latency_count[endpoint] += 1
        for bucket in self._buckets_s:
            if elapsed_s <= bucket:
                self.latency_bucket_counts[(endpoint, bucket)] += 1

    def record_adapter(self, adapter: str | None) -> None:
        self.adapter_requests_total[adapter or "base"] += 1

    def to_prometheus_text(
        self, runtime_name: str,
    ) -> str:
        """Render as text-exposition Prometheus format.

        Intentionally terse — less surface for format bugs. If
        you need histograms, summaries, or native OpenMetrics,
        pin ``prometheus_client`` and delete this method.
        """
        lines: list[str] = []
        lines.append(
            "# HELP mlx_runtime_info Runtime identifier + uptime"
        )
        lines.append("# TYPE mlx_runtime_info gauge")
        uptime = round(time.time() - self.started_at, 1)
        lines.append(
            f'mlx_runtime_info{{runtime="{runtime_name}"}} {uptime}'
        )

        lines.append(
            "# HELP mlx_requests_total Total HTTP requests by "
            "endpoint and status"
        )
        lines.append("# TYPE mlx_requests_total counter")
        for (endpoint, status), count in sorted(
            self.requests_total.items()
        ):
            lines.append(
                f'mlx_requests_total{{endpoint="{endpoint}",'
                f'status="{status}"}} {count}'
            )

        lines.append(
            "# HELP mlx_adapter_requests_total Total requests "
            "routed to each adapter"
        )
        lines.append("# TYPE mlx_adapter_requests_total counter")
        for adapter, count in sorted(
            self.adapter_requests_total.items()
        ):
            lines.append(
                f'mlx_adapter_requests_total{{adapter="{adapter}"}} '
                f"{count}"
            )

        lines.append(
            "# HELP mlx_request_seconds Request latency in seconds"
        )
        lines.append("# TYPE mlx_request_seconds histogram")
        for (endpoint, bucket), count in sorted(
            self.latency_bucket_counts.items()
        ):
            label_bucket = "+Inf" if bucket == float("inf") else bucket
            lines.append(
                f'mlx_request_seconds_bucket{{endpoint="{endpoint}",'
                f'le="{label_bucket}"}} {count}'
            )
        for endpoint, total in sorted(self.latency_sum.items()):
            lines.append(
                f'mlx_request_seconds_sum{{endpoint="{endpoint}"}} '
                f"{total:.3f}"
            )
        for endpoint, count in sorted(self.latency_count.items()):
            lines.append(
                f'mlx_request_seconds_count{{endpoint="{endpoint}"}} '
                f"{count}"
            )
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# App factory.
# ---------------------------------------------------------------------------


def create_app(
    runtime: MLXRuntime,
    *,
    model_aliases: dict[str, str] | None = None,
    model_namespace: str = "qwen3.6-35b",
    enable_cors: bool = True,
    title: str = "factory4life",
    version: str = "0.1.0",
) -> FastAPI:
    """Build a FastAPI app wired to ``runtime``.

    Parameters
    ----------
    runtime
        Anything satisfying :class:`MLXRuntime` — typically a
        :class:`FakeMLXRuntime` in tests or the real vanilla /
        MoE-native runtime in prod.
    model_aliases
        Optional client-facing → internal adapter-name map. Used
        to expose sementic aliases (``code``, ``reasoning``,
        ``chat``) that route to the real per-domain adapter
        names (``python``, ``reasoning``, ``chat-fr``). Returned
        verbatim by ``/v1/models`` ; resolved on dispatch.
    model_namespace
        Prefix for adapter ids in ``/v1/models`` responses.
        Default ``qwen3.6-35b`` gives ids like
        ``qwen3.6-35b-python``, matching the pattern T7 uses to
        resolve model → adapter.
    enable_cors
        If ``True`` (default), permissive CORS — fine for
        internal factory4life usage, tighten at deploy time.
    """

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup hook. Runtime is already built — we only warm
        # any per-app caches and log readiness.
        _LOG.info(
            "factory4life startup : runtime=%s adapters=%d",
            type(runtime).__name__,
            len(runtime.list_adapters()),
        )
        yield
        _LOG.info("factory4life shutdown")

    app = FastAPI(title=title, version=version, lifespan=lifespan)
    metrics = Metrics()

    # Stash runtime + config on app.state so tests can introspect.
    app.state.runtime = runtime
    app.state.metrics = metrics
    app.state.model_aliases = model_aliases or {}
    app.state.model_namespace = model_namespace

    if enable_cors:
        # Permissive for factory4life internal use ; tighten in
        # a production reverse proxy config.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # -----------------------------------------------------------------
    # Middleware : request id + latency recording.
    # -----------------------------------------------------------------

    @app.middleware("http")
    async def _track_request(
        request: Request, call_next,
    ):
        request_id = request.headers.get(
            "x-request-id",
        ) or uuid.uuid4().hex
        start = time.monotonic()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            # Re-raise after recording — FastAPI's exception
            # handler will turn it into a 500 JSON.
            raise
        finally:
            elapsed = time.monotonic() - start
            metrics.record_request(
                endpoint=request.url.path,
                status=status,
                elapsed_s=elapsed,
            )
        response.headers["x-request-id"] = request_id
        return response

    # -----------------------------------------------------------------
    # Routes.
    # -----------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness + readiness. Returns 200 as long as the
        runtime is alive ; no heavy ping to a backend model —
        that would bounce on startup.
        """
        rt_health = runtime.health()
        payload: dict[str, Any] = {
            "status": "ok",
            "uptime_s": round(time.time() - metrics.started_at, 1),
            "runtime": rt_health,
            "adapters_count": len(runtime.list_adapters()),
            # Surfaced at top level so k8s / autoscalers don't
            # have to parse the nested runtime dict for the one
            # capacity limit that matters.
            "max_context_tokens": rt_health.get(
                "max_context_tokens",
            ),
        }
        return payload

    @app.get("/v1/models")
    async def list_models() -> s.ModelList:
        """OpenAI-compat ``/v1/models``. Lists every adapter the
        runtime exposes, namespaced with ``model_namespace``.
        Aliases from ``model_aliases`` are added as extra
        ModelEntry rows so clients can see them too."""
        adapters = runtime.list_adapters()
        data: list[s.ModelEntry] = []
        for name in adapters:
            data.append(
                s.ModelEntry(id=f"{model_namespace}-{name}")
            )
        # Auto-router alias (T7 will implement routing).
        data.append(s.ModelEntry(id=f"{model_namespace}-auto"))
        # User-provided aliases.
        for alias_id in (model_aliases or {}).keys():
            data.append(s.ModelEntry(id=alias_id))
        return s.ModelList(data=data)

    @app.get("/metrics")
    async def metrics_endpoint() -> PlainTextResponse:
        text = metrics.to_prometheus_text(
            runtime_name=type(runtime).__name__,
        )
        return PlainTextResponse(
            content=text,
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/v1/internal/kv-status")
    async def kv_status() -> dict[str, Any]:
        """Observability endpoint for the KV cache + session pool.

        Not part of the OpenAI spec — intentionally namespaced
        under ``/v1/internal/`` so clients that enumerate ``/v1/*``
        don't stumble on it. Meant for operator dashboards and
        the autoscaling controller (which uses ``kv_bytes_free``
        to know when to open more serving processes).

        Graceful degradation : if the runtime doesn't implement
        ``kv_stats`` (older backends, test mocks without the
        method), we synthesize a minimum-viable payload from
        ``health()`` + ``list_adapters()`` so the endpoint never
        500s.
        """
        try:
            stats = runtime.kv_stats()
        except AttributeError:
            stats = {
                "runtime": type(runtime).__name__,
                "kv_stats_supported": False,
                "note": (
                    "runtime does not implement kv_stats() ; "
                    "falling back to health snapshot"
                ),
                **runtime.health(),
            }
        except NotImplementedError:
            stats = {
                "runtime": type(runtime).__name__,
                "kv_stats_supported": False,
                "note": "kv_stats() explicitly unimplemented",
            }
        return stats

    # -----------------------------------------------------------------
    # /v1/chat/completions — non-streaming + tool calling + JSON mode.
    # -----------------------------------------------------------------

    def _validate_tool_choice(
        req: s.ChatCompletionRequest,
    ) -> None:
        """Guard against malformed tool_choice / tools combos.

        OpenAI and Anthropic both 400 on ``tool_choice="required"``
        with empty ``tools``. LiteLLM propagates the error code so
        the client can fall back to a non-tool-calling pass.
        Silent demotion would cause agent chains to retry with
        the same malformed payload.
        """
        tc = req.tool_choice
        has_tools = bool(req.tools)

        if isinstance(tc, s.ToolChoice):
            if not has_tools:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "tool_choice specifies a function "
                                "but 'tools' is empty"
                            ),
                            "type": "invalid_request_error",
                            "param": "tool_choice",
                        }
                    },
                )
            declared = {t.function.name for t in req.tools or []}
            if tc.function.name not in declared:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                f"tool_choice.function.name "
                                f"{tc.function.name!r} not found in "
                                f"tools"
                            ),
                            "type": "invalid_request_error",
                            "param": "tool_choice",
                        }
                    },
                )
        elif tc == "required" and not has_tools:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "tool_choice='required' needs at least "
                            "one tool declared"
                        ),
                        "type": "invalid_request_error",
                        "param": "tool_choice",
                    }
                },
            )

    def _validate_response_format(
        req: s.ChatCompletionRequest,
    ) -> None:
        """``response_format=json_schema`` without ``json_schema``
        payload is a silent killer — the request looks valid but
        the runtime cannot enforce anything. Return 400 up front
        so Instructor gets a useful error."""
        rf = req.response_format
        if rf is None:
            return
        if rf.type == "json_schema" and rf.json_schema is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "response_format.type='json_schema' "
                            "requires 'json_schema' payload"
                        ),
                        "type": "invalid_request_error",
                        "param": "response_format",
                    }
                },
            )

    async def _sse_stream(
        request: s.ChatCompletionRequest,
        http_request: Request,
    ) -> AsyncIterator[bytes]:
        """Wrap ``runtime.generate_stream`` in the SSE wire format.

        Emits one ``data: <chunk_json>\\n\\n`` line per chunk and
        a final ``data: [DONE]\\n\\n`` sentinel. Observes
        ``http_request.is_disconnected()`` between chunks so we
        stop generating (and free the KV cache) when the client
        closes the connection — agent workloads cancel frequently
        on timeouts + chain short-circuits.

        Structured runtime errors mid-stream are serialised as a
        final SSE ``data: {"error": ...}`` frame before [DONE]
        — some LangChain versions parse that and raise cleanly,
        others fall through on [DONE], both graceful.
        """
        try:
            async for chunk in runtime.generate_stream(request):
                if await http_request.is_disconnected():
                    _LOG.info(
                        "client disconnected mid-stream, "
                        "cancelling runtime"
                    )
                    break
                payload = chunk.model_dump_json(exclude_none=True)
                yield f"data: {payload}\n\n".encode("utf-8")
        except RuntimeError_ as exc:
            error_payload = exc.to_error().model_dump_json()
            yield f"data: {error_payload}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    def _admit_context(
        request: s.ChatCompletionRequest,
    ) -> int:
        """Dynamic context admission.

        Returns the clamped ceiling for this request. Reads the
        live ``runtime.current_context_ceiling()`` (shrinks as
        the KV pool fills, grows as sessions close). Three code
        paths :

        - No explicit ``max_completion_tokens`` → return ceiling,
          runtime is free to use up to that many output tokens.
        - Explicit request ≤ ceiling → honor the request.
        - Explicit request > ceiling → raise 413
          ``context_length_exceeded`` with the ceiling in the
          error body so the client can retry with a smaller
          budget instead of silent truncation.

        Falls back to ``runtime.health()['max_context_tokens']``
        when the runtime has no ``current_context_ceiling`` method
        (older backends).
        """
        try:
            ceiling = runtime.current_context_ceiling()
        except (AttributeError, NotImplementedError):
            ceiling = int(
                runtime.health().get("max_context_tokens", 262_144)
            )

        req_max = (
            request.max_completion_tokens or request.max_tokens
        )
        if req_max is None:
            return ceiling
        if req_max > ceiling:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": {
                        "message": (
                            f"requested max_tokens ({req_max}) "
                            f"exceeds current ceiling ({ceiling}) — "
                            f"KV pool is near budget ; retry with "
                            f"a smaller value or wait for idle sessions"
                        ),
                        "type": "context_length_exceeded",
                        "param": "max_completion_tokens",
                        "code": "dynamic_ceiling",
                    }
                },
                headers={"X-Max-Context-Available": str(ceiling)},
            )
        return req_max

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: s.ChatCompletionRequest,
        http_request: Request,
    ):
        """Chat completions endpoint — non-streaming + streaming.

        Validation steps ordered so the cheapest / most
        user-impactful checks fire first :

        1. ``tool_choice`` / ``tools`` consistency (T7).
        2. ``response_format`` sanity (T7).
        3. Dynamic context admission (T9 extension) — clamp or
           reject per the live KV pool ceiling.
        4. ``stream=True`` → SSE stream (T8).
        5. Non-streaming → dispatch to ``runtime.generate``.
        6. Wrap runtime exceptions in OpenAI-shaped errors.
        """
        _validate_tool_choice(request)
        _validate_response_format(request)

        effective_max = _admit_context(request)
        # Stamp the clamp back onto the request so the runtime
        # sees it — preserves the ``max_completion_tokens ||
        # max_tokens`` fallback chain OpenAI introduced.
        if request.max_completion_tokens is None and request.max_tokens is None:
            request.max_completion_tokens = effective_max

        metrics.record_adapter(request.model)

        admission_headers = {
            "X-Max-Context-Available": str(effective_max),
        }

        if request.stream:
            return StreamingResponse(
                _sse_stream(request, http_request),
                media_type="text/event-stream",
                # Disable proxy buffering (nginx etc) that would
                # batch SSE frames and break per-token latency.
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                    **admission_headers,
                },
            )

        try:
            completion = await runtime.generate(request)
        except RuntimeError_ as exc:
            return JSONResponse(
                status_code=exc.http_status,
                content=exc.to_error().model_dump(),
                headers=admission_headers,
            )
        return JSONResponse(
            content=completion.model_dump(),
            headers=admission_headers,
        )

    return app


__all__ = ["Metrics", "create_app"]
