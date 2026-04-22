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
from fastapi.responses import JSONResponse, PlainTextResponse

from . import schemas as s
from .runtime import MLXRuntime

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
        payload: dict[str, Any] = {
            "status": "ok",
            "uptime_s": round(time.time() - metrics.started_at, 1),
            "runtime": runtime.health(),
            "adapters_count": len(runtime.list_adapters()),
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

    # -----------------------------------------------------------------
    # /v1/chat/completions stub — T7 will replace this body.
    # -----------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: s.ChatCompletionRequest,
    ) -> JSONResponse:
        """Non-streaming chat completions (T7 work).

        T6 ships a **minimal scaffold** that dispatches to
        ``runtime.generate`` for non-streaming requests and
        returns the result. Tool calling, JSON mode, streaming
        (SSE) land in T7 / T8.
        """
        if request.stream:
            # T8 will implement. For now, explicit 400 so clients
            # discover the limitation without silent buffering.
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "streaming not yet implemented",
                        "type": "not_implemented",
                        "param": "stream",
                    }
                },
            )
        metrics.record_adapter(request.model)
        completion = await runtime.generate(request)
        return JSONResponse(content=completion.model_dump())

    return app


__all__ = ["Metrics", "create_app"]
