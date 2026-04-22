"""Full 7-stage cognitive pipeline server, OpenAI-compat.

Stages: MetaRouter -> adapters -> Aeon recall -> MLX inference -> Negotiator
-> anti-bias -> Aeon write. See Factory 4 Life
docs/superpowers/specs/2026-04-22-kiki-router-gateway-design.md for rationale.

This module lands only the skeleton: FastAPI app factory, ``/health``, and
``/v1/models``. Orchestration logic (request routing, adapter swap, multi-
candidate generation) is implemented in PB-T3..PB-T10.

Design notes
------------
* All heavy imports (torch, mlx, transformers) live *inside* the
  ``_build_*`` factories so that pytest collection and lightweight CI do
  not pay the import cost.
* The 5 factories are top-level callables so tests can monkeypatch them
  with fakes without instantiating the real subsystems.
* The real constructors discovered during PB-T2 introspection often
  require arguments (embedders, detectors, base-model paths) that the
  skeleton cannot supply on its own. Where that happens, the factory
  builds a minimal real instance if feasible, or raises
  ``NotImplementedError`` with a pointer to the task that will wire the
  missing dependency. The test suite monkeypatches before these bodies
  ever execute.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

from src.serving.model_aliases import ModelAlias, build_aliases, lookup

log = logging.getLogger(__name__)


def _build_registry() -> tuple[CollectorRegistry, dict]:
    """Construct a fresh Prometheus registry + per-server metric objects.

    A fresh ``CollectorRegistry`` is used (not the global default) so
    multiple ``make_app`` calls in the same process (tests!) do not
    collide on duplicate metric names.
    """
    reg = CollectorRegistry()
    return reg, {
        "requests_total": Counter(
            "kiki_requests_total",
            "Requests by model and status",
            ["model", "status"],
            registry=reg,
        ),
        "stage_latency": Histogram(
            "kiki_stage_latency_seconds",
            "Per-stage latency",
            ["stage"],
            registry=reg,
        ),
        "queue_depth": Gauge(
            "kiki_queue_depth",
            "In-flight requests",
            registry=reg,
        ),
        "rejections_total": Counter(
            "kiki_rejections_total",
            "Rejected at queue boundary",
            ["reason"],
            registry=reg,
        ),
    }


@dataclass
class FullPipelineConfig:
    """Startup configuration for the full pipeline server."""

    base_model_path: Path = Path(
        "/Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B-4bit"
    )
    adapters_root: Path = Path(
        "/Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota"
    )
    # Optional trained weights for the MetaRouter. If None, factory
    # constructs a fresh (untrained) module with default dims.
    meta_router_weights: Path | None = None
    # Embedding model path for AeonPalace. If None, factory raises — the
    # caller is expected to provide one for the real pipeline; the test
    # suite monkeypatches this factory.
    aeon_embed_model_path: Path | None = None
    negotiator_k: int = 3
    antibias_max_retries: int = 1
    max_queue_depth: int = 5

    @classmethod
    def defaults(cls) -> "FullPipelineConfig":
        return cls()


# ---------------------------------------------------------------------------
# OpenAI-compatible request schemas + error helper
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False


def _err(
    status: int,
    msg: str,
    err_type: str,
    param: str | None = None,
) -> JSONResponse:
    """Return an OpenAI-style error envelope."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": msg, "type": err_type, "param": param}},
    )


# ---------------------------------------------------------------------------
# Factories — top-level so tests can monkeypatch them with fakes.
# Each does a LAZY import so pytest collection on Linux CI does not crash.
# ---------------------------------------------------------------------------

class _MLXRuntimeAdapter:
    """Thin adapter that exposes the orchestrator's ``apply(list[str])`` /
    ``generate(prompt, max_tokens, temperature)`` contract over the native
    ``mlx_lm`` stack.

    The real ``MoELoRARuntime`` in ``src.serving.moe_lora_runtime`` only
    consumes **custom MoE-LoRA** safetensors (with ``.experts.*`` and
    ``.router_*`` keys). The V4 SOTA stacks on Studio are **vanilla MLX
    LoRA** (``fine_tune_type: lora``, keys ``…lora_a`` / ``…lora_b``) and
    therefore cannot be loaded by ``MoELoRARuntime.load_adapter``. This
    adapter takes the pragmatic path:

    * ``mlx_lm.load(base_model)`` once at startup.
    * ``apply(adapters)``: resolve the single (or first) niche name to
      ``<adapters_root>/<name>`` and call
      ``mlx_lm.tuner.utils.load_adapters`` to patch the live model in
      place. Swapping to a *different* adapter re-loads the base model
      from disk to avoid LoRA-layer stacking.
    * ``apply([])`` is a no-op (degraded path for failed MetaRouter).
    * ``generate`` uses ``mlx_lm.generate`` with
      ``sampler=make_sampler(temp=temperature)``.
    """

    def __init__(self, base_model_path: str, adapters_root: str) -> None:
        from mlx_lm import load as mlx_load  # lazy heavy import

        self._base_model_path = base_model_path
        self._adapters_root = adapters_root
        self._model, self._tokenizer = mlx_load(base_model_path)
        self._current_adapter: str | None = None

    def apply(self, adapters: list[str]) -> None:
        if not adapters:
            # Degraded path: base-model only, no adapter.
            return
        # Only one active adapter supported in V1.0 (first wins).
        name = adapters[0]
        if name == self._current_adapter:
            return
        from mlx_lm import load as mlx_load
        from mlx_lm.tuner.utils import load_adapters

        adapter_path = str(Path(self._adapters_root) / name)
        if self._current_adapter is not None:
            # Swapping: reload base to drop previous LoRA layers cleanly.
            self._model, self._tokenizer = mlx_load(self._base_model_path)
        self._model = load_adapters(self._model, adapter_path)
        self._current_adapter = name

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=float(temperature))
        return mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=int(max_tokens),
            sampler=sampler,
        )


def _build_runtime(cfg: FullPipelineConfig) -> Any:
    """Construct an MLX-backed runtime adapter (see ``_MLXRuntimeAdapter``).

    The orchestrator only calls ``.apply(list[str])`` and ``.generate(...)``
    on the returned object. We deliberately do *not* instantiate the
    native ``MoELoRARuntime`` here because V4 SOTA stacks on Studio are
    vanilla MLX LoRA, which the MoE-LoRA runtime cannot load (schema
    mismatch in ``load_adapter_projections``). See adapter docstring.
    """
    return _MLXRuntimeAdapter(
        base_model_path=str(cfg.base_model_path),
        adapters_root=str(cfg.adapters_root),
    )


def _build_meta_router(cfg: FullPipelineConfig) -> Any:
    """Return a stub MetaRouter whose ``.route(query)`` raises at call time.

    Rationale: ``make_app(cfg)`` must succeed at startup so integration
    tests can exercise the *degraded* path. The orchestrator catches the
    exception at stage 2 and falls back to ``adapters=[]`` (base-model
    only). Wired to a real implementation in V1.1.
    """

    class _StubMetaRouter:
        def route(self, query: str) -> list[tuple[str, float]]:
            raise NotImplementedError(
                "MetaRouter.route wiring deferred to V1.1"
            )

    return _StubMetaRouter()


def _build_aeon(cfg: FullPipelineConfig) -> Any:
    """Return a stub AeonPalace whose recall/write raise at call time.

    Orchestrator wraps both stage 1 (recall) and stage 7 (write) in
    try/except and degrades to empty recall / skipped write on failure.
    """

    class _StubAeon:
        def recall(self, query: str, k: int = 3) -> list[dict]:
            raise NotImplementedError("Aeon.recall deferred to V1.1")

        def write(self, episode: dict) -> None:
            raise NotImplementedError("Aeon.write deferred to V1.1")

    return _StubAeon()


def _build_negotiator(cfg: FullPipelineConfig) -> Any:
    """Return a stub Negotiator whose async ``arbitrate`` raises at call time.

    Orchestrator catches and falls back to ``candidates[0]`` with an
    empty quality dict.
    """

    class _StubNegotiator:
        async def arbitrate(
            self, candidates: list[str]
        ) -> tuple[str, dict]:
            raise NotImplementedError(
                "Negotiator.arbitrate deferred to V1.1"
            )

    return _StubNegotiator()


def _build_antibias(cfg: FullPipelineConfig) -> Any:
    """Return a stub AntiBias whose async ``check`` raises at call time.

    Orchestrator catches and passes the Negotiator winner straight through.
    """

    class _StubAntiBias:
        async def check(
            self, text: str, ctx: dict | None = None
        ) -> tuple[str, dict]:
            raise NotImplementedError(
                "AntiBiasPipeline.check deferred to V1.1"
            )

    return _StubAntiBias()


@dataclass
class _State:
    """Live server state — attached to ``app.state.kiki``."""

    runtime: object
    meta_router: object
    aeon: object
    negotiator: object
    antibias: object
    aliases: list[ModelAlias] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    queue_depth: int = 0
    cfg: FullPipelineConfig | None = None


def _build_prompt(messages: list[ChatMessage], recalled: list[dict]) -> str:
    """Assemble a Qwen-style chat prompt with optional recalled context."""
    parts: list[str] = []
    if recalled:
        context_lines = "\n".join(f"- {r.get('text', '')}" for r in recalled)
        parts.append(
            f"<|im_start|>system\nPrior relevant context:\n"
            f"{context_lines}<|im_end|>"
        )
    for m in messages:
        parts.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def make_app(cfg: FullPipelineConfig) -> FastAPI:
    """Build a FastAPI app with the 5 subsystems wired into app.state."""
    app = FastAPI(title="kiki-router full pipeline", version="0.1.0")

    state = _State(
        runtime=_build_runtime(cfg),
        meta_router=_build_meta_router(cfg),
        aeon=_build_aeon(cfg),
        negotiator=_build_negotiator(cfg),
        antibias=_build_antibias(cfg),
        aliases=build_aliases(),
        cfg=cfg,
    )
    app.state.kiki = state
    reg, metrics = _build_registry()
    app.state.kiki_metrics = metrics
    app.state.kiki_registry = reg

    from fastapi import Response as _Response

    @app.get("/metrics")
    def metrics_endpoint() -> _Response:
        # Sample queue depth at scrape time so the gauge reflects the
        # in-flight count regardless of when requests completed.
        metrics["queue_depth"].set(state.queue_depth)
        return _Response(
            generate_latest(reg),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "components": {
                "runtime": state.runtime is not None,
                "meta_router": state.meta_router is not None,
                "aeon": state.aeon is not None,
                "negotiator": state.negotiator is not None,
                "antibias": state.antibias is not None,
            },
            "uptime_s": int(time.time() - state.start_time),
            "aliases": len(state.aliases),
        }

    @app.get("/v1/models")
    def list_models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": a.model_id,
                    "object": "model",
                    "created": int(state.start_time),
                    "owned_by": "kiki-router",
                }
                for a in state.aliases
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        # Queue guard first — reject before doing any work.
        if state.queue_depth >= cfg.max_queue_depth:
            metrics["rejections_total"].labels(reason="queue_full").inc()
            metrics["requests_total"].labels(
                model=req.model, status="429"
            ).inc()
            return _err(429, "queue full, retry later", "queue_full")

        alias = lookup(req.model)
        if alias is None:
            metrics["requests_total"].labels(
                model=req.model, status="404"
            ).inc()
            return _err(
                404,
                f"model {req.model!r} not found",
                "model_not_found",
                "model",
            )

        state.queue_depth += 1
        try:
            # Drive recall from the LAST user message if present; fall back
            # to the last message content if there is no user turn.
            user_text = next(
                (m.content for m in reversed(req.messages) if m.role == "user"),
                req.messages[-1].content,
            )

            # Stage 1 — Aeon recall (best-effort, non-blocking on failure).
            with metrics["stage_latency"].labels(stage="recall").time():
                try:
                    recalled = state.aeon.recall(user_text, k=3)
                except Exception as exc:  # noqa: BLE001 — recall never blocks
                    log.warning("Aeon recall failed: %s", exc)
                    recalled = []

            # Stage 2 — adapter selection. Meta aliases consult the
            # MetaRouter; niche aliases force the requested adapter.
            with metrics["stage_latency"].labels(stage="router").time():
                if alias.mode == "meta":
                    try:
                        selections = state.meta_router.route(user_text)
                        adapters = [name for (name, _w) in selections]
                    except Exception as exc:  # noqa: BLE001 — degrade to base
                        log.warning(
                            "MetaRouter failed, falling back to base: %s", exc
                        )
                        adapters = []
                else:
                    adapters = [alias.target]

            # Stage 3 — apply adapters on the MLX runtime.
            try:
                state.runtime.apply(adapters)
            except Exception as exc:  # noqa: BLE001 — surface as 503
                metrics["requests_total"].labels(
                    model=req.model, status="503"
                ).inc()
                log.error("adapter apply failed for %s: %s", adapters, exc)
                return _err(
                    503,
                    f"adapter apply failed: {exc}",
                    "adapter_apply_failed",
                )

            # Stage 4 — inference, K candidates (K = cfg.negotiator_k).
            prompt = _build_prompt(req.messages, recalled)
            max_toks = req.max_tokens if req.max_tokens is not None else 512
            temp = req.temperature if req.temperature is not None else 0.7
            candidates: list[str] = []
            with metrics["stage_latency"].labels(stage="inference").time():
                for i in range(state.cfg.negotiator_k):
                    try:
                        candidates.append(
                            state.runtime.generate(
                                prompt,
                                max_tokens=max_toks,
                                temperature=temp,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        metrics["requests_total"].labels(
                            model=req.model, status="500"
                        ).inc()
                        log.error("inference attempt %d failed: %s", i, exc)
                        return _err(
                            500,
                            f"inference failed: {exc}",
                            "inference_error",
                        )

            # Stage 5 — Negotiator arbitrates K candidates.
            with metrics["stage_latency"].labels(stage="negotiator").time():
                try:
                    winner, quality = await state.negotiator.arbitrate(
                        candidates
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "Negotiator failed, using candidate 0: %s", exc
                    )
                    winner, quality = candidates[0], {}

            # Stage 6 — AntiBias check on the winner.
            ab_ctx = {
                "quality": quality,
                "recalled": recalled,
                "adapters": adapters,
            }
            with metrics["stage_latency"].labels(stage="antibias").time():
                try:
                    final_text, ab_report = await state.antibias.check(
                        winner, ctx=ab_ctx
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("AntiBias failed, passing through: %s", exc)
                    final_text, ab_report = winner, {}

            # Stage 7 — Aeon write (best-effort, never block on failure).
            with metrics["stage_latency"].labels(stage="memory_write").time():
                try:
                    state.aeon.write({
                        "query": user_text,
                        "response": final_text,
                        "adapters": adapters,
                        "recalled": [r.get("text", "") for r in recalled],
                        "quality": quality,
                        "antibias": ab_report,
                    })
                except Exception as exc:  # noqa: BLE001 — write never blocks
                    log.warning("Aeon write failed (non-blocking): %s", exc)

            metrics["requests_total"].labels(
                model=req.model, status="200"
            ).inc()
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": final_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": max(1, len(user_text) // 4),
                    "completion_tokens": max(1, len(final_text) // 4),
                    "total_tokens": max(
                        1, (len(user_text) + len(final_text)) // 4
                    ),
                },
            }
        finally:
            state.queue_depth -= 1

    return app


def make_default_app() -> FastAPI:
    """uvicorn --factory entrypoint."""
    return make_app(FullPipelineConfig.defaults())
