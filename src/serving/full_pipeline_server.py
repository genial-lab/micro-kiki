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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.serving.model_aliases import ModelAlias, build_aliases, lookup

log = logging.getLogger(__name__)


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

def _build_runtime(cfg: FullPipelineConfig) -> Any:
    """Construct a ``MoELoRARuntime`` and load the base model.

    The real constructor takes a ``MoELoRAConfig | None`` — not a
    base_model_path — so we instantiate then call ``load_base_model``.
    Adapter loading per-request is the orchestrator's job (PB-T4+).
    """
    from src.serving.moe_lora_runtime import MoELoRARuntime  # lazy MLX/torch

    runtime = MoELoRARuntime()
    runtime.load_base_model(cfg.base_model_path)
    return runtime


def _build_meta_router(cfg: FullPipelineConfig) -> Any:
    """Construct the query-level adapter selector.

    The orchestrator expects ``state.meta_router.route(query) ->
    list[tuple[str, float]]`` where each tuple is ``(niche_name,
    weight)`` and ``niche_name`` is an entry in ``NICHE_DOMAINS``.

    The real ``MetaRouter`` in ``src.routing.router`` is a pure
    ``torch.nn.Module`` operating on a 768-dim embedding — no tokenizer,
    no text-to-embedding front-end — and ``src.routing.dispatcher``
    maps the 35 sigmoid outputs to 7 *meta-intents* (QUICK_REPLY,
    CODING, …), not to niche domain names. Neither exposes the
    ``.route(query)`` contract directly, and no trained weights are
    available locally yet.

    We therefore defer the real wrapper to PB-T11 (Studio integration)
    and raise here. The test suite monkeypatches this factory with a
    fake ``route(query)`` implementation, so unit tests remain green.
    """
    raise NotImplementedError(
        "MetaRouter.route(query) wrapper deferred to PB-T11 Studio "
        "integration. The real MetaRouter (src.routing.router) is a "
        "torch.nn.Module over 768-dim embeddings; the dispatcher "
        "(src.routing.dispatcher) maps to 7 meta-intents, not niche "
        "names. Tests monkeypatch this factory with fakes.",
    )


def _build_aeon(cfg: FullPipelineConfig) -> Any:
    """Construct an ``AeonPalace`` memory instance.

    Real constructor requires either an ``embed_fn`` or an embedding
    ``model_path``. No ``create_aeon_palace`` helper exists in
    ``src.memory.aeon``. If the config does not provide a model path, we
    raise ``NotImplementedError`` — PB-T5 will wire the real embedder.
    Tests monkeypatch this factory before it executes.
    """
    from src.memory.aeon import AeonPalace  # lazy sentence-transformers

    if cfg.aeon_embed_model_path is None:
        raise NotImplementedError(
            "AeonPalace requires an embedding model; set "
            "FullPipelineConfig.aeon_embed_model_path or wire a real "
            "embedder in PB-T5.",
        )
    return AeonPalace(model_path=str(cfg.aeon_embed_model_path))


def _build_negotiator(cfg: FullPipelineConfig) -> Any:
    """Construct a ``Negotiator`` (CAMP: extractor + judge + catfish).

    Real constructor is ``Negotiator(extractor, judge, catfish)`` — it
    does *not* accept a ``k`` kwarg or a generate_fn. Its three deps all
    currently depend on an async ``generate_fn`` that is the server's
    own inference function (circular). PB-T6/T7 will wire the real
    generate_fn and instantiate the three subsystems with it; until
    then, raise so real startup is explicit about the missing dep.
    Tests monkeypatch this factory before it executes.
    """
    raise NotImplementedError(
        "Negotiator requires ArgumentExtractor + AdaptiveJudge + "
        "CatfishModule, all of which need the orchestrator's generate_fn. "
        "Wire in PB-T6/T7.",
    )


def _build_antibias(cfg: FullPipelineConfig) -> Any:
    """Construct the ``AntiBiasPipeline``.

    Real constructor is ``AntiBiasPipeline(detector, generate_fn=None,
    config=None)``. ``max_retries`` lives on ``PipelineConfig``. The
    ``ReasoningBiasDetector`` itself depends on a generate_fn for LLM-
    backed bias detection, same circular dep as Negotiator. PB-T8 will
    wire it. Tests monkeypatch this factory before it executes.
    """
    raise NotImplementedError(
        "AntiBiasPipeline requires a ReasoningBiasDetector with a real "
        "generate_fn. Wire in PB-T8.",
    )


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
        alias = lookup(req.model)
        if alias is None:
            return _err(
                404,
                f"model {req.model!r} not found",
                "model_not_found",
                "model",
            )

        # Drive recall from the LAST user message if present; fall back to
        # the last message content if there is no user turn (edge case).
        user_text = next(
            (m.content for m in reversed(req.messages) if m.role == "user"),
            req.messages[-1].content,
        )

        # Stage 1 — Aeon recall (best-effort, non-blocking on failure).
        try:
            recalled = state.aeon.recall(user_text, k=3)
        except Exception as exc:  # noqa: BLE001 — recall must not block
            log.warning("Aeon recall failed: %s", exc)
            recalled = []

        # Stage 2 — adapter selection. Meta aliases consult the MetaRouter;
        # niche aliases force the requested adapter and skip the router.
        if alias.mode == "meta":
            try:
                selections = state.meta_router.route(user_text)
                adapters = [name for (name, _weight) in selections]
            except Exception as exc:  # noqa: BLE001 — degrade to base
                log.warning("MetaRouter failed, falling back to base: %s", exc)
                adapters = []
        else:
            adapters = [alias.target]

        # Stage 3 — apply adapters on the MLX runtime.
        try:
            state.runtime.apply(adapters)
        except Exception as exc:  # noqa: BLE001 — surface as 503
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
                log.error("inference attempt %d failed: %s", i, exc)
                return _err(
                    500,
                    f"inference failed: {exc}",
                    "inference_error",
                )

        # Stage 5 — Negotiator arbitrates K candidates → winner + quality signals
        try:
            winner, quality = await state.negotiator.arbitrate(candidates)
        except Exception as exc:  # noqa: BLE001
            log.warning("Negotiator failed, using candidate 0: %s", exc)
            winner, quality = candidates[0], {}

        # Pass winner + quality to AntiBias for stage 6 (tests can observe via AB monkeypatch).
        ab_ctx = {"quality": quality, "recalled": recalled, "adapters": adapters}
        try:
            final_text, ab_report = await state.antibias.check(winner, ctx=ab_ctx)
        except Exception as exc:  # noqa: BLE001
            log.warning("AntiBias failed, passing through: %s", exc)
            final_text, ab_report = winner, {}

        _ = (final_text, ab_report)

        # Stage 7 and final response shape land in PB-T8/T9.
        return _err(
            501,
            "pipeline partially wired (stages 1-5)",
            "not_implemented",
        )

    return app


def make_default_app() -> FastAPI:
    """uvicorn --factory entrypoint."""
    return make_app(FullPipelineConfig.defaults())
