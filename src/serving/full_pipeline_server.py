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
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Qwen3/Qwen3.6 emit a chain-of-thought inside <think>...</think>. Callers
# opt in to stripping via ``strip_thinking=True`` on the request — default
# is False so reasoning-heavy models (kiki-meta-reasoning, kiki-meta-research,
# kiki-niche-reasoning, …) keep their CoT visible to consumers that need it.
_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_PATTERN.sub("", text).lstrip()

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

# Attach our own StreamHandler so adapter-swap / eager-materialize /
# preload events surface in whatever captures stdout/stderr (e.g. the
# nohup log file). Uvicorn's default logging configuration only wires
# up ``uvicorn.access`` / ``uvicorn.error`` — custom module loggers
# silently drop their records otherwise.
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


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
        "adapter_swap": Histogram(
            "kiki_adapter_swap_seconds",
            "Adapter swap latency by method "
            "(unpatch = zero-swap, reload = full base reload fallback)",
            ["method"],
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
    # When True, strip <think>...</think> chain-of-thought from the final
    # response. Default False: keep CoT visible (reasoning-heavy models
    # benefit from it). Set True for short UI-bound chat turns.
    strip_thinking: bool = False


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
      place. Swapping to a *different* adapter unpatches the previous
      LoRA wrappers via :mod:`src.serving.mlx_unpatch` (O(10ms)) rather
      than reloading the 19 GB base from disk. A base reload is the
      safe fallback if unpatching raises.
    * ``apply([])`` is a no-op (degraded path for failed MetaRouter).
    * ``generate`` uses ``mlx_lm.generate`` with
      ``sampler=make_sampler(temp=temperature)``.
    """

    def __init__(
        self,
        base_model_path: str,
        adapters_root: str,
        preload_all: bool = True,
        eager_materialize: bool = True,
        swap_metric: Any | None = None,
    ) -> None:
        from mlx_lm import load as mlx_load  # lazy heavy import
        import mlx.core as mx

        self._base_model_path = base_model_path
        self._adapters_root = adapters_root
        self._swap_metric = swap_metric
        self._model, self._tokenizer = mlx_load(base_model_path)
        if eager_materialize:
            # MLX evaluates lazily by default. Force every parameter
            # (attention + MoE experts + shared_expert + embeddings)
            # to be materialized in Metal/unified memory at startup so
            # the first inference doesn't pay the lazy-eval cost for
            # the 256 experts × 48 layers of Qwen3.6-35B-A3B.
            try:
                mx.eval(self._model.parameters())
                log.info(
                    "eager-materialized all base model parameters "
                    "(including 256 MoE experts)"
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "eager materialization failed (%s), first inference "
                    "may pay lazy-eval cost",
                    exc,
                )
        self._current_adapter: str | None = None
        # Preload all adapter safetensors into RAM at startup. Studio has
        # 512 GB so caching 34 × ~500 MB LoRA state dicts (~17 GB total)
        # is free. Swapping then skips the disk read half of the cost —
        # the base reload is still required to drop previous LoRA layers
        # cleanly (mlx_lm has no unpatch primitive as of 0.31.3), but
        # disk I/O for the adapter itself is eliminated.
        self._adapter_cache: dict[str, object] = {}
        if preload_all:
            self._preload_adapters()

    def _preload_adapters(self) -> None:
        """Prime the filesystem cache for every adapter.safetensors.

        On a 512 GB Apple Silicon host the page cache readily absorbs the
        ~17 GB of LoRA weights (34 × ~500 MB), so subsequent mmap-based
        reads inside ``load_adapters`` skip disk I/O. This is a lightweight
        alternative to holding a parallel dict of MLX tensors in memory —
        ``mlx_lm.tuner.utils.load_adapters`` already handles LoRA rank,
        target modules, and scale from each adapter's ``adapter_config.json``;
        we don't try to reimplement that plumbing.
        """
        root = Path(self._adapters_root)
        if not root.exists():
            log.warning("adapters_root %s missing, no preload", root)
            return
        count = 0
        bytes_touched = 0
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            st = child / "adapters.safetensors"
            if not st.exists():
                continue
            try:
                with st.open("rb") as f:
                    while f.read(1 << 20):  # 1 MiB chunks
                        pass
                bytes_touched += st.stat().st_size
                self._adapter_cache[child.name] = True  # marker; actual
                                                        # weights live in page cache
                count += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("preload failed for %s: %s", child.name, exc)
        log.info(
            "primed page cache for %d adapters (%.1f GB)",
            count, bytes_touched / (1024**3),
        )

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
            # Zero-swap path: unpatch LoRA wrappers in place (O(10ms))
            # instead of reloading the 19 GB base from disk. Fall back
            # to a full reload if unpatch raises — keeps the behaviour
            # of the original implementation as a safety net.
            t0 = time.perf_counter()
            try:
                from src.serving.mlx_unpatch import unpatch_all_lora

                unpatch_all_lora(self._model)
                dt = time.perf_counter() - t0
                log.info(
                    "adapter unpatched in %.1f ms (%s -> %s)",
                    dt * 1000,
                    self._current_adapter,
                    name,
                )
                if self._swap_metric is not None:
                    self._swap_metric.labels(method="unpatch").observe(dt)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "unpatch_all_lora failed (%s); falling back to "
                    "full base reload",
                    exc,
                )
                t1 = time.perf_counter()
                self._model, self._tokenizer = mlx_load(self._base_model_path)
                if self._swap_metric is not None:
                    self._swap_metric.labels(method="reload").observe(
                        time.perf_counter() - t1
                    )
        # Observe the load_adapters call itself. When self._current_adapter
        # is None we emit method="initial" so dashboards can separate cold-
        # boot latency from steady-state swaps; otherwise the unpatch/reload
        # observations above already cover the cost that matters.
        t_apply = time.perf_counter()
        self._model = load_adapters(self._model, adapter_path)
        if self._swap_metric is not None and self._current_adapter is None:
            self._swap_metric.labels(method="initial").observe(
                time.perf_counter() - t_apply
            )
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


def _build_runtime(cfg: FullPipelineConfig, swap_metric: Any | None = None) -> Any:
    """Construct an MLX-backed runtime adapter (see ``_MLXRuntimeAdapter``).

    The orchestrator only calls ``.apply(list[str])`` and ``.generate(...)``
    on the returned object. We deliberately do *not* instantiate the
    native ``MoELoRARuntime`` here because V4 SOTA stacks on Studio are
    vanilla MLX LoRA, which the MoE-LoRA runtime cannot load (schema
    mismatch in ``load_adapter_projections``). See adapter docstring.

    ``swap_metric`` is the Prometheus ``kiki_adapter_swap_seconds``
    histogram; ``None`` keeps the runtime usable in tests without a
    registry.
    """
    return _MLXRuntimeAdapter(
        base_model_path=str(cfg.base_model_path),
        adapters_root=str(cfg.adapters_root),
        swap_metric=swap_metric,
    )


class _MetaRouterV4Adapter:
    """Real adapter over the trained router-v4 (``output/router-v4/``).

    Architecture (from ``output/router-v4/meta.json``): a 2-layer MLP
    ``Linear(384, 512) -> ReLU -> Linear(512, 34)`` with sigmoid applied
    at inference. Input embeddings come from
    ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim). Output is 34
    niche-domain sigmoids in the label-map order (``chat-fr``, …
    ``yaml-json``). No ``base`` index — when nothing exceeds threshold
    the orchestrator falls back to base-only.

    Exposes ``.route(query: str) -> list[tuple[str, float]]`` matching
    the orchestrator's contract, ordered top-K by score.
    """

    def __init__(
        self,
        weights_path: Path,
        meta_path: Path,
        threshold: float = 0.12,
        max_active: int = 4,
    ) -> None:
        import json
        import torch
        import torch.nn as nn
        from safetensors.torch import load_file

        meta = json.loads(Path(meta_path).read_text())
        self._domains: list[str] = list(meta["domains"])
        self._threshold = float(threshold)
        self._max_active = int(max_active)

        input_dim = int(meta.get("embedding_dim", 384))
        hidden_dim = int(meta.get("hidden_dim", 512))
        num_domains = int(meta.get("num_domains", 34))

        self._net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        )
        state_dict = load_file(str(weights_path))
        # safetensors keys use "0.weight" / "3.weight" indices matching
        # the Sequential positions (Linear at 0, Linear at 3 with the
        # non-parametric ReLU at 1 and an implicit Dropout slot at 2).
        # Our Sequential here has Linear at 0, ReLU at 1, Linear at 2 —
        # so remap key "3.*" -> "2.*".
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("0."):
                remapped[k] = v
            elif k.startswith("3."):
                remapped[f"2.{k.split('.', 1)[1]}"] = v
        self._net.load_state_dict(remapped, strict=True)
        self._net.eval()

        from sentence_transformers import SentenceTransformer

        self._encoder = SentenceTransformer(
            meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        )

    def route(self, query: str) -> list[tuple[str, float]]:
        import torch

        vec = self._encoder.encode(
            [query], convert_to_numpy=False, show_progress_bar=False
        )
        x = vec[0] if isinstance(vec, list) else vec[0]
        if not hasattr(x, "float"):
            x = torch.tensor(x, dtype=torch.float32)
        # Encoder may place tensors on MPS (Apple Silicon); the tiny
        # router MLP stays on CPU to avoid a device migration just for
        # a 34×512 matmul.
        x = x.detach().to("cpu", dtype=torch.float32)
        with torch.no_grad():
            logits = self._net(x.unsqueeze(0))[0]
            scores = torch.sigmoid(logits)
        pairs = [
            (self._domains[i], float(scores[i].item()))
            for i in range(len(self._domains))
            if float(scores[i].item()) > self._threshold
        ]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs[: self._max_active]


def _build_meta_router(cfg: FullPipelineConfig) -> Any:
    """Build a real router-v4 adapter if weights exist, else a stub.

    The weights live at ``<repo_root>/output/router-v4/router.safetensors``
    with the sibling ``meta.json`` describing the architecture and label
    order. When both files are present the returned object routes text
    queries via ``all-MiniLM-L6-v2`` → MLP → sigmoid. When they are
    absent (e.g., CI, fresh checkout), a stub is returned whose
    ``.route(query)`` raises so the orchestrator degrades to base-only.
    """
    repo_root = Path(__file__).resolve().parents[2]
    weights = repo_root / "output" / "router-v4" / "router.safetensors"
    meta = repo_root / "output" / "router-v4" / "meta.json"
    if weights.exists() and meta.exists():
        try:
            return _MetaRouterV4Adapter(weights_path=weights, meta_path=meta)
        except Exception as exc:  # noqa: BLE001
            log.warning("router-v4 load failed (%s), using stub", exc)

    # Fall back to stub (degraded path: base-only inference).

    class _StubMetaRouter:
        def route(self, query: str) -> list[tuple[str, float]]:
            raise NotImplementedError(
                "MetaRouter weights not available (output/router-v4/ missing). "
                "Orchestrator degrades to base-only at stage 2."
            )

    return _StubMetaRouter()


def _build_meta_router_legacy_stub(cfg: FullPipelineConfig) -> Any:
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


class _AeonAdapter:
    """Adapter exposing the orchestrator's ``.recall(q, k)`` / ``.write(dict)``
    contract over the real ``src.memory.aeon.AeonPalace``.

    * ``recall`` maps ``AeonPalace.recall`` (returning ``list[Episode]``)
      to ``list[dict]`` with a ``"text"`` key as the orchestrator builds
      prompts from ``r.get("text", "")``.
    * ``write`` flattens the orchestrator's episode dict (``query`` /
      ``response`` / ``adapters`` / …) into ``AeonPalace.write(content,
      domain, metadata)`` — the first active adapter becomes the domain,
      the response becomes the content, and the rest lands in metadata.
    """

    def __init__(self, palace: Any) -> None:
        self._palace = palace

    def recall(self, query: str, k: int = 3) -> list[dict]:
        episodes = self._palace.recall(query, top_k=int(k))
        return [
            {
                "text": ep.content,
                "domain": ep.domain,
                "ts": ep.timestamp.isoformat(),
            }
            for ep in episodes
        ]

    def write(self, episode: dict) -> None:
        content = (
            episode.get("response")
            or episode.get("query")
            or ""
        )
        if not content:
            return
        adapters = episode.get("adapters") or []
        domain = adapters[0] if adapters else "general"
        meta = {
            "query": episode.get("query", ""),
            "adapters": adapters,
            "quality": episode.get("quality", {}),
            "antibias": episode.get("antibias", {}),
            "recalled": episode.get("recalled", []),
        }
        self._palace.write(content=content, domain=domain, metadata=meta)


def _build_aeon(cfg: FullPipelineConfig) -> Any:
    """Construct a real AeonPalace (MiniLM-L6-v2 384-dim) wrapped in
    ``_AeonAdapter``. Falls back to a stub that raises at call time when
    sentence-transformers is unavailable.
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        from src.memory.aeon import AeonPalace

        encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        def _embed(text: str):
            vec = encoder.encode(
                [text], convert_to_numpy=True, show_progress_bar=False
            )
            return vec[0].astype(np.float32)

        palace = AeonPalace(dim=384, embed_fn=_embed)
        return _AeonAdapter(palace)
    except Exception as exc:  # noqa: BLE001
        log.warning("AeonPalace init failed (%s), using stub", exc)

    class _StubAeon:
        def recall(self, query: str, k: int = 3) -> list[dict]:
            raise NotImplementedError("Aeon.recall deferred to V1.1")

        def write(self, episode: dict) -> None:
            raise NotImplementedError("Aeon.write deferred to V1.1")

    return _StubAeon()


class _NegotiatorAdapter:
    """Adapter that maps the orchestrator's ``arbitrate(candidates) ->
    (winner, quality_dict)`` to the real ``Negotiator.negotiate(prompt,
    candidates) -> NegotiationResult``.

    The orchestrator does not currently pass the user prompt into stage 5
    (only the K candidates). The real pipeline only uses ``prompt`` for
    the Catfish module's dissent generation — when absent, Catfish still
    scores agreement/quality off the candidates themselves, which is
    enough for a useful arbitration signal.
    """

    def __init__(self, negotiator: Any) -> None:
        self._neg = negotiator

    async def arbitrate(self, candidates: list[str]) -> tuple[str, dict]:
        # Prompt is not threaded through the orchestrator today; an empty
        # string is acceptable for Catfish (no-dissent default when the
        # local generate_fn is None anyway).
        result = await self._neg.negotiate("", list(candidates))
        quality = {
            "winner_idx": int(result.winner_idx),
            "num_candidates": int(result.num_candidates),
        }
        if result.judge_result is not None:
            quality["agreement"] = float(
                getattr(result.judge_result, "agreement", 0.0)
            )
        if result.catfish_result is not None:
            quality["catfish_triggered"] = bool(
                getattr(result.catfish_result, "triggered", False)
            )
        return result.winner_response, quality


def _build_negotiator(cfg: FullPipelineConfig) -> Any:
    """Construct a real Negotiator (extractor + judge + catfish) with
    ``generate_fn=None`` — each dep falls back to its heuristic path
    (embedding similarity, agreement/quality thresholds) when no LLM
    client is wired. Wraps the result in ``_NegotiatorAdapter`` so the
    orchestrator's contract is preserved. Stub fallback if import fails.
    """
    try:
        from src.cognitive.argument_extractor import ArgumentExtractor
        from src.cognitive.catfish import CatfishModule
        from src.cognitive.judge import AdaptiveJudge
        from src.cognitive.negotiator import Negotiator

        extractor = ArgumentExtractor(generate_fn=None)
        judge = AdaptiveJudge(fast_client=None, deep_client=None)
        catfish = CatfishModule(generate_fn=None)
        return _NegotiatorAdapter(
            Negotiator(extractor=extractor, judge=judge, catfish=catfish)
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Negotiator init failed (%s), using stub", exc)

    class _StubNegotiator:
        async def arbitrate(
            self, candidates: list[str]
        ) -> tuple[str, dict]:
            raise NotImplementedError(
                "Negotiator.arbitrate deferred to V1.1"
            )

    return _StubNegotiator()


class _AntiBiasAdapter:
    """Adapter mapping orchestrator's ``check(text, ctx)`` contract to
    the real ``AntiBiasPipeline.process(prompt, response)``.
    """

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    async def check(
        self, text: str, ctx: dict | None = None
    ) -> tuple[str, dict]:
        ctx = ctx or {}
        prompt_proxy = ""
        recalled = ctx.get("recalled") or []
        if recalled:
            first = recalled[0]
            prompt_proxy = (
                first if isinstance(first, str) else first.get("text", "")
            )
        result = await self._pipeline.process(prompt_proxy, text)
        final = (
            getattr(result, "final_response", None)
            or getattr(result, "response", None)
            or text
        )
        report = {
            "bias_detected": bool(getattr(result, "bias_detected", False)),
            "rewritten": bool(getattr(result, "rewritten", False)),
        }
        detection = getattr(result, "detection", None)
        if detection is not None:
            report["bias_type"] = getattr(detection, "bias_type", "")
            report["confidence"] = float(
                getattr(detection, "confidence", 0.0)
            )
        return final, report


def _build_antibias(cfg: FullPipelineConfig) -> Any:
    """Construct a real AntiBiasPipeline (RBD + heuristic generate_fn).

    Wraps in ``_AntiBiasAdapter`` to match the orchestrator's ``check``
    contract. Stub fallback if construction fails.
    """
    try:
        from src.cognitive.antibias import AntiBiasPipeline, PipelineConfig
        from src.cognitive.rbd import ReasoningBiasDetector

        detector = ReasoningBiasDetector(generate_fn=None)
        pipeline = AntiBiasPipeline(
            detector=detector,
            generate_fn=None,
            config=PipelineConfig(max_retries=cfg.antibias_max_retries),
        )
        return _AntiBiasAdapter(pipeline)
    except Exception as exc:  # noqa: BLE001
        log.warning("AntiBias init failed (%s), using stub", exc)

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

    # Build the registry first so the runtime can be wired with the
    # adapter-swap histogram for per-swap observability.
    reg, metrics = _build_registry()
    state = _State(
        runtime=_build_runtime(cfg, swap_metric=metrics["adapter_swap"]),
        meta_router=_build_meta_router(cfg),
        aeon=_build_aeon(cfg),
        negotiator=_build_negotiator(cfg),
        antibias=_build_antibias(cfg),
        aliases=build_aliases(),
        cfg=cfg,
    )
    app.state.kiki = state
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

            # Opt-in Qwen3 chain-of-thought cleanup. Keep the raw CoT for
            # reasoning-heavy meta intents by default (caller opts in via
            # strip_thinking=True when short, UI-bound output is desired).
            if req.strip_thinking:
                final_text = _strip_think(final_text)

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
