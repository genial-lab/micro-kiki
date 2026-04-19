from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# NerveWmlAdvisor is imported lazily inside forward() so the routing
# module stays importable even when the sibling `nerve-wml` repo is not
# installed. Wiring recipe: docs/integration/micro-kiki-wiring.md in
# nerve-wml, gated by NERVE_WML_ENABLED=1 + NERVE_WML_CHECKPOINT_PATH.
_ADVISOR_SINGLETON = None
_ADVISOR_IMPORT_TRIED = False


def _get_nerve_wml_advisor():
    """Lazy singleton — returns None if disabled, unavailable, or erroring."""
    global _ADVISOR_SINGLETON, _ADVISOR_IMPORT_TRIED
    if os.environ.get("NERVE_WML_ENABLED", "0") != "1":
        return None
    if _ADVISOR_IMPORT_TRIED:
        return _ADVISOR_SINGLETON
    _ADVISOR_IMPORT_TRIED = True
    try:
        from bridge.kiki_nerve_advisor import NerveWmlAdvisor
        _ADVISOR_SINGLETON = NerveWmlAdvisor()
    except Exception as exc:  # noqa: BLE001 — never-raise contract
        logger.info("NerveWmlAdvisor unavailable: %s", exc)
        _ADVISOR_SINGLETON = None
    return _ADVISOR_SINGLETON

# ---------------------------------------------------------------------------
# Constants — importable without torch
# ---------------------------------------------------------------------------

CAPABILITY_NAMES = [
    "web_search",
    "self_critique_token",
    "self_critique_response",
    "self_critique_task",
    "deep_eval",
]

# 34 niche domain names (2026-04-17: expanded to 35 outputs = 34 niche + 1 base)
NICHE_DOMAINS: frozenset[str] = frozenset({
    "chat-fr",
    "components",
    "cpp",
    "devops",
    "docker",
    "dsp",
    "electronics",
    "embedded",
    "emc",
    "freecad",
    "html-css",
    "iot",
    "kicad-dsl",
    "kicad-pcb",
    "llm-ops",
    "llm-orch",
    "lua-upy",
    "math",
    "ml-training",
    "music-audio",
    "platformio",
    "power",
    "python",
    "reasoning",
    "rust",
    "security",
    "shell",
    "spice",
    "sql",
    "stm32",
    "typescript",
    "web-backend",
    "web-frontend",
    "yaml-json",
})

# ---------------------------------------------------------------------------
# Optional torch import — MetaRouter requires it; constants do not
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
    _NNModule = nn.Module
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    _NNModule = object  # type: ignore[misc,assignment]


class MetaRouter(_NNModule):  # type: ignore[misc,valid-type]
    """Sigmoid meta-router with domain + capability outputs.

    Requires torch at runtime. Use ``num_domains=35`` (default) for the
    34-niche + base layout.
    """

    # Ordered list of niche domain names (index 0-33).
    # Index 34 is the implicit "base" output (fallback to raw 35B, no adapter).
    _NICHE_DOMAIN_LIST: list[str] = sorted(NICHE_DOMAINS)

    def __init__(
        self,
        input_dim: int = 768,
        num_domains: int = 35,
        num_capabilities: int = 5,
    ) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.num_capabilities = num_capabilities
        total = num_domains + num_capabilities
        self.linear = nn.Linear(input_dim, total)  # type: ignore[name-defined]
        self.sigmoid = nn.Sigmoid()  # type: ignore[name-defined]

    def forward(  # type: ignore[name-defined]
        self,
        x: torch.Tensor,
        query_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        """Sigmoid meta-router forward.

        If NERVE_WML_ENABLED=1 and a NerveWmlAdvisor is available and
        `query_tokens` is provided, the advisor's 35-dim advice is mixed
        into the domain slice of the raw logits BEFORE sigmoid, with
        mixing weight `alpha = float(NERVE_WML_ALPHA or "0.1")`. The
        advisor is never-raising: any failure falls back to the vanilla
        pre-advisor logits.
        """
        logits = self.linear(x)
        advisor = _get_nerve_wml_advisor()
        if advisor is not None and query_tokens is not None:
            try:
                advice = advisor.advise(query_tokens)
                if advice is not None:
                    alpha = float(os.environ.get("NERVE_WML_ALPHA", "0.1"))
                    domain_slice = logits[..., : self.num_domains]
                    advisor_logits = torch.tensor(  # type: ignore[name-defined]
                        [advice.get(i, 0.0) for i in range(self.num_domains)],
                        dtype=domain_slice.dtype,
                        device=domain_slice.device,
                    )
                    logits = logits.clone()
                    logits[..., : self.num_domains] = (
                        (1.0 - alpha) * domain_slice + alpha * advisor_logits
                    )
            except Exception as exc:  # noqa: BLE001 — never-raise contract
                logger.debug("NerveWmlAdvisor.advise skipped: %s", exc)
        return self.sigmoid(logits)

    def get_domains(self, output: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        return output[:, : self.num_domains]

    def get_capabilities(self, output: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        return output[:, self.num_domains :]

    def get_active_domains(
        self,
        output: torch.Tensor,  # type: ignore[name-defined]
        threshold: float = 0.12,
        max_active: int = 4,
    ) -> list[list[int]]:
        domains = self.get_domains(output)
        results = []
        for row in domains:
            mask = row > threshold
            indices = mask.nonzero(as_tuple=True)[0].tolist()
            if len(indices) > max_active:
                scores = row[indices]
                top_k = scores.topk(max_active).indices
                indices = [indices[i] for i in top_k.tolist()]
            results.append(indices)
        return results

    def get_active_domains_named(
        self,
        output: torch.Tensor,  # type: ignore[name-defined]
        threshold: float = 0.12,
        max_active: int = 4,
    ) -> list[str]:
        """Return active niche domain names, or ["base"] when none exceed threshold.

        Only valid when num_domains == 11 (10 niche + 1 base output).
        """
        domains = self.get_domains(output)[0]  # first batch item
        niche_scores = domains[: len(self._NICHE_DOMAIN_LIST)]
        active = [
            self._NICHE_DOMAIN_LIST[i]
            for i, score in enumerate(niche_scores)
            if score.item() > threshold
        ]
        if len(active) > max_active:
            scored = sorted(
                active,
                key=lambda name: niche_scores[self._NICHE_DOMAIN_LIST.index(name)].item(),
                reverse=True,
            )
            active = scored[:max_active]
        return active if active else ["base"]

    def get_active_capabilities(
        self,
        output: torch.Tensor,  # type: ignore[name-defined]
        thresholds: dict[str, float],
    ) -> dict[str, bool]:
        caps = self.get_capabilities(output)[0]
        return {
            name: caps[i].item() > thresholds.get(name, 0.5)
            for i, name in enumerate(CAPABILITY_NAMES)
        }
