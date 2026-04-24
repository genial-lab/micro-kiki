from __future__ import annotations

import os
from typing import TYPE_CHECKING

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

# 35 niche domain names. Aligned with the trained v4 SOTA adapter set
# on Studio (lora-qwen36-35b-v4-sota/) and the `num_domains=35` default
# in MetaRouter. The router v4 safetensors shipped with 34 outputs; an
# updated weight file with 35 outputs is required before the MetaRouter
# meta-mode can rank `spice-sim` — niche-explicit calls work regardless.
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
    "spice-sim",
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


# ---------------------------------------------------------------------------
# Optional NerveWmlAdvisor — env-gated lazy import + singleton
# ---------------------------------------------------------------------------
#
# Contract (tests/routing/test_router_nerve_wml.py):
# - Default (NERVE_WML_ENABLED != "1"): no import attempt, no perf cost,
#   forward() is byte-identical to the pre-advisor baseline.
# - If NERVE_WML_ENABLED=1 but nerve-wml is not installed: advisor resolves
#   to None, forward() still returns vanilla sigmoid logits (never-raise).
# - With an advisor returning a dict {domain_idx: logit}, forward() blends
#   the advice into the domain slice of the raw logits PRE-sigmoid, using
#   alpha from NERVE_WML_ALPHA (default 0.5):
#       raw[..., :num_domains] = (1-alpha) * raw + alpha * advice_tensor
# - If query_tokens is None, the advisor is never called (common path for
#   training batches without query strings).

_ADVISOR_SINGLETON: object | None = None
_ADVISOR_IMPORT_TRIED: bool = False


def _get_nerve_wml_advisor() -> object | None:
    """Lazy import + memoize the nerve-wml advisor; never raises."""
    global _ADVISOR_SINGLETON, _ADVISOR_IMPORT_TRIED
    if _ADVISOR_SINGLETON is not None:
        return _ADVISOR_SINGLETON
    if _ADVISOR_IMPORT_TRIED:
        return None
    _ADVISOR_IMPORT_TRIED = True
    try:
        from nerve_wml.bridge.kiki_nerve_advisor import (  # noqa: PLC0415
            KikiNerveAdvisor,
        )

        _ADVISOR_SINGLETON = KikiNerveAdvisor()
    except Exception:  # noqa: BLE001
        _ADVISOR_SINGLETON = None
    return _ADVISOR_SINGLETON


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

    def forward(
        self,
        x: torch.Tensor,  # type: ignore[name-defined]
        query_tokens: list[int] | None = None,
    ) -> torch.Tensor:  # type: ignore[name-defined]
        raw = self.linear(x)
        if query_tokens is not None and os.environ.get("NERVE_WML_ENABLED") == "1":
            advisor = _get_nerve_wml_advisor()
            if advisor is not None:
                try:
                    advice = advisor.advise(query_tokens)  # type: ignore[attr-defined]
                    alpha = float(os.environ.get("NERVE_WML_ALPHA", "0.5"))
                    advice_tensor = torch.zeros(
                        self.num_domains, dtype=raw.dtype, device=raw.device
                    )
                    for idx, val in advice.items():
                        if 0 <= int(idx) < self.num_domains:
                            advice_tensor[int(idx)] = float(val)
                    raw = raw.clone()
                    raw[..., : self.num_domains] = (
                        (1.0 - alpha) * raw[..., : self.num_domains]
                        + alpha * advice_tensor
                    )
                except Exception:  # noqa: BLE001 — never-raise contract
                    pass
        return self.sigmoid(raw)

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
