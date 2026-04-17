from __future__ import annotations

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

# 10 niche domain names (pivot 2026-04-16: 32 → 10 niche + 1 base = 11 outputs)
NICHE_DOMAINS: frozenset[str] = frozenset({
    "kicad-dsl",
    "spice",
    "emc",
    "stm32",
    "embedded",
    "freecad",
    "platformio",
    "power",
    "dsp",
    "electronics",
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

    Requires torch at runtime. Use ``num_domains=11`` (default) for the
    10-niche + base layout, or ``num_domains=35`` for legacy backward compat.
    """

    # Ordered list of niche domain names for 11-output mode (index 0-9).
    # Index 10 is the implicit "base" output (fallback to raw 35B, no adapter).
    _NICHE_DOMAIN_LIST: list[str] = sorted(NICHE_DOMAINS)

    def __init__(
        self,
        input_dim: int = 768,
        num_domains: int = 11,
        num_capabilities: int = 5,
    ) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.num_capabilities = num_capabilities
        total = num_domains + num_capabilities
        self.linear = nn.Linear(input_dim, total)  # type: ignore[name-defined]
        self.sigmoid = nn.Sigmoid()  # type: ignore[name-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        return self.sigmoid(self.linear(x))

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
