"""unpatch_all_lora must restore base layers after LoRA wrapping."""
from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx.nn")

import mlx.nn as nn  # noqa: E402


def _is_lora_wrapper(m: object) -> bool:
    from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
    from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear

    return isinstance(
        m,
        (LoRALinear, LoRASwitchLinear, LoRAEmbedding, DoRALinear, DoRAEmbedding),
    )


class _ToyBlock(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

    def __call__(self, x):  # pragma: no cover - not exercised
        return self.q_proj(x) + self.k_proj(x)


class _ToyModel(nn.Module):
    def __init__(self, dim: int = 16, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = [_ToyBlock(dim) for _ in range(n_layers)]

    def __call__(self, x):  # pragma: no cover - not exercised
        for layer in self.layers:
            x = layer(x)
        return x


def test_unpatch_restores_linears_after_lora_wrap() -> None:
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from src.serving.mlx_unpatch import unpatch_all_lora

    model = _ToyModel()
    assert not any(_is_lora_wrapper(m) for _, m in model.named_modules())

    linear_to_lora_layers(
        model,
        num_layers=2,
        config={"rank": 4, "scale": 2.0, "dropout": 0.0},
    )
    assert any(_is_lora_wrapper(m) for _, m in model.named_modules())

    unpatch_all_lora(model)
    assert not any(_is_lora_wrapper(m) for _, m in model.named_modules())


def test_unpatch_no_op_on_clean_model() -> None:
    from src.serving.mlx_unpatch import unpatch_all_lora

    model = _ToyModel()
    unpatch_all_lora(model)
    assert not any(_is_lora_wrapper(m) for _, m in model.named_modules())


def test_unpatch_idempotent() -> None:
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from src.serving.mlx_unpatch import unpatch_all_lora

    model = _ToyModel()
    linear_to_lora_layers(
        model,
        num_layers=2,
        config={"rank": 4, "scale": 2.0, "dropout": 0.0},
    )
    unpatch_all_lora(model)
    unpatch_all_lora(model)
    assert not any(_is_lora_wrapper(m) for _, m in model.named_modules())


def test_unpatch_allows_rewrap() -> None:
    """After unpatch, a fresh LoRA wrap must succeed (no stacking errors)."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from src.serving.mlx_unpatch import unpatch_all_lora

    model = _ToyModel()
    linear_to_lora_layers(
        model,
        num_layers=2,
        config={"rank": 4, "scale": 2.0, "dropout": 0.0},
    )
    unpatch_all_lora(model)
    # Second wrap — would raise "Can't convert layer of type LoRALinear"
    # if unpatch had missed anything.
    linear_to_lora_layers(
        model,
        num_layers=2,
        config={"rank": 8, "scale": 4.0, "dropout": 0.0},
    )
    assert any(_is_lora_wrapper(m) for _, m in model.named_modules())
