# Plan 3: Meta-Router (32 Sigmoid) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a meta-router that selects which of 32 frozen Brainstacks to activate per prompt, using outcome discovery to generate training targets and sigmoid gating for multi-label activation.

**Architecture:** The router extracts a weighted blend of mid-layer and last-layer hidden states from the frozen Qwen3.5-4B base, projects through a learned attention mechanism with 32 domain query vectors, and outputs 32 independent sigmoid scores. Training targets are discovered by measuring per-stack loss reduction on a mixed dataset (32 forward passes per prompt), then combined 80/20 with prior domain labels.

**Tech Stack:** PyTorch (MPS backend for Mac M3 Ultra), safetensors for weight I/O, Qwen3.5-4B via transformers, frozen stacks from `output/micro-kiki/stacks/`

**Prerequisites:** Plan 2 completed -- 32 frozen stacks in `output/micro-kiki/stacks/`, base model in `models/Qwen3.5-4B`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `scripts/micro_kiki/__init__.py` | Package init |
| `scripts/micro_kiki/constants.py` | Shared constants (domain names, thresholds, dimensions) |
| `scripts/micro_kiki/meta_router.py` | Router nn.Module: Linear projection + global attention + 32 cross-attention + MLP fusion + 32 sigmoid |
| `scripts/micro_kiki/hidden_extractor.py` | Extract mid-layer + last-layer hidden states from base model |
| `scripts/micro_kiki/outcome_discovery.py` | Generate router training targets: base-only loss, per-stack loss, greedy search |
| `scripts/micro_kiki/router_dataset.py` | Dataset class for router training (hidden states + target vectors) |
| `scripts/micro_kiki/train_router.py` | Training loop: BCE loss, cosine LR, 8 epochs, confidence margin penalty |
| `scripts/micro_kiki/stack_manager.py` | Disk-offloaded stack loading with LRU cache |
| `scripts/micro_kiki/inference.py` | Full inference pipeline: tokenize -> extract hidden -> route -> load stacks -> forward |
| `scripts/micro_kiki/validate.py` | Validation suite: latency, accuracy, param count, chat floor |
| `configs/micro-kiki-router.yaml` | Router training hyperparameters |
| `tests/micro_kiki/__init__.py` | Test package init |
| `tests/micro_kiki/test_meta_router.py` | Test router forward, output shapes, sigmoid range, gradient flow |
| `tests/micro_kiki/test_hidden_extractor.py` | Test hidden state extraction shapes and blending |
| `tests/micro_kiki/test_outcome_discovery.py` | Test target generation logic with mock model |
| `tests/micro_kiki/test_router_dataset.py` | Test dataset loading and batching |
| `tests/micro_kiki/test_train_router.py` | Test training step, loss decrease, checkpoint save/load |
| `tests/micro_kiki/test_stack_manager.py` | Test LRU cache behavior, load/unload |
| `tests/micro_kiki/test_inference.py` | Test end-to-end inference pipeline with mocks |
| `output/micro-kiki/router/` | Saved router weights (router.safetensors) |

---

## Task 1: Router Model Architecture

Build the MetaRouter nn.Module, constants, config, and hidden state extractor. This is the foundational model that takes a blended hidden state (3072-dim) and outputs 32 independent sigmoid scores.

**Files:**
- Create: `scripts/micro_kiki/__init__.py`
- Create: `scripts/micro_kiki/constants.py`
- Create: `scripts/micro_kiki/meta_router.py`
- Create: `scripts/micro_kiki/hidden_extractor.py`
- Create: `configs/micro-kiki-router.yaml`
- Create: `tests/micro_kiki/__init__.py`
- Create: `tests/micro_kiki/test_meta_router.py`
- Create: `tests/micro_kiki/test_hidden_extractor.py`

- [ ] **Step 1: Write the failing tests for constants and router**

```python
# tests/micro_kiki/__init__.py
```

```python
# tests/micro_kiki/test_meta_router.py
"""Tests for the MetaRouter nn.Module and constants."""
import pytest
import torch


# --- Constants tests ---

def test_domain_names_length():
    from scripts.micro_kiki.constants import DOMAIN_NAMES
    assert len(DOMAIN_NAMES) == 32


def test_domain_names_are_strings():
    from scripts.micro_kiki.constants import DOMAIN_NAMES
    for name in DOMAIN_NAMES:
        assert isinstance(name, str)
        assert len(name) > 0


def test_domain_names_unique():
    from scripts.micro_kiki.constants import DOMAIN_NAMES
    assert len(set(DOMAIN_NAMES)) == 32


def test_chat_fr_is_index_zero():
    from scripts.micro_kiki.constants import DOMAIN_NAMES
    assert DOMAIN_NAMES[0] == "chat-fr"


def test_hidden_dim():
    from scripts.micro_kiki.constants import H_DIM
    assert H_DIM == 3072


def test_router_hidden_dim():
    from scripts.micro_kiki.constants import ROUTER_HIDDEN_DIM
    assert ROUTER_HIDDEN_DIM == 512


def test_num_domains():
    from scripts.micro_kiki.constants import NUM_DOMAINS
    assert NUM_DOMAINS == 32


def test_gate_threshold():
    from scripts.micro_kiki.constants import GATE_THRESHOLD
    assert GATE_THRESHOLD == 0.12


def test_chat_floor():
    from scripts.micro_kiki.constants import CHAT_FLOOR
    assert CHAT_FLOOR == 0.20


def test_max_active_stacks():
    from scripts.micro_kiki.constants import MAX_ACTIVE_STACKS
    assert MAX_ACTIVE_STACKS == 4


def test_mid_layer_weight():
    from scripts.micro_kiki.constants import MID_LAYER_WEIGHT, LAST_LAYER_WEIGHT
    assert abs(MID_LAYER_WEIGHT - 0.45) < 1e-6
    assert abs(LAST_LAYER_WEIGHT - 0.55) < 1e-6
    assert abs(MID_LAYER_WEIGHT + LAST_LAYER_WEIGHT - 1.0) < 1e-6


def test_loss_improvement_threshold():
    from scripts.micro_kiki.constants import LOSS_IMPROVEMENT_THRESHOLD
    assert LOSS_IMPROVEMENT_THRESHOLD == 0.01


def test_discovery_prior_blend():
    from scripts.micro_kiki.constants import DISCOVERY_WEIGHT, PRIOR_WEIGHT
    assert abs(DISCOVERY_WEIGHT - 0.80) < 1e-6
    assert abs(PRIOR_WEIGHT - 0.20) < 1e-6


# --- MetaRouter tests ---

@pytest.fixture
def router():
    from scripts.micro_kiki.meta_router import MetaRouter
    return MetaRouter(
        h_dim=3072,
        hidden_dim=512,
        num_domains=32,
        dropout=0.1,
        temperature_init=1.0,
    )


@pytest.fixture
def dummy_hidden():
    """Batch of 4 sequences, each a single pooled vector of dim 3072."""
    return torch.randn(4, 3072)


class TestMetaRouterShape:
    def test_output_shape(self, router, dummy_hidden):
        scores = router(dummy_hidden)
        assert scores.shape == (4, 32)

    def test_output_range_sigmoid(self, router, dummy_hidden):
        scores = router(dummy_hidden)
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_single_sample(self, router):
        h = torch.randn(1, 3072)
        scores = router(h)
        assert scores.shape == (1, 32)

    def test_large_batch(self, router):
        h = torch.randn(64, 3072)
        scores = router(h)
        assert scores.shape == (64, 32)


class TestMetaRouterGradients:
    def test_gradients_flow(self, router, dummy_hidden):
        dummy_hidden.requires_grad_(True)
        scores = router(dummy_hidden)
        loss = scores.sum()
        loss.backward()
        assert dummy_hidden.grad is not None
        assert dummy_hidden.grad.shape == (4, 3072)

    def test_all_parameters_have_gradients(self, router, dummy_hidden):
        scores = router(dummy_hidden)
        loss = scores.sum()
        loss.backward()
        for name, param in router.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestMetaRouterComponents:
    def test_temperature_is_learnable(self, router):
        assert hasattr(router, "temperature")
        assert router.temperature.requires_grad

    def test_temperature_positive(self, router):
        temp = router.get_temperature()
        assert temp.item() > 0

    def test_domain_queries_shape(self, router):
        assert router.domain_queries.shape == (32, 512)

    def test_global_query_shape(self, router):
        assert router.global_query.shape == (1, 512)

    def test_projection_reduces_dim(self, router):
        h = torch.randn(2, 3072)
        projected = router.input_proj(h)
        assert projected.shape == (2, 512)


class TestMetaRouterDeterminism:
    def test_eval_mode_deterministic(self, router):
        router.eval()
        h = torch.randn(2, 3072)
        with torch.no_grad():
            s1 = router(h)
            s2 = router(h)
        assert torch.allclose(s1, s2)


class TestMetaRouterParamCount:
    def test_under_2m_params(self, router):
        total = sum(p.numel() for p in router.parameters())
        assert total < 2_000_000, f"Router has {total:,} params, expected < 2M"

    def test_over_500k_params(self, router):
        total = sum(p.numel() for p in router.parameters())
        assert total > 500_000, f"Router has {total:,} params, expected > 500K"


class TestGetActiveStacks:
    def test_max_active_respected(self, router):
        scores = torch.ones(1, 32)  # All high
        active = router.get_active_stacks(scores, max_active=4)
        assert len(active[0]) <= 4

    def test_chat_floor_applied(self, router):
        scores = torch.zeros(1, 32)  # All zero
        active = router.get_active_stacks(
            scores, gate_threshold=0.12, chat_floor=0.20, max_active=4
        )
        domain_indices = {idx for idx, _ in active[0]}
        assert 0 in domain_indices  # chat-fr always present via floor

    def test_below_threshold_excluded(self, router):
        scores = torch.full((1, 32), 0.05)  # All below threshold
        scores[0, 0] = 0.01  # Even chat-fr raw is below
        active = router.get_active_stacks(
            scores, gate_threshold=0.12, chat_floor=0.20, max_active=4
        )
        # Only chat-fr (via floor) should be active
        assert len(active[0]) == 1
        assert active[0][0][0] == 0
```

```python
# tests/micro_kiki/test_hidden_extractor.py
"""Tests for hidden state extraction from base model."""
import pytest
import torch
import torch.nn as nn


class MockQwenModel(nn.Module):
    """Minimal mock of Qwen3.5-4B for hidden state extraction."""

    def __init__(self, num_layers: int = 40, h_dim: int = 3072):
        super().__init__()
        self.config = type("Config", (), {
            "num_hidden_layers": num_layers,
            "hidden_size": h_dim,
        })()
        self.num_layers = num_layers
        self.h_dim = h_dim

    def forward(self, input_ids, output_hidden_states=True, **kwargs):
        batch_size, seq_len = input_ids.shape
        hidden_states = tuple(
            torch.randn(batch_size, seq_len, self.h_dim)
            for _ in range(self.num_layers + 1)
        )
        return type("Output", (), {
            "hidden_states": hidden_states,
            "logits": torch.randn(batch_size, seq_len, 32000),
        })()


@pytest.fixture
def mock_model():
    return MockQwenModel(num_layers=40, h_dim=3072)


class TestBlendHiddenStates:
    def test_output_shape(self, mock_model):
        from scripts.micro_kiki.hidden_extractor import extract_blended_hidden
        input_ids = torch.randint(0, 32000, (2, 64))
        blended = extract_blended_hidden(mock_model, input_ids)
        assert blended.shape == (2, 3072)

    def test_single_sample(self, mock_model):
        from scripts.micro_kiki.hidden_extractor import extract_blended_hidden
        input_ids = torch.randint(0, 32000, (1, 32))
        blended = extract_blended_hidden(mock_model, input_ids)
        assert blended.shape == (1, 3072)

    def test_custom_weights(self, mock_model):
        from scripts.micro_kiki.hidden_extractor import extract_blended_hidden
        input_ids = torch.randint(0, 32000, (2, 64))
        blended = extract_blended_hidden(
            mock_model, input_ids,
            mid_weight=0.5, last_weight=0.5,
        )
        assert blended.shape == (2, 3072)

    def test_weights_must_sum_to_one(self):
        from scripts.micro_kiki.hidden_extractor import extract_blended_hidden
        model = MockQwenModel()
        input_ids = torch.randint(0, 32000, (1, 32))
        with pytest.raises(ValueError, match="must sum to 1.0"):
            extract_blended_hidden(model, input_ids, mid_weight=0.3, last_weight=0.3)


class TestMidLayerIndex:
    def test_mid_layer_is_half(self, mock_model):
        from scripts.micro_kiki.hidden_extractor import get_mid_layer_index
        assert get_mid_layer_index(mock_model) == 20

    def test_mid_layer_odd_count(self):
        from scripts.micro_kiki.hidden_extractor import get_mid_layer_index
        model = MockQwenModel(num_layers=41)
        assert get_mid_layer_index(model) == 20


class TestPoolingStrategy:
    def test_last_token_pooling(self):
        from scripts.micro_kiki.hidden_extractor import pool_last_token
        hidden = torch.randn(2, 10, 3072)
        pooled = pool_last_token(hidden)
        assert pooled.shape == (2, 3072)
        assert torch.allclose(pooled, hidden[:, -1, :])

    def test_mean_pooling(self):
        from scripts.micro_kiki.hidden_extractor import pool_mean
        hidden = torch.ones(2, 10, 3072)
        pooled = pool_mean(hidden)
        assert pooled.shape == (2, 3072)
        assert torch.allclose(pooled, torch.ones(2, 3072))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/clems/KIKI-Mac_tunner && .venv/bin/python -m pytest tests/micro_kiki/test_meta_router.py tests/micro_kiki/test_hidden_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.micro_kiki'`

- [ ] **Step 3: Create package init files**

```python
# scripts/micro_kiki/__init__.py
"""Micro-KIKI: 32-domain Brainstacks MoE with sigmoid meta-router."""
```

- [ ] **Step 4: Write constants module**

```python
# scripts/micro_kiki/constants.py
"""Shared constants for the Micro-KIKI meta-router pipeline."""

# --- Model dimensions (Qwen3.5-4B) ---
H_DIM: int = 3072
ROUTER_HIDDEN_DIM: int = 512
NUM_DOMAINS: int = 32

# --- Domain registry (curriculum order from spec) ---
DOMAIN_NAMES: list[str] = [
    "chat-fr",       # 0  - Phase 1
    "reasoning",     # 1
    "python",        # 2  - Phase 2
    "typescript",    # 3
    "cpp",           # 4
    "rust",          # 5
    "html-css",      # 6  - Phase 3
    "shell",         # 7
    "sql",           # 8
    "yaml-json",     # 9
    "docker",        # 10
    "kicad-dsl",     # 11
    "spice",         # 12
    "lua-upy",       # 13
    "embedded",      # 14 - Phase 4
    "stm32",         # 15
    "iot",           # 16
    "freecad",       # 17
    "platformio",    # 18
    "power",         # 19
    "emc",           # 20
    "dsp",           # 21
    "spice-sim",     # 22
    "electronics",   # 23
    "kicad-pcb",     # 24
    "web-frontend",  # 25 - Phase 5
    "web-backend",   # 26
    "music-audio",   # 27
    "devops",        # 28
    "llm-orch",      # 29
    "math",          # 30 - Phase 6
    "security",      # 31
]

DOMAIN_TO_INDEX: dict[str, int] = {
    name: idx for idx, name in enumerate(DOMAIN_NAMES)
}

# --- Inference gating ---
GATE_THRESHOLD: float = 0.12
CHAT_FLOOR: float = 0.20
MAX_ACTIVE_STACKS: int = 4

# --- Hidden state blending ---
MID_LAYER_WEIGHT: float = 0.45
LAST_LAYER_WEIGHT: float = 0.55

# --- Outcome discovery ---
LOSS_IMPROVEMENT_THRESHOLD: float = 0.01
DISCOVERY_WEIGHT: float = 0.80
PRIOR_WEIGHT: float = 0.20

# --- Training defaults ---
DEFAULT_NUM_EPOCHS: int = 8
DEFAULT_LEARNING_RATE: float = 3e-4
DEFAULT_DROPOUT: float = 0.1
DEFAULT_TEMPERATURE_INIT: float = 1.0

# --- Paths ---
STACKS_DIR: str = "output/micro-kiki/stacks"
ROUTER_DIR: str = "output/micro-kiki/router"
ROUTER_WEIGHTS_FILE: str = "router.safetensors"
BASE_MODEL_PATH: str = "models/Qwen3.5-4B"
```

- [ ] **Step 5: Write the MetaRouter module**

```python
# scripts/micro_kiki/meta_router.py
"""
Meta-Router for 32-domain Brainstacks MoE.

Architecture:
    Input: blended hidden state (h_dim=3072)
    -> Linear(3072, 512) projection
    -> Global attention (learned query)
    -> 32 x cross-attention (domain query vectors)
    -> MLP fusion (GELU, dropout)
    -> 32 sigmoid outputs with temperature scaling

Total: ~1.5M parameters, <5ms inference on CPU, <2ms on MPS/ANE.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttentionPool(nn.Module):
    """Single-head attention pooling with a learned query vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.scale = math.sqrt(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, dim) -- single vector per sample (already pooled from sequence).
        Returns:
            (batch, dim) -- attention-weighted representation.
        """
        # Treat each sample as a length-1 sequence for the attention mechanism
        x_seq = x.unsqueeze(1)  # (B, 1, D)
        k = self.key_proj(x_seq)  # (B, 1, D)
        v = self.value_proj(x_seq)  # (B, 1, D)
        q = self.query.expand(x.size(0), -1, -1)  # (B, 1, D)

        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, 1, 1)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)  # (B, 1, D)
        return out.squeeze(1)  # (B, D)


class DomainCrossAttention(nn.Module):
    """Cross-attention between domain query vectors and the global representation."""

    def __init__(self, dim: int, num_domains: int) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.domain_queries = nn.Parameter(
            torch.randn(num_domains, dim) * 0.02
        )
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.scale = math.sqrt(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, dim) -- global representation from attention pool.
        Returns:
            (batch, num_domains, dim) -- per-domain attended features.
        """
        batch_size = x.size(0)
        k = self.key_proj(x).unsqueeze(1)  # (B, 1, D)
        v = self.value_proj(x).unsqueeze(1)  # (B, 1, D)

        # Domain queries: (num_domains, D) -> (B, num_domains, D)
        q = self.domain_queries.unsqueeze(0).expand(batch_size, -1, -1)

        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, num_domains, 1)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)  # (B, num_domains, D)
        return out


class MLPFusion(nn.Module):
    """Per-domain MLP that fuses cross-attention output to a scalar gate."""

    def __init__(self, dim: int, num_domains: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // 2)
        self.fc2 = nn.Linear(dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.num_domains = num_domains

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_domains, dim) -- per-domain features.
        Returns:
            (batch, num_domains) -- raw logits (pre-sigmoid).
        """
        h = self.fc1(x)  # (B, num_domains, dim//2)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)  # (B, num_domains, 1)
        return h.squeeze(-1)  # (B, num_domains)


class MetaRouter(nn.Module):
    """
    32-sigmoid meta-router for Brainstacks domain selection.

    Takes a blended hidden state vector (mid + last layer) and outputs
    32 independent sigmoid scores indicating which stacks to activate.

    Args:
        h_dim: Hidden dimension of the base model (3072 for Qwen3.5-4B).
        hidden_dim: Internal projection dimension (512).
        num_domains: Number of domain stacks (32).
        dropout: Dropout rate for MLP fusion (0.1).
        temperature_init: Initial temperature for sigmoid scaling (1.0).
    """

    def __init__(
        self,
        h_dim: int = 3072,
        hidden_dim: int = 512,
        num_domains: int = 32,
        dropout: float = 0.1,
        temperature_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains

        # Stage 1: Project from model hidden dim to router hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Stage 2: Global attention pooling
        self.global_attn = GlobalAttentionPool(hidden_dim)

        # Stage 3: Domain cross-attention
        self.domain_cross_attn = DomainCrossAttention(hidden_dim, num_domains)

        # Stage 4: MLP fusion -> logits
        self.mlp_fusion = MLPFusion(hidden_dim, num_domains, dropout)

        # Learnable temperature (stored as log for positivity)
        self.temperature = nn.Parameter(
            torch.tensor(math.log(temperature_init))
        )

        # Expose domain_queries at top level for easy access in tests/inspection
        self.domain_queries = self.domain_cross_attn.domain_queries

        # Expose global query for inspection
        self.global_query = self.global_attn.query.squeeze(0)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers, small normal for queries."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_temperature(self) -> torch.Tensor:
        """Return the positive temperature value."""
        return self.temperature.exp()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (batch, h_dim) -- blended mid+last hidden state.
        Returns:
            (batch, num_domains) -- sigmoid scores in [0, 1].
        """
        # Project to router dim
        projected = self.input_proj(hidden)  # (B, hidden_dim)

        # Global attention pool
        pooled = self.global_attn(projected)  # (B, hidden_dim)

        # Residual connection
        pooled = pooled + projected

        # Domain cross-attention
        domain_features = self.domain_cross_attn(pooled)  # (B, num_domains, hidden_dim)

        # MLP fusion -> logits
        logits = self.mlp_fusion(domain_features)  # (B, num_domains)

        # Temperature-scaled sigmoid
        temp = self.get_temperature()
        scores = torch.sigmoid(logits / temp)

        return scores

    def get_active_stacks(
        self,
        scores: torch.Tensor,
        gate_threshold: float = 0.12,
        chat_floor: float = 0.20,
        max_active: int = 4,
    ) -> list[list[tuple[int, float]]]:
        """
        Apply inference rules to router scores.

        Args:
            scores: (batch, num_domains) sigmoid outputs.
            gate_threshold: Minimum score to consider a stack.
            chat_floor: Minimum score for chat-fr (domain 0).
            max_active: Maximum simultaneous stacks.

        Returns:
            List of lists (one per batch item), each containing
            (domain_index, score) tuples sorted by score descending.
        """
        batch_results: list[list[tuple[int, float]]] = []
        scores_np = scores.detach().cpu()

        for i in range(scores_np.size(0)):
            sample_scores = scores_np[i]

            # Apply chat floor: ensure chat-fr always meets minimum
            effective_scores = sample_scores.clone()
            effective_scores[0] = max(effective_scores[0].item(), chat_floor)

            # Filter by gate threshold
            active: list[tuple[int, float]] = []
            for domain_idx in range(self.num_domains):
                score = effective_scores[domain_idx].item()
                if score >= gate_threshold:
                    active.append((domain_idx, score))

            # Sort by score descending, take top max_active
            active.sort(key=lambda x: x[1], reverse=True)
            active = active[:max_active]

            batch_results.append(active)

        return batch_results
```

- [ ] **Step 6: Write the hidden state extractor**

```python
# scripts/micro_kiki/hidden_extractor.py
"""
Extract and blend mid-layer + last-layer hidden states from Qwen3.5-4B.

The meta-router needs a single vector per prompt that captures both
intermediate reasoning (mid-layer) and final representation (last-layer).
The blending ratio 0.45/0.55 is from the Brainstacks spec.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from scripts.micro_kiki.constants import MID_LAYER_WEIGHT, LAST_LAYER_WEIGHT


def get_mid_layer_index(model: nn.Module) -> int:
    """Return the index of the middle hidden layer."""
    num_layers = model.config.num_hidden_layers
    return num_layers // 2


def pool_last_token(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Pool by taking the last token's hidden state.

    Args:
        hidden_states: (batch, seq_len, h_dim)
    Returns:
        (batch, h_dim)
    """
    return hidden_states[:, -1, :]


def pool_mean(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Pool by averaging across the sequence dimension.

    Args:
        hidden_states: (batch, seq_len, h_dim)
    Returns:
        (batch, h_dim)
    """
    return hidden_states.mean(dim=1)


@torch.no_grad()
def extract_blended_hidden(
    model: nn.Module,
    input_ids: torch.Tensor,
    mid_weight: float = MID_LAYER_WEIGHT,
    last_weight: float = LAST_LAYER_WEIGHT,
    pool_fn: str = "last_token",
) -> torch.Tensor:
    """
    Run a forward pass and blend mid-layer + last-layer hidden states.

    Args:
        model: The base language model (Qwen3.5-4B).
        input_ids: (batch, seq_len) token IDs.
        mid_weight: Weight for mid-layer hidden state (default 0.45).
        last_weight: Weight for last-layer hidden state (default 0.55).
        pool_fn: Pooling strategy -- "last_token" or "mean".

    Returns:
        (batch, h_dim) blended hidden state vector.

    Raises:
        ValueError: If weights don't sum to 1.0.
    """
    if abs(mid_weight + last_weight - 1.0) > 1e-4:
        raise ValueError(
            f"mid_weight ({mid_weight}) + last_weight ({last_weight}) "
            f"must sum to 1.0, got {mid_weight + last_weight:.4f}"
        )

    pooler = pool_last_token if pool_fn == "last_token" else pool_mean

    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    mid_idx = get_mid_layer_index(model)
    mid_hidden = pooler(hidden_states[mid_idx])  # (B, h_dim)
    last_hidden = pooler(hidden_states[-1])       # (B, h_dim)

    blended = mid_weight * mid_hidden + last_weight * last_hidden
    return blended
```

- [ ] **Step 7: Write the config YAML**

```yaml
# configs/micro-kiki-router.yaml
# Meta-router training configuration for 32-domain Brainstacks

model:
  base_model_path: "models/Qwen3.5-4B"
  stacks_dir: "output/micro-kiki/stacks"
  router_dir: "output/micro-kiki/router"
  h_dim: 3072
  router_hidden_dim: 512
  num_domains: 32

hidden_extraction:
  mid_layer_weight: 0.45
  last_layer_weight: 0.55
  # Mid layer index computed as num_layers // 2 at runtime

training:
  num_epochs: 8
  batch_size: 16
  learning_rate: 3.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  lr_scheduler: "cosine"
  dropout: 0.1
  temperature_init: 1.0
  max_seq_len: 512  # For hidden state extraction (router only needs summary)
  gradient_clip: 1.0

outcome_discovery:
  loss_improvement_threshold: 0.01
  discovery_weight: 0.80
  prior_weight: 0.20
  max_prompts: 5000  # Subset of mixed dataset for discovery
  cache_dir: "output/micro-kiki/discovery_cache"

inference:
  gate_threshold: 0.12
  chat_floor: 0.20
  max_active_stacks: 4
  lru_cache_size: 6  # Keep up to 6 stacks in memory

data:
  mixed_dataset_path: "data/micro-kiki/mixed"
  discovery_targets_path: "data/micro-kiki/router_targets.pt"
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /Users/clems/KIKI-Mac_tunner && .venv/bin/python -m pytest tests/micro_kiki/test_meta_router.py tests/micro_kiki/test_hidden_extractor.py -v`
Expected: All tests PASS

- [ ] **Step 9: Commit**

```bash
cd /Users/clems/KIKI-Mac_tunner
git add scripts/micro_kiki/__init__.py scripts/micro_kiki/constants.py scripts/micro_kiki/meta_router.py scripts/micro_kiki/hidden_extractor.py configs/micro-kiki-router.yaml tests/micro_kiki/__init__.py tests/micro_kiki/test_meta_router.py tests/micro_kiki/test_hidden_extractor.py
git commit -m "feat(micro-kiki): add MetaRouter model, constants, hidden extractor, and config"
```

---

## Task 2: Outcome Discovery

Generate training targets for the router by measuring which stacks reduce loss on each prompt. For each prompt: compute base-only loss, then per-stack loss (32 forwards), then greedy-select improving stacks, and blend 80% discovery + 20% prior label.

**Files:**
- Create: `scripts/micro_kiki/outcome_discovery.py`
- Create: `tests/micro_kiki/test_outcome_discovery.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/micro_kiki/test_outcome_discovery.py
"""Tests for outcome discovery target generation."""
import pytest
import torch
import torch.nn as nn


class MockBaseModel(nn.Module):
    """Mock base model that returns predictable loss."""

    def __init__(self, base_loss: float = 2.0):
        super().__init__()
        self.base_loss = base_loss
        self.config = type("Config", (), {
            "num_hidden_layers": 40,
            "hidden_size": 3072,
        })()

    def forward(self, input_ids, labels=None, output_hidden_states=False, **kwargs):
        batch_size, seq_len = input_ids.shape
        loss = torch.tensor(self.base_loss)
        hidden_states = None
        if output_hidden_states:
            hidden_states = tuple(
                torch.randn(batch_size, seq_len, 3072) for _ in range(41)
            )
        return type("Output", (), {
            "loss": loss,
            "logits": torch.randn(batch_size, seq_len, 32000),
            "hidden_states": hidden_states,
        })()


class TestComputeBaseLoss:
    def test_returns_float(self):
        from scripts.micro_kiki.outcome_discovery import compute_loss
        model = MockBaseModel(base_loss=2.5)
        input_ids = torch.randint(0, 100, (1, 32))
        labels = input_ids.clone()
        loss = compute_loss(model, input_ids, labels)
        assert isinstance(loss, float)
        assert abs(loss - 2.5) < 1e-4


class TestGreedyStackSelection:
    def test_selects_improving_stacks(self):
        from scripts.micro_kiki.outcome_discovery import greedy_select_stacks
        base_loss = 2.0
        # Stacks 0, 5, 10 reduce loss by > 0.01; others don't
        stack_losses = [2.0] * 32
        stack_losses[0] = 1.5   # -0.50 improvement
        stack_losses[5] = 1.8   # -0.20 improvement
        stack_losses[10] = 1.98  # -0.02 improvement
        stack_losses[20] = 1.999  # -0.001, below threshold

        selected = greedy_select_stacks(base_loss, stack_losses, threshold=0.01)
        assert 0 in selected
        assert 5 in selected
        assert 10 in selected
        assert 20 not in selected

    def test_returns_empty_when_no_improvement(self):
        from scripts.micro_kiki.outcome_discovery import greedy_select_stacks
        base_loss = 2.0
        stack_losses = [2.005] * 32  # No stack improves
        selected = greedy_select_stacks(base_loss, stack_losses, threshold=0.01)
        assert len(selected) == 0

    def test_all_stacks_helpful(self):
        from scripts.micro_kiki.outcome_discovery import greedy_select_stacks
        base_loss = 2.0
        stack_losses = [1.5] * 32  # All stacks help
        selected = greedy_select_stacks(base_loss, stack_losses, threshold=0.01)
        assert len(selected) == 32


class TestBuildTargetVector:
    def test_shape(self):
        from scripts.micro_kiki.outcome_discovery import build_target_vector
        discovered = {0, 5, 10}
        prior_label = 2  # python domain
        target = build_target_vector(
            discovered_stacks=discovered,
            prior_domain=prior_label,
            num_domains=32,
            discovery_weight=0.8,
            prior_weight=0.2,
        )
        assert target.shape == (32,)

    def test_discovered_stacks_have_high_score(self):
        from scripts.micro_kiki.outcome_discovery import build_target_vector
        discovered = {0, 5}
        target = build_target_vector(
            discovered_stacks=discovered,
            prior_domain=0,
            num_domains=32,
            discovery_weight=0.8,
            prior_weight=0.2,
        )
        # Stack 0 is both discovered and prior -> 0.8 + 0.2 = 1.0
        assert target[0].item() == pytest.approx(1.0, abs=1e-4)
        # Stack 5 is discovered only -> 0.8
        assert target[5].item() == pytest.approx(0.8, abs=1e-4)
        # Stack 3 is neither -> 0.0
        assert target[3].item() == pytest.approx(0.0, abs=1e-4)

    def test_prior_only_stack(self):
        from scripts.micro_kiki.outcome_discovery import build_target_vector
        discovered = {0}
        target = build_target_vector(
            discovered_stacks=discovered,
            prior_domain=5,
            num_domains=32,
            discovery_weight=0.8,
            prior_weight=0.2,
        )
        # Stack 5 is prior only -> 0.2
        assert target[5].item() == pytest.approx(0.2, abs=1e-4)

    def test_values_in_range(self):
        from scripts.micro_kiki.outcome_discovery import build_target_vector
        discovered = {0, 1, 2, 3, 4}
        target = build_target_vector(
            discovered_stacks=discovered,
            prior_domain=0,
            num_domains=32,
            discovery_weight=0.8,
            prior_weight=0.2,
        )
        assert (target >= 0.0).all()
        assert (target <= 1.0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/clems/KIKI-Mac_tunner && .venv/bin/python -m pytest tests/micro_kiki/test_outcome_discovery.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the outcome discovery module**

```python
# scripts/micro_kiki/outcome_discovery.py
"""
Outcome Discovery for Meta-Router Training Targets.

For each prompt in the mixed dataset:
1. Compute base-only loss (no stacks).
2. Compute single-domain loss for each of 32 stacks (32 forwards).
3. Greedy search: select stacks that reduce loss > threshold.
4. Build target vector: 80% discovery + 20% prior label.

This generates the supervision signal for training the meta-router.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from tqdm import tqdm

from scripts.micro_kiki.constants import (
    DISCOVERY_WEIGHT,
    LOSS_IMPROVEMENT_THRESHOLD,
    NUM_DOMAINS,
    PRIOR_WEIGHT,
)

logger = logging.getLogger(__name__)


def compute_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute cross-entropy loss for a single forward pass.

    Args:
        model: Language model (base or with stack applied).
        input_ids: (batch, seq_len) token IDs.
        labels: (batch, seq_len) target token IDs.

    Returns:
        Scalar loss value as float.
    """
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
    return outputs.loss.item()


def greedy_select_stacks(
    base_loss: float,
    stack_losses: list[float],
    threshold: float = LOSS_IMPROVEMENT_THRESHOLD,
) -> set[int]:
    """
    Select stacks that individually reduce loss beyond the threshold.

    The greedy approach: each stack is evaluated independently against
    the base-only loss. A stack is selected if it reduces loss by more
    than `threshold`.

    Args:
        base_loss: Loss with base model only (no stacks).
        stack_losses: Loss with each individual stack applied (length 32).
        threshold: Minimum loss reduction to consider a stack helpful.

    Returns:
        Set of domain indices whose stacks reduce loss sufficiently.
    """
    selected: set[int] = set()
    for domain_idx, stack_loss in enumerate(stack_losses):
        improvement = base_loss - stack_loss
        if improvement > threshold:
            selected.add(domain_idx)
    return selected


def build_target_vector(
    discovered_stacks: set[int],
    prior_domain: int,
    num_domains: int = NUM_DOMAINS,
    discovery_weight: float = DISCOVERY_WEIGHT,
    prior_weight: float = PRIOR_WEIGHT,
) -> torch.Tensor:
    """
    Build the soft target vector combining discovery and prior label.

    Target[i] = discovery_weight * (i in discovered) + prior_weight * (i == prior)

    Args:
        discovered_stacks: Set of domain indices selected by greedy search.
        prior_domain: The domain index from the original dataset label.
        num_domains: Total number of domains (32).
        discovery_weight: Weight for discovered stacks (0.80).
        prior_weight: Weight for prior label (0.20).

    Returns:
        (num_domains,) tensor with soft targets in [0, 1].
    """
    target = torch.zeros(num_domains)

    for idx in discovered_stacks:
        target[idx] += discovery_weight

    target[prior_domain] += prior_weight

    # Clamp to [0, 1] in case discovery + prior overlap
    target = target.clamp(0.0, 1.0)

    return target


def discover_targets_for_prompt(
    base_model: nn.Module,
    load_stack_fn: Callable[[nn.Module, int], nn.Module],
    unload_stack_fn: Callable[[nn.Module, int], nn.Module],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    prior_domain: int,
    threshold: float = LOSS_IMPROVEMENT_THRESHOLD,
) -> torch.Tensor:
    """
    Run the full outcome discovery pipeline for a single prompt.

    Args:
        base_model: The frozen base language model.
        load_stack_fn: Callable(model, domain_idx) -> model with stack loaded.
        unload_stack_fn: Callable(model, domain_idx) -> model with stack removed.
        input_ids: (1, seq_len) token IDs for this prompt.
        labels: (1, seq_len) target token IDs.
        prior_domain: The domain index from the dataset label.
        threshold: Loss improvement threshold.

    Returns:
        (32,) soft target vector for this prompt.
    """
    # Step 1: Base-only loss
    base_loss = compute_loss(base_model, input_ids, labels)

    # Step 2: Per-stack losses (32 forward passes)
    stack_losses: list[float] = []
    for domain_idx in range(NUM_DOMAINS):
        model_with_stack = load_stack_fn(base_model, domain_idx)
        stack_loss = compute_loss(model_with_stack, input_ids, labels)
        stack_losses.append(stack_loss)
        unload_stack_fn(base_model, domain_idx)

    # Step 3: Greedy selection
    discovered = greedy_select_stacks(base_loss, stack_losses, threshold)

    # Step 4: Build target
    target = build_target_vector(discovered, prior_domain)

    return target


def run_outcome_discovery(
    base_model: nn.Module,
    load_stack_fn: Callable[[nn.Module, int], nn.Module],
    unload_stack_fn: Callable[[nn.Module, int], nn.Module],
    dataset: list[dict],
    output_path: str | Path,
    threshold: float = LOSS_IMPROVEMENT_THRESHOLD,
    max_prompts: int = 5000,
    device: str = "cpu",
) -> Path:
    """
    Run outcome discovery on the full dataset and save targets.

    Each entry in `dataset` must have:
        - "input_ids": list[int] -- tokenized prompt
        - "labels": list[int] -- tokenized target
        - "domain": int -- prior domain label index

    Args:
        base_model: Frozen base model.
        load_stack_fn: Load a stack onto the model.
        unload_stack_fn: Remove a stack from the model.
        dataset: List of tokenized examples.
        output_path: Where to save the discovery results.
        threshold: Loss improvement threshold.
        max_prompts: Maximum number of prompts to process.
        device: Torch device string.

    Returns:
        Path to saved targets file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_hidden_states: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    subset = dataset[:max_prompts]
    logger.info("Running outcome discovery on %d prompts...", len(subset))

    for i, example in enumerate(tqdm(subset, desc="Outcome discovery")):
        input_ids = torch.tensor([example["input_ids"]], device=device)
        labels = torch.tensor([example["labels"]], device=device)
        prior_domain = example["domain"]

        target = discover_targets_for_prompt(
            base_model=base_model,
            load_stack_fn=load_stack_fn,
            unload_stack_fn=unload_stack_fn,
            input_ids=input_ids,
            labels=labels,
            prior_domain=prior_domain,
            threshold=threshold,
        )
        all_targets.append(target)

        # Extract hidden state for this prompt (for router training input)
        from scripts.micro_kiki.hidden_extractor import extract_blended_hidden
        hidden = extract_blended_hidden(base_model, input_ids)
        all_hidden_states.append(hidden.squeeze(0).cpu())

        if (i + 1) % 100 == 0:
            logger.info("  Processed %d/%d prompts", i + 1, len(subset))

    # Save as a single .pt file
    result = {
        "hidden_states": torch.stack(all_hidden_states),  # (N, 3072)
        "targets": torch.stack(all_targets),              # (N, 32)
    }
    torch.save(result, output_path)
    logger.info("Saved discovery targets to %s (%d samples)", output_path, len(subset))

    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/clems/KIKI-Mac_tunner && .venv/bin/python -m pytest tests/micro_kiki/test_outcome_discovery.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/clems/KIKI-Mac_tunner
git add scripts/micro_kiki/outcome_discovery.py tests/micro_kiki/test_outcome_discovery.py
git commit -m "feat(micro-kiki): add outcome discovery for router training targets (32 forwards per prompt)"
```

---

## Task 3: Router Training Loop

Build the dataset loader for pre-computed discovery targets and the full training loop with BCE + confidence margin penalty, cosine LR with warmup, checkpointing, and safetensors export.

**Files:**
- Create: `scripts/micro_kiki/router_dataset.py`
- Create: `scripts/micro_kiki/train_router.py`
- Create: `tests/micro_kiki/test_router_dataset.py`
- Create: `tests/micro_kiki/test_train_router.py`

- [ ] **Step 1: Write the failing tests for RouterDataset**

```python
# tests/micro_kiki/test_router_dataset.py
"""Tests for the router training dataset."""
import pytest
import torch
from pathlib import Path


@pytest.fixture
def sample_data(tmp_path):
    """Create a sample discovery targets file."""
    n_samples = 100
    data = {
        "hidden_states": torch.randn(n_samples, 3072),
        "targets": torch.rand(n_samples, 32).clamp(0, 1),
    }
    path = tmp_path / "router_targets.pt"
    torch.save(data, path)
    return path


class TestRouterDataset:
    def test_length(self, sample_data):
        from scripts.micro_kiki.router_dataset import RouterDataset
        ds = RouterDataset(sample_data)
        assert len(ds) == 100

    def test_getitem_shapes(self, sample_data):
        from scripts.micro_kiki.router_dataset import RouterDataset
        ds = RouterDataset(sample_data)
        hidden, target = ds[0]
        assert hidden.shape == (3072,)
        assert target.shape == (32,)

    def test_target_range(self, sample_data):
        from scripts.micro_kiki.router_dataset import RouterDataset
        ds = RouterDataset(sample_data)
        for i in range(min(10, len(ds))):
            _, target = ds[i]
            assert (target >= 0.0).all()
            assert (target <= 1.0).all()

    def test_dataloader_batching(self, sample_data):
        from scripts.micro_kiki.router_dataset import RouterDataset
        from torch.utils.data import DataLoader
        ds = RouterDataset(sample_data)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        batch = next(iter(loader))
        assert batch[0].shape == (16, 3072)
        assert batch[1].shape == (16, 32)

    def test_train_val_split(self, sample_data):
        from scripts.micro_kiki.router_dataset import RouterDataset
        ds = RouterDataset(sample_data)
        train_ds, val_ds = ds.split(val_ratio=0.1)
        assert len(train_ds) == 90
        assert len(val_ds) == 10
```

- [ ] **Step 2: Write the failing tests for training loop**

```python
# tests/micro_kiki/test_train_router.py
"""Tests for the router training loop."""
import pytest
import torch
from pathlib import Path


@pytest.fixture
def sample_data(tmp_path):
    """Create sample discovery targets for training."""
    n_samples = 200
    data = {
        "hidden_states": torch.randn(n_samples, 3072),
        "targets": torch.rand(n_samples, 32).clamp(0, 1),
    }
    path = tmp_path / "router_targets.pt"
    torch.save(data, path)
    return path


class TestTrainStep:
    def test_loss_is_scalar(self):
        from scripts.micro_kiki.meta_router import MetaRouter
        from scripts.micro_kiki.train_router import train_step
        router = MetaRouter(h_dim=3072, hidden_dim=512, num_domains=32)
        optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
        hidden = torch.randn(8, 3072)
        targets = torch.rand(8, 32).clamp(0, 1)
        loss = train_step(router, optimizer, hidden, targets)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases_over_steps(self):
        from scripts.micro_kiki.meta_router import MetaRouter
        from scripts.micro_kiki.train_router import train_step
        torch.manual_seed(42)
        router = MetaRouter(h_dim=3072, hidden_dim=512, num_domains=32)
        optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
        # Fixed data to ensure loss can decrease
        hidden = torch.randn(32, 3072)
        targets = torch.rand(32, 32).clamp(0, 1)
        losses = []
        for _ in range(20):
            loss = train_step(router, optimizer, hidden, targets)
            losses.append(loss)
        # Loss at end should be lower than at start
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


class TestConfidenceMarginPenalty:
    def test_penalty_is_non_negative(self):
        from scripts.micro_kiki.train_router import confidence_margin_penalty
        scores = torch.rand(4, 32)
        penalty = confidence_margin_penalty(scores)
        assert penalty.item() >= 0

    def test_confident_scores_have_low_penalty(self):
        from scripts.micro_kiki.train_router import confidence_margin_penalty
        # Scores close to 0 or 1 = confident
        confident = torch.tensor([[0.01, 0.99, 0.02, 0.98] * 8])
        uncertain = torch.tensor([[0.45, 0.55, 0.48, 0.52] * 8])
        p_conf = confidence_margin_penalty(confident)
        p_unc = confidence_margin_penalty(uncertain)
        assert p_conf.item() < p_unc.item()


class TestCheckpointing:
    def test_save_and_load(self, tmp_path):
        from scripts.micro_kiki.meta_router import MetaRouter
        from scripts.micro_kiki.train_router import save_checkpoint, load_checkpoint
        router = MetaRouter(h_dim=3072, hidden_dim=512, num_domains=32)
        checkpoint_path = tmp_path / "router_checkpoint.pt"
        save_checkpoint(router, checkpoint_path, epoch=3, loss=0.42)

        router2 = MetaRouter(h_dim=3072, hidden_dim=512, num_domains=32)
        meta = load_checkpoint(router2, checkpoint_path)
        assert meta["epoch"] == 3
        assert abs(meta["loss"] - 0.42) < 1e-4

        # Verify weights match
        h = torch.randn(2, 3072)
        with torch.no_grad():
            s1 = router(h)
            s2 = router2(h)
        assert torch.allclose(s1, s2, atol=1e-5)


class TestTrainRouter:
    def test_full_training_produces_weights(self, sample_data, tmp_path):
        from scripts.micro_kiki.train_router import train_router
        output_dir = tmp_path / "router_output"
        train_router(
            data_path=sample_data,
            output_dir=output_dir,
            num_epochs=2,
            batch_size=32,
            learning_rate=1e-3,
            val_ratio=0.1,
        )
        assert (output_dir / "router.safetensors").exists() or (
            output_dir / "router_final.pt"
        ).exists()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/clems/KIKI-Mac_tunner && .venv/bin/python -m pytest tests/micro_kiki/test_router_dataset.py tests/micro_kiki/test_train_router.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write the RouterDataset**

```python
# scripts/micro_kiki/router_dataset.py
"""
Dataset for training the meta-router.

Loads pre-computed hidden states and target vectors from outcome discovery.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset


class RouterDataset(Dataset):
    """
    Dataset of (hidden_state, target_vector) pairs for router training.

    The data file is a .pt dict with:
        - "hidden_states": (N, 3072) float tensor
        - "targets": (N, 32) float tensor in [0, 1]
    """

    def __init__(self, data_path: str | Path) -> None:
        data = torch.load(data_path, weights_only=True)
        self.hidden_states: torch.Tensor = data["hidden_states"].float()
        self.targets: torch.Tensor = data["targets"].float()
        assert self.hidden_states.size(0) == self.targets.size(0)

    def __len__(self) -> int:
        return self.hidden_states.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hidden_states[idx], self.targets[idx]

    def split(
        self, val_ratio: float = 0.1
    ) -> tuple["RouterDataset", "RouterDataset"]:
        """
        Split into train/val subsets deterministically.

        Returns Subset wrappers, not new RouterDataset instances,
        to avoid data duplication.
        """
        n = len(self)
        val_size = int(n * val_ratio)
        train_size = n - val_size
        indices = list(range(n))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        return Subset(self, train_indices), Subset(self, val_indices)
```

- [ ] **Step 5: Write the training module**

```python
# scripts/micro_kiki/train_router.py
"""
Training loop for the 32-sigmoid meta-router.

Loss: BCE + confidence margin penalty.
Schedule: Cosine LR with warmup.
Duration: 8 epochs on ~5K discovery samples.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from torch.utils.data import DataLoader

from scripts.micro_kiki.constants import (
    DEFAULT_DROPOUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_TEMPERATURE_INIT,
    H_DIM,
    NUM_DOMAINS,
    ROUTER_HIDDEN_DIM,
)
from scripts.micro_kiki.meta_router import MetaRouter
from scripts.micro_kiki.router_dataset import RouterDataset

logger = logging.getLogger(__name__)


def confidence_margin_penalty(scores: torch.Tensor) -> torch.Tensor:
    """
    Penalize uncertain sigmoid outputs (close to 0.5).

    Encourages the router to be decisive: push scores toward 0 or 1.
    Penalty = mean of (0.25 - (score - 0.5)^2), which is maximized at 0.5.

    Args:
        scores: (batch, num_domains) sigmoid outputs in [0, 1].

    Returns:
        Scalar penalty value (non-negative).
    """
    deviation_sq = (scores - 0.5) ** 2
    # Max deviation_sq is 0.25 (at scores 0 or 1), min is 0 (at 0.5)
    penalty = (0.25 - deviation_sq).clamp(min=0.0).mean()
    return penalty


def compute_bce_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    margin_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute BCE loss with confidence margin penalty.

    Args:
        scores: (batch, 32) router sigmoid outputs.
        targets: (batch, 32) soft targets from outcome discovery.
        margin_weight: Weight for the confidence margin penalty.

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    bce = F.binary_cross_entropy(scores, targets, reduction="mean")
    margin = confidence_margin_penalty(scores)
    total = bce + margin_weight * margin
    metrics = {
        "bce": bce.item(),
        "margin_penalty": margin.item(),
        "total_loss": total.item(),
    }
    return total, metrics


def train_step(
    router: MetaRouter,
    optimizer: torch.optim.Optimizer,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    margin_weight: float = 0.1,
) -> float:
    """
    Execute one training step.

    Args:
        router: The meta-router model.
        optimizer: Optimizer instance.
        hidden: (batch, h_dim) blended hidden states.
        targets: (batch, num_domains) soft targets.
        margin_weight: Confidence margin penalty weight.

    Returns:
        Total loss value as float.
    """
    router.train()
    optimizer.zero_grad()
    scores = router(hidden)
    loss, _ = compute_bce_loss(scores, targets, margin_weight)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def save_checkpoint(
    router: MetaRouter,
    path: str | Path,
    epoch: int,
    loss: float,
) -> None:
    """Save router checkpoint with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": router.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(
    router: MetaRouter,
    path: str | Path,
) -> dict:
    """Load router checkpoint and return metadata."""
    data = torch.load(path, weights_only=False)
    router.load_state_dict(data["model_state_dict"])
    return {"epoch": data["epoch"], "loss": data["loss"]}


def save_safetensors(router: MetaRouter, output_dir: Path) -> Path:
    """Save router weights in safetensors format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "router.safetensors"
    state_dict = {k: v.contiguous() for k, v in router.state_dict().items()}
    save_file(state_dict, path)
    return path


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_router(
    data_path: str | Path,
    output_dir: str | Path,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = 16,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    margin_weight: float = 0.1,
    val_ratio: float = 0.1,
    h_dim: int = H_DIM,
    hidden_dim: int = ROUTER_HIDDEN_DIM,
    num_domains: int = NUM_DOMAINS,
    dropout: float = DEFAULT_DROPOUT,
    temperature_init: float = DEFAULT_TEMPERATURE_INIT,
    device: str = "cpu",
) -> Path:
    """
    Train the meta-router end-to-end.

    Args:
        data_path: Path to discovery targets .pt file.
        output_dir: Directory for checkpoints and final weights.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of steps for LR warmup.
        margin_weight: Weight for confidence margin penalty.
        val_ratio: Fraction of data for validation.
        h_dim: Base model hidden dimension.
        hidden_dim: Router internal dimension.
        num_domains: Number of domain stacks.
        dropout: Dropout rate.
        temperature_init: Initial sigmoid temperature.
        device: Torch device.

    Returns:
        Path to the saved router.safetensors.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = RouterDataset(data_path)
    train_ds, val_ds = dataset.split(val_ratio)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    logger.info(
        "Training data: %d train, %d val", len(train_ds), len(val_ds)
    )

    # Initialize router
    router = MetaRouter(
        h_dim=h_dim,
        hidden_dim=hidden_dim,
        num_domains=num_domains,
        dropout=dropout,
        temperature_init=temperature_init,
    ).to(device)

    total_params = sum(p.numel() for p in router.parameters())
    logger.info("Router parameters: %s", f"{total_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    # Training loop
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        router.train()
        epoch_loss = 0.0
        num_batches = 0

        for hidden, targets in train_loader:
            hidden = hidden.to(device)
            targets = targets.to(device)
            loss = train_step(router, optimizer, hidden, targets, margin_weight)
            scheduler.step()
            epoch_loss += loss
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Validate
        router.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for hidden, targets in val_loader:
                hidden = hidden.to(device)
                targets = targets.to(device)
                scores = router(hidden)
                loss, _ = compute_bce_loss(scores, targets, margin_weight)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        elapsed = time.time() - start_time

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | time=%.1fs",
            epoch + 1,
            num_epochs,
            avg_train_loss,
            avg_val_loss,
            elapsed,
        )

        # Checkpoint if best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                router,
                output_dir / "router_best.pt",
                epoch=epoch,
                loss=avg_val_loss,
            )

    # Save final weights in safetensors
    final_path = save_safetensors(router, output_dir)
    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
    logger.info("Saved final router to %s", final_path)

    # Also save the best checkpoint as safetensors
    best_ckpt = output_dir / "router_best.pt"
    if best_ckpt.exists():
        best_router = MetaRouter(
            h_dim=h_dim,
            hidden_dim=hidden_dim,
            num_domains=num_domains,
            dropout=dropout,
            temperature_init=temperature_init,
        )
        load_checkpoint(best_router, best_ckpt)
        save_file(
            {k: v.contiguous() for k, v in best_router.state_dict().items()},
            output_dir / "router_best.safetensors",
        )

    return final_path
```

- [ ] **Step 6: Run all training tests**

Run: `cd /Users/clems/KIKI-Mac_tunner && .venv/bin/python -m pytest tests/micro_kiki/test_router_dataset.py tests/micro_kiki/test_train_router.py -v`
Expected: All 10 tests PASS

- [ ] **Step 7: Commit**

```bash
cd /Users/clems/KIKI-Mac_tunner
git add scripts/micro_kiki/router_dataset.py scripts/micro_kiki/train_router.py tests/micro_kiki/test_router_dataset.py tests/micro_kiki/test_train_router.py
git commit -m "feat(micro-kiki): add router dataset, BCE+margin training loop, cosine LR schedule"
```

---

## Task 4: Inference Pipeline

Build the full end-to-end inference pipeline that tokenizes a prompt, extracts hidden states, runs the router, loads active stacks, applies them, and generates a response.

**Files:**
- Create: `scripts/micro_kiki/inference.py`
- Create: `tests/micro_kiki/test_inference.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/micro_kiki/test_inference.py
"""Tests for the full inference pipeline."""
import pytest
import time
import torch
import torch.nn as nn
from pathlib import Path


class MockTokenizer:
    """Minimal tokenizer mock."""

    def __init__(self):
        self.eos_token_id = 2

    def encode(self, text, return_tensors="pt"):
        ids = torch.randint(100, 32000, (1, len(text.split()) + 5))
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "Mock generated response."

    def __call__(self, text, return_tensors="pt", **kwargs):
        return {"input_ids": self.encode(text)}


class MockBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)  # So it has parameters
        self.config = type("Config", (), {
            "num_hidden_layers": 40,
            "hidden_size": 3072,
        })()

    def forward(self, input_ids, output_hidden_states=False, **kwargs):
        batch_size, seq_len = input_ids.shape
        hidden_states = None
        if output_hidden_states:
            hidden_states = tuple(
                torch.randn(batch_size, seq_len, 3072) for _ in range(41)
            )
        return type("Output", (), {
            "hidden_states": hidden_states,
            "logits": torch.randn(batch_size, seq_len, 32000),
        })()


@pytest.fixture
def router_weights(tmp_path):
    from scripts.micro_kiki.meta_router import MetaRouter
    router = MetaRouter(h_dim=3072, hidden_dim=512, num_domains=32)
    path = tmp_path / "router.safetensors"
    from safetensors.torch import save_file
    state_dict = {k: v.contiguous() for k, v in router.state_dict().items()}
    save_file(state_dict, path)
    return path


class TestInferencePipeline:
    def test_route_returns_active_stacks(self, router_weights):
        from scripts.micro_kiki.inference import InferencePipeline
        from scripts.micro_kiki.meta_router import MetaRouter
        from safetensors.torch import load_file

        router = MetaRouter(h_dim=3072, hidden_dim=512, num_domains=32)
        state_dict = load_file(str(router_weights))
        router.load_state_dict(state_dict)

        hidden = torch.randn(1, 3072)
        pipeline = InferencePipeline.__new__(InferencePipeline)
        pipeline.router = router
        pipeline.gate_threshold = 0.12
        pipeline.chat_floor = 0.20
        pipeline.max_active = 4

        active = pipeline.route(hidden)
        # Should return list of (domain_idx, score) tuples
        assert isinstance(active, list)
        assert len(active) <= 4
        for item in active:
            assert len(item) == 2
            domain_idx, score = item
            assert 0 <= domain_idx < 32
            assert score >= 0.12 or domain_idx == 0

    def test_chat_floor_always_active(self, router_weights):
        from scripts.micro_kiki.inference import InferencePipeline
        from scripts.micro_kiki.meta_router import MetaRouter
        from safetensors.torch import load_file

        router = MetaRouter(h_dim=3072, hidden_dim=512, num_domains=32)
        state_dict = load_file(str(router_weights))
        router.load_state_dict(state_dict)

        pipeline = InferencePipeline.__new__(InferencePipeline)
        pipeline.router = router
        pipeline.gate_threshold = 0.12
        pipeline.chat_floor = 0.20
        pipeline.max_active = 4

        # Run many times -- chat-fr (idx 0) should always appear
        chat_present_count = 0
        for _ in range(20):
            hidden = torch.randn(1, 3072)
            active = pipeline.route(hidden)
            domain_indices = [idx for idx, _ in active]
            if 0 in domain_indices:
                chat_present_count += 1
        # Chat floor 0.20 > gate 0.12, so should always be present
        assert chat_present_count == 20


class TestRouterLatency:
    def test_routing_under_10ms(self, router_weights):
        """Router inference must complete in < 10ms per the spec."""
        from scripts.micro_kiki.meta_router import MetaRouter
        from safetensors.torch import load_file

        router = MetaRouter(h_dim=3072, hidden_dim=512, num_domains=32)
        state_dict = load_file(str(router_weights))
        router.load_state_dict(state_dict)
        router.eval()

        hidden = torch.randn(1, 3072)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                router(hidden)

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()
                router(hidden)
                elapsed_ms = (time.perf_counter() - start) * 1000
                times.append(elapsed_ms)

        median_ms = sorted(times)[len(times) // 2]
        assert median_ms < 10.0, f"Router latency {median_ms:.2f}ms exceeds 10ms target"


class TestMaxActiveStacksEnforced:
    def test_never_more_than_4_stacks(self, router_weights):
        from scripts.micro_kiki.meta_router import MetaRouter
        from safetensors.torch import load_file

        router = MetaRouter(h_dim=3072, hidden_dim=512, num_domains=32)
        state_dict = load_file(str(router_weights))
        router.load_state_dict(state_dict)
        router.eval()

        for _ in range(50):
            hidden = torch.randn(1, 3072)
            with torch.no_grad():
                scores = router(hidden)
            active = router.get_active_stacks(
                scores, gate_threshold=0.12, chat_floor=0.20, max_active=4
            )
            for batch_active in active:
                assert len(batch_active) <= 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/clems/KIKI-Mac_tunner && .venv/bin/python -m pytest tests/micro_kiki/test_inference.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the inference pipeline**

```python
# scripts/micro_kiki/inference.py
"""
Full inference pipeline: router -> stack selection -> forward.

Flow:
1. Tokenize prompt
2. Run base model forward to extract hidden states
3. Blend mid + last hidden states
4. Router predicts 32 sigmoid scores
5. Apply gating rules (threshold, chat floor, max 4)
6. Load active stacks from disk (LRU cache)
7. Apply stacks to base model
8. Generate response

The router adds < 10ms overhead. Stack swapping takes < 2s from SSD.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from scripts.micro_kiki.constants import (
    CHAT_FLOOR,
    DOMAIN_NAMES,
    GATE_THRESHOLD,
    H_DIM,
    MAX_ACTIVE_STACKS,
    NUM_DOMAINS,
    ROUTER_HIDDEN_DIM,
)
from scripts.micro_kiki.hidden_extractor import extract_blended_hidden
from scripts.micro_kiki.meta_router import MetaRouter
from scripts.micro_kiki.stack_manager import StackManager

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    End-to-end inference with meta-router and dynamic stack loading.

    Usage:
        pipeline = InferencePipeline.from_paths(
            base_model=model,
            tokenizer=tokenizer,
            router_path="output/micro-kiki/router/router.safetensors",
            stacks_dir="output/micro-kiki/stacks",
        )
        response = pipeline.generate("Explique le fonctionnement d'un MOSFET")
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer,
        router: MetaRouter,
        stack_manager: StackManager,
        apply_stacks_fn=None,
        remove_stacks_fn=None,
        gate_threshold: float = GATE_THRESHOLD,
        chat_floor: float = CHAT_FLOOR,
        max_active: int = MAX_ACTIVE_STACKS,
        device: str = "cpu",
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.router = router
        self.stack_manager = stack_manager
        self.apply_stacks_fn = apply_stacks_fn
        self.remove_stacks_fn = remove_stacks_fn
        self.gate_threshold = gate_threshold
        self.chat_floor = chat_floor
        self.max_active = max_active
        self.device = device

    @classmethod
    def from_paths(
        cls,
        base_model: nn.Module,
        tokenizer,
        router_path: str | Path,
        stacks_dir: str | Path,
        apply_stacks_fn=None,
        remove_stacks_fn=None,
        gate_threshold: float = GATE_THRESHOLD,
        chat_floor: float = CHAT_FLOOR,
        max_active: int = MAX_ACTIVE_STACKS,
        cache_size: int = 6,
        device: str = "cpu",
    ) -> "InferencePipeline":
        """
        Create pipeline from file paths.

        Args:
            base_model: Frozen Qwen3.5-4B model.
            tokenizer: HF tokenizer for the base model.
            router_path: Path to router.safetensors.
            stacks_dir: Directory containing stack subdirectories.
            apply_stacks_fn: Callable(model, weights_dict) to apply stack adapters.
            remove_stacks_fn: Callable(model) to remove applied adapters.
            gate_threshold: Minimum score to activate a stack.
            chat_floor: Minimum score for chat-fr domain.
            max_active: Maximum simultaneous active stacks.
            cache_size: LRU cache size for stack manager.
            device: Torch device.

        Returns:
            Configured InferencePipeline.
        """
        # Load router
        router = MetaRouter(
            h_dim=H_DIM,
            hidden_dim=ROUTER_HIDDEN_DIM,
            num_domains=NUM_DOMAINS,
        )
        state_dict = load_file(str(router_path))
        router.load_state_dict(state_dict)
        router.eval()
        router.to(device)

        # Initialize stack manager
        stack_mgr = StackManager(stacks_dir, cache_size=cache_size, device=device)

        return cls(
            base_model=base_model,
            tokenizer=tokenizer,
            router=router,
            stack_manager=stack_mgr,
            apply_stacks_fn=apply_stacks_fn,
            remove_stacks_fn=remove_stacks_fn,
            gate_threshold=gate_threshold,
            chat_floor=chat_floor,
            max_active=max_active,
            device=device,
        )

    def route(self, hidden: torch.Tensor) -> list[tuple[int, float]]:
        """
        Run the router on a blended hidden state and return active stacks.

        Args:
            hidden: (1, h_dim) blended hidden state vector.

        Returns:
            List of (domain_idx, score) tuples, sorted by score descending.
            At most max_active entries. Chat-fr guaranteed if above gate.
        """
        self.router.eval()
        with torch.no_grad():
            scores = self.router(hidden)  # (1, 32)
        active_list = self.router.get_active_stacks(
            scores,
            gate_threshold=self.gate_threshold,
            chat_floor=self.chat_floor,
            max_active=self.max_active,
        )
        return active_list[0]  # Single sample

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict:
        """
        Full generation pipeline.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Dict with keys:
                - "response": Generated text.
                - "active_stacks": List of (domain_name, score).
                - "routing_time_ms": Router inference time.
                - "total_time_ms": Total generation time.
        """
        total_start = time.perf_counter()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # Extract hidden states for routing
        hidden = extract_blended_hidden(self.base_model, input_ids)

        # Route
        route_start = time.perf_counter()
        active = self.route(hidden.to(self.device))
        routing_time_ms = (time.perf_counter() - route_start) * 1000

        # Load active stacks
        active_names = [
            (DOMAIN_NAMES[idx], score) for idx, score in active
        ]
        logger.info(
            "Active stacks: %s (routing: %.1fms)",
            [(n, f"{s:.2f}") for n, s in active_names],
            routing_time_ms,
        )

        # Apply stacks to model
        if self.apply_stacks_fn is not None:
            stack_weights = self.stack_manager.load_active_stacks(active)
            self.apply_stacks_fn(self.base_model, stack_weights)

        # Generate
        with torch.no_grad():
            outputs = self.base_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )

        # Remove stacks after generation
        if self.remove_stacks_fn is not None:
            self.remove_stacks_fn(self.base_model)

        # Decode
        generated_ids = outputs[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        total_time_ms = (time.perf_counter() - total_start) * 1000

        return {
            "response": response,
            "active_stacks": active_names,
            "routing_time_ms": routing_time_ms,
            "total_time_ms": total_time_ms,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/clems/KIKI-Mac_tunner && .venv/bin/python -m pytest tests/micro_kiki/test_inference.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/clems/KIKI-Mac_tunner
git add scripts/micro_kiki/inference.py tests/micro_kiki/test_inference.py
git commit -m "feat(micro-kiki): add full inference pipeline with router + stack application"
```

---

## Task 5: Disk-Offloaded Stack Manager

Build the LRU-cached stack manager that loads adapter weights from disk on demand and evicts the least-recently-used stack when the cache is full. This keeps VRAM bounded to at most 4 stacks simultaneously.

**Files:**
- Create: `scripts/micro_kiki/stack_manager.py`
- Create: `tests/micro_kiki/test_stack_manager.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/micro_kiki/test_stack_manager.py
"""Tests for disk-offloaded stack management with LRU cache."""
import pytest
import torch
from pathlib import Path
from collections import OrderedDict


@pytest.fixture
def mock_stacks_dir(tmp_path):
    """Create mock stack weight files on disk."""
    stacks_dir = tmp_path / "stacks"
    stacks_dir.mkdir()
    domain_names = [
        "chat-fr", "reasoning", "python", "typescript", "cpp",
        "rust", "html-css", "shell", "sql", "yaml-json",
        "docker", "kicad-dsl", "spice", "lua-upy", "embedded",
        "stm32", "iot", "freecad", "platformio", "power",
        "emc", "dsp", "spice-sim", "electronics", "kicad-pcb",
        "web-frontend", "web-backend", "music-audio", "devops", "llm-orch",
        "math", "security",
    ]
    for name in domain_names:
        domain_dir = stacks_dir / name
        domain_dir.mkdir()
        # Create a small mock safetensors-like file
        weights = {"lora_a": torch.randn(16, 3072), "lora_b": torch.randn(3072, 16)}
        torch.save(weights, domain_dir / "adapter.pt")
    return stacks_dir


class TestStackManagerInit:
    def test_discovers_all_stacks(self, mock_stacks_dir):
        from scripts.micro_kiki.stack_manager import StackManager
        mgr = StackManager(mock_stacks_dir, cache_size=4)
        assert mgr.num_available_stacks == 32

    def test_cache_starts_empty(self, mock_stacks_dir):
        from scripts.micro_kiki.stack_manager import StackManager
        mgr = StackManager(mock_stacks_dir, cache_size=4)
        assert mgr.num_loaded == 0


class TestStackManagerLoad:
    def test_load_stack(self, mock_stacks_dir):
        from scripts.micro_kiki.stack_manager import StackManager
        mgr = StackManager(mock_stacks_dir, cache_size=4)
        weights = mgr.load(0)  # chat-fr
        assert weights is not None
        assert "lora_a" in weights
        assert mgr.num_loaded == 1

    def test_load_same_stack_twice_uses_cache(self, mock_stacks_dir):
        from scripts.micro_kiki.stack_manager import StackManager
        mgr = StackManager(mock_stacks_dir, cache_size=4)
        w1 = mgr.load(0)
        w2 = mgr.load(0)
        assert w1 is w2  # Same object from cache
        assert mgr.num_loaded == 1  # Only loaded once

    def test_load_respects_cache_size(self, mock_stacks_dir):
        from scripts.micro_kiki.stack_manager import StackManager
        mgr = StackManager(mock_stacks_dir, cache_size=3)
        mgr.load(0)
        mgr.load(1)
        mgr.load(2)
        assert mgr.num_loaded == 3
        # Loading a 4th should evict the LRU (stack 0)
        mgr.load(3)
        assert mgr.num_loaded == 3
        assert not mgr.is_loaded(0)
        assert mgr.is_loaded(3)

    def test_lru_eviction_order(self, mock_stacks_dir):
        from scripts.micro_kiki.stack_manager import StackManager
        mgr = StackManager(mock_stacks_dir, cache_size=3)
        mgr.load(0)
        mgr.load(1)
        mgr.load(2)
        # Access stack 0 again to make it recently used
        mgr.load(0)
        # Now load stack 3 -- should evict stack 1 (LRU)
        mgr.load(3)
        assert mgr.is_loaded(0)  # Recently used
        assert not mgr.is_loaded(1)  # Evicted (LRU)
        assert mgr.is_loaded(2)
        assert mgr.is_loaded(3)


class TestStackManagerUnload:
    def test_explicit_unload(self, mock_stacks_dir):
        from scripts.micro_kiki.stack_manager import StackManager
        mgr = StackManager(mock_stacks_dir, cache_size=4)
        mgr.load(0)
        assert mgr.num_loaded == 1
        mgr.unload(0)
        assert mgr.num_loaded == 0

    def test_unload_nonloaded_is_noop(self, mock_stacks_dir):
        from scripts.micro_kiki.stack_manager import StackManager
        mgr = StackManager(mock_stacks_dir, cache_size=4)
        mgr.unload(5)  # Should not raise
        assert mgr.num_loaded == 0

    def test_clear_all(self, mock_stacks_dir):
        from scripts.micro_kiki.stack_manager import StackManager
        mgr = StackManager(mock_stacks_dir, cache_size=4)
        mgr.load(0)
        mgr.load(1)
        mgr.load(2)
        mgr.clear()
        assert mgr.num_loaded == 0


class TestStackManagerInvalidIndex:
    def test_load_out_of_range(self, mock_stacks_dir):
        from scripts.micro_kiki.stack_manager import StackManager
        mgr = StackManager(mock_stacks_dir, cache_size=4)
        with pytest.raises(IndexError):
            mgr.load(99)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/clems/KIKI-Mac_tunner && .venv/bin/python -m pytest tests/micro_kiki/test_stack_manager.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the StackManager**

```python
# scripts/micro_kiki/stack_manager.py
"""
Disk-offloaded stack management with LRU cache.

Stacks live on disk as safetensors/pt files. The StackManager loads them
into memory on demand and evicts the least-recently-used stack when the
cache is full. This keeps VRAM usage bounded to max_active_stacks at a time.

Directory structure:
    output/micro-kiki/stacks/
    +-- chat-fr/
    |   +-- adapter.pt (or adapter.safetensors)
    +-- reasoning/
    |   +-- adapter.pt
    +-- ...
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import torch

from scripts.micro_kiki.constants import DOMAIN_NAMES, NUM_DOMAINS

logger = logging.getLogger(__name__)


class StackManager:
    """
    LRU-cached disk-offloaded stack loader.

    Maintains at most `cache_size` stacks in memory simultaneously.
    When the cache is full and a new stack is requested, the least
    recently used stack is evicted.
    """

    def __init__(
        self,
        stacks_dir: str | Path,
        cache_size: int = 6,
        device: str = "cpu",
    ) -> None:
        self.stacks_dir = Path(stacks_dir)
        self.cache_size = cache_size
        self.device = device

        # Ordered dict acts as LRU: most recently accessed at end
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()

        # Discover available stacks
        self._stack_paths: dict[int, Path] = {}
        self._discover_stacks()

    def _discover_stacks(self) -> None:
        """Scan stacks_dir for available domain adapters."""
        for idx, name in enumerate(DOMAIN_NAMES):
            domain_dir = self.stacks_dir / name
            if domain_dir.is_dir():
                # Look for adapter files in priority order
                for filename in ["adapter.safetensors", "adapter.pt"]:
                    adapter_path = domain_dir / filename
                    if adapter_path.exists():
                        self._stack_paths[idx] = adapter_path
                        break
        logger.info(
            "Discovered %d/%d stacks in %s",
            len(self._stack_paths),
            NUM_DOMAINS,
            self.stacks_dir,
        )

    @property
    def num_available_stacks(self) -> int:
        """Number of stacks found on disk."""
        return len(self._stack_paths)

    @property
    def num_loaded(self) -> int:
        """Number of stacks currently in the cache."""
        return len(self._cache)

    def is_loaded(self, domain_idx: int) -> bool:
        """Check if a stack is currently in the cache."""
        return domain_idx in self._cache

    def load(self, domain_idx: int) -> dict[str, torch.Tensor]:
        """
        Load a stack's weights, using cache if available.

        If the stack is already cached, moves it to the end (most recent).
        If not cached and cache is full, evicts the LRU entry first.

        Args:
            domain_idx: Domain index (0-31).

        Returns:
            Dict of weight name -> tensor.

        Raises:
            IndexError: If domain_idx is out of range or not available.
        """
        if domain_idx < 0 or domain_idx >= NUM_DOMAINS:
            raise IndexError(
                f"Domain index {domain_idx} out of range [0, {NUM_DOMAINS})"
            )
        if domain_idx not in self._stack_paths:
            raise IndexError(
                f"Stack for domain {domain_idx} ({DOMAIN_NAMES[domain_idx]}) "
                f"not found in {self.stacks_dir}"
            )

        # Cache hit: move to end (most recently used)
        if domain_idx in self._cache:
            self._cache.move_to_end(domain_idx)
            return self._cache[domain_idx]

        # Cache miss: evict LRU if needed
        while len(self._cache) >= self.cache_size:
            evicted_idx, evicted_weights = self._cache.popitem(last=False)
            # Let tensors be garbage collected
            del evicted_weights
            logger.debug(
                "Evicted stack %d (%s) from cache",
                evicted_idx,
                DOMAIN_NAMES[evicted_idx],
            )

        # Load from disk
        path = self._stack_paths[domain_idx]
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            weights = load_file(str(path), device=self.device)
        else:
            weights = torch.load(path, map_location=self.device, weights_only=True)

        self._cache[domain_idx] = weights
        logger.debug(
            "Loaded stack %d (%s) from %s",
            domain_idx,
            DOMAIN_NAMES[domain_idx],
            path,
        )
        return weights

    def unload(self, domain_idx: int) -> None:
        """Explicitly remove a stack from the cache."""
        if domain_idx in self._cache:
            del self._cache[domain_idx]

    def clear(self) -> None:
        """Remove all stacks from the cache."""
        self._cache.clear()

    def load_active_stacks(
        self,
        active_list: list[tuple[int, float]],
    ) -> dict[int, dict[str, torch.Tensor]]:
        """
        Load all stacks from an active list (from router output).

        Args:
            active_list: List of (domain_idx, score) tuples.

        Returns:
            Dict of domain_idx -> weight dict for all active stacks.
        """
        result: dict[int, dict[str, torch.Tensor]] = {}
        for domain_idx, score in active_list:
            result[domain_idx] = self.load(domain_idx)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/clems/KIKI-Mac_tunner && .venv/bin/python -m pytest tests/micro_kiki/test_stack_manager.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/clems/KIKI-Mac_tunner
git add scripts/micro_kiki/stack_manager.py tests/micro_kiki/test_stack_manager.py
git commit -m "feat(micro-kiki): add LRU-cached disk-offloaded stack manager"
```

---

## Task 6: Validation Suite

Build a comprehensive validation script that checks all spec criteria: parameter count < 2M, latency < 10ms, output range [0,1], max 4 active stacks, chat-fr floor, and routing accuracy on labeled data.

**Files:**
- Create: `scripts/micro_kiki/validate.py`

- [ ] **Step 1: Write the validation script**

```python
# scripts/micro_kiki/validate.py
"""
Validation script for the meta-router.

Checks:
1. Routing accuracy: correct domain selected for known-domain prompts.
2. Latency: router inference < 10ms median.
3. Max stacks: never more than 4 simultaneous.
4. Chat floor: chat-fr always active.
5. Parameter count: < 2M.
6. Output range: all scores in [0, 1].
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file

from scripts.micro_kiki.constants import (
    CHAT_FLOOR,
    DOMAIN_NAMES,
    GATE_THRESHOLD,
    H_DIM,
    MAX_ACTIVE_STACKS,
    NUM_DOMAINS,
    ROUTER_HIDDEN_DIM,
)
from scripts.micro_kiki.meta_router import MetaRouter

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    detail: str
    value: float | None = None
    target: float | None = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def num_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def num_total(self) -> int:
        return len(self.results)

    def summary(self) -> str:
        lines = [f"Validation: {self.num_passed}/{self.num_total} passed"]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            line = f"  [{status}] {r.name}: {r.detail}"
            if r.value is not None and r.target is not None:
                line += f" (value={r.value:.4f}, target={r.target:.4f})"
            lines.append(line)
        return "\n".join(lines)


def validate_parameter_count(router: MetaRouter) -> ValidationResult:
    """Check that router has < 2M parameters."""
    total = sum(p.numel() for p in router.parameters())
    passed = total < 2_000_000
    return ValidationResult(
        name="parameter_count",
        passed=passed,
        detail=f"{total:,} parameters",
        value=float(total),
        target=2_000_000.0,
    )


def validate_output_range(router: MetaRouter, num_samples: int = 100) -> ValidationResult:
    """Check that all outputs are in [0, 1]."""
    router.eval()
    all_valid = True
    with torch.no_grad():
        for _ in range(num_samples):
            h = torch.randn(1, H_DIM)
            scores = router(h)
            if (scores < 0).any() or (scores > 1).any():
                all_valid = False
                break
    return ValidationResult(
        name="output_range",
        passed=all_valid,
        detail="All outputs in [0, 1]" if all_valid else "Found out-of-range outputs",
    )


def validate_latency(
    router: MetaRouter,
    target_ms: float = 10.0,
    num_warmup: int = 10,
    num_runs: int = 200,
) -> ValidationResult:
    """Check that router inference median latency < target_ms."""
    router.eval()
    h = torch.randn(1, H_DIM)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            router(h)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            router(h)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

    times.sort()
    median_ms = times[len(times) // 2]
    p95_ms = times[int(len(times) * 0.95)]

    return ValidationResult(
        name="latency",
        passed=median_ms < target_ms,
        detail=f"median={median_ms:.2f}ms, p95={p95_ms:.2f}ms",
        value=median_ms,
        target=target_ms,
    )


def validate_max_stacks(
    router: MetaRouter, num_samples: int = 200
) -> ValidationResult:
    """Check that get_active_stacks never returns more than MAX_ACTIVE_STACKS."""
    router.eval()
    max_seen = 0
    with torch.no_grad():
        for _ in range(num_samples):
            h = torch.randn(1, H_DIM)
            scores = router(h)
            active = router.get_active_stacks(
                scores,
                gate_threshold=GATE_THRESHOLD,
                chat_floor=CHAT_FLOOR,
                max_active=MAX_ACTIVE_STACKS,
            )
            num_active = len(active[0])
            max_seen = max(max_seen, num_active)

    return ValidationResult(
        name="max_stacks",
        passed=max_seen <= MAX_ACTIVE_STACKS,
        detail=f"max active seen: {max_seen}",
        value=float(max_seen),
        target=float(MAX_ACTIVE_STACKS),
    )


def validate_chat_floor(
    router: MetaRouter, num_samples: int = 100
) -> ValidationResult:
    """Check that chat-fr is always present in active stacks (via floor)."""
    router.eval()
    chat_present = 0
    with torch.no_grad():
        for _ in range(num_samples):
            h = torch.randn(1, H_DIM)
            scores = router(h)
            active = router.get_active_stacks(
                scores,
                gate_threshold=GATE_THRESHOLD,
                chat_floor=CHAT_FLOOR,
                max_active=MAX_ACTIVE_STACKS,
            )
            domain_indices = {idx for idx, _ in active[0]}
            if 0 in domain_indices:
                chat_present += 1

    ratio = chat_present / num_samples
    return ValidationResult(
        name="chat_floor",
        passed=ratio == 1.0,
        detail=f"chat-fr present in {chat_present}/{num_samples} samples ({ratio:.0%})",
        value=ratio,
        target=1.0,
    )


def validate_routing_accuracy(
    router: MetaRouter,
    test_data: list[dict],
) -> ValidationResult:
    """
    Check that the router's top-1 prediction matches the ground truth domain
    for labeled test samples.

    Each entry in test_data must have:
        - "hidden": (h_dim,) tensor
        - "domain": int (ground truth domain index)
    """
    if not test_data:
        return ValidationResult(
            name="routing_accuracy",
            passed=True,
            detail="No test data provided, skipped",
        )

    router.eval()
    correct = 0
    total = len(test_data)

    with torch.no_grad():
        for sample in test_data:
            h = sample["hidden"].unsqueeze(0)
            scores = router(h)
            predicted = scores.argmax(dim=-1).item()
            if predicted == sample["domain"]:
                correct += 1

    accuracy = correct / total
    return ValidationResult(
        name="routing_accuracy",
        passed=accuracy >= 0.7,  # 70% minimum
        detail=f"{correct}/{total} correct ({accuracy:.1%})",
        value=accuracy,
        target=0.7,
    )


def run_validation(
    router_path: str | Path,
    test_data: list[dict] | None = None,
) -> ValidationReport:
    """
    Run the full validation suite on a trained router.

    Args:
        router_path: Path to router.safetensors.
        test_data: Optional labeled test data for accuracy check.

    Returns:
        ValidationReport with all check results.
    """
    # Load router
    router = MetaRouter(
        h_dim=H_DIM,
        hidden_dim=ROUTER_HIDDEN_DIM,
        num_domains=NUM_DOMAINS,
    )
    state_dict = load_file(str(router_path))
    router.load_state_dict(state_dict)
    router.eval()

    report = ValidationReport()
    report.results.append(validate_parameter_count(router))
    report.results.append(validate_output_range(router))
    report.results.append(validate_latency(router))
    report.results.append(validate_max_stacks(router))
    report.results.append(validate_chat_floor(router))

    if test_data is not None:
        report.results.append(validate_routing_accuracy(router, test_data))

    logger.info("\n%s", report.summary())
    return report


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Validate meta-router")
    parser.add_argument(
        "--router-path",
        type=str,
        default="output/micro-kiki/router/router.safetensors",
        help="Path to router weights",
    )
    args = parser.parse_args()

    report = run_validation(args.router_path)
    print(report.summary())
    sys.exit(0 if report.all_passed else 1)
```

- [ ] **Step 2: Run the full test suite**

Run: `cd /Users/clems/KIKI-Mac_tunner && .venv/bin/python -m pytest tests/micro_kiki/ -v --tb=short`
Expected: All tests PASS across all test files

- [ ] **Step 3: Commit**

```bash
cd /Users/clems/KIKI-Mac_tunner
git add scripts/micro_kiki/validate.py
git commit -m "feat(micro-kiki): add validation suite (param count, latency, routing accuracy, chat floor)"
```

- [ ] **Step 4: Run all tests one final time**

Run: `cd /Users/clems/KIKI-Mac_tunner && .venv/bin/python -m pytest tests/micro_kiki/ -v --tb=short 2>&1 | tail -30`
Expected: All tests PASS, 0 failures

- [ ] **Step 5: Final commit with plan**

```bash
cd /Users/clems/KIKI-Mac_tunner
git add docs/plans/2026-04-15-micro-kiki-plan3-meta-router.md
git commit -m "docs(micro-kiki): rewrite Plan 3 meta-router with 6-task structure"
```
