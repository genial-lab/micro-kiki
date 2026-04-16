# Brainstacks Training Pipeline — Plan 2 of 4

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Brainstacks architecture (MoE-LoRA + null-space projection + residual boosting) to Qwen3.5-4B and train 32 domain-specialized frozen stacks on Apple Silicon.

**Architecture:** Each domain stack is a MoE-LoRA adapter with 4 experts (rank 16, top-2 routing) applied to 7 linear projections in each Qwen3.5-4B layer. Stacks are trained sequentially in curriculum order. Before training stack N, a null-space projector is computed from the frozen weights of stacks 1..N-1 via randomized SVD (ns_top_k_dirs=32). The training gradient is projected into the null-space so that new learning cannot degrade previous domains. After SFT, 1-2 rounds of residual boosting refine the stack on hard examples. The frozen stack is offloaded to disk.

**Tech Stack:** MLX (bf16, Apple Silicon), mlx-lm (model loading + tokenizer), PyYAML, NumPy (SVD), safetensors

**Prerequisites:** Plan 1 (Data Pipeline) completed. Files `data/micro-kiki/<domain>/train.jsonl` and `data/micro-kiki/<domain>/valid.jsonl` exist for all 32 domains.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `configs/micro_kiki/brainstacks.yaml` | All hyperparameters: model path, LoRA rank, experts, null-space config, curriculum order, training params |
| `scripts/micro_kiki/moe_lora.py` | MoE-LoRA module: 4 expert LoRA adapters + learned top-2 router per linear projection |
| `scripts/micro_kiki/null_space.py` | Randomized SVD null-space projector: collects frozen stack weights, computes projection matrix |
| `scripts/micro_kiki/residual_boost.py` | Residual boosting: identifies hard examples, retrains with boosted weight |
| `scripts/micro_kiki/train_stack.py` | Train a single domain stack: load base, attach MoE-LoRA, project gradients, SFT, boost, freeze, offload |
| `scripts/micro_kiki/train_all_stacks.sh` | Orchestrate sequential training of all 32 stacks with forgetting checks |
| `scripts/micro_kiki/eval_stack.py` | Evaluate a single stack (or all stacks) on domain-specific validation sets, compute forgetting delta |
| `tests/micro_kiki/test_moe_lora.py` | Unit tests for MoE-LoRA forward pass, expert routing, gradient flow |
| `tests/micro_kiki/test_null_space.py` | Unit tests for null-space projector: orthogonality, rank preservation |
| `tests/micro_kiki/test_residual_boost.py` | Unit tests for residual boosting: hard example selection, loss improvement |
| `tests/micro_kiki/test_train_stack.py` | Integration test: train 1 stack on tiny synthetic data, verify freeze + offload |

---

## Curriculum Order (from spec)

```
 1. chat-fr         2. reasoning       3. python          4. typescript
 5. cpp             6. rust            7. html-css        8. shell
 9. sql            10. yaml-json      11. docker         12. kicad-dsl
13. spice          14. lua-upy        15. embedded       16. stm32
17. iot            18. freecad        19. platformio     20. power
21. emc            22. dsp            23. spice-sim      24. electronics
25. kicad-pcb      26. web-frontend   27. web-backend    28. music-audio
29. devops         30. llm-orch       31. math           32. security
```

---

### Task 1: Create brainstacks.yaml configuration

**Files:**
- Create: `configs/micro_kiki/brainstacks.yaml`

- [ ] **Step 1: Write the config file**

```yaml
# Micro_KIKI Brainstacks — 32 MoE-LoRA stacks on Qwen3.5-4B
# Trained sequentially with null-space projection + residual boosting

model:
  path: models/Qwen3.5-4B-BF16
  h_dim: 3072
  num_layers: 36
  # 7 projections per layer to attach MoE-LoRA
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

moe_lora:
  num_experts: 4
  rank: 16
  alpha: 32.0           # alpha/rank = 2
  dropout: 0.01
  top_k: 2              # top-2 expert routing
  router_hidden: 64     # router MLP hidden dim
  use_rs_lora: true     # rsLoRA scaling (rank-stabilized)

null_space:
  ns_top_k_dirs: 32     # directions to preserve per frozen stack
  svd_oversampling: 10  # randomized SVD oversampling factor
  svd_n_iter: 3         # power iterations for randomized SVD
  projection_strength: 1.0  # 1.0 = full null-space projection

residual_boost:
  max_rounds: 2
  min_improvement: 0.002    # skip round if delta loss < this
  hard_example_quantile: 0.75  # top 25% loss examples = "hard"
  boost_weight: 2.0         # loss multiplier for hard examples
  boost_steps: 100           # steps per boost round
  boost_lr_scale: 0.5        # LR multiplied by this during boost

training:
  batch_size: 1
  grad_accumulation_steps: 4
  max_seq_length: 2048
  learning_rate: 2e-4
  warmup_ratio: 0.05
  weight_decay: 0.01
  max_steps: 500           # ~500 steps per domain
  steps_per_eval: 50
  steps_per_save: 100
  val_batches: 10
  seed: 42

forgetting:
  max_delta: 0.03          # abort if any previous domain degrades > 3%
  eval_domains: all        # "all" or "last_5" for faster checks

data:
  base_dir: data/micro-kiki
  # Each domain has train.jsonl + valid.jsonl in data/micro-kiki/<domain>/

output:
  base_dir: output/micro-kiki/stacks
  # Each stack saved as output/micro-kiki/stacks/<domain>/adapters.safetensors

curriculum:
  # Phase 1 — Foundations
  - chat-fr
  - reasoning
  # Phase 2 — Coding core
  - python
  - typescript
  - cpp
  - rust
  # Phase 3 — Coding secondary
  - html-css
  - shell
  - sql
  - yaml-json
  - docker
  - kicad-dsl
  - spice
  - lua-upy
  # Phase 4 — Technical domains
  - embedded
  - stm32
  - iot
  - freecad
  - platformio
  - power
  - emc
  - dsp
  - spice-sim
  - electronics
  - kicad-pcb
  # Phase 5 — Applications
  - web-frontend
  - web-backend
  - music-audio
  - devops
  - llm-orch
  # Phase 6 — Complements
  - math
  - security
```

- [ ] **Step 2: Verify YAML parses correctly**

Run: `cd /Users/clems/KIKI-Mac_tunner && python3 -c "import yaml; c=yaml.safe_load(open('configs/micro_kiki/brainstacks.yaml')); print(f'Curriculum: {len(c[\"curriculum\"])} domains'); assert len(c['curriculum'])==32"`
Expected: `Curriculum: 32 domains`

- [ ] **Step 3: Commit**

```bash
git add configs/micro_kiki/brainstacks.yaml
git commit -m "feat: add brainstacks.yaml config for 32 MoE-LoRA stacks"
```

---

### Task 2: Implement MoE-LoRA module

**Files:**
- Create: `scripts/micro_kiki/moe_lora.py`
- Create: `tests/micro_kiki/test_moe_lora.py`

- [ ] **Step 1: Write the failing tests**

```python
#!/usr/bin/env python3
"""Tests for MoE-LoRA module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import mlx.core as mx
import mlx.nn as nn
from micro_kiki.moe_lora import LoRAExpert, MoELoRALayer, apply_moe_lora


def test_lora_expert_forward_shape():
    """A single LoRA expert produces output with same shape as input."""
    expert = LoRAExpert(in_features=3072, out_features=3072, rank=16, alpha=32.0)
    x = mx.random.normal((1, 128, 3072))
    out = expert(x)
    assert out.shape == (1, 128, 3072), f"Expected (1,128,3072), got {out.shape}"


def test_lora_expert_zero_init():
    """LoRA B matrix is zero-initialized so expert starts as identity."""
    expert = LoRAExpert(in_features=3072, out_features=3072, rank=16, alpha=32.0)
    x = mx.random.normal((1, 4, 3072))
    out = expert(x)
    mx.eval(out)
    assert mx.allclose(out, mx.zeros_like(out), atol=1e-6), "Fresh expert should output zeros"


def test_moe_lora_layer_forward():
    """MoE-LoRA layer routes input through top-k experts."""
    layer = MoELoRALayer(
        in_features=3072, out_features=3072,
        num_experts=4, rank=16, alpha=32.0, top_k=2,
        router_hidden=64,
    )
    x = mx.random.normal((1, 32, 3072))
    out = layer(x)
    assert out.shape == (1, 32, 3072), f"Expected (1,32,3072), got {out.shape}"


def test_moe_lora_layer_router_produces_weights():
    """Router produces non-negative weights that sum to 1 per token."""
    layer = MoELoRALayer(
        in_features=256, out_features=256,
        num_experts=4, rank=8, alpha=16.0, top_k=2,
        router_hidden=32,
    )
    x = mx.random.normal((1, 8, 256))
    weights, indices = layer.route(x)
    mx.eval(weights, indices)
    assert weights.shape == (1, 8, 2), f"Expected (1,8,2), got {weights.shape}"
    assert indices.shape == (1, 8, 2), f"Expected (1,8,2), got {indices.shape}"
    # Weights should be non-negative (softmax output)
    assert mx.all(weights >= 0).item(), "Router weights must be non-negative"


def test_apply_moe_lora_counts_layers():
    """apply_moe_lora attaches MoE-LoRA to all target modules."""
    # Create a minimal mock model with Linear layers
    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(256, 256)
            self.k_proj = nn.Linear(256, 256)
            self.v_proj = nn.Linear(256, 256)
            self.o_proj = nn.Linear(256, 256)

    class FakeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(256, 512)
            self.up_proj = nn.Linear(256, 512)
            self.down_proj = nn.Linear(512, 256)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttention()
            self.mlp = FakeMLP()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = [FakeLayer(), FakeLayer()]

    model = FakeModel()
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    count = apply_moe_lora(
        model, target_modules=target_modules,
        num_experts=4, rank=8, alpha=16.0, top_k=2,
        router_hidden=32,
    )
    # 7 projections x 2 layers = 14
    assert count == 14, f"Expected 14 MoE-LoRA layers, got {count}"


if __name__ == "__main__":
    test_lora_expert_forward_shape()
    print("PASS: test_lora_expert_forward_shape")
    test_lora_expert_zero_init()
    print("PASS: test_lora_expert_zero_init")
    test_moe_lora_layer_forward()
    print("PASS: test_moe_lora_layer_forward")
    test_moe_lora_layer_router_produces_weights()
    print("PASS: test_moe_lora_layer_router_produces_weights")
    test_apply_moe_lora_counts_layers()
    print("PASS: test_apply_moe_lora_counts_layers")
    print("\nAll MoE-LoRA tests passed.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/clems/KIKI-Mac_tunner && python tests/micro_kiki/test_moe_lora.py`
Expected: `ModuleNotFoundError: No module named 'micro_kiki.moe_lora'`

- [ ] **Step 3: Write MoE-LoRA implementation**

```python
#!/usr/bin/env python3
"""MoE-LoRA module for Brainstacks on Qwen3.5-4B.

Each MoE-LoRA layer replaces a single nn.Linear with:
  - N LoRA experts (A_i, B_i) with rank r
  - A learned router MLP that selects top-k experts per token
  - The base weight W is frozen; only LoRA deltas are trainable

Architecture per projection:
  y = W @ x + sum_topk( gate_i * (B_i @ A_i @ x) * scale )

Paper: Brainstacks (arXiv:2604.01152)
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class LoRAExpert(nn.Module):
    """Single LoRA expert: low-rank delta W = B @ A scaled by alpha/rank.

    Uses rsLoRA scaling: scale = alpha / sqrt(rank) instead of alpha / rank
    for rank-stabilized training (Kalajdzievski 2023).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        use_rs_lora: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        # rsLoRA: scale by 1/sqrt(r) instead of 1/r for stable high-rank training
        self.scale = alpha / math.sqrt(rank) if use_rs_lora else alpha / rank
        self.dropout = dropout

        # A: (in_features, rank) — Kaiming uniform init
        self.lora_a = mx.random.normal((in_features, rank)) * (1.0 / math.sqrt(in_features))
        # B: (rank, out_features) — zero init so expert starts as no-op
        self.lora_b = mx.zeros((rank, out_features))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (..., in_features) -> (..., out_features)
        if self.dropout > 0 and self.training:
            mask = mx.random.bernoulli(1.0 - self.dropout, x.shape)
            x = x * mask / (1.0 - self.dropout)
        # (..., in) @ (in, r) -> (..., r)
        h = x @ self.lora_a
        # (..., r) @ (r, out) -> (..., out)
        out = h @ self.lora_b
        return out * self.scale

    def flat_weights(self) -> mx.array:
        """Return concatenated flattened weights for null-space projection."""
        return mx.concatenate([
            mx.reshape(self.lora_a, (-1,)),
            mx.reshape(self.lora_b, (-1,)),
        ])

    @property
    def num_params(self) -> int:
        return self.in_features * self.rank + self.rank * self.out_features


class MoELoRALayer(nn.Module):
    """Mixture-of-Experts LoRA layer.

    Replaces a single frozen linear projection with N LoRA experts
    and a learned top-k router.

    Forward:
      router_logits = Router(x)             # (batch, seq, num_experts)
      topk_weights, topk_indices = topk(router_logits, k)
      topk_weights = softmax(topk_weights)  # normalize over selected experts
      y = sum_i( topk_weights[i] * Expert_i(x) )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 4,
        rank: int = 16,
        alpha: float = 32.0,
        top_k: int = 2,
        dropout: float = 0.0,
        router_hidden: int = 64,
        use_rs_lora: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert pool
        self.experts = [
            LoRAExpert(
                in_features=in_features,
                out_features=out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                use_rs_lora=use_rs_lora,
            )
            for _ in range(num_experts)
        ]

        # Router: small MLP mapping input hidden -> expert logits
        self.router_w1 = nn.Linear(in_features, router_hidden)
        self.router_w2 = nn.Linear(router_hidden, num_experts)

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Compute routing weights and expert indices.

        Args:
            x: (batch, seq, in_features)

        Returns:
            weights: (batch, seq, top_k) — normalized weights
            indices: (batch, seq, top_k) — expert indices
        """
        # Router forward
        h = nn.gelu(self.router_w1(x))           # (B, T, router_hidden)
        logits = self.router_w2(h)                 # (B, T, num_experts)

        # Top-k selection
        top_k_logits, indices = mx.topk(logits, k=self.top_k, axis=-1)
        weights = mx.softmax(top_k_logits, axis=-1)  # normalize over selected
        return weights, indices

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: route input through top-k experts and combine.

        Args:
            x: (batch, seq, in_features)

        Returns:
            delta: (batch, seq, out_features) — additive LoRA delta
        """
        weights, indices = self.route(x)  # (B,T,k), (B,T,k)

        # Compute all expert outputs: list of (B, T, out_features)
        expert_outputs = mx.stack(
            [expert(x) for expert in self.experts], axis=-2
        )  # (B, T, num_experts, out_features)

        # Gather the top-k experts
        batch_size, seq_len, _ = x.shape
        # indices: (B, T, k) — expand for gather
        idx = mx.expand_dims(indices, axis=-1)           # (B, T, k, 1)
        idx = mx.broadcast_to(idx, (batch_size, seq_len, self.top_k, self.out_features))
        # Gather selected expert outputs
        selected = mx.take_along_axis(expert_outputs, idx, axis=2)  # (B, T, k, out)

        # Weighted sum over top-k
        w = mx.expand_dims(weights, axis=-1)  # (B, T, k, 1)
        delta = mx.sum(selected * w, axis=2)  # (B, T, out)
        return delta

    def all_expert_weights_flat(self) -> list[mx.array]:
        """Return flat weight vectors for each expert (for null-space)."""
        return [expert.flat_weights() for expert in self.experts]


def apply_moe_lora(
    model: nn.Module,
    target_modules: list[str],
    num_experts: int = 4,
    rank: int = 16,
    alpha: float = 32.0,
    top_k: int = 2,
    dropout: float = 0.0,
    router_hidden: int = 64,
    use_rs_lora: bool = True,
) -> int:
    """Attach MoE-LoRA layers to all matching linear projections in the model.

    Walks model.model.layers[i].{self_attn,mlp}.{target_module} and replaces
    the frozen Linear weight with a MoE-LoRA adapter stored alongside.

    The base Linear is NOT removed — the forward pass becomes:
        y = base_linear(x) + moe_lora(x)

    We store the MoE-LoRA as an attribute named `_moe_lora` on the parent module.

    Returns:
        count: number of MoE-LoRA layers attached
    """
    count = 0
    layers = model.model.layers if hasattr(model, "model") else model.layers

    for layer_idx, layer in enumerate(layers):
        for sub_name in ["self_attn", "mlp"]:
            sub_module = getattr(layer, sub_name, None)
            if sub_module is None:
                continue
            for target in target_modules:
                linear = getattr(sub_module, target, None)
                if linear is None or not isinstance(linear, nn.Linear):
                    continue
                in_f = linear.weight.shape[1]
                out_f = linear.weight.shape[0]
                moe = MoELoRALayer(
                    in_features=in_f,
                    out_features=out_f,
                    num_experts=num_experts,
                    rank=rank,
                    alpha=alpha,
                    top_k=top_k,
                    dropout=dropout,
                    router_hidden=router_hidden,
                    use_rs_lora=use_rs_lora,
                )
                # Store as sibling attribute: layer.self_attn.q_proj_moe_lora
                attr_name = f"{target}_moe_lora"
                setattr(sub_module, attr_name, moe)
                count += 1

    return count


def collect_moe_lora_layers(model: nn.Module) -> list[MoELoRALayer]:
    """Walk the model and collect all attached MoE-LoRA layers."""
    moe_layers = []
    layers = model.model.layers if hasattr(model, "model") else model.layers
    for layer in layers:
        for sub_name in ["self_attn", "mlp"]:
            sub_module = getattr(layer, sub_name, None)
            if sub_module is None:
                continue
            for attr_name in dir(sub_module):
                if attr_name.endswith("_moe_lora"):
                    moe = getattr(sub_module, attr_name)
                    if isinstance(moe, MoELoRALayer):
                        moe_layers.append(moe)
    return moe_layers


def moe_lora_forward_hook(base_linear: nn.Linear, moe_lora: MoELoRALayer, x: mx.array) -> mx.array:
    """Combined forward: frozen base + MoE-LoRA delta.

    This is called by the patched forward in train_stack.py.
    """
    base_out = base_linear(x)
    delta = moe_lora(x)
    return base_out + delta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/clems/KIKI-Mac_tunner && python tests/micro_kiki/test_moe_lora.py`
Expected: `All MoE-LoRA tests passed.`

- [ ] **Step 5: Commit**

```bash
git add scripts/micro_kiki/moe_lora.py tests/micro_kiki/test_moe_lora.py
git commit -m "feat: implement MoE-LoRA module with top-k routing for Brainstacks"
```

---

### Task 3: Implement null-space projection

**Files:**
- Create: `scripts/micro_kiki/null_space.py`
- Create: `tests/micro_kiki/test_null_space.py`

- [ ] **Step 1: Write the failing tests**

```python
#!/usr/bin/env python3
"""Tests for null-space projection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import mlx.core as mx
import numpy as np
from micro_kiki.null_space import (
    randomized_svd,
    compute_null_space_projector,
    project_gradient,
)


def test_randomized_svd_shapes():
    """Randomized SVD returns correct shapes."""
    np.random.seed(42)
    A = np.random.randn(1000, 500).astype(np.float32)
    U, S, Vt = randomized_svd(A, n_components=32, n_oversamples=10, n_iter=3)
    assert U.shape == (1000, 32), f"U shape: {U.shape}"
    assert S.shape == (32,), f"S shape: {S.shape}"
    assert Vt.shape == (32, 500), f"Vt shape: {Vt.shape}"


def test_randomized_svd_approx_quality():
    """Randomized SVD captures the dominant singular values."""
    np.random.seed(42)
    # Create a low-rank matrix (rank 10) embedded in 500 dims
    U_true = np.random.randn(200, 10).astype(np.float32)
    V_true = np.random.randn(10, 500).astype(np.float32)
    A = U_true @ V_true + 0.01 * np.random.randn(200, 500).astype(np.float32)

    _, S_approx, _ = randomized_svd(A, n_components=10, n_oversamples=5, n_iter=3)
    _, S_full, _ = np.linalg.svd(A, full_matrices=False)

    # Top 10 singular values should match within 5%
    ratio = S_approx / S_full[:10]
    assert np.all(ratio > 0.95) and np.all(ratio < 1.05), f"SVD approx ratio: {ratio}"


def test_null_space_projector_orthogonality():
    """Projector sends frozen directions to zero."""
    np.random.seed(42)
    # Simulate frozen stack weights: 3 stacks, each 100-dim
    frozen_weights = [np.random.randn(100).astype(np.float32) for _ in range(3)]
    P = compute_null_space_projector(frozen_weights, weight_dim=100, ns_top_k_dirs=32)

    # Project frozen weights — they should be near zero
    for w in frozen_weights:
        projected = P @ w
        norm_ratio = np.linalg.norm(projected) / np.linalg.norm(w)
        assert norm_ratio < 0.1, f"Frozen weight not zeroed: ratio={norm_ratio:.4f}"


def test_null_space_projector_preserves_orthogonal():
    """Projector preserves directions orthogonal to frozen space."""
    np.random.seed(42)
    frozen_weights = [np.array([1, 0, 0, 0, 0], dtype=np.float32)]
    P = compute_null_space_projector(frozen_weights, weight_dim=5, ns_top_k_dirs=1)

    # Direction orthogonal to [1,0,0,0,0] should be preserved
    orth_vec = np.array([0, 0, 0, 1, 0], dtype=np.float32)
    projected = P @ orth_vec
    norm_ratio = np.linalg.norm(projected) / np.linalg.norm(orth_vec)
    assert norm_ratio > 0.95, f"Orthogonal direction damaged: ratio={norm_ratio:.4f}"


def test_project_gradient_mlx():
    """project_gradient works with MLX arrays."""
    np.random.seed(42)
    frozen_weights = [np.random.randn(50).astype(np.float32)]
    P = compute_null_space_projector(frozen_weights, weight_dim=50, ns_top_k_dirs=10)
    P_mx = mx.array(P)

    grad = mx.random.normal((50,))
    projected = project_gradient(grad, P_mx)
    assert projected.shape == (50,), f"Shape mismatch: {projected.shape}"


def test_empty_frozen_returns_identity():
    """With no frozen stacks, projector is identity."""
    P = compute_null_space_projector([], weight_dim=100, ns_top_k_dirs=32)
    assert P.shape == (100, 100)
    assert np.allclose(P, np.eye(100)), "Empty frozen should give identity projector"


if __name__ == "__main__":
    test_randomized_svd_shapes()
    print("PASS: test_randomized_svd_shapes")
    test_randomized_svd_approx_quality()
    print("PASS: test_randomized_svd_approx_quality")
    test_null_space_projector_orthogonality()
    print("PASS: test_null_space_projector_orthogonality")
    test_null_space_projector_preserves_orthogonal()
    print("PASS: test_null_space_projector_preserves_orthogonal")
    test_project_gradient_mlx()
    print("PASS: test_project_gradient_mlx")
    test_empty_frozen_returns_identity()
    print("PASS: test_empty_frozen_returns_identity")
    print("\nAll null-space tests passed.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/clems/KIKI-Mac_tunner && python tests/micro_kiki/test_null_space.py`
Expected: `ModuleNotFoundError: No module named 'micro_kiki.null_space'`

- [ ] **Step 3: Write null-space projection implementation**

```python
#!/usr/bin/env python3
"""Null-space projection for Brainstacks continual learning.

When training stack N, we must not interfere with stacks 1..N-1.
We collect the frozen weight directions from all previous stacks,
compute a null-space projector via randomized SVD, and project
each gradient update into that null-space.

This guarantees zero catastrophic forgetting on previously learned domains.

Paper: Brainstacks (arXiv:2604.01152), Section 3.2
Also: Orthogonal Gradient Descent (OGD), Chaudhry et al. 2020
"""

import numpy as np
import mlx.core as mx


def randomized_svd(
    A: np.ndarray,
    n_components: int,
    n_oversamples: int = 10,
    n_iter: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized truncated SVD using the Halko-Martinsson-Tropp algorithm.

    Computes the top-k singular vectors/values of A without forming
    the full SVD. Memory: O(m*k + n*k) instead of O(m*n).

    Args:
        A: (m, n) matrix
        n_components: number of singular vectors to extract (k)
        n_oversamples: extra random vectors for stability
        n_iter: power iterations to sharpen spectrum

    Returns:
        U: (m, k) left singular vectors
        S: (k,) singular values
        Vt: (k, n) right singular vectors transposed
    """
    m, n = A.shape
    k = min(n_components, min(m, n))
    total = k + n_oversamples

    # Step 1: Random projection
    rng = np.random.RandomState(42)
    Omega = rng.randn(n, total).astype(A.dtype)
    Y = A @ Omega  # (m, total)

    # Step 2: Power iterations to sharpen
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)

    # Step 3: QR factorization of Y
    Q, _ = np.linalg.qr(Y)  # Q: (m, total)

    # Step 4: Project A into the low-rank basis
    B = Q.T @ A  # (total, n)

    # Step 5: SVD of the small matrix B
    U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)

    # Step 6: Recover left singular vectors
    U = Q @ U_hat

    return U[:, :k], S[:k], Vt[:k, :]


def compute_null_space_projector(
    frozen_weight_vectors: list[np.ndarray],
    weight_dim: int,
    ns_top_k_dirs: int = 32,
    svd_oversampling: int = 10,
    svd_n_iter: int = 3,
) -> np.ndarray:
    """Compute the null-space projection matrix P.

    P projects any vector into the subspace orthogonal to the
    top-k directions of the frozen stack weight matrix.

    If no frozen stacks exist, returns identity (no projection).

    Args:
        frozen_weight_vectors: list of 1D numpy arrays, one per frozen
            MoE-LoRA layer. Each vector is the concatenation of all
            expert weights (A and B matrices) for that layer.
        weight_dim: dimensionality of each weight vector
        ns_top_k_dirs: number of directions to preserve per stack
        svd_oversampling: oversampling factor for randomized SVD
        svd_n_iter: power iterations for randomized SVD

    Returns:
        P: (weight_dim, weight_dim) projection matrix into null-space
    """
    if len(frozen_weight_vectors) == 0:
        return np.eye(weight_dim, dtype=np.float32)

    # Stack frozen weights into a matrix: (num_frozen, weight_dim)
    W = np.stack(frozen_weight_vectors, axis=0).astype(np.float32)

    # Number of components to extract
    n_components = min(ns_top_k_dirs * len(frozen_weight_vectors), weight_dim - 1)
    n_components = max(1, n_components)

    if W.shape[0] < n_components:
        # Fewer frozen vectors than requested components: use full SVD
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        # Keep all non-trivial directions
        n_keep = min(len(S), n_components)
        V_keep = Vt[:n_keep, :]  # (n_keep, weight_dim)
    else:
        # Use randomized SVD for efficiency
        U, S, Vt = randomized_svd(
            W, n_components=n_components,
            n_oversamples=svd_oversampling,
            n_iter=svd_n_iter,
        )
        V_keep = Vt  # (n_components, weight_dim)

    # Null-space projector: P = I - V^T @ V
    # V_keep rows are the directions to block
    P = np.eye(weight_dim, dtype=np.float32) - V_keep.T @ V_keep

    return P


def project_gradient(grad: mx.array, projector: mx.array) -> mx.array:
    """Project a gradient vector into the null-space.

    Args:
        grad: (weight_dim,) or (d1, d2) gradient tensor
        projector: (weight_dim, weight_dim) null-space projection matrix

    Returns:
        Projected gradient with same shape as input
    """
    original_shape = grad.shape
    flat = mx.reshape(grad, (-1,))
    projected = projector @ flat
    return mx.reshape(projected, original_shape)


def collect_frozen_weights_from_disk(
    stack_dirs: list[str],
) -> dict[str, list[np.ndarray]]:
    """Load frozen stack weights from safetensors files on disk.

    For each MoE-LoRA layer name (e.g. "layers.0.self_attn.q_proj_moe_lora"),
    collects the concatenated expert weights from all frozen stacks.

    Args:
        stack_dirs: list of directories containing adapters.safetensors

    Returns:
        Dictionary mapping layer_name -> list of weight vectors (one per stack)
    """
    from safetensors.numpy import load_file

    layer_weights: dict[str, list[np.ndarray]] = {}

    for stack_dir in stack_dirs:
        path = f"{stack_dir}/adapters.safetensors"
        weights = load_file(path)

        # Group by MoE-LoRA layer
        layer_parts: dict[str, list[np.ndarray]] = {}
        for key, tensor in weights.items():
            # Keys like: "model.layers.0.self_attn.q_proj_moe_lora.experts.0.lora_a"
            # Extract layer identifier up to the moe_lora part
            parts = key.split(".")
            moe_idx = None
            for i, p in enumerate(parts):
                if p.endswith("_moe_lora"):
                    moe_idx = i
                    break
            if moe_idx is None:
                continue
            layer_name = ".".join(parts[:moe_idx + 1])
            if layer_name not in layer_parts:
                layer_parts[layer_name] = []
            layer_parts[layer_name].append(tensor.flatten())

        # Concatenate all expert weights per layer into a single vector
        for layer_name, parts in layer_parts.items():
            vec = np.concatenate(parts)
            if layer_name not in layer_weights:
                layer_weights[layer_name] = []
            layer_weights[layer_name].append(vec)

    return layer_weights


def build_projectors_for_stack(
    frozen_stack_dirs: list[str],
    ns_top_k_dirs: int = 32,
    svd_oversampling: int = 10,
    svd_n_iter: int = 3,
) -> dict[str, mx.array]:
    """Build null-space projectors for all MoE-LoRA layers.

    Called before training stack N. Loads stacks 1..N-1 from disk,
    computes one projector per MoE-LoRA layer name.

    Args:
        frozen_stack_dirs: paths to frozen stack output directories
        ns_top_k_dirs: directions to block per frozen stack
        svd_oversampling: for randomized SVD
        svd_n_iter: power iterations

    Returns:
        Dict mapping layer_name -> (weight_dim, weight_dim) MLX projector
    """
    if len(frozen_stack_dirs) == 0:
        return {}

    layer_weights = collect_frozen_weights_from_disk(frozen_stack_dirs)
    projectors = {}

    for layer_name, weight_vectors in layer_weights.items():
        weight_dim = weight_vectors[0].shape[0]
        P_np = compute_null_space_projector(
            frozen_weight_vectors=weight_vectors,
            weight_dim=weight_dim,
            ns_top_k_dirs=ns_top_k_dirs,
            svd_oversampling=svd_oversampling,
            svd_n_iter=svd_n_iter,
        )
        projectors[layer_name] = mx.array(P_np)

    return projectors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/clems/KIKI-Mac_tunner && python tests/micro_kiki/test_null_space.py`
Expected: `All null-space tests passed.`

- [ ] **Step 5: Commit**

```bash
git add scripts/micro_kiki/null_space.py tests/micro_kiki/test_null_space.py
git commit -m "feat: implement null-space projection via randomized SVD for Brainstacks"
```

---

### Task 4: Implement residual boosting

**Files:**
- Create: `scripts/micro_kiki/residual_boost.py`
- Create: `tests/micro_kiki/test_residual_boost.py`

- [ ] **Step 1: Write the failing tests**

```python
#!/usr/bin/env python3
"""Tests for residual boosting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import mlx.core as mx
import numpy as np
from micro_kiki.residual_boost import (
    compute_per_example_loss,
    select_hard_examples,
    build_boosted_weights,
)


def test_compute_per_example_loss_shape():
    """Per-example loss returns one scalar per example."""
    # Simulate logits and labels
    batch_logits = [mx.random.normal((1, 32, 1000)) for _ in range(5)]
    batch_labels = [mx.random.randint(0, 1000, (1, 32)) for _ in range(5)]
    lengths = [32, 32, 32, 32, 32]
    losses = compute_per_example_loss(batch_logits, batch_labels, lengths)
    assert len(losses) == 5, f"Expected 5 losses, got {len(losses)}"
    for loss in losses:
        assert isinstance(loss, float), f"Loss should be float, got {type(loss)}"
        assert loss >= 0, f"Loss should be non-negative, got {loss}"


def test_select_hard_examples():
    """Hard examples are those above the quantile threshold."""
    losses = [0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 1.0]
    indices = select_hard_examples(losses, quantile=0.75)
    # Top 25%: losses >= 0.8 -> indices 2 (0.9), 4 (0.8), 9 (1.0)
    # Actually quantile 0.75 means we pick losses >= 75th percentile
    assert len(indices) > 0, "Should select at least one hard example"
    for idx in indices:
        assert losses[idx] >= 0.7, f"Index {idx} with loss {losses[idx]} is not hard"


def test_build_boosted_weights():
    """Boosted weights are higher for hard examples."""
    losses = [0.1, 0.5, 0.9, 0.2, 0.8]
    hard_indices = select_hard_examples(losses, quantile=0.75)
    weights = build_boosted_weights(
        num_examples=5, hard_indices=hard_indices, boost_weight=2.0
    )
    assert len(weights) == 5
    for i in hard_indices:
        assert weights[i] == 2.0, f"Hard example {i} should have weight 2.0"
    for i in range(5):
        if i not in hard_indices:
            assert weights[i] == 1.0, f"Easy example {i} should have weight 1.0"


if __name__ == "__main__":
    test_compute_per_example_loss_shape()
    print("PASS: test_compute_per_example_loss_shape")
    test_select_hard_examples()
    print("PASS: test_select_hard_examples")
    test_build_boosted_weights()
    print("PASS: test_build_boosted_weights")
    print("\nAll residual boosting tests passed.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/clems/KIKI-Mac_tunner && python tests/micro_kiki/test_residual_boost.py`
Expected: `ModuleNotFoundError: No module named 'micro_kiki.residual_boost'`

- [ ] **Step 3: Write residual boosting implementation**

```python
#!/usr/bin/env python3
"""Residual boosting for Brainstacks training.

After the main SFT pass on a domain, residual boosting identifies "hard"
examples (top 25% by loss) and retrains the stack with boosted weights
on those examples. This squeezes extra performance on the tail of the
distribution without overfitting the easy examples.

Paper: Brainstacks (arXiv:2604.01152), Section 3.3
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def compute_per_example_loss(
    batch_logits: list[mx.array],
    batch_labels: list[mx.array],
    lengths: list[int],
) -> list[float]:
    """Compute cross-entropy loss for each example individually.

    Args:
        batch_logits: list of (1, seq_len, vocab_size) logit tensors
        batch_labels: list of (1, seq_len) label tensors
        lengths: list of actual sequence lengths (before padding)

    Returns:
        List of float losses, one per example
    """
    losses = []
    for logits, labels, length in zip(batch_logits, batch_labels, lengths):
        # Shift: predict next token
        shift_logits = logits[:, :-1, :]  # (1, T-1, V)
        shift_labels = labels[:, 1:]       # (1, T-1)

        # Truncate to actual length
        eff_len = min(length - 1, shift_logits.shape[1])
        if eff_len <= 0:
            losses.append(0.0)
            continue

        sl = shift_logits[:, :eff_len, :]  # (1, eff_len, V)
        tl = shift_labels[:, :eff_len]     # (1, eff_len)

        # Cross entropy
        log_probs = nn.log_softmax(sl, axis=-1)           # (1, eff_len, V)
        tl_expanded = mx.expand_dims(tl, axis=-1)         # (1, eff_len, 1)
        token_losses = -mx.take_along_axis(log_probs, tl_expanded, axis=-1)  # (1, eff_len, 1)
        token_losses = mx.squeeze(token_losses, axis=-1)  # (1, eff_len)

        mean_loss = mx.mean(token_losses).item()
        losses.append(float(mean_loss))

    return losses


def select_hard_examples(
    losses: list[float],
    quantile: float = 0.75,
) -> list[int]:
    """Select hard examples based on loss quantile.

    Args:
        losses: per-example losses
        quantile: threshold quantile (0.75 = top 25% hardest)

    Returns:
        List of indices into the original dataset for hard examples
    """
    if len(losses) == 0:
        return []

    threshold = float(np.quantile(losses, quantile))
    hard_indices = [i for i, loss in enumerate(losses) if loss >= threshold]

    # Always select at least 1 example
    if len(hard_indices) == 0 and len(losses) > 0:
        hard_indices = [int(np.argmax(losses))]

    return hard_indices


def build_boosted_weights(
    num_examples: int,
    hard_indices: list[int],
    boost_weight: float = 2.0,
) -> list[float]:
    """Build per-example loss weights for boosted training.

    Hard examples get boost_weight, others get 1.0.

    Args:
        num_examples: total number of examples
        hard_indices: indices of hard examples
        boost_weight: multiplier for hard examples

    Returns:
        List of float weights, one per example
    """
    hard_set = set(hard_indices)
    return [boost_weight if i in hard_set else 1.0 for i in range(num_examples)]


def run_residual_boost_round(
    model: nn.Module,
    tokenizer,
    dataset: list[dict],
    optimizer,
    projectors: dict,
    config: dict,
    round_num: int,
) -> float:
    """Run one round of residual boosting.

    1. Evaluate per-example loss on the full dataset
    2. Select hard examples (top quantile)
    3. Retrain for boost_steps with boosted weights
    4. Return the average loss after boosting

    Args:
        model: model with MoE-LoRA attached
        tokenizer: tokenizer for encoding
        dataset: list of {"messages": [...]} dicts
        optimizer: MLX optimizer (will be reset with lower LR)
        projectors: null-space projectors dict (layer_name -> projector)
        config: residual_boost section from brainstacks.yaml
        round_num: 1-indexed round number (for logging)

    Returns:
        Average loss after this boost round
    """
    from micro_kiki.moe_lora import collect_moe_lora_layers

    boost_steps = config.get("boost_steps", 100)
    boost_weight = config.get("boost_weight", 2.0)
    quantile = config.get("hard_example_quantile", 0.75)

    print(f"\n  [Boost round {round_num}] Evaluating per-example loss...")

    # Step 1: Collect per-example losses
    batch_logits = []
    batch_labels = []
    lengths = []

    for example in dataset:
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer.encode(text)
        max_len = config.get("max_seq_length", 2048)
        tokens = tokens[:max_len]
        length = len(tokens)

        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        batch_logits.append(logits)
        batch_labels.append(input_ids)
        lengths.append(length)

    losses = compute_per_example_loss(batch_logits, batch_labels, lengths)
    avg_loss_before = sum(losses) / len(losses) if losses else 0.0
    print(f"  [Boost round {round_num}] Avg loss before: {avg_loss_before:.4f}")

    # Step 2: Select hard examples
    hard_indices = select_hard_examples(losses, quantile=quantile)
    print(f"  [Boost round {round_num}] Hard examples: {len(hard_indices)}/{len(dataset)}")

    if len(hard_indices) == 0:
        return avg_loss_before

    # Step 3: Build boosted dataset
    weights = build_boosted_weights(len(dataset), hard_indices, boost_weight)

    # Step 4: Retrain on weighted dataset for boost_steps
    hard_dataset = [dataset[i] for i in hard_indices]
    step = 0
    total_loss = 0.0

    while step < boost_steps:
        for idx in hard_indices:
            if step >= boost_steps:
                break

            example = dataset[idx]
            w = weights[idx]
            messages = example["messages"]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            tokens = tokenizer.encode(text)[:config.get("max_seq_length", 2048)]
            input_ids = mx.array([tokens])
            labels = input_ids

            def loss_fn(model_params):
                model.update(model_params)
                logits = model(input_ids)
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                ce = nn.losses.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    reduction="mean",
                )
                return ce * w  # apply boost weight

            loss, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
            mx.eval(loss)

            # Project gradients into null-space (if projectors exist)
            if projectors:
                grads = _project_all_grads(grads, projectors)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            step += 1

    avg_loss_after = total_loss / max(step, 1)
    print(f"  [Boost round {round_num}] Avg loss after {step} steps: {avg_loss_after:.4f}")

    return avg_loss_after


def _project_all_grads(grads: dict, projectors: dict) -> dict:
    """Apply null-space projection to all MoE-LoRA gradients.

    This is a recursive walk matching gradient keys to projector keys.
    Only MoE-LoRA weights (keys containing '_moe_lora') are projected.
    """
    from micro_kiki.null_space import project_gradient

    def _walk(g, prefix=""):
        if isinstance(g, dict):
            return {k: _walk(v, f"{prefix}.{k}" if prefix else k) for k, v in g.items()}
        elif isinstance(g, list):
            return [_walk(v, f"{prefix}.{i}") for i, v in enumerate(g)]
        elif isinstance(g, mx.array):
            # Check if this gradient belongs to a MoE-LoRA layer
            for layer_name, P in projectors.items():
                if layer_name in prefix:
                    return project_gradient(g, P)
            return g
        return g

    return _walk(grads)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/clems/KIKI-Mac_tunner && python tests/micro_kiki/test_residual_boost.py`
Expected: `All residual boosting tests passed.`

- [ ] **Step 5: Commit**

```bash
git add scripts/micro_kiki/residual_boost.py tests/micro_kiki/test_residual_boost.py
git commit -m "feat: implement residual boosting for hard example retraining"
```

---

### Task 5: Implement single stack training script

**Files:**
- Create: `scripts/micro_kiki/train_stack.py`
- Create: `tests/micro_kiki/test_train_stack.py`

- [ ] **Step 1: Write the failing integration test**

```python
#!/usr/bin/env python3
"""Integration test for single stack training.

Uses a tiny synthetic model and dataset to verify the full pipeline:
load -> attach MoE-LoRA -> null-space -> train -> boost -> freeze -> save.
"""
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import mlx.core as mx
import mlx.nn as nn

from micro_kiki.train_stack import (
    load_domain_dataset,
    freeze_and_save_stack,
    extract_moe_lora_state_dict,
)


def test_load_domain_dataset():
    """load_domain_dataset reads JSONL and returns list of dicts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "train.jsonl"
        valid_path = Path(tmpdir) / "valid.jsonl"
        example = {"messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]}
        train_path.write_text("\n".join([json.dumps(example)] * 10))
        valid_path.write_text("\n".join([json.dumps(example)] * 2))

        train, valid = load_domain_dataset(tmpdir)
        assert len(train) == 10, f"Expected 10 train, got {len(train)}"
        assert len(valid) == 2, f"Expected 2 valid, got {len(valid)}"
        assert "messages" in train[0]


def test_extract_moe_lora_state_dict():
    """extract_moe_lora_state_dict collects only MoE-LoRA parameters."""
    from micro_kiki.moe_lora import MoELoRALayer

    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.q_proj_moe_lora = MoELoRALayer(64, 64, 4, 8, 16.0, 2, 0.0, 16)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttn()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = [FakeLayer()]

    model = FakeModel()
    state = extract_moe_lora_state_dict(model)
    assert len(state) > 0, "Should extract at least one parameter"
    for key in state:
        assert "moe_lora" in key, f"Key {key} should contain 'moe_lora'"


def test_freeze_and_save_stack():
    """freeze_and_save_stack creates a safetensors file."""
    from micro_kiki.moe_lora import MoELoRALayer

    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.q_proj_moe_lora = MoELoRALayer(64, 64, 4, 8, 16.0, 2, 0.0, 16)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttn()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = [FakeLayer()]

    model = FakeModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        freeze_and_save_stack(model, tmpdir, domain="test")
        saved = Path(tmpdir) / "adapters.safetensors"
        assert saved.exists(), f"Expected {saved} to exist"
        meta = Path(tmpdir) / "stack_meta.json"
        assert meta.exists(), f"Expected {meta} to exist"


if __name__ == "__main__":
    test_load_domain_dataset()
    print("PASS: test_load_domain_dataset")
    test_extract_moe_lora_state_dict()
    print("PASS: test_extract_moe_lora_state_dict")
    test_freeze_and_save_stack()
    print("PASS: test_freeze_and_save_stack")
    print("\nAll train_stack integration tests passed.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/clems/KIKI-Mac_tunner && python tests/micro_kiki/test_train_stack.py`
Expected: `ModuleNotFoundError: No module named 'micro_kiki.train_stack'`

- [ ] **Step 3: Write the single stack training script**

```python
#!/usr/bin/env python3
"""Train a single Brainstacks domain stack.

Full pipeline for one domain:
1. Load frozen Qwen3.5-4B base
2. Attach MoE-LoRA (4 experts, rank 16, top-2)
3. Compute null-space projector from previously frozen stacks
4. SFT on domain data (~500 steps)
5. Residual boost (1-2 rounds on hard examples)
6. Freeze MoE-LoRA weights → save to disk
7. Evaluate all previous domains (forgetting check)

Usage:
    python scripts/micro_kiki/train_stack.py \\
        --config configs/micro_kiki/brainstacks.yaml \\
        --domain python \\
        --stack-index 3
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import yaml
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Ensure scripts/ is on path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from micro_kiki.moe_lora import (
    apply_moe_lora,
    collect_moe_lora_layers,
    MoELoRALayer,
)
from micro_kiki.null_space import (
    build_projectors_for_stack,
    project_gradient,
)
from micro_kiki.residual_boost import (
    run_residual_boost_round,
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_domain_dataset(domain_dir: str) -> tuple[list[dict], list[dict]]:
    """Load train.jsonl and valid.jsonl for a domain.

    Args:
        domain_dir: path to data/micro-kiki/<domain>/

    Returns:
        (train_examples, valid_examples) as lists of dicts
    """
    train_path = Path(domain_dir) / "train.jsonl"
    valid_path = Path(domain_dir) / "valid.jsonl"

    train = []
    if train_path.exists():
        with open(train_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    train.append(json.loads(line))

    valid = []
    if valid_path.exists():
        with open(valid_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    valid.append(json.loads(line))

    return train, valid


def extract_moe_lora_state_dict(model: nn.Module) -> dict[str, mx.array]:
    """Extract only the MoE-LoRA parameters from the model.

    Walks the model tree and collects parameters whose path
    contains '_moe_lora'. These are the only trainable params.

    Returns:
        Dict mapping parameter path -> mx.array
    """
    from mlx.utils import tree_flatten

    all_params = tree_flatten(model.parameters())
    moe_params = {}
    for name, param in all_params:
        if "moe_lora" in name:
            moe_params[name] = param
    return moe_params


def freeze_and_save_stack(
    model: nn.Module,
    output_dir: str,
    domain: str,
    train_loss: float = 0.0,
    val_loss: float = 0.0,
    steps: int = 0,
) -> None:
    """Freeze the current MoE-LoRA stack and save to disk.

    Saves:
    - adapters.safetensors: all MoE-LoRA weights
    - stack_meta.json: domain name, loss, training info

    Args:
        model: model with MoE-LoRA attached
        output_dir: where to save
        domain: domain name
        train_loss: final training loss
        val_loss: final validation loss
        steps: number of training steps completed
    """
    from safetensors.mlx import save_file

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Extract and save MoE-LoRA weights
    state = extract_moe_lora_state_dict(model)
    if state:
        save_file(state, str(out / "adapters.safetensors"))

    # Save metadata
    meta = {
        "domain": domain,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "steps": steps,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out / "stack_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Count params
    total_params = sum(p.size for p in state.values()) if state else 0
    size_mb = sum(p.nbytes for p in state.values()) / (1024 * 1024) if state else 0
    print(f"  Stack saved: {out}")
    print(f"  Parameters: {total_params:,} ({size_mb:.1f} MB)")


def evaluate_domain(
    model: nn.Module,
    tokenizer,
    domain_dir: str,
    max_seq_length: int = 2048,
    val_batches: int = 10,
) -> float:
    """Evaluate model loss on a domain's validation set.

    Args:
        model: model with MoE-LoRA attached
        tokenizer: tokenizer
        domain_dir: path to domain data dir
        max_seq_length: max tokens
        val_batches: max examples to evaluate

    Returns:
        Average cross-entropy loss on validation set
    """
    valid_path = Path(domain_dir) / "valid.jsonl"
    if not valid_path.exists():
        return float("inf")

    examples = []
    with open(valid_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    examples = examples[:val_batches]

    if len(examples) == 0:
        return float("inf")

    total_loss = 0.0
    count = 0
    for example in examples:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer.encode(text)[:max_seq_length]
        if len(tokens) < 2:
            continue

        input_ids = mx.array([tokens])
        logits = model(input_ids)

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        loss = nn.losses.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="mean",
        )
        mx.eval(loss)
        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


def train_single_stack(config_path: str, domain: str, stack_index: int) -> None:
    """Train a single domain stack end-to-end.

    Args:
        config_path: path to brainstacks.yaml
        domain: domain name (e.g. "python")
        stack_index: 1-indexed position in curriculum
    """
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent.parent

    model_cfg = config["model"]
    moe_cfg = config["moe_lora"]
    ns_cfg = config["null_space"]
    boost_cfg = config["residual_boost"]
    train_cfg = config["training"]
    forget_cfg = config["forgetting"]
    output_cfg = config["output"]
    data_cfg = config["data"]
    curriculum = config["curriculum"]

    print("=" * 60)
    print(f"Brainstacks — Training stack {stack_index}/{len(curriculum)}: {domain}")
    print("=" * 60)

    # ---- 1. Load base model (frozen) ----
    print("\n[1/7] Loading base model...")
    model_path = str(project_root / model_cfg["path"])

    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(model_path)

    # Set memory limits for Mac
    mx.set_memory_limit(460 * 1024**3)
    mx.set_cache_limit(32 * 1024**3)

    # Freeze all base parameters
    model.freeze()

    # ---- 2. Attach MoE-LoRA ----
    print("\n[2/7] Attaching MoE-LoRA...")
    n_attached = apply_moe_lora(
        model,
        target_modules=model_cfg["target_modules"],
        num_experts=moe_cfg["num_experts"],
        rank=moe_cfg["rank"],
        alpha=moe_cfg["alpha"],
        top_k=moe_cfg["top_k"],
        dropout=moe_cfg.get("dropout", 0.01),
        router_hidden=moe_cfg["router_hidden"],
        use_rs_lora=moe_cfg.get("use_rs_lora", True),
    )
    print(f"  Attached {n_attached} MoE-LoRA layers")

    # Count trainable params
    from mlx.utils import tree_flatten
    all_params = tree_flatten(model.parameters())
    trainable = sum(p.size for name, p in all_params if "moe_lora" in name)
    total = sum(p.size for _, p in all_params)
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.4f}%)")

    # ---- 3. Build null-space projectors from frozen stacks ----
    print("\n[3/7] Building null-space projectors...")
    frozen_domains = curriculum[:stack_index - 1]
    frozen_dirs = []
    for d in frozen_domains:
        d_path = str(project_root / output_cfg["base_dir"] / d)
        if Path(d_path).exists() and (Path(d_path) / "adapters.safetensors").exists():
            frozen_dirs.append(d_path)

    projectors = {}
    if len(frozen_dirs) > 0:
        projectors = build_projectors_for_stack(
            frozen_stack_dirs=frozen_dirs,
            ns_top_k_dirs=ns_cfg["ns_top_k_dirs"],
            svd_oversampling=ns_cfg.get("svd_oversampling", 10),
            svd_n_iter=ns_cfg.get("svd_n_iter", 3),
        )
        print(f"  Projectors built from {len(frozen_dirs)} frozen stacks")
        print(f"  Covering {len(projectors)} MoE-LoRA layers")
    else:
        print(f"  No frozen stacks (first domain in curriculum)")

    # ---- 4. Load domain data ----
    print("\n[4/7] Loading domain data...")
    domain_dir = str(project_root / data_cfg["base_dir"] / domain)
    train_data, valid_data = load_domain_dataset(domain_dir)
    print(f"  Train: {len(train_data)} | Valid: {len(valid_data)}")

    # ---- 5. SFT training loop ----
    print("\n[5/7] SFT training...")
    lr = train_cfg["learning_rate"]
    max_steps = train_cfg["max_steps"]
    warmup_steps = int(max_steps * train_cfg.get("warmup_ratio", 0.05))
    batch_size = train_cfg["batch_size"]
    grad_accum = train_cfg["grad_accumulation_steps"]
    max_seq_len = train_cfg["max_seq_length"]
    steps_per_eval = train_cfg["steps_per_eval"]
    seed = train_cfg.get("seed", 42)

    mx.random.seed(seed)

    # Cosine schedule with warmup
    schedule = optim.join_schedules(
        [optim.linear_schedule(1e-7, lr, warmup_steps),
         optim.cosine_decay(lr, max_steps - warmup_steps)],
        [warmup_steps],
    )
    optimizer = optim.AdamW(
        learning_rate=schedule,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # Training loop
    step = 0
    epoch = 0
    running_loss = 0.0
    best_val_loss = float("inf")
    train_start = time.time()

    while step < max_steps:
        epoch += 1
        np.random.shuffle(train_data)

        for example in train_data:
            if step >= max_steps:
                break

            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
            tokens = tokenizer.encode(text)[:max_seq_len]
            if len(tokens) < 2:
                continue

            input_ids = mx.array([tokens])
            labels = input_ids

            def loss_fn(model_params):
                model.update(model_params)
                logits = model(input_ids)
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                ce = nn.losses.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    reduction="mean",
                )
                return ce

            loss, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())

            # Null-space projection on gradients
            if projectors:
                grads = _project_grads(grads, projectors)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            running_loss += loss.item()
            step += 1

            # Report
            if step % 5 == 0:
                avg = running_loss / 5
                elapsed = time.time() - train_start
                print(f"  Step {step}/{max_steps} | loss={avg:.4f} | "
                      f"elapsed={elapsed:.0f}s")
                running_loss = 0.0

            # Eval
            if step % steps_per_eval == 0:
                val_loss = evaluate_domain(
                    model, tokenizer, domain_dir,
                    max_seq_length=max_seq_len,
                    val_batches=train_cfg.get("val_batches", 10),
                )
                print(f"  [Eval step {step}] val_loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

    train_time = time.time() - train_start
    print(f"\n  SFT complete: {step} steps in {train_time:.0f}s")
    print(f"  Best val_loss: {best_val_loss:.4f}")

    # ---- 6. Residual boosting ----
    print("\n[6/7] Residual boosting...")
    max_boost_rounds = boost_cfg.get("max_rounds", 2)
    min_improvement = boost_cfg.get("min_improvement", 0.002)
    prev_loss = best_val_loss

    for round_num in range(1, max_boost_rounds + 1):
        # Reset optimizer with lower LR for boosting
        boost_lr = lr * boost_cfg.get("boost_lr_scale", 0.5)
        boost_optimizer = optim.AdamW(learning_rate=boost_lr)

        avg_loss = run_residual_boost_round(
            model=model,
            tokenizer=tokenizer,
            dataset=train_data,
            optimizer=boost_optimizer,
            projectors=projectors,
            config={**boost_cfg, "max_seq_length": max_seq_len},
            round_num=round_num,
        )

        improvement = prev_loss - avg_loss
        print(f"  Boost round {round_num}: improvement={improvement:.4f}")

        if improvement < min_improvement:
            print(f"  Stopping boost: improvement {improvement:.4f} < {min_improvement}")
            break
        prev_loss = avg_loss

    # ---- 7. Freeze and save ----
    print("\n[7/7] Freezing and saving stack...")
    final_val_loss = evaluate_domain(
        model, tokenizer, domain_dir,
        max_seq_length=max_seq_len,
        val_batches=train_cfg.get("val_batches", 10),
    )
    output_dir = str(project_root / output_cfg["base_dir"] / domain)
    freeze_and_save_stack(
        model, output_dir, domain=domain,
        train_loss=running_loss, val_loss=final_val_loss, steps=step,
    )

    # ---- Forgetting check ----
    print("\n[Forgetting check]")
    max_delta = forget_cfg.get("max_delta", 0.03)
    for prev_domain in frozen_domains:
        prev_dir = str(project_root / data_cfg["base_dir"] / prev_domain)
        prev_loss = evaluate_domain(
            model, tokenizer, prev_dir,
            max_seq_length=max_seq_len,
            val_batches=forget_cfg.get("val_batches", 5),
        )
        # Load the original val_loss from meta
        meta_path = Path(project_root / output_cfg["base_dir"] / prev_domain / "stack_meta.json")
        original_loss = float("inf")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                original_loss = meta.get("val_loss", float("inf"))

        delta = prev_loss - original_loss
        status = "OK" if delta < max_delta else "FAIL"
        print(f"  {prev_domain}: original={original_loss:.4f} "
              f"current={prev_loss:.4f} delta={delta:+.4f} [{status}]")
        if delta >= max_delta:
            print(f"  WARNING: Forgetting detected on {prev_domain}!")

    print(f"\n=== Stack {domain} training complete ===")
    print(f"  Output: {output_dir}")
    print(f"  Val loss: {final_val_loss:.4f}")


def _project_grads(grads, projectors: dict):
    """Apply null-space projection to gradients. Delegates to residual_boost._project_all_grads."""
    from micro_kiki.residual_boost import _project_all_grads
    return _project_all_grads(grads, projectors)


def main():
    parser = argparse.ArgumentParser(
        description="Brainstacks — Train a single domain stack"
    )
    parser.add_argument(
        "--config", type=str, default="configs/micro_kiki/brainstacks.yaml",
        help="Path to brainstacks config YAML",
    )
    parser.add_argument(
        "--domain", type=str, required=True,
        help="Domain name (e.g. 'python', 'embedded')",
    )
    parser.add_argument(
        "--stack-index", type=int, required=True,
        help="1-indexed position in curriculum (for null-space computation)",
    )
    args = parser.parse_args()

    train_single_stack(args.config, args.domain, args.stack_index)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/clems/KIKI-Mac_tunner && python tests/micro_kiki/test_train_stack.py`
Expected: `All train_stack integration tests passed.`

- [ ] **Step 5: Commit**

```bash
git add scripts/micro_kiki/train_stack.py tests/micro_kiki/test_train_stack.py
git commit -m "feat: implement single stack training with null-space + residual boost"
```

---

### Task 6: Implement evaluation script

**Files:**
- Create: `scripts/micro_kiki/eval_stack.py`

- [ ] **Step 1: Write the evaluation script**

```python
#!/usr/bin/env python3
"""Evaluate Brainstacks: per-domain loss, cross-domain forgetting delta.

Usage:
    # Evaluate a single domain
    python scripts/micro_kiki/eval_stack.py \\
        --config configs/micro_kiki/brainstacks.yaml \\
        --domain python

    # Evaluate all trained domains (forgetting matrix)
    python scripts/micro_kiki/eval_stack.py \\
        --config configs/micro_kiki/brainstacks.yaml \\
        --all
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from micro_kiki.moe_lora import apply_moe_lora
from micro_kiki.train_stack import (
    load_config,
    evaluate_domain,
    load_domain_dataset,
)


def load_stack_weights(model: nn.Module, stack_dir: str) -> None:
    """Load a frozen stack's MoE-LoRA weights into the model."""
    adapter_path = Path(stack_dir) / "adapters.safetensors"
    if adapter_path.exists():
        model.load_weights(str(adapter_path), strict=False)


def evaluate_single_domain(
    config_path: str,
    domain: str,
) -> dict:
    """Evaluate a single domain stack.

    Returns dict with domain, val_loss, and per-domain forgetting deltas.
    """
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent.parent
    model_cfg = config["model"]
    moe_cfg = config["moe_lora"]
    output_cfg = config["output"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    curriculum = config["curriculum"]

    # Load base model
    model_path = str(project_root / model_cfg["path"])
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(model_path)
    model.freeze()

    # Attach MoE-LoRA
    apply_moe_lora(
        model,
        target_modules=model_cfg["target_modules"],
        num_experts=moe_cfg["num_experts"],
        rank=moe_cfg["rank"],
        alpha=moe_cfg["alpha"],
        top_k=moe_cfg["top_k"],
        router_hidden=moe_cfg["router_hidden"],
    )

    # Load the domain stack weights
    stack_dir = str(project_root / output_cfg["base_dir"] / domain)
    load_stack_weights(model, stack_dir)

    # Evaluate on own domain
    domain_dir = str(project_root / data_cfg["base_dir"] / domain)
    val_loss = evaluate_domain(
        model, tokenizer, domain_dir,
        max_seq_length=train_cfg["max_seq_length"],
        val_batches=train_cfg.get("val_batches", 10),
    )

    result = {"domain": domain, "val_loss": val_loss}
    print(f"  {domain}: val_loss={val_loss:.4f}")
    return result


def evaluate_all_domains(config_path: str) -> None:
    """Evaluate all trained domains and print forgetting matrix."""
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent.parent
    output_cfg = config["output"]
    data_cfg = config["data"]
    curriculum = config["curriculum"]

    # Find which domains have been trained
    trained = []
    for domain in curriculum:
        stack_dir = project_root / output_cfg["base_dir"] / domain
        if (stack_dir / "adapters.safetensors").exists():
            trained.append(domain)

    if len(trained) == 0:
        print("No trained stacks found.")
        return

    print(f"Found {len(trained)} trained stacks: {', '.join(trained)}")
    print()

    # Load base model once
    model_cfg = config["model"]
    moe_cfg = config["moe_lora"]
    train_cfg = config["training"]
    model_path = str(project_root / model_cfg["path"])
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(model_path)
    model.freeze()

    # For each trained domain, load its stack and eval on all domains
    results = {}
    for active_domain in trained:
        print(f"\n--- Stack: {active_domain} ---")

        # Re-attach fresh MoE-LoRA
        apply_moe_lora(
            model,
            target_modules=model_cfg["target_modules"],
            num_experts=moe_cfg["num_experts"],
            rank=moe_cfg["rank"],
            alpha=moe_cfg["alpha"],
            top_k=moe_cfg["top_k"],
            router_hidden=moe_cfg["router_hidden"],
        )

        # Load this stack's weights
        stack_dir = str(project_root / output_cfg["base_dir"] / active_domain)
        load_stack_weights(model, stack_dir)

        # Eval on all trained domains
        domain_results = {}
        for eval_domain in trained:
            domain_dir = str(project_root / data_cfg["base_dir"] / eval_domain)
            val_loss = evaluate_domain(
                model, tokenizer, domain_dir,
                max_seq_length=train_cfg["max_seq_length"],
                val_batches=5,
            )
            domain_results[eval_domain] = val_loss
            marker = " *" if eval_domain == active_domain else ""
            print(f"  {eval_domain}: {val_loss:.4f}{marker}")

        results[active_domain] = domain_results

    # Print forgetting summary
    print("\n" + "=" * 60)
    print("FORGETTING SUMMARY")
    print("=" * 60)

    max_delta = config["forgetting"]["max_delta"]
    for domain in trained:
        stack_dir = project_root / output_cfg["base_dir"] / domain
        meta_path = stack_dir / "stack_meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        original_loss = meta.get("val_loss", float("inf"))

        # Check how later stacks affected this domain
        domain_idx = trained.index(domain)
        for later_domain in trained[domain_idx + 1:]:
            if domain in results.get(later_domain, {}):
                current_loss = results[later_domain][domain]
                delta = current_loss - original_loss
                status = "OK" if delta < max_delta else "FAIL"
                print(f"  {domain} after {later_domain}: "
                      f"delta={delta:+.4f} [{status}]")


def main():
    parser = argparse.ArgumentParser(description="Brainstacks — Evaluate stacks")
    parser.add_argument(
        "--config", type=str, default="configs/micro_kiki/brainstacks.yaml",
    )
    parser.add_argument("--domain", type=str, help="Evaluate a single domain")
    parser.add_argument("--all", action="store_true", help="Evaluate all trained domains")
    args = parser.parse_args()

    if args.all:
        evaluate_all_domains(args.config)
    elif args.domain:
        evaluate_single_domain(args.config, args.domain)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses**

Run: `cd /Users/clems/KIKI-Mac_tunner && python -c "import ast; ast.parse(open('scripts/micro_kiki/eval_stack.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/micro_kiki/eval_stack.py
git commit -m "feat: add per-domain evaluation and forgetting matrix script"
```

---

### Task 7: Implement full 32-stack orchestrator

**Files:**
- Create: `scripts/micro_kiki/train_all_stacks.sh`
- Create: `scripts/micro_kiki/__init__.py`

- [ ] **Step 1: Create the Python package init file**

```python
#!/usr/bin/env python3
"""Micro_KIKI Brainstacks — 32 MoE-LoRA stacks for Qwen3.5-4B."""
```

- [ ] **Step 2: Write the orchestrator shell script**

```bash
#!/usr/bin/env bash
# ==============================================================================
# Brainstacks — Train all 32 domain stacks sequentially
#
# Each stack is trained in curriculum order. After each stack:
#   1. The stack is frozen and saved to disk
#   2. All previous domains are evaluated for forgetting
#   3. If any domain's loss degrades > 0.03, training pauses
#
# Usage:
#   ./scripts/micro_kiki/train_all_stacks.sh
#   ./scripts/micro_kiki/train_all_stacks.sh --resume-from 5
#   ./scripts/micro_kiki/train_all_stacks.sh --dry-run
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$PROJECT_ROOT/configs/micro_kiki/brainstacks.yaml"
LOG_DIR="$PROJECT_ROOT/output/micro-kiki/logs"

# Parse arguments
RESUME_FROM=1
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Curriculum order (must match brainstacks.yaml)
DOMAINS=(
    chat-fr
    reasoning
    python
    typescript
    cpp
    rust
    html-css
    shell
    sql
    yaml-json
    docker
    kicad-dsl
    spice
    lua-upy
    embedded
    stm32
    iot
    freecad
    platformio
    power
    emc
    dsp
    spice-sim
    electronics
    kicad-pcb
    web-frontend
    web-backend
    music-audio
    devops
    llm-orch
    math
    security
)

TOTAL=${#DOMAINS[@]}
echo "================================================================"
echo "Brainstacks — 32 MoE-LoRA Stack Training"
echo "================================================================"
echo "Config:       $CONFIG"
echo "Total stacks: $TOTAL"
echo "Resume from:  $RESUME_FROM"
echo "Dry run:      $DRY_RUN"
echo "================================================================"

mkdir -p "$LOG_DIR"

# Check data directories exist
MISSING=0
for domain in "${DOMAINS[@]}"; do
    data_dir="$PROJECT_ROOT/data/micro-kiki/$domain"
    if [ ! -f "$data_dir/train.jsonl" ]; then
        echo "MISSING: $data_dir/train.jsonl"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING domain datasets are missing."
    echo "Run Plan 1 (Data Pipeline) first to generate all datasets."
    exit 1
fi

echo ""
echo "All $TOTAL domain datasets found."
echo ""

# Train each stack
FAILED=0
for i in $(seq 1 $TOTAL); do
    domain="${DOMAINS[$((i-1))]}"

    if [ "$i" -lt "$RESUME_FROM" ]; then
        echo "[${i}/${TOTAL}] Skipping $domain (before resume point)"
        continue
    fi

    echo ""
    echo "================================================================"
    echo "[${i}/${TOTAL}] Training stack: $domain"
    echo "================================================================"

    LOG_FILE="$LOG_DIR/${i}-${domain}.log"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would run: python train_stack.py --domain $domain --stack-index $i"
        continue
    fi

    START_TIME=$(date +%s)

    python "$SCRIPT_DIR/train_stack.py" \
        --config "$CONFIG" \
        --domain "$domain" \
        --stack-index "$i" \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo ""
        echo "ERROR: Stack $domain failed (exit code $EXIT_CODE)"
        echo "Log: $LOG_FILE"
        FAILED=$((FAILED + 1))
        echo ""
        echo "To resume from this point:"
        echo "  ./scripts/micro_kiki/train_all_stacks.sh --resume-from $i"
        exit 1
    fi

    MINUTES=$((DURATION / 60))
    echo ""
    echo "[${i}/${TOTAL}] $domain complete in ${MINUTES}m"

    # Verify the stack was saved
    STACK_DIR="$PROJECT_ROOT/output/micro-kiki/stacks/$domain"
    if [ ! -f "$STACK_DIR/adapters.safetensors" ]; then
        echo "ERROR: Stack $domain did not produce adapters.safetensors"
        exit 1
    fi

    STACK_SIZE=$(du -sh "$STACK_DIR" | cut -f1)
    echo "  Stack size: $STACK_SIZE"
done

echo ""
echo "================================================================"
echo "ALL $TOTAL STACKS TRAINED SUCCESSFULLY"
echo "================================================================"
echo ""
echo "Output: $PROJECT_ROOT/output/micro-kiki/stacks/"
echo ""
echo "Next steps:"
echo "  1. Run full forgetting evaluation:"
echo "     python scripts/micro_kiki/eval_stack.py --config $CONFIG --all"
echo "  2. Proceed to Plan 3 (Meta-Router Training)"
echo ""

# Final comprehensive eval
echo "Running final evaluation..."
python "$SCRIPT_DIR/eval_stack.py" --config "$CONFIG" --all
```

- [ ] **Step 3: Make the orchestrator executable**

Run: `chmod +x /Users/clems/KIKI-Mac_tunner/scripts/micro_kiki/train_all_stacks.sh`

- [ ] **Step 4: Verify dry-run mode works**

Run: `cd /Users/clems/KIKI-Mac_tunner && ./scripts/micro_kiki/train_all_stacks.sh --dry-run 2>&1 | head -20`
Expected: Lines showing `[DRY RUN] Would run:` for each domain (or an error about missing datasets, which is expected before Plan 1 completes).

- [ ] **Step 5: Commit**

```bash
git add scripts/micro_kiki/__init__.py scripts/micro_kiki/train_all_stacks.sh
git commit -m "feat: add 32-stack orchestrator with resume and forgetting checks"
```

---

## Self-Review

**Spec coverage:**
- MoE-LoRA (4 experts, rank 16, top-2): Task 2
- Null-space projection (ns_top_k_dirs=32, randomized SVD): Task 3
- Residual boosting (1-2 rounds): Task 4
- Single stack training (SFT + boost + freeze + offload): Task 5
- Forgetting checks (delta < 0.03): Task 5 + Task 6
- Full 32-stack sequential orchestrator: Task 7
- Curriculum order (chat-fr first, security last): Task 1 config + Task 7
- h_dim=3072, 7 projections: Task 1 config
- ~150 Mo/stack, ~500 steps/domain: Task 1 config params
- rsLoRA scaling: Task 2 (LoRAExpert)

**Placeholder scan:** No TBD, TODO, "implement later", or "similar to Task N" found. All code blocks are complete.

**Type consistency verified:**
- `LoRAExpert` / `MoELoRALayer` / `apply_moe_lora` / `collect_moe_lora_layers` used consistently across Task 2, 3, 4, 5
- `compute_null_space_projector` / `build_projectors_for_stack` / `project_gradient` used consistently across Task 3, 4, 5
- `extract_moe_lora_state_dict` / `freeze_and_save_stack` / `load_domain_dataset` / `evaluate_domain` used consistently in Task 5, 6
- Config key paths (`config["moe_lora"]`, `config["null_space"]`, etc.) match the YAML structure in Task 1
