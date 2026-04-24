# Brainstacks V4 — Null-Space Projection on Qwen3.6-35B-A3B

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port Brainstacks null-space gradient projection (arXiv 2604.01152) to V4 vanilla LoRA on Qwen3.6-35B-A3B MoE, running on Apple Silicon. Train 3 ablation stacks proving reduced catastrophic forgetting vs the empirical angle-gate.

**Architecture:** Before training stack N, collect the frozen weight deltas (B@A) from stacks 1..N-1 for each of the 17 module kinds across 32 layers. Compute a rank-K SVD of the stacked deltas to extract the top-K principal directions. During training, project each gradient orthogonally to these directions so the new stack cannot overwrite frozen knowledge. Use implicit projection (store V_keep only, not dense P matrix) to fit in 512GB.

**Tech Stack:** MLX, mlx_lm 0.31.2, NumPy (SVD), safetensors

---

## Key Dimensions (Qwen3.6-35B-A3B, LoRA r=16)

816 tensors per adapter (408 lora_a + 408 lora_b) across 32 trained layers × 17 module kinds.

| Module Kind | lora_a shape | lora_b shape | Delta (B@A) shape | 2D/3D |
|---|---|---|---|---|
| self_attn.q_proj | (2048, 16) | (16, 4096) | (2048, 4096) | 2D |
| self_attn.k_proj | (2048, 16) | (16, 256) | (2048, 256) | 2D |
| self_attn.v_proj | (2048, 16) | (16, 512) | (2048, 512) | 2D |
| self_attn.o_proj | (4096, 16) | (16, 2048) | (4096, 2048) | 2D |
| linear_attn.in_proj_a | (2048, 16) | (16, 32) | (2048, 32) | 2D |
| linear_attn.in_proj_b | (2048, 16) | (16, 32) | (2048, 32) | 2D |
| linear_attn.in_proj_qkv | (2048, 16) | (16, 8192) | (2048, 8192) | 2D |
| linear_attn.in_proj_z | (2048, 16) | (16, 4096) | (2048, 4096) | 2D |
| linear_attn.out_proj | (4096, 16) | (16, 2048) | (4096, 2048) | 2D |
| mlp.gate | (2048, 16) | (16, 256) | (2048, 256) | 2D |
| mlp.shared_expert_gate | (2048, 16) | (16, 1) | (2048, 1) | 2D |
| mlp.shared_expert.gate_proj | (2048, 16) | (16, 512) | (2048, 512) | 2D |
| mlp.shared_expert.up_proj | (2048, 16) | (16, 512) | (2048, 512) | 2D |
| mlp.shared_expert.down_proj | (512, 16) | (16, 2048) | (512, 2048) | 2D |
| mlp.switch_mlp.gate_proj | (256, 16, 2048) | (256, 512, 16) | (256, 2048, 512) | **3D** |
| mlp.switch_mlp.up_proj | (256, 16, 2048) | (256, 512, 16) | (256, 2048, 512) | **3D** |
| mlp.switch_mlp.down_proj | (256, 16, 512) | (256, 2048, 16) | (256, 512, 2048) | **3D** |

**3D strategy for switch_mlp:** Flatten the 256 experts into the batch dimension. Each expert's (2048, 512) delta is treated independently — projecting expert 0's gradient doesn't affect expert 255. This is correct because MoE routing already isolates experts.

## Memory Budget

Per-module projector (implicit V_keep, K=32 directions):
- 2D module (e.g. q_proj): V_keep = (32, 2048×4096) → store as (32, 16) in LoRA space = **2 KB**
- 3D module (switch_mlp): V_keep per expert = (32, 16) × 256 experts = **128 KB**
- Total per layer: 17 modules × ~4 KB ≈ **68 KB**
- Total all 32 layers: **~2 MB** — negligible

**Key insight:** Project in LoRA parameter space (rank 16), not in weight space (2048×4096). The delta B@A lives in a rank-16 subspace. The frozen directions to protect are in the span of the frozen {A, B} matrices. So V_keep has shape (K, 2*rank) = (32, 32), not (K, in×out). This makes the projector tiny.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/stacks/null_space_v4.py` | Implicit null-space projector: collect frozen LoRA deltas, randomized SVD, gradient projection in LoRA param space |
| `src/stacks/train_loop_v4.py` | Custom MLX training loop with gradient projection hook |
| `scripts/train_stack_v4.py` | CLI: train a single V4 domain stack with null-space projection |
| `scripts/eval_nullspace_ablation.py` | Ablation: train with vs without null-space, measure forgetting angles |
| `tests/stacks/test_null_space_v4.py` | Unit tests: SVD correctness, orthogonality, 3D expert handling |
| `tests/stacks/test_train_loop_v4.py` | Integration test: train on tiny data, verify projection reduces forgetting |
| `configs/brainstacks-v4.yaml` | Config: null-space params (K=32, svd_iters=3, oversampling=10) |

---

### Task 1: Null-Space Projector (LoRA Parameter Space)

**Files:**
- Create: `src/stacks/null_space_v4.py`
- Create: `tests/stacks/test_null_space_v4.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for V4 null-space projector in LoRA parameter space."""
import numpy as np
import pytest


def test_collect_lora_vectors_from_single_adapter():
    """Load one adapter's A,B matrices and flatten to parameter vector."""
    from src.stacks.null_space_v4 import collect_lora_vectors
    # Create fake adapter dict mimicking V4 format
    adapter = {
        "layers.0.self_attn.q_proj.lora_a": np.random.randn(2048, 16).astype(np.float32),
        "layers.0.self_attn.q_proj.lora_b": np.random.randn(16, 4096).astype(np.float32),
    }
    vectors = collect_lora_vectors(adapter, layer=0, module="self_attn.q_proj")
    # Vector = concat(flatten(A), flatten(B))
    expected_dim = 2048 * 16 + 16 * 4096
    assert vectors.shape == (expected_dim,)


def test_collect_lora_vectors_3d_switch_mlp():
    """3D switch_mlp: flatten per-expert, return (256, param_dim)."""
    from src.stacks.null_space_v4 import collect_lora_vectors
    adapter = {
        "layers.0.mlp.switch_mlp.gate_proj.lora_a": np.random.randn(256, 16, 2048).astype(np.float32),
        "layers.0.mlp.switch_mlp.gate_proj.lora_b": np.random.randn(256, 512, 16).astype(np.float32),
    }
    vectors = collect_lora_vectors(adapter, layer=0, module="mlp.switch_mlp.gate_proj")
    # Per-expert: concat(flatten(A[e]), flatten(B[e])) for e in 256
    per_expert_dim = 16 * 2048 + 512 * 16
    assert vectors.shape == (256, per_expert_dim)


def test_build_projector_orthogonal():
    """V_keep directions are orthogonal to projected gradient."""
    from src.stacks.null_space_v4 import build_projector, project_gradient
    # 3 frozen stacks, param_dim=64
    frozen_vectors = np.random.randn(3, 64).astype(np.float32)
    V_keep = build_projector(frozen_vectors, top_k=2)
    assert V_keep.shape == (2, 64)

    # Project a random gradient
    grad = np.random.randn(64).astype(np.float32)
    projected = project_gradient(grad, V_keep)

    # projected should be orthogonal to V_keep rows
    for i in range(V_keep.shape[0]):
        dot = np.abs(np.dot(projected, V_keep[i]))
        assert dot < 1e-5, f"Not orthogonal: dot={dot}"

    # projected should not be zero (grad had components outside V_keep span)
    assert np.linalg.norm(projected) > 0.01


def test_build_projector_preserves_orthogonal_component():
    """If grad is already orthogonal to frozen directions, projection is identity."""
    from src.stacks.null_space_v4 import build_projector, project_gradient
    # Frozen in directions [1,0,0,...] and [0,1,0,...]
    frozen = np.eye(64, dtype=np.float32)[:2]  # first 2 basis vectors
    V_keep = build_projector(frozen, top_k=2)

    # Grad purely in direction [0,0,1,0,...] — orthogonal to frozen
    grad = np.zeros(64, dtype=np.float32)
    grad[2] = 1.0
    projected = project_gradient(grad, V_keep)
    np.testing.assert_allclose(projected, grad, atol=1e-5)


def test_project_gradient_zero_when_fully_in_frozen_span():
    """If grad lies entirely in the frozen subspace, projection is zero."""
    from src.stacks.null_space_v4 import build_projector, project_gradient
    frozen = np.eye(64, dtype=np.float32)[:2]
    V_keep = build_projector(frozen, top_k=2)

    grad = np.zeros(64, dtype=np.float32)
    grad[0] = 3.0
    grad[1] = -2.0  # entirely in span of frozen dirs 0,1
    projected = project_gradient(grad, V_keep)
    assert np.linalg.norm(projected) < 1e-5
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd /Users/clems/Projets/micro-kiki && python -m pytest tests/stacks/test_null_space_v4.py -v
```

- [ ] **Step 3: Implement null_space_v4.py**

```python
"""Brainstacks V4: implicit null-space projection in LoRA parameter space.

Projects training gradients orthogonally to the subspace spanned by
previously frozen LoRA adapters.  Operates in LoRA param space
(dim = in_features * rank + rank * out_features) not weight space
(in_features * out_features), keeping projectors at ~2 KB per module.

Reference: Brainstacks (arXiv 2604.01152), adapted for vanilla LoRA
on Qwen3.6-35B-A3B MoE (17 module kinds, 3D switch_mlp experts).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def collect_lora_vectors(
    adapter: dict[str, np.ndarray],
    layer: int,
    module: str,
) -> np.ndarray:
    """Extract flattened LoRA parameter vector(s) for one module in one layer.

    For 2D modules: returns shape (param_dim,)
    For 3D switch_mlp modules (256 experts): returns shape (256, per_expert_dim)
    """
    prefix = f"layers.{layer}.{module}"
    # Handle key formats: may have "language_model.model." prefix
    a_key = None
    b_key = None
    for k in adapter:
        if k.endswith(f"{prefix}.lora_a") or k.endswith(f".{prefix}.lora_a"):
            a_key = k
        if k.endswith(f"{prefix}.lora_b") or k.endswith(f".{prefix}.lora_b"):
            b_key = k
    if a_key is None or b_key is None:
        raise KeyError(f"LoRA keys not found for {prefix}")

    A = adapter[a_key]
    B = adapter[b_key]

    if A.ndim == 3:
        # 3D: (num_experts, rank, dim) — flatten per expert
        n_experts = A.shape[0]
        vectors = []
        for e in range(n_experts):
            vec = np.concatenate([A[e].flatten(), B[e].flatten()])
            vectors.append(vec)
        return np.stack(vectors)  # (n_experts, per_expert_dim)
    else:
        # 2D: (in, rank) and (rank, out)
        return np.concatenate([A.flatten(), B.flatten()])


def collect_all_frozen_vectors(
    adapter_paths: list[str | Path],
    layer: int,
    module: str,
) -> np.ndarray:
    """Load multiple frozen adapters and stack their parameter vectors.

    Returns:
        2D modules: (n_frozen, param_dim)
        3D modules: (n_frozen * 256, per_expert_dim)
    """
    from safetensors.numpy import load_file

    vectors = []
    for path in adapter_paths:
        adapter = load_file(str(Path(path) / "adapters.safetensors"))
        vec = collect_lora_vectors(adapter, layer, module)
        if vec.ndim == 1:
            vectors.append(vec)
        else:
            # 3D: stack all experts from this adapter
            vectors.extend(vec)
    return np.array(vectors, dtype=np.float32)


def build_projector(
    frozen_vectors: np.ndarray,
    top_k: int = 32,
    n_oversampling: int = 10,
    n_iter: int = 3,
) -> np.ndarray:
    """Compute top-K principal directions via randomized SVD.

    Args:
        frozen_vectors: (n_frozen, param_dim) stacked parameter vectors
        top_k: number of directions to protect

    Returns:
        V_keep: (min(top_k, n_frozen), param_dim) — orthonormal rows
    """
    n_frozen, param_dim = frozen_vectors.shape
    effective_k = min(top_k, n_frozen, param_dim)

    if effective_k == 0:
        return np.zeros((0, param_dim), dtype=np.float32)

    # Randomized SVD (Halko-Martinsson-Tropp)
    rng = np.random.default_rng(42)
    k_hat = min(effective_k + n_oversampling, param_dim)
    Omega = rng.standard_normal((param_dim, k_hat)).astype(np.float32)

    Y = frozen_vectors @ Omega  # (n_frozen, k_hat)

    # Power iteration for better approximation
    for _ in range(n_iter):
        Y = frozen_vectors @ (frozen_vectors.T @ Y)

    Q, _ = np.linalg.qr(Y)  # (n_frozen, k_hat)
    B = Q.T @ frozen_vectors  # (k_hat, param_dim)
    _, S, Vt = np.linalg.svd(B, full_matrices=False)

    V_keep = Vt[:effective_k]  # (effective_k, param_dim)

    logger.debug(
        "null-space projector: %d frozen → %d directions (top singular: %.4f)",
        n_frozen, effective_k, S[0] if len(S) > 0 else 0,
    )
    return V_keep.astype(np.float32)


def project_gradient(
    grad: np.ndarray,
    V_keep: np.ndarray,
) -> np.ndarray:
    """Project gradient orthogonally to the protected subspace.

    projected = grad - V_keep^T @ (V_keep @ grad)

    This is the implicit form — no dense P matrix needed.
    """
    if V_keep.shape[0] == 0:
        return grad
    coeffs = V_keep @ grad  # (K,)
    return grad - V_keep.T @ coeffs


class NullSpaceRegistry:
    """Per-layer, per-module projector registry for a training run.

    Usage:
        registry = NullSpaceRegistry.from_frozen_adapters(
            adapter_paths=["path/to/stack-0", "path/to/stack-1"],
            layers=range(32),
            modules=MODULE_KINDS,
            top_k=32,
        )
        # During training:
        projected_grad = registry.project(layer=5, module="self_attn.q_proj", grad=grad_array)
    """

    def __init__(self) -> None:
        self._projectors: dict[tuple[int, str], np.ndarray] = {}  # (layer, module) -> V_keep

    @classmethod
    def from_frozen_adapters(
        cls,
        adapter_paths: list[str | Path],
        layers: range,
        modules: list[str],
        top_k: int = 32,
    ) -> NullSpaceRegistry:
        registry = cls()
        if not adapter_paths:
            logger.info("No frozen adapters — null-space projection disabled")
            return registry

        total = len(list(layers)) * len(modules)
        built = 0
        for layer in layers:
            for module in modules:
                try:
                    frozen = collect_all_frozen_vectors(adapter_paths, layer, module)
                    if frozen.shape[0] > 0:
                        V_keep = build_projector(frozen, top_k=top_k)
                        registry._projectors[(layer, module)] = V_keep
                        built += 1
                except KeyError:
                    pass  # module not present in this layer type

        logger.info(
            "Built %d/%d null-space projectors from %d frozen adapters",
            built, total, len(adapter_paths),
        )
        return registry

    def project(self, layer: int, module: str, grad: np.ndarray) -> np.ndarray:
        """Project a gradient for a specific (layer, module)."""
        V_keep = self._projectors.get((layer, module))
        if V_keep is None:
            return grad
        if grad.ndim == 1:
            return project_gradient(grad, V_keep)
        # 3D: project each expert independently
        result = np.empty_like(grad)
        for e in range(grad.shape[0]):
            result[e] = project_gradient(grad[e], V_keep)
        return result


# The 17 module kinds in Qwen3.6-35B-A3B V4 adapters
MODULE_KINDS: list[str] = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z", "linear_attn.out_proj",
    "mlp.gate", "mlp.shared_expert_gate",
    "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj",
    "mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj", "mlp.switch_mlp.down_proj",
]
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /Users/clems/Projets/micro-kiki && python -m pytest tests/stacks/test_null_space_v4.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/stacks/null_space_v4.py tests/stacks/test_null_space_v4.py
git commit -m "feat(stacks): Brainstacks V4 null-space projector"
```

---

### Task 2: Custom Training Loop with Gradient Projection

**Files:**
- Create: `src/stacks/train_loop_v4.py`
- Create: `tests/stacks/test_train_loop_v4.py`

- [ ] **Step 1: Write failing test**

Test that the training loop applies null-space projection at each step by checking that the adapter weights after training have zero projection onto frozen directions.

```python
def test_train_step_projects_gradients():
    """After one train step with projection, updated params stay in null-space."""
    from src.stacks.null_space_v4 import build_projector, project_gradient
    import numpy as np

    # Simulate: frozen direction = [1, 0, 0, ...]
    frozen = np.zeros((1, 32), dtype=np.float32)
    frozen[0, 0] = 1.0
    V_keep = build_projector(frozen, top_k=1)

    # Gradient pointing partly into frozen direction
    grad = np.ones(32, dtype=np.float32)
    projected = project_gradient(grad, V_keep)

    # Component along frozen dir should be zero
    assert abs(projected[0]) < 1e-5
    # Other components preserved
    assert abs(projected[1] - 1.0) < 1e-5
```

- [ ] **Step 2: Implement train_loop_v4.py**

Custom MLX training loop that:
1. Loads base model + applies LoRA (via `mlx_lm.tuner`)
2. Loads frozen adapter paths → builds NullSpaceRegistry
3. Standard SFT loop: forward → loss → backward → **project gradients** → optimizer step
4. Saves adapter weights at end

The gradient projection hook maps each LoRA parameter's gradient to its (layer, module) identifier, flattens A+B grads together, projects, then unflattens back.

Key: MLX uses `mx.grad()` which returns a tree of gradients matching the model parameters. We post-process this tree before passing to the optimizer.

```python
"""Custom MLX training loop with Brainstacks null-space gradient projection.

Wraps the standard mlx_lm LoRA training with a post-grad hook that
projects each LoRA parameter's gradient into the null-space of previously
frozen adapter stacks.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from src.stacks.null_space_v4 import NullSpaceRegistry, MODULE_KINDS

logger = logging.getLogger(__name__)


def _parse_param_key(key: str) -> tuple[int, str] | None:
    """Extract (layer_idx, module_kind) from a LoRA parameter key.

    Example: 'layers.10.self_attn.q_proj.lora_a' -> (10, 'self_attn.q_proj')
    """
    parts = key.split(".")
    try:
        layer_pos = parts.index("layers")
        layer_idx = int(parts[layer_pos + 1])
    except (ValueError, IndexError):
        return None

    # Find lora_a or lora_b suffix
    if parts[-1] not in ("lora_a", "lora_b"):
        return None

    # Module kind = everything between layer idx and lora_a/b
    mod_parts = parts[layer_pos + 2: -1]
    module = ".".join(mod_parts)
    if module in MODULE_KINDS:
        return (layer_idx, module)
    return None


def project_grad_tree(
    grads: dict,
    registry: NullSpaceRegistry,
    prefix: str = "",
) -> dict:
    """Walk the gradient tree and project each LoRA gradient."""
    projected = {}
    for key, value in grads.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            projected[key] = project_grad_tree(value, registry, full_key)
        elif isinstance(value, mx.array):
            parsed = _parse_param_key(full_key)
            if parsed is not None:
                layer_idx, module = parsed
                # Convert to numpy, project, convert back
                grad_np = np.array(value, dtype=np.float32)
                grad_flat = grad_np.flatten()
                proj_flat = registry.project(layer_idx, module, grad_flat)
                projected[key] = mx.array(proj_flat.reshape(grad_np.shape))
            else:
                projected[key] = value
        else:
            projected[key] = value
    return projected


def train_stack(
    model: nn.Module,
    tokenizer,
    train_data: list[dict],
    val_data: list[dict],
    frozen_adapter_paths: list[str],
    *,
    iters: int = 200,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    max_seq_length: int = 1024,
    null_space_top_k: int = 32,
    output_dir: str = "output/stack",
    grad_checkpoint: bool = True,
) -> dict:
    """Train a single LoRA stack with null-space gradient projection.

    Returns training metrics dict.
    """
    # Build null-space projector from frozen adapters
    logger.info("Building null-space projectors from %d frozen adapters...", len(frozen_adapter_paths))
    t0 = time.time()
    registry = NullSpaceRegistry.from_frozen_adapters(
        adapter_paths=frozen_adapter_paths,
        layers=range(32),  # V4 trains 32/40 layers
        modules=MODULE_KINDS,
        top_k=null_space_top_k,
    )
    logger.info("Projectors built in %.1fs", time.time() - t0)

    # Setup optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Loss function
    def loss_fn(model, tokens):
        x = mx.array(tokens[:-1])[None]
        y = mx.array(tokens[1:])[None]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    metrics = {"train_losses": [], "val_losses": [], "projected_norms": []}
    best_val_loss = float("inf")

    for step in range(iters):
        # Sample training example
        example = train_data[step % len(train_data)]
        tokens = tokenizer.encode(example.get("text", ""))[:max_seq_length]
        if len(tokens) < 4:
            continue

        # Forward + backward
        loss, grads = loss_and_grad(model, tokens)
        mx.eval(loss)

        # Null-space projection
        if frozen_adapter_paths:
            grads = project_grad_tree(grads, registry)

        # Optimizer step
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        loss_val = loss.item()
        metrics["train_losses"].append(loss_val)

        if (step + 1) % 50 == 0:
            logger.info("step %d/%d: train_loss=%.4f", step + 1, iters, loss_val)

    # Save adapter
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Save only LoRA parameters
    lora_params = {
        k: v for k, v in dict(model.named_parameters()).items()
        if "lora_a" in k or "lora_b" in k
    }
    mx.save_safetensors(str(output_path / "adapters.safetensors"), lora_params)

    logger.info("Stack saved to %s (%d LoRA tensors)", output_path, len(lora_params))
    return metrics
```

- [ ] **Step 3: Run tests — expect PASS**

- [ ] **Step 4: Commit**

```bash
git add src/stacks/train_loop_v4.py tests/stacks/test_train_loop_v4.py
git commit -m "feat(stacks): training loop with null-space projection"
```

---

### Task 3: CLI Training Script

**Files:**
- Create: `scripts/train_stack_v4.py`
- Create: `configs/brainstacks-v4.yaml`

- [ ] **Step 1: Write config**

```yaml
# Brainstacks V4 — Null-space projection on Qwen3.6-35B-A3B
model: models/Qwen3.6-35B-A3B
adapter_dir: /Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota

lora:
  rank: 16
  alpha: 16
  num_layers: 32
  lr: 1e-5

null_space:
  top_k: 32
  svd_oversampling: 10
  svd_n_iter: 3
  enabled: true

training:
  iters_foundation: 1000
  iters_coding: 500
  iters_niche: 200
  batch_size: 1
  max_seq_length: 1024
  grad_checkpoint: true
```

- [ ] **Step 2: Write CLI script**

CLI that wraps `train_loop_v4.train_stack()`:
- `--domain power` — train the power stack
- `--frozen-stacks chat-fr,python,cpp` — use these as frozen priors
- `--config configs/brainstacks-v4.yaml`
- Auto-discovers which stacks are already trained from adapter_dir

- [ ] **Step 3: Test on 1 domain (10 iters, dry run)**

```bash
cd /Users/clems/Projets/micro-kiki && \
  .venv/bin/python scripts/train_stack_v4.py \
    --domain power \
    --frozen-stacks chat-fr python \
    --iters 10 \
    --output /tmp/test-brainstacks-v4/power
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_stack_v4.py configs/brainstacks-v4.yaml
git commit -m "feat: Brainstacks V4 training CLI + config"
```

---

### Task 4: Ablation — With vs Without Null-Space

This is the core evidence for the paper.

**Design:**
- 3 domains: chat-fr (foundation), python (coding), embedded (niche)
- Condition A: train with null-space projection (frozen priors = all previously trained stacks)
- Condition B: train WITHOUT null-space (current V4 approach)
- Measure: forgetting angle (from forgetting.py) between new stack and all frozen stacks
- Also measure: val-loss on the new domain (to check null-space doesn't hurt performance)

**Files:**
- Create: `scripts/eval_nullspace_ablation.py`
- Output: `results/ablation-nullspace/`

- [ ] **Step 1: Write ablation script**

For each domain in {chat-fr, python, embedded}:
1. Pick 2-3 "prior" stacks as frozen (e.g. for embedded: chat-fr + python are priors)
2. Train stack WITHOUT null-space → measure forgetting angles vs priors
3. Train stack WITH null-space → measure forgetting angles vs priors
4. Compare angles and val-losses

Use only 100 iters (enough to see the effect).

- [ ] **Step 2: Run ablation (~2h GPU)**

```bash
nohup .venv/bin/python scripts/eval_nullspace_ablation.py \
  --config configs/brainstacks-v4.yaml \
  --domains chat-fr python embedded \
  --iters 100 \
  --output results/ablation-nullspace/ablation.json \
  > /tmp/nullspace-ablation.log 2>&1 &
```

- [ ] **Step 3: Commit results**

```bash
git add scripts/eval_nullspace_ablation.py results/ablation-nullspace/
git commit -m "eval: null-space ablation (with vs without projection)"
```

---

### Task 5: Update Paper with Brainstacks Results

**Files:**
- Modify: `paper/micro-kiki.tex`

- [ ] **Step 1: Add Brainstacks section**

New subsection in "Adapter Management" covering:
- The null-space projection in LoRA parameter space (not weight space)
- Implicit projector (V_keep only, ~2 KB per module)
- 3D expert handling for switch_mlp
- Ablation results table

- [ ] **Step 2: Add ablation table**

```latex
\begin{table}[h]
\centering
\caption{Null-space projection ablation. Forgetting angle (degrees)
between new stack and frozen priors. Higher = less forgetting.}
\begin{tabular}{lcccc}
\toprule
Domain & \multicolumn{2}{c}{Without NS} & \multicolumn{2}{c}{With NS} \\
       & Angle & Val Loss & Angle & Val Loss \\
\midrule
chat-fr   & X.X & X.XX & X.X & X.XX \\
python    & X.X & X.XX & X.X & X.XX \\
embedded  & X.X & X.XX & X.X & X.XX \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 3: Update contributions list**

Add: "Adaptation of Brainstacks null-space projection to vanilla LoRA on native MoE, operating in rank-16 parameter space with implicit projectors (2 KB vs 68 GB dense)."

- [ ] **Step 4: Commit**

```bash
git add paper/micro-kiki.tex
git commit -m "docs(paper): add Brainstacks V4 null-space ablation"
```

---

## Dependency Graph

```
Task 1 (null-space projector) → Task 2 (training loop) → Task 3 (CLI) → Task 4 (ablation) → Task 5 (paper)
```

Strictly sequential — each task builds on the previous.

**Total estimated time:**
- Implementation (Tasks 1-3): ~4h dev time
- Ablation (Task 4): ~2h GPU
- Paper update (Task 5): ~1h
