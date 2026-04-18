# Text-JEPA mini for VQC Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the VQC router's sentence-transformers input embeddings with embeddings from a small Text-JEPA student encoder (MLP projector trained with a JEPA objective — L1 masked-prediction loss against an EMA teacher) and benchmark 35-class domain classification against baseline.

**Architecture:** Frozen `all-MiniLM-L6-v2` (384-d) → trainable 2-layer MLP student projector (384 → 128) + EMA teacher (same shape, no grad) + tiny MLP predictor (~1/10 params of student). Training uses multi-span masking of input tokens (contiguous 3–5 token blocks), L1 loss on masked-position targets only, stop-gradient on teacher. At inference the student outputs 128-d embeddings that feed the existing 6-qubit / 6-layer VQC router. A config flag (`use_text_jepa`) selects the embedding path so we can A/B test.

**Tech Stack:** Python 3.11, PyTorch 2.4 (CPU or MPS), `sentence-transformers>=3.0`, PennyLane (VQC), NumPy, pytest, uv.

**Out-of-scope parking lot (do NOT build):**
- Full V-JEPA 2 reproduction (video, 1B params, >1M hours corpus).
- Replacing or fine-tuning the LLM stack / 35B-A3B base.
- Multi-modal extension (vision, audio).
- Training the student on > 10k examples in this PoC (time-boxed to 3 days).
- Modifying the VQC circuit itself (qubits, layers, parameter-shift gradients).
- VICReg / Barlow Twins / contrastive objectives — collapse prevention relies on EMA + stop-gradient + asymmetry only.

**Success criterion:** VQC router accuracy with Text-JEPA input **≥ baseline accuracy** on held-out 35-class test set, OR **equal accuracy at 128-d latent vs 384-d baseline** (memory/compute win).
**Kill criterion:** accuracy drops > 5 points vs baseline, OR student embedding std collapses to `< 0.01` during training (monitor every epoch).

---

## File Structure

**New files:**
- `src/routing/text_jepa/__init__.py` — package init, public re-exports.
- `src/routing/text_jepa/masking.py` — `span_mask(seq_len, mask_ratio, min_span, max_span, rng)`.
- `src/routing/text_jepa/encoder.py` — `StudentEncoder` (MLP 384→256→128) and `TeacherEncoder` (EMA wrapper).
- `src/routing/text_jepa/predictor.py` — `Predictor` (tiny MLP 128→32→128).
- `src/routing/text_jepa/loss.py` — `masked_l1_loss(pred, target, mask)`.
- `src/routing/text_jepa/collapse.py` — `embedding_std(x)` + `CollapseMonitor`.
- `src/routing/text_jepa/trainer.py` — training loop (1 step + full epochs) calling everything above.
- `src/routing/text_jepa/dataset.py` — `load_domain_corpus(data_dir)` reading `data/final/<domain>/train.jsonl`.
- `src/routing/text_jepa/embed.py` — `TextJEPAEmbedder.embed(text) -> np.ndarray` for inference.
- `scripts/train_text_jepa.py` — CLI: train student + dump checkpoint to `models/text-jepa/student.pt`.
- `scripts/eval_text_jepa_vqc.py` — CLI: benchmark VQC with baseline vs Text-JEPA input, writes `results/text-jepa-vqc.json`.
- `results/text-jepa-vqc.md` — human-readable experiment notes (filled in Task 13).
- `configs/text_jepa.yaml` — training + inference config.

**Modified files:**
- `src/routing/hybrid_pipeline.py` — add `use_text_jepa: bool` to `HybridPipelineConfig`; swap embedder when true.
- `pyproject.toml` — add `torch` to `[embeddings]` optional deps if not already present there (already in `[train]`).

**New test files (mirror `src/` layout):**
- `tests/routing/text_jepa/__init__.py`
- `tests/routing/text_jepa/test_masking.py`
- `tests/routing/text_jepa/test_encoder.py`
- `tests/routing/text_jepa/test_predictor.py`
- `tests/routing/text_jepa/test_loss.py`
- `tests/routing/text_jepa/test_collapse.py`
- `tests/routing/text_jepa/test_trainer.py`
- `tests/routing/text_jepa/test_dataset.py`
- `tests/routing/text_jepa/test_embed.py`
- `tests/routing/text_jepa/test_hybrid_pipeline_integration.py`
- `tests/scripts/test_eval_text_jepa_vqc.py`

All training runs on CPU or MPS. No GPU required for this PoC.

---

## Task 1: Scaffolding — package skeleton

**Files:**
- Create: `src/routing/text_jepa/__init__.py`
- Create: `tests/routing/text_jepa/__init__.py`
- Create: `tests/routing/text_jepa/test_scaffold.py`

- [ ] **Step 1: Write the failing test**

Create `tests/routing/text_jepa/test_scaffold.py`:

```python
"""Scaffolding smoke test — package is importable."""
from __future__ import annotations


def test_package_imports():
    import src.routing.text_jepa as tj  # noqa: F401
    assert hasattr(tj, "__all__")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_scaffold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.routing.text_jepa'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/routing/text_jepa/__init__.py`:

```python
"""Text-JEPA: JEPA-style masked prediction for VQC router embeddings.

V-JEPA 2 adapted for text — frozen sentence-transformer backbone, trainable
student/EMA-teacher MLP projectors, tiny predictor, L1 masked loss.
"""
from __future__ import annotations

__all__ = [
    "span_mask",
    "StudentEncoder",
    "TeacherEncoder",
    "Predictor",
    "masked_l1_loss",
    "CollapseMonitor",
    "TextJEPAEmbedder",
]
```

Create `tests/routing/text_jepa/__init__.py` (empty file, marker only):

```python
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_scaffold.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/routing/text_jepa/__init__.py tests/routing/text_jepa/__init__.py tests/routing/text_jepa/test_scaffold.py
git commit -m "feat(text-jepa): package scaffolding"
```

---

## Task 2: Span masking utility

**Files:**
- Create: `src/routing/text_jepa/masking.py`
- Test: `tests/routing/text_jepa/test_masking.py`

The V-JEPA paper shows multi-block / span masking is critical. Random single-token masking collapses. We use contiguous spans of 3–5 tokens covering ~40% of the sequence.

- [ ] **Step 1: Write the failing test**

Create `tests/routing/text_jepa/test_masking.py`:

```python
"""Span-masking tests — contiguous token blocks, no random single-token masks."""
from __future__ import annotations

import numpy as np
import pytest


def test_span_mask_returns_bool_array_of_seq_len():
    from src.routing.text_jepa.masking import span_mask

    rng = np.random.default_rng(0)
    mask = span_mask(seq_len=32, mask_ratio=0.4, min_span=3, max_span=5, rng=rng)

    assert mask.dtype == np.bool_
    assert mask.shape == (32,)


def test_span_mask_ratio_within_tolerance():
    from src.routing.text_jepa.masking import span_mask

    rng = np.random.default_rng(1)
    mask = span_mask(seq_len=128, mask_ratio=0.4, min_span=3, max_span=5, rng=rng)

    ratio = mask.mean()
    assert 0.30 <= ratio <= 0.55  # allow ±0.15 — short sequences bounce


def test_span_mask_uses_contiguous_spans_only():
    """No isolated masked token surrounded by unmasked — each mask run >= min_span."""
    from src.routing.text_jepa.masking import span_mask

    rng = np.random.default_rng(2)
    mask = span_mask(seq_len=64, mask_ratio=0.4, min_span=3, max_span=5, rng=rng)

    # Find runs of True
    runs = []
    current = 0
    for bit in mask:
        if bit:
            current += 1
        elif current > 0:
            runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)

    assert all(r >= 3 for r in runs), f"Found span shorter than 3: {runs}"


def test_span_mask_rejects_invalid_ratio():
    from src.routing.text_jepa.masking import span_mask

    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        span_mask(seq_len=32, mask_ratio=1.5, min_span=3, max_span=5, rng=rng)


def test_span_mask_seed_reproducible():
    from src.routing.text_jepa.masking import span_mask

    a = span_mask(seq_len=48, mask_ratio=0.4, min_span=3, max_span=5, rng=np.random.default_rng(42))
    b = span_mask(seq_len=48, mask_ratio=0.4, min_span=3, max_span=5, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_masking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.routing.text_jepa.masking'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/routing/text_jepa/masking.py`:

```python
"""Span masking for Text-JEPA — contiguous 3-5 token blocks."""
from __future__ import annotations

import numpy as np


def span_mask(
    seq_len: int,
    mask_ratio: float,
    min_span: int,
    max_span: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a boolean mask of shape (seq_len,) with contiguous spans.

    Args:
        seq_len: Sequence length.
        mask_ratio: Target fraction of tokens to mask in [0, 1].
        min_span: Minimum contiguous span length (inclusive).
        max_span: Maximum contiguous span length (inclusive).
        rng: NumPy random generator for reproducibility.

    Returns:
        Boolean ndarray, True where token is masked.
    """
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
    if min_span < 1 or max_span < min_span:
        raise ValueError(f"invalid span bounds: min={min_span}, max={max_span}")

    mask = np.zeros(seq_len, dtype=np.bool_)
    target_masked = int(mask_ratio * seq_len)
    masked = 0
    guard = 0
    max_guard = seq_len * 4

    while masked < target_masked and guard < max_guard:
        span_len = int(rng.integers(min_span, max_span + 1))
        span_len = min(span_len, seq_len)
        start = int(rng.integers(0, seq_len - span_len + 1))
        end = start + span_len

        # Only count new masks
        newly_masked = int((~mask[start:end]).sum())
        mask[start:end] = True
        masked += newly_masked
        guard += 1

    return mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_masking.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/routing/text_jepa/masking.py tests/routing/text_jepa/test_masking.py
git commit -m "feat(text-jepa): span masking utility"
```

---

## Task 3: Student encoder (MLP projector)

**Files:**
- Create: `src/routing/text_jepa/encoder.py`
- Test: `tests/routing/text_jepa/test_encoder.py`

Student = 2-layer MLP on frozen MiniLM output. Input dim 384, hidden 256, output 128.

- [ ] **Step 1: Write the failing test**

Create `tests/routing/text_jepa/test_encoder.py`:

```python
"""Tests for StudentEncoder and TeacherEncoder."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_student_forward_shape():
    from src.routing.text_jepa.encoder import StudentEncoder

    model = StudentEncoder(input_dim=384, hidden_dim=256, output_dim=128)
    x = torch.randn(4, 16, 384)  # (batch, seq_len, input_dim)
    out = model(x)
    assert out.shape == (4, 16, 128)


def test_student_has_trainable_params():
    from src.routing.text_jepa.encoder import StudentEncoder

    model = StudentEncoder(input_dim=384, hidden_dim=256, output_dim=128)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0


def test_student_output_is_finite():
    from src.routing.text_jepa.encoder import StudentEncoder

    model = StudentEncoder(input_dim=384, hidden_dim=256, output_dim=128)
    x = torch.randn(2, 8, 384)
    out = model(x)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.routing.text_jepa.encoder'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/routing/text_jepa/encoder.py`:

```python
"""Student encoder — trainable MLP projector on top of frozen MiniLM embeddings."""
from __future__ import annotations

import torch
from torch import nn


class StudentEncoder(nn.Module):
    """2-layer MLP: input_dim -> hidden_dim -> output_dim with GELU."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., input_dim) → (..., output_dim)."""
        return self.net(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_encoder.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/routing/text_jepa/encoder.py tests/routing/text_jepa/test_encoder.py
git commit -m "feat(text-jepa): student MLP encoder"
```

---

## Task 4: Teacher EMA wrapper

**Files:**
- Modify: `src/routing/text_jepa/encoder.py`
- Modify: `tests/routing/text_jepa/test_encoder.py`

Teacher = EMA copy of student with `requires_grad=False`. Momentum update `θ_teacher ← m*θ_teacher + (1-m)*θ_student`.

- [ ] **Step 1: Write the failing test**

Append to `tests/routing/text_jepa/test_encoder.py`:

```python
def test_teacher_initialised_from_student():
    from src.routing.text_jepa.encoder import StudentEncoder, TeacherEncoder

    student = StudentEncoder(input_dim=384, hidden_dim=256, output_dim=128)
    teacher = TeacherEncoder(student)

    for ps, pt in zip(student.parameters(), teacher.parameters()):
        torch.testing.assert_close(ps.data, pt.data)


def test_teacher_params_frozen():
    from src.routing.text_jepa.encoder import StudentEncoder, TeacherEncoder

    student = StudentEncoder()
    teacher = TeacherEncoder(student)
    for p in teacher.parameters():
        assert p.requires_grad is False


def test_teacher_ema_update_mixes_params():
    from src.routing.text_jepa.encoder import StudentEncoder, TeacherEncoder

    student = StudentEncoder()
    teacher = TeacherEncoder(student)

    # Perturb student
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.ones_like(p))

    teacher.update(student, momentum=0.9)

    # Teacher moved 10% of the way toward student
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        diff = (ps.data - pt.data).abs().mean().item()
        # student moved by ~1, teacher by ~0.1 (with m=0.9)
        assert 0.05 < diff < 0.95


def test_teacher_forward_stop_gradient():
    from src.routing.text_jepa.encoder import StudentEncoder, TeacherEncoder

    student = StudentEncoder()
    teacher = TeacherEncoder(student)
    x = torch.randn(2, 8, 384, requires_grad=True)

    out = teacher(x)
    assert out.requires_grad is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_encoder.py::test_teacher_initialised_from_student -v`
Expected: FAIL with `ImportError: cannot import name 'TeacherEncoder'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/routing/text_jepa/encoder.py`:

```python
class TeacherEncoder(nn.Module):
    """EMA copy of a StudentEncoder. All params frozen; forward runs under no_grad."""

    def __init__(self, student: StudentEncoder) -> None:
        super().__init__()
        import copy

        self.net = copy.deepcopy(student.net)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, student: StudentEncoder, momentum: float) -> None:
        """θ_teacher ← m·θ_teacher + (1-m)·θ_student."""
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        for pt, ps in zip(self.parameters(), student.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_encoder.py -v`
Expected: PASS (7 passed — 3 from Task 3 + 4 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/routing/text_jepa/encoder.py tests/routing/text_jepa/test_encoder.py
git commit -m "feat(text-jepa): EMA teacher wrapper"
```

---

## Task 5: Predictor (tiny MLP)

**Files:**
- Create: `src/routing/text_jepa/predictor.py`
- Test: `tests/routing/text_jepa/test_predictor.py`

V-JEPA 2 uses a small predictor (22M for 1B encoder, ~1/45). At this scale we use 1/10: student ~130k params → predictor ~13k (MLP 128→32→128).

- [ ] **Step 1: Write the failing test**

Create `tests/routing/text_jepa/test_predictor.py`:

```python
"""Tests for the tiny Predictor module."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_predictor_forward_shape():
    from src.routing.text_jepa.predictor import Predictor

    pred = Predictor(latent_dim=128, hidden_dim=32)
    x = torch.randn(4, 16, 128)
    out = pred(x)
    assert out.shape == (4, 16, 128)


def test_predictor_smaller_than_student():
    """Plan spec: predictor ~1/10 of student params."""
    from src.routing.text_jepa.encoder import StudentEncoder
    from src.routing.text_jepa.predictor import Predictor

    student = StudentEncoder(input_dim=384, hidden_dim=256, output_dim=128)
    pred = Predictor(latent_dim=128, hidden_dim=32)

    n_student = sum(p.numel() for p in student.parameters())
    n_pred = sum(p.numel() for p in pred.parameters())
    assert n_pred * 5 < n_student, f"predictor too large: {n_pred} vs student {n_student}"


def test_predictor_output_finite():
    from src.routing.text_jepa.predictor import Predictor

    pred = Predictor(latent_dim=128, hidden_dim=32)
    out = pred(torch.randn(2, 4, 128))
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_predictor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.routing.text_jepa.predictor'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/routing/text_jepa/predictor.py`:

```python
"""Tiny predictor — MLP 128 → 32 → 128. Kept small per V-JEPA 2 asymmetry recipe."""
from __future__ import annotations

import torch
from torch import nn


class Predictor(nn.Module):
    """Asymmetric narrow-bottleneck MLP predictor."""

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., latent_dim) → (..., latent_dim)."""
        return self.net(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_predictor.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/routing/text_jepa/predictor.py tests/routing/text_jepa/test_predictor.py
git commit -m "feat(text-jepa): tiny predictor"
```

---

## Task 6: L1 masked loss

**Files:**
- Create: `src/routing/text_jepa/loss.py`
- Test: `tests/routing/text_jepa/test_loss.py`

V-JEPA 2 loss: L1 on masked positions only. Targets come from stop-gradient teacher.

- [ ] **Step 1: Write the failing test**

Create `tests/routing/text_jepa/test_loss.py`:

```python
"""Tests for masked L1 loss — compare with known inputs."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_loss_ignores_unmasked_positions():
    from src.routing.text_jepa.loss import masked_l1_loss

    pred = torch.zeros(1, 4, 2)
    target = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]])
    mask = torch.tensor([[False, True, False, False]])  # only pos 1

    loss = masked_l1_loss(pred, target, mask)
    # mean(|0-2|, |0-2|) over masked = 2.0
    assert loss.item() == pytest.approx(2.0)


def test_loss_zero_when_pred_equals_target():
    from src.routing.text_jepa.loss import masked_l1_loss

    pred = torch.randn(2, 8, 16)
    target = pred.clone()
    mask = torch.ones(2, 8, dtype=torch.bool)

    loss = masked_l1_loss(pred, target, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_raises_when_no_masked_positions():
    from src.routing.text_jepa.loss import masked_l1_loss

    pred = torch.zeros(1, 4, 2)
    target = torch.zeros(1, 4, 2)
    mask = torch.zeros(1, 4, dtype=torch.bool)

    with pytest.raises(ValueError):
        masked_l1_loss(pred, target, mask)


def test_loss_differentiable_wrt_pred():
    from src.routing.text_jepa.loss import masked_l1_loss

    pred = torch.randn(1, 4, 2, requires_grad=True)
    target = torch.randn(1, 4, 2)
    mask = torch.tensor([[True, False, True, False]])

    loss = masked_l1_loss(pred, target, mask)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_loss.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.routing.text_jepa.loss'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/routing/text_jepa/loss.py`:

```python
"""L1 loss on masked token positions — V-JEPA 2 recipe."""
from __future__ import annotations

import torch


def masked_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """L1 loss averaged over masked positions only.

    Args:
        pred: (batch, seq_len, dim).
        target: (batch, seq_len, dim) — must be detached (teacher stop-gradient).
        mask: (batch, seq_len) bool, True where position is masked.

    Returns:
        Scalar loss.
    """
    if mask.sum() == 0:
        raise ValueError("mask has zero masked positions")

    # Apply mask along seq dim, broadcast over feature dim
    diff = (pred - target).abs()  # (batch, seq, dim)
    mask_f = mask.unsqueeze(-1).to(diff.dtype)
    masked_sum = (diff * mask_f).sum()
    n_masked_elems = mask_f.sum() * diff.size(-1)
    return masked_sum / n_masked_elems
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_loss.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/routing/text_jepa/loss.py tests/routing/text_jepa/test_loss.py
git commit -m "feat(text-jepa): masked L1 loss"
```

---

## Task 7: Collapse monitor

**Files:**
- Create: `src/routing/text_jepa/collapse.py`
- Test: `tests/routing/text_jepa/test_collapse.py`

Without VICReg we still need a collapse watchdog: if the student's output std drops below a threshold (default 0.01) for 2 consecutive checks → abort training.

- [ ] **Step 1: Write the failing test**

Create `tests/routing/text_jepa/test_collapse.py`:

```python
"""Tests for collapse monitor."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_embedding_std_zero_on_constant():
    from src.routing.text_jepa.collapse import embedding_std

    x = torch.ones(4, 8, 128)
    assert embedding_std(x) == pytest.approx(0.0)


def test_embedding_std_positive_on_random():
    from src.routing.text_jepa.collapse import embedding_std

    x = torch.randn(4, 8, 128)
    assert embedding_std(x) > 0.5


def test_collapse_monitor_triggers_after_threshold():
    from src.routing.text_jepa.collapse import CollapseMonitor

    mon = CollapseMonitor(floor=0.01, patience=2)
    assert mon.check(std=0.5) is False
    assert mon.check(std=0.005) is False  # 1 strike
    assert mon.check(std=0.003) is True  # 2 strikes → collapse


def test_collapse_monitor_resets_on_recovery():
    from src.routing.text_jepa.collapse import CollapseMonitor

    mon = CollapseMonitor(floor=0.01, patience=2)
    mon.check(std=0.005)  # 1 strike
    mon.check(std=0.5)  # recover
    assert mon.check(std=0.005) is False  # 1 strike again, not 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_collapse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.routing.text_jepa.collapse'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/routing/text_jepa/collapse.py`:

```python
"""Representation-collapse watchdog — abort training if student std floors."""
from __future__ import annotations

import torch


def embedding_std(x: torch.Tensor) -> float:
    """Mean over feature dims of token-wise std. Collapse detector signal."""
    # flatten batch+seq, keep features
    flat = x.reshape(-1, x.size(-1))
    return float(flat.std(dim=0).mean().item())


class CollapseMonitor:
    """Flag collapse when std stays below `floor` for `patience` consecutive checks."""

    def __init__(self, floor: float = 0.01, patience: int = 2) -> None:
        self.floor = floor
        self.patience = patience
        self._strikes = 0

    def check(self, std: float) -> bool:
        """Returns True if collapse detected (caller should abort training)."""
        if std < self.floor:
            self._strikes += 1
        else:
            self._strikes = 0
        return self._strikes >= self.patience
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_collapse.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/routing/text_jepa/collapse.py tests/routing/text_jepa/test_collapse.py
git commit -m "feat(text-jepa): collapse monitor"
```

---

## Task 8: Training loop — single step on synthetic data

**Files:**
- Create: `src/routing/text_jepa/trainer.py`
- Test: `tests/routing/text_jepa/test_trainer.py`

Single-step API: `TextJEPATrainer.step(token_embeddings)` does:
1. `mask = span_mask(seq_len, ...)`
2. `student_out = student(token_embeddings)` on masked seq (mask positions zeroed)
3. `pred_out = predictor(student_out)` on masked positions
4. `teacher_out = teacher(token_embeddings)` on full seq (stop-grad)
5. `loss = masked_l1_loss(pred_out, teacher_out, mask)`
6. Back-prop on student + predictor
7. EMA update teacher

- [ ] **Step 1: Write the failing test**

Create `tests/routing/text_jepa/test_trainer.py`:

```python
"""Trainer tests — single step on synthetic data should reduce loss."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_trainer_single_step_runs():
    from src.routing.text_jepa.trainer import TextJEPATrainer

    trainer = TextJEPATrainer(
        input_dim=384,
        latent_dim=128,
        hidden_dim=256,
        predictor_hidden=32,
        lr=1e-3,
        ema_momentum=0.99,
        mask_ratio=0.4,
        min_span=3,
        max_span=5,
        seed=0,
    )
    # Batch of 2 sequences, 16 tokens, 384-d
    tokens = torch.randn(2, 16, 384)
    loss = trainer.step(tokens)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_trainer_reduces_loss_on_fixed_batch():
    """Overfitting on a single batch must reduce loss monotonically-ish."""
    from src.routing.text_jepa.trainer import TextJEPATrainer

    trainer = TextJEPATrainer(
        input_dim=64,
        latent_dim=32,
        hidden_dim=64,
        predictor_hidden=16,
        lr=1e-2,
        ema_momentum=0.99,
        mask_ratio=0.4,
        min_span=3,
        max_span=5,
        seed=0,
    )
    torch.manual_seed(0)
    tokens = torch.randn(2, 16, 64)
    first = trainer.step(tokens).item()
    for _ in range(50):
        trainer.step(tokens)
    last = trainer.step(tokens).item()
    assert last < first, f"loss did not decrease: first={first} last={last}"


def test_trainer_ema_updates_teacher():
    from src.routing.text_jepa.trainer import TextJEPATrainer

    trainer = TextJEPATrainer(
        input_dim=64, latent_dim=32, hidden_dim=64, predictor_hidden=16,
        lr=1e-3, ema_momentum=0.9, mask_ratio=0.4, min_span=3, max_span=5, seed=0,
    )
    before = [p.detach().clone() for p in trainer.teacher.parameters()]
    tokens = torch.randn(2, 16, 64)
    for _ in range(5):
        trainer.step(tokens)
    after = list(trainer.teacher.parameters())
    moved = any(
        not torch.allclose(b, a.data) for b, a in zip(before, after)
    )
    assert moved, "teacher did not update via EMA"


def test_trainer_detects_collapse():
    """If student output std floors, collapse flag should raise."""
    from src.routing.text_jepa.trainer import TextJEPATrainer

    trainer = TextJEPATrainer(
        input_dim=64, latent_dim=32, hidden_dim=64, predictor_hidden=16,
        lr=1e-3, ema_momentum=0.99, mask_ratio=0.4, min_span=3, max_span=5, seed=0,
        collapse_floor=10.0,  # absurd floor to force collapse
        collapse_patience=1,
    )
    tokens = torch.randn(2, 16, 64)
    trainer.step(tokens)
    assert trainer.collapsed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.routing.text_jepa.trainer'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/routing/text_jepa/trainer.py`:

```python
"""Text-JEPA training loop — student + teacher EMA + predictor + L1 masked loss."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim

from src.routing.text_jepa.collapse import CollapseMonitor, embedding_std
from src.routing.text_jepa.encoder import StudentEncoder, TeacherEncoder
from src.routing.text_jepa.loss import masked_l1_loss
from src.routing.text_jepa.masking import span_mask
from src.routing.text_jepa.predictor import Predictor


class TextJEPATrainer:
    """One-step JEPA trainer. Call :meth:`step` repeatedly."""

    def __init__(
        self,
        input_dim: int = 384,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        predictor_hidden: int = 32,
        lr: float = 1e-3,
        ema_momentum: float = 0.99,
        mask_ratio: float = 0.4,
        min_span: int = 3,
        max_span: int = 5,
        collapse_floor: float = 0.01,
        collapse_patience: int = 2,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        self.student = StudentEncoder(input_dim, hidden_dim, latent_dim)
        self.teacher = TeacherEncoder(self.student)
        self.predictor = Predictor(latent_dim, predictor_hidden)

        params = list(self.student.parameters()) + list(self.predictor.parameters())
        self.optimizer = optim.AdamW(params, lr=lr)

        self.ema_momentum = ema_momentum
        self.mask_ratio = mask_ratio
        self.min_span = min_span
        self.max_span = max_span
        self._rng = np.random.default_rng(seed)

        self.monitor = CollapseMonitor(floor=collapse_floor, patience=collapse_patience)
        self.collapsed = False

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        """One training step.

        Args:
            tokens: (batch, seq_len, input_dim) — frozen-backbone token embeddings.

        Returns:
            Scalar loss tensor.
        """
        batch, seq_len, _ = tokens.shape

        # Shared span mask for the batch (V-JEPA 2: same mask within a batch row is fine;
        # we share across batch for simplicity in this PoC)
        mask_np = span_mask(
            seq_len=seq_len,
            mask_ratio=self.mask_ratio,
            min_span=self.min_span,
            max_span=self.max_span,
            rng=self._rng,
        )
        mask = torch.from_numpy(mask_np).unsqueeze(0).expand(batch, -1).contiguous()

        # Student sees masked input: zero-out masked positions
        mask_f = mask.unsqueeze(-1).to(tokens.dtype)
        masked_tokens = tokens * (1.0 - mask_f)

        student_latent = self.student(masked_tokens)  # (B, S, D)
        pred = self.predictor(student_latent)  # (B, S, D)

        with torch.no_grad():
            target = self.teacher(tokens).detach()

        loss = masked_l1_loss(pred, target, mask)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # EMA update
        self.teacher.update(self.student, momentum=self.ema_momentum)

        # Collapse check on student latent
        std = embedding_std(student_latent.detach())
        if self.monitor.check(std):
            self.collapsed = True

        return loss.detach()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_trainer.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/routing/text_jepa/trainer.py tests/routing/text_jepa/test_trainer.py
git commit -m "feat(text-jepa): training loop + collapse guard"
```

---

## Task 9: Domain corpus loader

**Files:**
- Create: `src/routing/text_jepa/dataset.py`
- Test: `tests/routing/text_jepa/test_dataset.py`

Load the existing per-domain `.jsonl` files under `data/final/<domain>/train.jsonl`. Each line is `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`. We use the user content as input text, with its domain label.

- [ ] **Step 1: Write the failing test**

Create `tests/routing/text_jepa/test_dataset.py`:

```python
"""Tests for the 10-domain corpus loader."""
from __future__ import annotations

import json
from pathlib import Path


def test_load_domain_corpus_reads_jsonl(tmp_path: Path) -> None:
    from src.routing.text_jepa.dataset import load_domain_corpus

    d = tmp_path / "final"
    (d / "dsp").mkdir(parents=True)
    (d / "electronics").mkdir(parents=True)

    (d / "dsp" / "train.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "What is FFT?"}, {"role": "assistant", "content": "x"}]}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "IIR filter"}, {"role": "assistant", "content": "x"}]}) + "\n",
        encoding="utf-8",
    )
    (d / "electronics" / "train.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "Ohm law"}, {"role": "assistant", "content": "x"}]}) + "\n",
        encoding="utf-8",
    )

    samples = load_domain_corpus(d, domains=["dsp", "electronics"], max_per_domain=10)
    assert len(samples) == 3
    texts = {s.text for s in samples}
    assert "What is FFT?" in texts
    assert "Ohm law" in texts
    labels = {s.domain for s in samples}
    assert labels == {"dsp", "electronics"}


def test_load_domain_corpus_respects_max_per_domain(tmp_path: Path) -> None:
    from src.routing.text_jepa.dataset import load_domain_corpus

    d = tmp_path / "final" / "dsp"
    d.mkdir(parents=True)
    lines = [
        json.dumps({"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a"}]})
        for i in range(20)
    ]
    (d / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    samples = load_domain_corpus(d.parent, domains=["dsp"], max_per_domain=5)
    assert len(samples) == 5


def test_load_domain_corpus_skips_missing_dir(tmp_path: Path) -> None:
    from src.routing.text_jepa.dataset import load_domain_corpus

    (tmp_path / "final").mkdir()
    samples = load_domain_corpus(tmp_path / "final", domains=["missing"], max_per_domain=10)
    assert samples == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.routing.text_jepa.dataset'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/routing/text_jepa/dataset.py`:

```python
"""Domain corpus loader — reads data/final/<domain>/train.jsonl into (text, domain) pairs."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DomainSample:
    text: str
    domain: str


def load_domain_corpus(
    data_dir: Path | str,
    domains: list[str],
    max_per_domain: int,
) -> list[DomainSample]:
    """Load user-content messages from per-domain JSONL files.

    Args:
        data_dir: Root containing `<domain>/train.jsonl` subdirs.
        domains: Domain names to include.
        max_per_domain: Cap on samples per domain (first N lines).

    Returns:
        List of DomainSample, possibly empty if no files found.
    """
    data_dir = Path(data_dir)
    out: list[DomainSample] = []
    for domain in domains:
        path = data_dir / domain / "train.jsonl"
        if not path.exists():
            logger.warning("missing %s", path)
            continue
        with path.open(encoding="utf-8") as f:
            count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = rec.get("messages", [])
                user = next(
                    (m["content"] for m in msgs if m.get("role") == "user"), None
                )
                if not user:
                    continue
                out.append(DomainSample(text=user, domain=domain))
                count += 1
                if count >= max_per_domain:
                    break
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_dataset.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/routing/text_jepa/dataset.py tests/routing/text_jepa/test_dataset.py
git commit -m "feat(text-jepa): domain corpus loader"
```

---

## Task 10: TextJEPAEmbedder — inference wrapper

**Files:**
- Create: `src/routing/text_jepa/embed.py`
- Test: `tests/routing/text_jepa/test_embed.py`

At inference: tokenize text → get per-token MiniLM hidden states → run through trained `StudentEncoder` → mean-pool to a single vector of dim 128. This is what feeds the VQC.

- [ ] **Step 1: Write the failing test**

Create `tests/routing/text_jepa/test_embed.py`:

```python
"""Tests for TextJEPAEmbedder — inference-time embedding helper."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_embed_shape_mean_pooled():
    """Simulated backbone: return a shape matching (seq_len, 384)."""
    from src.routing.text_jepa.embed import TextJEPAEmbedder
    from src.routing.text_jepa.encoder import StudentEncoder

    student = StudentEncoder(input_dim=16, hidden_dim=32, output_dim=8)

    def fake_token_embed(text: str) -> torch.Tensor:
        # deterministic 5-token "encoding"
        return torch.randn(5, 16)

    embedder = TextJEPAEmbedder(
        student=student, token_embed_fn=fake_token_embed, latent_dim=8
    )
    vec = embedder.embed("hello world")
    assert vec.shape == (8,)


def test_embed_returns_numpy():
    import numpy as np

    from src.routing.text_jepa.embed import TextJEPAEmbedder
    from src.routing.text_jepa.encoder import StudentEncoder

    student = StudentEncoder(input_dim=16, hidden_dim=32, output_dim=8)
    embedder = TextJEPAEmbedder(
        student=student,
        token_embed_fn=lambda text: torch.randn(3, 16),
        latent_dim=8,
    )
    vec = embedder.embed("x")
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float64 or vec.dtype == np.float32


def test_embed_eval_mode_no_grad():
    """Calling embed must not accumulate gradients on student."""
    from src.routing.text_jepa.embed import TextJEPAEmbedder
    from src.routing.text_jepa.encoder import StudentEncoder

    student = StudentEncoder(input_dim=16, hidden_dim=32, output_dim=8)
    embedder = TextJEPAEmbedder(
        student=student,
        token_embed_fn=lambda text: torch.randn(3, 16),
        latent_dim=8,
    )
    embedder.embed("x")
    for p in student.parameters():
        assert p.grad is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_embed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.routing.text_jepa.embed'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/routing/text_jepa/embed.py`:

```python
"""Inference-time embedder — text → 128-d vector via trained student."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from src.routing.text_jepa.encoder import StudentEncoder


class TextJEPAEmbedder:
    """Produce a single vector per text by mean-pooling student token embeddings."""

    def __init__(
        self,
        student: StudentEncoder,
        token_embed_fn: Callable[[str], torch.Tensor],
        latent_dim: int = 128,
    ) -> None:
        """
        Args:
            student: trained StudentEncoder.
            token_embed_fn: Callable `str -> (seq_len, input_dim)` tensor.
                E.g. wraps a frozen sentence-transformers backbone.
            latent_dim: Output dim (for shape assertions).
        """
        self.student = student.eval()
        self.token_embed_fn = token_embed_fn
        self.latent_dim = latent_dim

    def embed(self, text: str) -> np.ndarray:
        """Return a single (latent_dim,) numpy vector."""
        with torch.no_grad():
            tokens = self.token_embed_fn(text)  # (S, input_dim)
            if tokens.dim() != 2:
                raise ValueError(f"token_embed_fn must return 2-D tensor, got {tokens.shape}")
            latent = self.student(tokens.unsqueeze(0))  # (1, S, latent_dim)
            pooled = latent.mean(dim=1).squeeze(0)  # (latent_dim,)
        assert pooled.shape == (self.latent_dim,), f"bad shape {pooled.shape}"
        return pooled.cpu().numpy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_embed.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/routing/text_jepa/embed.py tests/routing/text_jepa/test_embed.py
git commit -m "feat(text-jepa): inference embedder"
```

---

## Task 11: Training CLI script

**Files:**
- Create: `configs/text_jepa.yaml`
- Create: `scripts/train_text_jepa.py`
- Test: `tests/scripts/test_train_text_jepa.py`

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_train_text_jepa.py`:

```python
"""Smoke test for the training CLI — runs on synthetic data, 2 epochs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_train_script_runs_on_tiny_corpus(tmp_path: Path) -> None:
    # Create a tiny 2-domain corpus
    data = tmp_path / "data" / "final"
    for dom in ["dsp", "electronics"]:
        (data / dom).mkdir(parents=True)
        lines = [
            json.dumps({"messages": [{"role": "user", "content": f"{dom} question {i}"}, {"role": "assistant", "content": "a"}]})
            for i in range(8)
        ]
        (data / dom / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ckpt = tmp_path / "student.pt"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_text_jepa.py",
            "--data-dir", str(data),
            "--domains", "dsp,electronics",
            "--max-per-domain", "8",
            "--epochs", "2",
            "--batch-size", "2",
            "--output", str(ckpt),
            "--backbone", "random",  # deterministic fake backbone for CI
            "--seq-len", "8",
            "--input-dim", "32",
            "--latent-dim", "8",
            "--hidden-dim", "16",
        ],
        capture_output=True,
        text=True,
        cwd="/Users/electron/Documents/Projets/micro-kiki",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert ckpt.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/scripts/test_train_text_jepa.py -v`
Expected: FAIL — script does not exist yet (`FileNotFoundError` or non-zero exit).

- [ ] **Step 3: Write minimal implementation**

Create `configs/text_jepa.yaml`:

```yaml
# Text-JEPA training config (PoC defaults)
backbone: "models/niche-embeddings"  # sentence-transformers MiniLM 384-d
domains:
  - dsp
  - electronics
  - emc
  - embedded
  - freecad
  - kicad-dsl
  - platformio
  - power
  - spice
  - stm32
max_per_domain: 1000
seq_len: 32
input_dim: 384
hidden_dim: 256
latent_dim: 128
predictor_hidden: 32
batch_size: 16
epochs: 3
lr: 1.0e-3
ema_momentum: 0.99
mask_ratio: 0.4
min_span: 3
max_span: 5
collapse_floor: 0.01
collapse_patience: 2
seed: 42
output: "models/text-jepa/student.pt"
```

Create `scripts/train_text_jepa.py`:

```python
#!/usr/bin/env python3
"""Train the Text-JEPA student + predictor on the 10-domain corpus.

Usage:
    uv run python scripts/train_text_jepa.py --config configs/text_jepa.yaml
    uv run python scripts/train_text_jepa.py --help
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from src.routing.text_jepa.dataset import DomainSample, load_domain_corpus
from src.routing.text_jepa.trainer import TextJEPATrainer

logger = logging.getLogger(__name__)


def _random_backbone(seed: int, seq_len: int, input_dim: int):
    """Fake backbone for CI — returns deterministic random tokens per text."""
    rng = np.random.default_rng(seed)

    def _embed(text: str) -> torch.Tensor:
        local = np.random.default_rng(abs(hash(text)) % (2**32))
        arr = local.standard_normal((seq_len, input_dim)).astype(np.float32)
        return torch.from_numpy(arr)

    return _embed


def _st_backbone(model_dir: Path, seq_len: int):
    """Real sentence-transformers backbone — returns per-token hidden states."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_dir))
    tok = model.tokenizer
    transformer = model[0].auto_model

    def _embed(text: str) -> torch.Tensor:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")
        with torch.no_grad():
            out = transformer(**enc).last_hidden_state  # (1, S, 384)
        return out.squeeze(0).float()

    return _embed


def _make_batches(samples: list[DomainSample], batch_size: int, embed_fn, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [samples[j] for j in idx[i : i + batch_size]]
        tensors = [embed_fn(s.text) for s in chunk]
        yield torch.stack(tensors, dim=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--domains", default=None, help="comma-separated")
    p.add_argument("--max-per-domain", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--backbone", default=None, help="model dir or 'random' for CI")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--input-dim", type=int, default=None)
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    args = p.parse_args()

    cfg: dict = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())

    def pick(name_cli: str, name_cfg: str, default):
        v = getattr(args, name_cli.replace("-", "_"))
        if v is not None:
            return v
        return cfg.get(name_cfg, default)

    data_dir = Path(pick("data-dir", "data_dir", "data/final"))
    domains = pick("domains", "domains", ["dsp"])
    if isinstance(domains, str):
        domains = [d.strip() for d in domains.split(",") if d.strip()]
    max_per_domain = int(pick("max-per-domain", "max_per_domain", 1000))
    epochs = int(pick("epochs", "epochs", 3))
    batch_size = int(pick("batch-size", "batch_size", 16))
    output = Path(pick("output", "output", "models/text-jepa/student.pt"))
    backbone = pick("backbone", "backbone", "models/niche-embeddings")
    seq_len = int(pick("seq-len", "seq_len", 32))
    input_dim = int(pick("input-dim", "input_dim", 384))
    latent_dim = int(pick("latent-dim", "latent_dim", 128))
    hidden_dim = int(pick("hidden-dim", "hidden_dim", 256))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    samples = load_domain_corpus(data_dir, domains=domains, max_per_domain=max_per_domain)
    if not samples:
        logger.error("no samples loaded — check data-dir and domains")
        return 2
    logger.info("loaded %d samples across %d domains", len(samples), len(domains))

    if backbone == "random":
        embed_fn = _random_backbone(seed=0, seq_len=seq_len, input_dim=input_dim)
    else:
        embed_fn = _st_backbone(Path(backbone), seq_len=seq_len)

    trainer = TextJEPATrainer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        predictor_hidden=max(latent_dim // 4, 8),
        lr=float(cfg.get("lr", 1e-3)),
        ema_momentum=float(cfg.get("ema_momentum", 0.99)),
        mask_ratio=float(cfg.get("mask_ratio", 0.4)),
        min_span=int(cfg.get("min_span", 3)),
        max_span=int(cfg.get("max_span", 5)),
        collapse_floor=float(cfg.get("collapse_floor", 0.01)),
        collapse_patience=int(cfg.get("collapse_patience", 2)),
        seed=int(cfg.get("seed", 42)),
    )

    for epoch in range(epochs):
        losses: list[float] = []
        for batch in _make_batches(samples, batch_size, embed_fn, seed=epoch):
            loss = trainer.step(batch)
            losses.append(float(loss.item()))
            if trainer.collapsed:
                logger.error("COLLAPSE detected at epoch %d — aborting", epoch)
                return 3
        logger.info("epoch %d/%d avg_loss=%.4f", epoch + 1, epochs, float(np.mean(losses)))

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student_state_dict": trainer.student.state_dict(),
            "predictor_state_dict": trainer.predictor.state_dict(),
            "config": {
                "input_dim": input_dim,
                "latent_dim": latent_dim,
                "hidden_dim": hidden_dim,
                "seq_len": seq_len,
                "backbone": str(backbone),
            },
        },
        output,
    )
    logger.info("saved student checkpoint to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/scripts/test_train_text_jepa.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add configs/text_jepa.yaml scripts/train_text_jepa.py tests/scripts/test_train_text_jepa.py
git commit -m "feat(text-jepa): training CLI + config"
```

---

## Task 12: HybridPipeline integration — `use_text_jepa` flag

**Files:**
- Modify: `src/routing/hybrid_pipeline.py`
- Test: `tests/routing/text_jepa/test_hybrid_pipeline_integration.py`

Add a config flag and a getter for an embedder that returns either the existing MiniLM vector (baseline) or a Text-JEPA vector. The router then calls `.route(embedding)` unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/routing/text_jepa/test_hybrid_pipeline_integration.py`:

```python
"""Integration: HybridPipelineConfig carries use_text_jepa; build_embedder routes correctly."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_config_accepts_use_text_jepa():
    from src.routing.hybrid_pipeline import HybridPipelineConfig

    cfg = HybridPipelineConfig(use_text_jepa=True, text_jepa_checkpoint="dummy.pt")
    assert cfg.use_text_jepa is True
    assert cfg.text_jepa_checkpoint == "dummy.pt"


def test_config_defaults_use_text_jepa_false():
    from src.routing.hybrid_pipeline import HybridPipelineConfig

    cfg = HybridPipelineConfig()
    assert cfg.use_text_jepa is False


def test_build_text_jepa_embedder_roundtrip(tmp_path):
    """Train a tiny student, save it, load it via build_text_jepa_embedder, embed."""
    from src.routing.hybrid_pipeline import build_text_jepa_embedder
    from src.routing.text_jepa.encoder import StudentEncoder

    student = StudentEncoder(input_dim=32, hidden_dim=16, output_dim=8)
    ckpt_path = tmp_path / "student.pt"
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "predictor_state_dict": {},
            "config": {
                "input_dim": 32,
                "latent_dim": 8,
                "hidden_dim": 16,
                "seq_len": 4,
                "backbone": "random",
            },
        },
        ckpt_path,
    )

    embedder = build_text_jepa_embedder(ckpt_path)
    vec = embedder.embed("hello")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (8,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_hybrid_pipeline_integration.py -v`
Expected: FAIL — `HybridPipelineConfig` has no `use_text_jepa`, `build_text_jepa_embedder` does not exist.

- [ ] **Step 3: Write minimal implementation**

Edit `src/routing/hybrid_pipeline.py` — add the fields to `HybridPipelineConfig` (edit the dataclass block starting at line 41):

Replace the entire `HybridPipelineConfig` dataclass body with:

```python
@dataclass(frozen=True)
class HybridPipelineConfig:
    """Frozen configuration for the triple-hybrid routing pipeline.

    Args:
        use_quantum: Enable VQC router. Falls back to classical if False or
            PennyLane is unavailable.
        use_memory: Enable Aeon pre/post-inference memory.
        use_negotiator: Enable CAMP negotiator for multi-candidate arbitration.
            Disabled by default (adds latency).
        quantum_confidence_threshold: If quantum confidence is below this value
            the classical ModelRouter is also run as a backup and its result
            takes precedence.
        use_text_jepa: If True, VQC input embedding comes from the Text-JEPA
            student. Default False (baseline MiniLM direct embedding).
        text_jepa_checkpoint: Path to the student checkpoint saved by
            scripts/train_text_jepa.py. Required when use_text_jepa=True.
    """

    use_quantum: bool = True
    use_memory: bool = True
    use_negotiator: bool = False
    quantum_confidence_threshold: float = 0.7
    use_text_jepa: bool = False
    text_jepa_checkpoint: str | None = None
```

Then append at the end of `src/routing/hybrid_pipeline.py` (keep existing content intact):

```python


def build_text_jepa_embedder(checkpoint_path: str | Path):
    """Load a Text-JEPA student checkpoint and return a TextJEPAEmbedder.

    Args:
        checkpoint_path: Path produced by scripts/train_text_jepa.py.

    Returns:
        TextJEPAEmbedder ready for `.embed(text)`.
    """
    import numpy as np
    import torch

    from src.routing.text_jepa.embed import TextJEPAEmbedder
    from src.routing.text_jepa.encoder import StudentEncoder

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    student = StudentEncoder(
        input_dim=int(cfg["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        output_dim=int(cfg["latent_dim"]),
    )
    student.load_state_dict(ckpt["student_state_dict"])
    student.eval()

    seq_len = int(cfg["seq_len"])
    input_dim = int(cfg["input_dim"])
    backbone = cfg.get("backbone", "random")

    if backbone == "random":
        def _embed(text: str) -> torch.Tensor:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            return torch.from_numpy(rng.standard_normal((seq_len, input_dim)).astype(np.float32))
        token_fn = _embed
    else:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(str(backbone))
        tok = st.tokenizer
        transformer = st[0].auto_model

        def _embed_st(text: str) -> torch.Tensor:
            enc = tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
                padding="max_length",
            )
            with torch.no_grad():
                out = transformer(**enc).last_hidden_state
            return out.squeeze(0).float()
        token_fn = _embed_st

    return TextJEPAEmbedder(
        student=student,
        token_embed_fn=token_fn,
        latent_dim=int(cfg["latent_dim"]),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/routing/text_jepa/test_hybrid_pipeline_integration.py tests/routing/test_hybrid_pipeline.py -v`
Expected: PASS for all new tests; existing hybrid_pipeline tests unchanged (still pass).

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/routing/hybrid_pipeline.py tests/routing/text_jepa/test_hybrid_pipeline_integration.py
git commit -m "feat(routing): use_text_jepa flag + embedder factory"
```

---

## Task 13: Benchmark CLI — baseline vs Text-JEPA

**Files:**
- Create: `scripts/eval_text_jepa_vqc.py`
- Test: `tests/scripts/test_eval_text_jepa_vqc.py`

Script: for each condition (baseline MiniLM vs Text-JEPA student), (a) build embeddings for N samples/domain, (b) train the VQC on 80% train split, (c) measure 35-class accuracy on 20% test split, (d) write JSON with both conditions + latent dim + param count.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_eval_text_jepa_vqc.py`:

```python
"""Smoke test the eval script on a tiny synthetic corpus + untrained student."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pennylane = pytest.importorskip("pennylane")
torch = pytest.importorskip("torch")


def test_eval_script_produces_json(tmp_path: Path) -> None:
    from src.routing.text_jepa.encoder import StudentEncoder

    # Tiny corpus (use only 2 domains — faster and matches the eval script's n_classes)
    data = tmp_path / "data" / "final"
    for dom in ["dsp", "electronics"]:
        (data / dom).mkdir(parents=True)
        lines = []
        for i in range(20):
            text = f"{dom} unique question number {i}"
            lines.append(json.dumps({"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": "a"}]}))
        (data / dom / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Fake trained student checkpoint
    student = StudentEncoder(input_dim=16, hidden_dim=16, output_dim=8)
    ckpt = tmp_path / "student.pt"
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "predictor_state_dict": {},
            "config": {
                "input_dim": 16,
                "latent_dim": 8,
                "hidden_dim": 16,
                "seq_len": 4,
                "backbone": "random",
            },
        },
        ckpt,
    )

    out_json = tmp_path / "results.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_text_jepa_vqc.py",
            "--data-dir", str(data),
            "--domains", "dsp,electronics",
            "--max-per-domain", "20",
            "--epochs", "2",
            "--checkpoint", str(ckpt),
            "--output", str(out_json),
            "--backbone", "random",
            "--seq-len", "4",
            "--input-dim", "16",
        ],
        capture_output=True,
        text=True,
        cwd="/Users/electron/Documents/Projets/micro-kiki",
    )
    assert result.returncode == 0, f"stderr={result.stderr}\nstdout={result.stdout}"
    data_json = json.loads(out_json.read_text())
    assert "baseline" in data_json
    assert "text_jepa" in data_json
    assert "accuracy" in data_json["baseline"]
    assert "accuracy" in data_json["text_jepa"]
    assert 0.0 <= data_json["baseline"]["accuracy"] <= 1.0
    assert 0.0 <= data_json["text_jepa"]["accuracy"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/scripts/test_eval_text_jepa_vqc.py -v`
Expected: FAIL — script does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/eval_text_jepa_vqc.py`:

```python
#!/usr/bin/env python3
"""Benchmark VQC router: baseline MiniLM vs Text-JEPA student embeddings.

Writes a JSON blob with accuracy + param count for each condition.

Usage:
    uv run python scripts/eval_text_jepa_vqc.py \
        --data-dir data/final \
        --domains dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32 \
        --checkpoint models/text-jepa/student.pt \
        --output results/text-jepa-vqc.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig
from src.routing.text_jepa.dataset import load_domain_corpus
from src.routing.text_jepa.embed import TextJEPAEmbedder
from src.routing.text_jepa.encoder import StudentEncoder

logger = logging.getLogger(__name__)


def _make_token_fn(backbone: str, seq_len: int, input_dim: int):
    if backbone == "random":
        def _f(text: str) -> torch.Tensor:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            return torch.from_numpy(rng.standard_normal((seq_len, input_dim)).astype(np.float32))
        return _f

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(backbone))
    tok = st.tokenizer
    transformer = st[0].auto_model

    def _embed(text: str) -> torch.Tensor:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")
        with torch.no_grad():
            out = transformer(**enc).last_hidden_state
        return out.squeeze(0).float()
    return _embed


def _baseline_embed(token_fn, text: str) -> np.ndarray:
    """Mean-pool raw MiniLM tokens (no JEPA student)."""
    with torch.no_grad():
        toks = token_fn(text)  # (S, input_dim)
        pooled = toks.mean(dim=0)
    return pooled.cpu().numpy()


def _evaluate_vqc(
    embs: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    epochs: int,
    seed: int,
) -> dict:
    """Split 80/20, train a VQC on train, measure accuracy on test."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(embs))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]

    cfg = QuantumRouterConfig(n_qubits=6, n_layers=6, n_classes=n_classes)
    vqc = QuantumRouter(cfg)
    vqc.train(embs[tr], labels[tr].astype(int), epochs=epochs)

    # Predict on test
    correct = 0
    for e, y in zip(embs[te], labels[te]):
        qubits = vqc.circuit(vqc.weights, e)
        logits = qubits @ vqc.linear_w + vqc.linear_b
        pred = int(np.argmax(logits))
        if pred == int(y):
            correct += 1
    acc = correct / max(len(te), 1)
    n_params = vqc.weights.size + vqc.linear_w.size + vqc.linear_b.size
    return {
        "accuracy": acc,
        "n_test": int(len(te)),
        "n_params": int(n_params),
        "latent_dim": int(embs.shape[1]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--domains", required=True, help="comma-separated")
    p.add_argument("--max-per-domain", type=int, default=200)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backbone", default="models/niche-embeddings")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--input-dim", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    samples = load_domain_corpus(Path(args.data_dir), domains=domains, max_per_domain=args.max_per_domain)
    if not samples:
        logger.error("no samples loaded")
        return 2
    logger.info("loaded %d samples", len(samples))

    dom_to_idx = {d: i for i, d in enumerate(domains)}
    labels = np.array([dom_to_idx[s.domain] for s in samples], dtype=np.int64)

    token_fn = _make_token_fn(args.backbone, args.seq_len, args.input_dim)

    # Baseline: raw MiniLM mean-pool
    logger.info("computing baseline embeddings …")
    baseline_embs = np.stack([_baseline_embed(token_fn, s.text) for s in samples], axis=0)

    # Text-JEPA: load student and embed
    logger.info("loading Text-JEPA student from %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    student = StudentEncoder(
        input_dim=int(cfg["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        output_dim=int(cfg["latent_dim"]),
    )
    student.load_state_dict(ckpt["student_state_dict"])
    embedder = TextJEPAEmbedder(student=student, token_embed_fn=token_fn, latent_dim=int(cfg["latent_dim"]))
    logger.info("computing Text-JEPA embeddings …")
    jepa_embs = np.stack([embedder.embed(s.text) for s in samples], axis=0)

    n_classes = len(domains)

    logger.info("training+evaluating baseline VQC …")
    baseline_result = _evaluate_vqc(baseline_embs, labels, n_classes, epochs=args.epochs, seed=args.seed)
    logger.info("training+evaluating Text-JEPA VQC …")
    jepa_result = _evaluate_vqc(jepa_embs, labels, n_classes, epochs=args.epochs, seed=args.seed)

    out = {
        "baseline": baseline_result,
        "text_jepa": jepa_result,
        "domains": domains,
        "n_samples": int(len(samples)),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("baseline acc=%.3f  text_jepa acc=%.3f", baseline_result["accuracy"], jepa_result["accuracy"])
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/scripts/test_eval_text_jepa_vqc.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add scripts/eval_text_jepa_vqc.py tests/scripts/test_eval_text_jepa_vqc.py
git commit -m "feat(text-jepa): baseline vs JEPA VQC eval script"
```

---

## Task 14: Run the real PoC experiment — 10 domains × 1000 samples

**Files:**
- Create: `results/text-jepa-vqc.md`
- Create: `results/text-jepa-vqc.json` (generated)
- Create: `models/text-jepa/student.pt` (generated)

This task executes the pipeline on real data. No unit test — this is an experiment run. Document results inline.

- [ ] **Step 1: Train the student**

Run:
```bash
cd /Users/electron/Documents/Projets/micro-kiki
uv run python scripts/train_text_jepa.py --config configs/text_jepa.yaml
```

Expected stdout (last line): `saved student checkpoint to models/text-jepa/student.pt`.
Expected behavior: avg_loss decreases monotonically across 3 epochs; no `COLLAPSE detected` line.

If `COLLAPSE detected` appears → STOP, hit kill criterion. Document in `results/text-jepa-vqc.md` and end PoC.

- [ ] **Step 2: Run the evaluation**

Run:
```bash
cd /Users/electron/Documents/Projets/micro-kiki
uv run python scripts/eval_text_jepa_vqc.py \
  --data-dir data/final \
  --domains dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32 \
  --max-per-domain 1000 \
  --epochs 10 \
  --checkpoint models/text-jepa/student.pt \
  --output results/text-jepa-vqc.json \
  --backbone models/niche-embeddings \
  --seq-len 32 \
  --input-dim 384
```

Expected stdout (last info line): `baseline acc=X.XXX  text_jepa acc=Y.YYY`.
Expected file written: `results/text-jepa-vqc.json` with both conditions populated.

- [ ] **Step 3: Record results**

Create `results/text-jepa-vqc.md` with the template below, filling in the real numbers from `results/text-jepa-vqc.json`:

```markdown
# Text-JEPA mini VQC router — PoC results

**Date:** 2026-04-17 (replace with actual run date)
**Corpus:** 10 niche domains × 1000 samples from `data/final/`
**Backbone:** `models/niche-embeddings` (frozen MiniLM-L6-v2, 384-d)
**Student:** MLP 384 → 256 → 128
**Teacher:** EMA momentum 0.99
**Predictor:** MLP 128 → 32 → 128
**Loss:** L1 on span-masked positions (ratio 0.4, span 3-5 tokens)
**VQC:** 6 qubits, 6 StronglyEntanglingLayers, parameter-shift gradient, 10 training epochs.

## Results

| Condition   | Embedding dim | VQC test accuracy | VQC params |
|-------------|---------------|-------------------|------------|
| Baseline    | 384           | <FILL FROM JSON>  | <FILL>     |
| Text-JEPA   | 128           | <FILL FROM JSON>  | <FILL>     |

## Decision

- [ ] Success: Text-JEPA >= baseline → proceed to ablation (Task 15).
- [ ] Partial success: Text-JEPA ≥ baseline − 1 pt at 3× smaller latent → compute win.
- [ ] Kill: Text-JEPA < baseline − 5 pt OR student collapse → abort hypothesis.

## Notes

(Fill in observations about training loss curve, any collapse warnings, latency.)
```

- [ ] **Step 4: Sanity-check results JSON**

Run:
```bash
cat /Users/electron/Documents/Projets/micro-kiki/results/text-jepa-vqc.json
```

Expected output: a JSON object with `baseline.accuracy`, `text_jepa.accuracy`, `baseline.latent_dim == 384`, `text_jepa.latent_dim == 128`, `n_samples >= 1000`, `domains` list of length 10.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add results/text-jepa-vqc.md results/text-jepa-vqc.json
git commit -m "docs(text-jepa): PoC baseline-vs-JEPA VQC results"
```

Note: do NOT commit `models/text-jepa/student.pt` — large binary. It is produced on demand.
Add `models/text-jepa/` to `.gitignore` if not already covered.

---

## Task 15: Ablation — latent-dim sweep (only if Task 14 did not hit the kill criterion)

**Files:**
- Create: `scripts/ablate_text_jepa_dim.py`
- Test: `tests/scripts/test_ablate_text_jepa_dim.py`
- Create: `results/text-jepa-ablation.json`

Test whether JEPA preserves accuracy at smaller latent dim. Train 3 students with `latent_dim ∈ {64, 128, 256}` on the same corpus, evaluate each against the VQC.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_ablate_text_jepa_dim.py`:

```python
"""Smoke test the ablation script on a tiny corpus."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pennylane")
pytest.importorskip("torch")


def test_ablation_runs(tmp_path: Path) -> None:
    data = tmp_path / "data" / "final"
    for dom in ["dsp", "electronics"]:
        (data / dom).mkdir(parents=True)
        lines = [
            json.dumps({"messages": [{"role": "user", "content": f"{dom} Q {i}"}, {"role": "assistant", "content": "a"}]})
            for i in range(16)
        ]
        (data / dom / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_json = tmp_path / "ablation.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ablate_text_jepa_dim.py",
            "--data-dir", str(data),
            "--domains", "dsp,electronics",
            "--max-per-domain", "16",
            "--latent-dims", "4,8",
            "--train-epochs", "1",
            "--vqc-epochs", "1",
            "--output", str(out_json),
            "--backbone", "random",
            "--seq-len", "4",
            "--input-dim", "16",
            "--hidden-dim", "16",
        ],
        capture_output=True,
        text=True,
        cwd="/Users/electron/Documents/Projets/micro-kiki",
    )
    assert result.returncode == 0, f"stderr={result.stderr}\nstdout={result.stdout}"
    data_json = json.loads(out_json.read_text())
    assert "runs" in data_json
    assert len(data_json["runs"]) == 2
    for entry in data_json["runs"]:
        assert "latent_dim" in entry
        assert "accuracy" in entry
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/scripts/test_ablate_text_jepa_dim.py -v`
Expected: FAIL — script does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/ablate_text_jepa_dim.py`:

```python
#!/usr/bin/env python3
"""Latent-dim ablation: train Text-JEPA students at multiple latent dims and compare VQC accuracy.

Usage:
    uv run python scripts/ablate_text_jepa_dim.py \
        --data-dir data/final \
        --domains dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32 \
        --latent-dims 64,128,256 \
        --output results/text-jepa-ablation.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig
from src.routing.text_jepa.dataset import load_domain_corpus
from src.routing.text_jepa.embed import TextJEPAEmbedder
from src.routing.text_jepa.trainer import TextJEPATrainer

logger = logging.getLogger(__name__)


def _make_token_fn(backbone: str, seq_len: int, input_dim: int):
    if backbone == "random":
        def _f(text: str) -> torch.Tensor:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            return torch.from_numpy(rng.standard_normal((seq_len, input_dim)).astype(np.float32))
        return _f

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(backbone))
    tok = st.tokenizer
    transformer = st[0].auto_model

    def _embed(text: str) -> torch.Tensor:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")
        with torch.no_grad():
            out = transformer(**enc).last_hidden_state
        return out.squeeze(0).float()
    return _embed


def _train_student(
    samples,
    token_fn,
    input_dim: int,
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> TextJEPATrainer:
    trainer = TextJEPATrainer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        predictor_hidden=max(latent_dim // 4, 8),
        lr=1e-3,
        ema_momentum=0.99,
        mask_ratio=0.4,
        min_span=3,
        max_span=5,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    for epoch in range(epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            chunk = [samples[j] for j in idx[i : i + batch_size]]
            batch = torch.stack([token_fn(s.text) for s in chunk], dim=0)
            trainer.step(batch)
            if trainer.collapsed:
                logger.warning("collapse at latent_dim=%d epoch=%d", latent_dim, epoch)
                return trainer
    return trainer


def _eval_vqc(embs: np.ndarray, labels: np.ndarray, n_classes: int, epochs: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(embs))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]
    cfg = QuantumRouterConfig(n_qubits=6, n_layers=6, n_classes=n_classes)
    vqc = QuantumRouter(cfg)
    vqc.train(embs[tr], labels[tr].astype(int), epochs=epochs)
    correct = 0
    for e, y in zip(embs[te], labels[te]):
        qubits = vqc.circuit(vqc.weights, e)
        logits = qubits @ vqc.linear_w + vqc.linear_b
        if int(np.argmax(logits)) == int(y):
            correct += 1
    return {"accuracy": correct / max(len(te), 1), "n_test": int(len(te))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--domains", required=True)
    p.add_argument("--max-per-domain", type=int, default=500)
    p.add_argument("--latent-dims", required=True, help="comma-separated ints")
    p.add_argument("--train-epochs", type=int, default=3)
    p.add_argument("--vqc-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", required=True)
    p.add_argument("--backbone", default="models/niche-embeddings")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--input-dim", type=int, default=384)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    latent_dims = [int(x) for x in args.latent_dims.split(",")]

    samples = load_domain_corpus(Path(args.data_dir), domains=domains, max_per_domain=args.max_per_domain)
    if not samples:
        logger.error("no samples loaded")
        return 2

    dom_to_idx = {d: i for i, d in enumerate(domains)}
    labels = np.array([dom_to_idx[s.domain] for s in samples], dtype=np.int64)
    token_fn = _make_token_fn(args.backbone, args.seq_len, args.input_dim)
    n_classes = len(domains)

    runs: list[dict] = []
    for ld in latent_dims:
        logger.info("--- latent_dim=%d ---", ld)
        trainer = _train_student(
            samples=samples,
            token_fn=token_fn,
            input_dim=args.input_dim,
            latent_dim=ld,
            hidden_dim=args.hidden_dim,
            epochs=args.train_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        embedder = TextJEPAEmbedder(
            student=trainer.student, token_embed_fn=token_fn, latent_dim=ld
        )
        embs = np.stack([embedder.embed(s.text) for s in samples], axis=0)
        result = _eval_vqc(embs, labels, n_classes, epochs=args.vqc_epochs, seed=args.seed)
        result["latent_dim"] = ld
        result["collapsed"] = bool(trainer.collapsed)
        runs.append(result)
        logger.info("latent_dim=%d accuracy=%.3f collapsed=%s", ld, result["accuracy"], result["collapsed"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"runs": runs, "domains": domains, "n_samples": int(len(samples))}, indent=2)
    )
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/scripts/test_ablate_text_jepa_dim.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the real ablation, then commit**

Run:
```bash
cd /Users/electron/Documents/Projets/micro-kiki
uv run python scripts/ablate_text_jepa_dim.py \
  --data-dir data/final \
  --domains dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32 \
  --max-per-domain 500 \
  --latent-dims 64,128,256 \
  --train-epochs 3 \
  --vqc-epochs 10 \
  --output results/text-jepa-ablation.json
```

Expected: JSON with 3 runs, each with `latent_dim`, `accuracy`, `collapsed` keys.

Append the ablation numbers into `results/text-jepa-vqc.md` under a new `## Ablation — latent dim` section with a 3-row table.

Commit:

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add scripts/ablate_text_jepa_dim.py tests/scripts/test_ablate_text_jepa_dim.py results/text-jepa-ablation.json results/text-jepa-vqc.md
git commit -m "feat(text-jepa): latent-dim ablation"
```

---

## Task 16: Final suite sanity — everything still passes

**Files:** none created/modified — verification only.

- [ ] **Step 1: Run the full Text-JEPA test suite**

Run:
```bash
cd /Users/electron/Documents/Projets/micro-kiki
uv run python -m pytest tests/routing/text_jepa/ tests/scripts/test_train_text_jepa.py tests/scripts/test_eval_text_jepa_vqc.py tests/scripts/test_ablate_text_jepa_dim.py tests/routing/test_hybrid_pipeline.py tests/routing/test_quantum_router.py -v
```

Expected: all green. If anything fails, fix the regression before closing the PoC.

- [ ] **Step 2: Final verdict — edit `results/text-jepa-vqc.md`**

Check the three boxes (Success / Partial / Kill) based on the numbers in `results/text-jepa-vqc.json` and `results/text-jepa-ablation.json`. Fill the "Notes" section with observations.

- [ ] **Step 3: Commit the verdict**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add results/text-jepa-vqc.md
git commit -m "docs(text-jepa): PoC verdict + notes"
```

---

## Self-review (performed)

- **Spec coverage:** span-mask (Task 2), student encoder (3), EMA teacher (4), predictor (5), L1 masked loss (6), collapse guard (7), training loop (8), corpus loader (9), inference embedder (10), training CLI (11), router integration (12), benchmark CLI (13), real experiment run (14), latent-dim ablation (15), final regression check (16). All in-scope items from the brief are covered.
- **Placeholder scan:** no "TBD", "similar to Task N", "add error handling" — every code block is complete and self-contained.
- **Type consistency:** `StudentEncoder`, `TeacherEncoder`, `Predictor`, `TextJEPAEmbedder`, `TextJEPATrainer`, `DomainSample`, `HybridPipelineConfig`, `QuantumRouter`, `QuantumRouterConfig`, `span_mask`, `masked_l1_loss`, `embedding_std`, `CollapseMonitor`, `load_domain_corpus`, `build_text_jepa_embedder` — all names appear identically across defining tasks and consumer tasks.
- **Kill criterion** (> 5pt drop OR std collapse) and **success criterion** (≥ baseline OR equal at smaller dim) are explicit upfront and re-checked in Task 14 and Task 16.
- **Out-of-scope** parking lot stated at the top.
- **Span masking, EMA teacher, L1 loss, collapse detection** each have their own task (2, 4, 6, 7) + are composed in the trainer (8).
- **Commit style** matches repo convention: `feat(scope): imperative`, subjects ≤ 50 chars, no Co-Authored-By trailer (per `CLAUDE.md`).
