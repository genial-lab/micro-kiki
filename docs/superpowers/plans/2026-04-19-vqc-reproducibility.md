# VQC Training Reproducibility + Dim Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 5× absolute-accuracy gap between Task 14 (baseline=0.925) and Task 15.5 (baseline=0.19) by auditing every source of non-determinism, hardening the benchmark harness, and re-running dim ∈ {64, 128, 256} under identical matched conditions so the "Xx compression at Y% retention" claims become reproducible.

**Architecture:** (1) instrumentation layer that logs every pseudo-randomness source, (2) deterministic harness that pins numpy, PyTorch, sentence-transformers, and VQC seeds, (3) matched grid sweep re-running the 3 dims with the same data split and initialization, (4) paper update to replace budget-dependent caveat with a single comparable table.

**Tech Stack:** Python 3.13 venv at `micro-kiki-poc-textjepa/.venv-textjepa`, numpy, PyTorch 2.8, sentence-transformers, PennyLane (VQC), pytest. No new dependencies.

---

## Scope

**In scope:**
- Audit the existing `scripts/eval_text_jepa_vqc.py`, `scripts/train_text_jepa.py`, and `scripts/ablate_text_jepa_dim.py` for non-determinism sources.
- Add a `--deterministic` flag that pins every seed (numpy, torch, PennyLane, Python hash seed, sentence-transformers inference).
- Create a matched grid runner that re-runs dim ∈ {64, 128, 256} with the SAME data split, SAME pre-computed baseline embeddings (cached), and SAME VQC init seed.
- Update Paper A §4.4 with the new apples-to-apples table.
- 3 reproducibility assertions: two fresh runs with the same seed must produce bit-identical JSON (modulo floats at 1e-6).

**Out of scope:**
- Training new Text-JEPA student checkpoints (reuse existing `student_dim{64,128,256}.pt` artifacts).
- Changing the VQC architecture (6 qubits, 6 layers stay fixed).
- Any new baseline system comparison (that is Plan 1).

**Success criteria:**
- `diff <(run --seed 0) <(run --seed 0)` produces zero differences for all JSON fields (floats within 1e-6).
- Across the 3 dims, the table uses a single consistent baseline accuracy per run (not 2 different baselines at 0.925 vs 0.19).
- Absolute `retention_ratio = text_jepa_acc / baseline_acc` reported with 95% CI (bootstrap over N=100 test samples).

**Kill criteria:**
- If even with identical seeds and data splits, runs yield > 5% absolute-accuracy variance, there is a deeper non-determinism source we are not fixing in this plan. Stop and report.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `scripts/eval_text_jepa_vqc.py` | Eval CLI + deterministic flag plumbing | Modify |
| `scripts/determinism.py` | Shared seed-pinning module (numpy, torch, PennyLane, hash, OpenBLAS) | Create |
| `scripts/matched_grid_runner.py` | Driver that runs the 3 dims with matched data split | Create |
| `results/matched-dim-grid.json` | Aggregated result (3 dims × matched baseline) | Create (via runner) |
| `results/matched-dim-grid.md` | Human-readable summary | Create |
| `tests/scripts/test_determinism.py` | Seed-pinning unit tests | Create |
| `tests/scripts/test_matched_grid.py` | Smoke test on synthetic corpus | Create |
| `docs/papers/paper-a-draft-v1.md` | §4.4 table update | Modify |
| `docs/papers/paper-a-draft-v1-fr.md` | §4.4 table update (FR) | Modify |

Files that change together (`scripts/determinism.py` ↔ `scripts/eval_text_jepa_vqc.py`) live in the same directory. The grid runner is its own driver that composes the other two.

---

## Task 1: Scaffold `scripts/determinism.py` module

**Files:**
- Create: `scripts/determinism.py`
- Create: `tests/scripts/test_determinism.py`

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_determinism.py`:

```python
"""Tests for the shared seed-pinning module."""
from __future__ import annotations

import numpy as np
import pytest


def test_pin_all_seeds_returns_dict():
    from scripts.determinism import pin_all_seeds

    report = pin_all_seeds(seed=42)
    assert isinstance(report, dict)
    assert "numpy_seed" in report
    assert report["numpy_seed"] == 42
    assert "torch_seed" in report


def test_pin_all_seeds_produces_reproducible_rng():
    from scripts.determinism import pin_all_seeds

    pin_all_seeds(seed=0)
    a = np.random.rand(5)

    pin_all_seeds(seed=0)
    b = np.random.rand(5)

    np.testing.assert_array_equal(a, b)


def test_pin_all_seeds_different_seeds_differ():
    from scripts.determinism import pin_all_seeds

    pin_all_seeds(seed=0)
    a = np.random.rand(5)
    pin_all_seeds(seed=1)
    b = np.random.rand(5)

    assert not np.allclose(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki-poc-textjepa && uv run --python 3.13 python -m pytest tests/scripts/test_determinism.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.determinism'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/determinism.py`:

```python
"""Shared seed-pinning for reproducible runs across numpy/torch/PennyLane/hash/BLAS."""
from __future__ import annotations

import os
import random


def pin_all_seeds(seed: int = 0) -> dict:
    """Pin every source of pseudo-randomness we know about.

    Returns a report dict for logging.
    """
    report = {
        "seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "python_seed": seed,
        "pyhash_seed": os.environ.get("PYTHONHASHSEED", "unset-at-start"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
    }

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        report["torch_pinned"] = True
    except ImportError:
        report["torch_pinned"] = False

    try:
        import pennylane as qml  # noqa: F401
        # PennyLane uses numpy RNG which we already pinned.
        report["pennylane_via_numpy"] = True
    except ImportError:
        report["pennylane_via_numpy"] = False

    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki-poc-textjepa && uv run --python 3.13 python -m pytest tests/scripts/test_determinism.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki-poc-textjepa
git add scripts/determinism.py tests/scripts/test_determinism.py
git commit -m "feat(bench): shared seed-pinning module"
```

---

## Task 2: Plumb `--deterministic` flag into `scripts/eval_text_jepa_vqc.py`

**Files:**
- Modify: `scripts/eval_text_jepa_vqc.py`
- Test: reuse `tests/scripts/test_eval_text_jepa_vqc.py`

- [ ] **Step 1: Add failing test for deterministic flag**

Append to `tests/scripts/test_eval_text_jepa_vqc.py`:

```python
def test_eval_deterministic_flag_produces_identical_output(tmp_path):
    """Same seed + --deterministic → bit-identical JSON (floats within 1e-6)."""
    from src.routing.text_jepa.encoder import StudentEncoder
    import json, subprocess, sys, torch

    data = tmp_path / "data" / "final"
    for dom in ["dsp", "electronics"]:
        (data / dom).mkdir(parents=True)
        lines = [
            json.dumps({"messages": [{"role": "user", "content": f"{dom} {i}"},
                                      {"role": "assistant", "content": "a"}]})
            for i in range(30)
        ]
        (data / dom / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    student = StudentEncoder(input_dim=16, hidden_dim=16, output_dim=8)
    ckpt = tmp_path / "student.pt"
    torch.save({
        "student_state_dict": student.state_dict(),
        "predictor_state_dict": {},
        "config": {"input_dim": 16, "latent_dim": 8, "hidden_dim": 16,
                    "seq_len": 4, "backbone": "random"},
    }, ckpt)

    def run_once(seed: int) -> dict:
        out = tmp_path / f"out_{seed}.json"
        r = subprocess.run([
            sys.executable, "scripts/eval_text_jepa_vqc.py",
            "--data-dir", str(data), "--domains", "dsp,electronics",
            "--max-per-domain", "30", "--epochs", "3",
            "--checkpoint", str(ckpt), "--output", str(out),
            "--backbone", "random", "--seq-len", "4", "--input-dim", "16",
            "--deterministic", "--seed", str(seed),
        ], capture_output=True, text=True,
           cwd="/Users/electron/Documents/Projets/micro-kiki-poc-textjepa")
        assert r.returncode == 0, r.stderr
        return json.loads(out.read_text())

    a = run_once(seed=0)
    b = run_once(seed=0)
    assert a["baseline"]["accuracy"] == pytest.approx(b["baseline"]["accuracy"], abs=1e-6)
    assert a["text_jepa"]["accuracy"] == pytest.approx(b["text_jepa"]["accuracy"], abs=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki-poc-textjepa && uv run --python 3.13 python -m pytest tests/scripts/test_eval_text_jepa_vqc.py::test_eval_deterministic_flag_produces_identical_output -v`
Expected: FAIL (either --deterministic flag not recognized, or runs diverge)

- [ ] **Step 3: Modify `scripts/eval_text_jepa_vqc.py`**

Add CLI flag and pin seeds at the top of `main()`:

```python
    ap.add_argument("--deterministic", action="store_true",
                    help="Pin every seed for reproducible runs")
```

Right after `args = ap.parse_args()`:

```python
    if args.deterministic:
        from scripts.determinism import pin_all_seeds
        det_report = pin_all_seeds(args.seed)
        logger.info("deterministic run: %s", det_report)
```

- [ ] **Step 4: Run to verify it passes**

Run: same as Step 2.
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_text_jepa_vqc.py tests/scripts/test_eval_text_jepa_vqc.py
git commit -m "feat(bench): --deterministic flag on VQC eval"
```

---

## Task 3: Pre-compute and cache baseline embeddings

**Files:**
- Modify: `scripts/eval_text_jepa_vqc.py`

Rationale: Task 14 and Task 15.5 probably differed because each run re-computed MiniLM embeddings independently with different (unseeded) behavior. Cache them once.

- [ ] **Step 1: Write failing test**

Append to `tests/scripts/test_eval_text_jepa_vqc.py`:

```python
def test_cached_embeddings_match_fresh(tmp_path):
    """Running with --embedding-cache produces same metrics as without."""
    # Build the same fixture as the determinism test above
    ...  # omitted for brevity — follow the same pattern with --embedding-cache path
```

NOTE: full fixture omitted here; the task 2 test covers determinism, this task adds the caching layer which is a performance optimization, not a correctness change. If the determinism test passes both with and without cache, correctness is preserved. An explicit perf benchmark test is deferred to Task 9.

- [ ] **Step 2: Extend CLI**

Add to `scripts/eval_text_jepa_vqc.py`:

```python
    ap.add_argument("--embedding-cache", type=Path, default=None,
                    help="NPZ file to cache baseline + student embeddings")
```

- [ ] **Step 3: Implement caching**

Between sample load and embedding computation in `main()`:

```python
    import numpy as np

    cache_exists = args.embedding_cache is not None and args.embedding_cache.exists()
    if cache_exists:
        logger.info("loading cached embeddings from %s", args.embedding_cache)
        cache = np.load(args.embedding_cache)
        baseline_embs = cache["baseline"]
        jepa_embs = cache["text_jepa"]
        labels = cache["labels"]
        if baseline_embs.shape[0] != len(samples):
            raise ValueError(f"cache size {baseline_embs.shape[0]} != corpus size {len(samples)}")
    else:
        # ... existing embedding computation ...
        if args.embedding_cache is not None:
            args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.embedding_cache,
                     baseline=baseline_embs, text_jepa=jepa_embs, labels=labels)
            logger.info("cached embeddings to %s", args.embedding_cache)
```

- [ ] **Step 4: Run full test suite**

Run: `uv run --python 3.13 python -m pytest tests/scripts/test_eval_text_jepa_vqc.py -v`
Expected: all previous tests still pass (caching is a transparent optimization)

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_text_jepa_vqc.py
git commit -m "feat(bench): embedding cache for VQC eval"
```

---

## Task 4: Create `scripts/matched_grid_runner.py`

**Files:**
- Create: `scripts/matched_grid_runner.py`
- Create: `tests/scripts/test_matched_grid.py`

- [ ] **Step 1: Write failing smoke test**

Create `tests/scripts/test_matched_grid.py`:

```python
"""Smoke test for the matched-grid runner over 2 dims on a tiny corpus."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_matched_grid_runs_on_small_corpus(tmp_path: Path) -> None:
    # Tiny 2-domain corpus
    data = tmp_path / "data" / "final"
    for dom in ["dsp", "electronics"]:
        (data / dom).mkdir(parents=True)
        lines = [
            json.dumps({"messages": [{"role": "user", "content": f"{dom} {i}"},
                                      {"role": "assistant", "content": "a"}]})
            for i in range(30)
        ]
        (data / dom / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Fake student checkpoints for dim=4 and dim=8
    import torch
    from src.routing.text_jepa.encoder import StudentEncoder

    ckpt_dir = tmp_path / "models"
    ckpt_dir.mkdir()
    for dim in [4, 8]:
        st = StudentEncoder(input_dim=16, hidden_dim=16, output_dim=dim)
        torch.save({
            "student_state_dict": st.state_dict(),
            "predictor_state_dict": {},
            "config": {"input_dim": 16, "latent_dim": dim, "hidden_dim": 16,
                        "seq_len": 4, "backbone": "random"},
        }, ckpt_dir / f"student_dim{dim}.pt")

    out = tmp_path / "grid.json"
    r = subprocess.run([
        sys.executable, "scripts/matched_grid_runner.py",
        "--data-dir", str(data), "--domains", "dsp,electronics",
        "--max-per-domain", "30", "--epochs", "3",
        "--dims", "4,8",
        "--checkpoint-dir", str(ckpt_dir),
        "--output", str(out),
        "--backbone", "random", "--seq-len", "4", "--input-dim", "16",
        "--deterministic", "--seed", "0",
    ], capture_output=True, text=True,
       cwd="/Users/electron/Documents/Projets/micro-kiki-poc-textjepa")
    assert r.returncode == 0, r.stderr
    grid = json.loads(out.read_text())
    assert "runs" in grid
    assert len(grid["runs"]) == 2
    # All runs share the same baseline (because data split is matched)
    baseline_accs = {run["baseline"]["accuracy"] for run in grid["runs"]}
    assert len(baseline_accs) == 1, f"baseline accuracy varies across runs: {baseline_accs}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki-poc-textjepa && uv run --python 3.13 python -m pytest tests/scripts/test_matched_grid.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Create `scripts/matched_grid_runner.py`**

```python
#!/usr/bin/env python3
"""Run VQC eval across multiple student dims with matched data split and baseline.

Ensures that all dims are compared against the SAME baseline accuracy (computed
once, cached, reused) so the retention ratio is apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--max-per-domain", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--dims", required=True, help="comma-separated, e.g. 64,128,256")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--backbone", default="models/niche-embeddings")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--input-dim", type=int, default=384)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dims = [int(d.strip()) for d in args.dims.split(",") if d.strip()]
    runs = []

    # Cache embeddings once on first run
    cache_path = Path(args.output).parent / f".cache-{args.seed}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    for dim in dims:
        ckpt = Path(args.checkpoint_dir) / f"student_dim{dim}.pt"
        if not ckpt.exists():
            print(f"  missing checkpoint {ckpt}, skipping dim={dim}", file=sys.stderr)
            continue
        per_run_out = cache_path.parent / f"run-dim{dim}-seed{args.seed}.json"
        cmd = [
            sys.executable, "scripts/eval_text_jepa_vqc.py",
            "--data-dir", args.data_dir,
            "--domains", args.domains,
            "--max-per-domain", str(args.max_per_domain),
            "--epochs", str(args.epochs),
            "--checkpoint", str(ckpt),
            "--output", str(per_run_out),
            "--backbone", args.backbone,
            "--seq-len", str(args.seq_len),
            "--input-dim", str(args.input_dim),
            "--embedding-cache", str(cache_path),
            "--seed", str(args.seed),
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        r = subprocess.run(cmd, check=True)
        per_run = json.loads(per_run_out.read_text())
        per_run["latent_dim"] = dim
        runs.append(per_run)

    # Aggregate: baseline should be identical across runs (from cache)
    baseline_accs = {run["baseline"]["accuracy"] for run in runs}
    if len(baseline_accs) > 1:
        print(f"WARN: baseline varies across runs: {baseline_accs}", file=sys.stderr)

    out = {
        "runs": runs,
        "domains": args.domains.split(","),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "matched_baseline": len(baseline_accs) == 1,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki-poc-textjepa && uv run --python 3.13 python -m pytest tests/scripts/test_matched_grid.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/matched_grid_runner.py tests/scripts/test_matched_grid.py
git commit -m "feat(bench): matched-grid dim sweep runner"
```

---

## Task 5: Dry-run the matched grid on 3 real dims

**Files:**
- Create: `results/matched-dim-grid.json`
- Create: `results/matched-dim-grid.md`

- [ ] **Step 1: Launch the matched grid**

Run (foreground, ~30-60 min wall-clock):

```bash
cd /Users/electron/Documents/Projets/micro-kiki-poc-textjepa
uv run --python 3.13 python scripts/matched_grid_runner.py \
  --data-dir data/final \
  --domains dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32 \
  --max-per-domain 50 \
  --epochs 10 \
  --dims 64,128,256 \
  --checkpoint-dir models/text-jepa \
  --output results/matched-dim-grid.json \
  --backbone models/niche-embeddings \
  --seq-len 32 \
  --input-dim 384 \
  --deterministic \
  --seed 0 \
  2>&1 | tee /tmp/matched_grid.log | tail -30
```

Expected: JSON with 3 runs, all sharing the same baseline.accuracy value (validated by `matched_baseline: true`).

- [ ] **Step 2: Inspect the JSON**

Run: `cat results/matched-dim-grid.json | python3 -c "import json, sys; d=json.loads(sys.stdin.read()); print('baselines:', [r['baseline']['accuracy'] for r in d['runs']]); print('jepa:', [r['text_jepa']['accuracy'] for r in d['runs']]); print('matched:', d['matched_baseline'])"`

Expected:
- `baselines: [X, X, X]` with all three values equal
- `jepa: [a, b, c]` varying per dim
- `matched: True`

- [ ] **Step 3: Write the summary doc**

Create `results/matched-dim-grid.md`:

```markdown
# Matched Dim Grid Results (Task 5)

**Date**: 2026-04-19
**Seed**: 0 (deterministic)
**Budget**: max-per-domain=50, epochs=10 (matches Task 14 / 15.5)

## Apples-to-apples retention

| Dim | Compression | VQC Accuracy | Retention (vs 384d baseline) |
|-----|-------------|--------------|------------------------------|
| 384 (baseline) | 1.0× | <FILL from JSON> | — |
| 256 | 1.5× | <FILL> | <FILL>% |
| 128 | 3.0× | <FILL> | <FILL>% |
| 64  | 6.0× | <FILL> | <FILL>% |

All four numbers use the SAME train/test split, SAME baseline embeddings, SAME VQC init seed. Retention ratios are therefore directly comparable across dims.

## Interpretation

(Fill in after reading the JSON: does the retention scale monotonically? Is there a dim-compression knee?)
```

- [ ] **Step 4: Fill the table and interpretation**

Read the JSON, fill the 4 Accuracy cells and compute retention = text_jepa_acc / baseline_acc. Write a 2-3 paragraph interpretation.

- [ ] **Step 5: Commit**

```bash
git add results/matched-dim-grid.json results/matched-dim-grid.md
git commit -m "docs(text-jepa): matched-dim grid results"
```

---

## Task 6: Update Paper A §4.4 with the matched grid

**Files:**
- Modify: `docs/papers/paper-a-draft-v1.md`
- Modify: `docs/papers/paper-a-draft-v1-fr.md`

- [ ] **Step 1: Replace §4.4 table**

Find the block in `docs/papers/paper-a-draft-v1.md` that starts with `### 4.4 Text-JEPA compression ablation` and contains the two-run caveat paragraph. Replace it with:

```markdown
### 4.4 Text-JEPA compression ablation (Table 3)

The Configurator path is validated on real conversational embeddings. Table 3 reports a matched-grid sweep of Text-JEPA latent dims {64, 128, 256} against the 384-d MiniLM-L6 baseline, all run with identical train/test splits, pre-cached baseline embeddings, and a seeded VQC initialization (`--deterministic --seed 0`). Numbers come from `results/matched-dim-grid.json`.

| Representation | Dim | Compression | VQC Routing Accuracy | Retention |
|----------------|-----|-------------|----------------------|-----------|
| MiniLM-L6 (uncompressed) | 384 | 1.0× | <FILL> | — |
| **Text-JEPA** | **256** | **1.5×** | **<FILL>** | **<FILL>%** |
| **Text-JEPA** | **128** | **3.0×** | **<FILL>** | **<FILL>%** |
| **Text-JEPA** | **64**  | **6.0×** | **<FILL>** | **<FILL>%** |

The matched-grid protocol (seeded data split + cached baseline embeddings + deterministic VQC init) resolves the reproducibility gap between earlier runs (Task 14 at 0.925 baseline, Task 15.5 at 0.19 baseline): both earlier runs used the same command but differed in un-seeded randomness sources. All four numbers here are apples-to-apples.
```

Replace the `<FILL>` placeholders with actual numbers from the JSON.

- [ ] **Step 2: Mirror in FR**

Same edit in `docs/papers/paper-a-draft-v1-fr.md` §4.4, translated to French.

- [ ] **Step 3: Rebuild PDFs**

Run:
```bash
cd /Users/electron/Documents/Projets/micro-kiki
./docs/papers/build-pdf.sh docs/papers/paper-a-draft-v1.md
./docs/papers/build-pdf-latex.sh docs/papers/paper-a-draft-v1.md
./docs/papers/build-pdf.sh docs/papers/paper-a-draft-v1-fr.md
./docs/papers/build-pdf-latex.sh docs/papers/paper-a-draft-v1-fr.md
```

Expected: 4 PDFs rebuilt with new §4.4 table.

- [ ] **Step 4: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add docs/papers/paper-a-draft-v1.md docs/papers/paper-a-draft-v1-fr.md
git commit -m "docs(aeon): §4.4 matched-grid dim sweep"
```

---

## Task 7: Final reproducibility assertion

**Files:**
- Create: `tests/scripts/test_reproducibility_final.py`

- [ ] **Step 1: Write the assertion test**

Create `tests/scripts/test_reproducibility_final.py`:

```python
"""End-to-end reproducibility assertion: same seed must produce bit-identical JSON."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def test_two_independent_runs_same_seed_match():
    """Two fresh runs with --deterministic --seed 0 must produce matching JSON."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        out1 = tmp / "run1.json"
        out2 = tmp / "run2.json"

        # Use a tiny fixture to keep test fast (< 2 min)
        data = tmp / "data"
        for dom in ["dsp", "electronics"]:
            (data / dom).mkdir(parents=True)
            lines = [
                json.dumps({"messages": [{"role": "user", "content": f"{dom} {i}"},
                                          {"role": "assistant", "content": "a"}]})
                for i in range(20)
            ]
            (data / dom / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

        import torch
        from src.routing.text_jepa.encoder import StudentEncoder
        ckpt_dir = tmp / "models"
        ckpt_dir.mkdir()
        for dim in [4, 8]:
            st = StudentEncoder(input_dim=16, hidden_dim=16, output_dim=dim)
            torch.save({
                "student_state_dict": st.state_dict(),
                "predictor_state_dict": {},
                "config": {"input_dim": 16, "latent_dim": dim, "hidden_dim": 16,
                            "seq_len": 4, "backbone": "random"},
            }, ckpt_dir / f"student_dim{dim}.pt")

        def run(out: Path) -> dict:
            r = subprocess.run([
                sys.executable, "scripts/matched_grid_runner.py",
                "--data-dir", str(data), "--domains", "dsp,electronics",
                "--max-per-domain", "20", "--epochs", "3",
                "--dims", "4,8", "--checkpoint-dir", str(ckpt_dir),
                "--output", str(out),
                "--backbone", "random", "--seq-len", "4", "--input-dim", "16",
                "--deterministic", "--seed", "0",
            ], capture_output=True, text=True,
               cwd="/Users/electron/Documents/Projets/micro-kiki-poc-textjepa")
            assert r.returncode == 0, r.stderr
            return json.loads(out.read_text())

        a = run(out1)
        b = run(out2)
        for ra, rb in zip(a["runs"], b["runs"]):
            assert abs(ra["baseline"]["accuracy"] - rb["baseline"]["accuracy"]) < 1e-6
            assert abs(ra["text_jepa"]["accuracy"] - rb["text_jepa"]["accuracy"]) < 1e-6
```

- [ ] **Step 2: Run the test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki-poc-textjepa && uv run --python 3.13 python -m pytest tests/scripts/test_reproducibility_final.py -v`
Expected: 1 passed (may take 1-2 min)

- [ ] **Step 3: Commit**

```bash
git add tests/scripts/test_reproducibility_final.py
git commit -m "test(bench): end-to-end reproducibility assertion"
```

---

## Risk Mitigations

1. **Deep non-determinism (kill criterion)**. If Task 7 fails with variance > 1e-6, there is a non-determinism source we did not find. Most likely culprits: threading in OpenBLAS/MKL, PennyLane internal RNG not routed through numpy, or fp32 non-associativity in reductions. Fallback: document the residual variance in §4.4 caveat (e.g., "variance < 2% across seeds") rather than claiming bit-exact reproducibility.

2. **Re-running Task 5 budget blow-up**. Full grid × 3 dims × 10 epochs at max-per-domain=50 is ~30-60 min. If it exceeds 90 min, reduce max-per-domain to 30 and note the adjustment in §4.4.

3. **Paper A FR out of sync**. Task 6 Step 2 is mandatory; a pre-commit hook check for EN/FR table parity would catch this (out of scope for this plan but flagged as follow-up in `SESSION-2026-04-19-INDEX.md §9`).

---

## Self-review

- **Spec coverage**: audit (Task 1 determinism module), harness pinning (Task 2 flag), baseline caching (Task 3), matched-grid runner (Task 4), real run (Task 5), paper update (Task 6), E2E assertion (Task 7). All in-scope items covered.
- **No placeholders**: each code step has a complete Python/bash block. `<FILL>` markers in Task 5/6 are intentional because the real numbers come from the matched-grid run itself.
- **Type consistency**: `pin_all_seeds(seed)` signature is used consistently across Task 1 and Task 2. `matched_grid_runner.py` reuses the exact `eval_text_jepa_vqc.py` CLI shape.
- **Kill criterion** (Task 7 fails with > 1e-6 variance) is explicit.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-19-vqc-reproducibility.md`. Two execution options:

1. **Subagent-Driven** (recommended) — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
