# Forgetting gate — operator reference

## Purpose

After each sequential LoRA stack is trained, we must check it has not
catastrophically damaged any previously-trained stack. Per top-level
`CLAUDE.md`:

> Run forgetting check after EACH stack; rollback if `angle < 30°`
> **AND** win-rate drop `> 0.03`.

Phase 1a ships the **angle half** of that gate as a standalone CLI.
The win-rate half (paired evaluation vs. base + prior) is deferred to
phase 1b.

## Usage

```bash
python scripts/measure_forgetting.py \
    --prior-adapter output/stacks/stack-03-cpp/adapter_model.safetensors \
    --new-adapter   output/stacks/stack-04-rust/adapter_model.safetensors \
    --output        results/forgetting-stack04-vs-stack03.json
```

Loguru progress logs go to stderr; the JSON report is written to stdout
and (optionally) to `--output`. Exit code is always `0` — this tool is
informational, not gating.

## Output

```json
{
  "angle_degrees_mean": 72.3,
  "angle_degrees_per_module": {
    "self_attn.q_proj": 71.8,
    "self_attn.k_proj": 74.1,
    "self_attn.v_proj": 70.9,
    "self_attn.o_proj": 72.4,
    "linear_attn.in_proj_qkv": 83.2,
    "mlp.switch_mlp.down_proj": 45.7
  },
  "warning": null,
  "gate_status": "angle_only_partial",
  "note": "Win-rate half not measured; run paired eval for full gate."
}
```

- `angle_degrees_mean` — mean over all module groups present in both adapters.
- `angle_degrees_per_module` — per-module-kind breakdown (grouped across
  layers). Covers every module the adapter was trained on, including
  `self_attn.{q,k,v,o}_proj`, `linear_attn.*`, and MoE
  (`mlp.{gate,shared_expert,switch_mlp,shared_expert_gate}.*`).
- `warning` — `"angle below threshold"` when mean `< 30°`, else `null`.
- `gate_status` — always `"angle_only_partial"` in phase 1a.

## Interpretation

| Mean angle | Action |
|---|---|
| `> 60°` | Healthy — stacks nearly orthogonal. Proceed. |
| `30°–60°` | Watch, but no rollback signal. Proceed. |
| `< 30°` | **Potential forgetting.** Run phase 1b win-rate eval before deciding. Rollback only if `Δwin-rate > 0.03`. |

## Current limitation

Angle alone cannot trigger rollback. A stack with `angle=20°` may still
be harmless if win-rate on prior domains is unchanged. The full gate
(angle **AND** win-rate drop) requires the paired eval wiring in
`src/eval/forgetting.py::check_all_previous` — see phase 1b.

## Per-expert analysis

For MoE adapters the report also carries an optional
`angle_degrees_per_expert` field — keyed `module → expert_idx → angle`
with JSON-string expert indices. The field is **present only when the
input adapters expose true expert structure** (pre-pivot
`.experts.<N>` MoE-LoRA keys, or post-pivot MLX `switch_mlp.*` 3D
stacks). Plain LoRA adapters omit it entirely, so existing callers
that parse the JSON strictly remain unaffected.

```json
"angle_degrees_per_expert": {
  "mlp.switch_mlp.down_proj": { "0": 82.1, "1": 77.4, "2": 89.0 },
  "mlp.down_proj_moe_lora":   { "0": 71.2, "1": 65.8 }
}
```

Interpret it against the module-averaged view:

- **Spread wide, mean high (e.g. 60–95°)** — experts are genuinely
  specializing and the adapter lives in mostly orthogonal directions.
  The post-pivot `switch_mlp.*` 88° mean observation is real.
- **Spread wide, mean near 90°** — averaging artifact. Random high-dim
  vectors tend toward orthogonal under the mean; per-expert angles
  will vary widely, some low. Do not trust the mean alone.
- **Spread tight and low (< 30°)** — genuine forgetting on those
  experts; the averaged gate will catch it too.

Per-expert angles are **diagnostic only** — the gate threshold remains
on the averaged per-module view. Use them when the mean looks
suspiciously uniform or you need to localize which experts are driving
the signal.

## Pairwise sweep across a fleet of adapters

`scripts/run_forgetting_sweep.py` runs `measure_forgetting_signal()` for
every ordered pair `(prior, new)` in a directory of adapters and emits a
single JSON matrix. Useful for cross-checking N post-pivot adapters at
once without N² manual invocations.

```bash
python scripts/run_forgetting_sweep.py \
    --adapters-dir /path/to/lora-qwen36-35b \
    --output results/forgetting-matrix.json
```

Each immediate subdirectory containing `adapters.safetensors` is treated
as one adapter (subdir name → adapter label). Output shape:

```json
{
  "adapters": ["chat-fr", "python", ...],
  "pairs": [
    {
      "prior": "chat-fr",
      "new": "python",
      "angle_degrees_mean": 78.4,
      "angle_degrees_per_module": { "self_attn.q_proj": 80.1, ... },
      "gate_status": "angle_only_partial"
    },
    ...
  ],
  "flags": {
    "any_pair_below_30": false,
    "min_mean_angle": 66.56,
    "worst_pair": ["typescript", "python"],
    "angle_threshold_degrees": 30.0
  }
}
```

- **Exit 0** — every pair's mean angle ≥ 30°.
- **Exit 1** — at least one pair fell below 30° (worst pair surfaced in
  `flags.worst_pair`, minimum in `flags.min_mean_angle`). Treat as a
  forgetting-risk signal; run the full phase 1b win-rate gate on the
  flagged pair before accepting the new stack.
- **Exit 2** — fewer than two adapters discovered (CLI misuse).

Progress logs go to stderr via stdlib `logging`, so the script runs on
the Mac Studio training venv (torch/safetensors/numpy only — no
`loguru` dependency).

## Per-module gate

The **aggregate** gate above (mean angle across every module kind)
masks divergences concentrated in a single module. The 2026-04-18
post-pivot sweep of 5 adapters (chat-fr, cpp, python, reasoning,
typescript) passed the aggregate gate 20/20 while
`mlp.shared_expert_gate` alone hit **17.3°** on the python↔typescript
pair — below the 30° threshold.

`src.eval.forgetting.apply_per_module_gate` adds a fine-grained
complement:

```python
from src.eval.forgetting import apply_per_module_gate

decision = apply_per_module_gate(
    per_module_angles={...},       # from measure_forgetting_signal
    winrate_drop=0.04,             # baseline − measured
    angle_threshold=30.0,
    winrate_drop_threshold=0.03,
    ignore_modules=None,           # → DEFAULT_PER_MODULE_IGNORE
)
# decision.failed             — bool (AND-logic: any module <30° AND drop>0.03)
# decision.offending_modules  — list[str] of below-threshold modules (angle only)
# decision.min_angle_module   — lowest-angle non-ignored module
# decision.min_angle_value    — its angle in degrees
```

### Why two gates coexist

- **Aggregate gate** (`gate_status_aggregate`) — coarse monitoring.
  Survives small per-module noise and matches historical reports.
- **Per-module gate** (`gate_status_per_module`) — flags a single
  module blowing out even when the mean stays safe. Runs with the
  same AND-logic (angle AND winrate-drop) so a low angle alone is
  never enough to rollback.

`measure_forgetting_signal` returns both statuses; the CLI
(`scripts/measure_forgetting.py`) and `ForgettingEvaluator.
check_all_previous` exit non-zero / rollback when **either** gate
fails — stricter is safer.

### Why `mlp.shared_expert_gate` is ignored by default

The Qwen3.5-35B-A3B shared-expert gate has a rank-1 delta shape
(`(hidden, r) @ (r, 1)` → `(hidden, 1)`). Subspace angles on a
one-column matrix are structurally constrained — the measured angle
is an artifact of the shape, not a meaningful forgetting signal.

Empirically (post-pivot sweep):

- All 16 other module kinds stay > 42° across every pair in the
  5-adapter matrix.
- `mlp.shared_expert_gate` alone drops to 17.3° on two pairs
  (python↔typescript, bidirectional) while their aggregate mean
  angle is still 66.6°.

The default ignore list (`DEFAULT_PER_MODULE_IGNORE`) therefore
excludes `mlp.shared_expert_gate`. Override via the
``ignore_modules`` kwarg when you want the strictest possible view
(e.g. investigating a specific pair):

```python
apply_per_module_gate(per_module_angles=..., winrate_drop=...,
                      ignore_modules=set())   # consider every module
apply_per_module_gate(per_module_angles=..., winrate_drop=...,
                      ignore_modules={"mlp.shared_expert_gate",
                                       "mlp.gate"})  # add more
```

A CLI flag to flip this behaviour (e.g.
`--no-ignore-shared-expert-gate`) is a straightforward follow-up
when a human operator needs it; the API already supports the
override.

### Updated CLI exit-code policy

| Scenario | `gate_status_aggregate` | `gate_status_per_module` | exit |
|---|---|---|---|
| All modules safe, full gate passes | `pass` | `pass` | 0 |
| Aggregate fails (mean < 30° AND drop > 0.03) | `fail` | any | **1** |
| Per-module fails (any non-ignored module < 30° AND drop > 0.03) | any | `fail` | **1** |
| Angle-only (no win-rate inputs) | `angle_only_partial` | `angle_only_partial` | 0 (informational) |

`offending_modules` / `min_angle_module` / `min_angle_value` are
populated even in angle-only mode; they never force a non-zero exit
on their own (AND-logic with the win-rate drop is preserved).

## Win-rate scoring modes

The win-rate half of the gate compares generated responses against a
reference answer per prompt. Two scoring modes ship with the framework;
choose via `--scorer-module` (CLI) or the `scorer=` kwarg on
`measure_forgetting_signal()`.

### Containment (default)

`src.eval.scorers.containment_score` — async wrapper around the legacy
heuristic: score = 1.0 if the reference string is a substring of the
response (case-insensitive); otherwise the fraction of
whitespace-split reference tokens present in the response. Returns 0.0
for empty references.

**Use when:**
- References are short, unambiguous strings (single facts, short
  phrases, IDs, code tokens).
- You cannot or will not call an external judge (air-gapped, CI cost).
- You want a deterministic, reproducible score — the same
  response/reference always produces the same value.

**Don't use when:**
- References are long free-form answers (paraphrases score near 0).
- Correctness depends on semantics, not surface tokens.

Default behaviour, no flag needed:

```bash
python scripts/measure_forgetting.py \
    --prior-adapter ... --new-adapter ... \
    --eval-dataset  ... --generate-fn-module ... \
    --winrate-baseline-score 0.82
```

### LLM judge

`src.eval.scorers.JudgeScorer` — async callable that reuses
`src.eval.stack_eval.JUDGE_PROMPT` (the canonical per-stack evaluator
template) and calls the configured judge client. Returns the `score`
field from the judge's JSON response, clipped to `[0, 1]`.
Bad-JSON responses return `0.0` with a warning — the gate stays
robust when the judge is flaky.

**Use when:**
- References are long or free-form (essays, code explanations, French
  translation critique).
- You care about semantic equivalence, not surface tokens.
- A judge client is already available (Mistral-Large via
  `StackEvaluator`'s `judge_client`).

**Don't use when:**
- Budget / latency is tight (one judge call per prompt per adapter).
- The judge model lives behind a flaky or slow network boundary
  without retries.

Wire a `JudgeScorer` via a small wrapper module that exposes a
concrete scorer callable at import time, then point
`--scorer-module` at it:

```python
# my_judge_scorer.py
from src.eval.scorers import JudgeScorer
from src.serving.judge_client import make_judge_client  # your client

scorer = JudgeScorer(make_judge_client(), judge_model="mistral-large")
```

```bash
python scripts/measure_forgetting.py \
    --prior-adapter ... --new-adapter ... \
    --eval-dataset  ... --generate-fn-module src.serving.mlx_client:generate \
    --winrate-baseline-score 0.82 \
    --scorer-module my_judge_scorer:scorer
```

The scorer is resolved via the same `module:attr` locator shape used
by `--generate-fn-module`. Sync and async callables are both
accepted; async ones are driven via `asyncio.run()` per prompt.

### When to switch modes

Typical playbook: containment for fast iteration during stack
training, judge for the "real" post-stack gate check before
registering an adapter with the router. The gate thresholds
(`angle < 30°` AND `winrate_drop > 0.03`) are unchanged across
modes — only the per-prompt score function differs.

## Adapter health sanity

Before running the forgetting gate at all, check that each new adapter
actually *trained* — the 2026-04-19 pre-pivot MoE-LoRA audit
(`docs/research/2026-04-19-prepivot-moe-lora-audit.md`) found 35
adapters with every `lora_B` tensor stuck at zero-init, which makes
every forgetting angle trivially 0° and hides weeks of wasted GPU.

```bash
# single adapter
python scripts/validate_adapter_health.py path/to/adapter.safetensors

# entire fleet (recursive)
python scripts/validate_adapter_health.py --adapters-dir output/lora-qwen36-35b
```

Exit `1` means *every* `lora_B` tensor has Frobenius norm below
`--epsilon` (default `1e-6`). The default is chosen so numerical
noise from an honestly-trained adapter always clears it while a
pristine zero-init adapter cannot; pass `--epsilon 1e-4` if you
need a stricter floor (e.g. catching near-zero adapters whose
gradient path *did* fire but barely).

CI enforces this as a smoke check in `.github/workflows/validators.yml`.

### Audit 2026-04-19: full adapter sweep

`scripts/sweep_adapter_health.py` walks a directory and emits structured
per-adapter JSON (same health logic as `validate_adapter_health.py`,
but batched). Results for every MLX adapter on Mac Studio are in
`results/adapter-health-sweep.json`:

- **Post-pivot `output/micro-kiki/lora-qwen36-35b/`**: 35/35 final
  `adapters.safetensors` healthy (+28/28 intermediate checkpoints).
- **Pre-pivot `output/micro-kiki/stacks-v3-r16/`**: 35/35 final
  adapters degenerate (every `lora_B` tensor at exact zero, 512/512
  per adapter).
- **Surprises: none.** Both 2026-04-19 empirical predictions hold
  across the full fleet.

Run again with:

```bash
python scripts/sweep_adapter_health.py \
    --adapters-dir <dir> \
    --output results/<label>.json
```

## Running full gate on Mac Studio

The win-rate half of the gate requires a real base model loaded on
the Mac Studio MLX-LM server (`studio:8000`, see
`src/serving/AGENTS.md`). `src/serving/mlx_client.py` (async httpx
with exponential retry on 5xx / `ConnectError`) exposes a sync
`generate()` that plugs into `--generate-fn-module`.

Prerequisites:

- SSH chain reachable: `ssh grosmac -> ssh studio`.
- MLX-LM server running on `studio:8000` with the base model loaded
  and the target adapter hot-swappable (subprocess restart on swap
  is acceptable — the client does not assume in-process hot-swap).
- Held-out eval JSONL in `data/eval/<stack>/heldout.jsonl` shaped as
  `{"prompt": ..., "reference": ...}` (one entry per line).

Concrete invocation from the repo root on the Studio venv:

```bash
ssh grosmac "ssh studio 'cd ~/micro-kiki && \
    python scripts/measure_forgetting.py \
      --prior-adapter output/stacks/stack-03-cpp/adapters.safetensors \
      --new-adapter   output/stacks/stack-04-rust/adapters.safetensors \
      --eval-dataset  data/eval/stack-03-cpp/heldout.jsonl \
      --generate-fn-module src.serving.mlx_client:generate \
      --winrate-baseline-score 0.82 \
      --output results/forgetting-stack04-vs-stack03.json'"
```

Override the server target with the `MLX_HOST` / `MLX_MODEL`
environment variables (defaults `http://studio:8000` and
`qwen3.6-35b`). Expected JSON fields:

- `gate_status_aggregate` — `pass` / `fail` (no longer
  `angle_only_partial` once the three win-rate flags are set).
- `gate_status_per_module` — same shape, stricter (any non-ignored
  module below 30° AND drop > 0.03).
- `winrate_measured` / `winrate_drop` — the win-rate half computed
  against `--winrate-baseline-score`.
- `angle_degrees_mean` / `angle_degrees_per_module` — unchanged
  from the angle-only output.

## Reference

Full protocol, alternatives considered, and roadmap:
`.omc/brainstorm-oplora.md`.
