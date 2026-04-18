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

## Reference

Full protocol, alternatives considered, and roadmap:
`.omc/brainstorm-oplora.md`.
