# Brainstorm: OPLoRA wiring into MLX training pipeline

**Date:** 2026-04-18
**Domain:** ML / training infrastructure
**Status:** DRAFT — design only, no code / configs touched
**Owner:** clement@lelectronrare.fr
**Target paper:** OPLoRA, arXiv 2510.13003
**Relevant source tree:**
- `src/stacks/oplora.py` (23-line `orthogonal_projection` + `init_oplora_experts`)
- `src/stacks/trainer.py` (`StackTrainer`, `LoRAConfig`, PEFT path — torch stack, Mac-only)
- `src/eval/forgetting.py` (`GradientSubspaceAnalyzer`, `ForgettingEvaluator`, CLI `python -m src.eval.forgetting`)
- `configs/stack-*.yaml` (PEFT-style, has `init_lora_weights: oplora` from stack 04 onward)
- `configs/mlx-per-domain/*.yaml` (the actual MLX pipeline — NO `init_lora_weights` field)
- `configs/mlx-curriculum/{foundations,coding-core,...}.yaml` (phase-grouped MLX configs)
- `configs/micro_kiki/brainstacks.yaml` (curriculum, `null_space` legacy keys)
- `docs/training/README.md` (phase-1/2/3 MLX workflow)
- `tests/test_oplora.py` (3 passing torch-only tests)

## Problem Frame

### What OPLoRA prevents
Catastrophic forgetting across 32 **sequential** LoRA adapters. Each new adapter's `B` (down-projection) is initialised in the orthogonal complement of the subspace spanned by prior adapters' `B` matrices, so the new adapter's updates are geometrically prevented from colliding with prior ones.

`src/stacks/oplora.py` already implements the math in torch:
```python
def orthogonal_projection(prior_subspace, dim):
    q, _ = torch.linalg.qr(prior_subspace.float())
    return (I - q @ q.T)
```
and `init_oplora_experts(in_features, rank, num_experts, prior)` returns projected random inits. `tests/test_oplora.py` asserts `(proj @ prior).abs().max() < 0.01` and that cosine similarity between prior and projected init stays `< 0.2`.

### Current wiring state (observed, not claimed)
1. **PEFT / torch pipeline** (`StackTrainer`): accepts `init_lora_weights: oplora` via `PeftLoraConfig(init_lora_weights=...)`. Stack configs 04–32 all set this flag. But `StackTrainer.train()` **never touches** `src/stacks/oplora.py` — it just passes the string to PEFT, which does not know about our custom `init_oplora_experts`. So the flag is currently a no-op unless PEFT has a builtin `oplora` init (it does not, as of 0.x).
2. **MLX pipeline** (`python -m mlx_lm lora --config configs/mlx-curriculum/*.yaml`): the one actually used on Mac Studio per `docs/training/README.md`. YAML schema has no `init_lora_weights`; adapter weights are created internally by `mlx_lm`. **No hook into `oplora.py` exists.**
3. **Forgetting check** (`src/eval/forgetting.py`): functional, **but automatic measurement is not wired** — `check_all_previous()` raises `NotImplementedError` unless `results=[...pre-computed...]` is passed in. A human currently has to produce angle + win-rate numbers externally.

### Why OPLoRA is deferrable
- Curriculum ordering already front-loads the most foundational domains (`chat-fr → reasoning → python → ts → cpp → rust`), so the *first* stacks to risk interference train on the largest, cleanest data — they are the least likely to forget, and the most likely to dominate any subspace conflict.
- The forgetting gate (`angle < 30° AND Δwr > 0.03`) gives observable rollback semantics. We can detect damage before spending the compute on all 32.
- Rank-8/12/16 adapters on q/k/v/o are a tiny subspace of a 35B param model: collision probability is low without OPLoRA, simply because the adapter rank is `<< dim`.

### What gate rollback means (needs clarification)
`ForgettingReport.should_rollback = True` when `angle < 30° AND winrate_drop > 0.03` on any prior stack. In the code, rollback only *flags* the event — there is no automated re-train. Interpretation choices:
- **Soft rollback** (current behaviour): discard the newly trained adapter, keep curriculum pointer at the previous stack, try again with different hyperparams or OPLoRA init.
- **Hard rollback**: discard new adapter AND re-train the damaged prior stack. Not needed — the prior adapter weights on disk are unchanged; only *measured* win-rate dropped, which is a property of the combined stack, not the prior adapter itself.

Design decision for this experiment: **soft rollback** is enough. We never overwrite a prior `adapters.safetensors`. A failed new stack means "don't merge, re-init with OPLoRA."

---

## Approaches

### A. Offline projection (pre-training patch)

| | |
|---|---|
| **Core idea** | Before training stack N, read all prior stacks' `adapters.safetensors` from disk, compute `P = I - QQ^T` over the stacked `B` matrices, apply `P` to a fresh random `B` init, write to `init_oplora_<stack-N>.safetensors`. Hand that file to `mlx_lm lora` via `resume_adapter_file` or an explicit `init_adapter_file` override. |
| **Mechanism** | Python script outside the training loop. Uses the existing `src/stacks/oplora.py::orthogonal_projection` (torch) or a numpy mirror (`numpy.linalg.qr`). Reads `mx.safetensors` via `safetensors.numpy`. Writes new safetensors keyed by the MLX LoRA parameter layout (`model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_{a,b}`). |
| **Best for** | Projects where training runtime is a black box (we don't want to fork `mlx_lm`). Keeps OPLoRA math co-located in `src/stacks/oplora.py`. Reversible — generated init files live under `output/oplora_inits/`, removable with `rm`. |
| **Worst for** | Deep-geometry purists: projection is only applied to `B` at t=0. During training, `B` can drift back into prior subspace. OPLoRA paper actually also projects gradients during training (not done here). So "offline A" is OPLoRA-init, not full-OPLoRA. |
| **Hidden assumption** | That `mlx_lm lora` has a way to load an initial adapter file. `resume_adapter_file` exists (`docs/training/README.md` phase 2/3 resumes from phase 1). Needs verification that resuming zero steps from a synthesised file produces the expected starting weights and that `mlx_lm` does not silently re-init. |
| **Smallest prototype** | (1) write `scripts/build_oplora_init.py` that: loads N prior `adapters.safetensors`, stacks all B matrices per layer×projection, QR-decomposes, makes projected random init, writes init file. (2) train stack 04 with `resume_adapter_file: output/oplora_inits/stack-04.safetensors`, compare first-step loss vs. default. |
| **Hours** | Script: 3–4 h (torch+safetensors). Verification: 1 h. Per-stack overhead at run time: ~5–10 min CPU (QR of `rank × 40layers × 4projections × (N-1)_stacks` matrices). |

### B. Custom MLX hook

| | |
|---|---|
| **Core idea** | Fork `mlx_lm lora` (or wrap via subprocess + callback) to inject an `init_callback(module, prior_subspace)` that runs `orthogonal_projection` inside MLX at adapter creation. |
| **Mechanism** | Vendored copy of `mlx_lm/tuner/lora.py` with a patched `linear_to_lora_layers()`, reading prior subspace from a config-specified dir. Needs an MLX port of `orthogonal_projection` (`mx.linalg.qr` exists). |
| **Best for** | Long-term roadmap where OPLoRA gradient projection (not just init) becomes needed, or where the same hook carries other safety features (e.g., angle-logging during training). |
| **Worst for** | Short term. MLX compat risk: `mlx_lm` moves fast. Every `mlx_lm` upgrade forces a merge. `docs/specs/mlx-lm-fork-reference.md` already exists (we already maintain a fork) — but the fork is tuned for 3x Metal limit and per-phase specifics; piling an OPLoRA hook on top doubles maintenance surface. |
| **Hidden assumption** | MLX's `linear_to_lora_layers` is a clean injection point. Needs inspection. Also that the fork's release cadence can tolerate the patch. |
| **Smallest prototype** | ~8 h: fork mlx_lm tuner, port QR projection, write init hook, unit test angle on synthetic weights. |
| **Hours** | Prototype 8 h + ongoing 0.5 h per mlx_lm upgrade. |

### C. Skip + measure (empirical)

| | |
|---|---|
| **Core idea** | Do not wire OPLoRA. Train stacks 04, 05, 06 with `mlx_lm lora` default init (kaiming `A`, zero `B`). After each, run `src/eval/forgetting.py` with pre-computed angle + win-rate across all prior stacks. If gate trips → fall back to A. If all three pass → continue through stack 32 unassisted; spot-check at stacks 10, 20, 32. |
| **Mechanism** | No code changes. One `scripts/measure_forgetting.py` to compute angle + win-rate automatically (this needs writing either way — it is a prerequisite for A and B too). |
| **Best for** | Information value. We don't currently know whether OPLoRA is *necessary* on 35B-A3B — the OPLoRA paper was on dense 7B–13B. The native MoE base already isolates experts; attention-only LoRA at rank ≤16 may be below the interference threshold. |
| **Worst for** | Irreversible *compute* loss if a late stack (say, 25) trips the gate — we've spent 25×45min = ~19 h training before finding out. Mitigation: spot-check cadence every 3 stacks. |
| **Hidden assumption** | That the forgetting gate (`angle<30°` **AND** `Δwr>0.03`) actually catches problems. The `AND` is lenient — a stack with `angle=20°, Δwr=0.02` passes but may still be subtly damaging. Literature: OPLoRA paper uses win-rate alone. |
| **Smallest prototype** | Write `scripts/measure_forgetting.py`: (1) load base + adapter_k, sample 100 tokens per prior domain's eval set, compute per-token logit win-rate vs. base (using teacher scores as referee OR pairwise judge), (2) compute LoRA-B subspace angle vs. prior stacks using `GradientSubspaceAnalyzer.compute_angle`. Emits JSON for `python -m src.eval.forgetting --results-file`. |
| **Hours** | Measurement script: 4–6 h. Per-stack measurement time: 15–20 min (win-rate forward passes on 100 samples × N prior stacks). |

---

## Tension Map

| Axis | A (offline) | B (MLX hook) | C (skip+measure) |
|---|---|---|---|
| MLX code change | none | fork patch | none |
| Theoretical fidelity | init only | init + optional grad | none (empirical) |
| Reversibility | delete init file | revert fork | nothing to revert |
| Compute-loss risk | ~0 | ~0 | up to 19 h if late gate trip |
| Info value | low (known tech works) | low | high (tells us whether we even need OPLoRA) |
| MLX upstream drift risk | none | high | none |
| Unlocks gradient-projection OPLoRA later | no | yes | no |
| Requires measurement script | yes | yes | yes (blocker) |
| Wall-clock per stack added | 5–10 min | 1–2 min | 15–20 min (already required) |

Note: C does not *save* measurement cost; A and B both still need the measurement script to *validate* that OPLoRA is working. The measurement script is therefore the common prerequisite and should be built first regardless.

---

## Ranking

Score per axis, 0–10:

| Approach | Implementability | Reversibility | Info value | Total | Why it wins | Why it loses |
|---|---|---|---|---|---|---|
| A | 8 | 9 | 4 | 21 | Clean, local, uses existing `oplora.py`. Handles `resume_adapter_file` cleanly. | Only protects init. Does not answer "is OPLoRA needed?". |
| B | 3 | 4 | 5 | 12 | Future-proof (gradient projection, telemetry). | Fork maintenance, MLX churn, slowest to prototype. |
| C | 10 | 10 | 9 | 29 | No code written unless proved needed. Produces an empirical answer about the 35B-A3B regime. | 19 h compute risk if gate trips at stack 25. Mitigable with spot checks. |

---

## Recommendation

**C first, A as fallback. Never B unless the roadmap forces gradient-projection OPLoRA.**

Confidence: **medium-high**. The primary uncertainty is whether the measurement script will reliably produce monotonic angle/win-rate metrics with small sample sizes (n=100). That is a tooling risk independent of OPLoRA choice and has to be solved anyway.

### Go/no-go criteria
- **Go C**: measurement script is built and validated on stack-03 vs stack-01/stack-02 (should produce angle ≈ 90° and Δwr ≈ 0, because they were trained with default init on disjoint data — if it produces garbage, fix the tool before trusting it).
- **No-go C, jump straight to A**:
  - if preliminary measurement on stacks 01–03 shows `angle < 45°` pairwise — this means rank 16 is already colliding *without* OPLoRA even on disjoint foundational data, and we need the projection guarantee up front.
  - if dataset overlap across target domains is > 5% (audit via dedup report); OPLoRA is more important with overlapping inputs.

---

## Experimental Protocol (C — Skip and Measure)

**Preconditions** (must hold before starting):
1. Stacks 01, 02, 03 (chat-fr, reasoning, python) trained and saved under `~/KIKI-Mac_tunner/output/micro-kiki/stack-0{1,2,3}-*/adapters.safetensors` (per `docs/training/README.md`).
2. Held-out eval sets exist at `~/KIKI-Mac_tunner/data/micro-kiki/<domain>/valid.jsonl` for every curriculum domain.
3. Measurement script `scripts/measure_forgetting.py` (to be written, out of scope here) emits JSON of shape:
   ```json
   [{"stack_id": "chat-fr", "angle": 87.3, "winrate_base": 0.52, "winrate_adapted": 0.51}, ...]
   ```
   consumable by `python -m src.eval.forgetting --results-file <path>`.

**Numbered steps:**

1. **Baseline capture (pre-stack-04)**
   - For each trained stack {01,02,03}, run `scripts/measure_forgetting.py --stack <id> --prior-stacks <list-of-earlier>` and store output at `results/forgetting/baseline-{stack}.json`.
   - Expected: stack-02 vs stack-01 → `angle ∈ [60°, 90°]`, `winrate_drop ∈ [-0.02, +0.02]`. Stack-03 vs {01,02} similar.
   - If baseline itself shows `angle < 45°`, the tool or the training regime is suspect — STOP, investigate, do not proceed.

2. **Train stack 04 (typescript) with default init**
   - Use existing `configs/mlx-curriculum/coding-core.yaml` (rank 12, scale 2.0, seq 4096, 400 iters, LR 3e-5 cosine decay, warmup 40) via `python -m mlx_lm lora --config configs/mlx-curriculum/coding-core.yaml` after editing `data:` and `adapter_path:` for the typescript domain.
   - DO NOT set `init_adapter_file`. This is the control arm.
   - Record: wall-clock, val-loss trajectory, final val-loss, peak RSS, path to `adapters.safetensors`.

3. **Measure post-stack-04**
   - `python scripts/measure_forgetting.py --stack typescript --prior-stacks chat-fr,reasoning,python --n-samples 100 --output results/forgetting/stack-04.json`
   - `python -m src.eval.forgetting typescript --results-file results/forgetting/stack-04.json --output-dir results/forgetting/`
   - Inspect `results/forgetting/forgetting-typescript.json`. Gate is `angle < 30° AND winrate_drop > 0.03`.

4. **Apply gate**
   - **Pass** (no prior shows `should_rollback: True`) → proceed to stack 05.
   - **Fail** → stop. Move to Fallback Protocol A.
   - **Warning zone** (any prior has `angle < 45°` or `winrate_drop > 0.02`): proceed to stack 05 but increase measurement sample to `n-samples 300` on next stack to tighten the estimate.

5. **Train stack 05 (cpp), measure, gate** — repeat steps 2–4 with stack 05, prior list `{chat-fr, reasoning, python, typescript}`.

6. **Train stack 06 (rust), measure, gate** — repeat with stack 06, prior list `{...stacks 01..05}`.

7. **Decision point (after stack 06)**
   - All 3 pass → continue sequential training without OPLoRA. Measurement cadence relaxes to every 3 stacks (at stacks 09, 12, 15, 18, 21, 24, 27, 30, 32).
   - Any fail → Fallback Protocol A, re-train the failed stack only (soft rollback).

**Measurement cadence:** Every stack for 04–06 (gate window). Every 3 stacks thereafter (spot checks). Always at stack 32 (final).

---

## Fallback Protocol (A — Offline projection init)

Triggered when C's gate fails or preconditions are violated.

1. **Collect prior B matrices**
   - For each prior stack `s ∈ {stack-01..stack-(N-1)}`, load `~/KIKI-Mac_tunner/output/micro-kiki/stack-NN-<s>/adapters.safetensors`.
   - MLX LoRA parameter naming: `model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_{a,b}`. Collect `lora_b` tensors grouped by `(layer, projection)`.

2. **Compute projection per (layer, projection) pair**
   - For each `(i, p)`, concatenate all prior `lora_b[i,p]` along rank axis → `prior_subspace` of shape `(dim, sum_of_prior_ranks)`.
   - `P = I - QQ^T` via `src.stacks.oplora.orthogonal_projection(prior_subspace, dim)`. This is 40 layers × 4 projections × (N-1) stacks worth of QRs. Compute cost ≈ 5–10 min on CPU for a mid-curriculum stack.

3. **Build init adapter file**
   - For the new stack N at rank `r_N`, create:
     - `lora_a[i,p]` = kaiming_uniform init (standard MLX default).
     - `lora_b[i,p]` = `P @ randn(dim, r_N) * 0.01`.
   - Pack into safetensors matching MLX LoRA layout, write to `output/oplora_inits/stack-NN-<domain>.safetensors`.

4. **Verify projection quality (pre-flight)**
   - For each `(i, p)`: compute `angle(prior_subspace, new_b)` using `GradientSubspaceAnalyzer.compute_angle`. Expect `>= 85°` (near-orthogonal). If `< 85°` on any layer, `prior_subspace` is rank-saturated (sum of prior ranks ≥ dim — unlikely until late curriculum with dim=3072 and rank 16) — halve the new stack's rank or drop oldest prior from the subspace.

5. **Train with init file**
   - Pass to mlx_lm via `resume_adapter_file: output/oplora_inits/stack-NN-<domain>.safetensors` in the phase-1 config YAML. Set `iters` unchanged; mlx_lm resumes at step 0 from the supplied weights (no optimizer state — fresh Adam).
   - Train normally through all 3 phases.

6. **Post-train: same forgetting measurement as step 3 of Protocol C**
   - Expected improvement: `angle` at least +15° vs. the failed-gate C measurement. `winrate_drop` at or below 0.02.

7. **If A also fails**
   - Consider reducing rank for the new stack (OPLoRA's guarantee weakens as adapter rank approaches residual subspace dimension).
   - Consider dropping the newly trained stack from the curriculum entirely if its gain doesn't justify the cost.
   - Do NOT escalate to Protocol B without explicit human approval — fork maintenance is not a one-person job.

**Overhead:** 5–10 min projection + ~1 min safetensors write + 1 min verification per stack.

---

## Key Metrics

| Metric | Source | Threshold | Action on violation |
|---|---|---|---|
| Gradient subspace angle | `GradientSubspaceAnalyzer.compute_angle(base_grads, adapted_grads)` | `>= 30°` | If `< 30°` AND winrate_drop `> 0.03` → rollback |
| Win-rate on prior domain | `scripts/measure_forgetting.py` pairwise-judge or teacher-score on 100 held-out examples | `drop <= 0.03` | See above (AND-gate) |
| LoRA-B subspace saturation | `rank(concat(prior_b)) / dim` | `< 0.5` | If `>= 0.5`, drop oldest prior stack from projection subspace |
| Training val loss | `mlx_lm` stderr log | converges `<= 0.7` at phase-1 end (per `docs/training/README.md` table) | If not → retrain with shorter seq first |
| Peak Metal memory | `sysctl iogpu.wired_limit_mb` + MLX reporting | `<= 200 GB` phase 3 | If > 200 GB → lower batch or rank |
| Measurement confidence | std-dev of winrate across 3 resamples (n=100) | `std <= 0.015` | If > 0.015 → bump n_samples to 300 |

---

## Stop Conditions

**Abort whole experiment and escalate to human** if any of:
1. Baseline measurement (step 1) produces `angle < 45°` on stacks 01↔02 or 01↔03 — measurement tool or training regime is broken, need to diagnose before any OPLoRA decision is meaningful.
2. Both C and A fail the gate at the same stack (same domain, same prior list).
3. Three consecutive stacks in Protocol C trigger the warning zone (`angle < 45°` OR `winrate_drop > 0.02`) even without triggering the hard gate — indicates creeping damage that the AND-gate is failing to catch.
4. `mlx_lm` `resume_adapter_file` turns out not to be honored (Protocol A infrastructure assumption broken). Verify this in the Protocol-A smallest-prototype step before committing to fallback.
5. Measurement cost exceeds training cost (> 45 min per stack) — re-scope the experiment to cheaper proxy metrics (e.g., weight-space angle only, no win-rate).

**Graceful stop** (experiment done, no escalation):
- All 32 stacks trained under Protocol C with all measurements passing → commit run log, update `docs/training/README.md` with observed angles & win-rates table, close experiment, mark `src/stacks/oplora.py` as "available but unused" in `src/stacks/CLAUDE.md`.
- All 32 trained with one or more stacks using Protocol A fallback → commit, update README with list of stacks that required OPLoRA init, preserve `scripts/build_oplora_init.py` in-tree.

---

## Timeline

### Compute hours (Mac Studio wall-clock)

| Phase | Activity | Per-stack | Count | Total |
|---|---|---|---|---|
| Setup | Baseline measurement (steps 01–03 already trained) | — | 3 | 1 h |
| C-gate window | Train stack 04, 05, 06 × 3 phases | ~12 h (2+6+4 per `docs/training/README.md`) | 3 | 36 h |
| C-gate window | Measurement after each | 15–20 min × (1..5 priors) | 3 | 2 h |
| Post-gate curriculum | Train stacks 07–32 (26 stacks) | ~12 h | 26 | 312 h |
| Post-gate curriculum | Spot-check measurements at stacks 09, 12, 15, 18, 21, 24, 27, 30, 32 | ~30 min | 9 | 4.5 h |
| **Baseline total (no fallback)** | | | | **~356 h ≈ 15 days** |
| Fallback (if any) | Projection + re-train + measure | +12 h per triggered stack | 0..N | 0..N × 12 h |

### Human hours

| Phase | Activity | Hours |
|---|---|---|
| Prereq | Build `scripts/measure_forgetting.py` | 6 |
| Prereq | Validate on stacks 01–03 (baseline) | 2 |
| Protocol C | Monitor stack-04/05/06 runs + interpret gates | 3 |
| Decision | Post-stack-06 review, write-up | 2 |
| Fallback (if triggered) | Build `scripts/build_oplora_init.py` | 4 |
| Fallback (if triggered) | Verify pre-flight projection quality, wire into YAML | 2 |
| Documentation | Update `docs/training/README.md` with results table | 2 |
| **Human total (happy path)** | | **15 h** |
| **Human total (one fallback trigger)** | | **21 h** |

### Dependencies

- `scripts/measure_forgetting.py` is the critical path. It is a prerequisite for Protocol C, A, and any future B. Build it first, validate on already-trained stacks 01–03. Do not start stack 04 training until baseline numbers look sane.
- `docs/specs/mlx-lm-fork-reference.md` should be re-read before attempting Protocol A's `resume_adapter_file` assumption — the fork may have diverged.

---

## Open Questions (for human review before starting)

1. **Win-rate judge choice.** Teacher-scoring (Qwen3-Coder-480B MLX 4bit) vs. pairwise LLM-as-judge vs. exact-match/BLEU on eval completions. `docs/training/README.md` doesn't specify; `brainstacks.yaml` has `forgetting.max_delta: 0.03` but leaves judge unspecified. Recommendation: teacher log-prob of gold answer, cheap and deterministic.
2. **Rank of prior subspace when N large.** At stack 32 with rank 16 per prior, sum-of-prior-ranks = 31 × 16 = 496 vs. `dim=3072`. Still ~16% saturation — safe, but worth monitoring. Past what N does the projection stop being well-conditioned? Probably never at rank 16, but watch.
3. **PEFT pipeline alignment.** `configs/stack-*.yaml` have `init_lora_weights: oplora` but `StackTrainer.train()` doesn't wire this to `src/stacks/oplora.py::init_oplora_experts`. This is dead config even in the torch path. Should we either (a) delete the flag from configs or (b) implement the wiring in `StackTrainer`? This brainstorm does not resolve this — out of scope (MLX pipeline is the real target).
4. **Soft vs hard rollback confirmation.** If stack-15 breaks stack-03, do we re-run stack-03 fine-tuning with a projected-away stack-15 subspace (hard rollback, symmetric) or just discard stack-15 (soft)? This doc assumes soft. Confirm before codifying.
5. **`resume_adapter_file` semantics.** Does MLX `mlx_lm lora` start optimizer state fresh when resuming from a file with no optimizer state shard? Needs empirical verification before Protocol A is trusted.

---

## Summary
- OPLoRA math already exists (`src/stacks/oplora.py`, 23 lines, tested). Zero MLX integration.
- PEFT configs set `init_lora_weights: oplora` but `StackTrainer` doesn't propagate it — this is dead flag, unrelated to the real MLX training path.
- MLX is the actual training path per `docs/training/README.md`. No OPLoRA touchpoint there.
- The forgetting framework is half-wired: thresholds defined, report class defined, automatic measurement `NotImplementedError`. `scripts/measure_forgetting.py` is the common prerequisite for every approach.
- Recommended plan: Protocol C (train 04, 05, 06 plain, measure after each, gate on `angle<30° AND Δwr>0.03`). Fall back to Protocol A (offline projection → init safetensors → `resume_adapter_file`) on gate trip.
- Protocol B (MLX fork hook) is explicitly rejected for now: high maintenance, low marginal benefit vs. A.
- Timeline: ~15 days compute, ~15 h human (happy path). Fallback adds 12 h compute + 6 h human per triggered stack.
