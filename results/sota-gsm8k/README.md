# GSM8K sweep — 2026-04-22

Zero-shot GSM8K (N=200 sampled from 1319-problem test split).
Fixture downloaded from `openai/gsm8k` (MIT). Runner
`scripts/eval_gsm8k_v4.py`, prompt `{question}\nAnswer:`, answer
extraction via `####\s*(-?[\d,]+)` regex with last-number
fallback.

## Results

| Label | pass@1 | gen_s | Δ vs base |
|---|---|---|---|
| **gsm8k-base-200** (no adapter) | **0.455** (91/200) | 1172 | — |
| gsm8k-math-200 | 0.475 (95/200) | 2071 | **+0.020** (within ±3.5 pp noise) |
| gsm8k-reasoning-200 | **0.045** (9/200) | 400 | **−0.410** 🚨 |

## Findings

### Math adapter — net-neutral

- +2 pp on GSM8K is within the N=200 std error (±3.5 pp), so
  the signal is **indistinguishable from noise**.
- Combined with the −0.110 smoothing on HumanEval-cross
  (2026-04-22 Piste A), the math adapter produces no
  measurable net benefit under any routing regime.
- **Action** : do NOT ship as a dedicated math adapter to HF.
  Either re-train with a purer math dataset (e.g. MATH +
  GSM8K + AIME only) or drop the domain entirely.

### Reasoning adapter — broken

- **−41 pp** is catastrophic. The model drops from 45.5% to
  4.5% on GSM8K.
- Generation time is 3× shorter (400s vs 1172s base) — the
  model is emitting truncated / degenerate outputs. Likely
  causes :
  1. Chat template mismatch (training used a different
     prompt format than `{question}\nAnswer:`).
  2. EOS token poisoning — training data had early EOS.
  3. Loss converged on the training format but doesn't
     generalize to GSM8K's structure.
- **Action** :
  1. **Remove `reasoning` from `clemsail/micro-kiki-v4-sota`
     on HF immediately.**
  2. Inspect `data/v3/reasoning/` to diagnose the root cause.
  3. Consider retraining with a mixed `Answer:` / `####`
     format and explicit CoT examples.

## Publication gate

A single adapter that's +2 pp in-domain and −11 pp
out-of-domain is neutral at best. A single adapter that's
−41 pp in-domain is actively toxic. We can't ship 35 adapters
without per-adapter validation.

**Proposed gate for HF release** :

1. Adapter health : `lora_B` norm > ε — 76/76 pass (2026-04-22).
2. In-domain uplift : `pass@1(adapter) − pass@1(base) ≥ 0` on
   a domain-appropriate benchmark.
3. Out-of-domain bound : `pass@1(adapter) − pass@1(base) ≥ −0.15`
   on at least one unrelated benchmark.

Adapters failing (2) or (3) get flagged `+UNSAFE` in the HF
model card and are not listed in the auto-router.

## Next

Run `scripts/run_sota_publish.sh` to propagate this validation
across the 6 coding + non-coding adapters already trained,
then update `clemsail/micro-kiki-v4-sota` model card with the
pass/fail matrix.
