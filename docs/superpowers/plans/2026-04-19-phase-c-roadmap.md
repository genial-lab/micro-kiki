# Phase C — Paper A Main-Track Upgrade Roadmap

**Status:** Strategic roadmap (NOT an implementation plan — see sibling plans for execution)
**Created:** 2026-04-19
**Based on:** Session findings in `results/plan6-findings.md`, merged via commit `9e920ad`

## Context

Paper A draft (workshop-grade) was pivoted on 2026-04-19 from a compression-ratio claim to a 3-contribution framing:
1. **Methodological**: torch-native batched VQC @ 3000× PennyLane speedup (done, merged)
2. **Architectural**: learned projection rescues 4-qubit VQC on 10-class routing (done, merged)
3. **Empirical**: rigorous 3-axis ablation, ceiling ~0.40 quantified (done, merged)

To move from workshop-track to **main-track publishable**, 5 independent sub-projects must each land rigorously. Each produces a standalone result that survives reviewer scrutiny.

## The 5 sub-plans

| Sub | Title | Owner | Effort | Status |
|---|---|---|---|---|
| **C1** | Baseline comparison (classical classifiers vs VQC) | Done | ~1h actual | **Completed 2026-04-19 (commits `68dd690`..`73ceb40`)** — see `docs/paper-a/c1-baseline-results.md` |
| **C2** | Downstream LLM eval (router → actual model selection quality, not just accuracy) | TBD | 3-5 days | Needs brainstorm before plan |
| **C3** | Real dialogue corpus (replace `data/final` synthetic with real user queries) | TBD | 5-7 days | Needs data-source decision |
| **C4** | Scale test (10 → 35 domains with full MoE stack routing) | TBD | 3-4 days | Blocked by C1 + C3 |
| **C5** | Theoretical analysis (information-capacity bound formal proof, linking 4-qubit ceiling to classical coding theory) | Done | ~45min eng | **Completed 2026-04-19 (commits `41a3432`..`c370001`)** — see `docs/paper-a/c5-info-bound.tex`, bound=0.911 vs empirical 0.246 |

**Total: 18-29 days of focused work** (vs 6 months with PennyLane parameter-shift).

## Sequencing

```
     ┌────── C1 Baseline ──────┐
     │                          ├──► C4 Scale test (needs C1's classical baselines + C3's real data)
     └────── C3 Real corpus ────┘
                                  
C2 Downstream eval  (independent of C1/C3/C4, runs any time after torch-vqc merge — done)
C5 Theoretical     (fully independent, no code — can start T-0)
```

**Recommended order for solo execution:**
1. **C1 first** (2-3 days) — contextualises all numbers, unblocks rest of discussion
2. **C5 in parallel** (writing-heavy, can interleave) — theoretical framing strengthens C1's ceiling claim
3. **C2** (3-5 days) — shows VQC router is useful downstream, not just a classification exercise
4. **C3** (5-7 days) — required for reviewer credibility; synthetic data is the biggest soft spot
5. **C4 last** (3-4 days) — the "crown" showing 35-domain scale

## Acceptance criteria per sub-plan

Each must produce:
1. **Working code** in `src/` + `scripts/`, all tests green
2. **One results JSON** in `results/c<N>-<slug>.json` — machine-readable numbers
3. **One paper-facing doc** in `docs/paper-a/c<N>-<slug>.md` — narrative + figures
4. **Citation-ready bibtex** for any external work invoked

## Kill criteria

The project should **stop** (and retract Paper A entirely) if any of the following is encountered during C1–C5:

- **C1 kill**: classical LogReg on MiniLM raw embeddings hits >80% on 10-class routing. If the baseline is that strong, the VQC's ~40% ceiling is a story about inadequacy, not a publishable methodology.
- **C2 kill**: routing accuracy doesn't correlate with downstream LLM quality (i.e., "better" routes don't produce better answers). The router is worthless.
- **C3 kill**: real corpus shows <10 clearly-separable domains. The problem framing is synthetic.
- **C5 kill**: the 4-qubit ceiling is trivially derivable (like "obviously you can't classify 10 classes with 4 bits"). Nothing theoretical to publish.

If any kill criterion triggers, the contribution degrades to "tool release only" (`torch-vqc` repo already published as standalone — that stays).

## Current blockers & dependencies

- **Repo**: `genial-lab/micro-kiki` (main branch at `9e920ad`)
- **Standalone tool**: `electron-rare/torch-vqc` (v0.1.0 published 2026-04-19)
- **Compute**: Studio available for parallelism; M5 for iteration
- **Data for C3**: open question — real dialogue logs from where? Need to decide (Kiki/Yi sessions? mascarade? synthetic augmentation?)

## Handoff notes for future-you

- All 5 sub-plans are runnable serially by one engineer or in parallel if multiple. They only share data preparation (embedded samples), not code paths.
- C1 is the best first step because it's small, closed-form, and its output is directly used in C4 and in Paper A §4.
- **Do not start Paper A v2 writing until C1 is done.** Numbers from C1 populate the abstract.
