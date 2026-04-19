# C2: Downstream LLM Evaluation

**Setup.** 100-query eval (10/domain × 10 domains from `data/corpus-real`), routed through three routers to Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf on kxkm-ai (RTX 4090, llama.cpp HTTP). Each answer scored 0-5 by the **same model** (self-judging — see Caveats) using the rubric in `src/routing/llm_judge.py`. `chat_template_kwargs.enable_thinking=False` forced (Qwen3/3.5 thinking mode would consume all max_tokens silently).

**Routers.**
- **Random**: picks an expert uniformly among 10 (control).
- **Torch VQC (ours)**: trained per C3 on 400 remaining samples (train_acc = 0.300 — limited by the 4-qubit ceiling established in C1/C5).
- **Oracle**: always picks the ground-truth expert (upper bound).

"Expert" here means the system prompt `"You are an expert in {domain}."` — a prompt-based pseudo-adapter, NOT actual LoRA weight swapping.

## Results (100 queries, single seed, self-judged)

| Router | mean score | routing acc | score when routed correct | score when routed wrong |
|---|---|---|---|---|
| **Random** | **3.190** | 0.070 | 3.857 | 3.140 |
| **Torch VQC** | **2.650** | 0.170 | 3.059 | 2.566 |
| **Oracle** | **3.480** | 1.000 | 3.480 | N/A |

See Figure `c2-downstream-figure.pdf`.

## Kill criterion check — TRIGGERED

The plan's kill clause: "if `oracle - random < 0.3`, routing is POINTLESS". Measured: `3.480 - 3.190 = 0.290`. **0.29 < 0.30 → TRIGGERED** (by 0.01 margin).

Honest conclusion: **per-domain prompt-based specialization does not meaningfully improve answer quality** on this setup. The adapter-swap premise, as tested here, collapses marginally.

## Unexpected observations

1. **Random beats VQC by 0.54 rubric points** (3.19 vs 2.65). This is NOT seen in routing accuracy (VQC 0.170 > Random 0.070 — VQC routes better than chance) but IS seen in quality. Interpretation: **confident-but-wrong routing is worse than uncommitted random routing**. When VQC prompts the LLM with a wrong domain expert, the LLM commits to that specialization and produces an off-topic answer; when Random prompts with a mostly-wrong expert, the LLM (with no strong signal) drifts toward a safer generic answer.

2. **Random's score-when-correct (3.86) exceeds Oracle's score-when-correct (3.48)**. The 7 times Random happens to pick the right expert, the LLM produces its best answers. This artefact is likely self-judging bias: the judge may prefer answers less constrained by explicit expert-persona directives.

3. **VQC's score-when-wrong (2.57) is lowest** of all. Confirms point 1 — wrong routing with the VQC is the worst condition in the entire experiment.

## Caveats (matter for interpretation)

1. **Self-judging**: Qwen3.5-35B-A3B judges its own outputs. The rubric-score absolute calibration is questionable. Only RELATIVE orderings between routers should be trusted.

2. **Prompt-based pseudo-adapters**: Real per-domain LoRA adapters were NOT used (would require retraining 10 adapters out-of-session). System-prompt directives are a much weaker specialization signal than weight-level LoRA. A full adapter-swap experiment could produce larger oracle-random gaps.

3. **Small sample (n=100)**: 95% Wilson CI for the mean difference would be roughly ±0.4 pts — meaning the 0.29 gap is not statistically distinguishable from 0 at this sample size. More samples are needed for confident inference.

4. **VQC train_acc=0.300**: the VQC itself is a weak classifier on this 10-class task (consistent with C1/C3 findings). A stronger router would produce different dynamics.

## Implications for Paper A

This is a **negative result** that Paper A must report honestly. The three positive contributions remain:

1. **Methodological (C1-tested)**: `torch-vqc` enables rigorous quantum-ML benchmarking at 3000× speedup.
2. **Architectural (C1-tested)**: learned projection is necessary; raw-dim truncation is information-empty.
3. **Theoretical (C5-proven)**: Holevo+Fano bound quantifies the 4-qubit VQC ceiling (0.911 at MI=2.04 bits).

What C2 adds:
- **Downstream integration test** shows prompt-based routing does NOT pass the kill criterion on this setup.
- **Confidence-wrong pathology**: a weak-but-confident router can be actively harmful downstream. This is a novel observation worth highlighting.

Paper A §5 should include C2 as a limitation: "our VQC router, even when trained, does not yield downstream quality improvements via prompt-based specialization. Testing against actual LoRA adapters and larger eval sets is future work."

Alternative Paper A framing: position C2 as demonstrating that **quality of routing matters more than presence of routing** — confidently wrong is worse than uncommitted random, establishing a non-obvious specificity bound on when routing is useful.
