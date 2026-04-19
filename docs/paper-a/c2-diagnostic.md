# C2 Diagnostic — Why does routing fail downstream?

**Context.** The Phase C2 downstream eval triggered the kill criterion (oracle − random = 0.29 < 0.30) and revealed a confidently-wrong pathology where VQC (routing_acc 0.17) produced lower mean score (2.65) than Random (3.19). This diagnostic analyses the existing 100 per-query records (`results/c2-downstream.json`) to locate the effect, without running any new LLM calls.

## Per-domain breakdown

Figure `c2-diagnostic-per-domain.pdf`: for each of the 10 domains, the oracle-minus-vqc and vqc-minus-random score gaps.

Key observations (from `results/c2-diagnostic.json`):

- **Maximum oracle-vs-vqc gap: `platformio` at Δ=2.200** — the domain where VQC hurts most.
- **Most harmful to downstream quality: `platformio` at gap_vqc_vs_random=−2.100** — VQC's presence subtracts 2.1 rubric points vs having no router at all.
- **Domains where VQC helps (`gap_vqc_vs_random > 0`): 2 of 10** — routing is net-positive only on a minority.
- **Domains where VQC is strictly harmful (`gap_vqc_vs_random < −0.5`): 6 of 10** — confidently-wrong is the majority outcome.

Interpretation: the pathology is **not** uniformly distributed. It concentrates on platformio (and other domains where VQC systematically misroutes to a nearby "attractor" class — see below). A minority of domains show net benefit from routing.

## Correctness stratification

Figure `c2-diagnostic-stratified.pdf`: mean scores of the three routers on (a) queries where VQC routed correctly, (b) queries where VQC routed wrong.

Measured buckets:

| Bucket | n | vqc mean | oracle mean | random mean |
|---|---|---|---|---|
| `vqc_correct` | 17 | 3.059 | 3.471 | 3.706 |
| `vqc_wrong` | 83 | 2.566 | 3.482 | 3.084 |

**Confidently-wrong pathology test (bucket B):** VQC mean 2.566 vs Random mean 3.084 over the **same 83 queries**. Random beats VQC by **0.518 rubric points** in the bucket where VQC routed incorrectly. This confirms the pathology at stratified level — it is NOT an aggregation artefact.

**Surprising**: in bucket A (VQC routed correctly, n=17), VQC mean 3.059 is still **below** Random mean 3.706 by 0.647 points on the **same queries**. This is either (a) a self-judging artefact where the judge prefers outputs less constrained by explicit expert-persona strings, or (b) a genuine finding that "You are an expert in X" prompts harm quality even when X is correct. Worth isolating in future work.

## Qualitative top-10 review

See `c2-diagnostic-top10.md` for the 10 queries with the largest oracle-minus-vqc score gap (all have gap ≥ 4). The hand-written "Patterns observed" section identifies three distinct failure modes:

1. **Explicit persona-refusal on far-domain misrouting (#2, #7)**: the LLM given "You are an expert in kicad-dsl" for a FreeCAD or DSP question **refuses explicitly** ("KiCad DSL is not for 3D printing") and scores 0. The sharpest pathology and the main driver of the aggregate.
2. **Mode collapse on incompatible persona (#5)**: asking a stm32-persona LLM for Python PySpice code produces degenerate repeated imports. Rare but catastrophic when it hits.
3. **Near-identical content, large score delta (#3, #4, #6)**: VQC and Oracle produce essentially the same answer substance yet VQC scores 0 while Oracle scores 5. This is **judge inconsistency** inflating the measured gap.

Routing-error taxonomy in top-10:
- 4 out of 10 misroutes go to `kicad-dsl` — it acts as a "dumping ground" in the VQC's learned projection.
- Adjacent-domain misroutes (platformio↔embedded↔stm32) tend to pattern #3 (soft degradation).
- Far-domain misroutes (freecad↔kicad-dsl, dsp↔kicad-dsl) trigger pattern #1 (explicit refusal).

## Implications for Paper A §5

1. **The confidently-wrong pathology is real** at stratified level (bucket B: VQC −0.52 vs Random). Paper A §5 "Discussion and Limitations" should report this explicitly as the mechanism behind the aggregate negative result, not hide it.

2. **The platformio concentration** (gap_oracle_vs_vqc=2.2, gap_vqc_vs_random=−2.1) merits a sentence: the VQC's failure mode is domain-specific, not uniform. On the 2/10 domains where routing helps, the framework is directionally correct; the 6/10 harmful domains drive the aggregate collapse.

3. **Judge inconsistency is a methodological concern**: pattern #3 suggests n=100 is underpowered vs judge noise. A replication with independent non-self judge (Claude/GPT-4o-mini via API, or the Studio 480B teacher if available) or with n≥500 would improve confidence. Noted as limitation.

4. **The sibling LoRA experiment is JUSTIFIED** by the diagnostic. The dominant failure mode (#1 persona-refusal on far-domain misroutes) is **prompt-specific**: weight-level LoRA adapters do not commit the LLM to an explicit persona string, so they are unlikely to trigger the same refusal pathology. The experiment's hypothesis — that real LoRA will close the oracle-random gap — is empirically motivated. Proceed with writing the sibling spec.

## Kill-criterion check for the diagnostic itself

- `max(gap_oracle_vs_vqc) = 2.200` — **≥ 0.5**, diagnostic is load-bearing.
- `range(gap_vqc_vs_random) ≈ [−2.1, +something] ≫ 0.3` — effect is real, not noise floor.

**Verdict:** diagnostic is load-bearing. The C2 negative result has an identified mechanism (persona-refusal), a concentrated domain signature (platformio), and a methodological caveat (judge inconsistency). Paper A §5 can report these concretely.
