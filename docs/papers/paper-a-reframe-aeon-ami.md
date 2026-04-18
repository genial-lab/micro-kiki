# Paper A Reframe Plan — Aeon-as-AMI-Memory

**Date**: 2026-04-19
**Status**: DRAFT — PoC A (Text-JEPA) results integrated; LayerNorm(delta) case study backports confirmed
**Supersedes**: `paper-outline-triple-hybrid.md` as the v0.3 research paper lead direction

---

## 1. Motivation for the reframe

Le plan initial (`paper-outline-triple-hybrid.md`) positionnait micro-kiki comme "First hybrid quantum-neuromorphic-classical routing for domain-expert LLM inference". Trois raisons de pivoter :

1. **Cadre théorique plus solide**. LeCun's "A Path Towards Autonomous Machine Intelligence" (arXiv:2206.15331, 2022) fournit un cadre modulaire (7 modules) où nos composants se rangent naturellement. La communauté AMI/JEPA (I-JEPA, V-JEPA 2, LeJEPA/SIGReg arXiv:2511.08544) cherche précisément des implémentations concrètes du Module 7.

2. **Les résultats empiriques orientent le framing**. Le PoC B v2 (`2026-04-17-aeon-predictor-poc-alpha.md`) a prouvé que centrage DinoV3 + rollback produit +22 % MRR sur flux structurés (condition D), mais que stack-conditioning est fragile sous centrage (0 % win_stack D, 1 % E). La contribution principale n'est donc pas "quantum router" mais **working memory JEPA-alignée avec garde-fous runtime**.

3. **Quantique et neuromorphique pas mûrs pour papier unique**. VQC reste simulateur, SNN LAS en cours, Akida non livré. Mieux vaut isoler la contribution défendable (Aeon-Module-7) et déferer VQC/SNN à un Paper B.

## 2. New thesis

**Aeon is a candidate implementation of the Short-Term Memory module (Module 7) in LeCun's Autonomous Machine Intelligence architecture, built on JEPA-aligned principles: it predicts latent successor states via a numpy MLP, applies runtime anti-collapse mechanisms (DinoV3-style centering as the default; LayerNorm(delta) as the stack-preserving alternative), and detects representational collapse via a deterministic std-ratio tripwire that triggers weight rollback.** We demonstrate three aligned findings: (1) centering delivers +22 % MRR on structured streams without adding runtime cost (< 1 MB weights, < 2 s training per 1000 turns on M5); (2) LayerNorm(delta)-based anti-collapse preserves stack-conditioning signal (59% win_stack at 300 epochs, 0.447 predictive_mrr vs 0.090 null_mrr, condition L2); (3) Text-JEPA compression of the Configurator embedding achieves 3× compression (384→128 dims) while retaining 97% VQC routing accuracy (0.925 → 0.900). Together, these mechanisms form the AMI Module 7 substrate: working memory with anti-collapse robustness, stack-conditioning compatibility, and signal-compressible routing.

**Ce que nous NE revendiquons PAS** : (a) implémentation AMI complète — Module 7 seul, pas de Perception / World Model / Cost / Critic / Actor bouclés en feedback ; (b) world model génératif — transition latente, pas observation ou token ; (c) planification multi-étapes — horizon = 1 ; (d) validation complète quantique / neuromorphique — VQC + Text-JEPA Configurator mentionnés comme composants intégrés hors-scope ici, spin-off à Paper B ; (e) comparaison en production contre competitors — tous les expériments sur données synthétiques et conversationnelles, disclosed.

## 3. AMI module mapping

| AMI module | micro-kiki component | Claim strength | Notes |
|------------|----------------------|----------------|-------|
| **1. Configurator** | VQC router (6 qubits, 35 classes, ~180 params) + Text-JEPA compression | **Strong** | VQC: 86.8 % val_acc unbalanced, 53 % balanced. Text-JEPA: 3× compression (384→128 dims) retains 97% routing accuracy (0.925 → 0.900 on 10-domain classification). This is the Configurator path of AMI Module 1 in full pipeline. |
| **2. Perception** | n/a | **None** | Pas d'environnement externe ; entrées sont directement des embeddings texte (MiniLM-L6) |
| **3. World Model** | Aeon LatentMLP (h_t → h_{t+1}) | **Partial** | Prédit transitions latentes 1-pas, pas dynamique complète du monde |
| **4. Cost** | CAMP judge (arXiv:2604.00085) | **Partial** | Évaluation post-hoc ; pas de feedback d'apprentissage bouclé |
| **5. Critic** | n/a | **None** | Pas de value function |
| **6. Actor** | LLM stack (Qwen3.5-35B-A3B + LoRAs) | **Delegated** | Exécution déléguée à la stack LLM ; hors scope Paper A |
| **7. Short-Term Memory** | **Aeon (Atlas + Trace + LatentMLP + anti-collapse + rollback)** | **STRONG** | Claim principal du papier ; empirical backing PoC B v2 (centering + LayerNorm(delta)) + PoC A Text-JEPA (Configurator compression) |

Le papier se concentre sur les lignes 1 (compressed) et 7 (working memory). Les lignes 3, 4, 6 sont mentionnées dans la discussion comme points d'ancrage pour des papiers suivants.

## 4. Section-by-section outline

### §1 Introduction
LeCun's AMI framework and the open question of concrete Module 7 implementations for text/dialogue. JEPA successes in vision vs. the gap for symbolic state. Thesis statement (§2). *Draws from*: PoC B α §1–§2, arXiv:2206.15331, arXiv:2511.08544.

### §2 Related Work
JEPA family (I-JEPA, V-JEPA 2, LeJEPA/SIGReg) ; generative world models contrast (DreamerV3 arXiv:2301.04104, TD-MPC2 arXiv:2410.16662) ; DINO self-distillation (DINO/v2/v3, arXiv:2104.14294 / 2304.07193 / 2508.10104) ; memory-augmented LMs (MemGPT arXiv:2310.08560, Larimar arXiv:2403.11901, RETRO arXiv:2112.04426). *Draws from*: `related-work-aeon-predictor.md` (104 lines, directly reusable).

### §3 Aeon Architecture
Substrate (Atlas SIMD + Trace NetworkX) ; LatentMLP 384→256→384 numpy cosine loss (< 1 MB) ; DinoV3-style centering (stateless, no EMA) ; collapse detector + weight rollback (runtime safety) ; cold-start fallback (identity below 500 pairs). *Draws from*: `src/memory/aeon_predictor.py`, PoC B α §2.

### §4 Experimental protocol
Synthetic streams (random-walk + stack-structured), 1000 turns, 100 held-out queries, 5 ablations A–E. Metrics: Recall@5, MRR, win_pred %, win_stack %, final_loss. Full table from PoC B α §3.

### §5 Results
- **5.1 Centering delivers anti-collapse** — condition D MRR 0.413 → 0.498 (+22 %), E stabilizes at 0.500. Tested on structured streams (saturation regime).
- **5.2 LayerNorm(delta) restores stack signal** — condition L2 at 300 epochs: win_stack = 59%, predictive_mrr 0.447 vs null_mrr 0.090. Per-sample normalization of residual preserves stack-specific offsets that running-mean centering destroys.
- **5.3 Text-JEPA validates Configurator compression** — 3× compression (384→128) retains 97% VQC routing accuracy (0.925 → 0.900 on 10-domain classification). Real conversational embeddings.
- **5.4 Centering harms on random-walk** — A–B MRR 0.263 → 0.228 ; bounds claim to saturation regime. Disclosed.
- **5.5 Rollback activation** — Unit test + deterministic collapse detection; telemetry from long-run PoC B v2.
- **5.6 Cross-session persistence** — AeonSleep: 36 recalls / 14 turns vs 0 for raw LLM.

### §6 Discussion
Centering+rollback as Module 7 primitive ; centering↔stack interference hypothesis (per-stack µ/σ or learned stack adapter) ; saturation ceiling ; limitations (synthetic, horizon=1, no closed loop) ; roadmap to full AMI via VQC (Configurator) + LLM stack (Actor).

### §7 Conclusion
Summary of strong claim ; partial-AMI disclaimer ; code + weights release (Apache 2.0) ; future work (real conversations via PoC A Text-JEPA, per-stack centering, multi-step horizon).

### Appendices
A. Test coverage (33 tests) ; B. Compute budget (M5, ~2 s / 1000 turns, no GPU) ; C. Hyperparameters + seeds.

## 4. Empirical scorecard

**What PoC B v2 and PoC A Text-JEPA actually proved (strong claims, empirically backed)** :

| Finding | Evidence | Strength |
|---------|----------|----------|
| Centering delivers +22 % MRR on structured streams | Condition D vs baseline, MRR 0.413 → 0.498 | **Strong** |
| LayerNorm(delta) stack preservation | Condition L2 at 300 epochs: win_stack = 59%, predictive_mrr 0.447 vs null_mrr 0.090. Per-sample normalization of residual delta preserves stack-specific offsets. | **Strong** |
| Text-JEPA compression validates Configurator path | 3× embedding compression (384→128 dims) retains 97% VQC routing accuracy (baseline 0.925, Text-JEPA 0.900 on 10-domain classification). | **Strong** |
| Rollback on std-collapse works deterministically | Unit test `test_collapse_detector_triggers` | **Strong** |
| 100K-param numpy deployment feasible | Code size + < 1 MB weights, runtime measured on M5 | **Strong** |
| Cross-session memory via AeonSleep | AeonSleep existing design, 36 recalls / 14 turns | **Strong** |
| Cold-start fallback is graceful | `predict_next()` returns h_t when not ready | **Strong** |

**Weak / disclosed findings** :

| Finding | Evidence | Treatment |
|---------|----------|-----------|
| Centering destroys stack signal, LayerNorm(delta) restores it | A 23 % → D 0 % under centering; L2 59 % under LayerNorm(delta) | **Disclosed**, framed as two compatible anti-collapse strategies with different stack properties |
| Centering harms on random-walk (non-saturated retrieval) | A–B MRR 0.263 → 0.228 | **Disclosed**, bounds claim to saturation regime |
| Synthetic streams only | All experiments on random-walk + stack-structured | **Disclosed**, commits to real-data follow-up |

**What's still needed before submission** :

- LeJEPA baseline if code releases. <TBD — arXiv:2511.08544 code status>
- Serving-load latency (> 100 concurrent queries). <TBD>
- Centering on/off ablation at serving time. <TBD — eval script needed>
- Benchmark against LeJEPA baseline (if published). <TBD — comparative eval needed>

## 5. Stack-conditioning status update

**Stack-conditioning: validated via LayerNorm(delta).** Centering-based anti-collapse destroys the stack signal (0–1% win_stack across conditions D, E, F in PoC B v2). Replacing centering with per-sample LayerNorm of the residual delta restores stack signal to 59% win_stack at 300 epochs (condition L2, PoC B case study). **This is not a single-mechanism paper but a COMPATIBILITY STUDY of anti-collapse choices.** We present both: centering as the simpler, production-robust default; LayerNorm(delta) as the stack-preserving alternative. Real-deployment trade-off depends on the downstream task.

## 6. Reviewer anticipation

1. **"Not a real AMI implementation — LeCun has 7 modules, you touch one."** We never claim full AMI. Title + abstract scope to "candidate Module 7 implementation". §6.5 enumerates missing pieces. Framing is "building block", not "system".

2. **"Not a proper JEPA predictor (no masking, no teacher network)."** We claim methodological convergence on three principles: no EMA, no stop-gradient hacks, prediction in latent space. We do not reproduce I-JEPA / V-JEPA 2 architecturally. Centering is our specific mechanism, philosophically kin to SIGReg (arXiv:2511.08544).

3. **"Stack-conditioning was your PoC novelty and it failed."** Stack-conditioning works under LayerNorm(delta) anti-collapse but fails under DinoV3-style centering. The paper presents BOTH findings: centering as the simpler, robust default; LayerNorm(delta) as the stack-preserving alternative. This is not a single-mechanism paper but a **COMPATIBILITY STUDY of anti-collapse choices** — see §5 for full details.

4. **"All experiments synthetic."** Disclosed in §6.4. Committed to Paper A' follow-up on PoC A Text-JEPA real data (Text-JEPA compression validated on real conversational turns; centering has no synthetic-specific assumption). Rollback is data-agnostic (safety mechanism).

5. **"Why AMI-class without a world model or actor loop?"** Module 7 is the working-memory contribution. LeCun 2022 §3.6 describes Module 7 as standalone-describable. We scope to "Module 7 substrate", not "AMI system".

## 7. Venue targeting

**Primary** : NeurIPS 2026 Workshop on **World Models & Cognitive Architectures** (historique pour I-JEPA, V-JEPA, DreamerV3). Call expected May-June 2026, deadline typically July-Sept.

**Secondary (now plausible)** : ICLR 2027 workshop track on **Cognitive Architectures** or **Memory in LLMs** (with LayerNorm(delta) + Text-JEPA wins, the empirical portfolio strengthens for ICLR). Call expected September 2026, deadline typically November 2026.

**Tertiary** : ICML 2026 Workshop on **Cognitive Architectures for Language Agents** (call expected February-March, already passed; could submit late or pivot to post-acceptance iterate).

**Strategy** : submit NeurIPS 2026 workshop first for peer review + feedback (faster turnaround), simultaneously prepare ICLR 2027 submission (more pages, more experiments). Main-track ICLR possible if we secure one real-deployment story or additional baseline comparison by October 2026.

## 8. What to cut from the original paper

Reference: `paper-outline-triple-hybrid.md`, 359 lines.

**Cut or drastically reduce** (move to Paper B or SpikingKiki paper) :
- Quantum VQC deep-dive mechanics (§3.2 old, §5.1 old) — KEEP brief Text-JEPA integration story in §3 (1 paragraph on VQC router + compression path). Move SQA training / gate-optimization to Paper B.
- SNN LAS conversion details (§3.3, §5.3, §7.2) — move entirely to `spikingkiki-v3-final.md` paper.
- 32-domain LoRA training curriculum (§3.4, §5.2, §7.3) — keep only Qwen base identity mention ; full discussion goes to micro-kiki systems paper.
- End-to-end multi-turn cognitive pipeline latency breakdown (§5.4, §6.4) — trim to a half-page section focused on memory-specific latency.
- Negotiator CAMP arbitration (§5.4, §6.5) — keep 1 paragraph ; not the focus.

**Expand substantially** :
- Aeon architecture (§3.5 of old → entire §3 in new, 4-5 pages).
- DinoV3-style centering philosophy (new subsection).
- AMI Module 7 positioning (new §1 + §6.1).
- Stack-conditioning ablation full disclosure (§5.4 new).
- Runtime safety via rollback (new subsection §3.4).

Net effect: Paper A becomes ~12–14 pages focused on Aeon as Module 7 substrate. Paper B (VQC + SNN systems work) is spun off separately.

## 9. Writing timeline

Assuming PoC A Text-JEPA results land this week (Task 14 per project memory) :

- **Week 1 (2026-04-20 → 04-26)** : finalize this reframe, secure PoC A numbers, draft §1 + §2 + §3.
- **Week 2 (04-27 → 05-03)** : draft §4 + §5 + §6 + §7. Import figures from PoC B α eval script.
- **Week 3 (05-04 → 05-10)** : internal review (coauthor or careful self-review), address reviewer-anticipation objections preemptively, tighten prose.
- **Week 4 (05-11 → 05-17)** : final polish, appendices, reproducibility checklist, submit to chosen workshop or upload to arXiv.

**Realistic horizon** : 3–4 weeks for a workshop submission. Main-track ICLR would require +6–8 weeks of additional experiments.

## 10. Open decisions for author

Five decisions needed before drafting begins in earnest :

1. **Keep or drop the quantum framing entirely in Paper A?** CLOSED: **KEEP quantum as Configurator module.** Text-JEPA validates the VQC+Text-JEPA combo as the AMI Configurator (§3). Brief integration story (1 para in §3), full VQC SQA details → Paper B.

2. **Single paper (Aeon-as-Module-7) or split (A1 Aeon + A2 VQC Configurator)?** Current plan: single paper, VQC+Text-JEPA integrated in §3 as strong Configurator claim. Alternative: write A1 (Aeon+Centering+LayerNorm(delta)) now, defer A2 for VQC systems. **Recommendation: single paper A, §3 covers Configurator+compression, Paper B handles quantum details**.

3. **Cite PoC A (Text-JEPA) results prominently?** Current plan: YES, cite in §3 (Configurator) and §4 (scorecard). Text-JEPA is now a co-claimed validation, not optional. **Recommendation: cite as part of §3.1 Configurator section and Table in §4**.

4. **Include a theoretical section on "JEPA loss as working-memory regularizer"?** Current plan: no — keep paper empirical. Alternative: 1-page theoretical section positioning centering as a projection operator analogous to SIGReg's Cramér-Wold projections. **Recommendation: save theory for a companion short paper or tech report**.

5. **Workshop track or main track?** Current plan: workshop first (NeurIPS 2026 World Models, May-Sept). Alternative: skip workshop, go main ICLR 2027 (September-November, with extra experiments). **Recommendation: workshop first — faster turnaround, valuable reviewer feedback, pipeline to ICLR main-track with LayerNorm(delta)+Text-JEPA lifts**.

---

**Document metadata**  
Author: micro-kiki research team  
Review status: awaiting author sign-off on §10 open decisions  
Next step: once §10 is resolved, begin drafting §1 of the actual paper
