# Paper A Reframe Plan — Aeon-as-AMI-Memory

**Date**: 2026-04-19
**Status**: DRAFT — awaiting PoC A (Text-JEPA) results to finalize
**Supersedes**: `paper-outline-triple-hybrid.md` as the v0.3 research paper lead direction

---

## 1. Motivation for the reframe

Le plan initial (`paper-outline-triple-hybrid.md`) positionnait micro-kiki comme "First hybrid quantum-neuromorphic-classical routing for domain-expert LLM inference". Trois raisons de pivoter :

1. **Cadre théorique plus solide**. LeCun's "A Path Towards Autonomous Machine Intelligence" (arXiv:2206.15331, 2022) fournit un cadre modulaire (7 modules) où nos composants se rangent naturellement. La communauté AMI/JEPA (I-JEPA, V-JEPA 2, LeJEPA/SIGReg arXiv:2511.08544) cherche précisément des implémentations concrètes du Module 7.

2. **Les résultats empiriques orientent le framing**. Le PoC B v2 (`2026-04-17-aeon-predictor-poc-alpha.md`) a prouvé que centrage DinoV3 + rollback produit +22 % MRR sur flux structurés (condition D), mais que stack-conditioning est fragile sous centrage (0 % win_stack D, 1 % E). La contribution principale n'est donc pas "quantum router" mais **working memory JEPA-alignée avec garde-fous runtime**.

3. **Quantique et neuromorphique pas mûrs pour papier unique**. VQC reste simulateur, SNN LAS en cours, Akida non livré. Mieux vaut isoler la contribution défendable (Aeon-Module-7) et déferer VQC/SNN à un Paper B.

## 2. New thesis

**Aeon is a candidate implementation of the Short-Term Memory module (Module 7) in LeCun's Autonomous Machine Intelligence architecture, built on JEPA-aligned principles: it predicts latent successor states via a numpy MLP, shapes its predictions via runtime DinoV3-style centering (no EMA teacher, no stop-gradient), and detects representational collapse via a deterministic std-ratio tripwire that triggers weight rollback.** Nous démontrons que cette combinaison — centrage + rollback, déployés en production — donne +22 % MRR sur des flux structurés sans alourdir le coût runtime (< 1 MB poids, < 2 s training par 1000 turns sur M5).

**Ce que nous NE revendiquons PAS** : (a) implémentation AMI complète — Module 7 seul, pas de Configurator / Perception / World Model / Cost / Critic / Actor bouclés ; (b) world model génératif — transition latente, pas observation ou token ; (c) planification multi-étapes — horizon = 1 ; (d) validation quantique / neuromorphique — VQC et SNN mentionnés comme composants complémentaires hors-scope ; (e) stack-conditioning — ablation D montre l'effondrement sous centrage, disclosed comme future work.

## 3. AMI module mapping

| AMI module | micro-kiki component | Claim strength | Notes |
|------------|----------------------|----------------|-------|
| **1. Configurator** | VQC router (4 qubits, 72 params) | **Partial** | Existe, 86.8 % val_acc unbalanced, 53 % balanced — mentionné mais pas étudié dans ce papier |
| **2. Perception** | n/a | **None** | Pas d'environnement externe ; entrées sont directement des embeddings texte |
| **3. World Model** | Aeon LatentMLP (h_t → h_{t+1}) | **Partial** | Prédit transitions latentes 1-pas, pas dynamique complète du monde |
| **4. Cost** | CAMP judge (arXiv:2604.00085) | **Partial** | Évaluation post-hoc ; pas de feedback d'apprentissage bouclé |
| **5. Critic** | n/a | **None** | Pas de value function |
| **6. Actor** | LLM stack (Qwen3.5-35B-A3B + LoRAs) | **Delegated** | Exécution déléguée à la stack LLM ; hors scope Paper A |
| **7. Short-Term Memory** | **Aeon (Atlas + Trace + LatentMLP + centering + rollback)** | **STRONG** | Claim principal du papier ; empirical backing PoC B v2 |

Le papier se concentre sur la ligne 7. Les lignes 1, 3, 4 sont mentionnées dans la discussion comme points d'ancrage pour des papiers suivants.

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
- **5.1 Centering delivers** — condition D MRR 0.413 → 0.498 (+22 %), E stabilizes at 0.500.
- **5.2 Centering hurts on random-walk** — A–B MRR 0.263 → 0.228 ; bounds claim to saturation regime.
- **5.3 Rollback activation** — <TBD — awaiting telemetry from PoC B long-run>
- **5.4 Stack-conditioning ablation (honest failure)** — 23 % (A) → 0 % (D) → 1 % (E). Disclosed.
- **5.5 Cross-session persistence** — AeonSleep: 36 recalls / 14 turns vs 0 for raw LLM.

### §6 Discussion
Centering+rollback as Module 7 primitive ; centering↔stack interference hypothesis (per-stack µ/σ or learned stack adapter) ; saturation ceiling ; limitations (synthetic, horizon=1, no closed loop) ; roadmap to full AMI via VQC (Configurator) + LLM stack (Actor).

### §7 Conclusion
Summary of strong claim ; partial-AMI disclaimer ; code + weights release (Apache 2.0) ; future work (real conversations via PoC A Text-JEPA, per-stack centering, multi-step horizon).

### Appendices
A. Test coverage (33 tests) ; B. Compute budget (M5, ~2 s / 1000 turns, no GPU) ; C. Hyperparameters + seeds.

## 5. Empirical scorecard

**What PoC B v2 actually proved (strong claims, empirically backed)** :

| Finding | Evidence | Strength |
|---------|----------|----------|
| Centering delivers +22 % MRR on structured streams | Condition D vs baseline, MRR 0.413 → 0.498 | **Strong** |
| Rollback on std-collapse works deterministically | Unit test `test_collapse_detector_triggers` | **Strong** |
| 100K-param numpy deployment feasible | Code size + < 1 MB weights, runtime measured on M5 | **Strong** |
| Cross-session memory via AeonSleep | AeonSleep existing design, 36 recalls / 14 turns | **Strong** |
| Cold-start fallback is graceful | `predict_next()` returns h_t when not ready | **Strong** |

**Weak / disclosed findings** :

| Finding | Evidence | Treatment |
|---------|----------|-----------|
| Stack-conditioning fragile under centering | A 23 % → D 0 % → E 1 % win_stack | **Disclosed**, framed as future work (per-stack centering, stack adapter) |
| Centering harms on random-walk (non-saturated retrieval) | A–B MRR 0.263 → 0.228 | **Disclosed**, bounds claim to saturation regime |
| Synthetic streams only | All experiments on random-walk + stack-structured | **Disclosed**, commits to real-data follow-up |

**What's still needed before submission** :

- VQC Configurator smoke-test (pipe router output → Aeon, no regression). <TBD — awaiting PoC A Task 14>
- LeJEPA baseline if code releases. <TBD — arXiv:2511.08544 code status>
- Serving-load latency (> 100 concurrent queries). <TBD>
- Centering on/off ablation at serving time. <TBD — eval script needed>
- At least one real-data dataset (PoC A Text-JEPA turns). <TBD — awaiting PoC A results this week>

## 6. Reviewer anticipation

1. **"Not a real AMI implementation — LeCun has 7 modules, you touch one."** We never claim full AMI. Title + abstract scope to "candidate Module 7 implementation". §6.5 enumerates missing pieces. Framing is "building block", not "system".

2. **"Not a proper JEPA predictor (no masking, no teacher network)."** We claim methodological convergence on three principles: no EMA, no stop-gradient hacks, prediction in latent space. We do not reproduce I-JEPA / V-JEPA 2 architecturally. Centering is our specific mechanism, philosophically kin to SIGReg (arXiv:2511.08544).

3. **"Stack-conditioning was your PoC novelty and it failed."** Reframe acknowledges this head-on: centering + rollback **is** the novelty now. Stack-conditioning is future work. Honest A–E ablation table is a feature, not a bug.

4. **"All experiments synthetic."** Disclosed in §6.4. Committed to Paper A' follow-up on PoC A Text-JEPA real data. Centering has no synthetic-specific assumption (pure distribution matching) ; rollback is data-agnostic (safety mechanism).

5. **"Why AMI-class without a world model or actor loop?"** Module 7 is the working-memory contribution. LeCun 2022 §3.6 describes Module 7 as standalone-describable. We scope to "Module 7 substrate", not "AMI system".

## 7. Venue targeting

**Primary** : NeurIPS 2026 Workshop on **World Models & Cognitive Architectures** (historique pour I-JEPA, V-JEPA, DreamerV3). Call expected May-June 2026, deadline typically July-Sept.

**Secondary** : ICML 2026 Workshop on **Cognitive Architectures for Language Agents** or **Memory in Foundation Models** (tracks à surveiller, call expected February-March 2026).

**Tertiary** : ICLR 2027 main track (if we can bolt on PoC A Text-JEPA real-data results + at least one more experiment — would push to main track rather than workshop).

**Strategy** : submit workshop first for peer review + feedback, iterate, extend to journal (TMLR) or ICLR main.

## 8. What to cut from the original paper

Reference: `paper-outline-triple-hybrid.md`, 359 lines.

**Cut or drastically reduce** (move to Paper B or SpikingKiki paper) :
- Quantum VQC deep-dive (§3.2, §5.1, §6.1, §7.1) — keep a 1-paragraph mention in §6.5 discussion as "candidate Configurator for future integration".
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

1. **Keep or drop the quantum framing entirely in Paper A?** Current plan: drop it to §6.5 discussion paragraph. Alternative: one-sentence mention in abstract as "future integration target". **Recommendation: drop to discussion**.

2. **Single paper (Aeon-as-Module-7) or split (A1 Aeon + A2 VQC Configurator)?** Current plan: single paper, VQC spins off later. Alternative: write A1 (Aeon) now, plan A2 for NeurIPS 2027. **Recommendation: single paper A, defer A2**.

3. **Cite PoC A (Text-JEPA) even though it's a different experiment / repo (`micro-kiki-poc-textjepa` vs `micro-kiki-poc-aeon`)?** Current plan: cite if results are available before submission as "orthogonal validation on real conversational embeddings". **Recommendation: cite if available, don't block on it**.

4. **Include a theoretical section on "JEPA loss as working-memory regularizer"?** Current plan: no — keep paper empirical. Alternative: 1-page theoretical section positioning centering as a projection operator analogous to SIGReg's Cramér-Wold projections. **Recommendation: save theory for a companion short paper or tech report**.

5. **Workshop track or main track?** Current plan: workshop (NeurIPS 2026 World Models). Alternative: skip workshop, go main ICLR 2027. **Recommendation: workshop first — faster turnaround, valuable reviewer feedback, still cite-able**.

---

**Document metadata**  
Author: micro-kiki research team  
Review status: awaiting author sign-off on §10 open decisions  
Next step: once §10 is resolved, begin drafting §1 of the actual paper
