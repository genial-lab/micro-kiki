# Session Index — 2026-04-17 → 2026-04-19

Navigation master des livrables de la session. Organisé par type d'artefact, pas par chronologie.

## 1. Papers (publication-candidate)

### 1.1 Paper A — Aeon-as-AMI-Memory (workshop draft v1)

- **EN**: [`docs/papers/paper-a-draft-v1.md`](papers/paper-a-draft-v1.md) — 6205 mots, 9 sections + 2 appendices
- **FR**: [`docs/papers/paper-a-draft-v1-fr.md`](papers/paper-a-draft-v1-fr.md) — synced
- **Reframe plan**: [`docs/papers/paper-a-reframe-aeon-ami.md`](papers/paper-a-reframe-aeon-ami.md)
- **Related work**: [`docs/papers/related-work-aeon-predictor.md`](papers/related-work-aeon-predictor.md)
- **PDFs** (Typst + LaTeX × EN + FR = 4 fichiers): `docs/papers/pdf/paper-a-draft-v1*.pdf`
- **Status**: workshop-grade, pas main track. Cf. `paper-a-reframe-aeon-ami.md` §10 pour decisions ouvertes.

### 1.2 Case study companion

- **EN**: [`docs/papers/stack-conditioning-case-study.md`](papers/stack-conditioning-case-study.md) — 3387 mots
- **FR**: [`docs/papers/stack-conditioning-case-study-fr.md`](papers/stack-conditioning-case-study-fr.md)
- **Claim**: centering vs LayerNorm(δ) compatibility under one-hot conditioning. 2/4 fixes testés.

### 1.3 Paper B — SpikingKiki v3 (neuromorphic substrate)

- **EN**: [`docs/papers/spikingkiki-v3-final.md`](papers/spikingkiki-v3-final.md) — section 10 JEPA discussion ajoutée
- **FR**: [`docs/papers/spikingkiki-v3-final-fr.md`](papers/spikingkiki-v3-final-fr.md)
- **Target**: ICONS / Frontiers in Neuroscience

## 2. Research notes

- [`docs/research/vqc-cem-acceleration-vjepa2.md`](research/vqc-cem-acceleration-vjepa2.md) — verdict: no near-term quantum accel
- [`docs/research/vqc-conditioned-aeon-predictor.md`](research/vqc-conditioned-aeon-predictor.md) — D direction design (next sprint)
- [`docs/research/vqc-class-count-reconciliation.md`](research/vqc-class-count-reconciliation.md) — 6q/35-class consolidation
- [`docs/research/edge-deployment-genio-ventunoq.md`](research/edge-deployment-genio-ventunoq.md) — GenioBoard + VENTUNO Q scouting
- [`docs/research/devis-edge-deployment-2026-04-19.md`](research/devis-edge-deployment-2026-04-19.md) — devis détaillé 2 scénarios (4.6 k€ / 11.7 k€)

## 3. Plans (roadmap main-track)

- [`docs/superpowers/plans/2026-04-19-baseline-comparison.md`](superpowers/plans/2026-04-19-baseline-comparison.md) — 1935L, 13 tasks (MemGPT/Larimar/RETRO/HippoRAG)
- [`docs/superpowers/plans/2026-04-19-downstream-llm-eval.md`](superpowers/plans/2026-04-19-downstream-llm-eval.md) — 2078L, 20 tasks (judge-scored delta)
- [`docs/superpowers/plans/2026-04-19-real-dialogue-corpus.md`](superpowers/plans/2026-04-19-real-dialogue-corpus.md) — 1880L, 17 tasks (LMSYS-Chat-1M)
- [`docs/superpowers/plans/2026-04-19-scale-test.md`](superpowers/plans/2026-04-19-scale-test.md) — 1577L, 15 tasks (100k-1M scale)
- [`docs/superpowers/plans/2026-04-19-theoretical-analysis.md`](superpowers/plans/2026-04-19-theoretical-analysis.md) — 1132L, 14 tasks (Theorem A/B formal)

## 4. Code + tests (src/, scripts/, tests/)

### 4.1 Aeon predictor (src/memory/aeon_predictor.py)

- `PredictorConfig` + `LatentMLP` (numpy 2-layer + skip + LayerNorm(δ) option + per-stack centering option)
- `AeonPredictor` facade (ingest_latent, predict_next, fit_on_buffer, recall, buffer_size, ready)
- `detect_collapse` helper
- Tests: `tests/memory/test_aeon_predictor.py` (30+ tests, all passing)

### 4.2 Eval scripts

- `scripts/eval_aeon_predictor.py` — synthetic streams (random-walk + stack-structured) + 3-way ablation + CLI
- `scripts/eval_aeon_realdata.py` — real 10-domain corpus + exact/soft-domain metrics + 2 stream modes
- `scripts/eval_aeon_next_domain.py` — next-domain classification eval
- `scripts/eval_text_jepa_vqc.py` — Text-JEPA VQC routing benchmark
- `scripts/train_text_jepa.py` — Text-JEPA training CLI
- `scripts/ablate_text_jepa_dim.py` — latent-dim ablation

### 4.3 PDF build pipelines

- `docs/papers/build-pdf.sh` + `template.typ` — Typst pipeline (moderne, auto-detect FR)
- `docs/papers/build-pdf-latex.sh` + `latex-header.tex` — LaTeX/xelatex pipeline (academic, auto-detect FR)

## 5. Results (JSON + MD)

### 5.1 PoC B synthetic v2 (5 conditions + F + L)

- `results/2026-04-17-aeon-poc-{A,B,C,D,E}-v2.json` — centering ablation
- `results/2026-04-19-aeon-poc-{F1,F2}-per-stack*.json` — per-stack centering (FAIL)
- `results/2026-04-19-aeon-poc-{L1,L2,L3}-layernorm*.json` — LayerNorm(δ) validation (L2 SUCCESS at 59%)
- `results/2026-04-19-aeon-baseline-{no-centering,shared-centering}.json`

### 5.2 Real-data Aeon (5 runs)

- `results/aeon-realdata.json` — exact match interleaved (negative)
- `results/aeon-realdata-within-topic.json` — exact match within-topic (negative)
- `results/aeon-realdata-soft-interleaved.json` — soft-domain interleaved (+20pts ✅)
- `results/aeon-realdata-soft-withintopic.json` — soft-domain within-topic (saturated)
- `results/aeon-next-domain.json` — next-domain classification (chance)

### 5.3 Text-JEPA (PoC A)

- `results/text-jepa-vqc.json` + `.md` — Task 14 full-budget (0.925 / 0.900)
- `results/text-jepa-ablation.json` + `.md` — Task 15 dim sweep (budget-constrained)
- Note: dim=64 full-budget (Task 15.5) en cours 2026-04-19

## 6. Metrics at a glance (final)

| Mécanisme | Metric | Value | Verdict |
|-----------|--------|-------|---------|
| Centering (synthetic) | MRR D | 0.413 → 0.498 (+22%) | Strong |
| LayerNorm(δ) (synthetic) | win_stack L2 | 59% (300 ep, lr=5e-3) | Strong |
| Text-JEPA compression | VQC acc | 0.900 @128d vs 0.925 @384d (97% retention, 3×) | Strong |
| Real-data soft-domain | recall@5 interleaved | 11% → 31% (+181%, 4× MRR) | Strong ✅ |
| Per-stack centering | win_stack | 0% | FAIL (documented) |
| Combined LN(δ) + centering | win_stack | 1% | Catastrophe (documented) |

## 7. Git log (session commits, reverse chrono)

```
f5ba8c4 docs(aeon): FR sync Appendice B edge
5ef1840 docs: devis edge deployment (2 scenarios)
53682d9 docs: edge deployment scouting (Genio/VENTUNO Q)
00b77a9 docs(aeon): FR sync §4.5 real-data section
89ac05a docs(aeon): Paper A §4.5 real-data topic-switch
8defd95 docs(text-jepa): Task 16 verdict + ablation notes
605152d merge: PoC A Task 15 latent-dim ablation
b1969e9 merge: PoC A Text-JEPA VQC router
d30ffb8 merge: LayerNorm(delta) + reconciliation doc
...
(see `git log --oneline -60` for full)
```

## 8. Background tasks (still running or pending)

- **Task 2 dim=64 full-budget VQC eval** (PID 64431, 45+ min CPU): would validate "6× compression" claim if SUCCESS. Monitor via `ps aux | grep dim64`. Result file expected at `micro-kiki-poc-textjepa/results/text-jepa-dim64-full-budget.json`.
- **Quota reset Apr 23 21:00 CEST** for subagents (if needed for future plan execution).

## 9. Next-session decisions

1. **Submit Paper A workshop** (NeurIPS 2026 world-models workshop primary, ICLR 2027 workshop secondary) after author review pass.
2. **Execute 1-2 of the 5 plans** (P0 = Plan 1 baseline + Plan 2 downstream = ~6-10 weeks → main track candidate).
3. **Edge deployment devis** — décision go/no-go sur Devis A ou B. Si go, commande DigiKey + Arduino Store.
4. **Task 2 dim=64** — monitor, update Paper A §4.4 + Appendix B quand numbers landent.
5. **FR-EN sync discipline** : automatiser via pré-commit hook qui check l'existence du pair EN/FR et signale la divergence.

## 10. Honest take (from earlier assessment)

- **Ce qui est solide** : code propre, 4 mécanismes validés, case study diagnostic propre.
- **Ce qui est thin** : baselines absents, downstream LLM-quality absent, théorie observationnelle, data en synthétique + Q&A isolés.
- **Workshop submit** : réaliste avec l'état actuel.
- **Main track ICLR/NeurIPS** : nécessite les 2 P0 + idéalement 1-2 autres plans sur 3-6 mois.
- **Production micro-kiki deployment** : immédiat — le code est prêt pour intégration Factory 4 Life / Kill_LIFE.
