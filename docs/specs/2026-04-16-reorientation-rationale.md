# Reorientation Rationale: 32 Stacks → 10 Niche Domain Stacks

Date: 2026-04-16  
Status: Approved  
Decision: Reduce LoRA training scope from 32 domains to 10 niche hardware/EDA domains; drop all general-purpose and cross-domain stacks.

---

## Summary

Training chat-fr (stack-01) on Qwen3.5-35B-A3B produced a validation loss curve that makes the core assumption of this project untenable for 22 of the 32 planned domains: the base model already knows them. Forcing LoRA on well-represented domains causes overfitting within 300 iterations and wastes ~22 × 45 min = 16.5 h of Mac Studio training time on zero-gain adapters. The correct response is to reduce scope to the 10 domains where the base model demonstrably lacks coverage, enrich those datasets from HuggingFace mascarade collections, and simplify the router from 32 outputs to 11 (10 domain + 1 passthrough).

---

## 1. Evidence

### 1.1 Chat-fr training run (stack-01)

| Iteration | Val Loss | Train Loss | Observation |
|-----------|----------|------------|-------------|
| 0 | 1.695 | — | Starting point after warmup |
| 100 | 1.212 | 0.841 | Normal descent |
| 200 | 0.987 | 0.751 | Continued improvement |
| 300 | **0.722** | 0.633 | Best checkpoint |
| 400 | 0.789 | 0.633 | Val rising, train flat — overfitting onset |
| 500 | 0.856 | 0.633 | Confirmed overfitting |

Train loss frozen at 0.633 from iter ~280 onward while val loss climbs 18% from best. This is textbook memorization: the adapter has exhausted the generalization signal in the training split and is copying token sequences.

Root cause: the 35B base **already handles French conversation**. The 2,700-example fine-tuning set adds no new capability — it only introduces distribution shift that degrades the base model's broader generalisation.

### 1.2 Base model capability profile

Qwen3.5-35B-A3B out of the box:

| Capability | Coverage |
|-----------|---------|
| Languages | 201 (including French at near-native level) |
| Reasoning / thinking mode | Native — chain-of-thought with `<think>` tokens |
| Code (Python, TypeScript, C++, Rust, SQL, Shell, Lua, YAML) | Top-tier on EvalPlus, HumanEval, LiveCodeBench |
| Math | Competitive with specialized models on MATH-500 |
| Docker, web frontend/backend, DevOps | Well covered in pretraining corpus |
| Music theory, general audio | Adequate |
| LLM orchestration concepts | Strong, given model lineage |

The 35B was trained on a multi-trillion token corpus with heavy emphasis on code, multilingual text, and reasoning chains. Stacking a LoRA on top of these capabilities can only subtract performance (via forgetting or distribution shift) rather than add it.

### 1.3 Parameter/data ratio analysis

The 35B-A3B LoRA adapter at rank 16 exposes approximately 6.881 M trainable parameters per stack (attention projections only: q, k, v, o across all layers).

| Metric | Value |
|--------|-------|
| Trainable params per stack | ~6,881,000 |
| Training examples (chat-fr) | 2,700 |
| Params per example | **2,548** |
| Safe upper bound (rule of thumb) | ~100 params/example |
| Overshoot factor | **~25×** |

At 25× above the safe ratio, any domain with fewer than ~69,000 training examples will overfit a rank-16 adapter on the 35B base. Most of the 32 planned domains sit between 200 and 8,000 examples. Only the niche hardware domains, where we can aggregate multiple complementary sources, approach the required density.

### 1.4 Why niche hardware domains are different

The 10 retained domains share three properties that make LoRA beneficial:

1. **Underrepresented in pretraining**: KiCad DSL, SPICE netlist syntax, EMC compliance tables, STM32 HAL idioms, FreeCAD Python API — these are present in the pretraining corpus but at very low density. The base model hallucinates on component footprint names, net labels, and register maps.
2. **Structured and verifiable output**: A KiCad s-expression netlist is either syntactically valid or not. A SPICE simulation that references a non-existent model file fails deterministically. This makes domain-specific loss signal genuine rather than style-level noise.
3. **Dataset enrichment is feasible**: The `electron-rare/mascarade-*` HuggingFace collections (electronics, embedded, EDA tooling) can be merged with existing KIKI-Mac_tunner data for a 2–5× dataset size increase, pushing the params/example ratio into safe territory for these 10 domains.

---

## 2. Domains: KEEP (10)

These domains exhibit genuine capability gaps in the base model and have sufficient or enrichable datasets.

| # | Domain | Primary source | HF enrichment | Total est. examples |
|---|--------|---------------|--------------|-------------------|
| 01 | `kicad-dsl` | KIKI-Mac_tunner (1,980) | mascarade-kicad (2,645) | ~4,625 |
| 02 | `spice` | KIKI-Mac_tunner (2,675) | mascarade-spice (3,091) | ~5,766 |
| 03 | `emc` | KIKI-Mac_tunner (1,693) | mascarade-emc (3,360) | ~5,053 |
| 04 | `stm32` | KIKI-Mac_tunner (711) | mascarade-stm32 (2,012) | ~2,723 |
| 05 | `embedded` | KIKI-Mac_tunner (1,532 + 8,344) | mascarade-embedded (3,950) | ~13,826 |
| 06 | `freecad` | KIKI-Mac_tunner (219) | — | ~219 (needs augment) |
| 07 | `platformio` | KIKI-Mac_tunner (223) | — | ~223 (needs augment) |
| 08 | `power` | KIKI-Mac_tunner (1,238) | mascarade-power (3,267) | ~4,505 |
| 09 | `dsp` | KIKI-Mac_tunner (953) | mascarade-dsp (3,160) | ~4,113 |
| 10 | `electronics` | KIKI-Mac_tunner (1,900) | — | ~1,900 |

Notes:
- `freecad` and `platformio` are below the safety threshold even after known sources. These require teacher-distilled augmentation (Qwen3-Coder-480B) before training or must be dropped in a follow-up review.
- `embedded` benefits from three source streams and reaches ~14K examples — the strongest case for LoRA in this set.
- All HF enrichment pulls from `electron-rare/mascarade-*` collections (Apache 2.0 / CC-BY 4.0 compatible).

---

## 3. Domains: DROP (22)

These domains are dropped because the base model provides adequate-to-excellent coverage and the available datasets are insufficient to escape the overfitting regime.

| # | Domain | Reason for drop |
|---|--------|----------------|
| 01 | `chat-fr` | Empirically proven: overfits by iter 300, base model natively fluent in FR |
| 02 | `reasoning` | 35B has native thinking mode; LoRA would degrade CoT structure |
| 03 | `python` | Top-tier Python on EvalPlus; 2K examples cannot improve |
| 04 | `typescript` | Strong base coverage; no HF enrichment available at target density |
| 05 | `cpp` | Extensive C++ in pretraining (LLVM, Linux kernel, game engines) |
| 06 | `rust` | High base quality; Rust Book + crates.io well represented |
| 07 | `html-css` | Trivially covered; LoRA risk outweighs gain |
| 08 | `shell` | bash/zsh/fish well covered; LoRA adds style noise |
| 09 | `sql` | PostgreSQL/SQLite heavily represented; no niche gap |
| 10 | `yaml-json` | Structural format; base model handles edge cases correctly |
| 11 | `docker` | Dockerfile + Compose well covered; no hardware specificity |
| 12 | `lua-upy` | MicroPython/Lua present but dataset too small; defer to embedded LoRA |
| 13 | `iot` | Covered by embedded + stm32 stacks; redundant domain |
| 14 | `kicad-pcb` | Overlaps with kicad-dsl; consolidate into single stack |
| 15 | `spice-sim` | Overlaps with spice; consolidate |
| 16 | `web-frontend` | React/Vue/Svelte well covered; no niche gap |
| 17 | `web-backend` | FastAPI/Express/Django covered; no niche gap |
| 18 | `music-audio` | Niche but dataset (~500 examples) far below safety threshold |
| 19 | `devops` | CI/CD covered; no hardware specificity |
| 20 | `llm-orch` | Meta-domain; base model has strong LLM-about-LLM knowledge |
| 21 | `math` | 35B-A3B competitive on MATH-500; LoRA on math risks disrupting reasoning |
| 22 | `security` | Broad domain; existing dataset lacks specificity for meaningful gap-fill |

---

## 4. Architecture Changes

### 4.1 Router: 32 → 11 outputs

| Component | Before | After |
|-----------|--------|-------|
| Router outputs | 32 sigmoid scores | 10 domain scores + 1 passthrough |
| Routing logic | Multi-label sigmoid (max 4 active) | Argmax + confidence threshold; passthrough if max < 0.65 |
| Training targets | Outcome discovery (32 forward passes/prompt) | Domain classifier on 10 classes |
| Router training cost | ~4× base inference per sample | Single forward pass per sample |
| Active stacks cap | 4 (VRAM constraint) | 2 (sufficient for 10 specialised domains) |

Passthrough path: if router confidence < 0.65 on all 10 domain heads, the request is served directly by the base 35B-A3B with no adapter loaded. For the 22 dropped domains, this path handles all traffic — correctly, since the base model is already optimal.

### 4.2 Training schedule

| Metric | Before | After |
|--------|--------|-------|
| Total stacks | 32 | 10 |
| Time per stack | 45 min | 45 min |
| Total training time | 32 × 45 min = **24 h** | 10 × 45 min = **7.5 h** |
| Forgetting checks | 32 (after each stack) | 10 (after each stack) |
| Dataset prep overhead | 32 splits | 10 splits + HF merge step |

### 4.3 Multi-model routing (unchanged intent, refined allocation)

The base routing policy for the full system remains:

| Tier | Model | Use case |
|------|-------|---------|
| Fast | 35B-A3B + domain LoRA | Hardware/EDA queries (10 domains) |
| Fast | 35B-A3B base (passthrough) | General queries in dropped domains |
| Deep | Qwen3-Coder-480B-A35B | Complex multi-domain synthesis, teacher distillation |
| Code-specialized | devstral-v3 (GGUF Q4_K_M) | Firmware generation tasks alongside stm32/embedded stacks |

---

## 5. Data Enrichment Strategy

### 5.1 HuggingFace mascarade datasets

Source: `electron-rare/mascarade-*` repositories on HuggingFace Hub.

Merge procedure:
1. Download each mascarade dataset matching a KEEP domain.
2. Deduplicate against existing KIKI-Mac_tunner splits using SHA-256 on normalized prompt text.
3. Apply the same classifier used in the original pipeline to verify domain label integrity.
4. Re-split 80/10/10 (train/valid/test) on the merged corpus.
5. Re-run example count check: if merged total < 2,000 examples, flag for teacher augmentation before training.

### 5.2 Teacher augmentation for sub-threshold domains

Domains with merged counts below 2,000 examples (currently `freecad` ~219 and `platformio` ~223) require synthetic augmentation:

- Use Qwen3-Coder-480B-A35B (local Mac Studio) as teacher.
- Generate 1,000–2,000 additional QA pairs per domain using structured prompts derived from official documentation (FreeCAD Python API docs, PlatformIO registry).
- Verify with `distill_teacher.py --require-verify` flag (without this flag, all examples pass silently — known gotcha).
- Target: ≥2,000 verified examples per domain before training.

### 5.3 Params/example ratio targets post-enrichment

| Domain | Post-enrich examples | Params/example | Status |
|--------|---------------------|----------------|--------|
| embedded | ~13,826 | 498 | Near safe (good enough for niche) |
| spice | ~5,766 | 1,194 | Acceptable for structured output |
| emc | ~5,053 | 1,362 | Acceptable |
| kicad-dsl | ~4,625 | 1,487 | Acceptable |
| power | ~4,505 | 1,527 | Acceptable |
| dsp | ~4,113 | 1,673 | Acceptable |
| stm32 | ~2,723 | 2,527 | Marginal — monitor for overfitting |
| electronics | ~1,900 | 3,622 | High risk — augment or use rank 8 |
| freecad | ~1,500* | 4,587 | Requires teacher augment |
| platformio | ~1,500* | 4,587 | Requires teacher augment |

*Post-augmentation targets. Current raw counts are 219 and 223.

For high-ratio domains (electronics, freecad, platformio), the mitigation is to halve the LoRA rank from 16 to 8, reducing trainable params to ~3.4 M and improving the ratio ~2×. This is the only rank deviation from the project default.

---

## 6. Impact on Existing Code

### 6.1 Preserved (no changes required)

| Component | Location | Status |
|-----------|----------|--------|
| Cognitive layer (Aeon, Negotiator, Anti-bias) | `src/cognitive/` | Unchanged — domain-agnostic |
| MLX serving pipeline | `src/serving/` | Unchanged |
| Spiking neural network experiments | `src/spiking/` | Unchanged — orthogonal research track |
| Eval harness + forgetting check framework | `src/eval/`, `scripts/eval_stack.py` | Unchanged — applies to 10 stacks |
| OPLoRA projection for forgetting prevention | `src/stacks/oplora.py` | Unchanged |
| Curriculum training configs (3-phase, LR schedule) | `configs/micro_kiki/` | Unchanged — applies to each of the 10 stacks |
| Dispatcher (7 meta-intents) | `src/dispatch/` | Unchanged — meta-intent layer is above domain routing |

### 6.2 Changed

| Component | Location | Change required |
|-----------|----------|----------------|
| Router model | `scripts/micro_kiki/meta_router.py` | Reduce output dimension 32 → 11; add passthrough head |
| Router config | `configs/micro-kiki-router.yaml` | Update `num_domains: 32` → `num_domains: 10`, add passthrough threshold |
| Constants | `scripts/micro_kiki/constants.py` | Replace 32-domain list with 10-domain list + passthrough label |
| Training orchestration | `scripts/micro_kiki/train_all_stacks.sh` | Update curriculum order to 10 domains |
| Data pipeline | `scripts/micro_kiki/data_pipeline.py` | Add HF merge + dedup step; add sub-threshold augmentation check |
| PRD / plans | `docs/plans/` | Plans 2–3 reference 32 stacks; add deprecation note at top |

### 6.3 Deleted (safe to remove)

The following stack training work products for dropped domains can be removed once the router and data pipeline are updated:

- `data/micro-kiki/<dropped-domain>/` splits for all 22 dropped domains (saves ~15 GB estimated)
- Any partially trained adapter checkpoints for dropped domains in `output/micro-kiki/stacks/`

Stack-01 `chat-fr` checkpoint (overfitted, best iter 300, val 0.722) should be **archived but not used** — it demonstrates the failure mode and serves as a reference data point.

---

## 7. Decision Record

| Question | Decision |
|----------|----------|
| Keep 35B-A3B base? | Yes — architecture pivot confirmed, no regression |
| Train stacks for dropped domains? | No — base model handles them; LoRA degrades |
| Use rank 16 for all 10 stacks? | No — use rank 8 for electronics, freecad, platformio |
| Augment sub-threshold domains? | Yes — teacher distillation with `--require-verify` |
| Merge HF mascarade data? | Yes — dedup + re-classify before training |
| Router: 32 sigmoid → 35-class sigmoid (final)? | Yes — sigmoid outputs (not argmax) for 34 niches + base; confidence-driven routing via thresholds |
| Training time saving? | 16.5 h saved (22 dropped × 45 min) |
| Risk of further overfitting? | Mitigated by enrichment; stm32/electronics flagged for early stopping |

---

## 8. Next Steps

1. Update `scripts/micro_kiki/constants.py` with 10-domain list.
2. Run HF merge + dedup for 8 enrichable domains.
3. Run teacher augmentation for freecad and platformio (target 1,500 each).
4. Update `configs/micro-kiki-router.yaml`: `num_domains: 10`, add passthrough.
5. Rewrite `meta_router.py` for 11-output architecture.
6. Begin training stack-01 `kicad-dsl` (highest evidence of base model gap).
7. After each stack, run forgetting check; rollback if angle < 30° AND win-rate drop > 0.03.
8. Archive stack-01 `chat-fr` checkpoint to `output/micro-kiki/archive/chat-fr-overfit/`.
