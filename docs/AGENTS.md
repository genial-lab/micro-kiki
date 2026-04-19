<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# docs

## Purpose
All project documentation that isn't in root-level markdown: dated specs (architecture pivots, cognitive layer design, quantum-inspired v0.2, neuroscience v0.3, AeonSleep, DiffAttention, SpikingBrain, LAS conversion, energy methodology, MAP paper), dated plans (data pipeline, brainstacks training, meta-router, ANE pipeline, v0.2 roadmap), research notes (landscape scan, MoE research, 2026 SoTA survey), the training README, superpowers (agentic capabilities), and setup walkthroughs. Specs follow RFC2119 tone; plans are dated and numbered.

## Key Files
| File | Description |
|------|-------------|
| `cookbook-v0.3.md` | v0.3 (neuroscience-branch) end-to-end recipes |
| `data-sources.md` | Catalogue of public + private datasets per domain |
| `paper-outline-triple-hybrid.md` | Paper outline: triple-hybrid (MoE + DiffAttention + Spiking) |
| `setup-studio-neuro.md` | Mac Studio setup guide for the neuroscience branch |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `specs/` | Dated design specs. Architecture pivot to 35B (`2026-04-16-architecture-pivot-35b.md`), reorientation rationale, cognitive layer design, v0.2 quantum-inspired, v0.3 neuroscience, AeonSleep architecture, DiffAttention integration, energy methodology, LAS conversion framework, MAP paper spec + validation report, MLX-LM fork reference, SpikingBrain (7B / acquisition / quant), STM32H743 USB bootloader design, Ethernet bootloader update protocol |
| `plans/` | Dated implementation plans. `2026-04-15-micro-kiki-plan{1..4}-*.md` (data pipeline, brainstacks training, meta-router, ANE pipeline), `v0.2-roadmap.md` |
| `research/` | Research notes: `2026-04-16-micro-kiki-landscape.md`, `micro-kiki-moe-research.md`, `sota-training-2026.md`. **Pivot audit (2026-04-19)**: `2026-04-19-prepivot-moe-lora-audit.md` (35 pre-pivot adapters all `lora_B=0` → dead weights) + `2026-04-19-moe-lora-root-cause.md` (root-cause trace for the bug). |
| `superpowers/` | `plans/` (agentic capabilities, dataset gen+training paper, phases I-III foundations, reorientation) + `specs/` (agentic capabilities design) |
| `training/` | `README.md` — authoritative 3-phase MLX curriculum (seq 512->1280->4096, LR 8e-6->5e-6->3e-6, grad-checkpoint mandatory, Qwen3.6-35B-A3B-Opus-bf16 base). `forgetting-gate.md` — canonical forgetting-gate operator ref (angle + win-rate, `scripts/measure_forgetting.py` / `scripts/post_train_gate.py`). `e2e-smoke-runbook.md` — operator runbook for the dual-server real-adapter flow (`scripts/smoke_gate_on_studio.py`). |

## For AI Agents

### Working In This Directory
- File names use ISO dates: `YYYY-MM-DD-<slug>.md`. New specs/plans MUST follow this pattern.
- Specs use RFC2119 language (MUST / SHOULD / MAY).
- The architecture pivot of 2026-04-16 is locked: Qwen3.5-35B-A3B + standard LoRA on attention only. Earlier docs (v0.2 quantum, v0.3 neuroscience) remain for historical context — do not edit them retroactively; write a new dated spec instead.
- `training/README.md` is the canonical operational guide — keep it in sync with `configs/mlx-*.yaml` and `~/KIKI-Mac_tunner/configs/mlx-lm-micro-kiki-phase*.yaml`.
- NEVER drop a spec that is referenced from a plan or the root `CLAUDE.md`; supersede instead.
- Research notes are frozen snapshots — mark them with a "freeze date" at the top if you update them.

### Testing Requirements
No tests live here. Doc changes do not trigger CI gates, but `tests/test_phase_*_configs.py` indirectly guards that spec-declared invariants (intent coverage, capability indices) still hold in `configs/`.

### Common Patterns
- Code fences are language-tagged (```python, ```bash, ```yaml).
- Tables over prose when comparing configs or metrics.
- Papers referenced by arXiv id (OPLoRA 2510.13003, Aeon 2601.15311, CAMP 2604.00085, Catfish 2505.21503, KnowBias 2601.21864, RBD 2505.17100).
- File headers include a one-line purpose; dated specs state their status (Draft / Accepted / Superseded).

## Dependencies

### Internal
Documents behaviour implemented in `src/`, scripts in `scripts/`, configs in `configs/`. The training README points at `scripts/distill_with_local_teacher.py` and `src/eval/forgetting.py`.

### External
Links to HuggingFace models (`Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3-Coder-480B-A35B-Instruct`), arXiv papers, and `~/KIKI-Mac_tunner` on the Mac Studio (training rig).

<!-- MANUAL: -->
