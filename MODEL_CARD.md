---
license: apache-2.0
language:
  - fr
  - en
tags:
  - moe
  - lora
  - multi-domain
  - embedded-systems
  - cognitive
base_model: Qwen/Qwen3.6-35B-A3B
pipeline_tag: text-generation
---

# micro-kiki

**34-domain expert model** built on Qwen3.6-35B-A3B (MoE, 256 experts, 3B active/token) with LoRA adapters and a cognitive layer (memory palace + negotiator + anti-bias).

## Model Description

micro-kiki is a multi-domain language model designed for technical applications spanning electronics, firmware, CAD, manufacturing, and general-purpose conversation. It uses a router-based architecture that selects up to 4 domain-specific LoRA stacks per request.

| Property | Value |
|----------|-------|
| Base model | Qwen3.6-35B-A3B |
| Architecture | MoE (256 experts, 3B active/token) |
| Adapter | LoRA, rank tiers {4, 8, 12, 16, 32}, MLX scale = 20.0, 17 module kinds/layer (attention + MoE routers + shared_expert + switch_mlp) |
| Domains | 34 |
| Max active stacks | 4 |
| Context length | 262,144 tokens |
| Quantization | Q4_K_M (inference), BF16 (training) |
| License | Apache 2.0 |

### Adapter shape — 17 module kinds per layer

LoRA is applied (via `mlx_lm lora`) to:

- `linear_attn.{in_proj_a, in_proj_b, in_proj_qkv, in_proj_z, out_proj}` (GLA hybrid, 5 modules)
- `self_attn.{q_proj, k_proj, v_proj, o_proj}` (4 modules)
- `mlp.gate`, `mlp.shared_expert_gate` (MoE routers, 2 modules)
- `mlp.shared_expert.{down_proj, gate_proj, up_proj}` (3 modules)
- `mlp.switch_mlp.{down_proj, gate_proj, up_proj}` (3 modules)

Prior model-card revisions described "q/k/v/o attention projections only"; that was superseded 2026-04-18 after the real `adapter_config.json` was read and a forgetting audit (chat-fr ↔ reasoning, mean 79.4°, no catastrophic interference) demonstrated the 17-module surface holds. Pre-pivot MoE-LoRA adapters (stacks-v3-r16, 35 files) had `lora_B = 0` across all modules — archived to `scripts/legacy/`, see `docs/research/2026-04-19-prepivot-moe-lora-audit.md` and `docs/research/2026-04-19-moe-lora-root-cause.md`.

### Forgetting-gate enforcement

After every sequential stack trains, the adapter is admitted to the curriculum only if:

- `scripts/validate_adapter_health.py` → all `lora_B` matrices non-zero, AND
- `scripts/measure_forgetting.py` → per-module cosine angle ≥ 30° vs. every prior stack, OR win-rate drop ≤ 0.03 on cross-domain probes.

Canonical doc: `docs/training/forgetting-gate.md`. Operator runbook for the dual-server flow: `docs/training/e2e-smoke-runbook.md`. Empirical sweeps in `results/forgetting-matrix.json`, `results/adapter-health-sweep.json`, `results/smoke-gate.json`.

## Architecture

```
                         +-------------------+
                         |   Domain Router   |
                         | (classifier, top4)|
                         +--------+----------+
                                  |
              +----------+--------+--------+----------+
              |          |                 |          |
         +----v----+ +---v---+       +----v----+ +---v---+
         | Stack 1 | |Stack 2|  ...  |Stack 33 | |Stack34|
         | chat-fr | |python |       |ml-train | |securi.|
         +---------+ +-------+       +---------+ +-------+
              |          |                 |          |
              +----------+--------+--------+----------+
                                  |
                         +--------v----------+
                         |    Negotiator     |
                         | CAMP + Catfish    |
                         +--------+----------+
                                  |
                         +--------v----------+
                         |    Anti-Bias      |
                         | KnowBias + RBD   |
                         +--------+----------+
                                  |
                         +--------v----------+
                         |   Aeon Memory     |
                         | Atlas + Trace     |
                         +-------------------+
```

## Intended Use

- **French/English conversational AI** with domain expertise
- **Code generation** (Python, C/C++, Rust, TypeScript, embedded firmware)
- **Electronics design** (KiCad DSL, schematic review, component selection, SPICE)
- **Manufacturing** (process optimization, quality control)
- **Multi-domain routing** with cognitive arbitration

## Limitations

- Not designed for medical, legal, or financial advice
- Optimized for technical domains; general knowledge may be weaker than base model
- Requires Q4_K_M or higher quantization; quality degrades below Q4
- Maximum 4 concurrent LoRA stacks; performance varies with stack combinations
- Memory (Aeon) requires external backends (Qdrant/Neo4j) for production use

## Training Data — V3 (489K examples, 35 domains)

### Sources

| Source | Examples | Description |
|--------|----------|-------------|
| Claude CLI sessions | 50,116 | Real user-tool interactions extracted from 5 machines (GrosMac, kxkm-ai, Studio, Tower, CILS) |
| Codex/Copilot sessions | 2,529 | OpenAI Codex + GitHub Copilot sessions extracted from 4 machines |
| HuggingFace datasets | 364,045 | 19 open datasets (see below) |
| Opus teacher distillation | — | chat-fr, reasoning domains |
| Original curated | — | 32 domain seed datasets |

### HuggingFace Datasets

| Dataset | Examples | License |
|---------|----------|---------|
| CodeFeedback-Filtered-Instruction | 157,000 | Apache 2.0 |
| French-Alpaca-Instruct-110K | 110,000 | Apache 2.0 |
| Electronics StackExchange | 95,000 | CC-BY-SA-3.0 |
| CJJones/LLM_EE_Educational_Synthetic_Dialog | 50,000 | CC-BY-NC-SA-4.0 |
| MuratKomurcu/stm32-hal-dataset | 29,700 | MIT |
| redcathode/thingiverse-openscad | 7,400 | — |
| ThomasTheMaker/OpenSCAD | 4,900 | — |
| STEM-AI-mtl/Electrical-engineering | 1,100 | — |
| JITX open-components-database | 151 | — |
| Vrindarani/netlistgen | 106 | — |

### 34 Domains

| Group | Domains |
|-------|---------|
| Conversation | chat-fr, reasoning |
| Code | python, typescript, cpp, rust, html-css, shell, sql, yaml-json, lua-upy |
| Infrastructure | docker, devops, llm-orch, llm-ops, ml-training |
| Electronics | kicad-dsl, kicad-pcb, spice, electronics, components, power, emc, dsp |
| Hardware | embedded, stm32, iot, platformio |
| CAD | freecad |
| Web | web-frontend, web-backend |
| Other | music-audio, math, security |

**Changes from pre-v3 (32 domains):** 3 new domains (`components`, `llm-ops`, `ml-training`). `spice-sim` removed (merged into `spice`). Net: +2 → 34 total. The list must stay in sync across 3 config mirrors (`configs/micro_kiki/domains.yaml`, `configs/micro_kiki/brainstacks.yaml`, `configs/mlx-per-domain/*.yaml`) — enforced by `scripts/validate_domains.py`.

### New Domain: components

57K Q&A about electronic component specs, datasheets, sourcing, BOM, and cross-reference. Sources: Electronics StackExchange (filtered by component tags) + JITX open-components-database.

## Training — V3 (post-pivot 2026-04-16)

| Property | Value |
|----------|-------|
| Base model | Qwen3.6-35B-A3B (pivot from pre-v3 Qwen3.5-4B + MoE-LoRA) |
| Adapter | Standard LoRA, rank tiers {4, 8, 12, 16, 32}, MLX scale = 20.0, 17 module kinds/layer |
| Trainer | `mlx_lm lora` on Mac Studio M3 Ultra 512 GB (BF16) |
| Forgetting gate | `scripts/post_train_gate.py` — health + angle + win-rate after every stack (rollback if angle < 30° AND win-rate drop > 0.03) |
| Curriculum | Sequential, 34 stacks, foundations (rank 32) before niches — enforced by `scripts/validate_curriculum_order.py` |
| Platform (MLX) | Mac Studio M3 Ultra 512 GB |
| Platform (CUDA) | kxkm-ai RTX 4090 24 GB (Q4 inference only; **do not train** — 35B BF16 LoRA does not fit in 24 GB) |

**Pre-pivot warning.** 35 MoE-LoRA adapters from the pre-pivot pipeline (`stacks-v3-r16/`) have `lora_B = 0` across all modules and are effectively dead weights. They are archived under `scripts/legacy/`. Do not deploy them. See `docs/research/2026-04-19-prepivot-moe-lora-audit.md` (audit) and `docs/research/2026-04-19-moe-lora-root-cause.md` (root cause).

## Evaluation

| Metric | Value |
|--------|-------|
| Router accuracy (35-class) | [PENDING] |
| Forgetting check — per-module angle, post-pivot (chat-fr ↔ reasoning, `results/smoke-gate.json`) | mean 79.4°, winrate_drop −0.04, gate PASS |
| Forgetting sweep — post-pivot (`results/forgetting-matrix.json`) | 5 adapters × 20 pairs, all above 30° |
| Forgetting sweep — pre-pivot MoE-LoRA (`results/forgetting-matrix-prepivot.json`) | 35 adapters, all 0° (degenerate — `lora_B = 0`) |
| Adapter health sweep (`results/adapter-health-sweep.json`) | 35/35 post-pivot healthy · 35/35 pre-pivot degenerate |
| Perplexity (base) | [PENDING] |
| Perplexity (debiased) | [PENDING] |
| Aeon recall@1 | [PENDING] |
| Aeon recall@5 | [PENDING] |
| Aeon recall@10 | [PENDING] |
| Anti-bias flag rate | [PENDING] |
| Average inference latency | [PENDING] |

## Hardware Requirements

| Setup | RAM/VRAM | Use |
|-------|----------|-----|
| Mac Studio M3 Ultra | 512 GB unified | Training (BF16 LoRA) + serving (MLX) |
| RTX 4090 | 24 GB VRAM | Q4 inference (vLLM) |
| Apple Silicon 32 GB+ | 32 GB unified | Q4_K_M inference (MLX/llama.cpp) |

## Citation

```bibtex
@misc{micro-kiki-2026,
  title={micro-kiki: Multi-Domain Expert Model with Cognitive Layer},
  author={L'Electron Rare},
  year={2026},
  url={https://huggingface.co/electron-rare/micro-kiki}
}
```
