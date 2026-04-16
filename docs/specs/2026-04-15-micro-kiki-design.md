> **Architecture pivot — 2026-04-16**: The Qwen3.5-4B + custom MoE-LoRA design described in this document has been superseded. The active architecture uses Qwen3.5-35B-A3B (native MoE, 256 experts) with standard LoRA. See `docs/specs/2026-04-16-architecture-pivot-35b.md` for the pivot rationale and updated decisions. This document is preserved as historical reference.

# Micro_KIKI — 32 Expert Brainstacks MoE

## Overview

Fleet of 32 specialized domains on a Qwen3.5-4B base, assembled via Brainstacks (MoE-LoRA + null-space projection + sigmoid meta-router). Deployable on RTX 4090 24 GB.

## Architecture

```
┌──────────────────────────────────────────────────┐
│              SIGMOID META-ROUTER                  │
│     32 independent outputs per prompt             │
│     Input: mid-layer + last-layer hidden states   │
│     ~2M params, 5ms overhead                      │
└──────────┬───────────────────────────────────────┘
           │ activates relevant stacks (threshold 0.12)
           ▼
┌──────────────────────────────────────────────────┐
│         BASE MODEL: Qwen3.5-4B (frozen, Q4)      │
│         2.5 GB VRAM permanent                     │
│         GatedDeltaNet, 262K ctx, native thinking  │
├──────────────────────────────────────────────────┤
│  Stack 1: chat-fr       │  Stack 17: embedded     │
│  Stack 2: python        │  Stack 18: stm32        │
│  Stack 3: typescript    │  Stack 19: iot           │
│  Stack 4: cpp           │  Stack 20: freecad       │
│  Stack 5: rust          │  Stack 21: platformio    │
│  Stack 6: html-css      │  Stack 22: power         │
│  Stack 7: shell         │  Stack 23: emc           │
│  Stack 8: sql           │  Stack 24: dsp           │
│  Stack 9: yaml-json     │  Stack 25: spice-sim     │
│  Stack 10: kicad-dsl    │  Stack 26: electronics   │
│  Stack 11: spice        │  Stack 27: web-frontend  │
│  Stack 12: docker       │  Stack 28: web-backend   │
│  Stack 13: lua-upy      │  Stack 29: music-audio   │
│  Stack 14: math         │  Stack 30: devops        │
│  Stack 15: security     │  Stack 31: llm-orch      │
│  Stack 16: reasoning    │  Stack 32: kicad-pcb     │
└──────────────────────────────────────────────────┘
  Each stack: MoE-LoRA (4 experts, rank 16, top-2)
  ~150 MB/stack × 32 = ~4.8 GB disk
  2-4 stacks active simultaneously in VRAM (~1-2 GB)
```

## Hardware constraints

| Machine | VRAM | Role |
|---------|------|------|
| Mac M3 Ultra 512 GB | Unlimited | Stack training, teacher distillation |
| RTX 4090 24 GB (kxkm-ai) | 24 GB | Inference, Unsloth training |

### RTX 4090 VRAM budget (inference)

| Component | VRAM |
|-----------|------|
| Base Qwen3.5-4B Q4 | 2.5 GB |
| Meta-router | 0.01 GB |
| 2-4 active stacks | 0.6-1.2 GB |
| KV cache (4K ctx) | ~0.5 GB |
| **Total** | **~4-5 GB** |
| **Headroom** | **19 GB** |

## Base model: Qwen3.5-4B

### Why

| Criterion | Qwen3.5-4B | Gemma 4 E4B | Nemotron Nano 4B |
|-----------|-----------|------------|-----------------|
| MMLU-Pro | **79.1** | 69.4 | ~65 |
| GPQA-D | **76.2** | 58.6 | — |
| Native thinking | **Yes** | No | No |
| 262K context | **Yes** | No | No |
| Same family as 122B | **Yes** | No | No |
| French (201 languages) | **Yes** | Limited | No |
| Q4 VRAM | 2.5 GB | 3 GB | 2.5 GB |
| License | Apache 2.0 | Apache 2.0 | NVIDIA Open |

Qwen3.5-4B wins on every criterion except raw HumanEval (Gemma 4 E4B scores 85%+). But with thinking mode and the same DeltaNet architecture as our 122B/35B teachers, distillation will be optimal.

## Brainstacks — adaptations for 32 domains

### Parameters

| Param | Paper (5 dom.) | Micro_KIKI (32 dom.) |
|-------|----------------|---------------------|
| Base model | Gemma 3 12B | Qwen3.5-4B |
| `h_dim` | 3840 | 3072 (Qwen3.5-4B) |
| `ns_top_k_dirs` | 64 | **32** |
| Null-space used | 8.3% | **33%** (32×32/3072) |
| MoE experts/stack | 4 | 4 |
| LoRA rank | 16 | 16 |
| Residual boost rounds | 2-3 | 1-2 |
| Stack size | 567 MB (12B) | **~150 MB** (4B) |
| Total disk | 5.67 GB | **~4.8 GB** |
| Meta-router outputs | 5 | **32** |

### Curriculum order (sequential, each domain does not degrade prior ones)

```
Phase 1 — Foundations (scaffolding)
  1. chat-fr        : instruction-following + French
  2. reasoning      : meta-reasoning, thinking chains

Phase 2 — Coding core (procedural logic)
  3. python         : primary coding
  4. typescript     : web + types
  5. cpp            : systems + embedded
  6. rust           : safety + concurrency

Phase 3 — Coding secondary
  7. html-css       : frontend markup
  8. shell          : scripts, DevOps
  9. sql            : queries, schemas
  10. yaml-json     : configs, schemas
  11. docker        : containers
  12. kicad-dsl     : netlists, footprints
  13. spice         : simulations
  14. lua-upy       : embedded scripting

Phase 4 — Technical domains (upgrade kiki-*)
  15. embedded      : ESP-IDF, general firmware
  16. stm32         : STM32 HAL, CubeMX
  17. iot           : protocols, MQTT, BLE
  18. freecad       : mechanical CAD
  19. platformio    : build system
  20. power         : power supply, regulators
  21. emc           : EMC, filtering
  22. dsp           : signal processing
  23. spice-sim     : circuit simulation
  24. electronics   : analog, RF, components
  25. kicad-pcb     : PCB routing, DRC

Phase 5 — Applications
  26. web-frontend  : React, Vite, patterns
  27. web-backend   : FastAPI, Hono, Express
  28. music-audio   : audio DSP, TTS, instruments
  29. devops        : Docker, Tailscale, CI/CD
  30. llm-orch      : RAG, agents, LLM routing

Phase 6 — Complements
  31. math          : math/physics reasoning
  32. security      : crypto, auth, OWASP
```

## Distillation — progressive multi-teacher chain

```
Teachers (on Mac 512 GB):
  ├── Qwen3.5-122B-A10B Opus-v3 (in training, val 0.497)
  ├── Gemma 4 31B (18 GB bf16, fast)
  └── Devstral 2 123B (for coding)

Chain:
  122B → 35B → 4B (progressive, 80-88% quality retained)

Per domain:
  1. Generate ~2K specialized examples with the appropriate teacher
  2. Deduplicate cross-domain
  3. SFT via Brainstacks (inner loop + null-space)
```

### Teachers per domain

| Domains | Primary teacher | Secondary teacher |
|----------|------------------|-------------------|
| Coders (3-14) | Devstral 2 123B | Gemma 4 31B |
| Embedded (15-25) | 122B Opus-v3 | Existing kiki-* data |
| Reasoning/Math (1-2, 31) | 122B Opus-v3 | Opus API |
| Web/DevOps (26-30) | Gemma 4 31B | 122B Opus-v3 |
| Security (32) | 122B Opus-v3 | — |

## Data — sources and deduplication

### Existing sources

| Source | Examples | Domains |
|--------|----------|----------|
| final-opus-v3-1 | 11,880 | Reasoning, general |
| 10 kiki-* LoRA datasets | ~5,000 estimated | Embedded, hardware |
| CodeFeedback | 156K | Coding |
| OpenCodeReasoning | 735K | Python coding |
| Magicoder-OSS | 75K | Multi-language code |

### Budget per domain

| Tier | Examples/domain | Total 32 domains |
|------|-----------------|-------------------|
| Minimum viable | 500 | 16K |
| Recommended | 2,000 | 64K |
| Optimal | 5,000 | 160K |

### Cross-domain deduplication

Each example goes into **exactly 1 domain** (the one with the highest relevance score). No duplicates between stacks — null-space projection handles cross-domain transfer.

## Training pipeline

### Per stack (~30 min on Mac, ~20 min on RTX)

```
1. Load frozen Qwen3.5-4B base
2. Compute null-space projector of frozen stacks
3. Add MoE-LoRA stack (4 experts, rank 16, 7 projections)
4. SFT on ~2K domain examples (~500 steps)
5. Residual boost: round 2 if improvement > 0.002
6. Freeze → offload to CPU/disk
7. Evaluate all prior domains (forgetting check)
```

### Total time

| Phase | Mac only | Mac + RTX parallel |
|-------|----------|---------------------|
| Data distillation (32 × 2K) | ~48h | ~48h (Mac teacher) |
| Training 32 stacks | ~16h | **~8h** |
| Residual boost | ~8h | ~4h |
| Meta-router | ~2h | ~1h |
| Eval + iterate | ~4h | ~4h |
| **Total** | **~78h** | **~65h** |

## Meta-router — 32 sigmoids

### Architecture

```
Input: 0.45 × mid_hidden + 0.55 × last_hidden (Qwen3.5-4B h_dim=3072)
→ Linear(3072, 512)
→ Global attention (learned query)
→ 32 × cross-attention (domain query vectors)
→ MLP fusion (GELU, dropout 0.1)
→ 32 sigmoid outputs with temperature scaling
```

### Training (outcome discovery)

For each prompt in the mixed dataset:
1. Loss base-only
2. Loss with each individual stack (32 forwards)
3. Greedy search: add stacks that reduce loss by > 0.01
4. Target: 80% discovered + 20% prior label
5. BCE loss, 8 epochs, cosine LR

### Inference rules

- Chat floor: 0.20 (always active minimum)
- Gate threshold: 0.12 (below, stack not loaded)
- Max simultaneous stacks: 4 (VRAM constraint)

## Export and deployment

### RTX 4090 (kxkm-ai)

```
Base Q4: models/Qwen3.5-4B-Q4.gguf (2.5 GB)
Stacks: output/micro-kiki/stacks/ (32 × 150 MB)
Router: output/micro-kiki/router.safetensors (8 MB)
Inference: vLLM with LoRA switching OR custom script
```

### Mac Studio (local)

```
Base bf16: models/Qwen3.5-4B (8 GB)
MLX stacks: same format
Inference: mlx-lm with adapter switching
```

## Success criteria

| Metric | Target |
|--------|--------|
| Coding (HumanEval) | > 70% (Qwen3.5-4B base ~55%) |
| Reasoning (GPQA) | > 80% (base 76.2%) |
| Embedded (custom eval) | Correct answers on ESP-IDF, KiCad |
| French (custom eval) | Fluent, no code-switching |
| Zero forgetting | Delta < 0.03 on all prior domains |
| RTX 4090 VRAM | < 8 GB at inference |
| Router latency | < 10 ms |
| Stack swap | < 2s |

## Apple Silicon — ANE+GPU+CPU triple pipeline (Mac only)

On the Mac M3 Ultra, the ANE (Neural Engine) is idle while the GPU runs MoE inference. Three integrations exploit this otherwise-unused resource.

### Triple pipeline architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MAC M3 ULTRA 512 GB                        │
│                                                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │   GPU METAL      │  │   ANE (32 cores)  │  │    CPU      │ │
│  │   76 cores       │  │   ~2W, 14 tok/s   │  │  24 cores   │ │
│  │                  │  │                    │  │             │ │
│  │  Base Qwen3.5-4B │  │  A. GRPO scorer   │  │  Sigmoid    │ │
│  │  + 2-4 active    │  │  B. Draft 0.8B    │  │  router     │ │
│  │  stacks          │  │     (speculative)  │  │  (5ms)      │ │
│  │                  │  │  C. Meta-router    │  │             │ │
│  │  Primary         │  │     + embedding    │  │  Stack      │ │
│  │  generation      │  │                    │  │  offload    │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
│         ↕ unified memory (zero-copy)  ↕                       │
└─────────────────────────────────────────────────────────────┘
```

### A. ANE as scorer / quality filter

During GRPO training (Phase 3), the GPU generates K=4 responses per prompt.
The ANE scores each response in parallel via a lightweight reward model.

```
GPU: generates response[i+1]  ──────────────────────→
ANE: scores response[i]       ──→ reward = 0.85 ──→
                               (14 tok/s)
```

| Component | Unit | Model |
|-----------|------|-------|
| Generator | GPU Metal | Qwen3.5-4B + MoE stacks |
| Scorer | ANE CoreML | Qwen3.5-0.8B converted to CoreML |
| Reward head | ANE | Linear(h_dim, 1) on top of the scorer |

Gain: **free scoring** (zero impact on GPU generation speed).

### B. Speculative decoding via ANE

A draft Qwen3.5-0.8B model (0.5 GB) runs on the ANE. It proposes N tokens;
the GPU (4B + stacks) verifies them in a single forward pass.

```
ANE (draft 0.8B): proposes tokens [t1, t2, t3, t4, t5]  → 200+ tok/s
GPU (4B + stacks): verifies [t1✓, t2✓, t3✓, t4✗]         → 1 forward
                    accepts 3 tokens instead of 1
```

| Metric | Without speculative | With ANE speculative |
|--------|---------------------|----------------------|
| GPU tok/s | ~30-50 | ~30-50 |
| Effective tok/s | ~30-50 | **~60-100** (2-3x) |
| Extra VRAM | 0 | 0 (ANE is separate) |

The draft 0.8B shares the same tokenizer as the 4B (same Qwen3.5 family).
CoreML conversion is proven (we have already converted the 9B DeltaNet).

### C. ANE for meta-router + embedding

The meta-router (2M params) and the embedding layer are lightweight ops
that can run entirely on ANE, freeing the GPU for the MoE stacks.

```
Prompt arrives
  │
  ▼
ANE: embedding(tokens) → hidden states           (~0.5 ms)
ANE: meta_router(hidden) → 32 sigmoid scores     (~2 ms)
CPU: selects top-4 stacks, loads from SSD         (~50 ms)
  │
  ▼
GPU: forward(hidden, active stacks) → tokens      (bulk of compute)
```

The GPU forward only does the heavy MoE compute. Embedding + routing = free on ANE.

### Required CoreML models

| Model | Use | CoreML size | Conversion |
|--------|-----|-------------|------------|
| Qwen3.5-0.8B | Speculative draft | ~1 GB | TBD (ANEMLL or custom) |
| Meta-router | 32-stack routing | ~8 MB | Trivial (small MLP) |
| Embedding layer | Token → hidden | ~50 MB | Trivial |
| Reward scorer | GRPO scoring | ~1 GB | Clone of the draft + reward head |

**Note**: The 0.8B Qwen3.5 uses GatedDeltaNet like the 4B/9B.
Our DeltaNet → CoreML conversion (Phase 1 ANE research) applies directly.

### When to use the triple pipeline

| Scenario | GPU | ANE | CPU | Gain |
|----------|-----|-----|-----|------|
| Standard inference | 4B + stacks | Draft 0.8B (spec) | Router | **2-3x tok/s** |
| GRPO training | Generates K=4 | Scores responses | Router | **Free scoring** |
| SFT training | LoRA training | Idle | — | No gain |
| Batch scoring | Idle | Scores dataset | — | **14 tok/s continuous** |

### Impact on success criteria

| Metric | Without ANE | With ANE |
|--------|-------------|----------|
| Inference tok/s | 30-50 | **60-100** (speculative) |
| GRPO scoring overhead | +50% time | **~0%** (parallel) |
| Router latency | ~5 ms CPU | **~2 ms ANE** |
| Power draw | ~20W GPU only | ~22W (GPU+ANE) |

## Risks

| Risk | Mitigation |
|------|-----------|
| 32 domains saturate the null-space | Drop ns_top_k_dirs to 32 (33% space) |
| Base 4B too small for 32 specializations | Upgrade to Qwen3.5-9B (5.5 GB Q4, fits on RTX) |
| Brainstacks untested with Qwen3.5 | Port the Gemma code → Qwen (same transformers API) |
| kxkm-ai unreachable (Tailscale) | Training 100% on Mac, deploy GGUF via NFS |

## Addendum 2026-04-15 — Cognitive layer added

After the initial design, a cognitive layer was added (see `docs/specs/2026-04-15-cognitive-layer-design.md`):

- Training-free dispatcher (7 meta-intents derived from router)
- Aeon memory palace (neuro-symbolic, arxiv 2601.15311)
- CAMP + Catfish negotiator with adaptive judge
- KnowBias + RBD anti-bias pipeline

The original 32-domain + router architecture is preserved unchanged. The cognitive layer wraps it.
