# micro-kiki

A 32-domain expert system built on Qwen3.5-4B with MoE-LoRA stacks, cognitive layer (memory palace, negotiator, anti-bias), and triple-device serving. Fits on RTX 4090 24 GB.

## What

Five tightly integrated layers that turn a small base model into a specialist team:

1. **Base**: Qwen3.5-4B (Apache 2.0, GatedDeltaNet hybrid, 262K ctx, 201 languages, thinking mode), with **Differential Attention** (arxiv 2410.05258) applied to the 13 full-attention layers for better long-context retrieval, lower hallucinations, and reduced activation outliers
2. **32 MoE-LoRA stacks**: one per domain, 4 experts × rank 16 each, ~150 MB per stack, ~4.8 GB total
3. **Meta-router**: sigmoid 32 outputs + training-free dispatcher (7 meta-intents)
4. **Cognitive layer**:
   - **Aeon memory palace** (Atlas SIMD index + Trace neuro-symbolic graph) — persistent, spatial, temporal-aware memory
   - **Negotiator** (CAMP arbitration + Catfish dissent) — resolves conflicts between active stacks with adaptive judge (Qwen3.5-35B fast ↔ Mistral-Large-Opus deep)
   - **KnowBias + RBD anti-bias** — neuron-level debiasing (before+after stacks) + runtime detector
5. **Serving**: vLLM with dynamic LoRA (kxkm-ai RTX 4090) OR mlx-lm (Mac Studio), with Apple Neural Engine triple pipeline (draft, scorer, router) as an optional accelerator

## Architecture

```
  user prompt
      ↓
  [Dispatcher]    → 7 meta-intents (training-free, zero latency)
      ↓
  [Aeon recall]   → inject top memories into context
      ↓
  [Meta-router]   → sigmoid 32 → activate 2-4 stacks
      ↓
  [Base + stacks] → K candidate responses
      ↓
  [Negotiator]    → CAMP arbitration (Qwen35B) / Mistral-Large if deep needed
      ↓
  [Anti-bias]     → RBD flag → DeFrame re-gen if biased
      ↓
  [Aeon write]    → persist the turn
      ↓
  response
```

## Domains (32)

Organized in 6 curriculum phases:

1. **Foundations** (chat-fr, reasoning)
2. **Coding core** (python, typescript, cpp, rust)
3. **Coding secondary** (html-css, shell, sql, yaml-json, docker, kicad-dsl, spice, lua-upy)
4. **Technical** (embedded, stm32, iot, freecad, platformio, power, emc, dsp, spice-sim, electronics, kicad-pcb)
5. **Applications** (web-frontend, web-backend, music-audio, devops, llm-orch)
6. **Complements** (math, security)

Full list: `docs/specs/2026-04-15-micro-kiki-design.md`.

## Teachers

Used for distillation, preference pair generation, and the adaptive judge:

- **Mistral-Large-Opus** (123B, Studio) — reasoning, general, deep judge
- **Qwen3.5-122B-A10B Opus-v3** (Studio, same family as base) — primary distill teacher
- **Qwen3.5-35B-A3B Opus** (kxkm-ai) — fast teacher and default judge
- **Devstral-v3/v4** (kxkm-ai) — coding-specific teacher

## Hardware

| Machine | Role |
|---------|------|
| Mac Studio M3 Ultra 512 GB | Teacher serving + primary training + deep judge + ANE pipeline |
| RTX 4090 24 GB (kxkm-ai) | Inference + Unsloth training + fast judge |

## Research foundations

The design is grounded in 2025–2026 published work:

- **MoLoRA** (arxiv 2603.15965) — composable specialization via per-token routing; Qwen3-1.7B + MoLoRA beats Qwen3-8B on reasoning benchmarks
- **OPLoRA** (arxiv 2510.13003) — orthogonal projection prevents catastrophic forgetting across sequential domains
- **LoRA-Null** (arxiv 2503.02659) — null-space initialization preserves pre-trained knowledge
- **Subspace Geometry** (arxiv 2603.02224) — forgetting is governed by gradient-subspace overlap, not rank
- **Aeon** (arxiv 2601.15311) — neuro-symbolic memory palace for long-horizon agents
- **CAMP** (arxiv 2604.00085) — evidence-based arbitration beats majority voting
- **Catfish Agent** (arxiv 2505.21503) — structured dissent disrupts silent consensus
- **KnowBias** (arxiv 2601.21864) — neuron-level debiasing via targeted fine-tuning
- **RBD** (arxiv 2505.17100) — runtime reasoning-based bias detector
- **Temporal limits** (arxiv 2601.10132) — why 4B LLM is naze at quantitative forecasting, tools layer preferred

## Structure

> **Note:** this tree represents the **target layout** per the 106-step plan. Several directories (`src/memory/`, `src/cognitive/`, `deploy/`) do not exist yet — they are populated as ralph works through the plan. The current state reflects Phase I foundations only.

```
micro-kiki/
├── docs/
│   ├── specs/           # Design documents (frozen, source of truth)
│   ├── research/        # MoE research, benchmarks
│   └── plans/           # Implementation plan (.ralph drives from here)
├── src/
│   ├── base/            # Base model loading, quantization
│   ├── stacks/          # MoE-LoRA trainer + OPLoRA utilities
│   ├── routing/         # Router + dispatcher
│   ├── distill/         # Teacher clients + dataset generator + dedup
│   ├── memory/          # Aeon (atlas, trace, backends)
│   ├── cognitive/       # Argument extractor, judge, catfish, RBD, bias probe
│   ├── eval/            # Per-stack + forgetting + full suite
│   └── serving/         # vLLM / mlx-lm / ANE pipeline
├── configs/             # YAML per stack + meta-intents + judge config
├── data/                # Gitignored: raw + distilled + bias pairs
├── scripts/             # Orchestrators + one-shot utilities
├── deploy/              # systemd units, launchd plists
├── .ralph/              # Ralph loop (prd.json, CLAUDE.md, loop.py)
├── tests/
└── .claude/
    └── plans/           # Source of truth for ralph
```

## Status

14 phases, 107 implementation stories. Tracked in `.ralph/prd.json`.

- [x] Design (2026-04-15) — see `docs/specs/`
- [x] MoE approach research — see `docs/research/`
- [x] Implementation plan (107 stories, 14 phases)
- [ ] Phase I — Foundations (bootstrap base + loader + teacher client + smoke)
- [ ] Phase II — Data pipeline
- [ ] Phase III — First stack (chat-fr E2E)
- [ ] Phase IV — Router v0 + dispatcher (3 stacks)
- [ ] Phase V — Curriculum coding 04–14
- [ ] Phase VI — Technical stacks 15–25
- [ ] Phase VII — Apps + complements 26–32
- [ ] Phase VIII — Aeon memory palace
- [ ] Phase IX — Negotiator
- [ ] Phase X — Anti-bias (KnowBias + RBD)
- [ ] Phase XI — Serving deployment
- [ ] Phase XII — ANE triple pipeline (Mac-only)
- [ ] Phase XIII — Quantum-inspired (HyQuT, QMoE, Quantum-PEFT, classical simulators)
- [ ] Phase XIV — Release v0.2

## Execution

Driven by the ralph loop skill:

```bash
cd /Users/electron/Documents/Projets/micro-kiki
MAX_ITERATIONS=10 uv run .ralph/loop.py
```

Each iteration picks one incomplete story, implements it, runs quality gates, commits, and exits.

## Roadmap

- **v0.1** (shipped in plan history): 32 stacks + router + cognitive layer + serving + ANE
- **v0.2** (scope of current consolidated plan): v0.1 + quantum-inspired techniques (HyQuT hybrid VQC, QMoE routing with classical fallback, Quantum-PEFT adapters — all run on classical simulators by default)
- **v0.3** (planned): temporal context (real-time clock, location, news slice) + future-reasoner (CoT temporal chains, calendar-aware planning). Deferred because LLM 4B underperforms dedicated time-series ML on quantitative forecasting (arxiv 2601.10132); context-injection + tools approach is more appropriate than new stacks.

## Branches

- `main` — public v0.2 core (classical + quantum-inspired, runs on CPU/GPU simulators)
- `quantum` — public hybrid classical/QPU staging area (Qiskit, PennyLane, AWS Braket, IBM Runtime); every experiment must have a classical-simulator fallback
- Private QPU-only research lives at [`electron-rare/micro-kiki-quantum`](https://github.com/electron-rare/micro-kiki-quantum) (PRIVATE, $250/experiment budget, no classical fallback requirement)

## License

Apache 2.0. Base model (Qwen3.5-4B) is also Apache 2.0.

## Related

Part of the KIKI family:
- [KIKI-Mac_tunner](https://github.com/L-electron-Rare/KIKI-Mac_tunner) — MLX fine-tuning toolkit
- [KIKI-models-tuning](https://github.com/L-electron-Rare/KIKI-models-tuning) — Unsloth/LoRA training registry
- [kiki-forge](https://github.com/L-electron-Rare/kiki-forge) — multi-compute LLM training pipeline
