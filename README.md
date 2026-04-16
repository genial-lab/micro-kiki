# micro-kiki

A 32-domain expert system built on Qwen3.5-35B-A3B (native MoE, 256 experts, 3B active per token) with standard LoRA stacks, cognitive layer (memory palace, negotiator, anti-bias), and dual-machine serving. Trains on Mac Studio M3 Ultra 512 GB via MLX.

## What

Five tightly integrated layers that turn a native MoE base into a specialist team:

1. **Base**: Qwen3.5-35B-A3B (Apache 2.0, 262K ctx, 256 MoE experts, 3B active/token, native thinking mode). The model is already a MoE — no custom MoE-LoRA needed.
2. **32 standard LoRA stacks**: one per domain, rank 16, targeting q/k/v/o attention projections only. ~74 GB BF16 for training on Mac Studio.
3. **Domain router**: classifier-based adapter selection (max 4 active stacks) + training-free dispatcher (7 meta-intents)
4. **Cognitive layer**:
   - **Aeon memory palace** (Atlas SIMD index + Trace neuro-symbolic graph) — persistent, spatial, temporal-aware memory
   - **Negotiator** (CAMP arbitration + Catfish dissent) — resolves conflicts between active stacks with adaptive judge (Qwen3.5-35B fast / Mistral-Large deep)
   - **KnowBias + RBD anti-bias** — post-hoc neuron-level debiasing (applied twice on the merged model, after all 32 stacks are trained) + RBD runtime detector
5. **Serving**: MLX primary (Mac Studio BF16), vLLM Q4 inference (kxkm-ai RTX 4090)

## Architecture

```
  user prompt
      ↓
  [Dispatcher]    → 7 meta-intents (training-free, zero latency)
      ↓
  [Aeon recall]   → inject top memories into context
      ↓
  [Domain router] → classifier → activate 2-4 stacks
      ↓
  [Base + stacks] → K candidate responses
      ↓
  [Negotiator]    → CAMP arbitration (35B) / Mistral-Large if deep needed
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

## Base model: Qwen3.5-35B-A3B

### Why the pivot from 4B

The 4B base + custom MoE-LoRA approach was replaced after the 2026-04-16 pivot:

- Qwen3.5-35B-A3B is natively MoE (256 experts, only 3B active per token) — custom MoE-LoRA adapters on top are redundant
- Standard LoRA on attention projections is simpler, more reliable, and achieves better specialization
- Mac Studio M3 Ultra 512 GB handles BF16 LoRA training at ~195 GB peak memory
- Teacher is local: Qwen3-Coder-480B-A35B MLX 4bit (already on-machine, no network dependency)

See `docs/specs/2026-04-16-architecture-pivot-35b.md` for full rationale.

## Teachers

Used for distillation and the adaptive judge:

- **Qwen3-Coder-480B-A35B** (MLX 4bit, local Mac Studio) — primary distillation teacher for all domains
- **Mistral-Large-Opus** (123B, Studio) — deep judge
- **Qwen3.5-35B-A3B Opus** (kxkm-ai) — fast judge and secondary teacher

## Training

Stacks are trained via MLX LoRA using the KIKI-Mac_tunner pipeline.

```bash
# Train a stack (from KIKI-Mac_tunner)
./train.sh --config configs/mlx-lm-qwen35-35b-a3b-micro-kiki.yaml

# Run forgetting check after each stack
uv run python src/eval/forgetting.py --stack chat-fr
```

Config: `configs/mlx-lm-qwen35-35b-a3b-micro-kiki.yaml` in KIKI-Mac_tunner.

- LR: 1e-5, batch_size: 2, grad_accumulation: 8, rank: 16, 2000 iters
- Peak memory: ~195 GB, GPU Metal at 100%
- Data: `~/KIKI-Mac_tunner/data/micro-kiki/<domain>/` (63K+ examples across 32 domains)

See `docs/training/README.md` for the full training workflow.

## Hardware

| Machine | Role |
|---------|------|
| Mac Studio M3 Ultra 512 GB | BF16 training (MLX), teacher serving (480B), MLX inference |
| RTX 4090 24 GB (kxkm-ai) | Q4 inference only |
| Tower | Aeon backends (Qdrant, Neo4j), Piper TTS |

## Research foundations

- **OPLoRA** (arxiv 2510.13003) — orthogonal projection prevents catastrophic forgetting across sequential domains
- **LoRA-Null** (arxiv 2503.02659) — null-space initialization preserves pre-trained knowledge
- **Aeon** (arxiv 2601.15311) — neuro-symbolic memory palace for long-horizon agents
- **CAMP** (arxiv 2604.00085) — evidence-based arbitration beats majority voting
- **Catfish Agent** (arxiv 2505.21503) — structured dissent disrupts silent consensus
- **KnowBias** (arxiv 2601.21864) — neuron-level debiasing via targeted fine-tuning
- **RBD** (arxiv 2505.17100) — runtime reasoning-based bias detector

## Structure

```
micro-kiki/
├── docs/
│   ├── specs/           # Design documents (frozen, source of truth)
│   ├── training/        # Training workflow documentation
│   ├── research/        # Research references, benchmarks
│   └── plans/           # Implementation plan (.ralph drives from here)
├── src/
│   ├── base/            # Base model loading, quantization
│   ├── stacks/          # LoRA trainer + OPLoRA utilities
│   ├── routing/         # Router + dispatcher
│   ├── distill/         # Teacher clients + dataset generator + dedup
│   ├── memory/          # Aeon (atlas, trace, backends)
│   ├── cognitive/       # Argument extractor, judge, catfish, RBD, bias probe
│   ├── eval/            # Per-stack + forgetting + full suite
│   └── serving/         # vLLM / mlx-lm pipeline
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

14 phases, 108 implementation stories. Tracked in `.ralph/prd.json`. **40/108 done (37%)**.

- [x] Design (2026-04-15) — see `docs/specs/`
- [x] Architecture pivot to 35B-A3B (2026-04-16) — see `docs/specs/2026-04-16-architecture-pivot-35b.md`
- [x] MoE approach research — see `docs/research/`
- [x] Implementation plan (108 stories, 14 phases)
- [x] Phase I — Foundations (bootstrap base + loader + teacher client + smoke)
- [~] Phase II — Data pipeline (chat-fr distilled, datasets classified + deduped)
- [~] Phase III — First stack (chat-fr training active on Mac Studio)
- [x] Phase IV — Router v0 + dispatcher (3 stacks) — code done, training pending
- [ ] Phase V — Curriculum coding 04–14
- [ ] Phase VI — Technical stacks 15–25
- [ ] Phase VII — Apps + complements 26–32
- [x] Phase VIII — Aeon memory palace (atlas + trace + aeon API + backends)
- [x] Phase IX — Negotiator (judge + catfish + argument extractor)
- [~] Phase X — KnowBias + RBD (code done, bias dataset in progress)
- [x] Phase XI — Serving deployment (vLLM dynamic LoRA + MLX server)
- [~] Phase XII — ANE triple pipeline (stubs present, CoreML conversion pending)
- [x] Phase XIII — Quantum-inspired (classical simulators)
- [~] Phase XIV — E2E acceptance + Release

### Bottleneck

37 sequential GPU training stories (32 stacks + router retrains). Estimated: ~1h/stack × 32 = ~32h of compute on Mac Studio (BF16 LoRA, 2000 iters). Code scaffolding is complete.

## Execution

Driven by the ralph loop skill:

```bash
# Ralph loop
cd /Users/clems/micro-kiki
MAX_ITERATIONS=10 uv run .ralph/loop.py

# Train a specific stack (from KIKI-Mac_tunner)
cd ~/KIKI-Mac_tunner
./train.sh --config configs/mlx-lm-qwen35-35b-a3b-micro-kiki.yaml

# Forgetting check
cd /Users/clems/micro-kiki
uv run python src/eval/forgetting.py --stack <domain>
```

## Roadmap

- **v0.1** (shipped in plan history): 32 stacks + router + cognitive layer + serving
- **v0.2** (current): 35B-A3B base, MLX training, local 480B teacher, quantum-inspired techniques (classical simulators)
- **v0.3** (planned): temporal context + future-reasoner

## License

Apache 2.0. Base model (Qwen3.5-35B-A3B) is also Apache 2.0.

## Related

Part of the KIKI family:
- [KIKI-Mac_tunner](https://github.com/L-electron-Rare/KIKI-Mac_tunner) — MLX fine-tuning toolkit (training pipeline)
- [KIKI-models-tuning](https://github.com/L-electron-Rare/KIKI-models-tuning) — Unsloth/LoRA training registry
- [kiki-forge](https://github.com/L-electron-Rare/kiki-forge) — multi-compute LLM training pipeline
