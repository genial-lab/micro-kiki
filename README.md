# micro-kiki

A 32-domain expert system built on Qwen3.5-35B-A3B (native MoE, 256 experts, 3B active per token) with standard LoRA stacks, cognitive layer (memory palace, negotiator, anti-bias), and dual-machine serving. Trains on Mac Studio M3 Ultra 512 GB via MLX.

## What

Five tightly integrated layers that turn a native MoE base into a specialist team:

1. **Base**: Qwen3.5-35B-A3B (Apache 2.0, 262K ctx, 256 MoE experts, 3B active/token, native thinking mode). The model is already a MoE — no custom MoE-LoRA needed.
2. **10 niche LoRA stacks**: one per hardware/EDA domain where the base model has demonstrable gaps. Rank 16 (rank 8 for high-ratio domains), targeting q/k/v/o attention projections only. General-purpose domains (French, Python, TypeScript, etc.) are served by base model passthrough — fine-tuning those caused overfitting within 300 iterations.
3. **Multi-model router**: 11-output domain classifier (10 niche domains + 1 passthrough) with confidence threshold 0.65 + multi-model tier routing (35B+LoRA / 35B passthrough / 480B teacher / devstral-v3)
4. **Cognitive layer**:
   - **Aeon memory palace** (Atlas SIMD index + Trace neuro-symbolic graph) — persistent, spatial, temporal-aware memory
   - **Negotiator** (CAMP arbitration + Catfish dissent) — resolves conflicts between active stacks with adaptive judge (Qwen3.5-35B fast / Mistral-Large deep)
   - **KnowBias + RBD anti-bias** — post-hoc neuron-level debiasing + RBD runtime detector
5. **Serving**: MLX primary (Mac Studio BF16), vLLM Q4 inference (kxkm-ai RTX 4090)
6. **SNN conversion pipeline**: LAS (Latency-Aware Spiking) conversion of base and niche-adapted models to spiking neural networks (SpikingKiki-27B, SpikingKiki-35B, SpikingKiki-122B)

## Architecture

```
  user prompt
      ↓
  [Dispatcher]      → 7 meta-intents (training-free, zero latency)
      ↓
  [Aeon recall]     → inject top memories into context
      ↓
  [Domain router]   → 11-output classifier
      ↓                  ↓
  [confidence≥0.65] → [Base + niche stack(s)] → candidate response
  [confidence<0.65] → [Base passthrough]      → response (22 dropped domains)
      ↓
  [Negotiator]      → CAMP arbitration (35B) / Mistral-Large if deep needed
      ↓
  [Anti-bias]       → RBD flag → DeFrame re-gen if biased
      ↓
  [Aeon write]      → persist the turn
      ↓
  response
```

## Domains (10 niche — hardware/EDA only)

These are the 10 domains where Qwen3.5-35B-A3B has measurable capability gaps.
All other domains (French, Python, TypeScript, math, etc.) are served by base passthrough.

| # | Domain | Est. examples | Notes |
|---|--------|---------------|-------|
| 01 | `kicad-dsl` | ~4,625 | KIKI-Mac_tunner + mascarade-kicad |
| 02 | `spice` | ~5,766 | KIKI-Mac_tunner + mascarade-spice |
| 03 | `emc` | ~5,053 | KIKI-Mac_tunner + mascarade-emc |
| 04 | `stm32` | ~2,723 | Marginal — early stopping monitored |
| 05 | `embedded` | ~13,826 | 3 source streams — strongest case |
| 06 | `freecad` | ~1,500* | Needs teacher augment, rank 8 |
| 07 | `platformio` | ~1,500* | Needs teacher augment, rank 8 |
| 08 | `power` | ~4,505 | KIKI-Mac_tunner + mascarade-power |
| 09 | `dsp` | ~4,113 | KIKI-Mac_tunner + mascarade-dsp |
| 10 | `electronics` | ~1,900 | High overfitting risk, rank 8 |

*Post-augmentation target. Raw counts: 219 (freecad), 223 (platformio).

Rationale for scope reduction from 32 to 10: see `docs/specs/2026-04-16-reorientation-rationale.md`.

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

7 phases, 40 implementation stories. Tracked in `.ralph/prd.json`. **7/40 done (18%)** — carried over from v0.2 work.

- [x] Design + architecture pivot (2026-04-15/16) — see `docs/specs/`
- [x] Reorientation: 32 → 10 niche stacks (2026-04-16) — see `docs/specs/2026-04-16-reorientation-rationale.md`
- [ ] Phase 1 — Validation (benchmark base, merge datasets, confirm 22 known domains)
- [ ] Phase 2 — Niche Training (10 stacks + cross-stack forgetting check + eval)
- [x] Phase 3 — Router + Multi-model (11-output router + multi-model routing — code done)
- [ ] Phase 4 — Cognitive Layer (Aeon, Negotiator, anti-bias integration tests)
- [ ] Phase 5 — SNN Conversions (SpikingKiki-27B, 35B, 122B + energy benchmark)
- [x] Phase 6 — Serving + Deploy (MLX + vLLM + service units — code done)
- [x] Phase 7 — Release (config freeze + model card + HF publish — templates done)

### Bottleneck

10 sequential LoRA training runs. Estimated: ~45 min/stack × 10 = **~7.5 h** of compute on Mac Studio (BF16, 3-phase curriculum). SNN conversion adds ~70–170 h for SpikingKiki models (parallelizable with training).

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
- **v0.2** (archived): 35B-A3B base, MLX training, local 480B teacher — superseded by reorientation
- **v0.3** (current): 10 niche stacks + 11-output router + multi-model routing + SNN conversion pipeline
- **v0.4** (planned): temporal context + future-reasoner

## License

Apache 2.0. Base model (Qwen3.5-35B-A3B) is also Apache 2.0.

## Related

Part of the KIKI family:
- [KIKI-Mac_tunner](https://github.com/L-electron-Rare/KIKI-Mac_tunner) — MLX fine-tuning toolkit (training pipeline)
- [KIKI-models-tuning](https://github.com/L-electron-Rare/KIKI-models-tuning) — Unsloth/LoRA training registry
- [kiki-forge](https://github.com/L-electron-Rare/kiki-forge) — multi-compute LLM training pipeline
