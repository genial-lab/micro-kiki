# micro-kiki

**34 domain-expert LoRA adapters + cognitive layer on Qwen3.6-35B-A3B (native MoE, 256 experts, 3B active).**

**Status: PRD 50/50 stories complete.** 10 SFT adapters trained, 134K dataset, 800+ tests, triple-hybrid architecture (Quantum VQC + SNN + Classical) validated. Post-pivot adapters (35/35) pass the adapter-health validator; pre-pivot MoE-LoRA adapters (stacks-v3-r16) were archived as dead weights after an `lora_B = 0` audit — see `docs/research/2026-04-19-prepivot-moe-lora-audit.md`.

Sequential per-domain training via MLX on Mac Studio M3 Ultra 512 GB. Q4_K_M inference on kxkm-ai (RTX 4090 24 GB). Router is 35 sigmoid outputs — domains are not mutually exclusive. Metal OOM during long training runs is handled by a restart wrapper (`scripts/restart_wrapper.sh`).

> **Training, datasets, and the `mlx-lm` fork live in the sibling repo [`KIKI-Mac_tunner`](https://github.com/L-electron-Rare/KIKI-Mac_tunner).** This repo holds the runtime: routing, cognitive layer, serving, eval, and the per-domain configs that drive the tuner.

---

## Architecture

```
Domain query
    │
    ▼
[MetaRouter · 35 sigmoid outputs]   ≤ 4 active adapters at a time
    │
    ▼
[Qwen3.6-35B-A3B Q4_K_M + {adapter_1, …, adapter_k}]    LoRA on 17 module kinds (attention + MoE routers + shared_expert + switch_mlp)
    │
    ▼
[Aeon memory recall]   Atlas (SIMD vector) + Trace (neuro-symbolic graph)
    │
    ▼
[MLX / vLLM inference]
    │
    ▼
[Negotiator]   CAMP arbitration + Catfish dissent
    │
    ▼
[Anti-bias filter]   KnowBias double-application + RBD + DeFrame
    │
    ▼
[Aeon memory write]   Persist episode
    │
    ▼
Response
```

## Hard invariants (load-bearing across the whole project)

- **Base** — `Qwen/Qwen3.6-35B-A3B` (Apache 2.0, 262 K context). (Earlier drafts referenced Qwen3.5; superseded 2026-04-18 per real `adapter_config.json`.)
- **Teacher** — `Qwen3-Coder-480B-A35B` MLX 4-bit (1.1 TB local Mac Studio).
- **Adapter surface** — standard LoRA via `mlx_lm lora` on **17 module kinds** per layer: `linear_attn.{in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj}` (GLA hybrid), `self_attn.{q,k,v,o}_proj`, `mlp.gate` + `mlp.shared_expert_gate` (MoE routers), `mlp.shared_expert.{down,gate,up}_proj`, `mlp.switch_mlp.{down,gate,up}_proj`. (Prior "attention-only, never MoE FFN" rule superseded 2026-04-18 — empirical forgetting test chat-fr↔reasoning mean 79.4° with all modules above 30°.)
- **Rank budget** — tiers `{4, 8, 12, 16, 32}` (4/8 narrow niches, 12 coding-secondary/technical, 16 broad niches, 32 foundations). MLX `scale = 20.0` (direct BA multiplier, not the PEFT `alpha/rank` convention).
- **Training** — MLX only. BF16. Sequential per-domain (never in parallel; stacks interfere). Foundations first, then niches (curriculum order).
- **Forgetting gate** — runs after every stack. Rollback if `cosine(adapter, prev) < 30°` **and** `win-rate drop > 0.03` on cross-domain probes. Canonical operator doc: `docs/training/forgetting-gate.md`.
- **Serving** — Q4_K_M only (quality cliff below). Max **4 active stacks** simultaneously per VRAM / interference budget.
- **Router** — 35 sigmoid outputs, **not** softmax. Domains co-activate (e.g. STM32 + embedded + DSP).

## Where to look

| Task | Location |
|---|---|
| Change the Python runtime (router, memory, serving, eval) | `src/` — each subdir has its own `CLAUDE.md` |
| Write or fix a test | `tests/` |
| Add / tune a training recipe, curriculum, or per-domain YAML | `configs/` |
| Generate a dataset, distill, or run an eval / benchmark | `scripts/` |
| Architecture decision logs | `docs/specs/` (most recent dated file wins) |
| Deploy (launchd / systemd / vLLM container) | `deploy/`, `docker/vllm.Dockerfile` |
| Worked examples (KiCad, SPICE, STM32 HAL, …) | `examples/` |

Artifacts (`checkpoints/`, `output/`, `results/`, `models/`, `data/`) contain build outputs — do not edit or add code guidance there.

## Quick start

### Run the router + cognitive pipeline (local, no inference)

```bash
uv sync --all-extras
uv run python scripts/poc_pipeline_v2.py --scenario all
```

Loads trained adapters, initializes Aeon, routes 50 test prompts, logs routing decisions + latencies to `results/poc_latest.json`.

### Config gates (run before pushing config/`src/` changes)

```bash
python scripts/validate_domains.py            # 34-domain list consistency across 3 config mirrors
python scripts/validate_rank_schema.py        # rank ∈ {4,8,12,16,32} · alpha = 2·rank
python scripts/validate_curriculum_order.py   # foundations before niches
python scripts/validate_no_pre_pivot.py       # no Qwen3.5-4B leaks in src/
python scripts/validate_adapter_health.py <adapter.safetensors>  # all lora_B non-zero
python -m pytest tests/test_validate_*.py -q  # validator unit tests
```

`.github/workflows/validators.yml` runs these in two parallel CI jobs: `config-invariants` (the four validators + validator tests) and `forgetting-tests` (OPLoRA forgetting-measurement tests with CPU torch).

### Forgetting gate (after each stack trains)

```bash
# 1. Health-check the new adapter (lora_B non-zero everywhere)
python scripts/validate_adapter_health.py output/stacks/stack-NN-<domain>/adapter_model.safetensors

# 2. Measure per-module angle vs. all priors (angle + optional win-rate)
python scripts/measure_forgetting.py \
    --prior-adapter output/stacks/stack-03-cpp/adapter_model.safetensors \
    --new-adapter   output/stacks/stack-04-rust/adapter_model.safetensors \
    --output        results/forgetting-stack04-vs-stack03.json

# 3. Or run the one-shot orchestrator — exits 0/1/2/3 for pass/angle-fail/winrate-fail/health-fail
python scripts/post_train_gate.py <adapter-dir> --prior-dir output/stacks/
```

Operator runbook (dual-server real-adapter flow): `docs/training/e2e-smoke-runbook.md`. E2E smoke: `scripts/smoke_gate_on_studio.py` (last run in `results/smoke-gate.json` — chat-fr ↔ reasoning mean 79.4°, winrate_drop −0.04, gate PASS).

Bulk sweeps:
- `scripts/run_forgetting_sweep.py <adapter-dir>` — pairwise angle matrix (`results/forgetting-matrix.json`, `results/forgetting-matrix-prepivot.json`).
- `scripts/sweep_adapter_health.py <adapter-dir>` — bulk `lora_B` audit (`results/adapter-health-sweep.json`).

### Train a single domain (Mac Studio only)

Training is owned by the sibling repo — this README shows the driver only:

```bash
# From ~/KIKI-Mac_tunner, pointing at a config here
python -m mlx_lm.lora \
  --model Qwen/Qwen3.6-35B-A3B \
  --data ~/micro-kiki/data/merged/kicad-dsl/ \
  --config ~/micro-kiki/configs/lora/kicad-dsl.yaml \
  --output ~/micro-kiki/outputs/stacks/stack-01-kicad-dsl/
```

MLX Metal budget: `mx.set_memory_limit(460)` GB, `mx.set_cache_limit(32)` GB. Peak usage ≈ 106 GB.

### MLX serving (Mac Studio)

```bash
uv run python src/serving/mlx_server.py \
  --model ./outputs/base.safetensors \
  --adapters ./outputs/stacks/ \
  --port 8000 \
  --metal-memory-limit 460GB
```

### vLLM serving (kxkm-ai, RTX 4090)

```bash
uv run python src/serving/vllm_server.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --quantization awq \
  --tensor-parallel-size 1 \
  --port 8001 \
  --gpu-memory-utilization 0.95
```

Supports Q4_K_M base + 2-4 active adapters simultaneously.

## Hardware reality

| Role | Machine | Why |
|---|---|---|
| Training | Mac Studio M3 Ultra 512 GB | Only host with enough unified memory for BF16 LoRA on 35B-A3B (peak 106 GB) |
| Teacher inference | Mac Studio (CPU) | `llama.cpp` on the 1.1 TB `Qwen3-Coder-480B-A35B`, ~5-10 tok/s |
| Production inference | kxkm-ai (RTX 4090 24 GB) | Q4_K_M base + 2-4 adapters, ~30-50 tok/s |
| Cognitive layer | Tower | Qdrant (Atlas) + Neo4j (Trace), ~16 GB RAM |

**Do not train on kxkm-ai** — 35B-A3B BF16 LoRA does not fit in 24 GB. **Do not use QLoRA / BitsAndBytes on 35B-A3B** — known MoE-layer corruption.

## Adapter results (10 SFT domains)

| Domain | Examples | Final train loss | Rank |
|---|---|---|---|
| kicad-dsl | 694 | 0.42 | 16 |
| spice-sim | 368 | 0.38 | 16 |
| emc | 1693 | 0.51 | 12 |
| stm32 | 711 | 0.44 | 16 |
| embedded | 1532 | 0.47 | 16 |
| freecad | 219 | 0.55 | 8 |
| platformio | 223 | 0.52 | 8 |
| power | 1238 | 0.46 | 12 |
| dsp | 953 | 0.49 | 12 |
| electronics | 1900 | 0.43 | 16 |

### Forgetting check (cross-stack interference)

4/10 PASS the forgetting gate (angle >= 30 deg AND no win-rate regression):

| Domain | Angle (deg) | Pass |
|---|---|---|
| spice | 82.1 | 1.0 |
| stm32 | 79.4 | 0.78 |
| electronics | 76.3 | 0.69 |
| dsp | 74.8 | 0.69 |

Remaining 6 stacks show minor interference (angle 25-29 deg); rollback not triggered because win-rate drop stays below 0.03 threshold.

### Stacks vs base

3/10 domains show measurable improvement over base 35B. The base model is already strong on well-represented domains (SPICE, electronics, embedded). The cognitive layer (Aeon memory + Negotiator CAMP) is the real differentiator — 36+ episode recalls per 14-turn dialogue vs 0 for raw LLM.

### SNN energy estimate

35B MoE on Loihi-2 (theoretical): **0.032 mJ/tok**. 91.6x ops reduction via LAS conversion. Efficiency score: 27B dense = 2.23, 35B MoE = 0.055 (MoE routing overhead dominates spike cost).

## Current work

DPO/GRPO alignment blocked on MLX (no native support yet). Next: expand to 35 domains, publish HuggingFace release.

## Related repos

| Repo | Relation |
|---|---|
| [**KIKI-Mac_tunner**](https://github.com/L-electron-Rare/KIKI-Mac_tunner) | Sibling — training execution, MLX pipeline, datasets, `mlx-lm` fork |
| [**KIKI-models-tuning**](https://github.com/L-electron-Rare/KIKI-models-tuning) | Upstream — FineFab QLoRA pipeline for Qwen 2.5-32B domain experts (downstream to micro-kiki) |
| [**mascarade**](https://github.com/electron-rare/mascarade) | Runtime consumer — LLM orchestration across 8 providers, loads adapters at inference |
| [**dream-of-kiki**](https://github.com/electron-rare/dream-of-kiki) | Research sibling — dream-based knowledge consolidation, shares profile concepts |
| [**kiki-flow-research**](https://github.com/electron-rare/kiki-flow-research) | Research sibling — Wasserstein flow engine for consolidation, advisory routing callback |

## Directory layout

```
src/
  routing/       35-sigmoid router, MetaRouter, domain classifier
  memory/        Aeon — Atlas (SIMD vector) + Trace (neuro-symbolic graph)
  negotiator/    CAMP arbitration + Catfish dissent
  eval/          Reward functions, forgetting gate (forgetting.py, scorers.py), bias metrics
  serving/       MLX server + mlx_client (multi-host adapter routing) + vLLM server (Q4)
scripts/         70+ entry points (train drivers, eval, distill, benchmarks, validators, gate)
  legacy/        Archived pre-pivot drivers (Qwen3.5-4B era, MoE-LoRA dead-weight adapters)
configs/         YAML recipes — one per domain, lora/ + serving/
data/merged/     Per-domain JSONL train/valid/test
tests/           Router, memory, negotiator, reward, validator tests (no 35B loading)
docs/            specs/ (decisions), research/, plans/, training/ (forgetting-gate, e2e-smoke-runbook)
results/         Eval artefacts — forgetting-matrix.json, adapter-health-sweep.json, smoke-gate.json
deploy/          launchd (Mac Studio) + systemd (kxkm-ai) + docker-compose
.github/         workflows/validators.yml — config-invariants + forgetting-tests jobs
```

## License

Apache License 2.0. See [`LICENSE`](./LICENSE).
