<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# micro-kiki

## Purpose
micro-kiki is a 34-domain expert system built on the Qwen3.6-35B-A3B MoE base (256 experts, 3B active per token) via standard LoRA adapters covering 17 module kinds per layer (attention q/k/v/o + `linear_attn` GLA hybrid + MoE routers `mlp.gate`/`mlp.shared_expert_gate` + `mlp.shared_expert.*` + `mlp.switch_mlp.*`). On top of the adapter layer sits a cognitive stack: sigmoid meta-router + YAML dispatcher, Aeon memory palace (Atlas SIMD + Trace graph), CAMP/Catfish negotiator, and KnowBias+RBD anti-bias. Training runs sequentially per domain via MLX on a Mac Studio M3 Ultra 512 GB; inference runs on MLX (Mac) or vLLM Q4 (kxkm-ai RTX 4090). Distillation uses Qwen3-Coder-480B-A35B MLX 4bit as a local teacher. Version 0.2.0-dev.

## Architecture

```
query
  |
  v
[routing/]          --> 35-output sigmoid (34 niche domains + 1 base; capabilities served separately)
  |                    thresholds: 0.12 general, 0.20 chat-mode, max 4 stacks
  v
[orchestrator/]     --> dispatcher maps router output to 7 meta-intents
  |
  v
[stacks/] LoRA      --> 17 module kinds/layer: linear_attn.* + self_attn.{q,k,v,o}_proj
  |                    + mlp.gate + mlp.shared_expert_gate + mlp.shared_expert.* + mlp.switch_mlp.*
  |                    rank tiers {4,8,12,16,32}; MLX scale = 20.0
  v
[cognitive/]        --> negotiator (CAMP+Catfish), KnowBias, RBD, forgetting_gate
  |
  v
[memory/] Aeon      --> Atlas SIMD ANN + Trace graph; native or Qdrant/Neo4j
  |
  v
[serving/]          --> MLX primary (Mac, adapter-swap via mlx_client host-map), vLLM Q4 (kxkm-ai)
```

## Key Files
| File | Description |
|------|-------------|
| `CLAUDE.md` | Authoritative conventions (base model, adapter strategy, commits, Do/Don't) |
| `MODEL_CARD.md`, `MODEL_CARD-v0.3.md` | Published model cards |
| `COOKBOOK.md` | End-user recipes for training / inference |
| `README.md` | Short project handle (content cached line 1 only) |
| `BRANCH-neuroscience.md` | v0.3 neuroscience branch notes (SpikingBrain / LIF / LAS) |
| `MIGRATION.md` | v0.1 -> v0.2 -> v0.3 migration guide |
| `pyproject.toml` | hatchling build, Python 3.11+, optional extras: `train`, `mlx`, `serve`, `agentic`, `dev` |
| `VERSION` | `0.2.0-dev` |

## Subdirectories
| Directory | Purpose | AGENTS.md |
|-----------|---------|-----------|
| `src/` | Python package (`src/*` -> installed top-level via hatch) | yes |
| `tests/` | pytest suite, conftest fixtures | yes |
| `scripts/` | Training drivers, distillation, eval, pipeline helpers | yes |
| `configs/` | Per-stack YAML, MLX curricula, meta-intent & capability maps | yes |
| `docs/` | Specs, plans, research, superpowers, training READMEs | yes |
| `research/` | Exploratory work (ANE hybrid pipeline) | yes |
| `examples/` | Minimal usage snippets (chat, memory, bias, forgetting) | yes |
| `deploy/` | launchd (Mac) + systemd (Linux) units | yes |
| `docker/` | vllm Dockerfile | yes |
| `hardware/` | KiCad schematics (STM32H743 bootloader, SPI bus) | yes |
| `data/` | Datasets and distilled JSONL — data-only, no AGENTS.md |
| `outputs/` | Training outputs (adapters, checkpoints) — data-only, no AGENTS.md |
| `output/` | Legacy output dir — data-only, no AGENTS.md |
| `results/` | Eval result JSON — data-only, no AGENTS.md. `results/legacy/` holds pre-pivot eval artifacts (`stack-01-eval*.json`, `e2e-smoke.json`) |
| `scripts/legacy/` | Archived pre-pivot drivers (Qwen3.5-4B era + GPU prototyping). Reference-only, NOT on the 35B-A3B path |

## For AI Agents

### Working In This Repository
- READ `CLAUDE.md` first; it overrides defaults (base model, adapter rule, commit format).
- Enforce hard rules:
  - LoRA tunes the 17-module surface (attention + MoE routers + shared_expert + switch_mlp) as set by `mlx_lm lora`. The prior "never MoE FFN" rule was superseded 2026-04-18 after reading the real `adapter_config.json` and verifying empirical forgetting (chat-fr ↔ reasoning mean 79.4°, all modules > 30°).
  - NEVER use QLoRA / BitsAndBytes on 35B-A3B (MoE mixed-precision kernels break).
  - NEVER train on kxkm-ai (RTX 4090 24 GB cannot hold 35B BF16 LoRA).
  - NEVER drop below Q4 quantization at inference.
  - NEVER route > 4 stacks simultaneously.
  - NEVER train stacks in parallel — sequential curriculum only.
  - Run forgetting gate (health + angle + win-rate) after EVERY stack; rollback if angle < 30 AND win-rate drop > 0.03. `scripts/post_train_gate.py` is the one-shot entry; exit codes 0/1/2/3 = pass / angle-fail / winrate-fail / health-fail.
  - Never deploy pre-pivot MoE-LoRA adapters (stacks-v3-r16) — all have `lora_B = 0`; they live in `scripts/legacy/` for archival only.
- `UNSLOTH_COMPILE_DISABLE=1` before any training on the Mac Studio.
- Python 3.11+, ruff + black, line length 100, loguru for logging (no `print()`).
- Commits: `feat|fix|docs(<area>): <imperative>`, subject <= 50 chars, no `Co-Authored-By` trailer (pre-commit hook rejects it).

### CI invariants (validators)
Five standalone scripts act as fail-fast gates over configs, source, and trained adapters. `.github/workflows/validators.yml` runs them in two parallel jobs: `config-invariants` (four validators + validator tests) and `forgetting-tests` (OPLoRA forgetting-measurement tests, CPU-torch wheel). Run locally before pushing:
- `scripts/validate_domains.py` — 34-domain list must match across `configs/micro_kiki/domains.yaml`, `configs/micro_kiki/brainstacks.yaml`, and `configs/mlx-per-domain/*.yaml`.
- `scripts/validate_rank_schema.py` — LoRA `rank ∈ {4, 8, 12, 16, 32}` and `alpha == 2 × rank` per per-domain config.
- `scripts/validate_curriculum_order.py` — foundations (rank 32) precede every niche in the curriculum.
- `scripts/validate_no_pre_pivot.py` — no `Qwen3.5-4B` / `Qwen3-4B` / `[0.0] * 32` identifier leaks into `src/**/*.py` (docs are exempt).
- `scripts/validate_adapter_health.py <adapter.safetensors>` — all `lora_B` matrices non-zero (catches the pre-pivot MoE-LoRA dead-weight bug).

Unit tests for the validators live in `tests/test_validate_*.py`.

### Forgetting-gate tooling
Pipeline entries, layered thinnest → thickest:
- `src/eval/forgetting.py` — core: `measure_forgetting_signal`, `ForgettingReport`, `apply_and_gate_detailed`, `apply_per_module_gate` (ignores `mlp.shared_expert_gate` canary by default).
- `src/eval/scorers.py` — `containment_score` + `JudgeScorer` (wraps `StackEvaluator` judge).
- `src/serving/mlx_client.py` — async httpx client with `MLX_ADAPTER_HOST_MAP` env-var routing for the dual-server real-adapter flow.
- `scripts/measure_forgetting.py` — angle-only (partial) or full gate (angle + win-rate) with `--generate-fn-module` + `--scorer-module`.
- `scripts/run_forgetting_sweep.py` — pairwise matrix across a directory of adapters.
- `scripts/sweep_adapter_health.py` — bulk `lora_B` audit.
- `scripts/post_train_gate.py` — one-shot post-training orchestrator; chains health + measure_forgetting; exit codes 0/1/2/3 = pass / angle-fail / winrate-fail / health-fail.
- `scripts/smoke_gate_on_studio.py` — E2E smoke via programmatic `mlx_lm` adapter-weight swap.

Canonical docs: `docs/training/forgetting-gate.md` (gate semantics + CLI) and `docs/training/e2e-smoke-runbook.md` (dual-server operator runbook). Design + roadmap: `.omc/brainstorm-oplora.md`.

### Empirical artefacts
- `results/forgetting-matrix.json` — 5 post-pivot adapters × 20 pairs, all above 30°.
- `results/forgetting-matrix-prepivot.json` — 35 pre-pivot adapters, all 0° (degenerate).
- `results/adapter-health-sweep.json` — 70 adapters on Studio (35/35 post-pivot healthy, 35/35 pre-pivot degenerate).
- `results/smoke-gate.json` — E2E smoke (chat-fr ↔ reasoning mean 79.4°, winrate_drop −0.04, gate PASS).

### Testing Requirements
```bash
uv run python -m pytest                  # full
uv run python -m pytest tests/routing    # targeted
uv run python -m pytest -m "not integration"   # skip real-model / SSH tests
```
`asyncio_mode = auto` is set in pyproject; integration tests marked `@pytest.mark.integration` are opt-in.

### Common Patterns
- `from __future__ import annotations` at the top of every module.
- Explicit device placement; never rely on implicit CUDA.
- Dataclass configs with `frozen=True` or Pydantic v2 `BaseModel`.
- YAML configs loaded through `src.routing.dispatcher.load_intent_mapping` or equivalents.
- Adapter outputs live under `~/KIKI-Mac_tunner/output/micro-kiki/stack-NN-<domain>/`.

## Dependencies

### External (core)
`httpx>=0.27`, `pyyaml>=6.0`, `numpy>=1.26`, `huggingface-hub>=0.28`.

### Optional extras
- `train`: torch, transformers, peft, trl, accelerate, bitsandbytes, datasets
- `mlx`: mlx>=0.26, mlx-lm>=0.30
- `serve`: vllm, fastapi, uvicorn
- `agentic`: beautifulsoup4
- `dev`: pytest, pytest-asyncio

### External services / hardware
- Mac Studio M3 Ultra 512 GB — training + MLX serving + teacher (480B)
- kxkm-ai RTX 4090 24 GB — Q4 inference only
- Tower — Qdrant + Neo4j Aeon backends, Piper TTS
- HuggingFace: `Qwen/Qwen3.6-35B-A3B` (base), `Qwen/Qwen3-Coder-480B-A35B-Instruct` (teacher)

<!-- MANUAL: -->
