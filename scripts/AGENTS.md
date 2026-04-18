<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# scripts

## Purpose
Operator-facing command-line drivers for the full micro-kiki lifecycle: downloading the base model, dataset curation & distillation, per-stack training (MLX on Mac, small GPU variants on 4090), DPO/GRPO refinement on niche domains, router training, evaluation benches, forgetting checks, and POC pipelines. These scripts wrap `src/*` modules and are meant to be run with `uv run python scripts/<name>.py`. Subdirectory `micro_kiki/` houses the legacy / research-oriented shell + Python pipeline that predates the MLX pivot.

## Key Files
| File | Description |
|------|-------------|
| `download_base.py` | Fetch `Qwen/Qwen3.5-35B-A3B` (or fallback) from HF |
| `validate_domains.py` | CI gate: 32-domain list consistency across `configs/micro_kiki/domains.yaml`, `brainstacks.yaml`, and `configs/mlx-per-domain/*.yaml` |
| `validate_rank_schema.py` | CI gate: LoRA `rank ∈ {4,8,12,16,32}` and `alpha == 2 × rank` for every per-domain MLX config |
| `validate_curriculum_order.py` | CI gate: foundations (rank 32) precede every niche in `brainstacks.yaml` curriculum |
| `validate_no_pre_pivot.py` | CI gate: fails if `src/**/*.py` contains pre-pivot identifiers (`Qwen3.5-4B`, `Qwen3-4B`, `[0.0] * 32`) |
| `measure_forgetting.py` | OPLoRA phase-1a CLI: per-layer cos-angle between two adapter stacks (angle-only, informational). See `docs/training/forgetting-gate.md` |
| `train_stack.py` | Per-stack trainer (standard LoRA, attention-only) |
| `train_niches_mlxtune.py` | MLX per-domain niche trainer (ranks 4-16) |
| `train_dpo_niches.py`, `train_grpo_niches.py` | Preference-tuning on niche domains |
| `train_router.py`, `train_vqc_router.py` | Sigmoid meta-router trainers |
| `train_embeddings.py`, `train_forgetting_gate.py`, `train_qtha_stack.py` | Auxiliary trainers |
| `distill_domain.py` | Teacher distillation via Qwen3-Coder-480B local MLX |
| `generate_dataset_codex.py`, `generate_dataset_mcp.py`, `generate_bias_pairs.py`, `generate_dpo_pairs.py`, `expand_prompts_x25.py` | Synthetic data generators |
| `merge_datasets.py`, `merge_all_sources.py`, `dataset_quality_filter.py`, `curate_bias_dataset.py`, `scrape_stackexchange.py`, `import_kiki_datasets.py`, `import_kiki_datasets.sh` | Dataset curation |
| `eval_aeon.py`, `eval_base_knowbias.py`, `eval_niche_vs_base.py`, `group_eval.py`, `energy_bench.py`, `benchmark_base_vs_lora.py`, `benchmark_quantum_router.py` | Evaluation benches |
| `run_full_eval.sh`, `run_forgetting.sh` | Shell orchestrators |
| `smoke_spikingbrain.py` | Smoke flow (neuroscience branch) |
| `poc_micro_kiki.py`, `poc_pipeline.py`, `poc_pipeline_v2.py` | End-to-end POCs (v2 = VQC router + negotiator + 10 domains) |
| `aeon_compress_daemon.py` | Long-running compress loop (systemd unit available) |
| `release_hf.py` | Push adapters / merged model to HuggingFace |
| `convert_spikingkiki_35b.py`, `probe_spikingbrain_hf.py`, `quantize_spikingbrain.py`, `fork_qwen_diffattn.py` | v0.3 neuroscience branch tooling |
| `orchestrate_remote.sh`, `setup_neuro_env.sh` | Remote orchestration & environment bootstrap |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `micro_kiki/` | Legacy / research pipeline (pre-MLX pivot): `classify_domains.py`, `deduplicate.py`, `eval_stack.py`, `moe_lora.py`, `null_space.py`, `residual_boost.py`, `split_domains.py`, `pipeline_data.sh`, `train_all_stacks.sh` |
| `legacy/` | Archived pre-pivot drivers (Qwen3.5-4B era + GPU prototyping): `train_stack02.py`, `train_stack03.py`, `train_stack_kxkm.py`, `train_micro_kiki_v3_gpu.py`, `train_router_v0.py`, `train_router_kxkm.py`, `distill_fast.py`, `e2e_final.py`, `smoke_e2e.py`, `smoke_test_e2e.py`, `run_pipeline.sh`, `run_eval_stack01{,_fast}.py`, `run_eval_v2.py`, `run_eval_mini.py`, `run_eval_3.py`. Kept for reference only — NOT on the 35B-A3B path |

## For AI Agents

### Working In This Directory
- Scripts are run with `uv run python scripts/<name>.py` from repo root. Absolute paths only.
- Training scripts here are for non-MLX pathways (router training, niche prototyping). Full 35B LoRA training happens in `~/KIKI-Mac_tunner` via `python -m mlx_lm lora --config configs/...`, not here.
- NEVER run `train_stack.py` against the full 35B on kxkm-ai — it will OOM.
- Never resurrect anything in `legacy/` without re-reading `docs/specs/2026-04-16-architecture-pivot-35b.md`; those drivers assume the pre-pivot Qwen3.5-4B base.
- The four `validate_*.py` scripts are CI gates; run them locally after any edit to `configs/micro_kiki/*.yaml` or `configs/mlx-per-domain/*.yaml`.
- `measure_forgetting.py` is informational (exit 0 always); the rollback decision still needs the win-rate half — see `docs/training/forgetting-gate.md`.
- Set `UNSLOTH_COMPILE_DISABLE=1` before any trainer that uses unsloth kernels.
- `release_hf.py` writes to clemsail's HF org — double-check the repo name before pushing.
- After every stack, run `run_forgetting.sh` (rollback if angle < 30 AND win-rate drop > 0.03).
- `aeon_compress_daemon.py` is long-running — use the systemd unit in `deploy/systemd/aeon-compress.service`, not a bare `&`.

### Testing Requirements
Several scripts have counterparts in `tests/`:
- `validate_domains.py` -> `tests/test_validate_domains.py`
- `validate_rank_schema.py` -> `tests/test_validate_rank_schema.py`
- `validate_curriculum_order.py` -> `tests/test_validate_curriculum_order.py`
- `validate_no_pre_pivot.py` -> `tests/test_validate_no_pre_pivot.py`
- `measure_forgetting.py` -> `tests/scripts/test_measure_forgetting.py`
- `generate_dataset_codex.py` -> `test_generate_codex.py`
- `dataset_quality_filter.py` -> `test_quality_filter.py`
- `scrape_stackexchange.py` -> `test_scrape_stackexchange.py`
- `convert_spikingkiki_35b.py` -> `test_convert_spikingkiki.py`

Add a `tests/test_<script>.py` when logic is non-trivial (parsing, filtering, reward shaping). Don't test shell wrappers.

### Common Patterns
- CLI parsing with `argparse`; `--domain <name>` + `--prior-stacks <list>` is the common forgetting-check signature.
- Reads `configs/*.yaml` for stack/meta-intent/capability definitions.
- Writes artifacts into `outputs/` (training), `results/` (eval), `data/` (datasets).
- Shell scripts set `set -euo pipefail` and source the correct venv (`~/KIKI-Mac_tunner/.venv` on Mac, local `.venv` on kxkm-ai).

## Dependencies

### Internal
Uses nearly every `src/*` subpackage: `src.base.loader`, `src.distill.*`, `src.eval.forgetting`, `src.routing.*`, `src.cognitive.*`, `src.stacks.*`, `src.memory.*`, `src.spiking.*`.

### External
Depends on `train`, `mlx`, `serve`, and `agentic` extras depending on the script. Shell scripts also call `python -m mlx_lm lora` (requires `mlx-lm>=0.30`).

<!-- MANUAL: -->
