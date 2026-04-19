<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# scripts/micro_kiki

## Purpose
Core training + data-pipeline scripts for the 35-domain micro_kiki system. These are the scripts that operators actually invoke — `pipeline_data.sh` orchestrates the full data-prep flow end-to-end, `train_stack.py` runs the legacy MoE-LoRA single-domain training loop (null-space projection + residual boosting + forgetting check), `train_all_stacks.sh` sweeps the curriculum, and the rest are focused utilities for classification, deduplication, splitting, evaluation, and CoreML conversion. Post-2026-04-16, MLX-native training is preferred via `python -m mlx_lm lora --config ../../configs/mlx-per-domain/<domain>.yaml`; the MoE-LoRA path here is retained for experiments that exercise the null-space / residual-boost pipeline.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package marker. Docstring still says "32 MoE-LoRA stacks for Qwen3.5-4B" — pre-pivot artifact, safe to update opportunistically. |
| `pipeline_data.sh` | Full data pipeline orchestrator: download → classify → generate-missing → dedupe → split. Flags: `--skip-download`, `--skip-generate`, `--dry-run`, `--teacher <model>`. |
| `download_datasets.sh` | Download public datasets into `data/raw/`. |
| `classify_domains.py` | Classify raw examples into the 35 domains via `configs/micro_kiki/domains.yaml` keyword + regex rules. |
| `classify_parallel.py` | Parallel variant of `classify_domains.py` for large corpora. |
| `generate_missing.py` | Synthetic generation for sparse domains (< target examples) using a teacher model. |
| `deduplicate.py` | Cross-domain deduplication (exact + near-duplicate hashing). |
| `split_domains.py` | Train/valid split per domain (`valid_ratio` from `domains.yaml`). |
| `train_stack.py` | **Core training loop.** Loads `brainstacks.yaml`, freezes base, attaches MoE-LoRA, computes null-space projector from prior stacks, runs SFT (~500 steps), residual-boost round on hard examples, freezes + saves, evaluates all prior domains. Imports MoE-LoRA from the archived `scripts/legacy/moe_lora.py`. |
| `train_all_stacks.sh` | Sequential curriculum sweep over all 35 domains. |
| ~~`moe_lora.py`~~ | **ARCHIVED 2026-04-19** → `scripts/legacy/moe_lora.py`. Dual-mount topology stranded every `lora_b` at zero; see `docs/research/2026-04-19-moe-lora-root-cause.md`. |
| `null_space.py` | Randomized SVD → null-space projector for forgetting prevention (`ns_top_k_dirs: 32`). |
| `residual_boost.py` | Hard-example mining + reweighted training rounds (top-25% loss quantile, 2x weight, 100 boost steps). |
| `eval_stack.py` | Per-stack eval harness (win-rate, angle vs prior). |
| `poc_2stacks.py` | Minimal 2-stack reproduction for pipeline smoke-testing. |
| `convert_08b_coreml.py` | CoreML conversion utility (ANE experiments — see `research/ane-hybrid/` for status). |

## For AI Agents

### Working In This Directory

- **Entry points are the shell scripts** (`pipeline_data.sh`, `train_all_stacks.sh`). Start there before diving into a Python file.
- `train_stack.py` imports from `scripts/micro_kiki/` as a package (it does `sys.path.insert(0, parent)`). Keep `__init__.py` present; don't move modules without updating imports.
- This directory's `train_stack.py` is the **legacy MoE-LoRA** path. For the post-pivot standard-LoRA path, prefer `scripts/train_niches_mlxtune.py` or `python -m mlx_lm lora` directly.
- Shell scripts use `set -euo pipefail` — keep it. Derive paths from `$SCRIPT_DIR` / `$PROJECT_DIR` (see `pipeline_data.sh` for the pattern).
- The 35 domain slugs are load-bearing — any script that iterates domains should source them from `configs/micro_kiki/domains.yaml` or `brainstacks.yaml:curriculum`, never hardcode.
- Don't add GPU-only (CUDA) paths here. This directory is Mac-Studio-first (MLX + Metal). CUDA experiments belong in `scripts/` at the parent level.

### Testing Requirements

- Smoke test: `./pipeline_data.sh --dry-run` should complete without invoking the teacher or writing to disk.
- `train_stack.py` has a `--dry-run` mode in practice through `iters` override — use `iters: 10` via config override before a full run.
- `poc_2stacks.py` is the minimal pipeline smoke test; run it after any change to `null_space.py` or `residual_boost.py`. The underlying MoE-LoRA module is archived at `scripts/legacy/moe_lora.py` (see `docs/research/2026-04-19-moe-lora-root-cause.md`) — do not resurrect.
- Forgetting check is a post-condition of `train_stack.py`, not a separate test — rollback criteria: angle < 30° AND win-rate drop > 0.03.

### Common Patterns

- Python scripts: argparse CLI, `loguru` logging (per project convention — not `print`), `yaml.safe_load` for configs, `Path` objects for filesystem.
- Shell: `#!/bin/bash`, `set -euo pipefail`, long `--` flags, `SCRIPT_DIR`/`PROJECT_DIR` at the top, positional args via `while`/`case`.
- Configs loaded from `configs/micro_kiki/brainstacks.yaml` (training) or `configs/micro_kiki/domains.yaml` (data). Keep those two as the only sources of truth.
- Outputs go to `output/micro-kiki/stacks/<domain>/adapters.safetensors` (one adapter per stack).

## Dependencies

### Internal
- Reads `configs/micro_kiki/brainstacks.yaml` and `configs/micro_kiki/domains.yaml`.
- Writes to `data/micro-kiki/<domain>/train.jsonl`, `valid.jsonl` (relative to project root).
- Writes adapters to `output/micro-kiki/stacks/<domain>/`.
- Evaluated by `src/eval/` (forgetting gate) and `scripts/eval_*.py`.

### External
- MLX (`mx`, `mlx.nn`, `mlx.optimizers`), PyYAML, numpy, loguru.
- Teacher model at `models/Qwen3.5-35B-A3B-Opus-vlm` (default, overridable via `--teacher`).

<!-- MANUAL: -->
