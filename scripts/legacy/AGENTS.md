<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-19 | Updated: 2026-04-19 -->

# scripts/legacy

## Purpose
Archived pre-pivot drivers and modules kept on disk for historical
reference. **Nothing here is on the 35B-A3B training or inference
path.** The post-2026-04-16 pivot (see
`docs/specs/2026-04-16-architecture-pivot-35b.md`) replaced this
pipeline with standard LoRA on Qwen3.5-35B-A3B; anything in this
directory is load-bearing only for reading old commits, interpreting
old outputs, or running the pre-pivot pipeline against the known-dead
pre-pivot adapters.

## Key Files

| File | Description |
|------|-------------|
| `train_stack02.py`, `train_stack03.py`, `train_stack_kxkm.py` | Pre-pivot single-stack trainers (Qwen3.5-4B base). |
| `train_micro_kiki_v3_gpu.py` | Pre-pivot GPU prototype trainer. |
| `train_router_v0.py`, `train_router_kxkm.py` | Earlier router trainers; superseded by `scripts/train_router.py` and `scripts/train_vqc_router.py`. |
| `distill_fast.py` | Early distillation prototype; superseded by `scripts/distill_domain.py`. |
| `e2e_final.py`, `smoke_e2e.py`, `smoke_test_e2e.py` | Archived end-to-end smoke harnesses. |
| `run_pipeline.sh`, `run_eval_stack01.py`, `run_eval_stack01_fast.py`, `run_eval_v2.py`, `run_eval_mini.py`, `run_eval_3.py` | Archived eval drivers. |
| `moe_lora.py` | **Archived 2026-04-19.** Pre-pivot MoE-LoRA layer (4 experts, top-2 routing, rank 16, rsLoRA scaling). Moved from `scripts/micro_kiki/` after the 2026-04-19 audit showed every produced adapter had `lora_b` stuck at zero. Root cause: dual-mount topology — the module is attached both as a child inside `MoELoRALinear` (in forward path) and as a sibling `{target}_moe_lora` attribute (out of forward path); gradients flow to the child, save/load targets the sibling, sibling `lora_b` never moves off its zero init. **Do not attempt to fix or resurrect** — see `docs/research/2026-04-19-moe-lora-root-cause.md` and `docs/research/2026-04-19-prepivot-moe-lora-audit.md`. |

## For AI Agents

### Working In This Directory
- **Do not modify.** These files are archived, not maintained. New work goes in `scripts/` (parent) or `scripts/micro_kiki/`.
- Read-only references are fine: importing `scripts.legacy.moe_lora` from `scripts/micro_kiki/*.py` or `scripts/eval_v2_v3.py` is the one supported use case (replaying the pre-pivot pipeline against pre-pivot adapters).
- If you find yourself needing to fix a bug in `legacy/`, stop and confirm with the operator — the default answer is "use the post-pivot path instead."
- Never reintroduce `legacy/` paths into the post-pivot serving or training code (`src/serving/`, `src/stacks/`, `scripts/train_niches_mlxtune.py`, `scripts/measure_forgetting.py`, the `validate_*` scripts).

### Testing Requirements
- No tests target anything in this directory directly. `tests/test_moe_lora.py` covers `src.stacks.moe_lora` (a different, post-pivot stub) and `tests/test_moe_lora_runtime.py` covers `src.serving.moe_lora_runtime` (the post-pivot inference loader).

### Common Patterns
- These scripts predate the `configs/mlx-per-domain/` per-domain YAML and the `validate_*.py` CI gates. Expect hardcoded paths, expect Qwen3.5-4B references, expect `sys.path.insert` gymnastics.

## Dependencies

### Internal
- `scripts/micro_kiki/*.py` imports `moe_lora` from this directory via `from legacy.moe_lora import ...`.
- `scripts/eval_v2_v3.py::MLXBackend._get_apply_moe_lora` resolves this directory at runtime.

### External
- MLX (`mx`, `mlx.nn`, `mlx.optimizers`), numpy, PyYAML, loguru. Torch on the GPU prototypes only.

<!-- MANUAL: -->
