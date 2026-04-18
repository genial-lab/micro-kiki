# micro-kiki

35 domain-expert LoRAs + cognitive layer on Qwen3.5-35B-A3B (native MoE, 256 experts, 3B active). Sequential per-domain training via MLX on Mac Studio M3 Ultra 512 GB; Q4 inference on kxkm-ai (RTX 4090 24 GB).

> Training, datasets, and the `mlx-lm` fork live in the sibling repo `~/KIKI-Mac_tunner/`. This repo holds the runtime code (routing, cognitive layer, serving, eval) and configs that drive the tuner.

## Where to look

| I want to… | Go to |
|---|---|
| Understand / change the Python runtime (router, memory, serving, eval) | `src/CLAUDE.md` (recurse into `src/<area>/CLAUDE.md`) |
| Write or fix a test | `tests/CLAUDE.md` |
| Add / tune a training recipe, curriculum, or per-domain YAML | `configs/CLAUDE.md` |
| Generate a dataset, distill, or run an eval / benchmark script | `scripts/CLAUDE.md` |
| Check hardware/budget decisions or the pivot to 35B-A3B | `docs/specs/2026-04-16-architecture-pivot-35b.md` |
| Deploy (launchd / systemd / vLLM container) | `deploy/`, `docker/vllm.Dockerfile` |
| See a worked code example (KiCad, SPICE, STM32 HAL, …) | `examples/` |

Artifacts (`checkpoints/`, `output/`, `outputs/`, `results/`, `models/`, `data/`) contain build outputs and ingested datasets — do not add code guidance there.

## Hard invariants (load-bearing for the whole project)

- **Base**: `Qwen/Qwen3.5-35B-A3B` (Apache 2.0, 262K ctx). **Teacher**: `Qwen3-Coder-480B-A35B` MLX 4bit (local Mac Studio, 1.1 TB).
- **Adapter surface**: standard LoRA on q/k/v/o attention projections **only**. Never on MoE FFN layers — the MoE routing is already trained.
- **Rank budget**: 4/8 narrow niches, 12 coding-secondary/technical/apps, 16 broad niches, 32 foundations; alpha = 2×rank; scale 2.0.
- **Training**: MLX only. BF16. Sequential per-domain, curriculum order (foundations first). Never in parallel — stacks interfere.
- **Forgetting gate**: run after EACH stack; rollback if angle < 30° AND win-rate drop > 0.03.
- **Serving**: Q4_K_M for inference, never below Q4 (quality cliff). Max 4 active stacks simultaneously (VRAM + interference).
- **Router shape**: 35 sigmoid outputs (domains are not mutually exclusive) — not softmax.

## Never do this

- Don't train on kxkm-ai — 35B-A3B BF16 LoRA does not fit in 24 GB.
- Don't use QLoRA / BitsAndBytes on 35B-A3B (known MoE-layer issues).
- Don't LoRA-tune MoE FFN layers.
- Don't merge adapters into base — they are runtime-swappable.
- Don't skip the forgetting check, even for "small" stacks.
- Don't train router and stacks simultaneously.

## Agent workflow

1. Read the nested `CLAUDE.md` for the directory you're about to touch **before** editing. Claude Code auto-loads it.
2. If the task spans domains (e.g. new stack = configs + scripts + tests + eval), read each nested file for those dirs; do not assume the root covers their specifics.
3. For anything architectural, the authoritative decision log is `docs/specs/` — the most recent dated file wins over older ones.
4. The sibling `~/KIKI-Mac_tunner/` repo owns training execution. This repo owns configs, runtime, and eval. Don't duplicate training logic here.

## Commit conventions

- `feat(<phase>): …`, `docs(<area>): …`, `fix(<area>): …`
- Subject ≤ 50 chars (pre-commit hook enforces)
- **No `Co-Authored-By` trailer** — the hook rejects it.

## Language

- Conversation: French. Code, comments, commits, docs: English.
