# micro-kiki

41 domain-expert LoRAs (v4-sota, latest stack — incl. v2 retrains for cpp, html-css, python, rust, shell, typescript) + cognitive layer on Qwen3.6-35B-A3B (native MoE, 256 experts, 3B active; earlier drafts referenced Qwen3.5-35B-A3B — superseded 2026-04-18 per real `adapter_config.json`; v3 was 24 stacks, expanded to 41 in v4-sota 2026-04-26). Sequential per-domain training via MLX on Mac Studio M3 Ultra 512 GB; Q4 inference on kxkm-ai (RTX 4090 24 GB) **or local M1 Max 32 GB via mlx-lm — see Inference deployments below**.

> Training, datasets, and the `mlx-lm` fork live in the sibling repo `~/KIKI-Mac_tunner/`. This repo holds the runtime code (routing, cognitive layer, serving, eval) and configs that drive the tuner.

## Where to look

| I want to… | Go to |
|---|---|
| Understand / change the Python runtime (router, memory, serving, eval) | `src/CLAUDE.md` (recurse into `src/<area>/CLAUDE.md`) |
| Write or fix a test | `tests/CLAUDE.md` |
| Add / tune a training recipe, curriculum, or per-domain YAML | `configs/CLAUDE.md` |
| Generate a dataset, distill, or run an eval / benchmark script | `scripts/CLAUDE.md` |
| Check hardware/budget decisions or the pivot to 35B-A3B (3.5 → 3.6) | `docs/specs/2026-04-16-architecture-pivot-35b.md` |
| Deploy (launchd / systemd / vLLM container) | `deploy/`, `docker/vllm.Dockerfile` |
| See a worked code example (KiCad, SPICE, STM32 HAL, …) | `examples/` |

Artifacts (`checkpoints/`, `output/`, `outputs/`, `results/`, `models/`, `data/`) contain build outputs and ingested datasets — do not add code guidance there.

## Hard invariants (load-bearing for the whole project)

- **Base**: `Qwen/Qwen3.6-35B-A3B` (Apache 2.0, 262K ctx, 256 MoE experts, 3B active). **Teacher**: `Qwen3-Coder-480B-A35B` MLX 4bit (local Mac Studio, 1.1 TB).
- **Adapter surface**: standard LoRA via `mlx_lm lora` on **all 17 module kinds** per layer — `linear_attn.{in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj}` (GLA hybrid), `self_attn.{q,k,v,o}_proj`, `mlp.gate` + `mlp.shared_expert_gate` (MoE routers), `mlp.shared_expert.{down,gate,up}_proj`, `mlp.switch_mlp.{down,gate,up}_proj`. (Superseded 2026-04-18: prior rule "attention-only, never MoE FFN" contradicted real `adapter_config.json`; empirical forgetting test chat-fr↔reasoning mean 79.4°, all modules >30°, no catastrophic interference.)
- **Rank**: r=16 for all domains, alpha=16 (1:1 ratio per arXiv 2602.04998 "vanilla LoRA r=16 suffices when LR is tuned"; LR optimal ∝ r^(-1/2) per arXiv 2602.06204). Previous tiered ranks {4,8,12,16,32} superseded. 1.03B trainable params (2.96% of 35B).
- **Layers**: 32/40 (optimal — not 8, not 40). 8 layers undertrained; 40 layers overfits (V3 chat-fr 1.304).
- **Learning rate**: 1e-5 (MLX quantized/BF16).
- **Iters**: 1000 for foundations (chat-fr, reasoning, python), 500 for coding, 100-200 for niches.
- **Metal optimization** (hard invariant): `mx.set_memory_limit(460GB)` + `mx.set_cache_limit(32GB)` — required to prevent GPU Hang on M3 Ultra. Peak mem ~107 GB on 512 GB Studio.
- **DoRA**: NOT supported on Qwen3.6 MoE (SwitchLinear incompatible).
- **Training**: MLX only. BF16. Sequential per-domain, curriculum order (foundations first). Never in parallel — stacks interfere.
- **Forgetting gate**: run after EACH stack; rollback if angle < 30° AND win-rate drop > 0.03.
- **Serving**: Q4_K_M for inference, never below Q4 (quality cliff). Max 4 active stacks simultaneously (VRAM + interference).
- **Router shape**: 35 sigmoid outputs (domains are not mutually exclusive) — not softmax.

## Inference deployments (validated 2026-05-02)

Three distinct serving paths, picked by use case:

| Target | Hardware | Runtime | Models | Wrapper |
|---|---|---|---|---|
| **kxkm-ai** | RTX 4090 24 GB | vLLM Q4_K_M | base + adapters via vLLM LoRA hot-swap | (per `deploy/`) |
| **macM1** (M1 Max 32 GB) | Apple Silicon | mlx-lm 0.31.3 (Python 3.12 venv) | `Qwen3.6-35B-A3B-4bit` MLX (~19 GB) + 41 v4-sota LoRA adapters at `~/llm/adapters/v4-sota/<sub>/` | `~/llm/qwen.sh "..." [adapter] [--think]` |
| **macM1 — edge mode** | ANE (Neural Engine, ~1-2 W) | ANEMLL via CoreML (separate venv `~/llm/venv-anemll/`) | `anemll-qwen3-0.6b` and `anemll-qwen3-1.7b` (CoreML, on T7 SSD) | `~/llm/anemll.sh "..." [0.6b\|1.7b]` |

**macM1 stack gotchas (load-bearing):**
- `mlx-lm < 0.30` raises `Model type qwen3_5_moe not supported` — must use 0.31.x with Python ≥3.12. Earlier `~/llm/venv/` (Python 3.9) is broken for this base.
- **Chat template is mandatory** when invoking adapters via `--adapter-path` — without `tokenizer.apply_chat_template(..., enable_thinking=False)` the adapters loop ("J'ai lu les calculatrices..." on kicad-pcb). The `qwen.sh` wrapper handles this.
- Adapters can be **fused** into the base via `python -m mlx_lm fuse --model <base> --adapter-path <adapter> --save-path <out>` — cuts load time ~40 % at the cost of disk per fused variant (~19 GB each).
- macM1 ↔ studio sync **must use WAN ProxyJump** (`clems@86.207.130.39` → `192.168.13.100`). Tailscale falls into DERP relay and caps at ~0.7 MB/s; WAN direct = ~16 MB/s/flow, ~65 MB/s on 5 concurrent flows.

**ANEMLL caveat:** the 0.6B/1.7B CoreML models are convenient for low-power chat but **hallucinate badly on technical questions** (tested 2026-05-02: 0.6B answers "200 mm" trace width for 10A/2oz, 1.7B fabricates a snubber formula with a fictional inductor). Reserve for chat-fr / classification / batterie longue durée; hardware questions go to the 35B path.

## Never do this

- Don't train on kxkm-ai — 35B-A3B BF16 LoRA does not fit in 24 GB.
- Don't use QLoRA / BitsAndBytes on 35B-A3B (known MoE-layer issues).
- ~~Don't LoRA-tune MoE FFN layers~~ — superseded 2026-04-18: real adapters tune `switch_mlp` + `shared_expert`; empirical forgetting test shows stacks remain ~80° apart.
- Don't merge adapters into base — they are runtime-swappable.
- Don't skip the forgetting check, even for "small" stacks.
- Don't train router and stacks simultaneously.
- Don't use DoRA on Qwen3.6 MoE — SwitchLinear incompatible.
- Don't use alpha = 2*rank (old convention) — use alpha = rank (1:1 ratio per arXiv 2602.04998).

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
