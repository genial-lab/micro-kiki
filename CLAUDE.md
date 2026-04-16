# micro-kiki — project context for Claude Code

32 domain experts (LoRA) on Qwen3.5-35B-A3B MoE base with a cognitive layer (memory palace + negotiator + anti-bias). Trains on Mac Studio M3 Ultra 512 GB.

## Architecture pivot (2026-04-16)

Switched from Qwen3.5-4B + custom MoE-LoRA to Qwen3.5-35B-A3B (native MoE, 256 experts, 3B active) + standard LoRA. See `docs/specs/2026-04-16-architecture-pivot-35b.md`.

## Key decisions locked in

- **Base model**: Qwen3.5-35B-A3B (Apache 2.0, 262K ctx, 256 MoE experts, 3B active/token)
- **Adapter technique**: Standard LoRA rank 64, scale 64, targeting q/k/v/o attention projections (NOT MoE FFN layers)
- **Training**: 3-phase curriculum (seq 512→1280→4096), LR decay (8e-6→5e-6→3e-6), via MLX in KIKI-Mac_tunner
- **Init strategy**: OPLoRA projection for forgetting prevention (stacks >= 04)
- **Quantization**: Q4_K_M for inference, BF16 for training
- **Router**: domain classifier for adapter selection, max 4 active stacks
- **Dispatcher**: training-free YAML mapping from router output to 7 meta-intents
- **Memory**: Aeon (Atlas SIMD + Trace graph), native or Qdrant/Neo4j backends
- **Negotiator**: CAMP arbitration + Catfish dissent; adaptive judge (Qwen3.5-35B fast / Mistral-Large deep)
- **Anti-bias**: KnowBias double-application + RBD runtime detector
- **Teacher**: Qwen3-Coder-480B-A35B MLX 4bit (local Mac Studio, 1.1 TB)
- **Serving**: MLX primary (Mac Studio), vLLM Q4 inference (kxkm-ai)

## Hardware

- **Mac Studio M3 Ultra 512 GB**: Training (BF16 LoRA, 74 GB), teacher (480B), MLX serving
- **kxkm-ai RTX 4090 24 GB**: Q4 inference only
- **Tower**: Aeon backends (Qdrant, Neo4j), Piper TTS

## Don't

- Don't use QLoRA/BitsAndBytes on 35B-A3B MoE (known issues with MoE layers)
- Don't train on kxkm-ai (model too large for 24 GB BF16 LoRA)
- Don't LoRA-tune MoE FFN layers (only attention projections — the MoE routing is already learned)
- Don't drop below Q4 quantization (quality cliff)
- Don't route > 4 stacks simultaneously
- Don't train stacks in parallel (interference)
- Don't skip forgetting check after each stack

## Do

- Set `UNSLOTH_COMPILE_DISABLE=1` before training (MoE mixed-precision kernel fix)
- Train stacks sequentially, curriculum order
- Run forgetting check after EACH stack; rollback if angle < 30° AND win-rate drop > 0.03
- Use Qwen3-Coder-480B as teacher for distillation (local, no network dependency)
- Use gradient checkpointing during training (74 GB is tight)

## Commit conventions

- `feat(<phase>): <short imperative>` — new code
- `docs(<area>): <short imperative>` — docs
- `fix(<area>): <short imperative>` — bug fix
- Subject ≤ 50 chars (pre-commit hook enforces)
- No `Co-Authored-By` trailer (hook rejects it)

## External resources

- HuggingFace: `Qwen/Qwen3.5-35B-A3B` (base), `Qwen/Qwen3-Coder-480B-A35B-Instruct` (teacher)
- Papers: OPLoRA 2510.13003, Aeon 2601.15311, CAMP 2604.00085, Catfish 2505.21503, KnowBias 2601.21864, RBD 2505.17100
- Datasets: 32 domains in `~/KIKI-Mac_tunner/data/micro-kiki/` (classified + deduped + split)
