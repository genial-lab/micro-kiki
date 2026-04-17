# Stacks (Standard LoRA Domain Experts)

35 domain-specific LoRA stacks on Qwen3.5-35B-A3B (native MoE) base.

## Architecture pivot (2026-04-16)

The previous approach used custom MoE-LoRA (4 experts per projection, top-2 routing) on a 4B dense base. This has been replaced by **standard LoRA** on the Qwen3.5-35B-A3B native MoE base. The model is already a MoE (256 experts, 3B active per token) — adding custom MoE-LoRA on top is redundant.

## Architecture

- Standard LoRA via PEFT `LoraConfig`: rank 16, scale 32.0, dropout 0.01
- Target modules: q/k/v/o attention projections only — do NOT touch MoE FFN layers
- OPLoRA orthogonal projection for forgetting prevention (applied from stack 04 onward)
- Each stack = one domain adapter saved as `adapters.safetensors`

## Training

Training runs on Mac Studio M3 Ultra 512 GB via MLX LoRA (KIKI-Mac_tunner pipeline). Not on kxkm-ai — the model is too large (74 GB BF16) for 24 GB VRAM.

```bash
# From ~/KIKI-Mac_tunner
./train.sh --config configs/mlx-lm-qwen35-35b-a3b-micro-kiki.yaml
```

Config: `~/KIKI-Mac_tunner/configs/mlx-lm-qwen35-35b-a3b-micro-kiki.yaml`

Key parameters: LR 1e-5, batch_size 2, grad_accumulation 8, rank 16, 2000 iters, max_seq_length 4096.

See `docs/training/README.md` for the full workflow.

## Training Order

Sequential, curriculum order (foundations first). Never in parallel — stacks interfere.
Check `~/KIKI-Mac_tunner/configs/micro_kiki/brainstacks.yaml` for domain ordering and dataset mappings.

## Adding a New Stack

1. Ensure dataset is classified + deduped in `~/KIKI-Mac_tunner/data/micro-kiki/<domain>/`
2. Update `~/KIKI-Mac_tunner/configs/mlx-lm-qwen35-35b-a3b-micro-kiki.yaml` to point at the new domain
3. Train with BF16, Mac Studio only
4. Run forgetting check IMMEDIATELY after training
5. Test domain router with new stack active

## Forgetting Check (Critical)

After EACH stack trained:
- Measure angle between base and adapted weights
- If angle < 30° AND win-rate drop > 0.03 → rollback
- Framework: `uv run python src/eval/forgetting.py --stack <domain>`

## Memory Budget (Mac Studio, BF16 training)

Peak training memory: ~195 GB. Gradient checkpointing is required.
Inference: max 4 stacks active simultaneously.

## Anti-Patterns

- Don't skip forgetting check — even for "small" stacks
- Don't train on overlapping data across stacks (dedup enforces disjoint)
- Don't apply LoRA to MoE FFN layers — attention projections only
- Don't merge adapters into base — keep them as runtime LoRA
- Don't use QLoRA/BitsAndBytes on 35B-A3B MoE (known issues with MoE layers)
- Don't train on kxkm-ai (model too large for 24 GB BF16 LoRA)
