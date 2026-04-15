# Stacks (MoE-LoRA Domain Experts)

32 domain-specific LoRA stacks on Qwen3.5-4B base.

## Architecture

- MoLoRA: rank 16, 4 experts per projection, top-2 routing
- Init: PiSSA default, LoRA-Null alternative, OPLoRA for forgetting prevention
- Each stack = one domain (e.g., `stack_00_embedded_c`, `stack_01_pcb_design`)

## Training Order

Sequential, curriculum order (foundations first). Never parallel on same GPU.
Check `configs/` for domain ordering and dataset mappings.

## Adding a New Stack

1. Create `stack_NN_domain_name.py` with training config
2. Ensure dataset is classified + deduped (KIKI-Mac_tunner pipeline)
3. Train with BF16, single GPU (kxkm-ai RTX 4090)
4. Run forgetting check IMMEDIATELY after training
5. Test meta-router with new stack active

## Forgetting Check (Critical)

After EACH stack trained:
- Measure angle between base and adapted weights
- If angle < 30° AND win-rate drop > 0.03 → rollback
- Framework lives in `src/eval/`

## VRAM Budget

Max 4 stacks active simultaneously (24 GB limit).
Each active LoRA adapter ≈ 50-80 MB depending on rank.

## Anti-Patterns

- Don't skip forgetting check — even for "small" stacks
- Don't train on overlapping data across stacks (dedup enforces disjoint)
- Don't change rank/experts without revalidating VRAM budget
- Don't merge adapters into base — keep them as runtime LoRA
