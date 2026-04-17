# Handoff

## State
PRD at 21/50. POC v2 proven end-to-end with real MiniLM embeddings. 6/10 SFT adapters BLOCKED by Metal OOM: fork `mlx_lm_fork` without keys LoRA-ises too many layers → crash at ~iter 160. With keys → 0 trainable params. 4 working adapters (spice, emc, stm32, embedded) were trained with an earlier fork version. freecad succeeded (small dataset, finished before OOM). 5 remaining: platformio, power, dsp, electronics, kicad-dsl all fail.

## Next
1. **Debug the fork**: `ssh studio "diff ~/KIKI-Mac_tunner/lib/mlx_lm_fork/tuner/lora.py"` vs the version that trained spice. Check git log in `~/KIKI-Mac_tunner/lib/mlx_lm_fork/` for changes.
2. **Alternative**: train on kxkm-ai via vLLM/Unsloth instead of MLX (RTX 4090 24GB may fit rank 4-8 with QLoRA).
3. **After adapters**: story-14 forgetting check, story-15 eval, DPO pipeline.

## Context
- **ZERO compute on GrosMac** — all on Studio or kxkm-ai.
- The Metal `resource_limit(499000)` is about allocation COUNT not memory SIZE. Peak mem stays at 106GB but allocations accumulate.
- `save_every: 50` in config may help (forces checkpoint → GC).
- Training script: `scripts/train_niches_mlxtune.py`, wrapper: `scripts/train_one_by_one.sh`.
- Commit hook: no Co-Authored-By, subject ≤ 50 chars.
