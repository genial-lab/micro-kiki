# Handoff

## State
PRD 21/50. Restart wrapper running on Studio (PID 47114) for power domain (chunk 39/405). Val loss stagnates at ~1.95 — resume-adapter may not reload weights correctly (val loss resets to ~2.4 each chunk instead of continuing from ~1.5). 6/10 adapters exist: spice, emc, stm32, embedded, freecad + partial power. kicad-dsl and platformio have stale adapters (need delete+retrain). POC v2 proven with real MiniLM embeddings.

## Next
1. **Fix resume**: check if `--resume-adapter-file` in `mlx_lm_fork` actually loads LoRA weights. Try `ssh studio "strings ~/micro-kiki/outputs/sft-restart-all.log | grep -i 'loading\|resume\|adapter'"`. If broken, consider training in ONE long run with `mx.clear_cache()` injected in the fork's training loop.
2. **After fixing**: retrain power, dsp, electronics, kicad-dsl, platformio with working resume.
3. **Alternative**: train on kxkm-ai via Unsloth (RTX 4090, no Metal issues).

## Context
- **ZERO compute on GrosMac**.
- Metal `resource_limit(499000)` = allocation count, not memory. Crashes after ~30-60 iters regardless of peak mem.
- The restart wrapper (`scripts/train_with_restart.py --all --chunk 20`) works mechanically but resume quality is suspect.
- Commit hook: no Co-Authored-By, subject ≤ 50 chars.
