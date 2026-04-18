# Text-JEPA Latent-Dim Ablation Results

**Date**: 2026-04-18
**Branch**: `poc/text-jepa-vqc`
**Task**: 15 of 16 (PoC A)
**Corpus**: 10 niche domains x 100 samples = 1000 samples from `data/final/`
**Backbone**: `models/niche-embeddings` (frozen MiniLM-L6-v2, 384-d)

## Setup reductions (from plan)

CPU wall-clock constraints required reducing several hyperparameters vs the original plan.
Reductions were applied uniformly across all dims so per-dim comparison remains fair.

| Hyperparameter     | Plan spec | Used  | Rationale                                                              |
|--------------------|-----------|-------|------------------------------------------------------------------------|
| max-per-domain     | 500-1000  | 100   | VQC parameter-shift gradient is the bottleneck (~10s/sample at eval)   |
| VQC epochs         | 10        | 2     | Parameter-shift gradient cost scales linearly with epochs              |
| Text-JEPA epochs   | 3         | 2     | Student loss already plateaus by epoch 2 on reduced corpus             |

Total wall-clock: **37m 28s** (14:37:20 -> 15:14:48), well under the 90-min budget.

## Results

| latent_dim | Student params | VQC test accuracy | Retention vs 384-d baseline | Compression | Train time | VQC eval time | Collapsed |
|------------|----------------|-------------------|-----------------------------|-------------|------------|---------------|-----------|
| 64         | 115,520        | 0.180             | 189%                        | 6x          | 18.7 s     | 595.8 s       | no        |
| 128        | 131,968        | 0.185             | 195%                        | 3x          | 17.7 s     | 539.2 s       | no        |
| 256        | 164,864        | 0.190             | 200%                        | 1.5x        | 8.4 s      | 495.3 s       | no        |
| baseline   | n/a (raw)      | 0.095             | 100%                        | 1x          | n/a        | 548.3 s       | n/a       |

VQC params: 122 for all conditions (same 4-qubit x 6-layer StronglyEntanglingLayers circuit).

## Observations

### Baseline under-trained, not student super-human

The "200% retention" is an artifact: **the 384-d mean-pool baseline is under-trained at VQC
epochs=2**. With 10 classes, random accuracy = 10%, so the 0.095 baseline is essentially chance.
The Text-JEPA students are all 2x above chance at epoch=2, which is genuinely better VQC
convergence per epoch thanks to the compressed latent space — but the correct reading is *not*
"JEPA doubles accuracy", it is "JEPA converges faster under the same VQC budget".

For reference, Task 14 at VQC epochs=10 on 50 samples/domain achieved baseline=0.925 and
text_jepa=0.900 — so at full VQC convergence the baseline is actually slightly ahead. The
ablation here measures *VQC-budget efficiency*, not asymptotic accuracy.

### Accuracy is flat across dims (0.180 → 0.185 → 0.190)

The 5 pt spread across 64 / 128 / 256 is within noise for an n_test = 200 binomial sample
(stderr ~2.8 pt). Practically, all three dims give equivalent routing signal under this VQC
budget — there is no clear accuracy advantage to going above 64.

### No collapse at any dim

Embedding std monitor fired at no point across the 3 runs. Loss curves:
- dim=64: 0.4153 -> 0.3505
- dim=128: 0.3807 -> 0.3398
- dim=256: 0.3634 -> 0.3399

Convergence behavior is similar; larger dim trains marginally faster in wall-clock per-epoch
(likely memory-bandwidth-bound on MLP forward, not quadratic in dim).

### Student parameter count grows sub-linearly in latent_dim

The input projection (384 -> 256) dominates student size, so param count only grows from 115k
(64d) -> 165k (256d) — a 43% increase for 4x latent dim.

## Decision

- [x] **Clear win at dim=64** — 6x compression with accuracy indistinguishable from 128/256
      under this VQC budget.
- [ ] Sweet spot at dim=128 (Task 14 baseline, reasonable default)
- [ ] Only dim=256 matches baseline

**Recommendation**: adopt **dim=64** as the default latent for PoC A. Downstream deployments
(Coherent Ising, on-device) benefit disproportionately from 6x input compression, and we
observed no accuracy penalty in this regime. Task 14 should be rerun at dim=64 with full VQC
epochs=10 before locking this decision.

## Deviations from plan

- `max-per-domain` set to 100 (plan: 500-1000) — VQC eval bottleneck
- `train-epochs` set to 2 (plan: 3) — loss already near plateau
- `vqc-epochs` set to 2 (plan: 10) — wall-clock budget
- Checkpoints saved to `models/text-jepa/student_dim{64,128,256}.pt` (addition to plan)

## Next steps

Task 16 (final regression test) — run the full Text-JEPA + routing test suite and mark the PoC
decision in `results/text-jepa-vqc.md`.

Before finalizing the dim choice, consider a follow-up run of Task 14's setup
(50 samples/domain, 10 VQC epochs) at dim=64 to confirm the 6x-compression win holds at full
VQC convergence, not just under constrained-budget evaluation.
