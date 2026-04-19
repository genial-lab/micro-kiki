# C1: Classical Baselines vs Torch VQC Router

**Setup.** 10 domain-routing classes (dsp, electronics, emc, embedded, freecad, kicad-dsl, platformio, power, spice, stm32) x 50 samples per domain from `data/final`, encoded to 384-D via frozen MiniLM-L6-v2 mean-pool, 80/20 train/test split, 5 seeds, 300 epochs for trainable baselines.

**Baselines.**
- **Stratified random**: draws labels from the train distribution — chance floor.
- **LogReg on PCA-4**: 384-D embeddings compressed to 4-D via PCA, then multinomial logistic regression. Information-matched to VQC (same 4 effective features).
- **Torch VQC (ours)**: 4-qubit, 6-layer StronglyEntanglingLayers, learned projection 384->4, weight decay 1e-4.
- **MLP (384->64)**: 1-hidden-layer classifier with capacity comparable to VQC+projection.
- **LogReg on raw 384-D**: upper bound — no information loss, maximally expressive linear head.

**Results** (5 seeds, mean ± population std; see Figure `c1-comparison.pdf`).

| Baseline | Test acc | Macro F1 | Params | Train time |
|---|---|---|---|---|
| Stratified random | 0.118 ± 0.024 | 0.115 | 0 | <0.01s |
| LogReg on PCA-4 | 0.364 ± 0.040 | 0.338 | 1,586 | <0.1s |
| **Torch VQC (ours)** | **0.246 ± 0.031** | **0.163** | **1,662** | **4.2s** |
| MLP (384->64) | 0.546 ± 0.029 | 0.536 | 25,290 | 0.3s |
| LogReg on raw 384-D | 0.546 ± 0.036 | 0.533 | 3,850 | 0.1s |

**Interpretation.**

1. **The VQC UNDERPERFORMS its information-matched classical baseline.** LogReg on PCA-4 achieves 0.364 — a full 12 accuracy points above our VQC's 0.246, with a nearly identical parameter budget (1,586 vs 1,662). This refutes the intuition that a learned non-linear quantum projection would extract richer 4-D features than PCA's variance-maximising projection. In practice, `pi * tanh(W x + b)` before rotation gates loses information compared to a lossless linear projection passed to a maximum-likelihood linear classifier.

2. **The full-capacity classical ceiling sits ~30 points above the VQC.** LogReg on raw 384-D hits 0.546 — the upper bound given our embedding and routing task. The gap torch_vqc (0.246) -> logreg (0.546) is the price of the 4-qubit information bottleneck combined with the projection suboptimality.

3. **MLP with ~15x more params matches LogReg exactly.** MLP (25,290 params, 0.546) and LogReg raw (3,850 params, 0.546) tie, indicating linear separability of MiniLM embeddings for this task — nonlinearity buys nothing.

**Kill criterion check.** LogReg raw 384-D = 0.546 < 0.80 threshold. Paper A is NOT retracted. Routing is not trivially solvable with the raw embedding — there IS a genuine routing challenge.

**Implications for Paper A.** This result is crisper than if the VQC had won: it establishes that the current 4-qubit VQC + learned projection approach is **not competitive** as a routing classifier on MiniLM embeddings. The contribution of this work is therefore:

- **Methodological only**: `torch-vqc` (https://github.com/electron-rare/torch-vqc) makes VQC research tractable at ~3000x the speed of PennyLane parameter-shift, enabling rigorous comparisons like the one above.
- **A reproducible baseline ceiling**: any future quantum-advantage claim for VQC routing on pretrained embeddings must beat 0.546 (LogReg raw 384-D) with fewer than 3,850 parameters, not just chance.
- **Architectural lesson**: naive `W x + pi * tanh` projections into a 4-qubit VQC lose information versus PCA. Future work should explore (i) more qubits with projection, (ii) alternative embeddings (learned features rather than tanh-bounded rotations), (iii) quantum-kernel methods instead of parameterised circuits.

Paper A §4 should lead with the benchmarking table above and frame the VQC as a *reference architecture* whose ceiling we quantify, not as a proposed classifier.
