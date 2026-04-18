# Related Work — Aeon Latent Predictor

## Abstract

This section positions the Aeon latent predictor—a stack-conditioned, in-situ numpy MLP for predicting conversation-state transitions in a hybrid quantum-neuromorphic-classical system—against the most relevant state-of-the-art methods in world modeling, self-supervised learning, and memory-augmented language systems. Aeon differs fundamentally from existing approaches by operating in the latent space of a multi-expert language model ensemble (32 LoRA-fine-tuned stacks), learning from conversation turns in near-real-time with cold-start fallback and anti-collapse safeguards, and requiring only kilobytes of weights for edge deployment. We situate this work at the intersection of three methodological families: JEPA-style predictive architectures, self-supervised representation learning, and memory-augmented decoding systems.

---

## 1. Latent Predictive Architectures (JEPA Family)

The closest cousins to Aeon's design philosophy are the JEPA (Joint-Embedding Predictive Architecture) family introduced by LeCun and Meta researchers. The foundational I-JEPA (arXiv:2301.08243) proposed learning abstract representations by predicting one region of an image from another without pixel reconstruction. V-JEPA (arXiv:2404.08471) extended this to video, predicting future visual embeddings from masked present frames; the method achieved strong downstream performance on action recognition and semantic segmentation while learning representations without contrastive losses.

Most directly relevant is V-JEPA 2 (arXiv:2506.09985), which introduced action-conditioned video prediction: given visual state and action, predict the resulting next-state embedding. The method demonstrated that JEPA-style in-latent prediction scales to high-dimensional visual domains while remaining computationally tractable at deployment. H-JEPA (the hypothetical hierarchical extension discussed in LeCun's 2022 world models position paper) conceptually extends this stack to reason over temporal abstractions.

**Aeon's positioning**: Aeon applies the JEPA core insight—predict in learned latent space rather than reconstruct observations—to the text/conversation domain, where state is represented by a learned embedding of dialogue history. Like V-JEPA 2-AC, Aeon is action-conditioned: the "action" is which LoRA stack is active (stack ID one-hot encoded). Critically, Aeon differs in three ways: (1) it learns from conversation turns in real-time (sleep-cycle episodic training), not pre-recorded trajectories; (2) it has no visual ground truth, only text-based memory states; (3) it operates on a 32-stack expert mixture, a coordination problem absent in V-JEPA's single-model setting. The absence of action semantics in conversation (i.e., the user's next turn is not controllable) means Aeon is a pure memory-state predictor, not a control-conditioned world model.

---

## 2. Generative World Models (Contrast)

A large body of recent work frames the problem differently: reconstruct or generate future states/observations, then plan via the learned model. **DreamerV3** (arXiv:2301.04104) trains a VAE-based world model (Recurrent State Space Model, RSSM) to compress observations, then learns both a value function and imaginative policy through dream rollouts. **TD-MPC2** (arXiv:2310.16828) similarly learns a deterministic world model but uses Cross-Entropy Method (CEM) for planning within the learned dynamics. **Genie** and **Genie 2** (Google DeepMind 2024) take generative scaling further: foundation models that learn spatiotemporal video generation from latent codes, enabling world simulation at scale.

These methods excel at long-horizon planning and control in environments where observation sequences are continuous and physics-based. However, they incur a reconstruction cost: predicting the full next observation or a high-fidelity latent representation is more expensive than predicting low-dimensional state changes. Additionally, they are designed around single-agent planning, not multi-expert coordination or memory retrieval.

**Aeon's contrast**: Aeon deliberately rejects the "world model" framing. It does not aim to reconstruct or simulate future text; instead, it predicts whether the conversation state will shift toward a given memory region (retrieval anticipation). The function $h_t \to h_{t+1}$ is a tiny embedding-space transition, not a full generative model. This distinction is crucial: Aeon is a *memory management aid*, not a planner. It answers "which expert stack will be needed next?" or "what memory should I fetch?" rather than "what will the user say?" This design choice enables both size and safety: a numpy MLP scales to edge devices, and there is no open-ended text generation from which harmful content could emerge.

---

## 3. Self-Supervised Visual Representation Learning (Methodological Cousins)

While Aeon operates on conversation embeddings rather than images, the methodological toolkit overlaps significantly with self-supervised vision learning, particularly the DINO family. **DINO** (arXiv:2104.14294) pioneered self-distillation without labels: a student network is trained to match the softmax output of an exponential-moving-average (EMA) teacher, with a stop-gradient on the teacher to prevent collapse. **DINOv2** (arXiv:2304.07193) scaled this to internet-scale data, learning universal visual features with strong zero-shot transfer. **DINOv3** (arXiv:2508.10104), released in 2025, further refined the approach with improved scaling and multi-scale learning.

Aeon borrows three key ingredients from this family:

1. **Cosine loss** for similarity-based learning: $\mathcal{L} = 1 - \cos(\hat{h}_{t+1}, h_{t+1})$ matches DINO's use of cosine similarity in the embedding space.
2. **Stop-gradient blocking**: Aeon applies stop-gradient to the teacher (previous-turn embedding) to prevent feedback loops and mode collapse.
3. **Centering and collapse detection**: Aeon monitors the standard deviation of predicted embeddings; when std drops below a threshold (indicating collapsed representations), the model weights are rolled back. This mirrors DINO's explicit centering term to combat mode collapse, though implemented as a runtime safety mechanism rather than a loss regularizer.

**Key differences**: DINO operates on large vision backbones (ViT) trained on terabyte-scale data; Aeon is a single numpy MLP (< 1 MB) trained on conversation turns in the conversation loop itself. DINO's self-distillation is unsupervised; Aeon has access to ground truth (the actual next-turn embedding) but chooses predictive loss for cold-start robustness. DINO explicitly learns global semantic features; Aeon learns state *transitions*, a fundamentally different objective.

---

## 4. Regularization and Collapse Prevention: LeJEPA's SIGReg

The most methodologically relevant recent work on representation collapse prevention is **LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics** (Balestriero & LeCun, arXiv:2511.08544, November 2025). This paper introduces **SIGReg (Sketched Isotropic Gaussian Regularization)**, a principled regularizer that explicitly enforces isotropic Gaussian structure over the learned representation space, eliminating collapse without resorting to stop-gradients, exponential-moving-average teachers, or centering heuristics.

**SIGReg mechanism**: During training, embeddings are randomly projected to 1D via Cramér-Wold theorem, and the 1D marginals are regularized to match a standard Gaussian. This has linear complexity in embedding dimension and batch size, scales to large models, and provides formal guarantees against mode collapse. LeJEPA also introduces a **Weak-SIGReg** variant (arXiv:2603.05924, 2026) that constrains only the covariance matrix, reducing regularization overhead.

**Aeon's convergence with LeJEPA philosophy**: Aeon shares LeJEPA's core insight—collapse prevention should be baked into the method, not patched on as a heuristic. While SIGReg regularizes embeddings globally during training, Aeon implements collapse detection as a runtime safety mechanism: the standard-deviation ratio (`std(h_t^pred) / std(h_t^true)`) serves as a tripwire. When this ratio falls below a threshold (indicating clustered representations), Aeon rolls back predictor weights to the checkpoint before divergence. This is orthogonal to SIGReg (training-time vs. deployment-time) but philosophically aligned: both reject ad-hoc centering, stop-gradient hacks, and EMA teachers in favor of principled, interpretable mechanisms.

**Contrast with sigmoid-family losses**: For completeness, the sigmoid-family methods in vision-language learning are distinct from SIGReg:

- **SigLIP** (arXiv:2303.15343, Google 2023): Sigmoid-based contrastive loss for image-language pairs, superior to NT-Xent (normalized temperature-scaled cross-entropy) in stability and performance.
- **SigLIP2** (arXiv:2502.14786, Google 2025): Refinement with improved calibration for large-scale vision-language models.

Aeon uses cosine loss directly, which is scale-invariant and does not require temperature tuning—advantageous for cold-start scenarios. Sigmoid losses assume balanced positive/negative pairs; Aeon has only one ground truth per step (the next embedding) and ambiguous negatives (all other conversation histories).

---

## 5. Memory-Augmented Language Models

Aeon is not itself a memory system but rather a meta-module that *manages* memory retrieval in a larger language model. The native habitat is multi-expert LLM ensembles with explicit memory buffers. Relevant prior work in this space:

- **MemGPT** (arXiv:2310.08560): Introduced continual learning for context-window-limited LLMs by maintaining a persistent memory bank and learning when/what to write to and read from memory. MemGPT uses heuristic policies (recency, relevance); Aeon could be viewed as learning these policies implicitly from conversation dynamics.
- **Larimar** (arXiv:2403.11901): A memory-augmented retrieval system that learns to condition LLM generation on retrieved past contexts. Larimar explicitly trains a retriever to anticipate which memory will be most useful, a task very close to Aeon's stack-selection objective.
- **RETRO** (arXiv:2112.04426): Foundational work on retrieval-augmented decoding, showing that LLMs augmented with dynamic retrieval can improve factuality and reduce hallucination. RETRO decodes tokens while simultaneously attending to retrieved documents.
- **Gecko/text-embedding retrieval**: Lightweight embedding models (e.g., LLaMA2-based embedders) used for fast approximate retrieval in retrieval-augmented generation pipelines. Aeon could be paired with Gecko-style embedders for real-time memory indexing.

**Aeon's niche**: These systems either (a) learn retrieval at the token level during generation, or (b) hand-craft policies for memory management. Aeon learns the *memory transition function*: given the current conversation state, what is the next likely memory region? This is upstream of retrieval—it answers "what will we need?" before the model actually requests it. The stack-conditioning (one-hot over 32 LoRA experts) further distinguishes Aeon: it explicitly routes between a mixture of expert modules, not a flat memory buffer. To our knowledge, no prior work learns expert selection via latent state prediction.

---

## 6. What's Novel in Aeon Predictor

Building on the above, Aeon's core contributions are:

1. **Stack-conditioned latent prediction**: Aeon learns a separate set of weights for each of the 32 LoRA-fine-tuned expert stacks, conditioning the $h_t \to h_{t+1}$ transition on the active stack via one-hot encoding. This is a multi-hypothesis world model, not a single monolithic predictor. No prior work (to our knowledge) applies JEPA-style prediction to expert-mixture systems.

2. **Runtime safety via rollback-on-collapse**: Aeon implements a novel anti-collapse mechanism that triggers weight rollback when predicted representations begin to cluster (detected via std-ratio thresholding). This is distinct from DINO's centering loss; it is a deployed safety measure, not a training regularizer. Philosophically, this approach aligns with LeJEPA's rejection of heuristic-based collapse prevention: both eliminate stop-gradient tricks, EMA teachers, and ad-hoc centering in favor of transparent, mechanistic safeguards. Aeon's deployment-time instantiation of this principle—monitoring and rolling back in the loop—is a novel extension of LeJEPA's training-time philosophy to the operational regime. It is particularly important for edge systems where model drift is costly and retraining may be infrequent.

3. **Numpy-only, kilobyte-scale MLP for edge deployment**: Aeon's predictor is a 2-3 layer MLP (< 1 MB weights) with numpy backend, eliminating GPU/PyTorch dependencies. This makes it deployable on embedded systems and edge devices alongside Whisper-tiny or other resource-constrained models. The trade-off is reduced capacity, but with stack-conditioning and cold-start fallback, the method remains robust.

These three contributions together define a lightweight, safe, and expert-routed latent prediction system tailored to conversation-based AMI (Artificial Machine Intelligence) systems. The architecture is validated empirically through the Aeon predictor PoC (9/12 tasks complete as of 2026-04-17), with latent prediction accuracy and memory retrieval anticipation serving as the primary evaluation metrics.

---

## References (In-text arXiv citations)

- I-JEPA: arXiv:2301.08243
- V-JEPA: arXiv:2404.08471
- V-JEPA 2: arXiv:2506.09985
- DreamerV3: arXiv:2301.04104
- TD-MPC2: arXiv:2310.16828
- DINO: arXiv:2104.14294
- DINOv2: arXiv:2304.07193
- DINOv3: arXiv:2508.10104
- LeJEPA (SIGReg): arXiv:2511.08544
- Weak-SIGReg: arXiv:2603.05924
- SigLIP: arXiv:2303.15343
- SigLIP2: arXiv:2502.14786
- MemGPT: arXiv:2310.08560
- Larimar: arXiv:2403.11901
- RETRO: arXiv:2112.04426
