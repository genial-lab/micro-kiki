# Paper Outline: Hybrid Quantum-Neuromorphic-Classical Routing for Domain-Expert LLM Inference

## Title options
1. "Triple-Hybrid Routing: Quantum VQC, Spiking Neural Networks, and Classical LLMs for Domain-Expert Inference"
2. "From Qubits to Spikes to Tokens: A Hybrid Architecture for Energy-Efficient Domain Routing in Large Language Models"
3. "micro-kiki: Quantum-Neuromorphic-Classical Domain Routing for 35B MoE Language Models"

## Abstract (~200 words)
- Problem: Domain-expert LLMs require efficient routing to specialized adapters
- Gap: No existing system combines quantum, neuromorphic, and classical computing for LLM routing
- Contribution: Triple-hybrid pipeline — VQC domain classifier (4 qubits, PennyLane), SNN backbone (LAS-converted 35B MoE), classical fallback (Qwen3.5-35B-A3B + 10 LoRA niches)
- Results: [PENDING — fill after training + benchmarks]
- Claim: First demonstration of quantum-neuromorphic-classical routing for production-scale LLM inference

## 1. Introduction
- Multi-domain LLMs need routing (which expert/adapter to activate)
- Classical routers work but miss energy efficiency and quantum advantages
- Neuromorphic hardware (Akida, Loihi) enables ultra-low-power inference
- Quantum circuits (VQC) offer potential for high-dimensional classification
- **Thesis**: combining all three provides a complete spectrum from cloud (classical) to edge (SNN) to quantum-enhanced routing

## 2. Related Work

### 2.1 MoE-LoRA Systems
- MixLoRA (TUDB-Labs) — MoE LoRA in FFN blocks
- MoLA (NAACL 2025) — layer-wise LoRA expert allocation
- HMoRA — hierarchical token+task routing
- **Gap**: none add cognitive memory or SNN/quantum routing

### 2.2 ANN→SNN Conversion
- LAS (arxiv 2505.09659) — lossless conversion up to OPT-66B
- FAS — fast but lossy conversion
- SpikingBrain (BICLab) — native spiking LLM (7B, 76B)
- **Gap**: no MoE model converted to SNN, no quantum integration

### 2.3 Quantum ML for NLP
- IonQ quantum-enhanced LLM fine-tuning
- Quantum-Train (compression via VQC)
- QPA (quantum parameter adaptation)
- **Gap**: no quantum routing for domain selection in LLMs

### 2.4 Cognitive Architecture
- MAGMA (multi-graph agentic memory)
- A-Mem (agentic memory for LLM agents)
- **Gap**: no sleep consolidation, no SNN integration

## 3. Architecture

### 3.1 Overview
```
Query → Quantum VQC Router → domain classification
           ↓ (confidence > θ)          ↓ (confidence < θ)
     SNN SpikingKiki           Classical MetaRouter
           ↓                          ↓
     Model Router → 35B + LoRA adapter selection
           ↓
     Aeon Memory → context injection
           ↓
     Inference (MLX / vLLM)
           ↓
     Negotiator → quality arbitration
           ↓
     Aeon Memory → persistence
```

### 3.2 Quantum VQC Router
- 4 qubits, 6 variational layers
- AngleEmbedding + StronglyEntanglingLayers
- 4 PauliZ measurements → linear head → 11 classes
- Trained on domain-labeled embeddings
- PennyLane simulator (QPU optional)
- Parameter count: ~200 (vs ~3.4M classical router)

### 3.3 SNN Backbone (SpikingKiki)
- LAS conversion of Qwen3.5-35B-A3B (MoE, 256 experts)
- First SNN MoE at 35B scale
- Expert routing preserved via spike-coded top-K selection
- LIF neurons with surrogate gradient
- Target: 69% sparsity, energy reduction TBD

### 3.4 Classical Backbone
- Qwen3.5-35B-A3B (201 languages, thinking mode)
- 10 niche LoRA stacks (rank 4-16, adaptive per domain)
- Domains: kicad-dsl, spice, emc, stm32, embedded, freecad, platformio, power, dsp, electronics
- Training: mlx-tune on Mac Studio M3 Ultra (512 GB)

### 3.5 Cognitive Layer (Aeon)
- Atlas: SIMD vector index (cosine similarity)
- Trace: neuro-symbolic episodic graph
- AeonSleep: sleep consolidation (SleepGate + ForgettingGate + Consolidation)
- Pre-inference: recall top-K memories, inject into context
- Post-inference: persist turn as episode

## 4. Training

### 4.1 Niche LoRA Training
- 10 domains × adaptive hyperparameters
- Data: KIKI-Mac_tunner + HuggingFace mascarade datasets (2-9× enrichment)
- Teacher: Qwen3-Coder-480B-A35B (IQ1M GGUF, CPU inference)
- mlx-tune + Metal buffer fixes (set_cache_limit 32GB)
- Chat-fr overfitting analysis → evidence for niche-only strategy

### 4.2 Quantum Router Training
- Synthetic domain-labeled embeddings
- PennyLane parameter-shift rule
- Comparison: VQC accuracy vs classical sigmoid

### 4.3 LAS Conversion
- Qwen3.5-27B (dense, 30h)
- Qwen3.5-35B-A3B (MoE, 40h) — first MoE SNN
- Evaluation: accuracy retention, spike rate, energy estimate

## 5. Experiments

### 5.1 Routing Quality
- Quantum VQC vs classical sigmoid: accuracy, latency, parameter count
- Domain classification F1 per niche domain
- Confidence calibration analysis

### 5.2 Niche LoRA vs Base
- Per-domain benchmark: 35B base vs 35B+LoRA on niche prompts
- Forgetting check across stacks
- Val loss curves showing niche domains need LoRA, known domains don't

### 5.3 SNN Efficiency
- SpikingKiki-27B vs Qwen3.5-27B: accuracy, spike rate, theoretical energy
- SpikingKiki-35B-MoE vs Qwen3.5-35B-A3B: same metrics + MoE routing preservation
- Akida deployment (if hardware available): measured latency + power

### 5.4 End-to-End Pipeline
- Full triple-hybrid: quantum route → SNN/classical inference → Aeon memory → response
- Latency breakdown per component
- Memory accuracy at PI-depth-10

## 6. Results
[PENDING — fill after experiments]

## 7. Discussion
- Quantum advantage: honest assessment (likely no computational advantage at 11 classes, but demonstrates pipeline)
- SNN advantage: real energy reduction for edge deployment
- Classical advantage: production quality for most domains
- When to use which: decision tree based on deployment constraints
- Limitations: VQC simulator only, SNN is pseudo-spiking, Akida deployment not yet validated

## 8. Conclusion
- First triple-hybrid quantum-neuromorphic-classical system for LLM domain routing
- Open-source: all code, training scripts, and niche adapters released
- Future: QPU deployment, Akida physical validation, SpikingBrain-76B weights

## Hardware
- Mac Studio M3 Ultra 512 GB (training, serving, development)
- RTX 4090 24 GB (inference, distillation)
- BrainChip Akida Mini PCIe (planned, $300)

## Reproducibility
- All code at github.com/electron-rare/micro-kiki
- Datasets at huggingface.co/electron-rare/mascarade-*
- Training recipe documented in CLAUDE.md + configs/
