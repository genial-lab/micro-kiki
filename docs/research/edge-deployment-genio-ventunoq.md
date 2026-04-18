# Edge Deployment Scouting — GenioBoard & Arduino VENTUNO Q

**Date**: 2026-04-19
**Status**: scouting / research note (no engagement)
**Scope**: feasibility analysis for porting micro-kiki's Aeon memory predictor + VQC router + LLM serving stack to two candidate edge AI platforms.

## 1. Platforms at a glance

### 1.1 Grinn GenioBoard (MediaTek Genio 700)

- **SoC**: MediaTek Genio 700, 6 nm
- **CPU**: Octa-core (2× Cortex-A78 + 6× Cortex-A55)
- **NPU**: 4 TOPS (5th gen MediaTek NPU, INT8)
- **GPU**: Mali-G57
- **RAM options**: 4 / 8 / 16 GB LPDDR4X
- **OS**: Yocto Linux / Ubuntu / Android
- **Form factor**: SBC, compact, dual M.2
- **Peripherals**: GbE, USB 3.x, CSI/DSI, PCIe, CAN, industrial I/O
- **Family variants**: Genio 360/420 (entry), 520/720 (10 TOPS, 8th gen NPU)

### 1.2 Arduino VENTUNO Q (Qualcomm Dragonwing IQ8 + STM32H5)

- **Dual-brain**: MPU (Qualcomm IQ8-275) + MCU (STM32H5F5)
- **MPU compute**: NPU + CPU + GPU, up to **40 dense TOPS** combined
- **MCU**: STM32H5F5, sub-ms deterministic control, Arduino Core on Zephyr
- **Inter-proc bridge**: RPC (Remote Procedure Call)
- **MPU OS**: Linux (AI workloads)
- **Form factor**: Arduino shield-compatible SBC
- **Availability**: Q2 2026 (dev kits shipping)

## 2. Aeon predictor fit (trivial on both)

- **Aeon LatentMLP footprint**: ~100 K parameters, FP32 numpy, ~400 KB weights, < 2 s for 50-epoch training on 1000 pairs (Mac M5 CPU baseline).
- **Inference latency target**: < 1 ms per `predict_next`.
- **On GenioBoard (Cortex-A78 @ 2+ GHz)**: forward pass should run in ~100-300 µs, well below the target. NPU not needed — CPU suffices.
- **On VENTUNO Q MPU**: same magnitude, additional headroom for larger hidden dims (e.g. 512 hidden, 256 latent) without touching the NPU.
- **Memory budget**: Aeon + `AeonSleep` graph at 10 k turns < 50 MB. Both platforms have GB+ of RAM.

**Conclusion**: Aeon predictor is a non-issue. Either board ports cleanly with `uv venv` + `uv pip install numpy` + copy the `src/memory/aeon_predictor.py` module.

## 3. VQC router fit

- **VQC state**: 6 qubits, 6 StronglyEntanglingLayers, ~108 trainable params, PennyLane simulator.
- **Dependency**: PennyLane requires Python 3.11+ and a simulator backend (`default.qubit` NumPy).
- **Inference latency on CPU**: ~5-20 ms per classification (PennyLane overhead dominates).
- **GenioBoard**: Cortex-A78 x2 — workable. Consider batching multiple VQC evaluations to amortize PennyLane graph setup.
- **VENTUNO Q MPU**: same.

**Hardware quantum acceleration**: neither board has quantum HW, so VQC runs as simulator-only. No gain from dedicated NPU — PennyLane's `default.qubit` is pure NumPy.

**Conclusion**: VQC router is CPU-bound; both platforms handle it. NPU/GPU idle during VQC inference.

## 4. LLM serving fit

This is where the two platforms diverge.

### 4.1 GenioBoard Genio 700 (4 TOPS, 8 GB RAM typical)

- **Feasible models**: Qwen3-1.5B Q4 (~1 GB RAM, ~2-5 tok/s via CPU or NPU bridge), Phi-3-mini Q4, Gemma 2B Q4.
- **micro-kiki 35B-A3B base**: **not feasible** (would need ~20 GB RAM just for Q4 weights; NPU doesn't support MoE routing).
- **Role**: GenioBoard can host a **small fallback model** (1-3B) for local inference when cloud is unavailable.

### 4.2 VENTUNO Q (40 TOPS MPU + real-time MCU)

- **Feasible models**: Qwen3-3B Q4 (~2 GB RAM, ~5-10 tok/s via NPU), Llama3-8B Q4 (~4.5 GB, ~2-5 tok/s).
- **micro-kiki 35B-A3B base**: **still not feasible** (RAM ceiling).
- **Role**: VENTUNO Q can host a mid-sized local model (3-8 B) + leverage the MCU for deterministic actuation, making it a better fit for **embedded industrial agents** (Kill_LIFE, Zacus, KXKM Parallelator with intelligent battery management).

## 5. Strategic fit with micro-kiki product lines

### 5.1 Factory 4 Life / mascarade

- **Architecture today**: mascarade Python stack on VM (`192.168.0.119`) + ESP32 endpoints for actuation.
- **VENTUNO Q fit**: single board replaces VM + ESP32 for field-deployable micro-kiki clusters. One VENTUNO Q = one "mini-factory node". Split: MPU runs mascarade + Aeon + small LLM + VQC router; MCU runs deterministic PLC / motion / sensor loop.
- **Business case**: sell "Factory 4 Life Edge Node" = VENTUNO Q preloaded with Kill_LIFE methodology + Aeon memory + Mistral-7B fallback + MCU safety firmware.

### 5.2 Kill_LIFE / Zacus / KXKM Parallelator

- **Zacus escape room**: MCU-side gate/puzzle logic already lives on ESP32. VENTUNO Q could consolidate the ESP32 + Tower LLM stack into a single board per room. Cost: board ~USD 300-500 vs ESP32 + Tower. Gain: lower latency (no network hop to Tower), self-contained room.
- **KXKM Parallelator 16-ch battery manager**: MCU-side ML prediction + MPU-side data logging + ML model update. VENTUNO Q maps cleanly to the battery-management split.
- **Kill_LIFE methodology node**: one edge node per factory line running the Kill_LIFE decision loops.

### 5.3 GenioBoard variants for low-budget deployments

- **Genio 360 / 420**: sub-$150 BOM for low-volume commercial nodes where MCU companion isn't needed.
- **Genio 720**: comparable TOPS (10) without MCU companion, simpler for pure-AI deployments.
- **Use case**: **embedded demo units** for trade shows, test kitchens at Factory 4 Life, etc.

## 6. Latency projections for Paper A edge appendix

Based on architecture specs (not measured):

| Operation | Mac M5 (baseline) | GenioBoard Genio 700 | VENTUNO Q (MPU side) |
|-----------|-------------------|----------------------|----------------------|
| `AeonPredictor.predict_next` (single) | 200 µs | ~300 µs | ~250 µs |
| `AeonSleep.recall(k=10)` at 10 k turns | 1-2 ms | ~3-5 ms | ~2-4 ms |
| `QuantumRouter.classify` (VQC) | 5 ms | ~10-15 ms | ~8-12 ms |
| Small LLM inference (3B Q4, 10 tokens) | 2 s | ~5-10 s (CPU only) | ~1-2 s (NPU) |
| Full turn (ingest + predict + recall + LLM 3B) | ~2.2 s | ~5-10 s | ~1-2 s |

**Honest caveat**: these are extrapolations from public specs + known benchmarks of similar ARM Cortex-A cores and Qualcomm IQ8 early reviews. Real numbers require dev kits.

## 7. Recommendation (scouting level)

1. **Aeon + VQC core**: portable to either board, no architectural blocker.
2. **LLM serving**: VENTUNO Q wins at mid-size (3-8 B local) and dual-brain integration; GenioBoard is the budget option for 1-3 B models.
3. **Factory 4 Life / industrial deployment**: VENTUNO Q is the better strategic bet — dual-brain matches existing mascarade + ESP32 split natively.
4. **Paper A benchmark appendix**: projected latency table above (Section 6) can be added as Appendix B ("Edge deployment projections") with the explicit caveat that numbers are extrapolated, not measured.
5. **Next step if serious**: order one dev kit each (~€600-1000 combined) for a real latency + power benchmark. Budget 2-4 weeks of engineering to port Aeon + small LLM + measure.

## 8. Risks and open questions

- **PennyLane on MediaTek Genio**: not a tested combination in the wild. May require building from source; aarch64 wheels exist for PennyLane-Lightning, which is faster than `default.qubit`.
- **VENTUNO Q Q2 2026**: dev kits available from Q2 but supply + software maturity still unknown.
- **Qualcomm IQ8 NPU runtime**: Qualcomm's QNN SDK is the official path; ONNX Runtime + QNN Execution Provider works for many models but quantization support is model-dependent.
- **Commercial licensing**: Arduino VENTUNO Q at ~USD 500-800 retail; GenioBoard SBCs around USD 200-300. Economics work for industrial, tight for consumer.

## 9. References

- [MediaTek Genio 700 product page](https://genio.mediatek.com/genio-700)
- [Grinn GenioBoard article](https://www.mediatek.com/tek-talk-blogs/grinn-genioboard-edge-ai-sbc-powered-by-mediatek-genio-700)
- [Genioboard SBC Embed coverage](https://www.embedsbc.com/genioboard-mediatek-genio-sbc/)
- [Arduino VENTUNO Q product page](https://www.arduino.cc/product-ventuno-q/)
- [VENTUNO Q dual-brain on Electronics-Lab](https://www.electronics-lab.com/meet-the-arduino-ventuno-q-a-dual-brain-architecture-on-a-single-board/)
- [Arduino VENTUNO Q Embedded World 2026 first look (SBCwiki)](https://sbcwiki.com/news/articles/arduino-ventuno-q-first-look-ew26/)
- [Edge AI Vision Alliance coverage](https://www.edge-ai-vision.com/2026/03/arduino-announces-arduino-ventuno-q-powered-by-qualcomm-dragonwing-iq8-series/)
