# Branch `neuroscience`

v0.3 neuroscience-edition of micro-kiki. Research-grade cousin fork of v0.2:
- **SpikingBrain-76B MoE** as base (not Qwen3.5-4B)
- **AeonSleep** unified memory (fuses Aeon + SleepGate)
- **MAP-validated** cognitive layer (retrospective analysis of v0.2)
- **Neuromorphic edge** deployment targets (Akida PCIe + ESP32-S3 stretch)

## Workflow

- Experiment in this branch
- Learnings feed back to v0.2 if successful (e.g., AeonSleep consolidation could be backported to main's Phase VIII)
- Independent release cycle: aims for HuggingFace publish as `micro-kiki-v0.3-neuroscience`

## Hardware

- **Studio M3 Ultra 512 GB**: primary training + inference (SpikingBrain-76B fits in BF16)
- **RTX 4090 (kxkm-ai)**: Akida Mini PCIe host + driver testing
- **Akida Mini PCIe ~$300**: to be ordered once simulator validation passes
- **ESP32-S3**: STRETCH goal (custom SNN Xtensa port, optional)

## Dependencies (add progressively)

- spikingformer (Python lib)
- MLX or PyTorch (depends on SpikingBrain checkpoint format)
- BrainChip Akida SDK
- Intel Loihi 2 simulator (KAPOHO)
- Custom Xtensa SNN (Zacus firmware derivative)

## Papers

- MAP — Nature Communications 2025 (s41467-025-63804-5)
- SleepGate — arxiv 2603.14517
- SpikingBrain — arxiv 2509.05276
- BriLLM — OpenReview 2026
- Spikingformer — AAAI 2026

## Relationship to other branches/repos

- **main** (v0.2): stable core, don't break
- **quantum**: hybrid classical/QPU, orthogonal
- **micro-kiki-quantum** (private): true QPU research, orthogonal

v0.3 is its own world. Synchronize with main only via explicit merge of validated components.
