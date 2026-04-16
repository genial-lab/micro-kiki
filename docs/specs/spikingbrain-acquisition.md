# SpikingBrain Acquisition Spec

Story 12 of v0.3 neuroscience plan. Decides how stories 13-16 obtain a
working SpikingBrain checkpoint on the Studio M3 Ultra (512 GB unified
memory).

## 1. Reference

- Paper: *SpikingBrain Technical Report: Spiking Brain-inspired Large
  Models*, Yuqi Pan et al., arXiv:2509.05276 (2025).
- Lab: BICLab, Institute of Automation, Chinese Academy of Sciences.
- Code: https://github.com/BICLab/SpikingBrain-7B (Apache-2.0 repo, code
  only — weights hosted externally).
- Companion note: 36Kr coverage, 100x TTFT speedup on 4M-token context.

Two models are described:

| Model                | Params total | Params active | Base            | Role                 |
|----------------------|--------------|---------------|-----------------|----------------------|
| SpikingBrain-7B      | 7 B          | 7 B           | Qwen2.5-7B-base | linear LLM           |
| SpikingBrain-76B-A12B| 76 B         | 12 B          | Qwen2.5-7B-base | hybrid-linear MoE    |

Both use spike encoding + hybrid attention; the 76B adds MoE.

## 2. Available checkpoints (probe results — 2026-04-14)

### HuggingFace

No official `BICLab/*` organisation on HF. No official SpikingBrain-7B
or SpikingBrain-76B HF model cards. A few community forks of the GitHub
repo exist (`dansasser/`, `miguelgonez/`, `kustomzone/`) but they ship
code only, no weights, no provenance.

### ModelScope (primary distribution channel)

| Repo ID (modelscope.cn/models/...)        | Role                        | Notes                                   |
|-------------------------------------------|-----------------------------|-----------------------------------------|
| `Panyuqi/V1-7B-base`                      | 7B pre-trained              | Official, BICLab first author           |
| `Panyuqi/V1-7B-sft-s3-reasoning`          | 7B chat/SFT reasoning       | Official, preferred for inference       |
| `sherry12334/SpikingBrain-7B-VL`          | 7B vision-language          | Out of scope for v0.3                   |
| `Abel2076/SpikingBrain-7B-W8ASpike`       | 7B quantised W8A8 pseudo-spike | Pseudo-spiking at tensor level, not true SNN |
| `Panyuqi/V1-76B-*`                        | **NOT RELEASED**            | Paper reports metrics, weights withheld |

License on ModelScope repos: not explicitly stated in the GitHub README
(arxiv paper says research use; needs verification against the
ModelScope page ToS before any public redistribution from Studio).
Probe script reports the license field retrieved at run time.

### Key finding

**SpikingBrain-76B weights are not publicly available** as of 2026-04.
The paper reports benchmarks, the GitHub repo is named `SpikingBrain-7B`
and ships 7B artefacts only. The 76B-A12B checkpoint is referenced in
Table 2 of the paper but no download link is published by BICLab.

## 3. Primary path — SpikingBrain-7B chat (fallback becomes default)

Because 76B is unavailable, stories 13-16 target the 7B SFT checkpoint
as the operational artefact, with Studio memory targets scaled down
accordingly.

```bash
# Studio M3 Ultra, fresh clone
pip install modelscope
modelscope download \
  --model Panyuqi/V1-7B-sft-s3-reasoning \
  --local_dir models/spikingbrain-7b-sft
```

- Download size: ~14 GB BF16 (7B x 2 B/param).
- Wall time: ~15-30 min on residential 100 Mbit symmetric.
- Disk: 20 GB headroom (tokeniser + config + shards).
- Peak RAM on load (BF16): ~16 GB — trivial on Studio 512 GB.
- Peak RAM on load (W8ASpike quantised): ~8 GB — suitable for MPS.

### Decision 2026-04-16 — 7B adopted as production path

User approved the primary path above. Phase N-III targets SpikingBrain-7B
SFT as the shippable backbone. In parallel, Phase N-IV runs a **custom
multi-base SNN reproduction** via LAS (arxiv 2505.09659, lossless
ANN→SNN conversion — more modern than Spikingformer) on three bases:
Qwen3.5-27B (dense), Qwen3.5-122B-A10B (MoE), and Mistral-Large-Opus
123B (dense, already fused on Studio). Sections 4-5 below remain as
reference fallbacks; §3 + §4-bis together are the operative plan.

## 4-bis. Custom multi-base reproduction (Phase N-IV, stories 17-29)

Rather than wait on BICLab's 76B release, v0.3 reproduces the spiking
transformer principle across three large bases we already have compute
for:

| Base                       | Arch           | Studio state             | SpikingKiki output        |
|----------------------------|----------------|---------------------------|----------------------------|
| Qwen3.5-27B                | dense          | ~54 GB to fetch from HF   | `SpikingKiki-27B`          |
| Qwen3.5-122B-A10B          | hybrid + MoE   | cached (~244 GB BF16)     | `SpikingKiki-122B-A10B`    |
| Mistral-Large-Opus 123B    | dense          | fused (~233 GB BF16)      | `SpikingKiki-LargeOpus-123B` |

Compute (sequential on Studio, 240+ GB peaks forbid parallel runs):
~210-240 h wall time total. Story 29 cross-evaluates all three + the
7B baseline and picks the release variant. Full design in
`docs/specs/las-conversion-framework.md`.

## 5. Fallback A — 76B via author request

If BICLab releases 76B later (expected window: 2026 Q2-Q3 based on
paper's roadmap language "will be released"), path is:

```bash
modelscope download \
  --model Panyuqi/V1-76B-A12B-sft \
  --local_dir models/spikingbrain-76b
```

- Download size: ~152 GB BF16 (76 B x 2 B/param, dense) — active 24 GB
  during inference (MoE routes 12B/step).
- Wall time: ~4-8 h residential.
- Disk: 180 GB headroom.
- Peak RAM on load (BF16, dense weights resident): ~160 GB — fits in
  Studio 512 GB unified memory with >200 GB spare for activations + KV.
- Peak RAM with W8A8 quantisation: ~80 GB — comfortable.

Gate: re-run the probe script monthly; open an issue on
`BICLab/SpikingBrain-7B` asking for 76B release; email authors
(pan.yuqi@ia.ac.cn referenced in paper).

## 6. Fallback B — Spikingformer training-free conversion

Last-resort path if 76B never ships and even 7B BICLab weights become
unreachable. Spikingformer (Zhou et al., 2023; extended 2024) supports
training-free ANN→SNN conversion for transformer architectures.

- Input: `Qwen/Qwen2.5-7B-Instruct` from HuggingFace (Apache-2.0).
- Tool: `spikingjelly` + Spikingformer conversion script.
- Time: ~2-4 h on Studio M3 Ultra (MPS), 4-8 h CPU-only.
- Output: spiking Qwen2.5-7B approximating SpikingBrain-7B architecture
  (no hybrid-linear attention — vanilla softmax attention instead).

Drawback: the conversion does not reproduce SpikingBrain's hybrid-linear
attention or the continual pre-training on 150B tokens with spike
encoding. Expect ~70-80% of reference accuracy on reasoning benchmarks.

For v0.3 this fallback is marked "research-only" — it keeps the SNN
pipeline exercisable without the BICLab weights but does not ship in
the release artefact.

## 7. Studio environment requirements

Python extras (see story 13 for `pyproject.toml` wiring):

```
torch>=2.5            # MPS support on M3 Ultra
transformers>=4.45
accelerate>=0.34
spikingjelly>=0.0.0.0.14
modelscope>=1.20
safetensors
sentencepiece
tiktoken               # Qwen2.5 tokenizer
```

Disk (absolute minimum):

| Path                                | 7B    | 76B     |
|-------------------------------------|-------|---------|
| `models/spikingbrain-{7b,76b}/`     | 20 GB | 180 GB  |
| HF/ModelScope cache                 | 5 GB  | 20 GB   |
| Inference scratch (activations, KV) | 2 GB  | 40 GB   |
| **Total budget**                    | **27 GB** | **240 GB** |

RAM peaks (Studio M3 Ultra, 512 GB unified):

| Scenario            | 7B BF16 | 7B W8ASpike | 76B BF16 | 76B W8A8 |
|---------------------|---------|-------------|----------|----------|
| Load                | 16 GB   | 9 GB        | 160 GB   | 82 GB    |
| Inference (4k ctx)  | 20 GB   | 12 GB       | 200 GB   | 100 GB   |
| Inference (128k ctx)| 28 GB   | 18 GB       | 260 GB   | 140 GB   |

All scenarios fit. The 76B 128k-context case is the tightest at ~50% of
unified memory — safe when macOS has nothing else running.

## 8. Decision matrix

| Condition                             | Path        | Story 14 target                         |
|---------------------------------------|-------------|-----------------------------------------|
| 76B released on ModelScope            | Primary     | BF16 smoke, ≤480 GB peak                |
| 76B not released, 7B SFT accessible   | Fallback A* | 7B BF16 smoke, ≤20 GB peak              |
| 7B unreachable (region block, ToS)    | Fallback B  | Qwen2.5-7B + Spikingformer conversion   |
| Spikingformer conversion fails        | Abort       | Escalate: reconsider v0.3 scope         |

\* As of 2026-04-14, Fallback A is the default because 76B is not public.
Stories 14-16 acceptance criteria (memory, latency, energy) must be
rewritten against 7B numbers — see `docs/specs/micro-kiki-v0.3-neuroscience.md`
for follow-up amendment.

## 9. Open questions

1. ModelScope ToS / licence for re-hosting the 7B SFT weights inside
   the micro-kiki release artefact on HF? Needs legal check before
   mirroring.
2. Can ModelScope be accessed reliably from Europe without VPN? The
   probe script must capture latency + HTTP status to confirm.
3. Is the W8ASpike quantised variant numerically equivalent to true
   neuromorphic deployment on Akida / Loihi, or is it only a surrogate?
   Affects story 21 Akida simulator acceptance.
4. Will BICLab publish the 76B checkpoint? No public timeline. Probe
   script re-run monthly until release or v0.3 freeze (2026-06-01).
5. Energy benchmark (story 19) assumes true spikes. W8ASpike only
   emulates — record the caveat in N-IV report.

## 10. Probe script

`scripts/probe_spikingbrain_hf.py` queries:

- `https://huggingface.co/api/models?search=spikingbrain` (HF)
- `https://huggingface.co/api/models?author=BICLab` (HF org)
- `https://www.modelscope.cn/api/v1/models?Name=SpikingBrain` (MS)
- `https://www.modelscope.cn/api/v1/models?Name=V1-7B` (MS)
- `https://www.modelscope.cn/api/v1/models?Name=V1-76B` (MS)

and emits `results/spikingbrain-probe.json` with
`{timestamp, hf_repos:[], modelscope_repos:[], decision, notes}`.
