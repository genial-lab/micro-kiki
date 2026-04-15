# Cognitive layer design — 2026-04-15

> **v0.2 note**: this layer was originally specified as part of v0.1, then consolidated into the unified v0.2 plan at `.claude/plans/micro-kiki-v0.2-implementation.md`.

Companion to `2026-04-15-micro-kiki-design.md`. The base design (32 MoE-LoRA stacks + sigmoid meta-router on Qwen3.5-4B) is preserved verbatim. This document specifies the cognitive layer that wraps it: dispatcher, memory palace, negotiator, and anti-bias pipeline. Each component is grounded in a 2025–2026 published paper so we can explain and defend the choices.

Integration point in the main architecture is the per-turn pipeline documented in `README.md` ("Architecture" section). The cognitive layer inserts four steps: dispatcher before the router, Aeon recall before prompt assembly, negotiator between candidate generation and final selection, anti-bias between selection and emit (with Aeon write as the persistence hook).

## 1. Dispatcher (Phase IV)

**Goal.** Collapse the 32-dim sigmoid router output into 7 meta-intents (code, electronics, system, knowledge, creative, chat, security) with zero training cost.

**Design.** A YAML file (`configs/meta_intents.yaml`) maps each domain to one or more meta-intents, with per-intent weights. At runtime the dispatcher takes the router scores, multiplies by the weight matrix, and returns the top meta-intent plus the raw per-stack activations. The router still decides *which* stacks activate; the dispatcher only provides a coarse label used by the negotiator to pick judge and by Aeon to scope recall.

**Rationale.** Training a 7-class classifier on top of the router adds a moving part for no quality gain: the router is already a soft classifier, the dispatcher is just a view. Zero latency, zero forgetting risk.

## 2. Aeon memory palace (Phase VIII)

**Goal.** Persistent, queryable memory across sessions that is spatial (where), temporal (when), and symbolic (what/who).

**Design.** Two indexes sharing the same event store:

- **Atlas** — SIMD-accelerated vector index (HNSW, cosine) over embedding of each memory event. Serves "find me similar turns."
- **Trace** — neuro-symbolic graph (entities → relations → events) with timestamps. Serves "what happened when with whom."

Backends pluggable: native (sqlite + faiss) for dev, Qdrant + Neo4j for production. The recall step runs both indexes in parallel, merges the top-K, and injects the result into the prompt's system section. The write step is async and happens after emit.

**Rationale — Aeon, arxiv 2601.15311.** Pure vector memory is blind to time and relations; pure graph memory is blind to similarity. Aeon's dual-index formulation is state-of-the-art for long-horizon agents (2026) and we adopt it as-is. The 4B base is too small to fake memory inside its 262K context across sessions, so external memory is load-bearing.

## 3. Negotiator (Phase IX)

**Goal.** When 2–4 stacks fire on the same turn, produce a single coherent response instead of a blended mush or a majority-vote winner.

**Design.** Each active stack produces a candidate. The negotiator extracts the arguments (CAMP), runs a structured dissent pass (Catfish), then asks an LLM judge to pick the winner citing evidence. The judge is adaptive:

- **Default**: Qwen3.5-35B-A3B Opus on kxkm-ai (fast, ~150 tok/s).
- **Escalate** to Mistral-Large-Opus on Studio when judge confidence < 0.5, disagreement is structural, or dispatcher says "reasoning"/"security."

**Rationale — CAMP 2604.00085, Catfish 2505.21503.** Majority vote fails under groupthink (all stacks agree for the wrong reason) and ties. CAMP's evidence-based arbitration beats majority on contested prompts; Catfish's structured dissent catches silent consensus. We combine both because they address orthogonal failure modes.

## 4. Anti-bias — KnowBias + RBD (Phase X)

**Goal.** Surface and correct bias introduced by distilled stacks before the response leaves the system.

**Design.** Two-stage:

- **Train-time KnowBias.** Before the first stack is trained we probe the base model on the Hugging Face bias suite and record the top-k biased neurons per category. After every stack is trained we re-probe and fine-tune *only* those neurons on targeted debiasing pairs. Net effect: bias contribution is clamped both at base and stack level.
- **Run-time RBD.** Every candidate response goes through the RBD classifier (a small head over the base's final layer). If flagged, the response is re-generated with DeFrame reframing once; if re-flagged, the turn returns a neutral refusal.

**Rationale — KnowBias 2601.21864, RBD 2505.17100.** Neuron-level debiasing (KnowBias) is cheaper and more durable than RLHF debiasing and doesn't hurt downstream quality. RBD is the 2025 SOTA for runtime detection; it costs one forward pass through a tiny head. Together they cover offline and online.

## Why no future-reasoner stack

Per arxiv 2601.10132, 4B-class LLMs underperform dedicated time-series ML on quantitative forecasting by a wide margin. A dedicated stack would not close the gap. Temporal awareness is a v0.2 concern and will be handled by context injection (real-time clock, calendar, news slice) plus a tools layer, not a new MoE-LoRA stack. Tracked in `docs/plans/v0.2-roadmap.md`.

---

## Citation note

Several arXiv IDs in the 2601–2604 range may require verification — they were sourced from 2026 literature searches and may have been cited before formal publication. Before implementation of each component, verify the paper's existence and current version via arXiv search:

- Aeon (memory palace) — 2601.15311
- CAMP (arbitration) — 2604.00085
- Catfish Agent — 2505.21503
- KnowBias — 2601.21864
- RBD — 2505.17100
- Temporal limits — 2601.10132
- OPLoRA — 2510.13003
- LoRA-Null — 2503.02659
- Subspace Geometry — 2603.02224
- MoLoRA — 2603.15965

If a paper resolves to a different ID or title, update citations in README, CLAUDE.md, and this spec.
