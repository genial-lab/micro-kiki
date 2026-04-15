# micro-kiki — project context for Claude Code

32 domain experts (MoE-LoRA) on Qwen3.5-4B base with a cognitive layer (memory palace + negotiator + anti-bias). Fits RTX 4090 24 GB.

## Key decisions locked in

- **Base model**: Qwen3.5-4B (Apache 2.0, 262K ctx, GatedDeltaNet hybrid)
- **Base architecture mod**: Differential Attention (arxiv 2410.05258) applied only to the 13 full-attention layers (the 36 linear/GatedDeltaNet layers left untouched) — follows Dragon LLM's per-global-layer finding; benefits long ctx, reduces hallucinations, reduces activation outliers (helps Q4 quant)
- **Adapter technique**: MoE-LoRA (MoLoRA approach), rank 16, 4 experts per projection, top-2 routing
- **Init strategy**: PiSSA default, LoRA-Null alternative, OPLoRA projection for forgetting prevention
- **Quantization**: Q4_K_M for serving, BF16 for training
- **Router**: sigmoid meta-router, 32 outputs, threshold 0.12, chat floor 0.20, max 4 active stacks
- **Dispatcher**: training-free YAML mapping from router output to 7 meta-intents
- **Memory**: Aeon (Atlas SIMD + Trace graph), native or Qdrant/Neo4j backends
- **Negotiator**: CAMP arbitration + Catfish dissent; adaptive judge (Qwen3.5-35B fast / Mistral-Large deep)
- **Anti-bias**: KnowBias neuron debiasing before+after stacks + RBD runtime detector
- **Serving**: vLLM with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` (kxkm-ai) or mlx-lm (Mac Studio)

## Implementation plan

- `.claude/plans/micro-kiki-v0.2-implementation.md` — 106 stories, 14 phases
- Driven by ralph loop: `uv run .ralph/loop.py`
- Each story = one shippable unit with explicit files-to-touch + acceptance + dependencies
- Phase XIII (steps 100–102) adds quantum-inspired techniques (HyQuT hybrid VQC, QMoE routing with classical fallback, Quantum-PEFT adapters); all three run on classical simulators by default — QPU dispatch is optional and lives on the `quantum` branch

## Don't

- Don't change the base model without updating all specs (base is baked into OPLoRA projections, bias neuron IDs, etc.)
- Don't drop below Q4 quantization (quality cliff)
- Don't route > 4 stacks simultaneously (VRAM exceeds 24 GB)
- Don't train stacks in parallel on the same GPU (interference)
- Don't skip the forgetting check framework (step 14) — activating AFTER stacks 02-03 means those two are baseline-checked retroactively
- Don't add a future-reasoner stack — LLM 4B underperforms time-series ML; use tools layer in v0.2 instead

## Do

- Train stacks sequentially, curriculum order (foundations first)
- Run forgetting check after EACH new stack trained; rollback if angle < 30° AND win-rate drop > 0.03
- Keep per-domain datasets disjoint (dedup enforces this, step 6)
- Test meta-router after every 4 stacks added
- Use the adaptive judge: cheap Qwen35B first, escalate to Mistral-Large-Opus only if confidence < 0.5
- Run RBD on every response; only re-generate via DeFrame if bias flagged

## Commit conventions

- `feat(<phase>): <short imperative>` — new code
- `docs(<area>): <short imperative>` — docs
- `fix(<area>): <short imperative>` — bug fix
- Subject ≤ 50 chars (pre-commit hook enforces)
- Body lines ≤ 72 chars (hook enforces; wrap prose manually)
- Scope without dots (use `quantum` not `v0.2`)
- No `Co-Authored-By` trailer (hook rejects it)
- Large diffs (> 734 LOC) require body with `## Context`,
  `## Approach`, `## Changes`, `## Impact` sections
  (see `.ralph/CLAUDE.md` for the template)

## External resources used

- HuggingFace: `Qwen/Qwen3.5-4B` (base), `bofenghuang/mt-bench-french`, `manu` FrenchBench collection
- Papers: MoLoRA 2603.15965, OPLoRA 2510.13003, LoRA-Null 2503.02659, Aeon 2601.15311, CAMP 2604.00085, Catfish 2505.21503, KnowBias 2601.21864, RBD 2505.17100
- Teachers served at: Mistral-Large-Opus (Studio), Qwen3.5-35B-A3B Opus (kxkm-ai :8000)
