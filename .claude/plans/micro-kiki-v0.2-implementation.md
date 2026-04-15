# micro-kiki v0.2 — Implementation Plan

**Scope**: Build 32 domain-expert MoE-LoRA stacks on Qwen3.5-4B base, with sigmoid meta-router, end-to-end distillation pipeline, triple-device serving (RTX 4090 + Mac Studio + optional ANE), plus a **quantum-inspired pre-release overlay** (CompactifAI base compression, QTHA adapter pilot, tensor-network router prototype — 100% classical execution, no QPU).

**Derived from**: `docs/specs/2026-04-15-micro-kiki-design.md` + `docs/specs/2026-04-15-cognitive-layer-design.md` + `docs/specs/2026-04-15-micro-kiki-v0.2-quantum-inspired.md` + `docs/research/micro-kiki-moe-research.md`.

**Conventions**:
- Each step is ONE shippable unit of work — can be committed independently.
- Steps run sequentially unless marked `(parallel)`.
- Acceptance = what "done" looks like (tests / eval score / artifact exists).
- Dependencies = which prior step MUST be complete first.
- **Domain order locked** per design-doc curriculum: 1 chat-fr, 2 reasoning, 3 python, 4 typescript, 5 cpp, 6 rust, 7 html-css, 8 shell, 9 sql, 10 yaml-json, 11 docker, 12 kicad-dsl, 13 spice, 14 lua-upy, 15 embedded, 16 stm32, 17 iot, 18 freecad, 19 platformio, 20 power, 21 emc, 22 dsp, 23 spice-sim, 24 electronics, 25 kicad-pcb, 26 web-frontend, 27 web-backend, 28 music-audio, 29 devops, 30 llm-orch, 31 math, 32 security.

---

## Implementation Steps

### Phase I — Foundations (bootstrap)

1. **Download + verify Qwen3.5-4B base**
   - Files to touch: `scripts/download_base.py`, `models/qwen3.5-4b/` (gitignored)
   - Download from HuggingFace `Qwen/Qwen3.5-4B` (or mirror), verify SHA256, save BF16 safetensors at `models/qwen3.5-4b/bf16/`.
   - Also produce Q4_K_M GGUF at `models/qwen3.5-4b-q4.gguf` via `llama.cpp/convert_hf_to_gguf.py` + `llama-quantize`.
   - Acceptance: both files exist, loader smoke test passes (prompt "hello" → non-empty response), file sizes within ±5% of expected (~8 GB BF16, ~2.5 GB Q4).
   - Dependencies: none.

2. **Fork Qwen3.5-4B with Differential Attention on full-attn layers**
   - Files to touch: `scripts/fork_qwen_diffattn.py`, `src/base/diff_attention.py`, `models/qwen3.5-4b-diffattn/` (gitignored), `tests/test_diff_attention.py`, `docs/specs/diffattn-integration.md`
   - Implement the differential attention mechanism from arxiv 2410.05258 (ICLR 2025, Microsoft): attention scores = softmax(Q1·K1) − λ·softmax(Q2·K2) where λ is a learnable scalar per head, initialized to reinit_lambda * (depth_pre_train).
   - Apply **only to the 13 `full_attention` layers** of Qwen3.5-4B (leave the 36 `linear_attention` / GatedDeltaNet layers untouched). Rationale: Dragon LLM's empirical finding (substack post) is that DiffAttn applied to global attention only yields meaningful gains while per-layer application is marginal.
   - Approach: fork the Qwen3.5-4B weights; for each full_attn layer, duplicate Q and K projections (Q1/Q2, K1/K2); init Q2/K2 with small perturbation of Q1/K1 for warm start (avoid full re-train).
   - Short SkyLadder-style calibration pass on a small sample (~5K tokens) to stabilize λ — NOT a full retrain. Duration: ~30 min on 4090.
   - Acceptance: `models/qwen3.5-4b-diffattn/` exists with same file structure as base; forward pass matches within 2% perplexity on a 1K-sample held-out set; activation outliers (max/mean ratio per layer) reduced by ≥ 30% on full-attn layers vs vanilla Qwen3.5-4B; write spec documenting the exact implementation choices.
   - Dependencies: step 1.

3. **Write base model loader (`src/base/loader.py`)**
   - Files to touch: `src/base/loader.py`, `src/base/__init__.py`, `tests/test_loader.py`
   - Class `BaseModelLoader` with methods `load_bf16()`, `load_q4()`, `enable_lora_switching()` (Unsloth + PEFT hooks).
   - Expose context manager `with_stack(adapter_name)` that hot-swaps the LoRA weights.
   - Acceptance: `pytest tests/test_loader.py::test_load_and_switch` passes.
   - Dependencies: step 1.

4. **Write teacher client (`src/distill/teacher_client.py`)**
   - Files to touch: `src/distill/teacher_client.py`, `src/distill/__init__.py`, `tests/test_teacher_client.py`
   - OpenAI-compatible client wrapping: Mistral-Large-Opus (Studio mlx-lm server), Qwen3.5-122B-A10B Opus-v3 (Studio), Qwen3.5-35B-A3B Opus (kxkm-ai :8000), Devstral-v3/v4 (kxkm-ai).
   - Features: async HTTP via httpx, exponential backoff retry (3 attempts), on-disk response cache keyed by SHA256(prompt + model + params), Qwen3 thinking-mode toggle (`enable_thinking=False` for scoring).
   - Endpoints configurable via env vars `TEACHER_MISTRAL_URL`, `TEACHER_QWEN122_URL`, etc.
   - Acceptance: `pytest tests/test_teacher_client.py` — mocks HTTP, verifies cache hit, verifies retry on 500.
   - Dependencies: none.

5. **Smoke-test harness (`tests/conftest.py`)**
   - Files to touch: `tests/conftest.py`, `tests/test_smoke.py`
   - Pytest fixtures: `tmp_model_dir`, `mock_teacher`, `sample_prompts`.
   - Smoke tests: 1 per module (`test_loader_smoke`, `test_teacher_smoke`).
   - Acceptance: `uv run pytest` runs cleanly, all smoke tests green.
   - Dependencies: steps 3, 4.

### Phase II — Data pipeline

6. **Distilled dataset generator (`src/distill/generator.py`)**
   - Files to touch: `src/distill/generator.py`, `tests/test_generator.py`
   - Function `generate_examples(prompts: list[str], teacher: TeacherClient, n_per_prompt: int = 1) -> Dataset`.
   - Emits `jsonl` with `{prompt, completion, teacher_model, domain, hash}` rows.
   - Supports resume from checkpoint (scan existing jsonl, skip already-done hashes).
   - Acceptance: generates 10 examples from mock teacher, output jsonl valid, resume test passes.
   - Dependencies: step 4.

7. **Cross-domain dedup (`src/distill/dedup.py`)**
   - Files to touch: `src/distill/dedup.py`, `tests/test_dedup.py`
   - MinHash + LSH across ALL domain jsonls, flag examples appearing in > 1 domain, assign to highest-affinity domain only.
   - CLI: `python -m src.distill.dedup --input data/raw/ --output data/dedup/`
   - Acceptance: on synthetic dup set (3 domains, 30% overlap), produces disjoint partition; unit tests pass.
   - Dependencies: step 6.

8. **Audit data sources — write `docs/data-sources.md` (no download)**
   - Files to touch: `docs/data-sources.md`
   - For each of the 32 domains (in design-doc order), add a table row listing: (a) candidate local datasets under `~/Documents/Projets/Factory 4 Life/` (especially `KIKI-models-tuning/`), (b) existing HuggingFace datasets worth considering, (c) an `availability` column checked via `ls` / `find` (CONFIRMED / TBD / GAP), (d) any notes on licensing or language.
   - For French-language domains (chat-fr especially), explicitly reference these HF collections known to exist as of 2026:
     - `bofenghuang/mt-bench-french` — MT-Bench translation
     - `manu` HF user collection: FrenchBench evaluation datasets
     - Community 'OpenAssistant-FR' may or may not have clean HF mirror — verify
   - The data-sources.md file must list these specific HF IDs with a CONFIRMED / TBD column and licensing note.
   - NO download or fetch from HF. This is an inventory + planning step only.
   - Full curation of the `kiki-*` legacy datasets (actual import + dedup) is deferred to steps 37–46.
   - Acceptance: `docs/data-sources.md` exists with a header row and 32 data rows (one per domain); ≥ 70% of domains (≥ 23) have a CONFIRMED source; each GAP row has a one-line mitigation note (synthetic via teacher, scraping, manual seed, etc.); FR HF IDs listed with CONFIRMED/TBD + license.
   - Dependencies: none.

9. **Generate first distilled dataset (chat-fr, 2K examples)**
   - Files to touch: `data/prompts/chat-fr.jsonl`, `data/distilled/chat-fr.jsonl`, `scripts/distill_chat_fr.py`
   - Seed prompts (300–500) curated from the sources identified in step 8 for domain 1 (MTBench-FR, OpenAssistant-FR, manual).
   - Teacher: Mistral-Large-Opus (Studio). Target: 2000 completed examples.
   - Command: `uv run scripts/distill_chat_fr.py --teacher mistral-large-opus --n 2000 --out data/distilled/chat-fr.jsonl`
   - Acceptance: `wc -l data/distilled/chat-fr.jsonl == 2000`, random-sample manual quality check (5/5 OK), French ratio ≥ 95%.
   - Dependencies: steps 6, 8, Studio teacher reachable.

### Phase III — First stack (stack-01 chat-fr, prove E2E)

10. **MoE-LoRA stack trainer (`src/stacks/trainer.py`)**
   - Files to touch: `src/stacks/trainer.py`, `src/stacks/moe_lora.py`, `src/stacks/oplora.py`, `tests/test_trainer.py`
   - MoLoRA-style: 4 LoRA experts per attention projection (q, k, v, o), rank 16, top-2 routing per token, softmax gate.
   - **OPLoRA initialization (recommended)**: to prevent catastrophic forgetting across stacks, support Orthogonal Projection LoRA (arxiv 2510.13003, Oct 2025). `src/stacks/oplora.py` exposes orthogonal projection utilities that initialize each expert's A/B matrices in a subspace orthogonal to previously trained stacks' updates. Config lets each stack choose among: `molora-vanilla`, `oplora`, or `molora+oplora` hybrid (OPLoRA applied per-expert inside the MoLoRA mixture). Default for stacks ≥ 04 is `molora+oplora`.
   - Uses HuggingFace `peft` + custom `MoLoraConfig`. Training via `trl.SFTTrainer`.
   - Hyperparams from YAML config. Saves adapters to `outputs/stacks/stack-XX-<name>/`.
   - Acceptance: `pytest tests/test_trainer.py::test_forward_pass` — 1 step with dummy data, loss finite; `pytest tests/test_trainer.py::test_oplora_orthogonality` — OPLoRA init produces updates with cosine < 0.1 vs a fixed prior-stack subspace.
   - Dependencies: step 3.

11. **First stack config (`configs/stack-01-chat-fr.yaml`)**
    - Files to touch: `configs/stack-01-chat-fr.yaml`
    - Fields: `base_model: models/qwen3.5-4b-diffattn/` (DiffAttn fork from step 2; fallback `models/qwen3.5-4b/` if DiffAttn artifact missing), `num_experts: 4`, `lora_rank: 16`, `lora_alpha: 32`, `top_k: 2`, `learning_rate: 2e-4`, `batch_size: 4`, `grad_accum: 8`, `epochs: 3`, `seq_len: 4096`, `dataset: data/distilled/chat-fr.jsonl`.
    - Initialization options (add to YAML):
      ```yaml
      init_lora_weights: pissa   # options: pissa | lora-null | default
      pissa_niter: 4
      # lora-null: init in null space of activations (prevents pre-trained knowledge loss).
      #   Reference: arxiv 2503.02659 (LoRA-Null, Feb 2026)
      ```
    - Acceptance: file loads via `yaml.safe_load` in test, required keys present (including `init_lora_weights`).
    - Dependencies: step 10.

12. **Train stack-01 (chat-fr)**
    - Files to touch: `outputs/stacks/stack-01-chat-fr/` (gitignored, artifact-only)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-01-chat-fr.yaml`
    - Target: ~30 min on 4090 or ~15 min on Studio MLX.
    - Acceptance: final train loss < 1.5, eval loss (held-out 5% split) < 1.8, adapter files saved (< 200 MB).
    - Dependencies: steps 9, 10, 11.

13. **Per-stack eval harness (`src/eval/stack_eval.py`)**
    - Files to touch: `src/eval/stack_eval.py`, `data/eval/chat-fr.jsonl` (held-out 100 prompts + judge criteria)
    - Loads base + stack adapter, generates responses, LLM-judges with Mistral-Large-Opus (pairwise vs base-only).
    - Outputs score JSON: `{stack, n_prompts, win_rate_vs_base, avg_judge_score, sample_responses}`.
    - Acceptance: run on stack-01 produces valid JSON, win_rate > 0.55 (stack better than base).
    - Dependencies: step 12.

14. **Evaluate + commit stack-01 baseline**
    - Files to touch: `results/stack-01-baseline.json`, update `README.md` status section
    - Run eval, archive results, log tokens/sec at inference, commit adapter metadata (not weights) to git.
    - Acceptance: `results/` dir exists, JSON committed, README `- [x] First stack trained`.
    - Dependencies: step 13.

### Phase IV — Meta-router v0 (3 stacks)

15. **Dispatcher mapping (training-free meta-intents)**
    - Files to touch: `src/routing/dispatcher.py`, `configs/meta_intents.yaml`, `tests/test_dispatcher.py`
    - No new model. Derives 7 meta-intents from the 32-dim router sigmoid output via a YAML mapping table: `{quick-reply, coding, reasoning, creative, research, agentic, tool-use} → lists of domain indices`.
    - Function `dispatch(router_logits: Tensor) -> MetaIntent`: returns the dominant meta-intent for logging + downstream tool-use hints. Does NOT replace the router — it annotates the request.
    - Latency: zero (lookup + argmax). Zero additional VRAM.
    - Acceptance: `tests/test_dispatcher.py::test_mapping` — 10 synthetic sigmoid vectors produce expected meta-intents; YAML validates against a schema (each domain index 1..32 appears in exactly one bucket).
    - Dependencies: step 14 (router v0 already trained in step 19 post-shift; we keep it decoupled — dispatcher reads ANY 32-dim vector, testable with stub).

16. **Forgetting check framework (`src/eval/forgetting.py`)**
    - Files to touch: `src/eval/forgetting.py`, `scripts/run_forgetting.sh`
    - For each already-trained stack, run its eval set with the NEW stack loaded (active-swap test). Flag delta > 0.03 in win_rate_vs_base.
    - **Gradient subspace overlap metric** (2nd signal): in addition to win-rate delta, measure the geometric angle between the NEW stack's LoRA update gradient subspace and each prior stack's subspace. Small angle = high interference = forgetting risk. Reference: arxiv 2603.02224 (Subspace Geometry, 2026).
    - Output JSON per prior stack includes both `win_rate_delta` and `gradient_subspace_angle_deg`.
    - Meant to run automatically after EACH training step from now on (wrap `src.stacks.trainer` CLI or emit a shell hook).
    - Acceptance: script runs after any trained stack and emits `results/forgetting-stack-XX.json`; no stack has `gradient_subspace_angle_deg < 30°` AND `win_rate_delta > 0.03` simultaneously (either one alone is tolerable).
    - Dependencies: step 13.

17. **Train stack-02 (reasoning)**
    - Files to touch: `configs/stack-02-reasoning.yaml`, `data/prompts/reasoning.jsonl`, `data/distilled/reasoning.jsonl`, `data/eval/reasoning.jsonl`, `outputs/stacks/stack-02-reasoning/` (gitignored)
    - Seed prompts from GSM8K trainset + ARC-easy (design-doc domain 2). Distill 2K examples via Qwen3.5-35B Opus (Qwen3 thinking-mode teacher).
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-02-reasoning.yaml`
    - Duration: ~45 min on 4090 (or ~20 min on Studio MLX).
    - Acceptance: win_rate_vs_base ≥ 0.55 on `data/eval/reasoning.jsonl` (50-sample GSM8K subset); forgetting check passes (step 16 gates): no regression > 0.03 on stack-01 AND gradient-angle ≥ 30°.
    - Dependencies: steps 10, 13, 16 (forgetting framework).

18. **Train stack-03 (python)**
    - Files to touch: `configs/stack-03-python.yaml`, `data/prompts/python.jsonl`, `data/distilled/python.jsonl`, `data/eval/python.jsonl`, `outputs/stacks/stack-03-python/` (gitignored)
    - Seed prompts from CodeAlpaca + StackOverflow Python top-tagged. Teacher: Devstral-v3/v4 (kxkm-ai).
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-03-python.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on HumanEval 50-subset; forgetting check (step 16) passes against stacks 01–02.
    - Dependencies: steps 10, 13, 16, 17.

19. **Meta-router module (`src/routing/router.py`)**
    - Files to touch: `src/routing/router.py`, `src/routing/__init__.py`, `tests/test_router.py`
    - Architecture: extract base model last-hidden-state at layer N (N=16 for 4B), pool (mean + attention-query), feed to Linear(3072→512) → LayerNorm → GELU → Linear(512→32) → sigmoid.
    - ~2M params total. Training target: multi-label outcome discovery (binary per-domain, threshold 0.12, chat floor 0.20).
    - Acceptance: `test_router_forward` — dummy batch, outputs shape [B, 32], all in [0,1]; `test_router_init` — trainable params < 3 M.
    - Dependencies: step 3.

20. **Train router v0 on 3-stack mix**
    - Files to touch: `scripts/train_router.py`, `outputs/router/v0/router.pt`
    - Dataset: concat 3 domain jsonls (chat-fr, reasoning, python), label each example with its domain-id; also generate 1K "mixed-intent" examples (queries that span 2 domains) for multi-label training.
    - Loss: BCEWithLogitsLoss, train 3 epochs, lr 1e-3.
    - Acceptance: macro-F1 ≥ 0.85 on 3-domain eval; router file < 10 MB.
    - Dependencies: steps 14, 18, 19.

21. **Runtime adapter switcher (`src/serving/switchable.py`)**
    - Files to touch: `src/serving/switchable.py`, `tests/test_switchable.py`
    - Class `SwitchableModel`: holds base, registry of loaded adapters, method `apply_stacks(names: list[str])` — merges selected LoRAs (weighted sum or sequential application).
    - Cap active stacks at 4. Cache merged weights until stack-set changes.
    - Acceptance: switches between stack-01 ↔ stack-03 ↔ none in < 500 ms; generation before/after differs measurably.
    - Dependencies: steps 12, 18.

22. **End-to-end smoke test (3 stacks)**
    - Files to touch: `tests/test_e2e_3stacks.py`, `scripts/demo_e2e.py`
    - Prompt → router → top-2 sigmoid scores → SwitchableModel.apply_stacks → generate → print.
    - Test cases: "écris un haïku" → chat-fr dominant; "factor 231 into primes" → reasoning dominant; "python fastapi hello world" → python dominant.
    - Acceptance: all 3 test cases have expected domain in top-2 router picks; generation coherent (no garbled tokens).
    - Dependencies: steps 20, 21.

### Phase V — Curriculum coding core + secondary (stacks 04–14)

23. **Train stack-04 (typescript)**
    - Files to touch: `configs/stack-04-typescript.yaml`, `data/distilled/typescript.jsonl`, `data/eval/typescript.jsonl`, `outputs/stacks/stack-04-typescript/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-04-typescript.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against stacks 01–03.
    - Dependencies: steps 10, 18, 16.

24. **Train stack-05 (cpp)**
    - Files to touch: `configs/stack-05-cpp.yaml`, `data/distilled/cpp.jsonl`, `data/eval/cpp.jsonl`, `outputs/stacks/stack-05-cpp/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-05-cpp.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 23, 16.

25. **Train stack-06 (rust)**
    - Files to touch: `configs/stack-06-rust.yaml`, `data/distilled/rust.jsonl`, `data/eval/rust.jsonl`, `outputs/stacks/stack-06-rust/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-06-rust.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 24, 16.

26. **Train stack-07 (html-css)**
    - Files to touch: `configs/stack-07-html-css.yaml`, `data/distilled/html-css.jsonl`, `data/eval/html-css.jsonl`, `outputs/stacks/stack-07-html-css/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-07-html-css.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 25, 16.

27. **Train stack-08 (shell)**
    - Files to touch: `configs/stack-08-shell.yaml`, `data/distilled/shell.jsonl`, `data/eval/shell.jsonl`, `outputs/stacks/stack-08-shell/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-08-shell.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 26, 16.

28. **Re-train router v1 after stack 08**
    - Files to touch: `outputs/router/v1/router.pt`, `results/router-v1.json`
    - Re-run the router training script (step 20) with the 8-domain set; generate fresh mixed-intent examples covering new pairings.
    - Acceptance: macro-F1 ≥ 0.80 on 8-domain held-out eval; router still < 10 MB.
    - Dependencies: step 27.

29. **Train stack-09 (sql)**
    - Files to touch: `configs/stack-09-sql.yaml`, `data/distilled/sql.jsonl`, `data/eval/sql.jsonl`, `outputs/stacks/stack-09-sql/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-09-sql.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 28, 16.

30. **Train stack-10 (yaml-json)**
    - Files to touch: `configs/stack-10-yaml-json.yaml`, `data/distilled/yaml-json.jsonl`, `data/eval/yaml-json.jsonl`, `outputs/stacks/stack-10-yaml-json/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-10-yaml-json.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 29, 16.

31. **Train stack-11 (docker)**
    - Files to touch: `configs/stack-11-docker.yaml`, `data/distilled/docker.jsonl`, `data/eval/docker.jsonl`, `outputs/stacks/stack-11-docker/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-11-docker.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 30, 16.

32. **Train stack-12 (kicad-dsl)**
    - Files to touch: `configs/stack-12-kicad-dsl.yaml`, `data/distilled/kicad-dsl.jsonl`, `data/eval/kicad-dsl.jsonl`, `outputs/stacks/stack-12-kicad-dsl/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-12-kicad-dsl.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 31, 16.

33. **Re-train router v2 after stack 12**
    - Files to touch: `outputs/router/v2/router.pt`, `results/router-v2.json`
    - 12-domain set + mixed-intent refresh.
    - Acceptance: macro-F1 ≥ 0.80 on 12-domain held-out eval; router still < 10 MB.
    - Dependencies: step 32.

34. **Train stack-13 (spice)**
    - Files to touch: `configs/stack-13-spice.yaml`, `data/distilled/spice.jsonl`, `data/eval/spice.jsonl`, `outputs/stacks/stack-13-spice/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-13-spice.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 33, 16.

35. **Train stack-14 (lua-upy)**
    - Files to touch: `configs/stack-14-lua-upy.yaml`, `data/distilled/lua-upy.jsonl`, `data/eval/lua-upy.jsonl`, `outputs/stacks/stack-14-lua-upy/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-14-lua-upy.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 34, 16.

36. **Re-train router v3 after stack 14**
    - Files to touch: `outputs/router/v3/router.pt`, `results/router-v3.json`
    - 14-domain set + mixed-intent refresh.
    - Acceptance: macro-F1 ≥ 0.80 on 14-domain held-out eval; router still < 10 MB.
    - Dependencies: step 35.

### Phase VI — Technical domains (stacks 15–25)

37. **Curate dataset for domain-15 (embedded)**
    - Files to touch: `scripts/import_kiki_datasets.py` (create once, reused by steps 38–46), `data/distilled/embedded.jsonl`
    - Source: `~/Documents/Projets/Factory 4 Life/KIKI-models-tuning/` (or HF mirror `L-electron-Rare/KIKI-models-tuning`) — specifically the `kiki-embedded` or related dataset.
    - Run: `uv run scripts/import_kiki_datasets.py --domain embedded`
    - After import, run dedup: `python -m src.distill.dedup --input data/raw/embedded.jsonl --output data/distilled/embedded.jsonl`.
    - Acceptance: `data/distilled/embedded.jsonl` has ≥ 800 rows; dedup report shows < 5% cross-domain contamination.
    - Dependencies: step 7 (dedup); step 8 (audit).

38. **Curate dataset for domain-16 (stm32)**
    - Files to touch: `data/distilled/stm32.jsonl`
    - Source: `~/Documents/Projets/Factory 4 Life/KIKI-models-tuning/` (or HF mirror `L-electron-Rare/KIKI-models-tuning`) — specifically the `kiki-stm32` or related dataset.
    - Run: `uv run scripts/import_kiki_datasets.py --domain stm32` (reuses script from step 37).
    - After import, run dedup: `python -m src.distill.dedup --input data/raw/stm32.jsonl --output data/distilled/stm32.jsonl`.
    - Acceptance: `data/distilled/stm32.jsonl` has ≥ 800 rows; dedup report shows < 5% cross-domain contamination.
    - Dependencies: step 7 (dedup); step 8 (audit); step 37 (previous curation step, for script reuse).

39. **Curate dataset for domain-17 (iot)**
    - Files to touch: `data/distilled/iot.jsonl`
    - Source: `~/Documents/Projets/Factory 4 Life/KIKI-models-tuning/` (or HF mirror `L-electron-Rare/KIKI-models-tuning`) — specifically the `kiki-iot` or related dataset.
    - Run: `uv run scripts/import_kiki_datasets.py --domain iot`
    - After import, run dedup: `python -m src.distill.dedup --input data/raw/iot.jsonl --output data/distilled/iot.jsonl`.
    - Acceptance: `data/distilled/iot.jsonl` has ≥ 800 rows; dedup report shows < 5% cross-domain contamination.
    - Dependencies: step 7; step 8; step 38.

40. **Curate dataset for domain-18 (freecad)**
    - Files to touch: `data/distilled/freecad.jsonl`
    - Source: `~/Documents/Projets/Factory 4 Life/KIKI-models-tuning/` (or HF mirror `L-electron-Rare/KIKI-models-tuning`) — specifically the `kiki-freecad` or related dataset.
    - Run: `uv run scripts/import_kiki_datasets.py --domain freecad`
    - After import, run dedup: `python -m src.distill.dedup --input data/raw/freecad.jsonl --output data/distilled/freecad.jsonl`.
    - Acceptance: `data/distilled/freecad.jsonl` has ≥ 800 rows; dedup report shows < 5% cross-domain contamination.
    - Dependencies: step 7; step 8; step 39.

41. **Curate dataset for domain-19 (platformio)**
    - Files to touch: `data/distilled/platformio.jsonl`
    - Source: `~/Documents/Projets/Factory 4 Life/KIKI-models-tuning/` (or HF mirror `L-electron-Rare/KIKI-models-tuning`) — specifically the `kiki-platformio` or related dataset.
    - Run: `uv run scripts/import_kiki_datasets.py --domain platformio`
    - After import, run dedup: `python -m src.distill.dedup --input data/raw/platformio.jsonl --output data/distilled/platformio.jsonl`.
    - Acceptance: `data/distilled/platformio.jsonl` has ≥ 800 rows; dedup report shows < 5% cross-domain contamination.
    - Dependencies: step 7; step 8; step 40.

42. **Curate dataset for domain-20 (power)**
    - Files to touch: `data/distilled/power.jsonl`
    - Source: `~/Documents/Projets/Factory 4 Life/KIKI-models-tuning/` (or HF mirror `L-electron-Rare/KIKI-models-tuning`) — specifically the `kiki-power` or related dataset.
    - Run: `uv run scripts/import_kiki_datasets.py --domain power`
    - After import, run dedup: `python -m src.distill.dedup --input data/raw/power.jsonl --output data/distilled/power.jsonl`.
    - Acceptance: `data/distilled/power.jsonl` has ≥ 800 rows; dedup report shows < 5% cross-domain contamination.
    - Dependencies: step 7; step 8; step 41.

43. **Curate dataset for domain-21 (emc)**
    - Files to touch: `data/distilled/emc.jsonl`
    - Source: `~/Documents/Projets/Factory 4 Life/KIKI-models-tuning/` (or HF mirror `L-electron-Rare/KIKI-models-tuning`) — specifically the `kiki-emc` or related dataset.
    - Run: `uv run scripts/import_kiki_datasets.py --domain emc`
    - After import, run dedup: `python -m src.distill.dedup --input data/raw/emc.jsonl --output data/distilled/emc.jsonl`.
    - Acceptance: `data/distilled/emc.jsonl` has ≥ 800 rows; dedup report shows < 5% cross-domain contamination.
    - Dependencies: step 7; step 8; step 42.

44. **Curate dataset for domain-22 (dsp)**
    - Files to touch: `data/distilled/dsp.jsonl`
    - Source: `~/Documents/Projets/Factory 4 Life/KIKI-models-tuning/` (or HF mirror `L-electron-Rare/KIKI-models-tuning`) — specifically the `kiki-dsp` or related dataset.
    - Run: `uv run scripts/import_kiki_datasets.py --domain dsp`
    - After import, run dedup: `python -m src.distill.dedup --input data/raw/dsp.jsonl --output data/distilled/dsp.jsonl`.
    - Acceptance: `data/distilled/dsp.jsonl` has ≥ 800 rows; dedup report shows < 5% cross-domain contamination.
    - Dependencies: step 7; step 8; step 43.

45. **Curate dataset for domain-23 (spice-sim)**
    - Files to touch: `data/distilled/spice-sim.jsonl`
    - Source: `~/Documents/Projets/Factory 4 Life/KIKI-models-tuning/` (or HF mirror `L-electron-Rare/KIKI-models-tuning`) — specifically the `kiki-spice-sim` or related dataset.
    - Run: `uv run scripts/import_kiki_datasets.py --domain spice-sim`
    - After import, run dedup: `python -m src.distill.dedup --input data/raw/spice-sim.jsonl --output data/distilled/spice-sim.jsonl`.
    - Acceptance: `data/distilled/spice-sim.jsonl` has ≥ 800 rows; dedup report shows < 5% cross-domain contamination.
    - Dependencies: step 7; step 8; step 44.

46. **Curate dataset for domain-24 (electronics)**
    - Files to touch: `data/distilled/electronics.jsonl`
    - Source: `~/Documents/Projets/Factory 4 Life/KIKI-models-tuning/` (or HF mirror `L-electron-Rare/KIKI-models-tuning`) — specifically the `kiki-electronics` or related dataset.
    - Run: `uv run scripts/import_kiki_datasets.py --domain electronics`
    - After import, run dedup: `python -m src.distill.dedup --input data/raw/electronics.jsonl --output data/distilled/electronics.jsonl`.
    - Acceptance: `data/distilled/electronics.jsonl` has ≥ 800 rows; dedup report shows < 5% cross-domain contamination.
    - Dependencies: step 7; step 8; step 45.

47. **Curate dataset for domain-25 (kicad-pcb)**
    - Files to touch: `data/distilled/kicad-pcb.jsonl`
    - Source: `~/Documents/Projets/Factory 4 Life/KIKI-models-tuning/` (or HF mirror `L-electron-Rare/KIKI-models-tuning`) — specifically the `kiki-kicad-pcb` or related dataset.
    - Run: `uv run scripts/import_kiki_datasets.py --domain kicad-pcb`
    - After import, run dedup: `python -m src.distill.dedup --input data/raw/kicad-pcb.jsonl --output data/distilled/kicad-pcb.jsonl`.
    - Acceptance: `data/distilled/kicad-pcb.jsonl` has ≥ 800 rows; dedup report shows < 5% cross-domain contamination.
    - Dependencies: step 7; step 8; step 46.

48. **Train stack-15 (embedded)**
    - Files to touch: `configs/stack-15-embedded.yaml`, `data/eval/embedded.jsonl`, `outputs/stacks/stack-15-embedded/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-15-embedded.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 36, 37, 16.

49. **Train stack-16 (stm32)**
    - Files to touch: `configs/stack-16-stm32.yaml`, `data/eval/stm32.jsonl`, `outputs/stacks/stack-16-stm32/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-16-stm32.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 48, 38, 16.

50. **Train stack-17 (iot)**
    - Files to touch: `configs/stack-17-iot.yaml`, `data/eval/iot.jsonl`, `outputs/stacks/stack-17-iot/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-17-iot.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 49, 39, 16.

51. **Train stack-18 (freecad)**
    - Files to touch: `configs/stack-18-freecad.yaml`, `data/eval/freecad.jsonl`, `outputs/stacks/stack-18-freecad/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-18-freecad.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 50, 40, 16.

52. **Train stack-19 (platformio)**
    - Files to touch: `configs/stack-19-platformio.yaml`, `data/eval/platformio.jsonl`, `outputs/stacks/stack-19-platformio/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-19-platformio.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 51, 41, 16.

53. **Train stack-20 (power)**
    - Files to touch: `configs/stack-20-power.yaml`, `data/eval/power.jsonl`, `outputs/stacks/stack-20-power/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-20-power.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 52, 42, 16.

54. **Group eval after stack 20**
    - Files to touch: `results/group-eval-after-20.json`
    - Run router eval + forgetting check across ALL 20 stacks.
    - Acceptance: global macro-F1 ≥ 0.75; no stack-forgetting > 3%.
    - Dependencies: step 53.

55. **Train stack-21 (emc)**
    - Files to touch: `configs/stack-21-emc.yaml`, `data/eval/emc.jsonl`, `outputs/stacks/stack-21-emc/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-21-emc.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 54, 43, 16.

56. **Train stack-22 (dsp)**
    - Files to touch: `configs/stack-22-dsp.yaml`, `data/eval/dsp.jsonl`, `outputs/stacks/stack-22-dsp/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-22-dsp.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 55, 44, 16.

57. **Train stack-23 (spice-sim)**
    - Files to touch: `configs/stack-23-spice-sim.yaml`, `data/eval/spice-sim.jsonl`, `outputs/stacks/stack-23-spice-sim/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-23-spice-sim.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 56, 45, 16.

58. **Train stack-24 (electronics)**
    - Files to touch: `configs/stack-24-electronics.yaml`, `data/eval/electronics.jsonl`, `outputs/stacks/stack-24-electronics/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-24-electronics.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 57, 46, 16.

59. **Group eval after stack 24**
    - Files to touch: `results/group-eval-after-24.json`
    - Run router eval + forgetting check across ALL 24 stacks.
    - Acceptance: global macro-F1 ≥ 0.75; no stack-forgetting > 3%.
    - Dependencies: step 58.

60. **Train stack-25 (kicad-pcb)**
    - Files to touch: `configs/stack-25-kicad-pcb.yaml`, `data/eval/kicad-pcb.jsonl`, `outputs/stacks/stack-25-kicad-pcb/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-25-kicad-pcb.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 59, 47, 16.

61. **Group eval after stack 25 (end of Phase VI)**
    - Files to touch: `results/group-eval-after-25.json`
    - Run router eval + forgetting check across ALL 25 stacks.
    - Acceptance: global macro-F1 ≥ 0.75; no stack-forgetting > 3%.
    - Dependencies: step 60.

### Phase VII — Applications + complements (stacks 26–32)

62. **Generate distilled datasets for domains 26–32 (apps + complements)**
    - Files to touch: `data/prompts/web-frontend.jsonl`, `data/prompts/web-backend.jsonl`, `data/prompts/music-audio.jsonl`, `data/prompts/devops.jsonl`, `data/prompts/llm-orch.jsonl`, `data/prompts/math.jsonl`, `data/prompts/security.jsonl`, plus the corresponding `data/distilled/*.jsonl`.
    - For each domain: seed 300 prompts per domain via manual + HF sources noted in step 8's `data-sources.md`, then distill ~1.5K examples each via the appropriate teacher (Devstral for web/devops, Mistral-Large-Opus for math/security/llm-orch, kxkm Qwen35B for music-audio).
    - Command: `uv run scripts/distill_apps_domains.py`
    - Duration: ~3 h total (7 domains × 200 prompts × 2 s/call teacher avg).
    - Acceptance: 7 jsonl files in `data/distilled/`, each ≥ 1500 rows; dedup pass applied.
    - Dependencies: steps 6, 7, 8.

63. **Train stack-26 (web-frontend)**
    - Files to touch: `configs/stack-26-web-frontend.yaml`, `data/eval/web-frontend.jsonl`, `outputs/stacks/stack-26-web-frontend/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-26-web-frontend.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 61, 62, 16.

64. **Train stack-27 (web-backend)**
    - Files to touch: `configs/stack-27-web-backend.yaml`, `data/eval/web-backend.jsonl`, `outputs/stacks/stack-27-web-backend/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-27-web-backend.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 63, 62, 16.

65. **Train stack-28 (music-audio)**
    - Files to touch: `configs/stack-28-music-audio.yaml`, `data/eval/music-audio.jsonl`, `outputs/stacks/stack-28-music-audio/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-28-music-audio.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 64, 62, 16.

66. **Train stack-29 (devops)**
    - Files to touch: `configs/stack-29-devops.yaml`, `data/eval/devops.jsonl`, `outputs/stacks/stack-29-devops/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-29-devops.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 65, 62, 16.

67. **Train stack-30 (llm-orch)**
    - Files to touch: `configs/stack-30-llm-orch.yaml`, `data/eval/llm-orch.jsonl`, `outputs/stacks/stack-30-llm-orch/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-30-llm-orch.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 66, 62, 16.

68. **Train stack-31 (math)**
    - Files to touch: `configs/stack-31-math.yaml`, `data/eval/math.jsonl`, `outputs/stacks/stack-31-math/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-31-math.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 67, 62, 16.

69. **Train stack-32 (security)**
    - Files to touch: `configs/stack-32-security.yaml`, `data/eval/security.jsonl`, `outputs/stacks/stack-32-security/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-32-security.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; forgetting check (step 16) passes against prior stacks.
    - Dependencies: steps 10, 68, 62, 16.

70. **Final meta-router retrain (v-final, 32 outputs)**
    - Files to touch: `outputs/router/v-final/router.pt`, `results/router-final.json`
    - Full 32-domain training set; include 3K cross-domain multi-intent examples.
    - Acceptance: macro-F1 ≥ 0.82 on held-out 32-class eval; p95 latency < 8 ms on RTX 4090.
    - Dependencies: step 69.

71. **Full evaluation suite**
    - Files to touch: `scripts/run_full_eval.sh`, `results/full-eval-v0.1.json`
    - Run: HumanEval (+ pass@1), GSM8K, MMLU-Pro sample, 32 custom domain evals, latency benchmarks.
    - Acceptance: consolidated JSON report; compare against base-Qwen3.5-4B numbers; each domain shows ≥ 0.55 win rate; aggregate score ≥ +15% over base.
    - Dependencies: step 70.

### Phase VIII — Aeon Memory Palace (neuro-symbolic memory)

72. **Aeon architecture spec (`docs/specs/aeon-memory-palace.md`)**
    - Files to touch: `docs/specs/aeon-memory-palace.md`
    - Document: Atlas SIMD Page-Clustered Vector Index (data layout, page size, clustering algorithm, SIMD kernels for AVX-512 / NEON), Trace neuro-symbolic episodic graph (node types, edge types, causality, temporal validity windows), API contract (`write_episode`, `recall`, `walk`, `query_by_time`, `compress`).
    - Reference: arxiv 2601.15311 (Aeon, 2026).
    - Acceptance: spec document reviewed against paper, covers 4 core operations.
    - Dependencies: none.

73. **Atlas SIMD vector index (`src/memory/atlas.py` + `src/memory/atlas_simd.{c,rs}`)**
    - Files to touch: `src/memory/atlas.py`, `src/memory/atlas_simd.c` (or .rs), `tests/test_atlas.py`, `setup.py` (cython/rust binding)
    - Page-clustered layout: vectors stored in fixed-size pages (~4 KB), clusters via k-means. SIMD kernels for dot-product batched (AVX-512 on kxkm-ai, NEON on Mac Studio).
    - Python bindings via cython (simple) or PyO3/rust (fast but more deps).
    - Fallback: pure Python with numpy when SIMD not available (validity via CI).
    - Acceptance: `test_atlas::test_insert_recall` — insert 10k 3072-dim vectors, recall top-10 latency < 5 ms on kxkm-ai CPU, accuracy vs naive ≥ 99 %.
    - Dependencies: step 72.

74. **Trace episodic graph (`src/memory/trace.py`)**
    - Files to touch: `src/memory/trace.py`, `tests/test_trace.py`
    - Graph store (Neo4j or in-process RustworkX) holding episodes = subgraphs with: `event` nodes, `causality` edges (directed, weighted), `temporal_validity` intervals, `source` provenance.
    - Neuro-symbolic rules: simple Datalog-style rules for retrieval filtering (e.g., "recall all `decision` events with `causality > 0.5` targeting `project=X` within last 30 days").
    - Acceptance: 100 synthetic episodes inserted, 5 rule queries return expected subsets, round-trip latency < 20 ms.
    - Dependencies: step 72.

75. **Aeon unified API (`src/memory/aeon.py`)**
    - Files to touch: `src/memory/aeon.py`, `src/memory/__init__.py`, `tests/test_aeon_api.py`
    - High-level class `AeonPalace` wrapping Atlas + Trace. Methods: `write(content, domain, timestamp, links)`, `recall(query, top_k, filters)`, `walk(from_node, max_depth)`, `query_by_time(interval)`, `compress(older_than)`.
    - `compress` uses learned summarization (call teacher on chunk) to reduce tokens by 5-10× while preserving recoverable meaning.
    - Acceptance: end-to-end test writes 1k episodes, recalls 50 queries, walks 10 hops, compresses half → token count 5×+ smaller.
    - Dependencies: steps 73, 74.

76. **Aeon storage backends (Qdrant + Neo4j integration)**
    - Files to touch: `src/memory/backends/qdrant_atlas.py`, `src/memory/backends/neo4j_trace.py`, `configs/aeon-backends.yaml`
    - Make backends pluggable — Atlas-native (SIMD local files) OR Qdrant remote; Trace-native (RustworkX in-memory) OR Neo4j remote (Tower or Graphiti).
    - Config selects backend per environment (dev = local; prod = remote).
    - Acceptance: both backend pairs pass the same test suite; switch via env var `AEON_BACKEND={native,remote}`.
    - Dependencies: step 75.

77. **Aeon integration in serving pipeline**
    - Files to touch: `src/serving/aeon_hook.py`, update `src/serving/vllm_server.py` (or mlx_server) to call Aeon pre-inference
    - Pre-inference: call `AeonPalace.recall(prompt, top_k=8)` → inject top memories into system prompt.
    - Post-inference: `AeonPalace.write(conversation_turn, domain=dispatch_result, links=[previous_turn_id])`.
    - Acceptance: integration test — 10-turn conversation, turn 9 references info from turn 2, model correctly retrieves via Aeon.
    - Dependencies: steps 76, 94 (vllm server) OR 95 (mlx server).

78. **Aeon compression daemon**
    - Files to touch: `scripts/aeon_compress_daemon.py`, `deploy/systemd/aeon-compress.service`
    - Background process that runs `compress` periodically on episodes older than N days. Reduces storage + recall noise.
    - Acceptance: 1000 episodes older than N days shrunk by ≥ 5× in size, `recall` still finds the compressed versions with ≥ 95 % accuracy on a held-out query set.
    - Dependencies: step 75.

79. **Aeon eval + tuning**
    - Files to touch: `scripts/eval_aeon.py`, `results/aeon-eval.json`
    - Benchmark: MemoryBench (arxiv 2601.15311 supplementary) — 100 multi-turn conversations, measure recall@k, conversation coherence score, token budget saved via compression.
    - Acceptance: recall@5 ≥ 0.85, coherence score ≥ 0.80, token savings ≥ 3× vs no-memory.
    - Dependencies: step 77.

### Phase IX — Negotiator (adaptive judge)

80. **Negotiator design (`docs/specs/negotiator.md`)**
    - Files to touch: `docs/specs/negotiator.md`
    - Document: CAMP-style evidence-based arbitration (arxiv 2604.00085) + Catfish dissent injection (arxiv 2505.21503). Pipeline: K=2-4 stack candidates → extract arguments → judge weights arguments (not votes) → if suspect consensus (>95 % agreement, weak arguments), inject Catfish prompt → re-judge.
    - Acceptance: spec reviewed, decision thresholds (agreement %, argument quality score) defined.
    - Dependencies: none.

81. **Argument extraction (`src/cognitive/argument_extractor.py`)**
    - Files to touch: `src/cognitive/argument_extractor.py`, `tests/test_argument_extractor.py`
    - For each of K stack candidate responses, extract: claim, evidence, reasoning chain. LLM-based (use base model with stack-reasoning active, or dedicated small judge).
    - Acceptance: on 20 synthetic candidates, extraction returns structured `{claim, evidence, reasoning}` triples with schema validation.
    - Dependencies: step 10 (trainer, base ready).

82. **Adaptive judge (`src/cognitive/judge.py`)**
    - Files to touch: `src/cognitive/judge.py`, `src/cognitive/judge_backends.py`, `configs/judge.yaml`
    - Two backends:
        - **fast**: Qwen3.5-35B-A3B Opus on kxkm-ai :8000 (~160 tok/s, default)
        - **deep**: Mistral-Large-Opus on Studio (slow but stronger reasoning, used only if confidence low)
    - Routing logic: agreement_score in (0, 1). If > 0.9 → skip judge (trust consensus). If 0.5–0.9 → fast judge. If < 0.5 → deep judge.
    - Output: `{winner_idx, confidence, rationale}`.
    - Acceptance: on 30 synthetic K=3 candidate sets (mix of easy/hard disagreements), routing selects the right backend and judge returns non-empty rationale.
    - Dependencies: step 4 (teacher_client for endpoints), step 81.

83. **Catfish dissent module (`src/cognitive/catfish.py`)**
    - Files to touch: `src/cognitive/catfish.py`, `tests/test_catfish.py`
    - Inject a dissent agent when high-agreement but weak arguments detected. The Catfish re-generates a candidate with explicit "devil's advocate" system prompt using one of the active stacks.
    - Trigger: agreement > 0.95 AND avg_argument_quality < 0.6.
    - Acceptance: on 10 high-agreement weak-argument cases, Catfish produces a substantively different response (embedding similarity < 0.7 with consensus).
    - Dependencies: step 10 (trainer), step 81.

84. **Negotiator integration + eval**
    - Files to touch: `src/cognitive/negotiator.py`, update serving pipeline, `results/negotiator-eval.json`
    - End-to-end: prompt → K candidates → argument extraction → judge → Catfish if triggered → final response.
    - Eval: 50 prompts that are known to elicit stack disagreement. Measure: judge accuracy vs human-labeled ground truth, Catfish trigger rate, latency overhead.
    - Acceptance: judge agrees with ground truth ≥ 75 %; Catfish triggers on ≥ 80 % of planted suspect-consensus cases; latency overhead ≤ 300 ms when fast, ≤ 2 s when deep.
    - Dependencies: steps 82, 83.

### Phase X — Anti-bias (KnowBias neurons + RBD)

85. **Bias dataset curation (`data/bias/bias_pairs.jsonl`)**
    - Files to touch: `data/bias/bias_pairs.jsonl`, `scripts/curate_bias_dataset.py`
    - ~5000 pair examples: (biased_prompt, fair_prompt, expected_behavior). Mix: confirmation, anchoring, authority, recency, framing, stereotyping. Sources: StereoSet, CrowS-Pairs, BBQ, DeFrame seed set.
    - Acceptance: ≥ 5000 pairs, balanced across 6 bias types (≥ 500 each), 95 %+ valid JSONL.
    - Dependencies: none.

86. **Neuron probing (`src/cognitive/bias_probe.py`)**
    - Files to touch: `src/cognitive/bias_probe.py`, `tests/test_bias_probe.py`
    - PyTorch hooks on activations of Qwen3.5-4B layers. For each pair in dataset, record activation delta across all neurons. Rank neurons by |Δ_biased - Δ_fair|.
    - Output: `results/bias_neurons_base.json` (top N=200 bias neurons per layer).
    - Acceptance: 200 bias neurons identified in base model; probe runs in < 30 min on 4090 with the 5K dataset.
    - Dependencies: step 3 (loader), step 85.

87. **KnowBias fine-tune on BASE (pre-stacks)**
    - Files to touch: `scripts/knowbias_finetune_base.py`, `outputs/base-knowbias/` (gitignored), `configs/knowbias-base.yaml`
    - LoRA fine-tune targeting ONLY the identified bias neurons (narrow training, r=8). Objective: reduce activation delta between biased/fair pairs → no behavior change on non-bias tasks.
    - Reference: arxiv 2601.21864 (KnowBias, 2026).
    - Duration: ~2h on 4090.
    - Acceptance: bias score (StereoSet + CrowS-Pairs) drops ≥ 30 % vs base; MMLU-Pro + HumanEval capacity retention ≥ 95 % (no significant capacity loss).
    - Dependencies: step 86.

88. **Validate base debiased preserves capacity (`scripts/eval_base_knowbias.py`)**
    - Files to touch: `scripts/eval_base_knowbias.py`, `results/base-knowbias-eval.json`
    - Run full eval suite (HumanEval, GSM8K, MMLU-Pro) on base-knowbias vs base.
    - Acceptance: no benchmark drops more than 2 pp.
    - Dependencies: step 87.

89. **Re-anchor all 32 stack training on debiased base**
    - Files to touch: update training configs 11, 17-19, 23-35, 48-60, 63-69 (all stack configs) to use `outputs/base-knowbias/` as base path.
    - NOTE: this is a refactor step — does NOT retrain stacks. It just updates config files so FUTURE training uses the debiased base. Stacks already trained keep their artifacts; if retrain is desired, do so manually.
    - Acceptance: all stack config files have base_model_path → debiased base; ralph can validate with a grep.
    - Dependencies: step 88.

90. **KnowBias probing on MERGED model (post-stacks)**
    - Files to touch: `scripts/knowbias_probe_merged.py`, `results/bias_neurons_merged.json`
    - Re-run probing (step 86 logic) on the fully-merged model (base + 32 stacks fused). Stacks may have introduced new biases (domain-specific stereotypes).
    - Acceptance: second bias neuron set identified; compare overlap with base's — expect 40-60 % overlap, 40-60 % new.
    - Dependencies: step 86, step 70 (final router retrain post-all-stacks / merged model ready).

91. **KnowBias fine-tune on MERGED (post-stacks)**
    - Files to touch: `scripts/knowbias_finetune_merged.py`, `outputs/final-knowbias/` (gitignored)
    - Second KnowBias round on merged model using the post-stacks bias neurons.
    - Acceptance: bias score drops ≥ 20 % further; domain capacities retained (per-stack win_rate_vs_base ≥ -2 %).
    - Dependencies: step 90.

92. **RBD runtime plug-in (`src/cognitive/rbd.py`)**
    - Files to touch: `src/cognitive/rbd.py`, `tests/test_rbd.py`
    - Reasoning-based Bias Detector (arxiv 2505.17100). Post-inference check: given (prompt, response), a lightweight reasoning pass flags potential biases not caught by KnowBias (domain-specific, framing effects).
    - Uses Qwen3.5-4B-debiased itself as detector (self-consistent) OR Qwen1.7B as cheaper alternative.
    - Acceptance: on 100 responses (50 biased, 50 clean), RBD flags ≥ 80 % of biased, false-positive rate ≤ 15 %.
    - Dependencies: step 91.

93. **Anti-bias integration + eval**
    - Files to touch: `src/cognitive/antibias.py` (orchestrator), update serving pipeline, `results/antibias-eval.json`
    - Orchestration: RBD runs on every response; if flagged, trigger DeFrame-style re-generation with inverted framing (optional fallback).
    - Full eval: StereoSet, CrowS-Pairs, BBQ, custom framing dataset. Compare pre vs post full antibias stack.
    - Acceptance: aggregate bias score drops ≥ 50 % vs vanilla micro-kiki; capacity retention ≥ 95 %.
    - Dependencies: step 92.

### Phase XI — Serving deployment

94. **vLLM server with dynamic LoRA (`src/serving/vllm_server.py`)**
    - Files to touch: `src/serving/vllm_server.py`, `docker/vllm.Dockerfile`
    - vLLM 0.6+ with `--enable-lora --max-loras 4 --max-lora-rank 16`; expose OpenAI-compatible endpoint on :8100.
    - **Dynamic LoRA config**:
      - Set env var `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` at startup.
      - Use `/v1/load_lora_adapter` endpoint to dynamically load/unload adapters per request.
      - Register a `LoRAResolver` plugin pointing at `outputs/stacks/` directory so adapters can be referenced by name in OpenAI-style `model` field.
      - Caveat: this is a security risk in multi-tenant — keep service bound to localhost + tailscale only.
      - Reference: vLLM docs, unsloth hot-swap guide.
    - Router runs as a FastAPI sidecar on :8101 that rewrites incoming request with `lora_request` field.
    - Acceptance: `curl :8100/v1/chat/completions` with "write python factorial" → correct response, logs show stacks 03 + 02 active; dynamic `/v1/load_lora_adapter` test loads an unseen stack and serves a query without restart.
    - Dependencies: step 70.

95. **mlx-lm server for Mac Studio (`src/serving/mlx_server.py`)**
    - Files to touch: `src/serving/mlx_server.py`, `configs/mlx-server.json`
    - mlx-lm with adapter switching via subprocess restart (mlx doesn't support hot-swap cleanly — accept ~200ms penalty per switch) OR via `mlx_lm.server --adapter-path` per-request.
    - Acceptance: runs on Studio, serves 32 stacks (max 4 active), p95 first-token latency < 500 ms.
    - Dependencies: step 70.

96. **Persistent service unit (staged only, no live install)**
    - Files to touch: `deploy/launchd/cc.saillant.micro-kiki.plist` (Mac), `deploy/systemd/micro-kiki.service` (Linux), `deploy/README.md`.
    - Launchd plist for Studio, systemd for kxkm-ai. Autostart on boot, restart on crash.
    - NO live install. The CI/loop environment cannot `sudo` non-interactively. Unit files are STAGED in `deploy/` only.
    - `deploy/README.md` documents the manual install commands for an operator with sudo:
      - Mac: `cp deploy/launchd/cc.saillant.micro-kiki.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/cc.saillant.micro-kiki.plist`
      - Linux: `sudo cp deploy/systemd/micro-kiki.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now micro-kiki`
    - Acceptance: `deploy/launchd/cc.saillant.micro-kiki.plist` + `deploy/systemd/micro-kiki.service` exist and `plutil -lint` / `systemd-analyze verify` both pass on the staged files; `deploy/README.md` documents install commands; NO requirement that the service is actually running.
    - Dependencies: steps 94, 95.

97. **Integration test (router HTTP)**
    - Files to touch: `tests/test_integration_http.py`
    - Spawn router sidecar + vLLM (or mlx-server) in pytest fixture, send 20 queries spanning 10 domains, assert each hits correct top-2 stacks.
    - Acceptance: 20/20 queries routed correctly; test completes in < 60 s.
    - Dependencies: step 94 OR step 95.

### Phase XII — ANE triple pipeline (Mac-only)

98. **ANE draft model (`src/serving/ane_draft.py`)**
    - Files to touch: `src/serving/ane_draft.py`, `models/qwen3.5-0.8b.mlpackage`
    - Convert Qwen3.5-0.8B to CoreML (ANE-targeted), use as speculative decoder for the 4B base running on GPU (Metal Performance Shaders).
    - Target: 2-3× decode speedup on M3 Ultra.
    - Acceptance: benchmark before/after on 100-token generation, speedup ≥ 1.8×.
    - Dependencies: step 95.

99. **ANE scorer for GRPO (`src/serving/ane_scorer.py`)**
    - Files to touch: `src/serving/ane_scorer.py`
    - CoreML-compiled scorer (reward model) running on ANE for RL fine-tuning pipelines (future GRPO stacks).
    - Acceptance: scores 100 prompt-completion pairs in < 2 s on ANE; scores match GPU-ref within ±0.02.
    - Dependencies: step 98.

100. **ANE-resident meta-router (`src/serving/ane_router.py`)**
    - Files to touch: `src/serving/ane_router.py`
    - Compile the 2M-param router to CoreML, run on ANE (not CPU) to free CPU for decode orchestration.
    - Acceptance: latency drops from ~5 ms (CPU) to ~1 ms (ANE); identical outputs within fp16 tolerance.
    - Dependencies: step 70.

### Phase XIII — Quantum-inspired pre-release (classical only, no QPU)

101. **CompactifAI base compression**
    - Files to touch: `src/compress/compactifai.py`, `src/compress/__init__.py`, `scripts/compress_base.py`, `tests/test_compactifai.py`, `configs/compactifai.yaml`
    - Implement tensor-network truncation over self-attn (Q/K/V/O projections) and MLP (gate/up/down) layers of the **debiased, post-stacks-fused** base checkpoint. Use `quimb` for MPS/MPO decomposition and `opt_einsum` for contraction paths.
    - Expose CLI: `uv run python scripts/compress_base.py --bond-dim 32 --input models/qwen3.5-4b-fused/bf16 --output models/qwen3.5-4b-compact/chi32`.
    - Progressive chi schedule: produce chi in {64, 32, 16, 8} artifacts; evaluate each on HumanEval-sample + MMLU-Pro-sample.
    - Commands:
      - `uv add quimb opt_einsum tensornetwork`
      - `uv run pytest tests/test_compactifai.py`
      - `uv run python scripts/compress_base.py --bond-dim 32`
    - Acceptance: chi=32 artifact exists at `models/qwen3.5-4b-compact/chi32/`, on-disk size <= 1 GB, loader smoke test yields non-empty response, accuracy drop on HumanEval-sample < 5 pp vs BF16 baseline.
    - Dependencies: step 91 (post-stacks debiased base).

102. **QTHA adapter pilot on stack-02 (reasoning) — comparative study vs MoLoRA**
    - Files to touch: `src/stacks/qtha.py`, `src/stacks/__init__.py`, `scripts/train_qtha_stack.py`, `tests/test_qtha.py`, `configs/qtha-stack-02.yaml`
    - Implement `QTHAAdapter(nn.Module)`: decomposes the frozen base weights into a tensor-network factor + a trainable low-bond-dim correction. Parameter count target: ~500 K (vs 2 M MoLoRA rank-16). Training hook compatible with PEFT-style attach/detach.
    - Reuse the stack-02 reasoning dataset (`data/dedup/02-reasoning/`). Train on RTX 4090 via Unsloth-compatible path.
    - Eval: `win_rate_vs_base` on reasoning eval slice; compare to the v0.1-class MoLoRA stack-02 baseline (step 17).
    - Commands:
      - `uv run pytest tests/test_qtha.py`
      - `uv run python scripts/train_qtha_stack.py --config configs/qtha-stack-02.yaml`
      - `uv run python scripts/eval_stack.py --stack stack-02-reasoning-qtha --compare stack-02-reasoning`
    - Acceptance: adapter saved to `outputs/stacks/stack-02-reasoning-qtha/`, param count logged <= 600 K, `win_rate_vs_base` >= MoLoRA stack-02 - 1 pp, comparative report at `results/qtha-vs-molora.md` with a merge/fallback decision line.
    - Dependencies: step 17 (MoLoRA stack-02 baseline); step 101 optional (for compact base path).

103. **Tensor-network router prototype**
    - Files to touch: `src/routing/tn_router.py`, `src/routing/__init__.py`, `scripts/train_tn_router.py`, `tests/test_tn_router.py`, `configs/tn-router.yaml`
    - Implement `TNRouter(nn.Module)` gating function using matrix product states (MPS) over embedding input; produce per-domain scores via interference amplitudes.
    - Train on the final 32-domain router eval set (step 70). Fallback guard: if macro-F1 drops > 2 pp vs sigmoid, flag in report and do not merge into the serving path.
    - Commands:
      - `uv run pytest tests/test_tn_router.py`
      - `uv run python scripts/train_tn_router.py --config configs/tn-router.yaml`
      - `uv run python scripts/eval_router.py --router tn --compare sigmoid`
    - Acceptance: `outputs/routers/tn-v0/` artifact exists, comparative report at `results/tn-vs-sigmoid.md`, macro-F1 delta reported (+/- vs sigmoid), decision line ("merge" | "fallback") included.
    - Dependencies: step 70 (final sigmoid router + 32-domain eval set).

### Phase XIV — Release v0.2

104. **Freeze config + migration guide**
    - Files to touch: `MIGRATION.md`, `VERSION`, tag `v0.2.0`
    - Document breaking changes, adapter format, router protocol, quantum-inspired overlay artifacts. Freeze `configs/` (hash-lock).
    - Acceptance: `git tag v0.2.0` pushed; MIGRATION.md reviewed.
    - Dependencies: step 97, step 103.

105. **HuggingFace release**
    - Files to touch: `.github/workflows/hf-release.yml`, HF repo `electron-rare/micro-kiki-v0.2`
    - Upload: base-model reference (link to Qwen), compact base chi=32 (optional), 32 MoLoRA adapter .safetensors, QTHA stack-02 (optional), router.pt (+ tn-router.pt if merged), model card, eval results.
    - Acceptance: `huggingface-cli download electron-rare/micro-kiki-v0.2` works from clean env; model card renders.
    - Dependencies: step 104.

106. **Model card + cookbook**
    - Files to touch: `MODEL_CARD.md`, `COOKBOOK.md`, `examples/` dir
    - Document: intended use, limitations, training data, eval scores, quantum-inspired overlay results (CompactifAI / QTHA / TN-router), 5+ runnable examples (chat-fr, code, SPICE, KiCad, reasoning).
    - Acceptance: all examples in COOKBOOK run green in CI; model card passes HF lint.
    - Dependencies: step 105.

---

## Success criteria (v0.2)

- 32 stacks trained, each ≥ 0.55 win-rate vs base on its domain.
- Meta-router macro-F1 ≥ 0.82 across 32 domains.
- Serving stack runs on RTX 4090 24 GB (max 4 active stacks) AND on Mac Studio.
- Aggregate eval score ≥ +15% over base Qwen3.5-4B.
- Quantum-inspired overlay: CompactifAI chi=32 artifact ≤ 1 GB with < 5 pp drop; QTHA vs MoLoRA comparative report shipped; TN-router macro-F1 vs sigmoid reported with merge/fallback decision.
- Released on HuggingFace with model card + cookbook.

## Risks + mitigations

- **Forgetting across stacks** → step 16 forgetting check (win-rate delta + gradient subspace angle) after EACH stack; rollback if win-rate drop > 3% AND angle < 30°.
- **Router saturation at 32 domains** → re-trains at stacks 08/12/14/final (steps 28, 33, 36, 70); fallback to hierarchical router (6 phase-groups → intra-group classifier).
- **VRAM overrun** → hard cap 4 active stacks, Q4 base, KV cache eviction.
- **Teacher unavailability** → step 4 caches responses; fallback to Qwen3.5-35B (kxkm-ai) if Studio offline.
- **Quantum-inspired regressions** → each of steps 101-103 ships a fallback/merge decision line; artifacts are overlay-only and do not replace the classical serving path unless the decision is "merge".

**v0.3 roadmap (post-release)**: temporal context injection (real-time clock, location, news slice) and future-reasoner (CoT temporal chains, calendar-aware planning) are explicit non-goals for v0.2. They will ship in v0.3 as a tools layer, not as new stacks (LLM 4B underperforms dedicated time-series ML models per arxiv 2601.10132 — context injection is the right approach). True QPU experiments live on the `quantum` branch (public, hybrid with classical fallback) and in the private `micro-kiki-quantum` repo (QPU-only research).
