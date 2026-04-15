# Agentic Capabilities Design — micro-kiki

Date: 2026-04-15
Status: Approved
Scope: Web search, multi-level auto-critique, autonomous ralph loop

## Overview

Extend micro-kiki with runtime agentic capabilities (web search, 3-level auto-critique) and upgrade the ralph loop into a fully autonomous development agent. Hybrid approach: the sigmoid meta-router decides *what* to activate, the Python orchestrator handles *how* to execute.

## 1. Meta-Router Extension (32 → 37 outputs)

Add 5 capability flags to the existing sigmoid router:

| Flag | Threshold | Purpose |
|------|-----------|---------|
| `web_search` | 0.15 | Trigger web/docs/papers search |
| `self_critique_token` | 0.10 | Best-of-N / self-consistency |
| `self_critique_response` | 0.20 | Structured critique + correction |
| `self_critique_task` | 0.35 | Full agentic loop |
| `deep_eval` | 0.25 | Formal benchmark evaluation |

Same sigmoid architecture, same training approach. Thresholds calibrated by eval. Dispatcher YAML extended to map domain+capability combos to concrete actions.

VRAM impact: negligible (slightly wider linear layer).

## 2. Web Search — 3 Specialized Backends

| Tool | Backend | Scope |
|------|---------|-------|
| `search_web` | Exa API | General web, code, forums, technical docs |
| `search_papers` | Semantic Scholar API | Arxiv papers, citations, state of the art |
| `search_docs` | Targeted scraper + cache | Datasheets, library docs, API references |

### Flow

1. Router activates `web_search` flag
2. Model generates structured intent: `{"tool": "search_papers", "query": "...", "max_results": 5}`
3. Orchestrator executes, injects results into context
4. Model generates response with sources

### Cache & RAG

- SQLite cache: TTL 24h (web), 30d (papers)
- Relevant results indexed in Aeon for long-term memory enrichment
- `search_docs` maintains local index of frequently accessed datasheets/docs

### Deployment

All 3 backends run as lightweight services on Mac Studio (no GPU needed). Model on kxkm-ai calls via HTTP, or everything local on Mac Studio in MLX mode.

## 3. Auto-Critique — 3 Stacked Levels

### Level 1: Token-level (best-of-N)

Adaptive sampling driven by router confidence:
- Confidence > 0.8 → N=1 (no re-sampling)
- 0.5–0.8 → N=3
- < 0.5 → N=5

Scoring: mean log-probability + query coherence. Runs on same GPU as inference.

### Level 2: Response-level (self-refine)

Activated when `self_critique_response` > 0.20.

```
Initial response → Structured critique → Corrected response
```

- Critique template: `{factual_errors, missing_info, clarity, confidence}`
- Single correction pass (no infinite loops)
- If critique finds nothing substantial → initial response returned as-is
- Adaptive judge validates correction (Qwen3.5-35B cheap → Mistral-Large if confidence < 0.5)

### Level 3: Task-level (agentic loop)

Activated when `self_critique_task` > 0.35. Reserved for complex multi-step tasks.

```
Plan → Execute step → Evaluate result → Correct/continue → ... → Final result
```

- Max 5 iterations (hard cap)
- Each iteration can trigger `web_search` or levels 1-2
- Runs on Mac Studio (unified memory for long contexts)
- Negotiator (CAMP + Catfish) arbitrates between iterations

### Stacking

Levels are not mutually exclusive. A complex query can activate all 3:
1. Agentic loop plans steps (level 3)
2. Each step generates with best-of-N (level 1)
3. Consolidated response goes through self-refine (level 2)

Router decides which levels activate — no overhead for simple queries.

## 4. Ralph Loop — Full Autonomous Agent

### 4.1 Automatic Pre-Story Research

Before each story from the 102-step plan:
1. Extract keywords (domain, technique, cited papers)
2. `search_papers` → recent relevant papers
3. `search_web` → reference implementations (GitHub, blogs)
4. Results injected into implementing agent context
5. Sources saved in `.ralph/research/<step-NN>.md`

### 4.2 Code Auto-Critique

After each implementation:
```
Generate code → Lint + type check → Structured self-review → Correct → Tests
```

- Self-review template: `{bugs, edge_cases, perf, security, style}`
- Max 3 correction passes
- If tests pass and self-review clean → propose commit

### 4.3 Integrated Forgetting Check

After each stack trained, ralph automatically:
1. Compute gradient subspace angle (base vs adapted)
2. Win-rate on held-out set
3. If angle < 30° AND win-rate drop > 0.03 → automatic rollback + alert
4. Results logged in `.ralph/evals/<stack-NN>.json`

### 4.4 Complete Autonomous Loop

```
Story → Research → Implement → Critique → Tests → Forgetting check → Commit
  ↑                                                                      |
  └──────────────────── next story ─────────────────────────────────────┘
```

- Guardrails in `.ralph/guardrails.md`
- Hard stop if: 3 consecutive failures, rollback triggered, or token budget exceeded
- Progress tracked in `.ralph/progress.txt`
- Dry-run mode available (everything except commit)

Ralph runs on Mac Studio — calls kxkm-ai for GPU training, executes the rest locally.

## 5. Deployment Architecture

### Machine Allocation

| Machine | Role | Components |
|---------|------|------------|
| kxkm-ai (RTX 4090) | Training + light serving | Stack training, vLLM inference, best-of-N |
| Mac Studio (M3 Ultra) | Agentic serving + orchestration | MLX inference, auto-critique loops, ralph loop, web search backends, SQLite cache, Aeon |

### Communication

```
Mac Studio (orchestrator)
  ├── HTTP → kxkm-ai:8000 (vLLM inference)
  ├── HTTP → Exa API (web search)
  ├── HTTP → Semantic Scholar API (papers)
  ├── Local → scraper + SQLite cache (docs)
  └── Local → Aeon (memory)
```

### Operating Modes

| Mode | Machine | Usage |
|------|---------|-------|
| Simple inference | kxkm-ai only | Direct queries, no tools |
| Agentic inference | Mac Studio orchestrates, kxkm-ai serves | Web search, auto-critique, complex tasks |
| Training | kxkm-ai only | Stack training, ralph pilots from Mac Studio |
| Autonomous dev | Mac Studio | Full ralph loop, MLX for quick tests |

### Fallback

- kxkm-ai unavailable → Mac Studio takes over in MLX (slower but functional)
- Web API down → local cache + Aeon as fallback
- Semantic Scholar rate-limited → queue + retry with backoff

## 6. Implementation Plan — Phase 14

Additive phase after existing 13 phases. 15 new stories (103–117).

| Story | Description | Dependencies |
|-------|-------------|--------------|
| 103 | Extend sigmoid router 32→37 outputs | Phase IV (existing router) |
| 104 | Dispatcher YAML: capability flags → action mapping | Story 103 |
| 105 | `search_web` backend (Exa) + SQLite cache | None |
| 106 | `search_papers` backend (Semantic Scholar) + cache | None |
| 107 | `search_docs` backend (targeted scraper + local index) | None |
| 108 | HTTP orchestrator Mac Studio → kxkm-ai | Phase XIII (serving) |
| 109 | Adaptive best-of-N (level 1 auto-critique) | Story 103 |
| 110 | Self-refine pipeline (level 2) + critique template | Story 109 |
| 111 | Integrate adaptive judge into self-refine | Story 110 + Phase XII (Negotiator) |
| 112 | Task-level agentic loop (level 3) + hard cap | Stories 110, 105-107 |
| 113 | Web results injection into Aeon | Story 105 + Phase XI (Aeon) |
| 114 | Ralph: automatic pre-story research | Stories 105-106 |
| 115 | Ralph: code self-review + correction loop | None |
| 116 | Ralph: automated post-stack forgetting check | Phase IV (forgetting check) |
| 117 | Ralph: complete autonomous loop + guardrails | Stories 114-116 |

Stories 105-107 are independent → parallelizable.
Stories 114-116 are independent → parallelizable.

### Impact on Existing Phases

None. Phase 14 is purely additive. Contact points:
- Story 103 extends router (Phase IV)
- Story 111 uses Negotiator (Phase XII)
- Story 113 writes to Aeon (Phase XI)

## 7. Constraints & Invariants

- Max 4 active stacks + capabilities must stay within 24 GB VRAM on RTX 4090
- Auto-critique level 3 hard cap at 5 iterations — no infinite loops
- Ralph hard stop after 3 consecutive failures
- Web cache TTLs enforced (no stale data in Aeon)
- Best-of-N sampling N is never > 5 (latency bound)
- All web search results include source attribution
- Forgetting check is non-negotiable after every stack — ralph cannot skip it
