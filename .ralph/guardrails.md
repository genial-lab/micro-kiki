# Ralph Loop Guardrails

Constraints and boundaries for this ralph loop execution.

## Scope Boundaries

This loop implements the plan at: `.claude/plans/micro-kiki-v0.2-implementation.md`

### In Scope
- Stories defined in prd.json
- Files mentioned in the original plan
- Quality gates listed in CLAUDE.md

### Out of Scope
- Features not in the plan
- Refactoring unrelated code
- Dependency upgrades (unless specified)
- Documentation beyond code comments

## Quality Requirements

All changes must:
1. Pass defined quality gates
2. Include appropriate tests (if test infrastructure exists)
3. Follow existing code patterns
4. Not break existing functionality

## Commit Standards

- One commit per story
- Conventional commit format: `feat(ralph): {description}`
- Include `Story-Id: {story-id}` in commit body
- No unrelated changes in commits

## Blocking Conditions

Stop and document in progress.txt if:
- Quality gates fail after 3 attempts
- Story requires clarification not in plan
- External dependency is unavailable
- Circular dependency detected

## Project-specific constraints

- Never train stacks in parallel on the same GPU
- Never route > 4 stacks simultaneously (VRAM cap)
- Never drop below Q4 quantization for base model
- Never change base model without updating all specs
- Training stacks MUST run in curriculum order (foundations first)
- Forgetting check required after each new stack

## Phase 14 — Agentic Capabilities

### Additional constraints
- Web search backends must cache all results (SQLite, enforced TTLs)
- Best-of-N sampling: N never exceeds 5
- Agentic loop: hard cap at 5 iterations
- Self-refine: single correction pass only
- Ralph hard stop after 3 consecutive failures
- Forgetting check is non-negotiable after every stack training story
- All search results include source attribution
- HTTP bridge timeout: 120s max

### New packages
- `src/search/` — web search backends (exa, scholar, docs) + cache
- `src/critique/` — auto-critique levels 1-3
- `src/orchestrator/` — main engine + HTTP bridge
- `src/ralph/` — research, self-review, forgetting, autonomous loop

## Recovery

If the loop fails:
1. Check progress.txt for last successful story
2. Check git log for committed work
3. Review prd.json for story states
4. Resume with: `uv run .ralph/loop.py`

## Manual Override

To skip a problematic story:
```bash
# Edit prd.json, set passes: true for the story
# Add note to progress.txt explaining skip
# Run: uv run .ralph/loop.py
```
