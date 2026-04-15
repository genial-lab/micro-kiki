# Ralph Loop Guardrails — v0.3 Neuroscience

Constraints and boundaries for the v0.3 ralph loop on branch `neuroscience`.

## Scope Boundaries

This loop implements the plan at: `.claude/plans/micro-kiki-v0.3-neuroscience.md`.

### In Scope
- Stories defined in `.ralph-v0.3/prd.json` (26 stories)
- Files mentioned in the v0.3 plan
- Quality gates listed in `.ralph-v0.3/CLAUDE.md`

### Out of Scope
- v0.2 stack training / router / dispatcher artifacts (different base architecture)
- `main` branch work
- `quantum` branch work
- Private `micro-kiki-quantum` repo
- QPU calls (this is the neuroscience branch, not the QPU branch)
- Refactoring unrelated code
- Dependency upgrades unless the plan explicitly calls for them

## Cousin-fork discipline

- v0.3 has a different base model (SpikingBrain-76B vs Qwen3.5-4B v0.2). Weights do NOT transfer.
- Code MAY be copied from v0.2 when the v0.3 plan explicitly authorises it (Atlas SIMD index story-6, Trace graph story-7). These are pure classical components.
- If a story text references v0.2 infrastructure, port it as a fresh module under the v0.3 tree; do not symlink or git-subtree from main.

## Quality Requirements

All changes must:
1. Pass defined quality gates
2. Include appropriate tests (if test infrastructure exists)
3. Follow existing code patterns
4. Not break existing functionality
5. Be committed on the `neuroscience` branch

## Commit Standards

- One commit per story
- Conventional commit format: `feat(<scope>): <imperative>`
- Scope should be a sub-area: `neuroscience`, `aeonsleep`, `spiking`, `akida`, `map`, `loihi`
- Subject ≤ 50 chars (pre-commit hook enforces)
- Body lines ≤ 72 chars (pre-commit hook enforces; wrap prose manually)
- No `Co-Authored-By:` trailer (hook rejects)
- Large diffs (> 734 LOC) require the 4-section body template (see `.ralph-v0.3/CLAUDE.md`)

## Blocking Conditions

Stop and document in `.ralph-v0.3/progress.txt` if:
- Quality gates fail after 3 attempts
- Story requires clarification not in plan
- External dependency unavailable (especially SpikingBrain-76B checkpoint — use the fallback plan documented in story-12)
- Akida Mini PCIe card not delivered yet at story-22 (hold until hardware arrives, simulator-only alternative documented)
- Circular dependency detected

## Project-specific constraints

- **Hardware cost gate**: step 22 = $300 one-time order of Akida Mini PCIe. NEVER skip step 21 (simulator validation) before ordering hardware.
- **ESP32-S3 stretch goal (step 24)**: OPTIONAL. If time-boxed or release is near, skip and mark `passes: true` with a note in progress.txt. v0.3 ships without it.
- **SpikingBrain availability**: step 12 probes for checkpoint. If path (c) fallback to 7B is used, document the downgrade in all downstream story acceptance notes.
- **AeonSleep discipline**: steps 6-10 must all pass independently (Atlas, Trace, SleepTagger, ForgettingGate, Consolidation) before step 11 (unified API). Do NOT take shortcuts by merging components early.
- **Never** switch branches from inside a ralph iteration
- **Never** push to `main` or `quantum` from this loop

## Recovery

If the loop fails:
1. Check `.ralph-v0.3/progress.txt` for last successful story
2. Check `git log --oneline neuroscience` for committed work
3. Review `.ralph-v0.3/prd.json` for story states
4. Resume with: `uv run .ralph-v0.3/loop.py`

## Manual Override

To skip a problematic story:
```bash
# Edit .ralph-v0.3/prd.json, set passes: true for the story
# Add note to .ralph-v0.3/progress.txt explaining skip
# Run: uv run .ralph-v0.3/loop.py
```

Recommended skips for pragmatic releases:
- story-24 (ESP32-S3 stretch) — explicitly optional
- story-22 (Akida PCIe order) — if budget or shipping blocks, mark skipped and note simulator-only release
