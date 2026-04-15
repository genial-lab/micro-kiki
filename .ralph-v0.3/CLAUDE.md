# Ralph Loop Task — v0.3 Neuroscience

You are executing ONE iteration of the v0.3 ralph loop on the `neuroscience` branch. Complete ONE story, then exit.

## Your Task

1. Read `.ralph-v0.3/prd.json` to find the next incomplete story (passes: false, lowest priority)
2. Implement ONLY that story
3. Run quality gates
4. Commit your changes
5. Update progress
6. Exit

## Branch discipline

- You are on branch `neuroscience`. Do NOT switch branches.
- This is a **cousin fork** of v0.2, not an evolution. Do NOT import v0.2 stack/router/dispatcher artifacts.
- Code may be *copied* from v0.2 if the spec explicitly says so (Atlas, Trace).
- `main` and `quantum` are out of scope.

## Quality Gates

Run these commands before committing. ALL must pass:

- `uv run pytest` (once tests exist)

Light checks (best-effort, not blocking if tool not installed):

- `uv run python -c "import src"` (package imports cleanly)

If any gate fails, fix the issue before committing.

## Story Completion Protocol

When you complete a story:

1. **Check for changes and commit if needed:**
   ```bash
   git status --porcelain
   ```

   If there ARE changes:
   - Stage changes: `git add -A`
   - Commit with a conventional commit message describing your implementation
   - The `commit_quality_enforcer` hook validates format automatically
   - If commit is rejected, read the error, fix the message, retry (max 3 attempts)
   - After 3 failures, log the error to progress.txt and exit

   If there are NO changes, proceed directly to step 2.

2. **Update prd.json:**
   - Set `passes: true` for the completed story in `.ralph-v0.3/prd.json`

3. **Append to `.ralph-v0.3/progress.txt`:**
   ```
   [{timestamp}] Completed: {story-id} - {story-title}
   ```

4. **Exit immediately** — Do not start another story.

## Commit body requirements

This repo's pre-commit hook `PreToolUse:Bash` rejects commits with large diffs (> 734 LOC) unless the body contains all 4 sections below. Small diffs (< 300 LOC) can use just the subject + a 1-2 line summary.

**Required body template for large diffs**:

```
<subject ≤ 50 chars, no dots in scope>

## Context
<1-2 sentences: why this change? what problem/need?>

## Approach
<1-3 sentences: what strategy was chosen and why>

## Changes
<bullets of concrete modifications: files touched, functions added/modified, tests added>

## Impact
<1-2 sentences: what does this unlock, what metric improved, what risk>
```

**Convention reminders**:
- Subject: conventional commit `<type>(<scope>): <imperative>`, scope without dots (use `neuroscience` or a sub-scope like `aeonsleep`, `spiking`, `akida`)
- Subject ≤ 50 chars
- Body lines ≤ 72 chars (hook validates; wrap prose manually)
- NO `Co-Authored-By:` trailer (hook rejects)
- Commit via HEREDOC to preserve formatting:

```bash
git commit -m "$(cat <<'EOF'
feat(aeonsleep): sleep tagger conflict scoring

## Context
Story-8 of v0.3 plan. AeonSleep needs conflict-aware temporal
tagging before consolidation can run.

## Approach
Sentence-transformer similarity + rule-based logic, no LLM in
hot path per spec.

## Changes
- src/cognitive/sleep_tagger.py (scorer class)
- tests/test_sleep_tagger.py (synthetic planted conflicts)

## Impact
Unblocks story-10 (consolidation) and story-11 (AeonSleep API).
EOF
)"
```

## Guardrails

See `.ralph-v0.3/guardrails.md` for constraints and boundaries.

## Important

- Complete exactly ONE story per iteration
- Do not skip quality gates
- Do not modify stories you are not implementing
- If blocked, document in `.ralph-v0.3/progress.txt` and exit
- Trust the loop script to handle the next iteration
