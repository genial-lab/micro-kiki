# Ralph Loop Task

You are executing ONE iteration of a ralph loop. Complete ONE story, then exit.

## Your Task

1. Read `.ralph/prd.json` to find the next incomplete story (passes: false, lowest priority)
2. Implement ONLY that story
3. Run quality gates
4. Commit your changes
5. Update progress
6. Exit

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
   - Set `passes: true` for the completed story

3. **Append to progress.txt:**
   ```
   [{timestamp}] Completed: {story-id} - {story-title}
   ```

4. **Exit immediately** - Do not start another story

## Guardrails

See `.ralph/guardrails.md` for constraints and boundaries.

## Important

- Complete exactly ONE story per iteration
- Do not skip quality gates
- Do not modify stories you are not implementing
- If blocked, document in progress.txt and exit
- Trust the loop script to handle the next iteration
