# Testing

## Framework

pytest with uv: `uv run python -m pytest`

## Structure

- Mirror `src/` layout: `tests/stacks/`, `tests/routing/`, `tests/eval/`
- Test files: `test_*.py`
- Fixtures in `conftest.py` at each level

## Patterns

- AAA: Arrange, Act, Assert
- One behavior per test
- Use fixtures for model/tokenizer loading (session-scoped for heavy objects)
- Mock GPU operations in CI — real GPU tests tagged `@pytest.mark.gpu`

## What to Test

- Router thresholds and stack selection logic
- Forgetting check math (angle calculation, rollback decision)
- Dataset dedup correctness
- Config loading and validation
- Bias detection triggers

## What NOT to Test

- Actual model quality (that's eval, not unit tests)
- Third-party library internals (torch, transformers)

## Anti-Patterns

- Don't load real models in unit tests — mock or use tiny stubs
- Don't share mutable state between tests
- Don't test implementation details (private methods)
- Don't write tests that pass when code is broken
