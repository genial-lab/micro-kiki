"""Guardrail: fail if pre-pivot identifiers leak into src/**/*.py.

The 2026-04-16 pivot moved the project from Qwen3.5-4B + custom MoE-LoRA
(32 domains hard-coded as ``[0.0] * 32``, 37-slot variants in earlier drafts)
to Qwen3.5-35B-A3B + standard LoRA. Code referencing the pre-pivot base model
or the fixed-length placeholder vectors is almost certainly stale.

This validator greps ``src/**/*.py`` for any of the forbidden strings below
and fails CI if any match is found. Markdown docs (CLAUDE.md, AGENTS.md) are
excluded because they may legitimately mention old names in historical
context.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

FORBIDDEN_STRINGS: tuple[str, ...] = (
    "Qwen3.5-4B",
    "Qwen3-4B",
    "[0.0] * 32",
)


def _iter_python_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.py"))


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return list of (line_number, needle, line_text) hits for a single file."""
    hits: list[tuple[int, str, str]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                for needle in FORBIDDEN_STRINGS:
                    if needle in line:
                        hits.append((lineno, needle, line.rstrip("\n")))
    except (OSError, UnicodeDecodeError) as exc:
        # Treat unreadable .py as a hard failure — better loud than silent.
        hits.append((0, f"<read-error: {exc!s}>", ""))
    return hits


def validate() -> tuple[bool, str]:
    """Scan src/**/*.py for forbidden pre-pivot identifiers."""
    files = _iter_python_files(SRC_DIR)
    if not files:
        # Nothing under src/ is fine (could be an early checkout); not an error.
        return True, f"OK: no python files under {SRC_DIR} (nothing to scan)."

    all_hits: list[tuple[Path, int, str, str]] = []
    for path in files:
        for lineno, needle, text in scan_file(path):
            all_hits.append((path, lineno, needle, text))

    if not all_hits:
        return (
            True,
            (
                f"OK: scanned {len(files)} src/**/*.py files, no pre-pivot "
                f"identifiers found (forbidden: {list(FORBIDDEN_STRINGS)})."
            ),
        )

    lines = [f"DRIFT: {len(all_hits)} pre-pivot identifier hit(s) in src/**/*.py:"]
    for path, lineno, needle, text in all_hits:
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            rel = path
        snippet = text.strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        lines.append(f"  {rel}:{lineno}  [{needle}]  {snippet}")
    return False, "\n".join(lines)


def main() -> int:
    ok, msg = validate()
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
