#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["rich"]
# ///
"""
Ralph Loop Runner — v0.3 Neuroscience edition

Executes AI iterations until all v0.3 stories complete.
Reads .ralph-v0.3/prd.json. Runs on branch `neuroscience`.
Uses rich for beautiful terminal output with progress tracking.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()

RALPH_DIR = Path(".ralph-v0.3")
PRD_FILE = RALPH_DIR / "prd.json"
PROGRESS_FILE = RALPH_DIR / "progress.txt"


def load_prd() -> dict:
    """Load and return the PRD JSON."""
    with open(PRD_FILE) as f:
        return json.load(f)


def count_stories(prd: dict) -> tuple[int, int]:
    """Return (complete, total) story counts."""
    stories = prd["stories"]
    complete = sum(1 for s in stories if s["passes"])
    return complete, len(stories)


def get_next_story(prd: dict) -> dict | None:
    """Get the next incomplete story by priority."""
    incomplete = [s for s in prd["stories"] if not s["passes"]]
    return min(incomplete, key=lambda s: s["priority"]) if incomplete else None


def verify_branch() -> str | None:
    """Return current branch name, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def run_claude() -> int:
    """Run claude and return exit code."""
    result = subprocess.run([
        "claude",
        "--dangerously-skip-permissions",
        "--print",
        "Execute ralph loop iteration per .ralph-v0.3/CLAUDE.md"
    ])
    return result.returncode


def show_header(max_iterations: int, complete: int, total: int, branch: str):
    """Display the header panel with autonomous mode warning."""
    console.print(Panel.fit(
        f"[bold cyan]Ralph Loop Runner — v0.3 Neuroscience[/]\n\n"
        f"[bold yellow]AUTONOMOUS MODE ENABLED[/]\n"
        f"[dim]Commands execute without approval[/]\n\n"
        f"[dim]Branch:[/] [magenta]{branch}[/]  "
        f"[dim]Max:[/] [yellow]{max_iterations}[/]  "
        f"[dim]Progress:[/] [green]{complete}[/]/[cyan]{total}[/]",
        border_style="blue",
        title="ralph-v0.3",
        title_align="left"
    ))


def show_iteration(iteration: int, max_iterations: int, story: dict, remaining: int):
    """Display iteration info."""
    console.print()
    console.rule(f"[bold]Iteration {iteration}/{max_iterations}[/]", style="dim")
    console.print(f"[cyan]Next:[/] {story['id']} - {story['title']}")
    console.print(f"[dim]Remaining:[/] {remaining} stories")
    console.print()


def show_progress_bar(complete: int, total: int):
    """Display a simple progress indicator."""
    pct = (complete / total * 100) if total > 0 else 0
    filled = int(pct / 5)
    bar = "#" * filled + "." * (20 - filled)
    console.print(f"[cyan]Progress:[/] [{bar}] {complete}/{total} ({pct:.0f}%)")


def show_completion():
    """Display completion panel."""
    console.print()
    console.print(Panel.fit(
        "[bold green]All v0.3 stories complete![/]",
        border_style="green"
    ))


def show_max_reached(max_iterations: int, incomplete: int):
    """Display max iterations panel."""
    console.print()
    console.print(Panel.fit(
        f"[bold yellow]Max iterations reached ({max_iterations})[/]\n"
        f"[red]{incomplete} stories still incomplete[/]",
        border_style="yellow"
    ))


def main():
    max_iterations = int(os.environ.get("MAX_ITERATIONS", "10"))

    # Verify ralph directory
    if not RALPH_DIR.exists():
        console.print("[red]Error:[/] .ralph-v0.3 directory not found")
        console.print("[dim]You must be on branch `neuroscience` at repo root[/]")
        sys.exit(1)

    if not PRD_FILE.exists():
        console.print(f"[red]Error:[/] {PRD_FILE} not found")
        sys.exit(1)

    # Verify branch
    branch = verify_branch()
    if branch != "neuroscience":
        console.print(
            f"[red]Error:[/] expected branch `neuroscience`, got `{branch}`. "
            "Switch to the neuroscience branch before running this loop."
        )
        sys.exit(1)

    # Load PRD and show header
    prd = load_prd()
    complete, total = count_stories(prd)
    show_header(max_iterations, complete, total, branch)

    # Main loop
    for iteration in range(1, max_iterations + 1):
        prd = load_prd()  # Reload each iteration
        complete, total = count_stories(prd)
        incomplete = total - complete

        if incomplete == 0:
            show_completion()
            sys.exit(0)

        next_story = get_next_story(prd)
        show_iteration(iteration, max_iterations, next_story, incomplete)

        # Run claude with visible output
        with console.status("[bold green]Starting claude...[/]", spinner="dots"):
            pass  # Brief status then let subprocess take over

        exit_code = run_claude()

        if exit_code != 0:
            console.print(f"[yellow]Warning:[/] claude exited with code {exit_code}")

        # Show updated progress
        prd = load_prd()
        complete, total = count_stories(prd)
        show_progress_bar(complete, total)

    # Max iterations reached
    prd = load_prd()
    complete, total = count_stories(prd)
    incomplete = total - complete
    show_max_reached(max_iterations, incomplete)
    sys.exit(1 if incomplete > 0 else 0)


if __name__ == "__main__":
    main()
