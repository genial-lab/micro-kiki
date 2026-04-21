#!/usr/bin/env python3
"""HumanEval pilot eval for V4 SOTA adapters.

Loads the base Qwen3.6-35B-A3B model via mlx_lm, optionally applies a LoRA
adapter, then generates deterministic completions for a small HumanEval subset
and runs the provided unit tests in a sandbox to estimate pass@1.

Design constraint: single entry point, runs on Mac Studio (grosmac -> studio),
graceful fallback if mlx_lm is missing.

Usage (on Studio):
    python scripts/eval_humaneval_v4.py \
        --base-model /Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B \
        --adapter /Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota/python \
        --fixture /Users/clems/tmp/humaneval-run/humaneval_10.jsonl \
        --output /tmp/humaneval-python.json \
        --label python \
        --n 10
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import signal
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("humaneval_v4")

# ---------------------------------------------------------------------------
# Prompt & completion handling
# ---------------------------------------------------------------------------

STOP_SEQUENCES = (
    "\ndef ",
    "\nclass ",
    "\nif __name__",
    "\nprint(",
    "\n#",
    "\n```",
    "```",
    "\nassert ",
)


def _build_prompt(problem: dict[str, Any]) -> str:
    """Return the raw HumanEval prompt (signature + docstring)."""
    return problem["prompt"]


def _extract_completion(raw: str, prompt: str) -> str:
    """Strip the prompt echo (if any) and cut at the first stop sequence.

    Models often include the signature at the start; we keep the *body*
    candidate that composes against the original prompt.
    """
    text = raw
    # Drop prompt echo if mlx_lm returned the full sequence (defensive).
    if text.startswith(prompt):
        text = text[len(prompt):]
    # Remove fenced code block prefix if the model added one.
    if text.lstrip().startswith("```"):
        # chop first fence line
        tail = text.split("\n", 1)[1] if "\n" in text else ""
        text = tail
        # If a leading "python" language tag slipped in
        if text.lstrip().startswith("python\n"):
            text = text.lstrip()[len("python\n"):]
    # Truncate at first stop sequence.
    cut = len(text)
    for s in STOP_SEQUENCES:
        i = text.find(s)
        if i != -1 and i < cut:
            cut = i
    text = text[:cut]
    # Fix indentation: models sometimes emit 3-space first indent when body
    # should be 4 spaces. Detect and normalize.
    lines = text.split("\n")
    if lines and lines[0].startswith("   ") and not lines[0].startswith("    "):
        lines[0] = " " + lines[0]
        text = "\n".join(lines)
    return text


# ---------------------------------------------------------------------------
# Sandbox test runner
# ---------------------------------------------------------------------------


def _worker(prompt: str, completion: str, test: str, entry_point: str, q) -> None:
    """Runs inside a child process; writes (ok, err) to the queue."""
    try:
        # Tight timeout via SIGALRM inside the child
        signal.alarm(8)
        program = prompt + completion + "\n" + test + f"\ncheck({entry_point})\n"
        ns: dict[str, Any] = {"__name__": "__test__"}
        exec(compile(program, "<humaneval>", "exec"), ns)
        q.put(("ok", ""))
    except BaseException as e:  # noqa: BLE001 - sandbox
        q.put(("fail", f"{type(e).__name__}: {e}"))
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass


def _run_tests(prompt: str, completion: str, test: str, entry_point: str) -> tuple[bool, str]:
    """Execute the HumanEval check in an isolated process with a hard timeout."""
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(
        target=_worker, args=(prompt, completion, test, entry_point, q)
    )
    p.start()
    p.join(timeout=15)
    if p.is_alive():
        p.terminate()
        p.join(1)
        return False, "timeout"
    if q.empty():
        return False, "no_result"
    status, err = q.get()
    return status == "ok", err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_mlx(base_model: Path, adapter: Path | None, max_tokens: int):
    """Returns (generate_fn, tokenizer) where generate_fn(prompt)->str."""
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    logger.info("loading base model from %s", base_model)
    t0 = time.monotonic()
    model, tokenizer = load(str(base_model))
    logger.info("base loaded in %.1fs", time.monotonic() - t0)

    if adapter is not None:
        from mlx_lm.tuner.utils import load_adapters

        t1 = time.monotonic()
        logger.info("applying adapter: %s", adapter)
        load_adapters(model, str(adapter))
        logger.info("adapter applied in %.1fs", time.monotonic() - t1)

    sampler = make_sampler(temp=0.0)

    def _gen(prompt: str) -> str:
        return generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )

    return _gen


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, type=Path)
    p.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="LoRA adapter directory (containing adapters.safetensors). "
        "Omit for base-only run.",
    )
    p.add_argument("--fixture", required=True, type=Path,
                   help="JSONL file with HumanEval problems (task_id/prompt/test/entry_point)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--label", required=True, help="Run label (base|python|cpp|typescript...)")
    p.add_argument("--n", type=int, default=10, help="Number of problems (cap)")
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Stub report; do not load the model (for kxkm-ai offline test).",
    )
    args = p.parse_args()

    # Load fixture
    problems: list[dict[str, Any]] = []
    with args.fixture.open() as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    problems = problems[: args.n]
    logger.info("loaded %d problems from %s", len(problems), args.fixture)

    out: dict[str, Any] = {
        "label": args.label,
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "n_problems": len(problems),
        "max_tokens": args.max_tokens,
        "per_problem": [],
    }

    if args.dry_run:
        logger.warning("dry run — skipping mlx_lm")
        out["pass@1"] = None
        out["status"] = "dry_run"
        args.output.write_text(json.dumps(out, indent=2))
        return 0

    # Load the model
    try:
        gen = _load_mlx(args.base_model, args.adapter, args.max_tokens)
    except ImportError as e:
        logger.error("mlx_lm unavailable: %s", e)
        out["status"] = "mlx_unavailable"
        out["error"] = str(e)
        args.output.write_text(json.dumps(out, indent=2))
        return 2

    # Generate + test
    passed = 0
    total_gen_s = 0.0
    for i, prob in enumerate(problems, 1):
        prompt = _build_prompt(prob)
        t_g = time.monotonic()
        try:
            raw = gen(prompt)
        except Exception as e:  # noqa: BLE001
            logger.warning("gen failed for %s: %s", prob["task_id"], e)
            raw = ""
        gen_s = time.monotonic() - t_g
        total_gen_s += gen_s
        completion = _extract_completion(raw, prompt)
        ok, err = _run_tests(prompt, completion, prob["test"], prob["entry_point"])
        logger.info(
            "[%d/%d] %s: %s (%.1fs, err=%s)",
            i,
            len(problems),
            prob["task_id"],
            "PASS" if ok else "FAIL",
            gen_s,
            (err[:60] + "...") if err and len(err) > 60 else err,
        )
        if ok:
            passed += 1
        out["per_problem"].append(
            {
                "task_id": prob["task_id"],
                "entry_point": prob["entry_point"],
                "passed": ok,
                "gen_s": round(gen_s, 2),
                "completion": completion[:800],
                "error": err[:200] if err else "",
            }
        )

    out["pass@1"] = passed / len(problems) if problems else 0.0
    out["total_gen_s"] = round(total_gen_s, 1)
    out["status"] = "ok"
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(
        "DONE label=%s pass@1=%.2f (%d/%d), total_gen=%.1fs, wrote %s",
        args.label,
        out["pass@1"],
        passed,
        len(problems),
        total_gen_s,
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
