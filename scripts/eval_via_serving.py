#!/usr/bin/env python3
"""HumanEval runner driving the factory4life serving endpoint.

Counterpart to ``eval_humaneval_v4.py`` — instead of loading
``mlx_lm`` in-process and calling ``generate`` directly, this
script treats the serving layer (``POST /v1/chat/completions``)
as the unit under test. That surfaces :

- OpenAI-compat serialisation bugs (schemas / tools / streaming)
  that unit tests miss because they use Pydantic models directly.
- Dynamic-context admission pressure under a batched workload
  (164 problems back-to-back, same session-equivalent).
- KV budget pressure — the script reads
  ``X-Max-Context-Available`` after each call and logs it so
  post-run analysis shows how the ceiling moved during the eval.
- Proper parity with how ``factory4life`` agent chains will
  actually hit the model : via the HTTP API.

Usage (example — endpoint up locally or on Studio) :

    python scripts/eval_via_serving.py \\
        --base-url http://studio:8100 \\
        --model qwen3.6-35b-python \\
        --fixture /path/to/humaneval_164.jsonl \\
        --output results/humaneval-serving-python.json \\
        --label python-via-serving \\
        --n 164

The ``--api-key`` flag is a pass-through — local serving ignores
it, OpenAI-compat proxies (LiteLLM, vLLM) require it.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_via_serving")


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


# ---------------------------------------------------------------------------
# HTTP client — stdlib only so we don't force an httpx install.
# ---------------------------------------------------------------------------


def post_chat_completion(
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float = 120.0,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Send one chat completion. Returns (body, response_headers)."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.0,
        "max_completion_tokens": max_tokens,
        "seed": 0,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp_body = json.loads(resp.read().decode("utf-8"))
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        # Surface the structured error body so the caller can
        # log the reason (e.g. dynamic_ceiling) without breaking.
        err_body_raw = exc.read().decode("utf-8", errors="replace")
        try:
            err_body = json.loads(err_body_raw)
        except json.JSONDecodeError:
            err_body = {"error": {"raw": err_body_raw}}
        return (
            {"_http_error": exc.code, **err_body},
            {k.lower(): v for k, v in exc.headers.items()},
        )
    return resp_body, resp_headers


# ---------------------------------------------------------------------------
# Completion extraction + sandbox execution (borrowed from
# eval_humaneval_v4 so verdicts align across the two runners).
# ---------------------------------------------------------------------------


def _extract_completion(raw: str, prompt: str) -> str:
    text = raw
    if text.startswith(prompt):
        text = text[len(prompt):]
    if text.lstrip().startswith("```"):
        tail = text.split("\n", 1)[1] if "\n" in text else ""
        text = tail
        if text.lstrip().startswith("python\n"):
            text = text.lstrip()[len("python\n"):]
    cut = len(text)
    for s in STOP_SEQUENCES:
        i = text.find(s)
        if i != -1 and i < cut:
            cut = i
    text = text[:cut]
    lines = text.split("\n")
    if lines and lines[0].startswith("   ") and not lines[0].startswith("    "):
        lines[0] = " " + lines[0]
        text = "\n".join(lines)
    return text


def _worker(prompt, completion, test, entry_point, q) -> None:
    try:
        signal.alarm(8)
        program = prompt + completion + "\n" + test + f"\ncheck({entry_point})\n"
        ns: dict[str, Any] = {"__name__": "__test__"}
        exec(compile(program, "<humaneval>", "exec"), ns)
        q.put(("ok", ""))
    except BaseException as e:  # noqa: BLE001 — sandbox
        q.put(("fail", f"{type(e).__name__}: {e}"))
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass


def _run_tests(prompt, completion, test, entry_point) -> tuple[bool, str]:
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
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True,
                        help="e.g. http://studio:8100")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", required=True,
                        help="e.g. qwen3.6-35b-python")
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    problems: list[dict[str, Any]] = []
    with args.fixture.open() as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    problems = problems[: args.n]
    logger.info(
        "loaded %d problems from %s ; target=%s/%s",
        len(problems), args.fixture, args.base_url, args.model,
    )

    out: dict[str, Any] = {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "n_problems": len(problems),
        "max_tokens": args.max_tokens,
        "per_problem": [],
        "admission_trace": [],
    }

    passed = 0
    total_rt_s = 0.0
    for i, prob in enumerate(problems, 1):
        prompt = prob["prompt"]
        t0 = time.monotonic()
        body, headers = post_chat_completion(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
        )
        rt = time.monotonic() - t0
        total_rt_s += rt

        ceiling = headers.get("x-max-context-available")
        if ceiling is not None:
            out["admission_trace"].append(
                {"task_id": prob["task_id"], "ceiling": int(ceiling)}
            )

        if "_http_error" in body:
            logger.warning(
                "[%d/%d] %s: HTTP %d (%s)",
                i, len(problems), prob["task_id"],
                body["_http_error"],
                body.get("error", {}).get("type", "?"),
            )
            out["per_problem"].append(
                {
                    "task_id": prob["task_id"],
                    "passed": False,
                    "rt_s": round(rt, 2),
                    "error": body.get("error", {}).get("type", "unknown"),
                }
            )
            continue

        raw = ""
        try:
            raw = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            logger.warning(
                "malformed response for %s : %r",
                prob["task_id"], body,
            )

        completion = _extract_completion(raw, prompt)
        ok, err = _run_tests(
            prompt, completion, prob["test"], prob["entry_point"],
        )
        logger.info(
            "[%d/%d] %s: %s (%.1fs ceiling=%s err=%s)",
            i, len(problems), prob["task_id"],
            "PASS" if ok else "FAIL",
            rt,
            ceiling,
            (err[:60] + "...") if err and len(err) > 60 else err,
        )
        if ok:
            passed += 1
        out["per_problem"].append(
            {
                "task_id": prob["task_id"],
                "entry_point": prob["entry_point"],
                "passed": ok,
                "rt_s": round(rt, 2),
                "ceiling": int(ceiling) if ceiling else None,
                "completion": completion[:800],
                "error": err[:200] if err else "",
            }
        )

    out["pass@1"] = passed / len(problems) if problems else 0.0
    out["total_rt_s"] = round(total_rt_s, 1)
    out["status"] = "ok"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(
        "DONE label=%s pass@1=%.2f (%d/%d), total_rt=%.1fs, wrote %s",
        args.label,
        out["pass@1"],
        passed,
        len(problems),
        total_rt_s,
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
