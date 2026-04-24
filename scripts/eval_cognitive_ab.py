"""Cognitive layer A/B evaluation script.

Compares responses WITH the full cognitive pipeline (Aeon recall/write,
Negotiator, AntiBias) vs WITHOUT (raw_mode=True: routing + inference only).

Usage:
    python scripts/eval_cognitive_ab.py [--host HOST] [--port PORT] \
        [--model MODEL] [--output PATH]

The server must already be running with the raw_mode-capable build.
Do NOT run this against the old server — it will 400 on the raw_mode field.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 30 test prompts across 6 domains (5 per domain)
# ---------------------------------------------------------------------------

PROMPTS: list[dict[str, str]] = [
    # embedded
    {
        "domain": "embedded",
        "prompt": "Explain the difference between FreeRTOS tasks and interrupts on STM32",
    },
    {
        "domain": "embedded",
        "prompt": "How to implement a circular DMA buffer for UART on ARM Cortex-M4",
    },
    {
        "domain": "embedded",
        "prompt": "Design a state machine for a battery management system",
    },
    {
        "domain": "embedded",
        "prompt": "What are the trade-offs between polling and interrupt-driven I/O",
    },
    {
        "domain": "embedded",
        "prompt": "Implement a watchdog timer recovery strategy for safety-critical firmware",
    },
    # spice
    {
        "domain": "spice",
        "prompt": "Write a SPICE netlist for a common-emitter amplifier with voltage divider bias",
    },
    {
        "domain": "spice",
        "prompt": "Simulate a 555 timer in astable mode using SPICE",
    },
    {
        "domain": "spice",
        "prompt": "Create a SPICE model for a Zener diode voltage regulator",
    },
    {
        "domain": "spice",
        "prompt": "How to do Monte Carlo analysis in SPICE for component tolerances",
    },
    {
        "domain": "spice",
        "prompt": "Write a SPICE testbench for a full-bridge rectifier with filter",
    },
    # python
    {
        "domain": "python",
        "prompt": "Implement a thread-safe LRU cache in Python without external libraries",
    },
    {
        "domain": "python",
        "prompt": "Write a Python decorator that retries failed async functions with exponential backoff",
    },
    {
        "domain": "python",
        "prompt": "How to implement the observer pattern in Python using weakrefs",
    },
    {
        "domain": "python",
        "prompt": "Create a Python context manager for database transaction rollback",
    },
    {
        "domain": "python",
        "prompt": "Implement a trie data structure with prefix search in Python",
    },
    # chat-fr
    {
        "domain": "chat-fr",
        "prompt": "Explique les avantages et inconvénients du télétravail pour une PME française",
    },
    {
        "domain": "chat-fr",
        "prompt": "Rédige un résumé de la Révolution française en 200 mots",
    },
    {
        "domain": "chat-fr",
        "prompt": "Quels sont les critères pour choisir une école d'ingénieur en France",
    },
    {
        "domain": "chat-fr",
        "prompt": "Compare les systèmes de retraite français et allemand",
    },
    {
        "domain": "chat-fr",
        "prompt": "Donne des conseils pour préparer un entretien d'embauche en France",
    },
    # stm32
    {
        "domain": "stm32",
        "prompt": "Configure SPI with DMA on STM32F4 using HAL for SD card communication",
    },
    {
        "domain": "stm32",
        "prompt": "How to use the STM32 ADC with DMA for multi-channel sampling",
    },
    {
        "domain": "stm32",
        "prompt": "Implement CAN bus communication between two STM32 boards",
    },
    {
        "domain": "stm32",
        "prompt": "Configure the STM32 RTC with backup domain for timestamp logging",
    },
    {
        "domain": "stm32",
        "prompt": "How to implement OTA firmware update on STM32 via UART bootloader",
    },
    # power
    {
        "domain": "power",
        "prompt": "Design a synchronous buck converter for 12V to 3.3V at 5A",
    },
    {
        "domain": "power",
        "prompt": "Calculate the inductor value for a boost converter with 95% efficiency",
    },
    {
        "domain": "power",
        "prompt": "Compare linear regulators vs switching regulators for battery-powered devices",
    },
    {
        "domain": "power",
        "prompt": "Design a MOSFET gate driver circuit for a half-bridge inverter",
    },
    {
        "domain": "power",
        "prompt": "How to implement soft-start for a DC-DC converter to limit inrush current",
    },
]

# ---------------------------------------------------------------------------
# Per-domain technical keywords for coverage scoring
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "embedded": [
        "interrupt", "task", "rtos", "dma", "buffer", "state", "watchdog",
        "cortex", "priority", "preempt", "hal", "peripheral", "irq", "isr",
        "tick", "semaphore", "mutex", "queue", "stack", "handler",
    ],
    "spice": [
        "netlist", "spice", ".ac", ".dc", ".tran", ".op", "resistor", "capacitor",
        "inductor", "transistor", "bjt", "mosfet", "voltage", "current", "node",
        "model", "subckt", "monte", "simulation", "probe",
    ],
    "python": [
        "def", "class", "async", "await", "decorator", "context", "weakref",
        "thread", "lock", "cache", "lru", "trie", "observer", "generator",
        "yield", "dataclass", "exception", "typing", "protocol",
    ],
    "chat-fr": [
        "avantage", "inconvénient", "entreprise", "travail", "france",
        "française", "retraite", "école", "ingénieur", "entretien",
        "résumé", "révolution", "critère", "système", "conseil",
    ],
    "stm32": [
        "spi", "dma", "adc", "can", "rtc", "hal", "stm32", "uart", "bootloader",
        "ota", "gpio", "clock", "prescaler", "interrupt", "register",
        "cube", "firmware", "flash", "periph", "sample",
    ],
    "power": [
        "buck", "boost", "converter", "inductor", "capacitor", "mosfet",
        "switching", "linear", "regulator", "efficiency", "duty", "cycle",
        "ripple", "gate", "driver", "soft-start", "inrush", "voltage",
        "current", "pwm",
    ],
}


# ---------------------------------------------------------------------------
# Scoring heuristics (no LLM judge — deterministic, offline-safe)
# ---------------------------------------------------------------------------


def _score_response(prompt: str, response: str, domain: str) -> dict[str, Any]:
    """Return a dict of quality signals for a single response.

    Three signals:
    - length: character count (longer ≈ more complete up to a point)
    - keyword_count: number of domain-specific keywords found in the response
    - coherence: fraction of significant query tokens that appear in response

    All are raw counts / fractions; aggregation happens in the caller.
    """
    resp_lower = response.lower()
    prompt_lower = prompt.lower()

    length = len(response)

    keywords = DOMAIN_KEYWORDS.get(domain, [])
    keyword_count = sum(1 for kw in keywords if kw in resp_lower)

    # Coherence proxy: check query content words (len > 3) found in response.
    query_tokens = [
        t.strip(".,?!;:()[]\"'")
        for t in prompt_lower.split()
        if len(t.strip(".,?!;:()[]\"'")) > 3
    ]
    if query_tokens:
        matched = sum(1 for t in query_tokens if t in resp_lower)
        coherence = matched / len(query_tokens)
    else:
        coherence = 0.0

    return {
        "length": length,
        "keyword_count": keyword_count,
        "keyword_max": len(keywords),
        "coherence": round(coherence, 4),
    }


def _composite(scores: dict[str, Any]) -> float:
    """Single composite score: weighted sum of normalised signals.

    Weights: coherence 50%, keyword coverage 30%, length 20% (capped at 2000).
    """
    coherence = scores["coherence"]
    kw_max = scores["keyword_max"] or 1
    kw_norm = scores["keyword_count"] / kw_max
    length_norm = min(scores["length"] / 2000.0, 1.0)
    return round(0.5 * coherence + 0.3 * kw_norm + 0.2 * length_norm, 4)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _build_payload(
    prompt: str,
    model: str,
    max_tokens: int,
    raw_mode: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "raw_mode": raw_mode,
    }


def _extract_text(resp_json: dict[str, Any]) -> str:
    """Pull the assistant content from an OpenAI-style response envelope."""
    try:
        return resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_eval(
    host: str,
    port: int,
    model: str,
    max_tokens: int,
    timeout: float,
    output_path: Path,
) -> None:
    base_url = f"http://{host}:{port}"
    endpoint = f"{base_url}/v1/chat/completions"

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    with httpx.Client(timeout=timeout) as client:
        # Quick health check before spending time on 30 prompts.
        try:
            health = client.get(f"{base_url}/health")
            health.raise_for_status()
            log.info("Server healthy: %s", health.json())
        except Exception as exc:  # noqa: BLE001
            log.error("Server health check failed: %s", exc)
            raise SystemExit(1) from exc

        for i, item in enumerate(PROMPTS):
            domain = item["domain"]
            prompt = item["prompt"]
            log.info("[%02d/30] domain=%s prompt=%.60s…", i + 1, domain, prompt)

            row: dict[str, Any] = {
                "index": i,
                "domain": domain,
                "prompt": prompt,
                "full": None,
                "raw": None,
                "scores": None,
                "winner": None,
            }

            # --- Full pipeline ---
            t0 = time.perf_counter()
            try:
                r = client.post(
                    endpoint,
                    json=_build_payload(prompt, model, max_tokens, raw_mode=False),
                )
                r.raise_for_status()
                full_text = _extract_text(r.json())
            except Exception as exc:  # noqa: BLE001
                log.warning("full request failed: %s", exc)
                errors.append(f"[{i}] full: {exc}")
                full_text = ""
            full_latency = time.perf_counter() - t0

            # --- Raw mode ---
            t0 = time.perf_counter()
            try:
                r = client.post(
                    endpoint,
                    json=_build_payload(prompt, model, max_tokens, raw_mode=True),
                )
                r.raise_for_status()
                raw_text = _extract_text(r.json())
            except Exception as exc:  # noqa: BLE001
                log.warning("raw request failed: %s", exc)
                errors.append(f"[{i}] raw: {exc}")
                raw_text = ""
            raw_latency = time.perf_counter() - t0

            # --- Score ---
            full_scores = _score_response(prompt, full_text, domain)
            raw_scores = _score_response(prompt, raw_text, domain)
            full_composite = _composite(full_scores)
            raw_composite = _composite(raw_scores)

            winner = (
                "full" if full_composite > raw_composite
                else "raw" if raw_composite > full_composite
                else "tie"
            )

            row.update(
                {
                    "full": {
                        "text": full_text,
                        "latency_s": round(full_latency, 3),
                        "scores": full_scores,
                        "composite": full_composite,
                    },
                    "raw": {
                        "text": raw_text,
                        "latency_s": round(raw_latency, 3),
                        "scores": raw_scores,
                        "composite": raw_composite,
                    },
                    "winner": winner,
                }
            )
            results.append(row)
            log.info(
                "  full=%.4f  raw=%.4f  winner=%s",
                full_composite,
                raw_composite,
                winner,
            )

    # ---------------------------------------------------------------------------
    # Aggregate summary
    # ---------------------------------------------------------------------------

    total = len(results)
    full_wins = sum(1 for r in results if r["winner"] == "full")
    raw_wins = sum(1 for r in results if r["winner"] == "raw")
    ties = sum(1 for r in results if r["winner"] == "tie")

    avg_full = (
        sum(r["full"]["composite"] for r in results if r["full"]) / total
        if total else 0.0
    )
    avg_raw = (
        sum(r["raw"]["composite"] for r in results if r["raw"]) / total
        if total else 0.0
    )

    # Per-domain breakdown
    domains = sorted({r["domain"] for r in results})
    domain_summary: dict[str, Any] = {}
    for d in domains:
        d_rows = [r for r in results if r["domain"] == d]
        domain_summary[d] = {
            "full_avg": round(
                sum(r["full"]["composite"] for r in d_rows) / len(d_rows), 4
            ),
            "raw_avg": round(
                sum(r["raw"]["composite"] for r in d_rows) / len(d_rows), 4
            ),
            "full_wins": sum(1 for r in d_rows if r["winner"] == "full"),
            "raw_wins": sum(1 for r in d_rows if r["winner"] == "raw"),
            "ties": sum(1 for r in d_rows if r["winner"] == "tie"),
        }

    summary = {
        "total_prompts": total,
        "full_wins": full_wins,
        "raw_wins": raw_wins,
        "ties": ties,
        "avg_composite_full": round(avg_full, 4),
        "avg_composite_raw": round(avg_raw, 4),
        "cognitive_delta": round(avg_full - avg_raw, 4),
        "errors": errors,
        "per_domain": domain_summary,
    }

    output: dict[str, Any] = {
        "meta": {
            "model": model,
            "host": host,
            "port": port,
            "max_tokens": max_tokens,
            "prompts": len(PROMPTS),
        },
        "summary": summary,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    log.info("Results written to %s", output_path)

    # Print a brief console summary.
    print("\n=== Cognitive Layer A/B Summary ===")
    print(f"Prompts evaluated : {total}")
    print(f"Full pipeline wins: {full_wins}  ({full_wins/total*100:.1f}%)")
    print(f"Raw mode wins     : {raw_wins}  ({raw_wins/total*100:.1f}%)")
    print(f"Ties              : {ties}  ({ties/total*100:.1f}%)")
    print(f"Avg composite — full: {avg_full:.4f}  raw: {avg_raw:.4f}")
    print(f"Cognitive delta (full − raw): {avg_full - avg_raw:+.4f}")
    print("\nPer-domain breakdown:")
    for d, ds in domain_summary.items():
        delta = ds["full_avg"] - ds["raw_avg"]
        print(
            f"  {d:<12} full={ds['full_avg']:.4f}  raw={ds['raw_avg']:.4f}"
            f"  delta={delta:+.4f}  wins={ds['full_wins']}/{ds['raw_wins']}"
        )
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    print(f"\nFull results: {output_path}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="A/B eval: full cognitive pipeline vs raw_mode"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=9200, help="Server port")
    parser.add_argument(
        "--model",
        default="kiki-meta-coding",
        help="Model alias to test",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        dest="max_tokens",
        help="Max tokens per response",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--output",
        default="results/eval_cognitive_ab.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    run_eval(
        host=args.host,
        port=args.port,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
