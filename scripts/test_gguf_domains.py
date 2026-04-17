#!/usr/bin/env python3
"""Smoke-test micro-kiki V3 GGUF across key domains via llama-server."""
import argparse
import json
import subprocess
import sys
import time

import httpx

DOMAIN_PROMPTS = {
    "python": "Write a Python function that merges two sorted lists into one sorted list.",
    "embedded": "Explain how to configure UART on an ESP32 using ESP-IDF.",
    "kicad-dsl": "Write a KiCad symbol for a 4-pin voltage regulator with VIN, VOUT, GND, EN.",
    "spice": "Write an ngspice netlist for a 2nd-order Butterworth low-pass filter at 1kHz.",
    "electronics": "What is the difference between a MOSFET and a BJT for switching applications?",
    "components": "What are the specs and pinout of the AMS1117-3.3 voltage regulator?",
    "chat-fr": "Explique-moi le principe de fonctionnement d'un pont en H.",
    "reasoning": "A circuit has three resistors: 100\u03a9, 200\u03a9, 300\u03a9 in parallel. What is the total resistance? Show your work.",
    "shell": "Write a bash one-liner that finds all .c files modified in the last 24 hours and counts their total lines.",
    "docker": "Write a multi-stage Dockerfile for a Python FastAPI app with minimal final image size.",
}


def test_domain(client: httpx.Client, domain: str, prompt: str) -> dict:
    """Send prompt to llama-server and assess response."""
    resp = client.post("/v1/chat/completions", json={
        "model": "micro-kiki-v3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    # Qwen3 thinking mode: real content may be in reasoning_content
    reasoning = msg.get("reasoning_content") or ""
    if not content and reasoning:
        content = reasoning
    tokens = data.get("usage", {})

    return {
        "domain": domain,
        "prompt_len": len(prompt),
        "response_len": len(content),
        "tokens": tokens.get("completion_tokens", 0),
        "thinking_only": bool(reasoning and not msg.get("content")),
        "degenerate": len(content) < 20 or content.count("\n") < 1,
        "preview": content[:200],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080", help="llama-server URL")
    ap.add_argument("--gguf", default=None, help="GGUF path — starts llama-server automatically")
    ap.add_argument("--llama-server", default="llama-server", help="Path to llama-server binary")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    server_proc = None
    if args.gguf:
        print(f"Starting llama-server with {args.gguf}...")
        server_proc = subprocess.Popen([
            args.llama_server, "-m", args.gguf,
            "-c", "4096", "--port", str(args.port),
            "--log-disable",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
        args.url = f"http://localhost:{args.port}"

    try:
        client = httpx.Client(base_url=args.url)
        # Health check
        health = client.get("/health")
        assert health.status_code == 200, f"Server not ready: {health.status_code}"

        print(f"{'Domain':<15} {'Resp len':>8} {'Tokens':>7} {'OK?':>4}")
        print("-" * 40)

        results = []
        for domain, prompt in DOMAIN_PROMPTS.items():
            r = test_domain(client, domain, prompt)
            results.append(r)
            ok = "FAIL" if r["degenerate"] else " OK "
            print(f"{domain:<15} {r['response_len']:>8} {r['tokens']:>7} {ok}")

        passed = sum(1 for r in results if not r["degenerate"])
        print(f"\n{passed}/{len(results)} domains passed smoke test")

        # Save detailed results
        with open("output/micro-kiki/gguf/smoke-test-results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()


if __name__ == "__main__":
    main()
