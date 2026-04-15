#!/usr/bin/env python3
"""Probe HuggingFace + ModelScope for SpikingBrain checkpoints.

Emits ``results/spikingbrain-probe.json`` with the acquisition decision
per ``docs/specs/spikingbrain-acquisition.md`` section 7.

Usage:

    uv run python scripts/probe_spikingbrain_hf.py

Offline-safe: on network failure the script still writes a probe report
with ``decision = "offline"`` and notes listing the queries attempted.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys
import urllib.error
import urllib.request
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = RESULTS_DIR / "spikingbrain-probe.json"

HF_QUERIES = [
    "https://huggingface.co/api/models?search=spikingbrain&limit=50",
    "https://huggingface.co/api/models?author=BICLab&limit=50",
]
MS_KNOWN_REPOS = [
    "Panyuqi/V1-7B-base",
    "Panyuqi/V1-7B-sft-s3-reasoning",
    "Abel2076/SpikingBrain-7B-W8ASpike",
    "sherry12334/SpikingBrain-7B-VL",
    "Panyuqi/V1-76B-base",
    "Panyuqi/V1-76B-A12B-sft",
]
TIMEOUT_S = 10.0


def _get_json(url: str) -> tuple[int, Any | None, str | None]:
    req = urllib.request.Request(
        url, headers={"User-Agent": "micro-kiki-probe/0.3 (+contact@saillant.cc)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="replace")
            return status, json.loads(body), None
    except urllib.error.HTTPError as exc:
        return exc.code, None, f"HTTPError {exc.code}: {exc.reason}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return 0, None, f"{type(exc).__name__}: {exc}"


def probe_hf() -> list[dict[str, Any]]:
    repos: list[dict[str, Any]] = []
    for url in HF_QUERIES:
        status, payload, err = _get_json(url)
        if payload is None or not isinstance(payload, list):
            repos.append({"query": url, "status": status, "error": err})
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            repos.append(
                {
                    "query": url,
                    "status": status,
                    "repo_id": item.get("modelId") or item.get("id"),
                    "downloads": item.get("downloads"),
                    "likes": item.get("likes"),
                    "private": item.get("private", False),
                }
            )
    return repos


def probe_modelscope() -> list[dict[str, Any]]:
    repos: list[dict[str, Any]] = []
    for repo_id in MS_KNOWN_REPOS:
        url = f"https://modelscope.cn/api/v1/models/{repo_id}"
        status, payload, err = _get_json(url)
        if payload is None or not isinstance(payload, dict):
            repos.append({"query": url, "status": status, "error": err, "probed_id": repo_id})
            continue
        data = payload.get("Data") or payload.get("data") or payload
        if not isinstance(data, dict):
            data = {}
        repos.append(
            {
                "query": url,
                "status": status,
                "repo_id": repo_id,
                "exists": bool(data.get("IsPublished") or data.get("IsAccessible")),
                "license": data.get("License"),
                "downloads": data.get("Downloads"),
                "model_id": data.get("Id"),
            }
        )
    return repos


def decide(hf_repos: list[dict[str, Any]], ms_repos: list[dict[str, Any]]) -> tuple[str, list[str]]:
    notes: list[str] = []
    def _matches(repo_id: str, needles: tuple[str, ...]) -> bool:
        rid = repo_id.lower()
        return any(n.lower() in rid for n in needles)

    def _reachable(r: dict[str, Any]) -> bool:
        if not r.get("repo_id"):
            return False
        if "modelscope.cn" in str(r.get("query", "")) and not r.get("exists"):
            return False
        return True

    has_76b = any(
        _matches(str(r.get("repo_id", "")), ("76b", "v1-76", "spikingbrain-76"))
        for r in hf_repos + ms_repos
        if _reachable(r)
    )
    has_7b = any(
        _matches(str(r.get("repo_id", "")), ("v1-7b", "spikingbrain-7b", "spikingbrain-v1-7b"))
        for r in hf_repos + ms_repos
        if _reachable(r)
    )
    all_probes = hf_repos + ms_repos
    net_errors = [r for r in all_probes if r.get("error")]
    repo_hits = [r for r in all_probes if r.get("repo_id")]
    if net_errors and not repo_hits:
        return "offline", [
            "all probe endpoints failed or returned no repo_id",
            "re-run with network access before taking any path",
        ]
    if has_76b:
        notes.append("76B checkpoint located — take primary path per spec section 3")
        return "primary-76b", notes
    if has_7b:
        notes.append("only 7B family public — take Fallback A per spec section 4")
        notes.append("re-run monthly until v0.3 freeze (2026-06-01) in case 76B lands")
        return "fallback-a-7b", notes
    notes.append("no BICLab checkpoints reachable — consider Fallback B (Spikingformer)")
    return "fallback-b-spikingformer", notes


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hf_repos = probe_hf()
    ms_repos = probe_modelscope()
    decision, notes = decide(hf_repos, ms_repos)
    report = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "spec": "docs/specs/spikingbrain-acquisition.md",
        "hf_repos": hf_repos,
        "modelscope_repos": ms_repos,
        "decision": decision,
        "notes": notes,
    }
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PATH}")
    print(f"decision: {decision}")
    for note in notes:
        print(f"  - {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
