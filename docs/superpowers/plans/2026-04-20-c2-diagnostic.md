# C2 Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce post-hoc per-domain + correctness-stratified + top-10 qualitative analyses of `results/c2-downstream.json` (100 per-query records) and a paper-facing narrative, with zero new compute.

**Architecture:** Single Python script `scripts/c2_diagnostic.py` exposes three pure functions (`analyze_per_domain`, `analyze_stratified`, `top_10_by_gap`) tested independently, plus a `main()` orchestrator that loads the JSON, runs all three, renders two matplotlib PDFs, writes a top-10 markdown dump, and a machine-readable JSON summary.

**Tech Stack:** Python 3.14, stdlib `json` + `argparse`, numpy (aggregation only), matplotlib (figures). pytest for unit tests. No new deps.

Spec: `docs/superpowers/specs/2026-04-20-c2-diagnostic-design.md`.

---

## File Structure

**Files to create:**
- `scripts/c2_diagnostic.py` — 3 analysis functions + CLI orchestrator (~160 lines)
- `tests/scripts/test_c2_diagnostic.py` — 4 unit tests on synthetic fixtures (~100 lines)
- `results/c2-diagnostic.json` — machine-readable output
- `docs/paper-a/c2-diagnostic-per-domain.pdf` — per-domain gap figure
- `docs/paper-a/c2-diagnostic-stratified.pdf` — stratified-bucket figure
- `docs/paper-a/c2-diagnostic-top10.md` — auto-generated top-10 pairs + hand-written "Patterns observed"
- `docs/paper-a/c2-diagnostic.md` — paper-facing narrative

**Files to modify (optional, post-review):**
- `docs/paper-a/paper-a-v2.tex` — add §5.1 subsection citing diagnostic findings (separate commit).

---

### Task 1: Failing unit tests for the three analysis functions

**Files:**
- Create: `tests/scripts/test_c2_diagnostic.py`

- [ ] **Step 1: Create the test file**

Write to `tests/scripts/test_c2_diagnostic.py`:

```python
"""Unit tests for scripts/c2_diagnostic.py — synthetic fixtures only."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _mock_run(queries):
    """Build a c2-downstream.json-shaped dict for one router from a list of queries.

    Each query dict must contain: expected_domain, routed_domain, correct_route,
    score, question, answer.
    """
    correct = [r["score"] for r in queries if r["correct_route"]]
    wrong = [r["score"] for r in queries if not r["correct_route"]]
    n = len(queries)
    return {
        "per_query": queries,
        "mean_score": sum(r["score"] for r in queries) / n,
        "routing_accuracy": sum(r["correct_route"] for r in queries) / n,
        "mean_score_when_routed_correct": sum(correct) / len(correct) if correct else 0.0,
        "mean_score_when_routed_wrong": sum(wrong) / len(wrong) if wrong else 0.0,
        "n_queries": n,
    }


def _mock_downstream(per_query_by_router: dict[str, list[dict]]):
    """Build a full c2-downstream.json-shaped dict across 3 routers."""
    return {
        "config": {"n_eval": len(per_query_by_router["oracle"]), "self_judging": True},
        "results": {router: _mock_run(qs) for router, qs in per_query_by_router.items()},
    }


def _q(domain, routed, score, question="q", answer="a"):
    return {
        "question": question,
        "expected_domain": domain,
        "routed_domain": routed,
        "correct_route": domain == routed,
        "score": score,
        "answer": answer,
    }


def test_per_domain_mean_gap_single_query_per_domain():
    from scripts.c2_diagnostic import analyze_per_domain

    # 3 domains × 1 query; oracle perfect (5), vqc routes wrong (1), random (3)
    data = _mock_downstream({
        "oracle": [_q("a", "a", 5), _q("b", "b", 5), _q("c", "c", 5)],
        "vqc":    [_q("a", "b", 1), _q("b", "c", 1), _q("c", "a", 1)],
        "random": [_q("a", "x", 3), _q("b", "x", 3), _q("c", "x", 3)],
    })
    out = analyze_per_domain(data, domains=["a", "b", "c"])
    assert set(out.keys()) == {"a", "b", "c"}
    for d in ["a", "b", "c"]:
        assert out[d]["mean_oracle"] == 5.0
        assert out[d]["mean_vqc"] == 1.0
        assert out[d]["mean_random"] == 3.0
        assert out[d]["gap_oracle_vs_vqc"] == 4.0
        assert out[d]["gap_vqc_vs_random"] == -2.0


def test_stratified_two_buckets_mutually_exclusive():
    from scripts.c2_diagnostic import analyze_stratified

    # 4 queries: vqc correct on 2, wrong on 2. oracle always correct.
    data = _mock_downstream({
        "oracle": [_q("a","a",5), _q("b","b",4), _q("c","c",4), _q("d","d",3)],
        "vqc":    [_q("a","a",3), _q("b","b",4), _q("c","x",1), _q("d","y",1)],
        "random": [_q("a","a",4), _q("b","x",2), _q("c","c",5), _q("d","x",3)],
    })
    out = analyze_stratified(data)
    assert out["vqc_correct"]["n"] == 2
    assert out["vqc_wrong"]["n"] == 2
    # In vqc_correct bucket (queries 0 and 1): vqc scores [3,4], mean=3.5
    assert out["vqc_correct"]["vqc_mean_score"] == 3.5
    # In vqc_wrong bucket (queries 2 and 3): vqc scores [1,1], mean=1.0
    assert out["vqc_wrong"]["vqc_mean_score"] == 1.0
    # Totals should cover all 4 queries
    assert out["vqc_correct"]["n"] + out["vqc_wrong"]["n"] == 4


def test_top_10_gap_sorted_desc_ties_stable():
    from scripts.c2_diagnostic import top_10_by_gap

    # 12 queries with distinct gaps + 2 ties at the boundary
    queries_oracle = [_q("a", "a", 5, question=f"q{i}") for i in range(12)]
    vqc_scores = [5, 5, 5, 5, 5, 5, 5, 4, 4, 3, 2, 1]  # gaps = [0]*7 + [1,1,2,3,4]
    queries_vqc = [_q("a", "a", s, question=f"q{i}") for i, s in enumerate(vqc_scores)]
    data = _mock_downstream({
        "oracle": queries_oracle,
        "vqc":    queries_vqc,
        "random": [_q("a","a",3, question=f"q{i}") for i in range(12)],
    })
    out = top_10_by_gap(data, k=10)
    assert len(out) == 10
    # Top gap is q11 (oracle 5 - vqc 1 = 4)
    assert out[0]["question"] == "q11"
    assert out[0]["gap"] == 4
    # Q10 next (gap 3)
    assert out[1]["question"] == "q10"
    # The 10th slot covers gap=0 queries; ties resolved by original index → q0 first
    # Since 7 queries have gap=0 (indices 0..6) and only 5 slots remain (0,1,2,3,4 after gaps 4,3,2,1,1),
    # slots 6..9 are indices 0,1,2,3 — stable by input order
    assert all(o["gap"] >= 0 for o in out)


def test_full_pipeline_writes_json_and_figures(tmp_path):
    from scripts.c2_diagnostic import run

    # Minimal input: 2 domains × 3 queries each = 6 per router
    input_path = tmp_path / "c2-downstream.json"
    data = _mock_downstream({
        "oracle": [_q("a","a",5), _q("a","a",4), _q("a","a",5),
                   _q("b","b",3), _q("b","b",4), _q("b","b",5)],
        "vqc":    [_q("a","b",2), _q("a","a",3), _q("a","b",1),
                   _q("b","a",1), _q("b","b",4), _q("b","a",2)],
        "random": [_q("a","x",3), _q("a","y",3), _q("a","x",4),
                   _q("b","y",2), _q("b","x",3), _q("b","y",3)],
    })
    import json as _json
    input_path.write_text(_json.dumps(data))

    out_json = tmp_path / "c2-diagnostic.json"
    out_per_domain = tmp_path / "per-domain.pdf"
    out_stratified = tmp_path / "stratified.pdf"
    out_top10 = tmp_path / "top10.md"
    rc = run(
        input_path=input_path,
        out_json=out_json,
        out_per_domain_pdf=out_per_domain,
        out_stratified_pdf=out_stratified,
        out_top10_md=out_top10,
        domains=["a", "b"],
    )
    assert rc == 0
    assert out_json.exists()
    assert out_per_domain.exists()
    assert out_stratified.exists()
    assert out_top10.exists()
    report = _json.loads(out_json.read_text())
    assert set(report.keys()) == {"per_domain", "stratified", "top_gaps", "config"}
    assert set(report["per_domain"].keys()) == {"a", "b"}
```

- [ ] **Step 2: Run to verify all fail**

Run: `uv run python -m pytest tests/scripts/test_c2_diagnostic.py -v 2>&1 | tail -12`
Expected: 4/4 FAIL with `ModuleNotFoundError: No module named 'scripts.c2_diagnostic'`.

- [ ] **Step 3: Commit**

```bash
git add tests/scripts/test_c2_diagnostic.py
git commit -m "test(c2-diag): analysis + pipeline tests (red)"
```

Subject ≤50 chars, no Co-Authored-By.

---

### Task 2: Implement `analyze_per_domain`

**Files:**
- Create: `scripts/c2_diagnostic.py` (progressively — start with this function)

- [ ] **Step 1: Create the script with only the imports + `analyze_per_domain`**

Write to `scripts/c2_diagnostic.py`:

```python
#!/usr/bin/env python3
"""Post-hoc diagnostic on results/c2-downstream.json.

Produces three analyses (per-domain gap, correctness-stratified, top-10
qualitative) plus two matplotlib PDFs and a machine-readable JSON summary.

Usage:
    uv run python scripts/c2_diagnostic.py \\
        --input results/c2-downstream.json \\
        --out-json results/c2-diagnostic.json \\
        --out-per-domain-pdf docs/paper-a/c2-diagnostic-per-domain.pdf \\
        --out-stratified-pdf docs/paper-a/c2-diagnostic-stratified.pdf \\
        --out-top10-md docs/paper-a/c2-diagnostic-top10.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_DEFAULT_DOMAINS = [
    "dsp", "electronics", "emc", "embedded", "freecad",
    "kicad-dsl", "platformio", "power", "spice", "stm32",
]


def analyze_per_domain(data: dict, domains: list[str]) -> dict:
    """Return per-domain mean scores and gaps for vqc, oracle, random routers."""
    out: dict = {}
    for d in domains:
        per_domain: dict[str, float] = {}
        for router in ("oracle", "vqc", "random"):
            records = [
                r for r in data["results"][router]["per_query"]
                if r["expected_domain"] == d
            ]
            if not records:
                per_domain[f"mean_{router}"] = 0.0
                continue
            per_domain[f"mean_{router}"] = sum(r["score"] for r in records) / len(records)
        per_domain["gap_oracle_vs_vqc"] = per_domain["mean_oracle"] - per_domain["mean_vqc"]
        per_domain["gap_vqc_vs_random"] = per_domain["mean_vqc"] - per_domain["mean_random"]
        per_domain["n"] = sum(
            1 for r in data["results"]["oracle"]["per_query"] if r["expected_domain"] == d
        )
        out[d] = per_domain
    return out
```

- [ ] **Step 2: Run unit test for this function**

Run: `uv run python -m pytest tests/scripts/test_c2_diagnostic.py::test_per_domain_mean_gap_single_query_per_domain -v 2>&1 | tail -8`
Expected: PASSED.

- [ ] **Step 3: Commit**

```bash
git add scripts/c2_diagnostic.py
git commit -m "feat(c2-diag): analyze_per_domain"
```

---

### Task 3: Implement `analyze_stratified`

**Files:**
- Modify: `scripts/c2_diagnostic.py` (append function)

- [ ] **Step 1: Append `analyze_stratified` to `scripts/c2_diagnostic.py`**

Add immediately after `analyze_per_domain`:

```python
def analyze_stratified(data: dict) -> dict:
    """Split queries by VQC correctness, report per-router mean score in each bucket."""
    vqc_per_query = data["results"]["vqc"]["per_query"]
    oracle_per_query = data["results"]["oracle"]["per_query"]
    random_per_query = data["results"]["random"]["per_query"]

    correct_idx = [i for i, r in enumerate(vqc_per_query) if r["correct_route"]]
    wrong_idx = [i for i, r in enumerate(vqc_per_query) if not r["correct_route"]]

    def _bucket(indices: list[int]) -> dict:
        if not indices:
            return {
                "n": 0,
                "vqc_mean_score": 0.0,
                "oracle_mean_score": 0.0,
                "random_mean_score": 0.0,
            }
        return {
            "n": len(indices),
            "vqc_mean_score": sum(vqc_per_query[i]["score"] for i in indices) / len(indices),
            "oracle_mean_score": sum(oracle_per_query[i]["score"] for i in indices) / len(indices),
            "random_mean_score": sum(random_per_query[i]["score"] for i in indices) / len(indices),
        }

    return {
        "vqc_correct": _bucket(correct_idx),
        "vqc_wrong": _bucket(wrong_idx),
    }
```

- [ ] **Step 2: Run unit test**

Run: `uv run python -m pytest tests/scripts/test_c2_diagnostic.py::test_stratified_two_buckets_mutually_exclusive -v 2>&1 | tail -8`
Expected: PASSED.

- [ ] **Step 3: Commit**

```bash
git add scripts/c2_diagnostic.py
git commit -m "feat(c2-diag): analyze_stratified buckets"
```

---

### Task 4: Implement `top_10_by_gap`

**Files:**
- Modify: `scripts/c2_diagnostic.py` (append function)

- [ ] **Step 1: Append `top_10_by_gap` to `scripts/c2_diagnostic.py`**

Add after `analyze_stratified`:

```python
def top_10_by_gap(data: dict, k: int = 10) -> list[dict]:
    """Return top-k queries by oracle_score - vqc_score, descending, ties stable."""
    oracle_pq = data["results"]["oracle"]["per_query"]
    vqc_pq = data["results"]["vqc"]["per_query"]
    random_pq = data["results"]["random"]["per_query"]

    rows = []
    for i, (o, v, r) in enumerate(zip(oracle_pq, vqc_pq, random_pq)):
        rows.append({
            "index": i,
            "question": o["question"],
            "expected_domain": o["expected_domain"],
            "oracle_score": o["score"],
            "oracle_answer": o["answer"],
            "vqc_routed_domain": v["routed_domain"],
            "vqc_score": v["score"],
            "vqc_answer": v["answer"],
            "random_routed_domain": r["routed_domain"],
            "random_score": r["score"],
            "random_answer": r["answer"],
            "gap": o["score"] - v["score"],
        })

    # Sort: primary key = gap desc, secondary = index asc (stability)
    rows.sort(key=lambda x: (-x["gap"], x["index"]))
    return rows[:k]
```

- [ ] **Step 2: Run unit test**

Run: `uv run python -m pytest tests/scripts/test_c2_diagnostic.py::test_top_10_gap_sorted_desc_ties_stable -v 2>&1 | tail -8`
Expected: PASSED.

- [ ] **Step 3: Commit**

```bash
git add scripts/c2_diagnostic.py
git commit -m "feat(c2-diag): top_10_by_gap"
```

---

### Task 5: Orchestrator `run()` + figures + CLI

**Files:**
- Modify: `scripts/c2_diagnostic.py` (append orchestrator + main)

- [ ] **Step 1: Append the orchestrator + figure helpers + CLI**

Add after `top_10_by_gap`:

```python
def _render_per_domain_pdf(per_domain: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    domains_sorted = sorted(per_domain.keys(),
                            key=lambda d: per_domain[d]["gap_oracle_vs_vqc"],
                            reverse=True)
    ov = [per_domain[d]["gap_oracle_vs_vqc"] for d in domains_sorted]
    vr = [per_domain[d]["gap_vqc_vs_random"] for d in domains_sorted]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(domains_sorted))
    width = 0.38
    ax.bar(x - width / 2, ov, width, label="oracle - vqc", color="#3366cc", edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, vr, width, label="vqc - random", color="#cc3366", edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(domains_sorted, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score gap (rubric points)")
    ax.set_title("C2 diagnostic: per-domain score gaps")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_stratified_pdf(stratified: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    buckets = ["vqc_correct", "vqc_wrong"]
    routers = ["vqc", "oracle", "random"]
    values = np.array([
        [stratified[b][f"{r}_mean_score"] for r in routers] for b in buckets
    ])  # shape (2, 3)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(buckets))
    width = 0.27
    colors = {"vqc": "#6699ff", "oracle": "#66bb77", "random": "#bbbbbb"}
    for j, r in enumerate(routers):
        ax.bar(x + (j - 1) * width, values[:, j], width, label=r, color=colors[r],
               edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={stratified[b]['n']})" for b in buckets])
    ax.set_ylabel("Mean judge score (0-5)")
    ax.set_ylim(0, 5)
    ax.set_title("C2 diagnostic: scores stratified by VQC routing correctness")
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_top10_md(top_gaps: list[dict], out_path: Path) -> None:
    lines = ["# C2 diagnostic — Top-10 queries by oracle-vqc score gap",
             "",
             "*Auto-generated; append human-observed patterns at the bottom.*",
             ""]
    for k, row in enumerate(top_gaps, 1):
        lines.extend([
            f"## #{k} — gap = {row['gap']}",
            "",
            f"**Question:** {row['question']}",
            "",
            f"**Expected domain:** `{row['expected_domain']}`",
            "",
            f"### Oracle (routed to `{row['expected_domain']}`, score {row['oracle_score']})",
            "",
            "> " + row["oracle_answer"][:500].replace("\n", "\n> "),
            "",
            f"### VQC (routed to `{row['vqc_routed_domain']}`, score {row['vqc_score']})",
            "",
            "> " + row["vqc_answer"][:500].replace("\n", "\n> "),
            "",
            f"### Random (routed to `{row['random_routed_domain']}`, score {row['random_score']})",
            "",
            "> " + row["random_answer"][:500].replace("\n", "\n> "),
            "",
            "---",
            "",
        ])
    lines.extend([
        "## Patterns observed (hand-written by reviewer)",
        "",
        "_Edit after reading the 10 pairs above. Candidate patterns: persona mismatch,",
        "technical depth gap, answer length/tone, off-topic drift, hallucinated code._",
        "",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def run(
    *,
    input_path: Path,
    out_json: Path,
    out_per_domain_pdf: Path,
    out_stratified_pdf: Path,
    out_top10_md: Path,
    domains: list[str],
) -> int:
    data = json.loads(Path(input_path).read_text())
    per_domain = analyze_per_domain(data, domains=domains)
    stratified = analyze_stratified(data)
    top_gaps = top_10_by_gap(data, k=10)

    _render_per_domain_pdf(per_domain, out_per_domain_pdf)
    _render_stratified_pdf(stratified, out_stratified_pdf)
    _render_top10_md(top_gaps, out_top10_md)

    # Strip the large answer fields from the JSON summary
    top_gaps_slim = [
        {k: v for k, v in row.items() if not k.endswith("_answer")}
        for row in top_gaps
    ]

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps({
        "per_domain": per_domain,
        "stratified": stratified,
        "top_gaps": top_gaps_slim,
        "config": {
            "domains": domains,
            "input": str(input_path),
            "n_queries": len(data["results"]["oracle"]["per_query"]),
        },
    }, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-per-domain-pdf", type=Path, required=True)
    p.add_argument("--out-stratified-pdf", type=Path, required=True)
    p.add_argument("--out-top10-md", type=Path, required=True)
    p.add_argument("--domains", default=",".join(_DEFAULT_DOMAINS))
    args = p.parse_args()

    return run(
        input_path=args.input,
        out_json=args.out_json,
        out_per_domain_pdf=args.out_per_domain_pdf,
        out_stratified_pdf=args.out_stratified_pdf,
        out_top10_md=args.out_top10_md,
        domains=[d.strip() for d in args.domains.split(",") if d.strip()],
    )


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run all 4 unit tests to verify**

Run: `uv run python -m pytest tests/scripts/test_c2_diagnostic.py -v 2>&1 | tail -12`
Expected: 4/4 PASSED.

- [ ] **Step 3: Commit**

```bash
git add scripts/c2_diagnostic.py
git commit -m "feat(c2-diag): orchestrator + figures + CLI"
```

---

### Task 6: Real-data run — generate JSON + 2 PDFs + top-10 dump

**Files:**
- Create: `results/c2-diagnostic.json`, `docs/paper-a/c2-diagnostic-per-domain.pdf`, `docs/paper-a/c2-diagnostic-stratified.pdf`, `docs/paper-a/c2-diagnostic-top10.md`

- [ ] **Step 1: Run the diagnostic on the real C2 JSON**

Run:

```bash
uv run python scripts/c2_diagnostic.py \
    --input results/c2-downstream.json \
    --out-json results/c2-diagnostic.json \
    --out-per-domain-pdf docs/paper-a/c2-diagnostic-per-domain.pdf \
    --out-stratified-pdf docs/paper-a/c2-diagnostic-stratified.pdf \
    --out-top10-md docs/paper-a/c2-diagnostic-top10.md
```

Expected: exits 0, four files created. Runtime <5s.

- [ ] **Step 2: Sanity-check the JSON**

Run: `jq '{stratified, per_domain_max_gap: (.per_domain | to_entries | max_by(.value.gap_oracle_vs_vqc) | {domain: .key, gap: .value.gap_oracle_vs_vqc})}' results/c2-diagnostic.json`

Expected output shape:
- `stratified.vqc_correct.n + stratified.vqc_wrong.n == 100`
- Max per-domain `gap_oracle_vs_vqc` in [0.0, 5.0]
- Max domain's VQC mean at that domain roughly lower than oracle's.

**Kill criterion check:** if `max(gap_oracle_vs_vqc) < 0.5` AND every `|gap_vqc_vs_random[d]| < 0.3`, document the negative diagnostic in Task 7 and STOP before writing the sibling LoRA spec (per design doc).

- [ ] **Step 3: Commit the generated artefacts**

```bash
git add results/c2-diagnostic.json \
  docs/paper-a/c2-diagnostic-per-domain.pdf \
  docs/paper-a/c2-diagnostic-stratified.pdf \
  docs/paper-a/c2-diagnostic-top10.md
git commit -m "results(c2-diag): real-data analyses generated"
```

---

### Task 7: Hand-written "Patterns observed" + paper-facing narrative

**Files:**
- Modify: `docs/paper-a/c2-diagnostic-top10.md` (append "Patterns observed" human section)
- Create: `docs/paper-a/c2-diagnostic.md` (paper-facing narrative)

- [ ] **Step 1: Read the 10 pairs in `docs/paper-a/c2-diagnostic-top10.md`**

Open the file. For each of the 10 pairs, note concretely what differs between VQC and Oracle answers:
- Is VQC drifting off-topic (persona mismatch consequence)?
- Is VQC shorter/less technical?
- Does VQC hallucinate code or references?
- Is the routing mistake a neighbouring domain (dsp↔electronics) or far (dsp↔freecad)?

- [ ] **Step 2: Replace the placeholder "Patterns observed" section**

In `docs/paper-a/c2-diagnostic-top10.md`, replace the auto-inserted placeholder block (from `## Patterns observed (hand-written by reviewer)` to end of file) with a concrete bullet list of the patterns observed. Example structure to follow (content must reflect actual reading):

```markdown
## Patterns observed (hand-written by reviewer)

1. **Persona drift on adjacent domains (N/10 pairs)**: when VQC routes `dsp → electronics`, the Qwen LLM commits to the electronics expert persona and answers the DSP question with analog-circuit framing, missing the filter/FFT substance. Example: #3.

2. **Short answers on wrong persona (M/10)**: VQC answers are visibly shorter when the persona conflicts with the question's technical stack. Example: #5.

(Continue with actual observations...)

## Routing errors classified

- Adjacent-domain misrouting (dsp↔electronics, kicad↔electronics): K/10
- Far-domain misrouting (dsp↔freecad, power↔kicad): L/10

## Implications for diagnostic report

The X+Y pattern dominates the top-10 gap, confirming that the confidently-wrong pathology is driven by [persona commitment / technical-depth loss / answer truncation / other].
```

- [ ] **Step 3: Write the paper-facing narrative `docs/paper-a/c2-diagnostic.md`**

Create the file with this structure (fill concrete numbers from `results/c2-diagnostic.json`):

```markdown
# C2 Diagnostic — Why does routing fail downstream?

**Context.** The Phase C2 downstream eval triggered the kill criterion (oracle − random = 0.29 < 0.30) and revealed a confidently-wrong pathology where VQC (routing_acc 0.17) produced lower mean score (2.65) than Random (3.19). This diagnostic analyses the existing 100 per-query records (`results/c2-downstream.json`) to locate the effect, without running any new LLM calls.

## Per-domain breakdown

Figure `c2-diagnostic-per-domain.pdf`: for each of the 10 domains, the oracle-minus-vqc and vqc-minus-random score gaps.

Key observations (populate from `results/c2-diagnostic.json`):
- Maximum oracle-vs-vqc gap: {MAX_OVV_DOMAIN} at Δ={MAX_OVV}.
- Domains where VQC beats random (`gap_vqc_vs_random > 0`): {LIST or NONE}.
- Domains where VQC is strictly harmful (`gap_vqc_vs_random < −0.5`): {LIST}.

Interpretation: if the gap is concentrated on 2-3 domains, the confidently-wrong pathology is localised (likely adjacent-domain misrouting); if uniform, it is systemic.

## Correctness stratification

Figure `c2-diagnostic-stratified.pdf`: mean scores of the three routers on (a) queries where VQC routed correctly, (b) queries where VQC routed wrong.

Key numbers:
- Bucket A (vqc_correct, n={n_A}): vqc={vqc_A}, oracle={oracle_A}, random={random_A}.
- Bucket B (vqc_wrong, n={n_B}): vqc={vqc_B}, oracle={oracle_B}, random={random_B}.

**Pathology test:** in bucket B, if `vqc_B < random_B`, confidently-wrong is confirmed at stratified level (not aggregation artefact). Measured gap: {random_B - vqc_B}.

## Qualitative top-10 review

See `c2-diagnostic-top10.md` for the 10 queries with the largest oracle-minus-vqc gap. The "Patterns observed" section summarises the dominant failure modes identified by human review.

## Implications for Paper A §5

1. Paper A §5 "Discussion and Limitations" adds a subsection citing this diagnostic: the confidently-wrong pathology is {localised on N domains / systemic}, driven primarily by {persona drift / technical-depth loss / other} as evidenced by the top-10 qualitative review.

2. The sibling sub-project (real LoRA adapter experiment) is {justified / not justified} by the diagnostic:
   - If gap concentrated and driven by persona drift → real LoRA adapters with weight-level specialisation may close it. Proceed with the sibling spec.
   - If gap systemic and driven by LLM's intrinsic failure to leverage the expert hint → real LoRA adapters unlikely to help. Document, do NOT proceed.

3. The diagnostic itself is a {strength / weakness} of Paper A: it shows explicit, non-hand-wavy reasoning about WHY a negative result occurred, rather than presenting C2 as inexplicable.

## Kill-criterion check for the diagnostic itself

- `max(gap_oracle_vs_vqc) = {MAX_OVV}` — {> 0.5 so diagnostic is load-bearing / < 0.5 so pathology is at noise floor}
- `range(gap_vqc_vs_random) = {R}` — {> 0.3 across domains so effect is real / ≤ 0.3 so effect is noise}

**Verdict:** {diagnostic is load-bearing / diagnostic is at noise floor — C2 is underpowered rather than revealing a real pathology}.
```

Fill in the placeholder values `{...}` with measured numbers from `results/c2-diagnostic.json`. Delete any `{...}` that remains after editing.

- [ ] **Step 4: Commit hand-written content**

```bash
git add docs/paper-a/c2-diagnostic-top10.md docs/paper-a/c2-diagnostic.md
git commit -m "docs(c2-diag): patterns + paper narrative"
```

---

### Task 8: Push + verify no regression

- [ ] **Step 1: Check correct branch**

Run: `git branch --show-current`
Expected: `main`. If NOT main, run `git log --oneline -5` to identify your diagnostic commits, then checkout main and `git cherry-pick <sha>` for each commit.

- [ ] **Step 2: Run the full pytest suite on the diagnostic tests + existing quick tests**

Run:
```bash
uv run python -m pytest tests/scripts/test_c2_diagnostic.py \
  tests/routing/test_classical_baselines.py \
  tests/routing/test_llm_judge.py \
  tests/routing/test_downstream_harness.py -q 2>&1 | tail -6
```
Expected: all PASSED.

- [ ] **Step 3: Push**

```bash
git push origin main
```
Expected: clean push (no "Everything up-to-date" confusion — a new commit range).

---

## Self-review

**Spec coverage:**
- Analysis B (per-domain) ↔ Task 2 ✓
- Analysis C (stratified) ↔ Task 3 ✓
- Analysis E (top-10 qualitative) ↔ Tasks 4 + 7 ✓
- 4 output file types (JSON, 2 PDFs, top-10 MD) ↔ Task 5 orchestrator + Task 6 real run ✓
- Paper-facing narrative ↔ Task 7 ✓
- Unit tests for all three analyses + integration ↔ Task 1 (4 tests) ✓
- Kill criterion ↔ Task 6 Step 2 inline check + Task 7 Step 3 narrative ✓

No gap.

**Placeholder scan:** All code steps contain full code. No "TBD"/"TODO". The Task 7 narrative template has `{PLACEHOLDER}` fields deliberately — those are filled by the engineer from the generated JSON, not code placeholders. That's acceptable for a narrative template and explicitly called out.

**Type consistency:**
- `analyze_per_domain(data, domains)` returns `dict[str, dict]` — matches Task 2 impl and Task 5 orchestrator call site.
- `analyze_stratified(data)` returns `{vqc_correct, vqc_wrong}` — matches Task 3 impl and Task 1 test.
- `top_10_by_gap(data, k)` returns `list[dict]` with `gap` + per-router fields — matches Task 4 impl and Task 1 test.
- `run(input_path, out_json, out_per_domain_pdf, out_stratified_pdf, out_top10_md, domains)` — keyword-only args, matches Task 1 integration test and Task 5 `main()` wiring.

All consistent.

---

## Estimated time

- Task 1 (tests red): 10 min
- Task 2 (per-domain): 10 min
- Task 3 (stratified): 10 min
- Task 4 (top-10): 10 min
- Task 5 (orchestrator + figures): 25 min
- Task 6 (real run): 5 min compute + 5 min inspect
- Task 7 (hand-written patterns + narrative): 30 min human reading + writing
- Task 8 (push + regress): 5 min

**Total: ~1h40 engineering + 30 min human reading.** Fits comfortably in one session.
