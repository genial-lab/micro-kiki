"""Smoke test for scripts/eval_aeon_predictor.py — must finish < 5 s."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_eval_aeon_predictor_smoke(tmp_path):
    out = tmp_path / "result.json"
    t0 = time.time()
    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "eval_aeon_predictor.py"),
            "--dim", "16",
            "--n-turns", "80",
            "--n-queries", "10",
            "--cold-start", "4",
            "--epochs", "20",
            "--out", str(out),
            "--seed", "0",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=60,
    )
    elapsed = time.time() - t0
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    data = json.loads(out.read_text())
    for key in (
        "baseline_recall_at_5",
        "predictive_recall_at_5",
        "null_stack_recall_at_5",
        "baseline_mrr",
        "predictive_mrr",
        "null_stack_mrr",
        "win_rate_predictive",
        "win_rate_stack_vs_null",
    ):
        assert key in data
    assert elapsed < 5.0, f"smoke run too slow: {elapsed:.2f}s"
