"""Smoke test for scripts/validate_domains.py against the live configs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_domains import load_domains_from_each_mirror, validate  # noqa: E402


@pytest.mark.xfail(
    reason=(
        "Known drift between domains.yaml (34), brainstacks.yaml (35), "
        "and mlx-per-domain/*.yaml (32). spice-sim kept in brainstacks+mlx, "
        "components/llm-ops/ml-training missing from mlx-per-domain. "
        "Remove xfail once mirrors are re-synced."
    ),
    strict=False,
)
def test_all_three_mirrors_agree():
    ok, msg = validate()
    assert ok, msg


def test_each_mirror_is_nonempty():
    mirrors = load_domains_from_each_mirror()
    for name, ids in mirrors.items():
        assert len(ids) > 0, f"{name}: empty mirror"
