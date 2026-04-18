"""Smoke test for scripts/validate_domains.py against the live configs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_domains import load_domains_from_each_mirror, validate  # noqa: E402


def test_all_three_mirrors_agree():
    ok, msg = validate()
    assert ok, msg


def test_each_mirror_is_nonempty():
    mirrors = load_domains_from_each_mirror()
    for name, ids in mirrors.items():
        assert len(ids) > 0, f"{name}: empty mirror"
