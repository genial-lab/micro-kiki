from __future__ import annotations

import json
import pytest
from src.distill.dedup import deduplicate_domains


@pytest.fixture
def overlap_data(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shared = [{"prompt": f"shared-{i}", "completion": f"a-{i}", "domain": "x", "hash": f"sh{i}"} for i in range(3)]
    for domain in ["domain-a", "domain-b", "domain-c"]:
        unique = [{"prompt": f"{domain}-{i}", "completion": f"a-{i}", "domain": domain, "hash": f"{domain}-{i}"} for i in range(7)]
        (raw_dir / f"{domain}.jsonl").write_text("\n".join(json.dumps(e) for e in unique + shared))
    return raw_dir


class TestDedup:
    def test_produces_disjoint_output(self, overlap_data, tmp_path):
        output = tmp_path / "dedup"
        deduplicate_domains(overlap_data, output)
        all_hashes: dict[str, list[str]] = {}
        for f in output.glob("*.jsonl"):
            for line in f.read_text().strip().split("\n"):
                if line:
                    h = json.loads(line)["hash"]
                    all_hashes.setdefault(h, []).append(f.stem)
        assert all(len(v) == 1 for v in all_hashes.values())

    def test_preserves_unique_entries(self, overlap_data, tmp_path):
        output = tmp_path / "dedup"
        deduplicate_domains(overlap_data, output)
        total = sum(len([l for l in f.read_text().strip().split("\n") if l]) for f in output.glob("*.jsonl"))
        assert total == 24  # 3*7 unique + 3 shared

    def test_returns_stats(self, overlap_data, tmp_path):
        stats = deduplicate_domains(overlap_data, tmp_path / "dedup")
        assert stats["total_input"] == 30
        assert stats["total_output"] == 24
        assert stats["duplicates_removed"] == 6
