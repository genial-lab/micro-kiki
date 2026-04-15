from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def deduplicate_domains(input_dir: Path | str, output_dir: Path | str) -> dict:
    """Deduplicate across domain JSONL files. Each hash assigned to first-seen domain."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: dict[str, str] = {}
    domain_entries: dict[str, list[dict]] = {}
    total_input = 0

    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        domain = jsonl_file.stem
        entries = []
        for line in jsonl_file.read_text().strip().split("\n"):
            if not line:
                continue
            entry = json.loads(line)
            total_input += 1
            h = entry.get("hash", "")
            if h not in seen_hashes:
                seen_hashes[h] = domain
            entries.append(entry)
        domain_entries[domain] = entries

    total_output = 0
    for domain, entries in domain_entries.items():
        kept = [e for e in entries if seen_hashes.get(e.get("hash", "")) == domain]
        (output_dir / f"{domain}.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in kept)
        )
        total_output += len(kept)
        logger.info("Domain %s: %d -> %d", domain, len(entries), len(kept))

    return {"total_input": total_input, "total_output": total_output,
            "duplicates_removed": total_input - total_output, "domains": len(domain_entries)}
