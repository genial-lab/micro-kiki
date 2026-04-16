#!/usr/bin/env python3
"""Cross-domain deduplication for Micro_KIKI pipeline.

Ensures each example appears in exactly 1 domain. When duplicates exist
across domains, the example stays in the domain where it was classified
first (by priority: phase order, then alphabetical).

Usage:
    python scripts/micro_kiki/deduplicate.py \
        --classified-dir data/micro-kiki/classified \
        --generated-dir data/micro-kiki/generated \
        --output-dir data/micro-kiki/deduped \
        --config configs/micro_kiki/domains.yaml
"""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def dedup_key(example: dict) -> str:
    """Generate a SHA-256 hash key for deduplication.

    Uses the first 500 chars of user content + first 500 chars of assistant content.
    """
    messages = example.get("messages", [])
    user_content = ""
    assistant_content = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
        elif msg.get("role") == "assistant":
            assistant_content = msg.get("content", "")

    text = user_content.strip()[:500] + "\n###\n" + assistant_content.strip()[:500]
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_domain_jsonl(filepath: Path) -> list[dict]:
    """Load examples from a JSONL file."""
    examples = []
    if not filepath.exists():
        return examples
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def dedup_cross_domain(domain_data: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Remove duplicates across domains.

    Each example is kept in the first domain that claims it (by domain priority).
    Priority: lower phase number first, then alphabetical within phase.

    Args:
        domain_data: dict of domain_name -> list of examples

    Returns:
        dict of domain_name -> deduplicated list of examples
    """
    if not domain_data:
        return {}

    seen_hashes: set[str] = set()
    result: dict[str, list[dict]] = {}

    # Process domains in priority order (alphabetical for simplicity if no config)
    for domain_name in sorted(domain_data.keys()):
        deduped = []
        for example in domain_data[domain_name]:
            h = dedup_key(example)
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped.append(example)
        result[domain_name] = deduped

    return result


def run_dedup(config_path: str, classified_dir: str, generated_dir: str,
              output_dir: str) -> dict[str, int]:
    """Run full deduplication pipeline.

    Merges classified + generated data per domain, then deduplicates cross-domain.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    domains = config["domains"]
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load all domain data (classified + generated)
    domain_data: dict[str, list[dict]] = {}

    # Sort domains by phase then name for priority
    sorted_domains = sorted(
        domains.keys(),
        key=lambda n: (domains[n]["phase"], n),
    )

    for name in sorted_domains:
        examples = []
        # Load classified
        classified_file = Path(classified_dir) / f"{name}.jsonl"
        examples.extend(load_domain_jsonl(classified_file))
        # Load generated
        generated_file = Path(generated_dir) / f"{name}.jsonl"
        examples.extend(load_domain_jsonl(generated_file))
        domain_data[name] = examples

    print(f"Before dedup: {sum(len(v) for v in domain_data.values())} total examples")

    # Deduplicate
    deduped = dedup_cross_domain(domain_data)

    # Write output
    counts = {}
    total_removed = 0
    for name in sorted_domains:
        before = len(domain_data.get(name, []))
        after = len(deduped.get(name, []))
        removed = before - after
        total_removed += removed
        counts[name] = after

        outfile = out_path / f"{name}.jsonl"
        with open(outfile, "w") as f:
            for ex in deduped.get(name, []):
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        status = "OK" if after >= domains[name]["target"] * 0.5 else "LOW"
        print(f"  Phase {domains[name]['phase']} {name:<16} {after:>5} (removed {removed}) [{status}]")

    print(f"\nAfter dedup: {sum(counts.values())} total ({total_removed} removed)")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Cross-domain deduplication for Micro_KIKI")
    parser.add_argument("--config", default="configs/micro_kiki/domains.yaml")
    parser.add_argument("--classified-dir", default="data/micro-kiki/classified")
    parser.add_argument("--generated-dir", default="data/micro-kiki/generated")
    parser.add_argument("--output-dir", default="data/micro-kiki/deduped")
    args = parser.parse_args()

    run_dedup(args.config, args.classified_dir, args.generated_dir, args.output_dir)


if __name__ == "__main__":
    main()
