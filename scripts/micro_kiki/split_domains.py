#!/usr/bin/env python3
"""Split deduplicated domain data into train/valid sets.

Produces the final output structure:
    data/micro-kiki/<domain>/train.jsonl
    data/micro-kiki/<domain>/valid.jsonl

Usage:
    python scripts/micro_kiki/split_domains.py \
        --config configs/micro_kiki/domains.yaml \
        --input-dir data/micro-kiki/deduped \
        --output-dir data/micro-kiki
"""

import argparse
import json
import random
from pathlib import Path

import yaml


def split_examples(examples: list[dict], valid_ratio: float = 0.1,
                   seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split examples into train and valid sets.

    Args:
        examples: list of chat examples
        valid_ratio: fraction for validation (default 10%)
        seed: random seed for reproducibility

    Returns:
        (train_examples, valid_examples)
    """
    if not examples:
        return [], []

    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    valid_count = max(0, int(len(shuffled) * valid_ratio))

    # Minimum: at least 1 in train
    if valid_count >= len(shuffled):
        valid_count = len(shuffled) - 1
    if valid_count < 0:
        valid_count = 0

    valid = shuffled[:valid_count]
    train = shuffled[valid_count:]

    return train, valid


def load_jsonl(filepath: Path) -> list[dict]:
    """Load examples from JSONL file."""
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


def run_split(config_path: str, input_dir: str, output_dir: str) -> dict[str, dict[str, int]]:
    """Split all domains into train/valid.

    Returns dict of domain -> {"train": count, "valid": count}.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    domains = config["domains"]
    valid_ratio = config.get("valid_ratio", 0.1)
    in_path = Path(input_dir)
    out_base = Path(output_dir)

    results = {}
    total_train = 0
    total_valid = 0

    sorted_domains = sorted(
        domains.keys(),
        key=lambda n: (domains[n]["phase"], n),
    )

    for name in sorted_domains:
        examples = load_jsonl(in_path / f"{name}.jsonl")
        train, valid = split_examples(examples, valid_ratio=valid_ratio, seed=42)

        # Write to domain directory
        domain_dir = out_base / name
        domain_dir.mkdir(parents=True, exist_ok=True)

        train_file = domain_dir / "train.jsonl"
        with open(train_file, "w") as f:
            for ex in train:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        valid_file = domain_dir / "valid.jsonl"
        with open(valid_file, "w") as f:
            for ex in valid:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        results[name] = {"train": len(train), "valid": len(valid)}
        total_train += len(train)
        total_valid += len(valid)

        target = domains[name]["target"]
        pct = len(train) / target * 100 if target > 0 else 0
        status = "OK" if pct >= 50 else "LOW"
        print(f"  Phase {domains[name]['phase']} {name:<16} train={len(train):>5} valid={len(valid):>4} ({pct:.0f}% of target) [{status}]")

    print(f"\nTotal: {total_train} train + {total_valid} valid = {total_train + total_valid}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Split domain data into train/valid for Micro_KIKI")
    parser.add_argument("--config", default="configs/micro_kiki/domains.yaml")
    parser.add_argument("--input-dir", default="data/micro-kiki/deduped")
    parser.add_argument("--output-dir", default="data/micro-kiki")
    args = parser.parse_args()

    run_split(args.config, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
