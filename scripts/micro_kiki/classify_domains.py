#!/usr/bin/env python3
"""Classify examples into 1 of 35 Micro_KIKI domains.

Reads raw datasets from data/raw/, classifies each example using keyword
heuristics + regex patterns, and writes per-domain JSONL to data/micro-kiki/classified/.

Usage:
    python scripts/micro_kiki/classify_domains.py [--config configs/micro_kiki/domains.yaml]
                                                   [--input-dir data/raw]
                                                   [--output-dir data/micro-kiki/classified]
                                                   [--max-per-domain 3000]
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


def load_config(config_path: str) -> dict:
    """Load and validate domain configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert len(config["domains"]) == 35, f"Expected 35 domains, got {len(config['domains'])}"
    return config


def compile_patterns(domains: dict[str, dict]) -> dict[str, list[re.Pattern]]:
    """Pre-compile regex patterns for each domain."""
    compiled = {}
    for name, cfg in domains.items():
        compiled[name] = [re.compile(p, re.IGNORECASE) for p in cfg.get("patterns", [])]
    return compiled


def extract_text(example: dict) -> str:
    """Extract all text from an example for classification."""
    messages = example.get("messages", [])
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if content:
            parts.append(content)

    # Also handle non-chat formats
    for key in ("instruction", "input", "output", "question", "answer",
                "solution", "response", "prompt", "completion", "text"):
        val = example.get(key, "")
        if val and isinstance(val, str):
            parts.append(val)

    return "\n".join(parts)


def score_domain(text: str, domain_name: str, domain_cfg: dict,
                 compiled_patterns: list[re.Pattern]) -> float:
    """Score how well a text matches a domain. Higher = better match."""
    if not text.strip():
        return 0.0

    text_lower = text.lower()
    score = 0.0

    # Keyword matches (1 point each, diminishing returns)
    keyword_hits = 0
    for kw in domain_cfg.get("keywords", []):
        if kw.lower() in text_lower:
            keyword_hits += 1
    score += min(keyword_hits * 1.0, 5.0)

    # Regex pattern matches (3 points each — stronger signal)
    for pattern in compiled_patterns:
        matches = pattern.findall(text)
        if matches:
            score += min(len(matches) * 3.0, 12.0)

    return score


def classify_example(example: dict, domains: dict[str, dict],
                     _compiled: dict[str, list[re.Pattern]] | None = None) -> str | None:
    """Classify a single example into its best-matching domain.

    Returns the domain name, or None if no domain scores above threshold.
    """
    text = extract_text(example)
    if len(text.strip()) < 20:
        return None

    compiled = _compiled or compile_patterns(domains)

    best_domain = None
    best_score = 0.0

    for name, cfg in domains.items():
        s = score_domain(text, name, cfg, compiled.get(name, []))
        if s > best_score:
            best_score = s
            best_domain = name

    # Minimum threshold: at least 1 keyword hit
    if best_score < 1.0:
        return None

    return best_domain


def normalize_to_chat(example: dict) -> dict | None:
    """Normalize any example format to chat messages format.

    Returns None if the example cannot be converted.
    """
    # Already in chat format
    if "messages" in example and isinstance(example["messages"], list):
        msgs = example["messages"]
        if len(msgs) >= 2:
            return {"messages": msgs}

    # OpenCodeReasoning / CodeFeedback format
    user = (example.get("question") or example.get("instruction")
            or example.get("input") or example.get("prompt") or "")
    assistant = (example.get("solution") or example.get("output")
                 or example.get("response") or example.get("answer")
                 or example.get("completion") or "")
    reasoning = example.get("reasoning", "")

    if not user.strip() or not assistant.strip():
        return None

    # Normalize thinking tags
    assistant = assistant.replace("<think>", "<thinking>").replace("</think>", "</thinking>")

    # Add reasoning if present but not already wrapped
    if reasoning and "<thinking>" not in assistant:
        assistant = f"<thinking>\n{reasoning.strip()}\n</thinking>\n\n{assistant.strip()}"

    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


def load_all_raw(input_dir: Path) -> list[dict]:
    """Load all examples from all raw dataset directories."""
    examples = []
    if not input_dir.exists():
        print(f"WARNING: {input_dir} does not exist", file=sys.stderr)
        return examples

    for dataset_dir in sorted(input_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        count_before = len(examples)
        for filepath in sorted(dataset_dir.rglob("*")):
            if filepath.suffix == ".jsonl":
                with open(filepath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            examples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            elif filepath.suffix == ".json" and filepath.name not in (
                "dataset_info.json", "dataset_infos.json", "config.json"
            ):
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        examples.extend(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
            elif filepath.suffix == ".parquet" and HAS_PARQUET:
                try:
                    table = pq.read_table(filepath)
                    examples.extend(table.to_pylist())
                except Exception:
                    continue

        loaded = len(examples) - count_before
        if loaded > 0:
            print(f"  {dataset_dir.name}: {loaded} examples")

    return examples


def run_classification(config_path: str, input_dir: str, output_dir: str,
                       max_per_domain: int = 3000) -> dict[str, int]:
    """Run the full classification pipeline.

    Returns a dict of domain -> count.
    """
    config = load_config(config_path)
    domains = config["domains"]
    compiled = compile_patterns(domains)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data from {input_dir}...")
    raw_examples = load_all_raw(Path(input_dir))
    print(f"Loaded {len(raw_examples)} total raw examples")

    # Classify with multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    n_workers = min(mp.cpu_count(), 16)
    print(f"Classifying with {n_workers} workers...")

    domain_examples: dict[str, list[dict]] = {name: [] for name in domains}
    unclassified = 0
    unconvertible = 0

    # Process in chunks for better parallelism
    chunk_size = max(1000, len(raw_examples) // (n_workers * 4))
    chunks = [raw_examples[i:i+chunk_size] for i in range(0, len(raw_examples), chunk_size)]

    def process_chunk(chunk):
        results = []
        unc = 0
        uncv = 0
        for raw in chunk:
            chat = normalize_to_chat(raw)
            if chat is None:
                uncv += 1
                continue
            domain = classify_example(chat, domains, compiled)
            if domain is None:
                unc += 1
                continue
            results.append((domain, chat))
        return results, unc, uncv

    # Sequential but optimized (ProcessPoolExecutor can't pickle compiled regex easily)
    # Use threading instead for I/O-bound parquet loading
    processed = 0
    for ci, chunk in enumerate(chunks):
        results, unc, uncv = process_chunk(chunk)
        unclassified += unc
        unconvertible += uncv
        for domain, chat in results:
            if len(domain_examples[domain]) < max_per_domain:
                domain_examples[domain].append(chat)
        processed += len(chunk)
        if processed % 50000 < chunk_size:
            print(f"  {processed}/{len(raw_examples)} classifiés ({processed*100//len(raw_examples)}%)")

    # Write per-domain JSONL
    counts = {}
    for name, examples in domain_examples.items():
        outfile = out_path / f"{name}.jsonl"
        with open(outfile, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        counts[name] = len(examples)

    # Summary
    print(f"\n=== Classification Summary ===")
    print(f"Total raw: {len(raw_examples)}")
    print(f"Unconvertible: {unconvertible}")
    print(f"Unclassified: {unclassified}")
    print(f"Classified: {sum(counts.values())}")
    print()
    for name in sorted(counts, key=lambda n: domains[n]["phase"]):
        target = domains[name]["target"]
        count = counts[name]
        status = "OK" if count >= target * 0.5 else "SPARSE"
        print(f"  Phase {domains[name]['phase']} {name:<16} {count:>5}/{target} [{status}]")

    return counts


def main():
    parser = argparse.ArgumentParser(description="Classify examples into 35 Micro_KIKI domains")
    parser.add_argument("--config", default="configs/micro_kiki/domains.yaml",
                        help="Path to domain configuration YAML")
    parser.add_argument("--input-dir", default="data/raw",
                        help="Directory containing raw datasets")
    parser.add_argument("--output-dir", default="data/micro-kiki/classified",
                        help="Output directory for per-domain JSONL")
    parser.add_argument("--max-per-domain", type=int, default=3000,
                        help="Maximum examples per domain from public data (default: 3000)")
    args = parser.parse_args()

    run_classification(args.config, args.input_dir, args.output_dir, args.max_per_domain)


if __name__ == "__main__":
    main()
