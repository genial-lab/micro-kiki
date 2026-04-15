"""Train meta-router v0 on 3-stack domain mix."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_router_dataset(domain_files: dict[str, str], mixed_count: int = 1000) -> list[dict]:
    """Create multi-label dataset from domain JSONL files.

    Args:
        domain_files: {domain_name: path_to_jsonl}
        mixed_count: number of mixed-intent examples to generate
    """
    examples = []
    domain_names = sorted(domain_files.keys())
    domain_to_idx = {name: i for i, name in enumerate(domain_names)}

    for domain, path in domain_files.items():
        path = Path(path)
        if not path.exists():
            logger.warning("Missing %s, skipping", path)
            continue
        for line in path.read_text().strip().split("\n"):
            if not line:
                continue
            entry = json.loads(line)
            label = [0] * len(domain_names)
            label[domain_to_idx[domain]] = 1
            examples.append({"text": entry["prompt"], "label": label, "domain": domain})

    logger.info("Created %d router training examples from %d domains", len(examples), len(domain_names))
    return examples


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train meta-router v0")
    parser.add_argument("--output", default="outputs/router/v0/router.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    domain_files = {
        "chat-fr": "data/distilled/chat-fr.jsonl",
        "reasoning": "data/distilled/reasoning.jsonl",
        "python": "data/distilled/python.jsonl",
    }

    dataset = create_router_dataset(domain_files)
    logger.info("Dataset: %d examples", len(dataset))

    # Actual training requires torch — deferred to GPU machine
    logger.info("Router training requires torch. Run on kxkm-ai or Mac Studio.")
    logger.info("Output will be saved to %s", args.output)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
