"""Build the V4 router dataset: sample prompts per domain from data/v3/<domain>/*.jsonl.

Emits `data/router-v4/train.jsonl` and `data/router-v4/valid.jsonl` with records
`{"prompt": str, "domain": str}`. Domains list comes from `src.routing.router.NICHE_DOMAINS`.

Usage:
    python scripts/build_router_dataset.py --samples-per-domain 2000 --out data/router-v4
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.routing.router import NICHE_DOMAINS  # noqa: E402

logger = logging.getLogger(__name__)


def extract_prompt(record: dict) -> str | None:
    """Pull the user prompt out of either `messages` or `prompt` JSONL rows."""
    if "messages" in record:
        for msg in record["messages"]:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return None
    if "prompt" in record and isinstance(record["prompt"], str):
        return record["prompt"].strip() or None
    return None


def sample_domain(
    domain: str,
    src_train: Path,
    src_valid: Path,
    n_train: int,
    n_valid: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Reservoir-sample prompts from one domain's train/valid splits."""
    rng = random.Random(seed)

    def _load(path: Path, limit: int) -> list[dict]:
        if not path.exists():
            return []
        prompts: list[str] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = extract_prompt(record)
                if prompt:
                    prompts.append(prompt)
        rng.shuffle(prompts)
        prompts = prompts[:limit]
        return [{"prompt": p, "domain": domain} for p in prompts]

    return _load(src_train, n_train), _load(src_valid, n_valid)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build V4 router dataset")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data" / "v3"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "router-v4"))
    parser.add_argument("--samples-per-domain", type=int, default=2000)
    parser.add_argument("--valid-per-domain", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = sorted(NICHE_DOMAINS)
    logger.info("Building router dataset for %d domains", len(domains))

    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    label_map_path = out_dir / "label_map.json"

    n_train_total = 0
    n_valid_total = 0

    with train_path.open("w", encoding="utf-8") as f_train, valid_path.open(
        "w", encoding="utf-8"
    ) as f_valid:
        for i, domain in enumerate(domains):
            dom_dir = data_root / domain
            src_train = dom_dir / "train.jsonl"
            src_valid = dom_dir / "valid.jsonl"
            train_rows, valid_rows = sample_domain(
                domain,
                src_train,
                src_valid,
                args.samples_per_domain,
                args.valid_per_domain,
                seed=args.seed + i,
            )
            for row in train_rows:
                f_train.write(json.dumps(row, ensure_ascii=False) + "\n")
            for row in valid_rows:
                f_valid.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_train_total += len(train_rows)
            n_valid_total += len(valid_rows)
            logger.info(
                "%-14s  train=%5d  valid=%4d", domain, len(train_rows), len(valid_rows)
            )

    label_map = {name: i for i, name in enumerate(domains)}
    label_map_path.write_text(json.dumps(label_map, indent=2) + "\n", encoding="utf-8")

    logger.info("TOTAL train=%d  valid=%d", n_train_total, n_valid_total)
    logger.info("Wrote %s", train_path)
    logger.info("Wrote %s", valid_path)
    logger.info("Wrote %s", label_map_path)


if __name__ == "__main__":
    main()
