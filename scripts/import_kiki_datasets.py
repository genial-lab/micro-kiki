"""Import datasets from KIKI-models-tuning repo for technical domains.

Usage: uv run scripts/import_kiki_datasets.py --domain embedded
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

KIKI_SOURCES = {
    "embedded": "kiki-embedded",
    "stm32": "kiki-stm32",
    "iot": "kiki-iot",
    "freecad": "kiki-freecad",
    "platformio": "kiki-platformio",
    "power": "kiki-power",
    "emc": "kiki-emc",
    "dsp": "kiki-dsp",
    "spice-sim": "kiki-spice-sim",
    "electronics": "kiki-electronics",
    "kicad-pcb": "kiki-kicad-pcb",
}

LOCAL_KIKI_PATH = Path.home() / "Documents/Projets/Factory 4 Life/KIKI-models-tuning"
HF_ORG = "L-electron-Rare"


def find_local_dataset(domain: str) -> Path | None:
    """Look for dataset in local KIKI-models-tuning repo."""
    kiki_name = KIKI_SOURCES.get(domain, f"kiki-{domain}")
    candidates = [
        LOCAL_KIKI_PATH / "data" / f"{kiki_name}.jsonl",
        LOCAL_KIKI_PATH / "data" / kiki_name / "train.jsonl",
        LOCAL_KIKI_PATH / f"{kiki_name}.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def import_domain(domain: str, output_dir: str = "data/raw") -> Path:
    """Import dataset for a domain, trying local then HuggingFace."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{domain}.jsonl"

    # Try local first
    local = find_local_dataset(domain)
    if local:
        logger.info("Found local dataset: %s", local)
        # Copy and normalize format
        entries = []
        for line in local.read_text().strip().split("\n"):
            if not line:
                continue
            entry = json.loads(line)
            normalized = {
                "prompt": entry.get("prompt", entry.get("instruction", entry.get("input", ""))),
                "completion": entry.get("completion", entry.get("output", entry.get("response", ""))),
                "domain": domain,
                "source": "kiki-local",
            }
            if normalized["prompt"] and normalized["completion"]:
                entries.append(normalized)

        out_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries))
        logger.info("Imported %d examples for %s from local", len(entries), domain)
        return out_path

    # Try HuggingFace
    kiki_name = KIKI_SOURCES.get(domain, f"kiki-{domain}")
    hf_repo = f"{HF_ORG}/{kiki_name}"
    logger.info("Local not found. Trying HuggingFace: %s", hf_repo)

    try:
        from datasets import load_dataset
        ds = load_dataset(hf_repo, split="train")
        entries = []
        for row in ds:
            normalized = {
                "prompt": row.get("prompt", row.get("instruction", "")),
                "completion": row.get("completion", row.get("output", "")),
                "domain": domain,
                "source": f"hf:{hf_repo}",
            }
            if normalized["prompt"] and normalized["completion"]:
                entries.append(normalized)

        out_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries))
        logger.info("Imported %d examples for %s from HF", len(entries), domain)
        return out_path

    except Exception as e:
        logger.warning("HuggingFace import failed for %s: %s", domain, e)
        logger.info("Creating empty placeholder. Fill manually or via teacher distillation.")
        out_path.write_text("")
        return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Import KIKI datasets")
    parser.add_argument("--domain", required=True, choices=list(KIKI_SOURCES.keys()))
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    import_domain(args.domain, args.output_dir)
