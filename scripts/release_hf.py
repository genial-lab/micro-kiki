#!/usr/bin/env python3
"""Push model + adapters to HuggingFace Hub.

Dry-run by default — use --execute to actually push.

Usage:
    uv run python scripts/release_hf.py --repo electron-rare/micro-kiki --adapters outputs/stacks/
    uv run python scripts/release_hf.py --repo electron-rare/micro-kiki --adapters outputs/stacks/ --execute
    uv run python scripts/release_hf.py --help
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReleaseConfig:
    repo: str
    adapters: str
    model_card: str = "MODEL_CARD.md"
    execute: bool = False
    private: bool = False


def collect_files(adapters_dir: Path) -> list[Path]:
    """Collect all files to upload from the adapters directory."""
    if not adapters_dir.exists():
        logger.warning("Adapters directory does not exist: %s", adapters_dir)
        return []

    files: list[Path] = []
    for path in sorted(adapters_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            files.append(path)
    return files


def validate_config(config: ReleaseConfig) -> list[str]:
    """Validate release configuration, return list of issues."""
    issues: list[str] = []

    if "/" not in config.repo:
        issues.append(f"Repo must be in 'org/name' format, got: {config.repo}")

    adapters_path = Path(config.adapters)
    if not adapters_path.exists():
        issues.append(f"Adapters directory not found: {config.adapters}")

    model_card = Path(config.model_card)
    if not model_card.exists():
        issues.append(f"Model card not found: {config.model_card}")

    return issues


def dry_run(config: ReleaseConfig) -> dict:
    """List what would be uploaded without actually pushing."""
    adapters_path = Path(config.adapters)
    files = collect_files(adapters_path)

    model_card = Path(config.model_card)
    has_model_card = model_card.exists()

    total_size = sum(f.stat().st_size for f in files)
    total_size_mb = total_size / (1024 * 1024)

    report = {
        "repo": config.repo,
        "adapters_dir": str(adapters_path),
        "files_to_upload": [str(f) for f in files],
        "file_count": len(files),
        "total_size_mb": round(total_size_mb, 2),
        "model_card": str(model_card) if has_model_card else None,
        "private": config.private,
        "mode": "dry_run",
    }

    logger.info("DRY RUN — would upload to %s:", config.repo)
    logger.info("  Adapters dir: %s", adapters_path)
    logger.info("  Files: %d (%.2f MB)", len(files), total_size_mb)
    for f in files:
        size_kb = f.stat().st_size / 1024
        logger.info("    %s (%.1f KB)", f, size_kb)
    if has_model_card:
        logger.info("  Model card: %s", model_card)
    else:
        logger.warning("  Model card NOT found — will skip upload")

    return report


def execute_release(config: ReleaseConfig) -> dict:
    """Actually push to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: uv add huggingface_hub")
        return {"status": "error", "reason": "huggingface_hub not installed"}

    api = HfApi()
    adapters_path = Path(config.adapters)
    model_card = Path(config.model_card)

    logger.info("Creating/verifying repo: %s", config.repo)
    api.create_repo(config.repo, exist_ok=True, private=config.private)

    logger.info("Uploading adapters from %s", adapters_path)
    api.upload_folder(
        folder_path=str(adapters_path),
        repo_id=config.repo,
        path_in_repo="adapters",
    )

    if model_card.exists():
        logger.info("Uploading model card: %s", model_card)
        api.upload_file(
            path_or_fileobj=str(model_card),
            path_in_repo="README.md",
            repo_id=config.repo,
        )

    logger.info("Release complete: https://huggingface.co/%s", config.repo)
    return {"status": "success", "repo": config.repo, "url": f"https://huggingface.co/{config.repo}"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push model + adapters to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo", required=True,
        help="HuggingFace repo ID (e.g. electron-rare/micro-kiki)",
    )
    parser.add_argument(
        "--adapters", required=True,
        help="Path to adapters directory",
    )
    parser.add_argument(
        "--model-card", default="MODEL_CARD.md",
        help="Path to model card (default: MODEL_CARD.md)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually push to HuggingFace (default: dry-run only)",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create repo as private",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()
    config = ReleaseConfig(
        repo=args.repo,
        adapters=args.adapters,
        model_card=args.model_card,
        execute=args.execute,
        private=args.private,
    )

    issues = validate_config(config)
    if issues:
        for issue in issues:
            logger.warning("Validation: %s", issue)

    if config.execute:
        logger.info("EXECUTE mode — pushing to HuggingFace")
        execute_release(config)
    else:
        dry_run(config)
