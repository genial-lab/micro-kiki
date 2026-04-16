#!/usr/bin/env python3
"""Aeon compression daemon — periodically compresses old episodes.

Runs as a background process (or systemd service) that calls
``AeonPalace.compress()`` on episodes older than ``--max-age-days``.

Usage::

    uv run python scripts/aeon_compress_daemon.py --interval 3600 --max-age-days 30
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.memory.aeon import AeonPalace

logger = logging.getLogger(__name__)

_running = True


def _handle_signal(signum: int, _frame: object) -> None:
    global _running
    logger.info("Received signal %d, shutting down gracefully", signum)
    _running = False


def run_daemon(
    palace: AeonPalace,
    interval_s: int,
    max_age_days: int,
    summarize_fn: object = None,
) -> int:
    """Run the compression loop. Returns total episodes compressed."""
    total = 0
    while _running:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        count = palace.compress(older_than=cutoff, summarize_fn=summarize_fn)
        total += count
        if count > 0:
            logger.info(
                "Compressed %d episodes (cutoff %s), total %d",
                count, cutoff.isoformat(), total,
            )
        else:
            logger.debug("No episodes to compress (cutoff %s)", cutoff.isoformat())

        for _ in range(interval_s):
            if not _running:
                break
            time.sleep(1)

    logger.info("Daemon stopped. Total compressed: %d", total)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aeon compression daemon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interval", type=int, default=3600,
        help="Seconds between compression runs.",
    )
    parser.add_argument(
        "--max-age-days", type=int, default=30,
        help="Compress episodes older than N days.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run once and exit (no loop).",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    palace = AeonPalace()
    logger.info(
        "Starting Aeon compress daemon: interval=%ds, max_age=%dd",
        args.interval, args.max_age_days,
    )

    if args.once:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.max_age_days)
        count = palace.compress(older_than=cutoff)
        logger.info("One-shot: compressed %d episodes", count)
        return

    run_daemon(palace, args.interval, args.max_age_days)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
