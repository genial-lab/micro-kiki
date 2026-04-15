from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForgettingResult:
    angle: float
    winrate_base: float
    winrate_adapted: float
    winrate_drop: float
    passed: bool
    should_rollback: bool


class ForgettingChecker:
    """Automated forgetting check after stack training.

    Rollback trigger: angle < threshold AND winrate drop > threshold.
    Both conditions must be true simultaneously.
    """

    def __init__(
        self,
        eval_fn=None,
        evals_dir: Path | str = ".ralph/evals",
        angle_threshold: float = 30.0,
        winrate_drop_threshold: float = 0.03,
    ) -> None:
        self._eval_fn = eval_fn
        self._evals_dir = Path(evals_dir)
        self._evals_dir.mkdir(parents=True, exist_ok=True)
        self._angle_threshold = angle_threshold
        self._winrate_drop_threshold = winrate_drop_threshold

    def evaluate(
        self,
        angle: float,
        winrate_base: float,
        winrate_adapted: float,
    ) -> ForgettingResult:
        winrate_drop = winrate_base - winrate_adapted
        angle_low = angle < self._angle_threshold
        winrate_dropped = winrate_drop > self._winrate_drop_threshold
        should_rollback = angle_low and winrate_dropped

        result = ForgettingResult(
            angle=angle,
            winrate_base=winrate_base,
            winrate_adapted=winrate_adapted,
            winrate_drop=winrate_drop,
            passed=not should_rollback,
            should_rollback=should_rollback,
        )

        if should_rollback:
            logger.warning(
                "FORGETTING DETECTED: angle=%.1f° (<%.1f°), winrate drop=%.3f (>%.3f)",
                angle, self._angle_threshold, winrate_drop, self._winrate_drop_threshold,
            )
        else:
            logger.info("Forgetting check passed: angle=%.1f°, winrate_drop=%.3f", angle, winrate_drop)

        return result

    def save_result(self, stack_id: str, result: ForgettingResult) -> Path:
        path = self._evals_dir / f"{stack_id}.json"
        data = {
            "stack_id": stack_id,
            "timestamp": datetime.now().isoformat(),
            **asdict(result),
        }
        path.write_text(json.dumps(data, indent=2))
        return path
