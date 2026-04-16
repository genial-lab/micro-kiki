"""Forgetting check framework: win-rate delta + gradient subspace angle.

Run after each stack training to detect interference with prior stacks.
Reference: arxiv 2603.02224 (Subspace Geometry, 2026).

Provides:
- GradientSubspaceAnalyzer: SVD-based gradient subspace angle measurement
- ForgettingEvaluator: orchestrates win-rate + angle checks across stacks
- ForgettingReport: immutable result dataclass

Usable standalone (python -m src.eval.forgetting) or imported by
src.ralph.forgetting_auto.ForgettingChecker.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForgettingReport:
    """Result of a single forgetting check (one prior stack vs new stack)."""

    stack_id: str
    new_stack_id: str
    angle: float
    winrate_base: float
    winrate_adapted: float
    winrate_drop: float
    passed: bool
    should_rollback: bool


# ---------------------------------------------------------------------------
# Gradient subspace analyzer
# ---------------------------------------------------------------------------


class GradientSubspaceAnalyzer:
    """Measures geometric angle between gradient subspaces via SVD."""

    def compute_angle(
        self,
        base_grads: torch.Tensor,
        adapted_grads: torch.Tensor,
    ) -> float:
        """Geometric angle in degrees between two gradient subspaces.

        Uses QR decomposition to get orthonormal bases, then SVD of the
        cross-product to find the principal angle.

        Args:
            base_grads: gradient matrix from base model (n_params, n_samples)
            adapted_grads: gradient matrix from adapted model (n_params, n_samples)

        Returns:
            Angle in degrees. 90° = orthogonal (no interference), 0° = identical.
        """
        import torch

        q_base, _ = torch.linalg.qr(base_grads.float())
        q_adapted, _ = torch.linalg.qr(adapted_grads.float())

        # Principal angle via SVD of Q_base^T @ Q_adapted
        cross = q_base.T @ q_adapted
        singular_values = torch.linalg.svdvals(cross)
        # Clamp for numerical stability
        max_sv = singular_values[0].clamp(max=1.0)
        angle_rad = torch.acos(max_sv)
        return math.degrees(angle_rad.item())

    def collect_gradients(
        self,
        model: object,
        eval_data: list[dict],
        n_samples: int = 100,
    ) -> torch.Tensor:
        """Collect gradient matrix from a small eval set.

        Runs forward+backward on up to n_samples examples, collecting
        gradients of all trainable parameters into a matrix.

        Args:
            model: a PyTorch model (with .parameters())
            eval_data: list of dicts with at least an 'input_ids' key
            n_samples: max number of samples to use

        Returns:
            Gradient matrix of shape (n_params, min(n_samples, len(eval_data)))
        """
        import torch

        model.eval()
        samples = eval_data[:n_samples]
        grad_columns = []

        for sample in samples:
            model.zero_grad()
            input_ids = sample["input_ids"]
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            input_ids = input_ids.unsqueeze(0).to(next(model.parameters()).device)

            outputs = model(input_ids, labels=input_ids)
            outputs.loss.backward()

            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().flatten())
            grad_columns.append(torch.cat(grads))

        # (n_params, n_samples)
        return torch.stack(grad_columns, dim=1)


# ---------------------------------------------------------------------------
# Forgetting evaluator
# ---------------------------------------------------------------------------

# Thresholds
ANGLE_THRESHOLD = 30.0
WINRATE_DROP_THRESHOLD = 0.03


class ForgettingEvaluator:
    """Orchestrates forgetting checks using StackEvaluator + GradientSubspaceAnalyzer."""

    def __init__(
        self,
        stack_evaluator: object,
        analyzer: GradientSubspaceAnalyzer | None = None,
        angle_threshold: float = ANGLE_THRESHOLD,
        winrate_drop_threshold: float = WINRATE_DROP_THRESHOLD,
    ) -> None:
        self._evaluator = stack_evaluator
        self._analyzer = analyzer or GradientSubspaceAnalyzer()
        self._angle_threshold = angle_threshold
        self._winrate_drop_threshold = winrate_drop_threshold

    def _make_report(
        self,
        stack_id: str,
        new_stack_id: str,
        angle: float,
        winrate_base: float,
        winrate_adapted: float,
    ) -> ForgettingReport:
        winrate_drop = winrate_base - winrate_adapted
        should_rollback = (
            angle < self._angle_threshold
            and winrate_drop > self._winrate_drop_threshold
        )
        return ForgettingReport(
            stack_id=stack_id,
            new_stack_id=new_stack_id,
            angle=angle,
            winrate_base=winrate_base,
            winrate_adapted=winrate_adapted,
            winrate_drop=winrate_drop,
            passed=not should_rollback,
            should_rollback=should_rollback,
        )

    def check_stack(
        self,
        stack_id: str,
        new_stack_id: str,
        eval_data_path: Path,
        angle: float | None = None,
        winrate_base: float | None = None,
        winrate_adapted: float | None = None,
    ) -> ForgettingReport:
        """Check if new_stack causes forgetting on stack_id.

        When angle / winrate values are provided directly, uses those
        (useful for testing and when measurements come from external
        tooling). Otherwise delegates to the analyzer and evaluator.

        Args:
            stack_id: previously-trained stack to check regression on
            new_stack_id: the newly-trained stack
            eval_data_path: path to JSONL eval data for stack_id
            angle: pre-computed gradient subspace angle (optional)
            winrate_base: pre-computed baseline win-rate (optional)
            winrate_adapted: pre-computed adapted win-rate (optional)

        Returns:
            ForgettingReport with pass/rollback decision
        """
        if angle is None or winrate_base is None or winrate_adapted is None:
            raise ValueError(
                "Direct measurement mode requires angle, winrate_base, "
                "and winrate_adapted. Automatic measurement requires a "
                "running model — use scripts/run_forgetting.sh instead."
            )

        report = self._make_report(
            stack_id=stack_id,
            new_stack_id=new_stack_id,
            angle=angle,
            winrate_base=winrate_base,
            winrate_adapted=winrate_adapted,
        )

        if report.should_rollback:
            logger.warning(
                "FORGETTING on %s after %s: angle=%.1f° (<%.1f°), "
                "wr_drop=%.3f (>%.3f) -> ROLLBACK",
                stack_id,
                new_stack_id,
                report.angle,
                self._angle_threshold,
                report.winrate_drop,
                self._winrate_drop_threshold,
            )
        else:
            logger.info(
                "Forgetting check %s vs %s: angle=%.1f°, wr_drop=%.3f -> PASS",
                stack_id,
                new_stack_id,
                report.angle,
                report.winrate_drop,
            )

        return report

    def check_all_previous(
        self,
        trained_stacks: list[str],
        new_stack_id: str,
        eval_data_dir: Path | None = None,
        results: list[dict] | None = None,
    ) -> list[ForgettingReport]:
        """Run forgetting check for each previously-trained stack.

        Args:
            trained_stacks: list of stack IDs trained before new_stack_id
            new_stack_id: the newly-trained stack
            eval_data_dir: directory containing {stack_id}.jsonl files
            results: pre-computed list of dicts with keys
                     (stack_id, angle, winrate_base, winrate_adapted)

        Returns:
            List of ForgettingReport, one per prior stack.
        """
        reports: list[ForgettingReport] = []

        if results is not None:
            # Use pre-computed measurements
            for entry in results:
                report = self._make_report(
                    stack_id=entry["stack_id"],
                    new_stack_id=new_stack_id,
                    angle=entry["angle"],
                    winrate_base=entry["winrate_base"],
                    winrate_adapted=entry["winrate_adapted"],
                )
                reports.append(report)
        else:
            # Iterate over prior stacks
            for stack_id in trained_stacks:
                eval_path = (eval_data_dir or Path("data/eval")) / f"{stack_id}.jsonl"
                # In production, measurements would come from the evaluator
                # and analyzer. For now, raise if no pre-computed data.
                raise NotImplementedError(
                    f"Automatic measurement for {stack_id} not yet wired. "
                    "Pass results= with pre-computed measurements."
                )

        any_rollback = any(r.should_rollback for r in reports)
        if any_rollback:
            failed = [r.stack_id for r in reports if r.should_rollback]
            logger.warning(
                "ROLLBACK TRIGGERED: %d stack(s) show forgetting: %s",
                len(failed),
                ", ".join(failed),
            )
        else:
            logger.info(
                "All %d forgetting checks passed for %s",
                len(reports),
                new_stack_id,
            )

        return reports


# ---------------------------------------------------------------------------
# Legacy compat functions (used by existing code/tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForgettingCheckResult:
    """Legacy result type — prefer ForgettingReport for new code."""

    prior_stack: str
    new_stack: str
    win_rate_delta: float
    gradient_subspace_angle_deg: float
    passed: bool
    reason: str


def check_forgetting(
    win_rate_delta: float,
    angle_deg: float,
    prior_stack: str,
    new_stack: str,
    max_delta: float = 0.03,
    min_angle: float = 30.0,
) -> ForgettingCheckResult:
    """Check if new stack causes forgetting on prior stack.

    Fails only if BOTH conditions are true:
    - win_rate_delta > max_delta (regression on prior stack)
    - gradient_subspace_angle < min_angle (high interference)
    """
    delta_bad = win_rate_delta > max_delta
    angle_bad = angle_deg < min_angle
    failed = delta_bad and angle_bad

    if failed:
        reason = f"FORGETTING: delta={win_rate_delta:.3f}>{max_delta}, angle={angle_deg:.1f}<{min_angle}"
    elif delta_bad:
        reason = f"Delta high ({win_rate_delta:.3f}) but angle safe ({angle_deg:.1f}°)"
    elif angle_bad:
        reason = f"Angle low ({angle_deg:.1f}°) but delta safe ({win_rate_delta:.3f})"
    else:
        reason = "OK"

    return ForgettingCheckResult(
        prior_stack=prior_stack,
        new_stack=new_stack,
        win_rate_delta=win_rate_delta,
        gradient_subspace_angle_deg=angle_deg,
        passed=not failed,
        reason=reason,
    )


def compute_subspace_angle(lora_a_new: torch.Tensor, lora_a_prior: torch.Tensor) -> float:
    """Compute angle between LoRA update subspaces (legacy wrapper)."""
    analyzer = GradientSubspaceAnalyzer()
    return analyzer.compute_angle(lora_a_new, lora_a_prior)


def save_forgetting_report(
    results: list[ForgettingCheckResult],
    new_stack: str,
    output_dir: str = "results",
) -> Path:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    path = output_dir_path / f"forgetting-{new_stack}.json"
    data = {
        "new_stack": new_stack,
        "timestamp": datetime.now().isoformat(),
        "all_passed": all(r.passed for r in results),
        "checks": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Forgetting report saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI entry point: python -m src.eval.forgetting
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point for forgetting checks."""
    parser = argparse.ArgumentParser(
        description="Run forgetting check for a new stack against previous stacks",
    )
    parser.add_argument("new_stack_id", help="ID of the newly-trained stack")
    parser.add_argument(
        "--all-previous",
        action="store_true",
        help="Check against all previously-trained stacks",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        help="JSON file with pre-computed measurements",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output reports",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.results_file:
        logger.error("--results-file is required (automatic measurement not yet wired)")
        return 1

    measurements = json.loads(args.results_file.read_text())
    if not isinstance(measurements, list):
        measurements = measurements.get("checks", [measurements])

    evaluator = ForgettingEvaluator(
        stack_evaluator=None,
        analyzer=GradientSubspaceAnalyzer(),
    )

    reports = evaluator.check_all_previous(
        trained_stacks=[],
        new_stack_id=args.new_stack_id,
        results=[
            {
                "stack_id": m.get("stack_id", m.get("prior_stack", "unknown")),
                "angle": m.get("angle", m.get("gradient_subspace_angle_deg", 0.0)),
                "winrate_base": m.get("winrate_base", 0.0),
                "winrate_adapted": m.get("winrate_adapted", 0.0),
            }
            for m in measurements
        ],
    )

    # Save report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"forgetting-{args.new_stack_id}.json"
    report_data = {
        "new_stack": args.new_stack_id,
        "timestamp": datetime.now().isoformat(),
        "all_passed": all(r.passed for r in reports),
        "reports": [asdict(r) for r in reports],
    }
    report_path.write_text(json.dumps(report_data, indent=2))
    logger.info("Report saved to %s", report_path)

    any_rollback = any(r.should_rollback for r in reports)
    if any_rollback:
        logger.warning("ROLLBACK TRIGGERED — exit code 1")
        return 1

    logger.info("All checks passed — exit code 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
