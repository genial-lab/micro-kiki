"""Energy benchmark framework — theoretical FLOPs vs spikes ratio.

Story-32 implementation. Computes:
- Dense ANN FLOPs: 2 * params * seq_len
- SNN ops: spike_rate * params * timesteps * seq_len
- Energy ratio: snn_ops / dense_flops

CLI usage::

    uv run python scripts/energy_bench.py \
        --model-params 7e9 --spike-rate 0.3 --timesteps 4

Output: ``results/energy-bench.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EnergyBenchResult:
    """Result of a theoretical energy benchmark."""

    model_params: float
    seq_len: int
    spike_rate: float
    timesteps: int
    dense_flops: float
    snn_ops: float
    energy_ratio: float
    snn_energy_saving_pct: float
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def compute_energy(
    model_params: float,
    seq_len: int = 2048,
    spike_rate: float = 0.3,
    timesteps: int = 4,
) -> EnergyBenchResult:
    """Compute theoretical energy comparison between ANN and SNN.

    Parameters
    ----------
    model_params : float
        Total model parameters (e.g. 7e9 for 7B).
    seq_len : int
        Sequence length for the forward pass.
    spike_rate : float
        Average spike rate in [0, 1]. Lower = more sparse = more
        energy efficient. Typical SNN values: 0.1-0.4.
    timesteps : int
        Number of SNN integration timesteps T.

    Returns
    -------
    EnergyBenchResult
        Computed energy metrics.
    """
    if model_params <= 0:
        raise ValueError("model_params must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if not 0.0 < spike_rate <= 1.0:
        raise ValueError("spike_rate must be in (0, 1]")
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")

    # Dense ANN: 2 MACs per parameter per token
    dense_flops = 2.0 * model_params * seq_len

    # SNN: only active (spiking) neurons contribute ops per timestep
    # Each spike triggers an accumulate (1 op) vs multiply-accumulate (2 ops)
    # So SNN ops = spike_rate * params * timesteps * seq_len
    snn_ops = spike_rate * model_params * timesteps * seq_len

    energy_ratio = snn_ops / dense_flops if dense_flops > 0 else 0.0
    saving_pct = (1.0 - energy_ratio) * 100.0

    return EnergyBenchResult(
        model_params=model_params,
        seq_len=seq_len,
        spike_rate=spike_rate,
        timesteps=timesteps,
        dense_flops=dense_flops,
        snn_ops=snn_ops,
        energy_ratio=energy_ratio,
        snn_energy_saving_pct=saving_pct,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Theoretical FLOPs -> spikes energy ratio calculator"
    )
    parser.add_argument(
        "--model-params",
        type=float,
        required=True,
        help="Total model parameters (e.g. 7e9)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)",
    )
    parser.add_argument(
        "--spike-rate",
        type=float,
        default=0.3,
        help="Average spike rate in (0, 1] (default: 0.3)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=4,
        help="SNN integration timesteps T (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/energy-bench.json)",
    )
    args = parser.parse_args(argv)

    result = compute_energy(
        model_params=args.model_params,
        seq_len=args.seq_len,
        spike_rate=args.spike_rate,
        timesteps=args.timesteps,
    )

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        # Find project root (where pyproject.toml lives)
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        out_path = project_root / "results" / "energy-bench.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Also print to stdout
    print(json.dumps(result.to_dict(), indent=2))
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
