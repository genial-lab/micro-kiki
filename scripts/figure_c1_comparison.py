#!/usr/bin/env python3
"""Generate c1-comparison.pdf from c1-classical-vs-vqc.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("results/c1-classical-vs-vqc.json"))
    p.add_argument("--output", type=Path, default=Path("docs/paper-a/c1-comparison.pdf"))
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    order = ["stratified", "logreg_pca", "torch_vqc", "mlp", "logreg"]
    labels = {
        "stratified": "Stratified\nrandom",
        "logreg_pca": "LogReg\non PCA-4",
        "torch_vqc": "Torch VQC\n(ours)",
        "mlp": "MLP\n(384->64)",
        "logreg": "LogReg\non raw 384-D",
    }
    means = [data["aggregated"][n]["accuracy_mean"] for n in order]
    stds = [data["aggregated"][n]["accuracy_std"] for n in order]
    params = [data["aggregated"][n]["n_params"] for n in order]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    x = np.arange(len(order))
    colors = ["#bbbbbb", "#ffaa66", "#6699ff", "#66bb77", "#cc5555"]
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[n] for n in order], fontsize=9)
    ax.set_ylabel("Test accuracy (10-class routing)")
    ax.axhline(y=0.1, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.10)")
    ax.set_ylim(0, max(means) * 1.15 + 0.05)
    ax.legend(loc="upper left")
    for bar, mean, pcount in zip(bars, means, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.2f}\n({pcount:,}p)", ha="center", va="bottom", fontsize=8)
    ax.set_title("C1: Classical baselines vs Torch VQC router")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
