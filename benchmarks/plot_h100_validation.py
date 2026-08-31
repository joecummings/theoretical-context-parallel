"""Plot measured FA3 imbalance against the calibrated analytical model."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.imbalance import SimulationConfig, TheoreticalSimulator
from src.utils import make_sample_fn, read_batches
from src.visual import (
    plot_cost_histograms,
    plot_flash_cost_violin,
    plot_imbalance_vs_cp,
)


def predicted(distribution: str, cp: int, dp: int = 128) -> tuple[float, float]:
    dist_index = ["normal", "exponential"].index(distribution)
    rng = np.random.default_rng(dist_index)
    batches = read_batches(make_sample_fn(distribution, 8192, rng), 8192 * cp, dp // cp)
    simulator = TheoreticalSimulator(SimulationConfig(8192, 8192, dp, cp, n_steps=1))
    times = np.asarray([simulator.compute_cost(batch) for batch in batches]) * 1e3
    return float(times.mean()), float((times.max() - times.min()) / times.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--detail-dir",
        type=Path,
        help="Also write per-distribution H100 plots to this directory",
    )
    args = parser.parse_args()
    data = json.loads(args.results.read_text())

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"normal": "C0", "exponential": "C1"}
    for distribution in ["normal", "exponential"]:
        entries = [
            entry
            for entry in data["results"]
            if entry["causal"] and entry["distribution"] == distribution
        ]
        entries.sort(key=lambda entry: entry["cp"])
        cp = np.asarray([entry["cp"] for entry in entries])
        measured_mean = np.asarray([entry["mean_ms"] for entry in entries])
        measured_imbalance = np.asarray([entry["imbalance"] for entry in entries])
        model = [predicted(distribution, int(value)) for value in cp]
        model_mean = np.asarray([value[0] for value in model])
        model_imbalance = np.asarray([value[1] for value in model])

        axes[0].plot(
            cp,
            measured_mean,
            "o-",
            color=colors[distribution],
            label=f"{distribution}, H100",
        )
        axes[0].plot(
            cp,
            model_mean,
            "--",
            color=colors[distribution],
            label=f"{distribution}, model",
        )
        axes[1].plot(
            cp,
            measured_imbalance,
            "o-",
            color=colors[distribution],
            label=f"{distribution}, H100",
        )
        axes[1].plot(
            cp,
            model_imbalance,
            "--",
            color=colors[distribution],
            label=f"{distribution}, model",
        )

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Context-parallel degree")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("FA3 forward time (ms)")
    axes[0].set_title("Mean causal varlen attention time")
    axes[1].set_ylabel("(max - min) / mean")
    axes[1].set_title("Imbalance across DP/CP groups")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")

    if args.detail_dir is not None:
        args.detail_dir.mkdir(parents=True, exist_ok=True)
        for distribution in ["normal", "exponential"]:
            entries = [
                entry
                for entry in data["results"]
                if entry["causal"] and entry["distribution"] == distribution
            ]
            entries.sort(key=lambda entry: entry["cp"])
            cp_degrees = [entry["cp"] for entry in entries]
            imbalances = [entry["imbalance"] for entry in entries]
            costs = [
                [measurement["time_ms"] for measurement in entry["measurements"]]
                for entry in entries
            ]
            prefix = distribution[:3]
            plot_imbalance_vs_cp(
                cp_degrees,
                imbalances,
                distribution,
                args.detail_dir / f"{prefix}_imbalance_vs_cp.png",
            )
            plot_flash_cost_violin(
                cp_degrees,
                costs,
                distribution,
                args.detail_dir / f"{prefix}_flash_cost_violin.png",
                "Measured FA3 forward time (ms)",
            )
            plot_cost_histograms(
                128,
                cp_degrees,
                imbalances,
                costs,
                distribution,
                args.detail_dir / f"{prefix}_cost_histograms.png",
                "Measured FA3 forward time (ms)",
            )


if __name__ == "__main__":
    main()
