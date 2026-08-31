from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from src.imbalance import BaseSimulator, SimulationConfig, TheoreticalSimulator
from src.utils import (
    DistributionType,
    ImbalanceMetrics,
    compute_imbalance_metrics,
    make_sample_fn,
)

SimulatorType = Literal["theoretical", "real"]


@dataclass
class ExperimentResult:
    """Results from running a single experiment configuration."""

    cp: int
    distribution: str
    metrics: ImbalanceMetrics
    costs: np.ndarray  # shape: (n_steps, dp_no_cp)

    @property
    def first_step_costs(self) -> list[float]:
        return self.costs[0].tolist()


@dataclass
class ExperimentConfig:
    """Configuration for imbalance experiments."""

    batch_seq_len: int = 8192
    max_seq_len: int = 8192
    dp: int = 128
    n_steps: int = 100
    cp_degrees: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 64])
    distributions: list[DistributionType] = field(
        default_factory=lambda: ["exponential", "normal"]
    )
    tile_aware: bool = True
    seed: int | None = 0


class ImbalanceExperimentRunner:
    """Runs Ulysses imbalance experiments across multiple CP degrees and distributions."""

    def __init__(
        self, config: ExperimentConfig, simulator_type: SimulatorType = "theoretical"
    ):
        self.config = config
        self.simulator_type = simulator_type
        self._validate_cp_degrees()

    def _validate_cp_degrees(self) -> None:
        for cp in self.config.cp_degrees:
            if self.config.dp % cp != 0:
                raise ValueError(
                    f"DP ({self.config.dp}) must be divisible by CP ({cp})"
                )

    def _create_simulator(self, cp: int) -> BaseSimulator:
        sim_config = SimulationConfig(
            batch_seq_len=self.config.batch_seq_len,
            max_seq_len=self.config.max_seq_len,
            dp=self.config.dp,
            cp=cp,
            n_steps=self.config.n_steps,
            tile_aware=self.config.tile_aware,
        )

        if self.simulator_type == "theoretical":
            return TheoreticalSimulator(sim_config)
        else:
            from src.imbalance.real import RealSimulator

            return RealSimulator(sim_config)

    def run(self, verbose: bool = True) -> list[ExperimentResult]:
        """Run all experiments and return results."""
        results: list[ExperimentResult] = []

        for cp in self.config.cp_degrees:
            simulator = self._create_simulator(cp)

            for dist in self.config.distributions:
                dist_seed = None
                if self.config.seed is not None:
                    dist_seed = self.config.seed + self.config.distributions.index(dist)
                rng = np.random.default_rng(dist_seed)
                sample_fn = make_sample_fn(dist, self.config.max_seq_len, rng)
                costs = simulator.run(sample_fn)
                metrics = compute_imbalance_metrics(costs)

                results.append(
                    ExperimentResult(
                        cp=cp, distribution=dist, metrics=metrics, costs=costs
                    )
                )

                if verbose:
                    print(f"CP={cp:3d}, dist={dist:12s}: {metrics}")

        return results

    def run_and_plot(
        self, output_dir: Path | str, verbose: bool = True
    ) -> list[ExperimentResult]:
        """Run experiments and generate all plots."""
        from src.visual import (
            plot_cost_histograms,
            plot_flash_cost_violin,
            plot_imbalance_vs_cp,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.run(verbose=verbose)

        for dist in self.config.distributions:
            dist_results = [r for r in results if r.distribution == dist]
            cp_degrees = [r.cp for r in dist_results]
            imbalances = [r.metrics.imbalance for r in dist_results]
            costs = [r.first_step_costs for r in dist_results]

            prefix = dist[:3]
            cost_label = (
                "Measured FA3 forward time (ms)"
                if self.simulator_type == "real"
                else "Estimated FA3 forward time (s)"
            )

            plot_imbalance_vs_cp(
                cp_degrees,
                imbalances,
                dist,
                output_dir / f"{prefix}_imbalance_vs_cp.png",
            )

            plot_flash_cost_violin(
                cp_degrees,
                costs,
                dist,
                output_dir / f"{prefix}_flash_cost_violin.png",
                cost_label,
            )

            plot_cost_histograms(
                self.config.dp,
                cp_degrees,
                imbalances,
                costs,
                dist,
                output_dir / f"{prefix}_cost_histograms.png",
                cost_label,
            )

        return results
