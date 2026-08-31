from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.cost_model import H100, QWEN235, STRATEGIES, AttentionConfig, HardwareConfig
from src.utils import DistributionType, make_sample_fn, read_batches


@dataclass
class CostModelConfig:
    """Configuration for cost model experiments."""

    seq_len: int = 8192
    dp: int = 128
    cp_degrees: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64])
    strategies: list[str] = field(default_factory=lambda: ["ulysses", "ring", "zigzag"])
    distributions: list[DistributionType] = field(
        default_factory=lambda: ["exponential", "normal"]
    )
    hw: HardwareConfig = field(default_factory=lambda: H100)
    attention: AttentionConfig = field(default_factory=lambda: QWEN235)
    tile_aware: bool = True
    seed: int | None = 0


@dataclass
class ExperimentResult:
    """Results from running a single experiment configuration."""

    cp: int
    distribution: str
    strategy: str
    times: list[float]

    @property
    def max_time(self) -> float:
        return max(self.times)


class CostModelRunner:
    def __init__(self, config: CostModelConfig):
        self.config = config
        self._validate()

    def _validate(self) -> None:
        for cp in self.config.cp_degrees:
            if self.config.dp % cp != 0:
                raise ValueError(
                    f"DP ({self.config.dp}) must be divisible by CP ({cp})"
                )
            self.config.attention.local_heads(cp)
        for s in self.config.strategies:
            if s not in STRATEGIES:
                raise ValueError(
                    f"Unknown strategy: {s}. Available: {list(STRATEGIES.keys())}"
                )

    def run_single(
        self,
        cp: int,
        distribution: DistributionType,
        strategy_name: str,
        verbose: bool = True,
        batches: list[list[int]] | None = None,
    ) -> ExperimentResult:
        """Run comparison for a single configuration."""
        if batches is None:
            rng = np.random.default_rng(self.config.seed)
            sample_fn = make_sample_fn(distribution, self.config.seq_len, rng)
            batches = read_batches(
                sample_fn, self.config.seq_len * cp, self.config.dp // cp
            )

        strategy_cls = STRATEGIES[strategy_name]
        strategy = strategy_cls(
            cp,
            self.config.hw,
            self.config.attention,
            tile_aware=self.config.tile_aware,
        )

        total_times = [strategy.total_time(batch) for batch in batches]
        max_time = max(total_times)

        if verbose:
            print(f"  {strategy_name:8s} with {distribution}: max_time={max_time:.2e}s")

        return ExperimentResult(cp, distribution, strategy_name, total_times)

    def run(self, verbose: bool = True) -> list[ExperimentResult]:
        results = []

        for cp in self.config.cp_degrees:
            if verbose:
                print(f"CP={cp}")
            for dist in self.config.distributions:
                dist_seed = None
                if self.config.seed is not None:
                    dist_seed = self.config.seed + self.config.distributions.index(dist)
                rng = np.random.default_rng(dist_seed)
                sample_fn = make_sample_fn(dist, self.config.seq_len, rng)
                batches = read_batches(
                    sample_fn, self.config.seq_len * cp, self.config.dp // cp
                )
                for strategy_name in self.config.strategies:
                    results.append(
                        self.run_single(
                            cp,
                            distribution=dist,
                            strategy_name=strategy_name,
                            verbose=verbose,
                            batches=batches,
                        )
                    )

        return results

    def run_and_plot(
        self, output_dir: Path | str, verbose: bool = True
    ) -> list[ExperimentResult]:
        """Run experiments and generate plots."""
        from src.visual import plot_cost_model_summary, plot_strategy_comparison

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.run(verbose=verbose)

        for dist in self.config.distributions:
            prefix = dist[:3]
            dist_results = [r for r in results if r.distribution == dist]
            summary = {name: [] for name in self.config.strategies}

            for cp in self.config.cp_degrees:
                local_results = [r for r in dist_results if r.cp == cp]
                strategy_times = {}

                for r in local_results:
                    summary[r.strategy].append(r.max_time)
                    strategy_times[r.strategy] = r.times

                plot_strategy_comparison(
                    cp,
                    self.config.dp,
                    strategy_times,
                    output_dir / f"{prefix}_cp{cp}_comparison.png",
                )

            plot_cost_model_summary(
                self.config.cp_degrees, summary, output_dir / f"{prefix}_summary.png"
            )

        return results
