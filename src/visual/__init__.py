import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


# =============================================================================
# Imbalance plots
# =============================================================================


def plot_imbalance_vs_cp(
    cp_degrees: list[int],
    imbalances: list[float],
    distribution: str,
    output_path: Path | str,
) -> None:
    """Plot load imbalance as a function of context parallelism degree."""
    _, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        x=cp_degrees, y=imbalances, marker="o", markersize=10, linewidth=2.5, ax=ax
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context Parallel Size")
    ax.set_ylabel("Load Imbalance (max−min)/mean")
    ax.set_title(
        f"Flash Attention Load Imbalance vs Context Parallelism\n{distribution.title()} Distribution"
    )

    for cp, imb in zip(cp_degrees, imbalances):
        ax.annotate(
            f"{imb:.3f}",
            (cp, imb),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_flash_cost_violin(
    cp_degrees: list[int],
    flash_costs: list[list[float]],
    distribution: str,
    output_path: Path | str,
    cost_label: str = "Flash cost per GPU",
) -> None:
    df = pd.DataFrame(
        [
            {"CP": cp, "Flash Cost": cost}
            for cp, costs in zip(cp_degrees, flash_costs)
            for cost in costs
        ]
    )

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.violinplot(data=df, x="CP", y="Flash Cost", ax=axes[0], inner="quart", cut=0)
    axes[0].set_title("Flash Cost Distribution by Context Parallel Size")
    axes[0].set_xlabel("Context Parallel Size")
    axes[0].set_ylabel(cost_label)

    sns.boxplot(data=df, x="CP", y="Flash Cost", ax=axes[1], width=0.5)
    axes[1].set_title("Flash Cost Spread by Context Parallel Size")
    axes[1].set_xlabel("Context Parallel Size")
    axes[1].set_ylabel(cost_label)

    plt.suptitle(f"{distribution.title()} Distribution")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cost_histograms(
    dp: int,
    cp_degrees: list[int],
    imbalances: list[float],
    flash_costs: list[list[float]],
    distribution: str,
    output_path: Path | str,
    cost_label: str = "Flash cost per GPU",
) -> None:
    """Plot histograms of flash costs for each CP degree."""
    n_plots = len(cp_degrees)
    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    _, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = [axes] if n_plots == 1 else axes.flatten()
    palette = sns.color_palette("deep")

    for idx, cp in enumerate(cp_degrees):
        ax = axes[idx]
        costs = flash_costs[idx]

        sns.histplot(
            costs,
            bins=30,
            kde=True,
            color=palette[idx % len(palette)],
            ax=ax,
            alpha=0.7,
        )
        ax.axvline(
            mean(costs), color="crimson", linestyle="--", linewidth=2, label="mean"
        )
        ax.set_title(f"CP={cp}, DP/CP={dp // cp}\nimbalance={imbalances[idx]:.3f}")
        ax.set_xlabel(cost_label)
        ax.set_ylabel("Count")
        ax.legend()

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"Flash Cost Distribution by Context Parallel Size\n{distribution.title()} Distribution"
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Cost model plots
# =============================================================================


def plot_strategy_comparison(
    cp: int, dp: int, strategy_times: dict[str, list[float]], output_path: Path | str
) -> None:
    """Plot total time across DP ranks for each strategy."""
    _, ax = plt.subplots(figsize=(12, 6))

    colors = {"ulysses": "C0", "ring": "C1", "zigzag": "C2"}

    for name, times in strategy_times.items():
        ax.plot(times, "-", color=colors.get(name, "gray"), label=name, alpha=0.8)

    ax.set_xlabel("DP Rank")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Total Time per DP Rank (CP={cp}, DP={dp})")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cost_model_summary(
    cp_degrees: list[int], results: dict[str, list[float]], output_path: Path | str
) -> None:
    """Plot max time across CP degrees for each strategy.

    Args:
        cp_degrees: List of CP values
        results: {strategy_name: [max_time_per_cp]}
    """
    _, ax = plt.subplots(figsize=(8, 5))

    colors = {"ulysses": "C0", "ring": "C1", "zigzag": "C2"}

    for name, times in results.items():
        ax.plot(
            cp_degrees,
            times,
            "o-",
            color=colors.get(name, "gray"),
            label=name,
            markersize=8,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Context Parallel Size")
    ax.set_ylabel("Max Time (s)")
    ax.set_title("CP Strategy Comparison: Max Time vs CP Degree")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
