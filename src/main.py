"""CLI for context parallelism analysis.

Examples:
    # Ulysses load imbalance analysis (theoretical)
    python -m src.main imbalance theoretical
    python -m src.main imbalance theoretical --cp 1 2 4 8 --dp 64 --steps 50

    # Ulysses load imbalance analysis (real CUDA benchmark)
    python -m src.main imbalance real --cp 4 8 16 --steps 2

    # Cost model comparison (Ring vs ZigZag vs Ulysses)
    python -m src.main cost-model
    python -m src.main cost-model --cp 2 4 8 16 32 --strategies ulysses ring zigzag
"""

import argparse
import sys
from pathlib import Path


def add_imbalance_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "imbalance",
        help="Ulysses load imbalance analysis",
        description="Analyze load imbalance across DP ranks when using **Ulysses** attention",
    )

    parser.add_argument(
        "mode",
        choices=["theoretical", "real"],
        help="Simulation mode: 'theoretical' (analytical) or 'real' (GPU benchmark)",
    )

    parser.add_argument(
        "--cp",
        type=int,
        nargs="+",
        default=None,
        help="Context parallelism degrees (default: 1,2,4,8,16,64)",
    )

    parser.add_argument(
        "--dp", type=int, default=128, help="Data parallelism degree (default: 128)"
    )

    parser.add_argument(
        "--batch-seq-len",
        type=int,
        default=8192,
        help="Batch sequence length (default: 8192)",
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="Maximum sequence length (default: 8192)",
    )

    parser.add_argument(
        "--steps",
        "-n",
        type=int,
        default=None,
        help="Number of simulation steps (default: 100 for theoretical, 2 for real)",
    )

    parser.add_argument(
        "--distributions",
        "-d",
        nargs="+",
        choices=["exponential", "normal"],
        default=["exponential", "normal"],
        help="Distributions to simulate (default: both)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: figures/imbalance/<mode>)",
    )

    parser.add_argument(
        "--ideal-flops",
        action="store_true",
        help="Use mathematically useful pairs instead of FlashAttention tile work",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate plots")


def add_cost_model_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "cost-model",
        help="Compare CP strategies (Ulysses, Ring, ZigZag)",
        description="Compare compute vs communication costs for different context parallelism strategies",
    )

    parser.add_argument(
        "--cp",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64],
        help="Context parallelism degrees (default: 2,4,8,16,32,64)",
    )

    parser.add_argument(
        "--dp", type=int, default=128, help="Data parallelism degree (default: 128)"
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=8192,
        help="Sequence length per rank (default: 8192)",
    )

    parser.add_argument(
        "--strategies",
        "-s",
        nargs="+",
        choices=["ulysses", "ring", "zigzag"],
        default=["ulysses", "ring", "zigzag"],
        help="Strategies to compare (default: all)",
    )

    parser.add_argument(
        "--distributions",
        "-d",
        nargs="+",
        choices=["exponential", "normal"],
        default=["exponential", "normal"],
        help="Distributions to simulate (default: both)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="figures/cost_model",
        help="Output directory for plots (default: figures/cost_model)",
    )

    parser.add_argument(
        "--ideal-flops",
        action="store_true",
        help="Use mathematically useful pairs instead of FlashAttention tile work",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Context parallelism analytical toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    add_imbalance_parser(subparsers)
    add_cost_model_parser(subparsers)

    return parser.parse_args()


def run_imbalance(args) -> int:
    """Run imbalance analysis."""

    if args.cp is None:
        args.cp = [1, 2, 4, 8, 16, 64]

    if args.steps is None:
        args.steps = 2 if args.mode == "real" else 100

    if args.output is None:
        args.output = f"figures/imbalance/{args.mode}"

    for cp in args.cp:
        if args.dp % cp != 0:
            print(
                f"Error: DP ({args.dp}) must be divisible by CP ({cp})", file=sys.stderr
            )
            return 1

    if args.mode == "real":
        try:
            import torch

            if not torch.cuda.is_available():
                print(
                    "Error: CUDA not available. Use 'theoretical' mode.",
                    file=sys.stderr,
                )
                return 1
        except ImportError:
            print(
                "Error: PyTorch not installed. Required for 'real' mode.",
                file=sys.stderr,
            )
            return 1

    from src.imbalance.experiment import ExperimentConfig, ImbalanceExperimentRunner

    config = ExperimentConfig(
        batch_seq_len=args.batch_seq_len,
        max_seq_len=args.max_seq_len,
        dp=args.dp,
        n_steps=args.steps,
        cp_degrees=args.cp,
        distributions=args.distributions,
        tile_aware=not args.ideal_flops,
        seed=args.seed,
    )

    if not args.quiet:
        print(f"Running Ulysses imbalance analysis ({args.mode})")
        print(f"  CP degrees: {args.cp}")
        print(f"  DP: {args.dp}")
        print(f"  Steps: {args.steps}")
        print(f"  Distributions: {args.distributions}")
        print(f"  Output: {args.output}")
        print()

    runner = ImbalanceExperimentRunner(config, simulator_type=args.mode)
    if args.no_plot:
        runner.run(verbose=not args.quiet)
    else:
        runner.run_and_plot(Path(args.output), verbose=not args.quiet)

    if not args.quiet and not args.no_plot:
        print(f"\nPlots saved to {args.output}/")

    return 0


def run_cost_model(args) -> int:
    """Run cost model comparison."""
    for cp in args.cp:
        if args.dp % cp != 0:
            print(
                f"Error: DP ({args.dp}) must be divisible by CP ({cp})", file=sys.stderr
            )
            return 1

    from src.cost_model.experiment import CostModelConfig, CostModelRunner

    config = CostModelConfig(
        seq_len=args.seq_len,
        dp=args.dp,
        cp_degrees=args.cp,
        strategies=args.strategies,
        distributions=args.distributions,
        tile_aware=not args.ideal_flops,
        seed=args.seed,
    )

    if not args.quiet:
        print("Running CP strategy cost model comparison")
        print(f"  CP degrees: {args.cp}")
        print(f"  DP: {args.dp}")
        print(f"  Seq len: {args.seq_len}")
        print(f"  Strategies: {args.strategies}")
        print(f"  Distributions: {args.distributions}")
        print(f"  Output: {args.output}")
        print()

    runner = CostModelRunner(config)
    if args.no_plot:
        runner.run(verbose=not args.quiet)
    else:
        runner.run_and_plot(Path(args.output), verbose=not args.quiet)

    if not args.quiet and not args.no_plot:
        print(f"\nPlots saved to {args.output}/")

    return 0


def main() -> int:
    args = parse_args()

    if args.command == "imbalance":
        return run_imbalance(args)
    elif args.command == "cost-model":
        return run_cost_model(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
