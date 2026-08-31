"""Fit a linear FA3 cost model to benchmark JSON."""

import argparse
import json
from pathlib import Path

import numpy as np


def fit(results: list[dict], pair_key: str, causal: bool) -> dict:
    flops = []
    seconds = []
    for entry in results:
        if entry["causal"] != causal:
            continue
        for measurement in entry["measurements"]:
            flops.append(4 * entry["local_q_heads"] * 128 * measurement[pair_key])
            seconds.append(measurement["time_ms"] * 1e-3)
    x = np.asarray(flops)
    y = np.asarray(seconds)
    design = np.column_stack([np.ones_like(x), x])
    intercept, seconds_per_flop = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ np.array([intercept, seconds_per_flop])
    return {
        "causal": causal,
        "pair_basis": pair_key,
        "fixed_overhead_us": float(intercept * 1e6),
        "throughput_tflops": float(1 / seconds_per_flop / 1e12),
        "r_squared": float(
            1 - np.square(y - predicted).sum() / np.square(y - y.mean()).sum()
        ),
        "mape": float(np.mean(np.abs(predicted - y) / y)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    for causal in [True, False]:
        for pair_key in ["useful_pairs", "tiled_pairs"]:
            print(json.dumps(fit(data["results"], pair_key, causal), indent=2))


if __name__ == "__main__":
    main()
