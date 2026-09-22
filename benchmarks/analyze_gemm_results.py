"""Fit linear and HBM roofline models to projection GEMM measurements."""

import argparse
import json
from pathlib import Path

import numpy as np


def errors(actual, predicted) -> dict:
    return {
        "mape": float(np.mean(np.abs(predicted - actual) / actual)),
        "max_relative_error": float(np.max(np.abs(predicted - actual) / actual)),
    }


def fit(entries: list[dict], bandwidth: float) -> dict:
    work = np.array([entry["flops"] for entry in entries]) / 1e12
    seconds = np.array([entry["time_ms"] for entry in entries]) * 1e-3
    memory_seconds = np.array([entry["io_bytes"] for entry in entries]) / bandwidth
    design = np.column_stack([np.ones_like(work), work])
    latency, slope = np.linalg.lstsq(design, seconds, rcond=None)[0]
    linear = {
        "fixed_overhead_us": float(latency * 1e6),
        "throughput_tflops": float(1 / slope) if slope > 0 else None,
        "nonnegative_parameters": bool(latency >= 0 and slope > 0),
    } | errors(seconds, design @ [latency, slope])

    # Search a broad range so a bad model is visible rather than capped at GPU peak.
    rates = np.geomspace(100, 10000, 10000)
    base = np.maximum(work[:, None] / rates[None, :], memory_seconds[:, None])
    overhead = np.maximum(0, np.mean(seconds[:, None] - base, axis=0))
    predicted = base + overhead
    best = int(np.argmin(np.mean((predicted - seconds[:, None]) ** 2, axis=0)))
    roofline = {
        "fixed_overhead_us": float(overhead[best] * 1e6),
        "throughput_tflops": float(rates[best]),
        "bandwidth_bytes_per_second": bandwidth,
        "rate_at_search_boundary": best in (0, len(rates) - 1),
    } | errors(seconds, predicted[:, best])
    return {"num_shapes": len(entries), "linear": linear, "roofline": roofline}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--bandwidth", type=float, default=5.6e12)
    args = parser.parse_args()
    if not args.bandwidth > 0:
        parser.error("bandwidth must be positive")
    data = json.loads(args.results.read_text())
    entries = data["results"]
    if any(len(entry["samples_ms"]) != data["config"]["rounds"] for entry in entries):
        parser.error("Benchmark is incomplete")
    groups = {"all": entries}
    for projection in ["qkv", "output"]:
        groups[projection] = [e for e in entries if e["projection"] == projection]
        for tp in sorted({e["tp"] for e in entries}):
            groups[f"{projection}_tp{tp}"] = [e for e in groups[projection] if e["tp"] == tp]
    print(json.dumps({name: fit(group, args.bandwidth) for name, group in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
