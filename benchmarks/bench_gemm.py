"""Benchmark BF16 QKV/output GEMMs across TP and CP degrees."""

import argparse
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
import transformer_engine.pytorch as te
import triton
import triton.testing


def projection_shapes(tokens: int, tp: int, cp: int, gated_attention: bool = False) -> list[dict]:
    if tp <= 0 or 64 % tp or cp <= 0 or tokens <= 0 or tokens % (tp * cp):
        raise ValueError("TP must divide 64 and tokens must be divisible by TP * CP")
    rows = tokens // cp
    q_width = 64 // tp * 128
    kv_width = max(1, 8 // tp) * 128
    return [
        {
            "projection": "qgkv" if gated_attention else "qkv", "tp": tp, "cp": cp,
            "m": rows, "k": 7168, "n": q_width * (2 if gated_attention else 1) + 2 * kv_width,
        },
        {
            "projection": "output", "tp": tp, "cp": cp,
            "m": rows, "k": q_width, "n": 7168,
        },
    ]


@torch.no_grad()
def bench_one(x, linear, warmup: int, rep: int) -> dict:
    def fn():
        return linear(x)

    time_ms = float(
        triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    )
    m, k = x.shape
    n = linear.weight.shape[0]
    flops = 2 * m * k * n
    return {
        "time_ms": time_ms,
        "flops": flops,
        "io_bytes": (m * k + k * n + m * n) * x.element_size(),
        "throughput_tflops": flops / (time_ms * 1e9),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("gemm_gb300_results.json"))
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--tp", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--cp", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--gated-attention", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.seq_len, args.micro_batch_size, args.warmup, args.rep, args.rounds) <= 0:
        raise ValueError("Token counts, timing durations, and rounds must be positive")

    torch.manual_seed(args.seed)
    device_name = torch.cuda.get_device_name(0)
    try:
        te_version = version("transformer-engine")
    except PackageNotFoundError:
        te_version = "unknown"
    results = {
        "environment": {
            "device": device_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "triton": triton.__version__,
            "transformer_engine": te_version,
        },
        "config": vars(args) | {
            "output": str(args.output),
            "hidden_size": 7168,
            "q_heads": 64,
            "kv_heads": 8,
            "head_dim": 128,
            "dtype": "bfloat16",
            "backend": "transformer_engine.pytorch.Linear, BF16, no bias",
        },
        "results": [],
    }

    total_tokens = args.seq_len * args.micro_batch_size
    for tp in args.tp:
        for cp in args.cp:
            for shape in projection_shapes(total_tokens, tp, cp, args.gated_attention):
                m, k, n = shape["m"], shape["k"], shape["n"]
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                linear = te.Linear(
                    k, n, bias=False, params_dtype=torch.bfloat16, device="cuda"
                )
                measurements = [
                    bench_one(x, linear, args.warmup, args.rep)
                    for _ in range(args.rounds)
                ]
                times = np.array([entry["time_ms"] for entry in measurements])
                time_ms = float(times.mean())
                entry = shape | measurements[0] | {
                    "samples_ms": times.tolist(),
                    "time_ms": time_ms,
                    "mean_ms": float(times.mean()),
                    "std_ms": float(times.std()),
                    "throughput_tflops": measurements[0]["flops"] / (time_ms * 1e9),
                    "ulysses_supported": tp * cp <= 64,
                }
                results["results"].append(entry)
                print(
                    f"{shape['projection']:6s} tp={tp:2d} cp={cp:2d} "
                    f"{m}x{k}x{n}: mean={times.mean():.4f}ms"
                )
                del x, linear
                torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
