"""Benchmark the exact varlen/GQA shapes used by the imbalance simulation."""

import argparse
import json
import math
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
import triton

try:
    from flash_attn_3.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    from flash_attn_interface import flash_attn_varlen_func


def sample_length(distribution: str, max_seq_len: int, rng: np.random.Generator) -> int:
    mean = max_seq_len // 8
    if distribution == "exponential":
        value = rng.exponential(mean)
    else:
        value = rng.normal(mean, mean / 4)
    return int(np.clip(value, 10, max_seq_len))


def read_batches(
    distribution: str,
    max_seq_len: int,
    tokens_per_batch: int,
    count: int,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    saved = None
    batches = []
    for _ in range(count):
        batch = []
        total = 0
        while total < tokens_per_batch:
            length = (
                saved
                if saved is not None
                else sample_length(distribution, max_seq_len, rng)
            )
            saved = None
            remaining = tokens_per_batch - total
            if length > remaining:
                saved = length - remaining
                length = remaining
            batch.append(length)
            total += length
        batches.append(batch)
    return batches


def tiled_pairs(length: int, causal: bool, block: int = 128) -> int:
    tiles = math.ceil(length / block)
    tile_count = tiles * (tiles + 1) // 2 if causal else tiles * tiles
    return tile_count * block * block


def bench_one(
    q,
    k,
    v,
    batch: list[int],
    causal: bool,
    warmup: int,
    rep: int,
    max_seqlen_override: int | None = None,
) -> dict:
    cu = torch.tensor([0] + batch, device="cuda").cumsum(0).to(torch.int32)
    actual_max_seqlen = max(batch)
    max_seqlen = (
        actual_max_seqlen if max_seqlen_override is None else max_seqlen_override
    )

    def fn():
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu,
            cu,
            max_seqlen,
            max_seqlen,
            causal=causal,
        )

    time_ms = float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))
    useful = sum(s * (s + 1) // 2 if causal else s * s for s in batch)
    tiled = sum(tiled_pairs(s, causal) for s in batch)
    local_q_heads = q.shape[1]
    head_dim = q.shape[2]
    useful_flops = 4 * local_q_heads * head_dim * useful
    tiled_flops = 4 * local_q_heads * head_dim * tiled
    return {
        "time_ms": time_ms,
        "actual_max_seqlen": actual_max_seqlen,
        "reported_max_seqlen": max_seqlen,
        "num_sequences": len(batch),
        "useful_pairs": useful,
        "tiled_pairs": tiled,
        "useful_tflops": useful_flops / (time_ms * 1e9),
        "tiled_tflops": tiled_flops / (time_ms * 1e9),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("fa3_results.json"))
    parser.add_argument("--dp", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--cp", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "--fixed-max-seqlen",
        action="store_true",
        help="Pass the configured cap instead of the actual per-pack maximum",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device_name = torch.cuda.get_device_name(0)
    try:
        flash_version = version("flash-attn-3")
    except PackageNotFoundError:
        flash_version = "unknown"
    results = {
        "environment": {
            "device": device_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "flash_attn_3": flash_version,
        },
        "config": vars(args) | {"output": str(args.output)},
        "results": [],
    }

    for distribution_index, distribution in enumerate(["normal", "exponential"]):
        for cp in args.cp:
            if args.dp % cp or 64 % cp:
                raise ValueError("CP must divide DP and 64 query heads")
            local_q_heads = 64 // cp
            local_kv_heads = max(1, 8 // cp)
            total_tokens = args.max_seq_len * cp
            count = args.dp // cp
            batches = read_batches(
                distribution,
                args.max_seq_len,
                total_tokens,
                count,
                seed=distribution_index,
            )
            q = torch.randn(
                total_tokens, local_q_heads, 128, device="cuda", dtype=torch.bfloat16
            )
            k = torch.randn(
                total_tokens, local_kv_heads, 128, device="cuda", dtype=torch.bfloat16
            )
            v = torch.randn_like(k)
            for causal in [True, False]:
                measurements = [
                    bench_one(
                        q,
                        k,
                        v,
                        batch,
                        causal,
                        args.warmup,
                        args.rep,
                        args.max_seq_len if args.fixed_max_seqlen else None,
                    )
                    for batch in batches
                ]
                times = np.array([m["time_ms"] for m in measurements])
                entry = {
                    "distribution": distribution,
                    "cp": cp,
                    "causal": causal,
                    "local_q_heads": local_q_heads,
                    "local_kv_heads": local_kv_heads,
                    "measurements": measurements,
                    "mean_ms": float(times.mean()),
                    "std_ms": float(times.std()),
                    "imbalance": float((times.max() - times.min()) / times.mean()),
                }
                results["results"].append(entry)
                print(
                    f"{distribution:11s} cp={cp:2d} causal={causal!s:5s} "
                    f"n={count:3d} mean={times.mean():.4f}ms "
                    f"imbalance={entry['imbalance']:.4f}"
                )
            del q, k, v
            torch.cuda.empty_cache()

    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
