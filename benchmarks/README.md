# FlashAttention benchmark notes

`bench_fa3_varlen.py` reproduces the imbalance experiment using causal and
non-causal FA3 varlen GQA on one GPU. The checked-in results were measured on:

- NVIDIA H100 80GB HBM3
- CUDA 13.0, driver 580.95.05
- PyTorch 2.9.1+cu130
- flash-attn-3 3.0.0b1+2025121700base12170
- BF16, head dimension 128, 64 Q heads, 8 KV heads

The benchmark was run with:

```bash
python bench_fa3_varlen.py \
  --dp 128 --warmup 25 --rep 100 \
  --output results/fa3_h100_results.json
python analyze_fa3_results.py results/fa3_h100_results.json
python plot_h100_validation.py \
  results/fa3_h100_results.json \
  ../figures/imbalance/h100-fa3-validation.png \
  --detail-dir ../figures/imbalance/real
```

The fixed-maximum control used `--fixed-max-seqlen --cp 1 8 64 --dp 64`.
These are sequential samples of hypothetical DP/CP-group workloads on one
H100; they do not include concurrent distributed communication or system-level
straggler noise.

## FA4 on GB300

`bench_fa4_varlen.py` uses `flash_attn.cute.flash_attn_varlen_func` with the
same BF16, head-dimension-128 workloads. Install FA4, PyTorch, Triton, and NumPy
in the GPU environment, then run from the repository root:

```bash
python benchmarks/bench_fa4_varlen.py --dp 128 --warmup 25 --rep 100 \
  --output benchmarks/results/fa4_gb300_results.json
python benchmarks/analyze_fa3_results.py benchmarks/results/fa4_gb300_results.json
```

The analyzer accepts the same JSON schema. Timing uses median eager invocation
latency after compilation and warmup; `--warmup` and `--rep` are milliseconds.
`--block-q` and `--block-kv` only set analytical tile counts (default 128 each),
not FA4 kernel configuration. Verify those assumptions before using a tiled
fit. Results also include useful/tiled FLOPs and estimated Q/K/V/output bytes
for a roofline fit. The existing analyzer fits a linear compute-only model.

## GB300 projection GEMMs

`bench_gemm.py` benchmarks tier7_gen2 BF16 QKV and output projections using
Transformer Engine (`transformer_engine.pytorch.Linear`, no bias). It fixes
hidden size 7168, 64 Q heads, 8 KV heads, and head dimension 128. The default sweep uses
32768 tokens (microbatch 1), TP and CP each in {1, 2, 4, 8, 16, 32, 64}, and three
measurements per shape. Inputs and contiguous `[out_features, in_features]`
weights are allocated outside timing; output allocation is included. Timing
uses warmed eager calls, not CUDA graphs. Each TE module runs a local GEMM
with TP already reflected in its dimensions;
distributed collectives, normalization, FP8, and backward are excluded.

```bash
python benchmarks/bench_gemm.py --output benchmarks/results/gemm_te_gb300_results.json
python benchmarks/analyze_gemm_results.py benchmarks/results/gemm_te_gb300_results.json
```

Both projections process `32768 / CP` rows after the SP all-gather. TP above 8
replicates KV heads. All 25 TP/CP pairs can benchmark GEMMs, while pairs with
TP*CP > 64 are marked unsupported for full Ulysses. Global batch size does not
change per-invocation GEMM shapes.

These are ungated projection shapes; the gated tier7_gen2 projection also
includes an output-gate block and requires a separate calibration.

The analyzer reports unconstrained linear fits and nonnegative roofline fits
with fixed effective HBM bandwidth (default 5.6 TB/s), jointly and separately
by projection and TP. Errors are in-sample. Inspect residuals and timing
variation before adopting one shared throughput/overhead pair.

## Measured GEMM lookup

`calibrations/gemm_te211_gb300.json` preserves the 50 measured BF16, ungated,
no-bias TE 2.11 projection shapes on GB300 Max-Q, including environment and
three timing rounds per shape. `MeasuredGemmTimings` loads `time_ms` directly
for `AttentionBlock(..., measured_gemms=timings)`; unseen shapes fail rather
than using the fitted throughput. The table covers 32k packs, microbatch 1,
H=7168, and TP/CP each in 1, 2, 4, 8, 16.

## Gated attention calibration

Use `bench_gemm.py --gated-attention` for the larger QGKV projection and
output GEMMs. `bench_gate.py` benchmarks the compiled FP32 sigmoid/multiply
with BF16 inputs/output, for contiguous gates and strided slices from a
QGKV buffer. Compilation and correctness checks precede timing. These layouts
probe the cost-model operation; they do not reproduce every production split
and redistribution path. Both use the existing TP/CP sweep and mean timings.

Raw TE 2.11 GB300 results are in `calibrations/gemm_te211_gated_gb300.json`
and `calibrations/gate_te211_gb300.json`. The gated GEMM table is directly
loadable by `MeasuredGemmTimings`. `analyze_gate_results.py` fits nonnegative
invocation overhead and effective bandwidth separately per gate layout; this
fit is diagnostic, not a hardware HBM bandwidth measurement.

The benchmark defaults now include TP/CP 32 and 64 (49 pairs, 98 GEMMs).
`calibrations/gemm_te211_gated_gb300_extended.json` combines the original 50
measurements with 12 new QGKV/output measurements for (TP, CP) = (1,32),
(1,64), (2,32), (32,1), (32,2), (64,1). This covers all 28 valid pairs at
world size 64, plus the original pairs outside that world-size constraint;
it does not cover the full 49-pair grid. Raw additions are in
`calibrations/extended_gated/`. All use TE 2.11, BF16, no bias, and three
rounds of mean timings with the same 25 ms warmup and 100 ms repetition
window. Ungated GEMMs and measured gate tables still stop at degree 16.
