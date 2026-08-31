# Estimating Context Parallel

Analytical and empirical models for Ulysses, Ring, and ZigZag context
parallelism. The default model is causal, BF16, GQA-aware, and includes the
128-by-128 tile quantization used by FlashAttention-3 for head dimension 128.

The model estimates one attention forward pass. It is not an end-to-end
training model: projections, backward, collective latency, topology details,
and pipeline bubbles are outside its scope.

## Run

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
python -m src.main imbalance theoretical
python -m src.main cost-model
```

Use `--ideal-flops` to reproduce the theorem's smooth quadratic model and
`--no-plot` when only numerical output is needed.

GPU mode additionally requires Hopper-compatible FlashAttention-3, either as
`flash_attn_3.flash_attn_interface` or the legacy top-level
`flash_attn_interface`. It benchmarks causal varlen GQA using the actual maximum
sequence length in each pack.

## Important assumptions

- `dp` is the total device count; the number of independent DP/CP groups is
  `dp // cp`.
- Input sampling models concat-then-split packing. A document crossing a pack
  boundary becomes a new causal sequence.
- KV heads are replicated when Ulysses CP exceeds the GQA KV-head count.
- Ulysses QKV all-to-all, attention, and output all-to-all are added because
  they are data-dependent. Ring communication overlaps per-step computation.
- The tile-aware model counts scheduled matrix-multiply slots, then applies a
  roofline bound. The checked-in H100 calibration (`680 TFLOP/s` and `95 us`
  fixed cost) comes from `benchmarks/results/fa3_h100_results.json`; rerun
  `benchmarks/analyze_fa3_results.py` when changing FA or hardware.
