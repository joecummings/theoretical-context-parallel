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
