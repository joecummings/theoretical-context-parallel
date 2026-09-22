"""Benchmark compiled BF16 attention output gating across TP/CP shapes."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
import triton
import triton.testing


def apply_gate(x, gate):
    return (x * torch.sigmoid(gate.float())).to(x.dtype)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('gate_gb300_results.json'))
    parser.add_argument('--seq-len', type=int, default=32768)
    parser.add_argument('--tp', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--cp', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--warmup', type=int, default=25)
    parser.add_argument('--rep', type=int, default=100)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(0)
    results = dict(environment=dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                   cuda=torch.version.cuda, triton=triton.__version__, python=platform.python_version()),
                   config=vars(args) | {'output': str(args.output), 'dtype': 'bfloat16',
                   'backend': 'torch.compile sigmoid FP32 and multiply, BF16 output',
                   'layouts': ['contiguous', 'qgkv_slice']}, results=[])
    for tp in args.tp:
        for cp in args.cp:
            if tp <= 0 or 64 % tp or cp <= 0 or args.seq_len % (tp * cp):
                raise ValueError('Invalid TP/CP partition')
            m, n = args.seq_len // cp, 8192 // tp
            kv = max(1, 8 // tp) * 128
            for layout in ['contiguous', 'qgkv_slice']:
                x = torch.randn(m, n, device='cuda', dtype=torch.bfloat16)
                storage = torch.randn(m, n if layout == 'contiguous' else 2*n+2*kv,
                                      device='cuda', dtype=torch.bfloat16)
                gate = storage if layout == 'contiguous' else storage[:, n:2*n]
                torch._dynamo.reset()
                compiled = torch.compile(apply_gate, fullgraph=True, dynamic=False)
                actual = compiled(x, gate)
                torch.testing.assert_close(actual, apply_gate(x, gate))
                torch.cuda.synchronize()
                times = np.array([triton.testing.do_bench(lambda: compiled(x, gate),
                                  warmup=args.warmup, rep=args.rep) for _ in range(args.rounds)])
                entry = dict(tp=tp, cp=cp, m=m, n=n, layout=layout, gate_stride=list(gate.stride()),
                             io_bytes=3*m*n*2, time_ms=float(times.mean()), samples_ms=times.tolist(),
                             std_ms=float(times.std()))
                results['results'].append(entry)
                print(f'{layout} tp={tp} cp={cp} {m}x{n}: {times.mean():.6f} ms', flush=True)
                del actual, x, gate, storage, compiled
                torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
