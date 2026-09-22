"""Fixed-world TP/CP sweep over shared 32k token packs."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from src.cost_model import AttentionBlock, GB300, MeasuredGemmTimings, STRATEGIES
from src.utils import make_sample_fn, read_batches


DEGREES = [1, 2, 4, 8, 16, 32, 64]


def valid_pairs(world_size, tp_degrees, cp_degrees, strategy):
    for tp in tp_degrees:
        for cp in cp_degrees:
            if world_size % (tp * cp):
                continue
            if strategy == 'ulysses' and 64 % (tp * cp):
                continue
            yield tp, cp


def run(args):
    if min(args.world_size, args.global_packs, args.pack_tokens, args.hidden_size) <= 0:
        raise ValueError('World size, pack count, token count and hidden size must be positive')
    if any(t <= 0 or 64 % t for t in args.tp) or any(c <= 0 for c in args.cp):
        raise ValueError('TP must divide 64 query heads; CP must be positive')
    timings = MeasuredGemmTimings(args.gemm_calibration) if args.gemm_calibration else None
    configs = []
    for name in args.strategies:
        for tp, cp in valid_pairs(args.world_size, args.tp, args.cp, name):
            dp = args.world_size // (tp * cp)
            if args.global_packs % dp:
                raise ValueError(f'Global packs must divide evenly across DP={dp} (TP={tp}, CP={cp})')
            strategy = STRATEGIES[name](cp, tp=tp, hw=GB300)
            strategy._validate_batch([args.pack_tokens])
            if name == 'zigzag' and args.pack_tokens % (2 * cp):
                raise ValueError('Zigzag requires pack tokens divisible by 2 * CP')
            block = AttentionBlock(strategy, args.hidden_size, measured_gemms=timings,
                                   gated_attention=args.gated_attention)
            # Fail before running any sweep if measured projection shapes are absent.
            block._qkv_compute_time(args.pack_tokens)
            block._output_compute_time(args.pack_tokens)
            configs.append((name, tp, cp, dp, block))
    if not configs:
        raise ValueError('No valid TP/CP configurations for this world size')
    packs = {
        'equal_length': read_batches(lambda: max(1, args.pack_tokens // 8), args.pack_tokens, args.global_packs),
        'exponential': read_batches(make_sample_fn('exponential', args.pack_tokens,
                                                  np.random.default_rng(args.seed)),
                                    args.pack_tokens, args.global_packs),
    }
    rows = []
    for dist, batches in packs.items():
        for name, tp, cp, dp, block in configs:
            cache = {}
            times = []
            for batch in batches:
                key = tuple(batch)
                if key not in cache:
                    cache[key] = block.total_time(batch) * 1000
                times.append(cache[key])
            rounds = np.array(times).reshape(-1, dp)
            rows.append(dict(strategy=name, distribution=dist, tp=tp, cp=cp, dp=dp,
                             rounds=len(rounds), synchronized_batch_ms=float(rounds.max(axis=1).sum()),
                             independent_groups_batch_ms=float(rounds.sum(axis=0).max())))
            print(f'{name} {dist} TP={tp} CP={cp} DP={dp}: {rows[-1]["synchronized_batch_ms"]:.3f} ms', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'sweep.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = vars(args) | {'output': str(args.output), 'hardware': asdict(GB300),
                           'gemm_calibration': str(args.gemm_calibration) if args.gemm_calibration else None,
                           'gate_mode': 'analytical bandwidth plus configured invocation overhead'}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (args.output / 'packs.json').write_text(json.dumps(packs))
    return rows


def plot(rows, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    values = [r['synchronized_batch_ms'] for r in rows]
    norm = LogNorm(min(values), max(values) * (1.000001 if min(values) == max(values) else 1))
    cmap = plt.get_cmap('Oranges').copy()
    cmap.set_bad('#e9edf2')
    for dist in ['equal_length', 'exponential']:
        fig, axes = plt.subplots(1, len(args.strategies), figsize=(6*len(args.strategies), 7),
                                 squeeze=False, layout='constrained')
        for ax, name in zip(axes[0], args.strategies):
            matrix = np.full((len(args.cp), len(args.tp)), np.nan)
            for r in rows:
                if r['strategy'] == name and r['distribution'] == dist:
                    matrix[args.cp.index(r['cp']), args.tp.index(r['tp'])] = r['synchronized_batch_ms']
            im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm)
            ax.set_xticks(range(len(args.tp)), args.tp)
            ax.set_yticks(range(len(args.cp)), args.cp)
            ax.set_xlabel('TP'); ax.set_ylabel('CP'); ax.set_title(name.capitalize())
            for i, cp in enumerate(args.cp):
                for j, tp in enumerate(args.tp):
                    value = matrix[i, j]
                    if np.isfinite(value):
                        ax.text(j, i, f'{value:.0f}\nDP={args.world_size//(tp*cp)}', ha='center', va='center',
                                fontsize=8, color='white' if norm(value)>.65 else '#172033')
        mode = 'measured GEMMs' if args.gemm_calibration else 'roofline GEMMs'
        gate = 'gated' if args.gated_attention else 'ungated'
        fig.suptitle(f'GB300 · {gate} forward completion · {mode}\n{dist} · {args.world_size} GPUs · {args.global_packs} × {args.pack_tokens} tokens')
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.75, label='Completion (ms) · shared log scale')
        fig.supxlabel('Sum of slowest DP group per microbatch round. Analytical gate cost. Blank: invalid world/head partition.', fontsize=10)
        fig.savefig(args.output / f'completion_{dist}.png', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tp', type=int, nargs='+', default=DEGREES)
    parser.add_argument('--cp', type=int, nargs='+', default=DEGREES)
    parser.add_argument('--world-size', type=int, default=64)
    parser.add_argument('--global-packs', type=int, default=1536)
    parser.add_argument('--pack-tokens', type=int, default=32768)
    parser.add_argument('--hidden-size', type=int, default=7168)
    parser.add_argument('--strategies', nargs='+', choices=list(STRATEGIES), default=list(STRATEGIES))
    parser.add_argument('--gated-attention', action='store_true')
    parser.add_argument('--gemm-calibration', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('results/tp_cp_sweep'))
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    rows = run(args)
    if not args.no_plot:
        plot(rows, args)


if __name__ == '__main__':
    main()
