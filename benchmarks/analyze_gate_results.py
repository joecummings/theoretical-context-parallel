"""Fit nonnegative invocation overhead and bandwidth to gate timings."""

import argparse
import json
from pathlib import Path

import numpy as np


def fit(rows):
    gigabytes = np.array([r['io_bytes'] for r in rows]) / 1e9
    seconds = np.array([r['time_ms'] for r in rows]) * 1e-3
    design = np.column_stack([np.ones_like(gigabytes), gigabytes])
    unconstrained = np.linalg.lstsq(design, seconds, rcond=None)[0]
    candidates = [np.array([0, gigabytes @ seconds / (gigabytes @ gigabytes)]),
                  np.array([seconds.mean(), 0])]
    if np.all(unconstrained >= 0):
        candidates.append(unconstrained)
    overhead, slope = min(candidates, key=lambda c: np.mean((design @ c - seconds)**2))
    predicted = overhead + slope * gigabytes
    errors = np.abs(predicted - seconds) / seconds
    return dict(num_shapes=len(rows), fixed_overhead_us=float(overhead * 1e6),
                bandwidth_tb_per_second=float(1e-3 / slope) if slope > 0 else None,
                mape=float(errors.mean()), max_relative_error=float(errors.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = data['results']
    assert all(len(r['samples_ms']) == data['config']['rounds'] for r in rows)
    result = {layout: fit([r for r in rows if r['layout'] == layout])
              for layout in sorted({r['layout'] for r in rows})}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
