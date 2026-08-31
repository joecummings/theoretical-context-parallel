from typing import Callable, Literal

import numpy as np

DistributionType = Literal["exponential", "normal"]


def read_sample_from_exponential(
    max_seq_len: int, beta: float, rng: np.random.Generator | None = None
) -> int:
    """Sample sequence length from Exp(1/beta), clamped to [10, max_seq_len]."""
    source = np.random if rng is None else rng
    return int(np.clip(source.exponential(beta), 10, max_seq_len))


def read_sample_from_normal(
    max_seq_len: int,
    mu: float,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample sequence length from Normal(mu, sigma), clamped to [10, max_seq_len]."""
    source = np.random if rng is None else rng
    return int(np.clip(source.normal(mu, sigma), 10, max_seq_len))


def read_batches(
    read_sample_fn: Callable[[], int], batch_seq_len: int, n_batches: int
) -> list[list[int]]:
    """Fill exact-token-count batches, splitting a sample at pack boundaries.

    This models concat-then-split training data. A split fragment is treated as
    a new causal sequence; callers modeling indivisible documents need a
    padding/bin-packing policy instead.
    """
    saved_sample: int | None = None
    batches: list[list[int]] = []

    for _ in range(n_batches):
        batch: list[int] = []
        current_size = 0

        while current_size < batch_seq_len:
            sample = saved_sample if saved_sample is not None else read_sample_fn()
            saved_sample = None

            remaining = batch_seq_len - current_size
            if sample > remaining:
                saved_sample = sample - remaining
                sample = remaining

            current_size += sample
            batch.append(sample)

        batches.append(batch)

    return batches


def make_sample_fn(
    distribution: DistributionType,
    max_seq_len: int,
    rng: np.random.Generator | None = None,
) -> Callable[[], int]:
    mu = max_seq_len // 8

    if distribution == "exponential":
        return lambda: read_sample_from_exponential(max_seq_len, mu, rng)
    else:  # distribution == "normal"
        return lambda: read_sample_from_normal(max_seq_len, mu, mu / 4, rng)
