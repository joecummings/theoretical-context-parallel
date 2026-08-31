from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.utils.data import read_batches


@dataclass
class SimulationConfig:
    """Configuration for imbalance simulation."""

    batch_seq_len: int
    max_seq_len: int
    dp: int
    cp: int
    n_steps: int = 100
    num_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 128
    causal: bool = True
    tile_aware: bool = True
    flash_block_q: int = 128
    flash_block_kv: int = 128
    compute_flops: float = 680e12
    memory_bandwidth: float = 2.4e12
    flash_fixed_overhead: float = 95e-6
    dtype_bytes: int = 2

    def __post_init__(self):
        if self.dp % self.cp != 0:
            raise ValueError(f"DP must be divisible by CP: {self.dp} % {self.cp} != 0")
        if self.num_heads % self.cp != 0:
            raise ValueError(
                f"Num heads must be divisible by CP: {self.num_heads} % {self.cp} != 0"
            )
        if self.cp <= self.num_kv_heads and self.num_kv_heads % self.cp != 0:
            raise ValueError(
                f"CP ({self.cp}) must divide KV heads ({self.num_kv_heads})"
            )
        if self.cp > self.num_kv_heads and self.cp % self.num_kv_heads != 0:
            raise ValueError(
                "CP above the KV-head count must replicate KV heads evenly"
            )

    @property
    def dp_no_cp(self) -> int:
        return self.dp // self.cp

    @property
    def effective_batch_seq_len(self) -> int:
        return self.batch_seq_len * self.cp

    @property
    def local_nheads(self) -> int:
        return self.num_heads // self.cp

    @property
    def local_kv_nheads(self) -> int:
        return max(1, self.num_kv_heads // self.cp)


class BaseSimulator(ABC):
    """Base class for Ulysses load imbalance simulation."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    @abstractmethod
    def compute_cost(self, batch: list[int]) -> float:
        """Compute the flash attention cost for a single batch."""
        raise NotImplementedError

    def run(self, sample_fn: Callable[[], int]) -> np.ndarray:
        """Run simulation and return costs array of shape (n_steps, dp_no_cp)."""
        costs = []
        for _ in range(self.config.n_steps):
            dp_batches = read_batches(
                sample_fn, self.config.effective_batch_seq_len, self.config.dp_no_cp
            )
            step_costs = [self.compute_cost(batch) for batch in dp_batches]
            costs.append(step_costs)
        return np.array(costs)
