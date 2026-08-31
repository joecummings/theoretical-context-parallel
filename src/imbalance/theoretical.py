from src.imbalance.base import BaseSimulator, SimulationConfig
from src.utils.estimate import approximate_flash_cost


class TheoreticalSimulator(BaseSimulator):
    def __init__(self, config: SimulationConfig):
        super().__init__(config)

    def compute_cost(self, batch: list[int]) -> float:
        pairs = approximate_flash_cost(
            batch,
            causal=self.config.causal,
            tile_aware=self.config.tile_aware,
            block_q=self.config.flash_block_q,
            block_kv=self.config.flash_block_kv,
        )
        q_heads = self.config.local_nheads
        kv_heads = self.config.local_kv_nheads
        tokens = sum(batch)
        compute_time = (
            4 * q_heads * self.config.head_dim * pairs / self.config.compute_flops
        )
        elements = (
            2 * tokens * q_heads * self.config.head_dim
            + 2 * tokens * kv_heads * self.config.head_dim
        )
        memory_time = elements * self.config.dtype_bytes / self.config.memory_bandwidth
        return max(compute_time, memory_time) + self.config.flash_fixed_overhead
