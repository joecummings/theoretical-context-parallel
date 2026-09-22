from abc import ABC, abstractmethod
from src.cost_model.architecture import QWEN235, AttentionConfig
from src.cost_model.hardware import H100, HardwareConfig


class CPStrategy(ABC):
    name: str = "base"

    def __init__(
        self,
        cp: int,
        hw: HardwareConfig = H100,
        attn: AttentionConfig = QWEN235,
        tile_aware: bool = True,
        *,
        tp: int = 1,
    ):
        if cp <= 0:
            raise ValueError("CP must be positive")
        if tp <= 0:
            raise ValueError("TP must be positive")
        self.cp = cp
        self.tp = tp
        self.tp_q_heads, self.tp_kv_heads = attn.local_heads(tp)
        self.hw = hw
        self.attn = attn
        self.tile_aware = tile_aware

    def _validate_batch(self, batch: list[int]) -> None:
        if not batch or any(length <= 0 for length in batch):
            raise ValueError("batch must contain positive sequence lengths")
        if sum(batch) % (self.cp * self.tp):
            raise ValueError("total tokens must be divisible by CP * TP")

    def _flash_time(
        self,
        pairs: float,
        q_tokens: int,
        kv_tokens: int,
        q_heads: int,
        kv_heads: int,
    ) -> float:
        """Roofline estimate for one FlashAttention invocation."""
        dh = self.attn.head_dim
        compute_time = 4 * q_heads * dh * pairs / self.hw.flash_flops
        elements = 2 * q_tokens * q_heads * dh + 2 * kv_tokens * kv_heads * dh
        memory_time = elements * self.attn.dtype_bytes / self.hw.memory_bandwidth
        return max(compute_time, memory_time) + self.hw.flash_launch_latency

    @abstractmethod
    def total_time(self, batch: list[int]) -> float:
        """Total time for this strategy given a batch of sequence lengths."""
        raise NotImplementedError
