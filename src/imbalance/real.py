from typing import cast

import torch
import triton

from src.imbalance.base import BaseSimulator, SimulationConfig


class RealSimulator(BaseSimulator):
    """Benchmarks flash attention on GPU."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self._q, self._k, self._v = self._make_qkv()

    def _make_qkv(self) -> tuple[torch.Tensor, ...]:
        q_shape = (
            self.config.effective_batch_seq_len,
            self.config.local_nheads,
            self.config.head_dim,
        )
        kv_shape = (
            self.config.effective_batch_seq_len,
            self.config.local_kv_nheads,
            self.config.head_dim,
        )
        return (
            torch.randn(q_shape, dtype=torch.bfloat16, device="cuda"),
            torch.randn(kv_shape, dtype=torch.bfloat16, device="cuda"),
            torch.randn(kv_shape, dtype=torch.bfloat16, device="cuda"),
        )

    def _generate_cu_seqlens(self, batch: list[int]) -> torch.Tensor:
        seqlens = [0] + batch
        return torch.cumsum(torch.tensor(seqlens, device="cuda"), dim=0).to(torch.int32)

    def compute_cost(self, batch: list[int]) -> float:
        try:
            from flash_attn_3.flash_attn_interface import flash_attn_varlen_func
        except ImportError:
            from flash_attn_interface import flash_attn_varlen_func

        cu_seqlens = self._generate_cu_seqlens(batch)
        max_seqlen = max(batch)

        def fn():
            return flash_attn_varlen_func(
                self._q,
                self._k,
                self._v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                causal=self.config.causal,
            )

        return cast(float, triton.testing.do_bench(fn))
