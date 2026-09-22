from typing import cast

import numpy as np

from src.cost_model.architecture import QWEN235, AttentionConfig
from src.cost_model.flash import attention_pairs
from src.cost_model.hardware import H100, HardwareConfig
from src.cost_model.strategies.base import CPStrategy


class RingAttention(CPStrategy):
    """Ring attention with sequential KV block passing.

    Each rank holds a contiguous chunk of tokens. KV blocks are passed
    around the ring while Q blocks stay local. Causal masking means
    ranks only compute attention for KV positions <= Q positions.

    At each step, compute and communication overlap - we wait for whichever
    is slower before proceeding to the next step.
    """

    name = "ring"

    def __init__(
        self,
        cp: int,
        hw: HardwareConfig = H100,
        attn: AttentionConfig = QWEN235,
        tile_aware: bool = True,
        *,
        tp: int = 1,
    ):
        super().__init__(cp, hw, attn, tile_aware, tp=tp)

    def total_time(self, batch: list[int]) -> float:
        self._validate_batch(batch)
        offsets = np.cumsum([0] + batch)
        total_tokens = offsets[-1]
        if total_tokens % self.cp != 0:
            raise ValueError("total tokens must be divisible by CP")
        tokens_per_rank = total_tokens // self.cp

        rank_starts = np.arange(self.cp) * tokens_per_rank
        rank_ends = rank_starts + tokens_per_rank

        s_starts = offsets[:-1]
        s_ends = offsets[1:]

        nkvh = self.tp_kv_heads

        bytes_per_step = (
            tokens_per_rank * nkvh * self.attn.head_dim * 2 * self.attn.dtype_bytes
        )
        comm_time_per_step = bytes_per_step / self.hw.p2p_bandwidth(self.cp * self.tp)

        total_time = 0.0

        for step in range(self.cp):
            max_compute_at_step = 0.0

            for i in range(self.cp):
                q_rank = i
                kv_rank = i - step

                #  skip if kv comes from future positions
                if kv_rank < 0:
                    continue

                q_range = (rank_starts[q_rank], rank_ends[q_rank])
                kv_range = (rank_starts[kv_rank], rank_ends[kv_rank])

                # find overlap of each sample with Q and KV ranges
                q_lens = np.maximum(
                    0, np.minimum(q_range[1], s_ends) - np.maximum(q_range[0], s_starts)
                )
                kv_lens = np.maximum(
                    0,
                    np.minimum(kv_range[1], s_ends) - np.maximum(kv_range[0], s_starts),
                )

                active = (q_lens > 0) & (kv_lens > 0)
                if not np.any(active):
                    continue

                # Diagonal block (same rank): triangular attention
                # Off-diagonal block: rectangular attention
                pairs = sum(
                    attention_pairs(
                        int(q_len),
                        int(kv_len),
                        causal=step == 0,
                        tile_aware=self.tile_aware,
                        block_q=self.attn.flash_block_q,
                        block_kv=self.attn.flash_block_kv,
                    )
                    for q_len, kv_len in zip(q_lens[active], kv_lens[active])
                )
                flash_time = self._flash_time(
                    pairs,
                    int(np.sum(q_lens[active])),
                    int(np.sum(kv_lens[active])),
                    self.tp_q_heads,
                    self.tp_kv_heads,
                )
                max_compute_at_step = max(max_compute_at_step, flash_time)

            # wait for the slower of compute or communication
            comm_time = comm_time_per_step if step < self.cp - 1 else 0.0
            total_time += max(max_compute_at_step, comm_time)

        return cast(float, total_time)
