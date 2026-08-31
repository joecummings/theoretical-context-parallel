from typing import cast

import numpy as np

from src.cost_model.architecture import QWEN235, AttentionConfig
from src.cost_model.flash import attention_pairs
from src.cost_model.hardware import H100, HardwareConfig
from src.cost_model.strategies.base import CPStrategy


class ZigZagAttention(CPStrategy):
    """ZigZag ring attention for better load balancing.

    Instead of contiguous chunks, each rank gets two blocks:
    one from the start and one from the end of the sequence.

    At each step, compute and communication overlap - we wait for whichever
    is slower before proceeding to the next step.
    """

    name = "zigzag"

    def __init__(
        self,
        cp: int,
        hw: HardwareConfig = H100,
        attn: AttentionConfig = QWEN235,
        tile_aware: bool = True,
    ):
        super().__init__(cp, hw, attn, tile_aware)

    def _get_rank_chunks(self, rank: int, block_size: int) -> list[tuple[int, int]]:
        """Get the two token ranges assigned to a rank."""
        start1 = rank * block_size
        end1 = start1 + block_size
        start2 = (2 * self.cp - 1 - rank) * block_size
        end2 = start2 + block_size
        return [(start1, end1), (start2, end2)]

    def total_time(self, batch: list[int]) -> float:
        offsets = np.cumsum([0] + batch)
        total_tokens = offsets[-1]
        if total_tokens % (2 * self.cp) != 0:
            raise ValueError("total tokens must be divisible by 2 * CP")
        block_size = total_tokens // (2 * self.cp)

        s_starts = offsets[:-1]
        s_ends = offsets[1:]

        nkvh = self.attn.num_kv_heads

        # assumes bf16
        bytes_per_step = (
            2 * block_size * nkvh * self.attn.head_dim * 2 * self.attn.dtype_bytes
        )
        comm_time_per_step = bytes_per_step / self.hw.p2p_bandwidth(self.cp)

        total_time = 0.0

        for step in range(self.cp):
            max_compute_at_step = 0.0

            for i in range(self.cp):
                q_rank = i
                kv_rank = (i - step) % self.cp

                q_chunks = self._get_rank_chunks(q_rank, block_size)
                kv_chunks = self._get_rank_chunks(kv_rank, block_size)

                rank_ops = 0.0
                rank_q_tokens = 0
                rank_kv_tokens = 0

                for q_range in q_chunks:
                    for kv_range in kv_chunks:
                        if kv_range[0] > q_range[0]:
                            continue

                        q_lens = np.maximum(
                            0,
                            np.minimum(q_range[1], s_ends)
                            - np.maximum(q_range[0], s_starts),
                        )
                        kv_lens = np.maximum(
                            0,
                            np.minimum(kv_range[1], s_ends)
                            - np.maximum(kv_range[0], s_starts),
                        )

                        active = (q_lens > 0) & (kv_lens > 0)
                        if not np.any(active):
                            continue

                        rank_ops += sum(
                            attention_pairs(
                                int(q_len),
                                int(kv_len),
                                causal=q_range == kv_range,
                                tile_aware=self.tile_aware,
                                block_q=self.attn.flash_block_q,
                                block_kv=self.attn.flash_block_kv,
                            )
                            for q_len, kv_len in zip(q_lens[active], kv_lens[active])
                        )
                        rank_q_tokens += int(np.sum(q_lens[active]))
                        rank_kv_tokens += int(np.sum(kv_lens[active]))

                flash_time = self._flash_time(
                    rank_ops,
                    rank_q_tokens,
                    rank_kv_tokens,
                    self.attn.num_heads,
                    self.attn.num_kv_heads,
                )
                max_compute_at_step = max(max_compute_at_step, flash_time)

            comm_time = comm_time_per_step if step < self.cp - 1 else 0.0
            total_time += max(max_compute_at_step, comm_time)

        return cast(float, total_time)
