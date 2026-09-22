from src.cost_model.architecture import QWEN235, AttentionConfig
from src.cost_model.flash import batch_attention_pairs
from src.cost_model.hardware import H100, HardwareConfig
from src.cost_model.strategies.base import CPStrategy


class UlyssesAttention(CPStrategy):
    """Ulysses attention with all-to-all communication.

    Ulysses splits attention heads across CP ranks and uses all-to-all
    to exchange Q, K, V before local attention, then all-to-all again for output.

    Compose with AttentionBlock to include projections and sequence parallelism.
    All forward phases are added without communication/compute overlap.
    """

    name = "ulysses"

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
        self.local_q_heads, self.local_kv_heads = self.attn.local_heads(tp * cp)

    def _attn_compute_time(self, batch: list[int]) -> float:
        """Local attention compute time (heads split across TP and CP)."""
        seqlen = sum(batch)
        local_q_heads, local_kv_heads = self.local_q_heads, self.local_kv_heads
        pairs = batch_attention_pairs(
            batch,
            causal=self.attn.causal,
            tile_aware=self.tile_aware,
            block_q=self.attn.flash_block_q,
            block_kv=self.attn.flash_block_kv,
        )
        return self._flash_time(pairs, seqlen, seqlen, local_q_heads, local_kv_heads)

    def _attn_comm_time(self, total_seq_len: int) -> float:
        """QKV all-to-all before attention."""
        return self._attn_all_to_all_time(
            total_seq_len, self.local_q_heads + 2 * self.local_kv_heads
        )

    def _attn_output_comm_time(self, total_seq_len: int) -> float:
        """Output all-to-all after attention."""
        return self._attn_all_to_all_time(total_seq_len, self.local_q_heads)

    def _attn_all_to_all_time(self, total_seq_len: int, heads: int) -> float:
        if self.cp == 1:
            return 0.0
        # Local head counts include replicated KV heads.
        bytes_transferred = (
            total_seq_len * self.attn.head_dim * heads * self.attn.dtype_bytes
        )
        # TP ranks are contiguous; a CP group spans the TP x CP placement.
        return bytes_transferred / self.hw.p2p_bandwidth(self.cp * self.tp)

    def _compute_time(self, batch: list[int]) -> float:
        return self._attn_compute_time(batch)

    def _comm_time(self, total_seq_len: int) -> float:
        return (
            self._attn_comm_time(total_seq_len)
            + self._attn_output_comm_time(total_seq_len)
        )

    def total_time(self, batch: list[int]) -> float:
        self._validate_batch(batch)
        compute = self._compute_time(batch)
        comm = self._comm_time(sum(batch))
        return compute + comm
