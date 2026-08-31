from src.cost_model.architecture import QWEN235, AttentionConfig
from src.cost_model.flash import batch_attention_pairs
from src.cost_model.hardware import H100, HardwareConfig
from src.cost_model.strategies.base import CPStrategy


class UlyssesAttention(CPStrategy):
    """Ulysses attention with all-to-all communication.

    Ulysses splits attention heads across CP ranks and uses all-to-all
    to exchange Q, K, V before local attention, then all-to-all again for output.

    Without an explicitly modeled chunked pipeline, the dependent communication
    and attention phases are added.
    """

    name = "ulysses"

    def __init__(
        self,
        cp: int,
        hw: HardwareConfig = H100,
        attn: AttentionConfig = QWEN235,
        tile_aware: bool = True,
    ):
        super().__init__(cp, hw, attn, tile_aware)
        self.attn.local_heads(cp)  # validate head sharding

    def _compute_time(self, batch: list[int]) -> float:
        """Local attention compute time (heads split across CP)."""
        seqlen = sum(batch)
        local_q_heads, local_kv_heads = self.attn.local_heads(self.cp)
        pairs = batch_attention_pairs(
            batch,
            causal=self.attn.causal,
            tile_aware=self.tile_aware,
            block_q=self.attn.flash_block_q,
            block_kv=self.attn.flash_block_kv,
        )
        return self._flash_time(pairs, seqlen, seqlen, local_q_heads, local_kv_heads)

    def _comm_time(self, total_seq_len: int) -> float:
        """AllToAlls communication for Q, K, V, and O."""
        if self.cp == 1:
            return 0.0
        local_q_heads, local_kv_heads = self.attn.local_heads(self.cp)
        dh = self.attn.head_dim

        # Per-rank QKVO payload. KV heads are replicated when CP exceeds the
        # global KV-head count; treating Hkv / CP as fractional heads is invalid.
        bytes_transferred = (
            total_seq_len
            * dh
            * (2 * local_q_heads + 2 * local_kv_heads)
            * self.attn.dtype_bytes
        )
        return bytes_transferred / self.hw.p2p_bandwidth(self.cp)

    def total_time(self, batch: list[int]) -> float:
        """QKV all-to-all, attention, and output all-to-all are dependent."""
        compute = self._compute_time(batch)
        comm = self._comm_time(sum(batch))
        return compute + comm
