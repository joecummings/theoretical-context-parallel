from dataclasses import dataclass


@dataclass
class AttentionConfig:
    num_heads: int = 64
    head_dim: int = 128

    num_kv_heads: int = 8
    """Use GQA."""

    causal: bool = True
    dtype_bytes: int = 2
    flash_block_q: int = 128
    flash_block_kv: int = 128

    def local_heads(self, cp: int) -> tuple[int, int]:
        """Return local Q and KV heads for Ulysses, including KV replication."""
        if cp <= 0 or self.num_heads % cp != 0:
            raise ValueError(f"CP ({cp}) must divide query heads ({self.num_heads})")
        if cp <= self.num_kv_heads:
            if self.num_kv_heads % cp != 0:
                raise ValueError(
                    f"CP ({cp}) must divide KV heads ({self.num_kv_heads})"
                )
            local_kv_heads = self.num_kv_heads // cp
        else:
            if cp % self.num_kv_heads != 0:
                raise ValueError(
                    "CP above the KV-head count must be a multiple of the KV-head "
                    "count so each KV head can be replicated evenly"
                )
            local_kv_heads = 1
        local_q_heads = self.num_heads // cp
        if local_q_heads % local_kv_heads != 0:
            raise ValueError("local KV heads must divide local query heads")
        return local_q_heads, local_kv_heads


QWEN235 = AttentionConfig()
