import math


def useful_attention_pairs(
    q_len: int, kv_len: int | None = None, *, causal: bool
) -> int:
    """Number of unmasked Q/K pairs for one attention problem."""
    kv_len = q_len if kv_len is None else kv_len
    if q_len < 0 or kv_len < 0:
        raise ValueError("sequence lengths must be non-negative")
    if causal:
        if q_len != kv_len:
            raise ValueError("this model only supports causal self-attention")
        return q_len * (q_len + 1) // 2
    return q_len * kv_len


def tiled_attention_pairs(
    q_len: int,
    kv_len: int | None = None,
    *,
    causal: bool,
    block_q: int = 128,
    block_kv: int = 128,
) -> int:
    """Approximate tensor-core work after FlashAttention tile quantization.

    Masked elements in a scheduled tile still participate in the QK and PV
    matrix multiplications. This returns the number of tile slots, rather than
    the number of mathematically useful attention pairs.
    """
    kv_len = q_len if kv_len is None else kv_len
    if block_q <= 0 or block_kv <= 0:
        raise ValueError("FlashAttention block sizes must be positive")
    useful_attention_pairs(q_len, kv_len, causal=causal)  # validate inputs
    if q_len == 0 or kv_len == 0:
        return 0
    if not causal:
        return (
            math.ceil(q_len / block_q)
            * math.ceil(kv_len / block_kv)
            * block_q
            * block_kv
        )

    slots = 0
    for q_start in range(0, q_len, block_q):
        causal_k_end = min(kv_len, q_start + block_q)
        slots += block_q * math.ceil(causal_k_end / block_kv) * block_kv
    return slots


def attention_pairs(
    q_len: int,
    kv_len: int | None = None,
    *,
    causal: bool,
    tile_aware: bool,
    block_q: int = 128,
    block_kv: int = 128,
) -> int:
    if tile_aware:
        return tiled_attention_pairs(
            q_len,
            kv_len,
            causal=causal,
            block_q=block_q,
            block_kv=block_kv,
        )
    return useful_attention_pairs(q_len, kv_len, causal=causal)


def batch_attention_pairs(
    batch: list[int],
    *,
    causal: bool,
    tile_aware: bool,
    block_q: int = 128,
    block_kv: int = 128,
) -> int:
    return sum(
        attention_pairs(
            seq_len,
            causal=causal,
            tile_aware=tile_aware,
            block_q=block_q,
            block_kv=block_kv,
        )
        for seq_len in batch
    )
