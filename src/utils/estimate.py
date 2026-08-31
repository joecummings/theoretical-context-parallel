from src.cost_model.flash import batch_attention_pairs


def approximate_flash_cost(
    batch: list[int],
    *,
    causal: bool = True,
    tile_aware: bool = True,
    block_q: int = 128,
    block_kv: int = 128,
) -> float:
    """Approximate FlashAttention work in useful or scheduled Q/K pairs."""
    return float(
        batch_attention_pairs(
            batch,
            causal=causal,
            tile_aware=tile_aware,
            block_q=block_q,
            block_kv=block_kv,
        )
    )
