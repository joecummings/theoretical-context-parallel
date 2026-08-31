import unittest

from src.cost_model.flash import (
    batch_attention_pairs,
    tiled_attention_pairs,
    useful_attention_pairs,
)


class FlashCostTest(unittest.TestCase):
    def test_useful_causal_pairs(self) -> None:
        self.assertEqual(useful_attention_pairs(128, causal=True), 128 * 129 // 2)

    def test_tiled_causal_pairs_include_masked_slots(self) -> None:
        self.assertEqual(
            tiled_attention_pairs(129, causal=True, block_q=128, block_kv=128),
            128 * 128 + 128 * 256,
        )

    def test_tiled_noncausal_rectangular_pairs(self) -> None:
        self.assertEqual(
            tiled_attention_pairs(129, 257, causal=False, block_q=128, block_kv=128),
            2 * 3 * 128 * 128,
        )

    def test_batch_cost_is_additive(self) -> None:
        expected = sum(tiled_attention_pairs(s, causal=True) for s in [10, 128, 129])
        self.assertEqual(
            batch_attention_pairs([10, 128, 129], causal=True, tile_aware=True),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
