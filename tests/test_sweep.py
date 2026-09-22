from dataclasses import replace
import unittest

from src.cost_model import AttentionBlock, GB300, RingAttention, UlyssesAttention, ZigZagAttention
from src.cost_model.sweep import DEGREES, valid_pairs


class ExtendedSweepTest(unittest.TestCase):
    def test_world64_has_28_valid_pairs_and_new_endpoints(self):
        for name in ['ulysses', 'ring', 'zigzag']:
            pairs = list(valid_pairs(64, DEGREES, DEGREES, name))
            self.assertEqual(len(pairs), 28)
            for pair in [(1, 32), (1, 64), (2, 32), (32, 1), (32, 2), (64, 1)]:
                self.assertIn(pair, pairs)
            self.assertNotIn((32, 32), pairs)

    def test_head_limit_is_specific_to_ulysses(self):
        self.assertEqual(list(valid_pairs(4096, [64], [64], 'ulysses')), [])
        for name in ['ring', 'zigzag']:
            self.assertEqual(list(valid_pairs(4096, [64], [64], name)), [(64, 64)])

    def test_tp32_and64_gated_shapes_replicate_kv(self):
        hw = replace(GB300, gemm_flops=1, gemm_launch_latency=0, memory_bandwidth=float('inf'))
        for tp, q_width, gated_width in [(32, 256, 768), (64, 128, 512)]:
            block = AttentionBlock(UlyssesAttention(1, tp=tp, hw=hw), 7168, gated_attention=True)
            self.assertEqual(block.tp_kv_heads, 1)
            self.assertEqual(block._qkv_compute_time(32768), 2*32768*7168*gated_width)
            self.assertEqual(block._output_compute_time(32768), 2*32768*q_width*7168)

    def test_cp32_and64_supported_by_each_strategy(self):
        for cls in [UlyssesAttention, RingAttention, ZigZagAttention]:
            for cp in [32, 64]:
                with self.subTest(strategy=cls.name, cp=cp):
                    block = AttentionBlock(cls(cp, hw=GB300), 7168, gated_attention=True)
                    self.assertGreater(block.total_time([128]), 0)
