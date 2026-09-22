import unittest

from src.cost_model.block import AttentionBlock
from src.cost_model.architecture import QWEN235, AttentionConfig
from src.cost_model.hardware import HardwareConfig
from src.cost_model.strategies import RingAttention, UlyssesAttention, ZigZagAttention


class StrategyTest(unittest.TestCase):
    def test_gqa_kv_heads_are_replicated_above_cp8(self) -> None:
        self.assertEqual(QWEN235.local_heads(8), (8, 1))
        self.assertEqual(QWEN235.local_heads(16), (4, 1))
        self.assertEqual(QWEN235.local_heads(64), (1, 1))

    def test_ulysses_dependent_phases_are_added(self) -> None:
        strategy = UlyssesAttention(2, tile_aware=False)
        batch = [4096, 4096, 4096, 4096]
        self.assertAlmostEqual(
            strategy.total_time(batch),
            strategy._compute_time(batch) + strategy._comm_time(sum(batch)),
        )

    def test_ulysses_cp1_has_no_collective(self) -> None:
        self.assertEqual(UlyssesAttention(1)._comm_time(8192), 0.0)

    def test_ulysses_tp_sp_projection_shapes_and_collectives(self) -> None:
        hw = HardwareConfig(
            memory_bandwidth=float("inf"), nvlink_bandwidth=100.0, gemm_flops=1.0
        )
        attn = AttentionConfig(num_heads=8, num_kv_heads=2, head_dim=4)
        attention = UlyssesAttention(2, hw, attn, tp=2)
        strategy = AttentionBlock(attention, hidden_size=12)
        self.assertEqual(strategy._qkv_compute_time(16), 4608.0)
        self.assertEqual(strategy._output_compute_time(16), 3072.0)
        self.assertAlmostEqual(strategy._sp_all_gather_time(16), 0.96)
        self.assertAlmostEqual(strategy._sp_reduce_scatter_time(16), 0.96)
        self.assertAlmostEqual(attention._attn_comm_time(16), 5.12)
        self.assertAlmostEqual(attention._attn_output_comm_time(16), 2.56)
        self.assertAlmostEqual(strategy._sp_comm_time(16), 1.92)
        self.assertAlmostEqual(
            strategy.total_time([16]),
            4608.0 + attention._attn_compute_time([16]) + 3072.0 + 9.6,
        )
        self.assertEqual((attention.local_q_heads, attention.local_kv_heads), (2, 1))

    def test_ulysses_tp1_projections_have_no_sp_collectives(self) -> None:
        strategy = AttentionBlock(
            UlyssesAttention(1, hw=HardwareConfig(gemm_flops=1e15)), hidden_size=4096
        )
        self.assertEqual(strategy._sp_all_gather_time(128), 0.0)
        self.assertEqual(strategy._sp_reduce_scatter_time(128), 0.0)
        self.assertEqual(strategy.strategy._attn_comm_time(128), 0.0)
        self.assertEqual(strategy.strategy._attn_output_comm_time(128), 0.0)
        self.assertGreater(strategy._qkv_compute_time(128), 0.0)
        self.assertGreater(strategy._output_compute_time(128), 0.0)

    def test_ulysses_gemm_memory_bound_and_fixed_cost(self) -> None:
        hw = HardwareConfig(
            memory_bandwidth=100.0, gemm_flops=float("inf"), gemm_launch_latency=0.5
        )
        strategy = AttentionBlock(UlyssesAttention(1, hw), hidden_size=12)
        self.assertEqual(strategy._gemm_compute_time(2, 3, 4), 1.02)

    def test_ulysses_tp_placement_affects_cp_bandwidth(self) -> None:
        hw = HardwareConfig(nvlink_bandwidth=100.0, ib_bandwidth=10.0)
        strategy = UlyssesAttention(8, hw, tp=2)
        self.assertEqual((strategy.local_q_heads, strategy.local_kv_heads), (4, 1))
        self.assertEqual(strategy._comm_time(128), 128 * 128 * 10 * 2 / 10)

    def test_ulysses_rejects_invalid_projection_and_partition_config(self) -> None:
        for kwargs in [
            {"tp": 0}, {"tp": 3}, {"tp": 128},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                UlyssesAttention(1, **kwargs)
        for batch in [[], [0], [127], [-128, 256]]:
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                UlyssesAttention(2, tp=2).total_time(batch)

    def test_hardware_rejects_invalid_gemm_calibration(self) -> None:
        for kwargs in [
            {"gemm_flops": 0}, {"gemm_flops": -1},
            {"gemm_flops": float("nan")},
            {"gemm_launch_latency": -1},
            {"gemm_launch_latency": float("nan")},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HardwareConfig(**kwargs)

    def test_gemm_calibration_does_not_enable_projections(self) -> None:
        strategy = UlyssesAttention(1, hw=HardwareConfig(gemm_flops=1e15))
        self.assertEqual(strategy.total_time([128]), UlyssesAttention(1).total_time([128]))

    def test_block_requires_valid_hidden_size_and_calibration(self) -> None:
        for hidden_size, hw in [(4096, HardwareConfig()), (-1, HardwareConfig(gemm_flops=1e15))]:
            with self.subTest(hidden_size=hidden_size), self.assertRaises(ValueError):
                AttentionBlock(UlyssesAttention(1, hw), hidden_size)

    def test_block_composes_all_strategies(self) -> None:
        hw = HardwareConfig(memory_bandwidth=float("inf"), gemm_flops=1.0, nvlink_bandwidth=100.0)
        attn = AttentionConfig(num_heads=8, num_kv_heads=2, head_dim=4)
        for cls in (UlyssesAttention, RingAttention, ZigZagAttention):
            with self.subTest(strategy=cls.name):
                attention = cls(2, hw, attn, tp=2)
                block = AttentionBlock(attention, hidden_size=12)
                self.assertAlmostEqual(
                    block.total_time([16]) - attention.total_time([16]),
                    4608.0 + 3072.0 + 1.92,
                )

    def test_ring_tp_matches_explicit_local_head_configuration(self) -> None:
        hw = HardwareConfig(nvlink_bandwidth=100.0, ib_bandwidth=10.0, nvlink_domain_size=8)
        local_hw = HardwareConfig(nvlink_bandwidth=10.0, ib_bandwidth=10.0)
        local_attn = AttentionConfig(num_heads=4, num_kv_heads=1)
        for cls in (RingAttention, ZigZagAttention):
            with self.subTest(strategy=cls.name):
                actual = cls(2, hw, tp=16).total_time([128, 128])
                expected = cls(2, local_hw, local_attn).total_time([128, 128])
                self.assertAlmostEqual(actual, expected)

    def test_ring_cp_does_not_partition_heads(self) -> None:
        attn = AttentionConfig(num_heads=8, num_kv_heads=2, head_dim=4)
        with self.assertRaises(ValueError):
            UlyssesAttention(4, attn=attn, tp=4)
        for cls in (RingAttention, ZigZagAttention):
            with self.subTest(strategy=cls.name):
                attention = cls(4, attn=attn, tp=4)
                self.assertEqual((attention.tp_q_heads, attention.tp_kv_heads), (2, 1))
                self.assertGreater(attention.total_time([128]), 0.0)

    def test_zigzag_processes_wrapped_kv_blocks(self) -> None:
        # The second ring step's maximum is on a wrapped rank for this packing.
        hw = HardwareConfig(
            flash_flops=4 * 64 * 128,
            memory_bandwidth=float("inf"),
            nvlink_bandwidth=float("inf"),
            ib_bandwidth=float("inf"),
            flash_launch_latency=0.0,
        )
        strategy = ZigZagAttention(2, hw=hw, tile_aware=False)
        self.assertEqual(strategy.total_time([512, 3584]), 4_195_328)

    def test_strategy_rejects_dropped_tail_tokens(self) -> None:
        with self.assertRaises(ValueError):
            ZigZagAttention(4).total_time([8191])


if __name__ == "__main__":
    unittest.main()
