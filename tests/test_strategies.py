import unittest

from src.cost_model.architecture import QWEN235
from src.cost_model.hardware import HardwareConfig
from src.cost_model.strategies import UlyssesAttention, ZigZagAttention


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

    def test_zigzag_processes_wrapped_kv_blocks(self) -> None:
        # The second ring step's maximum is on a wrapped rank for this packing.
        hw = HardwareConfig(
            compute_flops=4 * 64 * 128,
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
