import unittest
from unittest.mock import patch

from src.cost_model.experiment import CostModelConfig, CostModelRunner


class ExperimentTest(unittest.TestCase):
    def test_dp_is_total_devices_not_number_of_cp_groups(self) -> None:
        config = CostModelConfig(
            seq_len=1024,
            dp=8,
            cp_degrees=[4],
            strategies=["ulysses"],
            distributions=["normal"],
        )
        result = CostModelRunner(config).run_single(4, "normal", "ulysses", False)
        self.assertEqual(len(result.times), 2)

    def test_all_strategies_share_the_same_sampled_batches(self) -> None:
        config = CostModelConfig(
            seq_len=1024,
            dp=8,
            cp_degrees=[2],
            strategies=["ulysses", "ring", "zigzag"],
            distributions=["normal"],
        )
        batches = [[256] * 8 for _ in range(4)]
        with patch(
            "src.cost_model.experiment.read_batches", return_value=batches
        ) as read:
            results = CostModelRunner(config).run(verbose=False)
        self.assertEqual(read.call_count, 1)
        self.assertEqual([len(result.times) for result in results], [4, 4, 4])


if __name__ == "__main__":
    unittest.main()
