import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from src.cost_model import AttentionBlock, GB300, MeasuredGemmTimings, UlyssesAttention


CALIBRATION = Path(__file__).resolve().parents[1] / 'benchmarks/calibrations/gemm_te211_gb300.json'


class MeasuredGemmTest(unittest.TestCase):
    def test_all_benchmark_shapes_load_in_seconds(self) -> None:
        timings = MeasuredGemmTimings(CALIBRATION)
        rows = json.loads(CALIBRATION.read_text())['results']
        self.assertEqual(len(rows), 50)
        for row in rows:
            with self.subTest(tp=row['tp'], cp=row['cp'], projection=row['projection']):
                self.assertAlmostEqual(timings.time(row['m'], row['k'], row['n']), row['time_ms'] / 1000, places=15)

    def test_block_uses_measurements_without_roofline_or_added_overhead(self) -> None:
        timings = MeasuredGemmTimings(CALIBRATION)
        hw = replace(GB300, gemm_flops=None, gemm_launch_latency=100, memory_bandwidth=1)
        attention = UlyssesAttention(16, tp=4, hw=hw)
        block = AttentionBlock(attention, 7168, measured_gemms=timings)
        qkv = 0.08895225318195991 / 1000
        output = 0.07998145997042326 / 1000
        self.assertAlmostEqual(block._qkv_compute_time(32768), qkv)
        self.assertAlmostEqual(block._output_compute_time(32768), output)
        expected = qkv + output + attention.total_time([32768]) + block._sp_comm_time(32768)
        self.assertAlmostEqual(block.total_time([32768]), expected)

    def test_missing_shape_does_not_fall_back_to_roofline(self) -> None:
        block = AttentionBlock(UlyssesAttention(1, hw=GB300), 7168,
                               measured_gemms=MeasuredGemmTimings(CALIBRATION))
        with self.assertRaisesRegex(ValueError, 'M=1024, K=7168, N=10240'):
            block.total_time([1024])

    def test_gated_projection_flops_include_tp_local_gate(self) -> None:
        hw = replace(GB300, gemm_flops=1, memory_bandwidth=float('inf'),
                     gemm_launch_latency=0)
        for tp, width in [(1, 18432), (2, 9216), (4, 4608), (8, 2304), (16, 1280)]:
            with self.subTest(tp=tp):
                attention = UlyssesAttention(2, tp=tp, hw=hw)
                ungated = AttentionBlock(attention, 7168)
                gated = AttentionBlock(attention, 7168, gated_attention=True)
                self.assertEqual(gated._qkv_compute_time(32768), 2 * 16384 * 7168 * width)
                self.assertEqual(gated._output_compute_time(32768), ungated._output_compute_time(32768))
                self.assertEqual(gated._sp_comm_time(32768), ungated._sp_comm_time(32768))

    def test_fused_gate_cost_and_forward_inclusion(self) -> None:
        hw = replace(GB300, memory_bandwidth=1000, gate_launch_latency=0.25)
        for tp, cp in [(1, 1), (4, 2), (16, 4)]:
            with self.subTest(tp=tp, cp=cp):
                attention = UlyssesAttention(cp, tp=tp, hw=hw)
                gated = AttentionBlock(attention, 7168, gated_attention=True)
                ungated = AttentionBlock(attention, 7168)
                elements = (32768 // cp) * (64 // tp) * 128
                gate_time = 3 * elements * 2 / 1000 + 0.25
                self.assertEqual(gated._gate_compute_time(32768), gate_time)
                self.assertEqual(ungated._gate_compute_time(32768), 0.0)
                projection_delta = (gated._qkv_compute_time(32768)
                                    - ungated._qkv_compute_time(32768))
                self.assertAlmostEqual(
                    gated.total_time([32768]) - ungated.total_time([32768]),
                    projection_delta + gate_time,
                )

    def test_gate_cost_also_applies_to_measured_gemms(self) -> None:
        timings = MeasuredGemmTimings(CALIBRATION)
        hw = replace(GB300, gate_launch_latency=1e-6)
        attention = UlyssesAttention(1, tp=16, hw=hw)
        block = AttentionBlock(attention, 7168, measured_gemms=timings,
                               gated_attention=True)
        expected = (timings.time(32768, 7168, 1280)
                    + timings.time(32768, 512, 7168)
                    + attention.total_time([32768])
                    + block._sp_comm_time(32768)
                    + 3 * 32768 * 512 * 2 / hw.memory_bandwidth + 1e-6)
        self.assertAlmostEqual(block.total_time([32768]), expected)

    def test_negative_or_nan_gate_latency_rejected(self) -> None:
        for latency in [-1, float('nan')]:
            with self.subTest(latency=latency), self.assertRaises(ValueError):
                replace(GB300, gate_launch_latency=latency)

    def test_gated_projection_uses_exact_shape_measurements(self) -> None:
        timings = MeasuredGemmTimings(CALIBRATION)
        block = AttentionBlock(UlyssesAttention(1, tp=4, hw=GB300), 7168,
                               measured_gemms=timings, gated_attention=True)
        with self.assertRaisesRegex(ValueError, 'M=32768, K=7168, N=4608'):
            block.total_time([32768])
        # TP16 gated projection has the same GEMM shape as TP8 ungated QKV.
        block = AttentionBlock(UlyssesAttention(1, tp=16, hw=GB300), 7168,
                               measured_gemms=timings, gated_attention=True)
        self.assertEqual(block._qkv_compute_time(32768), timings.time(32768, 7168, 1280))

    def test_dtype_must_match(self) -> None:
        timings = MeasuredGemmTimings(CALIBRATION)
        with self.assertRaisesRegex(ValueError, 'dtype=float16'):
            timings.time(32768, 7168, 10240, dtype='float16')
        attention = UlyssesAttention(1, hw=GB300)
        attention.attn = replace(attention.attn, dtype_bytes=4)
        with self.assertRaisesRegex(ValueError, 'dtype_bytes=2'):
            AttentionBlock(attention, 7168, measured_gemms=timings)

    def test_invalid_calibration_rejected(self) -> None:
        row = dict(m=2, k=3, n=4, time_ms=1)
        cases = [[], [row, row], [row | {'time_ms': -1}],
                 [row | {'time_ms': float('nan')}], [row | {'m': 0}]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibration.json'
            for rows in cases:
                with self.subTest(rows=rows):
                    path.write_text(json.dumps(dict(config={'dtype': 'bfloat16'}, environment={}, results=rows)))
                    with self.assertRaises(ValueError):
                        MeasuredGemmTimings(path)


if __name__ == '__main__':
    unittest.main()
