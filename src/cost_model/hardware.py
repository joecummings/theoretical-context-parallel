from dataclasses import dataclass


@dataclass
class HardwareConfig:
    """Hardware performance characteristics."""

    compute_flops: float = 680e12
    """Calibrated FA3 BF16 tile-equivalent throughput on H100."""

    memory_bandwidth: float = 2.4e12
    """HBM bandwidth for H100. 3.35 TB/s are reported, but we use ~0.7 of reported values to better estimate the real-world time."""

    nvlink_bandwidth: float = 400e9
    ib_bandwidth: float = 50e9

    flash_launch_latency: float = 95e-6
    """Calibrated fixed FA3 varlen invocation cost in seconds.

    This includes launch, scheduling, output allocation, and non-matmul work; it
    should not be interpreted as CUDA launch latency alone.
    """

    nvlink_domain_size: int = 8
    """Number of GPUs sharing an NVLink domain."""

    def p2p_bandwidth(self, cp: int) -> float:
        """Get p2p bandwidth based on CP degree.

        Assumes compact placement within the NVLink domain, then IB.
        """
        return self.nvlink_bandwidth if cp <= self.nvlink_domain_size else self.ib_bandwidth


H100 = HardwareConfig()
H200 = HardwareConfig(memory_bandwidth=3.4e12)

# Calibrated FA4 BF16 causal varlen forward on GB300 Max-Q, 128x128 model tiles.
# Fit: benchmarks/results/fa4_gb300_calibration.txt (causal, tiled_pairs).
# GB300 specs: https://www.nvidia.com/en-us/data-center/gb300-nvl72/
GB300 = HardwareConfig(
    compute_flops=1606e12,
    memory_bandwidth=5.6e12,  # 70% of 8 TB/s HBM bandwidth per GPU.
    nvlink_bandwidth=800e9,  # Effective one-way bandwidth (900 GB/s peak).
    ib_bandwidth=100e9,  # 800 Gb/s per GPU.
    flash_launch_latency=58e-6,
    nvlink_domain_size=72,
)
