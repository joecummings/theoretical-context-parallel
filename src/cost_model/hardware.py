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

    def p2p_bandwidth(self, cp: int) -> float:
        """Get p2p bandwidth based on CP degree.

        Assumes NVLink within node (cp <= 8) and IB across nodes.
        """
        return self.nvlink_bandwidth if cp <= 8 else self.ib_bandwidth


H100 = HardwareConfig()
H200 = HardwareConfig(memory_bandwidth=3.4e12)
