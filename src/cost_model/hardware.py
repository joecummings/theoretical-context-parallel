from dataclasses import dataclass


@dataclass
class HardwareConfig:
    """Hardware performance characteristics."""

    flash_flops: float = 680e12
    """Calibrated FA3 BF16 tile-equivalent throughput on H100."""

    gemm_flops: float | None = None
    """Effective projection GEMM throughput in FLOP/s; requires calibration."""

    gemm_launch_latency: float = 0.0
    """Fixed GEMM invocation cost in seconds; zero until calibrated."""

    gate_launch_latency: float = 0.0
    """Fixed fused output-gate invocation cost in seconds; uncalibrated by default."""

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

    def __post_init__(self) -> None:
        if self.gemm_flops is not None and not self.gemm_flops > 0:
            raise ValueError("gemm_flops must be positive")
        if not self.gemm_launch_latency >= 0:
            raise ValueError("gemm_launch_latency must be non-negative")

        if not self.gate_launch_latency >= 0:
            raise ValueError("gate_launch_latency must be non-negative")

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
# GEMMs: TE 2.11 BF16 ungated projections, hidden size 7168.
GB300 = HardwareConfig(
    flash_flops=1606e12,
    gemm_flops=1567e12,
    gemm_launch_latency=29e-6,
    memory_bandwidth=5.6e12,  # 70% of 8 TB/s HBM bandwidth per GPU.
    nvlink_bandwidth=800e9,  # Effective one-way bandwidth (900 GB/s peak).
    ib_bandwidth=100e9,  # 800 Gb/s per GPU.
    flash_launch_latency=58e-6,
    nvlink_domain_size=72,
)
