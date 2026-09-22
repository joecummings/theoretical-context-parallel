from src.cost_model.gemm import MeasuredGemmTimings
from src.cost_model.strategies.base import CPStrategy


class AttentionBlock:
    """Forward projections, optional output gating, and TP/SP collectives."""

    def __init__(
        self,
        strategy: CPStrategy,
        hidden_size: int,
        *,
        measured_gemms: MeasuredGemmTimings | None = None,
        gated_attention: bool = False,
    ):
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if measured_gemms is None and strategy.hw.gemm_flops is None:
            raise ValueError("Set hw.gemm_flops to estimate projections")
        if measured_gemms is not None and strategy.attn.dtype_bytes != 2:
            raise ValueError("Measured BF16 GEMMs require dtype_bytes=2")
        self.measured_gemms = measured_gemms
        self.strategy = strategy
        self.hidden_size = hidden_size
        self.gated_attention = gated_attention
        self.cp = strategy.cp
        self.tp = strategy.tp
        self.hw = strategy.hw
        self.attn = strategy.attn
        self.tp_q_heads = strategy.tp_q_heads
        self.tp_kv_heads = strategy.tp_kv_heads

    def _gemm_compute_time(self, m: int, k: int, n: int) -> float:
        if self.measured_gemms is not None:
            return self.measured_gemms.time(m, k, n)
        assert self.hw.gemm_flops is not None
        compute_time = 2 * m * k * n / self.hw.gemm_flops
        io_bytes = (m * k + k * n + m * n) * self.attn.dtype_bytes
        memory_time = io_bytes / self.hw.memory_bandwidth
        return max(compute_time, memory_time) + self.hw.gemm_launch_latency

    def _qkv_compute_time(self, total_tokens: int) -> float:
        qkv_width = (self.tp_q_heads + 2 * self.tp_kv_heads) * self.attn.head_dim
        if self.gated_attention:
            qkv_width += self.tp_q_heads * self.attn.head_dim
        return self._gemm_compute_time(
            total_tokens // self.cp, self.hidden_size, qkv_width
        )

    def _gate_compute_time(self, total_tokens: int) -> float:
        if not self.gated_attention:
            return 0.0
        elements = (
            (total_tokens // self.cp) * self.tp_q_heads * self.attn.head_dim
        )
        # Fused sigmoid/multiply: read gate and attention output, write result.
        io_bytes = 3 * elements * self.attn.dtype_bytes
        return io_bytes / self.hw.memory_bandwidth + self.hw.gate_launch_latency

    def _output_compute_time(self, total_tokens: int) -> float:
        return self._gemm_compute_time(
            total_tokens // self.cp,
            self.tp_q_heads * self.attn.head_dim,
            self.hidden_size,
        )

    def _sp_all_gather_time(self, total_tokens: int) -> float:
        if self.tp == 1:
            return 0.0
        # Each SP collective transfers (TP - 1) / TP of the gathered tensor.
        wire_bytes = (
            (total_tokens // self.cp) * self.hidden_size * self.attn.dtype_bytes
            * (self.tp - 1) / self.tp
        )
        return wire_bytes / self.hw.p2p_bandwidth(self.tp)

    def _sp_reduce_scatter_time(self, total_tokens: int) -> float:
        return self._sp_all_gather_time(total_tokens)

    def _sp_comm_time(self, total_tokens: int) -> float:
        return (
            self._sp_all_gather_time(total_tokens)
            + self._sp_reduce_scatter_time(total_tokens)
        )

    def total_time(self, batch: list[int]) -> float:
        """Seconds for one pack shared by the TP x CP group."""
        self.strategy._validate_batch(batch)
        total_tokens = sum(batch)
        projections = (
            self._qkv_compute_time(total_tokens)
            + self._output_compute_time(total_tokens)
        )
        return (
            projections
            + self.strategy.total_time(batch)
            + self._gate_compute_time(total_tokens)
            + self._sp_comm_time(total_tokens)
        )
