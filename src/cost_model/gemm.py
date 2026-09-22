import json
import math
from pathlib import Path


class MeasuredGemmTimings:
    """Exact BF16 forward GEMM timings from bench_gemm.py, in seconds."""

    def __init__(self, path: Path | str):
        data = json.loads(Path(path).read_text())
        self.dtype = data["config"]["dtype"]
        if self.dtype != "bfloat16":
            raise ValueError("Measured GEMM timing currently requires bfloat16")
        self.environment = data["environment"]
        self._times: dict[tuple[int, int, int, str], float] = {}
        for row in data["results"]:
            shape = (row["m"], row["k"], row["n"])
            if any(not isinstance(size, int) or size <= 0 for size in shape):
                raise ValueError(f"Invalid GEMM shape: {shape}")
            seconds = row["time_ms"] * 1e-3
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"Invalid GEMM time for {shape}: {seconds}")
            key = (*shape, self.dtype)
            if key in self._times:
                raise ValueError(f"Duplicate GEMM measurement for {key}")
            self._times[key] = seconds
        if not self._times:
            raise ValueError("No GEMM measurements found")

    def time(self, m: int, k: int, n: int, dtype: str = "bfloat16") -> float:
        key = (m, k, n, dtype)
        try:
            return self._times[key]
        except KeyError:
            raise ValueError(
                f"No measured GEMM time for M={m}, K={k}, N={n}, dtype={dtype}"
            ) from None
