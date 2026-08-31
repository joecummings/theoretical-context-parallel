from dataclasses import dataclass

import numpy as np


@dataclass
class ImbalanceMetrics:
    """Metrics for measuring load imbalance across DP ranks."""

    imbalance: float
    """(max - min) / mean, averaged over steps"""

    cv: float
    """Coefficient of variation"""

    variance: float
    """Mean variance across steps"""

    def __repr__(self) -> str:
        return f"ImbalanceMetrics(imbalance={self.imbalance:.4f}, cv={self.cv:.4f}, var={self.variance:.2e})"


def compute_imbalance_metrics(costs: np.ndarray) -> ImbalanceMetrics:
    """Compute load imbalance metrics from cost array.

    Args:
        costs: Array of shape (n_steps, n_ranks) with costs per rank per step
    """
    rank_max = costs.max(axis=1)
    rank_min = costs.min(axis=1)
    rank_mean = costs.mean(axis=1)

    imbalance = float(((rank_max - rank_min) / rank_mean).mean())
    cv = float((costs.std(axis=1) / rank_mean).mean())
    variance = float(costs.var(axis=1).mean())

    return ImbalanceMetrics(imbalance=imbalance, cv=cv, variance=variance)
