"""Small, tested readers used by the degradation reports."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .blocks import AMBIGUOUS, LOW_FIRST


def zero_crossing(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """First x where y reaches zero walking along x in the given order.

    Linear interpolation between the bracketing grid points; an exact zero is
    returned as is; None when y never changes sign. It reads a curve, it does
    not fit one.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    for i in range(len(y)):
        if y[i] == 0:
            return float(x[i])
        if i and np.sign(y[i]) != np.sign(y[i - 1]):
            return float(x[i - 1] + (x[i] - x[i - 1]) * y[i - 1] / (y[i - 1] - y[i]))
    return None


def calibrated_brier(ft_hat, ft_true, confidence: float) -> float:
    """Brier score of the event LOW_FIRST for a forecaster that states
    ``confidence`` in its own label. Ambiguous truth is excluded."""
    ft_hat, ft_true = np.asarray(ft_hat), np.asarray(ft_true)
    keep = ft_true != AMBIGUOUS
    prob_low = np.where(ft_hat[keep] == LOW_FIRST, confidence, 1.0 - confidence)
    outcome = (ft_true[keep] == LOW_FIRST).astype(float)
    return float(np.mean((prob_low - outcome) ** 2)) if keep.any() else float("nan")
