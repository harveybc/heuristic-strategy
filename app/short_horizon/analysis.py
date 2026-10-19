"""Small, tested readers used by the degradation reports."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

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


def path_mae(pred, target, rows):
    """MAE over every available predicted price of the selected rows, and per horizon.

    Only cells with a realised target count; a missing prediction where a target
    exists is an error (no silent drop).
    """
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    valid = np.isfinite(target) & np.asarray(rows, bool)[:, None]
    if not np.isfinite(pred[valid]).all():
        raise ValueError("prediction missing where a target exists")
    err = np.where(valid, np.abs(pred - target), np.nan)
    with np.errstate(invalid="ignore"):
        per_h = np.array([np.nanmean(err[:, j]) if valid[:, j].any() else np.nan for j in range(err.shape[1])])
    return {"mae": float(np.nanmean(err)) if valid.any() else float("nan"), "per_h": per_h,
            "count": int(valid.sum())}


def ruined(df):
    """Ruin flag per row: the stored flag when present, otherwise a drawdown of >= 100 % of the peak
    (marked-to-market equity reached zero), which is derivable from older result tables."""
    flag = df["max_drawdown_pct"] >= 100.0 if "max_drawdown_pct" in df.columns else False
    if "ruined" in df.columns:
        flag = flag | df["ruined"].astype(bool)
    return pd.Series(flag, index=df.index) if not isinstance(flag, pd.Series) else flag


def frequency_validity(df, min_per_week=3.0, max_per_day=4, min_valid_seeds=3):
    """Owner rule (plus: a ruined seed is never valid): a seed is valid with >= ``min_per_week`` trades per business week and
    <= ``max_per_day`` orders per business day; a level needs >= ``min_valid_seeds`` valid seeds.

    Returns (per-row bool Series, per-level bool Series indexed by ``level``).
    """
    seeds = (df["trades_per_week"] >= min_per_week) & (df["max_orders_per_day"] <= max_per_day)
    if "max_drawdown_pct" in df.columns or "ruined" in df.columns:
        seeds = seeds & ~ruined(df)
    levels = seeds.groupby(df["level"]).sum() >= min_valid_seeds
    return seeds, levels
