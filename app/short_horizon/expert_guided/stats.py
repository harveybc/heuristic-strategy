"""Uncertainty of a replay's Sharpe ratio (owner 2026-10-25: decide only when a difference exceeds its error).

The replay engine (``execution._account``) reports ``sharpe_daily_sqrt365`` = mean(daily return) / std(daily return,
ddof 1) * sqrt(365) over calendar days (closures forward-filled). This module gives that figure its error:

* ``sharpe_se_lo``: Lo (2002, "The Statistics of Sharpe Ratios", eq. for iid returns) asymptotic standard error of the
  daily Sharpe, SE = sqrt((1 + SR_d^2 / 2) / T), scaled by the same sqrt(365). A lower bound on the true error when
  returns are autocorrelated, which is why the bootstrap below is reported next to it.
* ``stationary_bootstrap_ci``: Politis & Romano (1994) stationary bootstrap of the daily returns (geometric block
  lengths with mean ``block`` days, circular wrap), ``n`` resamples, seeded; percentile interval of the annualized
  Sharpe. Resamples whose std is zero contribute a Sharpe of 0 (the engine's own convention).
* ``daily_returns``: the engine's return series from its ``daily_equity`` (calendar days) and the initial capital, so
  the point estimate recomputed here equals the engine's figure.

Pre-registered rules (written before any walk-forward result was read, 2026-10-25) live in ``HOLDS_RULE`` and
``SEPARABLE_RULE``; ``holds`` and ``separable`` implement them literally.
"""
from __future__ import annotations

import math

import numpy as np

ANNUALIZATION = math.sqrt(365)
HOLDS_RULE = ("an arm HOLDS only if its seed-ensemble validation Sharpe minus one Lo standard error is > 0 in BOTH walk-forward "
              "folds (wf2022 and wf2023); otherwise DOES_NOT_HOLD")
SEPARABLE_RULE = ("two arms are SEPARABLE on a window only when |Sharpe_a - Sharpe_b| > the pooled Lo standard error "
                  "sqrt(SE_a^2 + SE_b^2); otherwise NOT_SEPARABLE")


def daily_returns(daily_equity, initial_capital: float) -> np.ndarray:
    """The engine's daily return series: equity per calendar day (already forward-filled) preceded by the initial capital."""
    eq = np.r_[float(initial_capital), np.asarray(daily_equity, float)]
    return np.diff(eq) / eq[:-1]


def sharpe(ret: np.ndarray, annualization: float = ANNUALIZATION) -> float:
    ret = np.asarray(ret, float)
    if len(ret) < 2:
        return 0.0
    sd = ret.std(ddof=1)
    return float(ret.mean() / sd * annualization) if sd > 0 else 0.0


def sharpe_se_lo(ret: np.ndarray, annualization: float = ANNUALIZATION) -> float:
    """Lo (2002) iid asymptotic SE of the annualized Sharpe: sqrt((1 + SR_d^2/2) / T) * annualization."""
    ret = np.asarray(ret, float)
    T = len(ret)
    if T < 2:
        return float("nan")
    sd = ret.std(ddof=1)
    sr_d = float(ret.mean() / sd) if sd > 0 else 0.0
    return float(math.sqrt((1.0 + 0.5 * sr_d ** 2) / T) * annualization)


def stationary_bootstrap_indices(T: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """One stationary-bootstrap index vector of length T: blocks start uniformly, lengths ~ Geometric(1/block), wrap."""
    p = 1.0 / float(block)
    idx = np.empty(T, np.int64)
    i = 0
    while i < T:
        start = int(rng.integers(0, T))
        length = int(rng.geometric(p))
        k = min(length, T - i)
        idx[i:i + k] = (start + np.arange(k)) % T
        i += k
    return idx


def stationary_bootstrap_ci(ret: np.ndarray, block: float = 5.0, n: int = 2000, seed: int = 20261025, alpha: float = 0.05,
                            annualization: float = ANNUALIZATION) -> dict:
    """Percentile (1 - alpha) interval of the annualized Sharpe under the stationary bootstrap of the daily returns."""
    ret = np.asarray(ret, float)
    T = len(ret)
    if T < 2:
        return {"low": float("nan"), "high": float("nan"), "n": int(n), "block": float(block), "seed": int(seed), "alpha": alpha}
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for b in range(n):
        vals[b] = sharpe(ret[stationary_bootstrap_indices(T, block, rng)], annualization)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"low": float(lo), "high": float(hi), "n": int(n), "block": float(block), "seed": int(seed), "alpha": alpha,
            "boot_mean": float(vals.mean()), "boot_std": float(vals.std(ddof=1))}


def describe(ret: np.ndarray, point: float | None = None, **boot) -> dict:
    """Sharpe point (recomputed), Lo SE, bootstrap interval and the sample size, for one replay window."""
    ret = np.asarray(ret, float)
    sr = sharpe(ret)
    se = sharpe_se_lo(ret)
    out = {"sharpe": sr, "se_lo": se, "days": int(len(ret)), "sharpe_minus_se": (sr - se) if math.isfinite(se) else float("nan"),
           "bootstrap": stationary_bootstrap_ci(ret, **boot)}
    if point is not None:
        out["engine_sharpe"] = float(point)
        out["recomputed_equals_engine"] = bool(abs(float(point) - sr) <= 1e-9 * max(1.0, abs(sr)))
    return out


def holds(fold_stats: dict) -> dict:
    """HOLDS_RULE over {fold: {"sharpe":..., "se_lo":...}} of the ensemble on the validation window of each fold."""
    per = {f: bool(math.isfinite(v["se_lo"]) and (v["sharpe"] - v["se_lo"]) > 0) for f, v in fold_stats.items()}
    return {"rule": HOLDS_RULE, "per_fold": per, "verdict": "HOLDS" if per and all(per.values()) else "DOES_NOT_HOLD"}


def separable(a: dict, b: dict) -> dict:
    """SEPARABLE_RULE for two windows' {"sharpe", "se_lo"}."""
    pooled = math.sqrt(a["se_lo"] ** 2 + b["se_lo"] ** 2) if math.isfinite(a["se_lo"]) and math.isfinite(b["se_lo"]) else float("nan")
    delta = a["sharpe"] - b["sharpe"]
    return {"rule": SEPARABLE_RULE, "delta": float(delta), "pooled_se": float(pooled),
            "verdict": "SEPARABLE" if math.isfinite(pooled) and abs(delta) > pooled else "NOT_SEPARABLE"}
