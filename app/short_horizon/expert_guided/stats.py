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


SANITY_REFUSAL = "REFUSED_FAILED_SANITY"


def sanity_refusal(*inputs) -> dict | None:
    """A verdict that involves a FAILED_SANITY unit is refused (owner 2026-10-25: a run with zero trades, one-sided
    trades or no early close measures nothing): inputs carrying ``sanity == "FAILED_SANITY"`` name their unit."""
    bad = [x for x in inputs if isinstance(x, dict) and x.get("sanity") == "FAILED_SANITY"]
    if not bad:
        return None
    return {"verdict": SANITY_REFUSAL, "failed_units": [x.get("unit") for x in bad],
            "causes": [x.get("sanity_causes") for x in bad], "rule": "no verdict involves a FAILED_SANITY unit"}


def holds(fold_stats: dict) -> dict:
    """HOLDS_RULE over {fold: {"sharpe":..., "se_lo":...}} of the ensemble on the validation window of each fold.
    Refused (``SANITY_REFUSAL``) when any fold's input is a FAILED_SANITY unit."""
    ref = sanity_refusal(*fold_stats.values())
    if ref is not None:
        return {"rule": HOLDS_RULE, "per_fold": {}, **ref}
    per = {f: bool(math.isfinite(v["se_lo"]) and (v["sharpe"] - v["se_lo"]) > 0) for f, v in fold_stats.items()}
    return {"rule": HOLDS_RULE, "per_fold": per, "verdict": "HOLDS" if per and all(per.values()) else "DOES_NOT_HOLD"}


def separable(a: dict, b: dict) -> dict:
    """SEPARABLE_RULE for two windows' {"sharpe", "se_lo"}; refused when either side is a FAILED_SANITY unit."""
    ref = sanity_refusal(a, b)
    if ref is not None:
        return {"rule": SEPARABLE_RULE, **ref}
    pooled = math.sqrt(a["se_lo"] ** 2 + b["se_lo"] ** 2) if math.isfinite(a["se_lo"]) and math.isfinite(b["se_lo"]) else float("nan")
    delta = a["sharpe"] - b["sharpe"]
    return {"rule": SEPARABLE_RULE, "delta": float(delta), "pooled_se": float(pooled),
            "verdict": "SEPARABLE" if math.isfinite(pooled) and abs(delta) > pooled else "NOT_SEPARABLE"}


# ======================================================================================================================
# PROTOCOL v11 (audit A5-1 / A5-7 / A5-3 / A5-10, owner-approved 2026-10-26): the primary statistic is the per-trade net
# expectancy in sigma with a day-block bootstrap interval POOLED across folds; arm differences are PAIRED, TWO-SIDED
# tests on daily returns with multiplicity control; every strategy search reports a deflated Sharpe and the
# probability of backtest overfitting; the minimum detectable effect is printed before any run. The v1 rules above
# stay only for reading the old reports.
from statistics import NormalDist as _ND

ALPHA = 0.05
HOLDS_RULE_V2 = ("an arm HOLDS only if the lower bound of the (1 - alpha) day-block bootstrap interval of its per-trade net "
                 "expectancy (sigma, after costs), POOLED across every decision block of the protocol, is > 0; otherwise "
                 "DOES_NOT_HOLD; a FAILED_SANITY input refuses the verdict")
SEPARABLE_RULE_V2 = ("two arms are SEPARABLE only when the paired, two-sided test of their daily-return difference on the same "
                     "decision rows has p < alpha AFTER Romano-Wolf step-down across every arm x variant read on that "
                     "block; otherwise NOT_SEPARABLE; the sign of the difference is always reported")
NOISE_FLOOR_SHARPE_CALIBRATION = 4.1
NOISE_FLOOR_NOTE = ("calibration-block Sharpe of the strategy search on PURE-NOISE inputs = 4.1 (A6 controls, v1.3, wf2022, 1 seed): "
                    "a calibration Sharpe below this floor is NOT evidence of signal")


def _z(p: float) -> float:
    return _ND().inv_cdf(p)


def mde(se: float, alpha: float = ALPHA, power: float = 0.8, two_sided: bool = True) -> float:
    """Minimum detectable effect for a statistic with standard error ``se``: (z_{1-alpha(/2)} + z_power) * se."""
    za = _z(1 - alpha / 2) if two_sided else _z(1 - alpha)
    return float((za + _z(power)) * se)


def mde_table(se_by_window: dict, alpha: float = ALPHA, power: float = 0.8) -> dict:
    """MDE per window and for the pooled mean of k windows (SE / sqrt(k)), printed BEFORE any run."""
    out = {w: {"se": float(s), "mde_two_sided": mde(s, alpha, power, True), "mde_one_sided": mde(s, alpha, power, False)} for w, s in se_by_window.items()}
    if se_by_window:
        k = len(se_by_window)
        se_p = float(np.sqrt(np.mean(np.square(list(se_by_window.values()))) / k))
        out["POOLED"] = {"se": se_p, "mde_two_sided": mde(se_p, alpha, power, True), "mde_one_sided": mde(se_p, alpha, power, False), "windows": k}
    return out


def day_block_bootstrap_expectancy(trades_by_day: dict, block: float = 5.0, n: int = 2000, seed: int = 20261026,
                                   alpha: float = ALPHA) -> dict:
    """Per-trade net expectancy (sigma) with a day-block bootstrap interval. ``trades_by_day`` = {day_key: [net_sigma, ...]}
    for EVERY day of the pooled decision blocks (days without trades included as []); days are resampled with the
    stationary bootstrap, the expectancy of a resample is the mean over all trades of its days."""
    days = sorted(trades_by_day)
    T = len(days)
    allt = np.concatenate([np.asarray(trades_by_day[d], float) for d in days]) if days else np.zeros(0)
    if len(allt) == 0:
        return {"expectancy": float("nan"), "low": float("nan"), "high": float("nan"), "n_trades": 0, "n_days": T, "alpha": alpha}
    rng = np.random.default_rng(seed)
    per = [np.asarray(trades_by_day[d], float) for d in days]
    vals = np.empty(n)
    for b in range(n):
        idx = stationary_bootstrap_indices(T, block, rng)
        s = np.concatenate([per[i] for i in idx]) if T else np.zeros(0)
        vals[b] = s.mean() if len(s) else np.nan
    vals = vals[np.isfinite(vals)]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if len(vals) else (np.nan, np.nan)
    return {"expectancy": float(allt.mean()), "low": float(lo), "high": float(hi), "n_trades": int(len(allt)), "n_days": T,
            "block": float(block), "n": int(n), "seed": int(seed), "alpha": alpha, "sd_per_trade": float(allt.std(ddof=1)) if len(allt) > 1 else float("nan")}


def paired_block_bootstrap_diff(ra, rb, block: float = 5.0, n: int = 2000, seed: int = 20261026, alpha: float = ALPHA,
                                annualization: float = ANNUALIZATION) -> dict:
    """Two-sided paired test of Sharpe(a) - Sharpe(b) on ALIGNED daily returns: the same stationary-bootstrap index
    vector is applied to both series (pairing kept), p = two-sided percentile of the centred resamples."""
    ra, rb = np.asarray(ra, float), np.asarray(rb, float)
    if len(ra) != len(rb):
        raise ValueError("PAIRED_SERIES_LENGTH_MISMATCH")
    T = len(ra)
    delta = sharpe(ra, annualization) - sharpe(rb, annualization)
    if T < 3:
        return {"delta": float(delta), "p_two_sided": float("nan"), "low": float("nan"), "high": float("nan"), "days": T}
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for b in range(n):
        idx = stationary_bootstrap_indices(T, block, rng)
        vals[b] = sharpe(ra[idx], annualization) - sharpe(rb[idx], annualization)
    centred = vals - vals.mean()
    p = float(np.mean(np.abs(centred) >= abs(delta)))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"delta": float(delta), "p_two_sided": p, "low": float(lo), "high": float(hi), "days": T, "n": int(n), "block": float(block),
            "seed": int(seed), "sign": "a>b" if delta > 0 else ("a<b" if delta < 0 else "equal")}


def _hac_cov(X: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West (Bartlett) long-run covariance of the rows of X (T x k), demeaned."""
    Z = X - X.mean(0)
    T = len(Z)
    S = Z.T @ Z / T
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = Z[L:].T @ Z[:-L] / T
        S += w * (G + G.T)
    return S


def hac_sharpe_diff_test(ra, rb, lags: int | None = None, annualization: float = ANNUALIZATION) -> dict:
    """Ledoit & Wolf (2008) delta-method test of the Sharpe difference with a HAC (Newey-West) variance of the moment
    vector (r_a, r_a^2, r_b, r_b^2); two-sided normal p-value. ``lags`` default floor(4 (T/100)^(2/9))."""
    ra, rb = np.asarray(ra, float), np.asarray(rb, float)
    if len(ra) != len(rb):
        raise ValueError("PAIRED_SERIES_LENGTH_MISMATCH")
    T = len(ra)
    if T < 10:
        return {"delta": float(sharpe(ra, annualization) - sharpe(rb, annualization)), "se": float("nan"), "z": float("nan"), "p_two_sided": float("nan"), "days": T}
    lags = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0))) if lags is None else int(lags)
    X = np.column_stack([ra, ra ** 2, rb, rb ** 2])
    mu_a, g_a, mu_b, g_b = X.mean(0)
    v_a, v_b = g_a - mu_a ** 2, g_b - mu_b ** 2
    if v_a <= 0 or v_b <= 0:
        return {"delta": float("nan"), "se": float("nan"), "z": float("nan"), "p_two_sided": float("nan"), "days": T}
    sr = lambda mu, v: mu / np.sqrt(v)  # noqa: E731
    delta_d = sr(mu_a, v_a) - sr(mu_b, v_b)
    grad = np.array([(g_a) / v_a ** 1.5, -0.5 * mu_a / v_a ** 1.5, -(g_b) / v_b ** 1.5, 0.5 * mu_b / v_b ** 1.5])
    S = _hac_cov(X, lags)
    var = float(grad @ S @ grad) / T
    se_d = float(np.sqrt(max(var, 0.0)))
    z = delta_d / se_d if se_d > 0 else float("nan")
    p = float(2 * (1 - _ND().cdf(abs(z)))) if np.isfinite(z) else float("nan")
    return {"delta": float(delta_d * annualization), "se": float(se_d * annualization), "z": float(z), "p_two_sided": p, "days": T, "lags": lags}


def romano_wolf(stats: np.ndarray, boot: np.ndarray) -> np.ndarray:
    """Romano-Wolf (2005) step-down adjusted two-sided p-values for k hypotheses H_j: theta_j = 0, from the observed
    statistics ``stats`` (k,) and ``boot`` (B, k) bootstrap replicates of the SAME statistics (paired resampling);
    the replicates are centred at their mean before taking maxima (studentisation left to the caller)."""
    stats = np.asarray(stats, float)
    boot = np.asarray(boot, float)
    k = len(stats)
    C = np.abs(boot - boot.mean(0))
    order = np.argsort(-np.abs(stats))
    adj = np.empty(k)
    prev = 0.0
    for r, j in enumerate(order):
        rest = order[r:]
        maxs = C[:, rest].max(1)
        p = float(np.mean(maxs >= abs(stats[j])))
        prev = max(prev, p)                       # monotonicity enforcement of the step-down
        adj[j] = min(prev, 1.0)
    return adj


def deflated_sharpe(sr_daily: float, n_trials: int, T: int, skew: float = 0.0, kurt: float = 3.0, sr_var_trials: float | None = None) -> dict:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio: the probability that the observed DAILY Sharpe exceeds the
    expected maximum of ``n_trials`` null trials (E[max] from the trials' Sharpe variance, Euler-Mascheroni form),
    with the non-normality correction. ``sr_var_trials`` default = the iid variance 1/T."""
    if n_trials < 1 or T < 3:
        return {"dsr": float("nan"), "sr0": float("nan")}
    v = (1.0 / T) if sr_var_trials is None else float(sr_var_trials)
    gamma = 0.5772156649015329
    if n_trials == 1:
        sr0 = 0.0
    else:
        sr0 = float(np.sqrt(v) * ((1 - gamma) * _z(1 - 1.0 / n_trials) + gamma * _z(1 - 1.0 / (n_trials * np.e))))
    denom = float(np.sqrt(max(1 - skew * sr_daily + (kurt - 1) / 4.0 * sr_daily ** 2, 1e-12)))
    z = (sr_daily - sr0) * np.sqrt(T - 1) / denom
    return {"dsr": float(_ND().cdf(z)), "sr0": sr0, "z": float(z), "n_trials": int(n_trials), "T": int(T)}


def pbo(returns: np.ndarray, S: int = 16, max_combos: int = 200, seed: int = 20261026) -> dict:
    """Probability of backtest overfitting (Bailey et al. 2017, CSCV): ``returns`` (T, k) daily returns of k trials;
    T split into S blocks; for each train/test half-split the in-sample best trial's out-of-sample rank gives
    logit lambda; PBO = P(lambda < 0). At most ``max_combos`` seeded random half-splits."""
    R = np.asarray(returns, float)
    T, k = R.shape
    if k < 2 or T < 2 * S:
        return {"pbo": float("nan"), "combos": 0}
    blocks = np.array_split(np.arange(T), S)
    rng = np.random.default_rng(seed)
    lam = []
    for _ in range(max_combos):
        tr = np.sort(rng.choice(S, S // 2, replace=False))
        te = np.setdiff1d(np.arange(S), tr)
        itr = np.concatenate([blocks[i] for i in tr]); ite = np.concatenate([blocks[i] for i in te])
        sr_tr = np.array([sharpe(R[itr, j]) for j in range(k)]); sr_te = np.array([sharpe(R[ite, j]) for j in range(k)])
        best = int(np.argmax(sr_tr))
        rank = (np.sum(sr_te < sr_te[best]) + 1) / (k + 1.0)
        lam.append(float(np.log(rank / (1 - rank))))
    lam = np.asarray(lam)
    return {"pbo": float(np.mean(lam < 0)), "combos": int(len(lam)), "lambda_mean": float(lam.mean()), "S": int(S), "trials": int(k)}


def holds_v2(expectancy: dict, inputs=()) -> dict:
    """HOLDS_RULE_V2 on a pooled ``day_block_bootstrap_expectancy`` document; refused on FAILED_SANITY inputs."""
    ref = sanity_refusal(*inputs)
    if ref is not None:
        return {"rule": HOLDS_RULE_V2, **ref}
    ok = math.isfinite(expectancy.get("low", float("nan"))) and expectancy["low"] > 0
    return {"rule": HOLDS_RULE_V2, "expectancy": expectancy.get("expectancy"), "low": expectancy.get("low"), "high": expectancy.get("high"),
            "n_trades": expectancy.get("n_trades"), "verdict": "HOLDS" if ok else "DOES_NOT_HOLD"}


def separable_v2(delta: float, p_adjusted: float, alpha: float = ALPHA, inputs=()) -> dict:
    """SEPARABLE_RULE_V2 from a paired two-sided test's Romano-Wolf adjusted p-value; refused on FAILED_SANITY inputs."""
    ref = sanity_refusal(*inputs)
    if ref is not None:
        return {"rule": SEPARABLE_RULE_V2, **ref}
    sep = math.isfinite(p_adjusted) and p_adjusted < alpha
    return {"rule": SEPARABLE_RULE_V2, "delta": float(delta), "p_adjusted": float(p_adjusted), "alpha": alpha,
            "sign": "positive" if delta > 0 else ("negative" if delta < 0 else "zero"),
            "verdict": "SEPARABLE" if sep else "NOT_SEPARABLE"}


def sign_flip(values_by_window: dict) -> dict:
    """Two-sided diagnostic: how many windows are positive / negative and whether the sign is stable."""
    v = {k: float(x) for k, x in values_by_window.items() if x is not None and math.isfinite(float(x))}
    pos = sum(1 for x in v.values() if x > 0); neg = sum(1 for x in v.values() if x < 0)
    return {"windows": len(v), "positive": pos, "negative": neg, "stable_sign": bool(v) and (pos == len(v) or neg == len(v)), "values": v}


def noise_floor_note(calibration_sharpe: float) -> dict:
    """Whether a calibration-block Sharpe clears the pure-noise selection floor measured by the A6 controls."""
    s = float(calibration_sharpe) if calibration_sharpe is not None else float("nan")
    return {"calibration_sharpe": s, "noise_floor": NOISE_FLOOR_SHARPE_CALIBRATION, "note": NOISE_FLOOR_NOTE,
            "evidence": "ABOVE_NOISE_FLOOR" if math.isfinite(s) and s > NOISE_FLOOR_SHARPE_CALIBRATION else "NOT_EVIDENCE_BELOW_NOISE_FLOOR"}


def seed_mean_ci(values, alpha: float = ALPHA) -> dict:
    """Mean of per-seed values with ddof=1 standard deviation, standard error and a t interval (A5-10)."""
    x = np.asarray([float(v) for v in values], float)
    n = len(x)
    if n == 0:
        return {"mean": float("nan"), "sd": float("nan"), "se": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    try:
        from scipy.stats import t as _t
        q = float(_t.ppf(1 - alpha / 2, n - 1)) if n > 1 else float("nan")
    except Exception:  # pragma: no cover - scipy absent
        q = _z(1 - alpha / 2)
    return {"mean": float(x.mean()), "sd": sd, "se": se, "low": float(x.mean() - q * se) if n > 1 else float("nan"),
            "high": float(x.mean() + q * se) if n > 1 else float("nan"), "n": n, "ddof": 1}


def tie_band(a: dict, b: dict) -> dict:
    """Two seed-mean documents are TIED when |mean_a - mean_b| <= sqrt(se_a^2 + se_b^2) (A5-10 tie band)."""
    se = math.sqrt((a["se"] if math.isfinite(a["se"]) else 0.0) ** 2 + (b["se"] if math.isfinite(b["se"]) else 0.0) ** 2)
    d = a["mean"] - b["mean"]
    return {"delta": float(d), "band": float(se), "tied": bool(abs(d) <= se)}
