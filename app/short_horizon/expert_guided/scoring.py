"""Classifier scores and the executable signal replay used by E0.

* per-direction PR-AUC (average precision, one-vs-rest over ALL eligible rows, NONE included in the base
  rate), ROC-AUC, precision/recall/support at the frozen threshold; log-loss of the model vs the constant
  class prior fitted on TRAIN only (Musashi V3-04);
* a dependence-preserving uncertainty: moving-block bootstrap over ISO weeks of (PR-AUC - prevalence);
* ``SignalRunner``: replays a list of (decision bar, direction) through the SAME block simulator, profile,
  sizing, order rules and daily cap as the ZigZag v2 study (decision after the completed bar, fill at the
  next open, SL = 2 sigma, TP = k_tp sigma or the next hindsight pivot for the expert), on any window.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import zigzag as Z
from .. import zigzag_opt as ZO


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, bool)
    if y.sum() == 0:
        return math.nan
    order = np.argsort(-np.asarray(s, float), kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return math.nan
    r = pd.Series(np.asarray(s, float)).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def log_loss(y: np.ndarray, P: np.ndarray, eps: float = 1e-12) -> float:
    P = np.clip(np.asarray(P, float), eps, 1.0)
    return float(-np.mean(np.log(P[np.arange(len(y)), np.asarray(y, int)])))


def week_keys(ts_ns: np.ndarray) -> np.ndarray:
    t = pd.DatetimeIndex(pd.to_datetime(ts_ns, utc=True))
    iso = t.isocalendar()
    return (iso.year.to_numpy() * 100 + iso.week.to_numpy()).astype(np.int64)


def block_bootstrap_ap_excess(y, s, ts_ns, n_boot=1000, seed=0):
    """(mean, lo95, hi95) of AP - prevalence, resampling whole ISO weeks with replacement."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    wk = week_keys(ts_ns)
    uniq = np.unique(wk)
    groups = [np.flatnonzero(wk == w) for w in uniq]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        if y[idx].sum() == 0:
            continue
        vals.append(average_precision(y[idx], s[idx]) - y[idx].mean())
    vals = np.asarray(vals)
    return float(vals.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def class_scores(y, P, prior, ts_ns, labels=(1, 2), names=("LONG", "SHORT"), boot=True, seed=0):
    out = {"rows": int(len(y)), "log_loss": log_loss(y, P),
           "prior_log_loss": log_loss(y, np.tile(np.asarray(prior, float), (len(y), 1))),
           "class_counts": np.bincount(np.asarray(y, int), minlength=3).tolist()}
    out["log_loss_minus_prior"] = out["log_loss"] - out["prior_log_loss"]
    for c, name in zip(labels, names):
        yc = np.asarray(y) == c
        d = {"support": int(yc.sum()), "prevalence": float(yc.mean()), "pr_auc": average_precision(yc, P[:, c]),
             "roc_auc": roc_auc(yc, P[:, c])}
        d["pr_auc_minus_prevalence"] = d["pr_auc"] - d["prevalence"]
        if boot:
            m, lo, hi = block_bootstrap_ap_excess(yc, P[:, c], ts_ns, seed=seed + c)
            d.update({"boot_excess_mean": m, "boot_excess_lo95": lo, "boot_excess_hi95": hi})
        out[name] = d
    return out


def threshold_signals(P: np.ndarray, tau: float):
    """Direction 1 (LONG) / -1 (SHORT) / 0 at each row: max(pL, pS) >= tau, ties broken toward LONG."""
    pl, ps = P[:, 1], P[:, 2]
    d = np.where(pl >= ps, 1, -1)
    return np.where(np.maximum(pl, ps) >= tau, d, 0)


def precision_recall(y, sig):
    out = {}
    for c, d, name in ((1, 1, "LONG"), (2, -1, "SHORT")):
        pred = sig == d
        tp = int((pred & (y == c)).sum())
        out[name] = {"predicted": int(pred.sum()), "precision": tp / pred.sum() if pred.sum() else math.nan,
                     "recall": tp / (y == c).sum() if (y == c).sum() else math.nan}
    return out


class SignalRunner(ZO.Evaluator):
    """The v2 study's Evaluator on explicit windows (period name -> (start, end))."""

    def __init__(self, bars, cal, periods: dict, **kw):
        super().__init__(bars, cal, **kw)
        self.periods = periods

    def _window(self, period, margin_days=3):
        a, b = (pd.Timestamp(x, tz="UTC").value for x in self.periods[period])
        return a - margin_days * 86_400 * 10 ** 9, a, b

    def execute(self, signals, period, mode="causal", k_tp=1.5, piv=None, cap=4, weeks=None):
        """``weeks``: ISO weeks of the period holding eligible decision rows. The study's own orders/week
        divides by weeks holding a SIGNAL, which overstates a sparse policy; with ``weeks`` given the weekly
        rule is judged on every population week (declared, stricter)."""
        sig = sorted((int(b), int(d)) for b, d in signals)
        m, res = self.run(sig, piv if piv is not None else [], period, mode, k_tp=k_tp, cap=cap, executable=True)
        mm = res.metrics
        if weeks:
            m["orders_per_week_signal_weeks"] = m["orders_per_week"]
            m["orders_per_week"] = float(mm["trades"]) / weeks
            m["valid"] = bool(m["orders_per_week"] >= 3.0 and m["max_orders_per_day"] <= 4 and not m["ruined"]
                              and np.isfinite(m["sharpe"]))
        m.update({"sortino": float(mm["sortino_daily_sqrt365"]), "win_rate": float(mm["win_rate"]),
                  "profit_factor": float(mm["profit_factor"]), "gross": float(mm["gross_usd"]),
                  "cost_spread": float(mm["cost_spread_usd"]), "cost_slippage": float(mm["cost_slippage_usd"]),
                  "cost_commission": float(mm["cost_commission_usd"]), "cost_financing": float(mm["cost_financing_usd"]),
                  "exit_tp": int(mm["exit_tp"]), "exit_sl": int(mm["exit_sl"]), "exit_time": int(mm["exit_time"]),
                  "signals": len(sig), "entries": int(mm["trades"]), "closes": int(mm["trades"]),
                  "below_min_units": int(mm["below_min_units"]), "margin_refused": int(mm["margin_refused_orders"]),
                  "unfilled": int(mm["unfilled_orders"]), "refused_gap": int(mm["refused_entries"])})
        return m, res


def no_trade_metrics() -> dict:
    return {"sharpe": 0.0, "net": 0.0, "max_dd_pct": 0.0, "trades": 0, "orders_per_week": 0.0, "valid": False,
            "sortino": 0.0, "win_rate": 0.0}


def hindsight_signals(pivots):
    return Z.signals_from_pivots(pivots)


def confirmed_signals(pivots, conf):
    return Z.signals_from_pivots(pivots, conf=conf)
