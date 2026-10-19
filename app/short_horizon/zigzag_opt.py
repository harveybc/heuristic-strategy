"""ZigZag NSGA-II v2: search spaces with log-scale genes and repair, exact hypervolume, convergence rule,
read ledger, and the objective evaluators for (A) the hindsight expert and (B) the causally confirmed
version on the executable path (decision after the completed bar, fill at the next open).

DEVELOPMENT. 2020-2022 = decision; 2023 = JOINT development (it chooses the expert); 2024 = PREVIOUSLY
OBSERVED development (it was read in v1). Nothing here is a confirmation.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import execution as X
from . import zigzag as Z
from .blocks import OK

PIP = Z.PIP                      # 0.0001 price units
POINTS_PER_PIP = 10              # MetaTrader 5-digit EURUSD: 1 point = 0.00001 = 0.1 pip
PENALTY = {"sharpe": -10.0, "dd": 100.0, "lag": 1000.0}
PERIOD_STATUS = {"DECISION_2020_2022": "DECISION", "VALIDATION_2023": "JOINT_DEVELOPMENT",
                 "OBSERVED_2024": "PREVIOUSLY_OBSERVED_DEVELOPMENT"}
PERIODS = {"DECISION_2020_2022": ("2020-01-01", "2023-01-01"), "VALIDATION_2023": ("2023-01-01", "2024-01-01"),
           "OBSERVED_2024": ("2024-01-01", "2025-01-01")}


def executable_profile(base: X.RealisticProfile = X.RealisticProfile()) -> X.RealisticProfile:
    """The realistic profile with no extra entry delay: the block already starts at the next executable bar."""
    return replace(base, entry_delay_bars=0)


# ------------------------------------------------------------------ search spaces
@dataclass(frozen=True)
class Space:
    depth: tuple = (2, 200)
    deviation_pips: tuple = (1.0, 300.0)
    backstep: tuple = (1, 30)
    k_tp: tuple = None           # (lo, hi) for the confirmed version's TP multiple; None for (A)

    def lower(self):
        out = [math.log(self.depth[0]), math.log(self.deviation_pips[0]), float(self.backstep[0])]
        return out + ([float(self.k_tp[0])] if self.k_tp else [])

    def upper(self):
        out = [math.log(self.depth[1]), math.log(self.deviation_pips[1]), float(self.backstep[1])]
        return out + ([float(self.k_tp[1])] if self.k_tp else [])

    def decode(self, genes):
        """(parameters, repaired genes). Depth and Deviation are log-scale genes; Backstep < Depth is
        repaired by setting Backstep = Depth - 1 and writing the repaired value back into the genes."""
        lo, hi = self.lower(), self.upper()
        g = [min(max(float(x), a), b) for x, a, b in zip(genes, lo, hi)]
        depth = int(min(max(round(math.exp(g[0])), self.depth[0]), self.depth[1]))
        dev = float(min(max(round(math.exp(g[1]), 1), self.deviation_pips[0]), self.deviation_pips[1]))
        back = int(min(max(round(g[2]), self.backstep[0]), self.backstep[1]))
        if back >= depth:
            back = depth - 1
            g[2] = float(back)
        p = {"depth": depth, "deviation_pips": dev, "deviation_points": dev * POINTS_PER_PIP, "backstep": back}
        if self.k_tp:
            p["k_tp"] = round(g[3], 3)
        return p, g

    def on_bound(self, p):
        hits = []
        for name, (a, b) in (("depth", self.depth), ("deviation_pips", self.deviation_pips), ("backstep", self.backstep)):
            if p[name] <= a:
                hits.append(f"{name}=lower")
            elif p[name] >= b:
                hits.append(f"{name}=upper")
        if self.k_tp:
            if p["k_tp"] <= self.k_tp[0] + 1e-9:
                hits.append("k_tp=lower")
            elif p["k_tp"] >= self.k_tp[1] - 1e-9:
                hits.append("k_tp=upper")
        return hits


SPACE_A = Space()
SPACE_B = Space(k_tp=(0.25, 3.0))


# ------------------------------------------------------------------ hypervolume and convergence
def _hv2(points, ref):
    pts = sorted((p for p in points if p[0] < ref[0] and p[1] < ref[1]), key=lambda q: (q[0], q[1]))
    hv, best_y = 0.0, ref[1]
    for x, y in pts:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


def hypervolume(points, ref) -> float:
    """Exact hypervolume of ``points`` (minimization) dominated below ``ref``; 2 or 3 objectives."""
    pts = [list(map(float, p)) for p in points if all(a < r for a, r in zip(p, ref))]
    if not pts:
        return 0.0
    if len(ref) == 2:
        return _hv2(pts, ref)
    if len(ref) != 3:
        raise ValueError("2 or 3 objectives")
    zs = sorted({p[2] for p in pts}) + [float(ref[2])]
    hv = 0.0
    for z0, z1 in zip(zs[:-1], zs[1:]):
        slab = [p[:2] for p in pts if p[2] <= z0]
        hv += _hv2(slab, ref[:2]) * (z1 - z0)
    return hv


def converged(hv_history, min_generations: int = 80, window: int = 20, tol: float = 0.01) -> bool:
    """hv_history[g] = hypervolume after generation g (g = 0 is the initial population)."""
    if len(hv_history) - 1 < max(min_generations, window):
        return False
    a, b = float(hv_history[-1 - window]), float(hv_history[-1])
    return a > 0 and abs(b - a) / a < tol


def ledger(path, period: str, status: str, purpose: str, **extra) -> None:
    row = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "period": period, "status": status,
           "purpose": purpose, **extra}
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


# ------------------------------------------------------------------ evaluators
class Evaluator:
    """Objective evaluator on one bar set. ``bars`` and ``cal`` are the lake bars and the dev calendar."""

    def __init__(self, bars, cal, capital: float = 10_000.0, m_sl: float = 2.0, inset: float = PIP):
        self.bars, self.cal, self.capital, self.m_sl, self.inset = bars, cal, capital, m_sl, inset
        self.h, self.l = bars.HIGH.to_numpy(float), bars.LOW.to_numpy(float)
        self.sigma = Z.causal_sigma(bars)
        self.sigma_rule = Z.causal_range_sigma(bars)
        self.idx_ns = bars.index.as_unit("ns").asi8
        self.prof_exec = executable_profile()
        self.prof_v1 = X.RealisticProfile()
        self._piv = {}

    def pivots(self, p):
        key = (p["depth"], p["deviation_pips"], p["backstep"])
        if key not in self._piv:
            zp = Z.Params(*key)
            piv = Z.hindsight_pivots(self.h, self.l, zp)
            conf = Z.confirmation_bars_fast(self.h, self.l, piv, zp)
            if len(self._piv) > 4000:
                self._piv.clear()
            self._piv[key] = (piv, conf)
        return self._piv[key]

    def _window(self, period, margin_days=3):
        a, b = (pd.Timestamp(x, tz="UTC").value for x in PERIODS[period])
        return a - margin_days * 86_400 * 10 ** 9, a, b

    def run(self, signals, piv, period, mode, k_tp=1.0, cap=4, executable=True, leverage=100.0):
        lo_ns, a, b = self._window(period)
        sig = [(s, d) for s, d in signals if lo_ns <= self.idx_ns[s] < b]
        blocks = Z.signal_blocks(self.bars, self.cal, sig, executable=executable)
        orders = Z.reversal_orders(blocks, sig, piv, self.sigma, mode=mode, k_tp=k_tp, m_sl=self.m_sl, inset=self.inset,
                                   sigma_rule=self.sigma_rule)
        sig_blk = self.sigma_rule[np.minimum(blocks.bar_lo, len(self.bars) - 1)]
        checked, skipped = [], 0
        for k, o in enumerate(orders):
            if o is None:
                checked.append(None)
                continue
            try:
                X.check_order(o, float(blocks.ref[k]), float(sig_blk[k]), X.OrderRules())
                checked.append(o)
            except X.OrderRuleError:
                checked.append(None)
                skipped += 1
        prof = self.prof_exec if executable else self.prof_v1
        res = X.run_backtest(self.bars, blocks, checked, prof, self.cal, self.capital,
                             (pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")), profile=prof,
                             max_orders_per_day=cap, ruin_floor=0.0, order_rules=X.OrderRules(), sigma_block=sig_blk,
                             sizing=X.Sizing(rel_volume=0.10, leverage=leverage))
        m = res.metrics
        valid = bool(m["trades_per_week"] >= 3.0 and m["max_orders_per_day"] <= 4 and not m["ruined"]
                     and np.isfinite(m["sharpe_daily_sqrt365"]))
        refused_reopen = int(sum(1 for k in range(len(blocks)) if blocks.status[k] == "BEFORE_OPERATIONAL_START"
                                 and a <= blocks.start[k].value < b))
        out = {"sharpe": float(m["sharpe_daily_sqrt365"]), "max_dd_pct": float(m["max_drawdown_pct"]),
               "max_dd_usd": float(m["max_drawdown_usd"]), "net": float(m["net_profit_usd"]), "trades": int(m["trades"]),
               "orders_per_week": float(m["trades_per_week"]), "max_orders_per_day": int(m["max_orders_per_day"]),
               "share_exit_tp": float(m["share_exit_tp"]), "share_exit_sl": float(m["share_exit_sl"]),
               "share_exit_time": float(m["share_exit_time"]), "ruined": bool(m["ruined"]), "valid": valid,
               "rule_skipped_orders": skipped, "capped_orders": int(m["capped_orders"]),
               "refused_before_operational_start": refused_reopen}
        return out, res

    def lag(self, piv, conf, period):
        _, a, b = self._window(period)
        lags = [c - i for (i, _, _), c in zip(piv, conf) if c >= 0 and a <= self.idx_ns[i] < b]
        return (float(np.median(lags)) if lags else float("nan")), (float(np.mean(lags)) if lags else float("nan")), len(lags)

    def eval_A(self, p, period="DECISION_2020_2022", cap=4, executable=True):
        piv, conf = self.pivots(p)
        m, res = self.run(Z.signals_from_pivots(piv), piv, period, "hindsight", cap=cap, executable=executable)
        m["pivots"] = len(piv)
        return m, res

    def eval_B(self, p, period="DECISION_2020_2022", cap=4, executable=True, k_tp=None):
        piv, conf = self.pivots(p)
        sig = Z.signals_from_pivots(piv, conf=conf)
        m, res = self.run(sig, piv, period, "causal", k_tp=p.get("k_tp", 1.0) if k_tp is None else k_tp, cap=cap,
                          executable=executable)
        med, mean, n = self.lag(piv, conf, period)
        m.update({"lag_median": med, "lag_mean": mean, "confirmed_pivots": n, "pivots": len(piv)})
        return m, res


def objectives_A(m):
    return (m["sharpe"], m["max_dd_pct"]) if m["valid"] else (PENALTY["sharpe"], PENALTY["dd"])


def objectives_B(m):
    if not m["valid"] or not np.isfinite(m["lag_median"]):
        return (PENALTY["sharpe"], PENALTY["dd"], PENALTY["lag"])
    return (m["sharpe"], m["max_dd_pct"], m["lag_median"])


def to_min(obj):
    """Objectives as minimization vectors (Sharpe negated) for the hypervolume."""
    return [-obj[0]] + list(obj[1:])


REF_A = [-PENALTY["sharpe"], PENALTY["dd"]]
REF_B = [-PENALTY["sharpe"], PENALTY["dd"], PENALTY["lag"]]
