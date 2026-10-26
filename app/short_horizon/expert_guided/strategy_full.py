"""The FULL parametric upper layer (owner 2026-10-26): TP and SL per direction as explicit formulas of the head outputs,
the ORDER VOLUME per direction as an explicit formula, and every coefficient optimised with NSGA-II on the selection
block. Two declared families per quantity, compared in the simulation sweeps:

  TP / SL   family "cell"   TP_d, SL_d = the grid cell maximising the predicted net for side d (C3: outcomes grid relative
                            to the per-bar effective band; TP < SL only when the predicted net clears cost + 0.10 sigma)
            family "param"  TP_d = sigma_t * (a_d + b_d * q_d),  SL_d = sigma_t * (c_d + e_d * r_d)
                            q_d = calibrated P(TP first | side d), r_d = calibrated P(SL first | side d) (the 3-class opener's
                            calibrated probabilities at the reference cell); both bounded to the engine band [lo_t, cap_t]
                            and the business floors (env tp_bounds / sl_bounds)
  volume    family "risk"   v_d = min(1, risk_pct * balance / (SL distance * units_full))   (exec_variants.risk_units, F3)
            family "param"  v_d = clip(alpha_d + beta_d * conf_d - gamma_d * p_nop - delta_d * risk_d, 0, 1)
                            conf_d = P(TP first | d), risk_d = P(SL first | d);  v < v_min -> no order
  units = v * f * balance * rel_volume * leverage / price (integer; the engine's own sizing with percentual_order_size = f * v;
          leverage <= 1:50; rel_volume 0.10 from the replay contract); SL and TP on every order, always.
The open / close thresholds and the NOP veto stay C3's quantile genes (frozen on the calibration block); the best-of-day
observation window is a gene too. Nothing here reads the decision block.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from . import action_contract as ACT
from . import c3 as C3
from . import env as E
from . import exec_variants as XV

TP_FAMILIES = ("cell", "param")
VOL_FAMILIES = ("risk", "param")
SIDES = ("l", "s")
# coefficient genes (per side), with their declared bounds
COEF_BOUNDS = {"tp_a": [0.25, 2.0], "tp_b": [-1.5, 1.5], "sl_c": [0.5, 2.5], "sl_e": [-1.5, 1.5],
               "v_alpha": [0.0, 1.0], "v_beta": [-1.0, 1.0], "v_gamma": [0.0, 1.0], "v_delta": [0.0, 1.0]}
V_MIN_BOUNDS = [0.0, 0.5]
OBSERVE_BOUNDS = [0.0, 3.0]          # best-of-day observation window (rounded to an integer number of signals)
THRESHOLD_GENES = C3.GENES_Q         # open_rate_l/s, close_rem_rate_l/s, close_p_rate_l/s, allow_reverse, nop_veto_rate
GENE_SPACE_FULL = "calibration_quantile_plus_coefficients"


def gene_bounds(tp_family: str, vol_family: str) -> dict:
    """The optimiser's gene dictionary for a family pair (only the coefficients the families consume are genes)."""
    g = {k: list(v) for k, v in C3.Q_BOUNDS.items()}
    g["observe_bars"] = list(OBSERVE_BOUNDS)
    if tp_family == "param":
        for s in SIDES:
            for k in ("tp_a", "tp_b", "sl_c", "sl_e"):
                g[f"{k}_{s}"] = list(COEF_BOUNDS[k])
    if vol_family == "param":
        for s in SIDES:
            for k in ("v_alpha", "v_beta", "v_gamma", "v_delta"):
                g[f"{k}_{s}"] = list(COEF_BOUNDS[k])
        g["v_min"] = list(V_MIN_BOUNDS)
    return g


@dataclass(frozen=True)
class StrategyFull:
    base: C3.StrategyC3                      # thresholds (absolute, mapped from the quantile genes), reverse, veto, early_close
    tp_family: str
    vol_family: str
    observe_bars: int
    coef: dict                               # the coefficient genes consumed by the families (per side)
    v_min: float = 0.0
    risk_pct: float = 0.01

    def to_dict(self) -> dict:
        return {"base": self.base.to_dict(), "tp_family": self.tp_family, "vol_family": self.vol_family, "observe_bars": self.observe_bars,
                "coef": dict(self.coef), "v_min": self.v_min, "risk_pct": self.risk_pct}

    def no_early_close(self):
        return dataclasses.replace(self, base=self.base.no_early_close())

    @staticmethod
    def from_genes(g_abs: dict, veto: bool, tp_family: str = "cell", vol_family: str = "risk", risk_pct: float = 0.01) -> "StrategyFull":
        base = C3.StrategyC3.from_genes(g_abs, veto)
        coef = {k: float(v) for k, v in g_abs.items() if k.split("_")[0] in ("tp", "sl", "v") and k != "v_min"}
        return StrategyFull(base, tp_family, vol_family, int(round(float(g_abs.get("observe_bars", 0)))), coef, float(g_abs.get("v_min", 0.0)), risk_pct)


def strategy_from_dict(d: dict) -> StrategyFull:
    return StrategyFull(C3.StrategyC3(**d["base"]), d["tp_family"], d["vol_family"], int(d["observe_bars"]), dict(d["coef"]), float(d.get("v_min", 0.0)),
                        float(d.get("risk_pct", 0.01)))


def gene_map_full(view: dict, cal_rows, closer_ref: dict, cells, e_time, cost, bounds: dict) -> ACT.GeneMap:
    """C3's quantile map for the threshold genes plus pass-through ("copy") rules for every coefficient gene."""
    gm = C3.gene_map(view, cal_rows, closer_ref, cells, e_time, cost)
    rules = dict(gm.rules)
    for k in bounds:
        if k not in THRESHOLD_GENES:
            rules[k] = (k, None, "copy")
    return ACT.GeneMap(gm.refs, rules, gm.note + "; coefficient genes copied (FULL parametric layer)")


class PolicyFull(C3.PolicyC3):
    """C3's decision flow with the two TP/SL families and the two volume families. ``scored(cx)`` returns
    (action, tp_mult, sl_mult, volume v, score); ``__call__`` wraps it in the best-of-day gate that forwards v."""

    def __init__(self, table: dict, strategy: StrategyFull, cells=C3.CELLS, e_time=(0.0, 0.0), cost: float = 0.0, scfg=None):
        super().__init__(table, strategy.base, cells, e_time, cost, strategy.observe_bars)
        self.full, self.scfg = strategy, scfg

    def _probs(self, r, si):
        p3 = self._v("p3", r, si)
        return float(p3[0]), float(p3[1]), float(p3[2])

    def levels(self, r, si, bar, env):
        """(tp_mult, sl_mult, EV) of side si at row r under the strategy's TP/SL family; (nan, nan, nan) when no level is allowed."""
        cap, lo = self._cap(r)
        ptp, psl, pt = self._probs(r, si)
        if self.full.tp_family == "cell":
            ci, ev = C3.best_cell_ev(self._v("net", r, si), self._v("p3", r, si), self.cells, self.e_time[si], self.cost, cap=cap, lo=lo)
            if ci < 0 or not np.isfinite(ev):
                return float("nan"), float("nan"), float("nan")
            return float(self.cells[ci][0]), float(self.cells[ci][1]), float(ev)
        s = SIDES[si]
        c = self.full.coef
        tp = c[f"tp_a_{s}"] + c[f"tp_b_{s}"] * ptp
        sl = c[f"sl_c_{s}"] + c[f"sl_e_{s}"] * psl
        tb, sb = (self.scfg.tp_bounds, self.scfg.sl_bounds) if self.scfg is not None else ((0.25, 3.0), (0.5, 3.0))
        hi = cap if np.isfinite(cap) else max(tb[1], sb[1])
        lo_ = lo if np.isfinite(lo) else 0.0
        tp = float(min(max(tp, tb[0], lo_), tb[1], hi))
        sl = float(min(max(sl, sb[0], lo_), sb[1], hi))
        if tp < sl and float(self._v("net", r, si).max()) <= self.cost + C3.RR_MARGIN_SIGMA:   # the same RR rule as the cell family
            return float("nan"), float("nan"), float("nan")
        ev = ptp * tp - psl * sl + pt * self.e_time[si] - self.cost
        return tp, sl, float(ev)

    def volume(self, r, si, sl_mult, bar, env) -> float:
        ptp, psl, _ = self._probs(r, si)
        f = self.full
        if f.vol_family == "risk":
            if env is None or self.scfg is None:
                return 1.0
            b = int(bar)
            price = float(env.c[b])
            sig = float(env.decision_sigma(b)) if hasattr(env, "decision_sigma") else float(env.sigma[b + 1])
            full_u = self.scfg.sizing.units_max(env.balance, price)
            if full_u <= 0 or not np.isfinite(sig) or sig <= 0:
                return 0.0
            ru = XV.risk_units(env.balance, price, sl_mult * sig, f.risk_pct, self.scfg.sizing) or 0
            return float(min(1.0, ru / full_u))
        s = SIDES[si]
        c = f.coef
        pn = float(self._v("pn", r)) if f.base.nop_veto is not None or True else 0.0
        v = c[f"v_alpha_{s}"] + c[f"v_beta_{s}"] * ptp - c[f"v_gamma_{s}"] * pn - c[f"v_delta_{s}"] * psl
        return float(min(max(v, 0.0), 1.0))

    def open_decision_full(self, r, bar, env):
        s = self.s
        if s.nop_veto is not None and float(self._v("pn", r)) > s.nop_veto:
            return None
        best = None
        for si, d in ((0, E.LONG), (1, E.SHORT)):
            tp, sl, ev = self.levels(r, si, bar, env)
            if not np.isfinite(ev):
                continue
            mar = s.open_margin_l if si == 0 else s.open_margin_s
            if ev > mar and (best is None or ev > best[0]):
                v = self.volume(r, si, sl, bar, env)
                if v < max(self.full.v_min, 1e-9):
                    continue                                     # v below the declared minimum: no order
                best = (ev, d, tp, sl, v)
        return best

    def open_decision(self, r, with_score: bool = False):
        o = self.open_decision_full(r, None, None)
        if o is None:
            return None
        return (o[1], o[2], o[3], o[0]) if with_score else (o[1], o[2], o[3])

    def scored_full(self, cx):
        NEG = float("-inf")

        def f(bar, r, st, env):
            pos = int(st[0])
            if pos != 0:
                si = 0 if pos > 0 else 1
                if not self.close_now(r, si, bar, env.pos):
                    return E.HOLD, 1.5, 2.0, 1.0, NEG
                o = self.open_decision_full(r, bar, env)
                if o is not None and self.s.allow_reverse and o[1] != (E.LONG if pos > 0 else E.SHORT):
                    return o[1], float(o[2]), float(o[3]), float(o[4]), float(o[0])
                return E.CLOSE, 1.5, 2.0, 1.0, NEG
            o = self.open_decision_full(r, bar, env)
            return (E.HOLD, 1.5, 2.0, 1.0, NEG) if o is None else (o[1], float(o[2]), float(o[3]), float(o[4]), float(o[0]))
        return f

    def __call__(self, cx):
        self.last_gate = BestOfDayGateV(self.scored_full(cx), observe_bars=self.observe)
        return self.last_gate


class BestOfDayGateV:
    """exec_variants.BestOfDayGate for a scored policy that also returns the volume: (action, tp, sl, v, score) -> (action, tp, sl, v)."""

    def __init__(self, act, observe_bars: int = 0):
        self.inner = XV.BestOfDayGate(lambda bar, r, st, env: None, observe_bars=observe_bars)
        self.act, self.observe_bars = act, int(observe_bars)
        self.day, self.seen, self.best = None, 0, -math.inf
        self.stats = self.inner.stats

    def __call__(self, bar, r, st, env):
        action, tp, sl, v, score = self.act(bar, r, st, env)
        if int(st[0]) != 0:
            return action, tp, sl, v
        k = min(bar + 1, len(env.v) - 1)
        day = env.business_day(k)
        if day != self.day:
            self.day, self.seen, self.best = day, 0, -math.inf
            self.stats["days"] += 1
        ok, _ = env.entry_gate(bar)
        if not ok or action not in (1, 2):
            return (0 if not ok else action), tp, sl, v
        self.seen += 1
        prev_best = self.best
        self.best = max(self.best, float(score))
        if self.seen <= self.observe_bars:
            self.stats["deferred_observing"] += 1
            return 0, tp, sl, v
        if score < prev_best:
            self.stats["deferred_below_best"] += 1
            return 0, tp, sl, v
        self.stats["placed"] += 1
        return action, tp, sl, v


def rollout_with_volume(cx, period, act, cfg):
    """``action_contract.rollout_with_trades`` for a policy that returns (action, tp, sl, v): the engine's
    ``percentual_order_size`` is set to f * v for that decision (v = 1 -> the full rel_volume of the current balance)."""
    a, b = cx.ds.meta["periods"][period] if isinstance(period, str) else period
    env = E.BarEnv(cx.ds.bars, cx.ds.cal, cx.ds.sigma, cx.ds.sigma_rule, a, b, cfg)
    f_full = float(cfg.percentual_order_size)
    for bar in range(env.first, env.last):
        r = cx.rob[bar]
        st = env.state(bar)
        if r >= 0 and cx.obs_ok[r]:
            action, tp, sl, v = act(bar, r, st, env)
            if action in (E.LONG, E.SHORT):
                v = float(min(max(v, 0.0), 1.0))
                if v <= 0.0:
                    action = E.HOLD
                else:
                    # the order's fraction stays in force until the next entry: the trade record (written at the close)
                    # then carries the fraction the order was sized with, as the correctness check re-derives it
                    env.cfg = dataclasses.replace(cfg, percentual_order_size=f_full * v)
        else:
            action, tp, sl = E.HOLD, 1.5, 2.0
        env.step(bar, action, tp, sl)
    m = env.result()
    return m, list(env.trades)
