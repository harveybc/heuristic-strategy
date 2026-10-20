"""Bar-by-bar executable trading environment with the owner's business rules (E1, E0 replays).

Decision protocol (label != fill): the agent decides after the COMPLETED bar b (its observation is the
row available at start(b) + 1 h) and every fill happens at the OPEN of bar b + 1 or later.

Actions: HOLD, LONG, SHORT, CLOSE, each with a take-profit and a stop-loss multiple of the causal 24 h
volatility sigma known at the decision (``zigzag.causal_sigma`` at bar b + 1, which only uses bars <= b).
Owner hard rules enforced here for EVERY agent (expert, student, SAC replays):

* every entry carries an effective SL and TP computed from the reference close[b]; levels farther than
  ``OrderRules.effectiveness_c`` x the causal block range sigma or nearer than the broker minimum are either
  clipped into the effective band (``level_policy="clip"``, students) or the entry is refused
  (``level_policy="refuse"``, the expert replay, as in the block simulator);
* units = floor(percentual_order_size x rel_volume x balance x leverage / price), integer, OANDA minimum 1,
  refused above the free margin (``execution.Sizing``; rel_volume 0.10, leverage 100, size 1.0);
* no entry before the session's operational start (12 h after a reopen) nor at/after the entry cutoff; any
  open position is closed at the close of the last bar before the forced flat (12 h before the closure);
* at most ``max_entries_per_day`` (4) ENTRIES per business day (rollover to rollover, as the block
  simulator); the fifth is refused; the weekly rule (>= 3 orders per ISO week on average) is a validity
  flag of the run, never a forced entry;
* entries and closes are counted separately; LONG while short (or SHORT while long) is close-and-reverse:
  one close plus one entry, both costed, the entry subject to every entry rule; a repeated decision in the
  current direction is ignored (declared IGNORE);
* intrabar: open first (a gap beyond a level fills at the open), stop before take-profit when both are
  inside one bar, TP needs ``limit_through`` (the block simulator's ``_protect``);
* costs: ``RealisticProfile`` (hour-dependent spread, market/stop slippage, commission per side, swap per
  rollover by side with the triple rollover), all ASSUMED_NOT_BROKER_VERIFIED.

Accounting reuses ``execution._account`` so net profit, drawdown, Sharpe (daily, sqrt(365), calendar days
incl. closures) and Sortino have exactly the definitions of every other short-horizon result.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import execution as X
from ..calendar import HOUR
from ..zigzag_opt import executable_profile

HOLD, LONG, SHORT, CLOSE = 0, 1, 2, 3
ACTION_NAMES = ("HOLD", "LONG", "SHORT", "CLOSE")
H_NS = HOUR.value
STATE_NAMES = ("position", "unrealized_sigma", "holding_days", "sl_distance_sigma", "tp_distance_sigma",
               "hours_since_operational_start_days", "hours_to_forced_flat_days", "entries_today_share",
               "can_enter")


@dataclass(frozen=True)
class EnvConfig:
    capital: float = 10_000.0
    max_entries_per_day: int = 4
    min_orders_per_week: float = 3.0
    tp_bounds: tuple = (0.25, 3.0)          # sigma multiples
    sl_bounds: tuple = (0.5, 3.0)
    level_policy: str = "clip"               # "clip" (students) or "refuse" (expert replay)
    percentual_order_size: float = 1.0
    sizing: X.Sizing = field(default_factory=lambda: X.Sizing(rel_volume=0.10, leverage=100.0))
    profile: X.RealisticProfile = field(default_factory=executable_profile)
    rules: X.OrderRules = field(default_factory=X.OrderRules)
    same_direction: str = "IGNORE"
    reverse: str = "CLOSE_AND_OPEN"


@dataclass
class Position:
    direction: int
    units: int
    entry_bar: int
    entry_price: float
    sl: float
    tp: float
    sigma: float
    seg: int
    entry_cost: dict


class BarEnv:
    """One pass over the bars with start in ``[start, end)``; ``step(b, ...)`` decides after bar b."""

    def __init__(self, bars: pd.DataFrame, cal, sigma: np.ndarray, sigma_rule: np.ndarray, start, end,
                 cfg: EnvConfig = EnvConfig()):
        self.bars, self.cal, self.cfg = bars, cal, cfg
        self.v = bars.index.as_unit("ns").asi8
        self.o, self.h, self.l, self.c = (bars[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
        self.sigma, self.sigma_rule = np.asarray(sigma, float), np.asarray(sigma_rule, float)
        self.seg_of = cal.assign(bars.index)
        self.op = np.array([s.operational_start.value for s in cal.segments], np.int64)
        self.ff = np.array([s.force_flat.value for s in cal.segments], np.int64)
        self.ec = np.array([s.entry_cutoff.value for s in cal.segments], np.int64)
        self.mo = np.array([s.market_open.value for s in cal.segments], np.int64)
        a, b = (pd.Timestamp(x) for x in (start, end))
        a = a.tz_localize("UTC") if a.tzinfo is None else a
        b = b.tz_localize("UTC") if b.tzinfo is None else b
        self.split = (a, b)
        sel = np.flatnonzero((self.v >= a.value) & (self.v < b.value))
        if not len(sel):
            raise ValueError("no bars in the window")
        self.first, self.last = int(sel[0]), int(sel[-1])
        self.reset()

    # ------------------------------------------------------------------ bookkeeping
    def reset(self):
        self.pos: Position | None = None
        self.balance = self.cfg.capital
        self.trades, self.log = [], []
        self.entries_day: dict = {}
        self.counts = {k: 0 for k in ("entries", "closes", "close_TP", "close_SL", "close_FORCED_FLAT",
                                      "close_AGENT_CLOSE", "close_REVERSAL", "refused_BEFORE_OPERATIONAL_START",
                                      "refused_ENTRY_CUTOFF", "refused_DAILY_CAP", "refused_UNITS",
                                      "refused_LEVELS", "refused_GAP_BEYOND_LEVEL", "refused_NO_SIGMA",
                                      "refused_SESSION_END", "refused_RUIN", "ignored_same_direction",
                                      "levels_clipped", "decisions")}
        self.decisions = []
        return self.first

    def business_day(self, k: int):
        si = int(self.seg_of[k])
        return si, int((self.v[k] - self.mo[si]) // (24 * H_NS))

    def entry_gate(self, b: int):
        """(ok, reason) for an entry decided after bar b (fill at the open of b + 1); time/cap rules only."""
        k = b + 1
        if k > self.last or k >= len(self.v):
            return False, "SESSION_END"
        si = int(self.seg_of[b])
        if int(self.seg_of[k]) != si:
            return False, "SESSION_END"
        if self.v[k] < self.op[si]:
            return False, "BEFORE_OPERATIONAL_START"
        if self.v[k] >= self.ec[si] or self.v[k] + H_NS >= self.ff[si]:
            return False, "ENTRY_CUTOFF"
        if self.entries_day.get(self.business_day(k), 0) >= self.cfg.max_entries_per_day:
            return False, "DAILY_CAP"
        if self.balance <= 0:
            return False, "RUIN"
        s = self.sigma[k]
        if not (np.isfinite(s) and s > 0 and np.isfinite(self.sigma_rule[k]) and self.sigma_rule[k] > 0):
            return False, "NO_SIGMA"
        return True, "OK"

    def decision_sigma(self, b: int) -> float:
        k = min(b + 1, len(self.sigma) - 1)
        return float(self.sigma[k])

    def state(self, b: int) -> np.ndarray:
        """Agent state known after bar b (no future): see STATE_NAMES."""
        si = int(self.seg_of[b])
        t_end = self.v[b] + H_NS
        since = max(0.0, (t_end - self.op[si]) / H_NS) / 24.0
        to_ff = max(0.0, (self.ff[si] - t_end) / H_NS) / 24.0
        k = min(b + 1, len(self.v) - 1)
        day = self.business_day(k) if int(self.seg_of[k]) == si else (si, -1)
        share = self.entries_day.get(day, 0) / self.cfg.max_entries_per_day
        ok, _ = self.entry_gate(b)
        s = self.decision_sigma(b)
        s = s if np.isfinite(s) and s > 0 else 1e-4
        if self.pos is None:
            return np.array([0, 0, 0, 0, 0, min(since, 5.0), min(to_ff, 5.0), share, float(ok)], np.float32)
        p = self.pos
        unreal = p.direction * (self.c[b] - p.entry_price) / p.sigma
        hold = (b - p.entry_bar + 1) / 24.0
        return np.array([p.direction, np.clip(unreal, -10, 10), min(hold, 5.0), abs(self.c[b] - p.sl) / s,
                         abs(p.tp - self.c[b]) / s, min(since, 5.0), min(to_ff, 5.0), share, float(ok)], np.float32)

    # ------------------------------------------------------------------ levels and costs
    def _levels(self, b: int, d: int, tp_mult: float, sl_mult: float):
        """(tp, sl, clipped) from the reference close[b], or None when refused."""
        cfg, r = self.cfg, float(self.c[b])
        s, sr = self.decision_sigma(b), float(self.sigma_rule[b + 1])
        tp_m = float(np.clip(tp_mult, *cfg.tp_bounds))
        sl_m = float(np.clip(sl_mult, *cfg.sl_bounds))
        dist_tp, dist_sl = tp_m * s, sl_m * s
        lo, hi = cfg.rules.min_distance * (1 + 1e-9), cfg.rules.effectiveness_c * sr * (1 - 1e-9)   # float-safe band
        clipped = False
        if cfg.level_policy == "clip":
            if hi <= lo:
                return None
            ntp, nsl = min(max(dist_tp, lo), hi), min(max(dist_sl, lo), hi)
            clipped = (ntp != dist_tp) or (nsl != dist_sl)
            dist_tp, dist_sl = ntp, nsl
        tp, sl = r + d * dist_tp, r - d * dist_sl
        order = X.BlockOrder(d, "MARKET", None, tp, sl, 1, cfg.percentual_order_size)
        try:
            X.check_order(order, r, sr, cfg.rules, require_units=True)
        except X.OrderRuleError:
            return None
        return tp, sl, clipped

    def _fill_cost(self, units, k, kind):
        prof = self.cfg.profile
        si = int(self.seg_of[k])
        slip = {"MARKET": prof.slippage_market, "SL": prof.slippage_stop, "TP": 0.0}[kind]
        return {"spread": units * prof.spread_at(int(self.v[k]), int(self.mo[si])) / 2, "slippage": units * slip,
                "commission": units / prof.lot_units * prof.commission_per_lot_side}

    def _close(self, k: int, price: float, at_open: bool, reason: str, ambiguous: bool = False):
        p, prof = self.pos, self.cfg.profile
        kind = {"SL": "SL", "TP": "TP"}.get(reason, "MARKET")
        fc_out = self._fill_cost(p.units, k, kind)
        entry_ns = int(self.v[p.entry_bar])
        exit_ns = int(self.v[k]) + (0 if at_open else H_NS)
        lots = p.units / prof.lot_units
        rate = prof.swap_long_per_lot if p.direction > 0 else prof.swap_short_per_lot
        fin, rolls = 0.0, 0
        for i, r in enumerate(self.cal.rollover_ns(p.seg), start=1):
            if entry_ns < r <= exit_ns:
                rolls += 1
                fin -= rate * lots * (3 if i == prof.triple_swap_rollover_index else 1)
        gross = p.direction * p.units * (price - p.entry_price)
        entry_cost = sum(p.entry_cost.values())
        exit_cost = sum(fc_out.values()) + fin
        rec = {"block": -1, "segment": p.seg, "direction": p.direction, "units": int(p.units),
               "entry_time": pd.Timestamp(entry_ns, tz="UTC"), "exit_time": pd.Timestamp(exit_ns, tz="UTC"),
               "entry_bar_abs": int(p.entry_bar), "exit_bar_abs": int(k), "entry_price": p.entry_price,
               "exit_price": float(price),
               "exit_reason": {"TP": "TP", "SL": "SL", "FORCED_FLAT": "TIME"}.get(reason, "EARLY"),
               "close_reason": reason, "ambiguous": bool(ambiguous), "gross_usd": gross,
               "spread_usd": p.entry_cost["spread"] + fc_out["spread"],
               "slippage_usd": p.entry_cost["slippage"] + fc_out["slippage"],
               "commission_usd": p.entry_cost["commission"] + fc_out["commission"], "financing_usd": fin,
               "rollovers": rolls, "entry_cost_usd": entry_cost, "exit_cost_usd": exit_cost,
               "net_usd": gross - entry_cost - exit_cost, "notional_usd": p.units * (p.entry_price + price),
               "stop_loss": p.sl, "take_profit": p.tp, "margin_usd": p.units * p.entry_price / self.cfg.sizing.leverage,
               "percentual_order_size": self.cfg.percentual_order_size, "rel_volume": self.cfg.sizing.rel_volume}
        self.trades.append(rec)
        self.balance += rec["net_usd"]
        self.counts["closes"] += 1
        self.counts["close_" + reason] += 1
        self.pos = None

    # ------------------------------------------------------------------ the step
    def step(self, b: int, action: int, tp_mult: float = 1.5, sl_mult: float = 2.0) -> dict:
        """Decide after bar b, then simulate bar b + 1. Returns what happened (entry/close/refusal)."""
        if not (self.first <= b < self.last):
            raise IndexError("decision outside the window")
        k = b + 1
        self.counts["decisions"] += 1
        info = {"b": b, "action": int(action), "entry": False, "close": None, "refused": None}
        want_close, want_entry = False, 0
        p = self.pos
        if action == CLOSE and p is not None:
            want_close = True
        elif action in (LONG, SHORT):
            d = 1 if action == LONG else -1
            if p is not None and p.direction == d:
                self.counts["ignored_same_direction"] += 1
            else:
                want_entry = d
                want_close = p is not None
        # 1) the close fills at the open of b + 1
        if want_close:
            self._close(k, float(self.o[k]), True, "REVERSAL" if want_entry else "AGENT_CLOSE")
            info["close"] = "REVERSAL" if want_entry else "AGENT_CLOSE"
        # 2) the entry fills at the open of b + 1 when every rule allows it
        if want_entry:
            ok, why = self.entry_gate(b)
            lv = self._levels(b, want_entry, tp_mult, sl_mult) if ok else None
            if ok and lv is None:
                ok, why = False, "LEVELS"
            units = None
            if ok:
                units = self.cfg.sizing.order_units(self.balance, float(self.c[b]), self.cfg.percentual_order_size)
                if units is None:
                    ok, why = False, "UNITS"
            if ok:
                tp, sl, clipped = lv
                o = float(self.o[k])
                if not ((sl < o < tp) if want_entry > 0 else (tp < o < sl)):
                    ok, why = False, "GAP_BEYOND_LEVEL"
            if not ok:
                self.counts["refused_" + why] += 1
                info["refused"] = why
            else:
                self.counts["levels_clipped"] += int(clipped)
                self.pos = Position(want_entry, int(units), k, float(self.o[k]), sl, tp, self.decision_sigma(b),
                                    int(self.seg_of[k]), self._fill_cost(units, k, "MARKET"))
                day = self.business_day(k)
                self.entries_day[day] = self.entries_day.get(day, 0) + 1
                self.counts["entries"] += 1
                info["entry"] = True
        # 3) protections on bar b + 1 (open gap first, stop before take-profit)
        if self.pos is not None:
            p = self.pos
            hit = X._protect(p.direction, p.tp, p.sl, self.o[k], self.h[k], self.l[k], self.cfg.profile.limit_through)
            if hit is not None:
                reason, price, at_open, amb = hit
                self._close(k, float(price), bool(at_open), reason, amb)
                info["close"] = reason
        # 4) forced flat: close at the close of the last bar before the forced flat (or a session end)
        if self.pos is not None:
            si = int(self.seg_of[k])
            nxt_other = (k + 1 >= len(self.v)) or int(self.seg_of[k + 1]) != si
            if self.v[k] + H_NS >= self.ff[si] or nxt_other or k >= self.last:
                self._close(k, float(self.c[k]), False, "FORCED_FLAT")
                info["close"] = "FORCED_FLAT"
        self.decisions.append(b)
        return info

    def flatten_at_end(self):
        if self.pos is not None:
            self._close(self.last, float(self.c[self.last]), False, "FORCED_FLAT")

    # ------------------------------------------------------------------ results
    def result(self) -> dict:
        self.flatten_at_end()
        res = X._account(self.bars, self.trades, self.split, self.cfg.capital, self.decisions,
                         self.counts["refused_GAP_BEYOND_LEVEL"], orders_submitted=self.counts["entries"],
                         margin_closeout=self.cfg.sizing.margin_closeout)
        m = dict(res.metrics)
        m.update({f"n_{k}": int(v) for k, v in self.counts.items()})
        dec_v = self.v[np.asarray(self.decisions, int) + 1] if self.decisions else np.array([], np.int64)
        weeks = {tuple(pd.Timestamp(int(t), tz="UTC").isocalendar()[:2]) for t in dec_v}
        days = {}
        for t in self.trades:
            d = self.business_day(t["entry_bar_abs"])
            days[d] = days.get(d, 0) + 1
        m["weeks_with_decisions"] = len(weeks)
        m["orders_per_week"] = self.counts["entries"] / len(weeks) if weeks else 0.0
        m["max_entries_per_day"] = max(days.values(), default=0)
        m["valid_weekly_rule"] = bool(m["orders_per_week"] >= self.cfg.min_orders_per_week)
        m["valid"] = bool(m["valid_weekly_rule"] and m["max_entries_per_day"] <= self.cfg.max_entries_per_day
                          and not m["ruined"] and math.isfinite(m["sharpe_daily_sqrt365"]))
        m["equity"] = res.equity
        m["daily_equity"] = res.daily_equity
        return m


def summary(m: dict) -> dict:
    """Report-ready numbers (no series)."""
    keys = ("net_profit_usd", "sharpe_daily_sqrt365", "sortino_daily_sqrt365", "max_drawdown_pct", "max_drawdown_usd",
            "trades", "win_rate", "profit_factor", "orders_per_week", "max_entries_per_day", "valid", "valid_weekly_rule",
            "gross_usd", "cost_spread_usd", "cost_slippage_usd", "cost_commission_usd", "cost_financing_usd",
            "exposure_bar_share", "ruined")
    out = {k: m[k] for k in keys if k in m}
    out.update({k: v for k, v in m.items() if k.startswith("n_")})
    return out
