"""Explicit hourly-OHLC event simulator, costs and business metrics.

Intrabar rule (contract section 6): open first (gaps fill at the open), then
stop before take-profit when both are inside one bar; on an intrabar limit-fill
bar only the stop may trigger. Every cost is a cash debit charged once per fill.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd

from .blocks import Blocks
from .calendar import Calendar, HOUR


@dataclass(frozen=True)
class Costs:
    """All values ASSUMED_NOT_BROKER_VERIFIED (instrument config i6dmo-eurusd-costs v1)."""
    spread_price: float = 0.00020            # full spread; half charged per fill
    slippage_price: float = 0.00010          # per fill
    commission_per_lot_round_trip: float = 7.0
    financing_per_lot_per_rollover: float = 10.0  # debited in both directions
    lot_units: float = 100_000.0

    def scaled(self, m: float) -> "Costs":
        return replace(self, spread_price=self.spread_price * m,
                       slippage_price=self.slippage_price * m,
                       commission_per_lot_round_trip=self.commission_per_lot_round_trip * m,
                       financing_per_lot_per_rollover=self.financing_per_lot_per_rollover * m)

    def fill_costs(self, units: float) -> dict:
        u = abs(units)
        return {"spread": u * self.spread_price / 2, "slippage": u * self.slippage_price,
                "commission": u / self.lot_units * self.commission_per_lot_round_trip / 2}

    def financing(self, units: float, rollovers: int) -> float:
        return abs(units) / self.lot_units * self.financing_per_lot_per_rollover * rollovers

    def round_trip_price(self) -> float:
        return self.spread_price + 2 * self.slippage_price + \
            self.commission_per_lot_round_trip / self.lot_units


_DEFAULT_SPREAD_SCHEDULE = (  # (from, to) hours since the daily rollover -> full spread, price units
    (0, 1, 3.0e-4),    # first hour after rollover: liquidity gap
    (1, 3, 1.0e-4),
    (3, 9, 0.6e-4),    # Asian session on this bar clock
    (9, 21, 0.3e-4),   # London / New York core hours (raw ECN spread)
    (21, 23, 0.6e-4),
    (23, 24, 1.5e-4),  # last hour before rollover: widening
)


@dataclass(frozen=True)
class RealisticProfile:
    """Realistic execution and costs; every value ASSUMED_NOT_BROKER_VERIFIED.

    Spread depends on the hour since the configured daily rollover (rollover
    instants are segment open + 24 k on the calendar; no day name is read);
    the first hour after a segment (weekly) open uses ``weekly_open_spread``.
    Half the spread is charged on each fill, slippage on market and stop fills
    only, commission per side, and swap per rollover held by side (negative
    rate = cost) with a triple charge on the configured rollover index of the
    segment (third rollover after the weekly open = the Wednesday convention).
    """
    spread_schedule: tuple = _DEFAULT_SPREAD_SCHEDULE
    weekly_open_spread: float = 3.0e-4
    slippage_market: float = 0.2e-4
    slippage_stop: float = 0.5e-4
    limit_through: float = 1.0e-4
    entry_delay_bars: int = 1
    commission_per_lot_side: float = 3.5
    swap_long_per_lot: float = -7.0
    swap_short_per_lot: float = 2.5
    triple_swap_rollover_index: int = 3
    lot_units: float = 100_000.0

    def scaled(self, m: float) -> "RealisticProfile":
        return replace(self, spread_schedule=tuple((a, b, x * m) for a, b, x in self.spread_schedule),
                       weekly_open_spread=self.weekly_open_spread * m, slippage_market=self.slippage_market * m,
                       slippage_stop=self.slippage_stop * m, commission_per_lot_side=self.commission_per_lot_side * m,
                       swap_long_per_lot=self.swap_long_per_lot * m, swap_short_per_lot=self.swap_short_per_lot * m)

    def spread_for_hour(self, hour: float) -> float:
        for a, b, x in self.spread_schedule:
            if a <= hour < b:
                return x
        raise ValueError(f"hour {hour} outside the schedule")

    def spread_at(self, t_ns: int, open_ns: int) -> float:
        since = (t_ns - open_ns) / HOUR.value
        if since < 1.0:
            return self.weekly_open_spread
        return self.spread_for_hour(since % 24)

    def round_trip_price(self) -> float:
        """Edge-filter estimate: time-weighted mean spread + two market slippages + commission."""
        mean_spread = sum((b - a) * x for a, b, x in self.spread_schedule) / 24.0
        return mean_spread + 2 * self.slippage_market + 2 * self.commission_per_lot_side / self.lot_units


@dataclass(frozen=True)
class BlockOrder:
    direction: int            # +1 long, -1 short
    entry_type: str           # "MARKET" (first block open) or "LIMIT"
    entry_price: Optional[float]
    take_profit: float
    stop_loss: float
    units: float


@dataclass
class Fill:
    direction: int
    units: float
    entry_bar: int = -1
    entry_price: float = math.nan
    exit_bar: int = -1
    exit_price: float = math.nan
    exit_reason: str = ""
    exit_at_open: bool = False
    ambiguous: bool = False
    refused: bool = False
    entry_kind: str = ""


def _protect(d, tp, sl, o, h, l, through=0.0):
    """Full bar after the fill: (reason, price, at_open, ambiguous) or None.

    ``through``: the TP limit fills intrabar only if price trades beyond it by this much.
    """
    if d > 0:
        if o <= sl:
            return "SL", o, True, False
        if o >= tp:
            return "TP", o, True, False
        hit_sl, hit_tp = l <= sl, h >= tp + through
    else:
        if o >= sl:
            return "SL", o, True, False
        if o <= tp:
            return "TP", o, True, False
        hit_sl, hit_tp = h >= sl, l <= tp - through
    if hit_sl:
        return "SL", sl, False, bool(hit_tp)
    if hit_tp:
        return "TP", tp, False, False
    return None


def simulate_block(order: BlockOrder, o, h, l, c, entry_bars: Optional[int] = None,
                   limit_through: float = 0.0, entry_delay: int = 0) -> Optional[Fill]:
    """Simulate one block. ``entry_bars``: how many leading bars may host an entry.

    ``entry_delay``: decision latency in bars (the order exists from that bar on).
    ``limit_through``: limits (entry and TP) fill intrabar only when price trades
    beyond them by this much; an open already beyond the limit fills at the open.
    """
    n = len(o)
    entry_bars = n if entry_bars is None else min(entry_bars, n)
    d, tp, sl = order.direction, order.take_profit, order.stop_loss
    f = Fill(direction=d, units=order.units, entry_kind=order.entry_type)
    q = entry_delay
    if order.entry_type == "MARKET":
        if entry_bars < q + 1:
            return None
        if not (sl < o[q] < tp if d > 0 else tp < o[q] < sl):
            f.refused, f.units = True, 0
            return f
        f.entry_bar, f.entry_price, start = q, float(o[q]), q
    elif order.entry_type == "LIMIT":
        e = order.entry_price
        start = None
        for k in range(q, entry_bars):
            crossed_open = o[k] <= e if d > 0 else o[k] >= e
            touched = l[k] <= e - limit_through if d > 0 else h[k] >= e + limit_through
            if crossed_open:
                f.entry_bar, f.entry_price, start = k, float(o[k]), k
                break
            if touched:
                f.entry_bar, f.entry_price = k, float(e)
                stopped = l[k] <= sl if d > 0 else h[k] >= sl
                if stopped:
                    f.exit_bar, f.exit_price, f.exit_reason = k, float(sl), "SL"
                    return f
                start = k + 1
                break
        if start is None:
            return None
    else:
        raise ValueError(f"unknown entry type {order.entry_type}")
    for k in range(start, n):
        hit = _protect(d, tp, sl, o[k], h[k], l[k], limit_through)
        if hit:
            f.exit_bar = k
            f.exit_reason, price, f.exit_at_open, f.ambiguous = hit
            f.exit_price = float(price)
            return f
    f.exit_bar, f.exit_price, f.exit_reason = n - 1, float(c[n - 1]), "TIME"
    return f


@dataclass
class Result:
    trades: list
    metrics: dict
    equity: pd.Series          # marked-to-market at each split bar close
    daily_equity: pd.Series    # calendar days incl. closures (forward-filled)
    evaluated_blocks: list
    final_position_units: float
    _intervals: list

    def position_at(self, t) -> float:
        t = pd.Timestamp(t)
        return sum(u for a, b, u in self._intervals if a <= t < b)


def _realistic_costs(profile, v, seg_id, lo, fill, calendar, entry_ns, exit_ns):
    u, lots = abs(fill.units), abs(fill.units) / profile.lot_units
    open_ns = calendar.segments[seg_id].market_open.value
    entry_label, exit_label = int(v[lo + fill.entry_bar]), int(v[lo + fill.exit_bar])
    slip_in = profile.slippage_market if fill.entry_kind == "MARKET" else 0.0
    slip_out = {"SL": profile.slippage_stop, "TIME": profile.slippage_market}.get(fill.exit_reason, 0.0)
    fc_in = {"spread": u * profile.spread_at(entry_label, open_ns) / 2, "slippage": u * slip_in,
             "commission": lots * profile.commission_per_lot_side}
    fc_out = {"spread": u * profile.spread_at(exit_label, open_ns) / 2, "slippage": u * slip_out,
              "commission": lots * profile.commission_per_lot_side}
    rate = profile.swap_long_per_lot if fill.direction > 0 else profile.swap_short_per_lot
    fin, rolls = 0.0, 0
    for i, r in enumerate(calendar.rollover_ns(seg_id), start=1):
        if entry_ns < r <= exit_ns:
            rolls += 1
            fin -= rate * lots * (3 if i == profile.triple_swap_rollover_index else 1)
    return fc_in, fc_out, fin, rolls


def _trade_record(v, seg_id, lo, fill, costs, calendar, block_id, profile=None):
    """``v``: bar labels as int64 nanoseconds."""
    d, u = fill.direction, fill.units
    entry_ns = int(v[lo + fill.entry_bar])
    exit_ns = int(v[lo + fill.exit_bar]) + (0 if fill.exit_at_open else HOUR.value)
    entry_time, exit_time = pd.Timestamp(entry_ns, tz="UTC"), pd.Timestamp(exit_ns, tz="UTC")
    if profile is None:
        rolls = sum(1 for r in calendar.rollover_ns(seg_id) if entry_ns < r <= exit_ns)
        fc_in, fc_out = costs.fill_costs(u), costs.fill_costs(u)
        fin = costs.financing(u, rolls)
    else:
        fc_in, fc_out, fin, rolls = _realistic_costs(profile, v, seg_id, lo, fill, calendar, entry_ns, exit_ns)
    gross = d * u * (fill.exit_price - fill.entry_price)
    entry_cost = sum(fc_in.values())
    exit_cost = sum(fc_out.values()) + fin
    return {"block": block_id, "segment": int(seg_id), "direction": d, "units": u,
            "entry_time": entry_time, "exit_time": exit_time,
            "entry_bar_abs": int(lo + fill.entry_bar), "exit_bar_abs": int(lo + fill.exit_bar),
            "entry_price": fill.entry_price, "exit_price": fill.exit_price,
            "exit_reason": fill.exit_reason, "ambiguous": fill.ambiguous,
            "gross_usd": gross, "spread_usd": fc_in["spread"] + fc_out["spread"],
            "slippage_usd": fc_in["slippage"] + fc_out["slippage"],
            "commission_usd": fc_in["commission"] + fc_out["commission"],
            "financing_usd": fin, "rollovers": rolls,
            "entry_cost_usd": entry_cost, "exit_cost_usd": exit_cost,
            "net_usd": gross - entry_cost - exit_cost,
            "notional_usd": u * (fill.entry_price + fill.exit_price)}


def _entry_bars(v, lo, hi, cutoff):
    """Leading block bars that end at or before the entry cutoff (``v`` in ns)."""
    return int(np.searchsorted(v[lo:hi], (cutoff - HOUR).value, side="right"))


def _account(bars, trades, split, initial_capital, evaluated, refused, orders_submitted=None, unfilled=0):
    ts = bars.index
    a, b = (pd.Timestamp(x) for x in split)
    sel = np.flatnonzero((ts >= a) & (ts < b))
    if not len(sel):
        raise ValueError("split holds no bars")
    first, last = sel[0], sel[-1]
    n = last - first + 1
    close = bars.CLOSE.to_numpy(float)
    cash = np.zeros(n)
    unreal = np.zeros(n)
    in_pos = np.zeros(n, bool)
    intervals = []
    for t in trades:
        i, j = t["entry_bar_abs"] - first, t["exit_bar_abs"] - first
        cash[i] -= t["entry_cost_usd"]
        cash[j] += t["gross_usd"] - t["exit_cost_usd"]
        if j > i:
            unreal[i:j] = t["direction"] * t["units"] * (close[first + i:first + j] - t["entry_price"])
        in_pos[i:j + 1] = True
        intervals.append((t["entry_time"], t["exit_time"], t["units"]))
    equity = pd.Series(initial_capital + np.cumsum(cash) + unreal, index=ts[first:last + 1])
    day = equity.groupby(equity.index.normalize()).last()
    days = pd.date_range(day.index[0], day.index[-1], freq="D")
    daily = day.reindex(days).ffill()
    eq_d = np.r_[initial_capital, daily.to_numpy()]
    ret = np.diff(eq_d) / eq_d[:-1]
    sd = ret.std(ddof=1) if len(ret) > 1 else 0.0
    down = math.sqrt(np.mean(np.minimum(ret, 0) ** 2)) if len(ret) else 0.0
    curve = np.r_[initial_capital, equity.to_numpy()]
    peak = np.maximum.accumulate(curve)
    dd = peak - curve
    net = [t["net_usd"] for t in trades]
    wins = sum(x for x in net if x > 0)
    losses = -sum(x for x in net if x < 0)
    years = len(days) / 365.0
    notional = sum(t["notional_usd"] for t in trades)
    m = {"trades": len(trades), "refused_entries": refused,
         "net_profit_usd": float(equity.iloc[-1] - initial_capital),
         "end_equity_usd": float(equity.iloc[-1]),
         "max_drawdown_usd": float(dd.max()),
         "max_drawdown_pct": float((dd / peak).max() * 100),
         "sharpe_daily_sqrt365": float(ret.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0,
         "sortino_daily_sqrt365": float(ret.mean() / down * math.sqrt(365)) if down > 0
         else (math.inf if ret.mean() > 0 else 0.0),
         "win_rate": float(np.mean([x > 0 for x in net])) if net else 0.0,
         "profit_factor": float(wins / losses) if losses > 0 else (math.inf if wins > 0 else 0.0),
         "turnover_x_capital": float(notional / initial_capital),
         "turnover_x_capital_per_year": float(notional / initial_capital / years),
         "exposure_bar_share": float(in_pos.mean()),
         "gross_usd": float(sum(t["gross_usd"] for t in trades)),
         "cost_spread_usd": float(sum(t["spread_usd"] for t in trades)),
         "cost_slippage_usd": float(sum(t["slippage_usd"] for t in trades)),
         "cost_commission_usd": float(sum(t["commission_usd"] for t in trades)),
         "cost_financing_usd": float(sum(t["financing_usd"] for t in trades)),
         "ambiguous_stop_first": int(sum(t["ambiguous"] for t in trades)),
         "exit_tp": int(sum(t["exit_reason"] == "TP" for t in trades)),
         "exit_sl": int(sum(t["exit_reason"] == "SL" for t in trades)),
         "exit_time": int(sum(t["exit_reason"] == "TIME" for t in trades)),
         "evaluated_blocks": len(evaluated), "calendar_days": len(days),
         # participation: an unfilled limit or a refused entry is an abstention, not a loss
         "eligible_blocks": len(evaluated),
         "orders_submitted": int(len(trades) + refused + unfilled if orders_submitted is None else orders_submitted),
         "unfilled_orders": int(unfilled),
         "fill_rate": float(len(trades) / len(evaluated)) if len(evaluated) else 0.0,
         "net_per_eligible_block_usd": float((equity.iloc[-1] - initial_capital) / len(evaluated))
         if len(evaluated) else 0.0}
    # Ruin: marked-to-market equity at or below zero. Returns against a non-positive base are
    # meaningless, so a ruined run has no Sharpe or Sortino.
    m["ruined"] = bool(curve.min() <= 0)
    m["ruin_skipped_orders"] = 0
    if m["ruined"]:
        m["sharpe_daily_sqrt365"] = math.nan
        m["sortino_daily_sqrt365"] = math.nan
    recon = m["gross_usd"] - sum(m[f"cost_{c}_usd"] for c in ("spread", "slippage", "commission", "financing"))
    if abs(recon - m["net_profit_usd"]) > 1e-6 * max(1.0, abs(recon)):
        raise AssertionError("cash ledger does not reconcile with trade costs")
    return Result(trades, m, equity, daily, [int(k) for k in evaluated], 0.0, intervals)


def _business_day(blocks, calendar, k):
    seg = calendar.segments[blocks.segment[k]]
    return int(blocks.segment[k]), int((blocks.start[k] - seg.market_open) // pd.Timedelta(hours=24))


def run_backtest(bars: pd.DataFrame, blocks: Blocks, orders, costs, calendar: Calendar,
                 initial_capital: float, split, profile: Optional[RealisticProfile] = None,
                 max_orders_per_day: Optional[int] = None, ruin_floor: Optional[float] = None) -> Result:
    """``max_orders_per_day``: strategy-side cap per business day (rollover to rollover),
    applied at decision time in block order; later orders of a full day are not sent.
    ``ruin_floor``: stop-out; no new order once realized equity is at or below this level."""
    v = bars.index.as_unit("ns").asi8
    arr = [bars[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE")]
    evaluated = list(np.flatnonzero(blocks.in_split(split)))
    trades, refused, submitted, unfilled, capped, ruin_skipped = [], 0, 0, 0, 0, 0
    realized = initial_capital
    per_day, trades_day = {}, {}
    through = profile.limit_through if profile is not None else 0.0
    delay = profile.entry_delay_bars if profile is not None else 0
    for k in evaluated:
        order = orders[k]
        if order is None or order.units <= 0:
            continue
        if ruin_floor is not None and realized <= ruin_floor:
            ruin_skipped += 1
            continue
        day = _business_day(blocks, calendar, k)
        if max_orders_per_day is not None and per_day.get(day, 0) >= max_orders_per_day:
            capped += 1
            continue
        per_day[day] = per_day.get(day, 0) + 1
        submitted += 1
        lo, hi = blocks.bar_lo[k], blocks.bar_hi[k]
        seg = calendar.segments[blocks.segment[k]]
        fill = simulate_block(order, *(x[lo:hi] for x in arr),
                              entry_bars=_entry_bars(v, lo, hi, seg.entry_cutoff),
                              limit_through=through, entry_delay=delay)
        if fill is None:
            unfilled += 1
            continue
        if fill.refused:
            refused += 1
            continue
        trades_day[day] = trades_day.get(day, 0) + 1
        trades.append(_trade_record(v, blocks.segment[k], lo, fill, costs, calendar, int(k), profile))
        realized += trades[-1]["net_usd"]
    res = _account(bars, trades, split, initial_capital, evaluated, refused, submitted, unfilled)
    weeks = {tuple(blocks.start[k].isocalendar()[:2]) for k in evaluated}
    res.metrics.update({"weeks_with_eligible_blocks": len(weeks),
                        "trades_per_week": len(trades) / len(weeks) if weeks else 0.0,
                        "max_orders_per_day": max(per_day.values(), default=0),
                        "max_trades_per_day": max(trades_day.values(), default=0),
                        "capped_orders": capped, "ruin_skipped_orders": ruin_skipped,
                        "max_orders_per_day_cap": -1 if max_orders_per_day is None else int(max_orders_per_day)})
    return res


def run_segment_hold(bars, calendar, direction, costs, initial_capital, split, blocks,
                     leverage_cap=5.0, unit_step=1000):
    """Passive baseline: enter at the first bar at/after S, exit at the close before F,
    for every segment holding at least one evaluated block. Pays financing."""
    ts = bars.index
    v = ts.as_unit("ns").asi8
    o, c = bars.OPEN.to_numpy(float), bars.CLOSE.to_numpy(float)
    evaluated = list(np.flatnonzero(blocks.in_split(split)))
    segs = sorted({int(blocks.segment[k]) for k in evaluated})
    trades = []
    for si in segs:
        seg = calendar.segments[si]
        lo = int(np.searchsorted(v, seg.operational_start.value, side="left"))
        hi = int(np.searchsorted(v, (seg.force_flat - HOUR).value, side="right"))
        if hi - lo < 2 or ts[lo] + HOUR > seg.entry_cutoff:
            continue
        units = math.floor(leverage_cap * initial_capital / o[lo] / unit_step) * unit_step
        fill = Fill(direction=direction, units=units, entry_bar=0, entry_price=float(o[lo]),
                    exit_bar=hi - 1 - lo, exit_price=float(c[hi - 1]), exit_reason="TIME")
        trades.append(_trade_record(v, si, lo, fill, costs, calendar, -1))
    return _account(bars, trades, split, initial_capital, evaluated, 0)
