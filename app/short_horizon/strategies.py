"""Range-first policies, the short-horizon comparator and the baselines."""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from app.policies import prediction_entry_exit as P

from .blocks import AMBIGUOUS, HIGH_FIRST, LOW_FIRST, OK, Blocks
from .calendar import HOUR
from .execution import BlockOrder, Costs


@dataclass(frozen=True)
class RangeFirstParams:
    """Fixed before scores (contract section 5); never tuned on results."""
    inset: float = 0.10
    stop_buffer: float = 0.25
    min_edge_costs: float = 1.5
    risk_fraction: float = 0.01
    leverage_cap: float = 5.0
    unit_step: int = 1000


def size_units(entry_ref, stop, capital, p: RangeFirstParams) -> int:
    if not (math.isfinite(entry_ref) and entry_ref > 0):
        return 0
    risk = abs(entry_ref - stop)
    cap = p.leverage_cap * capital / entry_ref
    units = min(p.risk_fraction * capital / risk, cap) if (risk > 0 and math.isfinite(risk)) else cap
    return int(math.floor(units / p.unit_step) * p.unit_step)


def range_first_order(variant, H_hat, L_hat, ft_hat, ref, costs: Costs, p: RangeFirstParams, capital):
    if variant not in ("RF-M", "RF-L"):
        raise ValueError(variant)
    if not all(math.isfinite(x) for x in (H_hat, L_hat, ref)) or ft_hat not in (LOW_FIRST, HIGH_FIRST):
        return None
    R = H_hat - L_hat
    if R <= 0:
        return None
    edge = p.min_edge_costs * costs.round_trip_price()
    d = 1 if ft_hat == LOW_FIRST else -1
    if d > 0:
        tp, sl, e = H_hat - p.inset * R, L_hat - p.stop_buffer * R, L_hat + p.inset * R
    else:
        tp, sl, e = L_hat + p.inset * R, H_hat + p.stop_buffer * R, H_hat - p.inset * R
    limit = variant == "RF-L" and (e < ref if d > 0 else e > ref)
    entry = e if limit else ref
    if not (sl < entry < tp if d > 0 else tp < entry < sl):
        return None
    reward = d * (tp - entry)
    if reward <= 0 or reward < edge:
        return None
    units = size_units(entry, sl, capital, p)
    if units <= 0:
        return None
    return BlockOrder(d, "LIMIT" if limit else "MARKET", e if limit else None, tp, sl, units)


def range_first_orders(variant, blocks: Blocks, H_hat, L_hat, ft_hat, costs, p, capital):
    return [range_first_order(variant, float(H_hat[k]), float(L_hat[k]), int(ft_hat[k]),
                              float(blocks.ref[k]), costs, p, capital)
            if blocks.status[k] == OK else None for k in range(len(blocks))]


# ------------------------------------------------------------------ comparator (ii)
@functools.wraps(P.calculate_entry_geometry)
def entry_geometry(current_price, predictions, params):
    return P.calculate_entry_geometry(current_price=current_price,
                                      long_horizon_predictions=predictions, params=params)


def comparator_params(costs: Costs, min_edge_costs: float = 1.5) -> P.PredictionEntryExitParameters:
    """Plugin defaults; only profit_threshold is set, to the cost-aware edge in
    the policy's own 0.00001 point unit."""
    base = P.PredictionEntryExitParameters()
    return P.PredictionEntryExitParameters(
        profit_threshold=min_edge_costs * costs.round_trip_price() / base.pip_cost)


def comparator_horizons(block_hours: int) -> tuple:
    return (2, 3, 4, 6) if block_hours == 6 else (2, 3, 4, 6, 8, 12)


def first_barrier(direction, predictions, tp, sl):
    """Chronological rule: the first ordered sample beyond a barrier decides."""
    for x in predictions:
        if (x <= sl) if direction == "long" else (x >= sl):
            return "SL"
        if (x >= tp) if direction == "long" else (x <= tp):
            return "TP"
    return None


def comparator_order(ref, predictions, costs, params, capital, p: RangeFirstParams = RangeFirstParams()):
    path = [float(x) for x in predictions if math.isfinite(float(x))]
    if not path or not math.isfinite(ref):
        return None
    g = entry_geometry(ref, path, params)
    if g is None or first_barrier(g.direction, path, g.take_profit_price, g.stop_loss_price) != "TP":
        return None
    d = 1 if g.direction == "long" else -1
    units = size_units(ref, g.stop_loss_price, capital, p)
    if units <= 0:
        return None
    return BlockOrder(d, "MARKET", None, g.take_profit_price, g.stop_loss_price, units)


def comparator_orders(blocks: Blocks, paths, costs, capital):
    params = comparator_params(costs)
    return [comparator_order(float(blocks.ref[k]), paths[k], costs, params, capital)
            if blocks.status[k] == OK else None for k in range(len(blocks))]


def comparator_targets(bars, blocks: Blocks, horizons) -> np.ndarray:
    """Close of the bar labelled t0 + h - 1h, if observed inside the block; else NaN."""
    v = bars.index.as_unit("ns").asi8
    close = bars.CLOSE.to_numpy(float)
    out = np.full((len(blocks), len(horizons)), np.nan)
    for k in range(len(blocks)):
        if blocks.status[k] != OK:
            continue
        for j, h in enumerate(horizons):
            t = blocks.start[k] + (h - 1) * HOUR
            if t + HOUR > blocks.end[k]:
                continue
            i = int(np.searchsorted(v, t.value))
            if i < len(v) and v[i] == t.value:
                out[k, j] = close[i]
    return out


# ------------------------------------------------------------------ naive and baselines
def naive_forecasts(blocks: Blocks, kind: str, window: int = 20):
    """Causal naive (U, D, first touch) from completed blocks (end <= start)."""
    if kind not in ("PERSIST", "MEAN20"):
        raise ValueError(kind)
    n = len(blocks)
    u = np.full(n, np.nan)
    d = np.full(n, np.nan)
    ft = np.full(n, AMBIGUOUS, int)
    done, pending = [], []
    for k in range(n):
        still = []
        for a in pending:
            (done if blocks.end[a] <= blocks.start[k] else still).append(a)
        pending = still
        if kind == "PERSIST" and done:
            a = done[-1]
            u[k], d[k] = blocks.H[a] - blocks.ref[a], blocks.ref[a] - blocks.L[a]
            ft[k] = blocks.ft[a]
        elif kind == "MEAN20" and len(done) >= window:
            w = done[-window:]
            u[k] = np.mean([blocks.H[a] - blocks.ref[a] for a in w])
            d[k] = np.mean([blocks.ref[a] - blocks.L[a] for a in w])
            s = sum(int(blocks.ft[a]) for a in w)
            ft[k] = HIGH_FIRST if s > 0 else LOW_FIRST if s < 0 else AMBIGUOUS
        if blocks.status[k] == OK:
            pending.append(k)
    return u, d, ft


def passive_block_orders(blocks: Blocks, direction, capital, p: RangeFirstParams = RangeFirstParams()):
    out = []
    for k in range(len(blocks)):
        if blocks.status[k] != OK:
            out.append(None)
            continue
        units = size_units(float(blocks.ref[k]), -direction * math.inf, capital, p)
        out.append(BlockOrder(direction, "MARKET", None, direction * math.inf, -direction * math.inf, units))
    return out


def naive_range(blocks: Blocks, kind: str):
    from .noise import ambiguous_coin, resolve_ambiguous
    u, d, ft = naive_forecasts(blocks, kind)
    return blocks.ref + u, blocks.ref - d, resolve_ambiguous(ft, ambiguous_coin(len(blocks)))


def baseline_orders(blocks: Blocks, costs: Costs, p: RangeFirstParams, capital) -> dict:
    out = {"NO_TRADE": [None] * len(blocks),
           "BLOCK_BUY": passive_block_orders(blocks, 1, capital, p),
           "BLOCK_SELL": passive_block_orders(blocks, -1, capital, p)}
    for kind in ("PERSIST", "MEAN20"):
        H_hat, L_hat, ft = naive_range(blocks, kind)
        for variant in ("RF-M", "RF-L"):
            out[f"NAIVE_{kind}_{variant}"] = range_first_orders(variant, blocks, H_hat, L_hat, ft,
                                                                costs, p, capital)
    return out


# ------------------------------------------------------------------ price-path-only input (owner, 2026-10-19)
def path_to_range(paths):
    """Derive (H_hat, L_hat, first touch) from a predicted hourly price path.

    H_hat = max, L_hat = min over the finite predicted prices; first touch is
    the order of their first occurrences (argmax before argmin -> HIGH_FIRST).
    A flat path or one with no finite price gives AMBIGUOUS (no trade).
    """
    P = np.asarray(paths, float)
    n = len(P)
    finite = np.isfinite(P)
    has = finite.any(axis=1)
    pmax = np.where(finite, P, -np.inf)
    pmin = np.where(finite, P, np.inf)
    ih, il = pmax.argmax(axis=1), pmin.argmin(axis=1)
    rows = np.arange(n)
    H_hat = np.where(has, pmax[rows, ih], np.nan)
    L_hat = np.where(has, pmin[rows, il], np.nan)
    ft = np.where(il < ih, LOW_FIRST, np.where(ih < il, HIGH_FIRST, AMBIGUOUS))
    ft = np.where(has, ft, AMBIGUOUS).astype(int)
    return H_hat, L_hat, ft


# ------------------------------------------------------------------ business hard rule helpers
def causal_block_sigma(blocks: Blocks) -> np.ndarray:
    """Causal block-range scale: mean (H - L) of the last 20 completed blocks (NaN before)."""
    u, d, _ = naive_forecasts(blocks, "MEAN20")
    return u + d


def apply_order_rules(orders, blocks: Blocks, sigma, rules):
    """Strategy-side filter: an order that would break the SL/TP rule is not sent (counted)."""
    from .execution import OrderRuleError, check_order
    out, skipped = [], 0
    for k, o in enumerate(orders):
        if o is None:
            out.append(None)
            continue
        try:
            check_order(o, float(blocks.ref[k]), float(sigma[k]), rules)
            out.append(o)
        except OrderRuleError:
            out.append(None)
            skipped += 1
    return out, skipped
