"""Direction-free strategies on predicted volatility (owner pivot, 2026-10-19). DEVELOPMENT.

V1 OCO breakout (trades only on predicted expansion): buy-stop at r + k_entry*U_hat and sell-stop
at r - k_entry*D_hat; the first fill cancels the other. Broker-side TP = fill +/- k_tp*R_hat and
SL = fill -/+ k_sl*R_hat (R_hat = U_hat + D_hat), the SL MOVED as a trailing stop
(close -/+ k_trail*R_hat, effective from the next bar, only ever tightened, never removed).

V2 range reversion (trades only on predicted contraction): resting OCO limits at a causal
anchor (mean close of the previous B bars, never centred) +/- b*sigma_hat with
sigma_hat = R_hat/2; TP at the anchor, SL at the band -/+ m_sl*sigma_hat; time stop at block end.

Every parameter is fixed a priori here (contract section 16); none is tuned on 2024.
Toda orden lleva SL y TP calculados y efectivos desde la entrada.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .blocks import OK
from .calendar import HOUR
from .execution import BlockOrder, OrderRules, check_order, _protect


@dataclass(frozen=True)
class Leg:
    direction: int
    entry_type: str            # "STOP" or "LIMIT"
    entry_price: float
    tp_offset_or_price: float  # relative: distance from the fill; absolute: price
    sl_offset_or_price: float
    trail: Optional[float] = None
    relative: bool = True

    @property
    def tp_offset(self):
        return self.tp_offset_or_price

    @property
    def take_profit(self):
        """Planned absolute TP (from the planned entry)."""
        return self.entry_price + self.direction * self.tp_offset_or_price if self.relative else self.tp_offset_or_price

    @property
    def stop_loss(self):
        return self.entry_price - self.direction * self.sl_offset_or_price if self.relative else self.sl_offset_or_price


@dataclass(frozen=True)
class OcoOrder:
    legs: Tuple[Leg, ...]
    units: Optional[float] = None
    percentual_order_size: float = 1.0
    entry_type: str = "OCO"
    entry_price: Optional[float] = None
    direction: int = 0


@dataclass
class OcoFill:
    direction: int
    units: float
    entry_bar: int
    entry_price: float
    exit_bar: int = -1
    exit_price: float = math.nan
    exit_reason: str = ""
    exit_at_open: bool = False
    ambiguous: bool = False
    refused: bool = False
    entry_kind: str = ""
    ambiguous_entry: bool = False
    trailed: bool = False
    leg_tp: float = math.nan
    leg_sl: float = math.nan
    leg_sl_initial: float = math.nan


def _levels(leg: Leg, fill: float):
    if leg.relative:
        return fill + leg.direction * leg.tp_offset_or_price, fill - leg.direction * leg.sl_offset_or_price
    return leg.tp_offset_or_price, leg.sl_offset_or_price


def _trigger(leg: Leg, o, h, l, through):
    """(fill price, at_open) if the leg triggers on this bar, else None."""
    e, d = leg.entry_price, leg.direction
    if leg.entry_type == "STOP":
        if (o >= e) if d > 0 else (o <= e):
            return float(o), True
        if (h >= e) if d > 0 else (l <= e):
            return float(e), False
        return None
    if (o <= e) if d > 0 else (o >= e):
        return float(o), True
    if (l <= e - through) if d > 0 else (h >= e + through):
        return float(e), False
    return None


def _run_after_fill(leg, fill_price, at_open, k, o, h, l, c, through):
    """Manage one filled leg from bar k on; returns (exit_bar, exit_price, reason, at_open, ambiguous,
    trailed, final_sl, tp, initial_sl)."""
    d = leg.direction
    tp, sl = _levels(leg, fill_price)
    sl0, trailed = sl, False
    n = len(o)
    if not at_open:   # intrabar fill: its order relative to the bar's extremes is unknown -> stop only
        if (l[k] <= sl) if d > 0 else (h[k] >= sl):
            return k, sl, "SL", False, True, False, sl, tp, sl0
        start = k + 1
    else:
        hit = _protect(d, tp, sl, o[k], h[k], l[k], through)
        if hit:
            reason, price, op, amb = hit
            return k, float(price), reason, op, amb, False, sl, tp, sl0
        start = k + 1
    if leg.trail is not None:   # the bar k close moves the stop, effective from bar k + 1
        new = c[k] - d * leg.trail
        if (new > sl) if d > 0 else (new < sl):
            sl, trailed = new, True
    for j in range(start, n):
        hit = _protect(d, tp, sl, o[j], h[j], l[j], through)
        if hit:
            reason, price, op, amb = hit
            return j, float(price), reason, op, amb, trailed and reason == "SL", sl, tp, sl0
        if leg.trail is not None:
            new = c[j] - d * leg.trail
            if (new > sl) if d > 0 else (new < sl):
                sl, trailed = new, True
    return n - 1, float(c[n - 1]), "TIME", False, False, False, sl, tp, sl0


def simulate_oco(order: OcoOrder, o, h, l, c, entry_bars: Optional[int] = None,
                 limit_through: float = 0.0, entry_delay: int = 0) -> Optional[OcoFill]:
    n = len(o)
    entry_bars = n if entry_bars is None else min(entry_bars, n)
    for k in range(entry_delay, entry_bars):
        trig = [(leg, _trigger(leg, o[k], h[k], l[k], limit_through)) for leg in order.legs]
        trig = [(leg, t) for leg, t in trig if t is not None]
        if not trig:
            continue
        ambiguous_entry = False
        if len(trig) > 1:
            at_open = [x for x in trig if x[1][1]]
            if at_open:
                trig = at_open[:1]      # the observed open comes first
            else:
                # both legs inside one bar: unknown order -> take the leg with the worse outcome
                ambiguous_entry = True
                outcomes = []
                for leg, (price, op) in trig:
                    res = _run_after_fill(leg, price, op, k, o, h, l, c, limit_through)
                    outcomes.append((leg.direction * (res[1] - price), leg, price, op, res))
                outcomes.sort(key=lambda x: x[0])
                _, leg, price, op, res = outcomes[0]
                return _fill(order, leg, k, price, op, res, ambiguous_entry)
        leg, (price, op) = trig[0]
        res = _run_after_fill(leg, price, op, k, o, h, l, c, limit_through)
        return _fill(order, leg, k, price, op, res, ambiguous_entry)
    return None


def _fill(order, leg, k, price, op, res, ambiguous_entry):
    exit_bar, exit_price, reason, at_open, amb, trailed, sl, tp, sl0 = res
    return OcoFill(direction=leg.direction, units=order.units, entry_bar=k, entry_price=price,
                   exit_bar=exit_bar, exit_price=exit_price, exit_reason=reason, exit_at_open=at_open,
                   ambiguous=amb, entry_kind=leg.entry_type, ambiguous_entry=ambiguous_entry, trailed=trailed,
                   leg_tp=tp, leg_sl=sl, leg_sl_initial=sl0)


def check_oco(order: OcoOrder, sigma: float, rules: OrderRules, require_units: bool = False) -> None:
    for leg in order.legs:
        check_order(BlockOrder(leg.direction, "LIMIT", leg.entry_price, leg.take_profit, leg.stop_loss,
                               order.units), leg.entry_price, sigma, rules, require_units=require_units)


# ------------------------------------------------------------------ order builders
@dataclass(frozen=True)
class V1Params:
    expansion_threshold: float = 1.10   # trade only if R_hat / sigma_recent >= this
    k_entry: float = 0.5
    k_tp: float = 0.5
    k_sl: float = 0.5
    k_trail: float = 0.35


@dataclass(frozen=True)
class V2Params:
    contraction_threshold: float = 0.90  # trade only if R_hat / sigma_recent <= this
    b_band: float = 0.8                  # band = anchor -/+ b * sigma_hat, sigma_hat = R_hat / 2
    m_sl: float = 2.0                    # SL = band -/+ m * sigma_hat


def v1_order(blocks, k, U_hat, D_hat, sigma_recent, p: V1Params = V1Params()) -> Optional[OcoOrder]:
    if blocks.status[k] != OK or not all(map(math.isfinite, (U_hat, D_hat, sigma_recent))):
        return None
    if U_hat <= 0 or D_hat <= 0 or sigma_recent <= 0:
        return None
    R = U_hat + D_hat
    if R / sigma_recent < p.expansion_threshold:
        return None
    r = float(blocks.ref[k])
    up = Leg(1, "STOP", r + p.k_entry * U_hat, p.k_tp * R, p.k_sl * R, p.k_trail * R, relative=True)
    dn = Leg(-1, "STOP", r - p.k_entry * D_hat, p.k_tp * R, p.k_sl * R, p.k_trail * R, relative=True)
    return OcoOrder((up, dn))


def causal_anchor(bars, blocks, k) -> float:
    """Mean close of the B bars ending at the decision time t0 (only bars with label < t0)."""
    t0 = blocks.start[k]
    lo = t0 - blocks.hours * HOUR
    sel = bars.CLOSE[(bars.index >= lo) & (bars.index < t0)]
    return float(sel.mean()) if len(sel) else math.nan


def v2_order(bars, blocks, k, U_hat, D_hat, sigma_recent, p: V2Params = V2Params(),
             anchor: Optional[float] = None) -> Optional[OcoOrder]:
    if blocks.status[k] != OK or not all(map(math.isfinite, (U_hat, D_hat, sigma_recent))):
        return None
    if U_hat <= 0 or D_hat <= 0 or sigma_recent <= 0:
        return None
    R = U_hat + D_hat
    if R / sigma_recent > p.contraction_threshold:
        return None
    a = causal_anchor(bars, blocks, k) if anchor is None else anchor
    if not math.isfinite(a):
        return None
    s = R / 2
    r = float(blocks.ref[k])
    legs = []
    buy_e, sell_e = a - p.b_band * s, a + p.b_band * s
    if buy_e < r:                       # a buy-limit above the market would be marketable, not a reversion
        legs.append(Leg(1, "LIMIT", buy_e, a, buy_e - p.m_sl * s, None, relative=False))
    if sell_e > r:
        legs.append(Leg(-1, "LIMIT", sell_e, a, sell_e + p.m_sl * s, None, relative=False))
    return OcoOrder(tuple(legs)) if legs else None


# ------------------------------------------------------------------ range error families
def blend_excursion(truth, naive, w, eps_sigma, z):
    """A1r: naive + w (truth - naive) + eps; w = 0 is the MEAN20 naive, w = 1 the truth."""
    return np.asarray(naive, float) + w * (np.asarray(truth, float) - np.asarray(naive, float)) \
        + eps_sigma * np.asarray(z, float)
