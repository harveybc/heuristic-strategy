"""Hindsight labels: the ZigZag expert's action at every bar, its label support, and the state-aware
relabeling oracle used by DAgger.

NON-CAUSAL. Everything here reads the final (repainted) pivots and is a LABEL, never an input.

* ``pivot_labels``: the expert's action at bar b (the anticipatory target, Musashi F05 T1): LONG when b is
  a final VALLEY pivot, SHORT when b is a final PEAK, NONE otherwise. Signal bar = pivot bar (the
  executable expert decides after the completed pivot bar and fills at the next open).
* ``confirmed_labels``: the reactive comparator (F05 T2): LONG/SHORT at the knowable confirmation bar of a
  valley/peak (``zigzag.confirmation_bars_fast``); this target is causal by construction.
* ``label_support_end``: the last bar the hindsight label of bar b depends on. A bar's status (pivot or not)
  is final once the first final pivot at or after b is locked, i.e. at the confirmation bar of that pivot.
  -1 = the lock lies beyond the data: CENSORED, never labelled NONE (audit F01/A03).
* ``LegOracle``: the expert's leg-following rule evaluated in ANY agent state (DAgger relabeler). In the
  current ZigZag leg (direction of the next final pivot), with remaining move m = d* x (next pivot level -/+
  inset - close[b]): flat -> enter d* when the entry gate is open and m >= ``min_remaining_sigma`` sigma,
  else HOLD; positioned with the leg -> HOLD; positioned against it -> close-and-reverse when an entry is
  allowed and m is large enough, else CLOSE. TP label = m / sigma (clipped to the action bounds), SL label
  = the expert's 2 sigma. At the expert's own pivot bars from flat it reproduces the expert's entry.
"""
from __future__ import annotations

import numpy as np

from .. import zigzag as Z
from .env import CLOSE, HOLD, LONG, SHORT

NONE_CLASS, LONG_CLASS, SHORT_CLASS = 0, 1, 2
CLASS_NAMES = ("NONE", "LONG", "SHORT")


def pivot_labels(n: int, pivots) -> np.ndarray:
    y = np.zeros(n, np.int64)
    for i, kind, _ in pivots:
        y[i] = LONG_CLASS if kind == Z.VALLEY else SHORT_CLASS
    return y


def confirmed_labels(n: int, pivots, conf):
    """(labels, collisions): the later pivot wins when two confirmations share a bar."""
    y = np.zeros(n, np.int64)
    collisions = 0
    for (i, kind, _), c in sorted(zip(pivots, conf), key=lambda t: (t[1], t[0][0])):
        if c is None or c < 0 or c >= n:
            continue
        collisions += int(y[c] != 0)
        y[c] = LONG_CLASS if kind == Z.VALLEY else SHORT_CLASS
    return y, collisions


def label_support_end(n: int, pivots, conf) -> np.ndarray:
    """Bar index after which the hindsight label of each bar is final; -1 when censored."""
    out = np.full(n, -1, np.int64)
    piv_bars = np.array([i for i, _, _ in pivots], np.int64)
    conf = np.asarray(conf, np.int64)
    j = np.searchsorted(piv_bars, np.arange(n), side="left")
    has = j < len(piv_bars)
    c = np.where(has, conf[np.minimum(j, len(conf) - 1)], -1)
    out[has] = c[has]
    return out


class LegOracle:
    def __init__(self, close: np.ndarray, pivots, inset: float = Z.PIP, min_remaining_sigma: float = 1.0,
                 sl_mult: float = 2.0, tp_bounds=(0.25, 3.0)):
        n = len(close)
        self.close = np.asarray(close, float)
        piv_bars = np.array([i for i, _, _ in pivots], np.int64)
        kinds = np.array([k for _, k, _ in pivots], np.int64)
        lvls = np.array([v for _, _, v in pivots], float)
        j = np.searchsorted(piv_bars, np.arange(n), side="right")      # first pivot strictly after b
        has = j < len(piv_bars)
        jj = np.minimum(j, len(piv_bars) - 1)
        self.d_star = np.where(has, np.where(kinds[jj] == Z.PEAK, 1, -1), 0)
        target = np.where(has, lvls[jj] - self.d_star * inset, np.nan)
        self.remaining = np.where(has, self.d_star * (target - self.close), np.nan)
        self.min_rem, self.sl_mult, self.tp_bounds = min_remaining_sigma, sl_mult, tp_bounds

    def action(self, b: int, position: int, can_enter: bool, sigma: float):
        """(action, tp_mult, sl_mult) the expert takes after bar b in the agent's state."""
        d = int(self.d_star[b])
        if d == 0 or not np.isfinite(sigma) or sigma <= 0:
            return (CLOSE if position != 0 else HOLD), 1.0, self.sl_mult
        m = float(self.remaining[b]) / sigma
        tp = float(np.clip(m, *self.tp_bounds))
        enter = can_enter and m >= self.min_rem
        act_dir = LONG if d > 0 else SHORT
        if position == 0:
            return (act_dir if enter else HOLD), tp, self.sl_mult
        if position == d:
            return HOLD, tp, self.sl_mult
        return (act_dir if enter else CLOSE), tp, self.sl_mult


def reversal_pivots(high, low, threshold: float):
    """Classic reversal ZigZag on highs / lows (hindsight; labels only, never an input). A leg's extreme is fixed as a
    pivot at the first LATER bar whose low (high) lies >= ``threshold`` (price units) beyond it; a bar that makes a new
    extreme extends the leg and cannot also confirm a reversal from it (no intrabar-order assumption), so every pivot
    is confirmed strictly after its own bar. Returns (pivots [(bar, PEAK|VALLEY, level)], conf [bar it became final]).

    SYMMETRIC under price negation (2026-10-27): before the first pivot both running extremes are tracked and the
    first peak and the first valley obey the same rule (a peak needs the running high to be later than the running low
    and a drop of ``threshold`` on a bar that is not a new high; a valley the mirror conditions), so
    reversal_pivots(K - low, K - high) == mirror(reversal_pivots(high, low)) exactly. The legacy rule examined the
    high first and let a new-low bar confirm a peak while a new-high bar could not confirm a valley; only the pivots
    before and at the first confirmation can differ from ``reversal_pivots_legacy``."""
    high, low = np.asarray(high, float), np.asarray(low, float)
    n = len(high)
    piv, conf = [], []
    if n == 0:
        return piv, conf
    trend, hi_i, hi_v, lo_i, lo_v = 0, 0, high[0], 0, low[0]
    for i in range(1, n):
        if trend == 0:
            nh, nl = high[i] > hi_v, low[i] < lo_v
            pk = (not nh) and (hi_v - low[i] >= threshold) and hi_i > lo_i
            vl = (not nl) and (high[i] - lo_v >= threshold) and lo_i > hi_i
            if pk:
                piv.append((hi_i, Z.PEAK, hi_v)); conf.append(i)
                trend, lo_i, lo_v = -1, i, low[i]
                continue
            if vl:
                piv.append((lo_i, Z.VALLEY, lo_v)); conf.append(i)
                trend, hi_i, hi_v = 1, i, high[i]
                continue
            if nh:
                hi_i, hi_v = i, high[i]
            if nl:
                lo_i, lo_v = i, low[i]
        elif trend == 1:
            if high[i] > hi_v:
                hi_i, hi_v = i, high[i]
            elif hi_v - low[i] >= threshold:
                piv.append((hi_i, Z.PEAK, hi_v)); conf.append(i)
                trend, lo_i, lo_v = -1, i, low[i]
        else:
            if low[i] < lo_v:
                lo_i, lo_v = i, low[i]
            elif high[i] - lo_v >= threshold:
                piv.append((lo_i, Z.VALLEY, lo_v)); conf.append(i)
                trend, hi_i, hi_v = 1, i, high[i]
    return piv, conf


def reversal_pivots_legacy(high, low, threshold: float):
    """Classic reversal ZigZag on highs / lows (hindsight; labels only, never an input). A leg's extreme is fixed as a
    pivot at the first LATER bar whose low (high) lies >= ``threshold`` (price units) beyond it; a bar that makes a new
    extreme extends the leg and cannot also confirm a reversal from it (no intrabar-order assumption), so every pivot
    is confirmed strictly after its own bar. Returns (pivots [(bar, PEAK|VALLEY, level)], conf [bar it became final])."""
    high, low = np.asarray(high, float), np.asarray(low, float)
    n = len(high)
    piv, conf = [], []
    if n == 0:
        return piv, conf
    trend, hi_i, hi_v, lo_i, lo_v = 0, 0, high[0], 0, low[0]
    for i in range(1, n):
        if trend >= 0:
            if high[i] > hi_v:
                hi_i, hi_v = i, high[i]
            elif hi_v - low[i] >= threshold and (trend == 1 or hi_i > lo_i):
                piv.append((hi_i, Z.PEAK, hi_v)); conf.append(i)
                trend, lo_i, lo_v = -1, i, low[i]
                continue
            if trend == 0:
                if low[i] < lo_v:
                    lo_i, lo_v = i, low[i]
                elif high[i] - lo_v >= threshold and lo_i > hi_i:
                    piv.append((lo_i, Z.VALLEY, lo_v)); conf.append(i)
                    trend, hi_i, hi_v = 1, i, high[i]
        else:
            if low[i] < lo_v:
                lo_i, lo_v = i, low[i]
            elif high[i] - lo_v >= threshold:
                piv.append((lo_i, Z.VALLEY, lo_v)); conf.append(i)
                trend, hi_i, hi_v = 1, i, high[i]
    return piv, conf
