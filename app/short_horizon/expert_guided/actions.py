"""Continuous action contract for the SAC arms (E2) over the bar env, and mark-to-market equity.

Action x in [-1, 1]^3 (declared, fixed before any fit):
  x[0] target position with hysteresis: > +1/3 long, < -1/3 short, otherwise flat;
  x[1] TP sigma multiple, linearly mapped to [0.25, 3];  x[2] SL sigma multiple, linearly mapped to [0.5, 3].
Decoding against the current position: target == position -> HOLD; target flat while positioned -> CLOSE;
target opposite or from flat -> LONG/SHORT (the env applies every entry rule; reverse = close + entry).
Every entry therefore carries an SL and a TP (owner rule); the env clips them into the effective band.
"""
from __future__ import annotations

import numpy as np

from .env import CLOSE, HOLD, LONG, SHORT

TP_RANGE = (0.25, 3.0)
SL_RANGE = (0.5, 3.0)
DEAD_BAND = 1.0 / 3.0


def _lin(x, lo, hi):
    return lo + (float(np.clip(x, -1, 1)) + 1) / 2 * (hi - lo)


def _inv(v, lo, hi):
    return float(np.clip(2 * (float(np.clip(v, lo, hi)) - lo) / (hi - lo) - 1, -1, 1))


def decode(x, position: int):
    t = 1 if x[0] > DEAD_BAND else (-1 if x[0] < -DEAD_BAND else 0)
    tp, sl = _lin(x[1], *TP_RANGE), _lin(x[2], *SL_RANGE)
    if t == position:
        a = HOLD
    elif t == 0:
        a = CLOSE
    else:
        a = LONG if t > 0 else SHORT
    return a, tp, sl


def encode(action: int, position: int, tp: float, sl: float) -> np.ndarray:
    """The continuous action that decodes to ``action`` from ``position`` (demonstrations)."""
    if action == LONG:
        t = 1.0
    elif action == SHORT:
        t = -1.0
    elif action == CLOSE:
        t = 0.0
    else:
        t = float(position)
    return np.array([t, _inv(tp, *TP_RANGE), _inv(sl, *SL_RANGE)], np.float32)


def mark_to_market(env, k: int) -> float:
    """Equity at the close of bar k: realised balance + open P&L - the open trade's already-paid entry cost."""
    p = env.pos
    if p is None:
        return float(env.balance)
    return float(env.balance + p.direction * p.units * (env.c[k] - p.entry_price) - sum(p.entry_cost.values()))
