"""Market-time block grid and causal labels (high, low, first touch)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendar import Calendar, HOUR

HIGH_FIRST, LOW_FIRST, AMBIGUOUS = 1, -1, 0
OK, INSUFFICIENT_BARS, NO_REFERENCE = "OK", "INSUFFICIENT_BARS", "NO_REFERENCE"


def first_touch(high: np.ndarray, low: np.ndarray) -> int:
    """Order of the block extremes by first attainment; same bar -> AMBIGUOUS."""
    i_h = int(np.argmax(high == high.max()))
    i_l = int(np.argmax(low == low.min()))
    if i_l < i_h:
        return LOW_FIRST
    if i_h < i_l:
        return HIGH_FIRST
    return AMBIGUOUS


@dataclass
class Blocks:
    hours: int
    start: list
    end: list
    segment: np.ndarray
    bar_lo: np.ndarray
    bar_hi: np.ndarray
    ref_bar: np.ndarray
    ref: np.ndarray
    H: np.ndarray
    L: np.ndarray
    ft: np.ndarray
    status: np.ndarray

    def __len__(self):
        return len(self.start)

    def ft_oracle(self) -> np.ndarray:
        """Oracle first-touch: truth, with AMBIGUOUS replaced by the fixed coin."""
        from .noise import ambiguous_coin, resolve_ambiguous
        return resolve_ambiguous(self.ft, ambiguous_coin(len(self.ft)))

    def in_split(self, split) -> np.ndarray:
        a, b = (pd.Timestamp(x).value for x in split)
        start = np.array([s.value for s in self.start], dtype=np.int64)
        end = np.array([e.value for e in self.end], dtype=np.int64)
        return (self.status == OK) & (start >= a) & (end <= b)


def build_blocks(bars: pd.DataFrame, calendar: Calendar, block_hours: int,
                 min_fraction: float = 0.5) -> Blocks:
    ts = bars.index
    seg_of_bar = calendar.assign(ts)
    v = ts.as_unit("ns").asi8
    high, low, close = (bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    B = block_hours * HOUR
    rows = []
    for si, seg in enumerate(calendar.segments):
        k = 0
        while seg.market_open + k * B < seg.force_flat:
            start = max(seg.market_open + k * B, seg.operational_start)
            end = min(seg.market_open + (k + 1) * B, seg.force_flat)
            k += 1
            if end - start < min_fraction * B or start >= seg.entry_cutoff:
                continue
            lo = int(np.searchsorted(v, start.value, side="left"))
            hi = int(np.searchsorted(v, (end - HOUR).value, side="right"))
            length_h = (end - start) / HOUR
            status, ref_bar = OK, lo - 1
            if hi - lo < math.ceil(length_h * min_fraction):
                status = INSUFFICIENT_BARS
            elif ref_bar < 0 or seg_of_bar[ref_bar] != si:
                status = NO_REFERENCE
            if status == OK:
                Hk, Lk = high[lo:hi].max(), low[lo:hi].min()
                ftk = first_touch(high[lo:hi], low[lo:hi])
                refk = close[ref_bar]
            else:
                Hk = Lk = refk = np.nan
                ftk = AMBIGUOUS
            rows.append((start, end, si, lo, hi, ref_bar, refk, Hk, Lk, ftk, status))
    cols = list(zip(*rows)) if rows else [[]] * 11
    return Blocks(block_hours, list(cols[0]), list(cols[1]), np.array(cols[2], int),
                  np.array(cols[3], int), np.array(cols[4], int), np.array(cols[5], int),
                  np.array(cols[6], float), np.array(cols[7], float), np.array(cols[8], float),
                  np.array(cols[9], int), np.array(cols[10], object))
