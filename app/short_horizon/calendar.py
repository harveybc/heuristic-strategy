"""Configured session calendar: segments, entry delay, forced flat, rollovers.

The calendar is data (a versioned JSON), never inferred from day names. A
helper derives segments from observed bar gaps once, by a declared rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)


class CalendarError(ValueError):
    pass


@dataclass(frozen=True)
class Segment:
    segment_id: str
    market_open: pd.Timestamp
    market_close: pd.Timestamp
    operational_start: pd.Timestamp
    force_flat: pd.Timestamp
    entry_cutoff: pd.Timestamp

    def rollovers(self) -> list:
        """Daily financing instants O + 24k (k >= 1) strictly inside the segment."""
        out, k = [], 1
        while self.market_open + k * 24 * HOUR < self.market_close:
            out.append(self.market_open + k * 24 * HOUR)
            k += 1
        return out


def _ts(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def derive_segments(index: pd.DatetimeIndex, max_gap_hours: float = 24) -> list:
    """Segments = maximal runs of bars with no gap above ``max_gap_hours``.

    open = first bar label, close = last bar label + 1 h. Returned as plain
    dicts (ISO strings) so they can be written to the versioned calendar JSON.
    """
    if len(index) == 0:
        raise CalendarError("no bars")
    idx = pd.DatetimeIndex(index)
    gaps = np.diff(idx.as_unit("ns").asi8) / 3.6e12
    cuts = np.flatnonzero(gaps > max_gap_hours) + 1
    bounds = np.r_[0, cuts, len(idx)]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        o, c = idx[a], idx[b - 1] + HOUR
        segs.append({"segment_id": f"SEG-{o.strftime('%Y%m%dT%H')}",
                     "market_open": o.isoformat(), "market_close": c.isoformat()})
    return segs


class Calendar:
    def __init__(self, segments: Sequence[Segment], config: Mapping):
        self.segments = tuple(segments)
        self.config = dict(config)
        self._open = np.array([s.market_open.value for s in self.segments])
        self._close = np.array([s.market_close.value for s in self.segments])
        self._rollovers = [tuple(r.value for r in s.rollovers()) for s in self.segments]

    def rollover_ns(self, segment_index: int) -> tuple:
        return self._rollovers[segment_index]

    def assign(self, index) -> np.ndarray:
        """Segment index of every bar; a bar outside every segment is an error."""
        v = pd.DatetimeIndex(index).as_unit("ns").asi8
        pos = np.searchsorted(self._open, v, side="right") - 1
        bad = (pos < 0) | (v + HOUR.value > self._close[np.maximum(pos, 0)])
        if bad.any():
            raise CalendarError(f"{int(bad.sum())} bars outside the configured sessions, "
                                f"first {pd.DatetimeIndex(index)[bad][0]}")
        return pos

    def segment_of(self, t) -> int:
        v = _ts(t).value
        pos = int(np.searchsorted(self._open, v, side="right") - 1)
        if pos < 0 or v >= self._close[pos]:
            return -1
        return pos


def load_calendar(cfg: Mapping) -> Calendar:
    for key in ("calendar_id", "version", "segments", "entry_delay_after_open_hours",
                "force_flat_before_close_hours", "entry_cutoff_before_flat_hours"):
        if key not in cfg:
            raise CalendarError(f"calendar config needs explicit {key}")
    delay = float(cfg["entry_delay_after_open_hours"])
    flat = float(cfg["force_flat_before_close_hours"])
    cutoff = float(cfg["entry_cutoff_before_flat_hours"])
    if min(delay, flat, cutoff) < 0:
        raise CalendarError("offsets must be non-negative")
    segs = []
    for raw in cfg["segments"]:
        o, c = _ts(raw["market_open"]), _ts(raw["market_close"])
        s, f = o + delay * HOUR, c - flat * HOUR
        x = f - cutoff * HOUR
        if not o < c:
            raise CalendarError(f"{raw['segment_id']}: open must precede close")
        segs.append(Segment(raw["segment_id"], o, c, s, f, min(max(x, s), f)))
    for a, b in zip(segs, segs[1:]):
        if not a.market_close <= b.market_open:
            raise CalendarError(f"{a.segment_id} overlaps {b.segment_id}")
    if len({s.segment_id for s in segs}) != len(segs):
        raise CalendarError("duplicate segment_id")
    return Calendar(segs, cfg)
