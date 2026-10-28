"""Vendored copy of predictor tools/timing_event_clock.py (2026-10-21, commit 95863c88) for a single decision instant:
hours since / to the London 16:00 fix and to the New York 10:00 cut (DST per zone). Known in advance."""
from __future__ import annotations

import pandas as pd


def _since_to(decision_utc: pd.Timestamp, tz: str, hh: int, mm: int):
    loc = decision_utc.tz_convert(tz)
    today = pd.Timestamp(loc.tz_localize(None).normalize() + pd.Timedelta(hours=hh, minutes=mm)).tz_localize(tz)
    last = today if loc >= today else pd.Timestamp(today.tz_localize(None) - pd.Timedelta(days=1)).tz_localize(tz)
    nxt = pd.Timestamp(last.tz_localize(None) + pd.Timedelta(days=1)).tz_localize(tz)
    return (decision_utc - last).total_seconds() / 3600, (nxt - decision_utc).total_seconds() / 3600


def event_clock_at(decision_utc: pd.Timestamp) -> dict:
    ls, lt = _since_to(decision_utc, "Europe/London", 16, 0)
    _, nt = _since_to(decision_utc, "America/New_York", 10, 0)
    return {"lsince": ls, "lto": lt, "nyto": nt}
