"""Read-only OHLC loading with the TEST boundary enforced at read time."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")


class TestRowsError(ValueError):
    """A request or a frame would touch TEST rows (2025-01-01 and later)."""


def _utc(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ohlc(path, start, end_exclusive) -> pd.DataFrame:
    """Load ``[start, end_exclusive)`` of a lake OHLC parquet, filtered at read.

    Refuses any window reaching TEST. Columns are renamed to OPEN/HIGH/LOW/CLOSE
    and the frame is validated (unique sorted whole-hour labels, finite, coherent
    ranges). No row is created, shifted or filled.
    """
    start, end = _utc(start), _utc(end_exclusive)
    if end > TEST_START:
        raise TestRowsError(f"window end {end} reaches TEST rows (>= {TEST_START})")
    raw = pd.read_parquet(Path(path), filters=[("datetime", ">=", start), ("datetime", "<", end)])
    frame = raw.rename(columns={"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE"})
    idx = pd.DatetimeIndex(frame["datetime"])
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    frame = frame[["OPEN", "HIGH", "LOW", "CLOSE"]].astype(float)
    frame.index = idx.rename("DATE_TIME")
    if len(frame) and frame.index.max() >= TEST_START:
        raise TestRowsError("TEST rows present after filtering")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("bar labels must be unique and increasing")
    if ((frame.index.minute != 0) | (frame.index.second != 0)).any():
        raise ValueError("only whole-hour bar labels are supported")
    v = frame.to_numpy()
    if not np.isfinite(v).all():
        raise ValueError("non-finite OHLC")
    if ((frame.HIGH < frame[["OPEN", "CLOSE"]].max(axis=1)) |
            (frame.LOW > frame[["OPEN", "CLOSE"]].min(axis=1))).any():
        raise ValueError("incoherent OHLC range")
    return frame
