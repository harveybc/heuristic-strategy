"""Causal event-window strategy: fixture readers, exporter clock table, surprise scaling, family
estimates, event blocks and orders (DEVELOPMENT, FIXTURE_RESEARCH_ONLY inputs).

The economic-calendar fixture is an Investing.com export: research only, never published, never
fed to anything live; this module returns frames in memory and writes nothing. Its clock is NOT a
legal DST rule but an exporter artifact measured by the data lane on 1,571 anchor rows: fixed
UTC-5 from 2011-01 to 2018-01, then alternating UTC-4 / UTC-5 windows with the switch dates in
``EXPORTER_OFFSET_SWITCHES``. The ``actual_text`` column encodes good/bad, not the sign of the
surprise, so only the numeric surprise is used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .blocks import AMBIGUOUS, Blocks, OK
from .calendar import Calendar, HOUR
from .data import TEST_START, TestRowsError
from .execution import BlockOrder, RealisticProfile
from .vstrategies import Leg, OcoOrder

COLUMNS = ["date", "time", "country", "impact", "event", "actual_text", "unit", "actual", "forecast", "previous"]
CURRENCY = {"United States": "USD", "Euro Zone": "EUR", "United Kingdom": "GBP", "Japan": "JPY",
            "Switzerland": "CHF", "Canada": "CAD", "Australia": "AUD", "New Zealand": "NZD", "Germany": "EUR",
            "France": "EUR", "Italy": "EUR", "Spain": "EUR"}
IMPACT = {"High Volatility Expected": "High", "Moderate Volatility Expected": "Moderate",
          "Low Volatility Expected": "Low"}
# (switch date, hours to ADD to the exported local time to reach UTC) -- data-lane measurement, not DST
EXPORTER_OFFSET_SWITCHES = (("2011-01-01", 5), ("2018-02-28", 4), ("2018-09-12", 5), ("2019-02-01", 4),
                            ("2019-11-01", 5), ("2020-02-03", 4), ("2020-09-01", 5), ("2021-02-03", 4))
CLOCK_STATUS = "CLOCK_DETECTED_NOT_VERIFIED"
FIXTURE_LABEL = "FIXTURE_RESEARCH_ONLY"


# ------------------------------------------------------------------ fixture readers
def exporter_offset_hours(day) -> int:
    d = pd.Timestamp(day).tz_localize(None).normalize()
    off = EXPORTER_OFFSET_SWITCHES[0][1]
    for start, hours in EXPORTER_OFFSET_SWITCHES:
        if d >= pd.Timestamp(start):
            off = hours
    return off


def parse_calendar_fixture(path_or_buffer) -> pd.DataFrame:
    """Rows of the Investing.com export with the release instant in UTC via the exporter table.

    Refuses any TEST row (2025-01-01 and later). Numeric actual / forecast / previous only; the
    textual columns are dropped except the family (event name), country and impact.
    """
    raw = pd.read_csv(path_or_buffer, header=None, names=COLUMNS, dtype=str, on_bad_lines="skip",
                      keep_default_na=False)
    for c in ("country", "impact", "event", "actual", "forecast", "previous"):
        raw[c] = raw[c].str.strip()
    local = pd.to_datetime(raw["date"] + " " + raw["time"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    ok = local.notna()
    raw, local = raw[ok].reset_index(drop=True), local[ok].reset_index(drop=True)
    off = np.array([exporter_offset_hours(d) for d in local], dtype="int64")
    release = (local + pd.to_timedelta(off, unit="h")).dt.tz_localize("UTC")
    if (release >= TEST_START).any():
        raise TestRowsError("calendar fixture holds TEST rows (>= 2025-01-01)")
    num = lambda s: pd.to_numeric(s.str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")  # noqa
    out = pd.DataFrame({"release_utc": release, "country": raw["country"],
                        "currency": raw["country"].map(CURRENCY).fillna("OTHER"),
                        "impact": raw["impact"].map(IMPACT).fillna("Unknown"), "family": raw["event"],
                        "actual": num(raw["actual"]), "forecast": num(raw["forecast"]),
                        "previous": num(raw["previous"])})
    return out.sort_values("release_utc", kind="stable").reset_index(drop=True)


def parse_hourly_fixture(path) -> pd.DataFrame:
    """The hourly OHLC fixture as LABELLED (clock unverified); call ``detect_clock_offset`` next."""
    raw = pd.read_csv(path)
    idx = pd.DatetimeIndex(pd.to_datetime(raw["datetime"], dayfirst=True, format="mixed"))
    frame = raw.rename(columns={"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE"})
    frame = frame[["OPEN", "HIGH", "LOW", "CLOSE"]].astype(float)
    frame.index = idx.rename("DATE_TIME")
    frame = frame[~frame.index.duplicated()].sort_index()
    if frame.index.max() >= TEST_START.tz_localize(None):
        raise TestRowsError("hourly fixture holds TEST rows")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("bar labels must increase")
    return frame


def detect_clock_offset(bars_labelled: pd.DataFrame, releases_utc) -> dict:
    """Offset (hours) of the bar labels ahead of UTC, from the hour with the largest mean |close - open|
    on the release days. Status is always CLOCK_DETECTED_NOT_VERIFIED."""
    idx = bars_labelled.index
    idx = idx.tz_localize(None) if idx.tz is not None else idx
    move = (bars_labelled.CLOSE - bars_labelled.OPEN).abs().to_numpy()
    days = {pd.Timestamp(r).tz_convert("UTC").tz_localize(None).normalize() for r in releases_utc}
    hours = {pd.Timestamp(r).tz_convert("UTC").hour for r in releases_utc}
    if len(hours) != 1:
        raise ValueError("clock detection needs releases at one UTC hour")
    release_hour = hours.pop()
    on_day = np.array([d in days for d in idx.normalize()])
    by_hour = pd.Series(move[on_day]).groupby(idx.hour[on_day]).mean()
    if by_hour.empty:
        raise ValueError("no bars on the release days")
    peak = int(by_hour.idxmax())
    rest = by_hour.drop(peak).mean()
    return {"offset_hours": int((peak - release_hour + 12) % 24 - 12), "peak_label_hour": peak,
            "release_hour_utc": int(release_hour), "days": int(len(days)),
            "vol_ratio_peak_vs_other_hours": float(by_hour[peak] / rest) if rest > 0 else math.inf,
            "status": CLOCK_STATUS}


def bars_to_utc(bars_labelled: pd.DataFrame, offset_hours: int) -> pd.DataFrame:
    out = bars_labelled.copy()
    idx = out.index.tz_localize(None) if out.index.tz is not None else out.index
    out.index = (idx - offset_hours * HOUR).tz_localize("UTC").rename("DATE_TIME")
    return out


# ------------------------------------------------------------------ surprise
def universe(events: pd.DataFrame, currencies=("USD", "EUR"), impacts=("High",)) -> pd.DataFrame:
    m = events.currency.isin(currencies) & events.impact.isin(impacts)
    m &= np.isfinite(events.actual) & np.isfinite(events.forecast)
    return events[m].reset_index(drop=True)


def surprise(events: pd.DataFrame, min_prior: int = 6) -> pd.DataFrame:
    """Standardized surprise with an expanding robust scale (1.4826 MAD of actual - forecast) fitted on
    PRIOR releases of the same family only; NaN until ``min_prior`` prior releases exist."""
    out = events.sort_values("release_utc", kind="stable").reset_index(drop=True).copy()
    raw = (out.actual - out.forecast).to_numpy(float)
    scale = np.full(len(out), np.nan)
    for fam, idx in out.groupby("family").indices.items():
        for j, i in enumerate(idx):
            prior = raw[idx[:j]]
            prior = prior[np.isfinite(prior)]
            if len(prior) >= min_prior:
                mad = 1.4826 * np.median(np.abs(prior - np.median(prior)))
                if mad <= 0:
                    mad = prior.std(ddof=1)
                scale[i] = mad if mad > 0 else np.nan
    out["scale"] = scale
    out["surprise"] = raw / scale
    return out


# ------------------------------------------------------------------ post-release windows
def event_windows(bars: pd.DataFrame, events: pd.DataFrame, max_window: int = 12) -> pd.DataFrame:
    """Per event: release bar, entry bar (next bar), forward drifts drift_h = close[entry+h-1] - open[entry]
    for h = 1..bars_available (contiguous bars after the release, at most max_window, truncated at a
    session gap), realized vol over each window, per-event dissipation half-life (first h >= 1 with
    |ret_h| <= 0.5 |ret_release|, capped at the available bars) and the release-bar jump. Events with a
    missing release bar or fewer than two following bars get NaN."""
    v = bars.index.as_unit("ns").asi8
    o, c = bars.OPEN.to_numpy(float), bars.CLOSE.to_numpy(float)
    ret = np.r_[np.nan, np.diff(c)]
    rows = []
    for r in events.release_utc:
        r_ns = pd.Timestamp(r).tz_convert("UTC").floor("h").value
        k = int(np.searchsorted(v, r_ns, side="left"))
        rec = {"release_bar": -1, "entry_bar": -1, "jump": np.nan, "half_life": np.nan, "bars_available": 0}
        for h in range(1, max_window + 1):
            rec[f"drift_{h}"] = np.nan
            rec[f"realized_vol_{h}"] = np.nan
        if k < len(v) and v[k] == r_ns:
            # contiguous bars after the release bar, truncated at a session gap (e.g. the Friday close)
            avail = 0
            while k + avail + 1 < len(v) and avail < max_window and v[k + avail + 1] - v[k] == (avail + 1) * HOUR.value:
                avail += 1
            if avail >= 2:
                e = k + 1
                rec.update({"release_bar": k, "entry_bar": e, "jump": c[k] - o[k], "bars_available": avail})
                for h in range(1, avail + 1):
                    rec[f"drift_{h}"] = c[e + h - 1] - o[e]
                    rec[f"realized_vol_{h}"] = float(np.sqrt(np.sum(ret[e:e + h] ** 2)))
                j0 = abs(ret[k]) if np.isfinite(ret[k]) else abs(c[k] - o[k])
                hl = avail
                for h in range(1, avail + 1):
                    if abs(ret[k + h]) <= 0.5 * j0:
                        hl = h
                        break
                rec["half_life"] = float(hl)
        rows.append(rec)
    w = pd.DataFrame(rows)
    return pd.concat([events.reset_index(drop=True), w], axis=1)


# ------------------------------------------------------------------ family estimates
def _slope_through_origin(s, y):
    s, y = np.asarray(s, float), np.asarray(y, float)
    n = len(s)
    if n < 3 or np.sum(s * s) <= 0:
        return np.nan, np.nan, n
    b = float(np.sum(s * y) / np.sum(s * s))
    resid = y - b * s
    se = math.sqrt(np.sum(resid ** 2) / (n - 1) / np.sum(s * s))
    if se <= 0:
        return b, 0.0, n
    from scipy import stats
    p = float(2 * stats.t.sf(abs(b / se), df=n - 1))
    return b, p, n


def fit_families(windows: pd.DataFrame, window: int, decision_end, min_n: int = 20, alpha: float = 0.05):
    """Per family, on rows strictly before ``decision_end``: slope of drift_window on the surprise
    (through the origin), its p-value, the sign map and the take/skip gate (tradable iff n >= min_n and
    p < alpha). Also the mean |drift| and the mean half-life (fallbacks)."""
    col = f"drift_{window}"
    d = windows[(windows.release_utc < pd.Timestamp(decision_end)) & np.isfinite(windows.surprise)
                & np.isfinite(windows[col])]
    rows = {}
    for fam, g in d.groupby("family"):
        b, p, n = _slope_through_origin(g.surprise, g[col])
        rows[fam] = {"n": n, "slope": b, "p_value": p, "sign": int(np.sign(b)) if np.isfinite(b) and b != 0 else 0,
                     "tradable": bool(n >= min_n and np.isfinite(p) and p < alpha and b != 0),
                     "mean_abs_drift": float(np.mean(np.abs(g[col]))),
                     "mean_half_life": float(np.nanmean(g.half_life)) if g.half_life.notna().any() else np.nan}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("family")


def expanding_family_mean(windows: pd.DataFrame, col: str, min_prior: int = 6) -> np.ndarray:
    """Causal expected value per event: mean of ``col`` over PRIOR events of the same family."""
    out = np.full(len(windows), np.nan)
    vals = windows[col].to_numpy(float)
    for fam, idx in windows.groupby("family").indices.items():
        for j, i in enumerate(idx):
            prior = vals[idx[:j]]
            prior = prior[np.isfinite(prior)]
            if len(prior) >= min_prior:
                out[i] = prior.mean()
    return out


def rolling_prior_mean(windows: pd.DataFrame, col: str, n: int = 20) -> np.ndarray:
    """MEAN20-style naive: mean of ``col`` over the last ``n`` prior events of ANY family."""
    vals = windows[col].to_numpy(float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        prior = vals[max(0, i - n):i]
        prior = prior[np.isfinite(prior)]
        if len(prior) >= min(n, 5):
            out[i] = prior.mean()
    return out


# ------------------------------------------------------------------ blocks and orders
def _blocks(bars, cal, lo_arr, time_stop, ref_bar_arr):
    v = bars.index
    v_ns = v.as_unit("ns").asi8
    seg_of = cal.assign(v)
    close = bars.CLOSE.to_numpy(float)
    high, low = bars.HIGH.to_numpy(float), bars.LOW.to_numpy(float)
    start, end, seg, blo, bhi, rbar, ref, H, L, ft, status = ([] for _ in range(11))
    for lo, ts, rb in zip(lo_arr, time_stop, ref_bar_arr):
        lo, rb, ts = int(lo), int(rb), int(ts)
        st = OK
        if lo < 1 or lo >= len(v) or rb < 0:
            st, si = "NO_REFERENCE", 0
            s, e, hi = v[0], v[0] + HOUR, 1
        else:
            si = int(seg_of[lo])
            s = v[lo]
            fflat = cal.segments[si].force_flat
            e = min(s + max(ts, 1) * HOUR, fflat)
            hi = int(np.searchsorted(v_ns, (e - HOUR).value, side="right"))
            if hi <= lo or s >= cal.segments[si].entry_cutoff or seg_of[rb] != si or e <= s:
                st = "INSUFFICIENT_BARS"
        start.append(s), end.append(e), seg.append(si), blo.append(lo), bhi.append(max(hi, lo + 1))
        rbar.append(rb), ref.append(close[rb] if st == OK else np.nan)
        H.append(high[lo:hi].max() if st == OK else np.nan), L.append(low[lo:hi].min() if st == OK else np.nan)
        ft.append(AMBIGUOUS), status.append(st)
    return Blocks(0, start, end, np.array(seg, int), np.array(blo, int), np.array(bhi, int), np.array(rbar, int),
                  np.array(ref, float), np.array(H, float), np.array(L, float), np.array(ft, int),
                  np.array(status, object))


def event_blocks(bars, cal: Calendar, windows: pd.DataFrame, time_stop) -> Blocks:
    """One block per event starting at the entry bar (next bar after the release), ending at the time
    stop or the forced flat, whichever comes first. Reference = release-bar close (first available quote)."""
    return _blocks(bars, cal, windows.entry_bar.to_numpy(int), np.asarray(time_stop), windows.release_bar.to_numpy(int))


def latency_zero_blocks(bars, cal, windows, time_stop) -> Blocks:
    """ORACLE bound: entry at the release-bar open (as if the surprise were known with zero latency)."""
    rb = windows.release_bar.to_numpy(int)
    return _blocks(bars, cal, rb, np.asarray(time_stop) + 1, rb - 1)


def straddle_blocks(bars, cal, windows, time_stop) -> Blocks:
    """Straddle arm: the block opens on the release bar; reference = close of the bar before it."""
    rb = windows.release_bar.to_numpy(int)
    return _blocks(bars, cal, rb, np.asarray(time_stop) + 1, rb - 1)


def directional_orders(blocks: Blocks, direction, tp_distance, sl_distance, size):
    out = []
    for k in range(len(blocks)):
        d, tp, sl, ps = int(direction[k]), float(tp_distance[k]), float(sl_distance[k]), float(size[k])
        if blocks.status[k] != OK or d == 0 or not all(map(np.isfinite, (tp, sl, ps))) or tp <= 0 or sl <= 0 or ps <= 0:
            out.append(None)
            continue
        r = float(blocks.ref[k])
        out.append(BlockOrder(d, "MARKET", None, r + d * tp, r - d * sl, None, min(ps, 1.0)))
    return out


def straddle_orders(blocks: Blocks, entry_distance, tp_distance, sl_distance, size):
    out = []
    for k in range(len(blocks)):
        e, tp, sl, ps = (float(x[k]) for x in (entry_distance, tp_distance, sl_distance, size))
        if blocks.status[k] != OK or not all(map(np.isfinite, (e, tp, sl, ps))) or min(e, tp, sl, ps) <= 0:
            out.append(None)
            continue
        r = float(blocks.ref[k])
        out.append(OcoOrder((Leg(1, "STOP", r + e, tp, sl, None, True), Leg(-1, "STOP", r - e, tp, sl, None, True)),
                            None, min(ps, 1.0)))
    return out


def release_bar_labels(releases_utc, hours: int = 2) -> frozenset:
    """Bar labels (ns) of the release bar and the following ``hours - 1`` bars."""
    out = set()
    for r in releases_utc:
        base = pd.Timestamp(r).tz_convert("UTC").floor("h").value
        for h in range(hours):
            out.add(base + h * HOUR.value)
    return frozenset(out)


@dataclass(frozen=True)
class EventProfile(RealisticProfile):
    """Realistic profile with an event-widened spread (ASSUMED) on the release bar and the next one, and
    no extra entry delay (the event block already starts on the bar after the release)."""
    entry_delay_bars: int = 0
    event_spread: float = 0.0
    release_bars: frozenset = frozenset()

    def spread_at(self, t_ns: int, open_ns: int) -> float:
        if self.event_spread > 0 and int(t_ns) in self.release_bars:
            return self.event_spread
        return super().spread_at(t_ns, open_ns)

    def scaled(self, m: float) -> "EventProfile":
        base = RealisticProfile.scaled(self, m)
        return replace(base, event_spread=self.event_spread * m)


def causal_event_sigma(blocks: Blocks, n: int = 20) -> np.ndarray:
    """Mean realized range of the last ``n`` completed event blocks (any family), for the order rules."""
    rng = blocks.H - blocks.L
    out = np.full(len(blocks), np.nan)
    for k in range(len(blocks)):
        prior = rng[max(0, k - n):k]
        prior = prior[np.isfinite(prior)]
        if len(prior) >= 5:
            out[k] = prior.mean()
    return out
