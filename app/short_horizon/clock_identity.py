"""Pinned clock identity for vendor bar labels (EURUSD H1 lake) and its witness against a UTC root.

The lake file stores naive wall-clock labels under a ``UTC`` type tag; the tag is not the clock. A
clock identity maps every label to a UTC bar START through a declared tz-database rule (``zoneinfo``),
never through a hand-applied hour shift, and handles the two DST transition weeks explicitly:

* a label whose wall time does not exist (spring forward) is ``NONEXISTENT``;
* a label whose wall time occurs twice (fall back) is ``AMBIGUOUS``.

Both raise by default. For every rule declared here all transitions fall inside the FX weekend
closure, so a bar carrying such a label is a defect, not a choice.

The witness compares the mapped lake with an admitted UTC root keyed by
``(asset, utc_bar_start, bar_convention, source)``; OHLC agreement is supporting evidence only
(audit F04). A residual mismatch is classified CLOCK when the lake bar equals an admitted bar at
another hour, SOURCE otherwise. Only aggregates leave this module's witness.

CLI (writes CLOCK_IDENTITY.json, WITNESS.json, WITNESS_TABLE.md)::

    FINANCIAL_DATA_ROOT=... ADMITTED_ROOT=... python -m app.short_horizon.clock_identity \\
        --config config/short_horizon/eurusd_h1_bar_sources.json --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import TZPATH, ZoneInfo

import numpy as np
import pandas as pd

from app.short_horizon import data as D

SCHEMA = "heuristic_strategy.clock_identity.v1"
WITNESS_SCHEMA = "heuristic_strategy.clock_witness.v1"
LABEL_BAR_START = "LABEL_IS_BAR_START_IN_RULE_CLOCK"
UTC_BAR_START = "UTC_BAR_START"
AVAILABILITY = "UTC_BAR_START_PLUS_ONE_HOUR"
MAPPING_ENCODING = ("sha256(b'clock_identity.mapping.v1\\x00' + label_ns.int64_le + utc_ns.int64_le); "
                    "label_ns = naive label as datetime64[ns]; utc_ns = UTC bar start as datetime64[ns]")
OK, NONEXISTENT, AMBIGUOUS = "OK", "NONEXISTENT", "AMBIGUOUS"
OHLC = ["OPEN", "HIGH", "LOW", "CLOSE"]
_UTC = timezone.utc


class ClockError(ValueError):
    """A label cannot be mapped under the declared rule and policy."""


@dataclass(frozen=True)
class ClockRule:
    rule_id: str
    tz: str
    label_minus_wall_hours: int
    description: str

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)


RULES = {
    "AMERICA_NEW_YORK_WALL": ClockRule(
        "AMERICA_NEW_YORK_WALL", "America/New_York", 0,
        "label = America/New_York wall-clock time of the bar start (UTC-5 / UTC-4 on the US DST calendar)"),
    "EU_DST_UTC_MINUS_5": ClockRule(
        "EU_DST_UTC_MINUS_5", "Europe/London", -5,
        "label = Europe/London wall-clock time of the bar start minus 5 h: UTC-5 outside EU summer time, "
        "UTC-4 during it (EU transition instants, 01:00 UTC on the last Sundays of March and October)"),
    "FIXED_UTC_MINUS_5": ClockRule(
        "FIXED_UTC_MINUS_5", "Etc/GMT+5", 0,
        "label = UTC-5 all year, no DST (the vendor's documented convention)"),
}

# Computed from the tz database by ``transitions(rule, 2019, 2025)`` and pinned; a test asserts the
# installed tz database still produces exactly these instants (US rule since 2007, EU rule since 1996).
PINNED_TRANSITIONS = {
    "AMERICA_NEW_YORK_WALL": [
        {"utc": f"{d}T{h}:00:00+00:00", "label_offset_before_hours": b, "label_offset_after_hours": a}
        for d, h, b, a in [
            ("2019-03-10", "07", -5, -4), ("2019-11-03", "06", -4, -5),
            ("2020-03-08", "07", -5, -4), ("2020-11-01", "06", -4, -5),
            ("2021-03-14", "07", -5, -4), ("2021-11-07", "06", -4, -5),
            ("2022-03-13", "07", -5, -4), ("2022-11-06", "06", -4, -5),
            ("2023-03-12", "07", -5, -4), ("2023-11-05", "06", -4, -5),
            ("2024-03-10", "07", -5, -4), ("2024-11-03", "06", -4, -5),
            ("2025-03-09", "07", -5, -4), ("2025-11-02", "06", -4, -5)]],
    "EU_DST_UTC_MINUS_5": [
        {"utc": f"{d}T01:00:00+00:00", "label_offset_before_hours": b, "label_offset_after_hours": a}
        for d, b, a in [
            ("2019-03-31", -5, -4), ("2019-10-27", -4, -5),
            ("2020-03-29", -5, -4), ("2020-10-25", -4, -5),
            ("2021-03-28", -5, -4), ("2021-10-31", -4, -5),
            ("2022-03-27", -5, -4), ("2022-10-30", -4, -5),
            ("2023-03-26", -5, -4), ("2023-10-29", -4, -5),
            ("2024-03-31", -5, -4), ("2024-10-27", -4, -5),
            ("2025-03-30", -5, -4), ("2025-10-26", -4, -5)]],
    "FIXED_UTC_MINUS_5": [],
}


def rule(rule_id) -> ClockRule:
    if isinstance(rule_id, ClockRule):
        return rule_id
    if rule_id not in RULES:
        raise ClockError(f"UNKNOWN_CLOCK_RULE: {rule_id!r}; declared rules are {sorted(RULES)}")
    return RULES[rule_id]


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def tzdata_version(tz: str = "America/New_York") -> dict:
    """Version of the tz database that ``zoneinfo`` actually loads for ``tz`` (no paths recorded)."""
    for root in TZPATH:
        if os.path.isfile(os.path.join(root, *tz.split("/"))):
            zi = os.path.join(root, "tzdata.zi")
            version = None
            if os.path.isfile(zi):
                with open(zi, encoding="ascii", errors="replace") as f:
                    first = f.readline().strip()
                if first.startswith("# version"):
                    version = first.split()[-1]
            return {"version": version or "UNKNOWN", "provider": "zoneinfo.TZPATH system tz database"}
    try:
        import tzdata  # noqa: F401
        return {"version": getattr(tzdata, "IANA_VERSION", "UNKNOWN"), "provider": "tzdata python package"}
    except ImportError:  # pragma: no cover
        return {"version": "UNKNOWN", "provider": "UNKNOWN"}


# ----------------------------------------------------------------------------- label <-> UTC
def _naive_labels(labels) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(labels)
    return idx.tz_localize(None) if idx.tz is not None else idx


def _resolve(wall: datetime, zone: ZoneInfo):
    a = wall.replace(tzinfo=zone, fold=0).astimezone(_UTC)
    b = wall.replace(tzinfo=zone, fold=1).astimezone(_UTC)
    if a == b:
        return OK, a, a
    if a.astimezone(zone).replace(tzinfo=None) == wall:
        return AMBIGUOUS, min(a, b), max(a, b)
    return NONEXISTENT, None, None


def label_to_utc(labels, rule_id, nonexistent: str = "raise", ambiguous: str = "raise"):
    """Map naive labels to UTC bar starts. Returns ``(DatetimeIndex[UTC], status ndarray)``.

    ``nonexistent``: ``raise`` | ``flag`` (NaT). ``ambiguous``: ``raise`` | ``flag`` | ``earlier`` | ``later``.
    """
    r = rule(rule_id)
    if nonexistent not in ("raise", "flag") or ambiguous not in ("raise", "flag", "earlier", "later"):
        raise ClockError(f"UNKNOWN_TRANSITION_POLICY: nonexistent={nonexistent!r} ambiguous={ambiguous!r}")
    idx = _naive_labels(labels)
    zone, shift = r.zone, timedelta(hours=r.label_minus_wall_hours)
    out, status, cache = [], [], {}
    for ts in idx:
        if ts is pd.NaT:
            raise ClockError("NaT label")
        key = ts.value
        if key not in cache:
            cache[key] = _resolve(ts.to_pydatetime() - shift, zone)
        st, early, late = cache[key]
        if st == NONEXISTENT:
            if nonexistent == "raise":
                raise ClockError(f"NONEXISTENT label {ts} under {r.rule_id} ({r.tz})")
            out.append(pd.NaT)
        elif st == AMBIGUOUS:
            if ambiguous == "raise":
                raise ClockError(f"AMBIGUOUS label {ts} under {r.rule_id} ({r.tz})")
            out.append(pd.NaT if ambiguous == "flag" else (early if ambiguous == "earlier" else late))
        else:
            out.append(early)
        status.append(st)
    utc = pd.DatetimeIndex(out, tz="UTC") if out else pd.DatetimeIndex([], tz="UTC")
    return utc.as_unit("ns"), np.asarray(status, dtype=object)


def utc_to_label(utc, rule_id):
    """Inverse map: UTC instants -> ``(naive labels, fold ndarray)``; total (every instant has a label)."""
    r = rule(rule_id)
    zone, shift = r.zone, timedelta(hours=r.label_minus_wall_hours)
    idx = pd.DatetimeIndex(utc)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    labels, folds = [], []
    for ts in idx:
        w = ts.to_pydatetime().astimezone(zone)
        labels.append(w.replace(tzinfo=None) + shift)
        folds.append(w.fold)
    return pd.DatetimeIndex(labels).as_unit("ns"), np.asarray(folds, dtype=np.int64)


def label_to_utc_with_fold(labels, fold, rule_id) -> pd.DatetimeIndex:
    r = rule(rule_id)
    zone, shift = r.zone, timedelta(hours=r.label_minus_wall_hours)
    out = []
    for ts, f in zip(_naive_labels(labels), np.asarray(fold)):
        wall = ts.to_pydatetime() - shift
        st, _, _ = _resolve(wall, zone)
        if st == NONEXISTENT:
            raise ClockError(f"NONEXISTENT label {ts} under {r.rule_id}")
        out.append(wall.replace(tzinfo=zone, fold=int(f)).astimezone(_UTC))
    return pd.DatetimeIndex(out, tz="UTC").as_unit("ns")


def label_offset_hours(utc, rule_id) -> np.ndarray:
    """Signed label - UTC offset in hours for each UTC instant (e.g. -5 or -4)."""
    labels, _ = utc_to_label(utc, rule_id)
    idx = pd.DatetimeIndex(utc)
    idx = idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx
    return ((labels - idx) / pd.Timedelta(hours=1)).to_numpy().astype(int)


def transitions(rule_id, first_year: int, last_year: int) -> list:
    r = rule(rule_id)
    hours = pd.date_range(f"{first_year}-01-01", f"{last_year + 1}-01-01", freq="h", tz="UTC", inclusive="left")
    off = label_offset_hours(hours, r)
    ch = np.flatnonzero(np.diff(off) != 0) + 1
    return [{"utc": hours[i].isoformat(), "label_offset_before_hours": int(off[i - 1]),
             "label_offset_after_hours": int(off[i])} for i in ch]


def label_bounds(rule_id, start, end_exclusive):
    """Label window whose mapped UTC lies in ``[start, end)`` (the mapping is monotone)."""
    s, e = D._utc(start), D._utc(end_exclusive)
    lab, _ = utc_to_label(pd.DatetimeIndex([s, e]), rule_id)
    return lab[0], lab[1]


# ----------------------------------------------------------------------------- identity record
def mapping_digest(labels, utc) -> str:
    lab = _naive_labels(labels).as_unit("ns").asi8.astype("<i8")
    u = pd.DatetimeIndex(utc)
    u = (u.tz_convert("UTC").tz_localize(None) if u.tz is not None else u).as_unit("ns").asi8.astype("<i8")
    if len(lab) != len(u):
        raise ClockError("labels and utc differ in length")
    return hashlib.sha256(b"clock_identity.mapping.v1\x00" + lab.tobytes() + u.tobytes()).hexdigest()


def identity_record(rule_id, source_sha256: str, labels, utc, window, policy=None, *, asset="EURUSD", bar="1h",
                    transition_years=(2019, 2025), extra=None) -> dict:
    r = rule(rule_id)
    policy = policy or {"nonexistent": "raise", "ambiguous": "raise"}
    tr = transitions(r, *transition_years)
    u = pd.DatetimeIndex(utc)
    offs = pd.Series(label_offset_hours(u, r)).value_counts().sort_index() if len(u) else pd.Series(dtype=int)
    rec = {
        "schema": SCHEMA,
        "asset": asset,
        "bar": bar,
        "source": {"sha256": source_sha256, "label_column": "datetime",
                   "label_semantics": "naive wall-clock label of the bar start; the file's UTC type tag is not the clock"},
        "rule": {"id": r.rule_id, "tz": r.tz, "label_minus_wall_hours": r.label_minus_wall_hours,
                 "description": r.description, "implementation": "zoneinfo (PEP 495 fold), no hand-applied shift"},
        "tzdata": tzdata_version(r.tz),
        "transitions": {"years": list(transition_years), "instants": tr, "sha256": canonical_sha256(tr),
                        "matches_pinned": tr == PINNED_TRANSITIONS.get(r.rule_id)},
        "bar_convention_in": LABEL_BAR_START,
        "bar_convention_out": UTC_BAR_START,
        "availability_out": AVAILABILITY,
        "transition_policy": dict(policy),
        "mapping": {"window_utc": [str(D._utc(window[0]).isoformat()), str(D._utc(window[1]).isoformat())],
                    "rows": int(len(u)),
                    "first_utc": u.min().isoformat() if len(u) else None,
                    "last_utc": u.max().isoformat() if len(u) else None,
                    "label_offset_hours_counts": {str(int(k)): int(v) for k, v in offs.items()},
                    "encoding": MAPPING_ENCODING,
                    "digest_sha256": mapping_digest(labels, utc)},
    }
    if extra:
        rec.update(extra)
    rec["identity_sha256"] = canonical_sha256(rec)
    return rec


# ----------------------------------------------------------------------------- reading
def _read_lake_labels(path, lo, hi) -> pd.DataFrame:
    raw = pd.read_parquet(Path(path), columns=["datetime", "open", "high", "low", "close"],
                          filters=[("datetime", ">=", lo.tz_localize("UTC")), ("datetime", "<", hi.tz_localize("UTC"))])
    labels = _naive_labels(raw["datetime"])
    frame = raw[["open", "high", "low", "close"]].astype(float)
    frame.columns = OHLC
    frame.index = labels.rename("LABEL")
    if len(frame) and not (frame.index.is_unique and frame.index.is_monotonic_increasing):
        raise ValueError("lake labels must be unique and increasing")
    return frame


def map_lake(path, start, end_exclusive, rule_id, nonexistent="raise", ambiguous="raise") -> pd.DataFrame:
    """Lake bars whose MAPPED UTC bar start lies in ``[start, end)``. Labels at or past the label of
    ``end`` are never read, so TEST rows (>= 2025-01-01 UTC) are not touched."""
    start, end = D._utc(start), D._utc(end_exclusive)
    if end > D.TEST_START:
        raise D.TestRowsError(f"window end {end} reaches TEST rows (>= {D.TEST_START})")
    lo, hi = label_bounds(rule_id, start, end)
    lake = _read_lake_labels(path, lo, hi)
    utc, status = label_to_utc(lake.index, rule_id, nonexistent=nonexistent, ambiguous=ambiguous)
    keep = (~pd.isna(utc)) & (utc >= start) & (utc < end)
    out = lake.loc[keep].copy()
    out.insert(len(OHLC), "LABEL", lake.index[keep])
    out.index = utc[keep].rename(UTC_BAR_START)
    out.attrs["status_counts"] = {k: int(v) for k, v in pd.Series(status).value_counts().items()}
    if len(out) and out.index.max() >= D.TEST_START:
        raise D.TestRowsError("TEST rows present after mapping")
    if not out.index.is_unique:
        raise ClockError("mapped UTC bar starts are not unique")
    return out


def read_admitted(path, start, end_exclusive, with_source=False) -> pd.DataFrame:
    """Admitted root rows with UTC bar start in ``[start, end)`` (read-time filter, TEST refused)."""
    start, end = D._utc(start), D._utc(end_exclusive)
    if end > D.TEST_START:
        raise D.TestRowsError(f"window end {end} reaches TEST rows (>= {D.TEST_START})")
    cols = ["timestamp", "available_at", "BidOpen", "BidHigh", "BidLow", "BidClose"] + (["source_id"] if with_source else [])
    raw = pd.read_parquet(Path(path), columns=cols, filters=[("timestamp", ">=", start), ("timestamp", "<", end)])
    idx = pd.DatetimeIndex(raw["timestamp"])
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    avail = pd.DatetimeIndex(raw["available_at"])
    avail = avail.tz_localize("UTC") if avail.tz is None else avail.tz_convert("UTC")
    if len(idx) and not ((avail - idx) == pd.Timedelta(hours=1)).all():
        raise ValueError("AVAILABILITY: admitted rows must be available exactly one hour after the bar start")
    frame = raw[["BidOpen", "BidHigh", "BidLow", "BidClose"]].astype(float)
    frame.columns = OHLC
    if with_source:
        frame["SOURCE_FAMILY"] = raw["source_id"].astype(str).str.split(":").str[0].to_numpy()
    frame.index = idx.rename(UTC_BAR_START)
    if len(frame) and frame.index.max() >= D.TEST_START:
        raise D.TestRowsError("TEST rows present after filtering")
    if not (frame.index.is_unique and frame.index.is_monotonic_increasing):
        raise ValueError("admitted bar starts must be unique and increasing")
    return frame


# ----------------------------------------------------------------------------- joined view
def joined_view(lake_path, admitted_path, start, end_exclusive, rule_id, *, asset="EURUSD",
                lake_sha256: str, admitted_sha256: str, policy=None) -> pd.DataFrame:
    """Long view keyed by (asset, utc_bar_start, bar_convention, source); one row per source bar."""
    policy = policy or {"nonexistent": "raise", "ambiguous": "raise"}
    r = rule(rule_id)
    lm = map_lake(lake_path, start, end_exclusive, r, **policy)[OHLC]
    ad = read_admitted(admitted_path, start, end_exclusive)
    parts = []
    for frame, src in ((ad, f"admitted:{admitted_sha256}"), (lm, f"lake_mapped:{r.rule_id}:{lake_sha256}")):
        p = frame.reset_index().rename(columns={UTC_BAR_START: "utc_bar_start"})
        p.insert(0, "asset", asset)
        p.insert(2, "bar_convention", UTC_BAR_START)
        p.insert(3, "source", src)
        parts.append(p)
    view = pd.concat(parts, ignore_index=True).sort_values(["utc_bar_start", "source"], kind="mergesort")
    return view.reset_index(drop=True)


def comparison_view(view: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """Wide view: one row per (asset, utc_bar_start, bar_convention), admitted vs lake_mapped side by side."""
    v = view.copy()
    v["side"] = np.where(v["source"].str.startswith("admitted:"), "admitted", "lake_mapped")
    wide = v.pivot_table(index=["asset", "utc_bar_start", "bar_convention"], columns="side", values=OHLC, aggfunc="first")
    wide.columns = [f"{side}_{c}" for c, side in wide.columns]
    for side in ("admitted", "lake_mapped"):
        for c in OHLC:
            if f"{side}_{c}" not in wide:
                wide[f"{side}_{c}"] = np.nan
    wide["in_admitted"] = wide["admitted_CLOSE"].notna()
    wide["in_lake_mapped"] = wide["lake_mapped_CLOSE"].notna()
    eq = np.ones(len(wide), bool)
    for c in OHLC:
        eq &= (np.abs(wide[f"admitted_{c}"] - wide[f"lake_mapped_{c}"]) <= tol).fillna(False).to_numpy()
    wide["ohlc_equal"] = eq & wide["in_admitted"].to_numpy() & wide["in_lake_mapped"].to_numpy()
    return wide.reset_index()


# ----------------------------------------------------------------------------- witness
def _all_equal(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (np.abs(a - b) <= tol).all(axis=1)


def _half(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.year.astype(str) + "H" + np.where(idx.month <= 6, "1", "2"), index=idx)


def _counts_by(keys: pd.Series) -> dict:
    return {str(k): int(v) for k, v in keys.value_counts().sort_index().items()}


def _closure_mask(utc: pd.DatetimeIndex, intervals) -> np.ndarray:
    if not intervals or not len(utc):
        return np.zeros(len(utc), bool)
    lo = pd.DatetimeIndex([pd.Timestamp(a) for a, _ in intervals]).tz_convert("UTC")
    hi = pd.DatetimeIndex([pd.Timestamp(b) for _, b in intervals]).tz_convert("UTC")
    pos = np.searchsorted(lo.asi8, utc.asi8, side="right") - 1
    ok = pos >= 0
    out = np.zeros(len(utc), bool)
    out[ok] = utc.asi8[ok] < hi.asi8[pos[ok]]
    return out


def _rule_witness(lake_path, adm: pd.DataFrame, start, end, r: ClockRule, tol, closure, gap_mask_fn, max_shift):
    lm = map_lake(lake_path, start, end, r, nonexistent="flag", ambiguous="flag")
    status = lm.attrs["status_counts"]
    L, A = lm[OHLC], adm[OHLC]
    joined_idx = L.index.intersection(A.index)
    lake_only_idx = L.index.difference(A.index)
    adm_only_idx = A.index.difference(L.index)
    lj, aj = L.loc[joined_idx].to_numpy(), A.loc[joined_idx].to_numpy()
    exact = _all_equal(lj, aj, tol)
    bitwise = (lj == aj).all(axis=1)

    def neighbour_match(idx):
        vals = L.loc[idx].to_numpy()
        hit = np.zeros(len(idx), bool)
        for k in range(-max_shift, max_shift + 1):
            if k == 0:
                continue
            other = A.reindex(idx + pd.Timedelta(hours=k)).to_numpy()
            hit |= _all_equal(vals, other, tol)
        return hit

    res_idx = joined_idx[~exact]
    res_clock = neighbour_match(res_idx) if len(res_idx) else np.zeros(0, bool)
    any_field = (np.abs(lj[~exact] - aj[~exact]) <= tol).any(axis=1) if len(res_idx) else np.zeros(0, bool)
    src_partial = (~res_clock) & any_field
    src_all = (~res_clock) & (~any_field)
    lo_clock = neighbour_match(lake_only_idx) if len(lake_only_idx) else np.zeros(0, bool)
    lo_closed = _closure_mask(lake_only_idx, closure)

    fam = adm["SOURCE_FAMILY"] if "SOURCE_FAMILY" in adm else None
    src_fam = {}
    if fam is not None and len(res_idx):
        src_res = res_idx[~res_clock]
        src_fam = _counts_by(fam.loc[src_res])

    # monthly table on the UTC key
    def month_counts(idx, mask=None):
        idx = idx if mask is None else idx[mask]
        return pd.Series(1, index=idx).groupby(idx.strftime("%Y-%m")).sum() if len(idx) else pd.Series(dtype=int)

    cols = {
        "lake_mapped": month_counts(L.index), "admitted": month_counts(A.index), "joined": month_counts(joined_idx),
        "ohlc_exact": month_counts(joined_idx, exact), "ohlc_bitwise": month_counts(joined_idx, bitwise),
        "residual_clock": month_counts(res_idx, res_clock),
        "residual_source_partial": month_counts(res_idx, src_partial),
        "residual_source_all": month_counts(res_idx, src_all),
        "lake_only": month_counts(lake_only_idx), "lake_only_clock": month_counts(lake_only_idx, lo_clock),
        "lake_only_in_admitted_closure": month_counts(lake_only_idx, lo_closed),
        "admitted_only": month_counts(adm_only_idx),
    }
    months = pd.period_range(D._utc(start).tz_localize(None), D._utc(end).tz_localize(None) - pd.Timedelta(hours=1), freq="M")
    table = pd.DataFrame({k: v for k, v in cols.items()}).reindex([str(m) for m in months]).fillna(0).astype(int)
    table["residual_source"] = table["residual_source_partial"] + table["residual_source_all"]
    table["exact_rate"] = (table["ohlc_exact"] / table["joined"].where(table["joined"] > 0)).round(6)
    by_year = table.drop(columns="exact_rate").groupby(table.index.str[:4]).sum()
    by_year["exact_rate"] = (by_year["ohlc_exact"] / by_year["joined"].where(by_year["joined"] > 0)).round(6)
    totals = {k: int(v) for k, v in table.drop(columns="exact_rate").sum().items()}
    totals["exact_rate"] = round(totals["ohlc_exact"] / totals["joined"], 6) if totals["joined"] else None
    totals["transition_nonexistent"] = int(status.get(NONEXISTENT, 0))
    totals["transition_ambiguous"] = int(status.get(AMBIGUOUS, 0))

    gm = gap_mask_fn(joined_idx)
    gap = {"joined_in_us_eu_gap": int(gm.sum()), "exact_in_us_eu_gap": int((exact & gm).sum()),
           "lake_mapped_in_us_eu_gap": int(gap_mask_fn(L.index).sum()),
           "admitted_in_us_eu_gap": int(gap_mask_fn(A.index).sum())}

    clock_res = totals["residual_clock"] + totals["lake_only_clock"]
    transition_bad = totals["transition_nonexistent"] + totals["transition_ambiguous"]
    if not totals["joined"] or totals["ohlc_exact"] + totals["residual_clock"] < 0.5 * totals["joined"]:
        verdict = "INCONCLUSIVE_SOURCES_DISAGREE"
    elif clock_res == 0 and transition_bad == 0:
        verdict = "EXACT_CLOCK_MAPPING"
    else:
        verdict = "CLOCK_RESIDUALS_PRESENT"
    to_rows = lambda t: {idx: {k: (None if pd.isna(v) else (float(v) if k == "exact_rate" else int(v)))
                               for k, v in row.items()} for idx, row in t.iterrows()}
    return {"rule": {"id": r.rule_id, "tz": r.tz, "label_minus_wall_hours": r.label_minus_wall_hours},
            "verdict": verdict, "totals": totals, "us_eu_gap_periods": gap,
            "residual_source_by_admitted_family": src_fam,
            "by_year": to_rows(by_year), "by_month": to_rows(table),
            "lake_mapped_index": L.index}


def witness(lake_path, admitted_path, start, end_exclusive, rules=None, *, closure_intervals=None, tol=1e-9,
            max_shift_hours=3, lake_sha256=None, admitted_sha256=None) -> dict:
    """Aggregate-only comparison of the mapped lake with the admitted UTC root, per candidate rule."""
    rules = [rule(x) for x in (rules or list(RULES))]
    start, end = D._utc(start), D._utc(end_exclusive)
    adm = read_admitted(admitted_path, start, end, with_source=True)
    has_fam = adm["SOURCE_FAMILY"].nunique() > 0
    if not has_fam:
        adm = adm.drop(columns="SOURCE_FAMILY")

    def gap_mask(idx):
        if not len(idx):
            return np.zeros(0, bool)
        return label_offset_hours(idx, "AMERICA_NEW_YORK_WALL") != label_offset_hours(idx, "EU_DST_UTC_MINUS_5")

    out = {"schema": WITNESS_SCHEMA, "window_utc": [start.isoformat(), end.isoformat()],
           "key": ["asset", "utc_bar_start", "bar_convention", "source"],
           "evidence_rule": "OHLC agreement is supporting evidence only (F04); the key is the declared clock mapping",
           "tolerance_abs": tol, "neighbour_hours_for_clock_class": max_shift_hours,
           "classes": {"ohlc_exact": "all four OHLC within tolerance on the joined key",
                       "residual_clock": "joined, not equal, but equal to the admitted bar at another hour (+-1..3 h)",
                       "residual_source_partial": "joined, not equal, at least one OHLC field equal, no neighbour match",
                       "residual_source_all": "joined, no OHLC field equal, no neighbour match",
                       "lake_only_clock": "lake key absent from admitted but its OHLC equals an admitted bar at another hour",
                       "lake_only_in_admitted_closure": "lake key absent from admitted and inside an admitted closure interval",
                       "admitted_only": "admitted key with no lake bar under this rule"},
           "sources": {"lake_sha256": lake_sha256, "admitted_sha256": admitted_sha256},
           "read_bounds": {}, "rules": {}, "coverage_by_half": {"admitted": _counts_by(_half(adm.index)), "lake_mapped": {}}}
    for r in rules:
        lo, hi = label_bounds(r, start, end)
        out["read_bounds"][r.rule_id] = {"lake_label_ge": lo.isoformat(), "lake_label_lt": hi.isoformat(),
                                         "admitted_timestamp_lt": end.isoformat()}
        res = _rule_witness(lake_path, adm, start, end, r, tol, closure_intervals, gap_mask, max_shift_hours)
        out["coverage_by_half"]["lake_mapped"][r.rule_id] = _counts_by(_half(res.pop("lake_mapped_index")))
        out["rules"][r.rule_id] = res
    exact = [rid for rid, v in out["rules"].items() if v["verdict"] == "EXACT_CLOCK_MAPPING"]
    out["selected_rule"] = max(exact, key=lambda rid: out["rules"][rid]["totals"]["ohlc_exact"]) if exact else None
    out["witness_sha256"] = canonical_sha256(out)
    return out


def witness_table_md(w: dict) -> str:
    lines = [f"# Clock witness (generated from WITNESS.json, sha {w['witness_sha256'][:12]})", "",
             f"Window (UTC bar start): {w['window_utc'][0]} to {w['window_utc'][1]} (exclusive). "
             f"Selected rule: **{w['selected_rule']}**.", "", "## Verdict per rule", "",
             "| rule | verdict | joined | OHLC exact | exact rate | clock residual | source residual | lake-only (clock) | admitted-only | transition labels | gap-week joined / exact |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rid, v in w["rules"].items():
        t, g = v["totals"], v["us_eu_gap_periods"]
        lines.append(f"| {rid} | {v['verdict']} | {t['joined']} | {t['ohlc_exact']} | {t['exact_rate']} | {t['residual_clock']} | "
                     f"{t['residual_source']} | {t['lake_only']} ({t['lake_only_clock']}) | {t['admitted_only']} | "
                     f"{t['transition_nonexistent'] + t['transition_ambiguous']} | {g['joined_in_us_eu_gap']} / {g['exact_in_us_eu_gap']} |")
    for rid, v in w["rules"].items():
        lines += ["", f"## {rid}: per year", "",
                  "| year | lake mapped | admitted | joined | exact | exact rate | clock | source (partial/all) | lake-only | admitted-only |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for y, t in v["by_year"].items():
            lines.append(f"| {y} | {t['lake_mapped']} | {t['admitted']} | {t['joined']} | {t['ohlc_exact']} | {t['exact_rate']} | "
                         f"{t['residual_clock']} | {t['residual_source_partial']}/{t['residual_source_all']} | {t['lake_only']} | {t['admitted_only']} |")
    sel = w["selected_rule"] or next(iter(w["rules"]))
    lines += ["", f"## {sel}: per month", "",
              "| month | lake mapped | admitted | joined | exact | exact rate | clock | source | lake-only | admitted-only |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m, t in w["rules"][sel]["by_month"].items():
        lines.append(f"| {m} | {t['lake_mapped']} | {t['admitted']} | {t['joined']} | {t['ohlc_exact']} | {t['exact_rate']} | "
                     f"{t['residual_clock']} | {t['residual_source']} | {t['lake_only']} | {t['admitted_only']} |")
    lines += ["", "## Coverage by half-year (rows)", "", "| half | admitted | " + " | ".join(w["coverage_by_half"]["lake_mapped"]) + " |",
              "|---|---:|" + "---:|" * len(w["coverage_by_half"]["lake_mapped"])]
    for h, n in w["coverage_by_half"]["admitted"].items():
        lines.append(f"| {h} | {n} | " + " | ".join(str(v.get(h, 0)) for v in w["coverage_by_half"]["lake_mapped"].values()) + " |")
    return "\n".join(lines) + "\n"


def verify_identity(record: dict, lake_path) -> bool:
    """Recompute the mapping digest of ``record`` from the lake bytes (raises on any difference)."""
    body = {k: v for k, v in record.items() if k != "identity_sha256"}
    if canonical_sha256(body) != record["identity_sha256"]:
        raise ClockError("IDENTITY_DIGEST: record body differs from its identity_sha256")
    if D.file_sha256(lake_path) != record["source"]["sha256"]:
        raise ClockError("SOURCE_DIGEST: lake bytes differ from the identity's source sha256")
    w = record["mapping"]["window_utc"]
    pol = record["transition_policy"]
    m = map_lake(lake_path, w[0], w[1], record["rule"]["id"], **pol)
    if mapping_digest(m["LABEL"], m.index) != record["mapping"]["digest_sha256"]:
        raise ClockError("MAPPING_DIGEST: recomputed mapping differs from the pinned identity")
    return True


# ----------------------------------------------------------------------------- CLI
def _resolve_path(template: str, roots=None) -> str:
    roots = dict(roots or {})
    out = template
    while "<" in out and ">" in out:
        name = out[out.index("<") + 1:out.index(">")]
        val = roots.get(name, os.environ.get(name))
        if not val:
            raise ValueError(f"UNRESOLVED_ROOT: <{name}> (set the environment variable or pass roots)")
        out = out.replace(f"<{name}>", val)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    lake_spec, adm_spec = cfg["sources"]["lake_mapped"], cfg["sources"]["admitted"]
    lake_path, adm_path = _resolve_path(lake_spec["path"]), _resolve_path(adm_spec["path"])
    for p, spec in ((lake_path, lake_spec), (adm_path, adm_spec)):
        if D.file_sha256(p) != spec["sha256"]:
            raise ValueError(f"SOURCE_DIGEST: {spec['path']} differs from its pin")
    closure = None
    if adm_spec.get("closure_intervals"):
        cp = _resolve_path(adm_spec["closure_intervals"]["path"])
        if D.file_sha256(cp) != adm_spec["closure_intervals"]["sha256"]:
            raise ValueError("SOURCE_DIGEST: closure intervals differ from their pin")
        closure = json.loads(Path(cp).read_text())
    wc = cfg["witness"]
    w = witness(lake_path, adm_path, wc["window_utc"][0], wc["window_utc"][1], wc["rules"], closure_intervals=closure,
                tol=wc["tolerance_abs"], lake_sha256=lake_spec["sha256"], admitted_sha256=adm_spec["sha256"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "WITNESS.json").write_text(json.dumps(w, indent=1, sort_keys=True) + "\n")
    (out / "WITNESS_TABLE.md").write_text(witness_table_md(w))
    if w["selected_rule"] is None:
        print(json.dumps({"selected_rule": None, "witness_sha256": w["witness_sha256"]}))
        return 2
    rid = w["selected_rule"]
    iw = cfg["identity"]["window_utc"]
    pol = cfg["identity"]["transition_policy"]
    m = map_lake(lake_path, iw[0], iw[1], rid, **pol)
    refuted = {k: {"verdict": v["verdict"], "residual_clock": v["totals"]["residual_clock"],
                   "us_eu_gap_periods": v["us_eu_gap_periods"]} for k, v in w["rules"].items() if k != rid}
    rec = identity_record(rid, lake_spec["sha256"], m["LABEL"], m.index, iw, pol, asset=cfg["asset"], bar=cfg["bar"],
                          extra={"witness": {"sha256": w["witness_sha256"], "verdict": w["rules"][rid]["verdict"],
                                             "window_utc": w["window_utc"], "admitted_sha256": adm_spec["sha256"],
                                             "other_rules": refuted}})
    (out / "CLOCK_IDENTITY.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    verify_identity(rec, lake_path)
    print(json.dumps({"selected_rule": rid, "identity_sha256": rec["identity_sha256"],
                      "mapping_digest": rec["mapping"]["digest_sha256"], "witness_sha256": w["witness_sha256"],
                      "verdicts": {k: v["verdict"] for k, v in w["rules"].items()},
                      "by_year": {y: t["exact_rate"] for y, t in w["rules"][rid]["by_year"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
