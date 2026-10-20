"""Inputs of the expert-guided ladder: admitted bars + calendar, Musashi's admitted-core 15 observations,
the ZigZag v2 expert recomputed on the admitted clock, labels, support, eligibility and periods.

Pins (verified before use; a mismatch raises):

* admitted root ``eurusd.parquet`` sha256 c80847bb..., ``CLOSURE_INTERVALS.json`` 71078e8f...; UTC bar START,
  available at start + 1 h; the calendar = the complement of the closure intervals (exact: every admitted
  bar lies in a segment), entry delay 12 h after a reopen, forced flat 12 h before a closure, cutoff 0;
* observations: Musashi's prepared ``train.csv`` (2020-2023) and ``validation.csv`` (2024) of the closed
  campaign with CONTRACT.json 37e06192... (selection 553fc047...), read-only; the contract's input rule:
  13 observations at stride 2 OBSERVED rows including the current completed bar, causal rolling z-score
  over the 256 previous rows [i-256, i) per feature. DATE_TIME = availability = bar start + 1 h, verified
  by OPEN/HIGH/LOW/CLOSE equality with the admitted bar (a mismatch raises, nothing is shifted by hand);
* expert: MetaTrader ZigZag Depth 3 / Deviation 40.7 pips / Backstep 0 (demonstrations v2 parameters),
  recomputed on the admitted bars (the v2 seal 38125b0b... was computed on the lake label clock; the new
  demonstration digest is reported beside it, audit F08).

Periods by the decision time (DATE_TIME): TRAIN 2020-2022, INNER_A 2023H1, INNER_B 2023H2, OBS_2024.
A row is eligible when the entry gate of the executable path is open at its fill bar (operational window),
its window/scale warm-up is complete, its hindsight label is not censored and its label support ends
inside its own period (purged otherwise). Exclusions are counted by reason.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import bar_source as BS
from .. import calendar as CAL
from .. import zigzag as Z
from . import oracle as OR

PINS = {
    "admitted_sha256": "c80847bb149c7af1ad50e42301b5cd70608e6c37702ce8cc6a7b9eb70f36d2ca",
    "closure_sha256": "71078e8f9a95c4438405a9fa3dafdffdc0f44183bbf27ba828168c18065e8350",
    "contract_sha256": "37e0619224109416dc35a726831183975fd41b03823a30685630dc2896d38029",
    "selection_sha256": "553fc04707ee3d1d2d552a21af073bf942e3a1b880150f2b34245764446c09e2",
    "train_csv_sha256": "3d8a9dcdc52d8eccd7799f46eb082387f8d8c48c5e15e4893b3ffdfc97333817",
    "validation_csv_sha256": "52ef0adc908339a3bdbb02787a9304ae510daeaa5519a3ca46a585319f2948d3",
    "demonstrations_v2_lake_sha256": "38125b0b368a29b425b4f4ac192fe380b343894fbceedfe17c1e9b85c6564a24",
}
FEATURES = ["HIGH", "RAW_LOG_RANGE", "RAW_CLOSE_LOG_RETURN", "RAW_ELAPSED_HOURS", "ATR", "RSI", "MACD_Histogram",
            "ADX", "DI_minus", "Stochastic_D", "WilliamsR", "rolling_std_24", "hod_sin", "hod_cos", "dow_cos"]
EXPERT = Z.Params(depth=3, deviation_pips=40.7, backstep=0)
CONFIRMED_K_TP = 1.5            # the v2 study's (b) choice for this expert on 2020-2022
PERIODS = {"TRAIN": ("2020-01-01", "2023-01-01"), "INNER_A": ("2023-01-01", "2023-07-01"),
           "INNER_B": ("2023-07-01", "2024-01-01"), "OBS_2024": ("2024-01-01", "2025-01-01")}
PERIOD_STATUS = {"TRAIN": "FIT", "INNER_A": "JOINT_DEVELOPMENT_MONITOR_CALIBRATION",
                 "INNER_B": "JOINT_DEVELOPMENT_SCORED_AFTER_FREEZE",
                 "OBS_2024": "PREVIOUSLY_OBSERVED_DEVELOPMENT_READ_ONCE_AFTER_FREEZE"}
WIN, STRIDE, SCALE_W = 13, 2, 256
H_NS = 3_600_000_000_000


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_ns(x) -> int:
    t = pd.Timestamp(x)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return int(t.value)


def calendar_from_closures(closures, delay=12, flat=12, cutoff=0, calendar_id="ADMITTED_CLOSURE_COMPLEMENT"):
    c = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in closures]
    segs = [{"segment_id": f"SEG-{c[i][1].strftime('%Y%m%dT%H')}", "market_open": c[i][1].isoformat(),
             "market_close": c[i + 1][0].isoformat()} for i in range(len(c) - 1) if c[i][1] < c[i + 1][0]]
    return CAL.load_calendar({"calendar_id": calendar_id, "version": "2026-10-19.v1", "segments": segs,
                              "entry_delay_after_open_hours": delay, "force_flat_before_close_hours": flat,
                              "entry_cutoff_before_flat_hours": cutoff})


def causal_rolling_windows(X: np.ndarray, W: int = SCALE_W, stride: int = STRIDE, win: int = WIN):
    """(windows [n, win, f], ok [n]): z-score with the statistics of rows [i-W, i) (never row i), then the
    ``win`` slots i - stride*(win-1) ... i (oldest first). Prefix-invariant by construction."""
    n = len(X)
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])
    cs2 = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X ** 2, axis=0)])
    i = np.arange(n)
    lo = np.clip(i - W, 0, None)
    cnt = np.maximum(i - lo, 1)[:, None]
    mean = (cs[i] - cs[lo]) / cnt
    var = (cs2[i] - cs2[lo]) / cnt - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        Zs = (X - mean) / np.where(std > 0, std, np.nan)
    idx = i[:, None] - stride * np.arange(win - 1, -1, -1)[None, :]
    ok = (idx[:, 0] >= 0) & (i - W >= 0)
    xa = Zs[np.clip(idx, 0, n - 1)]
    ok &= np.isfinite(xa).all(axis=(1, 2))
    return xa.astype(np.float32), ok


@dataclass
class Dataset:
    bars: pd.DataFrame
    cal: object
    sigma: np.ndarray
    sigma_rule: np.ndarray
    pivots: list
    conf: list
    y_hind: np.ndarray          # per bar
    y_conf: np.ndarray          # per bar
    support_end: np.ndarray     # per bar
    row_bar: np.ndarray         # per observation row: decision bar b
    row_ts: np.ndarray          # per observation row: DATE_TIME (ns) = decision time
    windows: np.ndarray         # per observation row
    row_ok: np.ndarray          # window/scale complete
    period: np.ndarray          # per row
    eligible: np.ndarray        # per row (final population)
    exclusions: dict
    meta: dict
    masks: dict = None          # per row: the exclusion masks by reason

    def row_of_bar(self) -> np.ndarray:
        out = np.full(len(self.bars), -1, np.int64)
        ok = self.row_bar >= 0
        out[self.row_bar[ok]] = np.flatnonzero(ok)
        return out

    def observable(self) -> np.ndarray:
        """Rows the agent can act on: joined, window/scale complete, causal sigma known (labels not needed)."""
        m = self.masks
        return ~(m["not_joined"] | m["window_or_scale_warmup"] | m["no_causal_sigma"])

    def labelled(self) -> np.ndarray:
        """Rows usable as training/monitor/score examples: observable, label final and inside its period."""
        m = self.masks
        return self.observable() & ~(m["label_censored"] | m["label_support_crosses_period_end"] | m["outside_periods"])

    def rows(self, period: str) -> np.ndarray:
        return np.flatnonzero(self.eligible & (self.period == period))


def load(admitted_dir, csv_dir, *, verify=True, end="2025-01-01") -> Dataset:
    admitted_dir, csv_dir = Path(admitted_dir), Path(csv_dir)
    pq, clp = admitted_dir / "eurusd.parquet", admitted_dir / "CLOSURE_INTERVALS.json"
    if verify:
        if sha256(pq) != PINS["admitted_sha256"] or sha256(clp) != PINS["closure_sha256"]:
            raise ValueError("ADMITTED_PIN")
        for role in ("train", "validation"):
            if sha256(csv_dir / f"{role}.csv") != PINS[f"{role}_csv_sha256"]:
                raise ValueError(f"SEL15_PIN:{role}")
        if (csv_dir / "CONTRACT.json").exists() and sha256(csv_dir / "CONTRACT.json") != PINS["contract_sha256"]:
            raise ValueError("CONTRACT_PIN")
    bars = BS.load_ohlc(pq, "2020-01-01", end, source="admitted")
    cal = calendar_from_closures(json.loads(clp.read_text()))
    cal.assign(bars.index)
    return build(bars, cal, csv_dir)


def build(bars: pd.DataFrame, cal, csv_dir, receipts: bool = True) -> Dataset:
    csv_dir = Path(csv_dir)
    frames, rec = [], {}
    for role in ("train", "validation"):
        df = pd.read_csv(csv_dir / f"{role}.csv")
        if receipts and (csv_dir / f"{role}.json").exists():
            r = json.loads((csv_dir / f"{role}.json").read_text())
            if list(r["feature_columns"]) != FEATURES or len(df) != r["row_count"] or r["status"] != "ADMITTED":
                raise ValueError(f"SEL15_RECEIPT:{role}")
            rec[role] = {k: r.get(k) for k in ("input_sha256", "selection_sha256", "row_count", "period_start",
                                               "period_end", "timestamp_semantics", "calendar_sha256")}
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    ts = pd.DatetimeIndex(pd.to_datetime(df["DATE_TIME"], utc=True))
    ts_ns = np.array([t.value for t in ts], np.int64)
    if not (np.diff(ts_ns) > 0).all():
        raise ValueError("SEL15_TIMESTAMPS")
    X = df[FEATURES].to_numpy(float)
    if not np.isfinite(X).all():
        raise ValueError("SEL15_NONFINITE")
    windows, row_ok = causal_rolling_windows(X)
    # join: DATE_TIME = availability = bar start + 1 h, verified on OHLC
    v = bars.index.as_unit("ns").asi8
    pos = np.searchsorted(v, ts_ns - H_NS)
    inb = (pos < len(v)) & (v[np.minimum(pos, len(v) - 1)] == ts_ns - H_NS)
    row_bar = np.where(inb, pos, -1)
    o, h, l, c = (bars[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
    rb = np.clip(row_bar, 0, len(v) - 1)
    same = inb & np.isclose(o[rb], df.OPEN.to_numpy(float), atol=1e-9) & np.isclose(h[rb], df.HIGH.to_numpy(float), atol=1e-9) \
        & np.isclose(l[rb], df.LOW.to_numpy(float), atol=1e-9) & np.isclose(c[rb], df.CLOSE.to_numpy(float), atol=1e-9)
    in_window = ts_ns - H_NS < v[-1] + H_NS
    if (in_window & inb & ~same).any():
        raise ValueError(f"SEL15_JOIN_PRICE_MISMATCH: {int((in_window & inb & ~same).sum())} rows")
    # expert on the admitted clock
    piv = Z.hindsight_pivots(h, l, EXPERT)
    conf = Z.confirmation_bars_fast(h, l, piv, EXPERT)
    n = len(v)
    y_h = OR.pivot_labels(n, piv)
    y_c, collisions = OR.confirmed_labels(n, piv, conf)
    sup = OR.label_support_end(n, piv, conf)
    sigma, sigma_rule = Z.causal_sigma(bars), Z.causal_range_sigma(bars)
    # periods and eligibility
    period = np.full(len(df), "", object)
    for name, (a, b) in PERIODS.items():
        period[(ts_ns >= _utc_ns(a)) & (ts_ns < _utc_ns(b))] = name
    seg_of = cal.assign(bars.index)
    op = np.array([s.operational_start.value for s in cal.segments], np.int64)
    ff = np.array([s.force_flat.value for s in cal.segments], np.int64)
    ec = np.array([s.entry_cutoff.value for s in cal.segments], np.int64)
    k = np.clip(row_bar + 1, 0, n - 1)
    gate = (row_bar >= 0) & (row_bar + 1 < n) & (seg_of[rb] == seg_of[k]) & (v[k] >= op[seg_of[rb]]) \
        & (v[k] < ec[seg_of[rb]]) & (v[k] + H_NS < ff[seg_of[rb]])
    sig_ok = np.isfinite(sigma[k]) & (sigma[k] > 0) & np.isfinite(sigma_rule[k]) & (sigma_rule[k] > 0)
    sup_r = sup[rb]
    censored = (row_bar >= 0) & (sup_r < 0)
    p_end = np.array([_utc_ns(PERIODS[p][1]) if p else 0 for p in period], np.int64)
    sup_time = v[np.clip(sup_r, 0, n - 1)] + H_NS            # the support bar is known at its end
    purged = (~censored) & (row_bar >= 0) & (sup_time > p_end)
    reasons = {"outside_periods": period == "", "not_joined": row_bar < 0, "window_or_scale_warmup": ~row_ok,
               "entry_gate_closed": ~gate, "no_causal_sigma": ~sig_ok, "label_censored": censored,
               "label_support_crosses_period_end": purged}
    eligible = np.ones(len(df), bool)
    excl = {}
    for name, mask in reasons.items():
        excl[name] = {p: int((mask & (period == p)).sum()) for p in list(PERIODS) + [""]}
        eligible &= ~mask
    meta = {"pins": PINS, "receipts": rec, "expert": {"depth": EXPERT.depth, "deviation_pips": EXPERT.deviation_pips,
            "backstep": EXPERT.backstep, "source": "admitted", "pivots": len(piv),
            "confirmed": int(sum(1 for x in conf if x >= 0)), "confirmation_collisions": int(collisions)},
            "rows_total": int(len(df)), "rows_eligible": {p: int((eligible & (period == p)).sum()) for p in PERIODS},
            "input_contract": {"observations": WIN, "stride_rows": STRIDE, "includes_current_completed_bar": True,
                               "scaling": f"causal rolling z-score over rows [i-{SCALE_W}, i)", "features": FEATURES},
            "periods": PERIODS, "period_status": PERIOD_STATUS}
    return Dataset(bars, cal, sigma, sigma_rule, piv, conf, y_h, y_c, sup, row_bar, ts_ns, windows, row_ok, period,
                   eligible, excl, meta, reasons)
