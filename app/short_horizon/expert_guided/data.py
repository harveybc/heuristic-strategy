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

Input scaling (auditor's defects, 2026-10-21): ``scaling`` is a strict enum (``SCALINGS``); an unknown value
raises ``SCALING_UNKNOWN`` before any fit and every loader propagates it. The effective policy, its constants
and its NaN accounting live in ``Dataset.meta["scaling"]`` and are persisted next to every checkpoint as
``SCALER.json`` (``write_scaler``); a reload refuses to score without it or with another feature order
(``load_with_scaler``). Split contracts (``SPLIT_CONTRACTS``): ``dev3y`` is the default above; ``consolidated``
keeps TRAIN 2020-2023 with its last 6 months as the inner selection block and reads 2024 once; TEST (>= 2025)
stays closed in both (the admitted reader refuses it).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
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
PINS_V4 = {
    "features_all_sha256": "2cbe26e069a9c44912958bbd6d3d06d53524c38b4134b959145b5e6948e88cfb",
    "frozen_subset_sha256": "300cbc6e9f75182999401d4830926f45f78eb45c810165339c486f7035dc841c",
    "selection_manifest_sha256": "ceba29e938bc882ee09002919a02903238aa3eb336c97632c4e5ab752fbe401c",
}
FEATURES_V4 = ["RAW_BAR_LOG_RETURN", "cal.utc_session_us", "hurst_spectrum_width_w200", "px.close_loc_2"]
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
SCALINGS = ("fixed_train", "rolling_z")
# What a fixed-constant scaler does with a NON-STATIONARY input (owner 2026-10-22, after the E3-M run on the inherited
# core15 CSV: the raw HIGH level under fixed TRAIN constants became a date proxy -> TRAIN memorized, 2023 lost).
# "diff" (default): replace the feature by its first difference and rename it "<f>@diff"; "refuse": typed error;
# "keep": leave it, recorded. The same ADF/KPSS rule as the feature catalog (tools/layer1_pool_h1.py), on TRAIN only.
NONSTATIONARY = ("diff", "refuse", "keep")
SCALER_FILE = "SCALER.json"
TEST_CLOSED_FROM = "2025-01-01"
# Split contracts (arm recorded in every output). ``dev3y`` = the periods above (fit 2020-2022, 2023A monitor/selection,
# 2023B scored after freeze, 2024 observed once). ``consolidated`` = TRAIN 2020-2023 with an internal fold: the fit block
# is 2020-01..2023-07 and the LAST 6 MONTHS of TRAIN (2023-07..2024-01) are the inner selection block (early stopping and
# every model/arbiter choice come from inside TRAIN); VALIDATION 2024 is read once. The scaler constants of
# ``fixed_train`` are fitted on the fit block only, so the inner block is seen by no fitted statistic.
SPLIT_CONTRACTS = {
    "dev3y": {"periods": PERIODS, "status": PERIOD_STATUS, "scaler_fit": PERIODS["TRAIN"],
              "report": ("TRAIN", "INNER_A", "INNER_B"), "selection": "INNER_A", "read_once": "OBS_2024",
              "train": PERIODS["TRAIN"], "validation": PERIODS["OBS_2024"]},
    "consolidated": {"periods": {"TRAIN": ("2020-01-01", "2023-07-01"), "INNER_A": ("2023-07-01", "2024-01-01"),
                                 "OBS_2024": ("2024-01-01", "2025-01-01")},
                     "status": {"TRAIN": "FIT_BLOCK_OF_CONSOLIDATED_TRAIN_2020_2023",
                                "INNER_A": "INNER_SELECTION_BLOCK_LAST_6_MONTHS_OF_TRAIN",
                                "OBS_2024": "VALIDATION_2024_READ_ONCE"},
                     "scaler_fit": ("2020-01-01", "2023-07-01"), "report": ("TRAIN", "INNER_A"), "selection": "INNER_A",
                     "read_once": "OBS_2024", "train": ("2020-01-01", "2024-01-01"), "validation": ("2024-01-01", "2025-01-01")},
}
# Walk-forward folds INSIDE development (owner 2026-10-25: "arreglar primero el instrumento de medida"): the same shape
# as ``consolidated`` (fit block + its last 6 months as the inner selection block) rolled back so that a WHOLE YEAR after
# the fit is scored after freeze (INNER_B, never used for any choice). ``wf2022``: fit 2020-01..2022-01 (inner 2021H2),
# validation 2022; ``wf2023``: fit 2020-01..2023-01 (inner 2022H2), validation 2023. The scaler constants come from the
# fit block only. OBS_2024 keeps its single read so every arm of every fold is reported the same way. TEST stays closed.
# Diagnostic contracts: they never release anything; the principal contract of the clean round remains ``consolidated``.
WALK_FORWARD_FOLDS = ("wf2022", "wf2023")


def _wf_contract(fit_end: str, inner_start: str, val_end: str) -> dict:
    return {"periods": {"TRAIN": ("2020-01-01", inner_start), "INNER_A": (inner_start, fit_end), "INNER_B": (fit_end, val_end),
                        "OBS_2024": ("2024-01-01", "2025-01-01")},
            "status": {"TRAIN": "WALK_FORWARD_FIT_BLOCK", "INNER_A": "INNER_SELECTION_BLOCK_LAST_6_MONTHS_OF_FIT",
                       "INNER_B": "WALK_FORWARD_VALIDATION_YEAR_SCORED_AFTER_FREEZE",
                       "OBS_2024": "PREVIOUSLY_OBSERVED_DEVELOPMENT_READ_ONCE_AFTER_FREEZE"},
            "scaler_fit": ("2020-01-01", inner_start), "report": ("TRAIN", "INNER_A", "INNER_B"), "selection": "INNER_A",
            "read_once": "OBS_2024", "train": ("2020-01-01", fit_end), "validation": (fit_end, val_end)}


SPLIT_CONTRACTS["wf2022"] = _wf_contract("2022-01-01", "2021-07-01", "2023-01-01")
SPLIT_CONTRACTS["wf2023"] = _wf_contract("2023-01-01", "2022-07-01", "2024-01-01")


def check_scaling(scaling) -> str:
    """Strict enum: anything but the two declared policies is refused before any statistic is fitted."""
    if scaling not in SCALINGS:
        raise ValueError(f"SCALING_UNKNOWN:{scaling}")
    return scaling


def split_contract(name) -> dict:
    if name not in SPLIT_CONTRACTS:
        raise ValueError(f"SPLIT_CONTRACT_UNKNOWN:{name}")
    return SPLIT_CONTRACTS[name]


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
        before = m.get("before_eligible_from", np.zeros(len(self.row_bar), bool))   # shared-support option (matched arms)
        return ~(m["not_joined"] | m["window_or_scale_warmup"] | m["no_causal_sigma"] | before)

    def labelled(self) -> np.ndarray:
        """Rows usable as training/monitor/score examples: observable, label final and inside its period."""
        m = self.masks
        return self.observable() & ~(m["label_censored"] | m["label_support_crosses_period_end"] | m["outside_periods"])

    def rows(self, period: str) -> np.ndarray:
        return np.flatnonzero(self.eligible & (self.period == period))


# Fixed TRAIN-only scaling (owner 2026-10-21: the rolling z-score destroys amplitude information; "parámetros del
# z-score por característica solo en TRAIN y con eso se estandariza todo"): every feature is (x - mean_TRAIN) /
# std_TRAIN with the constants frozen on 2020-2022. No feature is exempt.
NATURAL_SCALE: dict = {}


def stationarity_decisions(X: np.ndarray, ts_ns: np.ndarray, features: list, train, policy: str = "diff"):
    """Measure every feature on the TRAIN rows (ADF, max lag 24; KPSS level, 24 lags) and decide what a fixed scaler may
    receive: ADF p > 0.05 AND KPSS p < 0.05 -> NON_STATIONARY. Returns (X2, features2, table). Under ``policy="diff"``
    the column becomes its first difference (NaN first row -> that window is not ok) and is renamed ``<f>@diff``; a
    feature already named ``...@diff``/``...@asinh`` (decided upstream by the catalog) is measured but never re-decided.
    ``refuse`` raises ``SCALING_NONSTATIONARY_INPUT:<f>``; ``keep`` records the finding only."""
    if policy not in NONSTATIONARY:
        raise ValueError(f"NONSTATIONARY_UNKNOWN:{policy}")
    from statsmodels.tsa.stattools import adfuller, kpss
    import warnings
    t0, t1 = _utc_ns(train[0]), _utc_ns(train[1])
    tr = (ts_ns >= t0) & (ts_ns < t1)
    X2, feats, table = X.astype(float).copy(), list(features), {}
    for j, f in enumerate(features):
        x = X[tr, j]
        x = x[np.isfinite(x)]
        rec = {"adf_p": None, "kpss_p": None, "decision": "NONE", "reason": "stationary or undecidable"}
        if len(x) >= 100 and np.std(x) > 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rec["adf_p"] = float(adfuller(x, maxlag=24, autolag=None)[1])
                    rec["kpss_p"] = float(kpss(x, regression="c", nlags=24)[1])
            except Exception as exc:  # recorded, never silent
                rec["error"] = str(exc)[:80]
        decided_upstream = "@" in f
        if rec["adf_p"] is not None and rec["adf_p"] > 0.05 and rec["kpss_p"] is not None and rec["kpss_p"] < 0.05 and not decided_upstream:
            rec["reason"] = f"non-stationary level on TRAIN (ADF p={rec['adf_p']:.3f}, KPSS p={rec['kpss_p']:.3f})"
            if policy == "refuse":
                raise ValueError(f"SCALING_NONSTATIONARY_INPUT:{f}")
            if policy == "diff":
                X2[:, j] = np.r_[np.nan, np.diff(X[:, j])]
                feats[j] = f"{f}@diff"
                rec["decision"] = "DIFF"
            else:
                rec["decision"] = "KEEP_NONSTATIONARY"
        table[f] = rec
    return X2, feats, table


def fixed_train_windows(X: np.ndarray, ts_ns: np.ndarray, features: list, stride: int = STRIDE, win: int = WIN,
                        train=("2020-01-01", "2023-01-01")):
    """Windows of the raw features scaled with TRAIN-only constants (causal: constants come from the past of every
    scored row); same slot layout as ``causal_rolling_windows``. Returns (windows, ok, scale_doc).

    NaN policy (explicit, 2026-10-21): the constants are nanmean/nanstd (ddof 0) over the TRAIN rows, so a warm-up
    or a gap inside TRAIN does not poison a feature; an EMPTY TRAIN (``SCALING_TRAIN_EMPTY``), a feature with no
    finite value inside TRAIN (``SCALING_FEATURE_ALL_NAN:<name>``) or a non-finite constant, e.g. an inf in the data
    (``SCALING_NONFINITE_CONSTANT:<name>``) are refused. Per feature the share of NaN inside TRAIN and the rule
    ``std_zero_to_one`` (std == 0 -> scale 1) are recorded; a row whose window holds any non-finite value is
    excluded (``rows_excluded_nan_windows``), separately from the index warm-up (``rows_excluded_warmup``)."""
    n = len(X)
    t0, t1 = _utc_ns(train[0]), _utc_ns(train[1])
    tr = (ts_ns >= t0) & (ts_ns < t1)
    n_tr = int(tr.sum())
    if n_tr == 0:
        raise ValueError("SCALING_TRAIN_EMPTY")
    Xtr = X[tr]
    fin_tr = np.isfinite(Xtr)
    n_fin = fin_tr.sum(0)
    for j, f in enumerate(features):
        if n_fin[j] == 0:
            raise ValueError(f"SCALING_FEATURE_ALL_NAN:{f}")
    # nan-aware: a feature with a warm-up or a gap inside TRAIN must not poison the constants of every row
    with np.errstate(invalid="ignore"):
        mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0)
    for j, f in enumerate(features):
        if not (np.isfinite(mu[j]) and np.isfinite(sd[j])):
            raise ValueError(f"SCALING_NONFINITE_CONSTANT:{f}")
    Z = np.empty_like(X, dtype=float)
    doc = {}
    for j, f in enumerate(features):
        nan_share = float(1.0 - n_fin[j] / n_tr)
        if f in NATURAL_SCALE:
            Z[:, j] = X[:, j] / NATURAL_SCALE[f]
            doc[f] = {"scale": "natural", "divisor": NATURAL_SCALE[f], "nan_share_train": nan_share, "fit_rows": int(n_fin[j])}
        else:
            s_ = sd[j] if sd[j] > 0 else 1.0
            Z[:, j] = (X[:, j] - mu[j]) / s_
            doc[f] = {"scale": "fixed_train_z", "mean": float(mu[j]), "std": float(s_), "std_raw": float(sd[j]),
                      "variance": float(sd[j] ** 2), "rule": ("std_zero_to_one" if sd[j] <= 0 else "z"),
                      "nan_share_train": nan_share, "fit_rows": int(n_fin[j])}
    i = np.arange(n)
    idx = i[:, None] - stride * np.arange(win - 1, -1, -1)[None, :]
    ok_idx = idx[:, 0] >= 0
    xa = Z[np.clip(idx, 0, n - 1)]
    fin_w = np.isfinite(xa).all(axis=(1, 2))
    ok = ok_idx & fin_w
    return xa.astype(np.float32), ok, {"method": "fixed_train", "train": list(train), "features": doc, "fit_rows": n_tr,
                                      "ddof": 0, "nan_policy": "nanmean/nanstd over TRAIN; empty TRAIN, all-NaN feature or "
                                                                "non-finite constant refused; NaN windows excluded",
                                      "rows_excluded_warmup": int((~ok_idx).sum()),
                                      "rows_excluded_nan_windows": int((ok_idx & ~fin_w).sum())}


def recipe_windows(X: np.ndarray, features: list, rec: dict, stride: int, win: int):
    """Apply a PINNED consumer recipe (heuristic-strategy tools/expert_guided_consumer_recipe.py, Musashi's audit A3
    2026-10-23) without measuring or fitting anything: the consumer-applied operations of each column (``DIFF`` or
    none, as recorded), then the recipe's fixed TRAIN constants, then the window layout. Returns (windows, ok, scale_doc,
    tensor_features) with the same scale_doc shape as ``fixed_train_windows`` (so SCALER.json is comparable)."""
    cols = rec["tensor"]["columns"]
    if [c["store_column"] for c in cols] != list(features):
        raise ValueError("RECIPE_COLUMN_ORDER_MISMATCH")
    if int(rec["tensor"]["window"]["observations"]) != int(win) or int(rec["tensor"]["window"]["stride_rows"]) != int(stride):
        raise ValueError("RECIPE_WINDOW_MISMATCH")
    sc = rec["scaler"]
    mean, std = np.asarray(sc["mean"], float), np.asarray(sc["std"], float)
    if len(mean) != len(cols) or len(std) != len(cols) or not (np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()):
        raise ValueError("RECIPE_SCALER_INVALID")
    X2 = X.astype(float).copy()
    tensor = []
    for j, c in enumerate(cols):
        ops = list(c.get("consumer_ops", []))
        if ops not in ([], ["DIFF"]):
            raise ValueError(f"RECIPE_CONSUMER_OP_UNKNOWN:{c['store_column']}:{ops}")
        if ops == ["DIFF"]:
            X2[:, j] = np.r_[np.nan, np.diff(X[:, j])]
        tensor.append(c["tensor_name"])
    if tensor != list(sc["features"]):
        raise ValueError("RECIPE_TENSOR_ORDER_MISMATCH")
    Z = (X2 - mean) / std
    n = len(X2)
    i = np.arange(n)
    idx = i[:, None] - stride * np.arange(win - 1, -1, -1)[None, :]
    ok_idx = idx[:, 0] >= 0
    xa = Z[np.clip(idx, 0, n - 1)]
    fin_w = np.isfinite(xa).all(axis=(1, 2))
    ok = ok_idx & fin_w
    feats_doc = {f: {"scale": "fixed_train_z", "mean": float(mean[k]), "std": float(std[k]), "std_raw": float(sc["std_raw"][k]),
                     "variance": float(sc["variance"][k]), "rule": sc["rule"][k], "nan_share_train": float(sc["nan_share_train"][k]),
                     "fit_rows": int(sc["fit_rows_per_feature"][k])} for k, f in enumerate(tensor)}
    doc = {"method": "fixed_train", "train": list(sc["fit_period"]), "features": feats_doc, "fit_rows": int(sc["fit_rows"]), "ddof": int(sc["ddof"]),
           "nan_policy": sc["nan_policy"], "rows_excluded_warmup": int((~ok_idx).sum()), "rows_excluded_nan_windows": int((ok_idx & ~fin_w).sum()),
           "stationarity": sc["stationarity"], "recipe": {"identity": rec["identity"], "applied": "pinned constants; nothing measured or fitted"}}
    return xa.astype(np.float32), ok, doc, tensor


def load(admitted_dir, csv_dir, *, verify=True, end="2025-01-01", scaling="rolling_z", contract="dev3y", nonstationary="diff", eligible_from=None) -> Dataset:
    check_scaling(scaling)
    split_contract(contract)
    admitted_dir, csv_dir = Path(admitted_dir), Path(csv_dir)
    pq, clp = admitted_dir / "eurusd.parquet", admitted_dir / "CLOSURE_INTERVALS.json"
    digests = {"eurusd.parquet": sha256(pq), "CLOSURE_INTERVALS.json": sha256(clp),
               **{f"{role}.csv": sha256(csv_dir / f"{role}.csv") for role in ("train", "validation")}}
    if (csv_dir / "CONTRACT.json").exists():
        digests["CONTRACT.json"] = sha256(csv_dir / "CONTRACT.json")
    if verify:
        if digests["eurusd.parquet"] != PINS["admitted_sha256"] or digests["CLOSURE_INTERVALS.json"] != PINS["closure_sha256"]:
            raise ValueError("ADMITTED_PIN")
        for role in ("train", "validation"):
            if digests[f"{role}.csv"] != PINS[f"{role}_csv_sha256"]:
                raise ValueError(f"SEL15_PIN:{role}")
        if "CONTRACT.json" in digests and digests["CONTRACT.json"] != PINS["contract_sha256"]:
            raise ValueError("CONTRACT_PIN")
    bars = BS.load_ohlc(pq, "2020-01-01", end, source="admitted")
    cal = calendar_from_closures(json.loads(clp.read_text()))
    cal.assign(bars.index)
    return build(bars, cal, csv_dir, scaling=scaling, contract=contract, nonstationary=nonstationary, eligible_from=eligible_from, sources={"loader": "core15", "verified": bool(verify),
                                                                                "sha256": digests})


def load_v4(admitted_dir, v4_dir, *, verify=True, end="2025-01-01", scaling="rolling_z", contract="dev3y", nonstationary="diff", eligible_from=None) -> Dataset:
    """Same population rules as ``load`` but the observation features are the four TIMING_PROXY_SELECTION v4
    inputs (frozen subset read from the snapshot, never hard-coded). The v4 matrix is stamped at bar start;
    the observation becomes available at bar close, so DATE_TIME = timestamp + 1 h (verified on OHLC).
    ``scaling`` is propagated (default unchanged: rolling_z); an unknown value is refused before any read."""
    check_scaling(scaling)
    split_contract(contract)
    admitted_dir, v4_dir = Path(admitted_dir), Path(v4_dir)
    pq, clp = admitted_dir / "eurusd.parquet", admitted_dir / "CLOSURE_INTERVALS.json"
    fp, sp = v4_dir / "eval_input" / "features_all.parquet", v4_dir / "FROZEN_SUBSET.json"
    digests = {"eurusd.parquet": sha256(pq), "CLOSURE_INTERVALS.json": sha256(clp), "features_all.parquet": sha256(fp),
               "FROZEN_SUBSET.json": sha256(sp)}
    if verify:
        if digests["eurusd.parquet"] != PINS["admitted_sha256"] or digests["CLOSURE_INTERVALS.json"] != PINS["closure_sha256"]:
            raise ValueError("ADMITTED_PIN")
        if digests["features_all.parquet"] != PINS_V4["features_all_sha256"] or digests["FROZEN_SUBSET.json"] != PINS_V4["frozen_subset_sha256"]:
            raise ValueError("V4_PIN")
    feats = list(json.loads(sp.read_text())["selected"])
    if feats != FEATURES_V4:
        raise ValueError(f"V4_SUBSET:{feats}")
    raw = pd.read_parquet(fp, columns=["timestamp", "OPEN", "HIGH", "LOW", "CLOSE"] + feats)
    raw = raw[raw["timestamp"] < pd.Timestamp(end, tz="UTC")]
    fin = np.isfinite(raw[feats].to_numpy(float)).all(1)
    first = int(np.argmax(fin))                 # declared warm-up (hurst w200, close_loc_2) is leading only
    if not fin[first:].all():
        raise ValueError(f"V4_INTERIOR_NONFINITE:{int((~fin[first:]).sum())}")
    raw = raw.iloc[first:]
    df = raw.drop(columns="timestamp")
    df.insert(0, "DATE_TIME", (raw["timestamp"] + pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00").values)
    bars = BS.load_ohlc(pq, "2020-01-01", end, source="admitted")
    cal = calendar_from_closures(json.loads(clp.read_text()))
    cal.assign(bars.index)
    return build(bars, cal, None, frame=df.reset_index(drop=True), features=feats, scaling=scaling, contract=contract, nonstationary=nonstationary, eligible_from=eligible_from,
                 frame_meta={"source": "TIMING_PROXY_SELECTION v4", "pins": PINS_V4},
                 sources={"loader": "v4", "verified": bool(verify), "sha256": digests})


def load_store(admitted_dir, store_parquet, features: list, *, verify=True, end="2025-01-01", scaling="fixed_train", nonstationary="diff", eligible_from=None, win: int = WIN, stride: int = STRIDE,
               manifest_sha256: str | None = None, contract="dev3y", recipe=None) -> Dataset:
    """Observations from an arbitrary feature store on the admitted grid (layer-1 pool, 2026-10-22): the store's
    MANIFEST must name this admitted source; ``features`` are the selected columns; the store timestamp is the bar
    START (availability = start + 1 h). Scaling default = fixed TRAIN-only standardization (owner's rule)."""
    check_scaling(scaling)
    split_contract(contract)
    admitted_dir, store_parquet = Path(admitted_dir), Path(store_parquet)
    pq, clp = admitted_dir / "eurusd.parquet", admitted_dir / "CLOSURE_INTERVALS.json"
    man = json.loads((store_parquet.parent / "MANIFEST.json").read_text())
    digests = {"eurusd.parquet": sha256(pq), "CLOSURE_INTERVALS.json": sha256(clp), "store.parquet": sha256(store_parquet),
               "MANIFEST.json": sha256(store_parquet.parent / "MANIFEST.json")}
    if verify:
        if digests["eurusd.parquet"] != PINS["admitted_sha256"] or digests["CLOSURE_INTERVALS.json"] != PINS["closure_sha256"]:
            raise ValueError("ADMITTED_PIN")
        if man.get("source_sha256") != PINS["admitted_sha256"] or digests["store.parquet"] != man.get("file_sha256"):
            raise ValueError("STORE_IDENTITY")
        if manifest_sha256 and digests["MANIFEST.json"] != manifest_sha256:
            raise ValueError("STORE_MANIFEST_PIN")
    missing = [f for f in features if f not in man["features"]]
    if missing:
        raise ValueError(f"STORE_FEATURES_MISSING:{missing}")
    if recipe is not None:                     # the pinned recipe binds the store bytes, its manifest and the columns
        if digests["store.parquet"] != recipe["store"]["sha256"] or digests["MANIFEST.json"] != recipe["store"]["manifest_sha256"]:
            raise ValueError("RECIPE_STORE_IDENTITY")
        if list(features) != [c["store_column"] for c in recipe["tensor"]["columns"]]:
            raise ValueError("RECIPE_COLUMN_ORDER_MISMATCH")
        if contract != recipe["split"]["arm"]:
            raise ValueError(f"RECIPE_SPLIT_MISMATCH:{contract}!={recipe['split']['arm']}")
    raw = pd.read_parquet(store_parquet, columns=["timestamp", "OPEN", "HIGH", "LOW", "CLOSE"] + [f for f in features if f not in ("OPEN", "HIGH", "LOW", "CLOSE")])
    raw = raw[pd.to_datetime(raw["timestamp"], utc=True) < pd.Timestamp(end, tz="UTC")]
    df = raw.drop(columns="timestamp")
    df.insert(0, "DATE_TIME", (pd.to_datetime(raw["timestamp"], utc=True) + pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00").values)
    fin = np.isfinite(df[features].to_numpy(float)).all(1)
    first = int(np.argmax(fin))
    df = df.iloc[first:].reset_index(drop=True)
    bars = BS.load_ohlc(pq, "2020-01-01", end, source="admitted")
    cal = calendar_from_closures(json.loads(clp.read_text()))
    cal.assign(bars.index)
    return build(bars, cal, None, frame=df, features=list(features), scaling=scaling, contract=contract, nonstationary=nonstationary, eligible_from=eligible_from, win=win, stride=stride,
                 recipe=recipe,
                 frame_meta={"source": "layer1 store", "store_sha256": man.get("file_sha256"), "n_features": len(features)},
                 sources={"loader": "store", "verified": bool(verify), "sha256": digests})


def build(bars: pd.DataFrame, cal, csv_dir, receipts: bool = True, *, frame=None, features=None,
          frame_meta=None, scaling="rolling_z", contract="dev3y", sources=None, nonstationary="diff",
          eligible_from=None, win: int = WIN, stride: int = STRIDE, recipe=None) -> Dataset:
    check_scaling(scaling)                       # strict enum, refused before any statistic is fitted
    if nonstationary not in NONSTATIONARY:
        raise ValueError(f"NONSTATIONARY_UNKNOWN:{nonstationary}")
    sc = split_contract(contract)
    periods, period_status = sc["periods"], sc["status"]
    features = FEATURES if features is None else list(features)
    rec = {}
    if frame is not None:
        df = frame
        rec = {"frame": frame_meta}
    else:
        csv_dir = Path(csv_dir)
        frames = []
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
    X = df[features].to_numpy(float)
    if frame is None and not np.isfinite(X).all():
        raise ValueError("SEL15_NONFINITE")      # the v4 matrix has a declared fractal warm-up (NaN -> window not ok)
    if recipe is not None:
        if scaling != "fixed_train":
            raise ValueError("RECIPE_REQUIRES_FIXED_TRAIN")
        windows, row_ok, scale_doc, features = recipe_windows(X, features, recipe, stride, win)
    elif scaling == "fixed_train":
        # a fixed-constant scaler may only receive stationary inputs: measured on TRAIN, decided and recorded here
        X, features, stationarity = stationarity_decisions(X, ts_ns, features, sc["scaler_fit"], nonstationary)
        windows, row_ok, scale_doc = fixed_train_windows(X, ts_ns, features, stride=stride, win=win, train=sc["scaler_fit"])
        scale_doc["stationarity"] = {"policy": nonstationary, "rule": "ADF p > 0.05 and KPSS p < 0.05 on TRAIN -> NON_STATIONARY",
                                     "features": stationarity,
                                     "differenced": [f for f, r in stationarity.items() if r["decision"] == "DIFF"]}
    else:
        windows, row_ok = causal_rolling_windows(X, stride=stride, win=win)
        scale_doc = {"method": "rolling_z", "rows": SCALE_W, "constants": None,
                     "note": f"no constants exist: every row is standardized with the statistics of rows [i-{SCALE_W}, i)",
                     "rows_excluded_warmup_or_nan": int((~row_ok).sum())}
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
    for name, (a, b) in periods.items():
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
    p_end = np.array([_utc_ns(periods[p][1]) if p else 0 for p in period], np.int64)
    sup_time = v[np.clip(sup_r, 0, n - 1)] + H_NS            # the support bar is known at its end
    purged = (~censored) & (row_bar >= 0) & (sup_time > p_end)
    reasons = {"outside_periods": period == "", "not_joined": row_bar < 0, "window_or_scale_warmup": ~row_ok,
               "entry_gate_closed": ~gate, "no_causal_sigma": ~sig_ok, "label_censored": censored,
               "label_support_crosses_period_end": purged}
    # shared-support option (matched comparisons, auditor 2026-10-22 §3.2): rows before ``eligible_from`` are never
    # eligible in ANY arm, so arms with different warm-ups (rolling 256 rows, relative columns 480 bars, ...) score
    # the same rows; the per-period support hash below makes the match verifiable across runs
    if eligible_from:
        reasons["before_eligible_from"] = ts_ns < _utc_ns(eligible_from)
    eligible = np.ones(len(df), bool)
    excl = {}
    for name, mask in reasons.items():
        excl[name] = {p: int((mask & (period == p)).sum()) for p in list(periods) + [""]}
        eligible &= ~mask
    meta = {"pins": PINS, "receipts": rec, "expert": {"depth": EXPERT.depth, "deviation_pips": EXPERT.deviation_pips,
            "backstep": EXPERT.backstep, "source": "admitted", "pivots": len(piv),
            "confirmed": int(sum(1 for x in conf if x >= 0)), "confirmation_collisions": int(collisions)},
            "rows_total": int(len(df)), "rows_eligible": {p: int((eligible & (period == p)).sum()) for p in periods},
            "support_hash": {p: hashlib.sha256(np.ascontiguousarray(ts_ns[eligible & (period == p)]).tobytes()).hexdigest()[:16]
                             for p in periods}, "eligible_from": eligible_from,
            "input_contract": {"observations": int(win), "stride_rows": int(stride), "includes_current_completed_bar": True,
                               "scaling": (f"causal rolling z-score over rows [i-{SCALE_W}, i)" if scaling != "fixed_train"
                                           else ("fixed TRAIN-only mean/std (2020-2022) for every feature" if contract == "dev3y"
                                                 else f"fixed TRAIN-only mean/std ({sc['scaler_fit'][0]}..{sc['scaler_fit'][1]}) for every feature")),
                               "scale_doc": scale_doc, "features": features},
            "periods": periods, "period_status": period_status,
            "split_contract": {"arm": contract, "train": list(sc["train"]), "validation": list(sc["validation"]),
                               "selection_period": sc["selection"], "read_once": sc["read_once"],
                               "test_closed_from": TEST_CLOSED_FROM},
            "scaling": {"policy": scaling, **scale_doc},
            "sources": sources}
    return Dataset(bars, cal, sigma, sigma_rule, piv, conf, y_h, y_c, sup, row_bar, ts_ns, windows, row_ok, period,
                   eligible, excl, meta, reasons)


# ----------------------------------------------------------------------------- scaler artifact (SCALER.json)
def code_version(root=None) -> dict:
    """Commit of the code that fitted the constants (no paths, no host); ``None`` outside a git checkout."""
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True,
                               text=True, timeout=10)
        if commit.returncode != 0:
            return {"commit": None, "dirty": None}
        return {"commit": commit.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def scaler_artifact(ds: Dataset, **extra) -> dict:
    """The effective input-scaling policy of ``ds`` as a self-contained record: names in tensor order, constants
    (fixed_train: mean/std/variance per feature, fit rows and period, NaN accounting, std==0 rule), the window and
    stride, the source digests, the split contract and the code commit. For ``rolling_z`` no constants exist and the
    record says so (window of rows). ``extra`` (e.g. run_id, agent_state) is stored under ``"run"``."""
    sc, feats = ds.meta["scaling"], list(ds.meta["input_contract"]["features"])
    art = {"artifact": "SCALER", "schema_version": 1, "policy": sc["policy"], "features": feats, "n_features": len(feats),
           "tensor_order": "windows[row, slot, feature] follow this feature order; slot layout = input_contract",
           "window": {"observations": int(ds.meta["input_contract"]["observations"]),
                      "stride_rows": int(ds.meta["input_contract"]["stride_rows"]), "includes_current_completed_bar": True},
           "split_contract": ds.meta["split_contract"], "sources": ds.meta.get("sources"), "code": code_version(),
           "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "run": extra or {}}
    if sc["policy"] == "fixed_train":
        fd = sc["features"]
        art.update({"fit_period": list(sc["train"]), "fit_rows": int(sc["fit_rows"]), "ddof": int(sc["ddof"]),
                    "mean": [float(fd[f]["mean"]) for f in feats], "std": [float(fd[f]["std"]) for f in feats],
                    "std_raw": [float(fd[f]["std_raw"]) for f in feats], "variance": [float(fd[f]["variance"]) for f in feats],
                    "rule": [fd[f]["rule"] for f in feats], "fit_rows_per_feature": [int(fd[f]["fit_rows"]) for f in feats],
                    "nan_share_train": [float(fd[f]["nan_share_train"]) for f in feats],
                    "rows_excluded_warmup": int(sc["rows_excluded_warmup"]),
                    "rows_excluded_nan_windows": int(sc["rows_excluded_nan_windows"]), "nan_policy": sc["nan_policy"],
                    "transform": "(x - mean) / std per feature; std = 1 where rule == std_zero_to_one",
                    "stationarity": sc.get("stationarity")})
    else:
        art.update({"rolling_window_rows": int(sc["rows"]), "mean": None, "std": None, "variance": None,
                    "fit_period": None, "fit_rows": None, "note": sc["note"],
                    "transform": f"(x_i - mean[i-{SCALE_W}, i)) / std[i-{SCALE_W}, i) per feature, recomputed per row"})
    return art


def write_scaler(out_dir, ds: Dataset, **extra) -> Path:
    """Persist ``SCALER.json`` next to the checkpoints written into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SCALER_FILE
    path.write_text(json.dumps(scaler_artifact(ds, **extra), indent=1))
    return path


def read_scaler(run_dir) -> dict:
    """The scaler artifact of a run directory; ``SCALER_MISSING`` when absent (no hidden refit of statistics)."""
    path = Path(run_dir) / SCALER_FILE
    if not path.exists():
        raise ValueError(f"SCALER_MISSING:{SCALER_FILE}")
    art = json.loads(path.read_text())
    if art.get("artifact") != "SCALER" or art.get("policy") not in SCALINGS or not isinstance(art.get("features"), list):
        raise ValueError("SCALER_MALFORMED")
    return art


def verify_scaler(art: dict, ds: Dataset, rtol=1e-9, atol=1e-12) -> dict:
    """Refuse to score unless the dataset's policy, feature names/order and constants equal the artifact's."""
    feats = list(ds.meta["input_contract"]["features"])
    if art["policy"] != ds.meta["scaling"]["policy"]:
        raise ValueError(f"SCALER_POLICY_MISMATCH:{art['policy']}!={ds.meta['scaling']['policy']}")
    if list(art["features"]) != feats:
        raise ValueError(f"SCALER_FEATURES_MISMATCH:artifact={art['features']}:dataset={feats}")
    ic, aw = ds.meta["input_contract"], art.get("window") or {}
    if aw and (int(aw.get("observations", -1)) != int(ic["observations"]) or int(aw.get("stride_rows", -1)) != int(ic["stride_rows"])):
        raise ValueError(f"SCALER_WINDOW_LAYOUT_MISMATCH:{aw}!=({ic['observations']},{ic['stride_rows']})")
    if art["policy"] == "fixed_train":
        fd = ds.meta["scaling"]["features"]
        mean = np.array([fd[f]["mean"] for f in feats]); std = np.array([fd[f]["std"] for f in feats])
        if list(art.get("fit_period") or []) != list(ds.meta["scaling"]["train"]):
            raise ValueError(f"SCALER_FIT_PERIOD_MISMATCH:{art.get('fit_period')}!={ds.meta['scaling']['train']}")
        if not (np.allclose(np.asarray(art["mean"], float), mean, rtol=rtol, atol=atol)
                and np.allclose(np.asarray(art["std"], float), std, rtol=rtol, atol=atol)):
            raise ValueError("SCALER_CONSTANTS_MISMATCH")
    elif int(art.get("rolling_window_rows", -1)) != int(ds.meta["scaling"]["rows"]):
        raise ValueError("SCALER_WINDOW_MISMATCH")
    return art


def load_with_scaler(run_dir, admitted_dir, csv_dir, *, contracts=("dev3y",), verify=True, end="2025-01-01"):
    """Dataset for scoring/refitting the checkpoints of ``run_dir``: the policy and the split contract come from
    its ``SCALER.json`` (never from a default), the dataset is rebuilt with them and verified against the artifact.
    Returns (dataset, artifact). ``contracts`` lists the split arms the caller supports."""
    art = read_scaler(run_dir)
    arm = (art.get("split_contract") or {}).get("arm", "dev3y")
    if arm not in contracts:
        raise ValueError(f"SPLIT_CONTRACT_UNSUPPORTED:{arm}")
    if (art.get("sources") or {}).get("loader", "core15") != "core15":
        raise ValueError(f"SCALER_LOADER_UNSUPPORTED:{art['sources'].get('loader')}")
    ds = load(admitted_dir, csv_dir, verify=verify, end=end, scaling=art["policy"], contract=arm)
    verify_scaler(art, ds)
    return ds, art
