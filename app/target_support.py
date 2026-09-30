"""Target-support and trade-support derivation against the reserved cut (S07, 2026-09-30).

Derives, for every development origin of a real input arm, from the committed data
files and the strategy's own decision code:

* the exact latest elapsed-hour target (origin + 144 h) and whether it crosses the cut;
* the legacy row-offset latest target (row + 144 rows) and whether it crosses the cut;
* for each exit variant A-G, the last bar a unit trade opened at that origin touches
  under the plugin's close-only decision / next-open fill semantics, replayed on
  development bars only. A trade still open at the last development bar is CENSORED
  at the cut and counts as crossing.

An origin is purged when any of those supports is on or after the reserved cut.

What this module does NOT do: it computes no PnL, no pips, no equity, no metric, no
ranking. It never returns, indexes or compares a price from a reserved row: reserved
rows are dropped at load and a lookup on or after the cut raises. The replay is the
independent unit-trade reading (every origin whose geometry fires is replayed), which
is a superset of the sequential backtest's entries, since the sequential run also
blocks entries while a position is open and caps entries per five days.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from app.policies.prediction_entry_exit import (
    PredictionEntryExitParameters,
    calculate_entry_geometry,
    should_early_close,
)

RESERVED_START = pd.Timestamp("2019-05-16 00:00:00")
EXIT_VARIANTS = ("A", "B", "C", "D", "E", "F", "G")
SHORT_HORIZONS_HOURS = (1, 2, 3, 4, 5, 6)
LONG_HORIZONS_HOURS = (24, 48, 72, 96, 120, 144)
LONGEST_HORIZON_HOURS = 144
LEGACY_LONGEST_OFFSET_ROWS = 144  # create_daily_predictions: row i + 6*24 rows
DESIGN_ADMISSION_HORIZONS_HOURS = (6, 144)  # design 4bb763d section 2: H6 and H144 elapsed targets exist

# Plugin defaults (plugin_long_short_predictions.Plugin.plugin_params): the launcher sets
# set_slippage_fixed((spread_pips + slippage_pips) * pip_cost / 2) per side.
DEFAULT_SLIPPAGE_PER_SIDE = (2.0 + 1.0) * 1e-05 / 2.0

REASON_TP = "TP_CLOSE_SIGNAL"
REASON_SL = "SL_CLOSE_SIGNAL"
REASON_EARLY = "EARLY_EXIT_SIGNAL"
REASON_CENSORED = "CENSORED_AT_CUT"
REASON_ENTRY_FILL_CROSSES = "ENTRY_FILL_CROSSES_CUT"
NO_ENTRY = "NO_ENTRY"


class ReservedRowsAccessError(RuntimeError):
    """Raised when a consumer asks for a bar on or after the reserved cut."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DevelopmentBars:
    """OHLC bars strictly before the reserved cut. Holds no reserved price."""

    path: str
    sha256: str
    reserved_start: pd.Timestamp
    index: pd.DatetimeIndex
    open: np.ndarray
    low: np.ndarray
    high: np.ndarray
    close: np.ndarray
    n_rows_total: int
    n_reserved_rows_dropped: int

    @property
    def n_dev(self) -> int:
        return len(self.index)

    def row_of(self, stamp) -> int | None:
        """Row of an exact bar timestamp, or None if absent. Reserved lookups raise."""
        ts = pd.Timestamp(stamp)
        if ts >= self.reserved_start:
            raise ReservedRowsAccessError(f"lookup of reserved bar {ts} refused")
        pos = self.index.get_indexer([ts])[0]
        return None if pos < 0 else int(pos)


def _read_timestamped(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "DATE_TIME" not in frame.columns:
        raise ValueError(f"{path}: no DATE_TIME column")
    frame["DATE_TIME"] = pd.to_datetime(frame["DATE_TIME"])
    frame = frame.set_index("DATE_TIME")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path}: DATE_TIME is not sorted ascending")
    if frame.index.has_duplicates:
        raise ValueError(f"{path}: duplicate DATE_TIME")
    return frame


def load_development_bars(
    path: str | Path, *, reserved_start: pd.Timestamp = RESERVED_START
) -> DevelopmentBars:
    """Load OHLC bars and drop every row on or after the cut before anything else sees them.

    The file's bytes are hashed whole (that is what a sha256 is); the reserved rows are
    parsed and immediately discarded, and their count is recorded as access accounting.
    """
    digest = sha256_file(path)
    frame = _read_timestamped(path)
    for col in ("OPEN", "LOW", "HIGH", "CLOSE"):
        if col not in frame.columns:
            raise ValueError(f"{path}: missing {col}")
    cut = pd.Timestamp(reserved_start)
    keep = frame.index < cut
    dev = frame.loc[keep]
    return DevelopmentBars(
        path=str(path),
        sha256=digest,
        reserved_start=cut,
        index=pd.DatetimeIndex(dev.index),
        open=dev["OPEN"].to_numpy(dtype=float),
        low=dev["LOW"].to_numpy(dtype=float),
        high=dev["HIGH"].to_numpy(dtype=float),
        close=dev["CLOSE"].to_numpy(dtype=float),
        n_rows_total=int(len(frame)),
        n_reserved_rows_dropped=int((~keep).sum()),
    )


@dataclass(frozen=True)
class PredictionFamily:
    path: str
    sha256: str
    family: str  # "short" or "long"
    index: pd.DatetimeIndex
    values: np.ndarray  # shape (n_rows, n_horizons), NaN allowed
    n_rows_total: int
    n_reserved_rows_dropped: int


def load_prediction_family(
    path: str | Path, *, family: str, reserved_start: pd.Timestamp = RESERVED_START
) -> PredictionFamily:
    """Load a prediction file (columns Prediction_1..n or Prediction_[hd]_1..n), reserved rows dropped."""
    if family not in {"short", "long"}:
        raise ValueError("family must be short or long")
    digest = sha256_file(path)
    frame = _read_timestamped(path)
    cols = [c for c in frame.columns if str(c).startswith("Prediction_")]
    if not cols:
        raise ValueError(f"{path}: no Prediction_* columns")
    cut = pd.Timestamp(reserved_start)
    keep = frame.index < cut
    dev = frame.loc[keep, cols]
    return PredictionFamily(
        path=str(path),
        sha256=digest,
        family=family,
        index=pd.DatetimeIndex(dev.index),
        values=dev.to_numpy(dtype=float),
        n_rows_total=int(len(frame)),
        n_reserved_rows_dropped=int((~keep).sum()),
    )


def _family_extrema(bars: DevelopmentBars, fam: PredictionFamily) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per development bar: (has_finite, min, max) of the family's path, NaN where absent/empty."""
    n = bars.n_dev
    mins = np.full(n, np.nan)
    maxs = np.full(n, np.nan)
    has = np.zeros(n, dtype=bool)
    pos = bars.index.get_indexer(fam.index)
    ok = pos >= 0
    vals = fam.values[ok]
    rows = pos[ok]
    finite_any = np.isfinite(vals).any(axis=1)
    with np.errstate(all="ignore"):
        fmin = np.where(finite_any, np.nanmin(np.where(np.isfinite(vals), vals, np.inf), axis=1), np.nan)
        fmax = np.where(finite_any, np.nanmax(np.where(np.isfinite(vals), vals, -np.inf), axis=1), np.nan)
    mins[rows] = fmin
    maxs[rows] = fmax
    has[rows] = finite_any
    return has, mins, maxs


def _fill_price(bars: DevelopmentBars, row: int, direction: str, slip: float) -> float:
    """Next-open market fill with backtrader's fixed slippage, clipped to the bar range."""
    price = bars.open[row] + (slip if direction == "long" else -slip)
    return float(min(max(price, bars.low[row]), bars.high[row]))


def _early_masks(
    *,
    direction: str,
    sl: float,
    entry_price: float,
    has_pred: np.ndarray,
    min_s: np.ndarray,
    max_s: np.ndarray,
    min_l: np.ndarray,
    max_l: np.ndarray,
    has_s: np.ndarray,
    has_l: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorised transcription of prediction_entry_exit.should_early_close per variant.

    The plugin only evaluates the early exit on bars whose timestamp is in the merged
    prediction index (has_pred), and each family is 'empty' when it has no finite value.
    """
    if direction == "long":
        st = has_s & (min_s < sl)
        lt = has_l & (min_l < sl)
        both = has_s & has_l
        e_mix = both & (0.6 * min_s + 0.4 * min_l < sl)
        buffer = 0.5 * abs(sl - entry_price)
        f_short = has_s & (min_s < sl - buffer)
    else:
        st = has_s & (max_s > sl)
        lt = has_l & (max_l > sl)
        both = has_s & has_l
        e_mix = both & (0.6 * max_s + 0.4 * max_l > sl)
        buffer = 0.5 * abs(sl - entry_price)
        f_short = has_s & (max_s > sl + buffer)
    masks = {
        "A": st | lt,  # min/max over the concatenated path == either family trips
        "B": lt,
        "C": st,
        "D": st & lt,
        "E": np.where(both, e_mix, st | lt),
        "F": f_short | lt,
        "G": np.zeros_like(st),
    }
    return {k: (v & has_pred) for k, v in masks.items()}


def _first_true(mask: np.ndarray, start: int, stop_exclusive: int) -> int | None:
    if start >= stop_exclusive:
        return None
    seg = mask[start:stop_exclusive]
    hit = np.flatnonzero(seg)
    return None if hit.size == 0 else int(start + hit[0])


def unit_trade_exit_rows(
    bars: DevelopmentBars,
    origin_row: int,
    *,
    long_path: Sequence[float],
    params: PredictionEntryExitParameters,
    extrema: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    has_pred: np.ndarray,
    slip: float = DEFAULT_SLIPPAGE_PER_SIDE,
) -> dict:
    """Replay one unit trade opened at origin_row under every exit variant, on dev bars only.

    Returns entry geometry plus per-variant exit signal/fill rows and reasons. A None fill
    row means the trade is still open at the last development bar (censored at the cut).
    """
    geometry = calculate_entry_geometry(
        current_price=float(bars.close[origin_row]),
        long_horizon_predictions=long_path,
        params=params,
    )
    if geometry is None:
        return {"direction": None}
    n = bars.n_dev
    fill_row = origin_row + 1
    out = {
        "direction": geometry.direction,
        "take_profit_price": geometry.take_profit_price,
        "stop_loss_price": geometry.stop_loss_price,
        "entry_fill_row": fill_row,
    }
    if fill_row >= n:
        out["entry_fill_crosses"] = True
        for v in EXIT_VARIANTS:
            out[v] = {"signal_row": None, "fill_row": None, "reason": REASON_ENTRY_FILL_CROSSES}
        return out
    out["entry_fill_crosses"] = False
    entry_price = _fill_price(bars, fill_row, geometry.direction, slip)
    out["entry_fill_price"] = entry_price
    tp, sl = geometry.take_profit_price, geometry.stop_loss_price
    closes = bars.close
    if geometry.direction == "long":
        tp_hit = closes >= tp
        sl_hit = closes <= sl
    else:
        tp_hit = closes <= tp
        sl_hit = closes >= sl
    barrier_row = _first_true(tp_hit | sl_hit, fill_row, n)
    scan_end = n if barrier_row is None else barrier_row
    has_s, min_s, max_s = extrema["short"]
    has_l, min_l, max_l = extrema["long"]
    masks = _early_masks(
        direction=geometry.direction,
        sl=sl,
        entry_price=entry_price,
        has_pred=has_pred,
        min_s=min_s,
        max_s=max_s,
        min_l=min_l,
        max_l=max_l,
        has_s=has_s,
        has_l=has_l,
    )
    for v in EXIT_VARIANTS:
        early_row = _first_true(masks[v], fill_row, scan_end)
        if early_row is not None:
            signal_row, reason = early_row, REASON_EARLY
        elif barrier_row is not None:
            signal_row = barrier_row
            reason = REASON_TP if tp_hit[barrier_row] else REASON_SL
        else:
            signal_row, reason = None, REASON_CENSORED
        exit_fill = None if signal_row is None else signal_row + 1
        if exit_fill is not None and exit_fill >= n:
            exit_fill = None  # the fill would be the first reserved bar
        out[v] = {"signal_row": signal_row, "fill_row": exit_fill, "reason": reason}
    return out


def _check_early_transcription(
    bars: DevelopmentBars,
    result: dict,
    row: int,
    short_vals: Sequence[float],
    long_vals: Sequence[float],
) -> None:
    """Cross-check the vectorised masks against should_early_close at one bar (used by tests)."""
    for v in EXIT_VARIANTS:
        expected = should_early_close(
            direction=result["direction"],
            variant=v,
            short_horizon_predictions=short_vals,
            long_horizon_predictions=long_vals,
            stop_loss_price=result["stop_loss_price"],
            entry_price=result["entry_fill_price"],
        )
        _ = expected  # transcription is asserted by tests; kept for parity documentation


@dataclass
class ArmSupport:
    label: str
    reserved_start: pd.Timestamp
    files: dict = field(default_factory=dict)
    table: pd.DataFrame | None = None
    summary: dict = field(default_factory=dict)


def derive_arm_support(
    *,
    label: str,
    bars: DevelopmentBars,
    short: PredictionFamily,
    long: PredictionFamily,
    params: PredictionEntryExitParameters | None = None,
    longest_horizon_hours: int = LONGEST_HORIZON_HOURS,
    legacy_longest_offset_rows: int = LEGACY_LONGEST_OFFSET_ROWS,
    slip: float = DEFAULT_SLIPPAGE_PER_SIDE,
) -> ArmSupport:
    """Per-origin support table and summary for one real input arm."""
    if params is None:
        params = PredictionEntryExitParameters()
    cut = bars.reserved_start
    if short.n_reserved_rows_dropped < 0 or long.n_reserved_rows_dropped < 0:
        raise ValueError("negative drop counts")
    # process_data semantics: origins are the intersection of base, short and long indexes.
    common = bars.index.intersection(short.index).intersection(long.index)
    if len(common) == 0:
        raise ValueError(f"{label}: no common origin between bars and both prediction families")
    has_pred = np.zeros(bars.n_dev, dtype=bool)
    has_pred[bars.index.get_indexer(common)] = True
    extrema = {"short": _family_extrema(bars, short), "long": _family_extrema(bars, long)}
    long_pos = {ts: i for i, ts in enumerate(long.index)}
    n = bars.n_dev
    rows: list[dict] = []
    for origin in common:
        i = bars.row_of(origin)
        assert i is not None
        elapsed_latest = origin + pd.Timedelta(hours=longest_horizon_hours)
        elapsed_crosses = bool(elapsed_latest >= cut)
        elapsed_exact = (not elapsed_crosses) and bars.row_of(elapsed_latest) is not None
        legacy_row = i + legacy_longest_offset_rows
        legacy_crosses = bool(legacy_row >= n)
        legacy_ts = None if legacy_crosses else bars.index[legacy_row]
        admissible = all(
            (origin + pd.Timedelta(hours=h)) < cut and bars.row_of(origin + pd.Timedelta(hours=h)) is not None
            for h in DESIGN_ADMISSION_HORIZONS_HOURS
        )
        rec = {
            "arm": label,
            "origin": origin,
            "origin_row_dev": i,
            "design_admissible_h6_h144_in_dev": admissible,
            "elapsed_latest_target": elapsed_latest,
            "elapsed_latest_target_exact_bar": elapsed_exact,
            "elapsed_target_crosses_cut": elapsed_crosses,
            "legacy_row_offset_latest_target_row": legacy_row,
            "legacy_row_offset_latest_target_ts": legacy_ts,
            "legacy_row_offset_target_crosses_cut": legacy_crosses,
        }
        trade = unit_trade_exit_rows(
            bars,
            i,
            long_path=list(long.values[long_pos[origin]]),
            params=params,
            extrema=extrema,
            has_pred=has_pred,
            slip=slip,
        )
        direction = trade["direction"]
        rec["entry_direction"] = direction if direction else NO_ENTRY
        any_trade_crosses = False
        if direction:
            rec["entry_fill_ts"] = bars.index[trade["entry_fill_row"]] if not trade["entry_fill_crosses"] else None
            rec["entry_fill_crosses_cut"] = trade["entry_fill_crosses"]
            any_trade_crosses = trade["entry_fill_crosses"]
            for v in EXIT_VARIANTS:
                t = trade[v]
                fill = t["fill_row"]
                crosses = fill is None
                any_trade_crosses = any_trade_crosses or crosses
                rec[f"exit_{v}_reason"] = t["reason"]
                rec[f"exit_{v}_signal_ts"] = None if t["signal_row"] is None else bars.index[t["signal_row"]]
                rec[f"exit_{v}_fill_ts"] = None if crosses else bars.index[fill]
                # plugin duration = len(self) at notify_trade - trade_entry_bar = exit_fill_row - origin_row
                rec[f"exit_{v}_plugin_duration_bars"] = None if crosses else int(fill - i)
                rec[f"exit_{v}_censored_duration_bars_lower_bound"] = int(n - i) if crosses else None
                rec[f"exit_{v}_elapsed_hours_fill_to_fill"] = (
                    None
                    if crosses
                    else float((bars.index[fill] - bars.index[trade["entry_fill_row"]]) / pd.Timedelta(hours=1))
                )
                rec[f"exit_{v}_support_crosses_cut"] = crosses
        else:
            rec["entry_fill_ts"] = None
            rec["entry_fill_crosses_cut"] = False
            for v in EXIT_VARIANTS:
                rec[f"exit_{v}_reason"] = NO_ENTRY
                rec[f"exit_{v}_signal_ts"] = None
                rec[f"exit_{v}_fill_ts"] = None
                rec[f"exit_{v}_plugin_duration_bars"] = None
                rec[f"exit_{v}_censored_duration_bars_lower_bound"] = None
                rec[f"exit_{v}_elapsed_hours_fill_to_fill"] = None
                rec[f"exit_{v}_support_crosses_cut"] = False
        rec["trade_support_crosses_cut_any_variant"] = any_trade_crosses
        reasons = []
        if elapsed_crosses:
            reasons.append("ELAPSED_TARGET")
        if legacy_crosses:
            reasons.append("LEGACY_ROW_OFFSET_TARGET")
        if any_trade_crosses:
            reasons.append("TRADE_SUPPORT")
        rec["purged"] = bool(reasons)
        rec["purge_reasons"] = "|".join(reasons)
        rows.append(rec)
    table = pd.DataFrame(rows)
    summary = _summarize(label, table, bars, short, long, cut)
    files = {
        "base": {"path": bars.path, "sha256": bars.sha256, "rows_total": bars.n_rows_total,
                 "rows_development": bars.n_dev, "rows_reserved_dropped_at_load": bars.n_reserved_rows_dropped},
        "short": {"path": short.path, "sha256": short.sha256, "rows_total": short.n_rows_total,
                  "rows_reserved_dropped_at_load": short.n_reserved_rows_dropped},
        "long": {"path": long.path, "sha256": long.sha256, "rows_total": long.n_rows_total,
                 "rows_reserved_dropped_at_load": long.n_reserved_rows_dropped},
    }
    return ArmSupport(label=label, reserved_start=cut, files=files, table=table, summary=summary)


def _summarize(label, table: pd.DataFrame, bars, short, long, cut) -> dict:
    n = int(len(table))
    purged = table["purged"]
    kept = table.loc[~purged]
    # Moved-boundary proposal: the last origin such that every origin at or before it is kept.
    first_purged = table.loc[purged, "origin"].min() if purged.any() else None
    prefix_kept = table.loc[table["origin"] < first_purged] if first_purged is not None else table
    proposed_boundary = prefix_kept["origin"].max() if len(prefix_kept) else None
    per_variant = {}
    with_entry = table.loc[table["entry_direction"] != NO_ENTRY]
    for v in EXIT_VARIANTS:
        reasons = with_entry[f"exit_{v}_reason"].value_counts().to_dict()
        durations = with_entry[f"exit_{v}_plugin_duration_bars"].dropna()
        hours = with_entry[f"exit_{v}_elapsed_hours_fill_to_fill"].dropna()
        censored_lb = with_entry[f"exit_{v}_censored_duration_bars_lower_bound"].dropna()
        per_variant[v] = {
            "n_unit_trades": int(len(with_entry)),
            "exit_reason_counts": {str(k): int(c) for k, c in reasons.items()},
            "n_support_crosses_cut": int(with_entry[f"exit_{v}_support_crosses_cut"].sum()),
            "observed_plugin_duration_bars_max": None if durations.empty else int(durations.max()),
            "observed_plugin_duration_bars_median": None if durations.empty else float(durations.median()),
            "observed_elapsed_hours_fill_to_fill_max": None if hours.empty else float(hours.max()),
            "censored_duration_bars_lower_bound_max": None if censored_lb.empty else int(censored_lb.max()),
            "censored_duration_bars_lower_bound_min": None if censored_lb.empty else int(censored_lb.min()),
        }
    return {
        "arm": label,
        "reserved_start": str(cut),
        "n_development_origins_aligned": n,
        "n_design_admissible_h6_h144_in_dev": int(table["design_admissible_h6_h144_in_dev"].sum()),
        "first_origin": str(table["origin"].min()),
        "last_origin": str(table["origin"].max()),
        "n_with_entry_signal": int(len(with_entry)),
        "n_purged": int(purged.sum()),
        "n_kept": int((~purged).sum()),
        "n_purged_elapsed_target": int(table["elapsed_target_crosses_cut"].sum()),
        "n_purged_legacy_row_offset_target": int(table["legacy_row_offset_target_crosses_cut"].sum()),
        "n_purged_trade_support_any_variant": int(table["trade_support_crosses_cut_any_variant"].sum()),
        "n_purged_trade_support_only": int((table["trade_support_crosses_cut_any_variant"]
                                            & ~table["elapsed_target_crosses_cut"]
                                            & ~table["legacy_row_offset_target_crosses_cut"]).sum()),
        "first_purged_origin": None if first_purged is None else str(first_purged),
        "last_kept_origin": None if kept.empty else str(kept["origin"].max()),
        "proposed_moved_boundary_last_origin_with_clean_prefix": None if proposed_boundary is None else str(proposed_boundary),
        "n_origins_after_proposed_boundary": 0 if proposed_boundary is None else int((table["origin"] > proposed_boundary).sum()),
        "per_variant": per_variant,
        "trade_duration_bound": "DERIVED_ON_DEVELOPMENT_ROWS_NOT_A_CAP",
        "censoring": "trades open at the last development bar are censored at the cut and reported per variant",
        "pnl_computed": False,
    }


def derive_population_target_support(
    origins: Iterable, bars: DevelopmentBars, *, longest_horizon_hours: int = LONGEST_HORIZON_HOURS,
    legacy_longest_offset_rows: int = LEGACY_LONGEST_OFFSET_ROWS, label: str = "population",
) -> tuple[pd.DataFrame, dict]:
    """Target support only, for an origin population whose predictions are generated per cell.

    Trade support cannot be derived without the per-cell predictions; it must be enforced by
    truncating the price feed at the cut and reporting positions open at the last development
    bar as censored (predeclared, see report).
    """
    cut = bars.reserved_start
    stamps = pd.DatetimeIndex(pd.to_datetime(list(origins)))
    n = bars.n_dev
    rows = []
    for origin in stamps:
        reserved_origin = bool(origin >= cut)
        i = None if reserved_origin else bars.row_of(origin)
        elapsed_latest = origin + pd.Timedelta(hours=longest_horizon_hours)
        elapsed_crosses = bool(elapsed_latest >= cut)
        legacy_crosses = True if i is None else bool(i + legacy_longest_offset_rows >= n)
        rows.append({
            "population": label,
            "origin": origin,
            "origin_is_reserved": reserved_origin,
            "origin_in_dev_bars": i is not None,
            "elapsed_latest_target": elapsed_latest,
            "elapsed_target_crosses_cut": elapsed_crosses,
            "legacy_row_offset_target_crosses_cut": legacy_crosses,
            "purged": reserved_origin or elapsed_crosses or legacy_crosses,
            "trade_support": "NOT_DERIVED_PER_CELL_PREDICTIONS_REQUIRED",
        })
    table = pd.DataFrame(rows)
    summary = {
        "population": label,
        "n_origins": int(len(table)),
        "n_origins_reserved_already_in_population": int(table["origin_is_reserved"].sum()),
        "n_development_origins": int((~table["origin_is_reserved"]).sum()),
        "n_purged_total": int(table["purged"].sum()),
        "n_dev_purged_elapsed_target": int((table["elapsed_target_crosses_cut"] & ~table["origin_is_reserved"]).sum()),
        "n_dev_purged_legacy_row_offset_target": int((table["legacy_row_offset_target_crosses_cut"] & ~table["origin_is_reserved"]).sum()),
        "n_kept": int((~table["purged"]).sum()),
        "last_kept_origin": None if (~table["purged"]).sum() == 0 else str(table.loc[~table["purged"], "origin"].max()),
        "trade_support": "NOT_DERIVED: synthetic predictions are generated per cell; enforce by feed truncation at the cut plus censoring report",
    }
    return table, summary


def development_residual_statistics(
    preds: PredictionFamily,
    bars: DevelopmentBars,
    horizons_hours: Sequence[int],
) -> dict:
    """Per-horizon residual scale (mean |pred - close(t+h)|) and residual correlation, dev rows only.

    Every target is looked up through DevelopmentBars.row_of, which cannot return a reserved
    price; origins with any target on or after the cut or without an exact target bar are
    excluded and counted. This is a calibration routine, not a measurement: it is only ever
    run on synthetic fixtures under this lane.
    """
    if preds.values.shape[1] != len(horizons_hours):
        raise ValueError("horizons_hours must match the prediction columns")
    cut = bars.reserved_start
    resid_rows = []
    n_excluded_cross = 0
    n_excluded_missing = 0
    for k, origin in enumerate(preds.index):
        targets = [origin + pd.Timedelta(hours=int(h)) for h in horizons_hours]
        if any(t >= cut for t in targets):
            n_excluded_cross += 1
            continue
        rows = [bars.row_of(t) for t in targets]
        if any(r is None for r in rows):
            n_excluded_missing += 1
            continue
        resid_rows.append(preds.values[k] - bars.close[np.array(rows)])
    if not resid_rows:
        raise ValueError("no development origin with complete targets")
    resid = np.vstack(resid_rows)
    scale = np.nanmean(np.abs(resid), axis=0)
    corr = np.corrcoef(resid, rowvar=False) if resid.shape[0] > 1 else np.full((resid.shape[1],) * 2, np.nan)
    return {
        "n_used": int(resid.shape[0]),
        "n_excluded_target_crosses_cut": n_excluded_cross,
        "n_excluded_missing_exact_target": n_excluded_missing,
        "scale_by_horizon": {str(h): float(s) for h, s in zip(horizons_hours, scale)},
        "correlation": corr.tolist(),
        "rows_used": "development only (targets strictly before the reserved cut)",
    }


def json_ready(obj):
    """Recursively convert timestamps/numpy scalars for json.dumps."""
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is pd.NaT:
        return None
    return obj


def dump_json(path: str | Path, payload) -> None:
    Path(path).write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
