"""Support corrections for the 2026-09-30 lane.

Replication baseline keeps the plugin's close-only decision and next-open
market fill. Protective broker orders are a separate named experiment.
This module does not score, does not fit a noise model, and does not start B0.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Sequence

import pandas as pd

from app.config import DEFAULT_VALUES
from app.data_processor import LEGACY_OFFSET_UNIT

ELAPSED_OFFSET_UNIT = "hours"
SHORT_HORIZONS_HOURS = (1, 2, 3, 4, 5, 6)
LONG_HORIZONS_HOURS = (24, 48, 72, 96, 120, 144)
REQUIRED_HORIZONS_HOURS = SHORT_HORIZONS_HOURS + LONG_HORIZONS_HOURS
LONGEST_REQUIRED_HORIZON_HOURS = 144

RESERVED_START = pd.Timestamp("2019-05-16 00:00:00")
DESIGN_RESERVED_START = RESERVED_START

TRADE_DURATION_UNBOUNDED = "UNBOUNDED_TRADE_DURATION"
NOT_SEPARATED_BY_ORIGIN_CUT = "NOT_SEPARATED_BY_ORIGIN_CUT"
CENSORING_REPORTED = "REPORTED_NO_TIME_CENSOR"
RESOLUTION_EXPLICIT = "explicit_resolution_not_recovered_historical_run"
SWEEP_241_EXIT_VARIANT = "NOT_CHECKED"
B0_STATUS = "B0_NOT_STARTED"

HISTORIC_CPU_CEILING_SECONDS = Decimal("3600")
RETAINED_SWEEP_CPU_SECONDS = Decimal("1461.693")
RETAINED_SWEEP_CELLS = 241

_HOLD_BOUND_KEYS = frozenset(
    {
        "max_holding_bars",
        "max_hold_bars",
        "max_holding_hours",
        "max_trade_duration",
        "max_bars_in_trade",
        "max_holding_period",
    }
)

CENSORING_NOTE = (
    "UNBOUNDED_TRADE_DURATION. The plugin does not censor a trade at a "
    "horizon, at six days, or at the reserved cut. stop() closes a position "
    "only because the feed ended."
)


class CalibrationSetError(ValueError):
    """Raised when a calibration fixture is not a development-only set."""


class DevelopmentOriginError(ValueError):
    """Raised when a supplied origin is not strictly before the reserved cut."""


def baseline_config() -> dict:
    """Exit-variant baseline recorded in config and in this return.

    The value E resolves the contradiction between the old signature default
    and plugin_params. historical_run_recovered is false: this is not a
    recovered historical run. The retained 241-cell manifest was not checked.
    """
    from app.plugins.plugin_long_short_predictions import Plugin

    signature = inspect.signature(Plugin.HeuristicStrategy.__init__)
    return {
        "exit_variant": DEFAULT_VALUES["exit_variant"],
        "plugin_params_exit_variant": Plugin.plugin_params["exit_variant"],
        "signature_default_exit_variant": signature.parameters["exit_variant"].default,
        "exit_variant_resolution": DEFAULT_VALUES["exit_variant_resolution"],
        "historical_run_recovered": DEFAULT_VALUES["historical_run_recovered"],
        "resolution": RESOLUTION_EXPLICIT,
        "sweep_241_exit_variant": DEFAULT_VALUES["sweep_241_exit_variant"],
        "fill_semantics": "close_only_decision_next_open_market_fill",
        "protective_broker_orders": "separate_named_experiment_not_in_baseline",
        "note": (
            "Explicit resolution to variant E. Not a recovered historical run."
        ),
    }


def plugin_trade_duration(parameter_names: Iterable[str] | None = None) -> str:
    """Holding bound of the long/short plugin.

    next() closes on a close-only TP, a close-only SL, or a variant early
    exit. max_trades_per_5days counts entries inside five days. It is not a
    maximum holding period. No six-day cap is invented here.
    """
    if parameter_names is None:
        from app.plugins.plugin_long_short_predictions import Plugin

        parameter_names = Plugin.plugin_params.keys()
    if set(parameter_names) & _HOLD_BOUND_KEYS:
        raise ValueError("holding bound present; refusing to invent a six-day cap")
    return TRADE_DURATION_UNBOUNDED


def _elapsed_column(hours: int) -> str:
    return f"elapsed_{int(hours)}h"


def create_elapsed_hour_predictions(
    frame: pd.DataFrame,
    horizons_hours: Sequence[int],
    *,
    price_column: str = "CLOSE",
) -> pd.DataFrame:
    """Predictions at exact elapsed hours.

    Column h is `price_column` at timestamp t + h hours. An origin is
    excluded when any requested horizon has no exact target bar. 144 hours
    is 144 elapsed hours, never 144 rows.
    """
    if ELAPSED_OFFSET_UNIT != "hours":
        raise RuntimeError("elapsed generator unit must be hours")
    if not horizons_hours:
        raise ValueError("horizons_hours is empty")
    if any(int(hours) <= 0 for hours in horizons_hours):
        raise ValueError("horizons must be positive elapsed hours")
    if price_column not in frame.columns:
        raise ValueError(f"missing price column {price_column}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("frame index must be a DatetimeIndex of bar timestamps")

    index = pd.DatetimeIndex(frame.index)
    if index.has_duplicates:
        raise ValueError("bar timestamps must be unique")

    prices = frame[price_column]
    lookup = {pd.Timestamp(ts): prices.iloc[i] for i, ts in enumerate(index)}
    rows: list[list] = []
    origins: list[pd.Timestamp] = []
    requested = tuple(int(hours) for hours in horizons_hours)
    for origin in index:
        origin_ts = pd.Timestamp(origin)
        values = []
        missing = False
        for hours in requested:
            target = origin_ts + pd.Timedelta(hours=hours)
            if target not in lookup:
                missing = True
                break
            values.append(lookup[target])
        if missing:
            continue
        rows.append(values)
        origins.append(origin_ts)

    out = pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(origins, name="DATE_TIME"),
        columns=[_elapsed_column(hours) for hours in requested],
    )
    out.attrs["offset_unit"] = ELAPSED_OFFSET_UNIT
    out.attrs["horizons_hours"] = requested
    out.attrs["legacy_offset_unit"] = LEGACY_OFFSET_UNIT
    return out


@dataclass(frozen=True)
class OriginSupport:
    origin: pd.Timestamp
    latest_target_timestamp: pd.Timestamp
    exact_target_bar_present: bool
    purged: bool
    trade_duration: str
    trade_exit_support: str
    cost_support: str
    censoring: str
    separation: str


@dataclass(frozen=True)
class SupportPopulation:
    reserved_start: pd.Timestamp
    reserved_window_rewritten: bool
    longest_horizon_hours: int
    required_horizons_hours: tuple[int, ...]
    trade_duration: str
    censoring: str
    censoring_note: str
    origins: tuple[OriginSupport, ...]
    n_development_origins: int
    n_purged: int
    n_exact_latest_target: int
    n_missing_exact_latest_target: int
    n_target_timestamp_before_cut: int
    n_entire_target_support_before_cut: int
    n_not_separated_by_origin_cut: int
    alternative_latest_development_origin: pd.Timestamp | None
    population_label: str


def _as_timestamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        raise ValueError("timestamps must be timezone-naive")
    return stamp


def derive_development_support(
    bars: pd.DataFrame,
    development_origins: Iterable,
    *,
    reserved_start: pd.Timestamp = DESIGN_RESERVED_START,
    longest_horizon_hours: int = LONGEST_REQUIRED_HORIZON_HOURS,
    required_horizons_hours: Sequence[int] | None = None,
    population_label: str = "UNLABELED",
) -> SupportPopulation:
    """Target support and trade-exit support for each development origin.

    The latest target is origin + longest_horizon_hours on the elapsed-hour
    path. An origin is purged when that timestamp is on or after
    reserved_start. Trade duration is unbounded, so no origin is treated as
    reserved-safe. The alternative boundary is reported and does not replace
    reserved_start.
    """
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars index must be a DatetimeIndex")
    bar_index = pd.DatetimeIndex(bars.index)
    if bar_index.has_duplicates:
        raise ValueError("bar timestamps must be unique")
    bar_keys = {pd.Timestamp(ts) for ts in bar_index}

    reserved = _as_timestamp(reserved_start)
    if reserved != DESIGN_RESERVED_START:
        raise ValueError("refusing to rewrite the reserved window")

    longest = int(longest_horizon_hours)
    if longest <= 0:
        raise ValueError("longest horizon must be positive")
    required = (
        tuple(int(hours) for hours in required_horizons_hours)
        if required_horizons_hours is not None
        else REQUIRED_HORIZONS_HOURS
    )
    if not required or any(hours <= 0 for hours in required):
        raise ValueError("required horizons must be positive elapsed hours")
    if max(required) != longest:
        raise ValueError("longest_horizon_hours must be the longest required horizon")

    trade_duration = plugin_trade_duration()
    if trade_duration != TRADE_DURATION_UNBOUNDED:
        raise ValueError("refusing to treat a new holding bound as a six-day purge")

    origin_list = tuple(_as_timestamp(origin) for origin in development_origins)
    if len(origin_list) != len(set(origin_list)):
        raise DevelopmentOriginError("duplicate development origin")
    for origin in origin_list:
        if origin >= reserved:
            raise DevelopmentOriginError(
                "development origin is on or after the reserved cut"
            )

    rows: list[OriginSupport] = []
    entire_before: list[pd.Timestamp] = []
    for origin in origin_list:
        latest = origin + pd.Timedelta(hours=longest)
        exact = latest in bar_keys
        purged = latest >= reserved
        all_targets_present = all(
            (origin + pd.Timedelta(hours=hours)) in bar_keys for hours in required
        )
        if all_targets_present and latest < reserved:
            entire_before.append(origin)
        rows.append(
            OriginSupport(
                origin=origin,
                latest_target_timestamp=latest,
                exact_target_bar_present=exact,
                purged=purged,
                trade_duration=trade_duration,
                trade_exit_support=trade_duration,
                cost_support=trade_duration,
                censoring=CENSORING_REPORTED,
                separation=NOT_SEPARATED_BY_ORIGIN_CUT,
            )
        )

    alternative = max(entire_before) if entire_before else None
    n_exact = sum(1 for row in rows if row.exact_target_bar_present)
    n_before = sum(1 for row in rows if row.latest_target_timestamp < reserved)
    return SupportPopulation(
        reserved_start=DESIGN_RESERVED_START,
        reserved_window_rewritten=False,
        longest_horizon_hours=longest,
        required_horizons_hours=required,
        trade_duration=trade_duration,
        censoring=CENSORING_REPORTED,
        censoring_note=CENSORING_NOTE,
        origins=tuple(rows),
        n_development_origins=len(rows),
        n_purged=sum(1 for row in rows if row.purged),
        n_exact_latest_target=n_exact,
        n_missing_exact_latest_target=len(rows) - n_exact,
        n_target_timestamp_before_cut=n_before,
        n_entire_target_support_before_cut=len(entire_before),
        n_not_separated_by_origin_cut=sum(
            1 for row in rows if row.separation == NOT_SEPARATED_BY_ORIGIN_CUT
        ),
        alternative_latest_development_origin=alternative,
        population_label=population_label,
    )


def synthetic_elapsed_support_frame() -> tuple[pd.DataFrame, tuple[pd.Timestamp, ...]]:
    """Nine constructed hourly timestamps. Not a market sample.

    Development origins are the five supplied timestamps, not every bar
    before the reserved cut. The later 15 May bars are target bars only.
    """
    rows = (
        ("2019-05-09 20:00:00", 1.0),
        ("2019-05-09 22:00:00", 1.1),
        ("2019-05-09 23:00:00", 1.2),
        ("2019-05-10 00:00:00", 1.3),
        ("2019-05-15 12:00:00", 1.4),
        ("2019-05-15 20:00:00", 1.5),
        ("2019-05-15 22:00:00", 1.6),
        ("2019-05-15 23:00:00", 1.7),
        ("2019-05-16 00:00:00", 1.8),
    )
    index = pd.DatetimeIndex((pd.Timestamp(stamp) for stamp, _ in rows), name="DATE_TIME")
    close = [price for _, price in rows]
    frame = pd.DataFrame(
        {
            "OPEN": [price + 0.01 for price in close],
            "LOW": [price - 0.01 for price in close],
            "HIGH": [price + 0.02 for price in close],
            "CLOSE": close,
        },
        index=index,
    )
    origins = (
        pd.Timestamp("2019-05-09 20:00:00"),
        pd.Timestamp("2019-05-09 22:00:00"),
        pd.Timestamp("2019-05-09 23:00:00"),
        pd.Timestamp("2019-05-10 00:00:00"),
        pd.Timestamp("2019-05-15 12:00:00"),
    )
    return frame, origins


def calibration_set(timestamps: Iterable) -> pd.DatetimeIndex:
    """Development origins accepted for later calibration. Does not fit noise.

    Any timestamp on or after 2019-05-16 00:00 is rejected. No correlation,
    scale, or noise model is estimated.
    """
    values = [_as_timestamp(stamp) for stamp in timestamps]
    if not values:
        raise CalibrationSetError("calibration set is empty")
    if len(values) != len(set(values)):
        raise CalibrationSetError("duplicate timestamps")
    if any(stamp >= DESIGN_RESERVED_START for stamp in values):
        raise CalibrationSetError(
            "calibration refuses a timestamp on or after 2019-05-16 00:00"
        )
    return pd.DatetimeIndex(sorted(values), name="DATE_TIME")


calibration_set.fits_noise_model = False


def cpu_reconciliation() -> dict:
    """Subtract the retained sweep from the historic ceiling. Do not start B0.

    The ceiling is the prior run's bound, not a fresh allowance. The
    remainder is not permission to spend.
    """
    remainder = HISTORIC_CPU_CEILING_SECONDS - RETAINED_SWEEP_CPU_SECONDS
    subtraction = (
        f"{HISTORIC_CPU_CEILING_SECONDS} - {RETAINED_SWEEP_CPU_SECONDS} = {remainder}"
    )
    return {
        "historic_ceiling_cpu_seconds": HISTORIC_CPU_CEILING_SECONDS,
        "ceiling_is_fresh_allowance": False,
        "retained_sweep_cells": RETAINED_SWEEP_CELLS,
        "retained_sweep_cpu_seconds": RETAINED_SWEEP_CPU_SECONDS,
        "subtraction": subtraction,
        "remainder_cpu_seconds": remainder,
        "remainder_is_permission": False,
        "status": B0_STATUS,
    }
