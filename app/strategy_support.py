"""Support corrections for the 2026-09-30 lane.

Replication baseline keeps the plugin's close-only decision and next-open
market fill. Protective broker orders are a separate named experiment.
This module does not score, does not fit a noise model, and does not start B0.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

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
# The design names this clock time. Comparisons use it as UTC only after the
# caller declares a source timezone. Historical CSV zones are not inferred.
RESERVED_START_UTC = pd.Timestamp("2019-05-16 00:00:00", tz="UTC")

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


class HorizonContractError(ValueError):
    """Raised when a horizon is not a positive unique integer."""


class TimestampContractError(ValueError):
    """Raised when timestamps are unordered, duplicated, or have no declared zone."""


class PriceContractError(ValueError):
    """Raised when a consumed price is missing or not finite."""


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
    return f"elapsed_{hours}h"


def _parse_hour(hours, *, what: str) -> int:
    """Accept a Python int only. bool, fractions, and numeric strings are rejected."""
    if isinstance(hours, bool) or type(hours) is not int:
        raise HorizonContractError(
            f"{what} must be a positive integer without coercion, got {hours!r}"
        )
    if hours <= 0:
        raise HorizonContractError(f"{what} must be a positive integer")
    return hours


def parse_horizons(horizons, *, what: str = "horizons_hours") -> tuple[int, ...]:
    if isinstance(horizons, (str, bytes)) or not isinstance(horizons, Sequence):
        raise HorizonContractError(f"{what} must be a sequence of positive integers")
    if len(horizons) == 0:
        raise HorizonContractError(f"{what} is empty")
    parsed = tuple(_parse_hour(hours, what=what) for hours in horizons)
    if len(set(parsed)) != len(parsed):
        raise HorizonContractError(f"{what} contains duplicate horizons")
    return parsed


def _finite_price(value, *, where: str) -> float:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        raise PriceContractError(f"non-finite price at {where}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PriceContractError(f"non-finite price at {where}") from exc
    if not math.isfinite(number):
        raise PriceContractError(f"non-finite price at {where}")
    return number


def _declared_zone(source_timezone: str) -> ZoneInfo:
    if not isinstance(source_timezone, str) or not source_timezone.strip():
        raise TimestampContractError(
            "source timezone must be declared; refusing to guess"
        )
    name = source_timezone.strip()
    if name.lower() in {"local", "naive", "guess", "none"}:
        raise TimestampContractError(
            "source timezone must be declared; refusing to guess"
        )
    try:
        return ZoneInfo(name)
    except Exception as exc:
        raise TimestampContractError(f"unknown source timezone {name!r}") from exc


def _same_zone(index_tz, declared: ZoneInfo, name: str) -> bool:
    key = getattr(index_tz, "key", None)
    label = str(index_tz)
    if key == name or label == name:
        return True
    if name == "UTC" and label in {"UTC", "tzutc()"}:
        return True
    return key is not None and key == getattr(declared, "key", None)


def normalize_timestamp_index(index, *, source_timezone: str) -> pd.DatetimeIndex:
    """Localize with the declared zone when naive, then convert to UTC.

    Ambiguous or nonexistent civil times raise. The zone of a historical
    series is never inferred.
    """
    declared = _declared_zone(source_timezone)
    name = source_timezone.strip()
    raw = pd.DatetimeIndex(pd.to_datetime(index))
    if not isinstance(raw, pd.DatetimeIndex):
        raise TimestampContractError("timestamps must form a DatetimeIndex")
    if raw.tz is None:
        try:
            aware = raw.tz_localize(declared, ambiguous="raise", nonexistent="raise")
        except Exception as exc:
            raise TimestampContractError(
                "naive timestamps could not be localized in the declared timezone"
            ) from exc
    else:
        if not _same_zone(raw.tz, declared, name):
            raise TimestampContractError(
                "index timezone does not match the declared source timezone"
            )
        aware = raw
    utc = pd.DatetimeIndex(aware.tz_convert("UTC"))
    if utc.has_duplicates:
        raise TimestampContractError("bar timestamps must be unique")
    if not bool(utc.is_monotonic_increasing):
        raise TimestampContractError("bar timestamps must be in temporal order")
    return pd.DatetimeIndex(utc, name=raw.name or "DATE_TIME")


def _normalize_one(value, *, source_timezone: str) -> pd.Timestamp:
    normalized = normalize_timestamp_index(
        pd.DatetimeIndex([pd.Timestamp(value)]),
        source_timezone=source_timezone,
    )
    return pd.Timestamp(normalized[0])


def create_elapsed_hour_predictions(
    frame: pd.DataFrame,
    horizons_hours: Sequence[int],
    *,
    price_column: str = "CLOSE",
    source_timezone: str | None = None,
) -> pd.DataFrame:
    """Predictions at exact elapsed hours, normalized to UTC.

    Column h is `price_column` at timestamp t + h hours. An origin is
    excluded when any requested horizon has no exact target bar. A missing
    bar is a gap, not a non-finite price. 144 hours is 144 elapsed hours,
    never 144 rows. `source_timezone` is required.
    """
    if ELAPSED_OFFSET_UNIT != "hours":
        raise RuntimeError("elapsed generator unit must be hours")
    requested = parse_horizons(horizons_hours)
    if price_column not in frame.columns:
        raise ValueError(f"missing price column {price_column}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TimestampContractError("frame index must be a DatetimeIndex of bar timestamps")

    index = normalize_timestamp_index(frame.index, source_timezone=source_timezone)
    aligned = frame.copy()
    aligned.index = index
    prices = aligned[price_column]
    lookup: dict[pd.Timestamp, float] = {}
    for stamp, raw in zip(index, prices.tolist()):
        key = pd.Timestamp(stamp)
        # Prices are validated when consumed. A gap is a missing key.
        lookup[key] = raw

    rows: list[list[float]] = []
    origins: list[pd.Timestamp] = []
    consumed: list[pd.Timestamp] = []
    for origin in index:
        origin_ts = pd.Timestamp(origin)
        values: list[float] = []
        targets: list[pd.Timestamp] = []
        missing = False
        for hours in requested:
            target = origin_ts + pd.Timedelta(hours=hours)
            if target not in lookup:
                missing = True
                break
            raw_price = lookup[target]
            values.append(_finite_price(raw_price, where=str(target)))
            targets.append(target)
        if missing:
            continue
        rows.append(values)
        origins.append(origin_ts)
        consumed.append(origin_ts)
        consumed.extend(targets)

    out = pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(origins, name="DATE_TIME"),
        columns=[_elapsed_column(hours) for hours in requested],
    )
    out.attrs["offset_unit"] = ELAPSED_OFFSET_UNIT
    out.attrs["horizons_hours"] = requested
    out.attrs["legacy_offset_unit"] = LEGACY_OFFSET_UNIT
    out.attrs["source_timezone"] = source_timezone.strip()
    out.attrs["normalized_timezone"] = "UTC"
    out.attrs["consumed_timestamps_utc"] = tuple(sorted(set(consumed)))
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
    source_timezone: str
    normalized_timezone: str
    reserved_start_utc: pd.Timestamp


def _as_timestamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        raise ValueError("timestamps must be timezone-naive")
    return stamp


def derive_development_support(
    bars: pd.DataFrame,
    development_origins: Iterable,
    *,
    source_timezone: str | None = None,
    reserved_start: pd.Timestamp = DESIGN_RESERVED_START,
    longest_horizon_hours: int = LONGEST_REQUIRED_HORIZON_HOURS,
    required_horizons_hours: Sequence[int] | None = None,
    population_label: str = "UNLABELED",
    price_column: str = "CLOSE",
) -> SupportPopulation:
    """Target support and trade-exit support for each development origin.

    The latest target is origin + longest_horizon_hours on the elapsed-hour
    path. An origin is purged when that UTC timestamp is on or after
    2019-05-16 00:00 UTC. Trade duration is unbounded, so no origin is
    treated as reserved-safe. The alternative boundary is reported and does
    not replace reserved_start. Horizons are not coerced. A present target
    with a non-finite price is rejected; a missing timestamp stays a gap.
    """
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TimestampContractError("bars index must be a DatetimeIndex")
    bar_index = normalize_timestamp_index(bars.index, source_timezone=source_timezone)
    declared = source_timezone.strip()
    reserved = _normalize_one(reserved_start, source_timezone=declared)
    if reserved != RESERVED_START_UTC:
        raise ValueError("refusing to rewrite the reserved window")

    longest = _parse_hour(longest_horizon_hours, what="longest_horizon_hours")
    required = (
        parse_horizons(required_horizons_hours, what="required_horizons_hours")
        if required_horizons_hours is not None
        else REQUIRED_HORIZONS_HOURS
    )
    if max(required) != longest:
        raise HorizonContractError(
            "longest_horizon_hours must be the longest required horizon"
        )

    trade_duration = plugin_trade_duration()
    if trade_duration != TRADE_DURATION_UNBOUNDED:
        raise ValueError("refusing to treat a new holding bound as a six-day purge")

    origin_list = tuple(
        _normalize_one(origin, source_timezone=declared) for origin in development_origins
    )
    if len(origin_list) != len(set(origin_list)):
        raise DevelopmentOriginError("duplicate development origin")
    for origin in origin_list:
        if origin >= RESERVED_START_UTC:
            raise DevelopmentOriginError(
                "development origin is on or after the reserved cut"
            )

    aligned = bars.copy()
    aligned.index = bar_index
    bar_keys = set(bar_index)
    if price_column in aligned.columns:
        for origin in origin_list:
            for hours in required:
                stamp = origin + pd.Timedelta(hours=hours)
                if stamp < RESERVED_START_UTC and stamp in bar_keys:
                    _finite_price(aligned.at[stamp, price_column], where=str(stamp))

    rows: list[OriginSupport] = []
    entire_before: list[pd.Timestamp] = []
    for origin in origin_list:
        latest = origin + pd.Timedelta(hours=longest)
        exact = latest in bar_keys
        purged = latest >= RESERVED_START_UTC
        all_targets_present = all(
            (origin + pd.Timedelta(hours=hours)) in bar_keys for hours in required
        )
        if all_targets_present and latest < RESERVED_START_UTC:
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
    n_before = sum(1 for row in rows if row.latest_target_timestamp < RESERVED_START_UTC)
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
        source_timezone=declared,
        normalized_timezone="UTC",
        reserved_start_utc=RESERVED_START_UTC,
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


def calibration_set(
    timestamps: Iterable,
    *,
    consumed: Iterable | None = None,
    source_timezone: str | None = None,
) -> pd.DatetimeIndex:
    """Development timestamps. Does not fit noise.

    A one-argument call only screens origin clocks and is not target
    admission. Pass `consumed` and `source_timezone` to admit the timestamps
    that are actually read: origins, targets, scales, and residuals. Any
    consumed UTC timestamp on or after 2019-05-16 00:00 is rejected.
    """
    if consumed is None and source_timezone is None:
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
    if source_timezone is None or consumed is None:
        raise CalibrationSetError(
            "calibration admission requires consumed support and a declared timezone"
        )
    origins = [_normalize_one(stamp, source_timezone=source_timezone) for stamp in timestamps]
    consumed_stamps = [
        _normalize_one(stamp, source_timezone=source_timezone) for stamp in consumed
    ]
    if not origins or not consumed_stamps:
        raise CalibrationSetError("calibration set is empty")
    if len(origins) != len(set(origins)) or len(consumed_stamps) != len(set(consumed_stamps)):
        raise CalibrationSetError("duplicate timestamps")
    if any(stamp >= RESERVED_START_UTC for stamp in origins):
        raise CalibrationSetError(
            "calibration refuses an origin on or after 2019-05-16 00:00 UTC"
        )
    if any(stamp >= RESERVED_START_UTC for stamp in consumed_stamps):
        raise CalibrationSetError(
            "consumed support is on or after 2019-05-16 00:00 UTC"
        )
    return pd.DatetimeIndex(sorted(origins), name="DATE_TIME")


calibration_set.fits_noise_model = False


@dataclass(frozen=True)
class DevelopmentParameters:
    """Scales and residuals on DEV support. Not a noise model."""

    admitted_origins: tuple[pd.Timestamp, ...]
    consumed_timestamps_utc: tuple[pd.Timestamp, ...]
    horizons_hours: tuple[int, ...]
    per_horizon_mean_abs_residual: tuple[tuple[int, float], ...]
    per_horizon_residual_sum: tuple[tuple[int, float], ...]
    fits_noise_model: bool = False

    @property
    def parameters(self) -> tuple:
        return (
            self.horizons_hours,
            self.admitted_origins,
            self.consumed_timestamps_utc,
            self.per_horizon_mean_abs_residual,
            self.per_horizon_residual_sum,
            self.fits_noise_model,
        )


def _dev_price_lookup(
    frame: pd.DataFrame,
    *,
    source_timezone: str,
    price_column: str,
) -> dict[pd.Timestamp, float]:
    """Prices strictly before the reserved cut. Later rows are not read."""
    if price_column not in frame.columns:
        raise ValueError(f"missing price column {price_column}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TimestampContractError("frame index must be a DatetimeIndex")
    index = normalize_timestamp_index(frame.index, source_timezone=source_timezone)
    aligned = frame.copy()
    aligned.index = index
    dev = aligned.loc[aligned.index < RESERVED_START_UTC]
    lookup: dict[pd.Timestamp, float] = {}
    for stamp, raw in zip(dev.index, dev[price_column].tolist()):
        key = pd.Timestamp(stamp)
        lookup[key] = _finite_price(raw, where=str(key))
    return lookup


def _residual_parameters(
    lookup: dict[pd.Timestamp, float],
    horizons: tuple[int, ...],
    origins: tuple[pd.Timestamp, ...],
) -> DevelopmentParameters:
    consumed: set[pd.Timestamp] = set()
    residual_sums = {hours: 0.0 for hours in horizons}
    abs_sums = {hours: 0.0 for hours in horizons}
    for origin in origins:
        origin_price = lookup[origin]
        consumed.add(origin)
        for hours in horizons:
            target = origin + pd.Timedelta(hours=hours)
            target_price = lookup[target]
            residual = target_price - origin_price
            residual_sums[hours] += residual
            abs_sums[hours] += abs(residual)
            consumed.add(target)
    count = len(origins)
    mean_abs = tuple(
        (hours, abs_sums[hours] / count) for hours in horizons
    )
    sums = tuple((hours, residual_sums[hours]) for hours in horizons)
    consumed_tuple = tuple(sorted(consumed))
    calibration_set(
        origins,
        consumed=consumed_tuple,
        source_timezone="UTC",
    )
    return DevelopmentParameters(
        admitted_origins=origins,
        consumed_timestamps_utc=consumed_tuple,
        horizons_hours=horizons,
        per_horizon_mean_abs_residual=mean_abs,
        per_horizon_residual_sum=sums,
        fits_noise_model=False,
    )


def admit_elapsed_hour_calibration(
    frame: pd.DataFrame,
    horizons_hours: Sequence[int],
    *,
    source_timezone: str,
    origins: Iterable | None = None,
    price_column: str = "CLOSE",
) -> DevelopmentParameters:
    """Admit origins whose targets, scale, and residual all sit inside DEV.

    Requested origins that would read a reserved timestamp raise. Prices at
    or after 2019-05-16 00:00 UTC are not read. No noise model is fit.
    """
    horizons = parse_horizons(horizons_hours)
    lookup = _dev_price_lookup(
        frame, source_timezone=source_timezone, price_column=price_column
    )
    if origins is None:
        requested = tuple(sorted(lookup))
        keep: list[pd.Timestamp] = []
        for origin in requested:
            targets = [origin + pd.Timedelta(hours=hours) for hours in horizons]
            if any(target >= RESERVED_START_UTC for target in targets):
                continue
            if any(target not in lookup for target in targets):
                continue
            if origin not in lookup:
                continue
            keep.append(origin)
        if not keep:
            raise CalibrationSetError("no development origin has full target support")
        return _residual_parameters(lookup, horizons, tuple(keep))

    normalized_origins = tuple(
        _normalize_one(origin, source_timezone=source_timezone) for origin in origins
    )
    if len(normalized_origins) != len(set(normalized_origins)):
        raise CalibrationSetError("duplicate timestamps")
    consumed_request: list[pd.Timestamp] = []
    for origin in normalized_origins:
        consumed_request.append(origin)
        for hours in horizons:
            target = origin + pd.Timedelta(hours=hours)
            consumed_request.append(target)
            if target >= RESERVED_START_UTC or origin >= RESERVED_START_UTC:
                calibration_set(
                    normalized_origins,
                    consumed=consumed_request,
                    source_timezone="UTC",
                )
            if origin not in lookup or target not in lookup:
                raise CalibrationSetError(
                    "requested origin does not have full development support"
                )
    return _residual_parameters(lookup, horizons, normalized_origins)


def fit_development_parameters(
    frame: pd.DataFrame,
    horizons_hours: Sequence[int],
    *,
    source_timezone: str,
    price_column: str = "CLOSE",
) -> DevelopmentParameters:
    """Fit DEV scales and residuals. Reserved rows are excluded before the read.

    The scale is the mean absolute residual of target close minus origin
    close, per horizon. That is not a noise model. Mutating a bar on or
    after the reserved cut cannot change the result.
    """
    return admit_elapsed_hour_calibration(
        frame,
        horizons_hours,
        source_timezone=source_timezone,
        origins=None,
        price_column=price_column,
    )


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
