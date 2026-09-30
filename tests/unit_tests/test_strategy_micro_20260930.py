"""Synthetic contract tests for elapsed-hour horizons and calibration support.

F2 and F3 expectations are fixed here. They do not call a broker, do not read
a market CSV, and do not fit a noise model.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app import strategy_support
from app.strategy_support import (
    CalibrationSetError,
    create_elapsed_hour_predictions,
    derive_development_support,
)


def _elapsed(frame, horizons, source_timezone="UTC"):
    try:
        return create_elapsed_hour_predictions(
            frame, horizons, source_timezone=source_timezone
        )
    except TypeError as exc:
        if "source_timezone" not in str(exc):
            raise
        return create_elapsed_hour_predictions(frame, horizons)


def _derive(frame, origins, **kwargs):
    kwargs.setdefault("population_label", "SYNTHETIC")
    try:
        return derive_development_support(
            frame, origins, source_timezone="UTC", **kwargs
        )
    except TypeError as exc:
        if "source_timezone" not in str(exc):
            raise
        return derive_development_support(frame, origins, **kwargs)


def _numbers(value):
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        yield float(value)
        return
    if isinstance(value, str):
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _numbers(item)


def _three_hour_frame():
    index = pd.to_datetime(
        ["2019-04-01 00:00", "2019-04-01 01:00", "2019-04-01 02:00"]
    )
    return pd.DataFrame({"CLOSE": [1.0, 2.0, 3.0]}, index=index)


def _support_frame():
    index = pd.to_datetime(["2019-05-09 20:00", "2019-05-09 21:00"])
    return pd.DataFrame(
        {"CLOSE": [1.0, 1.5], "OPEN": [1.0, 1.5], "HIGH": [1.1, 1.6], "LOW": [0.9, 1.4]},
        index=index,
    )


def test_f3_elapsed_rejects_bool_horizon():
    with pytest.raises(ValueError):
        _elapsed(_three_hour_frame(), [True])


def test_f3_elapsed_rejects_fractional_horizon():
    with pytest.raises(ValueError):
        _elapsed(_three_hour_frame(), [1.9])


def test_f3_elapsed_rejects_text_horizon():
    with pytest.raises(ValueError):
        _elapsed(_three_hour_frame(), ["1"])


def test_f3_elapsed_rejects_duplicate_horizons():
    with pytest.raises(ValueError):
        _elapsed(_three_hour_frame(), [1, 1])


def test_f3_elapsed_rejects_nonfinite_price():
    frame = _three_hour_frame()
    frame.iloc[1, 0] = float("nan")
    with pytest.raises(ValueError):
        _elapsed(frame, [1])


def test_f3_elapsed_rejects_infinite_price():
    frame = _three_hour_frame()
    frame.iloc[1, 0] = float("inf")
    with pytest.raises(ValueError):
        _elapsed(frame, [1])


def test_f3_derive_rejects_bool_horizon():
    frame = _support_frame()
    with pytest.raises(ValueError):
        _derive(
            frame,
            [frame.index[0]],
            required_horizons_hours=(True,),
            longest_horizon_hours=True,
        )


def test_f3_derive_rejects_fractional_horizon():
    frame = _support_frame()
    with pytest.raises(ValueError):
        _derive(
            frame,
            [frame.index[0]],
            required_horizons_hours=(1.9,),
            longest_horizon_hours=1.9,
        )


def test_f3_derive_rejects_text_horizon():
    frame = _support_frame()
    with pytest.raises(ValueError):
        _derive(
            frame,
            [frame.index[0]],
            required_horizons_hours=("1",),
            longest_horizon_hours="1",
        )


def test_f3_derive_rejects_duplicate_horizons():
    frame = _support_frame()
    with pytest.raises(ValueError):
        _derive(
            frame,
            [frame.index[0]],
            required_horizons_hours=(1, 1),
            longest_horizon_hours=1,
        )


def test_f3_derive_rejects_nonfinite_consumed_price():
    frame = _support_frame()
    frame.iloc[1, frame.columns.get_loc("CLOSE")] = float("nan")
    with pytest.raises(ValueError):
        _derive(
            frame,
            [frame.index[0]],
            required_horizons_hours=(1,),
            longest_horizon_hours=1,
        )


def test_f3_refuses_to_guess_source_timezone():
    with pytest.raises(ValueError):
        create_elapsed_hour_predictions(_three_hour_frame(), [1])


def test_f3_timestamps_must_be_in_temporal_order():
    index = pd.to_datetime(["2019-04-01 02:00", "2019-04-01 00:00"])
    frame = pd.DataFrame({"CLOSE": [1.0, 2.0]}, index=index)
    with pytest.raises(ValueError):
        _elapsed(frame, [2])


def test_f3_declared_zone_normalizes_to_utc():
    index = pd.to_datetime(["2019-05-15 22:00", "2019-05-15 23:00"])
    frame = pd.DataFrame({"CLOSE": [1.0, 2.0]}, index=index)
    try:
        result = create_elapsed_hour_predictions(
            frame, [1], source_timezone="Europe/Madrid"
        )
    except TypeError as exc:
        if "source_timezone" not in str(exc):
            raise
        pytest.fail("elapsed generator ignores a declared source timezone")
    assert result.index.tz is not None
    assert str(result.index.tz) == "UTC"
    assert result.index[0] == pd.Timestamp("2019-05-15 20:00", tz="UTC")
    assert float(result.iloc[0, 0]) == 2.0


def test_f3_market_gap_is_not_a_nonfinite_rejection():
    index = pd.to_datetime(
        ["2019-04-01 00:00", "2019-04-01 02:00", "2019-04-01 03:00"]
    )
    frame = pd.DataFrame({"CLOSE": [1.0, 2.0, 3.0]}, index=index)
    result = _elapsed(frame, [1])
    assert len(result) == 1
    assert float(result.iloc[0, 0]) == 3.0
    clock = pd.Timestamp(result.index[0])
    if clock.tzinfo is not None:
        clock = clock.tz_convert("UTC").tz_localize(None)
    assert clock == pd.Timestamp("2019-04-01 02:00")


def test_f2_admit_rejects_reserved_target():
    admit = getattr(strategy_support, "admit_elapsed_hour_calibration", None)
    if admit is None:
        pytest.fail(
            "F2: calibration admission sees origins only; "
            "2019-05-15 23:00 is accepted while +1h reads 2019-05-16 00:00"
        )
    crossing = pd.DataFrame(
        {"CLOSE": [1.0, 999.0]},
        index=pd.to_datetime(["2019-05-15 23:00", "2019-05-16 00:00"]),
    )
    with pytest.raises(CalibrationSetError):
        admit(
            crossing,
            [1],
            source_timezone="UTC",
            origins=(pd.Timestamp("2019-05-15 23:00"),),
        )


def test_f2_reserved_mutation_does_not_change_dev_parameters():
    fit = getattr(strategy_support, "fit_development_parameters", None)
    if fit is None:
        pytest.fail(
            "F2: no DEV fit inspects targets, scales, and residuals; "
            "a reserved price can still enter through origin-only admission"
        )
    index = pd.to_datetime(
        [
            "2019-05-15 21:00",
            "2019-05-15 22:00",
            "2019-05-15 23:00",
            "2019-05-16 00:00",
        ]
    )
    base = pd.DataFrame({"CLOSE": [1.0, 1.2, 1.4, 1.0]}, index=index)
    mutated = base.copy()
    mutated.iloc[-1, 0] = 999.0
    left = fit(base, [1], source_timezone="UTC")
    right = fit(mutated, [1], source_timezone="UTC")
    assert left.fits_noise_model is False
    assert right.fits_noise_model is False
    assert left.parameters == right.parameters
    assert all(abs(value) != 999.0 for value in _numbers(left.parameters))
    assert all(abs(value) != 999.0 for value in _numbers(right.parameters))
    admitted = tuple(pd.Timestamp(stamp) for stamp in left.admitted_origins)
    assert pd.Timestamp("2019-05-15 23:00", tz="UTC") not in admitted
    assert pd.Timestamp("2019-05-16 00:00", tz="UTC") not in admitted
    consumed = tuple(pd.Timestamp(stamp) for stamp in left.consumed_timestamps_utc)
    reserved = pd.Timestamp("2019-05-16 00:00", tz="UTC")
    assert consumed
    assert all(stamp < reserved for stamp in consumed)


# Decision oracles are literals. They are not calls to calculate_entry_geometry
# or should_early_close. Short family / long family.
OPEN_ORACLES = {
    "ideal/ideal": ("long", "short", "long"),
    "persistence/ideal": ("long", "short", "long"),
    "ideal/persistence": (),
    "persistence/persistence": (),
}
CLOSED_ORACLES = {
    "ideal/ideal": 3,
    "persistence/ideal": 3,
    "ideal/persistence": 0,
    "persistence/persistence": 0,
}
TRADE_CLOCKS = {
    "ideal/ideal": (
        ("2019-05-01T00:00:00+00:00", "2019-05-01T03:00:00+00:00", 3.0),
        ("2019-05-01T04:00:00+00:00", "2019-05-01T07:00:00+00:00", 3.0),
        ("2019-05-01T08:00:00+00:00", "2019-05-01T10:00:00+00:00", 2.0),
    ),
    "persistence/ideal": (
        ("2019-05-01T00:00:00+00:00", "2019-05-01T03:00:00+00:00", 3.0),
        ("2019-05-01T04:00:00+00:00", "2019-05-01T07:00:00+00:00", 3.0),
        ("2019-05-01T08:00:00+00:00", "2019-05-01T11:00:00+00:00", 3.0),
    ),
}
SUPPORT_ORIGINS = (
    "2019-05-01T00:00:00+00:00",
    "2019-05-01T04:00:00+00:00",
    "2019-05-01T08:00:00+00:00",
    "2019-05-01T09:00:00+00:00",
)
# request, direction, long-family TP, SL, short-family TP that must not be used
ENTRY_GEOMETRY_ORACLE = {
    "2019-05-01T00:00:00+00:00": ("buy", "long", 1.018, 0.9998, 1.027),
    "2019-05-01T04:00:00+00:00": ("sell", "short", 1.002, 1.0202, 0.993),
    "2019-05-01T08:00:00+00:00": ("buy", "long", 1.009, 0.9998, None),
}
IDEAL_EXIT_STAMP = "2019-05-01T09:00:00+00:00"
PERSIST_SL_STAMP = "2019-05-01T10:00:00+00:00"
TURN_STOP = 0.9998
IDEAL_SHORT_AT_EXIT = (0.99, 0.995, 1.0, 1.0, 1.0, 1.0)
IDEAL_LONG_MIN_AT_EXIT = 1.003
PERSIST_SHORT_AT_EXIT = 1.002
DECLARED_CAPITAL = 2_000_000.0
PLUGIN_CASH_OBSERVATION = 10_000.0
EXPECTED_COLUMNS = [f"Prediction_h_{i}" for i in range(1, 7)] + [
    f"Prediction_d_{i}" for i in range(1, 7)
]
SHORT_HORIZONS = (1, 2, 3, 4, 5, 6)
LONG_HORIZONS = (24, 48, 72, 96, 120, 144)


def _ohlc(stamp, open_, close):
    return (
        stamp,
        open_,
        max(open_, close) + 0.01,
        min(open_, close) - 0.01,
        close,
    )


def _frame(rows):
    return pd.DataFrame(
        {
            "OPEN": [row[1] for row in rows],
            "HIGH": [row[2] for row in rows],
            "LOW": [row[3] for row in rows],
            "CLOSE": [row[4] for row in rows],
        },
        index=pd.to_datetime([row[0] for row in rows]),
    )


def _micro_bars():
    """Rise, fall, then a turn. The 999 bar is the reserved cut."""
    rows = [
        _ohlc("2019-05-01 00:00", 1.000, 1.000),
        _ohlc("2019-05-01 01:00", 1.000, 1.005),
        _ohlc("2019-05-01 02:00", 1.005, 1.030),
        _ohlc("2019-05-01 03:00", 1.030, 1.030),
        _ohlc("2019-05-01 04:00", 1.020, 1.020),
        _ohlc("2019-05-01 05:00", 1.020, 1.015),
        _ohlc("2019-05-01 06:00", 1.015, 1.000),
        _ohlc("2019-05-01 07:00", 1.000, 1.002),
        _ohlc("2019-05-01 08:00", 1.000, 1.000),
        _ohlc("2019-05-01 09:00", 1.000, 1.002),
        _ohlc("2019-05-01 10:00", 1.001, 0.990),
        _ohlc("2019-05-01 11:00", 0.990, 0.995),
        _ohlc("2019-05-01 12:00", 1.000, 1.000),
        _ohlc("2019-05-01 13:00", 1.000, 1.000),
        _ohlc("2019-05-01 14:00", 1.000, 1.000),
        _ohlc("2019-05-01 15:00", 1.000, 1.000),
    ]
    long_closes = {
        "00:00": (1.010, 1.020, 1.012, 1.011, 1.011, 1.010),
        "04:00": (1.010, 1.000, 1.008, 1.009, 1.009, 1.010),
        "08:00": (1.005, 1.010, 1.004, 1.003, 1.003, 1.003),
        "09:00": (1.004, 1.004, 1.004, 1.003, 1.004, 1.004),
    }
    for offset in range(1, 7):
        for hour, closes in long_closes.items():
            rows.append(
                _ohlc(f"2019-05-{1 + offset:02d} {hour}", closes[offset - 1], closes[offset - 1])
            )
    rows.append(_ohlc("2019-05-16 00:00", 999.0, 999.0))
    return _frame(rows)


def _probe_bars():
    rows = [_ohlc("2019-05-01 00:00", 1.0, 1.0)]
    for hour in range(1, 7):
        rows.append(_ohlc(f"2019-05-01 {hour:02d}:00", 1.0, 1.001))
    for day in range(2, 8):
        rows.append(_ohlc(f"2019-05-{day:02d} 00:00", 1.02, 1.02))
    rows.append(_ohlc("2019-05-16 00:00", 999.0, 999.0))
    return _frame(rows)


def _arm_config(short_family, long_family, **extra):
    config = {
        "prediction_generator": "elapsed_hours",
        "offset_unit": "hours",
        "exit_variant": "E",
        "source_timezone": "UTC",
        "short_horizons_hours": SHORT_HORIZONS,
        "long_horizons_hours": LONG_HORIZONS,
        "short_family": short_family,
        "long_family": long_family,
        "initial_cash": DECLARED_CAPITAL,
        "price_column": "CLOSE",
    }
    config.update(extra)
    return config


def _event(result, stamp):
    found = [row for row in result["events"] if row["timestamp"] == stamp]
    assert len(found) == 1
    return found[0]


def _same_numbers(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert left[key] == pytest.approx(right[key])


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise AssertionError("non-finite SYNTHETIC number")
        return value
    raise AssertionError(f"unexpected measured type {type(value).__name__}")


def test_harness_rejects_a_generator_other_than_elapsed_hours():
    from app.elapsed_hour_harness import run_elapsed_hour_harness

    with pytest.raises(ValueError, match="elapsed_hours"):
        run_elapsed_hour_harness(
            _micro_bars(),
            {"prediction_generator": "rows", "exit_variant": "E"},
            work_directory=".",
        )


def test_synthetic_elapsed_hour_microexperiment(tmp_path):
    """Generator, plugin columns, decisions, fills, and a SYNTHETIC result."""
    import json
    from pathlib import Path

    from app.elapsed_hour_harness import _ObservedStrategy, run_elapsed_hour_harness
    from app.plugins.plugin_long_short_predictions import Plugin

    assert issubclass(_ObservedStrategy, Plugin.HeuristicStrategy)
    bars = _micro_bars()
    arms = {}
    for short_family, long_family in (
        ("ideal", "ideal"),
        ("persistence", "ideal"),
        ("ideal", "persistence"),
        ("persistence", "persistence"),
    ):
        pair = f"{short_family}/{long_family}"
        arms[pair] = run_elapsed_hour_harness(
            bars,
            _arm_config(short_family, long_family),
            work_directory=str(tmp_path / pair.replace("/", "_")),
        )

    ideal = arms["ideal/ideal"]
    mixed = arms["persistence/ideal"]
    assert ideal["prediction_generator"] == "elapsed_hours"
    assert ideal["offset_unit"] == "hours"
    assert ideal["legacy_offset_unit"] == "rows"
    assert ideal["exit_variant"] == "E"
    assert ideal["protective_orders"] is False
    assert ideal["fill_semantics"] == "close_only_decision_next_open_market_fill"
    assert ideal["label"] == "SYNTHETIC"
    assert ideal["plugin_columns"] == EXPECTED_COLUMNS
    assert ideal["support_origins"] == list(SUPPORT_ORIGINS)
    assert ideal["broker_first"] == "2019-05-01T00:00:00+00:00"
    assert ideal["broker_last"] == "2019-05-07T09:00:00+00:00"
    assert ideal["initial_cash"] == DECLARED_CAPITAL
    assert ideal["mae_aggregation"] == "macro_average_of_per_horizon_mae"

    for pair, result in arms.items():
        assert tuple(result["entry_sides"]) == OPEN_ORACLES[pair]
        assert result["n_closed_trades"] == CLOSED_ORACLES[pair]
        assert result["support_origins"] == list(SUPPORT_ORIGINS)
        assert result["plugin_columns"] == EXPECTED_COLUMNS
        assert result["exit_variant"] == "E"
        assert result["pending_position_count"] == 0
        assert result["pending_position_units"] == 0.0
        assert result["end_exposure_units"] == 0.0
        assert result["stop_close_obtained_fill"] is False
        assert result["marked_equity"] == pytest.approx(result["cash"])
        assert result["realized_pnl"] == pytest.approx(
            sum(trade["pnl"] for trade in result["trades"])
        )
        assert "pending_position_units" in result
        assert "marked_equity" in result
        assert "realized_pnl" in result
        _same_numbers(result["per_horizon_naive_mae"], ideal["per_horizon_naive_mae"])
        assert result["naive_macro_average"] == pytest.approx(ideal["naive_macro_average"])
        for key in (
            "spread_pips",
            "slippage_pips",
            "pip_cost",
            "commission_per_unit",
            "commission_per_lot",
            "swap_per_lot_per_day",
            "tp_multiplier",
            "sl_multiplier",
        ):
            assert result["costs"][key] == pytest.approx(ideal["costs"][key])
        assert result["costs"]["tp_multiplier"] == pytest.approx(0.9)
        assert result["costs"]["sl_multiplier"] == pytest.approx(2.0)
        assert result["costs"]["spread_pips"] == pytest.approx(2.0)
        assert result["costs"]["slippage_pips"] == pytest.approx(1.0)
        for event in result["events"]:
            assert event["open"] != pytest.approx(999.0)
            assert event["close"] != pytest.approx(999.0)
            assert event["timestamp"] < "2019-05-16T00:00:00+00:00"

    for value in ideal["per_horizon_mae"].values():
        assert value == pytest.approx(0.0)
    assert ideal["mae_macro_average"] == pytest.approx(0.0)
    flat = arms["persistence/persistence"]
    _same_numbers(flat["per_horizon_mae"], flat["per_horizon_naive_mae"])
    assert flat["mae_macro_average"] == pytest.approx(flat["naive_macro_average"])
    assert arms["ideal/persistence"]["n_closed_trades"] == 0
    assert arms["ideal/persistence"]["realized_pnl"] == pytest.approx(0.0)
    assert flat["realized_pnl"] == pytest.approx(0.0)

    for pair in ("ideal/ideal", "persistence/ideal"):
        result = arms[pair]
        assert len(result["trades"]) == 3
        assert len(result["fills"]) == 6
        assert result["fills"][0]["price"] == pytest.approx(1.000015)
        assert result["fills"][0]["is_buy"] is True
        for trade, (opened, closed, duration) in zip(result["trades"], TRADE_CLOCKS[pair]):
            assert trade["open_dt"] == opened
            assert trade["close_dt"] == closed
            assert trade["duration_bars"] == pytest.approx(duration)
            assert trade["volume"] == pytest.approx(1_000_000.0)
        for stamp, oracle in ENTRY_GEOMETRY_ORACLE.items():
            request, direction, take_profit, stop_loss, rejected_tp = oracle
            event = _event(result, stamp)
            assert event["request"] == request
            assert event["direction_after"] == direction
            assert event["position_before"] == pytest.approx(0.0)
            assert event["take_profit_after"] == pytest.approx(take_profit)
            assert event["stop_loss_after"] == pytest.approx(stop_loss)
            assert event["size"] == pytest.approx(1_000_000.0)
            if rejected_tp is not None:
                assert event["take_profit_after"] != pytest.approx(rejected_tp)
            assert len(event["predictions"]["short"]) == 6
            assert len(event["predictions"]["long"]) == 6

    ideal_exit = _event(ideal, IDEAL_EXIT_STAMP)
    assert ideal_exit["request"] == "close"
    assert ideal_exit["position_before"] == pytest.approx(1_000_000.0)
    assert ideal_exit["predictions"]["short"] == pytest.approx(IDEAL_SHORT_AT_EXIT)
    long_min = min(ideal_exit["predictions"]["long"])
    short_min = min(ideal_exit["predictions"]["short"])
    assert long_min == pytest.approx(IDEAL_LONG_MIN_AT_EXIT)
    blend = 0.6 * short_min + 0.4 * long_min
    assert blend < TURN_STOP
    assert not (long_min < TURN_STOP)

    persist_exit = _event(mixed, IDEAL_EXIT_STAMP)
    assert persist_exit["request"] == "none"
    assert persist_exit["position_before"] == pytest.approx(1_000_000.0)
    assert persist_exit["predictions"]["short"] == pytest.approx(
        [PERSIST_SHORT_AT_EXIT] * 6
    )
    persist_long_min = min(persist_exit["predictions"]["long"])
    persist_short = persist_exit["predictions"]["short"][0]
    persist_blend = 0.6 * persist_short + 0.4 * persist_long_min
    assert not (persist_blend < TURN_STOP)
    persist_stop = _event(mixed, PERSIST_SL_STAMP)
    assert persist_stop["request"] == "close"
    assert persist_stop["position_before"] == pytest.approx(1_000_000.0)
    assert _event(ideal, PERSIST_SL_STAMP)["request"] == "none"

    # Long family sets the entry. Short-family persistence leaves that entry.
    assert _event(ideal, "2019-05-01T00:00:00+00:00")["predictions"]["long"] == pytest.approx(
        (1.01, 1.02, 1.012, 1.011, 1.011, 1.01)
    )
    assert _event(mixed, "2019-05-01T00:00:00+00:00")["predictions"]["short"] == pytest.approx(
        [1.0] * 6
    )
    assert _event(arms["ideal/persistence"], "2019-05-01T00:00:00+00:00")["request"] == "none"

    cash_check = run_elapsed_hour_harness(
        bars,
        _arm_config("ideal", "ideal", initial_cash=PLUGIN_CASH_OBSERVATION),
        work_directory=str(tmp_path / "cash_10000"),
    )
    assert cash_check["initial_cash"] == PLUGIN_CASH_OBSERVATION
    assert cash_check["n_closed_trades"] == 1
    assert cash_check["trades"][0]["open_dt"] == "2019-05-01T04:00:00+00:00"
    assert _event(cash_check, "2019-05-01T00:00:00+00:00")["request"] == "buy"
    assert all(trade["open_dt"] != "2019-05-01T00:00:00+00:00" for trade in cash_check["trades"])
    assert cash_check["pending_position_units"] == pytest.approx(0.0)

    probe = run_elapsed_hour_harness(
        _probe_bars(),
        _arm_config(
            "ideal",
            "ideal",
            broker_session_end="2019-05-01 01:00",
        ),
        work_directory=str(tmp_path / "probe"),
    )
    assert tuple(probe["entry_sides"]) == ("long",)
    assert probe["n_closed_trades"] == 0
    assert probe["realized_pnl"] == pytest.approx(0.0)
    assert probe["pending_position_units"] == pytest.approx(1_000_000.0)
    assert probe["pending_position_count"] == 1
    assert probe["position_before_stop"] == pytest.approx(1_000_000.0)
    assert probe["position_after_stop"] == pytest.approx(1_000_000.0)
    assert probe["closed_trades_before_stop"] == 0
    assert probe["closed_trades_after_stop"] == 0
    assert probe["stop_close_applicable"] is True
    assert probe["stop_close_obtained_fill"] is False
    assert probe["broker_bar_count"] == 2
    assert probe["broker_last"] == "2019-05-01T01:00:00+00:00"
    assert probe["support_origins"] == ["2019-05-01T00:00:00+00:00"]
    assert probe["fills"][0]["price"] == pytest.approx(1.000015)
    assert probe["fills"][0]["size"] == pytest.approx(1_000_000.0)
    assert probe["marked_equity"] != pytest.approx(probe["realized_pnl"])
    assert probe["cash"] != pytest.approx(probe["marked_equity"])
    assert probe["terminal_convention"].startswith("pending positions stay open")

    document = {
        "label": "SYNTHETIC",
        "phase": "MEASURED",
        "experiment": "elapsed_hour_variant_E_micro",
        "b0_status": "NOT_STARTED",
        "sweep_not_run": True,
        "equal_mae_note": ideal["equal_mae_note"],
        "ideal_profit_is_not_required_to_be_maximal": True,
        "variant_e_trace": {
            "exit_uses_both_families": True,
            "long_family_sets_entry_take_profit_and_stop": True,
            "short_family_can_change_the_exit": True,
            "ideal_exit_request": "close",
            "persistence_short_exit_request_at_same_bar": "none",
            "ideal_blend_0_6_short_plus_0_4_long": blend,
            "long_min_alone_is_below_stop": False,
            "persistence_short_blend": persist_blend,
            "persistence_short_blend_is_below_stop": False,
            "turn_stop": TURN_STOP,
        },
        "declared_initial_cash": DECLARED_CAPITAL,
        "capital_reason": (
            "SYNTHETIC. The real plugin leaves margin unset, so the simulated "
            "broker requires full notional. Size is min(1000000, cash * 2). "
            "At cash 10000 a requested long of size 20000 does not fill. "
            "Declared cash 2000000 makes that same sizer hit the 1000000 cap, "
            "which the full-notional check accepts. TP, SL, spread, slippage, "
            "commission rate, and swap are unchanged. Leverage is not passed "
            "to the broker."
        ),
        "arms": arms,
        "broker_cash_check": {
            "label": "SYNTHETIC",
            "role": "observation_not_a_fifth_contrast_arm",
            "initial_cash": cash_check["initial_cash"],
            "n_closed_trades": cash_check["n_closed_trades"],
            "entry_sides": cash_check["entry_sides"],
            "trades": cash_check["trades"],
            "realized_pnl": cash_check["realized_pnl"],
            "pending_position_units": cash_check["pending_position_units"],
            "marked_equity": cash_check["marked_equity"],
            "cash": cash_check["cash"],
            "long_requested_at_open_and_not_closed": True,
        },
        "terminal_probe": probe,
        "b0_successor": {
            "status": "NOT_STARTED",
            "not_run": True,
            "historic_remainder_is_not_a_budget": True,
            "terminal_convention": (
                "pending positions stay open; no forced liquidation"
            ),
            "forced_liquidation_applied_to_microexperiment": False,
            "cost_basis": {
                "spread_pips": ideal["costs"]["spread_pips"],
                "slippage_pips": ideal["costs"]["slippage_pips"],
                "pip_cost": ideal["costs"]["pip_cost"],
                "commission_per_unit": ideal["costs"]["commission_per_unit"],
                "commission_note": (
                    "The plugin passes commission_per_lot / 100000 as a "
                    "percentage of price. It is not rewritten as a literal "
                    "7 currency-unit fee."
                ),
                "swap_per_lot_per_day": ideal["costs"]["swap_per_lot_per_day"],
                "tp_multiplier": ideal["costs"]["tp_multiplier"],
                "sl_multiplier": ideal["costs"]["sl_multiplier"],
                "exit_variant": "E",
                "fill_semantics": ideal["fill_semantics"],
                "protective_orders": False,
            },
        },
    }
    payload = _jsonable(document)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    lowered = encoded.lower()
    for token in ("/home", "harvey", "hostname", "postgres"):
        assert token not in lowered
    evidence = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "audits"
        / "evidence"
        / "STRATEGY_MICRO_20260930"
        / "MICRO.json"
    )
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(encoded + "\n", encoding="utf-8")
