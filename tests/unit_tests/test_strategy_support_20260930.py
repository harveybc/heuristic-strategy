from __future__ import annotations

import inspect

import pandas as pd
import pytest

from app.config import DEFAULT_VALUES
from app.data_processor import (
    LEGACY_OFFSET_UNIT,
    create_daily_predictions,
    create_hourly_predictions,
)
from app.plugins.plugin_long_short_predictions import Plugin
from app.strategy_support import (
    B0_STATUS,
    CENSORING_REPORTED,
    DESIGN_RESERVED_START,
    ELAPSED_OFFSET_UNIT,
    NOT_SEPARATED_BY_ORIGIN_CUT,
    REQUIRED_HORIZONS_HOURS,
    RESOLUTION_EXPLICIT,
    TRADE_DURATION_UNBOUNDED,
    CalibrationSetError,
    DevelopmentOriginError,
    baseline_config,
    calibration_set,
    cpu_reconciliation,
    create_elapsed_hour_predictions,
    derive_development_support,
    plugin_trade_duration,
    synthetic_elapsed_support_frame,
)


def test_baseline_config_is_variant_e_and_not_a_recovered_run() -> None:
    report = baseline_config()
    assert DEFAULT_VALUES["exit_variant"] == "E"
    assert DEFAULT_VALUES["historical_run_recovered"] is False
    assert DEFAULT_VALUES["exit_variant_resolution"] == RESOLUTION_EXPLICIT
    assert DEFAULT_VALUES["sweep_241_exit_variant"] == "NOT_CHECKED"
    assert report["exit_variant"] == "E"
    assert report["plugin_params_exit_variant"] == "E"
    assert report["signature_default_exit_variant"] == "E"
    assert report["historical_run_recovered"] is False
    assert report["resolution"] == RESOLUTION_EXPLICIT
    assert report["sweep_241_exit_variant"] == "NOT_CHECKED"
    assert "Not a recovered historical run" in report["note"]
    assert report["fill_semantics"] == "close_only_decision_next_open_market_fill"
    assert report["protective_broker_orders"] == "separate_named_experiment_not_in_baseline"
    assert Plugin.plugin_params["exit_variant"] == "E"
    signature = inspect.signature(Plugin.HeuristicStrategy.__init__)
    assert signature.parameters["exit_variant"].default == "E"


def test_plugin_keeps_close_only_market_exits_and_has_no_holding_cap() -> None:
    source = inspect.getsource(Plugin.HeuristicStrategy)
    assert "buy_bracket" not in source
    assert "sell_bracket" not in source
    assert "max_holding" not in source
    assert "max_bars" not in source
    assert plugin_trade_duration() == TRADE_DURATION_UNBOUNDED
    assert plugin_trade_duration(Plugin.plugin_params.keys()) == TRADE_DURATION_UNBOUNDED
    # The five-day counter limits entries. It is not a holding bound.
    assert plugin_trade_duration(["max_trades_per_5days", "exit_variant"]) == (
        TRADE_DURATION_UNBOUNDED
    )


def test_legacy_generators_are_row_offsets_and_still_exist() -> None:
    assert LEGACY_OFFSET_UNIT == "rows"
    assert create_hourly_predictions.offset_unit == "rows"
    assert create_daily_predictions.offset_unit == "rows"
    index = pd.to_datetime(
        ["2019-04-01 00:00", "2019-04-01 03:00", "2019-04-01 04:00"]
    )
    frame = pd.DataFrame({"CLOSE": [10.0, 30.0, 40.0]}, index=index)
    legacy = create_hourly_predictions(frame, 1)
    assert legacy.iloc[0, 0] == 30.0
    elapsed_one = create_elapsed_hour_predictions(frame, [1])
    assert list(elapsed_one.index) == [pd.Timestamp("2019-04-01 03:00")]
    assert pd.Timestamp("2019-04-01 00:00") not in elapsed_one.index
    assert elapsed_one.iloc[0, 0] == 40.0
    elapsed_three = create_elapsed_hour_predictions(frame, [3])
    assert list(elapsed_three.index) == [pd.Timestamp("2019-04-01 00:00")]
    assert elapsed_three.iloc[0, 0] == 30.0
    assert elapsed_three.attrs["offset_unit"] == "hours"


def test_elapsed_144_hours_is_not_144_rows() -> None:
    index = pd.to_datetime(
        ["2019-04-01 00:00", "2019-04-01 01:00", "2019-04-07 00:00"]
    )
    frame = pd.DataFrame(
        {
            "OPEN": [9.0, 8.0, 7.0],
            "LOW": [0.5, 0.6, 0.7],
            "HIGH": [9.5, 8.5, 7.5],
            "CLOSE": [1.0, 1.5, 4.0],
        },
        index=index,
    )
    elapsed = create_elapsed_hour_predictions(frame, [144])
    assert ELAPSED_OFFSET_UNIT == "hours"
    assert list(elapsed.columns) == ["elapsed_144h"]
    assert list(elapsed.index) == [pd.Timestamp("2019-04-01 00:00")]
    assert elapsed.iloc[0, 0] == 4.0
    assert elapsed.iloc[0, 0] != frame.iloc[0]["OPEN"]
    assert len(frame) < 144
    assert create_hourly_predictions(frame[["CLOSE"]], 144).empty
    assert create_daily_predictions(frame[["CLOSE"]], 6).empty
    both = create_elapsed_hour_predictions(frame, [1, 144])
    assert list(both.index) == [pd.Timestamp("2019-04-01 00:00")]
    assert both.iloc[0]["elapsed_1h"] == 1.5
    assert both.iloc[0]["elapsed_144h"] == 4.0
    assert pd.Timestamp("2019-04-01 01:00") not in both.index


def test_elapsed_24_hours_is_not_24_rows() -> None:
    index = pd.to_datetime(["2019-04-01 00:00", "2019-04-02 00:00"])
    frame = pd.DataFrame({"CLOSE": [1.0, 2.0]}, index=index)
    assert create_daily_predictions(frame, 1).empty
    elapsed = create_elapsed_hour_predictions(frame, [24])
    assert elapsed.iloc[0, 0] == 2.0
    assert elapsed.attrs["offset_unit"] == ELAPSED_OFFSET_UNIT


def test_synthetic_support_purges_targets_on_or_after_the_cut() -> None:
    frame, origins = synthetic_elapsed_support_frame()
    assert len(frame) == 9
    assert len(origins) == 5
    population = derive_development_support(
        frame,
        origins,
        required_horizons_hours=(1, 144),
        longest_horizon_hours=144,
        population_label="SYNTHETIC",
    )
    assert population.population_label == "SYNTHETIC"
    assert population.reserved_start == DESIGN_RESERVED_START
    assert population.reserved_window_rewritten is False
    assert population.trade_duration == TRADE_DURATION_UNBOUNDED
    assert population.censoring == CENSORING_REPORTED
    assert population.n_development_origins == 5
    assert population.n_purged == 2
    assert population.n_exact_latest_target == 4
    assert population.n_missing_exact_latest_target == 1
    assert population.n_target_timestamp_before_cut == 3
    assert population.n_entire_target_support_before_cut == 2
    assert population.n_not_separated_by_origin_cut == 5
    assert population.alternative_latest_development_origin == pd.Timestamp(
        "2019-05-09 23:00:00"
    )
    by_origin = {row.origin: row for row in population.origins}
    assert by_origin[pd.Timestamp("2019-05-10 00:00:00")].purged is True
    assert by_origin[pd.Timestamp("2019-05-10 00:00:00")].latest_target_timestamp == (
        DESIGN_RESERVED_START
    )
    late = by_origin[pd.Timestamp("2019-05-15 12:00:00")]
    assert late.purged is True
    assert late.exact_target_bar_present is False
    assert late.latest_target_timestamp == pd.Timestamp("2019-05-21 12:00:00")
    kept = by_origin[pd.Timestamp("2019-05-09 23:00:00")]
    assert kept.purged is False
    assert kept.separation == NOT_SEPARATED_BY_ORIGIN_CUT
    assert kept.trade_exit_support == TRADE_DURATION_UNBOUNDED
    assert kept.cost_support == TRADE_DURATION_UNBOUNDED
    assert kept.censoring == CENSORING_REPORTED

    full = derive_development_support(
        frame,
        origins,
        population_label="SYNTHETIC",
    )
    assert full.required_horizons_hours == REQUIRED_HORIZONS_HOURS
    assert full.longest_horizon_hours == 144
    assert full.n_purged == 2
    assert full.n_entire_target_support_before_cut == 0
    assert full.alternative_latest_development_origin is None
    assert full.n_not_separated_by_origin_cut == 5
    assert full.reserved_start == pd.Timestamp("2019-05-16 00:00:00")
    assert full.reserved_window_rewritten is False


def test_support_refuses_a_rewritten_reserved_window_and_reserved_origins() -> None:
    frame, origins = synthetic_elapsed_support_frame()
    with pytest.raises(ValueError, match="reserved window"):
        derive_development_support(
            frame,
            origins,
            reserved_start=pd.Timestamp("2019-05-09 00:00:00"),
            required_horizons_hours=(1, 144),
            longest_horizon_hours=144,
        )
    with pytest.raises(DevelopmentOriginError):
        derive_development_support(frame, [DESIGN_RESERVED_START])


def test_calibration_set_rejects_reserved_timestamps_and_does_not_fit_noise() -> None:
    frame, origins = synthetic_elapsed_support_frame()
    accepted = calibration_set(origins)
    assert accepted.max() < DESIGN_RESERVED_START
    assert calibration_set.fits_noise_model is False
    with pytest.raises(CalibrationSetError):
        calibration_set(frame.index)
    with pytest.raises(CalibrationSetError):
        calibration_set([DESIGN_RESERVED_START])
    with pytest.raises(CalibrationSetError):
        calibration_set(["2019-05-15 23:00:00", "2019-05-16 01:00:00"])
    source = inspect.getsource(calibration_set)
    assert "standard_normal" not in source
    assert "default_rng" not in source
    module_source = inspect.getsource(inspect.getmodule(calibration_set))
    assert "standard_normal" not in module_source
    assert "cholesky" not in module_source


def test_cpu_reconciliation_does_not_start_b0() -> None:
    report = cpu_reconciliation()
    assert report["subtraction"] == "3600 - 1461.693 = 2138.307"
    assert str(report["remainder_cpu_seconds"]) == "2138.307"
    assert report["ceiling_is_fresh_allowance"] is False
    assert report["remainder_is_permission"] is False
    assert report["retained_sweep_cells"] == 241
    assert report["status"] == B0_STATUS
    assert report["status"] == "B0_NOT_STARTED"
    assert not hasattr(cpu_reconciliation, "run_b0")
