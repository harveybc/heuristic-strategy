"""S07 tests: purge derivation, reserved-rows never-read guard, variant-E resolution, dev-only calibration.

Every fixture is synthetic and constructed here. No committed market file is read by these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.config import DEFAULT_VALUES
from app.plugins.plugin_long_short_predictions import Plugin
from app.policies.prediction_entry_exit import (
    PredictionEntryExitParameters,
    calculate_entry_geometry,
    should_early_close,
)
from app.strategy_support import baseline_config
from app.target_support import (
    EXIT_VARIANTS,
    NO_ENTRY,
    REASON_CENSORED,
    REASON_EARLY,
    REASON_TP,
    RESERVED_START,
    ReservedRowsAccessError,
    derive_arm_support,
    derive_population_target_support,
    development_residual_statistics,
    load_development_bars,
    load_prediction_family,
    unit_trade_exit_rows,
)

ROOT = Path(__file__).resolve().parents[2]
CUT = RESERVED_START


def _hourly_index(start: str, hours: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), periods=hours, freq="h", name="DATE_TIME")


def _write_bars(path: Path, index: pd.DatetimeIndex, close: np.ndarray) -> None:
    frame = pd.DataFrame(
        {"OPEN": close, "LOW": close - 0.0005, "HIGH": close + 0.0005, "CLOSE": close}, index=index
    )
    frame.reset_index().to_csv(path, index=False)


def _write_preds(path: Path, index: pd.DatetimeIndex, values: np.ndarray) -> None:
    cols = [f"Prediction_{i}" for i in range(1, values.shape[1] + 1)]
    pd.DataFrame(values, index=index, columns=cols).reset_index().to_csv(path, index=False)


def _fixture(tmp_path: Path, *, poison_reserved: bool = False, extra_reserved_rows: int = 0):
    """Ten development days of flat 1.1000 bars ending 2019-05-15 23:00, then reserved bars.

    Origin K (2019-05-06 00:00): long path forecasts +30 pips at every long horizon, and the
    close reaches TP two bars after the fill -> exits well before the cut, 144 h target on
    2019-05-12 00:00 (before the cut) -> KEPT.
    Origin X (2019-05-08 00:00): a SHORT with a distant stop (long path max +30 pips,
    min -100 pips) on prices that never reach TP/SL and no early exit fires -> still open at
    the last development bar -> CENSORED -> PURGED, although its 144 h target (2019-05-14
    00:00) is before the cut.
    Origin T (2019-05-10 01:00): long, exits by TP at bar T+3, but its 144 h target is
    2019-05-16 01:00 -> PURGED by elapsed target even though the trade exits.
    """
    dev = _hourly_index("2019-05-06 00:00", 10 * 24)  # ends 2019-05-15 23:00
    reserved = _hourly_index("2019-05-16 00:00", 48 + extra_reserved_rows)
    index = dev.append(reserved)
    close = np.full(len(index), 1.1000)
    k = 0  # origin K row
    close[k + 3] = 1.1000 + 0.00030  # TP for K is c + 0.9*30pips = +27 pips; bar k+3 close = +30 -> TP signal at k+3
    close[97 + 3] = 1.1000 + 0.00030  # TP for T (long, +30 pips) at bar 100; below X's short stop 1.1006
    if poison_reserved:
        close[len(dev):] = np.nan
        close[len(dev)] = 9.9
    _write_bars(tmp_path / "bars.csv", index, close)
    # Predictions on every development bar: flat (no entry) except the three origins.
    short = np.full((len(dev), 6), 1.1000)
    long = np.full((len(dev), 6), 1.1000)
    for row in (k, 97):  # K=2019-05-06 00:00, T=2019-05-10 01:00: long, +30 pips >= profit_threshold 5
        long[row, :] = 1.1000 + 0.00030
    long[48, :] = [1.1003, 1.0900, 1.0900, 1.0900, 1.0900, 1.0900]  # X: rr_sell 3.33 > rr_buy 0.3 -> short, SL 1.1006
    _write_preds(tmp_path / "short.csv", dev, short)
    _write_preds(tmp_path / "long.csv", dev, long)
    return tmp_path


def _derive(tmp_path: Path):
    bars = load_development_bars(tmp_path / "bars.csv", reserved_start=CUT)
    short = load_prediction_family(tmp_path / "short.csv", family="short", reserved_start=CUT)
    long = load_prediction_family(tmp_path / "long.csv", family="long", reserved_start=CUT)
    return bars, derive_arm_support(label="SYN", bars=bars, short=short, long=long)


def test_purge_keeps_origin_whose_trade_and_targets_end_before_the_cut(tmp_path: Path) -> None:
    _fixture(tmp_path)
    bars, arm = _derive(tmp_path)
    row = arm.table.set_index("origin").loc[pd.Timestamp("2019-05-06 00:00")]
    assert row["entry_direction"] == "long"
    assert row["elapsed_latest_target"] == pd.Timestamp("2019-05-12 00:00")
    assert not row["elapsed_target_crosses_cut"]
    assert not row["legacy_row_offset_target_crosses_cut"]
    for v in EXIT_VARIANTS:
        assert row[f"exit_{v}_reason"] == REASON_TP
        assert row[f"exit_{v}_fill_ts"] == pd.Timestamp("2019-05-06 04:00")  # signal at bar 3, fill at bar 4
        assert row[f"exit_{v}_plugin_duration_bars"] == 4
        assert not row[f"exit_{v}_support_crosses_cut"]
    assert not row["purged"]
    assert row["purge_reasons"] == ""


def test_purge_removes_origin_whose_trade_is_still_open_at_the_cut(tmp_path: Path) -> None:
    _fixture(tmp_path)
    bars, arm = _derive(tmp_path)
    row = arm.table.set_index("origin").loc[pd.Timestamp("2019-05-08 00:00")]
    assert row["entry_direction"] == "short"
    assert not row["elapsed_target_crosses_cut"]  # 144 h target 2019-05-14 00:00 is before the cut
    for v in EXIT_VARIANTS:
        assert row[f"exit_{v}_reason"] == REASON_CENSORED
        assert row[f"exit_{v}_fill_ts"] is None or pd.isna(row[f"exit_{v}_fill_ts"])
        assert row[f"exit_{v}_support_crosses_cut"]
        assert row[f"exit_{v}_censored_duration_bars_lower_bound"] == bars.n_dev - 48
    assert row["purged"]
    assert row["purge_reasons"] == "TRADE_SUPPORT"
    assert arm.summary["n_purged_trade_support_only"] >= 1


def test_purge_removes_origin_whose_elapsed_target_crosses_even_if_trade_exits(tmp_path: Path) -> None:
    _fixture(tmp_path)
    bars, arm = _derive(tmp_path)
    row = arm.table.set_index("origin").loc[pd.Timestamp("2019-05-10 01:00")]
    assert row["entry_direction"] == "long"
    assert row["exit_E_reason"] == REASON_TP and row["exit_E_fill_ts"] == pd.Timestamp("2019-05-10 05:00")
    assert not row["trade_support_crosses_cut_any_variant"]
    assert row["elapsed_latest_target"] == pd.Timestamp("2019-05-16 01:00")
    assert row["elapsed_target_crosses_cut"]
    assert row["purged"]
    assert "ELAPSED_TARGET" in row["purge_reasons"]
    # Legacy row offset: row 97 + 144 = 241 >= 240 development rows -> also crosses.
    assert row["legacy_row_offset_target_crosses_cut"]
    assert "LEGACY_ROW_OFFSET_TARGET" in row["purge_reasons"]


def test_no_entry_origins_are_purged_only_by_target_support(tmp_path: Path) -> None:
    _fixture(tmp_path)
    bars, arm = _derive(tmp_path)
    t = arm.table
    # Entries: K, X, T plus two SHORTS at the spike bars (rows 3 and 100), because the plugin
    # shorts when the close sits 30 pips above a flat long forecast (profit_sell >= threshold).
    entries = set(t.loc[t["entry_direction"] != NO_ENTRY, "origin"])
    assert entries == {pd.Timestamp(x) for x in ("2019-05-06 00:00", "2019-05-06 03:00", "2019-05-08 00:00",
                                                  "2019-05-10 01:00", "2019-05-10 04:00")}
    flat = t[t["entry_direction"] == NO_ENTRY]
    assert len(flat) == len(t) - 5
    assert (flat["trade_support_crosses_cut_any_variant"] == False).all()  # noqa: E712
    # Every origin from 2019-05-10 00:00 onward has origin + 144 h >= cut.
    late = flat[flat["origin"] >= pd.Timestamp("2019-05-10 00:00")]
    assert late["elapsed_target_crosses_cut"].all() and late["purged"].all()
    early = flat[flat["origin"] < pd.Timestamp("2019-05-10 00:00")]
    assert not early["elapsed_target_crosses_cut"].any()


def test_reserved_rows_are_never_read_poison_invariance(tmp_path: Path) -> None:
    """Replacing every reserved price with NaN/9.9 and adding reserved rows changes nothing."""
    (tmp_path / "clean").mkdir()
    (tmp_path / "poison").mkdir()
    clean = _fixture(tmp_path / "clean")
    poisoned = _fixture(tmp_path / "poison", poison_reserved=True, extra_reserved_rows=500)
    bars_c, arm_c = _derive(clean)
    bars_p, arm_p = _derive(poisoned)
    assert bars_c.n_reserved_rows_dropped == 48
    assert bars_p.n_reserved_rows_dropped == 548
    assert bars_c.n_dev == bars_p.n_dev == 240
    pd.testing.assert_frame_equal(arm_c.table, arm_p.table)
    sc = {k: v for k, v in arm_c.summary.items()}
    sp = {k: v for k, v in arm_p.summary.items()}
    assert sc == sp
    assert arm_c.files["base"]["sha256"] != arm_p.files["base"]["sha256"]  # different bytes, same derivation


def test_reserved_lookup_raises_and_bars_hold_no_reserved_price(tmp_path: Path) -> None:
    _fixture(tmp_path)
    bars = load_development_bars(tmp_path / "bars.csv", reserved_start=CUT)
    assert bars.index.max() == pd.Timestamp("2019-05-15 23:00")
    assert (bars.index < CUT).all()
    with pytest.raises(ReservedRowsAccessError):
        bars.row_of(CUT)
    with pytest.raises(ReservedRowsAccessError):
        bars.row_of("2019-06-01 00:00")
    assert bars.row_of("2019-05-15 23:00") == 239


def test_unit_trade_early_exit_masks_match_should_early_close(tmp_path: Path) -> None:
    """Vectorised variant masks agree with the plugin's scalar rule on a bar with predictions."""
    _fixture(tmp_path)
    bars = load_development_bars(tmp_path / "bars.csv", reserved_start=CUT)
    # Build a short family that trips the stop at bar 2 for variants that read the short family.
    dev = bars.index
    short = np.full((len(dev), 6), 1.1000)
    long = np.full((len(dev), 6), 1.1000)
    long[0, :] = 1.1000 + 0.00030  # long entry at row 0: SL = c - 2*max(drawdown,10pips)*pip = c - 20 pips
    short[2, :] = 1.1000 - 0.00050  # short min far below SL at bar 2
    _write_preds(tmp_path / "short2.csv", dev, short)
    _write_preds(tmp_path / "long2.csv", dev, long)
    sf = load_prediction_family(tmp_path / "short2.csv", family="short", reserved_start=CUT)
    lf = load_prediction_family(tmp_path / "long2.csv", family="long", reserved_start=CUT)
    arm = derive_arm_support(label="SYN2", bars=bars, short=sf, long=lf)
    row = arm.table.set_index("origin").loc[pd.Timestamp("2019-05-06 00:00")]
    params = PredictionEntryExitParameters()
    geometry = calculate_entry_geometry(current_price=1.1, long_horizon_predictions=list(long[0]), params=params)
    for v in EXIT_VARIANTS:
        scalar = should_early_close(
            direction="long", variant=v, short_horizon_predictions=list(short[2]),
            long_horizon_predictions=list(long[2]), stop_loss_price=geometry.stop_loss_price,
            entry_price=1.1 + 1.5e-5,  # next-open fill plus the configured half spread+slippage
        )
        if scalar:
            assert row[f"exit_{v}_reason"] == REASON_EARLY, v
            assert row[f"exit_{v}_signal_ts"] == pd.Timestamp("2019-05-06 02:00"), v
        else:
            assert row[f"exit_{v}_reason"] != REASON_EARLY, v
    assert row["exit_G_reason"] != REASON_EARLY
    assert row["exit_B_reason"] != REASON_EARLY  # B reads the long family only, which is flat
    assert row["exit_C_reason"] == REASON_EARLY


def test_population_target_support_counts_reserved_origins_already_present(tmp_path: Path) -> None:
    _fixture(tmp_path)
    bars = load_development_bars(tmp_path / "bars.csv", reserved_start=CUT)
    origins = list(pd.date_range("2019-05-06 00:00", "2019-05-20 00:00", freq="6h"))
    table, summary = derive_population_target_support(origins, bars, label="SYNPOP")
    assert summary["n_origins"] == len(origins)
    assert summary["n_origins_reserved_already_in_population"] == sum(1 for o in origins if o >= CUT)
    kept = table[~table["purged"]]
    assert kept["origin"].max() < pd.Timestamp("2019-05-10 00:00")
    assert (table["trade_support"] == "NOT_DERIVED_PER_CELL_PREDICTIONS_REQUIRED").all()


def test_variant_e_resolution_is_explicit_and_consistent_in_config_file() -> None:
    cfg = json.loads((ROOT / "config_replication_baseline_20260930.json").read_text())
    report = baseline_config()
    assert cfg["exit_variant"] == report["exit_variant"] == "E"
    assert cfg["historical_run_recovered"] is False and report["historical_run_recovered"] is False
    assert cfg["exit_variant_contradiction"]["fixed_reading"] == "reading_1_plugin_params"
    readings = cfg["exit_variant_contradiction"]
    assert readings["reading_1_plugin_params"]["value_before"] == "E"
    assert readings["reading_2_signature_default"]["value_before"] == "D"
    assert "D" in readings["reading_3_comment_block"]["value_before"]
    assert cfg["sweep_241_exit_variant"] == DEFAULT_VALUES["sweep_241_exit_variant"] == "E"
    assert cfg["sweep_241_manifest_sha256"] == DEFAULT_VALUES["sweep_241_manifest_sha256"]
    assert cfg["use_protective_broker_orders"] is False
    assert DEFAULT_VALUES["use_protective_broker_orders"] is False
    assert cfg["protective_orders_experiment"] == DEFAULT_VALUES["protective_orders_experiment"]
    assert cfg["fill_semantics"] == report["fill_semantics"]
    assert cfg["executed"] is False
    # The three code readings now agree with the fixed one.
    assert Plugin.plugin_params["exit_variant"] == "E"
    assert report["signature_default_exit_variant"] == "E"
    src = open(ROOT / "app/plugins/plugin_long_short_predictions.py").read()
    assert "D = both must agree (DEFAULT" not in src


def test_variant_e_rule_is_the_weighted_minimum_with_empty_family_fallback() -> None:
    sl = 1.0
    assert should_early_close(direction="long", variant="E", short_horizon_predictions=[0.99],
                              long_horizon_predictions=[1.02], stop_loss_price=sl, entry_price=1.01) is False  # 0.6*0.99+0.4*1.02=1.002, not < 1.0
    assert should_early_close(direction="long", variant="E", short_horizon_predictions=[0.98],
                              long_horizon_predictions=[1.02], stop_loss_price=sl, entry_price=1.01) is True
    assert should_early_close(direction="long", variant="E", short_horizon_predictions=[0.999],
                              long_horizon_predictions=[1.02], stop_loss_price=sl, entry_price=1.01) is False
    # empty short family -> either trigger
    assert should_early_close(direction="long", variant="E", short_horizon_predictions=[],
                              long_horizon_predictions=[0.99], stop_loss_price=sl, entry_price=1.01) is True


def test_calibration_uses_development_rows_only(tmp_path: Path) -> None:
    """Residual scale/correlation are identical when reserved prices are poisoned, and
    origins whose targets reach the cut are excluded and counted."""
    (tmp_path / "clean").mkdir()
    (tmp_path / "poison").mkdir()
    clean = _fixture(tmp_path / "clean")
    poisoned = _fixture(tmp_path / "poison", poison_reserved=True, extra_reserved_rows=100)
    horizons = (1, 2, 3, 4, 5, 6)
    stats = []
    for d in (clean, poisoned):
        bars = load_development_bars(d / "bars.csv", reserved_start=CUT)
        preds = load_prediction_family(d / "short.csv", family="short", reserved_start=CUT)
        stats.append(development_residual_statistics(preds, bars, horizons))
    assert json.dumps(stats[0], sort_keys=True) == json.dumps(stats[1], sort_keys=True)
    assert stats[0]["n_excluded_target_crosses_cut"] == 6  # the last six dev origins reach the cut at h<=6
    assert stats[0]["n_used"] == 240 - 6
    assert stats[0]["rows_used"].startswith("development only")
