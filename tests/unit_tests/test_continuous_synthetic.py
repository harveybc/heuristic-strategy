"""Continuous synthetic sensitivity, with all run artifacts under tmp_path."""

import pandas as pd
import pytest

from app.continuous_synthetic import build_trajectory, prepare_continuous, run_pilot
from app.sweep_executor import HORIZONS, synthetic_dev_frame


def test_every_eligible_hour_has_all_forecasts_and_separate_dev():
    bars = build_trajectory()
    prepared = prepare_continuous(bars, synthetic_dev_frame(), "ideal", "ideal")
    origins = prepared["elapsed_frame"].index
    assert len(origins) == len(bars) - max(HORIZONS)
    assert all(origins[i + 1] - origins[i] == pd.Timedelta(hours=1) for i in range(len(origins) - 1))
    assert prepared["elapsed_frame"].notna().all().all()
    assert set(prepared["per_horizon_mae"]) == {str(h) for h in HORIZONS}
    assert all(value == pytest.approx(0.0) for value in prepared["per_horizon_mae"].values())
    assert prepared["dev_last_utc"] < origins[0].isoformat()
    assert prepared["per_horizon_naive_mae"]["1"] > 0


def test_dev_overlap_and_incomplete_trajectory_are_rejected():
    bars = build_trajectory()
    with pytest.raises(ValueError, match="overlaps"):
        prepare_continuous(bars, bars[["CLOSE"]], "ideal", "ideal")
    with pytest.raises(ValueError, match="no complete"):
        prepare_continuous(bars.iloc[:100], synthetic_dev_frame(), "ideal", "ideal")
    reserved = bars.copy()
    reserved.loc[pd.Timestamp("2019-05-16"), :] = reserved.iloc[-1]
    with pytest.raises(ValueError, match="reserved cut"):
        prepare_continuous(reserved, synthetic_dev_frame(), "ideal", "ideal")


def test_paired_pilot_records_actual_causes_fills_and_exposure(tmp_path):
    report = run_pilot(str(tmp_path))
    assert report["label"] == "SYNTHETIC"
    assert report["b0_status"] == "B0_NOT_STARTED"
    assert report["financial_utility"] == "NOT_CLAIMED"
    assert len(report["cells"]) == 3
    assert {cell["name"] for cell in report["cells"]} == {
        "ideal_ideal", "persistence_persistence", "short_noise_long_ideal"
    }
    for cell in report["cells"]:
        assert len(cell["events"]) == len(build_trajectory())
        assert cell["accounting_equity"] == pytest.approx(cell["marked_equity"], abs=1e-6)
        assert cell["position_before_stop"] == pytest.approx(cell["position_after_stop"])
        assert set(cell["per_horizon_mae"]) == {str(h) for h in HORIZONS}
        assert all(row["close_cause"] in {None, "take_profit", "stop_loss", "early_prediction"}
                   for row in cell["events"])
        for row in cell["events"]:
            if row["request"] == "close":
                assert row["close_cause"] is not None
            if row["close_cause"] == "early_prediction":
                assert row["early_close_triggered"] is True
            if row["early_close_evaluated"]:
                assert row["position_before"] != 0
                assert row["close_cause"] not in {"take_profit", "stop_loss"}
                short = row["predictions"]["short"]
                long = row["predictions"]["long"]
                score = (0.6 * min(short) + 0.4 * min(long)
                         if row["direction_before"] == "long"
                         else 0.6 * max(short) + 0.4 * max(long))
                expected = (score < row["stop_loss_before"]
                            if row["direction_before"] == "long"
                            else score > row["stop_loss_before"])
                assert row["early_close_triggered"] is expected
                assert (row["request"] == "close") is expected
        for fill in cell["fills"]:
            assert fill["price"] == pytest.approx(fill["bar_open"], abs=0.0000151)
        fills_by_time = {fill["datetime"] for fill in cell["fills"]}
        for row in cell["events"]:
            if row["request"] in {"buy", "sell", "close"}:
                next_open = (pd.Timestamp(row["timestamp"]) + pd.Timedelta(hours=1)).isoformat()
                assert next_open in fills_by_time
    by_name = {cell["name"]: cell for cell in report["cells"]}
    assert all(row["close_cause"] != "early_prediction"
               for row in by_name["ideal_ideal"]["events"])
    assert sum(row["early_close_evaluated"] for row in by_name["ideal_ideal"]["events"]) == 127
    assert not any(row["early_close_triggered"] for row in by_name["ideal_ideal"]["events"])
    assert any(row["position_before"] != 0 and row["predictions"] is None
               and not row["early_close_evaluated"] for row in by_name["ideal_ideal"]["events"])
    assert all(row["request"] == "none"
               for row in by_name["persistence_persistence"]["events"])
    assert any(row["close_cause"] == "early_prediction"
               for row in by_name["short_noise_long_ideal"]["events"])
    assert sum(row["early_close_evaluated"] for row in by_name["short_noise_long_ideal"]["events"]) == 8
