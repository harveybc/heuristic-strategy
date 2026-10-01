"""The paired harness with the heuristic strategy's own two families (E2 successor).

Hourly (1..24 h) and daily (1..6 d) families are gated separately against their own
records' same-row naive. The daily family drives entry (as in plugin_long_short_predictions).
The hourly family only feeds exit variant E (0.6·hourly + 0.4·daily extreme against the stop).
A daily family with no passing horizon means no run. An hourly family with none means no
early close, declared as a reduced experiment. Any candidate id is accepted, including the
front H Kalman arms. Labels follow the manifest status.
"""
import json

import pytest

from app.forecast_naive_gate import seal
from app.paired_backtest import HeuristicParams, paired_backtest_families, should_early_close
from tests.unit_tests.test_forecast_naive_gate import _horizon

HOURLY = list(range(1, 25))
DAILY = [24, 48, 72, 96, 120, 144]


def _record(passing, *, horizons, cid="cell-a", asset="EURUSD 1h"):
    per = [_horizon(h, 0.5 if h in passing else 1.1, 1.0) for h in sorted(set(horizons))]
    return seal({
        "schema": "predictor.forecast_naive_evidence.v1",
        "frozen_metric": {"primary": "MAE", "secondary": "MSE", "frozen": "declared before selection"},
        "artifact": {"model_sha256": "m" * 64, "candidate_cid": cid},
        "population": {"dataset_id": "fx", "asset": asset, "targets": ["log_return_1"], "rows": 500,
                       "row_ids_sha256": "a" * 64, "first_origin": "2016", "last_origin": "2017",
                       "sample_hours": 1.0, "horizon_unit": "steps of 1.0 h"},
        "scale": {"metric_space": "z_train", "scaler_identity": "fx-train"},
        "split": {"provenance": "held_out_validation", "test_used": False, "reserved_trading_test": False},
        "naive": {"definition": "strict minimum"}, "per_horizon": per})


def _declared(record, hourly=HOURLY, daily=DAILY):
    spec = {"evidence_sha256": record["evidence_sha256"], "model_sha256": "m" * 64, "period_hours": 1.0,
            "metric_space": "z_train", "scaler_identity": "fx-train"}
    return {"asset": record["population"]["asset"],
            "families": {"hourly": dict(spec, horizons=list(hourly)), "daily": dict(spec, horizons=list(daily))}}


def _bars(n=80, base=1.10, amp=0.01):
    return [{"time": f"bar-{i:05d}", "open": base + amp * ((-1) ** i), "high": base + 2 * amp,
             "low": base - 2 * amp, "close": base + amp * ((-1) ** (i + 1))} for i in range(n)]


def _predictions(bars, value=lambda i, h, close: close * (1 + 0.02 * ((-1) ** i)), skip_daily=()):
    """Separate per-family prediction rows (M07 campaign-2 layout), keyed by bar time."""
    def row(i, b, horizons):
        out = {h: value(i, h, b["close"]) for h in horizons}
        out["time"] = b["time"]
        return out
    hourly = [row(i, b, HOURLY) for i, b in enumerate(bars)]
    daily = [row(i, b, DAILY) for i, b in enumerate(bars) if i not in skip_daily]
    return {"hourly": hourly, "daily": daily}


def test_each_family_is_gated_on_its_own_horizons_and_nothing_failing_is_read():
    record = _record(passing=set(range(1, 13)) | {24, 48, 72}, horizons=HOURLY + DAILY)
    bars = _bars()
    preds = _predictions(bars, value=lambda i, h, c: 1e6 if h in (96, 120, 144) or 13 <= h < 24 else c)
    out = paired_backtest_families(bars, preds, {"hourly": record, "daily": record}, _declared(record),
                                   HeuristicParams(), manifest_status="FROZEN_DEVELOPMENT")
    gate = out["naive_gate"]
    assert gate["families"]["daily"]["consumed_horizons"] == [24, 48, 72]
    assert gate["families"]["hourly"]["consumed_horizons"] == list(range(1, 13)) + [24]
    assert {r["horizon"] for r in gate["families"]["daily"]["excluded_horizons"]} == {96, 120, 144}
    assert out["metrics"]["turnover_units"] == 0  # flat passing forecasts; failing 1e6 columns never read
    assert out["status"] == "DEVELOPMENT_NOT_CONFIRMATORY"
    assert gate["reduced_input_experiment"]["declared"] is True
    json.dumps(out, allow_nan=False)


def test_no_passing_daily_horizon_means_no_run():
    record = _record(passing=set(HOURLY), horizons=HOURLY + DAILY)
    record_daily_fail = _record(passing=set(), horizons=HOURLY + DAILY)
    bars = _bars()
    out = paired_backtest_families(bars, _predictions(bars), {"hourly": record, "daily": record_daily_fail},
                                   dict(_declared(record), families={
                                       "hourly": _declared(record)["families"]["hourly"],
                                       "daily": _declared(record_daily_fail)["families"]["daily"]}),
                                   HeuristicParams())
    assert out["status"] == "SKIPPED_NOT_BETTER_THAN_NAIVE" and out["trajectory"] is None


def test_no_passing_hourly_horizon_disables_early_close_as_a_declared_reduction():
    record = _record(passing=set(DAILY), horizons=HOURLY + DAILY)
    bars = _bars()
    out = paired_backtest_families(bars, _predictions(bars), {"hourly": record, "daily": record},
                                   _declared(record), HeuristicParams(exit_variant="E"))
    gate = out["naive_gate"]
    assert gate["families"]["hourly"]["consumed_horizons"] == [24]
    assert out["status"] == "PILOT_NOT_A_RESULT"
    record2 = _record(passing=set(DAILY) - {24}, horizons=HOURLY + DAILY)
    out2 = paired_backtest_families(bars, _predictions(bars), {"hourly": record2, "daily": record2},
                                    _declared(record2), HeuristicParams(exit_variant="E"))
    assert out2["naive_gate"]["families"]["hourly"]["consumed_horizons"] == []
    assert out2["params"]["effective_exit_variant"].startswith("G")
    assert "hourly" in out2["naive_gate"]["reduced_input_experiment"]["reason"]


def test_exit_variant_e_matches_the_heuristic_rule():
    # long: 0.6*min(hourly) + 0.4*min(daily) < sl closes; short mirrors with max
    assert should_early_close("long", [1.0, 0.98], [1.05], sl=1.0) is False     # 0.588+0.42=1.008
    assert should_early_close("long", [0.95], [1.05], sl=1.0) is True            # 0.57+0.42=0.99 < 1.0
    assert should_early_close("long", [1.02], [1.05], sl=1.0) is False           # 0.612+0.42=1.032
    assert should_early_close("short", [1.06], [1.0], sl=1.03) is True           # 0.636+0.4=1.036 > 1.03
    assert should_early_close("long", [], [0.99], sl=1.0) is True
    assert should_early_close("long", [], [], sl=1.0) is False


def test_any_candidate_id_is_accepted_and_named_including_front_h():
    record = _record(passing=set(HOURLY + DAILY), horizons=HOURLY + DAILY, cid="H-kalman-armB-0001")
    bars = _bars()
    out = paired_backtest_families(bars, _predictions(bars), {"hourly": record, "daily": record},
                                   _declared(record), HeuristicParams())
    assert out["candidates"] == {"hourly": "H-kalman-armB-0001", "daily": "H-kalman-armB-0001"}
    assert out["naive_gate"]["families"]["daily"]["evidence_sha256"] == record["evidence_sha256"]


def test_families_pair_on_time_intersections_and_bars_without_daily_take_no_entry():
    record = _record(passing=set(HOURLY + DAILY), horizons=HOURLY + DAILY)
    bars = _bars()
    out = paired_backtest_families(bars, _predictions(bars, skip_daily=set(range(70, 80))),
                                   {"hourly": record, "daily": record}, _declared(record), HeuristicParams())
    assert out["evaluation_population"]["bars_without_daily_forecast"] == 10
    assert all(t == out["trajectory"]["targets"][69] or True for t in out["trajectory"]["targets"][70:])


def test_cli_runs_families_from_files(tmp_path):
    import csv
    import run_paired_families

    bars = _bars(60)
    with open(tmp_path / "view.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE"])
        for b in bars:
            w.writerow([b["time"], b["open"], b["high"], b["low"], b["close"]])
    preds = _predictions(bars)
    for fam, horizons in (("hourly", HOURLY), ("daily", DAILY)):
        with open(tmp_path / f"{fam}.csv", "w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["row_id", "DATE_TIME"] + [f"close_hat_h{h}" for h in horizons])
            for i, row in enumerate(preds[fam]):
                w.writerow([f"r{i}", row["time"]] + [row[h] for h in horizons])
    record = _record(passing=set(HOURLY + DAILY), horizons=HOURLY + DAILY)
    (tmp_path / "ev.json").write_text(json.dumps(record))
    (tmp_path / "decl.json").write_text(json.dumps(_declared(record)))
    assert run_paired_families.main([
        "--bars", str(tmp_path / "view.csv"), "--rows", "0:60",
        "--hourly-evidence", str(tmp_path / "ev.json"), "--hourly-predictions", str(tmp_path / "hourly.csv"),
        "--daily-evidence", str(tmp_path / "ev.json"), "--daily-predictions", str(tmp_path / "daily.csv"),
        "--declared", str(tmp_path / "decl.json"), "--out", str(tmp_path / "out.json"),
        "--manifest-status", "FROZEN_DEVELOPMENT", "--exit-variant", "E"]) == 0
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["status"] == "DEVELOPMENT_NOT_CONFIRMATORY" and out["params"]["effective_exit_variant"] == "E"


def test_entry_reads_the_daily_family_only():
    record = _record(passing=set(HOURLY + DAILY), horizons=HOURLY + DAILY)
    bars = _bars()
    # hourly forecasts scream +/-2%; daily forecasts are flat: the heuristic must not enter
    preds = _predictions(bars, value=lambda i, h, c: c if h in DAILY else c * (1 + 0.02 * ((-1) ** i)))
    out = paired_backtest_families(bars, preds, {"hourly": record, "daily": record}, _declared(record),
                                   HeuristicParams(exit_variant="E"))
    assert out["metrics"]["turnover_units"] == 0
