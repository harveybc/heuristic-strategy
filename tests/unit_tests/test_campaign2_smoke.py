"""Smoke: run_paired_families.py on synthetic campaign-2 records (EURUSD / GBPUSD 1h).

Irregular bars (weekend gaps), separate hourly and daily records and CSVs with M07's
columns (row_id, DATE_TIME, close_origin, y_hat_z_h*, logret_hat_h*, close_hat_h*,
n_rows_h*), exact asset strings, families paired on DATE_TIME with different origins.
"""
import csv
import json
import math
from datetime import datetime, timedelta

import pytest

import run_paired_families
from app.forecast_naive_gate import seal
from app.split_half import horizon_mae
from tests.unit_tests.test_forecast_naive_gate import _horizon

ASSETS = ["EURUSD 1h (heuristic-strategy 939f5e6 eurusd_hour_2005_2020.csv; DEVELOPMENT)",
          "EURUSD 1h (lake 5m->1h, HistData lineage; DEVELOPMENT)",
          "GBPUSD 1h (lake 5m->1h, HistData lineage; DEVELOPMENT)"]
HOURLY, DAILY = list(range(1, 25)), [24, 48, 72, 96, 120, 144]


def _irregular_bars(n=400, start=datetime(2016, 1, 4)):
    bars, t, i = [], start, 0
    while len(bars) < n:
        if t.weekday() < 5:  # weekend gap: no bars Saturday/Sunday
            c = 1.10 + 0.004 * math.sin(i / 7.0)
            bars.append({"time": t.strftime("%Y-%m-%d %H:%M:%S"), "close": c})
            i += 1
        t += timedelta(hours=1)
    return bars


def _write(tmp_path, asset, family, horizons, bars, origins, passing, cid):
    rows = []
    for k in origins:
        row = {"row_id": f"fx:row{k}", "DATE_TIME": bars[k]["time"], "close_origin": bars[k]["close"]}
        for h in horizons:
            n = min(h, len(bars) - 1 - k)  # elapsed hours -> bar steps differ on irregular bars
            row[f"n_rows_h{h}"] = n
            target = math.log(bars[k + n]["close"] / bars[k]["close"]) if n > 0 else 0.0
            row[f"logret_hat_h{h}"] = target * 0.5
            row[f"y_hat_z_h{h}"] = target * 50
            row[f"close_hat_h{h}"] = bars[k]["close"] * math.exp(target * 0.5)
        rows.append(row)
    path = tmp_path / f"PREDICTIONS_{cid}.csv"
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    record = seal({
        "schema": "predictor.forecast_naive_evidence.v1", "family": family,
        "frozen_metric": {"primary": "MAE", "secondary": "MSE", "frozen": "declared before selection"},
        "artifact": {"model_sha256": cid * 8, "candidate_cid": cid * 8},
        "population": {"dataset_id": "fx", "asset": asset, "targets": ["log_return_1"], "rows": len(origins),
                       "row_ids_sha256": "a" * 64, "first_origin": rows[0]["DATE_TIME"],
                       "last_origin": rows[-1]["DATE_TIME"], "sample_hours": 1.0, "horizon_unit": "steps of 1.0 h",
                       "feature_manifest": "vDh"},
        "scale": {"metric_space": "z_train", "scaler_identity": f"{family}-train"},
        "split": {"provenance": "chronological_oof", "test_used": False, "reserved_trading_test": False},
        "naive": {"definition": "strict minimum"},
        "per_horizon": [_horizon(h, 0.5 if h in passing else 1.2, 1.0, rows=len(origins)) for h in horizons]})
    ev = tmp_path / f"EVIDENCE_{cid}.json"
    ev.write_text(json.dumps(record))
    spec = {"evidence_sha256": record["evidence_sha256"], "model_sha256": cid * 8, "period_hours": 1.0,
            "metric_space": "z_train", "scaler_identity": f"{family}-train", "horizons": horizons}
    return ev, path, spec


@pytest.mark.parametrize("asset", ASSETS)
@pytest.mark.parametrize("daily_passing, expect", [(set(DAILY), "DEVELOPMENT_NOT_CONFIRMATORY"),
                                                   (set(), "SKIPPED_NOT_BETTER_THAN_NAIVE"),
                                                   ({24, 48}, "DEVELOPMENT_NOT_CONFIRMATORY")])
def test_campaign2_layout_runs_end_to_end(tmp_path, asset, daily_passing, expect):
    bars = _irregular_bars()
    with open(tmp_path / "view.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE"])
        for b in bars:
            w.writerow([b["time"], b["close"], b["close"] * 1.001, b["close"] * 0.999, b["close"]])
    hourly_origins = list(range(30, 370))
    daily_origins = list(range(30, 250))  # daily needs 144 h of support: fewer origins
    ev_h, p_h, s_h = _write(tmp_path, asset, "hourly", HOURLY, bars, hourly_origins, set(range(1, 13)), "aaaaaaaa")
    ev_d, p_d, s_d = _write(tmp_path, asset, "daily", DAILY, bars, daily_origins, daily_passing, "bbbbbbbb")
    (tmp_path / "decl.json").write_text(json.dumps({"asset": asset, "families": {"hourly": s_h, "daily": s_d}}))
    assert run_paired_families.main([
        "--bars", str(tmp_path / "view.csv"), "--rows", "30:370",
        "--hourly-evidence", str(ev_h), "--hourly-predictions", str(p_h),
        "--daily-evidence", str(ev_d), "--daily-predictions", str(p_d),
        "--declared", str(tmp_path / "decl.json"), "--out", str(tmp_path / "out.json"),
        "--manifest-status", "FROZEN_DEVELOPMENT", "--exit-variant", "E"]) == 0
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["status"] == expect
    assert out["naive_gate"]["families"]["hourly"]["consumed_horizons"] == list(range(1, 13))
    assert out["evaluation_population"]["bars_without_daily_forecast"] == 340 - 220
    assert out["candidates"] == {"hourly": "a" * 64, "daily": "b" * 64}
    if expect != "SKIPPED_NOT_BETTER_THAN_NAIVE":
        assert out["params"]["effective_exit_variant"] == "E"
        assert out["naive_gate"]["families"]["daily"]["consumed_horizons"] == sorted(daily_passing)


def test_split_half_uses_per_row_n_rows_on_irregular_bars():
    bars = _irregular_bars(200)
    k = 100  # pick an origin right before a weekend so n_rows < h
    row = {"time": bars[k]["time"], "n_rows_h24": 3, "logret_hat_h24": 0.0}
    table = horizon_mae(bars, [row], [24], 0.0)[24]
    actual = math.log(bars[k + 3]["close"] / bars[k]["close"])
    assert table["zero_return_mae"] == pytest.approx(abs(actual))
