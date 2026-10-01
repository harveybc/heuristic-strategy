"""Forecast-versus-naive gate for heuristic-strategy backtests (owner order b327b771 s5).

Unit cases decide from synthetic M04-format records (``predictor.forecast_naive_evidence.v1``).
The entry-point cases run the REAL ``run_processing_pipeline`` (what ``app/main.py``
calls) with the real ``ls_pred_strategy`` plugin on the bundled data, and the real
``app/main.py`` in a subprocess, and count strategy evaluations.
"""
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.forecast_naive_gate import (
    CONTRACT_SHA256, ELIGIBLE, SKIPPED, evaluate, gate_run, seal,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data"
SCALER_H = "eurusd-1h-train-standard:" + "b" * 64
SCALER_D = "eurusd-1d-train-standard:" + "d" * 64
PINNED_CONTRACT_SHA256 = "5a5685893504cf84112132c639e255d14393cbad74fe3b3ccc147363f7bfdc27"


def _horizon(h, model_mae, naive_mae, *, rows=500):
    def pair(m, n):
        if m is None or n is None or not (math.isfinite(m) and math.isfinite(n)):
            return {"skill": None, "delta": None, "status": "NOT_AVAILABLE", "reason": "missing or nonfinite metric"}
        if n == 0:
            return {"skill": None, "delta": m - n, "status": "NOT_AVAILABLE", "reason": "ZERO_NAIVE"}
        return {"skill": 1 - m / n, "delta": m - n, "status": "OK"}
    return {"horizon": h, "rows": rows, "model_MAE": model_mae, "naive_MAE": naive_mae,
            "model_MSE": model_mae ** 2, "naive_MSE": naive_mae ** 2,
            "MAE": pair(model_mae, naive_mae), "MSE": pair(model_mae ** 2, naive_mae ** 2)}


def _record(per_horizon, *, hours, scaler, model, rows=500):
    return seal({
        "schema": "predictor.forecast_naive_evidence.v1",
        "frozen_metric": {"primary": "MAE", "secondary": "MSE", "frozen": "declared before selection"},
        "artifact": {"candidate_cid": "c" * 64, "model_sha256": model, "weights_sha256": "w" * 64,
                     "predictor_revision": "r" * 40, "campaign_id": "test"},
        "population": {"dataset_id": "data-gov:fx:eurusd", "asset": "EURUSD", "targets": ["CLOSE"],
                       "rows": rows, "row_ids_sha256": "a" * 64, "first_origin": "2016-01-01T00:00:00+00:00",
                       "last_origin": "2016-12-31T23:00:00+00:00", "sample_hours": hours,
                       "horizon_unit": f"steps of {hours} h", "validation_sha256": "v" * 64},
        "scale": {"metric_space": "z_train", "scaler_identity": scaler, "reduction": "mean"},
        "split": {"provenance": "held_out_validation", "test_used": False, "reserved_trading_test": False},
        "naive": {"definition": "persistence"},
        "per_horizon": per_horizon})


def _evidence(hourly=None, daily=None):
    hourly = hourly or [_horizon(h, 0.10 * h, 0.12 * h) for h in range(1, 7)]
    daily = daily or [_horizon(h, 0.40 * h, 0.50 * h) for h in range(1, 7)]
    return {"hourly": _record(hourly, hours=1.0, scaler=SCALER_H, model="h" * 64),
            "daily": _record(daily, hours=24.0, scaler=SCALER_D, model="d" * 64)}


def _declared(evidence, **overrides):
    declared = {"asset": "EURUSD", "families": {}}
    for family, hours, scaler, model in (("hourly", 1.0, SCALER_H, "h" * 64), ("daily", 24.0, SCALER_D, "d" * 64)):
        declared["families"][family] = {
            "evidence_sha256": evidence[family]["evidence_sha256"], "model_sha256": model,
            "period_hours": hours, "metric_space": "z_train", "scaler_identity": scaler,
            "horizons": [1, 2, 3, 4, 5, 6], **overrides.get(family, {})}
    return declared


CONSUMED = {"hourly": 6, "daily": 6}


def _decide(evidence, declared=None, consumed=CONSUMED, asset="EURUSD"):
    return evaluate(evidence, declared if declared is not None else _declared(evidence), consumed, asset)


def _reasons(decision):
    return {f["reason"] for f in decision["failures"]}


# ------------------------------------------------------------ the required decisions


def test_contract_digest_is_pinned_and_carried():
    assert CONTRACT_SHA256 == PINNED_CONTRACT_SHA256
    assert _decide(_evidence())["contract_sha256"] == CONTRACT_SHA256


def test_genuinely_passing_configuration_is_eligible():
    decision = _decide(_evidence())
    assert decision["status"] == ELIGIBLE and decision["failures"] == [] and len(decision["horizons"]) == 12
    row = decision["horizons"][0]
    assert row["mae"]["baseline"] == pytest.approx(0.12) and row["mae"]["skill"] == pytest.approx(1 - 0.10 / 0.12)
    assert row["mse"]["baseline"] == pytest.approx(0.0144)
    assert decision["provenance"] == {"hourly": "held_out_validation", "daily": "held_out_validation"}


def test_one_failing_short_horizon_skips():
    hourly = [_horizon(h, 0.10 * h, 0.12 * h) for h in range(1, 7)]
    hourly[2] = _horizon(3, 0.40, 0.36)
    decision = _decide(_evidence(hourly=hourly))
    assert decision["status"] == SKIPPED
    assert [(f["family"], f["horizon"], f["reason"]) for f in decision["failures"]] == [
        ("hourly", 3, "not_better_than_naive")]


def test_one_failing_long_horizon_skips():
    daily = [_horizon(h, 0.40 * h, 0.50 * h) for h in range(1, 7)]
    daily[5] = _horizon(6, 3.1, 3.0)
    decision = _decide(_evidence(daily=daily))
    assert [(f["family"], f["horizon"]) for f in decision["failures"]] == [("daily", 6)]


def test_favourable_macro_mean_cannot_mask_a_failing_member():
    daily = [_horizon(h, 0.1, 1.0) for h in range(1, 7)]
    daily[0] = _horizon(1, 1.01, 1.00)
    decision = _decide(_evidence(daily=daily))
    assert decision["macro"]["daily"]["mae"]["model"] < decision["macro"]["daily"]["mae"]["baseline"]
    assert decision["status"] == SKIPPED and _reasons(decision) == {"not_better_than_naive"}


def test_equality_with_naive_is_not_a_pass():
    hourly = [_horizon(h, 0.10 * h, 0.12 * h) for h in range(1, 7)]
    hourly[0] = _horizon(1, 0.12, 0.12)
    assert _reasons(_decide(_evidence(hourly=hourly))) == {"tie_with_naive"}


def test_zero_naive_rejects_without_infinite_skill():
    hourly = [_horizon(h, 0.10 * h, 0.12 * h) for h in range(1, 7)]
    hourly[1] = _horizon(2, 0.0, 0.0)
    decision = _decide(_evidence(hourly=hourly))
    assert decision["status"] == SKIPPED and _reasons(decision) == {"naive_zero"}
    row = [r for r in decision["horizons"] if r["family"] == "hourly" and r["horizon"] == 2][0]
    assert row["mae"]["skill"] == "NOT_AVAILABLE"
    json.dumps(decision, allow_nan=False)


@pytest.mark.parametrize("damage", ["missing_horizon", "null_model", "null_naive"])
def test_missing_or_nan_metric_skips(damage):
    hourly = [_horizon(h, 0.10 * h, 0.12 * h) for h in range(1, 7)]
    if damage == "missing_horizon":
        hourly = hourly[:5]
    elif damage == "null_model":
        hourly[4]["model_MAE"] = None
    else:
        hourly[4]["naive_MAE"] = None
    decision = _decide(_evidence(hourly=hourly))
    assert decision["status"] == SKIPPED and _reasons(decision) == {"missing_metric"}


def test_nan_inside_a_record_is_refused():
    evidence = _evidence()
    declared = _declared(evidence)
    evidence["hourly"]["per_horizon"][0]["model_MAE"] = float("nan")
    decision = _decide(evidence, declared)
    assert decision["status"] == SKIPPED
    assert {"non_finite_metric", "evidence_digest_mismatch"} <= _reasons(decision)


@pytest.mark.parametrize("what, reason", [("rows", "rows_mismatch"), ("scaler", "scaler_mismatch"),
                                          ("period", "period_mismatch")])
def test_mismatched_rows_scaler_or_period_skips(what, reason):
    evidence = _evidence()
    declared = _declared(evidence)
    if what == "rows":
        hourly = [_horizon(h, 0.10 * h, 0.12 * h) for h in range(1, 7)]
        hourly[0]["rows"] = 499
        evidence = _evidence(hourly=hourly)
        declared = _declared(evidence)
    elif what == "scaler":
        declared["families"]["hourly"]["scaler_identity"] = "other"
    else:
        declared["families"]["daily"]["period_hours"] = 1.0
    assert _reasons(_decide(evidence, declared)) == {reason}


# ------------------------------------------------------------ provenance, identity, consumption


@pytest.mark.parametrize("split", ["reserved_trading_test", "test", "unknown", None])
def test_trading_test_or_unknown_provenance_is_refused(split):
    evidence = _evidence()
    evidence["daily"]["split"]["provenance"] = split
    evidence["daily"] = seal(evidence["daily"])
    decision = _decide(evidence)
    assert "provenance_not_admissible" in _reasons(decision) and decision["provenance"]["daily"] == split


def test_metric_switch_tampering_identity_and_asset_are_refused():
    evidence = _evidence()
    evidence["hourly"]["frozen_metric"]["primary"] = "MSE"
    evidence["hourly"] = seal(evidence["hourly"])
    assert "primary_metric_not_mae" in _reasons(_decide(evidence))
    tampered = _evidence()
    declared = _declared(tampered)
    tampered["hourly"]["per_horizon"][0]["model_MAE"] = 0.0
    assert "evidence_digest_mismatch" in _reasons(_decide(tampered, declared))
    swapped = _evidence()
    assert "candidate_mismatch" in _reasons(_decide(swapped, _declared(swapped, hourly={"model_sha256": "x" * 64})))
    assert "asset_mismatch" in _reasons(_decide(_evidence(), asset="USDJPY"))
    assert "asset_not_declared" in _reasons(_decide(_evidence(), asset=None))


def test_consumed_columns_must_match_the_declaration_and_nothing_is_dropped():
    assert "consumption_not_declared" in _reasons(_decide(_evidence(), consumed={"hourly": 6, "daily": 7}))
    off_grid = _declared(_evidence(), daily={"horizons": [24, 48, 72, 96, 120, 144]})
    decision = _decide(_evidence(), off_grid)
    assert {(f["family"], f["reason"]) for f in decision["failures"]} == {("daily", "missing_metric")}
    assert len(decision["failures"]) == 6


def test_api_auto_generated_and_unconfigured_runs_skip():
    assert gate_run({"prediction_source": "API"}, CONSUMED)["failures"][0]["reason"] == \
        "api_source_consumption_not_verifiable"
    assert gate_run({}, None)["failures"][0]["reason"] == "auto_generated_predictions_not_admissible"
    assert gate_run({"asset": "EURUSD"}, CONSUMED)["failures"][0]["reason"] == "evidence_not_configured"


def test_receipt_reports_seasonal_naive_without_deciding():
    decision = _decide(_evidence())
    assert decision["baselines"]["hourly"]["seasonal_naive"]["status"] == "NOT_AVAILABLE"
    hourly = [_horizon(h, 0.10 * h, 0.12 * h) for h in range(1, 7)]
    for entry in hourly[:5]:  # M04 b5ed0982 nested form; better than the model: reported only
        entry["seasonal_naive"] = {"naive_MAE": 0.01, "naive_MSE": 0.0001,
                                   "MAE": {"skill": None, "delta": None, "status": "OK"}, "MSE": {}}
    hourly[5]["seasonal_naive"] = {"status": "NOT_AVAILABLE",
                                   "reason": "target time minus one period is not inside the input window"}
    record = _record(hourly, hours=1.0, scaler=SCALER_H, model="h" * 64)
    record["seasonal_naive"] = {"period_steps": 5, "declared": True, "definition": "value one period earlier"}
    evidence = dict(_evidence(), hourly=seal(record))
    decision = _decide(evidence)
    assert decision["status"] == ELIGIBLE
    assert decision["baselines"]["hourly"]["seasonal_naive"]["period_steps"] == 5
    first = decision["horizons"][0]["seasonal_naive"]
    assert first["used_for_eligibility"] is False and first["mae"]["delta"] > 0
    assert decision["horizons"][5]["seasonal_naive"]["reason"].startswith("target time minus one period")


# ------------------------------------------------------------ the REAL backtest entry point


def _run_config(tmp_path, evidence, *, declared=None):
    files = {}
    if evidence is not None:
        declared = declared or _declared(evidence)
        for family, record in evidence.items():
            path = tmp_path / f"evidence_{family}.json"
            path.write_text(json.dumps(record))
            declared["families"][family]["file"] = str(path)
    return {
        "asset": "EURUSD", "forecast_evidence": declared,
        "base_dataset_file": str(DATA / "phase_2_3_base_d3.csv"),
        "hourly_predictions_file": str(DATA / "phase_2_3_cnn_1h_prediction_d3.csv"),
        "daily_predictions_file": str(DATA / "phase_2_3_cnn_1d_prediction_d3.csv"),
        "date_column": "DATE_TIME", "headers": True, "max_steps": 600, "force_date": False,
        "population_size": 2, "num_generations": 1, "crossover_probability": 0.5,
        "mutation_probability": 0.2, "disable_multiprocessing": True, "prediction_source": "CSV",
        "load_parameters": None, "use_normalization_json": None, "time_horizon": 6,
        "trades_csv_file": str(tmp_path / "trades.csv"), "summary_csv_file": str(tmp_path / "summary.csv"),
        "balance_plot_file": str(tmp_path / "balance_plot.png"),
        "save_parameters": str(tmp_path / "parameters.json"),
        "naive_gate_receipt_file": str(tmp_path / "naive_gate_receipt.json"), **files}


def _pipeline(tmp_path, monkeypatch, evidence, **kw):
    from app.data_processor import run_processing_pipeline
    from app.plugins.plugin_long_short_predictions import Plugin

    monkeypatch.chdir(tmp_path)
    plugin = Plugin()
    calls = []
    real = plugin.evaluate_candidate

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    plugin.evaluate_candidate = counting
    info, trades = run_processing_pipeline(_run_config(tmp_path, evidence, **kw), plugin)
    return info, trades, calls


def test_failing_gate_evaluates_the_strategy_zero_times_at_the_real_pipeline(tmp_path, monkeypatch):
    daily = [_horizon(h, 0.40 * h, 0.50 * h) for h in range(1, 7)]
    daily[3] = _horizon(4, 3.0, 2.0)
    info, trades, calls = _pipeline(tmp_path, monkeypatch, _evidence(daily=daily))
    assert calls == [] and trades is None and info["status"] == SKIPPED
    for name in ("trades.csv", "summary.csv", "parameters.json", "balance_plot.png"):
        assert not (tmp_path / name).exists()
    receipt = json.loads((tmp_path / "naive_gate_receipt.json").read_text())
    assert receipt["status"] == SKIPPED and receipt["contract_sha256"] == CONTRACT_SHA256
    assert [(f["family"], f["horizon"]) for f in receipt["failures"]] == [("daily", 4)]


def test_missing_evidence_evaluates_zero_times_at_the_real_pipeline(tmp_path, monkeypatch):
    info, _, calls = _pipeline(tmp_path, monkeypatch, None)
    assert calls == [] and info["forecast_naive_gate"]["failures"][0]["reason"] == "evidence_not_configured"


def test_passing_gate_lets_the_real_pipeline_evaluate(tmp_path, monkeypatch):
    info, _, calls = _pipeline(tmp_path, monkeypatch, _evidence())
    assert len(calls) >= 1 and "status" not in info
    assert json.loads((tmp_path / "naive_gate_receipt.json").read_text())["status"] == ELIGIBLE


def test_failing_gate_at_the_real_cli_entry_point_writes_no_trajectory(tmp_path):
    hourly = [_horizon(h, 0.10 * h, 0.12 * h) for h in range(1, 7)]
    hourly[0] = _horizon(1, 0.12, 0.12)
    config = _run_config(tmp_path, _evidence(hourly=hourly))
    config_path = tmp_path / "run_config.json"
    config_path.write_text(json.dumps(config))
    env = dict(os.environ, PYTHONPATH=str(ROOT), STRATEGY_QUIET="0")
    out = subprocess.run([sys.executable, str(ROOT / "app" / "main.py"), "--plugin", "ls_pred_strategy",
                          "--load_config", str(config_path), "--save_config", str(tmp_path / "config_out.json")],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "SKIPPED_NOT_BETTER_THAN_NAIVE: strategy not evaluated" in out.stdout
    assert "Running optimizer" not in out.stdout
    for name in ("trades.csv", "summary.csv", "parameters.json", "balance_plot.png"):
        assert not (tmp_path / name).exists()
    receipt = json.loads((tmp_path / "naive_gate_receipt.json").read_text())
    assert receipt["status"] == SKIPPED and receipt["failures"][0]["reason"] == "tie_with_naive"
