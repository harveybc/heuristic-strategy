"""S09: every runner that calls plugin.evaluate_candidate directly is gated at entry.

Contract digest is unchanged (5a568589...). Each REAL entry point is driven with no
admissible evidence and must evaluate the strategy ZERO times; the oracle runner
refuses without --diagnostic-oracle. Plugin evaluate_candidate methods are wrapped
at class level only to COUNT calls; the entry points themselves are real.
"""
import json
import sys
from pathlib import Path

import pytest

from app.forecast_naive_gate import CONTRACT_SHA256, ELIGIBLE, SKIPPED
from tests.unit_tests.test_forecast_naive_gate import _declared, _evidence

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CONTRACT = "5a5685893504cf84112132c639e255d14393cbad74fe3b3ccc147363f7bfdc27"


def _evidence_config(tmp_path, evidence=None):
    evidence = evidence or _evidence()
    declared = _declared(evidence)
    for family, record in evidence.items():
        path = tmp_path / f"ev_{family}.json"
        path.write_text(json.dumps(record))
        declared["families"][family]["file"] = str(path)
    path = tmp_path / "forecast_evidence.json"
    path.write_text(json.dumps(declared))
    return path


@pytest.fixture
def counted(monkeypatch):
    """Count evaluate_candidate on every strategy plugin class (the entry points stay real)."""
    calls = []
    from app.plugins import (plugin_api_predictions, plugin_direction_atr, plugin_long_short_predictions,
                             plugin_regime_wfo)
    for module in (plugin_direction_atr, plugin_long_short_predictions, plugin_regime_wfo, plugin_api_predictions):
        real = module.Plugin.evaluate_candidate

        def wrapper(self, *args, _real=real, _name=module.__name__, **kwargs):
            calls.append(_name)
            return _real(self, *args, **kwargs)
        monkeypatch.setattr(module.Plugin, "evaluate_candidate", wrapper)
    return calls


def test_contract_is_unchanged():
    assert CONTRACT_SHA256 == EXPECTED_CONTRACT


# ------------------------------------------------------------ RunnerGate semantics


def test_runner_gate_skips_without_evidence_and_counts(tmp_path):
    from app.runner_naive_gate import RunnerGate

    gate = RunnerGate("unit", evidence_config=None, receipt_path=tmp_path / "r.json", asset="EURUSD")
    assert gate.entry() is False
    assert gate.candidate("c1", {"hourly": 6, "daily": 6}) is False
    receipt = json.loads((tmp_path / "r.json").read_text())
    assert receipt["entry"]["status"] == SKIPPED and receipt["contract_sha256"] == CONTRACT_SHA256
    assert receipt["counts"] == {"evaluated": 0, "skipped": 1}


def test_runner_gate_decides_per_candidate_prediction_set(tmp_path):
    from app.runner_naive_gate import RunnerGate

    gate = RunnerGate("unit", evidence_config=_evidence_config(tmp_path), receipt_path=tmp_path / "r.json",
                      asset="EURUSD")
    assert gate.entry() is True
    assert gate.candidate("matches", {"hourly": 6, "daily": 6}) is True
    assert gate.candidate("seven_daily_columns", {"hourly": 6, "daily": 7}) is False
    assert gate.candidate("direction_probability", None) is False
    receipt = json.loads((tmp_path / "r.json").read_text())
    assert receipt["counts"] == {"evaluated": 1, "skipped": 2}
    skipped = [c for c in receipt["candidates"] if c["status"] == SKIPPED]
    assert {c["label"] for c in skipped} == {"seven_daily_columns", "direction_probability"}
    assert all(c["failures"] for c in skipped)
    evaluated = [c for c in receipt["candidates"] if c["status"] == ELIGIBLE][0]
    assert evaluated["horizons"][0]["mae"]["baseline"] is not None  # MAE beside its same-row naive


def test_runner_gate_skips_a_candidate_with_its_own_failing_evidence(tmp_path):
    from app.runner_naive_gate import RunnerGate
    from tests.unit_tests.test_forecast_naive_gate import _horizon

    gate = RunnerGate("unit", evidence_config=_evidence_config(tmp_path), receipt_path=tmp_path / "r.json",
                      asset="EURUSD")
    assert gate.entry() is True
    daily = [_horizon(h, 0.40 * h, 0.50 * h) for h in range(1, 7)]
    daily[2] = _horizon(3, 2.0, 1.5)
    bad = tmp_path / "bad"
    bad.mkdir()
    assert gate.candidate("own_evidence", {"hourly": 6, "daily": 6},
                          evidence_config=_evidence_config(bad, _evidence(daily=daily))) is False
    row = json.loads((tmp_path / "r.json").read_text())["candidates"][-1]
    assert [(f["family"], f["horizon"]) for f in row["failures"]] == [("daily", 3)]


def test_non_prediction_plugin_is_recorded_not_applicable(tmp_path):
    from app.plugins.plugin_regime_wfo import Plugin as RegimeWFO
    from app.runner_naive_gate import NOT_APPLICABLE, RunnerGate, plugin_consumes_learned_predictions

    assert plugin_consumes_learned_predictions(RegimeWFO()) is False
    from app.plugins.plugin_direction_atr import Plugin as DirectionATR
    assert plugin_consumes_learned_predictions(DirectionATR()) is True
    gate = RunnerGate("unit", receipt_path=tmp_path / "r.json", consumes_predictions=False)
    assert gate.entry() is True and gate.candidate("fold", None) is True
    assert json.loads((tmp_path / "r.json").read_text())["entry"]["status"] == NOT_APPLICABLE


# ------------------------------------------------------------ REAL entry points: zero evaluations


def test_run_wfo_with_a_prediction_plugin_evaluates_zero_times(tmp_path, monkeypatch, counted):
    import run_wfo

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_wfo, "load_csv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("data loaded")))
    monkeypatch.setattr(sys, "argv", ["run_wfo.py", "--plugin", "ls_pred_strategy",
                                      "--base_dataset_file", str(ROOT / "tests/data/phase_2_3_base_d3.csv"),
                                      "--naive_gate_receipt", str(tmp_path / "wfo.json"),
                                      "--save_results", str(tmp_path / "wfo_results.json"),
                                      "--save_trades", str(tmp_path / "wfo_trades.csv")])
    run_wfo.main()
    assert counted == []
    assert not (tmp_path / "wfo_results.json").exists() and not (tmp_path / "wfo_trades.csv").exists()
    assert json.loads((tmp_path / "wfo.json").read_text())["entry"]["status"] == SKIPPED


def test_walk_forward_library_refuses_without_an_allowed_gate():
    from app.runner_naive_gate import GateRefused
    from app.walk_forward_optimizer import run_walk_forward

    with pytest.raises(GateRefused):
        run_walk_forward(plugin=None, full_data=None, config={})


@pytest.mark.parametrize("module_name", ["run_phase_b_cnn", "run_phase_c_ensemble", "run_phase_d_neat"])
def test_phase_runners_evaluate_zero_times_without_evidence(tmp_path, monkeypatch, counted, module_name):
    monkeypatch.chdir(tmp_path)
    module = __import__(module_name)
    # Sandbox: if the gate were broken, nothing heavy or external may start
    # (phase D would otherwise launch NEAT training in the predictor checkout).
    import subprocess
    refuse = lambda *a, **k: (_ for _ in ()).throw(AssertionError("external work started before the gate"))
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    for name in ("run_neat_optimization", "load_cnn_model", "load_all_models_and_predict",
                 "generate_full_predictions", "load_neat_champion"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, refuse)
    loads = []
    monkeypatch.setattr("app.data_handler.load_csv", lambda *a, **k: loads.append(a) or (_ for _ in ()).throw(
        AssertionError("data loaded before the gate decided")))
    monkeypatch.setattr(sys, "argv", [f"{module_name}.py", "--naive_gate_receipt", str(tmp_path / "r.json")])
    module.main()
    assert counted == [] and loads == []
    receipt = json.loads((tmp_path / "r.json").read_text())
    assert receipt["runner"] == module_name and receipt["entry"]["status"] == SKIPPED


@pytest.mark.parametrize("module_name, label", [("run_phase_b_cnn", "cnn"), ("run_phase_c_ensemble", "ens"),
                                                 ("run_phase_d_neat", "neat")])
def test_phase_backtest_functions_refuse_a_direction_prediction_set(tmp_path, counted, module_name, label):
    from app.runner_naive_gate import CandidateSkipped, RunnerGate, install

    module = __import__(module_name)
    gate = RunnerGate(module_name, evidence_config=_evidence_config(tmp_path), receipt_path=tmp_path / "r.json",
                      asset="EURUSD")
    assert gate.entry() is True  # even an eligible price-forecast record ...
    install(gate)
    backtest = getattr(module, "run_cnn_backtest", None) or getattr(module, "run_backtest")
    with pytest.raises(CandidateSkipped):  # ... cannot admit a direction-probability set
        backtest(None, object(), {}, label=label)
    assert counted == []
    assert json.loads((tmp_path / "r.json").read_text())["counts"] == {"evaluated": 0, "skipped": 1}


def test_sweep_noise_evaluates_zero_times_and_calls_no_service(tmp_path, monkeypatch, counted):
    import requests
    import sweep_noise

    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("HTTP")))
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("HTTP")))
    sweep_noise.main(["--naive_gate_receipt", str(tmp_path / "r.json")])
    assert counted == []
    assert json.loads((tmp_path / "r.json").read_text())["entry"]["failures"][0]["reason"] == \
        "api_source_consumption_not_verifiable"


def test_api_plugin_driven_outside_the_pipeline_refuses(tmp_path, counted):
    from app.plugins.plugin_api_predictions import Plugin
    from app.runner_naive_gate import GateRefused

    with pytest.raises(GateRefused, match="SKIPPED_NOT_BETTER_THAN_NAIVE"):
        Plugin().evaluate_candidate([1, 1, 1, 0.5, 2.0], None, None, None,
                                    {"prediction_source": "API", "asset": "EURUSD",
                                     "naive_gate_receipt_file": str(tmp_path / "api.json")})
    assert json.loads((tmp_path / "api.json").read_text())["status"] == SKIPPED


# ------------------------------------------------------------ oracle: refuse unless diagnostic


def test_oracle_ceiling_refuses_without_the_diagnostic_flag(tmp_path, monkeypatch, counted):
    import run_oracle_ceiling

    monkeypatch.chdir(tmp_path)
    refuse = lambda *a, **k: (_ for _ in ()).throw(AssertionError("oracle work started"))
    monkeypatch.setattr(run_oracle_ceiling, "load_csv", refuse)
    monkeypatch.setattr(run_oracle_ceiling, "_oracle_main", refuse)
    assert run_oracle_ceiling.main(["--naive_gate_receipt", str(tmp_path / "o.json")]) == 2
    assert counted == []
    receipt = json.loads((tmp_path / "o.json").read_text())
    assert receipt["status"] == "REFUSED_ORACLE_WITHOUT_DIAGNOSTIC_FLAG"
    assert not list(tmp_path.glob("oracle_*"))


def test_oracle_diagnostic_run_stamps_outputs_and_records_the_bypass(tmp_path):
    import run_oracle_ceiling

    receipt = run_oracle_ceiling.diagnostic_bypass_receipt(tmp_path / "o.json")
    assert receipt["bypass"] == "--diagnostic-oracle" and receipt["contract_sha256"] == CONTRACT_SHA256
    assert receipt["result_class"] == "DIAGNOSTIC_ORACLE_NOT_A_STRATEGY_RESULT"
    stamped = run_oracle_ceiling.stamp({"total_profit": 1.0})
    assert stamped["result_class"] == "DIAGNOSTIC_ORACLE_NOT_A_STRATEGY_RESULT"
    rows = run_oracle_ceiling.stamp_rows([{"pnl": 1.0}])
    assert rows[0]["result_class"] == "DIAGNOSTIC_ORACLE_NOT_A_STRATEGY_RESULT"


# ------------------------------------------------------------ INCIDENT_S09-MUT-01: the sandbox holds


def _porcelain(path):
    import subprocess as sp
    if not (Path(path) / ".git").exists():
        return None
    return sp.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)


def test_sandbox_stops_phase_d_at_its_first_post_gate_side_effect(tmp_path, monkeypatch, counted):
    """With the entry gate broken (the d_entry mutant), main() must raise within 1 s at the
    first post-gate side effect, and the predictor checkout must be left exactly as it was."""
    import subprocess
    import time
    import app.runner_naive_gate as runner_gate
    import run_phase_d_neat

    before = _porcelain(run_phase_d_neat.PREDICTOR_DIR)
    monkeypatch.chdir(tmp_path)

    class BrokenGate:  # what the d_entry mutant amounts to: the gate always allows
        allowed = True

    monkeypatch.setattr(runner_gate, "entry_gate_for_runner", lambda *a, **k: BrokenGate())
    refuse = lambda *a, **k: (_ for _ in ()).throw(AssertionError("external work started before the gate"))
    real_popen, real_run = subprocess.Popen, subprocess.run
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    for name in ("run_neat_optimization", "load_neat_champion"):
        monkeypatch.setattr(run_phase_d_neat, name, refuse)
    monkeypatch.setattr("app.data_handler.load_csv", refuse)
    monkeypatch.setattr(sys, "argv", ["run_phase_d_neat.py", "--naive_gate_receipt", str(tmp_path / "r.json")])
    started = time.monotonic()
    with pytest.raises(AssertionError, match="external work started before the gate"):
        run_phase_d_neat.main()
    assert time.monotonic() - started < 1.0
    assert counted == []
    monkeypatch.setattr(subprocess, "Popen", real_popen)
    monkeypatch.setattr(subprocess, "run", real_run)
    assert _porcelain(run_phase_d_neat.PREDICTOR_DIR) == before
