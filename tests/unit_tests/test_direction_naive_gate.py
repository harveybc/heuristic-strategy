"""predictor.direction_naive_evidence.v1: the classification baseline gate (coordinator design).

Per consumed horizon, a direction model is ELIGIBLE only if its held-out/OOF balanced
accuracy AND log-loss both strictly beat BOTH baselines: the TRAIN majority class and
sign persistence of the last return. Same refusals as the forecast gate.
"""
import json
import math

import pytest

from app.direction_naive_gate import (
    DIRECTION_CONTRACT_SHA256, ELIGIBLE, SCHEMA, SKIPPED, evaluate_direction, seal, validate_record,
)


def _h(h, ba, ll, *, maj=(0.5, 0.693), sp=(0.52, 0.69), rows=400):
    return {"horizon": h, "rows": rows,
            "model": {"balanced_accuracy": ba, "log_loss": ll},
            "majority_class": {"balanced_accuracy": maj[0], "log_loss": maj[1]},
            "sign_persistence": {"balanced_accuracy": sp[0], "log_loss": sp[1]}}


def _record(per_horizon, *, model="l" * 64, provenance="held_out_validation"):
    return seal({
        "schema": SCHEMA,
        "frozen_metric": {"primary": ["balanced_accuracy", "log_loss"],
                          "rule": "both strictly better than both baselines", "frozen": "declared before selection"},
        "artifact": {"model_sha256": model},
        "population": {"dataset_id": "eurusd", "asset": "EURUSD", "rows": 400, "row_ids_sha256": "a" * 64,
                       "first_origin": "2016-01-01", "last_origin": "2016-12-31", "sample_hours": 4.0,
                       "horizon_unit": "steps of 4.0 h", "label_definition": "close[t+h] > close[t]"},
        "split": {"provenance": provenance, "test_used": False, "reserved_trading_test": False},
        "baselines": {"majority_class": {"definition": "TRAIN majority class; probability = TRAIN frequency"},
                      "sign_persistence": {"definition": "sign of the last return; probability from TRAIN"}},
        "per_horizon": per_horizon})


def _declared(records, horizons=(6,)):
    return {"asset": "EURUSD", "families": {
        family: {"evidence_sha256": record["evidence_sha256"], "model_sha256": record["artifact"]["model_sha256"],
                 "period_hours": 4.0, "horizons": list(horizons)} for family, record in records.items()}}


def _decide(long=None, short=None, consumed=None, horizons=(6,)):
    records = {"long": long or _record([_h(6, 0.58, 0.67)], model="l" * 64),
               "short": short or _record([_h(6, 0.57, 0.68)], model="s" * 64)}
    return evaluate_direction(records, _declared(records, horizons), consumed or {"long": 1, "short": 1}, "EURUSD")


def _reasons(decision):
    return {f["reason"] for f in decision["failures"]}


def test_contract_digest_is_pinned():
    assert DIRECTION_CONTRACT_SHA256 == "754f016f4513a2e37f1655190f0e8915a448bd68d39eea962d681f30473c0fd5"


def test_validator_accepts_a_complete_record_and_names_missing_fields():
    record = _record([_h(6, 0.58, 0.67)])
    validate_record(record)
    for path in ("split", "baselines", "per_horizon", "population", "artifact", "frozen_metric"):
        broken = {k: v for k, v in record.items() if k != path}
        with pytest.raises(ValueError, match=path):
            validate_record(broken)
    entry = dict(record["per_horizon"][0])
    del entry["sign_persistence"]
    with pytest.raises(ValueError, match="sign_persistence"):
        validate_record(dict(record, per_horizon=[entry]))


def test_genuinely_better_on_both_metrics_against_both_baselines_is_eligible():
    decision = _decide()
    assert decision["status"] == ELIGIBLE and decision["contract_sha256"] == DIRECTION_CONTRACT_SHA256
    row = decision["horizons"][0]
    assert row["balanced_accuracy"]["best_baseline"] == pytest.approx(0.52)
    assert row["log_loss"]["best_baseline"] == pytest.approx(0.69)


@pytest.mark.parametrize("entry, reason", [
    (_h(6, 0.51, 0.60), "balanced_accuracy_not_better"),          # beats majority, not sign persistence
    (_h(6, 0.60, 0.70), "log_loss_not_better"),                   # worse log-loss than sign persistence
    (_h(6, 0.52, 0.67), "balanced_accuracy_tie"),
    (_h(6, 0.60, 0.69), "log_loss_tie"),
    (_h(6, 0.40, 0.80), "balanced_accuracy_not_better"),
])
def test_both_metrics_must_strictly_beat_both_baselines(entry, reason):
    decision = _decide(long=_record([entry]))
    assert decision["status"] == SKIPPED and reason in _reasons(decision)


@pytest.mark.parametrize("damage", ["missing_horizon", "nan", "null", "missing_baseline_metric"])
def test_missing_or_non_finite_metrics_skip(damage):
    entry = _h(6, 0.58, 0.67)
    if damage == "missing_horizon":
        entry["horizon"] = 3
    elif damage == "nan":
        entry["model"]["log_loss"] = float("nan")
    elif damage == "null":
        entry["sign_persistence"]["balanced_accuracy"] = None
    else:
        del entry["majority_class"]["log_loss"]
    record = _record([entry]) if damage != "nan" else None
    if damage == "nan":
        record = _record([_h(6, 0.58, 0.67)])
        record["per_horizon"][0]["model"]["log_loss"] = float("nan")
    decision = _decide(long=record)
    assert decision["status"] == SKIPPED
    assert _reasons(decision) & {"missing_metric", "non_finite_metric", "evidence_digest_mismatch",
                                 "evidence_invalid"}
    json.dumps(decision, allow_nan=False)


@pytest.mark.parametrize("provenance", ["reserved_trading_test", "test", "unknown", None])
def test_provenance_refusals(provenance):
    decision = _decide(short=_record([_h(6, 0.57, 0.68)], model="s" * 64, provenance=provenance))
    assert "provenance_not_admissible" in _reasons(decision)


def test_both_families_must_pass_and_identity_binds():
    decision = _decide(short=_record([_h(6, 0.50, 0.70)], model="s" * 64))
    assert [(f["family"], f["horizon"]) for f in decision["failures"]] == [("short", 6), ("short", 6)]
    records = {"long": _record([_h(6, 0.58, 0.67)], model="l" * 64),
               "short": _record([_h(6, 0.57, 0.68)], model="s" * 64)}
    declared = _declared(records)
    declared["families"]["long"]["model_sha256"] = "x" * 64
    assert "candidate_mismatch" in _reasons(evaluate_direction(records, declared, {"long": 1, "short": 1}, "EURUSD"))
    assert "consumption_not_declared" in _reasons(_decide(consumed={"long": 2, "short": 1}))
    tampered = _record([_h(6, 0.58, 0.67)], model="l" * 64)
    good_sha = tampered["evidence_sha256"]
    tampered["per_horizon"][0]["model"]["balanced_accuracy"] = 0.99
    tampered["evidence_sha256"] = good_sha
    assert "evidence_digest_mismatch" in _reasons(_decide(long=tampered))


def test_phase_runners_admit_a_direction_set_only_through_the_direction_gate(tmp_path, monkeypatch):
    import run_phase_b_cnn
    from app.runner_naive_gate import CandidateSkipped, RunnerGate, install

    records = {"long": _record([_h(6, 0.58, 0.67)], model="l" * 64),
               "short": _record([_h(6, 0.57, 0.68)], model="s" * 64)}
    declared = _declared(records)
    for family, record in records.items():
        path = tmp_path / f"{family}.json"
        path.write_text(json.dumps(record))
        declared["families"][family]["file"] = str(path)
    config = tmp_path / "direction_evidence.json"
    config.write_text(json.dumps(declared))
    gate = RunnerGate("run_phase_b_cnn", evidence_config=config, receipt_path=tmp_path / "r.json",
                      asset="EURUSD", kind="direction")
    assert gate.entry() is True
    install(gate)
    from app.runner_naive_gate import require_installed
    require_installed("cnn", {"long": 1, "short": 1})  # admitted: no exception
    with pytest.raises(CandidateSkipped):
        require_installed("cnn_one_family", {"long": 1})
    receipt = json.loads((tmp_path / "r.json").read_text())
    assert receipt["counts"] == {"evaluated": 1, "skipped": 1}
    assert receipt["entry"]["schema"].startswith("heuristic_strategy.direction_naive_gate_decision")
