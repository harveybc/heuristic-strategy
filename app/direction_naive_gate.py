"""Classification baseline gate for direction predictions (stdlib only).

Coordinator design (from question 21): a direction-probability model may reach the
heuristic strategy only if, for every consumed horizon and every consumed family, its
held-out or chronological out-of-fold **balanced accuracy and log-loss both strictly
beat BOTH baselines**:

* ``majority_class``: always predicts the TRAIN majority class, with its probability
  set to the TRAIN class frequency;
* ``sign_persistence``: predicts the sign of the last return, with a probability
  estimated on TRAIN.

The evidence is ``predictor.direction_naive_evidence.v1`` (schema in
docs/contracts/predictor.direction_naive_evidence.v1.json). The refusals are the
forecast gate's: missing or non-finite metric, a tie, the reserved trading test,
unknown provenance, ``test_used``, a tampered or undeclared record, and an identity
or consumption mismatch. ``DIRECTION_CONTRACT_SHA256`` pins this rule set and is
carried by every decision.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

SCHEMA = "predictor.direction_naive_evidence.v1"
DECISION_SCHEMA = "heuristic_strategy.direction_naive_gate_decision.v1"
ELIGIBLE = "ELIGIBLE"
SKIPPED = "SKIPPED_NOT_BETTER_THAN_NAIVE"
NOT_AVAILABLE = "NOT_AVAILABLE"
ADMISSIBLE_PROVENANCE = ("held_out_validation", "chronological_oof")
BASELINES = ("majority_class", "sign_persistence")
METRICS = {"balanced_accuracy": "higher", "log_loss": "lower"}

DIRECTION_CONTRACT = {
    "contract": "heuristic_strategy.direction_naive_gate.v1",
    "design": "coordinator design from question 21",
    "evidence_schema": SCHEMA, "decision_schema": DECISION_SCHEMA,
    "metrics": METRICS, "baselines": list(BASELINES),
    "rule": "on every consumed horizon of every consumed family: model balanced_accuracy > max(baselines) "
            "and model log_loss < min(baselines), strictly, all finite",
    "admissible_provenance": list(ADMISSIBLE_PROVENANCE),
    "refuses": ["tie", "missing metric", "non-finite metric", "unknown provenance", "reserved trading test",
                "test_used", "tampered record", "undeclared record", "candidate mismatch", "asset mismatch",
                "period mismatch", "rows mismatch", "consumption mismatch"],
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


DIRECTION_CONTRACT_SHA256 = hashlib.sha256(_canonical(DIRECTION_CONTRACT).encode()).hexdigest()


def evidence_digest(record: Mapping[str, Any]) -> str | None:
    body = {k: v for k, v in record.items() if k != "evidence_sha256"}
    try:
        return hashlib.sha256(_canonical(body).encode()).hexdigest()
    except ValueError:
        return None


def seal(record: dict) -> dict:
    record = {k: v for k, v in record.items() if k != "evidence_sha256"}
    record["evidence_sha256"] = evidence_digest(record)
    return record


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


_REQUIRED = {
    "schema": None, "frozen_metric": ("primary", "rule"), "artifact": ("model_sha256",),
    "population": ("asset", "rows", "row_ids_sha256", "first_origin", "last_origin", "sample_hours",
                   "horizon_unit", "label_definition"),
    "split": ("provenance", "test_used", "reserved_trading_test"),
    "baselines": BASELINES, "per_horizon": None,
}


def validate_record(record: Mapping[str, Any]) -> None:
    """Structural validation; raises ValueError naming the first missing field."""
    for key, fields in _REQUIRED.items():
        if key not in record:
            raise ValueError(f"direction evidence lacks {key}")
        for field in fields or ():
            if field not in record[key]:
                raise ValueError(f"direction evidence lacks {key}.{field}")
    if record["schema"] != SCHEMA:
        raise ValueError(f"schema {record['schema']!r} is not {SCHEMA}")
    if not isinstance(record["per_horizon"], list) or not record["per_horizon"]:
        raise ValueError("direction evidence lacks per_horizon entries")
    for entry in record["per_horizon"]:
        for side in ("horizon", "rows", "model", *BASELINES):
            if side not in entry:
                raise ValueError(f"direction evidence per_horizon entry lacks {side}")
        for side in ("model", *BASELINES):
            for metric in METRICS:
                if metric not in entry[side]:
                    raise ValueError(f"direction evidence per_horizon.{side} lacks {metric}")


def evaluate_direction(evidence: Mapping[str, Any], declared: Mapping[str, Any] | None,
                       consumed: Mapping[str, int], asset: str | None,
                       families: tuple = ("long", "short")) -> dict[str, Any]:
    failures, horizons, provenance, population = [], [], {}, {}

    def fail(reason, family=None, horizon=None, detail=None):
        failures.append({"reason": reason, "family": family, "horizon": horizon, "detail": detail})

    def done():
        return json.loads(_canonical({
            "schema": DECISION_SCHEMA, "contract_sha256": DIRECTION_CONTRACT_SHA256,
            "status": ELIGIBLE if not failures else SKIPPED, "asset": asset, "provenance": provenance,
            "population": population, "horizons": horizons, "failures": failures}))

    if not isinstance(declared, Mapping):
        fail("evidence_not_configured", detail="no direction evidence declaration")
        return done()
    if declared.get("asset") != asset:
        fail("asset_mismatch", detail={"declared": declared.get("asset"), "run": asset})
    for family in families:
        spec = (declared.get("families") or {}).get(family)
        if not isinstance(spec, Mapping) or not isinstance(spec.get("horizons"), list):
            fail("consumption_not_declared", family)
            continue
        if consumed.get(family) != len(spec["horizons"]):
            fail("consumption_not_declared", family,
                 detail={"consumed": consumed.get(family), "declared": len(spec["horizons"])})
            continue
        record = evidence.get(family)
        if not isinstance(record, Mapping):
            fail("evidence_missing", family)
            continue
        try:
            validate_record(record)
        except ValueError as exc:
            fail("evidence_invalid", family, detail=str(exc))
            continue
        digest = evidence_digest(record)
        if digest is None or digest != record.get("evidence_sha256"):
            fail("evidence_digest_mismatch", family)
        if record.get("evidence_sha256") != spec.get("evidence_sha256"):
            fail("evidence_not_declared_record", family)
        split = record["split"]
        provenance[family] = split.get("provenance")
        if (split.get("provenance") not in ADMISSIBLE_PROVENANCE or split.get("test_used") is not False
                or split.get("reserved_trading_test") is not False):
            fail("provenance_not_admissible", family, detail=split)
        if record["artifact"].get("model_sha256") != spec.get("model_sha256"):
            fail("candidate_mismatch", family)
        pop = record["population"]
        if pop.get("asset") != declared.get("asset"):
            fail("asset_mismatch", family)
        if pop.get("sample_hours") != spec.get("period_hours"):
            fail("period_mismatch", family)
        population[family] = {k: pop.get(k) for k in ("asset", "rows", "row_ids_sha256", "first_origin",
                                                       "last_origin", "sample_hours", "label_definition")}
        entries = {e.get("horizon"): e for e in record["per_horizon"]}
        for horizon in spec["horizons"]:
            entry = entries.get(horizon)
            if entry is None:
                fail("missing_metric", family, horizon, "consumed horizon absent from the record")
                horizons.append({"family": family, "horizon": horizon, "status": NOT_AVAILABLE})
                continue
            if entry.get("rows") != pop.get("rows"):
                fail("rows_mismatch", family, horizon)
            row = {"family": family, "horizon": horizon, "rows": entry.get("rows")}
            for metric, better in METRICS.items():
                model = entry["model"].get(metric)
                bases = [entry[b].get(metric) for b in BASELINES]
                if model is None or any(b is None for b in bases):
                    fail("missing_metric", family, horizon, metric)
                    row[metric] = {"status": NOT_AVAILABLE, "reason": "missing metric"}
                    continue
                if not all(_finite(v) for v in (model, *bases)):
                    fail("non_finite_metric", family, horizon, metric)
                    row[metric] = {"status": NOT_AVAILABLE, "reason": "non-finite metric"}
                    continue
                best = max(bases) if better == "higher" else min(bases)
                row[metric] = {"model": model, "best_baseline": best,
                               "baselines": dict(zip(BASELINES, bases)), "delta": model - best}
                if model == best:
                    fail(f"{metric}_tie", family, horizon)
                elif (model < best) if better == "higher" else (model > best):
                    fail(f"{metric}_not_better", family, horizon, {"model": model, "best_baseline": best})
            horizons.append(row)
    return done()
