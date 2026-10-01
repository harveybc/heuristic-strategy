"""Forecast-versus-naive eligibility gate for heuristic-strategy backtests (stdlib only).

Owner order (predictor b327b771, section 5): predictions that have not beaten
persistence never reach the heuristic strategy, in any runner.  This module is
the backtest runner's copy of the rule.  It is written here rather than imported
from ``lts`` (no shared package exists), and it pins its rule set as
``CONTRACT`` with digest ``CONTRACT_SHA256``, which every decision carries.

Evidence: M04's frozen ``predictor.forecast_naive_evidence.v1`` record
(predictor ``tools/modular_forecast_evidence.py``), one per prediction family
(``hourly``, ``daily``).  The run config declares, under ``forecast_evidence``,
the asset and, per family, the record (``file``, ``evidence_sha256``), the model
(``model_sha256``), ``period_hours``, ``metric_space``, ``scaler_identity`` and
``horizons``: the record horizon each consumed prediction column maps onto.

ELIGIBLE only if every consumed horizon of every family satisfies all of these:
the record verifies and is the declared one; MAE is the frozen primary; the
provenance is held-out validation or chronological OOF with ``test_used`` and
``reserved_trading_test`` false; asset, period, scale, scaler and model match;
the horizon was scored on the population's rows; model and naive MAE are finite;
naive MAE is not zero; and model MAE < naive MAE strictly.  Otherwise the
decision is SKIPPED_NOT_BETTER_THAN_NAIVE, listing every failure.  A seasonal
naive is reported when the evidence holds one, and never decides.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

EVIDENCE_SCHEMA = "predictor.forecast_naive_evidence.v1"
DECISION_SCHEMA = "heuristic_strategy.forecast_naive_gate_decision.v1"
ELIGIBLE = "ELIGIBLE"
SKIPPED = "SKIPPED_NOT_BETTER_THAN_NAIVE"
NOT_AVAILABLE = "NOT_AVAILABLE"
FAMILIES = ("hourly", "daily")
ADMISSIBLE_PROVENANCE = ("held_out_validation", "chronological_oof")

CONTRACT = {
    "contract": "heuristic_strategy.forecast_naive_gate.v1",
    "owner_order": "predictor b327b771 section 5",
    "evidence_schema": EVIDENCE_SCHEMA,
    "decision_schema": DECISION_SCHEMA,
    "primary_metric": "MAE", "reported_metrics": ["MAE", "MSE"],
    "eligibility_baseline": "persistence (record naive, same rows)",
    "rule": "finite model_MAE < naive_MAE, strictly, on every consumed horizon of every family",
    "families": list(FAMILIES),
    "admissible_provenance": list(ADMISSIBLE_PROVENANCE),
    "requires": ["test_used false", "reserved_trading_test false", "evidence_sha256 verifies",
                 "declared evidence_sha256", "declared model_sha256", "asset", "period_hours",
                 "metric_space", "scaler_identity", "horizon rows == population rows",
                 "consumed-horizon mapping length == consumed prediction columns"],
    "refuses": ["tie", "zero naive", "missing metric", "non-finite metric", "unknown provenance",
                "reserved trading test", "metric switch", "tampered record", "auto-generated predictions",
                "API source (consumption not verifiable at entry)"],
    "seasonal_naive": "reported when present; never decides",
    "macro_mean": "reported; never overrides a failing member",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


CONTRACT_SHA256 = hashlib.sha256(_canonical(CONTRACT).encode()).hexdigest()


def evidence_digest(record: Mapping[str, Any]) -> str | None:
    """M04's verify(): sha256 of canonical JSON without ``evidence_sha256``."""
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


def paired(model: Any, baseline: Any) -> dict[str, Any]:
    """A metric beside its same-row baseline with delta/skill, or NOT_AVAILABLE and why."""
    if not (_finite(model) and _finite(baseline)):
        return {"model": model if _finite(model) else None,
                "baseline": baseline if _finite(baseline) else None,
                "delta": NOT_AVAILABLE, "skill": NOT_AVAILABLE,
                "skill_reason": "missing or non-finite model or baseline metric"}
    out = {"model": model, "baseline": baseline, "delta": model - baseline}
    if baseline == 0:
        out.update(skill=NOT_AVAILABLE, skill_reason="ZERO_NAIVE: zero baseline error, skill undefined")
    else:
        out["skill"] = 1.0 - model / baseline
    return out


def seasonal_row(entry: Mapping[str, Any], model_mae: Any) -> dict[str, Any]:
    """M04 b5ed0982: per_horizon[i].seasonal_naive is nested; report-only, never decides."""
    seasonal = entry.get("seasonal_naive")
    if not isinstance(seasonal, Mapping):
        return {"status": NOT_AVAILABLE, "reason": "the record carries no seasonal naive for this horizon"}
    if "naive_MAE" not in seasonal and "naive_MSE" not in seasonal:
        return {"status": NOT_AVAILABLE, "reason": seasonal.get("reason", "seasonal naive not available")}
    return {"mae": paired(model_mae, seasonal.get("naive_MAE")),
            "mse": paired(entry.get("model_MSE"), seasonal.get("naive_MSE")),
            "used_for_eligibility": False}


def _skip(reason: str, detail: Any = None, asset: Any = None) -> dict[str, Any]:
    return {"schema": DECISION_SCHEMA, "contract_sha256": CONTRACT_SHA256, "status": SKIPPED,
            "asset": asset, "provenance": {}, "population": {}, "baselines": {}, "horizons": [],
            "macro": {}, "failures": [{"reason": reason, "family": None, "horizon": None, "detail": detail}]}


def evaluate(evidence: Mapping[str, Any], declared: Mapping[str, Any] | None,
             consumed: Mapping[str, int], asset: str | None,
             families: tuple = FAMILIES) -> dict[str, Any]:
    """Decide from per-family records, the declaration and the consumed column counts.

    ``families`` names the prediction families this runner's strategy consumes
    (default: the backtest pipeline's hourly and daily). Every named family must
    pass; the rule itself (CONTRACT) is unchanged.
    """
    failures, horizons, provenance, population, macro, baselines = [], [], {}, {}, {}, {}

    def fail(reason, family=None, horizon=None, detail=None):
        failures.append({"reason": reason, "family": family, "horizon": horizon, "detail": detail})

    if not isinstance(declared, Mapping):
        return _skip("evidence_not_configured", "config.forecast_evidence is absent", asset)
    if not asset:
        fail("asset_not_declared", detail="config.asset is absent")
    elif declared.get("asset") != asset:
        fail("asset_mismatch", detail={"declared": declared.get("asset"), "run": asset})
    for family in families:
        spec = (declared.get("families") or {}).get(family)
        count = consumed.get(family)
        if not isinstance(spec, Mapping) or not isinstance(spec.get("horizons"), list):
            fail("consumption_not_declared", family, detail="no consumed-horizon mapping")
            continue
        if count is None or count != len(spec["horizons"]):
            fail("consumption_not_declared", family,
                 detail={"consumed_columns": count, "declared_horizons": len(spec["horizons"])})
            continue
        record = evidence.get(family)
        if not isinstance(record, Mapping):
            fail("evidence_missing", family)
            continue
        if record.get("schema") != EVIDENCE_SCHEMA:
            fail("evidence_schema_unsupported", family, detail=record.get("schema"))
            continue
        digest = evidence_digest(record)
        if digest is None or digest != record.get("evidence_sha256"):
            fail("evidence_digest_mismatch", family)
        if record.get("evidence_sha256") != spec.get("evidence_sha256"):
            fail("evidence_not_declared_record", family,
                 detail={"declared": spec.get("evidence_sha256"), "found": record.get("evidence_sha256")})
        if (record.get("frozen_metric") or {}).get("primary") != "MAE":
            fail("primary_metric_not_mae", family, detail=(record.get("frozen_metric") or {}).get("primary"))
        split = record.get("split") or {}
        provenance[family] = split.get("provenance")
        if (split.get("provenance") not in ADMISSIBLE_PROVENANCE or split.get("test_used") is not False
                or split.get("reserved_trading_test") is not False):
            fail("provenance_not_admissible", family, detail=split)
        artifact, pop, scale = record.get("artifact") or {}, record.get("population") or {}, record.get("scale") or {}
        if artifact.get("model_sha256") != spec.get("model_sha256"):
            fail("candidate_mismatch", family,
                 detail={"declared": spec.get("model_sha256"), "evidence": artifact.get("model_sha256")})
        if pop.get("asset") != declared.get("asset"):
            fail("asset_mismatch", family, detail={"declared": declared.get("asset"), "evidence": pop.get("asset")})
        if pop.get("sample_hours") != spec.get("period_hours"):
            fail("period_mismatch", family,
                 detail={"declared": spec.get("period_hours"), "evidence": pop.get("sample_hours")})
        if (scale.get("metric_space") != spec.get("metric_space")
                or scale.get("scaler_identity") != spec.get("scaler_identity")):
            fail("scaler_mismatch", family)
        population[family] = {k: pop.get(k) for k in ("dataset_id", "asset", "rows", "row_ids_sha256",
                                                      "first_origin", "last_origin", "sample_hours",
                                                      "horizon_unit")}
        population[family].update(evidence_sha256=record.get("evidence_sha256"),
                                  model_sha256=artifact.get("model_sha256"),
                                  metric_space=scale.get("metric_space"),
                                  scaler_identity=scale.get("scaler_identity"))
        seasonal = record.get("seasonal_naive")
        baselines[family] = {
            "eligibility_baseline": "persistence",
            "persistence": {"definition": (record.get("naive") or {}).get("definition")},
            "seasonal_naive": dict(seasonal) if isinstance(seasonal, Mapping) else {
                "status": NOT_AVAILABLE, "reason": "the record declares no seasonal_naive; never decides"}}
        entries = {e.get("horizon"): e for e in record.get("per_horizon") or [] if isinstance(e, Mapping)}
        sums = [0.0, 0.0, 0]
        for position, horizon in enumerate(spec["horizons"], start=1):
            entry = entries.get(horizon)
            row = {"family": family, "consumed_column": position, "horizon": horizon,
                   "horizon_unit": pop.get("horizon_unit"), "rows": None}
            if entry is None:
                row.update(mae=paired(None, None), mse=paired(None, None),
                           seasonal_naive={"status": NOT_AVAILABLE, "reason": "horizon absent"})
                horizons.append(row)
                fail("missing_metric", family, horizon, "consumed horizon absent from the record")
                continue
            model_mae, naive_mae = entry.get("model_MAE"), entry.get("naive_MAE")
            row.update(rows=entry.get("rows"), mae=paired(model_mae, naive_mae),
                       mse=paired(entry.get("model_MSE"), entry.get("naive_MSE")))
            row["seasonal_naive"] = seasonal_row(entry, model_mae)
            horizons.append(row)
            if entry.get("rows") != pop.get("rows"):
                fail("rows_mismatch", family, horizon,
                     {"horizon_rows": entry.get("rows"), "population_rows": pop.get("rows")})
            if model_mae is None or naive_mae is None:
                fail("missing_metric", family, horizon)
                continue
            if not (_finite(model_mae) and _finite(naive_mae)):
                fail("non_finite_metric", family, horizon)
                continue
            sums = [sums[0] + model_mae, sums[1] + naive_mae, sums[2] + 1]
            if naive_mae == 0:
                fail("naive_zero", family, horizon, "strict improvement over a zero naive error is impossible")
            elif model_mae == naive_mae:
                fail("tie_with_naive", family, horizon)
            elif not model_mae < naive_mae:
                fail("not_better_than_naive", family, horizon, {"model_MAE": model_mae, "naive_MAE": naive_mae})
        if sums[2]:
            macro[family] = {"mae": {"model": sums[0] / sums[2], "baseline": sums[1] / sums[2],
                                     "horizons": sums[2], "note": "reported; never overrides a failing member"}}
    decision = {"schema": DECISION_SCHEMA, "contract_sha256": CONTRACT_SHA256,
                "status": ELIGIBLE if not failures else SKIPPED, "asset": asset,
                "provenance": provenance, "population": population, "baselines": baselines,
                "horizons": horizons, "macro": macro, "failures": failures}
    return json.loads(_canonical(decision))


def gate_run(config: Mapping[str, Any], consumed: Mapping[str, int] | None) -> dict[str, Any]:
    """Load the declared records for a backtest run and decide; every fault is a SKIP."""
    asset = config.get("asset")
    if str(config.get("prediction_source", "CSV")).upper() == "API":
        return _skip("api_source_consumption_not_verifiable",
                     "per-tick API predictions cannot be bound to declared horizons before the run", asset)
    if consumed is None:
        return _skip("auto_generated_predictions_not_admissible",
                     "no prediction files: the plugin would synthesize predictions from the base data", asset)
    declared = config.get("forecast_evidence")
    if isinstance(declared, str):
        try:
            declared = json.loads(Path(declared).expanduser().read_text())
        except (OSError, ValueError):
            return _skip("evidence_unreadable", "forecast_evidence path", asset)
    evidence, load_failures = {}, []
    if isinstance(declared, Mapping):
        for family in FAMILIES:
            spec = (declared.get("families") or {}).get(family) or {}
            path = spec.get("file")
            try:
                record = json.loads(Path(path).expanduser().read_text()) if path else None
            except (OSError, ValueError):
                record = None
            if not isinstance(record, dict):
                load_failures.append({"reason": "evidence_unreadable" if path else "evidence_file_not_declared",
                                      "family": family, "horizon": None, "detail": path})
            else:
                evidence[family] = record
    decision = evaluate(evidence, declared, consumed, asset)
    if load_failures:
        decision["failures"] = load_failures + decision["failures"]
        decision["status"] = SKIPPED
    return decision
