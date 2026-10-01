"""Entry and per-candidate application of the forecast-versus-naive gate (S09).

Every runner that calls ``plugin.evaluate_candidate`` itself (outside
``run_processing_pipeline``) builds a ``RunnerGate`` before loading data or models.

``entry()`` decides on the run-level declared evidence. ``candidate()`` decides
for each evaluated prediction set: its consumed columns must bind to the
declaration, and a candidate may carry its own held-out/OOF evidence.  A skipped
candidate is recorded with its failures and its MAE beside the same-row naive.
Nothing is dropped silently: the receipt counts evaluated and skipped candidates.

The rule and its digest live in ``app.forecast_naive_gate`` (CONTRACT, unchanged).
A plugin that declares ``consumes_learned_predictions = False`` is recorded as
NOT_APPLICABLE_NO_LEARNED_PREDICTIONS; an undeclared plugin is treated as
consuming predictions (fail closed).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from app.forecast_naive_gate import CONTRACT_SHA256, ELIGIBLE, SKIPPED, evaluate, gate_run

RECEIPT_SCHEMA = "heuristic_strategy.runner_naive_gate_receipt.v1"
NOT_APPLICABLE = "NOT_APPLICABLE_NO_LEARNED_PREDICTIONS"


class GateRefused(RuntimeError):
    """A runner or library entry was reached without an allowing gate."""


class CandidateSkipped(GateRefused):
    """One prediction set was not eligible; it was not evaluated."""


def plugin_consumes_learned_predictions(plugin: Any) -> bool:
    return getattr(plugin, "consumes_learned_predictions", True) is not False


def _load_declared(path: str | Path | None) -> Mapping[str, Any] | None:
    if not path:
        return None
    try:
        value = json.loads(Path(path).expanduser().read_text())
    except (OSError, ValueError):
        return {"_unreadable": str(path)}
    return value if isinstance(value, Mapping) else {"_unreadable": str(path)}


class RunnerGate:
    def __init__(self, runner: str, evidence_config: str | Path | None = None,
                 receipt_path: str | Path | None = None, *, consumes_predictions: bool = True,
                 asset: str | None = None, prediction_source: str = "OFFLINE",
                 kind: str = "forecast") -> None:
        if kind not in ("forecast", "direction"):
            raise ValueError("kind must be forecast or direction")
        self.kind = kind
        self.runner = runner
        self.evidence_config = evidence_config
        self.receipt_path = Path(receipt_path) if receipt_path else None
        self.consumes_predictions = consumes_predictions
        self.asset = asset
        self.prediction_source = prediction_source
        self.entry_decision: dict[str, Any] | None = None
        self.candidates: list[dict[str, Any]] = []
        self.allowed = False

    def _config(self, declared):
        asset = self.asset or (declared or {}).get("asset")
        return {"asset": asset, "prediction_source": self.prediction_source, "forecast_evidence": declared}

    def _decide(self, declared, consumed):
        if isinstance(declared, Mapping) and "_unreadable" in declared:
            from app.forecast_naive_gate import _skip
            return _skip("evidence_unreadable", declared["_unreadable"], self.asset)
        if self.kind == "direction":
            return self._decide_direction(declared, consumed or {})
        return gate_run(self._config(declared), consumed)

    def _decide_direction(self, declared, consumed):
        """predictor.direction_naive_evidence.v1 against both classification baselines."""
        from app.direction_naive_gate import evaluate_direction
        records = {}
        for family, spec in ((declared or {}).get("families") or {}).items():
            try:
                records[family] = json.loads(Path(spec.get("file", "")).expanduser().read_text())
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        asset = self.asset or (declared or {}).get("asset")
        return evaluate_direction(records, declared, consumed, asset)

    def entry(self) -> bool:
        if not self.consumes_predictions:
            self.entry_decision = {"status": NOT_APPLICABLE, "contract_sha256": CONTRACT_SHA256,
                                   "reason": "the plugin declares consumes_learned_predictions = False"}
            self.allowed = True
        else:
            declared = _load_declared(self.evidence_config)
            consumed = None
            if isinstance(declared, Mapping) and "_unreadable" not in declared:
                consumed = {family: len(spec.get("horizons") or [])
                            for family, spec in (declared.get("families") or {}).items()
                            if isinstance(spec, Mapping)}
            self.entry_decision = self._decide(declared, consumed if declared else {})
            self.allowed = self.entry_decision["status"] == ELIGIBLE
        self.write()
        return self.allowed

    def candidate(self, label: str, consumed: Mapping[str, int] | None, *,
                  evidence_config: str | Path | None = None) -> bool:
        """Decide one prediction set before any strategy evaluation of it."""
        if self.entry_decision is None:
            raise GateRefused("candidate() before entry()")
        if not self.allowed:
            row = {"label": label, "status": SKIPPED, "failures": [
                {"reason": "entry_not_eligible", "family": None, "horizon": None,
                 "detail": self.entry_decision.get("status")}], "horizons": []}
        elif self.entry_decision["status"] == NOT_APPLICABLE:
            row = {"label": label, "status": NOT_APPLICABLE, "failures": [], "horizons": []}
        else:
            declared = _load_declared(evidence_config or self.evidence_config)
            if consumed is None and self.kind == "forecast":
                decision = {"status": SKIPPED, "horizons": [], "failures": [{
                    "reason": "consumption_not_declared", "family": None, "horizon": None,
                    "detail": "this prediction set is not an hourly/daily price-forecast family of the contract"}]}
            else:
                decision = self._decide(declared, dict(consumed))
            row = {"label": label, "status": decision["status"], "failures": decision["failures"],
                   "horizons": decision.get("horizons", []), "provenance": decision.get("provenance", {})}
        self.candidates.append(row)
        self.write()
        return row["status"] in (ELIGIBLE, NOT_APPLICABLE)

    def require(self, label: str, consumed: Mapping[str, int] | None, **kw) -> None:
        if not self.candidate(label, consumed, **kw):
            raise CandidateSkipped(f"{SKIPPED}: candidate {label!r} not evaluated")

    def receipt(self) -> dict[str, Any]:
        evaluated = sum(1 for c in self.candidates if c["status"] in (ELIGIBLE, NOT_APPLICABLE))
        return {"schema": RECEIPT_SCHEMA, "runner": self.runner, "contract_sha256": CONTRACT_SHA256,
                "entry": self.entry_decision, "candidates": self.candidates,
                "counts": {"evaluated": evaluated, "skipped": len(self.candidates) - evaluated}}

    def write(self) -> None:
        if self.receipt_path:
            self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
            self.receipt_path.write_text(json.dumps(self.receipt(), indent=2, sort_keys=True))


_INSTALLED: RunnerGate | None = None


def install(gate: RunnerGate | None) -> None:
    """Make a runner's gate visible to its module-level backtest helpers."""
    global _INSTALLED
    _INSTALLED = gate


def require_installed(label: str, consumed: Mapping[str, int] | None) -> None:
    """Called immediately before a direct evaluate_candidate; no gate means no evaluation."""
    if _INSTALLED is None:
        raise GateRefused(f"{SKIPPED}: no naive gate installed for candidate {label!r}")
    _INSTALLED.require(label, consumed)


def entry_gate_for_runner(runner: str, argv=None, *, consumes_predictions: bool = True,
                          asset: str = "EURUSD", prediction_source: str = "OFFLINE",
                          default_receipt: str | None = None, kind: str = "forecast") -> RunnerGate:
    """Parse --forecast_evidence / --naive_gate_receipt (unknown args ignored) and decide entry."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--forecast_evidence")
    parser.add_argument("--naive_gate_receipt", default=default_receipt or f"naive_gate_receipt_{runner}.json")
    known, _ = parser.parse_known_args(argv)
    gate = RunnerGate(runner, evidence_config=known.forecast_evidence, receipt_path=known.naive_gate_receipt,
                      consumes_predictions=consumes_predictions, asset=asset,
                      prediction_source=prediction_source, kind=kind)
    gate.entry()
    install(gate)
    if not gate.allowed:
        print(f"{SKIPPED}: {runner} evaluates nothing "
              f"({len(gate.entry_decision['failures'])} failure(s); receipt {known.naive_gate_receipt})")
    return gate
