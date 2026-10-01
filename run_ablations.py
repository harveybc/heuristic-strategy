#!/usr/bin/env python3
"""Direction and per-horizon ablations of the heuristic arm; only for a fully eligible candidate.

Runs, on the same episode: both / long_only / short_only over all declared horizons, and
"both" over each declared horizon alone. Each run is a separately declared experiment.
A candidate that fails the naive gate on ANY declared horizon is not ablated: the
reduced-input runs live in the main harness. (ETH R0: no config is eligible.)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.forecast_naive_gate import ELIGIBLE, evaluate  # noqa: E402
from app.paired_backtest import HeuristicParams, paired_backtest  # noqa: E402


def ablations(bars, predictions, record, declared, params: HeuristicParams = HeuristicParams(), *,
              family: str = "forecast", manifest_status: str = "UNKNOWN", missing_forecast: str = "refuse"):
    horizons = list(declared["families"][family]["horizons"])
    full = evaluate({family: record}, declared, {family: len(horizons)}, declared.get("asset"), families=(family,))
    if full["status"] != ELIGIBLE:
        return {"status": "NOT_ELIGIBLE_NO_ABLATION", "failures": full["failures"], "runs": []}
    runs = []
    for direction in ("both", "long_only", "short_only"):
        runs.append({"name": f"{direction}/all", "result": paired_backtest(
            bars, predictions, record, declared, replace(params, direction=direction), family=family,
            manifest_status=manifest_status, missing_forecast=missing_forecast)})
    for h in horizons:
        one = dict(declared, families={family: dict(declared["families"][family], horizons=[h])})
        runs.append({"name": f"both/h{h}", "result": paired_backtest(
            bars, predictions, record, one, replace(params, direction="both"), family=family,
            manifest_status=manifest_status, missing_forecast=missing_forecast)})
    return {"status": "ABLATIONS_RAN", "declared": "each run is a separately declared ablation", "runs": runs}


def main(argv=None):
    from run_paired_eth4h import load_bars, load_predictions
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--bars", "--rows", "--predictions", "--evidence", "--declared", "--out"):
        parser.add_argument(name, required=True)
    parser.add_argument("--manifest-status", default="UNKNOWN")
    parser.add_argument("--missing-forecast", default="refuse", choices=["refuse", "hold"])
    args = parser.parse_args(argv)
    declared = json.loads(Path(args.declared).read_text())
    horizons = declared["families"]["forecast"]["horizons"]
    out = ablations(load_bars(args.bars, args.rows), load_predictions(args.predictions, horizons),
                    json.loads(Path(args.evidence).read_text()), declared,
                    manifest_status=args.manifest_status, missing_forecast=args.missing_forecast)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"status": out["status"], "runs": [r["name"] for r in out["runs"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
