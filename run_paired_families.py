#!/usr/bin/env python3
"""Run the heuristic strategy over its own hourly and daily families (M07 campaign-2 layout).

    python run_paired_families.py --bars VIEW.csv --rows START:STOP \
        --hourly-evidence EVIDENCE_<h>.json --hourly-predictions PREDICTIONS_<h>.csv \
        --daily-evidence EVIDENCE_<d>.json --daily-predictions PREDICTIONS_<d>.csv \
        --declared declared.json --out result.json [--manifest-status FROZEN_DEVELOPMENT] [--exit-variant E]

Each family is gated on its own record. Rows pair on DATE_TIME intersections. Prices are
read from the close_hat_h<hours> columns. Nothing here touches a broker.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.paired_backtest import HeuristicParams, paired_backtest_families  # noqa: E402
from run_paired_eth4h import load_bars  # noqa: E402


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_family(path, horizons):
    with open(path, newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            item = {h: float(row[f"close_hat_h{h}"]) for h in horizons if row.get(f"close_hat_h{h}") not in (None, "")}
            if len(item) != len(horizons):
                continue  # a row lacking any declared horizon is not a forecast for this family
            item["time"] = row["DATE_TIME"]
            rows.append(item)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--bars", "--rows", "--hourly-evidence", "--hourly-predictions", "--daily-evidence",
                 "--daily-predictions", "--declared", "--out"):
        parser.add_argument(name, required=True)
    parser.add_argument("--manifest-status", default="UNKNOWN")
    parser.add_argument("--exit-variant", default="G", choices=["G", "E"])
    args = parser.parse_args(argv)
    declared = json.loads(Path(args.declared).read_text())
    records = {"hourly": json.loads(Path(args.hourly_evidence).read_text()),
               "daily": json.loads(Path(args.daily_evidence).read_text())}
    predictions = {f: load_family(getattr(args, f"{f}_predictions"), declared["families"][f]["horizons"])
                   for f in ("hourly", "daily")}
    params = HeuristicParams(exit_variant="E (0.6 hourly + 0.4 daily vs stop)" if args.exit_variant == "E"
                             else HeuristicParams().exit_variant)
    result = paired_backtest_families(load_bars(args.bars, args.rows), predictions, records, declared, params,
                                      manifest_status=args.manifest_status)
    result["inputs"] = {"bars_sha256": _sha(args.bars), "rows": args.rows, "declared_sha256": _sha(args.declared),
                        **{f"{f}_{k}_sha256": _sha(getattr(args, f"{f}_{k}"))
                           for f in ("hourly", "daily") for k in ("evidence", "predictions")}}
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    gate = result["naive_gate"]["families"]
    print(json.dumps({"status": result["status"], "metrics": result["metrics"],
                      "consumed": {f: g["consumed_horizons"] for f, g in gate.items()}}, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
