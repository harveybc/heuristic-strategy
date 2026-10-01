#!/usr/bin/env python3
"""Run the forecasting+heuristic arm on lane G's ETH 4h validation episode.

    python run_paired_eth4h.py --bars VIEW.csv --rows 13699:15895 \
        --predictions M07_validation_predictions.csv --evidence M07_evidence.json \
        --declared declared.json --out result.json [--manifest-status FROZEN]

``--bars`` is the model-ready view (DATE_TIME, OPEN, HIGH, LOW, CLOSE, ...), sliced to
the declared validation rows. ``--predictions`` has one row per forecast origin
(DATE_TIME plus one column per horizon, in price units), issued from data through
that bar's close. ``--declared`` is the consumer declaration that the
forecast-versus-naive gate checks. Only horizons that pass the gate are read
(see app/paired_backtest.py). Nothing here touches a broker.
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

from app.paired_backtest import HeuristicParams, paired_backtest  # noqa: E402


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bars(path, rows):
    start, stop = (int(x) for x in rows.split(":"))
    with open(path, newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    out = []
    for row in all_rows[start:stop]:
        out.append({"time": row["DATE_TIME"], "open": float(row["OPEN"]), "high": float(row["HIGH"]),
                    "low": float(row["LOW"]), "close": float(row["CLOSE"])})
    if len(out) != stop - start:
        raise ValueError("the view does not contain the declared rows")
    return out


def load_predictions(path, horizons):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        out = []
        for row in reader:
            item = {"time": row["DATE_TIME"]}
            for h in horizons:
                key = f"h_{h}" if f"h_{h}" in row else f"Prediction_{h}"
                item[h] = float(row[key]) if row.get(key) not in (None, "") else float("nan")
            out.append(item)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--bars", "--rows", "--predictions", "--evidence", "--declared", "--out"):
        parser.add_argument(name, required=True)
    parser.add_argument("--family", default="forecast")
    parser.add_argument("--manifest-status", default="UNKNOWN")
    args = parser.parse_args(argv)
    record = json.loads(Path(args.evidence).read_text())
    declared = json.loads(Path(args.declared).read_text())
    horizons = declared["families"][args.family]["horizons"]
    bars = load_bars(args.bars, args.rows)
    result = paired_backtest(bars, load_predictions(args.predictions, horizons), record, declared,
                             HeuristicParams(), family=args.family, manifest_status=args.manifest_status)
    result["inputs"] = {"bars_sha256": _sha(args.bars), "rows": args.rows,
                        "predictions_sha256": _sha(args.predictions), "evidence_file_sha256": _sha(args.evidence),
                        "declared_sha256": _sha(args.declared)}
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "metrics": result["metrics"],
                      "consumed": result["naive_gate"]["consumed_horizons"],
                      "excluded": [r["horizon"] for r in result["naive_gate"]["excluded_horizons"]]},
                     sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
