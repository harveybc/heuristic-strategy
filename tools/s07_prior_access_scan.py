#!/usr/bin/env python
"""S07 (2026-09-30): prior-access scan of reserved rows (>= 2019-05-16 00:00) in committed and retained artifacts.

Streams the first field of each CSV (no DataFrame, no price used), counts rows on or after
the cut, records the file sha256, and reads retained sweep manifests for origin_last / n_origins.
This is access accounting, not a market result. Usage:
  python tools/s07_prior_access_scan.py --out <dir> [--root <checkout>] [--retained-root <dir with run_out>]
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import time
from pathlib import Path

CUT = "2019-05-16 00:00:00"

CANDIDATES = [
    "tests/data/eurusd_hour_2005_2020.csv",
    "tests/data/phase_1c_direction_test_ohlc.csv",
    "tests/data/phase_1_base_d3.csv",
    "tests/data/phase_2_3_base_d3.csv",
    "tests/data/phase_2_3_base_d3_last_year.csv",
    "tests/data/ideal_predictions_hourly_d3.csv",
    "tests/data/ideal_predictions_daily_d3.csv",
    "tests/data/ann_predictions_hourly_d3.csv",
    "tests/data/ann_predictions_daily_d3.csv",
    "tests/data/lstm_predictions_hourly_d3.csv",
    "tests/data/phase_2_3_cnn_1h_prediction_d3.csv",
    "tests/data/phase_2_3_cnn_1d_prediction_d3.csv",
    "cnn_predictions_15yr.csv",
    "ensemble_predictions_15yr.csv",
    "oracle_ceiling_trades.csv",
    "phase_b_cnn_trades.csv",
    "phase_c_ensemble_trades.csv",
    "phase_d_neat_trades.csv",
    "trades.csv",
    "wfo_oos_trades.csv",
    "wfo_oos_trades_2017_2019.csv",
    "wfo_oos_trades_merged.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_first_field(path: Path) -> dict:
    """Count data rows whose first field (a timestamp string) is on or after the cut."""
    n = 0
    n_after = 0
    first = last_after = None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            n += 1
            stamp = row[0].strip()
            if first is None:
                first = stamp
            if len(stamp) >= 10 and stamp[:4].isdigit() and stamp >= CUT[: len(stamp)]:
                n_after += 1
                last_after = stamp
    return {"header_first_field": header[0] if header else None, "data_rows": n,
            "rows_first_field_on_or_after_cut": n_after, "first_row": first, "last_row_on_or_after_cut": last_after}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--retained-root", default=None, help="checkout holding untracked run_out/ (read-only)")
    args = parser.parse_args(argv)
    t0 = time.time()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"tool": "tools/s07_prior_access_scan.py", "cut": CUT, "root_files": {}, "retained_manifests": {},
              "wfo_results_2019_fold": {}, "year_loop_results_2019": {}}
    for rel in CANDIDATES:
        p = root / rel
        if not p.exists():
            report["root_files"][rel] = {"status": "ABSENT"}
            continue
        rec = scan_first_field(p)
        rec["sha256"] = sha256(p)
        rec["verdict"] = "CONTAINS_RESERVED_ROWS" if rec["rows_first_field_on_or_after_cut"] else "NO_RESERVED_ROWS"
        report["root_files"][rel] = rec
    for name in ("wfo_results.json", "wfo_results_2017_2019.json"):
        p = root / name
        if p.exists():
            try:
                data = json.loads(p.read_text())
                folds = data if isinstance(data, list) else data.get("fold_results", data.get("folds", []))
                hit = [f for f in folds if isinstance(f, dict) and f.get("test_year") == 2019]
                report["wfo_results_2019_fold"][name] = {"sha256": sha256(p), "n_2019_folds": len(hit),
                                                         "test_bars": [f.get("test_bars") for f in hit]}
            except Exception as exc:  # noqa: BLE001
                report["wfo_results_2019_fold"][name] = {"error": repr(exc)}
    for name in ("oracle_ceiling_results.json", "phase_b_cnn_results.json", "phase_c_ensemble_results.json", "phase_d_neat_results.json"):
        p = root / name
        if p.exists():
            txt = p.read_text()
            report["year_loop_results_2019"][name] = {"sha256": sha256(p), "mentions_year_2019": '"year": 2019' in txt}
    if args.retained_root:
        for m in sorted(glob.glob(os.path.join(args.retained_root, "run_out", "*", "manifest.json"))):
            try:
                man = json.loads(Path(m).read_text())
            except Exception as exc:  # noqa: BLE001
                report["retained_manifests"][m] = {"error": repr(exc)}
                continue
            rec = {"sha256": sha256(Path(m)), "n_origins": man.get("n_origins"), "origin_first": man.get("origin_first"),
                   "origin_last": man.get("origin_last"), "n_bars": man.get("n_bars"),
                   "exit_variant": (man.get("execution") or {}).get("exit_variant"),
                   "input_sha256": (man.get("files") or {}).get("input.csv") or man.get("source_sha256")}
            origins = Path(m).with_name("origins.csv")
            if origins.exists():
                rec["origins_csv"] = scan_first_field(origins)
                rec["origins_csv"]["sha256"] = sha256(origins)
            trades = glob.glob(os.path.join(os.path.dirname(m), "**", "trades.csv"), recursive=True)
            if trades:
                n_with = 0
                for tpath in trades:
                    with open(tpath) as f:
                        next(f, None)
                        if any(line[:19] >= CUT for line in f if line[:4].isdigit()):
                            n_with += 1
                rec["cell_trades_files"] = len(trades)
                rec["cell_trades_files_with_reserved_timestamps"] = n_with
            rec["verdict"] = ("RESERVED_ORIGINS_SCORED" if (rec.get("origin_last") or "") >= CUT else "NO_RESERVED_ORIGIN")
            report["retained_manifests"][os.path.relpath(m, args.retained_root)] = rec
    report["wall_seconds"] = round(time.time() - t0, 3)
    (out / "prior_access_scan.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    n_contain = sum(1 for r in report["root_files"].values() if r.get("verdict") == "CONTAINS_RESERVED_ROWS")
    print(f"files containing reserved rows: {n_contain}; retained manifests: {len(report['retained_manifests'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
