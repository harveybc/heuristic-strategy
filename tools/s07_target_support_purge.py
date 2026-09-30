#!/usr/bin/env python
"""S07 (2026-09-30): target-support and trade-support purge table against the reserved cut.

Read-only over committed CSVs. Computes no PnL, no metric, no ranking. Writes:
  <out>/purge_table.csv           one row per development origin per real arm
  <out>/purge_summary.json        counts, per-variant exit/censoring, moved-boundary proposal, file shas
  <out>/population_<label>.csv    target support for an origin population given with --population
Reserved rows are dropped at load and never indexed (app.target_support).

Usage (from the heuristic-strategy checkout root, PYTHONPATH=.):
  python tools/s07_target_support_purge.py --out <dir> [--population LABEL=origins.csv] [--arms R1,R2,R3]
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.target_support import (  # noqa: E402
    RESERVED_START,
    derive_arm_support,
    derive_population_target_support,
    dump_json,
    load_development_bars,
    load_prediction_family,
    sha256_file,
)

DATA = Path("tests/data")
ARMS = {
    # label: (short family file, long family file, base OHLC file)  -- design 4bb763d section 2
    "R1": (DATA / "ann_predictions_hourly_d3.csv", DATA / "ann_predictions_daily_d3.csv", DATA / "phase_1_base_d3.csv"),
    "R2": (DATA / "lstm_predictions_hourly_d3.csv", DATA / "ann_predictions_daily_d3.csv", DATA / "phase_1_base_d3.csv"),
    "R3": (DATA / "phase_2_3_cnn_1h_prediction_d3.csv", DATA / "phase_2_3_cnn_1d_prediction_d3.csv", DATA / "phase_2_3_base_d3.csv"),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default="R1,R2,R3")
    parser.add_argument("--population", action="append", default=[], help="LABEL=path/to/origins.csv (DATE_TIME column)")
    parser.add_argument("--population-base", default=str(DATA / "phase_2_3_base_d3.csv"))
    parser.add_argument("--reserved-start", default=str(RESERVED_START))
    args = parser.parse_args(argv)

    t0 = time.time()
    cut = pd.Timestamp(args.reserved_start)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "tool": "tools/s07_target_support_purge.py",
        "reserved_start": str(cut),
        "pnl_computed": False,
        "arms": {},
        "populations": {},
        "files": {},
    }
    tables = []
    bars_cache: dict[str, object] = {}
    for label in [a for a in args.arms.split(",") if a]:
        short_p, long_p, base_p = ARMS[label]
        if str(base_p) not in bars_cache:
            bars_cache[str(base_p)] = load_development_bars(base_p, reserved_start=cut)
        bars = bars_cache[str(base_p)]
        short = load_prediction_family(short_p, family="short", reserved_start=cut)
        long = load_prediction_family(long_p, family="long", reserved_start=cut)
        arm = derive_arm_support(label=label, bars=bars, short=short, long=long)
        summary["arms"][label] = arm.summary
        summary["files"][label] = arm.files
        tables.append(arm.table)
        print(f"[{label}] origins={arm.summary['n_development_origins_aligned']} purged={arm.summary['n_purged']} "
              f"kept={arm.summary['n_kept']} first_purged={arm.summary['first_purged_origin']}", flush=True)
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(out / "purge_table.csv", index=False)

    for spec in args.population:
        label, path = spec.split("=", 1)
        base = bars_cache.get(args.population_base) or load_development_bars(args.population_base, reserved_start=cut)
        origins = pd.read_csv(path)["DATE_TIME"]
        table, psum = derive_population_target_support(origins, base, label=label)
        psum["origins_file"] = {"path": path, "sha256": sha256_file(path)}
        psum["base_file"] = {"path": base.path, "sha256": base.sha256}
        summary["populations"][label] = psum
        table.to_csv(out / f"population_{label}.csv", index=False)
        print(f"[{label}] origins={psum['n_origins']} reserved_in_population={psum['n_origins_reserved_already_in_population']} "
              f"dev_purged_elapsed={psum['n_dev_purged_elapsed_target']} kept={psum['n_kept']}", flush=True)

    ru = resource.getrusage(resource.RUSAGE_SELF)
    summary["resources"] = {
        "wall_seconds": round(time.time() - t0, 3),
        "cpu_user_seconds": round(ru.ru_utime, 3),
        "cpu_sys_seconds": round(ru.ru_stime, 3),
        "max_rss_mib": round(ru.ru_maxrss / 1024, 1),
        "host_role": os.environ.get("S07_HOST_ROLE", "unspecified"),
    }
    dump_json(out / "purge_summary.json", summary)
    print(f"resources: {summary['resources']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
