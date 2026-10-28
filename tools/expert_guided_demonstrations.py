#!/usr/bin/env python3
"""Seal the ZigZag v2 expert's demonstrations recomputed on the ADMITTED clock (audit F08: a source/clock change
is a target identity change). Same parameters, executable path, profile, sizing, rules and cap as demonstrations
v2 (lake label clock, sha256 38125b0b...); only the bar source and the calendar (closure complement) differ.

  python tools/expert_guided_demonstrations.py --admitted <dir> --csv-dir <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import zigzag as Z  # noqa: E402
from app.short_horizon import zigzag_opt as ZO  # noqa: E402
from app.short_horizon.expert_guided import data as DG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = DG.load(a.admitted, a.csv_dir)
    ev = ZO.Evaluator(ds.bars, ds.cal)
    p = {"depth": DG.EXPERT.depth, "deviation_pips": DG.EXPERT.deviation_pips, "backstep": DG.EXPERT.backstep}
    trades, per = [], {}
    for period in ZO.PERIODS:
        m, res = ev.eval_A(p, period)
        per[period] = {k: m[k] for k in ("sharpe", "trades", "orders_per_week", "max_dd_pct", "valid")}
        trades += res.trades
    man = Z.write_demonstrations(Path(a.out), trades, DG.EXPERT, DG.PINS["admitted_sha256"],
                                 {"profile": "RealisticProfile, entry_delay_bars = 0 (executable path)", "sizing":
                                  "rel_volume 0.10, percentual_order_size 1.0, leverage 100", "sl": "2 x causal 24 h vol",
                                  "tp": "next hindsight pivot level -/+ 1 pip", "daily_cap": 4},
                                 extra={"version": "v2-admitted", "source": "admitted (UTC bar start, +1 h availability)",
                                        "calendar": "complement of CLOSURE_INTERVALS.json (71078e8f...), 12 h / 12 h",
                                        "sibling": "demonstrations v2 on the lake label clock, sha256 "
                                                   + DG.PINS["demonstrations_v2_lake_sha256"],
                                        "periods": {k: [v, ZO.PERIOD_STATUS[k]] for k, v in ZO.PERIODS.items()},
                                        "per_period": per})
    print(man["sha256"], man["rows"], per)


if __name__ == "__main__":
    main()
