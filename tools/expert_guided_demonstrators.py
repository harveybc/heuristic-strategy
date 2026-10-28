#!/usr/bin/env python3
"""Admission check for ZigZag demonstrators (owner 2026-10-22: several ZigZag signals as training-only
demonstrations). A demonstrator is admitted only if its OWN hindsight replay, executed by the native simulator with
the lane's business rules (expert config: SL/TP set on every order, sizing, costs, weekly forced flat, entry limits),
satisfies the activity rule there. Its hindsight profit is reported but cannot discriminate (a hindsight oracle is
always hugely profitable); the causal-confirmed counterpart (acting only once the pivot is confirmed) is reported as
the imitability information. Candidates vary the ZigZag DEPTH: the replicated MetaTrader Deviation test never acts
(a candidate is kept only when the bar IS the window extreme, so its distance to it is 0), so pivots depend on Depth
and Backstep only. Labels only, no model fitted, no validation period read. Writes DEMONSTRATORS.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import zigzag as Z  # noqa: E402
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from tools.expert_guided_e1t import EXPERT_CFG, STUDENT_CFG, Ctx, summary  # noqa: E402
from tools.expert_guided_e1t import MUSASHI_CFG  # noqa: E402
from tools.expert_guided_e3m import ExpertTrack, parse_track  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--specs", default="3,2,rev10", help="demonstrators: a MetaTrader ZigZag depth ('2') or a reversal threshold in pips ('rev10'); 3 = registered optimum")
    ap.add_argument("--exec-contract", choices=("native", "sac"), default="sac")
    ap.add_argument("--contract", default="consolidated")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = DG.load(a.admitted, a.csv_dir, scaling="fixed_train", contract=a.contract)
    cx = Ctx(ds)
    rows = []
    K = ("net_profit_usd", "sharpe_daily_sqrt365", "trades", "orders_per_week", "max_drawdown_pct", "win_rate", "valid")
    import dataclasses
    ECFG = dataclasses.replace(EXPERT_CFG, max_entries_per_day=MUSASHI_CFG.max_entries_per_day, sizing=MUSASHI_CFG.sizing) if a.exec_contract == "sac" else EXPERT_CFG
    SCFG = MUSASHI_CFG if a.exec_contract == "sac" else STUDENT_CFG
    for spec in a.specs.split(","):
        primary = spec.strip() == str(DG.EXPERT.depth)
        t = ExpertTrack(cx) if primary else parse_track(cx, spec)
        m = summary(cx.rollout("TRAIN", t.policy(cx), ECFG)[0])
        mc = summary(cx.rollout("TRAIN", t.confirmed_policy(), SCFG)[0])
        admitted = bool(m["valid"])
        rows.append({"name": t.name, "spec": spec.strip(), "primary": primary, "pivots": t.n_pivots, "exec_contract": a.exec_contract,
                     "hindsight_train": {k: m[k] for k in K if k in m}, "confirmed_train": {k: mc[k] for k in K if k in mc},
                     "admitted": admitted, "admission": "hindsight replay valid under the business activity rules"})
        print(json.dumps(rows[-1]), flush=True)
    doc = {"rule": "admitted iff the demonstrator's own hindsight TRAIN replay is valid under the business rules; hindsight profit is not a criterion; confirmed replay = imitability information",
           "contract": a.contract, "period": ds.meta["periods"]["TRAIN"], "demonstrators": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
