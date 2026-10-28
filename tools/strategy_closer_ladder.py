"""The closer at quality < 1 against the INFORMATION-LIMITED ORACLE (QUALIFICATION_SPEC D7, sealed before running).

For each cell, closer quality q in {1.00, 0.85, 0.70} and head-error autocorrelation phi in a panel: the strategy's own
run (opener and closer) and the same opener with the Bayes closer (a 2-state Kalman filter per position over the same
noisy head stream: A random walk, AR(1) error; close iff E[A | history] >= -EPS_CLOSE). Per side: E3 tick agreement,
E4 destructive-close share, mean regret / initial cash risk at SL, early-close value vs hold-to-SL, plus Sharpe, net,
trades, mean holding bars and the exit mix (the q0.85 Sharpe anomaly).

  python tools/strategy_closer_ladder.py --cells <QUEUE.json> --ids a:0,b:0 --params <json> --admitted <d> --csv-dir <d> \
      --qualities 1.0,0.85,0.70 --phis 0,0.587,0.98 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import strategy_params as SP  # noqa: E402
from app.short_horizon.expert_guided import strategy_zz as ZZ  # noqa: E402
from tools import strategy_behaviour_matrix as BM  # noqa: E402
from tools import strategy_output_simulation as S  # noqa: E402


class PolicyFilterTrue(ZZ.PolicyC3Z):
    """The strategy with its closer filter fed the simulator's TRUE model (a measurement variant: the strategy-vs-Bayes
    gap that the estimation error causes is the estimated-filter row minus this one)."""
    filter_source = "truth"


def measure(w, strat, view, cells, Pol, closer_exact) -> dict:
    roll = BM.rollout_policy(w, Pol, strat, view, cells)
    out = BM.closer_d7_stats(w, roll, closer_exact)
    out["_trades_key"] = [(int(t["entry_bar_abs"]), int(t["exit_bar_abs"]), int(t["direction"]), round(float(t["net_usd"]), 6)) for t in roll["trades"]]
    return out


def filter_models(closer) -> dict:
    """Per side: the filter model the strategy estimates on the calibration block and the true one."""
    out = {}
    for si, side in ((0, "BUY"), (1, "SELL")):
        est = ZZ.CloserFilterModel.estimate(closer.calibration_pairs(si))
        tru = ZZ.CloserFilterModel.truth(closer.filter_truth(si))
        e = est.summary() if est is not None else None
        t = tru.summary()
        if e is not None:
            t["obs_at_h"] = [{"h": o["h"], **dict(zip(("alpha", "beta", "sd"), tru.obs(o["h"])))} for o in e["obs_at_h"]]
        out[side] = {"estimated": e, "true": t}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--qualities", default="1.0,0.85,0.70")
    ap.add_argument("--phis", default="0,0.587,0.98")
    ap.add_argument("--rules", default="1.0:1:1", help="z_close:k_persist:closer_filter of the strategy's close rule, comma separated")
    ap.add_argument("--filter-truth", action="store_true", help="also run the first rule with the filter fed the TRUE model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    AC = S.contract_for_sim(str(S.CONTRACT))
    cx, SCFG = S.load_fold("wf2022", a.admitted, a.csv_dir)
    SCFG = S.with_variant(SCFG, C3.FULL_F3)
    params = SP.validate({**SP.DEFAULTS, **json.loads(Path(a.params).read_text())})   # a point from before the z_close / k_persist genes carries their defaults
    ZZ.configure(params)
    genes = SP.genes_c3(params)
    cells = {c["id"]: c for c in json.loads(Path(a.cells).read_text())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.ids.split(","):
        cid, seed = spec.rsplit(":", 1)
        t0 = time.time()
        w = BM.cell_world(cx, SCFG, AC, cells[cid], int(seed))
        strat, view0, cl = BM.strategy_for(w, genes, AC)
        exact = view0["closer"].with_quality(1.0)
        rec = {"cell": cid, "seed": int(seed), "params_sha256": SP.sha256_of(a.params), "rows": []}
        for phi in [float(x) for x in a.phis.split(",")]:
            S.CLOSER_NOISE_PHI = phi
            for q in [float(x) for x in a.qualities.split(",")]:
                if q >= 0.999 and phi != float(a.phis.split(",")[0]):
                    continue                                          # quality 1 carries no noise: once
                view = dict(view0)
                view["closer"] = view0["closer"].with_quality(q)
                row = {"quality": q, "phi": phi, "bayes": measure(w, strat, view, cl, BM.PolicyBayes, exact), "rules": {}}
                if q < 0.999:
                    row["filter_models"] = filter_models(view["closer"])
                for rule in a.rules.split(","):
                    z, k, f = (rule.split(":") + ["1"])[:3]
                    ZZ.PARAMS.update(z_close=float(z), k_persist=int(k), closer_filter=float(f))
                    row["rules"][rule] = measure(w, strat, view, cl, ZZ.PolicyC3Z, exact)
                    if a.filter_truth and rule == a.rules.split(",")[0] and q < 0.999:
                        row["rules"][rule + ":truth"] = measure(w, strat, view, cl, PolicyFilterTrue, exact)
                ZZ.PARAMS.update(z_close=float(params["z_close"]), k_persist=int(params["k_persist"]), closer_filter=float(params["closer_filter"]))
                row["strategy"] = row["rules"][a.rules.split(",")[0]]
                if q >= 0.999:
                    row["q1_bayes_equals_strategy"] = row["strategy"]["_trades_key"] == row["bayes"]["_trades_key"]
                for v in [row["bayes"]] + list(row["rules"].values()):
                    v.pop("_trades_key", None)
                rec["rows"].append(row)
                print(json.dumps({"cell": spec, "q": q, "phi": phi, "strategy": {k: row["strategy"][k] for k in ("sharpe", "net_usd", "trades", "mean_hold_bars")},
                                  "E4": [row["strategy"][s]["E4"] for s in ("BUY", "SELL")], "E4_bayes": [row["bayes"][s]["E4"] for s in ("BUY", "SELL")]}, default=float), flush=True)
        rec["sec"] = round(time.time() - t0, 1)
        (out / f"LADDER_{cid}_s{seed}.json").write_text(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    main()
