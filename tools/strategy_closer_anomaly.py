"""Sharpe anomaly: per closer quality, Sharpe, net, trades, mean holding bars, exit mix, E4 and daily-return moments."""
import json, sys, time, warnings, os
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ["CODE"])
from app.short_horizon.expert_guided import c3 as C3, strategy_params as SP, strategy_zz as ZZ
from tools import strategy_behaviour_matrix as BM, strategy_output_simulation as S
D_, pf = sys.argv[1], sys.argv[2]; ids = sys.argv[3].split(","); quals = [float(x) for x in sys.argv[4].split(",")]
cells = {c["id"]: c for c in json.load(open(D_ + "/QUEUE.json"))}
AC = S.contract_for_sim(str(S.CONTRACT)); cx, SCFG = S.load_fold("wf2022", D_ + "/data/admitted", D_ + "/data/sel15"); SCFG = S.with_variant(SCFG, C3.FULL_F3)
p = json.load(open(pf)); p = SP.validate({k: v for k, v in {**SP.DEFAULTS, **p}.items() if k in SP.BOUNDS or k in SP.LIST_KEYS or k in SP.META_KEYS}); ZZ.configure(p); genes = SP.genes_c3(p)
for spec in ids:
    cid, seed = spec.rsplit(":", 1)
    w = BM.cell_world(cx, SCFG, AC, cells[cid], int(seed))
    strat, view0, cl = BM.strategy_for(w, genes, AC)
    for q in quals:
        view = dict(view0); view["closer"] = view0["closer"].with_quality(q)
        roll = BM.rollout_cell(w, strat, view, cl)
        st = BM.exit_stats(w, roll, view0["closer"].with_quality(1.0))
        m, tr = roll["metrics"], roll["trades"]
        net = np.array([float(t["net_usd"]) for t in tr]) if tr else np.zeros(0)
        hold = [int(t["exit_bar_abs"]) - int(t["entry_bar_abs"]) for t in tr]
        print(json.dumps({"cell": spec, "q": q, "sharpe": float(m.get("sharpe_daily_sqrt365", np.nan)), "net_usd": float(m.get("net_profit_usd", np.nan)),
                          "trades": len(tr), "mean_hold_bars": float(np.mean(hold)) if hold else None, "mean_net_per_trade": float(net.mean()) if len(net) else None,
                          "win_rate": float((net > 0).mean()) if len(net) else None, "metric_keys": sorted(k for k in m if isinstance(m[k], (int, float)))[:60] if q == quals[0] and spec == ids[0] else None,
                          "m": {k: float(m[k]) for k in m if isinstance(m[k], (int, float)) and any(s in k for s in ("dd", "drawdown", "vol", "cost", "return", "profit", "sortino", "calmar"))},
                          "mix": {side: st[si]["mix"] for si, side in ((0, "BUY"), (1, "SELL"))}, "E4": [st[0]["destructive_share"], st[1]["destructive_share"]]}, default=float), flush=True)
