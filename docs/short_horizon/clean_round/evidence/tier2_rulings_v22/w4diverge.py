"""W4 'unjustified' early closes (remaining hold value > 0 at the close) against O*: per close, the tick, the head's A,
the path's A, the remaining value, and O*'s choice at that tick on the SAME trade (close iff path A >= -EPS_CLOSE),
plus the earliest tick of that trade where O*'s rule fires. usage: w4diverge.py QUEUE.json OUT.json cell:seed ..."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from tools import strategy_behaviour_matrix as BM
from tools import strategy_output_simulation as S
from app.short_horizon.expert_guided import c3 as C3, strategy_zz as ZZ, strategy_params as SP
import os
R = os.environ.get("BM_ROOT", "/var/tmp/behaviour-matrix-20261027")
cells = {c["id"]: c for c in json.load(open(sys.argv[1]))}
AC = S.contract_for_sim(S.CONTRACT)
cx, SCFG = S.load_fold("wf2022", R + "/data/admitted", R + "/data/sel15")
doc = SP.load("docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json")
SCFG = S.with_variant(SCFG, C3.exec_variant_of(doc)); ZZ.configure(doc); genes = SP.genes_c3(doc)
out = []
for spec in sys.argv[3:]:
    cid, seed = spec.split(":")
    w = BM.cell_world(cx, SCFG, AC, cells[cid], int(seed))
    strat, view, cl = BM.strategy_for(w, genes, AC)
    roll = BM.rollout_cell(w, strat, view, cl)
    closer = view["closer"]
    ec = [c for c in roll["closes"] if c[5]]
    rows = []
    for c in ec:
        r_, bar, si, rem, psl, a_head = int(c[0]), int(c[1]), int(c[2]), float(c[3]), float(c[4]), float(c[9])
        t = next((x for x in roll["trades"] if int(x["entry_bar_abs"]) <= bar < int(x["exit_bar_abs"]) + 1 and (0 if int(x["direction"]) > 0 else 1) == si), None)
        pos = (roll.get("positions") or {}).get((int(t["entry_bar_abs"]), si)) if t is not None else None
        if pos is None:
            rows.append({"bar": bar, "side": si, "remaining": rem, "head_A": a_head, "path_A": None, "note": "position not found"}); continue
        path = closer.path(pos)
        a_path = float(S.path_advantage(path, bar))
        first = None
        for b in range(int(pos.entry_bar), bar + 1):
            ap = S.path_advantage(path, b)
            if np.isfinite(ap) and ap >= -BM.EPS_CLOSE:
                first = b; break
        rows.append({"bar": bar, "side": si, "entry_bar": int(pos.entry_bar), "remaining": rem, "justified_sealed": bool(rem <= 0.0), "head_A": a_head, "path_A": a_path,
                     "head_minus_path": a_head - a_path, "Ostar_closes_at_this_tick": bool(np.isfinite(a_path) and a_path >= -BM.EPS_CLOSE),
                     "Ostar_first_close_tick_on_this_trade": first, "strategy_rule_fires": bool(a_head - 0.0 >= -ZZ.PARAMS["closer_adv_margin"])})
    unj = [x for x in rows if x.get("justified_sealed") is False]
    out.append({"record": f"{cid}_s{seed}", "early_closes": len(rows), "unjustified": len(unj), "W4_sealed_share": (sum(1 for x in rows if x.get("justified_sealed")) / len(rows)) if rows else None,
                "unjustified_closes": unj, "decision_differences_vs_Ostar": [x for x in rows if x.get("path_A") is not None and (not x["Ostar_closes_at_this_tick"] or x["Ostar_first_close_tick_on_this_trade"] != x["bar"])],
                "max_abs_head_minus_path": max((abs(x["head_minus_path"]) for x in rows if x.get("path_A") is not None), default=None)})
    print(json.dumps(out[-1], default=float)[:1500], flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1, default=float)
