#!/usr/bin/env python
"""PZ3 report (PZ_PREREGISTRO_2026_10_30 section 3): per witness PASS / FAIL / NOT_EVALUABLE / CONTROL_CONFLICT over the
tier-1 records of the behaviour matrix run with --strategy pz, the per-record governing verdict, the pre-registered
"passes the matrix" rule, and the first decision where PZ diverges from the truth oracle (side or slot) in each failing
record (the world rebuilt, PZ and the oracle rolled, inside the failing witness's window when it has one).

  summary  --results <dir> --out <json>
  diverge  --cells <TIER1_CELLS.json> --results <dir> --params <STRATEGY_PARAMS> --pz-params <PZ_PARAMS> --admitted <d> --csv-dir <d>
           [--seeds 0] --out <jsonl>
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def witness_table(recs: list) -> dict:
    """Per witness: PASS / FAIL / NOT_EVALUABLE / CONTROL_CONFLICT counts (governing form; the exit layer's effective list)."""
    tab = collections.defaultdict(collections.Counter)
    for d in recs:
        v = d.get("validity") or {}
        ne, cc = set(v.get("not_evaluable") or []), set(v.get("control_conflicts") or [])
        for k, st in (v.get("per_witness") or {}).items():
            if k in ne:
                tab[k]["NOT_EVALUABLE"] += 1
            elif k in cc or st.get("status") == "CONTROL_CONFLICT":
                tab[k]["CONTROL_CONFLICT"] += 1
            else:
                tab[k]["PASS" if st.get("pass") else "FAIL"] += 1
        ex = d.get("exit") or {}
        conf = set(ex.get("control_conflicts") or [])
        for k, w in (ex.get("witnesses") or {}).items():
            if not isinstance(w, dict) or "pass" not in w:
                continue
            if k in conf or w.get("status") == "CONTROL_CONFLICT":
                tab[k]["CONTROL_CONFLICT"] += 1
            elif k in (ex.get("defects_by_Ostar") or []):
                tab[k]["DEFECT_OF_CONTROL"] += 1
            else:
                tab[k]["PASS" if w["pass"] else "FAIL"] += 1
        w14 = (d.get("results") or {}).get("W14_mirror")
        if isinstance(w14, dict) and "pass" in w14:
            tab["W14_mirror"]["NOT_EVALUABLE" if w14.get("status") == "NOT_EVALUABLE" or w14.get("not_evaluable") else ("PASS" if w14["pass"] else "FAIL")] += 1
    return {k: dict(v) for k, v in sorted(tab.items())}


def record_verdict(d: dict) -> str:
    if d.get("error"):
        return "ERROR"
    if d.get("PASS_effective"):
        return "PASS"
    v = d.get("validity") or {}
    ex = d.get("exit") or {}
    real = [k for k in (v.get("failing") or []) if k not in set(v.get("control_conflicts") or [])] + list(ex.get("failing_effective") or [])
    w14 = (d.get("results") or {}).get("W14_mirror")
    if isinstance(w14, dict) and w14.get("pass") is False:
        real.append("W14_mirror")
    if real:
        return "FAIL"
    if v.get("not_evaluable"):
        return "NOT_EVALUABLE"
    if v.get("control_conflicts") or ex.get("control_conflicts"):
        return "CONTROL_CONFLICT"
    return "FAIL"


def cmd_summary(a):
    recs = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(f"{a.results}/CELL_*.json"))]
    verdicts = collections.Counter(record_verdict(d) for d in recs)
    failing = collections.Counter()
    for d in recs:
        for k in (d.get("validity") or {}).get("failing") or []:
            failing[k] += 1
        for k in (d.get("exit") or {}).get("failing_effective") or []:
            failing[k] += 1
    act = [d["results"]["summary"].get("opens", 0) for d in recs if "results" in d and "summary" in d["results"]]
    ts = [d.get("trade_summary") or {} for d in recs]
    hm = [((d.get("horizon_mix") or {}).get("executed") or {}) for d in recs]
    out = {"records": len(recs), "record_verdicts": dict(verdicts), "matrix_pass_preregistered": verdicts.get("FAIL", 0) == 0 and verdicts.get("ERROR", 0) == 0,
           "witness_table": witness_table(recs), "failing_counts": dict(failing.most_common()),
           "trades_per_week_median": float(np.median([d["results"]["W0_activity"]["value"] for d in recs if isinstance((d.get("results") or {}).get("W0_activity"), dict)])) if recs else float("nan"),
           "net_sigma_per_trade_median_of_records": float(np.nanmedian([t.get("net_sigma_per_trade_median", np.nan) for t in ts])) if ts else float("nan"),
           "horizon_mix_median": {h: float(np.nanmedian([x.get(h, np.nan) for x in hm])) for h in ("h3", "h6", "h12", "h24")},
           "levels_clipped_total": int(sum(((d.get("non_fills") or {}).get("levels_clipped") or 0) for d in recs)),
           "early_close_share_median": float(np.nanmedian([np.mean([t["exit_reason"] == "EARLY" for t in d["trades"]]) if d.get("trades") else np.nan for d in recs])),
           "sl_share_median": float(np.nanmedian([np.mean([t["exit_reason"] == "SL" for t in d["trades"]]) if d.get("trades") else np.nan for d in recs])),
           "pz_genes_sha256": sorted({(d.get("pz") or {}).get("genes_sha256") for d in recs} - {None}), "per_record": [{"id": d["id"], "seed": d["seed"], "verdict": record_verdict(d),
            "failing": (d.get("validity") or {}).get("failing"), "exit_failing": (d.get("exit") or {}).get("failing_effective"), "sharpe": d.get("sharpe_daily")} for d in recs]}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("records", "record_verdicts", "matrix_pass_preregistered", "failing_counts")}, default=float))


def cmd_diverge(a):
    from app.short_horizon.expert_guided import c3 as C3, strategy_params as SP, strategy_pz as PZ, strategy_zz as ZZ, menu as M
    from tools import strategy_behaviour_matrix as BM
    from tools import strategy_hierarchy as H
    from tools import strategy_output_simulation as S
    cells = {c["id"]: c for c in json.loads(Path(a.cells).read_text())}
    AC = S.contract_for_sim(S.CONTRACT)
    cx, SCFG = S.load_fold("wf2022", a.admitted, a.csv_dir)
    doc = SP.load(a.params)
    SCFG = S.with_variant(SCFG, C3.exec_variant_of(doc))
    ZZ.configure(doc)
    genes = SP.genes_c3(doc)
    BM.STRATEGY, BM.PZ_GENES = "pz", PZ.load_genes(a.pz_params)
    H.MENU_ORDERS_FN = PZ.truth_orders_fn(BM.PZ_GENES)
    seeds = {int(x) for x in a.seeds.split(",")}
    done = set()
    if Path(a.out).exists():
        done = {json.loads(x)["record"] for x in Path(a.out).read_text().splitlines() if x.strip()}
    for p in sorted(glob.glob(f"{a.results}/CELL_*.json")):
        d = json.loads(Path(p).read_text())
        if int(d["seed"]) not in seeds or record_verdict(d) == "PASS" or f"{d['id']}_s{d['seed']}" in done or d["id"] not in cells:
            continue
        fails = list((d.get("validity") or {}).get("failing") or []) + list((d.get("exit") or {}).get("failing_effective") or [])
        w = BM.cell_world(cx, SCFG, AC, cells[d["id"]], int(d["seed"]))
        strat, view, cl = BM.strategy_for(w, genes, AC)
        roll = BM.rollout_cell(w, strat, view, cl)
        ov = BM.oracle_view(w, view)
        oroll = BM.rollout_oracle(w, ov, cl)
        rb = np.asarray(w["cx"].ds.row_bar)
        meta = w.get("meta") or {}
        sl = {x[0]: x for x in roll["log"]}
        ol = {x[0]: x for x in oroll["log"]}
        out = {"record": f"{d['id']}_s{d['seed']}", "failing": fails, "first_divergence": {}}
        for wid in fails:
            lo_bar = 0
            if wid.startswith("W8") and meta.get("switch_bars"):
                lo_bar = int(meta["switch_bars"][0]) + BM.LAG
            if wid.startswith("W17") and meta.get("burst_windows"):
                lo_bar = int(meta["burst_windows"][0][0])
            key = (wid.split("_")[0], lo_bar)
            if key in out["first_divergence"]:
                continue
            first = None
            for r in sorted(set(sl) & set(ol)):
                if rb[r] < lo_bar:
                    continue
                x, y = sl[r], ol[r]
                sa, sb = (x[5], x[7 + x[5]] if x[5] >= 0 else -1), (y[5], y[7 + y[5]] if y[5] >= 0 else -1)
                if sa != sb:
                    tn = [[float(v) if np.isfinite(v) else None for v in BM.menu_truth_net(w["truth"], r, si)] for si in (0, 1)]
                    first = {"row": int(r), "bar": int(rb[r]), "pz": {"side": int(sa[0]), "h": (M.slot_h(sa[1]) if sa[1] >= 0 else None), "score": [x[1], x[2]], "cleared": [x[3], x[4]]},
                             "oracle": {"side": int(sb[0]), "h": (M.slot_h(sb[1]) if sb[1] >= 0 else None), "net": [y[1], y[2]]}, "truth_net_by_side_slot": tn,
                             "pz_p_up": [float(v) for v in view["pz_p_up"][r]]}
                    break
            out["first_divergence"][f"{wid}@{lo_bar}"] = first
        # the first early close the hold oracle disagrees with (W4 / E family): remaining hold value > 0 at a PZ close
        ec = [c for c in roll["closes"] if c[5] and float(c[3]) > BM.EPS_CLOSE]
        out["first_unjustified_close"] = ({"row": int(ec[0][0]), "bar": int(ec[0][1]), "side": int(ec[0][2]), "remaining_sigma": float(ec[0][3]), "hold_rule": bool(ec[0][6]), "urgent_rule": bool(ec[0][7])}
                                          if ec else None)
        out["closes_unjustified_share"] = float(len(ec) / max(1, sum(1 for c in roll["closes"] if c[5])))
        out["unjustified_by_rule"] = {"hold": int(sum(1 for c in ec if c[6])), "urgent_only": int(sum(1 for c in ec if c[7] and not c[6]))}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(out, default=float) + "\n")
        print(out["record"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--results", required=True)
    s.add_argument("--out", required=True)
    d = sub.add_parser("diverge")
    for k in ("--cells", "--results", "--params", "--pz-params", "--admitted", "--csv-dir", "--out"):
        d.add_argument(k, required=True)
    d.add_argument("--seeds", default="0")
    a = ap.parse_args()
    {"summary": cmd_summary, "diverge": cmd_diverge}[a.cmd](a)


if __name__ == "__main__":
    main()
