#!/usr/bin/env python3
"""Stage C2 of the causal-ladder design (2026-10-25): contract writer, outcome-table builder and the WF_C2 report.

  contract   write ACTION_CONTRACT_C2.json (``c2.default_contract``, sealed) and print its identity / sha256
  outcomes   build the stamped entry outcome table of one fold (engine replay of every entry; CPU) for E3-M --outcomes
  report     RESULTS_WF_C2: the seed ENSEMBLE of each WF_C2 unit on the validation year (INNER_B, scored after the
             freeze) against its two baselines on the same fold (WF baseline, WF_V1), every fitness variant, the
             pre-registered HOLDS / SEPARABLE rules, the early-close rule, OBS_2024 one read (never decides)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import action_contract as ACT  # noqa: E402
from app.short_horizon.expert_guided import c2 as C2  # noqa: E402
from app.short_horizon.expert_guided import stats as ST  # noqa: E402

FOLDS = ("wf2022", "wf2023")


def cmd_contract(a):
    doc = C2.default_contract()
    errs = C2.validate(doc)
    if errs:
        raise SystemExit(f"C2_CONTRACT_INVALID:{errs}")
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"identity": doc["identity"], "sha256": ACT.sha256_file(a.out), "encoding_audit": C2.encoding_audit(doc)["verdict"]}))


def cmd_outcomes(a):
    from app.short_horizon.expert_guided import data as DG
    from app.short_horizon.expert_guided import outcomes as OC
    from tools.expert_guided_e1t import Ctx
    from tools.expert_guided_e3m import effective_contracts
    rec = json.loads(Path(a.recipe).read_text())
    feats = [c["store_column"] for c in rec["tensor"]["columns"]]
    ds = DG.load_store(a.admitted, a.store, feats, scaling="fixed_train", contract=a.split_contract, recipe=rec)
    cx = Ctx(ds)
    SCFG, _ = effective_contracts(a.exec_contract, a.min_orders_per_week)
    t0 = time.time()
    t = OC.entry_table(cx, SCFG, progress=lambda i, n: print(json.dumps({"rows": f"{i}/{n}", "s": round(time.time() - t0, 1)}), flush=True))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    OC.save_table(a.out, t)
    elig = int(t["eligible"].sum())
    print(json.dumps({"out": a.out, "eligible_rows": elig, "refused": t["refused"], "stamp": t["stamp"]["sha256"], "seconds": round(time.time() - t0, 1)}))


# ----------------------------------------------------------------------------- report
def _ens(results: Path, unit: str):
    p = results / unit / "ENSEMBLE.json"
    return json.loads(p.read_text()) if p.exists() else None


def _win(en, period="INNER_B", fitness=None):
    """{"sharpe", "se_lo", "opw", "dd", "valid", "trades"} of an ensemble window (primary or a fitness variant)."""
    if en is None:
        return None
    if fitness is None:
        w = en["ensemble"].get(period)
    else:
        bf = (en.get("by_fitness") or {}).get(fitness)
        w = bf["windows"].get(period) if bf else None
    if not w:
        return None
    m = w["metrics"]
    return {"sharpe": float(w["sharpe"]), "se_lo": float(w["se_lo"]), "opw": float(m["orders_per_week"]), "dd": float(m["max_drawdown_pct"]),
            "valid": bool(m["valid"] and m["orders_per_week"] >= ACT.MIN_TRADES_PER_WEEK), "trades": int(m["trades"])}


def _audit(audit_dir: Path, unit: str, legacy: bool):
    """(early-close value USD, CI, closer recall) of an audited unit's ensemble on INNER_B (primary fitness for C2/v1)."""
    p = audit_dir / f"EARLY_CLOSE_{unit}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    ens = d.get("ensemble") or {}
    if legacy:
        x = (ens.get("legacy") or {}).get("INNER_B")
    else:
        prim = next((k for k, v in ens.items() if isinstance(v, dict) and v.get("primary")), None)
        x = (ens.get(prim) or {}).get("INNER_B") if prim else None
    if not x:
        return None
    return {"value_usd": float(x["early_close_value_usd"]), "ci95": x["early_close_value_usd_ci95"], "recall": float(x["closer"]["recall_sl_first"]),
            "auc": float(x["closer"]["auc_sl_first"]), "early_close_share": float(x["early_close_share"]), "flags": x.get("flags", []),
            "cut_winner_share": float(x["class_shares"]["CUT_WINNER"])}


def early_close_verdict(c2_by_fold: dict, base_by_fold: dict) -> dict:
    """The pre-registered early-close rule of WF_C2: value > 0 in BOTH folds and recall >= 2x the baseline's in both."""
    per = {}
    for f in FOLDS:
        c, b = c2_by_fold.get(f), base_by_fold.get(f)
        if c is None or b is None:
            per[f] = {"verdict": "NOT_MEASURED"}
            continue
        ok_v = c["value_usd"] > 0
        ok_r = math.isfinite(c["recall"]) and math.isfinite(b["recall"]) and c["recall"] >= 2.0 * b["recall"]
        per[f] = {"value_usd": c["value_usd"], "value_positive": ok_v, "recall": c["recall"], "baseline_recall": b["recall"], "recall_at_least_2x": ok_r,
                  "verdict": "PASS" if ok_v and ok_r else "FAIL"}
    allm = all(v["verdict"] != "NOT_MEASURED" for v in per.values())
    return {"per_fold": per, "verdict": ("FIXES_EARLY_CLOSE" if allm and all(v["verdict"] == "PASS" for v in per.values())
                                         else ("NOT_MEASURED" if not allm else "DOES_NOT_FIX_EARLY_CLOSE"))}


def fitness_choice(by_fold: dict) -> dict:
    """The pre-registered fitness rule: chosen on wf2022 by the highest ensemble Sharpe - 1 SE among variants valid out
    of sample (>= 2 trades/week); confirmed on wf2023 when Sharpe - 1 SE > 0 there with >= 2 trades/week."""
    a, b = by_fold.get("wf2022") or {}, by_fold.get("wf2023") or {}
    cands = [(f, w) for f, w in a.items() if w is not None and w["valid"] and math.isfinite(w["se_lo"])]
    if not cands:
        return {"chosen": None, "verdict": "NO_VALID_VARIANT_ON_WF2022"}
    f, w = max(cands, key=lambda x: x[1]["sharpe"] - x[1]["se_lo"])
    c = b.get(f)
    if c is None:
        return {"chosen": f, "wf2022": w, "wf2023": None, "verdict": "NOT_MEASURED"}
    conf = c["valid"] and math.isfinite(c["se_lo"]) and (c["sharpe"] - c["se_lo"]) > 0
    return {"chosen": f, "wf2022": w, "wf2023": c, "verdict": "CONFIRMED" if conf else "REFUTED"}


def build_report(spec: dict, results: Path, audit_dir: Path) -> dict:
    w = spec["walk_forward_c2"]
    doc = {"identity": spec["identity"], "stage": "WF_C2", "contract": w["action_contract"], "rule": w["rule"], "early_close_rule": w["early_close_rule"],
           "fitness_rule": w["fitness_rule"], "arms": {}}
    for arm in w["arms"]:
        nm = arm["name"]
        bl = w["baseline"][nm]
        rec = {"folds": {}, "fitness": {}}
        stats_c2, stats_b, stats_v1 = {}, {}, {}
        ec_c2, ec_b = {}, {}
        for f in w["folds"]:
            u = f"WF_C2_{f}_{nm}"
            ub, uv = bl["wf"].replace("<fold>", f), bl["wf_v1"].replace("<fold>", f)
            en, eb, ev = _ens(results, u), _ens(results, ub), _ens(results, uv)
            c, b, v = _win(en), _win(eb), _win(ev)
            fr = {"unit": u, "c2": c, "baseline_wf": {"unit": ub, **(b or {})} if b else None, "baseline_wf_v1": {"unit": uv, **(v or {})} if v else None,
                  "obs_2024_one_read": _win(en, "OBS_2024"), "status": "DONE" if en else "MISSING"}
            if c and b:
                fr["separable_vs_wf"] = ST.separable(c, b)
            if c and v:
                fr["separable_vs_wf_v1"] = ST.separable(c, v)
            rec["folds"][f] = fr
            if c:
                stats_c2[f] = c
            if b:
                stats_b[f] = b
            if v:
                stats_v1[f] = v
            rec["fitness"][f] = {fn: _win(en, "INNER_B", fn) for fn in ACT.FITNESS} if en else {}
            if en:
                for fn in ACT.FITNESS:
                    av = (en["by_fitness"].get(fn) or {}).get("archive_view") or {}
                    bu = av.get("best_unconstrained_by_fitness") or {}
                    ws = (bu.get("windows_not_selectable") or {}).get("INNER_B")
                    if ws and rec["fitness"][f].get(fn) is not None:
                        rec["fitness"][f][fn]["best_unconstrained_not_selectable"] = {"sharpe": ws["sharpe"], "opw": ws["metrics"]["orders_per_week"]}
            ec_c2[f], ec_b[f] = _audit(audit_dir, u, False), _audit(audit_dir, ub, True)
        rec["holds"] = ST.holds(stats_c2) if len(stats_c2) == len(w["folds"]) else {"verdict": "NOT_MEASURED", "per_fold": {}}
        rec["holds_baseline_wf"] = ST.holds(stats_b) if len(stats_b) == len(w["folds"]) else {"verdict": "NOT_MEASURED"}
        rec["holds_baseline_wf_v1"] = ST.holds(stats_v1) if len(stats_v1) == len(w["folds"]) else {"verdict": "NOT_MEASURED"}
        rec["invalid_by_activity"] = {f: (c is not None and not c["valid"]) for f, c in ((f, rec["folds"][f]["c2"]) for f in w["folds"])}
        rec["early_close"] = {"c2": ec_c2, "baseline_wf": ec_b, **early_close_verdict(ec_c2, ec_b)}
        rec["fitness_choice"] = fitness_choice(rec["fitness"])
        doc["arms"][nm] = rec
    return doc


def _f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def report_markdown(doc: dict) -> str:
    L = [f"# Stage WF_C2 — learning the consequences of our actions (spec `{doc['identity']}`, 2026-10-25)", "",
         "Seed ensemble on the validation year of each walk-forward fold (INNER_B, scored once after every choice was frozen on "
         "INNER_A), against the two baselines on the same fold. OBS_2024 is one read and never decides. TEST closed. These are "
         "replays of a learned policy under the replay contract, not a profitability claim.", "",
         f"Pre-registered rule: {doc['rule']}", "", f"Early-close rule: {doc['early_close_rule']}", "", f"Fitness rule: {doc['fitness_rule']}", ""]
    for nm, rec in doc["arms"].items():
        L += [f"## Arm `{nm}`", "", "| fold | C2 Sharpe ± SE | trades/wk | DD % | valid | WF baseline ± SE | WF_V1 ± SE | vs WF | vs WF_V1 | 2024 (one read) |",
              "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- |"]
        for f, fr in rec["folds"].items():
            c, b, v, o = fr["c2"], fr["baseline_wf"], fr["baseline_wf_v1"], fr["obs_2024_one_read"]
            L.append(f"| {f} | {_f(c and c['sharpe'])} ± {_f(c and c['se_lo'])} | {_f(c and c['opw'])} | {_f(c and c['dd'], 1)} | {c and c['valid']} | "
                     f"{_f(b and b.get('sharpe'))} ± {_f(b and b.get('se_lo'))} | {_f(v and v.get('sharpe'))} ± {_f(v and v.get('se_lo'))} | "
                     f"{(fr.get('separable_vs_wf') or {}).get('verdict', '—')} | {(fr.get('separable_vs_wf_v1') or {}).get('verdict', '—')} | {_f(o and o['sharpe'])} |")
        L += ["", f"- HOLDS (C2): **{rec['holds']['verdict']}**; baselines: WF {rec['holds_baseline_wf']['verdict']}, WF_V1 {rec['holds_baseline_wf_v1']['verdict']}",
              f"- Early close: **{rec['early_close']['verdict']}**"]
        for f, x in rec["early_close"]["per_fold"].items():
            c = rec["early_close"]["c2"].get(f)
            L.append(f"  - {f}: " + (x["verdict"] if x["verdict"] == "NOT_MEASURED" else
                     f"value {x['value_usd']:+.0f} USD (CI {c['ci95'][0]:+.0f}, {c['ci95'][1]:+.0f}), recall {x['recall']:.2f} vs baseline {x['baseline_recall']:.2f}, "
                     f"cut-winner share {c['cut_winner_share']:.2f}, AUC {c['auc']:.2f} → {x['verdict']}"))
        fc = rec["fitness_choice"]
        L += [f"- Fitness variant: chosen on wf2022 = `{fc.get('chosen')}`, wf2023 → **{fc['verdict']}**", "",
              "| fitness | wf2022 Sharpe ± SE | trades/wk | valid | wf2023 Sharpe ± SE | trades/wk | valid |", "| --- | --- | ---: | --- | --- | ---: | --- |"]
        for fn in ACT.FITNESS:
            a_, b_ = (rec["fitness"].get("wf2022") or {}).get(fn), (rec["fitness"].get("wf2023") or {}).get(fn)
            L.append(f"| {fn} | {_f(a_ and a_['sharpe'])} ± {_f(a_ and a_['se_lo'])} | {_f(a_ and a_['opw'])} | {a_ and a_['valid']} | "
                     f"{_f(b_ and b_['sharpe'])} ± {_f(b_ and b_['se_lo'])} | {_f(b_ and b_['opw'])} | {b_ and b_['valid']} |")
        L.append("")
    return "\n".join(L) + "\n"


def cmd_report(a):
    spec = json.loads(Path(a.spec).read_text())
    doc = build_report(spec, Path(a.results), Path(a.audit_dir))
    Path(a.out_json).write_text(json.dumps(doc, indent=1, default=float))
    Path(a.out_md).write_text(report_markdown(doc))
    print(json.dumps({nm: {"holds": r["holds"]["verdict"], "early_close": r["early_close"]["verdict"], "fitness": r["fitness_choice"]["verdict"]}
                      for nm, r in doc["arms"].items()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("contract")
    c.add_argument("--out", required=True)
    o = sub.add_parser("outcomes")
    for k in ("--admitted", "--store", "--recipe", "--split-contract", "--out"):
        o.add_argument(k, required=True)
    o.add_argument("--exec-contract", default="sac")
    o.add_argument("--min-orders-per-week", type=float, default=2.0)
    r = sub.add_parser("report")
    for k in ("--spec", "--results", "--audit-dir", "--out-json", "--out-md"):
        r.add_argument(k, required=True)
    a = ap.parse_args()
    {"contract": cmd_contract, "outcomes": cmd_outcomes, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
