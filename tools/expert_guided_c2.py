#!/usr/bin/env python3
"""Stage C2 of the causal-ladder design (2026-10-25): contract writer, outcome-table builder and the WF_C2 report.

  contract   write ACTION_CONTRACT_C2.json (``c2.default_contract``, sealed) and print its identity / sha256
  outcomes   build the stamped entry outcome table of one fold (engine replay of every entry; CPU) for E3-M --outcomes
  report     RESULTS_WF_C2: the seed ENSEMBLE of each WF_C2 unit on the validation year (INNER_B, scored after the
             freeze) against its two baselines on the same fold (WF baseline, WF_V1), every fitness variant, the
             pre-registered HOLDS / SEPARABLE rules, the early-close rule, OBS_2024 one read (never decides);
             ``--stage WF_C2G``: RESULTS_WF_C2G, the C2G arm (closers also read their order's geometry) against the C2
             arm of the same fold (the direct contrast) and the WF baseline, same rules
  contract --geometry   write ACTION_CONTRACT_C2G.json instead (``c2.default_contract_geometry``)
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
from tools import expert_guided_sanity_gate as SG  # noqa: E402

FOLDS = ("wf2022", "wf2023")


def cmd_contract(a):
    doc = C2.default_contract_geometry() if getattr(a, "geometry", False) else C2.default_contract()
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


def _ws(results: Path, unit: str, en, period="INNER_B"):
    """``_win`` of a unit plus its name and sanity status (a FAILED_SANITY unit is refused in every verdict)."""
    w = _win(en, period)
    return None if w is None else w | {"unit": unit} | SG.status_of(results, unit)


def _sanity_gate(rec: dict, results: Path, units: list) -> None:
    """Refuse every verdict of an arm that involves a FAILED_SANITY unit; list the units' sanity in the record."""
    st = {u: SG.status_of(results, u) for u in units}
    rec["sanity"] = st
    bad = [u for u, v in st.items() if v["sanity"] == "FAILED_SANITY"]
    if bad:
        ref = {"verdict": ST.SANITY_REFUSAL, "failed_units": bad, "causes": {u: st[u]["sanity_causes"] for u in bad}}
        rec["early_close"]["verdict"] = ST.SANITY_REFUSAL
        rec["early_close"]["refused"] = ref
        rec["fitness_choice"] = dict(rec["fitness_choice"], verdict=ST.SANITY_REFUSAL, refused=ref)


def _md_sanity(doc: dict) -> list:
    L = ["## Sanity gate (FAILED_SANITY units are excluded from every verdict; never shown as results)", "", "| unit | sanity | causes |", "| --- | --- | --- |"]
    for rec in doc["arms"].values():
        for u, v in (rec.get("sanity") or {}).items():
            L.append(f"| {u} | **{v['sanity']}** | {v.get('sanity_causes') or '—'} |")
    return L + [""]


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
            c, b, v = _ws(results, u, en), _ws(results, ub, eb), _ws(results, uv, ev)
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
        _sanity_gate(rec, results, [rec["folds"][f]["unit"] for f in w["folds"]] + [bl["wf"].replace("<fold>", f) for f in w["folds"]]
                     + [bl["wf_v1"].replace("<fold>", f) for f in w["folds"]])
        doc["arms"][nm] = rec
    return doc


def contrast_early_close(g_by_fold: dict, c2_by_fold: dict) -> dict:
    """The direct C2G vs C2 contrast of the early close (reported; the pre-registered verdict is the early-close rule)."""
    per = {}
    for f in FOLDS:
        g, c = g_by_fold.get(f), c2_by_fold.get(f)
        if g is None or c is None:
            per[f] = {"verdict": "NOT_MEASURED"}
            continue
        per[f] = {"value_usd_c2g": g["value_usd"], "value_usd_c2": c["value_usd"], "delta_value_usd": g["value_usd"] - c["value_usd"],
                  "recall_c2g": g["recall"], "recall_c2": c["recall"], "auc_c2g": g["auc"], "auc_c2": c["auc"],
                  "cut_winner_share_c2g": g["cut_winner_share"], "cut_winner_share_c2": c["cut_winner_share"],
                  "verdict": "C2G_BETTER" if g["value_usd"] > c["value_usd"] and g["recall"] > c["recall"] else
                             ("C2G_WORSE" if g["value_usd"] < c["value_usd"] and g["recall"] < c["recall"] else "MIXED")}
    return per


def build_report_c2g(spec: dict, results: Path, audit_dir: Path) -> dict:
    """RESULTS_WF_C2G: the C2G arm vs the C2 arm of the same fold (direct contrast) and the WF baseline, by the seed
    ENSEMBLES on INNER_B; HOLDS / SEPARABLE, activity, the early-close rule (vs the WF baseline) and the C2G-vs-C2
    early-close contrast; every fitness variant; OBS_2024 one read."""
    w = spec["walk_forward_c2g"]
    doc = {"identity": spec["identity"], "stage": "WF_C2G", "contract": w["action_contract"], "rule": w["rule"], "early_close_rule": w["early_close_rule"],
           "fitness_rule": w["fitness_rule"], "arms": {}}
    for arm in w["arms"]:
        nm = arm["name"]
        bl = w["baseline"][nm]
        rec = {"folds": {}, "fitness": {}}
        st_g, st_c, st_b = {}, {}, {}
        ec_g, ec_c, ec_b = {}, {}, {}
        for f in w["folds"]:
            u = f"WF_C2G_{f}_{nm}"
            uc, ub = bl["c2"].replace("<fold>", f), bl["wf"].replace("<fold>", f)
            en, ec, eb = _ens(results, u), _ens(results, uc), _ens(results, ub)
            g, c, b = _ws(results, u, en), _ws(results, uc, ec), _ws(results, ub, eb)
            fr = {"unit": u, "c2g": g, "c2": {"unit": uc, **(c or {})} if c else None, "baseline_wf": {"unit": ub, **(b or {})} if b else None,
                  "obs_2024_one_read": _win(en, "OBS_2024"), "status": "DONE" if en else "MISSING"}
            if g and c:
                fr["separable_vs_c2"] = ST.separable(g, c)
            if g and b:
                fr["separable_vs_wf"] = ST.separable(g, b)
            rec["folds"][f] = fr
            for d_, x in ((st_g, g), (st_c, c), (st_b, b)):
                if x:
                    d_[f] = x
            rec["fitness"][f] = {fn: _win(en, "INNER_B", fn) for fn in ACT.FITNESS} if en else {}
            ec_g[f], ec_c[f], ec_b[f] = _audit(audit_dir, u, False), _audit(audit_dir, uc, False), _audit(audit_dir, ub, True)
        n = len(w["folds"])
        rec["holds"] = ST.holds(st_g) if len(st_g) == n else {"verdict": "NOT_MEASURED", "per_fold": {}}
        rec["holds_c2"] = ST.holds(st_c) if len(st_c) == n else {"verdict": "NOT_MEASURED"}
        rec["holds_baseline_wf"] = ST.holds(st_b) if len(st_b) == n else {"verdict": "NOT_MEASURED"}
        rec["invalid_by_activity"] = {f: (x is not None and not x["valid"]) for f, x in ((f, rec["folds"][f]["c2g"]) for f in w["folds"])}
        rec["early_close"] = {"c2g": ec_g, "c2": ec_c, "baseline_wf": ec_b, **early_close_verdict(ec_g, ec_b)}
        rec["early_close_contrast_c2g_vs_c2"] = contrast_early_close(ec_g, ec_c)
        rec["fitness_choice"] = fitness_choice(rec["fitness"])
        _sanity_gate(rec, results, [rec["folds"][f]["unit"] for f in w["folds"]] + [bl["wf"].replace("<fold>", f) for f in w["folds"]]
                     + [bl["c2"].replace("<fold>", f) for f in w["folds"]])
        doc["arms"][nm] = rec
    return doc


def report_markdown_c2g(doc: dict) -> str:
    L = [f"# Stage WF_C2G — closers that see their own order (spec `{doc['identity']}`, 2026-10-25)", "",
         "C2 plus one change, the owner's extra arm: the closers alone also read the geometry of their own open order (signed "
         "distance to SL and TP in the order's sigma, hours held, side). Seed ensemble on the validation year (INNER_B, scored "
         "once after every choice was frozen on INNER_A) against the C2 arm and the WF baseline of the same fold. OBS_2024 one "
         "read, never decides. TEST closed. Replays of a learned policy under the replay contract, not a profitability claim.", "",
         f"Pre-registered rule: {doc['rule']}", "", f"Early-close rule: {doc['early_close_rule']}", "", f"Fitness rule: {doc['fitness_rule']}", ""]
    for nm, rec in doc["arms"].items():
        L += [f"## Arm `{nm}`", "", "| fold | C2G Sharpe ± SE | trades/wk | DD % | valid | C2 ± SE | WF baseline ± SE | vs C2 | vs WF | 2024 (one read) |",
              "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- |"]
        for f, fr in rec["folds"].items():
            g, c, b, o = fr["c2g"], fr["c2"], fr["baseline_wf"], fr["obs_2024_one_read"]
            L.append(f"| {f} | {_f(g and g['sharpe'])} ± {_f(g and g['se_lo'])} | {_f(g and g['opw'])} | {_f(g and g['dd'], 1)} | {g and g['valid']} | "
                     f"{_f(c and c.get('sharpe'))} ± {_f(c and c.get('se_lo'))} | {_f(b and b.get('sharpe'))} ± {_f(b and b.get('se_lo'))} | "
                     f"{(fr.get('separable_vs_c2') or {}).get('verdict', '—')} | {(fr.get('separable_vs_wf') or {}).get('verdict', '—')} | {_f(o and o['sharpe'])} |")
        L += ["", f"- HOLDS (C2G): **{rec['holds']['verdict']}**; C2 {rec['holds_c2']['verdict']}; WF baseline {rec['holds_baseline_wf']['verdict']}",
              f"- Early close (rule vs the WF baseline): **{rec['early_close']['verdict']}**"]
        for f, x in rec["early_close_contrast_c2g_vs_c2"].items():
            L.append(f"  - {f} C2G vs C2: " + (x["verdict"] if x["verdict"] == "NOT_MEASURED" else
                     f"value {x['value_usd_c2g']:+.0f} vs {x['value_usd_c2']:+.0f} USD, recall {x['recall_c2g']:.2f} vs {x['recall_c2']:.2f}, "
                     f"AUC {x['auc_c2g']:.2f} vs {x['auc_c2']:.2f}, cut-winner {x['cut_winner_share_c2g']:.2f} vs {x['cut_winner_share_c2']:.2f} → {x['verdict']}"))
        fc = rec["fitness_choice"]
        L += [f"- Fitness variant: chosen on wf2022 = `{fc.get('chosen')}`, wf2023 → **{fc['verdict']}**", ""]
    return "\n".join(L + _md_sanity(doc)) + "\n"


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
    return "\n".join(L + _md_sanity(doc)) + "\n"


def cmd_report(a):
    spec = json.loads(Path(a.spec).read_text())
    g = getattr(a, "stage", "WF_C2") == "WF_C2G"
    doc = (build_report_c2g if g else build_report)(spec, Path(a.results), Path(a.audit_dir))
    Path(a.out_json).write_text(json.dumps(doc, indent=1, default=float))
    Path(a.out_md).write_text((report_markdown_c2g if g else report_markdown)(doc))
    print(json.dumps({nm: {"holds": r["holds"]["verdict"], "early_close": r["early_close"]["verdict"], "fitness": r["fitness_choice"]["verdict"]}
                      for nm, r in doc["arms"].items()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("contract")
    c.add_argument("--out", required=True)
    c.add_argument("--geometry", action="store_true", help="the C2G contract (closers also read their order's geometry)")
    o = sub.add_parser("outcomes")
    for k in ("--admitted", "--store", "--recipe", "--split-contract", "--out"):
        o.add_argument(k, required=True)
    o.add_argument("--exec-contract", default="sac")
    o.add_argument("--min-orders-per-week", type=float, default=2.0)
    r = sub.add_parser("report")
    for k in ("--spec", "--results", "--audit-dir", "--out-json", "--out-md"):
        r.add_argument(k, required=True)
    r.add_argument("--stage", choices=("WF_C2", "WF_C2G"), default="WF_C2")
    a = ap.parse_args()
    {"contract": cmd_contract, "outcomes": cmd_outcomes, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
