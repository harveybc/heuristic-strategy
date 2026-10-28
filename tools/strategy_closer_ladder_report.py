"""QUALIFICATION_SPEC D7 verdicts from the closer-ladder result files (tools/strategy_closer_ladder.py).

Per family (cell) and side, for every closer quality < 1 and phi: E4 strategy <= E4 Bayes + 0.05, regret strategy <= regret
Bayes + 0.05, early-close value > 0 (VACUOUS when no SL-destined trade was closed early), and E4 / regret non-decreasing
as quality falls 1.00 -> 0.85 -> 0.70 (tolerance 0.02). Writes CLOSER_LADDER.json and CLOSER_LADDER.md.

  python tools/strategy_closer_ladder_report.py --dir <ladder dir> [--rule 1.0:1]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SIDES = ("BUY", "SELL")
TOL_REF, TOL_MONO = 0.05, 0.02


def _f(x):
    return float(x) if x is not None and isinstance(x, (int, float)) and math.isfinite(float(x)) else float("nan")


def _le(a, b, tol):
    a, b = _f(a), _f(b)
    if math.isnan(a) and math.isnan(b):
        return None
    if math.isnan(a) or math.isnan(b):
        return False
    return a <= b + tol


def verdicts(recs: list, rule: str) -> dict:
    out = {"rule": rule, "families": {}, "summary": {}}
    for rec in recs:
        fam = rec["cell"] + ":" + str(rec["seed"])
        rows = rec["rows"]
        q1 = next((r for r in rows if r["quality"] >= 0.999), None)
        fv = {"q1_bayes_equals_strategy": q1.get("q1_bayes_equals_strategy") if q1 else None, "cells": {}}
        for r in rows:
            st = r["rules"].get(rule, r["strategy"])
            key = f"q{r['quality']:.2f}_phi{r['phi']:g}"
            cell = {"sharpe": st["sharpe"], "net_usd": st["net_usd"], "trades": st["trades"], "mean_hold_bars": st["mean_hold_bars"],
                    "bayes_sharpe": r["bayes"]["sharpe"], "bayes_net_usd": r["bayes"]["net_usd"], "sides": {}}
            for s in SIDES:
                a, b = st[s], r["bayes"][s]
                sv = {"E4": a["E4"], "E4_bayes": b["E4"], "regret": a["regret_norm"], "regret_bayes": b["regret_norm"], "early": a["mix"]["early"], "early_bayes": b["mix"]["early"],
                      "early_close_value": a["early_close_value_norm"], "sl_destined_closed_early": a["sl_destined_closed_early"], "closes": a["closes"]}
                if r["quality"] < 0.999:
                    sv["pass_E4"] = _le(a["E4"], b["E4"], TOL_REF)
                    sv["pass_regret"] = _le(a["regret_norm"], b["regret_norm"], TOL_REF)
                    sv["pass_early_close_value"] = None if not a["sl_destined_closed_early"] else bool(_f(a["early_close_value_norm"]) > 0)
                cell["sides"][s] = sv
            fv["cells"][key] = cell
        for phi in sorted({r["phi"] for r in rows if r["quality"] < 0.999}):
            seq = [k for k in (f"q1.00_phi{rows[0]['phi']:g}", f"q0.85_phi{phi:g}", f"q0.70_phi{phi:g}") if k in fv["cells"]]
            for s in SIDES:
                for m in ("E4", "regret"):
                    v = [_f(fv["cells"][k]["sides"][s][m]) for k in seq]
                    ok = all((math.isnan(v[i]) or math.isnan(v[i + 1]) or v[i + 1] >= v[i] - TOL_MONO) for i in range(len(v) - 1))
                    fv.setdefault("monotone", {})[f"phi{phi:g}_{s}_{m}"] = ok
        out["families"][fam] = fv
    keys = sorted({k for f in out["families"].values() for k in f["cells"]})
    for k in keys:
        agg = {"E4": [], "E4_bayes": [], "regret": [], "regret_bayes": [], "early": [], "early_bayes": [], "sharpe": [], "bayes_sharpe": [], "net_usd": [], "bayes_net_usd": [],
               "trades": [], "mean_hold_bars": []}
        cnt = {"pass_E4": [0, 0], "pass_regret": [0, 0], "pass_early_close_value": [0, 0, 0]}
        for f in out["families"].values():
            c = f["cells"].get(k)
            if c is None:
                continue
            for m in ("sharpe", "bayes_sharpe", "net_usd", "bayes_net_usd", "trades", "mean_hold_bars"):
                agg[m].append(_f(c[m]))
            for s in SIDES:
                sv = c["sides"][s]
                for m in ("E4", "E4_bayes", "regret", "regret_bayes", "early", "early_bayes"):
                    agg[m].append(_f(sv[m]))
                for m in ("pass_E4", "pass_regret"):
                    if m in sv and sv[m] is not None:
                        cnt[m][0] += int(sv[m])
                        cnt[m][1] += 1
                if "pass_early_close_value" in sv:
                    v = sv["pass_early_close_value"]
                    cnt["pass_early_close_value"][0 if v is True else (1 if v is False else 2)] += 1
        mean = {m: (sum(x for x in v if not math.isnan(x)) / max(sum(1 for x in v if not math.isnan(x)), 1)) for m, v in agg.items()}
        out["summary"][k] = {"mean": mean, "counts": cnt}
    mono = [v for f in out["families"].values() for v in f.get("monotone", {}).values()]
    out["monotone_pass"] = [sum(mono), len(mono)]
    out["q1_equal"] = [sum(1 for f in out["families"].values() if f["q1_bayes_equals_strategy"]), len(out["families"])]
    return out


def markdown(v: dict) -> str:
    L = [f"# Closer ladder vs the information-limited Bayes oracle (rule z_close:k_persist = {v['rule']})", "",
         f"q1.00: strategy trades identical to the Bayes closer's in {v['q1_equal'][0]} / {v['q1_equal'][1]} families. Monotone in quality: {v['monotone_pass'][0]} / {v['monotone_pass'][1]} checks.", "",
         "| quality, phi | E4 strategy | E4 Bayes | regret strategy | regret Bayes | early share strategy | early share Bayes | Sharpe strategy | Sharpe Bayes | net strategy | net Bayes | trades | hold bars | pass E4 | pass regret | early-close value pass / fail / vacuous |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in v["summary"].items():
        m, c = s["mean"], s["counts"]
        L.append(f"| {k} | {m['E4']:.3f} | {m['E4_bayes']:.3f} | {m['regret']:.3f} | {m['regret_bayes']:.3f} | {m['early']:.3f} | {m['early_bayes']:.3f} | {m['sharpe']:.2f} | {m['bayes_sharpe']:.2f} | "
                 f"{m['net_usd']:,.0f} | {m['bayes_net_usd']:,.0f} | {m['trades']:.0f} | {m['mean_hold_bars']:.1f} | {c['pass_E4'][0]}/{c['pass_E4'][1]} | {c['pass_regret'][0]}/{c['pass_regret'][1]} | "
                 f"{c['pass_early_close_value'][0]}/{c['pass_early_close_value'][1]}/{c['pass_early_close_value'][2]} |")
    L += ["", "Means over the 24 families (E4, regret and early share also over both sides). Regret: (P* - P(exit)) / initial cash risk at the SL.", "",
          "## Per family and side (E4 strategy / Bayes, regret strategy / Bayes)", ""]
    keys = list(v["summary"])
    L.append("| family | side | " + " | ".join(keys) + " |")
    L.append("|---|---|" + "---|" * len(keys))
    for fam, f in v["families"].items():
        for s in SIDES:
            cells = []
            for k in keys:
                sv = f["cells"].get(k, {}).get("sides", {}).get(s)
                cells.append("" if sv is None else f"{_f(sv['E4']):.2f}/{_f(sv['E4_bayes']):.2f}, {_f(sv['regret']):.2f}/{_f(sv['regret_bayes']):.2f}")
            L.append(f"| {fam} | {s} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--rule", default="1.0:1")
    a = ap.parse_args()
    d = Path(a.dir)
    recs = [json.loads(p.read_text()) for p in sorted(d.glob("LADDER_*.json"))]
    v = verdicts(recs, a.rule)
    tag = a.rule.replace(":", "_")
    (d / f"CLOSER_LADDER_{tag}.json").write_text(json.dumps(v, indent=1, default=float))
    (d / f"CLOSER_LADDER_{tag}.md").write_text(markdown(v))
    print(markdown(v).split("## Per family")[0])


if __name__ == "__main__":
    main()
