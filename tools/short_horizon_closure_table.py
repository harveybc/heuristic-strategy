#!/usr/bin/env python3
"""Owner closure table, generated from the retained artifacts under docs/short_horizon/results.

Per measured predictor or strategy: error metric with its scale, the naive on the SAME rows, skill =
(err - naive) / naive (negative = better than the naive), a literature reference value with its source
where one exists, and a comparability verdict. Rows whose artifact is absent say NO_NEW_MEASUREMENT.
Nothing here is typed from memory except the literature column, which is static metadata.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.short_horizon_price_path_report import es  # noqa: E402

RESULTS = ROOT / "docs/short_horizon/results"

LIT = {
    "har": "HAR-RV vs. the previous-period RV naive: out-of-sample RMSE gain of roughly 10-30 % on daily FX/equity "
           "realized volatility (Corsi 2009, J. Financial Econometrics 7(2); Andersen, Bollerslev, Diebold & Labys 2003, "
           "Econometrica 71(2), R2 ~0,5 for next-day RV). Approximate recalled magnitudes.",
    "rw": "Random walk benchmark for the FX price level at horizons of days (Meese & Rogoff 1983, J. Int. Economics 14): "
          "no model beats the naive out of sample.",
    "abdv": "Announcement surprises move DM/$ (EUR/USD) within 5 minutes; a one-s.d. NFP surprise ~0,1-0,2 % with the "
            "conditional-mean jump immediate (Andersen, Bollerslev, Diebold & Vega 2003, AER 93(1)). Approximate recalled magnitudes.",
    "ederington": "Volatility in the hour after major U.S. releases is several times its normal level and decays within hours "
                  "(Ederington & Lee 1993, J. Finance 48(4); ABDV 2003).",
    "dir": "Directional accuracy of hourly/daily FX sign forecasts in the literature clusters at 0,50-0,56 "
           "(e.g. random-walk comparisons in Meese & Rogoff 1983; sign-test surveys).",
}


def _csv(path, **kw):
    return pd.read_csv(path, **kw) if Path(path).exists() else None


def _json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def _row(item, error, naive, skill, lit, comp, evidence):
    return {"item": item, "error": error, "naive_same_rows": naive, "skill": skill, "literature": lit,
            "comparability": comp, "evidence": evidence}


def _none(item, lit, evidence):
    return _row(item, "—", "—", math.nan, lit, "NO_NEW_MEASUREMENT", evidence)


def closure_rows(results: Path = RESULTS) -> list:
    rows = []
    # 1. price path (persistence is the naive itself; no real path predictor was scored this round)
    p = _csv(results / "predictors/predictores_resultados.csv")
    if p is not None and (p["mode"] == "path").any():
        for _, r in p[(p["mode"] == "path") & (p.cost_mult == 1)].drop_duplicates(["block_hours", "split"]).iterrows():
            rows.append(_row(f"Price path PERSISTENCE, B = {int(r.block_hours)} h, {r.split}",
                             f"path MAE {es(r.path_mae_mean * 1e4, 1)} pips (mean over horizons)",
                             "the same (persistence is the naive)", float(r.path_mae_ratio_mean) - 1.0, LIT["rw"],
                             "COMPARABLE: both say no skill over the random walk at <= 1 day",
                             "results/predictors/predictores_resultados.csv"))
    else:
        rows.append(_none("Price path predictors", LIT["rw"], "results/predictors/"))
    # 2. excursion / range predictors (calendar-range round)
    s = _csv(results / "calendar_range/scores.csv", dtype={"period": str})
    if s is not None:
        chosen = set()
        for B in (24, 6):
            ch = _json(results / f"calendar_range/CHOICE_B{B}.json")
            if ch:
                c = ch["chosen"]
                chosen.add((B, f"GBM_d{c['max_depth']}_n{c['max_iter']}"))
        d = s[(s.leverage_scenario == 100.0) & (s.direction == "coin")]
        d = d[d.model.isin(["HAR", "HARX"]) | pd.Series([(int(b), m) in chosen for b, m in zip(d.block_hours, d.model)], index=d.index)]
        for _, r in d.sort_values(["block_hours", "period", "model"]).iterrows():
            for side, col in (("range", "mae_r_ratio_mean20"), ("up excursion", "mae_u_ratio_mean20"),
                              ("down excursion", "mae_d_ratio_mean20")):
                ratio = float(r[col])
                extra = f" = {es(r.mae_r_pips, 1)} pips" if side == "range" else ""
                rows.append(_row(f"{r.model} {side}, B = {int(r.block_hours)} h, {r.period}",
                                 f"MAE {es(ratio, 3)} x MEAN20{extra}", "MEAN20 (1,000 x, same rows)", ratio - 1.0,
                                 LIT["har"],
                                 "COMPARABLE in kind (volatility-proxy forecast vs. a persistence-type naive); "
                                 "NOT_COMPARABLE in scale (block range/excursions, not daily RV; MAE not RMSE)",
                                 "results/calendar_range/scores.csv"))
        lg = s[(s.leverage_scenario == 100.0) & (s.direction == "logit")]
        for _, r in lg.sort_values(["block_hours", "period"]).drop_duplicates(["block_hours", "period"]).iterrows():
            acc = float(r.direction_accuracy)
            rows.append(_row(f"Logistic direction (first extreme), B = {int(r.block_hours)} h, {r.period}",
                             f"accuracy {es(acc, 3)}", "coin 0,500 (same rows)", (acc - 0.5) / 0.5, LIT["dir"],
                             "COMPARABLE: inside the literature's 0,50-0,56 band", "results/calendar_range/scores.csv"))
    else:
        rows.append(_none("Excursion / range predictors (HAR, HAR-X, GBM)", LIT["har"], "results/calendar_range/"))
    ev = _csv(results / "calendar_range/event_diagnostic.csv")
    if ev is not None:
        for B, g in ev.groupby("block_hours"):
            e = g[g.blocks == "event"].iloc[0]
            n = g[g.blocks == "no_event"].iloc[0]
            rows.append(_row(f"Event-day HAR range vs. no-event blocks, B = {int(B)} h (fixture 2020-01..2021-04)",
                             f"MAE {es(e.har_mae_r_ratio_mean20, 3)} x MEAN20 on event blocks (n = {int(e.n)})",
                             f"{es(n.har_mae_r_ratio_mean20, 3)} x MEAN20 on no-event blocks (n = {int(n.n)})",
                             (float(e.har_mae_r_ratio_mean20) - float(n.har_mae_r_ratio_mean20)) / float(n.har_mae_r_ratio_mean20),
                             LIT["ederington"], "COMPARABLE in direction only (events widen ranges and worsen HAR); descriptive",
                             "results/calendar_range/event_diagnostic.csv"))
    # 3. causal event study
    be = _json(results / "causal_event/BREAK_EVENS.json")
    rr = _csv(results / "causal_event/real_rule.csv")
    prep = _json(results / "causal_event/PREP.json")
    fam = _csv(results / "causal_event/families_decision.csv")
    if rr is not None and be is not None:
        u = rr[(rr.arm == "NEXT_BAR_UNGATED") & (rr.event_spread == "5pips") & (rr.leverage == 100.0) & (rr.stage != "decision")]
        for _, r in u.drop_duplicates("period").iterrows():
            req = be.get(r.period, {}).get("direction_accuracy")
            acc = float(r.direction_accuracy)
            rows.append(_row(f"Surprise-sign direction rule (ungated diagnostic), {r.period}",
                             f"accuracy {es(acc, 3)} (break-even {es(req, 2) if req else '—'})", "coin 0,500 (same rows)",
                             (acc - 0.5) / 0.5, LIT["abdv"],
                             "NOT_COMPARABLE: ABDV measure 5-minute conditional-mean jumps; here hourly bars after the release",
                             "results/causal_event/real_rule.csv, BREAK_EVENS.json"))
        g = rr[(rr.arm == "NEXT_BAR") & (rr.stage != "decision")]
        if not g.empty:
            rows.append(_row("Surprise-sign rule with the causal gate (NEXT_BAR), both periods",
                             f"0 trades ({int(fam.tradable.sum()) if fam is not None else 0} of {len(fam) if fam is not None else 0} families admitted; "
                             f"smallest p = {es(float(fam.p_value.min()), 3) if fam is not None else '—'})",
                             "—", math.nan, LIT["abdv"],
                             "NOT_COMPARABLE: no trade; literature effects are at 5-minute resolution",
                             "results/causal_event/families_decision.csv"))
    else:
        rows.append(_none("Causal event-window rule", LIT["abdv"], "results/causal_event/"))
    if prep is not None:
        rows.append(_row("Release-bar volatility ratio, NFP (hourly fixture)",
                         f"{es(prep['clock']['vol_ratio_peak_vs_other_hours'], 2)} x other hours of NFP days ({prep['clock']['days']} days); "
                         f"{es(prep['nfp_release_bar_vol_ratio_vs_previous_day'], 2)} x same hour one day earlier",
                         "1,00 (ratio to the normal level)", float(prep["nfp_release_bar_vol_ratio_vs_previous_day"]) - 1.0,
                         LIT["ederington"], "COMPARABLE in kind (event-study vol ratio); hourly vs. 5-minute resolution",
                         "results/causal_event/PREP.json"))
    # 4. volatility-only strategies (measured in an earlier round; pointer only)
    v = _json(results / "volatility_strategies/break_even_v1_v2.json")
    if v is not None:
        for strat in ("V1", "V2", "RFM_COIN"):
            hits = [x for x in v if x["strategy"] == strat and x["family"] == "a1r" and x["split"] == "VALIDATION_2024"]
            if not hits:
                continue
            x = hits[0]
            ideal = x.get("ideal_sharpe")
            rows.append(_row(f"{strat} ideal-information Sharpe, B = {int(x['block_hours'])} h, VALIDATION 2024 (A1r family)",
                             f"Sharpe {es(ideal, 2) if ideal is not None else 'ruined / invalid'}; break-even range ratio "
                             f"{es(x['be_rr'], 2) if x.get('be_rr') is not None else 'none'}; {es(x['ideal_trades_per_week'], 1)} trades/wk",
                             "Sharpe 0 (no information)", math.nan, "—",
                             "NO_NEW_MEASUREMENT this round (earlier round; strategy-level, no literature analogue)",
                             "results/volatility_strategies/break_even_v1_v2.json"))
    else:
        rows.append(_none("V1 / V2 volatility strategies", "—", "results/volatility_strategies/"))
    return rows


def markdown(rows) -> str:
    out = ["| Item | Error (model, scale) | Naive on the same rows | Skill (err − naive)/naive | Literature value and source | Comparability | Evidence |",
           "|---|---|---|---:|---|---|---|"]
    for r in rows:
        sk = "—" if r["skill"] is None or (isinstance(r["skill"], float) and math.isnan(r["skill"])) else es(r["skill"], 3)
        out.append(f"| {r['item']} | {r['error']} | {r['naive_same_rows']} | {sk} | {r['literature']} | {r['comparability']} | `{r['evidence']}` |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=RESULTS / "CLOSURE_TABLE_2026_10_19.md")
    a = ap.parse_args()
    rows = closure_rows(a.results)
    a.out.write_text("# Closure table 2026-10-19 (generated by tools/short_horizon_closure_table.py)\n\n"
                     "DEVELOPMENT. Skill < 0 means better than the naive. Literature magnitudes are approximate recalled values "
                     "with their sources; they were not reproduced here.\n\n" + markdown(rows))
    pd.DataFrame(rows).to_csv(a.out.with_suffix(".csv"), index=False)
    print(json.dumps({"rows": len(rows), "out": str(a.out.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
