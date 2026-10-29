"""Profile sanity tier report (QUALIFICATION_SPEC v2.1 D8): from a results folder of behaviour-matrix records, per
witness the PASS / FAIL / NOT_EVALUABLE counts under the governing forms (VALIDITY_SPEC v2), the exit-layer failures,
the horizon mix (executed shares, dominance flags), n_levels_clipped (must be 0 under full_f4), the sizing / engine
refusals by reason, the exec profile of every record and the trade summary (median net USD per record, net sigma per
trade, win rate, SL / TP pips). Writes TIER_REPORT.json and TIER_REPORT.md.

  python tools/strategy_tier_report.py --results <dir> --out <dir> [--expect-profile full_f4]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


TRADE_KEYS = ("direction", "sizing_q", "funding_reserve_per_unit", "loss_per_unit")


def trim(r: dict) -> dict:
    """What report() reads of one record (tier 2: 4 345 records, 3.8 GB of JSON, do not fit in memory whole): the trades
    reduced to their sizing fields, the oracle / O* readings reduced to the forms a DEFECT witness governs with."""
    if r.get("error"):
        return {"id": r.get("id"), "seed": r.get("seed"), "error": r["error"]}
    out = {k: r.get(k) for k in ("id", "seed", "validity", "non_fills", "exec_profile", "horizon_mix", "trade_summary", "closer_support", "w14", "PASS_effective")}
    ex = r.get("exit") or {}
    out["exit"] = {k: ex.get(k) for k in ("witnesses", "defects_by_Ostar", "z1_control", "PASS_effective", "failing_effective", "episode_summary", "episode_exit_mix", "closer_sweep")}
    res = r.get("results") or {}
    forms = {x.get("governing_form") for x in ((r.get("validity") or {}).get("per_witness") or {}).values()}
    keep = {"summary", "W0_activity", "W13_tpsl_track_sigma", "W8_early_close_at_switch_v3", "W14_mirror"} | forms
    out["results"] = {k: v for k, v in res.items() if k in keep}
    out["oracle"] = {k: v for k, v in (r.get("oracle") or {}).items() if k in forms}
    os_ = r.get("ostar") or {}
    out["ostar"] = {k: v for k, v in os_.items() if k in forms or k == "W8_early_close_at_switch_v3"}
    sz = {1: [], -1: []}                                              # the sizing fields as compact float32 arrays (memory: ~2.6 M trades in tier 2)
    for t in r.get("trades") or []:
        if "sizing_q" in t:
            sz[int(t["direction"])].append((t["sizing_q"], t["funding_reserve_per_unit"], t["loss_per_unit"]))
    out["_sizing"] = {d: np.asarray(v, np.float32).reshape(-1, 3) for d, v in sz.items()}
    return out


def load(d: str) -> list:
    out = []
    for p in sorted(Path(d).glob("CELL_*.json")):
        out.append(trim(json.loads(p.read_text())))
    return out


def report(recs: list, expect_profile: str | None) -> dict:
    wit = defaultdict(lambda: Counter())
    conflicts = []
    for r in recs:
        if r.get("error"):
            wit["ERROR"]["FAIL"] += 1
            continue
        for k, x in ((r.get("validity") or {}).get("per_witness") or {}).items():
            st = x["status"]
            if st == "DEFECT" and not x["pass"] and x.get("governing_form", k) != k:      # CONTROL_CONFLICT typed post hoc on records that predate it (ruling 2026-10-29)
                c = (r.get("ostar" if x.get("designated_control") == "O*" else "oracle") or {}).get(x["governing_form"])
                if isinstance(c, dict) and c.get("pass") is False:
                    st = "CONTROL_CONFLICT"
                    conflicts.append({"record": f"{r['id']}_s{r['seed']}", "witness": k, "form": x["governing_form"],
                                      "strategy": (r.get("results") or {}).get(x["governing_form"], {}).get("value"), "control": c.get("value")})
            if st == "CONTROL_CONFLICT":
                wit[k]["CONTROL_CONFLICT"] += 1
                continue
            wit[k]["NOT_EVALUABLE" if st == "NOT_EVALUABLE" else ("PASS" if x["pass"] else "FAIL")] += 1
            wit[k]["unvalidated_strict"] += int(st == "SEALED_UNVALIDATED")
            wit[k]["defect_corrected"] += int(st in ("DEFECT", "REPLAY_CONFLICT"))
        ex = r.get("exit") or {}
        for k, v in (ex.get("witnesses") or {}).items():
            if isinstance(v, dict) and "pass" in v:
                wit["exit:" + k]["PASS" if v["pass"] else "FAIL"] += 1
                wit["exit:" + k]["excused_by_Ostar"] += int(k in (ex.get("defects_by_Ostar") or []))
        w14 = (r.get("results") or {}).get("W14_mirror")
        if w14:
            wit["W14_mirror"]["NOT_EVALUABLE" if w14.get("status") == "NOT_EVALUABLE" else ("PASS" if w14.get("pass") else "FAIL")] += 1
    ok = [r for r in recs if not r.get("error")]
    eff = sum(1 for r in ok if r.get("PASS_effective"))
    clipped = [int((r.get("non_fills") or {}).get("levels_clipped") or 0) for r in ok]
    refused = Counter()
    for r in ok:
        for k, v in ((r.get("non_fills") or {}).get("engine_refused") or {}).items():
            if k.startswith("n_refused_") and v:
                refused[k[len("n_refused_"):]] += int(v)
    gate = Counter()
    for r in ok:
        for k, v in ((r.get("non_fills") or {}).get("gate") or {}).items():
            gate[k] += int(v)
    profiles = Counter((r.get("exec_profile") or {}).get("name", "unknown") for r in ok)
    hm = [r["horizon_mix"] for r in ok if r.get("horizon_mix")]
    executed = {h: float(np.nanmean([m["executed"].get(h, np.nan) for m in hm if m.get("n_executed")])) if any(m.get("n_executed") for m in hm) else float("nan") for h in ("h3", "h6", "h12", "h24")}
    flags = [r["id"] + f"_s{r['seed']}" for r in ok if (r.get("horizon_mix") or {}).get("flag_one_horizon_dominates_both_rules")]
    ts = [r["trade_summary"] for r in ok if r.get("trade_summary") and r["trade_summary"].get("n")]
    nets = [float(r["results"]["summary"]["net_usd"]) for r in ok if r.get("results", {}).get("summary")]
    sh = [float(x.get("sharpe", np.nan)) for r in ok for x in ((r.get("exit") or {}).get("closer_sweep") or []) if x.get("closer_quality") == 1.0]
    q = lambda key, i: float(np.nanmedian([t[key][i] for t in ts])) if ts else float("nan")  # noqa: E731
    # activity, early closes, per-family net in sigma per trade (record schema: never the compounded USD), W14 controls
    from tools.strategy_behaviour_matrix import cell_family
    opw = [float(r["results"]["W0_activity"]["value"]) for r in ok if isinstance((r.get("results") or {}).get("W0_activity"), dict)]
    ec = [(int(r["results"]["summary"]["early_closes"]), int(r["results"]["summary"]["opens"])) for r in ok if (r.get("results") or {}).get("summary")]
    fam = defaultdict(list)
    for r in ok:
        t = r.get("trade_summary") or {}
        if t.get("n"):
            fam[cell_family(r["id"])].append(float(t["net_sigma_per_trade_median"]))
    per_family = {k: {"records": len(v), "median_of_net_sigma_per_trade_medians": float(np.median(v))} for k, v in sorted(fam.items())}
    w14 = Counter((r.get("w14") or {}).get("control_status") for r in ok if r.get("w14"))
    w14_sym = [float((r.get("w14") or {}).get("code_symmetry_max_net_discrepancy_sigma") or 0.0) for r in ok if r.get("w14")]
    unmatched = sum(int((r.get("horizon_mix") or {}).get("n_unmatched_trades") or 0) for r in ok)
    w13 = Counter()
    for r in ok:
        x = (r.get("results") or {}).get("W13_tpsl_track_sigma")
        if isinstance(x, dict) and x.get("sealed_form"):
            w13["governing_" + ("PASS" if x["pass"] else "FAIL")] += 1
            w13["sealed_" + ("PASS" if x["sealed_form"].get("pass") else "FAIL")] += 1
    z1 = Counter((side, v.get("status")) for r in ok for side, v in ((r.get("exit") or {}).get("z1_control") or {}).items())
    w8b = Counter()
    for r in ok:
        x = (r.get("results") or {}).get("W8_early_close_at_switch_v3")
        if isinstance(x, dict):
            cb = (x.get("value") or {}).get("clause_b") or {}
            w8b["records"] += 1
            w8b["clause_b_" + str(cb.get("status"))] += 1
            w8b["not_evaluable_elements"] += int(cb.get("not_evaluable_zero_held_ticks") or 0)
        y = ((r.get("ostar") or {}).get("W8_early_close_at_switch_v3") or {}).get("value") or {}
        if y:
            w8b["Ostar_pass_with_sealed_clause_b_" + str(bool(y.get("pass_with_sealed_clause_b")))] += 1
    qs = {1: [], -1: []}
    for r in ok:
        if r.get("_sizing") is not None:
            for d_ in (1, -1):
                qs[d_].append(r["_sizing"][d_])
            continue
        for d_ in (1, -1):
            qs[d_].append(np.asarray([(float(t["sizing_q"]), float(t["funding_reserve_per_unit"]), float(t["loss_per_unit"])) for t in (r.get("trades") or [])
                                      if "sizing_q" in t and int(t["direction"]) == d_], np.float64).reshape(-1, 3))
    sizing_by_side = {}
    for d, v in qs.items():
        a = np.concatenate(v) if v else np.zeros((0, 3))
        sizing_by_side["BUY" if d > 0 else "SELL"] = ({"trades": int(len(a)), "q_p10_p50_p90": [float(np.percentile(a[:, 0], p_)) for p_ in (10, 50, 90)],
                                                      "share_q_below_1": float(np.mean(a[:, 0] < 1.0 - 1e-6)),
                                                      "funding_reserve_pips_p50_p90": [float(np.percentile(a[:, 1] * 1e4, p_)) for p_ in (50, 90)],
                                                      "loss_per_unit_pips_p50": float(np.median(a[:, 2] * 1e4))} if len(a) else {"trades": 0})
    eps_s, eps_o = [], []
    for r in ok:
        em = (r.get("exit") or {}).get("episode_exit_mix") or {}
        for si in ("0", "1"):
            a, b = (em.get("strategy") or {}).get(si) or {}, (em.get("O*") or {}).get(si) or {}
            if a.get("early") is not None and b.get("early") is not None:
                eps_s.append(float(a["early"])); eps_o.append(float(b["early"]))
    qq = lambda a: [float(np.nanpercentile(a, x)) for x in (10, 50, 90)] if a else []  # noqa: E731
    e_series = {"strategy_early_p10_p50_p90": qq(eps_s), "Ostar_early_p10_p50_p90": qq(eps_o),
                "max_abs_gap_to_Ostar": float(np.nanmax(np.abs(np.asarray(eps_s) - np.asarray(eps_o)))) if eps_s else float("nan")}
    sup = Counter(v.get("status") for r in ok for v in ((r.get("closer_support") or {}).values()))
    sup_passes = [int(v.get("extra_passes", 0)) for r in ok for v in ((r.get("closer_support") or {}).values())]
    out = {"records": len(recs), "errors": sum(1 for r in recs if r.get("error")), "pass_effective": eff, "fail_effective": len(ok) - eff,
           "exec_profiles": dict(profiles), "profile_expected": expect_profile, "profile_ok": (expect_profile is None or set(profiles) == {expect_profile}),
           "levels_clipped_total": int(sum(clipped)), "levels_clipped_records": sum(1 for c in clipped if c), "levels_clipped_ok": int(sum(clipped)) == 0,
           "engine_refusals_by_reason": dict(refused), "gate_counters": dict(gate),
           "horizon_mix_executed_mean": executed, "horizon_dominance_flags": flags, "records_with_executed_entries": sum(1 for m in hm if m.get("n_executed")),
           "witnesses": {k: dict(v) for k, v in sorted(wit.items())},
           "activity": {"trades_per_week_median": float(np.median(opw)) if opw else float("nan"), "trades_per_week_p10_p90": [float(np.percentile(opw, 10)), float(np.percentile(opw, 90))] if opw else [],
                        "early_close_share_pooled": (sum(a for a, _ in ec) / max(sum(b for _, b in ec), 1)) if ec else float("nan"),
                        "records_with_zero_trades": sum(1 for _, b in ec if b == 0)},
           "net_sigma_per_trade_by_family": per_family, "w14_control_status": dict(w14), "w14_code_symmetry_max_sigma": (max(w14_sym) if w14_sym else None),
           "horizon_unmatched_trades": unmatched, "control_conflicts": conflicts, "w13_columns": dict(w13), "z1_control": {f"{a}:{b}": n for (a, b), n in z1.items()}, "w8_clause_b": dict(w8b),
           "sizing_by_side": sizing_by_side, "e_series_episode_early_share": e_series,
           "closer_support": {"status": dict(sup), "records_extended": sum(1 for x in sup_passes if x > 0), "max_extra_passes": max(sup_passes) if sup_passes else 0},
           "trade_summary": {"records": len(ts), "median_net_usd_per_record": float(np.median(nets)) if nets else float("nan"), "net_sigma_per_trade_mean": float(np.nanmean([t["net_sigma_per_trade_mean"] for t in ts])) if ts else float("nan"),
                             "net_sigma_per_trade_median": float(np.nanmedian([t["net_sigma_per_trade_median"] for t in ts])) if ts else float("nan"), "win_rate_mean": float(np.nanmean([t["win_rate"] for t in ts])) if ts else float("nan"),
                             "trades_mean": float(np.mean([t["n"] for t in ts])) if ts else float("nan"), "sl_pips_median_of_medians": q("sl_pips_p10_p50_p90", 1), "tp_pips_median_of_medians": q("tp_pips_p10_p50_p90", 1),
                             "hold_bars_median_of_medians": q("hold_bars_p10_p50_p90", 1), "sharpe_mean": float(np.nanmean(sh)) if sh else float("nan")}}
    return out


def markdown(d: dict) -> str:
    L = ["# Profile sanity tier report", "", f"Records {d['records']} (errors {d['errors']}); PASS effective {d['pass_effective']}, FAIL {d['fail_effective']}; exec profiles {d['exec_profiles']} (expected {d['profile_expected']}: {'OK' if d['profile_ok'] else 'MISMATCH'}); "
         f"levels_clipped total {d['levels_clipped_total']} in {d['levels_clipped_records']} records ({'OK' if d['levels_clipped_ok'] else 'NOT ZERO'}).", "",
         "## Engine refusals by reason", "", "| reason | count |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in sorted(d["engine_refusals_by_reason"].items())] + [
         "", f"Gate counters: {d['gate_counters']}", "", "## Horizon mix (executed, mean over records with entries)", "",
         f"{d['horizon_mix_executed_mean']} over {d['records_with_executed_entries']} records; dominance flags: {d['horizon_dominance_flags'] or 'none'}", "",
         "## Activity", "", f"{d['activity']}", "", "## Net per family (median of the per-record median net sigma per trade)", "", "| family | records | median net sigma / trade |", "|---|---:|---:|"] + [
         f"| {k} | {v['records']} | {v['median_of_net_sigma_per_trade_medians']:.3f} |" for k, v in d["net_sigma_per_trade_by_family"].items()] + [
         "", f"W14 control status: {d['w14_control_status']}; swapped-reference code symmetry max {d['w14_code_symmetry_max_sigma']} sigma; trades without a matched menu slot: {d['horizon_unmatched_trades']}", "",
         f"Sizing by side (q, funding reserve in pips, loss per unit in pips): {d['sizing_by_side']}", "",
         f"E-series episode early-close share, strategy vs O*: {d['e_series_episode_early_share']}", "",
         f"Control conflicts (CONTROL_CONFLICT, never PASS): {d['control_conflicts']}", "",
         f"W13 columns (menu form governing / sealed form): {d['w13_columns']}; Z1 control: {d['z1_control']}; W8 clause (b): {d['w8_clause_b']}; closer support: {d['closer_support']}", "",
         "## Witnesses (governing forms)", "", "| witness | PASS | FAIL | NOT_EVALUABLE | CONTROL_CONFLICT | sealed-unvalidated (strict) | defect-corrected | excused by O* |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in d["witnesses"].items():
        L.append(f"| {k} | {v.get('PASS', 0)} | {v.get('FAIL', 0)} | {v.get('NOT_EVALUABLE', 0)} | {v.get('CONTROL_CONFLICT', 0)} | {v.get('unvalidated_strict', 0)} | {v.get('defect_corrected', 0)} | {v.get('excused_by_Ostar', 0)} |")
    t = d["trade_summary"]
    L += ["", "## Trade summary (record schema)", "", f"records {t['records']}; net sigma / trade mean {t['net_sigma_per_trade_mean']:.3f} (median {t['net_sigma_per_trade_median']:.3f}); win rate {t['win_rate_mean']:.3f}; trades / record {t['trades_mean']:.0f}; "
          f"SL pips median {t['sl_pips_median_of_medians']:.1f}; TP pips median {t['tp_pips_median_of_medians']:.1f}; hold bars median {t['hold_bars_median_of_medians']:.1f}; Sharpe mean {t['sharpe_mean']:.2f}", ""]
    if d.get("notes"):
        L += ["## Notes", ""] + [f"- {x}" for x in d["notes"]] + [""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-profile", default=None)
    ap.add_argument("--note", action="append", default=[], help="a declared note recorded in the report (repeatable)")
    a = ap.parse_args()
    d = report(load(a.results), a.expect_profile)
    d["notes"] = list(a.note)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "TIER_REPORT.json").write_text(json.dumps(d, indent=1, default=float))
    (out / "TIER_REPORT.md").write_text(markdown(d))
    print(markdown(d))


if __name__ == "__main__":
    main()
