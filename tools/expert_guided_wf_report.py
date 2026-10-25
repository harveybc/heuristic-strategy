#!/usr/bin/env python3
"""Reports with uncertainty for the clean round (owner 2026-10-25: "tabla con errores estándar, no sólo medias").

  b-ensemble     every finished arm of stages A / A_prime / B (plus seed extensions): per-seed Sharpe, 3/5-seed mean, the
                 seed-ensemble replay with its Lo SE and bootstrap interval (ENSEMBLE.json written by
                 tools/expert_guided_seed_ensemble.py), failed arms with their recorded cause, the 5-seed decision and
                 the Stage B ranking; owner closure table (control = same rows).
  wf-v1          stage WF_V1 (action contract v1): per fold, each v1 arm next to its baseline (the WF arm of the same base);
                 HOLDS / SEPARABLE on the INNER_B ensembles; INVALID when the ensemble trades < 2 per week on INNER_B.
  walk-forward   per fold x arm: per-seed and ensemble Sharpe on TRAIN / INNER_A / INNER_B (validation year, scored after
                 freeze) with SE and bootstrap; the PRE-REGISTERED verdicts (stats.HOLDS_RULE / SEPARABLE_RULE); 2024 one
                 read per unit; owner closure table (naive = control zz_hold on the same rows).

Every figure is a behaviour-cloning student replayed under the replay contract on development data: not a tradable
strategy, not a profitability claim. The hindsight labels enter training only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import stats as ST  # noqa: E402

CAVEAT = ("behaviour-cloning students replayed under the replay contract (1 entry/day, 1:50, >= 2 orders/week validity) on development "
          "data; hindsight labels enter training only; not a tradable strategy or a profitability claim; TEST (2025) never read")
LIT = "NOT_COMPARABLE: behaviour-cloning student under a private replay contract on EURUSD H1; no published value with this contract"


def fmt(x, f="{:.2f}"):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f.format(x)


def unit_doc(out: Path, unit: str, extra_units: list | None = None) -> dict:
    """One arm: E3M.json (+ pooled seed extensions) and its ENSEMBLE.json if present; FAILED with cause otherwise."""
    d = out / unit
    e3 = d / "E3M.json"
    if not e3.exists():
        cause = "no E3M.json"
        log = out / f"{unit}.log"
        if log.exists():
            txt = log.read_text(errors="replace")
            for key in ("E3M_RECIPE_CONFLICT", "MULTIRES_SPEC_NEEDS_PARAMS", "PRESSURE_STOP", "REFUSED", "Traceback"):
                if key in txt:
                    line = next((ln for ln in txt.splitlines() if key in ln), key)
                    cause = line.strip()[:200]
                    break
        return {"unit": unit, "status": "FAILED", "cause": cause}
    e = json.loads(e3.read_text())
    tr = dict(e["trials"]["FULL"])
    for u in extra_units or []:
        ep = json.loads((out / u / "E3M.json").read_text())
        tr.update(ep["trials"]["FULL"])
    periods = [p for p in ("TRAIN", "INNER_A", "INNER_B", "OBS_2024") if any(p in tr[s]["trading"] for s in tr)]
    seeds = {s: {p: {"sharpe": tr[s]["trading"][p]["sharpe_daily_sqrt365"], "dd": tr[s]["trading"][p]["max_drawdown_pct"],
                     "opw": tr[s]["trading"][p]["orders_per_week"], "win": tr[s]["trading"][p]["win_rate"], "valid": bool(tr[s]["trading"][p]["valid"])}
                 for p in periods if p in tr[s]["trading"]} for s in tr}
    ens_p = d / "ENSEMBLE.json"
    ens = json.loads(ens_p.read_text()) if ens_p.exists() else None
    doc = {"unit": unit, "status": "DONE", "seeds": seeds, "n_seeds": len(seeds), "periods": periods, "params": sum(e["params"]["FULL"].values()),
           "recipe": e.get("recipe"), "split": e["meta"]["split_contract"]["arm"], "support_hash": e["meta"]["support_hash"],
           "labels_mode": (e.get("labels") or {}).get("mode", "zigzag"), "aux_mode": e.get("aux_mode") or (e.get("aux_targets") or {}).get("mode") if isinstance(e.get("aux_targets"), dict) else e.get("aux_mode"),
           "agent_state": (e.get("agent_state") or {}).get("mode"), "wall_seconds": e.get("wall_seconds"), "ensemble": None}
    for p in periods:
        sh = [v[p]["sharpe"] for v in seeds.values() if p in v]
        doc.setdefault("seed_mean", {})[p] = st.mean(sh) if sh else None
        doc.setdefault("seed_std", {})[p] = (st.pstdev(sh) if len(sh) > 1 else 0.0) if sh else None
        doc.setdefault("all_valid", {})[p] = all(v[p]["valid"] for v in seeds.values() if p in v)
    if ens:
        doc["ensemble"] = {p: {"sharpe": v["sharpe"], "se_lo": v["se_lo"], "sharpe_minus_se": v["sharpe_minus_se"], "days": v["days"],
                               "ci95": [v["bootstrap"]["low"], v["bootstrap"]["high"]], "dd": v["metrics"]["max_drawdown_pct"],
                               "opw": v["metrics"]["orders_per_week"], "win": v["metrics"]["win_rate"], "valid": bool(v["metrics"]["valid"])}
                           for p, v in ens["ensemble"].items()}
        doc["ensemble_arbiter"] = ens["arbiter"]
        doc["ensemble_seeds"] = ens["seeds"]
        doc["ensemble_verification"] = ens["verification"]
        doc["per_seed_se"] = {s: {p: {"se_lo": v["se_lo"], "ci95": [v["bootstrap"]["low"], v["bootstrap"]["high"]]} for p, v in sv.items()} for s, sv in ens["per_seed"].items()}
    return doc


def closure_rows(units: list, control: str, period: str, label: str) -> list:
    """Owner closure table on one window: metric + scale, naive = the control arm on the same rows, skill with its pooled
    SE, literature (NOT_COMPARABLE typed), comparability. Ensemble figures when present, seed means otherwise."""
    by = {u["unit"]: u for u in units if u["status"] == "DONE"}
    c = by.get(control)
    rows = []
    for u in units:
        if u["status"] != "DONE" or u["unit"] == control:
            continue
        if c is None or period not in u.get("seed_mean", {}) or period not in c.get("seed_mean", {}):
            rows.append({"unit": u["unit"], "period": period, "metric": "Sharpe daily sqrt365 / max DD %", "model": "NO_NEW_MEASUREMENT", "naive": "control missing on this window",
                         "skill": "NOT_COMPARABLE: no control on the same rows", "literature": LIT, "comparability": label})
            continue
        if u.get("ensemble") and c.get("ensemble") and period in u["ensemble"] and period in c["ensemble"]:
            a, b = u["ensemble"][period], c["ensemble"][period]
            sep = ST.separable({"sharpe": a["sharpe"], "se_lo": a["se_lo"]}, {"sharpe": b["sharpe"], "se_lo": b["se_lo"]})
            model = f"ensemble Sharpe {fmt(a['sharpe'])} ± {fmt(a['se_lo'])} (SE) [95 % {fmt(a['ci95'][0])}, {fmt(a['ci95'][1])}] / DD {fmt(a['dd'], '{:.1f}')} % / {fmt(a['opw'], '{:.1f}')} orders/wk / valid {a['valid']}"
            naive = f"control {control} ensemble: Sharpe {fmt(b['sharpe'])} ± {fmt(b['se_lo'])} / DD {fmt(b['dd'], '{:.1f}')} %"
            skill = f"delta {sep['delta']:+.2f} vs pooled SE {fmt(sep['pooled_se'])}: {sep['verdict']}"
        else:
            ms, ns = u["seed_mean"][period], c["seed_mean"][period]
            model = f"seed-mean Sharpe {fmt(ms)} ± {fmt(u['seed_std'][period])} (seed std, n={u['n_seeds']})"
            naive = f"control {control}: seed-mean Sharpe {fmt(ns)} ± {fmt(c['seed_std'][period])}"
            skill = f"delta {ms - ns:+.2f} (seed std, no ensemble SE available)"
        rows.append({"unit": u["unit"], "period": period, "metric": "Sharpe daily sqrt365 of the student replay (dimensionless) / max DD %",
                     "model": model, "naive": naive, "skill": skill, "literature": LIT,
                     "comparability": f"same recipe/support ({u['support_hash'].get(period)}), same replay contract and costs; {label}"})
    return rows


def table_md(units: list, periods: tuple, title: str) -> list:
    md = [f"### {title}", "",
          "| unit | status | n seeds | " + " | ".join(f"{p} seeds" for p in periods) + " | " + " | ".join(f"{p} mean ± seed σ" for p in periods) + " | " +
          " | ".join(f"{p} ensemble ± SE [95 %]" for p in periods) + " | valid | params |",
          "| --- | --- | ---: | " + " | ".join("---" for _ in periods) + " | " + " | ".join("---" for _ in periods) + " | " + " | ".join("---" for _ in periods) + " | --- | ---: |"]
    for u in units:
        if u["status"] != "DONE":
            md.append(f"| {u['unit']} | **{u['status']}** ({u.get('cause', '')}) | | " + " | ".join("" for _ in periods) * 3 + " | | |")
            continue
        seeds_cells = ["/".join(fmt(u["seeds"][s][p]["sharpe"]) for s in u["seeds"] if p in u["seeds"][s]) for p in periods]
        mean_cells = [f"{fmt(u['seed_mean'].get(p))} ± {fmt(u['seed_std'].get(p))}" for p in periods]
        ens_cells = [(f"{fmt(u['ensemble'][p]['sharpe'])} ± {fmt(u['ensemble'][p]['se_lo'])} [{fmt(u['ensemble'][p]['ci95'][0])}, {fmt(u['ensemble'][p]['ci95'][1])}]"
                      if u.get("ensemble") and p in u["ensemble"] else "—") for p in periods]
        valid = all(u["all_valid"].get(p, True) for p in periods if p != "OBS_2024")
        md.append(f"| {u['unit']} | DONE | {u['n_seeds']} | " + " | ".join(seeds_cells) + " | " + " | ".join(mean_cells) + " | " + " | ".join(ens_cells) + f" | {valid} | {u['params']} |")
    return md + [""]


def closure_md(rows: list) -> list:
    md = ["| unit | period | metric | model | naive, same rows | skill | literature | comparability |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r['unit']} | {r['period']} | {r['metric']} | {r['model']} | {r['naive']} | {r['skill']} | {r['literature']} | {r['comparability']} |" for r in rows]
    return md + [""]


def cmd_b_ensemble(a):
    out = Path(a.results)
    spec = json.loads(Path(a.spec).read_text())
    pooled = {x: [f"A_{x}{spec['seed_extension']['out_suffix']}"] for x in spec.get("seed_extension", {}).get("arms", [])}
    units = []
    for stg in ("A", "A_prime", "B"):
        for arm in spec["stages"].get(stg, []):
            u = f"{stg}_{arm['name']}"
            units.append(unit_doc(out, u, pooled.get(arm["name"]) if stg == "A" else None))
    dec5 = json.loads(Path(a.decision_5seed).read_text()) if a.decision_5seed and Path(a.decision_5seed).exists() else None
    control = f"A_{spec['decision_control']}"
    done_b = [u for u in units if u["unit"].startswith("B_") and u["status"] == "DONE"]
    inherited = f"A_{dec5['chosen']}" if dec5 else None
    rank_b = sorted(done_b, key=lambda u: -(u["seed_mean"].get("INNER_A") or -1e9))
    doc = {"caveat": CAVEAT, "spec": spec["identity"], "units": units, "decision_5seed": dec5, "inherited_winner_unit": inherited,
           "stage_b_rule": spec.get("decision_b"), "stage_b_order": [u["unit"] for u in rank_b],
           "stage_b_chosen": (rank_b[0]["unit"] if rank_b else None),
           "closure": {"INNER_A": closure_rows(units, control, "INNER_A", "INNER_A = inner selection block of the consolidated contract (decides)"),
                       "OBS_2024": closure_rows(units, control, "OBS_2024", "2024 = ONE read per arm, never decides")},
           "rules": {"holds": ST.HOLDS_RULE, "separable": ST.SEPARABLE_RULE}}
    # stage B vs its inherited parent, with SE when both ensembles exist
    if inherited:
        par = next((u for u in units if u["unit"] == inherited and u["status"] == "DONE"), None)
        if par and par.get("ensemble"):
            doc["stage_b_vs_parent"] = {u["unit"]: ST.separable({"sharpe": u["ensemble"]["INNER_A"]["sharpe"], "se_lo": u["ensemble"]["INNER_A"]["se_lo"]},
                                                                 {"sharpe": par["ensemble"]["INNER_A"]["sharpe"], "se_lo": par["ensemble"]["INNER_A"]["se_lo"]})
                                        for u in done_b if u.get("ensemble") and "INNER_A" in u["ensemble"]}
    Path(a.out_json).write_text(json.dumps(doc, indent=1, default=float))
    md = [f"# Resultados etapa B y ensamble de semillas con errores estándar (spec `{spec['identity']}`, {a.date})", "",
          "Replays de estudiantes (clonación de comportamiento) bajo el contrato de replay sobre datos de desarrollo; etiquetas hindsight sólo en el "
          "entrenamiento; **no es una estrategia operable ni una medida de rentabilidad**; TEST cerrado. INNER_A decide; 2024 una lectura por brazo.", "",
          "Ensamble = media de probabilidades de las N semillas por módulo, un solo replay; SE = error estándar de Lo (2002) del Sharpe anualizado; "
          "[95 %] = bootstrap estacionario (bloque 5 días, 2 000 remuestreos).", ""]
    if dec5:
        md += ["## Decisión a 5 semillas (regla pre-registrada)", "", f"Elegido: **{dec5['chosen']}**; " + "; ".join(f"{r['arm']} media INNER_A {r['inner_a_sharpe_mean']:.3f} ({r['n_seeds']} semillas, válidas {r['all_valid']})" for r in dec5["arms"]), ""]
    md += table_md([u for u in units if u["unit"].startswith("A_") and not u["unit"].startswith("A_prime")], ("TRAIN", "INNER_A", "OBS_2024"), "Etapa A (las dos `both` con 5 semillas agrupadas)")
    md += table_md([u for u in units if u["unit"].startswith("A_prime")], ("TRAIN", "INNER_A", "OBS_2024"), "Etapa A' (multires como etiqueta de clase)")
    md += table_md([u for u in units if u["unit"].startswith("B_")], ("TRAIN", "INNER_A", "OBS_2024"), f"Etapa B (hereda {inherited})")
    if doc.get("stage_b_vs_parent"):
        md += ["**Etapa B frente a su padre (ensamble, INNER_A):** " + "; ".join(f"{k}: Δ {v['delta']:+.2f} vs SE agrupado {v['pooled_se']:.2f} → {v['verdict']}" for k, v in doc["stage_b_vs_parent"].items()), ""]
    md += ["## Tabla de cierre (regla del propietario)", ""] + closure_md(doc["closure"]["INNER_A"]) + closure_md(doc["closure"]["OBS_2024"])
    md += ["## Reglas pre-registradas", "", f"- {ST.HOLDS_RULE}", f"- {ST.SEPARABLE_RULE}", ""]
    Path(a.out_md).write_text("\n".join(md))
    print(json.dumps({"units": [(u["unit"], u["status"]) for u in units], "stage_b_chosen": doc["stage_b_chosen"], "out": a.out_md}))


def cmd_walk_forward(a):
    out = Path(a.results)
    spec = json.loads(Path(a.spec).read_text())
    wf = spec["walk_forward"]
    units = []
    for fold in wf["folds"]:
        for arm in wf["arms"]:
            units.append(unit_doc(out, f"WF_{fold}_{arm}") | {"fold": fold, "arm": arm})
    verdicts = {}
    for arm in wf["arms"]:
        per = {}
        for fold in wf["folds"]:
            u = next(x for x in units if x["unit"] == f"WF_{fold}_{arm}")
            if u["status"] == "DONE" and u.get("ensemble") and "INNER_B" in u["ensemble"]:
                per[fold] = {"sharpe": u["ensemble"]["INNER_B"]["sharpe"], "se_lo": u["ensemble"]["INNER_B"]["se_lo"]}
        verdicts[arm] = (ST.holds(per) if len(per) == len(wf["folds"]) else {"rule": ST.HOLDS_RULE, "per_fold": {}, "verdict": "NO_NEW_MEASUREMENT: a fold is missing or has no ensemble"}) | {"inputs": per}
    control = spec["decision_control"]
    sep = {}
    for fold in wf["folds"]:
        c = next((x for x in units if x["unit"] == f"WF_{fold}_{control}" and x["status"] == "DONE" and x.get("ensemble")), None)
        for arm in wf["arms"]:
            if arm == control:
                continue
            u = next((x for x in units if x["unit"] == f"WF_{fold}_{arm}" and x["status"] == "DONE" and x.get("ensemble")), None)
            if c and u and "INNER_B" in u["ensemble"] and "INNER_B" in c["ensemble"]:
                sep[f"{fold}/{arm}_vs_{control}"] = ST.separable({"sharpe": u["ensemble"]["INNER_B"]["sharpe"], "se_lo": u["ensemble"]["INNER_B"]["se_lo"]},
                                                                  {"sharpe": c["ensemble"]["INNER_B"]["sharpe"], "se_lo": c["ensemble"]["INNER_B"]["se_lo"]})
    closure = {}
    for fold in wf["folds"]:
        fu = [x for x in units if x["fold"] == fold]
        closure[fold] = closure_rows([dict(x, unit=x["unit"]) for x in fu], f"WF_{fold}_{control}", "INNER_B", f"{fold}: validation year scored after freeze (no choice made on it)")
    doc = {"caveat": CAVEAT, "spec": spec["identity"], "rule_preregistered": wf.get("rule"), "units": units, "verdicts": verdicts, "separability": sep, "closure": closure,
           "rules": {"holds": ST.HOLDS_RULE, "separable": ST.SEPARABLE_RULE}}
    Path(a.out_json).write_text(json.dumps(doc, indent=1, default=float))
    md = [f"# Walk-forward dentro del desarrollo: resultados con errores estándar (spec `{spec['identity']}`, {a.date})", "",
          "Dos pliegues: `wf2022` ajusta 2020-01→2022-01 (bloque interno 2021 S2) y puntúa **2022 entero** tras congelar; `wf2023` ajusta 2020-01→2023-01 (interno 2022 S2) y "
          "puntúa **2023 entero**. Constantes del escalador re-ajustadas por pliegue (receta derivada, identidad propia); misma representación, modelo, regularización y contrato. "
          "Replays de estudiantes; **no es una estrategia operable**; 2024 una lectura por unidad; TEST cerrado.", "",
          f"Regla pre-registrada (antes de leer): {wf.get('rule')}", ""]
    for fold in wf["folds"]:
        md += table_md([x for x in units if x["fold"] == fold], ("TRAIN", "INNER_A", "INNER_B", "OBS_2024"), f"Pliegue {fold} (INNER_B = año de validación)")
    md += ["## Veredictos", ""] + [f"- **{arm}**: {v['verdict']} — " + ", ".join(f"{f}: {fmt(x['sharpe'])} − SE {fmt(x['se_lo'])}" for f, x in v["inputs"].items()) for arm, v in verdicts.items()] + [""]
    if sep:
        md += ["## Separabilidad frente al control (ensamble, año de validación)", ""] + [f"- {k}: Δ {v['delta']:+.2f} vs SE agrupado {fmt(v['pooled_se'])} → {v['verdict']}" for k, v in sep.items()] + [""]
    md += ["## Tabla de cierre (regla del propietario)", ""]
    for fold in wf["folds"]:
        md += closure_md(closure[fold])
    Path(a.out_md).write_text("\n".join(md))
    print(json.dumps({"units": [(u["unit"], u["status"]) for u in units], "verdicts": {k: v["verdict"] for k, v in verdicts.items()}, "out": a.out_md}))


def v1_details(out: Path, unit: str) -> dict:
    """Per seed of a v1 unit: the NOP-veto decision, the calibrated ECE per specialist on the calibration block, the NOP
    information test and the out-of-sample activity (reported, never a choice)."""
    e3 = out / unit / "E3M.json"
    if not e3.exists():
        return {}
    e = json.loads(e3.read_text())
    det = {}
    for s_, rec in e["trials"]["FULL"].items():
        ac = rec.get("action_contract") or {}
        det[s_] = {"nop_veto_kept": (ac.get("strategy") or {}).get("nop_veto_kept"), "strategy": (ac.get("strategy") or {}).get("strategy"),
                   "ece_after": {m: c["after"]["ece"] for m, c in (ac.get("calibration") or {}).items()},
                   "nop_delta_logloss": (ac.get("nop_information_calibration_block") or {}).get("delta_logloss"),
                   "inner_b_opw": (ac.get("out_of_sample_activity") or {}).get("INNER_B_trades_per_week")}
    return det


def fitness_rows(out: Path, w: dict) -> tuple:
    """One row per (arm, fitness variant, fold) from the v1 ENSEMBLE.json files, and the pre-registered variant rule:
    chosen on wf2022's decision year (highest ensemble Sharpe - 1 SE among variants valid there with >= 2 trades/week),
    confirmed on wf2023 (Sharpe - 1 SE > 0 with >= 2 trades/week) or refuted; every variant is reported."""
    rows, rule = [], {}
    folds = list(w["folds"])
    for arm in w["arms"]:
        per = {}
        for fold in folds:
            ep = out / f"WF_V1_{fold}_{arm['name']}" / "ENSEMBLE.json"
            ens = json.loads(ep.read_text()) if ep.exists() else {}
            sc = (ens.get("ensemble_strategy") or {}).get("term_scales")
            if sc:
                rule.setdefault("_scales", {})[f"{fold}/{arm['name']}"] = sc
            for f in w.get("fitness_variants", []):
                bf = (ens.get("by_fitness") or {}).get(f)
                ib = (bf or {}).get("windows", {}).get("INNER_B")
                if ib is None:
                    rows.append({"arm": arm["name"], "fitness": f, "fold": fold, "status": "NO_NEW_MEASUREMENT"})
                    continue
                opw = ib["metrics"]["orders_per_week"]
                valid = bool(ib["metrics"]["valid"] and opw >= 2.0)
                av = bf.get("archive_view") or {}
                bu = av.get("best_unconstrained_by_fitness") or {}
                bub = (bu.get("windows_not_selectable") or {}).get("INNER_B")
                bs = av.get("best_sharpe_unconstrained") or {}
                bv = av.get("best_valid_by_fitness") or {}
                r = {"arm": arm["name"], "fitness": f, "fold": fold, "status": "DONE", "sharpe": ib["sharpe"], "se_lo": ib["se_lo"],
                     "lcb": ib["sharpe_minus_se"], "opw": opw, "dd": ib["metrics"]["max_drawdown_pct"], "valid": valid,
                     "verdict": "VALID" if valid else "INVALID (owner's rule: < 2 trades/week on INNER_B)",
                     "strategy": bf.get("strategy"), "valid_on_calibration": bf.get("valid_on_calibration"),
                     "chosen_calibration": bf.get("chosen_components"),
                     "archive": {"share_valid": av.get("share_valid"), "points": av.get("points"),
                                 "best_sharpe_unconstrained_calibration": bs.get("sharpe"), "best_sharpe_unconstrained_opw": bs.get("opw"),
                                 "best_valid_by_fitness_calibration": {"fitness": bv.get("fitness"), "sharpe": bv.get("sharpe"), "opw": bv.get("opw")} if bv else None,
                                 "best_unconstrained_by_fitness_calibration": {"fitness": bu.get("fitness"), "sharpe": bu.get("sharpe"), "opw": bu.get("opw"), "valid": bu.get("valid")} if bu else None,
                                 "best_unconstrained_INNER_B_not_selectable": None if bub is None else {"sharpe": bub["sharpe"], "se_lo": bub["se_lo"], "opw": bub["metrics"]["orders_per_week"],
                                                                                                         "dd": bub["metrics"]["max_drawdown_pct"], "label": "NOT SELECTABLE"}}}
                rows.append(r)
                per.setdefault(fold, {})[f] = r
        c0, c1 = (folds + [None, None])[:2]
        cand = [r for r in (per.get(c0) or {}).values() if r["valid"] and math.isfinite(r["lcb"])]
        if not cand:
            rule[arm["name"]] = {"chosen": None, "verdict": f"NO_VALID_VARIANT_ON_{c0}"}
            continue
        best = max(cand, key=lambda r: r["lcb"])
        conf = (per.get(c1) or {}).get(best["fitness"])
        ok = bool(conf and conf["valid"] and math.isfinite(conf["lcb"]) and conf["lcb"] > 0)
        rule[arm["name"]] = {"chosen": best["fitness"], "chosen_on": c0, "lcb_chosen_fold": best["lcb"], "confirm_fold": c1,
                             "lcb_confirm_fold": None if not conf else conf["lcb"], "opw_confirm_fold": None if not conf else conf["opw"],
                             "verdict": "CONFIRMED" if ok else "REFUTED"}
    return rows, rule


EVIDENCE_KEYS = ("D1", "D2", "D3", "D4", "D5", "E")


def _mean(xs):
    xs = [float(x) for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))]
    return st.mean(xs) if xs else None


def closure_evidence(out: Path, w: dict) -> dict:
    """The evidence column of OUTPUT_CONTRACT_V1_CLOSURE.md, from the WF_V1 artifacts (E3M.json per seed, ENSEMBLE.json)."""
    per = {k: [] for k in EVIDENCE_KEYS}
    opt, decay, nop_kept, nop_n, ens_kept, dll, ece_b, ece_a, br_b, br_a, ib_ece, front_opw, share_v, ib_opw, mae_tp, mae_sl, cov_hi, audits, shapes = \
        [], [], 0, 0, [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    nop_auc = []
    for fold in w["folds"]:
        for arm in w["arms"]:
            u = f"WF_V1_{fold}_{arm['name']}"
            e3 = out / u / "E3M.json"
            if not e3.exists():
                continue
            e = json.loads(e3.read_text())
            ac = e.get("action_contract") or {}
            audits.append((ac.get("encoding_audit") or {}).get("verdict"))
            shapes.append(all(v.get("status") == "MATCH" for v in (ac.get("head_shapes") or {}).values()) if ac.get("head_shapes") else None)
            for s_, rec in e["trials"]["FULL"].items():
                x = rec.get("action_contract") or {}
                stg = x.get("strategy") or {}
                ch = stg.get("chosen")
                fr = ((stg.get("runs") or {}).get("with_veto" if stg.get("nop_veto_kept") else "without_veto") or {}).get("front") or []
                if ch and fr:
                    opt.append(ch["sharpe"] - st.median([p["sharpe"] for p in fr]))
                    front_opw += [p["opw"] for p in fr]
                ib = (rec.get("trading") or {}).get("INNER_B")
                if ch and ib:
                    decay.append(ch["sharpe"] - ib["sharpe_daily_sqrt365"])
                for r in (stg.get("runs") or {}).values():
                    share_v.append(r.get("share_valid"))
                nop_n += 1
                nop_kept += int(bool(stg.get("nop_veto_kept")))
                dll.append((x.get("nop_information_calibration_block") or {}).get("delta_logloss"))
                cal = x.get("calibration") or {}
                if "nop" in cal:
                    nop_auc.append(cal["nop"]["after"]["auc"])
                for m, c in cal.items():
                    ece_b.append(c["before"]["ece"]); ece_a.append(c["after"]["ece"]); br_b.append(c["before"]["brier"]); br_a.append(c["after"]["brier"])
                for m, c in ((x.get("reported_only_INNER_B") or {}).get("heads") or {}).items():
                    ib_ece.append(c["calibrated"]["ece"])
                for m, t in (x.get("tp_sl_calibration_block") or {}).items():
                    mae_tp.append(t.get("tp_mae_sigma")); mae_sl.append(t.get("sl_mae_sigma"))
                    cov_hi.append(((t.get("bound_coverage") or {}).get("tp") or {}).get("share_at_hi"))
            ep = out / u / "ENSEMBLE.json"
            if ep.exists():
                en = json.loads(ep.read_text())
                ens_kept.append(bool((en.get("arbiter") or {}).get("nop_veto_kept")))
                ib_opw.append(((en.get("out_of_sample_activity") or {}).get("INNER_B_trades_per_week")))
    f = lambda x, p="{:.3f}": fmt(x, p)  # noqa: E731
    ev = {
        "D1": (f"optimismo re-medido en el bloque de calibración (elegido − mediana del frente válido): {f(_mean(opt), '{:+.2f}')} (auditoría: +1.08); "
               f"Sharpe elegido en INNER_A − Sharpe en INNER_B (nunca elegido allí): {f(_mean(decay), '{:+.2f}')} sobre {len(decay)} semillas; prueba de envenenamiento: ver la suite"),
        "D2": (f"veto NOP conservado en {nop_kept}/{nop_n} semillas y {sum(ens_kept)}/{len(ens_kept)} ensambles; AUC del NOP calibrado en INNER_A {f(_mean(nop_auc))}; "
               f"Δ log-loss medio con pNOP {f(_mean(dll), '{:+.4f}')}"),
        "D3": (f"ECE medio {f(_mean(ece_b))} → {f(_mean(ece_a))} y Brier {f(_mean(br_b))} → {f(_mean(br_a))} en INNER_A (antes → después de Platt); "
               f"ECE calibrado en INNER_B (sólo reportado) {f(_mean(ib_ece))}; tablas de fiabilidad de 15 bins en E3M.json y ENSEMBLE.json"),
        "D4": (f"órdenes/semana en los frentes válidos: mín {f(min(front_opw) if front_opw else None, '{:.2f}')}, mediana {f(st.median(front_opw) if front_opw else None, '{:.2f}')}, "
               f"máx {f(max(front_opw) if front_opw else None, '{:.2f}')}; % de puntos válidos en el archivo {f(100 * _mean(share_v) if _mean(share_v) is not None else None, '{:.0f}')} %; "
               f"trades/semana del ensamble en INNER_B: " + ", ".join(f(x, '{:.2f}') for x in ib_opw)),
        "D5": (f"MAE TP {f(_mean(mae_tp))} σ, SL {f(_mean(mae_sl))} σ en INNER_A; salidas TP en la cota superior {f(100 * _mean(cov_hi) if _mean(cov_hi) is not None else None, '{:.1f}')} %"),
        "E": (f"auditoría de codificación: {', '.join(sorted(set(str(a) for a in audits))) or 'NO_NEW_MEASUREMENT'} en {len(audits)} unidades; "
              f"forma de las cabezas = contrato en {sum(1 for x in shapes if x)}/{len(shapes)} unidades"),
    }
    if not audits:
        ev = {k: "NO_NEW_MEASUREMENT: no WF_V1 unit finished" for k in EVIDENCE_KEYS}
    return ev


def fill_closure(template: str, evidence: dict) -> str:
    """Replace ONLY the evidence markers; every other character of the template is kept."""
    outp = template
    for k in EVIDENCE_KEYS:
        outp = outp.replace(f"<!-- EVIDENCE:{k} -->", str(evidence.get(k, "NO_NEW_MEASUREMENT")).replace("|", "/"))
    return outp


def cmd_wf_v1(a):
    out = Path(a.results)
    spec = json.loads(Path(a.spec).read_text())
    w = spec["walk_forward_v1"]
    units, verdicts, sep, validity = [], {}, {}, {}
    for fold in w["folds"]:
        for arm in w["arms"]:
            u = unit_doc(out, f"WF_V1_{fold}_{arm['name']}") | {"fold": fold, "arm": arm["name"], "base": arm["base"], "v1": v1_details(out, f"WF_V1_{fold}_{arm['name']}")}
            b = unit_doc(out, f"WF_{fold}_{arm['base']}") | {"fold": fold, "arm": arm["base"], "baseline_of": arm["name"]}
            units += [u, b]
            ue, be = u.get("ensemble") or {}, b.get("ensemble") or {}
            if "INNER_B" in ue:
                opw = ue["INNER_B"]["opw"]
                validity[f"{fold}/{arm['name']}"] = {"INNER_B_trades_per_week": opw, "valid": bool(ue["INNER_B"]["valid"] and opw >= 2.0),
                                                     "verdict": "VALID" if ue["INNER_B"]["valid"] and opw >= 2.0 else "INVALID (owner's rule: < 2 trades/week on INNER_B)"}
            if "INNER_B" in ue and "INNER_B" in be:
                sep[f"{fold}/{arm['name']}_vs_{arm['base']}"] = ST.separable({"sharpe": ue["INNER_B"]["sharpe"], "se_lo": ue["INNER_B"]["se_lo"]},
                                                                             {"sharpe": be["INNER_B"]["sharpe"], "se_lo": be["INNER_B"]["se_lo"]})
    for arm in w["arms"]:
        per = {}
        for fold in w["folds"]:
            u = next(x for x in units if x["unit"] == f"WF_V1_{fold}_{arm['name']}")
            if u["status"] == "DONE" and u.get("ensemble") and "INNER_B" in u["ensemble"]:
                per[fold] = {"sharpe": u["ensemble"]["INNER_B"]["sharpe"], "se_lo": u["ensemble"]["INNER_B"]["se_lo"]}
        v = (ST.holds(per) if len(per) == len(w["folds"]) else {"rule": ST.HOLDS_RULE, "per_fold": {}, "verdict": "NO_NEW_MEASUREMENT: a fold is missing or has no ensemble"}) | {"inputs": per}
        if any(not validity.get(f"{f}/{arm['name']}", {}).get("valid", False) for f in w["folds"]) and v["verdict"] == "HOLDS":
            v["verdict"] = "INVALID: holds on Sharpe but trades fewer than 2 per week on INNER_B in some fold"
        verdicts[arm["name"]] = v
    closure = {}
    for fold in w["folds"]:
        fu = [x for x in units if x["fold"] == fold]
        rows = []
        for arm in w["arms"]:
            pair = [x for x in fu if x["unit"] in (f"WF_V1_{fold}_{arm['name']}", f"WF_{fold}_{arm['base']}")]
            rows += closure_rows(pair, f"WF_{fold}_{arm['base']}", "INNER_B", f"{fold}: v1 vs its baseline (same base arm, same fold), validation year scored after freeze")
        closure[fold] = rows
    frows, frule = fitness_rows(out, w)
    doc = {"caveat": CAVEAT, "spec": spec["identity"], "action_contract": w["action_contract"], "rule_preregistered": w["rule"], "units": units,
           "verdicts": verdicts, "separability_vs_baseline": sep, "out_of_sample_activity": validity, "closure": closure,
           "fitness_rule_preregistered": w.get("fitness_rule"), "fitness_rows": frows, "fitness_variant_rule": frule}
    Path(a.out_json).write_text(json.dumps(doc, indent=1, default=float))
    md = [f"# Contrato de salidas v1 en el walk-forward: resultados con errores estándar (spec `{spec['identity']}`, {a.date})", "",
          f"Contrato `{w['action_contract']['identity']}` (sha `{w['action_contract']['sha256'][:12]}…`): cinco especialistas calibrados (Platt en INNER_A), una estrategia "
          "paramétrica congelada en INNER_A por la mayor cota inferior (Sharpe − 1 EE) con ≥ 2.5 trades/semana; decisión en INNER_B por el ENSAMBLE de semillas. "
          "Línea base = el mismo brazo del walk-forward sin contrato. Replays de estudiantes; **no es una estrategia operable**; 2024 una lectura; TEST cerrado.", "",
          f"Regla pre-registrada (antes de leer): {w['rule']}", ""]
    for fold in w["folds"]:
        md += table_md([x for x in units if x["fold"] == fold], ("TRAIN", "INNER_A", "INNER_B", "OBS_2024"), f"Pliegue {fold}: v1 y su línea base")
    md += ["## Veredictos", ""] + [f"- **{k}**: {v['verdict']} — " + ", ".join(f"{f}: {fmt(x['sharpe'])} − SE {fmt(x['se_lo'])}" for f, x in v["inputs"].items()) for k, v in verdicts.items()] + [""]
    md += ["## Actividad fuera de muestra (INNER_B, ensamble)", ""] + [f"- {k}: {fmt(v['INNER_B_trades_per_week'])} trades/semana → {v['verdict']}" for k, v in validity.items()] + [""]
    if sep:
        md += ["## v1 frente a su línea base (ensamble, INNER_B)", ""] + [f"- {k}: Δ {v['delta']:+.2f} vs SE agrupado {fmt(v['pooled_se'])} → {v['verdict']}" for k, v in sep.items()] + [""]
    md += ["## Variantes de fitness de la estrategia (mismos especialistas y ensambles; sólo cambia el punto congelado)", "",
           f"Regla pre-registrada: {w.get('fitness_rule')}", "",
           "| brazo | fitness | pliegue | elegido: Sharpe INNER_B ± SE | Sharpe − SE | trades/semana | DD % | veredicto | archivo: % válidos | mejor Sharpe sin restricción (calibración, tr/sem) | mejor por fitness sin restricción → INNER_B (**no seleccionable**) | mejor válido por fitness (calibración) |",
           "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |"]
    for r in frows:
        if r["status"] != "DONE":
            md.append(f"| {r['arm']} | {r['fitness']} | {r['fold']} | — | — | — | — | {r['status']} | | | | |")
            continue
        ar = r["archive"]
        nb = ar.get("best_unconstrained_INNER_B_not_selectable")
        bv = ar.get("best_valid_by_fitness_calibration")
        md.append(f"| {r['arm']} | {r['fitness']} | {r['fold']} | {fmt(r['sharpe'])} ± {fmt(r['se_lo'])} | {fmt(r['lcb'])} | {fmt(r['opw'])} | {fmt(r['dd'], '{:.1f}')} | {r['verdict']} | "
                  f"{fmt(100 * ar['share_valid'] if ar.get('share_valid') is not None else None, '{:.0f}')} % | {fmt(ar.get('best_sharpe_unconstrained_calibration'))} ({fmt(ar.get('best_sharpe_unconstrained_opw'))}) | "
                  + (f"{fmt(nb['sharpe'])} ± {fmt(nb['se_lo'])}, {fmt(nb['opw'])} tr/sem — no seleccionable" if nb else "—") + " | "
                  + (f"{fmt(bv['fitness'], '{:.3g}')} (Sharpe {fmt(bv['sharpe'])}, {fmt(bv['opw'])} tr/sem)" if bv else "—") + " |")
    md += [""] + [f"- **{k}**: variante elegida en {v.get('chosen_on', '—')} = `{v['chosen']}` → {v['verdict']}" + (f" (Sharpe − SE en {v['confirm_fold']}: {fmt(v['lcb_confirm_fold'])}, {fmt(v['opw_confirm_fold'])} trades/semana)" if v.get("chosen") else "")
                   for k, v in frule.items() if not k.startswith("_")] + [""]
    if frule.get("_scales"):
        md += ["Escalas de los términos compuestos (regla pre-registrada: s_k = mediana |término| en el frente válido de INNER_A del ensamble; c_k = 1/s_k):", "",
               "| pliegue/brazo | " + " | ".join(f"s_{t} (c)" for t in ("sharpe", "r_ann", "opd", "dd")) + " | respaldo usado |", "| --- | ---: | ---: | ---: | ---: | --- |"]
        for k, sc in frule["_scales"].items():
            md.append(f"| {k} | " + " | ".join(f"{fmt(sc[t]['s'], '{:.4g}')} ({fmt(sc[t]['c'], '{:.4g}')})" for t in ("sharpe", "r_ann", "opd", "dd"))
                      + " | " + (", ".join(t for t in sc if sc[t]["fallback_used"]) or "ninguno") + " |")
        md.append("")
    md += ["## Detalle por semilla (veto NOP, ECE calibrada, información de NOP, actividad en INNER_B)", ""]
    for u in units:
        for s_, dd in (u.get("v1") or {}).items():
            md.append(f"- {u['unit']} semilla {s_}: veto NOP {'conservado' if dd['nop_veto_kept'] else 'desactivado'}; ECE tras calibrar "
                      + ", ".join(f"{m} {fmt(x, '{:.3f}')}" for m, x in dd["ece_after"].items()) + f"; Δ log-loss con pNOP {fmt(dd['nop_delta_logloss'], '{:+.4f}')}; INNER_B {fmt(dd['inner_b_opw'])} trades/semana")
    md += ["", "## Tabla de cierre (regla del propietario)", ""]
    for fold in w["folds"]:
        md += closure_md(closure[fold])
    ev = closure_evidence(out, w)
    doc["closure_evidence"] = ev
    Path(a.out_json).write_text(json.dumps(doc, indent=1, default=float))
    tpl = Path(a.closure_template).read_text() if getattr(a, "closure_template", None) else None
    if tpl:
        filled = fill_closure(tpl, ev)
        Path(a.closure_out).write_text(filled)
        md += ["## Tabla de cierre D1–D5 y codificación", ""] + [ln for ln in filled.splitlines() if ln.startswith("|")] + [""]
    Path(a.out_md).write_text("\n".join(md))
    print(json.dumps({"units": [(u["unit"], u["status"]) for u in units], "verdicts": {k: v["verdict"] for k, v in verdicts.items()}, "out": a.out_md}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("b-ensemble", cmd_b_ensemble), ("walk-forward", cmd_walk_forward), ("wf-v1", cmd_wf_v1)):
        p = sub.add_parser(name)
        p.add_argument("--spec", required=True)
        p.add_argument("--results", required=True, help="the package out/ directory")
        p.add_argument("--out-json", required=True)
        p.add_argument("--out-md", required=True)
        p.add_argument("--date", default="2026-10-25")
        if name == "b-ensemble":
            p.add_argument("--decision-5seed", default=None)
        if name == "wf-v1":
            p.add_argument("--closure-template", default=None, help="OUTPUT_CONTRACT_V1_CLOSURE.md with the evidence markers")
            p.add_argument("--closure-out", default=None)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
