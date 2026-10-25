#!/usr/bin/env python3
"""Reports with uncertainty for the clean round (owner 2026-10-25: "tabla con errores estándar, no sólo medias").

  b-ensemble     every finished arm of stages A / A_prime / B (plus seed extensions): per-seed Sharpe, 3/5-seed mean, the
                 seed-ensemble replay with its Lo SE and bootstrap interval (ENSEMBLE.json written by
                 tools/expert_guided_seed_ensemble.py), failed arms with their recorded cause, the 5-seed decision and
                 the Stage B ranking; owner closure table (control = same rows).
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
        verdicts[arm] = ST.holds(per) if len(per) == len(wf["folds"]) else {"rule": ST.HOLDS_RULE, "per_fold": per, "verdict": "NO_NEW_MEASUREMENT: a fold is missing or has no ensemble"}
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
    md += ["## Veredictos", ""] + [f"- **{arm}**: {v['verdict']} — " + ", ".join(f"{f}: {fmt(x['sharpe'])} − SE {fmt(x['se_lo'])}" for f, x in v["per_fold"].items()) for arm, v in verdicts.items()] + [""]
    if sep:
        md += ["## Separabilidad frente al control (ensamble, año de validación)", ""] + [f"- {k}: Δ {v['delta']:+.2f} vs SE agrupado {fmt(v['pooled_se'])} → {v['verdict']}" for k, v in sep.items()] + [""]
    md += ["## Tabla de cierre (regla del propietario)", ""]
    for fold in wf["folds"]:
        md += closure_md(closure[fold])
    Path(a.out_md).write_text("\n".join(md))
    print(json.dumps({"units": [(u["unit"], u["status"]) for u in units], "verdicts": {k: v["verdict"] for k, v in verdicts.items()}, "out": a.out_md}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("b-ensemble", cmd_b_ensemble), ("walk-forward", cmd_walk_forward)):
        p = sub.add_parser(name)
        p.add_argument("--spec", required=True)
        p.add_argument("--results", required=True, help="the package out/ directory")
        p.add_argument("--out-json", required=True)
        p.add_argument("--out-md", required=True)
        p.add_argument("--date", default="2026-10-25")
        if name == "b-ensemble":
            p.add_argument("--decision-5seed", default=None)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
