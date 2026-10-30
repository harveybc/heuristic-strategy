"""Strategy chain F4 (owner order 2026-10-29, "orquesta y ejecuta, cero babysitting"): the decision rules and the
bookkeeping of the unattended chain S1 -> S1R -> G2 -> S2 -> S3 -> Q -> R (tools/clean_round_chain/chain_f4/). Every
rule here is fixed BEFORE the chain runs (docs/short_horizon/clean_round/RUNBOOK_CADENA_ESTRATEGIA_F4.md); the shell
stages only call these subcommands and the existing tools.

  state     write a state file (<stage>.started / .done / .failed / FAILED_<stage>) with UTC time, code sha, inputs and
            outputs and their digests
  edge      S1R edge rule on STAGE1_RESULT.json: one extension point per axis whose frozen value sits on a grid edge
            (one step beyond, clipped to the sealed full_f4 bounds); a value AT a sealed bound is a BOUNDARY_HIT
  freeze    S1R: Deb ranking over the stage-1 points and the extension points (same canonical genes, same subset), the
            best frozen into a STRATEGY_PARAMS document (FROZEN_PARAMS.json) with FREEZE.json
  top5      S3: the top 5 points of the frozen ranking (the frozen point first) as a stage-3 point list
  s3        S3 decision from the grid tool's STAGE3_RESULT.json: re-run stage 2 once from the winner, or keep the champion
  cells     Q: the q-ladder / D9 cell files (the 24 level worlds of the DEAP subset at a given quality)
  noreverse Q: the champion with allow_reverse = 0 (the D6 / D3 arm)
  d9        Q: D9 instruments: draw-level cluster bootstrap with a Holm column, size = 1 trace equality, single-horizon
            ablations, the native counterfactual on every early close at q < 1
  report    R: REPORT_F4.md + REPORT_F4.json + PNG plots (Sharpe and profit against the head MAE and MAE / naive)
  status    STATUS.md / STATUS.json from files only
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import strategy_params as SP  # noqa: E402

STAGES = ("S1", "S1R", "G2", "S2", "S3", "Q", "R")
LEVEL_GRID = {"a_sl": (0.75, 1.0, 1.5), "a_tp": (0.75, 1.0, 1.5), "b_sl": (0.0, 0.5), "b_tp": (0.0, 0.5), "buffer_pips": (1.0, 3.0), "ref_p75": (0.0, 1.0)}
CATEGORICAL = ("ref_p75",)                                           # median / p75: both values are the whole domain, never an edge
Q_LADDER = (0.55, 0.62, 0.70, 0.80, 1.00)
BOOT_SEED, BOOT_N = 20261028, 10000


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ----------------------------------------------------------------------------- digests and state files
def digest(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    if p.is_file():
        return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    files = sorted(x for x in p.rglob("*") if x.is_file())
    h = hashlib.sha256()
    for x in files:                                                  # a manifest digest (relative path + size): result folders hold GBs
        h.update(f"{x.relative_to(p)}\t{x.stat().st_size}\n".encode())
    return {"path": str(p), "manifest_sha256": h.hexdigest(), "files": len(files)}


def cmd_state(a):
    d = {"stage": a.stage, "kind": a.kind, "utc": now(), "code_sha": a.code_sha, "reason": a.reason or None,
         "inputs": [digest(x) for x in (a.inputs.split(",") if a.inputs else [])], "outputs": [digest(x) for x in (a.outputs.split(",") if a.outputs else [])]}
    sd = Path(a.chain) / "state"
    sd.mkdir(parents=True, exist_ok=True)
    name = f"FAILED_{a.stage}" if a.kind == "FAILED" else f"{a.stage}.{a.kind}"
    (sd / name).write_text(json.dumps(d, indent=1))
    print(json.dumps({"state": name, "utc": d["utc"]}))


# ----------------------------------------------------------------------------- S1R: edge rule and freeze
def edge_rule(point: dict, grid: dict = LEVEL_GRID, bounds: dict | None = None) -> dict:
    """One extension per axis whose value is a grid edge: one grid step beyond, clipped to the sealed full_f4 bound; a
    value already at the sealed bound is BOUNDARY_HIT (recorded for the report, no extension)."""
    bounds = bounds or SP.bounds_of("full_f4")
    edges, ext, hits = [], [], []
    for k, vals in grid.items():
        if k in CATEGORICAL:
            continue
        vs = sorted(float(x) for x in vals)
        v = float(point[k])
        if v not in (vs[0], vs[-1]):
            continue
        low = v == vs[0]
        step = (vs[1] - vs[0]) if low else (vs[-1] - vs[-2])
        cand = v - step if low else v + step
        lo, hi = (float(x) for x in bounds[k])
        cc = min(max(cand, lo), hi)
        e = {"axis": k, "value": v, "side": "low" if low else "high", "sealed_bound": [lo, hi]}
        edges.append(e)
        if abs(cc - v) < 1e-12:
            hits.append(dict(e, status="BOUNDARY_HIT"))
        else:
            ext.append(dict(e, status="EXTENDED", to=round(cc, 10), clipped_to_bound=bool(abs(cc - cand) > 1e-12), point={**{kk: float(point[kk]) for kk in grid}, k: round(cc, 10)}))
    return {"rule": "S1R edge rule: one step beyond each edge axis within the sealed full_f4 bounds, one extension only; a value at a sealed bound is BOUNDARY_HIT",
            "frozen_point": {k: float(point[k]) for k in grid}, "edges": edges, "extensions": ext, "boundary_hits": hits,
            "extension_points": [e["point"] for e in ext]}


def cmd_edge(a):
    st1 = json.loads(Path(a.stage1).read_text())
    rk = st1["ranking"]
    if not rk:
        raise SystemExit("NO_RANKED_LEVEL_POINT")
    best = rk[0]                                                      # Deb: fewest failing witnesses, then the higher Sharpe
    res = edge_rule(best["params"])
    res["stage1_best"] = {k: best[k] for k in ("point", "rank", "fitness_sharpe", "n_failing_witnesses", "feasible")}
    Path(a.out).write_text(json.dumps(res, indent=1))
    Path(a.points_out).write_text(json.dumps(res["extension_points"], indent=1))
    print(json.dumps({"extensions": len(res["extension_points"]), "boundary_hits": [h["axis"] for h in res["boundary_hits"]]}))


def _ranked_from_grid_out(out_dir: str, label: str) -> list:
    from tools import strategy_level_grid as G
    meta = json.loads((Path(out_dir) / "RUN_META.json").read_text())
    allt = [json.loads(x) for x in (Path(out_dir) / "TASKS.jsonl").read_text().splitlines() if x.strip()]
    agg = G.aggregate(allt, meta["groups"])
    pts = meta.get("point_list") or G.grid_points()[: meta["points"]]
    rows = []
    for i, r in agg.items():
        rows.append({"source": label, "point": i, "params": pts[i], "fitness_sharpe": r["fitness_sharpe"], "n_failing_witnesses": r["n_failing_witnesses"],
                     "feasible": r["feasible"], "infeasible": bool(r["errors"]), "errors": r["errors"], "activity": r["activity"]})
    return rows


def deb_sort(rows: list) -> list:
    ok = [r for r in rows if not r.get("infeasible")]
    return sorted(ok, key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"]))


def cmd_freeze(a):
    base = SP.load(a.params)
    st1 = json.loads(Path(a.stage1).read_text())
    rows = [{"source": "stage1", **{k: r[k] for k in ("point", "params", "fitness_sharpe", "n_failing_witnesses", "feasible", "activity")}, "infeasible": False} for r in st1["ranking"]]
    if a.ext and (Path(a.ext) / "TASKS.jsonl").exists():
        rows += _ranked_from_grid_out(a.ext, "extension")
    ranked = deb_sort(rows)
    if not ranked:
        raise SystemExit("NO_RANKED_LEVEL_POINT")
    best = ranked[0]
    edge = json.loads(Path(a.edge).read_text()) if a.edge and Path(a.edge).exists() else {}
    doc = SP.validate({**base, **{k: float(v) for k, v in best["params"].items()}})
    Path(a.out_params).write_text(json.dumps(doc, indent=1))
    after = edge_rule(best["params"]) if best["source"] == "extension" else None
    res = {"utc": now(), "frozen": {"source": best["source"], "params": best["params"], "fitness_sharpe": best["fitness_sharpe"], "n_failing_witnesses": best["n_failing_witnesses"],
                                    "feasible": best["feasible"], "activity": best["activity"], "rank": 1},
           "params_sha256": SP.sha256_of(a.out_params), "boundary_hits": edge.get("boundary_hits", []), "extensions_run": edge.get("extensions", []),
           "edge_after_extension": ([e for e in after["edges"]] if after else []),
           "note": "one extension only (owner order): an extension point that wins stays frozen even when it sits on the new edge (EDGE_AFTER_EXTENSION, reported)",
           "ranking": [{"rank": i + 1, **{k: r[k] for k in ("source", "point", "params", "fitness_sharpe", "n_failing_witnesses", "feasible")}} for i, r in enumerate(ranked)],
           "infeasible_points": st1.get("infeasible_points", []) + [r for r in rows if r.get("infeasible")]}
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"frozen": res["frozen"]["params"], "source": best["source"], "sharpe": best["fitness_sharpe"], "failing": best["n_failing_witnesses"]}, default=float))


def cmd_top5(a):
    fr = json.loads(Path(a.freeze).read_text())
    pts = [r["params"] for r in fr["ranking"][: a.k]]
    Path(a.out).write_text(json.dumps(pts, indent=1))
    print(json.dumps({"points": len(pts), "frozen_index": 0}))


def s3_decision(stage3: dict) -> dict:
    wins = [r for r in stage3["rows"] if r.get("wins_by_more_than_1se")]
    if not wins:
        return {"rerun_stage2": False, "winner": None}
    w = max(wins, key=lambda r: r["paired_mean_dsharpe_vs_frozen"])
    return {"rerun_stage2": True, "winner": {"index": w["index"], "params": w["params"], "paired_mean_dsharpe_vs_frozen": w["paired_mean_dsharpe_vs_frozen"], "se": w["se"]}}


def cmd_s3(a):
    st3 = json.loads(Path(a.stage3).read_text())
    dec = s3_decision(st3)
    dec["rule"] = "S3: another top-5 point no worse on the constraint whose paired mean Sharpe gain over the frozen point exceeds 1 SE -> stage 2 re-run ONCE from it; then stop iterating"
    if dec["rerun_stage2"]:
        champ = SP.load(a.champion)
        doc = SP.validate({**champ, **{k: float(v) for k, v in dec["winner"]["params"].items()}})
        Path(a.out_params).write_text(json.dumps(doc, indent=1))
        dec["rerun_params_sha256"] = SP.sha256_of(a.out_params)
    Path(a.out).write_text(json.dumps(dec, indent=1, default=float))
    print(json.dumps({"rerun_stage2": dec["rerun_stage2"]}))


# ----------------------------------------------------------------------------- Q: cells and arms
def level_cells(subset: list) -> list:
    """The 24 level worlds of the DEAP subset (its quality < 1 entries), as matrix cells."""
    out = []
    for c in subset:
        if float(c.get("quality", 1.0)) < 0.999 and not c.get("pair"):
            d = {k: v for k, v in c.items() if k not in ("subset_key",)}
            out.append(d)
    return out


def cmd_cells(a):
    sub = json.loads(Path(a.subset).read_text())
    cells = level_cells(sub)[: a.limit] if a.limit else level_cells(sub)
    q = float(a.quality)
    out = [dict(c, quality=q) for c in cells]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"cells": len(out), "quality": q, "ids": ",".join(c["id"] for c in out)}))


def cmd_noreverse(a):
    d = SP.load(a.params)
    Path(a.out).write_text(json.dumps(SP.validate(dict(d, allow_reverse=0.0)), indent=1))


# ----------------------------------------------------------------------------- Q: D9 instruments
def load_records(d: str) -> dict:
    out = {}
    for p in sorted(Path(d).glob("CELL_*.json")):
        r = json.loads(p.read_text())
        if not r.get("error"):
            out[(r["id"], int(r["seed"]))] = r
    return out


def cluster_bootstrap(diffs: np.ndarray, n: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Draw-level cluster bootstrap: one draw = one (cell, seed) world; the mean paired difference, its 95 % CI and a
    two-sided bootstrap p-value (2 x the smaller tail share of resampled means beyond 0)."""
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {"n": int(len(d)), "mean": float(d.mean()) if len(d) else float("nan"), "ci95": [float("nan")] * 2, "p": float("nan")}
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    p = float(min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0))))
    return {"n": int(len(d)), "mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": p}


def holm(ps: list) -> list:
    """Holm step-down adjusted p-values (family = the comparisons of the gate step)."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: (np.inf if not np.isfinite(ps[i]) else ps[i]))
    adj, run = [float("nan")] * m, 0.0
    for r, i in enumerate(order):
        if not np.isfinite(ps[i]):
            continue
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


def paired(a: dict, b: dict, metric: str = "sharpe_daily") -> np.ndarray:
    ks = sorted(set(a) & set(b))
    return np.asarray([float(b[k].get(metric) if b[k].get(metric) is not None else np.nan) - float(a[k].get(metric) if a[k].get(metric) is not None else np.nan) for k in ks], float)


def trace_key(t: dict) -> tuple:
    return (int(t["entry_bar_abs"]), int(t["exit_bar_abs"]), int(t["direction"]), str(t.get("exit_reason")))


def trace_equality(sized: dict, unit: dict) -> dict:
    ks = sorted(set(sized) & set(unit))
    diff = []
    for k in ks:
        ta, tb = [trace_key(t) for t in sized[k]["trades"]], [trace_key(t) for t in unit[k]["trades"]]
        if ta != tb:
            first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            diff.append({"world": f"{k[0]}_s{k[1]}", "trades_sized": len(ta), "trades_unit": len(tb), "first_difference_index": first})
    return {"worlds": len(ks), "equal": len(ks) - len(diff), "different": diff, "missing_in_unit": sorted(f"{k[0]}_s{k[1]}" for k in set(sized) - set(unit)),
            "verdict": "PASS" if ks and not diff and not (set(sized) - set(unit)) else "FAIL"}


def counterfactual_pool(recs: dict) -> dict:
    out = {}
    for side in ("BUY", "SELL"):
        n = dest = band = 0.0
        regs = []
        for r in recs.values():
            x = (r.get("exit_counterfactual") or {}).get(side)
            if not x or not x.get("early_closes"):
                continue
            k = x["early_closes"]
            n += k
            dest += k * x["destructive_share"]
            band += k * x["band_share"]
            if np.isfinite(x["regret_vs_Ostar_cash_risk_mean"]):
                regs.append((k, x["regret_vs_Ostar_cash_risk_mean"]))
        out[side] = {"early_closes": int(n), "destructive_share": (dest / n) if n else float("nan"), "band_share": (band / n) if n else float("nan"),
                     "regret_vs_Ostar_cash_risk_mean": (sum(k * v for k, v in regs) / sum(k for k, _ in regs)) if regs else float("nan")}
    return out


def record_summary(recs: dict) -> dict:
    sh = [float(r.get("sharpe_daily")) for r in recs.values() if r.get("sharpe_daily") is not None]
    sig = [float(r["trade_summary"]["net_sigma_per_trade_median"]) for r in recs.values() if (r.get("trade_summary") or {}).get("n")]
    tpw = [float(r["results"]["W0_activity"]["value"]) for r in recs.values() if isinstance((r.get("results") or {}).get("W0_activity"), dict)]
    mae = [r["head_mae"] for r in recs.values() if r.get("head_mae")]
    eff = sum(1 for r in recs.values() if r.get("PASS_effective"))
    return {"records": len(recs), "pass_effective": eff, "sharpe_median": float(np.median(sh)) if sh else float("nan"), "sharpe_mean": float(np.mean(sh)) if sh else float("nan"),
            "net_sigma_per_trade_median": float(np.median(sig)) if sig else float("nan"), "trades_per_week_median": float(np.median(tpw)) if tpw else float("nan"),
            "head_net_mae_sigma": float(np.nanmean([m["net_mae_sigma"] for m in mae])) if mae else float("nan"),
            "head_naive_mae_sigma": float(np.nanmean([m["naive_mae_sigma"] for m in mae])) if mae else float("nan"),
            "head_mae_over_naive": float(np.nanmean([m["mae_over_naive"] for m in mae])) if mae else float("nan")}


def cmd_d9(a):
    q = Path(a.qroot)
    out = {"utc": now(), "bootstrap": {"draw": "the (cell, seed) world", "resamples": BOOT_N, "seed": BOOT_SEED, "metric": "daily Sharpe (sqrt 365), paired per world"}}
    ladder = {}
    for qq in Q_LADDER:
        d = q / f"ladder_q{qq:.2f}" / "results"
        if d.exists():
            ladder[f"{qq:.2f}"] = load_records(str(d))
    out["q_ladder"] = {k: record_summary(v) for k, v in ladder.items()}
    out["counterfactual_q_below_1"] = {k: counterfactual_pool(v) for k, v in ladder.items() if float(k) < 0.999}
    full = ladder.get("1.00", {})
    unit = load_records(str(q / "size_one" / "results")) if (q / "size_one" / "results").exists() else {}
    out["size_one_trace_equality"] = trace_equality(full, unit) if full and unit else {"verdict": "NOT_EVALUABLE"}
    comps = []
    for h in (6, 12, 24):
        d = q / f"ablation_h{h}" / "results"
        if d.exists() and full:
            ab = load_records(str(d))
            comps.append((f"champion_minus_h{h}_only", paired(ab, full)))
            out.setdefault("ablations", {})[f"h{h}"] = record_summary(ab)
    for nm, (da, db) in {"deap_minus_canonical_seeds345": ("m345_canonical", "m345_deap"), "deap_minus_noreverse_seeds345": ("m345_noreverse", "m345_deap")}.items():
        if (q / da / "results").exists() and (q / db / "results").exists():
            comps.append((nm, paired(load_records(str(q / da / "results")), load_records(str(q / db / "results")))))
    keys = sorted(ladder)
    for lo, hi in zip(keys, keys[1:]):
        comps.append((f"q{hi}_minus_q{lo}", paired(ladder[lo], ladder[hi])))
    bs = [(nm, cluster_bootstrap(d)) for nm, d in comps]
    adj = holm([b["p"] for _, b in bs])
    out["paired_comparisons"] = [dict(b, name=nm, p_holm=adj[i]) for i, (nm, b) in enumerate(bs)]
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"comparisons": len(bs), "size_one": out["size_one_trace_equality"].get("verdict"), "ladder": list(out["q_ladder"])}))


# ----------------------------------------------------------------------------- R: the report
def _plot(xs, ys, labels, xlabel, ylabel, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(xs, ys, color="#2a78d6", linewidth=2, marker="o", markersize=8, markeredgecolor="#fcfcfb", markeredgewidth=2)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#52514e")
    ax.set_xlabel(xlabel, color="#52514e")
    ax.set_ylabel(ylabel, color="#52514e")
    ax.set_title(title, color="#0b0b0b", fontsize=11, loc="left")
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cmd_report(a):
    c = Path(a.chain)
    w = c / "work"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rd = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None  # noqa: E731
    freeze, s3, d9 = rd(w / "S1R" / "FREEZE.json"), rd(w / "S3" / "S3_DECISION.json"), rd(w / "Q" / "D9.json")
    tier = rd(w / "Q" / "m012" / "report_tier" / "TIER_REPORT.json")
    d3 = {k: rd(w / "Q" / f"D3_{k}.json") for k in ("canonical_vs_deap", "noreverse_vs_deap")}
    champion = rd(w / "FINAL_CHAMPION.json")
    plots = []
    lad = (d9 or {}).get("q_ladder") or {}
    if lad:
        ks = sorted(lad)
        labs = [f"q{k}" for k in ks]
        for xk, xl in (("head_net_mae_sigma", "MAE de la cabeza de neto (sigma)"), ("head_mae_over_naive", "MAE / MAE ingenuo")):
            xs = [lad[k][xk] for k in ks]
            for yk, yl, tag in (("sharpe_median", "Sharpe diario (mediana por registro)", "sharpe"), ("net_sigma_per_trade_median", "neto por operación (sigma, mediana)", "profit")):
                ys = [lad[k][yk] for k in ks]
                fn = f"ladder_{tag}_vs_{'mae' if xk == 'head_net_mae_sigma' else 'mae_naive'}.png"
                _plot(xs, ys, labs, xl, yl, f"{yl} frente a {xl}", out / fn)
                plots.append(fn)
    resid = []
    try:
        sys.path.insert(0, str(ROOT))
        from tools import strategy_behaviour_matrix as BM
        resid = [x for x in BM.WITNESS_CHANGELOG if x.get("audit_request") == "most likely wrong"]
    except Exception as e:                                          # the report never fails on the residual list; it says so
        resid = [{"witness": "residual list unavailable", "defect": str(e)[:200]}]
    rep = {"utc": now(), "frozen_level_point": freeze, "stage3": s3, "champion": champion, "tier_full_matrix": tier, "d3": d3, "d9": d9, "plots": plots, "most_likely_wrong": resid}
    (out / "REPORT_F4.json").write_text(json.dumps(rep, indent=1, default=float))
    L = ["# Informe F4: cadena de estrategia (datos)", "", f"Generado {rep['utc']} a partir de los archivos de la cadena; ningún número se escribió a mano.", ""]
    if freeze:
        f = freeze["frozen"]
        L += ["## Punto de niveles congelado (etapa 1)", "", f"Parámetros: `{json.dumps(f['params'])}` (origen {f['source']}), Sharpe {f['fitness_sharpe']:.3f}, testigos que fallan {f['n_failing_witnesses']}, factible {f['feasible']}.",
              f"Límites alcanzados (BOUNDARY_HIT): {[h['axis'] for h in freeze['boundary_hits']] or 'ninguno'}; extensiones: {[e['axis'] for e in freeze['extensions_run']] or 'ninguna'}; puntos inviables: {len(freeze['infeasible_points'])}.", ""]
    if s3:
        L += ["## Etapa 3", "", f"Re-ejecución de la etapa 2: {s3.get('rerun_stage2')}; ganador: {json.dumps(s3.get('winner'), default=float)}", ""]
    if tier:
        L += ["## Matriz completa (semillas 0-2 y gemelos)", "", f"Registros {tier['records']}, PASS efectivo {tier['pass_effective']}, FAIL {tier['fail_effective']}; actividad {tier['activity']}.", ""]
        L += ["| testigo | PASS | FAIL | NOT_EVALUABLE | CONTROL_CONFLICT |", "|---|---:|---:|---:|---:|"] + [f"| {k} | {v.get('PASS', 0)} | {v.get('FAIL', 0)} | {v.get('NOT_EVALUABLE', 0)} | {v.get('CONTROL_CONFLICT', 0)} |" for k, v in tier["witnesses"].items()] + [""]
    for k, v in d3.items():
        if v:
            L += [f"## D3 {k} (semillas 3-5)", "", f"`{json.dumps({x: v.get(x) for x in ('status', 'cells', 'sharpe', 'net_usd')}, default=float)}`", ""]
    if d9:
        L += ["## Escalera de calidad", "", "| q | registros | Sharpe mediana | neto sigma/op | op/semana | MAE | MAE/ingenuo |", "|---|---:|---:|---:|---:|---:|---:|"]
        for k, v in sorted(lad.items()):
            L.append(f"| {k} | {v['records']} | {v['sharpe_median']:.3f} | {v['net_sigma_per_trade_median']:.3f} | {v['trades_per_week_median']:.2f} | {v['head_net_mae_sigma']:.3f} | {v['head_mae_over_naive']:.3f} |")
        L += ["", "## D9", "", f"Igualdad de trazas con tamaño 1: {d9['size_one_trace_equality'].get('verdict')}.", "",
              "| comparación pareada | n mundos | media | IC95 | p | p Holm |", "|---|---:|---:|---|---:|---:|"]
        for x in d9["paired_comparisons"]:
            L.append(f"| {x['name']} | {x['n']} | {x['mean']:.3f} | [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}] | {x['p']:.4f} | {x['p_holm']:.4f} |")
        L += ["", "Contrafactual nativo en cada cierre anticipado a q < 1 (cuota destructiva, banda ±0.02 sigma, arrepentimiento frente a O* en riesgo de caja):", "",
              f"`{json.dumps(d9['counterfactual_q_below_1'], default=float)}`", ""]
    L += ["## Gráficos", ""] + [f"- `{p}`" for p in plots] + ["", "## Lo más probablemente erróneo (para la auditoría)", ""]
    for x in resid:
        L.append(f"- **{x.get('witness')}**: {x.get('defect')}")
    (out / "REPORT_F4.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"report": str(out / "REPORT_F4.md"), "plots": plots}))


# ----------------------------------------------------------------------------- status
def _sh(cmd: list) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return f"unavailable: {e}"


def stage_progress(c: Path, cfg: dict) -> dict:
    w = c / "work"
    p = {}
    s1 = Path(cfg.get("S1_ROOT") or (w / "S1"))
    t = s1 / "out" / "TASKS.jsonl"
    if t.exists():
        n = sum(1 for _ in open(t))
        tot = json.loads((s1 / "out" / "RUN_META.json").read_text()).get("points", 144) * json.loads((s1 / "out" / "RUN_META.json").read_text()).get("groups", 78)
        last = (s1 / "out" / "PROGRESS.log").read_text().strip().splitlines()[-1] if (s1 / "out" / "PROGRESS.log").exists() else ""
        p["S1"] = {"tasks_done": n, "tasks_total": tot, "last_progress": last}
    for nm in ("S2", "S2b"):
        pl = w / nm / "PROGRESS.log"
        if pl.exists():
            gens = [json.loads(x) for x in pl.read_text().splitlines() if x.startswith("{") and '"generation"' in x]
            p[nm] = {"generations_done": len(gens), "last": gens[-1] if gens else None}
    qd = w / "Q"
    if qd.exists():
        p["Q"] = {d.name: len(list((d / "results").glob("CELL_*.json"))) for d in sorted(qd.iterdir()) if (d / "results").exists()}
    return p


def best_so_far(c: Path, cfg: dict) -> dict:
    out = {}
    for nm in ("S2b", "S2"):
        ar = c / "work" / nm / "ARCHIVE.jsonl"
        if ar.exists():
            best = None
            for x in open(ar):
                r = json.loads(x)
                k = (r["n_failing_witnesses"], -r["fitness_sharpe"])
                if best is None or k < best[0]:
                    best = (k, r)
            if best:
                r = best[1]
                out = {"source": nm, "fitness_sharpe": r["fitness_sharpe"], "n_failing_witnesses": r["n_failing_witnesses"], "activity": r.get("activity")}
                return out
    fz = c / "work" / "S1R" / "FREEZE.json"
    if fz.exists():
        f = json.loads(fz.read_text())["frozen"]
        return {"source": "S1R", "fitness_sharpe": f["fitness_sharpe"], "n_failing_witnesses": f["n_failing_witnesses"], "activity": f["activity"]}
    s1 = Path(cfg.get("S1_ROOT") or (c / "work" / "S1")) / "out"
    if (s1 / "TASKS.jsonl").exists():
        from tools import strategy_level_grid as G
        meta = json.loads((s1 / "RUN_META.json").read_text())
        agg = G.aggregate([json.loads(x) for x in open(s1 / "TASKS.jsonl") if x.strip()], meta["groups"])
        rk = sorted((r for r in agg.values() if not r["errors"]), key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"]))
        if rk:
            r = rk[0]
            return {"source": "S1 (points complete so far)", "point": r["point"], "fitness_sharpe": r["fitness_sharpe"], "n_failing_witnesses": r["n_failing_witnesses"], "activity": r["activity"]}
    return out


def read_cfg(c: Path) -> dict:
    cfg = {}
    f = c / "chain.env"
    if f.exists():
        for ln in f.read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"')
    return cfg


def cmd_status(a):
    c = Path(a.chain)
    cfg = read_cfg(c)
    st = c / "state"
    states = {}
    for s in STAGES:
        states[s] = ("FAILED" if (st / f"FAILED_{s}").exists() else "done" if (st / f"{s}.done").exists() else "running/started" if (st / f"{s}.started").exists() else "pending")
    cur = next((s for s in STAGES if states[s] != "done"), "FINAL")
    last_err = None
    for s in STAGES:
        for f in (st / f"FAILED_{s}", st / f"{s}.failed"):
            if f.exists():
                last_err = json.loads(f.read_text())
    adm = _sh(["python3", os.path.expanduser("~/.local/libexec/crispdm/crispdm_admission.py"), "state"])
    try:
        ad = json.loads(adm)
        mem = {"host_free_for_new_GiB": round(ad["host_free_for_new_bytes"] / 2 ** 30, 2), "mem_available_GiB": round(ad["mem_available_bytes"] / 2 ** 30, 2)}
    except Exception:
        mem = {"raw": adm[:300]}
    scopes = [ln.split()[0] for ln in _sh(["systemctl", "--user", "list-units", "--type=scope", "--state=active", "--no-legend", "crispdm-*"]).splitlines() if ln.strip()]
    d = {"utc": now(), "final": (c / "FINAL").exists(), "stop_flag": (c / "STOP").exists(), "current_stage": cur, "stages": states, "progress": stage_progress(c, cfg),
         "best_so_far": best_so_far(c, cfg), "memory": mem, "running_scopes": scopes, "last_error": last_err,
         "service": "strategy-chain-f4.service", "eta": cfg.get("ETA_NOTE", "")}
    (c / "STATUS.json").write_text(json.dumps(d, indent=1, default=float))
    L = [f"# Estado de la cadena F4 ({d['utc']})", "", f"- FINAL: {d['final']}   STOP: {d['stop_flag']}   etapa actual: **{cur}**",
         "- Etapas: " + ", ".join(f"{k}={v}" for k, v in states.items()), f"- Progreso: `{json.dumps(d['progress'], default=float)}`",
         f"- Mejor hasta ahora: `{json.dumps(d['best_so_far'], default=float)}`", f"- Memoria: `{json.dumps(mem)}`", f"- Ámbitos crispdm activos: {scopes or 'ninguno'}",
         f"- Último error: `{json.dumps(last_err, default=float) if last_err else 'ninguno'}`", f"- ETA: {d['eta']}", "",
         "## Cómo operar", "", f"- Estado: `bash {c}/status.sh`",
         f"- Parada suave (entre tareas y etapas): `touch {c}/STOP`  -> reanudar: `rm {c}/STOP && systemctl --user start strategy-chain-f4.service`",
         "- Parada dura (solo el dueño): `systemctl --user stop strategy-chain-f4.service` y, para el trabajo en curso, `systemctl --user stop <ámbito de la lista de arriba>`",
         f"- Reintentar una etapa fallida: `rm {c}/state/FAILED_<etapa> {c}/state/<etapa>.attempts && systemctl --user start strategy-chain-f4.service`",
         f"- Runbook: `{c}/RUNBOOK.md`"]
    (c / "STATUS.md").write_text("\n".join(L) + "\n")
    if not a.quiet:
        print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("state")
    for k in ("chain", "stage", "kind", "code_sha"):
        s.add_argument("--" + k.replace("_", "-"), dest=k, required=True)
    s.add_argument("--inputs", default="")
    s.add_argument("--outputs", default="")
    s.add_argument("--reason", default="")
    e = sub.add_parser("edge")
    e.add_argument("--stage1", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--points-out", required=True)
    f = sub.add_parser("freeze")
    for k in ("stage1", "params", "out", "out_params"):
        f.add_argument("--" + k.replace("_", "-"), dest=k, required=True)
    f.add_argument("--ext", default=None)
    f.add_argument("--edge", default=None)
    t = sub.add_parser("top5")
    t.add_argument("--freeze", required=True)
    t.add_argument("--k", type=int, default=5)
    t.add_argument("--out", required=True)
    s3 = sub.add_parser("s3")
    for k in ("stage3", "champion", "out", "out_params"):
        s3.add_argument("--" + k.replace("_", "-"), dest=k, required=True)
    ce = sub.add_parser("cells")
    ce.add_argument("--subset", required=True)
    ce.add_argument("--quality", required=True)
    ce.add_argument("--limit", type=int, default=0)
    ce.add_argument("--out", required=True)
    nr = sub.add_parser("noreverse")
    nr.add_argument("--params", required=True)
    nr.add_argument("--out", required=True)
    d9 = sub.add_parser("d9")
    d9.add_argument("--qroot", required=True)
    d9.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--chain", required=True)
    r.add_argument("--out", required=True)
    st = sub.add_parser("status")
    st.add_argument("--chain", required=True)
    st.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    {"state": cmd_state, "edge": cmd_edge, "freeze": cmd_freeze, "top5": cmd_top5, "s3": cmd_s3, "cells": cmd_cells, "noreverse": cmd_noreverse,
     "d9": cmd_d9, "report": cmd_report, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
