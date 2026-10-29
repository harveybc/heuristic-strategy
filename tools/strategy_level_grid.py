"""DEAP stage 1 under full_f4 (coordinator ruling 2026-10-29): the 144-point coarse grid of the absolute-level genes at
the canonical strategy genes, Sharpe-only fitness with Deb's constraint domination (fewer failing witnesses first, then
the higher mean daily Sharpe), the menu truth REBUILT per point (it depends on the level genes; the MENU_GENES_MISMATCH
guard refuses a policy evaluated on another point's truth).

  grid: a_sl, a_tp in {0.75, 1, 1.5}; b_sl, b_tp in {0, 0.5}; buffer_pips in {1, 3}; reference quantile in {median, p75}
        (ref_p75 0 / 1) -> 3 x 3 x 2 x 2 x 2 x 2 = 144 points; every other gene at the parameter file's value.

The evaluation is the DEAP's own (strategy_param_deap.CachedWorld.evaluate + judge_pairs), world-major to bound memory:
one task = (point, world group), a twin / mirror pair being one group (W14 / E6 judged inside the task); a worker
builds the task's worlds under the point's level genes, evaluates the point once, and releases them. Results append to
TASKS.jsonl (resumable: finished tasks are skipped on restart); <out>/STOP stops before the next task. The generation-0
degeneracy gate (strategy_param_deap.DEGENERACY_RULE) is applied to the first GATE_POINTS points as soon as they are
complete: a degenerate batch writes FAILED_DEGENERATE_GEN0 and the stage stops. ``report`` ranks the points and writes
STAGE1_RESULT.json (the best feasible point, its rank, its distance to the grid edges with an edge flag, activity next
to the Sharpe: trades / week, 1-bar re-open share, net sigma per trade, the calm-bar burst-gate share and W17).

  python tools/strategy_level_grid.py run --subset <SUBSET.json> --params <STRATEGY_PARAMS.json> --admitted <d> --csv-dir <d> --jobs 8 --out <dir>
  python tools/strategy_level_grid.py report --out <dir> [--params <STRATEGY_PARAMS.json>]
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import strategy_params as SP  # noqa: E402
from app.short_horizon.expert_guided import strategy_zz as ZZ  # noqa: E402
from tools import strategy_output_simulation as S  # noqa: E402
from tools import strategy_param_deap as D  # noqa: E402

LEVEL_GRID = {"a_sl": (0.75, 1.0, 1.5), "a_tp": (0.75, 1.0, 1.5), "b_sl": (0.0, 0.5), "b_tp": (0.0, 0.5), "buffer_pips": (1.0, 3.0), "ref_p75": (0.0, 1.0)}
GATE_POINTS = 16                                                     # the generation-0-sized batch the degeneracy gate reads (population 16)
BATCH_POINTS = 2                                                     # tasks are submitted 2 points at a time: STOP and the gate are read between batches


def grid_points() -> list:
    keys = tuple(LEVEL_GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(LEVEL_GRID[k] for k in keys))]


def point_doc(base: dict, pt: dict) -> dict:
    return SP.validate({**base, **pt})


def groups_of(subset: list) -> list:
    g = {}
    for c in subset:
        g.setdefault(c.get("pair") or c["subset_key"], []).append(c)
    return [(k, v) for k, v in sorted(g.items())]


_BASE = {}


def _task(arg):
    i, gkey, cells, doc = arg
    import warnings
    warnings.filterwarnings("ignore")
    t0 = time.time()
    ZZ.configure(doc)                                                # the menu truth is built with THIS point's level genes
    AC, cx = _BASE["AC"], _BASE["cx"]
    SCFG = S.with_variant(_BASE["SCFG0"], C3.exec_variant_of(doc))
    try:
        res = []
        for c in cells:
            w = D.CachedWorld(cx, SCFG, AC, c, 0)
            res.append(w.evaluate(doc))
            del w
            gc.collect()
        D.judge_pairs(res)                                           # W14 / E6 under the full_f4 rule; drops the swapped-reference control world
        err = None
    except Exception as e:                                           # recorded, never a silent pass: the point counts the group as failing
        res, err = [], f"{type(e).__name__}: {str(e)[:300]}"
    D._malloc_trim()
    slim = [{k: r.get(k) for k in ("key", "sharpe", "fails", "quality", "opens", "trades", "early_closes", "levels_clipped", "activity", "burst_gate", "w17", "w14_control_status")} |
            {"net_sigma_per_trade_median": (r.get("trade_stats") or {}).get("net_sigma_per_trade_median")} for r in res]
    return {"point": i, "group": gkey, "n_worlds": len(cells), "per_world": slim, "error": err, "sec": time.time() - t0}


def aggregate(tasks: list, n_groups: int) -> dict:
    """Per point: complete when every group returned; fitness = mean Sharpe over the worlds; constraint = the number of
    failing witnesses (+ one per world of a group that errored)."""
    by = {}
    for t in tasks:
        by.setdefault(t["point"], []).append(t)
    out = {}
    for i, ts in by.items():
        if len({t["group"] for t in ts}) < n_groups:
            continue
        pw = [w for t in ts for w in t["per_world"]]
        errs = [t for t in ts if t["error"]]
        n_fail = sum(len(w["fails"]) for w in pw) + sum(t["n_worlds"] for t in errs)
        act = [w["activity"] for w in pw if w.get("activity")]
        bg = [w["burst_gate"] for w in pw if isinstance(w.get("burst_gate"), dict)]
        w17 = [w["w17"] for w in pw if w.get("w17") is not None]
        out[i] = {"point": i, "fitness_sharpe": float(np.mean([w["sharpe"] for w in pw])) if pw else float("nan"), "n_failing_witnesses": int(n_fail), "feasible": n_fail == 0 and not errs,
                  "errors": [t["error"] for t in errs], "worlds": len(pw), "per_world": [{"key": w["key"], "quality": w["quality"], "fails": w["fails"]} for w in pw],
                  "activity": {"trades_per_week_mean": float(np.mean([a["trades_per_week"] for a in act])) if act else float("nan"),
                               "reopen_1bar_share_median": float(np.nanmedian([a["reopen_1bar_share"] for a in act])) if act else float("nan"),
                               "net_sigma_per_trade_median": float(np.nanmedian([w["net_sigma_per_trade_median"] if w["net_sigma_per_trade_median"] is not None else np.nan for w in pw])) if pw else float("nan"),
                               "calm_bar_burst_gate_share_raw_median": float(np.median([b["calm_trigger_share_raw"] for b in bg])) if bg else float("nan"),
                               "W17_fail_worlds": sum(1 for x in w17 if x is False), "W17_worlds": len(w17)},
                  "levels_clipped": int(sum(int(w.get("levels_clipped") or 0) for w in pw))}
    return out


def cmd_run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sub = json.loads(Path(a.subset).read_text())
    base = SP.load(a.params)
    if SP.exec_profile_of(base) != "full_f4":
        raise SystemExit("LEVEL_GRID_NEEDS_FULL_F4")
    if getattr(a, "points_json", None):                              # stage 3: an explicit list of level points (the top stage-1 points) at the --params genes
        pts = [{k: float(p[k]) for k in LEVEL_GRID} for p in json.loads(Path(a.points_json).read_text())]
    else:
        pts = grid_points()[: a.max_points] if a.max_points else grid_points()   # --max-points: a smoke run only (never the stage)
    groups = groups_of(D.with_swap_worlds(sub, base))                # full_f4: each pair group carries its side-swapped-reference control world
    meta = {"stage": 3 if getattr(a, "points_json", None) else 1, "grid": {k: list(v) for k, v in LEVEL_GRID.items()}, "points": len(pts), "point_list": pts, "subset": a.subset, "subset_worlds": len(sub), "groups": len(groups),
            "params_base_sha256": SP.sha256_of(a.params), "fitness": "mean daily Sharpe (sqrt 365) over the subset worlds (Sharpe only)",
            "constraint": "Deb constraint domination: failing witnesses first (the DEAP's constraint set: strategy_param_deap.CachedWorld.evaluate + judge_pairs)",
            "truth": "rebuilt per point (world-major tasks: (point, world group))", "gate": f"generation-0 degeneracy rule on the first {GATE_POINTS} points",
            "spec_v3_sha256": D.BM.SPEC_V3_SHA, "exit_spec_sha256": D.BM.EXIT_SPEC_SHA}
    (out / "RUN_META.json").write_text(json.dumps(meta, indent=1))
    log = lambda m: (print(m, flush=True), open(out / "PROGRESS.log", "a").write(m + "\n"))  # noqa: E731
    tf = out / "TASKS.jsonl"
    done = set()
    if tf.exists():
        for x in tf.read_text().splitlines():
            if x.strip():
                t = json.loads(x)
                done.add((t["point"], t["group"]))
    todo = [(i, k, cells, point_doc(base, p)) for i, p in enumerate(pts) for k, cells in groups if (i, k) not in done]
    log(json.dumps({"start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tasks_total": len(pts) * len(groups), "tasks_todo": len(todo), "jobs": a.jobs}))
    _BASE["AC"] = S.contract_for_sim(a.contract)
    cx, SCFG0 = S.load_fold(a.fold, a.admitted, a.csv_dir)
    _BASE["cx"], _BASE["SCFG0"] = cx, SCFG0
    gate_done = (out / "GATE_GEN0.json").exists()
    t0 = time.time()
    n = 0
    stop = None
    pool = mp.get_context("fork").Pool(a.jobs, maxtasksperchild=8)  # never terminated: a STOP or a failed gate ends submission between batches
    batch = len(groups) * BATCH_POINTS
    with open(tf, "a") as fh:
        for b0 in range(0, len(todo), batch):
            if (out / "STOP").exists():
                stop = {"STOP": True, "tasks_done": n}
                break
            for r in pool.imap(_task, todo[b0:b0 + batch], chunksize=1):
                fh.write(json.dumps(r, default=float) + "\n")
                fh.flush()
                n += 1
            el = time.time() - t0
            log(json.dumps({"tasks_done": n, "of": len(todo), "elapsed_h": round(el / 3600, 2), "eta_h": round(el / max(n, 1) * (len(todo) - n) / 3600, 2)}))
            if not gate_done:
                allt = [json.loads(x) for x in tf.read_text().splitlines() if x.strip()]
                agg = aggregate(allt, len(groups))
                if all(i in agg for i in range(GATE_POINTS)):
                    dg = D.degenerate_gen0([agg[i] for i in range(GATE_POINTS)])
                    (out / "GATE_GEN0.json").write_text(json.dumps(dg, indent=1))
                    gate_done = True
                    log(json.dumps({"gen0_degeneracy": dg}))
                    if dg["degenerate"]:
                        (out / "FAILED_DEGENERATE_GEN0").write_text(json.dumps(dg, indent=1))
                        stop = {"FAILED_DEGENERATE_GEN0": dg["reasons"]}
                        break
    pool.close()
    pool.join()
    if stop is not None:
        log(json.dumps(stop))
        sys.exit(3 if "FAILED_DEGENERATE_GEN0" in stop else 0)
    log(json.dumps({"done": True, "tasks_done": n, "wall_h": round((time.time() - t0) / 3600, 2)}))


def edge_report(pt: dict) -> dict:
    out = {}
    for k, vals in LEVEL_GRID.items():
        v = pt[k]
        out[k] = {"value": v, "grid": list(vals), "at_min": v == min(vals), "at_max": v == max(vals), "edge": bool(len(vals) > 2 and v in (min(vals), max(vals)))}
    return out


def cmd_report(a):
    out = Path(a.out)
    meta = json.loads((out / "RUN_META.json").read_text())
    allt = [json.loads(x) for x in (out / "TASKS.jsonl").read_text().splitlines() if x.strip()]
    agg = aggregate(allt, meta["groups"])
    pts = meta.get("point_list") or grid_points()[: meta["points"]]
    ranked = sorted(agg.values(), key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"]))
    for k, r in enumerate(ranked):
        r["rank"] = k + 1
        r["params"] = pts[r["point"]]
    best = next((r for r in ranked if r["feasible"]), None)
    res = {"points_complete": len(agg), "points_total": len(pts), "feasible_points": sum(1 for r in ranked if r["feasible"]),
           "best_feasible": None if best is None else {"point": best["point"], "rank": best["rank"], "params": best["params"], "fitness_sharpe": best["fitness_sharpe"],
                                                      "activity": best["activity"], "grid_edges": edge_report(best["params"])},
           "best_by_constraint_then_sharpe": None if not ranked else {"point": ranked[0]["point"], "params": ranked[0]["params"], "n_failing_witnesses": ranked[0]["n_failing_witnesses"], "fitness_sharpe": ranked[0]["fitness_sharpe"]},
           "ranking": [{k: r[k] for k in ("rank", "point", "params", "fitness_sharpe", "n_failing_witnesses", "feasible", "activity", "errors")} for r in ranked]}
    (out / "STAGE1_RESULT.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("points_complete", "points_total", "feasible_points", "best_feasible")}, default=float))


def cmd_compare(a):
    """Stage 3 (ruling 2026-10-29): the top stage-1 points re-evaluated with the stage-2 champion's genes; per point the
    paired per-world Sharpe difference against the frozen point, its mean and standard error; another level point wins
    when it is feasible (or no worse on the constraint) and its mean paired difference exceeds 1 SE -> re-run stage 2 once."""
    out = Path(a.out)
    meta = json.loads((out / "RUN_META.json").read_text())
    allt = [json.loads(x) for x in (out / "TASKS.jsonl").read_text().splitlines() if x.strip()]
    agg = aggregate(allt, meta["groups"])
    pts = meta["point_list"]
    sh = {i: {w["key"]: w["sharpe"] for t in allt if t["point"] == i for w in t["per_world"]} for i in agg}
    f = a.frozen
    rows = []
    for i in sorted(agg):
        keys = sorted(set(sh[i]) & set(sh[f]))
        d = np.array([sh[i][k] - sh[f][k] for k in keys], float)
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        rows.append({"index": i, "params": pts[i], "fitness_sharpe": agg[i]["fitness_sharpe"], "n_failing_witnesses": agg[i]["n_failing_witnesses"], "feasible": agg[i]["feasible"],
                     "paired_mean_dsharpe_vs_frozen": float(d.mean()) if len(d) else float("nan"), "se": se, "worlds": len(d),
                     "wins_by_more_than_1se": bool(i != f and len(d) > 1 and agg[i]["n_failing_witnesses"] <= agg[f]["n_failing_witnesses"] and d.mean() > se)})
    res = {"frozen_index": f, "frozen_params": pts[f], "rows": rows, "rerun_stage2": any(r["wins_by_more_than_1se"] for r in rows)}
    (out / "STAGE3_RESULT.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"rerun_stage2": res["rerun_stage2"], "rows": [{k: r[k] for k in ("index", "paired_mean_dsharpe_vs_frozen", "se", "n_failing_witnesses", "wins_by_more_than_1se")} for r in rows]}, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--subset", required=True)
    r.add_argument("--params", default=str(ROOT / "docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json"))
    r.add_argument("--fold", default="wf2022")
    r.add_argument("--admitted", required=True)
    r.add_argument("--csv-dir", required=True)
    r.add_argument("--contract", default=str(S.CONTRACT))
    r.add_argument("--jobs", type=int, default=8)
    r.add_argument("--max-points", type=int, default=0, help="smoke runs only: the first N grid points")
    r.add_argument("--points-json", default=None, help="stage 3: a JSON list of level points to evaluate at the --params genes (instead of the grid)")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True, help="a stage-3 run folder")
    c.add_argument("--frozen", type=int, required=True, help="the index (in the stage-3 point list) of the frozen stage-1 point")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
