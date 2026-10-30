#!/usr/bin/env python
"""PZ5 (PZ_PREREGISTRO_2026_10_30 section 5; owner order 2026-10-30: run it now, locally, within the 3 GiB budget): the
DEAP of the 10 PZ genes on the formal DEAP's subset, the formal procedure's staging and constraints.

Why staged: the PZ truth menu is the replay of PZ's own orders, which depend on the level genes (a_sl, a_tp, buffer);
a candidate with other level genes needs another truth (the formal chain's reason for its stage-1 level grid). So:
  stage 1  the level grid a_sl, a_tp in {0.75, 1, 1.5} x buffer in {1, 3} (the formal LEVEL_GRID values of these axes)
           at the canonical open / close genes; Deb ranking; one extension point per edge axis (the formal S1R rule)
  stage 2  the level point frozen; DEAP over (kappa, theta, alpha, tau0, rho, k_persist, p_urg): Sharpe only, Deb
           constraint domination by the number of failing witnesses, population 16, 30 generations, elitism 2, SBX 0.9 /
           polynomial mutation (the formal operators), the canonical point in generation 0, the gen-0 degeneracy gate
  stage 3  the top 5 stage-1 points re-evaluated with the stage-2 champion's genes (reported; the formal re-run rule
           applies: another level point winning by > 1 SE triggers one more stage 2 from it)
Constraints per world, as the formal DEAP (strategy_param_deap.CachedWorld.evaluate): perfect worlds -> the VALIDITY_SPEC
v2 failing witnesses + the light exit layer; q < 1 worlds -> the quality-independent witnesses + D7 per side against the
information-limited Bayes closer with the SAME (PZ) opener; pairs -> W14 / E6 (strategy_param_deap.judge_pairs).
Memory (3 GiB): WORLD-MAJOR evaluation -- each worker builds one world, evaluates every candidate of the batch on it,
releases it; per-(stage, generation, world) result files make a generation resumable; STOP flag between worlds.

CLI: measure | gate-mirror | stage1 | stage2 | stage3 | status
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import itertools
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import strategy_params as SP  # noqa: E402
from app.short_horizon.expert_guided import strategy_pz as PZ  # noqa: E402
from app.short_horizon.expert_guided import strategy_zz as ZZ  # noqa: E402
from tools import strategy_behaviour_matrix as BM  # noqa: E402
from tools import strategy_hierarchy as H  # noqa: E402
from tools import strategy_output_simulation as S  # noqa: E402
from tools import strategy_param_deap as D  # noqa: E402

SEED = 20261030
POPULATION, GENERATIONS, ELITE = 16, 30, 2
LEVEL = ("a_sl", "a_tp", "buffer_pips")
LEVEL_GRID = {"a_sl": (0.75, 1.0, 1.5), "a_tp": (0.75, 1.0, 1.5), "buffer_pips": (1.0, 3.0)}
GENOME2 = ("kappa", "theta", "alpha", "tau0", "rho", "k_persist", "q_sl")   # v1.2: q_sl categorical {0.5, 0.75, 0.9}; p_urg inert, out of the genome
_BASE = {}


# ----------------------------------------------------------------------------- genome
def decode(x: list, base: dict) -> dict:
    d = dict(PZ.validate_genes(base))
    for k, u in zip(GENOME2, x):
        u = min(max(float(u), 0.0), 1.0)
        if k in PZ.DISCRETE:
            opts = PZ.DISCRETE[k]
            d[k] = opts[min(int(u * len(opts)), len(opts) - 1)]
        else:
            lo, hi = PZ.BOUNDS[k]
            d[k] = min(max(lo + (hi - lo) * u, lo), hi)
    return PZ.validate_genes(d)


def encode(g: dict) -> list:
    g = PZ.validate_genes(g)
    out = []
    for k in GENOME2:
        if k in PZ.DISCRETE:
            opts = PZ.DISCRETE[k]
            out.append((opts.index(min(opts, key=lambda o: abs(o - g[k]))) + 0.5) / len(opts))
        else:
            lo, hi = PZ.BOUNDS[k]
            out.append((float(g[k]) - lo) / (hi - lo))
    return out


def level_points() -> list:
    return [dict(zip(LEVEL, v)) for v in itertools.product(*(LEVEL_GRID[k] for k in LEVEL))]


def edge_extensions(best: dict) -> list:
    """The formal S1R rule on the PZ level axes: one step beyond each edge value, clipped to the PZ bounds; a value already
    at its bound is a BOUNDARY_HIT (no extension)."""
    ext = []
    for k, vals in LEVEL_GRID.items():
        vs = sorted(vals)
        v = float(best[k])
        if v not in (vs[0], vs[-1]):
            continue
        low = v == vs[0]
        step = (vs[1] - vs[0]) if low else (vs[-1] - vs[-2])
        lo, hi = PZ.BOUNDS[k]
        c = min(max(v - step if low else v + step, lo), hi)
        if abs(c - v) > 1e-12:
            ext.append({**best, k: round(c, 10)})
    return ext


# ----------------------------------------------------------------------------- one world (built once per batch)
class PZWorld:
    def __init__(self, cx, SCFG, AC, cell: dict, genes: dict, seed: int = 0):
        BM.STRATEGY, BM.PZ_GENES = "pz", PZ.validate_genes(genes)
        H.MENU_ORDERS_FN = PZ.truth_orders_fn(BM.PZ_GENES)
        self.cell, self.level = cell, {k: float(BM.PZ_GENES[k]) for k in LEVEL}
        w = BM.cell_world(cx, SCFG, AC, cell, seed)
        self.w = w
        _, view, cells = BM.strategy_for(w, SP.genes_c3(SP.default_doc("full_f4")), AC)
        self.view0, self.cells = view, cells
        self.ov = BM.oracle_view(w, view)
        self.oroll = BM.rollout_oracle(w, self.ov, cells)
        self.owit = BM.witnesses(cell, w, self.oroll, cells)
        self.sroll = BM.rollout_oracle_star(w, self.ov, cells)
        self.swit = BM.witnesses(cell, w, self.sroll, cells)
        cl1 = view["closer"].with_quality(1.0)
        memo = w.setdefault("_exit_memo", {})
        memo["sroll"], memo["st_star"], memo["st_hold"] = self.sroll, BM.exit_stats(w, self.sroll, cl1), BM.exit_stats(w, self.oroll, cl1)
        self.closer1 = cl1
        self.cache_base = dict(view["closer"].cache)
        self.oracle_trades = BM.annotate_sigma(w, [dict(t) for t in self.oroll["trades"]]) if cell.get("pair") else None

    def _reset(self):
        bc = getattr(self.view0["closer"], "bc", None)
        if bc is not None:
            bc.rows.clear()
        cache = self.view0["closer"].cache
        cache.clear()
        cache.update(self.cache_base)

    def evaluate(self, genes: dict) -> dict:
        g = PZ.validate_genes(genes)
        if any(abs(float(g[k]) - self.level[k]) > 1e-12 for k in LEVEL):
            raise SystemExit("PZ_LEVEL_GENES_MISMATCH")                   # the truth was built with other level genes: refused, never silent
        self._reset()
        view = dict(self.view0)
        view["pz_genes"] = g
        if float(self.cell["quality"]) < 0.999 and abs(float(g["q_sl"]) - float(self.view0["pz_genes"]["q_sl"])) > 1e-12:
            w_, cx_ = self.w, self.w["cx"]                               # q_sl shapes the orders and p_s below perfect outputs: the view is re-derived
            bars = w_["bars"] if w_.get("bars") is not None else cx_.ds.bars
            view.update(PZ.view_entries(bars, cx_.ds.sigma, cx_.ds.row_bar, w_["table"]["cost_row"], w_["pz_hd"], g, **w_["pz_offsets"]))
            view["menu_stamp"] = None
        strat = PZ.strategy_object(g)
        xv = self.w["scfg"].variant
        if xv is not None and getattr(xv, "sizing_mode", "") == "risk_budget":
            self.w["scfg"] = dataclasses.replace(self.w["scfg"], variant=dataclasses.replace(xv, rho=float(g["rho"])))
        roll = BM.rollout_cell(self.w, strat, view, self.cells)
        wit = BM.witnesses(self.cell, self.w, roll, self.cells)
        perfect = float(self.cell["quality"]) >= 0.999
        if perfect and "W8_early_close_at_switch" in wit:
            wit["W8_early_close_at_switch_v3"] = BM.w8_switch_v3(self.w, roll, self.sroll, self.closer1)
        vres = BM.validity(wit, self.owit, self.swit, BM.expected_witness_ids(self.cell, self.w), menu=True)
        fails = [k for k in vres["failing"] if perfect or k in D.QUALITY_INDEPENDENT]
        ep_summary, d7 = None, None
        if perfect:
            ex = BM.exit_layer_of(self.cell, self.w, strat, view, self.cells, roll, self.oroll, self.ov, light=True)
            fails += list(ex["failing_effective"])
            ep_summary = ex.get("episode_summary")
        else:
            st_s = BM.closer_d7_stats(self.w, roll, self.closer1)
            broll = BM.rollout_policy(self.w, PolicyPZBayes, strat, view, self.cells)
            st_b = BM.closer_d7_stats(self.w, broll, self.closer1)
            fails += BM.d7_failures(st_s, st_b)
            d7 = {side: {"E4": st_s[side]["E4"], "E4_bayes": st_b[side]["E4"], "regret": st_s[side]["regret_norm"], "regret_bayes": st_b[side]["regret_norm"]} for side in ("BUY", "SELL")}
        sh = float(roll["metrics"].get("sharpe_daily_sqrt365", float("nan")))
        out = {"key": self.cell["subset_key"], "sharpe": sh if math.isfinite(sh) else -10.0, "fails": fails, "opens": wit["summary"]["opens"], "quality": float(self.cell["quality"]),
               "d7": d7, "trades": len(roll["trades"]), "early_closes": sum(1 for t in roll["trades"] if t.get("exit_reason") == "EARLY"),
               "trade_stats": D.trade_stats(self.w, roll["trades"]), "activity": D.activity_stats(self.w, roll["trades"]), "horizon_mix": BM.horizon_mix(roll, view, self.w),
               "validity_v2": {k: vres[k] for k in ("not_evaluable", "unvalidated", "defects", "replay_conflicts", "evaluated")}}
        if self.cell.get("pair"):
            out["_trades"], out["_episodes"], out["_pair"], out["_role"] = BM.annotate_sigma(self.w, roll["trades"]), ep_summary, self.cell["pair"], self.cell["role"]
            out["_oracle_trades"] = self.oracle_trades
        self._reset()
        return out


class PolicyPZBayes(PZ.PolicyPZ):
    """QUALIFICATION_SPEC D7 for PZ: the PZ opener and the information-limited Bayes closer (BM.PolicyBayes's close rule)."""
    close_now = BM.PolicyBayes.close_now


def safe_eval(w: PZWorld, genes: dict) -> dict:
    try:
        return w.evaluate(genes)
    except KeyboardInterrupt:
        raise
    except SystemExit as e:
        return {"key": w.cell["subset_key"], "sharpe": -10.0, "fails": ["PZ_EVALUATION_FAILED"], "opens": 0, "quality": float(w.cell["quality"]), "d7": None, "trades": 0,
                "early_closes": 0, "trade_stats": {}, "activity": {"trades_per_week": 0.0, "reopen_1bar_share": float("nan")}, "horizon_mix": None, "error": str(e)[:300],
                **({"_trades": [], "_episodes": None, "_pair": w.cell["pair"], "_role": w.cell["role"]} if w.cell.get("pair") else {})}


# ----------------------------------------------------------------------------- world-major batches
def groups_of(subset: list) -> list:
    """Worlds grouped so that a twin / mirror pair is evaluated by one worker (W14 / E6 are judged there)."""
    g = {}
    for c in subset:
        g.setdefault(c.get("pair") or c["subset_key"], []).append(c)
    return sorted(g.values(), key=lambda x: x[0]["subset_key"])


def _worker(conn, items: list, cands: list, level: dict, out_dir: str, stop_file: str):
    import warnings
    warnings.filterwarnings("ignore")
    AC, cx, SCFG = _BASE["AC"], _BASE["cx"], _BASE["SCFG"]
    for grp in items:
        if Path(stop_file).exists():                                 # the soft stop: between worlds; finished groups stay on disk
            break
        tag = grp[0].get("pair") or grp[0]["subset_key"]
        f = Path(out_dir) / f"G_{tag.replace('/', '_')}.json"
        if f.exists():
            continue
        t0 = time.time()
        res = [[] for _ in cands]
        for c in grp:
            w = PZWorld(cx, SCFG, AC, c, {**cands[0], **level})
            for i, g in enumerate(cands):
                res[i].append(safe_eval(w, {**g, **level}))
            del w
            gc.collect()
            D._malloc_trim()
        for r in res:
            D.judge_pairs(r)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"group": tag, "results": res, "sec": time.time() - t0, "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}, default=float))
        tmp.rename(f)
    conn.send("done")


def evaluate_batch(subset: list, cands: list, level: dict, out_dir: Path, jobs: int, log=print, stop_file: Path | None = None) -> list | None:
    """Every candidate (genes) on every subset world, world-major across ``jobs`` forked workers; None when STOP stopped it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "CANDIDATES.json").write_text(json.dumps({"level": level, "candidates": cands}, indent=1))
    gs = groups_of(subset)
    shares = [[] for _ in range(jobs)]
    for i, g in enumerate(gs):
        shares[i % jobs].append(g)
    ctx = mp.get_context("fork")
    procs = []
    for sh in shares:
        if not sh:
            continue
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(b, sh, cands, level, str(out_dir), str(stop_file or (out_dir / "STOP"))), daemon=False)
        p.start()
        procs.append((p, a))
    for p, a in procs:
        p.join()
    files = {json.loads(f.read_text())["group"]: json.loads(f.read_text()) for f in out_dir.glob("G_*.json")}
    if len(files) < len(gs):
        log(json.dumps({"batch_incomplete": str(out_dir), "groups_done": len(files), "of": len(gs), "stop": bool(stop_file and Path(stop_file).exists())}))
        return None
    per = [[] for _ in cands]
    for g in gs:
        tag = g[0].get("pair") or g[0]["subset_key"]
        for i, r in enumerate(files[tag]["results"]):
            per[i].extend(r)
    out = []
    for i, g in enumerate(cands):
        res = per[i]
        n_fail = sum(len(r["fails"]) for r in res)
        out.append({"pz_design_version": PZ.DESIGN_VERSION, "params": {**g, **level}, "fitness_sharpe": float(np.mean([r["sharpe"] for r in res])), "n_failing_witnesses": n_fail, "feasible": n_fail == 0,
                    "activity": {"trades_per_week_mean": float(np.mean([r["activity"]["trades_per_week"] for r in res])),
                                 "reopen_1bar_share_median": float(np.nanmedian([r["activity"]["reopen_1bar_share"] for r in res])),
                                 "net_sigma_per_trade_median": float(np.nanmedian([r["trade_stats"].get("net_sigma_per_trade_median", np.nan) for r in res]))},
                    "per_world": res})
    return out


def _load_base(a, genes_doc: dict | None = None):
    if not _BASE:
        _BASE["AC"] = S.contract_for_sim(a.contract)
        cx, SCFG = S.load_fold("wf2022", a.admitted, a.csv_dir)
        doc = SP.load(a.params)
        ZZ.configure(doc)
        _BASE["cx"], _BASE["SCFG"] = cx, S.with_variant(SCFG, C3.exec_variant_of(doc))


def _stop(root: Path) -> bool:
    return (root / "STOP").exists()


def _archive(root: Path, stage: str, gen, recs: list):
    with open(root / "ARCHIVE.jsonl", "a") as fh:
        for r in recs:
            fh.write(json.dumps({"stage": stage, "generation": gen, **r}, default=float) + "\n")


def rank(recs: list) -> list:
    return sorted(recs, key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"]))


# ----------------------------------------------------------------------------- CLI
def cmd_measure(a):
    _load_base(a)
    sub = json.loads(Path(a.subset).read_text())[: a.n]
    t0 = time.time()
    root = Path(a.out)
    res = evaluate_batch(sub, [PZ.validate_genes({})] * a.reps, {k: PZ.CANONICAL[k] for k in LEVEL}, root / "measure", a.jobs, stop_file=root / "STOP")
    print(json.dumps({"worlds": len(sub), "candidates": a.reps, "wall_sec": time.time() - t0, "fitness": [r["fitness_sharpe"] for r in res or []],
                      "failing": [r["n_failing_witnesses"] for r in res or []],
                      "group_sec": [json.loads(f.read_text())["sec"] for f in (root / "measure").glob("G_*.json")],
                      "maxrss_mb": [json.loads(f.read_text())["maxrss_mb"] for f in (root / "measure").glob("G_*.json")]}, default=float))


def cmd_stage1(a):
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    if _stop(root):
        print("STOP flag present"); sys.exit(75)
    _load_base(a)
    sub = json.loads(Path(a.subset).read_text())
    log = lambda m: (print(m, flush=True), open(root / "PROGRESS.log", "a").write(m + "\n"))  # noqa: E731
    pts = level_points()
    recs = []
    for i, lv in enumerate(pts):
        d = root / "stage1" / f"P{i:02d}"
        r = evaluate_batch(sub, [PZ.validate_genes({})], lv, d, a.jobs, log, root / "STOP")
        if r is None:
            sys.exit(75)
        if not (d / "ARCHIVED").exists():
            _archive(root, "S1", i, r)
            (d / "ARCHIVED").write_text("1")
        recs.append({"point": i, **{k: v for k, v in r[0].items() if k != "per_world"}, "per_world": r[0]["per_world"]})
        log(json.dumps({"stage1_point": i, "level": lv, "sharpe": r[0]["fitness_sharpe"], "failing": r[0]["n_failing_witnesses"]}))
        if i == 0 or (i == len(pts) - 1):
            pass
    dg = D.degenerate_gen0([{"n_failing_witnesses": x["n_failing_witnesses"], "per_world": x["per_world"]} for x in recs])
    log(json.dumps({"stage1_gen0_degeneracy": dg}))
    if dg["degenerate"]:
        (root / "FAILED_DEGENERATE_GEN0_S1").write_text(json.dumps(dg, indent=1)); sys.exit(3)
    rk = rank(recs)
    ext = edge_extensions({k: rk[0]["params"][k] for k in LEVEL})
    for j, lv in enumerate(ext):
        d = root / "stage1" / f"X{j:02d}"
        r = evaluate_batch(sub, [PZ.validate_genes({})], lv, d, a.jobs, log, root / "STOP")
        if r is None:
            sys.exit(75)
        if not (d / "ARCHIVED").exists():
            _archive(root, "S1R", j, r)
            (d / "ARCHIVED").write_text("1")
        recs.append({"point": f"X{j}", **{k: v for k, v in r[0].items() if k != "per_world"}, "per_world": r[0]["per_world"]})
    rk = rank(recs)
    (root / "STAGE1_RESULT.json").write_text(json.dumps({"ranking": [{k: v for k, v in x.items() if k != "per_world"} for x in rk], "extensions": ext,
                                                          "frozen_level": {k: rk[0]["params"][k] for k in LEVEL}}, indent=1, default=float))
    log(json.dumps({"stage1_done": True, "frozen_level": {k: rk[0]["params"][k] for k in LEVEL}, "sharpe": rk[0]["fitness_sharpe"], "failing": rk[0]["n_failing_witnesses"]}))


def cmd_stage2(a):
    root = Path(a.out)
    if _stop(root):
        print("STOP flag present"); sys.exit(75)
    _load_base(a)
    sub = json.loads(Path(a.subset).read_text())
    log = lambda m: (print(m, flush=True), open(root / "PROGRESS.log", "a").write(m + "\n"))  # noqa: E731
    st1 = json.loads((root / (a.stage1_result or "STAGE1_RESULT.json")).read_text())
    level = st1["frozen_level"]
    tag = a.tag or "stage2"
    ck = root / tag / "CHECKPOINT.pkl"
    (root / tag).mkdir(parents=True, exist_ok=True)
    from deap import tools as DT
    rnd = random.Random(SEED)
    nb = len(GENOME2)
    if ck.exists():
        st = pickle.loads(ck.read_bytes())
    else:
        ini = json.loads(Path(a.init).read_text()) if a.init else {}
        ini = {k: v for k, v in ini.get("champion", ini).items() if k in PZ.GENES}       # a stage-2 RESULT.json or a plain genes document
        base = PZ.validate_genes({**ini, **level})
        st = {"gen": 0, "pending": [encode(base)] + [[rnd.random() for _ in range(nb)] for _ in range(a.population - 1)], "pop": None, "rng": rnd.getstate()}
        ck.write_bytes(pickle.dumps(st))
    rnd.setstate(st["rng"])
    while st["gen"] <= a.generations:
        g = st["gen"]
        if _stop(root):
            log(json.dumps({"STOP": True, "generation": g})); sys.exit(75)
        if st["pending"] is not None:
            cands = [{k: v for k, v in decode(x, {**PZ.CANONICAL, **level}).items()} for x in st["pending"]]
            r = evaluate_batch(sub, cands, level, root / tag / f"gen{g:02d}", a.jobs, log, root / "STOP")
            if r is None:
                sys.exit(75)
            evald = [{"x": x, "fitness_sharpe": rr["fitness_sharpe"], "n_failing_witnesses": rr["n_failing_witnesses"], "feasible": rr["feasible"]} for x, rr in zip(st["pending"], r)]
            _archive(root, tag, g, r)
            if g == 0:
                dg = D.degenerate_gen0([{"n_failing_witnesses": x["n_failing_witnesses"], "per_world": x["per_world"]} for x in r])
                log(json.dumps({"gen0_degeneracy": dg}))
                if dg["degenerate"]:
                    (root / tag / "FAILED_DEGENERATE_GEN0").write_text(json.dumps(dg, indent=1)); sys.exit(3)
                pop = evald
            else:
                pop = sorted(st["pop"], key=lambda q: (q["n_failing_witnesses"], -q["fitness_sharpe"]))[:ELITE] + evald
            best = min(pop, key=lambda q: (q["n_failing_witnesses"], -q["fitness_sharpe"]))
            log(json.dumps({"generation": g, "best_sharpe": best["fitness_sharpe"], "best_failing": best["n_failing_witnesses"], "feasible_in_pop": sum(1 for q in pop if q["feasible"])}))
            st = {"gen": g + 1, "pending": None, "pop": pop, "rng": rnd.getstate()}
            ck.write_bytes(pickle.dumps(st))
            continue
        pop = st["pop"]
        kids = []
        while len(kids) < a.population - ELITE:
            def tour():
                x, y = rnd.sample(pop, 2)
                return x if D.better(x, y) else y
            p1, p2 = list(tour()["x"]), list(tour()["x"])
            random.seed(rnd.random())
            if rnd.random() < 0.9:
                p1, p2 = DT.cxSimulatedBinaryBounded(p1, p2, eta=15.0, low=[0.0] * nb, up=[1.0] * nb)
            for ch in (p1, p2):
                DT.mutPolynomialBounded(ch, eta=20.0, low=[0.0] * nb, up=[1.0] * nb, indpb=1.0 / nb)
                kids.append(list(ch))
        st = {"gen": st["gen"], "pending": kids[: a.population - ELITE], "pop": pop, "rng": rnd.getstate()}
        ck.write_bytes(pickle.dumps(st))
    best = min(st["pop"], key=lambda q: (q["n_failing_witnesses"], -q["fitness_sharpe"]))
    champ = decode(best["x"], {**PZ.CANONICAL, **level})
    (root / tag / "RESULT.json").write_text(json.dumps({"champion": champ, "fitness_sharpe": best["fitness_sharpe"], "n_failing_witnesses": best["n_failing_witnesses"],
                                                         "population": st["pop"]}, indent=1, default=float))
    log(json.dumps({"stage2_done": tag, "champion": champ, "sharpe": best["fitness_sharpe"], "failing": best["n_failing_witnesses"]}))


def cmd_stage3(a):
    """The formal S3 rule on PZ: the top 5 stage-1 level points (the frozen one first) re-evaluated with the stage-2
    champion's open / close genes; another point no worse on the constraint whose paired mean Sharpe gain over the frozen
    point exceeds 1 SE (over the subset worlds) -> one more stage 2 from it (tag stage2b); then stop iterating."""
    root = Path(a.out)
    if _stop(root):
        print("STOP flag present"); sys.exit(75)
    _load_base(a)
    sub = json.loads(Path(a.subset).read_text())
    log = lambda m: (print(m, flush=True), open(root / "PROGRESS.log", "a").write(m + "\n"))  # noqa: E731
    st1 = json.loads((root / "STAGE1_RESULT.json").read_text())
    champ = json.loads((root / "stage2" / "RESULT.json").read_text())["champion"]
    top = [{k: x["params"][k] for k in LEVEL} for x in st1["ranking"][:5]]
    rows = []
    for i, lv in enumerate(top):
        r = evaluate_batch(sub, [{k: v for k, v in champ.items() if k not in LEVEL}], lv, root / "stage3" / f"T{i}", a.jobs, log, root / "STOP")
        if r is None:
            sys.exit(75)
        rows.append({"index": i, "params": lv, "fitness_sharpe": r[0]["fitness_sharpe"], "n_failing_witnesses": r[0]["n_failing_witnesses"],
                     "per_world_sharpe": {w["key"]: w["sharpe"] for w in r[0]["per_world"]}})
    f0 = rows[0]
    for r in rows[1:]:
        keys = sorted(set(r["per_world_sharpe"]) & set(f0["per_world_sharpe"]))
        d = np.array([r["per_world_sharpe"][k] - f0["per_world_sharpe"][k] for k in keys], float)
        se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("inf")
        r.update({"paired_mean_dsharpe_vs_frozen": float(d.mean()), "se": se,
                  "wins_by_more_than_1se": bool(r["n_failing_witnesses"] <= f0["n_failing_witnesses"] and d.mean() > se)})
    import tools.strategy_chain_f4 as CF
    dec = CF.s3_decision({"rows": rows[1:]})
    out = {"rows": rows, "decision": dec, "champion_stage2": champ}
    if dec["rerun_stage2"]:
        (root / "STAGE1_RESULT_S3.json").write_text(json.dumps({**st1, "frozen_level": dec["winner"]["params"]}, indent=1, default=float))
    (root / "STAGE3_RESULT.json").write_text(json.dumps(out, indent=1, default=float))
    log(json.dumps({"stage3_done": True, "rerun_stage2": dec["rerun_stage2"]}))


def cmd_gate_mirror(a):
    """The pre-launch mirror property test (coordinator order 2026-10-28) for PZ: on every hard pair's zero-cost twin
    (S_) and exact mirror (M_) at perfect heads, the canonical point and ``n`` random genomes (the frozen level genes)
    must give mirrored ledgers (BM.mirror_check) and no per-run W14 verdict. PZ reads no side-specific reference or gene,
    so the sealed form IS the code-symmetry form. PASS writes <out>/PROPERTY_PASSED_<sha> (this lane's own gate folder)."""
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    _load_base(a)
    sub = {c["subset_key"]: c for c in json.loads(Path(a.subset).read_text())}
    level = json.loads(Path(a.level).read_text()) if a.level else {k: PZ.CANONICAL[k] for k in LEVEL}
    rng = random.Random(a.seed)
    gens = [("canonical", PZ.validate_genes({**PZ.CANONICAL, **level}))] + [(f"random_{i}", decode([rng.random() for _ in GENOME2], {**PZ.CANONICAL, **level})) for i in range(a.n)]
    AC, cx, SCFG = _BASE["AC"], _BASE["cx"], _BASE["SCFG"]
    bad = 0
    with open(root / "MIRROR_PROPERTY.jsonl", "a") as fh:
        for pid in a.pairs.split(","):
            if _stop(root):
                print("STOP"); sys.exit(75)
            t0 = time.time()
            trades = {}
            for role in ("S", "M"):
                w = PZWorld(cx, SCFG, AC, sub[f"{role}_{pid}@q1.00"], {**PZ.CANONICAL, **level})
                trades[role] = [safe_eval(w, g) for _, g in gens]
                del w
                gc.collect()
            for (name, g), rs, rm in zip(gens, trades["S"], trades["M"]):
                m = BM.mirror_check(rs.get("_trades", []), rm.get("_trades", []))
                per_run = [f for f in rs["fails"] + rm["fails"] if f.startswith("W14")]
                ok = bool(m["pass"] and not per_run and "error" not in rs and "error" not in rm)
                bad += int(not ok)
                fh.write(json.dumps({"pair": pid, "genome": name, "genes": g, "pass": ok, "n": [m["n_a"], m["n_b"]], "bars_match": m["bars_match"], "sides_flipped": m["sides_flipped"],
                                     "max_net_dev_sigma": m["max_net_discrepancy_ex_financing_sigma"], "per_run_w14": per_run}, default=float) + "\n")
                fh.flush()
            print(json.dumps({"pair": pid, "genomes": len(gens), "sec": round(time.time() - t0)}), flush=True)
    sha = a.code_sha or "unknown"
    res = {"code_sha": sha, "failures": bad, "pairs": a.pairs.split(","), "genomes": len(gens), "level": level}
    (root / f"MIRROR_PROPERTY_RESULT_{sha}.json").write_text(json.dumps(res, indent=1, default=float))
    if bad == 0:
        (root / f"PROPERTY_PASSED_{sha}").write_text(json.dumps(res, default=float))
    print(json.dumps(res, default=float))
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("measure", "stage1", "stage2", "stage3", "gate-mirror"):
        p = sub.add_parser(name)
        p.add_argument("--subset", required=True)
        p.add_argument("--admitted", required=True)
        p.add_argument("--csv-dir", required=True)
        p.add_argument("--contract", default=str(S.CONTRACT))
        p.add_argument("--params", required=True, help="STRATEGY_PARAMS (full_f4 profile, the engine contract)")
        p.add_argument("--jobs", type=int, default=2)
        p.add_argument("--out", required=True)
        if name == "gate-mirror":
            p.add_argument("--pairs", default="LHS_46,LHS_48,LHS_50,LHS_52,LHS_53,LHS_56,LHS_57,LHS_61,LHS_62")
            p.add_argument("--n", type=int, default=50)
            p.add_argument("--seed", type=int, default=20261028)
            p.add_argument("--level", default=None)
            p.add_argument("--code-sha", default=None)
        if name == "measure":
            p.add_argument("--n", type=int, default=2)
            p.add_argument("--reps", type=int, default=2)
        if name == "stage2":
            p.add_argument("--population", type=int, default=POPULATION)
            p.add_argument("--generations", type=int, default=GENERATIONS)
            p.add_argument("--stage1-result", default=None)
            p.add_argument("--init", default=None)
            p.add_argument("--tag", default=None)
    a = ap.parse_args()
    {"measure": cmd_measure, "stage1": cmd_stage1, "stage2": cmd_stage2, "stage3": cmd_stage3, "gate-mirror": cmd_gate_mirror}[a.cmd](a)


if __name__ == "__main__":
    main()
