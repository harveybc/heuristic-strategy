#!/usr/bin/env python
"""Strategy parameter optimisation by simulation (owner order 2026-10-27): DEAP with fitness = mean daily SHARPE over a
declared, stratified subset of behaviour-matrix worlds (every block / family, each world and its real-cost mirror, at
perfect heads AND at the operating quality 0.70); a candidate is FEASIBLE only if every sealed / effective witness
still passes on its perfect-head subset worlds and the quality-independent witnesses pass on the 0.70 worlds
(constraint domination by the number of failing witnesses; an infeasible point is never selected). No training: the
engine truth, the simulated heads and the ladder levels of every world are built ONCE per worker process; an
evaluation is the rollout of the parameterised C3Z strategy plus the witnesses. Population 16, 30 generations,
declared seed, elitism 2, every consumed coefficient of STRATEGY_PARAMS.json with its declared bounds as the genome.
Resumable (checkpoint per generation, archive of every evaluated point). The chosen point is re-verified on the FULL
matrix by the chain driver (tools/clean_round_chain/run_strategy_param_stage.sh).

CLI: subset (write the declared subset), measure (time one evaluation), run (the DEAP search), best (write the k-th
best feasible point as a STRATEGY_PARAMS file).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import strategy_params as SP  # noqa: E402
from app.short_horizon.expert_guided import strategy_zz as ZZ  # noqa: E402
from tools import strategy_behaviour_matrix as BM  # noqa: E402
from tools import strategy_hierarchy as H  # noqa: E402
from tools import strategy_output_simulation as S  # noqa: E402

SEED = 20261027
POPULATION, GENERATIONS, ELITE = 16, 30, 2
OPERATING_QUALITY = 0.70
# the genome: every STRATEGY_PARAMS coefficient the C3Z strategy consumes; the others are fixed, with the reason
GENOME = ("open_rate_l", "nop_veto", "close_rem_rate_l", "close_rem_rate_s", "close_p_l", "close_p_s", "allow_reverse", "observe_bars",
          "rr_margin_sigma", "ev_from_net", "close_margin_floor", "risk_reward_floor", "burst_window_bars", "burst_vol_ratio", "zz_room_tp_mult",
          "zz_fine_h", "zz_timing_gate", "zz_nop_gate", "zz_close_room", "close_p_requires_rem_below", "zz_theta_f", "zz_counter_net_frac", "zz_post_flip_bars", "closer_mode", "closer_adv_margin", "tp_mode", "k_tp_protect")
INTEGER = {"observe_bars", "burst_window_bars", "zz_fine_h", "zz_post_flip_bars"}
BINARY = {"allow_reverse", "ev_from_net", "zz_timing_gate", "zz_nop_gate", "closer_mode", "tp_mode"}
TIED = {"open_rate_s": "open_rate_l"}
FIXED = {"tp_a_l": "parametric TP / SL family: not consumed by C3Z (cell family)", "v_alpha_l": "parametric volume family: not consumed by C3Z (risk sizing)",
         "risk_pct": "execution contract (owner's sizing rule), not a strategy coefficient", "entry_cutoff_hours": "execution contract (Friday cutoff)",
         "max_entries_per_day": "execution contract (daily cap)", "confirmation_lag_bars": "a witness constant, not consumed by the strategy",
         "open_margin_eps": "numerical guard against float ties on the quantile sample", "forced_every_bars": "the exit audit's episode schedule, not a trading coefficient", "tp_grid": "the cell grid of the outcome heads", "sl_grid": "the cell grid of the outcome heads"}
QUALITY_INDEPENDENT = ("W16_sl_tp_every_order", "W17_no_open_in_burst", "W0_activity", "W12_missing", "W11_cutoff")


# ----------------------------------------------------------------------------- the declared subset
def build_subset(cells: list, per_level: int = 1, seed: int = SEED, mirror_at_quality: bool = False) -> list:
    """Stratified: per matrix level (every block / family) ``per_level`` base cells drawn with the declared seed; each
    world appears with its REAL-COST mirror (R_ = mirrored path, costs unchanged), at perfect heads and at quality 0.70."""
    rng = random.Random(seed)
    base = [c for c in cells if not c.get("mirror") and not c.get("sym_of")]
    by_level = {}
    for c in base:
        by_level.setdefault(c["level"], []).append(c)
    out = []
    for lvl in sorted(by_level):
        pick = rng.sample(by_level[lvl], min(per_level, len(by_level[lvl])))
        for c in pick:
            m = dict(c)
            m["id"], m["mirror"], m["mirror_real_cost_of"] = "R_" + c["id"], True, c["id"]
            m["legs_gen"], m["legs"], m["regime"] = list(c["legs"]), [H.MIRROR[x] for x in c["legs"]], H.MIRROR[c["regime"]]
            for w in (c, m):
                for q in ((1.0, OPERATING_QUALITY) if (mirror_at_quality or w is c) else (1.0,)):
                    x = dict(w)
                    x["quality"] = q
                    x["subset_key"] = f"{w['id']}@q{q:.2f}"
                    out.append(x)
    return out


# ----------------------------------------------------------------------------- one cached world and one evaluation
class CachedWorld:
    def __init__(self, cx, SCFG, AC, cell: dict, seed: int = 0):
        self.cell = cell
        w = BM.cell_world(cx, SCFG, AC, cell, seed)
        self.w = w
        genes0 = SP.genes_c3(SP.default_doc())
        _, view, cells = BM.strategy_for(w, genes0, AC)          # builds the view, the ladder (cached below) and the C3 gene map inputs
        self.view0, self.cells = view, cells
        bars = w.get("bars") if w.get("bars") is not None else w["cx"].ds.bars
        self.levels = ZZ.ladder_levels(bars, w["cx"].ds.sigma)
        self.bars = bars
        cx2, truth, table = w["cx"], w["truth"], w["table"]
        self.cal_rows = np.flatnonzero(cx2.obs_ok & (cx2.ds.period == BM.CAL))
        ref = C3.closer_reference_sides(table, S.closer_src_of(truth))
        self.gm = C3.gene_map(view, self.cal_rows, ref, cells, truth.e_time, truth.cost)
        self.ov = BM.oracle_view(w, view)
        orl = BM.rollout_oracle(w, self.ov, cells)                     # the oracle does not depend on the candidate: once per world
        self.oroll = orl
        owit = BM.witnesses(cell, w, orl, cells)
        self.defects = sorted(k for k, v in owit.items() if isinstance(v, dict) and "pass" in v and not k.endswith("_v2") and v["pass"] is False)   # per-cell rule
        self.oracle_sharpe = float(orl["metrics"].get("sharpe_daily_sqrt365", float("nan")))

    def evaluate(self, params: dict) -> dict:
        ZZ.configure(params)
        if "rr_margin_sigma" in params:
            C3.RR_MARGIN_SIGMA = float(params["rr_margin_sigma"])
        genes = SP.genes_c3(params)
        strat = C3.StrategyC3.from_genes(self.gm.to_absolute(genes), "nop_veto" in genes)
        view = dict(self.view0)
        q = ZZ.ladder_quantities(self.bars, self.w["cx"].ds.sigma, cached=self.levels)
        keep = np.isfinite(view["pn"])
        for k, v in ZZ.to_rows(q, self.w["cx"].ds.row_bar).items():
            view[k] = np.where(keep.reshape((-1,) + (1,) * (np.ndim(v) - 1)), v, np.nan)
        view["risk_burst"] = ZZ.burst_rows(self.bars, self.w["cx"].ds.sigma, self.w["cx"].ds.row_bar)
        if ZZ.PARAMS["ev_from_net"] >= 0.5:
            strat = dataclasses.replace(strat, open_margin=ZZ.net_open_margin(view, self.cal_rows, self.cells, float(genes["open_rate"])))
        self.w["ladder"] = q
        roll = BM.rollout_cell(self.w, strat, view, self.cells)
        wit = BM.witnesses(self.cell, self.w, roll, self.cells)
        perfect = float(self.cell["quality"]) >= 0.999
        fails = []
        for k, v in wit.items():
            if not (isinstance(v, dict) and "pass" in v) or k.endswith("_v2"):
                continue
            if not perfect and k not in QUALITY_INDEPENDENT:
                continue
            eff = wit.get(k + "_v2", v) if (k in self.defects and (k + "_v2") in wit) else v
            if eff["pass"] is False:
                fails.append(k)
        exit_fails = []
        if perfect:                                                   # the exit layer (sealed EXIT_SPEC) is a constraint: E0-E5, EP, E1', E2', E7, Z1, E8
            ex = BM.exit_layer_of(self.cell, self.w, strat, view, self.cells, roll, self.oroll, self.ov, light=True)
            exit_fails = list(ex["failing_effective"])
            fails += exit_fails
        sh = float(roll["metrics"].get("sharpe_daily_sqrt365", float("nan")))
        return {"key": self.cell["subset_key"], "sharpe": sh if math.isfinite(sh) else -10.0, "fails": fails, "opens": wit["summary"]["opens"],
                "net_usd": wit["summary"]["net_usd"], "quality": float(self.cell["quality"])}


# ----------------------------------------------------------------------------- genome <-> parameter document
def bounds() -> dict:
    return {k: list(SP.BOUNDS[k]) for k in GENOME}


def decode(x: list, base: dict) -> dict:
    d = dict(base)
    for k, u in zip(GENOME, x):
        lo, hi = SP.BOUNDS[k]
        v = lo + (hi - lo) * min(max(float(u), 0.0), 1.0)
        if k in INTEGER:
            v = int(round(v))
        if k in BINARY:
            v = 1.0 if v >= 0.5 * (lo + hi) else float(lo)
        d[k] = v
    for k, src in TIED.items():
        d[k] = d[src]
    return SP.validate(d)


def encode(doc: dict) -> list:
    return [(float(doc[k]) - SP.BOUNDS[k][0]) / (SP.BOUNDS[k][1] - SP.BOUNDS[k][0]) for k in GENOME]


# ----------------------------------------------------------------------------- the worker pool (one process per share of worlds)
_BASE = {}


def _worker(conn, share: list, fold: str, admitted: str, csv_dir: str, contract: str):
    import resource
    import warnings
    warnings.filterwarnings("ignore")
    AC, cx, SCFG = _BASE["AC"], _BASE["cx"], _BASE["SCFG"]             # loaded once in the parent, shared by fork (copy-on-write)
    worlds = []
    t0 = time.time()
    for c in share:
        worlds.append(CachedWorld(cx, SCFG, AC, c, 0))
    conn.send({"ready": [w.cell["subset_key"] for w in worlds], "build_sec": time.time() - t0, "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
               "defects": {w.cell["subset_key"]: w.defects for w in worlds}, "oracle_sharpe": {w.cell["subset_key"]: w.oracle_sharpe for w in worlds}})
    while True:
        msg = conn.recv()
        if msg is None:
            break
        t1 = time.time()
        res = [w.evaluate(msg) for w in worlds]
        conn.send({"results": res, "sec": time.time() - t1})


class Pool:
    def __init__(self, subset: list, jobs: int, fold: str, admitted: str, csv_dir: str, contract: str):
        ctx = mp.get_context("fork")
        if not _BASE:
            _BASE["AC"] = S.contract_for_sim(contract)
            cx, SCFG = S.load_fold(fold, admitted, csv_dir)
            _BASE["cx"], _BASE["SCFG"] = cx, S.with_variant(SCFG, C3.FULL_F3)
        shares = [subset[i::jobs] for i in range(jobs)]
        self.conns, self.procs = [], []
        for sh in shares:
            if not sh:
                continue
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, sh, fold, admitted, csv_dir, contract), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.ready = [c.recv() for c in self.conns]

    def evaluate(self, params: dict) -> dict:
        for c in self.conns:
            c.send(params)
        outs = [c.recv() for c in self.conns]
        res = [r for o in outs for r in o["results"]]
        n_fail = sum(len(r["fails"]) for r in res)
        sh = float(np.mean([r["sharpe"] for r in res]))
        return {"fitness_sharpe": sh, "n_failing_witnesses": n_fail, "feasible": n_fail == 0, "per_world": res, "wall_sec": max(o["sec"] for o in outs)}

    def close(self):
        for c in self.conns:
            c.send(None)
        for p in self.procs:
            p.join(timeout=30)


# ----------------------------------------------------------------------------- the search (constraint domination; elitism)
def better(a: dict, b: dict) -> bool:
    """Deb's constraint domination with one objective: fewer failing witnesses first, then the higher Sharpe."""
    if a["n_failing_witnesses"] != b["n_failing_witnesses"]:
        return a["n_failing_witnesses"] < b["n_failing_witnesses"]
    return a["fitness_sharpe"] > b["fitness_sharpe"]


def run_search(pool: Pool, base: dict, out: Path, population: int, generations: int, seed: int, log=print) -> dict:
    from deap import tools as DT
    rnd = random.Random(seed)
    ck = out / "CHECKPOINT.pkl"
    arch = out / "ARCHIVE.jsonl"
    if ck.exists():
        st = pickle.loads(ck.read_bytes())
        pop, gen0 = st["pop"], st["gen"] + 1
        rnd.setstate(st["rng"])
        log(json.dumps({"resume_from_generation": st["gen"]}))
    else:
        pop = [encode(base)] + [[rnd.random() for _ in GENOME] for _ in range(population - 1)]   # the canonical point is in generation 0
        pop = [{"x": x, **{k: v for k, v in _eval(pool, x, base, arch, 0).items()}} for x in pop]
        gen0 = 1
        ck.write_bytes(pickle.dumps({"pop": pop, "gen": 0, "rng": rnd.getstate()}))
    nb = len(GENOME)
    for g in range(gen0, generations + 1):
        pop.sort(key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"]))
        elite = [dict(r) for r in pop[:ELITE]]
        kids = []
        while len(kids) < population - ELITE:
            def tour():
                a, b = rnd.sample(pop, 2)
                return a if better(a, b) else b
            p1, p2 = list(tour()["x"]), list(tour()["x"])
            random.seed(rnd.random())
            if rnd.random() < 0.9:
                p1, p2 = DT.cxSimulatedBinaryBounded(p1, p2, eta=15.0, low=[0.0] * nb, up=[1.0] * nb)
            for ch in (p1, p2):
                DT.mutPolynomialBounded(ch, eta=20.0, low=[0.0] * nb, up=[1.0] * nb, indpb=1.0 / nb)
                kids.append(list(ch))
        kids = kids[:population - ELITE]
        pop = elite + [{"x": x, **_eval(pool, x, base, arch, g)} for x in kids]
        best = min(pop, key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"]))
        log(json.dumps({"generation": g, "best_sharpe": best["fitness_sharpe"], "best_failing": best["n_failing_witnesses"],
                        "feasible_in_pop": sum(1 for r in pop if r["feasible"])}))
        ck.write_bytes(pickle.dumps({"pop": pop, "gen": g, "rng": rnd.getstate()}))
    pop.sort(key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"]))
    return {"best": pop[0], "population": pop}


def _eval(pool: Pool, x: list, base: dict, arch: Path, gen: int) -> dict:
    doc = decode(x, base)
    r = pool.evaluate(doc)
    rec = {"generation": gen, "params": {k: doc[k] for k in GENOME}, "fitness_sharpe": r["fitness_sharpe"], "n_failing_witnesses": r["n_failing_witnesses"],
           "feasible": r["feasible"], "wall_sec": r["wall_sec"], "per_world": r["per_world"]}
    with open(arch, "a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")
    return {"fitness_sharpe": r["fitness_sharpe"], "n_failing_witnesses": r["n_failing_witnesses"], "feasible": r["feasible"]}


# ----------------------------------------------------------------------------- CLI
def cmd_subset(a):
    cells = json.loads(Path(a.cells).read_text())
    sub = build_subset(cells, a.per_level, SEED)
    Path(a.out).write_text(json.dumps(sub, indent=1))
    lv = {}
    for c in sub:
        lv[c["level"]] = lv.get(c["level"], 0) + 1
    print(json.dumps({"subset_worlds": len(sub), "levels": len(lv), "per_level": lv}))


def cmd_measure(a):
    sub = json.loads(Path(a.subset).read_text())[: a.n]
    base = SP.load(a.params)
    t0 = time.time()
    pool = Pool(sub, a.jobs, a.fold, a.admitted, a.csv_dir, a.contract)
    build = time.time() - t0
    times = []
    for _ in range(a.reps):
        t1 = time.time()
        r = pool.evaluate(base)
        times.append(time.time() - t1)
    pool.close()
    import resource
    print(json.dumps({"worlds": len(sub), "jobs": a.jobs, "build_sec_total": build, "build_sec_per_worker": [x["build_sec"] for x in pool.ready],
                      "worker_maxrss_mb": [x["maxrss_mb"] for x in pool.ready], "parent_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                      "eval_wall_sec": times, "eval_sec_per_world_per_job": float(np.mean(times)) * a.jobs / max(len(sub), 1),
                      "canonical_fitness_sharpe": r["fitness_sharpe"], "canonical_failing": r["n_failing_witnesses"],
                      "failing_by_world": {x["key"]: x["fails"] for x in r["per_world"] if x["fails"]}}, default=float))


def cmd_run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if BM.stop_requested(out) and os.environ.get("BEHAVIOUR_MATRIX_ALLOWED") != "1":
        print("STOP flag present: nothing started")
        raise SystemExit(75)
    sub = json.loads(Path(a.subset).read_text())
    base = SP.load(a.params)
    meta = {"seed": SEED, "population": a.population, "generations": a.generations, "elite": ELITE, "genome": GENOME, "bounds": bounds(), "fixed": FIXED, "tied": TIED,
            "subset": a.subset, "subset_worlds": len(sub), "operating_quality": OPERATING_QUALITY, "quality_independent_constraints": QUALITY_INDEPENDENT,
            "fitness": "mean daily Sharpe (sqrt 365) over the subset worlds", "constraint": "number of failing sealed / effective witnesses (perfect heads, per-cell rule) incl. the sealed exit layer E0-E5 / EP / E1' / E2' / E7 / Z1 / E8, + quality-independent witnesses (0.70)",
            "exit_spec_sha256": BM.EXIT_SPEC_SHA,
            "params_base_sha256": SP.sha256_of(a.params), "effective_rule": "per cell (corrected form only where the oracle fails the sealed form in that world)"}
    (out / "RUN_META.json").write_text(json.dumps(meta, indent=1, default=list))
    log = lambda m: (print(m, flush=True), open(out / "PROGRESS.log", "a").write(m + "\n"))  # noqa: E731
    t0 = time.time()
    pool = Pool(sub, a.jobs, a.fold, a.admitted, a.csv_dir, a.contract)
    log(json.dumps({"pool_ready_sec": time.time() - t0, "defects": {k: v for x in pool.ready for k, v in x["defects"].items() if v}}))
    res = run_search(pool, base, out, a.population, a.generations, SEED, log)
    pool.close()
    (out / "RESULT.json").write_text(json.dumps({"best": res["best"], "population": res["population"], "best_params": {k: v for k, v in zip(GENOME, res["best"]["x"])},
                                                 "wall_sec": time.time() - t0}, indent=1, default=float))
    log(json.dumps({"done": True, "best_sharpe": res["best"]["fitness_sharpe"], "best_failing": res["best"]["n_failing_witnesses"]}))


def cmd_best(a):
    """The k-th best FEASIBLE evaluated point of the archive as a STRATEGY_PARAMS document (for the full-matrix check)."""
    base = SP.load(a.params)
    recs = [json.loads(x) for x in Path(a.archive).read_text().splitlines() if x.strip()]
    seen, feas = set(), []
    for r in sorted(recs, key=lambda r: (r["n_failing_witnesses"], -r["fitness_sharpe"])):
        key = json.dumps(r["params"], sort_keys=True)
        if key in seen or not r["feasible"]:
            continue
        seen.add(key)
        feas.append(r)
    if a.k >= len(feas):
        print(json.dumps({"no_feasible_point_at_rank": a.k, "feasible_points": len(feas)}))
        raise SystemExit(3)
    r = feas[a.k]
    doc = dict(base)
    doc.update(r["params"])
    for k, src in TIED.items():
        doc[k] = doc[src]
    SP.validate(doc)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({"rank": a.k, "fitness_sharpe": r["fitness_sharpe"], "generation": r["generation"], "out": a.out, "sha256": SP.sha256_of(a.out)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--cells", required=True)
    s.add_argument("--per-level", type=int, default=1)
    s.add_argument("--out", required=True)
    for name in ("measure", "run"):
        r = sub.add_parser(name)
        r.add_argument("--subset", required=True)
        r.add_argument("--params", default=str(ROOT / "docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json"))
        r.add_argument("--fold", default="wf2022")
        r.add_argument("--admitted", required=True)
        r.add_argument("--csv-dir", required=True)
        r.add_argument("--contract", default=str(S.CONTRACT))
        r.add_argument("--jobs", type=int, default=10)
        if name == "measure":
            r.add_argument("--n", type=int, default=4)
            r.add_argument("--reps", type=int, default=2)
        else:
            r.add_argument("--population", type=int, default=POPULATION)
            r.add_argument("--generations", type=int, default=GENERATIONS)
            r.add_argument("--out", required=True)
    b = sub.add_parser("best")
    b.add_argument("--archive", required=True)
    b.add_argument("--params", default=str(ROOT / "docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json"))
    b.add_argument("--k", type=int, default=0)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    {"subset": cmd_subset, "measure": cmd_measure, "run": cmd_run, "best": cmd_best}[a.cmd](a)


if __name__ == "__main__":
    main()
