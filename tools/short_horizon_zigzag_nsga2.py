#!/usr/bin/env python3
"""ZigZag NSGA-II v2 driver (DEVELOPMENT): chunked, checkpointed, deterministic per seed.

  chunk     advance one optimization (opt in A1, A2, A4, B4; seed) by as many generations as fit in the
            CPU budget; checkpoint = population, fitness, hypervolume history and the evaluation cache
            (so a resumed chunk never re-simulates a parameter set). Writes DONE when the hypervolume of
            the last 20 generations changes < 1 % after >= 80 generations, or at max generations.
  finalize  union of the seeds' fronts per optimization, boundary report, 2023 (JOINT development) and
            2024 (PREVIOUSLY OBSERVED development) reads with a ledger, (b) Sharpe/lag of every (A) front
            point, the chosen expert, the fill-path cost (b+2 vs b+1) and demonstrations v2.

Optimizations: A{c} = hindsight expert (a), objectives Sharpe max / max drawdown % min, daily cap c in
{1, 2, 4}; B4 = causally confirmed version (b) on the executable path, objectives Sharpe max / max DD % min /
median confirmation lag min, with the TP multiple k_tp as a gene. Frequency rule, SL/TP/units rules and
realistic costs inside every evaluation; invalid -> penalty.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import zigzag as Z  # noqa: E402
from app.short_horizon import zigzag_opt as ZO  # noqa: E402
from tools.short_horizon_oracle import load_inputs  # noqa: E402

OPTS = {"A1": ("A", 1), "A2": ("A", 2), "A4": ("A", 4), "B4": ("B", 4)}
POP = 64


def space_for(kind, wide=False):
    if kind == "A":
        return ZO.Space(depth=(1, 400), deviation_pips=(0.5, 600.0), backstep=(0, 60)) if wide else ZO.SPACE_A
    return ZO.Space(depth=(1, 400), deviation_pips=(0.5, 600.0), backstep=(0, 60), k_tp=(0.1, 6.0)) if wide else ZO.SPACE_B


def key_of(p):
    return json.dumps(p, sort_keys=True)


def chunk(args):
    from deap import base, creator, tools
    run = Path(args.run) / args.opt / f"seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    if (run / "DONE").exists():
        print(json.dumps({"done": True}))
        return
    kind, cap = OPTS[args.opt]
    space = space_for(kind, args.wide)
    nobj = 2 if kind == "A" else 3
    bars, cal, digest = load_inputs(args.data)
    ev = ZO.Evaluator(bars, cal)
    if not hasattr(creator, f"Fit{nobj}"):
        creator.create(f"Fit{nobj}", base.Fitness, weights=(1.0, -1.0) if nobj == 2 else (1.0, -1.0, -1.0))
        creator.create(f"Ind{nobj}", list, fitness=getattr(creator, f"Fit{nobj}"))
    Ind = getattr(creator, f"Ind{nobj}")
    lo, hi = space.lower(), space.upper()
    tb = base.Toolbox()
    tb.register("mate", tools.cxSimulatedBinaryBounded, low=lo, up=hi, eta=15.0)
    tb.register("mutate", tools.mutPolynomialBounded, low=lo, up=hi, eta=20.0, indpb=1.0 / len(lo))
    tb.register("select", tools.selNSGA2)
    ck = run / "checkpoint.json"
    cache = {}
    if ck.exists():
        st = json.loads(ck.read_text())
        cache = st["cache"]
        pop = [Ind(g) for g in st["population"]]
        for ind, f in zip(pop, st["fitness"]):
            ind.fitness.values = tuple(f)
        pop = tb.select(pop, len(pop))
        gen, hv_hist, rng_state, evals, sims = st["generation"], st["hv"], st["rng"], st["evals"], st["sims"]
        rng = random.Random()
        rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
    else:
        rng = random.Random(1000 * args.seed + {"A1": 1, "A2": 2, "A4": 4, "B4": 7}[args.opt])
        pop, gen, hv_hist, evals, sims = [], 0, [], 0, 0

    def evaluate(ind):
        nonlocal evals, sims
        p, g = space.decode(ind)
        ind[:] = g
        k = key_of(p)
        evals += 1
        if k not in cache:
            sims += 1
            m, _ = (ev.eval_A(p, cap=cap) if kind == "A" else ev.eval_B(p, cap=cap))
            obj = ZO.objectives_A(m) if kind == "A" else ZO.objectives_B(m)
            cache[k] = {"obj": list(obj), "m": {x: m[x] for x in ("sharpe", "max_dd_pct", "orders_per_week", "valid", "trades")}
                        | ({"lag_median": m["lag_median"]} if kind == "B" else {})}
        return tuple(cache[k]["obj"])

    def hv_of(pop):
        front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        ref = ZO.REF_A if nobj == 2 else ZO.REF_B
        return ZO.hypervolume([ZO.to_min(i.fitness.values) for i in front], ref)

    t0 = time.process_time()
    if not pop:
        pop = [Ind([rng.uniform(a, b) for a, b in zip(lo, hi)]) for _ in range(POP)]
        for ind in pop:
            ind.fitness.values = evaluate(ind)
        pop = tb.select(pop, POP)
        hv_hist = [hv_of(pop)]
    done_reason = None
    while time.process_time() - t0 < args.budget_cpu_seconds:
        random.seed(rng.random())          # DEAP operators use the module RNG: reseed it from the chain RNG
        off = [Ind(list(i)) for i in tools.selTournamentDCD(pop, len(pop))]
        for a, b in zip(off[::2], off[1::2]):
            if rng.random() < 0.9:
                tb.mate(a, b)
            tb.mutate(a)
            tb.mutate(b)
            del a.fitness.values, b.fitness.values
        for ind in off:
            ind.fitness.values = evaluate(ind)
        pop = tb.select(pop + off, POP)
        gen += 1
        hv_hist.append(hv_of(pop))
        if ZO.converged(hv_hist, min_generations=args.min_generations, window=20, tol=0.01):
            done_reason = "CONVERGED"
            break
        if gen >= args.max_generations:
            done_reason = "MAX_GENERATIONS_NOT_CONVERGED"
            break
    st = rng.getstate()
    ck.write_text(json.dumps({"generation": gen, "population": [list(map(float, i)) for i in pop],
                              "fitness": [list(i.fitness.values) for i in pop], "hv": hv_hist,
                              "rng": [st[0], list(st[1]), st[2]], "evals": evals, "sims": sims, "cache": cache,
                              "opt": args.opt, "seed": args.seed, "wide": args.wide, "data_sha256": digest,
                              "space": {"lower": lo, "upper": hi}}))
    if done_reason:
        (run / "DONE").write_text(json.dumps({"reason": done_reason, "generation": gen, "hv_last": hv_hist[-1],
                                              "evals": evals, "sims": sims}) + "\n")
    print(json.dumps({"opt": args.opt, "seed": args.seed, "gen": gen, "hv": hv_hist[-1], "evals": evals, "sims": sims,
                      "cpu": round(time.process_time() - t0, 1), "done": done_reason}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["chunk"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--opt", choices=list(OPTS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--budget-cpu-seconds", type=float, default=95.0)
    ap.add_argument("--min-generations", type=int, default=80)
    ap.add_argument("--max-generations", type=int, default=200)
    a = ap.parse_args()
    chunk(a)


if __name__ == "__main__":
    main()
