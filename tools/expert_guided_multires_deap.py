#!/usr/bin/env python3
"""DEAP NSGA-II over the multi-resolution label thresholds (owner 2026-10-24: NOP and its thresholds are optimizable
values; "no hay que tomarlo a la ligera"). The genes are ``multires.Thresholds`` (theta_f, theta_mae, theta_vol,
theta_close, f_close_take; min_votes fixed at 1) on declared linear bounds; the objectives are EXACTLY the registered
optimum's study (ZigZag v2, front A4_wide): Sharpe (daily, sqrt 365) max and max drawdown % min of the LABEL
DEMONSTRATION (``multires.demonstration_policy``: decide after bar b, fill at open b+1, SL 2 sigma, TP = remaining gain
to the optimal close) replayed with ``Ctx.rollout`` on DECISION 2020-2022 under the replay contract
(``effective_contracts("sac", 2.0)``: 1 entry/day, 1:50, activity >= 2 orders/week as validity; invalid -> penalty, i.e.
dominated). Three seeds; convergence = hypervolume of the first front changes < 1 % over the last 20 generations after
>= 80; union of the seeds' fronts; VALIDATION_2023 (JOINT development) read for every union-front point with a ledger;
choice = best 2023 Sharpe among points valid on both; OBSERVED_2024 read ONCE for the chosen point only (ledger);
TEST (2025) is never read. Comparators on the same rows: the default-threshold demonstration and the registered
optimum's own demonstration (the owner's closure-table rule). HINDSIGHT LIMITATION: every figure here is a replay of
hindsight labels; it measures the quality of the TRAINING TARGET, never a tradable strategy, and the only real test is
whether the student generalizes on the inner block in the clean round.

  run   --admitted <dir> --csv-dir <dir> --out <dir> [--pop 48 --max-generations 200 --min-generations 80 --seeds 1,2,3
        --workers 3]   (CPU only; memory-capped by the caller)
Writes FINAL.json, CHOSEN_THRESHOLDS.json (ready for ``--multires-thresholds``), READ_LEDGER.jsonl, seed_<s>.json
checkpoints, CLOSURE_TABLE.md."""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon import zigzag_opt as ZO  # noqa: E402
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import multires as MR  # noqa: E402
from tools.expert_guided_e1t import Ctx, summary  # noqa: E402
from tools.expert_guided_e3m import effective_contracts  # noqa: E402

GENES = (("theta_f", 0.1, 0.9), ("theta_mae", 0.1, 1.0), ("theta_vol", 0.05, 1.0), ("theta_close", 0.02, 0.3), ("f_close_take", 0.5, 1.0))
LO, HI = [g[1] for g in GENES], [g[2] for g in GENES]
PERIODS = {"DECISION_2020_2022": ("2020-01-01", "2023-01-01"), "VALIDATION_2023": ("2023-01-01", "2024-01-01"),
           "OBSERVED_2024": ("2024-01-01", "2025-01-01")}
KEEP = ("sharpe_daily_sqrt365", "max_drawdown_pct", "orders_per_week", "valid", "trades", "win_rate", "net_profit_usd")
BUDGET = {"pop": 48, "max_generations": 200, "min_generations": 80, "window": 20, "tol": 0.01, "seeds": [1, 2, 3],
          "crossover_prob": 0.9, "eta_crossover": 15.0, "eta_mutation": 20.0, "indpb": 1 / len(GENES), "gene_rounding": 4}
_G = {}      # worker globals (fork): cx, tables, contract


def thresholds_of(genes) -> MR.Thresholds:
    return MR.Thresholds(**{name: float(round(v, BUDGET["gene_rounding"])) for (name, _, _), v in zip(GENES, genes)}, min_votes=1)


def key_of(genes):
    return tuple(round(float(v), BUDGET["gene_rounding"]) for v in genes)


def replay(cx, tb, th, period, cfg):
    m = summary(cx.rollout(PERIODS[period], MR.demonstration_policy(tb, th), cfg)[0])
    return {k: (bool(m[k]) if k == "valid" else float(m[k])) for k in KEEP if k in m}


def objectives(m):
    return (m["sharpe_daily_sqrt365"], m["max_drawdown_pct"]) if m["valid"] else (ZO.PENALTY["sharpe"], ZO.PENALTY["dd"])


def _eval(genes):
    return replay(_G["cx"], _G["tb"], thresholds_of(genes), "DECISION_2020_2022", _G["cfg"])


def nondominated(points):
    """points: list of dicts with 'obj' (sharpe, dd). Keep those not dominated (sharpe >=, dd <=, one strict)."""
    out = []
    for p in points:
        dom = any((q["obj"][0] >= p["obj"][0] and q["obj"][1] <= p["obj"][1] and q["obj"] != p["obj"]) for q in points)
        if not dom and p["key"] not in {o["key"] for o in out}:
            out.append(p)
    return out


def bound_hits(front):
    hits = {}
    for i, (name, lo, hi) in enumerate(GENES):
        tol = 1e-3 * (hi - lo)
        hits[name] = {"lower": int(sum(abs(p["genes"][i] - lo) <= tol for p in front)),
                      "upper": int(sum(abs(hi - p["genes"][i]) <= tol for p in front)), "n": len(front)}
    return hits


def closure_rows(final):
    """The owner's closure table (2026-09-21 rule) from FINAL.json: model metric + scale, naive on the same rows, skill,
    literature value + source, comparability. One row per period and comparator."""
    rows = []
    ch = final.get("chosen")
    if ch is None:
        return [{"period": "ALL", "metric": "Sharpe daily sqrt365 / max DD %", "model": "NO_VALID_POINT", "naive": "-", "skill": "-",
                 "literature": "NOT_COMPARABLE: hindsight demonstration under a private replay contract", "comparability": "NO_NEW_MEASUREMENT"}]
    for period in ("DECISION_2020_2022", "VALIDATION_2023", "OBSERVED_2024"):
        m = ch["metrics"].get(period)
        if m is None:
            continue
        model = f"Sharpe {m['sharpe_daily_sqrt365']:.2f} / DD {m['max_drawdown_pct']:.2f} % / {m['orders_per_week']:.1f} orders/wk / valid {m['valid']}"
        for name, comp in final["comparators"].items():
            c = comp.get(period)
            if c is None:
                rows.append({"period": period, "metric": "Sharpe daily sqrt365 / max DD % (hindsight label demonstration, replay contract)",
                             "model": model, "naive": f"{name}: NOT_READ", "skill": "NOT_COMPARABLE: comparator not read (single-read rule for 2024)",
                             "literature": "NOT_COMPARABLE: hindsight demonstration, no published comparator",
                             "comparability": "same rows available only for the chosen point"})
                continue
            rows.append({"period": period, "metric": "Sharpe daily sqrt365 / max DD % (hindsight label demonstration, replay contract)",
                         "model": model,
                         "naive": f"{name}: Sharpe {c['sharpe_daily_sqrt365']:.2f} / DD {c['max_drawdown_pct']:.2f} % / {c['orders_per_week']:.1f} orders/wk / valid {c['valid']}",
                         "skill": f"delta Sharpe {m['sharpe_daily_sqrt365'] - c['sharpe_daily_sqrt365']:+.2f}; delta DD {m['max_drawdown_pct'] - c['max_drawdown_pct']:+.2f} pp",
                         "literature": "NOT_COMPARABLE: hindsight demonstration under a private replay contract; no published value",
                         "comparability": "same bars, same contract (1 entry/day, 1:50, >= 2 orders/wk validity), same costs; both hindsight"})
    return rows


def closure_markdown(final) -> str:
    rows = closure_rows(final)
    md = ["# Multi-resolution thresholds: closure table (hindsight label demonstrations; NOT a strategy result)", "",
          "| period | metric | model (chosen thresholds) | naive, same rows | skill | literature value + source | comparability |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r['period']} | {r['metric']} | {r['model']} | {r['naive']} | {r['skill']} | {r['literature']} | {r['comparability']} |" for r in rows]
    ch = final.get("chosen")
    if ch:
        md += ["", "Chosen thresholds: `" + json.dumps(ch["thresholds"], sort_keys=True) + "`",
               f"Rule: best VALIDATION_2023 Sharpe among union-front points valid on 2020-2022 and 2023 ({final['union_front_size']} points; "
               f"{final['n_valid_both']} valid on both). Bound hits: " + json.dumps(final["bound_hits"])]
    md += ["", "Convergence:", "", "| seed | generations | stop | HV gen 0 | HV final | delta last 20 | evaluations | simulations | seconds |",
           "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in final["chains"]:
        hv = s["hv"]
        d = (abs(hv[-1] - hv[-1 - BUDGET["window"]]) / hv[-1 - BUDGET["window"]]) if len(hv) > BUDGET["window"] and hv[-1 - BUDGET["window"]] > 0 else float("nan")
        md.append(f"| {s['seed']} | {s['generations']} | {s['stop']} | {hv[0]:.2f} | {hv[-1]:.2f} | {100 * d:.3f} % | {s['evals']} | {s['sims']} | {s['seconds']:.0f} |")
    md += ["", final["limitation"]]
    return "\n".join(md) + "\n"


def chain(seed, pool, cache, out, budget):
    from deap import base, creator, tools
    if not hasattr(creator, "FitSD"):
        creator.create("FitSD", base.Fitness, weights=(1.0, -1.0))
        creator.create("IndSD", list, fitness=creator.FitSD)
    Ind = creator.IndSD
    tb = base.Toolbox()
    tb.register("mate", tools.cxSimulatedBinaryBounded, low=LO, up=HI, eta=budget["eta_crossover"])
    tb.register("mutate", tools.mutPolynomialBounded, low=LO, up=HI, eta=budget["eta_mutation"], indpb=budget["indpb"])
    tb.register("select", tools.selNSGA2)
    rng = random.Random(seed)
    ck = out / f"seed_{seed}.json"
    if ck.exists() and json.loads(ck.read_text()).get("stop"):
        return json.loads(ck.read_text())

    def evaluate(pop):
        todo = [i for i in pop if key_of(i) not in cache]
        keys = list({key_of(i) for i in todo})
        for k, m in zip(keys, pool.map(_eval, [list(k) for k in keys])):
            cache[k] = m
        for i in pop:
            i.fitness.values = objectives(cache[key_of(i)])
        return len(keys)

    def hv_of(pop):
        front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        return ZO.hypervolume([ZO.to_min(i.fitness.values) for i in front], ZO.REF_A)

    t0 = time.time()
    pop = [Ind([rng.uniform(a, b) for a, b in zip(LO, HI)]) for _ in range(budget["pop"])]
    evals = sims = 0
    sims += evaluate(pop)
    evals += len(pop)
    pop = tb.select(pop, len(pop))
    hv = [hv_of(pop)]
    gen, stop = 0, None
    while stop is None:
        random.seed(rng.random())
        off = [Ind(list(i)) for i in tools.selTournamentDCD(pop, len(pop))]
        for a, b in zip(off[::2], off[1::2]):
            if rng.random() < budget["crossover_prob"]:
                tb.mate(a, b)
            tb.mutate(a)
            tb.mutate(b)
            del a.fitness.values, b.fitness.values
        sims += evaluate(off)
        evals += len(off)
        pop = tb.select(pop + off, len(pop))
        gen += 1
        hv.append(hv_of(pop))
        if ZO.converged(hv, min_generations=budget["min_generations"], window=budget["window"], tol=budget["tol"]):
            stop = "CONVERGED"
        elif gen >= budget["max_generations"]:
            stop = "MAX_GENERATIONS_NOT_CONVERGED"
        rec = {"seed": seed, "generations": gen, "stop": stop, "hv": hv, "evals": evals, "sims": sims, "seconds": time.time() - t0,
               "population": [[float(x) for x in i] for i in pop], "fitness": [list(map(float, i.fitness.values)) for i in pop]}
        ck.write_text(json.dumps(rec))
        print(json.dumps({"seed": seed, "gen": gen, "hv": round(hv[-1], 3), "evals": evals, "sims": sims, "sec": round(time.time() - t0)}), flush=True)
    return rec


def cmd_run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = dict(BUDGET, pop=a.pop, max_generations=a.max_generations, min_generations=a.min_generations, seeds=[int(s) for s in a.seeds.split(",")])
    if budget["pop"] % 4:
        raise SystemExit("POP_MUST_BE_MULTIPLE_OF_4")
    ds = DG.load(a.admitted, a.csv_dir, scaling="fixed_train", contract="consolidated")
    cx = Ctx(ds)
    tb = MR.Tables.from_ctx(cx, MR.LADDER, DG.EXPERT)
    S, X_ = effective_contracts("sac", 2.0)
    _G.update(cx=cx, tb=tb, cfg=X_)
    led = out / "READ_LEDGER.jsonl"
    cache = {}
    cache_file = out / "CACHE.json"
    if cache_file.exists():
        cache.update({tuple(json.loads(k)): v for k, v in json.loads(cache_file.read_text()).items()})
    chains = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for seed in budget["seeds"]:
            chains.append(chain(seed, pool, cache, out, budget))
            cache_file.write_text(json.dumps({json.dumps(list(k)): v for k, v in cache.items()}))
    # union front on DECISION
    pts = {}
    for c in chains:
        for g in c["population"]:
            k = key_of(g)
            m = cache[k]
            pts[k] = {"key": k, "genes": list(k), "thresholds": dataclasses.asdict(thresholds_of(k)), "obj": list(objectives(m)), "metrics": {"DECISION_2020_2022": m}}
    front = nondominated(list(pts.values()))
    front.sort(key=lambda p: -p["obj"][0])
    # VALIDATION_2023: one ledger entry for the whole union front (joint development, as the ZigZag v2 study)
    for p in front:
        p["metrics"]["VALIDATION_2023"] = replay(cx, tb, thresholds_of(p["genes"]), "VALIDATION_2023", X_)
    ZO.ledger(led, "VALIDATION_2023", "JOINT_DEVELOPMENT", "multires thresholds union front validation", n_points=len(front))
    comps = {"default_thresholds": {}, "registered_optimum_demonstration": {}}
    for period in ("DECISION_2020_2022", "VALIDATION_2023"):
        comps["default_thresholds"][period] = replay(cx, tb, MR.Thresholds(), period, X_)
        m = summary(cx.rollout(PERIODS[period], cx.expert, X_)[0])
        comps["registered_optimum_demonstration"][period] = {k: (bool(m[k]) if k == "valid" else float(m[k])) for k in KEEP if k in m}
    ZO.ledger(led, "VALIDATION_2023", "JOINT_DEVELOPMENT", "comparators (default thresholds, registered optimum demonstration)", n_points=2)
    valid_both = [p for p in front if p["metrics"]["DECISION_2020_2022"]["valid"] and p["metrics"]["VALIDATION_2023"]["valid"]]
    chosen = max(valid_both, key=lambda p: p["metrics"]["VALIDATION_2023"]["sharpe_daily_sqrt365"]) if valid_both else None
    if chosen is not None:
        chosen["metrics"]["OBSERVED_2024"] = replay(cx, tb, thresholds_of(chosen["genes"]), "OBSERVED_2024", X_)
        ZO.ledger(led, "OBSERVED_2024", "PREVIOUSLY_OBSERVED_DEVELOPMENT", "multires chosen thresholds, single read", n_points=1)
        (out / "CHOSEN_THRESHOLDS.json").write_text(json.dumps(chosen["thresholds"], sort_keys=True, indent=1) + "\n")
    final = {"schema": "expert_guided_multires_deap.v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "budget_preregistered": budget, "genes": [{"name": n, "lower": lo, "upper": hi} for n, lo, hi in GENES], "min_votes": 1,
             "objectives": "Sharpe daily sqrt365 max, max drawdown % min, of the label demonstration on DECISION_2020_2022; invalid -> penalty " + json.dumps(ZO.PENALTY),
             "contract": {"exec": "sac", "min_orders_per_week": 2.0, "max_entries_per_day": X_.max_entries_per_day, "leverage_note": "1:50 (replay contract)"},
             "periods": PERIODS, "period_status": ZO.PERIOD_STATUS, "test_2025": "NEVER_READ",
             "data": {"admitted": ds.meta.get("pins"), "split_contract": ds.meta.get("split_contract"), "periods_of_rows": ds.meta.get("periods"),
                      "availability_note": "label availability follows the row's period end under the consolidated contract (TRAIN to 2023-07-01, INNER_A to 2024-01-01)"},
             "chains": [{k: v for k, v in c.items() if k not in ("population", "fitness")} for c in chains],
             "union_front": front, "union_front_size": len(front), "n_valid_both": len(valid_both), "bound_hits": bound_hits(front),
             "chosen": chosen, "comparators": comps,
             "choice_rule": "best VALIDATION_2023 Sharpe among union-front points valid on DECISION and VALIDATION_2023; OBSERVED_2024 read once for the chosen point only",
             "limitation": "HINDSIGHT LIMITATION: every figure is a replay of hindsight labels (pivots known after the fact); it measures the quality of the TRAINING TARGET under the replay contract, never a tradable strategy or a profitability claim; the only real test is the student's generalization on the inner block in the clean round."}
    (out / "FINAL.json").write_text(json.dumps(final, indent=1, default=float))
    (out / "CLOSURE_TABLE.md").write_text(closure_markdown(final))
    print(json.dumps({"front": len(front), "valid_both": len(valid_both), "chosen": None if chosen is None else chosen["thresholds"],
                      "bound_hits": final["bound_hits"]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--admitted", required=True)
    r.add_argument("--csv-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--pop", type=int, default=BUDGET["pop"])
    r.add_argument("--max-generations", type=int, default=BUDGET["max_generations"])
    r.add_argument("--min-generations", type=int, default=BUDGET["min_generations"])
    r.add_argument("--seeds", default=",".join(map(str, BUDGET["seeds"])))
    r.add_argument("--workers", type=int, default=3)
    r.set_defaults(f=cmd_run)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
