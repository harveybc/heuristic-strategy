#!/usr/bin/env python3
"""Hierarchical scenario catalog, synthetic generators and the TWO-STAGE optimisation (owner 2026-10-26).

L1 coarse regime (trend up / trend down / range) x volatility (low / high): synthetic paths on the REAL calendar (the
   engine's rules apply unchanged) and real sub-periods by realized-vol / trend terciles (strategy_output_simulation.quarter_tags)
L2 fine structure inside the coarse leg: oscillation amplitude and pivot density of the fine ZigZags (low / medium / high)
L3 events: shocks 2/3/4 sigma, regime switch mid-block, spread x5 24 h, missing runs 6-24 h, vol doubling (strategy_intervention)
L4 head quality: per-head AUC / MAE ladder, calibration drift x0.8 / x1.2, 1-bar lag, side asymmetry, openers-only /
   closers-only / NOP-only (strategy_output_simulation.quality_for)
L5 execution: cap 1/2/4, cutoff 12/24 h, risk 0.5/1/2 %, cost x1/2/3 (strategy_output_simulation.scfg_for)
Each leaf = generator config + expected behaviour + witness. Sampling = the full factorial L1 x L4 at the operating
quality + a Latin-hypercube design over the continuous knobs (drift, vol, fine amplitude, pivot density, quality,
cost factor, cap, cutoff, risk).

Stage A (synthetic): every coefficient optimised on the hierarchical dataset with the owner's composite fitness,
aggregated across leaves by MEAN and CVaR-10 % (worst decile) -> Pareto set + ROBUST REGION (points within 5 % of the
best mean fitness overall AND within 5 % of the best in every L1 regime) -> box bounds per gene.
Stage B (real): strategy_output_simulation.run with the gene bounds restricted to the Stage-A box, on the real
selection block, scored once on the decision block; the Stage-A optimum replayed on the same decision block gives the
out-of-block gain / loss of the re-optimisation.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import action_contract as ACT  # noqa: E402
from app.short_horizon.expert_guided import c2 as C2  # noqa: E402
from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import strategy_full as SF  # noqa: E402
from app.short_horizon import zigzag as Z  # noqa: E402
from tools import strategy_output_simulation as S  # noqa: E402

L1 = {"trend_up": +1.0, "trend_down": -1.0, "range": 0.0}
TREND_DRIFT_SD = 0.03                            # sustained drift of a trend leg in hourly-sd units per hour (~13 %/yr at sd 7e-4)
VOL = {"low": 0.7, "high": 1.6}                 # multiples of the real block's hourly return sd
L2 = {"low": (1.5, 0.5), "medium": (2.5, 1.0), "high": (4.0, 2.0)}   # (oscillation amplitude in sd units, pivot density multiple)
L4_FACTORIAL = ("base", "onesided", "opener_only", "closer_only", "nop_only", "cal12")
OPERATING_QUALITY = 0.70
PIP = 1e-4


# ----------------------------------------------------------------------------- synthetic generator (L1 x L2 on the real calendar)
def synthetic_path(index, base_sd: float, regime: str, vol: str, fine: str, seed: int, start: float = 1.10) -> pd.DataFrame:
    """Hourly OHLC on the given calendar index: drift by L1 (+/- TREND_DRIFT_SD sd per hour as a slow AR(1) drift: 0.03 sd
    per hour ~ 13 % per year for an hourly sd of 7e-4 -- the first version used 0.35 sd per hour, which collapsed a
    two-year trend path by e^-4 and made the engine refuse the short levels; see the mirror / trend witnesses), 0 for range,
    range = mean-reverting around the start), GARCH(1,1)-like volatility clustering scaled by VOL, plus an L2 fine
    oscillation (sinusoidal swings with random period 6-18 h, amplitude x and density x) that creates the fine ZigZag
    pivots. Witness: the realised drift sign, the sd ratio and the fine-pivot density match the leaf (tests)."""
    rng = np.random.default_rng(seed)
    n = len(index)
    sd = base_sd * VOL[vol]
    amp, dens = L2[fine]
    # GARCH-like variance
    h = np.full(n, sd ** 2)
    eps = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.08 * sd ** 2 + 0.10 * eps[t - 1] ** 2 + 0.82 * h[t - 1]
        eps[t] = math.sqrt(h[t]) * rng.standard_normal()
    drift = np.zeros(n)
    k = L1[regime]
    if k != 0.0:
        d = 0.0
        for t in range(n):
            d = 0.98 * d + 0.02 * (k * TREND_DRIFT_SD * sd) + 0.02 * sd * rng.standard_normal()
            drift[t] = d
    logp = np.log(start) + np.cumsum(drift + eps)
    if k == 0.0:                                          # range: pull towards the start level
        for t in range(1, n):
            logp[t] = logp[t] - 0.03 * (logp[t - 1] - math.log(start))
    # L2 fine oscillation: piecewise sinusoids with random periods, amplitude in sd units, density scales the frequency
    osc = np.zeros(n)
    t = 0
    while t < n:
        per = int(max(4, rng.integers(6, 19) / dens))
        a = amp * sd * rng.uniform(0.8, 1.2)
        seg = min(per, n - t)
        osc[t:t + seg] = a * np.sin(2 * np.pi * np.arange(seg) / per)
        t += seg
    logp = logp + osc
    c = np.exp(logp)
    intr = np.abs(rng.standard_normal(n)) * sd * 0.6
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + intr)
    lo = np.minimum(o, c) * (1 - intr)
    return pd.DataFrame({"OPEN": o, "HIGH": hi, "LOW": lo, "CLOSE": c}, index=index)


def fine_pivot_count(bars: pd.DataFrame, base_sd: float, mult: float = 2.0) -> int:
    """Fine ZigZag pivots: reversals of at least ``mult`` x base_sd x price (the fine-structure density witness)."""
    from app.short_horizon.expert_guided import oracle as OR
    h, l = bars["HIGH"].to_numpy(float), bars["LOW"].to_numpy(float)
    thr = float(mult * base_sd * np.nanmedian(bars["CLOSE"].to_numpy(float)))
    piv = OR.reversal_pivots(h, l, thr)[0]
    return int(len(piv))


def leaf_witness(bars: pd.DataFrame, base_sd: float, regime: str, vol: str, fine: str) -> dict:
    r = np.diff(np.log(bars["CLOSE"].to_numpy(float)))
    n_piv = fine_pivot_count(bars, base_sd)
    return {"drift_sign": float(np.sign(r.sum())), "expected_drift_sign": L1[regime], "sd_ratio": float(r.std() / base_sd), "expected_sd_mult": VOL[vol],
            "fine_pivots_per_100_bars": float(100.0 * n_piv / max(len(r), 1)), "fine": fine, "expected_fine_order": "low < medium < high"}


# ----------------------------------------------------------------------------- the hierarchy and the sampling design
def lhs(n: int, dims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = np.zeros((n, dims))
    for j in range(dims):
        perm = rng.permutation(n)
        u[:, j] = (perm + rng.random(n)) / n
    return u


KNOBS = {"drift": (-1.0, 1.0), "vol_mult": (0.6, 1.8), "fine_amp": (0.4, 2.0), "fine_dens": (0.5, 2.0), "quality": (0.55, 0.90),
         "cost_mult": (1.0, 3.0), "cap": (1, 4), "cutoff_h": (12.0, 24.0), "risk_pct": (0.005, 0.02)}


def catalog(n_lhs: int = 24, seed: int = 0) -> list:
    """Every leaf: id, level, generator config, expected behaviour, witness. L1 x L2 synthetic leaves at the operating
    quality, the full factorial L1 x L4 at the operating quality, and n_lhs Latin-hypercube leaves over the knobs."""
    leaves = []
    for reg in L1:
        for vol in VOL:
            for fine in L2:
                leaves.append({"id": f"L1L2_{reg}_{vol}_{fine}", "level": "L1xL2", "regime": reg, "vol": vol, "fine": fine, "variant": "base", "quality": OPERATING_QUALITY,
                               "exec": {}, "expected": f"openers follow the {reg} drift; closers pay in the {fine}-density fine swings; activity >= 2/wk",
                               "witness": "leaf_witness: drift sign, sd ratio, fine-pivot density"})
    for reg in L1:
        for vol in VOL:
            for v4 in L4_FACTORIAL:
                leaves.append({"id": f"L1L4_{reg}_{vol}_{v4}", "level": "L1xL4", "regime": reg, "vol": vol, "fine": "medium", "variant": v4, "quality": OPERATING_QUALITY,
                               "exec": {}, "expected": f"head-quality variant {v4} under {reg}/{vol}: the informative heads must still pay; uninformative ones must not open",
                               "witness": "simulated-head achieved quality per block (simulate_table)"})
    u = lhs(n_lhs, len(KNOBS), seed)
    for i in range(n_lhs):
        x = {k: lo + (hi - lo) * u[i, j] for j, (k, (lo, hi)) in enumerate(KNOBS.items())}
        reg = "trend_up" if x["drift"] > 0.33 else "trend_down" if x["drift"] < -0.33 else "range"
        vol = "high" if x["vol_mult"] > 1.15 else "low"
        fine = "high" if x["fine_amp"] > 1.4 else "low" if x["fine_amp"] < 0.7 else "medium"
        leaves.append({"id": f"LHS_{i:02d}", "level": "LHS", "regime": reg, "vol": vol, "fine": fine, "variant": "base", "quality": round(x["quality"], 3),
                       "exec": {"cost_mult": round(x["cost_mult"], 2), "cap": int(round(x["cap"])), "cutoff_h": round(x["cutoff_h"], 1), "risk_pct": round(x["risk_pct"], 4)},
                       "knobs": {k: round(v, 4) for k, v in x.items()}, "expected": "the robust region must keep this leaf's fitness within 5 % of its best",
                       "witness": "leaf_witness + the leaf's own fitness archive"})
    return leaves


# ----------------------------------------------------------------------------- Stage A
def leaf_world(cx_real, SCFG, AC, leaf: dict, calendar_period: tuple, seed: int) -> dict:
    """A synthetic world for one leaf: synthetic bars on the real calendar of ``calendar_period`` (and the bars around
    it), its own engine truth and simulated heads at the leaf's quality; the strategy is scored on ``calendar_period``."""
    ds = cx_real.ds
    r = np.diff(np.log(ds.bars["CLOSE"].to_numpy(float)))
    base_sd = float(np.nanstd(r))
    bars = synthetic_path(ds.bars.index, base_sd, leaf["regime"], leaf["vol"], leaf["fine"], seed)
    from tools.expert_guided_e1t import Ctx
    sigma, sigma_rule = Z.causal_sigma(bars), Z.causal_range_sigma(bars)
    ds2 = dataclasses.replace(ds, bars=bars, sigma=sigma, sigma_rule=sigma_rule)
    cx2 = Ctx(ds2)
    cx2.obs_ok = cx_real.obs_ok.copy()
    S2 = S.scfg_for("base", SCFG)
    ex = leaf.get("exec") or {}
    xv = S2.variant if S2.variant is not None else C3.FULL_F3
    if ex.get("cost_mult"):
        S2 = dataclasses.replace(S2, profile=S2.profile.scaled(float(ex["cost_mult"])))
    if ex.get("cap"):
        S2 = dataclasses.replace(S2, max_entries_per_day=int(ex["cap"]))
    if ex.get("cutoff_h") or ex.get("risk_pct"):
        S2 = dataclasses.replace(S2, variant=dataclasses.replace(xv, entry_cutoff_hours_before_flat=float(ex.get("cutoff_h", xv.entry_cutoff_hours_before_flat or 24.0)),
                                                                 risk_pct=float(ex.get("risk_pct", xv.risk_pct))))
    truth = S.build_truth(cx2, S2, "synthetic", AC["outcomes"]["cells"], AC, only_periods=(S.CAL_BLOCK, S.SEL_BLOCK))
    q = S.quality_for(leaf["variant"], float(leaf["quality"]))
    table, sim = S.simulate_table(cx2, S2, truth, q, seed)
    return {"cx": cx2, "scfg": S2, "truth": truth, "table": table, "sim": sim, "witness": leaf_witness(bars, base_sd, leaf["regime"], leaf["vol"], leaf["fine"]), "period": calendar_period}


def composite(c: dict, scales: dict) -> float:
    """The owner's composite: Sharpe + annualised return + orders/day, each divided by its scale (c_k = 1 / median |term|)."""
    if not (math.isfinite(c.get("sharpe", float("nan"))) and c.get("trades", 0) > 0):
        return float("nan")
    return float(c["sharpe"] / scales["sharpe"] + c["r_ann"] / scales["r_ann"] + c["opd"] / scales["opd"])


def cvar(x: np.ndarray, alpha: float = 0.10) -> float:
    x = np.sort(np.asarray(x, float)[np.isfinite(np.asarray(x, float))])
    if not len(x):
        return float("nan")
    k = max(1, int(math.ceil(alpha * len(x))))
    return float(x[:k].mean())


def _leaf_scorer(w, fam, bounds, cells, AC):
    """The per-leaf scoring closure (gene map frozen on the leaf's calibration block, per-bar cost, selection-block score)."""
    cx, SCFG, truth, table = w["cx"], w["scfg"], w["truth"], w["table"]

    cal_rows = np.flatnonzero(cx.obs_ok & (cx.ds.period == S.CAL_BLOCK))
    sel_rows = np.flatnonzero(cx.obs_ok & (cx.ds.period == S.SEL_BLOCK))
    view_rows = np.union1d(cal_rows, sel_rows)
    table["cost_row"] = S.row_costs(cx, SCFG, truth.cost - C3.cost_sigma_of(cx, SCFG, 0.0, cal_rows))   # per-bar cost BEFORE the view (the perbar gate)
    view = C2.restrict(table, view_rows)
    ref = C3.closer_reference_sides(table, S.closer_src_of(truth))
    sig = np.asarray(cx.ds.sigma, float)
    min_opw = float(AC["strategy"]["calibration_min_trades_per_week"])
    if fam is None:
        gm = C3.gene_map(view, cal_rows, ref, cells, truth.e_time, truth.cost)
        make = lambda g, _gm=gm: C3.StrategyC3.from_genes(_gm.to_absolute(g), "nop_veto_rate" in g)  # noqa: E731
        Pol, roll = S.PolicyC3PerBar, ACT.rollout_with_trades
    else:
        gm = SF.gene_map_full(view, cal_rows, ref, cells, truth.e_time, truth.cost, bounds)
        make = lambda g, _gm=gm: SF.StrategyFull.from_genes(_gm.to_absolute(g), "nop_veto_rate" in g, fam[0], fam[1])  # noqa: E731
        Pol, roll = S.policy_factory(S.PolicyFullPerBar, SCFG), SF.rollout_with_volume
    score = S.score_fn(cx, SCFG, cells, truth.e_time, truth.cost, AC, S.SEL_BLOCK, sig, min_opw, Pol, roll)
    return (score, view, make)

def stage_a(worlds: list, bounds: dict, family: str, AC, seed: int, population: int, generations: int, out: Path) -> dict:
    """NSGA-II over every gene with two objectives: the MEAN composite over the leaves and its CVaR-10 % (worst decile),
    under Deb's constraint domination on the leaf-average activity (CI lower bound >= 2/wk) and side share (<= 0.90);
    every evaluated point is archived with its per-leaf components."""
    from deap import base, creator, tools
    import random
    fam = S.FAMILIES[family]
    names = list(bounds)
    lo = [float(bounds[g][0]) for g in names]
    hi = [float(bounds[g][1]) for g in names]
    cells = [tuple(c) for c in AC["outcomes"]["cells"]]
    # per-leaf scoring closures (gene maps frozen on each leaf's calibration block)
    scorers = []
    kept, excluded = [], []
    for w in worlds:
        try:
            scorers.append(_leaf_scorer(w, fam, bounds, cells, AC))
            kept.append(w)
        except (SystemExit, ValueError, KeyError) as e:          # a leaf whose calibration block cannot freeze the gene map (e.g. no short-side
            w["excluded"] = str(e)                                # reference visits in a one-sided synthetic world) is excluded and recorded
            excluded.append({"id": w.get("leaf", {}).get("id"), "reason": str(e)})
        w.pop("truth", None); w.pop("table", None); w.pop("sim", None)      # memory: the scorer holds the restricted view only (78 leaves OOM-killed a 3G cap)
    worlds = kept
    if not scorers:
        raise SystemExit("STAGE_A_NO_FEASIBLE_LEAF")
    import gc
    gc.collect()
    archive = []
    scales = {"sharpe": 1.0, "r_ann": 0.2, "opd": 0.6}     # provisional; recalibrated from the first generation (declared rule c_k = 1/median|term|)

    global _SCORERS
    _SCORERS = scorers
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(min(int(a_workers()), len(scorers))) if len(scorers) > 1 else None

    def evaluate_point(g: dict) -> dict:
        if pool is None:
            comps = [score(view, make(g)) for score, view, make in scorers]
        else:
            comps = pool.map(_score_leaf, [(i, g) for i in range(len(scorers))])
        fit = np.array([composite(c, scales) for c in comps])
        opw_lb = np.array([c.get("opw_ci_lb", c["opw"]) if math.isfinite(c.get("opw_ci_lb", float("nan"))) else 0.0 for c in comps])
        side = np.array([c.get("side_max_share", 0.0) if math.isfinite(c.get("side_max_share", float("nan"))) else 0.0 for c in comps])
        agent = np.array([c.get("agent_closes", 0) for c in comps])
        mean_fit = float(np.nanmean(fit)) if np.isfinite(fit).any() else float("nan")
        cv = cvar(fit)
        viol = max(0.0, 2.0 - float(opw_lb.mean())) + max(0.0, float(side.mean()) - 0.90) + (0.0 if agent.sum() > 0 else 1.0) + (1.0 if not math.isfinite(mean_fit) else 0.0)
        return {"genes": g, "mean": mean_fit, "cvar10": cv, "valid": viol == 0.0, "violation": viol,
                "per_leaf": [{k: c.get(k) for k in ("sharpe", "r_ann", "opd", "opw", "opw_ci_lb", "side_max_share", "agent_closes", "trades", "dd", "lcb")} for c in comps]}

    if not hasattr(creator, "StageAFit"):
        creator.create("StageAFit", base.Fitness, weights=(1.0, 1.0))
        creator.create("StageAInd", list, fitness=creator.StageAFit)
    state = random.getstate()
    random.seed(int(seed))
    tb = base.Toolbox()
    tb.register("ind", lambda: creator.StageAInd([random.uniform(a, b) for a, b in zip(lo, hi)]))
    tb.register("pop", tools.initRepeat, list, tb.ind)
    tb.register("mate", tools.cxSimulatedBinaryBounded, low=lo, up=hi, eta=15.0)
    tb.register("mutate", tools.mutPolynomialBounded, low=lo, up=hi, eta=20.0, indpb=1.0 / len(names))
    tb.register("select", tools.selNSGA2)
    BIG = 1e6

    def objectives(r):
        if r["valid"]:
            return (r["mean"], r["cvar10"])
        return (-BIG - r["violation"], -BIG - r["violation"])

    cache = {}

    def ev(ind):
        key = tuple(round(float(x), 6) for x in ind)
        if key not in cache:
            r = evaluate_point(dict(zip(names, key)))
            r["generation"] = gen[0]
            cache[key] = r
            archive.append(r)
        return objectives(cache[key])
    gen = [0]
    pop = tb.pop(n=population)
    for ind in pop:
        ind.fitness.values = ev(ind)
    # recalibrate the composite scales on the first generation's valid points (c_k = 1 / median |term|)
    terms = {"sharpe": [], "r_ann": [], "opd": []}
    for r in archive:
        for c in r["per_leaf"]:
            if c["sharpe"] is not None and math.isfinite(c["sharpe"]):
                for k in terms:
                    terms[k].append(abs(float(c[k])))
    for k in terms:
        if terms[k] and float(np.median(terms[k])) > 1e-6:
            scales[k] = float(np.median(terms[k]))
    cache.clear()
    archive.clear()
    for ind in pop:
        ind.fitness.values = ev(ind)
    pop = tb.select(pop, len(pop))
    for g_ in range(1, generations + 1):
        gen[0] = g_
        off = tools.selTournamentDCD(pop, len(pop))
        off = [creator.StageAInd(list(o)) for o in off]
        for i in range(0, len(off) - 1, 2):
            if random.random() < 0.9:
                tb.mate(off[i], off[i + 1])
            tb.mutate(off[i])
            tb.mutate(off[i + 1])
        for ind in off:
            ind.fitness.values = ev(ind)
        pop = tb.select(pop + off, population)
        print(json.dumps({"stage_a": family, "gen": g_, "evals": len(archive), "best_mean": max((r["mean"] for r in archive if r["valid"] and math.isfinite(r["mean"])), default=None)}), flush=True)
    random.setstate(state)
    if pool is not None:
        pool.close()
    valid = [r for r in archive if r["valid"] and math.isfinite(r["mean"]) and math.isfinite(r["cvar10"])]
    pareto = _pareto(valid)
    region = robust_region_escalated(valid, worlds, names)
    doc = {"family": family, "genes": names, "bounds": bounds, "population": population, "generations": generations, "seed": seed, "scales": scales,
           "leaves": len(worlds), "leaves_excluded": excluded, "evaluations": len(archive), "valid_points": len(valid), "pareto": pareto, "robust_region": region,
           "best_mean": max(valid, key=lambda r: r["mean"]) if valid else None, "best_cvar": max(valid, key=lambda r: r["cvar10"]) if valid else None}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"STAGE_A_{family}.json").write_text(json.dumps(doc, indent=1, default=float))
    import gzip
    with gzip.open(out / f"STAGE_A_{family}_archive.json.gz", "wt") as fh:
        json.dump(archive, fh, default=float)
    return doc


_SCORERS = []


def a_workers() -> int:
    import os
    return int(os.environ.get("STAGE_A_WORKERS", "8"))


def _score_leaf(args):
    i, g = args
    score, view, make = _SCORERS[i]
    return score(view, make(g))


def _pareto(points: list) -> list:
    front = []
    for p in points:
        dom = any((q["mean"] >= p["mean"] and q["cvar10"] >= p["cvar10"] and (q["mean"] > p["mean"] or q["cvar10"] > p["cvar10"])) for q in points)
        if not dom:
            front.append({"genes": p["genes"], "mean": p["mean"], "cvar10": p["cvar10"]})
    return sorted(front, key=lambda r: -r["mean"])


TOLERANCES = (0.05, 0.10, 0.20, 0.30, 0.50)
MIN_ROBUST_POINTS = 4


def robust_region_escalated(valid: list, worlds: list, names: list) -> dict:
    """The owner's 5 % rule first; when it admits fewer than MIN_ROBUST_POINTS points (regimes trade off against each
    other: no point is near-best in trend-up AND trend-down AND range at once) the tolerance is widened along
    TOLERANCES until the region holds at least that many points. The tolerance used and the 5 % result are recorded."""
    first = robust_region(valid, worlds, names, TOLERANCES[0])
    for tol in TOLERANCES:
        r = robust_region(valid, worlds, names, tol)
        if r["points"] >= MIN_ROBUST_POINTS:
            r["tolerance_used"] = tol
            r["at_5pct"] = {"points": first["points"], "box": first["box"]}
            return r
    r = robust_region(valid, worlds, names, TOLERANCES[-1])
    r["tolerance_used"] = TOLERANCES[-1]
    r["at_5pct"] = {"points": first["points"], "box": first["box"]}
    return r


def robust_region(valid: list, worlds: list, names: list, tol: float = 0.05) -> dict:
    """Points within tol of the best mean fitness overall AND within tol of the best per-L1-regime mean fitness in every
    regime -> the box (min / max per gene) they span, with the count."""
    if not valid:
        return {"points": 0, "box": {}, "tolerance": tol, "rule": "within tol of the best mean overall and in every L1 regime"}
    regimes = {}
    for i, w in enumerate(worlds):
        regimes.setdefault(w["leaf"]["regime"], []).append(i)
    best = max(r["mean"] for r in valid)
    per_reg_best = {}
    for reg, idx in regimes.items():
        per_reg_best[reg] = max(float(np.nanmean([composite_from(r["per_leaf"][i]) for i in idx])) for r in valid)
    keep = []
    for r in valid:
        if r["mean"] < best - tol * abs(best):
            continue
        ok = True
        for reg, idx in regimes.items():
            m = float(np.nanmean([composite_from(r["per_leaf"][i]) for i in idx]))
            if m < per_reg_best[reg] - tol * abs(per_reg_best[reg]):
                ok = False
                break
        if ok:
            keep.append(r)
    box = {g: [min(r["genes"][g] for r in keep), max(r["genes"][g] for r in keep)] for g in names} if keep else {}
    return {"points": len(keep), "box": box, "per_regime_best": per_reg_best, "best_mean": best, "tolerance": tol,
            "rule": f"within {tol:.0%} of the best mean overall and within {tol:.0%} of the best mean in every L1 regime", "members": [r["genes"] for r in keep[:50]]}


_SCALES = {"sharpe": 1.0, "r_ann": 0.2, "opd": 0.6}


def composite_from(c: dict, scales: dict | None = None) -> float:
    s_ = scales or _SCALES
    if c.get("sharpe") is None or not math.isfinite(c["sharpe"]) or not c.get("trades"):
        return float("nan")
    return float(c["sharpe"] / s_["sharpe"] + c["r_ann"] / s_["r_ann"] + c["opd"] / s_["opd"])


# ----------------------------------------------------------------------------- CLI
def cmd_catalog(a):
    leaves = catalog(a.n_lhs, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(leaves, indent=1))
    print(json.dumps({"leaves": len(leaves), "levels": {k: sum(1 for x in leaves if x["level"] == k) for k in ("L1xL2", "L1xL4", "LHS")}}))


def cmd_stage_a(a):
    AC = S.contract_for_sim(a.contract)
    cx, SCFG = S.load_fold(a.fold, a.admitted, a.csv_dir)
    SCFG = S.with_variant(SCFG, C3.FULL_F3)
    leaves = json.loads(Path(a.catalog).read_text())
    if a.levels:
        leaves = [x for x in leaves if x["level"] in a.levels.split(",")]
    if a.max_leaves:
        leaves = leaves[:a.max_leaves]
    worlds = []
    t0 = time.time()
    for i, leaf in enumerate(leaves):
        w = leaf_world(cx, SCFG, AC, leaf, S.SEL_BLOCK, a.seed * 1000 + i)
        w["leaf"] = leaf
        w["achieved"] = w["sim"]["achieved"]
        worlds.append(w)
        print(json.dumps({"world": leaf["id"], "witness": w["witness"], "sec": round(time.time() - t0)}), flush=True)
    fam = S.FAMILIES[a.family]
    bounds = SF.gene_bounds(*fam) if fam else {k: list(v) for k, v in C3.Q_BOUNDS.items()}
    doc = stage_a(worlds, bounds, a.family, AC, a.seed, a.population, a.generations, Path(a.out))
    doc_leaves = [{"id": w["leaf"]["id"], "witness": w["witness"], "achieved": w["achieved"], "excluded": w.get("excluded")} for w in worlds]
    (Path(a.out) / f"STAGE_A_{a.family}_leaves.json").write_text(json.dumps(doc_leaves, indent=1, default=float))
    print(json.dumps({"family": a.family, "leaves": len(worlds), "evaluations": doc["evaluations"], "valid": doc["valid_points"], "pareto": len(doc["pareto"]),
                      "robust_points": doc["robust_region"]["points"], "best_mean": (doc["best_mean"] or {}).get("mean")}))


def cmd_robust_region(a):
    """Recompute the robust region of a finished Stage A from its archive (per-leaf components) and the catalog."""
    import gzip
    out = Path(a.out)
    doc = json.loads((out / f"STAGE_A_{a.family}.json").read_text())
    pts = json.loads(gzip.open(out / f"STAGE_A_{a.family}_archive.json.gz", "rt").read())
    pts = pts if isinstance(pts, list) else pts.get("archive", pts)
    leaves = json.loads((out / f"STAGE_A_{a.family}_leaves.json").read_text())
    cat = {x["id"]: x for x in json.loads(Path(a.catalog).read_text())}
    worlds = [{"leaf": cat[x["id"]]} for x in leaves if not x.get("excluded")]
    valid = [p for p in pts if p.get("valid") and p["mean"] is not None and np.isfinite(p["mean"])]
    region = robust_region_escalated(valid, worlds, doc["genes"])
    doc["robust_region_previous"] = doc.get("robust_region")
    doc["robust_region"] = region
    (out / f"STAGE_A_{a.family}.json").write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({"family": a.family, "points": region["points"], "tolerance_used": region.get("tolerance_used"), "at_5pct": region.get("at_5pct", {}).get("points"), "box": region["box"]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("catalog")
    c.add_argument("--n-lhs", type=int, default=24)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    s = sub.add_parser("stage-a")
    s.add_argument("--fold", default="wf2022")
    s.add_argument("--admitted", required=True)
    s.add_argument("--csv-dir", required=True)
    s.add_argument("--contract", default=str(S.CONTRACT))
    s.add_argument("--catalog", required=True)
    s.add_argument("--levels", default="")
    s.add_argument("--max-leaves", type=int, default=0)
    s.add_argument("--family", default="c3", choices=list(S.FAMILIES))
    s.add_argument("--population", type=int, default=32)
    s.add_argument("--generations", type=int, default=16)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    r = sub.add_parser("robust-region", help="recompute the robust region of a finished Stage A from its archive")
    r.add_argument("--family", default="c3", choices=list(S.FAMILIES))
    r.add_argument("--catalog", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"catalog": cmd_catalog, "stage-a": cmd_stage_a, "robust-region": cmd_robust_region}[a.cmd](a)


if __name__ == "__main__":
    main()
