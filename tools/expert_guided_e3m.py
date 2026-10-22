#!/usr/bin/env python3
"""E3-M: modular specialists guided by the ZigZag expert (owner idea 2026-10-20), PROGRAMMED, to run once the
capacity ladder names an architecture (or with the registered u=2 design if it names none). DEVELOPMENT.

Hypothesis: training one model per TASK (open / close) and per DIRECTION (long / short) on the same expert
labels may beat one joint 4-class policy. Declared 2 x 2 factorial on IDENTICAL rows, rounds, seeds, trunk:

  JOINT     one policy {HOLD, LONG, SHORT, CLOSE} (+ TP/SL heads)                     = the E1-T reference
  TASK      OPENER {HOLD, LONG, SHORT} (+ TP/SL) on flat states; CLOSER {HOLD, EXIT} on positioned states
  DIR       LONG-model {HOLD, OPEN, CLOSE} on flat-or-long states; SHORT-model likewise on flat-or-short
  FULL      OPEN-L, OPEN-S {no, yes} (+ TP/SL) on flat states; CLOSE-L, CLOSE-S {HOLD, EXIT} on own position

Labels, partitioned by the visited state. FLAT: the hindsight LegOracle action (open long / open short / hold).
POSITIONED (every variant, so the factorial is not confounded by the label): EXIT when, in hindsight, the open
order's SL is hit before its TP (or it ends negative at the forced flat), HOLD otherwise = the owner's early
close ("cut the trade the stop would kill"); SL and TP stay in force either way. Round 0 = the expert's own
trajectory for flat states plus EXPLORATION positions (the TRAIN-prior random policy's open orders, round 0
only) for the closers, because the expert's own orders end at TP and carry no EXIT example; rounds 1..3 =
DAgger with the COMPOSED policy visiting states (same as E1-T). A module whose partition has one class keeps
its previous round's model (ledgered), never crashes the run. Composition: positioned d -> closer_d
decides EXIT; after an EXIT (or when flat) the ARBITER chooses among the openers on the flat-equivalent state
(position fields zeroed) and may emit the opposite direction = close-and-reverse, as the oracle does.

Arbiter variants (chosen on INNER_A, ledgered; pL/pS = each direction's open probability, tp/sl = that
direction's sigma multiples from its own heads):
  prob            open argmax(pL, pS) when it exceeds tau (tau from a rate grid on INNER_A, as E0)
  expected_gain   score_d = p_d * tp_d - (1 - p_d) * sl_d - COST_SIGMA; open argmax when > 0
  gain_over_risk  score_d / sl_d; open argmax when > 0
Decision: variant x arbiter with the highest seed-mean INNER_A Sharpe among frequency-valid ones (ties ->
fewer models); INNER_B reported for all; 2024 read ONCE for the chosen. No TEST.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from tools.expert_guided_e0 import RATE_GRID, jsonable, ledger  # noqa: E402
from tools.expert_guided_e1t import EXPERT_CFG, MUSASHI_CFG, STUDENT_CFG, Ctx, cat, pack, summary  # noqa: E402

SEEDS = (20261019, 20261020, 20261021)
ROUNDS = 3
VARIANTS = ("JOINT", "TASK", "DIR", "FULL")
ARBITERS = ("prob", "expected_gain", "gain_over_risk")
COST_SIGMA = 0.1          # declared round-trip cost in sigma units for the gain arbiters (not broker-verified)
EXIT = 1                  # closer class


def flat_state(st):
    s = np.array(st, np.float32)
    s[:5] = 0.0
    return s


# ----------------------------------------------------------------------------- labels per partition
def partition(st):
    return int(st[0])     # 0 flat, 1 long, -1 short


def labels_for(variant, st, a):
    """{module: (class, is_open_label)} for one visited state with oracle/expert action a."""
    pos = partition(st)
    out = {}
    if variant == "JOINT":
        out["joint"] = (a, a in (E.LONG, E.SHORT))
    elif variant == "TASK":
        if pos == 0:
            out["opener"] = (a if a in (E.LONG, E.SHORT) else E.HOLD, a in (E.LONG, E.SHORT))
        else:
            out["closer"] = (EXIT if a != E.HOLD else 0, False)
    elif variant == "DIR":
        if pos == 0:
            out["long"] = (1 if a == E.LONG else 0, a == E.LONG)
            out["short"] = (1 if a == E.SHORT else 0, a == E.SHORT)
        else:
            out["long" if pos > 0 else "short"] = (2 if a != E.HOLD else 0, False)
    else:
        if pos == 0:
            out["open_l"] = (1 if a == E.LONG else 0, a == E.LONG)
            out["open_s"] = (1 if a == E.SHORT else 0, a == E.SHORT)
        else:
            out["close_l" if pos > 0 else "close_s"] = (EXIT if a != E.HOLD else 0, False)
    return out


MODULES = {"JOINT": {"joint": (4, 2)}, "TASK": {"opener": (3, 2), "closer": (2, 0)},
           "DIR": {"long": (3, 2), "short": (3, 2)},
           "FULL": {"open_l": (2, 2), "open_s": (2, 2), "close_l": (2, 0), "close_s": (2, 0)}}   # module -> (n_cls, n_reg)


def collect(cx, period, act, cfg=STUDENT_CFG):
    """Like Ctx.rollout(collect=True) but every positioned row also carries (direction, entry, sl, tp, ff_bar).
    ``period`` is a DG.PERIODS name or an explicit (start, end) pair (refit sweep, 2026-10-21)."""
    a, b = DG.PERIODS[period] if isinstance(period, str) else period
    env = E.BarEnv(cx.ds.bars, cx.ds.cal, cx.ds.sigma, cx.ds.sigma_rule, a, b, cfg)
    vis = []
    for bar in range(env.first, env.last):
        r = cx.rob[bar]
        st = env.state(bar)
        if r >= 0 and cx.obs_ok[r]:
            action, tp, sl = act(bar, r, st, env)
            if cx.lab_ok[r]:
                p = env.pos
                info = None if p is None else (p.direction, p.entry_price, p.sl, p.tp,
                                               int(np.searchsorted(cx.v, env.ff[int(env.seg_of[bar])])))
                vis.append((r, bar, st, action, tp, sl, info))
        else:
            action, tp, sl = E.HOLD, 1.5, 2.0
        env.step(bar, action, tp, sl)
    return env.result(), vis


def exit_label(cx, bar, info):
    """1 (EXIT) if the open order's SL is hit before its TP from bar+1 on (both in one bar -> SL first,
    conservative) or it ends negative at the forced flat; else 0 (HOLD)."""
    d, entry, sl, tp, ffb = info
    h, l, c = cx.h, cx.l, cx.c
    for k in range(bar + 1, min(ffb, len(c))):
        hit_sl = l[k] <= sl if d > 0 else h[k] >= sl
        hit_tp = h[k] >= tp if d > 0 else l[k] <= tp
        if hit_sl:
            return 1
        if hit_tp:
            return 0
    k = min(ffb, len(c)) - 1
    return int(d * (c[k] - entry) < 0)


def exit_resolution_bar(cx, bar, info):
    """Bar at which the open order's label becomes final (first touch or forced flat), or -1 if beyond the data."""
    d, entry, sl, tp, ffb = info
    h, l = cx.h, cx.l
    for k in range(bar + 1, min(ffb, len(cx.c))):
        if ((l[k] <= sl) if d > 0 else (h[k] >= sl)) or ((h[k] >= tp) if d > 0 else (l[k] <= tp)):
            return k
    return min(ffb, len(cx.c)) - 1 if ffb <= len(cx.c) else -1


def build_sets(variant, vis, cx, from_expert, final_before_bar=None):
    """vis rows -> {module: (X, y, R, Rm)}; flat labels from the expert's issued action (round 0) or the oracle,
    positioned labels from ``exit_label``. With ``final_before_bar`` (refit sweep) a row is used only when its label
    is FINAL before that bar: hindsight support end (flat rows) or the order's resolution bar (positioned rows)."""
    acc = {m: ([], [], [], []) for m in MODULES[variant]}
    for r, bar, st, a_exp, tp_exp, sl_exp, info in vis:
        if final_before_bar is not None:
            if info is not None:
                rb = exit_resolution_bar(cx, bar, info)
                if rb < 0 or rb >= final_before_bar:
                    continue
            else:
                se = int(cx.ds.support_end[bar])
                if se < 0 or se >= final_before_bar:
                    continue
        if info is not None:
            a, tp, sl = (E.CLOSE if exit_label(cx, bar, info) else E.HOLD), 1.5, 2.0
        elif from_expert:
            a, tp, sl = a_exp, float(np.clip(tp_exp, 0.25, 3.0)), float(np.clip(sl_exp, 0.5, 3.0))
        else:
            a, tp, sl = cx.oracle.action(bar, int(st[0]), bool(st[-1] > 0.5), float(cx.ds.sigma[bar + 1]))
            if a in (E.LONG, E.SHORT) and int(st[0]) != 0:
                a = E.HOLD                           # positioned rows are labelled by the exit rule only
        for m, (y, is_open) in labels_for(variant, st, a).items():
            X, Y, R, Rm = acc[m]
            X.append(np.r_[cx.Xw[r], st])
            Y.append(y)
            R.append((tp, sl))
            Rm.append(float(is_open))
    return {m: pack(*v) for m, v in acc.items() if len(v[1])}


# ----------------------------------------------------------------------------- composed policy
class Composed:
    def __init__(self, variant, pols, arbiter, tau=None):
        self.variant, self.p, self.arb, self.tau = variant, pols, arbiter, tau

    def _probs(self, name, x):
        lg, rg = self.p[name](x[None, :])
        z = lg[0] - lg[0].max()
        return np.exp(z) / np.exp(z).sum(), (None if rg is None else rg[0])

    def openers(self, x):
        """(pL, tpL, slL, pS, tpS, slS) on a flat-equivalent input."""
        v = self.variant
        if v == "JOINT":
            P, rg = self._probs("joint", x)
            return P[E.LONG], rg[0], rg[1], P[E.SHORT], rg[0], rg[1]
        if v == "TASK":
            P, rg = self._probs("opener", x)
            return P[E.LONG], rg[0], rg[1], P[E.SHORT], rg[0], rg[1]
        if v == "DIR":
            PL, rl = self._probs("long", x)
            PS, rs = self._probs("short", x)
            return PL[1], rl[0], rl[1], PS[1], rs[0], rs[1]
        PL, rl = self._probs("open_l", x)
        PS, rs = self._probs("open_s", x)
        return PL[1], rl[0], rl[1], PS[1], rs[0], rs[1]

    def exit_now(self, x, pos):
        v = self.variant
        if v == "JOINT":
            P, _ = self._probs("joint", x)
            a = int(np.argmax(P))
            return a == E.CLOSE or (a == E.LONG and pos < 0) or (a == E.SHORT and pos > 0)
        if v == "TASK":
            return int(np.argmax(self._probs("closer", x)[0])) == EXIT
        if v == "DIR":
            return int(np.argmax(self._probs("long" if pos > 0 else "short", x)[0])) == 2
        return int(np.argmax(self._probs("close_l" if pos > 0 else "close_s", x)[0])) == EXIT

    def arbitrate(self, pL, tpL, slL, pS, tpS, slS):
        if self.arb == "prob":
            d, p = (E.LONG, pL) if pL >= pS else (E.SHORT, pS)
            return d if p > self.tau else E.HOLD
        gL = pL * tpL - (1 - pL) * slL - COST_SIGMA
        gS = pS * tpS - (1 - pS) * slS - COST_SIGMA
        if self.arb == "gain_over_risk":
            gL, gS = gL / slL, gS / slS
        d, g = (E.LONG, gL) if gL >= gS else (E.SHORT, gS)
        return d if g > 0 else E.HOLD

    def __call__(self, cx):
        def f(bar, r, st, env):
            pos = int(st[0])
            x = np.r_[cx.Xw[r], st].astype(np.float32)
            if self.variant == "JOINT":            # the joint policy acts exactly as E1-T (argmax), no arbiter
                lg, rg = self.p["joint"](x[None, :])
                return int(np.argmax(lg[0])), float(rg[0, 0]), float(rg[0, 1])
            if pos != 0 and not self.exit_now(x, pos):
                return E.HOLD, 1.5, 2.0
            xf = np.r_[cx.Xw[r], flat_state(st)].astype(np.float32)
            pL, tpL, slL, pS, tpS, slS = self.openers(xf)
            d = self.arbitrate(pL, tpL, slL, pS, tpS, slS)
            if d == E.HOLD:
                return (E.CLOSE if pos != 0 else E.HOLD), 1.5, 2.0
            if d == pos:                           # closer said exit but the arbiter re-opens the same side: exit
                return E.CLOSE, 1.5, 2.0
            return d, float(tpL if d == E.LONG else tpS), float(slL if d == E.LONG else slS)
        return f

    def open_scores(self, cx, rows):
        """pmax per flat-equivalent row (for the prob arbiter's tau grid)."""
        out = []
        for r in rows:
            xf = np.r_[cx.Xw[r], np.zeros(9, np.float32)].astype(np.float32)
            pL, *_, pS, _, _ = self.openers(xf)
            out.append(max(pL, pS))
        return np.asarray(out)


# ----------------------------------------------------------------------------- run
def run(a):
    import torch
    from app.short_horizon.expert_guided import learn as L
    from app.short_horizon.expert_guided import temporal as T
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    led = out / "TRIAL_LEDGER.jsonl"
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    t0 = time.time()
    if a.store:
        feats = json.loads(Path(a.feature_list).read_text())["features"]
        ds = DG.load_store(a.admitted, a.store, feats, scaling=a.scaling)
    else:
        ds = DG.load(a.admitted, a.csv_dir, scaling=a.scaling) if a.features == "core15" else DG.load_v4(a.admitted, a.v4_dir)
    cx = Ctx(ds)
    nf, NS = ds.windows.shape[2], len(E.STATE_NAMES)
    ARCH = dict(units=a.units, branch=a.branch, core=a.core, depth=a.depth, head=a.head)
    variants = [v for v in VARIANTS if v in a.variants.split(",")]
    params = {v: {m: int(sum(p.numel() for p in T.TemporalPolicyNet(nf, DG.WIN, nc, n_reg=nr, n_state=NS, **ARCH).parameters()))
                  for m, (nc, nr) in MODULES[v].items()} for v in variants}
    ledger(led, rung="E3M", event="PREREGISTRATION", variants=variants, arbiters=list(ARBITERS), arch=ARCH,
           dropout=a.dropout, wd=a.wd, cost_sigma=COST_SIGMA, rounds=ROUNDS, seeds=SEEDS, params=params, device=device,
           decision="variant x arbiter: highest seed-mean INNER_A Sharpe among valid; ties -> fewer models; 2024 once")
    res = {"meta": ds.meta, "arch": ARCH, "params": params, "scaling": a.scaling, "variants": variants, "arbiters": list(ARBITERS),
           "cost_sigma": COST_SIGMA, "device": device, "comparators": {}, "trials": {}}

    def fit_module(v, m, D_tr, D_in, seed):
        nc, nr = MODULES[v][m]
        f = np.maximum(np.bincount(D_tr[1], minlength=nc) / len(D_tr[1]), 1e-3)   # 1/sqrt(freq), as E1-T
        w = 1 / np.sqrt(f)
        w = w / w[D_tr[1]].mean()
        model, hist, info = L.fit(D_tr[0], D_tr[1], D_in[0], D_in[1], seed=seed, n_cls=nc,
                                  R=D_tr[2] if nr else None, Rm=D_tr[3] if nr else None,
                                  Ri=D_in[2] if nr else None, Rmi=D_in[3] if nr else None, class_weight=w,
                                  dropout=a.dropout, lr=1e-3, wd=a.wd, batch=256, max_epochs=a.max_epochs, patience=10,
                                  device=device, net=lambda: T.TemporalPolicyNet(nf, DG.WIN, nc, n_reg=nr, n_state=NS,
                                                                                  dropout=a.dropout, **ARCH))
        return T.TorchPolicy(model), info

    # expert trajectory (round 0 demonstrations) and comparators
    cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    demo_vis = {p: collect(cx, p, cx.expert, EXPERT_CFG)[1] for p in ("TRAIN", "INNER_A")}
    _, vtr0 = collect(cx, "TRAIN", cx.expert, EXPERT_CFG)
    prior = (np.bincount(np.array([v[3] for v in vtr0], int), minlength=4) / len(vtr0)).tolist()
    explore = {p: [v for v in collect(cx, p, cx.prior_random(prior, 0), STUDENT_CFG)[1] if v[6] is not None]
               for p in ("TRAIN", "INNER_A")}       # exploration positions (round 0 closers only)
    res["exploration_positions"] = {p: len(v) for p, v in explore.items()}
    for p in ("TRAIN", "INNER_A", "INNER_B"):
        res["comparators"][p] = {"expert_demonstrations": summary(cx.rollout(p, cx.expert, EXPERT_CFG)[0]),
                                 "expert_confirmed": summary(cx.rollout(p, cx.confirmed, STUDENT_CFG)[0]),
                                 "no_trade": summary(cx.rollout(p, lambda *x: (E.HOLD, 1.5, 2.0), STUDENT_CFG)[0])}
    frozen = {}
    for v in variants:
        for seed in SEEDS:
            D_tr = build_sets(v, demo_vis["TRAIN"] + explore["TRAIN"], cx, True)
            D_in = build_sets(v, demo_vis["INNER_A"] + explore["INNER_A"], cx, True)
            pols, infos = {}, {}
            comp = None
            for rnd in range(ROUNDS + 1):
                if rnd > 0:
                    _, vtr = collect(cx, "TRAIN", comp(cx))
                    _, vin = collect(cx, "INNER_A", comp(cx))
                    for m, D in build_sets(v, vtr, cx, False).items():
                        D_tr[m] = cat(D_tr[m], D) if m in D_tr else D
                    for m, D in build_sets(v, vin, cx, False).items():
                        D_in[m] = cat(D_in[m], D) if m in D_in else D
                for m in MODULES[v]:
                    if m not in D_tr or m not in D_in or len(np.unique(D_tr[m][1])) < 2:
                        if m in pols:
                            ledger(led, rung="E3M", event="MODULE_KEPT", variant=v, module=m, seed=seed, round=rnd)
                            continue
                        raise SystemExit(f"E3M_EMPTY_PARTITION:{v}:{m}:round{rnd}")
                    ledger(led, rung="E3M", event="FIT", variant=v, module=m, seed=seed, round=rnd, rows_train=len(D_tr[m][1]),
                           rows_inner=len(D_in[m][1]), label_counts=np.bincount(D_tr[m][1]).tolist())
                    pols[m], infos[m] = fit_module(v, m, D_tr[m], D_in[m], seed)
                # DAgger rounds visit states with the prob arbiter at the median-rate tau (declared; not a choice)
                comp = Composed(v, pols, "prob", tau=0.5)
                if v != "JOINT":
                    pm = comp.open_scores(cx, cx.ds.rows("INNER_A"))
                    comp.tau = float(np.quantile(pm, 1 - 0.06))
            # arbiter trials on INNER_A only (every one ledgered)
            ia = cx.ds.rows("INNER_A")
            trials = []
            cands = [("prob", t) for t in RATE_GRID] if v != "JOINT" else [("argmax", None)]
            if v != "JOINT":
                cands += [("expected_gain", None), ("gain_over_risk", None)]
            pm = Composed(v, pols, "prob").open_scores(cx, ia) if v != "JOINT" else None
            for arb, rate in cands:
                tau = float(np.quantile(pm, 1 - rate)) if rate is not None else None
                c = Composed(v, pols, arb, tau)
                m = summary(cx.rollout("INNER_A", c(cx), STUDENT_CFG)[0])
                trials.append({"arbiter": arb, "rate": rate, "tau": tau, "sharpe": m["sharpe_daily_sqrt365"], "valid": m["valid"],
                               "orders_per_week": m["orders_per_week"], "net": m["net_profit_usd"]})
                ledger(led, rung="E3M", event="ARBITER_TRIAL", variant=v, seed=seed, **trials[-1])
            ok = [t for t in trials if t["valid"] and t["sharpe"] is not None and math.isfinite(t["sharpe"])]
            best = max(ok or trials, key=lambda t: t["sharpe"] if t["sharpe"] is not None and math.isfinite(t["sharpe"]) else -1e9)
            c = Composed(v, pols, best["arbiter"], best["tau"])
            frozen[(v, seed)] = c
            ledger(led, rung="E3M", event="FREEZE", variant=v, seed=seed, arbiter=best["arbiter"], tau=best["tau"], chosen_on="INNER_A")
            rec = {"fits": {m: {k: infos[m][k] for k in ("epochs", "best_epoch", "params", "stop_reason")} for m in infos},
                   "arbiter_trials": trials, "arbiter": best, "trading": {}, "musashi_contract": {}}
            for p in ("TRAIN", "INNER_A", "INNER_B"):
                rec["trading"][p] = summary(cx.rollout(p, c(cx), STUDENT_CFG)[0])
                rec["musashi_contract"][p] = summary(cx.rollout(p, c(cx), MUSASHI_CFG)[0])
            for m, pol in pols.items():
                torch.save(pol.m.state_dict(), out / f"{v}_{m}_{seed}.pt")
            res["trials"].setdefault(v, {})[str(seed)] = rec
            print(json.dumps({"variant": v, "seed": seed, "arbiter": best["arbiter"],
                              "sharpe": {p: round(rec["trading"][p]["sharpe_daily_sqrt365"], 2) for p in rec["trading"]},
                              "opw": round(rec["trading"]["INNER_B"]["orders_per_week"], 1), "s": round(time.time() - t0, 1)}), flush=True)
    # decision on INNER_A (seed means), 2024 once
    means = {}
    for v in variants:
        sh = [res["trials"][v][str(s)]["trading"]["INNER_A"]["sharpe_daily_sqrt365"] for s in SEEDS]
        valid = all(res["trials"][v][str(s)]["trading"]["INNER_A"]["valid"] for s in SEEDS)
        means[v] = {"inner_a_sharpe_mean": float(np.mean(sh)), "all_valid": valid, "n_models": len(MODULES[v])}
    ok = [v for v in variants if means[v]["all_valid"]]
    chosen = max(ok or variants, key=lambda v: (means[v]["inner_a_sharpe_mean"], -means[v]["n_models"]))
    res["decision"] = {"means": means, "chosen": chosen, "rule": "highest seed-mean INNER_A Sharpe among all-valid; ties -> fewer models"}
    ledger(led, rung="E3M", event="DECISION", **res["decision"])
    ledger(led, rung="E3M", event="READ", periods=["OBS_2024"], purpose=f"single read: chosen variant {chosen} + comparators")
    res["comparators"]["OBS_2024"] = {"expert_demonstrations": summary(cx.rollout("OBS_2024", cx.expert, EXPERT_CFG)[0]),
                                      "expert_confirmed": summary(cx.rollout("OBS_2024", cx.confirmed, STUDENT_CFG)[0]),
                                      "no_trade": summary(cx.rollout("OBS_2024", lambda *x: (E.HOLD, 1.5, 2.0), STUDENT_CFG)[0])}
    for seed in SEEDS:
        c = frozen[(chosen, seed)]
        rec = res["trials"][chosen][str(seed)]
        rec["trading"]["OBS_2024"] = summary(cx.rollout("OBS_2024", c(cx), STUDENT_CFG)[0])
        rec["musashi_contract"]["OBS_2024"] = summary(cx.rollout("OBS_2024", c(cx), MUSASHI_CFG)[0])
    res["wall_seconds"] = time.time() - t0
    (out / "E3M.json").write_text(json.dumps(jsonable(res), indent=1))
    print(json.dumps({"decision": res["decision"], "wall": round(res["wall_seconds"], 1)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--v4-dir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", choices=["v4", "core15"], default="core15")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--units", type=int, default=2)
    ap.add_argument("--branch", type=int, default=None)
    ap.add_argument("--core", type=int, default=None)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--scaling", choices=["rolling_z", "fixed_train"], default="rolling_z")
    ap.add_argument("--store", default="", help="layer-1 pool parquet (MANIFEST.json beside it)")
    ap.add_argument("--feature-list", default="", help="JSON with {'features': [...]} selected from the store")
    ap.add_argument("--cpu", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
