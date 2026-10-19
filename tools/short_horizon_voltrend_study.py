#!/usr/bin/env python3
"""Trend + volatility -> TP/SL ablation and the NSGA-II example for VolTrendStrategy (DEVELOPMENT).

Stages
  prep      lake EUR/USD H1 (pinned digest), blocks B = 24 / 6 h, causal trend models t0..t4 and vol
            models MEAN20 / HAR / ORACLE per block -> <run>/inputs_B{B}.npz, PREP.json
  ablation  defaults (m = 2, c_tp = 1, w_trend = 1, theta = 0) over {t0..t4} x {MEAN20, HAR, ORACLE},
            2023 decision / 2024 confirmation, leverage 10/50/100, c_tp sensitivity on 2023, and the
            degradation curves (trend blend, direction flips, vol blend) around the oracles -> cells.csv
  nsga      NSGA-II (DEAP) on 2020-2022 only, checkpointed per generation; validate the front on 2023,
            confirm the chosen point once on 2024 -> nsga_*.json, front.csv
Trend models (all causal, weekly expanding refits unless stated): t0 zero drift (coin direction), t1 AR(3)
on hourly returns + HAR vol exogenous, t2 ridge on 6 h / 1 d / 1 w / 1 m returns + calendar dummies,
t3 DLinear (one-sided MA(25) trend / seasonal decomposition of the 168-h return window, linear heads,
fit on 2020-2022, ridge alpha chosen on 2023, no refit), t4 oracle (realized block drift).
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
from app.short_horizon import blocks as BL  # noqa: E402
from app.short_horizon import execution as X  # noqa: E402
from app.short_horizon import noise as N  # noqa: E402
from app.short_horizon import predictors as PR  # noqa: E402
from app.short_horizon import strategies as S  # noqa: E402
from app.short_horizon import voltrend as VT  # noqa: E402
from tools.short_horizon_oracle import load_inputs  # noqa: E402

PERIODS = {"DECISION_2020_2022": ("2020-01-01", "2023-01-01"), "VALIDATION_2023": ("2023-01-01", "2024-01-01"),
           "CONFIRM_2024": ("2024-01-01", "2025-01-01")}
TREND = ("t0", "t1", "t2", "t3", "t4")
VOL = ("MEAN20", "HAR", "ORACLE")
LOOKBACK, MA = 168, 25
ALPHAS = (1.0, 10.0, 100.0)
C_TP_SENS = (0.5, 1.0, 1.5, 2.0)
TREND_BLEND = (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)
ACCURACY = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
VOL_BLEND = (1.0, 0.75, 0.5, 0.25, 0.0)
SEEDS = (0, 1, 2)


def utc(s):
    return pd.Timestamp(s, tz="UTC")


def per(name):
    a, b = PERIODS[name]
    return utc(a), utc(b)


# ------------------------------------------------------------------ features
def bar_features(bars, blocks, t):
    c = bars.CLOSE.to_numpy(float)
    r = np.r_[np.nan, np.diff(c)]
    idx = blocks.ref_bar[t.k.to_numpy(int)]
    f = {}
    for h in (6, 24, 120, 480):
        f[f"ret_{h}"] = np.array([c[i] - c[i - h] if i - h >= 0 else np.nan for i in idx])
    for j in (1, 2, 3):
        f[f"r_lag{j}"] = np.array([r[i - j + 1] if i - j + 1 >= 1 else np.nan for i in idx])
    F = pd.DataFrame(f, index=t.index)
    win = np.full((len(idx), LOOKBACK), np.nan)
    for n, i in enumerate(idx):
        if i - LOOKBACK + 1 >= 1:
            win[n] = r[i - LOOKBACK + 1:i + 1]
    return F, win


def dlinear_design(win):
    """One-sided moving average (kernel MA) trend part of the return window and the seasonal remainder."""
    k = np.ones(MA) / MA
    trend = np.full_like(win, np.nan)
    for n in range(win.shape[0]):
        if np.isfinite(win[n]).all():
            padded = np.r_[np.full(MA - 1, win[n, 0]), win[n]]
            trend[n] = np.convolve(padded, k, mode="valid")
    return np.hstack([trend, win - trend])


def ridge_predict(t, Xf, y, cadence, alpha, min_rows=100):
    """Weekly expanding ridge on standardized features (training rows only)."""
    out = np.full(len(t), np.nan)
    okx = np.isfinite(Xf).all(axis=1)
    for g, rows in PR.refit_groups(t, cadence):
        train = np.flatnonzero(okx[:g] & np.isfinite(y[:g]))
        if len(train) < min_rows:
            continue
        mu, sd = Xf[train].mean(0), Xf[train].std(0) + 1e-12
        Z = (Xf[train] - mu) / sd
        beta = PR._ridge(np.column_stack([np.ones(len(train)), Z]), y[train], lam=alpha)
        rr = [r for r in rows if okx[r]]
        out[rr] = np.column_stack([np.ones(len(rr)), (Xf[rr] - mu) / sd]) @ beta
    return out


def dlinear_predict(t, D, y, fit_end, alpha):
    okx = np.isfinite(D).all(axis=1)
    train = np.flatnonzero(okx & np.isfinite(y) & (t.start < fit_end).to_numpy())
    mu, sd = D[train].mean(0), D[train].std(0) + 1e-12
    beta = PR._ridge(np.column_stack([np.ones(len(train)), (D[train] - mu) / sd]), y[train], lam=alpha)
    out = np.full(len(t), np.nan)
    out[okx] = np.column_stack([np.ones(okx.sum()), (D[okx] - mu) / sd]) @ beta
    out[(t.start < fit_end).to_numpy()] = np.nan      # in-sample rows are not forecasts
    return out


def prep(args):
    out = Path(args.run)
    out.mkdir(parents=True, exist_ok=True)
    bars, cal, digest = load_inputs(args.data)
    info = {"data_sha256": digest, "blocks": {}}
    c = bars.CLOSE.to_numpy(float)
    for B in (24, 6):
        bl = BL.build_blocks(bars, cal, B)
        t = PR.block_table(bl, cal, B)
        n = len(bl)
        drift_t = np.array([c[bl.bar_hi[k] - 1] - bl.ref[k] for k in t.k])
        t["drift"] = drift_t
        y = t.drift.to_numpy(float)
        F, win = bar_features(bars, bl, t)
        har = PR.har_features(t, B)[:, 1:]
        trend = {"t0": np.zeros(len(t)), "t4": y.copy()}
        trend["t1"] = ridge_predict(t, np.column_stack([F[["r_lag1", "r_lag2", "r_lag3"]].to_numpy(float), har]), y, "weekly", 1.0)
        X2 = np.column_stack([F[["ret_6", "ret_24", "ret_120", "ret_480"]].to_numpy(float)] + PR._calendar_dummies(t, B))
        trend["t2"] = ridge_predict(t, X2, y, "weekly", 1.0)
        D = dlinear_design(win)
        fit_end, dec_a, dec_b = utc("2023-01-01"), utc("2023-01-01"), utc("2024-01-01")
        scores = {}
        for a in ALPHAS:
            pr = dlinear_predict(t, D, y, fit_end, a)
            sel = ((t.start >= dec_a) & (t.start < dec_b)).to_numpy() & np.isfinite(pr)
            scores[a] = float(np.mean(np.abs(pr[sel] - y[sel])))
        best = min(scores, key=scores.get)
        trend["t3"] = dlinear_predict(t, D, y, fit_end, best)
        hu, hd = PR.har_predict(t, B, "U", "weekly"), PR.har_predict(t, B, "D", "weekly")
        nU, nD, _ = S.naive_forecasts(bl, "MEAN20")
        U, D_ = bl.H - bl.ref, bl.ref - bl.L
        sigma = {"HAR": (hu + hd) / 2, "MEAN20": (nU[t.k] + nD[t.k]) / 2, "ORACLE": (U[t.k] + D_[t.k]) / 2}
        sigma_bar = pd.Series(sigma["MEAN20"]).rolling(20, min_periods=5).mean().shift(1).to_numpy()
        full = lambda v: np.where(np.isin(np.arange(n), t.k), np.interp(np.arange(n), t.k, v, left=np.nan, right=np.nan), np.nan)  # noqa
        def expand(v):
            o = np.full(n, np.nan)
            o[t.k.to_numpy(int)] = v
            return o
        np.savez(out / f"inputs_B{B}.npz", k=t.k.to_numpy(int), start=t.start.astype("int64").to_numpy(),
                 drift=expand(y), coin=N.ambiguous_coin(n), sigma_bar=expand(sigma_bar),
                 sigma_rule=S.causal_block_sigma(bl),
                 **{f"trend_{m}": expand(v) for m, v in trend.items()}, **{f"sigma_{m}": expand(v) for m, v in sigma.items()})
        info["blocks"][B] = {"n": int(n), "rows": int(len(t)), "dlinear_alpha_scores_2023": scores, "dlinear_alpha": best}
    (out / "PREP.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


# ------------------------------------------------------------------ context
class Ctx:
    def __init__(self, run: Path, data, B: int):
        self.bars, self.cal, self.digest = load_inputs(data)
        self.B = B
        self.blocks = BL.build_blocks(self.bars, self.cal, B)
        z = np.load(run / f"inputs_B{B}.npz")
        self.drift = z["drift"]
        self.inputs = {"trend": {m: z[f"trend_{m}"] for m in TREND}, "sigma": {m: z[f"sigma_{m}"] for m in VOL},
                       "sigma_bar": z["sigma_bar"], "coin": z["coin"], "sigma_rule": z["sigma_rule"]}
        self.starts = np.array([s.value for s in self.blocks.start])
        self.strat = VT.VolTrendStrategy(self.bars, self.blocks, self.cal, self.inputs)

    def mask(self, period):
        a, b = per(period)
        return (self.starts >= a.value) & (self.starts < b.value) & (self.blocks.status == BL.OK)

    def errors(self, trend_hat, sigma_hat, period):
        m = self.mask(period) & np.isfinite(trend_hat) & np.isfinite(sigma_hat) & np.isfinite(self.drift)
        d, s = self.drift[m], self.inputs["sigma"]["ORACLE"][m]
        s20 = self.inputs["sigma"]["MEAN20"][m]
        th = trend_hat[m]
        dirs = np.where(th == 0, self.inputs["coin"][m], np.sign(th))
        return {"drift_mae_pips": float(np.mean(np.abs(th - d)) * 1e4), "zero_drift_mae_pips": float(np.mean(np.abs(d)) * 1e4),
                "direction_accuracy": float(np.mean(dirs == np.sign(d))),
                "sigma_mae_pips": float(np.nanmean(np.abs(sigma_hat[m] - s)) * 1e4),
                "sigma_mae_x_mean20": float(np.nanmean(np.abs(sigma_hat[m] - s)) / np.nanmean(np.abs(s20 - s)))}

    def cell(self, p, period, extra=None, trend_override=None, sigma_override=None, seed=0):
        if trend_override is not None or sigma_override is not None:
            inputs = {**self.inputs, "trend": {**self.inputs["trend"]}, "sigma": {**self.inputs["sigma"]}}
            if trend_override is not None:
                inputs["trend"]["tx"] = trend_override
                p = {**p, "trend_model": "tx"}
            if sigma_override is not None:
                inputs["sigma"]["ORACLE"] = sigma_override
                p = {**p, "vol_model": "ORACLE"}
            strat = VT.VolTrendStrategy(self.bars, self.blocks, self.cal, inputs)
        else:
            strat = self.strat
        r = strat.evaluate(p, per(period), seed=seed)
        r.update(self.errors(strat.inputs["trend"][p["trend_model"]], strat.inputs["sigma"][p["vol_model"]], period))
        r.update({"period": period, "block_hours": self.B, "trend_model": p["trend_model"], "vol_model": p["vol_model"],
                  "c_tp": p["c_tp"], "m": p["m"], "w_trend": p["w_trend"], "theta": p["theta"], "tp_form": p["tp_form"],
                  "leverage": p["leverage"]})
        r.pop("objectives", None)
        if extra:
            r.update(extra)
        return r


def ablation(args):
    rows = []
    t0 = time.process_time()
    for B in (24, 6):
        ctx = Ctx(Path(args.run), args.data, B)
        base = {**VT.defaults(), "block_hours": B, "time_stop_hours": B}
        for tm in TREND:
            for vm in VOL:
                p = {**base, "trend_model": tm, "vol_model": vm}
                for period in ("VALIDATION_2023", "CONFIRM_2024"):
                    rows.append(ctx.cell(p, period, {"stage": "ablation_defaults"}))
                    if tm in ("t0", "t4") or vm == "HAR":
                        for lev in (10, 50):
                            rows.append(ctx.cell({**p, "leverage": lev}, period, {"stage": "leverage"}))
                for c in C_TP_SENS:
                    rows.append(ctx.cell({**p, "c_tp": c}, "VALIDATION_2023", {"stage": "c_tp_sensitivity"}))
                rows.append(ctx.cell({**p, "tp_form": "clipped"}, "VALIDATION_2023", {"stage": "tp_form_clipped"}))
        # degradation curves around the oracles (markers' context)
        drift, s_or, s20 = ctx.drift, ctx.inputs["sigma"]["ORACLE"], ctx.inputs["sigma"]["MEAN20"]
        for period in ("VALIDATION_2023", "CONFIRM_2024"):
            for w in TREND_BLEND:
                rows.append(ctx.cell({**base, "vol_model": "ORACLE"}, period, {"stage": "curve_trend_blend", "level": w},
                                     trend_override=w * drift))
            for w in VOL_BLEND:
                rows.append(ctx.cell({**base, "trend_model": "t4"}, period, {"stage": "curve_vol_blend", "level": w},
                                     sigma_override=s20 + w * (s_or - s20)))
            for acc in ACCURACY:
                for seed in SEEDS:
                    u = np.random.default_rng([seed, B, 2026_10_19]).uniform(size=len(drift))
                    elig = ctx.mask(period) & np.isfinite(drift)
                    sign = N.flip_to_accuracy(np.where(np.sign(drift) == 0, 1, np.sign(drift)).astype(int), elig, u, acc)
                    rows.append(ctx.cell({**base, "vol_model": "ORACLE"}, period, {"stage": "curve_direction", "level": acc},
                                         trend_override=sign * np.abs(drift), seed=seed))
    pd.DataFrame(rows).to_csv(Path(args.run) / "cells.csv", index=False)
    print(json.dumps({"rows": len(rows), "cpu_seconds": time.process_time() - t0}))


# ------------------------------------------------------------------ NSGA-II
GENES = [("c_tp", 0.0, 3.0), ("m", 0.5, 4.0), ("w_trend", 0.0, 2.0), ("theta", 0.0, 2.0), ("pos_a", 0.0, 1.0),
         ("pos_b", 0.0, 2.0), ("pos_c", 0.0, 2.0), ("pos_min", 0.05, 1.0), ("trend_idx", 0.0, 3.999), ("vol_idx", 0.0, 1.999),
         ("form_idx", 0.0, 1.999)]


def decode(ind, B):
    p = {**VT.defaults(), "block_hours": B, "time_stop_hours": B}
    for (name, lo, hi), v in zip(GENES, ind):
        v = min(max(float(v), lo), hi)
        if name == "trend_idx":
            p["trend_model"] = TREND[int(v)]          # t0..t3 (the oracle t4 is excluded from optimization)
        elif name == "vol_idx":
            p["vol_model"] = ("MEAN20", "HAR")[int(v)]
        elif name == "form_idx":
            p["tp_form"] = ("additive", "clipped")[int(v)]
        else:
            p[name] = v
    return p


def nsga(args):
    from deap import base, creator, tools
    run = Path(args.run)
    B = args.block_hours
    ctx = Ctx(run, args.data, B)
    ck = run / f"nsga_B{B}_checkpoint.json"
    if not hasattr(creator, "FitVT"):
        creator.create("FitVT", base.Fitness, weights=(1.0, -1.0))
        creator.create("IndVT", list, fitness=creator.FitVT)
    lo, hi = [g[1] for g in GENES], [g[2] for g in GENES]
    tb = base.Toolbox()
    tb.register("mate", tools.cxSimulatedBinaryBounded, low=lo, up=hi, eta=20.0)
    tb.register("mutate", tools.mutPolynomialBounded, low=lo, up=hi, eta=20.0, indpb=1.0 / len(GENES))
    tb.register("select", tools.selNSGA2)

    def evaluate(ind):
        r = ctx.strat.evaluate(decode(ind, B), per("DECISION_2020_2022"), seed=0)
        return r["objectives"]

    rng = random.Random(args.seed)
    if ck.exists():
        st = json.loads(ck.read_text())
        pop = [creator.IndVT(g) for g in st["population"]]
        for ind, f in zip(pop, st["fitness"]):
            ind.fitness.values = tuple(f)
        gen0, history = st["generation"], st["history"]
        pop = tb.select(pop, len(pop))          # assigns the crowding distance the DCD tournament needs
        rng = random.Random(args.seed + gen0 * 1000)
    else:
        pop = [creator.IndVT([rng.uniform(a, b) for a, b in zip(lo, hi)]) for _ in range(args.pop)]
        for ind in pop:
            ind.fitness.values = evaluate(ind)
        pop = tb.select(pop, len(pop))
        gen0, history = 0, []
    t0 = time.process_time()
    for gen in range(gen0 + 1, gen0 + args.generations + 1):
        offspring = [creator.IndVT(list(ind)) for ind in tools.selTournamentDCD(pop, len(pop))]
        for a, b in zip(offspring[::2], offspring[1::2]):
            if rng.random() < 0.9:
                tb.mate(a, b)
            tb.mutate(a), tb.mutate(b)
        for ind in offspring:
            ind.fitness.values = evaluate(ind)
        pop = tb.select(pop + offspring, args.pop)
        front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        history.append({"generation": gen, "front_size": len(front),
                        "best_sharpe": max(i.fitness.values[0] for i in pop), "min_dd": min(i.fitness.values[1] for i in pop)})
        ck.write_text(json.dumps({"generation": gen, "population": [list(map(float, i)) for i in pop],
                                  "fitness": [list(i.fitness.values) for i in pop], "history": history, "rng": [],
                                  "block_hours": B, "pop": args.pop, "seed": args.seed}, indent=1))
        if time.process_time() - t0 > args.budget_cpu_seconds:
            break
    front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    rows = []
    for ind in front:
        p = decode(ind, B)
        d = ctx.strat.evaluate(p, per("DECISION_2020_2022"))
        v = ctx.strat.evaluate(p, per("VALIDATION_2023"))
        rows.append({"manifest_sha256": VT.manifest_hash(p), **{k: p[k] for k in ("trend_model", "vol_model", "tp_form", "c_tp", "m",
                                                                             "w_trend", "theta", "pos_a", "pos_b", "pos_c", "pos_min")},
                     "dec_sharpe": d["sharpe"], "dec_max_dd_pct": d["max_dd"], "dec_net": d["net"], "dec_orders_per_week": d["orders_per_week"],
                     "dec_valid": d["valid"], "val_sharpe": v["sharpe"], "val_max_dd_pct": v["max_dd"], "val_net": v["net"],
                     "val_orders_per_week": v["orders_per_week"], "val_valid": v["valid"]})
    fr = pd.DataFrame(rows).sort_values("dec_sharpe", ascending=False)
    fr.to_csv(run / f"front_B{B}.csv", index=False)
    ok = fr[fr.dec_valid & fr.val_valid]
    chosen = (ok if not ok.empty else fr).sort_values("val_sharpe", ascending=False).iloc[0]
    p = {**VT.defaults(), "block_hours": B, "time_stop_hours": B,
         **{k: (chosen[k] if k in ("trend_model", "vol_model", "tp_form") else float(chosen[k]))
            for k in ("trend_model", "vol_model", "tp_form", "c_tp", "m", "w_trend", "theta", "pos_a", "pos_b", "pos_c", "pos_min")}}
    assert VT.manifest_hash(p) == chosen.manifest_sha256
    conf = ctx.strat.evaluate(p, per("CONFIRM_2024"))
    conf.pop("objectives", None)
    (run / f"nsga_B{B}_result.json").write_text(json.dumps(
        {"block_hours": B, "generations_done": history[-1]["generation"] if history else 0, "population": args.pop,
         "history": history, "front_points": len(fr), "front_valid_both": int(len(ok)),
         "chosen": {"parameters": p, "manifest_sha256": VT.manifest_hash(p), "decision": chosen[[c for c in fr.columns if c.startswith("dec_")]].to_dict(),
                    "validation_2023": chosen[[c for c in fr.columns if c.startswith("val_")]].to_dict(), "confirm_2024": conf},
         "note": "optimized on 2020-2022 only; 2023 used to choose the point; 2024 evaluated once"}, indent=1, default=str) + "\n")
    print(json.dumps({"generations": history[-1]["generation"] if history else 0, "front": len(fr), "chosen_val_sharpe": float(chosen.val_sharpe),
                      "confirm_2024_sharpe": conf["sharpe"], "cpu_seconds": time.process_time() - t0}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "ablation", "nsga"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--block-hours", type=int, default=24)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--budget-cpu-seconds", type=float, default=80.0)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    {"prep": prep, "ablation": ablation, "nsga": nsga}[a.stage](a)


if __name__ == "__main__":
    main()
