#!/usr/bin/env python3
"""Phase D: risk engine over the frozen E3-M FULL specialists (DEVELOPMENT; plan item 26/27, 2026-10-21).

The data phase confirmed ACTIVITY information (will an order touch a +/-2 sigma barrier before its timeout?) and no
DIRECTION information. This tool therefore does not search for direction; it asks whether risk rules driven by the
activity forecast and by volatility targeting improve the frozen policy's risk-adjusted result. DEAP is NOT used: the
plan reserves optimisation for a module with skill; a small grid is fixed here before any read.

  activity  HistGradientBoosting (depth 3, min leaf 200, 300 iterations, lr 0.05) on the data-phase CONFIRMED activity
            features (each at its confirmed lag) fitted on TRAIN 2020-2022 -> P(hit) for every grid bar (2020-2024);
            log-loss vs the TRAIN prior reported on INNER_A / INNER_B / 2024.
  evaluate  frozen FULL policies (3 seeds) wrapped by each rule of the grid (4 x 3 x 3 = 36):
              gate   theta in {off, 0.25, 0.30, 0.35}: no NEW entry when P(hit) < theta (an open order is kept; a
                     reversal into a gated entry becomes a plain CLOSE)
              size   {fixed 1.0, vol-target k=0.5, k=1.0}: percentual_order_size = clip(k * median TRAIN sigma / sigma_t, 0.1, 1)
              levels (tp x, sl x) in {(1, 1), (1.25, 0.75), (0.75, 1.25)} applied to the policy's own multiples
            choice: the highest seed-mean INNER_A Sharpe among rules whose three seeds are frequency-valid on INNER_A;
            the reference rule (off, fixed, (1, 1)) is the unmodified policy. INNER_B reported for all; 2024 read once
            for the chosen rule and the reference. Every order keeps effective SL/TP and integer units.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from tools.expert_guided_e0 import jsonable, ledger  # noqa: E402
from tools.expert_guided_e1t import STUDENT_CFG, Ctx, summary  # noqa: E402

GATES = (None, 0.25, 0.30, 0.35)
SIZES = (("fixed", None), ("voltarget", 0.5), ("voltarget", 1.0))
LEVELS = ((1.0, 1.0), (1.25, 0.75), (0.75, 1.25))
PERIODS = ("TRAIN", "INNER_A", "INNER_B")


def activity(a):
    from sklearn.ensemble import HistGradientBoostingClassifier
    inp = pd.read_parquet(a.screen_input)
    dec = json.loads(Path(a.decision).read_text())
    feats = sorted({(d["feature"], int(d["lag"])) for d in dec["activity_confirmed"]})
    ts = pd.DatetimeIndex(pd.to_datetime(inp["timestamp"])).tz_convert("UTC")
    X = pd.DataFrame({f"{f}@{L}": inp[f].shift(L).to_numpy(float) for f, L in feats}, index=ts)
    y = inp["label_hit"].to_numpy(float)
    per = inp["period"].to_numpy()
    tr = (per == "TRAIN") & np.isfinite(y)
    m = HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=200, max_iter=300, learning_rate=0.05, random_state=0)
    m.fit(X[tr], y[tr].astype(int))
    p = m.predict_proba(X)[:, 1]
    prior = float(y[tr].mean())

    def ll(mask):
        yy, pp = y[mask], np.clip(p[mask], 1e-6, 1 - 1e-6)
        return float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))), float(-np.mean(yy * np.log(prior) + (1 - yy) * np.log(1 - prior)))
    rep = {"features": [f"{f}@{L}" for f, L in feats], "train_prior": prior}
    years = pd.Series(ts.year, index=ts)
    for name, mask in (("TRAIN", tr), ("INNER_A", (per == "INNER_A") & np.isfinite(y)),
                       ("2023H2", (ts >= "2023-07-01") & (ts < "2024-01-01") & np.isfinite(y)),
                       ("2024", (years.to_numpy() == 2024) & np.isfinite(y))):
        if mask.sum():
            a_, b_ = ll(mask)
            rep[name] = {"rows": int(mask.sum()), "log_loss": a_, "prior_log_loss": b_, "minus_prior": a_ - b_}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": ts, "p_hit": p}).to_parquet(out / "P_HIT.parquet", index=False)
    (out / "ACTIVITY.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


def risk_wrap(policy_fn, p_hit_bar, sigma, sig_med, gate, size, levels):
    tpx, slx = levels

    def f(bar, r, st, env):
        a, tp, sl = policy_fn(bar, r, st, env)
        pos = int(st[0])
        new_entry = a in (E.LONG, E.SHORT) and (pos == 0 or (a == E.LONG and pos < 0) or (a == E.SHORT and pos > 0))
        if new_entry and gate is not None and not (p_hit_bar[bar] >= gate):
            return (E.CLOSE if pos != 0 else E.HOLD), 1.5, 2.0
        if new_entry:
            kind, k = size
            s = sigma[bar + 1] if bar + 1 < len(sigma) else np.nan
            pos_size = 1.0 if kind == "fixed" or not (np.isfinite(s) and s > 0) else float(np.clip(k * sig_med / s, 0.1, 1.0))
            env.cfg = dataclasses.replace(env.cfg, percentual_order_size=pos_size)     # EnvConfig is frozen
            return a, tp * tpx, sl * slx
        return a, tp, sl
    return f


def evaluate(a):
    import torch
    from app.short_horizon.expert_guided import temporal as T
    from tools.expert_guided_e3m import MODULES, Composed
    out = Path(a.out)
    led = out / "TRIAL_LEDGER.jsonl"
    t0 = time.time()
    ds = DG.load(a.admitted, a.csv_dir)
    cx = Ctx(ds)
    e3 = json.loads((Path(a.e3m_run) / "E3M.json").read_text())
    arch = e3["arch"]
    nf, NS = ds.windows.shape[2], len(E.STATE_NAMES)
    ph = pd.read_parquet(out / "P_HIT.parquet")
    pts = pd.DatetimeIndex(pd.to_datetime(ph["timestamp"])).tz_convert("UTC")
    p_bar = pd.Series(ph["p_hit"].to_numpy(float), index=pts).reindex(ds.bars.index).to_numpy()
    sigma = np.asarray(ds.sigma, float)
    tr0, tr1 = (DG._utc_ns(x) for x in DG.PERIODS["TRAIN"])
    v = ds.bars.index.as_unit("ns").asi8
    sig_med = float(np.nanmedian(sigma[(v >= tr0) & (v < tr1)]))
    pols = {}
    for seed, rec in e3["trials"]["FULL"].items():
        mods = {}
        for mname, (nc, nr) in MODULES["FULL"].items():
            net = T.TemporalPolicyNet(nf, DG.WIN, nc, n_reg=nr, n_state=NS, **arch)
            net.load_state_dict(torch.load(Path(a.e3m_run) / f"FULL_{mname}_{seed}.pt"))
            mods[mname] = T.TorchPolicy(net)
        pols[seed] = Composed("FULL", mods, rec["arbiter"]["arbiter"], rec["arbiter"].get("tau"))
    grid = list(itertools.product(GATES, SIZES, LEVELS))
    ledger(led, rung="D", event="PREREGISTRATION", grid=[str(g) for g in grid], sig_med=sig_med,
           choice="highest seed-mean INNER_A Sharpe among rules with all seeds frequency-valid on INNER_A")
    res = {"sig_med": sig_med, "rules": {}}

    def run(rule, period):
        g, s, lv = rule
        out_s = []
        for seed, comp in pols.items():
            cfg = E.EnvConfig(level_policy="clip")
            f = risk_wrap(comp(cx), p_bar, sigma, sig_med, g, s, lv)
            out_s.append(summary(cx.rollout(period, f, cfg)[0]))
        return out_s
    for rule in grid:
        key = f"gate={rule[0]}|size={rule[1][0]}{'' if rule[1][1] is None else rule[1][1]}|levels={rule[2]}"
        res["rules"][key] = {p: run(rule, p) for p in (("INNER_A", "INNER_B") if a.fast else PERIODS)}
        r = res["rules"][key]
        print(json.dumps({"rule": key, "inner_a": round(float(np.mean([m["sharpe_daily_sqrt365"] for m in r["INNER_A"]])), 3),
                          "inner_b": round(float(np.mean([m["sharpe_daily_sqrt365"] for m in r["INNER_B"]])), 3), "s": round(time.time() - t0)}), flush=True)
    ok = {k: v for k, v in res["rules"].items() if all(m["valid"] for m in v["INNER_A"])}
    chosen = max(ok or res["rules"], key=lambda k: np.mean([m["sharpe_daily_sqrt365"] for m in res["rules"][k]["INNER_A"]]))
    ref = "gate=None|size=fixed|levels=(1.0, 1.0)"
    res["chosen"], res["reference"] = chosen, ref
    ledger(led, rung="D", event="READ", periods=["OBS_2024"], purpose=f"single read: chosen {chosen} and reference")
    for k in {chosen, ref}:
        g = next(r for r in grid if f"gate={r[0]}|size={r[1][0]}{'' if r[1][1] is None else r[1][1]}|levels={r[2]}" == k)
        res["rules"][k]["OBS_2024"] = run(g, "OBS_2024")
    res["wall_seconds"] = time.time() - t0
    (out / "RISK_D.json").write_text(json.dumps(jsonable(res), indent=1))
    L = ["# Phase D: risk engine over the frozen FULL specialists (DEVELOPMENT)", "",
         f"Chosen on VAL A: **{chosen}**; reference = unmodified policy. Mean of 3 seeds.", "",
         "| Rule | Sharpe TRAIN | Sharpe VAL A | Sharpe VAL B | Sharpe 2024 | DD % VAL B | Orders/wk VAL B | Valid VAL A |",
         "|---|---|---|---|---|---|---|---|"]
    order = sorted(res["rules"], key=lambda k: -np.mean([m["sharpe_daily_sqrt365"] for m in res["rules"][k]["INNER_A"]]))
    for k in order:
        r = res["rules"][k]
        mean = lambda p, f: (f"{np.mean([m[f] for m in r[p]]):.2f}" if p in r else "—")  # noqa: E731
        L.append(f"| {k}{' **(chosen)**' if k == chosen else ''}{' *(reference)*' if k == ref else ''} | {mean('TRAIN', 'sharpe_daily_sqrt365')} | "
                 f"{mean('INNER_A', 'sharpe_daily_sqrt365')} | {mean('INNER_B', 'sharpe_daily_sqrt365')} | {mean('OBS_2024', 'sharpe_daily_sqrt365')} | "
                 f"{mean('INNER_B', 'max_drawdown_pct')} | {mean('INNER_B', 'orders_per_week')} | {all(m['valid'] for m in r['INNER_A'])} |")
    (out / "RISK_D.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"chosen": chosen, "wall": round(res["wall_seconds"])}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="verb", required=True)
    p1 = sp.add_parser("activity")
    p1.add_argument("--screen-input", required=True)
    p1.add_argument("--decision", required=True)
    p1.add_argument("--out", required=True)
    p2 = sp.add_parser("evaluate")
    p2.add_argument("--admitted", required=True)
    p2.add_argument("--csv-dir", required=True)
    p2.add_argument("--e3m-run", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--fast", action="store_true", help="plumbing: skip TRAIN")
    a = ap.parse_args()
    {"activity": activity, "evaluate": evaluate}[a.verb](a)


if __name__ == "__main__":
    main()
