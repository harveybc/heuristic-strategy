#!/usr/bin/env python3
"""Refit-cadence x super-window sweep for the expert-guided FULL specialists (owner order 2026-10-21). DEVELOPMENT.

The best out-of-sample policy of the lane so far is E3-M FULL (open-L / open-S / close-L / close-S, head 16, static
fit on 2020-2022). This sweep measures what periodic REFITTING adds, as we did for the predictors, with two declared
parameters: the cadence (monthly / weekly) and the SUPER-WINDOW (how much trailing history the refit sees; not the
model's 13-slot input window). Prequential protocol on 2023 (VAL A = S1 for choices, VAL B = S2 reported), 2024
read ONCE for the chosen arm; TEST 2025 closed.

Arms (fixed before any read): cadence in {monthly, weekly} x super_window in {1y, 2y, all} x init in {warm, scratch}
= 12, plus the static reference (= E3-M FULL, no refit). Cheap arms run first (monthly), then weekly; resumable.
At each refit date d (first bar of the month / ISO week): rows = eligible rows of [d - super_window, d) whose label is
FINAL before d (hindsight support end or the order's resolution bar); BC + 1 DAgger round (declared budget: the
static model used 3; the refit starts from the previous policy under 'warm'); early stopping on the mean of the
window's TRAIN loss and its last-10 %-of-rows holdout; arbiter 'prob' with tau chosen on that holdout (never on the
scored period). The refitted policy then acts on [d, next d) with the lane's full contract (SL/TP, integer units,
>= 3 orders/week, <= 4/day, forced flat); the scored segments are concatenated into one prequential equity curve.
3 seeds. Choice: highest seed-mean VAL A Sharpe among arms valid on VAL A; VAL B reported for all; 2024 once.
"""
from __future__ import annotations

import argparse
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
from tools.expert_guided_e0 import RATE_GRID, jsonable, ledger  # noqa: E402
from tools.expert_guided_e1t import EXPERT_CFG, STUDENT_CFG, Ctx, cat, summary  # noqa: E402
from tools.expert_guided_e3m import MODULES, Composed, build_sets, collect  # noqa: E402

SEEDS = (20261019, 20261020, 20261021)
CADENCES = {"monthly": "MS", "weekly": "W-MON"}
SUPER = {"1y": pd.DateOffset(years=1), "2y": pd.DateOffset(years=2), "all": None}
INITS = ("warm", "scratch")
ARMS = [f"{c}|{w}|{i}" for c in ("monthly", "weekly") for w in SUPER for i in INITS]
DAGGER_ROUNDS = 1
HOLDOUT_SHARE = 0.10


def _utc(x):
    return pd.Timestamp(x).tz_localize("UTC") if pd.Timestamp(x).tzinfo is None else pd.Timestamp(x).tz_convert("UTC")


def refit_dates(start, end, cadence):
    d = pd.date_range(_utc(start), _utc(end), freq=CADENCES[cadence], tz="UTC")
    d = d[(d >= _utc(start)) & (d < _utc(end))]
    if len(d) == 0 or d[0] > _utc(start):
        d = d.insert(0, _utc(start))
    return list(d) + [_utc(end)]


def fit_full(cx, ds, nf, NS, arch, a, seed, win_a, win_b, init_models, device, T, L):
    """One refit of the four FULL modules on [win_a, win_b): BC from the expert + exploration positions, then
    DAGGER_ROUNDS with the composed policy; labels only when final before win_b. Returns (policies, tau, info)."""
    import torch
    cap = int(np.searchsorted(cx.v, _utc(win_b).value))
    rng = np.random.default_rng(seed)
    vis_e = collect(cx, (win_a, win_b), cx.expert, EXPERT_CFG)[1]
    prior = (np.bincount(np.array([v[3] for v in vis_e], int), minlength=4) / max(len(vis_e), 1)).tolist()
    vis_x = [v for v in collect(cx, (win_a, win_b), cx.prior_random(prior, int(rng.integers(1 << 30))), STUDENT_CFG)[1] if v[6] is not None]
    D = build_sets("FULL", vis_e + vis_x, cx, True, final_before_bar=cap)
    pols, infos = dict(init_models or {}), {}

    def split(Dm):
        n = len(Dm[1])
        k = max(int(n * (1 - HOLDOUT_SHARE)), 1)
        return tuple(x[:k] for x in Dm), tuple(x[k:] for x in Dm)

    def fit_mod(m, Dm, init_state):
        nc, nr = MODULES["FULL"][m]
        tr, ho = split(Dm)
        if len(np.unique(tr[1])) < 2 or len(np.unique(ho[1])) < 2:
            return None, {"skipped": "SINGLE_CLASS"}
        f = np.maximum(np.bincount(tr[1], minlength=nc) / len(tr[1]), 1e-3)
        w = 1 / np.sqrt(f)
        w = w / w[tr[1]].mean()
        model, hist, info = L.fit(tr[0], tr[1], ho[0], ho[1], seed=seed, n_cls=nc, R=tr[2] if nr else None, Rm=tr[3] if nr else None,
                                  Ri=ho[2] if nr else None, Rmi=ho[3] if nr else None, class_weight=w, dropout=a.dropout, lr=a.lr,
                                  wd=a.wd, batch=256, max_epochs=a.max_epochs, patience=10, device=device, init_state=init_state,
                                  net=lambda: T.TemporalPolicyNet(nf, DG.WIN, nc, n_reg=nr, n_state=NS, dropout=a.dropout, **arch))
        return T.TorchPolicy(model), {k: info[k] for k in ("epochs", "best_epoch", "params", "stop_reason")}

    comp = None
    for rnd in range(DAGGER_ROUNDS + 1):
        if rnd > 0:
            vis = collect(cx, (win_a, win_b), comp(cx))[1]
            for m, Dm in build_sets("FULL", vis, cx, False, final_before_bar=cap).items():
                D[m] = cat(D[m], Dm) if m in D else Dm
        for m in MODULES["FULL"]:
            if m not in D:
                continue
            init = pols[m].m.state_dict() if (a_init_warm(a) and m in pols) else None
            pol, info = fit_mod(m, D[m], init)
            if pol is not None:
                pols[m], infos[(m, rnd)] = pol, info
        if any(m not in pols for m in MODULES["FULL"]):
            return None, None, {"skipped": "MODULE_MISSING", "have": sorted(pols)}
        comp = Composed("FULL", pols, "prob", tau=0.5)
    # tau on the holdout rows of the window (last 10 % of the window's bars), via the prob arbiter rate grid
    ho_a = cx.v[int(cap - max(int((cap - np.searchsorted(cx.v, _utc(win_a).value)) * HOLDOUT_SHARE), 24))]
    ho_rows = np.flatnonzero((cx.ds.row_ts >= ho_a) & (cx.ds.row_ts < _utc(win_b).value) & cx.obs_ok)
    pm = comp.open_scores(cx, ho_rows) if len(ho_rows) else np.array([0.5])
    best, best_sh = 0.5, -np.inf
    for rate in RATE_GRID:
        tau = float(np.quantile(pm, 1 - rate))
        m_ = summary(cx.rollout((pd.Timestamp(ho_a, unit="ns", tz="UTC"), _utc(win_b)), Composed("FULL", pols, "prob", tau)(cx), STUDENT_CFG)[0])
        sh = m_["sharpe_daily_sqrt365"]
        if sh is not None and math.isfinite(sh) and sh > best_sh:
            best, best_sh = tau, sh
    return pols, best, {"fits": {f"{m}_r{r}": v for (m, r), v in infos.items()}, "rows": {m: int(len(D[m][1])) for m in D}, "tau": best}


def a_init_warm(a):
    return a.init == "warm"


def rollout_range(cx, pol, a, b):
    return summary(cx.rollout((a, b), pol, STUDENT_CFG)[0])


def concat_metrics(segs: list[dict]) -> dict:
    """Prequential aggregate of consecutive segments: sum of trades / net, max of drawdowns, Sharpe from the pooled
    daily returns when the segment returns are available; else trade-weighted mean of segment Sharpes (declared)."""
    if not segs:
        return {"trades": 0, "net_profit_usd": 0.0, "max_drawdown_pct": 0.0, "orders_per_week": 0.0, "max_entries_per_day": 0,
                "sharpe_daily_sqrt365": 0.0, "valid": False, "empty": True}
    out = {"trades": int(sum(s["trades"] for s in segs)), "net_profit_usd": float(sum(s["net_profit_usd"] for s in segs)),
           "max_drawdown_pct": float(max(s["max_drawdown_pct"] for s in segs)),
           "orders_per_week": float(np.mean([s["orders_per_week"] for s in segs])),
           "max_entries_per_day": int(max(s.get("max_entries_per_day", 0) for s in segs))}
    w = np.array([max(s["trades"], 1) for s in segs], float)
    sh = np.array([s["sharpe_daily_sqrt365"] if s["sharpe_daily_sqrt365"] is not None else 0.0 for s in segs])
    out["sharpe_daily_sqrt365"] = float((w * sh).sum() / w.sum())
    out["valid"] = bool(out["orders_per_week"] >= 3.0 and out["max_entries_per_day"] <= 4)
    return out


def run(a):
    import torch
    from app.short_horizon.expert_guided import learn as L
    from app.short_horizon.expert_guided import temporal as T
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    led = out / "TRIAL_LEDGER.jsonl"
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    t0 = time.time()
    ds = DG.load(a.admitted, a.csv_dir)
    cx = Ctx(ds)
    cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    e3 = json.loads((Path(a.e3m_run) / "E3M.json").read_text())
    arch = e3["arch"]
    nf, NS = ds.windows.shape[2], len(E.STATE_NAMES)
    static = {}
    for seed in SEEDS:
        mods = {}
        for mname, (nc, nr) in MODULES["FULL"].items():
            net = T.TemporalPolicyNet(nf, DG.WIN, nc, n_reg=nr, n_state=NS, **arch)
            net.load_state_dict(torch.load(Path(a.e3m_run) / f"FULL_{mname}_{seed}.pt"))
            mods[mname] = T.TorchPolicy(net)
        static[seed] = (mods, e3["trials"]["FULL"][str(seed)]["arbiter"].get("tau"))
    arms = [x for x in ARMS if (not a.arms or x in a.arms.split(","))]
    ledger(led, rung="REFIT", event="PREREGISTRATION", arms=arms, dagger_rounds=DAGGER_ROUNDS, holdout_share=HOLDOUT_SHARE,
           seeds=SEEDS, arch=arch, device=device, choice="highest seed-mean VAL A Sharpe among arms valid on VAL A; VAL B reported; 2024 once")
    res_p = out / "REFIT.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else {"arms": {}, "static_reference": {}}
    if not res["static_reference"]:
        for seed, (mods, tau) in static.items():
            pol = Composed("FULL", mods, "prob", tau)(cx)
            res["static_reference"][str(seed)] = {p: rollout_range(cx, pol, *DG.PERIODS[p]) for p in ("INNER_A", "INNER_B")}
        res_p.write_text(json.dumps(jsonable(res), indent=1))
    seeds = SEEDS[:1] if a.smoke else SEEDS
    for arm in arms:
        if arm in res["arms"] and all(str(s) in res["arms"][arm] for s in seeds):
            continue
        cadence, sw, init = arm.split("|")
        a.init = init
        res["arms"].setdefault(arm, {})
        for seed in seeds:
            if str(seed) in res["arms"][arm]:
                continue
            ledger(led, rung="REFIT", event="ARM_SEED_START", arm=arm, seed=seed)
            dates = refit_dates("2023-01-01", "2024-01-01", cadence)
            if a.smoke:
                dates = dates[:3]
            segs, refits, prev = [], [], (static[seed][0] if init == "warm" else None)
            for d0, d1 in zip(dates, dates[1:]):
                win_a = _utc("2020-01-01") if SUPER[sw] is None else max(_utc("2020-01-01"), d0 - SUPER[sw])
                pols, tau, info = fit_full(cx, ds, nf, NS, arch, a, seed, win_a, d0, prev if init == "warm" else None, device, T, L)
                if pols is None:
                    refits.append({"refit_at": str(d0), **info})
                    m_ = rollout_range(cx, lambda *x: (E.HOLD, 1.5, 2.0), d0, d1)
                else:
                    prev = pols
                    refits.append({"refit_at": str(d0), "window": [str(win_a), str(d0)], **info})
                    m_ = rollout_range(cx, Composed("FULL", pols, "prob", tau)(cx), d0, d1)
                m_["segment"] = [str(d0), str(d1)]
                segs.append(m_)
                print(json.dumps({"arm": arm, "seed": seed, "refit": str(d0.date()), "seg_sharpe": round(m_["sharpe_daily_sqrt365"] or 0, 2),
                                  "trades": m_["trades"], "s": round(time.time() - t0)}), flush=True)
            ia = [s for s in segs if _utc(s["segment"][0]) < _utc("2023-07-01")]
            ib = [s for s in segs if _utc(s["segment"][0]) >= _utc("2023-07-01")]
            res["arms"][arm][str(seed)] = {"INNER_A": concat_metrics(ia), "INNER_B": concat_metrics(ib), "segments": segs, "refits": refits}
            res_p.write_text(json.dumps(jsonable(res), indent=1))
            ledger(led, rung="REFIT", event="ARM_SEED_DONE", arm=arm, seed=seed, inner_a=res["arms"][arm][str(seed)]["INNER_A"].get("sharpe_daily_sqrt365"))
    # decision on VAL A (static reference included as a candidate)
    cand = {"static": {str(s): res["static_reference"][str(s)] for s in SEEDS}, **res["arms"]}
    means = {k: {"val_a": float(np.mean([v[str(s)]["INNER_A"]["sharpe_daily_sqrt365"] for s in seeds])),
                 "val_b": float(np.mean([v[str(s)]["INNER_B"]["sharpe_daily_sqrt365"] for s in seeds])),
                 "valid_a": all(v[str(s)]["INNER_A"].get("valid") for s in seeds)} for k, v in cand.items() if all(str(s) in v for s in seeds)}
    ok = [k for k, m in means.items() if m["valid_a"]]
    chosen = max(ok or means, key=lambda k: means[k]["val_a"])
    res["decision"] = {"means": means, "chosen": chosen}
    if a.read_2024 and "OBS_2024" not in res.get("read_2024", {}):
        ledger(led, rung="REFIT", event="READ", periods=["OBS_2024"], purpose=f"single read of the chosen arm {chosen} (prequential through 2024) and static")
        r24 = {}
        for seed in SEEDS:
            pol = Composed("FULL", static[seed][0], "prob", static[seed][1])(cx)
            r24.setdefault("static", {})[str(seed)] = rollout_range(cx, pol, *DG.PERIODS["OBS_2024"])
            if chosen != "static":
                cadence, sw, init = chosen.split("|")
                a.init = init
                dates = refit_dates("2024-01-01", "2025-01-01", cadence)
                segs, prev = [], static[seed][0]
                for d0, d1 in zip(dates, dates[1:]):
                    win_a = _utc("2020-01-01") if SUPER[sw] is None else max(_utc("2020-01-01"), d0 - SUPER[sw])
                    pols, tau, info = fit_full(cx, ds, nf, NS, arch, a, seed, win_a, d0, prev if init == "warm" else None, device, T, L)
                    if pols is not None:
                        prev = pols
                    m_ = rollout_range(cx, Composed("FULL", pols, "prob", tau)(cx) if pols else (lambda *x: (E.HOLD, 1.5, 2.0)), d0, d1)
                    segs.append(m_)
                r24.setdefault(chosen, {})[str(seed)] = concat_metrics(segs)
        res["read_2024"] = {"OBS_2024": r24}
    res["wall_seconds"] = time.time() - t0
    res_p.write_text(json.dumps(jsonable(res), indent=1))
    L_ = ["# Refit cadence x super-window sweep (FULL specialists), DEVELOPMENT", "",
          "Prequential 2023 (VAL A = S1 choice, VAL B = S2 report); mean of 3 seeds; Sharpe = trade-weighted mean of segment Sharpes.", "",
          "| arm | Sharpe VAL A | Sharpe VAL B | valid VAL A |", "|---|---|---|---|"]
    for k, m in sorted(means.items(), key=lambda kv: -kv[1]["val_a"]):
        L_.append(f"| {k}{' **(chosen)**' if k == chosen else ''} | {m['val_a']:.2f} | {m['val_b']:.2f} | {m['valid_a']} |")
    if res.get("read_2024"):
        L_ += ["", "## 2024 (single read)", ""]
        for k, v in res["read_2024"]["OBS_2024"].items():
            L_.append(f"- {k}: Sharpe {np.mean([x['sharpe_daily_sqrt365'] for x in v.values()]):.2f}, net {np.mean([x['net_profit_usd'] for x in v.values()]):.0f} USD, "
                      f"DD {np.mean([x['max_drawdown_pct'] for x in v.values()]):.1f} %, orders/wk {np.mean([x['orders_per_week'] for x in v.values()]):.1f}")
    (out / "REFIT.md").write_text("\n".join(L_) + "\n")
    print(json.dumps({"chosen": chosen, "means": means, "wall": round(res["wall_seconds"])}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--e3m-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="", help="comma list; default all 12")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--read-2024", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="plumbing: 1 seed, 2 refit dates, few epochs")
    a = ap.parse_args()
    a.init = "warm"
    run(a)


if __name__ == "__main__":
    main()
