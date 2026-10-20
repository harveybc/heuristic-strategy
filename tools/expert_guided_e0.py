#!/usr/bin/env python3
"""E0 imitability of the ZigZag v2 expert from Musashi's admitted-core 15 observations (DEVELOPMENT).

  run     fit the student classifiers (hindsight = anticipatory target, confirmed = reactive comparator;
          3 seeds each; owner's regularisation and stopping rule), calibrate the temperature and choose the
          signal threshold on INNER_A only, freeze, score TRAIN and INNER_B, replay every policy in the
          executable simulator, then read 2024 ONCE for the frozen set -> <out>/E0.json + trial ledger
  report  Spanish figures and tables -> docs/short_horizon/results/expert_guided/e0/

Pre-registered gate (written to the ledger before the first fit; never changed after a read):
  G1  hindsight student, INNER_B: temperature-calibrated log-loss < TRAIN-prior log-loss, in >= 2 of 3 seeds
      and on the seed mean;
  G2  hindsight student, INNER_B: for BOTH directions PR-AUC - prevalence has a weekly block-bootstrap
      lower 95 % bound > 0, in >= 2 of 3 seeds;
  learnable signal = G1 AND G2 -> continue to E1; otherwise the ladder STOPS at E0 (a negative is a result).
  Trading (executed Sharpe vs no-trade, TRAIN-prior random and the confirmed expert) is reported, not gating.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import scoring as SC  # noqa: E402

SEEDS = (20261019, 20261020, 20261021)
RATE_GRID = (0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22)
GATE = {
    "G1_probability": "hindsight student INNER_B temperature-calibrated log-loss < TRAIN-prior log-loss in >= 2/3 seeds and on the seed mean",
    "G2_ranking": "hindsight student INNER_B: both directions PR-AUC - prevalence weekly-block-bootstrap lo95 > 0 in >= 2/3 seeds",
    "learnable_signal": "G1 AND G2 -> E1; else STOP at E0",
    "trading": "reported, not gating: executed Sharpe vs no-trade, TRAIN-prior random, confirmed expert",
}
MODEL = {"kind": "MLP on the flattened 13 x 15 window", "hidden": [128, 64], "dropout": 0.1, "optimizer": "AdamW",
         "lr": 1e-3, "weight_decay": 1e-4, "batch": 256, "max_epochs": 200, "patience": 10, "loss": "unweighted CE",
         "batchnorm": False}


def ledger(path, **row):
    row = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def strip(m):
    return {k: v for k, v in m.items() if not hasattr(v, "index")}


def run(a):
    import torch
    from app.short_horizon.expert_guided import learn as L
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    led = out / "TRIAL_LEDGER.jsonl"
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    t0 = time.time()
    ds = DG.load(a.admitted, a.csv_dir)
    ledger(led, rung="E0", event="PREREGISTRATION", gate=GATE, model=MODEL, seeds=SEEDS, rate_grid=RATE_GRID,
           periods=DG.PERIODS, period_status=DG.PERIOD_STATUS, device=device, max_epochs=a.max_epochs)
    X = ds.windows.reshape(len(ds.windows), -1).astype(np.float32)
    rows = {p: ds.rows(p) for p in DG.PERIODS}
    weeks = {p: len(np.unique(SC.week_keys(ds.row_ts[r]))) for p, r in rows.items()}
    elig_bars = set(int(b) for b in ds.row_bar[ds.eligible])
    runner = SC.SignalRunner(ds.bars, ds.cal, DG.PERIODS)
    res = {"meta": ds.meta, "exclusions": ds.exclusions, "weeks": weeks, "gate": GATE, "model": MODEL, "device": device,
           "students": {}, "comparators": {}}

    def execute(sig, period, **kw):
        m, r = runner.execute(sig, period, weeks=weeks[period], **kw)
        return strip(m)

    def restrict(sig):
        return [(b, d) for b, d in sig if b in elig_bars]

    # ---------------------------------------------------------------- comparators on TRAIN / INNER (2024 later)
    hs = restrict(SC.hindsight_signals(ds.pivots))
    cs = restrict(SC.confirmed_signals(ds.pivots, ds.conf))
    ledger(led, rung="E0", event="READ", periods=["TRAIN", "INNER_A", "INNER_B"], purpose="comparators (expert, confirmed, prior-random)")
    y_tr = {"hindsight": ds.y_hind[ds.row_bar[rows["TRAIN"]]], "confirmed": ds.y_conf[ds.row_bar[rows["TRAIN"]]]}
    prior = {t: (np.bincount(y, minlength=3) / len(y)).tolist() for t, y in y_tr.items()}
    res["train_prior"] = prior

    def comparators(periods):
        for p in periods:
            c = res["comparators"].setdefault(p, {})
            c["expert_hindsight_tp_pivot"] = execute(hs, p, mode="hindsight", piv=ds.pivots)
            c["expert_hindsight_timing_causal_tp"] = execute(hs, p, mode="causal", k_tp=DG.CONFIRMED_K_TP)
            c["expert_confirmed"] = execute(cs, p, mode="causal", k_tp=DG.CONFIRMED_K_TP)
            c["no_trade"] = SC.no_trade_metrics()
            pr = []
            for s in SEEDS:
                rng = np.random.default_rng(s)
                r = rows[p]
                u = rng.random(len(r))
                pl, ps = prior["hindsight"][1], prior["hindsight"][2]
                d = np.where(u < pl, 1, np.where(u < pl + ps, -1, 0))
                sig = [(int(b), int(x)) for b, x in zip(ds.row_bar[r], d) if x != 0]
                pr.append(execute(sig, p, mode="causal", k_tp=DG.CONFIRMED_K_TP))
            c["train_prior_random"] = pr

    comparators(["TRAIN", "INNER_A", "INNER_B"])

    # ---------------------------------------------------------------- students
    frozen = {}
    for target in ("hindsight", "confirmed"):
        ylab = ds.y_hind if target == "hindsight" else ds.y_conf
        y = ylab[ds.row_bar].astype(np.int64)
        st = res["students"].setdefault(target, {})
        for seed in SEEDS:
            ledger(led, rung="E0", event="FIT", target=target, seed=seed, fit="TRAIN", monitor="INNER_A")
            model, hist, info = L.fit(X[rows["TRAIN"]], y[rows["TRAIN"]], X[rows["INNER_A"]], y[rows["INNER_A"]],
                                      seed=seed, device=device, max_epochs=a.max_epochs, hidden=tuple(MODEL["hidden"]),
                                      dropout=MODEL["dropout"], lr=MODEL["lr"], wd=MODEL["weight_decay"], batch=MODEL["batch"],
                                      patience=MODEL["patience"])
            allr = np.flatnonzero(ds.eligible)
            P_all, _ = L.predict(model, X[allr], device=device)
            P = np.zeros((len(X), 3), np.float32)
            P[allr] = P_all
            T = L.fit_temperature(np.log(np.clip(P[rows["INNER_A"]], 1e-12, 1)), y[rows["INNER_A"]])
            Pc = L.apply_temperature(P, T)
            # threshold on INNER_A only (every candidate is a ledgered trial)
            ia = rows["INNER_A"]
            pmax = np.maximum(Pc[ia, 1], Pc[ia, 2])
            cands = []
            for rate in RATE_GRID:
                tau = float(np.quantile(pmax, 1 - rate))
                sig = SC.threshold_signals(Pc[ia], tau)
                m = execute([(int(b), int(d)) for b, d in zip(ds.row_bar[ia], sig) if d], "INNER_A", mode="causal",
                            k_tp=DG.CONFIRMED_K_TP)
                cands.append({"rate": rate, "tau": tau, "sharpe": m["sharpe"], "valid": m["valid"],
                              "orders_per_week": m["orders_per_week"], "net": m["net"]})
                ledger(led, rung="E0", event="THRESHOLD_TRIAL", target=target, seed=seed, period="INNER_A", rate=rate,
                       tau=tau, sharpe=m["sharpe"], valid=m["valid"])
            ok = [c for c in cands if c["valid"] and c["sharpe"] is not None and math.isfinite(c["sharpe"])]
            choice = max(ok or cands, key=lambda c: c["sharpe"] if math.isfinite(c["sharpe"]) else -1e9)
            frozen[(target, seed)] = (Pc, choice["tau"])
            ledger(led, rung="E0", event="FREEZE", target=target, seed=seed, temperature=T, tau=choice["tau"],
                   rate=choice["rate"], chosen_on="INNER_A")
            rec = {"fit": info, "history": hist, "temperature": T, "threshold_candidates": cands, "threshold": choice,
                   "scores": {}, "trading": {}}
            ledger(led, rung="E0", event="READ", target=target, seed=seed, periods=["TRAIN", "INNER_A", "INNER_B"],
                   purpose="scores of the frozen student")
            for p in ("TRAIN", "INNER_A", "INNER_B"):
                r = rows[p]
                rec["scores"][p] = {"raw": SC.class_scores(y[r], P[r], prior[target], ds.row_ts[r], boot=(p == "INNER_B"), seed=seed),
                                    "calibrated": SC.class_scores(y[r], Pc[r], prior[target], ds.row_ts[r],
                                                                  boot=(p == "INNER_B"), seed=seed)}
                sig = SC.threshold_signals(Pc[r], choice["tau"])
                rec["scores"][p]["at_threshold"] = SC.precision_recall(y[r], sig)
                rec["trading"][p] = execute([(int(b), int(d)) for b, d in zip(ds.row_bar[r], sig) if d], p, mode="causal",
                                            k_tp=DG.CONFIRMED_K_TP)
            st[str(seed)] = rec
            np.savez_compressed(out / f"pred_{target}_{seed}.npz", P=P, Pc=Pc, row_ts=ds.row_ts, eligible=ds.eligible)
            print(json.dumps({"target": target, "seed": seed, "epochs": info["epochs"], "best": info["best_epoch"],
                              "ib_ll": rec["scores"]["INNER_B"]["calibrated"]["log_loss_minus_prior"],
                              "ib_sharpe": rec["trading"]["INNER_B"]["sharpe"], "s": round(time.time() - t0, 1)}), flush=True)

    # ---------------------------------------------------------------- gate (INNER_B only)
    g1, g2 = [], []
    for seed in SEEDS:
        sc = res["students"]["hindsight"][str(seed)]["scores"]["INNER_B"]["calibrated"]
        g1.append(sc["log_loss_minus_prior"] < 0)
        g2.append(all(sc[d]["boot_excess_lo95"] > 0 for d in ("LONG", "SHORT")))
    mean_ll = float(np.mean([res["students"]["hindsight"][str(s)]["scores"]["INNER_B"]["calibrated"]["log_loss_minus_prior"]
                             for s in SEEDS]))
    G1 = sum(g1) >= 2 and mean_ll < 0
    G2 = sum(g2) >= 2
    res["gate_result"] = {"G1": bool(G1), "G1_per_seed": g1, "G1_mean_log_loss_minus_prior": mean_ll, "G2": bool(G2),
                          "G2_per_seed": g2, "learnable_signal": bool(G1 and G2),
                          "decision": "CONTINUE_TO_E1" if (G1 and G2) else "STOP_AT_E0"}
    ledger(led, rung="E0", event="GATE", **res["gate_result"])

    # ---------------------------------------------------------------- 2024: one read of the frozen set
    ledger(led, rung="E0", event="READ", periods=["OBS_2024"], purpose="single read of the frozen E0 set (students + comparators)")
    comparators(["OBS_2024"])
    r = rows["OBS_2024"]
    for (target, seed), (Pc, tau) in frozen.items():
        ylab = ds.y_hind if target == "hindsight" else ds.y_conf
        y = ylab[ds.row_bar].astype(np.int64)
        rec = res["students"][target][str(seed)]
        rec["scores"]["OBS_2024"] = {"calibrated": SC.class_scores(y[r], Pc[r], prior[target], ds.row_ts[r], boot=True, seed=seed)}
        sig = SC.threshold_signals(Pc[r], tau)
        rec["scores"]["OBS_2024"]["at_threshold"] = SC.precision_recall(y[r], sig)
        rec["trading"]["OBS_2024"] = execute([(int(b), int(d)) for b, d in zip(ds.row_bar[r], sig) if d], "OBS_2024",
                                             mode="causal", k_tp=DG.CONFIRMED_K_TP)
    res["wall_seconds"] = time.time() - t0
    (out / "E0.json").write_text(json.dumps(jsonable(res), indent=1))
    print(json.dumps({"gate": res["gate_result"], "wall": round(res["wall_seconds"], 1)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    run(a)


if __name__ == "__main__":
    main()
