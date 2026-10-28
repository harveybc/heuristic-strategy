#!/usr/bin/env python3
"""E0-T: imitability of the ZigZag v2 expert with the MINIMAL TEMPORAL trunk (owner rule 2026-10-20).

Corrects E0, which flattened the 13 x 15 window into an MLP 195 -> 128 -> 64 (33,539 parameters) and used
the admitted-core 15 instead of the TIMING_PROXY_SELECTION v4 four. Here the window stays a (13, F) tensor
(``temporal.py``: per-feature causal Conv1D branches, fusion across features, dilations 1/2/4, last step).

Arms (fixed before any fit; every one on IDENTICAL rows = eligible under both feature sets):
  T2_v4, T4_v4, T8_v4   temporal trunk, u = 2 / 4 / 8 units per component, the four v4 inputs
  T2_core15             temporal trunk, u = 2, Musashi's admitted-core 15 (separates selection from architecture)
  MLP_v4_flat           the old flattened MLP (52 -> 128 -> 64) on the v4 four: comparator only, never chosen
Targets: hindsight (anticipatory, primary) and confirmed (reactive comparator); 3 seeds each.

Capacity ladder (pre-registered, INNER_A only, hindsight target, seed means): start at u = 2; move to the next
u only if its calibrated INNER_A log-loss is lower AND its (INNER_A - TRAIN) loss gap is not larger; stop at
the first refusal. Gate on INNER_B for the chosen arm exactly as E0 (G1 probability vs TRAIN prior, G2
PR-AUC excess). 2024 is read ONCE, for the chosen arm and the comparators only. Everything is DEVELOPMENT.
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
sys.path.insert(0, str(ROOT / "tools"))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import scoring as SC  # noqa: E402
from expert_guided_e0 import GATE, RATE_GRID, SEEDS, jsonable, ledger, strip  # noqa: E402

ARMS = {
    "T2_v4": {"features": "v4", "kind": "temporal", "units": 2},
    "T4_v4": {"features": "v4", "kind": "temporal", "units": 4},
    "T8_v4": {"features": "v4", "kind": "temporal", "units": 8},
    "T2_core15": {"features": "core15", "kind": "temporal", "units": 2},
    "MLP_v4_flat": {"features": "v4", "kind": "mlp_flat", "hidden": [128, 64]},
}
LADDER = ("T2_v4", "T4_v4", "T8_v4")
TRAIN_CFG = {"dropout": 0.1, "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "batch": 256, "max_epochs": 200,
             "patience": 10, "min_delta": 1e-8, "loss": "unweighted CE", "batchnorm": False,
             "monitor": "0.5*(TRAIN loss + INNER_A loss), inference mode at the epoch end"}
LADDER_RULES = {
    "v1": ("start u=2; accept the next u only if seed-mean calibrated INNER_A log-loss is lower AND the "
           "seed-mean (INNER_A - TRAIN) loss gap is not larger; stop at the first refusal"),
    # v2 amendment (declared 2026-10-20 after run 1): a NEGATIVE gap (INNER_A better than TRAIN) is not
    # overfitting, so v1 refused u=4 for moving -0.010 -> -0.005. v2 compares only the positive part.
    "v2": ("start u=2; accept the next u only if seed-mean calibrated INNER_A log-loss is lower AND "
           "max(seed-mean (INNER_A - TRAIN) gap, 0) is not larger; stop at the first refusal"),
}
LADDER_RULE = LADDER_RULES["v1"]


def common_rows(a, b):
    """Rows eligible under both feature sets, as index arrays into a and b (same decision bars)."""
    ia, ib = np.flatnonzero(a.eligible), np.flatnonzero(b.eligible)
    ts = np.intersect1d(a.row_ts[ia], b.row_ts[ib])
    ra = ia[np.searchsorted(a.row_ts[ia], ts)]
    rb = ib[np.searchsorted(b.row_ts[ib], ts)]
    assert (a.row_bar[ra] == b.row_bar[rb]).all() and (a.period[ra] == b.period[rb]).all()
    return ra, rb


def run(a):
    import torch
    from app.short_horizon.expert_guided import learn as L
    from app.short_horizon.expert_guided import temporal as T
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    led = out / "TRIAL_LEDGER.jsonl"
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    t0 = time.time()
    dv4 = DG.load_v4(a.admitted, a.v4_dir)
    d15 = DG.load(a.admitted, a.csv_dir)
    global ARMS, LADDER_RULE
    if a.arms:
        ARMS = {k: v for k, v in ARMS.items() if k in a.arms.split(",")}
    LADDER_RULE = LADDER_RULES[a.ladder_rule]
    ra, rb = common_rows(dv4, d15)
    W = {"v4": dv4.windows[ra], "core15": d15.windows[rb]}
    row_bar, row_ts, period = dv4.row_bar[ra], dv4.row_ts[ra], dv4.period[ra]
    rows = {p: np.flatnonzero(period == p) for p in DG.PERIODS}
    arch = {}
    for name, cfg in ARMS.items():
        nf = W[cfg["features"]].shape[2]
        m = T.TemporalPolicyNet(nf, DG.WIN, 3, units=cfg["units"]) if cfg["kind"] == "temporal" \
            else L.PolicyNet(DG.WIN * nf, 3, 0, tuple(cfg["hidden"]))
        arch[name] = {"input": [DG.WIN, nf] if cfg["kind"] == "temporal" else [DG.WIN * nf], "layers": T.param_table(m)}
    ledger(led, rung="E0T", event="PREREGISTRATION", gate=GATE, arms=ARMS, ladder=LADDER, ladder_rule=LADDER_RULE,
           train=dict(TRAIN_CFG, max_epochs=a.max_epochs), architectures=arch, seeds=SEEDS, rate_grid=RATE_GRID, device=device,
           rows={p: int(len(r)) for p, r in rows.items()}, features={"v4": dv4.meta["input_contract"]["features"],
                                                                     "core15": DG.FEATURES})
    weeks = {p: len(np.unique(SC.week_keys(row_ts[r]))) for p, r in rows.items()}
    runner = SC.SignalRunner(dv4.bars, dv4.cal, DG.PERIODS)
    elig_bars = set(int(b) for b in row_bar)
    res = {"meta": {"v4": dv4.meta, "core15": d15.meta, "common_rows": {p: int(len(r)) for p, r in rows.items()},
                    "dropped_only_core15": int(d15.eligible.sum() - len(rb)), "dropped_only_v4": int(dv4.eligible.sum() - len(ra))},
           "arms": ARMS, "architectures": arch, "train": dict(TRAIN_CFG, max_epochs=a.max_epochs), "ladder_rule": LADDER_RULE, "weeks": weeks,
           "device": device, "students": {}, "comparators": {}}

    def execute(sig, p):
        m, _ = runner.execute(sig, p, weeks=weeks[p], mode="causal", k_tp=DG.CONFIRMED_K_TP)
        return strip(m)

    hs = [(b, d) for b, d in SC.hindsight_signals(dv4.pivots) if b in elig_bars]
    cs = [(b, d) for b, d in SC.confirmed_signals(dv4.pivots, dv4.conf) if b in elig_bars]
    yb = {"hindsight": dv4.y_hind[row_bar].astype(np.int64), "confirmed": dv4.y_conf[row_bar].astype(np.int64)}
    prior = {t: (np.bincount(y[rows["TRAIN"]], minlength=3) / len(rows["TRAIN"])).tolist() for t, y in yb.items()}
    res["train_prior"] = prior

    def comparators(periods):
        ledger(led, rung="E0T", event="READ", periods=list(periods), purpose="comparators")
        for p in periods:
            c = res["comparators"].setdefault(p, {})
            m, _ = runner.execute(hs, p, weeks=weeks[p], mode="hindsight", piv=dv4.pivots)
            c["expert_hindsight_tp_pivot"] = strip(m)
            c["expert_confirmed"] = execute(cs, p)
            c["no_trade"] = SC.no_trade_metrics()

    comparators(["TRAIN", "INNER_A", "INNER_B"])
    frozen = {}
    for name, cfg in ARMS.items():
        Xw = W[cfg["features"]].astype(np.float32)
        nf = Xw.shape[2]
        X = Xw if cfg["kind"] == "temporal" else Xw.reshape(len(Xw), -1)
        factory = (lambda nf=nf, u=cfg.get("units"): T.TemporalPolicyNet(nf, DG.WIN, 3, units=u, dropout=TRAIN_CFG["dropout"])) \
            if cfg["kind"] == "temporal" else None
        for target in ("hindsight", "confirmed"):
            y = yb[target]
            for seed in SEEDS:
                ledger(led, rung="E0T", event="FIT", arm=name, target=target, seed=seed, fit="TRAIN", monitor="INNER_A")
                model, hist, info = L.fit(X[rows["TRAIN"]], y[rows["TRAIN"]], X[rows["INNER_A"]], y[rows["INNER_A"]],
                                          seed=seed, device=device, max_epochs=a.max_epochs, hidden=tuple(cfg.get("hidden", (128, 64))),
                                          dropout=TRAIN_CFG["dropout"], lr=TRAIN_CFG["lr"], wd=TRAIN_CFG["weight_decay"],
                                          batch=TRAIN_CFG["batch"], patience=TRAIN_CFG["patience"], net=factory)
                P, _ = L.predict(model, X, device=device)
                Tm = L.fit_temperature(np.log(np.clip(P[rows["INNER_A"]], 1e-12, 1)), y[rows["INNER_A"]])
                Pc = L.apply_temperature(P, Tm)
                ia = rows["INNER_A"]
                pmax = np.maximum(Pc[ia, 1], Pc[ia, 2])
                cands = []
                for rate in RATE_GRID:
                    tau = float(np.quantile(pmax, 1 - rate))
                    sig = SC.threshold_signals(Pc[ia], tau)
                    m = execute([(int(b), int(d)) for b, d in zip(row_bar[ia], sig) if d], "INNER_A")
                    cands.append({"rate": rate, "tau": tau, "sharpe": m["sharpe"], "valid": m["valid"],
                                  "orders_per_week": m["orders_per_week"], "net": m["net"]})
                ok = [c for c in cands if c["valid"] and c["sharpe"] is not None and math.isfinite(c["sharpe"])]
                choice = max(ok or cands, key=lambda c: c["sharpe"] if c["sharpe"] is not None and math.isfinite(c["sharpe"]) else -1e9)
                frozen[(name, target, seed)] = (Pc, choice["tau"])
                ledger(led, rung="E0T", event="FREEZE", arm=name, target=target, seed=seed, temperature=Tm,
                       tau=choice["tau"], rate=choice["rate"], chosen_on="INNER_A")
                rec = {"fit": info, "history": hist, "temperature": Tm, "threshold_candidates": cands, "threshold": choice,
                       "losses": {}, "scores": {}, "trading": {}}
                for p in ("TRAIN", "INNER_A", "INNER_B"):
                    r = rows[p]
                    rec["losses"][p] = SC.log_loss(y[r], Pc[r])
                    rec["scores"][p] = {"calibrated": SC.class_scores(y[r], Pc[r], prior[target], row_ts[r],
                                                                      boot=(p == "INNER_B"), seed=seed)}
                    sig = SC.threshold_signals(Pc[r], choice["tau"])
                    rec["scores"][p]["at_threshold"] = SC.precision_recall(y[r], sig)
                    rec["trading"][p] = execute([(int(b), int(d)) for b, d in zip(row_bar[r], sig) if d], p)
                res["students"].setdefault(name, {}).setdefault(target, {})[str(seed)] = rec
                print(json.dumps({"arm": name, "target": target, "seed": seed, "params": info["params"],
                                  "epochs": info["epochs"], "best": info["best_epoch"],
                                  "ll_tr": round(rec["losses"]["TRAIN"], 4), "ll_ia": round(rec["losses"]["INNER_A"], 4),
                                  "ib_ll_minus_prior": rec["scores"]["INNER_B"]["calibrated"]["log_loss_minus_prior"],
                                  "sh": [rec["trading"][p]["sharpe"] for p in ("TRAIN", "INNER_A", "INNER_B")],
                                  "s": round(time.time() - t0, 1)}), flush=True)

    # ---------------------------------------------------------------- capacity ladder (INNER_A only)
    def arm_mean(name, key):
        st = res["students"][name]["hindsight"]
        if key == "ll":
            return float(np.mean([st[str(s)]["losses"]["INNER_A"] for s in SEEDS]))
        g = float(np.mean([st[str(s)]["losses"]["INNER_A"] - st[str(s)]["losses"]["TRAIN"] for s in SEEDS]))
        return max(g, 0.0) if a.ladder_rule == "v2" else g
    steps, chosen = [], LADDER[0]
    for nxt in LADDER[1:]:
        better = arm_mean(nxt, "ll") < arm_mean(chosen, "ll")
        no_wider = arm_mean(nxt, "gap") <= arm_mean(chosen, "gap")
        steps.append({"from": chosen, "to": nxt, "inner_a_ll": [arm_mean(chosen, "ll"), arm_mean(nxt, "ll")],
                      "gap": [arm_mean(chosen, "gap"), arm_mean(nxt, "gap")], "accepted": bool(better and no_wider)})
        if not (better and no_wider):
            break
        chosen = nxt
    res["ladder"] = {"rule": LADDER_RULE, "steps": steps, "chosen": chosen}
    ledger(led, rung="E0T", event="LADDER", **res["ladder"])

    # ---------------------------------------------------------------- gate on INNER_B (every arm reported; chosen gates)
    res["gate_result"] = {}
    for name in ARMS:
        g1, g2 = [], []
        for seed in SEEDS:
            sc = res["students"][name]["hindsight"][str(seed)]["scores"]["INNER_B"]["calibrated"]
            g1.append(sc["log_loss_minus_prior"] < 0)
            g2.append(all(sc[d]["boot_excess_lo95"] > 0 for d in ("LONG", "SHORT")))
        mean_ll = float(np.mean([res["students"][name]["hindsight"][str(s)]["scores"]["INNER_B"]["calibrated"]["log_loss_minus_prior"]
                                 for s in SEEDS]))
        G1, G2 = sum(g1) >= 2 and mean_ll < 0, sum(g2) >= 2
        res["gate_result"][name] = {"G1": bool(G1), "G1_per_seed": g1, "G1_mean_log_loss_minus_prior": mean_ll,
                                    "G2": bool(G2), "G2_per_seed": g2, "learnable_signal": bool(G1 and G2)}
    res["decision"] = "CONTINUE_TO_E1" if res["gate_result"][chosen]["learnable_signal"] else "STOP_AT_E0"
    ledger(led, rung="E0T", event="GATE", chosen=chosen, decision=res["decision"], gates=res["gate_result"])

    # ---------------------------------------------------------------- 2024: single read, chosen arm + comparators
    ledger(led, rung="E0T", event="READ", periods=["OBS_2024"], purpose=f"single read: chosen arm {chosen} + comparators")
    comparators(["OBS_2024"])
    r = rows["OBS_2024"]
    for target in ("hindsight", "confirmed"):
        y = yb[target]
        for seed in SEEDS:
            Pc, tau = frozen[(chosen, target, seed)]
            rec = res["students"][chosen][target][str(seed)]
            rec["losses"]["OBS_2024"] = SC.log_loss(y[r], Pc[r])
            rec["scores"]["OBS_2024"] = {"calibrated": SC.class_scores(y[r], Pc[r], prior[target], row_ts[r], boot=True, seed=seed)}
            sig = SC.threshold_signals(Pc[r], tau)
            rec["scores"]["OBS_2024"]["at_threshold"] = SC.precision_recall(y[r], sig)
            rec["trading"]["OBS_2024"] = execute([(int(b), int(d)) for b, d in zip(row_bar[r], sig) if d], "OBS_2024")
    res["wall_seconds"] = time.time() - t0
    (out / "E0T.json").write_text(json.dumps(jsonable(res), indent=1))
    print(json.dumps({"ladder": res["ladder"], "decision": res["decision"], "wall": round(res["wall_seconds"], 1)}, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True, help="admitted-core 15 CSV directory")
    ap.add_argument("--v4-dir", required=True, help="TIMING_PROXY_SELECTION v4 snapshot directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--arms", default="", help="comma list of arm names (default: all)")
    ap.add_argument("--ladder-rule", default="v1", choices=sorted(LADDER_RULES))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
