#!/usr/bin/env python3
"""E1-T: behaviour cloning + DAgger-lite with the MINIMAL TEMPORAL trunk (owner rule 2026-10-20) (DEVELOPMENT).

Copy of ``expert_guided_e1.py`` (unchanged) with one change of model: the window is NOT flattened into an MLP;
``temporal.TemporalPolicyNet`` reads it as (13, F) (per-feature causal Conv1D branches, fusion, dilations 1/2/4,
last step), the agent state is concatenated AFTER the trunk. Features (v4 four or admitted-core 15) and units u
come from ``--features`` / ``--units`` (chosen by tools/expert_guided_select_t.py from E0-T run B). Policy on
[13 x F window, agent state (env.STATE_NAMES)] -> {HOLD, LONG, SHORT, CLOSE} + (TP, SL) sigma multiples in [0.25, 3] / [0.5, 3]
(owner rule: always set; the env clips them into the effective band). Owner's stopping rule (mean of TRAIN and
INNER_A loss per epoch, patience 10), 3 seeds.

  round 0 (BC)   demonstrations = the ZigZag v2 expert's own trajectory in the env (its issued action at every
                 observable bar: LONG/SHORT at its pivot signals with TP = next pivot level, SL = 2 sigma; HOLD
                 otherwise), TRAIN fits, INNER_A (expert trajectory) monitors;
  rounds 1..3    the current policy acts deterministically (argmax) in the env on TRAIN and INNER_A; every visited
                 observable state is relabelled by the hindsight LegOracle in the policy's own state (position,
                 entry gate); the aggregated set is refitted from scratch (same seed); class weights 1/sqrt(freq)
                 of the aggregated TRAIN labels (floor 1e-3), recomputed each round (declared).
  The final policy is round 3 (fixed in advance: no round is chosen on any score).

Pre-registered gate (ledgered before the first fit):
  I1  round-3 policy, its own INNER_B visited states: oracle-label log-loss < the (Laplace-smoothed) TRAIN
      action-prior log-loss of the aggregated round-3 labels,
      in >= 2 of 3 seeds;
  I2  round-3 INNER_B Sharpe, seed mean > confirmed expert AND > TRAIN-prior random policy (seed mean), same env;
  learnable signal = I1 AND I2 -> E2; else STOP at E1. Sharpe > 0 (no-trade) is reported, not gating.
2024 is read once after the round-3 freeze of all seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from app.short_horizon.expert_guided import oracle as OR  # noqa: E402
from tools.expert_guided_e0 import jsonable, ledger  # noqa: E402

SEEDS = (20261019, 20261020, 20261021)
ROUNDS = 3
GATE = {"I1": "round-3 policy INNER_B own visited states: oracle-label log-loss < TRAIN action-prior log-loss in >= 2/3 seeds",
        "I2": "round-3 INNER_B Sharpe seed mean > confirmed expert AND > TRAIN-prior random (seed mean), same bar env",
        "learnable_signal": "I1 AND I2 -> E2; else STOP at E1", "no_trade": "Sharpe > 0 reported, not gating"}
MODEL = {"architecture": "TemporalPolicyNet (temporal.py)", "dropout": 0.1, "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "batch": 256,
         "max_epochs": 200, "patience": 10, "loss": "class-weighted CE (1/sqrt freq) + Huber on (TP, SL) for LONG/SHORT labels",
         "batchnorm": False, "decision": "argmax", "rounds": ROUNDS, "final_round": ROUNDS}
STUDENT_CFG = E.EnvConfig(level_policy="clip")
EXPERT_CFG = E.EnvConfig(level_policy="refuse", tp_bounds=(1e-6, 1e3), sl_bounds=(1e-6, 1e3))
# sensitivity only (Musashi's SAC contract: <= 1 entry/day, leverage 1:50); never used for any choice
MUSASHI_CFG = E.EnvConfig(level_policy="clip", max_entries_per_day=1,
                          sizing=E.X.Sizing(rel_volume=0.10, leverage=50.0))


def summary(m):
    out = E.summary(m)
    out["entries"], out["closes"] = out.get("n_entries"), out.get("n_closes")
    return out


class Ctx:
    def __init__(self, ds):
        self.ds = ds
        self.rob = ds.row_of_bar()
        self.v = ds.bars.index.as_unit("ns").asi8
        self.obs_ok = ds.observable()
        self.lab_ok = ds.labelled()
        self.Xw = ds.windows.reshape(len(ds.windows), -1).astype(np.float32)
        self.oracle = OR.LegOracle(ds.bars.CLOSE.to_numpy(float), ds.pivots)

    def rollout(self, period, act, cfg=STUDENT_CFG, collect=False):
        a, b = DG.PERIODS[period]
        env = E.BarEnv(self.ds.bars, self.ds.cal, self.ds.sigma, self.ds.sigma_rule, a, b, cfg)
        vis = []
        for bar in range(env.first, env.last):
            r = self.rob[bar]
            st = env.state(bar)
            if r >= 0 and self.obs_ok[r]:
                action, tp, sl = act(bar, r, st, env)
                if collect and self.lab_ok[r]:
                    vis.append((r, bar, st, action, tp, sl))
            else:
                action, tp, sl = E.HOLD, 1.5, 2.0
            env.step(bar, action, tp, sl)
        return env.result(), vis

    def relabel(self, vis):
        X, y, R, Rm = [], [], [], []
        for r, bar, st, *_ in vis:
            s = float(self.ds.sigma[bar + 1])
            a, tp, sl = self.oracle.action(bar, int(st[0]), bool(st[-1] > 0.5), s)
            X.append(np.r_[self.Xw[r], st])
            y.append(a)
            R.append((tp, sl))
            Rm.append(float(a in (E.LONG, E.SHORT)))
        return pack(X, y, R, Rm)

    def demo(self, vis):
        X, y, R, Rm = [], [], [], []
        for r, bar, st, a, tp, sl in vis:
            X.append(np.r_[self.Xw[r], st])
            y.append(a)
            R.append((float(np.clip(tp, 0.25, 3.0)), float(np.clip(sl, 0.5, 3.0))))
            Rm.append(float(a in (E.LONG, E.SHORT)))
        return pack(X, y, R, Rm)

    # ---- policies
    def expert(self, bar, r, st, env):
        y = self.ds.y_hind[bar]
        if y == 0:
            return E.HOLD, 1.5, 2.0
        s = float(self.ds.sigma[bar + 1])
        rem = self.oracle.remaining[bar]
        tp = rem / s if np.isfinite(rem) and s > 0 else -1.0
        if tp <= 0:
            return E.HOLD, 1.5, 2.0     # the hindsight order would be invalid (TP on the wrong side)
        return (E.LONG if y == OR.LONG_CLASS else E.SHORT), tp, 2.0

    def oracle_policy(self, bar, r, st, env):
        return self.oracle.action(bar, int(st[0]), bool(st[-1] > 0.5), float(self.ds.sigma[bar + 1]))

    def confirmed(self, bar, r, st, env):
        y = self.ds.y_conf[bar]
        return (E.HOLD if y == 0 else (E.LONG if y == OR.LONG_CLASS else E.SHORT)), DG.CONFIRMED_K_TP, 2.0

    def student(self, pol):
        def f(bar, r, st, env):
            lg, rg = pol(np.r_[self.Xw[r], st][None, :])
            return int(np.argmax(lg[0])), float(rg[0, 0]), float(rg[0, 1])
        return f

    def prior_random(self, prior, seed):
        rng = np.random.default_rng(seed)
        cum = np.cumsum(prior)

        def f(bar, r, st, env):
            return int(np.searchsorted(cum, rng.random() * cum[-1], side="right")), 1.5, 2.0
        return f


def pack(X, y, R, Rm):
    return (np.asarray(X, np.float32), np.asarray(y, np.int64), np.asarray(R, np.float32), np.asarray(Rm, np.float32))


def cat(a, b):
    return tuple(np.concatenate([x, z]) for x, z in zip(a, b))


def weights(y):
    f = np.maximum(np.bincount(y, minlength=4) / len(y), 1e-3)
    w = 1 / np.sqrt(f)
    return w / (w[y].mean())


def imitation_scores(pol, vis_lab, prior):
    X, y, _, _ = vis_lab
    if not len(y):
        return {"rows": 0}
    lg, _ = pol(X)
    z = lg - lg.max(1, keepdims=True)
    P = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    ll = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))
    pll = float(-np.mean(np.log(np.clip(np.asarray(prior)[y], 1e-12, 1))))
    return {"rows": int(len(y)), "log_loss": ll, "prior_log_loss": pll, "log_loss_minus_prior": ll - pll,
            "accuracy": float((P.argmax(1) == y).mean()), "label_counts": np.bincount(y, minlength=4).tolist(),
            "pred_counts": np.bincount(P.argmax(1), minlength=4).tolist()}


def run(a):
    import torch
    from app.short_horizon.expert_guided import learn as L
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    led = out / "TRIAL_LEDGER.jsonl"
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    t0 = time.time()
    from app.short_horizon.expert_guided import temporal as T
    ds = DG.load(a.admitted, a.csv_dir) if a.features == "core15" else DG.load_v4(a.admitted, a.v4_dir)
    cx = Ctx(ds)
    nf = ds.windows.shape[2]
    MODEL.update(features=a.features, feature_names=ds.meta["input_contract"]["features"], units=a.units,
                 params=int(sum(p.numel() for p in T.TemporalPolicyNet(nf, DG.WIN, 4, n_reg=2, n_state=len(E.STATE_NAMES), units=a.units).parameters())),
                 layers=T.param_table(T.TemporalPolicyNet(nf, DG.WIN, 4, n_reg=2, n_state=len(E.STATE_NAMES), units=a.units)))
    factory = lambda: T.TemporalPolicyNet(nf, DG.WIN, 4, n_reg=2, n_state=len(E.STATE_NAMES), units=a.units,  # noqa: E731
                                          dropout=MODEL["dropout"])
    ledger(led, rung="E1T", event="PREREGISTRATION", gate=GATE, model=MODEL, seeds=SEEDS, periods=DG.PERIODS,
           student_env=str(STUDENT_CFG), expert_env=str(EXPERT_CFG), max_epochs=a.max_epochs, device=device)
    res = {"meta": ds.meta, "gate": GATE, "model": MODEL, "device": device, "comparators": {}, "seeds": {},
           "env": {"student": str(STUDENT_CFG), "expert": str(EXPERT_CFG), "state_names": E.STATE_NAMES}}
    # demonstrations: the expert's own trajectory (TRAIN fit, INNER_A monitor)
    ledger(led, rung="E1T", event="READ", periods=["TRAIN", "INNER_A", "INNER_B"], purpose="expert/oracle/confirmed replays")
    demo = {}
    for p in ("TRAIN", "INNER_A"):
        m, vis = cx.rollout(p, cx.expert, EXPERT_CFG, collect=True)
        demo[p] = cx.demo(vis)
    prior = (np.bincount(demo["TRAIN"][1], minlength=4) / len(demo["TRAIN"][1])).tolist()
    res["train_action_prior_bc"] = prior

    def comparators(periods):
        for p in periods:
            c = res["comparators"].setdefault(p, {})
            c["expert_demonstrations"] = summary(cx.rollout(p, cx.expert, EXPERT_CFG)[0])
            c["oracle_leg_following"] = summary(cx.rollout(p, cx.oracle_policy, EXPERT_CFG)[0])
            c["expert_confirmed"] = summary(cx.rollout(p, cx.confirmed, STUDENT_CFG)[0])
            c["train_prior_random"] = [summary(cx.rollout(p, cx.prior_random(prior, s), STUDENT_CFG)[0]) for s in SEEDS]
            c["no_trade"] = summary(cx.rollout(p, lambda *x: (E.HOLD, 1.5, 2.0), STUDENT_CFG)[0])
    if a.measure:   # memory/time pilot: TRAIN/INNER_A only, one seed, rounds 0-1, nothing scored or written as a result
        comparators(["TRAIN", "INNER_A"])
        seeds, rounds, eval_p = SEEDS[:1], 1, ("TRAIN", "INNER_A")
    else:
        comparators(["TRAIN", "INNER_A", "INNER_B"])
        seeds, rounds, eval_p = SEEDS, ROUNDS, ("TRAIN", "INNER_A", "INNER_B")
    last = eval_p[-1]
    print(json.dumps({f"comparators_{last}": {k: (v["sharpe_daily_sqrt365"] if isinstance(v, dict) else
                                                 [x["sharpe_daily_sqrt365"] for x in v])
                                             for k, v in res["comparators"][last].items()}, "s": round(time.time() - t0, 1)}),
          flush=True)
    finals = {}
    for seed in seeds:
        D_tr, D_in = demo["TRAIN"], demo["INNER_A"]
        rec = res["seeds"].setdefault(str(seed), {"rounds": []})
        pol = None
        for rnd in range(rounds + 1):
            if rnd > 0:
                _, vtr = cx.rollout("TRAIN", cx.student(pol), collect=True)
                _, vin = cx.rollout("INNER_A", cx.student(pol), collect=True)
                D_tr, D_in = cat(D_tr, cx.relabel(vtr)), cat(D_in, cx.relabel(vin))
            w = weights(D_tr[1])
            ledger(led, rung="E1T", event="FIT", seed=seed, round=rnd, rows_train=len(D_tr[1]), rows_inner=len(D_in[1]),
                   class_weight=w.tolist())
            model, hist, info = L.fit(D_tr[0], D_tr[1], D_in[0], D_in[1], seed=seed, n_cls=4, R=D_tr[2], Rm=D_tr[3],
                                      Ri=D_in[2], Rmi=D_in[3], class_weight=w, net=factory,
                                      dropout=MODEL["dropout"], lr=MODEL["lr"], wd=MODEL["weight_decay"],
                                      batch=MODEL["batch"], max_epochs=a.max_epochs, patience=MODEL["patience"], device=device)
            pol = T.TorchPolicy(model)
            r = {"round": rnd, "fit": info, "history": hist, "rows_train": int(len(D_tr[1])), "rows_inner": int(len(D_in[1])),
                 "label_counts_train": np.bincount(D_tr[1], minlength=4).tolist(), "class_weight": w.tolist(),
                 "trading": {}, "imitation": {}}
            aprior = ((np.bincount(D_tr[1], minlength=4) + 1) / (len(D_tr[1]) + 4)).tolist()   # Laplace-smoothed TRAIN prior
            ledger(led, rung="E1T", event="READ", seed=seed, round=rnd, periods=["TRAIN", "INNER_A", "INNER_B"],
                   purpose="diagnostic per round; no round is chosen on it")
            for p in eval_p:
                m, vis = cx.rollout(p, cx.student(pol), collect=True)
                r["trading"][p] = summary(m)
                r["imitation"][p] = imitation_scores(pol, cx.relabel(vis), aprior)
            rec["rounds"].append(r)
            print(json.dumps({"seed": seed, "round": rnd, "epochs": info["epochs"], "best": info["best_epoch"],
                              "last_sharpe": r["trading"][eval_p[-1]]["sharpe_daily_sqrt365"],
                              "last_trades": r["trading"][eval_p[-1]]["trades"],
                              "last_ll_minus_prior": r["imitation"][eval_p[-1]].get("log_loss_minus_prior"),
                              "s": round(time.time() - t0, 1)}), flush=True)
        finals[seed] = (pol, aprior)
        ledger(led, rung="E1T", event="FREEZE", seed=seed, round=ROUNDS)
        torch.save(pol.m.state_dict(), out / f"policy_{seed}.pt")
    if a.measure:
        print(json.dumps({"measure_only": True, "wall": round(time.time() - t0, 1)}))
        return
    # gate
    fin = [res["seeds"][str(s)]["rounds"][-1] for s in SEEDS]
    i1 = [f["imitation"]["INNER_B"]["log_loss_minus_prior"] < 0 for f in fin]
    sh = float(np.mean([f["trading"]["INNER_B"]["sharpe_daily_sqrt365"] for f in fin]))
    cB = res["comparators"]["INNER_B"]
    conf = cB["expert_confirmed"]["sharpe_daily_sqrt365"]
    prr = float(np.mean([x["sharpe_daily_sqrt365"] for x in cB["train_prior_random"]]))
    I1, I2 = sum(i1) >= 2, (sh > conf and sh > prr)
    res["gate_result"] = {"I1": bool(I1), "I1_per_seed": i1, "I2": bool(I2), "inner_b_sharpe_mean": sh,
                          "confirmed": conf, "prior_random_mean": prr, "beats_no_trade": bool(sh > 0),
                          "learnable_signal": bool(I1 and I2), "decision": "CONTINUE_TO_E2" if (I1 and I2) else "STOP_AT_E1"}
    ledger(led, rung="E1T", event="GATE", **res["gate_result"])
    # 2024 once (skipped in the capacity sweep: 2024 is read only for the arm chosen on INNER_A)
    if a.skip_2024:
        ledger(led, rung="E1T", event="NO_READ", periods=["OBS_2024"], purpose="capacity sweep arm; 2024 not read")
    else:
        ledger(led, rung="E1T", event="READ", periods=["OBS_2024"], purpose="single read of the frozen round-3 policies + comparators")
        comparators(["OBS_2024"])
        for s in SEEDS:
            pol, aprior = finals[s]
            m, vis = cx.rollout("OBS_2024", cx.student(pol), collect=True)
            fr = res["seeds"][str(s)]["rounds"][-1]
            fr["trading"]["OBS_2024"] = summary(m)
            fr["imitation"]["OBS_2024"] = imitation_scores(pol, cx.relabel(vis), aprior)
            res.setdefault("musashi_contract_sensitivity", {})[str(s)] = {
                p: summary(cx.rollout(p, cx.student(pol), MUSASHI_CFG)[0]) for p in ("TRAIN", "INNER_A", "INNER_B", "OBS_2024")}
            eq = m["daily_equity"]
            np.savez_compressed(out / f"equity_2024_{s}.npz", t=eq.index.asi8, v=eq.to_numpy())
    res["wall_seconds"] = time.time() - t0
    (out / "E1T.json").write_text(json.dumps(jsonable(res), indent=1))
    print(json.dumps({"gate": res["gate_result"], "wall": round(res["wall_seconds"], 1)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--measure", action="store_true", help="pilot: no INNER_B / 2024 read, one seed, rounds 0-1")
    ap.add_argument("--features", choices=["v4", "core15"], required=True)
    ap.add_argument("--v4-dir", default="")
    ap.add_argument("--units", type=int, required=True)
    ap.add_argument("--skip-2024", action="store_true", help="capacity sweep: never read 2024")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
