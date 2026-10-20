#!/usr/bin/env python3
"""E2-T: SAC with vs without hindsight-expert demonstrations, MINIMAL TEMPORAL trunk (owner rule 2026-10-20).

Copy of ``expert_guided_e2.py`` (unchanged) with the model changed: the SB3 features extractor is
``temporal.TemporalExtractor`` (flat obs reshaped back to (13, F) -> per-feature causal Conv1D branches, fusion,
dilations 1/2/4, last step; agent state concatenated after the trunk) and net_arch = [u] for actor and critics;
features and u come from E0-T run B via tools/expert_guided_select_t.py. ORDER (owner 2026-10-20): E2-T runs
after E1-T whatever the E1-T gate says; it is not a gate pass. The E0 reference row is replaced by the frozen
E1-T round-3 policies. The design below is fixed before the first fit; nothing is tuned on INNER_B or 2024.

Arms (identical env, inputs, rules, seeds, budget, network, optimiser, stopping rule; shaping OFF):
  SAC          DemoSAC with demo_frac0 = 0, bc_lambda0 = 0 (= SB3 2.9 SAC update);
  SAC_DEMO     DemoSAC with demo replay (25 % of each batch at the start, linear decay to 0 over 50k updates) and a
               Q-filtered BC actor term (lambda 1.0, same decay) on the ZigZag v2 expert demonstrations recomputed on
               the admitted clock (expert's own trajectory in the same env, TRAIN segments only).
Budget 100k environment transitions per run (gradient step per transition after 1,000 warm-up), 3 seeds.
SAC: lr 3e-4, buffer 100k, batch 256, tau 0.005, gamma 0.99, entropy auto; extractor = temporal trunk (u units,
dropout 0.1) + state, net_arch [u], AdamW weight decay 1e-4, no BatchNorm.
Early stopping (Musashi's ES design, owner's rule): every 5,000 transitions the checkpoint is frozen and run
deterministically over the whole TRAIN and INNER_A periods in the bar env; score = mean of the two net daily Sharpe
ratios (ruined -> -10); best checkpoint kept; patience 10 evaluations. INNER_B is read after each run's freeze;
2024 is read once after every run is frozen.
Reference rows under identical accounting: the E0 hindsight-student classifier policy (its frozen temperature and
threshold), the expert as representable in the action space, the expert demonstrations, the confirmed expert,
TRAIN-prior random and no-trade.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import actions as A  # noqa: E402
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from app.short_horizon.expert_guided import scoring as SC  # noqa: E402
from tools.expert_guided_e0 import jsonable, ledger  # noqa: E402
from tools.expert_guided_e1t import EXPERT_CFG, MUSASHI_CFG, STUDENT_CFG, Ctx, summary  # noqa: E402

SEEDS = (20261019, 20261020, 20261021)
ARMS = {"SAC": {"demo_frac0": 0.0, "bc_lambda0": 0.0}, "SAC_DEMO": {"demo_frac0": 0.25, "bc_lambda0": 1.0}}
DESIGN = {"deviation": "E2-T runs after E1-T by owner order 2026-10-20, whatever the E1-T gate; not a gate pass",
          "arms": ARMS, "demo_decay_updates": 50_000, "budget_transitions": 100_000, "learning_starts": 1000,
          "sac": {"lr": 3e-4, "buffer": 100_000, "batch": 256, "tau": 0.005, "gamma": 0.99, "ent_coef": "auto",
                  "train_freq": 1, "gradient_steps": 1},
          "network": {"extractor": "TemporalExtractor (temporal.py)", "net_arch": "[u]", "optimizer": "AdamW",
                      "weight_decay": 1e-4, "batchnorm": False},
          "early_stopping": {"every": 5000, "score": "mean(net daily Sharpe TRAIN, INNER_A), frozen deterministic checkpoint; a checkpoint failing the frequency rule or ruined scores -10 (amendment 2026-10-20)",
                             "patience_evals": 10, "ruined_sharpe": -10.0},
          "episode": "one random TRAIN calendar segment (week); capital 10,000 USD per episode",
          "reward": "100 x delta mark-to-market equity / 10,000, net of all costs, UNSHAPED",
          "action": "x[0] target position (dead band 1/3), x[1] TP sigma in [0.25,3], x[2] SL sigma in [0.5,3]",
          "shaping": "OFF", "privileged_critic": "NO"}


def sharpe(m):
    s = m.get("sharpe_daily_sqrt365")
    return -10.0 if s is None or not np.isfinite(s) else float(s)


def es_sharpe(m):
    """Early-stopping score only (E2-T amendment, declared 2026-10-20 after the pilot picked a no-trade checkpoint):
    a checkpoint that breaks the frequency rule (>= 3 orders/week, <= 4/day) or is ruined scores -10, because a
    policy that does not trade is not a result (owner rule). Reports keep the raw Sharpe."""
    return sharpe(m) if m.get("valid") and not m.get("ruined") else -10.0


def actor_policy(cx, actor):
    def f(bar, r, st, env):
        a = actor(np.r_[cx.Xw[r], st].astype(np.float32))
        return A.decode(a, int(st[0]))
    return f


def demo_transitions(cx, period):
    from app.short_horizon.expert_guided.sac_demo import SegmentEnv
    env = SegmentEnv(cx, period, seed=0, sequential=True)
    O, Ac, R, O2, D = [], [], [], [], []
    for _ in range(len(env.segs)):
        obs, _ = env.reset()
        done = False
        while not done:
            r = cx.rob[env.b]
            st = env.env.state(env.b)
            a, tp, sl = cx.expert(env.b, r, st, env.env)
            x = A.encode(a, env.position(), tp, sl)
            obs2, rew, done, _, _ = env.step(x)
            O.append(obs), Ac.append(x), R.append([rew]), O2.append(obs2), D.append([float(done)])
            obs = obs2
    return {"obs": np.asarray(O, np.float32), "act": np.asarray(Ac, np.float32), "rew": np.asarray(R, np.float32),
            "next_obs": np.asarray(O2, np.float32), "done": np.asarray(D, np.float32)}


def run(a):
    import torch as th
    from stable_baselines3.common.callbacks import BaseCallback
    from app.short_horizon.expert_guided.sac_demo import DemoSAC, SegmentEnv
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    led = out / "TRIAL_LEDGER.jsonl"
    device = "cuda" if th.cuda.is_available() and not a.cpu else "cpu"
    t0 = time.time()
    from app.short_horizon.expert_guided import temporal as T
    ds = DG.load(a.admitted, a.csv_dir) if a.features == "core15" else DG.load_v4(a.admitted, a.v4_dir)
    cx = Ctx(ds)
    nf, u = ds.windows.shape[2], a.units
    NS = len(E.STATE_NAMES)
    DESIGN["network"].update(features=a.features, feature_names=ds.meta["input_contract"]["features"], units=u,
                             net_arch=[u])
    budget = a.budget
    seeds = SEEDS[:1] if a.measure else SEEDS
    arms = {"SAC_DEMO": ARMS["SAC_DEMO"]} if a.measure else ARMS
    eval_every = a.eval_every
    ledger(led, rung="E2T", event="DECLARED_DEVIATION", design=DESIGN, seeds=seeds, budget=budget, device=device,
           measure=a.measure, note="E2-T after E1-T by owner order; not a gate pass")
    res = {"meta": ds.meta, "design": DESIGN, "device": device, "runs": {}, "references": {}}
    demo = demo_transitions(cx, "TRAIN")
    res["demo"] = {"transitions": int(len(demo["rew"])), "reward_sum": float(demo["rew"].sum()),
                   "action_counts_target": np.bincount((np.sign(demo["act"][:, 0]) + 1).astype(int), minlength=3).tolist()}
    np.savez_compressed(out / "demo_transitions_train.npz", **demo)
    print(json.dumps({"demo": res["demo"], "s": round(time.time() - t0, 1)}), flush=True)

    # reference rows (TRAIN / INNER_A / INNER_B now, 2024 later)
    e1pol = {}
    for s in SEEDS:
        if a.measure and not (Path(a.e1_run) / f"policy_{s}.pt").exists():
            continue
        net = T.TemporalPolicyNet(nf, DG.WIN, 4, n_reg=2, n_state=NS, units=u)
        net.load_state_dict(th.load(Path(a.e1_run) / f"policy_{s}.pt"))
        e1pol[s] = cx.student(T.TorchPolicy(net))

    def expert_repr(bar, r, st, env):
        act, tp, sl = cx.expert(bar, r, st, env)
        return A.decode(A.encode(act, int(st[0]), tp, sl), int(st[0]))

    _, vis = cx.rollout("TRAIN", cx.expert, EXPERT_CFG, collect=True)       # the same TRAIN action prior as E1
    prior = (np.bincount(np.array([v[3] for v in vis], int), minlength=4) / len(vis)).tolist()
    res["train_action_prior"] = prior

    def references(periods):
        for p in periods:
            c = res["references"].setdefault(p, {})
            c["E1T_round3_policy"] = [summary(cx.rollout(p, e1pol[s], STUDENT_CFG)[0]) for s in e1pol]
            c["expert_representable_in_action_space"] = summary(cx.rollout(p, expert_repr, STUDENT_CFG)[0])
            c["expert_demonstrations"] = summary(cx.rollout(p, cx.expert, EXPERT_CFG)[0])
            c["expert_confirmed"] = summary(cx.rollout(p, cx.confirmed, STUDENT_CFG)[0])
            c["train_prior_random"] = [summary(cx.rollout(p, cx.prior_random(prior, s), STUDENT_CFG)[0]) for s in SEEDS]
            c["no_trade"] = summary(cx.rollout(p, lambda *x: (E.HOLD, 1.5, 2.0), STUDENT_CFG)[0])
    ledger(led, rung="E2T", event="READ", periods=["TRAIN", "INNER_A"] + ([] if a.measure else ["INNER_B"]),
           purpose="reference rows")
    references(["TRAIN", "INNER_A"] + ([] if a.measure else ["INNER_B"]))

    frozen = {}
    for arm, kw in arms.items():
        for seed in seeds:
            key = f"{arm}_{seed}"
            env = SegmentEnv(cx, "TRAIN", seed=seed)
            model = DemoSAC("MlpPolicy", env, demo=demo if kw["demo_frac0"] > 0 else None, demo_decay=50_000,
                            learning_rate=3e-4, buffer_size=100_000, learning_starts=1000, batch_size=256, tau=0.005,
                            gamma=0.99, train_freq=1, gradient_steps=1, ent_coef="auto", seed=seed, device=device, verbose=0,
                            policy_kwargs=dict(features_extractor_class=T.TemporalExtractor,
                                               features_extractor_kwargs=dict(n_feat=nf, win=DG.WIN, n_state=NS, units=u),
                                               net_arch=[u],
                                               optimizer_class=th.optim.AdamW, optimizer_kwargs=dict(weight_decay=1e-4)),
                            **kw)
            curve, best = [], {"score": -np.inf, "state": None, "step": 0, "wait": 0}

            class ES(BaseCallback):
                def _on_step(self):
                    n = self.num_timesteps
                    if n % eval_every:
                        return True
                    actor = T.TorchActor(model.actor.state_dict(), nf, DG.WIN, NS, u, [u])
                    mt = summary(cx.rollout("TRAIN", actor_policy(cx, actor))[0])
                    mi = summary(cx.rollout("INNER_A", actor_policy(cx, actor))[0])
                    score = 0.5 * (es_sharpe(mt) + es_sharpe(mi))
                    curve.append({"transitions": n, "train_sharpe": sharpe(mt), "inner_a_sharpe": sharpe(mi), "score": score,
                                  "gap_inner_minus_train": sharpe(mi) - sharpe(mt), "train_trades": mt["trades"],
                                  "train_valid": bool(mt.get("valid")), "inner_a_valid": bool(mi.get("valid")),
                                  "inner_a_trades": mi["trades"], "train_opw": mt["orders_per_week"],
                                  "train_net": mt["net_profit_usd"], "inner_a_net": mi["net_profit_usd"]})
                    if score > best["score"] + 1e-9:
                        best.update(score=score, state=copy.deepcopy(model.actor.state_dict()), step=n, wait=0)
                    else:
                        best["wait"] += 1
                    print(json.dumps({"run": key, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in curve[-1].items()},
                                      "s": round(time.time() - t0, 1)}), flush=True)
                    return best["wait"] < 10

            ledger(led, rung="E2T", event="FIT", arm=arm, seed=seed, budget=budget)
            tf = time.time()
            model.learn(total_timesteps=budget, callback=ES())
            stop = "patience" if best["wait"] >= 10 else "budget"
            model.actor.load_state_dict(best["state"])
            actor = T.TorchActor(model.actor.state_dict(), nf, DG.WIN, NS, u, [u])
            frozen[key] = actor
            th.save(best["state"], out / f"actor_{key}.pt")
            ledger(led, rung="E2T", event="FREEZE", arm=arm, seed=seed, best_transitions=best["step"], stop=stop)
            rec = {"arm": arm, "seed": seed, "curve": curve, "best_transitions": best["step"], "best_score": best["score"],
                   "stop": stop, "transitions": int(model.num_timesteps), "seconds": time.time() - tf,
                   "demo_log": model.demo_log, "trading": {}}
            periods = ("TRAIN", "INNER_A") if a.measure else ("TRAIN", "INNER_A", "INNER_B")
            ledger(led, rung="E2T", event="READ", arm=arm, seed=seed, periods=list(periods), purpose="frozen checkpoint")
            for p in periods:
                rec["trading"][p] = summary(cx.rollout(p, actor_policy(cx, actor))[0])
            res["runs"][key] = rec
            print(json.dumps({"run": key, "stop": stop, "best": best["step"],
                              "sharpe": {p: rec["trading"][p]["sharpe_daily_sqrt365"] for p in periods},
                              "s": round(time.time() - t0, 1)}), flush=True)
    if a.measure:
        (out / "E2_MEASURE.json").write_text(json.dumps(jsonable(res), indent=1))
        print(json.dumps({"measure_only": True, "wall": round(time.time() - t0, 1)}))
        return
    if a.skip_2024:
        ledger(led, rung="E2T", event="NO_READ", periods=["OBS_2024"], purpose="capacity sweep arm; 2024 not read")
        frozen = {}
    else:
        ledger(led, rung="E2T", event="READ", periods=["OBS_2024"], purpose="single read of every frozen E2 run and the references")
        references(["OBS_2024"])
    for key, actor in frozen.items():
        res["runs"][key]["trading"]["OBS_2024"] = summary(cx.rollout("OBS_2024", actor_policy(cx, actor))[0])
        res["runs"][key]["musashi_contract_sensitivity"] = {
            p: summary(cx.rollout(p, actor_policy(cx, actor), MUSASHI_CFG)[0]) for p in ("TRAIN", "INNER_A", "INNER_B", "OBS_2024")}
    res["wall_seconds"] = time.time() - t0
    (out / "E2T.json").write_text(json.dumps(jsonable(res), indent=1))
    print(json.dumps({"done": True, "wall": round(res["wall_seconds"], 1)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--e1-run", required=True, help="E1-T run dir (frozen round-3 policies as reference rows)")
    ap.add_argument("--features", choices=["v4", "core15"], required=True)
    ap.add_argument("--v4-dir", default="")
    ap.add_argument("--units", type=int, required=True)
    ap.add_argument("--skip-2024", action="store_true", help="capacity sweep: never read 2024")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=100_000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--measure", action="store_true", help="pilot: one seed, SAC_DEMO only, no INNER_B / 2024 read")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
