#!/usr/bin/env python3
"""E6 stack: four declared improvements on top of E5-B, run sequentially and resumably (owner order 2026-10-21).

Each stage reads the frozen E5-B members (Bayesian + bootstrap ensemble over the 32 candidate orders) and the frozen
E3-M FULL specialists; nothing is re-fitted unless the stage says so. Choices on INNER_A (2023 S1); INNER_B reported;
2024 read ONCE at the end for the stage chosen on INNER_A. DEVELOPMENT; no orders.

  A conformal   split-conformal band on the ensemble: on INNER_A, s_i = |real_i - mean_i| / std_i for the chosen
                candidates; q = the 0.8-quantile of s (declared 80 % lower-bound coverage); decision = trade when
                mean - q * std - margin > 0 (replaces the hand-tuned kappa) with P(NOP) < p_nop_max.
  B closers     the E3-M FULL closers (close-L / close-S: early exit when the stop would be hit before the target)
                applied to the open order of the best stage so far; SL/TP always stay in force.
  C defer       a 'wait' candidate: 3 extra members (same net) trained on the NEXT tick's best candidate result; the
                tick is skipped (HOLD) when the predicted next-best mean exceeds the current best utility + hysteresis.
  D zigzag arbiter  the decision imitated from the hindsight expert over the ORDER space: a 33-class policy
                (32 candidates + NOP) fitted by BC + 2 DAgger rounds on the states the policy visits; labels = the
                hindsight best candidate (NOP when none pays > 0.1 sigma); compared against the utility decision.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from tools.expert_guided_e0 import jsonable, ledger  # noqa: E402
from tools.expert_guided_e1t import STUDENT_CFG, Ctx, summary  # noqa: E402
from tools.expert_guided_e3m import MODULES, Composed, collect  # noqa: E402
from tools.expert_guided_e4q import CANDS, candidate_results  # noqa: E402
from tools.expert_guided_e5b import NOP, NOP_MARGIN, action_target, build_model, fit_member, predict  # noqa: E402

CONF_COVERAGE = 0.8
PERIODS3 = ("TRAIN", "INNER_A", "INNER_B")


def setup(a):
    import torch
    from app.short_horizon.expert_guided import temporal as T
    ds, scaler = DG.load_with_scaler(Path(a.e3m_run), a.admitted, a.csv_dir)     # policy from the E3-M run's SCALER.json
    DG.verify_scaler(DG.read_scaler(Path(a.e5b_run)), ds)                        # the E5B members must share it
    cx = Ctx(ds)
    cx.o, cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
    v = ds.bars.index.as_unit("ns").asi8
    seg = ds.cal.assign(ds.bars.index)
    ff = np.array([s.force_flat.value for s in ds.cal.segments], np.int64)
    cx.flat_bar = np.array([int(np.searchsorted(v + DG.H_NS, ff[int(seg[b])], side="right")) - 1 if seg[b] >= 0 else len(v) for b in range(len(v))])
    e3 = json.loads((Path(a.e3m_run) / "E3M.json").read_text())
    arch = e3["arch"]
    nf, NS = ds.windows.shape[2], len(E.STATE_NAMES)
    hold = lambda *x: (E.HOLD, 1.5, 2.0)  # noqa: E731
    rows, X, Y, fin = {}, {}, {}, {}
    ends = {p: int(np.searchsorted(v, DG._utc_ns(DG.PERIODS[p][1]))) for p in DG.PERIODS}
    for p in DG.PERIODS:
        vis = [vv for vv in collect(cx, p, hold, STUDENT_CFG)[1] if vv[2][-1] > 0.5]
        rows[p] = np.array([vv[0] for vv in vis], int)
        X[p] = np.stack([np.r_[cx.Xw[vv[0]], vv[2]] for vv in vis]).astype(np.float32)
        Y[p], fin[p] = candidate_results(cx, rows[p], ends[p])
    e5 = json.loads((Path(a.e5b_run) / "E5B.json").read_text())
    members = []
    for m in e5["members"]:
        net = build_model(T, nf, NS, arch, a.dropout, 1)
        net.load_state_dict(torch.load(Path(a.e5b_run) / f"member_{m}.pt"))
        members.append(net.to(a.device))
    closers = {}
    for seed, rec in e3["trials"]["FULL"].items():
        mods = {}
        for mname, (nc, nr) in MODULES["FULL"].items():
            net = T.TemporalPolicyNet(nf, DG.WIN, nc, n_reg=nr, n_state=NS, **arch)
            net.load_state_dict(torch.load(Path(a.e3m_run) / f"FULL_{mname}_{seed}.pt"))
            mods[mname] = T.TorchPolicy(net)
        closers[seed] = Composed("FULL", mods, "prob", rec["arbiter"].get("tau"), mask=E.state_mask(e3.get("agent_state", {}).get("mode", "full")))
    return ds, cx, arch, nf, NS, rows, X, Y, fin, e5, members, closers, T


def parse_setting(s):
    d = dict(kv.split("=") for kv in s.split("|"))
    return float(d["kappa"]), float(d["pnop"]), float(d["margin"]), d["size"]


def make_policy(cx, rows_p, j, size_arr, closer=None, defer_mask=None):
    row_pos = {int(r): i for i, r in enumerate(rows_p)}

    def f(bar, r, st, env):
        pos = int(st[0])
        if pos != 0:
            if closer is not None:
                x = np.r_[cx.Xw[r], st].astype(np.float32)
                if closer.exit_now(x, pos):
                    return E.CLOSE, 1.5, 2.0
            return E.HOLD, 1.5, 2.0
        if r not in row_pos:
            return E.HOLD, 1.5, 2.0
        i = row_pos[r]
        k = j[i]
        if k < 0 or (defer_mask is not None and defer_mask[i]):
            return E.HOLD, 1.5, 2.0
        if size_arr is not None:
            env.cfg = dataclasses.replace(env.cfg, percentual_order_size=float(size_arr[i]))
        d, tp, sl = CANDS[k]
        return (E.LONG if d > 0 else E.SHORT), tp, sl
    return f


def run(a):
    import torch
    from app.short_horizon.expert_guided import learn as L
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    led = out / "TRIAL_LEDGER.jsonl"
    a.device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    t0 = time.time()
    res_p = out / "E6.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else {"stages": {}}
    ds, cx, arch, nf, NS, rows, X, Y, fin, e5, members, closers, T = setup(a)
    DG.write_scaler(out, ds, rung="E6")           # next to the c_defer / d_arbiter checkpoints this stage writes
    kappa0, pnop0, margin0, size0 = parse_setting(e5["chosen"])
    pred = {p: predict(members, X[p], a.device, mc=(5 if a.smoke else 30)) for p in DG.PERIODS}
    std_ref = float(np.nanquantile(pred["TRAIN"][1].min(1), 0.9))

    def sizes(p, j, s_):
        return None if size0 == "fixed" else np.clip(1 - s_ / max(std_ref, 1e-6), 0.1, 1.0)

    def eval_policy(name, pol_fn, periods=PERIODS3):
        return {p: summary(cx.rollout(p, pol_fn(p), E.EnvConfig(level_policy="clip"))[0]) for p in periods}

    def save():
        res_p.write_text(json.dumps(jsonable(res), indent=1))

    # ---- reference: E5-B chosen setting re-evaluated here (identity check against E5B.json)
    def base_decision(p, kappa, pnop, margin):
        cm, cs, pm, _ = pred[p]
        util = cm - kappa * cs - margin
        j = util.argmax(1)
        u = util[np.arange(len(j)), j]
        s_ = cs[np.arange(len(j)), j]
        ok = (u > 0) & (pm[:, NOP] < pnop)
        return np.where(ok, j, -1), u, s_
    if "E5B_reference" not in res["stages"]:
        res["stages"]["E5B_reference"] = eval_policy("ref", lambda p: make_policy(cx, rows[p], *((lambda j, u, s_: (j, sizes(p, j, s_)))(*base_decision(p, kappa0, pnop0, margin0)))))
        save()
    # ---- A conformal
    if "A_conformal" not in res["stages"]:
        cm, cs, pm, _ = pred["INNER_A"]
        j, u, s_ = base_decision("INNER_A", 0.0, pnop0, 0.0)
        ok = fin["INNER_A"] & (j >= 0)
        real = Y["INNER_A"][np.arange(len(j)), np.maximum(j, 0)]
        sc = np.abs(real[ok] - cm[np.arange(len(j)), np.maximum(j, 0)][ok]) / np.maximum(s_[ok], 1e-6)
        q = float(np.quantile(sc, CONF_COVERAGE)) if ok.sum() > 50 else 1.0
        best, cand = None, {}
        for margin in (0.0, 0.05):
            for pn in (1.0, 0.7, 0.5):
                key = f"q={q:.2f}|pnop={pn}|margin={margin}"
                r_ = eval_policy(key, lambda p, pn=pn, margin=margin: make_policy(cx, rows[p], *((lambda j, u, s_: (j, sizes(p, j, s_)))(*base_decision(p, q, pn, margin)))))
                cand[key] = r_
                ledger(led, rung="E6A", event="TRIAL", setting=key, inner_a=r_["INNER_A"]["sharpe_daily_sqrt365"], valid=r_["INNER_A"]["valid"])
        ok_c = {k: v for k, v in cand.items() if v["INNER_A"]["valid"]}
        chosen = max(ok_c or cand, key=lambda k: cand[k]["INNER_A"]["sharpe_daily_sqrt365"] or -1e9)
        res["stages"]["A_conformal"] = {"q": q, "coverage": CONF_COVERAGE, "calibration_rows": int(ok.sum()), "trials": cand, "chosen": chosen, **cand[chosen]}
        save()
        print(json.dumps({"stage": "A", "q": round(q, 3), "chosen": chosen, "s": round(time.time() - t0)}), flush=True)
    qA = res["stages"]["A_conformal"]["q"]
    qA_kappa, qA_pn, qA_margin = qA, float(res["stages"]["A_conformal"]["chosen"].split("|")[1].split("=")[1]), float(res["stages"]["A_conformal"]["chosen"].split("|")[2].split("=")[1])

    def best_so_far():
        st = res["stages"]
        keys = [k for k in st if k != "E5B_reference" and "INNER_A" in st[k]]
        cand = {k: st[k]["INNER_A"]["sharpe_daily_sqrt365"] for k in ["E5B_reference"] + keys if st[k]["INNER_A"].get("valid")}
        return max(cand, key=cand.get) if cand else "E5B_reference"

    def decision_of(stage, p):
        if stage == "A_conformal":
            return base_decision(p, qA_kappa, qA_pn, qA_margin)
        return base_decision(p, kappa0, pnop0, margin0)
    # ---- B closers on the best stage so far
    if "B_closers" not in res["stages"]:
        base = best_so_far()
        trials = {}
        for seed, comp in closers.items():
            trials[seed] = eval_policy(f"closer{seed}", lambda p, comp=comp: make_policy(cx, rows[p], *((lambda j, u, s_: (j, sizes(p, j, s_)))(*decision_of(base, p))), closer=comp))
        mean = {p: {k: float(np.mean([trials[s][p][k] for s in trials])) for k in ("sharpe_daily_sqrt365", "net_profit_usd", "max_drawdown_pct", "orders_per_week")}
                for p in PERIODS3}
        for p in PERIODS3:
            mean[p]["valid"] = all(trials[s][p]["valid"] for s in trials)
        res["stages"]["B_closers"] = {"base_stage": base, "per_seed": trials, **mean}
        save()
        print(json.dumps({"stage": "B", "base": base, "inner_a": round(mean["INNER_A"]["sharpe_daily_sqrt365"], 3), "s": round(time.time() - t0)}), flush=True)
    # ---- C defer candidate
    if "C_defer" not in res["stages"]:
        nxt = {}
        for p in DG.PERIODS:
            yb = np.nanmax(np.nan_to_num(Y[p], nan=-9.0), 1)
            nxt[p] = np.r_[yb[1:], np.nan].astype(np.float32)          # next tick's best candidate result
            nxt[p][~np.r_[fin[p][1:], False]] = np.nan
        trm = fin["TRAIN"] & np.isfinite(nxt["TRAIN"])
        iam = fin["INNER_A"] & np.isfinite(nxt["INNER_A"])
        dmodels = []
        for mbr in range(1, (2 if a.smoke else 3) + 1):
            Yt = np.repeat(nxt["TRAIN"][trm][:, None], len(CANDS), 1)   # same target in every candidate slot (declared)
            Yi = np.repeat(nxt["INNER_A"][iam][:, None], len(CANDS), 1)
            m, info = fit_member(X["TRAIN"][trm], Yt, action_target(np.nan_to_num(Y["TRAIN"][trm], nan=-9.0)), X["INNER_A"][iam], Yi,
                                 action_target(np.nan_to_num(Y["INNER_A"][iam], nan=-9.0)), member=100 + mbr, T=T, L=L, nf=nf, NS=NS, arch=arch, a=a, device=a.device)
            dmodels.append(m)
            torch.save(m.state_dict(), out / f"c_defer_member_{mbr}.pt")
        base = best_so_far()
        trials = {}
        for hyst in (0.0, 0.05, 0.1):
            def pol(p, hyst=hyst):
                j, u, s_ = decision_of(base, p)
                nm = predict(dmodels, X[p], a.device, mc=(5 if a.smoke else 30))[0].mean(1)
                defer = (j >= 0) & (nm > u + hyst)
                return make_policy(cx, rows[p], j, sizes(p, j, s_), defer_mask=defer)
            trials[f"hyst={hyst}"] = eval_policy(f"defer{hyst}", pol)
        ok_c = {k: v for k, v in trials.items() if v["INNER_A"]["valid"]}
        chosen = max(ok_c or trials, key=lambda k: trials[k]["INNER_A"]["sharpe_daily_sqrt365"] or -1e9)
        res["stages"]["C_defer"] = {"base_stage": base, "trials": trials, "chosen": chosen, **trials[chosen]}
        save()
        print(json.dumps({"stage": "C", "chosen": chosen, "s": round(time.time() - t0)}), flush=True)
    # ---- D zigzag-imitated arbiter over the order space (33-class BC + DAgger)
    if "D_zigzag_arbiter" not in res["stages"]:
        from torch import nn

        def labels(p, idx):
            yy = np.nan_to_num(Y[p][idx], nan=-9.0)
            best = yy.argmax(1)
            val = yy[np.arange(len(idx)), best]
            return np.where(val > NOP_MARGIN, best + 1, 0)
        n_cls = len(CANDS) + 1
        results = {}
        for seed in (20261019,) if a.smoke else (20261019, 20261020, 20261021):
            Dtr = (X["TRAIN"][fin["TRAIN"]], labels("TRAIN", np.flatnonzero(fin["TRAIN"])))
            Din = (X["INNER_A"][fin["INNER_A"]], labels("INNER_A", np.flatnonzero(fin["INNER_A"])))
            pol = None
            for rnd in range(3):
                if rnd > 0:                                           # DAgger: relabel the flat states the policy visits
                    for p, D in (("TRAIN", "Dtr"), ("INNER_A", "Din")):
                        vis = [vv for vv in collect(cx, p, pol_fn(p), STUDENT_CFG)[1] if vv[2][-1] > 0.5 and vv[6] is None]
                        rr = np.array([vv[0] for vv in vis], int)
                        pos = {int(r): i for i, r in enumerate(rows[p])}
                        keep = np.array([r in pos and fin[p][pos[r]] for r in rr], bool)
                        if keep.sum():
                            Xn = np.stack([np.r_[cx.Xw[vv[0]], vv[2]] for vv, k in zip(vis, keep) if k]).astype(np.float32)
                            yn = labels(p, np.array([pos[int(r)] for r in rr[keep]]))
                            if D == "Dtr":
                                Dtr = (np.concatenate([Dtr[0], Xn]), np.concatenate([Dtr[1], yn]))
                            else:
                                Din = (np.concatenate([Din[0], Xn]), np.concatenate([Din[1], yn]))
                f = np.maximum(np.bincount(Dtr[1], minlength=n_cls) / len(Dtr[1]), 1e-3)
                w = 1 / np.sqrt(f)
                w = w / w[Dtr[1]].mean()
                model, hist, info = L.fit(Dtr[0], Dtr[1], Din[0], Din[1], seed=seed, n_cls=n_cls, class_weight=w, dropout=a.dropout, lr=a.lr,
                                          wd=a.wd, batch=256, max_epochs=a.max_epochs, patience=10, device=a.device,
                                          net=lambda: T.TemporalPolicyNet(nf, DG.WIN, n_cls, n_state=NS, dropout=a.dropout, **arch))
                tp_ = T.TorchPolicy(model)

                def pol_fn(p, tp_=tp_):
                    def f_(bar, r, st, env):
                        if int(st[0]) != 0:
                            return E.HOLD, 1.5, 2.0
                        lg, _ = tp_(np.r_[cx.Xw[r], st].astype(np.float32)[None, :])
                        k = int(np.argmax(lg[0]))
                        if k == 0:
                            return E.HOLD, 1.5, 2.0
                        d, tp, sl = CANDS[k - 1]
                        return (E.LONG if d > 0 else E.SHORT), tp, sl
                    return f_
                pol = tp_
            results[str(seed)] = eval_policy(f"zz{seed}", pol_fn)
            results[str(seed)]["fit"] = {k: info[k] for k in ("epochs", "best_epoch", "params")}
            torch.save(pol.m.state_dict(), out / f"d_arbiter_{seed}.pt")
            print(json.dumps({"stage": "D", "seed": seed, "inner_a": round(results[str(seed)]["INNER_A"]["sharpe_daily_sqrt365"], 3), "s": round(time.time() - t0)}), flush=True)
        mean = {p: {k: float(np.mean([results[s][p][k] for s in results])) for k in ("sharpe_daily_sqrt365", "net_profit_usd", "max_drawdown_pct", "orders_per_week")} for p in PERIODS3}
        for p in PERIODS3:
            mean[p]["valid"] = all(results[s][p]["valid"] for s in results)
        res["stages"]["D_zigzag_arbiter"] = {"per_seed": results, **mean}
        save()
    # ---- decision across stages on INNER_A, 2024 once
    st = res["stages"]
    cand = {k: st[k]["INNER_A"]["sharpe_daily_sqrt365"] for k in st if st[k]["INNER_A"].get("valid")}
    chosen = max(cand, key=cand.get) if cand else "E5B_reference"
    res["chosen_stage"] = chosen
    if "read_2024" not in res:
        ledger(led, rung="E6", event="READ", periods=["OBS_2024"], purpose=f"single read: chosen stage {chosen}")
        if chosen in ("E5B_reference", "A_conformal"):
            r24 = summary(cx.rollout("OBS_2024", make_policy(cx, rows["OBS_2024"], *((lambda j, u, s_: (j, sizes("OBS_2024", j, s_)))(*decision_of(chosen, "OBS_2024")))), E.EnvConfig(level_policy="clip"))[0])
        elif chosen == "B_closers":
            base = st["B_closers"]["base_stage"]
            r24 = {s: summary(cx.rollout("OBS_2024", make_policy(cx, rows["OBS_2024"], *((lambda j, u, s_: (j, sizes("OBS_2024", j, s_)))(*decision_of(base, "OBS_2024"))), closer=comp), E.EnvConfig(level_policy="clip"))[0]) for s, comp in closers.items()}
        elif chosen == "C_defer":
            base = st["C_defer"]["base_stage"]
            hyst = float(st["C_defer"]["chosen"].split("=")[1])
            dm = []
            for f_ in sorted(out.glob("c_defer_member_*.pt")):
                net = build_model(T, nf, NS, arch, a.dropout, 1)
                net.load_state_dict(torch.load(f_))
                dm.append(net.to(a.device))
            j, u, s_ = decision_of(base, "OBS_2024")
            nm = predict(dm, X["OBS_2024"], a.device, mc=(5 if a.smoke else 30))[0].mean(1)
            r24 = summary(cx.rollout("OBS_2024", make_policy(cx, rows["OBS_2024"], j, sizes("OBS_2024", j, s_), defer_mask=(j >= 0) & (nm > u + hyst)), E.EnvConfig(level_policy="clip"))[0])
        else:                                                                  # D: the saved 33-class arbiters
            r24 = {}
            for f_ in sorted(out.glob("d_arbiter_*.pt")):
                net = T.TemporalPolicyNet(nf, DG.WIN, len(CANDS) + 1, n_state=NS, dropout=a.dropout, **arch)
                net.load_state_dict(torch.load(f_))
                tp_ = T.TorchPolicy(net)

                def f24(bar, r, st_, env, tp_=tp_):
                    if int(st_[0]) != 0:
                        return E.HOLD, 1.5, 2.0
                    lg, _ = tp_(np.r_[cx.Xw[r], st_].astype(np.float32)[None, :])
                    k = int(np.argmax(lg[0]))
                    if k == 0:
                        return E.HOLD, 1.5, 2.0
                    d, tp, sl = CANDS[k - 1]
                    return (E.LONG if d > 0 else E.SHORT), tp, sl
                r24[f_.stem] = summary(cx.rollout("OBS_2024", f24, E.EnvConfig(level_policy="clip"))[0])
        res["read_2024"] = {"stage": chosen, "OBS_2024": r24}
        save()
    res["wall_seconds"] = time.time() - t0
    save()
    Lm = ["# E6 stack over E5-B (DEVELOPMENT)", "", f"Chosen on VAL A: **{chosen}**; 2024 read once for it.", "",
          "| stage | Sharpe TRAIN | VAL A | VAL B | DD % VAL B | orders/wk VAL A | valid VAL A |", "|---|---|---|---|---|---|---|"]
    for k, v in st.items():
        Lm.append(f"| {k} | {v['TRAIN']['sharpe_daily_sqrt365']:.2f} | {v['INNER_A']['sharpe_daily_sqrt365']:.2f} | {v['INNER_B']['sharpe_daily_sqrt365']:.2f} | "
                  f"{v['INNER_B']['max_drawdown_pct']:.1f} | {v['INNER_A']['orders_per_week']:.1f} | {v['INNER_A']['valid']} |")
    Lm += ["", f"2024: {json.dumps(jsonable(res['read_2024']), default=str)[:800]}"]
    (out / "E6.md").write_text("\n".join(Lm) + "\n")
    print(json.dumps({"chosen_stage": chosen, "wall": round(res["wall_seconds"])}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--e3m-run", required=True)
    ap.add_argument("--e5b-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
