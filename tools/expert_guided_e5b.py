#!/usr/bin/env python3
"""E5-B: Bayesian + bootstrap decision ensemble over candidate orders, with an explicit NOP expert (owner 2026-10-21).

What the owner asked for, built on E4-Q's candidate-order labels:
  * a Bayesian output layer (local-reparameterisation Bayesian linear, the DenseFlipout idea of the predictor era)
    plus MC-dropout in the trunk: every prediction is the MEAN of K stochastic passes and its UNCERTAINTY is their std;
  * bootstrap bagging: M members, each fitted on a week-block bootstrap of the TRAIN rows (bootstrap seed = member);
  * a NOP expert from the ZigZag-era hindsight: the 3-class target {NOP, BUY, SELL} = the side of the best candidate
    order at the tick when its hindsight net result exceeds ``NOP_MARGIN`` sigma, else NOP (nothing pays here, or the
    forced flat / the end of a leg is too close for any order to pay);
  * the uncertainty enters the decision (abstain when wide) AND the order size (smaller when uncertain), as the
    short/long prediction uncertainties entered the predictor strategy.

Network: the specialists' trunk (same arch) -> two Bayesian heads: 32 candidate mean net results (sigma) and 3 NOP/BUY/
SELL logits. Loss = Huber(candidates) + CE(action) + KL / N. Decision per tick from the ensemble (M members x K passes):
  utility(c) = mean(c) - kappa * std(c) - margin;  trade iff max utility > 0 and P(NOP) < p_nop_max;
  percentual_order_size = clip(1 - std(c*) / std_ref, 0.1, 1) when the 'uncertainty sizing' arm is on.
(kappa, p_nop_max, margin, sizing) chosen on INNER_A among valid settings; INNER_B reported; 2024 read ONCE.
Figure: predicted utility of the chosen order with its uncertainty band against the realized result (VAL A / VAL B).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from tools.expert_guided_e0 import jsonable, ledger  # noqa: E402
from tools.expert_guided_e1t import STUDENT_CFG, Ctx, summary  # noqa: E402
from tools.expert_guided_e3m import collect  # noqa: E402
from tools.expert_guided_e4q import CANDS, candidate_results  # noqa: E402

MEMBERS = 5
MC = 30
NOP_MARGIN = 0.10
GRID = {"kappa": (0.0, 0.5, 1.0), "p_nop_max": (1.0, 0.7, 0.5), "margin": (0.0, 0.05), "sizing": ("fixed", "uncertainty")}
NOP, BUY, SELL = 0, 1, 2


def action_target(Y):
    """3-class hindsight target from the candidate results: BUY/SELL = side of the best candidate when it pays more
    than NOP_MARGIN sigma; NOP otherwise."""
    best = np.nanargmax(Y, 1)
    val = Y[np.arange(len(Y)), best]
    side = np.array([CANDS[j][0] for j in best])
    return np.where(val > NOP_MARGIN, np.where(side > 0, BUY, SELL), NOP)


def build_model(T, nf, NS, arch, dropout, n_train):
    import torch
    from torch import nn

    class BayesLinear(nn.Module):
        """Mean-field Gaussian weights with local reparameterisation; KL to N(0, 1) scaled by 1/n_train."""

        def __init__(self, d_in, d_out):
            super().__init__()
            self.w_mu = nn.Parameter(torch.randn(d_in, d_out) * 0.05)
            self.w_rho = nn.Parameter(torch.full((d_in, d_out), -5.0))
            self.b_mu = nn.Parameter(torch.zeros(d_out))
            self.b_rho = nn.Parameter(torch.full((d_out,), -5.0))

        def forward(self, x):
            w_sig, b_sig = torch.nn.functional.softplus(self.w_rho), torch.nn.functional.softplus(self.b_rho)
            mu = x @ self.w_mu + self.b_mu
            var = (x ** 2) @ (w_sig ** 2) + b_sig ** 2
            return mu + torch.sqrt(var + 1e-12) * torch.randn_like(mu)

        def kl(self):
            w_sig, b_sig = torch.nn.functional.softplus(self.w_rho), torch.nn.functional.softplus(self.b_rho)
            kl = lambda m, s: 0.5 * torch.sum(s ** 2 + m ** 2 - 1.0 - 2.0 * torch.log(s + 1e-12))  # noqa: E731
            return kl(self.w_mu, w_sig) + kl(self.b_mu, b_sig)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = T.TemporalPolicyNet(nf, DG.WIN, 1, n_state=NS, dropout=dropout, **arch)
            d = self.base.cls.in_features
            self.base.cls = nn.Identity()
            self.cand = BayesLinear(d, len(CANDS))
            self.act = BayesLinear(d, 3)
            self.n_train = n_train

        def forward(self, x):
            z, _ = self.base(x)
            return self.cand(z), self.act(z)

        def kl(self):
            return (self.cand.kl() + self.act.kl()) / max(self.n_train, 1)
    return Net()


def fit_member(X, Y, A, Xi, Yi, Ai, *, member, T, L, nf, NS, arch, a, device):
    import copy
    import torch
    from torch import nn
    rng = np.random.default_rng(member)
    # week-block bootstrap of the TRAIN rows (contiguous blocks of 120 rows ~ one trading week)
    nb = int(math.ceil(len(X) / 120))
    pick = rng.integers(0, nb, nb)
    idx = np.concatenate([np.arange(b * 120, min((b + 1) * 120, len(X))) for b in pick])
    X, Y, A = X[idx], Y[idx], A[idx]
    L.set_seed(20261021 + member)
    model = build_model(T, nf, NS, arch, a.dropout, len(X)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    Xt, Yt, At = (torch.as_tensor(v, device=device) for v in (X, Y, A))
    Xv, Yv, Av = (torch.as_tensor(v, device=device) for v in (Xi, Yi, Ai))
    cw = torch.as_tensor(1 / np.sqrt(np.maximum(np.bincount(A, minlength=3) / len(A), 1e-3)), dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=cw / cw.mean())

    def loss_fn(xb, yb, ab):
        c, l = model(xb)
        return nn.functional.smooth_l1_loss(c, yb) + ce(l, ab) + model.kl()
    g = torch.Generator(device="cpu").manual_seed(member)
    best, best_state, best_ep, wait, hist = math.inf, None, 0, 0, []
    for ep in range(1, a.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X), generator=g).to(device)
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            loss = loss_fn(Xt[b], Yt[b], At[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr = float(np.mean([float(loss_fn(Xt, Yt, At)) for _ in range(3)]))
            va = float(np.mean([float(loss_fn(Xv, Yv, Av)) for _ in range(3)]))
        score = 0.5 * (tr + va)
        hist.append({"epoch": ep, "train": tr, "inner": va})
        if score < best - 1e-8:
            best, best_state, best_ep, wait = score, copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= 10:
                break
    model.load_state_dict(best_state)
    return model, {"member": member, "rows_boot": int(len(X)), "epochs": len(hist), "best_epoch": best_ep, "best_score": best,
                   "params": int(sum(p.numel() for p in model.parameters()))}


def predict(models, X, device, mc=MC):
    """Ensemble (members x MC passes with dropout + weight noise) -> candidate mean/std, action probability mean/std."""
    import torch
    C, P = [], []
    with torch.no_grad():
        for m in models:
            m.train()                                   # MC-dropout active (no BatchNorm anywhere); weights sampled per pass
            for _ in range(mc):
                cs, ps = [], []
                for i in range(0, len(X), 4096):
                    c, l = m(torch.as_tensor(X[i:i + 4096], device=device))
                    cs.append(c.cpu().numpy())
                    ps.append(torch.softmax(l, 1).cpu().numpy())
                C.append(np.concatenate(cs))
                P.append(np.concatenate(ps))
            m.eval()
    C, P = np.stack(C), np.stack(P)
    return C.mean(0), C.std(0), P.mean(0), P.std(0)


def decide(cm, cs, pm, kappa, p_nop_max, margin):
    util = cm - kappa * cs - margin
    j = util.argmax(1)
    u = util[np.arange(len(j)), j]
    ok = (u > 0) & (pm[:, NOP] < p_nop_max)
    return np.where(ok, j, -1), u, cs[np.arange(len(j)), j]


def run(a):
    import dataclasses
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
    cx.o, cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
    v = ds.bars.index.as_unit("ns").asi8
    seg = ds.cal.assign(ds.bars.index)
    ff = np.array([s.force_flat.value for s in ds.cal.segments], np.int64)
    cx.flat_bar = np.array([int(np.searchsorted(v + DG.H_NS, ff[int(seg[b])], side="right")) - 1 if seg[b] >= 0 else len(v) for b in range(len(v))])
    e3 = json.loads((Path(a.e3m_run) / "E3M.json").read_text())
    arch = e3["arch"]
    nf, NS = ds.windows.shape[2], len(E.STATE_NAMES)
    hold = lambda *x: (E.HOLD, 1.5, 2.0)  # noqa: E731
    rows, X, Y, fin, A = {}, {}, {}, {}, {}
    ends = {p: int(np.searchsorted(v, DG._utc_ns(DG.PERIODS[p][1]))) for p in DG.PERIODS}
    for p in DG.PERIODS:
        vis = [vv for vv in collect(cx, p, hold, STUDENT_CFG)[1] if vv[2][-1] > 0.5]
        rows[p] = np.array([vv[0] for vv in vis], int)
        X[p] = np.stack([np.r_[cx.Xw[vv[0]], vv[2]] for vv in vis]).astype(np.float32)
        Y[p], fin[p] = candidate_results(cx, rows[p], ends[p])
        A[p] = np.where(fin[p], action_target(np.nan_to_num(Y[p], nan=-9.0)), NOP)
    members = list(range(1, (2 if a.smoke else MEMBERS) + 1))
    ledger(led, rung="E5B", event="PREREGISTRATION", members=members, mc=MC, nop_margin=NOP_MARGIN, grid=GRID, arch=arch, device=device,
           action_counts_train={int(k): int(c) for k, c in zip(*np.unique(A["TRAIN"][fin["TRAIN"]], return_counts=True))},
           choice="(kappa, p_nop_max, margin, sizing) by INNER_A Sharpe among valid; INNER_B reported; 2024 once")
    tr, ia = fin["TRAIN"], fin["INNER_A"]
    models, infos = [], []
    for mbr in members:
        m, info = fit_member(X["TRAIN"][tr], Y["TRAIN"][tr].astype(np.float32), A["TRAIN"][tr], X["INNER_A"][ia],
                             Y["INNER_A"][ia].astype(np.float32), A["INNER_A"][ia], member=mbr, T=T, L=L, nf=nf, NS=NS, arch=arch, a=a, device=device)
        models.append(m)
        infos.append(info)
        torch.save(m.state_dict(), out / f"member_{mbr}.pt")
        print(json.dumps({**info, "s": round(time.time() - t0)}), flush=True)
    pred = {p: predict(models, X[p], device, mc=(5 if a.smoke else MC)) for p in DG.PERIODS}
    res = {"members": members, "mc": MC, "nop_margin": NOP_MARGIN, "arch": arch, "fits": infos,
           "rows_final": {p: int(fin[p].sum()) for p in DG.PERIODS},
           "action_prior_train": {int(k): int(c) for k, c in zip(*np.unique(A["TRAIN"][tr], return_counts=True))}}
    # NOP expert quality: log-loss vs prior and accuracy on INNER_A / INNER_B
    prior = np.bincount(A["TRAIN"][tr], minlength=3) / tr.sum()
    res["nop_expert"] = {}
    for p in ("INNER_A", "INNER_B"):
        pm = pred[p][2][fin[p]]
        y = A[p][fin[p]]
        ll = float(-np.mean(np.log(np.clip(pm[np.arange(len(y)), y], 1e-6, 1))))
        pl = float(-np.mean(np.log(np.clip(prior[y], 1e-6, 1))))
        res["nop_expert"][p] = {"rows": int(len(y)), "log_loss": ll, "prior_log_loss": pl, "minus_prior": ll - pl,
                                "accuracy": float(np.mean(pm.argmax(1) == y)), "nop_share_true": float(np.mean(y == NOP))}
    # uncertainty diagnostic: realized result by uncertainty tercile of the chosen candidate (kappa 0.5)
    res["uncertainty_vs_realized"] = {}
    for p in ("INNER_A", "INNER_B"):
        cm, cs, pm, _ = pred[p]
        j, u, s_ = decide(cm, cs, pm, 0.5, 1.0, 0.0)
        ok = fin[p] & (j >= 0)
        if ok.sum() > 30:
            real = Y[p][np.arange(len(j)), np.maximum(j, 0)]
            q = np.quantile(s_[ok], [1 / 3, 2 / 3])
            res["uncertainty_vs_realized"][p] = {f"std_tercile_{i}": float(np.mean(real[ok & (np.digitize(s_, q) == i)])) for i in range(3)}
    row_pos = {p: {int(r): i for i, r in enumerate(rows[p])} for p in DG.PERIODS}
    std_ref = {p: float(np.nanquantile(pred[p][1].min(1), 0.9)) for p in DG.PERIODS}

    def policy(p, kappa, p_nop_max, margin, sizing):
        cm, cs, pm, _ = pred[p]
        j, u, s_ = decide(cm, cs, pm, kappa, p_nop_max, margin)

        def f(bar, r, st, env):
            if int(st[0]) != 0 or r not in row_pos[p]:
                return E.HOLD, 1.5, 2.0
            i = row_pos[p][r]
            k = j[i]
            if k < 0:
                return E.HOLD, 1.5, 2.0
            if sizing == "uncertainty":
                env.cfg = dataclasses.replace(env.cfg, percentual_order_size=float(np.clip(1 - s_[i] / max(std_ref["TRAIN"], 1e-6), 0.1, 1.0)))
            d, tp, sl = CANDS[k]
            return (E.LONG if d > 0 else E.SHORT), tp, sl
        return f
    trials = {}
    for kappa, pn, margin, sizing in itertools.product(*GRID.values()):
        key = f"kappa={kappa}|pnop={pn}|margin={margin}|size={sizing}"
        trials[key] = {p: summary(cx.rollout(p, policy(p, kappa, pn, margin, sizing), E.EnvConfig(level_policy="clip"))[0]) for p in ("TRAIN", "INNER_A", "INNER_B")}
        ledger(led, rung="E5B", event="GRID_TRIAL", setting=key, inner_a=trials[key]["INNER_A"]["sharpe_daily_sqrt365"], valid=trials[key]["INNER_A"]["valid"])
    ok = {k: v for k, v in trials.items() if v["INNER_A"]["valid"]}
    chosen = max(ok or trials, key=lambda k: trials[k]["INNER_A"]["sharpe_daily_sqrt365"] or -1e9)
    res["trials"], res["chosen"] = trials, chosen
    ledger(led, rung="E5B", event="READ", periods=["OBS_2024"], purpose=f"single read: chosen {chosen}")
    kappa, pn, margin, sizing = chosen.split("|")
    kappa, pn, margin, sizing = float(kappa.split("=")[1]), float(pn.split("=")[1]), float(margin.split("=")[1]), sizing.split("=")[1]
    res["trials"][chosen]["OBS_2024"] = summary(cx.rollout("OBS_2024", policy("OBS_2024", kappa, pn, margin, sizing), E.EnvConfig(level_policy="clip"))[0])
    res["wall_seconds"] = time.time() - t0
    (out / "E5B.json").write_text(json.dumps(jsonable(res), indent=1))
    # figure: predicted utility with uncertainty band vs realized, chosen setting, VAL A and VAL B
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
        for ax, p in zip(axes, ("INNER_A", "INNER_B")):
            cm, cs, pm, _ = pred[p]
            j, u, s_ = decide(cm, cs, pm, kappa, pn, margin)
            okm = fin[p] & (j >= 0)
            idx = np.flatnonzero(okm)[:400]
            real = Y[p][idx, j[idx]]
            x = np.arange(len(idx))
            ax.fill_between(x, u[idx] - s_[idx], u[idx] + s_[idx], alpha=0.25, label="utilidad predicha ± incertidumbre")
            ax.plot(x, u[idx], lw=1, label="utilidad predicha (media)")
            ax.plot(x, real, ".", ms=3, alpha=0.6, label="resultado real (σ)")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"{p}: órdenes elegidas ({okm.sum()} en el periodo; primeras 400)")
            ax.set_ylabel("σ")
            ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(out / "e5b_incertidumbre.png", dpi=140)
    except Exception as exc:                            # the figure never blocks the result
        res["figure_error"] = str(exc)
    Lm = ["# E5-B: Bayesian + bootstrap decision ensemble with a NOP expert (DEVELOPMENT)", "", f"Chosen on VAL A: **{chosen}**.", "",
          "## NOP / BUY / SELL expert (hindsight best order, margin 0.1 sigma)", "", "| period | rows | log-loss − prior | accuracy | true NOP share |", "|---|---|---|---|---|"]
    for p, d_ in res["nop_expert"].items():
        Lm.append(f"| {p} | {d_['rows']} | {d_['minus_prior']:+.4f} | {d_['accuracy']:.3f} | {d_['nop_share_true']:.3f} |")
    Lm += ["", "## Realized result of the chosen orders by uncertainty tercile (sigma; lower std should be better)", "", f"{json.dumps(res['uncertainty_vs_realized'])}", "",
           "## Strategy (executable simulator, full contract)", "", "| setting | Sharpe TRAIN | VAL A | VAL B | 2024 | orders/wk VAL A | valid VAL A |", "|---|---|---|---|---|---|---|"]
    for k, v_ in sorted(trials.items(), key=lambda kv: -(kv[1]["INNER_A"]["sharpe_daily_sqrt365"] or -1e9))[:20]:
        s24 = v_.get("OBS_2024", {}).get("sharpe_daily_sqrt365")
        Lm.append(f"| {k}{' **(chosen)**' if k == chosen else ''} | {v_['TRAIN']['sharpe_daily_sqrt365']:.2f} | {v_['INNER_A']['sharpe_daily_sqrt365']:.2f} | "
                  f"{v_['INNER_B']['sharpe_daily_sqrt365']:.2f} | {'—' if s24 is None else f'{s24:.2f}'} | {v_['INNER_A']['orders_per_week']:.1f} | {v_['INNER_A']['valid']} |")
    (out / "E5B.md").write_text("\n".join(Lm) + "\n")
    print(json.dumps({"chosen": chosen, "nop_expert": res["nop_expert"], "wall": round(res["wall_seconds"])}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--e3m-run", required=True)
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
