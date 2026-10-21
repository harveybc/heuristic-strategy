#!/usr/bin/env python3
"""E4-Q: decide the ORDER, not the direction (owner framing 2026-10-21). DEVELOPMENT.

At every eligible tick the strategy evaluates every candidate order — long and short, each with a grid of TP / SL
multiples of the causal sigma — predicts the DISTRIBUTION of each candidate's net result, and executes the one with
the best predicted profit/risk, or nothing. This is the predictor-era rule (best predicted order among buy and sell at
each point) rebuilt on the specialists' trunk, and it is strictly more general than a direction label: a candidate
can win through asymmetric levels, timing or regime even when "up vs down" is 50/50.

  Candidates   side in {long, short} x TP in {1, 1.5, 2, 3} sigma x SL in {1, 1.5, 2, 3} sigma  = 32 (+ none)
  Labels       for every eligible origin t and every candidate: realized net result in sigma units of the order filled
               at open(t+1): +TP - cost if TP first, -SL - cost if SL first (same bar -> SL first), close-at-end
               result - cost at the forced flat or after 72 bars. Hindsight; used only as labels; a row enters
               fitting only when ALL its candidates are resolved before its period's cut.
  Model        the FULL specialists' temporal trunk (same arch) on [13 x 15 window, flat agent state] -> 32 x 3
               quantiles (0.1, 0.5, 0.9) of the net result, pinball loss; 3 seeds; early stopping on the mean of the
               TRAIN and INNER_A losses (patience 10); AdamW wd 1e-4, dropout 0.1, no BatchNorm.
  Decision     ensemble over seeds; utility(c) = q50 - lambda * (q50 - q10) - margin; abstain when the seeds'
               q50 spread > s_max; execute argmax if utility > 0; SL/TP are the candidate's own; the simulator applies
               every business rule (integer units, costs, forced flat, <= 4/day; >= 3/week validity).
               (lambda, s_max, margin) chosen on INNER_A among valid settings; INNER_B reported; 2024 read ONCE.
  Also reported: the unconditional mean result of every candidate (does ANY fixed order pay after costs?) and the
  realized result of the model's top-decile choices (does conditioning on the state create an edge?).
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

SEEDS = (20261019, 20261020, 20261021)
TPS, SLS = (1.0, 1.5, 2.0, 3.0), (1.0, 1.5, 2.0, 3.0)
CANDS = [(d, tp, sl) for d in (1, -1) for tp in TPS for sl in SLS]
QS = (0.1, 0.5, 0.9)
HORIZON = 72
COST_SIGMA = 0.06          # declared round-trip cost for the LABELS (~2.3 pips at sigma 38 pips); the simulator charges real costs
GRID = {"lam": (0.0, 0.5, 1.0), "s_max": (math.inf, 0.3, 0.15), "margin": (0.0, 0.05)}


def candidate_results(cx, rows, period_end_bar):
    """(Y [n, 32] net result in sigma, final [n]) for the given observation rows; final = every candidate resolved
    before period_end_bar (purge at the cut; censored beyond the data)."""
    n_b = len(cx.c)
    o, h, l, c = cx.o, cx.h, cx.l, cx.c
    Y = np.full((len(rows), len(CANDS)), np.nan)
    final = np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        b = int(cx.ds.row_bar[r])
        s = float(cx.ds.sigma[b + 1]) if b + 1 < n_b else np.nan
        if not np.isfinite(s) or s <= 0 or b + 1 >= n_b:
            continue
        ff = int(cx.flat_bar[b])
        end = min(b + HORIZON, ff)
        if end < b + 1:
            continue
        k0, k1 = b + 1, min(end, n_b - 1)
        fill = o[k0]
        hh, ll = h[k0:k1 + 1], l[k0:k1 + 1]
        up_max, dn_min = np.maximum.accumulate(hh), np.minimum.accumulate(ll)
        resolved_all = end < n_b
        last = np.nan if not resolved_all else c[k1]
        res_bar = k0
        for j, (d, tp, sl) in enumerate(CANDS):
            if d > 0:
                kt = int(np.argmax(up_max >= fill + tp * s)) if (up_max >= fill + tp * s).any() else -1
                ks = int(np.argmax(dn_min <= fill - sl * s)) if (dn_min <= fill - sl * s).any() else -1
            else:
                kt = int(np.argmax(dn_min <= fill - tp * s)) if (dn_min <= fill - tp * s).any() else -1
                ks = int(np.argmax(up_max >= fill + sl * s)) if (up_max >= fill + sl * s).any() else -1
            if ks >= 0 and (kt < 0 or ks <= kt):
                Y[i, j], rb = -sl - COST_SIGMA, k0 + ks
            elif kt >= 0:
                Y[i, j], rb = tp - COST_SIGMA, k0 + kt
            elif resolved_all:
                Y[i, j], rb = d * (last - fill) / s - COST_SIGMA, k1
            else:
                rb = n_b
            res_bar = max(res_bar, rb)
        final[i] = np.isfinite(Y[i]).all() and res_bar < period_end_bar
    return Y, final


class QNet:
    pass


def build_model(T, nf, NS, arch, dropout):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = T.TemporalPolicyNet(nf, DG.WIN, 1, n_state=NS, dropout=dropout, **arch)
            d = self.base.cls.in_features
            self.base.cls = nn.Identity()
            self.out = nn.Linear(d, len(CANDS) * len(QS))

        def forward(self, x):
            z, _ = self.base(x)
            return self.out(z).view(-1, len(CANDS), len(QS))
    return Net()


def pinball(pred, y):
    import torch
    q = torch.tensor(QS, device=pred.device).view(1, 1, -1)
    e = y.unsqueeze(-1) - pred
    return torch.mean(torch.maximum(q * e, (q - 1) * e))


def fit_q(X, Y, Xi, Yi, *, seed, T, L, nf, NS, arch, a, device):
    import copy
    import torch
    L.set_seed(seed)
    model = build_model(T, nf, NS, arch, a.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    Xt, Yt = torch.as_tensor(X, device=device), torch.as_tensor(Y, device=device)
    Xv, Yv = torch.as_tensor(Xi, device=device), torch.as_tensor(Yi, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, best_ep, wait, hist = math.inf, None, 0, 0, []
    for ep in range(1, a.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X), generator=g).to(device)
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            loss = pinball(model(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr = float(pinball(model(Xt), Yt))
            va = float(pinball(model(Xv), Yv))
        score = 0.5 * (tr + va)
        hist.append({"epoch": ep, "train": tr, "inner": va})
        if score < best - 1e-8:
            best, best_state, best_ep, wait = score, copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= 10:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs": len(hist), "best_epoch": best_ep, "best_score": best, "history": hist,
                   "params": int(sum(p.numel() for p in model.parameters()))}


def predict(models, X, device):
    import torch
    out = []
    with torch.no_grad():
        for m in models:
            P = []
            for i in range(0, len(X), 4096):
                P.append(m(torch.as_tensor(X[i:i + 4096], device=device)).cpu().numpy())
            out.append(np.concatenate(P))
    return np.stack(out)                       # seeds x n x 32 x 3


def decide(Q, lam, s_max, margin):
    """Q: seeds x n x 32 x 3 -> (chosen candidate index or -1, utility) per row."""
    mean = Q.mean(0)
    spread = Q[:, :, :, 1].std(0)
    util = mean[:, :, 1] - lam * (mean[:, :, 1] - mean[:, :, 0]) - margin
    util = np.where(spread <= s_max, util, -np.inf)
    j = util.argmax(1)
    u = util[np.arange(len(j)), j]
    return np.where(u > 0, j, -1), u


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
    cx.o, cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
    # forced-flat bar per bar from the env's own calendar (the segment's force_flat instant)
    v = ds.bars.index.as_unit("ns").asi8
    seg = ds.cal.assign(ds.bars.index)
    ff = np.array([s.force_flat.value for s in ds.cal.segments], np.int64)
    cx.flat_bar = np.array([int(np.searchsorted(v + DG.H_NS, ff[int(seg[b])], side="right")) - 1 if seg[b] >= 0 else len(v) for b in range(len(v))])
    e3 = json.loads((Path(a.e3m_run) / "E3M.json").read_text())
    arch = e3["arch"]
    nf, NS = ds.windows.shape[2], len(E.STATE_NAMES)
    # flat agent state at every observable bar: a HOLD-only rollout per period
    hold = lambda *x: (E.HOLD, 1.5, 2.0)  # noqa: E731
    rows, X = {}, {}
    for p in DG.PERIODS:
        vis = collect(cx, p, hold, STUDENT_CFG)[1]
        vis = [vv for vv in vis if vv[2][-1] > 0.5]                     # entry gate open (flat, inside the session)
        rows[p] = np.array([vv[0] for vv in vis], int)
        X[p] = np.stack([np.r_[cx.Xw[vv[0]], vv[2]] for vv in vis]).astype(np.float32) if len(vis) else np.zeros((0, cx.Xw.shape[1] + NS), np.float32)
    ends = {p: int(np.searchsorted(v, DG._utc_ns(DG.PERIODS[p][1]))) for p in DG.PERIODS}
    Y, fin = {}, {}
    for p in DG.PERIODS:
        Y[p], fin[p] = candidate_results(cx, rows[p], ends[p])
    ledger(led, rung="E4Q", event="PREREGISTRATION", candidates=CANDS, quantiles=QS, cost_sigma=COST_SIGMA, grid=GRID, seeds=SEEDS,
           arch=arch, rows={p: int(fin[p].sum()) for p in DG.PERIODS}, device=device,
           choice="(lambda, s_max, margin) by INNER_A Sharpe among valid; INNER_B reported; 2024 once")
    res = {"candidates": CANDS, "cost_sigma": COST_SIGMA, "arch": arch, "rows_final": {p: int(fin[p].sum()) for p in DG.PERIODS},
           "unconditional_mean_result_sigma": {p: {f"{'L' if d > 0 else 'S'}_tp{tp}_sl{sl}": float(np.nanmean(Y[p][fin[p], j]))
                                                   for j, (d, tp, sl) in enumerate(CANDS)} for p in ("TRAIN", "INNER_A", "INNER_B")}}
    tr, ia = fin["TRAIN"], fin["INNER_A"]
    models, infos = [], []
    for seed in SEEDS:
        ledger(led, rung="E4Q", event="FIT", seed=seed, rows_train=int(tr.sum()), rows_inner=int(ia.sum()))
        m, info = fit_q(X["TRAIN"][tr], Y["TRAIN"][tr].astype(np.float32), X["INNER_A"][ia], Y["INNER_A"][ia].astype(np.float32),
                        seed=seed, T=T, L=L, nf=nf, NS=NS, arch=arch, a=a, device=device)
        models.append(m)
        infos.append({k: info[k] for k in ("epochs", "best_epoch", "best_score", "params")})
        torch.save(m.state_dict(), out / f"q_{seed}.pt")
        print(json.dumps({"seed": seed, **infos[-1], "s": round(time.time() - t0)}), flush=True)
    res["fits"] = infos
    Q = {p: predict(models, X[p], device) for p in DG.PERIODS}
    # conditional edge diagnostic: realized result of the top-decile utility choices (lambda 0.5)
    diag = {}
    for p in ("TRAIN", "INNER_A", "INNER_B"):
        j, u = decide(Q[p], 0.5, math.inf, 0.0)
        ok = fin[p] & (j >= 0)
        real = Y[p][np.arange(len(j)), np.maximum(j, 0)]
        top = ok & (u >= np.quantile(u[ok], 0.9)) if ok.sum() > 20 else ok
        diag[p] = {"rows_chosen": int(ok.sum()), "mean_realized_sigma_chosen": float(np.mean(real[ok])) if ok.any() else None,
                   "mean_realized_sigma_top_decile": float(np.mean(real[top])) if top.any() else None,
                   "share_long": float(np.mean(np.array([CANDS[k][0] > 0 for k in j[ok]]))) if ok.any() else None}
    res["conditional_edge"] = diag
    # strategy in the executable simulator over the grid; choice on INNER_A
    row_pos = {p: {int(r): i for i, r in enumerate(rows[p])} for p in DG.PERIODS}

    def policy(p, lam, s_max, margin):
        j, u = decide(Q[p], lam, s_max, margin)

        def f(bar, r, st, env):
            if int(st[0]) != 0 or r not in row_pos[p]:
                return E.HOLD, 1.5, 2.0
            k = j[row_pos[p][r]]
            if k < 0:
                return E.HOLD, 1.5, 2.0
            d, tp, sl = CANDS[k]
            return (E.LONG if d > 0 else E.SHORT), tp, sl
        return f
    trials = {}
    for lam, s_max, margin in itertools.product(*GRID.values()):
        key = f"lam={lam}|smax={s_max}|margin={margin}"
        trials[key] = {p: summary(cx.rollout(p, policy(p, lam, s_max, margin), STUDENT_CFG)[0]) for p in ("TRAIN", "INNER_A", "INNER_B")}
        ledger(led, rung="E4Q", event="GRID_TRIAL", setting=key, inner_a=trials[key]["INNER_A"]["sharpe_daily_sqrt365"], valid=trials[key]["INNER_A"]["valid"])
        print(json.dumps({"setting": key, "sharpe": {p: round(trials[key][p]["sharpe_daily_sqrt365"] or 0, 2) for p in trials[key]},
                          "opw_ia": round(trials[key]["INNER_A"]["orders_per_week"], 1), "s": round(time.time() - t0)}), flush=True)
    ok = {k: v for k, v in trials.items() if v["INNER_A"]["valid"]}
    chosen = max(ok or trials, key=lambda k: trials[k]["INNER_A"]["sharpe_daily_sqrt365"] or -1e9)
    res["trials"], res["chosen"] = trials, chosen
    ledger(led, rung="E4Q", event="READ", periods=["OBS_2024"], purpose=f"single read: chosen {chosen}")
    lam, s_max, margin = (float(x.split("=")[1]) for x in chosen.split("|"))
    res["trials"][chosen]["OBS_2024"] = summary(cx.rollout("OBS_2024", policy("OBS_2024", lam, s_max, margin), STUDENT_CFG)[0])
    res["comparators"] = {p: {"no_trade": 0.0, "always_long_2_2": summary(cx.rollout(p, lambda b, r, st, env: (E.LONG if int(st[0]) == 0 else E.HOLD, 2.0, 2.0), STUDENT_CFG)[0])["sharpe_daily_sqrt365"]}
                          for p in ("INNER_A", "INNER_B")}
    res["wall_seconds"] = time.time() - t0
    (out / "E4Q.json").write_text(json.dumps(jsonable(res), indent=1))
    Lm = ["# E4-Q: best candidate order per tick (DEVELOPMENT)", "", f"Chosen on VAL A: **{chosen}**.", "",
          "## Does any FIXED order pay after costs? (mean realized result in sigma, TRAIN / VAL A / VAL B)", "",
          "| candidate | TRAIN | VAL A | VAL B |", "|---|---|---|---|"]
    for k in res["unconditional_mean_result_sigma"]["TRAIN"]:
        Lm.append(f"| {k} | " + " | ".join(f"{res['unconditional_mean_result_sigma'][p][k]:+.3f}" for p in ("TRAIN", "INNER_A", "INNER_B")) + " |")
    Lm += ["", "## Does conditioning on the state create an edge? (model's choices, lambda 0.5)", "", "| period | rows chosen | mean realized (sigma) | top-decile realized | share long |", "|---|---|---|---|---|"]
    for p, d_ in diag.items():
        Lm.append(f"| {p} | {d_['rows_chosen']} | {d_['mean_realized_sigma_chosen']} | {d_['mean_realized_sigma_top_decile']} | {d_['share_long']} |")
    Lm += ["", "## Strategy (executable simulator, full contract)", "", "| setting | Sharpe TRAIN | VAL A | VAL B | 2024 | orders/wk VAL A | valid VAL A |", "|---|---|---|---|---|---|---|"]
    for k, v in sorted(trials.items(), key=lambda kv: -(kv[1]["INNER_A"]["sharpe_daily_sqrt365"] or -1e9)):
        s24 = v.get("OBS_2024", {}).get("sharpe_daily_sqrt365")
        Lm.append(f"| {k}{' **(chosen)**' if k == chosen else ''} | {v['TRAIN']['sharpe_daily_sqrt365']:.2f} | {v['INNER_A']['sharpe_daily_sqrt365']:.2f} | "
                  f"{v['INNER_B']['sharpe_daily_sqrt365']:.2f} | {'—' if s24 is None else f'{s24:.2f}'} | {v['INNER_A']['orders_per_week']:.1f} | {v['INNER_A']['valid']} |")
    (out / "E4Q.md").write_text("\n".join(Lm) + "\n")
    print(json.dumps({"chosen": chosen, "diag": diag, "wall": round(res["wall_seconds"])}))


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
    run(ap.parse_args())


if __name__ == "__main__":
    main()
