#!/usr/bin/env python3
"""Strategy-free signal audit (pre-registered in docs/short_horizon/clean_round/SIGNAL_AUDIT_PREREGISTRATION_2026_10_26.md,
committed before any result): is there ANY out-of-sample predictive signal in the recipe's inputs for the targets the
strategy needs? No strategy layer, no threshold chosen on the scored block, no policy replay.

Per walk-forward fold (wf2022, wf2023): fit on TRAIN, choose hyper-parameters and calibrate on INNER_A, score INNER_B
once (OBS_2024 one reported-only read). Targets from the engine-exact C2 entry outcome tables (plus one extra 1σ/1σ
table built with the same engine function) and from the bars. Metrics with a stationary block bootstrap over days.

  run      build targets, fit (ii) HistGradientBoosting / (iii) logistic-ridge / (iv) naive, score, write fold JSON
  heads    collect the (i) already-trained heads' INNER_B reported-only metrics from E3M.json files
  report   combine fold JSONs + heads into SIGNAL_AUDIT.{json,md} with the pre-registered verdicts
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 20261026
BLOCK_DAYS = 5.0
N_BOOT = 2000
LAG_SLOTS = (-1, -2, -4, -7, 0)               # observed-row lags 0, 2, 6, 12, 24 of the 13 x stride-2 window
LAG_NAMES = ("l0", "l2", "l6", "l12", "l24")
CELLS_A = ((1.0, 1.0), (1.5, 1.5), (1.5, 2.5), (2.0, 2.5))
EXTRA_CELL = (1.0, 1.0)
FWD_H = (6, 12, 24)
EXIT_CELL = (1.5, 2.5)
EXIT_K = (3, 8)
HIT_TP, HIT_SL, HIT_TIME = 0, 1, 2
HGB_ITERS = (50, 100, 200, 400)
LOGIT_C = (0.001, 0.01, 0.1, 1.0)
RIDGE_A = (1.0, 10.0, 100.0, 1000.0)
AUC_LO_MIN = 0.52
SPEARMAN_LO_MIN = 0.02
RULE = ("SIGNAL_PRESENT for a target only if the lower 95 % CI of its INNER_B AUC is > 0.52 (or its Spearman lower CI > 0.02) "
        "in BOTH folds AND the top-decile precision exceeds the cost-inclusive break-even win rate in both folds; otherwise "
        "NO_USABLE_SIGNAL. If NO target is SIGNAL_PRESENT, further strategy-layer tuning is suspended and work moves to inputs "
        "(C3 invariance across years/regimes, data).")


# ----------------------------------------------------------------------------- metrics (pure)
def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties averaged (scipy when available, else the plain loop below)."""
    x = np.asarray(x, float)
    try:
        from scipy.stats import rankdata as _rd
        return _rd(x, method="average")
    except ImportError:
        pass
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), float)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, int)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def logloss(y, p) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def skill(score_model: float, score_ref: float) -> float:
    return float(1.0 - score_model / score_ref) if score_ref > 0 else float("nan")


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def r2(y, yhat, ref: float) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss = float(np.sum((y - ref) ** 2))
    return float(1.0 - np.sum((y - yhat) ** 2) / ss) if ss > 0 else float("nan")


def top_mask(score, frac: float) -> np.ndarray:
    """The top ``frac`` of rows by score (ties broken by order); at least one row."""
    score = np.asarray(score, float)
    k = max(1, int(round(frac * len(score))))
    idx = np.argsort(-score, kind="mergesort")[:k]
    m = np.zeros(len(score), bool)
    m[idx] = True
    return m


def breakeven_win_rate(net: np.ndarray, win: np.ndarray) -> float:
    """p* = -L / (W - L): the win rate at which E[net] = 0 given W = mean net of wins, L = mean net of the rest."""
    net, win = np.asarray(net, float), np.asarray(win, bool)
    if win.sum() == 0 or (~win).sum() == 0:
        return float("nan")
    W, L = float(net[win].mean()), float(net[~win].mean())
    return float(-L / (W - L)) if W != L else float("nan")


def breakeven_close_precision(delta: np.ndarray, y: np.ndarray) -> float:
    """(d): SL-first precision at which E[hold - close] = 0: mean(Δ|y=0) / (mean(Δ|y=0) - mean(Δ|y=1))."""
    delta, y = np.asarray(delta, float), np.asarray(y, int)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    d0, d1 = float(delta[y == 0].mean()), float(delta[y == 1].mean())
    return float(d0 / (d0 - d1)) if d0 != d1 else float("nan")


def stationary_indices(T: int, block: float, rng) -> np.ndarray:
    from app.short_horizon.expert_guided.stats import stationary_bootstrap_indices
    return stationary_bootstrap_indices(T, block, rng)


def day_groups(day: np.ndarray):
    """(unique days in order, list of row-index arrays per day)."""
    day = np.asarray(day)
    u, inv = np.unique(day, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(u))]
    return u, groups


def bootstrap(fn, day: np.ndarray, n: int = N_BOOT, block: float = BLOCK_DAYS, seed: int = SEED) -> dict:
    """Stationary block bootstrap over calendar days: each draw resamples whole days (blocks of mean length ``block``)
    and evaluates ``fn(idx) -> dict of floats`` on the drawn rows. Returns {metric: [lo, hi]} (2.5 / 97.5 %)."""
    u, groups = day_groups(day)
    T = len(u)
    if T < 2:
        return {}
    rng = np.random.default_rng(seed)
    vals = {}
    for _ in range(n):
        di = stationary_indices(T, block, rng)
        idx = np.concatenate([groups[i] for i in di])
        for k, v in fn(idx).items():
            vals.setdefault(k, []).append(v)
    out = {}
    for k, v in vals.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        out[k] = [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [float("nan"), float("nan")]
    return out


def classification_metrics(y, p, base_rate: float, econ: dict | None = None):
    """Point metrics + a function for the bootstrap. ``econ``: {"kind": "a"|"d"|"c", arrays aligned with y}."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)

    def m(idx):
        yy, pp = y[idx], p[idx]
        out = {"auc": auc(yy, pp), "brier_skill": skill(brier(yy, pp), brier(yy, np.full(len(yy), base_rate))),
               "logloss_skill": skill(logloss(yy, pp), logloss(yy, np.full(len(yy), base_rate)))}
        if econ is not None:
            out.update(econ_metrics(econ, idx, pp, yy))
        return out
    return m


def econ_metrics(econ: dict, idx, score, y) -> dict:
    out = {}
    for frac, tag in ((0.10, "top10"), (0.20, "top20")):
        if econ["kind"] in ("a", "d"):
            mk = top_mask(score, frac)
            out[f"{tag}_precision"] = float(np.mean(y[mk]))
            out[f"{tag}_mean_value"] = float(np.mean(np.asarray(econ["value"])[idx][mk]))
        elif econ["kind"] == "c":
            up, dn = top_mask(score, frac), top_mask(-np.asarray(score), frac)
            move, cost = np.asarray(econ["move"])[idx], np.asarray(econ["cost"])[idx]
            net = np.concatenate([move[up] - cost[up], -move[dn] - cost[dn]])
            corr = np.concatenate([(move[up] > 0), (move[dn] < 0)])
            out[f"{tag}_precision"] = float(np.mean(corr))
            out[f"{tag}_mean_value"] = float(np.mean(net))
    return out


def regression_metrics(y, yhat, mean_ref: float):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)

    def m(idx):
        yy, hh = y[idx], yhat[idx]
        out = {"spearman": spearman(yy, hh), "r2_vs_mean": r2(yy, hh, mean_ref), "r2_vs_zero": r2(yy, hh, 0.0)}
        for frac, tag in ((0.10, "top10"), (0.20, "top20")):
            mk = top_mask(hh, frac)
            out[f"{tag}_mean_value"] = float(np.mean(yy[mk]))
            out[f"{tag}_precision"] = float(np.mean(yy[mk] > 0))
        return out
    return m


def verdict(target_kind: str, by_fold: dict) -> dict:
    """Pre-registered rule. ``by_fold[fold][model] = {"point": {...}, "ci": {...}, "breakeven": p*|None}``.
    SIGNAL_PRESENT when one model of (ii)/(iii) passes both conditions in BOTH folds."""
    folds = sorted(by_fold)
    passing = []
    detail = {}
    for model in ("hgb", "linear"):
        ok_all = len(folds) == 2
        det = {}
        for f in folds:
            r = by_fold[f].get(model)
            if not r:
                ok_all = False
                det[f] = "missing"
                continue
            ci, pt = r["ci"], r["point"]
            if target_kind == "regression":
                stat_ok = ci.get("spearman", [float("nan")])[0] > SPEARMAN_LO_MIN
                econ_ok = pt.get("top10_mean_value", float("nan")) > 0
            elif target_kind == "c":
                stat_ok = ci.get("auc", [float("nan")])[0] > AUC_LO_MIN
                econ_ok = pt.get("top10_mean_value", float("nan")) > 0
            else:
                stat_ok = ci.get("auc", [float("nan")])[0] > AUC_LO_MIN
                be = r.get("breakeven")
                econ_ok = be is not None and math.isfinite(be) and pt.get("top10_precision", float("nan")) > be
            det[f] = {"statistical": bool(stat_ok), "economic": bool(econ_ok)}
            ok_all &= bool(stat_ok and econ_ok)
        detail[model] = det
        if ok_all:
            passing.append(model)
    return {"verdict": "SIGNAL_PRESENT" if passing else "NO_USABLE_SIGNAL", "passing_models": passing, "detail": detail}


# ----------------------------------------------------------------------------- targets (pure, given arrays)
def forward_sign_target(close: np.ndarray, row_bar: np.ndarray, rows: np.ndarray, h: int, bar_ns: np.ndarray,
                        period_end_ns: np.ndarray, h_ns: int):
    """y = 1 when CLOSE[bar(r+h)] > CLOSE[bar(r)] (h observed rows ahead); move = the signed price change; available when
    row r+h exists and its bar end (start + 1 h) <= the period end of row r. Returns (ok, y, move)."""
    n = len(row_bar)
    rows = np.asarray(rows, np.int64)
    rr = rows + h
    ok = rr < n
    b0 = row_bar[rows]
    b1 = np.where(ok, row_bar[np.clip(rr, 0, n - 1)], -1)
    ok &= (b1 >= 0) & (b0 >= 0)
    t_end = np.where(ok, bar_ns[np.clip(b1, 0, len(bar_ns) - 1)] + h_ns, np.iinfo(np.int64).max)
    ok &= t_end <= period_end_ns
    move = np.where(ok, close[np.clip(b1, 0, len(close) - 1)] - close[np.clip(b0, 0, len(close) - 1)], np.nan)
    return ok, (move > 0).astype(np.int64), move


def exit_decisions(entry_rows: np.ndarray, row_bar: np.ndarray, hit: np.ndarray, net: np.ndarray, exit_bar: np.ndarray,
                   k: int):
    """Strategy-free early-close decision rows: a position opened after row r0 (fill at bar(r0)+1) and still open after
    the bar of row r = r0 + k (exit_bar > bar(r)). Label 1 when its final engine outcome is SL or a negative forced flat.
    Returns (r0, r, y)."""
    n = len(row_bar)
    r0 = np.asarray(entry_rows, np.int64)
    r = r0 + k
    ok = (r < n) & (hit >= 0) & (exit_bar >= 0)
    br = np.where(ok, row_bar[np.clip(r, 0, n - 1)], -1)
    fill = row_bar[r0] + 1
    ok &= (br >= fill) & (exit_bar > br)
    y = ((hit == HIT_SL) | ((hit == HIT_TIME) & (net < 0))).astype(np.int64)
    return r0[ok], r[ok], y[ok]


# ----------------------------------------------------------------------------- models
def fit_classifier(kind: str, Xtr, ytr, Xa, ya):
    """Returns (predict_proba_fn, info): hyper-parameter chosen on INNER_A log-loss, Platt-calibrated on INNER_A."""
    from sklearn.linear_model import LogisticRegression
    best = None
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        for it in HGB_ITERS:
            m = HistGradientBoostingClassifier(max_leaf_nodes=15, min_samples_leaf=200, learning_rate=0.05, l2_regularization=1.0,
                                               max_iter=it, early_stopping=False, random_state=SEED)
            m.fit(Xtr, ytr)
            ll = logloss(ya, m.predict_proba(Xa)[:, 1])
            if best is None or ll < best[0]:
                best = (ll, m, {"max_iter": it})
        raw = lambda X, m=best[1]: m.predict_proba(X)[:, 1]  # noqa: E731
    else:
        mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0)
        sd = np.where(sd > 0, sd, 1.0)
        z = lambda X: np.nan_to_num((X - mu) / sd)  # noqa: E731
        for C in LOGIT_C:
            m = LogisticRegression(C=C, max_iter=3000)
            m.fit(z(Xtr), ytr)
            ll = logloss(ya, m.predict_proba(z(Xa))[:, 1])
            if best is None or ll < best[0]:
                best = (ll, m, {"C": C})
        raw = lambda X, m=best[1]: m.predict_proba(z(X))[:, 1]  # noqa: E731
        best[2]["coef"] = best[1].coef_[0].tolist()
    pa = np.clip(raw(Xa), 1e-6, 1 - 1e-6)
    lz = np.log(pa / (1 - pa))[:, None]
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(lz, ya) if len(np.unique(ya)) == 2 else None

    def proba(X):
        p = np.clip(raw(X), 1e-6, 1 - 1e-6)
        if platt is None:
            return p
        return platt.predict_proba(np.log(p / (1 - p))[:, None])[:, 1]
    info = dict(best[2])
    info["inner_a_logloss_raw"] = float(best[0])
    return proba, info


def fit_regressor(kind: str, Xtr, ytr, Xa, ya):
    best = None
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        for it in HGB_ITERS:
            m = HistGradientBoostingRegressor(max_leaf_nodes=15, min_samples_leaf=200, learning_rate=0.05, l2_regularization=1.0,
                                              max_iter=it, early_stopping=False, random_state=SEED)
            m.fit(Xtr, ytr)
            mse = float(np.mean((m.predict(Xa) - ya) ** 2))
            if best is None or mse < best[0]:
                best = (mse, m, {"max_iter": it})
        return (lambda X, m=best[1]: m.predict(X)), dict(best[2], inner_a_mse=best[0])
    from sklearn.linear_model import Ridge
    mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0)
    sd = np.where(sd > 0, sd, 1.0)
    z = lambda X: np.nan_to_num((X - mu) / sd)  # noqa: E731
    for a in RIDGE_A:
        m = Ridge(alpha=a).fit(z(Xtr), ytr)
        mse = float(np.mean((m.predict(z(Xa)) - ya) ** 2))
        if best is None or mse < best[0]:
            best = (mse, m, {"alpha": a})
    return (lambda X, m=best[1]: m.predict(z(X))), dict(best[2], inner_a_mse=best[0])


# ----------------------------------------------------------------------------- run one fold
def build_fold(a):
    from app.short_horizon.expert_guided import data as DG
    from app.short_horizon.expert_guided import env as E
    from app.short_horizon.expert_guided import outcomes as OC
    from tools.expert_guided_e1t import Ctx
    from tools.expert_guided_e3m import effective_contracts
    rec = json.loads(Path(a.recipe).read_text())
    feats = [c["store_column"] for c in rec["tensor"]["columns"]]
    names = [c.get("tensor_name", c["store_column"]) for c in rec["tensor"]["columns"]]
    ds = DG.load_store(a.admitted, a.store, feats, scaling="fixed_train", contract=a.split_contract, recipe=rec)
    cx = Ctx(ds)
    SCFG, _ = effective_contracts("sac", 2.0)
    t = OC.load_table(a.outcomes, cx, SCFG)
    t_extra = OC.entry_table(cx, SCFG, cells=(EXTRA_CELL,), rows=np.flatnonzero(t["eligible"]))
    return ds, cx, SCFG, t, t_extra, names, E, OC


def features(ds, rows):
    W = ds.windows[rows]
    return np.concatenate([W[:, s, :] for s in LAG_SLOTS], axis=1).astype(np.float32)


def cmd_run(a):
    t0 = time.time()
    ds, cx, SCFG, t, tx, names, E, OC = build_fold(a)
    periods = ("TRAIN", "INNER_A", "INNER_B", "OBS_2024")
    bar_ns = ds.bars.index.as_unit("ns").asi8
    close = ds.bars["CLOSE"].to_numpy(float)
    opn = ds.bars["OPEN"].to_numpy(float)
    sig = np.asarray(ds.sigma, float)
    rt_price = float(SCFG.profile.round_trip_price())
    day = (bar_ns // (86400 * 10**9))
    row_bar = np.asarray(ds.row_bar, np.int64)
    elig = np.flatnonzero(t["eligible"] & ds.row_ok)
    pend = {p: OC.period_end_ns(cx, p) for p in periods}

    def cost_sigma(rows):
        b = row_bar[rows] + 1
        s = sig[np.clip(b, 0, len(sig) - 1)]
        return np.where(np.isfinite(s) & (s > 0), rt_price / s, np.nan)

    targets = []                      # (name, kind, per-period dict of (rows, y, extra))
    for cell in CELLS_A:
        tab, ci = (tx, 0) if cell == EXTRA_CELL else (t, OC.cell_index(cell))
        for si, side in enumerate(("long", "short")):
            per = {}
            for p in periods:
                rows = elig[ds.period[elig] == p]
                ok = OC.available(cx, tab["exit_bar"][rows, si, ci], tab["censored"][rows, si, ci], p) & np.isfinite(tab["net_sigma"][rows, si, ci])
                rows = rows[ok]
                hit = tab["hit"][rows, si, ci]
                net = tab["net_sigma"][rows, si, ci].astype(float)
                per[p] = (rows, (hit == HIT_TP).astype(np.int64), net)
            targets.append((f"a_tp_first_{side}_tp{cell[0]:g}_sl{cell[1]:g}", "a", per))
            targets.append((f"b_net_{side}_tp{cell[0]:g}_sl{cell[1]:g}", "regression", {p: (v[0], v[2], v[2]) for p, v in per.items()}))
    for h in FWD_H:
        per = {}
        for p in periods:
            rows = elig[ds.period[elig] == p]
            ok, y, move = forward_sign_target(close, row_bar, rows, h, bar_ns, pend[p], E.H_NS)
            rows = rows[ok]
            s = sig[np.clip(row_bar[rows] + 1, 0, len(sig) - 1)]
            per[p] = (rows, y[ok], {"move": move[ok] / s, "cost": cost_sigma(rows)})
        targets.append((f"c_sign_fwd{h}", "c", per))
    ci_exit = OC.cell_index(EXIT_CELL)
    for k in EXIT_K:
        for si, side in enumerate(("long", "short")):
            d = 1 if si == 0 else -1
            per = {}
            r0_all = elig
            r0, r, y = exit_decisions(r0_all, row_bar, t["hit"][r0_all, si, ci_exit], t["net_sigma"][r0_all, si, ci_exit],
                                      t["exit_bar"][r0_all, si, ci_exit], k)
            for p in periods:
                m = (ds.period[r] == p) & ds.row_ok[r]
                m &= OC.available(cx, t["exit_bar"][r0, si, ci_exit], t["censored"][r0, si, ci_exit], p)
                rr, rr0, yy = r[m], r0[m], y[m]
                fill = row_bar[rr0] + 1
                s0 = sig[np.clip(fill, 0, len(sig) - 1)]
                close_now = d * (opn[np.clip(row_bar[rr] + 1, 0, len(opn) - 1)] - opn[fill]) / s0 - rt_price / s0
                delta = t["net_sigma"][rr0, si, ci_exit].astype(float) - close_now
                per[p] = (rr, yy, delta)
            targets.append((f"d_sl_first_{side}_k{k}", "d", per))

    out = {"fold": a.split_contract, "recipe": Path(a.recipe).name, "outcomes": Path(a.outcomes).name,
           "outcomes_stamp": t["stamp"]["sha256"], "extra_cell_stamp": tx["stamp"]["sha256"], "features": len(names) * len(LAG_SLOTS),
           "lags": dict(zip(LAG_NAMES, LAG_SLOTS)), "targets": {}}
    for name, kind, per in targets:
        rtr, ytr, xtr = per["TRAIN"]
        ra, ya, _ = per["INNER_A"]
        if len(rtr) < 200 or len(ra) < 50 or (kind != "regression" and len(np.unique(ytr)) < 2):
            out["targets"][name] = {"kind": kind, "skipped": "too few rows"}
            continue
        Xtr, Xa = features(ds, rtr), features(ds, ra)
        rec = {"kind": kind, "n": {p: int(len(per[p][0])) for p in periods}}
        if kind == "regression":
            mean_ref = float(np.mean(ytr))
            rec["train_mean"] = mean_ref
        else:
            base = float(np.mean(ytr))
            rec["train_base_rate"] = base
            be = None
            if kind == "a":
                be = breakeven_win_rate(xtr, ytr == 1)
            elif kind == "d":
                be = breakeven_close_precision(xtr, ytr)
            rec["breakeven"] = be
        rec["models"] = {}
        coef_out = None
        for model in ("hgb", "linear"):
            if kind == "regression":
                fn, info = fit_regressor(model, Xtr, ytr, Xa, ya)
            else:
                fn, info = fit_classifier(model, Xtr, ytr, Xa, ya)
                if model == "linear":
                    coef_out = info.pop("coef")
            mr = {"info": info}
            for p in ("INNER_B", "OBS_2024"):
                rp, yp, xp = per[p]
                if len(rp) < 30:
                    mr[p] = {"skipped": "too few rows"}
                    continue
                score = fn(features(ds, rp))
                dd = day[row_bar[rp]]
                if kind == "regression":
                    mfn = regression_metrics(yp, score, mean_ref)
                else:
                    econ = None
                    if kind in ("a", "d"):
                        econ = {"kind": kind, "value": xp}
                    elif kind == "c":
                        econ = {"kind": "c", "move": xp["move"], "cost": xp["cost"]}
                    mfn = classification_metrics(yp, score, base, econ)
                idx = np.arange(len(yp))
                mr[p] = {"n": int(len(yp)), "days": int(len(np.unique(dd))), "point": mfn(idx), "ci": bootstrap(mfn, dd)}
                if p == "INNER_B":
                    mr[p]["_score"] = score.tolist()
                    mr[p]["_y"] = np.asarray(yp, float).tolist()
                    mr[p]["_day"] = dd.tolist()
                    if kind in ("a", "d"):
                        mr[p]["_value"] = np.asarray(xp, float).tolist()
                    elif kind == "c":
                        mr[p]["_move"] = np.asarray(xp["move"], float).tolist()
                        mr[p]["_cost"] = np.asarray(xp["cost"], float).tolist()
            rec["models"][model] = mr
        # naive
        nv = {}
        rp, yp, xp = per["INNER_B"]
        if kind == "regression":
            nv["INNER_B"] = {"r2_vs_mean_of_train_mean": r2(yp, np.full(len(yp), rec["train_mean"]), float(np.mean(yp))),
                             "mean_target": float(np.mean(yp)), "share_positive": float(np.mean(np.asarray(yp) > 0))}
        else:
            nv["INNER_B"] = {"base_rate_inner_b": float(np.mean(yp)), "train_base_rate": rec["train_base_rate"],
                             "auc": 0.5, "brier_skill_of_train_base_rate": skill(brier(yp, np.full(len(yp), rec["train_base_rate"])),
                                                                                 brier(yp, np.full(len(yp), float(np.mean(yp)))))}
        rec["models"]["naive"] = nv
        if coef_out is not None and kind in ("a", "c"):
            rec["linear_coef_top"] = sorted(((names[i % len(names)] + "@" + LAG_NAMES[i // len(names)], float(c)) for i, c in enumerate(coef_out)),
                                            key=lambda x: -abs(x[1]))[:30]
        out["targets"][name] = rec
        print(json.dumps({"fold": a.split_contract, "target": name, "n_train": int(len(rtr)), "n_b": int(len(per["INNER_B"][0])),
                          "auc_hgb": (rec["models"]["hgb"].get("INNER_B", {}).get("point", {}) or {}).get("auc"),
                          "auc_lin": (rec["models"]["linear"].get("INNER_B", {}).get("point", {}) or {}).get("auc"),
                          "s": round(time.time() - t0, 1)}), flush=True)
    # feature-sign stability (descriptive): per calendar year, Spearman of current-slot features with the target
    if a.split_contract == "wf2023":
        stab = {}
        years = (2020, 2021, 2022)
        bar_year = np.asarray(ds.bars.index.year)
        for name in ("a_tp_first_long_tp1.5_sl1.5", "a_tp_first_short_tp1.5_sl1.5", "c_sign_fwd12"):
            rec = out["targets"].get(name)
            if not rec or "linear_coef_top" not in rec:
                continue
            kind = rec["kind"]
            per = next(pp for nn, kk, pp in targets if nn == name)
            rows = np.concatenate([per[p][0] for p in ("TRAIN", "INNER_A")])
            yv = np.concatenate([per[p][1] for p in ("TRAIN", "INNER_A")])
            base_names = []
            for fname, _c in rec["linear_coef_top"]:
                bn = fname.split("@")[0]
                if bn not in base_names:
                    base_names.append(bn)
                if len(base_names) == 15:
                    break
            res = {}
            Xc = ds.windows[rows][:, -1, :]
            yrs = bar_year[row_bar[rows]]
            for bn in base_names:
                j = names.index(bn)
                rho = {}
                for y_ in years:
                    m = yrs == y_
                    rho[str(y_)] = spearman(Xc[m, j], yv[m]) if m.sum() > 50 else float("nan")
                sg = [np.sign(v) for v in rho.values() if math.isfinite(v) and abs(v) > 0.01]
                res[bn] = {"rho_by_year": rho, "flips": bool(len(set(sg)) > 1)}
            stab[name] = {"years": list(years), "note": "2022 rows are wf2023 TRAIN (H1) + INNER_A (H2); descriptive only, never decides",
                          "features": res, "n_flipping": int(sum(v["flips"] for v in res.values())), "n_features": len(res)}
        out["feature_sign_stability"] = stab
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps({"out": a.out, "targets": len(out["targets"]), "seconds": out["seconds"]}))


# ----------------------------------------------------------------------------- (i) trained heads
def _num_leaves(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            if k in ("reliability",):
                continue
            yield from _num_leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield prefix, float(d)


def collect_heads(results_dir: str, patterns=("WF_V13_*", "WF_C2_13_*", "WF_C2G_13_*")) -> dict:
    out = {}
    for pat in patterns:
        for f in sorted(glob.glob(str(Path(results_dir) / pat / "E3M.json"))):
            unit = Path(f).parent.name
            d = json.loads(Path(f).read_text())
            per = {}
            for seed, v in d["trials"]["FULL"].items():
                rb = ((v.get("action_contract") or {}).get("reported_only_INNER_B") or {}).get("heads") or {}
                for head, hv in rb.items():
                    for path, val in _num_leaves(hv):
                        if path.split(".")[-1] in ("auc", "brier", "ece", "r2", "spearman", "mae", "positives", "n", "rows"):
                            per.setdefault(head, {}).setdefault(path, []).append(val)
            out[unit] = {h: {p: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "seeds": len(v)}
                             for p, v in hv.items()} for h, hv in per.items()}
    return out


def cmd_heads(a):
    doc = collect_heads(a.results)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({"out": a.out, "units": list(doc)}))


# ----------------------------------------------------------------------------- report
def _pooled(recs, kind):
    """Pool the two folds' INNER_B predictions (each from its own fold model) and recompute metrics + CI."""
    ys, ss, ds_, vals, moves, costs = [], [], [], [], [], []
    for r in recs:
        b = r.get("INNER_B") or {}
        if "_score" not in b:
            return None
        ys.append(np.asarray(b["_y"]))
        ss.append(np.asarray(b["_score"]))
        ds_.append(np.asarray(b["_day"]))
        if "_value" in b:
            vals.append(np.asarray(b["_value"]))
        if "_move" in b:
            moves.append(np.asarray(b["_move"]))
            costs.append(np.asarray(b["_cost"]))
    y, s, day = np.concatenate(ys), np.concatenate(ss), np.concatenate(ds_)
    if kind == "regression":
        fn = regression_metrics(y, s, 0.0)
    else:
        econ = {"kind": kind, "value": np.concatenate(vals)} if vals else ({"kind": "c", "move": np.concatenate(moves), "cost": np.concatenate(costs)} if moves else None)
        fn = classification_metrics(y, s, float(np.mean(y)), econ)
    return {"n": int(len(y)), "point": fn(np.arange(len(y))), "ci": bootstrap(fn, day)}


def build_report(fold_docs: list, heads: dict | None) -> dict:
    by_t = {}
    for fd in fold_docs:
        for name, rec in fd["targets"].items():
            by_t.setdefault(name, {})[fd["fold"]] = rec
    targets = {}
    for name, folds in sorted(by_t.items()):
        kind = next(iter(folds.values())).get("kind")
        vin = {}
        for f, rec in folds.items():
            if "skipped" in rec:
                continue
            vin[f] = {}
            for model in ("hgb", "linear"):
                b = rec["models"].get(model, {}).get("INNER_B")
                if b and "point" in b:
                    vin[f][model] = {"point": b["point"], "ci": b["ci"], "breakeven": rec.get("breakeven")}
        v = verdict(kind, vin)
        pooled = {}
        for model in ("hgb", "linear"):
            recs = [folds[f]["models"][model] for f in sorted(folds) if "models" in folds[f]]
            pooled[model] = _pooled(recs, kind) if len(recs) == 2 else None
        slim = {}
        for f, rec in folds.items():
            r2_ = {k: v_ for k, v_ in rec.items() if k != "models"}
            r2_["models"] = {}
            for model, mr in rec.get("models", {}).items():
                r2_["models"][model] = {p: ({k: v__ for k, v__ in x.items() if not k.startswith("_")} if isinstance(x, dict) else x) for p, x in mr.items()}
            slim[f] = r2_
        targets[name] = {"kind": kind, "verdict": v, "folds": slim, "pooled_inner_b": pooled}
    any_signal = any(t["verdict"]["verdict"] == "SIGNAL_PRESENT" for t in targets.values())
    stab = {fd["fold"]: fd.get("feature_sign_stability") for fd in fold_docs if fd.get("feature_sign_stability")}
    return {"rule": RULE, "prereg": "docs/short_horizon/clean_round/SIGNAL_AUDIT_PREREGISTRATION_2026_10_26.md",
            "overall": "AT_LEAST_ONE_SIGNAL_PRESENT" if any_signal else "NO_TARGET_HAS_USABLE_SIGNAL",
            "consequence": None if any_signal else "strategy-layer tuning suspended; work moves to inputs (C3 invariance, data)",
            "folds": [{k: fd[k] for k in ("fold", "recipe", "outcomes", "outcomes_stamp", "extra_cell_stamp", "features", "lags", "seconds")} for fd in fold_docs],
            "targets": targets, "feature_sign_stability": stab, "trained_heads_inner_b_reported_only": heads}


def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _ci(c, k, nd=3):
    v = (c or {}).get(k)
    return "[—]" if not v else f"[{_f(v[0], nd)}, {_f(v[1], nd)}]"


def markdown(doc: dict) -> str:
    L = ["# Signal audit: out-of-sample predictive signal of the recipe's inputs (strategy-free, 2026-10-26)", "",
         f"Pre-registration: `{doc['prereg']}` (committed before any result). TEST closed. INNER_B scored once per fold; "
         "CIs: stationary block bootstrap over days (block 5, 2 000 draws, seed 20261026).", "",
         f"**Overall: {doc['overall']}**" + (f" — {doc['consequence']}." if doc["consequence"] else "."), "",
         f"Rule: {doc['rule']}", "",
         "## Summary (INNER_B; per fold: AUC or Spearman [95 % CI] · skill · top-10 % precision or mean value vs break-even)", "",
         "| target | model | wf2022 (2022) | wf2023 (2023) | pooled | break-even (2022 / 2023) | verdict |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, t in doc["targets"].items():
        kind = t["kind"]
        for model in ("hgb", "linear"):
            cells = []
            bes = []
            for f in ("wf2022", "wf2023"):
                rec = t["folds"].get(f, {})
                b = (rec.get("models", {}).get(model, {}) or {}).get("INNER_B") or {}
                pt, ci = b.get("point") or {}, b.get("ci") or {}
                if kind == "regression":
                    cells.append(f"ρ {_f(pt.get('spearman'))} {_ci(ci, 'spearman')} · R²₀ {_f(pt.get('r2_vs_zero'))} · top10 net {_f(pt.get('top10_mean_value'))}σ")
                else:
                    cells.append(f"AUC {_f(pt.get('auc'))} {_ci(ci, 'auc')} · BS {_f(pt.get('brier_skill'))} · top10 {_f(pt.get('top10_precision'))}"
                                 + (f" / net {_f(pt.get('top10_mean_value'))}σ" if kind == "c" else f" / Δ̄ {_f(pt.get('top10_mean_value'))}σ" if kind == "d" else f" / net {_f(pt.get('top10_mean_value'))}σ"))
                bes.append(_f(rec.get("breakeven")) if kind in ("a", "d") else ("net > 0" if kind in ("c", "regression") else "—"))
            po = (t.get("pooled_inner_b") or {}).get(model) or {}
            ppt, pci = po.get("point") or {}, po.get("ci") or {}
            pcell = (f"ρ {_f(ppt.get('spearman'))} {_ci(pci, 'spearman')}" if kind == "regression" else f"AUC {_f(ppt.get('auc'))} {_ci(pci, 'auc')}") if po else "—"
            v = t["verdict"]
            vd = v["verdict"] + (" (" + ", ".join(v["passing_models"]) + ")" if v["passing_models"] else "")
            L.append(f"| {name} | {model} | {cells[0]} | {cells[1]} | {pcell} | {' / '.join(bes)} | {vd if model == 'hgb' else ''} |")
    L += ["", "Legend: BS = Brier skill vs the TRAIN base rate; top10 = precision in the 10 % highest-scored rows; net = realized "
          "mean engine net σ there (costs included; (c): long top / short bottom decile minus the real round-trip cost); "
          "Δ̄ (target d) = realized mean of hold − close-now there (< 0: closing pays); break-even = cost-inclusive win rate "
          "p* from TRAIN outcomes for (a), the SL-first precision at which closing equals holding for (d).", ""]
    L += ["## Detail per target and fold", ""]
    for name, t in doc["targets"].items():
        L.append(f"### {name} ({t['kind']}) — {t['verdict']['verdict']}")
        for f, rec in sorted(t["folds"].items()):
            if "skipped" in rec:
                L.append(f"- {f}: skipped ({rec['skipped']})")
                continue
            L.append(f"- {f}: n {rec.get('n')}; " + (f"TRAIN base rate {_f(rec.get('train_base_rate'))}; break-even {_f(rec.get('breakeven'))}"
                                                     if t["kind"] != "regression" else f"TRAIN mean {_f(rec.get('train_mean'))}"))
            for model in ("hgb", "linear"):
                mr = rec["models"].get(model, {})
                for p in ("INNER_B", "OBS_2024"):
                    b = mr.get(p) or {}
                    if "point" not in b:
                        continue
                    pt, ci = b["point"], b["ci"]
                    kv = "; ".join(f"{k} {_f(v)} {_ci(ci, k)}" for k, v in pt.items())
                    L.append(f"  - {model} {p}{' (reported only)' if p == 'OBS_2024' else ''} [{mr.get('info')}]: {kv}")
            nv = rec["models"].get("naive", {}).get("INNER_B")
            if nv:
                L.append(f"  - naive INNER_B: {json.dumps(nv)}")
        L.append("")
    stab = doc.get("feature_sign_stability") or {}
    if stab:
        L += ["## Feature-sign stability by year (descriptive, never decides)", ""]
        for fold, s in stab.items():
            for name, v in s.items():
                L.append(f"### {name} ({fold}): {v['n_flipping']} of {v['n_features']} top features flip sign across {v['years']}")
                L.append("| feature | " + " | ".join(str(y) for y in v["years"]) + " | flips |")
                L.append("| --- | " + " | ".join("---:" for _ in v["years"]) + " | --- |")
                for fn_, r in v["features"].items():
                    L.append(f"| {fn_} | " + " | ".join(_f(r["rho_by_year"][str(y)]) for y in v["years"]) + f" | {'YES' if r['flips'] else ''} |")
                L.append("")
    heads = doc.get("trained_heads_inner_b_reported_only") or {}
    if heads:
        L += ["## (i) Already-trained heads: INNER_B reported-only AUC (mean [min, max] over seeds; no CI, context only)", "",
              "| unit | head | AUC raw | AUC calibrated | Brier calibrated |", "| --- | --- | --- | --- | --- |"]
        for unit, hv in heads.items():
            for head, m in hv.items():
                def g(k):
                    x = m.get(k)
                    return f"{x['mean']:.3f} [{x['min']:.3f}, {x['max']:.3f}]" if x else "—"
                araw = next((k for k in m if k.endswith("raw.auc")), None)
                acal = next((k for k in m if k.endswith("calibrated.auc")), None)
                bcal = next((k for k in m if k.endswith("calibrated.brier")), None)
                if araw or acal:
                    L.append(f"| {unit} | {head} | {g(araw) if araw else '—'} | {g(acal) if acal else '—'} | {g(bcal) if bcal else '—'} |")
        L.append("")
    return "\n".join(L) + "\n"


def cmd_report(a):
    fold_docs = [json.loads(Path(p).read_text()) for p in a.folds]
    heads = json.loads(Path(a.heads).read_text()) if a.heads else None
    doc = build_report(fold_docs, heads)
    Path(a.out_json).write_text(json.dumps(doc, indent=1))
    Path(a.out_md).write_text(markdown(doc))
    print(json.dumps({"overall": doc["overall"], "signal": [k for k, v in doc["targets"].items() if v["verdict"]["verdict"] == "SIGNAL_PRESENT"]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    for k in ("admitted", "store", "recipe", "split_contract", "outcomes", "out"):
        r.add_argument("--" + k.replace("_", "-"), dest=k, required=True)
    r.set_defaults(fn=cmd_run)
    h = sp.add_parser("heads")
    h.add_argument("--results", required=True)
    h.add_argument("--out", required=True)
    h.set_defaults(fn=cmd_heads)
    rp = sp.add_parser("report")
    rp.add_argument("--folds", nargs="+", required=True)
    rp.add_argument("--heads")
    rp.add_argument("--out-json", required=True)
    rp.add_argument("--out-md", required=True)
    rp.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
