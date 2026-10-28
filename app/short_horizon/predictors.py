"""Cheap causal baseline predictors for the range-first inputs (DEVELOPMENT).

Every prediction for block k uses only blocks completed before k's start (blocks
never overlap, so these are the earlier rows of the table). Refits are expanding
and happen at the first block of each week (``weekly``) or business day (``daily``);
a refit uses only rows before that first block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .blocks import AMBIGUOUS, LOW_FIRST, OK

HOUR = pd.Timedelta(hours=1)


def block_table(blocks, calendar, B: int) -> pd.DataFrame:
    rows = []
    for k in range(len(blocks)):
        if blocks.status[k] != OK:
            continue
        seg = calendar.segments[blocks.segment[k]]
        since = (blocks.start[k] - seg.market_open) / HOUR
        day = int(since // 24)
        ft = int(blocks.ft[k])
        rows.append({"k": k, "start": blocks.start[k], "ref": float(blocks.ref[k]),
                     "U": float(blocks.H[k] - blocks.ref[k]), "D": float(blocks.ref[k] - blocks.L[k]),
                     "ft": ft, "y": np.nan if ft == AMBIGUOUS else float(ft == LOW_FIRST),
                     "posday": int((since % 24) // B), "dayidx": day,
                     "week": tuple(blocks.start[k].isocalendar()[:2]),
                     "day": (int(blocks.segment[k]), day)})
    return pd.DataFrame(rows)


def refit_groups(t: pd.DataFrame, cadence: str):
    key = {"weekly": "week", "daily": "day"}[cadence]
    groups, cur, start = [], None, 0
    for i, v in enumerate(t[key]):
        if v != cur:
            if cur is not None:
                groups.append((start, list(range(start, i))))
            cur, start = v, i
    if cur is not None:
        groups.append((start, list(range(start, len(t)))))
    return groups


def _windows(B: int):
    nd = max(1, 24 // B)
    return nd, 5 * nd, 21 * nd


def _calendar_dummies(t: pd.DataFrame, B: int):
    cols = []
    for d in range(1, 5):
        cols.append((t.dayidx == d).astype(float).to_numpy())
    for p in range(1, max(1, 24 // B)):
        cols.append((t.posday == p).astype(float).to_numpy())
    return cols


def har_features(t: pd.DataFrame, B: int) -> np.ndarray:
    nd, nw, nm = _windows(B)
    cols = [np.ones(len(t))]
    for c in ("U", "D"):
        past = t[c].shift(1)
        for n in (nd, nw, nm):
            cols.append(past.rolling(n, min_periods=n).mean().to_numpy())
    cols += _calendar_dummies(t, B)
    return np.column_stack(cols)


def _ridge(X, y, lam=1e-8):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def har_predict(t: pd.DataFrame, B: int, target: str, cadence: str, min_rows: int = 100) -> np.ndarray:
    """HAR-style linear regression of U or D on past daily/weekly/monthly means + calendar."""
    X = har_features(t, B)
    y = t[target].to_numpy(float)
    okx = np.isfinite(X).all(axis=1)
    out = np.full(len(t), np.nan)
    for g, rows in refit_groups(t, cadence):
        train = np.flatnonzero(okx[:g] & np.isfinite(y[:g]))
        if len(train) < min_rows:
            continue
        beta = _ridge(X[train], y[train])
        rr = [r for r in rows if okx[r]]
        out[rr] = X[rr] @ beta
    return out


def logit_features(t: pd.DataFrame, B: int) -> np.ndarray:
    nd, nw, _ = _windows(B)
    ref = t["ref"]
    lab = (2 * t["y"] - 1).fillna(0.0)
    cols = [(ref - ref.shift(1)).to_numpy(), (ref - ref.shift(nd)).to_numpy(), (ref - ref.shift(nw)).to_numpy(),
            lab.shift(1).to_numpy(), lab.shift(1).rolling(3, min_periods=3).mean().to_numpy(),
            (t["U"] - t["D"]).shift(1).to_numpy()]
    cols += _calendar_dummies(t, B)
    return np.column_stack(cols)


def _fit_logit(X, y, lam=1.0, iters=25):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    reg = lam * np.eye(Z.shape[1])
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        W = p * (1 - p)
        H = Z.T @ (Z * W[:, None]) + reg
        g = Z.T @ (p - y) + reg @ w
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w, mu, sd


def logit_predict(t: pd.DataFrame, B: int, cadence: str, min_rows: int = 100) -> np.ndarray:
    """Logistic regression of P(low first) on lagged returns, past order labels and calendar."""
    X = logit_features(t, B)
    y = t["y"].to_numpy(float)
    okx = np.isfinite(X).all(axis=1)
    out = np.full(len(t), np.nan)
    for g, rows in refit_groups(t, cadence):
        train = np.flatnonzero(okx[:g] & np.isfinite(y[:g]))
        if len(train) < min_rows:
            continue
        w, mu, sd = _fit_logit(X[train], y[train])
        rr = [r for r in rows if okx[r]]
        Z = np.column_stack([np.ones(len(rr)), (X[rr] - mu) / sd])
        out[rr] = np.clip(1 / (1 + np.exp(-np.clip(Z @ w, -30, 30))), 1e-4, 1 - 1e-4)
    return out


def base_rate_predict(t: pd.DataFrame, cadence: str, min_rows: int = 20) -> np.ndarray:
    y = t["y"].to_numpy(float)
    out = np.full(len(t), np.nan)
    for g, rows in refit_groups(t, cadence):
        past = y[:g][np.isfinite(y[:g])]
        if len(past) >= min_rows:
            out[rows] = np.clip(past.mean(), 0.01, 0.99)
    return out


def ewma_predict(values, halflife: float) -> np.ndarray:
    v = np.asarray(values, float)
    a = 1 - 0.5 ** (1 / halflife)
    out = np.full(len(v), np.nan)
    for i in range(1, len(v)):
        out[i] = v[i - 1] if i == 1 else a * v[i - 1] + (1 - a) * out[i - 1]
    return out


def select_halflife(t: pd.DataFrame, train_mask, grid=(2, 3, 5, 8, 13, 21, 34, 55)) -> float:
    """Half-life minimising MAE of U and D on TRAIN rows only (predictions there use TRAIN rows only
    when TRAIN is a prefix, which it is in time order)."""
    train = np.asarray(train_mask, bool)
    best, best_err = None, np.inf
    for hl in grid:
        err = 0.0
        for c in ("U", "D"):
            e = ewma_predict(t[c].to_numpy(), hl)
            ok = train & np.isfinite(e)
            err += np.mean(np.abs(e[ok] - t[c].to_numpy()[ok]))
        if err < best_err:
            best, best_err = hl, err
    return best


def mean_window_predict(values, n: int) -> np.ndarray:
    return pd.Series(values).shift(1).rolling(n, min_periods=n).mean().to_numpy()


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


# ------------------------------------------------------------------ small gradient-boosted model
def gbm_predict(t: pd.DataFrame, F: pd.DataFrame, target: str, cadence: str, cfg: dict, min_rows: int = 100,
                predict_from=None, embargo=pd.Timedelta(0)) -> np.ndarray:
    """Median (absolute-error) gradient boosting of U or D on causal features, refit at each group start on
    rows completed before it (and at least ``embargo`` before it). Only groups starting at or after
    ``predict_from`` are fitted. Fixed small capacity from ``cfg``."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = F.to_numpy(float)
    y = t[target].to_numpy(float)
    out = np.full(len(t), np.nan)
    for g, rows in refit_groups(t, cadence):
        if predict_from is not None and t["start"].iloc[g] < predict_from:
            continue
        cutoff = t["start"].iloc[g] - embargo
        train = np.flatnonzero((np.arange(len(t)) < g) & (t["start"] <= cutoff).to_numpy() & np.isfinite(y))
        if len(train) < min_rows:
            continue
        model = HistGradientBoostingRegressor(loss="absolute_error", max_depth=cfg.get("max_depth", 2),
                                              max_iter=cfg.get("max_iter", 100),
                                              learning_rate=cfg.get("learning_rate", 0.05),
                                              min_samples_leaf=cfg.get("min_samples_leaf", 30), random_state=0)
        use = np.isfinite(X[train]).sum(axis=0) >= 10        # columns with data in this training window
        if not use.any():
            continue
        model.fit(X[train][:, use], y[train])
        out[rows] = model.predict(X[rows][:, use])
    return out


def choose_gbm_config(t: pd.DataFrame, F: pd.DataFrame, fit_end, decision_end, grid, embargo,
                      cadence: str = "weekly", min_rows: int = 100):
    """Pick the capacity with the lowest |U error| + |D error| on blocks in [fit_end, decision_end),
    predicted with weekly refits on earlier blocks only (purged by ``embargo``). Rows at or after
    ``decision_end`` are never read."""
    sel = (t["start"] < decision_end).to_numpy()
    tt, FF = t[sel].reset_index(drop=True), F[sel].reset_index(drop=True)
    dec = (tt["start"] >= fit_end).to_numpy()
    scores = {}
    for cfg in grid:
        u = gbm_predict(tt, FF, "U", cadence, cfg, min_rows, predict_from=fit_end, embargo=embargo)
        d = gbm_predict(tt, FF, "D", cadence, cfg, min_rows, predict_from=fit_end, embargo=embargo)
        ok = dec & np.isfinite(u) & np.isfinite(d)
        scores[json_key(cfg)] = float(np.mean(np.abs(u[ok] - tt["U"].to_numpy()[ok]) + np.abs(d[ok] - tt["D"].to_numpy()[ok])))
    best = min(grid, key=lambda c: scores[json_key(c)])
    return best, scores


def json_key(cfg: dict) -> str:
    import json
    return json.dumps(cfg, sort_keys=True)


def linear_predict(t: pd.DataFrame, F: pd.DataFrame, target: str, cadence: str, min_rows: int = 100,
                   predict_from=None) -> np.ndarray:
    """HAR-X: least squares of U or D on the causal features (rows with complete features), weekly refit."""
    X = np.column_stack([np.ones(len(F)), F.to_numpy(float)])
    y = t[target].to_numpy(float)
    okx = np.isfinite(X).all(axis=1)
    out = np.full(len(t), np.nan)
    for g, rows in refit_groups(t, cadence):
        if predict_from is not None and t["start"].iloc[g] < predict_from:
            continue
        train = np.flatnonzero(okx[:g] & np.isfinite(y[:g]))
        if len(train) < min_rows:
            continue
        beta = _ridge(X[train], y[train], lam=1e-6)
        rr = [r for r in rows if okx[r]]
        out[rr] = X[rr] @ beta
    return out
