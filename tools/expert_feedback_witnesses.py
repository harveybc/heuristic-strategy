#!/usr/bin/env python3
"""Witnesses for the ZigZag-as-expert-feedback design (ZIGZAG_EXPERT_FEEDBACK_DESIGN_2026_10_26.md, §6).

Self-contained, synthetic, numpy only, CPU, no market data: a toy market with a planted causal signal, a reversal
ZigZag ladder with pivot CONFIRMATION bars, the per-tick expert quantities (f_open, f_close, remaining, MAE, votes,
f*, NOP kinds), the desired / undesired action sets D(t) / U(t) with the preference strength pi(t), the tick TYPES,
an exact toy engine giving Q(t, a) for every action at every tick (no market impact => exact counterfactuals), the
exact per-tick regret, a preference-weighted trainer, the per-tick ledger, and the seven witnesses W1..W7.

Every function here mirrors a production counterpart named in the design doc (§7 checklist); the witnesses prove the
DEFINITIONS behave as claimed. They do not touch the real pipeline, the store or TEST.
"""
from __future__ import annotations

import dataclasses
import json
import sys

import numpy as np

ACTIONS = ("OPEN_L", "OPEN_S", "HOLD", "CLOSE")
OPEN_L, OPEN_S, HOLD, CLOSE = 0, 1, 2, 3
LADDER = ("opt", "rev20", "rev15", "rev10", "rev8", "rev5")
LADDER_MULT = {"opt": 4.0, "rev20": 3.0, "rev15": 2.5, "rev10": 2.0, "rev8": 1.5, "rev5": 1.0}   # reversal in units of mean sigma
TYPES = ("forced_flat_window", "nop_hard", "at_pivot", "pre_pivot", "post_pivot", "mid_leg")
HINDSIGHT_NAMES = {"f_open", "f_close", "remaining", "mae", "leg_sigma", "votes", "fstar", "pi", "D", "U", "Q", "regret",
                   "pivot", "leg_start", "leg_end", "hit", "net"}


@dataclasses.dataclass(frozen=True)
class Thresholds:
    theta_f: float = 0.5
    theta_mae: float = 0.5
    theta_vol: float = 0.25
    theta_close: float = 0.1
    f_close_take: float = 0.9
    min_votes: int = 1


# ----------------------------------------------------------------------------------------------- synthetic market
def synth_market(n=3000, seed=0, kappa=0.25, rho=0.95, vol=1.0, n_noise=6):
    """Toy H1 market: an observable AR(1) factor z drives the next return (planted causal signal), plus noise
    features. Returns dict with o/h/l/c, causal sigma, observation matrix X (z first) and the factor z."""
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal()
    eps = rng.standard_normal(n)
    r = kappa * np.r_[0.0, z[:-1]] + vol * eps                      # return of bar t uses z_{t-1}: causal
    c = 100.0 + np.cumsum(r)
    o = np.r_[100.0, c[:-1]]
    wick = np.abs(rng.standard_normal(n)) * 0.3 * vol
    h = np.maximum(o, c) + wick
    l = np.minimum(o, c) - wick
    sig = np.full(n, vol)
    for t in range(24, n):                                           # causal rolling std of returns (window 24)
        sig[t] = max(1e-6, np.std(r[t - 24:t]))
    # causal phase features (what the real recipe carries as px.close_loc / hours since the confirmed pivot):
    # distance of the close to the trailing 24-bar min / max in sigma, computed from bars <= t only
    lo24 = np.array([np.min(l[max(0, t - 23):t + 1]) for t in range(n)]); hi24 = np.array([np.max(h[max(0, t - 23):t + 1]) for t in range(n)])
    up_room, dn_room = (hi24 - c) / sig, (c - lo24) / sig
    X = np.c_[z, up_room, dn_room, rng.standard_normal((n, n_noise))]
    return dict(o=o, h=h, l=l, c=c, sigma=sig, X=X, z=z, r=r, cost=np.full(n, COST))


# ----------------------------------------------------------------------------------------------- ZigZag with confirmation
def reversal_pivots(h, l, thr):
    """Reversal ZigZag: a pivot at the running extreme is CONFIRMED at the first bar whose price retraces ``thr`` from
    it. Returns list of (pivot_bar, kind, level, confirmation_bar), kind +1 = peak, -1 = valley."""
    n = len(h)
    piv = []
    mode, ext_i, ext_v = 0, 0, (h[0] + l[0]) / 2
    for t in range(n):
        if mode >= 0 and h[t] > (ext_v if mode == 1 else -np.inf) and mode == 1:
            ext_i, ext_v = t, h[t]
        elif mode <= 0 and l[t] < (ext_v if mode == -1 else np.inf) and mode == -1:
            ext_i, ext_v = t, l[t]
        if mode == 0:
            if h[t] - ext_v >= thr:
                mode, ext_i, ext_v = 1, t, h[t]
            elif ext_v - l[t] >= thr:
                mode, ext_i, ext_v = -1, t, l[t]
            continue
        if mode == 1 and ext_v - l[t] >= thr:
            piv.append((ext_i, 1, ext_v, t)); mode, ext_i, ext_v = -1, t, l[t]
        elif mode == -1 and h[t] - ext_v >= thr:
            piv.append((ext_i, -1, ext_v, t)); mode, ext_i, ext_v = 1, t, h[t]
    return piv


def leg_tables(m, pivots):
    """Per tick and side: f_open, f_close, remaining (sigma), MAE fraction, leg_sigma, leg_end, conf_bar of the end
    pivot. The leg containing t is [s, e) between consecutive pivots; counter-leg quantities are negative."""
    n = len(m["c"]); o, h, l, c, sig = m["o"], m["h"], m["l"], m["c"], m["sigma"]
    out = {k: {1: np.full(n, np.nan), -1: np.full(n, np.nan)} for k in ("f_open", "f_close", "remaining", "mae")}
    leg_sigma, leg_end, conf, leg_start = np.full(n, np.nan), np.full(n, -1), np.full(n, -1), np.full(n, -1)
    for (sb, sk, sv, _), (eb, ek, ev, ec) in zip(pivots[:-1], pivots[1:]):
        leg = abs(ev - sv)
        if leg <= 0:
            continue
        for t in range(sb, min(eb, n - 1)):
            leg_sigma[t], leg_end[t], conf[t], leg_start[t] = leg / sig[t], eb, ec, sb
            for d in (1, -1):
                out["f_open"][d][t] = d * (ev - o[t + 1]) / leg
                out["f_close"][d][t] = d * (c[t] - sv) / leg
                out["remaining"][d][t] = d * (ev - c[t]) / sig[t]
                path = h[t + 1:eb + 1] if d < 0 else l[t + 1:eb + 1]
                adverse = (np.max(path) - o[t + 1]) if d < 0 else (o[t + 1] - np.min(path))
                out["mae"][d][t] = max(0.0, adverse) / leg
    return dict(out, leg_sigma=leg_sigma, leg_end=leg_end, conf=conf, leg_start=leg_start)


def ladder_tables(m, mults=LADDER_MULT):
    base = float(np.mean(m["sigma"]))
    return {r: leg_tables(m, reversal_pivots(m["h"], m["l"], k * base)) for r, k in mults.items()}


def available(tabs, t, period_end):
    """Training-only availability: the END pivot of every ladder leg containing t is confirmed before period_end."""
    return all(0 <= tabs[r]["conf"][t] < period_end for r in tabs)


# ----------------------------------------------------------------------------------------------- expert feedback per tick
def expert_tick(tabs, t, th: Thresholds):
    """votes, f*, kind, D(t), U(t), pi(t) for a FLAT tick (§1)."""
    votes = {1: 0, -1: 0}; fo = {1: -np.inf, -1: -np.inf}
    for r in tabs:
        for d in (1, -1):
            f = tabs[r]["f_open"][d][t]
            if np.isfinite(f):
                fo[d] = max(fo[d], f)
                if f >= th.theta_f:
                    votes[d] += 1
    sides = [d for d in (1, -1) if votes[d] >= th.min_votes]
    if not sides:
        return dict(kind="nop_hard", fstar=np.nan, votes=votes, D={HOLD}, U={OPEN_L, OPEN_S}, pi=1.0, side=0)
    if len(sides) == 2:
        fs = max(fo[1], fo[-1])
        return dict(kind="conflict", fstar=fs, votes=votes, D={HOLD}, U=set(), pi=float(np.clip(1 - fs, 0, 1)), side=0)
    d = sides[0]; fs = fo[d]; opt = tabs["opt"]
    mae, ls = opt["mae"][d][t], opt["leg_sigma"][t]
    if fs < 0:
        return dict(kind="negative", fstar=fs, votes=votes, D={HOLD}, U={OPEN_L if d > 0 else OPEN_S}, pi=float(min(1, abs(fs))), side=d)
    if fs < th.theta_f or (np.isfinite(mae) and mae > th.theta_mae) or (np.isfinite(ls) and ls < th.theta_vol):
        return dict(kind="nop_graded", fstar=fs, votes=votes, D={HOLD}, U={OPEN_L if d > 0 else OPEN_S}, pi=float(np.clip(1 - fs, 0, 1)), side=d)
    consensus = votes[d] / len(tabs)
    return dict(kind="open", fstar=fs, votes=votes, D={OPEN_L if d > 0 else OPEN_S}, U={OPEN_S if d > 0 else OPEN_L, HOLD},
                pi=float(np.clip(consensus * fs, 0, 1)), side=d)


def expert_close(tabs, t, d, th: Thresholds):
    """D/U/pi for a POSITIONED tick of side d, from the optimum leg (§1)."""
    fc, rem_sig = tabs["opt"]["f_close"][d][t], tabs["opt"]["remaining"][d][t]
    if not (np.isfinite(fc) and np.isfinite(rem_sig)):
        return dict(D={HOLD}, U=set(), pi=0.0, f_close=np.nan, rem_frac=np.nan)
    rem_frac = 1.0 - fc
    close = rem_frac < th.theta_close or fc >= th.f_close_take
    pi = float(np.clip(abs(rem_frac - th.theta_close) / max(th.theta_close, 1e-9), 0, 1))
    return dict(D={CLOSE} if close else {HOLD}, U={HOLD} if close else {CLOSE}, pi=pi, f_close=fc, rem_frac=rem_frac)


def tick_type(tabs, t, period_end, flat_window=12, pre=2, post=2):
    """Exactly one type per tick, by priority (§5)."""
    opt = tabs["opt"]
    if t >= period_end - flat_window:
        return "forced_flat_window"
    if all(not np.isfinite(tabs[r]["f_open"][1][t]) for r in tabs):
        return "nop_hard"
    e, s_prev = opt["leg_end"][t], None
    starts = {int(x) for x in np.unique(opt["leg_end"]) if x >= 0}
    if t in starts:
        return "at_pivot"
    if e >= 0 and 0 < e - t <= pre:
        return "pre_pivot"
    if any(0 < t - s <= post for s in starts):
        return "post_pivot"
    return "mid_leg"


# ----------------------------------------------------------------------------------------------- exact toy engine (§2)
COST = 0.05          # round-trip cost in sigma
HORIZON = 48         # forced flat after this many bars (toy)


def _cost(m, t):
    return float(m["cost"][t]) if "cost" in m else COST


def q_open(m, t, d, tp, sl):
    """Exact outcome of opening side d at open[t+1] with levels tp/sl (sigma multiples): (net_sigma, hit, exit_bar).
    Conservative intrabar order: SL before TP. Cost = the tick's round-trip cost (spread blow-ups are per tick)."""
    n = len(m["c"]); e = m["o"][t + 1]; s = m["sigma"][t]; COST_T = _cost(m, t)
    up, dn = e + d * tp * s, e - d * sl * s
    for j in range(t + 1, min(n, t + 1 + HORIZON)):
        hi, lo = m["h"][j], m["l"][j]
        if (d > 0 and lo <= dn) or (d < 0 and hi >= dn):
            return -sl - COST_T, "SL", j
        if (d > 0 and hi >= up) or (d < 0 and lo <= up):
            return tp - COST_T, "TP", j
    j = min(n - 1, t + HORIZON)
    return d * (m["c"][j] - e) / s - COST_T, "TIME", j


def q_table(m, t, cells=((1.0, 1.0), (1.5, 1.5))):
    """Q(t, a) for every flat action: OPEN_L / OPEN_S over cells (best cell kept per side), HOLD = 0."""
    q = {HOLD: 0.0}
    for d, a in ((1, OPEN_L), (-1, OPEN_S)):
        q[a] = max(q_open(m, t, d, tp, sl)[0] for tp, sl in cells) if t + 1 < len(m["c"]) else -np.inf
    return q


def q_positioned(m, t, d, entry, tp_lvl, sl_lvl, entry_sig):
    """Q for a position: HOLD = continue to TP/SL/time from t+1 with the SAME levels; CLOSE = exit at open[t+1]."""
    n = len(m["c"]); COST_T = _cost(m, t); close_now = d * (m["o"][t + 1] - entry) / entry_sig - COST_T / 2
    for j in range(t + 1, min(n, t + 1 + HORIZON)):
        hi, lo = m["h"][j], m["l"][j]
        if (d > 0 and lo <= sl_lvl) or (d < 0 and hi >= sl_lvl):
            return {HOLD: d * (sl_lvl - entry) / entry_sig - COST_T / 2, CLOSE: close_now}
        if (d > 0 and hi >= tp_lvl) or (d < 0 and lo <= tp_lvl):
            return {HOLD: d * (tp_lvl - entry) / entry_sig - COST_T / 2, CLOSE: close_now}
    j = min(n - 1, t + HORIZON)
    return {HOLD: d * (m["c"][j] - entry) / entry_sig - COST_T / 2, CLOSE: close_now}


def regret(q, a):
    return max(q.values()) - q[a]


# ----------------------------------------------------------------------------------------------- policies, ledger (§5, W7)
def simulate(m, policy, t0, t1, cell=None):
    """Sequential replay with ONE position at a time; policy(t, state) -> action. Returns trades and per-tick ledger."""
    cell = cell or getattr(policy, "cell", (1.0, 1.0))
    trades, ledger, pos = [], [], None
    for t in range(t0, t1 - 1):
        if pos is None:
            q = q_table(m, t, cells=(cell,)); a = policy(t, None)
            if a in (OPEN_L, OPEN_S):
                d = 1 if a == OPEN_L else -1; e, s = m["o"][t + 1], m["sigma"][t]
                pos = dict(d=d, entry=e, sig=s, tp=e + d * cell[0] * s, sl=e - d * cell[1] * s, t_open=t)
            ledger.append(dict(t=t, state="flat", action=a, Q=q, regret=regret(q, a)))
        else:
            q = q_positioned(m, t, pos["d"], pos["entry"], pos["tp"], pos["sl"], pos["sig"]); a = policy(t, pos)
            a = a if a in (HOLD, CLOSE) else HOLD
            ledger.append(dict(t=t, state="pos", action=a, Q=q, regret=regret(q, a)))
            if a == CLOSE:
                trades.append(dict(t_open=pos["t_open"], t_close=t + 1, net=q[CLOSE], reason="AGENT")); pos = None
            else:
                hi, lo = m["h"][t + 1], m["l"][t + 1]; d = pos["d"]
                ct = _cost(m, t)
                if (d > 0 and lo <= pos["sl"]) or (d < 0 and hi >= pos["sl"]):
                    trades.append(dict(t_open=pos["t_open"], t_close=t + 1, net=d * (pos["sl"] - pos["entry"]) / pos["sig"] - ct / 2, reason="SL")); pos = None
                elif (d > 0 and hi >= pos["tp"]) or (d < 0 and lo <= pos["tp"]):
                    trades.append(dict(t_open=pos["t_open"], t_close=t + 1, net=d * (pos["tp"] - pos["entry"]) / pos["sig"] - ct / 2, reason="TP")); pos = None
                elif t + 1 - pos["t_open"] >= HORIZON:
                    trades.append(dict(t_open=pos["t_open"], t_close=t + 1, net=d * (m["c"][t + 1] - pos["entry"]) / pos["sig"] - ct / 2, reason="TIME")); pos = None
    return trades, ledger


def ledger_aggregate(ledger, trades):
    return dict(net=float(sum(x["net"] for x in trades)), trades=len(trades), mean_regret=float(np.mean([x["regret"] for x in ledger])) if ledger else 0.0)


# ----------------------------------------------------------------------------------------------- preference-weighted trainer (§3)
def preference_dataset(m, tabs, th, t0, t1, lam=2.0):
    """Rows (x_t, y_t, w_t) for the opener: y = +1 open long desired, -1 open short desired, 0 hold desired;
    w_t = 1 + lam * pi(t) * 1[expert confirmed by reality (D(t) meets argmax Q)]."""
    X, Y, W, F, PI = [], [], [], [], []
    for t in range(t0, t1 - 1):
        if not available(tabs, t, t1):
            continue
        ex = expert_tick(tabs, t, th); q = q_table(m, t)
        best = {a for a, v in q.items() if v >= max(q.values()) - 1e-12}
        confirmed = 1.0 if ex["D"] & best else 0.0
        y = 1 if OPEN_L in ex["D"] else (-1 if OPEN_S in ex["D"] else 0)
        X.append(m["X"][t]); Y.append(y); W.append(1.0 + lam * ex["pi"] * confirmed); F.append(ex["fstar"] if ex["side"] != 0 else 0.0); PI.append(ex["pi"])
    return np.array(X), np.array(Y), np.array(W), np.array(F), np.array(PI)


def fit_softmax(X, Y, W, epochs=600, lr=0.3, l2=1e-3, seed=0):
    """Weighted multinomial logistic regression (3 classes: short / hold / long) by full-batch gradient descent."""
    rng = np.random.default_rng(seed); k = 3
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Xb = np.c_[X, np.ones(len(X))]; Wt = rng.normal(0, 0.01, (Xb.shape[1], k)); T = Y + 1
    onehot = np.eye(k)[T]
    for _ in range(epochs):
        z = Xb @ Wt; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        g = Xb.T @ ((p - onehot) * W[:, None]) / W.sum() + l2 * Wt
        Wt -= lr * g
    return Wt


def predict(Wt, X):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    z = np.c_[X, np.ones(len(X))] @ Wt; z -= z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


def policy_from_model(Wt, X, thr=0.5):
    p = predict(Wt, X); a = np.full(len(X), HOLD)
    a[p[:, 2] > thr] = OPEN_L; a[p[:, 0] > thr] = OPEN_S
    return a, p


def agreement(actions, D_sets, pi):
    hit = np.array([1.0 if a in D else 0.0 for a, D in zip(actions, D_sets)])
    return float((hit * pi).sum() / max(pi.sum(), 1e-12))


def behaviour_curve(score, fstar, bins=5):
    """P(open) per f* bin and the Spearman correlation of the curve with the bin order."""
    edges = np.quantile(fstar, np.linspace(0, 1, bins + 1)); idx = np.clip(np.searchsorted(edges, fstar, side="right") - 1, 0, bins - 1)
    curve = np.array([score[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)])
    ok = np.isfinite(curve); r = _spearman(np.arange(bins)[ok], curve[ok]) if ok.sum() > 2 else np.nan
    return curve, float(r)


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def reward_model_inputs(X_t, action):
    """What the reward model r_phi(x_t, a) receives: the OBSERVATION row and the action one-hot, nothing else (W5)."""
    return {"obs": np.asarray(X_t, float), "action_onehot": np.eye(len(ACTIONS))[action]}


# ----------------------------------------------------------------------------------------------- witnesses W1..W7
def W1_W2(seed=0, n=4000, kappa=0.35, shuffle=False):
    th = Thresholds(); m = synth_market(n=n, seed=seed, kappa=kappa); tabs = ladder_tables(m)
    split = int(n * 0.6)
    X, Y, W, F, PI = preference_dataset(m, tabs, th, 30, split)
    rng = np.random.default_rng(seed + 1)
    if shuffle:                                        # negative control: preferences detached from the ticks
        perm = rng.permutation(len(Y)); Y, W, F, PI = Y[perm], W[perm], F[perm], PI[perm]
    Wt = fit_softmax(X, Y, W)
    Xo, Yo, Wo, Fo, PIo = preference_dataset(m, tabs, th, split, n)
    if shuffle:
        perm = rng.permutation(len(Yo)); Yo, Fo, PIo = Yo[perm], Fo[perm], PIo[perm]
    acts, p = policy_from_model(Wt, Xo)
    D_sets = [{OPEN_L} if y == 1 else ({OPEN_S} if y == -1 else {HOLD}) for y in Yo]
    A = agreement(acts, D_sets, PIo)
    chance = np.mean([agreement(rng.integers(0, 3, len(Yo)), D_sets, PIo) for _ in range(20)])
    mask = Yo != 0
    sides = np.where(Yo[mask] == 1, 1, -1); score = np.where(sides == 1, p[mask, 2], p[mask, 0])
    curve_f, rho_f = behaviour_curve(score, np.abs(Fo[mask]))             # diagnostic only: f* is hindsight timing
    hit = np.array([1.0 if a in D else 0.0 for a, D in zip(acts, D_sets)])
    open_t = Yo != 0                                                        # ticks where the expert wants an OPEN
    curve_pi, rho_pi = behaviour_curve(hit[open_t], PIo[open_t])            # claim: stronger open preference -> more agreement
    nop_agree = float(hit[~open_t].mean()) if (~open_t).any() else np.nan   # exposure: obedience where the expert says HOLD/NOP
    by_type = {"open": float(hit[open_t].mean()), "hold_or_nop": nop_agree}
    return dict(agreement=A, chance=chance, curve_pi=curve_pi.tolist(), spearman_pi=rho_pi, by_type=by_type,
                curve_fstar=curve_f.tolist(), spearman_fstar=rho_f, n_train=int(len(Y)), n_eval=int(len(Yo)))


def W3(seed=0, n=1500):
    m = synth_market(n=n, seed=seed)
    oracle = lambda t, pos: (max(q_table(m, t, cells=((1.0, 1.0),)), key=q_table(m, t, cells=((1.0, 1.0),)).get) if pos is None
                             else max(q_positioned(m, t, pos["d"], pos["entry"], pos["tp"], pos["sl"], pos["sig"]), key=q_positioned(m, t, pos["d"], pos["entry"], pos["tp"], pos["sl"], pos["sig"]).get))
    rng = np.random.default_rng(seed)
    rand = lambda t, pos: int(rng.integers(0, 3)) if pos is None else int(rng.choice([HOLD, CLOSE]))
    _, L_or = simulate(m, oracle, 30, n); _, L_rd = simulate(m, rand, 30, n)
    r_or = np.array([x["regret"] for x in L_or]); r_rd = np.array([x["regret"] for x in L_rd])
    return dict(oracle_max_regret=float(r_or.max()), random_mean_regret=float(r_rd.mean()), random_min=float(r_rd.min()), n=int(len(r_rd)))


def W4(seed=0, n=1500, cut=900):
    """Availability: labels of ticks available by ``cut`` are identical when every bar >= cut is perturbed."""
    th = Thresholds(); m = synth_market(n=n, seed=seed); tabs = ladder_tables(m)
    m2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
    rng = np.random.default_rng(seed + 7)
    for k in ("o", "h", "l", "c"):
        m2[k][cut:] = m2[k][cut:] * (1 + 0.3 * rng.standard_normal(n - cut)) + 5 * rng.standard_normal(n - cut)
    m2["h"][cut:] = np.maximum(m2["h"][cut:], np.maximum(m2["o"][cut:], m2["c"][cut:])); m2["l"][cut:] = np.minimum(m2["l"][cut:], np.minimum(m2["o"][cut:], m2["c"][cut:]))
    tabs2 = ladder_tables(m2)
    checked, diffs = 0, 0
    for t in range(30, cut - 1):
        if not available(tabs, t, cut):
            continue
        a, b = expert_tick(tabs, t, th), expert_tick(tabs2, t, th); checked += 1
        if a["D"] != b["D"] or a["U"] != b["U"] or abs(a["pi"] - b["pi"]) > 1e-12 or a["kind"] != b["kind"]:
            diffs += 1
    return dict(checked=checked, diffs=diffs)


def W5(seed=0, n=600):
    m = synth_market(n=n, seed=seed); t = 300
    inp = reward_model_inputs(m["X"][t], OPEN_L)
    static_ok = not (set(inp) & HINDSIGHT_NAMES)
    m2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
    m2["X"][t + 1:] += 100.0; m2["c"][t + 1:] += 100.0
    inp2 = reward_model_inputs(m2["X"][t], OPEN_L)
    return dict(static_ok=bool(static_ok), perturbation_invariant=bool(np.array_equal(inp["obs"], inp2["obs"]) and np.array_equal(inp["action_onehot"], inp2["action_onehot"])))


def W6(seed=0, n=1500):
    th = Thresholds(); m = synth_market(n=n, seed=seed); tabs = ladder_tables(m)
    counts, missing_D = {k: 0 for k in TYPES}, 0
    for t in range(30, n - 1):
        ty = tick_type(tabs, t, n); counts[ty] += 1
        if not expert_tick(tabs, t, th)["D"]:
            missing_D += 1
    return dict(counts=counts, total=sum(counts.values()), ticks=n - 31, missing_D=missing_D)


def W7(seed=0, n=1500):
    m = synth_market(n=n, seed=seed); rng = np.random.default_rng(seed)
    pol = lambda t, pos: (OPEN_L if m["z"][t] > 0.8 else (OPEN_S if m["z"][t] < -0.8 else HOLD)) if pos is None else (CLOSE if rng.random() < 0.05 else HOLD)
    trades, ledger = simulate(m, pol, 30, n)
    agg = ledger_aggregate(ledger, trades)
    engine_net = float(sum(x["net"] for x in trades))                 # the "engine" total is the trade list itself here
    closes_from_ledger = sum(1 for x in ledger if x["state"] == "pos" and x["action"] == CLOSE)
    return dict(ledger_net=agg["net"], engine_net=engine_net, equal=bool(abs(agg["net"] - engine_net) < 1e-9),
                agent_closes_ledger=closes_from_ledger, agent_closes_trades=sum(1 for x in trades if x["reason"] == "AGENT"), trades=len(trades))


def worked_example():
    """§1 worked example: a 10-bar toy up-leg, valley 100.0 at bar 0, peak 104.0 at bar 9, sigma 1, opens = closes."""
    c = np.array([100.0, 100.5, 101.2, 100.9, 101.8, 102.6, 103.1, 102.8, 103.6, 104.0, 103.2]); o = np.r_[c[0], c[:-1]]
    leg, Ls, Le = 4.0, 100.0, 104.0; th = Thresholds(); rows = []
    for t in range(0, 9):
        f_open = (Le - o[t + 1]) / leg; f_close = (c[t] - Ls) / leg; rem = Le - c[t]
        adverse = max(0.0, o[t + 1] - np.min(c[t + 1:10])); mae = adverse / leg
        close = (1 - f_close) < th.theta_close or f_close + 1e-9 >= th.f_close_take
        rows.append(dict(t=t, open_next=o[t + 1], f_open=round(f_open, 3), f_close=round(f_close, 3), remaining_sigma=round(rem, 2), mae=round(mae, 3),
                         open_desired=bool(f_open >= th.theta_f and mae <= th.theta_mae), close_desired=bool(close)))
    return rows


# ----------------------------------------------------------------------------------------------- §10 hierarchy: coarse = long horizon, fine = short horizon
COARSE, FINE = ("opt", "rev20"), ("rev10", "rev8", "rev5")


def coarse_oracle(tabs, t, d):
    """Long-horizon hindsight oracle (training-only): remaining room to the next COARSE pivot in sigma (signed: < 0
    against the coarse leg), coarse direction (+1/-1/0), bars to the coarse pivot."""
    rooms = [tabs[r]["remaining"][d][t] for r in COARSE if np.isfinite(tabs[r]["remaining"][d][t])]
    if not rooms:
        return dict(room=np.nan, direction=0, tau=np.nan)
    room = float(max(rooms)); e = tabs["opt"]["leg_end"][t]
    f = tabs["opt"]["f_open"][d][t]
    return dict(room=room, direction=int(np.sign(f)) if np.isfinite(f) else 0, tau=float(e - t) if e >= 0 else np.nan)


def fine_oracle(tabs, t, d, h=6):
    """Short-horizon hindsight oracle: P(a FINE pivot AGAINST side d within h bars) = 1 iff a fine leg in favour of d
    ends within h bars; bars to that pivot; fine remaining room; and whether a fine leg in favour of d started within
    the last `h` bars (entry-timing signal)."""
    counter, tau, room, fresh = 0, np.inf, np.nan, 0
    for r in FINE:
        f = tabs[r]["f_open"][d][t]; e = tabs[r]["leg_end"][t]; s0 = tabs[r]["leg_start"][t]
        if not np.isfinite(f) or e < 0:
            continue
        if f >= 0:                                   # fine leg in favour of d: its END is a counter-pivot
            if 0 < e - t <= h:
                counter = 1
            tau = min(tau, e - t)
            room = np.nanmax([room, tabs[r]["remaining"][d][t]])
            if 0 <= t - s0 <= h:
                fresh = 1
    return dict(counter_within_h=counter, tau=tau, room=room, fresh_favour=fresh)


def causal_alarm(m, t, d, entry_bar, thr_sigma=1.0):
    """The SHORT alarm as the strategy uses it: a fine reversal CONFIRMED on the realized path (price retraced
    thr_sigma*sigma from the favourable extreme reached since entry). Reads only bars <= t: no forecast."""
    if t <= entry_bar:
        return False
    seg_h, seg_l = m["h"][entry_bar + 1:t + 1], m["l"][entry_bar + 1:t + 1]
    ext = seg_h.max() if d > 0 else seg_l.min()
    retrace = (ext - m["c"][t]) if d > 0 else (m["c"][t] - ext)
    return bool(retrace >= thr_sigma * m["sigma"][entry_bar])


def causal_room(m, t, d, win=24):
    """Causal proxy of coarse room: distance from the close to the trailing `win`-bar extreme in the direction d."""
    hi = m["h"][max(0, t - win + 1):t + 1].max(); lo = m["l"][max(0, t - win + 1):t + 1].min()
    return float(((hi - m["c"][t]) if d > 0 else (m["c"][t] - lo)) / m["sigma"][t])


def make_strategy(m, tp=1.0, sl=2.0, z_open=0.6, trend_nop=0.5, room_min=0.6, alarm_thr=0.8, room_close=0.4, loss_close=0.5,
                  early_close=True, nop_veto=True, alarm=True, timing=True, obs_ok=None):
    """Toy causal strategy with the hierarchy consumption rule (§10): direction from the long-horizon proxy (z),
    timing from a fresh favourable move (causal), NOP veto on weak signal, early close when the causal fine alarm fires
    AND the causal coarse room is small; engine SL/TP always in force; HOLD on missing observations."""
    state = {}

    def act(t, pos):
        if obs_ok is not None and not obs_ok[t]:
            return HOLD
        z = m["z"][t]
        if pos is None:
            d = 1 if z > 0 else -1
            # NOP veto = causal "no room at any scale": realized 24-bar trend strength below trend_nop sigma-units
            trend = abs(m["c"][t] - m["c"][max(0, t - 24)]) / (m["sigma"][t] * np.sqrt(24.0))
            if nop_veto and trend < trend_nop:
                return HOLD
            if abs(z) < z_open:
                return HOLD
            if timing:
                fresh = (m["c"][t] > m["h"][max(0, t - 6):t].max()) if d > 0 else (m["c"][t] < m["l"][max(0, t - 6):t].min())
                if not fresh:
                    return HOLD
            if causal_room(m, t, d) < room_min:                  # the coarse room must still pay the TP
                return HOLD
            return OPEN_L if d > 0 else OPEN_S
        d = pos["d"]
        if early_close:
            fired = alarm and causal_alarm(m, t, d, pos["t_open"], alarm_thr)
            state["alarms"] = state.get("alarms", 0) + int(fired)
            # "SL before TP" on the realized path: the alarm fired AND (the trade is already under water by
            # loss_close sigma OR the coarse room no longer pays the TP)
            under = d * (m["c"][t] - pos["entry"]) / pos["sig"]
            if fired and (under <= -loss_close or causal_room(m, t, d) < room_close):
                return CLOSE
            if (not alarm) and causal_room(m, t, d) < room_close * 0.5:   # counterfactual "no alarm": room only
                return CLOSE
        return HOLD
    act.state = state; act.cell = (tp, sl)
    return act


def inject_jumps(m, ticks, size_sigma, seed=0, against_position=None, ramp_bars=3):
    """Unforeseen shock: a persistent level move of `size_sigma` AGAINST the position side (or the prevailing
    z-direction) ramped over the 3 bars after each tick."""
    m2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in m.items()}; n = len(m2["c"])
    side = dict(against_position or [])
    for t in ticks:
        d = side.get(t, 1 if m2["z"][t] > 0 else -1)
        total = -d * size_sigma * m2["sigma"][t]
        for j in range(ramp_bars):                            # an adverse ramp over ramp_bars bars (not a single gap)
            b = t + 1 + j
            if b >= n:
                break
            for k in ("o", "h", "l", "c"):
                m2[k][b:] += total / ramp_bars
        for b in range(t + 1, min(n, t + 1 + ramp_bars)):
            m2["h"][b] = max(m2["h"][b], m2["o"][b], m2["c"][b]); m2["l"][b] = min(m2["l"][b], m2["o"][b], m2["c"][b])
    return m2


def inject_regime_flip(m, t0, t1, seed=0):
    """Trend -> range -> trend: inside [t0, t1) the planted drift is removed (returns re-drawn with kappa = 0)."""
    rng = np.random.default_rng(seed + 11); m2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
    r = m2["r"].copy(); r[t0:t1] = rng.standard_normal(t1 - t0) * np.std(m["r"])
    c = 100.0 + np.cumsum(r); shift_o = m2["o"] - m2["c"]; wick_h = m2["h"] - np.maximum(m2["o"], m2["c"]); wick_l = np.minimum(m2["o"], m2["c"]) - m2["l"]
    m2["c"] = c; m2["o"] = np.r_[100.0, c[:-1]]; m2["h"] = np.maximum(m2["o"], m2["c"]) + wick_h; m2["l"] = np.minimum(m2["o"], m2["c"]) - wick_l; m2["r"] = r
    return m2


def inject_spread(m, t0, t1, factor=6.0):
    m2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in m.items()}; m2["cost"][t0:t1] *= factor; return m2


def shock_window_net(trades, windows):
    return float(sum(x["net"] for x in trades if any(a <= x["t_close"] <= b or a <= x["t_open"] <= b for a, b in windows)))


def W8_W10(seed=0, n=3000):
    """Intervention battery on the decision block [1800, 3000): shocks on the SAME path, counterfactual strategies."""
    m = synth_market(n=n, seed=seed, kappa=0.35); t0 = 1800
    # position-conditional shocks: the first 6 entries of the base strategy on the CLEAN path receive a 3-sigma adverse
    # ramp starting 2 bars after the fill (an unforeseen event while positioned); every counterfactual runs on the SAME
    # shocked path
    base0 = make_strategy(m); tr0, _ = simulate(m, base0, t0, n)
    jump_ticks = sorted(x["t_open"] + 1 for x in tr0[:6])
    sides = [(x["t_open"] + 1, 1 if m["z"][x["t_open"]] > 0 else -1) for x in tr0[:6]]
    windows = [(t, t + 24) for t in jump_ticks]
    out = {}
    for name, ramp in (("jumps_fast_3bars", 3), ("jumps_slow_6bars", 6)):
        mj = inject_jumps(m, jump_ticks, size_sigma=3.0, against_position=sides, ramp_bars=ramp)
        full = make_strategy(mj); tr_full, _ = simulate(mj, full, t0, n)
        noec = make_strategy(mj, early_close=False); tr_noec, _ = simulate(mj, noec, t0, n)
        noal = make_strategy(mj, alarm=False); tr_noal, _ = simulate(mj, noal, t0, n)
        loss_full, loss_noec, loss_noal = (shock_window_net(x, windows) for x in (tr_full, tr_noec, tr_noal))
        out[name] = dict(ticks=jump_ticks, ramp_bars=ramp, sigma_per_bar=3.0 / ramp, alarms_fired=int(full.state.get("alarms", 0)),
                         net_full=float(sum(x["net"] for x in tr_full)), net_no_early_close=float(sum(x["net"] for x in tr_noec)),
                         shock_net_full=loss_full, shock_net_no_early_close=loss_noec, shock_net_no_alarm=loss_noal,
                         loss_ratio_full_vs_no_early_close=(loss_full / loss_noec if loss_noec < 0 else np.nan),
                         agent_closes_full=sum(1 for x in tr_full if x["reason"] == "AGENT"))
    mr = inject_regime_flip(m, t0 + 300, t0 + 700, seed=seed); rw = [(t0 + 300, t0 + 700)]
    v = make_strategy(mr); tr_v, _ = simulate(mr, v, t0, n); nv = make_strategy(mr, nop_veto=False); tr_nv, _ = simulate(mr, nv, t0, n)
    inr = lambda tr: [x for x in tr if rw[0][0] <= x["t_open"] < rw[0][1]]
    out["regime_flip"] = dict(window=rw[0], trades_in_range_with_veto=len(inr(tr_v)), trades_in_range_without_veto=len(inr(tr_nv)),
                              net_in_range_with_veto=float(sum(x["net"] for x in inr(tr_v))), net_in_range_without_veto=float(sum(x["net"] for x in inr(tr_nv))),
                              net_full_with_veto=float(sum(x["net"] for x in tr_v)), net_full_without_veto=float(sum(x["net"] for x in tr_nv)))
    ms = inject_spread(m, t0 + 900, t0 + 1000, 6.0); sp = make_strategy(ms); tr_sp, _ = simulate(ms, sp, t0, n); base = make_strategy(m); tr_b, _ = simulate(m, base, t0, n)
    w = [(t0 + 900, t0 + 1000)]
    out["spread_blowup"] = dict(window=w[0], net_window_shocked=shock_window_net(tr_sp, w), net_window_base=shock_window_net(tr_b, w),
                                trades_window_shocked=sum(1 for x in tr_sp if w[0][0] <= x["t_open"] <= w[0][1]))
    obs_ok = np.ones(n, bool); obs_ok[t0 + 400:t0 + 460] = False
    mo = make_strategy(m, obs_ok=obs_ok); tr_o, L_o = simulate(m, mo, t0, n)
    opened_in_gap = sum(1 for x in tr_o if t0 + 400 <= x["t_open"] < t0 + 460)
    out["missing_observations"] = dict(window=(t0 + 400, t0 + 460), entries_during_gap=opened_in_gap, trades=len(tr_o),
                                       protections_still_run=all(x["reason"] in ("SL", "TP", "TIME", "AGENT") for x in tr_o))
    return out


if __name__ == "__main__":
    out = dict(W1=W1_W2(), W2=W1_W2(shuffle=True), W3=W3(), W4=W4(), W5=W5(), W6=W6(), W7=W7(), W8_W10=W8_W10(), example=worked_example())
    json.dump(out, sys.stdout, indent=1, default=float)
