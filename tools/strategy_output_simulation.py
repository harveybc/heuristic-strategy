#!/usr/bin/env python3
"""Strategy simulation with IDEAL and NOISY SIMULATED MODEL OUTPUTS (owner 2026-10-26): the real C3 strategy layer and
the real engine, fed by head outputs of controllable quality derived from the engine-exact outcome tables, so the
strategy's parameters are optimised and its robustness swept BEFORE any model is trained.

Chain, all real code paths: outcome tables (outcomes.entry_table / hold_value) -> simulated heads of declared quality
(binary AUC through a Gaussian latent, regression Spearman through additive noise; Platt / temperature calibration on
the calibration block) -> the C3 upper layer (c3.PolicyC3: NOP veto, per-side EV over the cell grid with the RR rule
and the per-bar band, per-side close thresholds, reverse rule, best-of-day gate) -> c3.freeze (NSGA-II, Deb constraint
domination, F0-F6, require_early_close) on the SELECTION block -> one replay on the DECISION block -> the 15 trading
correctness checks, the sanity fields, the early-close value, the per-trade ledger.

Subcommands: truth (cache the engine truth of a fold), run (one configuration, seeds), report (aggregate, plots, md).
TEST (rows >= 2025-01-01) is never read; the decision block is read once per configuration and seed and ledgered.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import action_contract as ACT  # noqa: E402
from app.short_horizon.expert_guided import c2 as C2  # noqa: E402
from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from app.short_horizon.expert_guided import outcomes as OC  # noqa: E402
from app.short_horizon.expert_guided import stats as ST  # noqa: E402
from app.short_horizon.expert_guided import strategy_full as SF  # noqa: E402

LADDER = (0.50, 0.52, 0.55, 0.58, 0.62, 0.66, 0.70, 0.75, 0.80, 0.90, 1.00)
HEADS = ("open", "net", "nop", "close")            # the four simulated head families (quality set per family)
VARIANTS = ("base", "block", "cost2", "cost3", "spike", "missing10", "missing30", "onesided", "asym", "closer_only", "opener_only", "nop_only",
            "regime", "cal08", "cal12", "lag1", "cap2", "cap4", "cutoff12", "risk05", "risk2", "inv_open", "const_net", "sat")
SCENARIOS = {  # the catalog (owner 2026-10-26), each with what it changes and where
    "base": "all heads at the level; contract gate and the per-bar-cost gate", "block": "noise shared within the day (rho 0.5)",
    "cost2": "every cost x2 (truth and replay)", "cost3": "every cost x3", "spike": "rollover-hour and weekly-open spread x2",
    "missing10": "10 % of observable rows unobservable (HOLD-on-missing)", "missing30": "30 % unobservable",
    "onesided": "short openers at chance (0.5), long at the level", "asym": "short openers 0.10 below the level",
    "closer_only": "openers and NOP at chance, closers at the level", "opener_only": "closers at chance", "nop_only": "only the NOP head informative",
    "regime": "quality drops by 0.10 from the MIDDLE of the decision block (regime shift mid-block)",
    "cal08": "miscalibration: calibrated logits x0.8 (under-confident)", "cal12": "calibrated logits x1.2 (over-confident)",
    "lag1": "delayed signal: every output is the previous observable row's", "cap2": "daily cap 2 entries", "cap4": "daily cap 4 entries",
    "cutoff12": "Friday entry cutoff 12 h before the flat (contract: 24 h)", "risk05": "risk 0.5 % per trade", "risk2": "risk 2 % per trade",
    "inv_open": "openers with INVERTED sign (AUC 1 - level)", "const_net": "the net head constant (0) everywhere", "sat": "probabilities saturated (clipped to {0.01, 0.99})"}
EXPLORATION_PRIOR = (0.85, 0.05, 0.05, 0.05)        # HOLD, LONG, SHORT, CLOSE: exploration visits for the closer references (declared)
SEL_BLOCK, CAL_BLOCK, DEC_BLOCK = "INNER_A", "INNER_A", "INNER_B"   # yearly folds: calibration = selection = the inner semester (declared)
CONTRACT = ROOT / "docs/short_horizon/clean_round/contracts/ACTION_CONTRACT_C3.json"
_ND = NormalDist()


# ----------------------------------------------------------------------------- quality <-> noise
def beta_for_auc(auc: float) -> float:
    """Gaussian latent s = beta*y' + eps (y' in {-1, +1}, eps ~ N(0, 1)): AUC = Phi(beta * 2 / sqrt(2)) / ... exactly
    AUC = Phi(beta / sqrt(2) * 2 / 2 ...); with the +-1 coding the class means differ by 2*beta, so AUC = Phi(2*beta/sqrt(2))
    = Phi(beta*sqrt(2)); beta = inv_Phi(AUC) / sqrt(2). AUC 1.0 -> a large finite beta (separable)."""
    a = min(max(float(auc), 0.0), 1.0)
    if a >= 1.0:
        return 12.0
    if a <= 0.0:
        return -12.0
    return _ND.inv_cdf(a) / math.sqrt(2.0)              # AUC < 0.5 -> a negative beta: an inverted head


def rho_of_level(level: float) -> float:
    """The regression correlation carrying the SAME information as a binary head of AUC = level: for a bivariate normal
    latent, AUC = 1/2 + arcsin(rho)/pi, so rho = sin(pi (level - 1/2)). Level 0.5 -> rho 0 (chance), 0.7 -> 0.588,
    1.0 -> 1 (oracle). Round 1 used rho = level, whose 'chance' floor still carried Spearman 0.48 (the defect the
    oracle / chance floor check exposed; recorded in the report)."""
    return float(min(1.0, max(0.0, math.sin(math.pi * (float(level) - 0.5)))))


def sigma_for_rho(rho: float, sd_true: float) -> float:
    """r = x + eps with eps ~ N(0, sigma^2): corr(r, x) = rho  ->  sigma = sd(x) * sqrt(1/rho^2 - 1). rho 1 -> 0."""
    r = min(max(float(rho), 0.0), 1.0)
    if r <= 1e-6:
        return float(sd_true) * 1e3                             # chance: the truth is drowned (corr ~ 0.001)
    return 0.0 if r >= 1.0 else float(sd_true) * math.sqrt(1.0 / r ** 2 - 1.0)


def auc_of(score, y) -> float:
    s, y = np.asarray(score, float), np.asarray(y, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks on ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return C2.spearman(a[ok], b[ok])


class Noise:
    """Deterministic noise per (key, seed) with an optional within-day shared component (block noise, rho_day).
    ``side_swap`` (the mirror twin of a world) draws side s from the key of side 1 - s, so the simulated heads of the
    mirror are the exact mirror of the original's (behaviour matrix W14)."""

    def __init__(self, seed: int, rho_day: float = 0.0, side_swap: bool = False):
        self.seed, self.rho, self.side_swap = int(seed), float(rho_day), bool(side_swap)

    @staticmethod
    def _swap(key):
        k = tuple(key)
        if k and k[0] in ("open", "net", "mae") and len(k) >= 3:
            return k[:2] + (1 - int(k[2]),) + k[3:]
        if k and k[0] == "close" and len(k) >= 2:
            inner = k[1:]
            if inner and inner[0] == "vis" and len(inner) >= 3:
                return ("close", "vis", inner[1], 1 - int(inner[2])) + tuple(inner[3:])
            if len(inner) >= 2 and isinstance(inner[1], (int, np.integer)) and inner[1] in (-1, 1):   # pos_key: (bar, direction, entry_bar, tp, sl)
                return ("close", inner[0], -int(inner[1])) + tuple(inner[2:])
        return k

    def _rng(self, *key):
        key = self._swap(key) if self.side_swap else tuple(key)
        h = hashlib.sha256(json.dumps([self.seed] + [str(k) for k in key]).encode()).digest()
        return np.random.default_rng(int.from_bytes(h[:8], "little"))

    def draw(self, key, day, n: int = 1) -> np.ndarray:
        e = self._rng(*key).standard_normal(n)
        if self.rho > 0:
            d = self._rng("day", day, key[0]).standard_normal(n)
            e = self.rho * d + math.sqrt(1.0 - self.rho ** 2) * e
        return e


# ----------------------------------------------------------------------------- the engine truth of a fold
@dataclasses.dataclass
class Truth:
    fold: str
    cells: list
    ref: int
    rows: np.ndarray            # observable rows (the table rows)
    period: np.ndarray
    day: np.ndarray             # bar-day index per row (for block noise)
    hit3: np.ndarray            # (n, 2) hit at the reference cell: 0 TP, 1 SL, 2 TIME, -1 unknown
    net: np.ndarray             # (n, 2, cells) true engine net in sigma (NaN unknown)
    nop: np.ndarray             # (n,) 1 = no allowed cell on either side beats the cost (reality-based NOP)
    e_time: list
    cost: float
    cap: np.ndarray
    lo: np.ndarray
    xvis: dict                  # exploration visits per block: row, bar, side, remaining, sl_first, key (hold-value truth)
    stamp: dict
    mae: np.ndarray | None = None   # (n, 2, cells) engine maximum adverse excursion in sigma (NaN unknown); the MAE head's truth


def load_fold(fold: str, admitted: str, csv_dir: str):
    from tools.expert_guided_e1t import Ctx
    from tools.expert_guided_e3m import effective_contracts
    ds = DG.load(admitted, csv_dir, scaling="fixed_train", contract=fold)
    cx = Ctx(ds)
    SCFG, _ = effective_contracts("sac", 2.0)
    return cx, SCFG


def with_variant(SCFG, xv):
    return dataclasses.replace(SCFG, variant=None if xv.is_legacy() else xv)


def pos_key(pos, bar: int) -> tuple:
    return (int(bar), int(pos.direction), int(pos.entry_bar), round(float(pos.tp), 7), round(float(pos.sl), 7))


def mirrored_policy(f):
    """The same exploration draws with LONG and SHORT swapped: a mirrored world explores the mirrored visits, so the
    closer references of the mirror twin are the exact mirror of the original's (side-symmetry witness)."""
    swap = {E.LONG: E.SHORT, E.SHORT: E.LONG}

    def g(bar, r, st, env):
        a, tp, sl = f(bar, r, st, env)
        return swap.get(a, a), tp, sl
    return g


def build_truth(cx, SCFG, fold: str, cells, contract: dict, explore_seed: int = 0, progress=None, only_periods=None, mirror_exploration: bool = False) -> Truth:
    """``only_periods``: restrict the entry table to the rows of these periods (Stage A synthetic worlds score the
    selection block only); the exploration visits and the NOP truth follow the same restriction. ``mirror_exploration``
    swaps the exploration's LONG / SHORT draws (the mirror twin of a world)."""
    cells = [tuple(c) for c in cells]
    rows_sel = None
    if only_periods:
        rows_sel = np.flatnonzero(cx.obs_ok & np.isin(cx.ds.period, list(only_periods)))
    et = OC.entry_table(cx, SCFG, cells=cells, progress=progress, rows=rows_sel)
    n = len(cx.ds.row_bar)
    ref = C3.cell_index(tuple(contract["outcomes"]["ref_cell"]), cells)
    rows = np.flatnonzero(cx.obs_ok)
    e_time = C3.e_time_of(cx, et, "TRAIN", tuple(contract["outcomes"]["ref_cell"]), cells)
    cal_rows = np.flatnonzero(cx.obs_ok & (cx.ds.period == CAL_BLOCK))
    from tools.expert_guided_e3m import COST_SIGMA                  # the declared round-trip allowance E3-M uses for the EV rule
    cost = C3.cost_sigma_of(cx, SCFG, float(COST_SIGMA), cal_rows)
    cap, lo = C3.cap_arrays(cx, SCFG)
    hit3 = et["hit"][:, :, ref].astype(np.int64)
    net = et["net_sigma"].astype(float)
    mae_t = et["mae"].astype(float) if "mae" in et else None
    # reality-based NOP: no allowed cell on either side has a true net above the cost (A2-8: open only with positive EV)
    nop = np.ones(n, np.int64)
    for r in (rows if rows_sel is None else rows_sel):
        best = -np.inf
        for si in (0, 1):
            x = net[r, si]
            if np.all(np.isfinite(x)):
                ok = C3.allowed_cells(x, cells, cost, cap=float(cap[r]), lo=float(lo[r]))
                if ok.any():
                    best = max(best, float(np.max(x[ok])))
        nop[r] = 0 if best > cost else 1
    v = cx.ds.bars.index.as_unit("ns").asi8
    day = np.full(n, -1, np.int64)
    rb = np.asarray(cx.ds.row_bar, np.int64)
    okb = rb >= 0
    day[okb] = (v[rb[okb]] // (24 * 3600 * 10 ** 9)).astype(np.int64)
    xvis = {}
    for p in (CAL_BLOCK, SEL_BLOCK, DEC_BLOCK):
        if p in xvis or (only_periods and p not in only_periods):
            continue
        xpol = C3.prior_random_cells(cx, list(EXPLORATION_PRIOR), explore_seed, cells)
        vis = OC.visit_positions(cx, p, mirrored_policy(xpol) if mirror_exploration else xpol, SCFG)
        xt = OC.exit_table(cx, SCFG, vis)
        keys = [pos_key(pos, bar) for _, bar, pos in vis if OC.make_env(cx, SCFG).first <= bar < OC.make_env(cx, SCFG).last] if False else None
        xvis[p] = {"row": xt["row"], "bar": xt["bar"], "side": xt["side"], "remaining": xt["remaining"], "sl_first": xt["sl_first"],
                   "censored": xt["censored"], "exit_bar": xt["exit_bar"]}
    return Truth(fold, [list(c) for c in cells], ref, rows, np.asarray(cx.ds.period), day, hit3, net, nop, e_time, cost, cap, lo, xvis,
                 {"entry_table": et["stamp"], "refused": {k: int(v_) for k, v_ in et["refused"].items()}, "explore_seed": explore_seed,
                  "exploration_prior": list(EXPLORATION_PRIOR)}, mae=mae_t)


def save_truth(path: Path, t: Truth):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rows=t.rows, period=np.asarray(t.period).astype("U16"), day=t.day, hit3=t.hit3, net=t.net, nop=t.nop, cap=t.cap, lo=t.lo,
                        meta=np.asarray(json.dumps({"fold": t.fold, "cells": t.cells, "ref": t.ref, "e_time": t.e_time, "cost": t.cost, "stamp": t.stamp})),
                        **{f"x__{p}__{k}": v for p, d in t.xvis.items() for k, v in d.items()})


def load_truth(path: Path) -> Truth:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    xvis = {}
    for k in z.files:
        if k.startswith("x__"):
            _, p, f = k.split("__", 2)
            xvis.setdefault(p, {})[f] = z[k]
    return Truth(meta["fold"], meta["cells"], int(meta["ref"]), z["rows"], np.asarray(z["period"]).astype(str), z["day"], z["hit3"], z["net"], z["nop"], meta["e_time"],
                 float(meta["cost"]), z["cap"], z["lo"], xvis, meta["stamp"])


# ----------------------------------------------------------------------------- simulated heads
@dataclasses.dataclass
class Quality:
    open: float = 1.0       # AUC of the TP-first / SL-first indicators at the reference cell (both sides)
    net: float = 1.0        # Spearman of the predicted net per cell with the true net
    nop: float = 1.0        # AUC of the NOP head
    close: float = 1.0      # AUC of the closer's SL-first head; the remaining-value regression uses the same rho
    open_s: float | None = None      # one-sided / asymmetric variants: the short side's opener quality (None = same as ``open``)
    decision_drop: float = 0.0       # regime variant: quality after the MIDDLE of the decision block = quality - drop
    logit_scale: float = 1.0         # calibration drift: calibrated logits x scale (0.8 under-, 1.2 over-confident)
    lag: int = 0                     # delayed signal: outputs shifted by this many observable rows
    saturate: bool = False           # probabilities clipped to {0.01, 0.99}
    const_net: bool = False          # the net regression head constant 0

    def side(self, si: int) -> float:
        return self.open if si == 0 or self.open_s is None else self.open_s


class SimCloser:
    """The closer table object the C3 policy reads: ``value(r, bar, pos)`` -> (remaining, P(SL first)) of the open order at
    that decision, simulated from the ENGINE's own hold-vs-close truth (outcomes.hold_value) with the declared quality;
    ``rows_values(m, rows, G)`` -> the calibration references on the exploration visits of that side."""

    def __init__(self, cx, SCFG, truth: Truth, q: Quality, noise: Noise, cal: dict, dec_rows_mask):
        self.cx, self.SCFG, self.t, self.q, self.noise, self.cal = cx, SCFG, truth, q, noise, cal
        self.env = OC.make_env(cx, SCFG)
        self.cache = {}
        self.dec = dec_rows_mask
        self.visit_lookup = {}
        for p, d in truth.xvis.items():
            for i in range(len(d["row"])):
                self.visit_lookup[(int(d["row"][i]), int(d["side"][i]))] = (float(d["remaining"][i]), int(d["sl_first"][i]), int(d["bar"][i]))

    def _quality(self, r):
        q = self.q.close
        if self.dec is not None and self.dec[r]:
            q = max(0.5, q - self.q.decision_drop)
        return q

    def simulate(self, rem_true: float, sl_first: int, key, day, r):
        q = self._quality(r)
        e = self.noise.draw(("close",) + tuple(key), day, 2)
        z = beta_for_auc(q) * (1.0 if sl_first else -1.0) + e[0]
        psl = float(_drift(ACT.platt(np.array([z]), self.cal["psl"]["a"], self.cal["psl"]["b"]), self.q)[0])
        rem = float(rem_true + sigma_for_rho(q, self.cal["rem_sd"]) * e[1])
        return rem, psl, z

    def rows_values(self, m, rows, G):
        si = C3.SIDE_OF[m]
        rem, psl = [], []
        for r in np.asarray(rows, np.int64):
            tr = self.visit_lookup.get((int(r), si))
            if tr is None:
                rem.append(np.nan)
                psl.append(np.nan)
                continue
            x, p, _ = self.simulate(tr[0], tr[1], ("vis", int(r), si), int(self.t.day[r]), int(r))
            rem.append(x)
            psl.append(p)
        return np.asarray(rem, float), np.asarray(psl, float)

    def value(self, r, bar, pos):
        k = pos_key(pos, bar)
        if k not in self.cache:
            o = OC.hold_value(self.env, pos, int(bar))
            self.cache[k] = (float(o["remaining"]), int(o["sl_first"]))
        rem_true, slf = self.cache[k]
        x, p, _ = self.simulate(rem_true, slf, k, int(self.t.day[r]), int(r))
        return x, p

    def restricted(self, keep):
        return self

    def with_ens_cal(self, cal):
        return self


def fit_closer_cal(truth: Truth, q: Quality, noise: Noise) -> dict:
    """Platt map of the closer's SL-first latent and the sd of the remaining value, both on the calibration-block visits."""
    d = truth.xvis[CAL_BLOCK]
    ok = np.isfinite(d["remaining"]) & ~d["censored"]
    y = d["sl_first"][ok].astype(int)
    z = np.array([beta_for_auc(q.close) * (1.0 if yy else -1.0) + noise.draw(("close", "vis", int(r), int(s)), int(truth.day[r]), 2)[0]
                  for r, s, yy in zip(d["row"][ok], d["side"][ok], y)])
    a, b = ACT.fit_platt(z, y) if len(np.unique(y)) > 1 else (1.0, 0.0)
    return {"psl": {"a": float(a), "b": float(b)}, "rem_sd": float(np.std(d["remaining"][ok])) if ok.sum() > 2 else 1.0,
            "rows": int(ok.sum()), "auc_latent": auc_of(z, y)}


def simulate_table(cx, SCFG, truth: Truth, q: Quality, seed: int, rho_day: float = 0.0, regime_dec_mask=None, nets_from=None, side_swap: bool = False) -> tuple:
    """The C3 table on every observable row: ``net`` (n, 2, cells), ``p3`` (n, 2, 3), ``pn`` (n,), ``closer``, ``cap``, ``lo``;
    plus the achieved qualities on the calibration block and the decision block (AUC / Spearman / MAE)."""
    noise = Noise(seed, rho_day, side_swap)
    n, nc = len(truth.period), len(truth.cells)
    rows = truth.rows
    cal = truth.period == CAL_BLOCK
    dec_all = truth.period == DEC_BLOCK
    dec_rows = rows[dec_all[rows]]
    dec = np.zeros(n, bool)                     # the regime shift applies from the MIDDLE of the decision block (declared)
    if len(dec_rows):
        dec[dec_rows[len(dec_rows) // 2:]] = True
    t = {"net": np.full((n, 2, nc), np.nan), "p3": np.full((n, 2, 3), np.nan), "pn": np.full(n, np.nan), "cap": truth.cap, "lo": truth.lo}
    if truth.mae is not None:
        t["mae"] = np.full((n, 2, nc), np.nan)
    lat = {"tp": np.full((n, 2), np.nan), "sl": np.full((n, 2), np.nan), "nop": np.full(n, np.nan)}
    sd_net = float(np.nanstd(truth.net[cal])) if np.isfinite(truth.net[cal]).any() else 1.0
    mu_net = float(np.nanmean(truth.net[cal])) if np.isfinite(truth.net[cal]).any() else 0.0   # the calibration-block naive (the regression's shrinkage target)
    qual = {}
    for r in rows:
        day = int(truth.day[r])
        qo = {si: q.side(si) for si in (0, 1)}
        qn, qnp = rho_of_level(q.net), q.nop                  # the net regression's rho from its LEVEL (same information as an AUC-level binary)
        if dec[r] and q.decision_drop > 0:
            qo = {si: max(0.5, v - q.decision_drop) for si, v in qo.items()}
            qn, qnp = rho_of_level(max(0.5, q.net - q.decision_drop)), max(0.5, qnp - q.decision_drop)
        for si in (0, 1):
            h = int(truth.hit3[r, si])
            e = noise.draw(("open", int(r), si), day, 2)
            b = beta_for_auc(qo[si])
            lat["tp"][r, si] = (b * (1.0 if h == OC.HIT_TP else -1.0) if h >= 0 else 0.0) + e[0]
            lat["sl"][r, si] = (b * (1.0 if h == OC.HIT_SL else -1.0) if h >= 0 else 0.0) + e[1]
            x = truth.net[r, si]
            en = noise.draw(("net", int(r), si), day, nc + 1)
            if q.const_net:
                t["net"][r, si, :] = 0.0                        # a constant net head: the RR rule alone decides the cell
            elif np.all(np.isfinite(x)):
                # a trained MSE head outputs the conditional mean E[x | r] = mu + rho^2 (r - mu) of its noisy view r = x + eps:
                # corr(E[x|r], x) = rho, MAE / naive = sqrt(1 - rho^2); chance (rho 0) returns the naive mean, the oracle returns x
                r_noisy = x + sigma_for_rho(qn, sd_net) * (0.5 * en[0] + math.sqrt(0.75) * en[1:])   # shared row + cell noise
                t["net"][r, si, :] = mu_net + (qn ** 2) * (r_noisy - mu_net)
            else:
                t["net"][r, si, :] = -1.0                       # no eligible entry at this row: the strategy must not open here
            if truth.mae is not None:
                xm = truth.mae[r, si]
                if np.all(np.isfinite(xm)):
                    em = noise.draw(("mae", int(r), si), day, nc)
                    sdm = float(np.nanstd(truth.mae[cal])) if np.isfinite(truth.mae[cal]).any() else 1.0
                    mum = float(np.nanmean(truth.mae[cal])) if np.isfinite(truth.mae[cal]).any() else 0.0
                    rn = xm + sigma_for_rho(qn, sdm) * em
                    t["mae"][r, si, :] = np.maximum(0.0, mum + (qn ** 2) * (rn - mum))
        e = noise.draw(("nop", int(r)), day, 1)
        lat["nop"][r] = beta_for_auc(qnp) * (1.0 if truth.nop[r] else -1.0) + e[0]
    # calibration on the calibration block only (Platt per binary latent), then the 3-class probabilities
    cr = rows[cal[rows]]
    cal_doc = {}
    for si in (0, 1):
        h = truth.hit3[cr, si]
        okr = cr[h >= 0]
        hh = truth.hit3[okr, si]
        for name, lab in (("tp", OC.HIT_TP), ("sl", OC.HIT_SL)):
            y = (hh == lab).astype(int)
            z = lat[name][okr, si]
            a, b = ACT.fit_platt(z, y) if len(np.unique(y)) > 1 else (1.0, 0.0)
            cal_doc[f"{name}_{si}"] = {"a": float(a), "b": float(b), "rows": int(len(y)), "auc_cal": auc_of(z, y)}
            lat[f"p_{name}_{si}"] = _drift(ACT.platt(np.nan_to_num(lat[name][:, si]), a, b), q)
        ptp, psl = lat[f"p_tp_{si}"], lat[f"p_sl_{si}"]
        pt = np.clip(1.0 - ptp - psl, 0.02, 1.0)
        p3 = np.stack([ptp, psl, pt], axis=1)
        p3 = p3 / p3.sum(axis=1, keepdims=True)
        t["p3"][rows, si, :] = p3[rows]
    yn = truth.nop[cr]
    a, b = ACT.fit_platt(lat["nop"][cr], yn) if len(np.unique(yn)) > 1 else (1.0, 0.0)
    t["pn"][rows] = _drift(ACT.platt(np.nan_to_num(lat["nop"][rows]), a, b), q)
    cal_doc["nop"] = {"a": float(a), "b": float(b), "auc_cal": auc_of(lat["nop"][cr], yn)}
    # achieved qualities per block
    for blk, mask in (("calibration", cal), ("decision", dec)):
        br = rows[mask[rows]]
        d = {}
        for si in (0, 1):
            h = truth.hit3[br, si]
            ok = h >= 0
            d[f"auc_tp_first_side{si}"] = auc_of(lat["tp"][br[ok], si], (h[ok] == OC.HIT_TP).astype(int))
            d[f"auc_sl_first_side{si}"] = auc_of(lat["sl"][br[ok], si], (h[ok] == OC.HIT_SL).astype(int))
            x, y = t["net"][br, si, :], truth.net[br, si, :]
            okn = np.all(np.isfinite(y), axis=1)
            d[f"spearman_net_side{si}"] = spearman(x[okn].ravel(), y[okn].ravel())
            err = np.abs(x[okn] - y[okn])
            naive = np.abs(y[okn] - np.nanmean(truth.net[cr, si, :]))
            d[f"mae_net_side{si}"] = float(err.mean()) if okn.any() else float("nan")
            d[f"mae_over_naive_net_side{si}"] = float(err.mean() / naive.mean()) if okn.any() and naive.mean() > 0 else float("nan")
        d["auc_nop"] = auc_of(lat["nop"][br], truth.nop[br])
        qual[blk] = d
    if q.lag > 0:                               # delayed signal: the outputs at observable row i are those of row i - lag (same side)
        for k in ("net", "p3", "pn"):
            arr = t[k][rows]
            t[k][rows[q.lag:]] = arr[:-q.lag]
    ccal = fit_closer_cal(truth, q, noise)
    t["closer"] = SimCloser(cx, SCFG, truth, q, noise, ccal, dec if q.decision_drop > 0 else None)
    return t, {"calibration_maps": cal_doc, "closer_calibration": ccal, "achieved": qual, "sd_net_cal": sd_net}


def _drift(p: np.ndarray, q: Quality) -> np.ndarray:
    """Calibration drift (logit x scale) and saturation, applied AFTER the Platt map (the strategy consumes these)."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    if q.logit_scale != 1.0:
        z = np.log(p / (1 - p)) * float(q.logit_scale)
        p = 1.0 / (1.0 + np.exp(-z))
    if q.saturate:
        p = np.where(p >= 0.5, 0.99, 0.01)
    return p


# ----------------------------------------------------------------------------- one configuration
def contract_for_sim(path=CONTRACT) -> dict:
    ac = json.loads(Path(path).read_text())
    ac = copy.deepcopy(ac)
    ac["strategy"]["optimizer"]["final_escalated"] = dict(ac["strategy"]["optimizer"]["final"])   # declared: no escalation in the sweep
    return ac


class PolicyC3PerBar(C3.PolicyC3):
    """Declared strategy variant measured by the sweeps (the sweeps' own fix, A4-2 revisited): the EV rule pays the
    decision bar's OWN round-trip cost (``table['cost_row']``: the hour's spread + slippage + commission in sigma plus
    the allowance) instead of one median cost, and the gate observes 0 signals (the first qualifying open is placed),
    so a day is never lost to the secretary rule while rollover-hour signals pay their real spread."""

    def __init__(self, table, strategy, cells=C3.CELLS, e_time=(0.0, 0.0), cost: float = 0.0, observe_bars: int = 0):
        super().__init__(table, strategy, cells, e_time, cost, observe_bars)

    def open_decision(self, r, with_score: bool = False):
        c = self.t.get("cost_row")
        if c is not None and np.isfinite(c[r]):
            saved, self.cost = self.cost, float(c[r])
            try:
                return super().open_decision(r, with_score)
            finally:
                self.cost = saved
        return super().open_decision(r, with_score)


POLICIES = {"contract": C3.PolicyC3, "perbar": PolicyC3PerBar}
FAMILIES = {"c3": None, "cell_param": ("cell", "param"), "param_param": ("param", "param"), "param_risk": ("param", "risk"), "cell_risk": ("cell", "risk")}
FULL_OPTIMIZER = {"population": 48, "generations": 24}      # declared budget of the FULL parametric layer (more genes than C3's 8)
FULL_PRIMARY = "sharpe_plus_return_plus_activity"             # the owner's composite (normalised coefficients) is the FULL layer's primary fitness


class PolicyFullPerBar(SF.PolicyFull):
    """PolicyFull with the per-bar cost in the EV (the ``perbar`` gate's cost rule)."""

    def levels(self, r, si, bar, env):
        c = self.t.get("cost_row")
        if c is not None and np.isfinite(c[r]):
            saved, self.cost = self.cost, float(c[r])
            try:
                return super().levels(r, si, bar, env)
            finally:
                self.cost = saved
        return super().levels(r, si, bar, env)


def contract_full(AC: dict, tp_family: str, vol_family: str) -> dict:
    """The C3 contract with the FULL gene space: coefficient genes added, the optimiser budget and the primary fitness
    declared; the activity-CI and side rules unchanged (still a v1.3-style block)."""
    ac = copy.deepcopy(AC)
    st = ac["strategy"]
    st["genes"] = SF.gene_bounds(tp_family, vol_family)
    st["gene_space"] = SF.GENE_SPACE_FULL
    st["optimizer"]["final"] = dict(FULL_OPTIMIZER)
    st["optimizer"]["final_escalated"] = dict(FULL_OPTIMIZER)
    st["fitness"]["default"] = FULL_PRIMARY
    st["tp_family"], st["vol_family"] = tp_family, vol_family
    st["formulas"] = {"tp_cell": "TP_d, SL_d = argmax_cell predicted_net_d within the band [lo_t, cap_t]; TP<SL only if net > cost + 0.10 sigma",
                      "tp_param": "TP_d = sigma_t (a_d + b_d q_d), SL_d = sigma_t (c_d + e_d r_d); q_d = P(TP first|d), r_d = P(SL first|d); bounded to [lo_t, cap_t] and env tp/sl bounds",
                      "vol_risk": "v_d = min(1, risk_pct balance / (SL distance units_full)) (engine risk sizing)",
                      "vol_param": "v_d = clip(alpha_d + beta_d P(TP first|d) - gamma_d P(NOP) - delta_d P(SL first|d), 0, 1); v < v_min -> no order",
                      "units": "units = v f balance rel_volume leverage / price (integer, leverage <= 1:50; rel_volume 0.10)"}
    ac["identity"] = ac["identity"] + f"_full_{tp_family}_{vol_family}"
    return ac


def scfg_full(SCFG, vol_family: str):
    """The engine config of a volume family: risk -> the engine's risk sizing (full_f3); param -> notional sizing scaled by v."""
    xv = SCFG.variant if SCFG.variant is not None else C3.FULL_F3
    if vol_family == "param":
        return dataclasses.replace(SCFG, variant=dataclasses.replace(xv, sizing_mode="notional_share"))
    return dataclasses.replace(SCFG, variant=dataclasses.replace(xv, sizing_mode="risk"))


def policy_cls(gate: str):
    if gate not in POLICIES:
        raise SystemExit(f"GATE_UNKNOWN:{gate}")
    return POLICIES[gate]


def row_costs(cx, SCFG, allowance: float) -> np.ndarray:
    """Per dataset row: the bar's own spread (schedule hour) + two market slippages + commission, in causal sigma, plus
    the declared allowance (the EV's cost term of the ``perbar`` policy)."""
    n = len(cx.ds.row_bar)
    out = np.full(n, np.nan)
    env = OC.make_env(cx, SCFG)
    pr = SCFG.profile
    sig = np.asarray(cx.ds.sigma, float)
    for r in range(n):
        b = int(cx.ds.row_bar[r])
        if 0 <= b < len(env.v) - 1 and np.isfinite(sig[b + 1]) and sig[b + 1] > 0:
            k = b + 1
            si = int(env.seg_of[k])
            sp = pr.spread_at(int(env.v[k]), int(env.mo[si]))
            out[r] = (sp + 2 * pr.slippage_market + 2 * pr.commission_per_lot_side / pr.lot_units) / sig[k] + float(allowance)
    return out


def score_fn(cx, SCFG, cells, e_time, cost, AC, period, sig, min_opw, Pol=C3.PolicyC3, roll=ACT.rollout_with_trades):
    def score_table(view, strat):
        m, trades = roll(cx, period, Pol(view, strat, cells, e_time, cost)(cx), SCFG)
        return ACT.window_components(m, trades, sig, SCFG.capital, min_opw, **ACT.score_kwargs(AC))
    return score_table


def policy_factory(Pol, scfg):
    """A PolicyFull factory bound to the engine config (the risk family needs the sizing rule)."""
    return lambda view, strat, cells, e_time, cost: Pol(view, strat, cells, e_time, cost, scfg=scfg)


def closer_src_of(truth: Truth) -> dict:
    d = truth.xvis[CAL_BLOCK]
    ok = np.isfinite(d["remaining"]) & ~d["censored"]
    out = {}
    for m in C3.CLOSERS:
        si = C3.SIDE_OF[m]
        out[m] = {"rows": d["row"][ok & (d["side"] == si)], "G": None}
    return out


def evaluate(cx, SCFG, table, strat, cells, e_time, cost, period, sig, AC, capital, with_checks=True, ledger_rows=None, Pol=C3.PolicyC3,
             roll=ACT.rollout_with_trades) -> dict:
    from tools import expert_guided_trading_report as TR
    pol = Pol(table, strat, cells, e_time, cost)
    m, trades = roll(cx, period, pol(cx), SCFG)
    comp = ACT.window_components(m, trades, sig, capital, float(AC["strategy"]["calibration_min_trades_per_week"]), **ACT.score_kwargs(AC))
    gate = pol.last_gate.stats if pol.last_gate is not None else {}
    m0, t0 = roll(cx, period, Pol(table, strat.no_early_close(), cells, e_time, cost)(cx), SCFG)
    out = {"metrics": {k: m[k] for k in ("net_profit_usd", "sharpe_daily_sqrt365", "max_drawdown_pct", "trades", "win_rate", "profit_factor",
                                        "orders_per_week", "n_entries_long", "n_entries_short", "n_close_TP", "n_close_SL", "n_close_FORCED_FLAT",
                                        "n_close_AGENT_CLOSE", "n_close_REVERSAL", "valid") if k in m},
           "components": comp, "sanity": ACT.seed_sanity(m), "best_of_day": gate,
           "no_early_close": {k: m0[k] for k in ("net_profit_usd", "sharpe_daily_sqrt365", "max_drawdown_pct", "trades") if k in m0},
           "early_close_value_usd": float(m["net_profit_usd"]) - float(m0["net_profit_usd"])}
    tr = ACT.trade_returns_sigma(trades, sig)
    tr = tr[np.isfinite(tr)]
    by_day = {}
    a, b = cx.ds.meta["periods"][period] if isinstance(period, str) else period
    for t_ in trades:
        by_day.setdefault(str(t_["exit_time"])[:10], []).append(float(t_["net_usd"]) / max(int(t_["units"]), 1) / float(sig[int(t_["entry_bar_abs"])]))
    v = cx.ds.bars.index.as_unit("ns").asi8
    inb = (v >= DG._utc_ns(a)) & (v < DG._utc_ns(b))
    days = sorted({str(d)[:10] for d in cx.ds.bars.index[inb]})
    ex = ST.day_block_bootstrap_expectancy({d: by_day.get(d, []) for d in days}) if len(tr) >= 2 else {"expectancy": float("nan"), "low": float("nan"), "high": float("nan"), "n_trades": int(len(tr))}
    out["expectancy_sigma"] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in ex.items()}
    side = {}
    for d_, name in ((1, "buy"), (-1, "sell")):
        ts = [t_ for t_ in trades if int(t_["direction"]) == d_]
        side[name] = {"n": len(ts), "win_rate": float(np.mean([t_["net_usd"] > 0 for t_ in ts])) if ts else float("nan"),
                      "net_usd": float(sum(t_["net_usd"] for t_ in ts))}
    out["sides"] = side
    if with_checks and trades:
        ev = E.BarEnv(cx.ds.bars, cx.ds.cal, cx.ds.sigma, cx.ds.sigma_rule, a, b, SCFG)
        df = TR.trade_frame(trades, ev)
        chk = TR.correctness(df, ev, SCFG, capital)
        out["correctness"] = {k: {"status": "PASS" if v["failed"] == 0 else "FAIL", "checked": v["checked"], "failed": v["failed"]} for k, v in chk.items()}
        out["correctness_failed"] = int(sum(v["failed"] for v in chk.values()))
        if ledger_rows is not None:
            ledger_rows.extend(df.assign(period=str(period)).to_dict("records"))
    elif with_checks:
        out["correctness"], out["correctness_failed"] = {}, 0
    return out


def quarter_tags(cx, period: str) -> list:
    """The decision block split into calendar quarters, each tagged by realized volatility (tercile over the quarters of
    2021-2023) and by trend efficiency |sum r| / sum |r| (tercile): the market-regime catalog (trending / ranging, high / low vol)."""
    import pandas as pd
    idx = cx.ds.bars.index
    v = idx.as_unit("ns").asi8
    r = np.r_[np.nan, np.diff(np.log(cx.ds.bars.CLOSE.to_numpy(float)))]
    allq = []
    for y in (2021, 2022, 2023):
        for qn in range(4):
            a = pd.Timestamp(f"{y}-{3 * qn + 1:02d}-01", tz="UTC")
            b = a + pd.offsets.QuarterBegin(startingMonth=1)
            m = (v >= a.value) & (v < b.value)
            rr = r[m]
            rr = rr[np.isfinite(rr)]
            if len(rr) < 100:
                continue
            allq.append({"start": str(a)[:10], "end": str(b)[:10], "vol": float(np.std(rr) * math.sqrt(24 * 260)), "trend": float(abs(rr.sum()) / np.abs(rr).sum())})
    vols = np.array([x["vol"] for x in allq])
    tr = np.array([x["trend"] for x in allq])
    tv, tt = np.quantile(vols, [1 / 3, 2 / 3]), np.quantile(tr, [1 / 3, 2 / 3])
    tagged = [dict(x, vol_tercile=("low" if x["vol"] <= tv[0] else "high" if x["vol"] > tv[1] else "mid"),
                   trend_tercile=("ranging" if x["trend"] <= tt[0] else "trending" if x["trend"] > tt[1] else "mixed")) for x in allq]
    if period is None:
        return tagged                                   # every quarter of 2021-2023 (donors for the regime splice)
    a0, b0 = cx.ds.meta["periods"][period]
    return [x for x in tagged if x["start"] >= str(a0) and x["end"] <= str(b0)]


def evaluate_quarters(cx, SCFG, table, strat, cells, e_time, cost, sig, AC, Pol, qtags, roll=ACT.rollout_with_trades) -> list:
    """The frozen strategy replayed on each quarter of the decision block (sub-periods of the same year; no new choice)."""
    out = []
    for x in qtags:
        d = evaluate(cx, SCFG, table, strat, cells, e_time, cost, (x["start"], x["end"]), sig, AC, SCFG.capital, False, None, Pol, roll)
        m = d["metrics"]
        out.append({**x, "sharpe": m.get("sharpe_daily_sqrt365"), "net_usd": m.get("net_profit_usd"), "trades": m.get("trades"), "opw": m.get("orders_per_week"),
                    "expectancy": d["expectancy_sigma"].get("expectancy"), "max_dd_pct": m.get("max_drawdown_pct")})
    return out


def apply_stage_a(AC: dict, stage_a: dict | None) -> dict:
    """Stage B: the gene bounds restricted to the Stage-A robust box (box ∩ declared bounds; a degenerate box keeps the
    declared bound); the Stage-A optimum is recorded for the out-of-block comparison."""
    if not stage_a:
        return AC
    box = (stage_a.get("robust_region") or {}).get("box") or {}
    ac = copy.deepcopy(AC)
    g = ac["strategy"]["genes"]
    for k in list(g):
        if k in box:
            lo, hi = float(box[k][0]), float(box[k][1])
            lo, hi = max(lo, float(g[k][0])), min(hi, float(g[k][1]))
            if hi - lo < 1e-9:
                hi = min(float(g[k][1]), lo + 0.02 * (float(g[k][1]) - float(g[k][0])) + 1e-9)
            g[k] = [lo, hi]
    ac["strategy"]["stage_a"] = {"family": stage_a.get("family"), "robust_points": (stage_a.get("robust_region") or {}).get("points"),
                                 "best_mean_genes": (stage_a.get("best_mean") or {}).get("genes"), "evaluations": stage_a.get("evaluations")}
    return ac


def run_config(cx, SCFG, truth: Truth, AC, q: Quality, seeds, variant: str, out_dir: Path, fold: str, freeze_per_seed: bool = True,
               missing_frac: float = 0.0, ledger: bool = True, gate: str = "contract", family: str = "c3", stage_a: dict | None = None) -> dict:
    fam = FAMILIES[family]
    if fam is None:
        Pol, roll, freeze_fn = policy_cls(gate), ACT.rollout_with_trades, None
        AC = apply_stage_a(AC, stage_a)
    else:
        tpf, vf = fam
        SCFG = scfg_full(SCFG, vf)
        AC = apply_stage_a(contract_full(AC, tpf, vf), stage_a)
        Pol = policy_factory(PolicyFullPerBar if gate == "perbar" else SF.PolicyFull, SCFG)
        roll = SF.rollout_with_volume
    cost_row = row_costs(cx, SCFG, truth.cost - C3.cost_sigma_of(cx, SCFG, 0.0, np.flatnonzero(cx.obs_ok & (cx.ds.period == CAL_BLOCK)))) if gate == "perbar" else None
    cells = [tuple(c) for c in truth.cells]
    sig = np.asarray(cx.ds.sigma, float)
    min_opw = float(AC["strategy"]["calibration_min_trades_per_week"])
    cal_rows = np.flatnonzero(cx.obs_ok & (cx.ds.period == CAL_BLOCK))
    sel_rows = np.flatnonzero(cx.obs_ok & (cx.ds.period == SEL_BLOCK))
    view_rows = np.union1d(cal_rows, sel_rows)
    rho_day = 0.5 if variant == "block" else 0.0
    res = {"fold": fold, "variant": variant, "gate": gate, "family": family, "quality": dataclasses.asdict(q), "seeds": list(seeds), "contract": AC["identity"],
           "exec_config": {"max_entries_per_day": SCFG.max_entries_per_day, "sizing_mode": getattr(SCFG.variant, "sizing_mode", None), "risk_pct": getattr(SCFG.variant, "risk_pct", None),
                           "cutoff_h": getattr(SCFG.variant, "entry_cutoff_hours_before_flat", None), "percentual_order_size": SCFG.percentual_order_size, "rel_volume": SCFG.sizing.rel_volume},
           "formulas": (AC["strategy"].get("formulas") if fam else None), "genes_bounds": AC["strategy"]["genes"], "primary_fitness": AC["strategy"]["fitness"]["default"],
           "stage_a": AC["strategy"].get("stage_a"),
           "exec_variant": (AC.get("exec_variant") or {}).get("identity"), "blocks": {"calibration": CAL_BLOCK, "selection": SEL_BLOCK, "decision": DEC_BLOCK},
           "cost_sigma": truth.cost, "e_time": truth.e_time, "per_seed": []}
    obs_backup = cx.obs_ok.copy()
    qtags = quarter_tags(cx, DEC_BLOCK)
    res["quarter_tags"] = qtags
    try:
        for seed in seeds:
            t0 = time.time()
            if missing_frac > 0:                                       # HOLD-on-missing: 10 % of observable rows unobservable (declared)
                rng = np.random.default_rng(1000 + int(seed))
                cx.obs_ok = obs_backup & (rng.random(len(obs_backup)) >= missing_frac)
            table, sim = simulate_table(cx, SCFG, truth, q, int(seed), rho_day)
            if cost_row is not None:
                table["cost_row"] = cost_row
            score_table = score_fn(cx, SCFG, cells, truth.e_time, truth.cost, AC, SEL_BLOCK, sig, min_opw, Pol, roll)
            rec = {"seed": int(seed), "simulation": sim}
            try:
                if fam is None:
                    fr = C3.freeze(table, view_rows, score_table, AC, int(seed), "final", AC["strategy"]["fitness"]["default"], closer_src_of(truth),
                                   cells, truth.e_time, truth.cost, cal_rows)
                    strat = C2.strategy_of(fr)
                else:
                    fr = freeze_full(table, view_rows, cal_rows, score_table, AC, int(seed), closer_src_of(truth), cells, truth.e_time, truth.cost, tpf, vf,
                                     float(getattr(SCFG.variant, "risk_pct", 0.01)))
                    strat = SF.strategy_from_dict(fr["strategy"]) if fr.get("strategy") else None
                    if strat is None:
                        raise SystemExit(f"STRATEGY_NO_SELECTABLE_POINT:{fr.get('note')}")
                    ap = out_dir / "archive" / f"{fold}_{variant}_{gate}_{family}_q{q.open:.2f}_s{seed}.json.gz"
                    ap.parent.mkdir(parents=True, exist_ok=True)
                    import gzip
                    with gzip.open(ap, "wt") as fh:
                        json.dump({k: r["points"] for k, r in fr["runs"].items()}, fh, default=float)
                    rec["archive"] = str(ap.relative_to(out_dir))
                rec["strategy"] = fr["strategy"]
                rec["chosen_calibration"] = {k: fr["chosen"].get(k) for k in ("sharpe", "lcb", "opw", "opw_ci_lb", "side_max_share", "agent_closes", "dd", "trades") if fr["chosen"]}
                rec["gene_map"] = {k: v for k, v in (fr.get("gene_map") or {}).items() if k != "references"}
                rec["chosen_genes"] = (fr.get("chosen") or {}).get("genes")
                rec["front_points"] = fr.get("fitness_diversity", {}).get("selectable_front_points")
                rec["by_fitness_chosen_genes"] = {f: (v["chosen"] or {}).get("genes") for f, v in fr["by_fitness"].items()}
                bench = early_close_benefit(cx, table, strat, SCFG, cells, truth.e_time, truth.cost, Pol, roll=roll)
                rec["early_close_benefit_calibration"] = {k: bench[k] for k in ("mean_daily_diff_usd", "lower_bound", "beneficial")}
                if not bench["beneficial"]:
                    strat = strat.no_early_close()
                    rec["early_close_disabled"] = True
                rec["strategy_effective"] = strat.to_dict()
                if seed == seeds[0]:                                   # one-at-a-time gene sensitivity on the SELECTION block (declared)
                    rec["sensitivity"] = (gene_sensitivity_full(cx, SCFG, table, fr, truth, AC, cells, sig, min_opw, view_rows, cal_rows, Pol, roll, tpf, vf)
                                          if fam else gene_sensitivity(cx, SCFG, table, fr, truth, AC, cells, sig, min_opw, view_rows, cal_rows, Pol=Pol))
                sa_genes = ((AC["strategy"].get("stage_a") or {}).get("best_mean_genes"))
                if sa_genes:                                           # Stage-A optimum on the real blocks: the out-of-block gain / loss of Stage B
                    view = C2.restrict(table, view_rows)
                    ref = C3.closer_reference_sides(table, closer_src_of(truth))
                    if fam is None:
                        gm_ = C3.gene_map(view, cal_rows, ref, cells, truth.e_time, truth.cost)
                        st_a = C3.StrategyC3.from_genes(gm_.to_absolute(sa_genes), "nop_veto" in sa_genes)
                    else:
                        gm_ = SF.gene_map_full(view, cal_rows, ref, cells, truth.e_time, truth.cost, AC["strategy"]["genes"])
                        st_a = SF.StrategyFull.from_genes(gm_.to_absolute(sa_genes), "nop_veto" in sa_genes, tpf, vf, float(getattr(SCFG.variant, "risk_pct", 0.01)))
                    if rec.get("early_close_disabled"):
                        st_a = st_a.no_early_close()
                    da = evaluate(cx, SCFG, table, st_a, cells, truth.e_time, truth.cost, DEC_BLOCK, sig, AC, SCFG.capital, False, None, Pol, roll)
                    sa_ = evaluate(cx, SCFG, table, st_a, cells, truth.e_time, truth.cost, SEL_BLOCK, sig, AC, SCFG.capital, False, None, Pol, roll)
                    rec["stage_a_optimum"] = {"genes": sa_genes, "selection": {k: sa_["metrics"].get(k) for k in ("sharpe_daily_sqrt365", "net_profit_usd", "orders_per_week", "trades")} | {"expectancy": sa_["expectancy_sigma"].get("expectancy")},
                                              "decision": {k: da["metrics"].get(k) for k in ("sharpe_daily_sqrt365", "net_profit_usd", "orders_per_week", "trades", "max_drawdown_pct", "valid")}
                                                          | {"expectancy": da["expectancy_sigma"].get("expectancy"), "expectancy_low": da["expectancy_sigma"].get("low"), "expectancy_high": da["expectancy_sigma"].get("high")}}
                # (owner) which fitness: every variant's chosen point, scored ONCE on the decision block (and on selection, for the gap)
                byf = {}
                for f, v in fr["by_fitness"].items():
                    if not v.get("strategy"):
                        byf[f] = {"valid": False, "note": v.get("note")}
                        continue
                    stf = (SF.strategy_from_dict if fam else C2.strategy_from_dict)(v["strategy"])
                    if rec.get("early_close_disabled"):
                        stf = stf.no_early_close()
                    dsel = evaluate(cx, SCFG, table, stf, cells, truth.e_time, truth.cost, SEL_BLOCK, sig, AC, SCFG.capital, False, None, Pol, roll)
                    ddec = evaluate(cx, SCFG, table, stf, cells, truth.e_time, truth.cost, DEC_BLOCK, sig, AC, SCFG.capital, False, None, Pol, roll)
                    byf[f] = {"valid": True, "genes": (v["chosen"] or {}).get("genes"), "nop_veto_kept": v.get("nop_veto_kept"),
                              "selection": {k: dsel["metrics"].get(k) for k in ("sharpe_daily_sqrt365", "net_profit_usd", "orders_per_week", "trades")} | {"expectancy": dsel["expectancy_sigma"].get("expectancy")},
                              "decision": {k: ddec["metrics"].get(k) for k in ("sharpe_daily_sqrt365", "net_profit_usd", "orders_per_week", "trades", "max_drawdown_pct", "valid")}
                                          | {"expectancy": ddec["expectancy_sigma"].get("expectancy"), "expectancy_low": ddec["expectancy_sigma"].get("low"), "lcb": ddec["components"].get("lcb")}}
                rec["by_fitness"] = byf
                rec["quarters"] = evaluate_quarters(cx, SCFG, table, strat, cells, truth.e_time, truth.cost, sig, AC, Pol, qtags, roll)
                rows_led = [] if ledger else None
                rec["selection"] = evaluate(cx, SCFG, table, strat, cells, truth.e_time, truth.cost, SEL_BLOCK, sig, AC, SCFG.capital, True, None, Pol, roll)
                rec["decision"] = evaluate(cx, SCFG, table, strat, cells, truth.e_time, truth.cost, DEC_BLOCK, sig, AC, SCFG.capital, True, rows_led, Pol, roll)
                if ledger and rows_led:
                    import pandas as pd
                    p = out_dir / "ledger" / f"{fold}_{variant}_{gate}_{family}_q{q.open:.2f}_s{seed}.csv.gz"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows_led).to_csv(p, index=False, compression="gzip")
                    rec["ledger"] = str(p.relative_to(out_dir))
            except SystemExit as e:
                rec["refused"] = str(e)[:400]
            rec["seconds"] = round(time.time() - t0, 1)
            res["per_seed"].append(rec)
            print(json.dumps({"fold": fold, "variant": variant, "family": family, "q": q.open, "seed": seed, "sec": rec["seconds"],
                              "dec_sharpe": (rec.get("decision") or {}).get("metrics", {}).get("sharpe_daily_sqrt365"),
                              "dec_net": (rec.get("decision") or {}).get("metrics", {}).get("net_profit_usd"),
                              "refused": rec.get("refused")}), flush=True)
    finally:
        cx.obs_ok = obs_backup
    return res


def early_close_benefit(cx, table, strat, SCFG, cells, e_time, cost, Pol, spec=C3.EARLY_CLOSE_BENEFIT, roll=ACT.rollout_with_trades) -> dict:
    """c3.early_close_benefit with the policy class of the gate variant (same statistic, same bootstrap)."""
    p = spec["block"]
    m1, t1 = roll(cx, p, Pol(table, strat, cells, e_time, cost)(cx), SCFG)
    m0, t0 = roll(cx, p, Pol(table, strat.no_early_close(), cells, e_time, cost)(cx), SCFG)
    d1, d0 = C3.daily_pnl(t1), C3.daily_pnl(t0)
    days = sorted(set(d1) | set(d0))
    diff = np.array([d1.get(d, 0.0) - d0.get(d, 0.0) for d in days])
    b = spec["bootstrap"]
    lb = C3.mean_diff_lower_bound(diff, b["block_days"], b["draws"], b["seed"], b["level"])
    return {"block": p, "days": int(len(days)), "mean_daily_diff_usd": float(diff.mean()) if len(diff) else float("nan"), "lower_bound": lb,
            "beneficial": bool(np.isfinite(lb) and lb > 0)}


def gene_sensitivity(cx, SCFG, table, fr: dict, truth: Truth, AC, cells, sig, min_opw, view_rows, cal_rows, delta: float = 0.1, Pol=C3.PolicyC3) -> list:
    """Each quantile gene of the frozen point moved by -delta / +delta (clipped to its bounds), the others fixed: the
    selection-block Sharpe and trades/week deltas (the strategy's local sensitivity, never read on the decision block)."""
    ch = fr.get("chosen") or {}
    genes = dict(ch.get("genes") or {})
    if not genes:
        return []
    view = C2.restrict(table, view_rows)
    ref = C3.closer_reference_sides(table, closer_src_of(truth))
    gm = C3.gene_map(view, cal_rows, ref, cells, truth.e_time, truth.cost)
    veto = bool(fr.get("nop_veto_kept"))
    bounds = AC["strategy"]["genes"]
    score_table = score_fn(cx, SCFG, cells, truth.e_time, truth.cost, AC, SEL_BLOCK, sig, min_opw, Pol)
    base = score_table(view, C3.StrategyC3.from_genes(gm.to_absolute(genes), veto))
    out = []
    for g in genes:
        row = {"gene": g, "value": float(genes[g]), "sharpe_base": base["sharpe"], "opw_base": base["opw"]}
        for sign, name in ((-1, "minus"), (1, "plus")):
            g2 = dict(genes)
            lo, hi = bounds[g]
            g2[g] = float(min(max(genes[g] + sign * delta, lo), hi))
            c = score_table(view, C3.StrategyC3.from_genes(gm.to_absolute(g2), veto))
            row[f"sharpe_{name}_delta"] = float(c["sharpe"] - base["sharpe"]) if math.isfinite(c["sharpe"]) and math.isfinite(base["sharpe"]) else float("nan")
            row[f"opw_{name}_delta"] = float(c["opw"] - base["opw"])
            row[f"valid_{name}"] = bool(c["valid"])
        out.append(row)
    return out


def freeze_full(table, view_rows, cal_rows, score_table, AC, seed, closer_src, cells, e_time, cost, tpf, vf, risk_pct) -> dict:
    """The FULL layer frozen on the selection view: C3's quantile map for the thresholds + copied coefficient genes;
    NSGA-II over every gene (``FULL_OPTIMIZER``); no combinatorial feasibility probe (gene space declared non-Q)."""
    view = C2.restrict(table, view_rows)
    ref = C3.closer_reference_sides(table, closer_src or {})
    bounds = AC["strategy"]["genes"]
    gm = SF.gene_map_full(view, cal_rows, ref, cells, e_time, cost, bounds)
    make = lambda g, veto, _gm=gm: SF.StrategyFull.from_genes(_gm.to_absolute(g), veto, tpf, vf, risk_pct)  # noqa: E731
    fr = ACT.freeze_strategy(table, view_rows, score_table, AC, seed, "final", AC["strategy"]["fitness"]["default"], gene_names=tuple(bounds),
                             make_strategy=make, restrict_fn=C2.restrict)
    fr["gene_map"] = gm.describe()
    return fr


def gene_sensitivity_full(cx, SCFG, table, fr, truth, AC, cells, sig, min_opw, view_rows, cal_rows, Pol, roll, tpf, vf, delta: float = 0.1) -> list:
    ch = fr.get("chosen") or {}
    genes = dict(ch.get("genes") or {})
    if not genes:
        return []
    view = C2.restrict(table, view_rows)
    ref = C3.closer_reference_sides(table, closer_src_of(truth))
    bounds = AC["strategy"]["genes"]
    gm = SF.gene_map_full(view, cal_rows, ref, cells, truth.e_time, truth.cost, bounds)
    veto = bool(fr.get("nop_veto_kept"))
    rp = float(getattr(SCFG.variant, "risk_pct", 0.01))
    score_table = score_fn(cx, SCFG, cells, truth.e_time, truth.cost, AC, SEL_BLOCK, sig, min_opw, Pol, roll)
    base = score_table(view, SF.StrategyFull.from_genes(gm.to_absolute(genes), veto, tpf, vf, rp))
    out = []
    for g in genes:
        lo, hi = bounds[g]
        step = delta * (hi - lo) if (hi - lo) > 1.0 else delta
        row = {"gene": g, "value": float(genes[g]), "sharpe_base": base["sharpe"], "opw_base": base["opw"]}
        for sign, name in ((-1, "minus"), (1, "plus")):
            g2 = dict(genes)
            g2[g] = float(min(max(genes[g] + sign * step, lo), hi))
            c = score_table(view, SF.StrategyFull.from_genes(gm.to_absolute(g2), veto, tpf, vf, rp))
            row[f"sharpe_{name}_delta"] = float(c["sharpe"] - base["sharpe"]) if math.isfinite(c["sharpe"]) and math.isfinite(base["sharpe"]) else float("nan")
            row[f"opw_{name}_delta"] = float(c["opw"] - base["opw"])
            row[f"valid_{name}"] = bool(c["valid"])
        out.append(row)
    return out


def quality_for(variant: str, level: float) -> Quality:
    q = Quality(level, level, level, level)
    if variant == "onesided":
        q.open_s = 0.5
    elif variant == "asym":
        q.open_s = max(0.5, level - 0.10)
    elif variant == "closer_only":
        q.open, q.net, q.nop = 0.5, 0.5, 0.5
    elif variant == "opener_only":
        q.close = 0.5
    elif variant == "nop_only":
        q.open, q.net, q.close = 0.5, 0.5, 0.5
    elif variant == "regime":
        q.decision_drop = 0.10
    elif variant == "cal08":
        q.logit_scale = 0.8
    elif variant == "cal12":
        q.logit_scale = 1.2
    elif variant == "lag1":
        q.lag = 1
    elif variant == "inv_open":
        q.open = 1.0 - level
    elif variant == "const_net":
        q.const_net = True
    elif variant == "sat":
        q.saturate = True
    return q


def scfg_for(variant: str, SCFG):
    """The engine configuration of a scenario (cost / spread / cap / cutoff / risk are DECLARED execution variants)."""
    xv = SCFG.variant if SCFG.variant is not None else C3.FULL_F3
    if variant == "cost2":
        return dataclasses.replace(SCFG, profile=SCFG.profile.scaled(2.0))
    if variant == "cost3":
        return dataclasses.replace(SCFG, profile=SCFG.profile.scaled(3.0))
    if variant == "spike":                      # rollover-hour spread doubled (the spike the A4 audit measured at hour 0)
        sch = tuple((a, b, (x * 2.0 if a == 0 else x)) for a, b, x in SCFG.profile.spread_schedule)
        return dataclasses.replace(SCFG, profile=dataclasses.replace(SCFG.profile, spread_schedule=sch, weekly_open_spread=SCFG.profile.weekly_open_spread * 2.0))
    if variant == "cap2":
        return dataclasses.replace(SCFG, max_entries_per_day=2)
    if variant == "cap4":
        return dataclasses.replace(SCFG, max_entries_per_day=4)
    if variant == "cutoff12":
        return dataclasses.replace(SCFG, variant=dataclasses.replace(xv, entry_cutoff_hours_before_flat=12.0))
    if variant == "risk05":
        return dataclasses.replace(SCFG, variant=dataclasses.replace(xv, risk_pct=0.005))
    if variant == "risk2":
        return dataclasses.replace(SCFG, variant=dataclasses.replace(xv, risk_pct=0.02))
    return SCFG


TRUTH_OF = {"cost2": "cost2", "cost3": "cost3", "spike": "spike", "cutoff12": "cutoff12"}   # scenarios whose engine truth differs from base


# ----------------------------------------------------------------------------- CLI
def cmd_truth(a):
    AC = contract_for_sim(a.contract)
    cx, SCFG = load_fold(a.fold, a.admitted, a.csv_dir)
    SCFG = with_variant(SCFG, C3.FULL_F3)
    for v in a.variants.split(","):
        S = scfg_for(v, SCFG)
        t = build_truth(cx, S, a.fold, AC["outcomes"]["cells"], AC, progress=lambda i, n: print(json.dumps({"truth": v, "rows": f"{i}/{n}"}), flush=True))
        save_truth(Path(a.out) / f"TRUTH_{a.fold}_{v}.npz", t)
        print(json.dumps({"truth": v, "fold": a.fold, "cost_sigma": t.cost, "e_time": t.e_time, "nop_share_cal": float(t.nop[t.rows][t.period[t.rows] == CAL_BLOCK].mean()),
                          "visits": {p: int(len(d["row"])) for p, d in t.xvis.items()}, "refused": t.stamp["refused"]}), flush=True)


def cmd_run(a):
    AC = contract_for_sim(a.contract)
    cx, SCFG = load_fold(a.fold, a.admitted, a.csv_dir)
    SCFG = with_variant(SCFG, C3.FULL_F3)
    truth_variant = TRUTH_OF.get(a.variant, "base")
    truth = load_truth(Path(a.truth_dir) / f"TRUTH_{a.fold}_{truth_variant}.npz")
    S = scfg_for(a.variant, SCFG)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    for lv in [float(x) for x in a.levels.split(",")]:
        q = quality_for(a.variant, lv)
        p = out / ((f"RUN_{a.fold}_{a.variant}_{a.gate}_q{lv:.2f}.json" if a.family == "c3" else f"RUN_{a.fold}_{a.variant}_{a.gate}_{a.family}_q{lv:.2f}.json").replace("RUN_", f"RUN_{a.tag}" if a.tag else "RUN_"))
        if p.exists() and not a.force:
            print(json.dumps({"skip": str(p)}), flush=True)
            continue
        miss = {"missing10": 0.10, "missing30": 0.30}.get(a.variant, 0.0)
        stage_a = json.loads(Path(a.bounds_from).read_text()) if a.bounds_from else None
        res = run_config(cx, S, truth, AC, q, seeds, a.variant, out, a.fold, missing_frac=miss, ledger=not a.no_ledger, gate=a.gate, family=a.family, stage_a=stage_a)
        res["scenario"] = SCENARIOS.get(a.variant, "")
        res["tag"] = a.tag
        res["level"] = lv
        p.write_text(json.dumps(res, indent=1, default=float))


def _agg(runs: list) -> list:
    rows = []
    for r in runs:
        for s in r["per_seed"]:
            d = s.get("decision") or {}
            m, c, ex = d.get("metrics", {}), d.get("components", {}), d.get("expectancy_sigma", {})
            sel = (s.get("selection") or {}).get("metrics", {})
            rows.append({"fold": r["fold"], "variant": r["variant"], "gate": r.get("gate", "contract"), "family": r.get("family", "c3"), "tag": r.get("tag", ""), "level": r["level"], "seed": s["seed"], "refused": s.get("refused"),
                         "auc_cal": s.get("simulation", {}).get("achieved", {}).get("calibration", {}).get("auc_tp_first_side0"),
                         "auc_dec": s.get("simulation", {}).get("achieved", {}).get("decision", {}).get("auc_tp_first_side0"),
                         "spearman_dec": s.get("simulation", {}).get("achieved", {}).get("decision", {}).get("spearman_net_side0"),
                         "mae_dec": s.get("simulation", {}).get("achieved", {}).get("decision", {}).get("mae_net_side0"),
                         "mae_over_naive_dec": s.get("simulation", {}).get("achieved", {}).get("decision", {}).get("mae_over_naive_net_side0"),
                         "sel_sharpe": sel.get("sharpe_daily_sqrt365"), "sel_net": sel.get("net_profit_usd"),
                         "dec_sharpe": m.get("sharpe_daily_sqrt365"), "dec_net": m.get("net_profit_usd"), "dec_dd": m.get("max_drawdown_pct"),
                         "dec_trades": m.get("trades"), "dec_opw": m.get("orders_per_week"), "dec_win": m.get("win_rate"), "dec_pf": m.get("profit_factor"),
                         "dec_long": m.get("n_entries_long"), "dec_short": m.get("n_entries_short"), "dec_agent_closes": c.get("agent_closes"),
                         "dec_exp": ex.get("expectancy"), "dec_exp_lo": ex.get("low"), "dec_exp_hi": ex.get("high"), "dec_lcb": c.get("lcb"),
                         "dec_early_close_value": d.get("early_close_value_usd"), "buy_win": (d.get("sides") or {}).get("buy", {}).get("win_rate"),
                         "sell_win": (d.get("sides") or {}).get("sell", {}).get("win_rate"), "checks_failed": d.get("correctness_failed"),
                         "early_close_disabled": bool(s.get("early_close_disabled", False)), "front_points": s.get("front_points"),
                         "genes": json.dumps((s.get("strategy") or {}))})
    return rows


def cmd_report(a):
    import pandas as pd
    runs = [json.loads(p.read_text()) for p in sorted(Path(a.runs).glob("RUN_*.json"))]
    df = pd.DataFrame(_agg(runs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "SWEEP_ROWS.csv", index=False)
    from tools import strategy_output_simulation_report as R
    ints = [json.loads(p.read_text()) for p in sorted(Path(a.interventions).glob("INT_*.json"))] if a.interventions else []
    sa = {}
    cat, done = [], {}
    if a.hier:
        for p in sorted(Path(a.hier).glob("STAGE_A_*.json")):
            if "_leaves" in p.name or "archive" in p.name:
                continue
            d = json.loads(p.read_text())
            sa[d["family"]] = d
        cp = Path(a.hier) / "CATALOG.json"
        cat = json.loads(cp.read_text()) if cp.exists() else []
        for p in sorted(Path(a.hier).glob("STAGE_A_*_leaves.json")):
            for x in json.loads(p.read_text()):
                done[x["id"]] = done.get(x["id"], 0) + 1
    R.write(df, runs, out, a.docs, ints, sa, cat, done, Path(a.contracts) if a.contracts else None)
    print(json.dumps({"rows": int(len(df)), "out": str(out)}))


def stop_requested(out_path) -> bool:
    """Owner STOP flag (2026-10-27): a file named STOP in the work root (any ancestor of the output path, three levels up)
    makes every NEW job of the lane exit before doing anything; running jobs finish their current file."""
    from pathlib import Path as _P
    p = _P(str(out_path)).resolve()
    for q in [p] + list(p.parents)[:3]:
        if (q / "STOP").exists():
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("truth", "run"):
        s = sub.add_parser(name)
        s.add_argument("--fold", required=True)
        s.add_argument("--admitted", required=True)
        s.add_argument("--csv-dir", required=True)
        s.add_argument("--contract", default=str(CONTRACT))
        s.add_argument("--out", required=True)
    sub.choices["truth"].add_argument("--variants", default="base,cost2,cost3,spike,cutoff12")
    r = sub.choices["run"]
    r.add_argument("--truth-dir", required=True)
    r.add_argument("--variant", default="base", choices=VARIANTS)
    r.add_argument("--levels", default=",".join(str(x) for x in LADDER))
    r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--gate", default="contract", choices=list(POLICIES))
    r.add_argument("--family", default="c3", choices=list(FAMILIES))
    r.add_argument("--bounds-from", default=None, help="Stage-A JSON (STAGE_A_<family>.json): restrict the genes to its robust box (Stage B)")
    r.add_argument("--tag", default="")
    r.add_argument("--force", action="store_true")
    r.add_argument("--no-ledger", action="store_true")
    rp = sub.add_parser("report")
    rp.add_argument("--runs", required=True)
    rp.add_argument("--out", required=True)
    rp.add_argument("--docs", required=True)
    rp.add_argument("--interventions", default=None)
    rp.add_argument("--hier", default=None, help="the two-stage directory (CATALOG.json, STAGE_A_<family>.json, *_leaves.json)")
    rp.add_argument("--contracts", default=None, help="where STRATEGY_PARAMS_FROZEN_v1.json is written")
    a = ap.parse_args()
    if stop_requested(a.out):
        print("STOP flag present in the work root: nothing started (owner order 2026-10-27)")
        raise SystemExit(75)
    {"truth": cmd_truth, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
