#!/usr/bin/env python3
"""E3-M: modular specialists guided by the ZigZag expert (owner idea 2026-10-20), PROGRAMMED, to run once the
capacity ladder names an architecture (or with the registered u=2 design if it names none). DEVELOPMENT.

Hypothesis: training one model per TASK (open / close) and per DIRECTION (long / short) on the same expert
labels may beat one joint 4-class policy. Declared 2 x 2 factorial on IDENTICAL rows, rounds, seeds, trunk:

  JOINT     one policy {HOLD, LONG, SHORT, CLOSE} (+ TP/SL heads)                     = the E1-T reference
  TASK      OPENER {HOLD, LONG, SHORT} (+ TP/SL) on flat states; CLOSER {HOLD, EXIT} on positioned states
  DIR       LONG-model {HOLD, OPEN, CLOSE} on flat-or-long states; SHORT-model likewise on flat-or-short
  FULL      OPEN-L, OPEN-S {no, yes} (+ TP/SL) on flat states; CLOSE-L, CLOSE-S {HOLD, EXIT} on own position

Labels, partitioned by the visited state. FLAT: the hindsight LegOracle action (open long / open short / hold).
POSITIONED (every variant, so the factorial is not confounded by the label): EXIT when, in hindsight, the open
order's SL is hit before its TP (or it ends negative at the forced flat), HOLD otherwise = the owner's early
close ("cut the trade the stop would kill"); SL and TP stay in force either way. Round 0 = the expert's own
trajectory for flat states plus EXPLORATION positions (the TRAIN-prior random policy's open orders, round 0
only) for the closers, because the expert's own orders end at TP and carry no EXIT example; rounds 1..3 =
DAgger with the COMPOSED policy visiting states (same as E1-T). A module whose partition has one class keeps
its previous round's model (ledgered), never crashes the run. Composition: positioned d -> closer_d
decides EXIT; after an EXIT (or when flat) the ARBITER chooses among the openers on the flat-equivalent state
(position fields zeroed) and may emit the opposite direction = close-and-reverse, as the oracle does.

Arbiter variants (chosen on INNER_A, ledgered; pL/pS = each direction's open probability, tp/sl = that
direction's sigma multiples from its own heads):
  prob            open argmax(pL, pS) when it exceeds tau (tau from a rate grid on INNER_A, as E0)
  expected_gain   score_d = p_d * tp_d - (1 - p_d) * sl_d - COST_SIGMA; open argmax when > 0
  gain_over_risk  score_d / sl_d; open argmax when > 0
Decision: variant x arbiter with the highest seed-mean INNER_A Sharpe among frequency-valid ones (ties ->
fewer models); INNER_B reported for all; 2024 read ONCE for the chosen. No TEST.
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
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from app.short_horizon.expert_guided import oracle as OR  # noqa: E402
from app.short_horizon import zigzag as Z  # noqa: E402
from tools.expert_guided_e0 import RATE_GRID, jsonable, ledger  # noqa: E402
from tools.expert_guided_e1t import EXPERT_CFG, MUSASHI_CFG, STUDENT_CFG, Ctx, cat, pack, summary  # noqa: E402

SEEDS = (20261019, 20261020, 20261021)
ROUNDS = 3
VARIANTS = ("JOINT", "TASK", "DIR", "FULL")
ARBITERS = ("prob", "expected_gain", "gain_over_risk")
COST_SIGMA = 0.1          # declared round-trip cost in sigma units for the gain arbiters (not broker-verified)
EXIT = 1                  # closer class


def flat_state(st):
    s = np.array(st, np.float32)
    s[:5] = 0.0
    return s


# ----------------------------------------------------------------------------- labels per partition
def partition(st):
    return int(st[0])     # 0 flat, 1 long, -1 short


def labels_for(variant, st, a):
    """{module: (class, is_open_label)} for one visited state with oracle/expert action a."""
    pos = partition(st)
    out = {}
    if variant == "JOINT":
        out["joint"] = (a, a in (E.LONG, E.SHORT))
    elif variant == "TASK":
        if pos == 0:
            out["opener"] = (a if a in (E.LONG, E.SHORT) else E.HOLD, a in (E.LONG, E.SHORT))
        else:
            out["closer"] = (EXIT if a != E.HOLD else 0, False)
    elif variant == "DIR":
        if pos == 0:
            out["long"] = (1 if a == E.LONG else 0, a == E.LONG)
            out["short"] = (1 if a == E.SHORT else 0, a == E.SHORT)
        else:
            out["long" if pos > 0 else "short"] = (2 if a != E.HOLD else 0, False)
    else:
        if pos == 0:
            out["open_l"] = (1 if a == E.LONG else 0, a == E.LONG)
            out["open_s"] = (1 if a == E.SHORT else 0, a == E.SHORT)
        else:
            out["close_l" if pos > 0 else "close_s"] = (EXIT if a != E.HOLD else 0, False)
    return out


MODULES = {"JOINT": {"joint": (4, 2)}, "TASK": {"opener": (3, 2), "closer": (2, 0)},
           "DIR": {"long": (3, 2), "short": (3, 2)},
           "FULL": {"open_l": (2, 2), "open_s": (2, 2), "close_l": (2, 0), "close_s": (2, 0)}}   # module -> (n_cls, n_reg)


def collect(cx, period, act, cfg=STUDENT_CFG):
    """Like Ctx.rollout(collect=True) but every positioned row also carries (direction, entry, sl, tp, ff_bar).
    ``period`` is a period name of the dataset's split contract (== DG.PERIODS for dev3y) or an explicit (start, end)
    pair (refit sweep, 2026-10-21)."""
    a, b = cx.ds.meta["periods"][period] if isinstance(period, str) else period
    env = E.BarEnv(cx.ds.bars, cx.ds.cal, cx.ds.sigma, cx.ds.sigma_rule, a, b, cfg)
    vis = []
    for bar in range(env.first, env.last):
        r = cx.rob[bar]
        st = env.state(bar)
        if r >= 0 and cx.obs_ok[r]:
            action, tp, sl = act(bar, r, st, env)
            if cx.lab_ok[r]:
                p = env.pos
                info = None if p is None else (p.direction, p.entry_price, p.sl, p.tp,
                                               int(np.searchsorted(cx.v, env.ff[int(env.seg_of[bar])])))
                vis.append((r, bar, st, action, tp, sl, info))
        else:
            action, tp, sl = E.HOLD, 1.5, 2.0
        env.step(bar, action, tp, sl)
    return env.result(), vis


def exit_label(cx, bar, info):
    """1 (EXIT) if the open order's SL is hit before its TP from bar+1 on (both in one bar -> SL first,
    conservative) or it ends negative at the forced flat; else 0 (HOLD)."""
    d, entry, sl, tp, ffb = info
    h, l, c = cx.h, cx.l, cx.c
    for k in range(bar + 1, min(ffb, len(c))):
        hit_sl = l[k] <= sl if d > 0 else h[k] >= sl
        hit_tp = h[k] >= tp if d > 0 else l[k] <= tp
        if hit_sl:
            return 1
        if hit_tp:
            return 0
    k = min(ffb, len(c)) - 1
    return int(d * (c[k] - entry) < 0)


def exit_resolution_bar(cx, bar, info):
    """Bar at which the open order's label becomes final (first touch or forced flat), or -1 if beyond the data."""
    d, entry, sl, tp, ffb = info
    h, l = cx.h, cx.l
    for k in range(bar + 1, min(ffb, len(cx.c))):
        if ((l[k] <= sl) if d > 0 else (h[k] >= sl)) or ((h[k] >= tp) if d > 0 else (l[k] <= tp)):
            return k
    return min(ffb, len(cx.c)) - 1 if ffb <= len(cx.c) else -1


class ExpertTrack:
    """One ZigZag demonstrator (owner 2026-10-22: several ZigZag signals as TRAINING-ONLY demonstrations; never an
    input). The primary track is the dataset's own expert (pivots, labels, support and oracle unchanged); an extra
    track recomputes hindsight pivots with its own deviation on the same admitted bars. A flat row's label from an
    extra track is used only when that track's label support (pivot confirmation) ends inside the row's period, so a
    slower expert cannot leak a later bar across a period boundary."""

    def __init__(self, cx, params=None, reversal_pips=None):
        ds = cx.ds
        self.primary = params is None and reversal_pips is None
        self.params = DG.EXPERT if params is None else params
        self.reversal_pips = reversal_pips
        if self.primary:
            self.y, self.sup, self.oracle, self.n_pivots = ds.y_hind, ds.support_end, cx.oracle, len(ds.pivots)
            self.y_conf = ds.y_conf
        else:
            h, l = ds.bars["HIGH"].to_numpy(float), ds.bars["LOW"].to_numpy(float)
            if reversal_pips is not None:
                piv, conf = OR.reversal_pivots(h, l, reversal_pips * Z.PIP)
            else:
                piv = Z.hindsight_pivots(h, l, params)
                conf = Z.confirmation_bars_fast(h, l, piv, params)
            n = len(h)
            self.y, self.sup = OR.pivot_labels(n, piv), OR.label_support_end(n, piv, conf)
            self.y_conf = OR.confirmed_labels(n, piv, conf)[0]
            self.oracle, self.n_pivots = OR.LegOracle(ds.bars["CLOSE"].to_numpy(float), piv), len(piv)
            ends = {p: DG._utc_ns(b) for p, (a_, b) in ds.meta["periods"].items()}
            self._end = np.array([ends.get(p, 0) for p in ds.period], np.int64)
            self._v = ds.bars.index.as_unit("ns").asi8
        self.name = (f"zz_rev{reversal_pips:g}pips" if reversal_pips is not None
                     else f"zz_d{self.params.depth}_dev{self.params.deviation_pips:g}_b{self.params.backstep}")

    def valid(self, bar, r):
        if self.primary:
            return True                          # the dataset's labelled() mask already applies this track's support
        se = int(self.sup[bar])
        return se >= 0 and self._v[se] + DG.H_NS <= self._end[r]

    def confirmed_policy(self):
        """The causal-confirmed counterpart (acts only once the pivot is confirmed): how much of the hindsight edge
        survives the confirmation lag (imitability information, reported; never an input)."""
        def f(bar, r, st, env):
            y = self.y_conf[bar]
            return (E.HOLD if y == 0 else (E.LONG if y == OR.LONG_CLASS else E.SHORT)), DG.CONFIRMED_K_TP, 2.0
        return f

    def policy(self, cx):
        def f(bar, r, st, env):
            y = self.y[bar]
            if y == 0:
                return E.HOLD, 1.5, 2.0
            s = float(cx.ds.sigma[bar + 1])
            rem = self.oracle.remaining[bar]
            tp = rem / s if np.isfinite(rem) and s > 0 else -1.0
            if tp <= 0:
                return E.HOLD, 1.5, 2.0
            return (E.LONG if y == OR.LONG_CLASS else E.SHORT), tp, 2.0
        return f


def parse_track(cx, spec: str):
    """'2' -> MetaTrader ZigZag depth 2 (registered deviation/backstep); 'rev10' -> reversal ZigZag with 10 pips."""
    spec = str(spec).strip()
    if spec.startswith("rev"):
        return ExpertTrack(cx, reversal_pips=float(spec[3:]))
    return ExpertTrack(cx, Z.Params(depth=int(spec), deviation_pips=DG.EXPERT.deviation_pips, backstep=DG.EXPERT.backstep))


class AuxTargets:
    """Hindsight auxiliary targets (owner 2026-10-22; training-only, never an observation). Horizon of every target =
    the optimal (registered) expert's next pivot, capped by 72 bars and the forced flat.
      opportunity(side): best favourable excursion of an entry at the next open, in causal sigma, net of the declared
                         cost: how much the open could have earned at most before the optimum.
      hold_gain(side):   best further gain from the current close of an open position: how much more the trade could
                         still earn relative to closing now (= regret of an exit before the optimum).
    A target is used only when its horizon ends inside the row's period (purge by the real label support)."""
    BOUND = (-0.1, 3.0)                      # covers >= 99 % of TRAIN targets (p1 -0.06, p99 2.1-2.2; SIGNAL_PROFILE)

    def __init__(self, cx, horizon=72, cost=COST_SIGMA):
        ds = cx.ds
        o, h, l, c = (ds.bars[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
        v = ds.bars.index.as_unit("ns").asi8
        n = len(c)
        piv = np.array([i for i, _, _ in ds.pivots], np.int64)
        nxt = piv[np.minimum(np.searchsorted(piv, np.arange(n), side="right"), len(piv) - 1)]
        seg = ds.cal.assign(ds.bars.index)
        ff = np.array([s.force_flat.value for s in ds.cal.segments], np.int64)
        flat = np.array([int(np.searchsorted(v + DG.H_NS, ff[int(seg[b])], side="right")) - 1 if seg[b] >= 0 else n - 1
                         for b in range(n)])
        sig = np.asarray(ds.sigma, float)
        self.opp = {1: np.full(n, np.nan), -1: np.full(n, np.nan)}
        self.hold = {1: np.full(n, np.nan), -1: np.full(n, np.nan)}
        self.end_ns = np.full(n, np.iinfo(np.int64).max, np.int64)
        for b in range(n - 2):
            end = int(min(nxt[b], b + horizon, flat[b], n - 1))
            s = sig[b + 1]
            if end <= b + 1 or not np.isfinite(s) or s <= 0:
                continue
            hh, ll = h[b + 1:end + 1], l[b + 1:end + 1]
            self.opp[1][b] = (hh.max() - o[b + 1]) / s - cost
            self.opp[-1][b] = (o[b + 1] - ll.min()) / s - cost
            self.hold[1][b] = (hh.max() - c[b]) / s
            self.hold[-1][b] = (c[b] - ll.min()) / s
            self.end_ns[b] = v[end] + DG.H_NS
        ends = {p: DG._utc_ns(e) for p, (s_, e) in ds.meta["periods"].items()}
        self.row_end = np.array([ends.get(p, 0) for p in ds.period], np.int64)

    def _get(self, arr, bar, r):
        x = arr[bar]
        ok = np.isfinite(x) and self.end_ns[bar] <= self.row_end[r]
        return (float(x) if ok else 0.0), float(ok)

    def opportunity(self, bar, r, side):
        return self._get(self.opp[side], bar, r)

    def hold_gain(self, bar, r, side):
        return self._get(self.hold[side], bar, r)


def merge_sets(*parts):
    out = {}
    for D in parts:
        for m, d in D.items():
            out[m] = cat(out[m], d) if m in out else d
    return out


def build_sets(variant, vis, cx, from_expert, final_before_bar=None, mask=None, tracks=None, track=None, aux=None):
    """vis rows -> {module: (X, y, R, Rm)}; flat labels from the expert's issued action (round 0) or the oracle,
    positioned labels from ``exit_label``. With ``final_before_bar`` (refit sweep) a row is used only when its label
    is FINAL before that bar: hindsight support end (flat rows) or the order's resolution bar (positioned rows).
    ``mask`` (agent-state ablation): the state slots fed to the network; labels and partitions use the real state.
    ``track`` (round 0, from_expert): the demonstrator that drove ``vis``; its flat rows are kept only where its own
    label support is inside the period. ``tracks`` (DAgger rounds): every flat row is labelled by EACH track whose
    support is valid there (multi-expert consensus); positioned rows keep the single exit-rule label."""
    acc = {m: ([], [], [], []) for m in MODULES[variant]}
    for r, bar, st, a_exp, tp_exp, sl_exp, info in vis:
        if final_before_bar is not None:
            if info is not None:
                rb = exit_resolution_bar(cx, bar, info)
                if rb < 0 or rb >= final_before_bar:
                    continue
            else:
                se = int(cx.ds.support_end[bar])
                if se < 0 or se >= final_before_bar:
                    continue
        if info is not None:
            labs = [((E.CLOSE if exit_label(cx, bar, info) else E.HOLD), 1.5, 2.0)]
        elif from_expert:
            if track is not None and not track.valid(bar, r):
                continue
            labs = [(a_exp, float(np.clip(tp_exp, 0.25, 3.0)), float(np.clip(sl_exp, 0.5, 3.0)))]
        else:
            labs = []
            for orc in ([None] if tracks is None else [t for t in tracks if t.valid(bar, r)]):
                o = cx.oracle if orc is None else orc.oracle
                a, tp, sl = o.action(bar, int(st[0]), bool(st[-1] > 0.5), float(cx.ds.sigma[bar + 1]))
                if a in (E.LONG, E.SHORT) and int(st[0]) != 0:
                    a = E.HOLD                       # positioned rows are labelled by the exit rule only
                labs.append((a, tp, sl))
        for a, tp, sl in labs:
            for m, (y, is_open) in labels_for(variant, st, a).items():
                X, Y, R, Rm = acc[m]
                X.append(np.r_[cx.Xw[r], E.masked_state(st, mask)])
                Y.append(y)
                if aux is None:
                    R.append((tp, sl))
                    Rm.append(float(is_open))
                elif m in ("open_l", "open_s"):         # per-target masks: TP/SL on opening rows, opportunity on every flat row
                    ov, ok = aux.opportunity(bar, r, 1 if m == "open_l" else -1)
                    R.append((tp, sl, ov))
                    Rm.append((float(is_open), float(is_open), ok))
                else:                                    # closers: hold gain relative to the optimum
                    hv, ok = aux.hold_gain(bar, r, 1 if m == "close_l" else -1)
                    R.append((hv,))
                    Rm.append((ok,))
    return {m: pack(*v) for m, v in acc.items() if len(v[1])}


# ----------------------------------------------------------------------------- composed policy
class Composed:
    """``mask`` (agent-state ablation): the network input keeps only the unmasked state slots (constant zeros
    elsewhere); the routing by partition (which module acts) and the env use the real state."""

    def __init__(self, variant, pols, arbiter, tau=None, mask=None):
        self.variant, self.p, self.arb, self.tau, self.mask = variant, pols, arbiter, tau, mask

    def _probs(self, name, x):
        lg, rg = self.p[name](x[None, :])
        z = lg[0] - lg[0].max()
        return np.exp(z) / np.exp(z).sum(), (None if rg is None else rg[0])

    def openers(self, x):
        """(pL, tpL, slL, pS, tpS, slS) on a flat-equivalent input."""
        v = self.variant
        if v == "JOINT":
            P, rg = self._probs("joint", x)
            return P[E.LONG], rg[0], rg[1], P[E.SHORT], rg[0], rg[1]
        if v == "TASK":
            P, rg = self._probs("opener", x)
            return P[E.LONG], rg[0], rg[1], P[E.SHORT], rg[0], rg[1]
        if v == "DIR":
            PL, rl = self._probs("long", x)
            PS, rs = self._probs("short", x)
            return PL[1], rl[0], rl[1], PS[1], rs[0], rs[1]
        PL, rl = self._probs("open_l", x)
        PS, rs = self._probs("open_s", x)
        return PL[1], rl[0], rl[1], PS[1], rs[0], rs[1]

    def exit_now(self, x, pos):
        v = self.variant
        if v == "JOINT":
            P, _ = self._probs("joint", x)
            a = int(np.argmax(P))
            return a == E.CLOSE or (a == E.LONG and pos < 0) or (a == E.SHORT and pos > 0)
        if v == "TASK":
            return int(np.argmax(self._probs("closer", x)[0])) == EXIT
        if v == "DIR":
            return int(np.argmax(self._probs("long" if pos > 0 else "short", x)[0])) == 2
        return int(np.argmax(self._probs("close_l" if pos > 0 else "close_s", x)[0])) == EXIT

    def arbitrate(self, pL, tpL, slL, pS, tpS, slS):
        if self.arb == "prob":
            d, p = (E.LONG, pL) if pL >= pS else (E.SHORT, pS)
            return d if p > self.tau else E.HOLD
        gL = pL * tpL - (1 - pL) * slL - COST_SIGMA
        gS = pS * tpS - (1 - pS) * slS - COST_SIGMA
        if self.arb == "gain_over_risk":
            gL, gS = gL / slL, gS / slS
        d, g = (E.LONG, gL) if gL >= gS else (E.SHORT, gS)
        return d if g > 0 else E.HOLD

    def __call__(self, cx):
        def f(bar, r, st, env):
            pos = int(st[0])
            s_in = E.masked_state(st, self.mask)
            x = np.r_[cx.Xw[r], s_in].astype(np.float32)
            if self.variant == "JOINT":            # the joint policy acts exactly as E1-T (argmax), no arbiter
                lg, rg = self.p["joint"](x[None, :])
                return int(np.argmax(lg[0])), float(rg[0, 0]), float(rg[0, 1])
            if pos != 0 and not self.exit_now(x, pos):
                return E.HOLD, 1.5, 2.0
            xf = np.r_[cx.Xw[r], flat_state(s_in)].astype(np.float32)
            pL, tpL, slL, pS, tpS, slS = self.openers(xf)
            d = self.arbitrate(pL, tpL, slL, pS, tpS, slS)
            if d == E.HOLD:
                return (E.CLOSE if pos != 0 else E.HOLD), 1.5, 2.0
            if d == pos:                           # closer said exit but the arbiter re-opens the same side: exit
                return E.CLOSE, 1.5, 2.0
            return d, float(tpL if d == E.LONG else tpS), float(slL if d == E.LONG else slS)
        return f

    def open_scores(self, cx, rows):
        """pmax per flat-equivalent row (for the prob arbiter's tau grid)."""
        out = []
        for r in rows:
            xf = np.r_[cx.Xw[r], np.zeros(9, np.float32)].astype(np.float32)
            pL, *_, pS, _, _ = self.openers(xf)
            out.append(max(pL, pS))
        return np.asarray(out)


# ----------------------------------------------------------------------------- run
def run(a):
    import torch
    from app.short_horizon.expert_guided import learn as L
    from app.short_horizon.expert_guided import temporal as T
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    led = out / "TRIAL_LEDGER.jsonl"
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    t0 = time.time()
    verify = not a.no_verify
    DG.check_scaling(a.scaling)                   # typed refusal before any read
    sc = DG.split_contract(a.split_contract)
    mask, inventory = E.state_mask(a.agent_state), E.state_inventory(a.agent_state)
    import dataclasses
    if a.exec_contract == "sac":       # owner 2026-10-22: the replay contract (max 1 entry/day, leverage 1:50) everywhere
        SCFG = MUSASHI_CFG
        ECFG = dataclasses.replace(EXPERT_CFG, max_entries_per_day=MUSASHI_CFG.max_entries_per_day, sizing=MUSASHI_CFG.sizing)
    else:
        SCFG, ECFG = STUDENT_CFG, EXPERT_CFG
    if a.aux_targets and a.variants != "FULL":
        raise SystemExit("E3M_AUX_TARGETS_ONLY_FOR_FULL")
    if (a.window, a.stride) != (DG.WIN, DG.STRIDE) and not a.store:
        raise SystemExit("E3M_WINDOW_ONLY_ON_STORE_PATH")
    if a.store:
        feats = json.loads(Path(a.feature_list).read_text())["features"]
        ds = DG.load_store(a.admitted, a.store, feats, scaling=a.scaling, verify=verify, contract=a.split_contract, nonstationary=a.nonstationary, eligible_from=a.eligible_from,
                           win=a.window, stride=a.stride)
    elif a.features == "core15":
        ds = DG.load(a.admitted, a.csv_dir, scaling=a.scaling, verify=verify, contract=a.split_contract, nonstationary=a.nonstationary, eligible_from=a.eligible_from)
    else:
        ds = DG.load_v4(a.admitted, a.v4_dir, scaling=a.scaling, verify=verify, contract=a.split_contract, nonstationary=a.nonstationary, eligible_from=a.eligible_from)
    st = ds.meta["scaling"].get("stationarity")
    if st:
        print(json.dumps({"stationarity_policy": st["policy"], "differenced": st["differenced"],
                          "table": {f: {k: r[k] for k in ("adf_p", "kpss_p", "decision")} for f, r in st["features"].items()}}), flush=True)
    if ds.meta["scaling"]["policy"] != a.scaling:
        raise SystemExit(f"E3M_SCALING_NOT_EFFECTIVE:{a.scaling}!={ds.meta['scaling']['policy']}")
    cx = Ctx(ds)
    nf, NS = ds.windows.shape[2], len(E.STATE_NAMES)
    WIN_E = int(ds.meta["input_contract"]["observations"])
    if ds.windows.shape[1] != WIN_E:
        raise SystemExit(f"E3M_WINDOW_LAYOUT:{ds.windows.shape[1]}!={WIN_E}")
    DEV = tuple(sc["report"])                     # dev3y: TRAIN, INNER_A, INNER_B; consolidated: TRAIN, INNER_A (inner block)
    ARCH = dict(units=a.units, branch=a.branch, core=a.core, depth=a.depth, head=a.head)
    MODS = dict(MODULES)
    BOUNDS = {m: ((0.25, 3.0), (0.5, 3.0)) for v_ in MODULES for m in MODULES[v_]}
    if a.aux_targets:
        MODS["FULL"] = {"open_l": (2, 3), "open_s": (2, 3), "close_l": (2, 1), "close_s": (2, 1)}
        BOUNDS.update({"open_l": ((0.25, 3.0), (0.5, 3.0), AuxTargets.BOUND), "open_s": ((0.25, 3.0), (0.5, 3.0), AuxTargets.BOUND),
                       "close_l": (AuxTargets.BOUND,), "close_s": (AuxTargets.BOUND,)})
    if a.groups:
        # extractor groups: {"groups": {name: [columns]}, "units": {name: u}} matched to the tensor order by base name
        gspec = json.loads(Path(a.groups).read_text())
        tensor = [f.split("@")[0] for f in ds.meta["input_contract"]["features"]]
        names, idx, gu = [], [], []
        for g, cols in gspec["groups"].items():
            ii = [tensor.index(c.split("@")[0]) for c in cols if c.split("@")[0] in tensor]
            if ii:
                names.append(g); idx.append(ii); gu.append(int(gspec.get("units", {}).get(g, a.branch or a.units)))
        if sorted(i for g in idx for i in g) != list(range(nf)):
            raise SystemExit("E3M_GROUPS_DO_NOT_PARTITION_THE_TENSOR")
        ARCH.update(groups=idx, group_units=gu, group_names=names)
    AUX = AuxTargets(cx) if a.aux_targets else None
    tracks = [ExpertTrack(cx)] + [ExpertTrack(cx, Z.Params(depth=DG.EXPERT.depth, deviation_pips=float(d), backstep=DG.EXPERT.backstep))
                                  for d in []] + [parse_track(cx, d) for d in (a.extra_experts.split(",") if a.extra_experts else [])]
    variants = [v for v in VARIANTS if v in a.variants.split(",")]
    params = {v: {m: int(sum(p.numel() for p in T.TemporalPolicyNet(nf, WIN_E, nc, n_reg=nr, n_state=NS, reg_bounds=BOUNDS[m], **ARCH).parameters()))
                  for m, (nc, nr) in MODS[v].items()} for v in variants}
    agent_state = {"mode": a.agent_state, "inventory": inventory, "n_state_slots": NS,
                   "included": [n for n, v in inventory.items() if v == "included"],
                   "constant_zero": [n for n, v in inventory.items() if v == "constant_zero"], "params": params,
                   "note": "omitted slots are constant zeros in the network input only; tensor shape and parameter count "
                           "unchanged; the env keeps its real state and enforces every business rule"}
    print(json.dumps({"agent_state": a.agent_state, "inventory": inventory, "params": params, "split_contract": a.split_contract,
                      "scaling": ds.meta["scaling"]["policy"], "verify": verify}), flush=True)
    ledger(led, rung="E3M", event="PREREGISTRATION", variants=variants, arbiters=list(ARBITERS), arch=ARCH,
           dropout=a.dropout, wd=a.wd, cost_sigma=COST_SIGMA, rounds=ROUNDS, seeds=SEEDS, params=params, device=device,
           scaling=a.scaling, split_contract=ds.meta["split_contract"], agent_state=agent_state, verify=verify,
           experts=[t.name for t in tracks], window=[WIN_E, int(ds.meta["input_contract"]["stride_rows"])],
           exec_contract=a.exec_contract, aux_targets=bool(a.aux_targets),
           extractor_groups=ARCH.get("group_names"),
           decision="variant x arbiter: highest seed-mean INNER_A Sharpe among valid; ties -> fewer models; 2024 once")
    scaler_path = DG.write_scaler(out, ds, rung="E3M", agent_state=a.agent_state, split_contract=a.split_contract)
    res = {"exec_contract": a.exec_contract, "aux_targets": ({"opportunity_at_open": "best excursion before the optimal pivot (sigma, net of cost)",
                                                              "hold_gain_at_close": "best further gain vs closing now (sigma)",
                                                              "bound": AuxTargets.BOUND, "horizon": "next optimal pivot, <= 72 bars, <= forced flat"}
                                                             if a.aux_targets else None),
           "meta": ds.meta, "arch": ARCH, "params": params, "window": {"observations": WIN_E, "stride_rows": int(ds.meta["input_contract"]["stride_rows"])},
           "experts": [{"name": t.name, "primary": t.primary, "pivots": t.n_pivots} for t in tracks], "scaling": a.scaling, "variants": variants, "arbiters": list(ARBITERS),
           "cost_sigma": COST_SIGMA, "device": device, "comparators": {}, "trials": {}, "verify": verify,
           "split_contract": ds.meta["split_contract"], "agent_state": agent_state, "scaler_file": scaler_path.name,
           "probe": {}}
    probe_rows = cx.ds.rows("INNER_A")
    probe_rows = probe_rows[np.linspace(0, len(probe_rows) - 1, min(8, len(probe_rows))).astype(int)] if len(probe_rows) else probe_rows

    def probe(pol):
        """Logits of the saved module on fixed rows with a zero (flat-equivalent) state: ties the checkpoint to the tensors."""
        X = np.stack([np.r_[cx.Xw[r], np.zeros(NS, np.float32)] for r in probe_rows]).astype(np.float32) if len(probe_rows) else np.zeros((0, cx.Xw.shape[1] + NS), np.float32)
        lg, rg = pol(X)
        return {"rows": probe_rows.tolist(), "row_ts_ns": [int(cx.ds.row_ts[r]) for r in probe_rows], "state": "zeros",
                "logits": lg.tolist(), "reg": None if rg is None else rg.tolist()}

    def fit_module(v, m, D_tr, D_in, seed):
        nc, nr = MODS[v][m]
        f = np.maximum(np.bincount(D_tr[1], minlength=nc) / len(D_tr[1]), 1e-3)   # 1/sqrt(freq), as E1-T
        w = 1 / np.sqrt(f)
        w = w / w[D_tr[1]].mean()
        model, hist, info = L.fit(D_tr[0], D_tr[1], D_in[0], D_in[1], seed=seed, n_cls=nc,
                                  R=D_tr[2] if nr else None, Rm=D_tr[3] if nr else None,
                                  Ri=D_in[2] if nr else None, Rmi=D_in[3] if nr else None, class_weight=w,
                                  dropout=a.dropout, lr=1e-3, wd=a.wd, batch=256, max_epochs=a.max_epochs, patience=10,
                                  device=device, net=lambda: T.TemporalPolicyNet(nf, WIN_E, nc, n_reg=nr, n_state=NS,
                                                                                  dropout=a.dropout, reg_bounds=BOUNDS[m], **ARCH))
        return T.TorchPolicy(model), info

    # expert trajectory (round 0 demonstrations) and comparators
    cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    demo_vis = {p: collect(cx, p, cx.expert, ECFG)[1] for p in ("TRAIN", "INNER_A")}
    extra_vis = {(t.name, p): collect(cx, p, t.policy(cx), ECFG)[1] for t in tracks[1:] for p in ("TRAIN", "INNER_A")}
    _, vtr0 = collect(cx, "TRAIN", cx.expert, ECFG)
    prior = (np.bincount(np.array([v[3] for v in vtr0], int), minlength=4) / len(vtr0)).tolist()
    explore = {p: [v for v in collect(cx, p, cx.prior_random(prior, 0), SCFG)[1] if v[6] is not None]
               for p in ("TRAIN", "INNER_A")}       # exploration positions (round 0 closers only)
    res["exploration_positions"] = {p: len(v) for p, v in explore.items()}
    for p in DEV:
        for t in tracks[1:]:
            res["comparators"].setdefault(p, {})[f"expert_demonstrations_{t.name}"] = summary(cx.rollout(p, t.policy(cx), ECFG)[0])
        res["comparators"].setdefault(p, {})
        res["comparators"][p] |= {"expert_demonstrations": summary(cx.rollout(p, cx.expert, ECFG)[0]),
                                 "expert_confirmed": summary(cx.rollout(p, cx.confirmed, SCFG)[0]),
                                 "no_trade": summary(cx.rollout(p, lambda *x: (E.HOLD, 1.5, 2.0), SCFG)[0])}
    frozen = {}
    for v in variants:
        for seed in SEEDS:
            D_tr = merge_sets(build_sets(v, demo_vis["TRAIN"] + explore["TRAIN"], cx, True, mask=mask, aux=AUX),
                              *[build_sets(v, extra_vis[(t.name, "TRAIN")], cx, True, mask=mask, track=t, aux=AUX) for t in tracks[1:]])
            D_in = merge_sets(build_sets(v, demo_vis["INNER_A"] + explore["INNER_A"], cx, True, mask=mask, aux=AUX),
                              *[build_sets(v, extra_vis[(t.name, "INNER_A")], cx, True, mask=mask, track=t, aux=AUX) for t in tracks[1:]])
            pols, infos = {}, {}
            comp = None
            for rnd in range(ROUNDS + 1):
                if rnd > 0:
                    _, vtr = collect(cx, "TRAIN", comp(cx), SCFG)
                    _, vin = collect(cx, "INNER_A", comp(cx), SCFG)
                    multi = tracks if len(tracks) > 1 else None
                    for m, D in build_sets(v, vtr, cx, False, mask=mask, tracks=multi, aux=AUX).items():
                        D_tr[m] = cat(D_tr[m], D) if m in D_tr else D
                    for m, D in build_sets(v, vin, cx, False, mask=mask, tracks=multi, aux=AUX).items():
                        D_in[m] = cat(D_in[m], D) if m in D_in else D
                for m in MODS[v]:
                    if m not in D_tr or m not in D_in or len(np.unique(D_tr[m][1])) < 2:
                        if m in pols:
                            ledger(led, rung="E3M", event="MODULE_KEPT", variant=v, module=m, seed=seed, round=rnd)
                            continue
                        raise SystemExit(f"E3M_EMPTY_PARTITION:{v}:{m}:round{rnd}")
                    ledger(led, rung="E3M", event="FIT", variant=v, module=m, seed=seed, round=rnd, rows_train=len(D_tr[m][1]),
                           rows_inner=len(D_in[m][1]), label_counts=np.bincount(D_tr[m][1]).tolist())
                    pols[m], infos[m] = fit_module(v, m, D_tr[m], D_in[m], seed)
                # DAgger rounds visit states with the prob arbiter at the median-rate tau (declared; not a choice)
                comp = Composed(v, pols, "prob", tau=0.5, mask=mask)
                if v != "JOINT":
                    pm = comp.open_scores(cx, cx.ds.rows("INNER_A"))
                    comp.tau = float(np.quantile(pm, 1 - 0.06))
            # arbiter trials on INNER_A only (every one ledgered)
            ia = cx.ds.rows("INNER_A")
            trials = []
            cands = [("prob", t) for t in RATE_GRID] if v != "JOINT" else [("argmax", None)]
            if v != "JOINT":
                cands += [("expected_gain", None), ("gain_over_risk", None)]
            pm = Composed(v, pols, "prob", mask=mask).open_scores(cx, ia) if v != "JOINT" else None
            for arb, rate in cands:
                tau = float(np.quantile(pm, 1 - rate)) if rate is not None else None
                c = Composed(v, pols, arb, tau, mask=mask)
                m = summary(cx.rollout("INNER_A", c(cx), SCFG)[0])
                trials.append({"arbiter": arb, "rate": rate, "tau": tau, "sharpe": m["sharpe_daily_sqrt365"], "valid": m["valid"],
                               "orders_per_week": m["orders_per_week"], "net": m["net_profit_usd"]})
                ledger(led, rung="E3M", event="ARBITER_TRIAL", variant=v, seed=seed, **trials[-1])
            ok = [t for t in trials if t["valid"] and t["sharpe"] is not None and math.isfinite(t["sharpe"])]
            best = max(ok or trials, key=lambda t: t["sharpe"] if t["sharpe"] is not None and math.isfinite(t["sharpe"]) else -1e9)
            c = Composed(v, pols, best["arbiter"], best["tau"], mask=mask)
            frozen[(v, seed)] = c
            ledger(led, rung="E3M", event="FREEZE", variant=v, seed=seed, arbiter=best["arbiter"], tau=best["tau"], chosen_on="INNER_A")
            rec = {"fits": {m: {k: infos[m][k] for k in ("epochs", "best_epoch", "params", "stop_reason")} for m in infos},
                   "arbiter_trials": trials, "arbiter": best, "trading": {}, "musashi_contract": {}}
            for p in DEV:
                rec["trading"][p] = summary(cx.rollout(p, c(cx), SCFG)[0])
                rec["musashi_contract"][p] = summary(cx.rollout(p, c(cx), MUSASHI_CFG)[0])
            for m, pol in pols.items():
                torch.save(pol.m.state_dict(), out / f"{v}_{m}_{seed}.pt")
                res["probe"][f"{v}_{m}_{seed}"] = probe(pol)
            (out / "PROBE.json").write_text(json.dumps({"split_contract": a.split_contract, "agent_state": a.agent_state,
                                                        "scaling": a.scaling, "probe": res["probe"]}, indent=1))
            res["trials"].setdefault(v, {})[str(seed)] = rec
            print(json.dumps({"variant": v, "seed": seed, "arbiter": best["arbiter"],
                              "sharpe": {p: round(rec["trading"][p]["sharpe_daily_sqrt365"], 2) for p in rec["trading"]},
                              "opw": round(rec["trading"][DEV[-1]]["orders_per_week"], 1), "s": round(time.time() - t0, 1)}), flush=True)
    # decision on INNER_A (seed means), 2024 once
    means = {}
    for v in variants:
        sh = [res["trials"][v][str(s)]["trading"]["INNER_A"]["sharpe_daily_sqrt365"] for s in SEEDS]
        valid = all(res["trials"][v][str(s)]["trading"]["INNER_A"]["valid"] for s in SEEDS)
        means[v] = {"inner_a_sharpe_mean": float(np.mean(sh)), "all_valid": valid, "n_models": len(MODULES[v])}
    ok = [v for v in variants if means[v]["all_valid"]]
    chosen = max(ok or variants, key=lambda v: (means[v]["inner_a_sharpe_mean"], -means[v]["n_models"]))
    res["decision"] = {"means": means, "chosen": chosen, "rule": "highest seed-mean INNER_A Sharpe among all-valid; ties -> fewer models"}
    ledger(led, rung="E3M", event="DECISION", **res["decision"])
    ledger(led, rung="E3M", event="READ", periods=["OBS_2024"], purpose=f"single read: chosen variant {chosen} + comparators")
    res["comparators"]["OBS_2024"] = {"expert_demonstrations": summary(cx.rollout("OBS_2024", cx.expert, ECFG)[0]),
                                      "expert_confirmed": summary(cx.rollout("OBS_2024", cx.confirmed, SCFG)[0]),
                                      "no_trade": summary(cx.rollout("OBS_2024", lambda *x: (E.HOLD, 1.5, 2.0), SCFG)[0])}
    for seed in SEEDS:
        c = frozen[(chosen, seed)]
        rec = res["trials"][chosen][str(seed)]
        rec["trading"]["OBS_2024"] = summary(cx.rollout("OBS_2024", c(cx), SCFG)[0])
        rec["musashi_contract"]["OBS_2024"] = summary(cx.rollout("OBS_2024", c(cx), MUSASHI_CFG)[0])
    res["wall_seconds"] = time.time() - t0
    (out / "E3M.json").write_text(json.dumps(jsonable(res), indent=1))
    print(json.dumps({"decision": res["decision"], "wall": round(res["wall_seconds"], 1)}))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--admitted", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--v4-dir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", choices=["v4", "core15"], default="core15")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--units", type=int, default=2)
    ap.add_argument("--branch", type=int, default=None)
    ap.add_argument("--core", type=int, default=None)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--scaling", choices=list(DG.SCALINGS), default="rolling_z")
    ap.add_argument("--window", type=int, default=DG.WIN, help="observations per window (store path only; default the inherited 13)")
    ap.add_argument("--stride", type=int, default=DG.STRIDE, help="observed rows between window slots (store path only; default 2)")
    ap.add_argument("--groups", default="", help="extractor groups JSON {groups: {name: [cols]}, units: {name: u}}; default one extractor per feature")
    ap.add_argument("--exec-contract", choices=("native", "sac"), default="native",
                    help="native: max 4 entries/day, leverage 1:100 (inherited); sac: max 1 entry/day, leverage 1:50 (the replay contract) "
                         "for demonstrations, DAgger rollouts, arbiter choice and every reported period")
    ap.add_argument("--aux-targets", action="store_true", help="FULL only: hindsight opportunity (openers) and hold-gain (closers) heads")
    ap.add_argument("--extra-experts", default="", help="comma list of ZigZag DEPTHS added as training-only demonstrators "
                    "(the replicated MetaTrader Deviation test is inert: pivots depend on Depth and Backstep only)")
    ap.add_argument("--eligible-from", default=None, help="shared-support option: rows before this UTC date are ineligible in every arm (matched comparisons)")
    ap.add_argument("--nonstationary", choices=list(DG.NONSTATIONARY), default="diff",
                    help="fixed_train only: a feature measured non-stationary on TRAIN (ADF/KPSS) is differenced (default), refused or kept")
    ap.add_argument("--agent-state", choices=list(E.AGENT_STATE_MODES), default="full",
                    help="ablation: feedback slots fed to the network (omitted slots = constant zeros; same shape/params)")
    ap.add_argument("--split-contract", choices=list(DG.SPLIT_CONTRACTS), default="dev3y",
                    help="dev3y = fit 2020-2022 / 2023A / 2023B / 2024 once; consolidated = TRAIN 2020-2023 with its last "
                         "6 months as the inner selection block, 2024 once; TEST stays closed")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the source pins (fixture runs only; recorded as verify=false in every output)")
    ap.add_argument("--store", default="", help="layer-1 pool parquet (MANIFEST.json beside it)")
    ap.add_argument("--feature-list", default="", help="JSON with {'features': [...]} selected from the store")
    ap.add_argument("--cpu", action="store_true")
    return ap


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
