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
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import action_contract as ACT  # noqa: E402
from app.short_horizon.expert_guided import c2 as C2  # noqa: E402
from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from app.short_horizon.expert_guided import multires as MR  # noqa: E402
from app.short_horizon.expert_guided import oracle as OR  # noqa: E402
from app.short_horizon.expert_guided import signals as SG  # noqa: E402
from app.short_horizon import zigzag as Z  # noqa: E402
from tools.expert_guided_e0 import RATE_GRID, jsonable, ledger  # noqa: E402
from tools.expert_guided_e1t import EXPERT_CFG, MUSASHI_CFG, STUDENT_CFG, Ctx, cat, pack, summary  # noqa: E402

SEEDS = (20261019, 20261020, 20261021)
SEEDS_DEFAULT = SEEDS


def parse_seeds(text) -> tuple:
    """``--seeds``: comma list of distinct positive integers (default = the inherited three; a seed extension of an arm
    runs the new seeds only, into its own --out, and is pooled by the launcher's decide)."""
    try:
        seeds = tuple(int(x) for x in str(text).split(",") if x.strip())
    except ValueError as ex:
        raise SystemExit(f"E3M_SEEDS_MALFORMED:{text}") from ex
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) <= 0:
        raise SystemExit(f"E3M_SEEDS_MALFORMED:{text}")
    return seeds
ROUNDS = 3
VARIANTS = ("JOINT", "TASK", "DIR", "FULL")
ARBITERS = ("prob", "expected_gain", "gain_over_risk")
COST_SIGMA = 0.1          # declared round-trip cost in sigma units for the gain arbiters (not broker-verified)
EXIT = 1                  # closer class
DROP_CENSORED_MODES = ("opportunity", "multires")   # a censored signal drops the row; "both" keeps the exit rule


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


def forced_flat_bars(ds):
    """Per bar: the last bar whose availability (start + 1 h) is at or before the forced flat of its weekly segment."""
    v = ds.bars.index.as_unit("ns").asi8
    n = len(v)
    seg = ds.cal.assign(ds.bars.index)
    ff = np.array([s.force_flat.value for s in ds.cal.segments], np.int64)
    return np.array([int(np.searchsorted(v + DG.H_NS, ff[int(seg[b])], side="right")) - 1 if seg[b] >= 0 else n - 1
                     for b in range(n)])


class SignalLabels:
    """Per-agent hindsight labels (docs/short_horizon/clean_round/EXPERT_SIGNAL_PER_AGENT.md rev 2; owner 2026-10-22;
    Musashi's instructions 2026-10-22: a candidate experiment, not a certified expert). Training-only, never an input.
      mode "zigzag"       every label as before (byte-identical default);
      mode "opportunity"  flat rows: OPEN side d when opportunity_d >= theta_open and adverse_to_peak_d <= sl_max, NOP
                          when neither side qualifies (both sides: the larger net opportunity; a tie is NOP); TP head =
                          opportunity, SL head = adverse_to_peak (bounded as the other heads); positioned rows: CLOSE /
                          HOLD by the declared closer rule on the order's OWN SL distance (hold_through | optimum);
      mode "both"         OR with a fixed truth table: the ZigZag side wins when it names one, otherwise the signal's
                          side, NOP only when neither names a side; closers CLOSE when either rule says so.
    Availability (label_available_at): a row's label is used only when its horizon AND the ZigZag that defines it are
    final inside the row's period: avail = max(end, support_end[end]) where support_end[end] is the confirmation bar
    of the first optimal pivot at/after the horizon end (the price at the pivot bar is not proof that the hindsight
    ZigZag was already stable there); censored when there is none. Censored rows are DROPPED, never filled."""
    MODES = ("zigzag", "opportunity", "both")
    RULES = ("hold_through", "optimum")
    TP_BOUNDS, SL_BOUNDS = (0.25, 3.0), (0.5, 3.0)

    def __init__(self, cx, mode="opportunity", closer_rule="hold_through", params=None):
        if mode not in self.MODES:
            raise ValueError(f"LABELS_MODE_UNKNOWN:{mode}")
        if closer_rule not in self.RULES:
            raise ValueError(f"CLOSER_RULE_UNKNOWN:{closer_rule}")
        ds = cx.ds
        self.mode, self.rule, self.p = mode, closer_rule, params or SG.SignalParams()
        o, h, l, c = (ds.bars[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
        self.c, self.sigma = c, np.asarray(ds.sigma, float)
        v = ds.bars.index.as_unit("ns").asi8
        n = len(c)
        piv = [(int(i), 1 if k == Z.PEAK else -1, float(lv)) for i, k, lv in ds.pivots]
        self.sig, self.hz = SG.hindsight_signals(o, h, l, c, self.sigma, piv, forced_flat_bars(ds), self.p)
        self.open = {d: SG.opener_labels(self.sig, d, self.p) for d in (1, -1)}
        sup = np.asarray(ds.support_end, np.int64)
        self.avail_bar, self.avail_ns = {}, {}
        for d in (1, -1):
            end = np.asarray(self.hz[d]["bar"], np.int64)
            se = np.where(end >= 0, sup[np.clip(end, 0, n - 1)], -1)
            avail = np.where((end >= 0) & (se >= 0), np.maximum(end, se), -1)
            self.avail_bar[d] = avail
            self.avail_ns[d] = np.where(avail >= 0, v[np.clip(avail, 0, n - 1)] + DG.H_NS, np.iinfo(np.int64).max)
        ends = {p: DG._utc_ns(e) for p, (s_, e) in ds.meta["periods"].items()}
        self.row_end = np.array([ends.get(p, 0) for p in ds.period], np.int64)
        self.n = {k: 0 for k in ("flat_used", "flat_censored", "flat_both_sides", "closer_used", "closer_censored",
                                 "closer_signal_censored_exit_rule_only", "truncated_used")}

    def available(self, bar, r, d) -> bool:
        return bool(self.avail_ns[d][bar] <= self.row_end[r])

    def flat_action(self, bar, r):
        """(action, tp, sl) of the opener / NOP signal at a flat row; None when either side is not final inside the
        row's period (NOP needs both sides)."""
        if not (self.available(bar, r, 1) and self.available(bar, r, -1)):
            return None
        L, S = self.open[1][bar], self.open[-1][bar]
        if not (np.isfinite(L) and np.isfinite(S)):
            return None
        if L == 0 and S == 0:
            return E.HOLD, 1.5, 2.0
        if L == 1 and S == 1:
            self.n["flat_both_sides"] += 1
            oL, oS = self.sig["opportunity"][1][bar], self.sig["opportunity"][-1][bar]
            if oL == oS:
                return E.HOLD, 1.5, 2.0
            d = 1 if oL > oS else -1
        else:
            d = 1 if L == 1 else -1
        if self.sig["truncated"][d][bar] == 1:
            self.n["truncated_used"] += 1
        tp = float(np.clip(self.sig["opportunity"][d][bar], *self.TP_BOUNDS))
        sl = float(np.clip(self.sig["adverse_to_peak"][d][bar], *self.SL_BOUNDS))
        return (E.LONG if d > 0 else E.SHORT), tp, sl

    def positioned_action(self, bar, r, info):
        """E.CLOSE / E.HOLD by the declared closer rule for the open order (its own SL distance in causal sigma);
        None when the signal is censored (horizon not final inside the period, or no pivot inside under `optimum`)."""
        d, entry, sl, tp, ffb = info
        if not self.available(bar, r, d):
            return None
        s = float(self.sigma[bar + 1])
        if not np.isfinite(s) or s <= 0:
            return None
        sl_dist = d * (self.c[bar] - sl) / s
        g, adv_hold = self.sig["hold_gain"][d][bar], self.sig["hold_adverse"][d][bar]
        if not (np.isfinite(g) and np.isfinite(adv_hold)):
            return None
        close = bool(g < self.p.theta_close or adv_hold >= sl_dist)
        if self.rule == "optimum":
            nxt = self.sig["next_adverse_inside"][d][bar]
            if not np.isfinite(nxt):
                return None
            close = close or nxt == 1
        return E.CLOSE if close else E.HOLD

    def profile(self, ds, period="TRAIN"):
        """Label properties on the period's eligible rows taken as flat origins (they describe labels, not learning)."""
        rows = ds.rows(period)
        acts, raw_tp, trunc = [], [], 0
        for r in rows:
            b = int(ds.row_bar[r])
            fa = self.flat_action(b, r)
            if fa is None:
                acts.append(None)
                continue
            acts.append(fa[0])
            if fa[0] in (E.LONG, E.SHORT):
                d = 1 if fa[0] == E.LONG else -1
                raw_tp.append(float(self.sig["opportunity"][d][b]))
                trunc += int(self.sig["truncated"][d][b] == 1)
        raw = np.array(raw_tp, float)
        return {"period": period, "rows": len(acts), "censored": sum(a is None for a in acts),
                "open_long": sum(a == E.LONG for a in acts), "open_short": sum(a == E.SHORT for a in acts),
                "nop": sum(a == E.HOLD for a in acts), "truncated_among_opens": trunc,
                "opportunity_raw_p50_p95": [float(np.percentile(raw, 50)), float(np.percentile(raw, 95))] if len(raw) else None,
                "tp_head_saturated_share": float(np.mean((raw < self.TP_BOUNDS[0]) | (raw > self.TP_BOUNDS[1]))) if len(raw) else None}

    def describe(self, ds):
        return {"mode": self.mode, "closer_rule": self.rule, "params": dataclasses.asdict(self.p),
                "tp_head": "opportunity (sigma, net of cost) bounded " + str(self.TP_BOUNDS),
                "sl_head": "adverse_to_peak (sigma) bounded " + str(self.SL_BOUNDS),
                "availability_rule": "max(horizon end, confirmation of the first optimal pivot at/after it) + 1 h <= period end; censored rows dropped",
                "both_truth_table": "ZigZag side if it names one; else signal side; NOP only when neither; closers CLOSE if either rule",
                "counts": self.n, "profile_train": self.profile(ds, "TRAIN")}


def _signal_flat(labels, mode, bar, r, lab):
    """Flat-row label(s) under the signal modes; None when the row is dropped (censored signal)."""
    if mode == "both" and lab[0] in (E.LONG, E.SHORT):
        labels.n["flat_used"] += 1
        return [lab]                                        # the ZigZag side wins when it names one
    fa = labels.flat_action(bar, r)
    if fa is None:
        labels.n["flat_censored"] += 1
        return None
    labels.n["flat_used"] += 1
    return [fa]


def merge_sets(*parts):
    out = {}
    for D in parts:
        for m, d in D.items():
            out[m] = cat(out[m], d) if m in out else d
    return out


def check_sample_weights(a) -> bool:
    """``--sample-weights`` is a multires-only treatment (the fitness weights exist in no other mode)."""
    if getattr(a, "sample_weights", False) and a.labels != "multires":
        raise SystemExit(f"SAMPLE_WEIGHTS_REQUIRE_MULTIRES:labels={a.labels}")
    return bool(getattr(a, "sample_weights", False))


def row_weight(labels, label_mode, sample_weights, info, bar, r):
    """Per-row loss weight: the multires label fitness of a FLAT row when ``--sample-weights`` is on (positioned rows
    and every other mode weigh 1)."""
    if not sample_weights or info is not None or label_mode != "multires" or labels is None:
        return 1.0
    return float(labels.weight(bar, r))


def build_sets(variant, vis, cx, from_expert, final_before_bar=None, mask=None, tracks=None, track=None, aux=None,
               labels=None, label_mode="zigzag", sample_weights=False, nop_labels=None, rows_out=None):
    """vis rows -> {module: (X, y, R, Rm)}; flat labels from the expert's issued action (round 0) or the oracle,
    positioned labels from ``exit_label``. With ``final_before_bar`` (refit sweep) a row is used only when its label
    is FINAL before that bar: hindsight support end (flat rows) or the order's resolution bar (positioned rows).
    ``mask`` (agent-state ablation): the state slots fed to the network; labels and partitions use the real state.
    ``track`` (round 0, from_expert): the demonstrator that drove ``vis``; its flat rows are kept only where its own
    label support is inside the period. ``tracks`` (DAgger rounds): every flat row is labelled by EACH track whose
    support is valid there (multi-expert consensus); positioned rows keep the single exit-rule label.
    ``labels`` / ``label_mode`` (per-agent signal, rev 2): "zigzag" leaves every label as above; "opportunity" and
    "both" relabel flat rows by ``SignalLabels.flat_action`` and positioned rows by the declared closer rule (see
    ``SignalLabels``); a censored signal drops the row under "opportunity" and leaves the exit rule alone under "both".
    ``sample_weights`` (multires only): the returned tuples carry a fifth element, the per-row loss weight (label
    fitness on flat rows, 1 on positioned rows); default False returns the 4-tuples unchanged.
    ``nop_labels`` (action contract v1, default None = unchanged): an ``ACT.NopLabels``; every flat row that produced
    opener labels also yields one ``nop`` sample, consistent with them (the arm opens -> NOP 0), returned as a 5-tuple
    with its weight. ``rows_out`` (default None): {module: [dataset row of every sample, in order]} for calibration."""
    use_sig = labels is not None and label_mode != "zigzag"
    acc = {m: ([], [], [], [], []) for m in MODULES[variant]}
    if nop_labels is not None:
        acc["nop"] = ([], [], [], [], [])
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
            ex = E.CLOSE if exit_label(cx, bar, info) else E.HOLD
            if use_sig:
                a_sig = labels.positioned_action(bar, r, info)
                if label_mode in DROP_CENSORED_MODES:
                    if a_sig is None:
                        labels.n["closer_censored"] += 1
                        continue
                    ex = a_sig
                elif a_sig is None:
                    labels.n["closer_signal_censored_exit_rule_only"] += 1
                else:
                    ex = E.CLOSE if (ex == E.CLOSE or a_sig == E.CLOSE) else E.HOLD
                labels.n["closer_used"] += 1
            labs = [(ex, 1.5, 2.0)]
        elif from_expert:
            if track is not None and not track.valid(bar, r):
                continue
            labs = [(a_exp, float(np.clip(tp_exp, 0.25, 3.0)), float(np.clip(sl_exp, 0.5, 3.0)))]
            if use_sig:
                labs = _signal_flat(labels, label_mode, bar, r, labs[0])
                if labs is None:
                    continue
        else:
            labs = []
            for orc in ([None] if tracks is None else [t for t in tracks if t.valid(bar, r)]):
                o = cx.oracle if orc is None else orc.oracle
                a, tp, sl = o.action(bar, int(st[0]), bool(st[-1] > 0.5), float(cx.ds.sigma[bar + 1]))
                if a in (E.LONG, E.SHORT) and int(st[0]) != 0:
                    a = E.HOLD                       # positioned rows are labelled by the exit rule only
                labs.append((a, tp, sl))
            if use_sig:
                labs = [x for lab in labs for x in (_signal_flat(labels, label_mode, bar, r, lab) or [])]
        wt = row_weight(labels, label_mode, sample_weights, info, bar, r)
        if nop_labels is not None and info is None and labs:
            nl = nop_labels.label(bar, r, any(x[0] in (E.LONG, E.SHORT) for x in labs))
            if nl is not None:
                X, Y, R, Rm, W = acc["nop"]
                X.append(np.r_[cx.Xw[r], E.masked_state(st, mask)])
                Y.append(nl[0])
                R.append((0.0, 0.0))
                Rm.append(0.0)
                W.append(nl[1])
                if rows_out is not None:
                    rows_out.setdefault("nop", []).append(int(r))
        for a, tp, sl in labs:
            for m, (y, is_open) in labels_for(variant, st, a).items():
                X, Y, R, Rm, W = acc[m]
                X.append(np.r_[cx.Xw[r], E.masked_state(st, mask)])
                Y.append(y)
                W.append(wt)
                if rows_out is not None:
                    rows_out.setdefault(m, []).append(int(r))
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
    return {m: (pack(*v[:4]) + (np.asarray(v[4], np.float32),) if (sample_weights or m == "nop") else pack(*v[:4])) for m, v in acc.items() if len(v[1])}


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


# ----------------------------------------------------------------------------- action contract v1 (owner 2026-10-25)
def realized_excursions(cx, horizon=72):
    """Per bar and side, in causal sigma at the fill bar: the favourable and the adverse excursion of an entry at the
    next open, up to the optimal expert's next pivot, ``horizon`` bars and the forced flat (the AuxTargets horizon).
    Diagnostics of the TP / SL heads only (never a label, never an input)."""
    ds = cx.ds
    o, h, l = (ds.bars[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW"))
    n = len(o)
    piv = np.array([i for i, _, _ in ds.pivots], np.int64)
    nxt = piv[np.minimum(np.searchsorted(piv, np.arange(n), side="right"), len(piv) - 1)] if len(piv) else np.full(n, n - 1)
    flat = forced_flat_bars(ds)
    sig = np.asarray(ds.sigma, float)
    fav = {1: np.full(n, np.nan), -1: np.full(n, np.nan)}
    adv = {1: np.full(n, np.nan), -1: np.full(n, np.nan)}
    for b in range(n - 2):
        end = int(min(nxt[b], b + horizon, flat[b], n - 1))
        s = sig[b + 1]
        if end <= b + 1 or not np.isfinite(s) or s <= 0:
            continue
        hh, ll = h[b + 1:end + 1].max(), l[b + 1:end + 1].min()
        fav[1][b], adv[1][b] = (hh - o[b + 1]) / s, (o[b + 1] - ll) / s
        fav[-1][b], adv[-1][b] = (o[b + 1] - ll) / s, (hh - o[b + 1]) / s
    return fav, adv


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(ok.sum())}
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)   # noqa: E731
    return {"pearson": float(np.corrcoef(x[ok], y[ok])[0, 1]), "spearman": float(np.corrcoef(rk(x[ok]), rk(y[ok]))[0, 1]), "n": int(ok.sum())}


def tp_sl_diagnostics(pols, D, rows, cx, exc) -> dict:
    """TP / SL heads of each opener on its OPEN-labelled rows (the Huber loss mask): MAE in sigma against the targets,
    and correlation with the realized favourable (TP) / adverse (SL) excursion of the row's bar."""
    fav, adv = exc
    out = {}
    for m, side in (("open_l", 1), ("open_s", -1)):
        if m not in D or m not in rows:
            continue
        n = len(rows[m])
        X, R, Rm = D[m][0][:n], D[m][2][:n], D[m][3][:n]
        k = np.asarray(Rm, float) > 0
        if not k.any():
            out[m] = {"rows": 0}
            continue
        _, rg = ACT.module_outputs(pols[m], X[k])
        bars = cx.ds.row_bar[np.asarray(rows[m], np.int64)[k]]
        out[m] = {"rows": int(k.sum()), "tp_mae_sigma": float(np.mean(np.abs(rg[:, 0] - R[k, 0]))), "sl_mae_sigma": float(np.mean(np.abs(rg[:, 1] - R[k, 1]))),
                  "tp_target_mean": float(R[k, 0].mean()), "sl_target_mean": float(R[k, 1].mean()), "sl_target_std": float(R[k, 1].std()),
                  "tp_vs_realized_favourable": _corr(rg[:, 0], fav[side][bars]), "sl_vs_realized_adverse": _corr(rg[:, 1], adv[side][bars]),
                  "mask": "rows whose label is an OPEN of this side (labels_for is_open)",
                  "bound_coverage": {q: {"min": float(rg[:, j].min()), "max": float(rg[:, j].max()), "lo": lo_, "hi": hi_,
                                         "share_at_lo": float(np.mean(rg[:, j] <= lo_ + 0.01 * (hi_ - lo_))), "share_at_hi": float(np.mean(rg[:, j] >= hi_ - 0.01 * (hi_ - lo_))),
                                         "target_share_at_bounds": float(np.mean((R[k, j] <= lo_ + 1e-9) | (R[k, j] >= hi_ - 1e-9)))}
                                     for j, (q, lo_, hi_) in enumerate((("tp", 0.25, 3.0), ("sl", 0.5, 3.0)))}}
    return out


def calibrate_modules(pols, D, bins=15):
    """Platt map per specialist on its calibration-block samples D[m] (INNER_A), with AUC / Brier / ECE / reliability
    before (raw softmax) and after."""
    cal = {}
    for m, pol in pols.items():
        z, _ = ACT.module_outputs(pol, D[m][0])
        y = np.asarray(D[m][1], int)
        a_, b_ = ACT.fit_platt(z, y)
        cal[m] = {"a": a_, "b": b_, "temperature": (1.0 / a_) if a_ > 0 else float("inf"), "rows": int(len(y)), "positives": int(y.sum()),
                  "single_class": bool(y.min() == y.max()) if len(y) else True,
                  "before": ACT.head_metrics(ACT.sigmoid(z), y, bins), "after": ACT.head_metrics(ACT.platt(z, a_, b_), y, bins)}
    return cal


def report_metrics(pols, cal, D, bins=15):
    """Head metrics on another block with the FROZEN calibration (reported only; nothing is fitted here)."""
    out = {}
    for m, pol in pols.items():
        if m not in D:
            continue
        z, _ = ACT.module_outputs(pol, D[m][0])
        y = np.asarray(D[m][1], int)
        out[m] = {"raw": ACT.head_metrics(ACT.sigmoid(z), y, bins), "calibrated": ACT.head_metrics(ACT.platt(z, cal[m]["a"], cal[m]["b"]), y, bins)}
    return out


NO_TRADE = ACT.Strategy(2.0, 1e9, 2.0, False, None)


def v1_seed(cx, seed, d):
    """One seed under the action contract v1. ``d``: contract, mods, labels, label_mode, mask, tracks, demo_vis,
    extra_vis, explore, nop_mr, fit_module, SCFG, led, NS, rows_all, cost, cal_rows, exc. Every fitted or chosen quantity
    (early stopping, Platt maps, visitation and final strategies) reads TRAIN / INNER_A only: D_tr / D_in come from
    TRAIN / INNER_A visits, the Platt maps from D_in, and the strategy optimiser sees the table restricted to the
    INNER_A rows (``ACT.freeze_strategy``). Returns (pols, infos, rec_extra, policy)."""
    AC, v, mask, tracks, SCFG = d["contract"], "FULL", d["mask"], d["tracks"], d["SCFG"]
    kw = dict(mask=mask, labels=d["labels"], label_mode=d["label_mode"])
    nopl = ACT.NopLabels(d["nop_mr"])
    rows0 = {}
    D_tr = merge_sets(build_sets(v, d["demo_vis"]["TRAIN"] + d["explore"]["TRAIN"], cx, True, nop_labels=nopl, **kw),
                      *[build_sets(v, d["extra_vis"][(t.name, "TRAIN")], cx, True, track=t, **kw) for t in tracks[1:]])
    D_in = merge_sets(build_sets(v, d["demo_vis"]["INNER_A"] + d["explore"]["INNER_A"], cx, True, nop_labels=nopl, rows_out=rows0, **kw),
                      *[build_sets(v, d["extra_vis"][(t.name, "INNER_A")], cx, True, track=t, rows_out=rows0, **kw) for t in tracks[1:]])
    min_opw = float(AC["strategy"]["calibration_min_trades_per_week"])

    sig = np.asarray(cx.ds.sigma, float)
    primary = d.get("fitness") or ACT.DEFAULT_FITNESS

    def score_table(view, strat):
        m, trades = ACT.rollout_with_trades(cx, "INNER_A", ACT.ContractPolicy(view, strat)(cx), SCFG)
        return ACT.window_components(m, trades, sig, SCFG.capital, min_opw, **ACT.score_kwargs(AC))

    pols, infos, comp, rounds = {}, {}, None, []
    for rnd in range(ROUNDS + 1):
        if rnd > 0:
            _, vtr = collect(cx, "TRAIN", comp(cx), SCFG)
            _, vin = collect(cx, "INNER_A", comp(cx), SCFG)
            multi = tracks if len(tracks) > 1 else None
            for m, D in build_sets(v, vtr, cx, False, tracks=multi, nop_labels=nopl, **kw).items():
                D_tr[m] = cat(D_tr[m], D) if m in D_tr else D
            for m, D in build_sets(v, vin, cx, False, tracks=multi, nop_labels=nopl, **kw).items():
                D_in[m] = cat(D_in[m], D) if m in D_in else D
        for m in d["mods"]:
            if m not in D_tr or m not in D_in or len(np.unique(D_tr[m][1])) < 2:
                if m in pols:
                    ledger(d["led"], rung="E3M", event="MODULE_KEPT", variant=v, module=m, seed=seed, round=rnd)
                    continue
                raise SystemExit(f"E3M_EMPTY_PARTITION:{v}:{m}:round{rnd}")
            ledger(d["led"], rung="E3M", event="FIT", variant=v, module=m, seed=seed, round=rnd, rows_train=len(D_tr[m][1]),
                   rows_inner=len(D_in[m][1]), label_counts=np.bincount(D_tr[m][1]).tolist())
            pols[m], infos[m] = d["fit_module"](v, m, D_tr[m], D_in[m], seed)
        cal = calibrate_modules(pols, D_in)
        table = ACT.calibrate_table(ACT.raw_table(pols, cx.Xw, d["rows_all"], d["NS"], d["cost"]), cal)
        if rnd == ROUNDS:
            break
        fr = ACT.freeze_strategy(table, d["cal_rows"], score_table, AC, seed, "visitation", primary)
        st_ = ACT.strategy_of(fr)
        rounds.append({"round": rnd, "visitation_strategy": fr["strategy"], "valid": fr["valid"], "nop_veto_kept": fr["nop_veto_kept"],
                       "chosen": fr["chosen"], "fallback": None if st_ else "inherited prob arbiter at the 6 % rate (no valid strategy on INNER_A)",
                       "archive": {k: r["points"] for k, r in fr["runs"].items()}, "term_scales": fr["term_scales"]})
        ledger(d["led"], rung="E3M", event="V1_VISITATION_STRATEGY", seed=seed, round=rnd, strategy=fr["strategy"], chosen=fr["chosen"])
        if st_ is not None:
            comp = ACT.ContractPolicy(table, st_)
        else:
            comp = Composed(v, pols, "prob", tau=0.5, mask=mask)
            comp.tau = float(np.quantile(comp.open_scores(cx, cx.ds.rows("INNER_A")), 1 - 0.06))
    fr = ACT.freeze_strategy(table, d["cal_rows"], score_table, AC, seed, "final", primary)
    st_ = ACT.strategy_of(fr) or NO_TRADE
    ledger(d["led"], rung="E3M", event="FREEZE", variant=v, seed=seed, arbiter="action_contract_v1", strategy=fr["strategy"], chosen=fr["chosen"],
           nop_veto_kept=fr["nop_veto_kept"], chosen_on="INNER_A")
    seen, nr, ny = set(), [], []
    for r, y, w, kind, arm_open in nopl.rows:
        if cx.ds.period[r] == "INNER_A" and r not in seen:
            seen.add(r)
            nr.append(r)
            ny.append(y)
    extra = {"calibration": {m: {k: c[k] for k in ("a", "b", "temperature", "rows", "positives", "single_class", "before", "after")} for m, c in cal.items()},
             "strategy": fr, "visitation_rounds": rounds, "nop_label_consistency": nopl.table,
             "nop_information_calibration_block": ACT.nop_information(table, nr, ny),
             "tp_sl_calibration_block": tp_sl_diagnostics(pols, D_in, rows0, cx, d["exc"]),
             "best_epochs": {m: int(infos[m]["best_epoch"]) for m in infos}}
    return pols, infos, extra, ACT.ContractPolicy(table, st_), table


def v1_round0_sets(cx, period, d, rows_out=None):
    """Round-0 label sets of a block (demonstrations + exploration positions + extra demonstrators, NOP included): the
    INNER_B report (frozen maps, reported only) and the ensemble's single Platt refit on INNER_A."""
    v, kw = "FULL", dict(mask=d["mask"], labels=d["labels"], label_mode=d["label_mode"])
    nopl = ACT.NopLabels(d["nop_mr"])
    return merge_sets(build_sets(v, d["demo_vis"][period] + d["explore"][period], cx, True, nop_labels=nopl, rows_out=rows_out, **kw),
                      *[build_sets(v, d["extra_vis"][(t.name, period)], cx, True, track=t, rows_out=rows_out, **kw) for t in d["tracks"][1:]])


def v1_context(cx, e3m_like: dict, ECFG, SCFG, periods=("TRAIN", "INNER_A")) -> dict:
    """Rebuild the label machinery of a v1 run from its recorded label arguments (E3-M itself and the seed ensemble):
    the arm's labels, the demonstrator tracks, the NOP ladder labels, the round-0 visits and the exploration positions
    (the TRAIN-prior random policy, seed 0, as the run). Deterministic: no network involved."""
    la = e3m_like["label_args"]
    ds = cx.ds
    cx.h, cx.l, cx.c = (ds.bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    mode = la["labels"]
    if mode == "multires":
        th = MR.Thresholds.from_json(la["multires_thresholds_json"]) if la.get("multires_thresholds_json") else MR.Thresholds()
        labels = MR.MultiResLabels(cx, th, MR.LADDER, DG.EXPERT)
    elif mode != "zigzag":
        labels = SignalLabels(cx, mode, la["closer_rule"], SG.SignalParams(theta_open=la["theta_open"], theta_close=la["theta_close"]))
    else:
        labels = None
    tracks = [ExpertTrack(cx)] + [parse_track(cx, x) for x in (la["extra_experts"].split(",") if la.get("extra_experts") else [])]
    _, vtr0 = collect(cx, "TRAIN", cx.expert, ECFG)
    prior = (np.bincount(np.array([q[3] for q in vtr0], int), minlength=4) / len(vtr0)).tolist()
    d = {"labels": labels, "label_mode": mode, "tracks": tracks, "mask": E.state_mask("none"),
         "nop_mr": MR.MultiResLabels(cx, MR.Thresholds(), MR.LADDER, DG.EXPERT),
         "demo_vis": {p: collect(cx, p, cx.expert, ECFG)[1] for p in periods},
         "extra_vis": {(t.name, p): collect(cx, p, t.policy(cx), ECFG)[1] for t in tracks[1:] for p in periods},
         "explore": {p: [q for q in collect(cx, p, cx.prior_random(prior, 0), SCFG)[1] if q[6] is not None] for p in periods}}
    return d


# ----------------------------------------------------------------------------- run
def effective_contracts(exec_contract: str, min_orders_per_week=None):
    """(student/evaluation config, expert-demonstration config) actually used by a run. ``sac`` = the replay contract
    (owner 2026-10-22: max 1 entry/day, leverage 1:50 everywhere); ``min_orders_per_week`` = the weekly activity rule
    (period average). One definition, read by the run and by the consumer recipe (identity-covered)."""
    if exec_contract == "sac":
        S = MUSASHI_CFG
        X_ = dataclasses.replace(EXPERT_CFG, max_entries_per_day=MUSASHI_CFG.max_entries_per_day, sizing=MUSASHI_CFG.sizing)
    elif exec_contract == "native":
        S, X_ = STUDENT_CFG, EXPERT_CFG
    else:
        raise ValueError(f"EXEC_CONTRACT_UNKNOWN:{exec_contract}")
    if min_orders_per_week is not None:
        S = dataclasses.replace(S, min_orders_per_week=float(min_orders_per_week))
        X_ = dataclasses.replace(X_, min_orders_per_week=float(min_orders_per_week))
    return S, X_


def run(a):
    import torch
    from app.short_horizon.expert_guided import learn as L
    from app.short_horizon.expert_guided import temporal as T
    out = Path(a.out)
    global SEEDS
    SEEDS = parse_seeds(getattr(a, "seeds", None) or ",".join(map(str, SEEDS_DEFAULT)))
    out.mkdir(parents=True, exist_ok=True)
    led = out / "TRIAL_LEDGER.jsonl"
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    t0 = time.time()
    verify = not a.no_verify
    DG.check_scaling(a.scaling)                   # typed refusal before any read
    sc = DG.split_contract(a.split_contract)
    AC = None
    C2MODE = False
    C3MODE = False
    if getattr(a, "outcomes", None) and not a.action_contract:
        raise SystemExit("OUTCOMES_REQUIRE_C2_ACTION_CONTRACT")
    if getattr(a, "strategy_fitness", "") and not a.action_contract:
        raise SystemExit("STRATEGY_FITNESS_REQUIRES_ACTION_CONTRACT")
    if a.action_contract:
        # output contract v1 (owner 2026-10-25): five calibrated specialists + one frozen parametric strategy; every
        # choice on INNER_A, decision on the post-freeze INNER_B; typed refusals before any data is read
        _acdoc = json.loads(Path(a.action_contract).read_text())
        C3MODE = C3.is_c3(_acdoc)                                              # stage C3 (2026-10-26): the audit's label / target fixes
        C2MODE = C2.is_c2(_acdoc) or C3MODE                                    # stage C2 (2026-10-25): action consequences, no teacher
        AC = C3.load(a.action_contract) if C3MODE else (C2.load(a.action_contract) if C2MODE else ACT.load(a.action_contract))
        ACT.require_decision_block(AC, sc, a.split_contract)
        if a.variants != "FULL":
            raise SystemExit("ACTION_CONTRACT_ONLY_FOR_FULL")
        if a.agent_state != "none":
            raise SystemExit("ACTION_CONTRACT_REQUIRES_AGENT_STATE_NONE")
        if a.aux_targets or getattr(a, "sample_weights", False):
            raise SystemExit("ACTION_CONTRACT_V1_REFUSES_AUX_TARGETS_AND_SAMPLE_WEIGHTS")
    mask, inventory = E.state_mask(a.agent_state), E.state_inventory(a.agent_state)
    SCFG, ECFG = effective_contracts(a.exec_contract, a.min_orders_per_week)
    if a.aux_targets and a.variants != "FULL":
        raise SystemExit("E3M_AUX_TARGETS_ONLY_FOR_FULL")
    if (a.window, a.stride) != (DG.WIN, DG.STRIDE) and not a.store:
        raise SystemExit("E3M_WINDOW_ONLY_ON_STORE_PATH")
    RECIPE = None
    if a.recipe:
        # the pinned shared recipe (audit A3 2026-10-23): columns, consumer operations, scaler constants, window and
        # groups come from the recipe; explicit flags that would contradict it are refused instead of overriding it
        RECIPE = json.loads(Path(a.recipe).read_text())
        if not str(RECIPE.get("identity", "")).startswith("recipe_") or RECIPE.get("schema") != "expert_guided_consumer_recipe.v2":
            raise SystemExit("E3M_RECIPE_MALFORMED")
        if a.feature_list or a.groups or a.scaling != "fixed_train":
            raise SystemExit("E3M_RECIPE_CONFLICT: --feature-list/--groups/--scaling come from the recipe")
        w = RECIPE["tensor"]["window"]
        if (a.window, a.stride) not in ((DG.WIN, DG.STRIDE), (int(w["observations"]), int(w["stride_rows"]))):
            raise SystemExit("E3M_RECIPE_CONFLICT: --window/--stride differ from the recipe")
        a.window, a.stride = int(w["observations"]), int(w["stride_rows"])
        if a.split_contract != RECIPE["split"]["arm"]:
            raise SystemExit(f"E3M_RECIPE_CONFLICT: --split-contract {a.split_contract} != {RECIPE['split']['arm']}")
        if a.agent_state != "none":
            raise SystemExit("E3M_RECIPE_CONFLICT: the shared recipe has no agent feedback (agent_state none)")
        rr = RECIPE.get("regularization", {})
        if (float(a.dropout), float(a.wd)) != (float(rr.get("dropout", -1)), float(rr.get("wd", -1))):
            raise SystemExit("E3M_RECIPE_CONFLICT: --dropout/--wd differ from the recipe's regularization")
    if a.store:
        feats = ([c["store_column"] for c in RECIPE["tensor"]["columns"]] if RECIPE else json.loads(Path(a.feature_list).read_text())["features"])
        ds = DG.load_store(a.admitted, a.store, feats, scaling=a.scaling, verify=verify, contract=a.split_contract, nonstationary=a.nonstationary, eligible_from=a.eligible_from,
                           win=a.window, stride=a.stride, recipe=RECIPE)
        if RECIPE and list(ds.meta["input_contract"]["features"]) != list(RECIPE["scaler"]["features"]):
            raise SystemExit("E3M_RECIPE_TENSOR_ORDER")
    elif RECIPE:
        raise SystemExit("E3M_RECIPE_REQUIRES_STORE")
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
    if (a.trunk, a.fusion_units, a.core_blocks, a.readout) != ("u", 8, 2, "flatten"):   # residual_tcn (2026-10-26): declared; defaults byte-identical
        ARCH.update(trunk=a.trunk, fusion_units=a.fusion_units, core_blocks=a.core_blocks, readout=a.readout)
    MODS = dict(MODULES)
    BOUNDS = {m: ((0.25, 3.0), (0.5, 3.0)) for v_ in MODULES for m in MODULES[v_]}
    if a.aux_targets:
        MODS["FULL"] = {"open_l": (2, 3), "open_s": (2, 3), "close_l": (2, 1), "close_s": (2, 1)}
        ob, cb = (MR.MultiResAux.BOUND, MR.MultiResAux.CLOSE_BOUND) if a.aux_targets == "multires" else (AuxTargets.BOUND, AuxTargets.BOUND)
        BOUNDS.update({"open_l": ((0.25, 3.0), (0.5, 3.0), ob), "open_s": ((0.25, 3.0), (0.5, 3.0), ob),
                       "close_l": (cb,), "close_s": (cb,)})
    if a.groups or RECIPE:
        # extractor groups: {"groups": {name: [columns]}, "units": {name: u}} matched to the tensor order by base name
        gspec = RECIPE["representation"]["groups"] if RECIPE else json.loads(Path(a.groups).read_text())
        tensor = [f.split("@")[0] for f in ds.meta["input_contract"]["features"]]
        names, idx, gu = [], [], []
        for g, cols in gspec["groups"].items():
            ii = [tensor.index(c.split("@")[0]) for c in cols if c.split("@")[0] in tensor]
            if ii:
                names.append(g); idx.append(ii); gu.append(int(gspec.get("units", {}).get(g, a.branch or a.units)))
        if sorted(i for g in idx for i in g) != list(range(nf)):
            raise SystemExit("E3M_GROUPS_DO_NOT_PARTITION_THE_TENSOR")
        gk = [str(gspec.get("kinds", {}).get(g, "conv")) for g in names]     # T3: extractor type per group (conv | pointwise)
        if RECIPE:
            r = RECIPE["representation"]["resolved"]
            if (names, idx, gu, gk) != (r["names"], r["indices"], r["units"], r["kinds"]):
                raise SystemExit("E3M_RECIPE_GROUPS_DO_NOT_RESOLVE")
            m = RECIPE["representation"]["model"]
            if (a.units, a.depth, a.head, a.core) != (m["units"], m["depth"], m["head"], m["core"]):
                raise SystemExit("E3M_RECIPE_CONFLICT: --units/--depth/--head/--core differ from the recipe model")
        ARCH.update(groups=idx, group_units=gu, group_names=names, group_kinds=gk)
    AUX = (MR.MultiResAux(cx, MR.LADDER, DG.EXPERT) if a.aux_targets == "multires" else AuxTargets(cx)) if a.aux_targets else None   # the optimum member of the ladder needs the registered params
    LMODE = a.labels
    SW = check_sample_weights(a)
    if a.labels == "multires":
        # owner 2026-10-24: ladder candidates scored against the optimum; thresholds from JSON (defaults declared in multires.Thresholds)
        TH = MR.Thresholds.from_json(Path(a.multires_thresholds).read_text()) if a.multires_thresholds else MR.Thresholds()
        LABELS = MR.MultiResLabels(cx, TH, MR.LADDER, DG.EXPERT)
    else:
        LABELS = (SignalLabels(cx, a.labels, a.closer_rule, SG.SignalParams(theta_open=a.theta_open, theta_close=a.theta_close))
                  if a.labels != "zigzag" else None)
    tracks = [ExpertTrack(cx)] + [ExpertTrack(cx, Z.Params(depth=DG.EXPERT.depth, deviation_pips=float(d), backstep=DG.EXPERT.backstep))
                                  for d in []] + [parse_track(cx, d) for d in (a.extra_experts.split(",") if a.extra_experts else [])]
    variants = [v for v in VARIANTS if v in a.variants.split(",")]
    NSM = {}                                  # per-module state width; empty = every module NS (byte-identical default)
    if AC and C3MODE:
        _cells = [tuple(c) for c in AC["outcomes"]["cells"]]
        MODS["FULL"] = C3.modules(_cells)
        BOUNDS.update({m: C3.reg_bounds(m, _cells) for m in C3.SPECIALISTS})
        NSM = {m: NS + C2.N_GEO for m in C2.CLOSERS}
    elif AC and C2MODE:
        MODS["FULL"] = C2.modules()
        BOUNDS.update({m: C2.reg_bounds(m) for m in C2.SPECIALISTS})
        if C2.is_c2g(AC):                     # C2G (owner 2026-10-25): the closers alone also read their order's geometry
            NSM = {m: NS + C2.N_GEO for m in C2.CLOSERS}
    elif AC:
        MODS["FULL"] = dict(MODS["FULL"], nop=(2, 0))
        BOUNDS["nop"] = ((0.25, 3.0), (0.5, 3.0))
    params = {v: {m: int(sum(p.numel() for p in T.TemporalPolicyNet(nf, WIN_E, nc, n_reg=nr, n_state=NSM.get(m, NS), reg_bounds=BOUNDS[m], **ARCH).parameters()))
                  for m, (nc, nr) in MODS[v].items()} for v in variants}
    HEADS = None
    if AC:                                    # runtime encoding check: the networks' heads match the contract's declared heads
        HEADS = ACT.check_head_shapes(AC, {m: T.TemporalPolicyNet(nf, WIN_E, nc, n_reg=nr, n_state=NSM.get(m, NS), reg_bounds=BOUNDS[m], **ARCH)
                                           for m, (nc, nr) in MODS["FULL"].items()})
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
           exec_contract=a.exec_contract, aux_targets=bool(a.aux_targets), aux_mode=(a.aux_targets or None), min_orders_per_week=SCFG.min_orders_per_week,
           extractor_groups=ARCH.get("group_names"), labels=a.labels, closer_rule=a.closer_rule if LABELS else None,
           action_contract=(AC["identity"] if AC else None),
           decision="variant x arbiter: highest seed-mean INNER_A Sharpe among valid; ties -> fewer models; 2024 once")
    scaler_path = DG.write_scaler(out, ds, rung="E3M", agent_state=a.agent_state, split_contract=a.split_contract)
    res = {"exec_contract": a.exec_contract, "min_orders_per_week": SCFG.min_orders_per_week, "seeds": list(SEEDS),
           "aux_targets": (AUX.describe() if a.aux_targets == "multires" else
                           {"opportunity_at_open": "best excursion before the optimal pivot (sigma, net of cost)",
                            "hold_gain_at_close": "best further gain vs closing now (sigma)",
                            "bound": AuxTargets.BOUND, "horizon": "next optimal pivot, <= 72 bars, <= forced flat"}
                           if a.aux_targets else None),
           "labels": (LABELS.describe(ds) | {"weights_in_loss": SW}) if LMODE == "multires" else (LABELS.describe(ds) if LABELS else {"mode": "zigzag"}),
           "recipe": ({"identity": RECIPE["identity"], "sha256": DG.sha256(a.recipe)} if RECIPE else None),
           "meta": ds.meta, "arch": ARCH, "params": params, "window": {"observations": WIN_E, "stride_rows": int(ds.meta["input_contract"]["stride_rows"])},
           "experts": [{"name": t.name, "primary": t.primary, "pivots": t.n_pivots} for t in tracks], "scaling": a.scaling, "variants": variants, "arbiters": list(ARBITERS),
           "cost_sigma": COST_SIGMA, "device": device, "comparators": {}, "trials": {}, "verify": verify,
           "split_contract": ds.meta["split_contract"], "agent_state": agent_state, "scaler_file": scaler_path.name,
           "probe": {}}
    if NSM:                                   # C2G only: the ensemble / audit rebuild each module with its own state width
        res["n_state_by_module"] = {m: NSM.get(m, NS) for m in MODS["FULL"]}
    if AC:
        res["action_contract"] = {"identity": AC["identity"], "sha256": ACT.sha256_file(a.action_contract), "contract": AC,
                                  "encoding_audit": (C2.encoding_audit(AC) if C2MODE else ACT.encoding_audit(AC)), "head_shapes": HEADS,
                                  "calibration_block": "INNER_A", "decision_block": "INNER_B", "read_once": "OBS_2024",
                                  "label_args": {"labels": a.labels, "closer_rule": a.closer_rule, "theta_open": a.theta_open, "theta_close": a.theta_close,
                                                 "extra_experts": a.extra_experts,
                                                 "multires_thresholds_json": Path(a.multires_thresholds).read_text() if a.multires_thresholds else ""}}
    if LMODE == "multires" and SW:
        res["labels"]["sample_weights"] = "consumed: every flat training row's CE term is scaled by its label fitness (--sample-weights)"
    probe_rows = cx.ds.rows("INNER_A")
    probe_rows = probe_rows[np.linspace(0, len(probe_rows) - 1, min(8, len(probe_rows))).astype(int)] if len(probe_rows) else probe_rows

    def probe(pol):
        """Logits of the saved module on fixed rows with a zero (flat-equivalent) state: ties the checkpoint to the tensors."""
        ns = int(getattr(getattr(pol, "m", None), "n_state", NS) or NS)      # C2G closers: NS + geometry (zeros here too)
        X = np.stack([np.r_[cx.Xw[r], np.zeros(ns, np.float32)] for r in probe_rows]).astype(np.float32) if len(probe_rows) else np.zeros((0, cx.Xw.shape[1] + ns), np.float32)
        lg, rg = pol(X)
        return {"rows": probe_rows.tolist(), "row_ts_ns": [int(cx.ds.row_ts[r]) for r in probe_rows], "state": "zeros",
                "logits": lg.tolist(), "reg": None if rg is None else rg.tolist()}

    FIT_EXTRA = {}                            # C3: loss-balance kwargs set by the contract's runner (default: unchanged)

    def fit_module(v, m, D_tr, D_in, seed):
        nc, nr = MODS[v][m]
        f = np.maximum(np.bincount(D_tr[1], minlength=nc) / len(D_tr[1]), 1e-3)   # 1/sqrt(freq), as E1-T
        w = 1 / np.sqrt(f)
        w = w / w[D_tr[1]].mean()
        model, hist, info = L.fit(D_tr[0], D_tr[1], D_in[0], D_in[1], seed=seed, n_cls=nc,
                                  R=D_tr[2] if nr else None, Rm=D_tr[3] if nr else None,
                                  Ri=D_in[2] if nr else None, Rmi=D_in[3] if nr else None, class_weight=w,
                                  sample_weight=D_tr[4] if len(D_tr) > 4 else None,
                                  dropout=a.dropout, lr=1e-3, wd=a.wd, batch=256, max_epochs=a.max_epochs, patience=10,
                                  device=device, net=lambda: T.TemporalPolicyNet(nf, WIN_E, nc, n_reg=nr, n_state=NSM.get(m, NS),
                                                                                  dropout=a.dropout, reg_bounds=BOUNDS[m], **ARCH), **FIT_EXTRA)
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
    if C2MODE:
        _ctx = {"cx": cx, "ds": ds, "a": a, "AC": AC, "res": res, "out": out, "SCFG": SCFG, "ECFG": ECFG, "MUSASHI_CFG": MUSASHI_CFG,
                "NS": NS, "led": led, "ledger": ledger, "summary": summary, "DEV": DEV, "SEEDS": SEEDS, "prior": prior,
                "fit_module": fit_module, "probe": probe, "ROUNDS": ROUNDS, "EXPERT": DG.EXPERT, "fit_extra": FIT_EXTRA, "sc": sc,
                "ARCH": ARCH, "split_name": a.split_contract, "RECIPE": RECIPE}
        res = C3.run_e3m(_ctx) if C3MODE else C2.run_e3m(_ctx)
        res["wall_seconds"] = time.time() - t0
        (out / "E3M.json").write_text(json.dumps(jsonable(res), indent=1))
        print(json.dumps({"decision": res["decision"], "wall": round(res["wall_seconds"], 1)}))
        return
    frozen = {}
    V1 = None
    if AC:
        rows_all = np.flatnonzero(cx.obs_ok)
        V1 = {"contract": AC, "mods": MODS["FULL"], "labels": LABELS, "label_mode": LMODE, "mask": mask, "tracks": tracks,
              "demo_vis": demo_vis, "extra_vis": extra_vis, "explore": explore, "fit_module": fit_module, "SCFG": SCFG, "led": led, "NS": NS,
              "nop_mr": MR.MultiResLabels(cx, MR.Thresholds(), MR.LADDER, DG.EXPERT), "rows_all": rows_all,
              "cost": ACT.row_cost_sigma(SCFG.profile.round_trip_price(), ds.sigma, ds.row_bar, AC["strategy"]["cost_sigma"]),
              "cal_rows": np.flatnonzero(cx.obs_ok & (ds.period == AC["blocks"]["calibration"])), "exc": realized_excursions(cx),
              "fitness": a.strategy_fitness or AC["strategy"]["fitness"]["default"]}
        res["action_contract"]["primary_fitness"] = V1["fitness"]
        res["action_contract"]["cost"] = {"cost_sigma_allowance": AC["strategy"]["cost_sigma"], "round_trip_price": SCFG.profile.round_trip_price(),
                                          "per_row_sigma_p50": float(np.nanmedian(V1["cost"][rows_all])), "units": "sigma multiples (price / causal sigma price)"}
        ib = {}
    for v in variants:
        for seed in SEEDS:
            if AC:
                pols, infos, extra, c, table = v1_seed(cx, seed, V1)
                frozen[(v, seed)] = c
                if not ib:
                    ib_vis = {"demo_vis": {"INNER_B": collect(cx, "INNER_B", cx.expert, ECFG)[1]},
                              "extra_vis": {(t.name, "INNER_B"): collect(cx, "INNER_B", t.policy(cx), ECFG)[1] for t in tracks[1:]},
                              "explore": {"INNER_B": [q for q in collect(cx, "INNER_B", cx.prior_random(prior, 0), SCFG)[1] if q[6] is not None]}}
                    ib_rows = {}
                    ib["sets"] = v1_round0_sets(cx, "INNER_B", dict(V1, **ib_vis), rows_out=ib_rows)
                    ib["rows"] = ib_rows
                calp = {m: extra["calibration"][m] for m in pols}
                extra["reported_only_INNER_B"] = {"heads": report_metrics(pols, calp, ib["sets"]),
                                                  "tp_sl": tp_sl_diagnostics(pols, ib["sets"], ib["rows"], cx, V1["exc"]),
                                                  "note": "frozen calibration and heads; nothing fitted or chosen on INNER_B"}
                chosen = extra["strategy"]["chosen"]
                best = {"arbiter": "action_contract_v1", "rate": None, "tau": None, "strategy": extra["strategy"]["strategy"],
                        "nop_veto_kept": extra["strategy"]["nop_veto_kept"], "sharpe": None if chosen is None else chosen["sharpe"],
                        "lcb": None if chosen is None else chosen["lcb"], "valid": chosen is not None}
                rec = {"fits": {m: {k: infos[m][k] for k in ("epochs", "best_epoch", "params", "stop_reason")} for m in infos},
                       "arbiter_trials": (extra["strategy"]["runs"]["with_veto"]["front"] + extra["strategy"]["runs"]["without_veto"]["front"]),
                       "arbiter": best, "trading": {}, "musashi_contract": {}, "action_contract": extra}
                for p in DEV:
                    rec["trading"][p] = summary(cx.rollout(p, c(cx), SCFG)[0])
                    rec["musashi_contract"][p] = summary(cx.rollout(p, c(cx), MUSASHI_CFG)[0])
                for f, bf in extra["strategy"]["by_fitness"].items():        # every fitness variant, same trained seed (no refit)
                    if bf["strategy"] is not None:
                        cf = ACT.ContractPolicy(table, ACT.strategy_from_dict(bf["strategy"]))
                        bf["trading"] = {p: summary(cx.rollout(p, cf(cx), SCFG)[0]) for p in DEV}
                    bu = bf["archive_view"]["best_unconstrained_by_fitness"]
                    if bu is not None:                                       # what the fitness would do without the rule: NOT SELECTABLE
                        cu = ACT.ContractPolicy(table, ACT.strategy_from_dict(bu["strategy"]))
                        bu["trading_not_selectable"] = {p: summary(cx.rollout(p, cu(cx), SCFG)[0]) for p in DEV}
                rec["action_contract"]["out_of_sample_activity"] = {
                    "INNER_B_trades_per_week": rec["trading"].get("INNER_B", {}).get("orders_per_week"),
                    "owner_rule": "below 2 trades/week on INNER_B = INVALID", "valid_INNER_B": rec["trading"].get("INNER_B", {}).get("valid")}
                ci_p = {p: ACT.period_activity(cx, p, c(cx), SCFG) for p in DEV} if ACT.is_v13(AC["strategy"]) else {}
                if ci_p:
                    rec["action_contract"]["activity_ci90"] = ci_p
                for m, pol in pols.items():
                    torch.save(pol.m.state_dict(), out / f"{v}_{m}_{seed}.pt")
                    res["probe"][f"{v}_{m}_{seed}"] = probe(pol)
                (out / "PROBE.json").write_text(json.dumps({"split_contract": a.split_contract, "agent_state": a.agent_state,
                                                            "scaling": a.scaling, "probe": res["probe"]}, indent=1))
                res["trials"].setdefault(v, {})[str(seed)] = rec
                print(json.dumps({"variant": v, "seed": seed, "arbiter": "action_contract_v1", "strategy": best["strategy"],
                                  "sharpe": {p: round(rec["trading"][p]["sharpe_daily_sqrt365"], 2) for p in rec["trading"]},
                                  "opw": {p: round(rec["trading"][p]["orders_per_week"], 2) for p in rec["trading"]}, "s": round(time.time() - t0, 1),
                                  "seed_sanity": {p: ACT.seed_sanity(rec["trading"][p], ci_p.get(p)) for p in rec["trading"]}}), flush=True)
                continue
            D_tr = merge_sets(build_sets(v, demo_vis["TRAIN"] + explore["TRAIN"], cx, True, mask=mask, aux=AUX, labels=LABELS, label_mode=LMODE, sample_weights=SW),
                              *[build_sets(v, extra_vis[(t.name, "TRAIN")], cx, True, mask=mask, track=t, aux=AUX, labels=LABELS, label_mode=LMODE, sample_weights=SW) for t in tracks[1:]])
            D_in = merge_sets(build_sets(v, demo_vis["INNER_A"] + explore["INNER_A"], cx, True, mask=mask, aux=AUX, labels=LABELS, label_mode=LMODE, sample_weights=SW),
                              *[build_sets(v, extra_vis[(t.name, "INNER_A")], cx, True, mask=mask, track=t, aux=AUX, labels=LABELS, label_mode=LMODE, sample_weights=SW) for t in tracks[1:]])
            pols, infos = {}, {}
            comp = None
            for rnd in range(ROUNDS + 1):
                if rnd > 0:
                    _, vtr = collect(cx, "TRAIN", comp(cx), SCFG)
                    _, vin = collect(cx, "INNER_A", comp(cx), SCFG)
                    multi = tracks if len(tracks) > 1 else None
                    for m, D in build_sets(v, vtr, cx, False, mask=mask, tracks=multi, aux=AUX, labels=LABELS, label_mode=LMODE, sample_weights=SW).items():
                        D_tr[m] = cat(D_tr[m], D) if m in D_tr else D
                    for m, D in build_sets(v, vin, cx, False, mask=mask, tracks=multi, aux=AUX, labels=LABELS, label_mode=LMODE, sample_weights=SW).items():
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
    DEC = AC["blocks"]["decision"] if AC else "INNER_A"
    for v in variants:
        sh = [res["trials"][v][str(s)]["trading"][DEC]["sharpe_daily_sqrt365"] for s in SEEDS]
        valid = all(res["trials"][v][str(s)]["trading"][DEC]["valid"] for s in SEEDS)
        means[v] = {"inner_a_sharpe_mean": float(np.mean(sh)), "all_valid": valid, "n_models": len(MODULES[v])}
        if AC:
            means[v] = {"decision_block": DEC, "seed_mean_sharpe": float(np.mean(sh)), "all_valid": valid, "n_models": len(MODS[v]),
                        "note": "one variant (FULL) under the contract: nothing to choose here; arms are decided between runs by the seed ENSEMBLE on INNER_B"}
    ok = [v for v in variants if means[v]["all_valid"]]
    chosen = max(ok or variants, key=lambda v: (means[v].get("inner_a_sharpe_mean", means[v].get("seed_mean_sharpe")), -means[v]["n_models"]))
    res["decision"] = {"means": means, "chosen": chosen, "rule": "highest seed-mean INNER_A Sharpe among all-valid; ties -> fewer models" if not AC
                       else "action contract v1: single FULL variant; decision between arms by the seed ensemble on INNER_B (launcher / report)"}
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
    ap.add_argument("--exec-variant", choices=["contract", "legacy", "full_f3_12h"], default="contract",
                    help="C3 only: the contract's declared execution variant (full_f3) or the audited legacy engine as a declared control")
    ap.add_argument("--trunk", choices=["u", "residual_tcn"], default="u", help="trunk variant (residual_tcn = Musashi's minimal grouped TCN, declared)")
    ap.add_argument("--fusion-units", type=int, default=8)
    ap.add_argument("--core-blocks", type=int, default=2)
    ap.add_argument("--readout", choices=["flatten", "last"], default="flatten")
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
    ap.add_argument("--min-orders-per-week", type=float, default=None, help="activity rule for valid (default: the env's inherited 3.0; owner 2026-10-22: 2.0)")
    ap.add_argument("--aux-targets", nargs="?", const="opportunity", choices=("opportunity", "multires"), default=None,
                    help="FULL only: regression heads trained on hindsight targets (training-only). Bare flag / 'opportunity': "
                         "opportunity (openers) and hold-gain (closers) in sigma; 'multires': f_open / f_close relative to the "
                         "optimum leg (multires.MultiResAux), the class labels stay those of --labels")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS_DEFAULT)), help="comma list of seeds (default the inherited three)")
    ap.add_argument("--labels", choices=list(SignalLabels.MODES) + [MR.MultiResLabels.MODE], default="zigzag",
                    help="training labels: zigzag (the expert's pivots, unchanged default), opportunity (per-agent hindsight "
                         "signal rev 2: openers by opportunity/adverse, closers by --closer-rule on the order's own SL), both (OR, fixed truth table), "
                         "multires (owner 2026-10-24: ladder candidates scored against the optimum; NOP hard/graded; closers by remaining/f_close)")
    ap.add_argument("--multires-thresholds", default="", help="JSON file with multires.Thresholds fields (default: the declared defaults)")
    ap.add_argument("--sample-weights", action="store_true", help="multires only: weigh every flat training row by its label fitness in the CE loss (default off)")
    ap.add_argument("--closer-rule", choices=list(SignalLabels.RULES), default="hold_through",
                    help="closer counterfactual for --labels opportunity|both: hold_through (default) or optimum (declared contrast)")
    ap.add_argument("--theta-open", type=float, default=SG.SignalParams.theta_open, help="minimum net opportunity (sigma) to label an entry")
    ap.add_argument("--theta-close", type=float, default=SG.SignalParams.theta_close, help="further gain (sigma) below which closing now is right")
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
    ap.add_argument("--recipe", default="", help="pinned shared consumer recipe (v2): columns, consumer operations, scaler constants, "
                    "window, groups and model come from it; nothing is measured or refitted (audit A3)")
    ap.add_argument("--feature-list", default="", help="JSON with {'features': [...]} selected from the store")
    ap.add_argument("--action-contract", default="", help="output contract v1 JSON (ACTION_CONTRACT_v1.json): five calibrated specialists "
                    "(open_l, open_s, nop, close_l, close_s) and one parametric strategy frozen on INNER_A, decided on the post-freeze INNER_B; default off")
    ap.add_argument("--outcomes", default="", help="C2 only: the stamped entry outcome table (npz); built and saved there when missing")
    ap.add_argument("--strategy-fitness", choices=[""] + list(ACT.FITNESS), default="",
                    help="action contract v1 only: the fitness variant that picks the frozen strategy point (default: the contract's, sharpe_lcb); "
                         "every variant is still optimised and recorded")
    ap.add_argument("--cpu", action="store_true")
    return ap


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
