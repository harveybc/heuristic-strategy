"""Multi-resolution ZigZag labels scored against the OPTIMUM (owner 2026-10-24; design ZIGZAG_MULTIRESOLUTION_DESIGN.md
§4). Training-side only: labels, weights and targets. NON-CAUSAL: everything here reads final pivots; nothing is ever an
observation.

The ladder R = {optimum, rev20, rev15, rev10, rev8, rev5} generates CANDIDATE points (each resolution's own pivot bars
say "open long / open short here"); the optimum alone SCORES them. For a decision bar b (decide after the completed bar,
fill at o[b+1]) let [s, e] be the optimum leg containing b (pivot s <= b < pivot e) and leg = |level[e] - level[s]|:

  f_open[d](b)   = d * (level[e] - o[b+1]) / leg          in [-1, 1]
                   1 = the optimal open; < 1 a late entry; NEGATIVE when b lies in the optimum's contrary leg (the
                   owner's rule: always relative to the optimum): -(adverse move from o[b+1] to the end of that leg) / leg
  mae[d](b)      = adverse excursion of the entry at o[b+1] until the optimal close e, over leg (same-direction legs only)
  leg_sigma(b)   = leg / sigma[b+1]   (the optimal leg in causal sigma: small = the leg does not pay the volatility)
  f_close[d](b)  = d * (c[b] - level[s_d]) / leg_d        for a position of side d: gain of a trade opened at the optimal
                   open s_d (the last start pivot of a d-leg at/before b) and closed at c[b]; 1 only at the intrabar
                   extreme, < 1 before, falling after the optimal close e_d
  remaining[d](b)= d * (level[e_d] - c[b]) / leg_d  before e_d (>= 0); d * (c[b] - level[e_d]) / leg_d after it (<= 0: the
                   give-back from the optimal close, so the optimum having passed always reads as nothing remaining)

Labels per flat origin (``origin_label``), with thresholds ``Thresholds``:
  NOP hard      no resolution votes a side                                        weight 1
  negative      the voted side's f_open < 0 (open against the optimum)            NOP, weight min(1, |f|)
  conflict      both sides voted                                                  NOP graded, weight 1 - f*
  NOP graded    f* < theta_f or mae > theta_mae or leg_sigma < theta_vol          weight 1 - f*
  open d        otherwise                                                         weight f*, TP = remaining gain in sigma
Closer (``close_label``): CLOSE when remaining < theta_close or f_close >= f_close_take, else HOLD; f_close is the
bounded auxiliary regression target of that side.

Availability (same discipline as ``SignalLabels``): a flat label at (b, row) is usable only when every pivot it depends
on is confirmed inside the row's period: the optimum leg end e (its confirmation bar) and, for every resolution of the
ladder, the confirmation that makes its label at b final (``label_support_end``); a closer label needs e confirmed.
Censored rows are DROPPED, never filled.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

import numpy as np

from .. import zigzag as Z
from . import env as E
from . import oracle as OR

H_NS = 3_600_000_000_000
LADDER = ("3", "rev20", "rev15", "rev10", "rev8", "rev5")
F_CLOSE_BOUND = (-1.0, 1.5)            # aux regression target bound (f_close may exceed 1 at the intrabar extreme only)
TP_BOUNDS, SL_DEMO = (0.25, 3.0), 2.0  # as the current demonstrators: TP = remaining gain in sigma, SL = 2 sigma
NOP, OPEN_L, OPEN_S = 0, 1, 2
KINDS = ("nop_hard", "negative", "conflict", "nop_graded", "open")


@dataclass(frozen=True)
class Thresholds:
    theta_f: float = 0.5        # minimum fitness relative to the optimum to label an open
    theta_mae: float = 0.5      # maximum adverse excursion before the optimal close, as a fraction of the leg
    theta_vol: float = 0.25     # minimum optimal leg in causal sigma (the smallest effective TP)
    theta_close: float = 0.1    # remaining gain (fraction of the leg) below which closing now is right
    f_close_take: float = 0.9   # fitness already captured at which the closer takes the gain
    min_votes: int = 1          # resolutions that must name a side

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Thresholds":
        d = json.loads(text)
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"MULTIRES_THRESHOLDS_UNKNOWN:{sorted(unknown)}")
        return cls(**{k: (int(v) if k == "min_votes" else float(v)) for k, v in d.items()})


def _pivot_arrays(pivots):
    pb = np.array([int(p[0]) for p in pivots], np.int64)
    pk = np.array([int(p[1]) for p in pivots], np.int64)
    pl = np.array([float(p[2]) for p in pivots], float)
    return pb, pk, pl


def leg_tables(o, h, l, c, sigma, pivots):
    """Per bar and side, the optimum-relative quantities above (NaN where undefined). ``pivots`` = the optimum's final
    alternating pivots [(bar, PEAK|VALLEY, level)]. Returns a dict of arrays; ``leg_end[b]`` is the pivot INDEX of e
    (the bar whose confirmation makes every opener quantity at b final) and ``close_end[d][b]`` that of e_d."""
    o, h, l, c, sigma = (np.asarray(x, float) for x in (o, h, l, c, sigma))
    n = len(c)
    pb, pk, pl = _pivot_arrays(pivots)
    f_open = {1: np.full(n, np.nan), -1: np.full(n, np.nan)}
    mae = {1: np.full(n, np.nan), -1: np.full(n, np.nan)}
    leg_sigma, leg_end = np.full(n, np.nan), np.full(n, -1, np.int64)
    for i in range(len(pb) - 1):
        s, e = int(pb[i]), int(pb[i + 1])
        leg = abs(pl[i + 1] - pl[i])
        dl = 1 if pk[i] == Z.VALLEY else -1                      # direction of the optimum leg [s, e]
        if leg <= 0 or e <= s:
            continue
        # suffix extremes of the path b+1 .. e for the adverse excursion of a same-direction entry
        suf_min, suf_max = np.empty(e - s), np.empty(e - s)
        mn, mx = np.inf, -np.inf
        for k in range(e, s, -1):
            mn, mx = min(mn, l[k]), max(mx, h[k])
            suf_min[k - s - 1], suf_max[k - s - 1] = mn, mx
        for b in range(s, min(e, n - 1)):
            op = o[b + 1]
            leg_end[b] = i + 1
            for d in (1, -1):
                f_open[d][b] = d * (pl[i + 1] - op) / leg
            if dl > 0:
                mae[1][b] = (op - suf_min[b - s]) / leg
            else:
                mae[-1][b] = (suf_max[b - s] - op) / leg
            sg = sigma[b + 1]
            leg_sigma[b] = leg / sg if np.isfinite(sg) and sg > 0 else np.nan
    f_close = {1: np.full(n, np.nan), -1: np.full(n, np.nan)}
    remaining = {1: np.full(n, np.nan), -1: np.full(n, np.nan)}
    close_end = {1: np.full(n, -1, np.int64), -1: np.full(n, -1, np.int64)}
    for d in (1, -1):
        start_kind = Z.VALLEY if d > 0 else Z.PEAK
        starts = np.flatnonzero(pk == start_kind)
        starts = starts[starts + 1 < len(pb)]                  # a start whose leg end exists
        if not len(starts):
            continue
        sb = pb[starts]
        j = np.searchsorted(sb, np.arange(n), side="right") - 1   # last d-leg start at/before b
        for b in range(n):
            if j[b] < 0:
                continue
            si = int(starts[j[b]])
            leg = d * (pl[si + 1] - pl[si])
            if leg <= 0:
                continue
            f_close[d][b] = d * (c[b] - pl[si]) / leg
            # before the optimal close: what the leg still offers (>= 0); after it: the give-back from the optimal
            # close level (<= 0 by construction), so "remaining < theta_close" fires once the optimum has passed
            remaining[d][b] = d * (pl[si + 1] - c[b]) / leg if b <= pb[si + 1] else d * (c[b] - pl[si + 1]) / leg
            close_end[d][b] = si + 1
    return {"f_open": f_open, "mae": mae, "leg_sigma": leg_sigma, "leg_end": leg_end,
            "f_close": f_close, "remaining": remaining, "close_end": close_end, "n": n}


def resolution_pivots(h, l, spec: str, optimum: list, optimum_conf, params=None):
    """(pivots, conf) of one ladder member: the optimum itself (its pivots and confirmation bars) or a reversal ZigZag
    'revP' (P pips) or a MetaTrader depth 'N' with the registered deviation/backstep."""
    spec = str(spec).strip()
    if params is not None and spec == str(params.depth):
        return list(optimum), list(optimum_conf)
    if spec.startswith("rev"):
        return OR.reversal_pivots(h, l, float(spec[3:]) * Z.PIP)
    if params is None:
        raise ValueError(f"MULTIRES_SPEC_NEEDS_PARAMS:{spec}")
    p = Z.Params(depth=int(spec), deviation_pips=params.deviation_pips, backstep=params.backstep)
    piv = Z.hindsight_pivots(h, l, p)
    return piv, Z.confirmation_bars_fast(h, l, piv, p)


@dataclass
class Tables:
    """Everything a label needs, per bar (hindsight) plus the per-row period ends for availability."""
    legs: dict
    votes: dict                 # {1: n_long votes per bar, -1: n_short votes per bar}
    dep_res: np.ndarray         # per bar: max over the ladder of label_support_end (−1 = some resolution censored)
    conf: np.ndarray            # per optimum pivot index: its confirmation bar (−1 censored)
    o: np.ndarray
    sigma: np.ndarray
    v_ns: np.ndarray            # bar start times
    row_bar: np.ndarray
    row_end_ns: np.ndarray      # per row: end of its period
    ladder: tuple

    @classmethod
    def build(cls, o, h, l, c, sigma, optimum, optimum_conf, resolutions: dict, v_ns, row_bar, row_end_ns):
        """``resolutions`` = {spec: (pivots, conf)} for every ladder member (the optimum included)."""
        n = len(c)
        legs = leg_tables(o, h, l, c, sigma, optimum)
        votes = {1: np.zeros(n, np.int64), -1: np.zeros(n, np.int64)}
        dep = np.zeros(n, np.int64)
        for spec, (piv, conf) in resolutions.items():
            y = OR.pivot_labels(n, piv)
            votes[1] += (y == OR.LONG_CLASS)
            votes[-1] += (y == OR.SHORT_CLASS)
            sup = OR.label_support_end(n, piv, conf)
            dep = np.where((dep < 0) | (sup < 0), -1, np.maximum(dep, sup))
        conf = np.asarray(list(optimum_conf), np.int64)
        return cls(legs, votes, dep, conf, np.asarray(o, float), np.asarray(sigma, float), np.asarray(v_ns, np.int64),
                   np.asarray(row_bar, np.int64), np.asarray(row_end_ns, np.int64), tuple(resolutions))

    @classmethod
    def from_ctx(cls, cx, ladder=LADDER, params=None):
        ds = cx.ds
        o, h, l, c = (ds.bars[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
        v = ds.bars.index.as_unit("ns").asi8
        pb = np.array([int(p[0]) for p in ds.pivots], np.int64)
        conf = getattr(ds, "conf", None)
        conf = np.asarray(ds.support_end, np.int64)[pb] if conf is None else np.asarray(conf, np.int64)
        res = {spec: resolution_pivots(h, l, spec, ds.pivots, conf, params) for spec in ladder}
        ends = {p: _utc_ns(e) for p, (s_, e) in ds.meta["periods"].items()}
        row_end = np.array([ends.get(p, 0) for p in ds.period], np.int64)
        return cls.build(o, h, l, c, np.asarray(ds.sigma, float), ds.pivots, conf, res, v, ds.row_bar, row_end)

    # ---- availability
    def _avail_ns(self, dep_bar: int) -> int:
        if dep_bar < 0 or dep_bar >= len(self.v_ns):
            return np.iinfo(np.int64).max
        return int(self.v_ns[dep_bar]) + H_NS

    def open_available(self, bar: int, row: int) -> bool:
        ei = int(self.legs["leg_end"][bar])
        if ei < 0 or self.dep_res[bar] < 0:
            return False
        dep = max(int(self.conf[ei]), int(self.dep_res[bar])) if self.conf[ei] >= 0 else -1
        return self._avail_ns(dep) <= int(self.row_end_ns[row])

    def close_available(self, bar: int, row: int, d: int) -> bool:
        ei = int(self.legs["close_end"][d][bar])
        if ei < 0 or self.conf[ei] < 0:
            return False
        return self._avail_ns(int(self.conf[ei])) <= int(self.row_end_ns[row])


def origin_label(tb: Tables, bar: int, th: Thresholds):
    """(cls, weight, tp_sigma, kind, fitness) of the flat origin ``bar`` (availability checked by the caller); None when
    a needed quantity is undefined."""
    L = tb.legs
    sides = [d for d in (1, -1) if tb.votes[d][bar] >= th.min_votes]
    if not sides:
        return NOP, 1.0, None, "nop_hard", np.nan
    f = {d: float(L["f_open"][d][bar]) for d in sides}
    if any(not np.isfinite(x) for x in f.values()):
        return None
    if len(sides) == 2:
        fs = max(f.values())
        return NOP, float(np.clip(1.0 - fs, 0.0, 1.0)), None, "conflict", fs
    d = sides[0]
    fs = f[d]
    if fs < 0:
        return NOP, float(min(1.0, abs(fs))), None, "negative", fs
    m, ls = float(L["mae"][d][bar]), float(L["leg_sigma"][bar])
    if not (np.isfinite(m) and np.isfinite(ls)):
        return None
    if fs < th.theta_f or m > th.theta_mae or ls < th.theta_vol:
        return NOP, float(np.clip(1.0 - fs, 0.0, 1.0)), None, "nop_graded", fs
    return (OPEN_L if d > 0 else OPEN_S), fs, fs * ls, "open", fs


def close_label(tb: Tables, bar: int, d: int, th: Thresholds):
    """(E.CLOSE | E.HOLD, f_close) for a position of side d at ``bar``; None when undefined."""
    fc, rem = float(tb.legs["f_close"][d][bar]), float(tb.legs["remaining"][d][bar])
    if not (np.isfinite(fc) and np.isfinite(rem)):
        return None
    return (E.CLOSE if (rem < th.theta_close or fc >= th.f_close_take) else E.HOLD), fc


def demonstration_policy(tb: Tables, th: Thresholds):
    """(bar, r, st, env) -> (action, tp, sl): the label-only demonstrator on the executable path (decide after bar b,
    fill at o[b+1], SL 2 sigma, TP = remaining gain to the optimal close in sigma). Acts only where its label is
    available inside the row's period; HOLD elsewhere. Scores a threshold vector by replay."""
    def act(bar, r, st, env):
        pos = int(st[0])
        if pos == 0:
            if not (bool(st[-1] > 0.5) and tb.open_available(bar, r)):
                return E.HOLD, 1.5, SL_DEMO
            lab = origin_label(tb, bar, th)
            if lab is None or lab[0] == NOP:
                return E.HOLD, 1.5, SL_DEMO
            return (E.LONG if lab[0] == OPEN_L else E.SHORT), float(np.clip(lab[2], *TP_BOUNDS)), SL_DEMO
        if not tb.close_available(bar, r, pos):
            return E.HOLD, 1.5, SL_DEMO
        cl = close_label(tb, bar, pos, th)
        return (E.HOLD if cl is None else cl[0]), 1.5, SL_DEMO
    return act


class MultiResLabels:
    """E3-M adapter with the ``SignalLabels`` interface (flat_action / positioned_action / available / describe), so
    ``build_sets`` consumes it under ``label_mode="multires"`` (censored rows dropped). ``weight(bar, r)`` and
    ``aux_close(bar, r, d)`` expose the sample weight and the bounded f_close target of the last/any row."""
    MODE = "multires"

    def __init__(self, cx, thresholds: Thresholds | None = None, ladder=LADDER, params=None):
        self.th = thresholds or Thresholds()
        self.tb = Tables.from_ctx(cx, ladder, params)
        self.ladder = tuple(ladder)
        self.n = {k: 0 for k in ("flat_used", "flat_censored", "closer_used", "closer_censored")}
        self.kinds = {k: 0 for k in KINDS}

    def available(self, bar, r, d=None) -> bool:
        return self.tb.open_available(bar, r) if d is None else self.tb.close_available(bar, r, d)

    def label(self, bar, r):
        """Full label tuple (cls, weight, tp_sigma, kind, fitness) or None when censored/undefined."""
        if not self.tb.open_available(bar, r):
            return None
        return origin_label(self.tb, bar, self.th)

    def flat_action(self, bar, r):
        lab = self.label(bar, r)
        if lab is None:
            return None
        self.kinds[lab[3]] += 1
        if lab[0] == NOP:
            return E.HOLD, 1.5, SL_DEMO
        return (E.LONG if lab[0] == OPEN_L else E.SHORT), float(np.clip(lab[2], *TP_BOUNDS)), SL_DEMO

    def weight(self, bar, r) -> float:
        lab = self.label(bar, r)
        return 0.0 if lab is None else float(lab[1])

    def positioned_action(self, bar, r, info):
        d = int(info[0])
        if not self.tb.close_available(bar, r, d):
            return None
        cl = close_label(self.tb, bar, d, self.th)
        return None if cl is None else cl[0]

    def aux_close(self, bar, r, d):
        """(f_close bounded, ok) for the closer regression head of side d."""
        if not self.tb.close_available(bar, r, d):
            return 0.0, 0.0
        cl = close_label(self.tb, bar, d, self.th)
        return (0.0, 0.0) if cl is None else (float(np.clip(cl[1], *F_CLOSE_BOUND)), 1.0)

    def profile(self, ds, period="TRAIN"):
        """Label properties on the period's eligible rows taken as flat origins, and the closer labels of a position of
        each side held at those rows (labels, not learning)."""
        rows = ds.rows(period)
        kinds = {k: 0 for k in KINDS}
        cens, w, fit, cls = 0, {k: [] for k in KINDS}, [], {NOP: 0, OPEN_L: 0, OPEN_S: 0}
        close = {1: {"CLOSE": 0, "HOLD": 0, "censored": 0}, -1: {"CLOSE": 0, "HOLD": 0, "censored": 0}}
        for r in rows:
            b = int(ds.row_bar[r])
            lab = self.label(b, r)
            if lab is None:
                cens += 1
            else:
                kinds[lab[3]] += 1
                w[lab[3]].append(lab[1])
                cls[lab[0]] += 1
                if np.isfinite(lab[4]):
                    fit.append(lab[4])
            for d in (1, -1):
                if not self.tb.close_available(b, r, d):
                    close[d]["censored"] += 1
                    continue
                cl = close_label(self.tb, b, d, self.th)
                if cl is None:
                    close[d]["censored"] += 1
                else:
                    close[d]["CLOSE" if cl[0] == E.CLOSE else "HOLD"] += 1
        fit = np.asarray(fit, float)
        return {"period": period, "rows": int(len(rows)), "censored": cens, "kinds": kinds,
                "classes": {"nop": cls[NOP], "open_long": cls[OPEN_L], "open_short": cls[OPEN_S]},
                "weight_mean": {k: (float(np.mean(v)) if v else None) for k, v in w.items()},
                "fitness_p10_p50_p90": [float(np.percentile(fit, q)) for q in (10, 50, 90)] if len(fit) else None,
                "closer": {"long": close[1], "short": close[-1]}}

    def describe(self, ds):
        return {"mode": self.MODE, "ladder": list(self.ladder), "thresholds": dataclasses.asdict(self.th),
                "fitness": "relative to the optimum leg (owner 2026-10-24): open = d*(level[e]-o[b+1])/leg, close = d*(c[b]-level[s_d])/leg_d",
                "nop": "hard when no resolution votes; graded (1 - f*) when f* < theta_f, mae > theta_mae or leg/sigma < theta_vol; "
                       "negative (|f|) when the voted side opens against the optimum; conflict (both sides) graded",
                "closer_rule": "CLOSE when remaining < theta_close or f_close >= f_close_take; aux target f_close bounded " + str(F_CLOSE_BOUND),
                "tp_head": "remaining gain to the optimal close in sigma bounded " + str(TP_BOUNDS), "sl_head": f"{SL_DEMO} sigma (demonstrator)",
                "availability_rule": "confirmation of the optimum leg end AND every ladder label support + 1 h <= period end; censored rows dropped",
                "sample_weights": "computed and profiled; the class-weighted loss does not consume them yet (declared)",
                "counts": self.n, "kinds_used": self.kinds,
                "profile": {p: self.profile(ds, p) for p in ds.meta["periods"] if p in set(ds.period) and p != "OBS_2024"}}


def _utc_ns(x) -> int:
    import pandas as pd
    return int(pd.Timestamp(x, tz="UTC").as_unit("ns").value)
