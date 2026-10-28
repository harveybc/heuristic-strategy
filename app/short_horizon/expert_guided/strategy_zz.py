"""C3Z: the C3 strategy consuming the multi-resolution ZigZag hierarchy (ZIGZAG_EXPERT_FEEDBACK_DESIGN §1 and §10).

Ideal head outputs per tick (perfect = the hindsight ladder on the path; real heads estimate them causally):
  coarse (long horizon, levels COARSE): remaining room g(t,d) = max_r d * (L_e^r - close_t) / sigma_t, direction of the
          coarse leg, bars to the coarse pivot;
  fine (short horizon, levels FINE): P(fine pivot AGAINST d within h) = 1[a fine leg in favour of d ends within h],
          bars to the next fine pivot, fresh favourable fine leg (started within h);
  ladder: f_open^r(t,d), f* = max_r f_open, votes n_d, MAE/leg and leg/sigma at the optimum level -> NOP hard / graded.
The outcome heads of C3 (P(TP/SL/TIME), net per cell, MAE per cell, closer remaining / P(SL first), NOP) stay the
reality judge: they choose the cell and the EV; the hierarchy gates the open and arms the early-close alarm (§10.3):

  OPEN side d only if   EV_d(best cell) > open_margin + eps  (C3, pooled margin)
                        and net(cell) >= risk_reward_floor * MAE(cell)          (no open when the possible gain is small
                                                                                  relative to the adverse move)
                        and not burst(t)                                         (causal realized-vol burst gate)
                        and coarse room g(t,d) >= zz_room_tp_mult * TP(cell)     (the long horizon pays the TP)
                        and no fine counter-pivot against d within h             (short-horizon timing)
                        and not NOP-hard on the ladder                           (no resolution names a side)
  CLOSE early when      predicted remaining < -close_margin_d
                        or (P(SL first) > close_p_d and predicted remaining < close_p_requires_rem_below)
                        or (fine alarm against the position and coarse room < zz_close_room)
  Every order carries the engine's broker-side SL and TP of the chosen cell.
Every coefficient lives in STRATEGY_PARAMS.json (strategy_params.py); this module reads them through ``configure``.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from .. import zigzag as Z
from . import c3 as C3
from . import env as E
from . import oracle as OR

DEFAULT_LEVELS = ("rev800", "rev400", "rev200", "rev100", "rev60", "rev20", "rev15", "3", "rev10", "rev8", "rev5")
# roles: the coarser levels rev400 / rev200 were added because rev100 captured the planted long-horizon switch with F1 0.05
# (behaviour matrix, H4 cell); the expert Depth-3 ZigZag ('3') behaves as a fine level on hourly synthetic paths (F1 0.91
# against the planted short oscillation). The per-cell ladder-to-horizon capture is measured and reported.
ROLES = {"coarse": ("rev800", "rev400", "rev200", "rev100"), "middle": ("rev60", "rev20", "rev15"), "fine": ("3", "rev10", "rev8", "rev5")}
EXPERT = Z.Params(depth=3, deviation_pips=40.7, backstep=0)

PARAMS = {"closer_mode": 1.0, "closer_adv_margin": 0.02, "tp_mode": 1.0, "k_tp_protect": 1.5, "forced_every_bars": 24.0, "zz_post_flip_bars": 240.0, "zz_counter_net_frac": 1.2, "observe_bars": 0.0, "ev_from_net": 1.0, "close_margin_floor": 0.0, "risk_reward_floor": 0.5, "burst_window_bars": 6, "burst_vol_ratio": 2.0, "zz_room_tp_mult": 1.75, "zz_fine_h": 6,
          "zz_timing_gate": 1.0, "zz_nop_gate": 1.0, "zz_close_room": 0.5, "close_p_requires_rem_below": 0.0, "zz_theta_f": 0.5, "zz_reverse_flip_bars": 960, "z_close": 1.0, "k_persist": 1}


def configure(doc: dict):
    """Load the C3Z coefficients from a validated STRATEGY_PARAMS document (and the C3 module constants it owns)."""
    for k in PARAMS:
        if k in doc:
            PARAMS[k] = float(doc[k])
    if "open_margin_eps" in doc:
        C3.OPEN_MARGIN_EPS = float(doc["open_margin_eps"])
    if "rr_margin_sigma" in doc:
        C3.RR_MARGIN_SIGMA = float(doc["rr_margin_sigma"])


# ----------------------------------------------------------------------------- the ladder quantities (ideal head outputs)
def _raw_pivots(h, l, spec: str):
    if spec.startswith("rev"):
        return OR.reversal_pivots(h, l, float(spec[3:]) * Z.PIP)[0]
    return Z.hindsight_pivots(h, l, Z.Params(depth=int(spec), deviation_pips=EXPERT.deviation_pips, backstep=EXPERT.backstep))


def level_pivots(h, l, spec: str):
    """[(bar, PEAK|VALLEY, level)] of one ladder member: the PRODUCTION implementations (reversal ZigZag in pips, the
    expert MetaTrader ZigZag Depth 3 / Deviation 40.7 / Backstep 0), both symmetric under price negation since
    2026-10-27 (the matrix certifies the ladder the strategy will see)."""
    return _raw_pivots(np.asarray(h, float), np.asarray(l, float), spec)


def _level_arrays(o, h, l, c, sigma, piv):
    n = len(c)
    out = {k: np.full((n, 2), np.nan) for k in ("room", "f_open", "mae_leg")}
    out["leg_dir"] = np.zeros(n)
    out["leg_end"] = np.full(n, -1, np.int64)
    out["leg_start"] = np.full(n, -1, np.int64)
    out["leg_sigma"] = np.full(n, np.nan)
    if len(piv) < 2:
        return out
    pb = np.array([p[0] for p in piv], np.int64)
    pl = np.array([p[2] for p in piv], float)
    for i in range(len(pb) - 1):
        s, e = int(pb[i]), int(pb[i + 1])
        if e <= s:
            continue
        leg = abs(pl[i + 1] - pl[i])
        dl = 1.0 if pl[i + 1] > pl[i] else -1.0
        idx = np.arange(s, min(e, n - 1))
        if not len(idx) or leg <= 0:
            continue
        sg = sigma[idx]
        nxt_o = o[idx + 1]
        out["leg_dir"][idx] = dl
        out["leg_end"][idx] = e
        out["leg_start"][idx] = s
        out["leg_sigma"][idx] = np.where(np.isfinite(sigma[idx + 1]) & (sigma[idx + 1] > 0), leg / sigma[idx + 1], np.nan)
        # adverse excursion of an entry in the leg's direction until the leg's end
        seg_l, seg_h = l[s + 1:e + 1], h[s + 1:e + 1]
        suf_min = np.minimum.accumulate(seg_l[::-1])[::-1]
        suf_max = np.maximum.accumulate(seg_h[::-1])[::-1]
        for j, d in enumerate((1.0, -1.0)):
            out["room"][idx, j] = np.where(np.isfinite(sg) & (sg > 0), d * (pl[i + 1] - c[idx]) / sg, np.nan)
            out["f_open"][idx, j] = d * (pl[i + 1] - nxt_o) / leg
        k = idx - s
        if dl > 0:
            out["mae_leg"][idx, 0] = (nxt_o - suf_min[k]) / leg
        else:
            out["mae_leg"][idx, 1] = (suf_max[k] - nxt_o) / leg
    return out


def ladder_levels(bars, sigma, levels=DEFAULT_LEVELS) -> dict:
    """The per-level part of the ladder (pivots and per-bar leg arrays): independent of every strategy coefficient,
    computed once per world and cached by the parameter optimiser."""
    o, hi, lo, c = (np.asarray(bars[k], float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
    sig = np.asarray(sigma, float)
    piv = {r: level_pivots(hi, lo, r) for r in levels}
    return {"pivots": piv, "arr": {r: _level_arrays(o, hi, lo, c, sig, piv[r]) for r in levels}, "levels": tuple(levels), "n": len(c)}


def ladder_quantities(bars, sigma, levels=DEFAULT_LEVELS, roles=ROLES, h: int | None = None, theta_f: float | None = None,
                      theta_mae: float = 0.5, theta_vol: float = 0.25, cached: dict | None = None) -> dict:
    """Per bar (hindsight; the IDEAL head outputs): coarse room / direction / tau, fine counter-pivot within h / tau /
    fresh favourable leg, f*, votes, NOP hard / graded, plus the per-level pivots (for the ladder-to-horizon mapping).
    ``cached`` = ladder_levels(...) of the same path (the aggregation below is the only part that depends on h / theta_f)."""
    h = int(PARAMS["zz_fine_h"] if h is None else h)
    theta_f = float(PARAMS["zz_theta_f"] if theta_f is None else theta_f)
    lv = cached if cached is not None else ladder_levels(bars, sigma, levels)
    piv, arr, levels, n = lv["pivots"], lv["arr"], lv["levels"], lv["n"]
    coarse = [r for r in roles["coarse"] if r in arr]
    fine = [r for r in roles["fine"] if r in arr]
    room = np.nanmax(np.stack([arr[r]["room"] for r in coarse]), axis=0) if coarse else np.full((n, 2), np.nan)
    c0 = coarse[0] if coarse else levels[0]
    direction = np.zeros(n)
    tau_coarse = np.full(n, np.nan)
    for r in (coarse or [c0]):                                   # the coarsest level that HAS a leg at the bar names the coarse direction
        a_ = arr[r]
        fill = (direction == 0) & (a_["leg_dir"] != 0)
        direction = np.where(fill, a_["leg_dir"], direction)
        tau_coarse = np.where(fill, a_["leg_end"] - np.arange(n), tau_coarse)
    counter = np.zeros((n, 2), np.int8)
    fresh = np.zeros((n, 2), np.int8)
    tau_fine = np.full(n, np.inf)
    t_idx = np.arange(n)
    for r in fine:
        a = arr[r]
        e, s0 = a["leg_end"], a["leg_start"]
        ok = e >= 0
        for j, d in enumerate((1.0, -1.0)):
            favour = ok & (a["leg_dir"] == d)
            counter[:, j] |= (favour & (e - t_idx > 0) & (e - t_idx <= h)).astype(np.int8)
            fresh[:, j] |= (favour & (t_idx - s0 >= 0) & (t_idx - s0 <= h)).astype(np.int8)
        tau_fine = np.where(ok, np.minimum(tau_fine, e - t_idx), tau_fine)
    f_stack = np.stack([arr[r]["f_open"] for r in levels])
    fstar = np.nanmax(f_stack, axis=0)
    votes = np.nansum(f_stack >= theta_f, axis=0)
    nop_hard = (votes[:, 0] == 0) & (votes[:, 1] == 0)
    m_opt = arr["3"]["mae_leg"] if "3" in arr else arr[c0]["mae_leg"]
    ls = arr["3"]["leg_sigma"] if "3" in arr else arr[c0]["leg_sigma"]
    nop_graded = (fstar < theta_f) | (np.nan_to_num(m_opt, nan=0.0) > theta_mae) | (np.nan_to_num(ls, nan=np.inf)[:, None] < theta_vol)
    flip_age = np.zeros(n)                                         # bars since the coarse direction last changed
    for t in range(1, n):
        flip_age[t] = 0.0 if direction[t] != direction[t - 1] else flip_age[t - 1] + 1.0
    return {"room": room, "direction": direction, "tau_coarse": tau_coarse, "flip_age": flip_age, "counter": counter, "fresh": fresh, "tau_fine": tau_fine,
            "fstar": fstar, "votes": votes, "nop_hard": nop_hard, "nop_graded": nop_graded, "pivots": piv, "levels": tuple(levels), "h": h}


# ----------------------------------------------------------------------------- the hierarchy heads below perfect quality (README:
# ZigZag supervision belongs to the training plane; its live counterpart is a model prediction with error)
def _ar1(rng, n: int, phi: float) -> np.ndarray:
    e = rng.standard_normal(n)
    out = np.empty(n)
    out[0] = e[0]
    k = math.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + k * e[t]
    return out


def degrade_quantities(q: dict, rho: float, seed: int) -> dict:
    """The hierarchy heads at quality rho (the same rho = sin(pi (level - 1/2)) as every other head): direction,
    counter-pivot, fresh-leg and NOP-hard flags misclassified through a latent score z = rho * (+-1) + sqrt(1 - rho^2) u
    with persistent noise u (AR(1)), so errors come in runs and the flips are misdated; flip_age recomputed from the
    noisy direction; the coarse room as a conditional-mean regression with correlation rho. Per-bar misclassification
    rate Phi(-rho / sqrt(1 - rho^2)) (0.23 at quality 0.70)."""
    if rho >= 1.0:
        return q
    rng = np.random.default_rng(int(seed))
    n = len(q["direction"])
    k = math.sqrt(max(1.0 - rho * rho, 0.0))
    out = dict(q)
    d = np.asarray(q["direction"], float)
    zd = rho * d + k * _ar1(rng, n, 0.995)
    out["direction"] = np.where(d != 0, np.sign(zd), 0.0)
    fa = np.zeros(n)
    for t in range(1, n):
        fa[t] = 0.0 if out["direction"][t] != out["direction"][t - 1] else fa[t - 1] + 1.0
    out["flip_age"] = fa
    for key, phi in (("counter", 0.9), ("fresh", 0.9)):
        y = np.asarray(q[key], float)
        z = rho * (2 * y - 1) + k * np.stack([_ar1(rng, n, phi), _ar1(rng, n, phi)], axis=1)
        out[key] = (z > 0).astype(np.int8)
    y = np.asarray(q["nop_hard"], float)
    out["nop_hard"] = (rho * (2 * y - 1) + k * _ar1(rng, n, 0.95)) > 0
    room = np.asarray(q["room"], float)
    fin = np.isfinite(room)
    if fin.any():
        mu, sd = float(np.mean(room[fin])), float(np.std(room[fin]))
        u = np.stack([_ar1(rng, n, 0.99), _ar1(rng, n, 0.99)], axis=1)
        out["room"] = np.where(fin, mu + rho * rho * (room - mu) + rho * k * sd * u, np.nan)
    out["head_mode"] = f"ladder rho={rho:.3f}"
    return out


def causal_quantities(bars, sigma, levels=DEFAULT_LEVELS, roles=ROLES, h: int | None = None, cached: dict | None = None) -> dict:
    """CAUSAL-CONFIRMED control: what any model can at least reach. Each production ZigZag pivot becomes known only
    when it is CONFIRMED (price moves the level's threshold away from it); at bar t the coarse direction is that of
    the last confirmed coarse leg, flip_age counts from its confirmation, a counter-pivot / fresh leg is an against /
    favourable fine pivot confirmed within the last h bars; the forward-looking room and the NOP-hard vote have no
    causal counterpart (NaN / off: their gates do not act)."""
    h = int(PARAMS["zz_fine_h"] if h is None else h)
    lv = cached if cached is not None else ladder_levels(bars, sigma, levels)
    piv, levels, n = lv["pivots"], lv["levels"], lv["n"]
    hi, lo = np.asarray(bars["HIGH"], float), np.asarray(bars["LOW"], float)

    def confirmed(spec):
        thr = (float(spec[3:]) if spec.startswith("rev") else EXPERT.deviation_pips) * Z.PIP
        dirs = np.zeros(n)
        conf = []
        for (bp, kind, lvl) in piv.get(spec, []):
            seg = (lo[bp + 1:] <= lvl - thr) if kind == Z.PEAK else (hi[bp + 1:] >= lvl + thr)
            if seg.any():
                conf.append((bp + 1 + int(np.argmax(seg)), kind))
        conf.sort()
        last = 0.0
        j = 0
        for t in range(n):
            while j < len(conf) and conf[j][0] <= t:
                last = -1.0 if conf[j][1] == Z.PEAK else 1.0
                j += 1
            dirs[t] = last
        return dirs, conf

    coarse = [r for r in roles["coarse"] if r in piv]
    fine = [r for r in roles["fine"] if r in piv]
    direction = np.zeros(n)
    for r in coarse:
        dr, _ = confirmed(r)
        direction = np.where(direction == 0, dr, direction)
    counter = np.zeros((n, 2), np.int8)
    fresh = np.zeros((n, 2), np.int8)
    for r in fine:
        _, conf = confirmed(r)
        for (tc, kind) in conf:
            win = slice(tc, min(n, tc + h + 1))
            if kind == Z.PEAK:                                       # a confirmed peak: against a long, favourable to a short
                counter[win, 0] = 1
                fresh[win, 1] = 1
            else:
                counter[win, 1] = 1
                fresh[win, 0] = 1
    fa = np.zeros(n)
    for t in range(1, n):
        fa[t] = 0.0 if direction[t] != direction[t - 1] else fa[t - 1] + 1.0
    nan2 = np.full((n, 2), np.nan)
    return {"room": nan2.copy(), "direction": direction, "tau_coarse": np.full(n, np.nan), "flip_age": fa, "counter": counter, "fresh": fresh,
            "tau_fine": np.full(n, np.nan), "fstar": nan2.copy(), "votes": np.zeros((n, 2)), "nop_hard": np.zeros(n, bool), "nop_graded": np.zeros((n, 2), bool),
            "pivots": piv, "levels": tuple(levels), "h": h, "head_mode": "causal-confirmed"}


def head_quantities(bars, sigma, quality: float, mode: str | None, seed: int, cached: dict | None = None) -> tuple:
    """(heads the strategy reads, hindsight quantities): 'perfect' (hindsight; at quality < 1 this is the OLD, INVALID
    inflated setting), 'ladder' (default below quality 1: the hierarchy heads carry the world's quality), 'causal'
    (the causal-confirmed control)."""
    q = ladder_quantities(bars, sigma, cached=cached)
    rho = float(min(1.0, max(0.0, math.sin(math.pi * (float(quality) - 0.5)))))   # the rho of every head (strategy_output_simulation.rho_of_level)
    m = mode or ("ladder" if float(quality) < 0.999 else "perfect")
    if m == "causal":
        return causal_quantities(bars, sigma, cached=cached), q
    if m == "ladder":
        return degrade_quantities(q, rho, 7919 * int(seed) + 17), q
    return q, q


def to_rows(q: dict, row_bar) -> dict:
    """The per-row table entries of the ideal hierarchy heads (row r reads its decision bar)."""
    rb = np.asarray(row_bar, np.int64)
    ok = (rb >= 0) & (rb < len(q["direction"]))
    b = np.where(ok, rb, 0)
    nan2 = lambda x: np.where(ok[:, None], x[b], np.nan)  # noqa: E731
    return {"zz_room": nan2(q["room"]), "zz_dir": np.where(ok, q["direction"][b], 0.0), "zz_counter": nan2(q["counter"].astype(float)),
            "zz_fresh": nan2(q["fresh"].astype(float)), "zz_tau_fine": np.where(ok, q["tau_fine"][b], np.nan),
            "zz_nop_hard": np.where(ok, q["nop_hard"][b].astype(float), np.nan), "zz_fstar": nan2(q["fstar"]),
            "zz_flip_age": np.where(ok, q["flip_age"][b], np.nan) if "flip_age" in q else np.full(len(rb), np.inf)}


def burst_rows(bars, sigma, row_bar, window: int | None = None, ratio: float | None = None) -> np.ndarray:
    """Causal burst flag per row: realized vol over the last ``window`` bars (scaled to 24 h) > ``ratio`` x causal sigma."""
    w = int(PARAMS["burst_window_bars"] if window is None else window)
    k = float(PARAMS["burst_vol_ratio"] if ratio is None else ratio)
    c = np.asarray(bars["CLOSE"], float)
    d2 = np.r_[0.0, np.diff(c) ** 2]
    cs = np.cumsum(d2)
    rv = np.full(len(c), np.nan)
    rv[w:] = np.sqrt((cs[w:] - cs[:-w]) * 24.0 / w)
    sig = np.asarray(sigma, float)
    flag = np.where(np.isfinite(rv) & np.isfinite(sig) & (sig > 0), rv > k * sig, False)
    rb = np.asarray(row_bar, np.int64)
    ok = (rb >= 0) & (rb < len(c))
    return np.where(ok, flag[np.where(ok, rb, 0)], False)


def net_open_margin(view: dict, rows, cells, open_rate: float) -> float:
    """The pooled open margin on the NET reference (ev_from_net): the (1 - open_rate) quantile, over the calibration rows
    and both sides, of the best allowed cell's predicted net (risk-reward floor applied) -- the same rule C3 applies to
    its EV reference."""
    vals = []
    for r in np.asarray(rows, np.int64):
        for si in (0, 1):
            net = view["net"][r, si]
            if not np.all(np.isfinite(net)):
                continue
            cap, lo = view.get("cap"), view.get("lo")
            ok = C3.allowed_cells(net, cells, 0.0, C3.RR_MARGIN_SIGMA, float(cap[r]) if cap is not None else float("nan"), float(lo[r]) if lo is not None else float("nan"))
            if view.get("mae") is not None:
                m = view["mae"][r, si]
                ok &= np.where(np.isfinite(m), net >= PARAMS["risk_reward_floor"] * m, True)
            if ok.any():
                vals.append(float(np.max(np.where(ok, net, -np.inf))))
    return max(float(np.quantile(vals, 1.0 - float(open_rate))), 0.0) if vals else float("inf")   # never open a predicted loss


# ----------------------------------------------------------------------------- the policy
class PolicyC3Z(C3.PolicyC3):
    """C3 + the hierarchy gates and close rule above. ``log`` (per flat decision) and ``closes`` (per positioned check)
    record every input of the decision and which gate / trigger acted (the per-side audit and the horizon attribution)."""

    def __init__(self, table, strategy, cells=C3.CELLS, e_time=(0.0, 0.0), cost: float = 0.0, observe_bars: int = 0):
        super().__init__(table, strategy, cells, e_time, cost, observe_bars)
        self.log, self.closes, self.attr, self.positions = [], [], [], {}

    def _cost_at(self, r):
        c = self.t.get("cost_row")
        return float(c[r]) if c is not None and np.isfinite(c[r]) else self.cost

    def side_choice(self, r, si, cost):
        net = np.asarray(self._v("net", r, si), float)
        p3 = np.asarray(self._v("p3", r, si), float)
        cap, lo = self._cap(r)
        # the predicted net is the ENGINE net (every cost inside): the TP < SL rule asks net > rr_margin, not net > cost + rr_margin
        ok = C3.allowed_cells(net, self.cells, 0.0 if PARAMS["ev_from_net"] >= 0.5 else cost, C3.RR_MARGIN_SIGMA, cap, lo)
        mae = self.t.get("mae")
        if mae is not None:
            m = np.asarray(mae[r, si], float)
            ok &= np.where(np.isfinite(m), net >= PARAMS["risk_reward_floor"] * m, True)
        if not ok.any():
            return -1, float("nan"), "rr_floor_or_band"
        ci = C3.argmax_cell(net, ok)
        tp, sl = float(self.cells[ci][0]), float(self.cells[ci][1])
        self._tp_order = tp
        mfe = self.t.get("mfe")
        if PARAMS["tp_mode"] >= 0.5 and mfe is not None and np.isfinite(mfe[r, si, ci]):
            self._tp_order = max(tp, PARAMS["k_tp_protect"] * float(mfe[r, si, ci]))   # protective envelope beyond the predicted excursion
        if PARAMS["ev_from_net"] >= 0.5:
            ev = float(net[ci])                                    # the engine net of the chosen cell (every cost inside) ranks the sides
        else:
            ev = float(p3[0]) * tp - float(p3[1]) * sl + float(p3[2]) * float(self.e_time[si]) - float(cost)
        return ci, ev, ""

    def open_decision(self, r, with_score: bool = False):
        s = self.s
        cost = self._cost_at(r)
        gates = {}
        vetoed = s.nop_veto is not None and float(self._v("pn", r)) > s.nop_veto
        burst = bool(self.t.get("risk_burst") is not None and self.t["risk_burst"][r])
        nop_hard = bool(PARAMS["zz_nop_gate"] >= 0.5 and self.t.get("zz_nop_hard") is not None and self.t["zz_nop_hard"][r] == 1.0)
        evs, cis, cleared, tps = [float("nan")] * 2, [-1, -1], [False, False], [float("nan")] * 2
        for si in (0, 1):                                        # C3 (reality judge): cell, EV, pooled margin, risk-reward floor
            ci, ev, why = self.side_choice(r, si, cost)
            cis[si], evs[si] = ci, ev
            tps[si] = getattr(self, "_tp_order", float("nan")) if ci >= 0 else float("nan")
            if ci < 0:
                gates[si] = why
                continue
            if not (np.isfinite(ev) and ev > s.open_margin + C3.OPEN_MARGIN_EPS):
                gates[si] = "margin"
                continue
            cleared[si] = True
        blocked = "nop_veto" if vetoed else ("burst" if burst else ("nop_hard" if nop_hard else ""))
        chosen = -1
        if cleared[0] or cleared[1]:
            if cleared[0] and cleared[1]:
                chosen = -1 if abs(evs[0] - evs[1]) <= C3.TIE_EPS else (0 if evs[0] > evs[1] else 1)
            else:
                chosen = 0 if cleared[0] else 1
        if chosen >= 0 and not blocked:                          # the hierarchy VETOES the winner, it never redirects to the other side
            room, ctr, fr = self.t.get("zz_room"), self.t.get("zz_counter"), self.t.get("zz_fresh")
            tp = float(self.cells[cis[chosen]][0])
            fa = self.t.get("zz_flip_age")
            settled = bool(fa is None or not np.isfinite(fa[r]) or fa[r] > PARAMS["zz_post_flip_bars"])     # right after a coarse flip, no exception
            strong = settled and bool(np.isfinite(evs[chosen]) and evs[chosen] >= PARAMS["zz_counter_net_frac"] * tp)   # the outcome heads say it pays most of its TP
            if room is not None and np.isfinite(room[r, chosen]) and room[r, chosen] < PARAMS["zz_room_tp_mult"] * tp \
                    and not (fr is not None and fr[r, chosen] == 1.0) and not strong:   # against the long horizon only at a confirmed fine counter-pivot or a strong predicted net
                blocked = "coarse_room"
            elif PARAMS["zz_timing_gate"] >= 0.5 and ctr is not None and ctr[r, chosen] == 1.0:
                blocked = "fine_counter_pivot"
        if blocked:
            chosen_before = chosen
            chosen = -1
        else:
            chosen_before = chosen
        self.log.append((int(r), evs[0], evs[1], cleared[0], cleared[1], chosen, bool(vetoed), cis[0], cis[1]))
        self.attr.append({"row": int(r), "blocked": blocked, "gate_l": gates.get(0, ""), "gate_s": gates.get(1, ""), "chosen": chosen, "winner_before_veto": chosen_before,
                          "zz_dir": float(self.t["zz_dir"][r]) if self.t.get("zz_dir") is not None else 0.0,
                          "fresh": (float(self.t["zz_fresh"][r, chosen]) if chosen >= 0 and self.t.get("zz_fresh") is not None else float("nan"))})
        if chosen < 0:
            return None
        d = E.LONG if chosen == 0 else E.SHORT
        out = (d, tps[chosen] if np.isfinite(tps[chosen]) else self.cells[cis[chosen]][0], self.cells[cis[chosen]][1])
        return out + (evs[chosen],) if with_score else out

    def reverse_ok(self, r) -> bool:
        """A reversal (close + open of the other side on the same decision) only within zz_reverse_flip_bars of a coarse
        direction flip; otherwise a plain close. The behaviour matrix found the reversals of the run-2 DEAP point outside
        every switch window (LHS_46 s0: 21 reversals, 1 inside, sealed W5 validated by the hold oracle)."""
        fa = self.t.get("zz_flip_age")
        return bool(fa is None or not np.isfinite(fa[r]) or fa[r] <= PARAMS["zz_reverse_flip_bars"])

    def close_now(self, r, si, bar, pos) -> bool:
        self.positions.setdefault((int(pos.entry_bar), int(si)), dataclasses.replace(pos))
        if not self.s.early_close:
            return False
        cl = self.t["closer"]
        if PARAMS["closer_mode"] >= 0.5 and hasattr(cl, "advantage"):
            # close when A_hat - z_close x s_hat >= -margin (the head's calibrated conditional mean and predictive sd), the
            # condition holding on k_persist consecutive executable ticks of this position; the broker SL stays the backstop
            if hasattr(cl, "advantage_full"):
                a, s_hat = cl.advantage_full(r, bar, pos)[:2]
            else:
                a, s_hat = cl.advantage(r, bar, pos), 0.0
            cond = bool(np.isfinite(a) and (a - PARAMS["z_close"] * (s_hat if np.isfinite(s_hat) else 0.0)) >= -PARAMS["closer_adv_margin"])
            key = (int(pos.entry_bar), int(si))
            self._persist = getattr(self, "_persist", {})
            self._persist[key] = self._persist.get(key, 0) + 1 if cond else 0
            dec = bool(cond and self._persist[key] >= int(round(PARAMS["k_persist"])))
            rem, psl = cl.value(r, bar, pos)
            self.closes.append((int(r), int(bar), int(si), float(rem), float(psl), dec, False, False, False, float(a), "advantage"))
            return dec
        rem, psl = cl.value(r, bar, pos)
        mar = max(self.s.close_margin_l if si == 0 else self.s.close_margin_s, PARAMS["close_margin_floor"])   # never close a position worth holding
        p = self.s.close_p_l if si == 0 else self.s.close_p_s
        trig_rem = bool(rem < -mar)
        trig_sl = bool(psl > p and rem < PARAMS["close_p_requires_rem_below"])
        ctr, room = self.t.get("zz_counter"), self.t.get("zz_room")
        alarm = bool(ctr is not None and ctr[r, si] == 1.0)
        weak = bool(room is not None and np.isfinite(room[r, si]) and room[r, si] < PARAMS["zz_close_room"])
        trig_zz = alarm and weak and bool(rem < PARAMS["close_p_requires_rem_below"])   # the alarm never overrides a positive remaining value
        dec = trig_rem or trig_sl or trig_zz
        self.closes.append((int(r), int(bar), int(si), float(rem), float(psl), dec, trig_rem, trig_sl, trig_zz, float("nan"), "legacy"))
        return dec


class PolicyForced(PolicyC3Z):
    """FORCED ENTRY (exit audit independent of the opener): at every observable decision bar on the schedule
    (bar // every_bars changes), open the side that alternates with the schedule index, whatever the heads say, with
    the strategy's own TP / SL envelope for that side (cell or protective TP; the best band-allowed cell by predicted
    net, or 1 / 1 sigma when none is allowed); every exit is the strategy's closer (or the broker SL / TP / cutoff)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.last_slot = -1

    def open_decision(self, r, with_score: bool = False):
        rb = self.t.get("row_bar")
        bar = int(rb[r]) if rb is not None else int(r)
        every = max(int(PARAMS["forced_every_bars"]), 1)
        slot = bar // every
        if slot == self.last_slot:
            return None
        self.last_slot = slot
        si = int(slot % 2)
        net = np.asarray(self._v("net", r, si), float)
        cap, lo = self._cap(r)
        ok = C3.allowed_cells(net, self.cells, 0.0, 1e9, cap, lo) | np.zeros(len(self.cells), bool)
        if ok.any():
            ci = C3.argmax_cell(net, ok)
            tp, sl = float(self.cells[ci][0]), float(self.cells[ci][1])
            mfe = self.t.get("mfe")
            if PARAMS["tp_mode"] >= 0.5 and mfe is not None and np.isfinite(mfe[r, si, ci]):
                tp = max(tp, PARAMS["k_tp_protect"] * float(mfe[r, si, ci]))
        else:
            ci, tp, sl = -1, 1.0, 1.0
        self.log.append((int(r), float("nan"), float("nan"), si == 0, si == 1, si, False, ci if si == 0 else -1, ci if si == 1 else -1))
        out = (E.LONG if si == 0 else E.SHORT, tp, sl)
        return out + (0.0,) if with_score else out
