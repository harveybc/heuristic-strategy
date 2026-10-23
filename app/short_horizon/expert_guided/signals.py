"""Hindsight expert signals per agent (owner 2026-10-22): the ideal learning signal of each specialist, derived from
the future price path up to the OPTIMAL expert's next FAVOURABLE pivot. Training-side only: labels, targets and
masks; never an observation. Every value at bar b needs the bars up to its horizon, so a row may use it only when
that horizon ends inside the row's period (purge by real label support).

Horizon of side d at bar b (decision after the completed bar b; execution at the next open o[b+1]):
  H[d](b) = the next optimal pivot of the FAVOURABLE kind for d (peak for a long, valley for a short) after b,
            capped by b + max_bars, the forced flat of b's week and the end of the data. ``horizon_reason[d]`` says
            which one ended it: "pivot" (the favourable extreme lies inside the window) or "cap" / "flat" / "end"
            (the window was TRUNCATED: the values below are then lower bounds taken inside the window, and the
            labels carry ``truncated = 1`` so a consumer can censor them; nothing beyond the window is ever read,
            not even the kind of a later pivot — auditor's witness 2026-10-22).
The same horizon serves both agents of side d, so an opener and a closer looking at the same bar never contradict
each other (auditor's witness 3): a long worth opening is a long worth holding.
Per bar b and side d in {+1, -1}, all in causal sigma (sigma[b+1] uses bars <= b):
  opportunity[d]        = max favourable excursion of an entry at o[b+1] over (b+1 .. H], net of cost
                          -> what an OPENER could earn at most before the optimum ("cuánto es lo máximo que podrá ganar");
                          an upper bound on HIGH/LOW, not an executable exit price
  adverse_to_peak[d]    = max adverse excursion from o[b+1] until the bar of that favourable peak
                          -> the SL an opener needs to survive until its best exit
  bars_to_peak[d]       = bars from b+1 to the favourable peak
  hold_gain[d]          = max further favourable move from c[b] over (b+1 .. H]
                          -> what a CLOSER gives up by closing now ("cuánto más pudo haber ganado respecto al óptimo")
  hold_adverse[d]       = max adverse move from c[b] until the bar of that further peak -> what a CLOSER risks by holding
Two CLOSER counterfactuals are declared (auditor 2026-10-22 §3: define HOLD vs CLOSE per task and contrast them in the
inner block instead of forcing agreement):
  closer_rule = "hold_through"  CLOSE iff hold_gain[d] < theta_close or hold_adverse[d] >= sl_dist[d]: the position is
                                kept through a survivable dip when a higher favourable extreme lies ahead inside the
                                window (consistent with the opener at the same bar: a long worth opening is worth holding)
  closer_rule = "optimum"       the optimal expert closes at every favourable pivot and re-enters: CLOSE iff the next
                                optimal pivot of ANY kind inside the window is adverse (the optimum exit has passed) or
                                hold_gain[d] < theta_close or hold_adverse[d] >= sl_dist[d]. ``next_adverse_inside[d]``
                                is NaN when no pivot of any kind lies inside the window (censored, never guessed).
Labels (declared thresholds, in sigma):
  opener[d]   = 1 if opportunity[d] >= theta_open and adverse_to_peak[d] <= sl_max   (a profitable, survivable entry)
  nop         = 1 if neither opener label is 1
The closer's label depends on the order's SL: without agent feedback, two identical market observations with
different SLs can demand different labels (partial observability, recorded; the SL is NOT fed back as an input).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalParams:
    max_bars: int = 72
    cost_sigma: float = 0.06
    theta_open: float = 0.5        # minimum net opportunity to call an entry (>= the smallest effective TP, 0.25σ, + cost)
    sl_max: float = 2.0            # an entry must survive its path to the peak inside the lane's SL (2σ)
    theta_close: float = 0.1       # below this further gain, closing now is right (cost-sized)


def _next_after(bars, n):
    """For every bar b, the first entry of ``bars`` strictly after b (n when none)."""
    bars = np.asarray(bars, np.int64)
    if not len(bars):
        return np.full(n, n)
    j = np.searchsorted(bars, np.arange(n), side="right")
    return np.where(j < len(bars), bars[np.minimum(j, len(bars) - 1)], n)


def hindsight_signals(o, h, l, c, sigma, pivots, flat_bar, p: SignalParams = SignalParams()):
    """``pivots``: [(bar, kind, level)] of the optimal expert (kind +1 peak / -1 valley). O(n * max_bars).
    Returns (signals: dict of arrays (n,) per side incl. ``truncated``, horizons: {d: {"bar": (n,), "reason": (n,) str}})."""
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    sigma = np.asarray(sigma, float)
    n = len(c)
    pb = np.asarray([i for i, _, _ in pivots], np.int64)
    pk = np.asarray([k for _, k, _ in pivots], np.int64)
    flat = np.asarray(flat_bar, np.int64)
    out = {k: {1: np.full(n, np.nan), -1: np.full(n, np.nan)} for k in
           ("opportunity", "adverse_to_peak", "bars_to_peak", "hold_gain", "hold_adverse", "truncated", "next_adverse_inside")}
    hz = {d: {"bar": np.full(n, -1, np.int64), "reason": np.full(n, "", dtype=object)} for d in (1, -1)}
    nxt_fav = {1: _next_after(pb[pk == 1], n), -1: _next_after(pb[pk == -1], n)}
    nxt_any = _next_after(pb, n)
    kind_next = np.zeros(n, np.int64)
    if len(pb):
        j = np.searchsorted(pb, np.arange(n), side="right")
        kind_next = np.where(j < len(pb), pk[np.minimum(j, len(pb) - 1)], 0)
    for b in range(n - 2):
        s = sigma[b + 1]
        if not np.isfinite(s) or s <= 0:
            continue
        e = o[b + 1]
        for d in (1, -1):
            cands = {"pivot": int(nxt_fav[d][b]), "cap": b + p.max_bars, "flat": int(flat[b]), "end": n - 1}
            reason = min(cands, key=lambda k: (cands[k], ("pivot", "flat", "cap", "end").index(k)))
            end = cands[reason]
            if end <= b + 1:
                continue
            hz[d]["bar"][b], hz[d]["reason"][b] = end, reason
            hh, ll = h[b + 1:end + 1], l[b + 1:end + 1]
            fav = (hh - e) if d > 0 else (e - ll)
            adv = (e - ll) if d > 0 else (hh - e)
            k = int(np.argmax(fav))
            out["opportunity"][d][b] = fav[k] / s - p.cost_sigma
            out["adverse_to_peak"][d][b] = max(float(adv[:k + 1].max()), 0.0) / s
            out["bars_to_peak"][d][b] = k + 1
            favc = (hh - c[b]) if d > 0 else (c[b] - ll)
            advc = (c[b] - ll) if d > 0 else (hh - c[b])
            kc = int(np.argmax(favc))
            out["hold_gain"][d][b] = max(float(favc[kc]), 0.0) / s
            out["hold_adverse"][d][b] = max(float(advc[:kc + 1].max()), 0.0) / s
            out["truncated"][d][b] = 0.0 if reason == "pivot" else 1.0
            # the next optimal pivot of ANY kind, used only when it lies INSIDE this window (never read beyond it)
            if b < nxt_any[b] <= end and kind_next[b] != 0:
                out["next_adverse_inside"][d][b] = float(kind_next[b] != d)
    return out, hz


def opener_labels(sig, d, p: SignalParams = SignalParams()):
    opp, adv = sig["opportunity"][d], sig["adverse_to_peak"][d]
    ok = np.isfinite(opp) & np.isfinite(adv)
    return np.where(ok, ((opp >= p.theta_open) & (adv <= p.sl_max)).astype(float), np.nan)


def nop_labels(sig, p: SignalParams = SignalParams()):
    a, b = opener_labels(sig, 1, p), opener_labels(sig, -1, p)
    ok = np.isfinite(a) & np.isfinite(b)
    return np.where(ok, ((a == 0) & (b == 0)).astype(float), np.nan)


def closer_labels(sig, d, sl_dist, p: SignalParams = SignalParams(), rule: str = "hold_through"):
    """``sl_dist``: the open order's SL distance from the current close, in sigma (per bar; NaN where no order).
    ``rule``: "hold_through" (default) or "optimum" (see the module docstring)."""
    g, a = sig["hold_gain"][d], sig["hold_adverse"][d]
    ok = np.isfinite(g) & np.isfinite(a) & np.isfinite(sl_dist)
    close = (g < p.theta_close) | (a >= sl_dist)
    if rule == "optimum":
        adv = sig["next_adverse_inside"][d]
        ok &= np.isfinite(adv)
        close = close | (adv == 1)
    elif rule != "hold_through":
        raise ValueError(f"CLOSER_RULE_UNKNOWN:{rule}")
    return np.where(ok, close.astype(float), np.nan)
