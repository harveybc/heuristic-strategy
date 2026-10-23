"""Hindsight expert signals per agent (owner 2026-10-22): the ideal learning signal of each specialist, derived from
the future price path up to the OPTIMAL expert's next pivot. Training-side only: labels, targets and masks; never an
observation. Every value at bar b is known only after its horizon bar, so a row may use it only when that horizon
ends inside the row's period (purge by real label support, as for the ZigZag labels).

Two horizons, one per agent kind (decision after the completed bar b; execution at the next open o[b+1]):
  OPENER of side d:   H_open[d](b) = the next optimal pivot of the FAVOURABLE kind (peak for a long, valley for a
                      short) after b, capped by b + max_bars and the forced flat. The anticipatory ZigZag label marks
                      the bars BEFORE a valley as LONG; the optimum exit of that entry is the following peak, so the
                      horizon of an opener is the favourable extreme, and the dip before it is the risk it must survive.
  CLOSER of side d:   H_close(b) = the next optimal pivot of ANY kind after b (same caps). If that pivot is of the
                      favourable kind the optimum still lies ahead (hold); if it is adverse, the optimum exit has
                      passed (close).
Per bar b and side d in {+1, -1}:
  opportunity[d]        = max favourable excursion of an entry at o[b+1] over (b+1 .. H_open], causal sigma, net of cost
                          -> what an OPENER could earn at most before the optimum ("cuánto es lo máximo que podrá ganar")
  adverse_to_peak[d]    = max adverse excursion from o[b+1] until the bar of that favourable peak (sigma)
                          -> the SL an opener would need to survive until its best exit
  bars_to_peak[d]       = bars from b+1 to the favourable peak
  hold_gain[d]          = max further favourable move from c[b] over (b+1 .. H_close] (sigma)
                          -> what a CLOSER gives up by closing now ("cuánto más pudo haber ganado respecto al óptimo")
  hold_adverse[d]       = max adverse move from c[b] over (b+1 .. H_close] (sigma) -> what a CLOSER risks by holding
  next_is_favourable[d] = 1 if H_close(b) is a pivot of the favourable kind for d
Labels (declared thresholds, in sigma):
  opener[d]   = 1 if opportunity[d] >= theta_open and adverse_to_peak[d] <= sl_max   (a profitable, survivable entry)
  nop         = 1 if neither opener label is 1
  closer[d]   = 1 (CLOSE) if not next_is_favourable[d] or hold_gain[d] < theta_close or hold_adverse[d] >= sl_dist[d]
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
    """For every bar b, the first entry of ``bars`` strictly after b (n - 1 when none)."""
    bars = np.asarray(bars, np.int64)
    if not len(bars):
        return np.full(n, n - 1)
    return bars[np.minimum(np.searchsorted(bars, np.arange(n), side="right"), len(bars) - 1)]


def hindsight_signals(o, h, l, c, sigma, pivots, flat_bar, p: SignalParams = SignalParams()):
    """``pivots``: [(bar, kind, level)] of the optimal expert (kind +1 peak / -1 valley). O(n * max_bars).
    Returns (signals dict of arrays (n,) per side, horizons dict {"open": {d: (n,)}, "close": (n,)}); -1 = no horizon."""
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    sigma = np.asarray(sigma, float)
    n = len(c)
    pb = np.asarray([i for i, _, _ in pivots], np.int64)
    pk = np.asarray([k for _, k, _ in pivots], np.int64)
    cap = np.minimum.reduce([np.arange(n) + p.max_bars, np.asarray(flat_bar, np.int64), np.full(n, n - 1)])
    nxt_any = np.minimum(_next_after(pb, n), cap)
    kind_any = np.zeros(n, np.int64)
    if len(pb):
        j = np.minimum(np.searchsorted(pb, np.arange(n), side="right"), len(pb) - 1)
        kind_any = np.where(pb[j] > np.arange(n), pk[j], 0)
    nxt_fav = {1: np.minimum(_next_after(pb[pk == 1], n), cap), -1: np.minimum(_next_after(pb[pk == -1], n), cap)}
    out = {k: {1: np.full(n, np.nan), -1: np.full(n, np.nan)} for k in
           ("opportunity", "adverse_to_peak", "bars_to_peak", "hold_gain", "hold_adverse", "next_is_favourable")}
    hz = {"open": {1: np.full(n, -1, np.int64), -1: np.full(n, -1, np.int64)}, "close": np.full(n, -1, np.int64)}
    for b in range(n - 2):
        s = sigma[b + 1] if b + 1 < n else np.nan
        if not np.isfinite(s) or s <= 0:
            continue
        e = o[b + 1]
        for d in (1, -1):
            end = int(nxt_fav[d][b])
            if end > b + 1:
                hz["open"][d][b] = end
                hh, ll = h[b + 1:end + 1], l[b + 1:end + 1]
                fav = (hh - e) if d > 0 else (e - ll)
                adv = (e - ll) if d > 0 else (hh - e)
                k = int(np.argmax(fav))
                out["opportunity"][d][b] = fav[k] / s - p.cost_sigma
                out["adverse_to_peak"][d][b] = max(float(adv[:k + 1].max()), 0.0) / s
                out["bars_to_peak"][d][b] = k + 1
        end = int(nxt_any[b])
        if end > b + 1:
            hz["close"][b] = end
            hh, ll = h[b + 1:end + 1], l[b + 1:end + 1]
            for d in (1, -1):
                out["hold_gain"][d][b] = max(float(((hh - c[b]) if d > 0 else (c[b] - ll)).max()), 0.0) / s
                out["hold_adverse"][d][b] = max(float(((c[b] - ll) if d > 0 else (hh - c[b])).max()), 0.0) / s
                out["next_is_favourable"][d][b] = float(kind_any[b] == d) if kind_any[b] != 0 else np.nan
    return out, hz


def opener_labels(sig, d, p: SignalParams = SignalParams()):
    opp, adv = sig["opportunity"][d], sig["adverse_to_peak"][d]
    ok = np.isfinite(opp) & np.isfinite(adv)
    return np.where(ok, ((opp >= p.theta_open) & (adv <= p.sl_max)).astype(float), np.nan)


def nop_labels(sig, p: SignalParams = SignalParams()):
    a, b = opener_labels(sig, 1, p), opener_labels(sig, -1, p)
    ok = np.isfinite(a) & np.isfinite(b)
    return np.where(ok, ((a == 0) & (b == 0)).astype(float), np.nan)


def closer_labels(sig, d, sl_dist, p: SignalParams = SignalParams()):
    """``sl_dist``: the open order's SL distance from the current close, in sigma (per bar; NaN where no order)."""
    g, a, f = sig["hold_gain"][d], sig["hold_adverse"][d], sig["next_is_favourable"][d]
    ok = np.isfinite(g) & np.isfinite(a) & np.isfinite(sl_dist) & np.isfinite(f)
    return np.where(ok, ((f == 0) | (g < p.theta_close) | (a >= sl_dist)).astype(float), np.nan)
